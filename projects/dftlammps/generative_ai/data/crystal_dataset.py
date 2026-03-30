"""
Crystal Dataset Module
======================

Dataset classes for crystal structure data.
Supports loading from CIF files, Materials Project, and custom formats.
"""

import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any, Callable
import numpy as np
from pathlib import Path
import json


class CrystalDataset(Dataset):
    """
    Base dataset for crystal structures.
    
    Expects data in the format:
    {
        "atom_types": List[int],  # Atomic numbers
        "frac_coords": List[List[float]],  # Fractional coordinates (N, 3)
        "lattice": List[float],  # Lattice parameters (6,) or (3, 3)
        "properties": Optional[Dict[str, float]]  # Target properties
    }
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        data_list: Optional[List[Dict]] = None,
        transform: Optional[Callable] = None,
        max_atoms: int = 100,
        properties_to_predict: Optional[List[str]] = None
    ):
        """
        Args:
            data_path: Path to data file (json, npz)
            data_list: List of data dictionaries
            transform: Optional transform to apply
            max_atoms: Maximum number of atoms per structure
            properties_to_predict: List of property names to include
        """
        self.transform = transform
        self.max_atoms = max_atoms
        self.properties_to_predict = properties_to_predict or []
        
        if data_list is not None:
            self.data = data_list
        elif data_path is not None:
            self.data = self.load_data(data_path)
        else:
            raise ValueError("Either data_path or data_list must be provided")
        
    def load_data(self, path: str) -> List[Dict]:
        """Load data from file."""
        path = Path(path)
        
        if path.suffix == ".json":
            with open(path) as f:
                return json.load(f)
        elif path.suffix == ".npz":
            data = np.load(path, allow_pickle=True)
            return [dict(zip(data.keys(), vals)) for vals in zip(*data.values())]
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Extract atom types
        atom_types = torch.tensor(item["atom_types"], dtype=torch.long)
        num_atoms = len(atom_types)
        
        # Pad or truncate to max_atoms
        if num_atoms > self.max_atoms:
            atom_types = atom_types[:self.max_atoms]
            num_atoms = self.max_atoms
        
        atom_types_padded = torch.zeros(self.max_atoms, dtype=torch.long)
        atom_types_padded[:num_atoms] = atom_types
        
        # Extract fractional coordinates
        frac_coords = torch.tensor(item["frac_coords"], dtype=torch.float32)
        if frac_coords.shape[0] > self.max_atoms:
            frac_coords = frac_coords[:self.max_atoms]
        
        frac_coords_padded = torch.zeros(self.max_atoms, 3)
        frac_coords_padded[:num_atoms] = frac_coords
        
        # Extract lattice
        lattice = torch.tensor(item["lattice"], dtype=torch.float32)
        if lattice.shape == (3, 3):
            # Convert lattice matrix to parameters (a, b, c, alpha, beta, gamma)
            lattice = self.lattice_matrix_to_params(lattice)
        elif lattice.shape != (6,):
            raise ValueError(f"Invalid lattice shape: {lattice.shape}")
        
        # Create mask
        mask = torch.zeros(self.max_atoms, dtype=torch.bool)
        mask[:num_atoms] = True
        
        # Build output
        output = {
            "atom_types": atom_types_padded,
            "frac_coords": frac_coords_padded,
            "lattice": lattice,
            "mask": mask,
            "num_atoms": num_atoms
        }
        
        # Extract properties
        if "properties" in item and self.properties_to_predict:
            props = item["properties"]
            prop_values = [props.get(p, float('nan')) for p in self.properties_to_predict]
            output["properties"] = torch.tensor(prop_values, dtype=torch.float32)
        
        if self.transform:
            output = self.transform(output)
        
        return output
    
    @staticmethod
    def lattice_matrix_to_params(matrix: torch.Tensor) -> torch.Tensor:
        """Convert 3x3 lattice matrix to parameters (a, b, c, alpha, beta, gamma)."""
        a = torch.norm(matrix[0])
        b = torch.norm(matrix[1])
        c = torch.norm(matrix[2])
        
        alpha = torch.acos(torch.dot(matrix[1], matrix[2]) / (b * c))
        beta = torch.acos(torch.dot(matrix[0], matrix[2]) / (a * c))
        gamma = torch.acos(torch.dot(matrix[0], matrix[1]) / (a * b))
        
        return torch.tensor([a, b, c, alpha, beta, gamma])


class MPDataset(CrystalDataset):
    """
    Dataset for Materials Project data.
    
    Loads data from Materials Project format or exported CSV/JSON.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        chemsys: Optional[str] = None,
        nsites: Optional[tuple] = None,
        properties: Optional[List[str]] = None,
        data_path: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            api_key: Materials Project API key
            chemsys: Chemical system filter (e.g., "Li-P-S")
            nsites: Range of number of sites (min, max)
            properties: Properties to fetch
            data_path: Local data path (if not using API)
        """
        self.api_key = api_key
        self.chemsys = chemsys
        self.nsites = nsites
        self.mp_properties = properties or [
            "band_gap",
            "formation_energy_per_atom",
            "energy_per_atom",
            "volume",
            "density"
        ]
        
        if data_path is None and api_key is not None:
            data_list = self.fetch_from_api()
            super().__init__(data_list=data_list, **kwargs)
        else:
            super().__init__(data_path=data_path, **kwargs)
    
    def fetch_from_api(self) -> List[Dict]:
        """Fetch data from Materials Project API."""
        try:
            from mp_api.client import MPRester
        except ImportError:
            raise ImportError("mp-api is required for fetching from Materials Project")
        
        data_list = []
        
        with MPRester(self.api_key) as mpr:
            # Build query
            criteria = {}
            if self.chemsys:
                criteria["chemsys"] = self.chemsys
            if self.nsites:
                criteria["nsites"] = {"$gte": self.nsites[0], "$lte": self.nsites[1]}
            
            # Search
            docs = mpr.materials.search(
                **criteria,
                fields=["material_id", "structure", "formula"] + self.mp_properties
            )
            
            for doc in docs:
                structure = doc.structure
                
                data = {
                    "atom_types": [site.specie.Z for site in structure],
                    "frac_coords": structure.frac_coords.tolist(),
                    "lattice": structure.lattice.parameters,
                    "properties": {k: getattr(doc, k, None) for k in self.mp_properties}
                }
                
                data_list.append(data)
        
        return data_list


class SyntheticCrystalDataset(CrystalDataset):
    """
    Synthetic dataset for testing and development.
    Generates random crystal structures.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        min_atoms: int = 5,
        max_atoms: int = 50,
        elements: Optional[List[int]] = None,
        **kwargs
    ):
        self.num_samples = num_samples
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.elements = elements or list(range(1, 20))  # H to Ca
        
        data_list = self.generate_data()
        super().__init__(data_list=data_list, **kwargs)
    
    def generate_data(self) -> List[Dict]:
        """Generate synthetic crystal data."""
        data_list = []
        
        for _ in range(self.num_samples):
            num_atoms = np.random.randint(self.min_atoms, self.max_atoms + 1)
            
            data = {
                "atom_types": np.random.choice(self.elements, num_atoms).tolist(),
                "frac_coords": np.random.rand(num_atoms, 3).tolist(),
                "lattice": [
                    np.random.uniform(3, 10),  # a
                    np.random.uniform(3, 10),  # b
                    np.random.uniform(3, 10),  # c
                    np.random.uniform(80, 100),  # alpha
                    np.random.uniform(80, 100),  # beta
                    np.random.uniform(80, 100),  # gamma
                ],
                "properties": {
                    "band_gap": np.random.uniform(0, 5),
                    "formation_energy": np.random.uniform(-5, 0)
                }
            }
            
            data_list.append(data)
        
        return data_list


def collate_crystal_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for crystal batches.
    Handles variable-sized structures by padding.
    """
    # Stack tensors
    result = {}
    
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        
        if key == "num_atoms":
            result[key] = torch.tensor(values)
        else:
            result[key] = torch.stack(values)
    
    return result
