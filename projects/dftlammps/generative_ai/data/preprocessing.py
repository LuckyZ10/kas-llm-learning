"""
Crystal Preprocessing Module
============================

Preprocessing utilities for crystal structures:
- Normalization
- Data augmentation
- Symmetry handling
- Wyckoff position processing
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, List
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import warnings


class CrystalPreprocessor:
    """
    Preprocessor for crystal structures.
    
    Handles normalization, augmentation, and format conversion.
    """
    
    def __init__(
        self,
        normalize_coords: bool = True,
        normalize_lattice: bool = True,
        augment_rotation: bool = False,
        center_structure: bool = True
    ):
        self.normalize_coords = normalize_coords
        self.normalize_lattice = normalize_lattice
        self.augment_rotation = augment_rotation
        self.center_structure = center_structure
        
        # Statistics for normalization (computed from training data)
        self.coord_mean = 0.5
        self.coord_std = 0.289  # std of uniform [0,1]
        self.lattice_mean = None
        self.lattice_std = None
    
    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply preprocessing."""
        data = data.copy()
        
        # Center structure
        if self.center_structure:
            data = self.center(data)
        
        # Normalize coordinates
        if self.normalize_coords:
            data = self.normalize_coordinates(data)
        
        # Normalize lattice
        if self.normalize_lattice:
            data = self.normalize_lattice_params(data)
        
        # Data augmentation
        if self.augment_rotation:
            data = self.random_rotation(data)
        
        return data
    
    def center(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Center structure at origin."""
        frac_coords = data["frac_coords"].clone()
        mask = data.get("mask", torch.ones(frac_coords.shape[0], dtype=torch.bool))
        
        # Compute centroid
        centroid = frac_coords[mask].mean(dim=0)
        
        # Center
        frac_coords[mask] = frac_coords[mask] - centroid
        frac_coords[mask] = frac_coords[mask] % 1.0  # Wrap to [0, 1)
        
        data["frac_coords"] = frac_coords
        return data
    
    def normalize_coordinates(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Normalize coordinates to zero mean, unit variance."""
        frac_coords = data["frac_coords"].clone()
        
        # Normalize
        frac_coords = (frac_coords - self.coord_mean) / self.coord_std
        
        data["frac_coords"] = frac_coords
        data["coord_mean"] = torch.tensor(self.coord_mean)
        data["coord_std"] = torch.tensor(self.coord_std)
        
        return data
    
    def denormalize_coordinates(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Denormalize coordinates."""
        frac_coords = data["frac_coords"].clone()
        
        mean = data.get("coord_mean", self.coord_mean)
        std = data.get("coord_std", self.coord_std)
        
        frac_coords = frac_coords * std + mean
        frac_coords = frac_coords % 1.0  # Wrap to [0, 1)
        
        data["frac_coords"] = frac_coords
        return data
    
    def normalize_lattice_params(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Normalize lattice parameters."""
        lattice = data["lattice"].clone()
        
        # Log-scale for lengths, degrees for angles
        # lattice: [a, b, c, alpha, beta, gamma]
        
        # Normalize lengths (log scale)
        lattice[:3] = torch.log(lattice[:3])
        
        # Normalize angles (already roughly centered at 90)
        lattice[3:] = (lattice[3:] - 90.0) / 30.0  # Normalize around 90 degrees
        
        data["lattice"] = lattice
        return data
    
    def denormalize_lattice_params(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Denormalize lattice parameters."""
        lattice = data["lattice"].clone()
        
        # Denormalize lengths
        lattice[:3] = torch.exp(lattice[:3])
        
        # Denormalize angles
        lattice[3:] = lattice[3:] * 30.0 + 90.0
        
        data["lattice"] = lattice
        return data
    
    def random_rotation(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply random rotation (for data augmentation)."""
        # For fractional coordinates, we need to rotate in Cartesian space
        # This is a simplified version
        
        # Generate random rotation matrix
        angle_x = np.random.uniform(0, 2 * np.pi)
        angle_y = np.random.uniform(0, 2 * np.pi)
        angle_z = np.random.uniform(0, 2 * np.pi)
        
        Rx = torch.tensor([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)]
        ], dtype=torch.float32)
        
        Ry = torch.tensor([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ], dtype=torch.float32)
        
        Rz = torch.tensor([
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        R = Rz @ Ry @ Rx
        
        # Apply to fractional coordinates (approximate for orthogonal cells)
        # For exact rotation, need to convert to Cartesian
        data["frac_coords"] = data["frac_coords"] @ R
        data["frac_coords"] = data["frac_coords"] % 1.0
        
        return data
    
    def compute_statistics(self, dataset):
        """Compute normalization statistics from dataset."""
        all_coords = []
        all_lattices = []
        
        for data in dataset:
            mask = data.get("mask", torch.ones(data["frac_coords"].shape[0], dtype=torch.bool))
            all_coords.append(data["frac_coords"][mask])
            all_lattices.append(data["lattice"])
        
        coords = torch.cat(all_coords)
        lattices = torch.stack(all_lattices)
        
        self.coord_mean = coords.mean().item()
        self.coord_std = coords.std().item()
        
        self.lattice_mean = lattices.mean(dim=0)
        self.lattice_std = lattices.std(dim=0)


class WyckoffPositionEncoder:
    """
    Encoder for Wyckoff positions.
    
    Maps Wyckoff positions to continuous embeddings.
    """
    
    def __init__(self, embed_dim: int = 64):
        self.embed_dim = embed_dim
        
        # Wyckoff position encoding
        # 27 letters + 1 for general position
        self.position_embed = torch.nn.Embedding(28, embed_dim)
        
        # Multiplicity encoding
        self.multiplicity_embed = torch.nn.Embedding(100, embed_dim)
    
    def encode(
        self,
        wyckoff_letters: List[str],
        multiplicities: List[int]
    ) -> torch.Tensor:
        """
        Encode Wyckoff positions.
        
        Args:
            wyckoff_letters: List of Wyckoff letters (e.g., ['a', 'b', 'c'])
            multiplicities: List of multiplicities
            
        Returns:
            Encoded positions (N, embed_dim)
        """
        # Convert letters to indices
        letter_to_idx = {chr(ord('a') + i): i for i in range(26)}
        letter_to_idx['\u03b1'] = 26  # alpha for general position
        
        indices = torch.tensor([letter_to_idx.get(l, 27) for l in wyckoff_letters])
        mult_indices = torch.tensor(multiplicities)
        
        return self.position_embed(indices) + self.multiplicity_embed(mult_indices)


class SpaceGroupAnalyzer:
    """
    Analyzer for space group information.
    
    Extracts symmetry information from crystal structures.
    """
    
    def __init__(self, symprec: float = 0.01):
        self.symprec = symprec
    
    def analyze(self, structure: Structure) -> Dict:
        """Analyze space group of structure."""
        analyzer = SpacegroupAnalyzer(structure, symprec=self.symprec)
        
        return {
            "space_group_number": analyzer.get_space_group_number(),
            "space_group_symbol": analyzer.get_space_group_symbol(),
            "crystal_system": analyzer.get_crystal_system(),
            "point_group": analyzer.get_point_group(),
            "wyckoff_positions": analyzer.get_symmetry_dataset().wyckoffs,
            "equivalent_atoms": analyzer.get_symmetry_dataset().equivalent_atoms.tolist()
        }
    
    def get_symmetry_operations(self, structure: Structure):
        """Get symmetry operations for structure."""
        analyzer = SpacegroupAnalyzer(structure, symprec=self.symprec)
        return analyzer.get_symmetry_operations()


def structure_to_tensors(structure: Structure) -> Dict[str, torch.Tensor]:
    """Convert pymatgen Structure to tensors."""
    return {
        "atom_types": torch.tensor([site.specie.Z for site in structure], dtype=torch.long),
        "frac_coords": torch.tensor(structure.frac_coords, dtype=torch.float32),
        "lattice": torch.tensor(structure.lattice.parameters, dtype=torch.float32)
    }


def tensors_to_structure(data: Dict[str, torch.Tensor]) -> Structure:
    """Convert tensors to pymatgen Structure."""
    atom_types = data["atom_types"].cpu().numpy()
    frac_coords = data["frac_coords"].cpu().numpy()
    lattice = data["lattice"].cpu().numpy()
    
    # Handle batched data
    if atom_types.ndim > 1:
        atom_types = atom_types[0]
        frac_coords = frac_coords[0]
        lattice = lattice[0]
    
    # Filter out padding
    mask = atom_types > 0
    atom_types = atom_types[mask]
    frac_coords = frac_coords[mask]
    
    # Create lattice
    lattice_obj = Lattice.from_parameters(*lattice)
    
    # Create structure
    from pymatgen.core import Element
    species = [Element.from_Z(int(z)) for z in atom_types]
    
    structure = Structure(lattice_obj, species, frac_coords)
    
    return structure
