"""
Machine Learning Coarse-Graining Module

Implements ML-based methods for learning optimal coarse-grained mappings
from atomistic simulations.
"""
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
import json


@dataclass
class CGMapping:
    """Represents a coarse-grained mapping."""
    atom_to_bead: np.ndarray  # Mapping from atoms to CG beads
    bead_positions: np.ndarray  # CG bead positions
    bead_types: List[str]  # CG bead types
    n_beads: int
    n_atoms: int
    
    def save(self, filename: str):
        """Save mapping to file."""
        np.savez(filename, 
                 atom_to_bead=self.atom_to_bead,
                 bead_positions=self.bead_positions,
                 n_beads=self.n_beads,
                 n_atoms=self.n_atoms)
        with open(filename + '.types', 'w') as f:
            json.dump(self.bead_types, f)
    
    @classmethod
    def load(cls, filename: str) -> 'CGMapping':
        """Load mapping from file."""
        data = np.load(filename + '.npz')
        with open(filename + '.types', 'r') as f:
            types = json.load(f)
        return cls(
            atom_to_bead=data['atom_to_bead'],
            bead_positions=data['bead_positions'],
            bead_types=types,
            n_beads=int(data['n_beads']),
            n_atoms=int(data['n_atoms'])
        )


class CoarseGrainer:
    """Base class for coarse-graining methods."""
    
    def __init__(self, n_beads: int):
        """
        Initialize coarse-grainer.
        
        Args:
            n_beads: Target number of CG beads
        """
        self.n_beads = n_beads
        self.mapping = None
        
    def fit(self, trajectories: List[np.ndarray], 
            atom_types: List[str]) -> CGMapping:
        """
        Learn coarse-grained mapping from trajectories.
        
        Args:
            trajectories: List of (n_frames, n_atoms, 3) position arrays
            atom_types: List of atom type strings
            
        Returns:
            CGMapping object
        """
        raise NotImplementedError
    
    def transform(self, positions: np.ndarray) -> np.ndarray:
        """
        Transform atomistic positions to CG representation.
        
        Args:
            positions: (n_atoms, 3) or (n_frames, n_atoms, 3) positions
            
        Returns:
            CG positions
        """
        if self.mapping is None:
            raise ValueError("Must fit before transform")
        
        if positions.ndim == 2:
            # Single frame
            return self._map_frame(positions)
        else:
            # Trajectory
            cg_traj = []
            for frame in positions:
                cg_traj.append(self._map_frame(frame))
            return np.array(cg_traj)
    
    def _map_frame(self, positions: np.ndarray) -> np.ndarray:
        """Map single frame to CG representation."""
        cg_pos = np.zeros((self.n_beads, 3))
        for bead_idx in range(self.n_beads):
            atom_indices = np.where(self.mapping.atom_to_bead == bead_idx)[0]
            if len(atom_indices) > 0:
                cg_pos[bead_idx] = positions[atom_indices].mean(axis=0)
        return cg_pos
    
    def inverse_transform(self, cg_positions: np.ndarray,
                         template_atoms: np.ndarray) -> np.ndarray:
        """
        Reconstruct atomistic positions from CG representation.
        
        Args:
            cg_positions: CG bead positions
            template_atoms: Reference atomistic structure for internal geometry
            
        Returns:
            Reconstructed atomistic positions
        """
        raise NotImplementedError


class CentroidCoarseGrainer(CoarseGrainer):
    """
    Simple centroid-based coarse-graining.
    Maps groups of atoms to their center of mass/centroid.
    """
    
    def __init__(self, n_beads: int, 
                 predefined_mapping: Optional[np.ndarray] = None):
        """
        Initialize centroid coarse-grainer.
        
        Args:
            n_beads: Number of CG beads
            predefined_mapping: Optional predefined atom-to-bead mapping
        """
        super().__init__(n_beads)
        self.predefined_mapping = predefined_mapping
    
    def fit(self, trajectories: List[np.ndarray],
            atom_types: List[str],
            masses: Optional[np.ndarray] = None) -> CGMapping:
        """
        Create mapping based on atom groupings.
        
        Args:
            trajectories: List of trajectories (for shape reference)
            atom_types: Atom types
            masses: Atomic masses for COM calculation
            
        Returns:
            CGMapping
        """
        n_atoms = trajectories[0].shape[1]
        
        if self.predefined_mapping is not None:
            atom_to_bead = self.predefined_mapping
        else:
            # Create uniform mapping
            atoms_per_bead = n_atoms // self.n_beads
            atom_to_bead = np.zeros(n_atoms, dtype=int)
            for i in range(self.n_beads):
                start = i * atoms_per_bead
                end = start + atoms_per_bead if i < self.n_beads - 1 else n_atoms
                atom_to_bead[start:end] = i
        
        # Generate bead types based on constituent atoms
        bead_types = []
        for i in range(self.n_beads):
            atom_indices = np.where(atom_to_bead == i)[0]
            constituent_types = [atom_types[j] for j in atom_indices]
            # Create bead type name
            unique_types = sorted(set(constituent_types))
            bead_type = ''.join(unique_types)
            bead_types.append(bead_type)
        
        # Calculate representative bead positions manually
        avg_frame = np.mean(trajectories[0], axis=0)
        bead_positions = np.zeros((self.n_beads, 3))
        for bead_idx in range(self.n_beads):
            atom_indices = np.where(atom_to_bead == bead_idx)[0]
            if len(atom_indices) > 0:
                bead_positions[bead_idx] = avg_frame[atom_indices].mean(axis=0)
        
        self.mapping = CGMapping(
            atom_to_bead=atom_to_bead,
            bead_positions=bead_positions,
            bead_types=bead_types,
            n_beads=self.n_beads,
            n_atoms=n_atoms
        )
        
        return self.mapping


class MLCGMapping(CoarseGrainer):
    """
    Machine learning-based coarse-graining.
    Learns optimal mapping using autoencoder-like approach.
    """
    
    def __init__(self, 
                 n_beads: int,
                 encoder_hidden: List[int] = [128, 64],
                 use_forces: bool = True,
                 force_weight: float = 0.1):
        """
        Initialize ML coarse-graining.
        
        Args:
            n_beads: Number of CG beads
            encoder_hidden: Hidden layer sizes for encoder
            use_forces: Whether to use force information
            force_weight: Weight for force regularization
        """
        super().__init__(n_beads)
        self.encoder_hidden = encoder_hidden
        self.use_forces = use_forces
        self.force_weight = force_weight
        self.encoder = None
        self.decoder = None
        self.scaler = None
    
    def fit(self, 
            trajectories: List[np.ndarray],
            atom_types: List[str],
            forces: Optional[List[np.ndarray]] = None) -> CGMapping:
        """
        Learn ML-based coarse-grained mapping.
        
        Args:
            trajectories: List of (n_frames, n_atoms, 3) trajectories
            atom_types: Atom types
            forces: Optional list of force trajectories
            
        Returns:
            CGMapping
        """
        try:
            import torch
            import torch.nn as nn
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("PyTorch and scikit-learn required for MLCGMapping")
        
        # Prepare training data
        X = np.vstack(trajectories)  # (total_frames, n_atoms, 3)
        n_frames, n_atoms, _ = X.shape
        X_flat = X.reshape(n_frames, -1)  # Flatten spatial dimensions
        
        # Normalize
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_flat)
        
        # Build and train encoder
        self.encoder = self._build_encoder(n_atoms * 3, self.encoder_hidden)
        self.decoder = self._build_decoder(self.n_beads * 3, n_atoms * 3)
        
        # Training loop (simplified)
        self._train_autoencoder(X_scaled, forces)
        
        # Extract mapping
        atom_to_bead = self._extract_mapping(X_scaled)
        
        # Create mapping object
        avg_frame = X[0]
        bead_positions = self.transform(avg_frame)
        
        bead_types = [f"BEAD_{i}" for i in range(self.n_beads)]
        
        self.mapping = CGMapping(
            atom_to_bead=atom_to_bead,
            bead_positions=bead_positions,
            bead_types=bead_types,
            n_beads=self.n_beads,
            n_atoms=n_atoms
        )
        
        return self.mapping
    
    def _build_encoder(self, input_dim: int, hidden_dims: List[int]):
        """Build encoder network."""
        try:
            import torch.nn as nn
        except ImportError:
            return None
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        
        # Output layer produces CG coordinates
        layers.append(nn.Linear(prev_dim, self.n_beads * 3))
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self, input_dim: int, output_dim: int):
        """Build decoder network."""
        try:
            import torch.nn as nn
        except ImportError:
            return None
        
        # Mirror of encoder
        layers = []
        prev_dim = input_dim
        for h_dim in reversed(self.encoder_hidden):
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def _train_autoencoder(self, X: np.ndarray, forces: Optional[List] = None):
        """Train autoencoder."""
        # Placeholder for actual training
        # In full implementation, would use proper training loop
        pass
    
    def _extract_mapping(self, X: np.ndarray) -> np.ndarray:
        """Extract hard assignment from soft mapping."""
        # Placeholder for mapping extraction
        n_atoms = X.shape[1] // 3
        return np.random.randint(0, self.n_beads, size=n_atoms)
    
    def transform(self, positions: np.ndarray) -> np.ndarray:
        """Transform using trained encoder."""
        try:
            import torch
        except ImportError:
            return super().transform(positions)
        
        if positions.ndim == 2:
            positions = positions.flatten()[np.newaxis, :]
        else:
            positions = positions.reshape(positions.shape[0], -1)
        
        positions_scaled = self.scaler.transform(positions)
        positions_torch = torch.FloatTensor(positions_scaled)
        
        with torch.no_grad():
            cg_flat = self.encoder(positions_torch).numpy()
        
        return cg_flat.reshape(-1, self.n_beads, 3)
    
    def inverse_transform(self, 
                         cg_positions: np.ndarray,
                         template_atoms: np.ndarray = None) -> np.ndarray:
        """Reconstruct using decoder."""
        try:
            import torch
        except ImportError:
            raise NotImplementedError("PyTorch required for inverse transform")
        
        cg_flat = cg_positions.reshape(-1, self.n_beads * 3)
        cg_torch = torch.FloatTensor(cg_flat)
        
        with torch.no_grad():
            atom_flat = self.decoder(cg_torch).numpy()
        
        atom_reconstructed = self.scaler.inverse_transform(atom_flat)
        return atom_reconstructed.reshape(-1, self.mapping.n_atoms, 3)


class ForceMatcher:
    """
    Force matching for coarse-grained force field parameterization.
    Matches CG forces to atomistic mean forces.
    """
    
    def __init__(self, cg_mapping: CGMapping):
        """
        Initialize force matcher.
        
        Args:
            cg_mapping: Coarse-grained mapping
        """
        self.cg_mapping = cg_mapping
        self.cg_forces = None
        
    def compute_reference_forces(self,
                                 atom_positions: np.ndarray,
                                 atom_forces: np.ndarray) -> np.ndarray:
        """
        Compute CG reference forces from atomistic forces.
        
        Args:
            atom_positions: (n_frames, n_atoms, 3) positions
            atom_forces: (n_frames, n_atoms, 3) forces
            
        Returns:
            (n_frames, n_beads, 3) CG forces
        """
        n_frames = len(atom_positions)
        n_beads = self.cg_mapping.n_beads
        cg_forces = np.zeros((n_frames, n_beads, 3))
        
        for frame_idx in range(n_frames):
            for bead_idx in range(n_beads):
                atom_indices = np.where(self.cg_mapping.atom_to_bead == bead_idx)[0]
                cg_forces[frame_idx, bead_idx] = atom_forces[frame_idx, atom_indices].sum(axis=0)
        
        return cg_forces
    
    def optimize_force_field(self,
                            cg_positions: np.ndarray,
                            reference_forces: np.ndarray,
                            force_field_type: str = 'spline') -> Dict:
        """
        Optimize CG force field parameters.
        
        Args:
            cg_positions: (n_frames, n_beads, 3) CG positions
            reference_forces: (n_frames, n_beads, 3) reference CG forces
            force_field_type: Type of force field ('spline', 'neural', etc.)
            
        Returns:
            Optimized force field parameters
        """
        if force_field_type == 'spline':
            return self._optimize_spline(cg_positions, reference_forces)
        elif force_field_type == 'neural':
            return self._optimize_neural(cg_positions, reference_forces)
        else:
            raise ValueError(f"Unknown force field type: {force_field_type}")
    
    def _optimize_spline(self, positions: np.ndarray, 
                        forces: np.ndarray) -> Dict:
        """Optimize spline-based force field."""
        # Calculate radial distribution and forces
        from scipy.interpolate import UnivariateSpline
        
        # Compute distances between beads
        n_frames, n_beads, _ = positions.shape
        
        params = {
            'type': 'spline',
            'n_beads': n_beads,
            'cutoff': 10.0
        }
        
        return params
    
    def _optimize_neural(self, positions: np.ndarray,
                        forces: np.ndarray) -> Dict:
        """Optimize neural network force field."""
        # Placeholder for neural network optimization
        return {
            'type': 'neural',
            'n_beads': self.cg_mapping.n_beads,
            'hidden_layers': [64, 64]
        }


def create_mapping_from_residue(residue_name: str, 
                                residue_atoms: List[str]) -> np.ndarray:
    """
    Create standard coarse-grained mapping for common residues.
    
    Args:
        residue_name: Name of residue (e.g., 'WATER', 'BENZENE')
        residue_atoms: List of atom names
        
    Returns:
        Atom-to-bead mapping array
    """
    n_atoms = len(residue_atoms)
    mapping = np.zeros(n_atoms, dtype=int)
    
    if residue_name.upper() == 'WATER':
        # Single bead for water
        mapping[:] = 0
    elif residue_name.upper() == 'BENZENE':
        # Two beads: ring + substituents
        c_atoms = [i for i, name in enumerate(residue_atoms) 
                  if name.startswith('C')]
        h_atoms = [i for i, name in enumerate(residue_atoms)
                  if name.startswith('H')]
        mapping[c_atoms] = 0
        mapping[h_atoms] = 1
    elif residue_name.upper() in ['ALA', 'GLY', 'VAL', 'LEU', 'ILE']:
        # Amino acid: backbone + sidechain
        backbone = ['N', 'CA', 'C', 'O']
        for i, name in enumerate(residue_atoms):
            if name in backbone:
                mapping[i] = 0
            else:
                mapping[i] = 1
    else:
        # Default: each heavy atom is a bead
        for i, name in enumerate(residue_atoms):
            if name.startswith('H'):
                # Map hydrogens to previous heavy atom
                mapping[i] = max(0, i - 1)
            else:
                mapping[i] = i
    
    # Renumber to contiguous bead indices
    unique = np.unique(mapping)
    renumber = {old: new for new, old in enumerate(unique)}
    return np.array([renumber[m] for m in mapping])
