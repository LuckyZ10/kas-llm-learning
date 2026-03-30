"""
Utility functions for multiscale coupling.
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
import json

class UnitConverter:
    """Unit conversion utilities for multiscale simulations."""
    
    # Conversion factors
    EV_TO_KCAL = 23.0605
    ANG_TO_BOHR = 1.88973
    BOHR_TO_ANG = 0.529177
    KCAL_TO_EV = 1.0 / 23.0605
    
    @staticmethod
    def energy_eV_to_kcal(energy_eV: float) -> float:
        """Convert energy from eV to kcal/mol."""
        return energy_eV * UnitConverter.EV_TO_KCAL
    
    @staticmethod
    def energy_kcal_to_eV(energy_kcal: float) -> float:
        """Convert energy from kcal/mol to eV."""
        return energy_kcal * UnitConverter.KCAL_TO_EV
    
    @staticmethod
    def length_ang_to_bohr(length_ang: float) -> float:
        """Convert length from Angstrom to Bohr."""
        return length_ang * UnitConverter.ANG_TO_BOHR
    
    @staticmethod
    def length_bohr_to_ang(length_bohr: float) -> float:
        """Convert length from Bohr to Angstrom."""
        return length_bohr * UnitConverter.BOHR_TO_ANG


class AtomSelection:
    """Tools for selecting atoms in QM/MM simulations."""
    
    @staticmethod
    def select_sphere(positions: np.ndarray, 
                      center: np.ndarray, 
                      radius: float) -> np.ndarray:
        """
        Select atoms within a spherical region.
        
        Args:
            positions: (N, 3) array of atomic positions
            center: (3,) array of sphere center
            radius: selection radius in Angstrom
            
        Returns:
            Boolean array of selected atoms
        """
        distances = np.linalg.norm(positions - center, axis=1)
        return distances <= radius
    
    @staticmethod
    def select_index_range(start: int, end: int, total_atoms: int) -> np.ndarray:
        """Select atoms by index range."""
        mask = np.zeros(total_atoms, dtype=bool)
        mask[start:end] = True
        return mask
    
    @staticmethod
    def select_residue(residue_ids: np.ndarray, 
                       target_residues: List[int]) -> np.ndarray:
        """Select atoms by residue ID."""
        return np.isin(residue_ids, target_residues)


class BoundaryHandler:
    """Handle QM/MM boundary conditions."""
    
    def __init__(self, qm_atoms: np.ndarray, mm_atoms: np.ndarray):
        """
        Initialize boundary handler.
        
        Args:
            qm_atoms: Boolean mask for QM atoms
            mm_atoms: Boolean mask for MM atoms
        """
        self.qm_atoms = qm_atoms
        self.mm_atoms = mm_atoms
        self.boundary_pairs = []
        
    def find_boundary_bonds(self, bonds: np.ndarray):
        """
        Find bonds crossing QM/MM boundary.
        
        Args:
            bonds: (M, 2) array of bond pairs (0-indexed)
        """
        self.boundary_pairs = []
        for bond in bonds:
            i, j = bond
            if self.qm_atoms[i] and self.mm_atoms[j]:
                self.boundary_pairs.append((i, j))
            elif self.qm_atoms[j] and self.mm_atoms[i]:
                self.boundary_pairs.append((j, i))
    
    def add_link_atoms(self, positions: np.ndarray, 
                       link_atom_type: str = 'H') -> Tuple[np.ndarray, List[int]]:
        """
        Add link atoms at QM/MM boundary.
        
        Args:
            positions: (N, 3) array of atomic positions
            link_atom_type: type of link atom (default H)
            
        Returns:
            New positions with link atoms, list of link atom indices
        """
        link_positions = []
        link_indices = []
        
        for qm_idx, mm_idx in self.boundary_pairs:
            # Place link atom along bond at standard distance
            qm_pos = positions[qm_idx]
            mm_pos = positions[mm_idx]
            direction = qm_pos - mm_pos
            direction = direction / np.linalg.norm(direction)
            
            # C-H bond length ~1.09 Angstrom
            if link_atom_type == 'H':
                link_dist = 1.09
            else:
                link_dist = 1.0
                
            link_pos = qm_pos - direction * link_dist
            link_positions.append(link_pos)
            link_indices.append(len(positions) + len(link_positions) - 1)
        
        if link_positions:
            new_positions = np.vstack([positions, np.array(link_positions)])
        else:
            new_positions = positions
            
        return new_positions, link_indices


class Logger:
    """Simple logging utility for multiscale simulations."""
    
    def __init__(self, log_file: Optional[str] = None, verbose: bool = True):
        self.log_file = log_file
        self.verbose = verbose
        
    def log(self, message: str, level: str = 'INFO'):
        """Log a message."""
        formatted = f"[{level}] {message}"
        if self.verbose:
            print(formatted)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(formatted + '\n')
    
    def info(self, message: str):
        self.log(message, 'INFO')
        
    def warning(self, message: str):
        self.log(message, 'WARNING')
        
    def error(self, message: str):
        self.log(message, 'ERROR')


def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config(config: Dict, config_path: str):
    """Save configuration to JSON file."""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
