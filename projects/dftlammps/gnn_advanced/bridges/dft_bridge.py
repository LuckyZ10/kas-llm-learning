"""
DFT-GNN Bridge
==============

Integrates GNN models with DFT calculations for:
- Active learning workflows
- Data generation for training
- Hybrid GNN-DFT calculations
- Uncertainty quantification
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass
import json
import os


@dataclass
class DFTConfig:
    """Configuration for DFT calculations."""
    code: str = "vasp"  # or "qe", "gpaw", "pyscf"
    functional: str = "PBE"
    basis_set: str = "paw"  # or specific basis
    encut: float = 520.0  # eV
    kpoints: Tuple[int, int, int] = (3, 3, 3)
    ismear: int = 0
    sigma: float = 0.1
    ibrion: int = 2  # relaxation method
    nsw: int = 100   # max relaxation steps
    ediff: float = 1e-5
    ediffg: float = -0.01


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning."""
    uncertainty_threshold: float = 0.5
    max_new_samples: int = 100
    query_strategy: str = "uncertainty"  # or "diversity", "random"
    n_initial: int = 10
    n_iterations: int = 10


class DFTGNNBridge:
    """
    Bridge between GNN models and DFT calculations.
    
    Supports:
    - Hybrid GNN-DFT workflows
    - Active learning for data generation
    - Uncertainty quantification
    - Dataset management
    """
    
    def __init__(self, gnn_model: nn.Module, dft_config: DFTConfig = None,
                 al_config: ActiveLearningConfig = None, device: str = "cpu"):
        """
        Args:
            gnn_model: GNN model for energy/force prediction
            dft_config: DFT calculation configuration
            al_config: Active learning configuration
            device: Device for GNN computations
        """
        self.gnn_model = gnn_model.to(device)
        self.gnn_model.eval()
        self.dft_config = dft_config or DFTConfig()
        self.al_config = al_config or ActiveLearningConfig()
        self.device = device
        
        self.dataset = {
            'structures': [],
            'energies': [],
            'forces': [],
            'stresses': [],
        }
    
    def predict_with_uncertainty(self, atomic_numbers: torch.Tensor,
                                  pos: torch.Tensor,
                                  edge_index: torch.Tensor,
                                  n_samples: int = 10,
                                  noise_scale: float = 0.01) -> Dict[str, torch.Tensor]:
        """
        Predict energy/forces with uncertainty using MC dropout.
        
        Args:
            atomic_numbers: Atom types [N]
            pos: Positions [N, 3]
            edge_index: Edge indices [2, E]
            n_samples: Number of MC samples
            noise_scale: Input noise scale
        
        Returns:
            Dictionary with mean, std, and individual predictions
        """
        self.gnn_model.train()  # Enable dropout
        
        predictions = []
        for _ in range(n_samples):
            # Add noise to positions
            pos_noisy = pos + torch.randn_like(pos) * noise_scale
            
            with torch.no_grad():
                result = self.gnn_model(
                    atomic_numbers=atomic_numbers,
                    pos=pos_noisy,
                    edge_index=edge_index,
                    compute_forces=True
                )
                predictions.append(result)
        
        self.gnn_model.eval()
        
        # Compute statistics
        energies = torch.stack([p['energy'] for p in predictions])
        forces = torch.stack([p['forces'] for p in predictions])
        
        return {
            'energy_mean': energies.mean(dim=0),
            'energy_std': energies.std(dim=0),
            'forces_mean': forces.mean(dim=0),
            'forces_std': forces.std(dim=0),
            'predictions': predictions,
        }
    
    def should_run_dft(self, atomic_numbers: torch.Tensor, pos: torch.Tensor,
                       edge_index: torch.Tensor) -> Tuple[bool, float]:
        """
        Determine if DFT calculation is needed based on uncertainty.
        
        Returns:
            (needs_dft, uncertainty_score)
        """
        result = self.predict_with_uncertainty(atomic_numbers, pos, edge_index)
        
        # Use energy uncertainty as criterion
        uncertainty = result['energy_std'].item()
        needs_dft = uncertainty > self.al_config.uncertainty_threshold
        
        return needs_dft, uncertainty
    
    def run_dft(self, atomic_numbers: np.ndarray, pos: np.ndarray,
                cell: Optional[np.ndarray] = None,
                pbc: Optional[List[bool]] = None) -> Dict[str, np.ndarray]:
        """
        Run DFT calculation using external code.
        
        This is a template - actual implementation would call
        VASP, Quantum ESPRESSO, GPAW, or PySCF.
        
        Args:
            atomic_numbers: Atomic numbers [N]
            pos: Positions [N, 3] in Angstrom
            cell: Unit cell [3, 3] (optional)
            pbc: Periodic boundary conditions [3] (optional)
        
        Returns:
            Dictionary with energy, forces, stress
        """
        # This would interface with actual DFT code
        # For now, return dummy data
        
        print(f"Running {self.dft_config.code} calculation...")
        
        # Generate input files (template)
        if self.dft_config.code.lower() == "vasp":
            self._generate_vasp_input(atomic_numbers, pos, cell, pbc)
        elif self.dft_config.code.lower() == "qe":
            self._generate_qe_input(atomic_numbers, pos, cell, pbc)
        
        # Placeholder results
        energy = np.random.randn() * 0.1  # Dummy energy
        forces = np.random.randn(*pos.shape) * 0.01  # Dummy forces
        stress = np.random.randn(3, 3) * 0.001  # Dummy stress
        
        return {
            'energy': energy,
            'forces': forces,
            'stress': stress,
        }
    
    def _generate_vasp_input(self, atomic_numbers: np.ndarray, pos: np.ndarray,
                             cell: Optional[np.ndarray], pbc: Optional[List[bool]]):
        """Generate VASP input files."""
        # POSCAR
        with open('POSCAR', 'w') as f:
            f.write('Generated by DFT-GNN Bridge\n')
            f.write('1.0\n')
            
            if cell is not None:
                for row in cell:
                    f.write(f'{row[0]:.10f} {row[1]:.10f} {row[2]:.10f}\n')
            else:
                # Large box for molecules
                f.write('20.0 0.0 0.0\n')
                f.write('0.0 20.0 0.0\n')
                f.write('0.0 0.0 20.0\n')
            
            # Unique elements
            unique_elements = np.unique(atomic_numbers)
            element_symbols = self._get_element_symbols(unique_elements)
            
            f.write(' '.join(element_symbols) + '\n')
            
            # Counts
            counts = [np.sum(atomic_numbers == z) for z in unique_elements]
            f.write(' '.join(map(str, counts)) + '\n')
            
            f.write('Cartesian\n')
            
            # Sort positions by element
            for z in unique_elements:
                mask = atomic_numbers == z
                for p in pos[mask]:
                    f.write(f'{p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n')
        
        # INCAR
        with open('INCAR', 'w') as f:
            f.write(f'SYSTEM = GNN-DFT Bridge\n')
            f.write(f'ENCUT = {self.dft_config.encut}\n')
            f.write(f'ISMEAR = {self.dft_config.ismear}\n')
            f.write(f'SIGMA = {self.dft_config.sigma}\n')
            f.write(f'IBRION = {self.dft_config.ibrion}\n')
            f.write(f'NSW = {self.dft_config.nsw}\n')
            f.write(f'EDIFF = {self.dft_config.ediff}\n')
            f.write(f'EDIFFG = {self.dft_config.ediffg}\n')
            f.write('ISIF = 2\n')
            f.write('LWAVE = .FALSE.\n')
            f.write('LCHARG = .FALSE.\n')
        
        # KPOINTS
        with open('KPOINTS', 'w') as f:
            f.write('Automatic mesh\n')
            f.write('0\n')
            f.write('Gamma\n')
            f.write(f'{self.dft_config.kpoints[0]} {self.dft_config.kpoints[1]} {self.dft_config.kpoints[2]}\n')
            f.write('0 0 0\n')
    
    def _generate_qe_input(self, atomic_numbers: np.ndarray, pos: np.ndarray,
                           cell: Optional[np.ndarray], pbc: Optional[List[bool]]):
        """Generate Quantum ESPRESSO input file."""
        # Template for QE input
        with open('pw.in', 'w') as f:
            f.write('\u0026CONTROL\n')
            f.write("  calculation='scf',\n")
            f.write('  restart_mode=\'from_scratch\',\n')
            f.write('  pseudo_dir=\'./\',\n')
            f.write('  outdir=\'./tmp/\',\n')
            f.write('\u0026END\n')
            f.write('\u0026SYSTEM\n')
            
            unique_elements = np.unique(atomic_numbers)
            f.write(f'  nat={len(atomic_numbers)},\n')
            f.write(f'  ntyp={len(unique_elements)},\n')
            f.write(f'  ecutwfc={self.dft_config.encut / 13.6057:.1f},\n')  # Convert eV to Ry
            f.write('\u0026END\n')
            f.write('\u0026ELECTRONS\n')
            f.write('  conv_thr=1d-8,\n')
            f.write('\u0026END\n')
            
            # Atomic positions
            f.write('ATOMIC_POSITIONS (angstrom)\n')
            for z, p in zip(atomic_numbers, pos):
                symbol = self._get_element_symbols([z])[0]
                f.write(f'{symbol} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n')
            
            # Cell
            if cell is not None:
                f.write('CELL_PARAMETERS (angstrom)\n')
                for row in cell:
                    f.write(f'{row[0]:.10f} {row[1]:.10f} {row[2]:.10f}\n')
    
    def _get_element_symbols(self, atomic_numbers: np.ndarray) -> List[str]:
        """Convert atomic numbers to element symbols."""
        symbols = {
            1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
            9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
            16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
            23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
            30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
            37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc',
            44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
            51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 79: 'Au',
        }
        return [symbols.get(z, f'X{z}') for z in atomic_numbers]
    
    def active_learning_loop(self, initial_structures: List[Dict],
                             n_iterations: Optional[int] = None) -> Dict:
        """
        Active learning loop for generating training data.
        
        Args:
            initial_structures: List of initial structure dictionaries
            n_iterations: Number of iterations (uses config if None)
        
        Returns:
            Final dataset
        """
        n_iterations = n_iterations or self.al_config.n_iterations
        
        # Add initial structures to dataset
        for struct in initial_structures:
            self.dataset['structures'].append(struct)
        
        for iteration in range(n_iterations):
            print(f"\n=== Active Learning Iteration {iteration + 1}/{n_iterations} ===")
            
            # Train GNN on current dataset (placeholder)
            self._train_gnn()
            
            # Identify uncertain structures
            uncertain_structures = []
            
            for struct in self._generate_candidate_structures():
                needs_dft, uncertainty = self.should_run_dft(
                    struct['atomic_numbers'],
                    struct['pos'],
                    struct['edge_index']
                )
                
                if needs_dft:
                    struct['uncertainty'] = uncertainty
                    uncertain_structures.append(struct)
            
            # Limit new samples
            uncertain_structures = sorted(
                uncertain_structures,
                key=lambda x: x['uncertainty'],
                reverse=True
            )[:self.al_config.max_new_samples]
            
            print(f"Selected {len(uncertain_structures)} structures for DFT")
            
            # Run DFT on uncertain structures
            for struct in uncertain_structures:
                result = self.run_dft(
                    struct['atomic_numbers'].cpu().numpy(),
                    struct['pos'].cpu().numpy(),
                    struct.get('cell'),
                    struct.get('pbc')
                )
                
                # Add to dataset
                self.dataset['structures'].append(struct)
                self.dataset['energies'].append(result['energy'])
                self.dataset['forces'].append(result['forces'])
                self.dataset['stresses'].append(result['stress'])
            
            print(f"Dataset size: {len(self.dataset['structures'])}")
        
        return self.dataset
    
    def _train_gnn(self):
        """Train GNN on current dataset."""
        if len(self.dataset['structures']) == 0:
            return
        
        print(f"Training GNN on {len(self.dataset['structures'])} structures...")
        # This would call the training loop
        # For now, just a placeholder
    
    def _generate_candidate_structures(self) -> List[Dict]:
        """Generate candidate structures for active learning."""
        # This would generate perturbed structures, new compositions, etc.
        # Placeholder: return empty list
        return []
    
    def save_dataset(self, path: str):
        """Save dataset to disk."""
        # Convert to serializable format
        save_data = {
            'structures': [
                {
                    'atomic_numbers': s['atomic_numbers'].cpu().numpy().tolist(),
                    'pos': s['pos'].cpu().numpy().tolist(),
                }
                for s in self.dataset['structures']
            ],
            'energies': [float(e) for e in self.dataset['energies']],
            'forces': [f.tolist() for f in self.dataset['forces']],
            'stresses': [s.tolist() for s in self.dataset['stresses']],
        }
        
        with open(path, 'w') as f:
            json.dump(save_data, f)
        
        print(f"Dataset saved to {path}")
    
    def load_dataset(self, path: str):
        """Load dataset from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.dataset['structures'] = [
            {
                'atomic_numbers': torch.tensor(s['atomic_numbers']),
                'pos': torch.tensor(s['pos']),
            }
            for s in data['structures']
        ]
        self.dataset['energies'] = [np.array(e) for e in data['energies']]
        self.dataset['forces'] = [np.array(f) for f in data['forces']]
        self.dataset['stresses'] = [np.array(s) for s in data['stresses']]
        
        print(f"Loaded {len(self.dataset['structures'])} structures from {path}")
