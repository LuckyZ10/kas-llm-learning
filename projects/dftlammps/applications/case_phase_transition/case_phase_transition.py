#!/usr/bin/env python3
"""
Case Study: Phase Transition Dynamics
=====================================

Study of phase transitions using metadynamics and umbrella sampling.

Examples:
- Solid-liquid phase transitions
- Martensitic transformations
- Order-disorder transitions
- Ferroelectric phase transitions

Workflow:
1. Identify relevant collective variables
2. Run metadynamics to explore phase space
3. Reconstruct free energy surface
4. Analyze transition mechanism
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging

# ASE
from ase import Atoms
from ase.io import read, write
from ase.build import bulk
from ase.optimize import BFGS

# Local imports
from dftlammps.md_advanced.enhanced_sampling import (
    Metadynamics, MetadynamicsConfig,
    UmbrellaSampling, UmbrellaSamplingConfig,
    DistanceCV, CoordinationNumberCV
)
from dftlammps.md_advanced.free_energy import (
    FreeEnergyPerturbation, FEPConfig
)
from dftlammps.mlip_training import load_model, UnifiedMLIPCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PhaseTransitionConfig:
    """Configuration for phase transition study.
    
    Attributes:
        initial_phase: Initial phase structure file
        final_phase: Final phase structure file (optional)
        transition_type: Type of transition
        sampling_method: 'metadynamics' or 'umbrella_sampling'
        collective_variables: List of CV definitions
        temperature: Simulation temperature (K)
        pressure: Simulation pressure (GPa, optional)
        mlip_type: MLIP type for energy/forces
        mlip_model_path: Path to trained model
    """
    initial_phase: str
    final_phase: Optional[str] = None
    transition_type: str = "solid_liquid"
    sampling_method: str = "metadynamics"
    collective_variables: Optional[List[Dict]] = None
    temperature: float = 1000.0
    pressure: Optional[float] = None
    mlip_type: str = "mace"
    mlip_model_path: Optional[str] = None
    n_simulation_steps: int = 500000


class PhaseTransitionAnalyzer:
    """Analyze phase transitions using enhanced sampling."""
    
    def __init__(self, config: PhaseTransitionConfig):
        self.config = config
        self.calculator = None
        self._setup_calculator()
        self.fes_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    
    def _setup_calculator(self):
        """Setup MLIP calculator."""
        self.calculator = UnifiedMLIPCalculator(
            mlip_type=self.config.mlip_type,
            model_path=self.config.mlip_model_path
        )
    
    def define_collective_variables(self, atoms: Atoms) -> List[Dict]:
        """Define CVs based on transition type."""
        if self.config.collective_variables is not None:
            return self.config.collective_variables
        
        cvs = []
        
        if self.config.transition_type == "solid_liquid":
            # For solid-liquid: coordination number as CV
            # Solid has higher coordination, liquid lower
            
            # Define groups for coordination
            n_atoms = len(atoms)
            all_atoms = list(range(n_atoms))
            
            cvs.append({
                'name': 'coordination',
                'type': 'coordination',
                'group1': all_atoms[:n_atoms//2],
                'group2': all_atoms[n_atoms//2:],
                'r0': 3.5
            })
            
            # Also use Q6 as CV
            cvs.append({
                'name': 'Q6',
                'type': 'bond_orientational',
                'l': 6
            })
        
        elif self.config.transition_type == "martensitic":
            # For martensitic: lattice parameter ratio
            cvs.append({
                'name': 'c_over_a',
                'type': 'lattice_parameter',
                'direction': 'c/a'
            })
        
        elif self.config.transition_type == "order_disorder":
            # Warren-Cowley short-range order parameter
            cvs.append({
                'name': 'SRO',
                'type': 'short_range_order',
                'species': list(set(atoms.get_chemical_symbols()))
            })
        
        return cvs
    
    def run_metadynamics(self, atoms: Atoms,
                        cvs: List[Dict]) -> Dict:
        """Run metadynamics simulation."""
        logger.info("Starting metadynamics simulation")
        
        config = MetadynamicsConfig(
            collective_variables=cvs,
            gaussian_height=2.0,  # kcal/mol
            gaussian_width=0.1,
            hill_frequency=500,
            well_tempered=True,
            bias_factor=10.0,
            temperature=self.config.temperature,
            nsteps=self.config.n_simulation_steps,
            output_dir="./metadynamics_phase_transition",
            use_plumed=True
        )
        
        mtd = Metadynamics(config)
        
        # Generate PLUMED input
        mtd.write_plumed_script("./plumed_phase_transition.dat")
        
        # Run simulation (would be with actual MD)
        result = {
            'success': True,
            'hills_file': './HILLS',
            'colvar_file': './COLVAR',
            'config': config
        }
        
        return result
    
    def run_umbrella_sampling(self, atoms_initial: Atoms,
                             atoms_final: Optional[Atoms] = None) -> Dict:
        """Run umbrella sampling along reaction coordinate."""
        logger.info("Starting umbrella sampling")
        
        # Define reaction coordinate (e.g., lattice constant)
        if atoms_final is not None:
            # Interpolate between phases
            cv_initial = self._compute_phase_coordinate(atoms_initial)
            cv_final = self._compute_phase_coordinate(atoms_final)
        else:
            # Guess final state
            cv_initial = 0.0
            cv_final = 1.0
        
        # Setup windows
        n_windows = 20
        cv_path = np.linspace(cv_initial, cv_final, n_windows)
        
        cvs = self.define_collective_variables(atoms_initial)
        
        config = UmbrellaSamplingConfig(
            collective_variables=cvs,
            reaction_path=[[cv] for cv in cv_path],
            kappa=100.0,
            n_windows=n_windows,
            nsteps_per_window=50000,
            temperature=self.config.temperature,
            output_dir="./umbrella_sampling_phase_transition"
        )
        
        us = UmbrellaSampling(config)
        
        # Run windows
        results = us.run_all_windows(atoms_initial, parallel=False)
        
        # Analyze with WHAM
        pmf = us.analyze_windows()
        
        return {
            'success': True,
            'windows': results,
            'pmf': pmf
        }
    
    def _compute_phase_coordinate(self, atoms: Atoms) -> float:
        """Compute scalar coordinate representing phase."""
        # Simple example: use density or lattice parameter
        volume = atoms.get_volume()
        n_atoms = len(atoms)
        
        return volume / n_atoms  # Volume per atom
    
    def analyze_fes(self, hills_file: str,
                   colvar_file: str) -> Dict:
        """Analyze free energy surface from metadynamics."""
        logger.info("Reconstructing free energy surface")
        
        # Load data
        hills = pd.read_csv(hills_file, sep=r'\\s+', comment='#')
        colvar = pd.read_csv(colvar_file, sep=r'\\s+', comment='#')
        
        # Reconstruct FES
        if len(self.config.collective_variables) == 1:
            grid, fes = self._reconstruct_fes_1d(hills)
        else:
            grid, fes = self._reconstruct_fes_2d(hills)
        
        self.fes_data = (grid, fes)
        
        # Find minima (phases)
        minima = self._find_fes_minima(grid, fes)
        
        # Find transition state
        ts = self._find_transition_state(grid, fes, minima)
        
        # Compute barrier
        if minima and ts is not None:
            barrier = fes[ts] - min(minima.values())
        else:
            barrier = None
        
        return {
            'grid': grid,
            'fes': fes,
            'minima': minima,
            'transition_state': ts,
            'barrier': barrier
        }
    
    def _reconstruct_fes_1d(self, hills: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct 1D FES from hills."""
        cv_range = (hills['cv1'].min() - 0.5, hills['cv1'].max() + 0.5)
        grid = np.linspace(cv_range[0], cv_range[1], 200)
        
        bias = np.zeros_like(grid)
        
        for _, hill in hills.iterrows():
            center = hill['cv1']
            sigma = hill['sigma_cv1']
            height = hill['height']
            
            bias += height * np.exp(-0.5 * ((grid - center) / sigma) ** 2)
        
        fes = -bias
        fes -= fes.min()
        
        return grid, fes
    
    def _reconstruct_fes_2d(self, hills: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct 2D FES from hills."""
        # Simplified 2D reconstruction
        x_range = (hills['cv1'].min() - 0.5, hills['cv1'].max() + 0.5)
        y_range = (hills['cv2'].min() - 0.5, hills['cv2'].max() + 0.5)
        
        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x, y)
        
        bias = np.zeros_like(X)
        
        for _, hill in hills.iterrows():
            cx = hill['cv1']
            cy = hill['cv2']
            sx = hill['sigma_cv1']
            sy = hill['sigma_cv2']
            height = hill['height']
            
            bias += height * np.exp(-0.5 * (
                ((X - cx) / sx) ** 2 + ((Y - cy) / sy) ** 2
            ))
        
        fes = -bias
        fes -= fes.min()
        
        return (X, Y), fes
    
    def _find_fes_minima(self, grid: np.ndarray,
                        fes: np.ndarray) -> Dict[int, float]:
        """Find local minima in FES."""
        from scipy.signal import find_peaks
        
        if fes.ndim == 1:
            # 1D case: find minima as negative peaks of -fes
            peaks, _ = find_peaks(-fes, height=-np.inf)
            
            minima = {int(p): float(fes[p]) for p in peaks}
            
            # Also include boundaries
            minima[0] = float(fes[0])
            minima[len(fes)-1] = float(fes[-1])
            
            return minima
        else:
            # 2D case: simplified
            min_idx = np.unravel_index(np.argmin(fes), fes.shape)
            return {min_idx: float(fes[min_idx])}
    
    def _find_transition_state(self, grid: np.ndarray,
                              fes: np.ndarray,
                              minima: Dict) -> Optional[int]:
        """Find transition state (maximum between minima)."""
        if fes.ndim > 1:
            return None
        
        if len(minima) < 2:
            return None
        
        # Find maximum between two lowest minima
        sorted_minima = sorted(minima.items(), key=lambda x: x[1])
        
        if len(sorted_minima) >= 2:
            idx1 = sorted_minima[0][0]
            idx2 = sorted_minima[1][0]
            
            # Ensure idx1 < idx2
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
            
            # Find maximum in between
            if idx2 > idx1 + 1:
                ts_region = fes[idx1:idx2]
                ts_rel = np.argmax(ts_region)
                return idx1 + ts_rel
        
        return None
    
    def compute_phase_diagram_point(self,
                                   temperatures: List[float],
                                   pressures: List[float]) -> pd.DataFrame:
        """Compute phase stability at different T,P conditions."""
        results = []
        
        for T in temperatures:
            for P in pressures:
                # Run calculation at this T,P
                self.config.temperature = T
                self.config.pressure = P
                
                # Load initial structure
                atoms = read(self.config.initial_phase)
                
                # Quick energy evaluation
                atoms.calc = self.calculator
                energy = atoms.get_potential_energy()
                
                # Compute PV term
                if P is not None:
                    pv = P * atoms.get_volume() * 0.0062415  # Convert to eV
                else:
                    pv = 0
                
                # Approximate free energy (very simplified)
                # F ≈ E + PV - TS
                # Entropy estimated from Debye model or similar
                S = 3 * 8.617e-5 * T  # Rough estimate
                F = energy + pv - T * S
                
                results.append({
                    'temperature': T,
                    'pressure': P,
                    'energy': energy,
                    'free_energy': F
                })
        
        return pd.DataFrame(results)
    
    def run_full_analysis(self) -> Dict:
        """Run complete phase transition analysis."""
        logger.info("Starting full phase transition analysis")
        
        # Load initial structure
        atoms = read(self.config.initial_phase)
        
        # Define CVs
        cvs = self.define_collective_variables(atoms)
        
        # Run enhanced sampling
        if self.config.sampling_method == 'metadynamics':
            sampling_result = self.run_metadynamics(atoms, cvs)
        else:
            if self.config.final_phase:
                atoms_final = read(self.config.final_phase)
            else:
                atoms_final = None
            sampling_result = self.run_umbrella_sampling(atoms, atoms_final)
        
        # Analyze results if available
        if sampling_result['success'] and 'hills_file' in sampling_result:
            fes_analysis = self.analyze_fes(
                sampling_result['hills_file'],
                sampling_result.get('colvar_file', './COLVAR')
            )
            sampling_result['fes_analysis'] = fes_analysis
        
        return sampling_result


def example_tin_phase_transition():
    """Example: Tin phase transition (diamond to beta-tin)."""
    
    # Create alpha-tin (diamond structure)
    alpha_sn = bulk('Sn', 'diamond', a=6.49)
    
    # Create beta-tin (tetragonal)
    beta_sn = bulk('Sn', 'tetragonal', a=5.83, c=3.18)
    
    # Save structures
    write('alpha_sn.xyz', alpha_sn)
    write('beta_sn.xyz', beta_sn)
    
    config = PhaseTransitionConfig(
        initial_phase='alpha_sn.xyz',
        final_phase='beta_sn.xyz',
        transition_type='martensitic',
        sampling_method='metadynamics',
        temperature=300,
        mlip_type='chgnet'
    )
    
    analyzer = PhaseTransitionAnalyzer(config)
    result = analyzer.run_full_analysis()
    
    return result


def example_melting_silicon():
    """Example: Silicon melting transition."""
    
    # Create silicon
    si = bulk('Si', 'diamond', a=5.43)
    si = si * (3, 3, 3)  # Supercell
    
    write('si_crystal.xyz', si)
    
    config = PhaseTransitionConfig(
        initial_phase='si_crystal.xyz',
        transition_type='solid_liquid',
        sampling_method='metadynamics',
        temperature=1800,  # Near melting point
        mlip_type='mace'
    )
    
    analyzer = PhaseTransitionAnalyzer(config)
    result = analyzer.run_full_analysis()
    
    return result


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase Transition Analysis')
    parser.add_argument('--initial', '-i', required=True, help='Initial phase structure')
    parser.add_argument('--final', '-f', help='Final phase structure (optional)')
    parser.add_argument('--type', '-t', default='solid_liquid',
                       choices=['solid_liquid', 'martensitic', 'order_disorder'],
                       help='Transition type')
    parser.add_argument('--method', '-m', default='metadynamics',
                       choices=['metadynamics', 'umbrella'],
                       help='Sampling method')
    parser.add_argument('--temperature', type=float, default=1000,
                       help='Temperature (K)')
    parser.add_argument('--mlip', default='chgnet',
                       choices=['mace', 'chgnet', 'orb'],
                       help='MLIP type')
    parser.add_argument('--model', help='Path to trained model')
    parser.add_argument('--output', '-o', default='./phase_transition_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    config = PhaseTransitionConfig(
        initial_phase=args.initial,
        final_phase=args.final,
        transition_type=args.type,
        sampling_method=args.method,
        temperature=args.temperature,
        mlip_type=args.mlip,
        mlip_model_path=args.model
    )
    
    analyzer = PhaseTransitionAnalyzer(config)
    result = analyzer.run_full_analysis()
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'phase_transition_result.json', 'w') as f:
        json.dump({
            'success': result['success'],
            'sampling_method': args.method
        }, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()
