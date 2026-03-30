#!/usr/bin/env python3
"""
Case Study: Catalytic Reaction Mechanism
========================================

Analysis of catalytic reaction mechanisms using reaction path search.

Examples:
- CO oxidation on metal surfaces
- Nitrogen reduction (NRR)
- Oxygen reduction (ORR)
- CO2 reduction

Workflow:
1. Build catalyst surface with adsorbates
2. Identify reaction intermediates
3. Search for minimum energy paths
4. Compute barriers and rate constants
5. Build microkinetic model
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
import logging
from collections import defaultdict

# ASE
from ase import Atoms
from ase.io import read, write
from ase.build import fcc111, add_adsorbate
from ase.constraints import FixAtoms
from ase.optimize import BFGS, FIRE
from ase.neighborlist import NeighborList, natural_cutoffs

# Local imports
from dftlammps.md_advanced.rare_events import (
    NEB, NEBConfig, StringMethod, StringMethodConfig, 
    DimerMethod, DimerConfig
)
from dftlammps.md_advanced.reaction_analysis import (
    ReactionPathSearcher, RateConstantCalculator,
    TSTConfig
)
from dftlammps.mlip_training import load_model, UnifiedMLIPCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CatalyticReactionConfig:
    """Configuration for catalytic reaction study.
    
    Attributes:
        catalyst_structure: Catalyst surface structure file
        reactants: List of reactant species
        products: List of product species
        active_site: Active site position or index
        surface_miller: Miller indices for surface
        supercell: Supercell size
        reaction_temperature: Temperature (K)
        mlip_type: MLIP type
        mlip_model_path: Path to trained model
    """
    catalyst_structure: Optional[str] = None
    catalyst_type: str = "Pt"
    surface_miller: Tuple[int, int, int] = (1, 1, 1)
    supercell: Tuple[int, int, int] = (3, 3, 4)
    reactants: List[str] = None
    products: List[str] = None
    reaction_type: str = "co_oxidation"
    active_site: Optional[Union[int, Tuple[float, float, float]]] = None
    reaction_temperature: float = 500.0
    pressure: float = 1.0  # atm
    mlip_type: str = "chgnet"
    mlip_model_path: Optional[str] = None


class CatalystBuilder:
    """Build catalyst surface structures."""
    
    def __init__(self, config: CatalyticReactionConfig):
        self.config = config
    
    def build_surface(self, element: str = None,
                     miller: Tuple[int, int, int] = None,
                     supercell: Tuple[int, int, int] = None) -> Atoms:
        """Build catalyst surface."""
        if element is None:
            element = self.config.catalyst_type
        if miller is None:
            miller = self.config.surface_miller
        if supercell is None:
            supercell = self.config.supercell
        
        # Build surface based on crystal structure
        if miller == (1, 1, 1):
            surface = fcc111(element, size=supercell, vacuum=10.0)
        else:
            # Generic surface (simplified)
            surface = fcc111(element, size=supercell, vacuum=10.0)
        
        # Fix bottom layers
        mask = [atom.tag > 2 for atom in surface]
        surface.set_constraint(FixAtoms(mask=mask))
        
        return surface
    
    def add_adsorbate(self, surface: Atoms,
                     species: str,
                     position: Tuple[float, float] = (0, 0),
                     height: float = 2.0) -> Atoms:
        """Add adsorbate to surface."""
        atoms = surface.copy()
        
        # Define adsorbate geometries (simplified)
        adsorbates = {
            'CO': Atoms('CO', positions=[[0, 0, 0], [0, 0, 1.13]]),
            'O': Atoms('O'),
            'O2': Atoms('O2', positions=[[0, 0, 0], [0, 0, 1.21]]),
            'CO2': Atoms('CO2', positions=[[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]]),
            'H': Atoms('H'),
            'H2': Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]]),
            'N': Atoms('N'),
            'N2': Atoms('N2', positions=[[0, 0, 0], [0, 0, 1.10]]),
            'NH': Atoms('NH', positions=[[0, 0, 0], [0, 0, 1.00]]),
            'NH2': Atoms('NH2', positions=[[0, 0, 0], [0.5, 0, 1.00], [-0.5, 0, 1.00]]),
            'NH3': Atoms('NH3'),  # Approximate
            'OH': Atoms('OH', positions=[[0, 0, 0], [0, 0, 0.98]]),
            'H2O': Atoms('H2O', positions=[[0, 0, 0], [0.96, 0, 0.3], [-0.96, 0, 0.3]]),
        }
        
        if species in adsorbates:
            ads = adsorbates[species]
            add_adsorbate(atoms, ads, height, position)
        else:
            logger.warning(f"Unknown adsorbate: {species}")
        
        return atoms
    
    def find_adsorption_sites(self, surface: Atoms,
                             site_type: str = 'all') -> Dict[str, List[Tuple[float, float]]]:
        """Find high-symmetry adsorption sites."""
        cell = surface.cell[:2, :2]
        
        # FCC (111) sites
        sites = {
            'top': [(0, 0), (1/3, 1/3), (2/3, 2/3)],
            'bridge': [(1/6, 1/6), (1/2, 1/2), (5/6, 5/6)],
            'fcc': [(1/3, 1/3)],
            'hcp': [(2/3, 2/3)]
        }
        
        # Convert fractional to Cartesian
        for site_type_key in sites:
            sites[site_type_key] = [
                tuple(np.dot(frac, cell)) for frac in sites[site_type_key]
            ]
        
        if site_type == 'all':
            return sites
        else:
            return {site_type: sites.get(site_type, [])}


class ReactionPathwayAnalyzer:
    """Analyze catalytic reaction pathways."""
    
    def __init__(self, config: CatalyticReactionConfig):
        self.config = config
        self.calculator = None
        self._setup_calculator()
    
    def _setup_calculator(self):
        """Setup MLIP calculator."""
        self.calculator = UnifiedMLIPCalculator(
            mlip_type=self.config.mlip_type,
            model_path=self.config.mlip_model_path
        )
    
    def relax_structure(self, atoms: Atoms,
                       fmax: float = 0.05,
                       max_steps: int = 500) -> Atoms:
        """Relax atomic structure."""
        atoms = atoms.copy()
        atoms.calc = self.calculator
        
        optimizer = BFGS(atoms, logfile=None)
        optimizer.run(fmax=fmax, steps=max_steps)
        
        return atoms
    
    def identify_intermediates(self, surface: Atoms) -> List[Dict]:
        """Identify possible reaction intermediates."""
        intermediates = []
        
        # Based on reaction type
        if self.config.reaction_type == 'co_oxidation':
            # CO + 1/2 O2 -> CO2
            intermediates = [
                {'name': 'CO*', 'species': ['CO']},
                {'name': 'O*', 'species': ['O']},
                {'name': 'O2*', 'species': ['O2']},
                {'name': 'CO-O*', 'species': ['CO', 'O']},
                {'name': 'CO2*', 'species': ['CO2']},
            ]
        elif self.config.reaction_type == 'nrr':
            # N2 reduction to NH3
            intermediates = [
                {'name': 'N2*', 'species': ['N2']},
                {'name': 'N*', 'species': ['N']},
                {'name': 'NH*', 'species': ['NH']},
                {'name': 'NH2*', 'species': ['NH2']},
                {'name': 'NH3*', 'species': ['NH3']},
            ]
        elif self.config.reaction_type == 'orr':
            # O2 reduction
            intermediates = [
                {'name': 'O2*', 'species': ['O2']},
                {'name': 'OOH*', 'species': ['O2', 'H']},
                {'name': 'O*', 'species': ['O']},
                {'name': 'OH*', 'species': ['OH']},
                {'name': 'H2O*', 'species': ['H2O']},
            ]
        
        return intermediates
    
    def compute_reaction_energy(self, initial: Atoms,
                               final: Atoms) -> float:
        """Compute reaction energy ΔE."""
        initial_relaxed = self.relax_structure(initial)
        final_relaxed = self.relax_structure(final)
        
        e_initial = initial_relaxed.get_potential_energy()
        e_final = final_relaxed.get_potential_energy()
        
        return e_final - e_initial
    
    def compute_barrier_neb(self, initial: Atoms,
                           final: Atoms,
                           n_images: int = 7) -> Dict:
        """Compute reaction barrier using NEB."""
        logger.info("Running NEB calculation")
        
        config = NEBConfig(
            n_images=n_images,
            k_spring=5.0,
            climb=True,
            fmax=0.05,
            neb_method='ase'
        )
        
        neb = NEB(config)
        
        # Create images
        images = neb.interpolate_images(initial, final)
        
        # Run NEB
        result = neb.run_ase_neb(images, calculator=self.calculator)
        
        return result
    
    def compute_barrier_dimer(self, initial: Atoms,
                             final: Atoms,
                             mode_direction: np.ndarray = None) -> Dict:
        """Find TS using dimer method."""
        logger.info("Running dimer method")
        
        config = DimerConfig(
            dimer_distance=0.01,
            max_steps=500,
            fmax=0.05
        )
        
        dimer = DimerMethod(config)
        
        # Start from midpoint
        midpoint = initial.copy()
        midpoint.positions = (initial.positions + final.positions) / 2
        
        if mode_direction is None:
            mode_direction = final.positions - initial.positions
        
        result = dimer.run(midpoint, mode_direction.flatten(), 
                          calculator=self.calculator)
        
        return result
    
    def build_energy_profile(self, intermediates: List[Atoms]) -> pd.DataFrame:
        """Build reaction energy profile."""
        energies = []
        
        for i, intermediate in enumerate(intermediates):
            relaxed = self.relax_structure(intermediate)
            energy = relaxed.get_potential_energy()
            
            energies.append({
                'step': i,
                'energy': energy
            })
        
        df = pd.DataFrame(energies)
        
        # Normalize to first intermediate
        df['relative_energy'] = df['energy'] - df['energy'].iloc[0]
        
        return df
    
    def compute_rate_constants(self, barriers: List[float],
                              temperatures: List[float] = None) -> pd.DataFrame:
        """Compute rate constants for barriers."""
        if temperatures is None:
            temperatures = [self.config.reaction_temperature]
        
        config = TSTConfig(temperature=self.config.reaction_temperature)
        calculator = RateConstantCalculator(config)
        
        results = []
        
        for i, barrier in enumerate(barriers):
            for T in temperatures:
                k = calculator.compute_tst_rate(barrier, temperature=T)
                
                results.append({
                    'step': i,
                    'barrier_ev': barrier,
                    'temperature_k': T,
                    'rate_constant_s': k,
                    'log10_k': np.log10(k)
                })
        
        return pd.DataFrame(results)
    
    def analyze_selectivity(self, pathways: List[List[Atoms]]) -> Dict:
        """Analyze selectivity between competing pathways."""
        pathway_data = []
        
        for i, pathway in enumerate(pathways):
            # Compute overall barrier
            energies = []
            for state in pathway:
                relaxed = self.relax_structure(state)
                energies.append(relaxed.get_potential_energy())
            
            barriers = [energies[j+1] - energies[j] for j in range(len(energies)-1)]
            max_barrier = max(barriers) if barriers else 0
            
            pathway_data.append({
                'pathway_id': i,
                'max_barrier': max_barrier,
                'overall_energy': energies[-1] - energies[0]
            })
        
        # Compute selectivity ratios
        if len(pathway_data) >= 2:
            kB = 8.617e-5  # eV/K
            T = self.config.reaction_temperature
            
            delta_G = pathway_data[0]['max_barrier'] - pathway_data[1]['max_barrier']
            selectivity = np.exp(-delta_G / (kB * T))
            
            pathway_data[0]['selectivity_ratio'] = selectivity
            pathway_data[1]['selectivity_ratio'] = 1.0 / selectivity
        
        return {
            'pathways': pathway_data,
            'preferred_pathway': 0 if pathway_data[0]['max_barrier'] < 
                                 pathway_data[1]['max_barrier'] else 1
        }
    
    def run_microkinetic_model(self, energy_profile: pd.DataFrame,
                              initial_concentrations: Dict[str, float],
                              simulation_time: float = 100.0) -> pd.DataFrame:
        """Run simplified microkinetic model."""
        # Simplified ODE integration for reaction kinetics
        # This is a basic implementation
        
        from scipy.integrate import odeint
        
        # Define rate equations
        def rates(y, t, k_fwd, k_rev):
            # Simple first-order kinetics
            dydt = []
            for i in range(len(y) - 1):
                r_fwd = k_fwd[i] * y[i]
                r_rev = k_rev[i] * y[i + 1]
                dydt.append(r_rev - r_fwd)
            dydt.append(k_fwd[-1] * y[-2] - k_rev[-1] * y[-1])
            return dydt
        
        # Placeholder rates
        k_fwd = [1.0] * len(energy_profile)
        k_rev = [0.5] * len(energy_profile)
        
        y0 = list(initial_concentrations.values())
        t = np.linspace(0, simulation_time, 1000)
        
        solution = odeint(rates, y0, t, args=(k_fwd, k_rev))
        
        results = pd.DataFrame(solution, columns=[f'species_{i}' for i in range(len(y0))])
        results['time'] = t
        
        return results


def example_co_oxidation_pt():
    """Example: CO oxidation on Pt(111)."""
    
    config = CatalyticReactionConfig(
        catalyst_type='Pt',
        surface_miller=(1, 1, 1),
        supercell=(3, 3, 4),
        reaction_type='co_oxidation',
        reaction_temperature=500,
        mlip_type='chgnet'
    )
    
    builder = CatalystBuilder(config)
    analyzer = ReactionPathwayAnalyzer(config)
    
    # Build surface
    surface = builder.build_surface()
    
    # Initial state: CO* + O*
    initial = builder.add_adsorbate(surface, 'CO', position=(0, 0), height=2.0)
    initial = builder.add_adsorbate(initial, 'O', position=(2.8, 1.6), height=2.0)
    
    # Final state: CO2*
    final = builder.add_adsorbate(surface, 'CO2', position=(1.4, 0.8), height=2.5)
    
    # Run NEB
    result = analyzer.compute_barrier_neb(initial, final, n_images=5)
    
    logger.info(f"CO oxidation barrier: {result.get('barrier', 0):.3f} eV")
    
    return result


def example_n2_reduction():
    """Example: Nitrogen reduction reaction."""
    
    config = CatalyticReactionConfig(
        catalyst_type='Fe',
        reaction_type='nrr',
        reaction_temperature=400,
        mlip_type='mace'
    )
    
    builder = CatalystBuilder(config)
    analyzer = ReactionPathwayAnalyzer(config)
    
    # Build surface
    surface = builder.build_surface()
    
    # Identify intermediates
    intermediates = analyzer.identify_intermediates(surface)
    logger.info(f"NRR intermediates: {[im['name'] for im in intermediates]}")
    
    # Build energy profile for associative pathway
    pathway = []
    for im in intermediates[:4]:
        state = surface.copy()
        for species in im['species']:
            state = builder.add_adsorbate(state, species)
        pathway.append(state)
    
    profile = analyzer.build_energy_profile(pathway)
    
    return profile


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Catalytic Reaction Analysis')
    parser.add_argument('--catalyst', '-c', default='Pt', help='Catalyst element')
    parser.add_argument('--miller', type=int, nargs=3, default=[1, 1, 1],
                       help='Miller indices')
    parser.add_argument('--reaction', '-r', default='co_oxidation',
                       choices=['co_oxidation', 'nrr', 'orr', 'co2_reduction'],
                       help='Reaction type')
    parser.add_argument('--temperature', '-t', type=float, default=500,
                       help='Temperature (K)')
    parser.add_argument('--mlip', default='chgnet',
                       choices=['mace', 'chgnet', 'orb'],
                       help='MLIP type')
    parser.add_argument('--model', help='Path to trained model')
    parser.add_argument('--output', '-o', default='./catalysis_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    config = CatalyticReactionConfig(
        catalyst_type=args.catalyst,
        surface_miller=tuple(args.miller),
        reaction_type=args.reaction,
        reaction_temperature=args.temperature,
        mlip_type=args.mlip,
        mlip_model_path=args.model
    )
    
    # Build catalyst and analyze
    builder = CatalystBuilder(config)
    analyzer = ReactionPathwayAnalyzer(config)
    
    surface = builder.build_surface()
    intermediates = analyzer.identify_intermediates(surface)
    
    logger.info(f"Found {len(intermediates)} intermediates for {args.reaction}")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'intermediates.json', 'w') as f:
        json.dump(intermediates, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()
