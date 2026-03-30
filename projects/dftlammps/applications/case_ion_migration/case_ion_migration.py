#!/usr/bin/env python3
"""
Case Study: Ion Migration Barriers
==================================

Computation of ion migration barriers using NEB with ML potentials.

Workflow:
1. Generate initial structure with ion at starting position
2. Identify migration path using Voronoi analysis
3. Run NEB calculation with ML potential
4. Analyze barriers and diffusion coefficients

Example: Li-ion migration in solid electrolyte (Li3PS4)
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

# ASE
from ase import Atoms
from ase.io import read, write
from ase.build import bulk
from ase.optimize import BFGS, FIRE
from ase.neighborlist import NeighborList, natural_cutoffs

# Local imports
from dftlammps.md_advanced.rare_events import NEB, NEBConfig
from dftlammps.mlip_training import load_model, UnifiedMLIPCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IonMigrationConfig:
    """Configuration for ion migration calculation.
    
    Attributes:
        ion_symbol: Symbol of migrating ion ('Li', 'Na', etc.)
        host_structure: Host structure filename or Atoms object
        start_site: Starting site index or position
        end_site: Ending site index or position
        n_images: Number of NEB images
        spring_constant: NEB spring constant
        mlip_type: MLIP type ('mace', 'chgnet', 'orb')
        mlip_model_path: Path to trained MLIP model
    """
    ion_symbol: str = "Li"
    host_structure: Optional[str] = None
    start_site: Optional[Union[int, Tuple[float, float, float]]] = None
    end_site: Optional[Union[int, Tuple[float, float, float]]] = None
    n_images: int = 7
    spring_constant: float = 5.0
    fmax: float = 0.05
    mlip_type: str = "chgnet"
    mlip_model_path: Optional[str] = None
    temperature: float = 300.0  # K


class MigrationPathAnalyzer:
    """Analyze possible migration paths in structure."""
    
    def __init__(self, atoms: Atoms, ion_symbol: str):
        self.atoms = atoms
        self.ion_symbol = ion_symbol
        self.ion_indices = [i for i, s in enumerate(atoms.get_chemical_symbols()) 
                          if s == ion_symbol]
        
    def find_equivalent_sites(self) -> List[int]:
        """Find crystallographically equivalent sites for ion."""
        sites = []
        
        for i, symbol in enumerate(self.atoms.get_chemical_symbols()):
            if symbol == self.ion_symbol:
                sites.append(i)
        
        return sites
    
    def find_nearest_neighbors(self, site_idx: int,
                              cutoff: float = 4.0) -> List[Tuple[int, float]]:
        """Find nearest neighboring sites."""
        neighbors = []
        
        pos_i = self.atoms.positions[site_idx]
        
        for j in self.ion_indices:
            if j == site_idx:
                continue
            
            pos_j = self.atoms.positions[j]
            dist = np.linalg.norm(pos_j - pos_i)
            
            if dist < cutoff:
                neighbors.append((j, dist))
        
        # Sort by distance
        neighbors.sort(key=lambda x: x[1])
        
        return neighbors
    
    def identify_channels(self, 
                         voronoi_tolerance: float = 0.5) -> List[Dict]:
        """Identify ion migration channels using Voronoi analysis."""
        from scipy.spatial import Voronoi
        
        # Compute Voronoi tessellation of ion positions
        ion_positions = self.atoms.positions[self.ion_indices]
        
        vor = Voronoi(ion_positions)
        
        channels = []
        
        # Analyze Voronoi ridges for possible channels
        for ridge_points, ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
            if -1 in ridge_vertices:
                continue  # Unbounded ridge
            
            i, j = ridge_points
            
            # Midpoint of ridge
            midpoint = (ion_positions[i] + ion_positions[j]) / 2
            
            # Check if midpoint is clear (no atoms blocking)
            channel_radius = self._estimate_channel_radius(midpoint)
            
            if channel_radius > voronoi_tolerance:
                channels.append({
                    'site_i': self.ion_indices[i],
                    'site_j': self.ion_indices[j],
                    'distance': np.linalg.norm(ion_positions[i] - ion_positions[j]),
                    'channel_radius': channel_radius,
                    'midpoint': midpoint
                })
        
        # Sort by channel radius
        channels.sort(key=lambda x: x['channel_radius'], reverse=True)
        
        return channels
    
    def _estimate_channel_radius(self, point: np.ndarray) -> float:
        """Estimate minimum radius at point (bottleneck)."""
        min_radius = float('inf')
        
        for pos in self.atoms.positions:
            dist = np.linalg.norm(pos - point)
            if dist < min_radius:
                min_radius = dist
        
        return min_radius / 2  # Approximate channel radius
    
    def suggest_migration_paths(self, max_paths: int = 5) -> List[Tuple[int, int]]:
        """Suggest likely migration paths."""
        paths = []
        
        # Method 1: Voronoi channels
        channels = self.identify_channels()
        
        for channel in channels[:max_paths]:
            paths.append((channel['site_i'], channel['site_j']))
        
        # Method 2: Nearest neighbors
        for site in self.ion_indices[:3]:
            neighbors = self.find_nearest_neighbors(site)
            for neighbor_idx, dist in neighbors[:3]:
                path = (site, neighbor_idx)
                if path not in paths and (neighbor_idx, site) not in paths:
                    paths.append(path)
        
        return paths[:max_paths]


class IonMigrationNEB:
    """NEB calculation for ion migration."""
    
    def __init__(self, config: IonMigrationConfig):
        self.config = config
        self.neb_config = NEBConfig(
            n_images=config.n_images,
            k_spring=config.spring_constant,
            climb=True,
            fmax=config.fmax,
            neb_method='ase'
        )
        self.neb = NEB(self.neb_config)
        self.calculator = None
        self._setup_calculator()
    
    def _setup_calculator(self):
        """Setup MLIP calculator."""
        self.calculator = UnifiedMLIPCalculator(
            mlip_type=self.config.mlip_type,
            model_path=self.config.mlip_model_path
        )
    
    def create_initial_final(self, atoms: Atoms,
                            start_idx: int,
                            end_idx: int) -> Tuple[Atoms, Atoms]:
        """Create initial and final configurations."""
        initial = atoms.copy()
        final = atoms.copy()
        
        # For vacancy mechanism, swap ion to final site
        # For interstitial, add ion
        
        # Simple case: ion hops from start to end
        # (assuming end is currently vacant)
        
        return initial, final
    
    def create_interpolation(self, initial: Atoms, final: Atoms,
                            start_idx: int, end_idx: int) -> List[Atoms]:
        """Create interpolated images for NEB."""
        images = [initial.copy()]
        
        # Get positions
        start_pos = initial.positions[start_idx].copy()
        end_pos = initial.positions[end_idx].copy()
        
        for i in range(1, self.config.n_images + 1):
            t = i / (self.config.n_images + 1)
            
            image = initial.copy()
            
            # Interpolate migrating ion position
            image.positions[start_idx] = start_pos * (1 - t) + end_pos * t
            
            images.append(image)
        
        images.append(final.copy())
        
        return images
    
    def relax_endpoints(self, atoms: Atoms,
                       fmax: float = 0.01,
                       max_steps: int = 500) -> Atoms:
        """Relax structure endpoints."""
        atoms = atoms.copy()
        atoms.calc = self.calculator
        
        optimizer = BFGS(atoms, logfile=None)
        optimizer.run(fmax=fmax, steps=max_steps)
        
        return atoms
    
    def run_neb(self, initial: Atoms, final: Atoms,
               migrating_ion_idx: int) -> Dict:
        """Run NEB calculation for migration."""
        logger.info("Running NEB calculation for ion migration")
        
        # Create images with migrating ion interpolated
        images = self.create_interpolation(initial, final, 
                                          migrating_ion_idx, 
                                          migrating_ion_idx)
        
        # Set calculator for all images
        for image in images:
            image.calc = self.calculator
        
        # Run NEB
        result = self.neb.run_ase_neb(images, calculator=self.calculator)
        
        # Add migration-specific analysis
        if result['success']:
            result['migration_analysis'] = self._analyze_migration(result)
        
        return result
    
    def _analyze_migration(self, neb_result: Dict) -> Dict:
        """Analyze migration results."""
        energies = neb_result['energies']
        
        # Find barrier
        barrier = np.max(energies) - energies[0]
        
        # Attempt frequency estimation (simplified)
        # Typical value for ion diffusion: 10^12 - 10^13 Hz
        nu0 = 1e13  # Hz
        
        # Compute rate constant using Arrhenius
        kB = 8.617e-5  # eV/K
        k = nu0 * np.exp(-barrier / (kB * self.config.temperature))
        
        # Diffusion coefficient (simplified 1D estimate)
        # D = (1/6) * a² * k * f
        # where a is hop distance, f is correlation factor (~1)
        
        return {
            'barrier_ev': float(barrier),
            'barrier_kj_mol': float(barrier * 96.485),
            'rate_constant_s': float(k),
            'attempt_frequency_hz': nu0,
            'temperature_k': self.config.temperature
        }
    
    def compute_diffusion_coefficient(self, barrier: float,
                                     hop_distance: float,
                                     dimension: int = 3) -> float:
        """Compute diffusion coefficient from barrier."""
        kB = 8.617e-5  # eV/K
        nu0 = 1e13  # Hz
        
        # Jump frequency
        gamma = nu0 * np.exp(-barrier / (kB * self.config.temperature))
        
        # Diffusion coefficient
        # D = (1/2d) * λ² * Γ * f
        # where λ is jump distance, Γ is jump rate, f is correlation factor
        correlation_factor = 1.0
        
        D = (hop_distance ** 2 * gamma * correlation_factor) / (2 * dimension)
        
        return D
    
    def run_full_analysis(self, atoms: Atoms,
                         start_site: int,
                         end_site: int) -> Dict:
        """Run complete migration analysis."""
        logger.info(f"Analyzing {self.config.ion_symbol} migration from site "
                   f"{start_site} to {end_site}")
        
        # Relax endpoints
        initial = self.relax_endpoints(atoms.copy())
        
        # Create final by moving ion
        final = initial.copy()
        final.positions[start_site] = initial.positions[end_site].copy()
        final = self.relax_endpoints(final)
        
        # Run NEB
        neb_result = self.run_neb(initial, final, start_site)
        
        # Compute additional properties
        hop_distance = np.linalg.norm(
            initial.positions[end_site] - initial.positions[start_site]
        )
        
        if neb_result['success']:
            barrier = neb_result['migration_analysis']['barrier_ev']
            D = self.compute_diffusion_coefficient(barrier, hop_distance)
            
            neb_result['migration_analysis']['diffusion_coefficient'] = float(D)
            neb_result['migration_analysis']['hop_distance_ang'] = float(hop_distance)
        
        return neb_result


def example_li3ps4_migration():
    """Example: Li migration in Li3PS4."""
    
    # Create simple Li3PS4-like structure
    # This is a simplified example - real structure would be from DFT/CIF
    
    from ase.build import bulk
    from ase import Atoms
    
    # Approximate Li3PS4 structure (gamma phase, Pnma)
    # Simplified representation
    
    a, b, c = 12.6, 6.0, 12.4  # Approximate lattice parameters
    
    # Create approximate positions
    positions = [
        # Li sites (partial occupancy in reality)
        [0.0, 0.25, 0.0],
        [0.25, 0.25, 0.25],
        [0.5, 0.75, 0.5],
        # P sites
        [0.25, 0.25, 0.0],
        # S sites
        [0.1, 0.25, 0.1],
        [0.4, 0.25, 0.4],
        [0.6, 0.75, 0.6],
        [0.9, 0.75, 0.9],
    ]
    
    symbols = ['Li'] * 3 + ['P'] + ['S'] * 4
    
    atoms = Atoms(symbols=symbols,
                  scaled_positions=positions[:len(symbols)],
                  cell=[a, b, c],
                  pbc=True)
    
    # For real application, load actual Li3PS4 structure
    # atoms = read('Li3PS4.cif')
    
    config = IonMigrationConfig(
        ion_symbol='Li',
        mlip_type='chgnet',
        n_images=5,
        spring_constant=5.0
    )
    
    migration = IonMigrationNEB(config)
    
    # Find migration paths
    analyzer = MigrationPathAnalyzer(atoms, 'Li')
    paths = analyzer.suggest_migration_paths(max_paths=3)
    
    logger.info(f"Found {len(paths)} possible migration paths")
    
    results = []
    for start, end in paths[:3]:
        result = migration.run_full_analysis(atoms, start, end)
        results.append(result)
        
        if result['success']:
            logger.info(f"Path {start}->{end}: "
                       f"Barrier = {result['migration_analysis']['barrier_ev']:.3f} eV")
    
    return results


def main():
    """Main entry point for ion migration analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ion Migration Barrier Calculation')
    parser.add_argument('--structure', '-s', required=True, help='Structure file')
    parser.add_argument('--ion', '-i', default='Li', help='Ion symbol')
    parser.add_argument('--start', type=int, default=0, help='Starting site index')
    parser.add_argument('--end', type=int, default=1, help='Ending site index')
    parser.add_argument('--mlip', default='chgnet', 
                       choices=['mace', 'chgnet', 'orb'],
                       help='MLIP type')
    parser.add_argument('--model', help='Path to trained model')
    parser.add_argument('--output', '-o', default='./migration_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Load structure
    atoms = read(args.structure)
    
    # Setup configuration
    config = IonMigrationConfig(
        ion_symbol=args.ion,
        mlip_type=args.mlip,
        mlip_model_path=args.model
    )
    
    # Run calculation
    migration = IonMigrationNEB(config)
    result = migration.run_full_analysis(atoms, args.start, args.end)
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'migration_result.json', 'w') as f:
        json.dump({
            'success': result['success'],
            'barrier_ev': result.get('migration_analysis', {}).get('barrier_ev'),
            'diffusion_coefficient': result.get('migration_analysis', {}).get('diffusion_coefficient')
        }, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()
