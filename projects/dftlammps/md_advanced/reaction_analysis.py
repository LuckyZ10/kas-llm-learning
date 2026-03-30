#!/usr/bin/env python3
"""
Reaction Analysis
=================

Tools for reaction path analysis and kinetics:
- Automatic reaction path search
- Reaction coordinate definition
- Rate constant calculation
- Kinetic Monte Carlo preprocessing

References:
- Henkelman et al. - Dimer/NEB methods
- Peters et al. - Growing string method
- Voter - Accelerated MD and KMC
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from functools import partial
import logging
from itertools import combinations
import networkx as nx

# ASE
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS, FIRE, LBFGS
from ase.neighborlist import NeighborList, natural_cutoffs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReactionPathSearchConfig:
    """Configuration for automatic reaction path search.
    
    Attributes:
        search_method: 'neb', 'string', 'dimer', or 'gsm'
        n_initials: Number of initial path guesses
        barrier_threshold: Maximum barrier to consider (eV)
        max_paths: Maximum number of paths to find
        local_minima_tolerance: RMSD threshold for identical minima
    """
    search_method: str = "neb"
    n_initials: int = 10
    barrier_threshold: float = 3.0
    max_paths: int = 5
    local_minima_tolerance: float = 0.3
    output_dir: str = "./reaction_paths"


@dataclass
class ReactionCoordinateConfig:
    """Configuration for reaction coordinate analysis.
    
    Attributes:
        cv_type: 'distance', 'angle', 'coordination', 'pca', or 'path'
        n_components: Number of principal components
        resolution: Grid resolution for path CV
    """
    cv_type: str = "path"
    n_components: int = 2
    resolution: int = 100
    smooth_factor: float = 0.1


@dataclass
class RateConstantConfig:
    """Configuration for rate constant calculation.
    
    Attributes:
        method: 'tst', 'stst', 'rpmd', or 'instanton'
        temperature_range: (T_min, T_max) in Kelvin
        n_temperatures: Number of temperature points
        include_tunneling: Include quantum tunneling
        tunneling_method: 'wigner', 'eckart', or 'skst'
    """
    method: str = "tst"
    temperature_range: Tuple[float, float] = (300.0, 1000.0)
    n_temperatures: int = 20
    include_tunneling: bool = False
    tunneling_method: str = "wigner"


@dataclass
class KMCConfig:
    """Configuration for Kinetic Monte Carlo preprocessing.
    
    Attributes:
        max_events: Maximum number of events to catalog
        hop_distance: Maximum hop distance for diffusion
        neighbor_cutoff: Cutoff for neighbor list
        supercell_size: Supercell for KMC simulation
        temperature: KMC simulation temperature
        simulation_time: Total simulation time (seconds)
    """
    max_events: int = 1000
    hop_distance: float = 3.5
    neighbor_cutoff: float = 4.0
    supercell_size: Tuple[int, int, int] = (4, 4, 4)
    temperature: float = 300.0
    simulation_time: float = 1.0  # seconds


class ReactionPathSearcher:
    """Automatic reaction path search using various methods."""
    
    def __init__(self, config: ReactionPathSearchConfig):
        self.config = config
        self.discovered_paths: List[Dict] = []
        self.minima_database: List[Atoms] = []
    
    def generate_initial_paths(self, reactant: Atoms, product: Atoms) -> List[List[Atoms]]:
        """Generate initial path guesses."""
        paths = []
        
        # Linear interpolation
        linear = self._linear_interpolation(reactant, product, n_images=10)
        paths.append(linear)
        
        # Perturbed paths for exploration
        for i in range(self.config.n_initials - 1):
            perturbed = self._perturb_path(linear, amplitude=0.3)
            paths.append(perturbed)
        
        return paths
    
    def _linear_interpolation(self, initial: Atoms, final: Atoms,
                             n_images: int = 10) -> List[Atoms]:
        """Create linear interpolated path."""
        path = [initial.copy()]
        
        for i in range(1, n_images + 1):
            t = i / (n_images + 1)
            image = initial.copy()
            image.positions = initial.positions * (1 - t) + final.positions * t
            path.append(image)
        
        path.append(final.copy())
        return path
    
    def _perturb_path(self, path: List[Atoms], amplitude: float = 0.3) -> List[Atoms]:
        """Perturb a path for exploration."""
        perturbed = []
        
        for image in path:
            new_image = image.copy()
            noise = np.random.randn(*image.positions.shape) * amplitude
            new_image.positions += noise
            perturbed.append(new_image)
        
        return perturbed
    
    def optimize_path(self, path: List[Atoms],
                     calculator: Optional[Any] = None) -> Dict:
        """Optimize path using NEB or string method."""
        from .rare_events import NEB, NEBConfig
        
        config = NEBConfig(
            n_images=len(path) - 2,
            neb_method='ase',
            climb=True,
            output_dir=self.config.output_dir
        )
        
        neb = NEB(config)
        result = neb.run_ase_neb(path, calculator)
        
        return result
    
    def search_all_paths(self, reactant: Atoms, product: Atoms,
                        calculator: Optional[Any] = None) -> List[Dict]:
        """Search for multiple reaction paths."""
        logger.info("Starting reaction path search")
        
        initial_paths = self.generate_initial_paths(reactant, product)
        
        for i, path in enumerate(initial_paths):
            if len(self.discovered_paths) >= self.config.max_paths:
                break
            
            logger.info(f"Optimizing path {i + 1}/{len(initial_paths)}")
            
            result = self.optimize_path(path, calculator)
            
            if result['success'] and result['barrier'] < self.config.barrier_threshold:
                # Check if this is a new path
                if self._is_new_path(result):
                    self.discovered_paths.append(result)
                    logger.info(f"Found new path with barrier: {result['barrier']:.3f} eV")
        
        # Sort by barrier height
        self.discovered_paths.sort(key=lambda x: x['barrier'])
        
        return self.discovered_paths
    
    def _is_new_path(self, new_path: Dict) -> bool:
        """Check if path is unique compared to discovered paths."""
        new_energies = np.array(new_path['energies'])
        
        for path in self.discovered_paths:
            existing_energies = np.array(path['energies'])
            
            if len(new_energies) == len(existing_energies):
                rmsd = np.sqrt(np.mean((new_energies - existing_energies) ** 2))
                if rmsd < 0.1:  # eV
                    return False
        
        return True
    
    def catalog_local_minima(self, trajectories: List[List[Atoms]],
                           calculator: Optional[Any] = None) -> List[Atoms]:
        """Catalog local minima from trajectories."""
        for traj in trajectories:
            for atoms in traj:
                if calculator:
                    atoms.calc = calculator
                    try:
                        forces = atoms.get_forces()
                        max_force = np.max(np.linalg.norm(forces, axis=1))
                        
                        # If forces are small, might be a minimum
                        if max_force < 0.1:
                            if self._is_new_minimum(atoms):
                                self.minima_database.append(atoms.copy())
                    except:
                        pass
        
        return self.minima_database
    
    def _is_new_minimum(self, atoms: Atoms) -> bool:
        """Check if atoms configuration is a new minimum."""
        for existing in self.minima_database:
            rmsd = self._compute_rmsd(atoms, existing)
            if rmsd < self.config.local_minima_tolerance:
                return False
        return True
    
    def _compute_rmsd(self, atoms1: Atoms, atoms2: Atoms) -> float:
        """Compute RMSD between two configurations."""
        if len(atoms1) != len(atoms2):
            return float('inf')
        
        # Simple RMSD (no alignment)
        diff = atoms1.positions - atoms2.positions
        return np.sqrt(np.mean(diff ** 2))


class ReactionCoordinate:
    """Define and analyze reaction coordinates."""
    
    def __init__(self, config: ReactionCoordinateConfig):
        self.config = config
        self.cv_function: Optional[Callable] = None
        self.path_images: Optional[List[Atoms]] = None
    
    def define_path_cv(self, path_images: List[Atoms]) -> Callable:
        """Define path-based collective variable.
        
        s(R) = (i + u) / (N - 1)
        where i is closest image and u is progress to next image
        """
        self.path_images = path_images
        
        def path_cv(atoms: Atoms) -> float:
            positions = atoms.positions.flatten()
            
            # Find closest image
            min_dist = float('inf')
            closest_idx = 0
            
            for i, image in enumerate(path_images):
                image_pos = image.positions.flatten()
                dist = np.linalg.norm(positions - image_pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            
            # Compute progress to next image
            if closest_idx < len(path_images) - 1:
                current = path_images[closest_idx].positions.flatten()
                next_img = path_images[closest_idx + 1].positions.flatten()
                
                v = next_img - current
                v_norm = v / np.linalg.norm(v)
                
                displacement = positions - current
                u = np.dot(displacement, v_norm) / np.linalg.norm(v)
                u = max(0, min(1, u))
            else:
                u = 0
            
            s = (closest_idx + u) / (len(path_images) - 1)
            return s
        
        self.cv_function = path_cv
        return path_cv
    
    def compute_pca_coordinates(self, trajectories: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute PCA-based reaction coordinates."""
        from sklearn.decomposition import PCA
        
        # Flatten trajectories
        X = np.vstack([t.reshape(t.shape[0], -1) for t in trajectories])
        
        # Fit PCA
        pca = PCA(n_components=self.config.n_components)
        pca.fit(X)
        
        self.pca = pca
        
        return pca.components_, pca.explained_variance_ratio_
    
    def define_distance_cv(self, atom_pairs: List[Tuple[int, int]]) -> Callable:
        """Define distance-based collective variable."""
        def distance_cv(atoms: Atoms) -> float:
            value = 0.0
            for i, j in atom_pairs:
                dist = np.linalg.norm(atoms.positions[i] - atoms.positions[j])
                value += dist
            return value / len(atom_pairs)
        
        self.cv_function = distance_cv
        return distance_cv
    
    def define_coordination_cv(self, central_atoms: List[int],
                              ligand_atoms: List[int],
                              cutoff: float = 3.0) -> Callable:
        """Define coordination number collective variable."""
        def coordination_cv(atoms: Atoms) -> float:
            coord = 0.0
            for i in central_atoms:
                for j in ligand_atoms:
                    if i != j:
                        dist = np.linalg.norm(atoms.positions[i] - atoms.positions[j])
                        # Switching function
                        coord += 1.0 / (1.0 + np.exp(10 * (dist - cutoff)))
            return coord
        
        self.cv_function = coordination_cv
        return coordination_cv
    
    def discretize_path(self, path_images: List[Atoms],
                       n_points: int = 50) -> np.ndarray:
        """Discretize path into evenly spaced points."""
        cv = self.define_path_cv(path_images)
        
        # Compute CV values for each image
        cv_values = [cv(img) for img in path_images]
        
        # Interpolate to uniform grid
        grid = np.linspace(0, 1, n_points)
        interpolated = np.interp(grid, cv_values, 
                                range(len(path_images)))
        
        return grid
    
    def project_trajectory(self, trajectory: List[Atoms]) -> np.ndarray:
        """Project trajectory onto reaction coordinate."""
        if self.cv_function is None:
            raise ValueError("CV function not defined")
        
        cv_values = [self.cv_function(atoms) for atoms in trajectory]
        return np.array(cv_values)


class RateConstantCalculator:
    """Calculate rate constants using various methods."""
    
    def __init__(self, config: RateConstantConfig):
        self.config = config
        self.temperatures: np.ndarray = np.linspace(
            config.temperature_range[0],
            config.temperature_range[1],
            config.n_temperatures
        )
    
    def compute_tst_rate(self, barrier: float, 
                        entropy: float = 0.0,
                        temperature: Optional[float] = None) -> float:
        """Compute TST rate constant.
        
        k = (k_B T / h) exp(-ΔG‡ / k_B T)
        """
        if temperature is None:
            temperature = self.config.temperature_range[0]
        
        kB = 8.617e-5  # eV/K
        h = 4.136e-15  # eV*s
        
        # Free energy of activation
        delta_g = barrier - temperature * entropy * 8.617e-5  # Convert entropy units
        
        # Rate constant
        k = (kB * temperature / h) * np.exp(-delta_g / (kB * temperature))
        
        return k
    
    def compute_arrhenius_rate(self, ea: float, a_factor: float,
                              temperature: Optional[float] = None) -> float:
        """Compute Arrhenius rate constant.
        
        k = A exp(-Ea / k_B T)
        """
        if temperature is None:
            temperature = self.config.temperature_range[0]
        
        kB = 8.617e-5  # eV/K
        
        return a_factor * np.exp(-ea / (kB * temperature))
    
    def wigner_tunneling(self, barrier: float, imag_freq: float,
                        temperature: Optional[float] = None) -> float:
        """Compute Wigner tunneling correction."""
        if temperature is None:
            temperature = self.config.temperature_range[0]
        
        h = 4.136e-15  # eV*s
        kB = 8.617e-5  # eV/K
        
        # Imaginary frequency in Hz (convert from THz)
        nu = abs(imag_freq) * 1e12
        
        beta = 1.0 / (kB * temperature)
        
        kappa = 1.0 + (1.0 / 24.0) * (beta * h * nu) ** 2
        
        return kappa
    
    def compute_rates_vs_temperature(self, barrier: float,
                                     ts_analysis: Dict) -> pd.DataFrame:
        """Compute rate constants across temperature range."""
        results = []
        
        for T in self.temperatures:
            k_tst = self.compute_tst_rate(barrier, temperature=T)
            
            if self.config.include_tunneling:
                imag_freq = ts_analysis.get('imaginary_frequency', 0)
                if imag_freq:
                    kappa = self.wigner_tunneling(barrier, imag_freq, T)
                    k_tst *= kappa
            
            results.append({
                'temperature': T,
                'k_tst': k_tst,
                'log10_k': np.log10(k_tst)
            })
        
        return pd.DataFrame(results)
    
    def fit_arrhenius(self, rates_df: pd.DataFrame) -> Dict:
        """Fit Arrhenius parameters from rate data."""
        T = rates_df['temperature'].values
        k = rates_df['k_tst'].values
        
        # Linear fit to ln(k) vs 1/T
        x = 1.0 / T
        y = np.log(k)
        
        # Linear regression
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        
        kB = 8.617e-5  # eV/K
        ea = -slope * kB  # eV
        a_factor = np.exp(intercept)
        
        return {
            'ea_ev': ea,
            'ea_kj_mol': ea * 96.485,
            'a_factor': a_factor,
            'r_squared': np.corrcoef(x, y)[0, 1] ** 2
        }
    
    def estimate_half_life(self, rate_constant: float,
                          concentration: float = 1.0) -> float:
        """Estimate half-life from rate constant."""
        return np.log(2) / (rate_constant * concentration)


class KMCPreprocessor:
    """Preprocess data for Kinetic Monte Carlo simulations."""
    
    def __init__(self, config: KMCConfig):
        self.config = config
        self.events: List[Dict] = []
        self.states: Dict[str, Dict] = {}
    
    def identify_diffusion_events(self, atoms: Atoms,
                                 mobile_species: List[str]) -> List[Dict]:
        """Identify possible diffusion events from structure."""
        events = []
        
        # Build neighbor list
        cutoffs = natural_cutoffs(atoms)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)
        
        symbols = atoms.get_chemical_symbols()
        
        for i, symbol in enumerate(symbols):
            if symbol in mobile_species:
                # Find neighbors within hop distance
                neighbors = []
                indices, offsets = nl.get_neighbors(i)
                
                for j, offset in zip(indices, offsets):
                    pos_i = atoms.positions[i]
                    pos_j = atoms.positions[j] + np.dot(offset, atoms.cell)
                    
                    dist = np.linalg.norm(pos_j - pos_i)
                    
                    if dist < self.config.hop_distance:
                        neighbors.append((j, dist, pos_j - pos_i))
                
                # Create diffusion events
                for j, dist, direction in neighbors:
                    event = {
                        'type': 'diffusion',
                        'atom_i': i,
                        'atom_j': j,
                        'distance': dist,
                        'direction': direction,
                        'species': symbol,
                        'rate': None  # To be computed from barrier
                    }
                    events.append(event)
        
        return events
    
    def catalog_reaction_events(self, barriers: Dict[str, float],
                               rate_method: str = 'tst') -> List[Dict]:
        """Catalog reaction events with rate constants."""
        events = []
        
        for reaction_name, barrier in barriers.items():
            # Compute rate constant
            if rate_method == 'tst':
                calculator = RateConstantCalculator(RateConstantConfig())
                rate = calculator.compute_tst_rate(barrier)
            else:
                rate = 0.0
            
            event = {
                'name': reaction_name,
                'type': 'reaction',
                'barrier': barrier,
                'rate': rate,
                'rate_300k': rate
            }
            events.append(event)
        
        self.events = events
        return events
    
    def build_kmc_lattice(self, atoms: Atoms,
                         supercell: Optional[Tuple[int, int, int]] = None) -> Dict:
        """Build lattice for KMC simulation."""
        if supercell is None:
            supercell = self.config.supercell_size
        
        # Create supercell
        from ase.build import make_supercell
        
        P = np.diag(supercell)
        super_atoms = make_supercell(atoms, P)
        
        # Identify sites
        sites = []
        for i, (pos, symbol) in enumerate(zip(super_atoms.positions, 
                                               super_atoms.get_chemical_symbols())):
            sites.append({
                'index': i,
                'position': pos,
                'species': symbol,
                'occupied': True
            })
        
        return {
            'lattice': super_atoms,
            'sites': sites,
            'supercell': supercell
        }
    
    def generate_kmc_input(self, output_file: str = "kmc_input.json"):
        """Generate input file for KMC simulation."""
        kmc_data = {
            'temperature': self.config.temperature,
            'simulation_time': self.config.simulation_time,
            'events': self.events,
            'n_events': len(self.events)
        }
        
        with open(output_file, 'w') as f:
            json.dump(kmc_data, f, indent=2, default=str)
        
        logger.info(f"KMC input written to {output_file}")
    
    def estimate_event_rates(self, barriers: Dict[str, float],
                            attempt_frequency: float = 1e13) -> Dict[str, float]:
        """Estimate event rates from barriers using Arrhenius."""
        kB = 8.617e-5  # eV/K
        T = self.config.temperature
        
        rates = {}
        for name, barrier in barriers.items():
            rate = attempt_frequency * np.exp(-barrier / (kB * T))
            rates[name] = rate
        
        return rates
    
    def create_state_graph(self, states: List[Atoms],
                          transitions: List[Tuple[int, int, float]]) -> nx.DiGraph:
        """Create directed graph of states and transitions."""
        G = nx.DiGraph()
        
        # Add states as nodes
        for i, state in enumerate(states):
            G.add_node(i, atoms=state)
        
        # Add transitions as edges
        for i, j, barrier in transitions:
            rate = self.estimate_event_rates({f'{i}-{j}': barrier})[f'{i}-{j}']
            G.add_edge(i, j, barrier=barrier, rate=rate)
        
        return G
    
    def find_diffusion_pathways(self, start_idx: int, end_idx: int,
                               max_length: int = 10) -> List[List[int]]:
        """Find possible diffusion pathways between sites."""
        # This would use the state graph
        # For now, return simple path
        return [[start_idx, end_idx]]
    
    def compute_residence_time(self, rate_out: float) -> float:
        """Compute mean residence time on a site."""
        return 1.0 / rate_out if rate_out > 0 else float('inf')
    
    def estimate_diffusion_coefficient(self, hop_distance: float,
                                      hop_rate: float,
                                      dimension: int = 3) -> float:
        """Estimate diffusion coefficient from hopping parameters."""
        # D = (1/2d) * Γ * a² for isotropic diffusion
        return (hop_rate * hop_distance ** 2) / (2 * dimension)


class ReactionNetwork:
    """Build and analyze reaction networks."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.species: Set[str] = set()
    
    def add_reaction(self, reactants: List[str], products: List[str],
                    rate_constant: float, barrier: float):
        """Add reaction to network."""
        reaction_id = f"{'+'.join(reactants)}->{'+'.join(products)}"
        
        self.graph.add_edge(
            tuple(reactants), tuple(products),
            rate=rate_constant,
            barrier=barrier,
            id=reaction_id
        )
        
        self.species.update(reactants)
        self.species.update(products)
    
    def find_pathways(self, start: str, end: str, max_length: int = 5) -> List[List[str]]:
        """Find reaction pathways between species."""
        try:
            paths = list(nx.all_simple_paths(
                self.graph.to_undirected(),
                (start,), (end,),
                cutoff=max_length
            ))
            return paths
        except nx.NetworkXNoPath:
            return []
    
    def compute_steady_state(self, initial_concentrations: Dict[str, float]) -> Dict[str, float]:
        """Compute steady-state concentrations."""
        # Simplified - would solve ODE system
        return initial_concentrations
    
    def rate_control_analysis(self) -> Dict[str, float]:
        """Perform degree of rate control analysis."""
        # Placeholder for DRC analysis
        return {}


# Export public API
__all__ = [
    'ReactionPathSearchConfig',
    'ReactionCoordinateConfig',
    'RateConstantConfig',
    'KMCConfig',
    'ReactionPathSearcher',
    'ReactionCoordinate',
    'RateConstantCalculator',
    'KMCPreprocessor',
    'ReactionNetwork'
]
