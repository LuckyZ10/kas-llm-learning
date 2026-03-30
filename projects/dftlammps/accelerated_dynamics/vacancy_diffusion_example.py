"""
Vacancy Diffusion Long Timescale Simulation
===========================================

This example demonstrates the use of hyperdynamics and KMC to study
vacancy diffusion in materials over experimentally relevant timescales.

Vacancy diffusion is important for:
- Creep deformation
- Radiation damage recovery
- Solute transport
- Phase transformations

Methods:
--------
1. Hyperdynamics: Accelerate MD to capture individual hops
2. KMC: State-to-state simulation for long timescales
3. Rate extraction: Get rates from NEB or MD

Key observables:
----------------
- Diffusion coefficient D(T)
- Activation energy Q
- Correlation factor f
- Jump frequency

Systems:
--------
- Vacancies in bulk metals (Cu, Al, Ni)
- Vacancies in ceramics (MgO, Al2O3)
- Grain boundary diffusion
- Dislocation core diffusion

Example:
--------
    from dftlammps.accelerated_dynamics import (
        HyperdynamicsConfig, HyperdynamicsSimulation, RateCatalog
    )
    
    # Method 1: Hyperdynamics
    config = HyperdynamicsConfig(boost_method='bond_boost', q_cutoff=0.2)
    sim = HyperdynamicsSimulation(config)
    results = sim.run(atoms, n_steps=1000000)
    
    # Method 2: KMC with rates from NEB
    catalog = build_vacancy_rate_catalog(atoms, n_images=7)
    kmc_results = run_kmc(catalog, initial_state, n_steps=1000000)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def create_bulk_with_vacancy(element: str = 'Cu',
                             lattice_constant: float = 3.615,
                             supercell: Tuple[int, int, int] = (4, 4, 4),
                             vacancy_site: Optional[int] = None) -> Any:
    """Create bulk material with single vacancy.
    
    Args:
        element: Chemical element
        lattice_constant: Lattice constant in Angstrom
        supercell: Supercell dimensions
        vacancy_site: Site index for vacancy (None = center)
        
    Returns:
        ASE Atoms object
    """
    try:
        from ase import Atoms
        from ase.build import bulk
    except ImportError:
        raise ImportError("ASE required")
    
    # Create bulk structure
    atoms = bulk(element, 'fcc', a=lattice_constant, cubic=True)
    
    # Make supercell
    atoms = atoms * supercell
    
    # Create vacancy by removing atom
    if vacancy_site is None:
        # Remove center atom
        n_atoms = len(atoms)
        vacancy_site = n_atoms // 2
    
    del atoms[vacancy_site]
    
    logger.info(f"Created {element} supercell with vacancy: {len(atoms)} atoms")
    
    return atoms


def get_neighbor_sites(atoms: Any, 
                       vacancy_idx: int,
                       coordination: str = 'fcc') -> List[int]:
    """Get neighbor sites for vacancy hopping.
    
    Args:
        atoms: Atomic configuration
        vacancy_idx: Current vacancy index (after removal, use nearest atom)
        coordination: Crystal structure type
        
    Returns:
        List of neighbor site indices
    """
    positions = atoms.get_positions()
    
    # Find nearest neighbors of vacancy position
    # In practice, this requires knowledge of lattice structure
    
    if coordination == 'fcc':
        n_neighbors = 12  # FCC coordination
    elif coordination == 'bcc':
        n_neighbors = 8   # BCC coordination
    else:
        n_neighbors = 12
    
    # Find n_neighbors closest atoms to vacancy site
    # This is simplified - real implementation would use lattice vectors
    
    logger.info(f"Found {n_neighbors} neighbor sites for {coordordination}")
    
    return list(range(n_neighbors))


class VacancyDiffusionAnalyzer:
    """Analyzer for vacancy diffusion.
    
    Provides methods to:
    - Track vacancy trajectory
    - Calculate diffusion coefficient
    - Compute jump statistics
    - Analyze correlation effects
    """
    
    def __init__(self, 
                 trajectory: np.ndarray,
                 lattice_vectors: np.ndarray,
                 initial_vacancy_pos: np.ndarray):
        """Initialize analyzer.
        
        Args:
            trajectory: Atomic positions [n_frames, n_atoms, 3]
            lattice_vectors: Lattice vectors [3, 3]
            initial_vacancy_pos: Initial vacancy position
        """
        self.trajectory = trajectory
        self.lattice = lattice_vectors
        self.initial_vacancy = initial_vacancy_pos
        
        self.n_frames = len(trajectory)
        self.vacancy_trajectory = []
    
    def track_vacancy(self, method: str = 'site_occupation') -> np.ndarray:
        """Track vacancy position over time.
        
        Args:
            method: Tracking method ('site_occupation' or 'centroid')
            
        Returns:
            Vacancy trajectory [n_frames, 3]
        """
        vacancy_traj = np.zeros((self.n_frames, 3))
        
        if method == 'site_occupation':
            # Track by finding empty lattice site
            for frame in range(self.n_frames):
                # Find site with no atom within threshold
                vacancy_traj[frame] = self._find_empty_site(frame)
        
        self.vacancy_trajectory = vacancy_traj
        return vacancy_traj
    
    def _find_empty_site(self, frame: int) -> np.ndarray:
        """Find empty lattice site at given frame."""
        # Simplified - would check all lattice sites
        return self.initial_vacancy.copy()
    
    def calculate_diffusion_coefficient(self,
                                        timestep: float,
                                        dimensionality: int = 3) -> float:
        """Calculate vacancy diffusion coefficient.
        
        D = MSD / (2 * d * t)
        
        Args:
            timestep: Time step in seconds
            dimensionality: Spatial dimension (1, 2, or 3)
            
        Returns:
            Diffusion coefficient in m^2/s
        """
        if len(self.vacancy_trajectory) == 0:
            self.track_vacancy()
        
        # Calculate MSD
        displacements = self.vacancy_trajectory - self.vacancy_trajectory[0]
        msd = np.mean(np.sum(displacements**2, axis=1))
        
        # Total time
        total_time = self.n_frames * timestep
        
        # D = MSD / (2 * d * t)
        D = msd * 1e-20 / (2 * dimensionality * total_time)  # Angstrom^2 to m^2
        
        return D
    
    def count_jumps(self, jump_threshold: float = 1.5) -> Tuple[int, np.ndarray]:
        """Count number of vacancy jumps.
        
        Args:
            jump_threshold: Minimum distance for a jump (Angstrom)
            
        Returns:
            (n_jumps, jump_times)
        """
        if len(self.vacancy_trajectory) == 0:
            self.track_vacancy()
        
        n_jumps = 0
        jump_times = []
        
        for i in range(1, self.n_frames):
            disp = np.linalg.norm(self.vacancy_trajectory[i] - 
                                 self.vacancy_trajectory[i-1])
            if disp > jump_threshold:
                n_jumps += 1
                jump_times.append(i)
        
        return n_jumps, np.array(jump_times)
    
    def calculate_jump_frequency(self, timestep: float) -> float:
        """Calculate average jump frequency.
        
        Args:
            timestep: Time step in seconds
            
        Returns:
            Jump frequency in Hz
        """
        n_jumps, _ = self.count_jumps()
        total_time = self.n_frames * timestep
        return n_jumps / total_time if total_time > 0 else 0.0
    
    def calculate_correlation_factor(self) -> float:
        """Calculate correlation factor for vacancy diffusion.
        
        For simple cubic: f = 0.653
        For FCC: f = 0.781
        For BCC: f = 0.727
        
        Returns:
            Correlation factor
        """
        # Simplified - real calculation requires detailed analysis
        # of successive jump directions
        
        # FCC correlation factor
        return 0.781


def build_vacancy_rate_catalog(atoms: Any,
                                neighbor_distances: List[float],
                                activation_energies: List[float],
                                prefactor: float = 1e13) -> 'RateCatalog':
    """Build rate catalog for vacancy diffusion.
    
    Args:
        atoms: Atomic configuration
        neighbor_distances: List of neighbor distances
        activation_energies: List of activation energies (eV)
        prefactor: Attempt frequency (Hz)
        
    Returns:
        RateCatalog
    """
    from dftlammps.accelerated_dynamics import RateCatalog, RateProcess, ProcessType
    
    catalog = RateCatalog()
    
    # Add hopping processes to each neighbor
    for i, (dist, Ea) in enumerate(zip(neighbor_distances, activation_energies)):
        process = RateProcess(
            name=f'vacancy_hop_{i}',
            initial_state='state_0',
            final_state='state_0',  # Same state (vacancy moved)
            rate=prefactor * np.exp(-Ea / (8.617333e-5 * 300)),  # at 300K
            activation_energy=Ea,
            prefactor=prefactor,
            process_type=ProcessType.DIFFUSION,
            description=f'Vacancy hop to neighbor at {dist:.2f} Angstrom'
        )
        catalog.add_process(process)
    
    logger.info(f"Built rate catalog with {len(catalog)} processes")
    
    return catalog


def run_vacancy_hyperdynamics(atoms: Any,
                              temperature: float = 1000.0,
                              q_cutoff: float = 0.2,
                              delta_v_max: float = 0.5,
                              n_steps: int = 100000) -> Dict:
    """Run hyperdynamics for vacancy diffusion.
    
    Args:
        atoms: Initial configuration
        temperature: Temperature in K
        q_cutoff: Bond-boost cutoff
        delta_v_max: Maximum bias (eV)
        n_steps: Number of MD steps
        
    Returns:
        Results dictionary
    """
    from dftlammps.accelerated_dynamics import (
        HyperdynamicsConfig, HyperdynamicsSimulation, BoostMethod
    )
    
    config = HyperdynamicsConfig(
        boost_method=BoostMethod.BOND_BOOST,
        q_cutoff=q_cutoff,
        delta_v_max=delta_v_max,
        temperature=temperature,
        target_boost=1000.0
    )
    
    sim = HyperdynamicsSimulation(config)
    
    # Run simulation
    # In real implementation, would use actual MD engine
    logger.info(f"Setup hyperdynamics: T={temperature}K, q_cutoff={q_cutoff}")
    
    return {
        'config': config,
        'simulation': sim
    }


def run_vacancy_kmc(initial_state: Any,
                   activation_energy: float = 0.5,
                   prefactor: float = 1e13,
                   temperature: float = 1000.0,
                   n_steps: int = 1000000) -> Dict:
    """Run KMC for vacancy diffusion.
    
    Args:
        initial_state: Initial state
        activation_energy: Vacancy migration energy (eV)
        prefactor: Attempt frequency (Hz)
        temperature: Temperature (K)
        n_steps: Number of KMC steps
        
    Returns:
        Results dictionary
    """
    from dftlammps.accelerated_dynamics import (
        KMCConfig, KMCSimulator, RateCatalog, RateProcess, State, ProcessType
    )
    
    # Build rate catalog
    catalog = RateCatalog()
    
    # FCC vacancy has 12 nearest neighbors
    for i in range(12):
        process = RateProcess(
            name=f'vacancy_hop_{i}',
            initial_state='state_0',
            final_state='state_0',
            rate=0,  # Will be calculated from Arrhenius
            activation_energy=activation_energy,
            prefactor=prefactor,
            process_type=ProcessType.DIFFUSION
        )
        catalog.add_process(process)
    
    # Setup and run KMC
    config = KMCConfig(
        temperature=temperature,
        n_steps=n_steps,
        max_time=1e6  # 1 second
    )
    
    sim = KMCSimulator(config, catalog)
    results = sim.run(initial_state)
    
    return {
        'results': results,
        'diffusion_coefficient': estimate_diffusion_from_kmc(results, 'fcc')
    }


def estimate_diffusion_from_kmc(kmc_results: Any,
                                structure: str = 'fcc',
                                lattice_constant: float = 3.615) -> float:
    """Estimate diffusion coefficient from KMC results.
    
    Args:
        kmc_results: KMCResults object
        structure: Crystal structure
        lattice_constant: Lattice constant (Angstrom)
        
    Returns:
        Diffusion coefficient (m^2/s)
    """
    # D = (1/6) * f * a^2 * Gamma
    # f = correlation factor
    # a = jump distance
    # Gamma = jump frequency
    
    correlation_factors = {
        'fcc': 0.781,
        'bcc': 0.727,
        'sc': 0.653
    }
    
    f = correlation_factors.get(structure, 0.781)
    a = lattice_constant / np.sqrt(2) * 1e-10  # FCC nearest neighbor to meters
    
    # Get jump frequency from KMC
    n_jumps = sum(kmc_results.event_counts.values())
    Gamma = n_jumps / kmc_results.total_time
    
    D = (1.0/6.0) * f * a**2 * Gamma
    
    return D


def compare_methods(temperatures: List[float],
                   activation_energy: float = 0.5,
                   prefactor: float = 1e13) -> Dict:
    """Compare direct MD, hyperdynamics, and KMC for vacancy diffusion.
    
    Args:
        temperatures: List of temperatures (K)
        activation_energy: Migration energy (eV)
        prefactor: Attempt frequency (Hz)
        
    Returns:
        Comparison results
    """
    k_B = 8.617333e-5  # eV/K
    
    results = {
        'temperatures': temperatures,
        'direct_md': [],
        'hyperdynamics': [],
        'kmc': []
    }
    
    for T in temperatures:
        rate = prefactor * np.exp(-activation_energy / (k_B * T))
        
        # Direct MD: limited to ~1 microsecond per day on large systems
        md_time = 1e-6  # seconds
        md_jumps = rate * md_time
        results['direct_md'].append(md_jumps)
        
        # Hyperdynamics: 100-1000x acceleration
        boost = 1000.0
        hd_jumps = rate * md_time * boost
        results['hyperdynamics'].append(hd_jumps)
        
        # KMC: can reach seconds
        kmc_time = 1.0  # seconds
        kmc_jumps = rate * kmc_time
        results['kmc'].append(kmc_jumps)
    
    return results


# Example usage
def example_vacancy_diffusion():
    """Example workflow for vacancy diffusion simulation."""
    
    print("=" * 60)
    print("VACANCY DIFFUSION LONG TIMESCALE SIMULATION")
    print("=" * 60)
    
    # Create system
    atoms = create_bulk_with_vacancy(element='Cu', supercell=(4, 4, 4))
    print(f"\nSystem: Cu supercell with vacancy ({len(atoms)} atoms)")
    
    # Method comparison
    print("\n" + "-" * 60)
    print("Method Comparison for Vacancy Diffusion in Cu")
    print("-" * 60)
    
    temps = [300, 500, 700, 900, 1100]
    comparison = compare_methods(temps, activation_energy=0.68)  # Cu vacancy EM
    
    print(f"\n{'T (K)':<10} {'MD Jumps':<15} {'HD Jumps':<15} {'KMC Jumps':<15}")
    print("-" * 60)
    
    for i, T in enumerate(temps):
        print(f"{T:<10} {comparison['direct_md'][i]:<15.2e} "
              f"{comparison['hyperdynamics'][i]:<15.2e} "
              f"{comparison['kmc'][i]:<15.2e}")
    
    print("\nKey points:")
    print("  - Direct MD: Limited to ~1 μs/day")
    print("  - Hyperdynamics: 100-1000x acceleration")
    print("  - KMC: Can reach seconds to hours")
    print("  - All methods converge to same D(T) in their valid regimes")
    
    print("\n" + "=" * 60)
    
    return atoms


if __name__ == "__main__":
    example_vacancy_diffusion()
