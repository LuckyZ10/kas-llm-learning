"""
Surface Catalytic Reaction Acceleration
=======================================

This example demonstrates the use of hyperdynamics and KMC to study
catalytic reactions on surfaces with rare event acceleration.

Surface reactions are challenging for MD because:
- Activation barriers are typically 0.5-2 eV
- Reaction rates are slow at experimental conditions
- Multiple reaction pathways compete

Key reactions:
--------------
- CO oxidation on Pt, Pd, Au
- Hydrogen evolution on metals
- Nitrogen reduction (Haber-Bosch)
- Methanol synthesis
- Selective hydrogenation

Methods:
--------
1. Hyperdynamics: Accelerate individual reaction events
2. KMC: Long-timescale catalytic cycles
3. Microkinetic modeling: Reaction network analysis

Key observables:
----------------
- Turnover frequency (TOF)
- Reaction selectivity
- Apparent activation energy
- Rate-determining step
- Coverage effects

Example:
--------
    from dftlammps.accelerated_dynamics import (
        HyperdynamicsConfig, HyperdynamicsSimulation
    )
    
    # Setup catalytic surface
    surface = setup_cat_surface('Pt', '111', size=(4, 4))
    add_adsorbates(surface, ['CO', 'O'], coverage=0.25)
    
    # Run accelerated dynamics
    config = HyperdynamicsConfig(boost_method='coordinate_boost')
    sim = HyperdynamicsSimulation(config)
    results = sim.run(surface, n_steps=1000000)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def setup_cat_surface(element: str = 'Pt',
                     facet: str = '111',
                     size: Tuple[int, int] = (4, 4),
                     n_layers: int = 4,
                     vacuum: float = 10.0) -> Any:
    """Setup catalytic surface.
    
    Args:
        element: Surface element
        facet: Surface facet
        size: Surface supercell size
        n_layers: Number of atomic layers
        vacuum: Vacuum spacing in Angstrom
        
    Returns:
        ASE Atoms object
    """
    try:
        from ase import Atoms
        from ase.build import fcc111, bcc110, hcp0001
        from ase.constraints import FixAtoms
    except ImportError:
        raise ImportError("ASE required")
    
    # Build surface based on element and facet
    if facet == '111':
        surface = fcc111(element, size=size, vacuum=vacuum, periodic=True)
    elif facet == '100':
        # Use generic FCC surface
        from ase.build import fcc100
        surface = fcc100(element, size=size, vacuum=vacuum, periodic=True)
    elif facet == '110':
        from ase.build import fcc110
        surface = fcc110(element, size=size, vacuum=vacuum, periodic=True)
    else:
        raise ValueError(f"Unknown facet: {facet}")
    
    # Fix bottom layers
    c = FixAtoms(indices=[atom.index for atom in surface if atom.position[2] < 
                         surface.get_positions()[:,2].mean() - 1.0])
    surface.set_constraint(c)
    
    logger.info(f"Created {element}({facet}) surface: {len(surface)} atoms")
    
    return surface


def add_adsorbates(surface: Any,
                   adsorbates: List[str],
                   sites: Optional[List[str]] = None,
                   coverage: float = 0.25) -> Any:
    """Add adsorbates to surface.
    
    Args:
        surface: Surface atoms
        adsorbates: List of adsorbate species
        sites: Adsorption sites (None = automatic)
        coverage: Surface coverage
        
    Returns:
        Modified surface with adsorbates
    """
    try:
        from ase.build import molecule, add_adsorbate
    except ImportError:
        raise ImportError("ASE required")
    
    # Calculate number of adsorbates based on coverage
    n_surface_atoms = len([a for a in surface if a.position[2] > 
                          surface.get_positions()[:,2].mean()])
    n_ads = int(n_surface_atoms * coverage)
    
    # Add adsorbates
    for ads in adsorbates:
        if ads == 'CO':
            mol = molecule('CO')
            # Position C down
            mol.rotate(180, 'y')
        elif ads == 'O':
            mol = Atoms('O', positions=[[0, 0, 0]])
        elif ads == 'H':
            mol = Atoms('H', positions=[[0, 0, 0]])
        elif ads == 'N':
            mol = Atoms('N', positions=[[0, 0, 0]])
        elif ads == 'OH':
            mol = Atoms('OH', positions=[[0, 0, 0], [0.96, 0, 0]])
        elif ads == 'H2O':
            mol = molecule('H2O')
        else:
            continue
        
        # Add to random fcc hollow sites
        # Simplified - would use proper site detection
        for i in range(n_ads // len(adsorbates)):
            x = np.random.uniform(0, surface.get_cell()[0, 0])
            y = np.random.uniform(0, surface.get_cell()[1, 1])
            add_adsorbate(surface, mol, height=1.5, position=(x, y))
    
    logger.info(f"Added {len(adsorbates)} types of adsorbates")
    
    return surface


def get_adsorption_sites(surface: Any,
                        site_type: str = 'fcc') -> np.ndarray:
    """Get positions of adsorption sites.
    
    Args:
        surface: Surface atoms
        site_type: Type of site ('fcc', 'hcp', 'bridge', 'top')
        
    Returns:
        Array of site positions
    """
    # Simplified - real implementation would detect sites from topology
    cell = surface.get_cell()[:2, :2]
    
    # Generate grid of sites
    nx, ny = 4, 4
    sites = []
    
    for i in range(nx):
        for j in range(ny):
            x = (i + 0.5) * cell[0, 0] / nx
            y = (j + 0.5) * cell[1, 1] / ny
            sites.append([x, y])
    
    return np.array(sites)


class CatalyticReactionAnalyzer:
    """Analyzer for catalytic reactions.
    
    Provides methods to:
    - Track reaction events
    - Calculate turnover frequencies
    - Analyze reaction mechanisms
    - Compute selectivity
    """
    
    def __init__(self,
                 trajectory: Any,
                 species_list: List[str],
                 binding_energies: Optional[Dict] = None):
        """Initialize analyzer.
        
        Args:
            trajectory: MD trajectory
            species_list: List of chemical species
            binding_energies: Binding energies for species (eV)
        """
        self.trajectory = trajectory
        self.species_list = species_list
        self.binding_energies = binding_energies or {}
        
        self.reaction_events = []
        self.surface_species = defaultdict(int)
    
    def identify_reaction_events(self,
                                 bond_cutoffs: Optional[Dict] = None) -> List[Dict]:
        """Identify chemical reaction events in trajectory.
        
        Args:
            bond_cutoffs: Dictionary of bond distance cutoffs
            
        Returns:
            List of reaction events
        """
        if bond_cutoffs is None:
            # Default bond cutoffs in Angstrom
            bond_cutoffs = {
                ('C', 'O'): 1.5,
                ('O', 'O'): 1.5,
                ('C', 'C'): 1.8,
                ('H', 'O'): 1.2,
                ('H', 'C'): 1.2,
                ('N', 'N'): 1.5,
                ('N', 'H'): 1.2,
                ('N', 'O'): 1.5,
            }
        
        events = []
        
        # Analyze each frame for bond changes
        # Simplified - would compare bonding networks
        
        logger.info(f"Identified {len(events)} reaction events")
        return events
    
    def calculate_turnover_frequency(self,
                                     reaction: str,
                                     active_sites: int,
                                     simulation_time: float) -> float:
        """Calculate turnover frequency (TOF).
        
        TOF = (number of product molecules) / (number of active sites * time)
        
        Args:
            reaction: Reaction name
            active_sites: Number of active sites
            simulation_time: Simulation time in seconds
            
        Returns:
            TOF in s^-1
        """
        n_products = sum(1 for e in self.reaction_events 
                        if e.get('reaction') == reaction)
        
        tof = n_products / (active_sites * simulation_time)
        
        return tof
    
    def calculate_selectivity(self,
                             products: List[str]) -> Dict[str, float]:
        """Calculate product selectivity.
        
        Selectivity_i = (rate to product i) / (total rate to all products)
        
        Args:
            products: List of product names
            
        Returns:
            Selectivity dictionary
        """
        product_counts = defaultdict(int)
        
        for event in self.reaction_events:
            prod = event.get('product')
            if prod in products:
                product_counts[prod] += 1
        
        total = sum(product_counts.values())
        
        selectivity = {}
        if total > 0:
            for prod in products:
                selectivity[prod] = product_counts[prod] / total
        
        return selectivity
    
    def analyze_coverage_effects(self) -> Dict:
        """Analyze how coverage affects reaction rates.
        
        Returns:
            Coverage analysis
        """
        # Track coverage over time
        coverage_history = []
        
        for atoms in self.trajectory:
            # Count adsorbates
            n_ads = len([a for a in atoms if a.symbol not in ['Pt', 'Pd', 'Au', 'Cu', 'Ni']])
            coverage = n_ads / len(atoms)  # Simplified
            coverage_history.append(coverage)
        
        return {
            'mean_coverage': np.mean(coverage_history),
            'coverage_fluctuation': np.std(coverage_history)
        }
    
    def identify_rate_determining_step(self,
                                       reaction_network: Dict) -> str:
        """Identify the rate-determining step.
        
        The RDS has the largest degree of rate control (DRC).
        
        Args:
            reaction_network: Dictionary of reactions and rates
            
        Returns:
            Name of rate-determining step
        """
        # Simplified - would compute degree of rate control
        max_rate = 0
        rds = None
        
        for step, rate in reaction_network.items():
            if rate > max_rate:
                max_rate = rate
                rds = step
        
        return rds or "unknown"


def build_reaction_network(reactions: List[Tuple[str, str, float, float]]) -> 'RateCatalog':
    """Build reaction network for KMC.
    
    Args:
        reactions: List of (name, reactants->products, Ea, prefactor)
        
    Returns:
        RateCatalog
    """
    from dftlammps.accelerated_dynamics import (
        RateCatalog, RateProcess, ProcessType
    )
    
    catalog = RateCatalog()
    
    for name, equation, Ea, A in reactions:
        # Parse equation: "A + B -> C + D"
        reactants, products = equation.split('->')
        
        process = RateProcess(
            name=name,
            initial_state=reactants.strip(),
            final_state=products.strip(),
            rate=0,  # Will be calculated from Arrhenius
            activation_energy=Ea,
            prefactor=A,
            process_type=ProcessType.REACTION,
            description=equation
        )
        catalog.add_process(process)
    
    return catalog


def run_cat_hyperdynamics(surface: Any,
                         reactants: List[str],
                         temperature: float = 500.0,
                         boost_method: str = 'coordinate_boost',
                         n_steps: int = 1000000) -> Dict:
    """Run hyperdynamics for catalytic reaction.
    
    Args:
        surface: Catalytic surface with adsorbates
        reactants: List of reactant species
        temperature: Temperature in K
        boost_method: Boost method
        n_steps: Number of MD steps
        
    Returns:
        Results dictionary
    """
    from dftlammps.accelerated_dynamics import (
        HyperdynamicsConfig, HyperdynamicsSimulation, BoostMethod
    )
    
    # Setup configuration
    if boost_method == 'coordinate_boost':
        cv = ['distance:0,1']  # Example: CO bond distance
        method = BoostMethod.COORDINATE_BOOST
    else:
        method = BoostMethod.BOND_BOOST
        cv = None
    
    config = HyperdynamicsConfig(
        boost_method=method,
        coordinate_cv=cv,
        delta_v_max=1.0,
        temperature=temperature,
        target_boost=1000.0
    )
    
    sim = HyperdynamicsSimulation(config)
    
    logger.info(f"Setup catalytic hyperdynamics: T={temperature}K, "
               f"method={boost_method}")
    
    return {
        'config': config,
        'simulation': sim
    }


def run_microkinetic_model(reactions: List[Dict],
                          temperature: float = 500.0,
                          pressure: Dict[str, float] = None,
                          initial_coverage: Dict[str, float] = None) -> Dict:
    """Run microkinetic model for catalytic reactions.
    
    Microkinetic modeling solves the coupled rate equations for
    all surface species.
    
    Args:
        reactions: List of reaction dictionaries
        temperature: Temperature in K
        pressure: Gas phase pressures (bar)
        initial_coverage: Initial surface coverage
        
    Returns:
        Results dictionary
    """
    try:
        from scipy.integrate import odeint
    except ImportError:
        raise ImportError("SciPy required for microkinetic modeling")
    
    k_B = 8.617333e-5  # eV/K
    
    # Setup species
    species = set()
    for r in reactions:
        species.update(r.get('reactants', []))
        species.update(r.get('products', []))
    
    species = list(species)
    species_idx = {s: i for i, s in enumerate(species)}
    
    # Initial conditions
    if initial_coverage is None:
        theta0 = np.zeros(len(species))
        theta0[species_idx.get('*', 0)] = 1.0  # Empty sites
    else:
        theta0 = np.array([initial_coverage.get(s, 0) for s in species])
    
    # Rate equations
    def rates(theta, t):
        dtheta = np.zeros(len(species))
        
        for r in reactions:
            # Calculate rate
            Ea = r.get('activation_energy', 0)
            A = r.get('prefactor', 1e13)
            k = A * np.exp(-Ea / (k_B * temperature))
            
            # Forward rate (simplified)
            rate = k
            for reactant in r.get('reactants', []):
                if reactant in species_idx:
                    rate *= theta[species_idx[reactant]]
            
            # Update derivatives
            for reactant in r.get('reactants', []):
                if reactant in species_idx:
                    dtheta[species_idx[reactant]] -= rate
            for product in r.get('products', []):
                if product in species_idx:
                    dtheta[species_idx[product]] += rate
        
        return dtheta
    
    # Integrate
    t = np.linspace(0, 1e6, 10000)  # Time in arbitrary units
    theta = odeint(rates, theta0, t)
    
    return {
        'time': t,
        'coverage': theta,
        'species': species,
        'steady_state_coverage': theta[-1]
    }


def sabatier_principle_plot(adsorption_energies: np.ndarray,
                           activation_energies: np.ndarray,
                           temperature: float = 500.0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Sabatier principle plot.
    
    The Sabatier principle states that optimal catalysts have
    intermediate binding energies - not too weak, not too strong.
    
    Args:
        adsorption_energies: Array of adsorption energies (eV)
        activation_energies: Corresponding activation energies (eV)
        temperature: Temperature (K)
        
    Returns:
        (adsorption_energies, turnover_frequencies)
    """
    k_B = 8.617333e-5
    
    # Simplified model: TOF depends on activation energy
    tof = np.exp(-activation_energies / (k_B * temperature))
    
    # Normalize
    tof = tof / np.max(tof)
    
    return adsorption_energies, tof


def example_co_oxidation():
    """Example: CO oxidation on Pt(111)."""
    
    print("=" * 60)
    print("SURFACE CATALYTIC REACTION: CO OXIDATION ON Pt(111)")
    print("=" * 60)
    
    # Setup surface
    surface = setup_cat_surface('Pt', '111', size=(4, 4))
    add_adsorbates(surface, ['CO', 'O'], coverage=0.25)
    
    print(f"\nSystem: Pt(111) with CO and O adsorbates")
    print(f"Total atoms: {len(surface)}")
    
    # Reaction network
    print("\nReaction Network:")
    print("  1. CO + * → CO*")
    print("  2. O2 + 2* → 2O*")
    print("  3. CO* + O* → CO2* + *")
    print("  4. CO2* → CO2 + *")
    
    # Example activation energies (eV)
    reactions = [
        ('CO_adsorption', 'CO + * -> CO*', 0.0, 1e13),
        ('O2_dissociation', 'O2 + 2* -> 2O*', 0.2, 1e13),
        ('CO_oxidation', 'CO* + O* -> CO2* + *', 0.8, 1e13),
        ('CO2_desorption', 'CO2* -> CO2 + *', 0.5, 1e13),
    ]
    
    print("\nActivation Energies:")
    for name, eq, Ea, A in reactions:
        print(f"  {name}: Ea = {Ea:.2f} eV")
    
    # Method comparison
    print("\n" + "-" * 60)
    print("Method Comparison for Catalytic Reactions")
    print("-" * 60)
    
    methods = {
        'Direct MD': '10 ps (barrier too high)',
        'Hyperdynamics': '10 ns with 1000x boost',
        'KMC': 'Seconds to hours',
        'Microkinetic': 'Steady-state solution'
    }
    
    for method, timescale in methods.items():
        print(f"  {method:<20}: {timescale}")
    
    # Expected results
    print("\n" + "-" * 60)
    print("Expected Results at 500 K:")
    print("-" * 60)
    
    k_B = 8.617333e-5
    T = 500.0
    
    for name, eq, Ea, A in reactions:
        if Ea > 0:
            rate = A * np.exp(-Ea / (k_B * T))
            print(f"  {name}: k = {rate:.2e} s^-1")
    
    print("\nKey challenges:")
    print("  - High activation barriers (0.5-2 eV)")
    print("  - Rare event statistics")
    print("  - Coverage-dependent rates")
    print("  - Multiple competing pathways")
    
    print("\n" + "=" * 60)
    
    return surface


def example_hydrogen_evolution():
    """Example: Hydrogen evolution reaction (HER)."""
    
    print("\n" + "=" * 60)
    print("SURFACE CATALYTIC REACTION: HYDROGEN EVOLUTION")
    print("=" * 60)
    
    # HER mechanism
    print("\nVolmer-Heyrovsky Mechanism:")
    print("  1. H+ + e- + * → H* (Volmer)")
    print("  2. H* + H+ + e- → H2 + * (Heyrovsky)")
    print("  3. 2H* → H2 + 2* (Tafel)")
    
    print("\nFree Energy Diagram (eV):")
    print("  H+ + e- + * : 0.0")
    print("  H*          : ΔG_H")
    print("  H2 + *      : 0.0")
    
    print("\nOptimal catalysts have ΔG_H ≈ 0:")
    print("  Pt: ΔG_H ≈ -0.09 eV (near optimal)")
    print("  MoS2 edge: ΔG_H ≈ 0.08 eV")
    print("  Strong binding (-ΔG_H >> 0): Poisoned surface")
    print("  Weak binding (ΔG_H >> 0): Hard to activate")
    
    print("=" * 60)


# Main example
if __name__ == "__main__":
    example_co_oxidation()
    example_hydrogen_evolution()
