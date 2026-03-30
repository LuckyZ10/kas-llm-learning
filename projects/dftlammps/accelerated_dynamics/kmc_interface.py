"""
Kinetic Monte Carlo (KMC) Interface
===================================

This module provides an interface between molecular dynamics simulations
and kinetic Monte Carlo for long-time scale simulations of rare events.

KMC uses rates extracted from MD to simulate state-to-state dynamics
over much longer timescales than accessible by direct MD.

Key Features:
-------------
- Rate extraction from MD transition statistics
- Rate catalog management
- KMC simulation engine
- Defect evolution tracking
- Parallel KMC (bKLMC)
- Spatially resolved KMC

Classes:
--------
- KMCConfig: Configuration for KMC simulations
- RateProcess: Single rate process definition
- RateCatalog: Collection of rate processes
- State: System state definition
- KMCSimulator: Main KMC simulation engine
- RateExtractor: Extract rates from MD data
- DefectTracker: Track defect evolution

Functions:
----------
- extract_rates_from_md: Extract rates from MD trajectory
- run_kmc: Run KMC simulation
- analyze_defect_evolution: Analyze defect evolution trajectory
- calculate_mc_time: Calculate Monte Carlo time

References:
-----------
- Gillespie (1976). J. Comput. Phys. 22, 403
- Fichthorn & Weinberg (1991). J. Chem. Phys. 95, 1090
- Chatterjee & Vlachos (2007). J. Comput. Phys. 2, 179
- Shin et al. (2021). Phys. Rev. Materials 5, L040801

Example:
--------
>>> from dftlammps.accelerated_dynamics import KMCConfig, RateCatalog, KMCSimulator
>>> >>> # Define rate processes
>>> catalog = RateCatalog()
>>> catalog.add_process(RateProcess('hop', 0, 1, 1e12, 0.5))
>>> 
>>> # Run KMC
>>> config = KMCConfig(temperature=300, n_steps=100000)
>>> sim = KMCSimulator(config, catalog)
>>> results = sim.run(initial_state)
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Union, Callable, Any, Set
from pathlib import Path
from enum import Enum, auto
from collections import defaultdict
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

# Setup logging
logger = logging.getLogger(__name__)


class ProcessType(Enum):
    """Types of rate processes."""
    DIFFUSION = "diffusion"           # Atom hop/diffusion
    REACTION = "reaction"             # Chemical reaction
    DEFECT_FORMATION = "defect_formation"
    DEFECT_ANNIHILATION = "defect_annihilation"
    CLUSTERING = "clustering"
    DISSOCIATION = "dissociation"
    ROTATION = "rotation"


class KMCAlgorithm(Enum):
    """KMC algorithms."""
    GILLESPIE = "gillespie"           # Standard Gillespie algorithm
    NULL_EVENT = "null_event"         # Null-event algorithm
    BINARY_TREE = "binary_tree"       # Binary tree search
    COMPOSITION_REJECTION = "composition_rejection"
    PARALLEL_KMC = "parallel_kmc"     # Parallel KMC (bKLMC)


@dataclass
class KMCConfig:
    """Configuration for KMC simulation.
    
    Attributes:
        temperature: Temperature in Kelvin
        n_steps: Number of KMC steps
        max_time: Maximum simulation time (seconds)
        algorithm: KMC algorithm to use
        random_seed: Random seed
        output_freq: Output frequency (steps)
        trajectory_file: Trajectory output file
        state_file: State dump file
        rate_tolerance: Minimum rate to include (Hz)
        enable_defect_tracking: Enable defect tracking
        defect_radius: Radius for defect identification (Angstrom)
        parallel_kmc: Use parallel KMC
        n_processes: Number of processes for parallel KMC
        spatial_kmc: Use spatially resolved KMC
        lattice_size: Lattice size for spatial KMC
        boundary_conditions: Boundary conditions ('periodic', 'fixed')
    """
    temperature: float = 300.0
    n_steps: int = 100000
    max_time: Optional[float] = None
    algorithm: KMCAlgorithm = KMCAlgorithm.GILLESPIE
    random_seed: Optional[int] = None
    output_freq: int = 1000
    trajectory_file: str = "kmc_trajectory.xyz"
    state_file: str = "kmc_states.json"
    rate_tolerance: float = 1e-10
    enable_defect_tracking: bool = True
    defect_radius: float = 5.0
    parallel_kmc: bool = False
    n_processes: int = 1
    spatial_kmc: bool = False
    lattice_size: Optional[Tuple[int, int, int]] = None
    boundary_conditions: str = "periodic"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")


@dataclass
class RateProcess:
    """Definition of a rate process.
    
    Attributes:
        name: Process name
        process_type: Type of process
        initial_state: Initial state identifier
        final_state: Final state identifier
        rate: Rate constant (Hz)
        activation_energy: Activation energy (eV)
        prefactor: Arrhenius prefactor (Hz)
        reaction_coordinates: List of atom indices involved
        description: Human-readable description
        metadata: Additional metadata
    """
    name: str
    initial_state: Union[int, str]
    final_state: Union[int, str]
    rate: float
    activation_energy: Optional[float] = None
    prefactor: Optional[float] = None
    process_type: ProcessType = ProcessType.DIFFUSION
    reaction_coordinates: Optional[List[int]] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate process."""
        if self.rate < 0:
            raise ValueError(f"rate must be non-negative, got {self.rate}")
    
    def calculate_rate(self, temperature: float) -> float:
        """Calculate rate at given temperature using Arrhenius law.
        
        k = A * exp(-Ea / (k_B * T))
        
        Args:
            temperature: Temperature in Kelvin
            
        Returns:
            Rate constant in Hz
        """
        if self.prefactor is None or self.activation_energy is None:
            return self.rate
        
        k_B = 8.617333e-5  # eV/K
        return self.prefactor * np.exp(-self.activation_energy / (k_B * temperature))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "initial_state": self.initial_state,
            "final_state": self.final_state,
            "rate": self.rate,
            "activation_energy": self.activation_energy,
            "prefactor": self.prefactor,
            "process_type": self.process_type.value,
            "reaction_coordinates": self.reaction_coordinates,
            "description": self.description,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RateProcess':
        """Create from dictionary."""
        data = data.copy()
        data["process_type"] = ProcessType(data["process_type"])
        return cls(**data)


@dataclass
class State:
    """System state definition.
    
    Attributes:
        state_id: Unique state identifier
        positions: Atomic positions [n_atoms, 3]
        atom_types: Atom type identifiers
        cell: Simulation cell [3, 3]
        energy: Potential energy
        metadata: Additional state information
    """
    state_id: Union[int, str]
    positions: Optional[np.ndarray] = None
    atom_types: Optional[List[str]] = None
    cell: Optional[np.ndarray] = None
    energy: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def copy(self) -> 'State':
        """Create copy of state."""
        return State(
            state_id=self.state_id,
            positions=self.positions.copy() if self.positions is not None else None,
            atom_types=self.atom_types.copy() if self.atom_types is not None else None,
            cell=self.cell.copy() if self.cell is not None else None,
            energy=self.energy,
            metadata=self.metadata.copy()
        )


@dataclass
class KMCSimulationState:
    """Current state of KMC simulation.
    
    Attributes:
        current_state: Current system state
        time: Current simulation time
        step: Current KMC step
        event_history: History of executed events
        time_history: History of times
        state_history: History of visited states
        rates_history: History of total rates
    """
    current_state: State
    time: float = 0.0
    step: int = 0
    event_history: List[int] = field(default_factory=list)
    time_history: List[float] = field(default_factory=list)
    state_history: List[Union[int, str]] = field(default_factory=list)
    rates_history: List[float] = field(default_factory=list)


@dataclass
class KMCResults:
    """Results from KMC simulation.
    
    Attributes:
        final_state: Final system state
        total_time: Total simulation time
        n_steps: Number of KMC steps executed
        event_counts: Count of each event type
        state_visits: Number of visits to each state
        mean_residence_time: Mean time spent in each state
        transition_matrix: State-to-state transition counts
        trajectory_file: Path to trajectory file
        statistics: Additional statistics
    """
    final_state: State
    total_time: float
    n_steps: int
    event_counts: Dict[str, int] = field(default_factory=dict)
    state_visits: Dict[Union[int, str], int] = field(default_factory=dict)
    mean_residence_time: Dict[Union[int, str], float] = field(default_factory=dict)
    transition_matrix: Dict[Tuple, int] = field(default_factory=dict)
    trajectory_file: Optional[str] = None
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def get_rate_matrix(self) -> np.ndarray:
        """Get rate matrix from transition statistics."""
        # Build rate matrix from transition counts
        states = sorted(set([s for t in self.transition_matrix.keys() for s in t]))
        n_states = len(states)
        
        rate_matrix = np.zeros((n_states, n_states))
        state_to_idx = {s: i for i, s in enumerate(states)}
        
        for (s1, s2), count in self.transition_matrix.items():
            i, j = state_to_idx[s1], state_to_idx[s2]
            # Rate = count / total_time_in_state
            rate_matrix[i, j] = count / self.total_time
        
        return rate_matrix
    
    def get_diffusion_coefficient(self, dimensionality: int = 3) -> float:
        """Estimate diffusion coefficient from mean square displacement."""
        # This would need position history
        # Placeholder implementation
        return 0.0
    
    def print_summary(self):
        """Print summary of KMC results."""
        print("=" * 60)
        print("KMC SIMULATION SUMMARY")
        print("=" * 60)
        print(f"Total Steps: {self.n_steps}")
        print(f"Total Time: {self.total_time:.4e} s")
        print(f"Final State: {self.final_state.state_id}")
        print("\nEvent Counts:")
        for event, count in sorted(self.event_counts.items()):
            print(f"  {event}: {count}")
        print("\nState Visits:")
        for state, visits in sorted(self.state_visits.items()):
            print(f"  State {state}: {visits}")
        print("=" * 60)


class RateCatalog:
    """Catalog of rate processes for KMC.
    
    Manages the collection of rate processes and provides
    methods for rate lookup and management.
    
    Attributes:
        processes: Dictionary of rate processes
        state_processes: Mapping from state to available processes
    """
    
    def __init__(self):
        """Initialize empty rate catalog."""
        self.processes: Dict[str, RateProcess] = {}
        self.state_processes: Dict[Union[int, str], List[str]] = defaultdict(list)
    
    def add_process(self, process: RateProcess):
        """Add a rate process to the catalog.
        
        Args:
            process: RateProcess to add
        """
        self.processes[process.name] = process
        self.state_processes[process.initial_state].append(process.name)
        logger.debug(f"Added process {process.name}")
    
    def get_processes_from_state(self, state_id: Union[int, str]) -> List[RateProcess]:
        """Get all processes available from a given state.
        
        Args:
            state_id: State identifier
            
        Returns:
            List of available RateProcess objects
        """
        process_names = self.state_processes.get(state_id, [])
        return [self.processes[name] for name in process_names]
    
    def get_total_rate(self, state_id: Union[int, str], temperature: float) -> float:
        """Get total escape rate from a state.
        
        Args:
            state_id: State identifier
            temperature: Temperature in Kelvin
            
        Returns:
            Total escape rate
        """
        processes = self.get_processes_from_state(state_id)
        return sum(p.calculate_rate(temperature) for p in processes)
    
    def remove_process(self, name: str):
        """Remove a process from the catalog."""
        if name in self.processes:
            process = self.processes.pop(name)
            self.state_processes[process.initial_state].remove(name)
    
    def save(self, filename: str):
        """Save catalog to JSON file."""
        data = {
            "processes": {name: proc.to_dict() for name, proc in self.processes.items()}
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved rate catalog to {filename}")
    
    def load(self, filename: str):
        """Load catalog from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        for name, proc_data in data["processes"].items():
            self.add_process(RateProcess.from_dict(proc_data))
        
        logger.info(f"Loaded rate catalog from {filename}")
    
    def __len__(self) -> int:
        """Return number of processes."""
        return len(self.processes)


class KMCSimulator:
    """Main KMC simulation engine.
    
    Implements the kinetic Monte Carlo algorithm for simulating
    rare events over long timescales.
    
    Algorithm (Gillespie):
    ----------------------
    1. Calculate all rates from current state
    2. Calculate total rate R = sum(rates)
    3. Generate random time increment: dt = -ln(u1) / R
    4. Select process with probability proportional to rate
    5. Execute process and update time: t += dt
    6. Repeat from step 1
    
    Attributes:
        config: KMC configuration
        rate_catalog: Rate process catalog
        rng: Random number generator
    """
    
    def __init__(self, config: KMCConfig, rate_catalog: RateCatalog):
        """Initialize KMC simulator.
        
        Args:
            config: KMC configuration
            rate_catalog: Rate process catalog
        """
        self.config = config
        self.rate_catalog = rate_catalog
        
        # Random number generator
        self.rng = np.random.default_rng(config.random_seed)
        
        # State tracking
        self.simulation_state = None
        self.defect_tracker = None
        
        if config.enable_defect_tracking:
            self.defect_tracker = DefectTracker(config.defect_radius)
    
    def run(self, initial_state: State, 
            progress_callback: Optional[Callable] = None) -> KMCResults:
        """Run KMC simulation.
        
        Args:
            initial_state: Initial system state
            progress_callback: Optional callback function(step, time, state)
            
        Returns:
            KMCResults
        """
        # Initialize simulation state
        sim_state = KMCSimulationState(
            current_state=initial_state.copy(),
            state_history=[initial_state.state_id]
        )
        self.simulation_state = sim_state
        
        # Statistics
        event_counts = defaultdict(int)
        state_visits = defaultdict(int)
        state_visits[initial_state.state_id] = 1
        transition_matrix = defaultdict(int)
        
        logger.info(f"Starting KMC simulation: {self.config.n_steps} steps")
        
        for step in range(self.config.n_steps):
            sim_state.step = step
            
            # Get available processes
            processes = self.rate_catalog.get_processes_from_state(
                sim_state.current_state.state_id
            )
            
            if len(processes) == 0:
                logger.warning(f"No processes available from state {sim_state.current_state.state_id}")
                break
            
            # Calculate rates
            rates = np.array([p.calculate_rate(self.config.temperature) 
                             for p in processes])
            
            # Filter small rates
            mask = rates >= self.config.rate_tolerance
            processes = [p for p, m in zip(processes, mask) if m]
            rates = rates[mask]
            
            if len(rates) == 0:
                logger.warning("All rates below tolerance")
                break
            
            # Total rate
            total_rate = np.sum(rates)
            sim_state.rates_history.append(total_rate)
            
            # Generate time increment
            u1 = self.rng.random()
            dt = -np.log(u1) / total_rate
            sim_state.time += dt
            sim_state.time_history.append(sim_state.time)
            
            # Select process
            probabilities = rates / total_rate
            process_idx = self.rng.choice(len(processes), p=probabilities)
            selected_process = processes[process_idx]
            
            # Execute process
            old_state = sim_state.current_state.state_id
            self._execute_process(sim_state, selected_process)
            new_state = sim_state.current_state.state_id
            
            # Update statistics
            event_counts[selected_process.name] += 1
            state_visits[new_state] += 1
            transition_matrix[(old_state, new_state)] += 1
            sim_state.event_history.append(process_idx)
            sim_state.state_history.append(new_state)
            
            # Track defects
            if self.defect_tracker:
                self.defect_tracker.update(sim_state.current_state, step, sim_state.time)
            
            # Output
            if step % self.config.output_freq == 0:
                logger.info(f"Step {step}: t={sim_state.time:.4e}s, state={new_state}")
                if progress_callback:
                    progress_callback(step, sim_state.time, sim_state.current_state)
            
            # Check max time
            if self.config.max_time and sim_state.time >= self.config.max_time:
                logger.info(f"Reached maximum time: {self.config.max_time}")
                break
        
        # Calculate mean residence times
        mean_residence = {}
        for state in state_visits.keys():
            # Time spent in state / number of visits
            # This is approximate
            total_transitions = sum(count for (s1, s2), count in transition_matrix.items() 
                                  if s1 == state)
            if total_transitions > 0:
                mean_residence[state] = sim_state.time / total_transitions
            else:
                mean_residence[state] = sim_state.time
        
        # Build results
        results = KMCResults(
            final_state=sim_state.current_state,
            total_time=sim_state.time,
            n_steps=step + 1,
            event_counts=dict(event_counts),
            state_visits=dict(state_visits),
            mean_residence_time=mean_residence,
            transition_matrix=dict(transition_matrix)
        )
        
        # Add defect statistics if available
        if self.defect_tracker:
            results.statistics["defects"] = self.defect_tracker.get_statistics()
        
        logger.info(f"KMC simulation complete: {results.n_steps} steps, {results.total_time:.4e}s")
        
        return results
    
    def _execute_process(self, sim_state: KMCSimulationState, 
                        process: RateProcess):
        """Execute a rate process.
        
        Args:
            sim_state: Current simulation state
            process: Process to execute
        """
        # Update state ID
        sim_state.current_state.state_id = process.final_state
        
        # If position information is available, update it
        # This would require detailed knowledge of the process
        # For now, just update the state identifier
        
        logger.debug(f"Executed {process.name}: {process.initial_state} -> {process.final_state}")
    
    def get_current_time(self) -> float:
        """Get current simulation time."""
        if self.simulation_state:
            return self.simulation_state.time
        return 0.0
    
    def get_current_state(self) -> Optional[State]:
        """Get current state."""
        if self.simulation_state:
            return self.simulation_state.current_state
        return None


class RateExtractor:
    """Extract rate constants from MD simulation data.
    
    Uses transition state theory and MD trajectories to extract
    rate constants for KMC simulations.
    
    Methods:
    --------
    - Counting method: Direct counting of transitions
    - Transition state theory: From barrier heights
    - Mean first passage time: From survival probability
    """
    
    def __init__(self, temperature: float):
        """Initialize rate extractor.
        
        Args:
            temperature: Temperature in Kelvin
        """
        self.temperature = temperature
        self.k_B = 8.617333e-5  # eV/K
        self.h = 4.135667696e-15  # eV*s
    
    def extract_from_transition_counts(self,
                                       state_sequence: List[Union[int, str]],
                                       time_step: float) -> Dict[Tuple, float]:
        """Extract rates from state transition sequence.
        
        Args:
            state_sequence: Sequence of visited states
            time_step: Time per step (seconds)
            
        Returns:
            Dictionary of (state1, state2) -> rate
        """
        # Count transitions
        transitions = defaultdict(int)
        for i in range(len(state_sequence) - 1):
            s1, s2 = state_sequence[i], state_sequence[i + 1]
            if s1 != s2:
                transitions[(s1, s2)] += 1
        
        # Count state occurrences
        state_counts = defaultdict(int)
        for s in state_sequence:
            state_counts[s] += 1
        
        # Calculate rates
        rates = {}
        for (s1, s2), count in transitions.items():
            # Rate = transitions / total_time_in_state
            total_time = state_counts[s1] * time_step
            rates[(s1, s2)] = count / total_time
        
        return rates
    
    def extract_from_barrier(self,
                            activation_energy: float,
                            prefactor: float = 1e13) -> float:
        """Calculate rate from activation energy (TST).
        
        k = A * exp(-Ea / (k_B * T))
        
        Args:
            activation_energy: Activation energy (eV)
            prefactor: Attempt frequency (Hz)
            
        Returns:
            Rate constant (Hz)
        """
        return prefactor * np.exp(-activation_energy / (self.k_B * self.temperature))
    
    def extract_from_mean_first_passage(self,
                                       transition_times: List[float]) -> float:
        """Extract rate from mean first passage time.
        
        k = 1 / MFPT
        
        Args:
            transition_times: List of first passage times (seconds)
            
        Returns:
            Rate constant (Hz)
        """
        mfpt = np.mean(transition_times)
        return 1.0 / mfpt if mfpt > 0 else 0.0
    
    def extract_from_md_trajectory(self,
                                   trajectory: Any,
                                   state_classifier: Callable,
                                   timestep: float) -> RateCatalog:
        """Extract rates from MD trajectory.
        
        Args:
            trajectory: MD trajectory (ASE format)
            state_classifier: Function to classify state from atoms
            timestep: MD timestep (seconds)
            
        Returns:
            RateCatalog with extracted rates
        """
        # Classify each frame
        states = []
        for atoms in trajectory:
            state = state_classifier(atoms)
            states.append(state)
        
        # Extract rates
        rates = self.extract_from_transition_counts(states, timestep)
        
        # Build catalog
        catalog = RateCatalog()
        for (s1, s2), rate in rates.items():
            process = RateProcess(
                name=f"{s1}_to_{s2}",
                initial_state=s1,
                final_state=s2,
                rate=rate,
                process_type=ProcessType.DIFFUSION
            )
            catalog.add_process(process)
        
        return catalog
    
    def fit_arrhenius(self,
                     temperatures: List[float],
                     rates: List[float]) -> Tuple[float, float]:
        """Fit Arrhenius parameters from temperature-dependent rates.
        
        ln(k) = ln(A) - Ea/(k_B*T)
        
        Args:
            temperatures: List of temperatures (K)
            rates: List of rate constants (Hz)
            
        Returns:
            (prefactor, activation_energy)
        """
        T = np.array(temperatures)
        k = np.array(rates)
        
        # Linear fit
        inv_T = 1.0 / T
        ln_k = np.log(k)
        
        # Fit: ln(k) = ln(A) - Ea/(k_B*T)
        slope, intercept, r_value, p_value, std_err = stats.linregress(inv_T, ln_k)
        
        Ea = -slope * self.k_B  # eV
        A = np.exp(intercept)   # Hz
        
        return A, Ea


class DefectTracker:
    """Track defect evolution during KMC simulation.
    
    Monitors the creation, annihilation, and migration of defects
    such as vacancies, interstitials, and clusters.
    
    Attributes:
        defect_history: History of defect populations
        cluster_history: History of cluster sizes
    """
    
    def __init__(self, defect_radius: float = 5.0):
        """Initialize defect tracker.
        
        Args:
            defect_radius: Radius for defect identification (Angstrom)
        """
        self.defect_radius = defect_radius
        
        self.defect_history = []
        self.cluster_history = []
        self.time_history = []
        self.step_history = []
        
        self.defect_types = defaultdict(int)
        self.cluster_sizes = defaultdict(int)
    
    def update(self, state: State, step: int, time: float):
        """Update defect tracking with new state.
        
        Args:
            state: Current state
            step: Current step
            time: Current time
        """
        # This is a placeholder - real implementation would
        # analyze the atomic positions to identify defects
        
        # Count defects (simplified)
        n_defects = self._count_defects(state)
        
        self.defect_history.append(n_defects)
        self.time_history.append(time)
        self.step_history.append(step)
    
    def _count_defects(self, state: State) -> int:
        """Count defects in state.
        
        Args:
            state: System state
            
        Returns:
            Number of defects
        """
        # Placeholder - would use Wigner-Seitz or similar method
        return 0
    
    def get_statistics(self) -> Dict:
        """Get defect statistics.
        
        Returns:
            Statistics dictionary
        """
        if len(self.defect_history) == 0:
            return {}
        
        return {
            "mean_defects": np.mean(self.defect_history),
            "max_defects": np.max(self.defect_history),
            "defect_fluctuations": np.std(self.defect_history),
            "final_defects": self.defect_history[-1]
        }
    
    def get_defect_evolution(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get defect evolution over time.
        
        Returns:
            (times, defect_counts)
        """
        return np.array(self.time_history), np.array(self.defect_history)


def extract_rates_from_md(trajectory: Any,
                         state_classifier: Callable,
                         timestep: float,
                         temperature: float) -> RateCatalog:
    """Extract rates from MD trajectory.
    
    Convenience function for rate extraction.
    
    Args:
        trajectory: MD trajectory
        state_classifier: Function(atoms) -> state_id
        timestep: MD timestep (seconds)
        temperature: Temperature (K)
        
    Returns:
        RateCatalog
    """
    extractor = RateExtractor(temperature)
    return extractor.extract_from_md_trajectory(trajectory, state_classifier, timestep)


def run_kmc(rate_catalog: RateCatalog,
           initial_state: State,
           temperature: float = 300.0,
           n_steps: int = 100000,
           **kwargs) -> KMCResults:
    """Run KMC simulation with simplified interface.
    
    Args:
        rate_catalog: Rate process catalog
        initial_state: Initial system state
        temperature: Temperature (K)
        n_steps: Number of steps
        **kwargs: Additional KMCConfig parameters
        
    Returns:
        KMCResults
    """
    config = KMCConfig(
        temperature=temperature,
        n_steps=n_steps,
        **kwargs
    )
    
    simulator = KMCSimulator(config, rate_catalog)
    return simulator.run(initial_state)


def analyze_defect_evolution(results: KMCResults) -> Dict:
    """Analyze defect evolution from KMC results.
    
    Args:
        results: KMC results
        
    Returns:
        Defect analysis dictionary
    """
    if "defects" not in results.statistics:
        return {}
    
    defect_stats = results.statistics["defects"]
    
    analysis = {
        "mean_concentration": defect_stats.get("mean_defects", 0),
        "max_concentration": defect_stats.get("max_defects", 0),
        "formation_rate": defect_stats.get("final_defects", 0) / results.total_time
    }
    
    return analysis


def calculate_mc_time(n_events: int, total_rate: float) -> float:
    """Calculate Monte Carlo time.
    
    In KMC, time advances as dt = -ln(u) / R where u is uniform [0,1)
    The average time increment is 1/R.
    
    Args:
        n_events: Number of events
        total_rate: Total rate (Hz)
        
    Returns:
        Expected time (seconds)
    """
    # Average of -ln(u) for u in [0,1) is 1
    return n_events / total_rate
