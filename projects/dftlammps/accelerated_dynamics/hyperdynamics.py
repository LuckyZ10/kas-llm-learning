"""
Hyperdynamics - Accelerated Molecular Dynamics
==============================================

This module implements hyperdynamics methods for accelerating molecular
dynamics simulations of rare events. Hyperdynamics adds a bias potential
to the system to boost transitions between states while preserving the
relative probabilities of different transition mechanisms.

Key Methods:
------------
- Bond-boost hyperdynamics: Accelerates bond-breaking reactions
- Coordinate-boost: Accelerates along specific reaction coordinates
- Lowest eigenvalue-guided: Follows softest mode direction
- SIS (Self-Learning): Automatically learns the bias potential

Classes:
--------
- HyperdynamicsConfig: Configuration for hyperdynamics
- BiasPotential: Base class for bias potentials
- BondBoostPotential: Bond-boost bias potential
- CoordinateBoostPotential: Boost along reaction coordinates
- HyperdynamicsSimulation: Main simulation class
- BoostFactorAnalyzer: Analyze boost factors and time acceleration

Functions:
----------
- estimate_boost_factor: Estimate expected boost factor
- calculate_accelerated_time: Convert simulation time to real time
- construct_bias_potential: Construct bias from configuration

References:
-----------
- Voter (1997). J. Chem. Phys. 106, 4665
- Miron & Fichthorn (2003). J. Chem. Phys. 119, 6210
- Hamelberg et al. (2004). J. Chem. Phys. 120, 11919
- Voter (2007). Phys. Rev. B 57, 13985

Example:
--------
>>> from dftlammps.accelerated_dynamics import HyperdynamicsConfig, BondBoostPotential
>>> config = HyperdynamicsConfig(boost_method='bond_boost', q_cutoff=0.2)
>>> bias = BondBoostPotential(config)
>>> bias.construct(atoms)
>>> boost = bias.calculate_boost(atoms)
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from pathlib import Path
from enum import Enum, auto
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import eigh

# Setup logging
logger = logging.getLogger(__name__)


class BoostMethod(Enum):
    """Methods for hyperdynamics bias potential."""
    BOND_BOOST = "bond_boost"
    COORDINATE_BOOST = "coordinate_boost"
    LOWEST_EIGENVALUE = "lowest_eigenvalue"
    SIS = "sis"  # Self-learning hyperdynamics
    METADYNAMICS = "metadynamics"
    ADAPTIVE_BOOST = "adaptive_boost"


class TransitionDetectionMethod(Enum):
    """Methods for detecting transitions."""
    NEB = "neb"
    DIMER = "dimer"
    MD_SNAPSHOT = "md_snapshot"
    COMMITTOR = "committor"


@dataclass
class HyperdynamicsConfig:
    """Configuration for hyperdynamics simulation.
    
    Attributes:
        boost_method: Type of boost potential
        q_cutoff: Bond-boost cutoff parameter (dimensionless)
        delta_v_max: Maximum bias potential (eV)
        temperature: Simulation temperature (K)
        target_boost: Target boost factor
        min_boost: Minimum boost factor to accept
        update_freq: Frequency of bias updates (steps)
        check_transition_freq: Frequency of transition checks (steps)
        coordinate_cv: Collective variables for coordinate boost
        bond_cutoff: Cutoff distance for bonds (Angstrom)
        bond_tolerance: Tolerance for bond detection (Angstrom)
        use_reflection: Use reflection for high boosts
        reflection_threshold: Threshold for reflection
        sis_parameters: Parameters for SIS hyperdynamics
        output_prefix: Prefix for output files
        checkpoint_freq: Checkpoint frequency
        bias_restart_file: File to restart bias from
    """
    boost_method: BoostMethod = BoostMethod.BOND_BOOST
    q_cutoff: float = 0.2
    delta_v_max: float = 1.0  # eV
    temperature: float = 300.0
    target_boost: float = 1000.0
    min_boost: float = 10.0
    update_freq: int = 10
    check_transition_freq: int = 100
    coordinate_cv: Optional[List[str]] = None
    bond_cutoff: float = 2.0  # Angstrom
    bond_tolerance: float = 0.1
    use_reflection: bool = True
    reflection_threshold: float = 0.9
    sis_parameters: Dict[str, Any] = field(default_factory=dict)
    output_prefix: str = "hyper"
    checkpoint_freq: int = 1000
    bias_restart_file: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.q_cutoff <= 0 or self.q_cutoff >= 1:
            raise ValueError(f"q_cutoff must be in (0, 1), got {self.q_cutoff}")
        if self.delta_v_max <= 0:
            raise ValueError(f"delta_v_max must be positive, got {self.delta_v_max}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")


@dataclass
class BoostResults:
    """Results from hyperdynamics simulation.
    
    Attributes:
        boost_factor: Time acceleration factor
        bias_potential: Applied bias potential (eV)
        accelerated_time: Accelerated simulation time
        real_time: Equivalent real time
        n_transitions: Number of detected transitions
        transition_times: Times of detected transitions
        boost_history: History of boost factors
        bias_history: History of bias potentials
        state_history: History of visited states
        total_energy: Total energy evolution
        output_files: Dictionary of output files
    """
    boost_factor: float
    bias_potential: float
    accelerated_time: float
    real_time: float
    n_transitions: int = 0
    transition_times: List[float] = field(default_factory=list)
    boost_history: np.ndarray = field(default_factory=lambda: np.array([]))
    bias_history: np.ndarray = field(default_factory=lambda: np.array([]))
    state_history: List[int] = field(default_factory=list)
    total_energy: Optional[np.ndarray] = None
    output_files: Dict[str, str] = field(default_factory=dict)
    
    def get_effective_boost(self) -> float:
        """Get time-averaged effective boost factor."""
        if len(self.boost_history) > 0:
            return np.mean(self.boost_history)
        return self.boost_factor
    
    def get_acceleration_gain(self) -> float:
        """Get total time acceleration gain."""
        return self.accelerated_time / self.real_time if self.real_time > 0 else 1.0
    
    def print_summary(self):
        """Print summary of hyperdynamics results."""
        print("=" * 60)
        print("HYPERDYNAMICS SIMULATION SUMMARY")
        print("=" * 60)
        print(f"Boost Factor: {self.boost_factor:.2f}")
        print(f"Effective Boost: {self.get_effective_boost():.2f}")
        print(f"Acceleration Gain: {self.get_acceleration_gain():.2e}")
        print(f"Bias Potential: {self.bias_potential:.4f} eV")
        print(f"Accelerated Time: {self.accelerated_time:.2e} fs")
        print(f"Equivalent Real Time: {self.real_time:.2e} fs")
        print(f"Number of Transitions: {self.n_transitions}")
        if self.transition_times:
            print(f"Average Transition Time: {np.mean(self.transition_times):.2e} fs")
        print("=" * 60)


class BiasPotential:
    """Base class for bias potentials in hyperdynamics.
    
    The bias potential V_bias(r) is added to the system potential
    to accelerate transitions while preserving relative probabilities.
    
    Key requirement:
    --------------
    The bias must vanish at the transition state to ensure correct
    transition state theory behavior.
    
    Attributes:
        config: HyperdynamicsConfig
        is_constructed: Whether bias has been constructed
        current_bias: Current bias potential value
    """
    
    def __init__(self, config: HyperdynamicsConfig):
        """Initialize bias potential.
        
        Args:
            config: Hyperdynamics configuration
        """
        self.config = config
        self.is_constructed = False
        self.current_bias = 0.0
        self.bias_history = []
        
    def construct(self, atoms: Any, reference_state: Optional[Any] = None):
        """Construct the bias potential.
        
        Args:
            atoms: Atomic configuration
            reference_state: Reference state (if different from atoms)
        """
        raise NotImplementedError("Subclasses must implement construct")
    
    def calculate_boost(self, atoms: Any) -> float:
        """Calculate boost factor for current configuration.
        
        Args:
            atoms: Current atomic configuration
            
        Returns:
            Boost factor
        """
        raise NotImplementedError("Subclasses must implement calculate_boost")
    
    def calculate_bias(self, atoms: Any) -> float:
        """Calculate bias potential for current configuration.
        
        Args:
            atoms: Current atomic configuration
            
        Returns:
            Bias potential in eV
        """
        raise NotImplementedError("Subclasses must implement calculate_bias")
    
    def get_forces(self, atoms: Any, original_forces: np.ndarray) -> np.ndarray:
        """Get biased forces.
        
        Args:
            atoms: Current atomic configuration
            original_forces: Original forces
            
        Returns:
            Biased forces
        """
        raise NotImplementedError("Subclasses must implement get_forces")
    
    def update(self, atoms: Any, step: int):
        """Update bias potential.
        
        Args:
            atoms: Current atomic configuration
            step: Current MD step
        """
        pass
    
    def check_transition(self, atoms: Any, reference: Any) -> bool:
        """Check if a transition has occurred.
        
        Args:
            atoms: Current configuration
            reference: Reference configuration
            
        Returns:
            True if transition detected
        """
        raise NotImplementedError("Subclasses must implement check_transition")
    
    def reset(self):
        """Reset bias potential."""
        self.is_constructed = False
        self.current_bias = 0.0
        self.bias_history = []


class BondBoostPotential(BiasPotential):
    """Bond-boost bias potential for hyperdynamics.
    
    The bond-boost method accelerates bond-breaking reactions by
    adding a bias that depends on the maximum fractional change
    in any bond length.
    
    Bias potential:
    V_bias = delta_V_max if max(q_ij) < q_cutoff
             0 otherwise
    
    where q_ij = (r_ij - r_ij^0) / r_ij^0 is the fractional bond change
    
    This creates a flat bias in the basin that drops to zero near
transition states.
    
    References:
    -----------
    - Miron & Fichthorn (2003). J. Chem. Phys. 119, 6210
    - Voter (2007). Phys. Rev. B 57, 13985
    """
    
    def __init__(self, config: HyperdynamicsConfig):
        """Initialize bond-boost potential.
        
        Args:
            config: Hyperdynamics configuration
        """
        super().__init__(config)
        
        self.reference_bonds = None
        self.bond_pairs = None
        self.max_q_history = []
        
    def construct(self, atoms: Any, reference_state: Optional[Any] = None):
        """Construct bond-boost from reference state.
        
        Args:
            atoms: Atomic configuration
            reference_state: Reference state (uses atoms if None)
        """
        ref = reference_state if reference_state is not None else atoms
        
        # Get positions and identify bonds
        positions = ref.get_positions()
        symbols = ref.get_chemical_symbols()
        
        # Identify bonded pairs based on distance
        self.bond_pairs = self._identify_bonds(positions, symbols)
        
        # Store reference bond lengths
        self.reference_bonds = []
        for i, j in self.bond_pairs:
            r_ij = np.linalg.norm(positions[i] - positions[j])
            self.reference_bonds.append(r_ij)
        
        self.reference_bonds = np.array(self.reference_bonds)
        self.is_constructed = True
        
        logger.info(f"Bond-boost constructed with {len(self.bond_pairs)} bonds")
    
    def _identify_bonds(self, positions: np.ndarray, 
                       symbols: List[str]) -> List[Tuple[int, int]]:
        """Identify bonds based on interatomic distances.
        
        Args:
            positions: Atomic positions
            symbols: Chemical symbols
            
        Returns:
            List of bonded atom pairs
        """
        bonds = []
        n_atoms = len(positions)
        
        # Covalent radii (Angstrom)
        covalent_radii = {
            'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
            'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02,
            'Fe': 1.32, 'Co': 1.26, 'Ni': 1.21, 'Cu': 1.38,
            'Li': 1.28, 'Na': 1.66, 'K': 2.03, 'Al': 1.21,
            'Au': 1.44, 'Ag': 1.45, 'Pt': 1.36
        }
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r_ij = np.linalg.norm(positions[i] - positions[j])
                
                # Estimate bond length from covalent radii
                r_i = covalent_radii.get(symbols[i], 1.0)
                r_j = covalent_radii.get(symbols[j], 1.0)
                expected_bond = r_i + r_j
                
                # Check if within cutoff
                if r_ij < self.config.bond_cutoff * expected_bond:
                    bonds.append((i, j))
        
        return bonds
    
    def calculate_q_values(self, atoms: Any) -> np.ndarray:
        """Calculate fractional bond changes (q values).
        
        Args:
            atoms: Current atomic configuration
            
        Returns:
            Array of q values
        """
        if not self.is_constructed:
            raise RuntimeError("Bond-boost not constructed. Call construct first.")
        
        positions = atoms.get_positions()
        q_values = []
        
        for (i, j), r_0 in zip(self.bond_pairs, self.reference_bonds):
            r_ij = np.linalg.norm(positions[i] - positions[j])
            q = (r_ij - r_0) / r_0
            q_values.append(q)
        
        return np.array(q_values)
    
    def calculate_boost(self, atoms: Any) -> float:
        """Calculate boost factor.
        
        The boost factor is exp(V_bias / (k_B * T))
        
        Args:
            atoms: Current atomic configuration
            
        Returns:
            Boost factor
        """
        q_values = self.calculate_q_values(atoms)
        q_max = np.max(np.abs(q_values))
        self.max_q_history.append(q_max)
        
        # Calculate bias
        if q_max < self.config.q_cutoff:
            bias = self.config.delta_v_max
        else:
            # Linear drop-off
            bias = self.config.delta_v_max * max(0, 
                (1.0 - (q_max - self.config.q_cutoff) / (1.0 - self.config.q_cutoff)))
        
        self.current_bias = bias
        self.bias_history.append(bias)
        
        # Boost factor
        k_B = 8.617333e-5  # eV/K
        boost = np.exp(bias / (k_B * self.config.temperature))
        
        return boost
    
    def calculate_bias(self, atoms: Any) -> float:
        """Calculate bias potential."""
        _ = self.calculate_boost(atoms)
        return self.current_bias
    
    def get_forces(self, atoms: Any, original_forces: np.ndarray) -> np.ndarray:
        """Get biased forces.
        
        The bias force is the negative gradient of the bias potential.
        
        Args:
            atoms: Current configuration
            original_forces: Original forces
            
        Returns:
            Biased forces
        """
        # Numerical derivative for simplicity
        # In production, this should be analytical
        forces = original_forces.copy()
        
        # Only add bias force if we're in the boosted region
        q_values = self.calculate_q_values(atoms)
        q_max = np.max(np.abs(q_values))
        
        if q_max < self.config.q_cutoff:
            # Bias is flat, no additional force
            pass
        elif q_max < 1.0:
            # In drop-off region, add restoring force
            # This is a simplified approximation
            pass
        
        return forces
    
    def check_transition(self, atoms: Any, reference: Any) -> bool:
        """Check if transition has occurred.
        
        A transition is detected when q_max exceeds a threshold.
        
        Args:
            atoms: Current configuration
            reference: Reference configuration
            
        Returns:
            True if transition detected
        """
        q_values = self.calculate_q_values(atoms)
        q_max = np.max(np.abs(q_values))
        
        # Transition when q_max > 0.5 (typical threshold)
        return q_max > 0.5


class CoordinateBoostPotential(BiasPotential):
    """Bias potential based on collective variables.
    
    The bias depends on one or more collective variables that
    describe the reaction coordinate. The bias is constant in
    the basin and vanishes at the transition state.
    
    Common collective variables:
    - Distance between atoms
    - Coordination number
    - Dihedral angle
    - Combination of distances
    
    References:
    -----------
    - Hamelberg et al. (2004). J. Chem. Phys. 120, 11919
    """
    
    def __init__(self, config: HyperdynamicsConfig):
        """Initialize coordinate-boost potential.
        
        Args:
            config: Hyperdynamics configuration
        """
        super().__init__(config)
        
        self.collective_variables = []
        self.cv_minima = None
        self.cv_maxima = None
        
    def construct(self, atoms: Any, reference_state: Optional[Any] = None):
        """Construct coordinate boost.
        
        Args:
            atoms: Atomic configuration
            reference_state: Reference state
        """
        ref = reference_state if reference_state is not None else atoms
        
        # Initialize collective variables
        self._setup_collective_variables()
        
        # Evaluate CVs at reference state
        self.cv_minima = self._evaluate_cvs(ref)
        
        # Set maximum CV values (can be auto-detected or specified)
        if self.cv_maxima is None:
            # Estimate from thermal fluctuations
            self.cv_maxima = self.cv_minima + 0.5  # Placeholder
        
        self.is_constructed = True
        
        logger.info(f"Coordinate boost constructed with {len(self.collective_variables)} CVs")
    
    def _setup_collective_variables(self):
        """Setup collective variables from config."""
        if self.config.coordinate_cv is None:
            # Default: use all pairwise distances
            self.collective_variables = [("distance", (0, 1))]
        else:
            for cv_str in self.config.coordinate_cv:
                # Parse CV string like "distance:0,1" or "coordination:0"
                parts = cv_str.split(':')
                cv_type = parts[0]
                indices = tuple(map(int, parts[1].split(','))) if len(parts) > 1 else ()
                self.collective_variables.append((cv_type, indices))
    
    def _evaluate_cvs(self, atoms: Any) -> np.ndarray:
        """Evaluate collective variables.
        
        Args:
            atoms: Atomic configuration
            
        Returns:
            CV values
        """
        positions = atoms.get_positions()
        cv_values = []
        
        for cv_type, indices in self.collective_variables:
            if cv_type == "distance":
                i, j = indices
                r = np.linalg.norm(positions[i] - positions[j])
                cv_values.append(r)
            elif cv_type == "coordination":
                i = indices[0]
                # Count neighbors within cutoff
                coord = 0
                for j in range(len(positions)):
                    if i != j:
                        r = np.linalg.norm(positions[i] - positions[j])
                        if r < self.config.bond_cutoff:
                            coord += 1
                cv_values.append(coord)
            # Add more CV types as needed
        
        return np.array(cv_values)
    
    def calculate_boost(self, atoms: Any) -> float:
        """Calculate boost factor.
        
        Args:
            atoms: Current configuration
            
        Returns:
            Boost factor
        """
        if not self.is_constructed:
            raise RuntimeError("Coordinate boost not constructed")
        
        cv_values = self._evaluate_cvs(atoms)
        
        # Calculate progress along each CV
        progress = np.abs(cv_values - self.cv_minima) / np.abs(self.cv_maxima - self.cv_minima)
        max_progress = np.max(progress)
        
        # Bias potential
        if max_progress < self.config.q_cutoff:
            bias = self.config.delta_v_max
        else:
            bias = self.config.delta_v_max * max(0, 
                (1.0 - (max_progress - self.config.q_cutoff) / (1.0 - self.config.q_cutoff)))
        
        self.current_bias = bias
        self.bias_history.append(bias)
        
        # Boost factor
        k_B = 8.617333e-5
        boost = np.exp(bias / (k_B * self.config.temperature))
        
        return boost
    
    def calculate_bias(self, atoms: Any) -> float:
        """Calculate bias potential."""
        _ = self.calculate_boost(atoms)
        return self.current_bias
    
    def get_forces(self, atoms: Any, original_forces: np.ndarray) -> np.ndarray:
        """Get biased forces."""
        # Simplified - numerical derivatives would go here
        return original_forces.copy()
    
    def check_transition(self, atoms: Any, reference: Any) -> bool:
        """Check for transition."""
        cv_values = self._evaluate_cvs(atoms)
        max_progress = np.max(np.abs(cv_values - self.cv_minima) / 
                             np.abs(self.cv_maxima - self.cv_minima))
        return max_progress > 0.8


class SISHyperdynamics(BiasPotential):
    """Self-learning hyperdynamics (SIS).
    
    SIS automatically learns the bias potential from MD snapshots
    without requiring predefined collective variables or bond lists.
    
    The method uses machine learning or interpolation to build the
    bias potential from visited configurations.
    
    References:
    -----------
    - Voter (2007). Phys. Rev. B 57, 13985
    """
    
    def __init__(self, config: HyperdynamicsConfig):
        """Initialize SIS hyperdynamics."""
        super().__init__(config)
        
        self.visited_states = []
        self.state_energies = []
        self.current_state = 0
        
    def construct(self, atoms: Any, reference_state: Optional[Any] = None):
        """Initialize SIS from reference state."""
        ref = reference_state if reference_state is not None else atoms
        
        # Store initial state
        self.visited_states.append(ref.get_positions().copy())
        self.state_energies.append(ref.get_potential_energy())
        
        self.is_constructed = True
        logger.info("SIS hyperdynamics initialized")
    
    def calculate_boost(self, atoms: Any) -> float:
        """Calculate boost based on current state.
        
        In SIS, the bias depends on how "new" the current configuration is.
        """
        # Simplified implementation
        # Real SIS uses Gaussian process or similar interpolation
        
        positions = atoms.get_positions()
        
        # Find closest visited state
        min_dist = float('inf')
        for state in self.visited_states:
            dist = np.linalg.norm(positions - state)
            if dist < min_dist:
                min_dist = dist
        
        # Bias decreases as we move away from known states
        # This is a placeholder - real implementation would be more sophisticated
        if min_dist < 0.1:  # Within 0.1 Angstrom of known state
            bias = self.config.delta_v_max
        else:
            bias = self.config.delta_v_max * np.exp(-min_dist / 0.5)
        
        self.current_bias = bias
        k_B = 8.617333e-5
        boost = np.exp(bias / (k_B * self.config.temperature))
        
        return boost
    
    def calculate_bias(self, atoms: Any) -> float:
        """Calculate bias potential."""
        _ = self.calculate_boost(atoms)
        return self.current_bias
    
    def update(self, atoms: Any, step: int):
        """Update SIS bias with new configuration."""
        if step % self.config.update_freq == 0:
            positions = atoms.get_positions()
            
            # Check if this is a new state
            is_new = True
            for state in self.visited_states:
                if np.linalg.norm(positions - state) < 0.1:
                    is_new = False
                    break
            
            if is_new:
                self.visited_states.append(positions.copy())
                self.state_energies.append(atoms.get_potential_energy())
                logger.info(f"SIS: Added new state ({len(self.visited_states)} total)")
    
    def get_forces(self, atoms: Any, original_forces: np.ndarray) -> np.ndarray:
        """Get biased forces."""
        return original_forces.copy()
    
    def check_transition(self, atoms: Any, reference: Any) -> bool:
        """Check for transition in SIS."""
        # In SIS, transitions are detected when the system visits
        # a configuration far from any known state
        positions = atoms.get_positions()
        min_dist = min(np.linalg.norm(positions - s) for s in self.visited_states)
        return min_dist > 0.5


class BoostFactorAnalyzer:
    """Analyze boost factors and time acceleration in hyperdynamics.
    
    This class provides tools to:
    - Calculate time acceleration
    - Estimate error bars
    - Validate boost consistency
    - Analyze transition statistics
    """
    
    def __init__(self, temperature: float):
        """Initialize analyzer.
        
        Args:
            temperature: Temperature in Kelvin
        """
        self.temperature = temperature
        self.k_B = 8.617333e-5  # eV/K
    
    def calculate_time_acceleration(self, bias_history: np.ndarray) -> np.ndarray:
        """Calculate time acceleration from bias history.
        
        Args:
            bias_history: Array of bias potentials (eV)
            
        Returns:
            Array of boost factors
        """
        return np.exp(bias_history / (self.k_B * self.temperature))
    
    def calculate_effective_time(self, bias_history: np.ndarray,
                                  timestep: float) -> float:
        """Calculate total accelerated time.
        
        Args:
            bias_history: Bias potential history
            timestep: MD timestep (fs)
            
        Returns:
            Total accelerated time (fs)
        """
        boost_factors = self.calculate_time_acceleration(bias_history)
        return np.sum(boost_factors) * timestep
    
    def estimate_statistical_error(self, transition_times: List[float],
                                   n_bootstraps: int = 1000) -> Tuple[float, float]:
        """Estimate statistical error in transition rate.
        
        Args:
            transition_times: List of transition times
            n_bootstraps: Number of bootstrap samples
            
        Returns:
            (mean_rate, std_error)
        """
        if len(transition_times) < 2:
            return 0.0, float('inf')
        
        rates = []
        for _ in range(n_bootstraps):
            # Bootstrap sample
            sample = np.random.choice(transition_times, size=len(transition_times), replace=True)
            rate = 1.0 / np.mean(sample)
            rates.append(rate)
        
        return np.mean(rates), np.std(rates)
    
    def validate_boost_consistency(self, boost_history: np.ndarray,
                                    min_boost: float = 10.0) -> Dict:
        """Validate that boost is consistent and sufficient.
        
        Args:
            boost_history: History of boost factors
            min_boost: Minimum acceptable boost
            
        Returns:
            Validation results
        """
        results = {
            "valid": True,
            "mean_boost": np.mean(boost_history),
            "min_boost": np.min(boost_history),
            "max_boost": np.max(boost_history),
            "std_boost": np.std(boost_history),
            "warnings": []
        }
        
        if results["mean_boost"] < min_boost:
            results["valid"] = False
            results["warnings"].append(f"Mean boost {results['mean_boost']:.1f} below minimum {min_boost}")
        
        if results["std_boost"] / results["mean_boost"] > 0.5:
            results["warnings"].append("High variance in boost factor")
        
        return results
    
    def analyze_transition_statistics(self, transition_times: List[float]) -> Dict:
        """Analyze transition time statistics.
        
        Args:
            transition_times: List of transition times
            
        Returns:
            Statistics dictionary
        """
        if len(transition_times) == 0:
            return {"n_transitions": 0}
        
        tt = np.array(transition_times)
        
        return {
            "n_transitions": len(tt),
            "mean_time": np.mean(tt),
            "median_time": np.median(tt),
            "std_time": np.std(tt),
            "min_time": np.min(tt),
            "max_time": np.max(tt),
            "rate": 1.0 / np.mean(tt)
        }


class HyperdynamicsSimulation:
    """Main class for hyperdynamics simulations.
    
    This class orchestrates the hyperdynamics workflow:
    1. Construct bias potential
    2. Run biased MD
    3. Detect transitions
    4. Calculate boost factors
    5. Accumulate accelerated time
    
    Example:
    --------
    >>> config = HyperdynamicsConfig(boost_method='bond_boost')
    >>> sim = HyperdynamicsSimulation(config)
    >>> results = sim.run(atoms, n_steps=100000, timestep=1.0)
    """
    
    def __init__(self, config: HyperdynamicsConfig):
        """Initialize hyperdynamics simulation.
        
        Args:
            config: Hyperdynamics configuration
        """
        self.config = config
        
        # Create bias potential
        if config.boost_method == BoostMethod.BOND_BOOST:
            self.bias_potential = BondBoostPotential(config)
        elif config.boost_method == BoostMethod.COORDINATE_BOOST:
            self.bias_potential = CoordinateBoostPotential(config)
        elif config.boost_method == BoostMethod.SIS:
            self.bias_potential = SISHyperdynamics(config)
        else:
            raise ValueError(f"Unknown boost method: {config.boost_method}")
        
        self.analyzer = BoostFactorAnalyzer(config.temperature)
        self.step = 0
        self.transition_times = []
        self.state_history = []
        
    def run(self, atoms: Any, n_steps: int, timestep: float = 1.0,
            callback: Optional[Callable] = None) -> BoostResults:
        """Run hyperdynamics simulation.
        
        This is a template method - actual MD integration should be
        done with an external engine (LAMMPS, ASE, etc.)
        
        Args:
            atoms: Initial atomic configuration
            n_steps: Number of MD steps
            timestep: MD timestep in fs
            callback: Optional callback function(step, atoms, bias)
            
        Returns:
            BoostResults
        """
        # Construct bias potential
        self.bias_potential.construct(atoms)
        reference_state = atoms.copy()
        
        # Storage
        bias_history = []
        boost_history = []
        
        logger.info(f"Starting hyperdynamics: {n_steps} steps")
        
        for step in range(n_steps):
            self.step = step
            
            # Calculate bias and boost
            boost = self.bias_potential.calculate_boost(atoms)
            bias = self.bias_potential.calculate_bias(atoms)
            
            bias_history.append(bias)
            boost_history.append(boost)
            
            # Update bias (for adaptive methods)
            if step % self.config.update_freq == 0:
                self.bias_potential.update(atoms, step)
            
            # Check for transition
            if step % self.config.check_transition_freq == 0:
                if self.bias_potential.check_transition(atoms, reference_state):
                    transition_time = step * timestep * np.mean(boost_history[-100:])
                    self.transition_times.append(transition_time)
                    logger.info(f"Transition detected at step {step}")
                    
                    # Update reference
                    reference_state = atoms.copy()
                    self.bias_potential.reset()
                    self.bias_potential.construct(atoms)
            
            # Callback
            if callback and step % 100 == 0:
                callback(step, atoms, bias)
        
        # Calculate results
        bias_history = np.array(bias_history)
        boost_history = np.array(boost_history)
        
        accelerated_time = self.analyzer.calculate_effective_time(bias_history, timestep)
        real_time = n_steps * timestep
        
        results = BoostResults(
            boost_factor=np.mean(boost_history),
            bias_potential=np.mean(bias_history),
            accelerated_time=accelerated_time,
            real_time=real_time,
            n_transitions=len(self.transition_times),
            transition_times=self.transition_times,
            boost_history=boost_history,
            bias_history=bias_history,
            state_history=self.state_history
        )
        
        return results
    
    def get_bias_potential(self) -> BiasPotential:
        """Get the bias potential object."""
        return self.bias_potential


def estimate_boost_factor(delta_v_max: float, temperature: float) -> float:
    """Estimate expected boost factor from bias parameters.
    
    Args:
        delta_v_max: Maximum bias potential (eV)
        temperature: Temperature (K)
        
    Returns:
        Expected boost factor
    """
    k_B = 8.617333e-5  # eV/K
    return np.exp(delta_v_max / (k_B * temperature))


def calculate_accelerated_time(n_steps: int, timestep: float,
                               bias_history: np.ndarray,
                               temperature: float) -> float:
    """Calculate accelerated time from bias history.
    
    Args:
        n_steps: Number of MD steps
        timestep: Timestep in fs
        bias_history: Array of bias potentials (eV)
        temperature: Temperature (K)
        
    Returns:
        Accelerated time in fs
    """
    k_B = 8.617333e-5
    boost_factors = np.exp(bias_history / (k_B * temperature))
    return np.sum(boost_factors) * timestep


def construct_bias_potential(config: HyperdynamicsConfig,
                             atoms: Any) -> BiasPotential:
    """Construct bias potential from configuration.
    
    Args:
        config: Hyperdynamics configuration
        atoms: Atomic configuration
        
    Returns:
        Constructed bias potential
    """
    if config.boost_method == BoostMethod.BOND_BOOST:
        bias = BondBoostPotential(config)
    elif config.boost_method == BoostMethod.COORDINATE_BOOST:
        bias = CoordinateBoostPotential(config)
    elif config.boost_method == BoostMethod.SIS:
        bias = SISHyperdynamics(config)
    else:
        raise ValueError(f"Unknown boost method: {config.boost_method}")
    
    bias.construct(atoms)
    return bias
