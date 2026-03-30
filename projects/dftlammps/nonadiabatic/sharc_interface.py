"""
SHARC Interface Module for Multi-Reference Non-Adiabatic Dynamics
=================================================================

This module provides an interface to SHARC (Surface Hopping including 
ARbitrary Couplings) for performing non-adiabatic dynamics with 
multi-reference methods (CASSCF, MRCI, etc.) and spin-orbit coupling.

Features:
- CASSCF/MRCI wave function handling
- Spin-orbit coupling calculation
- Surface hopping with spin-flip transitions
- Multiple electronic state evolution
- Trajectory analysis with spin resolution

References:
- Mai, S.; Marquetand, P.; Gonzalez, L. Chem. Sci. 2018, 9, 6819-6827
- Richter, M.; Marquetand, P.; Gonzalez-Vazquez, J. et al. JCTC 2011, 7, 1253
"""

import numpy as np
import os
import re
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging
import warnings
from enum import Enum

# Optional imports
try:
    import scipy
    from scipy import linalg, integrate, interpolate
    from scipy.linalg import expm, logm, sqrtm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Some features may be limited.")

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import ase
    from ase import Atoms
    from ase.io import read, write
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False


logger = logging.getLogger(__name__)


class MultiReferenceMethod(Enum):
    """Available multi-reference electronic structure methods."""
    CASSCF = "casscf"
    MRCI = "mrci"
    NEVPT2 = "nevpt2"
    CASPT2 = "caspt2"
    XMS_CASPT2 = "xms_caspt2"
    DDCI = "ddci"


class SpinOrbitMethod(Enum):
    """Methods for spin-orbit coupling calculation."""
    BREIT_PAULI = "breit_pauli"
    ATOMIC_MEAN_FIELD = "amf"
    FULL_BREIT_PAULI = "full_bp"
    ONE_ELECTRON = "one_electron"
    SPIN_FREE = "spin_free"


@dataclass
class SHARCConfig:
    """Configuration for SHARC simulations."""
    
    # Electronic structure
    method: MultiReferenceMethod = MultiReferenceMethod.CASSCF
    nstates: int = 5  # Number of electronic states
    nstates_mspin: int = None  # Per spin multiplicity (None = auto)
    spin_multiplicities: List[int] = field(default_factory=lambda: [1, 3])  # Singlets, triplets
    
    # Active space
    nelcas: int = 4  # Active electrons
    norbcas: int = 4  # Active orbitals
    
    # Spin-orbit coupling
    soc_method: SpinOrbitMethod = SpinOrbitMethod.BREIT_PAULI
    soc_states: List[int] = field(default_factory=lambda: [0, 1, 2])  # States to include SOC
    include_soc: bool = True
    
    # Surface hopping
    surfhop_model: str = "fssh"  # fssh, gfsm, mssh
    decoherence: str = "edc"  # edc, idc, afssh, none
    ekincorrect: str = "none"  # Velocity rescaling method
    reflect_frustrated: str = "no"  # Reflect frustrated hops
    
    # Laser parameters (for laser-driven dynamics)
    laser: bool = False
    n_pulses: int = 0
    laser_file: str = "laser.dat"
    
    # Dynamics parameters
    dt: float = 0.5  # Time step in fs
    nsteps: int = 1000
    time_max: float = None  # Maximum time (overrides nsteps)
    
    # Output control
    output_steps: int = 1
    printlevel: int = 2
    
    # Quantum chemistry program
    qmcode: str = "molpro"  # molpro, orca, gaussian, bagel
    qmscript: str = "QM.sh"
    
    # Parallelization
    nprocs: int = 1
    mpi_parallel: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        d = asdict(self)
        d['method'] = self.method.value
        d['soc_method'] = self.soc_method.value
        return d
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'SHARCConfig':
        """Create from dictionary."""
        d = d.copy()
        d['method'] = MultiReferenceMethod(d.get('method', 'casscf'))
        d['soc_method'] = SpinOrbitMethod(d.get('soc_method', 'breit_pauli'))
        return cls(**d)
    
    def save(self, filepath: str):
        """Save to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'SHARCConfig':
        """Load from JSON file."""
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))


@dataclass
class SpinState:
    """Represents a spin-adapted electronic state."""
    
    energy: float  # Energy in Hartree
    multiplicity: int  # Spin multiplicity (1=singlet, 2=doublet, 3=triplet, etc.)
    ms: float  # Spin projection (Sz)
    irrep: str  # Irreducible representation
    state_index: int  # State index within multiplicity
    coefficient: np.ndarray = field(default_factory=lambda: np.array([]))  # CI coefficients
    
    @property
    def is_singlet(self) -> bool:
        return self.multiplicity == 1
    
    @property
    def is_triplet(self) -> bool:
        return self.multiplicity == 3
    
    def get_spin_label(self) -> str:
        """Get spin state label (S0, T1, etc.)."""
        labels = {1: 'S', 2: 'D', 3: 'T', 4: 'Q', 5: '5'}
        spin_label = labels.get(self.multiplicity, f'{self.multiplicity}')
        return f"{spin_label}{self.state_index}"


@dataclass
class SpinOrbitState:
    """Represents a spin-orbit coupled electronic state (eigenstate of H+HSO)."""
    
    energy: float  # Energy including SOC in Hartree
    label: str  # State label
    components: List[Tuple[SpinState, complex]]  # (spin state, coefficient) pairs
    dipole_moment: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    def get_multiplicity_weights(self) -> Dict[int, float]:
        """Get weight of each multiplicity in this state."""
        weights = defaultdict(float)
        for spin_state, coeff in self.components:
            weights[spin_state.multiplicity] += abs(coeff)**2
        return dict(weights)
    
    def get_major_component(self) -> Tuple[SpinState, float]:
        """Get dominant spin component."""
        max_coeff = 0
        major = None
        for spin_state, coeff in self.components:
            if abs(coeff) > max_coeff:
                max_coeff = abs(coeff)
                major = spin_state
        return major, max_coeff**2


@dataclass
class SpinOrbitMatrix:
    """Spin-orbit coupling matrix in spin-adapted basis."""
    
    spin_states: List[SpinState]  # Basis states
    so_matrix: np.ndarray  # SOC matrix in cm^-1
    h_elec: np.ndarray  # Electronic Hamiltonian (diagonal)
    
    def get_total_hamiltonian(self) -> np.ndarray:
        """Get total Hamiltonian H = H_el + H_SO."""
        h_total = np.diag(self.h_elec)  # Convert to Hartree
        h_total += self.so_matrix / 219474.63  # Convert cm^-1 to Hartree
        return h_total
    
    def diagonalize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Diagonalize total Hamiltonian to get SOC eigenstates."""
        h_total = self.get_total_hamiltonian()
        eigvals, eigvecs = np.linalg.eigh(h_total)
        return eigvals, eigvecs
    
    def get_soc_eigenstates(self) -> List[SpinOrbitState]:
        """Get list of SOC eigenstates."""
        eigvals, eigvecs = self.diagonalize()
        
        states = []
        for i, (e, vec) in enumerate(zip(eigvals, eigvecs.T)):
            components = []
            for j, spin_state in enumerate(self.spin_states):
                if abs(vec[j]) > 1e-6:
                    components.append((spin_state, vec[j]))
            
            # Generate label
            major, weight = SpinOrbitState(0, "", components).get_major_component()
            label = f"{major.get_spin_label()} ({weight*100:.0f}%)" if major else f"State{i}"
            
            states.append(SpinOrbitState(
                energy=e,
                label=label,
                components=components
            ))
        
        return states
    
    def get_soc_between(self, i: int, j: int) -> float:
        """Get SOC matrix element between two states in cm^-1."""
        return self.so_matrix[i, j]


@dataclass
class SHARCTrajectory:
    """Trajectory data from SHARC simulation."""
    
    times: np.ndarray  # Time in fs
    positions: np.ndarray  # Nuclear positions [ntimes, natoms, 3]
    velocities: np.ndarray  # Nuclear velocities [ntimes, natoms, 3]
    energies: np.ndarray  # Energies [ntimes, nstates]
    active_state: np.ndarray  # Active state index [ntimes]
    amplitudes: np.ndarray  # Electronic amplitudes [ntimes, nstates]
    hopping_events: List[Dict] = field(default_factory=list)
    
    # Spin-specific data
    spin_multiplicities: List[int] = field(default_factory=list)
    soc_energies: Optional[np.ndarray] = None  # SOC-corrected energies
    
    def get_spin_populations(self) -> Dict[int, np.ndarray]:
        """Get population of each spin multiplicity over time."""
        if not self.spin_multiplicities:
            return {}
        
        populations = defaultdict(lambda: np.zeros(len(self.times)))
        
        for t in range(len(self.times)):
            probs = np.abs(self.amplitudes[t])**2
            for i, mult in enumerate(self.spin_multiplicities):
                populations[mult][t] += probs[i]
        
        return dict(populations)
    
    def get_intersystem_crossing_rate(self) -> float:
        """Calculate rate of intersystem crossing events."""
        isc_events = [
            h for h in self.hopping_events
            if h.get('spin_change', False)
        ]
        total_time = self.times[-1] - self.times[0]
        return len(isc_events) / total_time if total_time > 0 else 0.0


class MultiReferenceInterface:
    """
    Interface for multi-reference electronic structure calculations.
    
    Handles CASSCF, MRCI, and other multi-reference methods through
    various quantum chemistry programs (MOLPRO, ORCA, Gaussian, etc.).
    """
    
    def __init__(self, config: SHARCConfig, workdir: str = "."):
        self.config = config
        self.workdir = Path(workdir)
        self.spin_states = []
        self.energies = None
        self.ci_coefficients = None
        
    def generate_molpro_input(self,
                               geometry: Atoms,
                               filename: str = "molpro.inp") -> str:
        """
        Generate MOLPRO input for CASSCF/MRCI calculation.
        
        Parameters
        ----------
        geometry : Atoms
            Molecular geometry
        filename : str
            Output filename
            
        Returns
        -------
        str : Path to input file
        """
        # MOLPRO input template
        inp_lines = [
            "***, SHARC Calculation",
            "memory,500,m",
            "",
            "! Basis set",
            "basis=cc-pVDZ",
            "",
            "! Geometry",
            "geometry={",
        ]
        
        # Add atomic coordinates
        for atom in geometry:
            inp_lines.append(
                f"{atom.symbol:3s} {atom.position[0]:12.6f} "
                f"{atom.position[1]:12.6f} {atom.position[2]:12.6f}"
            )
        
        inp_lines.extend([
            "}",
            "",
            "! CASSCF calculation",
            f"{{casscf",
            f"closed,0",  # Adjust based on system
            f"occ,{self.config.norbcas + 10}",  # Total orbitals
            f"wf,{geometry.get_number_of_electrons()},1,{self.config.spin_multiplicities[0]-1}",
            f"state,{self.config.nstates}",
        ])
        
        # Add states for each multiplicity
        for mult in self.config.spin_multiplicities[1:]:
            inp_lines.append(f"wf,{geometry.get_number_of_electrons()},1,{mult-1}")
            inp_lines.append(f"state,{self.config.nstates_mspin or self.config.nstates}")
        
        inp_lines.extend([
            "}",
            "",
            "! MRCI if requested",
        ])
        
        if self.config.method == MultiReferenceMethod.MRCI:
            inp_lines.extend([
                "{mrci",
                f"state,{self.config.nstates}",
                "}",
            ])
        
        inp_lines.extend([
            "",
            "! Save results",
            "---",
        ])
        
        # Write input
        filepath = self.workdir / filename
        with open(filepath, 'w') as f:
            f.write('\n'.join(inp_lines))
        
        logger.info(f"MOLPRO input written to {filepath}")
        return str(filepath)
    
    def generate_orca_input(self,
                           geometry: Atoms,
                           filename: str = "orca.inp") -> str:
        """Generate ORCA input for CASSCF calculation."""
        
        inp_lines = [
            "! CASSCF DLPNO-NEVPT2 def2-TZVP VeryTightSCF",
            "%maxcore 4000",
            "",
            "%casscf",
            f"  nel {self.config.nelcas}",
            f"  norb {self.config.norbcas}",
            "  mult ",
        ]
        
        # Multiplicities
        inp_lines[-1] += ','.join(map(str, self.config.spin_multiplicities))
        inp_lines.append(f"  nroots ")
        inp_lines[-1] += ','.join([str(self.config.nstates)] * len(self.config.spin_multiplicities))
        inp_lines.append("end")
        
        # Geometry
        inp_lines.extend([
            "",
            "* xyz 0 1",
        ])
        
        for atom in geometry:
            inp_lines.append(
                f"{atom.symbol:3s} {atom.position[0]:12.6f} "
                f"{atom.position[1]:12.6f} {atom.position[2]:12.6f}"
            )
        
        inp_lines.append("*")
        
        filepath = self.workdir / filename
        with open(filepath, 'w') as f:
            f.write('\n'.join(inp_lines))
        
        return str(filepath)
    
    def parse_molpro_output(self, filepath: str) -> Dict:
        """Parse MOLPRO output for energies and wave functions."""
        
        results = {
            'energies': [],
            'ci_coefficients': [],
            'multiplicities': []
        }
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract CASSCF energies
        energy_pattern = r'!MCSCF STATE\s+(\d+)\.\d+\s+Energy\s+([-\d.]+)'
        for match in re.finditer(energy_pattern, content):
            state_num = int(match.group(1))
            energy = float(match.group(2))
            results['energies'].append(energy)
        
        # Extract MRCI energies if present
        mrci_pattern = r'!MRCI STATE\s+(\d+)\.\d+\s+Energy\s+([-\d.]+)'
        if re.search(mrci_pattern, content):
            results['energies'] = []  # Replace with MRCI
            for match in re.finditer(mrci_pattern, content):
                results['energies'].append(float(match.group(2)))
        
        return results
    
    def calculate_spin_orbit_coupling(self,
                                       states: List[SpinState]) -> SpinOrbitMatrix:
        """
        Calculate spin-orbit coupling matrix.
        
        Parameters
        ----------
        states : List[SpinState]
            Spin-adapted electronic states
            
        Returns
        -------
        SpinOrbitMatrix
        """
        nstates = len(states)
        so_matrix = np.zeros((nstates, nstates))
        h_elec = np.array([s.energy for s in states])
        
        # Calculate SOC matrix elements
        # This is a placeholder - actual SOC calculation requires
        # quantum chemistry program interface
        for i, state_i in enumerate(states):
            for j, state_j in enumerate(states):
                if i == j:
                    so_matrix[i, j] = 0.0
                else:
                    # Selection rules for SOC
                    # ΔS = 0, ±1 and ΔMs = 0, ±1
                    dS = abs(state_i.multiplicity - state_j.multiplicity) / 2
                    dMs = abs(state_i.ms - state_j.ms)
                    
                    if dS <= 1 and dMs <= 1:
                        # Non-zero SOC (placeholder value)
                        so_matrix[i, j] = np.random.random() * 100  # cm^-1
                    else:
                        so_matrix[i, j] = 0.0
        
        return SpinOrbitMatrix(
            spin_states=states,
            so_matrix=so_matrix,
            h_elec=h_elec
        )


class SHARCSurfaceHopping:
    """
    SHARC surface hopping dynamics engine.
    
    Handles:
    - Spin-orbit coupled surface hopping
    - Intersystem crossings
    - Laser-driven dynamics
    - Multiplicity changes
    """
    
    def __init__(self, config: SHARCConfig):
        self.config = config
        self.time = 0.0
        self.amplitudes = None
        self.active_state = 0
        self.rng = np.random.default_rng()
        
    def initialize(self,
                   nstates: int,
                   initial_state: int = 0,
                   initial_amplitudes: Optional[np.ndarray] = None):
        """
        Initialize surface hopping simulation.
        
        Parameters
        ----------
        nstates : int
            Number of states
        initial_state : int
            Initial active state
        initial_amplitudes : np.ndarray, optional
            Initial electronic amplitudes
        """
        self.nstates = nstates
        self.active_state = initial_state
        
        if initial_amplitudes is not None:
            self.amplitudes = initial_amplitudes.copy()
        else:
            self.amplitudes = np.zeros(nstates, dtype=complex)
            self.amplitudes[initial_state] = 1.0
        
        self.density_matrix = np.outer(self.amplitudes, self.amplitudes.conj())
        logger.info(f"Initialized SHARC: {nstates} states, initial state {initial_state}")
    
    def propagate_electronic_soc(self,
                                  hamiltonian: np.ndarray,
                                  nac: np.ndarray,
                                  soc: np.ndarray,
                                  dt: float):
        """
        Propagate electronic wave function with SOC.
        
        Parameters
        ----------
        hamiltonian : np.ndarray
            Electronic Hamiltonian [nstates, nstates]
        nac : np.ndarray
            Non-adiabatic coupling [nstates, nstates]
        soc : np.ndarray
            Spin-orbit coupling matrix [nstates, nstates]
        dt : float
            Time step in fs
        """
        hbar = 0.6582119  # eV·fs
        
        # Total Hamiltonian: H = H_el + H_SO - iℏ * NAC
        H_total = hamiltonian + soc
        
        for i in range(self.nstates):
            for j in range(i+1, self.nstates):
                H_total[i, j] += -1j * hbar * nac[i, j]
                H_total[j, i] += -1j * hbar * nac[j, i]
        
        # Time evolution
        if SCIPY_AVAILABLE:
            U = expm(-1j * H_total * dt / hbar)
            self.amplitudes = U @ self.amplitudes
        else:
            # Euler propagation
            self.amplitudes -= 1j * H_total @ self.amplitudes * dt / hbar
        
        # Renormalize
        norm = np.linalg.norm(self.amplitudes)
        if norm > 0:
            self.amplitudes /= norm
        
        self.density_matrix = np.outer(self.amplitudes, self.amplitudes.conj())
    
    def calculate_sharc_hopping_prob(self,
                                      from_state: int,
                                      to_state: int,
                                      nac: float,
                                      soc: float,
                                      dt: float) -> float:
        """
        Calculate SHARC hopping probability including SOC effects.
        
        Parameters
        ----------
        from_state : int
            Current state
        to_state : int
            Target state
        nac : float
            Non-adiabatic coupling
        soc : float
            Spin-orbit coupling
        dt : float
            Time step
            
        Returns
        -------
        float : Hopping probability
        """
        # SHARC hopping probability with SOC correction
        c_from = self.amplitudes[from_state]
        c_to = self.amplitudes[to_state]
        
        if abs(c_from)**2 < 1e-10:
            return 0.0
        
        # Total coupling (NAC + SOC contribution)
        total_coupling = nac + soc / 0.6582119  # Convert SOC to same units as NAC
        
        # Population flux
        g = -2 * np.real(c_to.conj() * c_from * total_coupling) / abs(c_from)**2
        g *= dt
        
        return max(0.0, g)
    
    def attempt_hop_sharc(self,
                         from_state: int,
                         to_state: int,
                         energies: np.ndarray,
                         hop_prob: float) -> Tuple[bool, float]:
        """
        Attempt surface hop in SHARC framework.
        
        Parameters
        ----------
        from_state : int
            Current state
        to_state : int
            Target state
        energies : np.ndarray
            State energies
        hop_prob : float
            Hopping probability
            
        Returns
        -------
        Tuple of (success, rescaling_factor)
        """
        dE = energies[to_state] - energies[from_state]
        
        # Check energetic feasibility
        ke = 1.0  # Placeholder kinetic energy
        
        if dE > ke:
            # Frustrated hop
            if self.config.reflect_frustrated == "yes":
                return False, -1.0  # Reflect
            else:
                return False, 1.0
        
        # Random decision
        if self.rng.random() < hop_prob:
            # Accept hop
            rescaling = np.sqrt(max(0, 1 - dE / ke))
            return True, rescaling
        else:
            return False, 1.0
    
    def apply_sharc_decoherence(self,
                                 energies: np.ndarray,
                                 dt: float):
        """
        Apply SHARC decoherence correction.
        
        Parameters
        ----------
        energies : np.ndarray
            State energies
        dt : float
            Time step
        """
        method = self.config.decoherence
        
        if method == "edc":
            # Energy-based decoherence
            tau = 0.1  # fs^-1
            for j in range(self.nstates):
                if j != self.active_state:
                    dE = abs(energies[j] - energies[self.active_state])
                    rate = tau * dE
                    decay = np.exp(-rate * dt)
                    self.amplitudes[j] *= decay
        
        elif method == "idc":
            # Instantaneous decoherence
            self.amplitudes.fill(0.0)
            self.amplitudes[self.active_state] = 1.0
        
        # Renormalize
        norm = np.linalg.norm(self.amplitudes)
        if norm > 0:
            self.amplitudes /= norm
        
        self.density_matrix = np.outer(self.amplitudes, self.amplitudes.conj())
    
    def run_sharc_trajectory(self,
                             energy_calculator: Callable,
                             nac_calculator: Callable,
                             soc_calculator: Callable,
                             positions: np.ndarray,
                             velocities: np.ndarray,
                             nsteps: int) -> SHARCTrajectory:
        """
        Run full SHARC trajectory.
        
        Parameters
        ----------
        energy_calculator : Callable
            Function to calculate electronic energies
        nac_calculator : Callable
            Function to calculate NAC
        soc_calculator : Callable
            Function to calculate SOC
        positions : np.ndarray
            Initial positions
        velocities : np.ndarray
            Initial velocities
        nsteps : int
            Number of steps
            
        Returns
        -------
        SHARCTrajectory
        """
        dt = self.config.dt
        
        # Storage
        times = np.zeros(nsteps)
        pos_traj = np.zeros((nsteps,) + positions.shape)
        vel_traj = np.zeros((nsteps,) + velocities.shape)
        energies_traj = np.zeros((nsteps, self.nstates))
        active_traj = np.zeros(nsteps, dtype=int)
        amp_traj = np.zeros((nsteps, self.nstates), dtype=complex)
        hops = []
        
        # Initialize
        pos_traj[0] = positions
        vel_traj[0] = velocities
        active_traj[0] = self.active_state
        amp_traj[0] = self.amplitudes
        
        # Get initial energies
        result = energy_calculator(positions)
        energies_traj[0] = result['energies']
        forces = result.get('forces', np.zeros_like(positions))
        
        # Main loop
        for step in range(1, nsteps):
            # Nuclear propagation
            masses = np.ones(positions.shape[0])
            v_half = velocities + 0.5 * forces / masses[:, None] * dt
            positions = positions + v_half * dt
            
            # Get new electronic data
            result = energy_calculator(positions)
            energies = result['energies']
            forces = result.get('forces', np.zeros_like(positions))
            
            # Calculate NAC and SOC
            nac = nac_calculator(positions)
            soc = soc_calculator(positions)
            
            # Electronic propagation
            H = np.diag(energies)
            self.propagate_electronic_soc(H, nac, soc, dt)
            
            # Attempt hops
            for j in range(self.nstates):
                if j != self.active_state:
                    hop_prob = self.calculate_sharc_hopping_prob(
                        self.active_state, j, 
                        nac[self.active_state, j],
                        soc[self.active_state, j],
                        dt
                    )
                    
                    success, rescaling = self.attempt_hop_sharc(
                        self.active_state, j, energies, hop_prob
                    )
                    
                    if success:
                        old_state = self.active_state
                        self.active_state = j
                        velocities *= rescaling
                        
                        hops.append({
                            'time': self.time,
                            'from': old_state,
                            'to': j,
                            'probability': hop_prob,
                            'spin_change': abs(energies[old_state] - energies[j]) > 0.5
                        })
                        break
            
            # Decoherence
            self.apply_sharc_decoherence(energies, dt)
            
            # Finish velocity update
            velocities = v_half + 0.5 * forces / masses[:, None] * dt
            
            # Update time
            self.time += dt
            
            # Store
            times[step] = self.time
            pos_traj[step] = positions
            vel_traj[step] = velocities
            energies_traj[step] = energies
            active_traj[step] = self.active_state
            amp_traj[step] = self.amplitudes
        
        return SHARCTrajectory(
            times=times,
            positions=pos_traj,
            velocities=vel_traj,
            energies=energies_traj,
            active_state=active_traj,
            amplitudes=amp_traj,
            hopping_events=hops
        )


class SHARCInputGenerator:
    """Generator for SHARC input files."""
    
    def __init__(self, config: SHARCConfig):
        self.config = config
    
    def generate_input(self,
                      geometry: Atoms,
                      filename: str = "input") -> str:
        """
        Generate SHARC main input file.
        
        Parameters
        ----------
        geometry : Atoms
            Molecular geometry
        filename : str
            Output filename
            
        Returns
        -------
        str : Path to input file
        """
        lines = [
            "! SHARC Input File",
            "! Generated by dftlammps",
            "",
            "{",
        ]
        
        # Dynamics section
        lines.extend([
            "  dynamics:",
            f"    dt: {self.config.dt}",
            f"    nsteps: {self.config.nsteps}",
            f"    surf: {self.config.surfhop_model}",
            f"    decoherence: {self.config.decoherence}",
            f"    ekincorrect: {self.config.ekincorrect}",
            f"    reflect_frustrated: {self.config.reflect_frustrated}",
        ])
        
        # States
        lines.extend([
            "",
            "  states:",
            f"    n_states: {self.config.nstates}",
            f"    spin_multiplicities: {self.config.spin_multiplicities}",
        ])
        
        # SOC
        if self.config.include_soc:
            lines.extend([
                "",
                "  spin_orbit:",
                f"    method: {self.config.soc_method.value}",
                f"    states: {self.config.soc_states}",
            ])
        
        # Laser
        if self.config.laser:
            lines.extend([
                "",
                "  laser:",
                f"    n_pulses: {self.config.n_pulses}",
                f"    file: {self.config.laser_file}",
            ])
        
        # QM setup
        lines.extend([
            "",
            "  qm:",
            f"    code: {self.config.qmcode}",
            f"    script: {self.config.qmscript}",
        ])
        
        lines.extend([
            "}",
            "",
            "! Geometry",
            f"{len(geometry)}",
            "",
        ])
        
        for atom in geometry:
            lines.append(
                f"{atom.symbol:3s} {atom.position[0]:12.6f} "
                f"{atom.position[1]:12.6f} {atom.position[2]:12.6f}"
            )
        
        filepath = Path(filename)
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        
        return str(filepath)
    
    def generate_qm_script(self, filename: str = "QM.sh") -> str:
        """Generate QM interface script."""
        
        script = f"""#!/bin/bash
# QM Interface script for SHARC
# Generated by dftlammps

# Input/output files
input=$1
output=$2

# Run QM calculation based on selected code
case "{self.config.qmcode}" in
    molpro)
        molpro -n {self.config.nprocs} molpro.inp
        ;;
    orca)
        orca orca.inp > orca.out
        ;;
    gaussian)
        g16 < gaussian.gjf > gaussian.log
        ;;
    *)
        echo "Unknown QM code: {self.config.qmcode}"
        exit 1
        ;;
esac

# Parse output and write to SHARC format
python3 parse_qm_output.py $output
"""
        
        filepath = Path(filename)
        with open(filepath, 'w') as f:
            f.write(script)
        
        os.chmod(filepath, 0o755)
        return str(filepath)
    
    def generate_laser_file(self,
                           pulse_params: List[Dict],
                           filename: str = "laser.dat") -> str:
        """
        Generate laser pulse file for laser-driven dynamics.
        
        Parameters
        ----------
        pulse_params : List[Dict]
            List of pulse parameters with keys:
            - omega: carrier frequency in eV
            - phi: carrier envelope phase
            - dx, dy, dz: polarization direction
            - sigma: pulse width in fs
            - t0: peak time in fs
            - fwhm: full width at half maximum
            - intensity: intensity in W/cm^2
        filename : str
            Output filename
            
        Returns
        -------
        str : Path to laser file
        """
        lines = ["! SHARC Laser File", ""]
        
        for i, pulse in enumerate(pulse_params):
            lines.extend([
                f"! Pulse {i+1}",
                f"{pulse.get('omega', 3.0):.6f}    ! omega (eV)",
                f"{pulse.get('phi', 0.0):.6f}    ! phi",
                f"{pulse.get('dx', 0.0):.6f}    ! dx",
                f"{pulse.get('dy', 0.0):.6f}    ! dy",
                f"{pulse.get('dz', 1.0):.6f}    ! dz",
                f"{pulse.get('sigma', 10.0):.6f}    ! sigma (fs)",
                f"{pulse.get('t0', 50.0):.6f}    ! t0 (fs)",
                f"{pulse.get('fwhm', 23.0):.6f}    ! FWHM (fs)",
                f"{pulse.get('intensity', 1e12):.6e}    ! Intensity (W/cm^2)",
                "",
            ])
        
        filepath = Path(filename)
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        
        return str(filepath)


class SHARCAnalyzer:
    """Analysis tools for SHARC trajectories."""
    
    def __init__(self, trajectory: SHARCTrajectory):
        self.trajectory = trajectory
    
    def analyze_intersystem_crossing(self) -> Dict:
        """
        Analyze intersystem crossing (ISC) dynamics.
        
        Returns
        -------
        Dict with ISC analysis
        """
        # Identify ISC events
        isc_events = [
            h for h in self.trajectory.hopping_events
            if h.get('spin_change', False)
        ]
        
        # Categorize by multiplicity change
        isc_types = defaultdict(int)
        for event in isc_events:
            from_mult = self._get_multiplicity(event['from'])
            to_mult = self._get_multiplicity(event['to'])
            isc_types[f"{from_mult}->{to_mult}"] += 1
        
        # Calculate ISC rates
        total_time = self.trajectory.times[-1]
        isc_rate = len(isc_events) / total_time if total_time > 0 else 0.0
        
        return {
            'total_isc_events': len(isc_events),
            'isc_rate_fs': isc_rate,
            'isc_rate_ns': isc_rate * 1e6,  # Convert to ns^-1
            'isc_by_type': dict(isc_types),
            'isc_yield': len(isc_events) / max(len(self.trajectory.hopping_events), 1)
        }
    
    def _get_multiplicity(self, state_idx: int) -> int:
        """Get spin multiplicity for a state index."""
        if state_idx < len(self.trajectory.spin_multiplicities):
            return self.trajectory.spin_multiplicities[state_idx]
        return 1  # Default to singlet
    
    def analyze_state_populations(self) -> Dict:
        """Analyze state populations over time."""
        
        populations = np.abs(self.trajectory.amplitudes)**2
        
        # Time-averaged populations
        avg_pops = np.mean(populations, axis=0)
        
        # Final populations
        final_pops = populations[-1]
        
        # Equilibration time (time to reach within 10% of final)
        equil_times = []
        for i in range(populations.shape[1]):
            target = final_pops[i]
            for t, pop in enumerate(populations[:, i]):
                if abs(pop - target) < 0.1 * target:
                    equil_times.append(self.trajectory.times[t])
                    break
            else:
                equil_times.append(self.trajectory.times[-1])
        
        return {
            'average_populations': avg_pops,
            'final_populations': final_pops,
            'equilibration_times': np.array(equil_times),
            'max_population_state': np.argmax(final_pops)
        }
    
    def calculate_photophysical_quantities(self,
                                           initial_state: int = 0) -> Dict:
        """
        Calculate photophysical quantities like quantum yields.
        
        Parameters
        ----------
        initial_state : int
            Initial excited state
            
        Returns
        -------
        Dict with photophysical data
        """
        populations = np.abs(self.trajectory.amplitudes)**2
        
        # Fluorescence quantum yield
        # Population that remains in singlet states
        singlet_mask = np.array([m == 1 for m in self.trajectory.spin_multiplicities])
        singlet_pop = np.sum(populations[:, singlet_mask], axis=1)
        
        # Phosphorescence quantum yield
        triplet_mask = np.array([m == 3 for m in self.trajectory.spin_multiplicities])
        triplet_pop = np.sum(populations[:, triplet_mask], axis=1)
        
        # Intersystem crossing efficiency
        isc_efficiency = triplet_pop[-1] / (singlet_pop[-1] + triplet_pop[-1] + 1e-10)
        
        return {
            'singlet_yield': singlet_pop[-1],
            'triplet_yield': triplet_pop[-1],
            'isc_efficiency': isc_efficiency,
            'fluorescence_quantum_yield': singlet_pop[-1],
            'phosphorescence_quantum_yield': triplet_pop[-1] * isc_efficiency
        }
    
    def analyze_coherence(self) -> Dict:
        """Analyze electronic coherence between states."""
        
        # Density matrix evolution
        rho = np.zeros((len(self.trajectory.times), 
                       self.trajectory.amplitudes.shape[1],
                       self.trajectory.amplitudes.shape[1]), dtype=complex)
        
        for t in range(len(self.trajectory.times)):
            amp = self.trajectory.amplitudes[t]
            rho[t] = np.outer(amp, amp.conj())
        
        # Coherence measure: sum of off-diagonal elements
        coherence = np.zeros(len(self.trajectory.times))
        for t in range(len(self.trajectory.times)):
            coherence[t] = np.sum(np.abs(rho[t])) - np.sum(np.abs(np.diag(rho[t])))
        
        # Decoherence time (time for coherence to drop to 1/e)
        if coherence[0] > 0:
            threshold = coherence[0] / np.e
            decoherence_time = None
            for t, coh in enumerate(coherence):
                if coh < threshold:
                    decoherence_time = self.trajectory.times[t]
                    break
        else:
            decoherence_time = 0.0
        
        return {
            'coherence_vs_time': coherence,
            'initial_coherence': coherence[0],
            'final_coherence': coherence[-1],
            'decoherence_time': decoherence_time
        }
    
    def visualize_trajectory(self, output_dir: str = "./sharc_results"):
        """Generate visualizations of SHARC trajectory."""
        
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        times = self.trajectory.times
        
        # Plot 1: State energies
        ax = axes[0, 0]
        for i in range(self.trajectory.energies.shape[1]):
            ax.plot(times, self.trajectory.energies[:, i], label=f'State {i}')
        ax.set_xlabel('Time (fs)')
        ax.set_ylabel('Energy (eV)')
        ax.set_title('State Energies')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Active state
        ax = axes[0, 1]
        ax.plot(times, self.trajectory.active_state, 'k-', lw=0.5)
        ax.set_xlabel('Time (fs)')
        ax.set_ylabel('Active State')
        ax.set_title('Active State Trajectory')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Populations
        ax = axes[1, 0]
        populations = np.abs(self.trajectory.amplitudes)**2
        for i in range(populations.shape[1]):
            ax.plot(times, populations[:, i], label=f'State {i}')
        ax.set_xlabel('Time (fs)')
        ax.set_ylabel('Population')
        ax.set_title('State Populations')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Spin populations if available
        ax = axes[1, 1]
        spin_pops = self.trajectory.get_spin_populations()
        for mult, pop in spin_pops.items():
            label = {1: 'Singlets', 2: 'Doublets', 3: 'Triplets'}.get(mult, f'Mult={mult}')
            ax.plot(times, pop, label=label)
        ax.set_xlabel('Time (fs)')
        ax.set_ylabel('Population')
        ax.set_title('Spin State Populations')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sharc_analysis.png", dpi=150)
        plt.close()
        
        logger.info(f"SHARC visualizations saved to {output_dir}")


def demo_sharc_workflow():
    """Demonstrate SHARC workflow with synthetic data."""
    
    print("=" * 60)
    print("SHARC Surface Hopping Demo")
    print("=" * 60)
    
    # Configuration
    config = SHARCConfig(
        method=MultiReferenceMethod.CASSCF,
        nstates=6,
        spin_multiplicities=[1, 3],  # Singlets and triplets
        nelcas=4,
        norbcas=4,
        include_soc=True,
        dt=0.5,
        nsteps=500,
        surfhop_model="fssh",
        decoherence="edc"
    )
    
    # Create spin states (synthetic)
    spin_states = []
    # Singlets
    for i in range(3):
        spin_states.append(SpinState(
            energy=i * 2.0,  # eV
            multiplicity=1,
            ms=0,
            irrep="A",
            state_index=i,
            coefficient=np.random.randn(10)
        ))
    # Triplets
    for i in range(3):
        for ms in [-1, 0, 1]:
            spin_states.append(SpinState(
                energy=i * 2.0 + 0.5,  # Slightly lower
                multiplicity=3,
                ms=ms,
                irrep="B",
                state_index=i,
                coefficient=np.random.randn(10)
            ))
    
    # Calculate SOC
    mr_interface = MultiReferenceInterface(config)
    soc_matrix = mr_interface.calculate_spin_orbit_coupling(spin_states)
    
    print(f"\nSOC Matrix Shape: {soc_matrix.so_matrix.shape}")
    print(f"SOC Eigenvalues: {soc_matrix.diagonalize()[0][:5]}")
    
    # Get SOC eigenstates
    soc_states = soc_matrix.get_soc_eigenstates()
    print(f"\nNumber of SOC states: {len(soc_states)}")
    for i, state in enumerate(soc_states[:5]):
        major, weight = state.get_major_component()
        print(f"  State {i}: {state.label}, E={state.energy:.3f} eV")
    
    # Run SHARC dynamics
    print("\nRunning SHARC dynamics...")
    sh = SHARCSurfaceHopping(config)
    sh.initialize(len(soc_states), initial_state=1)
    
    # Placeholder calculators
    def energy_calc(pos):
        return {
            'energies': np.linspace(0, 5, len(soc_states)) + np.random.randn(len(soc_states))*0.1,
            'forces': np.random.randn(3, 3) * 0.01
        }
    
    def nac_calc(pos):
        n = len(soc_states)
        nac = np.random.randn(n, n) * 0.01
        nac = (nac - nac.T)  # Anti-symmetric
        return nac
    
    def soc_calc(pos):
        return soc_matrix.get_total_hamiltonian() * 0.01
    
    trajectory = sh.run_sharc_trajectory(
        energy_calc, nac_calc, soc_calc,
        np.random.randn(3, 3),
        np.random.randn(3, 3) * 0.01,
        config.nsteps
    )
    
    # Analyze
    print(f"\nTrajectory completed: {len(trajectory.hopping_events)} hops")
    
    analyzer = SHARCAnalyzer(trajectory)
    
    isc_analysis = analyzer.analyze_intersystem_crossing()
    print(f"\nISC Analysis:")
    print(f"  Total ISC events: {isc_analysis['total_isc_events']}")
    print(f"  ISC rate: {isc_analysis['isc_rate_fs']:.4f} fs^-1")
    
    pop_analysis = analyzer.analyze_state_populations()
    print(f"\nState Populations (final):")
    for i, pop in enumerate(pop_analysis['final_populations'][:5]):
        print(f"  State {i}: {pop:.4f}")
    
    # Visualize
    if MATPLOTLIB_AVAILABLE:
        print("\nGenerating visualizations...")
        analyzer.visualize_trajectory()
        print("Done! Check sharc_results/ directory")
    
    return config, soc_matrix, trajectory, analyzer


if __name__ == "__main__":
    demo_sharc_workflow()
