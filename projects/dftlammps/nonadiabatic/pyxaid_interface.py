"""
PYXAID Interface Module for Non-Adiabatic Molecular Dynamics
==============================================================

This module provides an interface to PYXAID (PYthon eXtended Ab Initio Dynamics)
for performing non-adiabatic molecular dynamics simulations using VASP TD-DFT
wave functions.

Features:
- VASP TD-DFT wave function extraction and processing
- Non-adiabatic coupling vector calculations
- Surface hopping molecular dynamics (SH-FSSH)
- Carrier lifetime calculations
- Excited state population analysis

References:
- Akimov, A. V.; Prezhdo, O. V. J. Chem. Theory Comput. 2013, 9, 4959-4972
- Akimov, A. V.; Prezhdo, O. V. J. Chem. Theory Comput. 2014, 10, 789-804
"""

import numpy as np
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings

# Optional imports with graceful degradation
try:
    import scipy
    from scipy import linalg, integrate, interpolate, sparse
    from scipy.sparse import csr_matrix, csc_matrix
    from scipy.sparse.linalg import eigsh, eigs
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Some features may be limited.")

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyBboxPatch
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import ase
    from ase import Atoms
    from ase.io import read, write
    from ase.units import fs, Bohr, Hartree, eV
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False


# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class PYXAIDConfig:
    """Configuration class for PYXAID simulations."""
    
    # Electronic structure parameters
    nstates: int = 10  # Number of electronic states
    nac_method: str = "nac"  # NAC calculation method: 'nac', 'ct', 'pt2'
    orbital_basis: str = "mo"  # Basis for NAC: 'mo', 'ao', 'cube'
    
    # Surface hopping parameters
    hopping_method: str = "fssh"  # FSSH, GFSH, MSSH, etc.
    decoherence_method: str = "edc"  # EDC, ID-A, AFSSH
    decoherence_rate: float = 0.1  # Decoherence rate in fs^-1
    quantum_subsystem: str = "exciton"  # 'electron', 'hole', 'exciton'
    
    # Dynamics parameters
    dt: float = 0.5  # Time step in fs
    nsteps: int = 1000  # Number of MD steps
    temperature: float = 300.0  # Temperature in K
    
    # Output control
    output_freq: int = 10  # Output frequency
    save_wf: bool = False  # Save wave functions
    save_nac: bool = True  # Save NAC matrices
    
    # Parallelization
    nprocs: int = 1  # Number of processors
    parallel_mode: str = "openmp"  # 'openmp', 'mpi', 'serial'
    
    # VASP specific
    vasp_prec: str = "Normal"  # VASP precision
    vasp_algo: str = "Normal"  # Algorithm
    vasp_nelm: int = 60  # Electronic steps
    vasp_nsw: int = 0  # Ionic steps (0 for single point)
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'nstates': self.nstates,
            'nac_method': self.nac_method,
            'orbital_basis': self.orbital_basis,
            'hopping_method': self.hopping_method,
            'decoherence_method': self.decoherence_method,
            'decoherence_rate': self.decoherence_rate,
            'quantum_subsystem': self.quantum_subsystem,
            'dt': self.dt,
            'nsteps': self.nsteps,
            'temperature': self.temperature,
            'output_freq': self.output_freq,
            'save_wf': self.save_wf,
            'save_nac': self.save_nac,
            'nprocs': self.nprocs,
            'parallel_mode': self.parallel_mode,
            'vasp_prec': self.vasp_prec,
            'vasp_algo': self.vasp_algo,
            'vasp_nelm': self.vasp_nelm,
            'vasp_nsw': self.vasp_nsw,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'PYXAIDConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'PYXAIDConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))


@dataclass
class ElectronicState:
    """Represents an electronic state with energy and wave function."""
    
    energy: float  # Energy in eV
    wavefunction: np.ndarray  # Wave function coefficients
    state_id: int  # State index
    spin: int = 0  # Spin (0 for alpha, 1 for beta)
    symmetry: str = "A"  # Irreducible representation
    
    @property
    def occupation(self) -> float:
        """Get state occupation (placeholder for TD-DFT)."""
        return 0.0
    
    def get_transition_dipole(self, other: 'ElectronicState') -> np.ndarray:
        """Calculate transition dipole moment between states."""
        # Simplified calculation
        return np.zeros(3)


@dataclass  
class NACMatrix:
    """Non-adiabatic coupling matrix between electronic states."""
    
    states_i: List[int]  # Initial state indices
    states_j: List[int]  # Final state indices
    times: np.ndarray  # Time array in fs
    nac_data: np.ndarray  # NAC values [ntimes, n_pairs]
    velocities: Optional[np.ndarray] = None  # Nuclear velocities
    
    def get_coupling(self, t: float, i: int, j: int) -> float:
        """Get NAC between states i and j at time t."""
        t_idx = np.argmin(np.abs(self.times - t))
        pair_idx = self._get_pair_index(i, j)
        return self.nac_data[t_idx, pair_idx]
    
    def _get_pair_index(self, i: int, j: int) -> int:
        """Get index in NAC data array for state pair (i,j)."""
        for idx, (si, sj) in enumerate(zip(self.states_i, self.states_j)):
            if (si == i and sj == j) or (si == j and sj == i):
                return idx
        raise ValueError(f"State pair ({i}, {j}) not found")
    
    def average_coupling(self, i: int, j: int) -> float:
        """Calculate time-averaged NAC between states."""
        pair_idx = self._get_pair_index(i, j)
        return np.mean(np.abs(self.nac_data[:, pair_idx]))
    
    def coupling_distribution(self, i: int, j: int) -> Tuple[float, float]:
        """Get mean and std of NAC distribution."""
        pair_idx = self._get_pair_index(i, j)
        vals = np.abs(self.nac_data[:, pair_idx])
        return np.mean(vals), np.std(vals)


@dataclass
class HoppingEvent:
    """Records a surface hopping event."""
    
    time: float  # Time of hop in fs
    from_state: int  # Initial state
    to_state: int  # Final state
    hop_prob: float  # Hopping probability
    successful: bool  # Whether hop was accepted
    kinetic_energy: float  # Kinetic energy after hop
    rescaling_factor: float  # Velocity rescaling factor
    
    def __str__(self) -> str:
        status = "SUCCESS" if self.successful else "FAILED"
        return f"Hop {self.from_state}->{self.to_state} at {self.time:.2f} fs [{status}]"


@dataclass
class DynamicsTrajectory:
    """Stores trajectory data from surface hopping simulation."""
    
    times: np.ndarray  # Time array in fs
    positions: np.ndarray  # Nuclear positions [ntimes, natoms, 3]
    velocities: np.ndarray  # Nuclear velocities [ntimes, natoms, 3]
    energies: np.ndarray  # Total energies [ntimes]
    state_energies: np.ndarray  # State energies [ntimes, nstates]
    active_state: np.ndarray  # Active state index [ntimes]
    populations: np.ndarray  # Electronic populations [ntimes, nstates]
    hopping_events: List[HoppingEvent] = field(default_factory=list)
    
    def get_state_lifetime(self, state: int, threshold: float = 0.5) -> float:
        """Calculate average lifetime of a given state."""
        in_state = self.populations[:, state] > threshold
        if not np.any(in_state):
            return 0.0
        
        # Find contiguous segments
        segments = []
        start = None
        for i, val in enumerate(in_state):
            if val and start is None:
                start = i
            elif not val and start is not None:
                segments.append((start, i))
                start = None
        if start is not None:
            segments.append((start, len(in_state)))
        
        # Calculate average duration
        lifetimes = []
        for s, e in segments:
            lifetimes.append(self.times[e-1] - self.times[s])
        
        return np.mean(lifetimes) if lifetimes else 0.0
    
    def get_transition_rate(self, i: int, j: int) -> float:
        """Calculate transition rate between states i and j."""
        hops_ij = [h for h in self.hopping_events 
                   if h.from_state == i and h.to_state == j and h.successful]
        if not hops_ij:
            return 0.0
        total_time = self.times[-1] - self.times[0]
        return len(hops_ij) / total_time  # hops per fs


class VASPTDDFTInterface:
    """
    Interface for extracting TD-DFT data from VASP calculations.
    
    Handles WAVECAR, WAVEDER, and vasprun.xml parsing for excited state
    properties and wave functions.
    """
    
    def __init__(self, workdir: str = "."):
        self.workdir = Path(workdir)
        self.eigenvalues = None
        self.occupations = None
        self.kpoints = None
        self.wavefunctions = None
        self.excitation_energies = None
        self.oscillator_strengths = None
        self.transition_dipoles = None
        
    def read_wavcar(self, filepath: Optional[str] = None) -> Dict:
        """
        Read VASP WAVECAR file for wave function information.
        
        Parameters
        ----------
        filepath : str, optional
            Path to WAVECAR file. If None, looks in workdir.
            
        Returns
        -------
        Dict containing wave function data
        """
        if filepath is None:
            filepath = self.workdir / "WAVECAR"
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"WAVECAR not found: {filepath}")
        
        logger.info(f"Reading WAVECAR: {filepath}")
        
        # VASP WAVECAR binary format reader
        # This is a simplified version - full implementation needs detailed format knowledge
        with open(filepath, 'rb') as f:
            # Read header
            header = np.fromfile(f, dtype=np.float64, count=1)
            if header[0] == 45200:  # VASP 5 format
                recl = 8
            elif header[0] == 45210:  # VASP 4 format
                recl = 4
            else:
                raise ValueError(f"Unknown WAVECAR format: {header[0]}")
            
            # Read record length and number of spins, kpoints, bands
            rec1 = np.fromfile(f, dtype=np.float64, count=recl)
            nspin = int(rec1[0])
            nkpts = int(rec1[1])
            nbands = int(rec1[2])
            
            # Store basic info
            wf_data = {
                'nspin': nspin,
                'nkpts': nkpts,
                'nbands': nbands,
                'energies': [],
                'occupations': [],
                'coefficients': []
            }
            
            # Read k-point data (simplified)
            for ispin in range(nspin):
                for ikpt in range(nkpts):
                    # Read k-point coordinates
                    kpt_rec = np.fromfile(f, dtype=np.float64, count=4)
                    wf_data['kpoints'] = kpt_rec[:3]
                    
                    # Read band energies and occupations
                    band_rec = np.fromfile(f, dtype=np.float64, count=nbands*3)
                    energies = band_rec[::3]
                    occupations = band_rec[1::3]
                    wf_data['energies'].append(energies)
                    wf_data['occupations'].append(occupations)
        
        self.wavefunctions = wf_data
        return wf_data
    
    def read_vasprun(self, filepath: Optional[str] = None) -> Dict:
        """
        Parse vasprun.xml for TD-DFT information.
        
        Parameters
        ----------
        filepath : str, optional
            Path to vasprun.xml. If None, looks in workdir.
            
        Returns
        -------
        Dict with TD-DFT data
        """
        if filepath is None:
            filepath = self.workdir / "vasprun.xml"
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"vasprun.xml not found: {filepath}")
        
        logger.info(f"Reading vasprun.xml: {filepath}")
        
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            logger.error("xml.etree.ElementTree not available")
            return {}
        
        tddft_data = {
            'eigenvalues': [],
            'excitation_energies': [],
            'oscillator_strengths': [],
            'transition_dipoles': []
        }
        
        # Parse XML (this is simplified - full VASP XML is complex)
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        # Look for TD-DFT section
        for elem in root.iter():
            if 'excitation' in elem.tag.lower():
                # Extract excitation data
                for child in elem:
                    if 'energy' in child.tag.lower():
                        tddft_data['excitation_energies'].append(
                            float(child.text) * eV if 'ase' in dir() else float(child.text)
                        )
                    elif 'oscillator' in child.tag.lower():
                        tddft_data['oscillator_strengths'].append(float(child.text))
        
        self.excitation_energies = np.array(tddft_data['excitation_energies'])
        self.oscillator_strengths = np.array(tddft_data['oscillator_strengths'])
        
        return tddft_data
    
    def extract_excited_states(self, nstates: int = 10) -> List[ElectronicState]:
        """
        Extract excited state information from VASP output.
        
        Parameters
        ----------
        nstates : int
            Number of excited states to extract
            
        Returns
        -------
        List of ElectronicState objects
        """
        states = []
        
        # Read TD-DFT data if available
        if self.excitation_energies is None:
            try:
                self.read_vasprun()
            except Exception as e:
                logger.warning(f"Could not read TD-DFT data: {e}")
        
        # Create electronic states
        for i in range(min(nstates, len(self.excitation_energies) if self.excitation_energies is not None else 0)):
            state = ElectronicState(
                energy=self.excitation_energies[i] if self.excitation_energies is not None else 0.0,
                wavefunction=np.zeros(1),  # Placeholder
                state_id=i,
                spin=0,
                symmetry="A"
            )
            states.append(state)
        
        return states
    
    def get_ground_state_energy(self) -> float:
        """Get ground state total energy."""
        # This would typically come from OUTCAR or vasprun.xml
        outcar_path = self.workdir / "OUTCAR"
        if outcar_path.exists():
            with open(outcar_path, 'r') as f:
                for line in f:
                    if 'TOTEN' in line:
                        match = re.search(r'TOTEN\s+=\s+([-\d.]+)', line)
                        if match:
                            return float(match.group(1))
        return 0.0


class NonAdiabaticCouplingCalculator:
    """
    Calculator for non-adiabatic coupling vectors and matrices.
    
    Implements various methods for NAC calculation:
    - Finite difference approximation
    - Overlap-based methods (Hammes-Schiffer/Tully)
    - Wave function projection methods
    """
    
    def __init__(self, config: Optional[PYXAIDConfig] = None):
        self.config = config or PYXAIDConfig()
        self.nac_cache = {}
        
    def calculate_overlap_nac(self, 
                              psi_t: np.ndarray, 
                              psi_tdt: np.ndarray,
                              dt: float) -> np.ndarray:
        """
        Calculate NAC using wave function overlap method.
        
        d_jk ≈ <ψ_j(t)|ψ_k(t+dt)> / (2dt) for j ≠ k
        
        Parameters
        ----------
        psi_t : np.ndarray
            Wave functions at time t [nstates, nbasis]
        psi_tdt : np.ndarray
            Wave functions at time t+dt [nstates, nbasis]
        dt : float
            Time step
            
        Returns
        -------
        np.ndarray : NAC matrix [nstates, nstates]
        """
        # Calculate overlap matrix
        overlap = np.dot(psi_t.conj(), psi_tdt.T)
        
        # NAC from overlap
        nstates = psi_t.shape[0]
        nac = np.zeros((nstates, nstates))
        
        for j in range(nstates):
            for k in range(j+1, nstates):
                # Hammes-Schiffer/Tully formula
                nac[j, k] = overlap[j, k] / (2 * dt)
                nac[k, j] = -nac[j, k]  # Anti-symmetric
        
        return nac
    
    def calculate_finite_difference_nac(self,
                                        energies: np.ndarray,
                                        positions: np.ndarray,
                                        velocities: np.ndarray) -> np.ndarray:
        """
        Calculate NAC using finite difference approximation.
        
        d_jk = <ψ_j|d/dt|ψ_k> = <ψ_j|d/dR|ψ_k> · Ṙ
        
        Parameters
        ----------
        energies : np.ndarray
            State energies [ntimes, nstates]
        positions : np.ndarray
            Nuclear positions [ntimes, natoms, 3]
        velocities : np.ndarray
            Nuclear velocities [ntimes, natoms, 3]
            
        Returns
        -------
        np.ndarray : NAC matrix [ntimes, nstates, nstates]
        """
        ntimes, nstates = energies.shape
        nac = np.zeros((ntimes, nstates, nstates))
        
        # Calculate energy gradients
        for t in range(1, ntimes-1):
            for j in range(nstates):
                for k in range(j+1, nstates):
                    # Energy gap
                    dE = energies[t, k] - energies[t, j]
                    
                    # Position derivative of overlap (simplified)
                    # In practice, this requires wave function overlaps
                    dR = positions[t+1] - positions[t-1]
                    
                    # NAC approximation
                    if abs(dE) > 1e-10:
                        # Simplified coupling
                        nac[t, j, k] = np.sum(dR * velocities[t]) / dE
                        nac[t, k, j] = -nac[t, j, k]
        
        return nac
    
    def calculate_ct_nac(self,
                         diabatic_states: List[np.ndarray],
                         adiabatic_states: List[np.ndarray],
                         h_diabatic: np.ndarray) -> np.ndarray:
        """
        Calculate NAC using charge transfer (CT) method.
        
        Useful for donor-acceptor systems.
        
        Parameters
        ----------
        diabatic_states : List[np.ndarray]
            Diabatic state wave functions
        adiabatic_states : List[np.ndarray]
            Adiabatic state wave functions  
        h_diabatic : np.ndarray
            Diabatic Hamiltonian
            
        Returns
        -------
        np.ndarray : NAC matrix
        """
        nstates = len(adiabatic_states)
        nac = np.zeros((nstates, nstates))
        
        # Transform to adiabatic basis
        # This is a simplified implementation
        for j in range(nstates):
            for k in range(j+1, nstates):
                # Coupling from diabatic states
                coupling = 0.0
                for d1, d2 in zip(diabatic_states, diabatic_states):
                    overlap = np.dot(d1.conj(), d2)
                    coupling += overlap * h_diabatic[0, 1]  # Simplified
                
                nac[j, k] = coupling
                nac[k, j] = -coupling
        
        return nac
    
    def compute_nac_trajectory(self,
                               wavefunctions: List[np.ndarray],
                               energies: np.ndarray,
                               dt: float) -> NACMatrix:
        """
        Compute NAC for entire trajectory.
        
        Parameters
        ----------
        wavefunctions : List[np.ndarray]
            Wave functions at each time step
        energies : np.ndarray
            State energies [ntimes, nstates]
        dt : float
            Time step
            
        Returns
        -------
        NACMatrix object
        """
        ntimes = len(wavefunctions)
        nstates = wavefunctions[0].shape[0]
        
        # State pairs
        states_i = []
        states_j = []
        for i in range(nstates):
            for j in range(i+1, nstates):
                states_i.append(i)
                states_j.append(j)
        
        npairs = len(states_i)
        nac_data = np.zeros((ntimes, npairs))
        times = np.arange(ntimes) * dt
        
        # Calculate NAC at each time step
        for t in range(ntimes - 1):
            nac_mat = self.calculate_overlap_nac(
                wavefunctions[t], wavefunctions[t+1], dt
            )
            
            for p, (i, j) in enumerate(zip(states_i, states_j)):
                nac_data[t, p] = nac_mat[i, j]
        
        return NACMatrix(
            states_i=states_i,
            states_j=states_j,
            times=times,
            nac_data=nac_data
        )


class SurfaceHoppingDynamics:
    """
    Surface hopping molecular dynamics simulator.
    
    Implements various surface hopping algorithms:
    - Fewest Switches Surface Hopping (FSSH)
    - Global Flux Surface Hopping (GFSH)
    - Multi-State Surface Hopping (MSSH)
    
    With decoherence corrections:
    - Energy-based Decoherence Correction (EDC)
    - Instantaneous Decoherence (ID-A)
    - Augmented FSSH (AFSSH)
    """
    
    def __init__(self, config: Optional[PYXAIDConfig] = None):
        self.config = config or PYXAIDConfig()
        self.nac_calculator = NonAdiabaticCouplingCalculator(config)
        self.current_state = 0
        self.time = 0.0
        self.rng = np.random.default_rng()
        
    def initialize_simulation(self, 
                              positions: np.ndarray,
                              velocities: np.ndarray,
                              initial_state: int = 0):
        """
        Initialize surface hopping simulation.
        
        Parameters
        ----------
        positions : np.ndarray
            Initial nuclear positions [natoms, 3] in Angstrom
        velocities : np.ndarray
            Initial velocities [natoms, 3] in Angstrom/fs
        initial_state : int
            Initial electronic state
        """
        self.positions = positions.copy()
        self.velocities = velocities.copy()
        self.current_state = initial_state
        self.time = 0.0
        
        # Initialize electronic amplitudes
        nstates = self.config.nstates
        self.amplitudes = np.zeros(nstates, dtype=complex)
        self.amplitudes[initial_state] = 1.0
        
        # Initialize density matrix
        self.density_matrix = np.outer(self.amplitudes, self.amplitudes.conj())
        
        logger.info(f"Initialized SH simulation: state={initial_state}, "
                   f"nstates={nstates}, dt={self.config.dt} fs")
    
    def propagate_electronic(self, 
                             energies: np.ndarray,
                             nac_matrix: np.ndarray,
                             dt: float):
        """
        Propagate electronic wave function using time-dependent Schrödinger equation.
        
        Parameters
        ----------
        energies : np.ndarray
            State energies [nstates]
        nac_matrix : np.ndarray
            Non-adiabatic coupling matrix [nstates, nstates]
        dt : float
            Time step
        """
        nstates = len(energies)
        
        # Build effective Hamiltonian: H_jk = E_j δ_jk - iℏ d_jk
        H = np.zeros((nstates, nstates), dtype=complex)
        
        # Diagonal: energies
        for j in range(nstates):
            H[j, j] = energies[j]
        
        # Off-diagonal: -iℏ * NAC (ℏ ≈ 0.658 eV·fs)
        hbar = 0.6582119  # eV·fs
        for j in range(nstates):
            for k in range(j+1, nstates):
                H[j, k] = -1j * hbar * nac_matrix[j, k]
                H[k, j] = -1j * hbar * nac_matrix[k, j]
        
        # Time propagation: |ψ(t+dt)> = exp(-iHdt/ℏ)|ψ(t)>
        # Using matrix exponential
        if SCIPY_AVAILABLE:
            from scipy.linalg import expm
            U = expm(-1j * H * dt / hbar)
            self.amplitudes = U @ self.amplitudes
        else:
            # Simple Euler propagation (less accurate)
            self.amplitudes -= 1j * H @ self.amplitudes * dt / hbar
        
        # Renormalize
        norm = np.sqrt(np.sum(np.abs(self.amplitudes)**2))
        if norm > 0:
            self.amplitudes /= norm
        
        # Update density matrix
        self.density_matrix = np.outer(self.amplitudes, self.amplitudes.conj())
    
    def calculate_hopping_probability(self, 
                                      from_state: int,
                                      to_state: int,
                                      nac: float,
                                      dt: float) -> float:
        """
        Calculate surface hopping probability.
        
        Parameters
        ----------
        from_state : int
            Current active state
        to_state : int
            Target state
        nac : float
            Non-adiabatic coupling
        dt : float
            Time step
            
        Returns
        -------
        float : Hopping probability
        """
        # FSSH hopping probability formula
        # g_jk = -2 Re[ (c_k* c_j d_jk) / (c_j* c_j) ] * dt
        
        c_from = self.amplitudes[from_state]
        c_to = self.amplitudes[to_state]
        
        if abs(c_from)**2 < 1e-10:
            return 0.0
        
        # Population transfer term
        pop_term = -2 * np.real(c_to.conj() * c_from * nac) / abs(c_from)**2
        
        # Hopping probability
        g = pop_term * dt
        
        return max(0.0, g)
    
    def attempt_hop(self,
                    from_state: int,
                    to_state: int,
                    energies: np.ndarray,
                    hop_prob: float) -> Tuple[bool, float]:
        """
        Attempt a surface hop with velocity rescaling.
        
        Parameters
        ----------
        from_state : int
            Current state
        to_state : int
            Target state
        energies : np.ndarray
            State energies
        hop_prob : float
            Calculated hopping probability
            
        Returns
        -------
        Tuple of (success, rescaling_factor)
        """
        # Check if hop is energetically allowed
        dE = energies[to_state] - energies[from_state]
        
        # Calculate kinetic energy
        ke = 0.5 * np.sum(self.velocities**2)  # Assuming mass=1 for now
        
        # Simple energy criterion
        if dE > ke:
            # Hop is frustrated - not enough energy
            return False, 1.0
        
        # Generate random number
        zeta = self.rng.random()
        
        if zeta < hop_prob:
            # Hop is accepted - rescale velocities
            if abs(dE) < 1e-10:
                return True, 1.0
            
            # Velocity rescaling to conserve energy
            # v_new = v_old * sqrt(1 - dE/KE)
            rescaling = np.sqrt(1 - dE / ke) if ke > dE else 0.0
            
            return True, rescaling
        else:
            return False, 1.0
    
    def apply_decoherence_correction(self, energies: np.ndarray, dt: float):
        """
        Apply decoherence correction to electronic amplitudes.
        
        Parameters
        ----------
        energies : np.ndarray
            State energies
        dt : float
            Time step
        """
        method = self.config.decoherence_method.lower()
        
        if method == "edc":
            # Energy-based Decoherence Correction
            self._apply_edc(energies, dt)
        elif method == "id-a":
            # Instantaneous Decoherence - Accepting
            self._apply_ida(energies)
        elif method == "afssh":
            # Augmented FSSH
            self._apply_afssh(energies, dt)
    
    def _apply_edc(self, energies: np.ndarray, dt: float):
        """Apply Energy-based Decoherence Correction."""
        tau = self.config.decoherence_rate  # fs^-1
        active = self.current_state
        
        for j in range(len(energies)):
            if j != active:
                # Decoherence rate depends on energy gap
                dE = abs(energies[j] - energies[active])
                rate = tau * dE  # Simplified
                
                # Collapse amplitude
                decay = np.exp(-rate * dt)
                self.amplitudes[j] *= decay
        
        # Renormalize
        norm = np.sqrt(np.sum(np.abs(self.amplitudes)**2))
        if norm > 0:
            self.amplitudes /= norm
    
    def _apply_ida(self, energies: np.ndarray):
        """Apply Instantaneous Decoherence - Accepting."""
        active = self.current_state
        
        # Collapse all amplitudes to active state
        self.amplitudes.fill(0.0)
        self.amplitudes[active] = 1.0
        self.density_matrix = np.outer(self.amplitudes, self.amplitudes.conj())
    
    def _apply_afssh(self, energies: np.ndarray, dt: float):
        """Apply Augmented FSSH decoherence."""
        # Simplified AFSSH implementation
        # In full implementation, this requires tracking auxiliary trajectories
        tau = self.config.decoherence_rate
        
        for j in range(len(energies)):
            if j != self.current_state:
                decay = np.exp(-tau * dt)
                self.amplitudes[j] *= decay
        
        # Renormalize
        norm = np.sqrt(np.sum(np.abs(self.amplitudes)**2))
        if norm > 0:
            self.amplitudes /= norm
    
    def run_trajectory(self,
                       energy_calculator: Callable,
                       nsteps: Optional[int] = None,
                       callback: Optional[Callable] = None) -> DynamicsTrajectory:
        """
        Run surface hopping trajectory.
        
        Parameters
        ----------
        energy_calculator : Callable
            Function to calculate energies and forces
        nsteps : int, optional
            Number of steps (default from config)
        callback : Callable, optional
            Callback function called each step
            
        Returns
        -------
        DynamicsTrajectory object
        """
        if nsteps is None:
            nsteps = self.config.nsteps
        
        dt = self.config.dt
        nstates = self.config.nstates
        
        # Storage arrays
        times = np.zeros(nsteps)
        positions = np.zeros((nsteps,) + self.positions.shape)
        velocities = np.zeros((nsteps,) + self.velocities.shape)
        energies_arr = np.zeros(nsteps)
        state_energies = np.zeros((nsteps, nstates))
        active_states = np.zeros(nsteps, dtype=int)
        populations = np.zeros((nsteps, nstates))
        hopping_events = []
        
        # Initial values
        positions[0] = self.positions
        velocities[0] = self.velocities
        active_states[0] = self.current_state
        populations[0] = np.abs(self.amplitudes)**2
        
        # Get initial energies
        result = energy_calculator(self.positions)
        state_energies[0] = result['energies']
        energies_arr[0] = state_energies[0, self.current_state]
        forces = result.get('forces', np.zeros_like(self.positions))
        
        # Main MD loop
        for step in range(1, nsteps):
            # Verlet integration for nuclei
            # v(t+dt/2) = v(t) + F(t)*dt/(2m)
            # r(t+dt) = r(t) + v(t+dt/2)*dt
            # v(t+dt) = v(t+dt/2) + F(t+dt)*dt/(2m)
            
            masses = np.ones(self.positions.shape[0])  # Placeholder masses
            
            # Half-step velocity
            v_half = self.velocities + 0.5 * forces / masses[:, None] * dt
            
            # Full-step position
            self.positions = self.positions + v_half * dt
            
            # Get new energies and forces
            result = energy_calculator(self.positions)
            new_energies = result['energies']
            forces = result.get('forces', np.zeros_like(self.positions))
            
            # Full-step velocity
            self.velocities = v_half + 0.5 * forces / masses[:, None] * dt
            
            # Calculate NAC (simplified - needs wave functions)
            nac = np.zeros((nstates, nstates))
            
            # Propagate electronic wave function
            self.propagate_electronic(new_energies, nac, dt)
            
            # Attempt surface hops
            for j in range(nstates):
                if j != self.current_state:
                    hop_prob = self.calculate_hopping_probability(
                        self.current_state, j, nac[self.current_state, j], dt
                    )
                    
                    success, rescaling = self.attempt_hop(
                        self.current_state, j, new_energies, hop_prob
                    )
                    
                    if success:
                        # Record hopping event
                        event = HoppingEvent(
                            time=self.time,
                            from_state=self.current_state,
                            to_state=j,
                            hop_prob=hop_prob,
                            successful=True,
                            kinetic_energy=0.5*np.sum(self.velocities**2),
                            rescaling_factor=rescaling
                        )
                        hopping_events.append(event)
                        
                        # Update state and rescale velocities
                        self.current_state = j
                        self.velocities *= rescaling
                        break
            
            # Apply decoherence correction
            self.apply_decoherence_correction(new_energies, dt)
            
            # Update time
            self.time += dt
            
            # Store data
            times[step] = self.time
            positions[step] = self.positions
            velocities[step] = self.velocities
            energies_arr[step] = new_energies[self.current_state]
            state_energies[step] = new_energies
            active_states[step] = self.current_state
            populations[step] = np.abs(self.amplitudes)**2
            
            # Callback
            if callback and step % self.config.output_freq == 0:
                callback(step, self.time, self.current_state, populations[step])
        
        return DynamicsTrajectory(
            times=times,
            positions=positions,
            velocities=velocities,
            energies=energies_arr,
            state_energies=state_energies,
            active_state=active_states,
            populations=populations,
            hopping_events=hopping_events
        )


class CarrierLifetimeAnalyzer:
    """
    Analyzer for carrier lifetimes and recombination dynamics.
    
    Calculates:
    - Electron/hole lifetimes
    - Exciton lifetimes
    - Recombination rates
    - Diffusion coefficients
    """
    
    def __init__(self, trajectory: DynamicsTrajectory):
        self.trajectory = trajectory
        
    def calculate_lifetime(self, 
                          state: int,
                          method: str = "fit") -> Dict:
        """
        Calculate carrier lifetime for a given state.
        
        Parameters
        ----------
        state : int
            State index
        method : str
            Analysis method: 'fit', 'average', 'first_passage'
            
        Returns
        -------
        Dict with lifetime information
        """
        pops = self.trajectory.populations[:, state]
        times = self.trajectory.times
        
        if method == "fit":
            # Fit exponential decay
            # P(t) = P0 * exp(-t/τ)
            from scipy.optimize import curve_fit
            
            def exp_decay(t, tau, P0):
                return P0 * np.exp(-t / tau)
            
            try:
                popt, pcov = curve_fit(exp_decay, times, pops, p0=[100.0, pops[0]])
                tau = popt[0]
                P0 = popt[1]
                tau_err = np.sqrt(pcov[0, 0])
            except:
                tau = np.nan
                P0 = pops[0]
                tau_err = np.nan
            
            return {
                'lifetime': tau,
                'lifetime_error': tau_err,
                'P0': P0,
                'method': 'exponential_fit'
            }
        
        elif method == "average":
            # Average lifetime: τ = ∫ t*P(t)dt / ∫ P(t)dt
            tau = np.trapz(times * pops, times) / np.trapz(pops, times)
            return {
                'lifetime': tau,
                'method': 'average'
            }
        
        elif method == "first_passage":
            # Time when population drops below threshold
            threshold = 0.5 * pops[0]
            below_thresh = np.where(pops < threshold)[0]
            if len(below_thresh) > 0:
                tau = times[below_thresh[0]]
            else:
                tau = times[-1]
            
            return {
                'lifetime': tau,
                'threshold': threshold,
                'method': 'first_passage'
            }
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def calculate_recombination_rate(self,
                                     electron_state: int = 1,
                                     hole_state: int = 0) -> float:
        """
        Calculate electron-hole recombination rate.
        
        Parameters
        ----------
        electron_state : int
            Electron state index
        hole_state : int
            Hole state index
            
        Returns
        -------
        float : Recombination rate in fs^-1
        """
        # Count recombination events (hops from electron to hole state)
        recomb_events = [
            h for h in self.trajectory.hopping_events
            if h.from_state == electron_state and h.to_state == hole_state
            and h.successful
        ]
        
        total_time = self.trajectory.times[-1] - self.trajectory.times[0]
        rate = len(recomb_events) / total_time if total_time > 0 else 0.0
        
        return rate
    
    def calculate_diffusion_coefficient(self,
                                        state: Optional[int] = None) -> float:
        """
        Calculate carrier diffusion coefficient from mean square displacement.
        
        Parameters
        ----------
        state : int, optional
            State to analyze (None for all states)
            
        Returns
        -------
        float : Diffusion coefficient in Å²/fs
        """
        if state is not None:
            # Filter trajectory for specific state
            mask = self.trajectory.active_state == state
            positions = self.trajectory.positions[mask]
            times = self.trajectory.times[mask]
        else:
            positions = self.trajectory.positions
            times = self.trajectory.times
        
        if len(positions) < 2:
            return 0.0
        
        # Calculate center of mass positions
        com = np.mean(positions, axis=1)  # [ntimes, 3]
        
        # Mean square displacement
        msd = np.zeros(len(com))
        for i in range(len(com)):
            dr = com[i] - com[0]
            msd[i] = np.sum(dr**2)
        
        # Diffusion coefficient: D = MSD / (6t) for 3D
        # Fit linear region
        valid = times > 0
        if np.any(valid):
            D = np.mean(msd[valid] / (6 * times[valid]))
        else:
            D = 0.0
        
        return D
    
    def analyze_spectral_diffusion(self,
                                   state: int,
                                   window_size: int = 50) -> Dict:
        """
        Analyze spectral diffusion (energy fluctuations) for a state.
        
        Parameters
        ----------
        state : int
            State index
        window_size : int
            Window size for running average
            
        Returns
        -------
        Dict with spectral diffusion analysis
        """
        energies = self.trajectory.state_energies[:, state]
        times = self.trajectory.times
        
        # Energy fluctuation
        dE = energies - np.mean(energies)
        
        # Autocorrelation function
        acf = self._calculate_acf(dE)
        
        # Frequency domain analysis
        if SCIPY_AVAILABLE:
            from scipy.fft import fft, fftfreq
            
            dt = times[1] - times[0]
            freqs = fftfreq(len(dE), dt)
            spectrum = np.abs(fft(dE))**2
            
            # Find dominant frequency
            pos_mask = freqs > 0
            dominant_idx = np.argmax(spectrum[pos_mask])
            dominant_freq = freqs[pos_mask][dominant_idx]
        else:
            freqs = None
            spectrum = None
            dominant_freq = None
        
        return {
            'energy_mean': np.mean(energies),
            'energy_std': np.std(energies),
            'energy_range': np.max(energies) - np.min(energies),
            'autocorrelation': acf,
            'frequencies': freqs,
            'spectrum': spectrum,
            'dominant_frequency': dominant_freq
        }
    
    def _calculate_acf(self, signal: np.ndarray) -> np.ndarray:
        """Calculate autocorrelation function."""
        n = len(signal)
        signal = signal - np.mean(signal)
        
        # Compute autocorrelation using FFT
        if SCIPY_AVAILABLE:
            f_signal = np.fft.fft(signal, n=2*n)
            acf = np.fft.ifft(f_signal * np.conj(f_signal))[:n].real
            acf = acf / acf[0]  # Normalize
        else:
            # Direct computation
            acf = np.correlate(signal, signal, mode='full')[n-1:]
            acf = acf / acf[0]
        
        return acf


class PYXAIDWorkflow:
    """
    Complete workflow for PYXAID non-adiabatic dynamics simulations.
    
    Orchestrates the entire process from VASP setup to analysis.
    """
    
    def __init__(self, config: Optional[PYXAIDConfig] = None):
        self.config = config or PYXAIDConfig()
        self.vasp_interface = None
        self.dynamics = None
        self.trajectory = None
        
    def setup_vasp_calculation(self,
                                structure_file: str,
                                kpoints: List[int] = [1, 1, 1],
                                nbands: int = None,
                                nstates: int = None):
        """
        Setup VASP calculation for TD-DFT.
        
        Parameters
        ----------
        structure_file : str
            Path to structure file (POSCAR, cif, etc.)
        kpoints : List[int]
            K-point grid
        nbands : int
            Number of bands
        nstates : int
            Number of excited states
        """
        if nstates is None:
            nstates = self.config.nstates
        
        # Generate VASP INCAR for TD-DFT
        incar_content = f"""# TD-DFT Calculation for PYXAID
PREC = {self.config.vasp_prec}
ALGO = {self.config.vasp_algo}
NELM = {self.config.vasp_nelm}
NSW = {self.config.vasp_nsw}
ISMEAR = 0
SIGMA = 0.05
EDIFF = 1E-6
EDIFFG = -0.01

# Electronic structure
NBANDS = {nbands or nstates*2}
LOPTICS = .TRUE.
CSHIFT = 0.1

# TD-DFT
ALGO = Exact
NOMEGA = {nstates}
OMEGAMAX = 20.0

# Output
LWAVE = .TRUE.
LCHARG = .TRUE.
LVTOT = .TRUE.
"""
        
        # Write INCAR
        with open("INCAR", 'w') as f:
            f.write(incar_content)
        
        # Generate KPOINTS
        kpoints_content = f"""Automatic mesh
0
Gamma
{' '.join(map(str, kpoints))}
0 0 0
"""
        with open("KPOINTS", 'w') as f:
            f.write(kpoints_content)
        
        logger.info(f"VASP input files created for {nstates} excited states")
    
    def run_dynamics_simulation(self,
                                 initial_structure,
                                 initial_state: int = 0,
                                 temperature: float = None,
                                 nsteps: int = None) -> DynamicsTrajectory:
        """
        Run complete surface hopping dynamics simulation.
        
        Parameters
        ----------
        initial_structure : Atoms or str
            Initial atomic structure
        initial_state : int
            Initial electronic state
        temperature : float
            Temperature in K
        nsteps : int
            Number of MD steps
            
        Returns
        -------
        DynamicsTrajectory
        """
        if temperature is None:
            temperature = self.config.temperature
        if nsteps is None:
            nsteps = self.config.nsteps
        
        # Initialize structure
        if isinstance(initial_structure, str):
            if ASE_AVAILABLE:
                atoms = read(initial_structure)
            else:
                raise ImportError("ASE required for structure file reading")
        else:
            atoms = initial_structure
        
        # Generate initial velocities (Maxwell-Boltzmann)
        if ASE_AVAILABLE:
            from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
            MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
            positions = atoms.get_positions()
            velocities = atoms.get_velocities() * 0.01  # Convert to Angstrom/fs
        else:
            positions = np.array([[0.0, 0.0, 0.0]])  # Placeholder
            velocities = np.random.randn(*positions.shape) * 0.01
        
        # Initialize dynamics
        self.dynamics = SurfaceHoppingDynamics(self.config)
        self.dynamics.initialize_simulation(positions, velocities, initial_state)
        
        # Define energy calculator (placeholder - needs VASP integration)
        def energy_calculator(pos):
            # This should call VASP or use precomputed data
            nstates = self.config.nstates
            # Placeholder energies
            energies = np.linspace(0, 5, nstates) + np.random.randn(nstates) * 0.1
            forces = np.random.randn(*positions.shape) * 0.01
            return {'energies': energies, 'forces': forces}
        
        # Run trajectory
        logger.info(f"Starting dynamics: {nsteps} steps, state {initial_state}")
        self.trajectory = self.dynamics.run_trajectory(
            energy_calculator,
            nsteps=nsteps,
            callback=lambda s, t, st, pop: print(f"Step {s}: t={t:.1f} fs, state={st}")
                      if s % 100 == 0 else None
        )
        
        logger.info(f"Dynamics completed: {len(self.trajectory.hopping_events)} hops")
        return self.trajectory
    
    def analyze_results(self) -> Dict:
        """
        Analyze dynamics results.
        
        Returns
        -------
        Dict with analysis results
        """
        if self.trajectory is None:
            raise ValueError("No trajectory available. Run dynamics first.")
        
        analyzer = CarrierLifetimeAnalyzer(self.trajectory)
        
        results = {
            'state_lifetimes': {},
            'recombination_rates': {},
            'diffusion_coefficients': {},
            'hopping_statistics': {}
        }
        
        # Calculate lifetimes for each state
        for state in range(self.config.nstates):
            lifetime_data = analyzer.calculate_lifetime(state, method='fit')
            results['state_lifetimes'][f'S{state}'] = lifetime_data
        
        # Hopping statistics
        hop_counts = defaultdict(int)
        for hop in self.trajectory.hopping_events:
            if hop.successful:
                hop_counts[f"{hop.from_state}->{hop.to_state}"] += 1
        
        results['hopping_statistics'] = dict(hop_counts)
        
        return results
    
    def visualize_results(self, output_dir: str = "./pyxaid_results"):
        """
        Generate visualization of dynamics results.
        
        Parameters
        ----------
        output_dir : str
            Directory for output files
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available for visualization")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot state populations
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Population dynamics
        ax = axes[0, 0]
        times = self.trajectory.times
        for i in range(self.config.nstates):
            ax.plot(times, self.trajectory.populations[:, i], label=f'S{i}')
        ax.set_xlabel('Time (fs)')
        ax.set_ylabel('Population')
        ax.set_title('State Populations')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Active state
        ax = axes[0, 1]
        ax.plot(times, self.trajectory.active_state, 'k-', lw=0.5)
        ax.set_xlabel('Time (fs)')
        ax.set_ylabel('Active State')
        ax.set_title('Active State Trajectory')
        ax.grid(True, alpha=0.3)
        
        # Energy evolution
        ax = axes[1, 0]
        ax.plot(times, self.trajectory.energies, 'b-')
        ax.set_xlabel('Time (fs)')
        ax.set_ylabel('Energy (eV)')
        ax.set_title('Total Energy')
        ax.grid(True, alpha=0.3)
        
        # State energies heatmap
        ax = axes[1, 1]
        im = ax.imshow(self.trajectory.state_energies.T, aspect='auto',
                       extent=[times[0], times[-1], 0, self.config.nstates],
                       cmap='viridis', origin='lower')
        ax.set_xlabel('Time (fs)')
        ax.set_ylabel('State')
        ax.set_title('State Energies')
        plt.colorbar(im, ax=ax, label='Energy (eV)')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/dynamics_analysis.png", dpi=150)
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")


# Utility functions

def estimate_hopping_probability(energy_gap: float,
                                  nac: float,
                                  dt: float = 0.5) -> float:
    """
    Estimate surface hopping probability from Landau-Zener formula.
    
    Parameters
    ----------
    energy_gap : float
        Energy gap between states in eV
    nac : float
        Non-adiabatic coupling in eV
    dt : float
        Time step in fs
        
    Returns
    -------
    float : Estimated hopping probability
    """
    # Simplified Landau-Zener estimate
    # P_LZ = exp(-π/2 * ΔE² / (ℏ |d12| |v|))
    hbar = 0.6582119  # eV·fs
    
    # Assume typical velocity
    v = 0.01  # Å/fs
    
    if abs(nac * v) > 1e-10:
        lz_exponent = -np.pi/2 * energy_gap**2 / (hbar * abs(nac) * v)
        return np.exp(lz_exponent)
    else:
        return 0.0


def create_pyxid_input(config: PYXAIDConfig,
                        output_file: str = "pyxaid_input.json"):
    """
    Create PYXAID input file from configuration.
    
    Parameters
    ----------
    config : PYXAIDConfig
        Configuration object
    output_file : str
        Output file path
    """
    config.save(output_file)
    logger.info(f"PYXAID input saved to {output_file}")


# Example usage and demonstration

def demo_pyxaid_workflow():
    """Demonstrate PYXAID workflow with synthetic data."""
    
    print("=" * 60)
    print("PYXAID Non-Adiabatic Dynamics Demo")
    print("=" * 60)
    
    # Create configuration
    config = PYXAIDConfig(
        nstates=5,
        dt=0.5,
        nsteps=500,
        hopping_method="fssh",
        decoherence_method="edc",
        temperature=300.0
    )
    
    # Create workflow
    workflow = PYXAIDWorkflow(config)
    
    # Create synthetic initial structure
    if ASE_AVAILABLE:
        from ase.build import molecule
        atoms = molecule('H2O')
        atoms.center(vacuum=5.0)
    else:
        atoms = None
        print("ASE not available, using placeholder structure")
    
    # Run dynamics (with placeholder energy calculator)
    print(f"\nRunning {config.nsteps} steps of surface hopping dynamics...")
    trajectory = workflow.run_dynamics_simulation(
        atoms or np.array([[0.0, 0.0, 0.0]]),
        initial_state=1,
        temperature=config.temperature,
        nsteps=config.nsteps
    )
    
    # Analyze results
    print("\nAnalyzing results...")
    results = workflow.analyze_results()
    
    # Print summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Total simulation time: {trajectory.times[-1]:.1f} fs")
    print(f"Number of hopping events: {len(trajectory.hopping_events)}")
    print(f"\nState Lifetimes:")
    for state, data in results['state_lifetimes'].items():
        tau = data.get('lifetime', 0)
        if not np.isnan(tau):
            print(f"  {state}: τ = {tau:.1f} fs")
    
    print(f"\nHopping Statistics:")
    for hop_type, count in results['hopping_statistics'].items():
        print(f"  {hop_type}: {count} hops")
    
    # Create visualizations
    if MATPLOTLIB_AVAILABLE:
        print("\nGenerating visualizations...")
        workflow.visualize_results()
        print("Done! Check pyxaid_results/ directory")
    
    return workflow, trajectory, results


if __name__ == "__main__":
    demo_pyxaid_workflow()
