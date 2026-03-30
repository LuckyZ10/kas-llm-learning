"""
Excited State Dynamics Module
=============================

Comprehensive module for simulating excited state processes including:
- Photoexcited carrier relaxation
- Exciton dissociation and recombination
- Energy transfer pathways
- Charge separation dynamics

Integrates PYXAID and SHARC interfaces for complete excited state treatment.

Author: dftlammps development team
"""

import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from enum import Enum
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor

# Local imports
from .pyxaid_interface import (
    PYXAIDConfig, PYXAIDWorkflow, SurfaceHoppingDynamics,
    DynamicsTrajectory, CarrierLifetimeAnalyzer, ElectronicState
)
from .sharc_interface import (
    SHARCConfig, SHARCSurfaceHopping, SHARCTrajectory,
    MultiReferenceMethod, SpinOrbitState
)

# Optional imports
try:
    import scipy
    from scipy import linalg, integrate, interpolate, optimize
    from scipy.sparse import csr_matrix, csc_matrix
    from scipy.sparse.linalg import eigsh
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
    from matplotlib.collections import LineCollection
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import ase
    from ase import Atoms
    from ase.io import read, write
    from ase.build import bulk, molecule
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False


logger = logging.getLogger(__name__)


class ExcitedProcess(Enum):
    """Types of excited state processes."""
    CARRIER_RELAXATION = "carrier_relaxation"
    EXCITON_FORMATION = "exciton_formation"
    EXCITON_DISSOCIATION = "exciton_dissociation"
    CHARGE_SEPARATION = "charge_separation"
    CHARGE_RECOMBINATION = "charge_recombination"
    ENERGY_TRANSFER = "energy_transfer"
    INTERSYSTEM_CROSSING = "intersystem_crossing"


@dataclass
class ExcitonState:
    """
    Represents an exciton state with electron-hole binding.
    
    Attributes:
        binding_energy: Exciton binding energy in eV (positive = bound)
        radius: Exciton Bohr radius in nm
        oscillator_strength: Oscillator strength
        coherence_length: Spatial coherence length in nm
        electron_state: Index of electron state
        hole_state: Index of hole state
    """
    
    energy: float  # Total energy in eV
    binding_energy: float  # Binding energy in eV
    radius: float  # Bohr radius in nm
    oscillator_strength: float
    electron_state: int
    hole_state: int
    coherence_length: float = 1.0  # nm
    
    # Wave function characterization
    electron_wavefunction: Optional[np.ndarray] = None
    hole_wavefunction: Optional[np.ndarray] = None
    
    @property
    def is_bound(self) -> bool:
        """Check if exciton is bound (positive binding energy)."""
        return self.binding_energy > 0
    
    @property
    def is_wannier(self) -> bool:
        """Check if Wannier-Mott exciton (large radius)."""
        return self.radius > 1.0  # nm
    
    @property
    def is_frenkel(self) -> bool:
        """Check if Frenkel exciton (small radius)."""
        return self.radius < 0.5  # nm
    
    def get_dissociation_rate(self, electric_field: float = 0.0) -> float:
        """
        Estimate exciton dissociation rate using Onsager theory.
        
        Parameters
        ----------
        electric_field : float
            Applied electric field in V/nm
            
        Returns
        -------
        float : Dissociation rate in s^-1
        """
        # Simplified Onsager model
        kT = 0.0259  # eV at 300K
        
        # Field-dependent dissociation probability
        if self.is_bound:
            # Poole-Frenkel effect
            barrier_reduction = np.sqrt(electric_field * 1e9) * 0.001  # eV
            effective_binding = max(0, self.binding_energy - barrier_reduction)
            
            # Arrhenius-like rate
            rate = 1e12 * np.exp(-effective_binding / kT)  # s^-1
            return rate
        else:
            return 1e15  # Already unbound, fast separation


@dataclass
class CarrierState:
    """
    Represents a free carrier (electron or hole).
    
    Attributes:
        charge: +1 for hole, -1 for electron
        effective_mass: Effective mass in units of m0
        mobility: Carrier mobility in cm^2/(V·s)
        energy: Energy relative to band edge in eV
        state_index: Quantum state index
    """
    
    charge: int  # +1 or -1
    energy: float  # eV
    effective_mass: float  # m0 units
    mobility: float  # cm^2/(V·s)
    state_index: int
    position: Optional[np.ndarray] = None  # Spatial position
    velocity: Optional[np.ndarray] = None  # Group velocity
    
    @property
    def is_electron(self) -> bool:
        return self.charge == -1
    
    @property
    def is_hole(self) -> bool:
        return self.charge == +1
    
    def get_thermal_velocity(self, T: float = 300.0) -> float:
        """Calculate thermal velocity at temperature T."""
        kB = 8.617e-5  # eV/K
        m0 = 9.109e-31  # kg
        meff = self.effective_mass * m0
        
        # v_th = sqrt(3kT/m)
        v_th = np.sqrt(3 * kB * T / (meff * 1e-3))  # nm/fs
        return v_th
    
    def get_diffusion_coefficient(self, T: float = 300.0) -> float:
        """Calculate diffusion coefficient from Einstein relation."""
        kB = 8.617e-5  # eV/K
        q = 1.0  # elementary charge
        
        # D = μkT/q
        D = self.mobility * kB * T / q  # cm^2/s
        return D * 1e16  # Convert to nm^2/fs


@dataclass
class EnergyTransferPathway:
    """
    Represents an energy transfer pathway between chromophores.
    
    Attributes:
        donor: Donor state index
        acceptor: Acceptor state index
        mechanism: Transfer mechanism (FRET, Dexter, etc.)
        rate: Transfer rate in s^-1
        efficiency: Quantum efficiency
        distance: Donor-acceptor distance in nm
    """
    
    donor: int
    acceptor: int
    mechanism: str  # 'FRET', 'Dexter', 'coulomb', 'exchange'
    rate: float  # s^-1
    efficiency: float
    distance: float  # nm
    
    # FRET specific
    overlap_integral: Optional[float] = None  # M^-1 cm^3
    orientation_factor: Optional[float] = None  # κ²
    
    # Dexter specific
    exchange_integral: Optional[float] = None  # eV
    
    def get_characteristic_time(self) -> float:
        """Get characteristic transfer time."""
        return 1e15 / self.rate if self.rate > 0 else np.inf  # fs


@dataclass
class ChargeSeparationState:
    """
    Represents a charge-separated state (e.g., in donor-acceptor systems).
    
    Attributes:
        electron_position: Position of electron
        hole_position: Position of hole
        separation_distance: Electron-hole separation in nm
        geminate: Whether this is a geminate pair
        free_charge: Whether charges have fully separated
    """
    
    electron_position: np.ndarray
    hole_position: np.ndarray
    electron_energy: float
    hole_energy: float
    geminate: bool = True
    free_charge: bool = False
    
    @property
    def separation_distance(self) -> float:
        """Calculate electron-hole separation distance."""
        return np.linalg.norm(self.electron_position - self.hole_position)
    
    @property
    def coulomb_binding(self) -> float:
        """Calculate Coulomb binding energy."""
        eps0 = 8.854e-12  # F/m
        epsr = 3.0  # Typical organic semiconductor
        e = 1.602e-19  # C
        
        r = self.separation_distance * 1e-9  # Convert to m
        E_binding = -e**2 / (4 * np.pi * eps0 * epsr * r)  # J
        return E_binding * 6.242e+18  # Convert to eV
    
    def get_recombination_probability(self) -> float:
        """Calculate geminate recombination probability."""
        # Onsager theory for recombination
        kT = 0.0259  # eV
        r_c = 1.44 / 3.0  # Coulomb radius in nm (ε=3)
        
        # Probability to escape Coulomb attraction
        if self.separation_distance > 0:
            P_escape = np.exp(-r_c / self.separation_distance)
        else:
            P_escape = 0.0
        
        return 1 - P_escape


class ExcitonDynamics:
    """
    Simulator for exciton dynamics including formation, diffusion, and dissociation.
    """
    
    def __init__(self, 
                 exciton_states: Optional[List[ExcitonState]] = None,
                 temperature: float = 300.0):
        self.exciton_states = exciton_states or []
        self.temperature = temperature
        self.kT = 8.617e-5 * temperature  # eV
        self.time = 0.0
        
    def add_exciton_state(self, state: ExcitonState):
        """Add an exciton state to the system."""
        self.exciton_states.append(state)
    
    def calculate_formation_rate(self,
                                  electron: CarrierState,
                                  hole: CarrierState,
                                  capture_radius: float = 1.0) -> float:
        """
        Calculate exciton formation rate from free carriers.
        
        Parameters
        ----------
        electron : CarrierState
            Electron state
        hole : CarrierState
            Hole state
        capture_radius : float
            Capture radius in nm
            
        Returns
        -------
        float : Formation rate in s^-1
        """
        # Langevin recombination theory
        # k_f = γ * (μ_e + μ_h) / ε
        
        q = 1.602e-19  # C
        eps0 = 8.854e-12  # F/m
        epsr = 3.0
        
        # Sum of mobilities (convert to SI)
        mu_sum = (electron.mobility + hole.mobility) * 1e-4  # m^2/(V·s)
        
        # Langevin prefactor
        gamma = q * mu_sum / (eps0 * epsr)
        
        # Effective concentration (simplified)
        n_eff = 1e24  # m^-3 (placeholder)
        
        rate = gamma * n_eff  # s^-1
        return rate
    
    def calculate_diffusion_coefficient(self, 
                                       exciton: ExcitonState) -> float:
        """
        Calculate exciton diffusion coefficient.
        
        Uses Einstein relation and effective mass approximation.
        """
        # D = μkT/q where μ = qτ/m*
        # For excitons, use reduced mass
        
        me = 0.5  # Reduced effective mass (placeholder)
        tau = 100e-15  # Scattering time (s)
        
        # Mobility
        q = 1.602e-19
        m0 = 9.109e-31
        mu = q * tau / (me * m0)  # m^2/(V·s)
        
        # Diffusion coefficient
        D = mu * self.kT / q  # m^2/s
        return D * 1e14  # Convert to nm^2/fs
    
    def simulate_diffusion(self,
                          initial_exciton: ExcitonState,
                          diffusion_time: float,
                          dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate exciton diffusion using random walk.
        
        Parameters
        ----------
        initial_exciton : ExcitonState
            Initial exciton state
        diffusion_time : float
            Total diffusion time in fs
        dt : float
            Time step in fs
            
        Returns
        -------
        Tuple of (times, positions)
        """
        nsteps = int(diffusion_time / dt)
        D = self.calculate_diffusion_coefficient(initial_exciton)
        
        # Random walk in 3D
        times = np.arange(nsteps) * dt
        positions = np.zeros((nsteps, 3))
        
        # Diffusion step length
        step_length = np.sqrt(6 * D * dt)
        
        current_pos = np.zeros(3)
        for i in range(1, nsteps):
            # Random step
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            current_pos += direction * step_length
            positions[i] = current_pos.copy()
        
        return times, positions
    
    def calculate_dissociation_yield(self,
                                     electric_field: float = 0.0) -> Dict:
        """
        Calculate exciton dissociation yield under electric field.
        
        Parameters
        ----------
        electric_field : float
            Electric field in V/nm
            
        Returns
        -------
        Dict with dissociation analysis
        """
        yields = {}
        
        for exciton in self.exciton_states:
            if not exciton.is_bound:
                yields[exciton] = 1.0
                continue
            
            # Dissociation rate
            k_diss = exciton.get_dissociation_rate(electric_field)
            
            # Competing processes
            k_rad = 1e9  # Radiative decay (s^-1)
            k_nonrad = 1e10  # Non-radiative decay (s^-1)
            
            # Quantum yield
            total_rate = k_diss + k_rad + k_nonrad
            diss_yield = k_diss / total_rate
            
            yields[exciton] = {
                'dissociation_yield': diss_yield,
                'dissociation_rate': k_diss,
                'lifetime': 1 / total_rate
            }
        
        return yields
    
    def analyze_exciton_population(self,
                                   time_points: np.ndarray,
                                   populations: np.ndarray) -> Dict:
        """
        Analyze exciton population dynamics.
        
        Parameters
        ----------
        time_points : np.ndarray
            Time array in fs
        populations : np.ndarray
            Population array [ntimes, nstates]
            
        Returns
        -------
        Dict with population analysis
        """
        results = {
            'state_lifetimes': {},
            'average_coherence': 0.0,
            'dominant_pathways': []
        }
        
        # Calculate lifetimes
        for i, exciton in enumerate(self.exciton_states):
            pop = populations[:, i]
            
            # Exponential fit for lifetime
            if pop[0] > 0:
                log_pop = np.log(pop / pop[0])
                valid = np.isfinite(log_pop) & (log_pop < 0)
                if np.sum(valid) > 2:
                    slope = np.polyfit(time_points[valid], log_pop[valid], 1)[0]
                    if slope < 0:
                        lifetime = -1 / slope  # fs
                        results['state_lifetimes'][i] = lifetime
        
        # Coherence analysis
        if populations.shape[1] > 1:
            # Calculate off-diagonal coherence (simplified)
            coherence = 0.0
            for t in range(len(time_points)):
                for i in range(populations.shape[1]):
                    for j in range(i+1, populations.shape[1]):
                        coherence += np.sqrt(populations[t, i] * populations[t, j])
            results['average_coherence'] = coherence / len(time_points)
        
        return results


class CarrierDynamics:
    """
    Simulator for photoexcited carrier relaxation and transport.
    """
    
    def __init__(self,
                 electron_states: Optional[List[CarrierState]] = None,
                 hole_states: Optional[List[CarrierState]] = None,
                 temperature: float = 300.0):
        self.electron_states = electron_states or []
        self.hole_states = hole_states or []
        self.temperature = temperature
        self.kT = 8.617e-5 * temperature
        
    def add_carrier(self, carrier: CarrierState):
        """Add a carrier state."""
        if carrier.is_electron:
            self.electron_states.append(carrier)
        else:
            self.hole_states.append(carrier)
    
    def calculate_cooling_rate(self,
                               carrier: CarrierState,
                               phonon_energy: float = 0.08) -> float:
        """
        Calculate hot carrier cooling rate via phonon emission.
        
        Parameters
        ----------
        carrier : CarrierState
            Hot carrier state
        phonon_energy : float
            Optical phonon energy in eV
            
        Returns
        -------
        float : Cooling rate in s^-1
        """
        # Simplified hot phonon bottleneck model
        # Rate depends on excess energy and phonon coupling
        
        excess_energy = carrier.energy  # Above band edge
        
        if excess_energy <= 0:
            return 0.0
        
        # Number of phonons needed
        n_phonons = int(excess_energy / phonon_energy)
        
        # Coupling strength (typical for organics: 0.1 eV)
        g = 0.1  # eV
        
        # Cooling rate (simplified)
        gamma0 = 1e13  # Base rate s^-1
        rate = gamma0 * (g / phonon_energy)**2 * n_phonons
        
        return rate
    
    def simulate_hot_carrier_relaxation(self,
                                       initial_carrier: CarrierState,
                                       max_time: float = 1000.0) -> Dict:
        """
        Simulate hot carrier relaxation dynamics.
        
        Parameters
        ----------
        initial_carrier : CarrierState
            Initial hot carrier state
        max_time : float
            Maximum simulation time in fs
            
        Returns
        -------
        Dict with relaxation dynamics
        """
        dt = 1.0  # fs
        nsteps = int(max_time / dt)
        
        times = np.arange(nsteps) * dt
        energies = np.zeros(nsteps)
        temperatures = np.zeros(nsteps)
        
        current_energy = initial_carrier.energy
        
        for i in range(nsteps):
            energies[i] = current_energy
            
            # Calculate carrier temperature from energy
            # E = 3/2 kT for 3D
            temperatures[i] = current_energy / (1.5 * self.kT)
            
            # Cooling
            carrier = CarrierState(
                charge=initial_carrier.charge,
                energy=current_energy,
                effective_mass=initial_carrier.effective_mass,
                mobility=initial_carrier.mobility,
                state_index=initial_carrier.state_index
            )
            
            rate = self.calculate_cooling_rate(carrier)
            cooling = rate * 1e-15 * 0.08  # Energy loss per fs
            
            current_energy = max(0, current_energy - cooling)
            
            if current_energy < 0.01:  # Near band edge
                break
        
        return {
            'times': times[:i+1],
            'energies': energies[:i+1],
            'temperatures': temperatures[:i+1],
            'cooling_time': times[i] if i < nsteps else max_time
        }
    
    def calculate_recombination_rate(self,
                                     electron: CarrierState,
                                     hole: CarrierState,
                                     mechanism: str = "langevin") -> float:
        """
        Calculate electron-hole recombination rate.
        
        Parameters
        ----------
        electron : CarrierState
            Electron state
        hole : CarrierState
            Hole state
        mechanism : str
            Recombination mechanism
            
        Returns
        -------
        float : Recombination rate in s^-1
        """
        if mechanism == "langevin":
            # Langevin recombination
            q = 1.602e-19
            eps0 = 8.854e-12
            epsr = 3.0
            
            mu_sum = (electron.mobility + hole.mobility) * 1e-4
            gamma = q * mu_sum / (eps0 * epsr)
            
            # For given concentrations
            n = p = 1e24  # m^-3
            rate = gamma * min(n, p)
            
        elif mechanism == "trap_assisted":
            # SRH recombination
            # Simplified
            rate = 1e8  # s^-1
            
        elif mechanism == "radiative":
            # Radiative recombination
            B = 1e-16  # cm^3/s (typical for organics)
            n = 1e18  # cm^-3
            rate = B * n
            
        else:
            rate = 1e6  # Default
        
        return rate
    
    def simulate_transport(self,
                          carrier: CarrierState,
                          electric_field: np.ndarray,
                          simulation_time: float,
                          dt: float = 1.0) -> Dict:
        """
        Simulate carrier transport under electric field.
        
        Parameters
        ----------
        carrier : CarrierState
            Carrier to simulate
        electric_field : np.ndarray
            Electric field vector in V/nm
        simulation_time : float
            Simulation time in fs
        dt : float
            Time step in fs
            
        Returns
        -------
        Dict with trajectory data
        """
        nsteps = int(simulation_time / dt)
        
        times = np.arange(nsteps) * dt
        positions = np.zeros((nsteps, 3))
        velocities = np.zeros((nsteps, 3))
        
        # Drift velocity
        # v_d = μE
        v_drift = carrier.mobility * electric_field * 1e-7  # Convert units
        
        # Thermal velocity
        v_thermal = carrier.get_thermal_velocity(self.temperature)
        
        for i in range(1, nsteps):
            # Drift + diffusion
            positions[i] = positions[i-1] + v_drift * dt * 1e-6  # nm
            
            # Add random thermal motion
            thermal_step = np.random.randn(3) * v_thermal * dt * 0.1
            positions[i] += thermal_step
            
            velocities[i] = (positions[i] - positions[i-1]) / dt
        
        return {
            'times': times,
            'positions': positions,
            'velocities': velocities,
            'drift_velocity': np.mean(velocities, axis=0),
            'displacement': positions[-1] - positions[0]
        }


class EnergyTransferNetwork:
    """
    Network model for energy transfer in molecular aggregates.
    """
    
    def __init__(self, n_sites: int = 10):
        self.n_sites = n_sites
        self.sites = []
        self.transfer_rates = np.zeros((n_sites, n_sites))
        self.energies = np.zeros(n_sites)
        self.positions = np.zeros((n_sites, 3))
        
    def set_site_energy(self, site: int, energy: float):
        """Set energy of a site."""
        self.energies[site] = energy
    
    def set_site_position(self, site: int, position: np.ndarray):
        """Set spatial position of a site."""
        self.positions[site] = position
    
    def calculate_fret_rate(self,
                           donor: int,
                           acceptor: int,
                           R0: float = 3.0) -> float:
        """
        Calculate Förster resonance energy transfer rate.
        
        Parameters
        ----------
        donor : int
            Donor site index
        acceptor : int
            Acceptor site index
        R0 : float
            Förster radius in nm
            
        Returns
        -------
        float : FRET rate in s^-1
        """
        distance = np.linalg.norm(self.positions[donor] - self.positions[acceptor])
        
        # FRET rate: k_FRET = (1/τ_D) * (R0/R)^6
        tau_D = 1e-9  # Donor lifetime (s)
        
        rate = (1 / tau_D) * (R0 / distance)**6 if distance > 0 else 0
        return rate
    
    def calculate_dexter_rate(self,
                             donor: int,
                             acceptor: int,
                             J: float = 0.01,
                             decay_length: float = 0.1) -> float:
        """
        Calculate Dexter (electron exchange) energy transfer rate.
        
        Parameters
        ----------
        donor : int
            Donor site index
        acceptor : int
            Acceptor site index
        J : float
            Exchange integral in eV
        decay_length : float
            Decay length in nm
            
        Returns
        -------
        float : Dexter rate in s^-1
        """
        distance = np.linalg.norm(self.positions[donor] - self.positions[acceptor])
        
        # Dexter rate: k_Dexter ∝ J² * exp(-2R/L)
        k0 = 1e12  # s^-1
        rate = k0 * (J / 0.1)**2 * np.exp(-2 * distance / decay_length)
        
        return rate
    
    def build_rate_matrix(self, 
                         mechanism: str = "FRET",
                         **kwargs) -> np.ndarray:
        """
        Build full rate matrix for energy transfer network.
        
        Parameters
        ----------
        mechanism : str
            Transfer mechanism ('FRET', 'Dexter', 'combined')
            
        Returns
        -------
        np.ndarray : Rate matrix [n_sites, n_sites]
        """
        rates = np.zeros((self.n_sites, self.n_sites))
        
        for i in range(self.n_sites):
            for j in range(i+1, self.n_sites):
                if mechanism in ["FRET", "combined"]:
                    k_fret = self.calculate_fret_rate(i, j, **kwargs.get('fret', {}))
                    rates[i, j] += k_fret
                
                if mechanism in ["Dexter", "combined"]:
                    k_dexter = self.calculate_dexter_rate(i, j, **kwargs.get('dexter', {}))
                    rates[i, j] += k_dexter
                
                # Detailed balance for reverse rates
                dE = self.energies[j] - self.energies[i]
                rates[j, i] = rates[i, j] * np.exp(-dE / (8.617e-5 * 300))
        
        # Diagonal: negative sum of outgoing rates
        for i in range(self.n_sites):
            rates[i, i] = -np.sum(rates[i, :])
        
        self.transfer_rates = rates
        return rates
    
    def simulate_energy_transfer(self,
                                 initial_population: np.ndarray,
                                 times: np.ndarray) -> np.ndarray:
        """
        Simulate energy transfer dynamics using master equation.
        
        Parameters
        ----------
        initial_population : np.ndarray
            Initial population on each site
        times : np.ndarray
            Time points for output
            
        Returns
        -------
        np.ndarray : Population evolution [ntimes, n_sites]
        """
        # Master equation: dP/dt = K * P
        # Solution: P(t) = exp(K*t) * P(0)
        
        populations = np.zeros((len(times), self.n_sites))
        populations[0] = initial_population
        
        if SCIPY_AVAILABLE:
            from scipy.integrate import odeint
            
            def master_eq(P, t):
                return self.transfer_rates @ P
            
            populations = odeint(master_eq, initial_population, times)
        else:
            # Simple Euler integration
            for i in range(1, len(times)):
                dt = times[i] - times[i-1]
                populations[i] = populations[i-1] + self.transfer_rates @ populations[i-1] * dt * 1e-15
        
        return populations
    
    def find_transfer_pathways(self,
                               donor: int,
                               acceptor: int,
                               max_hops: int = 5) -> List[List[int]]:
        """
        Find possible energy transfer pathways.
        
        Parameters
        ----------
        donor : int
            Starting site
        acceptor : int
            Target site
        max_hops : int
            Maximum number of hops
            
        Returns
        -------
        List of pathways (each pathway is a list of site indices)
        """
        pathways = []
        
        def dfs(current, target, path, visited):
            if len(path) > max_hops:
                return
            
            if current == target:
                pathways.append(path.copy())
                return
            
            for next_site in range(self.n_sites):
                if next_site not in visited and self.transfer_rates[current, next_site] > 0:
                    visited.add(next_site)
                    path.append(next_site)
                    dfs(next_site, target, path, visited)
                    path.pop()
                    visited.remove(next_site)
        
        visited = {donor}
        dfs(donor, acceptor, [donor], visited)
        
        return pathways


class ExcitedStateDynamicsWorkflow:
    """
    Complete workflow for excited state dynamics simulations.
    
    Integrates multiple methods for comprehensive excited state treatment.
    """
    
    def __init__(self, 
                 pyxaid_config: Optional[PYXAIDConfig] = None,
                 sharc_config: Optional[SHARCConfig] = None):
        self.pyxaid_config = pyxaid_config or PYXAIDConfig()
        self.sharc_config = sharc_config or SHARCConfig()
        
        self.exciton_dynamics = ExcitonDynamics(temperature=300.0)
        self.carrier_dynamics = CarrierDynamics(temperature=300.0)
        self.energy_network = None
        
        self.trajectories = []
        self.analyses = {}
        
    def setup_exciton_system(self,
                            exciton_energies: List[float],
                            binding_energies: List[float],
                            radii: List[float]):
        """Setup exciton states for the system."""
        
        for i, (E, Eb, r) in enumerate(zip(exciton_energies, binding_energies, radii)):
            exciton = ExcitonState(
                energy=E,
                binding_energy=Eb,
                radius=r,
                oscillator_strength=1.0,
                electron_state=i,
                hole_state=i
            )
            self.exciton_dynamics.add_exciton_state(exciton)
        
        logger.info(f"Setup {len(exciton_energies)} exciton states")
    
    def setup_carrier_system(self,
                           electron_masses: List[float],
                           hole_masses: List[float],
                           mobilities: Dict[str, float]):
        """Setup carrier states for the system."""
        
        # Electron states
        for i, me in enumerate(electron_masses):
            carrier = CarrierState(
                charge=-1,
                energy=i * 0.1,
                effective_mass=me,
                mobility=mobilities.get('electron', 1.0),
                state_index=i
            )
            self.carrier_dynamics.add_carrier(carrier)
        
        # Hole states
        for i, mh in enumerate(hole_masses):
            carrier = CarrierState(
                charge=+1,
                energy=i * 0.1,
                effective_mass=mh,
                mobility=mobilities.get('hole', 0.1),
                state_index=i
            )
            self.carrier_dynamics.add_carrier(carrier)
        
        logger.info(f"Setup {len(electron_masses)} electron and {len(hole_masses)} hole states")
    
    def setup_energy_network(self,
                            positions: np.ndarray,
                            site_energies: np.ndarray,
                            connectivity: Optional[np.ndarray] = None):
        """
        Setup energy transfer network.
        
        Parameters
        ----------
        positions : np.ndarray
            Site positions [n_sites, 3]
        site_energies : np.ndarray
            Site energies [n_sites]
        connectivity : np.ndarray, optional
            Connectivity matrix
        """
        n_sites = len(positions)
        self.energy_network = EnergyTransferNetwork(n_sites)
        
        for i in range(n_sites):
            self.energy_network.set_site_position(i, positions[i])
            self.energy_network.set_site_energy(i, site_energies[i])
        
        logger.info(f"Setup energy network with {n_sites} sites")
    
    def run_exciton_dissociation(self,
                                  initial_exciton_idx: int = 0,
                                  electric_field: float = 0.0) -> Dict:
        """
        Run exciton dissociation simulation.
        
        Parameters
        ----------
        initial_exciton_idx : int
            Index of initial exciton state
        electric_field : float
            Applied electric field in V/nm
            
        Returns
        -------
        Dict with dissociation analysis
        """
        if initial_exciton_idx >= len(self.exciton_dynamics.exciton_states):
            raise ValueError("Invalid exciton index")
        
        exciton = self.exciton_dynamics.exciton_states[initial_exciton_idx]
        
        # Calculate dissociation yield
        yields = self.exciton_dynamics.calculate_dissociation_yield(electric_field)
        
        # Simulate diffusion
        times, positions = self.exciton_dynamics.simulate_diffusion(
            exciton, 1000.0
        )
        
        results = {
            'exciton': exciton,
            'dissociation_yield': yields.get(exciton, {}).get('dissociation_yield', 0),
            'diffusion_trajectory': {'times': times, 'positions': positions},
            'diffusion_length': np.std(positions)  # nm
        }
        
        self.analyses['exciton_dissociation'] = results
        return results
    
    def run_carrier_relaxation(self,
                               initial_electron_energy: float = 0.5,
                               initial_hole_energy: float = 0.3) -> Dict:
        """
        Run hot carrier relaxation simulation.
        
        Parameters
        ----------
        initial_electron_energy : float
            Initial electron energy in eV (above CBM)
        initial_hole_energy : float
            Initial hole energy in eV (below VBM)
            
        Returns
        -------
        Dict with relaxation analysis
        """
        # Create hot carriers
        hot_electron = CarrierState(
            charge=-1,
            energy=initial_electron_energy,
            effective_mass=0.5,
            mobility=1.0,
            state_index=0
        )
        
        hot_hole = CarrierState(
            charge=+1,
            energy=initial_hole_energy,
            effective_mass=0.8,
            mobility=0.1,
            state_index=0
        )
        
        # Simulate relaxation
        e_relax = self.carrier_dynamics.simulate_hot_carrier_relaxation(hot_electron)
        h_relax = self.carrier_dynamics.simulate_hot_carrier_relaxation(hot_hole)
        
        results = {
            'electron_relaxation': e_relax,
            'hole_relaxation': h_relax,
            'total_cooling_time': max(e_relax['cooling_time'], h_relax['cooling_time'])
        }
        
        self.analyses['carrier_relaxation'] = results
        return results
    
    def run_energy_transfer(self,
                           donor_site: int = 0,
                           acceptor_site: int = -1,
                           simulation_time: float = 1000.0) -> Dict:
        """
        Run energy transfer simulation.
        
        Parameters
        ----------
        donor_site : int
            Initial excitation site
        acceptor_site : int
            Target site
        simulation_time : float
            Simulation time in fs
            
        Returns
        -------
        Dict with energy transfer analysis
        """
        if self.energy_network is None:
            raise ValueError("Energy network not setup")
        
        if acceptor_site == -1:
            acceptor_site = self.energy_network.n_sites - 1
        
        # Build rate matrix
        rates = self.energy_network.build_rate_matrix(mechanism="FRET")
        
        # Initial population
        P0 = np.zeros(self.energy_network.n_sites)
        P0[donor_site] = 1.0
        
        # Time evolution
        times = np.linspace(0, simulation_time, 1000)
        populations = self.energy_network.simulate_energy_transfer(P0, times)
        
        # Find pathways
        pathways = self.energy_network.find_transfer_pathways(
            donor_site, acceptor_site
        )
        
        results = {
            'times': times,
            'populations': populations,
            'transfer_efficiency': populations[-1, acceptor_site],
            'transfer_time': self._calculate_transfer_time(times, populations[:, acceptor_site]),
            'pathways': pathways
        }
        
        self.analyses['energy_transfer'] = results
        return results
    
    def _calculate_transfer_time(self, times: np.ndarray, population: np.ndarray) -> float:
        """Calculate characteristic transfer time."""
        # Time to reach 50% of final population
        target = 0.5 * population[-1]
        idx = np.where(population >= target)[0]
        return times[idx[0]] if len(idx) > 0 else times[-1]
    
    def run_surface_hopping_dynamics(self,
                                     structure,
                                     initial_state: int = 1,
                                     method: str = "pyxaid") -> Union[DynamicsTrajectory, SHARCTrajectory]:
        """
        Run surface hopping dynamics simulation.
        
        Parameters
        ----------
        structure : Atoms or structure object
            Initial structure
        initial_state : int
            Initial electronic state
        method : str
            Method: 'pyxaid' or 'sharc'
            
        Returns
        -------
        Trajectory object
        """
        if method == "pyxaid":
            workflow = PYXAIDWorkflow(self.pyxaid_config)
            trajectory = workflow.run_dynamics_simulation(
                structure, initial_state=initial_state
            )
        elif method == "sharc":
            sh = SHARCSurfaceHopping(self.sharc_config)
            sh.initialize(self.sharc_config.nstates, initial_state)
            
            # Placeholder calculators
            def energy_calc(pos):
                return {
                    'energies': np.random.randn(self.sharc_config.nstates),
                    'forces': np.random.randn(*pos.shape) * 0.01
                }
            
            def nac_calc(pos):
                return np.random.randn(self.sharc_config.nstates, 
                                      self.sharc_config.nstates) * 0.01
            
            def soc_calc(pos):
                return np.random.randn(self.sharc_config.nstates, 
                                      self.sharc_config.nstates) * 0.001
            
            positions = np.random.randn(5, 3)
            velocities = np.random.randn(5, 3) * 0.01
            
            trajectory = sh.run_sharc_trajectory(
                energy_calc, nac_calc, soc_calc,
                positions, velocities, self.sharc_config.nsteps
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.trajectories.append(trajectory)
        return trajectory
    
    def visualize_results(self, output_dir: str = "./excited_state_results"):
        """Generate visualizations for all analyses."""
        
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available for visualization")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot 1: Exciton dissociation
        if 'exciton_dissociation' in self.analyses:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            data = self.analyses['exciton_dissociation']
            traj = data['diffusion_trajectory']
            
            # Diffusion trajectory
            ax = axes[0]
            pos = traj['positions']
            ax.plot(pos[:, 0], pos[:, 1], 'b-', alpha=0.6)
            ax.plot(pos[0, 0], pos[0, 1], 'go', markersize=10, label='Start')
            ax.plot(pos[-1, 0], pos[-1, 1], 'ro', markersize=10, label='End')
            ax.set_xlabel('X (nm)')
            ax.set_ylabel('Y (nm)')
            ax.set_title('Exciton Diffusion')
            ax.legend()
            ax.axis('equal')
            ax.grid(True, alpha=0.3)
            
            # MSD
            ax = axes[1]
            msd = np.sum((pos - pos[0])**2, axis=1)
            ax.plot(traj['times'], msd, 'b-')
            ax.set_xlabel('Time (fs)')
            ax.set_ylabel('MSD (nm²)')
            ax.set_title('Mean Square Displacement')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/exciton_dynamics.png", dpi=150)
            plt.close()
        
        # Plot 2: Carrier relaxation
        if 'carrier_relaxation' in self.analyses:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            e_relax = self.analyses['carrier_relaxation']['electron_relaxation']
            h_relax = self.analyses['carrier_relaxation']['hole_relaxation']
            
            # Electron cooling
            ax = axes[0]
            ax.plot(e_relax['times'], e_relax['energies'], 'b-', label='Electron')
            ax.plot(h_relax['times'], h_relax['energies'], 'r-', label='Hole')
            ax.set_xlabel('Time (fs)')
            ax.set_ylabel('Energy (eV)')
            ax.set_title('Hot Carrier Cooling')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Temperature
            ax = axes[1]
            ax.plot(e_relax['times'], e_relax['temperatures'], 'b-', label='Electron')
            ax.plot(h_relax['times'], h_relax['temperatures'], 'r-', label='Hole')
            ax.set_xlabel('Time (fs)')
            ax.set_ylabel('Temperature (K)')
            ax.set_title('Carrier Temperature')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/carrier_relaxation.png", dpi=150)
            plt.close()
        
        # Plot 3: Energy transfer
        if 'energy_transfer' in self.analyses:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            data = self.analyses['energy_transfer']
            times = data['times']
            pops = data['populations']
            
            # Population evolution
            ax = axes[0]
            for i in range(min(5, pops.shape[1])):
                ax.plot(times, pops[:, i], label=f'Site {i}')
            ax.set_xlabel('Time (fs)')
            ax.set_ylabel('Population')
            ax.set_title('Energy Transfer Dynamics')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Network visualization
            ax = axes[1]
            pos = self.energy_network.positions[:, :2]
            ax.scatter(pos[:, 0], pos[:, 1], s=100, c=self.energy_network.energies, 
                      cmap='viridis')
            
            # Draw connections
            for i in range(self.energy_network.n_sites):
                for j in range(i+1, self.energy_network.n_sites):
                    if self.energy_network.transfer_rates[i, j] > 0:
                        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], 
                               'k-', alpha=0.2, lw=0.5)
            
            ax.set_xlabel('X (nm)')
            ax.set_ylabel('Y (nm)')
            ax.set_title('Energy Transfer Network')
            ax.axis('equal')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/energy_transfer.png", dpi=150)
            plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        
        report = []
        report.append("=" * 60)
        report.append("EXCITED STATE DYNAMICS ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Exciton dissociation
        if 'exciton_dissociation' in self.analyses:
            data = self.analyses['exciton_dissociation']
            report.append("EXCITON DISSOCIATION")
            report.append("-" * 40)
            report.append(f"Binding energy: {data['exciton'].binding_energy:.3f} eV")
            report.append(f"Exciton radius: {data['exciton'].radius:.3f} nm")
            report.append(f"Dissociation yield: {data['dissociation_yield']:.3f}")
            report.append(f"Diffusion length: {data['diffusion_length']:.3f} nm")
            report.append("")
        
        # Carrier relaxation
        if 'carrier_relaxation' in self.analyses:
            data = self.analyses['carrier_relaxation']
            report.append("HOT CARRIER RELAXATION")
            report.append("-" * 40)
            report.append(f"Electron cooling time: {data['electron_relaxation']['cooling_time']:.1f} fs")
            report.append(f"Hole cooling time: {data['hole_relaxation']['cooling_time']:.1f} fs")
            report.append(f"Total cooling time: {data['total_cooling_time']:.1f} fs")
            report.append("")
        
        # Energy transfer
        if 'energy_transfer' in self.analyses:
            data = self.analyses['energy_transfer']
            report.append("ENERGY TRANSFER")
            report.append("-" * 40)
            report.append(f"Transfer efficiency: {data['transfer_efficiency']:.3f}")
            report.append(f"Transfer time: {data['transfer_time']:.1f} fs")
            report.append(f"Number of pathways: {len(data['pathways'])}")
            report.append("")
        
        report.append("=" * 60)
        
        return '\n'.join(report)


def demo_excited_state_dynamics():
    """Demonstrate excited state dynamics workflow."""
    
    print("=" * 70)
    print("EXCITED STATE DYNAMICS DEMONSTRATION")
    print("=" * 70)
    
    # Create workflow
    workflow = ExcitedStateDynamicsWorkflow()
    
    # Setup exciton system
    print("\n1. Setting up exciton system...")
    workflow.setup_exciton_system(
        exciton_energies=[2.0, 2.2, 2.5],
        binding_energies=[0.4, 0.35, 0.3],
        radii=[1.5, 1.2, 1.0]
    )
    
    # Setup carrier system
    print("2. Setting up carrier system...")
    workflow.setup_carrier_system(
        electron_masses=[0.3, 0.5],
        hole_masses=[0.5, 0.8],
        mobilities={'electron': 1.0, 'hole': 0.1}
    )
    
    # Setup energy network
    print("3. Setting up energy transfer network...")
    positions = np.random.randn(10, 3) * 5  # 10 sites
    energies = np.linspace(2.0, 1.5, 10)
    workflow.setup_energy_network(positions, energies)
    
    # Run simulations
    print("\n4. Running exciton dissociation simulation...")
    exciton_results = workflow.run_exciton_dissociation(
        initial_exciton_idx=0, electric_field=0.1
    )
    print(f"   Dissociation yield: {exciton_results['dissociation_yield']:.3f}")
    print(f"   Diffusion length: {exciton_results['diffusion_length']:.2f} nm")
    
    print("\n5. Running hot carrier relaxation...")
    carrier_results = workflow.run_carrier_relaxation(
        initial_electron_energy=0.8, initial_hole_energy=0.5
    )
    print(f"   Electron cooling time: {carrier_results['electron_relaxation']['cooling_time']:.1f} fs")
    print(f"   Hole cooling time: {carrier_results['hole_relaxation']['cooling_time']:.1f} fs")
    
    print("\n6. Running energy transfer simulation...")
    transfer_results = workflow.run_energy_transfer(
        donor_site=0, acceptor_site=9, simulation_time=2000.0
    )
    print(f"   Transfer efficiency: {transfer_results['transfer_efficiency']:.3f}")
    print(f"   Transfer time: {transfer_results['transfer_time']:.1f} fs")
    print(f"   Number of pathways: {len(transfer_results['pathways'])}")
    
    # Generate report
    print("\n" + "=" * 70)
    print(workflow.generate_report())
    
    # Visualizations
    if MATPLOTLIB_AVAILABLE:
        print("\n7. Generating visualizations...")
        workflow.visualize_results()
        print("   Done! Check excited_state_results/ directory")
    
    return workflow


if __name__ == "__main__":
    demo_excited_state_dynamics()
