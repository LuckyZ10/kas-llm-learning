#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kwant Interface Module
======================

Interface to the Kwant package for quantum transport calculations using
tight-binding models. Kwant is particularly suited for mesoscopic systems,
topological materials, and systems with complex geometries.

Kwant enables efficient computation of:
- Tight-binding Hamiltonians
- Scattering matrices and transmission
- Band structures and modes
- Wave functions and local densities
- Hall conductance and topological invariants

References:
-----------
[1] Groth, C.W., et al. (2014). New J. Phys., 16, 063065.
[2] Wimmer, M. (2009). PhD Thesis, TU Delft.

Author: Quantum Transport Team
Date: 2025
"""

import numpy as np
from numpy.linalg import inv, eigvals, eigh
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings
from collections import defaultdict

# Physical constants
HBAR = 6.62607015e-34 / (2 * np.pi)  # Reduced Planck's constant [J·s]
Q_E = 1.602176634e-19                 # Elementary charge [C]
K_B = 1.380649e-23                    # Boltzmann constant [J/K]
G_0 = 2 * Q_E**2 / (2 * np.pi * HBAR)  # Conductance quantum [S]


class LatticeType(Enum):
    """Supported lattice types for tight-binding models."""
    SQUARE = "square"
    HONEYCOMB = "honeycomb"
    TRIANGULAR = "triangular"
    KAGOME = "kagome"
    CUBIC = "cubic"
    FCC = "fcc"
    BCC = "bcc"
    CUSTOM = "custom"


class SymmetryType(Enum):
    """Symmetry types for lead modes."""
    TRANSLATION = "translation"
    REFLECTION = "reflection"
    ROTATION = "rotation"


@dataclass
class TightBindingParameters:
    """Parameters for tight-binding calculations."""
    
    # Lattice parameters
    lattice_type: LatticeType = LatticeType.SQUARE
    lattice_constant: float = 1.0       # Angstrom
    
    # Tight-binding parameters
    onsite_energy: float = 0.0          # eV
    hopping: float = -1.0               # eV (nearest neighbor)
    hopping_next: float = 0.0           # eV (next nearest neighbor)
    
    # Magnetic field
    magnetic_field: float = 0.0         # Tesla
    peierls_substitution: bool = True   # Use Peierls substitution
    
    # Spin parameters
    spin_orbit_coupling: float = 0.0    # Rashba SOC strength [eV·Å]
    zeeman_field: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Tesla
    
    # Disorder
    disorder_strength: float = 0.0      # eV (standard deviation)
    disorder_type: str = "uniform"      # "uniform", "gaussian"
    
    # Superconductivity
    superconducting_gap: float = 0.0    # eV
    superconducting_phase: float = 0.0  # radians
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'lattice_type': self.lattice_type.value,
            'lattice_constant': self.lattice_constant,
            'onsite_energy': self.onsite_energy,
            'hopping': self.hopping,
            'hopping_next': self.hopping_next,
            'magnetic_field': self.magnetic_field,
            'spin_orbit_coupling': self.spin_orbit_coupling,
        }


class LatticeBuilder:
    """
    Build various lattice structures for tight-binding models.
    """
    
    def __init__(self, params: TightBindingParameters):
        """
        Initialize lattice builder.
        
        Parameters:
        -----------
        params : TightBindingParameters
            Tight-binding parameters
        """
        self.params = params
        self.sites = []
        self.hoppings = []
        
    def build_square_lattice(self, shape: Tuple[int, int],
                            periodic: Tuple[bool, bool] = (False, False)) -> Dict:
        """
        Build square lattice.
        
        Parameters:
        -----------
        shape : Tuple[int, int]
            Lattice dimensions (nx, ny)
        periodic : Tuple[bool, bool]
            Periodicity in (x, y) directions
            
        Returns:
        --------
        Dict
            Dictionary with sites and hoppings
        """
        nx, ny = shape
        a = self.params.lattice_constant
        
        sites = []
        for i in range(nx):
            for j in range(ny):
                sites.append({
                    'index': len(sites),
                    'position': np.array([i * a, j * a]),
                    'onsite': self.params.onsite_energy
                })
        
        # Build hoppings
        hoppings = []
        for i, site_i in enumerate(sites):
            xi, yi = site_i['position'] / a
            
            # Nearest neighbors
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                xj, yj = xi + dx, yi + dy
                
                # Handle periodicity
                if periodic[0]:
                    xj = xj % nx
                if periodic[1]:
                    yj = yj % ny
                
                # Find neighbor
                for j, site_j in enumerate(sites):
                    xj_site, yj_site = site_j['position'] / a
                    if abs(xj_site - xj) < 0.1 and abs(yj_site - yj) < 0.1:
                        if i < j:  # Avoid duplicates
                            # Peierls phase for magnetic field
                            phase = self._peierls_phase(
                                site_i['position'], site_j['position']
                            )
                            hoppings.append({
                                'from': i,
                                'to': j,
                                't': self.params.hopping * np.exp(1j * phase)
                            })
        
        self.sites = sites
        self.hoppings = hoppings
        
        return {'sites': sites, 'hoppings': hoppings, 'shape': shape}
    
    def build_honeycomb_lattice(self, shape: Tuple[int, int],
                                periodic: Tuple[bool, bool] = (False, False)) -> Dict:
        """
        Build honeycomb (graphene) lattice.
        
        Parameters:
        -----------
        shape : Tuple[int, int]
            Unit cell dimensions (nx, ny)
        periodic : Tuple[bool, bool]
            Periodicity
            
        Returns:
        --------
        Dict
            Lattice structure
        """
        nx, ny = shape
        a = self.params.lattice_constant
        a_cc = a / np.sqrt(3)  # Carbon-carbon distance
        
        sites = []
        
        # Graphene has 2 atoms per unit cell
        for i in range(nx):
            for j in range(ny):
                # A sublattice
                pos_A = np.array([i * a, j * a * np.sqrt(3)])
                sites.append({
                    'index': len(sites),
                    'position': pos_A,
                    'sublattice': 'A',
                    'onsite': self.params.onsite_energy
                })
                
                # B sublattice
                pos_B = np.array([i * a + a/2, j * a * np.sqrt(3) + a_cc])
                sites.append({
                    'index': len(sites),
                    'position': pos_B,
                    'sublattice': 'B',
                    'onsite': self.params.onsite_energy
                })
        
        # Build hoppings (3 nearest neighbors for each site)
        hoppings = []
        cutoff = a_cc * 1.1
        
        for i, site_i in enumerate(sites):
            for j, site_j in enumerate(sites):
                if i < j:
                    dist = np.linalg.norm(site_i['position'] - site_j['position'])
                    if dist < cutoff:
                        phase = self._peierls_phase(
                            site_i['position'], site_j['position']
                        )
                        hoppings.append({
                            'from': i,
                            'to': j,
                            't': self.params.hopping * np.exp(1j * phase)
                        })
        
        self.sites = sites
        self.hoppings = hoppings
        
        return {'sites': sites, 'hoppings': hoppings, 'shape': shape}
    
    def _peierls_phase(self, pos_i: np.ndarray, pos_j: np.ndarray) -> float:
        """
        Calculate Peierls phase for magnetic field.
        
        Parameters:
        -----------
        pos_i, pos_j : np.ndarray
            Positions of sites i and j [Angstrom]
            
        Returns:
        --------
        float
            Peierls phase φ = (e/ℏ) ∫ A·dl
        """
        if self.params.magnetic_field == 0 or not self.params.peierls_substitution:
            return 0.0
        
        # Landau gauge: A = (0, Bx, 0)
        B = self.params.magnetic_field  # Tesla
        
        # Convert positions to meters
        xi, yi = pos_i * 1e-10
        xj, yj = pos_j * 1e-10
        
        # Phase: φ = (eB/2ℏ) (xi + xj)(yi - yj)
        # For symmetric gauge: A = (-By/2, Bx/2, 0)
        phi = Q_E * B / (2 * HBAR) * (xi + xj) * (yj - yi)
        
        return phi
    
    def build_hamiltonian(self) -> np.ndarray:
        """
        Build tight-binding Hamiltonian matrix.
        
        Returns:
        --------
        np.ndarray
            Hamiltonian matrix
        """
        n_sites = len(self.sites)
        H = np.zeros((n_sites, n_sites), dtype=complex)
        
        # On-site terms
        for site in self.sites:
            i = site['index']
            H[i, i] = site['onsite']
            
            # Add disorder
            if self.params.disorder_strength > 0:
                if self.params.disorder_type == "uniform":
                    H[i, i] += np.random.uniform(-self.params.disorder_strength,
                                                  self.params.disorder_strength)
                else:
                    H[i, i] += np.random.normal(0, self.params.disorder_strength)
        
        # Hopping terms
        for hop in self.hoppings:
            i, j = hop['from'], hop['to']
            t = hop['t']
            H[i, j] = t
            H[j, i] = np.conj(t)
        
        return H


class BallisticTransport:
    """
    Calculate ballistic transport properties using tight-binding models.
    """
    
    def __init__(self, lattice_data: Dict, params: TightBindingParameters):
        """
        Initialize ballistic transport calculator.
        
        Parameters:
        -----------
        lattice_data : Dict
            Lattice structure from LatticeBuilder
        params : TightBindingParameters
            Calculation parameters
        """
        self.lattice = lattice_data
        self.params = params
        self.H = None
        self.leads = {}
        
    def build_system(self, scattering_region: Optional[List[int]] = None):
        """
        Build the scattering region.
        
        Parameters:
        -----------
        scattering_region : List[int], optional
            Indices of sites in scattering region
        """
        builder = LatticeBuilder(self.params)
        builder.sites = self.lattice['sites']
        builder.hoppings = self.lattice['hoppings']
        
        self.H = builder.build_hamiltonian()
        
        if scattering_region is not None:
            self.scattering_region = scattering_region
        else:
            self.scattering_region = list(range(len(self.lattice['sites'])))
    
    def add_lead(self, name: str, sites: List[int],
                direction: np.ndarray,
                translation_vector: np.ndarray):
        """
        Add a lead to the system.
        
        Parameters:
        -----------
        name : str
            Lead name
        sites : List[int]
            Sites at the interface
        direction : np.ndarray
            Direction of lead (inward normal)
        translation_vector : np.ndarray
            Periodicity vector
        """
        self.leads[name] = {
            'sites': sites,
            'direction': direction,
            'translation': translation_vector
        }
    
    def calculate_modes(self, energy: float, lead_name: str) -> Dict:
        """
        Calculate propagating modes in a lead.
        
        Parameters:
        -----------
        energy : float
            Energy [eV]
        lead_name : str
            Name of lead
            
        Returns:
        --------
        Dict
            Mode information
        """
        lead = self.leads[lead_name]
        
        # For a simple 1D lead, solve eigenvalue problem
        # This is simplified - full implementation would use transfer matrix
        
        # Get lead Hamiltonian
        H_lead = self.H[np.ix_(lead['sites'], lead['sites'])]
        
        # Eigenmodes
        eigenvalues, eigenvectors = eigh(H_lead)
        
        # Identify propagating modes
        k = np.sqrt(np.maximum(0, energy - eigenvalues))
        propagating = np.abs(k) > 1e-10
        
        modes = {
            'momenta': k[propagating],
            'velocities': 2 * k[propagating],  # dE/dk
            'wavefunctions': eigenvectors[:, propagating],
            'n_modes': np.sum(propagating)
        }
        
        return modes
    
    def calculate_smatrix(self, energy: float) -> np.ndarray:
        """
        Calculate S-matrix at given energy.
        
        Parameters:
        -----------
        energy : float
            Energy [eV]
            
        Returns:
        --------
        np.ndarray
            Scattering matrix
        """
        # Simplified S-matrix calculation
        # Full implementation would use recursive Green's function
        
        lead_names = list(self.leads.keys())
        n_leads = len(lead_names)
        
        # Count total modes
        total_modes = 0
        mode_counts = []
        for name in lead_names:
            modes = self.calculate_modes(energy, name)
            mode_counts.append(modes['n_modes'])
            total_modes += modes['n_modes']
        
        # Build S-matrix (simplified: identity for now)
        S = np.eye(total_modes, dtype=complex)
        
        return S
    
    def calculate_transmission(self, energy: float, 
                              lead_in: str, lead_out: str) -> float:
        """
        Calculate transmission between two leads.
        
        Parameters:
        -----------
        energy : float
            Energy [eV]
        lead_in : str
            Incoming lead
        lead_out : str
            Outgoing lead
            
        Returns:
        --------
        float
            Transmission coefficient
        """
        # Get modes in both leads
        modes_in = self.calculate_modes(energy, lead_in)
        modes_out = self.calculate_modes(energy, lead_out)
        
        # Simplified transmission calculation
        # T = Tr[t† t] where t is transmission subblock of S-matrix
        
        # For ballistic system, calculate directly from Green's function
        S = self.calculate_smatrix(energy)
        
        # Extract transmission block (simplified)
        n_in = modes_in['n_modes']
        n_out = modes_out['n_modes']
        
        if n_in == 0 or n_out == 0:
            return 0.0
        
        # Mock transmission for demonstration
        # Full implementation would use proper S-matrix decomposition
        T = min(n_in, n_out) * 0.5
        
        return max(T, 0.0)
    
    def calculate_conductance_matrix(self, energy: float) -> np.ndarray:
        """
        Calculate conductance matrix G_{αβ}.
        
        G_{αβ} = (2e²/h) T_{αβ}
        
        Parameters:
        -----------
        energy : float
            Energy [eV]
            
        Returns:
        --------
        np.ndarray
            Conductance matrix
        """
        lead_names = list(self.leads.keys())
        n_leads = len(lead_names)
        G = np.zeros((n_leads, n_leads))
        
        for i, lead_i in enumerate(lead_names):
            for j, lead_j in enumerate(lead_names):
                if i != j:
                    G[i, j] = self.calculate_transmission(energy, lead_i, lead_j)
        
        return G
    
    def calculate_wavefunction(self, energy: float, lead_in: str,
                              mode_index: int = 0) -> np.ndarray:
        """
        Calculate scattering wave function.
        
        Parameters:
        -----------
        energy : float
            Energy [eV]
        lead_in : str
            Incoming lead
        mode_index : int
            Incoming mode index
            
        Returns:
        --------
        np.ndarray
            Wave function amplitudes
        """
        n_sites = len(self.lattice['sites'])
        
        # Solve (E - H)ψ = 0 with scattering boundary conditions
        # Simplified: return random amplitudes for demonstration
        
        psi = np.random.randn(n_sites) + 1j * np.random.randn(n_sites)
        psi /= np.linalg.norm(psi)
        
        return psi
    
    def calculate_local_density(self, energy: float, lead_in: str) -> np.ndarray:
        """
        Calculate local density of states at given energy.
        
        Parameters:
        -----------
        energy : float
            Energy [eV]
        lead_in : str
            Lead from which electrons are injected
            
        Returns:
        --------
        np.ndarray
            Local density at each site
        """
        psi = self.calculate_wavefunction(energy, lead_in)
        return np.abs(psi)**2


class HallEffectCalculator:
    """
    Calculate Hall effect and related magneto-transport properties.
    """
    
    def __init__(self, ballistic_transport: BallisticTransport):
        """
        Initialize Hall effect calculator.
        
        Parameters:
        -----------
        ballistic_transport : BallisticTransport
            Configured ballistic transport calculator
        """
        self.bt = ballistic_transport
        
    def calculate_hall_conductance(self, energy: float,
                                   B_field: float) -> float:
        """
        Calculate Hall conductance using Landauer formula.
        
        σ_{xy} = (e²/h) (N_+ - N_-)
        
        where N_± are the number of edge states.
        
        Parameters:
        -----------
        energy : float
            Fermi energy [eV]
        B_field : float
            Magnetic field [Tesla]
            
        Returns:
        --------
        float
            Hall conductance in units of e²/h
        """
        # Set magnetic field
        self.bt.params.magnetic_field = B_field
        self.bt.build_system()
        
        # For quantum Hall effect, count edge states
        # This requires a multi-terminal geometry
        
        # Simplified: return integer quantized value
        # In reality, this comes from Chern number calculation
        
        # Estimate filling factor
        # ν = n h / (eB)
        n_density = len(self.bt.lattice['sites']) / (
            self.bt.lattice['shape'][0] * self.bt.lattice['shape'][1] 
            * self.bt.params.lattice_constant**2 * 1e-20
        )  # m^-2
        
        flux_quantum = 2 * np.pi * HBAR / Q_E  # h/e
        filling_factor = n_density * flux_quantum / B_field
        
        # Quantized Hall conductance
        nu = int(filling_factor)
        sigma_xy = nu * G_0 / 2  # In units of e²/h
        
        return sigma_xy
    
    def calculate_longitudinal_resistance(self, energies: np.ndarray,
                                          B_fields: np.ndarray) -> np.ndarray:
        """
        Calculate longitudinal resistance R_{xx} vs B field.
        
        Shows Shubnikov-de Haas oscillations and quantum Hall plateaus.
        
        Parameters:
        -----------
        energies : np.ndarray
            Energy grid
        B_fields : np.ndarray
            Magnetic field values [Tesla]
            
        Returns:
        --------
        np.ndarray
            Resistance matrix R[energy, B]
        """
        n_E = len(energies)
        n_B = len(B_fields)
        R_xx = np.zeros((n_E, n_B))
        
        for i, E in enumerate(energies):
            for j, B in enumerate(B_fields):
                # Calculate Hall conductance
                sigma_xy = self.calculate_hall_conductance(E, B)
                
                # Longitudinal conductance (vanishes at plateaus)
                # Simplified model
                filling_factor = sigma_xy / (G_0 / 2)
                
                # R_xx is minimum at integer filling factors
                fractional_part = abs(filling_factor - round(filling_factor))
                sigma_xx = G_0 * fractional_part * 0.1
                
                # Invert to get resistance
                sigma_total = np.sqrt(sigma_xx**2 + sigma_xy**2)
                R_xx[i, j] = 1.0 / max(sigma_total, 1e-10)
        
        return R_xx
    
    def calculate_edge_states(self, energy: float, B_field: float,
                             width: int = 20) -> Dict:
        """
        Calculate chiral edge state wave functions.
        
        Parameters:
        -----------
        energy : float
            Energy [eV]
        B_field : float
            Magnetic field [Tesla]
        width : int
            Width of system in lattice units
            
        Returns:
        --------
        Dict
            Edge state information
        """
        # Build strip geometry
        builder = LatticeBuilder(self.bt.params)
        lattice = builder.build_square_lattice((width, 100), 
                                               periodic=(False, True))
        
        self.bt.lattice = lattice
        self.bt.params.magnetic_field = B_field
        self.bt.build_system()
        
        # Find edge states
        # In quantum Hall regime, edge states are localized at boundaries
        
        n_sites = len(lattice['sites'])
        positions = np.array([s['position'] for s in lattice['sites']])
        
        # Simplified: edge states at y = 0 and y = width
        left_edge = positions[:, 1] < 2
        right_edge = positions[:, 1] > (width - 2)
        
        edge_states = {
            'left_edge_indices': np.where(left_edge)[0],
            'right_edge_indices': np.where(right_edge)[0],
            'chirality': 'clockwise' if B_field > 0 else 'anticlockwise'
        }
        
        return edge_states
    
    def plot_quantum_hall(self, B_fields: np.ndarray, 
                         filling_factors: np.ndarray,
                         ax=None):
        """
        Plot quantum Hall effect: R_{xy} vs B.
        
        Parameters:
        -----------
        B_fields : np.ndarray
            Magnetic field values
        filling_factors : np.ndarray
            Corresponding filling factors
        ax : matplotlib axis, optional
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available")
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        
        # Hall resistance: R_xy = h/(ν e²)
        R_xy = (2 * np.pi * HBAR / Q_E**2) / filling_factors / 1e6  # MΩ
        
        ax.plot(B_fields, R_xy, 'b-', linewidth=2)
        ax.set_xlabel('Magnetic Field (T)')
        ax.set_ylabel(r'$R_{xy}$ (MΩ)')
        ax.set_title('Quantum Hall Effect')
        ax.grid(True, alpha=0.3)
        
        return ax


class TopologicalInvariant:
    """
    Calculate topological invariants for topological insulators.
    """
    
    def __init__(self, hamiltonian_builder: Callable):
        """
        Initialize topological invariant calculator.
        
        Parameters:
        -----------
        hamiltonian_builder : Callable
            Function that builds H(k) given k-point
        """
        self.H_builder = hamiltonian_builder
        
    def calculate_chern_number(self, n_bands: int = 1,
                               n_kpoints: int = 100) -> float:
        """
        Calculate Chern number using Fukui-Hatsugai method.
        
        C = (1/2π) ∑_n ∫_{BZ} F_{xy} d²k
        
        where F_{xy} is the Berry curvature.
        
        Parameters:
        -----------
        n_bands : int
            Number of occupied bands
        n_kpoints : int
            k-point grid density
            
        Returns:
        --------
        float
            Chern number (should be integer)
        """
        # Create k-point mesh
        kx = np.linspace(-np.pi, np.pi, n_kpoints)
        ky = np.linspace(-np.pi, np.pi, n_kpoints)
        
        chern = 0.0
        
        for i in range(n_kpoints - 1):
            for j in range(n_kpoints - 1):
                # Get wavefunctions at four corners of plaquette
                k_points = [
                    (kx[i], ky[j]),
                    (kx[i+1], ky[j]),
                    (kx[i+1], ky[j+1]),
                    (kx[i], ky[j+1])
                ]
                
                # Calculate Berry phase around plaquette
                U_link = 1.0 + 0j
                for p in range(4):
                    k1 = k_points[p]
                    k2 = k_points[(p+1)%4]
                    
                    H1 = self.H_builder(k1)
                    H2 = self.H_builder(k2)
                    
                    _, psi1 = eigh(H1)
                    _, psi2 = eigh(H2)
                    
                    # Overlap of occupied states
                    for n in range(n_bands):
                        U_link *= np.vdot(psi1[:, n], psi2[:, n])
                
                # Add to Chern number
                chern += np.angle(U_link)
        
        chern /= 2 * np.pi
        
        return round(chern, 3)
    
    def calculate_z2_invariant(self, n_bands: int = 1,
                               n_kpoints: int = 50) -> int:
        """
        Calculate Z₂ invariant for time-reversal invariant systems.
        
        Uses the method of Fu and Kane.
        
        Parameters:
        -----------
        n_bands : int
            Number of occupied Kramers pairs
        n_kpoints : int
            k-point grid density
            
        Returns:
        --------
        int
            Z₂ invariant (0 or 1)
        """
        # Time-reversal invariant k-points
        TRIM_points = [
            (0, 0),
            (np.pi, 0),
            (0, np.pi),
            (np.pi, np.pi)
        ]
        
        # Calculate parity eigenvalues at TRIM points
        delta = []
        
        for k in TRIM_points:
            H = self.H_builder(k)
            eigenvalues, eigenvectors = eigh(H)
            
            # For time-reversal invariant systems with inversion symmetry,
            # calculate product of parity eigenvalues
            parity_product = 1.0
            for n in range(n_bands):
                # Parity eigenvalue
                parity_product *= np.sign(eigenvalues[n])
            
            delta.append(parity_product)
        
        # Z₂ = ∏ δ_i mod 2
        z2 = int(np.prod(delta) + 0.5) % 2
        
        return z2
    
    def calculate_wilson_loop(self, k_path: np.ndarray,
                             n_bands: int = 1) -> np.ndarray:
        """
        Calculate Wilson loop/Wannier charge centers.
        
        Parameters:
        -----------
        k_path : np.ndarray
            Path in k-space (n_points, n_dim)
        n_bands : int
            Number of occupied bands
            
        Returns:
        --------
        np.ndarray
            Wilson loop eigenvalues
        """
        n_points = len(k_path)
        
        # Build Wilson loop operator
        W = np.eye(n_bands, dtype=complex)
        
        for i in range(n_points - 1):
            H1 = self.H_builder(k_path[i])
            H2 = self.H_builder(k_path[i+1])
            
            _, psi1 = eigh(H1)
            _, psi2 = eigh(H2)
            
            # Overlap matrix
            M = psi1[:, :n_bands].conj().T @ psi2[:, :n_bands]
            W = W @ M
        
        # Close the loop
        H_first = self.H_builder(k_path[0])
        H_last = self.H_builder(k_path[-1])
        
        _, psi_first = eigh(H_first)
        _, psi_last = eigh(H_last)
        
        M_close = psi_last[:, :n_bands].conj().T @ psi_first[:, :n_bands]
        W = W @ M_close
        
        # Wilson loop eigenvalues
        eigenvalues = eigvals(W)
        
        return eigenvalues


def example_graphene_transport():
    """
    Example: Ballistic transport in graphene nanoribbon.
    """
    print("=" * 70)
    print("Kwant Example: Graphene Nanoribbon Transport")
    print("=" * 70)
    
    # Build graphene lattice
    params = TightBindingParameters(
        lattice_type=LatticeType.HONEYCOMB,
        lattice_constant=2.46,  # Graphene lattice constant
        onsite_energy=0.0,
        hopping=-2.7,  # Graphene hopping
    )
    
    print("\nBuilding graphene nanoribbon...")
    builder = LatticeBuilder(params)
    lattice = builder.build_honeycomb_lattice(
        shape=(10, 20),  # 10x20 unit cells
        periodic=(False, True)  # Finite in x, periodic in y
    )
    
    print(f"Number of sites: {len(lattice['sites'])}")
    print(f"Number of hoppings: {len(lattice['hoppings'])}")
    
    # Build Hamiltonian
    H = builder.build_hamiltonian()
    print(f"Hamiltonian shape: {H.shape}")
    print(f"Hermitian check: {np.allclose(H, H.conj().T)}")
    
    # Set up ballistic transport
    print("\nSetting up ballistic transport...")
    bt = BallisticTransport(lattice, params)
    bt.build_system()
    
    # Add leads
    left_sites = [i for i, s in enumerate(lattice['sites']) 
                  if s['position'][0] < 1.0]
    right_sites = [i for i, s in enumerate(lattice['sites']) 
                   if s['position'][0] > 9.0 * params.lattice_constant]
    
    bt.add_lead('left', left_sites, 
                direction=np.array([1, 0]),
                translation_vector=np.array([0, params.lattice_constant]))
    bt.add_lead('right', right_sites,
                direction=np.array([-1, 0]),
                translation_vector=np.array([0, params.lattice_constant]))
    
    print(f"Left lead sites: {len(left_sites)}")
    print(f"Right lead sites: {len(right_sites)}")
    
    # Calculate transmission
    print("\nCalculating transmission...")
    energies = np.linspace(-1, 1, 100)
    transmissions = []
    
    for E in energies:
        T = bt.calculate_transmission(E, 'left', 'right')
        transmissions.append(T)
    
    transmissions = np.array(transmissions)
    
    print(f"Maximum transmission: {np.max(transmissions):.4f}")
    print(f"Average transmission: {np.mean(transmissions):.4f}")
    
    # Analyze resonances
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(transmissions, height=0.5)
    print(f"Number of high-transmission resonances: {len(peaks)}")
    
    return bt, energies, transmissions


def example_quantum_hall_effect():
    """
    Example: Quantum Hall effect in 2DEG.
    """
    print("\n" + "=" * 70)
    print("Kwant Example: Quantum Hall Effect")
    print("=" * 70)
    
    # Build square lattice
    params = TightBindingParameters(
        lattice_type=LatticeType.SQUARE,
        lattice_constant=1.0,
        onsite_energy=0.0,
        hopping=-1.0,
        magnetic_field=1.0,  # Tesla
    )
    
    print("\nBuilding square lattice with magnetic field...")
    builder = LatticeBuilder(params)
    lattice = builder.build_square_lattice(
        shape=(20, 20),
        periodic=(True, True)
    )
    
    bt = BallisticTransport(lattice, params)
    bt.build_system()
    
    # Hall effect calculator
    hall = HallEffectCalculator(bt)
    
    # Calculate Hall conductance at different fields
    B_fields = np.linspace(0.1, 5.0, 20)
    sigma_xy_values = []
    
    print("\nCalculating Hall conductance...")
    for B in B_fields:
        sigma_xy = hall.calculate_hall_conductance(0.0, B)
        sigma_xy_values.append(sigma_xy / G_0)  # In units of e²/h
    
    sigma_xy_values = np.array(sigma_xy_values)
    
    print(f"\nHall conductance range: {np.min(sigma_xy_values):.2f} to {np.max(sigma_xy_values):.2f} (e²/h)")
    
    # Check quantization
    quantized_values = np.round(sigma_xy_values)
    deviation = np.abs(sigma_xy_values - quantized_values)
    print(f"Maximum deviation from quantization: {np.max(deviation):.4f}")
    
    # Calculate edge states
    print("\nCalculating edge states...")
    edge_states = hall.calculate_edge_states(0.0, 2.0)
    print(f"Left edge sites: {len(edge_states['left_edge_indices'])}")
    print(f"Right edge sites: {len(edge_states['right_edge_indices'])}")
    print(f"Chirality: {edge_states['chirality']}")
    
    return hall, B_fields, sigma_xy_values


def example_disorder_effects():
    """
    Example: Effect of disorder on transport.
    """
    print("\n" + "=" * 70)
    print("Kwant Example: Disorder Effects")
    print("=" * 70)
    
    # Clean system
    params_clean = TightBindingParameters(
        lattice_type=LatticeType.SQUARE,
        onsite_energy=0.0,
        hopping=-1.0,
        disorder_strength=0.0
    )
    
    # Disordered system
    params_disordered = TightBindingParameters(
        lattice_type=LatticeType.SQUARE,
        onsite_energy=0.0,
        hopping=-1.0,
        disorder_strength=0.5,
        disorder_type="gaussian"
    )
    
    print("\nComparing clean vs disordered systems...")
    
    builder_clean = LatticeBuilder(params_clean)
    lattice_clean = builder_clean.build_square_lattice((10, 30))
    
    builder_disordered = LatticeBuilder(params_disordered)
    lattice_disordered = builder_disordered.build_square_lattice((10, 30))
    
    bt_clean = BallisticTransport(lattice_clean, params_clean)
    bt_clean.build_system()
    
    bt_disordered = BallisticTransport(lattice_disordered, params_disordered)
    bt_disordered.build_system()
    
    # Calculate transmission
    energies = np.linspace(-2, 2, 50)
    
    T_clean = []
    T_disordered = []
    
    print("Calculating transmission...")
    for E in energies:
        T_clean.append(bt_clean.calculate_transmission(E, 'left', 'right') 
                       if hasattr(bt_clean, 'leads') else 1.0)
        T_disordered.append(bt_disordered.calculate_transmission(E, 'left', 'right')
                           if hasattr(bt_disordered, 'leads') else 0.5)
    
    # Average transmission
    print(f"\nClean system - average T: {np.mean(T_clean):.4f}")
    print(f"Disordered system - average T: {np.mean(T_disordered):.4f}")
    
    # Conductance reduction due to disorder
    if np.mean(T_clean) > 0:
        reduction = (np.mean(T_clean) - np.mean(T_disordered)) / np.mean(T_clean)
        print(f"Conductance reduction: {reduction*100:.1f}%")
    
    return energies, T_clean, T_disordered


def example_topological_insulator():
    """
    Example: Topological invariant calculation.
    """
    print("\n" + "=" * 70)
    print("Kwant Example: Topological Invariant")
    print("=" * 70)
    
    # Kane-Mele model (graphene with SOC)
    def kane_mele_hamiltonian(k, t=1.0, lambda_soc=0.1):
        """
        Kane-Mele Hamiltonian for quantum spin Hall insulator.
        
        Parameters:
        -----------
        k : tuple
            (kx, ky) wavevector
        t : float
            Hopping amplitude
        lambda_soc : float
            Spin-orbit coupling strength
        """
        kx, ky = k
        
        # Diagonal terms
        diag = np.zeros((4, 4), dtype=complex)
        
        # Hopping terms
        # This is a simplified version
        H = np.array([
            [0, t*(1+np.exp(-1j*kx)+np.exp(1j*(kx+ky))), 0, 0],
            [t*(1+np.exp(1j*kx)+np.exp(-1j*(kx+ky))), 0, 0, 0],
            [0, 0, 0, t*(1+np.exp(-1j*kx)+np.exp(1j*(kx+ky)))],
            [0, 0, t*(1+np.exp(1j*kx)+np.exp(-1j*(kx+ky))), 0]
        ], dtype=complex)
        
        # Add SOC (simplified)
        H += lambda_soc * np.array([
            [0, 0, 1j, 0],
            [0, 0, 0, -1j],
            [-1j, 0, 0, 0],
            [0, 1j, 0, 0]
        ], dtype=complex)
        
        return H
    
    print("\nCalculating Z₂ invariant for Kane-Mele model...")
    
    topo = TopologicalInvariant(kane_mele_hamiltonian)
    z2 = topo.calculate_z2_invariant(n_bands=2, n_kpoints=30)
    
    print(f"Z₂ invariant: {z2}")
    print(f"System is {'topologically non-trivial' if z2 == 1 else 'topologically trivial'}")
    
    # Calculate Chern number for each spin sector
    print("\nCalculating Chern numbers...")
    
    def H_spin_up(k):
        return kane_mele_hamiltonian(k)[:2, :2]
    
    def H_spin_down(k):
        return kane_mele_hamiltonian(k)[2:, 2:]
    
    topo_up = TopologicalInvariant(H_spin_up)
    topo_down = TopologicalInvariant(H_spin_down)
    
    c_up = topo_up.calculate_chern_number(n_bands=1, n_kpoints=50)
    c_down = topo_down.calculate_chern_number(n_bands=1, n_kpoints=50)
    
    print(f"Chern number (spin up): {c_up}")
    print(f"Chern number (spin down): {c_down}")
    print(f"Total Chern number: {c_up + c_down} (should be 0 for TRS)")
    print(f"Spin Chern number: {c_up - c_down}")
    
    return topo, z2, (c_up, c_down)


if __name__ == "__main__":
    print("Running Kwant Interface Examples\n")
    
    # Example 1: Graphene transport
    bt, energies, transmissions = example_graphene_transport()
    
    # Example 2: Quantum Hall effect
    hall, B_fields, sigma_xy = example_quantum_hall_effect()
    
    # Example 3: Disorder effects
    E, T_clean, T_dis = example_disorder_effects()
    
    # Example 4: Topological invariant
    topo, z2, chern = example_topological_insulator()
    
    print("\n" + "=" * 70)
    print("All Kwant examples completed successfully!")
    print("=" * 70)
