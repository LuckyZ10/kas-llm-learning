"""
Weyl Semimetal Module
=====================

This module provides comprehensive tools for studying Weyl semimetals:
- Weyl point location and classification
- Chirality calculation
- Fermi arc surface states
- Berry curvature near Weyl points
- Magnetotransport properties

Weyl semimetals are 3D topological materials with gapless Weyl points
in the bulk and Fermi arcs on surfaces.

References:
- Wan et al., PRL 107, 127601 (2011)
- Burkov, PRL 107, 127205 (2011)
- Weng et al., PRX 5, 011029 (2015) - TaAs
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, fsolve
from scipy.interpolate import RegularGridInterpolator
from abc import ABC, abstractmethod

try:
    from pymatgen.core import Structure, Lattice, Element
    from pymatgen.io.vasp import Poscar, Kpoints, Incar, Outcar, Vasprun
    HAS_PMG = True
except ImportError:
    HAS_PMG = False
    warnings.warn("Pymatgen not available.")

try:
    import ase
    from ase.io import read, write
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    warnings.warn("ASE not available.")


class WeylType(Enum):
    """Types of Weyl points."""
    TYPE_I = auto()    # |v_⊥| < |v_∥| (tilted cones)
    TYPE_II = auto()   # |v_⊥| > |v_∥| (overtilted cones)
    MULTIPOLE = auto() # Higher-order Weyl points


class FermiArcType(Enum):
    """Types of Fermi arcs."""
    SINGLE = auto()    # Single Fermi arc connecting Weyl points
    DOUBLE = auto()    # Double Fermi arc
    CLOSED = auto()    # Closed Fermi surface


@dataclass
class WeylSemimetalConfig:
    """Configuration for Weyl semimetal calculations."""
    # Search parameters
    search_window: Tuple[float, float] = (-1.0, 1.0)  # Energy window around E_F
    k_mesh_fine: Tuple[int, int, int] = (50, 50, 50)  # Fine mesh for search
    gap_threshold: float = 0.001  # eV, threshold for gap closing
    
    # Chirality calculation
    sphere_radius: float = 0.05   # Radius for surface integration
    num_theta: int = 20           # Angular mesh for sphere
    num_phi: int = 20
    
    # Fermi arc calculation
    surface_thickness: int = 10   # Number of layers for surface
    k_mesh_surface: Tuple[int, int] = (100, 100)
    
    # Output
    save_weyl_data: bool = True
    plot_fermi_arcs: bool = True
    plot_chirality: bool = True


@dataclass
class WeylPointData:
    """Detailed data for a Weyl point."""
    k_point: np.ndarray           # Position in fractional coordinates
    energy: float                 # Energy relative to Fermi level
    chirality: int                # +1 or -1
    weyl_type: WeylType
    
    # Velocity tensor (linearized Hamiltonian)
    velocity_tensor: Optional[np.ndarray] = None  # 3x3 matrix
    
    # Band structure near Weyl point
    bands: Optional[np.ndarray] = None  # Band energies
    
    # Quality metrics
    gap_closing: float = 0.0      # Minimum gap at Weyl point
    convergence: float = 0.0      # Convergence metric
    
    def get_tilting_parameter(self) -> float:
        """
        Calculate tilting parameter C_t = |v_⊥| / |v_∥|.
        
        Type I: C_t < 1
        Type II: C_t > 1
        """
        if self.velocity_tensor is None:
            return 0.0
        
        # Diagonalize velocity tensor
        v_parallel = self.velocity_tensor[2, 2]  # Along k_z
        v_perp = np.sqrt(self.velocity_tensor[0, 0]**2 + 
                        self.velocity_tensor[1, 1]**2)
        
        if abs(v_parallel) > 1e-10:
            return v_perp / abs(v_parallel)
        return float('inf')
    
    def classify_weyl_type(self) -> WeylType:
        """Classify as Type I or Type II Weyl point."""
        C_t = self.get_tilting_parameter()
        
        if C_t < 0.9:
            return WeylType.TYPE_I
        elif C_t > 1.1:
            return WeylType.TYPE_II
        else:
            return WeylType.MULTIPOLE


@dataclass
class ChiralityResult:
    """Result of chirality calculation."""
    chirality: int              # +1 or -1
    berry_flux: float           # Berry phase flux through sphere
    winding_number: int         # Winding number
    
    # Surface integration details
    theta_mesh: np.ndarray
    phi_mesh: np.ndarray
    berry_curvature_surface: np.ndarray
    
    def is_consistent(self) -> bool:
        """Check if chirality is consistent with Berry flux."""
        expected_flux = 2 * np.pi * self.chirality
        return abs(self.berry_flux - expected_flux) < 0.1


@dataclass
class FermiArcData:
    """Data for a Fermi arc."""
    weyl_start: WeylPointData   # Starting Weyl point
    weyl_end: WeylPointData     # Ending Weyl point (opposite chirality)
    
    # Surface k-points along arc
    k_surface: np.ndarray       # Shape: (n, 2) for surface BZ
    energies: np.ndarray        # Shape: (n,)
    spectral_weight: np.ndarray # Surface state weight
    
    arc_type: FermiArcType = FermiArcType.SINGLE
    length: float = 0.0         # Arc length in k-space
    
    def get_arc_direction(self) -> np.ndarray:
        """Get direction vector along the arc."""
        if len(self.k_surface) < 2:
            return np.array([0, 0])
        direction = self.k_surface[-1] - self.k_surface[0]
        return direction / np.linalg.norm(direction)


@dataclass
class MagnetotransportResult:
    """Result of magnetotransport calculations."""
    # Chiral anomaly contribution
    sigma_chiral: float         # Chiral anomaly conductivity
    
    # Negative magnetoresistance
    nmr_ratio: float            # Negative magnetoresistance ratio
    
    # Field angle dependence
    theta_angles: np.ndarray    # Polar angles
    phi_angles: np.ndarray      # Azimuthal angles
    conductivity_tensor: np.ndarray  # σ(B, θ, φ)
    
    def get_negative_magnetoresistance(self, B_field: float) -> float:
        """Calculate negative magnetoresistance at given field."""
        # Δρ/ρ ∝ -B² for chiral anomaly
        return -self.nmr_ratio * B_field**2


class WeylPointLocator:
    """
    Locate Weyl points in the Brillouin zone.
    
    Uses gap closing detection and Berry curvature analysis.
    """
    
    def __init__(
        self,
        calc_dir: str = ".",
        config: Optional[WeylSemimetalConfig] = None,
    ):
        """
        Initialize Weyl point locator.
        
        Args:
            calc_dir: Calculation directory with VASP/Wannier90 output
            config: Weyl semimetal configuration
        """
        self.calc_dir = Path(calc_dir)
        self.config = config or WeylSemimetalConfig()
        self.hamiltonian_data = None
    
    def search_weyl_points(
        self,
        k_mesh: Optional[Tuple[int, int, int]] = None,
        energy_window: Optional[Tuple[float, float]] = None,
    ) -> List[WeylPointData]:
        """
        Search for Weyl points in the BZ.
        
        Args:
            k_mesh: k-point mesh for search
            energy_window: Energy window around E_F
            
        Returns:
            List of Weyl points
        """
        if k_mesh is None:
            k_mesh = self.config.k_mesh_fine
        
        if energy_window is None:
            energy_window = self.config.search_window
        
        # Generate initial k-mesh
        k_points = self._generate_k_mesh(k_mesh)
        
        # Find gap closing points
        gap_closing_points = self._find_gap_closings(k_points, energy_window)
        
        # Refine Weyl point positions
        weyl_points = []
        for k_init in gap_closing_points:
            weyl = self._refine_weyl_point(k_init, energy_window)
            if weyl:
                weyl_points.append(weyl)
        
        # Remove duplicates
        weyl_points = self._remove_duplicates(weyl_points)
        
        # Calculate chirality for each Weyl point
        for weyl in weyl_points:
            chirality_result = self._calculate_chirality(weyl)
            weyl.chirality = chirality_result.chirality
        
        return weyl_points
    
    def _generate_k_mesh(
        self,
        k_mesh: Tuple[int, int, int],
    ) -> np.ndarray:
        """Generate uniform k-point mesh."""
        kx = np.linspace(0, 1, k_mesh[0], endpoint=False)
        ky = np.linspace(0, 1, k_mesh[1], endpoint=False)
        kz = np.linspace(0, 1, k_mesh[2], endpoint=False)
        
        k_points = []
        for i in kx:
            for j in ky:
                for k in kz:
                    k_points.append([i, j, k])
        
        return np.array(k_points)
    
    def _find_gap_closings(
        self,
        k_points: np.ndarray,
        energy_window: Tuple[float, float],
    ) -> List[np.ndarray]:
        """
        Find points where band gap closes.
        
        Returns initial guesses for Weyl points.
        """
        gap_closing = []
        
        for k in k_points:
            gap, bands = self._calculate_gap(k)
            
            # Check if gap is within energy window and below threshold
            if (energy_window[0] <= bands[0] <= energy_window[1] and
                gap < self.config.gap_threshold):
                gap_closing.append(k)
        
        return gap_closing
    
    def _calculate_gap(self, k_point: np.ndarray) -> Tuple[float, np.ndarray]:
        """Calculate band gap at a k-point."""
        # Get eigenvalues at k
        bands = self._get_eigenvalues(k_point)
        
        # Find gap at Fermi level
        occupied = bands[bands < 0]
        unoccupied = bands[bands >= 0]
        
        if len(occupied) > 0 and len(unoccupied) > 0:
            gap = np.min(unoccupied) - np.max(occupied)
        else:
            gap = float('inf')
        
        return gap, bands
    
    def _get_eigenvalues(self, k_point: np.ndarray) -> np.ndarray:
        """Get eigenvalues at a k-point from Hamiltonian."""
        # Placeholder - would use Wannier Hamiltonian or VASP
        return np.array([-1.0, 1.0])
    
    def _refine_weyl_point(
        self,
        k_init: np.ndarray,
        energy_window: Tuple[float, float],
    ) -> Optional[WeylPointData]:
        """
        Refine Weyl point position using optimization.
        
        Minimize gap near initial guess.
        """
        def gap_function(k):
            gap, _ = self._calculate_gap(k)
            return gap
        
        # Use bounded minimization
        result = minimize(
            gap_function,
            k_init,
            method='Nelder-Mead',
            options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000},
        )
        
        if result.success:
            k_weyl = result.x % 1.0  # Fold into BZ
            gap, bands = self._calculate_gap(k_weyl)
            
            # Get velocity tensor
            velocity_tensor = self._calculate_velocity_tensor(k_weyl)
            
            return WeylPointData(
                k_point=k_weyl,
                energy=bands[0] if len(bands) > 0 else 0.0,
                chirality=0,  # Will be calculated later
                weyl_type=WeylType.TYPE_I,
                velocity_tensor=velocity_tensor,
                bands=bands,
                gap_closing=gap,
                convergence=result.fun,
            )
        
        return None
    
    def _calculate_velocity_tensor(self, k_point: np.ndarray) -> np.ndarray:
        """Calculate velocity tensor at a Weyl point."""
        dk = 0.001
        
        # Calculate derivatives of Hamiltonian
        velocity_tensor = np.zeros((3, 3))
        
        for i in range(3):
            dk_vec = np.zeros(3)
            dk_vec[i] = dk
            
            H_plus = self._get_hamiltonian(k_point + dk_vec)
            H_minus = self._get_hamiltonian(k_point - dk_vec)
            
            # dH/dk
            dH = (H_plus - H_minus) / (2 * dk)
            
            # Velocity tensor elements
            velocity_tensor[i, i] = np.trace(dH).real / dH.shape[0]
        
        return velocity_tensor
    
    def _get_hamiltonian(self, k_point: np.ndarray) -> np.ndarray:
        """Get Hamiltonian matrix at k-point."""
        # Placeholder
        return np.eye(2)
    
    def _remove_duplicates(
        self,
        weyl_points: List[WeylPointData],
        tolerance: float = 0.05,
    ) -> List[WeylPointData]:
        """Remove duplicate Weyl points."""
        unique = []
        
        for wp in weyl_points:
            is_duplicate = False
            for existing in unique:
                dist = np.linalg.norm(wp.k_point - existing.k_point)
                if dist < tolerance:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(wp)
        
        return unique
    
    def _calculate_chirality(self, weyl: WeylPointData) -> ChiralityResult:
        """Calculate chirality using Berry curvature integration."""
        calculator = ChiralityCalculator(self.calc_dir, self.config)
        return calculator.calculate_chirality(weyl)


class ChiralityCalculator:
    """
    Calculate chirality of Weyl points using Berry curvature.
    """
    
    def __init__(
        self,
        calc_dir: str = ".",
        config: Optional[WeylSemimetalConfig] = None,
    ):
        """
        Initialize chirality calculator.
        
        Args:
            calc_dir: Calculation directory
            config: Configuration
        """
        self.calc_dir = Path(calc_dir)
        self.config = config or WeylSemimetalConfig()
    
    def calculate_chirality(self, weyl: WeylPointData) -> ChiralityResult:
        """
        Calculate chirality of a Weyl point.
        
        Uses surface integration of Berry curvature on a sphere
        surrounding the Weyl point:
        C = (1/2π) ∮_S Ω · dS
        """
        # Generate mesh on sphere
        theta = np.linspace(0, np.pi, self.config.num_theta)
        phi = np.linspace(0, 2*np.pi, self.config.num_phi)
        THETA, PHI = np.meshgrid(theta, phi)
        
        # Calculate Berry curvature on sphere
        berry_on_sphere = np.zeros_like(THETA)
        
        for i in range(len(theta)):
            for j in range(len(phi)):
                # Point on sphere
                r = self.config.sphere_radius
                k_point = weyl.k_point + r * np.array([
                    np.sin(theta[i]) * np.cos(phi[j]),
                    np.sin(theta[i]) * np.sin(phi[j]),
                    np.cos(theta[i]),
                ])
                
                # Calculate Berry curvature
                berry = self._calculate_berry_curvature(k_point)
                berry_on_sphere[j, i] = np.dot(berry, self._surface_normal(theta[i], phi[j]))
        
        # Integrate Berry curvature over sphere
        flux = self._integrate_over_sphere(berry_on_sphere, theta, phi)
        
        # Chirality is flux / (2π)
        chirality = int(round(flux / (2 * np.pi)))
        
        return ChiralityResult(
            chirality=chirality,
            berry_flux=flux,
            winding_number=chirality,
            theta_mesh=theta,
            phi_mesh=phi,
            berry_curvature_surface=berry_on_sphere,
        )
    
    def _calculate_berry_curvature(self, k_point: np.ndarray) -> np.ndarray:
        """Calculate Berry curvature at a k-point."""
        # Placeholder - real implementation would use Hamiltonian
        return np.zeros(3)
    
    def _surface_normal(self, theta: float, phi: float) -> np.ndarray:
        """Get surface normal vector on sphere."""
        return np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ])
    
    def _integrate_over_sphere(
        self,
        f: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray,
    ) -> float:
        """Integrate function over sphere surface."""
        r = self.config.sphere_radius
        
        # dS = r² sin(θ) dθ dφ
        integrand = f * r**2 * np.sin(THETA := np.meshgrid(theta, phi)[0])
        
        # Trapezoidal integration
        dtheta = theta[1] - theta[0] if len(theta) > 1 else 1
        dphi = phi[1] - phi[0] if len(phi) > 1 else 1
        
        integral = np.trapz(np.trapz(integrand, theta, axis=1), phi)
        
        return integral


class FermiArcCalculator:
    """
    Calculate Fermi arc surface states connecting Weyl points.
    """
    
    def __init__(
        self,
        calc_dir: str = ".",
        config: Optional[WeylSemimetalConfig] = None,
    ):
        """
        Initialize Fermi arc calculator.
        
        Args:
            calc_dir: Calculation directory
            config: Configuration
        """
        self.calc_dir = Path(calc_dir)
        self.config = config or WeylSemimetalConfig()
    
    def calculate_fermi_arcs(
        self,
        weyl_points: List[WeylPointData],
        surface_normal: np.ndarray = np.array([0, 0, 1]),
    ) -> List[FermiArcData]:
        """
        Calculate Fermi arcs for given Weyl points.
        
        Args:
            weyl_points: List of Weyl points
            surface_normal: Normal vector to surface
            
        Returns:
            List of Fermi arcs
        """
        # Separate by chirality
        positive = [wp for wp in weyl_points if wp.chirality == 1]
        negative = [wp for wp in weyl_points if wp.chirality == -1]
        
        fermi_arcs = []
        
        # Pair up Weyl points with opposite chirality
        for p in positive:
            # Find closest negative Weyl point
            if negative:
                distances = [np.linalg.norm(p.k_point - n.k_point) for n in negative]
                closest_idx = np.argmin(distances)
                n = negative[closest_idx]
                
                # Calculate Fermi arc between them
                arc = self._calculate_single_arc(p, n, surface_normal)
                fermi_arcs.append(arc)
        
        return fermi_arcs
    
    def _calculate_single_arc(
        self,
        weyl_p: WeylPointData,
        weyl_n: WeylPointData,
        surface_normal: np.ndarray,
    ) -> FermiArcData:
        """Calculate single Fermi arc between two Weyl points."""
        # Project Weyl points onto surface
        k_p = self._project_to_surface(weyl_p.k_point, surface_normal)
        k_n = self._project_to_surface(weyl_n.k_point, surface_normal)
        
        # Generate k-points along arc
        num_points = 100
        k_arc = np.linspace(k_p, k_n, num_points)
        
        # Calculate surface state energies and weights
        energies = np.zeros(num_points)
        weights = np.zeros(num_points)
        
        for i, k in enumerate(k_arc):
            energies[i], weights[i] = self._calculate_surface_state(k, surface_normal)
        
        # Calculate arc length
        length = np.sum(np.linalg.norm(np.diff(k_arc, axis=0), axis=1))
        
        return FermiArcData(
            weyl_start=weyl_p,
            weyl_end=weyl_n,
            k_surface=k_arc,
            energies=energies,
            spectral_weight=weights,
            length=length,
        )
    
    def _project_to_surface(
        self,
        k_point: np.ndarray,
        surface_normal: np.ndarray,
    ) -> np.ndarray:
        """Project 3D k-point onto 2D surface BZ."""
        # Simplified: assume surface normal is z
        if np.allclose(surface_normal, [0, 0, 1]):
            return k_point[:2]
        else:
            # General projection
            return k_point[:2]  # Simplified
    
    def _calculate_surface_state(
        self,
        k_surface: np.ndarray,
        surface_normal: np.ndarray,
    ) -> Tuple[float, float]:
        """Calculate surface state at a surface k-point."""
        # Placeholder - real implementation would use slab Hamiltonian
        return 0.0, 0.0
    
    def plot_fermi_arcs(
        self,
        fermi_arcs: List[FermiArcData],
        weyl_points: List[WeylPointData],
        save_path: Optional[str] = None,
    ):
        """
        Plot Fermi arcs in surface Brillouin zone.
        
        Args:
            fermi_arcs: List of Fermi arcs
            weyl_points: List of Weyl points
            save_path: Path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("Matplotlib required for plotting")
            return
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot Fermi arcs
        for arc in fermi_arcs:
            ax.plot(arc.k_surface[:, 0], arc.k_surface[:, 1], 
                   'b-', linewidth=2, label='Fermi Arc')
        
        # Plot Weyl points
        for wp in weyl_points:
            color = 'red' if wp.chirality == 1 else 'blue'
            marker = '+' if wp.chirality == 1 else '_'
            ax.scatter(wp.k_point[0], wp.k_point[1], 
                      c=color, marker=marker, s=200, zorder=5)
        
        ax.set_xlabel('$k_x$ (2π/a)')
        ax.set_ylabel('$k_y$ (2π/a)')
        ax.set_title('Fermi Arcs in Surface BZ')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


class MagnetotransportCalculator:
    """
    Calculate magnetotransport properties of Weyl semimetals.
    """
    
    def __init__(
        self,
        weyl_points: List[WeylPointData],
        config: Optional[WeylSemimetalConfig] = None,
    ):
        """
        Initialize magnetotransport calculator.
        
        Args:
            weyl_points: List of Weyl points
            config: Configuration
        """
        self.weyl_points = weyl_points
        self.config = config or WeylSemimetalConfig()
    
    def calculate_chiral_anomaly(
        self,
        B_field: np.ndarray,
        temperature: float = 0.0,
    ) -> float:
        """
        Calculate chiral anomaly contribution to conductivity.
        
        σ_chiral ∝ (e³B/4π²ℏ²) Σ_i C_i μ_i
        
        where C_i is chirality and μ_i is chemical potential difference.
        
        Args:
            B_field: Magnetic field vector (Tesla)
            temperature: Temperature (K)
            
        Returns:
            Chiral anomaly conductivity
        """
        e = 1.602e-19
        hbar = 1.055e-34
        
        # Simplified calculation
        B_mag = np.linalg.norm(B_field)
        
        # Sum over Weyl points
        chiral_sum = sum(wp.chirality for wp in self.weyl_points)
        
        # Chiral anomaly conductivity (simplified formula)
        sigma = (e**3 * B_mag / (4 * np.pi**2 * hbar**2)) * chiral_sum
        
        return sigma * 1e-4  # Convert to S/cm
    
    def calculate_negative_magnetoresistance(
        self,
        B_fields: np.ndarray,
        B_angle: float = 0.0,
    ) -> np.ndarray:
        """
        Calculate negative magnetoresistance.
        
        Δρ/ρ = -a B_∥² (for parallel field)
        
        Args:
            B_fields: Array of magnetic field magnitudes
            B_angle: Angle between B and current
            
        Returns:
            Magnetoresistance ratio Δρ/ρ
        """
        # Simplified: quadratic dependence on parallel field
        B_parallel = B_fields * np.cos(B_angle)
        
        # Adler-Bell-Jackiw anomaly contribution
        a = 0.01  # Material-dependent coefficient
        nmr = -a * B_parallel**2
        
        return nmr
    
    def calculate_full_transport(
        self,
        B_fields: np.ndarray,
        theta_angles: np.ndarray,
        phi_angles: np.ndarray,
    ) -> MagnetotransportResult:
        """
        Calculate full magnetotransport tensor.
        
        Args:
            B_fields: Magnetic field magnitudes
            theta_angles: Polar angles
            phi_angles: Azimuthal angles
            
        Returns:
            Magnetotransport results
        """
        # Initialize conductivity tensor
        n_theta, n_phi = len(theta_angles), len(phi_angles)
        sigma = np.zeros((n_theta, n_phi, 3, 3))
        
        for i, theta in enumerate(theta_angles):
            for j, phi in enumerate(phi_angles):
                B = np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ])
                
                # Calculate conductivity for this field direction
                sigma[i, j] = self._calculate_conductivity_tensor(B)
        
        # Chiral anomaly contribution at specific field
        sigma_chiral = self.calculate_chiral_anomaly(np.array([0, 0, 1]))
        
        # Negative magnetoresistance ratio
        nmr = 0.1  # 10% at 1 Tesla (example)
        
        return MagnetotransportResult(
            sigma_chiral=sigma_chiral,
            nmr_ratio=nmr,
            theta_angles=theta_angles,
            phi_angles=phi_angles,
            conductivity_tensor=sigma,
        )
    
    def _calculate_conductivity_tensor(self, B_field: np.ndarray) -> np.ndarray:
        """Calculate conductivity tensor for given B field."""
        # Placeholder - would use Boltzmann transport
        sigma = np.eye(3) * 1000  # Base conductivity in S/cm
        
        # Add magnetoresistance effects
        B_mag = np.linalg.norm(B_field)
        if B_mag > 0:
            # Weak-field magnetoresistance
            sigma *= (1 - 0.01 * B_mag**2)
        
        return sigma


# Convenience functions

def locate_weyl_points(
    calc_dir: str,
    k_mesh: Tuple[int, int, int] = (50, 50, 50),
) -> List[WeylPointData]:
    """
    Locate Weyl points in a material.
    
    Args:
        calc_dir: Calculation directory
        k_mesh: k-point mesh for search
        
    Returns:
        List of Weyl points
    """
    config = WeylSemimetalConfig(k_mesh_fine=k_mesh)
    locator = WeylPointLocator(calc_dir, config)
    return locator.search_weyl_points(k_mesh)


def calculate_fermi_arcs(
    weyl_points: List[WeylPointData],
    calc_dir: str,
) -> List[FermiArcData]:
    """
    Calculate Fermi arcs from Weyl points.
    
    Args:
        weyl_points: List of Weyl points
        calc_dir: Calculation directory
        
    Returns:
        List of Fermi arcs
    """
    config = WeylSemimetalConfig()
    calculator = FermiArcCalculator(calc_dir, config)
    return calculator.calculate_fermi_arcs(weyl_points)


def analyze_weyl_semimetal(
    calc_dir: str,
) -> Dict[str, Any]:
    """
    Complete analysis of a Weyl semimetal.
    
    Args:
        calc_dir: Calculation directory
        
    Returns:
        Dictionary with complete analysis results
    """
    results = {
        "calc_dir": calc_dir,
        "weyl_points": [],
        "fermi_arcs": [],
        "chiral_anomaly": None,
    }
    
    # Locate Weyl points
    weyl_points = locate_weyl_points(calc_dir)
    results["weyl_points"] = weyl_points
    results["num_weyl_points"] = len(weyl_points)
    
    # Count by chirality
    num_pos = sum(1 for wp in weyl_points if wp.chirality == 1)
    num_neg = sum(1 for wp in weyl_points if wp.chirality == -1)
    results["num_positive"] = num_pos
    results["num_negative"] = num_neg
    results["total_charge"] = num_pos - num_neg
    
    # Calculate Fermi arcs
    if len(weyl_points) >= 2:
        fermi_arcs = calculate_fermi_arcs(weyl_points, calc_dir)
        results["fermi_arcs"] = fermi_arcs
        results["num_fermi_arcs"] = len(fermi_arcs)
    
    # Magnetotransport
    if weyl_points:
        transport_calc = MagnetotransportCalculator(weyl_points)
        chiral_sigma = transport_calc.calculate_chiral_anomaly(np.array([0, 0, 1]))
        results["chiral_anomaly_sigma"] = chiral_sigma
    
    return results


# Example usage
if __name__ == "__main__":
    print("Weyl Semimetal Analysis Module")
    print("=" * 40)
    
    print("\nFeatures:")
    print("- Weyl point location and classification")
    print("- Chirality calculation via Berry curvature")
    print("- Fermi arc surface states")
    print("- Magnetotransport properties")
    print("- Chiral anomaly effects")
    
    print("\nExample materials:")
    print("- TaAs (Type-I Weyl semimetal)")
    print("- MoTe2 (Type-II Weyl semimetal)")
    print("- YbMnBi2 (Magnetic Weyl semimetal)")
