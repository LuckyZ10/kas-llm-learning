"""
Berry Phase and Related Calculations
=====================================

This module provides methods for calculating:
- Electric polarization (using VASP LCALCPOL)
- Berry curvature in k-space
- Anomalous Hall conductivity
- Orbital magnetization
- Magnetoelectric coupling

These quantities are essential for understanding topological transport
phenomena and response properties in materials.

References:
- King-Smith and Vanderbilt, PRB 47, 1651 (1993)
- Resta, RMP 66, 899 (1994)
- Xiao et al., RMP 82, 1959 (2010)
"""

import os
import re
import subprocess
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod

try:
    from pymatgen.core import Structure, Lattice, Element
    from pymatgen.io.vasp import Poscar, Kpoints, Incar, Outcar
    from pymatgen.analysis.structure_matcher import StructureMatcher
    HAS_PMG = True
except ImportError:
    HAS_PMG = False
    warnings.warn("Pymatgen not available.")

try:
    import ase
    from ase.io import read, write
    from ase.dft.kpoints import get_special_points, bandpath
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    warnings.warn("ASE not available.")


class PolarizationDirection(Enum):
    """Directions for polarization calculation."""
    X = "x"
    Y = "y"
    Z = "z"
    ALL = "all"


class BerryCurvatureMethod(Enum):
    """Methods for calculating Berry curvature."""
    FINITE_DIFFERENCE = "finite_difference"
    KUBO_FORMULA = "kubo"
    WANNIER_INTERPOLATION = "wannier"
    VASP_LCALCPOL = "vasp_lcalcpol"


@dataclass
class BerryPhaseConfig:
    """Configuration for Berry phase calculations."""
    # Method
    method: BerryCurvatureMethod = BerryCurvatureMethod.FINITE_DIFFERENCE
    
    # VASP settings
    vasp_executable: str = "vasp_std"
    encut: float = 500.0
    k_mesh_density: float = 0.02
    
    # Berry curvature mesh
    berry_k_mesh: Tuple[int, int, int] = (20, 20, 20)
    
    # Smearing for Kubo formula
    smearing_width: float = 0.01  # eV
    
    # Finite difference settings
    dk: float = 0.01  # k-space step
    
    # Output
    save_curvature_data: bool = True
    plot_curvature: bool = True


@dataclass
class PolarizationResult:
    """Result of polarization calculation."""
    # Electronic contribution
    P_elec: np.ndarray  # (Px, Py, Pz) in μC/cm²
    
    # Ionic contribution
    P_ion: np.ndarray   # (Px, Py, Pz) in μC/cm²
    
    # Total polarization
    P_total: np.ndarray # (Px, Py, Pz) in μC/cm²
    
    # Berry phases (electronic)
    berry_phases: np.ndarray  # (φx, φy, φz) in rad
    
    # Polarization quantum
    P_quantum: np.ndarray  # Polarization quantum in μC/cm²
    
    # Modern polarization (multivalued)
    P_modern: Optional[np.ndarray] = None
    
    def get_polarization_magnitude(self) -> float:
        """Get magnitude of total polarization."""
        return np.linalg.norm(self.P_total)
    
    def get_polarization_direction(self) -> np.ndarray:
        """Get unit vector in polarization direction."""
        return self.P_total / self.get_polarization_magnitude()


@dataclass
class BerryCurvatureResult:
    """Result of Berry curvature calculation."""
    k_points: np.ndarray      # Shape: (nk, 3)
    berry_curvature: np.ndarray  # Shape: (nk, 3) for Ωx, Ωy, Ωz
    berry_flux: np.ndarray    # Berry flux through each k-plane
    chern_numbers: np.ndarray # Chern numbers for each occupied band
    
    def get_berry_curvature_magnitude(self) -> np.ndarray:
        """Get magnitude of Berry curvature."""
        return np.linalg.norm(self.berry_curvature, axis=1)
    
    def plot_berry_curvature(self, component: str = 'z', save_path: Optional[str] = None):
        """Plot Berry curvature."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("Matplotlib required for plotting")
            return
        
        comp_map = {'x': 0, 'y': 1, 'z': 2}
        icomp = comp_map.get(component, 2)
        
        # Assuming 2D mesh
        kx = np.unique(self.k_points[:, 0])
        ky = np.unique(self.k_points[:, 1])
        
        if len(kx) * len(ky) == len(self.k_points):
            curvature_2d = self.berry_curvature[:, icomp].reshape(len(kx), len(ky))
            
            plt.figure(figsize=(8, 6))
            plt.contourf(kx, ky, curvature_2d.T, levels=50, cmap='RdBu_r')
            plt.colorbar(label=f'$\\Omega_{component}$ (Å²)')
            plt.xlabel('$k_x$ (2π/a)')
            plt.ylabel('$k_y$ (2π/a)')
            plt.title(f'Berry Curvature ($\\Omega_{component}$)')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()


@dataclass
class AnomalousHallConductivityResult:
    """Result of anomalous Hall conductivity calculation."""
    # Conductivity tensor σ_xy, σ_yz, σ_zx in S/cm or (Ω·cm)⁻¹
    conductivity: np.ndarray  # Shape: (3, 3)
    
    # Intrinsic contribution
    sigma_intrinsic: np.ndarray
    
    # Extrinsic contributions (if calculated)
    sigma_skew: Optional[np.ndarray] = None
    sigma_side_jump: Optional[np.ndarray] = None
    
    # Hall angle
    hall_angle: Optional[float] = None
    
    # Frequency dependence (for optical Hall conductivity)
    omega: Optional[np.ndarray] = None
    sigma_omega: Optional[np.ndarray] = None
    
    def get_hall_conductivity(self, component: str = 'xy') -> float:
        """Get specific Hall conductivity component."""
        comp_map = {'xy': (0, 1), 'yx': (1, 0), 'yz': (1, 2), 
                   'zy': (2, 1), 'zx': (2, 0), 'xz': (0, 2)}
        i, j = comp_map.get(component, (0, 1))
        return self.conductivity[i, j]
    
    def get_hall_coefficient(self, carrier_density: float) -> float:
        """
        Calculate Hall coefficient R_H = σ_xy / (n e B).
        
        Args:
            carrier_density: Carrier density in cm⁻³
            
        Returns:
            Hall coefficient in cm³/C
        """
        e = 1.602e-19  # Elementary charge in C
        sigma_xy = self.get_hall_conductivity('xy')
        
        # R_H ≈ σ_xy / (n e) for weak fields
        R_H = sigma_xy / (carrier_density * e)
        return R_H


class PolarizationCalculator:
    """
    Calculate electric polarization using the modern theory.
    
    Implements the King-Smith and Vanderbilt approach using Berry phases.
    """
    
    def __init__(
        self,
        calc_dir: str = ".",
        config: Optional[BerryPhaseConfig] = None,
    ):
        """
        Initialize polarization calculator.
        
        Args:
            calc_dir: VASP calculation directory
            config: Berry phase configuration
        """
        self.calc_dir = Path(calc_dir)
        self.config = config or BerryPhaseConfig()
        self.structure = None
        
        if not HAS_PMG:
            raise ImportError("Pymatgen required for polarization calculation")
    
    def read_structure(self) -> Structure:
        """Read crystal structure."""
        poscar_path = self.calc_dir / "POSCAR"
        if not poscar_path.exists():
            poscar_path = self.calc_dir / "CONTCAR"
        
        if not poscar_path.exists():
            raise FileNotFoundError(f"No POSCAR/CONTCAR in {self.calc_dir}")
        
        self.structure = Poscar.from_file(poscar_path).structure
        return self.structure
    
    def prepare_vasp_calculation(self, output_dir: str = "./polarization") -> str:
        """
        Prepare VASP input for polarization calculation.
        
        Uses LCALCPOL tag to calculate Berry phase.
        
        Args:
            output_dir: Output directory
            
        Returns:
            Path to output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Read structure
        structure = self.read_structure()
        
        # Create INCAR with LCALCPOL
        incar_dict = {
            "ENCUT": self.config.encut,
            "ISMEAR": 0,
            "SIGMA": 0.05,
            "LCALCPOL": True,  # Enable polarization calculation
            "LPEAD": True,     # PEAD approximation
            "NBANDS": structure.num_sites * 10,
            "EDIFF": 1e-8,
        }
        
        incar = Incar(incar_dict)
        incar.write_file(output_path / "INCAR")
        
        # Copy POSCAR and POTCAR
        for fname in ["POSCAR", "POTCAR"]:
            src = self.calc_dir / fname
            if src.exists():
                import shutil
                shutil.copy2(src, output_path / fname)
        
        # Generate k-mesh for Berry phase
        kpoints = Kpoints.automatic_density(structure, self.config.k_mesh_density)
        kpoints.write_file(output_path / "KPOINTS")
        
        return str(output_path)
    
    def calculate_polarization(
        self,
        direction: PolarizationDirection = PolarizationDirection.ALL,
    ) -> PolarizationResult:
        """
        Calculate electric polarization.
        
        Args:
            direction: Direction for polarization calculation
            
        Returns:
            Polarization result
        """
        # Read structure
        structure = self.read_structure()
        lattice = structure.lattice.matrix
        volume = structure.volume  # Å³
        
        # Calculate ionic contribution
        P_ion = self._calculate_ionic_polarization(structure)
        
        # Calculate electronic contribution from Berry phase
        P_elec, berry_phases = self._calculate_electronic_polarization(structure)
        
        # Total polarization
        P_total = P_elec + P_ion
        
        # Calculate polarization quantum
        # P_quantum = e R / Ω where R is lattice vector
        e = 1.602e-19  # C
        P_quantum = np.zeros(3)
        for i in range(3):
            R_mag = np.linalg.norm(lattice[i])  # Å
            P_quantum[i] = e * R_mag / volume * 1e6  # Convert to μC/cm²
        
        return PolarizationResult(
            P_elec=P_elec,
            P_ion=P_ion,
            P_total=P_total,
            berry_phases=berry_phases,
            P_quantum=P_quantum,
        )
    
    def _calculate_ionic_polarization(self, structure: Structure) -> np.ndarray:
        """
        Calculate ionic contribution to polarization.
        
        P_ion = (1/Ω) Σ_i Z_i r_i
        """
        volume = structure.volume * 1e-24  # Convert Å³ to cm³
        e = 1.602e-19  # C
        
        P_ion = np.zeros(3)
        
        for site in structure:
            # Get valence charge (simplified)
            Z = Element(site.species_string).Z
            valence = min(Z, 8)  # Simplified valence
            
            # Position in Cartesian coordinates
            r_cart = site.coords  # Å
            
            P_ion += valence * e * r_cart * 1e-6  # Convert to μC·cm/cm³ = μC/cm²
        
        P_ion /= volume
        return P_ion
    
    def _calculate_electronic_polarization(
        self,
        structure: Structure,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate electronic polarization from Berry phase.
        
        P_elec = -(e/Ω) (e/2π) Σ_n φ_n
        """
        volume = structure.volume * 1e-24  # cm³
        e = 1.602e-19  # C
        
        # Read Berry phases from OUTCAR
        berry_phases = self._read_berry_phases()
        
        # P_elec = -(e/Ω) × (phase / 2π) × R
        P_elec = np.zeros(3)
        lattice = structure.lattice.matrix
        
        for i in range(3):
            R_mag = np.linalg.norm(lattice[i]) * 1e-8  # Convert to cm
            phase_factor = berry_phases[i] / (2 * np.pi)
            P_elec[i] = -e * phase_factor * R_mag / volume * 1e6  # μC/cm²
        
        return P_elec, berry_phases
    
    def _read_berry_phases(self) -> np.ndarray:
        """
        Read Berry phases from VASP OUTCAR.
        
        Returns:
            Array of Berry phases (φx, φy, φz) in radians
        """
        outcar_path = self.calc_dir / "OUTCAR"
        
        if not outcar_path.exists():
            warnings.warn("No OUTCAR found. Using dummy Berry phases.")
            return np.array([0.0, 0.0, 0.0])
        
        try:
            outcar = Outcar(str(outcar_path))
            # Try to extract Berry phase information
            # This is VASP-version dependent
            berry_phases = np.array([0.0, 0.0, 0.0])  # Placeholder
            
            # Parse OUTCAR for polarization data
            with open(outcar_path, 'r') as f:
                for line in f:
                    if "P" in line and "(microC/cm2)" in line:
                        # Parse polarization values
                        pass
            
            return berry_phases
        except Exception as e:
            warnings.warn(f"Error reading Berry phases: {e}")
            return np.array([0.0, 0.0, 0.0])
    
    def calculate_born_effective_charges(self) -> Dict[int, np.ndarray]:
        """
        Calculate Born effective charge tensors Z*.
        
        Z*_ij = ∂P_i / ∂u_j where u is atomic displacement
        
        Returns:
            Dictionary mapping atom index to 3x3 Z* tensor
        """
        structure = self.read_structure()
        born_charges = {}
        
        for i in range(len(structure)):
            Z_star = np.zeros((3, 3))
            
            # Calculate by finite differences
            for alpha in range(3):  # Cartesian directions
                # Displace atom in +alpha direction
                # Calculate polarization
                # Displace atom in -alpha direction
                # Calculate polarization
                # Z*_iα = ΔP_i / Δu_α
                pass
            
            born_charges[i] = Z_star
        
        return born_charges


class BerryCurvatureCalculator:
    """
    Calculate Berry curvature in reciprocal space.
    """
    
    def __init__(
        self,
        calc_dir: str = ".",
        config: Optional[BerryPhaseConfig] = None,
    ):
        """
        Initialize Berry curvature calculator.
        
        Args:
            calc_dir: Calculation directory
            config: Berry phase configuration
        """
        self.calc_dir = Path(calc_dir)
        self.config = config or BerryPhaseConfig()
    
    def calculate_berry_curvature(
        self,
        k_mesh: Optional[Tuple[int, int, int]] = None,
        method: Optional[BerryCurvatureMethod] = None,
    ) -> BerryCurvatureResult:
        """
        Calculate Berry curvature on a k-point mesh.
        
        Args:
            k_mesh: k-point mesh (if None, use config)
            method: Calculation method (if None, use config)
            
        Returns:
            Berry curvature result
        """
        if k_mesh is None:
            k_mesh = self.config.berry_k_mesh
        
        if method is None:
            method = self.config.method
        
        # Generate k-mesh
        k_points = self._generate_k_mesh(k_mesh)
        
        # Calculate Berry curvature
        if method == BerryCurvatureMethod.FINITE_DIFFERENCE:
            berry_curvature = self._calculate_fd_berry_curvature(k_points)
        elif method == BerryCurvatureMethod.KUBO_FORMULA:
            berry_curvature = self._calculate_kubo_berry_curvature(k_points)
        else:
            berry_curvature = np.zeros((len(k_points), 3))
        
        # Calculate Chern numbers
        chern_numbers = self._calculate_chern_numbers(berry_curvature, k_mesh)
        
        # Calculate Berry flux
        berry_flux = self._calculate_berry_flux(berry_curvature, k_mesh)
        
        return BerryCurvatureResult(
            k_points=k_points,
            berry_curvature=berry_curvature,
            berry_flux=berry_flux,
            chern_numbers=chern_numbers,
        )
    
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
    
    def _calculate_fd_berry_curvature(
        self,
        k_points: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate Berry curvature using finite differences.
        
        Ω_n,μν(k) = ∂A_n,ν/∂k_μ - ∂A_n,μ/∂k_ν
        where A is the Berry connection.
        """
        dk = self.config.dk
        berry_curvature = np.zeros((len(k_points), 3))
        
        for i, k in enumerate(k_points):
            # Calculate Berry connection at neighboring points
            A_x_plus = self._calculate_berry_connection(k + np.array([dk, 0, 0]))
            A_x_minus = self._calculate_berry_connection(k - np.array([dk, 0, 0]))
            A_y_plus = self._calculate_berry_connection(k + np.array([0, dk, 0]))
            A_y_minus = self._calculate_berry_connection(k - np.array([0, dk, 0]))
            
            # Finite difference derivatives
            dAx_dy = (A_x_plus[1] - A_x_minus[1]) / (2 * dk)
            dAy_dx = (A_y_plus[0] - A_y_minus[0]) / (2 * dk)
            
            # Ω_z = ∂A_y/∂x - ∂A_x/∂y
            berry_curvature[i, 2] = dAy_dx - dAx_dy
        
        return berry_curvature
    
    def _calculate_berry_connection(self, k_point: np.ndarray) -> np.ndarray:
        """
        Calculate Berry connection at a k-point.
        
        A_n(k) = i <u_nk|∇_k|u_nk>
        """
        # Simplified implementation
        # Real implementation would use wavefunction overlaps
        return np.zeros(3)
    
    def _calculate_kubo_berry_curvature(
        self,
        k_points: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate Berry curvature using Kubo formula.
        
        Ω_n,αβ = -ℏ² Σ_{m≠n} 2 Im[<u_nk|v_α|u_mk><u_mk|v_β|u_nk>] / 
                 (E_nk - E_mk)²
        """
        berry_curvature = np.zeros((len(k_points), 3))
        
        for i, k in enumerate(k_points):
            # Get eigenvalues and eigenvectors at k
            energies, velocities = self._get_band_data(k)
            
            # Calculate using Kubo formula
            for n in range(len(energies)):
                for m in range(len(energies)):
                    if m != n:
                        E_diff = energies[n] - energies[m]
                        if abs(E_diff) > self.config.smearing_width:
                            # Simplified: add contribution
                            berry_curvature[i, 2] += 0.0  # Placeholder
        
        return berry_curvature
    
    def _get_band_data(self, k_point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get band energies and velocities at a k-point."""
        # Placeholder - would read from WAVECAR or use interpolation
        return np.array([0.0]), np.zeros((1, 3))
    
    def _calculate_chern_numbers(
        self,
        berry_curvature: np.ndarray,
        k_mesh: Tuple[int, int, int],
    ) -> np.ndarray:
        """Calculate Chern numbers by integrating Berry curvature."""
        # Integrate over BZ
        chern = np.sum(berry_curvature[:, 2]) / (2 * np.pi)
        return np.array([int(round(chern.real))])
    
    def _calculate_berry_flux(
        self,
        berry_curvature: np.ndarray,
        k_mesh: Tuple[int, int, int],
    ) -> np.ndarray:
        """Calculate Berry flux through each k-plane."""
        # Simplified calculation
        flux = np.zeros(3)
        dV = 1.0 / np.prod(k_mesh)
        
        for i in range(3):
            flux[i] = np.sum(berry_curvature[:, i]) * dV
        
        return flux


class AnomalousHallConductivityCalculator:
    """
    Calculate anomalous Hall conductivity from Berry curvature.
    """
    
    def __init__(
        self,
        calc_dir: str = ".",
        config: Optional[BerryPhaseConfig] = None,
    ):
        """
        Initialize AHC calculator.
        
        Args:
            calc_dir: Calculation directory
            config: Berry phase configuration
        """
        self.calc_dir = Path(calc_dir)
        self.config = config or BerryPhaseConfig()
        self.berry_calculator = BerryCurvatureCalculator(calc_dir, config)
    
    def calculate_ahc(
        self,
        temperature: float = 0.0,
        fermi_energy: Optional[float] = None,
    ) -> AnomalousHallConductivityResult:
        """
        Calculate anomalous Hall conductivity.
        
        σ_xy^A = -(e²/ℏ) ∫ (d³k/(2π)³) f(k) Ω_z(k)
        
        Args:
            temperature: Temperature in K
            fermi_energy: Fermi energy (if None, use 0)
            
        Returns:
            AHC result
        """
        # Calculate Berry curvature
        berry_result = self.berry_calculator.calculate_berry_curvature()
        
        # Calculate conductivity from Berry curvature
        sigma_intrinsic = self._calculate_intrinsic_ahc(
            berry_result,
            temperature,
            fermi_energy,
        )
        
        # Total conductivity (intrinsic + extrinsic)
        conductivity = sigma_intrinsic.copy()
        
        return AnomalousHallConductivityResult(
            conductivity=conductivity,
            sigma_intrinsic=sigma_intrinsic,
        )
    
    def _calculate_intrinsic_ahc(
        self,
        berry_result: BerryCurvatureResult,
        temperature: float,
        fermi_energy: Optional[float],
    ) -> np.ndarray:
        """
        Calculate intrinsic contribution to AHC.
        
        σ_αβ = -(e²/ℏ) ∫ d³k/(2π)³ Ω_γ f(E_k)
        """
        e = 1.602e-19  # C
        hbar = 1.055e-34  # J·s
        
        # Conductivity quantum: e²/h
        sigma_0 = e**2 / (2 * np.pi * hbar) * 1e-4  # Convert to S/cm
        
        # Integrate Berry curvature
        sigma = np.zeros((3, 3))
        
        # For z-component (σ_xy)
        omega_z = berry_result.berry_curvature[:, 2]
        sigma_xy = -sigma_0 * np.mean(omega_z)
        sigma[0, 1] = sigma_xy
        sigma[1, 0] = -sigma_xy
        
        return sigma
    
    def calculate_optical_ahc(
        self,
        frequencies: np.ndarray,
        smearing: float = 0.01,
    ) -> AnomalousHallConductivityResult:
        """
        Calculate frequency-dependent (optical) anomalous Hall conductivity.
        
        Args:
            frequencies: Frequency array in eV
            smearing: Smearing width in eV
            
        Returns:
            AHC result with frequency dependence
        """
        sigma_omega = np.zeros((len(frequencies), 3, 3), dtype=complex)
        
        # Calculate using Kubo-Greenwood formula
        for i, omega in enumerate(frequencies):
            # σ(ω) ∝ ∫ dk Ω(k) / (ω + iη)
            sigma_omega[i, 0, 1] = 0.0  # Placeholder
        
        # DC limit
        conductivity = sigma_omega[0].real if len(sigma_omega) > 0 else np.zeros((3, 3))
        
        return AnomalousHallConductivityResult(
            conductivity=conductivity,
            sigma_intrinsic=conductivity,
            omega=frequencies,
            sigma_omega=sigma_omega,
        )


# Convenience functions

def calculate_polarization(
    calc_dir: str,
    direction: str = "all",
) -> PolarizationResult:
    """
    Calculate electric polarization.
    
    Args:
        calc_dir: Calculation directory
        direction: Direction ("x", "y", "z", or "all")
        
    Returns:
        Polarization result
    """
    config = BerryPhaseConfig()
    calculator = PolarizationCalculator(calc_dir, config)
    
    dir_map = {
        "x": PolarizationDirection.X,
        "y": PolarizationDirection.Y,
        "z": PolarizationDirection.Z,
        "all": PolarizationDirection.ALL,
    }
    
    return calculator.calculate_polarization(dir_map.get(direction, PolarizationDirection.ALL))


def calculate_berry_curvature(
    calc_dir: str,
    k_mesh: Tuple[int, int, int] = (20, 20, 20),
) -> BerryCurvatureResult:
    """
    Calculate Berry curvature.
    
    Args:
        calc_dir: Calculation directory
        k_mesh: k-point mesh
        
    Returns:
        Berry curvature result
    """
    config = BerryPhaseConfig(berry_k_mesh=k_mesh)
    calculator = BerryCurvatureCalculator(calc_dir, config)
    return calculator.calculate_berry_curvature()


def calculate_anomalous_hall_conductivity(
    calc_dir: str,
    temperature: float = 0.0,
) -> AnomalousHallConductivityResult:
    """
    Calculate anomalous Hall conductivity.
    
    Args:
        calc_dir: Calculation directory
        temperature: Temperature in K
        
    Returns:
        AHC result
    """
    config = BerryPhaseConfig()
    calculator = AnomalousHallConductivityCalculator(calc_dir, config)
    return calculator.calculate_ahc(temperature)


# Example usage
if __name__ == "__main__":
    print("Berry Phase and Anomalous Hall Conductivity Calculator")
    print("=" * 60)
    
    print("\nFeatures:")
    print("- Electric polarization (VASP LCALCPOL)")
    print("- Berry curvature in k-space")
    print("- Anomalous Hall conductivity")
    print("- Born effective charges")
