"""
Thermal Properties Calculation Module
======================================

Advanced thermodynamic property calculations from phonon data.

Features:
- Lattice heat capacity vs temperature curves
- Helmholtz free energy F(T)
- Vibrational entropy S(T)
- Thermal expansion coefficient (QHA)
- Debye temperature calculation
- Quasi-Harmonic Approximation (QHA) for finite temperature properties

Author: DFTLammps Phonon Team
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from numpy.polynomial import polynomial as P
from scipy import integrate
from scipy.optimize import minimize_scalar, brentq
from scipy.interpolate import interp1d, CubicSpline
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Phonopy imports
try:
    from phonopy import Phonopy
    from phonopy.qha import QHA
    from phonopy.units import THzToEv, EvToTHz, THzToCm, Kb, THz
    PHONOPY_AVAILABLE = True
except ImportError:
    PHONOPY_AVAILABLE = False

# Pymatgen for EOS fitting
try:
    from pymatgen.analysis.eos import EOS
    PMG_AVAILABLE = True
except ImportError:
    PMG_AVAILABLE = False

logger = logging.getLogger(__name__)

# Physical constants
KB = 1.380649e-23  # Boltzmann constant (J/K)
H = 6.62607015e-34  # Planck constant (J*s)
H_BAR = H / (2 * np.pi)  # Reduced Planck constant (J*s)
NA = 6.02214076e23  # Avogadro's number
EV_TO_J = 1.602176634e-19  # eV to Joules
ANGSTROM_TO_M = 1e-10  # Angstrom to meters


@dataclass
class ThermalConfig:
    """Configuration for thermal property calculations."""
    
    # Temperature range
    t_min: float = 0.0  # K
    t_max: float = 1000.0  # K
    t_step: float = 10.0  # K
    temperatures: Optional[np.ndarray] = None
    
    # QHA parameters
    eos_model: str = 'vinet'  # 'birch_murnaghan', 'murnaghan', 'vinet'
    pressure: float = 0.0  # GPa
    t_max_qha: float = 1000.0  # Maximum temperature for QHA
    
    # Numerical parameters
    cutoff_frequency: float = 1e-3  # THz - ignore modes below this
    integration_method: str = 'tetrahedron'  # 'tetrahedron' or 'gaussian'
    
    # Output
    output_dir: str = "./thermal_output"
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        if self.temperatures is None:
            self.temperatures = np.arange(self.t_min, self.t_max + self.t_step, self.t_step)


@dataclass  
class ThermalResults:
    """Container for thermal property calculation results."""
    
    # Basic thermal properties
    temperatures: np.ndarray = field(default_factory=lambda: np.array([]))
    heat_capacity_v: np.ndarray = field(default_factory=lambda: np.array([]))  # J/mol/K
    heat_capacity_p: Optional[np.ndarray] = None  # J/mol/K (requires QHA)
    entropy: np.ndarray = field(default_factory=lambda: np.array([]))  # J/mol/K
    free_energy: np.ndarray = field(default_factory=lambda: np.array([]))  # kJ/mol
    internal_energy: np.ndarray = field(default_factory=lambda: np.array([]))  # kJ/mol
    
    # QHA properties
    thermal_expansion: Optional[np.ndarray] = None  # 1/K
    bulk_modulus: Optional[np.ndarray] = None  # GPa
    gruneisen_parameter: Optional[np.ndarray] = None
    volume_temperature: Optional[np.ndarray] = None  # Angstrom^3
    lattice_parameters_temp: Optional[Dict[str, np.ndarray]] = None
    
    # Debye model
    debye_temperature: Optional[float] = None  # K
    debye_temperature_temp: Optional[np.ndarray] = None  # K vs T
    
    # Metadata
    n_atoms: int = 0
    formula_unit: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'temperatures': self.temperatures.tolist(),
            'heat_capacity_v_J_mol_K': self.heat_capacity_v.tolist(),
            'heat_capacity_p_J_mol_K': self.heat_capacity_p.tolist() if self.heat_capacity_p is not None else None,
            'entropy_J_mol_K': self.entropy.tolist(),
            'free_energy_kJ_mol': self.free_energy.tolist(),
            'internal_energy_kJ_mol': self.internal_energy.tolist(),
            'thermal_expansion_K': self.thermal_expansion.tolist() if self.thermal_expansion is not None else None,
            'bulk_modulus_GPa': self.bulk_modulus.tolist() if self.bulk_modulus is not None else None,
            'gruneisen_parameter': self.gruneisen_parameter.tolist() if self.gruneisen_parameter is not None else None,
            'debye_temperature_K': self.debye_temperature,
            'n_atoms': self.n_atoms,
            'formula_unit': self.formula_unit
        }
    
    def save(self, filepath: str):
        """Save results to file."""
        ext = Path(filepath).suffix
        if ext == '.json':
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        elif ext == '.npz':
            np.savez_compressed(
                filepath,
                temperatures=self.temperatures,
                heat_capacity_v=self.heat_capacity_v,
                entropy=self.entropy,
                free_energy=self.free_energy,
                internal_energy=self.internal_energy,
                thermal_expansion=self.thermal_expansion,
                bulk_modulus=self.bulk_modulus,
                debye_temperature=self.debye_temperature
            )
        else:
            raise ValueError(f"Unsupported file format: {ext}")


class ThermalPropertyCalculator:
    """
    Advanced thermal property calculator from phonon data.
    
    This class provides methods to calculate various thermodynamic
    properties including heat capacity, entropy, free energy, and
    thermal expansion using the quasi-harmonic approximation.
    """
    
    def __init__(self, config: Optional[ThermalConfig] = None):
        """
        Initialize thermal property calculator.
        
        Args:
            config: Configuration for thermal calculations
        """
        self.config = config or ThermalConfig()
        self.results: Optional[ThermalResults] = None
        
        # Store phonon objects at different volumes for QHA
        self._phonopy_volumes: Dict[float, Phonopy] = {}
        self._energies_volumes: Optional[Tuple[np.ndarray, np.ndarray]] = None
        
        logger.info("Initialized ThermalPropertyCalculator")
    
    def calculate_from_phonopy(
        self,
        phonopy: Phonopy,
        temperatures: Optional[np.ndarray] = None
    ) -> ThermalResults:
        """
        Calculate thermal properties from Phonopy object.
        
        Args:
            phonopy: Phonopy object with force constants set
            temperatures: Temperature array (K)
            
        Returns:
            ThermalResults object
        """
        if not PHONOPY_AVAILABLE:
            raise ImportError("Phonopy is required")
        
        if phonopy.force_constants is None:
            raise ValueError("Force constants not set in Phonopy object")
        
        temps = temperatures or self.config.temperatures
        
        # Run mesh calculation
        mesh = phonopy.mesh_numbers or (20, 20, 20)
        phonopy.run_mesh(mesh)
        
        # Calculate thermal properties
        phonopy.run_thermal_properties(
            t_min=temps.min(),
            t_max=temps.max(),
            t_step=temps[1] - temps[0] if len(temps) > 1 else 10
        )
        
        tp_dict = phonopy.get_thermal_properties_dict()
        
        # Create results
        self.results = ThermalResults(
            temperatures=np.array(tp_dict['temperatures']),
            heat_capacity_v=np.array(tp_dict['heat_capacity']),  # J/mol/K
            entropy=np.array(tp_dict['entropy']),  # J/mol/K
            free_energy=np.array(tp_dict['free_energy']),  # kJ/mol
            internal_energy=np.array(tp_dict['internal_energy']) if 'internal_energy' in tp_dict else None,
            n_atoms=phonopy.unitcell.get_number_of_atoms(),
            formula_unit=''.join(phonopy.unitcell.symbols)
        )
        
        # Calculate internal energy if not provided
        if self.results.internal_energy is None:
            self.results.internal_energy = self._calculate_internal_energy(
                self.results.temperatures,
                self.results.free_energy,
                self.results.entropy
            )
        
        # Calculate Debye temperature
        self.results.debye_temperature = self._calculate_debye_temperature(
            phonopy, self.results
        )
        
        logger.info(f"Calculated thermal properties for {len(temps)} temperatures")
        return self.results
    
    def _calculate_internal_energy(
        self,
        temperatures: np.ndarray,
        free_energy: np.ndarray,
        entropy: np.ndarray
    ) -> np.ndarray:
        """
        Calculate internal energy from free energy and entropy.
        
        U = F + TS
        """
        # Convert units: free_energy in kJ/mol, entropy in J/mol/K
        # Need consistent units
        U = free_energy + temperatures * entropy / 1000  # kJ/mol
        return U
    
    def calculate_heat_capacity_classical(
        self,
        temperatures: np.ndarray,
        n_modes: int
    ) -> np.ndarray:
        """
        Calculate classical (Dulong-Petit) heat capacity.
        
        C_v = N_modes * k_B (per unit cell)
        
        Args:
            temperatures: Temperature array (K)
            n_modes: Number of phonon modes
            
        Returns:
            Classical heat capacity in J/mol/K
        """
        # Per mole
        return np.full_like(temperatures, n_modes * KB * NA, dtype=float)
    
    def calculate_debye_cv(
        self,
        temperatures: np.ndarray,
        debye_temp: float,
        n_atoms: int
    ) -> np.ndarray:
        """
        Calculate heat capacity using Debye model.
        
        Args:
            temperatures: Temperature array (K)
            debye_temp: Debye temperature (K)
            n_atoms: Number of atoms per formula unit
            
        Returns:
            Heat capacity in J/mol/K
        """
        from scipy.special import spence  # Dilogarithm
        
        x = debye_temp / temperatures
        
        # Debye function
        def debye_function(x):
            if x < 1e-10:
                return 1.0 - x**2 / 20 + x**4 / 560
            
            # Numerical integration of x^4 * exp(x) / (exp(x) - 1)^2
            # from 0 to x_D/T
            integrand = lambda t: t**4 * np.exp(t) / (np.exp(t) - 1)**2
            result, _ = integrate.quad(integrand, 0, x)
            return 3 * result / x**3
        
        cv = np.array([9 * n_atoms * NA * KB * (T/debye_temp)**3 * 
                      debye_function(debye_temp/T) for T in temperatures])
        
        return cv
    
    def _calculate_debye_temperature(
        self,
        phonopy: Phonopy,
        thermal_results: ThermalResults,
        method: str = 'entropy_match'
    ) -> float:
        """
        Calculate Debye temperature from phonon data.
        
        Methods:
        - 'entropy_match': Match entropy at high T
        - 'low_t': Low-temperature limit from sound velocity
        - 'cv_fit': Fit Debye model to Cv data
        
        Args:
            phonopy: Phonopy object
            thermal_results: Thermal results
            method: Method for Debye temperature calculation
            
        Returns:
            Debye temperature in K
        """
        n_atoms = phonopy.unitcell.get_number_of_atoms()
        
        if method == 'entropy_match':
            # Match entropy at high temperature
            # At high T, S_phonon ≈ 3N*k_B * (1 + ln(T/θ_D))
            # Use entropy at T = θ_D/2
            temps = thermal_results.temperatures
            entropy = thermal_results.entropy
            
            # Find temperature where S ≈ 3N*k_B * ln(2)
            target_S = 3 * n_atoms * KB * NA * np.log(2)  # J/mol/K
            
            # Interpolate to find T where S = target_S
            if entropy[-1] > target_S:
                interp = interp1d(entropy, temps, kind='cubic')
                T_half = interp(target_S)
                theta_D = 2 * T_half
            else:
                # Extrapolate
                slope = (entropy[-1] - entropy[-5]) / (temps[-1] - temps[-5])
                T_half = temps[-1] + (target_S - entropy[-1]) / slope
                theta_D = 2 * T_half
                
        elif method == 'cv_fit':
            # Fit Debye model to Cv data at low T
            temps = thermal_results.temperatures[thermal_results.temperatures <= 100]
            cv = thermal_results.heat_capacity_v[thermal_results.temperatures <= 100]
            
            def residual(theta_D):
                cv_debye = self.calculate_debye_cv(temps, theta_D, n_atoms)
                return np.sum((cv - cv_debye)**2)
            
            result = minimize_scalar(residual, bounds=(10, 2000), method='bounded')
            theta_D = result.x
            
        elif method == 'low_t':
            # Calculate from sound velocity (requires elastic constants)
            # θ_D = (ħ/k_B) * (6π²N/V)^(1/3) * v_avg
            # This is simplified - real implementation needs elastic constants
            volume = np.linalg.det(phonopy.unitcell.cell)  # Angstrom^3
            
            # Estimate average sound velocity from max phonon frequency
            # v_s ≈ ω_max / q_max
            # This is a rough approximation
            if hasattr(phonopy, 'frequencies') and phonopy.frequencies is not None:
                omega_max = np.max(phonopy.frequencies) * 2 * np.pi * 1e12  # rad/s
                n = (n_atoms / volume)**(1/3) * 1e10  # m^-1
                v_avg = omega_max / n
                
                theta_D = (H_BAR / KB) * (6 * np.pi**2 * n_atoms / volume * 1e-30)**(1/3) * v_avg
            else:
                theta_D = 300.0  # Default guess
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.info(f"Calculated Debye temperature: {theta_D:.1f} K")
        return theta_D
    
    def calculate_gruneisen_parameter(
        self,
        temperatures: np.ndarray,
        volumes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculate Grüneisen parameter.
        
        γ = α_V * B_T * V / C_V
        
        Where:
        - α_V: volumetric thermal expansion coefficient
        - B_T: isothermal bulk modulus
        - V: volume
        - C_V: heat capacity at constant volume
        
        Requires QHA calculation first.
        
        Args:
            temperatures: Temperature array (K)
            volumes: Volume array for different temperatures (optional)
            
        Returns:
            Grüneisen parameter vs temperature
        """
        if self.results is None:
            raise ValueError("Run thermal calculation first")
        
        if self.results.thermal_expansion is None:
            raise ValueError("QHA calculation required for Grüneisen parameter")
        
        alpha = self.results.thermal_expansion  # 1/K
        B_T = self.results.bulk_modulus * 1e9  # Convert GPa to Pa
        V = self.results.volume_temperature * 1e-30  # Convert Å³ to m³
        C_V = self.results.heat_capacity_v / NA  # J/K per unit cell
        
        gamma = alpha * B_T * V / C_V
        
        self.results.gruneisen_parameter = gamma
        return gamma
    
    def run_qha(
        self,
        volumes: np.ndarray,
        electronic_energies: np.ndarray,
        phonopy_objects: Dict[float, Phonopy],
        temperatures: Optional[np.ndarray] = None,
        eos_model: Optional[str] = None
    ) -> ThermalResults:
        """
        Run Quasi-Harmonic Approximation (QHA) calculation.
        
        QHA accounts for thermal expansion by calculating phonon properties
        at different volumes and minimizing the free energy.
        
        Args:
            volumes: Array of volumes (Angstrom^3)
            electronic_energies: Array of electronic total energies (eV)
            phonopy_objects: Dictionary mapping volume to Phonopy object
            temperatures: Temperature array (K)
            eos_model: Equation of state model ('vinet', 'birch_murnaghan', etc.)
            
        Returns:
            ThermalResults with QHA properties
        """
        if not PHONOPY_AVAILABLE:
            raise ImportError("Phonopy is required for QHA")
        
        temps = temperatures or self.config.temperatures
        eos_model = eos_model or self.config.eos_model
        
        # Store for later use
        self._phonopy_volumes = phonopy_objects
        self._energies_volumes = (volumes, electronic_energies)
        
        # Get thermal properties at each volume
        thermal_props = {}
        for vol, ph in phonopy_objects.items():
            thermal_props[vol] = self._get_thermal_properties(ph, temps)
        
        # Calculate total free energy at each volume and temperature
        # F(V,T) = E_elec(V) + F_vib(V,T)
        
        free_energies = np.zeros((len(volumes), len(temps)))
        for i, vol in enumerate(volumes):
            E_elec = electronic_energies[i]
            F_vib = thermal_props[vol]['free_energy']  # kJ/mol
            # Convert to eV per unit cell
            F_vib_ev = F_vib * 1000 / (NA * EV_TO_J)  # kJ/mol to eV
            free_energies[i] = E_elec + F_vib_ev
        
        # Find equilibrium volume at each temperature
        # by minimizing F(V,T)
        V_eq = np.zeros(len(temps))
        for j, T in enumerate(temps):
            # Interpolate F(V) and find minimum
            interp = CubicSpline(volumes, free_energies[:, j])
            
            # Find minimum
            result = minimize_scalar(
                lambda v: interp(v),
                bounds=(volumes.min(), volumes.max()),
                method='bounded'
            )
            V_eq[j] = result.x
        
        # Calculate thermal expansion coefficient
        alpha = np.gradient(V_eq, temps) / V_eq
        
        # Calculate bulk modulus vs T
        B_T = np.zeros(len(temps))
        for j, T in enumerate(temps):
            # B_T = V * d²F/dV²
            interp = CubicSpline(volumes, free_energies[:, j])
            d2F_dV2 = interp.derivative(2)(V_eq[j])
            B_T[j] = V_eq[j] * d2F_dV2 * 160.2177  # Convert to GPa
        
        # Interpolate other properties at equilibrium volume
        cv_eq = np.zeros(len(temps))
        s_eq = np.zeros(len(temps))
        f_eq = np.zeros(len(temps))
        
        for j, T in enumerate(temps):
            v = V_eq[j]
            # Find closest volumes and interpolate
            idx = np.argsort(np.abs(volumes - v))[:4]
            
            cv_vals = [thermal_props[volumes[i]]['heat_capacity'][j] for i in idx]
            s_vals = [thermal_props[volumes[i]]['entropy'][j] for i in idx]
            f_vals = [free_energies[i, j] for i in idx]
            
            cv_eq[j] = np.interp(v, volumes[idx], cv_vals)
            s_eq[j] = np.interp(v, volumes[idx], s_vals)
            f_eq[j] = np.interp(v, volumes[idx], f_vals)
        
        # Create results
        self.results = ThermalResults(
            temperatures=temps,
            heat_capacity_v=cv_eq,
            heat_capacity_p=None,  # Requires additional calculation
            entropy=s_eq,
            free_energy=f_eq * 96.485,  # Convert eV to kJ/mol
            internal_energy=None,
            thermal_expansion=alpha,
            bulk_modulus=B_T,
            volume_temperature=V_eq,
            n_atoms=phonopy_objects[volumes[0]].unitcell.get_number_of_atoms()
        )
        
        logger.info(f"Completed QHA calculation for {len(temps)} temperatures")
        return self.results
    
    def _get_thermal_properties(
        self,
        phonopy: Phonopy,
        temperatures: np.ndarray
    ) -> Dict:
        """Get thermal properties from Phonopy at given temperatures."""
        phonopy.run_thermal_properties(
            t_min=temperatures.min(),
            t_max=temperatures.max(),
            t_step=temperatures[1] - temperatures[0] if len(temperatures) > 1 else 10
        )
        
        tp = phonopy.get_thermal_properties_dict()
        return {
            'temperatures': np.array(tp['temperatures']),
            'heat_capacity': np.array(tp['heat_capacity']),
            'entropy': np.array(tp['entropy']),
            'free_energy': np.array(tp['free_energy'])
        }
    
    def calculate_qha_single_volume(
        self,
        phonopy: Phonopy,
        volumes: np.ndarray,
        energies: np.ndarray,
        temperatures: Optional[np.ndarray] = None
    ) -> ThermalResults:
        """
        Simplified QHA using single phonon calculation.
        
        Uses the approximation that phonon frequencies scale with volume.
        This is less accurate but computationally cheaper than full QHA.
        
        Args:
            phonopy: Phonopy object at reference volume
            volumes: Array of volumes
            energies: Electronic energies at each volume
            temperatures: Temperature array
            
        Returns:
            ThermalResults with approximate QHA properties
        """
        temps = temperatures or self.config.temperatures
        
        # Use Phonopy's built-in QHA
        qha = QHA(
            volumes,
            energies,
            phonopy,  # Reference phonopy
            t_max=temps.max(),
            eos=self.config.eos_model
        )
        
        # Get thermal expansion
        thermal_expansion = qha.get_thermal_expansion()
        
        # Get equilibrium volume vs T
        vol_temp = qha.get_equilibrium_volume()
        
        # Create results
        self.results = ThermalResults(
            temperatures=temps[:len(thermal_expansion)],
            thermal_expansion=thermal_expansion,
            volume_temperature=vol_temp,
            n_atoms=phonopy.unitcell.get_number_of_atoms()
        )
        
        logger.info("Completed single-volume QHA calculation")
        return self.results
    
    def calculate_lattice_parameters_vs_temperature(
        self,
        structure_at_volumes: Dict[float, Any],
        temperatures: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Calculate lattice parameters vs temperature from QHA results.
        
        Args:
            structure_at_volumes: Dictionary mapping volume to structure
            temperatures: Temperature array (uses results if None)
            
        Returns:
            Dictionary with 'a', 'b', 'c', 'alpha', 'beta', 'gamma' arrays
        """
        if self.results is None or self.results.volume_temperature is None:
            raise ValueError("Run QHA calculation first")
        
        temps = temperatures or self.results.temperatures
        V_eq = self.results.volume_temperature
        
        # Interpolate lattice parameters from volume
        volumes = np.array(list(structure_at_volumes.keys()))
        
        # Get lattice parameters at each volume
        params = {'a': [], 'b': [], 'c': [], 'alpha': [], 'beta': [], 'gamma': []}
        
        for vol, struct in structure_at_volumes.items():
            lattice = struct.lattice if hasattr(struct, 'lattice') else struct.cell
            params['a'].append(lattice.a)
            params['b'].append(lattice.b)
            params['c'].append(lattice.c)
            params['alpha'].append(lattice.alpha)
            params['beta'].append(lattice.beta)
            params['gamma'].append(lattice.gamma)
        
        # Interpolate to equilibrium volumes
        result = {}
        for key in params:
            interp = interp1d(volumes, params[key], kind='cubic', fill_value='extrapolate')
            result[key] = interp(V_eq)
        
        self.results.lattice_parameters_temp = result
        return result
    
    def calculate_c_p_from_c_v(
        self,
        temperatures: np.ndarray,
        c_v: np.ndarray,
        thermal_expansion: np.ndarray,
        bulk_modulus: np.ndarray,
        volume: np.ndarray
    ) -> np.ndarray:
        """
        Calculate C_p from C_v using thermodynamic relation.
        
        C_p = C_v + α² * B_T * T * V
        
        Args:
            temperatures: Temperature array (K)
            c_v: Heat capacity at constant volume (J/mol/K)
            thermal_expansion: Volumetric thermal expansion coefficient (1/K)
            bulk_modulus: Isothermal bulk modulus (GPa)
            volume: Volume per mole (m³/mol)
            
        Returns:
            Heat capacity at constant pressure (J/mol/K)
        """
        alpha = thermal_expansion
        B_T = bulk_modulus * 1e9  # GPa to Pa
        V = volume
        T = temperatures
        
        c_p = c_v + alpha**2 * B_T * T * V
        
        return c_p
    
    def fit_debye_model(
        self,
        temperatures: np.ndarray,
        heat_capacity: np.ndarray,
        n_atoms: int
    ) -> Tuple[float, np.ndarray]:
        """
        Fit Debye model to heat capacity data.
        
        Args:
            temperatures: Temperature array (K)
            heat_capacity: Heat capacity data (J/mol/K)
            n_atoms: Number of atoms per formula unit
            
        Returns:
            Tuple of (Debye temperature, fitted heat capacity)
        """
        from scipy.optimize import curve_fit
        
        def debye_cv(T, theta_D):
            return self.calculate_debye_cv(T, theta_D, n_atoms)
        
        # Fit at low temperatures (T < θ_D/2)
        mask = temperatures < 300
        
        popt, _ = curve_fit(debye_cv, temperatures[mask], heat_capacity[mask], 
                           p0=[300], bounds=([10], [2000]))
        
        theta_D = popt[0]
        cv_fitted = debye_cv(temperatures, theta_D)
        
        return theta_D, cv_fitted
    
    def plot_heat_capacity(
        self,
        results: Optional[ThermalResults] = None,
        show_debye: bool = False,
        show_classical: bool = False,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot heat capacity vs temperature.
        
        Args:
            results: ThermalResults (uses self.results if None)
            show_debye: Overlay Debye model fit
            show_classical: Show classical limit
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        results = results or self.results
        if results is None:
            raise ValueError("No results available")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        temps = results.temperatures
        cv = results.heat_capacity_v
        
        # Plot calculated Cv
        ax.plot(temps, cv, 'b-', lw=2, label='Cᵥ (calculated)')
        
        # Plot Cp if available
        if results.heat_capacity_p is not None:
            ax.plot(temps, results.heat_capacity_p, 'r--', lw=2, label='Cₚ (QHA)')
        
        # Show classical limit
        if show_classical:
            n_modes = 3 * results.n_atoms
            c_classical = self.calculate_heat_capacity_classical(temps, n_modes)
            ax.axhline(y=c_classical[0], color='gray', linestyle=':', 
                      lw=1.5, label='Classical limit (Dulong-Petit)')
        
        # Show Debye fit
        if show_debye:
            theta_D, cv_debye = self.fit_debye_model(temps, cv, results.n_atoms)
            ax.plot(temps, cv_debye, 'g--', lw=1.5, 
                   label=f'Debye model (θ_D={theta_D:.1f}K)')
        
        ax.set_xlabel('Temperature (K)', fontsize=12)
        ax.set_ylabel('Heat Capacity (J/mol/K)', fontsize=12)
        ax.set_title('Heat Capacity vs Temperature', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved heat capacity plot to {save_path}")
        
        return fig
    
    def plot_free_energy_entropy(
        self,
        results: Optional[ThermalResults] = None,
        figsize: Tuple[int, int] = (12, 5),
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot free energy and entropy vs temperature.
        
        Args:
            results: ThermalResults
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        results = results or self.results
        if results is None:
            raise ValueError("No results available")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        temps = results.temperatures
        
        # Free energy
        ax1.plot(temps, results.free_energy, 'b-', lw=2)
        ax1.set_xlabel('Temperature (K)', fontsize=12)
        ax1.set_ylabel('Helmholtz Free Energy (kJ/mol)', fontsize=12)
        ax1.set_title('F(T) = U - TS', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Entropy
        ax2.plot(temps, results.entropy, 'r-', lw=2)
        ax2.set_xlabel('Temperature (K)', fontsize=12)
        ax2.set_ylabel('Entropy (J/mol/K)', fontsize=12)
        ax2.set_title('Vibrational Entropy', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved F-S plot to {save_path}")
        
        return fig
    
    def plot_thermal_expansion(
        self,
        results: Optional[ThermalResults] = None,
        figsize: Tuple[int, int] = (12, 5),
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot thermal expansion and related properties.
        
        Args:
            results: ThermalResults with QHA data
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        results = results or self.results
        if results is None or results.thermal_expansion is None:
            raise ValueError("QHA results required")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        temps = results.temperatures
        
        # Volume vs T
        ax = axes[0]
        V0 = results.volume_temperature[0]
        delta_V = (results.volume_temperature - V0) / V0 * 100
        ax.plot(temps, delta_V, 'b-', lw=2)
        ax.set_xlabel('Temperature (K)', fontsize=12)
        ax.set_ylabel('ΔV/V₀ (%)', fontsize=12)
        ax.set_title('Thermal Expansion (Volume)', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Thermal expansion coefficient
        ax = axes[1]
        alpha = results.thermal_expansion * 1e6  # Convert to 10⁻⁶/K
        ax.plot(temps, alpha, 'r-', lw=2)
        ax.set_xlabel('Temperature (K)', fontsize=12)
        ax.set_ylabel('αᵥ (10⁻⁶ K⁻¹)', fontsize=12)
        ax.set_title('Thermal Expansion Coefficient', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved thermal expansion plot to {save_path}")
        
        return fig
    
    def plot_complete_thermal_properties(
        self,
        results: Optional[ThermalResults] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10)
    ) -> Figure:
        """
        Create comprehensive thermal properties visualization.
        
        Args:
            results: ThermalResults
            save_path: Path to save figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        results = results or self.results
        if results is None:
            raise ValueError("No results available")
        
        has_qha = results.thermal_expansion is not None
        
        n_rows = 3 if has_qha else 2
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(n_rows, 2, hspace=0.3, wspace=0.3)
        
        temps = results.temperatures
        
        # Row 1: Cv and Entropy
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(temps, results.heat_capacity_v, 'b-', lw=2)
        if results.heat_capacity_p is not None:
            ax1.plot(temps, results.heat_capacity_p, 'r--', lw=2, label='Cₚ')
        ax1.set_xlabel('T (K)')
        ax1.set_ylabel('Cv (J/mol/K)')
        ax1.set_title('Heat Capacity')
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(temps, results.entropy, 'r-', lw=2)
        ax2.set_xlabel('T (K)')
        ax2.set_ylabel('S (J/mol/K)')
        ax2.set_title('Entropy')
        ax2.grid(True, alpha=0.3)
        
        # Row 2: Free Energy and Internal Energy
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(temps, results.free_energy, 'g-', lw=2)
        ax3.set_xlabel('T (K)')
        ax3.set_ylabel('F (kJ/mol)')
        ax3.set_title('Helmholtz Free Energy')
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(gs[1, 1])
        if results.internal_energy is not None:
            ax4.plot(temps, results.internal_energy, 'm-', lw=2)
            ax4.set_xlabel('T (K)')
            ax4.set_ylabel('U (kJ/mol)')
            ax4.set_title('Internal Energy')
            ax4.grid(True, alpha=0.3)
        
        # Row 3: QHA properties (if available)
        if has_qha:
            ax5 = fig.add_subplot(gs[2, 0])
            V0 = results.volume_temperature[0]
            delta_V = (results.volume_temperature - V0) / V0 * 100
            ax5.plot(temps, delta_V, 'b-', lw=2)
            ax5.set_xlabel('T (K)')
            ax5.set_ylabel('ΔV/V₀ (%)')
            ax5.set_title('Thermal Expansion')
            ax5.grid(True, alpha=0.3)
            
            ax6 = fig.add_subplot(gs[2, 1])
            ax6.plot(temps, results.bulk_modulus, 'r-', lw=2)
            ax6.set_xlabel('T (K)')
            ax6.set_ylabel('B_T (GPa)')
            ax6.set_title('Bulk Modulus')
            ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Thermal Properties Summary', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved complete thermal properties plot to {save_path}")
        
        return fig
    
    def save_results(
        self,
        filepath: str,
        format: str = 'json'
    ):
        """
        Save results to file.
        
        Args:
            filepath: Output file path
            format: 'json' or 'npz'
        """
        if self.results is None:
            raise ValueError("No results to save")
        
        if format == 'json':
            self.results.save(filepath)
        elif format == 'npz':
            base = Path(filepath).with_suffix('.npz')
            self.results.save(str(base))
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Saved results to {filepath}")


class QHACalculator:
    """
    Specialized calculator for Quasi-Harmonic Approximation.
    
    This class handles the full QHA workflow including:
    - Volume-dependent phonon calculations
    - Equation of state fitting
    - Thermal expansion calculation
    """
    
    def __init__(self, config: Optional[ThermalConfig] = None):
        """Initialize QHA calculator."""
        self.config = config or ThermalConfig()
        self.thermal_calc = ThermalPropertyCalculator(config)
    
    def prepare_volume_expansions(
        self,
        structure: Any,
        n_volumes: int = 7,
        volume_range: Tuple[float, float] = (0.9, 1.1)
    ) -> List[Tuple[float, Any]]:
        """
        Prepare structures at different volumes for QHA.
        
        Args:
            structure: Reference structure
            n_volumes: Number of volume points
            volume_range: (min, max) volume scaling factors
            
        Returns:
            List of (volume, structure) tuples
        """
        import copy
        
        # Get reference volume
        if hasattr(structure, 'volume'):
            V0 = structure.volume
        elif hasattr(structure, 'get_volume'):
            V0 = structure.get_volume()
        else:
            # Assume ASE Atoms
            V0 = structure.get_cell().volume
        
        # Generate volume scaling factors
        scales = np.linspace(volume_range[0]**(1/3), 
                            volume_range[1]**(1/3), n_volumes)
        
        structures = []
        for scale in scales:
            new_struct = copy.deepcopy(structure)
            new_struct.scale_lattice(V0 * scale**3)
            structures.append((V0 * scale**3, new_struct))
        
        logger.info(f"Prepared {n_volumes} volume expansions from {V0 * volume_range[0]:.2f} "
                   f"to {V0 * volume_range[1]:.2f} Å³")
        
        return structures
    
    def fit_equation_of_state(
        self,
        volumes: np.ndarray,
        energies: np.ndarray,
        eos_model: Optional[str] = None
    ) -> Dict:
        """
        Fit equation of state to energy-volume data.
        
        Args:
            volumes: Volumes (Angstrom^3)
            energies: Energies (eV)
            eos_model: EOS model name
            
        Returns:
            Dictionary with fitted parameters
        """
        eos_model = eos_model or self.config.eos_model
        
        if PMG_AVAILABLE:
            eos = EOS(eos_model)
            fit = eos.fit(volumes, energies)
            
            return {
                'model': eos_model,
                'e0': fit.e0,  # Equilibrium energy
                'v0': fit.v0,  # Equilibrium volume
                'b0': fit.b0,  # Bulk modulus
                'b1': fit.b1,  # Pressure derivative
                'eos_fit': fit
            }
        else:
            # Simple polynomial fit
            coeffs = np.polyfit(volumes, energies, 3)
            poly = np.poly1d(coeffs)
            
            # Find minimum
            v0 = volumes[np.argmin(energies)]
            
            # Calculate bulk modulus
            d2E_dV2 = np.polyder(poly, 2)(v0)
            b0 = v0 * d2E_dV2 * 160.2177  # Convert to GPa
            
            return {
                'model': 'polynomial',
                'e0': poly(v0),
                'v0': v0,
                'b0': b0,
                'coefficients': coeffs
            }


def calculate_thermal_properties_from_phonopy(
    phonopy: Phonopy,
    temperatures: Optional[np.ndarray] = None,
    run_qha: bool = False,
    volumes: Optional[np.ndarray] = None,
    electronic_energies: Optional[np.ndarray] = None
) -> ThermalResults:
    """
    Convenience function to calculate thermal properties.
    
    Args:
        phonopy: Phonopy object
        temperatures: Temperature array
        run_qha: Whether to run QHA calculation
        volumes: Array of volumes for QHA
        electronic_energies: Electronic energies for QHA
        
    Returns:
        ThermalResults object
    """
    calc = ThermalPropertyCalculator()
    
    if run_qha and volumes is not None and electronic_energies is not None:
        # This requires phonopy objects at each volume
        # Simplified version - user should use full QHA workflow
        logger.warning("Full QHA requires phonopy objects at each volume. "
                      "Use QHACalculator for complete workflow.")
    
    return calc.calculate_from_phonopy(phonopy, temperatures)


def analyze_anharmonicity(
    phonopy: Phonopy,
    temperatures: np.ndarray,
    gruneisen_gamma: Optional[np.ndarray] = None
) -> Dict:
    """
    Analyze anharmonic effects from phonon data.
    
    Args:
        phonopy: Phonopy object
        temperatures: Temperature array
        gruneisen_gamma: Mode Grüneisen parameters (optional)
        
    Returns:
        Dictionary with anharmonicity metrics
    """
    results = {
        'temperatures': temperatures,
        'anharmonic_metrics': {}
    }
    
    # Calculate frequency shifts with temperature (if QHA available)
    # This would require phonon data at different volumes
    
    # Calculate thermal expansion from Grüneisen parameters
    if gruneisen_gamma is not None:
        results['average_gruneisen'] = np.mean(gruneisen_gamma)
        results['gruneisen_temperature_dependence'] = None  # Would need QHA
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Thermal Properties Calculator')
    parser.add_argument('--phonopy', type=str, 
                       help='Path to phonopy_params.yaml or force constants')
    parser.add_argument('--mesh', type=int, nargs=3, default=[20, 20, 20],
                       help='q-point mesh')
    parser.add_argument('--tmin', type=float, default=0.0,
                       help='Minimum temperature')
    parser.add_argument('--tmax', type=float, default=1000.0,
                       help='Maximum temperature')
    parser.add_argument('--tstep', type=float, default=10.0,
                       help='Temperature step')
    parser.add_argument('--outdir', type=str, default='./thermal_output',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("Thermal Properties Calculator - use within Python for full functionality")
    print("Example:")
    print("  from dftlammps.phonon.thermal_properties import ThermalPropertyCalculator")
    print("  calc = ThermalPropertyCalculator()")
    print("  results = calc.calculate_from_phonopy(phonopy_obj)")
