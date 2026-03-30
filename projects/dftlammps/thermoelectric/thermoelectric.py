"""
thermoelectric.py

Thermoelectric Transport Module

This module provides tools for calculating thermoelectric properties
including Seebeck coefficient, electrical and thermal conductivity,
and thermoelectric figure of merit ZT.

References:
- Mahan & Sofo, PNAS 93, 7436 (1996) - Best thermoelectric
- Snyder & Toberer, Nature Mater. 7, 105 (2008) - Complex thermoelectrics
- Madsen & Singh, Comput. Phys. Commun. 175, 67 (2006) - BoltzTraP
"""

import numpy as np
from scipy import integrate, interpolate
from scipy.optimize import minimize_scalar
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings


@dataclass
class TransportCoefficients:
    """
    Transport coefficients for thermoelectric calculations.
    """
    
    # Electrical conductivity (S/m or 1/Ω·m)
    sigma: np.ndarray
    
    # Seebeck coefficient (V/K)
    seebeck: np.ndarray
    
    # Electronic thermal conductivity (W/m·K)
    kappa_e: np.ndarray
    
    # Lattice thermal conductivity (W/m·K)
    kappa_l: Optional[np.ndarray] = None
    
    # Temperatures at which coefficients are evaluated
    temperatures: Optional[np.ndarray] = None
    
    def calculate_zt(self) -> np.ndarray:
        """
        Calculate thermoelectric figure of merit:
        
        ZT = S²σT / κ
        
        where κ = κ_e + κ_l
        """
        if self.temperatures is None:
            raise ValueError("Temperatures must be specified for ZT calculation")
        
        S = np.abs(self.seebeck)
        kappa_total = self.kappa_e
        
        if self.kappa_l is not None:
            kappa_total = kappa_total + self.kappa_l
        
        zt = (S**2 * self.sigma * self.temperatures) / kappa_total
        
        return zt
    
    def calculate_power_factor(self) -> np.ndarray:
        """
        Calculate power factor: PF = S²σ
        """
        return self.seebeck**2 * self.sigma


class SeebeckCalculator:
    """
    Calculator for Seebeck coefficient from electronic structure.
    """
    
    def __init__(self, temperatures: np.ndarray = None):
        self.temperatures = temperatures or np.linspace(100, 800, 20)
        self.kB = 8.617e-5  # Boltzmann constant in eV/K
    
    def calculate_from_transmission(self,
                                   energies: np.ndarray,
                                   transmission: np.ndarray,
                                   fermi_level: float = 0.0,
                                   method: str = "cutler_mott") -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Seebeck coefficient from transmission function.
        
        Methods:
        - "cutler_mott": S = -(π²/3e)(k_B T) (∂lnσ/∂E)|_Ef
        - "mott": Direct Mott formula
        """
        seebeck_values = []
        
        for T in self.temperatures:
            if method == "cutler_mott":
                S = self._cutler_mott_formula(energies, transmission, 
                                              fermi_level, T)
            elif method == "mott":
                S = self._mott_formula(energies, transmission,
                                      fermi_level, T)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            seebeck_values.append(S)
        
        return self.temperatures, np.array(seebeck_values)
    
    def _cutler_mott_formula(self, energies: np.ndarray,
                            transmission: np.ndarray,
                            fermi_level: float,
                            temperature: float) -> float:
        """
        Cutler-Mott formula for Seebeck coefficient.
        
        S = -(π²/3e)(k_B T) (∂lnσ/∂E)|_Ef
        """
        # Calculate conductivity as function of energy
        # σ(E) ∝ T(E)
        
        # Interpolate transmission
        interp = interpolate.interp1d(energies, transmission, 
                                      kind='cubic', fill_value=0)
        
        # Calculate derivative at Fermi level
        dE = 0.001  # eV
        E_plus = fermi_level + dE
        E_minus = fermi_level - dE
        
        if E_plus > energies[-1] or E_minus < energies[0]:
            return 0.0
        
        sigma_plus = interp(E_plus)
        sigma_minus = interp(E_minus)
        
        if sigma_plus > 0 and sigma_minus > 0:
            dln_sigma_dE = (np.log(sigma_plus) - np.log(sigma_minus)) / (2 * dE)
        else:
            dln_sigma_dE = 0
        
        # Seebeck coefficient
        e_charge = 1.602e-19  # C
        kB = 8.617e-5  # eV/K
        
        S = -(np.pi**2 / 3) * (kB * temperature) * dln_sigma_dE / e_charge
        
        return S
    
    def _mott_formula(self, energies: np.ndarray,
                     transmission: np.ndarray,
                     fermi_level: float,
                     temperature: float) -> float:
        """
        Direct Mott formula implementation.
        """
        # Integrate for transport coefficients
        # L_n = ∫ (E - μ)ⁿ T(E) (-∂f/∂E) dE
        
        kB = 8.617e-5
        
        def integrand_L1(E):
            # L1 integrand: (E - μ) × T(E) × (-df/dE)
            T_E = np.interp(E, energies, transmission, left=0, right=0)
            
            # Fermi-Dirac derivative
            x = (E - fermi_level) / (kB * temperature)
            if np.abs(x) > 100:
                return 0
            
            df_dE = -1 / (kB * temperature) * np.exp(x) / (1 + np.exp(x))**2
            
            return (E - fermi_level) * T_E * (-df_dE)
        
        def integrand_L0(E):
            # L0 integrand: T(E) × (-df/dE)
            T_E = np.interp(E, energies, transmission, left=0, right=0)
            
            x = (E - fermi_level) / (kB * temperature)
            if np.abs(x) > 100:
                return 0
            
            df_dE = -1 / (kB * temperature) * np.exp(x) / (1 + np.exp(x))**2
            
            return T_E * (-df_dE)
        
        # Numerical integration
        L1, _ = integrate.quad(integrand_L1, energies[0], energies[-1])
        L0, _ = integrate.quad(integrand_L0, energies[0], energies[-1])
        
        if L0 > 0:
            S = -L1 / (L0 * temperature)
        else:
            S = 0
        
        return S
    
    def calculate_from_band_structure(self,
                                     band_energies: np.ndarray,
                                     kpoints: np.ndarray,
                                     fermi_level: float) -> float:
        """
        Calculate Seebeck coefficient from band structure.
        
        Uses the relaxation time approximation and Boltzmann transport.
        """
        # This would use Boltzmann transport equation
        # For now, return a placeholder
        
        warnings.warn("Band structure Seebeck calculation requires BTE solver")
        return 100e-6  # 100 μV/K placeholder


class ConductivityCalculator:
    """
    Calculator for electrical and thermal conductivity.
    """
    
    def __init__(self):
        self.kB = 8.617e-5  # eV/K
        self.e_charge = 1.602e-19  # C
        self.hbar = 6.582e-16  # eV·s
    
    def calculate_electrical_conductivity(self,
                                         energies: np.ndarray,
                                         transmission: np.ndarray,
                                         temperature: float,
                                         fermi_level: float = 0.0) -> float:
        """
        Calculate electrical conductivity from transmission.
        
        σ = (2e²/h) ∫ T(E) (-∂f/∂E) dE
        """
        G0 = 2 * self.e_charge**2 / self.hbar  # Quantum of conductance
        
        def integrand(E):
            T_E = np.interp(E, energies, transmission, left=0, right=0)
            
            # Fermi-Dirac derivative
            x = (E - fermi_level) / (self.kB * temperature)
            if np.abs(x) > 100:
                return 0
            
            df_dE = -1 / (self.kB * temperature) * np.exp(x) / (1 + np.exp(x))**2
            
            return T_E * (-df_dE)
        
        # Integrate
        integral, _ = integrate.quad(integrand, energies[0], energies[-1])
        
        # Conductance (in units of G0)
        conductance = integral
        
        # Convert to conductivity (requires geometry)
        # For now return normalized value
        return conductance
    
    def calculate_thermal_conductivity_electronic(self,
                                                 energies: np.ndarray,
                                                 transmission: np.ndarray,
                                                 temperature: float,
                                                 fermi_level: float = 0.0) -> float:
        """
        Calculate electronic thermal conductivity.
        
        κ_e = (2/h) ∫ (E-μ)² T(E) (-∂f/∂E) dE
        """
        def integrand(E):
            T_E = np.interp(E, energies, transmission, left=0, right=0)
            
            x = (E - fermi_level) / (self.kB * temperature)
            if np.abs(x) > 100:
                return 0
            
            df_dE = -1 / (self.kB * temperature) * np.exp(x) / (1 + np.exp(x))**2
            
            return (E - fermi_level)**2 * T_E * (-df_dE)
        
        integral, _ = integrate.quad(integrand, energies[0], energies[-1])
        
        # Convert to thermal conductivity
        # L0 = (π²/3)(k_B/e)² (Lorenz number)
        kappa = integral / temperature
        
        return kappa


class ZTOptimizer:
    """
    Optimizer for thermoelectric figure of merit ZT.
    
    Provides strategies for band engineering and doping optimization.
    """
    
    def __init__(self, transport_coeffs: TransportCoefficients):
        self.coeffs = transport_coeffs
    
    def optimize_doping(self,
                       doping_range: Tuple[float, float],
                       temperature: float = 300.0) -> Dict:
        """
        Find optimal doping level for maximum ZT.
        
        Args:
            doping_range: (min_doping, max_doping) in cm^-3
            temperature: Operating temperature
            
        Returns:
            Dictionary with optimal doping and ZT
        """
        # This is a simplified implementation
        # Real implementation would recalculate transport coefficients
        # at each doping level
        
        doping_values = np.linspace(doping_range[0], doping_range[1], 100)
        
        zt_values = []
        for doping in doping_values:
            # Estimate ZT at this doping level
            # Simplified model: ZT depends on carrier concentration
            
            # Power factor typically peaks at intermediate doping
            # Thermal conductivity increases with doping
            
            # Simplified model
            pf = self._estimate_power_factor(doping)
            kappa = self._estimate_thermal_conductivity(doping)
            
            zt = pf * temperature / kappa
            zt_values.append(zt)
        
        zt_values = np.array(zt_values)
        max_idx = np.argmax(zt_values)
        
        return {
            'optimal_doping': doping_values[max_idx],
            'max_zt': zt_values[max_idx],
            'doping_range': doping_values,
            'zt_values': zt_values
        }
    
    def _estimate_power_factor(self, doping: float) -> float:
        """
        Estimate power factor as function of doping.
        
        PF = S²σ typically peaks at intermediate carrier concentrations.
        """
        # Simplified model: parabolic dependence
        # Peak at around 1e19 cm^-3
        
        log_doping = np.log10(doping)
        optimal_log_doping = 19.0
        
        # Gaussian-like peak
        pf = np.exp(-(log_doping - optimal_log_doping)**2 / 2)
        
        return pf
    
    def _estimate_thermal_conductivity(self, doping: float) -> float:
        """
        Estimate thermal conductivity as function of doping.
        """
        # κ increases with doping due to electronic contribution
        base_kappa = 1.0  # W/mK
        electronic_kappa = 0.1 * (doping / 1e19)**0.5
        
        return base_kappa + electronic_kappa
    
    def band_engineering_strategy(self, 
                                 target_temperature: float = 300.0) -> Dict:
        """
        Provide band engineering strategies for improved ZT.
        
        Returns:
            Dictionary with recommended strategies
        """
        strategies = {
            'band_convergence': {
                'description': 'Converge multiple valleys at Fermi level',
                'effect': 'Increases density of states and Seebeck coefficient',
                'implementation': 'Alloying or strain engineering'
            },
            'quantum_confinement': {
                'description': 'Use nanostructures to enhance Seebeck',
                'effect': 'Sharpens density of states',
                'implementation': 'Nanowires or quantum dots'
            },
            'band_flattening': {
                'description': 'Flatten bands near band edge',
                'effect': 'Increases effective mass and Seebeck',
                'implementation': 'Heavy element doping'
            },
            'phonon_glass_electron_crystal': {
                'description': 'Minimize κ_l while maintaining good σ',
                'effect': 'Reduces denominator in ZT',
                'implementation': 'Complex crystal structures'
            }
        }
        
        # Calculate expected improvements
        baseline_zt = np.max(self.coeffs.calculate_zt())
        
        recommendations = {
            'baseline_zt': baseline_zt,
            'strategies': strategies,
            'projected_zt': {
                'with_band_convergence': baseline_zt * 1.5,
                'with_nanostructuring': baseline_zt * 1.8,
                'with_pg_ec': baseline_zt * 2.0,
            }
        }
        
        return recommendations


class ThermoelectricDevice:
    """
    Full thermoelectric device simulation including heat and current flow.
    """
    
    def __init__(self,
                 n_type_material: TransportCoefficients,
                 p_type_material: TransportCoefficients,
                 length: float = 1e-3,  # m
                 area: float = 1e-6):  # m²
        self.n_type = n_type_material
        self.p_type = p_type_material
        self.length = length
        self.area = area
    
    def calculate_efficiency(self,
                            hot_side_temp: float,
                            cold_side_temp: float) -> float:
        """
        Calculate thermoelectric conversion efficiency.
        
        η = (T_hot - T_cold) / T_hot × (√(1 + ZT) - 1) / (√(1 + ZT) + T_cold/T_hot)
        """
        T_hot = hot_side_temp
        T_cold = cold_side_temp
        
        # Average ZT
        zt_n = np.mean(self.n_type.calculate_zt())
        zt_p = np.mean(self.p_type.calculate_zt())
        zt_avg = (zt_n + zt_p) / 2
        
        # Carnot efficiency
        carnot = (T_hot - T_cold) / T_hot
        
        # Thermoelectric efficiency
        eta = carnot * (np.sqrt(1 + zt_avg) - 1) / \
              (np.sqrt(1 + zt_avg) + T_cold / T_hot)
        
        return eta
    
    def calculate_power_output(self,
                              hot_side_temp: float,
                              cold_side_temp: float,
                              load_resistance: float = None) -> float:
        """
        Calculate electrical power output.
        """
        # Temperature difference
        delta_T = hot_side_temp - cold_side_temp
        
        # Seebeck voltage
        S_n = np.mean(np.abs(self.n_type.seebeck))
        S_p = np.mean(np.abs(self.p_type.seebeck))
        S_total = S_n + S_p
        
        voltage = S_total * delta_T
        
        # Internal resistance
        sigma_n = np.mean(self.n_type.sigma)
        sigma_p = np.mean(self.p_type.sigma)
        
        R_n = self.length / (sigma_n * self.area)
        R_p = self.length / (sigma_p * self.area)
        R_internal = R_n + R_p
        
        # Load matching
        if load_resistance is None:
            load_resistance = R_internal
        
        # Current
        I = voltage / (R_internal + load_resistance)
        
        # Power
        power = I**2 * load_resistance
        
        return power
    
    def optimize_geometry(self,
                         hot_side_temp: float,
                         cold_side_temp: float) -> Dict:
        """
        Find optimal leg geometry for maximum power or efficiency.
        """
        # Length-to-area ratio optimization
        # For maximum power density: specific geometry needed
        
        return {
            'optimal_length': self.length,
            'optimal_area': self.area,
            'aspect_ratio': self.length / np.sqrt(self.area),
            'max_power_density': 0.0  # Placeholder
        }


def example_seebeck_calculation():
    """
    Example: Calculate Seebeck coefficient from transmission.
    """
    print("=" * 60)
    print("Example: Seebeck Coefficient Calculation")
    print("=" * 60)
    
    # Create a simple transmission function
    energies = np.linspace(-2, 2, 500)
    
    # Lorentzian-shaped transmission centered at 0.2 eV
    E0 = 0.2
    gamma = 0.1
    transmission = 0.5 / ((energies - E0)**2 + gamma**2)
    transmission = np.clip(transmission, 0, 1)
    
    # Calculate Seebeck
    seebeck_calc = SeebeckCalculator(temperatures=np.linspace(100, 500, 5))
    
    temps, seebeck = seebeck_calc.calculate_from_transmission(
        energies, transmission, fermi_level=0.0, method="cutler_mott"
    )
    
    print(f"\nSeebeck coefficient:")
    for T, S in zip(temps, seebeck):
        print(f"  T = {T:.0f} K: S = {S*1e6:.1f} μV/K")
    
    return temps, seebeck


def example_zt_optimization():
    """
    Example: ZT optimization.
    """
    print("\n" + "=" * 60)
    print("Example: ZT Optimization")
    print("=" * 60)
    
    # Create transport coefficients
    temps = np.linspace(300, 800, 20)
    
    # Model material properties
    sigma = 1e5 * np.ones_like(temps)  # S/m
    seebeck = 200e-6 * (temps / 300)**0.5  # V/K
    kappa_e = 1.0 * np.ones_like(temps)  # W/mK
    kappa_l = 1.5 * np.ones_like(temps)  # W/mK
    
    coeffs = TransportCoefficients(
        sigma=sigma,
        seebeck=seebeck,
        kappa_e=kappa_e,
        kappa_l=kappa_l,
        temperatures=temps
    )
    
    # Calculate ZT
    zt = coeffs.calculate_zt()
    max_zt_idx = np.argmax(zt)
    
    print(f"\nThermoelectric performance:")
    print(f"  Max ZT: {zt[max_zt_idx]:.2f} at T = {temps[max_zt_idx]:.0f} K")
    print(f"  Power factor at 300K: {coeffs.calculate_power_factor()[0]*1e3:.2f} mW/m·K²")
    
    # Optimization
    optimizer = ZTOptimizer(coeffs)
    
    doping_result = optimizer.optimize_doping((1e17, 1e21), temperature=300)
    print(f"\nDoping optimization:")
    print(f"  Optimal doping: {doping_result['optimal_doping']:.2e} cm⁻³")
    print(f"  Max ZT: {doping_result['max_zt']:.2f}")
    
    strategies = optimizer.band_engineering_strategy()
    print(f"\nBand engineering strategies:")
    for name, data in strategies['strategies'].items():
        print(f"  - {name}: {data['description']}")
    
    return coeffs, zt


def example_device_simulation():
    """
    Example: Thermoelectric device simulation.
    """
    print("\n" + "=" * 60)
    print("Example: Thermoelectric Device")
    print("=" * 60)
    
    temps = np.linspace(300, 600, 10)
    
    # N-type material
    n_type = TransportCoefficients(
        sigma=5e4 * np.ones_like(temps),
        seebeck=-150e-6 * np.ones_like(temps),
        kappa_e=0.5 * np.ones_like(temps),
        kappa_l=1.0 * np.ones_like(temps),
        temperatures=temps
    )
    
    # P-type material
    p_type = TransportCoefficients(
        sigma=5e4 * np.ones_like(temps),
        seebeck=150e-6 * np.ones_like(temps),
        kappa_e=0.5 * np.ones_like(temps),
        kappa_l=1.0 * np.ones_like(temps),
        temperatures=temps
    )
    
    # Create device
    device = ThermoelectricDevice(n_type, p_type)
    
    # Calculate efficiency
    T_hot = 600
    T_cold = 300
    
    efficiency = device.calculate_efficiency(T_hot, T_cold)
    power = device.calculate_power_output(T_hot, T_cold)
    
    carnot = (T_hot - T_cold) / T_hot
    
    print(f"\nDevice performance:")
    print(f"  Carnot efficiency: {carnot*100:.1f}%")
    print(f"  Thermoelectric efficiency: {efficiency*100:.2f}%")
    print(f"  Relative to Carnot: {efficiency/carnot*100:.1f}%")
    print(f"  Power output: {power*1e3:.2f} mW")
    
    return device


if __name__ == "__main__":
    # Run examples
    temps, seebeck = example_seebeck_calculation()
    coeffs, zt = example_zt_optimization()
    device = example_device_simulation()
    
    print("\n" + "=" * 60)
    print("Thermoelectric Module - Test Complete")
    print("=" * 60)
