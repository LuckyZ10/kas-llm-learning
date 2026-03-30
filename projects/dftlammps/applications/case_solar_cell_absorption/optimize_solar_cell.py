"""
Solar Cell Absorption Optimization Case Study
==============================================

Application of GW+BSE methods to optimize solar cell light absorption:
- Perovskite absorbers
- Silicon heterojunctions  
- Multi-junction cells

Key analysis:
- Absorption coefficient from BSE
- Optimal thickness calculation
- Quantum efficiency estimation
- Exciton dissociation analysis

Author: DFTLammps Research Team
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

# Import our modules
import sys
sys.path.insert(0, '/root/.openclaw/workspace')

from dftlammps.many_body import (
    YamboInterface, GWApproximation, BSEType,
    QPParameters, BSEParameters, ExcitonProperties,
    MaterialParameters
)
from dftlammps.optical_advanced import OpticalParameters


@dataclass
class SolarCellParameters:
    """Solar cell design parameters."""
    material: str = "CH3NH3PbI3"
    bandgap: float = 1.6  # eV
    thickness: float = 500e-7  # cm (500 nm)
    diffusion_length: float = 1e-4  # cm (1 μm)
    
    # Optical properties
    refractive_index: float = 2.5
    absorption_onset: float = 1.5  # eV
    
    # Device parameters
    fill_factor: float = 0.8
    voc_factor: float = 0.8  # V_oc = E_g/e * voc_factor


class SolarAbsorptionOptimizer:
    """
    Optimizer for solar cell absorption using GW+BSE calculations.
    """
    
    def __init__(self, cell_params: SolarCellParameters):
        self.params = cell_params
        
        # Physical constants
        self.h = 4.136e-15  # eV·s
        self.c = 2.998e10   # cm/s
        self.q = 1.602e-19  # C
        
        # Solar spectrum (AM1.5G) - simplified
        self.am15g = self._load_am15g()
    
    def _load_am15g(self) -> Dict:
        """
        Load AM1.5G solar spectrum.
        
        Returns energy grid and photon flux.
        """
        # Energy range
        energies = np.linspace(0.3, 4.5, 500)  # eV
        
        # Simplified AM1.5G (ASTM G173)
        # Photon flux in photons/(cm²·s·eV)
        photon_flux = np.zeros_like(energies)
        
        for i, E in enumerate(energies):
            if E < 4.0:
                # Approximate spectrum
                photon_flux[i] = 2.0e17 * np.exp(-E / 0.3) * (E / 1.5)**(-3)
        
        return {
            'energies': energies,
            'photon_flux': photon_flux
        }
    
    def calculate_absorption_coefficient(self,
                                         bse_spectrum: Dict) -> np.ndarray:
        """
        Calculate absorption coefficient from BSE spectrum.
        
        α(E) = (2E / ħc) κ(E)
        
        where κ is extinction coefficient from ε2.
        
        Args:
            bse_spectrum: BSE absorption spectrum
            
        Returns:
            Absorption coefficient in cm^-1
        """
        energies = bse_spectrum['energies']
        epsilon2 = bse_spectrum['epsilon2']
        
        # Extinction coefficient
        epsilon1 = np.ones_like(energies) * self.params.refractive_index**2
        kappa = np.sqrt((np.sqrt(epsilon1**2 + epsilon2**2) - epsilon1) / 2)
        
        # Absorption coefficient
        alpha = (2 * energies * 1.602e-12) / (6.582e-16 * self.c) * kappa
        
        # Convert to cm^-1
        alpha_cm = alpha / 100
        
        return alpha_cm
    
    def optimal_thickness(self,
                         absorption_coefficient: np.ndarray,
                         energies: np.ndarray,
                         target_efficiency: float = 0.9) -> float:
        """
        Calculate optimal absorber thickness.
        
        Trade-off between absorption and charge collection.
        
        Args:
            absorption_coefficient: Absorption coefficient
            energies: Energy grid
            target_efficiency: Target absorption efficiency
            
        Returns:
            Optimal thickness in cm
        """
        # Maximum thickness based on diffusion length
        L_max = self.params.diffusion_length
        
        # Find thickness that gives target absorption
        # For single pass: T = exp(-αd)
        # Absorbed fraction: 1 - exp(-αd)
        
        # Average absorption coefficient above bandgap
        above_gap = energies > self.params.bandgap
        if np.any(above_gap):
            alpha_avg = np.mean(absorption_coefficient[above_gap])
        else:
            alpha_avg = 1e4  # cm^-1
        
        # Thickness for target absorption
        d_opt = -np.log(1 - target_efficiency) / alpha_avg
        
        # Limit by diffusion length
        d_opt = min(d_opt, L_max)
        
        return d_opt
    
    def quantum_efficiency(self,
                          absorption_coefficient: np.ndarray,
                          thickness: float) -> np.ndarray:
        """
        Calculate external quantum efficiency (EQE).
        
        EQE(E) = (1 - R(E)) × (1 - exp(-α(E)d)) × η_collection
        
        Args:
            absorption_coefficient: Absorption coefficient
            thickness: Absorber thickness
            
        Returns:
            EQE as function of energy
        """
        # Reflectance (simplified)
        n = self.params.refractive_index
        R = ((n - 1) / (n + 1))**2
        
        # Absorbed fraction
        absorbed = 1 - np.exp(-absorption_coefficient * thickness)
        
        # Collection efficiency (simplified)
        # η_collection = 1 / (1 + d/L_diff)
        eta_collection = 1 / (1 + thickness / self.params.diffusion_length)
        
        eqe = (1 - R) * absorbed * eta_collection
        
        return eqe
    
    def short_circuit_current(self,
                             eqe: np.ndarray,
                             energies: np.ndarray) -> float:
        """
        Calculate short-circuit current density.
        
        J_sc = q ∫ EQE(E) × Φ(E) dE
        
        where Φ(E) is photon flux from AM1.5G.
        
        Args:
            eqe: External quantum efficiency
            energies: Energy grid
            
        Returns:
            J_sc in mA/cm²
        """
        # Interpolate AM1.5G to calculation grid
        photon_flux = np.interp(
            energies,
            self.am15g['energies'],
            self.am15g['photon_flux']
        )
        
        # Integrate
        integrand = eqe * photon_flux
        J_sc = self.q * np.trapezoid(integrand, energies)
        
        # Convert to mA/cm²
        J_sc_mA = J_sc * 1000
        
        return J_sc_mA
    
    def power_conversion_efficiency(self,
                                   J_sc: float,
                                   voc: Optional[float] = None) -> float:
        """
        Estimate power conversion efficiency.
        
        η = (J_sc × V_oc × FF) / P_in
        
        Args:
            J_sc: Short-circuit current (mA/cm²)
            voc: Open-circuit voltage (V) [optional]
            
        Returns:
            Efficiency (fraction)
        """
        if voc is None:
            voc = self.params.bandgap * self.params.voc_factor
        
        FF = self.params.fill_factor
        
        # Input power (AM1.5G)
        P_in = 100  # mW/cm²
        
        efficiency = (J_sc * voc * FF) / P_in
        
        return efficiency
    
    def exciton_dissociation_analysis(self,
                                      exciton_binding: float,
                                      electric_field: float = 1e5) -> Dict:
        """
        Analyze exciton dissociation efficiency.
        
        For organic/perovskite solar cells, excitons must dissociate.
        
        Args:
            exciton_binding: Exciton binding energy (eV)
            electric_field: Built-in field (V/cm)
            
        Returns:
            Dissociation analysis
        """
        # Onsager-Braun model (simplified)
        # Exciton dissociation rate depends on binding energy and field
        
        # Critical radius for field-assisted dissociation
        kT = 0.0259  # eV at 300K
        
        # Field-assisted dissociation efficiency
        # η ∝ exp(-E_b/kT) × (1 + eE_a/kT) for E_a = exciton radius
        
        # Simplified model
        a_B = 2e-7  # cm (exciton Bohr radius ~2 nm)
        
        # Dissociation probability
        P_diss = np.exp(-exciton_binding / kT) * (1 + electric_field * a_B / kT)
        P_diss = min(P_diss, 1.0)
        
        return {
            'binding_energy': exciton_binding,
            'electric_field': electric_field,
            'dissociation_probability': P_diss,
            'geminate_recombination': 1 - P_diss,
            'thermal_energy': kT
        }
    
    def optimize_material(self,
                         candidate_materials: List[Dict]) -> Dict:
        """
        Optimize material selection based on GW+BSE predictions.
        
        Args:
            candidate_materials: List of material dictionaries with:
                - name
                - bandgap
                - absorption_spectrum
                - exciton_binding
                
        Returns:
            Optimization results
        """
        results = []
        
        for material in candidate_materials:
            # Calculate performance metrics
            bandgap = material['bandgap']
            
            # Simulate absorption coefficient
            energies = np.linspace(0.3, 4.5, 500)
            alpha = np.zeros_like(energies)
            
            # Simplified absorption edge
            for i, E in enumerate(energies):
                if E >= bandgap:
                    alpha[i] = 1e5 * np.sqrt(E - bandgap)
            
            # Optimal thickness
            d_opt = self.optimal_thickness(alpha, energies)
            
            # EQE
            eqe = self.quantum_efficiency(alpha, d_opt)
            
            # J_sc
            J_sc = self.short_circuit_current(eqe, energies)
            
            # Efficiency
            eta = self.power_conversion_efficiency(J_sc)
            
            # Exciton analysis
            exciton_analysis = self.exciton_dissociation_analysis(
                material.get('exciton_binding', 0.01)
            )
            
            results.append({
                'material': material['name'],
                'bandgap': bandgap,
                'optimal_thickness_nm': d_opt * 1e7,
                'J_sc_mA_cm2': J_sc,
                'efficiency_percent': eta * 100,
                'exciton_dissociation': exciton_analysis['dissociation_probability']
            })
        
        # Find best material
        best = max(results, key=lambda x: x['efficiency_percent'])
        
        return {
            'candidates': results,
            'best_material': best,
            'optimization_criteria': 'max_efficiency'
        }


def run_solar_cell_case_study():
    """
    Run complete solar cell case study.
    """
    print("="*70)
    print("SOLAR CELL ABSORPTION OPTIMIZATION - CASE STUDY")
    print("="*70)
    
    # Define solar cell parameters
    cell_params = SolarCellParameters(
        material="CH3NH3PbI3",
        bandgap=1.6,
        thickness=500e-7,
        diffusion_length=1e-4,
        refractive_index=2.5
    )
    
    optimizer = SolarAbsorptionOptimizer(cell_params)
    
    print("\n1. Material Properties")
    print("-" * 50)
    print(f"Material: {cell_params.material}")
    print(f"Bandgap: {cell_params.bandgap} eV")
    print(f"Current thickness: {cell_params.thickness*1e7:.0f} nm")
    print(f"Diffusion length: {cell_params.diffusion_length*1e4:.1f} μm")
    
    # Simulate BSE absorption spectrum
    print("\n2. BSE Absorption Spectrum")
    print("-" * 50)
    
    energies = np.linspace(0.5, 4.0, 500)
    
    # Simulate realistic absorption spectrum
    epsilon2 = np.zeros_like(energies)
    
    # Exciton peak
    exciton_energy = cell_params.bandgap - 0.02  # 20 meV binding
    exciton_width = 0.02
    epsilon2 += 10 * np.exp(-(energies - exciton_energy)**2 / (2 * exciton_width**2))
    
    # Continuum absorption
    for i, E in enumerate(energies):
        if E >= cell_params.bandgap:
            epsilon2[i] += 5 * np.sqrt(E - cell_params.bandgap)
    
    bse_spectrum = {
        'energies': energies,
        'epsilon2': epsilon2
    }
    
    # Calculate absorption coefficient
    alpha = optimizer.calculate_absorption_coefficient(bse_spectrum)
    
    print(f"Peak absorption coefficient: {np.max(alpha):.2e} cm⁻¹")
    print(f"Absorption coefficient at bandgap: {alpha[np.argmin(np.abs(energies - cell_params.bandgap))]:.2e} cm⁻¹")
    
    # Optimal thickness
    print("\n3. Optimal Thickness Analysis")
    print("-" * 50)
    
    d_opt = optimizer.optimal_thickness(alpha, energies, target_efficiency=0.95)
    print(f"Optimal thickness: {d_opt*1e7:.0f} nm")
    print(f"Current vs optimal: {cell_params.thickness*1e7:.0f} nm → {d_opt*1e7:.0f} nm")
    
    # Quantum efficiency
    print("\n4. Quantum Efficiency")
    print("-" * 50)
    
    eqe = optimizer.quantum_efficiency(alpha, d_opt)
    J_sc = optimizer.short_circuit_current(eqe, energies)
    
    print(f"Short-circuit current: {J_sc:.2f} mA/cm²")
    
    # Device performance
    print("\n5. Device Performance")
    print("-" * 50)
    
    efficiency = optimizer.power_conversion_efficiency(J_sc)
    print(f"Power conversion efficiency: {efficiency*100:.2f}%")
    
    # Exciton analysis
    print("\n6. Exciton Dissociation Analysis")
    print("-" * 50)
    
    exciton_analysis = optimizer.exciton_dissociation_analysis(
        exciton_binding=0.02,
        electric_field=1e5
    )
    
    print(f"Exciton binding energy: {exciton_analysis['binding_energy']*1000:.1f} meV")
    print(f"Dissociation probability: {exciton_analysis['dissociation_probability']*100:.1f}%")
    
    # Material optimization
    print("\n7. Material Optimization")
    print("-" * 50)
    
    candidates = [
        {'name': 'CH3NH3PbI3', 'bandgap': 1.6, 'exciton_binding': 0.02},
        {'name': 'CsPbI3', 'bandgap': 1.73, 'exciton_binding': 0.02},
        {'name': 'GaAs', 'bandgap': 1.42, 'exciton_binding': 0.004},
        {'name': 'CIGS', 'bandgap': 1.2, 'exciton_binding': 0.01},
        {'name': 'CdTe', 'bandgap': 1.5, 'exciton_binding': 0.01}
    ]
    
    optimization = optimizer.optimize_material(candidates)
    
    print("\nCandidate Materials:")
    for result in optimization['candidates']:
        print(f"  {result['material']:12s}: η = {result['efficiency_percent']:.2f}%, "
              f"J_sc = {result['J_sc_mA_cm2']:.2f} mA/cm², "
              f"d_opt = {result['optimal_thickness_nm']:.0f} nm")
    
    print(f"\nBest material: {optimization['best_material']['material']}")
    print(f"Efficiency: {optimization['best_material']['efficiency_percent']:.2f}%")
    
    # Save results
    results = {
        'material': cell_params.material,
        'bandgap': cell_params.bandgap,
        'optimal_thickness_nm': d_opt * 1e7,
        'J_sc_mA_cm2': J_sc,
        'efficiency_percent': efficiency * 100,
        'exciton_binding_meV': exciton_analysis['binding_energy'] * 1000,
        'material_optimization': optimization
    }
    
    with open('solar_cell_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("Results saved to solar_cell_results.json")
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = run_solar_cell_case_study()
