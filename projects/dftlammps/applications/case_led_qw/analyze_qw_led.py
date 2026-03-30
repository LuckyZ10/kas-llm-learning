"""
Quantum Well LED Luminescence Case Study
=========================================

Analysis of quantum well LED emission using GW+BSE:
- Quantum confinement effects
- Excitonic enhancement
- Light extraction efficiency
- Temperature dependence

Systems studied:
- GaN/AlGaN QWs (UV-LEDs)
- InGaN/GaN QWs (blue/green LEDs)
- InAs/GaAs QWs (IR LEDs)

Author: DFTLammps Research Team
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

import sys
sys.path.insert(0, '/root/.openclaw/workspace')

from dftlammps.many_body import (
    ExcitonProperties, MaterialParameters,
    BerkeleyGWInterface, GWFlavor
)
from dftlammps.defect_excited import DefectLuminescence, VibronicParameters


@dataclass
class QuantumWellParameters:
    """Quantum well structural parameters."""
    # Materials
    well_material: str = "InGaN"
    barrier_material: str = "GaN"
    
    # Geometry
    well_width: float = 3.0  # nm
    barrier_width: float = 10.0  # nm
    n_wells: int = 3
    
    # Composition
    indium_fraction: float = 0.2  # For InGaN
    
    # Electronic properties
    bandgap_well: float = 2.7  # eV (In0.2Ga0.8N)
    bandgap_barrier: float = 3.4  # eV (GaN)
    conduction_offset: float = 0.5  # ΔEc/ΔEg
    
    # Optical properties
    refractive_index: float = 2.5
    
    # Defect properties
    threading_dislocation: float = 1e8  # cm^-2


class QuantumWellLED:
    """
    Quantum Well LED emission calculator.
    """
    
    def __init__(self, qw_params: QuantumWellParameters):
        self.params = qw_params
        
        # Constants
        self.hbar = 6.582e-16  # eV·s
        self.kB = 8.617e-5     # eV/K
        self.c = 2.998e10      # cm/s
        self.m0 = 9.109e-31    # kg
        
        # Effective masses (InGaN)
        self.m_e = 0.2 * self.m0
        self.m_hh = 1.0 * self.m0
    
    def quantum_confinement_energy(self,
                                    particle_mass: float,
                                    well_width: float) -> float:
        """
        Calculate quantum confinement energy for infinite well.
        
        E_n = (ħ² π² n²) / (2 m* L²)
        
        Args:
            particle_mass: Effective mass
            well_width: Well width in nm
            
        Returns:
            Confinement energy in eV
        """
        L = well_width * 1e-9  # Convert to m
        
        # Ground state (n=1)
        E1_J = (np.pi**2 * (1.055e-34)**2) / (2 * particle_mass * L**2)
        E1_eV = E1_J / 1.602e-19
        
        return E1_eV
    
    def calculate_transition_energy(self,
                                    include_qc: bool = True) -> float:
        """
        Calculate QW transition energy including confinement.
        
        Args:
            include_qc: Include quantum confinement
            
        Returns:
            Transition energy in eV
        """
        E_g = self.params.bandgap_well
        
        if include_qc:
            # Electron confinement
            E_e = self.quantum_confinement_energy(self.m_e, self.params.well_width)
            
            # Hole confinement
            E_h = self.quantum_confinement_energy(self.m_hh, self.params.well_width)
            
            # Quantum-confined Stark effect (simplified)
            # Due to piezoelectric fields in InGaN/GaN
            qcse_shift = -0.05  # eV (redshift due to field)
            
            E_transition = E_g + E_e + E_h + qcse_shift
        else:
            E_transition = E_g
        
        return E_transition
    
    def exciton_properties_qw(self) -> Dict:
        """
        Calculate 2D exciton properties in quantum well.
        
        Returns:
            Exciton properties dictionary
        """
        # Reduced mass
        mu = (self.m_e * self.m_hh) / (self.m_e + self.m_hh)
        mu_rel = mu / self.m0
        
        # Dielectric constant
        epsilon = 8.9  # GaN
        
        # 2D exciton binding energy
        # E_b^2D = 4 × E_b^3D (for infinite well)
        # E_b^3D = μ/(ε² m0) × Ry
        Ry = 13.6  # eV
        E_b_3d = (mu_rel / epsilon**2) * Ry
        E_b_2d = 4 * E_b_3d  # Enhanced binding in 2D
        
        # 2D Bohr radius
        a_0 = 0.529  # Angstrom
        a_b_2d = a_0 * epsilon / mu_rel / 2  # Smaller in 2D
        
        # Oscillator strength enhancement in 2D
        f_2d = 2 / (a_b_2d * 1e-8)  # per cm^-2
        
        return {
            'binding_energy': E_b_2d,
            'bohr_radius_angstrom': a_b_2d,
            'rydberg_energy': E_b_3d,
            'oscillator_strength_per_area': f_2d,
            'reduced_mass': mu_rel
        }
    
    def luminescence_spectrum(self,
                             temperature: float = 300,
                             broadening: float = 0.02) -> Dict:
        """
        Calculate QW photoluminescence spectrum.
        
        Args:
            temperature: Temperature in K
            broadening: Inhomogeneous broadening in eV
            
        Returns:
            PL spectrum dictionary
        """
        # Energy grid
        energies = np.linspace(1.5, 4.0, 500)
        
        # Transition energy
        E_trans = self.calculate_transition_energy()
        
        # Exciton properties
        exciton = self.exciton_properties_qw()
        
        # Free carrier contribution (band-to-band)
        # Broadened by thermal distribution
        kT = self.kB * temperature
        
        carrier_pl = np.zeros_like(energies)
        for i, E in enumerate(energies):
            if E >= E_trans:
                # Joint density of states (2D: step function)
                # Broadened by thermal distribution
                carrier_pl[i] = np.exp(-(E - E_trans) / kT)
        
        # Exciton contribution
        E_exc = E_trans - exciton['binding_energy'] / 1000  # meV to eV
        
        # Exciton lineshape
        exciton_pl = np.exp(-(energies - E_exc)**2 / (2 * broadening**2))
        
        # Total PL (sum of exciton and free carrier)
        # Ratio depends on temperature and screening
        exciton_fraction = 0.7  # At room temperature
        pl_total = exciton_fraction * exciton_pl + (1 - exciton_fraction) * carrier_pl
        
        # Normalize
        pl_total /= np.max(pl_total)
        
        return {
            'energies': energies,
            'pl_intensity': pl_total,
            'exciton_peak': E_exc,
            'band_edge': E_trans,
            'exciton_binding': exciton['binding_energy'],
            'temperature': temperature,
            'fwhm': self.calculate_fwhm(energies, pl_total)
        }
    
    def calculate_fwhm(self, energies: np.ndarray, intensity: np.ndarray) -> float:
        """
        Calculate full width at half maximum.
        """
        half_max = np.max(intensity) / 2
        above_half = np.where(intensity > half_max)[0]
        
        if len(above_half) > 1:
            fwhm = energies[above_half[-1]] - energies[above_half[0]]
        else:
            fwhm = 0
        
        return fwhm
    
    def internal_quantum_efficiency(self,
                                   temperature: float = 300) -> float:
        """
        Calculate internal quantum efficiency (IQE).
        
        IQE = radiative_rate / (radiative_rate + nonradiative_rate)
        
        Args:
            temperature: Temperature in K
            
        Returns:
            IQE (fraction)
        """
        # Radiative rate (B coefficient × carrier density)
        # For GaN: B ≈ 2×10^-11 cm³/s
        B = 2e-11  # cm³/s
        n = 1e18   # cm^-3 (typical injection)
        
        R_rad = B * n**2
        
        # Non-radiative rate (Shockley-Read-Hall)
        # Limited by threading dislocations
        N_disl = self.params.threading_dislocation
        capture_cross = 1e-15  # cm²
        v_th = 1e7  # cm/s (thermal velocity)
        
        tau_nr = 1 / (N_disl * capture_cross * v_th)
        R_nonrad = n / tau_nr
        
        # Auger recombination (high injection)
        C = 1e-30  # cm^6/s
        R_auger = C * n**3
        
        IQE = R_rad / (R_rad + R_nonrad + R_auger)
        
        return IQE
    
    def light_extraction_efficiency(self,
                                   extraction_geometry: str = "planar") -> float:
        """
        Calculate light extraction efficiency.
        
        Args:
            extraction_geometry: LED geometry
            
        Returns:
            Extraction efficiency
        """
        n = self.params.refractive_index
        
        if extraction_geometry == "planar":
            # Critical angle
            theta_c = np.arcsin(1 / n)
            
            # Escape cone fraction
            eta_extract = (1 - np.cos(theta_c)) / 2
            
            # Fresnel reflection
            R_fresnel = ((n - 1) / (n + 1))**2
            
            eta_extract *= (1 - R_fresnel)
            
        elif extraction_geometry == "patterned":
            # Surface patterning improves extraction
            eta_extract = 0.5  # ~50% for patterned surfaces
            
        elif extraction_geometry == "roughened":
            # Surface roughening
            eta_extract = 0.6
            
        else:
            eta_extract = 0.1  # Default
        
        return eta_extract
    
    def external_quantum_efficiency(self,
                                   temperature: float = 300,
                                   geometry: str = "planar") -> float:
        """
        Calculate external quantum efficiency (EQE).
        
        EQE = IQE × extraction_efficiency
        
        Args:
            temperature: Temperature
            geometry: LED geometry
            
        Returns:
            EQE
        """
        IQE = self.internal_quantum_efficiency(temperature)
        eta_extract = self.light_extraction_efficiency(geometry)
        
        return IQE * eta_extract
    
    def temperature_dependence(self,
                              temperature_range: Tuple[float, float] = (100, 500),
                              n_points: int = 20) -> Dict:
        """
        Calculate temperature dependence of LED properties.
        
        Args:
            temperature_range: Temperature range in K
            n_points: Number of points
            
        Returns:
            Temperature-dependent data
        """
        temperatures = np.linspace(temperature_range[0], temperature_range[1], n_points)
        
        peak_energies = []
        fwhm_values = []
        iqe_values = []
        eqe_values = []
        
        for T in temperatures:
            # Spectrum
            spectrum = self.luminescence_spectrum(T)
            peak_energies.append(spectrum['exciton_peak'])
            fwhm_values.append(spectrum['fwhm'])
            
            # Efficiencies
            iqe = self.internal_quantum_efficiency(T)
            eqe = self.external_quantum_efficiency(T)
            
            iqe_values.append(iqe)
            eqe_values.append(eqe)
        
        return {
            'temperatures': temperatures,
            'peak_energies': np.array(peak_energies),
            'fwhm': np.array(fwhm_values),
            'IQE': np.array(iqe_values),
            'EQE': np.array(eqe_values)
        }
    
    def droop_analysis(self,
                      current_densities: np.ndarray) -> Dict:
        """
        Analyze efficiency droop at high current.
        
        Args:
            current_densities: Current density array (A/cm²)
            
        Returns:
            Droop analysis
        """
        # Efficiency vs current
        # Droop due to Auger recombination and carrier leakage
        
        efficiencies = []
        
        for J in current_densities:
            # Convert current to carrier density (simplified)
            n = J / (1.602e-19 * 1e7)  # cm^-3 (rough estimate)
            
            # IQE with Auger
            B = 2e-11
            C = 1e-30
            
            R_rad = B * n**2
            R_auger = C * n**3
            
            IQE = R_rad / (R_rad + R_auger + 1e6)  # + constant nonrad
            
            efficiencies.append(IQE * self.light_extraction_efficiency())
        
        # Find peak efficiency
        peak_idx = np.argmax(efficiencies)
        peak_eff = efficiencies[peak_idx]
        peak_current = current_densities[peak_idx]
        
        # Droop at high current
        high_current_eff = efficiencies[-1]
        droop = (peak_eff - high_current_eff) / peak_eff
        
        return {
            'current_densities': current_densities,
            'EQE': np.array(efficiencies),
            'peak_efficiency': peak_eff,
            'peak_current_density': peak_current,
            'droop_percentage': droop * 100
        }


def run_qw_led_case_study():
    """
    Run complete quantum well LED case study.
    """
    print("="*70)
    print("QUANTUM WELL LED LUMINESCENCE - CASE STUDY")
    print("="*70)
    
    # Define QW parameters
    qw_params = QuantumWellParameters(
        well_material="InGaN",
        barrier_material="GaN",
        well_width=3.0,
        indium_fraction=0.2,
        bandgap_well=2.7,  # In0.2Ga0.8N
        bandgap_barrier=3.4,  # GaN
        n_wells=3
    )
    
    led = QuantumWellLED(qw_params)
    
    print("\n1. Quantum Well Structure")
    print("-" * 50)
    print(f"Well material: {qw_params.well_material}")
    print(f"Barrier material: {qw_params.barrier_material}")
    print(f"Well width: {qw_params.well_width} nm")
    print(f"Number of wells: {qw_params.n_wells}")
    print(f"Indium fraction: {qw_params.indium_fraction}")
    
    # Quantum confinement
    print("\n2. Quantum Confinement Analysis")
    print("-" * 50)
    
    E_e = led.quantum_confinement_energy(led.m_e, qw_params.well_width)
    E_h = led.quantum_confinement_energy(led.m_hh, qw_params.well_width)
    
    print(f"Electron confinement: {E_e*1000:.1f} meV")
    print(f"Hole confinement: {E_h*1000:.1f} meV")
    
    E_trans = led.calculate_transition_energy()
    print(f"Transition energy: {E_trans:.3f} eV ({1240/E_trans:.0f} nm)")
    
    # Exciton properties
    print("\n3. Exciton Properties (2D)")
    print("-" * 50)
    
    exciton = led.exciton_properties_qw()
    print(f"Binding energy: {exciton['binding_energy']:.1f} meV")
    print(f"Bohr radius: {exciton['bohr_radius_angstrom']:.1f} Å")
    print(f"Oscillator strength: {exciton['oscillator_strength_per_area']:.2e} cm⁻²")
    
    # Luminescence spectrum
    print("\n4. Luminescence Spectrum")
    print("-" * 50)
    
    spectrum = led.luminescence_spectrum(temperature=300)
    print(f"Exciton peak: {spectrum['exciton_peak']:.3f} eV")
    print(f"Band edge: {spectrum['band_edge']:.3f} eV")
    print(f"FWHM: {spectrum['fwhm']*1000:.1f} meV")
    
    # Quantum efficiencies
    print("\n5. Quantum Efficiencies")
    print("-" * 50)
    
    IQE = led.internal_quantum_efficiency(300)
    eta_extract = led.light_extraction_efficiency("planar")
    EQE = led.external_quantum_efficiency(300)
    
    print(f"Internal quantum efficiency: {IQE*100:.1f}%")
    print(f"Light extraction efficiency: {eta_extract*100:.1f}%")
    print(f"External quantum efficiency: {EQE*100:.1f}%")
    
    # Temperature dependence
    print("\n6. Temperature Dependence")
    print("-" * 50)
    
    temp_data = led.temperature_dependence((100, 500), 10)
    print(f"Peak shift 100K → 500K: {(temp_data['peak_energies'][-1] - temp_data['peak_energies'][0])*1000:.1f} meV")
    print(f"IQE at 300K: {temp_data['IQE'][5]*100:.1f}%")
    print(f"IQE at 500K: {temp_data['IQE'][-1]*100:.1f}%")
    
    # Droop analysis
    print("\n7. Efficiency Droop Analysis")
    print("-" * 50)
    
    currents = np.linspace(1, 200, 50)  # A/cm²
    droop_data = led.droop_analysis(currents)
    
    print(f"Peak EQE: {droop_data['peak_efficiency']*100:.1f}%")
    print(f"Peak current: {droop_data['peak_current_density']:.1f} A/cm²")
    print(f"Droop at 200 A/cm²: {droop_data['droop_percentage']:.1f}%")
    
    # Save results
    results = {
        'material': qw_params.well_material,
        'well_width_nm': qw_params.well_width,
        'transition_energy_eV': E_trans,
        'wavelength_nm': 1240 / E_trans,
        'exciton_binding_meV': exciton['binding_energy'],
        'FWHM_meV': spectrum['fwhm'] * 1000,
        'IQE_percent': IQE * 100,
        'EQE_percent': EQE * 100,
        'peak_droop_current': droop_data['peak_current_density']
    }
    
    with open('qw_led_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("Results saved to qw_led_results.json")
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = run_qw_led_case_study()
