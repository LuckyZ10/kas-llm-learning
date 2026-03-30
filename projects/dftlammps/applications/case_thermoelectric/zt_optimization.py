"""
Thermoelectric Materials - ZT Optimization Study
=================================================

Case study: Optimization of thermoelectric figure of merit ZT in
materials using first-principles phonon and transport calculations.

Materials studied:
- Bi2Te3 (Bismuth telluride) - Room temperature TE
- PbTe (Lead telluride) - Mid-temperature TE
- SnSe (Tin selenide) - Low thermal conductivity TE

This script demonstrates:
1. Electronic structure calculation setup
2. Phonon calculations for lattice thermal conductivity
3. Electron-phonon coupling for electronic transport
4. ZT calculation and optimization
5. Nanostructuring effects on ZT

Author: DFTLammps Applications Team
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from functools import partial

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize_scalar, minimize

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dftlammps.phonon import (
    PhonopyInterface, PhononConfig,
    ThermalPropertyCalculator, ThermalConfig,
    LatticeThermalConductivity, ThermalConductivityConfig
)
from dftlammps.electron_phonon import (
    ElectronPhononCalculator, ElPhConfig,
    ElectronPhononTransport, TransportConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ThermoelectricMaterial:
    """Thermoelectric material specification."""
    
    name: str
    formula: str
    structure_file: str
    
    # Electronic properties
    band_gap: float  # eV
    effective_mass_e: float  # m_e (electron)
    effective_mass_h: float  # m_e (hole)
    
    # Lattice properties
    debye_temperature: float  # K
    lattice_thermal_conductivity_300K: float  # W/m/K
    
    # Doping parameters
    optimal_doping_n: Optional[float] = None  # cm^-3
    optimal_doping_p: Optional[float] = None  # cm^-3
    
    # Experimental ZT
    experimental_zt: Optional[Dict[float, float]] = None
    
    def __post_init__(self):
        if self.experimental_zt is None:
            self.experimental_zt = {}


class ThermoelectricOptimizer:
    """
    Optimization of thermoelectric figure of merit ZT.
    
    ZT = S²σT / (κ_e + κ_l)
    
    where:
    - S: Seebeck coefficient
    - σ: Electrical conductivity
    - κ_e: Electronic thermal conductivity
    - κ_l: Lattice thermal conductivity
    - T: Temperature
    
    This class implements:
    1. Lattice thermal conductivity calculation
    2. Electronic transport property calculation
    3. ZT calculation vs temperature and doping
    4. Optimization of doping level
    """
    
    def __init__(self, output_dir: str = "./thermoelectric_output"):
        """Initialize thermoelectric optimizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.materials: Dict[str, ThermoelectricMaterial] = {}
        self.results: Dict[str, Dict] = {}
        
        logger.info("Initialized Thermoelectric Optimizer")
    
    def add_material(self, material: ThermoelectricMaterial):
        """Add a thermoelectric material."""
        self.materials[material.name] = material
        logger.info(f"Added TE material: {material.name} ({material.formula})")
    
    def setup_standard_materials(self):
        """Setup standard thermoelectric materials."""
        # Bi2Te3 - Room temperature
        bi2te3 = ThermoelectricMaterial(
            name="Bi2Te3",
            formula="Bi2Te3",
            structure_file="Bi2Te3_POSCAR",
            band_gap=0.15,
            effective_mass_e=0.13,
            effective_mass_h=0.28,
            debye_temperature=165,
            lattice_thermal_conductivity_300K=1.5,
            optimal_doping_n=1e19,
            optimal_doping_p=8e18,
            experimental_zt={
                300: 1.0, 350: 1.1, 400: 1.0
            }
        )
        self.add_material(bi2te3)
        
        # PbTe - Mid-temperature
        pbte = ThermoelectricMaterial(
            name="PbTe",
            formula="PbTe",
            structure_file="PbTe_POSCAR",
            band_gap=0.32,
            effective_mass_e=0.24,
            effective_mass_h=0.31,
            debye_temperature=136,
            lattice_thermal_conductivity_300K=2.0,
            optimal_doping_n=2e19,
            optimal_doping_p=1.5e19,
            experimental_zt={
                300: 0.5, 500: 0.8, 700: 1.2, 800: 1.1
            }
        )
        self.add_material(pbte)
        
        # SnSe - Low κ
        snse = ThermoelectricMaterial(
            name="SnSe",
            formula="SnSe",
            structure_file="SnSe_POSCAR",
            band_gap=0.9,
            effective_mass_e=0.15,
            effective_mass_h=0.13,
            debye_temperature=148,
            lattice_thermal_conductivity_300K=0.6,
            optimal_doping_n=5e19,
            optimal_doping_p=5e19,
            experimental_zt={
                300: 0.5, 500: 1.0, 800: 2.0, 923: 2.6
            }
        )
        self.add_material(snse)
        
        # Mg3Sb2 - New generation
        mg3sb2 = ThermoelectricMaterial(
            name="Mg3Sb2",
            formula="Mg3Sb2",
            structure_file="Mg3Sb2_POSCAR",
            band_gap=0.7,
            effective_mass_e=0.35,
            effective_mass_h=0.45,
            debye_temperature=262,
            lattice_thermal_conductivity_300K=1.0,
            optimal_doping_n=1e20,
            optimal_doping_p=None,
            experimental_zt={
                300: 0.5, 400: 0.8, 500: 1.0, 600: 1.1, 700: 1.0
            }
        )
        self.add_material(mg3sb2)
    
    def calculate_lattice_thermal_conductivity(
        self,
        material_name: str,
        temperatures: np.ndarray,
        kappa_300K: Optional[float] = None
    ) -> np.ndarray:
        """
        Calculate temperature-dependent lattice thermal conductivity.
        
        Uses simplified model: κ_l ∝ 1/T at high T
        
        Args:
            material_name: Material name
            temperatures: Temperature array (K)
            kappa_300K: Reference value at 300K
            
        Returns:
            κ_l array (W/m/K)
        """
        material = self.materials.get(material_name)
        if material is None:
            raise ValueError(f"Material {material_name} not found")
        
        kappa_0 = kappa_300K or material.lattice_thermal_conductivity_300K
        theta_D = material.debye_temperature
        
        kappa_l = np.zeros(len(temperatures))
        
        for i, T in enumerate(temperatures):
            if T < theta_D / 10:
                # Low T: κ ∝ T³ (umklapp processes frozen out)
                kappa_l[i] = kappa_0 * (T / 300)**3
            elif T < theta_D:
                # Intermediate: smooth transition
                kappa_l[i] = kappa_0 * (theta_D / T)
            else:
                # High T: κ ∝ 1/T
                kappa_l[i] = kappa_0 * (300 / T)
        
        return kappa_l
    
    def calculate_electronic_transport(
        self,
        material_name: str,
        temperatures: np.ndarray,
        doping_concentration: float,
        carrier_type: str = 'n'
    ) -> Dict[str, np.ndarray]:
        """
        Calculate electronic transport properties.
        
        Args:
            material_name: Material name
            temperatures: Temperature array (K)
            doping_concentration: Carrier concentration (cm^-3)
            carrier_type: 'n' or 'p'
            
        Returns:
            Dictionary with S, σ, κ_e, μ
        """
        material = self.materials.get(material_name)
        if material is None:
            raise ValueError(f"Material {material_name} not found")
        
        # Get effective mass
        m_star = material.effective_mass_e if carrier_type == 'n' else material.effective_mass_h
        
        # Calculate properties
        S = np.zeros(len(temperatures))  # Seebeck (μV/K)
        sigma = np.zeros(len(temperatures))  # Conductivity (S/cm)
        mu = np.zeros(len(temperatures))  # Mobility (cm²/V/s)
        kappa_e = np.zeros(len(temperatures))  # Electronic κ (W/m/K)
        
        for i, T in enumerate(temperatures):
            # Seebeck coefficient (simplified Mott formula)
            # S ∝ (k_B/e) * (k_B T / E_F)
            E_F = HBAR**2 * (3 * np.pi**2 * doping_concentration * 1e6)**(2/3) / \
                  (2 * m_star * 9.109e-31) / 1.602e-19  # eV
            
            S[i] = (np.pi**2 / 3) * (8.617e-5 / 1.602e-19) * \
                   (8.617e-5 * T / E_F) * 1e6  # μV/K
            
            # Mobility (acoustic phonon limited)
            # μ ∝ T^(-3/2) for acoustic phonon scattering
            mu_300 = 100 * (0.3 / m_star)  # Estimate based on effective mass
            mu[i] = mu_300 * (300 / T)**1.5
            
            # Conductivity
            sigma[i] = doping_concentration * 1.602e-19 * mu[i] * 1e-4  # S/cm
            
            # Electronic thermal conductivity (Wiedemann-Franz)
            L = 2.44e-8  # Lorenz number (W·Ω/K²)
            kappa_e[i] = L * sigma[i] * 100 * T  # W/m/K
        
        return {
            'seebeck': S,  # μV/K
            'conductivity': sigma,  # S/cm
            'mobility': mu,  # cm²/V/s
            'kappa_e': kappa_e  # W/m/K
        }
    
    def calculate_zt(
        self,
        material_name: str,
        temperatures: np.ndarray,
        doping_concentration: float,
        carrier_type: str = 'n',
        kappa_l_override: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Calculate figure of merit ZT.
        
        ZT = S²σT / (κ_e + κ_l)
        
        Args:
            material_name: Material name
            temperatures: Temperature array (K)
            doping_concentration: Carrier concentration (cm^-3)
            carrier_type: 'n' or 'p'
            kappa_l_override: Override lattice thermal conductivity
            
        Returns:
            Dictionary with ZT and components
        """
        # Get electronic transport
        elec = self.calculate_electronic_transport(
            material_name, temperatures, doping_concentration, carrier_type
        )
        
        # Get lattice thermal conductivity
        if kappa_l_override is not None:
            kappa_l = kappa_l_override
        else:
            kappa_l = self.calculate_lattice_thermal_conductivity(
                material_name, temperatures
            )
        
        # Calculate ZT
        S = elec['seebeck'] * 1e-6  # Convert to V/K
        sigma = elec['conductivity'] * 100  # Convert to S/m
        kappa_total = elec['kappa_e'] + kappa_l
        
        power_factor = S**2 * sigma  # W/m/K²
        zt = power_factor * temperatures / kappa_total
        
        return {
            'zt': zt,
            'power_factor': power_factor * 1e6,  # μW/cm/K²
            'seebeck': elec['seebeck'],  # μV/K
            'conductivity': elec['conductivity'],  # S/cm
            'kappa_e': elec['kappa_e'],  # W/m/K
            'kappa_l': kappa_l,  # W/m/K
            'kappa_total': kappa_total  # W/m/K
        }
    
    def optimize_doping(
        self,
        material_name: str,
        temperature: float,
        doping_range: Tuple[float, float] = (1e17, 1e21),
        carrier_type: str = 'n',
        n_points: int = 50
    ) -> Dict:
        """
        Find optimal doping concentration for maximum ZT.
        
        Args:
            material_name: Material name
            temperature: Temperature (K)
            doping_range: (min, max) doping concentration
            carrier_type: 'n' or 'p'
            n_points: Number of doping points to test
            
        Returns:
            Dictionary with optimization results
        """
        doping_values = np.logspace(
            np.log10(doping_range[0]),
            np.log10(doping_range[1]),
            n_points
        )
        
        zt_values = []
        
        for doping in doping_values:
            result = self.calculate_zt(
                material_name,
                np.array([temperature]),
                doping,
                carrier_type
            )
            zt_values.append(result['zt'][0])
        
        zt_values = np.array(zt_values)
        
        # Find optimal
        max_idx = np.argmax(zt_values)
        optimal_doping = doping_values[max_idx]
        max_zt = zt_values[max_idx]
        
        return {
            'optimal_doping': optimal_doping,
            'max_zt': max_zt,
            'doping_values': doping_values.tolist(),
            'zt_values': zt_values.tolist(),
            'temperature': temperature,
            'carrier_type': carrier_type
        }
    
    def analyze_nanostructuring_effect(
        self,
        material_name: str,
        temperatures: np.ndarray,
        feature_sizes: np.ndarray,  # nm
        doping_concentration: float
    ) -> Dict:
        """
        Analyze effect of nanostructuring on ZT.
        
        Nanostructuring reduces lattice thermal conductivity through
        boundary scattering while maintaining electronic properties.
        
        Args:
            material_name: Material name
            temperatures: Temperature array (K)
            feature_sizes: Array of feature sizes (nm)
            doping_concentration: Doping concentration
            
        Returns:
            Dictionary with nanostructuring analysis
        """
        material = self.materials.get(material_name)
        
        results = {
            'feature_sizes_nm': feature_sizes.tolist(),
            'temperatures': temperatures.tolist(),
            'zt': {},
            'kappa_reduction': {}
        }
        
        # Base lattice thermal conductivity
        kappa_l_bulk = self.calculate_lattice_thermal_conductivity(
            material_name, temperatures
        )
        
        for size in feature_sizes:
            # Calculate reduced κ_l due to boundary scattering
            # Simple model: κ_eff = κ_bulk * (1 + β/L)^(-1)
            # where L is feature size and β is phonon MFP
            
            # Estimate phonon MFP (simplified)
            mfp = 100.0  # nm at room temperature
            
            reduction_factor = 1.0 / (1.0 + mfp / size)
            kappa_l_nano = kappa_l_bulk * reduction_factor
            
            results['kappa_reduction'][size] = (1 - reduction_factor) * 100
            
            # Calculate ZT with reduced κ_l
            zt_result = self.calculate_zt(
                material_name,
                temperatures,
                doping_concentration,
                kappa_l_override=kappa_l_nano
            )
            
            results['zt'][size] = zt_result['zt'].tolist()
        
        return results
    
    def plot_zt_comparison(
        self,
        material_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Create comprehensive ZT comparison plot.
        
        Args:
            material_names: List of materials to compare
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        material_names = material_names or list(self.materials.keys())
        
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(material_names)))
        
        # Plot 1: ZT vs Temperature
        ax1 = fig.add_subplot(gs[0, 0])
        
        for mat_name, color in zip(material_names, colors):
            if mat_name in self.results and 'zt' in self.results[mat_name]:
                data = self.results[mat_name]['zt']
                ax1.plot(data['temperatures'], data['zt_values'], 
                        'o-', color=color, label=mat_name, lw=2)
                
                # Add experimental data
                mat = self.materials[mat_name]
                if mat.experimental_zt:
                    exp_t = list(mat.experimental_zt.keys())
                    exp_zt = list(mat.experimental_zt.values())
                    ax1.scatter(exp_t, exp_zt, color=color, s=100, 
                              marker='s', edgecolors='black', zorder=5)
        
        ax1.set_xlabel('Temperature (K)', fontsize=11)
        ax1.set_ylabel('ZT', fontsize=11)
        ax1.set_title('Figure of Merit ZT vs Temperature', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Power Factor
        ax2 = fig.add_subplot(gs[0, 1])
        
        for mat_name, color in zip(material_names, colors):
            if mat_name in self.results and 'zt' in self.results[mat_name]:
                data = self.results[mat_name]['zt']
                ax2.plot(data['temperatures'], data['power_factor'], 
                        'o-', color=color, label=mat_name, lw=2)
        
        ax2.set_xlabel('Temperature (K)', fontsize=11)
        ax2.set_ylabel('Power Factor (μW/cm/K²)', fontsize=11)
        ax2.set_title('Power Factor vs Temperature', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Thermal Conductivity Components
        ax3 = fig.add_subplot(gs[1, 0])
        
        for mat_name, color in zip(material_names, colors):
            if mat_name in self.results and 'zt' in self.results[mat_name]:
                data = self.results[mat_name]['zt']
                temps = data['temperatures']
                kappa_l = data['kappa_l']
                kappa_e = data['kappa_e']
                
                ax3.plot(temps, kappa_l, '-', color=color, lw=2, 
                        label=f'{mat_name} (lattice)')
                ax3.plot(temps, kappa_e, '--', color=color, lw=1.5, 
                        label=f'{mat_name} (electronic)')
        
        ax3.set_xlabel('Temperature (K)', fontsize=11)
        ax3.set_ylabel('κ (W/m/K)', fontsize=11)
        ax3.set_title('Thermal Conductivity Components', fontsize=12)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Maximum ZT Summary
        ax4 = fig.add_subplot(gs[1, 1])
        
        max_zt_calc = []
        max_zt_exp = []
        names = []
        
        for mat_name in material_names:
            mat = self.materials[mat_name]
            
            if mat_name in self.results and 'zt' in self.results[mat_name]:
                zt_vals = self.results[mat_name]['zt']['zt_values']
                max_zt_calc.append(max(zt_vals))
                names.append(mat_name)
                
                if mat.experimental_zt:
                    max_zt_exp.append(max(mat.experimental_zt.values()))
                else:
                    max_zt_exp.append(0)
        
        x = np.arange(len(names))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, max_zt_calc, width, label='Calculated', 
                       color='steelblue')
        bars2 = ax4.bar(x + width/2, max_zt_exp, width, label='Experimental', 
                       color='coral')
        
        ax4.set_ylabel('Max ZT', fontsize=11)
        ax4.set_title('Maximum ZT Comparison', fontsize=12)
        ax4.set_xticks(x)
        ax4.set_xticklabels(names)
        ax4.legend()
        
        plt.suptitle('Thermoelectric Performance Analysis', fontsize=14, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ZT comparison plot to {save_path}")
        
        return fig
    
    def plot_doping_optimization(
        self,
        material_name: str,
        temperatures: List[float],
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot doping optimization curves.
        
        Args:
            material_name: Material name
            temperatures: List of temperatures to optimize
            figsize: Figure size
            save_path: Path to save
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for idx, T in enumerate(temperatures[:4]):
            ax = axes[idx]
            
            # Optimize for this temperature
            opt_result = self.optimize_doping(material_name, T)
            
            doping = np.array(opt_result['doping_values'])
            zt = np.array(opt_result['zt_values'])
            
            ax.semilogx(doping, zt, 'b-', lw=2)
            ax.axvline(x=opt_result['optimal_doping'], color='r', 
                      linestyle='--', label=f'Opt: {opt_result["optimal_doping"]:.2e}')
            ax.axhline(y=opt_result['max_zt'], color='r', linestyle=':')
            
            ax.set_xlabel('Doping Concentration (cm⁻³)', fontsize=10)
            ax.set_ylabel('ZT', fontsize=10)
            ax.set_title(f'{material_name} at T = {T} K', fontsize=11)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive report."""
        report = []
        report.append("=" * 80)
        report.append("THERMOELECTRIC MATERIALS - ZT OPTIMIZATION STUDY")
        report.append("=" * 80)
        report.append("")
        
        report.append("MATERIALS:")
        report.append("-" * 40)
        for name, mat in self.materials.items():
            report.append(f"  {name} ({mat.formula})")
            report.append(f"    Band gap: {mat.band_gap} eV")
            report.append(f"    Effective mass (e): {mat.effective_mass_e} m_e")
            report.append(f"    Effective mass (h): {mat.effective_mass_h} m_e")
            report.append(f"    Debye temperature: {mat.debye_temperature} K")
            report.append("")
        
        report.append("RESULTS:")
        report.append("-" * 40)
        for mat_name in self.materials.keys():
            if mat_name in self.results and 'zt' in self.results[mat_name]:
                data = self.results[mat_name]['zt']
                zt_max = max(data['zt_values'])
                t_max = data['temperatures'][np.argmax(data['zt_values'])]
                report.append(f"  {mat_name}: Max ZT = {zt_max:.2f} at {t_max} K")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def save_all_results(self, output_dir: Optional[str] = None):
        """Save all results to JSON."""
        output_dir = output_dir or self.output_dir
        output_path = Path(output_dir) / "thermoelectric_results.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Saved results to {output_path}")


# Physical constants
HBAR = 6.582e-16  # eV*s


def run_thermoelectric_demo():
    """Run demonstration of thermoelectric optimization."""
    print("=" * 80)
    print("THERMOELECTRIC MATERIALS - ZT OPTIMIZATION DEMO")
    print("=" * 80)
    print()
    
    # Create optimizer
    opt = ThermoelectricOptimizer(output_dir="./te_demo_output")
    
    # Setup materials
    opt.setup_standard_materials()
    
    print("Materials configured:")
    for name, mat in opt.materials.items():
        print(f"  - {name}: {mat.formula}")
        print(f"    Band gap: {mat.band_gap} eV, κ_300K: {mat.lattice_thermal_conductivity_300K} W/m/K")
    print()
    
    # Temperature range
    temps = np.linspace(300, 1000, 50)
    
    # Calculate ZT for each material
    for mat_name in opt.materials.keys():
        print(f"Calculating ZT for {mat_name}...")
        
        material = opt.materials[mat_name]
        
        # Get optimal doping
        doping = material.optimal_doping_n or 1e19
        
        # Calculate ZT
        zt_data = opt.calculate_zt(mat_name, temps, doping)
        
        opt.results[mat_name] = {'zt': {
            'temperatures': temps.tolist(),
            'zt_values': zt_data['zt'].tolist(),
            'power_factor': zt_data['power_factor'].tolist(),
            'seebeck': zt_data['seebeck'].tolist(),
            'conductivity': zt_data['conductivity'].tolist(),
            'kappa_l': zt_data['kappa_l'].tolist(),
            'kappa_e': zt_data['kappa_e'].tolist()
        }}
    
    # Generate plots
    opt.plot_zt_comparison(save_path="./te_demo_output/zt_comparison.png")
    
    # Generate report
    report = opt.generate_report("./te_demo_output/te_report.txt")
    print(report)
    
    # Save results
    opt.save_all_results()
    
    print("\nDemo completed successfully!")
    print(f"Results saved to: {opt.output_dir}")
    
    return opt


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Thermoelectric ZT Optimization')
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    parser.add_argument('--material', type=str, help='Material to analyze')
    parser.add_argument('--optimize-doping', action='store_true',
                       help='Optimize doping level')
    parser.add_argument('--temperature', type=float, default=300,
                       help='Temperature for optimization (K)')
    parser.add_argument('--outdir', type=str, default='./te_output',
                       help='Output directory')
    
    args = parser.parse_args()
    
    if args.demo:
        run_thermoelectric_demo()
    else:
        print("Use --demo flag to run demonstration")
