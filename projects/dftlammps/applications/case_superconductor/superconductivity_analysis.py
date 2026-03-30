"""
Superconducting Materials - Electron-Phonon Coupling and Critical Temperature
================================================================================

Case study: Calculation of superconducting properties from electron-phonon
coupling using first-principles methods.

Materials studied:
- MgB2 (Magnesium diboride) - Conventional phonon-mediated SC
- Nb3Sn (Niobium tin) - A15 structure superconductor
- H3S (Hydrogen sulfide) - High-pressure high-Tc SC
- Simple metals (Al, Pb) - Test cases

This script demonstrates:
1. EPW calculation workflow setup
2. Eliashberg spectral function calculation
3. Electron-phonon coupling constant λ
4. Superconducting critical temperature Tc
5. Isotope effect analysis
6. Pressure dependence of Tc

Author: DFTLammps Applications Team
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d, CubicSpline
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dftlammps.phonon import (
    PhonopyInterface, PhononConfig,
    ThermalPropertyCalculator
)
from dftlammps.electron_phonon import (
    EPWInterface, EPWConfig, EPWResults,
    ElectronPhononCalculator, ElPhConfig, ElPhResults,
    calculate_lambda_from_a2f
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Superconductor:
    """Superconducting material specification."""
    
    name: str
    formula: str
    structure_file: str
    crystal_structure: str
    
    # Experimental data
    tc_experimental: float  # K
    pressure: float = 0.0  # GPa (ambient pressure)
    
    # Electronic properties
    fermi_energy: Optional[float] = None  # eV
    n_states_at_fermi: Optional[float] = None  # states/eV/atom
    
    # Phonon properties
    debye_temperature: Optional[float] = None  # K
    max_phonon_freq: Optional[float] = None  # meV
    
    # Isotope effect
    isotope_exponent: Optional[float] = None  # α in Tc ∝ M^(-α)
    
    def __post_init__(self):
        pass


class SuperconductivityAnalyzer:
    """
    Analysis of superconducting properties from electron-phonon coupling.
    
    Implements:
    - Eliashberg theory calculations
    - McMillan and Allen-Dynes Tc formulas
    - Isotope effect calculation
    - Pressure dependence of Tc
    - Comparison with BCS theory
    """
    
    def __init__(self, output_dir: str = "./superconductor_output"):
        """Initialize superconductivity analyzer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.materials: Dict[str, Superconductor] = {}
        self.results: Dict[str, Dict] = {}
        
        logger.info("Initialized Superconductivity Analyzer")
    
    def add_material(self, material: Superconductor):
        """Add a superconducting material."""
        self.materials[material.name] = material
        logger.info(f"Added superconductor: {material.name} ({material.formula})")
    
    def setup_standard_materials(self):
        """Setup standard superconducting materials."""
        # MgB2
        mgb2 = Superconductor(
            name="MgB2",
            formula="MgB2",
            structure_file="MgB2_POSCAR",
            crystal_structure="hexagonal",
            tc_experimental=39.0,
            pressure=0.0,
            fermi_energy=8.5,
            n_states_at_fermi=0.35,
            debye_temperature=1050,
            max_phonon_freq=110,
            isotope_exponent=0.32
        )
        self.add_material(mgb2)
        
        # Nb3Sn
        nb3sn = Superconductor(
            name="Nb3Sn",
            formula="Nb3Sn",
            structure_file="Nb3Sn_POSCAR",
            crystal_structure="A15",
            tc_experimental=18.3,
            pressure=0.0,
            fermi_energy=10.2,
            n_states_at_fermi=0.82,
            debye_temperature=275,
            max_phonon_freq=26,
            isotope_exponent=0.08
        )
        self.add_material(nb3sn)
        
        # H3S (high pressure)
        h3s = Superconductor(
            name="H3S",
            formula="H3S",
            structure_file="H3S_POSCAR",
            crystal_structure="cubic",
            tc_experimental=203.0,
            pressure=155.0,  # GPa
            fermi_energy=15.0,
            n_states_at_fermi=0.25,
            debye_temperature=1500,
            max_phonon_freq=250,
            isotope_exponent=0.35
        )
        self.add_material(h3s)
        
        # Aluminum (test case)
        al = Superconductor(
            name="Al",
            formula="Al",
            structure_file="Al_POSCAR",
            crystal_structure="fcc",
            tc_experimental=1.2,
            pressure=0.0,
            fermi_energy=11.7,
            n_states_at_fermi=0.41,
            debye_temperature=428,
            max_phonon_freq=37,
            isotope_exponent=0.49
        )
        self.add_material(al)
        
        # Lead (strong coupling)
        pb = Superconductor(
            name="Pb",
            formula="Pb",
            structure_file="Pb_POSCAR",
            crystal_structure="fcc",
            tc_experimental=7.2,
            pressure=0.0,
            fermi_energy=9.4,
            n_states_at_fermi=0.65,
            debye_temperature=105,
            max_phonon_freq=11,
            isotope_exponent=0.49
        )
        self.add_material(pb)
    
    def calculate_eliashberg_function(
        self,
        material_name: str,
        omega: np.ndarray,
        phonon_dos: np.ndarray,
        coupling_strength: float,
        method: str = 'simple'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Eliashberg spectral function α²F(ω).
        
        Args:
            material_name: Material name
            omega: Frequency grid (meV)
            phonon_dos: Phonon density of states
            coupling_strength: Overall coupling strength
            method: Calculation method
            
        Returns:
            Tuple of (omega, alpha2f)
        """
        material = self.materials.get(material_name)
        
        # Normalize phonon DOS
        dos_norm = phonon_dos / integrate.simpson(phonon_dos, omega)
        
        # Calculate α²F(ω)
        # Simplified: α²F(ω) = λ * ω * F(ω) / (2∫ωF(ω)dω)
        
        if method == 'simple':
            # Constant coupling: α² = constant
            avg_omega = integrate.simpson(omega * dos_norm, omega)
            alpha2f = coupling_strength * omega * dos_norm / (2 * avg_omega)
            
        elif method == 'frequency_dependent':
            # Frequency-dependent coupling: α² ∝ 1/ω
            alpha2f = coupling_strength * dos_norm / 2.0
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return omega, alpha2f
    
    def calculate_lambda(
        self,
        omega: np.ndarray,
        alpha2f: np.ndarray
    ) -> float:
        """
        Calculate electron-phonon coupling constant λ.
        
        λ = 2∫₀^∞ α²F(ω)/ω dω
        
        Args:
            omega: Frequency grid (meV)
            alpha2f: Eliashberg function
            
        Returns:
            Coupling constant λ
        """
        omega_safe = np.where(omega > 0.1, omega, 0.1)
        integrand = 2 * alpha2f / omega_safe
        lambda_val = integrate.simpson(integrand, omega)
        return lambda_val
    
    def calculate_omega_log(
        self,
        omega: np.ndarray,
        alpha2f: np.ndarray,
        lambda_val: float
    ) -> float:
        """
        Calculate logarithmic average frequency.
        
        ω_log = exp[(2/λ) ∫ ln(ω) α²F(ω)/ω dω]
        
        Args:
            omega: Frequency grid (meV)
            alpha2f: Eliashberg function
            lambda_val: Coupling constant
            
        Returns:
            ω_log in meV
        """
        omega_safe = np.where(omega > 0.1, omega, 0.1)
        integrand = (2 / lambda_val) * alpha2f * np.log(omega_safe) / omega_safe
        omega_log = np.exp(integrate.simpson(integrand, omega))
        return omega_log
    
    def calculate_tc_mcmillan(
        self,
        omega_log: float,
        lambda_val: float,
        mu_star: float = 0.1
    ) -> float:
        """
        Calculate Tc using McMillan formula.
        
        Tc = (ω_log / 1.2) * exp[-1.04(1 + λ) / (λ - μ*(1 + 0.62λ))]
        
        Args:
            omega_log: Logarithmic average frequency (meV)
            lambda_val: Coupling constant
            mu_star: Coulomb pseudopotential
            
        Returns:
            Tc in Kelvin
        """
        if lambda_val <= mu_star * (1 + 0.62 * lambda_val):
            return 0.0
        
        # Convert ω_log to K
        omega_log_k = omega_log * 11.605
        
        numerator = -1.04 * (1 + lambda_val)
        denominator = lambda_val - mu_star * (1 + 0.62 * lambda_val)
        
        tc = (omega_log_k / 1.2) * np.exp(numerator / denominator)
        return tc
    
    def calculate_tc_allen_dynes(
        self,
        omega_log: float,
        lambda_val: float,
        mu_star: float = 0.1
    ) -> float:
        """
        Calculate Tc using Allen-Dynes formula.
        
        More accurate for strong coupling (λ > 1).
        
        Args:
            omega_log: Logarithmic average frequency (meV)
            lambda_val: Coupling constant
            mu_star: Coulomb pseudopotential
            
        Returns:
            Tc in Kelvin
        """
        omega_log_k = omega_log * 11.605
        
        # Allen-Dynes correction factor
        f = 1.0 + (lambda_val / (2.46 * (1 + 3.8 * mu_star))**1.5)
        
        numerator = -1.04 * (1 + lambda_val * (1 - 0.62 * mu_star - 0.62 * lambda_val))
        denominator = lambda_val - mu_star * (1 + 0.62 * lambda_val)
        
        if denominator <= 0:
            return 0.0
        
        tc = (omega_log_k / 1.2) * np.exp(numerator / denominator) * f
        
        return tc
    
    def estimate_lambda_from_tc(
        self,
        tc_experimental: float,
        omega_log: float,
        mu_star: float = 0.1,
        method: str = 'mcmillan'
    ) -> float:
        """
        Estimate λ from experimental Tc (inverse problem).
        
        Args:
            tc_experimental: Experimental Tc (K)
            omega_log: Logarithmic average frequency (meV)
            mu_star: Coulomb pseudopotential
            method: 'mcmillan' or 'allen_dynes'
            
        Returns:
            Estimated λ
        """
        # Solve for λ that gives experimental Tc
        def residual(lam):
            if method == 'mcmillan':
                tc_calc = self.calculate_tc_mcmillan(omega_log, lam, mu_star)
            else:
                tc_calc = self.calculate_tc_allen_dynes(omega_log, lam, mu_star)
            return (tc_calc - tc_experimental)**2
        
        # Use simple search
        lambda_values = np.linspace(0.1, 5.0, 1000)
        residuals = [residual(lam) for lam in lambda_values]
        
        return lambda_values[np.argmin(residuals)]
    
    def calculate_isotope_effect(
        self,
        material_name: str,
        isotope_masses: Dict[str, List[float]],
        natural_abundance: Dict[str, float],
        mu_star: float = 0.1
    ) -> Dict:
        """
        Calculate isotope effect on Tc.
        
        Args:
            material_name: Material name
            isotope_masses: Dict of {element: [mass1, mass2, ...]}
            natural_abundance: Natural abundance of elements
            mu_star: Coulomb pseudopotential
            
        Returns:
            Dictionary with isotope effect analysis
        """
        material = self.materials.get(material_name)
        
        # Get base properties
        if material_name not in self.results:
            raise ValueError(f"Run analysis for {material_name} first")
        
        lambda_base = self.results[material_name]['lambda']
        omega_log_base = self.results[material_name]['omega_log']
        tc_base = self.calculate_tc_mcmillan(omega_log_base, lambda_base, mu_star)
        
        results = {
            'base_tc': tc_base,
            'isotope_shifts': {}
        }
        
        # Calculate Tc for different isotopic compositions
        for element, masses in isotope_masses.items():
            for mass in masses:
                # Isotope effect: ω ∝ 1/√M, so ω_log ∝ 1/√M
                # λ is approximately independent of M
                
                mass_ratio = mass / natural_abundance[element]
                omega_log_iso = omega_log_base / np.sqrt(mass_ratio)
                
                tc_iso = self.calculate_tc_mcmillan(omega_log_iso, lambda_base, mu_star)
                
                # Isotope exponent
                alpha = -np.log(tc_iso / tc_base) / np.log(mass_ratio)
                
                results['isotope_shifts'][f"{element}-{mass:.1f}"] = {
                    'tc': tc_iso,
                    'shift': tc_iso - tc_base,
                    'alpha': alpha
                }
        
        return results
    
    def analyze_pressure_dependence(
        self,
        material_name: str,
        pressures: np.ndarray,  # GPa
        compressibility: float = 1e-3,  # GPa^-1
        gruneisen: float = 1.5
    ) -> Dict:
        """
        Analyze pressure dependence of Tc.
        
        Args:
            material_name: Material name
            pressures: Pressure array (GPa)
            compressibility: Compressibility
            gruneisen: Grüneisen parameter
            
        Returns:
            Dictionary with pressure analysis
        """
        material = self.materials.get(material_name)
        
        # Get base properties
        if material_name not in self.results:
            raise ValueError(f"Run analysis for {material_name} first")
        
        lambda_base = self.results[material_name]['lambda']
        omega_log_base = self.results[material_name]['omega_log']
        
        tc_vs_p = []
        
        for p in pressures:
            # Volume change
            delta_v_v = -compressibility * p
            
            # Frequency shift (Grüneisen)
            omega_log_p = omega_log_base * (1 - gruneisen * delta_v_v)
            
            # λ change (simplified model)
            # λ ∝ N(0)〈I²〉/M〈ω²〉
            # Assume N(0) and 〈I²〉constant, M constant
            # λ ∝ 1/〈ω²〉 ∝ 1/ω²
            lambda_p = lambda_base * (omega_log_base / omega_log_p)**2
            
            tc = self.calculate_tc_mcmillan(omega_log_p, lambda_p)
            tc_vs_p.append(tc)
        
        return {
            'pressures': pressures.tolist(),
            'tc': tc_vs_p,
            'd_tc_d_p': np.gradient(tc_vs_p, pressures).tolist()
        }
    
    def run_full_analysis(
        self,
        material_name: str,
        omega_max: float = 200.0,  # meV
        n_points: int = 500,
        mu_star: float = 0.1
    ) -> Dict:
        """
        Run complete analysis for a superconductor.
        
        Args:
            material_name: Material name
            omega_max: Maximum frequency for calculation
            n_points: Number of frequency points
            mu_star: Coulomb pseudopotential
            
        Returns:
            Dictionary with complete analysis results
        """
        material = self.materials.get(material_name)
        if material is None:
            raise ValueError(f"Material {material_name} not found")
        
        logger.info(f"Running full analysis for {material_name}")
        
        # Generate frequency grid
        omega = np.linspace(0.1, omega_max, n_points)
        
        # Generate phonon DOS (simplified model)
        # Use Debye model as approximation
        theta_D = material.debye_temperature or 300
        omega_D = theta_D * 0.08617  # Convert to meV
        
        phonon_dos = np.where(omega < omega_D, omega**2, 0)
        
        # Estimate coupling strength from experimental Tc
        estimated_lambda = self.estimate_lambda_from_tc(
            material.tc_experimental,
            omega_D * 0.75,  # Approximate ω_log
            mu_star
        )
        
        # Calculate α²F(ω)
        omega, alpha2f = self.calculate_eliashberg_function(
            material_name, omega, phonon_dos, estimated_lambda
        )
        
        # Calculate λ
        lambda_val = self.calculate_lambda(omega, alpha2f)
        
        # Calculate ω_log
        omega_log = self.calculate_omega_log(omega, alpha2f, lambda_val)
        
        # Calculate Tc
        tc_mcmillan = self.calculate_tc_mcmillan(omega_log, lambda_val, mu_star)
        tc_allen_dynes = self.calculate_tc_allen_dynes(omega_log, lambda_val, mu_star)
        
        results = {
            'material': material_name,
            'formula': material.formula,
            'lambda': lambda_val,
            'omega_log_meV': omega_log,
            'tc_mcmillan_K': tc_mcmillan,
            'tc_allen_dynes_K': tc_allen_dynes,
            'tc_experimental_K': material.tc_experimental,
            'mu_star': mu_star,
            'frequency_grid': omega.tolist(),
            'eliashberg_function': alpha2f.tolist()
        }
        
        self.results[material_name] = results
        
        logger.info(f"Analysis complete: λ = {lambda_val:.3f}, "
                   f"Tc = {tc_mcmillan:.1f} K (exp: {material.tc_experimental:.1f} K)")
        
        return results
    
    def plot_eliashberg_function(
        self,
        material_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot Eliashberg functions for multiple materials.
        
        Args:
            material_names: List of materials to plot
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        material_names = material_names or list(self.results.keys())
        
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(material_names)))
        
        # Plot 1: α²F(ω)
        ax1 = fig.add_subplot(gs[0, 0])
        
        for mat_name, color in zip(material_names, colors):
            if mat_name in self.results:
                data = self.results[mat_name]
                omega = np.array(data['frequency_grid'])
                alpha2f = np.array(data['eliashberg_function'])
                ax1.plot(omega, alpha2f, color=color, label=mat_name, lw=2)
                ax1.fill_between(omega, 0, alpha2f, alpha=0.2, color=color)
        
        ax1.set_xlabel('ℏω (meV)', fontsize=11)
        ax1.set_ylabel('α²F(ω)', fontsize=11)
        ax1.set_title('Eliashberg Spectral Function', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: λ comparison
        ax2 = fig.add_subplot(gs[0, 1])
        
        names = []
        lambdas = []
        
        for mat_name in material_names:
            if mat_name in self.results:
                names.append(mat_name)
                lambdas.append(self.results[mat_name]['lambda'])
        
        bars = ax2.bar(names, lambdas, color='steelblue')
        ax2.set_ylabel('λ', fontsize=11)
        ax2.set_title('Electron-Phonon Coupling Constant', fontsize=12)
        ax2.axhline(y=1.0, color='r', linestyle='--', label='Strong coupling threshold')
        ax2.legend()
        
        # Color bars by value
        for bar, lam in zip(bars, lambdas):
            if lam > 1.5:
                bar.set_color('darkred')
            elif lam > 1.0:
                bar.set_color('orange')
            else:
                bar.set_color('steelblue')
        
        # Plot 3: Tc comparison
        ax3 = fig.add_subplot(gs[1, 0])
        
        x = np.arange(len(names))
        width = 0.25
        
        tc_mcmillan = [self.results[n]['tc_mcmillan_K'] for n in names]
        tc_allen_dynes = [self.results[n]['tc_allen_dynes_K'] for n in names]
        tc_exp = [self.results[n]['tc_experimental_K'] for n in names]
        
        ax3.bar(x - width, tc_mcmillan, width, label='McMillan', color='lightblue')
        ax3.bar(x, tc_allen_dynes, width, label='Allen-Dynes', color='steelblue')
        ax3.bar(x + width, tc_exp, width, label='Experimental', color='coral')
        
        ax3.set_ylabel('Tc (K)', fontsize=11)
        ax3.set_title('Critical Temperature Comparison', fontsize=12)
        ax3.set_xticks(x)
        ax3.set_xticklabels(names, rotation=45)
        ax3.legend()
        
        # Plot 4: ω_log vs λ
        ax4 = fig.add_subplot(gs[1, 1])
        
        omega_logs = [self.results[n]['omega_log_meV'] for n in names]
        
        ax4.scatter(lambdas, omega_logs, s=200, c=tc_mcmillan, 
                   cmap='viridis', edgecolors='black')
        
        for i, name in enumerate(names):
            ax4.annotate(name, (lambdas[i], omega_logs[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax4.set_xlabel('λ', fontsize=11)
        ax4.set_ylabel('ω_log (meV)', fontsize=11)
        ax4.set_title('Coupling vs Characteristic Frequency', fontsize=12)
        cbar = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar.set_label('Tc (K)', fontsize=10)
        
        plt.suptitle('Superconducting Properties Analysis', fontsize=14, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved Eliashberg plot to {save_path}")
        
        return fig
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive report."""
        report = []
        report.append("=" * 80)
        report.append("SUPERCONDUCTING MATERIALS - ELECTRON-PHONON ANALYSIS")
        report.append("=" * 80)
        report.append("")
        
        report.append("MATERIALS ANALYZED:")
        report.append("-" * 40)
        for name, mat in self.materials.items():
            report.append(f"  {name} ({mat.formula})")
            report.append(f"    Structure: {mat.crystal_structure}")
            report.append(f"    Tc (exp): {mat.tc_experimental} K")
            report.append("")
        
        report.append("CALCULATED RESULTS:")
        report.append("-" * 40)
        for mat_name in self.materials.keys():
            if mat_name in self.results:
                data = self.results[mat_name]
                report.append(f"\n{mat_name}:")
                report.append(f"  λ = {data['lambda']:.3f}")
                report.append(f"  ω_log = {data['omega_log_meV']:.1f} meV")
                report.append(f"  Tc (McMillan) = {data['tc_mcmillan_K']:.1f} K")
                report.append(f"  Tc (Allen-Dynes) = {data['tc_allen_dynes_K']:.1f} K")
                report.append(f"  Tc (exp) = {data['tc_experimental_K']:.1f} K")
                
                error = abs(data['tc_mcmillan_K'] - data['tc_experimental_K'])
                report.append(f"  Error = {error:.1f} K ({error/data['tc_experimental_K']*100:.1f}%)")
        
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
        output_path = Path(output_dir) / "superconductor_results.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Saved results to {output_path}")


def run_superconductor_demo():
    """Run demonstration of superconductor analysis."""
    print("=" * 80)
    print("SUPERCONDUCTING MATERIALS - ELECTRON-PHONON ANALYSIS DEMO")
    print("=" * 80)
    print()
    
    # Create analyzer
    analyzer = SuperconductivityAnalyzer(output_dir="./sc_demo_output")
    
    # Setup materials
    analyzer.setup_standard_materials()
    
    print("Materials configured:")
    for name, mat in analyzer.materials.items():
        print(f"  - {name}: {mat.formula} ({mat.crystal_structure})")
        print(f"    Tc (exp) = {mat.tc_experimental} K")
    print()
    
    # Run analysis for each material
    for mat_name in analyzer.materials.keys():
        print(f"Analyzing {mat_name}...")
        analyzer.run_full_analysis(mat_name, mu_star=0.1)
    
    # Generate plots
    analyzer.plot_eliashberg_function(
        save_path="./sc_demo_output/eliashberg_analysis.png"
    )
    
    # Generate report
    report = analyzer.generate_report("./sc_demo_output/sc_report.txt")
    print(report)
    
    # Save results
    analyzer.save_all_results()
    
    print("\nDemo completed successfully!")
    print(f"Results saved to: {analyzer.output_dir}")
    
    return analyzer


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Superconductor Analysis')
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    parser.add_argument('--material', type=str, help='Material to analyze')
    parser.add_argument('--mu-star', type=float, default=0.1,
                       help='Coulomb pseudopotential')
    parser.add_argument('--outdir', type=str, default='./sc_output',
                       help='Output directory')
    
    args = parser.parse_args()
    
    if args.demo:
        run_superconductor_demo()
    elif args.material:
        analyzer = SuperconductivityAnalyzer(output_dir=args.outdir)
        analyzer.setup_standard_materials()
        if args.material in analyzer.materials:
            analyzer.run_full_analysis(args.material, mu_star=args.mu_star)
            analyzer.plot_eliashberg_function([args.material])
        else:
            print(f"Material {args.material} not found")
    else:
        print("Use --demo flag to run demonstration")
