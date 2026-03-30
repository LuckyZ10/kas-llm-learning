"""
Thermal Barrier Coating (TBC) Materials - Thermal Conductivity Study
======================================================================

Case study: Investigation of lattice thermal conductivity in ceramic
thermal barrier coating materials using first-principles calculations.

Materials studied:
- Yttria-stabilized zirconia (YSZ)
- Rare earth zirconates (RE2Zr2O7)
- Hafnium oxide (HfO2)

This script demonstrates:
1. Structure preparation for TBC materials
2. Phonon calculations with supercell approach
3. Three-phonon scattering calculations (Phono3py)
4. Thermal conductivity analysis vs temperature
5. Point defect and grain boundary scattering effects
6. Comparison with experimental data

Author: DFTLammps Applications Team
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import warnings

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dftlammps.phonon import (
    PhonopyInterface, PhononConfig, DFTCode,
    ThermalPropertyCalculator, ThermalConfig,
    LatticeThermalConductivity, ThermalConductivityConfig, ConductivityMethod
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TBCMaterial:
    """Thermal barrier coating material specification."""
    
    name: str
    formula: str
    structure_file: str
    space_group: str
    
    # Supercell parameters
    supercell_matrix: np.ndarray = field(default_factory=lambda: np.diag([2, 2, 2]))
    
    # DFT parameters
    ecutwfc: float = 60.0  # Ry
    k_mesh: Tuple[int, int, int] = (6, 6, 6)
    
    # Phono3py parameters
    kappa_mesh: Tuple[int, int, int] = (11, 11, 11)
    
    # Material properties
    experimental_kappa: Optional[Dict[float, float]] = None
    melting_point: Optional[float] = None  # K
    
    def __post_init__(self):
        if isinstance(self.supercell_matrix, list):
            self.supercell_matrix = np.array(self.supercell_matrix)


class TBCThermalConductivityStudy:
    """
    Comprehensive study of thermal conductivity in TBC materials.
    
    This class implements a complete workflow for:
    - Structure preparation
    - Force constant calculations
    - Phonon dispersion analysis
    - Thermal conductivity calculations
    - Defect scattering analysis
    """
    
    def __init__(self, output_dir: str = "./tbc_study_output"):
        """Initialize TBC study."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.materials: Dict[str, TBCMaterial] = {}
        self.results: Dict[str, Any] = {}
        
        logger.info("Initialized TBC Thermal Conductivity Study")
    
    def add_material(self, material: TBCMaterial):
        """Add a material to the study."""
        self.materials[material.name] = material
        logger.info(f"Added material: {material.name} ({material.formula})")
    
    def setup_standard_materials(self):
        """Setup standard TBC materials for study."""
        # YSZ (Yttria-stabilized Zirconia)
        ysz = TBCMaterial(
            name="YSZ",
            formula="Zr0.92Y0.08O1.96",
            structure_file="YSZ_POSCAR",
            space_group="Fm-3m",
            supercell_matrix=np.diag([2, 2, 2]),
            ecutwfc=60.0,
            k_mesh=(6, 6, 6),
            kappa_mesh=(11, 11, 11),
            experimental_kappa={
                300: 2.5, 500: 2.1, 800: 1.8, 1000: 1.6, 1200: 1.4
            },
            melting_point=2953
        )
        self.add_material(ysz)
        
        # Gd2Zr2O7 (Gadolinium Zirconate)
        gzo = TBCMaterial(
            name="Gd2Zr2O7",
            formula="Gd2Zr2O7",
            structure_file="Gd2Zr2O7_POSCAR",
            space_group="Fd-3m",
            supercell_matrix=np.diag([2, 2, 2]),
            ecutwfc=60.0,
            k_mesh=(4, 4, 4),
            kappa_mesh=(9, 9, 9),
            experimental_kappa={
                300: 1.6, 500: 1.4, 800: 1.2, 1000: 1.1, 1200: 1.0
            },
            melting_point=2573
        )
        self.add_material(gzo)
        
        # HfO2 (Hafnia)
        hfo2 = TBCMaterial(
            name="HfO2",
            formula="HfO2",
            structure_file="HfO2_POSCAR",
            space_group="P21/c",
            supercell_matrix=np.diag([2, 2, 2]),
            ecutwfc=70.0,
            k_mesh=(6, 6, 6),
            kappa_mesh=(11, 11, 11),
            experimental_kappa={
                300: 1.2, 500: 1.0, 800: 0.9, 1000: 0.8, 1200: 0.75
            },
            melting_point=3033
        )
        self.add_material(hfo2)
    
    def run_phonon_calculation(
        self,
        material_name: str,
        structure_path: Optional[str] = None,
        run_displacements: bool = True,
        run_force_constants: bool = True
    ) -> Dict:
        """
        Run phonon calculation for a material.
        
        Args:
            material_name: Name of the material
            structure_path: Path to structure file
            run_displacements: Generate displacement structures
            run_force_constants: Calculate force constants
            
        Returns:
            Dictionary with phonon results
        """
        material = self.materials.get(material_name)
        if material is None:
            raise ValueError(f"Material {material_name} not found")
        
        structure_path = structure_path or material.structure_file
        
        logger.info(f"Starting phonon calculation for {material_name}")
        
        # Setup phonon configuration
        config = PhononConfig(
            structure_path=structure_path,
            supercell_matrix=material.supercell_matrix,
            dft_code=DFTCode.VASP,
            output_dir=str(self.output_dir / material_name / "phonon")
        )
        
        phonon = PhonopyInterface(config)
        
        results = {}
        
        # Generate displacements
        if run_displacements:
            logger.info("Generating displacement structures...")
            displacements = phonon.create_displacements(
                structure_path,
                distance=0.01,
                is_plusminus='auto'
            )
            results['n_displacements'] = len(displacements)
            logger.info(f"Generated {len(displacements)} displacements")
        
        # Note: Force constants would be calculated from DFT
        # For this example, we'll skip actual DFT calculations
        
        self.results[material_name] = {'phonon': results}
        return results
    
    def run_thermal_conductivity_calculation(
        self,
        material_name: str,
        fc2_path: Optional[str] = None,
        fc3_path: Optional[str] = None,
        temperatures: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Run thermal conductivity calculation using Phono3py.
        
        Args:
            material_name: Name of material
            fc2_path: Path to second-order force constants
            fc3_path: Path to third-order force constants
            temperatures: Temperature array (K)
            
        Returns:
            Dictionary with thermal conductivity results
        """
        material = self.materials.get(material_name)
        if material is None:
            raise ValueError(f"Material {material_name} not found")
        
        temps = temperatures or np.array([300, 400, 500, 600, 800, 1000, 1200])
        
        logger.info(f"Starting thermal conductivity calculation for {material_name}")
        
        # Setup thermal conductivity configuration
        config = ThermalConductivityConfig(
            mesh=material.kappa_mesh,
            temperatures=temps,
            method=ConductivityMethod.RTA,
            include_isotope=True,
            output_dir=str(self.output_dir / material_name / "kappa")
        )
        
        kappa_calc = LatticeThermalConductivity(config)
        
        # Load force constants if provided
        if fc2_path and fc3_path:
            kappa_calc.set_force_constants(
                fc2_path, 
                fc3_path,
                material.structure_file
            )
            
            # Run calculation
            results = kappa_calc.run_thermal_conductivity_rta(
                temperatures=temps
            )
            
            # Save results
            kappa_calc.plot_kappa_vs_temperature(
                save_path=str(self.output_dir / material_name / "kappa_vs_T.png")
            )
            
            kappa_data = {
                'temperatures': results.temperatures.tolist(),
                'kappa_scalar': results.kappa_scalar.tolist() if results.kappa_scalar is not None else [],
                'kappa_tensor': results.kappa_tensor.tolist()
            }
            
            # Store results
            if material_name not in self.results:
                self.results[material_name] = {}
            self.results[material_name]['kappa'] = kappa_data
            
            logger.info(f"Completed κ calculation for {material_name}")
            return kappa_data
        else:
            logger.warning("Force constants not provided - skipping κ calculation")
            return {}
    
    def analyze_defect_scattering(
        self,
        material_name: str,
        defect_concentrations: Dict[str, float],
        grain_size: float = 1e6  # Angstrom (100 µm)
    ) -> Dict:
        """
        Analyze effect of defects on thermal conductivity.
        
        Args:
            material_name: Material name
            defect_concentrations: Dict of {defect_type: concentration}
            grain_size: Grain size for boundary scattering
            
        Returns:
            Dictionary with defect analysis results
        """
        logger.info(f"Analyzing defect scattering for {material_name}")
        
        # Get base thermal conductivity
        if material_name in self.results and 'kappa' in self.results[material_name]:
            kappa_base = np.array(self.results[material_name]['kappa']['kappa_scalar'])
            temps = np.array(self.results[material_name]['kappa']['temperatures'])
        else:
            logger.warning("No base κ data available")
            return {}
        
        results = {
            'base_kappa': kappa_base.tolist(),
            'defects': {}
        }
        
        # Calculate κ reduction for each defect type
        for defect_type, concentration in defect_concentrations.items():
            # Simplified model: κ_defect = κ_base / (1 + A * c)
            # where A depends on defect type
            
            if 'vacancy' in defect_type.lower():
                A = 10.0
            elif 'substitution' in defect_type.lower():
                A = 5.0
            elif 'interstitial' in defect_type.lower():
                A = 15.0
            else:
                A = 8.0
            
            kappa_defect = kappa_base / (1 + A * concentration)
            
            results['defects'][defect_type] = {
                'concentration': concentration,
                'kappa': kappa_defect.tolist(),
                'reduction_percent': ((1 - kappa_defect / kappa_base) * 100).tolist()
            }
        
        # Grain boundary scattering
        # Effect increases at low temperatures
        kappa_gb = kappa_base * np.tanh(temps / 300.0)
        
        results['grain_boundary'] = {
            'grain_size_A': grain_size,
            'kappa': kappa_gb.tolist()
        }
        
        self.results[material_name]['defect_analysis'] = results
        
        return results
    
    def compare_with_experiment(
        self,
        material_name: str
    ) -> Dict:
        """
        Compare calculated thermal conductivity with experimental data.
        
        Args:
            material_name: Material name
            
        Returns:
            Comparison results
        """
        material = self.materials.get(material_name)
        if material is None:
            raise ValueError(f"Material {material_name} not found")
        
        if material.experimental_kappa is None:
            logger.warning(f"No experimental data for {material_name}")
            return {}
        
        if material_name not in self.results or 'kappa' not in self.results[material_name]:
            logger.warning(f"No calculated data for {material_name}")
            return {}
        
        # Get calculated values
        calc_temps = np.array(self.results[material_name]['kappa']['temperatures'])
        calc_kappa = np.array(self.results[material_name]['kappa']['kappa_scalar'])
        
        # Get experimental values
        exp_temps = np.array(list(material.experimental_kappa.keys()))
        exp_kappa = np.array(list(material.experimental_kappa.values()))
        
        # Interpolate calculated to experimental temperatures
        calc_kappa_interp = np.interp(exp_temps, calc_temps, calc_kappa)
        
        # Calculate differences
        diff = calc_kappa_interp - exp_kappa
        percent_error = (diff / exp_kappa) * 100
        
        comparison = {
            'temperatures': exp_temps.tolist(),
            'calculated': calc_kappa_interp.tolist(),
            'experimental': exp_kappa.tolist(),
            'difference': diff.tolist(),
            'percent_error': percent_error.tolist(),
            'mean_absolute_error': float(np.mean(np.abs(percent_error))),
            'rms_error': float(np.sqrt(np.mean(percent_error**2)))
        }
        
        self.results[material_name]['comparison'] = comparison
        
        logger.info(f"Mean absolute error for {material_name}: "
                   f"{comparison['mean_absolute_error']:.2f}%")
        
        return comparison
    
    def plot_comparison_all_materials(
        self,
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Create comprehensive comparison plot for all materials.
        
        Args:
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
        
        materials = list(self.materials.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(materials)))
        
        # Plot 1: κ vs T for all materials
        ax1 = fig.add_subplot(gs[0, 0])
        for mat_name, color in zip(materials, colors):
            if mat_name in self.results and 'kappa' in self.results[mat_name]:
                data = self.results[mat_name]['kappa']
                temps = data['temperatures']
                kappa = data['kappa_scalar']
                ax1.plot(temps, kappa, 'o-', color=color, label=mat_name, lw=2)
                
                # Add experimental data if available
                mat = self.materials[mat_name]
                if mat.experimental_kappa:
                    exp_t = list(mat.experimental_kappa.keys())
                    exp_k = list(mat.experimental_kappa.values())
                    ax1.scatter(exp_t, exp_k, color=color, s=100, marker='s', 
                               edgecolors='black', zorder=5)
        
        ax1.set_xlabel('Temperature (K)', fontsize=11)
        ax1.set_ylabel('κ (W/m/K)', fontsize=11)
        ax1.set_title('Thermal Conductivity vs Temperature', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: κ at 1000K comparison
        ax2 = fig.add_subplot(gs[0, 1])
        mat_names = []
        kappa_1000_calc = []
        kappa_1000_exp = []
        
        for mat_name in materials:
            mat = self.materials[mat_name]
            if mat_name in self.results and 'kappa' in self.results[mat_name]:
                data = self.results[mat_name]['kappa']
                temps = np.array(data['temperatures'])
                kappa = np.array(data['kappa_scalar'])
                k_1000 = np.interp(1000, temps, kappa)
                kappa_1000_calc.append(k_1000)
                mat_names.append(mat_name)
                
                if mat.experimental_kappa and 1000 in mat.experimental_kappa:
                    kappa_1000_exp.append(mat.experimental_kappa[1000])
                else:
                    kappa_1000_exp.append(None)
        
        x = np.arange(len(mat_names))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, kappa_1000_calc, width, label='Calculated', color='steelblue')
        
        exp_values = [v if v is not None else 0 for v in kappa_1000_exp]
        exp_mask = [v is not None for v in kappa_1000_exp]
        bars2 = ax2.bar(x + width/2, exp_values, width, label='Experimental', color='coral')
        
        ax2.set_ylabel('κ at 1000K (W/m/K)', fontsize=11)
        ax2.set_title('Thermal Conductivity Comparison at 1000K', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(mat_names)
        ax2.legend()
        
        # Plot 3: Error analysis
        ax3 = fig.add_subplot(gs[1, 0])
        errors = []
        names_with_error = []
        
        for mat_name in materials:
            if mat_name in self.results and 'comparison' in self.results[mat_name]:
                errors.append(self.results[mat_name]['comparison']['mean_absolute_error'])
                names_with_error.append(mat_name)
        
        if errors:
            bars = ax3.bar(names_with_error, errors, color='lightcoral')
            ax3.set_ylabel('Mean Absolute Error (%)', fontsize=11)
            ax3.set_title('Calculation Accuracy', fontsize=12)
            ax3.axhline(y=20, color='r', linestyle='--', label='20% threshold')
            ax3.legend()
        
        # Plot 4: Summary table
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        table_data = []
        for mat_name in materials:
            mat = self.materials[mat_name]
            row = [
                mat_name,
                mat.formula,
                f"{mat.melting_point} K" if mat.melting_point else "N/A"
            ]
            table_data.append(row)
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Material', 'Formula', 'Melting Point'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Material Properties Summary', fontsize=12)
        
        plt.suptitle('Thermal Barrier Coating Materials: Thermal Conductivity Study', 
                    fontsize=14, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {save_path}")
        
        return fig
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive report of the study.
        
        Args:
            output_file: Path to output file
            
        Returns:
            Report text
        """
        report = []
        report.append("=" * 80)
        report.append("THERMAL BARRIER COATING MATERIALS - THERMAL CONDUCTIVITY STUDY")
        report.append("=" * 80)
        report.append("")
        
        report.append("MATERIALS STUDIED:")
        report.append("-" * 40)
        for name, mat in self.materials.items():
            report.append(f"  {name}: {mat.formula}")
            report.append(f"    Space group: {mat.space_group}")
            report.append(f"    Melting point: {mat.melting_point} K")
            report.append("")
        
        report.append("RESULTS SUMMARY:")
        report.append("-" * 40)
        for mat_name in self.materials.keys():
            if mat_name in self.results:
                report.append(f"\n{mat_name}:")
                
                if 'kappa' in self.results[mat_name]:
                    kappa_data = self.results[mat_name]['kappa']
                    report.append(f"  Thermal conductivity calculated")
                    report.append(f"  Temperature range: {min(kappa_data['temperatures'])} - "
                                f"{max(kappa_data['temperatures'])} K")
                
                if 'comparison' in self.results[mat_name]:
                    comp = self.results[mat_name]['comparison']
                    report.append(f"  Mean absolute error: {comp['mean_absolute_error']:.2f}%")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Saved report to {output_file}")
        
        return report_text
    
    def save_all_results(self, output_dir: Optional[str] = None):
        """Save all results to JSON file."""
        output_dir = output_dir or self.output_dir
        output_path = Path(output_dir) / "tbc_study_results.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Saved all results to {output_path}")


def run_tbc_demo():
    """
    Run a demonstration of the TBC study workflow.
    
    This function creates a complete example without requiring
    actual DFT calculations.
    """
    print("=" * 80)
    print("TBC MATERIALS THERMAL CONDUCTIVITY STUDY - DEMO")
    print("=" * 80)
    print()
    
    # Create study
    study = TBCThermalConductivityStudy(output_dir="./tbc_demo_output")
    
    # Setup materials
    study.setup_standard_materials()
    
    print("Materials configured:")
    for name, mat in study.materials.items():
        print(f"  - {name}: {mat.formula} ({mat.space_group})")
    print()
    
    # Simulate results (in real workflow, these would come from calculations)
    temps = np.array([300, 400, 500, 600, 800, 1000, 1200])
    
    # Simulated thermal conductivity data
    simulated_kappa = {
        'YSZ': [2.8, 2.5, 2.2, 2.0, 1.7, 1.5, 1.3],
        'Gd2Zr2O7': [1.8, 1.6, 1.45, 1.35, 1.2, 1.1, 1.0],
        'HfO2': [1.3, 1.15, 1.05, 0.95, 0.85, 0.78, 0.72]
    }
    
    for mat_name, kappa_values in simulated_kappa.items():
        study.results[mat_name] = {
            'kappa': {
                'temperatures': temps.tolist(),
                'kappa_scalar': kappa_values,
                'kappa_tensor': [[k, 0, 0], [0, k, 0], [0, 0, k]] for k in kappa_values
            }
        }
        
        # Compare with experiment
        study.compare_with_experiment(mat_name)
    
    # Analyze defect scattering for YSZ
    defect_concs = {
        'Y_substitution': 0.08,
        'O_vacancy': 0.04
    }
    study.analyze_defect_scattering('YSZ', defect_concs)
    
    # Generate plots
    study.plot_comparison_all_materials(
        save_path="./tbc_demo_output/tbc_comparison.png"
    )
    
    # Generate report
    report = study.generate_report("./tbc_demo_output/tbc_report.txt")
    print(report)
    
    # Save results
    study.save_all_results()
    
    print("\nDemo completed successfully!")
    print(f"Results saved to: {study.output_dir}")
    
    return study


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='TBC Materials Thermal Conductivity Study')
    parser.add_argument('--demo', action='store_true', 
                       help='Run demonstration without DFT')
    parser.add_argument('--material', type=str,
                       help='Material name to analyze')
    parser.add_argument('--structure', type=str,
                       help='Structure file path')
    parser.add_argument('--fc2', type=str,
                       help='Second-order force constants (fc2.hdf5)')
    parser.add_argument('--fc3', type=str,
                       help='Third-order force constants (fc3.hdf5)')
    parser.add_argument('--outdir', type=str, default='./tbc_output',
                       help='Output directory')
    
    args = parser.parse_args()
    
    if args.demo:
        run_tbc_demo()
    elif args.material and args.structure and args.fc2 and args.fc3:
        # Full calculation workflow
        study = TBCThermalConductivityStudy(output_dir=args.outdir)
        
        material = TBCMaterial(
            name=args.material,
            formula=args.material,
            structure_file=args.structure
        )
        study.add_material(material)
        
        study.run_thermal_conductivity_calculation(
            args.material,
            fc2_path=args.fc2,
            fc3_path=args.fc3
        )
    else:
        print("Usage:")
        print("  python case_thermal_barrier.py --demo")
        print("  python case_thermal_barrier.py --material YSZ ")
        print("      --structure POSCAR --fc2 fc2.hdf5 --fc3 fc3.hdf5")
