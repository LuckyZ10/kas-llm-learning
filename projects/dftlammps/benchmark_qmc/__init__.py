"""
DFT-QMC Benchmark and Validation Module
========================================

Provides tools for comparing DFT results with QMC benchmarks
and assessing XC functional accuracy.

Features:
- DFT vs QMC comparison
- XC functional error assessment
- Best functional recommendations
- Statistical analysis

Author: QMC Expert Module
Date: 2026-03-09
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import warnings


@dataclass
class BenchmarkResult:
    """Result from a benchmark calculation."""
    system: str
    method: str
    energy: float
    energy_error: Optional[float] = None
    binding_energy: Optional[float] = None
    lattice_constant: Optional[float] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'system': self.system,
            'method': self.method,
            'energy': self.energy,
            'energy_error': self.energy_error,
            'binding_energy': self.binding_energy,
            'lattice_constant': self.lattice_constant,
            'metadata': self.metadata
        }


@dataclass
class ComparisonResult:
    """Comparison between DFT and QMC results."""
    system: str
    dft_method: str
    qmc_method: str
    dft_energy: float
    qmc_energy: float
    energy_diff: float
    relative_error: float
    within_error: bool
    
    def to_dict(self) -> Dict:
        return {
            'system': self.system,
            'dft_method': self.dft_method,
            'qmc_method': self.qmc_method,
            'dft_energy': self.dft_energy,
            'qmc_energy': self.qmc_energy,
            'energy_diff': self.energy_diff,
            'relative_error': self.relative_error,
            'within_error': self.within_error
        }


class QMCBenchmarkDatabase:
    """
    Database of high-accuracy QMC reference data.
    
    Contains reference values for:
    - G2/99 test set
    - S22 non-covalent interactions
    - Solids and materials
    """
    
    def __init__(self):
        """Initialize benchmark database with reference data."""
        self.reference_data = self._load_reference_data()
    
    def _load_reference_data(self) -> Dict:
        """Load built-in reference data."""
        # G2/99 test set - atomization energies (in eV)
        g2_data = {
            'H2': {'atomization_energy': 4.75, 'source': 'QMC'},
            'N2': {'atomization_energy': 9.91, 'source': 'QMC'},
            'O2': {'atomization_energy': 5.23, 'source': 'QMC'},
            'CO': {'atomization_energy': 11.24, 'source': 'QMC'},
            'CO2': {'atomization_energy': 16.96, 'source': 'QMC'},
            'H2O': {'atomization_energy': 10.09, 'source': 'QMC'},
            'CH4': {'atomization_energy': 17.53, 'source': 'QMC'},
            'NH3': {'atomization_energy': 13.12, 'source': 'QMC'},
            'C2H2': {'atomization_energy': 17.60, 'source': 'QMC'},
            'C2H4': {'atomization_energy': 24.46, 'source': 'QMC'},
            'C2H6': {'atomization_energy': 30.42, 'source': 'QMC'},
            'CH3OH': {'atomization_energy': 23.18, 'source': 'QMC'},
            'CH3Cl': {'atomization_energy': 16.19, 'source': 'QMC'},
        }
        
        # S22 test set - interaction energies (in kcal/mol)
        s22_data = {
            'ammonia_dimer': {'interaction_energy': -3.17, 'type': 'HB'},
            'water_dimer': {'interaction_energy': -5.02, 'type': 'HB'},
            'formic_acid_dimer': {'interaction_energy': -18.75, 'type': 'HB'},
            'formamide_dimer': {'interaction_energy': -16.06, 'type': 'HB'},
            'uracil_dimer_HB': {'interaction_energy': -20.65, 'type': 'HB'},
            'pyridoxine_aminopyridine': {'interaction_energy': -16.71, 'type': 'HB'},
            'adenine_thymine_WC': {'interaction_energy': -16.37, 'type': 'HB'},
            'methane_dimer': {'interaction_energy': -0.53, 'type': 'D'},
            'ethene_dimer': {'interaction_energy': -1.51, 'type': 'D'},
            'benzene_methane': {'interaction_energy': -1.50, 'type': 'D'},
            'benzene_dimer_stack': {'interaction_energy': -2.73, 'type': 'D'},
            'pyrazine_dimer': {'interaction_energy': -4.42, 'type': 'D'},
            'uracil_dimer_stack': {'interaction_energy': -10.12, 'type': 'D'},
            'indole_benzene_stack': {'interaction_energy': -5.22, 'type': 'D'},
            'adenine_thymine_stack': {'interaction_energy': -12.23, 'type': 'D'},
            'ethene_ethyne': {'interaction_energy': -1.53, 'type': 'M'},
            'benzene_water': {'interaction_energy': -3.28, 'type': 'M'},
            'benzene_ammonia': {'interaction_energy': -2.35, 'type': 'M'},
            'benzene_HCN': {'interaction_energy': -4.46, 'type': 'M'},
            'benzene_dimer_T': {'interaction_energy': -2.74, 'type': 'M'},
            'indole_benzene_T': {'interaction_energy': -5.73, 'type': 'M'},
            'phenol_dimer': {'interaction_energy': -7.05, 'type': 'HB'},
        }
        
        # Solid state data - cohesive energies (in eV/atom)
        solids_data = {
            'Si': {'cohesive_energy': 4.63, 'lattice_constant': 5.43, 'source': 'DMC'},
            'Ge': {'cohesive_energy': 3.85, 'lattice_constant': 5.66, 'source': 'DMC'},
            'C_diamond': {'cohesive_energy': 7.58, 'lattice_constant': 3.57, 'source': 'DMC'},
            'Li': {'cohesive_energy': 1.63, 'lattice_constant': 3.49, 'source': 'DMC'},
            'Na': {'cohesive_energy': 1.11, 'lattice_constant': 4.29, 'source': 'DMC'},
            'Mg': {'cohesive_energy': 1.51, 'lattice_constant': 3.21, 'source': 'DMC'},
            'Al': {'cohesive_energy': 3.43, 'lattice_constant': 4.05, 'source': 'DMC'},
            'LiH': {'cohesive_energy': 4.55, 'lattice_constant': 4.08, 'source': 'DMC'},
            'NaCl': {'cohesive_energy': 3.34, 'lattice_constant': 5.64, 'source': 'DMC'},
            'MgO': {'cohesive_energy': 5.18, 'lattice_constant': 4.21, 'source': 'DMC'},
        }
        
        return {
            'G2': g2_data,
            'S22': s22_data,
            'solids': solids_data
        }
    
    def get_reference(self, system: str, test_set: str = 'G2') -> Optional[Dict]:
        """
        Get reference data for a system.
        
        Parameters:
        -----------
        system : str
            System name
        test_set : str
            Test set name ('G2', 'S22', 'solids')
            
        Returns:
        --------
        Reference data dictionary or None
        """
        if test_set in self.reference_data:
            return self.reference_data[test_set].get(system)
        return None
    
    def list_systems(self, test_set: str = 'G2') -> List[str]:
        """List all systems in a test set."""
        if test_set in self.reference_data:
            return list(self.reference_data[test_set].keys())
        return []


class DFTBenchmarkAnalyzer:
    """
    Analyze and compare DFT results with QMC references.
    """
    
    def __init__(self, database: Optional[QMCBenchmarkDatabase] = None):
        """
        Initialize analyzer.
        
        Parameters:
        -----------
        database : Optional[QMCBenchmarkDatabase]
            Reference database (creates new if None)
        """
        self.db = database or QMCBenchmarkDatabase()
        self.results: List[BenchmarkResult] = []
        self.comparisons: List[ComparisonResult] = []
    
    def add_dft_result(self,
                      system: str,
                      functional: str,
                      energy: float,
                      binding_energy: Optional[float] = None):
        """
        Add DFT calculation result.
        
        Parameters:
        -----------
        system : str
            System name
        functional : str
            XC functional used
        energy : float
            Total energy
        binding_energy : Optional[float]
            Binding energy if applicable
        """
        result = BenchmarkResult(
            system=system,
            method=f"DFT/{functional}",
            energy=energy,
            binding_energy=binding_energy
        )
        self.results.append(result)
    
    def add_qmc_result(self,
                      system: str,
                      qmc_method: str,
                      energy: float,
                      energy_error: float,
                      binding_energy: Optional[float] = None):
        """Add QMC calculation result."""
        result = BenchmarkResult(
            system=system,
            method=f"QMC/{qmc_method}",
            energy=energy,
            energy_error=energy_error,
            binding_energy=binding_energy
        )
        self.results.append(result)
    
    def compare_with_reference(self,
                               system: str,
                               dft_functional: str,
                               test_set: str = 'G2') -> Optional[ComparisonResult]:
        """
        Compare DFT result with QMC reference.
        
        Parameters:
        -----------
        system : str
            System name
        dft_functional : str
            DFT functional
        test_set : str
            Test set to use
            
        Returns:
        --------
        ComparisonResult or None
        """
        # Find DFT result
        dft_result = None
        for r in self.results:
            if r.system == system and dft_functional in r.method:
                dft_result = r
                break
        
        if dft_result is None:
            warnings.warn(f"No DFT result found for {system} with {dft_functional}")
            return None
        
        # Get reference
        ref = self.db.get_reference(system, test_set)
        if ref is None:
            warnings.warn(f"No reference data for {system}")
            return None
        
        # Compute comparison
        qmc_energy = ref.get('atomization_energy') or ref.get('interaction_energy') or \
                    ref.get('cohesive_energy')
        
        if qmc_energy is None:
            return None
        
        energy_diff = dft_result.binding_energy - qmc_energy if dft_result.binding_energy \
                     else dft_result.energy - qmc_energy
        
        relative_error = abs(energy_diff / qmc_energy) if qmc_energy != 0 else abs(energy_diff)
        
        comparison = ComparisonResult(
            system=system,
            dft_method=dft_functional,
            qmc_method='Reference',
            dft_energy=dft_result.energy,
            qmc_energy=qmc_energy,
            energy_diff=energy_diff,
            relative_error=relative_error,
            within_error=relative_error < 0.05  # 5% threshold
        )
        
        self.comparisons.append(comparison)
        return comparison
    
    def analyze_functional_accuracy(self,
                                   test_set: str = 'G2') -> Dict:
        """
        Analyze accuracy of different XC functionals.
        
        Parameters:
        -----------
        test_set : str
            Test set to analyze
            
        Returns:
        --------
        Dict with functional statistics
        """
        # Group results by functional
        functional_results: Dict[str, List[float]] = {}
        
        for comp in self.comparisons:
            func = comp.dft_method
            if func not in functional_results:
                functional_results[func] = []
            functional_results[func].append(comp.relative_error)
        
        # Compute statistics
        stats = {}
        for func, errors in functional_results.items():
            stats[func] = {
                'mean_error': np.mean(errors),
                'std_error': np.std(errors),
                'max_error': np.max(errors),
                'mae': np.mean(np.abs(errors)),
                'rmse': np.sqrt(np.mean(np.array(errors) ** 2)),
                'n_systems': len(errors)
            }
        
        return stats
    
    def rank_functionals(self, test_set: str = 'G2') -> List[Tuple[str, float]]:
        """
        Rank XC functionals by accuracy.
        
        Returns:
        --------
        List of (functional, score) tuples, sorted by score
        """
        stats = self.analyze_functional_accuracy(test_set)
        
        # Score based on MAE (lower is better)
        rankings = [(func, data['mae']) 
                   for func, data in stats.items()]
        
        return sorted(rankings, key=lambda x: x[1])
    
    def recommend_functional(self,
                            system_type: str = 'general',
                            property_type: str = 'energy') -> str:
        """
        Recommend best XC functional based on benchmarks.
        
        Parameters:
        -----------
        system_type : str
            Type of system ('general', 'molecular', 'solid', 'surface')
        property_type : str
            Property of interest ('energy', 'barrier', 'band_gap')
            
        Returns:
        --------
        Recommended functional name
        """
        # Predefined recommendations based on literature
        recommendations = {
            'general': {
                'energy': 'PBE0',
                'barrier': 'B3LYP',
                'band_gap': 'HSE06'
            },
            'molecular': {
                'energy': 'CCSD(T)',  # For reference
                'barrier': 'M06-2X',
                'reaction': 'PBE0'
            },
            'solid': {
                'energy': 'PBEsol',
                'band_gap': 'HSE06',
                'lattice': 'PBEsol'
            },
            'surface': {
                'energy': 'RPBE',
                'adsorption': 'BEEF-vdW'
            }
        }
        
        rec = recommendations.get(system_type, recommendations['general'])
        return rec.get(property_type, 'PBE')
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate benchmark report.
        
        Parameters:
        -----------
        output_file : Optional[str]
            File to write report to
            
        Returns:
        --------
        Report as string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("DFT-QMC BENCHMARK REPORT")
        lines.append("=" * 70)
        lines.append("")
        
        # Functional rankings
        lines.append("FUNCTIONAL RANKINGS")
        lines.append("-" * 70)
        rankings = self.rank_functionals()
        for i, (func, score) in enumerate(rankings[:10], 1):
            lines.append(f"{i:2d}. {func:20s}  MAE = {score:.4f}")
        lines.append("")
        
        # Detailed comparisons
        lines.append("DETAILED COMPARISONS")
        lines.append("-" * 70)
        for comp in self.comparisons:
            status = "✓" if comp.within_error else "✗"
            lines.append(f"{status} {comp.system:20s} {comp.dft_method:15s} "
                        f"ΔE = {comp.energy_diff:8.4f} ({comp.relative_error*100:5.2f}%)")
        lines.append("")
        
        # Recommendations
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 70)
        lines.append(f"General purpose: {self.recommend_functional('general', 'energy')}")
        lines.append(f"Molecules:       {self.recommend_functional('molecular', 'energy')}")
        lines.append(f"Solids:          {self.recommend_functional('solid', 'energy')}")
        lines.append(f"Band gaps:       {self.recommend_functional('solid', 'band_gap')}")
        lines.append("")
        
        report = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
        
        return report
    
    def save_results(self, filename: str):
        """Save all results to JSON file."""
        data = {
            'results': [r.to_dict() for r in self.results],
            'comparisons': [c.to_dict() for c in self.comparisons]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_results(self, filename: str):
        """Load results from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.results = [BenchmarkResult(**r) for r in data['results']]
        self.comparisons = [ComparisonResult(**c) for c in data['comparisons']]


def run_g2_benchmark(dft_calculator_fn: Callable,
                    functionals: List[str] = None) -> Dict:
    """
    Run G2/99 benchmark for specified functionals.
    
    Parameters:
    -----------
    dft_calculator_fn : Callable
        Function that takes (system, functional) and returns energy
    functionals : List[str]
        List of functionals to test
        
    Returns:
    --------
    Dict with benchmark results
    """
    if functionals is None:
        functionals = ['LDA', 'PBE', 'B3LYP', 'PBE0']
    
    db = QMCBenchmarkDatabase()
    analyzer = DFTBenchmarkAnalyzer(db)
    
    systems = db.list_systems('G2')
    
    for functional in functionals:
        print(f"Running G2 benchmark for {functional}...")
        
        for system in systems:
            try:
                energy = dft_calculator_fn(system, functional)
                analyzer.add_dft_result(system, functional, energy)
                
                # Compare with reference
                analyzer.compare_with_reference(system, functional, 'G2')
            except Exception as e:
                warnings.warn(f"Failed for {system} with {functional}: {e}")
    
    return analyzer.analyze_functional_accuracy('G2')


def run_s22_benchmark(dft_calculator_fn: Callable,
                     functionals: List[str] = None) -> Dict:
    """Run S22 non-covalent interactions benchmark."""
    if functionals is None:
        functionals = ['PBE', 'PBE-D3', 'B3LYP-D3', 'vdW-DF']
    
    db = QMCBenchmarkDatabase()
    analyzer = DFTBenchmarkAnalyzer(db)
    
    systems = db.list_systems('S22')
    
    for functional in functionals:
        print(f"Running S22 benchmark for {functional}...")
        
        for system in systems:
            try:
                energy = dft_calculator_fn(system, functional)
                analyzer.add_dft_result(system, functional, energy)
                analyzer.compare_with_reference(system, functional, 'S22')
            except Exception as e:
                warnings.warn(f"Failed for {system} with {functional}: {e}")
    
    return analyzer.analyze_functional_accuracy('S22')


# Type hint for the function parameter
from typing import Callable
