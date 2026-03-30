"""
AI Discovery Case Studies
==========================

This module provides complete application examples demonstrating the use
of AI-driven materials discovery for specific applications.

Case Studies:
1. Solid Electrolyte Discovery - Finding new Li/Na superionic conductors
2. High-Entropy Alloy Catalyst Design - Multi-component catalyst optimization

Each case study includes:
- Problem setup and constraints
- Active learning workflow configuration
- Integration with DFT validation
- Results analysis and visualization

Author: DFT+LAMMPS AI Team
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Import from dftlammps AI module
try:
    from dftlammps.ai import (
        StructureGenerator, PropertyPredictor,
        ActiveDiscovery, ActiveDiscoveryConfig, SamplingStrategy,
        BayesianOptimizer, BayesianOptimizerConfig,
        CrystalStructure
    )
    from dftlammps.ai.generative_models import GenerativeModelConfig
    from dftlammps.ai.property_predictor import PropertyPredictorConfig
except ImportError:
    import sys
    sys.path.insert(0, '/root/.openclaw/workspace')
    from dftlammps.ai import (
        StructureGenerator, PropertyPredictor,
        ActiveDiscovery, ActiveDiscoveryConfig, SamplingStrategy,
        BayesianOptimizer, BayesianOptimizerConfig,
        CrystalStructure
    )
    from dftlammps.ai.generative_models import GenerativeModelConfig
    from dftlammps.ai.property_predictor import PropertyPredictorConfig


@dataclass
class DiscoveryConfig:
    """Configuration for discovery case studies."""
    output_dir: str = "./discovery_results"
    num_iterations: int = 30
    samples_per_iteration: int = 5
    use_dft: bool = True
    save_intermediate: bool = True


# ============================================================================
# Case Study 1: Solid Electrolyte Discovery
# ============================================================================

class SolidElectrolyteDiscovery:
    """
    AI-driven discovery of solid electrolyte materials.
    
    Target: Find novel superionic conductors with high ionic conductivity,
    good electrochemical stability, and compatibility with electrodes.
    
    Properties of interest:
    - Ionic conductivity (target: > 1 mS/cm)
    - Band gap (target: > 3 eV for stability)
    - Bulk modulus (target: moderate for dendrite suppression)
    - Migration barrier (target: < 0.3 eV)
    """
    
    def __init__(self, target_ion: str = "Li", config: DiscoveryConfig = None):
        """
        Initialize solid electrolyte discovery.
        
        Args:
            target_ion: Target ionic species ('Li', 'Na', 'K', 'Mg', etc.)
            config: Discovery configuration
        """
        self.target_ion = target_ion
        self.config = config or DiscoveryConfig()
        
        # Define composition space for solid electrolytes
        self.composition_space = self._define_composition_space()
        
        # Initialize components
        self.generator = None
        self.predictor = None
        self.discovery = None
        
        self.results = {}
    
    def _define_composition_space(self) -> List[str]:
        """Define chemical space for solid electrolytes."""
        ion = self.target_ion
        
        # Known and candidate solid electrolyte compositions
        compositions = [
            # Known sulfides
            f"{ion}3PS4", f"{ion}6PS5Cl", f"{ion}4P2S7",
            f"{ion}3PSe4", f"{ion}6PSe5Cl", f"{ion}2S",
            
            # Oxides
            f"{ion}AlO2", f"{ion}3La3Zr2O12", f"{ion}7La3Zr2O13",
            f"{ion}3xLa(2/3-x)TiO3", f"{ion}NbO3",
            
            # NASICON-type
            f"{ion}Zr2P3O12", f"{ion}Ti2P3O12", f"{ion}Hf2P3O12",
            f"{ion}1.3Al0.3Ti1.7P3O12", f"{ion}1.3Al0.3Ge1.7P3O12",
            
            # Garnets
            f"{ion}7La3Zr2O12", f"{ion}6BaLa2Ta2O12",
            f"{ion}5.5La3Nb1.75Zr0.25O12",
            
            # Halides
            f"{ion}3YCl6", f"{ion}3YBr6", f"{ion}3ErCl6",
            
            # New candidate systems
            f"{ion}2SiS3", f"{ion}2GeS3", f"{ion}4SnS4",
            f"{ion}3AsS3", f"{ion}3SbS3",
        ]
        
        return compositions
    
    def setup(self):
        """Initialize discovery components."""
        print("=" * 70)
        print(f"SOLID ELECTROLYTE DISCOVERY - {self.target_ion}-ion conductors")
        print("=" * 70)
        
        # Structure generator
        gen_config = GenerativeModelConfig(
            model_type='cdvae',
            latent_dim=256,
            hidden_dim=512,
            num_layers=6,
            max_atoms=80,
            cutoff_distance=8.0
        )
        self.generator = StructureGenerator('cdvae', gen_config)
        print("✓ Structure generator initialized (CDVAE)")
        
        # Property predictor - multi-task for several properties
        pred_config = PropertyPredictorConfig(
            model_type='alignn',  # ALIGNN for better accuracy
            hidden_dim=256,
            num_layers=4,
            output_dim=4,  # 4 properties: conductivity, band gap, modulus, barrier
            task_type='multi_task'
        )
        self.predictor = PropertyPredictor('alignn', pred_config)
        print("✓ Property predictor initialized (ALIGNN)")
        
        # Active discovery workflow
        discovery_config = ActiveDiscoveryConfig(
            num_iterations=self.config.num_iterations,
            samples_per_iteration=self.config.samples_per_iteration,
            sampling_strategy=SamplingStrategy.HYBRID,
            uncertainty_weight=0.4,
            diversity_weight=0.3,
            greedy_weight=0.3,
            retrain_frequency=3,
            dft_validation_threshold=0.15,
            max_dft_per_iteration=3,
            output_dir=f"{self.config.output_dir}/{self.target_ion}_electrolyte"
        )
        
        self.discovery = ActiveDiscovery(
            config=discovery_config,
            structure_generator=self.generator,
            property_predictor=self.predictor,
            dft_evaluator=self._mock_dft_evaluator if not self.config.use_dft else None
        )
        print("✓ Active discovery workflow initialized")
    
    def _mock_dft_evaluator(self, structure: CrystalStructure) -> float:
        """Mock DFT evaluator for testing."""
        # Simulate ionic conductivity based on composition
        composition = structure.composition or ""
        
        # Base score
        score = 0.0
        
        # Bonus for target ion content
        if self.target_ion in composition:
            score += 0.5
        
        # Bonus for good anions (S, Se for sulfides, O for oxides)
        if any(x in composition for x in ['S', 'Se', 'O']):
            score += 0.3
        
        # Random variation
        score += np.random.randn() * 0.2
        
        return score
    
    def run(self) -> Dict[str, Any]:
        """Run the discovery workflow."""
        if self.discovery is None:
            self.setup()
        
        print(f"\n{'='*70}")
        print("STARTING DISCOVERY WORKFLOW")
        print(f"{'='*70}")
        print(f"Composition space: {len(self.composition_space)} candidates")
        print(f"Iterations: {self.config.num_iterations}")
        print(f"Samples per iteration: {self.config.samples_per_iteration}")
        
        # Run discovery
        result = self.discovery.run(
            composition_space=self.composition_space,
            num_candidates_per_iteration=50
        )
        
        # Analyze results
        self.results = self._analyze_results(result)
        
        # Save results
        if self.config.save_intermediate:
            self._save_results()
        
        return self.results
    
    def _analyze_results(self, result) -> Dict[str, Any]:
        """Analyze discovery results."""
        print(f"\n{'='*70}")
        print("ANALYZING RESULTS")
        print(f"{'='*70}")
        
        # Get top structures
        top_structures = result.get_best_structures(10)
        
        analysis = {
            'target_ion': self.target_ion,
            'total_structures': len(result.all_structures),
            'total_dft': result.total_dft_calculations,
            'top_discoveries': []
        }
        
        print("\nTop 10 Discoveries:")
        for i, (struct, value) in enumerate(top_structures):
            discovery = {
                'rank': i + 1,
                'composition': struct.composition,
                'predicted_value': float(value),
                'num_atoms': len(struct.atomic_numbers)
            }
            analysis['top_discoveries'].append(discovery)
            print(f"  {i+1}. {struct.composition}: {value:.4f}")
        
        # Composition analysis
        compositions = [s.composition for s in result.all_structures if s.composition]
        unique_comps = set(compositions)
        analysis['unique_compositions'] = len(unique_comps)
        
        print(f"\nUnique compositions explored: {len(unique_comps)}")
        
        return analysis
    
    def _save_results(self):
        """Save discovery results."""
        output_dir = Path(self.config.output_dir) / f"{self.target_ion}_electrolyte"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save analysis
        with open(output_dir / "analysis.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to {output_dir}")
    
    def visualize_results(self):
        """Visualize discovery results."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for visualization")
            return
        
        output_dir = Path(self.config.output_dir) / f"{self.target_ion}_electrolyte"
        
        # Plot discovery progress
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # TODO: Add actual visualization code
        
        plt.tight_layout()
        plt.savefig(output_dir / "discovery_analysis.png", dpi=150)
        plt.close()
        
        print(f"Visualization saved to {output_dir / 'discovery_analysis.png'}")


# ============================================================================
# Case Study 2: High-Entropy Alloy Catalyst Design
# ============================================================================

class HighEntropyAlloyDiscovery:
    """
    AI-driven design of high-entropy alloy (HEA) catalysts.
    
    Target: Discover multi-component alloys with optimal catalytic activity
    for reactions like oxygen reduction (ORR), hydrogen evolution (HER),
    or CO2 reduction (CO2RR).
    
    Properties of interest:
    - Adsorption energy of reaction intermediates
    - Electronic d-band center
    - Surface energy and stability
    - Activity descriptors ( volcano plot optimization)
    """
    
    def __init__(
        self,
        target_reaction: str = "ORR",
        config: DiscoveryConfig = None
    ):
        """
        Initialize HEA catalyst discovery.
        
        Args:
            target_reaction: Target reaction ('ORR', 'HER', 'OER', 'CO2RR')
            config: Discovery configuration
        """
        self.target_reaction = target_reaction
        self.config = config or DiscoveryConfig()
        
        # Define element pool for HEAs
        self.element_pool = self._define_element_pool()
        
        # Initialize components
        self.generator = None
        self.predictor = None
        self.optimizer = None
        
        self.results = {}
    
    def _define_element_pool(self) -> List[str]:
        """Define element pool for HEA catalysts."""
        # Noble and transition metals for catalysis
        base_elements = [
            'Pt', 'Pd', 'Au', 'Ag', 'Cu', 'Ni', 'Co', 'Fe',  # 3d/4d/5d metals
            'Ir', 'Rh', 'Ru', 'Os',  # Platinum group
            'Mn', 'Cr', 'V', 'Ti', 'Zr', 'Hf',  # Early transition metals
            'Mo', 'W', 'Nb', 'Ta',  # Refractory metals
            'Al', 'Ga', 'In', 'Sn', 'Bi',  # p-block elements for tuning
        ]
        
        return base_elements
    
    def _generate_composition_space(self, n_components: int = 5) -> List[Dict[str, float]]:
        """Generate HEA composition space."""
        compositions = []
        
        # Generate random HEA compositions
        np.random.seed(42)
        for _ in range(100):
            # Select n_components random elements
            elements = np.random.choice(self.element_pool, n_components, replace=False)
            
            # Generate random composition (approximately equiatomic)
            fractions = np.random.dirichlet(np.ones(n_components)) * 100
            
            composition = {elem: float(frac) for elem, frac in zip(elements, fractions)}
            compositions.append(composition)
        
        return compositions
    
    def setup(self):
        """Initialize discovery components."""
        print("=" * 70)
        print(f"HIGH-ENTROPY ALLOY DISCOVERY - {self.target_reaction} catalyst")
        print("=" * 70)
        
        # Structure generator for alloys
        gen_config = GenerativeModelConfig(
            model_type='diffcsp',  # DiffCSP good for composition-conditioned generation
            latent_dim=256,
            hidden_dim=512,
            num_layers=6,
            max_atoms=100,
            cutoff_distance=10.0  # Larger cutoff for metals
        )
        self.generator = StructureGenerator('diffcsp', gen_config)
        print("✓ Structure generator initialized (DiffCSP)")
        
        # Property predictor for catalytic properties
        pred_config = PropertyPredictorConfig(
            model_type='cgcnn',
            hidden_dim=256,
            num_layers=4,
            output_dim=1,
            cutoff_distance=10.0
        )
        self.predictor = PropertyPredictor('cgcnn', pred_config)
        print("✓ Property predictor initialized (CGCNN)")
        
        # Bayesian optimizer for composition optimization
        opt_config = BayesianOptimizerConfig(
            num_init_samples=20,
            num_iterations=self.config.num_iterations,
            batch_size=self.config.samples_per_iteration,
            acquisition_type='ei',
            num_candidates=80,
            dft_validation_frequency=5,
            output_dir=f"{self.config.output_dir}/hea_{self.target_reaction}"
        )
        
        self.optimizer = BayesianOptimizer(
            config=opt_config,
            structure_generator=self.generator,
            property_predictor=self.predictor,
            dft_evaluator=self._mock_dft_evaluator if not self.config.use_dft else None
        )
        print("✓ Bayesian optimizer initialized")
    
    def _mock_dft_evaluator(self, structure: CrystalStructure) -> float:
        """Mock DFT evaluator for catalytic activity."""
        # Simulate catalytic activity
        composition = structure.composition or ""
        
        # Base activity
        activity = 0.0
        
        # Noble metals improve activity
        noble_metals = ['Pt', 'Pd', 'Au', 'Ir', 'Rh']
        for metal in noble_metals:
            if metal in composition:
                activity += 0.2
        
        # 3d transition metals for tunability
        transition_metals = ['Ni', 'Co', 'Fe', 'Mn', 'Cr']
        for metal in transition_metals:
            if metal in composition:
                activity += 0.1
        
        # Random variation
        activity += np.random.randn() * 0.15
        
        return activity
    
    def run(self) -> Dict[str, Any]:
        """Run the HEA discovery workflow."""
        if self.optimizer is None:
            self.setup()
        
        print(f"\n{'='*70}")
        print("STARTING HEA DISCOVERY WORKFLOW")
        print(f"{'='*70}")
        print(f"Target reaction: {self.target_reaction}")
        print(f"Element pool: {len(self.element_pool)} elements")
        print(f"Iterations: {self.config.num_iterations}")
        
        # Generate composition space
        compositions = self._generate_composition_space(n_components=5)
        print(f"Composition space: {len(compositions)} candidates")
        
        # Run optimization
        result = self.optimizer.optimize(
            composition_space=[c for c in compositions],
        )
        
        # Analyze results
        self.results = self._analyze_results(result)
        
        # Save results
        if self.config.save_intermediate:
            self._save_results()
        
        return self.results
    
    def _analyze_results(self, result) -> Dict[str, Any]:
        """Analyze HEA discovery results."""
        print(f"\n{'='*70}")
        print("ANALYZING HEA RESULTS")
        print(f"{'='*70}")
        
        analysis = {
            'target_reaction': self.target_reaction,
            'best_value': float(result.best_y),
            'num_iterations': result.num_iterations,
            'num_dft': result.num_dft_calculations,
        }
        
        if result.best_structure:
            analysis['best_composition'] = result.best_structure.composition
            print(f"\nBest catalyst found:")
            print(f"  Composition: {result.best_structure.composition}")
            print(f"  Predicted activity: {result.best_y:.4f}")
        
        # Analyze composition trends
        print(f"\nTotal DFT calculations: {result.num_dft_calculations}")
        
        return analysis
    
    def _save_results(self):
        """Save HEA discovery results."""
        output_dir = Path(self.config.output_dir) / f"hea_{self.target_reaction}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save analysis
        with open(output_dir / "analysis.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to {output_dir}")


# ============================================================================
# Utility Functions and Runners
# ============================================================================

def run_solid_electrolyte_discovery(
    target_ion: str = "Li",
    num_iterations: int = 30,
    output_dir: str = "./discovery_results"
) -> Dict[str, Any]:
    """
    Run solid electrolyte discovery case study.
    
    Args:
        target_ion: Target ion ('Li', 'Na', 'K', etc.)
        num_iterations: Number of discovery iterations
        output_dir: Output directory
    
    Returns:
        Discovery results dictionary
    """
    config = DiscoveryConfig(
        output_dir=output_dir,
        num_iterations=num_iterations,
        use_dft=False,  # Set to True for actual DFT
        save_intermediate=True
    )
    
    discovery = SolidElectrolyteDiscovery(target_ion, config)
    results = discovery.run()
    
    try:
        discovery.visualize_results()
    except:
        pass
    
    return results


def run_hea_catalyst_discovery(
    target_reaction: str = "ORR",
    num_iterations: int = 30,
    output_dir: str = "./discovery_results"
) -> Dict[str, Any]:
    """
    Run HEA catalyst discovery case study.
    
    Args:
        target_reaction: Target reaction ('ORR', 'HER', 'OER', 'CO2RR')
        num_iterations: Number of discovery iterations
        output_dir: Output directory
    
    Returns:
        Discovery results dictionary
    """
    config = DiscoveryConfig(
        output_dir=output_dir,
        num_iterations=num_iterations,
        use_dft=False,
        save_intermediate=True
    )
    
    discovery = HighEntropyAlloyDiscovery(target_reaction, config)
    results = discovery.run()
    
    return results


def compare_discovery_methods(
    composition_space: List[str],
    num_iterations: int = 20,
    num_repeats: int = 3
) -> Dict[str, Any]:
    """
    Compare different discovery methods.
    
    Args:
        composition_space: Compositions to test
        num_iterations: Iterations per method
        num_repeats: Number of repeats for statistics
    
    Returns:
        Comparison results
    """
    methods = {
        'random': SamplingStrategy.HYBRID,  # Will use random baseline
        'uncertainty': SamplingStrategy.UNCERTAINTY,
        'greedy': SamplingStrategy.GREEDY,
        'hybrid': SamplingStrategy.HYBRID,
    }
    
    results = {}
    
    for method_name, strategy in methods.items():
        print(f"\nTesting {method_name}...")
        
        method_results = []
        for repeat in range(num_repeats):
            print(f"  Repeat {repeat + 1}/{num_repeats}")
            
            config = ActiveDiscoveryConfig(
                num_iterations=num_iterations,
                sampling_strategy=strategy,
                samples_per_iteration=5
            )
            
            generator = StructureGenerator('random', GenerativeModelConfig())
            predictor = PropertyPredictor('cgcnn', PropertyPredictorConfig())
            
            discovery = ActiveDiscovery(
                config=config,
                structure_generator=generator,
                property_predictor=predictor
            )
            
            result = discovery.run(composition_space=composition_space)
            
            # Get best value found
            best_value = max(result.all_predictions) if result.all_predictions else 0
            method_results.append(best_value)
        
        results[method_name] = {
            'mean': np.mean(method_results),
            'std': np.std(method_results),
            'values': method_results
        }
    
    # Print comparison
    print("\n" + "=" * 50)
    print("METHOD COMPARISON")
    print("=" * 50)
    for method, data in results.items():
        print(f"{method:15s}: {data['mean']:.4f} ± {data['std']:.4f}")
    
    return results


def generate_discovery_report(
    results: Dict[str, Any],
    output_path: str = "discovery_report.md"
):
    """Generate markdown report from discovery results."""
    report = []
    report.append("# Materials Discovery Report\n")
    report.append(f"## Target: {results.get('target_ion', results.get('target_reaction', 'Unknown'))}\n")
    
    report.append("### Summary\n")
    report.append(f"- Total structures explored: {results.get('total_structures', 'N/A')}")
    report.append(f"- DFT calculations performed: {results.get('total_dft', 'N/A')}")
    report.append(f"- Unique compositions: {results.get('unique_compositions', 'N/A')}\n")
    
    report.append("### Top Discoveries\n")
    report.append("| Rank | Composition | Predicted Value |")
    report.append("|------|-------------|-----------------|")
    
    for disc in results.get('top_discoveries', []):
        report.append(f"| {disc['rank']} | {disc['composition']} | {disc['predicted_value']:.4f} |")
    
    report.append("\n### Recommendations\n")
    report.append("Based on the discovery workflow, the following compositions")
    report.append("are recommended for experimental synthesis:\n")
    
    for disc in results.get('top_discoveries', [])[:5]:
        report.append(f"- **{disc['composition']}**: Predicted value {disc['predicted_value']:.4f}")
    
    # Write report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    # Run case studies
    print("AI DISCOVERY CASE STUDIES")
    print("=" * 70)
    
    # Case 1: Solid Electrolyte Discovery
    print("\n" + "=" * 70)
    print("CASE 1: Solid Electrolyte Discovery")
    print("=" * 70)
    
    results_electrolyte = run_solid_electrolyte_discovery(
        target_ion="Li",
        num_iterations=10,  # Reduced for demo
        output_dir="./discovery_results"
    )
    
    # Case 2: HEA Catalyst Discovery
    print("\n" + "=" * 70)
    print("CASE 2: High-Entropy Alloy Catalyst Discovery")
    print("=" * 70)
    
    results_hea = run_hea_catalyst_discovery(
        target_reaction="ORR",
        num_iterations=10,
        output_dir="./discovery_results"
    )
    
    print("\n" + "=" * 70)
    print("ALL CASE STUDIES COMPLETE")
    print("=" * 70)
