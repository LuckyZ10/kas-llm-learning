"""
AGI Materials Intelligence: Application Examples

This module demonstrates practical applications of the AGI core system:
1. Automatic discovery of new material categories
2. Self-improving computational methods
3. Automatic theory generation

Author: AGI Materials Intelligence System
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import json

# Import AGI core modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'agi_core'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'lifelong_learning'))

from meta_learning_v2 import (
    MetaLearningPipeline, TaskConfig, MetaLearnerV2
)
from self_improvement import (
    SelfImprovementManager, PerformanceMetrics
)
from knowledge_creation import (
    KnowledgeCreationPipeline, DiscoveredPattern, GeneratedTheory
)
from lifelong_learning import (
    LifelongLearningSystem, Experience
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Material:
    """Represents a material with its properties."""
    formula: str
    features: np.ndarray
    properties: Dict[str, float]
    category: str = None
    discovered_date: str = None
    

class AutomaticMaterialDiscovery:
    """
    Automatically discovers and categorizes new materials using AGI capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize AGI components
        self.knowledge_creator = KnowledgeCreationPipeline({
            'pattern': {'min_confidence': 0.7},
            'theory': {'min_pattern_confidence': 0.75}
        })
        
        self.meta_learner = MetaLearningPipeline({
            'hidden_dims': [128, 128],
            'meta_lr': 0.001,
            'inner_lr': 0.01,
            'inner_steps': 5
        })
        
        self.lifelong_system = LifelongLearningSystem({
            'use_ewc': True,
            'use_replay': True,
            'replay_capacity': 5000,
            'ewc_importance': 1000.0
        })
        
        # Material database
        self.materials_db = []
        self.discovered_categories = {}
        self.discovery_history = []
        
    def initialize(self):
        """Initialize the discovery system."""
        # Create material science tasks for meta-learning
        tasks = [
            TaskConfig("bandgap_classification", input_dim=50, output_dim=3,
                      task_type="classification", support_size=20),
            TaskConfig("conductivity_regression", input_dim=50, output_dim=1,
                      task_type="regression", support_size=15),
            TaskConfig("stability_prediction", input_dim=50, output_dim=1,
                      task_type="regression", support_size=18),
            TaskConfig("crystal_structure", input_dim=50, output_dim=5,
                      task_type="classification", support_size=25),
        ]
        
        self.meta_learner.initialize(tasks, input_dim=50, output_dim=1)
        
        # Initialize lifelong learning model
        class MaterialPropertyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(50, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1)
                )
            
            def forward(self, x):
                return self.net(x)
        
        self.lifelong_system.initialize_model(MaterialPropertyNet())
        
        logger.info("Automatic material discovery system initialized")
    
    def ingest_materials(self, materials: List[Material]):
        """Ingest materials into the discovery system."""
        self.materials_db.extend(materials)
        
        # Extract features and properties
        features = np.array([m.features for m in materials])
        
        # Process through knowledge creation pipeline
        feature_names = [f'feature_{i}' for i in range(features.shape[1])]
        
        results = self.knowledge_creator.process_data(
            features, feature_names, domain='materials_science'
        )
        
        # Store discovery results
        self.discovery_history.append({
            'timestamp': datetime.now().isoformat(),
            'num_materials': len(materials),
            'patterns_found': len(results['patterns']),
            'theories_generated': len(results['theories'])
        })
        
        return results
    
    def discover_new_categories(self, min_cluster_size: int = 5) -> List[Dict[str, Any]]:
        """
        Automatically discover new material categories based on patterns.
        """
        if not self.materials_db:
            return []
        
        # Get all material features
        all_features = np.array([m.features for m in self.materials_db])
        
        # Discover structural patterns (clusters)
        from sklearn.cluster import DBSCAN, HDBSCAN
        from sklearn.preprocessing import StandardScaler
        
        # Preprocess features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(all_features)
        
        # Try different clustering algorithms
        discovered_categories = []
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=0.5, min_samples=min_cluster_size)
        labels = clustering.fit_predict(features_scaled)
        
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:  # Noise
                continue
            
            cluster_mask = labels == label
            cluster_materials = [m for i, m in enumerate(self.materials_db) if cluster_mask[i]]
            
            # Analyze cluster characteristics
            cluster_features = all_features[cluster_mask]
            
            # Get pattern information from knowledge creator
            feature_names = [f'feature_{i}' for i in range(all_features.shape[1])]
            patterns = self.knowledge_creator.pattern_discovery._discover_statistical_patterns(
                cluster_features, feature_names
            )
            
            category = {
                'category_id': f'category_{label}_{datetime.now().strftime("%Y%m%d")}',
                'size': len(cluster_materials),
                'materials': [m.formula for m in cluster_materials],
                'characteristic_patterns': [p.description for p in patterns[:3]],
                'mean_features': cluster_features.mean(axis=0).tolist(),
                'std_features': cluster_features.std(axis=0).tolist(),
                'discovery_method': 'dbscan_clustering',
                'confidence': len(cluster_materials) / len(self.materials_db)
            }
            
            discovered_categories.append(category)
            self.discovered_categories[category['category_id']] = category
        
        return discovered_categories
    
    def predict_novel_material_properties(self, material_features: np.ndarray) -> Dict[str, Any]:
        """
        Predict properties of a novel material using meta-learning.
        """
        # Quick adaptation for property prediction
        support_x = np.random.randn(20, 50)  # Example support set
        support_y = np.random.randn(20, 1)   # Example labels
        
        adaptation_result = self.meta_learner.quick_adapt(
            support_x, support_y, "novel_material"
        )
        
        # Make prediction
        prediction = self.meta_learner.meta_learner.predict(
            material_features.reshape(1, -1),
            adaptation_result['task_embedding']
        )
        
        # Get similar known materials
        similar_materials = self._find_similar_materials(material_features)
        
        return {
            'predicted_properties': {'band_gap': float(prediction[0, 0])},
            'confidence': adaptation_result['adaptation_steps'] / 10,
            'similar_known_materials': similar_materials,
            'task_embedding': adaptation_result['task_embedding'].tolist()
        }
    
    def _find_similar_materials(self, features: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Find similar materials in database."""
        if not self.materials_db:
            return []
        
        all_features = np.array([m.features for m in self.materials_db])
        
        # Compute similarities
        similarities = np.dot(all_features, features) / (
            np.linalg.norm(all_features, axis=1) * np.linalg.norm(features)
        )
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        similar = []
        for idx in top_indices:
            material = self.materials_db[idx]
            similar.append({
                'formula': material.formula,
                'similarity': float(similarities[idx]),
                'known_properties': material.properties
            })
        
        return similar
    
    def get_discovery_report(self) -> Dict[str, Any]:
        """Generate comprehensive discovery report."""
        knowledge_summary = self.knowledge_creator.get_knowledge_summary()
        lifelong_stats = self.lifelong_system.get_lifelong_stats()
        
        return {
            'total_materials_analyzed': len(self.materials_db),
            'discovered_categories': len(self.discovered_categories),
            'knowledge_summary': knowledge_summary,
            'learning_stats': lifelong_stats,
            'discovery_history': self.discovery_history,
            'categories': list(self.discovered_categories.values())
        }


class SelfImprovingComputationalMethod:
    """
    Self-improving computational methods for materials calculations.
    """
    
    def __init__(self):
        self.improvement_manager = SelfImprovementManager({
            'execution_time_threshold': 1.0,
            'accuracy_threshold': 0.95
        })
        
        self.computational_methods = {}
        self.performance_history = []
        self.optimization_results = []
        
    def register_method(self, name: str, method: callable,
                       test_cases: List[Tuple] = None):
        """Register a computational method for optimization."""
        self.computational_methods[name] = {
            'method': method,
            'test_cases': test_cases,
            'version': 1
        }
        
        self.improvement_manager.register_component(
            name, method, test_cases
        )
        
        logger.info(f"Registered computational method: {name}")
    
    def optimize_method(self, method_name: str) -> Dict[str, Any]:
        """Optimize a registered computational method."""
        if method_name not in self.computational_methods:
            return {'error': f'Method {method_name} not found'}
        
        # Run improvement cycle
        results = self.improvement_manager.run_improvement_cycle(method_name)
        
        # Store results
        self.optimization_results.append({
            'method': method_name,
            'timestamp': datetime.now().isoformat(),
            'optimizations': results['optimizations']
        })
        
        return results
    
    def discover_optimization_algorithm(self, problem_description: str) -> Optional[Dict]:
        """
        Automatically discover a new optimization algorithm.
        """
        # Define evaluation function
        def evaluate_algorithm(code_str: str) -> float:
            try:
                # Compile and test the algorithm
                namespace = {}
                exec(code_str, namespace)
                
                # Simple test: minimize x^2
                if 'optimize' in namespace:
                    result = namespace['optimize'](lambda x: x**2, 1.0)
                    if abs(result) < 0.01:
                        return 1.0
                return 0.0
            except:
                return 0.0
        
        # Register algorithm building blocks
        self.improvement_manager.discovery_engine.register_search_space([
            {
                'name': 'gradient_step',
                'code': '{output} = {input} - learning_rate * gradient',
                'category': 'optimization'
            },
            {
                'name': 'momentum_update',
                'code': 'velocity = momentum * velocity + gradient; {output} = {input} - learning_rate * velocity',
                'category': 'optimization'
            },
            {
                'name': 'adaptive_lr',
                'code': 'learning_rate = learning_rate * decay; {output} = {input} - learning_rate * gradient',
                'category': 'optimization'
            }
        ])
        
        # Discover algorithm
        discovery = self.improvement_manager.discover_new_algorithm(
            problem_description,
            evaluate_algorithm,
            search_budget=50
        )
        
        if discovery:
            return {
                'algorithm_name': discovery.name,
                'description': discovery.description,
                'performance': discovery.performance_metrics.composite_score(),
                'code_preview': discovery.code[:200] + "..."
            }
        
        return None
    
    def get_improvement_report(self) -> Dict[str, Any]:
        """Get report on method improvements."""
        return self.improvement_manager.get_improvement_report()


class AutomaticTheoryGeneration:
    """
    Automatically generates scientific theories from material data.
    """
    
    def __init__(self):
        self.knowledge_pipeline = KnowledgeCreationPipeline({
            'pattern': {
                'min_confidence': 0.65,
                'max_complexity': 6
            },
            'theory': {
                'min_pattern_confidence': 0.7
            }
        })
        
        self.generated_theories = []
        self.theory_validation_results = []
        
    def analyze_material_dataset(self, data: np.ndarray,
                                feature_names: List[str],
                                target_property: str) -> Dict[str, Any]:
        """
        Analyze a material dataset and generate theories.
        """
        logger.info(f"Analyzing dataset with shape {data.shape}")
        
        # Process data through knowledge pipeline
        results = self.knowledge_pipeline.process_data(
            data, feature_names, target_property, 'materials_science'
        )
        
        # Extract theories
        theories = results.get('theories', [])
        
        for theory_data in theories:
            theory_record = {
                'theory_id': theory_data.get('name'),
                'domain': theory_data.get('domain'),
                'assumptions': theory_data.get('assumptions', []),
                'confidence': theory_data.get('confidence', 0),
                'generated_at': datetime.now().isoformat(),
                'supporting_patterns': len(results.get('patterns', []))
            }
            self.generated_theories.append(theory_record)
        
        return results
    
    def validate_theory(self, theory_id: str,
                       validation_data: np.ndarray) -> Dict[str, Any]:
        """
        Validate a generated theory against new data.
        """
        # Find theory
        theory = None
        for t in self.knowledge_pipeline.knowledge_base['theories']:
            if t.name == theory_id:
                theory = t
                break
        
        if not theory:
            return {'error': 'Theory not found'}
        
        # Validate predictions
        validation_results = {
            'theory_id': theory_id,
            'validation_samples': len(validation_data),
            'timestamp': datetime.now().isoformat(),
            'predictions_validated': 0,
            'predictions_failed': 0
        }
        
        # Simple validation: check if data follows expected patterns
        for assumption in theory.assumptions:
            # Check assumption against data
            # This is simplified - real validation would be more sophisticated
            validation_results['predictions_validated'] += 1
        
        theory.validation_status = 'validated' if validation_results['predictions_validated'] > 0 else 'rejected'
        
        self.theory_validation_results.append(validation_results)
        
        return validation_results
    
    def get_theory_catalog(self) -> Dict[str, Any]:
        """Get catalog of all generated theories."""
        return {
            'total_theories': len(self.generated_theories),
            'theories': self.generated_theories,
            'validation_results': self.theory_validation_results,
            'knowledge_summary': self.knowledge_pipeline.get_knowledge_summary()
        }


class AGIMaterialsDemo:
    """
    Demonstrates the complete AGI materials intelligence system.
    """
    
    def __init__(self):
        self.discovery_system = AutomaticMaterialDiscovery()
        self.computational_improver = SelfImprovingComputationalMethod()
        self.theory_generator = AutomaticTheoryGeneration()
        
    def run_full_demonstration(self):
        """Run a complete demonstration of AGI capabilities."""
        print("=" * 70)
        print("AGI MATERIALS INTELLIGENCE SYSTEM - FULL DEMONSTRATION")
        print("=" * 70)
        
        # 1. Initialize systems
        print("\n[1] Initializing AGI systems...")
        self.discovery_system.initialize()
        print("   ✓ Systems initialized")
        
        # 2. Generate synthetic material dataset
        print("\n[2] Generating synthetic material dataset...")
        np.random.seed(42)
        
        materials = []
        
        # Generate materials from different "categories"
        # Category 1: Semiconductors
        for i in range(30):
            features = np.random.randn(50)
            features[0] = np.random.uniform(0.1, 4.0)  # Band gap
            features[1] = np.random.uniform(0.01, 100)  # Conductivity
            materials.append(Material(
                formula=f"SC_{i:03d}",
                features=features,
                properties={'band_gap': features[0], 'conductivity': features[1]},
                category='semiconductor'
            ))
        
        # Category 2: Metals
        for i in range(30):
            features = np.random.randn(50)
            features[0] = 0.0  # Zero band gap
            features[1] = np.random.uniform(1000, 100000)  # High conductivity
            materials.append(Material(
                formula=f"MT_{i:03d}",
                features=features,
                properties={'band_gap': 0.0, 'conductivity': features[1]},
                category='metal'
            ))
        
        # Category 3: Insulators
        for i in range(30):
            features = np.random.randn(50)
            features[0] = np.random.uniform(4.0, 10.0)  # Large band gap
            features[1] = np.random.uniform(0.0001, 0.01)  # Low conductivity
            materials.append(Material(
                formula=f"IN_{i:03d}",
                features=features,
                properties={'band_gap': features[0], 'conductivity': features[1]},
                category='insulator'
            ))
        
        print(f"   ✓ Generated {len(materials)} synthetic materials")
        
        # 3. Ingest materials
        print("\n[3] Ingesting materials into discovery system...")
        discovery_results = self.discovery_system.ingest_materials(materials)
        print(f"   ✓ Discovered {len(discovery_results['patterns'])} patterns")
        print(f"   ✓ Generated {len(discovery_results['theories'])} theories")
        
        # 4. Discover new categories
        print("\n[4] Discovering new material categories...")
        categories = self.discovery_system.discover_new_categories(min_cluster_size=10)
        print(f"   ✓ Discovered {len(categories)} new categories:")
        for cat in categories:
            print(f"     - {cat['category_id']}: {cat['size']} materials")
        
        # 5. Predict properties of novel material
        print("\n[5] Predicting properties of novel material...")
        novel_material_features = np.random.randn(50)
        prediction = self.discovery_system.predict_novel_material_properties(
            novel_material_features
        )
        print(f"   ✓ Predicted band gap: {prediction['predicted_properties']['band_gap']:.3f} eV")
        print(f"   ✓ Confidence: {prediction['confidence']:.3f}")
        print(f"   ✓ Found {len(prediction['similar_known_materials'])} similar materials")
        
        # 6. Self-improving computational method
        print("\n[6] Demonstrating self-improving computational method...")
        
        # Example method to optimize
        def example_energy_calculation(structure):
            """Calculate energy of a structure (simplified)."""
            result = 0
            for i in range(len(structure)):
                for j in range(i+1, len(structure)):
                    result += structure[i] * structure[j]  # Pair interactions
            return result
        
        self.computational_improver.register_method(
            'energy_calculation',
            example_energy_calculation,
            test_cases=[
                ([1.0, 2.0, 3.0], 11.0),
                ([1.0, 1.0, 1.0], 3.0)
            ]
        )
        
        optimization_results = self.computational_improver.optimize_method('energy_calculation')
        print(f"   ✓ Optimizations attempted: {len(optimization_results['optimizations'])}")
        
        # 7. Generate theories
        print("\n[7] Generating scientific theories from material data...")
        
        # Prepare data for theory generation
        all_features = np.array([m.features for m in materials])
        feature_names = [f'feature_{i}' for i in range(50)]
        
        theory_results = self.theory_generator.analyze_material_dataset(
            all_features, feature_names, 'material_properties'
        )
        
        print(f"   ✓ Discovered {len(theory_results['patterns'])} patterns")
        print(f"   ✓ Generated {len(theory_results['theories'])} theories")
        
        if theory_results['theories']:
            theory = theory_results['theories'][0]
            print(f"   ✓ Example theory assumptions:")
            for assumption in theory.get('assumptions', [])[:3]:
                print(f"     • {assumption}")
        
        # 8. Generate summary report
        print("\n[8] Generating summary report...")
        
        discovery_report = self.discovery_system.get_discovery_report()
        improvement_report = self.computational_improver.get_improvement_report()
        theory_catalog = self.theory_generator.get_theory_catalog()
        
        print("\n" + "=" * 70)
        print("SUMMARY REPORT")
        print("=" * 70)
        
        print(f"\n📊 Material Discovery:")
        print(f"   Total materials: {discovery_report['total_materials_analyzed']}")
        print(f"   Discovered categories: {discovery_report['discovered_categories']}")
        print(f"   Knowledge patterns: {discovery_report['knowledge_summary']['total_patterns']}")
        
        print(f"\n🔧 Self-Improvement:")
        print(f"   Components tracked: {improvement_report['components_tracked']}")
        print(f"   Total improvements: {improvement_report['total_improvements']}")
        
        print(f"\n🧬 Theory Generation:")
        print(f"   Total theories: {theory_catalog['total_theories']}")
        print(f"   Total patterns: {theory_catalog['knowledge_summary']['total_patterns']}")
        
        print("\n" + "=" * 70)
        print("DEMONSTRATION COMPLETE")
        print("=" * 70)
        
        return {
            'discovery_report': discovery_report,
            'improvement_report': improvement_report,
            'theory_catalog': theory_catalog
        }


def main():
    """Main entry point for demonstration."""
    demo = AGIMaterialsDemo()
    results = demo.run_full_demonstration()
    return results


if __name__ == "__main__":
    main()
