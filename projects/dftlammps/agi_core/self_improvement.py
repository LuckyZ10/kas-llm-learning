"""
Self-Improvement Module: Autonomous Code and Algorithm Optimization

This module enables the AGI system to:
- Optimize its own code and algorithms
- Discover new algorithms automatically
- Improve performance through self-reflection

Author: AGI Materials Intelligence System
"""

import ast
import inspect
import re
import time
import traceback
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import hashlib
import json
import pickle
from pathlib import Path
import logging
from datetime import datetime
import sys
import types
from abc import ABC, abstractmethod
import copy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for code evaluation."""
    execution_time: float = 0.0
    memory_usage: float = 0.0
    accuracy: float = 0.0
    complexity_score: float = 0.0  # Code complexity
    efficiency_score: float = 0.0  # Combined efficiency metric
    stability_score: float = 0.0   # Consistency across runs
    iteration_count: int = 0
    
    def composite_score(self) -> float:
        """Calculate composite performance score."""
        # Weighted combination of metrics
        weights = {
            'accuracy': 0.4,
            'efficiency': 0.3,
            'stability': 0.2,
            'simplicity': 0.1
        }
        
        efficiency = 1.0 / (1.0 + self.execution_time + self.memory_usage / 1000)
        simplicity = 1.0 / (1.0 + self.complexity_score)
        
        score = (
            weights['accuracy'] * self.accuracy +
            weights['efficiency'] * efficiency +
            weights['stability'] * self.stability_score +
            weights['simplicity'] * simplicity
        )
        return score


@dataclass
class OptimizationAttempt:
    """Record of an optimization attempt."""
    timestamp: str
    original_code_hash: str
    optimized_code_hash: str
    optimization_type: str
    performance_before: PerformanceMetrics
    performance_after: PerformanceMetrics
    improvement_ratio: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class AlgorithmDiscovery:
    """Record of a discovered algorithm."""
    name: str
    description: str
    code: str
    discovery_date: str
    performance_metrics: PerformanceMetrics
    use_cases: List[str] = field(default_factory=list)
    parent_algorithms: List[str] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)


class CodeAnalyzer:
    """Analyzes Python code for optimization opportunities."""
    
    def __init__(self):
        self.complexity_patterns = {
            'nested_loops': re.compile(r'for.*:\s*\n.*for.*:', re.MULTILINE),
            'recursive_calls': re.compile(r'def\s+\w+.*:\s*\n.*\w+\(', re.MULTILINE),
            'list_append_in_loop': re.compile(r'for.*:\s*\n.*\.append\(', re.MULTILINE),
            'string_concatenation': re.compile(r'\+\s*[\'"]'),
            'repeated_computation': re.compile(r'(\w+\([^)]*\)).*\1'),
        }
        
    def analyze(self, code: str, func_name: str = None) -> Dict[str, Any]:
        """Analyze code for optimization opportunities."""
        analysis = {
            'complexity_score': 0,
            'optimization_opportunities': [],
            'code_structure': {},
            'suggestions': []
        }
        
        try:
            # Parse AST
            tree = ast.parse(code)
            
            # Calculate cyclomatic complexity
            analysis['complexity_score'] = self._calculate_complexity(tree)
            
            # Find optimization opportunities
            analysis['optimization_opportunities'] = self._find_optimizations(code, tree)
            
            # Analyze code structure
            analysis['code_structure'] = self._analyze_structure(tree)
            
            # Generate suggestions
            analysis['suggestions'] = self._generate_suggestions(
                analysis['optimization_opportunities']
            )
            
        except SyntaxError as e:
            analysis['error'] = str(e)
            
        return analysis
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity of code."""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, 
                                ast.ExceptHandler, ast.With,
                                ast.Assert, ast.comprehension)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity
    
    def _find_optimizations(self, code: str, tree: ast.AST) -> List[Dict]:
        """Find potential optimization opportunities."""
        opportunities = []
        
        # Check for nested loops
        for match in self.complexity_patterns['nested_loops'].finditer(code):
            opportunities.append({
                'type': 'nested_loops',
                'line': code[:match.start()].count('\n') + 1,
                'suggestion': 'Consider vectorization or algorithmic optimization'
            })
        
        # Check for list comprehension opportunities
        for match in self.complexity_patterns['list_append_in_loop'].finditer(code):
            opportunities.append({
                'type': 'list_comprehension',
                'line': code[:match.start()].count('\n') + 1,
                'suggestion': 'Consider using list comprehension or generator'
            })
        
        # Find repeated function calls
        for match in self.complexity_patterns['repeated_computation'].finditer(code):
            opportunities.append({
                'type': 'caching',
                'line': code[:match.start()].count('\n') + 1,
                'suggestion': 'Cache repeated computations'
            })
        
        return opportunities
    
    def _analyze_structure(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze code structure."""
        structure = {
            'num_functions': 0,
            'num_classes': 0,
            'num_imports': 0,
            'num_loops': 0,
            'num_conditionals': 0
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                structure['num_functions'] += 1
            elif isinstance(node, ast.ClassDef):
                structure['num_classes'] += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                structure['num_imports'] += 1
            elif isinstance(node, (ast.For, ast.While)):
                structure['num_loops'] += 1
            elif isinstance(node, ast.If):
                structure['num_conditionals'] += 1
        
        return structure
    
    def _generate_suggestions(self, opportunities: List[Dict]) -> List[str]:
        """Generate optimization suggestions."""
        suggestions = []
        
        type_counts = defaultdict(int)
        for opp in opportunities:
            type_counts[opp['type']] += 1
        
        if type_counts['nested_loops'] > 0:
            suggestions.append(
                f"Found {type_counts['nested_loops']} nested loops. "
                "Consider vectorization with NumPy or algorithmic improvements."
            )
        
        if type_counts['list_comprehension'] > 0:
            suggestions.append(
                f"Found {type_counts['list_comprehension']} opportunities for "
                "list comprehensions."
            )
        
        if type_counts['caching'] > 0:
            suggestions.append(
                f"Found {type_counts['caching']} repeated computations. "
                "Consider memoization."
            )
        
        return suggestions


class CodeOptimizer:
    """Automatically optimizes Python code."""
    
    def __init__(self):
        self.analyzer = CodeAnalyzer()
        self.optimization_history = []
        self.cache = {}
        
    def optimize(self, func: Callable, test_cases: List[Tuple] = None,
                metric_func: Callable = None) -> Dict[str, Any]:
        """
        Optimize a function automatically.
        
        Args:
            func: Function to optimize
            test_cases: List of (args, expected_output) for validation
            metric_func: Custom metric function for evaluation
        """
        # Get source code
        try:
            source = inspect.getsource(func)
        except (TypeError, OSError):
            return {'error': 'Cannot get source code'}
        
        # Analyze code
        analysis = self.analyzer.analyze(source)
        
        # Generate optimized versions
        optimized_versions = self._generate_optimizations(source, analysis)
        
        # Evaluate each version
        best_version = None
        best_score = -float('inf')
        
        baseline_metrics = self._evaluate_performance(func, test_cases, metric_func)
        
        for version_code in optimized_versions:
            try:
                # Compile and test optimized code
                optimized_func = self._compile_function(version_code, func.__name__)
                metrics = self._evaluate_performance(optimized_func, test_cases, metric_func)
                
                if metrics.composite_score() > best_score:
                    best_score = metrics.composite_score()
                    best_version = {
                        'code': version_code,
                        'metrics': metrics,
                        'improvement': metrics.composite_score() / baseline_metrics.composite_score() - 1
                    }
            except Exception as e:
                logger.debug(f"Optimization failed: {e}")
                continue
        
        return {
            'original_code': source,
            'analysis': analysis,
            'optimized_version': best_version,
            'baseline_metrics': baseline_metrics,
            'all_versions': len(optimized_versions)
        }
    
    def _generate_optimizations(self, code: str, analysis: Dict) -> List[str]:
        """Generate optimized versions of code."""
        optimizations = []
        
        # Apply loop unrolling for small loops
        loop_unrolled = self._unroll_small_loops(code)
        if loop_unrolled != code:
            optimizations.append(loop_unrolled)
        
        # Apply caching decorator
        cached_version = self._add_caching(code)
        if cached_version != code:
            optimizations.append(cached_version)
        
        # Apply list comprehension conversion
        list_comp_version = self._convert_to_list_comprehension(code)
        if list_comp_version != code:
            optimizations.append(list_comp_version)
        
        # Apply NumPy vectorization hints
        vectorized = self._add_vectorization_hints(code)
        if vectorized != code:
            optimizations.append(vectorized)
        
        return optimizations
    
    def _unroll_small_loops(self, code: str) -> str:
        """Unroll small loops for better performance."""
        # Simple pattern: for i in range(n) where n is small constant
        pattern = r'for\s+(\w+)\s+in\s+range\((\d+)\):\s*\n\s+([^\n]+)'
        
        def unroll_match(match):
            var, n, body = match.groups()
            n = int(n)
            if n <= 4:  # Only unroll small loops
                unrolled = ''
                for i in range(n):
                    unrolled += body.replace(var, str(i)) + '\n'
                return unrolled
            return match.group(0)
        
        return re.sub(pattern, unroll_match, code)
    
    def _add_caching(self, code: str) -> str:
        """Add caching decorator to functions."""
        # Add functools.lru_cache decorator
        if 'def ' in code and 'lru_cache' not in code:
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('def '):
                    indent = len(line) - len(line.lstrip())
                    lines.insert(i, ' ' * indent + '@functools.lru_cache(maxsize=None)')
                    # Add import if needed
                    if 'import functools' not in code:
                        lines.insert(0, 'import functools')
                    break
            return '\n'.join(lines)
        return code
    
    def _convert_to_list_comprehension(self, code: str) -> str:
        """Convert simple loops to list comprehensions."""
        # Pattern: result = []; for x in y: result.append(z)
        pattern = r'(\w+)\s*=\s*\[\]\s*\n\s*for\s+(\w+)\s+in\s+(\w+):\s*\n\s*\1\.append\(([^)]+)\)'
        replacement = r'\1 = [\4 for \2 in \3]'
        return re.sub(pattern, replacement, code)
    
    def _add_vectorization_hints(self, code: str) -> str:
        """Add hints for NumPy vectorization."""
        # Add numba JIT compilation for numerical functions
        if 'def ' in code and 'numba' not in code and any(
            op in code for op in ['np.', 'numpy.', 'math.']):
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('def '):
                    indent = len(line) - len(line.lstrip())
                    lines.insert(i, ' ' * indent + '@numba.jit(nopython=True)')
                    if 'import numba' not in code:
                        lines.insert(0, 'import numba')
                    break
            return '\n'.join(lines)
        return code
    
    def _compile_function(self, code: str, func_name: str) -> Callable:
        """Compile code string into callable function."""
        namespace = {}
        exec(code, namespace)
        return namespace[func_name]
    
    def _evaluate_performance(self, func: Callable, test_cases: List[Tuple],
                             metric_func: Callable = None) -> PerformanceMetrics:
        """Evaluate performance of a function."""
        metrics = PerformanceMetrics()
        
        # Measure execution time
        times = []
        for _ in range(10):
            start = time.time()
            if test_cases:
                for args, expected in test_cases:
                    try:
                        result = func(*args) if isinstance(args, tuple) else func(args)
                    except:
                        result = func(args)
            else:
                # Run with dummy data
                result = func()
            times.append(time.time() - start)
        
        metrics.execution_time = np.median(times)
        metrics.stability_score = 1.0 / (1.0 + np.std(times))
        
        # Measure accuracy
        if test_cases:
            correct = 0
            for args, expected in test_cases:
                try:
                    result = func(*args) if isinstance(args, tuple) else func(args)
                    if metric_func:
                        if metric_func(result, expected):
                            correct += 1
                    else:
                        if np.allclose(result, expected, rtol=1e-3):
                            correct += 1
                except:
                    pass
            metrics.accuracy = correct / len(test_cases)
        else:
            metrics.accuracy = 1.0
        
        # Estimate complexity
        try:
            source = inspect.getsource(func)
            analysis = self.analyzer.analyze(source)
            metrics.complexity_score = analysis.get('complexity_score', 0)
        except:
            pass
        
        return metrics


class AlgorithmDiscoveryEngine:
    """
    Automatically discovers new algorithms through search and evolution.
    """
    
    def __init__(self):
        self.discovered_algorithms = []
        self.search_space = []
        self.evolution_history = []
        
    def register_search_space(self, components: List[Dict]):
        """
        Register building blocks for algorithm composition.
        
        Each component is a dict with:
        - name: component name
        - code: code string
        - inputs: expected inputs
        - outputs: produced outputs
        - category: functional category
        """
        self.search_space.extend(components)
    
    def discover_algorithm(self, problem_description: str,
                          evaluation_func: Callable,
                          search_budget: int = 100) -> AlgorithmDiscovery:
        """
        Automatically discover an algorithm for a given problem.
        
        Args:
            problem_description: Description of the problem
            evaluation_func: Function to evaluate candidate algorithms
            search_budget: Number of candidates to evaluate
        """
        logger.info(f"Starting algorithm discovery for: {problem_description}")
        
        best_algorithm = None
        best_score = -float('inf')
        
        # Generate candidate algorithms
        candidates = self._generate_candidates(search_budget)
        
        for i, candidate in enumerate(candidates):
            try:
                # Evaluate candidate
                score = evaluation_func(candidate['code'])
                
                if score > best_score:
                    best_score = score
                    best_algorithm = candidate
                    
                logger.debug(f"Candidate {i}: score={score:.4f}")
                
            except Exception as e:
                logger.debug(f"Candidate {i} failed: {e}")
                continue
        
        if best_algorithm:
            discovery = AlgorithmDiscovery(
                name=f"discovered_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                description=problem_description,
                code=best_algorithm['code'],
                discovery_date=datetime.now().isoformat(),
                performance_metrics=PerformanceMetrics(accuracy=best_score),
                use_cases=[problem_description]
            )
            self.discovered_algorithms.append(discovery)
            return discovery
        
        return None
    
    def _generate_candidates(self, n: int) -> List[Dict]:
        """Generate candidate algorithms by composing components."""
        candidates = []
        
        # Simple composition: chain 2-4 components
        for _ in range(n):
            num_components = np.random.randint(2, 5)
            selected = np.random.choice(
                self.search_space, 
                size=min(num_components, len(self.search_space)),
                replace=False
            )
            
            # Compose into algorithm
            code = self._compose_components(selected)
            
            candidates.append({
                'components': [c['name'] for c in selected],
                'code': code
            })
        
        return candidates
    
    def _compose_components(self, components: List[Dict]) -> str:
        """Compose components into a complete algorithm."""
        code_lines = [
            "def discovered_algorithm(data):",
            "    # Auto-discovered algorithm",
            ""
        ]
        
        prev_var = "data"
        for i, comp in enumerate(components):
            lines = comp['code'].split('\n')
            for line in lines:
                # Replace input placeholder
                line = line.replace('{input}', prev_var)
                # Add output assignment
                if i < len(components) - 1:
                    line = line.replace('{output}', f'step_{i}')
                    prev_var = f'step_{i}'
                else:
                    line = line.replace('{output}', 'result')
                code_lines.append("    " + line)
        
        code_lines.append("    return result")
        return '\n'.join(code_lines)
    
    def evolve_algorithm(self, algorithm: AlgorithmDiscovery,
                        generations: int = 10,
                        population_size: int = 20) -> AlgorithmDiscovery:
        """
        Evolve an existing algorithm through genetic programming.
        """
        logger.info(f"Evolving algorithm: {algorithm.name}")
        
        # Initialize population
        population = [algorithm.code] * population_size
        
        for gen in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for code in population:
                try:
                    score = self._evaluate_algorithm_fitness(code)
                    fitness_scores.append(score)
                except:
                    fitness_scores.append(0)
            
            # Select top performers
            sorted_indices = np.argsort(fitness_scores)[::-1]
            elite = [population[i] for i in sorted_indices[:5]]
            
            # Create next generation
            new_population = elite.copy()
            
            while len(new_population) < population_size:
                parent1 = elite[np.random.randint(len(elite))]
                parent2 = elite[np.random.randint(len(elite))]
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)
            
            population = new_population
            
            logger.info(f"Generation {gen}: best fitness = {max(fitness_scores):.4f}")
        
        # Return best algorithm
        best_idx = np.argmax(fitness_scores)
        best_code = population[best_idx]
        
        evolved = AlgorithmDiscovery(
            name=f"{algorithm.name}_evolved",
            description=algorithm.description + " (evolved)",
            code=best_code,
            discovery_date=datetime.now().isoformat(),
            performance_metrics=PerformanceMetrics(accuracy=max(fitness_scores)),
            parent_algorithms=[algorithm.name],
            use_cases=algorithm.use_cases
        )
        
        return evolved
    
    def _evaluate_algorithm_fitness(self, code: str) -> float:
        """Evaluate fitness of an algorithm."""
        # Simplified fitness: code quality + execution success
        try:
            compile(code, '<string>', 'exec')
            
            # Prefer shorter, simpler code
            lines = code.strip().split('\n')
            complexity_penalty = len(lines) * 0.01
            
            return 1.0 - complexity_penalty
        except:
            return 0.0
    
    def _crossover(self, parent1: str, parent2: str) -> str:
        """Crossover two algorithm codes."""
        lines1 = parent1.split('\n')
        lines2 = parent2.split('\n')
        
        # Single point crossover
        point = min(len(lines1), len(lines2)) // 2
        child_lines = lines1[:point] + lines2[point:]
        
        return '\n'.join(child_lines)
    
    def _mutate(self, code: str, mutation_rate: float = 0.1) -> str:
        """Mutate algorithm code."""
        lines = code.split('\n')
        
        for i in range(len(lines)):
            if np.random.random() < mutation_rate:
                # Random mutation
                mutation_type = np.random.choice(['delete', 'duplicate', 'modify'])
                
                if mutation_type == 'delete' and len(lines) > 3:
                    lines.pop(i)
                elif mutation_type == 'duplicate':
                    lines.insert(i, lines[i])
                elif mutation_type == 'modify':
                    # Add a comment or modify slightly
                    lines[i] = lines[i] + "  # modified"
        
        return '\n'.join(lines)


class SelfReflectionEngine:
    """
    Enables the system to reflect on its own performance and identify improvements.
    """
    
    def __init__(self):
        self.reflection_logs = []
        self.improvement_suggestions = []
        self.performance_history = defaultdict(list)
        
    def reflect_on_performance(self, module_name: str, 
                              metrics: PerformanceMetrics,
                              context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Reflect on performance of a module and suggest improvements.
        """
        reflection = {
            'timestamp': datetime.now().isoformat(),
            'module': module_name,
            'metrics': metrics,
            'context': context or {},
            'analysis': {},
            'suggestions': []
        }
        
        # Track performance history
        self.performance_history[module_name].append(metrics.composite_score())
        
        # Analyze trends
        history = self.performance_history[module_name]
        if len(history) >= 3:
            # Check for degradation
            recent_avg = np.mean(history[-3:])
            previous_avg = np.mean(history[-6:-3]) if len(history) >= 6 else history[0]
            
            if recent_avg < previous_avg * 0.9:
                reflection['analysis']['degradation'] = True
                reflection['suggestions'].append({
                    'type': 'performance_degradation',
                    'severity': 'high',
                    'message': f"Performance degraded by {(1 - recent_avg/previous_avg)*100:.1f}%",
                    'action': 'Review recent changes and consider rollback'
                })
        
        # Check for bottlenecks
        if metrics.execution_time > 1.0:  # Slow execution
            reflection['suggestions'].append({
                'type': 'optimization',
                'severity': 'medium',
                'message': f"Execution time ({metrics.execution_time:.3f}s) exceeds threshold",
                'action': 'Consider algorithmic optimization or caching'
            })
        
        if metrics.memory_usage > 1000:  # High memory
            reflection['suggestions'].append({
                'type': 'memory',
                'severity': 'medium',
                'message': f"Memory usage ({metrics.memory_usage:.1f}MB) is high",
                'action': 'Consider memory-efficient data structures'
            })
        
        if metrics.accuracy < 0.9:  # Low accuracy
            reflection['suggestions'].append({
                'type': 'accuracy',
                'severity': 'high',
                'message': f"Accuracy ({metrics.accuracy:.2f}) below threshold",
                'action': 'Review model architecture or training data'
            })
        
        self.reflection_logs.append(reflection)
        return reflection
    
    def identify_recurring_issues(self) -> List[Dict]:
        """Identify patterns of recurring issues."""
        issue_counts = defaultdict(lambda: {'count': 0, 'modules': set()})
        
        for log in self.reflection_logs:
            for suggestion in log['suggestions']:
                issue_type = suggestion['type']
                issue_counts[issue_type]['count'] += 1
                issue_counts[issue_type]['modules'].add(log['module'])
        
        recurring = []
        for issue_type, data in issue_counts.items():
            if data['count'] >= 3:  # Threshold for recurring
                recurring.append({
                    'issue_type': issue_type,
                    'occurrences': data['count'],
                    'affected_modules': list(data['modules']),
                    'priority': 'high' if data['count'] > 5 else 'medium'
                })
        
        return recurring
    
    def generate_improvement_plan(self) -> Dict[str, Any]:
        """Generate comprehensive improvement plan."""
        plan = {
            'generated_at': datetime.now().isoformat(),
            'recurring_issues': self.identify_recurring_issues(),
            'module_specific': {},
            'system_wide_suggestions': []
        }
        
        # Analyze each module
        for module, history in self.performance_history.items():
            if len(history) >= 5:
                trend = np.polyfit(range(len(history)), history, 1)[0]
                plan['module_specific'][module] = {
                    'trend': 'improving' if trend > 0.01 else 'degrading' if trend < -0.01 else 'stable',
                    'trend_slope': trend,
                    'latest_score': history[-1],
                    'avg_score': np.mean(history)
                }
        
        # System-wide suggestions
        all_scores = [s for scores in self.performance_history.values() for s in scores]
        if all_scores:
            overall_avg = np.mean(all_scores)
            if overall_avg < 0.7:
                plan['system_wide_suggestions'].append({
                    'priority': 'critical',
                    'message': 'Overall system performance below threshold',
                    'action': 'Conduct comprehensive system review'
                })
        
        return plan


class SelfImprovementManager:
    """
    Main manager for self-improvement capabilities.
    Coordinates code optimization, algorithm discovery, and self-reflection.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.code_optimizer = CodeOptimizer()
        self.discovery_engine = AlgorithmDiscoveryEngine()
        self.reflection_engine = SelfReflectionEngine()
        
        self.improvement_history = []
        self.active_improvements = {}
        
        # Performance thresholds
        self.thresholds = {
            'execution_time': self.config.get('execution_time_threshold', 1.0),
            'memory_usage': self.config.get('memory_usage_threshold', 1000),
            'accuracy': self.config.get('accuracy_threshold', 0.9)
        }
        
    def register_component(self, name: str, func: Callable, 
                          test_cases: List[Tuple] = None,
                          metric_func: Callable = None):
        """Register a component for self-improvement."""
        self.active_improvements[name] = {
            'func': func,
            'test_cases': test_cases,
            'metric_func': metric_func,
            'last_optimized': None,
            'version_history': []
        }
        
    def run_improvement_cycle(self, target_component: str = None) -> Dict[str, Any]:
        """
        Run a complete self-improvement cycle.
        """
        cycle_results = {
            'timestamp': datetime.now().isoformat(),
            'optimizations': [],
            'reflections': [],
            'discoveries': []
        }
        
        components = ([target_component] if target_component 
                     else list(self.active_improvements.keys()))
        
        for component_name in components:
            if component_name not in self.active_improvements:
                continue
            
            component = self.active_improvements[component_name]
            
            # 1. Measure current performance
            baseline_metrics = self.code_optimizer._evaluate_performance(
                component['func'],
                component['test_cases'],
                component['metric_func']
            )
            
            # 2. Reflect on performance
            reflection = self.reflection_engine.reflect_on_performance(
                component_name,
                baseline_metrics,
                context={'test_cases': len(component['test_cases'] or [])}
            )
            cycle_results['reflections'].append(reflection)
            
            # 3. Optimize if needed
            needs_optimization = (
                baseline_metrics.execution_time > self.thresholds['execution_time'] or
                baseline_metrics.memory_usage > self.thresholds['memory_usage'] or
                baseline_metrics.accuracy < self.thresholds['accuracy']
            )
            
            if needs_optimization:
                logger.info(f"Optimizing {component_name}...")
                optimization_result = self.code_optimizer.optimize(
                    component['func'],
                    component['test_cases'],
                    component['metric_func']
                )
                cycle_results['optimizations'].append({
                    'component': component_name,
                    'result': optimization_result
                })
            
            # 4. Record improvement
            self.improvement_history.append({
                'component': component_name,
                'baseline': baseline_metrics,
                'reflection': reflection,
                'optimized': needs_optimization
            })
        
        return cycle_results
    
    def discover_new_algorithm(self, problem_description: str,
                              evaluation_func: Callable,
                              search_budget: int = 100) -> Optional[AlgorithmDiscovery]:
        """Discover a new algorithm for a specific problem."""
        discovery = self.discovery_engine.discover_algorithm(
            problem_description,
            evaluation_func,
            search_budget
        )
        
        if discovery:
            logger.info(f"Discovered new algorithm: {discovery.name}")
            
            # Evolve the algorithm
            evolved = self.discovery_engine.evolve_algorithm(discovery, generations=5)
            
            return evolved
        
        return None
    
    def get_improvement_report(self) -> Dict[str, Any]:
        """Generate comprehensive improvement report."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_improvements': len(self.improvement_history),
            'components_tracked': len(self.active_improvements),
            'performance_summary': {},
            'improvement_plan': self.reflection_engine.generate_improvement_plan(),
            'discovered_algorithms': [
                {
                    'name': alg.name,
                    'description': alg.description,
                    'performance': alg.performance_metrics.composite_score(),
                    'discovery_date': alg.discovery_date
                }
                for alg in self.discovery_engine.discovered_algorithms
            ]
        }
        
        # Performance by component
        for name, component in self.active_improvements.items():
            history = [
                h for h in self.improvement_history 
                if h['component'] == name
            ]
            if history:
                recent = history[-1]['baseline']
                report['performance_summary'][name] = {
                    'current_score': recent.composite_score(),
                    'accuracy': recent.accuracy,
                    'execution_time': recent.execution_time,
                    'improvement_count': len(history)
                }
        
        return report
    
    def save_state(self, path: str):
        """Save self-improvement state."""
        state = {
            'improvement_history': self.improvement_history,
            'active_improvements': list(self.active_improvements.keys()),
            'discovered_algorithms': self.discovery_engine.discovered_algorithms,
            'reflection_logs': self.reflection_engine.reflection_logs,
            'config': self.config
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Self-improvement state saved to {path}")
    
    def load_state(self, path: str):
        """Load self-improvement state."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.improvement_history = state['improvement_history']
        self.discovery_engine.discovered_algorithms = state['discovered_algorithms']
        self.reflection_engine.reflection_logs = state['reflection_logs']
        
        logger.info(f"Self-improvement state loaded from {path}")


# Predefined optimization templates
OPTIMIZATION_TEMPLATES = {
    'loop_vectorization': '''
# Vectorized version using NumPy
import numpy as np

@numba.jit(nopython=True)
def vectorized_{func_name}(data):
    return np.sum(data ** 2, axis=1)  # Example vectorization
''',
    'memoization': '''
from functools import lru_cache

@lru_cache(maxsize=None)
def cached_{func_name}(*args):
    # Cached version of function
    return original_function(*args)
''',
    'parallel_processing': '''
from multiprocessing import Pool

def parallel_{func_name}(data_chunks):
    with Pool() as pool:
        results = pool.map({func_name}, data_chunks)
    return results
''',
    'gpu_acceleration': '''
import cupy as cp

def gpu_{func_name}(data):
    data_gpu = cp.asarray(data)
    result_gpu = {func_name}(data_gpu)
    return cp.asnumpy(result_gpu)
'''
}


def demonstrate_self_improvement():
    """Demonstrate self-improvement capabilities."""
    print("=" * 60)
    print("Self-Improvement System Demonstration")
    print("=" * 60)
    
    # Create manager
    manager = SelfImprovementManager({
        'execution_time_threshold': 0.1,
        'accuracy_threshold': 0.95
    })
    
    # Example function to optimize
    def slow_function(n):
        """Intentionally slow function for demonstration."""
        result = []
        for i in range(n):
            for j in range(n):
                result.append(i * j)
        return sum(result)
    
    # Register component
    manager.register_component(
        'slow_function',
        slow_function,
        test_cases=[((100,), sum(i*j for i in range(100) for j in range(100)))],
    )
    
    # Run improvement cycle
    print("\nRunning improvement cycle...")
    results = manager.run_improvement_cycle()
    
    print(f"\nReflections: {len(results['reflections'])}")
    print(f"Optimizations attempted: {len(results['optimizations'])}")
    
    # Get report
    report = manager.get_improvement_report()
    print(f"\nTotal improvements tracked: {report['total_improvements']}")
    print(f"Components tracked: {report['components_tracked']}")
    
    # Demonstrate algorithm discovery
    print("\n" + "=" * 60)
    print("Algorithm Discovery Demo")
    print("=" * 60)
    
    # Register some components
    manager.discovery_engine.register_search_space([
        {
            'name': 'normalize',
            'code': '{output} = ({input} - np.mean({input})) / np.std({input})',
            'category': 'preprocessing'
        },
        {
            'name': 'power_transform',
            'code': '{output} = np.power({input}, 2)',
            'category': 'feature_engineering'
        },
        {
            'name': 'aggregate',
            'code': '{output} = np.sum({input}, axis=0)',
            'category': 'aggregation'
        }
    ])
    
    def simple_evaluator(code):
        # Simple evaluator for demo
        return 0.8 if 'normalize' in code else 0.5
    
    discovery = manager.discover_new_algorithm(
        "Optimize material feature processing",
        simple_evaluator,
        search_budget=10
    )
    
    if discovery:
        print(f"\nDiscovered algorithm: {discovery.name}")
        print(f"Performance: {discovery.performance_metrics.composite_score():.4f}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_self_improvement()
