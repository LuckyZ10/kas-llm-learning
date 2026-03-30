"""
Physics Law Validator

Validates that AI models and discovered equations respect fundamental
physical laws and constraints.
"""

import torch
import numpy as np
from typing import Dict, List, Callable, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import warnings


class ValidationResult(Enum):
    """Validation result types."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_TESTED = "not_tested"


@dataclass
class PhysicsTest:
    """A physics validation test."""
    name: str
    description: str
    test_fn: Callable
    required: bool = True
    tolerance: float = 1e-6
    
    def run(self, *args, **kwargs) -> Tuple[ValidationResult, Dict]:
        """Run the test and return result."""
        try:
            passed, details = self.test_fn(*args, **kwargs, tolerance=self.tolerance)
            result = ValidationResult.PASS if passed else ValidationResult.FAIL
            return result, details
        except Exception as e:
            return ValidationResult.FAIL, {'error': str(e)}


class PhysicsLawValidator:
    """
    Validates AI models against physical laws.
    
    Checks:
    - Conservation laws (energy, momentum, angular momentum)
    - Symmetries (translational, rotational, time-reversal)
    - Physical constraints (positivity, boundedness)
    - Dimensional consistency
    - Thermodynamic laws
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize validator.
        
        Args:
            tolerance: Numerical tolerance for tests
        """
        self.tolerance = tolerance
        self.tests: Dict[str, PhysicsTest] = {}
        self.results: Dict[str, Dict] = {}
        
        # Register default tests
        self._register_default_tests()
    
    def _register_default_tests(self):
        """Register default physics validation tests."""
        # Conservation law tests
        self.add_test(PhysicsTest(
            name="energy_conservation",
            description="Check if energy is conserved",
            test_fn=self._test_energy_conservation,
            tolerance=self.tolerance
        ))
        
        self.add_test(PhysicsTest(
            name="linear_momentum_conservation",
            description="Check if linear momentum is conserved",
            test_fn=self._test_linear_momentum_conservation,
            tolerance=self.tolerance
        ))
        
        self.add_test(PhysicsTest(
            name="angular_momentum_conservation",
            description="Check if angular momentum is conserved",
            test_fn=self._test_angular_momentum_conservation,
            tolerance=self.tolerance
        ))
        
        # Symmetry tests
        self.add_test(PhysicsTest(
            name="translational_symmetry",
            description="Check translational invariance",
            test_fn=self._test_translational_symmetry,
            tolerance=self.tolerance
        ))
        
        self.add_test(PhysicsTest(
            name="rotational_symmetry",
            description="Check rotational invariance",
            test_fn=self._test_rotational_symmetry,
            tolerance=self.tolerance
        ))
        
        self.add_test(PhysicsTest(
            name="time_reversal_symmetry",
            description="Check time-reversal symmetry",
            test_fn=self._test_time_reversal_symmetry,
            tolerance=self.tolerance
        ))
        
        # Physical constraint tests
        self.add_test(PhysicsTest(
            name="energy_positivity",
            description="Check if energy is positive definite",
            test_fn=self._test_energy_positivity,
            tolerance=self.tolerance
        ))
        
        self.add_test(PhysicsTest(
            name="force_action_reaction",
            description="Check Newton's 3rd law",
            test_fn=self._test_action_reaction,
            tolerance=self.tolerance
        ))
    
    def add_test(self, test: PhysicsTest):
        """Add a validation test."""
        self.tests[test.name] = test
    
    def validate_model(
        self,
        model: torch.nn.Module,
        test_data: Dict[str, torch.Tensor],
        test_names: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Validate a model against physics laws.
        
        Args:
            model: Neural network model
            test_data: Dictionary with test data
            test_names: Specific tests to run (None = all)
            
        Returns:
            Dictionary of test results
        """
        results = {}
        
        tests_to_run = test_names or list(self.tests.keys())
        
        for test_name in tests_to_run:
            if test_name in self.tests:
                test = self.tests[test_name]
                result, details = test.run(model, test_data)
                results[test_name] = {
                    'result': result,
                    'details': details,
                    'description': test.description
                }
        
        self.results = results
        return results
    
    def validate_trajectory(
        self,
        trajectory: np.ndarray,
        time_step: float = 0.001
    ) -> Dict[str, Dict]:
        """
        Validate a trajectory against physics laws.
        
        Args:
            trajectory: Trajectory array [n_steps, n_particles, 6]
                       (positions and velocities)
            time_step: Time step size
            
        Returns:
            Dictionary of test results
        """
        results = {}
        
        # Extract positions and velocities
        n_steps = trajectory.shape[0]
        
        if trajectory.shape[-1] == 6:
            positions = trajectory[:, :, :3]
            velocities = trajectory[:, :, 3:]
        else:
            positions = trajectory
            velocities = np.gradient(positions, time_step, axis=0)
        
        # Check energy conservation
        energies = self._compute_energies(positions, velocities)
        energy_drift = np.abs(energies[-1] - energies[0]) / energies[0]
        
        results['energy_conservation'] = {
            'result': ValidationResult.PASS if energy_drift < 0.01 else ValidationResult.FAIL,
            'energy_drift': float(energy_drift),
            'energy_mean': float(np.mean(energies)),
            'energy_std': float(np.std(energies))
        }
        
        # Check momentum conservation
        momenta = np.sum(velocities, axis=1)  # [n_steps, 3]
        momentum_drift = np.std(momenta, axis=0).mean()
        
        results['momentum_conservation'] = {
            'result': ValidationResult.PASS if momentum_drift < self.tolerance else ValidationResult.FAIL,
            'momentum_drift': float(momentum_drift),
            'mean_momentum': momenta.mean(axis=0).tolist()
        }
        
        return results
    
    def _test_energy_conservation(
        self,
        model: torch.nn.Module,
        test_data: Dict[str, torch.Tensor],
        tolerance: float = 1e-6
    ) -> Tuple[bool, Dict]:
        """Test if model conserves energy."""
        if 'positions' not in test_data or 'velocities' not in test_data:
            return False, {'error': 'Missing position or velocity data'}
        
        pos = test_data['positions']
        vel = test_data['velocities']
        
        # Get energy predictions
        with torch.no_grad():
            if hasattr(model, 'forward'):
                output = model(pos)
                if isinstance(output, dict) and 'energy' in output:
                    energies = output['energy']
                else:
                    energies = output
            else:
                return False, {'error': 'Model does not have forward method'}
        
        # Check energy variance
        energy_var = torch.var(energies).item()
        passed = energy_var < tolerance
        
        return passed, {
            'energy_variance': energy_var,
            'energy_mean': torch.mean(energies).item()
        }
    
    def _test_linear_momentum_conservation(
        self,
        model: torch.nn.Module,
        test_data: Dict[str, torch.Tensor],
        tolerance: float = 1e-6
    ) -> Tuple[bool, Dict]:
        """Test if model conserves linear momentum."""
        if 'velocities' not in test_data or 'masses' not in test_data:
            return False, {'error': 'Missing velocity or mass data'}
        
        vel = test_data['velocities']
        masses = test_data['masses']
        
        # Compute total momentum
        momentum = torch.sum(masses.unsqueeze(-1) * vel, dim=1)
        momentum_var = torch.var(momentum, dim=0).mean().item()
        
        passed = momentum_var < tolerance
        
        return passed, {
            'momentum_variance': momentum_var,
            'mean_momentum': torch.mean(momentum, dim=0).tolist()
        }
    
    def _test_angular_momentum_conservation(
        self,
        model: torch.nn.Module,
        test_data: Dict[str, torch.Tensor],
        tolerance: float = 1e-6
    ) -> Tuple[bool, Dict]:
        """Test if model conserves angular momentum."""
        if 'positions' not in test_data or 'velocities' not in test_data:
            return False, {'error': 'Missing position or velocity data'}
        
        pos = test_data['positions']
        vel = test_data['velocities']
        masses = test_data.get('masses', torch.ones(pos.shape[0], pos.shape[1]))
        
        # Compute angular momentum L = r x p
        p = masses.unsqueeze(-1) * vel
        L = torch.cross(pos, p, dim=-1)
        L_total = torch.sum(L, dim=1)
        
        L_var = torch.var(L_total, dim=0).mean().item()
        passed = L_var < tolerance
        
        return passed, {
            'angular_momentum_variance': L_var,
            'mean_L': torch.mean(L_total, dim=0).tolist()
        }
    
    def _test_translational_symmetry(
        self,
        model: torch.nn.Module,
        test_data: Dict[str, torch.Tensor],
        tolerance: float = 1e-6
    ) -> Tuple[bool, Dict]:
        """Test translational invariance."""
        if 'positions' not in test_data:
            return False, {'error': 'Missing position data'}
        
        pos = test_data['positions']
        shift = torch.randn(3, device=pos.device)
        
        with torch.no_grad():
            output1 = model(pos)
            output2 = model(pos + shift)
            
            if isinstance(output1, dict):
                # Compare energies (should be invariant)
                if 'energy' in output1:
                    diff = torch.abs(output1['energy'] - output2['energy']).max().item()
                else:
                    return False, {'error': 'No energy output'}
            else:
                diff = torch.abs(output1 - output2).max().item()
        
        passed = diff < tolerance
        
        return passed, {'max_difference': diff}
    
    def _test_rotational_symmetry(
        self,
        model: torch.nn.Module,
        test_data: Dict[str, torch.Tensor],
        tolerance: float = 1e-6
    ) -> Tuple[bool, Dict]:
        """Test rotational invariance."""
        if 'positions' not in test_data:
            return False, {'error': 'Missing position data'}
        
        pos = test_data['positions']
        
        # Random rotation matrix
        angle = np.random.uniform(0, 2 * np.pi)
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        
        R = self._rotation_matrix(axis, angle)
        R_tensor = torch.tensor(R, dtype=pos.dtype, device=pos.device)
        
        with torch.no_grad():
            output1 = model(pos)
            pos_rotated = torch.matmul(pos, R_tensor.T)
            output2 = model(pos_rotated)
            
            if isinstance(output1, dict) and 'energy' in output1:
                diff = torch.abs(output1['energy'] - output2['energy']).max().item()
            else:
                diff = torch.abs(output1 - output2).max().item()
        
        passed = diff < tolerance
        
        return passed, {'max_difference': diff}
    
    def _test_time_reversal_symmetry(
        self,
        model: torch.nn.Module,
        test_data: Dict[str, torch.Tensor],
        tolerance: float = 1e-6
    ) -> Tuple[bool, Dict]:
        """Test time-reversal symmetry."""
        if 'velocities' not in test_data:
            return False, {'error': 'Missing velocity data'}
        
        # Time reversal: v -> -v
        vel = test_data['velocities']
        
        with torch.no_grad():
            output1 = model(test_data)
            test_data_reversed = test_data.copy()
            test_data_reversed['velocities'] = -vel
            output2 = model(test_data_reversed)
            
            # Energy should be the same
            if isinstance(output1, dict) and 'energy' in output1:
                diff = torch.abs(output1['energy'] - output2['energy']).max().item()
                passed = diff < tolerance
                return passed, {'max_difference': diff}
        
        return False, {'error': 'Could not test time reversal'}
    
    def _test_energy_positivity(
        self,
        model: torch.nn.Module,
        test_data: Dict[str, torch.Tensor],
        tolerance: float = 1e-6
    ) -> Tuple[bool, Dict]:
        """Test if energy is positive definite."""
        with torch.no_grad():
            output = model(test_data)
            
            if isinstance(output, dict) and 'energy' in output:
                energies = output['energy']
                min_energy = torch.min(energies).item()
                passed = min_energy >= -tolerance
                return passed, {'minimum_energy': min_energy}
        
        return False, {'error': 'No energy output'}
    
    def _test_action_reaction(
        self,
        model: torch.nn.Module,
        test_data: Dict[str, torch.Tensor],
        tolerance: float = 1e-6
    ) -> Tuple[bool, Dict]:
        """Test Newton's 3rd law (action-reaction)."""
        with torch.no_grad():
            output = model(test_data)
            
            if isinstance(output, dict) and 'forces' in output:
                forces = output['forces']
                # Sum of all forces should be zero for isolated system
                total_force = torch.sum(forces, dim=1)
                max_force = torch.abs(total_force).max().item()
                passed = max_force < tolerance
                return passed, {'max_total_force': max_force}
        
        return False, {'error': 'No force output'}
    
    def _rotation_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Generate rotation matrix from axis and angle."""
        # Rodrigues' rotation formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R
    
    def _compute_energies(
        self,
        positions: np.ndarray,
        velocities: np.ndarray
    ) -> np.ndarray:
        """Compute kinetic + potential energy (simplified)."""
        # Kinetic energy
        kinetic = 0.5 * np.sum(velocities ** 2, axis=(1, 2))
        
        # Potential energy (simplified harmonic)
        # In practice, use the actual potential
        potential = 0.5 * np.sum(positions ** 2, axis=(1, 2))
        
        return kinetic + potential
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of validation results."""
        if not self.results:
            return {'status': 'No results available'}
        
        summary = {
            'total_tests': len(self.results),
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'not_tested': 0,
            'tests': {}
        }
        
        for test_name, result in self.results.items():
            status = result['result']
            summary['tests'][test_name] = {
                'status': status.value,
                'description': result['description']
            }
            
            if status == ValidationResult.PASS:
                summary['passed'] += 1
            elif status == ValidationResult.FAIL:
                summary['failed'] += 1
            elif status == ValidationResult.WARNING:
                summary['warnings'] += 1
            else:
                summary['not_tested'] += 1
        
        summary['success_rate'] = summary['passed'] / summary['total_tests']
        
        return summary
    
    def print_report(self):
        """Print validation report."""
        summary = self.get_summary()
        
        print("=" * 60)
        print("PHYSICS VALIDATION REPORT")
        print("=" * 60)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Warnings: {summary['warnings']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print("-" * 60)
        
        for test_name, test_info in summary['tests'].items():
            status = test_info['status'].upper()
            print(f"[{status}] {test_name}: {test_info['description']}")
        
        print("=" * 60)
