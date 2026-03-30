"""
Cross-Scale Validation Tools

Provides validation methods for checking consistency between
different scales in multiscale simulations.
"""
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
import json


@dataclass
class ValidationResult:
    """Result of a validation check."""
    test_name: str
    passed: bool
    score: float
    details: Dict
    message: str


class CrossScaleValidator:
    """
    Main validator for cross-scale consistency.
    """
    
    def __init__(self, tolerance: float = 1e-3):
        """
        Initialize validator.
        
        Args:
            tolerance: Tolerance for numerical comparisons
        """
        self.tolerance = tolerance
        self.results = []
    
    def validate_energy_conservation(self,
                                    energies: np.ndarray,
                                    times: np.ndarray = None) -> ValidationResult:
        """
        Check energy conservation in simulation.
        
        Args:
            energies: Array of total energies
            times: Optional time points
            
        Returns:
            ValidationResult
        """
        if len(energies) < 2:
            return ValidationResult(
                test_name='energy_conservation',
                passed=False,
                score=0.0,
                details={'n_samples': len(energies)},
                message='Insufficient data points'
            )
        
        # Check energy drift
        e_mean = np.mean(energies)
        e_std = np.std(energies)
        
        # Linear fit to check drift
        if times is None:
            times = np.arange(len(energies))
        
        coeffs = np.polyfit(times, energies, 1)
        drift = coeffs[0]  # Slope
        
        # Score based on drift relative to energy scale
        drift_score = abs(drift * len(energies) / e_mean) if e_mean != 0 else 0
        
        passed = drift_score < self.tolerance * 10  # Relaxed tolerance for drift
        
        return ValidationResult(
            test_name='energy_conservation',
            passed=passed,
            score=drift_score,
            details={
                'mean_energy': float(e_mean),
                'std_energy': float(e_std),
                'drift': float(drift),
                'drift_score': float(drift_score)
            },
            message=f"Energy drift: {drift:.6e} per step" if passed else \
                   f"Energy drift too large: {drift:.6e}"
        )
    
    def validate_force_consistency(self,
                                  atom_forces: np.ndarray,
                                  cg_forces: np.ndarray,
                                  atom_to_cg_mapping: np.ndarray) -> ValidationResult:
        """
        Check consistency between atom and CG forces.
        
        Args:
            atom_forces: (N_atoms, 3) atomic forces
            cg_forces: (N_cg, 3) coarse-grained forces
            atom_to_cg_mapping: Mapping from atoms to CG beads
            
        Returns:
            ValidationResult
        """
        # Sum atom forces for each CG bead
        n_cg = len(cg_forces)
        summed_atom_forces = np.zeros_like(cg_forces)
        
        for cg_idx in range(n_cg):
            atom_indices = np.where(atom_to_cg_mapping == cg_idx)[0]
            summed_atom_forces[cg_idx] = atom_forces[atom_indices].sum(axis=0)
        
        # Compare
        diff = np.linalg.norm(summed_atom_forces - cg_forces, axis=1)
        mag = np.linalg.norm(cg_forces, axis=1) + 1e-8
        
        relative_error = np.mean(diff / mag)
        
        passed = relative_error < self.tolerance * 100  # Relaxed for CG
        
        return ValidationResult(
            test_name='force_consistency',
            passed=passed,
            score=float(relative_error),
            details={
                'max_diff': float(np.max(diff)),
                'mean_diff': float(np.mean(diff)),
                'relative_error': float(relative_error)
            },
            message=f"Force consistency: {relative_error:.4f}" if passed else \
                   f"Force inconsistency detected: {relative_error:.4f}"
        )
    
    def validate_thermodynamic_consistency(self,
                                          atom_energies: np.ndarray,
                                          cg_energies: np.ndarray,
                                          temperature: float = 300.0) -> ValidationResult:
        """
        Check thermodynamic consistency between scales.
        
        Args:
            atom_energies: Energies from atomistic simulation
            cg_energies: Energies from CG simulation
            temperature: Temperature in Kelvin
            
        Returns:
            ValidationResult
        """
        # Calculate free energy difference (simplified)
        # Using BAR or MBAR would be more accurate
        
        atom_mean = np.mean(atom_energies)
        atom_std = np.std(atom_energies)
        
        cg_mean = np.mean(cg_energies)
        cg_std = np.std(cg_energies)
        
        # Energy difference
        energy_diff = abs(atom_mean - cg_mean)
        
        # Fluctuation comparison (heat capacity proxy)
        fluct_ratio = atom_std / (cg_std + 1e-8)
        
        passed = energy_diff < self.tolerance * abs(atom_mean) * 10
        
        return ValidationResult(
            test_name='thermodynamic_consistency',
            passed=passed,
            score=float(energy_diff / abs(atom_mean)) if atom_mean != 0 else 0,
            details={
                'atom_mean': float(atom_mean),
                'atom_std': float(atom_std),
                'cg_mean': float(cg_mean),
                'cg_std': float(cg_std),
                'energy_diff': float(energy_diff),
                'fluctuation_ratio': float(fluct_ratio)
            },
            message=f"Energy difference: {energy_diff:.4f}" if passed else \
                   f"Large energy difference: {energy_diff:.4f}"
        )
    
    def validate_structure_consistency(self,
                                      atom_positions: np.ndarray,
                                      cg_positions: np.ndarray,
                                      atom_to_cg_mapping: np.ndarray,
                                      cg_mapping: 'CGMapping') -> ValidationResult:
        """
        Check structural consistency between scales.
        
        Args:
            atom_positions: Atomic positions
            cg_positions: CG positions
            atom_to_cg_mapping: Atom-to-CG mapping
            cg_mapping: CGMapping object
            
        Returns:
            ValidationResult
        """
        # Compute CG positions from atoms
        computed_cg = np.zeros_like(cg_positions)
        
        for cg_idx in range(len(cg_positions)):
            atom_indices = np.where(atom_to_cg_mapping == cg_idx)[0]
            if len(atom_indices) > 0:
                computed_cg[cg_idx] = atom_positions[atom_indices].mean(axis=0)
        
        # Compare
        rmsd = np.sqrt(np.mean((computed_cg - cg_positions) ** 2))
        
        passed = rmsd < self.tolerance * 100  # Relaxed for CG
        
        return ValidationResult(
            test_name='structure_consistency',
            passed=passed,
            score=float(rmsd),
            details={
                'rmsd': float(rmsd),
                'max_deviation': float(np.max(np.abs(computed_cg - cg_positions)))
            },
            message=f"Structure RMSD: {rmsd:.4f} Å" if passed else \
                   f"Large structural deviation: {rmsd:.4f} Å"
        )
    
    def run_all_validations(self,
                           atom_data: Dict,
                           cg_data: Dict,
                           mapping: 'CGMapping') -> List[ValidationResult]:
        """
        Run all validation checks.
        
        Args:
            atom_data: Dictionary with atom-scale data
            cg_data: Dictionary with CG-scale data
            mapping: CGMapping object
            
        Returns:
            List of ValidationResults
        """
        results = []
        
        # Energy conservation
        if 'energies' in atom_data:
            results.append(self.validate_energy_conservation(
                atom_data['energies'],
                atom_data.get('times')
            ))
        
        # Force consistency
        if 'forces' in atom_data and 'forces' in cg_data:
            results.append(self.validate_force_consistency(
                atom_data['forces'],
                cg_data['forces'],
                mapping.atom_to_bead
            ))
        
        # Thermodynamic consistency
        if 'energies' in atom_data and 'energies' in cg_data:
            results.append(self.validate_thermodynamic_consistency(
                atom_data['energies'],
                cg_data['energies'],
                atom_data.get('temperature', 300.0)
            ))
        
        # Structure consistency
        if 'positions' in atom_data and 'positions' in cg_data:
            results.append(self.validate_structure_consistency(
                atom_data['positions'],
                cg_data['positions'],
                mapping.atom_to_bead,
                mapping
            ))
        
        self.results = results
        return results
    
    def generate_report(self) -> str:
        """Generate validation report."""
        lines = ["=" * 60]
        lines.append("Cross-Scale Validation Report")
        lines.append("=" * 60)
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        lines.append(f"\nOverall: {passed}/{total} tests passed")
        lines.append("")
        
        for result in self.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            lines.append(f"{status}: {result.test_name}")
            lines.append(f"  Score: {result.score:.6f}")
            lines.append(f"  {result.message}")
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


class EnergyConsistencyCheck:
    """
    Detailed energy consistency checks for QM/MM calculations.
    """
    
    def __init__(self):
        self.energy_components = {}
    
    def check_qm_mm_partitioning(self,
                                 total_energy: float,
                                 qm_energy: float,
                                 mm_energy: float,
                                 coupling_energy: float) -> ValidationResult:
        """
        Verify additive QM/MM energy partitioning.
        
        Args:
            total_energy: Total QM/MM energy
            qm_energy: QM subsystem energy
            mm_energy: MM subsystem energy
            coupling_energy: QM/MM coupling energy
            
        Returns:
            ValidationResult
        """
        expected = qm_energy + mm_energy + coupling_energy
        diff = abs(total_energy - expected)
        
        # Check with reasonable tolerance
        tolerance = 1e-4 * abs(total_energy)
        passed = diff < tolerance
        
        return ValidationResult(
            test_name='qm_mm_energy_partitioning',
            passed=passed,
            score=diff,
            details={
                'total': float(total_energy),
                'qm': float(qm_energy),
                'mm': float(mm_energy),
                'coupling': float(coupling_energy),
                'expected': float(expected),
                'difference': float(diff)
            },
            message=f"Energy partitioning correct" if passed else \
                   f"Energy mismatch: {diff:.6f}"
        )
    
    def check_virial_consistency(self,
                                stress_tensor: np.ndarray,
                                forces: np.ndarray,
                                positions: np.ndarray,
                                volume: float) -> ValidationResult:
        """
        Check virial consistency (pressure from forces vs stress).
        
        Args:
            stress_tensor: (3, 3) stress tensor
            forces: (N, 3) forces
            positions: (N, 3) positions
            volume: System volume
            
        Returns:
            ValidationResult
        """
        # Calculate virial
        virial = np.zeros((3, 3))
        for i in range(len(forces)):
            for alpha in range(3):
                for beta in range(3):
                    virial[alpha, beta] += forces[i, alpha] * positions[i, beta]
        
        # Compare to stress tensor
        # stress = virial / volume (simplified)
        calculated_stress = virial / volume
        
        diff = np.linalg.norm(stress_tensor - calculated_stress)
        
        passed = diff < 1e-3  # Tolerance in stress units
        
        return ValidationResult(
            test_name='virial_consistency',
            passed=passed,
            score=float(diff),
            details={
                'input_stress': stress_tensor.tolist(),
                'calculated_stress': calculated_stress.tolist(),
                'difference': float(diff)
            },
            message="Virial consistent" if passed else \
                   f"Virial mismatch: {diff:.4f}"
        )
    
    def check_gradient_consistency(self,
                                  energy_func: Callable,
                                  positions: np.ndarray,
                                  forces: np.ndarray,
                                  dx: float = 1e-5) -> ValidationResult:
        """
        Check that forces are consistent with energy gradients.
        
        Args:
            energy_func: Function to compute energy at given positions
            positions: Atomic positions
            forces: Forces from calculation
            dx: Finite difference step
            
        Returns:
            ValidationResult
        """
        n_atoms = len(positions)
        numerical_forces = np.zeros_like(forces)
        
        for i in range(n_atoms):
            for alpha in range(3):
                # Forward difference
                pos_plus = positions.copy()
                pos_plus[i, alpha] += dx
                e_plus = energy_func(pos_plus)
                
                # Backward difference
                pos_minus = positions.copy()
                pos_minus[i, alpha] -= dx
                e_minus = energy_func(pos_minus)
                
                # Central difference
                numerical_forces[i, alpha] = -(e_plus - e_minus) / (2 * dx)
        
        # Compare
        diff = np.linalg.norm(forces - numerical_forces)
        rel_diff = diff / (np.linalg.norm(forces) + 1e-8)
        
        passed = rel_diff < 1e-3
        
        return ValidationResult(
            test_name='gradient_consistency',
            passed=passed,
            score=float(rel_diff),
            details={
                'absolute_diff': float(diff),
                'relative_diff': float(rel_diff)
            },
            message=f"Gradient check passed: {rel_diff:.6f}" if passed else \
                   f"Gradient mismatch: {rel_diff:.6f}"
        )


def compare_trajectories(ref_trajectory: np.ndarray,
                        test_trajectory: np.ndarray,
                        align: bool = True) -> Dict:
    """
    Compare two trajectories.
    
    Args:
        ref_trajectory: Reference trajectory (n_frames, n_atoms, 3)
        test_trajectory: Test trajectory (n_frames, n_atoms, 3)
        align: Whether to align trajectories
        
    Returns:
        Comparison metrics
    """
    if len(ref_trajectory) != len(test_trajectory):
        return {'error': 'Trajectory lengths do not match'}
    
    n_frames = len(ref_trajectory)
    
    rmsds = []
    for i in range(n_frames):
        ref = ref_trajectory[i]
        test = test_trajectory[i]
        
        if align:
            # Simple centering (full Kabsch alignment would be better)
            ref -= ref.mean(axis=0)
            test -= test.mean(axis=0)
        
        rmsd = np.sqrt(np.mean((ref - test) ** 2))
        rmsds.append(rmsd)
    
    return {
        'mean_rmsd': float(np.mean(rmsds)),
        'max_rmsd': float(np.max(rmsds)),
        'std_rmsd': float(np.std(rmsds)),
        'rmsds': [float(r) for r in rmsds]
    }
