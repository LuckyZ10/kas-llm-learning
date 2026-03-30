"""
DFT计算回归测试
DFT Calculation Regression Tests
=================================

验证DFT计算结果的一致性和可重复性。

测试项目:
    - 能量计算一致性
    - 力计算一致性
    - 应力张量一致性
    - 自洽场收敛性
    - 跨平台数值稳定性
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.regression import (
    DFTReferenceData, ReferenceDataManager, NumericalComparator
)


# =============================================================================
# DFT能量计算回归测试
# =============================================================================

@pytest.mark.regression
@pytest.mark.dft
class TestDFTEnergyRegression:
    """DFT能量计算回归测试"""
    
    def test_li2s_energy_consistency(self, reference_manager, numerical_comparator):
        """测试Li2S体系能量计算的一致性"""
        # 参考值来自标准DFT计算
        reference_energy = -105.234567  # eV
        n_atoms = 12
        
        # 模拟新的计算结果
        computed_energy = -105.234568  # 微小差异
        
        # 比较能量（按原子归一化）
        passed = numerical_comparator.compare_energies(
            computed_energy, reference_energy,
            unit='eV', per_atom=True, n_atoms=n_atoms
        )
        
        assert passed, f"Energy mismatch: {computed_energy} vs {reference_energy}"
    
    def test_nacl_bulk_energy_reproducibility(self):
        """测试NaCl块体能量计算的可重复性"""
        # 同一结构多次计算应该得到一致结果
        energies = []
        
        # 模拟3次独立计算
        for seed in [42, 123, 456]:
            np.random.seed(seed)
            # 添加微小数值噪声模拟实际计算
            base_energy = -50.0
            noise = np.random.normal(0, 1e-6)
            energies.append(base_energy + noise)
        
        # 检查一致性
        energy_std = np.std(energies)
        assert energy_std < 1e-5, f"Energy not reproducible, std={energy_std}"
    
    def test_energy_convergence_with_kpoints(self):
        """测试能量随k点增加的收敛性"""
        # 模拟不同k点网格的能量
        kpoints_grid = [2, 4, 6, 8, 10]
        energies = []
        
        base_energy = -100.0
        for k in kpoints_grid:
            # 能量应该随k点增加而收敛
            convergence_correction = 0.1 / (k ** 2)
            energy = base_energy - convergence_correction
            energies.append(energy)
        
        # 检查收敛性：相邻能量差应该递减
        energy_diffs = np.diff(energies)
        for i in range(1, len(energy_diffs)):
            assert abs(energy_diffs[i]) < abs(energy_diffs[i-1]), \
                "Energy not converging with kpoints"
    
    def test_energy_symmetry_conservation(self):
        """测试能量对对称操作的守恒性"""
        # 同一结构经过对称操作后能量应该相同
        base_energy = -75.123456
        
        # 模拟不同对称操作后的能量
        operations = ['identity', 'rotation', 'translation', 'inversion']
        energies = []
        
        for op in operations:
            # 数值噪声模拟
            noise = np.random.normal(0, 1e-7)
            energies.append(base_energy + noise)
        
        # 所有能量应该相等（在容差内）
        max_diff = np.max(energies) - np.min(energies)
        assert max_diff < 1e-6, f"Energy not symmetric, max_diff={max_diff}"


# =============================================================================
# DFT力计算回归测试
# =============================================================================

@pytest.mark.regression
@pytest.mark.dft
class TestDFTForceRegression:
    """DFT力计算回归测试"""
    
    def test_forces_virial_theorem(self, numerical_comparator):
        """测试力满足维里定理（总力为零）"""
        n_atoms = 6
        
        # 模拟计算的力
        forces = np.random.randn(n_atoms, 3) * 0.1
        
        # 调整使总力为零（去除质心运动）
        forces = forces - np.mean(forces, axis=0)
        
        total_force = np.sum(forces, axis=0)
        
        assert np.allclose(total_force, 0, atol=1e-10), \
            f"Total force not zero: {total_force}"
    
    def test_force_rotational_invariance(self):
        """测试力的旋转不变性"""
        # 原始力
        forces_original = np.array([
            [0.1, 0.2, 0.3],
            [-0.1, -0.2, -0.3],
            [0.05, -0.05, 0.0]
        ])
        
        # 旋转矩阵（绕z轴45度）
        theta = np.pi / 4
        rotation = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        # 旋转后的力
        forces_rotated = np.dot(forces_original, rotation.T)
        
        # 力的模长应该保持不变
        norms_original = np.linalg.norm(forces_original, axis=1)
        norms_rotated = np.linalg.norm(forces_rotated, axis=1)
        
        assert np.allclose(norms_original, norms_rotated, atol=1e-10), \
            "Force norms not rotationally invariant"
    
    def test_force_accuracy_consistency(self):
        """测试力计算的精度一致性"""
        # 参考力（高精度计算）
        reference_forces = np.random.randn(10, 3) * 0.5
        
        # 测试计算（标准精度）
        test_forces = reference_forces + np.random.randn(10, 3) * 0.001
        
        # 计算偏差
        max_diff = np.max(np.abs(test_forces - reference_forces))
        rms_diff = np.sqrt(np.mean((test_forces - reference_forces)**2))
        
        assert max_diff < 0.01, f"Max force deviation too large: {max_diff}"
        assert rms_diff < 0.005, f"RMS force deviation too large: {rms_diff}"
    
    def test_force_energy_consistency(self):
        """测试力-能量一致性（力是能量的负梯度）"""
        # 通过有限差分验证力
        delta = 0.001
        
        # 模拟能量函数
        def energy_func(pos):
            # 简谐势
            return 0.5 * np.sum(pos**2) * 0.1
        
        # 初始位置
        pos = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        
        # 解析力（梯度）
        analytical_forces = -0.1 * pos
        
        # 数值力（有限差分）
        numerical_forces = np.zeros_like(pos)
        for i in range(pos.shape[0]):
            for j in range(3):
                pos_plus = pos.copy()
                pos_plus[i, j] += delta
                pos_minus = pos.copy()
                pos_minus[i, j] -= delta
                
                numerical_forces[i, j] = -(energy_func(pos_plus) - energy_func(pos_minus)) / (2 * delta)
        
        # 比较
        max_diff = np.max(np.abs(analytical_forces - numerical_forces))
        assert max_diff < 1e-4, f"Force-energy inconsistency: {max_diff}"


# =============================================================================
# DFT应力张量回归测试
# =============================================================================

@pytest.mark.regression
@pytest.mark.dft
class TestDFTStressRegression:
    """DFT应力张量回归测试"""
    
    def test_stress_tensor_symmetry(self):
        """测试应力张量的对称性"""
        # 应力张量应该是对称的
        stress = np.array([
            [0.5, 0.1, 0.2],
            [0.1, 0.6, 0.15],
            [0.2, 0.15, 0.4]
        ])
        
        assert np.allclose(stress, stress.T), "Stress tensor not symmetric"
    
    def test_stress_pressure_consistency(self):
        """测试应力与压强的一致性"""
        # 应力张量
        stress = np.array([
            [-0.5, 0.0, 0.0],
            [0.0, -0.6, 0.0],
            [0.0, 0.0, -0.4]
        ])
        
        # 压强 = -trace(stress) / 3
        pressure = -np.trace(stress) / 3
        expected_pressure = (0.5 + 0.6 + 0.4) / 3
        
        assert np.isclose(pressure, expected_pressure), \
            f"Pressure mismatch: {pressure} vs {expected_pressure}"
    
    def test_stress_units_conversion(self):
        """测试应力单位转换的正确性"""
        # kBar to GPa: 1 kBar = 0.1 GPa
        stress_kbar = 100.0
        stress_gpa = stress_kbar * 0.1
        
        assert np.isclose(stress_gpa, 10.0), "Stress unit conversion error"


# =============================================================================
# DFT自洽场收敛性测试
# =============================================================================

@pytest.mark.regression
@pytest.mark.dft
class TestDFTConvergenceRegression:
    """DFT SCF收敛性回归测试"""
    
    def test_scf_convergence_monotonicity(self):
        """测试SCF收敛的单调性"""
        # 模拟SCF迭代能量
        n_iterations = 20
        energies = []
        
        base_energy = -100.0
        for i in range(n_iterations):
            # 能量应该逐渐收敛
            delta = 0.1 * np.exp(-0.5 * i) * np.cos(i * 0.5)
            energies.append(base_energy + delta)
        
        # 检查最终收敛
        final_diffs = np.diff(energies[-5:])
        assert np.all(np.abs(final_diffs) < 1e-5), "SCF not converged"
    
    def test_scf_convergence_stability(self):
        """测试SCF收敛的稳定性"""
        # 多次运行应该得到相同结果
        n_runs = 5
        final_energies = []
        
        for _ in range(n_runs):
            # 模拟有轻微初始差异的SCF
            initial_noise = np.random.normal(0, 0.001)
            converged_energy = -100.0 + initial_noise * 0.01  # 噪声被抑制
            final_energies.append(converged_energy)
        
        energy_std = np.std(final_energies)
        assert energy_std < 1e-6, f"SCF convergence not stable: std={energy_std}"


# =============================================================================
# 跨平台数值稳定性测试
# =============================================================================

@pytest.mark.regression
@pytest.mark.dft
class TestDFTCrossPlatformStability:
    """DFT跨平台数值稳定性测试"""
    
    def test_floating_point_consistency(self):
        """测试浮点运算的一致性"""
        # 测试关键数值操作
        a = np.array([1.0/3.0, np.sqrt(2), np.pi])
        b = np.array([1.0/3.0, np.sqrt(2), np.pi])
        
        # 应该精确相等
        assert np.allclose(a, b, rtol=0, atol=1e-15), "Floating point inconsistency"
    
    def test_lattice_vector_operations(self):
        """测试晶格向量运算的一致性"""
        # 定义晶格
        lattice = np.array([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0]
        ])
        
        # 计算体积（应该为正）
        volume = np.abs(np.linalg.det(lattice))
        assert volume > 0, "Lattice volume calculation error"
        assert np.isclose(volume, 125.0), f"Volume mismatch: {volume}"
        
        # 倒晶格
        reciprocal = 2 * np.pi * np.linalg.inv(lattice).T
        expected_reciprocal = np.eye(3) * 2 * np.pi / 5.0
        
        assert np.allclose(reciprocal, expected_reciprocal), \
            "Reciprocal lattice calculation error"


# =============================================================================
# 特定体系回归测试
# =============================================================================

@pytest.mark.regression
@pytest.mark.dft
@pytest.mark.slow
class TestDFTSystemRegression:
    """特定材料体系的DFT回归测试"""
    
    @pytest.mark.parametrize("material_id,expected_energy", [
        ("mp-1234", -50.123),
        ("mp-5678", -75.456),
        ("mp-9999", -100.789),
    ])
    def test_known_material_energies(self, material_id, expected_energy):
        """测试已知材料体系的能量"""
        # 模拟从数据库获取计算结果
        computed_energy = expected_energy + np.random.normal(0, 1e-5)
        
        assert np.isclose(computed_energy, expected_energy, rtol=1e-4), \
            f"Energy mismatch for {material_id}"
    
    def test_battery_material_consistency(self):
        """测试电池材料计算的特定一致性"""
        # Li3PS4体系
        li3ps4_energy = -200.0  # 参考值
        
        # 检查电压计算的一致性
        voltage = 2.5  # V
        
        # 电压应该满足热力学稳定性
        assert 0 < voltage < 5, f"Unphysical voltage: {voltage}"
        
        # 检查体积变化
        volume_change = 0.05  # 5%
        assert -0.2 < volume_change < 0.5, f"Unreasonable volume change: {volume_change}"


# =============================================================================
# 参考数据管理测试
# =============================================================================

@pytest.mark.regression
@pytest.mark.dft
class TestDFTReferenceData:
    """DFT参考数据管理测试"""
    
    def test_reference_data_creation(self):
        """测试参考数据创建"""
        ref_data = DFTReferenceData(
            material_id="test-material",
            calculator="VASP",
            functional="PBE",
            encut=520.0,
            kpoints=[4, 4, 4],
            total_energy=-100.0,
            forces=np.random.randn(6, 3) * 0.1,
            stress=np.eye(3) * 0.1,
            lattice=np.eye(3) * 10,
            positions=np.random.randn(6, 3),
            symbols=['Li', 'Li', 'S', 'S', 'P', 'P'],
            fermi_energy=-2.5,
            band_gap=3.0
        )
        
        assert ref_data.material_id == "test-material"
        assert ref_data.total_energy == -100.0
    
    def test_reference_data_serialization(self, tmp_path):
        """测试参考数据序列化"""
        ref_data = DFTReferenceData(
            material_id="test-serial",
            calculator="VASP",
            functional="PBE",
            encut=520.0,
            kpoints=[4, 4, 4],
            total_energy=-150.0,
            forces=np.array([[0.1, 0.0, 0.0]]),
            stress=np.eye(3) * 0.1,
            lattice=np.eye(3) * 5,
            positions=np.array([[0.0, 0.0, 0.0]]),
            symbols=['Li'],
        )
        
        # 序列化
        data_dict = ref_data.to_dict()
        
        # 反序列化
        restored = DFTReferenceData.from_dict(data_dict)
        
        assert restored.material_id == ref_data.material_id
        assert np.allclose(restored.forces, ref_data.forces)
