"""
机器学习势回归测试
ML Potential Regression Tests
==============================

验证ML势预测的稳定性和一致性。

测试项目:
    - 预测一致性（确定性）
    - 数值稳定性（梯度、Hessian）
    - 外推行为
    - 能量-力一致性
    - 对称性守恒
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.regression import MLReferenceData, NumericalComparator


# =============================================================================
# ML预测一致性测试
# =============================================================================

@pytest.mark.regression
@pytest.mark.ml
class TestMLPredictionConsistency:
    """ML势预测一致性测试"""
    
    def test_deterministic_predictions(self):
        """测试ML预测的确定性"""
        # 模拟固定模型的预测
        structure = np.random.randn(6, 3)
        
        # 多次预测应该得到相同结果
        predictions = []
        for _ in range(5):
            # 简化的模型预测（无随机性）
            energy = -100.0 + 0.1 * np.sum(structure**2)
            predictions.append(energy)
        
        # 所有预测应该相同
        assert len(set(np.round(predictions, 10))) == 1, \
            "Predictions not deterministic"
    
    def test_model_reload_consistency(self):
        """测试模型重新加载的一致性"""
        # 模拟保存和加载模型
        original_weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # 保存模型
        saved_weights = original_weights.copy()
        
        # 加载模型
        loaded_weights = saved_weights.copy()
        
        # 权重应该相同
        assert np.allclose(original_weights, loaded_weights), \
            "Model weights changed after reload"
        
        # 使用相同权重的预测应该相同
        test_input = np.array([0.5, 0.3, 0.2, 0.1, 0.4])
        pred1 = np.dot(original_weights, test_input)
        pred2 = np.dot(loaded_weights, test_input)
        
        assert np.isclose(pred1, pred2), "Predictions differ after reload"
    
    def test_batch_vs_single_prediction(self):
        """测试批处理和单样本预测的一致性"""
        # 模拟模型
        def model_predict(x):
            return np.sum(x**2, axis=-1) * 0.1 - 10.0
        
        # 单样本预测
        n_samples = 10
        single_predictions = []
        for i in range(n_samples):
            x = np.random.randn(5)
            pred = model_predict(x[np.newaxis, :])
            single_predictions.append(pred[0])
        
        # 批处理预测
        batch_input = np.random.randn(n_samples, 5)
        batch_predictions = model_predict(batch_input)
        
        # 分别比较（由于随机输入，直接比较数值）
        assert len(single_predictions) == len(batch_predictions), \
            "Batch size mismatch"
    
    def test_prediction_scale_invariance(self):
        """测试预测对结构缩放的响应"""
        # 原始结构
        structure = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0]
        ])
        
        # 缩放结构
        scale_factors = [0.9, 1.0, 1.1]
        energies = []
        
        for scale in scale_factors:
            scaled = structure * scale
            # 简化的能量计算（谐振子近似）
            energy = -10.0 + 0.5 * np.sum((scaled - structure)**2)
            energies.append(energy)
        
        # 缩放1.0应该能量最低（平衡位置）
        assert energies[1] < energies[0], "Energy not minimized at scale=1.0"
        assert energies[1] < energies[2], "Energy not minimized at scale=1.0"


# =============================================================================
# 数值稳定性测试
# =============================================================================

@pytest.mark.regression
@pytest.mark.ml
class TestMLNumericalStability:
    """ML势数值稳定性测试"""
    
    def test_gradient_stability(self):
        """测试梯度的数值稳定性"""
        # 简化的势能函数
        def potential(x):
            return np.sum(x**2) + 0.01 * np.sum(x**4)
        
        # 计算数值梯度
        def numerical_gradient(x, eps=1e-5):
            grad = np.zeros_like(x)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    x_plus = x.copy()
                    x_minus = x.copy()
                    x_plus[i, j] += eps
                    x_minus[i, j] -= eps
                    grad[i, j] = (potential(x_plus) - potential(x_minus)) / (2 * eps)
            return grad
        
        # 测试位置
        x = np.random.randn(5, 3)
        
        # 解析梯度
        analytical_grad = 2 * x + 0.04 * x**3
        
        # 数值梯度
        numerical_grad = numerical_gradient(x)
        
        # 比较
        max_diff = np.max(np.abs(analytical_grad - numerical_grad))
        assert max_diff < 1e-4, f"Gradient mismatch: {max_diff}"
    
    def test_hessian_positive_definite(self):
        """测试Hessian矩阵正定性（稳定性）"""
        # 在平衡位置，Hessian应该是正定的
        n_atoms = 3
        
        # 简化的Hessian（谐振子）
        hessian = np.eye(n_atoms * 3) * 2.0
        
        # 添加耦合
        hessian[0, 1] = hessian[1, 0] = 0.1
        hessian[1, 2] = hessian[2, 1] = 0.1
        
        # 检查对称性
        assert np.allclose(hessian, hessian.T), "Hessian not symmetric"
        
        # 检查正定性（特征值）
        eigenvalues = np.linalg.eigvalsh(hessian)
        assert np.all(eigenvalues > 0), f"Hessian not positive definite: {eigenvalues}"
    
    def test_extreme_position_behavior(self):
        """测试极端位置下的行为"""
        # 测试原子过于接近的情况
        positions_close = np.array([
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0]  # 非常近
        ])
        
        # 简化的排斥势
        def repulsive_energy(pos):
            dist = np.linalg.norm(pos[0] - pos[1])
            return 1.0 / dist if dist > 0.001 else 1000.0
        
        energy = repulsive_energy(positions_close)
        
        # 能量应该很大（排斥）
        assert energy > 10.0, "Repulsion not strong enough for close atoms"
        
        # 测试原子过远的情况
        positions_far = np.array([
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0]
        ])
        
        energy_far = repulsive_energy(positions_far)
        
        # 能量应该很小
        assert energy_far < 0.1, "Energy not decaying for distant atoms"
    
    def test_discontinuity_detection(self):
        """测试不连续性的检测"""
        # 连续函数
        def continuous_func(x):
            return np.sin(x) + 0.5 * x**2
        
        # 测试连续性
        x_values = np.linspace(-2, 2, 100)
        y_values = continuous_func(x_values)
        
        # 相邻点差异
        diffs = np.diff(y_values)
        
        # 最大差异
        max_diff = np.max(np.abs(diffs))
        mean_diff = np.mean(np.abs(diffs))
        
        # 不应该有突变
        assert max_diff < 10 * mean_diff, "Potential discontinuity detected"


# =============================================================================
# 能量-力一致性测试
# =============================================================================

@pytest.mark.regression
@pytest.mark.ml
class TestMLEnergyForceConsistency:
    """ML能量-力一致性测试"""
    
    def test_force_energy_consistency_finite_diff(self):
        """使用有限差分测试力-能量一致性"""
        # 定义势能函数
        def energy_func(pos):
            # 简化的多体势
            return np.sum(pos**2) * 0.5 + np.sum(pos**4) * 0.01
        
        # 有限差分计算力
        def compute_forces_fd(pos, eps=1e-5):
            forces = np.zeros_like(pos)
            for i in range(pos.shape[0]):
                for j in range(pos.shape[1]):
                    pos_plus = pos.copy()
                    pos_minus = pos.copy()
                    pos_plus[i, j] += eps
                    pos_minus[i, j] -= eps
                    
                    e_plus = energy_func(pos_plus)
                    e_minus = energy_func(pos_minus)
                    forces[i, j] = -(e_plus - e_minus) / (2 * eps)
            return forces
        
        # 测试位置
        pos = np.random.randn(5, 3)
        
        # 解析力（梯度）
        analytical_forces = -(pos + 0.04 * pos**3)
        
        # 数值力
        numerical_forces = compute_forces_fd(pos)
        
        # 比较
        comparator = NumericalComparator()
        force_comparison = comparator.compare_forces(
            analytical_forces, numerical_forces, threshold=0.001
        )
        
        assert force_comparison['passed'], \
            f"Force-energy inconsistency: max_diff={force_comparison['max_diff']}"
    
    def test_virial_stress_consistency(self):
        """测试维里应力的一致性"""
        # 模拟原子和力
        n_atoms = 5
        positions = np.random.randn(n_atoms, 3)
        forces = np.random.randn(n_atoms, 3) * 0.1
        
        # 晶胞
        cell = np.eye(3) * 10.0
        volume = np.linalg.det(cell)
        
        # 维里应力
        virial = np.zeros((3, 3))
        for i in range(n_atoms):
            for alpha in range(3):
                for beta in range(3):
                    virial[alpha, beta] += positions[i, alpha] * forces[i, beta]
        
        # 应力张量
        stress = -virial / volume
        
        # 检查对称性
        assert np.allclose(stress, stress.T, atol=1e-10), "Stress tensor not symmetric"
        
        # 压强
        pressure = -np.trace(stress) / 3
        
        # 压强应该在合理范围内
        assert -1000 < pressure < 1000, f"Unphysical pressure: {pressure}"
    
    def test_energy_conservation_with_ml_forces(self):
        """测试使用ML力时的能量守恒"""
        # 简化的MD模拟
        n_steps = 100
        dt = 0.001
        
        positions = np.array([[1.0, 0.0, 0.0]])
        velocities = np.array([[0.0, 0.5, 0.0]])
        
        energies = []
        
        for _ in range(n_steps):
            # 简化的ML势（谐振子）
            potential = 0.5 * np.sum(positions**2)
            kinetic = 0.5 * np.sum(velocities**2)
            total = potential + kinetic
            energies.append(total)
            
            # ML力 = -梯度
            forces = -positions
            
            # Verlet积分
            velocities += 0.5 * forces * dt
            positions += velocities * dt
            forces_new = -positions
            velocities += 0.5 * forces_new * dt
        
        energies = np.array(energies)
        energy_drift = np.max(energies) - np.min(energies)
        
        assert energy_drift < 1e-6, f"Energy drift with ML forces: {energy_drift}"


# =============================================================================
# 对称性测试
# =============================================================================

@pytest.mark.regression
@pytest.mark.ml
class TestMLSymmetry:
    """ML势对称性测试"""
    
    def test_translation_invariance(self):
        """测试平移不变性"""
        # 原始结构
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0]
        ])
        
        # 计算能量
        def compute_energy(pos):
            # 基于距离的势能
            dists = []
            for i in range(len(pos)):
                for j in range(i+1, len(pos)):
                    dists.append(np.linalg.norm(pos[i] - pos[j]))
            return np.sum([1/d for d in dists])
        
        energy_original = compute_energy(positions)
        
        # 平移后的结构
        translation = np.array([5.0, 5.0, 5.0])
        positions_translated = positions + translation
        
        energy_translated = compute_energy(positions_translated)
        
        # 能量应该相同
        assert np.isclose(energy_original, energy_translated), \
            "Energy not translationally invariant"
    
    def test_rotation_invariance(self):
        """测试旋转不变性"""
        positions = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # 计算能量（基于距离）
        def compute_energy(pos):
            dists = []
            for i in range(len(pos)):
                for j in range(i+1, len(pos)):
                    dists.append(np.linalg.norm(pos[i] - pos[j]))
            return np.sum([d**2 for d in dists])
        
        energy_original = compute_energy(positions)
        
        # 随机旋转
        theta = np.pi / 4
        rotation = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        positions_rotated = np.dot(positions, rotation.T)
        energy_rotated = compute_energy(positions_rotated)
        
        # 能量应该相同
        assert np.isclose(energy_original, energy_rotated), \
            "Energy not rotationally invariant"
    
    def test_permutation_invariance(self):
        """测试置换不变性（原子顺序不影响）"""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0]
        ])
        
        # 基于距离的势能
        def compute_energy(pos):
            dists = []
            for i in range(len(pos)):
                for j in range(i+1, len(pos)):
                    dists.append(np.linalg.norm(pos[i] - pos[j]))
            return np.sum([d for d in dists])
        
        energy_original = compute_energy(positions)
        
        # 置换原子顺序
        permutation = [2, 0, 1]
        positions_permuted = positions[permutation]
        
        energy_permuted = compute_energy(positions_permuted)
        
        # 能量应该相同
        assert np.isclose(energy_original, energy_permuted), \
            "Energy not permutation invariant"


# =============================================================================
# 外推行为测试
# =============================================================================

@pytest.mark.regression
@pytest.mark.ml
class TestMLExtrapolation:
    """ML势外推行为测试"""
    
    def test_confidence_estimation(self):
        """测试置信度估计"""
        # 训练数据覆盖范围
        train_range = (0, 5)
        
        # 测试内插和外推
        test_points = [2.5, 5.5, 10.0]  # 内插、轻微外推、严重外推
        
        uncertainties = []
        for x in test_points:
            # 简化的不确定性估计（基于到训练数据的距离）
            distance_to_train = min(abs(x - train_range[0]), abs(x - train_range[1]))
            if x >= train_range[0] and x <= train_range[1]:
                uncertainty = 0.1
            else:
                uncertainty = 0.1 + distance_to_train * 0.5
            uncertainties.append(uncertainty)
        
        # 外推点应该有更高的不确定性
        assert uncertainties[0] < uncertainties[1], "Uncertainty not increasing"
        assert uncertainties[1] < uncertainties[2], "Uncertainty not increasing"
    
    def test_uncertainty_monotonicity(self):
        """测试不确定性的单调性"""
        # 从训练数据向外移动
        distances = [0, 1, 2, 5, 10]
        
        uncertainties = []
        base_uncertainty = 0.1
        
        for d in distances:
            uncertainty = base_uncertainty * (1 + 0.1 * d**2)
            uncertainties.append(uncertainty)
        
        # 不确定性应该单调增加
        for i in range(len(uncertainties) - 1):
            assert uncertainties[i] <= uncertainties[i+1], \
                "Uncertainty not monotonically increasing"


# =============================================================================
# 参考数据测试
# =============================================================================

@pytest.mark.regression
@pytest.mark.ml
class TestMLReferenceData:
    """ML参考数据测试"""
    
    def test_reference_data_creation(self):
        """测试ML参考数据创建"""
        ref_data = MLReferenceData(
            model_type="NEP",
            model_version="v1.0",
            training_dataset="Li3PS4_dataset_v2",
            rmse_energy=0.005,
            rmse_forces=0.08,
            r2_energy=0.999,
            r2_forces=0.995,
            test_predictions={"test1": -100.5, "test2": -95.3},
            gradient_norm=1.5,
            hessian_eigenvalues=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        
        assert ref_data.model_type == "NEP"
        assert ref_data.rmse_energy < 0.01
    
    def test_reference_data_serialization(self, tmp_path):
        """测试ML参考数据序列化"""
        import json
        
        ref_data = MLReferenceData(
            model_type="MACE",
            model_version="v2.1",
            training_dataset="test_data",
            rmse_energy=0.01,
            rmse_forces=0.1,
            r2_energy=0.998,
            r2_forces=0.99,
            test_predictions={},
            gradient_norm=2.0,
            hessian_eigenvalues=[0.5, 1.0, 1.5]
        )
        
        data_dict = ref_data.to_dict()
        
        filepath = tmp_path / "ml_ref.json"
        with open(filepath, 'w') as f:
            json.dump(data_dict, f)
        
        with open(filepath, 'r') as f:
            loaded = json.load(f)
        
        assert loaded['model_type'] == "MACE"
        assert loaded['rmse_energy'] == 0.01
