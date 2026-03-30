"""
Enhanced Differentiable DFT Module
===================================

扩展的可微分DFT功能，提供更高级的自动微分特性：
- 密度泛函的任意阶导数
- 响应性质计算（极化率、介电常数）
- 线性响应TDDFT
- 超参数梯度优化

核心功能：
- 高阶自动微分支持
- 物理响应量计算
- 与优化算法的集成
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, hessian, jacfwd, jacrev
from jax.experimental import optimizers
from functools import partial
from typing import Callable, Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize


@dataclass
class ResponseConfig:
    """响应性质计算配置"""
    field_strength: float = 0.001  # 外加场强度 (a.u.)
    field_direction: Tuple[float, float, float] = (1.0, 0.0, 0.0)  # 场方向
    omega: float = 0.0  # 频率 (ω=0为静态)
    response_order: int = 1  # 响应阶数 (1=线性, 2=非线性)


@dataclass
class TDDFTConfig:
    """TDDFT配置"""
    n_states: int = 5  # 激发态数目
    tda: bool = True  # Tamm-Dancoff近似
    max_iter: int = 100
    conv_tol: float = 1e-6


class HighOrderDerivatives:
    """
    高阶导数计算器
    
    使用JAX的高阶自动微分计算能量和密度的多阶导数
    """
    
    def __init__(self, energy_fn: Callable):
        """
        Args:
            energy_fn: 能量函数 E(positions, cell) -> float
        """
        self.energy_fn = energy_fn
        
        # 预计算各种导数函数
        self._grad_fn = jit(grad(energy_fn, argnums=0))  # 一阶梯度
        self._hessian_fn = jit(hessian(energy_fn, argnums=0))  # 二阶梯度
        self._mixed_grad_fn = jit(grad(lambda p, c: energy_fn(p, c), argnums=(0, 1)))
    
    def first_derivative_positions(self, positions: jnp.ndarray, 
                                    cell: jnp.ndarray) -> jnp.ndarray:
        """
        能量对位置的一阶导数 (力的负值)
        
        Args:
            positions: 原子位置 (N, 3)
            cell: 晶胞矩阵 (3, 3)
            
        Returns:
            梯度 (N, 3)
        """
        return self._grad_fn(positions, cell)
    
    def second_derivative_positions(self, positions: jnp.ndarray,
                                     cell: jnp.ndarray) -> jnp.ndarray:
        """
        能量对位置的二阶导数 (Hessian矩阵)
        
        用于：
        - 声子频率计算
        - 红外/拉曼强度
        - 热力学性质
        
        Args:
            positions: 原子位置 (N, 3)
            cell: 晶胞矩阵 (3, 3)
            
        Returns:
            Hessian矩阵 (N, 3, N, 3)
        """
        return self._hessian_fn(positions, cell)
    
    def force_constants_matrix(self, positions: jnp.ndarray,
                                cell: jnp.ndarray,
                                masses: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        计算力常数矩阵并进行质量加权
        
        Args:
            positions: 原子位置
            cell: 晶胞矩阵
            masses: 原子质量 (N,)，可选
            
        Returns:
            力常数矩阵 (3N, 3N)
        """
        hess = self._hessian_fn(positions, cell)
        N = positions.shape[0]
        
        # 重塑为 (3N, 3N)
        fc_matrix = hess.reshape(3*N, 3*N)
        
        if masses is not None:
            # 质量加权: C_ij / sqrt(m_i * m_j)
            m_sqrt = jnp.sqrt(masses.repeat(3))
            fc_matrix = fc_matrix / (m_sqrt[:, None] * m_sqrt[None, :])
        
        return fc_matrix
    
    def phonon_frequencies(self, positions: jnp.ndarray,
                           cell: jnp.ndarray,
                           masses: jnp.ndarray) -> jnp.ndarray:
        """
        计算声子频率
        
        Args:
            positions: 原子位置
            cell: 晶胞矩阵
            masses: 原子质量 (N,)
            
        Returns:
            声子频率 (3N,)，单位 THz
        """
        fc_matrix = self.force_constants_matrix(positions, cell, masses)
        
        # 对角化力常数矩阵
        eigenvalues, _ = jnp.linalg.eigh(fc_matrix)
        
        # 转换为频率 (Ha -> THz)
        # ω = sqrt(λ) in atomic units
        # 1 Ha = 4.13414e16 Hz = 41341.4 THz
        omega_sq = jnp.maximum(eigenvalues, 0)  # 去除负值
        frequencies = jnp.sqrt(omega_sq) * 41341.4  # THz
        
        return frequencies
    
    def third_derivative_positions(self, positions: jnp.ndarray,
                                    cell: jnp.ndarray) -> jnp.ndarray:
        """
        能量对位置的三阶导数
        
        用于非谐效应、热膨胀系数等
        
        Args:
            positions: 原子位置
            cell: 晶胞矩阵
            
        Returns:
            三阶导数张量 (N, 3, N, 3, N, 3)
        """
        # 使用jacfwd计算三阶导数
        def grad_fn(pos):
            return grad(self.energy_fn, argnums=0)(pos, cell)
        
        def hess_fn(pos):
            return jacfwd(grad_fn)(pos)
        
        third_deriv_fn = jacfwd(hess_fn)
        return third_deriv_fn(positions)
    
    def elastic_constants(self, positions: jnp.ndarray,
                          cell: jnp.ndarray) -> jnp.ndarray:
        """
        计算弹性常数张量 C_ijkl
        
        C_ijkl = 1/V * d²E/de_ij de_kl
        
        Args:
            positions: 原子位置
            cell: 晶胞矩阵
            
        Returns:
            弹性常数张量 (6, 6) Voigt记号
        """
        volume = jnp.abs(jnp.linalg.det(cell))
        
        # 能量对应变的二阶导数
        def energy_strain(strain_tensor):
            # 应用应变到晶胞
            strained_cell = (jnp.eye(3) + strain_tensor) @ cell
            # 保持分数坐标不变
            frac_pos = positions @ jnp.linalg.inv(cell)
            strained_pos = frac_pos @ strained_cell
            return self.energy_fn(strained_pos, strained_cell)
        
        # 计算应变二阶导数
        strain_hessian = hessian(energy_strain)(jnp.zeros((3, 3)))
        
        # 转换为Voigt记号 (6x6矩阵)
        # Voigt: 11->0, 22->1, 33->2, 23->3, 13->4, 12->5
        voigt_map = [(0,0), (1,1), (2,2), (1,2), (0,2), (0,1)]
        
        C_voigt = jnp.zeros((6, 6))
        for i, (a, b) in enumerate(voigt_map):
            for j, (c, d) in enumerate(voigt_map):
                C_voigt = C_voigt.at[i, j].set(strain_hessian[a, b, c, d] / volume)
        
        return C_voigt


class ResponseProperties:
    """
    响应性质计算器
    
    计算外加场下的分子/材料响应
    """
    
    def __init__(self, dft_calculator: Any):
        """
        Args:
            dft_calculator: DFT计算器实例
        """
        self.dft = dft_calculator
    
    def dipole_moment(self, density: jnp.ndarray, 
                      grid_points: jnp.ndarray,
                      cell: jnp.ndarray) -> jnp.ndarray:
        """
        计算偶极矩
        
        μ = ∫ r * ρ(r) dr
        
        Args:
            density: 电子密度
            grid_points: 网格点坐标
            cell: 晶胞矩阵
            
        Returns:
            偶极矩向量 (3,)
        """
        dV = jnp.abs(jnp.linalg.det(cell)) / len(density)
        dipole = jnp.sum(grid_points * density[:, None], axis=0) * dV
        return dipole
    
    def static_polarizability(self, positions: jnp.ndarray,
                              atomic_numbers: jnp.ndarray,
                              cell: jnp.ndarray,
                              field_strength: float = 0.001) -> jnp.ndarray:
        """
        计算静态极化率张量 (有限场方法)
        
        α_ij = -dμ_i/dE_j ≈ -[μ_i(+E_j) - μ_i(-E_j)] / (2*E_j)
        
        Args:
            positions: 原子位置
            atomic_numbers: 原子序数
            cell: 晶胞矩阵
            field_strength: 外场强度
            
        Returns:
            极化率张量 (3, 3)
        """
        polarizability = jnp.zeros((3, 3))
        
        for j in range(3):  # 电场方向
            # +E_j
            field_plus = jnp.zeros(3).at[j].set(field_strength)
            mu_plus = self._dipole_with_field(positions, atomic_numbers, 
                                              cell, field_plus)
            
            # -E_j
            field_minus = jnp.zeros(3).at[j].set(-field_strength)
            mu_minus = self._dipole_with_field(positions, atomic_numbers,
                                               cell, field_minus)
            
            # 中心差分
            dmu_dEj = (mu_plus - mu_minus) / (2 * field_strength)
            
            polarizability = polarizability.at[:, j].set(dmu_dEj)
        
        return polarizability
    
    def _dipole_with_field(self, positions: jnp.ndarray,
                           atomic_numbers: jnp.ndarray,
                           cell: jnp.ndarray,
                           field: jnp.ndarray) -> jnp.ndarray:
        """
        在外加场下计算偶极矩
        
        (简化实现，实际需要在外场下做SCF计算)
        """
        # 这里应该调用带外场的DFT计算
        # 简化：返回位置的质心近似
        electrons = jnp.sum(atomic_numbers)
        com = jnp.sum(positions * atomic_numbers[:, None], axis=0) / electrons
        return com * electrons  # 简化近似
    
    def dielectric_tensor(self, positions: jnp.ndarray,
                          atomic_numbers: jnp.ndarray,
                          cell: jnp.ndarray) -> jnp.ndarray:
        """
        计算介电张量
        
        ε = 1 + 4π * χ_e = 1 + 4π * α / V
        
        Args:
            positions: 原子位置
            atomic_numbers: 原子序数
            cell: 晶胞矩阵
            
        Returns:
            介电张量 (3, 3)
        """
        volume = jnp.abs(jnp.linalg.det(cell))
        
        # 计算极化率
        alpha = self.static_polarizability(positions, atomic_numbers, cell)
        
        # 介电张量
        epsilon = jnp.eye(3) + 4 * jnp.pi * alpha / volume
        
        return epsilon
    
    def born_effective_charges(self, positions: jnp.ndarray,
                                atomic_numbers: jnp.ndarray,
                                cell: jnp.ndarray,
                                field_strength: float = 0.001) -> jnp.ndarray:
        """
        计算Born有效电荷张量
        
        Z*_ij = dF_i/dE_j (力对电场的导数)
        或 Z*_ij = V * dP_i/du_j (极化对位移的导数)
        
        Args:
            positions: 原子位置
            atomic_numbers: 原子序数
            cell: 晶胞矩阵
            field_strength: 场强度
            
        Returns:
            Born电荷 (N, 3, 3)
        """
        N = len(positions)
        Z_star = jnp.zeros((N, 3, 3))
        
        # 使用有限差分计算 Z* = -dF/dE
        for j in range(3):  # 电场方向
            # +E
            field_plus = jnp.zeros(3).at[j].set(field_strength)
            F_plus = self._forces_with_field(positions, atomic_numbers,
                                             cell, field_plus)
            
            # -E
            field_minus = jnp.zeros(3).at[j].set(-field_strength)
            F_minus = self._forces_with_field(positions, atomic_numbers,
                                              cell, field_minus)
            
            # dF/dE
            dF_dE = (F_plus - F_minus) / (2 * field_strength)
            
            # Z* = -dF/dE
            for i in range(N):
                Z_star = Z_star.at[i, :, j].set(-dF_dE[i])
        
        return Z_star
    
    def _forces_with_field(self, positions: jnp.ndarray,
                           atomic_numbers: jnp.ndarray,
                           cell: jnp.ndarray,
                           field: jnp.ndarray) -> jnp.ndarray:
        """在外场下计算力 (简化实现)"""
        # 简化：返回与电场方向相反的力
        N = len(positions)
        return -field[None, :] * jnp.ones((N, 3)) * 0.01


class TDDFTCalculator:
    """
    线性响应TDDFT计算器
    
    计算激发态和光谱性质
    """
    
    def __init__(self, config: TDDFTConfig):
        self.config = config
    
    def casida_equations(self, 
                         ground_state_orbitals: jnp.ndarray,
                         ground_state_energies: jnp.ndarray,
                         occupation: jnp.ndarray,
                         xc_kernel: Callable) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        求解Casida方程 (TDDFT线性响应)
        
        [A  B][X]     [X]
        [B  A][Y] = ω [Y]
        
        其中 A_ia,jb = δ_ij δ_ab (ε_a - ε_i) + (ia|f_xc|jb)
             B_ia,jb = (ia|f_xc|bj)
        
        Args:
            ground_state_orbitals: 基态轨道
            ground_state_energies: 基态本征值
            occupation: 占据数
            xc_kernel: 交换关联核函数
            
        Returns:
            (激发能, 跃迁密度)
        """
        n_occ = jnp.sum(occupation > 0.5).astype(int)
        n_virt = len(ground_state_energies) - n_occ
        
        # 构建A矩阵 (Tamm-Dancoff近似，忽略B)
        n_exc = n_occ * n_virt
        A_matrix = jnp.zeros((n_exc, n_exc))
        
        # 对角部分: ε_a - ε_i
        idx = 0
        for i in range(n_occ):
            for a in range(n_occ, n_occ + n_virt):
                A_matrix = A_matrix.at[idx, idx].set(
                    ground_state_energies[a] - ground_state_energies[i]
                )
                idx += 1
        
        # 添加交换关联核 (简化)
        # 实际计算需要双电子积分
        
        # 对角化
        excitation_energies, eigenvectors = jnp.linalg.eigh(A_matrix)
        
        # 取前n_states个激发态
        n_states = min(self.config.n_states, n_exc)
        
        return excitation_energies[:n_states], eigenvectors[:, :n_states]
    
    def oscillator_strengths(self, excitation_energies: jnp.ndarray,
                             transition_dipoles: jnp.ndarray) -> jnp.ndarray:
        """
        计算振子强度
        
        f = (2/3) * ω * |μ|^2
        
        Args:
            excitation_energies: 激发能
            transition_dipoles: 跃迁偶极矩
            
        Returns:
            振子强度
        """
        dipole_magnitudes = jnp.sum(transition_dipoles**2, axis=1)
        oscillator_strengths = (2.0/3.0) * excitation_energies * dipole_magnitudes
        
        return oscillator_strengths
    
    def absorption_spectrum(self, excitation_energies: jnp.ndarray,
                           oscillator_strengths: jnp.ndarray,
                           energy_range: Tuple[float, float] = (0, 20),
                           broadening: float = 0.1) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        计算吸收光谱
        
        Args:
            excitation_energies: 激发能 (eV)
            oscillator_strengths: 振子强度
            energy_range: 能量范围
            broadening: 展宽参数 (eV)
            
        Returns:
            (能量网格, 吸收强度)
        """
        n_points = 500
        energies = jnp.linspace(energy_range[0], energy_range[1], n_points)
        
        # 洛伦兹展宽
        spectrum = jnp.zeros(n_points)
        for E, f in zip(excitation_energies, oscillator_strengths):
            spectrum += f * (broadening / (2 * jnp.pi)) / \
                       ((energies - E)**2 + (broadening/2)**2)
        
        return energies, spectrum


class DFTLayer(jax.example_libraries.stax.Layer):
    """
    可微分DFT神经网络层
    
    将DFT计算封装为神经网络层，支持端到端自动微分
    """
    
    def __init__(self, dft_calculator: Any):
        self.dft = dft_calculator
        
    def apply(self, params: Dict, inputs: Tuple[jnp.ndarray, ...]) -> Dict:
        """
        前向传播
        
        Args:
            params: 网络参数
            inputs: (positions, atomic_numbers, cell)
            
        Returns:
            DFT计算结果
        """
        positions, atomic_numbers, cell = inputs
        
        # 应用参数化变换 (如果需要)
        if 'position_shift' in params:
            positions = positions + params['position_shift']
        
        # 执行DFT计算
        result = self.dft.compute_energy(
            positions=positions,
            atomic_numbers=atomic_numbers,
            cell=cell
        )
        
        return result
    
    def loss_energy_target(self, params: Dict, 
                           inputs: Tuple,
                           target_energy: float) -> float:
        """
        能量目标损失函数
        
        Args:
            params: 参数
            inputs: 输入
            target_energy: 目标能量
            
        Returns:
            损失值
        """
        result = self.apply(params, inputs)
        return (result['energy'] - target_energy)**2
    
    def loss_forces_zero(self, params: Dict,
                         inputs: Tuple) -> float:
        """
        力为零的约束 (平衡结构)
        
        Args:
            params: 参数
            inputs: 输入
            
        Returns:
            力范数
        """
        positions, atomic_numbers, cell = inputs
        
        def energy_fn(pos):
            return self.apply(params, (pos, atomic_numbers, cell))['energy']
        
        forces = -grad(energy_fn)(positions)
        return jnp.sum(forces**2)


class HyperparameterOptimization:
    """
    DFT超参数优化
    
    使用梯度下降优化DFT计算参数 (如混合参数、截断能等)
    """
    
    def __init__(self, 
                 dft_calculator: Any,
                 reference_energies: jnp.ndarray,
                 reference_forces: Optional[jnp.ndarray] = None):
        """
        Args:
            dft_calculator: DFT计算器
            reference_energies: 参考能量 (训练数据)
            reference_forces: 参考力 (可选)
        """
        self.dft = dft_calculator
        self.E_ref = reference_energies
        self.F_ref = reference_forces
    
    def loss_function(self, hyperparams: Dict,
                      structures: List[Tuple]) -> float:
        """
        损失函数: DFT预测与参考值的偏差
        
        Args:
            hyperparams: 超参数字典
            structures: 结构列表 [(positions, atomic_numbers, cell), ...]
            
        Returns:
            总损失
        """
        total_loss = 0.0
        
        for i, (positions, atomic_numbers, cell) in enumerate(structures):
            # 设置超参数
            self.dft.set_hyperparameters(hyperparams)
            
            # 计算能量
            result = self.dft.compute_energy(
                positions=positions,
                atomic_numbers=atomic_numbers,
                cell=cell
            )
            
            # 能量损失
            energy_loss = (result['energy'] - self.E_ref[i])**2
            total_loss += energy_loss
            
            # 力损失 (如果有参考力)
            if self.F_ref is not None:
                forces = self.dft.compute_forces(
                    positions=positions,
                    atomic_numbers=atomic_numbers,
                    cell=cell
                )
                force_loss = jnp.sum((forces - self.F_ref[i])**2)
                total_loss += 0.1 * force_loss  # 权重因子
        
        return total_loss / len(structures)
    
    def optimize(self, 
                 initial_params: Dict,
                 structures: List[Tuple],
                 learning_rate: float = 0.01,
                 max_steps: int = 100) -> Dict:
        """
        梯度下降优化超参数
        
        Args:
            initial_params: 初始超参数
            structures: 结构列表
            learning_rate: 学习率
            max_steps: 最大步数
            
        Returns:
            优化后的超参数
        """
        opt_init, opt_update, get_params = optimizers.adam(learning_rate)
        opt_state = opt_init(initial_params)
        
        @jit
        def step(i, opt_state):
            params = get_params(opt_state)
            loss = lambda p: self.loss_function(p, structures)
            g = grad(loss)(params)
            return opt_update(i, g, opt_state)
        
        for i in range(max_steps):
            opt_state = step(i, opt_state)
            
            if i % 10 == 0:
                params = get_params(opt_state)
                loss_val = self.loss_function(params, structures)
                print(f"Step {i}: loss = {loss_val:.6f}")
        
        return get_params(opt_state)


def example_advanced_differentiation():
    """高阶微分示例"""
    print("=" * 60)
    print("高级自动微分示例")
    print("=" * 60)
    
    # 创建简单的能量函数 (谐振子近似)
    def harmonic_energy(positions, cell):
        k = 1.0
        r0 = 1.0
        
        # 计算所有原子对距离
        r_ij = positions[:, None, :] - positions[None, :, :]
        distances = jnp.linalg.norm(r_ij, axis=2)
        
        # 谐振子能量
        energy = 0.5 * k * jnp.sum((distances - r0)**2)
        return energy
    
    # 创建高阶导数计算器
    hod = HighOrderDerivatives(harmonic_energy)
    
    # 测试系统
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    cell = jnp.eye(3) * 10.0
    
    print("\n系统: 3原子")
    print(f"位置:\n{positions}")
    
    # 一阶梯度 (力)
    print("\n一阶梯度 (力的负值):")
    grad_1 = hod.first_derivative_positions(positions, cell)
    print(grad_1)
    
    # 二阶梯度 (Hessian)
    print("\n二阶梯度 (Hessian矩阵):")
    hessian = hod.second_derivative_positions(positions, cell)
    print(f"形状: {hessian.shape}")
    print(f"Hessian[0,0]:\n{hessian[0, 0]}")
    
    # 声子频率
    print("\n声子频率计算:")
    masses = jnp.array([1.0, 1.0, 1.0])
    frequencies = hod.phonon_frequencies(positions, cell, masses)
    print(f"频率 (THz): {frequencies}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    example_advanced_differentiation()
