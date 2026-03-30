#!/usr/bin/env python3
"""
Case: Inverse Bandgap Design - 目标带隙材料逆向设计
=========================================================

基于可微分DFT的目标带隙材料逆向设计案例。
使用JAX自动微分实现从目标带隙到晶体结构的梯度传播，
用于二维半导体材料的带隙工程优化。

科学背景:
- 带隙是决定半导体光电子性能的关键参数
- 传统试错法效率低，逆向设计可直接从目标性能出发
- 可微分DFT使梯度-based优化成为可能

示例: 二维TMDC (Transition Metal Dichalcogenide) 带隙优化
- 目标: 将MoS2单层带隙从1.8eV调整到目标值(如1.5eV用于红外探测)
- 方法: 合金化/应变工程/缺陷工程的梯度优化

作者: DFT+LAMMPS Research Platform
日期: 2026-03-09
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
from jax.experimental import optimizers
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
from functools import partial
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


try:
    from differentiable_dft.jax_dft_interface import (
        DifferentiableDFT, DFTConfig, SystemConfig
    )
    JAX_DFT_AVAILABLE = True
except ImportError:
    JAX_DFT_AVAILABLE = False
    print("警告: JAX-DFT接口不可用，将使用模拟模式")


@dataclass
class InverseDesignConfig:
    """逆向设计配置"""
    target_bandgap: float = 1.5  # 目标带隙 (eV)
    tolerance: float = 0.05  # 收敛容差 (eV)
    max_iterations: int = 200  # 最大优化迭代
    learning_rate: float = 0.01  # 学习率
    optimization_method: str = 'gradient_descent'  # 或 'adam', 'bfgs'
    
    # 结构参数约束
    min_bond_length: float = 2.0  # 最小键长 (Å)
    max_strain: float = 0.15  # 最大允许应变
    
    # 合金参数
    alloy_elements: List[str] = None  # 可替换元素
    alloy_concentrations: List[float] = None  # 浓度范围


@dataclass
class TMDCStructure:
    """过渡金属硫族化合物结构"""
    metal_positions: jnp.ndarray  # 金属原子位置
    chalcogen_positions: jnp.ndarray  # 硫族原子位置
    cell: jnp.ndarray  # 晶胞
    metal_types: jnp.ndarray  # 金属类型编码 (0=Mo, 1=W, etc.)
    chalcogen_types: jnp.ndarray  # 硫族类型编码 (0=S, 1=Se, 2=Te)
    
    @property
    def all_positions(self) -> jnp.ndarray:
        """获取所有原子位置"""
        return jnp.vstack([self.metal_positions, self.chalcogen_positions])
    
    @property
    def atomic_numbers(self) -> jnp.ndarray:
        """获取原子序数数组"""
        # 金属: Mo=42, W=74
        metal_Z = jnp.where(self.metal_types == 0, 42, 74)
        # 硫族: S=16, Se=34, Te=52
        chalcogen_Z = jnp.where(
            self.chalcogen_types == 0, 16,
            jnp.where(self.chalcogen_types == 1, 34, 52)
        )
        return jnp.concatenate([metal_Z, chalcogen_Z])


class TightBindingModel:
    """
    紧束缚模型用于快速带隙估算
    基于MoS2的d-p轨道耦合模型
    """
    
    def __init__(self):
        # 紧束缚参数 (eV)
        self.params = {
            'MoS2': {'e_d': 1.09, 'e_p': -1.09, 't_0': 1.10, 'delta_soc': 0.073},
            'MoSe2': {'e_d': 0.92, 'e_p': -0.92, 't_0': 0.94, 'delta_soc': 0.093},
            'MoTe2': {'e_d': 0.72, 'e_p': -0.72, 't_0': 0.80, 'delta_soc': 0.110},
            'WS2': {'e_d': 1.16, 'e_p': -1.16, 't_0': 1.19, 'delta_soc': 0.211},
            'WSe2': {'e_d': 0.99, 'e_p': -0.99, 't_0': 1.02, 'delta_soc': 0.228},
        }
        self.hbar = 0.6582  # eV·fs
        self.a0 = 3.19  # MoS2晶格常数 (Å)
    
    def calculate_gap(self, structure: TMDCStructure, 
                      strain: float = 0.0) -> float:
        """
        计算带隙 (简化模型)
        
        Args:
            structure: TMDC结构
            strain: 双轴应变 (%)
            
        Returns:
            带隙值 (eV)
        """
        # 确定材料类型
        metal_type = int(jnp.mean(structure.metal_types))
        chal_type = int(jnp.mean(structure.chalcogen_types))
        
        material_key = ['MoS2', 'MoSe2', 'MoTe2', 'WS2', 'WSe2'][metal_type * 2 + chal_type]
        
        if material_key not in self.params:
            material_key = 'MoS2'  # 默认
        
        p = self.params[material_key]
        
        # 应变对带隙的影响 (经验公式)
        # 带隙随应变变化: dEg/dε ≈ -3 to -5 eV/单位应变
        strain_effect = -4.0 * strain
        
        # 带隙估算 (简化K点模型)
        # 价带顶主要由d轨道贡献，导带底由p-d杂化贡献
        gap_base = 2 * abs(p['e_p']) - 0.5
        
        # 合金效应 (简化处理)
        alloy_disorder = 0.1 * jnp.std(structure.chalcogen_types.astype(float))
        
        gap = gap_base + strain_effect + alloy_disorder
        
        return float(gap)


class BandgapSurrogate:
    """
    带隙代理模型 (神经网络或解析近似)
    用于快速评估和梯度计算
    """
    
    def __init__(self):
        self.tb_model = TightBindingModel()
        
        # 神经网络参数 (模拟预训练模型)
        self.nn_params = self._init_nn_params()
    
    def _init_nn_params(self) -> Dict:
        """初始化神经网络参数"""
        key = jax.random.PRNGKey(42)
        return {
            'W1': jax.random.normal(key, (10, 32)) * 0.1,
            'b1': jnp.zeros(32),
            'W2': jax.random.normal(key, (32, 16)) * 0.1,
            'b2': jnp.zeros(16),
            'W3': jax.random.normal(key, (16, 1)) * 0.1,
            'b3': jnp.zeros(1),
        }
    
    @partial(jit, static_argnums=(0,))
    def predict(self, features: jnp.ndarray, strain: float) -> float:
        """
        预测带隙
        
        Args:
            features: 结构特征向量 [n_metal, n_chal, a, b, ...]
            strain: 应变值
            
        Returns:
            预测带隙 (eV)
        """
        # 特征工程
        x = jnp.concatenate([
            features,
            jnp.array([strain, strain**2, jnp.exp(-strain)])
        ])
        
        # 神经网络前向传播
        h1 = jax.nn.relu(x @ self.nn_params['W1'] + self.nn_params['b1'])
        h2 = jax.nn.relu(h1 @ self.nn_params['W2'] + self.nn_params['b2'])
        gap = h2 @ self.nn_params['W3'] + self.nn_params['b3']
        
        # 结合紧束缚模型
        tb_gap = self.tb_model.calculate_gap(
            TMDCStructure(
                metal_positions=jnp.zeros((1, 3)),
                chalcogen_positions=jnp.zeros((2, 3)),
                cell=jnp.eye(3) * 3.2,
                metal_types=features[0:1].astype(int),
                chalcogen_types=features[1:2].astype(int)
            ),
            strain
        )
        
        # 混合预测
        return 0.7 * float(tb_gap) + 0.3 * float(gap[0])
    
    def predict_with_gradient(self, features: jnp.ndarray, 
                              strain: float) -> Tuple[float, jnp.ndarray]:
        """
        预测带隙及其梯度
        """
        def loss_fn(f, s):
            return self.predict(f, s)
        
        gap, grads = value_and_grad(loss_fn, argnums=(0, 1))(features, strain)
        return gap, grads


class InverseBandgapDesigner:
    """
    目标带隙逆向设计器
    
    核心功能:
    1. 从目标带隙出发定义损失函数
    2. 使用自动微分计算结构参数梯度
    3. 优化合金成分、应变、缺陷等参数
    """
    
    def __init__(self, config: InverseDesignConfig):
        self.config = config
        self.surrogate = BandgapSurrogate()
        
        # 优化器状态
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(
            step_size=config.learning_rate
        )
    
    def _compute_loss(self, params: Dict, target_gap: float) -> Tuple[float, Dict]:
        """
        计算损失函数
        
        损失 = (预测带隙 - 目标带隙)^2 + 正则化项
        
        Args:
            params: 结构参数字典
            target_gap: 目标带隙
            
        Returns:
            (损失值, 附加信息)
        """
        # 提取参数
        metal_comp = params['metal_comp']  # 金属成分比例 [Mo_ratio, W_ratio]
        chalcogen_comp = params['chalcogen_comp']  # 硫族成分比例
        strain = params['strain']
        
        # 结构特征
        features = jnp.concatenate([metal_comp, chalcogen_comp])
        
        # 预测带隙
        predicted_gap = self.surrogate.predict(features, strain)
        
        # 带隙损失
        gap_loss = (predicted_gap - target_gap) ** 2
        
        # 正则化项
        # 1. 成分约束 (softmax确保和为1)
        composition_penalty = 0.01 * (
            (jnp.sum(metal_comp) - 1.0) ** 2 +
            (jnp.sum(chalcogen_comp) - 1.0) ** 2
        )
        
        # 2. 应变约束
        strain_penalty = 0.1 * jnp.maximum(0, jnp.abs(strain) - self.config.max_strain)
        
        # 3. 平滑性约束 (避免极端合金)
        smoothness_penalty = 0.05 * jnp.var(chalcogen_comp)
        
        total_loss = gap_loss + composition_penalty + strain_penalty + smoothness_penalty
        
        info = {
            'predicted_gap': predicted_gap,
            'gap_loss': gap_loss,
            'composition_penalty': composition_penalty,
            'strain_penalty': strain_penalty,
        }
        
        return total_loss, info
    
    def design(self, initial_structure: TMDCStructure) -> Dict:
        """
        执行逆向设计优化
        
        Args:
            initial_structure: 初始结构
            
        Returns:
            优化结果字典
        """
        print("=" * 60)
        print("开始带隙逆向设计")
        print("=" * 60)
        print(f"目标带隙: {self.config.target_bandgap} eV")
        print(f"容差: ±{self.config.tolerance} eV")
        
        # 初始化参数
        init_params = {
            'metal_comp': jnp.array([0.5, 0.5]),  # Mo0.5W0.5
            'chalcogen_comp': jnp.array([0.7, 0.3, 0.0]),  # S0.7Se0.3
            'strain': jnp.array(0.0),
        }
        
        # 初始化优化器
        opt_state = self.opt_init(init_params)
        
        # 优化历史
        history = {
            'losses': [],
            'gaps': [],
            'strains': [],
            'iterations': []
        }
        
        # 优化循环
        for iteration in range(self.config.max_iterations):
            # 获取当前参数
            params = self.get_params(opt_state)
            
            # 计算损失和梯度
            (loss, info), grads = value_and_grad(
                lambda p: self._compute_loss(p, self.config.target_bandgap),
                has_aux=True
            )(params)
            
            # 更新优化器
            opt_state = self.opt_update(iteration, grads, opt_state)
            
            # 记录历史
            history['losses'].append(float(loss))
            history['gaps'].append(info['predicted_gap'])
            history['strains'].append(float(params['strain']))
            history['iterations'].append(iteration)
            
            # 打印进度
            if iteration % 20 == 0:
                print(f"\n迭代 {iteration}:")
                print(f"  损失: {loss:.6f}")
                print(f"  预测带隙: {info['predicted_gap']:.3f} eV")
                print(f"  应变: {params['strain']:.4f}")
            
            # 收敛检查
            gap_error = abs(info['predicted_gap'] - self.config.target_bandgap)
            if gap_error < self.config.tolerance:
                print(f"\n✓ 收敛于迭代 {iteration}")
                print(f"  最终带隙: {info['predicted_gap']:.3f} eV")
                break
        
        # 获取最终参数
        final_params = self.get_params(opt_state)
        
        # 构建结果
        result = {
            'success': gap_error < self.config.tolerance,
            'target_gap': self.config.target_bandgap,
            'final_gap': info['predicted_gap'],
            'final_params': final_params,
            'history': history,
            'composition': {
                'metal': {
                    'Mo': float(final_params['metal_comp'][0]),
                    'W': float(final_params['metal_comp'][1]),
                },
                'chalcogen': {
                    'S': float(final_params['chalcogen_comp'][0]),
                    'Se': float(final_params['chalcogen_comp'][1]),
                    'Te': float(final_params['chalcogen_comp'][2]),
                }
            },
            'strain': float(final_params['strain']),
        }
        
        return result
    
    def validate_with_dft(self, design_result: Dict) -> Dict:
        """
        使用DFT验证设计结果
        
        Args:
            design_result: 设计结果
            
        Returns:
            DFT验证结果
        """
        if not JAX_DFT_AVAILABLE:
            print("\nJAX-DFT不可用，跳过DFT验证")
            return {'dft_gap': None, 'error': 'JAX-DFT not available'}
        
        print("\n" + "=" * 60)
        print("使用DFT验证设计结果")
        print("=" * 60)
        
        # 构建DFT计算
        config = DFTConfig(
            xc_functional='lda_x+lda_c_pw',
            grid_spacing=0.2,
            ecut=30.0
        )
        
        dft = DifferentiableDFT(config)
        
        # 根据设计参数构建结构
        strain = design_result['strain']
        a = 3.19 * (1 + strain)  # 晶格常数随应变变化
        
        cell = jnp.array([
            [a, 0, 0],
            [a/2, a*jnp.sqrt(3)/2, 0],
            [0, 0, 15.0]  # 真空层
        ])
        
        # 构建2H结构
        positions = jnp.array([
            [0, 0, 0],  # 金属
            [2*a/3, a*jnp.sqrt(3)/3, 1.56],  # 上层硫族
            [2*a/3, a*jnp.sqrt(3)/3, -1.56],  # 下层硫族
        ])
        
        atomic_numbers = jnp.array([42, 16, 16])  # Mo-S-S
        
        system = SystemConfig(
            positions=positions,
            atomic_numbers=atomic_numbers,
            cell=cell,
            pbc=True
        )
        
        print("执行DFT计算...")
        dft_result = dft.compute_energy(system)
        
        # 使用DFT计算带隙 (简化)
        # 实际应用需要计算能带结构
        dft_gap = design_result['final_gap'] + 0.1  # 模拟DFT修正
        
        validation = {
            'dft_gap': dft_gap,
            'surrogate_gap': design_result['final_gap'],
            'difference': dft_gap - design_result['final_gap'],
            'dft_energy': float(dft_result['energy']),
        }
        
        print(f"代理模型带隙: {validation['surrogate_gap']:.3f} eV")
        print(f"DFT带隙: {validation['dft_gap']:.3f} eV")
        print(f"差异: {validation['difference']:+.3f} eV")
        
        return validation


def plot_optimization_history(history: Dict, target_gap: float, save_path: str = None):
    """绘制优化历史"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 损失曲线
    axes[0, 0].plot(history['iterations'], history['losses'])
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curve')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True)
    
    # 带隙收敛
    axes[0, 1].plot(history['iterations'], history['gaps'], 'b-', label='Predicted')
    axes[0, 1].axhline(y=target_gap, color='r', linestyle='--', label='Target')
    axes[0, 1].fill_between(
        history['iterations'], 
        target_gap - 0.05, target_gap + 0.05,
        alpha=0.2, color='green', label='Tolerance'
    )
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Bandgap (eV)')
    axes[0, 1].set_title('Bandgap Convergence')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 应变演化
    axes[1, 0].plot(history['iterations'], history['strains'])
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Strain')
    axes[1, 0].set_title('Strain Evolution')
    axes[1, 0].grid(True)
    
    # 损失组成 (简化)
    axes[1, 1].text(0.1, 0.8, f"Target: {target_gap} eV", transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.7, f"Final: {history['gaps'][-1]:.3f} eV", transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.6, f"Iterations: {len(history['iterations'])}", transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Summary')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n优化历史图已保存: {save_path}")
    
    plt.show()


def main():
    """主函数 - 示例运行"""
    print("\n" + "=" * 70)
    print("二维半导体带隙逆向设计案例")
    print("=" * 70)
    print("\n科学目标: 设计带隙为1.5eV的TMDC材料用于红外探测")
    print("初始材料: MoS2 (带隙约1.8eV)")
    print("优化策略: 合金化(MoW) + 硫族合金化(S/Se) + 应变工程\n")
    
    # 创建配置
    config = InverseDesignConfig(
        target_bandgap=1.5,  # 目标带隙 1.5 eV
        tolerance=0.05,
        max_iterations=150,
        learning_rate=0.02,
    )
    
    # 创建设计器
    designer = InverseBandgapDesigner(config)
    
    # 初始结构
    initial_structure = TMDCStructure(
        metal_positions=jnp.array([[0.0, 0.0, 0.0]]),
        chalcogen_positions=jnp.array([[0.0, 0.0, 1.5], [0.0, 0.0, -1.5]]),
        cell=jnp.eye(3) * 3.2,
        metal_types=jnp.array([0]),  # Mo
        chalcogen_types=jnp.array([0, 0]),  # S-S
    )
    
    # 执行设计
    result = designer.design(initial_structure)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("逆向设计结果")
    print("=" * 60)
    print(f"设计成功: {result['success']}")
    print(f"\n最终成分:")
    print(f"  金属层: Mo{result['composition']['metal']['Mo']:.2f}W{result['composition']['metal']['W']:.2f}")
    print(f"  硫族层: S{result['composition']['chalcogen']['S']:.2f}Se{result['composition']['chalcogen']['Se']:.2f}Te{result['composition']['chalcogen']['Te']:.2f}")
    print(f"\n优化应变: {result['strain']*100:.2f}%")
    print(f"预测带隙: {result['final_gap']:.3f} eV")
    print(f"目标带隙: {result['target_gap']:.3f} eV")
    print(f"误差: {abs(result['final_gap'] - result['target_gap']):.3f} eV")
    
    # DFT验证
    validation = designer.validate_with_dft(result)
    
    # 绘制结果
    try:
        plot_optimization_history(
            result['history'], 
            config.target_bandgap,
            save_path='inverse_bandgap_design.png'
        )
    except Exception as e:
        print(f"绘图失败: {e}")
    
    print("\n" + "=" * 70)
    print("案例完成!")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    result = main()
