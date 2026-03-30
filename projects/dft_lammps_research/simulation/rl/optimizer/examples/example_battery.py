#!/usr/bin/env python3
"""
电池材料优化示例

演示如何使用RL优化器优化固态电解质材料。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import logging

from rl_optimizer import (
    CrystalStructureEnv,
    CrystalState,
    SACAgent,
    BatteryOptimizer,
    BatteryConfig,
    EnvConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_battery_optimization():
    """电池材料优化示例"""
    
    print("=" * 60)
    print("电池材料优化示例")
    print("=" * 60)
    
    # 创建配置
    battery_config = BatteryConfig(
        target_ionic_conductivity=1e-3,  # S/cm
        target_voltage_min=2.0,  # V
        target_voltage_max=5.0,  # V
        n_episodes=100,  # 训练episode数
        max_steps_per_episode=30
    )
    
    env_config = EnvConfig(
        max_steps=50,
        max_atoms=50,
        element_set=['Li', 'Na', 'P', 'S', 'Cl', 'O', 'Ge', 'Si', 'Al']
    )
    
    # 创建优化器
    optimizer = BatteryOptimizer(
        config=battery_config,
        env_config=env_config,
        agent_type='sac'
    )
    
    print("\n1. 开始训练优化器...")
    print(f"   训练episode数: {battery_config.n_episodes}")
    print(f"   每episode最大步数: {battery_config.max_steps_per_episode}")
    
    # 训练
    results = optimizer.train(n_episodes=50)  # 减少数量用于演示
    
    print("\n2. 训练完成!")
    print(f"   发现的最佳结构数: {len(results['best_structures'])}")
    
    if results['best_structures']:
        print("\n3. 最佳结构:")
        for i, struct in enumerate(results['best_structures'][:5]):
            print(f"   {i+1}. Episode {struct['episode']}: "
                  f"Reward={struct['reward']:.4f}, "
                  f"Formula={struct['formula']}")
    
    # 评估最佳结构
    print("\n4. 评估最佳结构...")
    if results['best_structures']:
        best_struct = results['best_structures'][0]['structure']
        properties = optimizer.evaluate(best_struct)
        
        print(f"   离子电导率: {properties['ionic_conductivity']:.2e} S/cm")
        print(f"   工作电压: {properties['voltage']:.2f} V")
        print(f"   稳定性: {properties['stability']:.2f}")
        print(f"   成本: ${properties['cost']:.2f}/kWh")
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)
    
    return optimizer, results


def example_single_optimization():
    """单次优化示例"""
    
    print("\n" + "=" * 60)
    print("单次优化示例")
    print("=" * 60)
    
    # 创建初始结构 (Li3PS4简化版)
    lattice = np.eye(3) * 8.0
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.25, 0.25, 0.25],
        [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75],
        [0.25, 0.75, 0.75],
    ])
    elements = ['Li', 'Li', 'Li', 'P', 'S', 'S', 'S', 'S']
    
    initial_structure = CrystalState(
        lattice=lattice,
        positions=positions,
        elements=elements
    )
    
    print(f"初始结构: {initial_structure.get_composition()}")
    
    # 创建优化器
    optimizer = BatteryOptimizer()
    
    # 执行优化
    print("\n优化中...")
    optimized_structure = optimizer.optimize(initial_structure)
    
    print(f"\n优化后结构: {optimized_structure.get_composition()}")
    
    # 评估
    properties = optimizer.evaluate(optimized_structure)
    print(f"\n性质预测:")
    print(f"  离子电导率: {properties['ionic_conductivity']:.2e} S/cm")
    print(f"  工作电压: {properties['voltage']:.2f} V")
    
    return optimized_structure


if __name__ == '__main__':
    # 运行示例
    optimizer, results = example_battery_optimization()
    
    # 运行单次优化
    # optimized = example_single_optimization()
