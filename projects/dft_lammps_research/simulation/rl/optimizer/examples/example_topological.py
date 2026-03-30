#!/usr/bin/env python3
"""
拓扑材料发现示例

演示如何使用RL发现拓扑材料。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import logging

from rl_optimizer import (
    TopologicalOptimizer,
    TopologicalConfig,
    EnvConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_topological_discovery():
    """拓扑材料发现示例"""
    
    print("=" * 60)
    print("拓扑材料发现示例")
    print("=" * 60)
    
    # 创建配置
    topo_config = TopologicalConfig(
        target_invariant='Z2',
        target_gap_range=(0.1, 0.5),  # eV
        prefer_2d=False,
        prefer_heavy_elements=True,
        n_episodes=50,
        max_steps_per_episode=30
    )
    
    env_config = EnvConfig(
        max_steps=30,
        max_atoms=20,  # 小团簇
        element_set=['Bi', 'Sb', 'Te', 'Se', 'Pb', 'Sn', 'W', 'Mo']
    )
    
    # 创建优化器
    optimizer = TopologicalOptimizer(
        config=topo_config,
        env_config=env_config
    )
    
    print(f"\n优化目标:")
    print(f"  拓扑不变量: {topo_config.target_invariant}")
    print(f"  目标带隙: {topo_config.target_gap_range} eV")
    print(f"  优先重元素: {topo_config.prefer_heavy_elements}")
    
    print("\n开始发现...")
    
    # 训练
    results = optimizer.train(n_episodes=30)
    
    print(f"\n发现完成!")
    print(f"发现的拓扑材料数: {results['n_discovered']}")
    
    if results['discovered_materials']:
        print("\n发现的拓扑材料:")
        for i, material in enumerate(results['discovered_materials'][:5]):
            formula = material['formula']
            print(f"\n  {i+1}. Episode {material['episode']}: {formula}")
            print(f"     奖励: {material['reward']:.4f}")
            
            # 验证拓扑性质
            validation = optimizer.validate_topology(material['structure'])
            print(f"     拓扑性质验证:")
            print(f"       是否为拓扑材料: {validation['is_topological']}")
            print(f"       置信度: {validation['confidence']:.2f}")
            if 'estimated_gap' in validation:
                print(f"       估计带隙: {validation['estimated_gap']:.3f} eV")
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)
    
    return optimizer, results


def example_dft_preparation():
    """准备DFT计算示例"""
    
    print("\n" + "=" * 60)
    print("DFT计算准备示例")
    print("=" * 60)
    
    # 创建一个示例结构
    from rl_optimizer import CrystalState
    
    lattice = np.array([
        [4.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 10.0]
    ])
    
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.25],
        [0.5, 0.0, 0.25],
    ])
    
    elements = ['Bi', 'Bi', 'Te', 'Te']
    
    structure = CrystalState(
        lattice=lattice,
        positions=positions,
        elements=elements
    )
    
    print(f"\n示例结构: {structure.get_composition()}")
    
    # 创建优化器并生成DFT输入
    optimizer = TopologicalOptimizer()
    
    output_file = '/tmp/Bi2Te2_poscar.vasp'
    optimizer.generate_dft_input(structure, output_file)
    
    print(f"\nDFT输入文件已生成: {output_file}")
    
    # 读取并显示
    with open(output_file, 'r') as f:
        print("\n文件内容:")
        print(f.read())


if __name__ == '__main__':
    # 运行拓扑材料发现
    optimizer, results = example_topological_discovery()
    
    # 准备DFT计算
    # example_dft_preparation()
