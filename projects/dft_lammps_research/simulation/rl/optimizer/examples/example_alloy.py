#!/usr/bin/env python3
"""
合金优化示例

演示如何使用多目标RL优化器优化合金组成。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import logging

from rl_optimizer import (
    AlloyOptimizer,
    AlloyConfig,
    EnvConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_alloy_optimization():
    """合金优化示例"""
    
    print("=" * 60)
    print("多目标合金优化示例")
    print("=" * 60)
    
    # 创建配置
    alloy_config = AlloyConfig(
        alloy_type='lightweight',
        base_elements=['Al', 'Mg', 'Ti'],
        target_strength_range=(300, 800),  # MPa
        target_ductility_range=(0.1, 0.3),
        max_density=3.0,  # g/cm³
        population_size=50,
        n_generations=50
    )
    
    env_config = EnvConfig(
        element_set=['Al', 'Mg', 'Ti', 'Cu', 'Zn', 'Mn', 'Si', 'V', 'Cr', 'Zr']
    )
    
    # 创建优化器
    optimizer = AlloyOptimizer(
        config=alloy_config,
        env_config=env_config
    )
    
    print(f"\n优化目标:")
    print(f"  合金类型: {alloy_config.alloy_type}")
    print(f"  基体元素: {alloy_config.base_elements}")
    print(f"  目标强度: {alloy_config.target_strength_range} MPa")
    print(f"  目标延展性: {alloy_config.target_ductility_range}")
    print(f"  最大密度: {alloy_config.max_density} g/cm³")
    
    print(f"\n优化参数:")
    print(f"  种群大小: {alloy_config.population_size}")
    print(f"  代数: {alloy_config.n_generations}")
    
    print("\n开始多目标优化...")
    
    # 执行优化
    pareto_front = optimizer.optimize()
    
    print(f"\n优化完成!")
    print(f"帕累托前沿大小: {len(pareto_front)}")
    
    # 获取最佳合金
    best_alloys = optimizer.get_best_alloys(n=5)
    
    print("\n最佳合金组成:")
    for i, alloy in enumerate(best_alloys):
        composition_str = ' '.join(f"{elem}{frac:.2f}" 
                                   for elem, frac in alloy['composition'].items() 
                                   if frac > 0.01)
        print(f"\n  {i+1}. {composition_str}")
        print(f"     强度: {alloy['strength']:.1f} MPa")
        print(f"     延展性: {alloy['ductility']:.3f}")
        print(f"     密度: {alloy['density']:.2f} g/cm³")
        
        # 评估详细性质
        properties = optimizer.evaluate_alloy(alloy['composition'])
        print(f"     成本: ${properties['cost_per_kg']:.2f}/kg")
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)
    
    return optimizer, pareto_front


def example_alloy_types():
    """比较不同类型的合金"""
    
    print("\n" + "=" * 60)
    print("不同类型合金优化比较")
    print("=" * 60)
    
    alloy_types = ['lightweight', 'high_strength', 'corrosion_resistant']
    
    for alloy_type in alloy_types:
        print(f"\n{alloy_type.replace('_', ' ').title()} Alloy:")
        
        config = AlloyConfig(
            alloy_type=alloy_type,
            population_size=30,
            n_generations=30
        )
        
        optimizer = AlloyOptimizer(config=config)
        pareto_front = optimizer.optimize()
        
        best_alloys = optimizer.get_best_alloys(n=1)
        if best_alloys:
            alloy = best_alloys[0]
            print(f"  最佳组成: {alloy['composition']}")
            print(f"  强度: {alloy['strength']:.1f} MPa")
            print(f"  延展性: {alloy['ductility']:.3f}")
            print(f"  密度: {alloy['density']:.2f} g/cm³")


if __name__ == '__main__':
    # 运行合金优化
    optimizer, pareto_front = example_alloy_optimization()
    
    # 比较不同类型
    # example_alloy_types()
