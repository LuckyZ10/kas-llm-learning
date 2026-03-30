#!/usr/bin/env python3
"""
催化剂优化示例

演示如何使用RL优化器优化催化剂活性位点。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import logging

from rl_optimizer import (
    CatalystOptimizer,
    CatalystConfig,
    EnvConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_catalyst_optimization():
    """催化剂优化示例"""
    
    print("=" * 60)
    print("催化剂优化示例")
    print("=" * 60)
    
    # 创建配置
    catalyst_config = CatalystConfig(
        reaction_type='ORR',  # 氧还原反应
        target_activity=1.0,
        target_selectivity=0.9,
        max_precious_metal_content=0.1,
        n_episodes=50,
        max_steps_per_episode=30
    )
    
    env_config = EnvConfig(
        max_steps=30,
        max_atoms=30,  # 催化剂团簇
        element_set=['Pt', 'Pd', 'Fe', 'Co', 'Ni', 'Cu', 'Mn', 'N', 'C']
    )
    
    # 创建优化器
    optimizer = CatalystOptimizer(
        config=catalyst_config,
        env_config=env_config
    )
    
    print(f"\n优化目标: {catalyst_config.reaction_type}催化剂")
    print(f"目标活性: {catalyst_config.target_activity}")
    print(f"目标选择性: {catalyst_config.target_selectivity}")
    print(f"最大贵金属含量: {catalyst_config.max_precious_metal_content * 100}%")
    
    print("\n开始训练...")
    
    # 训练
    results = optimizer.train(n_episodes=20)
    
    print("\n训练完成!")
    print(f"发现的最佳结构数: {len(results['best_structures'])}")
    
    if results['best_structures']:
        print("\n最佳催化剂:")
        for i, struct in enumerate(results['best_structures'][:3]):
            formula = struct['structure'].get_composition()
            print(f"  {i+1}. Episode {struct['episode']}: "
                  f"Reward={struct['reward']:.4f}")
            print(f"     组成: {formula}")
            
            # 评估活性位点
            sites = optimizer.evaluate_active_sites(struct['structure'])
            print(f"     识别到的活性位点数: {len(sites)}")
            if sites:
                print(f"     最佳活性位点: {sites[0]['element']} (配位数: {sites[0]['coordination']})")
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)
    
    return optimizer, results


def example_different_reactions():
    """比较不同反应的催化剂优化"""
    
    print("\n" + "=" * 60)
    print("不同反应类型催化剂比较")
    print("=" * 60)
    
    reactions = ['ORR', 'HER', 'OER', 'CO2RR']
    
    for reaction in reactions:
        print(f"\n{reaction}催化剂优化:")
        
        config = CatalystConfig(
            reaction_type=reaction,
            n_episodes=10  # 简化
        )
        
        optimizer = CatalystOptimizer(config=config)
        results = optimizer.train(n_episodes=10)
        
        if results['best_structures']:
            best = results['best_structures'][0]
            print(f"  最佳奖励: {best['reward']:.4f}")
            print(f"  组成: {best['structure'].get_composition()}")


if __name__ == '__main__':
    # 运行催化剂优化
    optimizer, results = example_catalyst_optimization()
    
    # 比较不同反应
    # example_different_reactions()
