#!/usr/bin/env python3
"""
强化学习材料优化引擎 - 使用示例

本文件演示如何使用rl_optimizer模块进行材料优化。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def example_1_basic_rl():
    """示例1: 基础RL优化"""
    print("\n" + "=" * 70)
    print("示例1: 基础强化学习材料优化")
    print("=" * 70)
    
    from rl_optimizer import (
        CrystalStructureEnv,
        CrystalState,
        SACAgent,
        SACConfig,
        EnvConfig,
    )
    
    # 创建环境配置
    env_config = EnvConfig(
        max_steps=20,
        max_atoms=20,
        element_set=['Li', 'O', 'P', 'S', 'Si']
    )
    
    # 创建环境
    env = CrystalStructureEnv(config=env_config)
    
    # 创建SAC智能体
    agent_config = SACConfig(
        state_dim=env.state_rep.get_feature_dim(),
        action_dim=10,
        learning_rate=3e-4
    )
    agent = SACAgent(agent_config)
    
    print("\n训练智能体...")
    
    # 训练几个episode
    for episode in range(5):
        state = env.reset()
        episode_reward = 0
        
        for step in range(20):
            # 选择动作
            action = agent.select_action(state, deterministic=False)
            
            # 执行动作
            result = env.step(action)
            
            # 存储经验
            agent.store_transition(
                state, action, result.reward,
                result.state, result.done
            )
            
            episode_reward += result.reward
            
            # 更新
            agent.update()
            
            if result.done:
                break
            
            state = result.state
        
        print(f"Episode {episode}: Reward = {episode_reward:.4f}")
    
    # 获取最终结构
    final_structure = env.get_structure()
    print(f"\n最终结构: {final_structure.get_composition()}")
    
    return env, agent


def example_2_battery():
    """示例2: 电池材料优化"""
    print("\n" + "=" * 70)
    print("示例2: 电池材料优化")
    print("=" * 70)
    
    from rl_optimizer import BatteryOptimizer, BatteryConfig
    
    config = BatteryConfig(
        n_episodes=10,  # 演示用
        max_steps_per_episode=20
    )
    
    optimizer = BatteryOptimizer(config=config)
    
    print("\n开始电池材料优化...")
    results = optimizer.train(n_episodes=10)
    
    print(f"\n训练完成!")
    print(f"发现 {len(results['best_structures'])} 个候选结构")
    
    return optimizer, results


def example_3_alloy():
    """示例3: 合金多目标优化"""
    print("\n" + "=" * 70)
    print("示例3: 合金多目标优化")
    print("=" * 70)
    
    from rl_optimizer import AlloyOptimizer, AlloyConfig
    
    config = AlloyConfig(
        alloy_type='lightweight',
        population_size=20,
        n_generations=10
    )
    
    optimizer = AlloyOptimizer(config=config)
    
    print("\n开始合金多目标优化...")
    pareto_front = optimizer.optimize()
    
    print(f"\n优化完成!")
    print(f"帕累托前沿大小: {len(pareto_front)}")
    
    # 显示最佳合金
    best_alloys = optimizer.get_best_alloys(n=3)
    print("\n最佳合金:")
    for i, alloy in enumerate(best_alloys):
        print(f"  {i+1}. 强度={alloy['strength']:.1f}MPa, "
              f"延展性={alloy['ductility']:.3f}, "
              f"密度={alloy['density']:.2f}g/cm³")
    
    return optimizer, pareto_front


def example_4_dft_coupling():
    """示例4: DFT/MD耦合"""
    print("\n" + "=" * 70)
    print("示例4: DFT/ML耦合")
    print("=" * 70)
    
    from rl_optimizer import (
        DFTCoupling,
        MLCoupling,
        ActiveLearningCoupling,
    )
    
    from rl_optimizer.environment import CrystalState
    
    # 创建DFT耦合器
    dft = DFTCoupling(calculator='vasp')
    
    # 创建ML耦合器
    ml = MLCoupling(potential_type='nep')
    
    # 创建主动学习耦合器
    al_coupling = ActiveLearningCoupling(
        dft_coupling=dft,
        ml_coupling=ml,
        uncertainty_threshold=0.2
    )
    
    # 创建测试结构
    lattice = np.eye(3) * 5.0
    positions = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    structure = CrystalState(
        lattice=lattice,
        positions=positions,
        elements=['Li', 'O']
    )
    
    print("\n计算能量...")
    
    # 使用主动学习耦合计算能量
    for i in range(5):
        energy, method = al_coupling.calculate_energy(structure)
        print(f"  结构{i+1}: 能量={energy:.4f} eV (使用{method.upper()})")
        
        # 稍微修改结构
        structure.positions += np.random.normal(0, 0.1, structure.positions.shape)
    
    # 查看统计
    stats = al_coupling.get_stats()
    print(f"\n统计信息:")
    print(f"  DFT调用次数: {stats['n_dft_calls']}")
    print(f"  ML调用次数: {stats['n_ml_calls']}")
    print(f"  DFT比例: {stats['dft_ratio']*100:.1f}%")
    
    return al_coupling


def example_5_explainability():
    """示例5: 可解释性分析"""
    print("\n" + "=" * 70)
    print("示例5: 可解释性分析")
    print("=" * 70)
    
    from rl_optimizer import (
        TrajectoryAnalyzer,
        ChemicalIntuitionExtractor,
    )
    
    # 创建轨迹分析器
    analyzer = TrajectoryAnalyzer()
    
    # 模拟一些轨迹数据
    for _ in range(10):
        trajectory = []
        for step in range(20):
            trajectory.append({
                'step': step,
                'reward': np.random.random(),
                'action_type': np.random.choice(['add', 'remove', 'move'])
            })
        analyzer.add_trajectory(trajectory)
    
    print("\n分析轨迹...")
    
    # 分析动作分布
    action_dist = analyzer.analyze_action_distribution()
    print(f"\n动作分布:")
    for action, freq in action_dist['frequencies'].items():
        print(f"  {action}: {freq*100:.1f}%")
    
    # 分析奖励进展
    reward_prog = analyzer.analyze_reward_progression()
    print(f"\n奖励进展:")
    print(f"  平均最大奖励: {reward_prog['mean_max_reward']:.4f}")
    
    # 识别模式
    patterns = analyzer.identify_common_patterns()
    print(f"\n常见模式:")
    for pattern in patterns[:3]:
        print(f"  {pattern['pattern']} (支持度: {pattern['support']})")
    
    return analyzer


def example_6_visualization():
    """示例6: 可视化"""
    print("\n" + "=" * 70)
    print("示例6: 可视化")
    print("=" * 70)
    
    from rl_optimizer.visualization import OptimizationPlotter
    
    plotter = OptimizationPlotter()
    
    # 添加一些示例数据
    for i in range(100):
        reward = np.sin(i / 10) + np.random.normal(0, 0.1)
        plotter.add_point(i, reward)
    
    print("\n生成奖励曲线图...")
    
    # 保存路径
    output_path = '/tmp/rl_optimizer_reward_curve.png'
    plotter.plot_reward_curve(save_path=output_path)
    
    print(f"图表已保存到: {output_path}")
    
    return plotter


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("强化学习材料优化引擎 - 综合示例")
    print("=" * 70)
    
    # 运行所有示例
    try:
        example_1_basic_rl()
    except Exception as e:
        print(f"示例1错误: {e}")
    
    try:
        example_2_battery()
    except Exception as e:
        print(f"示例2错误: {e}")
    
    try:
        example_3_alloy()
    except Exception as e:
        print(f"示例3错误: {e}")
    
    try:
        example_4_dft_coupling()
    except Exception as e:
        print(f"示例4错误: {e}")
    
    try:
        example_5_explainability()
    except Exception as e:
        print(f"示例5错误: {e}")
    
    try:
        example_6_visualization()
    except Exception as e:
        print(f"示例6错误: {e}")
    
    print("\n" + "=" * 70)
    print("所有示例完成!")
    print("=" * 70)
