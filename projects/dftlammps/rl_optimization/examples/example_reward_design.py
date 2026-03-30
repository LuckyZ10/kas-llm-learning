"""
Reward Design Example
=====================

演示如何设计和使用奖励函数
"""

import numpy as np
from dftlammps.rl_optimization.rewards.reward_design import (
    RewardDesigner,
    PropertyReward,
    ValidityReward,
    DiversityReward,
    NoveltyReward,
    MultiObjectiveReward,
    PreferenceLearning,
    RewardShaping,
)


def basic_reward_example():
    """基础奖励函数示例"""
    print("=" * 60)
    print("Basic Reward Functions")
    print("=" * 60)
    
    # 属性奖励
    print("\n1. Property Reward")
    property_reward = PropertyReward(
        property_name='bandgap',
        target_value=1.5,
        tolerance=0.3
    )
    
    test_states = [
        {'bandgap': 1.4},
        {'bandgap': 2.0},
        {'bandgap': 1.5},
    ]
    
    for state in test_states:
        reward = property_reward(state)
        print(f"  Bandgap = {state['bandgap']:.1f} eV, Reward = {reward:.3f}")
    
    # 有效性奖励
    print("\n2. Validity Reward")
    validity_reward = ValidityReward()
    
    valid_state = {'num_atoms': 10, 'atoms': ['C', 'H', 'O']}
    invalid_state = {'num_atoms': 0}
    
    print(f"  Valid state: {validity_reward(valid_state):.3f}")
    print(f"  Invalid state: {validity_reward(invalid_state):.3f}")
    
    # 多样性奖励
    print("\n3. Diversity Reward")
    diversity_reward = DiversityReward()
    
    samples = [
        {'atoms': ['C', 'C', 'H', 'H']},
        {'atoms': ['C', 'N', 'O', 'H']},
        {'atoms': ['O', 'O', 'H', 'H']},
    ]
    
    for sample in samples:
        reward = diversity_reward(sample)
        print(f"  Sample {sample['atoms']}: Diversity = {reward:.3f}")
        diversity_reward.add_sample(sample)


def composite_reward_example():
    """组合奖励函数示例"""
    print("\n" + "=" * 60)
    print("Composite Reward Design")
    print("=" * 60)
    
    # 使用奖励设计器
    designer = RewardDesigner()
    
    # 添加组件
    designer.add_component(
        'property',
        PropertyReward('bandgap', 1.5, tolerance=0.3),
        weight=0.5
    )
    
    designer.add_component(
        'validity',
        ValidityReward(),
        weight=0.3
    )
    
    designer.add_component(
        'diversity',
        DiversityReward(),
        weight=0.2
    )
    
    # 评估样本
    test_state = {
        'bandgap': 1.4,
        'num_atoms': 8,
        'atoms': ['C', 'N', 'O', 'H', 'H', 'H', 'H', 'H']
    }
    
    results = designer.evaluate(test_state)
    print("\nReward breakdown:")
    for name, value in results.items():
        print(f"  {name}: {value:.3f}")
    
    # 构建最终奖励函数
    composite_reward = designer.build()
    total_reward = composite_reward(test_state)
    print(f"\nComposite reward: {total_reward:.3f}")


def multi_objective_reward_example():
    """多目标奖励示例"""
    print("\n" + "=" * 60)
    print("Multi-Objective Reward")
    print("=" * 60)
    
    # 定义多个属性奖励
    bandgap_reward = PropertyReward('bandgap', target_value=1.5, tolerance=0.3)
    conductivity_reward = PropertyReward('conductivity', target_value=0.8, tolerance=0.2)
    stability_reward = PropertyReward('stability', target_value=0.9, tolerance=0.1)
    
    # 组合
    multi_reward = MultiObjectiveReward([
        (bandgap_reward, 0.4),
        (conductivity_reward, 0.3),
        (stability_reward, 0.3),
    ], method='weighted_sum')
    
    # 测试不同状态
    test_states = [
        {'bandgap': 1.4, 'conductivity': 0.7, 'stability': 0.9},
        {'bandgap': 2.0, 'conductivity': 0.9, 'stability': 0.8},
        {'bandgap': 1.5, 'conductivity': 0.8, 'stability': 0.9},
    ]
    
    print("\nEvaluating states:")
    for i, state in enumerate(test_states):
        reward = multi_reward(state)
        print(f"  State {i+1}: bandgap={state['bandgap']:.1f}, "
              f"conductivity={state['conductivity']:.1f}, "
              f"stability={state['stability']:.1f}")
        print(f"         Reward: {reward:.3f}")


def preference_learning_example():
    """偏好学习示例"""
    print("\n" + "=" * 60)
    print("Preference Learning")
    print("=" * 60)
    
    # 创建偏好学习器
    pref_learning = PreferenceLearning(state_dim=10, learning_rate=1e-3)
    
    # 生成合成偏好数据
    # 假设我们希望状态范数小的更好
    print("\nGenerating synthetic preferences...")
    
    for i in range(200):
        s1 = np.random.randn(10)
        s2 = np.random.randn(10)
        
        # 偏好: 范数小的更好
        if np.linalg.norm(s1) < np.linalg.norm(s2):
            preference = 0  # 偏好s1
        else:
            preference = 1  # 偏好s2
        
        pref_learning.add_preference(s1, s2, preference)
    
    # 训练
    print("Training preference model...")
    for epoch in range(10):
        stats = pref_learning.train_step(batch_size=32)
        if epoch == 0:
            print(f"  Initial loss: {stats['loss']:.3f}")
    
    print(f"  Final loss: {stats['loss']:.3f}")
    
    # 测试预测
    print("\nTesting predictions:")
    
    # 好的状态 (范数小)
    good_state = np.array([0.1, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # 差的状态 (范数大)
    bad_state = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    prob, pref = pref_learning.predict_preference(good_state, bad_state)
    print(f"  Good vs Bad: Prefer good = {pref == 0} (prob={prob:.3f})")
    
    prob, pref = pref_learning.predict_preference(bad_state, good_state)
    print(f"  Bad vs Good: Prefer good = {pref == 1} (prob={1-prob:.3f})")


def reward_shaping_example():
    """奖励整形示例"""
    print("\n" + "=" * 60)
    print("Reward Shaping")
    print("=" * 60)
    
    # 创建势函数
    def potential(state):
        """基于到目标距离的势函数"""
        if isinstance(state, dict) and 'position' in state:
            target = np.array([0.0, 0.0])
            return -np.linalg.norm(np.array(state['position']) - target)
        return 0.0
    
    shaping = RewardShaping(potential_func=potential, gamma=0.99)
    
    # 模拟轨迹
    print("\nSimulating trajectory with reward shaping:")
    
    trajectory = [
        {'position': np.array([1.0, 1.0])},
        {'position': np.array([0.7, 0.7])},
        {'position': np.array([0.4, 0.4])},
        {'position': np.array([0.1, 0.1])},
        {'position': np.array([0.0, 0.0])},
    ]
    
    base_rewards = [0, 0, 0, 0, 10]  # 只有最后一步有奖励
    
    total_shaped = 0
    for i in range(len(trajectory) - 1):
        state = trajectory[i]
        next_state = trajectory[i + 1]
        base_reward = base_rewards[i]
        
        shaped = shaping.shape_reward(state, base_reward, next_state, done=(i == len(trajectory) - 2))
        total_shaped += shaped
        
        dist = np.linalg.norm(state['position'])
        print(f"  Step {i+1}: pos=({state['position'][0]:.1f}, {state['position'][1]:.1f}), "
              f"dist={dist:.2f}, base={base_reward:.1f}, shaped={shaped:.2f}")
    
    print(f"\nTotal shaped reward: {total_shaped:.2f}")


def main():
    print("=" * 70)
    print("Reward Function Design Examples")
    print("=" * 70)
    
    # 运行示例
    basic_reward_example()
    composite_reward_example()
    multi_objective_reward_example()
    preference_learning_example()
    reward_shaping_example()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
