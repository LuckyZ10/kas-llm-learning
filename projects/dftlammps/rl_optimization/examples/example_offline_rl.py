"""
Offline RL for Material Design Example
======================================

演示如何使用离线RL进行材料设计
"""

import numpy as np
import torch
from dftlammps.rl_optimization.models.offline_rl import (
    CQL, IQL, DecisionTransformer, OfflineRLConfig
)
from dftlammps.rl_optimization.environments.material_env import (
    CompositionEnv, MaterialEnvConfig
)
from dftlammps.rl_optimization.training.offline_trainer import (
    OfflineRLTrainer, OfflineDataset, OfflineRLTrainingConfig
)


def generate_synthetic_dataset(
    num_trajectories: int = 100,
    traj_length: int = 20,
    state_dim: int = 30,
    action_dim: int = 10
) -> list:
    """生成合成数据集用于演示"""
    trajectories = []
    
    for _ in range(num_trajectories):
        traj = []
        
        # 随机初始状态
        state = np.random.randn(state_dim)
        
        for t in range(traj_length):
            # 随机动作
            action = np.random.randn(action_dim) * 0.5
            
            # 模拟奖励 (基于状态的某个函数)
            reward = np.exp(-np.linalg.norm(state) / 10) + np.random.randn() * 0.1
            
            traj.append((state.copy(), action, reward))
            
            # 状态转移
            state = state + action * 0.1 + np.random.randn(state_dim) * 0.05
        
        trajectories.append(traj)
    
    return trajectories


def train_cql_example():
    """CQL训练示例"""
    print("\n" + "=" * 60)
    print("CQL (Conservative Q-Learning) Example")
    print("=" * 60)
    
    # 生成数据集
    print("\nGenerating synthetic dataset...")
    trajectories = generate_synthetic_dataset(num_trajectories=50)
    print(f"Dataset: {len(trajectories)} trajectories")
    
    # 创建离线数据集
    dataset = OfflineDataset(
        trajectories,
        normalize_states=True,
        normalize_rewards=True
    )
    print(f"Total experiences: {len(dataset)}")
    
    # 创建CQL代理
    config = OfflineRLConfig(
        state_dim=30,
        action_dim=10,
        hidden_dim=128,
        num_layers=2,
        cql_alpha=1.0,
    )
    
    cql_agent = CQL(config)
    print(f"CQL agent created")
    
    # 创建训练器
    train_config = OfflineRLTrainingConfig(
        num_epochs=20,
        batch_size=32,
        eval_frequency=5,
    )
    
    trainer = OfflineRLTrainer(cql_agent, dataset, train_config)
    
    # 训练
    print("\nTraining CQL...")
    history = trainer.train()
    
    print(f"\nFinal loss: {history['losses'][-1]:.4f}")
    
    return cql_agent


def train_iql_example():
    """IQL训练示例"""
    print("\n" + "=" * 60)
    print("IQL (Implicit Q-Learning) Example")
    print("=" * 60)
    
    # 生成数据集
    print("\nGenerating synthetic dataset...")
    trajectories = generate_synthetic_dataset(num_trajectories=50)
    
    dataset = OfflineDataset(trajectories)
    
    # 创建IQL代理
    config = OfflineRLConfig(
        state_dim=30,
        action_dim=10,
        hidden_dim=128,
        num_layers=2,
        iql_tau=0.7,
        iql_beta=3.0,
    )
    
    iql_agent = IQL(config)
    print(f"IQL agent created")
    
    # 训练
    train_config = OfflineRLTrainingConfig(num_epochs=20, batch_size=32)
    trainer = OfflineRLTrainer(iql_agent, dataset, train_config)
    
    print("\nTraining IQL...")
    history = trainer.train()
    
    print(f"\nFinal loss: {history['losses'][-1]:.4f}")
    
    return iql_agent


def train_decision_transformer_example():
    """Decision Transformer训练示例"""
    print("\n" + "=" * 60)
    print("Decision Transformer Example")
    print("=" * 60)
    
    # 生成数据集
    print("\nGenerating synthetic dataset...")
    trajectories = generate_synthetic_dataset(num_trajectories=30, traj_length=15)
    
    dataset = OfflineDataset(trajectories)
    
    # 创建Decision Transformer
    config = OfflineRLConfig(
        state_dim=30,
        action_dim=10,
        hidden_dim=128,
        dt_max_len=20,
        dt_n_heads=4,
        dt_n_layers=3,
    )
    
    dt_agent = DecisionTransformer(config)
    print(f"Decision Transformer created")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Num layers: {config.dt_n_layers}")
    print(f"  Num heads: {config.dt_n_heads}")
    
    # 训练
    from dftlammps.rl_optimization.training.offline_trainer import DecisionTransformerTrainer
    
    train_config = OfflineRLTrainingConfig(num_epochs=20, batch_size=8)
    trainer = DecisionTransformerTrainer(dt_agent, dataset, train_config)
    
    print("\nTraining Decision Transformer...")
    history = trainer.train()
    
    print(f"\nFinal loss: {history['losses'][-1]:.4f}")
    
    return dt_agent


def compare_algorithms():
    """比较不同离线RL算法"""
    print("\n" + "=" * 60)
    print("Comparing Offline RL Algorithms")
    print("=" * 60)
    
    results = {}
    
    # CQL
    print("\n1. Training CQL...")
    cql_agent = train_cql_example()
    results['CQL'] = cql_agent
    
    # IQL
    print("\n2. Training IQL...")
    iql_agent = train_iql_example()
    results['IQL'] = iql_agent
    
    # Decision Transformer
    print("\n3. Training Decision Transformer...")
    dt_agent = train_decision_transformer_example()
    results['Decision Transformer'] = dt_agent
    
    print("\n" + "=" * 60)
    print("Comparison complete!")
    print("=" * 60)


def main():
    print("=" * 70)
    print("Offline RL for Material Design Examples")
    print("=" * 70)
    
    # 运行各个示例
    train_cql_example()
    train_iql_example()
    train_decision_transformer_example()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
