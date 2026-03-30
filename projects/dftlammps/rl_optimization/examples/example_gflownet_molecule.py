"""
GFlowNet Molecular Generation Example
=====================================

演示如何使用GFlowNet进行分子生成
"""

import numpy as np
import torch
from dftlammps.rl_optimization.models.gflownet import (
    GFlowNet, GFlowNetConfig, MoleculeGFlowNet
)
from dftlammps.rl_optimization.environments.molecule_env import (
    MolecularGraphEnv, MoleculeEnvConfig
)
from dftlammps.rl_optimization.training.gflownet_trainer import (
    GFlowNetTrainer, GFlowNetTrainingConfig
)
from dftlammps.rl_optimization.rewards.reward_design import (
    PropertyReward, ValidityReward, DiversityReward, MultiObjectiveReward
)


def main():
    print("=" * 70)
    print("GFlowNet Molecular Generation Example")
    print("=" * 70)
    
    # 配置
    atom_types = ['C', 'N', 'O', 'S', 'H', 'F', 'Cl']
    bond_types = ['SINGLE', 'DOUBLE', 'TRIPLE']
    max_atoms = 15
    
    # 创建环境配置
    env_config = MoleculeEnvConfig(
        max_atoms=max_atoms,
        atom_types=atom_types,
        bond_types=bond_types,
    )
    
    # 创建环境
    env = MolecularGraphEnv(env_config)
    print(f"\nEnvironment created:")
    print(f"  Atom types: {atom_types}")
    print(f"  Max atoms: {max_atoms}")
    print(f"  Action dim: {env.action_dim}")
    
    # 创建GFlowNet配置
    gfn_config = GFlowNetConfig(
        state_dim=env.state_dim if hasattr(env, 'state_dim') else 200,
        action_dim=env.action_dim,
        hidden_dim=256,
        num_layers=3,
        learning_rate=1e-3,
    )
    
    # 创建GFlowNet
    gfn = GFlowNet(gfn_config)
    print(f"\nGFlowNet created:")
    print(f"  Parameters: {sum(p.numel() for p in gfn.parameters()):,}")
    
    # 创建训练配置
    train_config = GFlowNetTrainingConfig(
        num_iterations=500,
        batch_size=16,
        num_samples_per_iter=16,
        eval_frequency=50,
        max_steps_per_trajectory=max_atoms * 2,
        use_replay_buffer=True,
    )
    
    # 创建训练器
    trainer = GFlowNetTrainer(gfn, env, train_config)
    
    # 训练
    print(f"\nTraining for {train_config.num_iterations} iterations...")
    history = trainer.train()
    
    # 生成样本
    print("\nGenerating samples...")
    samples = trainer.generate_samples(num_samples=20, epsilon=0.0)
    
    # 评估
    rewards = [s['reward'] for s in samples]
    print(f"\nGeneration Results:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Mean reward: {np.mean(rewards):.3f}")
    print(f"  Max reward: {max(rewards):.3f}")
    
    # 显示一些生成的分子
    print("\nTop 5 molecules:")
    samples.sort(key=lambda x: x['reward'], reverse=True)
    for i, sample in enumerate(samples[:5]):
        mol_data = sample['sample']
        print(f"  {i+1}. Atoms: {mol_data.get('num_atoms', 0)}, "
              f"Bonds: {mol_data.get('num_bonds', 0)}, "
              f"Reward: {sample['reward']:.3f}")
    
    # 保存模型
    save_path = "gflownet_molecule.pt"
    trainer.save(save_path)
    print(f"\nModel saved to {save_path}")
    
    print("\n" + "=" * 70)
    print("Example completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
