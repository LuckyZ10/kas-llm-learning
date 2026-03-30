"""
KAS DRL - Training Script
训练脚本入口
"""
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from training.environment import KASAgentEnv, CurriculumEnv
from training.trainer import Trainer, TrainingConfig
from core.state_space import StateSpace, AgentState, TaskFeatures, UserFeedback, TelemetryStateTracker
from core.action_space import ActionSpace
from core.reward import RewardFunction, RewardConfig, InteractionOutcome
from algorithms.ppo import PPOAgent, PPOConfig
from algorithms.sac import SACAgent, SACConfig
from algorithms.ddpg import DDPGAgent, DDPGConfig


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_agent(config: dict, device: str):
    """创建Agent"""
    agent_type = config['agent']['type'].lower()
    state_dim = config['agent']['state_dim']
    action_dim = config['agent']['action_dim']
    
    if agent_type == 'ppo':
        ppo_config = PPOConfig(
            lr=config['agent'].get('lr', 3e-4),
            gamma=config['agent'].get('gamma', 0.99),
            gae_lambda=config['agent'].get('gae_lambda', 0.95),
            clip_epsilon=config['agent'].get('clip_epsilon', 0.2),
            value_coef=config['agent'].get('value_coef', 0.5),
            entropy_coef=config['agent'].get('entropy_coef', 0.01),
            max_grad_norm=config['agent'].get('max_grad_norm', 0.5),
            ppo_epochs=config['agent'].get('ppo_epochs', 10),
            batch_size=config['agent'].get('batch_size', 64),
            hidden_dim=config['agent'].get('hidden_dim', 256)
        )
        return PPOAgent(state_dim, action_dim, ppo_config, device)
    
    elif agent_type == 'sac':
        sac_config = SACConfig(
            lr=config['agent'].get('lr', 3e-4),
            gamma=config['agent'].get('gamma', 0.99),
            tau=config['agent'].get('tau', 0.005),
            alpha=config['agent'].get('alpha', 0.2),
            automatic_entropy_tuning=config['agent'].get('automatic_entropy_tuning', True),
            buffer_size=config['agent'].get('buffer_size', 100000),
            batch_size=config['agent'].get('batch_size', 256),
            hidden_dim=config['agent'].get('hidden_dim', 256),
            updates_per_step=config['agent'].get('updates_per_step', 1)
        )
        return SACAgent(state_dim, action_dim, sac_config, device)
    
    elif agent_type == 'ddpg':
        ddpg_config = DDPGConfig(
            actor_lr=config['agent'].get('actor_lr', 1e-4),
            critic_lr=config['agent'].get('critic_lr', 1e-3),
            gamma=config['agent'].get('gamma', 0.99),
            tau=config['agent'].get('tau', 0.005),
            buffer_size=config['agent'].get('buffer_size', 100000),
            batch_size=config['agent'].get('batch_size', 64),
            hidden_dim=config['agent'].get('hidden_dim', 256),
            noise_std=config['agent'].get('noise_std', 0.1)
        )
        return DDPGAgent(state_dim, action_dim, ddpg_config, device)
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def create_env(config: dict):
    """创建环境"""
    env_config = config.get('environment', {})
    
    # 可以使用课程学习环境
    use_curriculum = env_config.get('use_curriculum', False)
    
    if use_curriculum:
        from training.environment import CurriculumEnv
        return CurriculumEnv(config=env_config)
    else:
        return KASAgentEnv(config=env_config)


def train(args):
    """训练主函数"""
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = config['training'].get('device', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    print(f"Using device: {device}")
    print(f"Training agent: {config['agent']['type']}")
    
    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # 创建环境
    env = create_env(config)
    print(f"Environment created: {type(env).__name__}")
    
    # 创建Agent
    agent = create_agent(config, device)
    print(f"Agent created: {type(agent).__name__}")
    
    # 创建训练配置
    train_config = TrainingConfig(
        num_episodes=config['training'].get('num_episodes', 1000),
        max_steps_per_episode=config['training'].get('max_steps_per_episode', 500),
        eval_interval=config['training'].get('eval_interval', 50),
        save_interval=config['training'].get('save_interval', 100),
        log_dir=config['training'].get('log_dir', './logs'),
        experiment_name=config['training'].get('experiment_name', 'kas_drl'),
        use_early_stopping=config['training'].get('use_early_stopping', True),
        patience=config['training'].get('patience', 50),
        min_delta=config['training'].get('min_delta', 0.01),
        device=device
    )
    
    # 创建训练器
    trainer = Trainer(agent, env, train_config)
    
    # 加载已有模型（如果指定）
    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    
    history = trainer.train()
    
    # 打印最终结果
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    print(f"Final mean reward: {np.mean(history['episode_rewards'][-100:]):.2f}")
    print(f"Best reward: {max(history['episode_rewards']):.2f}")
    print(f"Logs saved to: {trainer.log_path}")


def evaluate(args):
    """评估模型"""
    config = load_config(args.config)
    device = config['training'].get('device', 'cpu')
    
    # 创建环境和Agent
    env = create_env(config)
    agent = create_agent(config, device)
    
    # 加载模型
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        agent.load(args.checkpoint)
    else:
        print("Warning: No checkpoint specified, using random agent")
    
    # 评估
    print("\nEvaluating...")
    rewards = []
    
    for episode in range(args.num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(500):
            action = agent.select_action(state, deterministic=True)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            
            if done:
                break
        
        rewards.append(episode_reward)
        print(f"Episode {episode+1}/{args.num_episodes}: Reward = {episode_reward:.2f}")
    
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    print(f"Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Min reward: {np.min(rewards):.2f}")
    print(f"Max reward: {np.max(rewards):.2f}")


def main():
    parser = argparse.ArgumentParser(description='KAS DRL Training')
    parser.add_argument('command', choices=['train', 'eval'], help='Command to run')
    parser.add_argument('--config', '-c', type=str, required=True, help='Config file path')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path to load')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-episodes', type=int, default=10, help='Number of evaluation episodes')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'eval':
        evaluate(args)


if __name__ == '__main__':
    main()
