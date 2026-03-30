"""
KAS DRL - Main Entry Point
KAS深度学习与强化学习增强模块主入口
"""
import argparse
import sys
from pathlib import Path

# 确保模块可以被导入
sys.path.insert(0, str(Path(__file__).parent))


def print_banner():
    """打印横幅"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║          KAS Deep Reinforcement Learning Module           ║
    ║                                                           ║
    ║  Features:                                                ║
    ║    • PPO/DDPG/SAC algorithms for Agent optimization       ║
    ║    • MAML/Reptile meta-learning for quick adaptation      ║
    ║    • Transformer/LSTM for telemetry modeling              ║
    ║    • Online learning with continual learning support      ║
    ║    • Seamless integration with existing LLMClient         ║
    ║    • Graceful fallback and progressive deployment         ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def cmd_train(args):
    """训练命令"""
    from train import main as train_main
    sys.argv = ['train', 'train', '--config', args.config]
    if args.seed:
        sys.argv.extend(['--seed', str(args.seed)])
    train_main()


def cmd_eval(args):
    """评估命令"""
    from train import main as train_main
    sys.argv = ['train', 'eval', '--config', args.config, '--num-episodes', str(args.episodes)]
    if args.checkpoint:
        sys.argv.extend(['--checkpoint', args.checkpoint])
    train_main()


def cmd_meta_train(args):
    """元学习训练命令"""
    from train_meta import main as meta_main
    sys.argv = ['train_meta', '--config', args.config, '--method', args.method]
    if args.seed:
        sys.argv.extend(['--seed', str(args.seed)])
    meta_main()


def cmd_demo(args):
    """演示命令"""
    print("Running KAS DRL Demo...")
    
    import torch
    import numpy as np
    from training.environment import KASAgentEnv
    from algorithms.ppo import PPOAgent, PPOConfig
    from core.state_space import StateSpace
    from core.action_space import ActionSpace
    from core.reward import RewardFunction
    
    # 创建环境
    env = KASAgentEnv()
    
    # 创建Agent
    agent = PPOAgent(
        state_dim=28,  # 环境观测维度
        action_dim=21,  # 动作维度
        config=PPOConfig(),
        device='cpu'
    )
    
    # 运行几个回合
    for episode in range(5):
        state = env.reset()
        episode_reward = 0
        
        for step in range(100):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(reward, next_state, done)
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step + 1}")
    
    print("\nDemo completed!")


def main():
    print_banner()
    
    parser = argparse.ArgumentParser(
        description='KAS Deep Reinforcement Learning Module',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='Train DRL agent')
    train_parser.add_argument('--config', '-c', type=str, required=True, help='Config file')
    train_parser.add_argument('--seed', type=int, help='Random seed')
    train_parser.set_defaults(func=cmd_train)
    
    # 评估命令
    eval_parser = subparsers.add_parser('eval', help='Evaluate trained agent')
    eval_parser.add_argument('--config', '-c', type=str, required=True, help='Config file')
    eval_parser.add_argument('--checkpoint', type=str, help='Model checkpoint')
    eval_parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
    eval_parser.set_defaults(func=cmd_eval)
    
    # 元学习训练命令
    meta_parser = subparsers.add_parser('meta-train', help='Train meta-learning model')
    meta_parser.add_argument('--config', '-c', type=str, required=True, help='Config file')
    meta_parser.add_argument('--method', type=str, choices=['maml', 'reptile'], default='maml')
    meta_parser.add_argument('--seed', type=int, help='Random seed')
    meta_parser.set_defaults(func=cmd_meta_train)
    
    # 演示命令
    demo_parser = subparsers.add_parser('demo', help='Run demo')
    demo_parser.set_defaults(func=cmd_demo)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == '__main__':
    main()
