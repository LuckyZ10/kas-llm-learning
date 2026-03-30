"""
KAS Meta-Learning Training Script
元学习训练脚本
"""
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from meta_learning.maml import MAML, ProjectTaskSampler, MAMLAgentAdapter
from meta_learning.reptile import Reptile, ReptileAgentPolicy
from meta_learning.encoders import (
    ProjectFeatureEncoder, 
    TelemetryLSTMEncoder,
    MultiModalProjectEncoder,
    ProjectEmbeddingStore
)
from training.trainer import MetaTrainer, TrainingConfig


def create_project_embeddings(num_projects: int, feature_dim: int) -> tuple:
    """创建模拟项目嵌入"""
    # 生成项目特征
    embeddings = torch.randn(num_projects, feature_dim)
    
    # 生成项目标签（例如项目类型、复杂度等）
    labels = torch.randn(num_projects, 64)
    
    return embeddings, labels


def train_maml(config: dict, device: str):
    """训练MAML"""
    print("Training MAML meta-learner...")
    
    # 创建模拟数据
    num_projects = config['project_data']['num_projects']
    feature_dim = config['encoder']['output_dim']
    
    embeddings, labels = create_project_embeddings(num_projects, feature_dim)
    embeddings = embeddings.to(device)
    labels = labels.to(device)
    
    # 创建任务采样器
    task_sampler = ProjectTaskSampler(
        embeddings,
        labels,
        k_shot=config['meta_learning']['k_shot'],
        q_query=config['meta_learning']['q_query']
    )
    
    # 创建基础模型（这里用一个简单的MLP作为示例）
    from algorithms.ppo import ActorNetwork
    base_model = ActorNetwork(
        state_dim=config['encoder']['output_dim'],
        action_dim=21,  # 动作空间维度
        hidden_dim=config['meta_learning'].get('hidden_dim', 256)
    ).to(device)
    
    # 创建MAML
    from meta_learning.maml import MAMLConfig
    maml_config = MAMLConfig(
        inner_lr=config['meta_learning']['inner_lr'],
        outer_lr=config['meta_learning']['outer_lr'],
        num_inner_steps=config['meta_learning']['num_inner_steps'],
        num_tasks_per_batch=config['meta_learning']['num_tasks_per_batch'],
        first_order=config['meta_learning']['first_order']
    )
    
    maml = MAML(base_model, maml_config)
    
    # 创建训练配置
    train_config = TrainingConfig(
        log_dir=config['training'].get('log_dir', './logs'),
        experiment_name=config['training'].get('experiment_name', 'kas_maml'),
        device=device
    )
    
    # 创建训练器
    trainer = MetaTrainer(maml, task_sampler, train_config)
    
    # 训练
    num_iterations = config['meta_learning'].get('num_meta_iterations', 1000)
    history = trainer.train(num_iterations)
    
    # 保存模型
    save_path = Path(train_config.log_dir) / train_config.experiment_name / 'maml_model.pt'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': maml.model.state_dict(),
        'config': config,
        'history': history
    }, save_path)
    
    print(f"MAML model saved to {save_path}")
    print(f"Final task loss: {history['task_losses'][-1]:.4f}")
    
    return maml, history


def train_reptile(config: dict, device: str):
    """训练Reptile"""
    print("Training Reptile meta-learner...")
    
    # 创建模拟数据
    num_projects = config['project_data']['num_projects']
    feature_dim = config['encoder']['output_dim']
    
    embeddings, labels = create_project_embeddings(num_projects, feature_dim)
    embeddings = embeddings.to(device)
    labels = labels.to(device)
    
    # 创建任务采样器
    task_sampler = ProjectTaskSampler(
        embeddings,
        labels,
        k_shot=config['meta_learning']['k_shot'],
        q_query=config['meta_learning']['q_query']
    )
    
    # 创建基础模型
    from algorithms.ppo import ActorNetwork
    base_model = ActorNetwork(
        state_dim=config['encoder']['output_dim'],
        action_dim=21,
        hidden_dim=config['meta_learning'].get('hidden_dim', 256)
    ).to(device)
    
    # 创建Reptile
    from meta_learning.reptile import ReptileConfig
    reptile_config = ReptileConfig(
        inner_lr=config['meta_learning']['inner_lr'],
        meta_lr=config['meta_learning']['outer_lr'],
        inner_steps=config['meta_learning']['num_inner_steps'],
        meta_batch_size=config['meta_learning']['num_tasks_per_batch']
    )
    
    reptile = Reptile(base_model, reptile_config)
    
    # 训练
    num_iterations = config['meta_learning'].get('num_meta_iterations', 1000)
    history = reptile.train(task_sampler, num_iterations)
    
    # 保存
    train_config = TrainingConfig(
        log_dir=config['training'].get('log_dir', './logs'),
        experiment_name=config['training'].get('experiment_name', 'kas_reptile')
    )
    save_path = Path(train_config.log_dir) / train_config.experiment_name / 'reptile_model.pt'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': reptile.model.state_dict(),
        'config': config,
        'history': history
    }, save_path)
    
    print(f"Reptile model saved to {save_path}")
    print(f"Final task loss: {history['task_losses'][-1]:.4f}")
    
    return reptile, history


def main():
    parser = argparse.ArgumentParser(description='KAS Meta-Learning Training')
    parser.add_argument('--config', '-c', type=str, required=True, help='Config file path')
    parser.add_argument('--method', type=str, choices=['maml', 'reptile'], default='maml',
                        help='Meta-learning method')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = config['training'].get('device', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Using device: {device}")
    print(f"Meta-learning method: {args.method}")
    
    # 训练
    if args.method == 'maml':
        train_maml(config, device)
    else:
        train_reptile(config, device)


if __name__ == '__main__':
    main()
