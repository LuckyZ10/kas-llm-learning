"""
持续学习演示 (Continual Learning Demo)
演示如何不断积累跨领域知识

作者: DFT-LAMMPS Team
版本: 1.0.0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from collections import deque
import copy


@dataclass
class ContinualLearningConfig:
    """持续学习配置"""
    feature_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 64
    
    # 回放缓冲区
    buffer_size: int = 1000
    replay_ratio: float = 0.3
    
    # EWC正则化
    ewc_lambda: float = 100.0
    fisher_sample_size: int = 200
    
    # 渐进式神经网络
    use_progressive: bool = True
    
    learning_rate: float = 1e-3
    num_epochs_per_domain: int = 50
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleMaterialNetwork(nn.Module):
    """简单材料网络"""
    
    def __init__(self, input_dim: int = 128, output_dim: int = 64):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, 192),
            nn.ReLU(),
            nn.BatchNorm1d(192),
            nn.Dropout(0.2),
            
            nn.Linear(192, output_dim),
            nn.ReLU()
        )
        
        self.classifier = nn.Linear(output_dim, 10)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.network(x)
        logits = self.classifier(features)
        return features, logits


class ExperienceReplay:
    """经验回放 - 缓解灾难性遗忘"""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, features: np.ndarray, labels: np.ndarray):
        """添加样本到缓冲区"""
        for feat, label in zip(features, labels):
            self.buffer.append((feat, label))
    
    def sample(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """从缓冲区采样"""
        if len(self.buffer) == 0:
            return np.array([]), np.array([])
        
        n_samples = min(n_samples, len(self.buffer))
        indices = np.random.choice(len(self.buffer), n_samples, replace=False)
        
        sampled_features = []
        sampled_labels = []
        
        for idx in indices:
            feat, label = self.buffer[idx]
            sampled_features.append(feat)
            sampled_labels.append(label)
        
        return np.array(sampled_features), np.array(sampled_labels)
    
    def __len__(self) -> int:
        return len(self.buffer)


class EWCRegularizer:
    """EWC (Elastic Weight Consolidation) 正则化器"""
    
    def __init__(self, model: nn.Module, importance: float = 100.0):
        self.model = model
        self.importance = importance
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.means = {}
        self.fisher = {}
    
    def update_fisher(self, dataloader: torch.utils.data.DataLoader):
        """更新Fisher信息矩阵"""
        self.model.train()
        
        # 初始化Fisher
        for name, param in self.params.items():
            self.fisher[name] = torch.zeros_like(param)
        
        # 计算Fisher信息
        for features, labels in dataloader:
            features = features.to(next(self.model.parameters()).device)
            labels = labels.to(next(self.model.parameters()).device)
            
            self.model.zero_grad()
            _, logits = self.model(features)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher[name] += param.grad.data ** 2
        
        # 平均
        for name in self.fisher:
            self.fisher[name] /= len(dataloader)
        
        # 保存当前参数
        for name, param in self.params.items():
            self.means[name] = param.data.clone()
    
    def penalty(self, model: nn.Module) -> torch.Tensor:
        """计算EWC惩罚项"""
        loss = 0.0
        
        for name, param in model.named_parameters():
            if name in self.means and name in self.fisher:
                loss += (self.fisher[name] * (param - self.means[name]) ** 2).sum()
        
        return self.importance * loss


class ProgressiveNeuralNetwork:
    """渐进式神经网络 - 为每个任务添加新列"""
    
    def __init__(self, config: ContinualLearningConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        self.columns = []
        self.adapters = []
        self.task_count = 0
    
    def add_column(self):
        """为新任务添加新列"""
        column = SimpleMaterialNetwork(
            self.config.feature_dim,
            self.config.output_dim
        ).to(self.device)
        
        # 冻结之前的列
        for prev_column in self.columns:
            for param in prev_column.parameters():
                param.requires_grad = False
        
        self.columns.append(column)
        self.task_count += 1
        
        return column
    
    def forward(
        self,
        x: torch.Tensor,
        task_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 横向连接
        all_features = []
        
        for i, column in enumerate(self.columns):
            if i <= task_id:
                feat, _ = column(x)
                all_features.append(feat)
        
        # 合并特征
        if len(all_features) > 1:
            combined = torch.cat(all_features, dim=1)
            # 使用当前列的分类器
            logits = self.columns[task_id].classifier(
                all_features[-1][:, :self.config.output_dim]
            )
        else:
            combined = all_features[0]
            logits = self.columns[task_id].classifier(combined)
        
        return combined, logits


class ContinualMaterialLearner:
    """持续材料学习器"""
    
    def __init__(self, config: ContinualLearningConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 当前模型
        self.model = SimpleMaterialNetwork(
            config.feature_dim,
            config.output_dim
        ).to(self.device)
        
        # 经验回放
        self.replay_buffer = ExperienceReplay(config.buffer_size)
        
        # EWC
        self.ewc = None
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        # 任务历史
        self.task_history = []
        self.current_task = 0
    
    def learn_new_domain(
        self,
        domain_name: str,
        features: np.ndarray,
        labels: np.ndarray,
        use_ewc: bool = True,
        use_replay: bool = True
    ) -> Dict[str, List[float]]:
        """
        学习新领域
        
        Args:
            domain_name: 领域名称
            features: 特征
            labels: 标签
            use_ewc: 是否使用EWC
            use_replay: 是否使用经验回放
        """
        print(f"\nLearning new domain: {domain_name}")
        
        # 更新EWC (如果启用)
        if use_ewc and self.current_task > 0:
            print("  Updating EWC...")
            self.ewc = EWCRegularizer(self.model, self.config.ewc_lambda)
            
            # 用之前的数据计算Fisher
            if len(self.replay_buffer) > 0:
                replay_feats, replay_labels = self.replay_buffer.sample(
                    min(self.config.fisher_sample_size, len(self.replay_buffer))
                )
                
                replay_dataset = torch.utils.data.TensorDataset(
                    torch.FloatTensor(replay_feats),
                    torch.LongTensor(replay_labels)
                )
                replay_loader = torch.utils.data.DataLoader(
                    replay_dataset,
                    batch_size=32
                )
                
                self.ewc.update_fisher(replay_loader)
        
        # 训练
        history = {"loss": [], "accuracy": []}
        
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(features),
            torch.LongTensor(labels)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=True
        )
        
        for epoch in range(self.config.num_epochs_per_domain):
            self.model.train()
            
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_features, batch_labels in dataloader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # 前向传播
                _, logits = self.model(batch_features)
                loss = F.cross_entropy(logits, batch_labels)
                
                # EWC惩罚
                if use_ewc and self.ewc is not None:
                    loss += self.ewc.penalty(self.model)
                
                # 经验回放
                if use_replay and len(self.replay_buffer) > 0:
                    replay_feats, replay_labels = self.replay_buffer.sample(
                        int(len(batch_features) * self.config.replay_ratio)
                    )
                    
                    if len(replay_feats) > 0:
                        replay_feats = torch.FloatTensor(replay_feats).to(self.device)
                        replay_labels = torch.LongTensor(replay_labels).to(self.device)
                        
                        _, replay_logits = self.model(replay_feats)
                        replay_loss = F.cross_entropy(replay_logits, replay_labels)
                        
                        loss += replay_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 统计
                epoch_loss += loss.item()
                _, predicted = logits.max(1)
                epoch_total += batch_labels.size(0)
                epoch_correct += predicted.eq(batch_labels).sum().item()
            
            avg_loss = epoch_loss / len(dataloader)
            accuracy = 100. * epoch_correct / epoch_total
            
            history["loss"].append(avg_loss)
            history["accuracy"].append(accuracy)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")
        
        # 添加到回放缓冲区
        self.replay_buffer.add(features, labels)
        
        # 记录任务
        self.task_history.append({
            "task_id": self.current_task,
            "domain": domain_name,
            "n_samples": len(features)
        })
        
        self.current_task += 1
        
        return history
    
    def evaluate_on_all_domains(
        self,
        domain_data: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, float]:
        """
        评估在所有领域上的性能
        
        检查灾难性遗忘程度
        """
        self.model.eval()
        
        results = {}
        
        print("\nEvaluating on all domains:")
        
        for domain_name, (features, labels) in domain_data.items():
            features_tensor = torch.FloatTensor(features).to(self.device)
            labels_tensor = torch.LongTensor(labels).to(self.device)
            
            with torch.no_grad():
                _, logits = self.model(features_tensor)
                _, predicted = logits.max(1)
                accuracy = (predicted == labels_tensor).float().mean().item()
            
            results[domain_name] = accuracy
            print(f"  {domain_name}: {accuracy:.4f}")
        
        return results
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """获取知识摘要"""
        return {
            "num_tasks_learned": self.current_task,
            "task_history": self.task_history,
            "replay_buffer_size": len(self.replay_buffer),
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "ewc_enabled": self.ewc is not None
        }


def generate_domain_data(
    domain_name: str,
    n_samples: int = 500,
    n_classes: int = 10,
    feature_dim: int = 128
) -> Tuple[np.ndarray, np.ndarray]:
    """生成领域数据"""
    np.random.seed(hash(domain_name) % 2**32)
    
    features = np.random.randn(n_samples, feature_dim)
    labels = np.random.randint(0, n_classes, n_samples)
    
    # 添加领域特定的模式
    domain_offset = hash(domain_name) % 10
    features[:, 0] += domain_offset * 0.3
    
    return features, labels


# 演示代码
if __name__ == "__main__":
    print("=" * 70)
    print("持续学习演示 (Continual Learning Demo)")
    print("=" * 70)
    
    # 定义一系列材料领域
    domains = [
        "battery_cathode",
        "battery_anode",
        "catalyst_oxide",
        "catalyst_metal",
        "semiconductor_2d",
        "metal_alloy",
        "ceramic_oxide"
    ]
    
    print(f"\nWill learn {len(domains)} domains sequentially:")
    for i, domain in enumerate(domains, 1):
        print(f"  {i}. {domain}")
    
    # 配置
    config = ContinualLearningConfig()
    learner = ContinualMaterialLearner(config)
    
    # 存储所有领域数据用于评估
    all_domain_data = {}
    
    # 顺序学习每个领域
    print("\n" + "=" * 70)
    print("Sequential Learning of Domains")
    print("=" * 70)
    
    for i, domain in enumerate(domains):
        print(f"\n{'='*70}")
        print(f"Task {i+1}/{len(domains)}: {domain}")
        print('='*70)
        
        # 生成数据
        features, labels = generate_domain_data(domain)
        all_domain_data[domain] = (features, labels)
        
        # 学习
        history = learner.learn_new_domain(
            domain,
            features,
            labels,
            use_ewc=True,
            use_replay=True
        )
        
        # 每学2个领域评估一次
        if (i + 1) % 2 == 0:
            print(f"\nIntermediate evaluation after learning {i+1} domains:")
            eval_results = learner.evaluate_on_all_domains(all_domain_data)
    
    # 最终评估
    print("\n" + "=" * 70)
    print("Final Evaluation - Checking for Catastrophic Forgetting")
    print("=" * 70)
    
    final_results = learner.evaluate_on_all_domains(all_domain_data)
    
    # 计算平均性能
    avg_performance = np.mean(list(final_results.values()))
    print(f"\nAverage performance across all domains: {avg_performance:.4f}")
    
    # 知识摘要
    print("\n" + "=" * 70)
    print("Knowledge Summary")
    print("=" * 70)
    
    summary = learner.get_knowledge_summary()
    
    for key, value in summary.items():
        if isinstance(value, list):
            print(f"{key}:")
            for item in value:
                print(f"  {item}")
        else:
            print(f"{key}: {value}")
    
    print("\n" + "=" * 70)
    print("Continual Learning Demo Complete!")
    print("=" * 70)
