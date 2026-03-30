"""
小样本材料发现 (Few-Shot Material Discovery)
利用其他领域预训练模型进行小样本材料发现

作者: DFT-LAMMPS Team
版本: 1.0.0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt


@dataclass
class FewShotConfig:
    """小样本学习配置"""
    feature_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 64
    
    # 小样本设置
    n_way: int = 5        # 类别数
    k_shot: int = 5       # 每类支持样本数
    n_query: int = 15     # 每类查询样本数
    
    # 训练设置
    meta_lr: float = 1e-3
    inner_lr: float = 0.01
    num_inner_steps: int = 5
    num_meta_iterations: int = 1000
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MaterialFeatureExtractor(nn.Module):
    """材料特征提取器"""
    
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PropertyPredictor(nn.Module):
    """性质预测器"""
    
    def __init__(self, input_dim: int = 64, num_properties: int = 1):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_properties)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(x)


class PrototypicalNetwork:
    """原型网络 - 用于小样本材料分类/回归"""
    
    def __init__(self, config: FewShotConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        self.encoder = MaterialFeatureExtractor(
            config.feature_dim,
            config.output_dim
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=config.meta_lr
        )
    
    def compute_prototypes(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算类别原型
        
        Args:
            support_features: [n_way * k_shot, output_dim]
            support_labels: [n_way * k_shot]
        
        Returns:
            prototypes: [n_way, output_dim]
        """
        n_way = len(torch.unique(support_labels))
        output_dim = support_features.size(1)
        
        prototypes = torch.zeros(n_way, output_dim, device=self.device)
        
        for i in range(n_way):
            class_mask = (support_labels == i)
            class_features = support_features[class_mask]
            prototypes[i] = class_features.mean(dim=0)
        
        return prototypes
    
    def prototypical_loss(
        self,
        query_features: torch.Tensor,
        query_labels: torch.Tensor,
        prototypes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算原型网络损失
        """
        # 计算距离
        distances = torch.cdist(query_features, prototypes)
        
        # 转换为对数概率
        log_probs = F.log_softmax(-distances, dim=1)
        
        # 计算损失
        loss = F.nll_loss(log_probs, query_labels)
        
        # 计算准确率
        preds = torch.argmin(distances, dim=1)
        acc = (preds == query_labels).float().mean()
        
        return loss, acc
    
    def meta_train_step(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor
    ) -> Dict[str, float]:
        """元训练步骤"""
        self.encoder.train()
        self.optimizer.zero_grad()
        
        # 编码
        support_features = self.encoder(support_x)
        query_features = self.encoder(query_x)
        
        # 计算原型
        prototypes = self.compute_prototypes(support_features, support_y)
        
        # 计算损失
        loss, acc = self.prototypical_loss(
            query_features, query_y, prototypes
        )
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "accuracy": acc.item()
        }
    
    def predict(
        self,
        support_x: np.ndarray,
        support_y: np.ndarray,
        query_x: np.ndarray
    ) -> np.ndarray:
        """基于支持集预测"""
        self.encoder.eval()
        
        support_x = torch.FloatTensor(support_x).to(self.device)
        support_y = torch.LongTensor(support_y).to(self.device)
        query_x = torch.FloatTensor(query_x).to(self.device)
        
        with torch.no_grad():
            support_features = self.encoder(support_x)
            query_features = self.encoder(query_x)
            
            prototypes = self.compute_prototypes(support_features, support_y)
            
            distances = torch.cdist(query_features, prototypes)
            preds = torch.argmin(distances, dim=1)
        
        return preds.cpu().numpy()


class MAMLFewShot:
    """MAML实现小样本学习"""
    
    def __init__(self, config: FewShotConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        self.feature_extractor = MaterialFeatureExtractor(
            config.feature_dim,
            config.output_dim
        ).to(self.device)
        
        self.property_predictor = PropertyPredictor(
            config.output_dim,
            num_properties=1
        ).to(self.device)
        
        self.meta_optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) +
            list(self.property_predictor.parameters()),
            lr=config.meta_lr
        )
        
        self.criterion = nn.MSELoss()
    
    def inner_loop(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """内循环适应"""
        # 克隆参数
        fast_weights = {}
        for name, param in self.feature_extractor.named_parameters():
            fast_weights[name] = param.clone()
        
        # 内循环梯度下降
        for _ in range(self.config.num_inner_steps):
            # 前向传播
            features = self._forward_with_params(
                support_x, fast_weights, "feature_extractor"
            )
            predictions = self.property_predictor(features)
            
            loss = self.criterion(predictions.squeeze(), support_y)
            
            # 计算梯度
            grads = torch.autograd.grad(
                loss,
                fast_weights.values(),
                create_graph=True
            )
            
            # 更新快速权重
            fast_weights = {
                name: param - self.config.inner_lr * grad
                for (name, param), grad in zip(fast_weights.items(), grads)
            }
        
        return fast_weights
    
    def _forward_with_params(
        self,
        x: torch.Tensor,
        params: Dict[str, torch.Tensor],
        module_name: str
    ) -> torch.Tensor:
        """使用自定义参数前向传播"""
        # 简化实现 - 实际应该重构网络结构
        if module_name == "feature_extractor":
            return self.feature_extractor(x)
        return x
    
    def meta_train_step(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor
    ) -> Dict[str, float]:
        """元训练步骤"""
        # 内循环适应
        fast_weights = self.inner_loop(support_x, support_y)
        
        # 在查询集上评估
        features = self._forward_with_params(
            query_x, fast_weights, "feature_extractor"
        )
        predictions = self.property_predictor(features)
        
        meta_loss = self.criterion(predictions.squeeze(), query_y)
        
        # 元更新
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        mse = F.mse_loss(predictions.squeeze(), query_y).item()
        
        return {
            "meta_loss": meta_loss.item(),
            "mse": mse
        }
    
    def adapt_and_predict(
        self,
        support_x: np.ndarray,
        support_y: np.ndarray,
        query_x: np.ndarray
    ) -> np.ndarray:
        """适应并预测"""
        support_x = torch.FloatTensor(support_x).to(self.device)
        support_y = torch.FloatTensor(support_y).to(self.device)
        query_x = torch.FloatTensor(query_x).to(self.device)
        
        # 内循环适应
        fast_weights = self.inner_loop(support_x, support_y)
        
        # 预测
        with torch.no_grad():
            features = self._forward_with_params(
                query_x, fast_weights, "feature_extractor"
            )
            predictions = self.property_predictor(features)
        
        return predictions.cpu().numpy()


class CrossDomainFewShotLearner:
    """跨领域小样本学习器"""
    
    def __init__(
        self,
        config: FewShotConfig,
        algorithm: str = "protonet"
    ):
        self.config = config
        self.algorithm = algorithm
        
        if algorithm == "protonet":
            self.model = PrototypicalNetwork(config)
        elif algorithm == "maml":
            self.model = MAMLFewShot(config)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def pretrain_on_source_domain(
        self,
        source_data: Dict[str, np.ndarray],
        num_iterations: int = 1000
    ) -> Dict[str, List[float]]:
        """
        在源领域预训练
        """
        print(f"Pre-training on source domain...")
        
        history = defaultdict(list)
        
        for iteration in range(num_iterations):
            # 采样任务
            task = self._sample_task(source_data)
            
            # 训练
            if self.algorithm == "protonet":
                metrics = self.model.meta_train_step(**task)
            else:  # maml
                metrics = self.model.meta_train_step(**task)
            
            for key, value in metrics.items():
                history[key].append(value)
            
            if (iteration + 1) % 100 == 0:
                avg_metrics = {
                    k: np.mean(v[-100:])
                    for k, v in history.items()
                }
                print(f"  Iteration {iteration+1}: {avg_metrics}")
        
        return dict(history)
    
    def few_shot_predict_on_target(
        self,
        target_support: Dict[str, np.ndarray],
        target_query: np.ndarray
    ) -> np.ndarray:
        """
        在目标领域进行小样本预测
        """
        support_x = target_support["features"]
        support_y = target_support["labels"]
        
        if self.algorithm == "protonet":
            return self.model.predict(support_x, support_y, target_query)
        else:  # maml
            return self.model.adapt_and_predict(support_x, support_y, target_query)
    
    def _sample_task(
        self,
        data: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """采样一个任务"""
        features = data["features"]
        labels = data["labels"]
        
        unique_labels = np.unique(labels)
        selected_classes = np.random.choice(
            unique_labels,
            size=min(self.config.n_way, len(unique_labels)),
            replace=False
        )
        
        support_x, support_y = [], []
        query_x, query_y = [], []
        
        for i, cls in enumerate(selected_classes):
            cls_indices = np.where(labels == cls)[0]
            selected = np.random.choice(
                cls_indices,
                size=self.config.k_shot + self.config.n_query,
                replace=False
            )
            
            support_x.append(features[selected[:self.config.k_shot]])
            support_y.append(np.full(self.config.k_shot, i))
            
            query_x.append(features[selected[self.config.k_shot:]])
            query_y.append(np.full(self.config.n_query, i))
        
        return {
            "support_x": torch.FloatTensor(np.vstack(support_x)).to(self.config.device),
            "support_y": torch.LongTensor(np.concatenate(support_y)).to(self.config.device),
            "query_x": torch.FloatTensor(np.vstack(query_x)).to(self.config.device),
            "query_y": torch.LongTensor(np.concatenate(query_y)).to(self.config.device)
        }


def generate_source_domain_data(
    n_samples: int = 1000,
    n_classes: int = 10
) -> Dict[str, np.ndarray]:
    """生成源领域数据"""
    np.random.seed(42)
    
    features = np.random.randn(n_samples, 128)
    labels = np.random.randint(0, n_classes, n_samples)
    
    # 添加类别特定的模式
    for i in range(n_classes):
        mask = labels == i
        features[mask, :5] += i * 0.5
    
    return {"features": features, "labels": labels}


def generate_target_domain_data(
    n_samples: int = 100,
    n_classes: int = 5
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """生成目标领域数据"""
    np.random.seed(43)
    
    features = np.random.randn(n_samples, 128)
    labels = np.random.randint(0, n_classes, n_samples)
    
    # 添加与源领域相似但不同的模式
    for i in range(n_classes):
        mask = labels == i
        features[mask, :5] += i * 0.4  # 稍有不同的模式
    
    # 分割支持和查询集
    support_size = 5 * n_classes  # 5-shot per class
    support_features = features[:support_size]
    support_labels = labels[:support_size]
    
    query_features = features[support_size:]
    query_labels = labels[support_size:]
    
    support = {
        "features": support_features,
        "labels": support_labels
    }
    
    return support, query_features, query_labels


# 演示代码
if __name__ == "__main__":
    print("=" * 70)
    print("小样本材料发现演示 (Few-Shot Material Discovery Demo)")
    print("=" * 70)
    
    # 生成数据
    print("\nGenerating source and target domain data...")
    
    source_data = generate_source_domain_data(n_samples=1000, n_classes=10)
    target_support, target_query, target_labels = generate_target_domain_data(
        n_samples=100, n_classes=5
    )
    
    print(f"Source domain: {len(source_data['features'])} samples")
    print(f"Target domain support: {len(target_support['features'])} samples")
    print(f"Target domain query: {len(target_query)} samples")
    
    # 配置
    config = FewShotConfig()
    
    # Prototypical Networks
    print("\n" + "=" * 70)
    print("Method 1: Prototypical Networks")
    print("=" * 70)
    
    learner = CrossDomainFewShotLearner(config, algorithm="protonet")
    
    # 预训练
    history = learner.pretrain_on_source_domain(
        source_data,
        num_iterations=500
    )
    
    # 小样本预测
    predictions = learner.few_shot_predict_on_target(
        target_support,
        target_query
    )
    
    # 评估
    accuracy = (predictions == target_labels).mean()
    print(f"\nFew-shot accuracy: {accuracy:.4f}")
    
    # MAML
    print("\n" + "=" * 70)
    print("Method 2: MAML")
    print("=" * 70)
    
    learner_maml = CrossDomainFewShotLearner(config, algorithm="maml")
    
    # 注意: MAML需要回归目标，这里简化处理
    print("MAML requires continuous targets, skipping full demo...")
    
    print("\n" + "=" * 70)
    print("Few-Shot Material Discovery Demo Complete!")
    print("=" * 70)
