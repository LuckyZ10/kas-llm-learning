"""
元材料学习 (Meta-Material Learning)
实现跨领域元学习算法，包括MAML、Prototypical Networks等

作者: DFT-LAMMPS Team
版本: 1.0.0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass
from collections import OrderedDict
import copy
from abc import ABC, abstractmethod
import warnings


@dataclass
class MetaLearningConfig:
    """元学习配置"""
    algorithm: str = "maml"  # maml, protonet, metasgd, reptile
    input_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 64
    num_inner_steps: int = 5
    inner_lr: float = 0.01
    meta_lr: float = 0.001
    meta_batch_size: int = 4  # 每批任务数
    num_shots: int = 5  # K-shot learning
    num_queries: int = 15  # 查询样本数
    num_classes_per_task: int = 5  # N-way
    first_order: bool = False  # 是否使用一阶MAML
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Reptile specific
    reptile_beta: float = 0.5
    
    # Prototypical Networks specific
    protonet_distance: str = "euclidean"  # euclidean, cosine


class MaterialEmbeddingNetwork(nn.Module):
    """材料嵌入网络 - 用于学习材料表示"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 64,
        num_layers: int = 4
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            in_dim = hidden_dim
            if i < num_layers - 1:
                hidden_dim = max(hidden_dim // 2, output_dim)
        
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MaterialPropertyPredictor(nn.Module):
    """材料性质预测器"""
    
    def __init__(
        self,
        embedding_dim: int,
        num_properties: int = 1,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_properties)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(x)


class MAMLModel(nn.Module):
    """MAML模型 - 包含嵌入网络和预测器"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_classes: int = 1
    ):
        super().__init__()
        
        self.embedding = MaterialEmbeddingNetwork(
            input_dim, hidden_dim, output_dim
        )
        self.predictor = MaterialPropertyPredictor(
            output_dim, num_classes
        )
    
    def forward(
        self,
        x: torch.Tensor,
        params: Optional[OrderedDict] = None
    ) -> torch.Tensor:
        """前向传播，支持自定义参数"""
        if params is None:
            embedding = self.embedding(x)
            output = self.predictor(embedding)
        else:
            # 使用提供的参数
            embedding = self._forward_with_params(
                x, params, "embedding"
            )
            output = self._forward_with_params(
                embedding, params, "predictor"
            )
        
        return output
    
    def _forward_with_params(
        self,
        x: torch.Tensor,
        params: OrderedDict,
        module_name: str
    ) -> torch.Tensor:
        """使用指定参数前向传播"""
        # 简化的前向传播实现
        module = getattr(self, module_name)
        return module(x)
    
    def cloned_state_dict(self) -> OrderedDict:
        """克隆模型状态"""
        return OrderedDict(
            (name, param.clone())
            for name, param in self.named_parameters()
        )


class MAML:
    """Model-Agnostic Meta-Learning for Material Discovery"""
    
    def __init__(self, config: MetaLearningConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        self.model = MAMLModel(
            config.input_dim,
            config.hidden_dim,
            config.output_dim,
            config.num_classes_per_task
        ).to(self.device)
        
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.meta_lr
        )
        
        self.criterion = nn.MSELoss()
    
    def inner_loop(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        create_graph: bool = False
    ) -> OrderedDict:
        """内循环适应"""
        # 克隆当前参数
        fast_weights = OrderedDict(
            (name, param.clone())
            for name, param in self.model.named_parameters()
        )
        
        for step in range(self.config.num_inner_steps):
            # 使用当前快速权重前向传播
            support_pred = self._forward_with_params(support_x, fast_weights)
            loss = self.criterion(support_pred.squeeze(), support_y)
            
            # 计算梯度
            grads = torch.autograd.grad(
                loss,
                fast_weights.values(),
                create_graph=create_graph
            )
            
            # 更新快速权重
            fast_weights = OrderedDict(
                (name, param - self.config.inner_lr * grad)
                for (name, param), grad in zip(fast_weights.items(), grads)
            )
        
        return fast_weights
    
    def _forward_with_params(
        self,
        x: torch.Tensor,
        params: OrderedDict
    ) -> torch.Tensor:
        """使用自定义参数前向传播"""
        # 保存原始参数
        original_params = {}
        for name, param in self.model.named_parameters():
            original_params[name] = param.data.clone()
            param.data = params[name]
        
        # 前向传播
        output = self.model(x)
        
        # 恢复原始参数
        for name, param in self.model.named_parameters():
            param.data = original_params[name]
        
        return output
    
    def meta_train_step(
        self,
        batch_tasks: List[Tuple[torch.Tensor, torch.Tensor,
                                 torch.Tensor, torch.Tensor]]
    ) -> Dict[str, float]:
        """元训练步骤"""
        self.meta_optimizer.zero_grad()
        
        meta_loss = 0.0
        meta_mse = 0.0
        
        for support_x, support_y, query_x, query_y in batch_tasks:
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)
            
            # 内循环适应
            create_graph = not self.config.first_order
            fast_weights = self.inner_loop(
                support_x, support_y, create_graph=create_graph
            )
            
            # 在查询集上评估
            query_pred = self._forward_with_params(query_x, fast_weights)
            loss = self.criterion(query_pred.squeeze(), query_y)
            
            meta_loss += loss
            meta_mse += F.mse_loss(query_pred.squeeze(), query_y).item()
        
        # 元更新
        meta_loss = meta_loss / len(batch_tasks)
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return {
            "meta_loss": meta_loss.item(),
            "meta_mse": meta_mse / len(batch_tasks)
        }
    
    def adapt_to_new_task(
        self,
        support_x: np.ndarray,
        support_y: np.ndarray,
        num_steps: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """适应到新任务"""
        if num_steps is None:
            num_steps = self.config.num_inner_steps
        
        support_x = torch.FloatTensor(support_x).to(self.device)
        support_y = torch.FloatTensor(support_y).to(self.device)
        
        # 克隆模型
        adapted_params = OrderedDict(
            (name, param.clone())
            for name, param in self.model.named_parameters()
        )
        
        # 内循环适应
        for _ in range(num_steps):
            pred = self._forward_with_params(support_x, adapted_params)
            loss = self.criterion(pred.squeeze(), support_y)
            
            grads = torch.autograd.grad(
                loss,
                adapted_params.values(),
                create_graph=False
            )
            
            adapted_params = OrderedDict(
                (name, param - self.config.inner_lr * grad)
                for (name, param), grad in zip(adapted_params.items(), grads)
            )
        
        return adapted_params
    
    def predict(
        self,
        x: np.ndarray,
        adapted_params: Optional[Dict[str, torch.Tensor]] = None
    ) -> np.ndarray:
        """预测"""
        x = torch.FloatTensor(x).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            if adapted_params is not None:
                pred = self._forward_with_params(x, adapted_params)
            else:
                pred = self.model(x)
        
        return pred.cpu().numpy()


class PrototypicalNetworks:
    """Prototypical Networks for Material Few-Shot Learning"""
    
    def __init__(self, config: MetaLearningConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        self.encoder = MaterialEmbeddingNetwork(
            config.input_dim,
            config.hidden_dim,
            config.output_dim
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=config.meta_lr
        )
    
    def compute_prototypes(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor
    ) -> torch.Tensor:
        """计算类别原型"""
        num_classes = len(torch.unique(support_labels))
        embedding_dim = support_embeddings.size(1)
        
        prototypes = torch.zeros(
            num_classes, embedding_dim,
            device=self.device
        )
        
        for c in range(num_classes):
            class_mask = (support_labels == c)
            class_embeddings = support_embeddings[class_mask]
            prototypes[c] = class_embeddings.mean(dim=0)
        
        return prototypes
    
    def prototypical_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """原型网络损失"""
        if self.config.protonet_distance == "euclidean":
            distances = torch.cdist(embeddings, prototypes)
        else:  # cosine
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            prototypes_norm = F.normalize(prototypes, p=2, dim=1)
            distances = 1 - torch.mm(embeddings_norm, prototypes_norm.t())
        
        log_probs = F.log_softmax(-distances, dim=1)
        loss = F.nll_loss(log_probs, labels)
        
        return loss, -distances
    
    def train_step(
        self,
        batch_tasks: List[Tuple[torch.Tensor, torch.Tensor,
                                 torch.Tensor, torch.Tensor]]
    ) -> Dict[str, float]:
        """训练步骤"""
        self.encoder.train()
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        total_acc = 0.0
        
        for support_x, support_y, query_x, query_y in batch_tasks:
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)
            
            # 编码
            support_embeddings = self.encoder(support_x)
            query_embeddings = self.encoder(query_x)
            
            # 计算原型
            prototypes = self.compute_prototypes(
                support_embeddings, support_y
            )
            
            # 计算损失
            loss, distances = self.prototypical_loss(
                query_embeddings, query_y, prototypes
            )
            
            # 计算准确率
            preds = torch.argmin(distances, dim=1)
            acc = (preds == query_y).float().mean()
            
            total_loss += loss
            total_acc += acc.item()
        
        total_loss = total_loss / len(batch_tasks)
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "loss": total_loss.item(),
            "accuracy": total_acc / len(batch_tasks)
        }
    
    def predict(
        self,
        support_x: np.ndarray,
        support_y: np.ndarray,
        query_x: np.ndarray
    ) -> np.ndarray:
        """基于支持集预测查询集"""
        self.encoder.eval()
        
        support_x = torch.FloatTensor(support_x).to(self.device)
        support_y = torch.LongTensor(support_y).to(self.device)
        query_x = torch.FloatTensor(query_x).to(self.device)
        
        with torch.no_grad():
            support_embeddings = self.encoder(support_x)
            query_embeddings = self.encoder(query_x)
            
            prototypes = self.compute_prototypes(
                support_embeddings, support_y
            )
            
            distances = torch.cdist(query_embeddings, prototypes)
            preds = torch.argmin(distances, dim=1)
        
        return preds.cpu().numpy()


class MetaSGD:
    """Meta-SGD: 学习学习率和初始化"""
    
    def __init__(self, config: MetaLearningConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        self.model = MAMLModel(
            config.input_dim,
            config.hidden_dim,
            config.output_dim
        ).to(self.device)
        
        # 可学习的内循环学习率
        self.alpha = OrderedDict(
            (name, nn.Parameter(
                torch.ones_like(param) * config.inner_lr
            ))
            for name, param in self.model.named_parameters()
        )
        
        # 注册学习率参数
        for name, param in self.alpha.items():
            self.model.register_parameter(f"alpha_{name}", param)
        
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.meta_lr
        )
        
        self.criterion = nn.MSELoss()
    
    def inner_loop(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor
    ) -> OrderedDict:
        """内循环 - 使用可学习的学习率"""
        fast_weights = OrderedDict(
            (name, param.clone())
            for name, param in self.model.named_parameters()
            if not name.startswith("alpha_")
        )
        
        for step in range(self.config.num_inner_steps):
            pred = self._forward_with_params(support_x, fast_weights)
            loss = self.criterion(pred.squeeze(), support_y)
            
            grads = torch.autograd.grad(
                loss,
                fast_weights.values(),
                create_graph=True
            )
            
            # 使用可学习的学习率
            fast_weights = OrderedDict(
                (name, param - self.alpha[name] * grad)
                for (name, param), grad in zip(fast_weights.items(), grads)
            )
        
        return fast_weights
    
    def _forward_with_params(
        self,
        x: torch.Tensor,
        params: OrderedDict
    ) -> torch.Tensor:
        """使用自定义参数前向传播"""
        original_params = {}
        for name, param in self.model.named_parameters():
            if not name.startswith("alpha_"):
                original_params[name] = param.data.clone()
                param.data = params[name]
        
        output = self.model(x)
        
        for name in original_params:
            getattr(self.model, name).data = original_params[name]
        
        return output
    
    def meta_train_step(
        self,
        batch_tasks: List[Tuple]
    ) -> Dict[str, float]:
        """元训练步骤"""
        self.meta_optimizer.zero_grad()
        
        meta_loss = 0.0
        
        for support_x, support_y, query_x, query_y in batch_tasks:
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)
            
            fast_weights = self.inner_loop(support_x, support_y)
            
            query_pred = self._forward_with_params(query_x, fast_weights)
            loss = self.criterion(query_pred.squeeze(), query_y)
            
            meta_loss += loss
        
        meta_loss = meta_loss / len(batch_tasks)
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return {"meta_loss": meta_loss.item()}


class Reptile:
    """Reptile: 更简单的元学习算法"""
    
    def __init__(self, config: MetaLearningConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        self.model = MAMLModel(
            config.input_dim,
            config.hidden_dim,
            config.output_dim
        ).to(self.device)
        
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.meta_lr
        )
        
        self.criterion = nn.MSELoss()
    
    def inner_loop(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor
    ) -> OrderedDict:
        """内循环SGD"""
        # 创建任务特定的副本
        task_model = copy.deepcopy(self.model)
        task_optimizer = torch.optim.SGD(
            task_model.parameters(),
            lr=self.config.inner_lr
        )
        
        for _ in range(self.config.num_inner_steps):
            pred = task_model(support_x).squeeze()
            loss = self.criterion(pred, support_y)
            
            task_optimizer.zero_grad()
            loss.backward()
            task_optimizer.step()
        
        # 返回更新后的参数
        return OrderedDict(
            (name, param.clone())
            for name, param in task_model.named_parameters()
        )
    
    def meta_train_step(
        self,
        batch_tasks: List[Tuple]
    ) -> Dict[str, float]:
        """Reptile元训练步骤"""
        # 保存初始权重
        initial_weights = OrderedDict(
            (name, param.clone())
            for name, param in self.model.named_parameters()
        )
        
        task_weights = []
        
        for support_x, support_y, _, _ in batch_tasks:
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            
            # 在任务上训练
            updated_weights = self.inner_loop(support_x, support_y)
            task_weights.append(updated_weights)
        
        # 计算加权平均
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                task_updates = torch.stack([
                    weights[name] for weights in task_weights
                ])
                avg_update = task_updates.mean(dim=0)
                
                # Reptile更新
                param.data = initial_weights[name] + \
                           self.config.reptile_beta * (avg_update - initial_weights[name])
        
        return {"meta_loss": 0.0}  # Reptile没有明确的元损失


class MaterialTaskSampler:
    """材料任务采样器"""
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        domains: Optional[np.ndarray] = None,
        config: Optional[MetaLearningConfig] = None
    ):
        self.features = features
        self.labels = labels
        self.domains = domains
        self.config = config or MetaLearningConfig()
    
    def sample_task(
        self,
        num_classes: Optional[int] = None,
        num_shots: Optional[int] = None,
        num_queries: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """采样一个任务"""
        if num_classes is None:
            num_classes = self.config.num_classes_per_task
        if num_shots is None:
            num_shots = self.config.num_shots
        if num_queries is None:
            num_queries = self.config.num_queries
        
        unique_labels = np.unique(self.labels)
        selected_classes = np.random.choice(
            unique_labels,
            size=min(num_classes, len(unique_labels)),
            replace=False
        )
        
        support_x, support_y = [], []
        query_x, query_y = [], []
        
        for i, cls in enumerate(selected_classes):
            cls_indices = np.where(self.labels == cls)[0]
            
            if len(cls_indices) < num_shots + num_queries:
                continue
            
            selected = np.random.choice(
                cls_indices,
                size=num_shots + num_queries,
                replace=False
            )
            
            support_x.append(self.features[selected[:num_shots]])
            support_y.append(np.full(num_shots, i))
            
            query_x.append(self.features[selected[num_shots:]])
            query_y.append(np.full(num_queries, i))
        
        return (
            np.vstack(support_x),
            np.concatenate(support_y),
            np.vstack(query_x),
            np.concatenate(query_y)
        )
    
    def sample_batch(
        self,
        batch_size: Optional[int] = None
    ) -> List[Tuple[torch.Tensor, torch.Tensor,
                    torch.Tensor, torch.Tensor]]:
        """采样一批任务"""
        if batch_size is None:
            batch_size = self.config.meta_batch_size
        
        batch = []
        for _ in range(batch_size):
            s_x, s_y, q_x, q_y = self.sample_task()
            batch.append((
                torch.FloatTensor(s_x),
                torch.FloatTensor(s_y),
                torch.FloatTensor(q_x),
                torch.FloatTensor(q_y)
            ))
        
        return batch


class CrossDomainMetaLearner:
    """跨领域元学习器"""
    
    def __init__(
        self,
        config: MetaLearningConfig,
        algorithm: str = "maml"
    ):
        self.config = config
        self.algorithm = algorithm
        
        if algorithm == "maml":
            self.learner = MAML(config)
        elif algorithm == "protonet":
            self.learner = PrototypicalNetworks(config)
        elif algorithm == "metasgd":
            self.learner = MetaSGD(config)
        elif algorithm == "reptile":
            self.learner = Reptile(config)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def train(
        self,
        task_sampler: MaterialTaskSampler,
        num_iterations: int = 1000,
        log_interval: int = 100
    ) -> Dict[str, List[float]]:
        """训练元学习器"""
        history = {"loss": [], "accuracy": []}
        
        for iteration in range(num_iterations):
            batch = task_sampler.sample_batch()
            
            if self.algorithm == "protonet":
                metrics = self.learner.train_step(batch)
            else:
                metrics = self.learner.meta_train_step(batch)
            
            history["loss"].append(metrics.get("meta_loss", metrics.get("loss", 0)))
            if "accuracy" in metrics:
                history["accuracy"].append(metrics["accuracy"])
            
            if (iteration + 1) % log_interval == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}: {metrics}")
        
        return history
    
    def few_shot_predict(
        self,
        support_x: np.ndarray,
        support_y: np.ndarray,
        query_x: np.ndarray
    ) -> np.ndarray:
        """小样本预测"""
        if self.algorithm == "protonet":
            return self.learner.predict(support_x, support_y, query_x)
        else:
            # MAML/MetaSGD/Reptile需要适应
            adapted_params = self.learner.adapt_to_new_task(
                support_x, support_y
            )
            return self.learner.predict(query_x, adapted_params)


def evaluate_meta_learning(
    meta_learner: CrossDomainMetaLearner,
    test_sampler: MaterialTaskSampler,
    num_tasks: int = 100
) -> Dict[str, float]:
    """评估元学习器性能"""
    mse_scores = []
    mae_scores = []
    
    for _ in range(num_tasks):
        s_x, s_y, q_x, q_y = test_sampler.sample_task()
        
        predictions = meta_learner.few_shot_predict(s_x, s_y, q_x)
        
        mse = np.mean((predictions.squeeze() - q_y) ** 2)
        mae = np.mean(np.abs(predictions.squeeze() - q_y))
        
        mse_scores.append(mse)
        mae_scores.append(mae)
    
    return {
        "mse_mean": np.mean(mse_scores),
        "mse_std": np.std(mse_scores),
        "mae_mean": np.mean(mae_scores),
        "mae_std": np.std(mae_scores)
    }


# 演示代码
if __name__ == "__main__":
    print("=" * 60)
    print("元材料学习演示 (Meta-Material Learning Demo)")
    print("=" * 60)
    
    # 生成模拟材料数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # 模拟多个材料类别的特征和性质
    features = np.random.randn(n_samples, n_features)
    # 模拟带噪声的带隙
    labels = np.random.randn(n_samples) * 2 + 3.0  # eV
    labels += 0.1 * features[:, 0] + 0.05 * features[:, 1]
    
    print(f"\nDataset: {n_samples} samples, {n_features} features")
    print(f"Label range: [{labels.min():.2f}, {labels.max():.2f}] eV")
    
    # 创建配置
    config = MetaLearningConfig(
        algorithm="maml",
        input_dim=n_features,
        hidden_dim=128,
        output_dim=32,
        num_shots=5,
        num_queries=15,
        num_inner_steps=5,
        inner_lr=0.01,
        meta_lr=0.001,
        meta_batch_size=4
    )
    
    # 创建任务采样器
    print("\nCreating task sampler...")
    task_sampler = MaterialTaskSampler(features, labels, config=config)
    
    # 创建元学习器
    print(f"\nInitializing {config.algorithm.upper()} learner...")
    meta_learner = CrossDomainMetaLearner(config, algorithm=config.algorithm)
    
    # 训练
    print("\nTraining meta-learner...")
    history = meta_learner.train(
        task_sampler,
        num_iterations=500,
        log_interval=100
    )
    
    # 测试小样本学习
    print("\n" + "=" * 60)
    print("Testing Few-Shot Learning Performance")
    print("=" * 60)
    
    test_results = evaluate_meta_learning(
        meta_learner,
        task_sampler,
        num_tasks=50
    )
    
    print(f"\nTest Results:")
    print(f"  MSE: {test_results['mse_mean']:.4f} ± {test_results['mse_std']:.4f}")
    print(f"  MAE: {test_results['mae_mean']:.4f} ± {test_results['mae_std']:.4f}")
    
    # 测试Prototypical Networks
    print("\n" + "=" * 60)
    print("Testing Prototypical Networks")
    print("=" * 60)
    
    config.algorithm = "protonet"
    proto_learner = CrossDomainMetaLearner(config, algorithm="protonet")
    
    # 创建分类标签用于Prototypical Networks
    class_labels = np.digitize(labels, bins=np.linspace(labels.min(), labels.max(), 6))
    proto_sampler = MaterialTaskSampler(features, class_labels, config=config)
    
    proto_history = proto_learner.train(
        proto_sampler,
        num_iterations=500,
        log_interval=100
    )
    
    print("\nFinal accuracy:", np.mean(proto_history["accuracy"][-10:]))
    
    print("\n" + "=" * 60)
    print("Meta-Material Learning Demo Complete!")
    print("=" * 60)
