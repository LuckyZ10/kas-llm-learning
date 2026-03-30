"""
预训练GNN模型模块
实现大规模材料图的预训练、自监督学习和迁移学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, GraphSAGE, TransformerConv,
    global_mean_pool, global_max_pool, global_add_pool,
    MessagePassing, radius_graph
)
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_mean, scatter_add, scatter_max
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import warnings


@dataclass
class PretrainConfig:
    """预训练配置"""
    # 模型架构
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    
    # 预训练任务
    mask_ratio: float = 0.15
    contrastive_temp: float = 0.07
    
    # 优化器
    lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 10000
    
    # 训练
    batch_size: int = 64
    max_epochs: int = 100
    
    # 物理约束
    use_forces: bool = True
    force_weight: float = 10.0


class AtomEmbedding(nn.Module):
    """原子嵌入层 - 支持100+种元素"""
    
    def __init__(self, num_elements: int = 118, embedding_dim: int = 128):
        super().__init__()
        self.element_embedding = nn.Embedding(num_elements + 1, embedding_dim)
        
        # 可学习的原子属性
        self.atomic_radius = nn.Embedding(num_elements + 1, 1)
        self.electronegativity = nn.Embedding(num_elements + 1, 1)
        self.ionization_energy = nn.Embedding(num_elements + 1, 1)
        
        # 初始化
        nn.init.xavier_uniform_(self.element_embedding.weight)
        nn.init.normal_(self.atomic_radius.weight, mean=1.0, std=0.1)
        nn.init.normal_(self.electronegativity.weight, mean=2.5, std=0.5)
        nn.init.normal_(self.ionization_energy.weight, mean=10.0, std=2.0)
    
    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """
        Args:
            atomic_numbers: [N] 原子序数
        Returns:
            [N, embedding_dim + 3] 原子特征
        """
        elem_emb = self.element_embedding(atomic_numbers)
        radius = self.atomic_radius(atomic_numbers)
        en = self.electronegativity(atomic_numbers)
        ie = self.ionization_energy(atomic_numbers)
        
        return torch.cat([elem_emb, radius, en, ie], dim=-1)


class DistanceExpansion(nn.Module):
    """距离展开层 - Gaussian径向基函数"""
    
    def __init__(self, num_rbf: int = 50, cutoff: float = 10.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        
        # 可学习的中心点
        self.centers = nn.Parameter(torch.linspace(0, cutoff, num_rbf))
        # 可学习的宽度
        self.widths = nn.Parameter(torch.ones(num_rbf) * 0.5)
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distances: [E] 原子间距离
        Returns:
            [E, num_rbf] 展开后的距离特征
        """
        distances = distances.unsqueeze(-1)  # [E, 1]
        gamma = 1.0 / (2 * self.widths ** 2)
        rbf = torch.exp(-gamma * (distances - self.centers) ** 2)
        
        # 应用平滑截断
        cutoff_val = self.cosine_cutoff(distances.squeeze(-1))
        return rbf * cutoff_val.unsqueeze(-1)
    
    def cosine_cutoff(self, distances: torch.Tensor) -> torch.Tensor:
        """余弦截断函数"""
        cutoffs = 0.5 * (torch.cos(np.pi * distances / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs


class InteractionBlock(nn.Module):
    """交互块 - 消息传递的核心"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 多头注意力
        self.attention = TransformerConv(
            hidden_dim, hidden_dim // num_heads,
            heads=num_heads, dropout=dropout,
            edge_dim=hidden_dim
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 残差连接缩放
        self.alpha = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, hidden_dim] 节点特征
            edge_index: [2, E] 边索引
            edge_attr: [E, hidden_dim] 边特征
        Returns:
            [N, hidden_dim] 更新后的节点特征
        """
        # 自注意力 + 残差
        h = self.norm1(x)
        h = self.attention(h, edge_index, edge_attr)
        x = x + self.alpha * h
        
        # FFN + 残差
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h
        
        return x


class PretrainedGNN(nn.Module):
    """
    大规模预训练GNN模型
    支持多种预训练策略和下游任务迁移
    """
    
    def __init__(self, config: PretrainConfig = None):
        super().__init__()
        self.config = config or PretrainConfig()
        
        # 原子嵌入
        self.atom_embedding = AtomEmbedding(
            embedding_dim=self.config.hidden_dim // 2
        )
        
        # 投影到隐藏维度
        self.atom_proj = nn.Sequential(
            nn.Linear(self.config.hidden_dim // 2 + 3, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.SiLU()
        )
        
        # 距离展开
        self.distance_expansion = DistanceExpansion(
            num_rbf=self.config.hidden_dim,
            cutoff=10.0
        )
        
        # 交互层
        self.interaction_blocks = nn.ModuleList([
            InteractionBlock(
                self.config.hidden_dim,
                self.config.num_heads,
                self.config.dropout
            )
            for _ in range(self.config.num_layers)
        ])
        
        # 输出头
        self.energy_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, 1)
        )
        
        self.force_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, 3)
        )
        
        # 掩码token预测头（用于预训练）
        self.mask_predictor = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, 119)  # 118种元素 + mask token
        )
        
        # 图级表示头（用于对比学习）
        self.graph_proj = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, 128)  # 低维投影
        )
    
    def forward(self, data: Data, return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            data: PyG Data对象
            return_features: 是否返回中间特征
        
        Returns:
            包含能量、力等预测结果的字典
        """
        # 原子特征
        x = self.atom_embedding(data.atomic_numbers)
        x = self.atom_proj(x)
        
        # 构建边（如果没有提供）
        if not hasattr(data, 'edge_index') or data.edge_index is None:
            edge_index = radius_graph(
                data.pos, r=10.0, batch=data.batch
            )
        else:
            edge_index = data.edge_index
        
        # 边特征
        edge_vec = data.pos[edge_index[0]] - data.pos[edge_index[1]]
        edge_dist = torch.norm(edge_vec, dim=-1)
        edge_attr = self.distance_expansion(edge_dist)
        
        # 消息传递
        features = [x]
        for block in self.interaction_blocks:
            x = block(x, edge_index, edge_attr)
            features.append(x)
        
        # 多尺度特征融合
        x = torch.stack(features[-3:], dim=0).mean(dim=0)
        
        # 能量预测（图级）
        atomic_energies = self.energy_head(x)
        energy = scatter_add(atomic_energies, data.batch, dim=0)
        
        # 力预测（原子级）
        forces = self.force_head(x)
        
        output = {
            'energy': energy.squeeze(-1),
            'forces': forces,
            'atomic_features': x
        }
        
        if return_features:
            output['features'] = features
            output['graph_embedding'] = self.graph_proj(
                scatter_mean(x, data.batch, dim=0)
            )
        
        return output
    
    def predict_mask(self, atomic_features: torch.Tensor) -> torch.Tensor:
        """预测被掩码的原子类型"""
        return self.mask_predictor(atomic_features)
    
    def get_graph_embedding(self, data: Data) -> torch.Tensor:
        """获取图级嵌入"""
        output = self.forward(data, return_features=True)
        return output['graph_embedding']


class ContrastiveLearner(nn.Module):
    """
    对比学习模块
    用于学习材料图的良好表示
    """
    
    def __init__(self, encoder: PretrainedGNN, temperature: float = 0.07):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature
        
        # 动量编码器（用于对比学习）
        self.momentum_encoder = PretrainedGNN(encoder.config)
        self._update_momentum_encoder(0)
        
        # 队列存储负样本
        self.register_buffer("queue", torch.randn(128, 65536))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue = F.normalize(self.queue, dim=0)
    
    def _update_momentum_encoder(self, momentum: float = 0.999):
        """更新动量编码器"""
        for param_q, param_k in zip(
            self.encoder.parameters(), 
            self.momentum_encoder.parameters()
        ):
            param_k.data = momentum * param_k.data + (1 - momentum) * param_q.data
    
    def forward(self, data1: Data, data2: Data) -> torch.Tensor:
        """
        对比学习前向传播
        data1, data2: 同一材料的两种增强视图
        """
        # 编码两个视图
        z1 = self.encoder.get_graph_embedding(data1)
        z2 = self.momentum_encoder.get_graph_embedding(data2)
        
        # 归一化
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # 计算相似度
        l_pos = torch.einsum('nc,nc->n', [z1, z2]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [z1, self.queue.clone().detach()])
        
        # InfoNCE损失
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        
        # 更新队列
        self._dequeue_and_enqueue(z2)
        
        return loss
    
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """更新队列"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # 替换队列中的旧样本
        if ptr + batch_size <= self.queue.shape[1]:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            remaining = self.queue.shape[1] - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
        
        ptr = (ptr + batch_size) % self.queue.shape[1]
        self.queue_ptr[0] = ptr


class MaskedAtomPrediction(nn.Module):
    """
    掩码原子预测任务
    类似BERT的MLM任务，预测被掩码的原子类型
    """
    
    def __init__(self, encoder: PretrainedGNN, mask_ratio: float = 0.15):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播并计算掩码预测损失
        
        Returns:
            loss, predictions, targets
        """
        # 随机选择掩码位置
        num_atoms = data.atomic_numbers.shape[0]
        num_mask = int(num_atoms * self.mask_ratio)
        mask_indices = torch.randperm(num_atoms)[:num_mask]
        
        # 保存原始标签
        targets = data.atomic_numbers[mask_indices].clone()
        
        # 应用掩码（使用特殊token 118表示mask）
        masked_atomic_numbers = data.atomic_numbers.clone()
        masked_atomic_numbers[mask_indices] = 118
        
        # 创建新的data对象
        masked_data = Data(
            atomic_numbers=masked_atomic_numbers,
            pos=data.pos,
            batch=data.batch,
            edge_index=data.edge_index if hasattr(data, 'edge_index') else None
        )
        
        # 前向传播
        output = self.encoder(masked_data, return_features=True)
        
        # 预测被掩码的原子
        masked_features = output['atomic_features'][mask_indices]
        predictions = self.encoder.predict_mask(masked_features)
        
        # 计算损失
        loss = self.criterion(predictions, targets)
        
        return loss, predictions, targets


class PropertyPredictionHead(nn.Module):
    """属性预测头 - 用于下游任务微调"""
    
    def __init__(self, input_dim: int, num_tasks: int, task_types: List[str] = None):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_types = task_types or ['regression'] * num_tasks
        
        # 共享层
        self.shared = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.SiLU(),
            nn.Dropout(0.1)
        )
        
        # 任务特定层
        self.heads = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(num_tasks)
        ])
        
        # 任务权重（可学习）
        self.task_weights = nn.Parameter(torch.ones(num_tasks))
    
    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            graph_embedding: [B, input_dim] 图级嵌入
        Returns:
            [B, num_tasks] 预测结果
        """
        h = self.shared(graph_embedding)
        
        predictions = []
        for head in self.heads:
            predictions.append(head(h))
        
        return torch.cat(predictions, dim=-1)
    
    def compute_loss(self, predictions: torch.Tensor, 
                     targets: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        计算多任务损失
        
        Args:
            predictions: [B, num_tasks]
            targets: [B, num_tasks]
            mask: [B, num_tasks] 有效掩码
        """
        losses = []
        for i, task_type in enumerate(self.task_types):
            pred = predictions[:, i]
            target = targets[:, i]
            
            if mask is not None:
                valid = mask[:, i]
                pred = pred[valid]
                target = target[valid]
            
            if task_type == 'regression':
                loss = F.mse_loss(pred, target)
            elif task_type == 'classification':
                loss = F.cross_entropy(pred.unsqueeze(-1), target.long())
            else:
                loss = F.l1_loss(pred, target)
            
            losses.append(loss * self.task_weights[i])
        
        return sum(losses) / self.num_tasks


class PretrainingTrainer:
    """预训练器 - 管理整个预训练流程"""
    
    def __init__(self, model: PretrainedGNN, config: PretrainConfig, 
                 device: str = 'cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # 预训练任务
        self.masked_prediction = MaskedAtomPrediction(model, config.mask_ratio)
        self.contrastive_learner = ContrastiveLearner(model, config.contrastive_temp)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.max_epochs
        )
        
        self.epoch = 0
        self.step = 0
    
    def pretrain_step(self, batch: Batch, task_weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        单步预训练
        
        Args:
            batch: 数据批次
            task_weights: 各任务的权重
        
        Returns:
            损失字典
        """
        task_weights = task_weights or {'mask': 1.0, 'contrastive': 0.5, 'energy': 1.0}
        
        self.model.train()
        self.optimizer.zero_grad()
        
        batch = batch.to(self.device)
        losses = {}
        total_loss = 0
        
        # 1. 掩码原子预测
        if task_weights.get('mask', 0) > 0:
            mask_loss, _, _ = self.masked_prediction(batch)
            losses['mask'] = mask_loss.item()
            total_loss += task_weights['mask'] * mask_loss
        
        # 2. 能量/力预测
        if task_weights.get('energy', 0) > 0:
            output = self.model(batch)
            
            if hasattr(batch, 'energy'):
                energy_loss = F.mse_loss(output['energy'], batch.energy)
                losses['energy'] = energy_loss.item()
                total_loss += task_weights['energy'] * energy_loss
            
            if hasattr(batch, 'forces') and self.config.use_forces:
                force_loss = F.mse_loss(output['forces'], batch.forces)
                losses['force'] = force_loss.item()
                total_loss += self.config.force_weight * force_loss
        
        # 3. 对比学习（需要数据增强）
        # 这里简化处理，实际应用中需要生成两个视图
        
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        losses['total'] = total_loss.item()
        self.step += 1
        
        return losses
    
    def finetune(self, train_loader, val_loader=None, num_epochs: int = 100,
                 property_head: PropertyPredictionHead = None):
        """
        微调模型用于特定下游任务
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            property_head: 属性预测头
        """
        if property_head is None:
            property_head = PropertyPredictionHead(
                self.config.hidden_dim, num_tasks=1
            ).to(self.device)
        
        # 冻结部分层（可选）
        # for param in self.model.atom_embedding.parameters():
        #     param.requires_grad = False
        
        optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(property_head.parameters()),
            lr=self.config.lr / 10,  # 微调使用更小学习率
            weight_decay=self.config.weight_decay
        )
        
        for epoch in range(num_epochs):
            self.model.train()
            property_head.train()
            
            train_loss = 0
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                # 获取图表示
                output = self.model(batch, return_features=True)
                graph_emb = output['graph_embedding']
                
                # 属性预测
                predictions = property_head(graph_emb)
                
                # 计算损失
                if hasattr(batch, 'y'):
                    loss = property_head.compute_loss(
                        predictions, batch.y, getattr(batch, 'mask', None)
                    )
                    
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
            
            avg_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}")
            
            # 验证
            if val_loader is not None:
                self._validate(val_loader, property_head)
    
    def _validate(self, val_loader, property_head: PropertyPredictionHead):
        """验证"""
        self.model.eval()
        property_head.eval()
        
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                output = self.model(batch, return_features=True)
                predictions = property_head(output['graph_embedding'])
                
                if hasattr(batch, 'y'):
                    loss = property_head.compute_loss(
                        predictions, batch.y, getattr(batch, 'mask', None)
                    )
                    val_loss += loss.item()
        
        avg_loss = val_loss / len(val_loader)
        print(f"Val Loss: {avg_loss:.4f}")
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        torch.save({
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        print(f"Checkpoint loaded from {path}")


class FewShotLearner:
    """
    少样本学习器
    利用预训练模型快速适应新任务
    """
    
    def __init__(self, pretrained_model: PretrainedGNN, config: PretrainConfig):
        self.model = pretrained_model
        self.config = config
        self.device = next(pretrained_model.parameters()).device
    
    def protonet_fit(self, support_set: Batch, num_classes: int = 2):
        """
        Prototypical Network 少样本学习
        
        Args:
            support_set: 支持集
            num_classes: 类别数
        """
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(support_set, return_features=True)
            embeddings = output['graph_embedding']
            labels = support_set.y
        
        # 计算每个类别的原型
        prototypes = []
        for c in range(num_classes):
            mask = labels == c
            if mask.any():
                proto = embeddings[mask].mean(dim=0)
                prototypes.append(proto)
        
        self.prototypes = torch.stack(prototypes)
    
    def protonet_predict(self, query_set: Batch) -> torch.Tensor:
        """
        Prototypical Network 预测
        
        Returns:
            预测概率 [N, num_classes]
        """
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(query_set, return_features=True)
            embeddings = output['graph_embedding']
        
        # 计算到每个原型的距离
        distances = torch.cdist(embeddings, self.prototypes)
        logits = -distances
        
        return F.softmax(logits, dim=-1)
    
    def maml_adapt(self, task_batch: List[Batch], inner_lr: float = 0.01, 
                   inner_steps: int = 5):
        """
        MAML (Model-Agnostic Meta-Learning) 适应
        
        Args:
            task_batch: 任务批次列表
            inner_lr: 内循环学习率
            inner_steps: 内循环步数
        """
        meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
        meta_loss = 0
        for task_data in task_batch:
            # 克隆模型参数
            adapted_params = {
                name: param.clone() 
                for name, param in self.model.named_parameters()
            }
            
            # 内循环适应
            for _ in range(inner_steps):
                output = self.model(task_data)
                loss = F.mse_loss(output['energy'], task_data.energy)
                
                # 手动计算梯度并更新
                grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
                for (name, param), grad in zip(self.model.named_parameters(), grads):
                    adapted_params[name] = param - inner_lr * grad
            
            # 外循环损失
            # 使用适应后的参数进行预测
            # ... (实现细节)
        
        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()


# ============== 数据增强工具 ==============

def augment_structure(data: Data, aug_type: str = 'rotation', 
                      magnitude: float = 0.1) -> Data:
    """
    结构数据增强
    
    Args:
        data: 原始数据
        aug_type: 增强类型 ('rotation', 'translation', 'noise', 'dropout')
        magnitude: 增强强度
    
    Returns:
        增强后的数据
    """
    new_data = data.clone()
    
    if aug_type == 'rotation':
        # 随机旋转
        angle = torch.rand(3) * 2 * np.pi * magnitude
        R = rotation_matrix(angle)
        new_data.pos = new_data.pos @ R.T
    
    elif aug_type == 'translation':
        # 随机平移
        translation = torch.randn(3) * magnitude
        new_data.pos = new_data.pos + translation
    
    elif aug_type == 'noise':
        # 添加高斯噪声
        noise = torch.randn_like(new_data.pos) * magnitude
        new_data.pos = new_data.pos + noise
    
    elif aug_type == 'dropout':
        # 随机删除原子
        num_keep = int(data.pos.shape[0] * (1 - magnitude))
        keep_idx = torch.randperm(data.pos.shape[0])[:num_keep]
        new_data.pos = data.pos[keep_idx]
        new_data.atomic_numbers = data.atomic_numbers[keep_idx]
    
    return new_data


def rotation_matrix(angles: torch.Tensor) -> torch.Tensor:
    """生成3D旋转矩阵"""
    rx = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(angles[0]), -torch.sin(angles[0])],
        [0, torch.sin(angles[0]), torch.cos(angles[0])]
    ])
    ry = torch.tensor([
        [torch.cos(angles[1]), 0, torch.sin(angles[1])],
        [0, 1, 0],
        [-torch.sin(angles[1]), 0, torch.cos(angles[1])]
    ])
    rz = torch.tensor([
        [torch.cos(angles[2]), -torch.sin(angles[2]), 0],
        [torch.sin(angles[2]), torch.cos(angles[2]), 0],
        [0, 0, 1]
    ])
    return rz @ ry @ rx


# ============== 使用示例 ==============

def example_pretrained_gnn():
    """预训练GNN使用示例"""
    
    # 配置
    config = PretrainConfig(
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        mask_ratio=0.15,
        lr=1e-4
    )
    
    # 创建模型
    model = PretrainedGNN(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建示例数据
    num_atoms = 20
    data = Data(
        atomic_numbers=torch.randint(1, 100, (num_atoms,)),
        pos=torch.randn(num_atoms, 3) * 10,
        batch=torch.zeros(num_atoms, dtype=torch.long),
        energy=torch.tensor([1.5]),
        forces=torch.randn(num_atoms, 3)
    )
    
    # 前向传播
    output = model(data)
    print(f"Energy prediction: {output['energy'].item():.4f} eV")
    print(f"Forces shape: {output['forces'].shape}")
    
    # 掩码预测
    masked_pred = MaskedAtomPrediction(model)
    loss, preds, targets = masked_pred(data)
    print(f"Mask prediction loss: {loss.item():.4f}")
    
    return model, output


if __name__ == "__main__":
    example_pretrained_gnn()
