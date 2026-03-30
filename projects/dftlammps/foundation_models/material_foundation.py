"""
基础模型 (Foundation Models) 模块
实现材料科学领域的大型预训练模型架构
支持多任务预训练和少样本学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class FoundationModelConfig:
    """基础模型配置"""
    
    # 模型架构
    hidden_dim: int = 512
    num_layers: int = 12
    num_heads: int = 16
    ffn_dim: int = 2048
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # 输入输出
    max_atoms: int = 1000
    num_elements: int = 118
    max_neighbors: int = 50
    cutoff: float = 10.0
    
    # 预训练任务
    num_pretrain_tasks: int = 10
    
    # 多任务学习
    task_weights: Optional[Dict[str, float]] = None
    
    # 少样本学习
    protonet_dim: int = 128
    
    def __post_init__(self):
        if self.task_weights is None:
            self.task_weights = {
                'energy': 1.0,
                'forces': 10.0,
                'bandgap': 1.0,
                'bulk_modulus': 1.0,
                'shear_modulus': 1.0,
                'formation_energy': 1.0
            }


# ============== 位置编码 ==============

class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码 - 用于序列位置信息"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class SpatialPositionalEncoding(nn.Module):
    """空间位置编码 - 编码原子3D位置"""
    
    def __init__(self, d_model: int, num_freq: int = 16):
        super().__init__()
        self.d_model = d_model
        self.num_freq = num_freq
        
        # 频率编码
        frequencies = 2.0 ** torch.linspace(0, num_freq - 1, num_freq)
        self.register_buffer('frequencies', frequencies)
    
    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos: [N, 3] 3D位置
        Returns:
            [N, d_model] 编码后的位置
        """
        # 对每个坐标应用正弦编码
        pos_expanded = pos.unsqueeze(-1) * self.frequencies  # [N, 3, num_freq]
        
        sin_enc = torch.sin(pos_expanded)
        cos_enc = torch.cos(pos_expanded)
        
        # 展平并投影到d_model
        encoding = torch.cat([sin_enc, cos_enc], dim=-1)  # [N, 3, 2*num_freq]
        encoding = encoding.view(pos.shape[0], -1)  # [N, 6*num_freq]
        
        # 投影到目标维度
        if encoding.shape[-1] != self.d_model:
            projection = nn.Linear(encoding.shape[-1], self.d_model).to(pos.device)
            encoding = projection(encoding)
        
        return encoding


# ============== 核心Transformer层 ==============

class GraphMultiHeadAttention(nn.Module):
    """图多头注意力 - 带边约束的自注意力"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [N, d_model] 节点特征
            edge_index: [2, E] 边索引
            edge_attr: [E, d_edge] 边特征
            mask: [N, N] 注意力掩码
        """
        batch_size = x.shape[0]
        
        # 线性投影并分头
        Q = self.q_linear(x).view(batch_size, self.num_heads, self.d_k)
        K = self.k_linear(x).view(batch_size, self.num_heads, self.d_k)
        V = self.v_linear(x).view(batch_size, self.num_heads, self.d_k)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 应用掩码（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用边约束（只关注邻域）
        if edge_index is not None:
            # 创建基于边的掩码
            edge_mask = torch.zeros(batch_size, batch_size, device=x.device)
            edge_mask[edge_index[0], edge_index[1]] = 1
            scores = scores.masked_fill(edge_mask.unsqueeze(1) == 0, -1e9)
        
        # 软最大化
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力到值
        context = torch.matmul(attn, V)  # [batch, heads, d_k]
        context = context.view(batch_size, self.d_model)
        
        output = self.out_linear(context)
        
        return output


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层 - 适配图结构"""
    
    def __init__(self, config: FoundationModelConfig):
        super().__init__()
        
        self.self_attn = GraphMultiHeadAttention(
            config.hidden_dim,
            config.num_heads,
            config.attention_dropout
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        
        # 预归一化vs后归一化 - 这里使用预归一化
        self.pre_norm = True
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [N, hidden_dim] 节点特征
            edge_index: [2, E] 边索引
            edge_attr: [E, d_edge] 边特征
        """
        if self.pre_norm:
            # 预归一化
            x = x + self.self_attn(self.norm1(x), edge_index, edge_attr)
            x = x + self.ffn(self.norm2(x))
        else:
            # 后归一化
            x = self.norm1(x + self.self_attn(x, edge_index, edge_attr))
            x = self.norm2(x + self.ffn(x))
        
        return x


# ============== 基础模型主体 ==============

class MaterialFoundationModel(nn.Module):
    """
    材料基础大模型
    专为材料科学设计的大规模预训练Transformer
    """
    
    def __init__(self, config: FoundationModelConfig = None):
        super().__init__()
        self.config = config or FoundationModelConfig()
        
        # 原子嵌入
        self.atom_embedding = nn.Embedding(
            self.config.num_elements + 1,
            self.config.hidden_dim
        )
        
        # 空间位置编码
        self.spatial_encoding = SpatialPositionalEncoding(
            self.config.hidden_dim
        )
        
        # 类型嵌入（区分不同任务）
        self.task_embedding = nn.Embedding(20, self.config.hidden_dim)
        
        # 边嵌入
        self.edge_embedding = nn.Sequential(
            nn.Linear(64, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(self.config)
            for _ in range(self.config.num_layers)
        ])
        
        # 最终层归一化
        self.final_norm = nn.LayerNorm(self.config.hidden_dim)
        
        # 输出头
        self._build_output_heads()
        
        # 初始化
        self._init_weights()
    
    def _build_output_heads(self):
        """构建多任务输出头"""
        
        # 能量预测（图级）
        self.energy_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, 1)
        )
        
        # 力预测（原子级）
        self.force_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, 3)
        )
        
        # 多任务属性预测头
        self.property_heads = nn.ModuleDict({
            'bandgap': nn.Linear(self.config.hidden_dim, 1),
            'bulk_modulus': nn.Linear(self.config.hidden_dim, 1),
            'shear_modulus': nn.Linear(self.config.hidden_dim, 1),
            'formation_energy': nn.Linear(self.config.hidden_dim, 1),
            'magnetic_moment': nn.Linear(self.config.hidden_dim, 1),
            'phonon_dos': nn.Linear(self.config.hidden_dim, 100),  # 频谱预测
        })
        
        # 对比学习投影头
        self.contrastive_proj = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, 128)
        )
    
    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, data: Data, task_id: int = 0,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            data: PyG Data对象
            task_id: 任务ID（用于任务嵌入）
            return_features: 是否返回中间特征
        
        Returns:
            预测结果字典
        """
        # 原子嵌入 + 位置编码 + 任务嵌入
        x = self.atom_embedding(data.atomic_numbers)
        x = x + self.spatial_encoding(data.pos)
        x = x + self.task_embedding(
            torch.full((x.shape[0],), task_id, dtype=torch.long, device=x.device)
        )
        
        # 构建边（如果未提供）
        if not hasattr(data, 'edge_index') or data.edge_index is None:
            edge_index = self._build_edges(data.pos, data.batch)
        else:
            edge_index = data.edge_index
        
        # 计算边特征
        edge_attr = self._compute_edge_features(data.pos, edge_index)
        edge_emb = self.edge_embedding(edge_attr)
        
        # Transformer编码
        all_features = [x]
        for layer in self.layers:
            x = layer(x, edge_index, edge_emb)
            all_features.append(x)
        
        x = self.final_norm(x)
        
        # 多尺度特征融合
        x_fused = torch.stack(all_features[-3:], dim=0).mean(dim=0)
        
        # 图级特征
        graph_features = global_mean_pool(x_fused, data.batch)
        
        # 预测
        energy = self.energy_head(graph_features).squeeze(-1)
        forces = self.force_head(x_fused)
        
        # 多任务属性预测
        properties = {}
        for name, head in self.property_heads.items():
            properties[name] = head(graph_features).squeeze(-1)
        
        output = {
            'energy': energy,
            'forces': forces,
            'graph_features': graph_features,
            **properties
        }
        
        if return_features:
            output['atom_features'] = x_fused
            output['all_features'] = all_features
            output['contrastive_embedding'] = F.normalize(
                self.contrastive_proj(graph_features), dim=-1
            )
        
        return output
    
    def _build_edges(self, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """构建边"""
        from torch_geometric.nn import radius_graph
        return radius_graph(
            pos,
            r=self.config.cutoff,
            batch=batch,
            max_num_neighbors=self.config.max_neighbors
        )
    
    def _compute_edge_features(self, pos: torch.Tensor, 
                               edge_index: torch.Tensor) -> torch.Tensor:
        """计算边特征"""
        row, col = edge_index
        edge_vec = pos[col] - pos[row]
        edge_dist = torch.norm(edge_vec, dim=-1, keepdim=True)
        
        # 径向基函数
        d_min, d_max = 0.0, self.config.cutoff
        n_rbf = 64
        
        rbf = torch.exp(
            -n_rbf * (edge_dist - torch.linspace(d_min, d_max, n_rbf, device=pos.device)) ** 2
        )
        
        return rbf
    
    def get_embeddings(self, data: Data, pooling: str = 'mean') -> torch.Tensor:
        """获取图嵌入"""
        output = self.forward(data, return_features=True)
        
        if pooling == 'mean':
            return output['graph_features']
        elif pooling == 'contrastive':
            return output['contrastive_embedding']
        else:
            return output['atom_features']


# ============== 多任务学习 ==============

class MultiTaskTrainer:
    """多任务训练器"""
    
    def __init__(self, model: MaterialFoundationModel, 
                 config: FoundationModelConfig, device: str = 'cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        self.task_history = []
    
    def compute_multitask_loss(self, predictions: Dict[str, torch.Tensor],
                                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算多任务损失
        
        Args:
            predictions: 模型预测
            targets: 真实标签
        
        Returns:
            总损失, 各任务损失字典
        """
        losses = {}
        total_loss = 0.0
        
        # 能量损失
        if 'energy' in targets and 'energy' in predictions:
            energy_loss = F.mse_loss(predictions['energy'], targets['energy'])
            losses['energy'] = energy_loss.item()
            total_loss += self.config.task_weights.get('energy', 1.0) * energy_loss
        
        # 力损失
        if 'forces' in targets and 'forces' in predictions:
            force_loss = F.mse_loss(predictions['forces'], targets['forces'])
            losses['force'] = force_loss.item()
            total_loss += self.config.task_weights.get('forces', 10.0) * force_loss
        
        # 属性损失
        for prop_name in self.config.task_weights.keys():
            if prop_name in targets and prop_name in predictions:
                prop_loss = F.mse_loss(predictions[prop_name], targets[prop_name])
                losses[prop_name] = prop_loss.item()
                total_loss += self.config.task_weights.get(prop_name, 1.0) * prop_loss
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses
    
    def train_step(self, batch: Batch) -> Dict[str, float]:
        """单步训练"""
        self.model.train()
        self.optimizer.zero_grad()
        
        batch = batch.to(self.device)
        
        # 前向传播
        predictions = self.model(batch)
        
        # 准备目标
        targets = {}
        for key in ['energy', 'forces', 'bandgap', 'bulk_modulus', 
                    'shear_modulus', 'formation_energy']:
            if hasattr(batch, key):
                targets[key] = getattr(batch, key)
        
        # 计算损失
        loss, loss_dict = self.compute_multitask_loss(predictions, targets)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        return loss_dict
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """评估"""
        self.model.eval()
        
        all_losses = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                predictions = self.model(batch)
                
                targets = {}
                for key in ['energy', 'forces', 'bandgap']:
                    if hasattr(batch, key):
                        targets[key] = getattr(batch, key)
                
                _, loss_dict = self.compute_multitask_loss(predictions, targets)
                all_losses.append(loss_dict)
        
        # 平均损失
        avg_losses = {}
        for key in all_losses[0].keys():
            avg_losses[key] = np.mean([l[key] for l in all_losses])
        
        return avg_losses


# ============== 少样本学习 ==============

class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network 少样本学习
    """
    
    def __init__(self, encoder: MaterialFoundationModel, 
                 latent_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.latent_dim = latent_dim
        
        # 投影到度量空间
        self.projection = nn.Sequential(
            nn.Linear(encoder.config.hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
    
    def forward(self, data: Data) -> torch.Tensor:
        """获取嵌入"""
        graph_emb = self.encoder.get_embeddings(data)
        return F.normalize(self.projection(graph_emb), dim=-1)
    
    def compute_prototypes(self, support_data: List[Data], 
                          support_labels: torch.Tensor) -> torch.Tensor:
        """
        计算类别原型
        
        Args:
            support_data: 支持集数据列表
            support_labels: 支持集标签
        
        Returns:
            原型 [num_classes, latent_dim]
        """
        # 编码支持集
        embeddings = []
        for data in support_data:
            emb = self.forward(data)
            embeddings.append(emb)
        
        embeddings = torch.cat(embeddings, dim=0)
        
        # 计算每个类别的原型（平均）
        num_classes = support_labels.max().item() + 1
        prototypes = torch.zeros(num_classes, self.latent_dim, device=embeddings.device)
        
        for c in range(num_classes):
            mask = support_labels == c
            if mask.any():
                prototypes[c] = embeddings[mask].mean(dim=0)
        
        return prototypes
    
    def predict(self, query_data: List[Data], 
                prototypes: torch.Tensor) -> torch.Tensor:
        """
        预测查询样本
        
        Args:
            query_data: 查询集数据
            prototypes: 类别原型
        
        Returns:
            预测概率 [num_queries, num_classes]
        """
        # 编码查询集
        query_embeddings = []
        for data in query_data:
            emb = self.forward(data)
            query_embeddings.append(emb)
        
        query_embeddings = torch.cat(query_embeddings, dim=0)
        
        # 计算到原型的距离
        distances = torch.cdist(query_embeddings, prototypes)
        
        # 转换为概率（距离越近概率越高）
        logits = -distances
        return F.softmax(logits, dim=-1)
    
    def compute_loss(self, support_data: List[Data], 
                     support_labels: torch.Tensor,
                     query_data: List[Data],
                     query_labels: torch.Tensor) -> torch.Tensor:
        """计算少样本分类损失"""
        prototypes = self.compute_prototypes(support_data, support_labels)
        predictions = self.predict(query_data, prototypes)
        
        return F.cross_entropy(predictions, query_labels)


class MAMLAdapter:
    """
    MAML (Model-Agnostic Meta-Learning) 适配器
    """
    
    def __init__(self, model: MaterialFoundationModel, 
                 inner_lr: float = 0.01, num_inner_steps: int = 5):
        self.model = model
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
    
    def inner_loop(self, task_data: Data, task_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        内循环适应
        
        Args:
            task_data: 任务数据
            task_labels: 任务标签
        
        Returns:
            适应后的参数
        """
        # 克隆当前参数
        adapted_params = {
            name: param.clone()
            for name, param in self.model.named_parameters()
        }
        
        # 内循环梯度下降
        for _ in range(self.num_inner_steps):
            predictions = self.model.forward_with_params(task_data, adapted_params)
            loss = F.mse_loss(predictions['energy'], task_labels)
            
            # 计算梯度
            grads = torch.autograd.grad(
                loss, adapted_params.values(),
                create_graph=True, allow_unused=True
            )
            
            # 更新参数
            for (name, param), grad in zip(adapted_params.items(), grads):
                if grad is not None:
                    adapted_params[name] = param - self.inner_lr * grad
        
        return adapted_params
    
    def meta_train_step(self, task_batch: List[Tuple[Data, torch.Tensor]]) -> torch.Tensor:
        """
        元训练步骤
        
        Args:
            task_batch: 任务批次 [(data, labels), ...]
        """
        meta_loss = 0
        
        for task_data, task_labels in task_batch:
            # 划分支持和查询集
            n_support = len(task_labels) // 2
            support_data = task_data[:n_support]
            query_data = task_data[n_support:]
            
            support_labels = task_labels[:n_support]
            query_labels = task_labels[n_support:]
            
            # 内循环适应
            adapted_params = self.inner_loop(support_data, support_labels)
            
            # 外循环损失
            query_predictions = self.model.forward_with_params(query_data, adapted_params)
            meta_loss += F.mse_loss(query_predictions['energy'], query_labels)
        
        return meta_loss / len(task_batch)


# ============== 零样本学习 ==============

class ZeroShotPredictor:
    """
    零样本材料性质预测
    利用预训练模型的泛化能力
    """
    
    def __init__(self, model: MaterialFoundationModel):
        self.model = model
        self.known_properties = {}
    
    def register_property(self, name: str, 
                         description: str,
                         reference_materials: List[Tuple[Data, float]]):
        """
        注册新属性（无需训练）
        
        Args:
            name: 属性名称
            description: 属性描述
            reference_materials: 参考材料 [(data, value), ...]
        """
        # 计算参考材料的嵌入
        embeddings = []
        values = []
        
        for data, value in reference_materials:
            emb = self.model.get_embeddings(data)
            embeddings.append(emb)
            values.append(value)
        
        embeddings = torch.cat(embeddings, dim=0)
        values = torch.tensor(values, device=embeddings.device)
        
        # 存储属性原型
        self.known_properties[name] = {
            'embeddings': embeddings,
            'values': values,
            'description': description
        }
    
    def predict(self, data: Data, property_name: str) -> Tuple[float, float]:
        """
        零样本预测
        
        Returns:
            (预测值, 置信度)
        """
        if property_name not in self.known_properties:
            raise ValueError(f"Unknown property: {property_name}")
        
        prop_info = self.known_properties[property_name]
        
        # 计算查询材料的嵌入
        query_emb = self.model.get_embeddings(data)
        
        # 基于相似度的加权预测
        similarities = F.cosine_similarity(
            query_emb, prop_info['embeddings']
        )
        
        # 加权平均
        weights = F.softmax(similarities * 10, dim=0)
        prediction = (weights * prop_info['values']).sum()
        
        # 置信度基于最大相似度
        confidence = similarities.max().item()
        
        return prediction.item(), confidence


# ============== 预训练任务 ==============

class PretrainingTasks:
    """预训练任务集合"""
    
    @staticmethod
    def masked_atom_prediction(model: MaterialFoundationModel, 
                                batch: Batch, mask_ratio: float = 0.15) -> torch.Tensor:
        """掩码原子类型预测"""
        # 随机掩码原子
        mask = torch.rand(batch.atomic_numbers.shape[0]) < mask_ratio
        
        # 保存原始标签
        original = batch.atomic_numbers[mask].clone()
        
        # 掩码
        batch.atomic_numbers[mask] = 0  # 使用0作为MASK token
        
        # 前向传播
        output = model(batch)
        
        # 预测被掩码的原子（简化版）
        # 实际实现需要额外的分类头
        
        # 恢复原始值
        batch.atomic_numbers[mask] = original
        
        return torch.tensor(0.0)  # 占位
    
    @staticmethod
    def contrastive_learning(model: MaterialFoundationModel,
                             batch1: Batch, batch2: Batch,
                             temperature: float = 0.07) -> torch.Tensor:
        """对比学习"""
        # 编码两个视图
        emb1 = model.get_embeddings(batch1, pooling='contrastive')
        emb2 = model.get_embeddings(batch2, pooling='contrastive')
        
        # 归一化
        emb1 = F.normalize(emb1, dim=-1)
        emb2 = F.normalize(emb2, dim=-1)
        
        # 相似度矩阵
        similarity = torch.matmul(emb1, emb2.T) / temperature
        
        # InfoNCE损失
        labels = torch.arange(emb1.shape[0], device=emb1.device)
        loss = F.cross_entropy(similarity, labels)
        
        return loss
    
    @staticmethod
    def denoising(model: MaterialFoundationModel,
                  batch: Batch, noise_scale: float = 0.1) -> torch.Tensor:
        """去噪任务 - 预测添加的噪声"""
        # 添加噪声
        noise = torch.randn_like(batch.pos) * noise_scale
        noisy_pos = batch.pos + noise
        
        # 保存原始位置
        original_pos = batch.pos.clone()
        batch.pos = noisy_pos
        
        # 预测噪声（通过力）
        output = model(batch)
        
        # 力应该指向去噪方向
        predicted_noise = -output['forces'] * noise_scale
        
        # 恢复位置
        batch.pos = original_pos
        
        # 损失
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss


# ============== 使用示例 ==============

def example_foundation_model():
    """基础模型使用示例"""
    
    print("=" * 70)
    print("材料基础大模型示例")
    print("=" * 70)
    
    # 配置
    config = FoundationModelConfig(
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        ffn_dim=1024,
        dropout=0.1
    )
    
    # 创建模型
    model = MaterialFoundationModel(config)
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建示例数据
    num_atoms = 25
    data = Data(
        atomic_numbers=torch.randint(1, 50, (num_atoms,)),
        pos=torch.randn(num_atoms, 3) * 8,
        batch=torch.zeros(num_atoms, dtype=torch.long),
        energy=torch.tensor([2.5]),
        forces=torch.randn(num_atoms, 3)
    )
    
    # 前向传播
    print("\n前向传播:")
    output = model(data, return_features=True)
    
    print(f"  能量: {output['energy'].item():.4f} eV")
    print(f"  力形状: {output['forces'].shape}")
    print(f"  带隙: {output['bandgap'].item():.4f} eV")
    print(f"  体积模量: {output['bulk_modulus'].item():.4f} GPa")
    print(f"  剪切模量: {output['shear_modulus'].item():.4f} GPa")
    print(f"  图特征形状: {output['graph_features'].shape}")
    print(f"  对比嵌入形状: {output['contrastive_embedding'].shape}")
    
    # 多任务训练器示例
    print("\n多任务训练器:")
    trainer = MultiTaskTrainer(model, config, device='cpu')
    
    # 创建批次
    batch = Batch.from_data_list([data, data])
    loss_dict = trainer.train_step(batch)
    
    print(f"  训练损失:")
    for key, value in loss_dict.items():
        print(f"    {key}: {value:.4f}")
    
    # 少样本学习示例
    print("\n少样本学习 (Prototypical Network):")
    protonet = PrototypicalNetwork(model, latent_dim=128)
    
    # 模拟支持集和查询集
    support_data = [data]
    support_labels = torch.tensor([0])
    query_data = [data]
    
    prototypes = protonet.compute_prototypes(support_data, support_labels)
    print(f"  原型形状: {prototypes.shape}")
    
    predictions = protonet.predict(query_data, prototypes)
    print(f"  预测概率: {predictions}")
    
    print("\n" + "=" * 70)
    print("基础模型示例完成！")
    print("=" * 70)
    
    return model, output


if __name__ == "__main__":
    example_foundation_model()
