#!/usr/bin/env python3
"""
表示学习模块 - 材料结构的状态表示

包含:
- CrystalGraphEncoder: 晶体图神经网络编码器
- CompositionEncoder: 化学组成编码器
- StateEncoder: 通用状态编码器
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EncoderConfig:
    """编码器配置"""
    input_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 128
    n_layers: int = 3
    dropout: float = 0.1


class CrystalGraphEncoder(nn.Module):
    """
    晶体图神经网络编码器
    
    使用消息传递网络编码晶体结构。
    参考: Xie & Grossman, "Crystal Graph Convolutional Neural Networks", 2018
    """
    
    def __init__(
        self,
        n_elements: int = 100,
        hidden_dim: int = 64,
        n_conv_layers: int = 3,
        output_dim: int = 128
    ):
        super().__init__()
        
        self.n_elements = n_elements
        self.hidden_dim = hidden_dim
        
        # 原子嵌入
        self.atom_embedding = nn.Embedding(n_elements, hidden_dim)
        
        # 图卷积层
        self.conv_layers = nn.ModuleList([
            CrystalGraphConv(hidden_dim, hidden_dim)
            for _ in range(n_conv_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        lattice: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            atomic_numbers: (n_atoms,) 原子序数
            positions: (n_atoms, 3) 原子位置
            lattice: (3, 3) 晶格矩阵
            edge_index: (2, n_edges) 边索引
            
        Returns:
            (output_dim,) 结构表示
        """
        # 初始嵌入
        x = self.atom_embedding(atomic_numbers)
        
        # 构建边 (如果没有提供)
        if edge_index is None:
            edge_index = self._build_edges(positions, lattice)
        
        # 图卷积
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
        
        # 全局池化 (平均)
        x = x.mean(dim=0)
        
        # 输出
        out = self.output_layer(x)
        
        return out
    
    def _build_edges(
        self,
        positions: torch.Tensor,
        lattice: torch.Tensor,
        cutoff: float = 5.0
    ) -> torch.Tensor:
        """构建边 (基于距离)"""
        n_atoms = len(positions)
        
        # 转换为笛卡尔坐标
        cart_pos = positions @ lattice
        
        # 计算距离矩阵
        distances = torch.cdist(cart_pos, cart_pos)
        
        # 创建边索引
        edge_index = []
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if distances[i, j] < cutoff:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
        
        if edge_index:
            return torch.tensor(edge_index, dtype=torch.long).t()
        else:
            return torch.zeros((2, 0), dtype=torch.long)


class CrystalGraphConv(nn.Module):
    """晶体图卷积层"""
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        
        self.linear = nn.Linear(in_dim, out_dim)
        self.gate_linear = nn.Linear(in_dim * 2, out_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """消息传递"""
        if edge_index.shape[1] == 0:
            return self.linear(x)
        
        # 聚合邻居信息
        row, col = edge_index
        neighbor_features = x[col]
        
        # 门控机制
        gate_input = torch.cat([x[row], neighbor_features], dim=-1)
        gate = torch.sigmoid(self.gate_linear(gate_input))
        
        # 加权消息
        messages = gate * self.linear(neighbor_features)
        
        # 聚合
        out = torch.zeros_like(x)
        out.index_add_(0, row, messages)
        
        # 残差连接
        out = out + self.linear(x)
        
        return out


class CompositionEncoder(nn.Module):
    """
    化学组成编码器
    
    将化学组成编码为固定维度的向量。
    """
    
    def __init__(
        self,
        n_elements: int = 100,
        embedding_dim: int = 32,
        output_dim: int = 128
    ):
        super().__init__()
        
        # 元素嵌入
        self.element_embedding = nn.Embedding(n_elements, embedding_dim)
        
        # 组成编码器
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, composition: torch.Tensor) -> torch.Tensor:
        """
        编码组成
        
        Args:
            composition: (n_elements,) 各元素的原子分数
            
        Returns:
            (output_dim,) 组成表示
        """
        # 元素索引
        indices = torch.arange(len(composition), device=composition.device)
        
        # 元素嵌入
        elem_emb = self.element_embedding(indices)
        
        # 加权平均
        weighted_emb = (elem_emb * composition.unsqueeze(-1)).sum(dim=0)
        
        # 编码
        out = self.encoder(weighted_emb)
        
        return out


class StateEncoder(nn.Module):
    """
    通用状态编码器
    
    组合结构编码和组成编码。
    """
    
    def __init__(
        self,
        structure_encoder: nn.Module,
        composition_encoder: nn.Module,
        output_dim: int = 128
    ):
        super().__init__()
        
        self.structure_encoder = structure_encoder
        self.composition_encoder = composition_encoder
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(structure_encoder.output_layer[-1].out_features + 
                     composition_encoder.encoder[-1].out_features,
                     256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(
        self,
        structure_input: Dict[str, torch.Tensor],
        composition: torch.Tensor
    ) -> torch.Tensor:
        """编码完整状态"""
        # 结构编码
        struct_emb = self.structure_encoder(**structure_input)
        
        # 组成编码
        comp_emb = self.composition_encoder(composition)
        
        # 融合
        combined = torch.cat([struct_emb, comp_emb], dim=-1)
        out = self.fusion(combined)
        
        return out


class AttentionBasedEncoder(nn.Module):
    """
    基于注意力的编码器
    
    使用自注意力机制编码原子环境。
    """
    
    def __init__(
        self,
        n_elements: int = 100,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        output_dim: int = 128
    ):
        super().__init__()
        
        # 原子嵌入
        self.atom_embedding = nn.Embedding(n_elements, d_model)
        
        # 位置编码
        self.position_embedding = nn.Linear(3, d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 输出层
        self.output_layer = nn.Linear(d_model, output_dim)
    
    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            atomic_numbers: (n_atoms,) 原子序数
            positions: (n_atoms, 3) 原子位置
            
        Returns:
            (output_dim,) 结构表示
        """
        # 嵌入
        atom_emb = self.atom_embedding(atomic_numbers)
        pos_emb = self.position_embedding(positions)
        
        x = atom_emb + pos_emb
        
        # Transformer (添加批次维度)
        x = x.unsqueeze(0)  # (1, n_atoms, d_model)
        x = self.transformer(x)
        x = x.squeeze(0)  # (n_atoms, d_model)
        
        # 全局平均池化
        x = x.mean(dim=0)
        
        # 输出
        out = self.output_layer(x)
        
        return out
    
    def get_attention_weights(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """获取注意力权重"""
        # 这需要修改PyTorch Transformer以返回注意力权重
        # 简化实现
        return torch.zeros(len(atomic_numbers), len(atomic_numbers))
