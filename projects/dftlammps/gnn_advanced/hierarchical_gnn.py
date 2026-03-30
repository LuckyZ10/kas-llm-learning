"""
层次图神经网络模块
实现原子-化学键-晶胞多层次表示学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    MessagePassing, global_mean_pool, global_max_pool, 
    global_add_pool, AttentionalAggregation,
    radius_graph, knn_graph
)
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_mean, scatter_add, scatter_max
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings


@dataclass
class HierarchicalConfig:
    """层次GNN配置"""
    # 原子级别
    atom_hidden_dim: int = 128
    atom_num_layers: int = 4
    
    # 化学键级别
    bond_hidden_dim: int = 64
    bond_num_layers: int = 2
    
    # 晶胞级别
    cell_hidden_dim: int = 256
    cell_num_layers: int = 2
    
    # 注意力设置
    num_heads: int = 8
    attention_dropout: float = 0.1
    
    # 物理参数
    atom_cutoff: float = 5.0  # 原子邻域截断
    bond_cutoff: float = 8.0  # 化学键截断
    max_neighbors: int = 32
    
    # 训练
    dropout: float = 0.1
    use_layer_norm: bool = True


# ============== 原子级别层 ==============

class AtomConv(MessagePassing):
    """原子级别卷积 - 捕获局部化学环境"""
    
    def __init__(self, hidden_dim: int, num_radial: int = 32):
        super().__init__(aggr='add')
        
        self.hidden_dim = hidden_dim
        
        # 边嵌入（距离相关）
        self.edge_embed = nn.Sequential(
            nn.Linear(num_radial, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 消息函数
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 更新函数
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 门控
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        
        # 层归一化
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, hidden_dim] 原子特征
            edge_index: [2, E] 边索引
            edge_attr: [E, num_radial] 边属性（径向基）
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """消息函数"""
        # 边嵌入
        edge_emb = self.edge_embed(edge_attr)
        
        # 拼接并生成消息
        msg = self.message_net(torch.cat([x_i, x_j, edge_emb], dim=-1))
        
        # 门控
        gate = torch.sigmoid(self.gate(edge_emb))
        return msg * gate
    
    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """更新函数"""
        # 门控残差连接
        update = self.update_net(torch.cat([aggr_out, x], dim=-1))
        return self.norm(x + update)


class AtomLevel(nn.Module):
    """原子级别表示学习"""
    
    def __init__(self, config: HierarchicalConfig):
        super().__init__()
        self.config = config
        
        # 原子嵌入
        self.atom_embed = nn.Embedding(100, config.atom_hidden_dim)
        
        # 径向基函数
        self.rbf = GaussianBasis(50, config.atom_cutoff)
        
        # 卷积层
        self.conv_layers = nn.ModuleList([
            AtomConv(config.atom_hidden_dim, 50)
            for _ in range(config.atom_num_layers)
        ])
        
        # 自注意力池化
        self.attention_pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(config.atom_hidden_dim, config.atom_hidden_dim),
                nn.Tanh(),
                nn.Linear(config.atom_hidden_dim, 1)
            ),
            nn=nn.Sequential(
                nn.Linear(config.atom_hidden_dim, config.atom_hidden_dim),
                nn.SiLU()
            )
        )
        
        # 读出头
        self.atom_readout = nn.Sequential(
            nn.Linear(config.atom_hidden_dim, config.atom_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.atom_hidden_dim, config.atom_hidden_dim)
        )
    
    def forward(self, atomic_numbers: torch.Tensor, pos: torch.Tensor,
                batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            atomic_numbers: [N] 原子序数
            pos: [N, 3] 原子位置
            batch: [N] 批次索引
        
        Returns:
            atom_features: [N, hidden_dim] 原子特征
            atom_embeddings: [N, hidden_dim] 原子嵌入
            graph_atom_features: [B, hidden_dim] 图级原子特征
        """
        # 初始化特征
        x = self.atom_embed(atomic_numbers)
        
        # 构建邻域图
        edge_index = radius_graph(
            pos, r=self.config.atom_cutoff,
            batch=batch, max_num_neighbors=self.config.max_neighbors
        )
        
        # 计算边特征
        row, col = edge_index
        edge_vec = pos[col] - pos[row]
        edge_dist = torch.norm(edge_vec, dim=-1)
        edge_attr = self.rbf(edge_dist)
        
        # 消息传递
        all_features = [x]
        for conv in self.conv_layers:
            x = conv(x, edge_index, edge_attr)
            all_features.append(x)
        
        # 多尺度特征融合
        atom_features = torch.stack(all_features[-3:], dim=0).mean(dim=0)
        atom_embeddings = self.atom_readout(atom_features)
        
        # 图级特征
        graph_atom_features = self.attention_pool(atom_embeddings, batch)
        
        return atom_features, atom_embeddings, graph_atom_features


# ============== 化学键级别层 ==============

class BondConv(MessagePassing):
    """化学键级别卷积 - 处理化学键网络"""
    
    def __init__(self, hidden_dim: int):
        super().__init__(aggr='mean')
        
        self.hidden_dim = hidden_dim
        
        # 边类型嵌入（单键、双键、三键、芳香键等）
        self.bond_type_embed = nn.Embedding(5, hidden_dim)
        
        # 消息网络
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 更新网络
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, bond_features: torch.Tensor, bond_index: torch.Tensor,
                bond_types: torch.Tensor, atom_to_bond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bond_features: [num_bonds, hidden_dim] 化学键特征
            bond_index: [2, num_bond_edges] 化学键之间的连接
            bond_types: [num_bonds] 化学键类型
            atom_to_bond: 映射信息
        """
        bond_type_emb = self.bond_type_embed(bond_types)
        
        return self.propagate(
            bond_index, x=bond_features, 
            bond_type_emb=bond_type_emb
        )
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor,
                bond_type_emb_i: torch.Tensor) -> torch.Tensor:
        """消息函数"""
        return self.message_net(torch.cat([x_i, x_j, bond_type_emb_i], dim=-1))
    
    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """更新函数"""
        return x + self.update_net(torch.cat([aggr_out, x], dim=-1))


class BondLevel(nn.Module):
    """化学键级别表示学习"""
    
    def __init__(self, config: HierarchicalConfig):
        super().__init__()
        self.config = config
        
        # 从原子特征创建化学键特征
        self.bond_init = nn.Sequential(
            nn.Linear(config.atom_hidden_dim * 2, config.bond_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.bond_hidden_dim, config.bond_hidden_dim)
        )
        
        # 化学键卷积
        self.bond_convs = nn.ModuleList([
            BondConv(config.bond_hidden_dim)
            for _ in range(config.bond_num_layers)
        ])
        
        # 化学键读出
        self.bond_readout = nn.Sequential(
            nn.Linear(config.bond_hidden_dim, config.bond_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.bond_hidden_dim, config.bond_hidden_dim)
        )
    
    def forward(self, atom_features: torch.Tensor, atom_pos: torch.Tensor,
                atomic_numbers: torch.Tensor, batch: torch.Tensor,
                edge_index: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从原子特征构建化学键网络
        
        Args:
            atom_features: [N, atom_hidden_dim] 原子特征
            atom_pos: [N, 3] 原子位置
            atomic_numbers: [N] 原子序数
            batch: [N] 批次索引
            edge_index: 预定义的化学键（可选）
        
        Returns:
            bond_features: [num_bonds, bond_hidden_dim] 化学键特征
            graph_bond_features: [B, bond_hidden_dim] 图级化学键特征
        """
        # 如果没有预定义化学键，基于距离和原子类型推断
        if edge_index is None:
            edge_index = self._infer_bonds(atom_pos, atomic_numbers, batch)
        
        # 创建化学键特征
        row, col = edge_index
        bond_input = torch.cat([atom_features[row], atom_features[col]], dim=-1)
        bond_features = self.bond_init(bond_input)
        
        # 推断化学键类型（简化版）
        bond_types = self._classify_bonds(atom_features, edge_index)
        
        # 构建化学键图（键与键相邻如果共享原子）
        bond_edge_index = self._build_bond_graph(edge_index)
        
        # 消息传递
        for conv in self.bond_convs:
            bond_features = conv(bond_features, bond_edge_index, bond_types, None)
        
        # 图级特征
        # 计算每个晶胞的化学键数
        num_bonds_per_graph = scatter_add(
            torch.ones(edge_index.shape[1], device=edge_index.device),
            batch[row], dim=0
        ).long()
        
        bond_batch = torch.repeat_interleave(
            torch.arange(len(num_bonds_per_graph), device=edge_index.device),
            num_bonds_per_graph
        )
        
        graph_bond_features = scatter_mean(bond_features, bond_batch, dim=0)
        
        return bond_features, graph_bond_features
    
    def _infer_bonds(self, pos: torch.Tensor, atomic_numbers: torch.Tensor,
                     batch: torch.Tensor) -> torch.Tensor:
        """基于距离和共价半径推断化学键"""
        # 共价半径（简化）
        covalent_radii = torch.tensor([
            0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58,  # H-Ne
            1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, 2.03, 1.76,  # Na-Ca
        ], device=pos.device)
        
        # 使用较宽松的截断
        edge_index = radius_graph(
            pos, r=self.config.bond_cutoff,
            batch=batch, max_num_neighbors=8
        )
        
        row, col = edge_index
        
        # 基于共价半径筛选
        dist = torch.norm(pos[col] - pos[row], dim=-1)
        
        # 获取原子半径
        radii_i = covalent_radii[torch.clamp(atomic_numbers[row] - 1, 0, 19)]
        radii_j = covalent_radii[torch.clamp(atomic_numbers[col] - 1, 0, 19)]
        
        # 如果距离小于共价半径之和的1.3倍，认为是化学键
        bond_threshold = (radii_i + radii_j) * 1.3
        bond_mask = dist < bond_threshold
        
        return edge_index[:, bond_mask]
    
    def _classify_bonds(self, atom_features: torch.Tensor,
                        edge_index: torch.Tensor) -> torch.Tensor:
        """简单化学键类型分类"""
        # 简化：基于特征相似度推断键级
        row, col = edge_index
        similarity = F.cosine_similarity(atom_features[row], atom_features[col], dim=-1)
        
        # 相似度阈值分类
        bond_types = torch.zeros(edge_index.shape[1], dtype=torch.long, device=edge_index.device)
        bond_types[similarity > 0.8] = 1  # 双键
        bond_types[similarity > 0.9] = 2  # 三键
        bond_types[similarity < 0.3] = 3  # 配位键
        
        return bond_types
    
    def _build_bond_graph(self, bond_index: torch.Tensor) -> torch.Tensor:
        """构建化学键图（键相邻如果共享原子）"""
        num_bonds = bond_index.shape[1]
        
        # 创建原子到化学键的映射
        atom_to_bonds = {}
        for bond_idx, (i, j) in enumerate(bond_index.T.tolist()):
            if i not in atom_to_bonds:
                atom_to_bonds[i] = []
            if j not in atom_to_bonds:
                atom_to_bonds[j] = []
            atom_to_bonds[i].append(bond_idx)
            atom_to_bonds[j].append(bond_idx)
        
        # 构建键边
        bond_edges = []
        for atom, bonds in atom_to_bonds.items():
            for i in range(len(bonds)):
                for j in range(i + 1, len(bonds)):
                    bond_edges.append([bonds[i], bonds[j]])
                    bond_edges.append([bonds[j], bonds[i]])
        
        if len(bond_edges) == 0:
            # 如果没有共享原子的键，创建自环
            return torch.stack([
                torch.arange(num_bonds),
                torch.arange(num_bonds)
            ], dim=0)
        
        return torch.tensor(bond_edges, dtype=torch.long, device=bond_index.device).T


# ============== 晶胞级别层 ==============

class CellLevel(nn.Module):
    """晶胞级别表示学习"""
    
    def __init__(self, config: HierarchicalConfig):
        super().__init__()
        self.config = config
        
        input_dim = config.atom_hidden_dim + config.bond_hidden_dim
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, config.cell_hidden_dim),
            nn.LayerNorm(config.cell_hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout)
        )
        
        # Transformer层处理晶胞间关系
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.cell_hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.cell_hidden_dim * 4,
            dropout=config.attention_dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.cell_num_layers
        )
        
        # 晶胞读出
        self.cell_readout = nn.Sequential(
            nn.Linear(config.cell_hidden_dim, config.cell_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.cell_hidden_dim, config.cell_hidden_dim)
        )
    
    def forward(self, atom_level_features: torch.Tensor,
                bond_level_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            atom_level_features: [B, atom_hidden_dim] 原子级特征
            bond_level_features: [B, bond_hidden_dim] 化学键级特征
        
        Returns:
            cell_features: [B, cell_hidden_dim] 晶胞级特征
        """
        # 融合多尺度特征
        combined = torch.cat([atom_level_features, bond_level_features], dim=-1)
        features = self.fusion(combined)
        
        # 添加batch维度用于Transformer
        # [B, hidden_dim] -> [1, B, hidden_dim] (假设单个batch内处理)
        features = features.unsqueeze(0)
        
        # Transformer处理
        features = self.transformer(features)
        
        # 移除batch维度
        features = features.squeeze(0)
        
        # 读出
        cell_features = self.cell_readout(features)
        
        return cell_features


# ============== 完整模型 ==============

class HierarchicalGNN(nn.Module):
    """
    层次GNN模型
    整合原子、化学键、晶胞三个层次的表示
    """
    
    def __init__(self, config: HierarchicalConfig = None):
        super().__init__()
        self.config = config or HierarchicalConfig()
        
        # 三个层次
        self.atom_level = AtomLevel(self.config)
        self.bond_level = BondLevel(self.config)
        self.cell_level = CellLevel(self.config)
        
        # 跨层注意力
        self.cross_attention = CrossLevelAttention(
            self.config.atom_hidden_dim,
            self.config.bond_hidden_dim,
            self.config.cell_hidden_dim
        )
        
        # 输出头
        total_dim = (
            self.config.atom_hidden_dim + 
            self.config.bond_hidden_dim + 
            self.config.cell_hidden_dim
        )
        
        self.energy_head = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(total_dim // 2, 1)
        )
        
        self.property_head = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(total_dim // 2, 32)  # 多任务属性预测
        )
    
    def forward(self, data: Data, return_hierarchy: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            data: PyG Data对象
            return_hierarchy: 是否返回所有层次的特征
        
        Returns:
            预测结果字典
        """
        # 原子级别
        atom_features, atom_embeddings, graph_atom_features = self.atom_level(
            data.atomic_numbers, data.pos, data.batch
        )
        
        # 化学键级别
        bond_features, graph_bond_features = self.bond_level(
            atom_embeddings, data.pos, data.atomic_numbers, data.batch,
            getattr(data, 'bond_index', None)
        )
        
        # 晶胞级别
        cell_features = self.cell_level(graph_atom_features, graph_bond_features)
        
        # 跨层注意力
        fused_features = self.cross_attention(
            graph_atom_features, graph_bond_features, cell_features
        )
        
        # 能量预测
        energy = self.energy_head(fused_features).squeeze(-1)
        
        # 属性预测
        properties = self.property_head(fused_features)
        
        output = {
            'energy': energy,
            'properties': properties,
            'fused_features': fused_features
        }
        
        if return_hierarchy:
            output['hierarchy'] = {
                'atom_features': atom_features,
                'atom_embeddings': atom_embeddings,
                'graph_atom_features': graph_atom_features,
                'bond_features': bond_features,
                'graph_bond_features': graph_bond_features,
                'cell_features': cell_features
            }
        
        return output
    
    def get_atom_attention(self, data: Data, target_level: str = 'bond') -> torch.Tensor:
        """
        获取原子级别的注意力权重
        用于可视化哪些原子对特定层次最重要
        
        Args:
            data: 输入数据
            target_level: 'bond' 或 'cell'
        """
        with torch.no_grad():
            output = self.forward(data, return_hierarchy=True)
            
            # 计算原子对晶胞级预测的贡献
            atom_features = output['hierarchy']['atom_embeddings']
            
            # 简单的注意力权重（特征范数）
            attention_weights = torch.norm(atom_features, dim=-1)
            
            # 归一化到每个晶胞
            attention_weights = attention_weights / scatter_add(
                attention_weights, data.batch, dim=0
            )[data.batch]
        
        return attention_weights


class CrossLevelAttention(nn.Module):
    """跨层注意力 - 融合不同层次的特征"""
    
    def __init__(self, atom_dim: int, bond_dim: int, cell_dim: int):
        super().__init__()
        
        # 投影到相同维度
        self.atom_proj = nn.Linear(atom_dim, cell_dim)
        self.bond_proj = nn.Linear(bond_dim, cell_dim)
        
        # 自注意力融合
        self.attention = nn.MultiheadAttention(cell_dim, num_heads=4, batch_first=True)
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(cell_dim * 3, cell_dim),
            nn.LayerNorm(cell_dim),
            nn.SiLU()
        )
    
    def forward(self, atom_features: torch.Tensor,
                bond_features: torch.Tensor,
                cell_features: torch.Tensor) -> torch.Tensor:
        """
        跨层注意力融合
        
        Args:
            atom_features: [B, atom_dim]
            bond_features: [B, bond_dim]
            cell_features: [B, cell_dim]
        """
        # 投影
        atom_proj = self.atom_proj(atom_features)
        bond_proj = self.bond_proj(bond_features)
        
        # 堆叠用于注意力
        # [B, 3, cell_dim]
        features = torch.stack([atom_proj, bond_proj, cell_features], dim=1)
        
        # 自注意力
        attended, _ = self.attention(features, features, features)
        
        # 拼接原始和注意力特征
        combined = torch.cat([
            atom_proj, bond_proj, cell_features,
            attended.mean(dim=1)  # 池化后的注意力特征
        ], dim=-1)
        
        # 简化：直接拼接所有特征
        all_features = torch.cat([atom_features, bond_features, cell_features], dim=-1)
        
        return all_features


# ============== 辅助模块 ==============

class GaussianBasis(nn.Module):
    """高斯基函数"""
    
    def __init__(self, num_basis: int = 50, cutoff: float = 10.0):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff
        
        centers = torch.linspace(0, cutoff, num_basis)
        self.register_buffer('centers', centers)
        self.widths = nn.Parameter(torch.ones(num_basis) * 0.5)
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        distances = distances.unsqueeze(-1)
        gamma = 1.0 / (2 * self.widths ** 2)
        rbf = torch.exp(-gamma * (distances - self.centers) ** 2)
        
        # 平滑截断
        cutoff_val = 0.5 * (torch.cos(np.pi * distances / self.cutoff) + 1.0)
        cutoff_val = cutoff_val * (distances < self.cutoff).float()
        
        return rbf * cutoff_val


# ============== 使用示例 ==============

def example_hierarchical_gnn():
    """层次GNN使用示例"""
    
    print("=" * 60)
    print("层次GNN示例")
    print("=" * 60)
    
    # 配置
    config = HierarchicalConfig(
        atom_hidden_dim=128,
        bond_hidden_dim=64,
        cell_hidden_dim=256,
        atom_num_layers=4,
        bond_num_layers=2,
        cell_num_layers=2
    )
    
    # 创建模型
    model = HierarchicalGNN(config)
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建示例数据（批量）
    batch_size = 4
    total_atoms = 0
    atomic_numbers_list = []
    pos_list = []
    batch_list = []
    
    for b in range(batch_size):
        num_atoms = torch.randint(10, 30, (1,)).item()
        atomic_numbers_list.append(torch.randint(1, 30, (num_atoms,)))
        pos_list.append(torch.randn(num_atoms, 3) * 10)
        batch_list.append(torch.full((num_atoms,), b, dtype=torch.long))
        total_atoms += num_atoms
    
    data = Data(
        atomic_numbers=torch.cat(atomic_numbers_list),
        pos=torch.cat(pos_list),
        batch=torch.cat(batch_list)
    )
    
    print(f"\n批量数据:")
    print(f"  总原子数: {total_atoms}")
    print(f"  晶胞数: {batch_size}")
    
    # 前向传播
    output = model(data, return_hierarchy=True)
    
    print(f"\n输出:")
    print(f"  能量形状: {output['energy'].shape}")
    print(f"  能量值: {output['energy'].tolist()}")
    print(f"  属性预测形状: {output['properties'].shape}")
    
    # 层次特征
    hierarchy = output['hierarchy']
    print(f"\n层次特征:")
    print(f"  原子特征: {hierarchy['atom_features'].shape}")
    print(f"  图原子特征: {hierarchy['graph_atom_features'].shape}")
    print(f"  化学键特征: {hierarchy['bond_features'].shape}")
    print(f"  图化学键特征: {hierarchy['graph_bond_features'].shape}")
    print(f"  晶胞特征: {hierarchy['cell_features'].shape}")
    
    # 注意力可视化
    attention = model.get_atom_attention(data)
    print(f"\n原子注意力权重:")
    print(f"  形状: {attention.shape}")
    print(f"  范围: [{attention.min():.4f}, {attention.max():.4f}]")
    
    print("\n" + "=" * 60)
    print("层次GNN示例完成！")
    print("=" * 60)
    
    return model, output


if __name__ == "__main__":
    example_hierarchical_gnn()
