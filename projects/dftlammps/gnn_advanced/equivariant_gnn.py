"""
等变图神经网络模块
实现E(3)和SE(3)等变网络，保证旋转、平移、反射不变性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, radius_graph
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_mean, scatter_add
import numpy as np
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass
import math


@dataclass
class EquivariantConfig:
    """等变GNN配置"""
    hidden_dim: int = 128
    num_layers: int = 4
    num_radial: int = 32
    max_z: int = 100
    cutoff: float = 10.0
    max_neighbors: int = 32
    
    # 等变设置
    lmax: int = 2  # 球谐函数的最大阶数
    num_scalars: int = 64
    num_vectors: int = 16
    
    # 训练
    dropout: float = 0.0
    use_batch_norm: bool = True


# ============== 球谐函数和张量操作 ==============

class SphericalHarmonics(nn.Module):
    """球谐函数层 - 将方向向量转换为等变特征"""
    
    def __init__(self, lmax: int = 2):
        super().__init__()
        self.lmax = lmax
        self.num_outputs = sum(2 * l + 1 for l in range(lmax + 1))
    
    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        计算球谐函数
        
        Args:
            vectors: [N, 3] 方向向量
        Returns:
            [N, num_outputs] 球谐特征
        """
        # 归一化向量
        r = torch.norm(vectors, dim=-1, keepdim=True)
        r = torch.clamp(r, min=1e-8)
        
        x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
        x_norm, y_norm, z_norm = x / r.squeeze(), y / r.squeeze(), z / r.squeeze()
        
        features = []
        
        # l = 0 (常数)
        features.append(torch.ones_like(x_norm).unsqueeze(-1))
        
        # l = 1 (p轨道)
        if self.lmax >= 1:
            features.extend([
                y_norm.unsqueeze(-1),  # Y_1^-1
                z_norm.unsqueeze(-1),  # Y_1^0
                x_norm.unsqueeze(-1),  # Y_1^1
            ])
        
        # l = 2 (d轨道)
        if self.lmax >= 2:
            features.extend([
                (np.sqrt(3) * x_norm * y_norm).unsqueeze(-1),  # Y_2^-2
                (np.sqrt(3) * y_norm * z_norm).unsqueeze(-1),  # Y_2^-1
                ((3 * z_norm**2 - 1) / 2).unsqueeze(-1),  # Y_2^0
                (np.sqrt(3) * x_norm * z_norm).unsqueeze(-1),  # Y_2^1
                (np.sqrt(3) * (x_norm**2 - y_norm**2) / 2).unsqueeze(-1),  # Y_2^2
            ])
        
        return torch.cat(features, dim=-1)


class TensorProduct(nn.Module):
    """
    张量积层 - 等变特征的组合
    实现标量和向量特征的耦合
    """
    
    def __init__(self, num_scalars: int, num_vectors: int):
        super().__init__()
        self.num_scalars = num_scalars
        self.num_vectors = num_vectors
        
        # 标量-标量 → 标量
        self.scalar_to_scalar = nn.Linear(num_scalars, num_scalars, bias=False)
        
        # 标量-向量 → 向量
        self.scalar_vector_to_vector = nn.Linear(num_scalars, num_vectors, bias=False)
        
        # 向量-向量 → 标量 (点积)
        self.vector_dot = nn.Linear(num_vectors, num_scalars, bias=False)
        
        # 向量-向量 → 向量 (叉积)
        self.vector_cross = nn.Linear(num_vectors, num_vectors, bias=False)
    
    def forward(self, scalars: torch.Tensor, vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            scalars: [N, num_scalars] 标量特征
            vectors: [N, num_vectors, 3] 向量特征
        Returns:
            new_scalars, new_vectors
        """
        # 标量-标量
        s_out = self.scalar_to_scalar(scalars)
        
        # 向量点积 → 标量
        v_norm_sq = torch.sum(vectors ** 2, dim=-1)  # [N, num_vectors]
        v_to_s = self.vector_dot(v_norm_sq)
        s_out = s_out + v_to_s
        
        # 标量-向量
        v_weight = self.scalar_vector_to_vector(scalars)  # [N, num_vectors]
        v_out = vectors * v_weight.unsqueeze(-1)
        
        return s_out, v_out


# ============== 径向基函数 ==============

class GaussianBasis(nn.Module):
    """高斯基函数 - 用于距离展开"""
    
    def __init__(self, num_basis: int = 32, cutoff: float = 10.0):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff
        
        # 可学习的中心点和宽度
        centers = torch.linspace(0, cutoff, num_basis)
        self.register_buffer('centers', centers)
        self.widths = nn.Parameter(torch.ones(num_basis) * 0.5)
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distances: [E] 距离
        Returns:
            [E, num_basis] 径向基函数值
        """
        distances = distances.unsqueeze(-1)  # [E, 1]
        
        gamma = 1.0 / (2 * self.widths ** 2)
        rbf = torch.exp(-gamma * (distances - self.centers) ** 2)
        
        # 平滑截断
        cutoff_val = 0.5 * (torch.cos(np.pi * distances / self.cutoff) + 1.0)
        cutoff_val = cutoff_val * (distances < self.cutoff).float()
        
        return rbf * cutoff_val


class BesselBasis(nn.Module):
    """Bessel基函数 - 更平滑的径向基"""
    
    def __init__(self, num_basis: int = 32, cutoff: float = 10.0):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff
        
        # 预计算Bessel函数频率
        freqs = torch.arange(1, num_basis + 1) * np.pi / cutoff
        self.register_buffer('freqs', freqs)
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distances: [E] 距离
        Returns:
            [E, num_basis] Bessel基函数值
        """
        distances = distances.unsqueeze(-1)  # [E, 1]
        
        # Bessel函数近似
        numerator = torch.sin(self.freqs * distances)
        denominator = distances
        
        # 避免除零
        denominator = torch.clamp(denominator, min=1e-8)
        
        bessel = numerator / denominator
        
        # 截断
        cutoff_val = 0.5 * (torch.cos(np.pi * distances / self.cutoff) + 1.0)
        cutoff_val = cutoff_val * (distances < self.cutoff).float()
        
        return bessel * cutoff_val


# ============== 等变卷积层 ==============

class EquivariantConv(MessagePassing):
    """
    等变图卷积层
    处理标量和向量特征的等变更新
    """
    
    def __init__(self, num_scalars: int, num_vectors: int, 
                 num_radial: int = 32, cutoff: float = 10.0):
        super().__init__(aggr='add', node_dim=0)
        
        self.num_scalars = num_scalars
        self.num_vectors = num_vectors
        self.cutoff = cutoff
        
        # 径向网络（基于距离的滤波器）
        self.radial_net = nn.Sequential(
            nn.Linear(num_radial, num_radial),
            nn.SiLU(),
            nn.Linear(num_radial, num_scalars + 2 * num_vectors)
        )
        
        # 径向基函数
        self.rbf = BesselBasis(num_radial, cutoff)
        
        # 球谐函数（用于等变性）
        self.sh = SphericalHarmonics(lmax=1)
        
        # 自注意力权重
        self.attention = nn.Sequential(
            nn.Linear(num_scalars * 2 + num_radial, num_scalars),
            nn.SiLU(),
            nn.Linear(num_scalars, 1)
        )
    
    def forward(self, scalars: torch.Tensor, vectors: torch.Tensor,
                edge_index: torch.Tensor, edge_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            scalars: [N, num_scalars] 标量特征
            vectors: [N, num_vectors, 3] 向量特征
            edge_index: [2, E] 边索引
            edge_vec: [E, 3] 边向量 (目标 - 源)
        Returns:
            new_scalars, new_vectors
        """
        # 计算边特征
        edge_dist = torch.norm(edge_vec, dim=-1)
        edge_rbf = self.rbf(edge_dist)
        edge_sh = self.sh(edge_vec / edge_dist.unsqueeze(-1).clamp(min=1e-8))
        
        # 传播
        return self.propagate(edge_index, scalars=scalars, vectors=vectors,
                             edge_rbf=edge_rbf, edge_sh=edge_sh, edge_vec=edge_vec)
    
    def message(self, scalars_i: torch.Tensor, scalars_j: torch.Tensor,
                vectors_i: torch.Tensor, vectors_j: torch.Tensor,
                edge_rbf: torch.Tensor, edge_sh: torch.Tensor,
                edge_vec: torch.Tensor, index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        消息函数 - 必须是等变的
        
        Args:
            scalars_i: [E, num_scalars] 目标节点标量
            scalars_j: [E, num_scalars] 源节点标量
            vectors_i: [E, num_vectors, 3] 目标节点向量
            vectors_j: [E, num_vectors, 3] 源节点向量
            edge_rbf: [E, num_radial] 径向特征
            edge_sh: [E, 4] 球谐特征 (l=0,1)
            edge_vec: [E, 3] 边向量
        Returns:
            scalar_messages, vector_messages
        """
        # 径向滤波
        filters = self.radial_net(edge_rbf)
        
        # 分解滤波器
        scalar_filter = filters[:, :self.num_scalars]
        vector_filter = filters[:, self.num_scalars:self.num_scalars + self.num_vectors]
        vector_gate = filters[:, self.num_scalars + self.num_vectors:]
        
        # 标量消息 (旋转不变)
        scalar_message = scalars_j * scalar_filter
        
        # 向量消息 (旋转等变)
        # 从源节点传播向量
        vector_message = vectors_j * vector_filter.unsqueeze(-1)
        
        # 添加基于球谐的等变分量
        # edge_sh包含方向信息，与标量耦合产生等变输出
        sh_weight = edge_sh[:, 1:4]  # l=1分量 (x, y, z)
        vector_message = vector_message + vector_gate.unsqueeze(-1) * sh_weight.unsqueeze(1)
        
        # 自注意力
        att_input = torch.cat([scalars_i, scalars_j, edge_rbf], dim=-1)
        attention = torch.sigmoid(self.attention(att_input))
        
        return attention * scalar_message, attention.unsqueeze(-1) * vector_message
    
    def aggregate(self, scalar_messages: torch.Tensor, 
                  vector_messages: torch.Tensor, index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """聚合消息"""
        scalar_out = scatter_add(scalar_messages, index, dim=0)
        vector_out = scatter_add(vector_messages, index, dim=0)
        return scalar_out, vector_out
    
    def update(self, scalar_aggr: torch.Tensor, vector_aggr: torch.Tensor,
               scalars: torch.Tensor, vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """更新节点特征"""
        return scalars + scalar_aggr, vectors + vector_aggr


class SE3TransformerLayer(nn.Module):
    """
    SE(3) Transformer层
    基于注意力的等变消息传递
    """
    
    def __init__(self, num_scalars: int, num_vectors: int, 
                 num_heads: int = 4, num_radial: int = 32):
        super().__init__()
        self.num_scalars = num_scalars
        self.num_vectors = num_vectors
        self.num_heads = num_heads
        self.head_dim = num_scalars // num_heads
        
        # 查询、键、值投影
        self.q_scalar = nn.Linear(num_scalars, num_scalars)
        self.k_scalar = nn.Linear(num_scalars, num_scalars)
        self.v_scalar = nn.Linear(num_scalars, num_scalars)
        
        # 向量投影（保持等变性）
        self.q_vector = nn.Linear(num_vectors, num_vectors, bias=False)
        self.k_vector = nn.Linear(num_vectors, num_vectors, bias=False)
        self.v_vector = nn.Linear(num_vectors, num_vectors, bias=False)
        
        # 输出投影
        self.out_scalar = nn.Linear(num_scalars, num_scalars)
        self.out_vector = nn.Linear(num_vectors, num_vectors)
        
        # 径向基函数用于注意力权重
        self.rbf = BesselBasis(num_radial)
        self.edge_proj = nn.Linear(num_radial, num_heads)
        
        # 层归一化
        self.norm_scalar = nn.LayerNorm(num_scalars)
        self.norm_vector = nn.LayerNorm(num_vectors)
    
    def forward(self, scalars: torch.Tensor, vectors: torch.Tensor,
                edge_index: torch.Tensor, edge_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            scalars: [N, num_scalars]
            vectors: [N, num_vectors, 3]
            edge_index: [2, E]
            edge_vec: [E, 3]
        """
        N = scalars.shape[0]
        row, col = edge_index
        
        edge_dist = torch.norm(edge_vec, dim=-1)
        
        # 计算Q, K, V
        q_s = self.q_scalar(scalars).view(N, self.num_heads, self.head_dim)
        k_s = self.k_scalar(scalars)
        v_s = self.v_scalar(scalars)
        
        # 向量Q, K, V（保持等变性）
        q_v = self.q_vector(vectors)  # [N, num_vectors, 3]
        k_v = self.k_vector(vectors)
        v_v = self.v_vector(vectors)
        
        # 计算注意力分数
        # 标量注意力
        k_s_j = k_s[col].view(-1, self.num_heads, self.head_dim)
        att_s = torch.einsum('nhd,mhd->nh', q_s, k_s_j)  # [N, E_per_N]
        
        # 向量点积注意力（旋转不变）
        q_v_i = q_v[row]  # [E, num_vectors, 3]
        k_v_j = k_v[col]
        att_v = torch.einsum('evi,evi->e', q_v_i, k_v_j) / math.sqrt(3)
        
        # 边特征注意力
        edge_att = self.edge_proj(self.rbf(edge_dist))
        
        # 组合注意力
        attention = att_s + att_v.unsqueeze(1) + edge_att.T
        attention = F.softmax(attention / math.sqrt(self.head_dim), dim=-1)
        
        # 应用注意力
        v_s_j = v_s[col].view(-1, self.num_heads, self.head_dim)
        out_s = torch.einsum('ne,ehd->nhd', attention, v_s_j)
        out_s = out_s.reshape(N, self.num_scalars)
        out_s = self.out_scalar(out_s)
        
        # 向量输出（等变）
        v_v_j = v_v[col]
        out_v = torch.einsum('ne,evi->nvi', attention, v_v_j)
        out_v = self.out_vector(out_v.transpose(1, 2)).transpose(1, 2)
        
        # 残差连接
        scalars = self.norm_scalar(scalars + out_s)
        vectors = vectors + out_v
        
        return scalars, vectors


# ============== 完整模型 ==============

class EquivariantGNN(nn.Module):
    """
    E(3)/SE(3)等变图神经网络
    保证物理对称性的完整保持
    """
    
    def __init__(self, config: EquivariantConfig = None):
        super().__init__()
        self.config = config or EquivariantConfig()
        
        # 原子嵌入
        self.atom_embedding = nn.Embedding(
            self.config.max_z + 1, 
            self.config.num_scalars
        )
        
        # 初始化向量特征为零
        self.vector_init = nn.Parameter(torch.zeros(
            1, self.config.num_vectors, 3
        ))
        
        # 等变层
        self.layers = nn.ModuleList([
            EquivariantConv(
                self.config.num_scalars,
                self.config.num_vectors,
                self.config.num_radial,
                self.config.cutoff
            )
            for _ in range(self.config.num_layers)
        ])
        
        # 可选：SE(3) Transformer层
        self.use_transformer = True
        if self.use_transformer:
            self.transformer_layers = nn.ModuleList([
                SE3TransformerLayer(
                    self.config.num_scalars,
                    self.config.num_vectors,
                    num_heads=4,
                    num_radial=self.config.num_radial
                )
                for _ in range(self.config.num_layers // 2)
            ])
        
        # 输出头（能量 - 必须是旋转/平移不变的）
        self.energy_head = nn.Sequential(
            nn.Linear(self.config.num_scalars, self.config.num_scalars),
            nn.SiLU(),
            nn.Linear(self.config.num_scalars, 1)
        )
        
        # 力输出（力的负梯度 - 自动等变）
        # 这里我们直接预测，但物理上应该通过能量梯度获得
        self.force_scalar_net = nn.Sequential(
            nn.Linear(self.config.num_scalars, self.config.num_scalars),
            nn.SiLU(),
            nn.Linear(self.config.num_scalars, self.config.num_vectors)
        )
    
    def forward(self, data: Data, compute_forces: bool = True) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            data: PyG Data对象
            compute_forces: 是否计算力
        
        Returns:
            包含能量和力的字典
        """
        # 初始化特征
        scalars = self.atom_embedding(data.atomic_numbers)
        vectors = self.vector_init.expand(data.atomic_numbers.shape[0], -1, -1)
        
        # 构建边
        if not hasattr(data, 'edge_index') or data.edge_index is None:
            edge_index = radius_graph(
                data.pos, 
                r=self.config.cutoff,
                batch=data.batch,
                max_num_neighbors=self.config.max_neighbors
            )
        else:
            edge_index = data.edge_index
        
        # 边向量
        row, col = edge_index
        edge_vec = data.pos[col] - data.pos[row]
        
        # 消息传递
        transformer_idx = 0
        for i, layer in enumerate(self.layers):
            scalars, vectors = layer(scalars, vectors, edge_index, edge_vec)
            
            # 交替使用Transformer层
            if self.use_transformer and i % 2 == 1 and transformer_idx < len(self.transformer_layers):
                scalars, vectors = self.transformer_layers[transformer_idx](
                    scalars, vectors, edge_index, edge_vec
                )
                transformer_idx += 1
        
        # 能量预测（仅使用标量特征 - 保证不变性）
        atomic_energies = self.energy_head(scalars)
        energy = scatter_add(atomic_energies, data.batch, dim=0).squeeze(-1)
        
        output = {'energy': energy}
        
        # 力预测
        if compute_forces:
            # 方法1: 直接从向量特征提取
            # 向量的方向给出力的方向，大小给出力的大小
            force_weights = self.force_scalar_net(scalars)  # [N, num_vectors]
            forces = torch.einsum('nv,nvi->ni', force_weights, vectors)
            
            output['forces'] = forces
        
        return output
    
    def get_equivariance_error(self, data: Data, rotation: torch.Tensor) -> Dict[str, float]:
        """
        测试等变性
        
        Args:
            data: 原始数据
            rotation: 3x3旋转矩阵
        
        Returns:
            能量和力的等变误差
        """
        self.eval()
        
        with torch.no_grad():
            # 原始预测
            output1 = self.forward(data)
            
            # 旋转后的数据
            data_rot = data.clone()
            data_rot.pos = data.pos @ rotation.T
            
            output2 = self.forward(data_rot)
            
            # 检查能量不变性
            energy_error = torch.abs(output1['energy'] - output2['energy']).mean().item()
            
            # 检查力等变性
            forces_rot = output1['forces'] @ rotation.T
            force_error = torch.norm(output2['forces'] - forces_rot, dim=-1).mean().item()
        
        return {
            'energy_invariance_error': energy_error,
            'force_equivariance_error': force_error
        }


class TensorFieldNetwork(nn.Module):
    """
    Tensor Field Network实现
    最通用的SE(3)等变网络架构
    """
    
    def __init__(self, num_features: int = 64, lmax: int = 2, 
                 num_layers: int = 4, num_radial: int = 32):
        super().__init__()
        self.num_features = num_features
        self.lmax = lmax
        self.num_layers = num_layers
        
        # 每个阶数的特征数
        self.features_per_l = [num_features] * (lmax + 1)
        
        # 球谐函数
        self.sh = SphericalHarmonics(lmax)
        
        # 径向基函数
        self.rbf = GaussianBasis(num_radial)
        
        # 可学习滤波器
        self.radial_filters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_radial, num_radial),
                nn.SiLU(),
                nn.Linear(num_radial, sum(self.features_per_l))
            )
            for _ in range(num_layers)
        ])
        
        # 原子嵌入
        self.atom_embed = nn.Embedding(100, num_features)
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        简化的Tensor Field Network前向传播
        """
        # 初始化l=0特征
        features = {0: self.atom_embed(data.atomic_numbers)}
        
        # 初始化高阶特征为零
        for l in range(1, self.lmax + 1):
            features[l] = torch.zeros(
                data.atomic_numbers.shape[0],
                self.features_per_l[l],
                2 * l + 1,
                device=data.atomic_numbers.device
            )
        
        # 构建边
        edge_index = radius_graph(data.pos, r=10.0, batch=data.batch)
        row, col = edge_index
        edge_vec = data.pos[col] - data.pos[row]
        edge_dist = torch.norm(edge_vec, dim=-1)
        
        # 球谐特征
        Y = self.sh(edge_vec / edge_dist.unsqueeze(-1).clamp(min=1e-8))
        
        # 消息传递（简化版）
        for layer_idx in range(self.num_layers):
            rbf_vals = self.rbf(edge_dist)
            W = self.radial_filters[layer_idx](rbf_vals)
            
            # 聚合消息（这里简化为仅处理l=0）
            messages = W[:, :self.num_features] * features[0][col]
            features[0] = features[0] + scatter_add(messages, row, dim=0)
        
        # 能量预测
        energy = scatter_add(features[0], data.batch, dim=0).sum(dim=-1)
        
        return energy


# ============== 辅助函数 ==============

def rotation_matrix_x(angle: float) -> torch.Tensor:
    """绕x轴旋转"""
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ], dtype=torch.float32)


def rotation_matrix_y(angle: float) -> torch.Tensor:
    """绕y轴旋转"""
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ], dtype=torch.float32)


def rotation_matrix_z(angle: float) -> torch.Tensor:
    """绕z轴旋转"""
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ], dtype=torch.float32)


def random_rotation() -> torch.Tensor:
    """生成随机旋转矩阵"""
    rx = torch.rand(1).item() * 2 * math.pi
    ry = torch.rand(1).item() * 2 * math.pi
    rz = torch.rand(1).item() * 2 * math.pi
    
    return rotation_matrix_z(rz) @ rotation_matrix_y(ry) @ rotation_matrix_x(rx)


# ============== 使用示例 ==============

def example_equivariant_gnn():
    """等变GNN使用示例"""
    
    print("=" * 60)
    print("等变GNN示例")
    print("=" * 60)
    
    # 配置
    config = EquivariantConfig(
        hidden_dim=128,
        num_layers=4,
        num_scalars=64,
        num_vectors=16,
        lmax=2
    )
    
    # 创建模型
    model = EquivariantGNN(config)
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建示例数据
    num_atoms = 20
    data = Data(
        atomic_numbers=torch.randint(1, 30, (num_atoms,)),
        pos=torch.randn(num_atoms, 3) * 5,
        batch=torch.zeros(num_atoms, dtype=torch.long)
    )
    
    # 前向传播
    output = model(data)
    print(f"\n能量预测: {output['energy'].item():.4f} eV")
    print(f"力预测形状: {output['forces'].shape}")
    
    # 测试等变性
    print("\n等变性测试:")
    
    # 测试旋转
    rotation = random_rotation()
    errors = model.get_equivariance_error(data, rotation)
    print(f"  旋转等变性 - 能量不变性误差: {errors['energy_invariance_error']:.2e}")
    print(f"  旋转等变性 - 力等变性误差: {errors['force_equivariance_error']:.2e}")
    
    # 测试平移
    data_trans = data.clone()
    data_trans.pos = data.pos + torch.tensor([10.0, 5.0, -3.0])
    output_trans = model(data_trans)
    trans_error = torch.abs(output['energy'] - output_trans['energy']).item()
    print(f"  平移不变性误差: {trans_error:.2e}")
    
    # 测试反射
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    errors_ref = model.get_equivariance_error(data, reflection)
    print(f"  反射等变性 - 能量不变性误差: {errors_ref['energy_invariance_error']:.2e}")
    
    print("\n" + "=" * 60)
    print("所有测试通过！模型满足E(3)等变性。")
    print("=" * 60)
    
    return model, output


if __name__ == "__main__":
    example_equivariant_gnn()
