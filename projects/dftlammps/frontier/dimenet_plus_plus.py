"""
dimenet_plus_plus.py
DimeNet++方向消息传递神经网络

DimeNet++是2021年提出的用于分子性质预测的先进图神经网络,
使用方向消息传递捕获角度信息, 在QM9等数据集上达到SOTA。
2024年扩展应用于周期性材料和势函数任务。

References:
- Gasteiger et al. (2020) "Directional Message Passing for Molecular Graphs"
- Gasteiger et al. (2021) "Fast and Uncertainty-Aware Directional Message Passing"
- 2024进展: DimeNet++用于周期性材料和势函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import numpy as np
import math


@dataclass
class MolecularGraph:
    """分子图数据结构"""
    atomic_numbers: torch.Tensor   # [n_atoms]
    positions: torch.Tensor        # [n_atoms, 3]
    edge_index: torch.Tensor       # [2, n_edges]
    edge_directions: torch.Tensor  # [n_edges, 3] 单位方向向量
    edge_distances: torch.Tensor   # [n_edges]
    triplets: torch.Tensor         # [n_triplets, 3] (j->i, i->k, j->k) 索引
    triplet_angles: torch.Tensor   # [n_triplets] 角度
    batch: Optional[torch.Tensor] = None


class SphericalBasisLayer(nn.Module):
    """
    球谐基函数层
    
    将边方向编码为球谐函数特征
    """
    
    def __init__(self, num_spherical: int = 7, num_radial: int = 6):
        super().__init__()
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        
    def forward(self, directions: torch.Tensor) -> torch.Tensor:
        """
        计算球谐特征
        
        Args:
            directions: [n_edges, 3] 单位方向向量 (x, y, z)
        Returns:
            [n_edges, num_spherical] 球谐特征
        """
        x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
        
        features = []
        
        # l=0 (s轨道)
        features.append(torch.ones_like(x).unsqueeze(-1))
        
        # l=1 (p轨道)
        if self.num_spherical >= 3:
            features.append(torch.stack([y, z, x], dim=-1))
        
        # l=2 (d轨道) - 简化版
        if self.num_spherical >= 6:
            features.append(torch.stack([
                x * y,  # xy
                y * z,  # yz
                3 * z**2 - 1,  # 3z²-1
                x * z,  # xz
                x**2 - y**2  # x²-y²
            ], dim=-1))
        
        result = torch.cat(features, dim=-1)
        return result[:, :self.num_spherical]


class BesselBasisLayer(nn.Module):
    """Bessel基函数层 - 径向特征"""
    
    def __init__(
        self,
        num_radial: int = 16,
        cutoff: float = 5.0,
        envelope_exponent: int = 5
    ):
        super().__init__()
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope_exponent = envelope_exponent
        
        # Bessel函数频率
        freq = torch.arange(1, num_radial + 1) * np.pi / cutoff
        self.register_buffer('freq', freq)
    
    def envelope(self, d: torch.Tensor) -> torch.Tensor:
        """多项式包络函数"""
        p = self.envelope_exponent + 1
        a = -(p + 1) * (p + 2) / 2
        b = p * (p + 2)
        c = -p * (p + 1) / 2
        
        x = d / self.cutoff
        return (1 + a * x**p + b * x**(p + 1) + c * x**(p + 2)) * (d < self.cutoff).float()
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        计算Bessel基特征
        
        Args:
            distances: [n_edges] 距离
        Returns:
            [n_edges, num_radial] Bessel特征
        """
        d = distances.unsqueeze(-1)
        
        # 归一化Bessel函数
        bessel = torch.sqrt(2 / self.cutoff) * torch.sin(self.freq * d) / d
        
        # 应用包络
        return bessel * self.envelope(distances).unsqueeze(-1)


class EmbeddingBlock(nn.Module):
    """初始嵌入块"""
    
    def __init__(
        self,
        num_elements: int = 100,
        hidden_channels: int = 128,
        num_radial: int = 16
    ):
        super().__init__()
        
        self.atom_embedding = nn.Embedding(num_elements, hidden_channels)
        
        # 边特征嵌入
        self.edge_proj = nn.Linear(num_radial, hidden_channels * 3)
        
    def forward(
        self,
        atomic_numbers: torch.Tensor,
        radial_basis: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        初始嵌入
        
        Returns:
            h: 节点特征
            m: 消息特征
            s: 源节点特征
            t: 目标节点特征
        """
        # 节点嵌入
        h = self.atom_embedding(atomic_numbers - 1)
        
        # 边嵌入
        edge_emb = self.edge_proj(radial_basis)
        s, t, m = torch.split(edge_emb, h.shape[-1], dim=-1)
        
        return h, m, s, t


class InteractionBlock(nn.Module):
    """
    DimeNet++交互块
    
    核心创新: 使用方向消息传递结合角度信息
    """
    
    def __init__(
        self,
        hidden_channels: int = 128,
        num_bilinear: int = 8,
        num_spherical: int = 7,
        num_radial: int = 16,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_dense_output: int = 3
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_bilinear = num_bilinear
        
        # 径向网络
        self.radial_net = nn.Sequential(
            nn.Linear(num_radial, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 球谐投影
        self.spherical_proj = nn.Linear(num_spherical, num_bilinear)
        
        # 消息转换
        self.msg_before_skip = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU()
            )
            for _ in range(num_before_skip)
        ])
        
        self.msg_after_skip = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU()
            )
            for _ in range(num_after_skip)
        ])
        
        # 更新网络
        self.update_net = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 输出投影
        self.output_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU()
            )
            for _ in range(num_dense_output)
        ])
        
        # 可学习的双线性组合
        self.bilinear = nn.Parameter(torch.randn(
            num_bilinear, hidden_channels, hidden_channels
        ))
        
    def forward(
        self,
        h: torch.Tensor,
        m: torch.Tensor,
        radial_basis: torch.Tensor,
        spherical_basis: torch.Tensor,
        edge_index: torch.Tensor,
        triplets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        交互块前向传播
        
        Args:
            h: [n_atoms, hidden] 节点特征
            m: [n_edges, hidden] 消息特征
            radial_basis: [n_edges, num_radial]
            spherical_basis: [n_edges, num_spherical]
            edge_index: [2, n_edges]
            triplets: [n_triplets, 3]
        """
        src, dst = edge_index
        
        # 径向过滤
        radial_filter = self.radial_net(radial_basis)
        
        # 方向过滤 (球谐)
        spherical_filter = self.spherical_proj(spherical_basis)
        
        # 通过三元组聚合消息
        # triplets: [j->i索引, i->k索引, j->k索引]
        if triplets.shape[0] > 0:
            idx_ji = triplets[:, 0]
            idx_kj = triplets[:, 1]
            
            # 获取kj边的消息
            m_kj = m[idx_kj]
            
            # 应用径向过滤
            m_kj_filtered = m_kj * radial_filter[idx_kj]
            
            # 应用方向过滤 (通过球谐)
            # 简化: 使用角度信息调制消息
            if spherical_filter.shape[1] > 0:
                angle_weight = torch.sigmoid(spherical_filter[idx_ji].mean(dim=-1, keepdim=True))
                m_kj_filtered = m_kj_filtered * angle_weight
        else:
            m_kj_filtered = torch.zeros_like(m)
        
        # 聚合消息到目标节点
        aggregated = torch.zeros_like(h)
        aggregated.index_add_(0, dst, m_kj_filtered)
        
        # 更新节点特征
        combined = torch.cat([h, aggregated], dim=-1)
        h_new = self.update_net(combined)
        
        # 残差连接
        h = h + h_new
        
        # 生成新的消息
        m_new = torch.zeros_like(m)
        for layer in self.output_proj:
            m_new = m_new + layer(m_kj_filtered)
        
        m = m + m_new
        
        return h, m


class OutputBlock(nn.Module):
    """输出块 - 预测目标性质"""
    
    def __init__(
        self,
        hidden_channels: int = 128,
        num_radial: int = 16,
        num_dense: int = 3,
        output_dim: int = 1
    ):
        super().__init__()
        
        # 径向网络
        self.radial_net = nn.Sequential(
            nn.Linear(num_radial, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 密集层
        self.dense_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU()
            )
            for _ in range(num_dense)
        ])
        
        # 最终输出
        self.output_layer = nn.Linear(hidden_channels, output_dim)
        
    def forward(
        self,
        h: torch.Tensor,
        radial_basis: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        计算原子级输出
        
        Args:
            h: [n_atoms, hidden] 节点特征
            radial_basis: [n_edges, num_radial]
            edge_index: [2, n_edges]
        """
        src, dst = edge_index
        
        # 径向过滤
        W = self.radial_net(radial_basis)
        
        # 传播到边
        h_src = h[src]
        h_edge = h_src * W
        
        # 聚合
        aggregated = torch.zeros(h.shape[0], h_edge.shape[1], device=h.device)
        aggregated.index_add_(0, dst, h_edge)
        
        # 密集层
        x = aggregated
        for layer in self.dense_layers:
            x = layer(x)
        
        return self.output_layer(x)


class DimeNetPlusPlus(nn.Module):
    """
    DimeNet++完整模型
    
    用于分子性质预测的端到端模型
    """
    
    def __init__(
        self,
        num_elements: int = 100,
        hidden_channels: int = 128,
        num_blocks: int = 4,
        num_bilinear: int = 8,
        num_spherical: int = 7,
        num_radial: int = 16,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        output_dim: int = 1,
        extensive: bool = False  # 是否预测广延量
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.cutoff = cutoff
        self.output_dim = output_dim
        self.extensive = extensive
        
        # 基函数层
        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial)
        
        # 嵌入块
        self.embedding = EmbeddingBlock(num_elements, hidden_channels, num_radial)
        
        # 交互块
        self.interaction_blocks = nn.ModuleList([
            InteractionBlock(
                hidden_channels=hidden_channels,
                num_bilinear=num_bilinear,
                num_spherical=num_spherical,
                num_radial=num_radial,
                num_before_skip=num_before_skip,
                num_after_skip=num_after_skip
            )
            for _ in range(num_blocks)
        ])
        
        # 输出块
        self.output_blocks = nn.ModuleList([
            OutputBlock(
                hidden_channels=hidden_channels,
                num_radial=num_radial,
                num_dense=num_output_layers,
                output_dim=output_dim
            )
            for _ in range(num_blocks + 1)
        ])
        
    def build_graph(
        self,
        positions: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        构建分子图
        
        Returns:
            edge_index: [2, n_edges]
            edge_directions: [n_edges, 3] 单位向量
            edge_distances: [n_edges]
            triplets: [n_triplets, 3]
        """
        n_atoms = positions.shape[0]
        device = positions.device
        
        # 计算距离矩阵
        dists = torch.cdist(positions, positions)
        
        # 掩码: 距离小于cutoff且不是自身
        mask = (dists < self.cutoff) & (dists > 0.01)
        
        # 获取边索引
        edge_index = torch.nonzero(mask, as_tuple=False).t()
        
        # 限制邻居数量
        if edge_index.shape[1] > n_atoms * self.max_num_neighbors:
            # 选择最近的邻居
            edge_dists = dists[mask]
            _, top_indices = torch.topk(
                edge_dists, 
                min(n_atoms * self.max_num_neighbors, edge_dists.shape[0]),
                largest=False
            )
            edge_index = edge_index[:, top_indices]
        
        # 计算边向量和距离
        src, dst = edge_index
        edge_vectors = positions[dst] - positions[src]
        edge_distances = torch.norm(edge_vectors, dim=-1)
        edge_directions = edge_vectors / edge_distances.unsqueeze(-1)
        
        # 构建三元组
        triplets = self._build_triplets(edge_index, n_atoms)
        
        return edge_index, edge_directions, edge_distances, triplets
    
    def _build_triplets(
        self,
        edge_index: torch.Tensor,
        n_atoms: int
    ) -> torch.Tensor:
        """
        构建三元组 (j->i->k)
        
        用于捕获角度信息
        """
        n_edges = edge_index.shape[1]
        device = edge_index.device
        
        # 找到共享中心节点的边对
        # edge_index[0] = 源节点, edge_index[1] = 目标节点
        
        triplets = []
        for i in range(n_atoms):
            # 找到所有j->i边 (i是目标)
            incoming = torch.where(edge_index[1] == i)[0]
            
            # 找到所有i->k边 (i是源)
            outgoing = torch.where(edge_index[0] == i)[0]
            
            # 组合三元组
            for idx_ji in incoming:
                j = edge_index[0][idx_ji]
                for idx_ik in outgoing:
                    k = edge_index[1][idx_ik]
                    if j != k:  # 避免自环
                        # 找到j->k边
                        mask = (edge_index[0] == j) & (edge_index[1] == k)
                        idx_jk = torch.where(mask)[0]
                        if len(idx_jk) > 0:
                            triplets.append([idx_ji.item(), idx_ik.item(), idx_jk[0].item()])
        
        if len(triplets) == 0:
            return torch.zeros((0, 3), dtype=torch.long, device=device)
        
        return torch.tensor(triplets, dtype=torch.long, device=device)
    
    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        DimeNet++前向传播
        
        Args:
            atomic_numbers: [n_atoms] 原子序数
            positions: [n_atoms, 3] 位置
            batch: [n_atoms] 批次索引
        """
        # 构建图
        edge_index, edge_directions, edge_distances, triplets = self.build_graph(
            positions, batch
        )
        
        # 计算基函数
        radial_basis = self.rbf(edge_distances)
        spherical_basis = self.sbf(edge_directions)
        
        # 初始嵌入
        h, m, s, t = self.embedding(atomic_numbers, radial_basis)
        
        # 初始化输出
        P = self.output_blocks[0](h, radial_basis, edge_index)
        
        # 交互块
        for i in range(self.num_blocks):
            h, m = self.interaction_blocks[i](
                h, m, radial_basis, spherical_basis,
                edge_index, triplets
            )
            P = P + self.output_blocks[i + 1](h, radial_basis, edge_index)
        
        # 全局聚合
        if self.extensive:
            # 求和 (广延量如能量)
            if batch is None:
                output = P.sum(dim=0, keepdim=True)
            else:
                num_graphs = batch.max().item() + 1
                output = torch.zeros(num_graphs, P.shape[1], device=P.device)
                output.index_add_(0, batch, P)
        else:
            # 平均 (强度量)
            if batch is None:
                output = P.mean(dim=0, keepdim=True)
            else:
                num_graphs = batch.max().item() + 1
                output = torch.zeros(num_graphs, P.shape[1], device=P.device)
                output.index_add_(0, batch, P)
                
                counts = torch.zeros(num_graphs, device=P.device)
                counts.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
                output = output / counts.unsqueeze(-1).clamp(min=1)
        
        return {
            'output': output,
            'atomic_outputs': P,
            'hidden': h
        }
    
    def predict_forces(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """
        预测原子力 (-dE/dr)
        """
        positions_grad = positions.requires_grad_(True)
        
        output = self.forward(atomic_numbers, positions_grad)
        energy = output['output'].sum()
        
        forces = -torch.autograd.grad(
            energy, positions_grad,
            create_graph=self.training
        )[0]
        
        return forces


class DimeNetPlusPlusPeriodic(DimeNetPlusPlus):
    """
    DimeNet++用于周期性材料
    
    扩展原模型支持晶胞和周期性边界条件
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def build_periodic_graph(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """构建周期性图"""
        n_atoms = positions.shape[0]
        device = positions.device
        
        edge_list = []
        vector_list = []
        
        # 生成周期镜像
        shifts = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    shift = torch.tensor([i, j, k], dtype=torch.float32, device=device)
                    shift = shift * pbc.float()
                    cart_shift = shift @ cell
                    shifts.append(cart_shift)
        
        shifts = torch.stack(shifts)
        
        # 寻找邻居
        for i in range(n_atoms):
            for shift in shifts:
                shifted_pos = positions + shift.unsqueeze(0)
                vectors = shifted_pos - positions[i:i+1]
                dists = torch.norm(vectors, dim=-1)
                
                mask = (dists < self.cutoff) & (dists > 0.01)
                neighbors = torch.where(mask)[0]
                
                for j in neighbors[:self.max_num_neighbors]:
                    edge_list.append([i, j.item()])
                    vector_list.append(vectors[j].cpu().numpy())
        
        if len(edge_list) == 0:
            return (
                torch.zeros((2, 0), dtype=torch.long, device=device),
                torch.zeros((0, 3), device=device),
                torch.zeros(0, device=device),
                torch.zeros((0, 3), dtype=torch.long, device=device)
            )
        
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t()
        edge_vectors = torch.tensor(vector_list, dtype=torch.float32, device=device)
        edge_distances = torch.norm(edge_vectors, dim=-1)
        edge_directions = edge_vectors / edge_distances.unsqueeze(-1)
        
        # 构建三元组
        triplets = self._build_triplets(edge_index, n_atoms)
        
        return edge_index, edge_directions, edge_distances, triplets


def train_dimenet_pp(
    model: DimeNetPlusPlus,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int = 300,
    lr: float = 1e-3,
    device: str = 'cuda'
) -> Dict[str, List[float]]:
    """训练DimeNet++模型"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50
    )
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            atomic_numbers = batch['atomic_numbers'].to(device)
            positions = batch['positions'].to(device)
            target = batch['target'].to(device)
            
            optimizer.zero_grad()
            output = model(atomic_numbers, positions)
            loss = criterion(output['output'], target)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                atomic_numbers = batch['atomic_numbers'].to(device)
                positions = batch['positions'].to(device)
                target = batch['target'].to(device)
                
                output = model(atomic_numbers, positions)
                loss = criterion(output['output'], target)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_dimenet_pp.pt')
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    return history


if __name__ == "__main__":
    print("=" * 60)
    print("DimeNet++ Demo - Directional Message Passing")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 创建模型
    print("\n1. Creating DimeNet++ model")
    model = DimeNetPlusPlus(
        num_elements=20,
        hidden_channels=64,
        num_blocks=2,
        num_spherical=5,
        num_radial=8,
        cutoff=5.0,
        output_dim=1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试分子
    print("\n2. Testing on H2O molecule")
    
    atomic_numbers = torch.tensor([8, 1, 1], device=device)
    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.96, 0.0, 0.0],
        [-0.24, 0.93, 0.0]
    ], device=device)
    
    model.eval()
    with torch.no_grad():
        output = model(atomic_numbers, positions)
    
    print(f"Predicted property: {output['output'].item():.4f}")
    print(f"Number of atoms: {len(atomic_numbers)}")
    
    # 测试力预测
    print("\n3. Testing force prediction")
    forces = model.predict_forces(atomic_numbers, positions)
    print(f"Forces shape: {forces.shape}")
    print(f"Forces:\n{forces}")
    
    # 周期性测试
    print("\n4. Testing periodic version")
    periodic_model = DimeNetPlusPlusPeriodic(
        num_elements=20,
        hidden_channels=32,
        num_blocks=1,
        cutoff=5.0
    ).to(device)
    
    # NaCl晶胞
    cell = torch.tensor([
        [5.64, 0, 0],
        [0, 5.64, 0],
        [0, 0, 5.64]
    ], device=device)
    pbc = torch.tensor([1, 1, 1], device=device)
    
    atomic_numbers_bulk = torch.tensor([11, 17], device=device)  # Na, Cl
    positions_bulk = torch.tensor([
        [0.0, 0.0, 0.0],
        [2.82, 2.82, 2.82]
    ], device=device)
    
    with torch.no_grad():
        edge_index, directions, distances, triplets = periodic_model.build_periodic_graph(
            positions_bulk, cell, pbc
        )
    
    print(f"Periodic neighbors found: {edge_index.shape[1]}")
    print(f"Triplet angles captured: {triplets.shape[0]}")
    
    print("\n" + "=" * 60)
    print("DimeNet++ Demo completed!")
    print("Key features:")
    print("- Directional message passing with angles")
    print("- Bessel and spherical basis functions")
    print("- Efficient 3-body interaction modeling")
    print("- Support for periodic systems")
    print("=" * 60)
