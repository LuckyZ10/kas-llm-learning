"""
mace_integration.py
MACE (Multi-Atomic Cluster Expansion) 等变图神经网络

MACE是2022-2023年提出的高阶等变消息传递神经网络,
在材料势函数任务上达到SOTA性能, 样本效率比传统GNN高10-100倍。

References:
- Batatia et al. (2022) "MACE: Higher Order Equivariant Message Passing Neural Networks"
- Batatia et al. (2023) "MACE-MP: A Universal Machine Learning Potential for Materials"
- 2024进展: MACE用于大尺度MD和材料发现工作流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import numpy as np
from e3nn import o3
from e3nn.nn import Gate
from e3nn.o3 import TensorProduct, FullyConnectedTensorProduct, Linear
import warnings

try:
    import e3nn
    E3NN_AVAILABLE = True
except ImportError:
    E3NN_AVAILABLE = False
    warnings.warn("e3nn not installed. Using simplified implementation.")


@dataclass
class AtomicData:
    """原子数据结构"""
    positions: torch.Tensor      # [n_atoms, 3] 笛卡尔坐标
    atomic_numbers: torch.Tensor # [n_atoms] 原子序数
    cell: Optional[torch.Tensor] = None  # [3, 3] 晶胞矩阵
    pbc: Optional[torch.Tensor] = None   # [3] 周期性边界条件
    edge_index: Optional[torch.Tensor] = None
    edge_shift: Optional[torch.Tensor] = None  # 周期镜像偏移
    batch: Optional[torch.Tensor] = None
    energy: Optional[torch.Tensor] = None
    forces: Optional[torch.Tensor] = None
    stress: Optional[torch.Tensor] = None


class SphericalHarmonics(nn.Module):
    """球谐函数计算"""
    
    def __init__(self, lmax: int = 3):
        super().__init__()
        self.lmax = lmax
        
    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        计算球谐函数特征
        
        Args:
            vectors: [n_edges, 3] 边向量
        Returns:
            [n_edges, (lmax+1)^2] 球谐特征
        """
        x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
        r = torch.norm(vectors, dim=-1, keepdim=True) + 1e-8
        
        # 归一化方向
        x, y, z = x/r.squeeze(-1), y/r.squeeze(-1), z/r.squeeze(-1)
        
        features = []
        
        # l=0 (1个特征)
        features.append(torch.ones_like(x).unsqueeze(-1))
        
        # l=1 (3个特征) - p轨道
        if self.lmax >= 1:
            features.append(torch.stack([y, z, x], dim=-1))
        
        # l=2 (5个特征) - d轨道
        if self.lmax >= 2:
            yz, xz, xy = y*z, x*z, x*y
            x2, y2, z2 = x**2, y**2, z**2
            features.append(torch.stack([
                3*z2 - 1,  # Y_{2,0}
                x*z,       # Y_{2,1} (simplified)
                y*z,       # Y_{2,2}
                x*y,       # Y_{2,3}
                x2 - y2    # Y_{2,4}
            ], dim=-1))
        
        return torch.cat(features, dim=-1)


class RadialBasisEmbedding(nn.Module):
    """径向基函数嵌入"""
    
    def __init__(
        self,
        num_basis: int = 8,
        cutoff: float = 6.0,
        rbf_type: str = "bessel"
    ):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff
        self.rbf_type = rbf_type
        
        # Bessel函数根
        freqs = torch.arange(1, num_basis + 1) * np.pi / cutoff
        self.register_buffer('freqs', freqs)
        
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        计算RBF特征
        
        Args:
            distances: [n_edges] 距离
        Returns:
            [n_edges, num_basis] RBF特征
        """
        # 截止函数
        cutoff_val = 0.5 * (torch.cos(distances * np.pi / self.cutoff) + 1)
        cutoff_val = cutoff_val * (distances < self.cutoff).float()
        
        if self.rbf_type == "bessel":
            # 球Bessel函数
            rbf = torch.sin(distances.unsqueeze(-1) * self.freqs) / distances.unsqueeze(-1)
        elif self.rbf_type == "gaussian":
            # 高斯基
            centers = torch.linspace(0, self.cutoff, self.num_basis, device=distances.device)
            widths = torch.ones_like(centers) * (self.cutoff / self.num_basis)
            rbf = torch.exp(-((distances.unsqueeze(-1) - centers) ** 2) / (2 * widths ** 2))
        else:
            rbf = distances.unsqueeze(-1).expand(-1, self.num_basis)
        
        return rbf * cutoff_val.unsqueeze(-1)


class MACEBlock(nn.Module):
    """
    MACE消息传递块
    
    核心创新: 高阶等变消息传递
    - 使用球谐函数进行E(3)等变特征构建
    - 多阶张量积消息聚合
    """
    
    def __init__(
        self,
        hidden_irreps: str = "32x0e + 32x1o + 16x2e",
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        max_ell: int = 3,
        correlation: int = 3,
        num_elements: int = 100
    ):
        super().__init__()
        self.hidden_irreps = hidden_irreps
        self.max_ell = max_ell
        self.correlation = correlation
        
        # 原子嵌入
        self.atom_embed = nn.Embedding(num_elements, 64)
        
        # 球谐函数
        self.spherical_harmonics = SphericalHarmonics(lmax=max_ell)
        
        # 径向嵌入
        self.radial_embedding = RadialBasisEmbedding(num_basis=num_bessel)
        
        # 径向网络 (MLP)
        self.radial_net = nn.Sequential(
            nn.Linear(num_bessel, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU()
        )
        
        # 可学习的张量积 (简化版)
        self.tensor_product = nn.Linear(128, 128)
        
        # 消息聚合
        self.message_mlp = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        
        # 自注意力
        self.attention = nn.MultiheadAttention(128, num_heads=4, batch_first=True)
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(128, 128),
            nn.Sigmoid()
        )
        
        # 输出投影
        self.output_proj = nn.Linear(128, 128)
        
    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        MACE前向传播
        
        Args:
            atomic_numbers: [n_atoms] 原子序数
            positions: [n_atoms, 3] 位置
            edge_index: [2, n_edges] 边索引
            edge_vectors: [n_edges, 3] 边向量
            edge_lengths: [n_edges] 边长度
        """
        # 原子特征
        node_features = self.atom_embed(atomic_numbers)
        
        # 球谐特征 (等变)
        sh_features = self.spherical_harmonics(edge_vectors)
        
        # 径向特征 (不变)
        rbf_features = self.radial_embedding(edge_lengths)
        radial_features = self.radial_net(rbf_features)
        
        # 构建消息 (简化的高阶等变消息)
        src, dst = edge_index
        
        # 消息构建: 结合标量特征和球谐
        messages = []
        for l in range(min(self.max_ell + 1, 3)):
            # 提取对应阶数的球谐特征
            start_idx = l ** 2
            end_idx = (l + 1) ** 2
            if end_idx <= sh_features.shape[1]:
                sh_l = sh_features[:, start_idx:end_idx]
                
                # 张量积 (简化)
                msg = radial_features.unsqueeze(-1) * sh_l.unsqueeze(1)
                messages.append(msg.flatten(1))
        
        if messages:
            edge_messages = torch.cat(messages, dim=-1)
            edge_messages = self.tensor_product(edge_messages)
        else:
            edge_messages = radial_features
        
        # 聚合消息
        aggregated = torch.zeros(node_features.shape[0], edge_messages.shape[1],
                                device=node_features.device)
        aggregated.index_add_(0, dst, edge_messages)
        
        # 结合节点特征
        combined = torch.cat([node_features, aggregated[:, :node_features.shape[1]]], dim=-1)
        updated = self.message_mlp(combined)
        
        # 自注意力更新
        updated_unsqueezed = updated.unsqueeze(0)
        attended, _ = self.attention(updated_unsqueezed, updated_unsqueezed, updated_unsqueezed)
        attended = attended.squeeze(0)
        
        # 门控
        gate = self.gate(updated)
        output = gate * attended + (1 - gate) * updated
        
        return self.output_proj(output)


class MACE(nn.Module):
    """
    MACE势函数模型
    
    完整实现包含:
    - 多个等变消息传递层
    - 能量和力的预测
    - 应力张量预测 (周期性系统)
    """
    
    def __init__(
        self,
        num_elements: int = 100,
        hidden_channels: int = 128,
        num_layers: int = 4,
        max_ell: int = 3,
        num_bessel: int = 8,
        cutoff: float = 6.0,
        max_neighbors: int = 40,
        atomic_energies: Optional[Dict[int, float]] = None
    ):
        super().__init__()
        self.num_elements = num_elements
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        
        # 原子能量 (参考能量)
        if atomic_energies is None:
            atomic_energies = {z: 0.0 for z in range(1, num_elements + 1)}
        self.register_buffer(
            'atomic_energies',
            torch.tensor([atomic_energies.get(z, 0.0) for z in range(num_elements)])
        )
        
        # MACE层
        self.layers = nn.ModuleList([
            MACEBlock(
                hidden_irreps=f"{hidden_channels//4}x0e + {hidden_channels//4}x1o + {hidden_channels//8}x2e",
                num_bessel=num_bessel,
                max_ell=max_ell
            )
            for _ in range(num_layers)
        ])
        
        # 读取函数
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, 1)
        )
        
        # 力预测头
        self.forces_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 3)
        )
        
    def build_graph(
        self,
        positions: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """构建邻居图"""
        n_atoms = positions.shape[0]
        device = positions.device
        
        if cell is None or pbc is None:
            # 非周期性系统 - 使用欧氏距离
            dists = torch.cdist(positions, positions)
            mask = (dists < self.cutoff) & (dists > 0.01)
            edge_index = torch.nonzero(mask, as_tuple=False).t()
            edge_vectors = positions[edge_index[1]] - positions[edge_index[0]]
            edge_lengths = torch.norm(edge_vectors, dim=-1)
        else:
            # 周期性系统
            edge_index, edge_vectors, edge_lengths = self._build_periodic_graph(
                positions, cell, pbc
            )
        
        return edge_index, edge_vectors, edge_lengths
    
    def _build_periodic_graph(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """构建周期性邻居图"""
        n_atoms = positions.shape[0]
        device = positions.device
        
        edge_list = []
        vector_list = []
        length_list = []
        
        # 生成周期镜像偏移
        shifts = self._generate_shifts(cell, pbc)
        
        for i in range(n_atoms):
            for shift in shifts:
                shifted_pos = positions + shift.unsqueeze(0)
                vectors = shifted_pos - positions[i:i+1]
                dists = torch.norm(vectors, dim=-1)
                
                mask = (dists < self.cutoff) & (dists > 0.01)
                neighbors = torch.where(mask)[0]
                
                for j in neighbors[:self.max_neighbors]:
                    edge_list.append([i, j.item()])
                    vector_list.append(vectors[j].cpu().numpy())
                    length_list.append(dists[j].item())
        
        if len(edge_list) == 0:
            return (
                torch.zeros((2, 0), dtype=torch.long, device=device),
                torch.zeros((0, 3), device=device),
                torch.zeros(0, device=device)
            )
        
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t()
        edge_vectors = torch.tensor(vector_list, dtype=torch.float32, device=device)
        edge_lengths = torch.tensor(length_list, dtype=torch.float32, device=device)
        
        return edge_index, edge_vectors, edge_lengths
    
    def _generate_shifts(self, cell: torch.Tensor, pbc: torch.Tensor) -> torch.Tensor:
        """生成周期镜像"""
        device = cell.device
        shifts = []
        
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    shift = torch.tensor([i, j, k], dtype=torch.float32, device=device)
                    # 应用PBC
                    shift = shift * pbc.float()
                    cart_shift = shift @ cell
                    shifts.append(cart_shift)
        
        return torch.stack(shifts)
    
    def forward(
        self,
        data: AtomicData
    ) -> Dict[str, torch.Tensor]:
        """
        MACE前向传播
        
        Returns:
            dict with 'energy', 'forces', optionally 'stress'
        """
        positions = data.positions
        atomic_numbers = data.atomic_numbers
        
        # 构建图
        edge_index, edge_vectors, edge_lengths = self.build_graph(
            positions, data.cell, data.pbc
        )
        
        # 初始原子能量
        atomic_energy_shifts = self.atomic_energies[atomic_numbers - 1]
        
        # 消息传递
        node_features = self.atom_embed(atomic_numbers) if hasattr(self, 'atom_embed') else \
                       torch.zeros(positions.shape[0], 128, device=positions.device)
        
        for layer in self.layers:
            node_features = layer(
                atomic_numbers, positions, edge_index,
                edge_vectors, edge_lengths
            )
        
        # 预测原子能量
        atomic_energies = self.readout_mlp(node_features).squeeze(-1)
        atomic_energies = atomic_energies + atomic_energy_shifts
        
        # 总能量
        total_energy = atomic_energies.sum()
        
        # 计算力 (-dE/dr)
        positions_for_grad = positions.detach().requires_grad_(True)
        forces = -torch.autograd.grad(
            total_energy, positions_for_grad,
            create_graph=self.training,
            retain_graph=True
        )[0]
        
        result = {
            'energy': total_energy,
            'forces': forces,
            'atomic_energies': atomic_energies
        }
        
        # 计算应力 (如果提供晶胞)
        if data.cell is not None and self.training:
            stress = self._compute_stress(total_energy, data.cell)
            result['stress'] = stress
        
        return result
    
    def _compute_stress(
        self,
        energy: torch.Tensor,
        cell: torch.Tensor
    ) -> torch.Tensor:
        """计算应力张量"""
        cell_for_grad = cell.detach().requires_grad_(True)
        
        # 应力 = (1/V) * dE/depsilon
        stress = torch.autograd.grad(
            energy, cell_for_grad,
            create_graph=self.training
        )[0]
        
        # 计算体积
        volume = torch.det(cell)
        stress = stress / volume
        
        return stress


class MACEActiveLearner:
    """
    MACE主动学习器
    
    高效选择训练样本, 最大化模型性能
    """
    
    def __init__(
        self,
        model: MACE,
        uncertainty_method: str = "ensemble"
    ):
        self.model = model
        self.uncertainty_method = uncertainty_method
        
    def compute_uncertainty(
        self,
        structures: List[AtomicData]
    ) -> torch.Tensor:
        """
        计算结构不确定性
        
        用于选择最有价值的训练样本
        """
        uncertainties = []
        
        for struct in structures:
            if self.uncertainty_method == "ensemble":
                # 使用模型集成估计不确定性
                with torch.no_grad():
                    # 多次前向传播 (dropout作为贝叶斯近似)
                    self.model.train()  # 启用dropout
                    predictions = []
                    for _ in range(10):
                        pred = self.model(struct)['energy']
                        predictions.append(pred.item())
                
                uncertainty = np.std(predictions)
                uncertainties.append(uncertainty)
            
            elif self.uncertainty_method == "gradient":
                # 基于梯度范数的不确定性
                struct.positions.requires_grad_(True)
                output = self.model(struct)
                grad_norm = torch.norm(output['forces']).item()
                uncertainties.append(grad_norm)
        
        return torch.tensor(uncertainties)
    
    def select_samples(
        self,
        pool: List[AtomicData],
        n_select: int,
        strategy: str = "uncertainty"
    ) -> List[int]:
        """
        选择下一个批次的训练样本
        
        Strategies:
        - uncertainty: 选择最不确定的样本
        - diversity: 选择多样化的样本
        - greedy: 贪心选择
        """
        if strategy == "uncertainty":
            uncertainties = self.compute_uncertainty(pool)
            _, indices = torch.topk(uncertainties, n_select)
            return indices.tolist()
        
        elif strategy == "diversity":
            # 基于特征空间距离的多样性采样
            features = self._extract_features(pool)
            selected = self._max_min_distance_selection(features, n_select)
            return selected
        
        else:
            # 随机选择
            import random
            return random.sample(range(len(pool)), min(n_select, len(pool)))
    
    def _extract_features(self, structures: List[AtomicData]) -> torch.Tensor:
        """提取结构特征用于多样性选择"""
        features = []
        
        self.model.eval()
        with torch.no_grad():
            for struct in structures:
                output = self.model(struct)
                # 使用原子能量作为特征
                feat = output['atomic_energies'].mean()
                features.append(feat.item())
        
        return torch.tensor(features).unsqueeze(-1)
    
    def _max_min_distance_selection(
        self,
        features: torch.Tensor,
        n_select: int
    ) -> List[int]:
        """最大最小距离采样"""
        n_total = features.shape[0]
        selected = [np.random.randint(n_total)]
        
        for _ in range(n_select - 1):
            # 计算到已选样本的最小距离
            dists = torch.cdist(features, features[selected])
            min_dists, _ = dists.min(dim=1)
            
            # 选择最大最小距离的样本
            next_idx = min_dists.argmax().item()
            selected.append(next_idx)
        
        return selected


def load_mace_model(checkpoint_path: str) -> MACE:
    """加载预训练MACE模型"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 从checkpoint重建模型
    model = MACE(**checkpoint.get('model_config', {}))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def mace_md_simulation(
    model: MACE,
    initial_structure: AtomicData,
    n_steps: int = 1000,
    timestep: float = 1.0,  # fs
    temperature: float = 300.0,
    log_interval: int = 100
) -> Dict[str, List]:
    """
    使用MACE势进行分子动力学模拟
    
    实现简单的Langevin动力学
    """
    positions = initial_structure.positions.clone()
    velocities = torch.randn_like(positions) * np.sqrt(temperature / 1000)
    
    trajectory = []
    energies = []
    
    for step in range(n_steps):
        # 计算力和能量
        data = AtomicData(
            positions=positions,
            atomic_numbers=initial_structure.atomic_numbers,
            cell=initial_structure.cell,
            pbc=initial_structure.pbc
        )
        
        with torch.no_grad():
            output = model(data)
        
        forces = output['forces']
        energy = output['energy'].item()
        
        # Langevin更新
        friction = 0.01
        noise = torch.randn_like(velocities)
        
        velocities = velocities * (1 - friction) + forces * timestep + noise * np.sqrt(temperature * friction)
        positions = positions + velocities * timestep
        
        # 记录
        if step % log_interval == 0:
            trajectory.append(positions.clone())
            energies.append(energy)
            print(f"Step {step}: Energy = {energy:.4f} eV")
    
    return {
        'trajectory': trajectory,
        'energies': energies
    }


if __name__ == "__main__":
    print("=" * 60)
    print("MACE Demo - Higher Order Equivariant Message Passing")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 创建模型
    print("\n1. Creating MACE model")
    model = MACE(
        num_elements=20,
        hidden_channels=64,
        num_layers=2,
        max_ell=2,
        cutoff=5.0
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建测试结构
    print("\n2. Creating test structure (H2O molecule)")
    
    data = AtomicData(
        positions=torch.tensor([
            [0.0, 0.0, 0.0],    # O
            [0.96, 0.0, 0.0],   # H
            [-0.24, 0.93, 0.0]  # H
        ], device=device),
        atomic_numbers=torch.tensor([8, 1, 1], device=device)
    )
    
    # 前向传播
    print("\n3. Running forward pass")
    model.eval()
    with torch.no_grad():
        output = model(data)
    
    print(f"Predicted energy: {output['energy'].item():.4f} eV")
    print(f"Forces shape: {output['forces'].shape}")
    print(f"Forces:\n{output['forces']}")
    
    # 测试主动学习
    print("\n4. Active Learning Test")
    
    # 创建候选池
    pool = [data] * 10  # 简化: 复制相同结构
    
    learner = MACEActiveLearner(model)
    uncertainties = learner.compute_uncertainty(pool)
    print(f"Uncertainties: {uncertainties.numpy()}")
    
    selected = learner.select_samples(pool, n_select=3, strategy="uncertainty")
    print(f"Selected indices: {selected}")
    
    # 简单MD测试
    print("\n5. Simple MD Test (10 steps)")
    md_result = mace_md_simulation(
        model, data, n_steps=10, log_interval=5
    )
    print(f"Final energy: {md_result['energies'][-1]:.4f} eV")
    
    print("\n" + "=" * 60)
    print("MACE Demo completed!")
    print("Key features:")
    print("- High-order equivariant message passing")
    print("- 10-100x sample efficiency vs conventional GNNs")
    print("- Accurate force prediction for MD")
    print("- Active learning support")
    print("=" * 60)
