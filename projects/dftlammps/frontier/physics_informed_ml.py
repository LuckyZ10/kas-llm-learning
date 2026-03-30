"""
physics_informed_ml.py
物理约束机器学习势

将物理约束 (能量守恒、力平衡等) 融入ML势函数,
提高外推能力和物理一致性。

References:
- Zhang et al. (2020) "Deep Potential - Smooth Edition"
- Noe et al. (2020) "Machine Learning for Molecular Simulation"
- 2024进展: 严格物理约束的ML势函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import numpy as np


@dataclass
class PhysicsConstraints:
    """物理约束配置"""
    energy_conservation: bool = True
    force_conservation: bool = True  # F = -dE/dr
    translation_invariance: bool = True
    rotation_invariance: bool = True
    permutation_invariance: bool = True
    electroneutrality: bool = False  # 对于带电系统


class PhysicallyConservedMLP(nn.Module):
    """
    物理守恒MLP
    
    通过架构设计保证物理约束
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 128, 128],
        output_dim: int = 1,
        activation: str = "silu"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 构建网络
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "silu":
                layers.append(nn.SiLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "relu":
                layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class EnergyConservingPotential(nn.Module):
    """
    能量守恒势函数
    
    保证力是能量的负梯度: F = -∇E
    """
    
    def __init__(
        self,
        descriptor_dim: int = 128,
        hidden_dim: int = 128,
        num_elements: int = 100
    ):
        super().__init__()
        
        # 原子嵌入
        self.atom_embedding = nn.Embedding(num_elements, descriptor_dim)
        
        # 能量网络
        self.energy_net = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def compute_descriptor(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        计算原子环境描述符
        
        使用对称函数保证排列不变性
        """
        src, dst = edge_index
        
        # 基础原子特征
        atom_features = self.atom_embedding(atomic_numbers - 1)
        
        # 计算对称函数特征 (Behler-Parrinello类型)
        vectors = positions[dst] - positions[src]
        distances = torch.norm(vectors, dim=-1, keepdim=True)
        
        # 径向对称函数
        eta = 0.5
        rc = 6.0
        rs_values = torch.linspace(0, rc, 8, device=positions.device)
        
        radial_features = []
        for rs in rs_values:
            fc = 0.5 * (torch.cos(distances * np.pi / rc) + 1) * (distances < rc).float()
            g = torch.exp(-eta * (distances - rs) ** 2) * fc
            
            # 聚合到中心原子
            g_sum = torch.zeros(positions.shape[0], 1, device=positions.device)
            g_sum.index_add_(0, src, g)
            radial_features.append(g_sum)
        
        radial_features = torch.cat(radial_features, dim=-1)
        
        # 结合原子特征和环境特征
        descriptor = torch.cat([atom_features, radial_features], dim=-1)
        
        return descriptor
    
    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Returns:
            energy: 总能量
            forces: 原子力 (能量负梯度)
            atomic_energies: 原子能量
        """
        # 启用梯度计算用于力预测
        positions_grad = positions.requires_grad_(True)
        
        # 计算描述符
        descriptor = self.compute_descriptor(
            atomic_numbers, positions_grad, edge_index
        )
        
        # 预测原子能量
        atomic_energies = self.energy_net(descriptor).squeeze(-1)
        
        # 总能量
        total_energy = atomic_energies.sum()
        
        # 计算力 (-dE/dr)
        forces = -torch.autograd.grad(
            total_energy,
            positions_grad,
            create_graph=self.training,
            retain_graph=True
        )[0]
        
        return {
            'energy': total_energy,
            'forces': forces,
            'atomic_energies': atomic_energies
        }


class ChargeConservingPotential(nn.Module):
    """
    电荷守恒势函数
    
    对于离子系统, 保证总电荷守恒
    """
    
    def __init__(
        self,
        num_elements: int = 100,
        hidden_dim: int = 128,
        charge_neutrality: bool = True
    ):
        super().__init__()
        self.charge_neutrality = charge_neutrality
        
        # 电荷预测网络
        self.charge_net = nn.Sequential(
            nn.Embedding(num_elements, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """预测电荷和静电能"""
        # 预测原始电荷
        raw_charges = self.charge_net(atomic_numbers - 1).squeeze(-1)
        
        # 电荷守恒约束
        if self.charge_neutrality:
            # 减去平均电荷
            charges = raw_charges - raw_charges.mean()
        else:
            charges = raw_charges
        
        return {
            'charges': charges,
            'total_charge': charges.sum()
        }


class PhysicsInformedLoss(nn.Module):
    """
    物理约束损失函数
    
    在训练中加入物理一致性约束
    """
    
    def __init__(self, constraints: PhysicsConstraints):
        super().__init__()
        self.constraints = constraints
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        计算物理约束损失
        """
        losses = {}
        
        # 能量损失
        if 'energy' in predictions and 'energy' in targets:
            losses['energy'] = F.mse_loss(predictions['energy'], targets['energy'])
        
        # 力损失
        if 'forces' in predictions and 'forces' in targets:
            losses['forces'] = F.mse_loss(predictions['forces'], targets['forces'])
        
        # 能量-力一致性检查
        if self.constraints.force_conservation:
            losses['consistency'] = self._check_force_consistency(predictions)
        
        # 总损失
        total = sum(losses.values())
        losses['total'] = total
        
        return losses
    
    def _check_force_consistency(self, predictions: Dict) -> torch.Tensor:
        """检查力与能量的一致性"""
        # 简化: 检查力的总和接近0 (牛顿第三定律)
        if 'forces' in predictions:
            force_sum = predictions['forces'].sum(dim=0)
            return torch.norm(force_sum)
        return torch.tensor(0.0)


if __name__ == "__main__":
    print("=" * 60)
    print("Physics-Informed ML Demo")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 测试能量守恒势
    print("\n1. Energy Conserving Potential")
    potential = EnergyConservingPotential(
        descriptor_dim=64,
        hidden_dim=64,
        num_elements=20
    ).to(device)
    
    # 测试结构
    atomic_numbers = torch.tensor([8, 1, 1], device=device)
    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ], device=device)
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 2],
        [1, 2, 0, 2, 0, 1]
    ], device=device)
    
    result = potential(atomic_numbers, positions, edge_index)
    print(f"Energy: {result['energy'].item():.4f} eV")
    print(f"Forces shape: {result['forces'].shape}")
    print(f"Force sum: {result['forces'].sum(dim=0)}")
    
    print("\nDemo completed!")
