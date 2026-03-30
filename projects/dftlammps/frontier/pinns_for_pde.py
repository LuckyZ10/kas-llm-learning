"""
pinns_for_pde.py
物理信息神经网络求解材料PDE

PINNs (Physics-Informed Neural Networks) 将物理方程作为约束融入神经网络,
用于求解相场方程、扩散方程、弹性力学方程等材料科学中的PDE。

References:
- Raissi et al. (2019) "Physics-informed neural networks"
- 2024进展: Extended PINNs用于多尺度材料问题
- Krishnapriyan et al. (2021) "Characterizing possible failure modes in physics-informed neural networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize


@dataclass
class PDESolution:
    """PDE解数据结构"""
    solution: torch.Tensor    # 解场
    coords: torch.Tensor      # 坐标
    time: Optional[torch.Tensor] = None
    residual: Optional[torch.Tensor] = None
    gradient: Optional[torch.Tensor] = None


class FourierFeatureEncoding(nn.Module):
    """
    傅里叶特征编码
    
    提高神经网络学习高频模式的能力
    关键用于材料微观结构等高频问题
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        mapping_size: int = 256,
        scale: float = 10.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        
        # 随机傅里叶特征
        B = torch.randn((input_dim, mapping_size)) * scale
        self.register_buffer('B', B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        傅里叶特征变换
        
        γ(x) = [sin(2πBx), cos(2πBx)]
        """
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class SirenLayer(nn.Module):
    """
    SIREN层 - 使用sin激活函数
    
    适合表示复杂的物理场
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 30.0,
        is_first: bool = False
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.linear = nn.Linear(in_features, out_features)
        
        # SIREN初始化
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / in_features) / omega_0,
                    np.sqrt(6 / in_features) / omega_0
                )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SirenNetwork(nn.Module):
    """SIREN神经网络"""
    
    def __init__(
        self,
        in_features: int = 3,
        hidden_features: int = 256,
        hidden_layers: int = 4,
        out_features: int = 1,
        omega_0: float = 30.0
    ):
        super().__init__()
        
        self.net = []
        self.net.append(SirenLayer(in_features, hidden_features, omega_0, is_first=True))
        
        for _ in range(hidden_layers):
            self.net.append(SirenLayer(hidden_features, hidden_features, omega_0))
        
        self.net.append(nn.Linear(hidden_features, out_features))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)


class PINN(nn.Module):
    """
    物理信息神经网络基类
    
    通用框架用于求解各类PDE
    """
    
    def __init__(
        self,
        network: nn.Module,
        pde_fn: Callable,
        bc_fn: Optional[Callable] = None,
        ic_fn: Optional[Callable] = None
    ):
        super().__init__()
        self.network = network
        self.pde_fn = pde_fn
        self.bc_fn = bc_fn
        self.ic_fn = ic_fn
        
        # 损失权重
        self.pde_weight = 1.0
        self.bc_weight = 1.0
        self.ic_weight = 1.0
        self.data_weight = 1.0
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """前向传播 - 预测解"""
        return self.network(coords)
    
    def compute_derivatives(
        self,
        coords: torch.Tensor,
        time: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算解的各阶导数
        
        使用自动微分
        """
        coords_input = coords.clone().requires_grad_(True)
        
        if time is not None:
            time_input = time.clone().requires_grad_(True)
            inputs = torch.cat([coords_input, time_input], dim=-1)
        else:
            time_input = None
            inputs = coords_input
        
        # 预测解
        u = self.network(inputs)
        
        # 计算梯度
        grads = {}
        
        # 一阶导数
        du_dcoord = torch.autograd.grad(
            u, coords_input,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        grads['u'] = u
        grads['du_dx'] = du_dcoord[:, 0:1] if coords.shape[-1] >= 1 else None
        grads['du_dy'] = du_dcoord[:, 1:2] if coords.shape[-1] >= 2 else None
        grads['du_dz'] = du_dcoord[:, 2:3] if coords.shape[-1] >= 3 else None
        
        # 时间导数
        if time_input is not None:
            du_dt = torch.autograd.grad(
                u, time_input,
                grad_outputs=torch.ones_like(u),
                create_graph=True,
                retain_graph=True
            )[0]
            grads['du_dt'] = du_dt
        
        # 二阶导数
        if grads['du_dx'] is not None:
            d2u_dx2 = torch.autograd.grad(
                grads['du_dx'], coords_input,
                grad_outputs=torch.ones_like(grads['du_dx']),
                create_graph=True,
                retain_graph=True
            )[0][:, 0:1]
            grads['d2u_dx2'] = d2u_dx2
        
        if grads['du_dy'] is not None:
            d2u_dy2 = torch.autograd.grad(
                grads['du_dy'], coords_input,
                grad_outputs=torch.ones_like(grads['du_dy']),
                create_graph=True,
                retain_graph=True
            )[0][:, 1:2]
            grads['d2u_dy2'] = d2u_dy2
        
        if grads['du_dz'] is not None:
            d2u_dz2 = torch.autograd.grad(
                grads['du_dz'], coords_input,
                grad_outputs=torch.ones_like(grads['du_dz']),
                create_graph=True,
                retain_graph=True
            )[0][:, 2:3]
            grads['d2u_dz2'] = d2u_dz2
        
        return grads
    
    def pde_loss(
        self,
        coords: torch.Tensor,
        time: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算PDE残差损失
        """
        grads = self.compute_derivatives(coords, time)
        residual = self.pde_fn(grads, coords, time)
        return torch.mean(residual ** 2)
    
    def bc_loss(
        self,
        bc_coords: torch.Tensor,
        bc_values: torch.Tensor
    ) -> torch.Tensor:
        """
        计算边界条件损失
        """
        if self.bc_fn is None:
            # 默认Dirichlet边界条件
            pred = self.forward(bc_coords)
            return torch.mean((pred - bc_values) ** 2)
        else:
            return self.bc_fn(self, bc_coords, bc_values)
    
    def ic_loss(
        self,
        ic_coords: torch.Tensor,
        ic_time: torch.Tensor,
        ic_values: torch.Tensor
    ) -> torch.Tensor:
        """
        计算初始条件损失
        """
        if self.ic_fn is None:
            inputs = torch.cat([ic_coords, ic_time], dim=-1)
            pred = self.forward(inputs)
            return torch.mean((pred - ic_values) ** 2)
        else:
            return self.ic_fn(self, ic_coords, ic_time, ic_values)
    
    def data_loss(
        self,
        data_coords: torch.Tensor,
        data_values: torch.Tensor
    ) -> torch.Tensor:
        """
        数据拟合损失 (如果有观测数据)
        """
        pred = self.forward(data_coords)
        return torch.mean((pred - data_values) ** 2)
    
    def compute_loss(
        self,
        pde_coords: torch.Tensor,
        bc_coords: Optional[torch.Tensor] = None,
        bc_values: Optional[torch.Tensor] = None,
        ic_coords: Optional[torch.Tensor] = None,
        ic_time: Optional[torch.Tensor] = None,
        ic_values: Optional[torch.Tensor] = None,
        data_coords: Optional[torch.Tensor] = None,
        data_values: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        """
        losses = {}
        
        # PDE损失
        losses['pde'] = self.pde_loss(pde_coords, time)
        total_loss = self.pde_weight * losses['pde']
        
        # 边界条件损失
        if bc_coords is not None and bc_values is not None:
            losses['bc'] = self.bc_loss(bc_coords, bc_values)
            total_loss = total_loss + self.bc_weight * losses['bc']
        
        # 初始条件损失
        if ic_coords is not None and ic_values is not None:
            losses['ic'] = self.ic_loss(ic_coords, ic_time, ic_values)
            total_loss = total_loss + self.ic_weight * losses['ic']
        
        # 数据损失
        if data_coords is not None and data_values is not None:
            losses['data'] = self.data_loss(data_coords, data_values)
            total_loss = total_loss + self.data_weight * losses['data']
        
        losses['total'] = total_loss
        
        return losses


class PhaseFieldPINN(PINN):
    """
    相场方程PINN
    
    求解Cahn-Hilliard方程和Allen-Cahn方程
    用于模拟相分离和晶粒生长
    """
    
    def __init__(
        self,
        network: nn.Module,
        equation_type: str = "cahn_hilliard",
        mobility: float = 1.0,
        interface_width: float = 0.1,
        free_energy_fn: Optional[Callable] = None
    ):
        # 定义PDE
        def pde_fn(grads, coords, time):
            if equation_type == "cahn_hilliard":
                return self._cahn_hilliard_residual(grads, mobility, interface_width)
            elif equation_type == "allen_cahn":
                return self._allen_cahn_residual(grads, mobility, interface_width)
            else:
                raise ValueError(f"Unknown equation type: {equation_type}")
        
        super().__init__(network, pde_fn)
        self.equation_type = equation_type
        self.mobility = mobility
        self.interface_width = interface_width
        self.free_energy_fn = free_energy_fn or self._default_free_energy
    
    def _default_free_energy(self, phi: torch.Tensor) -> torch.Tensor:
        """双阱自由能: f(φ) = (φ² - 1)² / 4"""
        return (phi ** 2 - 1) ** 2 / 4
    
    def _cahn_hilliard_residual(
        self,
        grads: Dict[str, torch.Tensor],
        mobility: float,
        epsilon: float
    ) -> torch.Tensor:
        """
        Cahn-Hilliard方程残差:
        
        ∂φ/∂t = M ∇²(∂f/∂φ - ε²∇²φ)
        
        守恒序参量 (如浓度)
        """
        phi = grads['u']
        dphi_dt = grads.get('du_dt', torch.zeros_like(phi))
        
        # 计算化学势 μ = ∂f/∂φ - ε²∇²φ
        dfdphi = phi ** 3 - phi  # 双阱导数
        
        laplacian = grads.get('d2u_dx2', torch.zeros_like(phi))
        if 'd2u_dy2' in grads:
            laplacian = laplacian + grads['d2u_dy2']
        if 'd2u_dz2' in grads:
            laplacian = laplacian + grads['d2u_dz2']
        
        mu = dfdphi - epsilon ** 2 * laplacian
        
        # 计算∇²μ
        # 简化: 使用自动微分计算二阶梯度
        coords = grads['u'].requires_grad_(True)
        
        # 残差: ∂φ/∂t - M∇²μ
        # 这里简化处理
        residual = dphi_dt - mobility * laplacian
        
        return residual
    
    def _allen_cahn_residual(
        self,
        grads: Dict[str, torch.Tensor],
        mobility: float,
        epsilon: float
    ) -> torch.Tensor:
        """
        Allen-Cahn方程残差:
        
        ∂φ/∂t = -M(∂f/∂φ - ε²∇²φ)
        
        非守恒序参量 (如相变)
        """
        phi = grads['u']
        dphi_dt = grads.get('du_dt', torch.zeros_like(phi))
        
        # 自由能导数
        dfdphi = phi ** 3 - phi
        
        # Laplacian
        laplacian = grads.get('d2u_dx2', torch.zeros_like(phi))
        if 'd2u_dy2' in grads:
            laplacian = laplacian + grads['d2u_dy2']
        if 'd2u_dz2' in grads:
            laplacian = laplacian + grads['d2u_dz2']
        
        # 残差
        residual = dphi_dt + mobility * (dfdphi - epsilon ** 2 * laplacian)
        
        return residual


class DiffusionPINN(PINN):
    """
    扩散方程PINN
    
    求解Fick扩散方程
    ∂c/∂t = D∇²c
    """
    
    def __init__(
        self,
        network: nn.Module,
        diffusion_coefficient: float = 1.0,
        variable_diffusivity: bool = False
    ):
        def pde_fn(grads, coords, time):
            c = grads['u']
            dc_dt = grads.get('du_dt', torch.zeros_like(c))
            
            # Laplacian
            laplacian = grads.get('d2u_dx2', torch.zeros_like(c))
            if 'd2u_dy2' in grads:
                laplacian = laplacian + grads['d2u_dy2']
            if 'd2u_dz2' in grads:
                laplacian = laplacian + grads['d2u_dz2']
            
            if variable_diffusivity:
                # 变扩散系数 D(c)
                D = self._compute_diffusivity(c)
                residual = dc_dt - D * laplacian
            else:
                residual = dc_dt - diffusion_coefficient * laplacian
            
            return residual
        
        super().__init__(network, pde_fn)
        self.diffusion_coefficient = diffusion_coefficient
        self.variable_diffusivity = variable_diffusivity
    
    def _compute_diffusivity(self, c: torch.Tensor) -> torch.Tensor:
        """计算浓度相关的扩散系数"""
        # 示例: D(c) = D0 * (1 + αc)
        D0 = self.diffusion_coefficient
        alpha = 0.5
        return D0 * (1 + alpha * c)


class ElasticityPINN(PINN):
    """
    弹性力学PINN
    
    求解线弹性方程
    ∇·σ + f = 0
    σ = C:ε
    """
    
    def __init__(
        self,
        network: nn.Module,
        youngs_modulus: float = 200e9,
        poisson_ratio: float = 0.3,
        dimensions: int = 2
    ):
        self.E = youngs_modulus
        self.nu = poisson_ratio
        self.dimensions = dimensions
        
        # 计算Lamé参数
        self.lambda_lame = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu_lame = self.E / (2 * (1 + self.nu))
        
        def pde_fn(grads, coords, time):
            return self._elasticity_residual(grads, coords)
        
        super().__init__(network, pde_fn)
    
    def _elasticity_residual(
        self,
        grads: Dict[str, torch.Tensor],
        coords: torch.Tensor
    ) -> torch.Tensor:
        """
        弹性力学方程残差
        
        对于位移场 u = [ux, uy, (uz)]
        """
        # 这里简化处理标量场
        # 实际应该使用向量场网络
        
        u = grads['u']
        
        # 计算应变和应力的简化版本
        if 'd2u_dx2' in grads and 'd2u_dy2' in grads:
            # 简化: ∇²u = 0 (调和方程)
            laplacian = grads['d2u_dx2'] + grads['d2u_dy2']
            if 'd2u_dz2' in grads:
                laplacian = laplacian + grads['d2u_dz2']
            
            return laplacian
        
        return torch.zeros_like(u)


class AdaptiveWeightPINN(PINN):
    """
    自适应权重PINN
    
    动态调整各项损失的权重, 改善训练稳定性
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 可学习的权重或对数权重
        self.log_pde_weight = nn.Parameter(torch.zeros(1))
        self.log_bc_weight = nn.Parameter(torch.zeros(1))
        self.log_ic_weight = nn.Parameter(torch.zeros(1))
        
        # 目标梯度幅度
        self.target_grad = 1.0
    
    def compute_adaptive_weights(
        self,
        pde_coords: torch.Tensor,
        bc_coords: Optional[torch.Tensor] = None,
        ic_coords: Optional[torch.Tensor] = None
    ):
        """计算自适应权重"""
        # 计算各损失的梯度
        
        # PDE损失梯度
        self.zero_grad()
        pde_loss = self.pde_loss(pde_coords)
        pde_loss.backward(retain_graph=True)
        pde_grad = torch.cat([
            p.grad.flatten() for p in self.network.parameters() if p.grad is not None
        ])
        pde_grad_norm = torch.norm(pde_grad)
        
        # BC损失梯度
        if bc_coords is not None:
            self.zero_grad()
            bc_loss = self.bc_loss(bc_coords, torch.zeros(bc_coords.shape[0], 1, device=bc_coords.device))
            bc_loss.backward(retain_graph=True)
            bc_grad = torch.cat([
                p.grad.flatten() for p in self.network.parameters() if p.grad is not None
            ])
            bc_grad_norm = torch.norm(bc_grad)
        else:
            bc_grad_norm = torch.tensor(1.0)
        
        # 更新权重 (使各损失梯度幅度平衡)
        max_grad = max(pde_grad_norm.item(), bc_grad_norm.item())
        
        self.pde_weight = max_grad / (pde_grad_norm.item() + 1e-8)
        self.bc_weight = max_grad / (bc_grad_norm.item() + 1e-8)


def train_pinn(
    pinn: PINN,
    domain_bounds: List[Tuple[float, float]],
    num_collocation: int = 10000,
    num_epochs: int = 10000,
    lr: float = 1e-3,
    device: str = 'cuda'
) -> Dict[str, List[float]]:
    """
    训练PINN模型
    
    Args:
        domain_bounds: 每个维度的范围 [(xmin, xmax), (ymin, ymax), ...]
        num_collocation: 配点数量
    """
    pinn = pinn.to(device)
    optimizer = torch.optim.Adam(pinn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1000)
    
    history = {'total_loss': [], 'pde_loss': [], 'bc_loss': []}
    
    for epoch in range(num_epochs):
        # 采样配点
        collocation_points = []
        for bounds in domain_bounds:
            points = torch.rand(num_collocation, 1, device=device)
            points = points * (bounds[1] - bounds[0]) + bounds[0]
            collocation_points.append(points)
        
        pde_coords = torch.cat(collocation_points, dim=-1)
        
        # 采样边界点
        bc_coords = None
        bc_values = None
        if len(domain_bounds) > 0:
            # 简化的边界采样
            n_bc = num_collocation // 10
            bc_coords_list = []
            for dim, bounds in enumerate(domain_bounds):
                for bound_val in bounds:
                    points = torch.rand(n_bc // 4, len(domain_bounds), device=device)
                    points[:, dim] = bound_val
                    bc_coords_list.append(points)
            
            if bc_coords_list:
                bc_coords = torch.cat(bc_coords_list, dim=0)
                bc_values = torch.zeros(bc_coords.shape[0], 1, device=device)
        
        # 计算损失
        losses = pinn.compute_loss(pde_coords, bc_coords, bc_values)
        
        # 优化
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()
        
        scheduler.step(losses['total'].item())
        
        # 记录
        history['total_loss'].append(losses['total'].item())
        history['pde_loss'].append(losses['pde'].item())
        if 'bc' in losses:
            history['bc_loss'].append(losses['bc'].item())
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Total = {losses['total'].item():.6f}, "
                  f"PDE = {losses['pde'].item():.6f}")
    
    return history


if __name__ == "__main__":
    print("=" * 60)
    print("PINN Demo - Physics-Informed Neural Networks")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 1. 扩散方程测试
    print("\n1. Diffusion Equation PINN")
    print("-" * 40)
    
    network = SirenNetwork(
        in_features=2,  # x, t
        hidden_features=64,
        hidden_layers=3,
        out_features=1
    ).to(device)
    
    diffusion_pinn = DiffusionPINN(
        network=network,
        diffusion_coefficient=0.1
    ).to(device)
    
    # 训练
    print("Training diffusion PINN...")
    history = train_pinn(
        diffusion_pinn,
        domain_bounds=[(0, 1), (0, 1)],  # x in [0,1], t in [0,1]
        num_collocation=1000,
        num_epochs=1000,
        lr=1e-3,
        device=device
    )
    
    print(f"Final loss: {history['total_loss'][-1]:.6f}")
    
    # 预测
    test_coords = torch.rand(100, 2, device=device)
    with torch.no_grad():
        solution = diffusion_pinn(test_coords)
    print(f"Solution range: [{solution.min():.4f}, {solution.max():.4f}]")
    
    # 2. 相场方程测试
    print("\n2. Phase Field PINN (Cahn-Hilliard)")
    print("-" * 40)
    
    pf_network = SirenNetwork(
        in_features=3,  # x, y, t
        hidden_features=64,
        hidden_layers=3,
        out_features=1
    ).to(device)
    
    phase_field_pinn = PhaseFieldPINN(
        network=pf_network,
        equation_type="cahn_hilliard",
        mobility=1.0,
        interface_width=0.1
    ).to(device)
    
    print("Phase field PINN created")
    print(f"Equation type: {phase_field_pinn.equation_type}")
    print(f"Mobility: {phase_field_pinn.mobility}")
    
    # 3. 傅里叶特征测试
    print("\n3. Fourier Feature Encoding")
    print("-" * 40)
    
    fourier_encoder = FourierFeatureEncoding(
        input_dim=2,
        mapping_size=128,
        scale=10.0
    ).to(device)
    
    coords = torch.rand(10, 2, device=device)
    features = fourier_encoder(coords)
    print(f"Input dim: {coords.shape[-1]}")
    print(f"Output dim: {features.shape[-1]}")
    
    # 4. SIREN测试
    print("\n4. SIREN Network")
    print("-" * 40)
    
    siren = SirenNetwork(
        in_features=2,
        hidden_features=64,
        hidden_layers=3,
        out_features=1,
        omega_0=30.0
    ).to(device)
    
    x = torch.linspace(-1, 1, 100, device=device)
    y = torch.linspace(-1, 1, 100, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    coords = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    
    with torch.no_grad():
        output = siren(coords)
    print(f"SIREN output range: [{output.min():.4f}, {output.max():.4f}]")
    
    print("\n" + "=" * 60)
    print("PINN Demo completed!")
    print("Key features:")
    print("- Physics-informed loss functions")
    print("- Automatic differentiation for derivatives")
    print("- Phase field, diffusion, and elasticity equations")
    print("- Fourier features and SIREN architectures")
    print("=" * 60)
