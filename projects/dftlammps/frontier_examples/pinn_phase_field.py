"""
pinn_phase_field.py
PINN求解相场方程

演示如何使用物理信息神经网络求解Cahn-Hilliard和Allen-Cahn相场方程,
用于模拟相分离和晶粒生长。

应用场景:
- 合金相分离
- 晶粒生长模拟
- 相变动力学
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import sys
sys.path.insert(0, '/root/.openclaw/workspace')

from dftlammps.frontier.pinns_for_pde import (
    PhaseFieldPINN, SirenNetwork, FourierFeatureEncoding,
    train_pinn
)


def solve_cahn_hilliard_1d(
    domain_size: float = 1.0,
    nx: int = 100,
    nt: int = 100,
    total_time: float = 1.0,
    mobility: float = 0.01,
    interface_width: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用PINN求解1D Cahn-Hilliard方程
    
    ∂φ/∂t = M ∇²(∂f/∂φ - ε²∇²φ)
    
    Args:
        domain_size: 空间域大小
        nx: 空间网格数
        nt: 时间步数
        total_time: 总时间
        mobility: 迁移率M
        interface_width: 界面宽度ε
    
    Returns:
        x, t, phi: 空间坐标、时间坐标、解场
    """
    print("=" * 60)
    print("Solving 1D Cahn-Hilliard Equation with PINN")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 创建网络
    print("\n[1] Creating SIREN network...")
    network = SirenNetwork(
        in_features=2,  # x, t
        hidden_features=128,
        hidden_layers=4,
        out_features=1,
        omega_0=30.0
    ).to(device)
    
    print(f"    Parameters: {sum(p.numel() for p in network.parameters()):,}")
    
    # 创建PINN
    print("\n[2] Setting up Phase-Field PINN...")
    pinn = PhaseFieldPINN(
        network=network,
        equation_type="cahn_hilliard",
        mobility=mobility,
        interface_width=interface_width
    ).to(device)
    
    # 生成训练数据
    print("\n[3] Generating training data...")
    
    # 空间和时间网格
    x = np.linspace(0, domain_size, nx)
    t = np.linspace(0, total_time, nt)
    
    # 配点
    np.random.seed(42)
    n_collocation = 5000
    x_collocation = torch.rand(n_collocation, 1, device=device) * domain_size
    t_collocation = torch.rand(n_collocation, 1, device=device) * total_time
    pde_coords = torch.cat([x_collocation, t_collocation], dim=-1)
    
    # 初始条件
    x_ic = torch.linspace(0, domain_size, nx, device=device).unsqueeze(-1)
    t_ic = torch.zeros_like(x_ic)
    ic_coords = torch.cat([x_ic, t_ic], dim=-1)
    
    # 初始条件: 小的随机扰动
    np.random.seed(42)
    ic_values = torch.tensor(
        0.5 + 0.1 * np.random.randn(nx, 1),
        dtype=torch.float32,
        device=device
    )
    
    # 边界条件 (Neumann: ∂φ/∂x = 0)
    bc_x_left = torch.zeros(50, 1, device=device)
    bc_x_right = torch.ones(50, 1, device=device) * domain_size
    bc_t = torch.rand(50, 1, device=device) * total_time
    
    bc_coords_left = torch.cat([bc_x_left, bc_t], dim=-1)
    bc_coords_right = torch.cat([bc_x_right, bc_t], dim=-1)
    bc_coords = torch.cat([bc_coords_left, bc_coords_right], dim=0)
    bc_values = torch.zeros(100, 1, device=device)  # 零梯度
    
    print(f"    Collocation points: {n_collocation}")
    print(f"    Initial condition points: {nx}")
    print(f"    Boundary condition points: {len(bc_coords)}")
    
    # 训练
    print("\n[4] Training PINN...")
    optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    
    n_epochs = 5000
    losses = []
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # PDE损失
        loss_pde = pinn.pde_loss(pde_coords[:, 0:1], pde_coords[:, 1:2])
        
        # 初始条件损失
        pred_ic = pinn(ic_coords)
        loss_ic = torch.nn.functional.mse_loss(pred_ic, ic_values)
        
        # 边界条件损失
        pred_bc = pinn(bc_coords)
        # Neumann边界条件: 梯度为0
        bc_coords_grad = bc_coords.requires_grad_(True)
        pred_bc_grad = pinn(bc_coords_grad)
        grad_bc = torch.autograd.grad(
            pred_bc_grad.sum(), bc_coords_grad,
            create_graph=True
        )[0]
        dphi_dx = grad_bc[:, 0:1]  # x方向的梯度
        loss_bc = torch.mean(dphi_dx ** 2)
        
        # 总损失
        loss = loss_pde + 10 * loss_ic + loss_bc
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        
        if epoch % 1000 == 0:
            print(f"    Epoch {epoch}: Loss = {loss.item():.6f}, "
                  f"PDE = {loss_pde.item():.6f}, IC = {loss_ic.item():.6f}")
    
    # 预测
    print("\n[5] Generating solution...")
    
    pinn.eval()
    with torch.no_grad():
        X, T = np.meshgrid(x, t)
        xt = np.stack([X.flatten(), T.flatten()], axis=-1)
        xt_tensor = torch.tensor(xt, dtype=torch.float32, device=device)
        
        phi_pred = pinn(xt_tensor).cpu().numpy()
        phi = phi_pred.reshape(nt, nx)
    
    print("    Solution generated!")
    
    return x, t, phi


def solve_allen_cahn_2d(
    domain_size: Tuple[float, float] = (1.0, 1.0),
    nx: int = 64,
    ny: int = 64,
    nt: int = 50,
    total_time: float = 0.5,
    mobility: float = 1.0,
    interface_width: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    使用PINN求解2D Allen-Cahn方程
    
    ∂φ/∂t = -M(∂f/∂φ - ε²∇²φ)
    """
    print("\n" + "=" * 60)
    print("Solving 2D Allen-Cahn Equation with PINN")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建网络
    network = SirenNetwork(
        in_features=3,  # x, y, t
        hidden_features=128,
        hidden_layers=4,
        out_features=1,
        omega_0=30.0
    ).to(device)
    
    # 创建PINN
    pinn = PhaseFieldPINN(
        network=network,
        equation_type="allen_cahn",
        mobility=mobility,
        interface_width=interface_width
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in pinn.parameters()):,}")
    
    # 简化: 使用更少的点
    n_collocation = 2000
    
    x_collocation = torch.rand(n_collocation, 1, device=device) * domain_size[0]
    y_collocation = torch.rand(n_collocation, 1, device=device) * domain_size[1]
    t_collocation = torch.rand(n_collocation, 1, device=device) * total_time
    pde_coords = torch.cat([x_collocation, y_collocation, t_collocation], dim=-1)
    
    # 训练 (简化版)
    print("Training...")
    optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    
    for epoch in range(2000):
        optimizer.zero_grad()
        
        loss_pde = pinn.pde_loss(pde_coords[:, 0:2], pde_coords[:, 2:3])
        
        loss_pde.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"  Epoch {epoch}: Loss = {loss_pde.item():.6f}")
    
    # 预测
    x = np.linspace(0, domain_size[0], nx)
    y = np.linspace(0, domain_size[1], ny)
    t = np.linspace(0, total_time, nt)
    
    pinn.eval()
    solutions = []
    
    with torch.no_grad():
        for ti in t:
            X, Y = np.meshgrid(x, y)
            xy = np.stack([X.flatten(), Y.flatten()], axis=-1)
            t_vec = np.ones((xy.shape[0], 1)) * ti
            xyt = np.concatenate([xy, t_vec], axis=-1)
            
            xyt_tensor = torch.tensor(xyt, dtype=torch.float32, device=device)
            phi_pred = pinn(xyt_tensor).cpu().numpy()
            solutions.append(phi_pred.reshape(ny, nx))
    
    return x, y, t, np.array(solutions)


def visualize_phase_field_1d(
    x: np.ndarray,
    t: np.ndarray,
    phi: np.ndarray,
    output_file: str = 'phase_field_1d.png'
):
    """可视化1D相场结果"""
    print(f"\n[6] Visualizing 1D results...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 时空演化
    ax1 = axes[0]
    im = ax1.imshow(phi, aspect='auto', origin='lower',
                    extent=[x.min(), x.max(), t.min(), t.max()],
                    cmap='RdBu_r', vmin=-0.5, vmax=1.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_title('Phase Field Evolution (Cahn-Hilliard)')
    plt.colorbar(im, ax=ax1, label='φ')
    
    # 不同时刻的剖面
    ax2 = axes[1]
    for i in [0, len(t)//4, len(t)//2, 3*len(t)//4, -1]:
        ax2.plot(x, phi[i], label=f't={t[i]:.2f}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('φ')
    ax2.set_title('Profiles at Different Times')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"    Saved to {output_file}")


def visualize_phase_field_2d(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    phi: np.ndarray,
    output_file: str = 'phase_field_2d.png'
):
    """可视化2D相场结果"""
    print(f"\nVisualizing 2D results...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 显示不同时刻
    time_indices = np.linspace(0, len(t)-1, 6, dtype=int)
    
    for idx, ti in enumerate(time_indices):
        ax = axes[idx]
        im = ax.imshow(phi[ti], origin='lower',
                      extent=[x.min(), x.max(), y.min(), y.max()],
                      cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f't = {t[ti]:.3f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"    Saved to {output_file}")


def analyze_microstructure(
    phi: np.ndarray,
    threshold: float = 0.0
) -> Dict[str, float]:
    """
    分析微观结构特征
    
    计算域大小、界面数量等
    """
    print("\n[7] Analyzing microstructure...")
    
    # 二值化
    binary = (phi > threshold).astype(int)
    
    # 计算体积分数
    volume_fraction = np.mean(binary)
    
    # 计算界面长度 (简化)
    if phi.ndim == 2:
        grad_x = np.abs(np.diff(phi, axis=1, append=phi[:, -1:]))
        grad_y = np.abs(np.diff(phi, axis=0, append=phi[-1:, :]))
        interface_length = np.sum((grad_x + grad_y) > 0.1)
    else:
        interface_length = 0.0
    
    stats = {
        'volume_fraction': float(volume_fraction),
        'interface_length': float(interface_length),
        'phi_min': float(np.min(phi)),
        'phi_max': float(np.max(phi)),
        'phi_mean': float(np.mean(phi)),
        'phi_std': float(np.std(phi))
    }
    
    print(f"    Volume fraction: {volume_fraction:.3f}")
    print(f"    Interface length: {interface_length:.1f}")
    print(f"    φ range: [{stats['phi_min']:.3f}, {stats['phi_max']:.3f}]")
    
    return stats


def compare_with_finite_difference(
    domain_size: float = 1.0,
    nx: int = 100,
    nt: int = 100,
    total_time: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    比较PINN和有限差分解
    """
    print("\n" + "=" * 60)
    print("Comparing PINN vs Finite Difference")
    print("=" * 60)
    
    # PINN解
    x, t, phi_pinn = solve_cahn_hilliard_1d(
        domain_size, nx, nt, total_time
    )
    
    # 简化有限差分 (显式格式)
    phi_fd = np.zeros((nt, nx))
    dx = domain_size / (nx - 1)
    dt = total_time / (nt - 1)
    
    # 初始条件
    np.random.seed(42)
    phi_fd[0] = 0.5 + 0.1 * np.random.randn(nx)
    
    # 时间推进
    M = 0.01
    epsilon = 0.05
    
    for n in range(nt - 1):
        for i in range(1, nx - 1):
            # 化学势
            mu = phi_fd[n, i]**3 - phi_fd[n, i] - epsilon**2 * (
                phi_fd[n, i+1] - 2*phi_fd[n, i] + phi_fd[n, i-1]
            ) / dx**2
            
            # Laplacian of mu
            laplacian_mu = (mu - 2*phi_fd[n, i] + phi_fd[n, i]) / dx**2
            
            phi_fd[n+1, i] = phi_fd[n, i] + dt * M * laplacian_mu
        
        # 边界条件
        phi_fd[n+1, 0] = phi_fd[n+1, 1]
        phi_fd[n+1, -1] = phi_fd[n+1, -2]
    
    # 比较
    error = np.abs(phi_pinn - phi_fd)
    mean_error = np.mean(error)
    
    print(f"\nMean absolute error: {mean_error:.6f}")
    print(f"Max error: {np.max(error):.6f}")
    
    # 可视化比较
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1 = axes[0]
    im1 = ax1.imshow(phi_pinn, aspect='auto', origin='lower', cmap='RdBu_r')
    ax1.set_title('PINN Solution')
    plt.colorbar(im1, ax=ax1)
    
    ax2 = axes[1]
    im2 = ax2.imshow(phi_fd, aspect='auto', origin='lower', cmap='RdBu_r')
    ax2.set_title('Finite Difference Solution')
    plt.colorbar(im2, ax=ax2)
    
    ax3 = axes[2]
    im3 = ax3.imshow(error, aspect='auto', origin='lower', cmap='hot')
    ax3.set_title(f'Absolute Error (Mean: {mean_error:.4f})')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=150)
    print("    Comparison saved to comparison.png")
    
    return phi_pinn, phi_fd


if __name__ == "__main__":
    print("=" * 60)
    print("PINN Phase Field Demo")
    print("=" * 60)
    
    # 演示1: 1D Cahn-Hilliard
    print("\n" + "#" * 60)
    print("# Demo 1: 1D Cahn-Hilliard Equation")
    print("#" * 60)
    
    x, t, phi = solve_cahn_hilliard_1d(
        domain_size=1.0,
        nx=100,
        nt=50,
        total_time=0.5,
        mobility=0.01,
        interface_width=0.05
    )
    
    # 可视化
    visualize_phase_field_1d(x, t, phi)
    
    # 分析
    stats = analyze_microstructure(phi[-1])
    
    # 演示2: 2D Allen-Cahn
    print("\n" + "#" * 60)
    print("# Demo 2: 2D Allen-Cahn Equation")
    print("#" * 60)
    
    x2d, y2d, t2d, phi2d = solve_allen_cahn_2d(
        domain_size=(1.0, 1.0),
        nx=32,
        ny=32,
        nt=10,
        total_time=0.1
    )
    
    visualize_phase_field_2d(x2d, y2d, t2d, phi2d)
    
    # 演示3: 比较
    print("\n" + "#" * 60)
    print("# Demo 3: PINN vs Finite Difference")
    print("#" * 60)
    
    phi_pinn, phi_fd = compare_with_finite_difference(
        domain_size=1.0,
        nx=50,
        nt=50,
        total_time=0.05
    )
    
    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
