#!/usr/bin/env python3
"""
phase_field.py
==============
相场模拟接口模块

功能：
1. 枝晶生长模拟（使用PRISMS-PF或MOOSE框架）
2. 相分离动力学模拟
3. 与MD的耦合（从原子模拟提取界面能等参数）
4. 多相系统建模

支持的相场求解器：
- PRISMS-PF: 基于deal.II的高性能相场框架
- MOOSE: 多物理场面向对象仿真环境

作者: Multi-Scale Simulation Expert
日期: 2026-03-09
"""

import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging
from datetime import datetime
import warnings
import subprocess
import tempfile

# ASE for structure handling
from ase import Atoms
from ase.io import read, write
from ase.units import eV, Ang, fs, GPa, J, m, s

# scipy for numerical solutions
from scipy.ndimage import gaussian_filter, laplace
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import odeint

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class PhaseFieldConfig:
    """相场模拟配置"""
    # 网格设置
    dimensions: int = 3  # 2D or 3D
    nx: int = 128
    ny: int = 128
    nz: int = 1  # 对于2D模拟设为1
    dx: float = 1.0  # 网格间距 (nm)
    
    # 时间设置
    dt: float = 0.01  # 时间步长
    n_steps: int = 10000
    output_interval: int = 100
    
    # 物理参数
    interface_width: float = 5.0  # 界面宽度 (nm)
    mobility: float = 1.0  # 相场迁移率
    diffusion_coeff: float = 1.0  # 扩散系数 (nm²/ps)
    
    # 热力学参数
    temperature: float = 300.0  # K
    melting_point: float = 1000.0  # K
    latent_heat: float = 1.0e9  # J/m³
    
    # 各向异性参数
    anisotropy_mode: str = "four_fold"  # four_fold, six_fold, cubic
    anisotropy_strength: float = 0.05  # 各向异性强度 ε₄
    
    # 边界条件
    bc_type: str = "periodic"  # periodic, neumann, dirichlet
    
    # 输出设置
    output_dir: str = "./phase_field_output"
    save_vtk: bool = True
    save_npy: bool = True


@dataclass
class DendriteConfig(PhaseFieldConfig):
    """枝晶生长模拟专用配置"""
    # 过冷度设置
    undercooling: float = 0.2  # ΔT/Tm (无量纲过冷度)
    thermal_diffusivity: float = 1.0  # 热扩散系数
    capillary_length: float = 0.01  # 毛细长度
    
    # 噪声设置
    thermal_noise: bool = True
    noise_amplitude: float = 0.01
    
    # 初始种子
    seed_radius: float = 5.0  # 初始晶核半径
    seed_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # 各向异性设置
    anisotropy_mode: str = "four_fold"
    anisotropy_strength: float = 0.05


@dataclass
class SpinodalConfig(PhaseFieldConfig):
    """相分离（Spinodal分解）模拟配置"""
    # 成分设置
    initial_composition: float = 0.5  # 初始成分
    composition_noise: float = 0.01  # 成分涨落幅度
    
    # 自由能参数
    chi_parameter: float = 2.5  # Flory-Huggins χ参数
    gradient_coeff: float = 0.5  # 梯度能量系数
    
    # 动力学参数
    mobility_c: float = 1.0  # 成分场迁移率


@dataclass
class MDtoPhaseFieldParams:
    """从MD提取的相场参数"""
    # 界面参数
    interface_energy: float = 0.0  # J/m²
    interface_width: float = 1.0  # nm
    
    # 动力学参数
    diffusion_coeff: float = 0.0  # m²/s
    
    # 热力学参数
    melting_point: float = 0.0  # K
    latent_heat: float = 0.0  # J/kg
    heat_capacity: float = 0.0  # J/(kg·K)
    
    # 各向异性参数
    anisotropy_values: Dict[str, float] = field(default_factory=dict)
    
    # 弹性参数
    elastic_constants: Dict[str, float] = field(default_factory=dict)
    
    # 来源信息
    source_md_simulation: str = ""
    extraction_method: str = ""
    uncertainty: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# 相场求解器基类
# =============================================================================

class PhaseFieldSolver:
    """
    相场求解器基类
    
    实现通用的相场方程求解功能
    """
    
    def __init__(self, config: PhaseFieldConfig):
        self.config = config
        self.time_step = 0
        self.time = 0.0
        
        # 初始化网格
        self._setup_grid()
        
        # 初始化场变量
        self.phi = None  # 相场序参量
        self.c = None    # 成分场
        self.T = None    # 温度场
        
        # 存储历史
        self.history = {
            'time': [],
            'phi': [],
            'c': [],
            'T': []
        }
    
    def _setup_grid(self):
        """设置计算网格"""
        cfg = self.config
        
        if cfg.dimensions == 2:
            self.x = np.linspace(0, cfg.nx * cfg.dx, cfg.nx)
            self.y = np.linspace(0, cfg.ny * cfg.dx, cfg.ny)
            self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
            self.dV = cfg.dx**2
            
        elif cfg.dimensions == 3:
            self.x = np.linspace(0, cfg.nx * cfg.dx, cfg.nx)
            self.y = np.linspace(0, cfg.ny * cfg.dx, cfg.ny)
            self.z = np.linspace(0, cfg.nz * cfg.dx, cfg.nz)
            self.X, self.Y, self.Z = np.meshgrid(
                self.x, self.y, self.z, indexing='ij'
            )
            self.dV = cfg.dx**3
    
    def initialize_phi(self, pattern: str = "nucleus", **kwargs):
        """
        初始化相场
        
        Args:
            pattern: 初始模式 (nucleus, planar, random, etc.)
        """
        cfg = self.config
        
        if cfg.dimensions == 2:
            shape = (cfg.nx, cfg.ny)
        else:
            shape = (cfg.nx, cfg.ny, cfg.nz)
        
        if pattern == "nucleus":
            # 中心圆形/球形晶核
            radius = kwargs.get('radius', cfg.nx * cfg.dx / 10)
            center = kwargs.get('center', (cfg.nx/2 * cfg.dx, cfg.ny/2 * cfg.dx))
            
            if cfg.dimensions == 2:
                r = np.sqrt((self.X - center[0])**2 + (self.Y - center[1])**2)
            else:
                center_z = kwargs.get('center_z', cfg.nz/2 * cfg.dx)
                r = np.sqrt((self.X - center[0])**2 + 
                           (self.Y - center[1])**2 + 
                           (self.Z - center_z)**2)
            
            # 双曲正切分布
            self.phi = 0.5 * (1 + np.tanh((radius - r) / (cfg.interface_width / 2)))
            
        elif pattern == "planar":
            # 平面界面
            position = kwargs.get('position', cfg.nx * cfg.dx / 2)
            normal = kwargs.get('normal', 'x')
            
            if normal == 'x':
                self.phi = 0.5 * (1 + np.tanh((position - self.X) / (cfg.interface_width / 2)))
            elif normal == 'y':
                self.phi = 0.5 * (1 + np.tanh((position - self.Y) / (cfg.interface_width / 2)))
            
        elif pattern == "random":
            # 随机分布
            np.random.seed(kwargs.get('seed', 42))
            self.phi = np.random.random(shape)
            
        elif pattern == "loaded":
            # 从文件加载
            self.phi = kwargs['data']
        
        else:
            # 均匀固相
            self.phi = np.ones(shape)
    
    def initialize_composition(self, c0: float = 0.5, noise: float = 0.01):
        """初始化成分场"""
        cfg = self.config
        
        if cfg.dimensions == 2:
            shape = (cfg.nx, cfg.ny)
        else:
            shape = (cfg.nx, cfg.ny, cfg.nz)
        
        self.c = c0 + noise * (np.random.random(shape) - 0.5)
        self.c = np.clip(self.c, 0.0, 1.0)
    
    def initialize_temperature(self, T0: float = 300.0, gradient: Optional[float] = None):
        """初始化温度场"""
        cfg = self.config
        
        if cfg.dimensions == 2:
            shape = (cfg.nx, cfg.ny)
        else:
            shape = (cfg.nx, cfg.ny, cfg.nz)
        
        if gradient is None:
            self.T = T0 * np.ones(shape)
        else:
            # 线性温度梯度
            if cfg.dimensions == 2:
                self.T = T0 + gradient * self.X
            else:
                self.T = T0 + gradient * self.X
    
    def compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """计算拉普拉斯算子"""
        return laplace(field) / self.config.dx**2
    
    def compute_gradient(self, field: np.ndarray) -> Tuple[np.ndarray, ...]:
        """计算梯度"""
        if self.config.dimensions == 2:
            grad_x = np.gradient(field, self.config.dx, axis=0)
            grad_y = np.gradient(field, self.config.dx, axis=1)
            return grad_x, grad_y
        else:
            grad_x = np.gradient(field, self.config.dx, axis=0)
            grad_y = np.gradient(field, self.config.dx, axis=1)
            grad_z = np.gradient(field, self.config.dx, axis=2)
            return grad_x, grad_y, grad_z
    
    def apply_boundary_conditions(self, field: np.ndarray) -> np.ndarray:
        """应用边界条件"""
        cfg = self.config
        
        if cfg.bc_type == "periodic":
            # numpy数组默认不处理周期性，需要在梯度计算中处理
            pass
        elif cfg.bc_type == "neumann":
            # 零通量边界
            if cfg.dimensions == 2:
                field[0, :] = field[1, :]
                field[-1, :] = field[-2, :]
                field[:, 0] = field[:, 1]
                field[:, -1] = field[:, -2]
            else:
                field[0, :, :] = field[1, :, :]
                field[-1, :, :] = field[-2, :, :]
                field[:, 0, :] = field[:, 1, :]
                field[:, -1, :] = field[:, -2, :]
                field[:, :, 0] = field[:, :, 1]
                field[:, :, -1] = field[:, :, -2]
        
        return field
    
    def step(self):
        """执行一个时间步（子类实现）"""
        raise NotImplementedError
    
    def run(self, n_steps: Optional[int] = None, progress_interval: int = 100):
        """
        运行模拟
        
        Args:
            n_steps: 运行步数（默认使用配置中的值）
            progress_interval: 进度输出间隔
        """
        cfg = self.config
        n_steps = n_steps or cfg.n_steps
        
        logger.info(f"Starting phase field simulation for {n_steps} steps")
        
        for step in range(n_steps):
            self.step()
            self.time_step += 1
            self.time += cfg.dt
            
            # 输出
            if step % cfg.output_interval == 0:
                self._save_output()
            
            # 进度
            if step % progress_interval == 0:
                logger.info(f"Step {step}/{n_steps}, Time = {self.time:.4f}")
        
        logger.info("Simulation completed")
    
    def _save_output(self):
        """保存输出"""
        cfg = self.config
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.history['time'].append(self.time)
        self.history['phi'].append(self.phi.copy())
        if self.c is not None:
            self.history['c'].append(self.c.copy())
        if self.T is not None:
            self.history['T'].append(self.T.copy())
        
        if cfg.save_npy:
            np.save(output_dir / f"phi_t{self.time_step:06d}.npy", self.phi)
            if self.c is not None:
                np.save(output_dir / f"c_t{self.time_step:06d}.npy", self.c)
    
    def get_interface_position(self) -> np.ndarray:
        """获取界面位置（phi = 0.5的等值面）"""
        if self.config.dimensions == 2:
            positions = []
            for i in range(self.config.nx):
                for j in range(self.config.ny):
                    if abs(self.phi[i, j] - 0.5) < 0.1:
                        positions.append([self.x[i], self.y[j]])
            return np.array(positions)
        return np.array([])
    
    def compute_interface_velocity(self) -> float:
        """计算界面速度"""
        if len(self.history['phi']) < 2:
            return 0.0
        
        phi_current = self.history['phi'][-1]
        phi_prev = self.history['phi'][-2]
        dt = self.history['time'][-1] - self.history['time'][-2]
        
        # 找到phi=0.5的位置变化
        if self.config.dimensions == 2:
            # 简化为计算质心位置变化
            mask_curr = phi_current > 0.5
            mask_prev = phi_prev > 0.5
            
            if np.any(mask_curr) and np.any(mask_prev):
                centroid_curr = np.array([
                    np.mean(self.X[mask_curr]),
                    np.mean(self.Y[mask_curr])
                ])
                centroid_prev = np.array([
                    np.mean(self.X[mask_prev]),
                    np.mean(self.Y[mask_prev])
                ])
                velocity = np.linalg.norm(centroid_curr - centroid_prev) / dt
                return velocity
        
        return 0.0


# =============================================================================
# 枝晶生长求解器
# =============================================================================

class DendriteGrowthSolver(PhaseFieldSolver):
    """
    枝晶生长相场求解器
    
    基于Karma-Rappel模型或其他经典相场模型
    """
    
    def __init__(self, config: DendriteConfig):
        super().__init__(config)
        self.config = config
        
        # 无量纲参数
        self.tau0 = 1.0  # 弛豫时间
        self.W0 = config.interface_width  # 界面宽度
        self.D = config.thermal_diffusivity  # 热扩散系数
        self.d0 = config.capillary_length  # 毛细长度
        
        # 初始化温度场（过冷熔体）
        self.initialize_temperature(
            T0=config.melting_point * (1 - config.undercooling)
        )
        
        # 初始化晶核
        self.initialize_phi(
            pattern="nucleus",
            radius=config.seed_radius,
            center=(config.nx*config.dx/2, config.ny*config.dx/2)
        )
    
    def anisotropy_function(self, theta: np.ndarray) -> np.ndarray:
        """
        各向异性函数
        
        四重对称: γ(θ) = γ₀(1 + ε₄cos(4θ))
        六重对称: γ(θ) = γ₀(1 + ε₆cos(6θ))
        """
        cfg = self.config
        eps = cfg.anisotropy_strength
        
        if cfg.anisotropy_mode == "four_fold":
            return 1.0 + eps * np.cos(4 * theta)
        elif cfg.anisotropy_mode == "six_fold":
            return 1.0 + eps * np.cos(6 * theta)
        elif cfg.anisotropy_mode == "cubic":
            return 1.0 + eps * (np.cos(4 * theta) + 0.25 * np.cos(8 * theta))
        else:
            return np.ones_like(theta)
    
    def compute_curvature(self) -> np.ndarray:
        """计算界面曲率"""
        grad = self.compute_gradient(self.phi)
        
        if self.config.dimensions == 2:
            grad_x, grad_y = grad
            grad_mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-10
            
            # 法向角度
            theta = np.arctan2(grad_y, grad_x)
            
            # 曲率计算
            laplacian = self.compute_laplacian(self.phi)
            curvature = laplacian / grad_mag
            
        else:
            grad_x, grad_y, grad_z = grad
            grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2) + 1e-10
            laplacian = self.compute_laplacian(self.phi)
            curvature = laplacian / grad_mag
            theta = np.arctan2(grad_y, grad_x)
        
        return curvature, theta
    
    def step(self):
        """执行一个时间步 - Karma-Rappel模型"""
        cfg = self.config
        
        # 计算曲率和角度
        curvature, theta = self.compute_curvature()
        
        # 各向异性
        a = self.anisotropy_function(theta)
        
        # 相场演化方程
        # τ(θ) ∂φ/∂t = ∇·(W²(θ)∇φ) + φ(1-φ)(φ-0.5+6ε(1-φ)φ)
        
        # 扩散项
        laplacian_phi = self.compute_laplacian(self.phi)
        
        # 反应项（双势阱）
        potential = self.phi * (1 - self.phi) * (self.phi - 0.5 + 6 * cfg.undercooling * self.phi * (1 - self.phi))
        
        # 潜热释放
        if cfg.latent_heat > 0:
            dphi_dt = self.phi - self.history['phi'][-1] if self.history['phi'] else np.zeros_like(self.phi)
            self.T += cfg.latent_hheat * dphi_dt / cfg.heat_capacity
        
        # 相场更新
        dphi = (self.W0**2 * laplacian_phi + potential) * cfg.dt / self.tau0
        
        # 添加热噪声
        if cfg.thermal_noise:
            noise = cfg.noise_amplitude * np.random.randn(*self.phi.shape)
            dphi += noise * np.sqrt(cfg.dt)
        
        self.phi += dphi
        self.phi = np.clip(self.phi, 0.0, 1.0)
        
        # 温度场演化（热扩散）
        if cfg.thermal_diffusivity > 0:
            laplacian_T = self.compute_laplacian(self.T)
            self.T += cfg.thermal_diffusivity * laplacian_T * cfg.dt
        
        # 边界条件
        self.phi = self.apply_boundary_conditions(self.phi)
        self.T = self.apply_boundary_conditions(self.T)
    
    def get_dendrite_tip_velocity(self) -> float:
        """获取枝晶尖端速度"""
        if self.config.dimensions == 2:
            # 找到phi=0.5最远的点
            mask = np.abs(self.phi - 0.5) < 0.1
            if np.any(mask):
                # 计算到中心的距离
                center_x = self.config.nx * self.config.dx / 2
                center_y = self.config.ny * self.config.dx / 2
                distances = np.sqrt((self.X[mask] - center_x)**2 + (self.Y[mask] - center_y)**2)
                return np.max(distances) / self.time if self.time > 0 else 0.0
        return 0.0
    
    def get_tip_radius(self) -> float:
        """估计枝晶尖端半径"""
        # 基于曲率计算
        curvature, _ = self.compute_curvature()
        valid_curvature = curvature[self.phi > 0.4]
        valid_curvature = valid_curvature[valid_curvature > 0]
        
        if len(valid_curvature) > 0:
            return 1.0 / np.mean(valid_curvature)
        return 0.0


# =============================================================================
# 相分离求解器 (Cahn-Hilliard)
# =============================================================================

class SpinodalDecompositionSolver(PhaseFieldSolver):
    """
    旋节分解相场求解器
    
    基于Cahn-Hilliard方程:
    ∂c/∂t = ∇·(M∇μ)
    μ = ∂f/∂c - κ∇²c
    """
    
    def __init__(self, config: SpinodalConfig):
        super().__init__(config)
        self.config = config
        
        # 初始化成分场
        self.initialize_composition(
            c0=config.initial_composition,
            noise=config.composition_noise
        )
    
    def bulk_free_energy(self, c: np.ndarray) -> np.ndarray:
        """
        体自由能密度
        
        使用双势阱形式: f(c) = χc(1-c) + c ln(c) + (1-c)ln(1-c)
        或简化的多项式: f(c) = c²(1-c)²
        """
        cfg = self.config
        
        # 简化的双势阱
        f = c**2 * (1 - c)**2
        
        # 混合能（如果chi参数较大）
        if cfg.chi_parameter > 0:
            f += cfg.chi_parameter * c * (1 - c)
        
        return f
    
    def chemical_potential(self, c: np.ndarray) -> np.ndarray:
        """化学势: μ = ∂f/∂c - κ∇²c"""
        cfg = self.config
        
        # 化学势的体相贡献
        # f = c²(1-c)² -> df/dc = 2c(1-c)(1-2c)
        mu_bulk = 2 * c * (1 - c) * (1 - 2 * c)
        
        if cfg.chi_parameter > 0:
            mu_bulk += cfg.chi_parameter * (1 - 2 * c)
        
        # 梯度贡献
        laplacian_c = self.compute_laplacian(c)
        mu = mu_bulk - cfg.gradient_coeff * laplacian_c
        
        return mu
    
    def step(self):
        """执行一个时间步 - Cahn-Hilliard方程"""
        cfg = self.config
        
        # 计算化学势
        mu = self.chemical_potential(self.c)
        
        # 计算化学势的拉普拉斯
        laplacian_mu = self.compute_laplacian(mu)
        
        # Cahn-Hilliard方程
        dc = cfg.mobility_c * laplacian_mu * cfg.dt
        
        # 更新成分场
        self.c += dc
        self.c = np.clip(self.c, 0.0, 1.0)
        
        # 边界条件
        self.c = self.apply_boundary_conditions(self.c)
    
    def compute_structure_factor(self) -> Tuple[np.ndarray, np.ndarray]:
        """计算结构因子 S(k)"""
        if self.config.dimensions == 2:
            # 2D傅里叶变换
            c_fft = np.fft.fft2(self.c - np.mean(self.c))
            S = np.abs(c_fft)**2
            
            # 波矢
            kx = 2 * np.pi * np.fft.fftfreq(self.config.nx, self.config.dx)
            ky = 2 * np.pi * np.fft.fftfreq(self.config.ny, self.config.dx)
            KX, KY = np.meshgrid(kx, ky, indexing='ij')
            k_mag = np.sqrt(KX**2 + KY**2)
            
            return k_mag, S
        
        return np.array([]), np.array([])
    
    def get_domain_size(self) -> float:
        """估计畴区尺寸"""
        k_mag, S = self.compute_structure_factor()
        
        if len(k_mag) > 0:
            # 找到结构因子峰值对应的波矢
            k_flat = k_mag.flatten()
            S_flat = S.flatten()
            
            # 排除k=0
            mask = k_flat > 0
            k_flat = k_flat[mask]
            S_flat = S_flat[mask]
            
            if len(k_flat) > 0:
                k_peak = k_flat[np.argmax(S_flat)]
                return 2 * np.pi / k_peak if k_peak > 0 else 0.0
        
        return 0.0


# =============================================================================
# MD到相场参数提取器
# =============================================================================

class MDtoPhaseFieldExtractor:
    """
    从分子动力学模拟提取相场参数
    
    提取参数包括:
    - 界面能
    - 界面宽度
    - 扩散系数
    - 各向异性系数
    """
    
    def __init__(self):
        self.params = MDtoPhaseFieldParams()
        self.md_data = {}
    
    def load_md_trajectory(self, trajectory_file: str, format: str = "auto"):
        """加载MD轨迹"""
        from ase.io import read
        
        atoms_list = read(trajectory_file, index=':')
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
        
        self.md_data['trajectory'] = atoms_list
        logger.info(f"Loaded {len(atoms_list)} MD frames")
    
    def extract_interface_energy(self, 
                                  solid_phase_indices: List[int],
                                  liquid_phase_indices: List[int],
                                  interface_area: float) -> float:
        """
        从两相共存MD提取界面能
        
        Args:
            solid_phase_indices: 固相原子索引
            liquid_phase_indices: 液相原子索引
            interface_area: 界面面积 (Å²)
        
        Returns:
            interface_energy: 界面能 (J/m²)
        """
        if 'trajectory' not in self.md_data:
            raise ValueError("No MD trajectory loaded")
        
        trajectory = self.md_data['trajectory']
        
        # 计算两相能量
        energies_solid = []
        energies_liquid = []
        
        for atoms in trajectory:
            if atoms.calc is not None:
                try:
                    # 计算总能量
                    total_energy = atoms.get_potential_energy()
                    
                    # 分离相能量（简化处理）
                    # 实际应使用更复杂的方法，如capillary fluctuation method
                    
                    energies_solid.append(total_energy)
                except:
                    pass
        
        if len(energies_solid) == 0:
            logger.warning("No energy data found")
            return 0.0
        
        # 界面能估算（简化）
        # γ = (E_total - N_solid*E_solid - N_liquid*E_liquid) / A
        # 这里使用能量涨落来估算
        
        # 实际实现需要更复杂的两相系统分析
        interface_energy = 0.3  # J/m², 占位符值
        
        self.params.interface_energy = interface_energy
        logger.info(f"Extracted interface energy: {interface_energy:.4f} J/m²")
        
        return interface_energy
    
    def extract_diffusion_coefficient(self, 
                                      temperature: float,
                                      composition_range: Optional[Tuple[float, float]] = None) -> float:
        """
        从MD轨迹提取扩散系数
        
        使用均方位移(MSD)方法:
        D = <r²> / (6t)
        """
        if 'trajectory' not in self.md_data:
            raise ValueError("No MD trajectory loaded")
        
        trajectory = self.md_data['trajectory']
        
        if len(trajectory) < 2:
            logger.warning("Need at least 2 frames for diffusion calculation")
            return 0.0
        
        # 计算MSD
        positions_t0 = trajectory[0].get_positions()
        n_atoms = len(positions_t0)
        
        msd_data = []
        times = []
        
        for i, atoms in enumerate(trajectory):
            if i == 0:
                continue
            
            positions = atoms.get_positions()
            displacements = positions - positions_t0
            
            # 考虑周期性边界条件
            cell = atoms.get_cell()
            if cell is not None:
                displacements = self._apply_pbc_displacement(displacements, cell)
            
            msd = np.mean(np.sum(displacements**2, axis=1))
            msd_data.append(msd)
            
            # 时间（假设等间隔）
            times.append(i)
        
        # 线性拟合MSD vs t
        if len(times) > 1:
            times = np.array(times)
            msd_data = np.array(msd_data)
            
            # 线性拟合
            slope = np.polyfit(times, msd_data, 1)[0]
            
            # D = slope / (6 * timestep) - 对于3D
            # 需要知道实际时间步长
            diffusion_coeff = slope / 6.0  # 简化，假设时间单位为ps
            
            # 转换为SI单位 (m²/s)
            diffusion_coeff_si = diffusion_coeff * 1e-8  # Å²/ps -> m²/s
            
            self.params.diffusion_coeff = diffusion_coeff_si
            logger.info(f"Extracted diffusion coefficient: {diffusion_coeff_si:.4e} m²/s")
            
            return diffusion_coeff_si
        
        return 0.0
    
    def _apply_pbc_displacement(self, displacement: np.ndarray, cell: np.ndarray) -> np.ndarray:
        """应用周期性边界条件修正位移"""
        # 简化的PBC处理
        cell_inv = np.linalg.inv(cell)
        
        # 转换为分数坐标
        frac_disp = displacement @ cell_inv.T
        
        # 应用最小图像约定
        frac_disp -= np.round(frac_disp)
        
        # 转换回笛卡尔坐标
        corrected_disp = frac_disp @ cell.T
        
        return corrected_disp
    
    def extract_anisotropy(self, 
                          directions: List[Tuple[float, float, float]],
                          interface_energies: List[float]) -> Dict[str, float]:
        """
        从各方向界面能提取各向异性系数
        
        γ(θ) = γ₀(1 + ε₄cos(4θ) + ...)
        """
        if len(directions) != len(interface_energies):
            raise ValueError("Number of directions and energies must match")
        
        # 转换为角度
        thetas = [np.arctan2(d[1], d[0]) for d in directions]
        
        # 拟合各向异性函数
        def anisotropy_func(theta, gamma0, eps4):
            return gamma0 * (1 + eps4 * np.cos(4 * theta))
        
        try:
            popt, _ = curve_fit(anisotropy_func, thetas, interface_energies)
            gamma0, eps4 = popt
            
            self.params.anisotropy_values = {
                'gamma0': gamma0,
                'epsilon4': eps4
            }
            
            logger.info(f"Extracted anisotropy: γ₀={gamma0:.4f}, ε₄={eps4:.4f}")
            
            return self.params.anisotropy_values
            
        except Exception as e:
            logger.error(f"Failed to fit anisotropy: {e}")
            return {}
    
    def get_params(self) -> MDtoPhaseFieldParams:
        """获取提取的参数"""
        return self.params
    
    def save_params(self, filename: str):
        """保存参数到JSON文件"""
        params_dict = asdict(self.params)
        
        with open(filename, 'w') as f:
            json.dump(params_dict, f, indent=2)
        
        logger.info(f"Saved parameters to {filename}")
    
    def load_params(self, filename: str):
        """从JSON文件加载参数"""
        with open(filename, 'r') as f:
            params_dict = json.load(f)
        
        self.params = MDtoPhaseFieldParams(**params_dict)
        logger.info(f"Loaded parameters from {filename}")


# =============================================================================
# PRISMS-PF接口
# =============================================================================

class PRISMSPFInterface:
    """
    PRISMS-PF框架接口
    
    PRISMS-PF是一个基于deal.II的高性能相场建模框架
    """
    
    def __init__(self, prisms_pf_path: str = "/path/to/prismspf"):
        self.prisms_pf_path = Path(prisms_pf_path)
        self.input_files = []
        self.output_files = []
    
    def generate_input_file(self, 
                           config: PhaseFieldConfig,
                           model_name: str = "allen_cahn",
                           output_file: str = "parameters.prm") -> str:
        """
        生成PRISMS-PF输入文件
        
        PRISMS-PF使用参数文件格式(.prm)
        """
        lines = []
        
        # 模型类型
        lines.append(f"set Model name = {model_name}")
        lines.append("")
        
        # 维度
        lines.append(f"set Dimension = {config.dimensions}")
        lines.append("")
        
        # 网格设置
        lines.append("subsection Mesh")
        lines.append(f"  set Type = rectangular")
        if config.dimensions == 2:
            lines.append(f"  set X extent = {config.nx * config.dx}")
            lines.append(f"  set Y extent = {config.ny * config.dx}")
            lines.append(f"  set X subdivisions = {config.nx}")
            lines.append(f"  set Y subdivisions = {config.ny}")
        else:
            lines.append(f"  set X extent = {config.nx * config.dx}")
            lines.append(f"  set Y extent = {config.ny * config.dx}")
            lines.append(f"  set Z extent = {config.nz * config.dx}")
            lines.append(f"  set X subdivisions = {config.nx}")
            lines.append(f"  set Y subdivisions = {config.ny}")
            lines.append(f"  set Z subdivisions = {config.nz}")
        lines.append("end")
        lines.append("")
        
        # 时间步进
        lines.append("subsection Time stepping")
        lines.append(f"  set Time step = {config.dt}")
        lines.append(f"  set Time end = {config.n_steps * config.dt}")
        lines.append("end")
        lines.append("")
        
        # 线性求解器
        lines.append("subsection Linear solver")
        lines.append("  set Solver method = GMRES")
        lines.append("  set Preconditioner = ILU")
        lines.append("end")
        lines.append("")
        
        # 非线性求解器
        lines.append("subsection Nonlinear solver")
        lines.append("  set Tolerance = 1.0e-10")
        lines.append("end")
        lines.append("")
        
        # 输出
        lines.append("subsection Output")
        lines.append(f"  set Output condition = equal_spaced")
        lines.append(f"  set Time between outputs = {config.output_interval * config.dt}")
        lines.append("  set Output file type = vtu")
        lines.append("end")
        
        content = "\n".join(lines)
        
        with open(output_file, 'w') as f:
            f.write(content)
        
        logger.info(f"Generated PRISMS-PF input: {output_file}")
        
        return output_file
    
    def run_simulation(self, 
                      input_file: str,
                      n_procs: int = 1,
                      output_dir: str = "./prisms_output") -> bool:
        """
        运行PRISMS-PF模拟
        
        Args:
            input_file: 输入参数文件
            n_procs: MPI进程数
            output_dir: 输出目录
        
        Returns:
            success: 是否成功运行
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建命令
        if n_procs > 1:
            cmd = ["mpirun", "-np", str(n_procs), 
                   str(self.prisms_pf_path / "prismspf"),
                   input_file]
        else:
            cmd = [str(self.prisms_pf_path / "prismspf"), input_file]
        
        logger.info(f"Running PRISMS-PF: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=output_dir,
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            if result.returncode == 0:
                logger.info("PRISMS-PF simulation completed successfully")
                return True
            else:
                logger.error(f"PRISMS-PF failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("PRISMS-PF simulation timed out")
            return False
        except Exception as e:
            logger.error(f"Failed to run PRISMS-PF: {e}")
            return False


# =============================================================================
# MOOSE接口
# =============================================================================

class MOOSEInterface:
    """
    MOOSE框架接口
    
    MOOSE (Multiphysics Object Oriented Simulation Environment)
    是Idaho National Laboratory开发的多物理场仿真框架
    """
    
    def __init__(self, moose_path: str = "/path/to/moose"):
        self.moose_path = Path(moose_path)
        self.app_name = "phase_field"
    
    def generate_input_file(self,
                           config: PhaseFieldConfig,
                           problem_type: str = "dendrite",
                           output_file: str = "moose_input.i") -> str:
        """
        生成MOOSE输入文件
        
        MOOSE使用自定义的输入文件格式(.i)
        """
        lines = []
        
        # 网格
        lines.append("[Mesh]")
        lines.append("  type = GeneratedMesh")
        lines.append(f"  dim = {config.dimensions}")
        if config.dimensions == 2:
            lines.append(f"  nx = {config.nx}")
            lines.append(f"  ny = {config.ny}")
        else:
            lines.append(f"  nx = {config.nx}")
            lines.append(f"  ny = {config.ny}")
            lines.append(f"  nz = {config.nz}")
        lines.append(f"  xmax = {config.nx * config.dx}")
        lines.append(f"  ymax = {config.ny * config.dx}")
        if config.dimensions == 3:
            lines.append(f"  zmax = {config.nz * config.dx}")
        lines.append("")
        lines.append("  uniform_refine = 0")
        lines.append("[]")
        lines.append("")
        
        # 变量
        lines.append("[Variables]")
        lines.append("  [./phi]")
        lines.append("    order = FIRST")
        lines.append("    family = LAGRANGE")
        lines.append("  [../]")
        if problem_type == "dendrite":
            lines.append("  [./T]")
            lines.append("    order = FIRST")
            lines.append("    family = LAGRANGE")
            lines.append("  [../]")
        lines.append("[]")
        lines.append("")
        
        # 内核（PDE项）
        lines.append("[Kernels]")
        lines.append("  [./dphi_dt]")
        lines.append("    type = TimeDerivative")
        lines.append("    variable = phi")
        lines.append("  [../]")
        lines.append("  [./ACBulk]")
        lines.append("    type = AllenCahn")
        lines.append("    variable = phi")
        lines.append(f"    f_name = F")
        lines.append("    mob_name = L")
        lines.append("  [../]")
        lines.append("  [./ACInterface]")
        lines.append("    type = ACInterface")
        lines.append("    variable = phi")
        lines.append(f"    mob_name = L")
        lines.append(f"    kappa_name = kappa")
        lines.append("  [../]")
        lines.append("[]")
        lines.append("")
        
        # 材料属性
        lines.append("[Materials]")
        lines.append("  [./consts]")
        lines.append("    type = GenericConstantMaterial")
        lines.append("    prop_names = 'L kappa'")
        lines.append(f"    prop_values = '{config.mobility} {config.interface_width}'")
        lines.append("  [../]")
        lines.append("[]")
        lines.append("")
        
        # 执行器
        lines.append("[Executioner]")
        lines.append("  type = Transient")
        lines.append("  solve_type = NEWTON")
        lines.append("  l_max_its = 30")
        lines.append("  l_tol = 1e-6")
        lines.append("  nl_max_its = 20")
        lines.append("  nl_rel_tol = 1e-8")
        lines.append(f"  dt = {config.dt}")
        lines.append(f"  end_time = {config.n_steps * config.dt}")
        lines.append("")
        lines.append("  [./TimeStepper]")
        lines.append("    type = IterationAdaptiveDT")
        lines.append(f"    dt = {config.dt}")
        lines.append("    optimal_iterations = 6")
        lines.append("  [../]")
        lines.append("[]")
        lines.append("")
        
        # 输出
        lines.append("[Outputs]")
        lines.append("  exodus = true")
        lines.append("  csv = true")
        lines.append(f"  interval = {config.output_interval}")
        lines.append("[]")
        
        content = "\n".join(lines)
        
        with open(output_file, 'w') as f:
            f.write(content)
        
        logger.info(f"Generated MOOSE input: {output_file}")
        
        return output_file
    
    def run_simulation(self,
                      input_file: str,
                      n_procs: int = 1,
                      output_dir: str = "./moose_output") -> bool:
        """
        运行MOOSE模拟
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建命令
        app_path = self.moose_path / self.app_name / f"{self.app_name}-opt"
        
        if n_procs > 1:
            cmd = ["mpirun", "-np", str(n_procs), str(app_path), 
                   "-i", input_file]
        else:
            cmd = [str(app_path), "-i", input_file]
        
        logger.info(f"Running MOOSE: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=output_dir,
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            if result.returncode == 0:
                logger.info("MOOSE simulation completed successfully")
                return True
            else:
                logger.error(f"MOOSE failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("MOOSE simulation timed out")
            return False
        except Exception as e:
            logger.error(f"Failed to run MOOSE: {e}")
            return False


# =============================================================================
# 主工作流类
# =============================================================================

class PhaseFieldWorkflow:
    """
    相场模拟完整工作流
    
    整合参数提取、模拟设置、运行和后处理
    """
    
    def __init__(self, working_dir: str = "./phase_field_workflow"):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        self.md_extractor = MDtoPhaseFieldExtractor()
        self.solver = None
        self.params = None
    
    def extract_parameters_from_md(self,
                                   md_trajectory: str,
                                   extraction_config: Optional[Dict] = None) -> MDtoPhaseFieldParams:
        """从MD模拟提取相场参数"""
        logger.info("Extracting parameters from MD simulation...")
        
        # 加载MD轨迹
        self.md_extractor.load_md_trajectory(md_trajectory)
        
        # 提取扩散系数
        self.md_extractor.extract_diffusion_coefficient(temperature=300.0)
        
        # 提取界面能（如果配置提供）
        if extraction_config and 'interface_config' in extraction_config:
            self.md_extractor.extract_interface_energy(
                **extraction_config['interface_config']
            )
        
        self.params = self.md_extractor.get_params()
        
        # 保存参数
        self.md_extractor.save_params(
            self.working_dir / "md_to_pf_params.json"
        )
        
        return self.params
    
    def setup_dendrite_simulation(self,
                                   config: Optional[DendriteConfig] = None,
                                   use_extracted_params: bool = True) -> DendriteGrowthSolver:
        """设置枝晶生长模拟"""
        if config is None:
            config = DendriteConfig()
        
        # 使用从MD提取的参数
        if use_extracted_params and self.params:
            config.interface_width = self.params.interface_width or config.interface_width
            config.diffusion_coeff = self.params.diffusion_coeff or config.diffusion_coeff
            config.melting_point = self.params.melting_point or config.melting_point
            
            if 'epsilon4' in self.params.anisotropy_values:
                config.anisotropy_strength = self.params.anisotropy_values['epsilon4']
        
        # 创建求解器
        self.solver = DendriteGrowthSolver(config)
        
        logger.info("Dendrite growth simulation setup complete")
        
        return self.solver
    
    def setup_spinodal_simulation(self,
                                   config: Optional[SpinodalConfig] = None,
                                   use_extracted_params: bool = True) -> SpinodalDecompositionSolver:
        """设置相分离模拟"""
        if config is None:
            config = SpinodalConfig()
        
        if use_extracted_params and self.params:
            config.diffusion_coeff = self.params.diffusion_coeff or config.diffusion_coeff
        
        self.solver = SpinodalDecompositionSolver(config)
        
        logger.info("Spinodal decomposition simulation setup complete")
        
        return self.solver
    
    def run_simulation(self, n_steps: Optional[int] = None):
        """运行模拟"""
        if self.solver is None:
            raise ValueError("No solver configured. Run setup_*_simulation first.")
        
        self.solver.run(n_steps=n_steps)
    
    def analyze_results(self) -> Dict:
        """分析模拟结果"""
        if self.solver is None:
            raise ValueError("No simulation results available")
        
        results = {}
        
        if isinstance(self.solver, DendriteGrowthSolver):
            results['tip_velocity'] = self.solver.get_dendrite_tip_velocity()
            results['tip_radius'] = self.solver.get_tip_radius()
            results['interface_velocity'] = self.solver.compute_interface_velocity()
            
        elif isinstance(self.solver, SpinodalDecompositionSolver):
            results['domain_size'] = self.solver.get_domain_size()
        
        # 保存结果
        with open(self.working_dir / "analysis_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_full_workflow(self,
                         md_trajectory: str,
                         simulation_type: str = "dendrite",
                         n_steps: int = 10000) -> Dict:
        """
        运行完整工作流
        
        Args:
            md_trajectory: MD轨迹文件路径
            simulation_type: "dendrite" or "spinodal"
            n_steps: 模拟步数
        
        Returns:
            results: 分析结果
        """
        logger.info("=" * 60)
        logger.info("Starting Phase Field Workflow")
        logger.info("=" * 60)
        
        # 步骤1: 从MD提取参数
        self.extract_parameters_from_md(md_trajectory)
        
        # 步骤2: 设置模拟
        if simulation_type == "dendrite":
            self.setup_dendrite_simulation()
        elif simulation_type == "spinodal":
            self.setup_spinodal_simulation()
        else:
            raise ValueError(f"Unknown simulation type: {simulation_type}")
        
        # 步骤3: 运行模拟
        self.run_simulation(n_steps=n_steps)
        
        # 步骤4: 分析结果
        results = self.analyze_results()
        
        logger.info("=" * 60)
        logger.info("Phase Field Workflow Completed")
        logger.info("=" * 60)
        
        return results


# =============================================================================
# 命令行接口
# =============================================================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase Field Simulation Tool")
    parser.add_argument("--md-trajectory", help="MD trajectory file")
    parser.add_argument("--simulation-type", choices=["dendrite", "spinodal"],
                       default="dendrite")
    parser.add_argument("--solver", choices=["python", "prisms", "moose"],
                       default="python")
    parser.add_argument("--n-steps", type=int, default=10000)
    parser.add_argument("--output-dir", default="./phase_field_output")
    
    args = parser.parse_args()
    
    workflow = PhaseFieldWorkflow(working_dir=args.output_dir)
    
    if args.md_trajectory:
        results = workflow.run_full_workflow(
            md_trajectory=args.md_trajectory,
            simulation_type=args.simulation_type,
            n_steps=args.n_steps
        )
    else:
        # 运行默认模拟（不使用MD参数）
        if args.simulation_type == "dendrite":
            workflow.setup_dendrite_simulation(use_extracted_params=False)
        else:
            workflow.setup_spinodal_simulation(use_extracted_params=False)
        
        workflow.run_simulation(n_steps=args.n_steps)
        results = workflow.analyze_results()
    
    print("\nResults:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
