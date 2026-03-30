"""
Grain Boundary Simulator
========================
固态电解质晶界迁移模拟器

模拟多晶材料中的晶界迁移过程。
适用于固态电解质、金属陶瓷等。

物理模型:
- 多序参量Allen-Cahn模型
- 曲率驱动迁移
- 溶质拖曳效应
- 电场/应力场耦合
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
import logging

from ..core.allen_cahn import AllenCahnSolver, AllenCahnConfig
from ..core.cahn_hilliard import CahnHilliardSolver

logger = logging.getLogger(__name__)


@dataclass
class GBConfig(AllenCahnConfig):
    """
    晶界迁移配置
    
    Attributes:
        # 微观结构参数
        n_grains: 晶粒数量
        initial_grain_size: 初始平均晶粒尺寸 (μm)
        grain_orientation: 晶粒取向分布
        
        # 迁移参数
        gb_mobility: 晶界迁移率 (m²/J·s)
        gb_energy: 晶界能 (J/m²)
        
        # 溶质效应
        solute_concentration: 溶质浓度
        solute_drag_coefficient: 溶质拖曳系数
        
        # 外场
        applied_stress: 外加应力 (MPa)
        electric_field: 电场强度 (V/m)
        
        # 各向异性
        anisotropy_mode: 各向异性模式 ('isotropic', 'weak', 'strong')
        anisotropy_coefficient: 各向异性系数
    """
    # 晶粒参数
    n_grains: int = 10
    initial_grain_size: float = 1.0  # μm
    
    # 晶界性质
    gb_mobility: float = 1e-12  # m²/J·s
    gb_energy: float = 0.5  # J/m²
    gb_thickness: float = 1.0  # nm
    
    # 溶质效应
    solute_concentration: float = 0.0
    solute_drag_coefficient: float = 0.0
    
    # 外场
    applied_stress: float = 0.0  # MPa
    electric_field: float = 0.0  # V/m
    
    # 各向异性
    anisotropy_mode: str = "isotropic"
    anisotropy_coefficient: float = 0.0
    
    # 特殊晶界
    include_special_boundaries: bool = False
    special_boundary_fraction: float = 0.0
    
    def __post_init__(self):
        super().__post_init__()
        
        # 转换迁移率到相场单位
        # M_pf = M * γ * δ (无量纲化)
        self.L_eff = self.gb_mobility * self.gb_energy * 1e9  # 简化转换


class GrainBoundarySimulator(AllenCahnSolver):
    """
    晶界迁移模拟器
    
    模拟多晶材料中的晶界迁移和晶粒生长。
    """
    
    def __init__(self, config: Optional[GBConfig] = None):
        """
        初始化晶界模拟器
        
        Args:
            config: 晶界配置
        """
        self.config = config or GBConfig()
        super().__init__(self.config)
        
        # 晶粒追踪
        self.grain_ids = None
        self.grain_sizes = {}
        self.grain_orientations = {}
        
        # 溶质场 (如果启用)
        self.solute_field = None
        if self.config.solute_concentration > 0:
            self.solute_field = np.ones((self.config.nx, self.config.ny)) * \
                               self.config.solute_concentration
        
        # 外场
        self.stress_field = None
        if self.config.applied_stress != 0:
            self._init_stress_field()
        
        logger.info(f"Grain boundary simulator initialized")
        logger.info(f"n_grains={self.config.n_grains}, "
                   f"M_gb={self.config.gb_mobility:.2e}")
    
    def initialize_fields(self, voronoi_seeds: Optional[List[Tuple]] = None,
                         grain_orientations: Optional[List[float]] = None,
                         seed: Optional[int] = None):
        """
        初始化晶粒结构
        
        Args:
            voronoi_seeds: Voronoi晶核位置
            grain_orientations: 晶粒取向列表
            seed: 随机种子
        """
        if seed is not None:
            np.random.seed(seed)
        
        shape = (self.config.nx, self.config.ny)
        
        # 生成Voronoi结构
        if voronoi_seeds is None:
            # 随机生成晶核
            voronoi_seeds = [
                (np.random.randint(0, self.config.nx),
                 np.random.randint(0, self.config.ny))
                for _ in range(self.config.n_grains)
            ]
        
        # 为每个晶粒创建序参量
        for i in range(self.config.n_grains):
            name = f"grain_{i}"
            self.eta[name] = np.zeros(shape)
        
        # 分配网格点到最近晶核
        for i in range(self.config.nx):
            for j in range(self.config.ny):
                # 计算到各晶核的距离
                distances = [
                    (i - sx)**2 + (j - sy)**2
                    for sx, sy in voronoi_seeds
                ]
                
                # 分配到最近晶核
                nearest_grain = np.argmin(distances)
                self.eta[f"grain_{nearest_grain}"][i, j] = 1.0
        
        # 平滑过渡区
        self._smooth_grain_boundaries()
        
        # 初始化取向
        if grain_orientations is None:
            grain_orientations = np.random.uniform(0, 360, self.config.n_grains)
        
        self.grain_orientations = {
            f"grain_{i}": grain_orientations[i]
            for i in range(self.config.n_grains)
        }
        
        # 更新fields
        self.fields.update(self.eta)
        
        # 初始化晶粒ID图
        self._update_grain_ids()
        
        logger.info(f"Initialized {self.config.n_grains} grains")
    
    def _smooth_grain_boundaries(self, smoothing_width: int = 3):
        """平滑晶界区域"""
        from scipy.ndimage import gaussian_filter
        
        for name in self.eta:
            self.eta[name] = gaussian_filter(self.eta[name], sigma=smoothing_width)
    
    def _init_stress_field(self):
        """初始化应力场"""
        shape = (self.config.nx, self.config.ny)
        
        # 均匀应力场 (简化)
        self.stress_field = {
            'sigma_xx': np.full(shape, self.config.applied_stress),
            'sigma_yy': np.zeros(shape),
            'sigma_xy': np.zeros(shape)
        }
    
    def _update_grain_ids(self):
        """更新晶粒ID图"""
        shape = (self.config.nx, self.config.ny)
        self.grain_ids = np.full(shape, -1, dtype=int)
        
        for i, name in enumerate(self.eta.keys()):
            mask = self.eta[name] > 0.5
            self.grain_ids[mask] = i
    
    def _compute_local_curvature(self, eta_field: np.ndarray) -> np.ndarray:
        """
        计算局部曲率
        
        κ = ∇·(∇η/|∇η|)
        
        Args:
            eta_field: 序参量场
            
        Returns:
            curvature: 曲率场
        """
        # 计算梯度
        grad_x, grad_y = self.compute_gradient(eta_field)
        
        # 归一化
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mag = np.maximum(grad_mag, 1e-10)
        
        nx = grad_x / grad_mag
        ny = grad_y / grad_mag
        
        # 散度
        dnx_dx = np.gradient(nx, self.config.dx, axis=0)
        dny_dy = np.gradient(ny, self.config.dy, axis=1)
        
        curvature = dnx_dx + dny_dy
        
        return curvature
    
    def _compute_solute_drag(self, grain_boundary_mask: np.ndarray) -> np.ndarray:
        """
        计算溶质拖曳效应
        
        Args:
            grain_boundary_mask: 晶界掩码
            
        Returns:
            drag_force: 拖曳力场
        """
        if self.solute_field is None:
            return np.zeros((self.config.nx, self.config.ny))
        
        # 溶质在晶界的富集
        solute_gb = self.solute_field * grain_boundary_mask
        
        # 拖曳力 ∝ 溶质浓度梯度
        drag_force = self.config.solute_drag_coefficient * solute_gb
        
        return drag_force
    
    def _apply_anisotropy(self, gb_energy: float, orientation: float) -> float:
        """
        应用晶界能各向异性
        
        Args:
            gb_energy: 基准晶界能
            orientation: 晶界取向 (度)
            
        Returns:
            gamma_aniso: 各向异性晶界能
        """
        if self.config.anisotropy_mode == "isotropic":
            return gb_energy
        
        # 简化的各向异性模型
        theta_rad = np.deg2rad(orientation)
        
        if self.config.anisotropy_mode == "weak":
            # 弱各向异性: γ = γ0 * (1 + ε*cos(4θ))
            epsilon = self.config.anisotropy_coefficient
            gamma_aniso = gb_energy * (1 + epsilon * np.cos(4 * theta_rad))
        elif self.config.anisotropy_mode == "strong":
            # 强各向异性 (cusp模型)
            epsilon = self.config.anisotropy_coefficient
            gamma_aniso = gb_energy * np.maximum(0, 1 + epsilon * np.cos(4 * theta_rad))
        else:
            gamma_aniso = gb_energy
        
        return gamma_aniso
    
    def evolve_step(self) -> Dict:
        """
        执行晶界迁移步骤
        
        Returns:
            info: 演化信息
        """
        max_deta = 0.0
        
        # 计算各晶粒的曲率驱动力
        curvatures = {}
        for name in self.eta:
            curvatures[name] = self._compute_local_curvature(self.eta[name])
        
        # 晶界掩码
        total_eta = sum(self.eta.values())
        gb_mask = (total_eta > 0.1) & (total_eta < 0.9)
        
        # 溶质拖曳
        drag = self._compute_solute_drag(gb_mask)
        
        # 演化各序参量
        for i, name in enumerate(self.eta.keys()):
            eta_old = self.eta[name].copy()
            
            # 曲率驱动力
            curvature = curvatures[name]
            
            # 各向异性晶界能
            orientation = self.grain_orientations.get(name, 0)
            gamma = self._apply_anisotropy(self.config.gb_energy, orientation)
            
            # 总驱动力
            driving_force = gamma * curvature - drag
            
            # 外场贡献 (简化)
            if self.stress_field is not None:
                stress_effect = self.stress_field['sigma_xx'] * 1e-6  # 简化转换
                driving_force += stress_effect
            
            # 序参量演化
            # df/dη ≈ 2η(1-η)(1-2η)
            df_deta = 2 * eta_old * (1 - eta_old) * (1 - 2 * eta_old)
            laplacian_eta = self.compute_laplacian(eta_old)
            
            # 变分导数包含驱动力
            delta_F = df_deta - self.config.kappa * laplacian_eta - driving_force * 0.1
            
            # 演化
            L_eff = self.config.L_eff if hasattr(self.config, 'L_eff') else self.config.L
            eta_new = eta_old - self.config.dt * L_eff * delta_F
            
            # 限制范围
            eta_new = np.clip(eta_new, 0, 1)
            
            self.eta[name] = eta_new
            self.fields[name] = eta_new
            
            deta = np.abs(eta_new - eta_old).max()
            max_deta = max(max_deta, deta)
        
        # 归一化 (确保每个点至少有一个主导相)
        self._normalize_order_params()
        
        # 更新晶粒ID
        self._update_grain_ids()
        
        converged = max_deta < self.config.tolerance
        
        info = {
            'step': self.step,
            'time': self.time,
            'deta_max': max_deta,
            'n_grains': self.get_grain_count(),
            'avg_grain_size': self.get_average_grain_size(),
            'gb_length': self.get_grain_boundary_length(),
            'converged': converged
        }
        
        return info
    
    def get_grain_count(self) -> int:
        """获取当前晶粒数"""
        unique_ids = np.unique(self.grain_ids)
        return len(unique_ids[unique_ids >= 0])
    
    def get_grain_size_distribution(self) -> Dict[int, float]:
        """
        获取晶粒尺寸分布
        
        Returns:
            size_dist: 晶粒ID到尺寸的字典
        """
        unique_ids = np.unique(self.grain_ids)
        size_dist = {}
        
        for gid in unique_ids:
            if gid >= 0:
                size = np.sum(self.grain_ids == gid)
                size_dist[gid] = size * self.config.dx**2  # nm²
        
        return size_dist
    
    def get_gb_velocity(self) -> float:
        """
        估算平均晶界迁移速率
        
        Returns:
            velocity: 迁移速率 (nm/s)
        """
        if len(self.history['time']) < 2:
            return 0.0
        
        # 基于平均晶粒尺寸变化估算
        current_size = self.get_average_grain_size()
        
        # 简化的速率估算
        dt = self.config.dt * self.config.save_interval
        velocity = current_size / (self.time + 1e-10)  # 简化估算
        
        return velocity
    
    def get_ion_conductivity_factor(self) -> float:
        """
        估算晶界对离子电导率的影响
        
        Returns:
            conductivity_factor: 电导率修正因子 (0-1)
        """
        # 晶界体积分数
        gb_volume = np.sum(self.get_grain_structure() == -1) / self.grain_ids.size
        
        # 简化的阻滞因子
        # σ_eff = σ_bulk * (1 - f_gb)^n
        n = 2  # 指数因子
        factor = (1 - gb_volume)**n
        
        return factor
