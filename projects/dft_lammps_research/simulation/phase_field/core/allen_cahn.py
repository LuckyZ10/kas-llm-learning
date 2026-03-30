"""
Allen-Cahn Equation Solver
==========================
Allen-Cahn方程求解器

用于描述序参量的非保守演化，适用于晶粒生长、马氏体相变等。

方程形式:
    ∂η/∂t = -L(δF/δη) = -L(f'(η) - κ∇²η)

其中:
    η: 序参量 (0 ≤ η ≤ 1)
    L: 动力学系数 (弛豫速率)
    f(η): 双阱势函数
    κ: 梯度能量系数
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Callable
from dataclasses import dataclass, field
import logging

from . import BasePhaseFieldModel, PhaseFieldConfig

logger = logging.getLogger(__name__)


@dataclass
class AllenCahnConfig(PhaseFieldConfig):
    """
    Allen-Cahn模型配置
    
    Attributes:
        L: 动力学系数 (1/s)
        kappa: 梯度能量系数 (eV/nm²)
        barrier_height: 势垒高度 (eV)
        n_order_params: 序参量数量 (多序参量模型)
        initial_structure: 初始结构类型
    """
    # 物理参数
    L: float = 1.0  # 动力学系数
    kappa: float = 1.0  # 梯度能量系数
    barrier_height: float = 1.0  # 势垒高度
    
    # 多序参量
    n_order_params: int = 1  # 序参量数量
    
    # 初始条件
    initial_structure: str = "random"  # random, nucleation, grains
    initial_nuclei: List[Tuple[float, float, float]] = field(default_factory=list)  # 核位置列表
    
    # 数值参数
    solver_type: str = "semi_implicit"  # semi_implicit, explicit
    
    def __post_init__(self):
        super().__post_init__()
        if self.n_order_params < 1:
            raise ValueError("n_order_params must be at least 1")


class AllenCahnSolver(BasePhaseFieldModel):
    """
    Allen-Cahn方程求解器
    
    支持单序参量和多序参量模型。
    多序参量模型用于描述多晶系统，每个序参量代表一个晶粒取向。
    """
    
    def __init__(self, config: Optional[AllenCahnConfig] = None):
        """
        初始化Allen-Cahn求解器
        
        Args:
            config: Allen-Cahn配置
        """
        self.config = config or AllenCahnConfig()
        super().__init__(self.config)
        
        # 初始化序参量场
        self.eta = {}  # 序参量字典
        
        logger.info(f"Allen-Cahn solver initialized")
        logger.info(f"L={self.config.L}, κ={self.config.kappa}, "
                   f"n_params={self.config.n_order_params}")
    
    def initialize_fields(self, eta_init: Optional[Dict[str, np.ndarray]] = None,
                         seed: Optional[int] = None):
        """
        初始化序参量场
        
        Args:
            eta_init: 初始序参量场字典 (可选)
            seed: 随机种子 (可选)
        """
        if seed is not None:
            np.random.seed(seed)
        
        shape = (self.config.nx, self.config.ny) if self.ndim == 2 else \
                (self.config.nx, self.config.ny, self.config.nz)
        
        if eta_init is not None:
            # 使用给定的初始场
            for name, field in eta_init.items():
                self.eta[name] = field.copy()
                self.fields[name] = self.eta[name]
        else:
            # 根据初始结构类型生成
            if self.config.initial_structure == "random":
                self._init_random(shape)
            elif self.config.initial_structure == "nucleation":
                self._init_nucleation(shape)
            elif self.config.initial_structure == "grains":
                self._init_grains(shape)
            elif self.config.initial_structure == "interface":
                self._init_interface(shape)
            else:
                raise ValueError(f"Unknown initial structure: {self.config.initial_structure}")
        
        logger.info(f"Initialized {len(self.eta)} order parameter fields")
    
    def _init_random(self, shape: Tuple):
        """随机初始条件"""
        for i in range(self.config.n_order_params):
            name = f"eta_{i}"
            # 随机值并归一化
            eta_i = np.random.random(shape)
            self.eta[name] = eta_i
            self.fields[name] = eta_i
    
    def _init_nucleation(self, shape: Tuple):
        """形核初始条件"""
        # 初始化基体为0
        for i in range(self.config.n_order_params):
            name = f"eta_{i}"
            self.eta[name] = np.zeros(shape)
        
        # 在指定位置放置核
        nuclei = self.config.initial_nuclei
        if not nuclei:
            # 自动生成随机核位置
            n_nuclei = max(3, self.config.nx // 20)
            for _ in range(n_nuclei):
                x = np.random.randint(0, self.config.nx)
                y = np.random.randint(0, self.config.ny)
                if self.ndim == 3:
                    z = np.random.randint(0, self.config.nz)
                    nuclei.append((x, y, z))
                else:
                    nuclei.append((x, y, 0))
        
        # 为每个核分配一个序参量
        for idx, (x, y, z) in enumerate(nuclei):
            param_idx = idx % self.config.n_order_params
            name = f"eta_{param_idx}"
            
            # 创建高斯形核
            xi, yi = np.indices(shape[:2])
            dist_sq = (xi - x)**2 + (yi - y)**2
            
            if self.ndim == 2:
                nucleus = np.exp(-dist_sq / (2 * (shape[0]//10)**2))
                self.eta[name] = np.maximum(self.eta[name], nucleus)
            else:
                for iz in range(shape[2]):
                    dist_sq_3d = dist_sq + (iz - z)**2
                    nucleus = np.exp(-dist_sq_3d / (2 * (shape[0]//10)**2))
                    self.eta[name][:, :, iz] = np.maximum(
                        self.eta[name][:, :, iz], nucleus[:, :]
                    )
        
        # 归一化
        self._normalize_order_params()
        
        for name in self.eta:
            self.fields[name] = self.eta[name]
    
    def _init_grains(self, shape: Tuple):
        """多晶初始条件 (Voronoi图)"""
        from scipy.spatial import Voronoi
        
        # 生成随机晶核
        n_grains = self.config.n_order_params
        points = np.random.rand(n_grains, 2) * [self.config.nx, self.config.ny]
        
        # 创建网格
        xi, yi = np.indices(shape[:2])
        
        # 为每个晶粒分配序参量
        for i in range(n_grains):
            name = f"eta_{i}"
            # 计算到每个网格点的距离
            dist = np.sqrt((xi - points[i, 0])**2 + (yi - points[i, 1])**2)
            
            # 找到最近的晶核
            for j in range(n_grains):
                if j != i:
                    dist_j = np.sqrt((xi - points[j, 0])**2 + (yi - points[j, 1])**2)
                    dist = np.minimum(dist, dist_j)
            
            # 平滑过渡
            self.eta[name] = np.exp(-dist / (shape[0]//20))
            if self.ndim == 3:
                self.eta[name] = np.repeat(self.eta[name][:, :, np.newaxis], shape[2], axis=2)
        
        self._normalize_order_params()
        
        for name in self.eta:
            self.fields[name] = self.eta[name]
    
    def _init_interface(self, shape: Tuple):
        """平面界面初始条件"""
        # 创建一个简单的平面界面
        x = np.linspace(0, 1, shape[0])
        
        if self.ndim == 2:
            eta_profile = 0.5 * (1 + np.tanh((x - 0.5) * shape[0] / 10))
            self.eta["eta_0"] = np.tile(eta_profile[:, np.newaxis], (1, shape[1]))
        else:
            eta_profile = 0.5 * (1 + np.tanh((x - 0.5) * shape[0] / 10))
            self.eta["eta_0"] = np.tile(
                eta_profile[:, np.newaxis, np.newaxis], 
                (1, shape[1], shape[2])
            )
        
        self.fields["eta_0"] = self.eta["eta_0"]
    
    def _normalize_order_params(self):
        """归一化序参量 (确保每个点只有一个主要相)"""
        if len(self.eta) <= 1:
            return
        
        # 创建数组堆叠
        eta_array = np.stack(list(self.eta.values()), axis=0)
        
        # 找到每个点最大的序参量
        max_idx = np.argmax(eta_array, axis=0)
        
        # 归一化：最大值为1，其他为0 (硬约束)
        # 或者使用softmax (软约束)
        for i, name in enumerate(self.eta.keys()):
            mask = (max_idx == i)
            self.eta[name] = mask.astype(float) * 0.9 + 0.05
    
    def _double_well_potential(self, eta: np.ndarray) -> np.ndarray:
        """
        双阱势函数
        
        f(η) = A * η² * (1-η)²
        
        Returns:
            f: 势函数值
        """
        A = self.config.barrier_height
        return A * eta**2 * (1 - eta)**2
    
    def _double_well_derivative(self, eta: np.ndarray) -> np.ndarray:
        """
        双阱势导数
        
        f'(η) = 2A * η * (1-η) * (1-2η)
        
        Returns:
            df_deta: 势函数导数
        """
        A = self.config.barrier_height
        return 2 * A * eta * (1 - eta) * (1 - 2 * eta)
    
    def compute_energy(self) -> float:
        """
        计算系统总自由能
        
        Returns:
            energy: 总自由能
        """
        total_energy = 0.0
        
        for name, eta in self.eta.items():
            # 体相自由能
            f_bulk = self._double_well_potential(eta)
            E_bulk = np.sum(f_bulk) * self.config.dx * self.config.dy
            
            # 梯度能
            gradients = self.compute_gradient(eta)
            grad_squared = sum(g**2 for g in gradients)
            E_grad = np.sum(0.5 * self.config.kappa * grad_squared) * self.config.dx * self.config.dy
            
            total_energy += E_bulk + E_grad
        
        return total_energy
    
    def compute_chemical_potential(self, field_name: str) -> np.ndarray:
        """
        计算序参量的变分导数 δF/δη
        
        Args:
            field_name: 序参量名称
            
        Returns:
            df_deta: 变分导数
        """
        if field_name not in self.eta:
            raise KeyError(f"Order parameter '{field_name}' not found")
        
        eta = self.eta[field_name]
        
        # δF/δη = f'(η) - κ∇²η
        df_bulk = self._double_well_derivative(eta)
        laplacian_eta = self.compute_laplacian(eta)
        
        df_deta = df_bulk - self.config.kappa * laplacian_eta
        
        return df_deta
    
    def evolve_step(self) -> Dict:
        """
        执行单步演化
        
        Returns:
            info: 演化信息字典
        """
        max_deta = 0.0
        
        for name in self.eta:
            eta_old = self.eta[name].copy()
            
            # 计算变分导数
            df_deta = self.compute_chemical_potential(name)
            
            if self.config.solver_type == "explicit":
                # 显式: η^{n+1} = η^n - dt * L * δF/δη
                eta_new = eta_old - self.config.dt * self.config.L * df_deta
            elif self.config.solver_type == "semi_implicit":
                # 半隐式 (简化的线性化处理)
                # 对于双阱势，使用线性稳定性分析
                A = self.config.barrier_height
                # 线性化: f'(η) ≈ 2A*η - A
                eta_new = eta_old - self.config.dt * self.config.L * df_deta
            else:
                raise ValueError(f"Unknown solver type: {self.config.solver_type}")
            
            # 应用边界条件
            eta_new = self.bc_handler.apply(eta_new)
            
            # 限制序参量范围 [0, 1]
            eta_new = np.clip(eta_new, 0.0, 1.0)
            
            # 更新
            self.eta[name] = eta_new
            self.fields[name] = eta_new
            
            # 跟踪最大变化
            deta = np.abs(eta_new - eta_old).max()
            max_deta = max(max_deta, deta)
        
        # 多序参量归一化
        if self.config.n_order_params > 1:
            self._normalize_order_params()
        
        # 收敛判断
        converged = max_deta < self.config.tolerance
        
        info = {
            'step': self.step,
            'time': self.time,
            'deta_max': max_deta,
            'energy': self.compute_energy(),
            'converged': converged
        }
        
        return info
    
    def get_grain_structure(self, threshold: float = 0.5) -> np.ndarray:
        """
        获取晶粒结构
        
        Args:
            threshold: 序参量阈值
            
        Returns:
            grain_ids: 晶粒标记数组
        """
        if len(self.eta) == 1:
            # 单序参量：返回二值结构
            return (self.eta["eta_0"] > threshold).astype(int)
        
        # 多序参量：找到每个点最大的序参量
        eta_array = np.stack(list(self.eta.values()), axis=0)
        grain_ids = np.argmax(eta_array, axis=0)
        
        # 标记低于阈值的区域为-1 (晶界)
        max_eta = np.max(eta_array, axis=0)
        grain_ids[max_eta < threshold] = -1
        
        return grain_ids
    
    def get_grain_boundary_length(self, threshold: float = 0.5) -> float:
        """
        计算晶界总长度
        
        Args:
            threshold: 序参量阈值
            
        Returns:
            gb_length: 晶界长度 (nm)
        """
        grain_structure = self.get_grain_structure(threshold)
        
        # 找到晶界点
        if self.ndim == 2:
            # 使用边缘检测
            from scipy import ndimage
            edges = ndimage.binary_dilation(grain_structure == -1) ^ (grain_structure == -1)
            gb_length = np.sum(edges) * self.config.dx
        else:
            # 3D情况
            gb_volume = np.sum(grain_structure == -1) * self.config.dx**3
            # 估算表面积
            gb_length = gb_volume / self.config.dx
        
        return gb_length
    
    def get_average_grain_size(self) -> float:
        """
        计算平均晶粒尺寸
        
        Returns:
            grain_size: 平均晶粒尺寸 (nm)
        """
        grain_structure = self.get_grain_structure()
        
        if len(self.eta) == 1:
            # 单序参量：计算相畴尺寸
            phase_volume = np.sum(grain_structure) * self.config.dx**self.ndim
            domain_size = phase_volume**(1/self.ndim)
        else:
            # 多序参量：计算平均晶粒尺寸
            unique, counts = np.unique(grain_structure[grain_structure >= 0], return_counts=True)
            if len(unique) > 0:
                avg_volume = np.mean(counts) * self.config.dx**self.ndim
                domain_size = avg_volume**(1/self.ndim)
            else:
                domain_size = 0.0
        
        return domain_size
