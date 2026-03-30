"""
Cahn-Hilliard Equation Solver
=============================
Cahn-Hilliard方程求解器

用于描述成分场的演化，适用于旋节分解、有序化等过程。

方程形式:
    ∂c/∂t = ∇·(M∇μ)
    μ = δF/δc = f'(c) - κ∇²c

其中:
    c: 成分场
    M: 迁移率
    μ: 化学势
    F: 自由能泛函
    κ: 梯度能量系数
"""

import numpy as np
from typing import Dict, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
import logging

from . import BasePhaseFieldModel, PhaseFieldConfig, BoundaryConditionType

logger = logging.getLogger(__name__)


@dataclass
class CahnHilliardConfig(PhaseFieldConfig):
    """
    Cahn-Hilliard模型配置
    
    Attributes:
        M: 迁移率 (nm²/s)
        kappa: 梯度能量系数 (eV/nm²)
        c0: 初始平均成分
        noise_amplitude: 初始扰动幅度
        free_energy_type: 自由能函数类型
        free_energy_params: 自由能函数参数
    """
    # 物理参数
    M: float = 1.0  # 迁移率
    kappa: float = 1.0  # 梯度能量系数
    
    # 初始条件
    c0: float = 0.5  # 初始平均成分
    noise_amplitude: float = 0.05  # 初始扰动幅度
    
    # 自由能函数
    free_energy_type: str = "double_well"  # double_well, regular_solution, polynomial
    free_energy_params: Dict[str, float] = field(default_factory=lambda: {
        'A': 1.0,  # 双阱势垒高度
        'B': 1.0,  # 成分范围
    })
    
    # 数值参数
    solver_type: str = "semi_implicit"  # semi_implicit, explicit, spectral
    
    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.c0 <= 1:
            raise ValueError("c0 must be in [0, 1]")


class CahnHilliardSolver(BasePhaseFieldModel):
    """
    Cahn-Hilliard方程求解器
    
    用于模拟二元合金的相分离、有序化等过程。
    """
    
    def __init__(self, config: Optional[CahnHilliardConfig] = None):
        """
        初始化Cahn-Hilliard求解器
        
        Args:
            config: Cahn-Hilliard配置
        """
        self.config = config or CahnHilliardConfig()
        super().__init__(self.config)
        
        # 初始化成分场
        self.c = None
        self.mu = None
        
        # 预计算拉普拉斯算子的傅里叶空间表示 (用于谱方法)
        self._init_spectral_operators()
        
        logger.info(f"Cahn-Hilliard solver initialized")
        logger.info(f"M={self.config.M}, κ={self.config.kappa}, c0={self.config.c0}")
    
    def _init_spectral_operators(self):
        """初始化谱方法算子"""
        if self.config.solver_type == "spectral":
            # 创建波矢
            kx = 2 * np.pi * np.fft.fftfreq(self.config.nx, self.config.dx)
            ky = 2 * np.pi * np.fft.fftfreq(self.config.ny, self.config.dy)
            
            if self.ndim == 2:
                self.k_squared = kx[:, np.newaxis]**2 + ky[np.newaxis, :]**2
            else:
                kz = 2 * np.pi * np.fft.fftfreq(self.config.nz, self.config.dz)
                kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing='ij')
                self.k_squared = kx_grid**2 + ky_grid**2 + kz_grid**2
            
            # 避免除零
            self.k_squared[0, 0] = 1e-10
    
    def initialize_fields(self, c0: Optional[np.ndarray] = None, 
                         seed: Optional[int] = None):
        """
        初始化成分场
        
        Args:
            c0: 初始成分场 (可选)
            seed: 随机种子 (可选)
        """
        if seed is not None:
            np.random.seed(seed)
        
        if c0 is not None:
            # 使用给定的初始场
            self.c = c0.copy()
        else:
            # 生成随机初始扰动
            shape = (self.config.nx, self.config.ny) if self.ndim == 2 else \
                    (self.config.nx, self.config.ny, self.config.nz)
            
            noise = self.config.noise_amplitude * (np.random.random(shape) - 0.5)
            self.c = self.config.c0 + noise
            
            # 确保成分在[0, 1]范围内
            self.c = np.clip(self.c, 0.001, 0.999)
        
        self.fields['c'] = self.c
        
        # 初始化化学势
        self._update_chemical_potential()
        
        logger.info(f"Initialized concentration field: mean={self.c.mean():.4f}, "
                   f"std={self.c.std():.4f}")
    
    def _update_chemical_potential(self):
        """更新化学势"""
        # 化学势 μ = f'(c) - κ∇²c
        df_dc = self._compute_bulk_chemical_potential(self.c)
        laplacian_c = self.compute_laplacian(self.c)
        
        self.mu = df_dc - self.config.kappa * laplacian_c
        
        # 应用边界条件
        self.mu = self.bc_handler.apply(self.mu)
    
    def _compute_bulk_chemical_potential(self, c: np.ndarray) -> np.ndarray:
        """
        计算体相化学势 f'(c)
        
        Args:
            c: 成分场
            
        Returns:
            df_dc: 化学势
        """
        f_type = self.config.free_energy_type
        params = self.config.free_energy_params
        
        if f_type == "double_well":
            # 双阱势: f(c) = A * c² * (1-c)²
            # f'(c) = 2A * c * (1-c) * (1-2c)
            A = params.get('A', 1.0)
            df_dc = 2 * A * c * (1 - c) * (1 - 2 * c)
            
        elif f_type == "regular_solution":
            # 正规溶液: f(c) = c*ln(c) + (1-c)*ln(1-c) + Ω*c*(1-c)
            # f'(c) = ln(c/(1-c)) + Ω*(1-2c)
            Omega = params.get('Omega', 1.0)
            # 添加小量避免log(0)
            eps = 1e-10
            df_dc = np.log((c + eps) / (1 - c + eps)) + Omega * (1 - 2 * c)
            
        elif f_type == "polynomial":
            # 多项式: f(c) = a1*c + a2*c² + a3*c³ + a4*c⁴
            a1 = params.get('a1', 0.0)
            a2 = params.get('a2', 0.0)
            a3 = params.get('a3', 0.0)
            a4 = params.get('a4', 1.0)
            df_dc = a1 + 2*a2*c + 3*a3*c**2 + 4*a4*c**3
            
        elif f_type == "custom":
            # 自定义函数
            custom_func = params.get('custom_func')
            if custom_func is None:
                raise ValueError("custom_func must be provided for custom free energy")
            df_dc = custom_func(c)
        else:
            raise ValueError(f"Unknown free energy type: {f_type}")
        
        return df_dc
    
    def _compute_bulk_free_energy(self, c: np.ndarray) -> np.ndarray:
        """计算体相自由能密度"""
        f_type = self.config.free_energy_type
        params = self.config.free_energy_params
        
        if f_type == "double_well":
            A = params.get('A', 1.0)
            return A * c**2 * (1 - c)**2
            
        elif f_type == "regular_solution":
            Omega = params.get('Omega', 1.0)
            eps = 1e-10
            return c * np.log(c + eps) + (1 - c) * np.log(1 - c + eps) + Omega * c * (1 - c)
            
        elif f_type == "polynomial":
            a1 = params.get('a1', 0.0)
            a2 = params.get('a2', 0.0)
            a3 = params.get('a3', 0.0)
            a4 = params.get('a4', 1.0)
            return a1*c + a2*c**2 + a3*c**3 + a4*c**4
        else:
            return np.zeros_like(c)
    
    def compute_energy(self) -> float:
        """
        计算系统总自由能
        
        Returns:
            energy: 总自由能
        """
        # 体相自由能
        f_bulk = self._compute_bulk_free_energy(self.c)
        E_bulk = np.trapezoid(np.trapezoid(f_bulk)) * self.config.dx * self.config.dy
        
        # 梯度能
        gradients = self.compute_gradient(self.c)
        grad_squared = sum(g**2 for g in gradients)
        E_grad = np.trapezoid(np.trapezoid(0.5 * self.config.kappa * grad_squared)) * self.config.dx * self.config.dy
        
        return E_bulk + E_grad
    
    def compute_chemical_potential(self, field_name: str = 'c') -> np.ndarray:
        """
        计算化学势
        
        Args:
            field_name: 场名称 (应该是'c')
            
        Returns:
            mu: 化学势场
        """
        self._update_chemical_potential()
        return self.mu.copy()
    
    def evolve_step(self) -> Dict:
        """
        执行单步演化
        
        Returns:
            info: 演化信息字典
        """
        if self.config.solver_type == "semi_implicit":
            c_new = self._evolve_semi_implicit()
        elif self.config.solver_type == "explicit":
            c_new = self._evolve_explicit()
        elif self.config.solver_type == "spectral":
            c_new = self._evolve_spectral()
        else:
            raise ValueError(f"Unknown solver type: {self.config.solver_type}")
        
        # 应用边界条件
        c_new = self.bc_handler.apply(c_new)
        
        # 确保物理约束
        c_new = np.clip(c_new, 0.0, 1.0)
        
        # 计算变化
        dc = np.abs(c_new - self.c).max()
        
        # 更新场
        self.c = c_new
        self.fields['c'] = self.c
        self._update_chemical_potential()
        
        # 收敛判断
        converged = dc < self.config.tolerance
        
        info = {
            'step': self.step,
            'time': self.time,
            'dc_max': dc,
            'c_mean': self.c.mean(),
            'c_std': self.c.std(),
            'energy': self.compute_energy(),
            'converged': converged
        }
        
        return info
    
    def _evolve_explicit(self) -> np.ndarray:
        """
        显式欧拉法演化
        
        ∂c/∂t = M∇²μ
        """
        laplacian_mu = self.compute_laplacian(self.mu)
        c_new = self.c + self.config.dt * self.config.M * laplacian_mu
        return c_new
    
    def _evolve_semi_implicit(self) -> np.ndarray:
        """
        半隐式法演化
        
        使用算子分裂技术提高稳定性
        """
        # 简化的半隐式方案
        # c^{n+1} = c^n + dt * M * ∇²μ^n
        
        # 对于双阱势，可以使用线性化处理
        if self.config.free_energy_type == "double_well":
            A = self.config.free_energy_params.get('A', 1.0)
            kappa = self.config.kappa
            M = self.config.M
            dt = self.config.dt
            
            # 线性化: f'(c) ≈ 2A*c - A (小c近似)
            # 或者使用完整的化学势
            laplacian_mu = self.compute_laplacian(self.mu)
            c_new = self.c + dt * M * laplacian_mu
        else:
            # 通用显式方案
            laplacian_mu = self.compute_laplacian(self.mu)
            c_new = self.c + self.config.dt * self.config.M * laplacian_mu
        
        return c_new
    
    def _evolve_spectral(self) -> np.ndarray:
        """
        谱方法演化
        
        在傅里叶空间求解，对于周期性边界条件非常高效
        """
        # 转换到傅里叶空间
        c_hat = np.fft.fftn(self.c)
        mu_hat = np.fft.fftn(self.mu)
        
        # 演化方程: ∂ĉ/∂t = -M*k²*μ̂
        k2 = self.k_squared
        c_hat_new = c_hat - self.config.dt * self.config.M * k2 * mu_hat
        
        # 转换回实空间
        c_new = np.fft.ifftn(c_hat_new).real
        
        return c_new
    
    def get_phase_fraction(self, threshold: float = 0.5) -> Tuple[float, float]:
        """
        计算两相体积分数
        
        Args:
            threshold: 相分离阈值
            
        Returns:
            f1, f2: 两相体积分数
        """
        total = self.c.size
        f1 = np.sum(self.c > threshold) / total
        f2 = 1 - f1
        return f1, f2
    
    def get_domain_size(self) -> float:
        """
        计算特征畴尺寸
        
        Returns:
            domain_size: 特征畴尺寸 (nm)
        """
        # 使用结构因子计算
        c_fft = np.fft.fftn(self.c - self.c.mean())
        S = np.abs(c_fft)**2
        
        # 找到最大结构因子对应的波矢
        kx = 2 * np.pi * np.fft.fftfreq(self.config.nx, self.config.dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.config.ny, self.config.dy)
        
        if self.ndim == 2:
            k_mag = np.sqrt(kx[:, np.newaxis]**2 + ky[np.newaxis, :]**2)
        else:
            kz = 2 * np.pi * np.fft.fftfreq(self.config.nz, self.config.dz)
            kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing='ij')
            k_mag = np.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)
        
        # 排除零频
        mask = k_mag > 0
        if np.any(mask):
            k_peak = k_mag[mask][np.argmax(S[mask])]
            domain_size = 2 * np.pi / k_peak if k_peak > 0 else self.config.nx * self.config.dx
        else:
            domain_size = self.config.nx * self.config.dx
        
        return domain_size
    
    def export_results(self, filename: str):
        """
        导出结果
        
        Args:
            filename: 输出文件名
        """
        import json
        
        results = {
            'config': {
                'nx': self.config.nx,
                'ny': self.config.ny,
                'nz': self.config.nz,
                'dx': self.config.dx,
                'M': self.config.M,
                'kappa': self.config.kappa,
                'c0': self.config.c0,
            },
            'final_state': {
                'c_mean': float(self.c.mean()),
                'c_std': float(self.c.std()),
                'c_min': float(self.c.min()),
                'c_max': float(self.c.max()),
            },
            'evolution': {
                'time': [float(t) for t in self.history['time']],
                'energy': [float(e) for e in self.history['energy']],
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        # 导出场数据
        np.savez(f"{filename[:-5]}_fields.npz", 
                 c=self.c, mu=self.mu)
        
        logger.info(f"Results exported to {filename}")
