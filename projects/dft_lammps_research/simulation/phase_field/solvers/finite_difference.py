"""
Finite Difference Solver
========================
有限差分求解器

提供各种有限差分格式用于相场方程的数值求解。
支持显式、隐式、Crank-Nicolson等格式。
"""

import numpy as np
from typing import Dict, Optional, Tuple, Callable
from dataclasses import dataclass
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve, cg, bicgstab
import logging

logger = logging.getLogger(__name__)


@dataclass
class FDSConfig:
    """有限差分求解器配置"""
    # 差分格式
    spatial_order: int = 2  # 空间精度阶数 (2, 4, 6)
    time_scheme: str = "explicit"  # explicit, implicit, crank_nicolson
    
    # 迭代参数
    max_iter: int = 1000
    tolerance: float = 1e-6
    
    # 线性求解器
    linear_solver: str = "direct"  # direct, cg, bicgstab
    preconditioner: Optional[str] = "ilu"
    
    # 稳定性参数
    cfl_number: float = 0.25  # CFL数限制


class FiniteDifferenceSolver:
    """
    有限差分求解器
    
    提供高阶有限差分格式和高效的时间积分方案。
    """
    
    def __init__(self, config: Optional[FDSConfig] = None):
        """
        初始化有限差分求解器
        
        Args:
            config: 有限差分配置
        """
        self.config = config or FDSConfig()
        
        # 差分系数
        self._init_difference_coefficients()
        
        logger.info(f"Finite difference solver initialized")
        logger.info(f"Spatial order: {self.config.spatial_order}, "
                   f"Time scheme: {self.config.time_scheme}")
    
    def _init_difference_coefficients(self):
        """初始化差分系数"""
        order = self.config.spatial_order
        
        # 一阶导数系数 (中心差分)
        if order == 2:
            self.coeffs_1st = np.array([-1/2, 0, 1/2])
            self.coeffs_2nd = np.array([1, -2, 1])
        elif order == 4:
            self.coeffs_1st = np.array([1/12, -2/3, 0, 2/3, -1/12])
            self.coeffs_2nd = np.array([-1/12, 4/3, -5/2, 4/3, -1/12])
        elif order == 6:
            self.coeffs_1st = np.array([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
            self.coeffs_2nd = np.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])
        else:
            raise ValueError(f"Unsupported spatial order: {order}")
        
        self.stencil_width = (len(self.coeffs_1st) - 1) // 2
    
    def laplacian_2d(self, field: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """
        计算2D拉普拉斯算子
        
        Args:
            field: 输入场
            dx, dy: 网格间距
            
        Returns:
            laplacian: 拉普拉斯场
        """
        nx, ny = field.shape
        stencil = self.stencil_width
        
        laplacian = np.zeros_like(field)
        
        if self.config.spatial_order == 2:
            # 标准5点格式
            laplacian[1:-1, 1:-1] = (
                (field[2:, 1:-1] - 2*field[1:-1, 1:-1] + field[:-2, 1:-1]) / dx**2 +
                (field[1:-1, 2:] - 2*field[1:-1, 1:-1] + field[1:-1, :-2]) / dy**2
            )
        elif self.config.spatial_order == 4:
            # 9点高阶格式
            for i in range(stencil, nx - stencil):
                for j in range(stencil, ny - stencil):
                    d2x = sum(self.coeffs_2nd[k+stencil] * field[i+k, j] 
                             for k in range(-stencil, stencil+1)) / dx**2
                    d2y = sum(self.coeffs_2nd[k+stencil] * field[i, j+k] 
                             for k in range(-stencil, stencil+1)) / dy**2
                    laplacian[i, j] = d2x + d2y
        
        return laplacian
    
    def laplacian_3d(self, field: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
        """计算3D拉普拉斯算子"""
        nx, ny, nz = field.shape
        laplacian = np.zeros_like(field)
        
        if self.config.spatial_order == 2:
            laplacian[1:-1, 1:-1, 1:-1] = (
                (field[2:, 1:-1, 1:-1] - 2*field[1:-1, 1:-1, 1:-1] + field[:-2, 1:-1, 1:-1]) / dx**2 +
                (field[1:-1, 2:, 1:-1] - 2*field[1:-1, 1:-1, 1:-1] + field[1:-1, :-2, 1:-1]) / dy**2 +
                (field[1:-1, 1:-1, 2:] - 2*field[1:-1, 1:-1, 1:-1] + field[1:-1, 1:-1, :-2]) / dz**2
            )
        
        return laplacian
    
    def gradient_2d(self, field: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算2D梯度
        
        Returns:
            (grad_x, grad_y): 梯度分量
        """
        nx, ny = field.shape
        stencil = self.stencil_width
        
        grad_x = np.zeros_like(field)
        grad_y = np.zeros_like(field)
        
        if self.config.spatial_order == 2:
            grad_x[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2 * dx)
            grad_y[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2 * dy)
        else:
            for i in range(stencil, nx - stencil):
                for j in range(ny):
                    grad_x[i, j] = sum(self.coeffs_1st[k+stencil] * field[i+k, j] 
                                      for k in range(-stencil, stencil+1)) / dx
            for i in range(nx):
                for j in range(stencil, ny - stencil):
                    grad_y[i, j] = sum(self.coeffs_1st[k+stencil] * field[i, j+k] 
                                      for k in range(-stencil, stencil+1)) / dy
        
        return grad_x, grad_y
    
    def divergence_2d(self, fx: np.ndarray, fy: np.ndarray, 
                      dx: float, dy: float) -> np.ndarray:
        """
        计算2D散度
        
        ∇·F = ∂fx/∂x + ∂fy/∂y
        """
        dfx_dx = (fx[2:, :] - fx[:-2, :]) / (2 * dx)
        dfy_dy = (fy[:, 2:] - fy[:, :-2]) / (2 * dy)
        
        # 对齐数组尺寸
        div = np.zeros_like(fx)
        div[1:-1, 1:-1] = dfx_dx[:, 1:-1] + dfy_dy[1:-1, :]
        
        return div
    
    def time_step_explicit(self, field: np.ndarray, 
                          rhs_func: Callable, dt: float) -> np.ndarray:
        """
        显式时间步进 (Euler)
        
        u^{n+1} = u^n + dt * rhs(u^n)
        """
        rhs = rhs_func(field)
        field_new = field + dt * rhs
        return field_new
    
    def time_step_rk4(self, field: np.ndarray,
                     rhs_func: Callable, dt: float) -> np.ndarray:
        """
        4阶Runge-Kutta时间步进
        
        提供更高的时间精度和稳定性
        """
        k1 = rhs_func(field)
        k2 = rhs_func(field + 0.5 * dt * k1)
        k3 = rhs_func(field + 0.5 * dt * k2)
        k4 = rhs_func(field + dt * k3)
        
        field_new = field + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        return field_new
    
    def solve_poisson_2d(self, rhs: np.ndarray, dx: float, dy: float,
                        boundary_values: Optional[Dict] = None) -> np.ndarray:
        """
        求解2D泊松方程 ∇²φ = rhs
        
        使用直接稀疏矩阵求解或迭代法
        """
        nx, ny = rhs.shape
        n_total = nx * ny
        
        if self.config.linear_solver == "direct":
            # 构建稀疏矩阵
            main_diag = -2 * (1/dx**2 + 1/dy**2) * np.ones(n_total)
            off_x = (1/dx**2) * np.ones(n_total - 1)
            off_y = (1/dy**2) * np.ones(n_total - ny)
            
            # 移除跨边界连接
            for i in range(1, nx):
                off_x[i*ny - 1] = 0
            
            diagonals = [main_diag, off_x, off_x, off_y, off_y]
            offsets = [0, 1, -1, ny, -ny]
            
            A = diags(diagonals, offsets, format='csr')
            
            # 求解
            rhs_flat = rhs.flatten()
            phi_flat = spsolve(A, rhs_flat)
            phi = phi_flat.reshape(nx, ny)
            
        elif self.config.linear_solver == "cg":
            # 共轭梯度法
            A = self._build_laplacian_matrix_2d(nx, ny, dx, dy)
            rhs_flat = rhs.flatten()
            phi_flat, info = cg(A, rhs_flat, tol=self.config.tolerance, 
                               maxiter=self.config.max_iter)
            phi = phi_flat.reshape(nx, ny)
            
        else:
            raise ValueError(f"Unknown linear solver: {self.config.linear_solver}")
        
        return phi
    
    def _build_laplacian_matrix_2d(self, nx: int, ny: int, dx: float, dy: float) -> csr_matrix:
        """构建2D拉普拉斯算子矩阵"""
        n_total = nx * ny
        
        main_diag = -2 * (1/dx**2 + 1/dy**2) * np.ones(n_total)
        off_x = (1/dx**2) * np.ones(n_total - 1)
        off_y = (1/dy**2) * np.ones(n_total - ny)
        
        # 周期性边界处理
        for i in range(1, nx):
            off_x[i*ny - 1] = 0
        
        diagonals = [main_diag, off_x, off_x, off_y, off_y]
        offsets = [0, 1, -1, ny, -ny]
        
        return diags(diagonals, offsets, format='csr')
    
    def check_cfl(self, dt: float, dx: float, diffusion_coeff: float) -> bool:
        """
        检查CFL稳定性条件
        
        Args:
            dt: 时间步长
            dx: 空间步长
            diffusion_coeff: 扩散系数
            
        Returns:
            stable: 是否稳定
        """
        cfl = diffusion_coeff * dt / dx**2
        stable = cfl <= self.config.cfl_number
        
        if not stable:
            logger.warning(f"CFL condition violated: {cfl:.4f} > {self.config.cfl_number}")
        
        return stable
    
    def adaptive_time_step(self, field: np.ndarray, rhs_func: Callable,
                          dt_current: float, dx: float) -> float:
        """
        自适应时间步长选择
        
        基于CFL条件和变化率
        """
        rhs = rhs_func(field)
        max_rate = np.abs(rhs).max()
        
        if max_rate > 0:
            dt_suggested = 0.1 * np.abs(field).max() / max_rate
            dt_new = min(dt_suggested, dt_current * 1.1)  # 增长限制
            dt_new = max(dt_new, dt_current * 0.5)  # 最小值限制
        else:
            dt_new = dt_current
        
        return dt_new
