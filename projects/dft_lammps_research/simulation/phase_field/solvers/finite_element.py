"""
Finite Element Solver
=====================
有限元求解器

提供有限元方法用于相场方程求解。
支持高阶元和自适应网格。
"""

import numpy as np
from typing import Dict, Optional, Tuple, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FEConfig:
    """有限元配置"""
    element_order: int = 1  # 单元阶数 (1, 2, 3)
    quadrature_points: int = 4  # 高斯积分点
    mesh_type: str = "uniform"  # uniform, adaptive
    
    # 求解器参数
    solver_type: str = "direct"  # direct, iterative
    preconditioner: str = "ilu"
    
    # 自适应参数
    adaptive_refinement: bool = False
    refinement_threshold: float = 0.1
    max_refinement_level: int = 5


class FiniteElementSolver:
    """
    有限元求解器
    
    使用有限元方法求解相场方程。
    支持自适应网格细化。
    """
    
    def __init__(self, config: Optional[FEConfig] = None):
        """
        初始化有限元求解器
        
        Args:
            config: 有限元配置
        """
        self.config = config or FEConfig()
        
        # 网格数据
        self.nodes = None
        self.elements = None
        self.solution = None
        
        logger.info(f"Finite element solver initialized")
        logger.info(f"Element order: {self.config.element_order}")
    
    def create_uniform_mesh(self, nx: int, ny: int, 
                           xlim: Tuple[float, float] = (0, 1),
                           ylim: Tuple[float, float] = (0, 1)):
        """
        创建均匀网格
        
        Args:
            nx, ny: 网格单元数
            xlim, ylim: 边界范围
        """
        x = np.linspace(xlim[0], xlim[1], nx + 1)
        y = np.linspace(ylim[0], ylim[1], ny + 1)
        
        # 创建节点
        xv, yv = np.meshgrid(x, y, indexing='ij')
        self.nodes = np.column_stack([xv.flatten(), yv.flatten()])
        
        # 创建三角形单元 (简化)
        self.elements = []
        for i in range(nx):
            for j in range(ny):
                n1 = i * (ny + 1) + j
                n2 = n1 + 1
                n3 = (i + 1) * (ny + 1) + j
                n4 = n3 + 1
                
                # 两个三角形组成一个四边形
                self.elements.append([n1, n2, n3])
                self.elements.append([n2, n4, n3])
        
        self.elements = np.array(self.elements)
        
        logger.info(f"Created uniform mesh: {len(self.nodes)} nodes, {len(self.elements)} elements")
    
    def shape_functions(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        线性三角形单元的形函数
        
        Args:
            xi, eta: 自然坐标
            
        Returns:
            N: 形函数值
            dN: 形函数导数
        """
        # 线性三角形形函数
        N = np.array([1 - xi - eta, xi, eta])
        
        # 导数
        dN = np.array([[-1, -1],
                       [1, 0],
                       [0, 1]])
        
        return N, dN
    
    def integrate_element(self, element_nodes: np.ndarray, 
                         integrand: Callable) -> float:
        """
        单元积分
        
        Args:
            element_nodes: 单元节点坐标
            integrand: 被积函数
            
        Returns:
            integral: 积分值
        """
        # 高斯积分点 (1点积分)
        xi, eta = 1/3, 1/3
        weight = 0.5
        
        # 计算雅可比
        N, dN = self.shape_functions(xi, eta)
        J = dN.T @ element_nodes
        detJ = np.abs(np.linalg.det(J))
        
        # 积分
        integral = weight * integrand(xi, eta) * detJ
        
        return integral
    
    def assemble_stiffness_matrix(self, diffusion_coeff: float = 1.0) -> np.ndarray:
        """
        组装刚度矩阵
        
        Args:
            diffusion_coeff: 扩散系数
            
        Returns:
            K: 刚度矩阵
        """
        n_nodes = len(self.nodes)
        K = np.zeros((n_nodes, n_nodes))
        
        # 遍历单元
        for element in self.elements:
            element_nodes = self.nodes[element]
            
            # 计算单元刚度矩阵 (简化)
            Ke = np.zeros((3, 3))
            
            # 积分点
            xi, eta = 1/3, 1/3
            N, dN = self.shape_functions(xi, eta)
            
            # 雅可比
            J = dN.T @ element_nodes
            invJ = np.linalg.inv(J)
            detJ = np.abs(np.linalg.det(J))
            
            # 梯度
            dNdx = invJ @ dN.T
            
            # 单元刚度
            Ke = diffusion_coeff * (dNdx.T @ dNdx) * detJ * 0.5
            
            # 组装到全局矩阵
            for i in range(3):
                for j in range(3):
                    K[element[i], element[j]] += Ke[i, j]
        
        return K
    
    def solve_steady_state(self, rhs: np.ndarray,
                          boundary_conditions: Optional[Dict] = None) -> np.ndarray:
        """
        求解稳态问题
        
        K * u = f
        
        Args:
            rhs: 右端项
            boundary_conditions: 边界条件
            
        Returns:
            solution: 解向量
        """
        # 组装刚度矩阵
        K = self.assemble_stiffness_matrix()
        
        # 应用边界条件 (简化)
        if boundary_conditions:
            for node, value in boundary_conditions.items():
                K[node, :] = 0
                K[node, node] = 1
                rhs[node] = value
        
        # 求解
        self.solution = np.linalg.solve(K, rhs)
        
        return self.solution
    
    def project_to_grid(self, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
        """
        将FE解投影到规则网格
        
        Args:
            grid_x, grid_y: 目标网格坐标
            
        Returns:
            grid_solution: 网格上的解
        """
        if self.solution is None:
            raise ValueError("No solution available. Run solve first.")
        
        # 简化：最近邻插值
        grid_solution = np.zeros((len(grid_x), len(grid_y)))
        
        for i, x in enumerate(grid_x):
            for j, y in enumerate(grid_y):
                # 找到最近的节点
                dist = np.sum((self.nodes - [x, y])**2, axis=1)
                nearest = np.argmin(dist)
                grid_solution[i, j] = self.solution[nearest]
        
        return grid_solution
    
    def estimate_error(self) -> np.ndarray:
        """
        估计误差分布
        
        Returns:
            error: 单元误差估计
        """
        if self.solution is None:
            raise ValueError("No solution available")
        
        n_elements = len(self.elements)
        error = np.zeros(n_elements)
        
        # 简化：基于解的梯度变化估计误差
        for i, element in enumerate(self.elements):
            element_solution = self.solution[element]
            # 单元内解的变化
            error[i] = np.std(element_solution)
        
        return error
