#!/usr/bin/env python3
"""
continuum.py
============
连续介质力学模拟模块

功能：
1. 力学响应模拟（有限元分析）
2. 热传导分析
3. 耦合热-力问题
4. 与MD/DFT的耦合（从原子模拟提取材料参数）

支持的求解器：
- FEniCS/dolfinx: 高性能有限元求解
- Scipy/Sfepy: Python有限元工具
- 自定义实现：简单问题的直接求解

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

# 科学计算
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve, cg, gmres, bicgstab
from scipy.linalg import solve, cholesky, lu_factor, lu_solve
from scipy.interpolate import griddata, LinearNDInterpolator
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# 有限元专用（如果可用）
try:
    import meshio
    MESHIO_AVAILABLE = True
except ImportError:
    MESHIO_AVAILABLE = False
    warnings.warn("meshio not available. Mesh I/O features disabled.")

try:
    from skfem import *
    from skfem.helpers import grad, dot, ddot
    SKFEM_AVAILABLE = True
except ImportError:
    SKFEM_AVAILABLE = False
    warnings.warn("scikit-fem not available. Some FEM features disabled.")

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
class ElasticProperties:
    """弹性性质"""
    # 弹性常数 (GPa)
    C11: float = 100.0
    C12: float = 50.0
    C44: float = 50.0
    
    # 工程常数
    E: float = 0.0  # 杨氏模量 (GPa) - 如果为0则自动计算
    nu: float = 0.0  # 泊松比
    G: float = 0.0   # 剪切模量
    
    # 密度
    density: float = 1000.0  # kg/m³
    
    def __post_init__(self):
        """从弹性常数计算工程常数（如果未提供）"""
        if self.E == 0.0:
            # 对于立方晶系
            self.E = (self.C11 - self.C12) * (self.C11 + 2*self.C12) / (self.C11 + self.C12)
            self.nu = self.C12 / (self.C11 + self.C12)
            self.G = self.C44


@dataclass
class ThermalProperties:
    """热物理性质"""
    thermal_conductivity: float = 100.0  # W/(m·K)
    specific_heat: float = 500.0  # J/(kg·K)
    thermal_expansion: float = 1.0e-5  # 1/K
    density: float = 1000.0  # kg/m³
    
    # 热扩散系数
    @property
    def thermal_diffusivity(self) -> float:
        """热扩散系数 α = k/(ρ·Cp)"""
        return self.thermal_conductivity / (self.density * self.specific_heat)


@dataclass
class MaterialModel:
    """材料模型"""
    name: str = "elastic"
    elastic: ElasticProperties = field(default_factory=ElasticProperties)
    thermal: ThermalProperties = field(default_factory=ThermalProperties)
    
    # 塑性参数
    yield_stress: float = 0.0  # GPa
    hardening_modulus: float = 0.0  # GPa
    
    # 损伤参数
    damage_threshold: float = 1.0
    fracture_toughness: float = 0.0  # MPa·m^0.5


@dataclass
class FEMConfig:
    """有限元配置"""
    # 网格设置
    mesh_type: str = "structured"  # structured, unstructured, gmsh
    element_type: str = "quad"  # quad, tri, hex, tet
    element_order: int = 1
    
    # 网格尺寸
    lx: float = 100.0  # nm
    ly: float = 100.0
    lz: float = 100.0
    nx: int = 50
    ny: int = 50
    nz: int = 1  # 对于2D设为1
    
    # 维度
    dimensions: int = 2
    
    # 求解器设置
    solver_type: str = "direct"  # direct, cg, gmres
    preconditioner: str = "ilu"  # ilu, jacobi, none
    tolerance: float = 1e-8
    max_iterations: int = 10000


@dataclass
class MechanicsConfig:
    """力学分析配置"""
    fem_config: FEMConfig = field(default_factory=FEMConfig)
    
    # 分析类型
    analysis_type: str = "static"  # static, dynamic, modal, buckling
    
    # 载荷设置
    loads: List[Dict] = field(default_factory=list)
    
    # 边界条件
    bc_displacement: List[Dict] = field(default_factory=list)  # 位移边界
    bc_traction: List[Dict] = field(default_factory=list)      # 力边界
    
    # 时间步进（用于动力学）
    dt: float = 0.01
    n_steps: int = 1000
    
    # 输出
    output_stress: bool = True
    output_strain: bool = True
    output_displacement: bool = True


@dataclass
class ThermalConfig:
    """热传导分析配置"""
    fem_config: FEMConfig = field(default_factory=FEMConfig)
    
    # 分析类型
    analysis_type: str = "transient"  # steady, transient
    
    # 边界条件
    bc_temperature: List[Dict] = field(default_factory=list)  # 温度边界
    bc_heat_flux: List[Dict] = field(default_factory=list)    # 热流边界
    bc_convection: List[Dict] = field(default_factory=list)   # 对流边界
    
    # 热源
    heat_sources: List[Dict] = field(default_factory=list)
    
    # 初始条件
    initial_temperature: float = 300.0  # K
    
    # 时间步进
    dt: float = 0.001  # s
    n_steps: int = 1000
    
    # 输出
    output_interval: int = 10


@dataclass
class CoupledConfig:
    """耦合热-力分析配置"""
    mechanics: MechanicsConfig = field(default_factory=MechanicsConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    
    # 耦合参数
    coupling_scheme: str = "staggered"  # staggered, monolithic
    coupling_strength: str = "one_way"  # one_way, two_way
    
    # 时间步进
    dt: float = 0.001
    n_steps: int = 1000


# =============================================================================
# 有限元网格类
# =============================================================================

class FEMMesh:
    """
    有限元网格类
    
    管理节点坐标、单元连接和边界信息
    """
    
    def __init__(self, config: FEMConfig):
        self.config = config
        self.nodes = None  # (n_nodes, n_dim)
        self.elements = None  # (n_elements, n_nodes_per_element)
        self.node_sets = {}  # 命名节点集
        self.element_sets = {}  # 命名单元集
        
        self._generate_mesh()
    
    def _generate_mesh(self):
        """生成网格"""
        cfg = self.config
        
        if cfg.dimensions == 2:
            self._generate_2d_mesh()
        elif cfg.dimensions == 3:
            self._generate_3d_mesh()
    
    def _generate_2d_mesh(self):
        """生成2D结构化网格"""
        cfg = self.config
        
        # 节点坐标
        x = np.linspace(0, cfg.lx, cfg.nx + 1)
        y = np.linspace(0, cfg.ly, cfg.ny + 1)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        self.nodes = np.column_stack([X.flatten(), Y.flatten()])
        self.n_nodes = len(self.nodes)
        
        # 单元连接
        if cfg.element_type == "quad":
            # 四边形单元
            self.n_elements = cfg.nx * cfg.ny
            self.elements = np.zeros((self.n_elements, 4), dtype=int)
            
            e = 0
            for i in range(cfg.nx):
                for j in range(cfg.ny):
                    n1 = i * (cfg.ny + 1) + j
                    n2 = n1 + 1
                    n3 = n2 + (cfg.ny + 1)
                    n4 = n3 - 1
                    self.elements[e] = [n1, n2, n3, n4]
                    e += 1
            
            self.nodes_per_element = 4
            
        elif cfg.element_type == "tri":
            # 三角形单元（每个四边形分成两个三角形）
            self.n_elements = cfg.nx * cfg.ny * 2
            self.elements = np.zeros((self.n_elements, 3), dtype=int)
            
            e = 0
            for i in range(cfg.nx):
                for j in range(cfg.ny):
                    n1 = i * (cfg.ny + 1) + j
                    n2 = n1 + 1
                    n3 = n2 + (cfg.ny + 1)
                    n4 = n3 - 1
                    
                    self.elements[e] = [n1, n2, n4]
                    self.elements[e + 1] = [n2, n3, n4]
                    e += 2
            
            self.nodes_per_element = 3
    
    def _generate_3d_mesh(self):
        """生成3D结构化网格"""
        cfg = self.config
        
        # 节点坐标
        x = np.linspace(0, cfg.lx, cfg.nx + 1)
        y = np.linspace(0, cfg.ly, cfg.ny + 1)
        z = np.linspace(0, cfg.lz, cfg.nz + 1)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        self.nodes = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
        self.n_nodes = len(self.nodes)
        
        # 单元连接（六面体）
        self.n_elements = cfg.nx * cfg.ny * cfg.nz
        self.elements = np.zeros((self.n_elements, 8), dtype=int)
        
        ny1 = cfg.ny + 1
        nz1 = cfg.nz + 1
        
        e = 0
        for i in range(cfg.nx):
            for j in range(cfg.ny):
                for k in range(cfg.nz):
                    n1 = i * ny1 * nz1 + j * nz1 + k
                    n2 = n1 + 1
                    n3 = n2 + nz1
                    n4 = n3 - 1
                    n5 = n1 + ny1 * nz1
                    n6 = n5 + 1
                    n7 = n6 + nz1
                    n8 = n7 - 1
                    
                    self.elements[e] = [n1, n2, n3, n4, n5, n6, n7, n8]
                    e += 1
        
        self.nodes_per_element = 8
    
    def get_boundary_nodes(self, boundary: str) -> np.ndarray:
        """
        获取边界节点
        
        Args:
            boundary: "left", "right", "bottom", "top", "front", "back"
        """
        cfg = self.config
        eps = 1e-10
        
        if cfg.dimensions == 2:
            if boundary == "left":
                return np.where(np.abs(self.nodes[:, 0]) < eps)[0]
            elif boundary == "right":
                return np.where(np.abs(self.nodes[:, 0] - cfg.lx) < eps)[0]
            elif boundary == "bottom":
                return np.where(np.abs(self.nodes[:, 1]) < eps)[0]
            elif boundary == "top":
                return np.where(np.abs(self.nodes[:, 1] - cfg.ly) < eps)[0]
        
        return np.array([])
    
    def compute_jacobian(self, element_idx: int, xi: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        计算雅可比矩阵和行列式
        
        Args:
            element_idx: 单元索引
            xi: 局部坐标 (在参考单元中)
        
        Returns:
            J: 雅可比矩阵
            detJ: 行列式
        """
        nodes = self.nodes[self.elements[element_idx]]
        
        # 形函数梯度（线性单元）
        if self.nodes_per_element == 4:  # 四边形
            dN_dxi = np.array([[-0.25, 0.25, 0.25, -0.25],
                               [-0.25, -0.25, 0.25, 0.25]])
        elif self.nodes_per_element == 3:  # 三角形
            x = nodes[:, 0]
            y = nodes[:, 1]
            area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
            dN_dxi = np.array([[y[1]-y[2], y[2]-y[0], y[0]-y[1]],
                               [x[2]-x[1], x[0]-x[2], x[1]-x[0]]]) / (2*area)
            return dN_dxi, 2*area
        else:
            # 默认线性形函数梯度
            dN_dxi = np.eye(self.config.dimensions)
        
        # J = dN_dxi * node_coords
        J = dN_dxi @ nodes
        detJ = np.linalg.det(J)
        
        return J, detJ
    
    def export_to_vtk(self, filename: str, **field_data):
        """导出到VTK格式（用于可视化）"""
        if not MESHIO_AVAILABLE:
            logger.warning("meshio not available. Cannot export to VTK.")
            return
        
        # 准备单元数据
        cell_type = "quad" if self.nodes_per_element == 4 else "triangle"
        if self.config.dimensions == 3:
            cell_type = "hexahedron"
        
        cells = [(cell_type, self.elements)]
        
        # 添加场数据
        point_data = {}
        for name, data in field_data.items():
            if len(data) == self.n_nodes:
                point_data[name] = data
        
        mesh = meshio.Mesh(self.nodes, cells, point_data=point_data)
        mesh.write(filename)
        
        logger.info(f"Exported mesh to {filename}")


# =============================================================================
# 有限元求解器基类
# =============================================================================

class FEMSolver:
    """
    有限元求解器基类
    
    实现通用的有限元组装和求解功能
    """
    
    def __init__(self, mesh: FEMMesh, material: MaterialModel):
        self.mesh = mesh
        self.material = material
        self.dof_per_node = 1  # 将被子类覆盖
        
        # 全局矩阵和向量
        self.K = None  # 刚度矩阵
        self.F = None  # 载荷向量
        self.U = None  # 解向量
    
    def assemble_stiffness(self):
        """组装刚度矩阵（子类实现）"""
        raise NotImplementedError
    
    def assemble_load(self):
        """组装载荷向量（子类实现）"""
        raise NotImplementedError
    
    def apply_boundary_conditions(self, 
                                  bc_nodes: np.ndarray,
                                  bc_values: np.ndarray):
        """应用边界条件（罚函数法）"""
        penalty = 1e15
        
        for node, value in zip(bc_nodes, bc_values):
            for dof in range(self.dof_per_node):
                global_dof = node * self.dof_per_node + dof
                self.K[global_dof, global_dof] += penalty
                self.F[global_dof] += penalty * value
    
    def solve(self) -> np.ndarray:
        """
        求解线性系统
        
        Returns:
            U: 解向量
        """
        n_dof = self.mesh.n_nodes * self.dof_per_node
        
        # 转换为稀疏矩阵格式
        if not isinstance(self.K, csr_matrix):
            self.K = csr_matrix(self.K)
        
        # 求解
        try:
            if self.mesh.config.solver_type == "direct":
                self.U = spsolve(self.K, self.F)
            elif self.mesh.config.solver_type == "cg":
                self.U, info = cg(self.K, self.F, tol=self.mesh.config.tolerance,
                                  maxiter=self.mesh.config.max_iterations)
                if info != 0:
                    logger.warning(f"CG solver did not converge, info={info}")
            elif self.mesh.config.solver_type == "gmres":
                self.U, info = gmres(self.K, self.F, tol=self.mesh.config.tolerance,
                                     maxiter=self.mesh.config.max_iterations)
            else:
                self.U = spsolve(self.K, self.F)
        
        except Exception as e:
            logger.error(f"Solver failed: {e}")
            self.U = np.zeros(n_dof)
        
        return self.U
    
    def get_element_stiffness(self, element_idx: int) -> np.ndarray:
        """计算单元刚度矩阵（子类实现）"""
        raise NotImplementedError


# =============================================================================
# 热传导求解器
# =============================================================================

class ThermalSolver(FEMSolver):
    """
    热传导有限元求解器
    
    求解方程: ρ·Cp·∂T/∂t = ∇·(k∇T) + Q
    """
    
    def __init__(self, mesh: FEMMesh, material: MaterialModel, config: ThermalConfig):
        super().__init__(mesh, material)
        self.thermal_config = config
        self.dof_per_node = 1
        
        # 场变量
        self.T = None  # 温度场
        self.T_prev = None  # 上一时间步温度
    
    def assemble_stiffness(self, transient: bool = False, dt: float = None):
        """
        组装热传导刚度矩阵
        
        K_ij = ∫(k ∇N_i · ∇N_j)dΩ + (1/dt)∫(ρCp N_i N_j)dΩ (瞬态项)
        """
        n_nodes = self.mesh.n_nodes
        k = np.zeros((n_nodes, n_nodes))
        
        # 材料参数
        thermal_cond = self.material.thermal.thermal_conductivity
        rho = self.material.thermal.density
        cp = self.material.thermal.specific_heat
        
        # 数值积分点（高斯积分）
        gauss_points, gauss_weights = self._get_gauss_points()
        
        # 遍历所有单元
        for e in range(self.mesh.n_elements):
            element_nodes = self.mesh.elements[e]
            k_e = np.zeros((self.mesh.nodes_per_element, self.mesh.nodes_per_element))
            
            for xi, w in zip(gauss_points, gauss_weights):
                # 计算雅可比
                J, detJ = self.mesh.compute_jacobian(e, xi)
                if abs(detJ) < 1e-15:
                    continue
                
                # 形函数梯度
                dN_dxi = self._shape_function_grad(xi)
                dN_dx = np.linalg.solve(J, dN_dxi)
                
                # 刚度矩阵贡献
                k_e += thermal_cond * (dN_dx.T @ dN_dx) * detJ * w
                
                # 瞬态项质量矩阵
                if transient and dt:
                    N = self._shape_function(xi)
                    k_e += (rho * cp / dt) * np.outer(N, N) * detJ * w
            
            # 组装到全局矩阵
            for i, ni in enumerate(element_nodes):
                for j, nj in enumerate(element_nodes):
                    k[ni, nj] += k_e[i, j]
        
        self.K = k
    
    def assemble_load(self, heat_source: Optional[np.ndarray] = None, 
                     transient: bool = False, dt: float = None):
        """
        组装热载荷向量
        
        F_i = ∫(Q N_i)dΩ + (1/dt)∫(ρCp T_prev N_i)dΩ
        """
        n_nodes = self.mesh.n_nodes
        f = np.zeros(n_nodes)
        
        rho = self.material.thermal.density
        cp = self.material.thermal.specific_heat
        
        # 体积热源
        if heat_source is not None:
            gauss_points, gauss_weights = self._get_gauss_points()
            
            for e in range(self.mesh.n_elements):
                element_nodes = self.mesh.elements[e]
                f_e = np.zeros(self.mesh.nodes_per_element)
                
                for xi, w in zip(gauss_points, gauss_weights):
                    J, detJ = self.mesh.compute_jacobian(e, xi)
                    if abs(detJ) < 1e-15:
                        continue
                    
                    N = self._shape_function(xi)
                    # 插值热源到积分点
                    q = np.mean(heat_source[element_nodes]) if len(heat_source) > 0 else 0
                    f_e += q * N * detJ * w
                
                for i, ni in enumerate(element_nodes):
                    f[ni] += f_e[i]
        
        # 瞬态项
        if transient and dt and self.T_prev is not None:
            f += (rho * cp / dt) * self.T_prev
        
        self.F = f
    
    def apply_temperature_bc(self, bc_nodes: np.ndarray, T_values: np.ndarray):
        """应用温度边界条件"""
        self.apply_boundary_conditions(bc_nodes, T_values)
    
    def solve_steady(self, heat_source: Optional[np.ndarray] = None) -> np.ndarray:
        """求解稳态热传导"""
        self.assemble_stiffness(transient=False)
        self.assemble_load(heat_source=heat_source)
        
        # 应用温度边界条件
        for bc in self.thermal_config.bc_temperature:
            nodes = self.mesh.get_boundary_nodes(bc['boundary'])
            values = np.full(len(nodes), bc['value'])
            self.apply_temperature_bc(nodes, values)
        
        self.T = self.solve()
        return self.T
    
    def solve_transient(self, n_steps: Optional[int] = None, 
                        dt: Optional[float] = None) -> List[np.ndarray]:
        """
        求解瞬态热传导
        
        Returns:
            temperature_history: 温度场历史列表
        """
        cfg = self.thermal_config
        n_steps = n_steps or cfg.n_steps
        dt = dt or cfg.dt
        
        # 初始化
        self.T_prev = np.full(self.mesh.n_nodes, cfg.initial_temperature)
        temperature_history = [self.T_prev.copy()]
        
        # 组装刚度矩阵（只执行一次）
        self.assemble_stiffness(transient=True, dt=dt)
        
        logger.info(f"Starting transient thermal analysis for {n_steps} steps")
        
        for step in range(n_steps):
            # 组装载荷向量
            self.assemble_load(transient=True, dt=dt)
            
            # 应用边界条件
            for bc in cfg.bc_temperature:
                nodes = self.mesh.get_boundary_nodes(bc['boundary'])
                values = np.full(len(nodes), bc['value'])
                self.apply_temperature_bc(nodes, values)
            
            # 求解
            self.T = self.solve()
            self.T_prev = self.T.copy()
            
            # 存储
            if step % cfg.output_interval == 0:
                temperature_history.append(self.T.copy())
            
            if step % 100 == 0:
                logger.info(f"Thermal step {step}/{n_steps}")
        
        return temperature_history
    
    def compute_heat_flux(self) -> np.ndarray:
        """
        计算热流密度 q = -k ∇T
        
        Returns:
            heat_flux: (n_nodes, n_dim) 热流密度向量
        """
        if self.T is None:
            raise ValueError("Temperature field not available. Run solve first.")
        
        k = self.material.thermal.thermal_conductivity
        flux = np.zeros((self.mesh.n_nodes, self.mesh.config.dimensions))
        
        # 在单元中心计算梯度，然后外推到节点
        for e in range(self.mesh.n_elements):
            element_nodes = self.mesh.elements[e]
            T_e = self.T[element_nodes]
            
            # 单元中心梯度
            xi_center = np.zeros(self.mesh.config.dimensions)
            J, detJ = self.mesh.compute_jacobian(e, xi_center)
            dN_dxi = self._shape_function_grad(xi_center)
            dN_dx = np.linalg.solve(J, dN_dxi)
            
            grad_T = dN_dx.T @ T_e
            q = -k * grad_T
            
            # 分配到节点（简单平均）
            for ni in element_nodes:
                flux[ni] += q
        
        # 平均
        node_count = np.zeros(self.mesh.n_nodes)
        for e in range(self.mesh.n_elements):
            for ni in self.mesh.elements[e]:
                node_count[ni] += 1
        
        flux /= node_count[:, np.newaxis]
        
        return flux
    
    def _get_gauss_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取高斯积分点和权重"""
        if self.mesh.config.dimensions == 2:
            if self.mesh.nodes_per_element == 4:  # 四边形
                # 2x2高斯积分
                gp = 1.0 / np.sqrt(3)
                points = np.array([[-gp, -gp], [gp, -gp], [gp, gp], [-gp, gp]])
                weights = np.ones(4)
                return points, weights
            elif self.mesh.nodes_per_element == 3:  # 三角形
                # 单点积分
                points = np.array([[1/3, 1/3]])
                weights = np.array([0.5])
                return points, weights
        
        # 默认
        return np.array([[0.0, 0.0]]), np.array([1.0])
    
    def _shape_function(self, xi: np.ndarray) -> np.ndarray:
        """计算形函数值"""
        if self.mesh.nodes_per_element == 4:
            # 双线性形函数
            return 0.25 * np.array([
                (1 - xi[0]) * (1 - xi[1]),
                (1 + xi[0]) * (1 - xi[1]),
                (1 + xi[0]) * (1 + xi[1]),
                (1 - xi[0]) * (1 + xi[1])
            ])
        elif self.mesh.nodes_per_element == 3:
            # 三角形线性形函数（仅在参考单元中定义）
            return np.array([1 - xi[0] - xi[1], xi[0], xi[1]])
        
        return np.ones(self.mesh.nodes_per_element) / self.mesh.nodes_per_element
    
    def _shape_function_grad(self, xi: np.ndarray) -> np.ndarray:
        """计算形函数梯度（参考单元中）"""
        if self.mesh.nodes_per_element == 4:
            return 0.25 * np.array([
                [-(1 - xi[1]), -(1 - xi[0])],
                [(1 - xi[1]), -(1 + xi[0])],
                [(1 + xi[1]), (1 + xi[0])],
                [-(1 + xi[1]), (1 - xi[0])]
            ]).T
        elif self.mesh.nodes_per_element == 3:
            return np.array([[-1, -1], [1, 0], [0, 1]]).T
        
        return np.eye(self.mesh.config.dimensions)


# =============================================================================
# 力学求解器
# =============================================================================

class MechanicsSolver(FEMSolver):
    """
    力学有限元求解器
    
    求解方程: ∇·σ + f = 0
    其中 σ = C : ε
    """
    
    def __init__(self, mesh: FEMMesh, material: MaterialModel, config: MechanicsConfig):
        super().__init__(mesh, material)
        self.mechanics_config = config
        self.dof_per_node = mesh.config.dimensions  # 每个节点2或3个自由度
        
        # 场变量
        self.displacement = None  # 位移场 (n_nodes, n_dim)
        self.strain = None        # 应变场 (n_elements, n_strain_components)
        self.stress = None        # 应力场 (n_elements, n_stress_components)
    
    def get_elastic_matrix(self) -> np.ndarray:
        """获取弹性矩阵D（平面应力/应变或3D）"""
        cfg = self.mesh.config
        mat = self.material.elastic
        
        if cfg.dimensions == 2:
            # 平面应变假设
            factor = mat.E / ((1 + mat.nu) * (1 - 2 * mat.nu))
            D = factor * np.array([
                [1 - mat.nu, mat.nu, 0],
                [mat.nu, 1 - mat.nu, 0],
                [0, 0, (1 - 2 * mat.nu) / 2]
            ])
        else:
            # 3D弹性矩阵
            C11 = mat.C11
            C12 = mat.C12
            C44 = mat.C44
            D = np.array([
                [C11, C12, C12, 0, 0, 0],
                [C12, C11, C12, 0, 0, 0],
                [C12, C12, C11, 0, 0, 0],
                [0, 0, 0, C44, 0, 0],
                [0, 0, 0, 0, C44, 0],
                [0, 0, 0, 0, 0, C44]
            ])
        
        return D
    
    def assemble_stiffness(self):
        """
        组装刚度矩阵
        
        K = ∫ B^T D B dΩ
        """
        n_dof = self.mesh.n_nodes * self.dof_per_node
        K = lil_matrix((n_dof, n_dof))
        
        D = self.get_elastic_matrix()
        n_strain = D.shape[0]
        
        gauss_points, gauss_weights = self._get_gauss_points()
        
        for e in range(self.mesh.n_elements):
            element_nodes = self.mesh.elements[e]
            n_nodes_e = len(element_nodes)
            
            k_e = np.zeros((n_nodes_e * self.dof_per_node, n_nodes_e * self.dof_per_node))
            
            for xi, w in zip(gauss_points, gauss_weights):
                # 雅可比
                J, detJ = self.mesh.compute_jacobian(e, xi)
                if abs(detJ) < 1e-15:
                    continue
                
                # 形函数梯度
                dN_dxi = self._shape_function_grad(xi)
                dN_dx = np.linalg.solve(J, dN_dxi)
                
                # B矩阵
                B = self._get_B_matrix(dN_dx)
                
                # 单元刚度
                k_e += B.T @ D @ B * detJ * w
            
            # 组装
            for i, ni in enumerate(element_nodes):
                for j, nj in enumerate(element_nodes):
                    for di in range(self.dof_per_node):
                        for dj in range(self.dof_per_node):
                            row = ni * self.dof_per_node + di
                            col = nj * self.dof_per_node + dj
                            K[row, col] += k_e[i * self.dof_per_node + di, 
                                               j * self.dof_per_node + dj]
        
        self.K = K.tocsr()
    
    def assemble_load(self, body_force: Optional[np.ndarray] = None):
        """
        组装载荷向量
        
        F = ∫ N^T f dΩ + 边界项
        """
        n_dof = self.mesh.n_nodes * self.dof_per_node
        F = np.zeros(n_dof)
        
        # 体力
        if body_force is not None:
            gauss_points, gauss_weights = self._get_gauss_points()
            
            for e in range(self.mesh.n_elements):
                element_nodes = self.mesh.elements[e]
                f_e = np.zeros(len(element_nodes) * self.dof_per_node)
                
                for xi, w in zip(gauss_points, gauss_weights):
                    J, detJ = self.mesh.compute_jacobian(e, xi)
                    if abs(detJ) < 1e-15:
                        continue
                    
                    N = self._shape_function(xi)
                    
                    # 体力在积分点的值
                    if body_force.ndim == 1:
                        f_body = np.full(self.dof_per_node, body_force[0])
                    else:
                        f_body = body_force[element_nodes[0]]
                    
                    # 组装
                    for i, ni in enumerate(element_nodes):
                        for d in range(self.dof_per_node):
                            f_e[i * self.dof_per_node + d] += N[i] * f_body[d] * detJ * w
                
                # 添加到全局
                for i, ni in enumerate(element_nodes):
                    for d in range(self.dof_per_node):
                        F[ni * self.dof_per_node + d] += f_e[i * self.dof_per_node + d]
        
        self.F = F
    
    def apply_displacement_bc(self, bc_nodes: np.ndarray, 
                              bc_values: np.ndarray,
                              direction: Optional[int] = None):
        """
        应用位移边界条件
        
        Args:
            bc_nodes: 边界节点索引
            bc_values: 位移值
            direction: 方向 (0=x, 1=y, 2=z)，如果为None则所有方向
        """
        penalty = 1e15
        
        if direction is None:
            # 所有方向
            for i, node in enumerate(bc_nodes):
                for d in range(self.dof_per_node):
                    dof = node * self.dof_per_node + d
                    self.K[dof, dof] += penalty
                    self.F[dof] += penalty * bc_values[i]
        else:
            # 特定方向
            for i, node in enumerate(bc_nodes):
                dof = node * self.dof_per_node + direction
                self.K[dof, dof] += penalty
                self.F[dof] += penalty * bc_values[i]
    
    def apply_traction_bc(self, boundary: str, traction: np.ndarray):
        """
        应用面力边界条件
        
        Args:
            boundary: 边界名称
            traction: 面力向量 (n_dim,)
        """
        # 简化实现：将面力转换为等效节点力
        bc_nodes = self.mesh.get_boundary_nodes(boundary)
        
        # 平均分配到边界节点
        n_nodes = len(bc_nodes)
        if n_nodes > 0:
            force_per_node = traction / n_nodes
            
            for node in bc_nodes:
                for d in range(self.dof_per_node):
                    self.F[node * self.dof_per_node + d] += force_per_node[d]
    
    def solve_static(self, body_force: Optional[np.ndarray] = None) -> np.ndarray:
        """求解静力学问题"""
        self.assemble_stiffness()
        self.assemble_load(body_force=body_force)
        
        # 应用边界条件
        for bc in self.mechanics_config.bc_displacement:
            nodes = self.mesh.get_boundary_nodes(bc['boundary'])
            values = np.full(len(nodes), bc['value'])
            direction = bc.get('direction', None)
            self.apply_displacement_bc(nodes, values, direction)
        
        # 面力边界
        for bc in self.mechanics_config.bc_traction:
            self.apply_traction_bc(bc['boundary'], np.array(bc['traction']))
        
        # 求解
        U = self.solve()
        
        # 重塑为 (n_nodes, n_dim)
        self.displacement = U.reshape(self.mesh.n_nodes, self.dof_per_node)
        
        # 计算应力和应变
        self._compute_stress_strain()
        
        return self.displacement
    
    def _compute_stress_strain(self):
        """计算单元应力和应变"""
        n_elements = self.mesh.n_elements
        n_strain = 3 if self.mesh.config.dimensions == 2 else 6
        
        self.strain = np.zeros((n_elements, n_strain))
        self.stress = np.zeros((n_elements, n_strain))
        
        D = self.get_elastic_matrix()
        
        for e in range(n_elements):
            element_nodes = self.mesh.elements[e]
            u_e = self.displacement[element_nodes].flatten()
            
            # 在单元中心计算
            xi_center = np.zeros(self.mesh.config.dimensions)
            J, detJ = self.mesh.compute_jacobian(e, xi_center)
            dN_dxi = self._shape_function_grad(xi_center)
            dN_dx = np.linalg.solve(J, dN_dxi)
            
            # B矩阵
            B = self._get_B_matrix(dN_dx)
            
            # 应变
            epsilon = B @ u_e
            self.strain[e] = epsilon
            
            # 应力
            sigma = D @ epsilon
            self.stress[e] = sigma
    
    def _get_B_matrix(self, dN_dx: np.ndarray) -> np.ndarray:
        """
        构建应变-位移矩阵B
        
        Args:
            dN_dx: 形函数空间导数 (n_dim, n_nodes)
        
        Returns:
            B: 应变-位移矩阵
        """
        n_nodes = dN_dx.shape[1]
        n_dim = self.mesh.config.dimensions
        
        if n_dim == 2:
            # 平面应变
            B = np.zeros((3, n_nodes * 2))
            for i in range(n_nodes):
                B[0, i * 2] = dN_dx[0, i]      # ε_xx
                B[1, i * 2 + 1] = dN_dx[1, i]  # ε_yy
                B[2, i * 2] = dN_dx[1, i]      # γ_xy
                B[2, i * 2 + 1] = dN_dx[0, i]
        else:
            # 3D
            B = np.zeros((6, n_nodes * 3))
            for i in range(n_nodes):
                B[0, i * 3] = dN_dx[0, i]      # ε_xx
                B[1, i * 3 + 1] = dN_dx[1, i]  # ε_yy
                B[2, i * 3 + 2] = dN_dx[2, i]  # ε_zz
                B[3, i * 3] = dN_dx[1, i]      # γ_xy
                B[3, i * 3 + 1] = dN_dx[0, i]
                B[4, i * 3 + 1] = dN_dx[2, i]  # γ_yz
                B[4, i * 3 + 2] = dN_dx[1, i]
                B[5, i * 3] = dN_dx[2, i]      # γ_xz
                B[5, i * 3 + 2] = dN_dx[0, i]
        
        return B
    
    def get_von_mises_stress(self) -> np.ndarray:
        """计算Von Mises等效应力"""
        if self.stress is None:
            return np.zeros(self.mesh.n_elements)
        
        if self.mesh.config.dimensions == 2:
            # 平面应变
            sigma = self.stress
            vm = np.sqrt(sigma[:, 0]**2 - sigma[:, 0]*sigma[:, 1] + 
                        sigma[:, 1]**2 + 3*sigma[:, 2]**2)
        else:
            # 3D
            sigma = self.stress
            s11, s22, s33 = sigma[:, 0], sigma[:, 1], sigma[:, 2]
            s12, s23, s13 = sigma[:, 3], sigma[:, 4], sigma[:, 5]
            vm = np.sqrt(0.5 * ((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2 + 
                                6*(s12**2 + s23**2 + s13**2)))
        
        return vm
    
    def _get_gauss_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取高斯积分点"""
        if self.mesh.config.dimensions == 2:
            if self.mesh.nodes_per_element == 4:
                gp = 1.0 / np.sqrt(3)
                return np.array([[-gp, -gp], [gp, -gp], [gp, gp], [-gp, gp]]), np.ones(4)
            elif self.mesh.nodes_per_element == 3:
                return np.array([[1/3, 1/3]]), np.array([0.5])
        
        return np.array([[0.0, 0.0]]), np.array([1.0])
    
    def _shape_function(self, xi: np.ndarray) -> np.ndarray:
        """形函数"""
        if self.mesh.nodes_per_element == 4:
            return 0.25 * np.array([
                (1 - xi[0]) * (1 - xi[1]),
                (1 + xi[0]) * (1 - xi[1]),
                (1 + xi[0]) * (1 + xi[1]),
                (1 - xi[0]) * (1 + xi[1])
            ])
        elif self.mesh.nodes_per_element == 3:
            return np.array([1 - xi[0] - xi[1], xi[0], xi[1]])
        
        return np.ones(self.mesh.nodes_per_element) / self.mesh.nodes_per_element
    
    def _shape_function_grad(self, xi: np.ndarray) -> np.ndarray:
        """形函数梯度"""
        if self.mesh.nodes_per_element == 4:
            return 0.25 * np.array([
                [-(1 - xi[1]), -(1 - xi[0])],
                [(1 - xi[1]), -(1 + xi[0])],
                [(1 + xi[1]), (1 + xi[0])],
                [-(1 + xi[1]), (1 - xi[0])]
            ]).T
        elif self.mesh.nodes_per_element == 3:
            return np.array([[-1, -1], [1, 0], [0, 1]]).T
        
        return np.eye(self.mesh.config.dimensions)


# =============================================================================
# 热-力耦合求解器
# =============================================================================

class CoupledThermalMechanicsSolver:
    """
    热-力耦合求解器
    
    求解顺序耦合问题:
    1. 求解温度场
    2. 计算热应力
    """
    
    def __init__(self, 
                 mesh: FEMMesh,
                 material: MaterialModel,
                 config: CoupledConfig):
        self.mesh = mesh
        self.material = material
        self.config = config
        
        self.thermal_solver = ThermalSolver(mesh, material, config.thermal)
        self.mechanics_solver = MechanicsSolver(mesh, material, config.mechanics)
        
        # 结果
        self.temperature_history = []
        self.displacement_history = []
        self.stress_history = []
    
    def solve(self, n_steps: Optional[int] = None) -> Dict:
        """
        求解耦合问题
        
        使用顺序耦合（staggered）方案:
        1. 求解瞬态热传导
        2. 在每个输出步计算热应力
        """
        cfg = self.config
        n_steps = n_steps or cfg.n_steps
        
        logger.info(f"Starting coupled thermal-mechanics analysis for {n_steps} steps")
        
        # 步骤1: 求解瞬态热传导
        self.temperature_history = self.thermal_solver.solve_transient(
            n_steps=n_steps,
            dt=cfg.dt
        )
        
        logger.info(f"Thermal solution complete. Computing mechanical response...")
        
        # 步骤2: 在每个时间步计算热应力
        for step, T in enumerate(self.temperature_history):
            # 计算热应变对应的等效节点力
            # ε_th = α * (T - T_ref)
            alpha = self.material.thermal.thermal_expansion
            T_ref = self.config.thermal.initial_temperature
            
            delta_T = T - T_ref
            
            # 热应变作为初始应变处理
            # 这里简化为通过等效体力来考虑
            thermal_force = self._compute_thermal_force(delta_T)
            
            # 求解力学问题
            self.mechanics_solver.assemble_stiffness()
            self.mechanics_solver.assemble_load(body_force=thermal_force)
            
            # 应用边界条件
            for bc in cfg.mechanics.bc_displacement:
                nodes = self.mesh.get_boundary_nodes(bc['boundary'])
                values = np.full(len(nodes), bc['value'])
                direction = bc.get('direction', None)
                self.mechanics_solver.apply_displacement_bc(nodes, values, direction)
            
            U = self.mechanics_solver.solve()
            self.mechanics_solver.displacement = U.reshape(
                self.mesh.n_nodes, self.mesh.config.dimensions
            )
            self.mechanics_solver._compute_stress_strain()
            
            # 存储结果
            self.displacement_history.append(self.mechanics_solver.displacement.copy())
            self.stress_history.append(self.mechanics_solver.stress.copy())
            
            if step % 10 == 0:
                logger.info(f"Coupled step {step}/{len(self.temperature_history)}")
        
        logger.info("Coupled analysis complete")
        
        return {
            'n_steps': len(self.temperature_history),
            'max_displacement': np.max(np.abs(self.displacement_history[-1])),
            'max_stress': np.max(self.stress_history[-1])
        }
    
    def _compute_thermal_force(self, delta_T: np.ndarray) -> np.ndarray:
        """
        计算热载荷等效节点力
        
        简化实现：将温度变化转换为等效力
        """
        alpha = self.material.thermal.thermal_expansion
        E = self.material.elastic.E
        
        # 热应力 = E * α * ΔT
        # 简化为沿y方向的等效体力
        force = np.zeros((self.mesh.n_nodes, self.mesh.config.dimensions))
        force[:, 1] = E * alpha * delta_T * 1e6  # 转换为合适的单位
        
        return force


# =============================================================================
# FEniCS接口
# =============================================================================

class FEniCSInterface:
    """
    FEniCS/dolfinx接口
    
    为复杂的连续介质问题提供更高级的求解能力
    """
    
    def __init__(self):
        self.has_fenics = False
        try:
            import dolfinx
            import ufl
            from mpi4py import MPI
            self.has_fenics = True
            self.dolfinx = dolfinx
            self.ufl = ufl
            self.MPI = MPI
            logger.info("FEniCS/dolfinx interface initialized")
        except ImportError:
            logger.warning("FEniCS/dolfinx not available. Install with: pip install fenics-dolfinx")
    
    def create_mesh(self, mesh: FEMMesh):
        """从我们的网格格式创建FEniCS网格"""
        if not self.has_fenics:
            raise ImportError("FEniCS not available")
        
        # 使用meshio转换
        if MESHIO_AVAILABLE:
            meshio_mesh = meshio.Mesh(
                mesh.nodes,
                [("quad" if mesh.nodes_per_element == 4 else "triangle", mesh.elements)]
            )
            
            # 保存为XDMF格式
            temp_file = "temp_mesh.xdmf"
            meshio_mesh.write(temp_file)
            
            # 读取到FEniCS
            from dolfinx.io import XDMFFile
            with XDMFFile(self.MPI.COMM_WORLD, temp_file, "r") as file:
                fenics_mesh = file.read_mesh(name="Grid")
            
            return fenics_mesh
        
        return None
    
    def solve_thermal_fenics(self, 
                            mesh: FEMMesh,
                            material: MaterialModel,
                            bc_temperature: Dict,
                            heat_source: Optional[Callable] = None) -> np.ndarray:
        """使用FEniCS求解热传导问题"""
        if not self.has_fenics:
            raise ImportError("FEniCS not available")
        
        # 创建FEniCS网格
        fenics_mesh = self.create_mesh(mesh)
        if fenics_mesh is None:
            raise RuntimeError("Failed to create FEniCS mesh")
        
        # 定义函数空间
        V = self.dolfinx.fem.functionspace(fenics_mesh, ("CG", 1))
        
        # 定义试函数和测试函数
        u = self.ufl.TrialFunction(V)
        v = self.ufl.TestFunction(V)
        
        # 材料参数
        k = material.thermal.thermal_conductivity
        
        # 变分形式
        a = k * self.ufl.dot(self.ufl.grad(u), self.ufl.grad(v)) * self.ufl.dx
        
        # 热源项
        if heat_source:
            f = heat_source
            L = f * v * self.ufl.dx
        else:
            L = 0 * v * self.ufl.dx
        
        # 边界条件
        # (简化实现，实际应使用facet tags)
        
        # 求解
        problem = self.dolfinx.fem.petsc.LinearProblem(
            a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        uh = problem.solve()
        
        return uh.x.array


# =============================================================================
# 主工作流类
# =============================================================================

class ContinuumWorkflow:
    """
    连续介质模拟完整工作流
    """
    
    def __init__(self, working_dir: str = "./continuum_workflow"):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        self.mesh = None
        self.material = None
        self.solver = None
    
    def setup_material(self,
                      elastic_props: Optional[Dict] = None,
                      thermal_props: Optional[Dict] = None) -> MaterialModel:
        """
        设置材料模型
        
        可以从DFT计算结果自动提取弹性常数
        """
        if elastic_props:
            elastic = ElasticProperties(**elastic_props)
        else:
            elastic = ElasticProperties()
        
        if thermal_props:
            thermal = ThermalProperties(**thermal_props)
        else:
            thermal = ThermalProperties()
        
        self.material = MaterialModel(elastic=elastic, thermal=thermal)
        
        logger.info("Material model configured")
        
        return self.material
    
    def generate_mesh(self, config: FEMConfig) -> FEMMesh:
        """生成有限元网格"""
        self.mesh = FEMMesh(config)
        
        logger.info(f"Generated mesh: {self.mesh.n_nodes} nodes, {self.mesh.n_elements} elements")
        
        return self.mesh
    
    def run_thermal_analysis(self, config: ThermalConfig) -> Dict:
        """运行热传导分析"""
        if self.mesh is None:
            raise ValueError("Mesh not generated. Call generate_mesh first.")
        if self.material is None:
            raise ValueError("Material not configured. Call setup_material first.")
        
        self.solver = ThermalSolver(self.mesh, self.material, config)
        
        if config.analysis_type == "steady":
            T = self.solver.solve_steady()
            results = {
                'temperature': T,
                'max_temp': np.max(T),
                'min_temp': np.min(T),
                'avg_temp': np.mean(T)
            }
        else:  # transient
            T_history = self.solver.solve_transient()
            results = {
                'temperature_history': T_history,
                'final_temperature': T_history[-1],
                'n_steps': len(T_history)
            }
        
        # 保存结果
        self._save_thermal_results(results)
        
        return results
    
    def run_mechanics_analysis(self, config: MechanicsConfig) -> Dict:
        """运行力学分析"""
        if self.mesh is None:
            raise ValueError("Mesh not generated. Call generate_mesh first.")
        if self.material is None:
            raise ValueError("Material not configured. Call setup_material first.")
        
        self.solver = MechanicsSolver(self.mesh, self.material, config)
        
        displacement = self.solver.solve_static()
        von_mises = self.solver.get_von_mises_stress()
        
        results = {
            'displacement': displacement,
            'stress': self.solver.stress,
            'strain': self.solver.strain,
            'von_mises_stress': von_mises,
            'max_displacement': np.max(np.abs(displacement)),
            'max_von_mises': np.max(von_mises)
        }
        
        # 保存结果
        self._save_mechanics_results(results)
        
        return results
    
    def run_coupled_analysis(self, config: CoupledConfig) -> Dict:
        """运行热-力耦合分析"""
        if self.mesh is None:
            raise ValueError("Mesh not generated. Call generate_mesh first.")
        if self.material is None:
            raise ValueError("Material not configured. Call setup_material first.")
        
        solver = CoupledThermalMechanicsSolver(self.mesh, self.material, config)
        results = solver.solve()
        
        results.update({
            'temperature_history': solver.temperature_history,
            'displacement_history': solver.displacement_history,
            'stress_history': solver.stress_history
        })
        
        return results
    
    def _save_thermal_results(self, results: Dict):
        """保存热分析结果"""
        # 保存温度场
        if 'temperature' in results:
            np.save(self.working_dir / "temperature.npy", results['temperature'])
        
        if 'temperature_history' in results:
            np.save(self.working_dir / "temperature_history.npy", 
                   np.array(results['temperature_history']))
        
        # 保存到JSON
        summary = {k: v for k, v in results.items() 
                  if k not in ['temperature', 'temperature_history', 'final_temperature']}
        if 'final_temperature' in results:
            summary['final_max_temp'] = float(np.max(results['final_temperature']))
            summary['final_min_temp'] = float(np.min(results['final_temperature']))
        
        with open(self.working_dir / "thermal_results.json", 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _save_mechanics_results(self, results: Dict):
        """保存力学分析结果"""
        # 保存到NPY
        np.save(self.working_dir / "displacement.npy", results['displacement'])
        np.save(self.working_dir / "stress.npy", results['stress'])
        np.save(self.working_dir / "von_mises.npy", results['von_mises_stress'])
        
        # 导出到VTK
        self.mesh.export_to_vtk(
            str(self.working_dir / "mechanics_result.vtu"),
            displacement=results['displacement'][:, 0],  # x分量
            von_mises=results['von_mises_stress']
        )
        
        # 保存摘要
        summary = {
            'max_displacement': float(results['max_displacement']),
            'max_von_mises': float(results['max_von_mises'])
        }
        
        with open(self.working_dir / "mechanics_results.json", 'w') as f:
            json.dump(summary, f, indent=2)


# =============================================================================
# 命令行接口
# =============================================================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Continuum Mechanics Simulation Tool")
    parser.add_argument("--analysis", choices=["thermal", "mechanics", "coupled"],
                       default="thermal")
    parser.add_argument("--nx", type=int, default=50)
    parser.add_argument("--ny", type=int, default=50)
    parser.add_argument("--lx", type=float, default=100.0)
    parser.add_argument("--ly", type=float, default=100.0)
    parser.add_argument("--youngs-modulus", type=float, default=100.0)
    parser.add_argument("--poisson-ratio", type=float, default=0.3)
    parser.add_argument("--thermal-conductivity", type=float, default=100.0)
    parser.add_argument("--output-dir", default="./continuum_output")
    
    args = parser.parse_args()
    
    # 创建工作流
    workflow = ContinuumWorkflow(working_dir=args.output_dir)
    
    # 设置材料
    elastic_props = {
        'E': args.youngs_modulus,
        'nu': args.poisson_ratio
    }
    thermal_props = {
        'thermal_conductivity': args.thermal_conductivity
    }
    workflow.setup_material(elastic_props=elastic_props, thermal_props=thermal_props)
    
    # 生成网格
    fem_config = FEMConfig(
        dimensions=2,
        lx=args.lx, ly=args.ly,
        nx=args.nx, ny=args.ny
    )
    workflow.generate_mesh(fem_config)
    
    # 运行分析
    if args.analysis == "thermal":
        thermal_config = ThermalConfig(
            analysis_type="steady",
            bc_temperature=[
                {'boundary': 'left', 'value': 400.0},
                {'boundary': 'right', 'value': 300.0}
            ]
        )
        results = workflow.run_thermal_analysis(thermal_config)
        print(f"Max temperature: {results['max_temp']:.2f} K")
        
    elif args.analysis == "mechanics":
        mechanics_config = MechanicsConfig(
            bc_displacement=[
                {'boundary': 'left', 'value': 0.0, 'direction': 0},
                {'boundary': 'bottom', 'value': 0.0, 'direction': 1}
            ],
            bc_traction=[
                {'boundary': 'right', 'traction': [100.0, 0.0]}
            ]
        )
        results = workflow.run_mechanics_analysis(mechanics_config)
        print(f"Max displacement: {results['max_displacement']:.4f}")
        print(f"Max Von Mises stress: {results['max_von_mises']:.2f} GPa")
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
