"""
Electrochemical Phase Field Model
=================================
电化学相场模型

耦合离子传输、电化学反应和相变的综合模型。
适用于锂离子电池电极、固态电解质等体系。

控制方程:
1. 成分场演化 (Cahn-Hilliard)
   ∂c/∂t = ∇·(M∇μ) + R
   
2. 电荷守恒
   ∇·(σ∇φ) = -ρ_e
   
3. 反应动力学 (Butler-Volmer)
   j = j_0 [exp(α_aη_f/RT) - exp(-α_cη_f/RT)]

其中:
    c: Li离子浓度
    φ: 电势
    μ: 电化学势 μ = μ_0 + RTln(a) + zFφ
    R: 反应源项
    σ: 离子/电子电导率
"""

import numpy as np
from typing import Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
import logging

from .cahn_hilliard import CahnHilliardSolver, CahnHilliardConfig

logger = logging.getLogger(__name__)


@dataclass
class ElectrochemicalConfig(CahnHilliardConfig):
    """
    电化学相场配置
    
    Attributes:
        # 电化学参数
        max_concentration: 最大Li浓度 (mol/m³)
        temperature: 温度 (K)
        
        # 传输参数
        D_Li: Li扩散系数 (m²/s)
        sigma_ionic: 离子电导率 (S/m)
        sigma_electronic: 电子电导率 (S/m)
        
        # 反应参数
        exchange_current_density: 交换电流密度 (A/m²)
        alpha_a: 阳极传递系数
        alpha_c: 阴极传递系数
        
        # 热力学参数
        E0: 标准电极电势 (V)
        Omega: 偏摩尔体积 (m³/mol)
        
        # 应力效应
        include_mechanical: 是否包含力学效应
        Young_modulus: 杨氏模量 (GPa)
        Poisson_ratio: 泊松比
    """
    # 电化学参数
    max_concentration: float = 2.3e4  # mol/m³ (对应LiCoO2)
    temperature: float = 298.15  # K (25°C)
    
    # 传输参数
    D_Li: float = 1e-14  # m²/s
    sigma_ionic: float = 1e-4  # S/m
    sigma_electronic: float = 1e2  # S/m
    
    # 反应参数 (Butler-Volmer)
    exchange_current_density: float = 10.0  # A/m²
    alpha_a: float = 0.5  # 阳极传递系数
    alpha_c: float = 0.5  # 阴极传递系数
    
    # 热力学参数
    E0: float = 3.9  # V vs Li/Li+
    Omega: float = 2e-6  # m³/mol
    
    # 力学参数
    include_mechanical: bool = False
    Young_modulus: float = 100.0  # GPa
    Poisson_ratio: float = 0.25
    
    # 边界条件
    applied_current: float = 0.0  # A/m² (0表示开路)
    applied_voltage: Optional[float] = None  # V
    
    # 数值参数
    solve_poisson: bool = True  # 是否求解泊松方程
    coupling_strength: float = 1.0  # 电化学-相场耦合强度
    
    def __post_init__(self):
        super().__post_init__()
        # 物理常数
        self.R = 8.314  # J/(mol·K)
        self.F = 96485.0  # C/mol
        self.kB = 1.380649e-23  # J/K


class ElectrochemicalPhaseField(CahnHilliardSolver):
    """
    电化学相场模型
    
    耦合相分离和电化学反应的综合模型。
    适用于锂离子电池电极材料、固态电解质等。
    """
    
    def __init__(self, config: Optional[ElectrochemicalConfig] = None):
        """
        初始化电化学相场模型
        
        Args:
            config: 电化学配置
        """
        self.config = config or ElectrochemicalConfig()
        
        # 先调用父类初始化
        super().__init__(self.config)
        
        # 电化学场变量
        self.phi = None  # 电势场 (V)
        self.j_Li = None  # Li离子通量
        self.i_total = None  # 总电流密度
        
        # 反应源项
        self.reaction_rate = None
        
        # 应力场 (如果启用)
        self.stress = None
        self.strain = None
        
        logger.info(f"Electrochemical phase field initialized")
        logger.info(f"T={self.config.temperature}K, E0={self.config.E0}V")
    
    def initialize_fields(self, c0: Optional[np.ndarray] = None,
                         phi0: Optional[np.ndarray] = None,
                         seed: Optional[int] = None):
        """
        初始化电化学场
        
        Args:
            c0: 初始浓度场 (归一化，0-1)
            phi0: 初始电势场
            seed: 随机种子
        """
        # 初始化浓度场 (调用父类)
        if c0 is None:
            super().initialize_fields(seed=seed)
        else:
            self.c = c0.copy()
            self.fields['c'] = self.c
            self._update_chemical_potential()
        
        # 初始化电势场
        if phi0 is not None:
            self.phi = phi0.copy()
        else:
            self._initialize_potential()
        
        self.fields['phi'] = self.phi
        
        # 初始化其他场
        shape = self.c.shape
        self.j_Li = np.zeros(shape + (self.ndim,))
        self.reaction_rate = np.zeros(shape)
        
        logger.info(f"Electrochemical fields initialized")
        logger.info(f"c range: [{self.c.min():.4f}, {self.c.max():.4f}]")
        logger.info(f"φ range: [{self.phi.min():.4f}, {self.phi.max():.4f}]")
    
    def _initialize_potential(self):
        """初始化电势场"""
        shape = (self.config.nx, self.config.ny) if self.ndim == 2 else \
                (self.config.nx, self.config.ny, self.config.nz)
        
        # 开路电势
        self.phi = self._open_circuit_potential(self.c)
        
        # 添加边界条件
        if self.config.applied_voltage is not None:
            # 施加电压边界条件
            self.phi[0, :] = self.config.applied_voltage
            self.phi[-1, :] = 0.0
    
    def _open_circuit_potential(self, c: np.ndarray) -> np.ndarray:
        """
        计算开路电势 (OCV)
        
        基于热力学关系: E = E0 - (∂f/∂c) / F
        
        Args:
            c: 归一化浓度
            
        Returns:
            phi: 开路电势
        """
        # 简化的OCV模型 (基于成分依赖)
        # 实际应从DFT计算获得
        df_dc = self._compute_bulk_chemical_potential(c)
        
        # 转换为实际单位
        mu_Li = df_dc * self.config.max_concentration  # J/mol
        
        # OCV = E0 - μ_Li / F
        phi = self.config.E0 - mu_Li / self.config.F
        
        return phi
    
    def _exchange_current_density(self, c: np.ndarray) -> np.ndarray:
        """
        计算局部交换电流密度
        
        j0 = j0_0 * sqrt(c * (1-c))
        
        Args:
            c: 归一化浓度
            
        Returns:
            j0: 交换电流密度
        """
        j0_0 = self.config.exchange_current_density
        # 避免sqrt(0)
        c_eff = np.clip(c * (1 - c), 1e-10, 0.25)
        j0 = j0_0 * np.sqrt(c_eff)
        return j0
    
    def _butler_volmer(self, eta: np.ndarray, j0: np.ndarray) -> np.ndarray:
        """
        Butler-Volmer方程
        
        j = j0 [exp(α_a*F*η/RT) - exp(-α_c*F*η/RT)]
        
        Args:
            eta: 过电位 (V)
            j0: 交换电流密度 (A/m²)
            
        Returns:
            j: 反应电流密度 (A/m²)
        """
        RT_F = self.config.R * self.config.temperature / self.config.F
        
        alpha_a = self.config.alpha_a
        alpha_c = self.config.alpha_c
        
        # 避免指数溢出
        exp_arg_a = np.clip(alpha_a * eta / RT_F, -50, 50)
        exp_arg_c = np.clip(-alpha_c * eta / RT_F, -50, 50)
        
        j = j0 * (np.exp(exp_arg_a) - np.exp(exp_arg_c))
        
        return j
    
    def _compute_electrochemical_potential(self, c: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        计算电化学势
        
        μ = μ_chem + zFφ = ∂f/∂c + F*φ (对于Li+)
        
        Args:
            c: 归一化浓度
            phi: 电势
            
        Returns:
            mu: 电化学势
        """
        # 化学势部分
        mu_chem = self._compute_bulk_chemical_potential(c)
        
        # 电化学势 (归一化)
        mu = mu_chem + phi / self.config.E0
        
        return mu
    
    def _solve_poisson_equation(self):
        """
        求解泊松方程 (电势分布)
        
        ∇·(σ∇φ) = -ρ_e
        
        简化为拉普拉斯方程 (假设电中性近似)
        ∇²φ = 0
        """
        if not self.config.solve_poisson:
            return
        
        # 使用迭代法求解
        # 简化的SOR方法
        omega = 1.5  # 松弛因子
        n_iter = 50
        
        phi_old = self.phi.copy()
        
        for _ in range(n_iter):
            phi_new = phi_old.copy()
            
            if self.ndim == 2:
                # 内部点
                phi_new[1:-1, 1:-1] = 0.25 * (
                    phi_old[2:, 1:-1] + phi_old[:-2, 1:-1] +
                    phi_old[1:-1, 2:] + phi_old[1:-1, :-2]
                )
            else:
                phi_new[1:-1, 1:-1, 1:-1] = (1/6) * (
                    phi_old[2:, 1:-1, 1:-1] + phi_old[:-2, 1:-1, 1:-1] +
                    phi_old[1:-1, 2:, 1:-1] + phi_old[1:-1, :-2, 1:-1] +
                    phi_old[1:-1, 1:-1, 2:] + phi_old[1:-1, 1:-1, :-2]
                )
            
            # SOR更新
            phi_old = omega * phi_new + (1 - omega) * phi_old
            
            # 应用边界条件
            if self.config.applied_voltage is not None:
                phi_old[0, :] = self.config.applied_voltage
                phi_old[-1, :] = 0.0
        
        self.phi = phi_old
        self.fields['phi'] = self.phi
    
    def _compute_reaction_rate(self) -> np.ndarray:
        """
        计算电化学反应速率
        
        R = j / (F * c_max)  (归一化单位)
        
        Returns:
            R: 反应源项
        """
        # 计算过电位
        phi_eq = self._open_circuit_potential(self.c)
        eta_over = self.phi - phi_eq  # 过电位
        
        # 交换电流密度
        j0 = self._exchange_current_density(self.c)
        
        # Butler-Volmer电流
        j = self._butler_volmer(eta_over, j0)
        
        # 转换为浓度变化率 (归一化)
        R = j / (self.config.F * self.config.max_concentration)
        
        # 转换为相场时间单位
        R = R * 1e9  # 缩放因子
        
        return R
    
    def _compute_li_flux(self) -> np.ndarray:
        """
        计算Li离子通量
        
        J = -M∇μ = -D∇c - (D*zF/RT)*c*∇φ
        
        Returns:
            J: 通量向量 (nx, ny, [nz], ndim)
        """
        # 浓度梯度
        grad_c = self.compute_gradient(self.c)
        
        # 电势梯度
        grad_phi = self.compute_gradient(self.phi)
        
        # 有效扩散系数 (依赖于浓度)
        D_eff = self.config.D_Li * self.c * (1 - self.c) * 1e18  # m²/s -> nm²/s
        
        # 化学扩散项
        J_diff = [-D_eff * g for g in grad_c]
        
        # 电迁移项
        RT_F = self.config.R * self.config.temperature / self.config.F
        migration_factor = D_eff / RT_F
        J_migr = [-migration_factor * self.c * g for g in grad_phi]
        
        # 总通量
        J = np.stack([jd + jm for jd, jm in zip(J_diff, J_migr)], axis=-1)
        
        return J
    
    def evolve_step(self) -> Dict:
        """
        执行电化学-相场耦合演化步骤
        
        Returns:
            info: 演化信息
        """
        # 1. 更新电势场
        self._solve_poisson_equation()
        
        # 2. 计算Li通量
        self.j_Li = self._compute_li_flux()
        
        # 3. 计算反应速率
        self.reaction_rate = self._compute_reaction_rate()
        
        # 4. 更新化学势
        self._update_chemical_potential()
        
        # 5. 演化浓度场 (Cahn-Hilliard + 反应源项)
        # ∂c/∂t = ∇·(M∇μ) + R
        laplacian_mu = self.compute_laplacian(self.mu)
        
        # 相场演化
        dc_pf = self.config.dt * self.config.M * laplacian_mu
        
        # 反应贡献
        dc_reaction = self.config.dt * self.reaction_rate * self.config.coupling_strength
        
        c_new = self.c + dc_pf + dc_reaction
        
        # 应用边界条件
        c_new = self.bc_handler.apply(c_new)
        
        # 确保物理约束
        c_new = np.clip(c_new, 0.001, 0.999)
        
        # 计算变化
        dc_max = np.abs(c_new - self.c).max()
        
        # 更新场
        self.c = c_new
        self.fields['c'] = self.c
        
        # 收敛判断
        converged = dc_max < self.config.tolerance
        
        info = {
            'step': self.step,
            'time': self.time,
            'dc_max': dc_max,
            'c_mean': self.c.mean(),
            'c_std': self.c.std(),
            'phi_mean': self.phi.mean(),
            'reaction_rate_max': np.abs(self.reaction_rate).max(),
            'energy': self.compute_energy(),
            'converged': converged
        }
        
        return info
    
    def compute_voltage(self) -> float:
        """
        计算电池电压
        
        Returns:
            voltage: 电池电压 (V)
        """
        # 平均电势差
        voltage = self.phi.mean() - self.phi.min()
        return voltage
    
    def compute_capacity(self) -> float:
        """
        计算当前容量
        
        Returns:
            capacity: 归一化容量 (0-1)
        """
        return self.c.mean()
    
    def get_intercalation_current(self) -> float:
        """
        计算总嵌锂电流
        
        Returns:
            current: 总电流 (A/m²)
        """
        return np.abs(self.reaction_rate).mean() * self.config.F * self.config.max_concentration * 1e-9
