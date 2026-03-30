"""
Mechano-Chemical Coupling Model
===============================
力学-化学耦合模型

描述应力场对扩散和相变的影响。
适用于薄膜生长、应力诱导相变、裂纹扩展等。

控制方程:
1. 成分场演化 (含应力耦合)
   ∂c/∂t = ∇·(M∇μ)
   μ = μ_chem + μ_elastic
   μ_elastic = -Ωσ_h (Ω: 偏摩尔体积, σ_h: 静水应力)

2. 力学平衡
   ∇·σ = 0
   σ = C:(ε - ε^0)
   ε^0 = η(c - c_0)I (本征应变)

其中:
    σ: 应力张量
    ε: 总应变
    ε^0: 本征应变 (由成分变化引起)
    C: 弹性刚度张量
    η: 晶格膨胀系数
"""

import numpy as np
from typing import Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
import logging

from .cahn_hilliard import CahnHilliardSolver, CahnHilliardConfig

logger = logging.getLogger(__name__)


@dataclass
class MechanoChemicalConfig(CahnHilliardConfig):
    """
    力学-化学耦合配置
    
    Attributes:
        # 力学参数
        E: 杨氏模量 (GPa)
        nu: 泊松比
        
        # 化学-力学耦合参数
        Omega: 偏摩尔体积 (m³/mol)
        eta_0: 晶格膨胀系数
        c_0: 参考浓度
        
        # 初始应力
        initial_stress: 初始应力张量 (GPa)
        
        # 外部载荷
        external_load: 外部机械载荷 (GPa)
        
        # 求解选项
        solve_elasticity: 是否求解弹性问题
        plane_stress: 平面应力近似 (2D)
    """
    # 弹性参数
    E: float = 100.0  # GPa
    nu: float = 0.25  # 泊松比
    
    # 耦合参数
    Omega: float = 1e-5  # m³/mol
    eta_0: float = 0.1  # 晶格膨胀系数
    c_0: float = 0.5  # 参考浓度 (归一化)
    
    # 初始/边界条件
    initial_stress: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # σ_xx, σ_yy, σ_zz
    external_load: float = 0.0  # GPa
    
    # 求解选项
    solve_elasticity: bool = True
    plane_stress: bool = True  # 2D时使用平面应力
    elastic_solver: str = "fourier"  # fourier, finite_difference
    
    def __post_init__(self):
        super().__post_init__()
        # 计算拉梅常数
        self.lambda_lame = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))  # GPa
        self.mu_lame = self.E / (2 * (1 + self.nu))  # GPa (剪切模量)
        
        # 弹性刚度矩阵分量 (各向同性)
        self.C11 = self.lambda_lame + 2 * self.mu_lame
        self.C12 = self.lambda_lame


class MechanoChemicalSolver(CahnHilliardSolver):
    """
    力学-化学耦合求解器
    
    耦合Cahn-Hilliard方程和弹性力学方程，
    描述应力对扩散和相变的影响。
    """
    
    def __init__(self, config: Optional[MechanoChemicalConfig] = None):
        """
        初始化力学-化学耦合求解器
        
        Args:
            config: 力学-化学配置
        """
        self.config = config or MechanoChemicalConfig()
        super().__init__(self.config)
        
        # 力学场变量
        self.stress = {}  # 应力分量
        self.strain = {}  # 应变分量
        self.displacement = {}  # 位移场
        
        # 本征应变
        self.eigenstrain = None
        
        # 弹性化学势贡献
        mu_elastic = 0.0
        
        logger.info(f"Mechano-chemical solver initialized")
        logger.info(f"E={self.config.E}GPa, ν={self.config.nu}")
        logger.info(f"Ω={self.config.Omega}, η_0={self.config.eta_0}")
    
    def initialize_fields(self, c0: Optional[np.ndarray] = None,
                         u0: Optional[Dict[str, np.ndarray]] = None,
                         seed: Optional[int] = None):
        """
        初始化力学-化学场
        
        Args:
            c0: 初始浓度场
            u0: 初始位移场 {'ux': ..., 'uy': ...}
            seed: 随机种子
        """
        # 初始化浓度场 (调用父类)
        if c0 is None:
            super().initialize_fields(seed=seed)
        else:
            self.c = c0.copy()
            self.fields['c'] = self.c
            self._update_chemical_potential()
        
        # 计算本征应变
        self._compute_eigenstrain()
        
        # 初始化位移场
        shape = self.c.shape
        if u0 is not None:
            self.displacement['ux'] = u0.get('ux', np.zeros(shape))
            self.displacement['uy'] = u0.get('uy', np.zeros(shape))
            if self.ndim == 3:
                self.displacement['uz'] = u0.get('uz', np.zeros(shape))
        else:
            self.displacement['ux'] = np.zeros(shape)
            self.displacement['uy'] = np.zeros(shape)
            if self.ndim == 3:
                self.displacement['uz'] = np.zeros(shape)
        
        # 计算初始应力和应变
        if self.config.solve_elasticity:
            self._solve_elasticity()
        else:
            self._init_constant_stress()
        
        self.fields.update(self.stress)
        
        logger.info(f"Mechano-chemical fields initialized")
    
    def _compute_eigenstrain(self):
        """计算本征应变"""
        # ε^0 = η_0 * (c - c_0) * I (各向同性膨胀)
        self.eigenstrain = self.config.eta_0 * (self.c - self.config.c_0)
    
    def _init_constant_stress(self):
        """初始化常应力状态"""
        shape = self.c.shape
        sigma_xx, sigma_yy, sigma_zz = self.config.initial_stress
        
        self.stress['sigma_xx'] = np.full(shape, sigma_xx)
        self.stress['sigma_yy'] = np.full(shape, sigma_yy)
        self.stress['sigma_xy'] = np.zeros(shape)
        
        if self.ndim == 3:
            self.stress['sigma_zz'] = np.full(shape, sigma_zz)
            self.stress['sigma_xz'] = np.zeros(shape)
            self.stress['sigma_yz'] = np.zeros(shape)
    
    def _solve_elasticity(self):
        """
        求解弹性力学问题
        
        使用傅里叶谱方法求解各向同性弹性问题
        """
        if self.config.elastic_solver == "fourier":
            self._solve_elasticity_fourier()
        else:
            self._solve_elasticity_fd()
    
    def _solve_elasticity_fourier(self):
        """
        使用傅里叶方法求解弹性问题
        
        基于Eshelby夹杂理论的谱方法实现
        """
        # 本征应变 (变换到傅里叶空间)
        eigen_fft = np.fft.fftn(self.eigenstrain)
        
        # 波矢
        kx = 2 * np.pi * np.fft.fftfreq(self.config.nx, self.config.dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.config.ny, self.config.dy)
        
        if self.ndim == 2:
            kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
            k_squared = kx_grid**2 + ky_grid**2
            k_squared[0, 0] = 1e-10  # 避免除零
            
            # 各向同性弹性问题的Green函数
            # 应力与傅里叶空间的本征应变关系
            C11 = self.config.C11
            C12 = self.config.C12
            
            # 计算应力 (简化公式)
            # σ_ij = C_ijkl * (ε_kl - ε^0_kl)
            # 在傅里叶空间: σ̃_ij = ...
            
            # 对于平面应力/平面应变，公式有所不同
            if self.config.plane_stress:
                factor = self.config.E / (1 - self.config.nu**2)
            else:
                factor = self.config.E / ((1 + self.config.nu) * (1 - 2 * self.config.nu))
            
            # 简化：使用局部近似计算应力
            # 实际实现需要更复杂的谱方法
            trace_eigen = eigen_fft  # 各向同性膨胀的本征应变迹
            
            # 反变换得到应力
            sigma_h = -factor * np.fft.ifftn(trace_eigen).real * self.eigenstrain
            
        else:
            # 3D情况
            kz = 2 * np.pi * np.fft.fftfreq(self.config.nz, self.config.dz)
            kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing='ij')
            k_squared = kx_grid**2 + ky_grid**2 + kz_grid**2
            k_squared[0, 0, 0] = 1e-10
            
            # 3D弹性解
            trace_eigen = eigen_fft
            sigma_h = -self.config.E / (1 - 2 * self.config.nu) * \
                     np.fft.ifftn(trace_eigen).real * self.eigenstrain
        
        # 存储应力分量 (简化：各向同性)
        shape = self.c.shape
        self.stress['sigma_xx'] = sigma_h
        self.stress['sigma_yy'] = sigma_h
        self.stress['sigma_xy'] = np.zeros(shape)
        
        if self.ndim == 3:
            self.stress['sigma_zz'] = sigma_h
            self.stress['sigma_xz'] = np.zeros(shape)
            self.stress['sigma_yz'] = np.zeros(shape)
    
    def _solve_elasticity_fd(self):
        """使用有限差分法求解弹性问题"""
        # 简化实现：使用迭代法
        # 实际实现需要更复杂的有限差分格式
        self._solve_elasticity_fourier()
    
    def _compute_hydrostatic_stress(self) -> np.ndarray:
        """
        计算静水应力
        
        σ_h = (σ_xx + σ_yy + σ_zz) / 3
        
        Returns:
            sigma_h: 静水应力
        """
        sigma_h = (self.stress['sigma_xx'] + self.stress['sigma_yy']) / 2
        
        if self.ndim == 3 and 'sigma_zz' in self.stress:
            sigma_h = (self.stress['sigma_xx'] + self.stress['sigma_yy'] + 
                      self.stress['sigma_zz']) / 3
        
        return sigma_h
    
    def _compute_elastic_chemical_potential(self) -> np.ndarray:
        """
        计算弹性化学势贡献
        
        μ_elastic = -Ω * σ_h
        
        Returns:
            mu_elastic: 弹性化学势
        """
        sigma_h = self._compute_hydrostatic_stress()
        
        # 转换为无量纲单位
        # Ω (m³/mol) * σ_h (GPa) / (RT) -> 无量纲
        RT = 8.314 * 298.15  # J/mol at 25°C
        Omega_m3 = self.config.Omega
        sigma_h_Pa = sigma_h * 1e9  # GPa -> Pa
        
        mu_elastic = -Omega_m3 * sigma_h_Pa / RT
        
        return mu_elastic
    
    def compute_chemical_potential(self, field_name: str = 'c') -> np.ndarray:
        """
        计算化学势 (含力学贡献)
        
        μ = μ_chem + μ_elastic
        
        Args:
            field_name: 场名称
            
        Returns:
            mu: 总化学势
        """
        # 化学部分
        mu_chem = super().compute_chemical_potential(field_name)
        
        # 弹性部分
        mu_elastic = self._compute_elastic_chemical_potential()
        
        # 总化学势
        mu_total = mu_chem + mu_elastic
        
        return mu_total
    
    def compute_energy(self) -> float:
        """
        计算总能量 (含弹性贡献)
        
        Returns:
            energy: 总自由能
        """
        # 化学部分
        E_chem = super().compute_energy()
        
        # 弹性部分
        # E_elastic = 0.5 * ∫ σ:ε dV
        sigma_xx = self.stress['sigma_xx']
        sigma_yy = self.stress['sigma_yy']
        sigma_xy = self.stress['sigma_xy']
        
        # 简化的弹性能量计算
        strain_xx = self.eigenstrain
        strain_yy = self.eigenstrain
        
        E_elastic_density = 0.5 * (sigma_xx * strain_xx + sigma_yy * strain_yy + 
                                    2 * sigma_xy * 0)  # 剪切应变简化
        
        E_elastic = np.sum(E_elastic_density) * self.config.dx * self.config.dy
        
        return E_chem + E_elastic
    
    def evolve_step(self) -> Dict:
        """
        执行力学-化学耦合演化步骤
        
        Returns:
            info: 演化信息
        """
        # 1. 更新本征应变
        self._compute_eigenstrain()
        
        # 2. 求解弹性问题
        if self.config.solve_elasticity:
            self._solve_elasticity()
        
        # 3. 更新化学势 (含弹性贡献)
        self._update_chemical_potential()
        
        # 4. 演化浓度场 (调用父类演化)
        info = super().evolve_step()
        
        # 添加力学信息
        sigma_h = self._compute_hydrostatic_stress()
        info['sigma_h_mean'] = float(sigma_h.mean())
        info['sigma_h_max'] = float(np.abs(sigma_h).max())
        
        return info
    
    def get_stress_distribution(self) -> Dict[str, np.ndarray]:
        """
        获取应力分布
        
        Returns:
            stress: 应力分量字典
        """
        return {k: v.copy() for k, v in self.stress.items()}
    
    def get_max_principal_stress(self) -> np.ndarray:
        """
        计算最大主应力
        
        Returns:
            sigma_1: 最大主应力场
        """
        sigma_xx = self.stress['sigma_xx']
        sigma_yy = self.stress['sigma_yy']
        sigma_xy = self.stress['sigma_xy']
        
        # 主应力公式
        sigma_avg = (sigma_xx + sigma_yy) / 2
        radius = np.sqrt(((sigma_xx - sigma_yy) / 2)**2 + sigma_xy**2)
        
        sigma_1 = sigma_avg + radius
        
        return sigma_1
    
    def check_crack_criterion(self, fracture_toughness: float = 1.0) -> np.ndarray:
        """
        检查裂纹萌生准则
        
        Args:
            fracture_toughness: 断裂韧性 (MPa·m^0.5)
            
        Returns:
            crack_risk: 裂纹风险场 (True表示可能萌生裂纹)
        """
        sigma_1 = self.get_max_principal_stress()
        
        # 简化的裂纹准则: σ_1 > K_IC / sqrt(π*a)
        # 假设特征裂纹尺寸a
        a = self.config.dx * 1e-9  # nm -> m
        threshold = fracture_toughness / np.sqrt(np.pi * a) * 1e-6  # 转换为GPa
        
        crack_risk = sigma_1 > threshold
        
        return crack_risk
