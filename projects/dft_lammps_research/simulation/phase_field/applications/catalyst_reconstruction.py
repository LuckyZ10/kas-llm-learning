"""
Catalyst Reconstruction Simulator
=================================
催化剂表面重构模拟器

模拟催化剂在反应条件下的表面重构过程。
适用于电催化、热催化等体系。

物理模型:
- 表面相场模型 (2D)
- 吸附-脱附动力学
- 电化学势影响
- 应变效应
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
import logging

from ..core.allen_cahn import AllenCahnSolver, AllenCahnConfig
from ..core.electrochemical import ElectrochemicalPhaseField

logger = logging.getLogger(__name__)


@dataclass
class CatalystConfig(AllenCahnConfig):
    """
    催化剂表面重构配置
    
    Attributes:
        # 催化剂表面
        surface_miller: 表面晶面指数
        surface_size: 表面尺寸 (nm)
        
        # 吸附物种
        adsorbates: 吸附物种列表
        coverage_initial: 初始覆盖度
        
        # 反应条件
        temperature: 温度 (K)
        pressure: 气体压力 (Pa)
        potential: 电极电势 (V vs RHE)
        
        # 吸附能 (来自DFT)
        adsorption_energies: 各物种吸附能 (eV)
        interaction_energies: 吸附质相互作用能
        
        # 动力学参数
        adsorption_rate: 吸附速率
        desorption_rate: 脱附速率
        diffusion_rate: 表面扩散速率
        
        # 重构参数
        reconstruction_barrier: 重构势垒 (eV)
        strain_coupling: 应变耦合系数
    """
    # 表面参数
    surface_miller: Tuple[int, int, int] = (1, 1, 1)
    surface_size: Tuple[float, float] = (10.0, 10.0)  # nm
    
    # 吸附物种
    adsorbates: List[str] = field(default_factory=lambda: ['empty', 'O', 'OH', 'H'])
    coverage_initial: Dict[str, float] = field(default_factory=lambda: {
        'empty': 0.9, 'O': 0.05, 'OH': 0.03, 'H': 0.02
    })
    
    # 反应条件
    temperature: float = 298.15  # K
    pressure: Dict[str, float] = field(default_factory=lambda: {'O2': 1e5})  # Pa
    potential: float = 0.0  # V vs RHE
    pH: float = 7.0
    
    # 能量参数 (来自DFT)
    adsorption_energies: Dict[str, float] = field(default_factory=lambda: {
        'O': -1.5, 'OH': -1.0, 'H': -0.5
    })
    interaction_energies: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'O': {'O': 0.3, 'OH': 0.2},
        'OH': {'O': 0.2, 'OH': 0.1}
    })
    
    # 动力学参数
    adsorption_rate: float = 1.0  # 归一化
    desorption_rate: float = 0.1
    diffusion_rate: float = 0.5
    
    # 重构参数
    reconstruction_barrier: float = 0.5  # eV
    strain_coupling: float = 0.1
    
    # 表面结构
    include_step_sites: bool = False
    step_density: float = 0.0  # 台阶密度


class CatalystReconstructor(AllenCahnSolver):
    """
    催化剂表面重构模拟器
    
    模拟催化剂表面在电化学条件下的重构过程，
    包括吸附诱导重构和电化学驱动重构。
    """
    
    def __init__(self, config: Optional[CatalystConfig] = None):
        """
        初始化催化剂模拟器
        
        Args:
            config: 催化剂配置
        """
        self.config = config or CatalystConfig()
        super().__init__(self.config)
        
        # 吸附质覆盖度场
        self.coverage = {}  # 各物种的覆盖度场
        
        # 表面结构场
        self.surface_height = None  # 表面高度场 (重构)
        
        # 吸附诱导应变
        self.adsorption_strain = None
        
        # 反应速率
        self.reaction_rates = {}
        
        logger.info(f"Catalyst reconstructor initialized")
        logger.info(f"Surface: {self.config.surface_miller}")
        logger.info(f"Adsorbates: {self.config.adsorbates}")
    
    def initialize_fields(self, custom_coverage: Optional[Dict] = None,
                         surface_roughness: float = 0.0,
                         seed: Optional[int] = None):
        """
        初始化表面场
        
        Args:
            custom_coverage: 自定义覆盖度
            surface_roughness: 表面粗糙度
            seed: 随机种子
        """
        if seed is not None:
            np.random.seed(seed)
        
        shape = (self.config.nx, self.config.ny)
        
        # 初始化各吸附物种的覆盖度场
        coverage_init = custom_coverage or self.config.coverage_initial
        
        for species in self.config.adsorbates:
            c_init = coverage_init.get(species, 0.0)
            # 添加空间涨落
            noise = 0.01 * (np.random.random(shape) - 0.5)
            self.coverage[species] = np.clip(c_init + noise, 0, 1)
            self.fields[f'coverage_{species}'] = self.coverage[species]
        
        # 归一化覆盖度
        self._normalize_coverage()
        
        # 初始化表面高度场
        self.surface_height = np.zeros(shape)
        if surface_roughness > 0:
            self._add_surface_roughness(surface_roughness)
        
        self.fields['surface_height'] = self.surface_height
        
        # 初始化吸附诱导应变
        self.adsorption_strain = np.zeros(shape)
        
        # 初始化表面结构序参量
        self._init_surface_structure()
        
        logger.info(f"Initialized surface with {len(self.config.adsorbates)} species")
    
    def _normalize_coverage(self):
        """归一化覆盖度 (总和为1)"""
        total = sum(self.coverage.values())
        total = np.maximum(total, 1e-10)
        
        for species in self.coverage:
            self.coverage[species] /= total
            self.coverage[species] = np.clip(self.coverage[species], 0, 1)
            self.fields[f'coverage_{species}'] = self.coverage[species]
    
    def _add_surface_roughness(self, roughness: float):
        """添加表面粗糙度"""
        from scipy.ndimage import gaussian_filter
        
        noise = np.random.randn(self.config.nx, self.config.ny)
        self.surface_height = gaussian_filter(noise, sigma=2) * roughness
    
    def _init_surface_structure(self):
        """初始化表面结构序参量"""
        # 定义不同的表面结构 (如(1x1), (2x1)重构等)
        self.eta['bulk_structure'] = np.ones((self.config.nx, self.config.ny))
        self.eta['reconstructed'] = np.zeros((self.config.nx, self.config.ny))
        
        self.fields.update(self.eta)
    
    def _compute_adsorption_energy_local(self, species: str) -> np.ndarray:
        """
        计算局域吸附能
        
        包含吸附质相互作用和电化学贡献
        
        Args:
            species: 吸附物种
            
        Returns:
            E_ads: 局域吸附能场
        """
        # 基础吸附能
        E_base = self.config.adsorption_energies.get(species, 0.0)
        
        # 吸附质相互作用
        E_interaction = np.zeros((self.config.nx, self.config.ny))
        interactions = self.config.interaction_energies.get(species, {})
        
        for other_species, strength in interactions.items():
            if other_species in self.coverage:
                E_interaction += strength * self.coverage[other_species]
        
        # 电化学贡献 (对于电催化)
        # ΔE = -e * (φ - φ_PZC) * χ
        if species in ['O', 'OH'] and self.config.potential != 0:
            # 电化学修正 (简化模型)
            delta_E_electro = -0.5 * self.config.potential
            E_interaction += delta_E_electro
        
        E_ads = E_base + E_interaction
        
        return E_ads
    
    def _compute_reconstruction_driving_force(self) -> np.ndarray:
        """
        计算表面重构的驱动力
        
        由吸附诱导的应力驱动
        
        Returns:
            driving_force: 驱动力场
        """
        # 总覆盖度
        total_coverage = sum(self.coverage.values()) - self.coverage.get('empty', 0)
        
        # 吸附诱导应变
        strain = self.config.strain_coupling * total_coverage
        self.adsorption_strain = strain
        
        # 重构驱动力 ∝ 应变能
        driving_force = strain**2 - self.config.reconstruction_barrier
        
        return driving_force
    
    def _compute_adsorption_rate(self, species: str) -> np.ndarray:
        """
        计算吸附速率
        
        Args:
            species: 吸附物种
            
        Returns:
            rate: 吸附速率场
        """
        if species == 'empty':
            return np.zeros((self.config.nx, self.config.ny))
        
        # 可用位点
        empty_sites = self.coverage.get('empty', np.zeros((self.config.nx, self.config.ny)))
        
        # 碰撞频率 (简化)
        if species == 'O':
            gas_pressure = self.config.pressure.get('O2', 1e5)
            collision_freq = gas_pressure / 1e5  # 归一化
        elif species == 'H':
            gas_pressure = self.config.pressure.get('H2', 1e5)
            collision_freq = gas_pressure / 1e5
        else:
            collision_freq = 1.0
        
        # 吸附速率
        rate = self.config.adsorption_rate * collision_freq * empty_sites
        
        return rate
    
    def _compute_desorption_rate(self, species: str) -> np.ndarray:
        """
        计算脱附速率
        
        Args:
            species: 吸附物种
            
        Returns:
            rate: 脱附速率场
        """
        if species == 'empty':
            return np.zeros((self.config.nx, self.config.ny))
        
        # 吸附能
        E_ads = self._compute_adsorption_energy_local(species)
        
        # Arrhenius脱附速率
        kB_T = 8.617e-5 * self.config.temperature  # eV
        rate_constant = self.config.desorption_rate * np.exp(E_ads / kB_T)
        
        rate = rate_constant * self.coverage[species]
        
        return rate
    
    def _compute_surface_diffusion(self, species: str) -> np.ndarray:
        """
        计算表面扩散通量
        
        Args:
            species: 吸附物种
            
        Returns:
            diffusion: 扩散贡献
        """
        c = self.coverage[species]
        
        # 表面扩散 (Cahn-Hilliard类型)
        laplacian_c = self.compute_laplacian(c)
        
        D_surf = self.config.diffusion_rate
        diffusion_term = D_surf * laplacian_c
        
        return diffusion_term
    
    def evolve_step(self) -> Dict:
        """
        执行表面演化步骤
        
        Returns:
            info: 演化信息
        """
        max_dc = 0.0
        
        # 1. 演化覆盖度场
        for species in self.config.adsorbates:
            if species == 'empty':
                continue
            
            c_old = self.coverage[species].copy()
            
            # 吸附
            r_ads = self._compute_adsorption_rate(species)
            
            # 脱附
            r_des = self._compute_desorption_rate(species)
            
            # 扩散
            diffusion = self._compute_surface_diffusion(species)
            
            # 总变化
            dc_dt = r_ads - r_des + diffusion
            
            c_new = c_old + self.config.dt * dc_dt
            c_new = np.clip(c_new, 0, 1)
            
            self.coverage[species] = c_new
            self.fields[f'coverage_{species}'] = c_new
            
            max_dc = max(max_dc, np.abs(c_new - c_old).max())
        
        # 归一化覆盖度
        self._normalize_coverage()
        
        # 2. 演化表面结构 (重构)
        driving_force = self._compute_reconstruction_driving_force()
        
        eta_rec_old = self.eta['reconstructed'].copy()
        
        # Allen-Cahn类型演化
        dfd_eta = 2 * eta_rec_old * (1 - eta_rec_old) * (1 - 2*eta_rec_old)
        laplacian_eta = self.compute_laplacian(eta_rec_old)
        
        # 包含重构驱动力
        delta_F = dfd_eta - self.config.kappa * laplacian_eta - driving_force
        
        eta_rec_new = eta_rec_old - self.config.dt * self.config.L * delta_F
        eta_rec_new = np.clip(eta_rec_new, 0, 1)
        
        self.eta['reconstructed'] = eta_rec_new
        self.fields['reconstructed'] = eta_rec_new
        
        # 更新表面高度
        self.surface_height = eta_rec_new * 0.2  # 重构导致的高度变化
        self.fields['surface_height'] = self.surface_height
        
        # 计算反应活性
        activity = self._compute_catalytic_activity()
        
        converged = max_dc < self.config.tolerance
        
        info = {
            'step': self.step,
            'time': self.time,
            'max_coverage_change': max_dc,
            'reconstruction_fraction': float(eta_rec_new.mean()),
            'catalytic_activity': activity,
            'converged': converged
        }
        
        # 添加各物种平均覆盖度
        for species in self.config.adsorbates:
            info[f'coverage_{species}'] = float(self.coverage[species].mean())
        
        return info
    
    def _compute_catalytic_activity(self) -> float:
        """
        计算催化活性
        
        基于活性位点数量和吸附能
        
        Returns:
            activity: 归一化催化活性
        """
        # 简化模型：活性与O/OH覆盖度相关
        o_coverage = self.coverage.get('O', np.zeros((self.config.nx, self.config.ny)))
        oh_coverage = self.coverage.get('OH', np.zeros((self.config.nx, self.config.ny)))
        
        # 最优覆盖度假设为0.25
        optimal_coverage = 0.25
        
        # 活性 ∝ exp(-(c - c_opt)²)
        activity_o = np.exp(-(o_coverage.mean() - optimal_coverage)**2 / 0.1)
        activity_oh = np.exp(-(oh_coverage.mean() - optimal_coverage)**2 / 0.1)
        
        total_activity = 0.5 * (activity_o + activity_oh)
        
        return float(total_activity)
    
    def get_surface_structure_factor(self) -> float:
        """
        计算表面结构因子
        
        Returns:
            S: 结构因子 (指示有序程度)
        """
        # 重构区域的有序度
        reconstructed = self.eta['reconstructed']
        
        # FFT分析
        fft = np.fft.fft2(reconstructed)
        S = np.abs(fft)**2
        
        # 峰值强度
        peak_intensity = S.max()
        
        return float(peak_intensity / S.sum())
    
    def get_active_site_density(self) -> float:
        """
        计算活性位点密度
        
        Returns:
            density: 活性位点密度 (sites/nm²)
        """
        # 识别边界区域 (不同吸附物种交界)
        boundaries = np.zeros((self.config.nx, self.config.ny))
        
        species_list = [s for s in self.config.adsorbates if s != 'empty']
        
        for i in range(len(species_list)):
            for j in range(i+1, len(species_list)):
                s1 = species_list[i]
                s2 = species_list[j]
                
                # 找到界面
                interface = np.abs(self.coverage[s1] - self.coverage[s2]) > 0.5
                boundaries += interface
        
        # 活性位点密度
        area = self.config.nx * self.config.ny * self.config.dx**2  # nm²
        density = np.sum(boundaries > 0) / area
        
        return density
