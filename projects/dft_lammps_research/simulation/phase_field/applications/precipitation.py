"""
Precipitation Simulator
=======================
合金沉淀相演化模拟器

模拟合金中的沉淀相形成和粗化过程。
适用于铝合金、镍基超合金等。

物理模型:
- 多组元Cahn-Hilliard方程
- 弹性应力耦合
- 形核-生长-粗化全阶段
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
import logging

from ..core.cahn_hilliard import CahnHilliardSolver, CahnHilliardConfig
from ..core.allen_cahn import AllenCahnSolver

logger = logging.getLogger(__name__)


@dataclass
class PrecipConfig(CahnHilliardConfig):
    """
    沉淀相演化配置
    
    Attributes:
        # 合金体系
        n_components: 组元数 (主元素 + 合金元素)
        element_names: 元素名称列表
        nominal_composition: 名义成分
        
        # 沉淀相参数
        precipitate_composition: 沉淀相成分
        matrix_composition: 基体成分
        equilibrium_composition: 平衡成分
        
        # 形核参数
        nucleation_sites: 形核位置 ('random', 'dislocations', 'grain_boundaries')
        nucleation_density: 形核密度 (m^-3)
        critical_radius: 临界形核半径 (nm)
        
        # 弹性参数
        lattice_mismatch: 晶格错配度
        elastic_anisotropy: 弹性各向异性系数
        
        # 温度历史
        temperature_profile: 温度曲线
        aging_time: 时效时间 (hours)
    """
    # 合金组元
    n_components: int = 2
    element_names: List[str] = field(default_factory=lambda: ['Al', 'Cu'])
    nominal_composition: Dict[str, float] = field(default_factory=lambda: {'Al': 0.96, 'Cu': 0.04})
    
    # 沉淀相成分 (θ'相: Al2Cu)
    precipitate_composition: Dict[str, float] = field(default_factory=lambda: {'Al': 0.67, 'Cu': 0.33})
    matrix_composition: Dict[str, float] = field(default_factory=lambda: {'Al': 0.98, 'Cu': 0.02})
    equilibrium_composition: Dict[str, float] = field(default_factory=lambda: {
        'matrix': 0.02, 'precipitate': 0.33
    })
    
    # 形核参数
    nucleation_sites: str = "random"
    nucleation_density: float = 1e21  # m^-3
    critical_radius: float = 1.0  # nm
    
    # 弹性参数
    lattice_mismatch: float = 0.02  # 2%错配
    elastic_anisotropy: float = 1.0
    
    # 时效处理
    temperature: float = 473.15  # K (200°C)
    aging_time: float = 10.0  # hours
    
    # 析出动力学
    diffusivity_prefactor: float = 1e-5  # m²/s
    activation_energy: float = 1.0  # eV
    
    def __post_init__(self):
        super().__post_init__()
        
        # 根据温度计算扩散系数
        kB = 8.617e-5  # eV/K
        self.D = self.diffusivity_prefactor * np.exp(
            -self.activation_energy / (kB * self.temperature)
        )
        # 转换为相场单位
        self.M = self.D * 1e18  # m²/s -> nm²/s


class PrecipitationSimulator(CahnHilliardSolver):
    """
    沉淀相演化模拟器
    
    模拟合金时效过程中的沉淀相形核、生长和粗化。
    """
    
    def __init__(self, config: Optional[PrecipConfig] = None):
        """
        初始化沉淀模拟器
        
        Args:
            config: 沉淀相配置
        """
        self.config = config or PrecipConfig()
        super().__init__(self.config)
        
        # 多组元浓度场
        self.concentration = {}  # 各组元的浓度场
        
        # 序参量场 (区分基体和沉淀相)
        self.eta_precipitate = None
        
        # 形核追踪
        self.nuclei = []
        self.precipitates = []
        
        # 弹性场
        self.strain_field = None
        self.stress_field = None
        
        logger.info(f"Precipitation simulator initialized")
        logger.info(f"System: {'-'.join(self.config.element_names)}")
        logger.info(f"Aging at {self.config.temperature-273.15:.0f}°C")
    
    def initialize_fields(self, uniform_composition: bool = True,
                         seed: Optional[int] = None):
        """
        初始化浓度场
        
        Args:
            uniform_composition: 使用均匀初始成分
            seed: 随机种子
        """
        if seed is not None:
            np.random.seed(seed)
        
        shape = (self.config.nx, self.config.ny)
        
        # 初始化合金元素浓度场 (以Cu在Al-Cu合金为例)
        alloy_element = self.config.element_names[1] if len(self.config.element_names) > 1 else 'solute'
        c_nominal = self.config.nominal_composition.get(alloy_element, 0.04)
        
        if uniform_composition:
            # 均匀初始成分 + 热涨落
            noise = 0.005 * (np.random.random(shape) - 0.5)
            self.concentration[alloy_element] = c_nominal + noise
        else:
            # 非均匀初始条件
            self.concentration[alloy_element] = self._create_clustered_initial(c_nominal)
        
        # 主元素浓度 (由归一化条件确定)
        matrix_element = self.config.element_names[0]
        self.concentration[matrix_element] = 1.0 - self.concentration[alloy_element]
        
        # 设置父类的c场 (归一化的合金元素浓度)
        self.c = self.concentration[alloy_element].copy()
        self.fields['c'] = self.c
        
        # 初始化沉淀相序参量
        self.eta_precipitate = np.zeros(shape)
        
        # 根据浓度场识别初始沉淀相
        c_eq_matrix = self.config.equilibrium_composition['matrix']
        c_eq_precip = self.config.equilibrium_composition['precipitate']
        
        # 高浓度区域标记为潜在沉淀相
        threshold = (c_eq_matrix + c_eq_precip) / 2
        self.eta_precipitate[self.c > threshold] = 0.1
        
        self.fields['eta_precipitate'] = self.eta_precipitate
        
        # 初始化化学势
        self._update_chemical_potential()
        
        logger.info(f"Initialized concentration field: mean={self.c.mean():.4f}")
    
    def _create_clustered_initial(self, c_nominal: float) -> np.ndarray:
        """创建聚集的初始分布 (促进形核)"""
        shape = (self.config.nx, self.config.ny)
        field = np.full(shape, c_nominal)
        
        # 添加一些高浓度区域
        n_clusters = max(1, self.config.nx // 20)
        for _ in range(n_clusters):
            cx = np.random.randint(0, self.config.nx)
            cy = np.random.randint(0, self.config.ny)
            radius = np.random.randint(2, 5)
            
            y, x = np.ogrid[:self.config.ny, :self.config.nx]
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            field[mask] = min(c_nominal * 2, 0.3)  # 增加局部浓度
        
        return field
    
    def _compute_elastic_interaction(self) -> np.ndarray:
        """
        计算弹性相互作用能
        
        沉淀相与基体间的弹性应变能
        
        Returns:
            E_elastic: 弹性相互作用能场
        """
        # 简化模型：弹性能正比于沉淀相体积分数和错配度
        eta = self.eta_precipitate
        delta = self.config.lattice_mismatch
        E_modulus = 100e9  # 100 GPa in Pa
        
        # 应变能密度: E = 0.5 * E * (δ*η)²
        E_elastic = 0.5 * E_modulus * (delta * eta)**2
        
        # 转换为无量纲单位 (eV/nm³)
        E_elastic = E_elastic * 1e-18 / 1.602e-19
        
        return E_elastic
    
    def _nucleation_model(self) -> int:
        """
        形核模型
        
        基于经典形核理论计算新形核数量
        
        Returns:
            n_new: 新形核数量
        """
        # 计算过饱和度
        c_matrix = self.c[self.eta_precipitate < 0.5].mean()
        c_eq = self.config.equilibrium_composition['matrix']
        supersaturation = c_matrix / c_eq if c_eq > 0 else 1.0
        
        if supersaturation <= 1.0:
            return 0
        
        # 简化的形核率计算
        # J = J0 * exp(-ΔG*/kT)
        # ΔG* ∝ 1/(ln S)²
        
        Delta_G_star = 10.0 / (np.log(supersaturation)**2 + 1e-10)  # 简化的形核势垒
        
        kT = 8.617e-5 * self.config.temperature  # eV
        
        nucleation_rate = self.config.nucleation_density * np.exp(-Delta_G_star / kT)
        
        # 在当前时间步的形核数量
        dt = self.config.dt
        domain_volume = (self.config.nx * self.config.dx * 1e-9)**3  # m³
        
        n_new = int(nucleation_rate * domain_volume * dt)
        
        return max(0, n_new)
    
    def _place_nucleus(self, radius: Optional[float] = None):
        """
        放置一个新形核
        
        Args:
            radius: 形核半径 (网格单位)
        """
        if radius is None:
            radius = int(self.config.critical_radius / self.config.dx)
        
        # 随机选择位置 (在基体区域)
        matrix_mask = self.eta_precipitate < 0.3
        matrix_indices = np.argwhere(matrix_mask)
        
        if len(matrix_indices) == 0:
            return
        
        idx = np.random.choice(len(matrix_indices))
        cx, cy = matrix_indices[idx]
        
        # 创建球形形核
        y, x = np.ogrid[:self.config.ny, :self.config.nx]
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        
        # 设置序参量
        self.eta_precipitate[mask] = 1.0
        
        # 设置沉淀相成分
        c_precip = self.config.equilibrium_composition['precipitate']
        self.c[mask] = c_precip
        
        # 记录形核
        self.nuclei.append({
            'time': self.time,
            'position': (cx, cy),
            'radius': radius
        })
    
    def evolve_step(self) -> Dict:
        """
        执行沉淀相演化步骤
        
        Returns:
            info: 演化信息
        """
        # 1. 形核
        n_new_nuclei = self._nucleation_model()
        for _ in range(n_new_nuclei):
            self._place_nucleus()
        
        # 2. 浓度场演化 (Cahn-Hilliard + 弹性耦合)
        # 化学势包含弹性贡献
        E_elastic = self._compute_elastic_interaction()
        
        # 修改化学势
        mu_chem = self._compute_bulk_chemical_potential(self.c)
        laplacian_c = self.compute_laplacian(self.c)
        self.mu = mu_chem - self.config.kappa * laplacian_c + 0.1 * E_elastic
        
        # 标准Cahn-Hilliard演化
        laplacian_mu = self.compute_laplacian(self.mu)
        dc_dt = self.config.M * laplacian_mu
        
        c_new = self.c + self.config.dt * dc_dt
        c_new = np.clip(c_new, 0.001, 0.999)
        c_new = self.bc_handler.apply(c_new)
        
        dc_max = np.abs(c_new - self.c).max()
        self.c = c_new
        self.fields['c'] = self.c
        
        # 3. 沉淀相序参量演化
        # Allen-Cahn类型演化
        dfd_eta = 2 * self.eta_precipitate * (1 - self.eta_precipitate) * (1 - 2*self.eta_precipitate)
        laplacian_eta = self.compute_laplacian(self.eta_precipitate)
        
        L_eta = self.config.M * 0.1  # 序参量弛豫速率
        deta_dt = -L_eta * (dfd_eta - self.config.kappa * laplacian_eta)
        
        # 浓度约束：只在富溶质区域形成沉淀相
        c_threshold = (self.config.equilibrium_composition['matrix'] + 
                      self.config.equilibrium_composition['precipitate']) / 2
        deta_dt *= (self.c > c_threshold * 0.8)
        
        eta_new = self.eta_precipitate + self.config.dt * deta_dt
        eta_new = np.clip(eta_new, 0, 1)
        
        self.eta_precipitate = eta_new
        self.fields['eta_precipitate'] = self.eta_precipitate
        
        # 4. 更新沉淀相列表
        self._update_precipitate_list()
        
        converged = dc_max < self.config.tolerance
        
        info = {
            'step': self.step,
            'time': self.time,
            'dc_max': dc_max,
            'n_new_nuclei': n_new_nuclei,
            'n_precipitates': len(self.precipitates),
            'c_mean': self.c.mean(),
            'c_std': self.c.std(),
            'energy': self.compute_energy(),
            'converged': converged
        }
        
        return info
    
    def _update_precipitate_list(self):
        """更新沉淀相列表"""
        # 识别连通区域
        from scipy import ndimage
        
        precipitate_mask = self.eta_precipitate > 0.5
        labeled_array, num_features = ndimage.label(precipitate_mask)
        
        self.precipitates = []
        
        for i in range(1, num_features + 1):
            mask = labeled_array == i
            
            # 计算沉淀相属性
            size = np.sum(mask)
            centroid = ndimage.center_of_mass(mask)
            
            # 等效半径
            radius = np.sqrt(size / np.pi) * self.config.dx
            
            self.precipitates.append({
                'id': i,
                'size': size,
                'radius': radius,
                'centroid': centroid,
                'volume_fraction': size / self.c.size
            })
    
    def get_precipitate_statistics(self) -> Dict:
        """
        获取沉淀相统计信息
        
        Returns:
            stats: 统计信息字典
        """
        if not self.precipitates:
            return {
                'number_density': 0,
                'average_radius': 0,
                'volume_fraction': 0
            }
        
        radii = [p['radius'] for p in self.precipitates]
        volume_fractions = [p['volume_fraction'] for p in self.precipitates]
        
        domain_volume = (self.config.nx * self.config.dx * 1e-9)**3  # m³
        
        stats = {
            'count': len(self.precipitates),
            'number_density': len(self.precipitates) / domain_volume,  # m^-3
            'average_radius': np.mean(radii),  # nm
            'radius_std': np.std(radii),
            'max_radius': max(radii),
            'min_radius': min(radii),
            'total_volume_fraction': sum(volume_fractions),
            'nucleation_count': len(self.nuclei)
        }
        
        return stats
    
    def get_size_distribution(self, n_bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取沉淀相尺寸分布
        
        Args:
            n_bins: 分箱数量
            
        Returns:
            bins, counts: 尺寸分布
        """
        if not self.precipitates:
            return np.array([]), np.array([])
        
        radii = [p['radius'] for p in self.precipitates]
        
        counts, bins = np.histogram(radii, bins=n_bins)
        
        return bins, counts
    
    def estimate_hardness_increase(self) -> float:
        """
        估算硬度增加
        
        基于沉淀强化理论
        Δσ ∝ f^(1/2) / r
        
        Returns:
            hardness_increase: 相对硬度增加
        """
        stats = self.get_precipitate_statistics()
        
        f = stats['total_volume_fraction']
        r = stats['average_radius']
        
        if r > 0:
            # 简化模型
            delta_sigma = np.sqrt(f) / r * 100  # 相对值
        else:
            delta_sigma = 0
        
        return delta_sigma
