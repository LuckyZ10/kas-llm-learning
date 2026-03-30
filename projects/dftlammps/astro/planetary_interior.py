"""
行星内部模拟模块
===============
模拟行星内部的物理状态和动力学过程，包括：
- 行星结构模型
- 高压物态方程
- 内核动力学
- 磁场发电机

应用场景：
- 系外行星内部结构反演
- 地球深部物理
- 气态巨行星建模
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlanetType(Enum):
    """行星类型"""
    TERRESTRIAL = "terrestrial"  # 类地行星
    SUPER_EARTH = "super_earth"
    MINI_NEPTUNE = "mini_neptune"
    ICE_GIANT = "ice_giant"  # 天王星、海王星
    GAS_GIANT = "gas_giant"  # 木星、土星
    HOT_JUPITER = "hot_jupiter"
    LAVA_WORLD = "lava_world"


class CoreComposition(Enum):
    """核心成分"""
    IRON = "iron"
    IRON_NICKEL = "iron_nickel"
    ROCKY = "rocky"
    ICE_ROCK_MIX = "ice_rock_mixture"


class MantleComposition(Enum):
    """幔成分"""
    SILICATE = "silicate"
    WATER_ICE = "water_ice"
    HIGH_PRESSURE_ICE = "high_pressure_ice"
    SUPERIONIC_WATER = "superionic_water"


@dataclass
class LayerProperties:
    """行星层性质"""
    inner_radius: float    # m
    outer_radius: float    # m
    density: float         # kg/m^3
    pressure_top: float    # Pa
    pressure_bottom: float # Pa
    temperature_top: float # K
    temperature_bottom: float # K
    composition: str
    phase: str


@dataclass
class MagneticField:
    """磁场属性"""
    surface_strength: float    # T
    dipole_moment: float       # A·m^2
    dipole_tilt: float         # degrees
    quadrupole_ratio: float
    dynamo_region: Tuple[float, float]  # (r_inner, r_outer)


class PlanetaryStructureModel:
    """行星结构模型"""
    
    def __init__(self,
                 planet_type: PlanetType,
                 mass: float,      # Earth masses
                 radius: float):   # Earth radii
        self.planet_type = planet_type
        self.mass = mass * 5.97e24  # kg
        self.radius = radius * 6.37e6  # m
        
        self.G = 6.67e-11
        self.layers: List[LayerProperties] = []
        
        logger.info(f"行星模型: {planet_type.value}, M={mass:.2f} M⊕, R={radius:.2f} R⊕")
    
    def build_terrestrial_structure(self,
                                   core_mass_fraction: float = 0.3,
                                   core_composition: CoreComposition = CoreComposition.IRON,
                                   mantle_composition: MantleComposition = MantleComposition.SILICATE) -> List[LayerProperties]:
        """
        构建类地行星内部结构
        
        简化模型：均匀密度核心 + 均匀密度地幔
        """
        # 核心参数
        if core_composition == CoreComposition.IRON:
            rho_core = 12000  # kg/m^3
        elif core_composition == CoreComposition.IRON_NICKEL:
            rho_core = 11000
        else:
            rho_core = 10000
        
        # 地幔密度
        if mantle_composition == MantleComposition.SILICATE:
            rho_mantle = 4500
        else:
            rho_mantle = 4000
        
        # 通过质量守恒和核心质量分数求核心半径
        # M = (4/3)π * (ρ_core * r_core^3 + ρ_mantle * (r_planet^3 - r_core^3))
        # 设 x = (r_core/r_planet)^3
        # M = (4/3)π * r_planet^3 * (ρ_core * x + ρ_mantle * (1 - x))
        
        M_total = self.mass
        V_total = (4/3) * np.pi * self.radius**3
        
        # 核心质量分数方程求解
        # f_core = (ρ_core * r_core^3) / M_total
        r_core = ((3 * core_mass_fraction * M_total) / (4 * np.pi * rho_core)) ** (1/3)
        
        # 实际密度调整
        V_core = (4/3) * np.pi * r_core**3
        V_mantle = V_total - V_core
        
        M_core = rho_core * V_core
        M_mantle = M_total - M_core
        rho_mantle_adjusted = M_mantle / V_mantle
        
        # 压力估计 (简化)
        P_center = (3/8) * np.pi * self.G * rho_core**2 * r_core**2
        P_core_mantle = P_center * (rho_mantle_adjusted / rho_core)
        P_surface = 0
        
        # 温度估计
        T_surface = 300  # K
        T_core_mantle = 3000 + 500 * (self.mass / 5.97e24)  # 随质量增加
        T_center = T_core_mantle + 1000
        
        # 构建层
        layers = [
            LayerProperties(
                inner_radius=0,
                outer_radius=r_core,
                density=rho_core,
                pressure_top=P_core_mantle,
                pressure_bottom=P_center,
                temperature_top=T_core_mantle,
                temperature_bottom=T_center,
                composition=core_composition.value,
                phase="solid" if self.mass < 5 * 5.97e24 else "liquid"
            ),
            LayerProperties(
                inner_radius=r_core,
                outer_radius=self.radius,
                density=rho_mantle_adjusted,
                pressure_top=P_surface,
                pressure_bottom=P_core_mantle,
                temperature_top=T_surface,
                temperature_bottom=T_core_mantle,
                composition=mantle_composition.value,
                phase="solid"
            )
        ]
        
        self.layers = layers
        return layers
    
    def build_gas_giant_structure(self) -> List[LayerProperties]:
        """构建气态巨行星结构"""
        # 木星型结构
        r_core = 0.15 * self.radius  # 岩石/冰核
        r_metallic = 0.6 * self.radius  # 金属氢区域
        r_molecular = self.radius  # 分子氢大气
        
        layers = [
            LayerProperties(
                inner_radius=0,
                outer_radius=r_core,
                density=25000,
                pressure_top=40e6,  # 40 Mbar
                pressure_bottom=50e6,
                temperature_top=20000,
                temperature_bottom=25000,
                composition="rock_ice_core",
                phase="solid"
            ),
            LayerProperties(
                inner_radius=r_core,
                outer_radius=r_metallic,
                density=3000,
                pressure_top=2e6,
                pressure_bottom=40e6,
                temperature_top=5000,
                temperature_bottom=20000,
                composition="metallic_hydrogen",
                phase="liquid"
            ),
            LayerProperties(
                inner_radius=r_metallic,
                outer_radius=r_molecular,
                density=200,
                pressure_top=1e5,  # 1 bar
                pressure_bottom=2e6,
                temperature_top=100,
                temperature_bottom=5000,
                composition="molecular_hydrogen_helium",
                phase="gas"
            )
        ]
        
        self.layers = layers
        return layers
    
    def calculate_gravity_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算重力加速度随深度的变化
        
        g(r) = G * M(r) / r^2
        """
        radii = np.linspace(0, self.radius, 100)
        gravity = []
        
        for r in radii:
            # 计算r以内的质量
            M_enclosed = 0
            for layer in self.layers:
                if layer.outer_radius <= r:
                    # 整层在内
                    V = (4/3) * np.pi * (layer.outer_radius**3 - layer.inner_radius**3)
                    M_enclosed += layer.density * V
                elif layer.inner_radius < r < layer.outer_radius:
                    # 部分在内
                    V = (4/3) * np.pi * (r**3 - layer.inner_radius**3)
                    M_enclosed += layer.density * V
            
            if r > 0:
                g = self.G * M_enclosed / r**2
            else:
                g = 0
            
            gravity.append(g)
        
        return radii, np.array(gravity)
    
    def calculate_moment_of_inertia_factor(self) -> float:
        """
        计算转动惯量因子 C/(MR^2)
        
        均匀球体: 0.4
        点质量: 0
        行星内部密度分层会降低该值
        """
        if not self.layers:
            return 0.4  # 默认值
        
        # 计算转动惯量
        C = 0
        for layer in self.layers:
            # 球壳的转动惯量
            C += (8/15) * np.pi * layer.density * (
                layer.outer_radius**5 - layer.inner_radius**5
            )
        
        # 归一化
        C_factor = C / (self.mass * self.radius**2)
        
        return C_factor
    
    def calculate_surface_gravity(self) -> float:
        """计算表面重力加速度"""
        return self.G * self.mass / self.radius**2
    
    def calculate_escape_velocity(self) -> float:
        """计算逃逸速度"""
        return np.sqrt(2 * self.G * self.mass / self.radius)


class HighPressureEOS:
    """高压物态方程"""
    
    def __init__(self):
        self.eos_params = self._init_eos_parameters()
    
    def _init_eos_parameters(self) -> Dict:
        """初始化EOS参数"""
        return {
            'iron': {
                'V0': 6.7,      # cm^3/mol
                'K0': 160,      # GPa
                'K0_prime': 5.5,
                'gamma0': 2.0,  # Grüneisen参数
                'q': 1.0
            },
            'silicate_perovskite': {
                'V0': 24.4,
                'K0': 253,
                'K0_prime': 4.0,
                'gamma0': 1.5,
                'q': 1.0
            },
            'water': {
                'V0': 18.0,
                'K0': 2.2,
                'K0_prime': 4.0,
                'gamma0': 0.9,
                'q': 1.0
            },
            'hydrogen': {
                'V0': 15.0,
                'K0': 0.05,
                'K0_prime': 7.0,
                'gamma0': 1.0,
                'q': 1.0
            }
        }
    
    def birch_murnaghan_energy(self,
                              volume: float,
                              material: str) -> float:
        """
        Birch-Murnaghan物态方程 - 能量形式
        
        E(V) = E0 + (9V0B0/16) * [(V0/V)^(2/3) - 1]^2 *
               {6 - 4(V0/V)^(2/3) + B0'[(V0/V)^(2/3) - 1]}
        """
        params = self.eos_params.get(material, self.eos_params['iron'])
        
        V0 = params['V0']
        B0 = params['K0']
        B0_prime = params['K0_prime']
        
        eta = (V0 / volume) ** (2/3)
        
        energy = (9 * V0 * B0 / 16) * ((eta - 1)**3 * B0_prime + 
                                        (eta - 1)**2 * (6 - 4*eta))
        
        return energy  # kJ/mol (需单位转换)
    
    def birch_murnaghan_pressure(self,
                                volume: float,
                                material: str) -> float:
        """
        Birch-Murnaghan物态方程 - 压力形式
        
        P(V) = (3B0/2) * [(V0/V)^(7/3) - (V0/V)^(5/3)] *
               {1 + (3/4)(B0' - 4)[(V0/V)^(2/3) - 1]}
        """
        params = self.eos_params.get(material, self.eos_params['iron'])
        
        V0 = params['V0']
        B0 = params['K0']
        B0_prime = params['K0_prime']
        
        eta = V0 / volume
        eta_2_3 = eta ** (2/3)
        
        P = (3 * B0 / 2) * (eta ** (7/3) - eta ** (5/3)) * \
            (1 + 0.75 * (B0_prime - 4) * (eta_2_3 - 1))
        
        return P  # GPa
    
    def vinet_pressure(self,
                      volume: float,
                      material: str) -> float:
        """
        Vinet物态方程 (适用于高压)
        
        P(V) = 3B0 * (1 - (V/V0)^(1/3)) / (V/V0)^(2/3) *
               exp{(3/2)(B0' - 1)(1 - (V/V0)^(1/3))}
        """
        params = self.eos_params.get(material, self.eos_params['iron'])
        
        V0 = params['V0']
        B0 = params['K0']
        B0_prime = params['K0_prime']
        
        x = (volume / V0) ** (1/3)
        
        P = 3 * B0 * (1 - x) / x**2 * np.exp(1.5 * (B0_prime - 1) * (1 - x))
        
        return P
    
    def thomas_fermi_pressure(self,
                             density: float,
                             atomic_number: float,
                             atomic_weight: float) -> float:
        """
        Thomas-Fermi物态方程 (适用于极高压力)
        
        适用于 P > 10 TPa
        """
        # 简化的TF模型
        # P ∝ ρ^(5/3) for完全简并电子气
        const = 0.0234  # 原子单位
        
        n_e = density / atomic_weight * atomic_number  # 电子数密度 (原子单位)
        
        P = const * n_e ** (5/3)  # 原子单位
        
        # 转换为GPa
        P_GPa = P * 29421
        
        return P_GPa


class CoreDynamics:
    """行星核心动力学"""
    
    def __init__(self,
                 core_radius: float,
                 core_density: float,
                 rotation_rate: float = None):
        self.core_radius = core_radius
        self.core_density = core_density
        self.rotation_rate = rotation_rate or 7.29e-5  # rad/s (地球)
        
        self.mu0 = 4 * np.pi * 1e-7  # 真空磁导率
        self.sigma = 1e6  # 电导率 (S/m)
    
    def estimate_magnetic_diffusivity(self) -> float:
        """估计磁扩散率"""
        return 1 / (self.mu0 * self.sigma)
    
    def calculate_magnetic_reynolds_number(self,
                                          velocity: float,
                                          length_scale: float = None) -> float:
        """
        计算磁雷诺数
        
        Rm = μ0 σ v L
        
        Rm > 10-100 通常需要维持发电机
        """
        L = length_scale or self.core_radius
        Rm = self.mu0 * self.sigma * velocity * L
        return Rm
    
    def estimate_dynamo_field_strength(self,
                                      convective_velocity: float,
                                      density_ratio: float = 1.0) -> float:
        """
        估计发电机磁场强度
        
        使用能量平衡估计
        """
        # 动能密度
        rho = self.core_density
        v = convective_velocity
        
        # 动能通量
        kinetic_flux = 0.5 * rho * v**3
        
        # 转换为磁场能量 (假设效率1%)
        efficiency = 0.01
        magnetic_energy = efficiency * kinetic_flux
        
        # 磁场强度 B = sqrt(2 * mu0 * E_magnetic)
        B = np.sqrt(2 * self.mu0 * magnetic_energy)
        
        # 表面场强 (衰减)
        B_surface = B * (self.core_radius / 6.37e6)**3 * density_ratio
        
        return B_surface
    
    def calculate_rossby_number(self,
                               velocity: float,
                               length_scale: float) -> float:
        """
        计算Rossby数
        
        Ro = v / (2ΩL)
        
        Ro << 1: 旋转主导
        Ro > 1: 惯性主导
        """
        Ro = velocity / (2 * self.rotation_rate * length_scale)
        return Ro
    
    def calculate_ekman_number(self,
                              viscosity: float) -> float:
        """
        计算Ekman数
        
        E = ν / (2ΩL^2)
        
        E << 1: 旋转对流
        """
        E = viscosity / (2 * self.rotation_rate * self.core_radius**2)
        return E


class MagneticDynamoModel:
    """磁场发电机模型"""
    
    def __init__(self, planet_model: PlanetaryStructureModel):
        self.planet = planet_model
        self.core = None
        
        # 找出核心层
        for layer in planet_model.layers:
            if 'core' in layer.composition:
                self.core = layer
                break
    
    def calculate_dynamo_criteria(self) -> Dict[str, float]:
        """计算发电机维持条件"""
        if self.core is None:
            return {}
        
        dynamics = CoreDynamics(
            self.core.outer_radius - self.core.inner_radius,
            self.core.density
        )
        
        # 对流速度估计 (基于热通量)
        # 简化的对流速度估计
        F_conv = 1e3  # W/m^2 热通量估计
        v_conv = (F_conv / self.core.density / 1000)**(1/3)  # m/s
        
        Rm = dynamics.calculate_magnetic_reynolds_number(v_conv)
        Ro = dynamics.calculate_rossby_number(v_conv, self.core.outer_radius)
        
        # 估计磁场强度
        B_est = dynamics.estimate_dynamo_field_strength(v_conv)
        
        return {
            'magnetic_reynolds_number': Rm,
            'rossby_number': Ro,
            'estimated_field_strength_T': B_est,
            'dynamo_likely': Rm > 10 and Ro < 1
        }
    
    def generate_magnetic_field_model(self,
                                     multipole_cutoff: int = 3) -> Dict[str, np.ndarray]:
        """
        生成多极磁场模型
        
        Returns:
            球谐系数
        """
        criteria = self.calculate_dynamo_criteria()
        
        if not criteria.get('dynamo_likely', False):
            logger.warning("发电机可能不活跃")
        
        # 简化：假设轴对称偶极场为主
        g10 = criteria.get('estimated_field_strength_T', 1e-4)  # 偶极矩
        
        # 高阶项 (随阶数衰减)
        harmonics = {'g_1_0': g10}
        
        for l in range(2, multipole_cutoff + 1):
            # 近似衰减
            harmonics[f'g_{l}_0'] = g10 / (l**2)
            harmonics[f'h_{l}_0'] = g10 / (l**2) * 0.1  # 较小的高阶项
        
        return harmonics


# 应用案例: 系外行星内部结构
def exoplanet_interior_example():
    """系外行星内部结构研究示例"""
    logger.info("=" * 60)
    logger.info("系外行星内部结构研究示例")
    logger.info("=" * 60)
    
    # 1. 地球参考模型
    logger.info("\n--- 地球内部结构 ---")
    earth = PlanetaryStructureModel(PlanetType.TERRESTRIAL, mass=1.0, radius=1.0)
    earth_layers = earth.build_terrestrial_structure(
        core_mass_fraction=0.32,
        core_composition=CoreComposition.IRON_NICKEL
    )
    
    logger.info(f"行星层数: {len(earth_layers)}")
    for i, layer in enumerate(earth_layers):
        logger.info(f"  层{i+1}: {layer.composition}")
        logger.info(f"    半径范围: {layer.inner_radius/1000:.0f} - {layer.outer_radius/1000:.0f} km")
        logger.info(f"    密度: {layer.density:.0f} kg/m³")
        logger.info(f"    压力: {layer.pressure_top/1e9:.1f} - {layer.pressure_bottom/1e9:.1f} GPa")
    
    # 物理参数
    C_factor = earth.calculate_moment_of_inertia_factor()
    g_surface = earth.calculate_surface_gravity()
    v_escape = earth.calculate_escape_velocity()
    
    logger.info(f"\n地球物理参数:")
    logger.info(f"  转动惯量因子 C/MR²: {C_factor:.3f} (观测值: 0.3307)")
    logger.info(f"  表面重力: {g_surface:.2f} m/s²")
    logger.info(f"  逃逸速度: {v_escape/1000:.1f} km/s")
    
    # 重力剖面
    radii, gravity = earth.calculate_gravity_profile()
    logger.info(f"  中心重力: {gravity[0]:.3f} m/s²")
    logger.info(f"  地核-地幔边界重力: {gravity[len(gravity)//5]:.2f} m/s²")
    
    # 2. 超级地球 (5 M⊕, 1.5 R⊕)
    logger.info("\n--- 超级地球内部结构 ---")
    super_earth = PlanetaryStructureModel(PlanetType.SUPER_EARTH, mass=5.0, radius=1.5)
    super_earth_layers = super_earth.build_terrestrial_structure(
        core_mass_fraction=0.35
    )
    
    logger.info(f"超级地球层数: {len(super_earth_layers)}")
    for layer in super_earth_layers:
        logger.info(f"  {layer.composition}: R={layer.outer_radius/1000:.0f} km, "
                   f"ρ={layer.density:.0f} kg/m³")
    
    C_factor_se = super_earth.calculate_moment_of_inertia_factor()
    g_surface_se = super_earth.calculate_surface_gravity()
    
    logger.info(f"  转动惯量因子: {C_factor_se:.3f}")
    logger.info(f"  表面重力: {g_surface_se:.2f} m/s² ({g_surface_se/g_surface:.1f}倍地球)")
    
    # 3. 木星型气态巨行星
    logger.info("\n--- 气态巨行星内部结构 ---")
    jupiter = PlanetaryStructureModel(PlanetType.GAS_GIANT, mass=318.0, radius=11.2)
    jupiter_layers = jupiter.build_gas_giant_structure()
    
    logger.info(f"木星层数: {len(jupiter_layers)}")
    for layer in jupiter_layers:
        logger.info(f"  {layer.composition}:")
        logger.info(f"    半径范围: {layer.inner_radius/1000:.0f} - {layer.outer_radius/1000:.0f} km")
        logger.info(f"    密度: {layer.density:.0f} kg/m³")
        logger.info(f"    压力: {layer.pressure_top/1e9:.0f} - {layer.pressure_bottom/1e9:.0f} GPa")
    
    # 4. 高压物态方程
    logger.info("\n--- 高压物态方程计算 ---")
    eos = HighPressureEOS()
    
    # 铁核压力-体积关系
    volumes = np.linspace(4, 7, 20)  # cm^3/mol
    pressures_fe = [eos.birch_murnaghan_pressure(v, 'iron') for v in volumes]
    
    logger.info(f"铁核压力范围: {min(pressures_fe):.1f} - {max(pressures_fe):.1f} GPa")
    
    # 极高压力 (核心中心)
    rho_center = 13000  # kg/m^3
    V_center = 55.85 / (rho_center / 1000)  # cm^3/mol
    P_center = eos.birch_murnaghan_pressure(V_center, 'iron')
    logger.info(f"估计地核中心压力: {P_center:.0f} GPa")
    
    # 5. 发电机磁场
    logger.info("\n--- 发电机磁场模型 ---")
    dynamo_earth = MagneticDynamoModel(earth)
    criteria = dynamo_earth.calculate_dynamo_criteria()
    
    logger.info(f"磁雷诺数: {criteria['magnetic_reynolds_number']:.1f}")
    logger.info(f"Rossby数: {criteria['rossby_number']:.2e}")
    logger.info(f"估计表面磁场: {criteria['estimated_field_strength_T']*1e6:.0f} μT")
    logger.info(f"发电机活跃: {criteria['dynamo_likely']}")
    
    harmonics = dynamo_earth.generate_magnetic_field_model()
    logger.info(f"多极展开系数: g_1_0 = {harmonics.get('g_1_0', 0)*1e6:.0f} μT")
    
    return {
        'earth': earth,
        'super_earth': super_earth,
        'jupiter': jupiter,
        'eos': eos,
        'dynamo_criteria': criteria
    }


if __name__ == "__main__":
    exoplanet_interior_example()
