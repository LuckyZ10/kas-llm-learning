"""
白矮星壳层模拟模块
=================
模拟白矮星和中子星壳层的极端物理条件，包括：
- 超密物质物态方程
- 壳层结晶化
- 核燃烧过程
- 冷却演化

应用场景：
- I型超新星前身星研究
- 脉冲星地壳物理
- 致密物质状态研究
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompactObjectType(Enum):
    """致密天体类型"""
    WHITE_DWARF_CO = "carbon_oxygen_white_dwarf"
    WHITE_DWARF_ONe = "oxygen_neon_white_dwarf"
    NEUTRON_STAR = "neutron_star"
    QUARK_STAR = "quark_star"


class ShellComposition(Enum):
    """壳层成分"""
    HYDROGEN = "hydrogen"
    HELIUM = "helium"
    CARBON = "carbon"
    OXYGEN = "oxygen"
    NEON = "neon"
    MAGNESIUM = "magnesium"
    IRON = "iron"


class CrystallizationState(Enum):
    """结晶化状态"""
    LIQUID = "liquid"
    CRYSTALLIZING = "crystallizing"
    CRYSTALLINE = "crystalline"
    QUANTUM_LIQUID = "quantum_liquid"


@dataclass
class NuclearReaction:
    """核反应"""
    reactants: List[str]
    products: List[str]
    Q_value: float          # MeV
    rate: float             # 反应率 (s^-1)
    temperature_threshold: float  # K
    density_threshold: float      # g/cm^3


@dataclass
class ShellLayer:
    """壳层层结构"""
    composition: ShellComposition
    inner_radius: float     # m
    outer_radius: float     # m
    mass: float             # kg
    density: float          # kg/m^3
    temperature: float      # K
    pressure: float         # Pa
    state: CrystallizationState
    ion_coupling: float     # Γ参数


@dataclass
class CoulombCrystal:
    """库仑晶体性质"""
    lattice_type: str       # bcc, fcc
    melting_temperature: float  # K
    lattice_spacing: float  # m
    Debye_temperature: float  # K
    shear_modulus: float    # Pa


class WhiteDwarfModel:
    """白矮星模型"""
    
    def __init__(self,
                 mass: float,       # Solar masses
                 central_density: float = None,  # kg/m^3
                 effective_temperature: float = 10000):  # K
        self.mass = mass * 1.989e30  # kg
        self.T_eff = effective_temperature
        self.R_sun = 6.96e8
        self.M_sun = 1.989e30
        self.G = 6.67e-11
        self.kB = 1.38e-23
        self.h = 6.626e-34
        self.m_u = 1.66e-27  # 原子质量单位
        self.e = 1.6e-19
        
        # Chandrasekhar质量
        self.M_ch = 1.44  # M_sun
        
        # 计算半径 (质量-半径关系)
        self.radius = self._calculate_radius()
        
        self.central_density = central_density or self._estimate_central_density()
        
        # 壳层结构
        self.shells: List[ShellLayer] = []
        
        logger.info(f"白矮星模型: M={mass:.3f} M☉, R={self.radius/1000:.0f} km, "
                   f"Teff={effective_temperature}K")
    
    def _calculate_radius(self) -> float:
        """
        计算白矮星半径 (质量-半径关系)
        
        对于非相对论性简并电子气:
        R ∝ M^(-1/3)
        
        考虑相对论修正
        """
        # 简化的质量-半径关系
        # R ≈ R_earth * (M/M_chandra)^(-1/3) * (1 - (M/M_chandra)^2)^(1/2)
        
        R_earth = 6.37e6
        x = self.mass / self.M_ch / self.M_sun
        
        if x < 0.1:
            radius = R_earth * x**(-1/3) * 0.01
        else:
            radius = R_earth * (1/x)**(1/3) * 0.012 * np.sqrt(1 - x**2)
        
        return max(radius, 1000)  # 最小1km
    
    def _estimate_central_density(self) -> float:
        """估计中心密度"""
        # 平均密度
        avg_density = 3 * self.mass / (4 * np.pi * self.radius**3)
        # 中心密度通常比平均密度高3-10倍
        return avg_density * 5
    
    def build_shell_structure(self,
                             composition_profile: Dict[str, float] = None) -> List[ShellLayer]:
        """
        构建壳层结构
        
        标准分层：H → He → C/O → O/Ne → Mg
        """
        if composition_profile is None:
            # CO白矮星默认剖面
            composition_profile = {
                'outer': {'H': 0.0, 'He': 0.0, 'C': 0.5, 'O': 0.5},
            }
        
        shells = []
        
        # 质量分层 (简化)
        mass_fractions = {
            ShellComposition.OXYGEN: 0.4,
            ShellComposition.CARBON: 0.4,
            ShellComposition.HELIUM: 0.15,
            ShellComposition.HYDROGEN: 0.05
        }
        
        r_outer = self.radius
        
        for comp, mass_frac in mass_fractions.items():
            layer_mass = self.mass * mass_frac
            
            # 简化的密度剖面
            if comp == ShellComposition.HYDROGEN:
                density = 1e5  # kg/m^3
            elif comp == ShellComposition.HELIUM:
                density = 1e6
            else:
                density = 1e8
            
            # 计算层厚度
            volume = layer_mass / density
            r_inner = (r_outer**3 - 3*volume/(4*np.pi))**(1/3)
            
            # 温度 (从表面向内增加)
            T_layer = self.T_eff * (r_outer / self.radius)**(-0.5)
            
            # 压力
            P_layer = self.G * self.mass * density / r_outer
            
            # 离子耦合参数
            Gamma = self._calculate_ion_coupling(density, T_layer, comp)
            
            # 确定状态
            if Gamma < 1:
                state = CrystallizationState.LIQUID
            elif Gamma < 180:
                state = CrystallizationState.CRYSTALLIZING
            else:
                state = CrystallizationState.CRYSTALLINE
            
            layer = ShellLayer(
                composition=comp,
                inner_radius=r_inner,
                outer_radius=r_outer,
                mass=layer_mass,
                density=density,
                temperature=T_layer,
                pressure=P_layer,
                state=state,
                ion_coupling=Gamma
            )
            
            shells.append(layer)
            r_outer = r_inner
        
        self.shells = shells
        return shells
    
    def _calculate_ion_coupling(self,
                               density: float,
                               temperature: float,
                               composition: ShellComposition) -> float:
        """
        计算离子耦合参数 Γ
        
        Γ = (Z*e)^2 / (4πε0 * a * k_B * T)
        
        其中 a = (3/(4πn_i))^(1/3) 是Wigner-Seitz半径
        
        Γ >~ 180: 结晶化
        """
        Z = self._get_atomic_number(composition)
        A = self._get_atomic_mass(composition)
        
        # 离子数密度
        n_i = density / (A * self.m_u)
        
        # Wigner-Seitz半径
        a_ws = (3 / (4 * np.pi * n_i))**(1/3)
        
        # 库仑能量 / 热能
        Gamma = (Z * self.e)**2 / (4 * np.pi * 8.85e-12 * a_ws * self.kB * temperature)
        
        return Gamma
    
    def _get_atomic_number(self, composition: ShellComposition) -> int:
        """获取原子序数"""
        Z = {
            ShellComposition.HYDROGEN: 1,
            ShellComposition.HELIUM: 2,
            ShellComposition.CARBON: 6,
            ShellComposition.OXYGEN: 8,
            ShellComposition.NEON: 10,
            ShellComposition.MAGNESIUM: 12,
            ShellComposition.IRON: 26
        }
        return Z.get(composition, 6)
    
    def _get_atomic_mass(self, composition: ShellComposition) -> float:
        """获取原子质量数"""
        A = {
            ShellComposition.HYDROGEN: 1,
            ShellComposition.HELIUM: 4,
            ShellComposition.CARBON: 12,
            ShellComposition.OXYGEN: 16,
            ShellComposition.NEON: 20,
            ShellComposition.MAGNESIUM: 24,
            ShellComposition.IRON: 56
        }
        return A.get(composition, 12)
    
    def calculate_crystallization_luminosity(self) -> float:
        """
        计算结晶化释放的光度
        
        L_crystallization ≈ (4πR³/3) * ρ * ε_coulomb * (dX_crystal/dt)
        """
        if not self.shells:
            return 0
        
        # 库仑能量释放 (约0.1-1 keV/ion)
        epsilon_coulomb = 0.5 * 1e3 * self.e  # J/ion
        
        # 结晶化速率 (简化)
        dX_dt = 1e-15  # s^-1
        
        total_luminosity = 0
        for shell in self.shells:
            if shell.state == CrystallizationState.CRYSTALLIZING:
                n_ion = shell.density / (self._get_atomic_mass(shell.composition) * self.m_u)
                volume = (4/3) * np.pi * (shell.outer_radius**3 - shell.inner_radius**3)
                L = volume * n_ion * epsilon_coulomb * dX_dt
                total_luminosity += L
        
        return total_luminosity


class NeutronStarCrust:
    """中子星壳层模型"""
    
    def __init__(self,
                 mass: float = 1.4,  # Solar masses
                 radius: float = 10):  # km
        self.mass = mass * 1.989e30  # kg
        self.radius = radius * 1000  # m
        self.G = 6.67e-11
        self.hbar = 1.055e-34
        self.m_n = 1.67e-27  # 中子质量
        self.e = 1.6e-19
        
        # 核饱和密度
        self.rho_nuclear = 2.8e17  # kg/m^3
        
        logger.info(f"中子星模型: M={mass} M☉, R={radius} km")
    
    def build_crust_structure(self,
                             n_layers: int = 50) -> List[Dict]:
        """
        构建中子星壳层结构
        
        从外层(中子滴出点)到内层(均匀核物质)
        """
        layers = []
        
        # 中子滴出密度
        rho_drip = 4e14  # kg/m^3
        
        # 对数密度网格
        rhos = np.logspace(np.log10(rho_drip), 
                          np.log10(self.rho_nuclear), n_layers)
        
        for i, rho in enumerate(rhos):
            # 计算该深度的半径 (简化模型)
            r = self.radius * (1 - 0.1 * (i / n_layers))
            
            # 压力 (简并压主导)
            P = self._calculate_pressure(rho)
            
            # 温度 (壳层冷却)
            T = 1e8 * (rho / rho_drip)**(-0.25)  # K
            
            # 核组成 (随密度变化)
            if rho < 1e15:
                nuclei = "Fe-56"
                Z, A = 26, 56
            elif rho < 1e16:
                nuclei = "Kr-86 to Mo-100"
                Z, A = 36, 86
            elif rho < 1e17:
                nuclei = "Zr-100 to Sn-150"
                Z, A = 40, 100
            else:
                nuclei = "nuclear_pasta"
                Z, A = 0, 0
            
            # 电子丰度
            Y_e = Z / A if A > 0 else 0.05
            
            # 晶格类型
            if rho < 5e16:
                lattice = "bcc"
            elif rho < 2e17:
                lattice = "rod_phase"  # 核意面
            else:
                lattice = "uniform"
            
            layers.append({
                'radius': r,
                'density': rho,
                'pressure': P,
                'temperature': T,
                'nuclei': nuclei,
                'Z': Z,
                'A': A,
                'Y_e': Y_e,
                'lattice': lattice
            })
        
        return layers
    
    def _calculate_pressure(self, density: float) -> float:
        """
        计算壳层压力
        
        电子简并压 (相对论性)
        """
        # 简化的简并压公式
        rho_6 = density / 1e6
        
        if rho_6 < 1e6:
            # 非相对论性
            P = 1e22 * (rho_6)**(5/3)  # Pa
        else:
            # 极端相对论性
            P = 1.24e25 * rho_6**(4/3)
        
        return P
    
    def calculate_crustal_moment_of_inertia(self) -> float:
        """
        计算壳层的转动惯量
        
        壳层贡献通常占总转动惯量的 1-5%
        """
        I_crust = 0
        
        # 简化的壳层密度剖面
        n_shell = 20
        for i in range(n_shell):
            r_inner = self.radius * (1 - 0.1 * (i+1) / n_shell)
            r_outer = self.radius * (1 - 0.1 * i / n_shell)
            
            rho_avg = 1e15 * (1 + i / n_shell)  # kg/m^3
            
            mass_shell = (4/3) * np.pi * (r_outer**3 - r_inner**3) * rho_avg
            
            # 球壳的转动惯量 (I = (2/3)MR² 近似)
            r_avg = (r_inner + r_outer) / 2
            I_crust += (2/3) * mass_shell * r_avg**2
        
        return I_crust
    
    def calculate_shear_modulus(self, density: float,
                               lattice_spacing: float) -> float:
        """
        计算壳层剪切模量
        
        μ = 0.1194 * (n_i * Z*e)^2 / a
        """
        n_i = density / (56 * 1.66e-27)  # Fe核数密度
        Z = 26
        a = lattice_spacing
        
        mu = 0.1194 * (n_i * Z * self.e)**2 / (4 * np.pi * 8.85e-12 * a)
        
        return mu
    
    def estimate_crustal_breaking_strain(self) -> float:
        """
        估计壳层断裂应变
        
        典型值: 0.001 - 0.1
        """
        # 库仑晶体断裂应变
        # 取决于缺陷密度和温度
        strain = 0.01 + 0.09 * np.random.rand()
        return strain


class NuclearBurningSimulator:
    """核燃烧模拟器"""
    
    def __init__(self):
        self.reactions = self._init_reaction_network()
    
    def _init_reaction_network(self) -> List[NuclearReaction]:
        """初始化反应网络"""
        return [
            # 氢燃烧
            NuclearReaction(['H', 'H'], ['D', 'e+', 'nu_e'], 0.42, 0, 1e7, 1e4),
            NuclearReaction(['D', 'H'], ['He3', 'gamma'], 5.49, 0, 1e6, 1e4),
            NuclearReaction(['He3', 'He3'], ['He4', 'H', 'H'], 12.86, 0, 1e7, 1e5),
            
            # 氦燃烧 (3α)
            NuclearReaction(['He4', 'He4'], ['Be8'], 0.091, 0, 1e8, 1e6),
            NuclearReaction(['Be8', 'He4'], ['C12', 'gamma'], 7.367, 0, 1e8, 1e6),
            
            # 碳燃烧
            NuclearReaction(['C12', 'C12'], ['Ne20', 'He4'], 4.62, 0, 5e8, 1e9),
            NuclearReaction(['C12', 'C12'], ['Na23', 'p'], 2.24, 0, 5e8, 1e9),
            
            # 氧燃烧
            NuclearReaction(['O16', 'O16'], ['Si28', 'He4'], 9.59, 0, 1e9, 1e10),
            NuclearReaction(['O16', 'O16'], ['P31', 'p'], 7.68, 0, 1e9, 1e10),
            
            # 硅燃烧 (准平衡)
            NuclearReaction(['Si28', 'He4'], ['S32', 'gamma'], 6.95, 0, 3e9, 1e12),
        ]
    
    def calculate_burning_rate(self,
                              reaction: NuclearReaction,
                              temperature: float,
                              density: float,
                              composition: Dict[str, float]) -> float:
        """
        计算核燃烧速率
        
        使用简化Arrhenius型速率
        """
        if temperature < reaction.temperature_threshold:
            return 0
        
        if density < reaction.density_threshold:
            return 0
        
        # Gamow峰能量
        E_gamow = 0.5 * reaction.temperature_threshold * 8.617e-5  # eV to K
        
        # 反应率
        rate = reaction.rate * np.exp(-E_gamow / temperature)
        
        # 考虑反应物丰度
        for reactant in reaction.reactants:
            rate *= composition.get(reactant, 0)
        
        return rate
    
    def simulate_shell_burning(self,
                              shell: ShellLayer,
                              time_years: float,
                              dt_years: float = 1) -> Dict:
        """
        模拟壳层核燃烧
        
        用于新星和热核超新星爆发
        """
        logger.info(f"模拟壳层燃烧: {shell.composition.value}")
        
        # 初始成分
        if shell.composition == ShellComposition.HYDROGEN:
            X_H, X_He, X_C, X_O = 0.7, 0.28, 0.02, 0.0
        elif shell.composition == ShellComposition.HELIUM:
            X_H, X_He, X_C, X_O = 0.0, 0.98, 0.02, 0.0
        else:
            X_H, X_He, X_C, X_O = 0.0, 0.0, 0.5, 0.5
        
        n_steps = int(time_years / dt_years)
        
        history = {
            'time': [],
            'X_H': [], 'X_He': [], 'X_C': [], 'X_O': [],
            'luminosity': [],
            'temperature': []
        }
        
        for step in range(n_steps):
            t = step * dt_years
            
            composition = {'H': X_H, 'He': X_He, 'C': X_C, 'O': X_O}
            
            # 能量产生率
            epsilon = 0
            for reaction in self.reactions:
                rate = self.calculate_burning_rate(
                    reaction, shell.temperature, shell.density, composition
                )
                epsilon += rate * reaction.Q_value * 1.6e-13  # MeV to J
            
            # 温度变化 (简化的热平衡)
            shell.temperature += epsilon * dt_years * 365.25 * 24 * 3600 * 1e-10
            
            # 成分演化 (简化)
            if shell.composition == ShellComposition.HYDROGEN:
                X_H -= epsilon * 1e-20
                X_He += epsilon * 1e-20
            
            # 记录
            if step % 100 == 0:
                history['time'].append(t)
                history['X_H'].append(X_H)
                history['X_He'].append(X_He)
                history['X_C'].append(X_C)
                history['X_O'].append(X_O)
                history['luminosity'].append(epsilon * shell.mass)
                history['temperature'].append(shell.temperature)
        
        # 转换为数组
        for key in history:
            history[key] = np.array(history[key])
        
        return history


class CoolingModel:
    """白矮星冷却模型"""
    
    def __init__(self, white_dwarf: WhiteDwarfModel):
        self.wd = white_dwarf
        self.sigma = 5.67e-8  # Stefan-Boltzmann
        self.c = 3e8
        
    def calculate_cooling_track(self,
                               age_range: Tuple[float, float] = (1e6, 1e10)) -> Dict:
        """
        计算冷却轨迹
        
        白矮星冷却: T_eff ∝ t^(-1/4) (简化)
        """
        ages = np.logspace(np.log10(age_range[0]), 
                          np.log10(age_range[1]), 100)
        
        temperatures = []
        luminosities = []
        
        # 参考点 (太阳年龄的DA白矮星)
        T_ref = 10000  # K
        age_ref = 4.6e9  # years
        
        for age in ages:
            # Mestel冷却定律简化
            T_eff = T_ref * (age / age_ref)**(-0.4)
            
            # 光度
            L = 4 * np.pi * self.wd.radius**2 * self.sigma * T_eff**4
            
            temperatures.append(T_eff)
            luminosities.append(L)
        
        return {
            'age_years': ages,
            'effective_temperature': np.array(temperatures),
            'luminosity_Lsun': np.array(luminosities) / 3.828e26
        }
    
    def estimate_crystallization_age(self) -> float:
        """
        估计结晶化年龄
        
        当中心温度低于结晶化温度时发生
        """
        # 简化的结晶化年龄估计
        # 依赖于质量和成分
        
        mass_factor = (self.wd.mass / self.wd.M_sun)**(-2)
        
        age_crystallization = 1e9 * mass_factor  # years
        
        return age_crystallization


# 应用案例: 中子星壳层研究
def neutron_star_crust_example():
    """中子星壳层研究示例"""
    logger.info("=" * 60)
    logger.info("中子星壳层研究示例")
    logger.info("=" * 60)
    
    # 1. 中子星模型
    ns = NeutronStarCrust(mass=1.4, radius=12)
    
    # 2. 构建壳层结构
    crust = ns.build_crust_structure(n_layers=20)
    
    logger.info(f"中子星壳层结构 ({len(crust)}层):")
    for i, layer in enumerate(crust[::5]):  # 每5层显示
        logger.info(f"  层{i*5}: R={layer['radius']/1000:.1f} km, "
                   f"ρ={layer['density']:.2e} kg/m³, "
                   f"P={layer['pressure']:.2e} Pa, "
                   f"核: {layer['nuclei']}")
    
    # 3. 壳层物理参数
    I_crust = ns.calculate_crustal_moment_of_inertia()
    I_total = 0.35 * ns.mass * ns.radius**2  # 中子星总转动惯量
    
    logger.info(f"\n壳层转动惯量: {I_crust:.2e} kg·m²")
    logger.info(f"占总转动惯量: {I_crust/I_total*100:.1f}%")
    
    # 剪切模量
    mu = ns.calculate_shear_modulus(1e15, 1e-14)
    logger.info(f"壳层剪切模量 (ρ=1e15 kg/m³): {mu:.2e} Pa")
    
    # 断裂应变
    strain = ns.estimate_crustal_breaking_strain()
    logger.info(f"估计断裂应变: {strain:.3f}")
    
    # 4. 白矮星模型
    logger.info("\n--- 白矮星模型 ---")
    wd = WhiteDwarfModel(mass=0.6, effective_temperature=15000)
    
    shells = wd.build_shell_structure()
    
    logger.info(f"白矮星壳层 ({len(shells)}层):")
    for shell in shells:
        logger.info(f"  {shell.composition.value}: "
                   f"R={shell.inner_radius/1000:.0f}-{shell.outer_radius/1000:.0f} km, "
                   f"Γ={shell.ion_coupling:.0f}, 状态: {shell.state.value}")
    
    # 5. 结晶化
    L_cryst = wd.calculate_crystallization_luminosity()
    logger.info(f"\n结晶化释放光度: {L_cryst/3.828e26:.2e} L☉")
    
    # 6. 核燃烧
    burning = NuclearBurningSimulator()
    
    if shells:
        he_shell = shells[1] if len(shells) > 1 else shells[0]
        burn_history = burning.simulate_shell_burning(
            he_shell, time_years=1000, dt_years=1
        )
        
        logger.info(f"\n氦壳层燃烧模拟:")
        logger.info(f"  最终He丰度: {burn_history['X_He'][-1]:.3f}")
        logger.info(f"  峰值光度: {np.max(burn_history['luminosity'])/3.828e26:.2e} L☉")
    
    # 7. 冷却轨迹
    cooling = CoolingModel(wd)
    track = cooling.calculate_cooling_track()
    
    logger.info(f"\n白矮星冷却:")
    logger.info(f"  10亿年有效温度: {track['effective_temperature'][50]:.0f} K")
    logger.info(f"  100亿年有效温度: {track['effective_temperature'][-1]:.0f} K")
    
    age_cryst = cooling.estimate_crystallization_age()
    logger.info(f"  预计结晶化年龄: {age_cryst/1e9:.1f} Gyr")
    
    return {
        'neutron_star_crust': crust,
        'white_dwarf_shells': shells,
        'cooling_track': track
    }


if __name__ == "__main__":
    neutron_star_crust_example()
