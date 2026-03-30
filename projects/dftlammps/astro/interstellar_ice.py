"""
星际冰模拟模块
=============
模拟分子云和恒星形成区域的冰相化学，包括：
- 冰相分子形成动力学
- 宇宙射线辐射化学
- 热脱附过程
- 冰层结构和相变

应用场景：
- 原行星盘化学演化
- 彗星成分形成
- 前生命分子合成
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IceComponent(Enum):
    """冰组分枚举"""
    H2O = "water"
    CO = "carbon_monoxide"
    CO2 = "carbon_dioxide"
    CH3OH = "methanol"
    NH3 = "ammonia"
    CH4 = "methane"
    H2CO = "formaldehyde"
    HCOOH = "formic_acid"
    CH3CHO = "acetaldehyde"
    NH2CHO = "formamide"
    HNCO = "isocyanic_acid"
    OCS = "carbonyl_sulfide"


class IcePhase(Enum):
    """冰相态"""
    AMORPHOUS = "amorphous"
    CUBIC = "cubic_crystalline"
    HEXAGONAL = "hexagonal_crystalline"
    POROUS = "porous"
    COMPACT = "compact"


class GrainType(Enum):
    """尘埃颗粒类型"""
    SILICATE = "silicate"
    CARBONACEOUS = "carbonaceous"
    PAH = "polycyclic_aromatic_hydrocarbon"
    MIXED = "mixed_composition"


@dataclass
class AstrochemicalReaction:
    """天体化学反应"""
    reactants: List[str]
    products: List[str]
    rate_constant: float  # cm^3/s (气相) 或 1/s (表面)
    activation_energy: float  # K
    reaction_type: str  # 'gas_phase', 'grain_surface', 'photon', 'cosmic_ray'
    branching_ratio: float = 1.0


@dataclass
class IceLayer:
    """冰层结构"""
    component: IceComponent
    thickness: float  # nm
    density: float    # g/cm^3
    temperature: float  # K
    crystallinity: float  # 0-1
    porosity: float   # 0-1
    coverage: float   # 表面覆盖度 (0-1)


@dataclass
class GrainSurface:
    """尘埃颗粒表面"""
    grain_type: GrainType
    radius: float     # nm
    temperature: float  # K
    ice_layers: List[IceLayer]
    active_sites: int  # 表面活性位点数
    
    def get_surface_area(self) -> float:
        """计算表面积 (cm^2)"""
        return 4 * np.pi * (self.radius * 1e-7)**2
    
    def get_total_ice_thickness(self) -> float:
        """计算总冰层厚度"""
        return sum(layer.thickness for layer in self.ice_layers)


class InterstellarIceSimulator:
    """星际冰模拟器"""
    
    def __init__(self,
                 gas_density: float = 1e4,  # cm^-3
                 gas_temperature: float = 10.0,  # K
                 visual_extinction: float = 10.0,
                 cosmic_ray_ionization_rate: float = 1e-17):  # s^-1
        self.gas_density = gas_density
        self.gas_temperature = gas_temperature
        self.Av = visual_extinction
        self.zeta = cosmic_ray_ionization_rate
        
        # 物理常数
        self.kB = 1.38e-16  # erg/K
        self.h = 6.626e-27  # erg·s
        self.amu = 1.66e-24  # g
        
        self._init_ice_properties()
        self._init_reaction_network()
        
        logger.info(f"星际冰模拟器初始化: n={gas_density:.0e} cm^-3, T={gas_temperature}K")
    
    def _init_ice_properties(self):
        """初始化冰物理性质"""
        self.ice_properties = {
            IceComponent.H2O: {
                'binding_energy': 5770,  # K
                'density': 0.92,  # g/cm^3
                'diffusion_barrier': 2300,  # K (表面扩散)
                'formation_enthalpy': -242,  # kJ/mol
            },
            IceComponent.CO: {
                'binding_energy': 1150,
                'density': 1.03,
                'diffusion_barrier': 400,
                'formation_enthalpy': -110
            },
            IceComponent.CO2: {
                'binding_energy': 2990,
                'density': 1.56,
                'diffusion_barrier': 1100,
                'formation_enthalpy': -393
            },
            IceComponent.CH3OH: {
                'binding_energy': 5530,
                'density': 0.79,
                'diffusion_barrier': 2400,
                'formation_enthalpy': -239
            },
            IceComponent.NH3: {
                'binding_energy': 3060,
                'density': 0.73,
                'diffusion_barrier': 1200,
                'formation_enthalpy': -46
            },
            IceComponent.CH4: {
                'binding_energy': 1090,
                'density': 0.49,
                'diffusion_barrier': 450,
                'formation_enthalpy': -75
            },
        }
    
    def _init_reaction_network(self):
        """初始化反应网络"""
        self.reactions = [
            # 表面反应
            AstrochemicalReaction(['H', 'H'], ['H2'], 1e-12, 0, 'grain_surface'),
            AstrochemicalReaction(['H', 'O'], ['OH'], 1e-10, 0, 'grain_surface'),
            AstrochemicalReaction(['H', 'OH'], ['H2O'], 1e-10, 0, 'grain_surface'),
            AstrochemicalReaction(['O', 'CO'], ['CO2'], 1e-12, 1000, 'grain_surface'),
            AstrochemicalReaction(['H', 'CO'], ['HCO'], 1e-10, 0, 'grain_surface'),
            AstrochemicalReaction(['H', 'HCO'], ['H2CO'], 1e-10, 0, 'grain_surface'),
            AstrochemicalReaction(['H', 'H2CO'], ['CH3O'], 1e-10, 0, 'grain_surface'),
            AstrochemicalReaction(['H', 'CH3O'], ['CH3OH'], 1e-10, 0, 'grain_surface'),
            AstrochemicalReaction(['N', 'H'], ['NH'], 1e-10, 0, 'grain_surface'),
            AstrochemicalReaction(['NH', 'H'], ['NH2'], 1e-10, 0, 'grain_surface'),
            AstrochemicalReaction(['NH2', 'H'], ['NH3'], 1e-10, 0, 'grain_surface'),
            
            # 宇宙射线诱导反应
            AstrochemicalReaction(['H2O'], ['OH', 'H'], 1e-15, 0, 'cosmic_ray'),
            AstrochemicalReaction(['CH3OH'], ['CH2OH', 'H'], 1e-16, 0, 'cosmic_ray'),
            
            # 光化学反应
            AstrochemicalReaction(['H2O'], ['OH', 'H'], 1e-10, 0, 'photon'),
            AstrochemicalReaction(['CO'], ['C', 'O'], 1e-10, 0, 'photon'),
        ]
    
    def calculate_accretion_rate(self,
                                 species: str,
                                 gas_abundance: float,
                                 grain_surface: GrainSurface) -> float:
        """
        计算气体在颗粒表面的吸积速率
        
        R_acc = n_gas * v_th * sigma_grain * S
        """
        # 热速度
        mass = self._get_species_mass(species)
        v_thermal = np.sqrt(8 * self.kB * self.gas_temperature / (np.pi * mass))
        
        # 粘着系数 (sticking coefficient)
        S = 1.0  # 低温下通常约等于1
        
        rate = (gas_abundance * self.gas_density * v_thermal * 
                grain_surface.get_surface_area() * S)
        
        return rate  # molecules/s
    
    def _get_species_mass(self, species: str) -> float:
        """获取物种质量 (g)"""
        masses = {
            'H': 1, 'H2': 2, 'C': 12, 'N': 14, 'O': 16,
            'CO': 28, 'CO2': 44, 'H2O': 18, 'CH3OH': 32,
            'NH3': 17, 'CH4': 16, 'OH': 17, 'HCO': 29
        }
        return masses.get(species, 16) * self.amu
    
    def calculate_thermal_desorption_rate(self,
                                         component: IceComponent,
                                         grain_temperature: float) -> float:
        """
        计算热脱附速率
        
        k_des = ν * exp(-E_bind / T)
        
        Args:
            component: 冰组分
            grain_temperature: 颗粒温度 (K)
        
        Returns:
            脱附速率 (s^-1)
        """
        props = self.ice_properties.get(component, {})
        E_bind = props.get('binding_energy', 1000)  # K
        
        # 尝试频率 (~10^13 s^-1)
        nu = 1e13
        
        rate = nu * np.exp(-E_bind / grain_temperature)
        
        return rate
    
    def calculate_diffusion_rate(self,
                                component: IceComponent,
                                grain_temperature: float) -> float:
        """
        计算表面扩散速率
        
        k_diff = ν * exp(-E_diff / T)
        """
        props = self.ice_properties.get(component, {})
        E_diff = props.get('diffusion_barrier', 500)  # K
        
        nu = 1e13
        rate = nu * np.exp(-E_diff / grain_temperature)
        
        return rate
    
    def simulate_ice_evolution(self,
                              initial_composition: Dict[IceComponent, float],
                              grain_surface: GrainSurface,
                              time_years: float,
                              dt_years: float = 100) -> Dict:
        """
        模拟冰成分演化
        
        Args:
            initial_composition: 初始成分 (相对丰度)
            grain_surface: 颗粒表面
            time_years: 模拟时间
            dt_years: 时间步长
        
        Returns:
            演化历史
        """
        logger.info(f"模拟冰演化: {time_years:.0e}年")
        
        # 初始化表面丰度 (单层数)
        surface_coverage = {comp: init * 10 for comp, init in initial_composition.items()}
        
        n_steps = int(time_years / dt_years)
        history = {comp: [] for comp in IceComponent}
        history['time'] = []
        
        for step in range(n_steps):
            time = step * dt_years
            
            # 吸积过程
            for comp in IceComponent:
                gas_abundance = 1e-4 if comp == IceComponent.H2O else 1e-6
                accretion = self.calculate_accretion_rate(
                    comp.name, gas_abundance, grain_surface
                ) * dt_years * 365.25 * 24 * 3600 / 1e6  # 转换为单层
                
                surface_coverage[comp] = surface_coverage.get(comp, 0) + accretion
            
            # 表面化学反应
            surface_coverage = self._surface_chemistry(
                surface_coverage, grain_surface.temperature, dt_years
            )
            
            # 热脱附
            for comp in list(surface_coverage.keys()):
                desorption = self.calculate_thermal_desorption_rate(
                    comp, grain_surface.temperature
                ) * dt_years * 365.25 * 24 * 3600
                
                surface_coverage[comp] *= np.exp(-desorption)
            
            # 记录历史
            if step % max(1, n_steps // 100) == 0:
                history['time'].append(time)
                for comp in IceComponent:
                    history[comp].append(surface_coverage.get(comp, 0))
        
        # 转换为数组
        for key in history:
            history[key] = np.array(history[key])
        
        return history
    
    def _surface_chemistry(self,
                          coverage: Dict[IceComponent, float],
                          temperature: float,
                          dt_years: float) -> Dict[IceComponent, float]:
        """表面化学反应"""
        new_coverage = coverage.copy()
        dt_s = dt_years * 365.25 * 24 * 3600
        
        for reaction in self.reactions:
            if reaction.reaction_type != 'grain_surface':
                continue
            
            # 检查反应物是否存在
            reactant_components = []
            for r in reaction.reactants:
                found = False
                for comp in IceComponent:
                    if comp.name == r.lower() or comp.value == r.lower():
                        reactant_components.append(comp)
                        found = True
                        break
                if not found:
                    break
            else:
                # 计算反应速率
                k = reaction.rate_constant * np.exp(-reaction.activation_energy / temperature)
                
                # 简化的速率计算
                rate = k * min([coverage.get(r, 0) for r in reactant_components]) * dt_s
                
                # 更新覆盖度
                for r in reactant_components:
                    new_coverage[r] = max(0, new_coverage.get(r, 0) - rate)
        
        return new_coverage
    
    def calculate_infrared_spectrum(self,
                                   ice_composition: Dict[IceComponent, float],
                                   temperature: float = 10.0) -> Dict[str, np.ndarray]:
        """
        计算冰的红外吸收光谱
        
        Returns:
            波数和吸光度
        """
        # 冰的特征红外带 (cm^-1)
        ir_bands = {
            IceComponent.H2O: [(3280, 100, 200), (1650, 20, 50), (800, 10, 30)],  # OH stretch, bend, libration
            IceComponent.CO: [(2139, 5, 20)],
            IceComponent.CO2: [(2340, 10, 40), (660, 5, 20)],
            IceComponent.CH3OH: [(3265, 15, 60), (2825, 10, 40), (1125, 8, 30)],
            IceComponent.NH3: [(3370, 20, 80), (1070, 5, 25)],
            IceComponent.CH4: [(3010, 5, 20), (1300, 3, 15)],
        }
        
        wavenumbers = np.linspace(500, 4000, 1000)
        absorbance = np.zeros_like(wavenumbers)
        
        for component, abundance in ice_composition.items():
            if component in ir_bands:
                for band_center, width, strength in ir_bands[component]:
                    # 高斯线型
                    band = strength * abundance * np.exp(
                        -((wavenumbers - band_center) / width)**2
                    )
                    absorbance += band
        
        return {
            'wavenumber_cm-1': wavenumbers,
            'absorbance': absorbance
        }
    
    def calculate_desorption_temperature(self,
                                        component: IceComponent,
                                        heating_rate: float = 1.0) -> float:
        """
        计算程序升温脱附温度
        
        使用Redhead方程近似
        T_des = E_bind / (R * ln(ν * E_bind / (β * R * T_des^2)))
        
        Args:
            component: 冰组分
            heating_rate: 升温速率 (K/s)
        
        Returns:
            峰值脱附温度 (K)
        """
        props = self.ice_properties.get(component, {})
        E_bind = props.get('binding_energy', 1000)
        
        # Redhead方程迭代求解
        T_guess = E_bind / 30  # 初始猜测
        
        for _ in range(10):
            T_new = E_bind / np.log(
                1e13 * E_bind / (heating_rate * T_guess**2)
            )
            if abs(T_new - T_guess) < 0.1:
                break
            T_guess = T_new
        
        return T_guess
    
    def estimate_complex_organic_formation(self,
                                        ice_composition: Dict[IceComponent, float],
                                        irradiation_dose: float,  # eV/molecule
                                        temperature: float = 10.0) -> Dict[str, float]:
        """
        估计复杂有机分子形成
        
        Args:
            ice_composition: 冰成分
            irradiation_dose: 辐射剂量
            temperature: 温度
        
        Returns:
            COMs丰度
        """
        # 复杂有机分子 (COMs)
        coms = {
            'glycine': 0.0,
            'alanine': 0.0,
            'glycolaldehyde': 0.0,
            'ethylene_glycol': 0.0,
            'formamide': 0.0,
            'acetamide': 0.0
        }
        
        # 基于H2O:CH3OH:CO比例估计COMs形成
        h2o_abundance = ice_composition.get(IceComponent.H2O, 0)
        ch3oh_abundance = ice_composition.get(IceComponent.CH3OH, 0)
        co_abundance = ice_composition.get(IceComponent.CO, 0)
        
        # 辐射化学产额 (简化模型)
        G_factor = irradiation_dose / 100  # 归一化
        
        # COMs产额 (相对于CH3OH)
        if ch3oh_abundance > 0:
            coms['glycolaldehyde'] = 0.01 * ch3oh_abundance * G_factor
            coms['ethylene_glycol'] = 0.005 * ch3oh_abundance * G_factor
            coms['formamide'] = 0.008 * h2o_abundance * G_factor
        
        return coms


# 应用案例演示
def interstellar_ice_application():
    """星际冰应用示例"""
    logger.info("=" * 60)
    logger.info("星际冰应用示例")
    logger.info("=" * 60)
    
    # 1. 初始化模拟器
    simulator = InterstellarIceSimulator(
        gas_density=1e4,  # cm^-3
        gas_temperature=10.0,  # K
        visual_extinction=10.0,
        cosmic_ray_ionization_rate=1e-17
    )
    
    # 2. 创建尘埃颗粒表面
    grain = GrainSurface(
        grain_type=GrainType.SILICATE,
        radius=100,  # nm
        temperature=15.0,  # K
        ice_layers=[],
        active_sites=1000
    )
    
    logger.info(f"尘埃颗粒: {grain.radius} nm, {grain.grain_type.value}")
    logger.info(f"表面积: {grain.get_surface_area():.2e} cm²")
    
    # 3. 吸积速率计算
    logger.info(f"\n吸积速率:")
    for species in ['H2O', 'CO', 'CO2']:
        acc_rate = simulator.calculate_accretion_rate(
            species, 1e-4 if species == 'H2O' else 1e-6, grain
        )
        logger.info(f"  {species}: {acc_rate:.2e} molecules/s")
    
    # 4. 脱附温度计算
    logger.info(f"\n热脱附温度 (升温速率1K/s):")
    for component in [IceComponent.H2O, IceComponent.CO, IceComponent.CO2, IceComponent.CH3OH]:
        t_des = simulator.calculate_desorption_temperature(component, 1.0)
        props = simulator.ice_properties.get(component, {})
        logger.info(f"  {component.value}: {t_des:.0f} K (结合能: {props.get('binding_energy', 0)} K)")
    
    # 5. 冰演化模拟
    initial_comp = {
        IceComponent.H2O: 1.0,
        IceComponent.CO: 0.3,
        IceComponent.CO2: 0.1
    }
    
    evolution = simulator.simulate_ice_evolution(
        initial_comp, grain, time_years=1e5, dt_years=100
    )
    
    logger.info(f"\n冰演化模拟 ({len(evolution['time'])}时间步):")
    logger.info(f"  H2O最终丰度: {evolution[IceComponent.H2O][-1]:.2f} 单层")
    logger.info(f"  CO最终丰度: {evolution[IceComponent.CO][-1]:.2f} 单层")
    
    # 6. 红外光谱
    final_composition = {
        IceComponent.H2O: evolution[IceComponent.H2O][-1],
        IceComponent.CO: evolution[IceComponent.CO][-1],
        IceComponent.CO2: evolution[IceComponent.CO2][-1]
    }
    
    ir_spectrum = simulator.calculate_infrared_spectrum(final_composition)
    
    # 找主要吸收峰
    peaks = []
    for i in range(1, len(ir_spectrum['absorbance']) - 1):
        if (ir_spectrum['absorbance'][i] > ir_spectrum['absorbance'][i-1] and
            ir_spectrum['absorbance'][i] > ir_spectrum['absorbance'][i+1] and
            ir_spectrum['absorbance'][i] > 10):
            peaks.append((ir_spectrum['wavenumber_cm-1'][i], 
                         ir_spectrum['absorbance'][i]))
    
    logger.info(f"\n红外光谱主要吸收峰:")
    for peak, intensity in peaks[:5]:
        logger.info(f"  {peak:.0f} cm^-1: {intensity:.1f}")
    
    # 7. 复杂有机分子形成
    coms = simulator.estimate_complex_organic_formation(
        final_composition,
        irradiation_dose=50,  # eV/molecule
        temperature=15.0
    )
    
    logger.info(f"\n复杂有机分子(COMs)形成估计:")
    for molecule, abundance in coms.items():
        if abundance > 0:
            logger.info(f"  {molecule}: {abundance:.2e} 相对丰度")
    
    return {
        'evolution': evolution,
        'ir_spectrum': ir_spectrum,
        'coms': coms
    }


if __name__ == "__main__":
    interstellar_ice_application()
