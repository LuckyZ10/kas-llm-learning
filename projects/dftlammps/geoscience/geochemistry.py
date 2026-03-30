"""
地球化学模拟模块
===============
地球化学过程模拟，包括：
- 岩浆分异结晶
- 流体-岩石相互作用
- 元素分配系数计算
- 同位素分馏

应用场景：
- 成矿预测
- 火山活动研究
- 地热系统分析
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MagmaType(Enum):
    """岩浆类型"""
    BASALT = "basalt"
    ANDESITE = "andesite"
    DACITE = "dacite"
    RHYOLITE = "rhyolite"
    KOMATIITE = "komatiite"
    PICRITE = "picrite"
    PHONOLITE = "phonolite"


class ReservoirType(Enum):
    """储库类型"""
    MANTLE = "mantle"
    CRUST = "crust"
    OCEAN = "ocean"
    ATMOSPHERE = "atmosphere"
    SEDIMENT = "sediment"


class ElementCategory(Enum):
    """元素地球化学分类"""
    COMPATIBLE = "compatible"      # 优先进入固体
    INCOMPATIBLE = "incompatible"  # 优先进入熔体
    HIGHLY_INCOMPATIBLE = "highly_incompatible"
    MODERATELY_INCOMPATIBLE = "moderately_incompatible"


@dataclass
class ElementProperties:
    """元素地球化学性质"""
    symbol: str
    atomic_number: int
    atomic_mass: float
    ionic_radius: Dict[str, float]  # 配位数: 半径(Å)
    electronegativity: float
    partition_coefficients: Dict[str, float]  # 矿物: 分配系数
    category: ElementCategory
    isotope_systems: List[str]


@dataclass
class MineralMeltSystem:
    """矿物-熔体系统"""
    mineral: str
    melt_composition: Dict[str, float]
    temperature: float    # K
    pressure: float       # GPa
    oxygen_fugacity: float  # log(fO2)
    water_content: float  # wt%


@dataclass
class FluidComposition:
    """流体成分"""
    ph: float
    salinity: float     # wt% NaCl eq
    temperature: float  # K
    components: Dict[str, float]  # mol/kg


class PartitionCoefficientDatabase:
    """元素分配系数数据库"""
    
    def __init__(self):
        self.partition_coeffs: Dict[str, Dict[str, float]] = {}
        self._init_partition_data()
    
    def _init_partition_data(self):
        """初始化分配系数数据"""
        # 典型分配系数 (D = C_solid / C_melt)
        self.partition_coeffs = {
            # 橄榄石
            'olivine': {
                'Ni': 10.0, 'Co': 5.0, 'Mn': 1.0, 'Mg': 3.0,
                'Fe': 1.5, 'Ca': 0.01, 'Al': 0.01, 'Si': 0.8,
                'Ti': 0.01, 'Na': 0.01, 'K': 0.001, 'Rb': 0.0001,
                'Sr': 0.0001, 'Ba': 0.0001, 'La': 0.0001, 'Ce': 0.0001,
                'Nd': 0.0001, 'Sm': 0.0001, 'Eu': 0.0001, 'Yb': 0.0001,
                'Lu': 0.0001, 'Sc': 0.3, 'V': 0.5, 'Cr': 2.0
            },
            # 斜方辉石
            'orthopyroxene': {
                'Ni': 5.0, 'Co': 3.0, 'Mg': 2.0, 'Fe': 1.2,
                'Ca': 0.1, 'Al': 0.1, 'Si': 1.0, 'Ti': 0.1,
                'Na': 0.05, 'K': 0.001, 'Rb': 0.0001, 'Sr': 0.001,
                'La': 0.001, 'Ce': 0.002, 'Nd': 0.005, 'Sm': 0.01,
                'Eu': 0.01, 'Yb': 0.05, 'Lu': 0.05, 'Sc': 2.0
            },
            # 单斜辉石
            'clinopyroxene': {
                'Ni': 2.0, 'Co': 2.0, 'Mg': 1.5, 'Fe': 1.0,
                'Ca': 2.0, 'Al': 0.3, 'Si': 1.0, 'Ti': 0.5,
                'Na': 0.3, 'K': 0.01, 'Rb': 0.001, 'Sr': 0.1,
                'La': 0.1, 'Ce': 0.2, 'Nd': 0.4, 'Sm': 0.6,
                'Eu': 0.6, 'Yb': 1.0, 'Lu': 1.2, 'Sc': 3.0
            },
            # 石榴石
            'garnet': {
                'Ni': 0.5, 'Co': 1.0, 'Mg': 0.8, 'Fe': 1.5,
                'Ca': 1.5, 'Al': 2.0, 'Si': 1.0, 'Ti': 0.5,
                'Na': 0.01, 'K': 0.001, 'Rb': 0.0001, 'Sr': 0.001,
                'La': 0.001, 'Ce': 0.005, 'Nd': 0.05, 'Sm': 0.2,
                'Eu': 0.3, 'Yb': 15.0, 'Lu': 20.0, 'Y': 10.0,
                'Sc': 5.0, 'V': 2.0, 'Cr': 3.0
            },
            # 斜长石
            'plagioclase': {
                'Ni': 0.01, 'Co': 0.01, 'Mg': 0.01, 'Fe': 0.01,
                'Ca': 2.0, 'Al': 2.0, 'Si': 1.0, 'Ti': 0.01,
                'Na': 2.5, 'K': 0.3, 'Rb': 0.1, 'Sr': 2.0,
                'La': 0.5, 'Ce': 0.4, 'Nd': 0.2, 'Sm': 0.1,
                'Eu': 2.0, 'Yb': 0.02, 'Lu': 0.01
            },
            # 云母
            'mica': {
                'Rb': 3.0, 'K': 3.0, 'Ba': 5.0, 'Sr': 0.1,
                'Ni': 3.0, 'Li': 5.0, 'Cs': 10.0
            }
        }
        
        logger.info(f"分配系数数据库: {len(self.partition_coeffs)}种矿物")
    
    def get_partition_coefficient(self, element: str, mineral: str,
                                   temperature: Optional[float] = None) -> float:
        """
        获取分配系数
        
        Args:
            element: 元素符号
            mineral: 矿物名称
            temperature: 温度 (K)，用于温度修正
        
        Returns:
            分配系数 D
        """
        if mineral not in self.partition_coeffs:
            return 0.1  # 默认值
        
        D = self.partition_coeffs[mineral].get(element, 0.1)
        
        # 温度修正 (Blundy-Wood模型简化)
        if temperature is not None:
            T_ref = 1473  # K
            D = D * np.exp(-5000 * (1/temperature - 1/T_ref))
        
        return D
    
    def calculate_bulk_partition_coefficient(self,
                                            element: str,
                                            mineral_modes: Dict[str, float],
                                            temperature: float) -> float:
        """
        计算总体分配系数
        
        D_bulk = Σ(X_i * D_i)
        
        Args:
            element: 元素符号
            mineral_modes: 矿物模式丰度 (总和=1)
            temperature: 温度 (K)
        
        Returns:
            总体分配系数
        """
        D_bulk = 0.0
        for mineral, mode in mineral_modes.items():
            D_i = self.get_partition_coefficient(element, mineral, temperature)
            D_bulk += mode * D_i
        
        return D_bulk


class FractionalCrystallizationModel:
    """分离结晶模型"""
    
    def __init__(self, partition_db: PartitionCoefficientDatabase):
        self.partition_db = partition_db
    
    def rayleigh_fractionation(self,
                              initial_concentration: float,
                              partition_coefficient: float,
                              fraction_remaining: float) -> float:
        """
        Rayleigh分馏方程
        
        C_liquid = C0 * F^(D-1)
        
        Args:
            initial_concentration: 初始浓度
            partition_coefficient: 分配系数 D
            fraction_remaining: 剩余熔体比例 F
        >Returns:
            熔体中元素浓度
        """
        return initial_concentration * (fraction_remaining ** (partition_coefficient - 1))
    
    def simulate_fractional_crystallization(self,
                                           initial_magma: Dict[str, float],
                                           crystallizing_assemblage: Dict[str, float],
                                           steps: int = 50) -> Dict[str, np.ndarray]:
        """
        模拟分离结晶过程
        
        Args:
            initial_magma: 初始岩浆成分
            crystallizing_assemblage: 结晶矿物组合
            steps: 计算步数
        
        Returns:
            演化轨迹
        """
        logger.info("模拟分离结晶...")
        
        F_values = np.linspace(1.0, 0.1, steps)  # 从100%到10%熔体
        
        results = {'F': F_values}
        
        # 对每个元素进行计算
        for element in ['SiO2', 'MgO', 'FeO', 'CaO', 'Al2O3', 'Na2O', 'K2O',
                       'Ni', 'Cr', 'Rb', 'Sr', 'La', 'Ce', 'Nd', 'Sm', 'Eu', 'Yb']:
            
            C0 = initial_magma.get(element, 1.0)
            
            # 计算总体分配系数
            D_bulk = self.partition_db.calculate_bulk_partition_coefficient(
                element, crystallizing_assemblage, 1473
            )
            
            # Rayleigh分馏
            concentrations = []
            for F in F_values:
                C = self.rayleigh_fractionation(C0, D_bulk, F)
                concentrations.append(C)
            
            results[element] = np.array(concentrations)
        
        # 计算镁数 Mg#
        results['Mg_number'] = results.get('MgO', np.zeros(steps)) / \
                              (results.get('MgO', np.zeros(steps)) + 
                               results.get('FeO', np.zeros(steps))) * 100
        
        return results
    
    def calculate_cumulate_composition(self,
                                      liquid_composition: Dict[str, float],
                                      crystallizing_minerals: Dict[str, float],
                                      temperature: float) -> Dict[str, float]:
        """
        计算堆晶岩成分
        
        Args:
            liquid_composition: 熔体成分
            crystallizing_minerals: 结晶矿物
            temperature: 温度
        
        Returns:
            堆晶岩成分
        """
        cumulate = {}
        
        for element in liquid_composition.keys():
            # 堆晶 = 各矿物的加权平均
            element_in_cumulate = 0
            for mineral, proportion in crystallizing_minerals.items():
                D = self.partition_db.get_partition_coefficient(element, mineral, temperature)
                element_in_cumulate += proportion * D * liquid_composition[element]
            
            cumulate[element] = element_in_cumulate
        
        return cumulate


class FluidRockInteraction:
    """流体-岩石相互作用模型"""
    
    def __init__(self, partition_db: PartitionCoefficientDatabase):
        self.partition_db = partition_db
        self.solubility_constants = self._init_solubility()
    
    def _init_solubility(self) -> Dict[str, float]:
        """初始化溶解度常数"""
        return {
            'SiO2': 0.01,   # mol/kg
            'Al2O3': 0.001,
            'FeO': 0.005,
            'MgO': 0.01,
            'CaO': 0.02,
            'Na2O': 0.1,
            'K2O': 0.1,
            'Cu': 0.001,
            'Au': 1e-6,
            'Ag': 0.0001,
            'Zn': 0.001,
            'Pb': 0.0005
        }
    
    def calculate_fluid_rock_ratio(self,
                                  rock_composition: Dict[str, float],
                                  fluid_composition: FluidComposition,
                                  reacted_rock: Dict[str, float]) -> float:
        """
        计算流体/岩石比例
        
        使用质量平衡方程
        """
        # 简化的计算
        # 基于主要元素的迁移
        ratios = []
        for element in ['SiO2', 'CaO', 'Na2O', 'K2O']:
            if element in rock_composition and element in reacted_rock:
                loss = rock_composition[element] - reacted_rock[element]
                if loss > 0 and element in fluid_composition.components:
                    ratio = loss / fluid_composition.components[element]
                    ratios.append(ratio)
        
        return np.mean(ratios) if ratios else 1.0
    
    def simulate_hydrothermal_alteration(self,
                                        host_rock: Dict[str, float],
                                        fluid: FluidComposition,
                                        duration_years: float,
                                        temperature_gradient: float = 50) -> Dict:
        """
        模拟热液蚀变
        
        Args:
            host_rock: 原岩成分
            fluid: 流体成分
            duration_years: 持续时间
            temperature_gradient: 温度梯度 (K/km)
        
        Returns:
            蚀变结果
        """
        logger.info(f"模拟热液蚀变: {duration_years}年")
        
        # 温度历史
        max_temp = fluid.temperature
        time_steps = int(duration_years)
        temperatures = max_temp - np.linspace(0, 200, time_steps)  # 冷却
        
        # 蚀变矿物组合随温度变化
        alteration_minerals = {
            'high_temp': {'epidote': 0.3, 'chlorite': 0.4, 'actinolite': 0.3},
            'medium_temp': {'chlorite': 0.5, 'sericite': 0.3, 'albite': 0.2},
            'low_temp': {'sericite': 0.4, 'quartz': 0.3, 'clay': 0.3}
        }
        
        # 元素淋滤
        leached_elements = {}
        for element in host_rock.keys():
            # 温度越高，淋滤越强
            leaching_factor = 0.1 + 0.3 * (max_temp - 300) / 1000
            leached_elements[element] = host_rock[element] * leaching_factor
        
        # 成矿元素富集
        ore_enrichment = self._calculate_ore_enrichment(
            host_rock, fluid, duration_years
        )
        
        return {
            'original_rock': host_rock,
            'altered_rock': {k: v - leached_elements.get(k, 0) 
                           for k, v in host_rock.items()},
            'leached_elements': leached_elements,
            'alteration_minerals': alteration_minerals,
            'temperature_history': temperatures,
            'ore_enrichment': ore_enrichment,
            'alteration_index': sum(leached_elements.values()) / sum(host_rock.values())
        }
    
    def _calculate_ore_enrichment(self,
                                  rock: Dict[str, float],
                                  fluid: FluidComposition,
                                  duration: float) -> Dict[str, float]:
        """计算成矿元素富集"""
        enrichment = {}
        
        ore_elements = ['Cu', 'Au', 'Ag', 'Zn', 'Pb', 'Mo', 'W', 'Sn']
        
        for element in ore_elements:
            if element in fluid.components:
                # 富集 = 流体浓度 * 时间因子 * 沉淀效率
                precip_efficiency = 0.3  # 30%沉淀
                time_factor = min(duration / 1000, 1.0)
                
                enrichment[element] = (fluid.components[element] * 
                                      time_factor * precip_efficiency)
        
        return enrichment
    
    def calculate_metasomatic_front(self,
                                   initial_profile: np.ndarray,
                                   diffusion_coefficient: float,
                                   time: float,
                                   fluid_velocity: float) -> np.ndarray:
        """
        计算交代前锋推进
        
        使用对流-扩散方程
        """
        # 简化的解析解
        x = np.linspace(0, 10, len(initial_profile))  # meters
        
        # Peclet数
        Pe = fluid_velocity * x[-1] / diffusion_coefficient
        
        # 浓度剖面
        if Pe > 1:
            # 对流主导
            profile = initial_profile * np.exp(-x / (fluid_velocity * time))
        else:
            # 扩散主导
            profile = initial_profile * (1 - 0.5 * (1 + np.tanh((x - fluid_velocity * time) / 
                                                               np.sqrt(4 * diffusion_coefficient * time))))
        
        return profile


class IsotopeSystem:
    """同位素体系"""
    
    def __init__(self):
        self.decay_constants = {
            'Rb87': 1.42e-11,  # yr^-1
            'Sm147': 6.54e-12,
            'Lu176': 1.87e-11,
            'Re187': 1.67e-11,
            'K40': 5.54e-10,
            'U238': 1.55e-10,
            'U235': 9.85e-10,
            'Th232': 4.95e-11
        }
    
    def calculate_radiogenic_growth(self,
                                   parent_initial: float,
                                   daughter_initial: float,
                                   decay_constant: float,
                                   time: float) -> Tuple[float, float]:
        """
        计算放射成因同位素增长
        
        N_daughter = N_daughter_initial + N_parent * (e^(λt) - 1)
        """
        parent_remaining = parent_initial * np.exp(-decay_constant * time)
        daughter_radiogenic = parent_initial * (1 - np.exp(-decay_constant * time))
        daughter_total = daughter_initial + daughter_radiogenic
        
        return parent_remaining, daughter_total
    
    def calculate_model_age(self,
                           parent_daughter_ratio: float,
                           isotopic_ratio: float,
                           initial_ratio: float,
                           decay_constant: float) -> float:
        """
        计算模式年龄
        
        t = 1/λ * ln(1 + (D/Ds - D0/Ds) / P/Ds)
        """
        age = (1 / decay_constant) * np.log(
            1 + (isotopic_ratio - initial_ratio) / parent_daughter_ratio
        )
        return age
    
    def sm_nd_system(self,
                    sm_ppm: float,
                    nd_ppm: float,
                    initial_ratio: float,
                    time_years: float) -> Dict:
        """Sm-Nd同位素体系"""
        decay_const = self.decay_constants['Sm147']
        
        sm144 = sm_ppm / 1.0  # 简化
        nd144 = nd_ppm / 1.0
        
        sm147_nd144 = (sm_ppm / nd_ppm) * (0.15 / 0.53)  # 同位素丰度校正
        
        # 现在比值
        nd143_nd144_now = initial_ratio + sm147_nd144 * (np.exp(decay_const * time_years) - 1)
        
        return {
            'sm147_nd144': sm147_nd144,
            'nd143_nd144': nd143_nd144_now,
            'epsilon_nd': ((nd143_nd144_now / 0.512638) - 1) * 10000,
            'model_age_Ma': time_years / 1e6
        }


# 应用案例演示
def geochemistry_application_example():
    """地球化学应用示例"""
    logger.info("=" * 60)
    logger.info("地球化学应用示例")
    logger.info("=" * 60)
    
    # 1. 初始化数据库
    partition_db = PartitionCoefficientDatabase()
    
    # 2. 分配系数查询
    logger.info(f"\n分配系数示例:")
    for mineral in ['olivine', 'clinopyroxene', 'garnet']:
        D_ni = partition_db.get_partition_coefficient('Ni', mineral)
        D_rb = partition_db.get_partition_coefficient('Rb', mineral)
        logger.info(f"  {mineral}: D(Ni)={D_ni:.2f}, D(Rb)={D_rb:.4f}")
    
    # 3. 总体分配系数
    mantle_mode = {
        'olivine': 0.60,
        'orthopyroxene': 0.25,
        'clinopyroxene': 0.10,
        'garnet': 0.05
    }
    
    D_bulk_ni = partition_db.calculate_bulk_partition_coefficient(
        'Ni', mantle_mode, 1500
    )
    D_bulk_rb = partition_db.calculate_bulk_partition_coefficient(
        'Rb', mantle_mode, 1500
    )
    logger.info(f"\n地幔总体分配系数:")
    logger.info(f"  D_bulk(Ni)={D_bulk_ni:.2f} (相容元素)")
    logger.info(f"  D_bulk(Rb)={D_bulk_rb:.4f} (不相容元素)")
    
    # 4. 分离结晶模拟
    fc_model = FractionalCrystallizationModel(partition_db)
    
    # 初始玄武质岩浆
    basalt = {
        'SiO2': 50.0, 'TiO2': 1.5, 'Al2O3': 15.0, 'FeO': 10.0,
        'MgO': 8.0, 'CaO': 10.0, 'Na2O': 2.5, 'K2O': 0.5,
        'Ni': 100, 'Cr': 300, 'Rb': 5, 'Sr': 300, 'La': 10
    }
    
    # 结晶矿物组合 (橄榄石+辉石+斜长石)
    crystallizing = {
        'olivine': 0.30,
        'clinopyroxene': 0.40,
        'plagioclase': 0.30
    }
    
    fc_results = fc_model.simulate_fractional_crystallization(
        basalt, crystallizing, steps=50
    )
    
    logger.info(f"\n分离结晶演化:")
    logger.info(f"  初始MgO: {fc_results['MgO'][0]:.1f}% -> 最终: {fc_results['MgO'][-1]:.1f}%")
    logger.info(f"  初始Ni: {fc_results['Ni'][0]:.0f}ppm -> 最终: {fc_results['Ni'][-1]:.0f}ppm")
    logger.info(f"  初始Rb: {fc_results['Rb'][0]:.0f}ppm -> 最终: {fc_results['Rb'][-1]:.0f}ppm")
    logger.info(f"  Mg#: {fc_results['Mg_number'][0]:.1f} -> {fc_results['Mg_number'][-1]:.1f}")
    
    # 5. 流体-岩石相互作用
    fluid_rock = FluidRockInteraction(partition_db)
    
    # 花岗岩成分
    granite = {
        'SiO2': 70.0, 'Al2O3': 15.0, 'K2O': 5.0, 'Na2O': 3.0,
        'CaO': 2.0, 'FeO': 2.0, 'MgO': 1.0, 'Cu': 50, 'Au': 0.001
    }
    
    # 热液流体
    hydrothermal_fluid = FluidComposition(
        ph=4.5,
        salinity=10.0,
        temperature=573,
        components={'Cu': 0.01, 'Au': 1e-5, 'SiO2': 0.02, 'NaCl': 0.5}
    )
    
    alteration = fluid_rock.simulate_hydrothermal_alteration(
        granite, hydrothermal_fluid, duration_years=10000
    )
    
    logger.info(f"\n热液蚀变结果:")
    logger.info(f"  蚀变指数: {alteration['alteration_index']:.3f}")
    logger.info(f"  蚀变矿物: {list(alteration['alteration_minerals'].keys())}")
    logger.info(f"  成矿富集: {alteration['ore_enrichment']}")
    
    # 6. 同位素体系
    isotope = IsotopeSystem()
    
    # Sm-Nd体系
    sm_nd = isotope.sm_nd_system(
        sm_ppm=5.0,
        nd_ppm=25.0,
        initial_ratio=0.511,
        time_years=2e9
    )
    
    logger.info(f"\nSm-Nd同位素体系:")
    logger.info(f"  ¹⁴⁷Sm/¹⁴⁴Nd: {sm_nd['sm147_nd144']:.4f}")
    logger.info(f"  ¹⁴³Nd/¹⁴⁴Nd: {sm_nd['nd143_nd144']:.6f}")
    logger.info(f"  εNd: {sm_nd['epsilon_nd']:.1f}")
    
    return {
        'partition_coefficients': {
            'Ni': D_bulk_ni,
            'Rb': D_bulk_rb
        },
        'fractional_crystallization': fc_results,
        'alteration': alteration,
        'isotope': sm_nd
    }


if __name__ == "__main__":
    geochemistry_application_example()
