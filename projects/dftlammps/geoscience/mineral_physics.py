"""
矿物物理模块
============
地球深部矿物的物理性质计算，包括：
- 高温高压下的弹性性质
- 相变与相图
- 热力学性质
- 地震波速预测

应用场景：
- 地球深部结构探测
- 地幔对流模拟
- 矿物资源勘探
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MineralPhase(Enum):
    """矿物相枚举"""
    # 橄榄石系
    OLIVINE_ALPHA = "forsterite"  # α相，上地幔
    OLIVINE_BETA = "wadsleyite"   # β相，过渡带
    OLIVINE_GAMMA = "ringwoodite" # γ相，过渡带底部
    
    # 辉石系
    CLINOENSTATITE = "clinoenstatite"
    ORTHOENSTATITE = "orthoenstatite"
    MAJORITE = "majorite"  # 高压辉石
    
    # 钙钛矿系
    PEROVSKITE_MG = "magnesium_silicate_perovskite"  # Bridgmanite
    POST_PEROVSKITE = "post_perovskite"  # 核幔边界
    
    # 氧化物
    PEROCLASE = "periclase"  # MgO
    CORUNDUM = "corundum"    # Al2O3
    STISHOVITE = "stishovite"  # SiO2高压相
    
    # 其他
    GARNET = "pyrope_garnet"
    SPINEL = "spinel"
    AKIMOTOITE = "akimotoite"  # 钛铁矿结构


class CrystalSystem(Enum):
    """晶系枚举"""
    CUBIC = "cubic"
    TETRAGONAL = "tetragonal"
    ORTHORHOMBIC = "orthorhombic"
    HEXAGONAL = "hexagonal"
    MONOCLINIC = "monoclinic"
    TRICLINIC = "triclinic"


@dataclass
class ElasticTensor:
    """弹性张量"""
    C11: float
    C12: float
    C13: float = None
    C22: float = None
    C23: float = None
    C33: float = None
    C44: float = None
    C55: float = None
    C66: float = None
    
    def __post_init__(self):
        # 默认各向同性情况
        if self.C13 is None:
            self.C13 = self.C12
        if self.C22 is None:
            self.C22 = self.C11
        if self.C33 is None:
            self.C33 = self.C11
        if self.C44 is None:
            self.C44 = (self.C11 - self.C12) / 2
        if self.C55 is None:
            self.C55 = self.C44
        if self.C66 is None:
            self.C66 = self.C44
    
    def get_bulk_modulus_voigt(self) -> float:
        """Voigt平均体模量"""
        K = (self.C11 + self.C22 + self.C33 + 
             2*(self.C12 + self.C13 + self.C23)) / 9
        return K
    
    def get_shear_modulus_voigt(self) -> float:
        """Voigt平均剪切模量"""
        G = (self.C11 + self.C22 + self.C33 - 
             self.C12 - self.C13 - self.C23 + 
             3*(self.C44 + self.C55 + self.C66)) / 15
        return G
    
    def get_elastic_anisotropy(self) -> float:
        """计算弹性各向异性"""
        A = 2 * self.C44 / (self.C11 - self.C12)
        return A


@dataclass
class MineralProperties:
    """矿物物理性质"""
    density: float          # kg/m^3
    molar_mass: float       # g/mol
    bulk_modulus: float     # GPa
    shear_modulus: float    # GPa
    thermal_expansion: float  # 1/K
    heat_capacity: float    # J/(mol·K)
    thermal_conductivity: float  # W/(m·K)
    elastic_tensor: Optional[ElasticTensor] = None
    crystal_system: Optional[CrystalSystem] = None


@dataclass
class ThermodynamicState:
    """热力学状态"""
    pressure: float     # GPa
    temperature: float  # K
    volume: float       # m^3/mol
    energy: float       # kJ/mol
    entropy: float      # J/(mol·K)


class MineralDatabase:
    """矿物物理性质数据库"""
    
    def __init__(self):
        self.minerals: Dict[MineralPhase, MineralProperties] = {}
        self._init_mineral_data()
    
    def _init_mineral_data(self):
        """初始化矿物数据（STP条件下）"""
        self.minerals = {
            MineralPhase.OLIVINE_ALPHA: MineralProperties(
                density=3220,
                molar_mass=140.69,
                bulk_modulus=129.0,
                shear_modulus=79.0,
                thermal_expansion=2.7e-5,
                heat_capacity=118.0,
                thermal_conductivity=5.1,
                elastic_tensor=ElasticTensor(328, 68, 68, 200, 78, 235, 67),
                crystal_system=CrystalSystem.ORTHORHOMBIC
            ),
            
            MineralPhase.OLIVINE_BETA: MineralProperties(
                density=3530,
                molar_mass=140.69,
                bulk_modulus=169.0,
                shear_modulus=113.0,
                thermal_expansion=2.4e-5,
                heat_capacity=115.0,
                thermal_conductivity=6.5,
                crystal_system=CrystalSystem.ORTHORHOMBIC
            ),
            
            MineralPhase.OLIVINE_GAMMA: MineralProperties(
                density=3560,
                molar_mass=140.69,
                bulk_modulus=185.0,
                shear_modulus=119.0,
                thermal_expansion=2.0e-5,
                heat_capacity=112.0,
                thermal_conductivity=7.0,
                elastic_tensor=ElasticTensor(354, 82, 82, 354, 82, 354, 128),
                crystal_system=CrystalSystem.CUBIC
            ),
            
            MineralPhase.PEROVSKITE_MG: MineralProperties(
                density=4100,
                molar_mass=100.39,
                bulk_modulus=253.0,
                shear_modulus=173.0,
                thermal_expansion=1.8e-5,
                heat_capacity=95.0,
                thermal_conductivity=10.0,
                crystal_system=CrystalSystem.ORTHORHOMBIC
            ),
            
            MineralPhase.POST_PEROVSKITE: MineralProperties(
                density=4930,
                molar_mass=100.39,
                bulk_modulus=230.0,
                shear_modulus=150.0,
                thermal_expansion=1.5e-5,
                heat_capacity=90.0,
                thermal_conductivity=12.0,
                crystal_system=CrystalSystem.ORTHORHOMBIC
            ),
            
            MineralPhase.PEROCLASE: MineralProperties(
                density=3580,
                molar_mass=40.30,
                bulk_modulus=160.0,
                shear_modulus=131.0,
                thermal_expansion=3.0e-5,
                heat_capacity=37.0,
                thermal_conductivity=60.0,
                elastic_tensor=ElasticTensor(297, 95, 95, 297, 95, 297, 156),
                crystal_system=CrystalSystem.CUBIC
            ),
            
            MineralPhase.STISHOVITE: MineralProperties(
                density=4290,
                molar_mass=60.08,
                bulk_modulus=298.0,
                shear_modulus=220.0,
                thermal_expansion=1.5e-5,
                heat_capacity=43.0,
                thermal_conductivity=12.0,
                crystal_system=CrystalSystem.TETRAGONAL
            ),
            
            MineralPhase.MAJORITE: MineralProperties(
                density=3560,
                molar_mass=401.6,
                bulk_modulus=165.0,
                shear_modulus=90.0,
                thermal_expansion=2.0e-5,
                heat_capacity=320.0,
                thermal_conductivity=4.0,
                crystal_system=CrystalSystem.CUBIC
            ),
        }
        
        logger.info(f"矿物数据库初始化: {len(self.minerals)}种矿物")
    
    def get_mineral(self, phase: MineralPhase) -> Optional[MineralProperties]:
        """获取矿物性质"""
        return self.minerals.get(phase)
    
    def list_phases(self) -> List[MineralPhase]:
        """列出所有矿物相"""
        return list(self.minerals.keys())


class HighPressureCalculator:
    """高压矿物物理计算器"""
    
    def __init__(self, mineral_db: MineralDatabase):
        self.db = mineral_db
        self.birch_murnaghan_order = 3
    
    def calculate_eos(self,
                     phase: MineralPhase,
                     pressures: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算状态方程 (Equation of State)
        
        使用Birch-Murnaghan EOS:
        E(V) = E0 + (9V0B0/16) * [(V0/V)^(2/3) - 1]^2 * 
               {6 - 4(V0/V)^(2/3) + B0'[(V0/V)^(2/3) - 1]}
        
        Args:
            phase: 矿物相
            pressures: 压力数组 (GPa)
        
        Returns:
            体积、能量等随压力的变化
        """
        mineral = self.db.get_mineral(phase)
        if mineral is None:
            raise ValueError(f"未知矿物相: {phase}")
        
        # STP参数
        V0 = mineral.molar_mass / mineral.density * 1e-6  # m^3/mol
        B0 = mineral.bulk_modulus  # GPa
        B0_prime = 4.0  # 典型值
        
        volumes = []
        energies = []
        
        for P in pressures:
            # 从压力反推体积（使用Murnaghan EOS）
            # P = (B0/B0') * [(V/V0)^(-B0') - 1]
            V = V0 * (1 + B0_prime * P / B0) ** (-1/B0_prime)
            volumes.append(V)
            
            # 能量（相对于STP）
            eta = (V0 / V) ** (2/3)
            E = (9 * V0 * B0 / 16) * ((eta - 1)**3 * B0_prime + 
                                       (eta - 1)**2 * (6 - 4*eta))
            energies.append(E)
        
        return {
            'pressure_GPa': pressures,
            'volume_m3_mol': np.array(volumes),
            'energy_kJ_mol': np.array(energies),
            'V0': V0,
            'B0': B0,
            'B0_prime': B0_prime
        }
    
    def calculate_elastic_moduli_hpht(self,
                                     phase: MineralPhase,
                                     pressure: float,
                                     temperature: float) -> Dict[str, float]:
        """
        计算高温高压下的弹性模量
        
        使用经验修正:
        K(P,T) = K0 + dK/dP*(P-P0) - dK/dT*(T-T0)
        
        Args:
            phase: 矿物相
            pressure: 压力 (GPa)
            temperature: 温度 (K)
        
        Returns:
            弹性模量
        """
        mineral = self.db.get_mineral(phase)
        if mineral is None:
            return {}
        
        K0 = mineral.bulk_modulus
        G0 = mineral.shear_modulus
        
        # 压力导数（典型值）
        dK_dP = 4.0
        dG_dP = 1.5
        
        # 温度导数（典型值）
        dK_dT = -0.02  # GPa/K
        dG_dT = -0.015  # GPa/K
        
        K = K0 + dK_dP * pressure + dK_dT * (temperature - 300)
        G = G0 + dG_dP * pressure + dG_dT * (temperature - 300)
        
        # 密度随P,T变化
        density = self._calculate_density(phase, pressure, temperature)
        
        return {
            'bulk_modulus_GPa': K,
            'shear_modulus_GPa': G,
            'youngs_modulus_GPa': 9*K*G / (3*K + G),
            'poisson_ratio': (3*K - 2*G) / (2*(3*K + G)),
            'density_kg_m3': density
        }
    
    def _calculate_density(self, phase: MineralPhase,
                          pressure: float, temperature: float) -> float:
        """计算高温高压下的密度"""
        mineral = self.db.get_mineral(phase)
        rho0 = mineral.density
        alpha = mineral.thermal_expansion
        B0 = mineral.bulk_modulus
        
        # 体积模量修正
        B = B0 + 4.0 * pressure
        
        # 压缩
        compression = 1 - pressure / (B + pressure)
        
        # 热膨胀
        thermal = 1 - alpha * (temperature - 300)
        
        return rho0 / (compression * thermal)
    
    def calculate_seismic_velocities(self,
                                    phase: MineralPhase,
                                    pressure: float,
                                    temperature: float) -> Dict[str, float]:
        """
        计算地震波速
        
        Vp = sqrt((K + 4G/3) / rho)
        Vs = sqrt(G / rho)
        
        Args:
            phase: 矿物相
            pressure: 压力 (GPa)
            temperature: 温度 (K)
        
        Returns:
            波速信息
        """
        moduli = self.calculate_elastic_moduli_hpht(phase, pressure, temperature)
        
        K = moduli['bulk_modulus_GPa'] * 1e9  # Pa
        G = moduli['shear_modulus_GPa'] * 1e9  # Pa
        rho = moduli['density_kg_m3']
        
        Vp = np.sqrt((K + 4*G/3) / rho) / 1000  # km/s
        Vs = np.sqrt(G / rho) / 1000  # km/s
        
        return {
            'Vp_km_s': Vp,
            'Vs_km_s': Vs,
            'Vp_Vs_ratio': Vp / Vs,
            'acoustic_impedance': rho * Vp * 1000,  # kg/(m^2·s)
            'bulk_sound_velocity': np.sqrt(K / rho) / 1000  # km/s
        }


class PhaseTransitionCalculator:
    """相变计算器"""
    
    def __init__(self, mineral_db: MineralDatabase):
        self.db = mineral_db
        self.transition_boundaries = self._init_boundaries()
    
    def _init_boundaries(self) -> Dict:
        """初始化相边界数据"""
        return {
            # 橄榄石到瓦兹利石 (α-β)
            ('alpha', 'beta'): {
                'dP_dT_MPa_K': 3.0,
                'P0_GPa': 13.5,
                'T0_K': 1200,
                'clapeyron_slope': 3.0
            },
            # 瓦兹利石到林伍德石 (β-γ)
            ('beta', 'gamma'): {
                'dP_dT_MPa_K': 4.0,
                'P0_GPa': 18.0,
                'T0_K': 1500,
                'clapeyron_slope': 4.0
            },
            # 林伍德石分解 (γ-钙钛矿+方镁石)
            ('gamma', 'perovskite+periclase'): {
                'dP_dT_MPa_K': -2.5,
                'P0_GPa': 23.0,
                'T0_K': 1600,
                'clapeyron_slope': -2.5
            },
            # 钙钛矿到后钙钛矿
            ('perovskite', 'post_perovskite'): {
                'dP_dT_MPa_K': 8.0,
                'P0_GPa': 120.0,
                'T0_K': 2500,
                'clapeyron_slope': 8.0
            }
        }
    
    def get_transition_pressure(self,
                               phase_from: str,
                               phase_to: str,
                               temperature: float) -> Optional[float]:
        """
        获取相变压力
        
        使用Clapeyron方程:
        dP/dT = ΔS/ΔV
        
        Args:
            phase_from: 初始相
            phase_to: 目标相
            temperature: 温度 (K)
        
        Returns:
            相变压力 (GPa)
        """
        key = (phase_from, phase_to)
        if key not in self.transition_boundaries:
            return None
        
        boundary = self.transition_boundaries[key]
        dP_dT = boundary['clapeyron_slope']  # MPa/K
        P0 = boundary['P0_GPa']
        T0 = boundary['T0_K']
        
        # P = P0 + (dP/dT) * (T - T0)
        P_transition = P0 + (dP_dT / 1000) * (temperature - T0)
        
        return P_transition
    
    def construct_phase_diagram(self,
                               system: str,
                               T_range: Tuple[float, float] = (300, 3000),
                               P_range: Tuple[float, float] = (0, 140)) -> Dict:
        """
        构建相图
        
        Args:
            system: 系统 (如 "Mg2SiO4")
            T_range: 温度范围 (K)
            P_range: 压力范围 (GPa)
        
        Returns:
            相图数据
        """
        logger.info(f"构建{system}相图")
        
        temperatures = np.linspace(T_range[0], T_range[1], 100)
        
        phase_boundaries = {}
        
        # Mg2SiO4系统
        if system == "Mg2SiO4":
            transitions = [
                ('alpha', 'beta'),
                ('beta', 'gamma'),
                ('gamma', 'perovskite+periclase')
            ]
            
            for trans in transitions:
                pressures = []
                for T in temperatures:
                    P = self.get_transition_pressure(trans[0], trans[1], T)
                    if P is not None and P_range[0] <= P <= P_range[1]:
                        pressures.append(P)
                    else:
                        pressures.append(np.nan)
                
                phase_boundaries[f"{trans[0]}-{trans[1]}"] = {
                    'temperature': temperatures,
                    'pressure': np.array(pressures)
                }
        
        return {
            'system': system,
            'temperature_range': T_range,
            'pressure_range': P_range,
            'boundaries': phase_boundaries,
            'stable_phases': self._identify_stable_phases(temperatures, system)
        }
    
    def _identify_stable_phases(self, temperatures: np.ndarray,
                                system: str) -> List[str]:
        """识别稳定相区域"""
        # 简化版本
        if system == "Mg2SiO4":
            return ['alpha', 'beta', 'gamma', 'perovskite+periclase']
        return []
    
    def calculate_transition_enthalpy(self,
                                     phase_from: MineralPhase,
                                     phase_to: MineralPhase) -> float:
        """计算相变焓"""
        # 简化的相变焓估计
        enthalpies = {
            (MineralPhase.OLIVINE_ALPHA, MineralPhase.OLIVINE_BETA): 25.0,  # kJ/mol
            (MineralPhase.OLIVINE_BETA, MineralPhase.OLIVINE_GAMMA): 20.0,
            (MineralPhase.OLIVINE_GAMMA, MineralPhase.PEROVSKITE_MG): 120.0,
        }
        
        return enthalpies.get((phase_from, phase_to), 50.0)


class MantleComposition:
    """地幔成分模型"""
    
    def __init__(self):
        # 地幔岩石圈成分 (Pyrolite模型)
        self.pyrolite = {
            MineralPhase.OLIVINE_ALPHA: 0.57,
            MineralPhase.ORTHOENSTATITE: 0.17,
            MineralPhase.CLINOENSTATITE: 0.12,
            MineralPhase.GARNET: 0.14
        }
        
        # 洋中脊玄武岩源区
        self.morb_source = {
            MineralPhase.OLIVINE_ALPHA: 0.50,
            MineralPhase.ORTHOENSTATITE: 0.25,
            MineralPhase.CLINOENSTATITE: 0.15,
            MineralPhase.GARNET: 0.10
        }
    
    def get_density_profile(self,
                          depth_km: np.ndarray,
                          model: str = "pyrolite") -> np.ndarray:
        """
        计算地幔密度剖面
        
        Args:
            depth_km: 深度数组 (km)
            model: 成分模型
        
        Returns:
            密度剖面 (kg/m^3)
        """
        composition = self.pyrolite if model == "pyrolite" else self.morb_source
        
        densities = []
        for depth in depth_km:
            # 压力梯度 ~30 MPa/km
            pressure = depth * 0.03  # GPa
            
            # 温度梯度 (地热线)
            if depth < 100:
                temperature = 300 + depth * 5  # K
            else:
                temperature = 1300 + (depth - 100) * 0.3
            
            # 确定当前相
            phase = self._get_phase_at_conditions(pressure, temperature)
            
            # 平均密度
            avg_density = 0
            for mineral, fraction in composition.items():
                # 这里需要更复杂的计算
                base_density = 3200  # kg/m^3
                # 压力和温度修正
                density = base_density * (1 + pressure * 0.01) * (1 - (temperature - 300) * 1e-5)
                avg_density += fraction * density
            
            densities.append(avg_density)
        
        return np.array(densities)
    
    def _get_phase_at_conditions(self, pressure: float, 
                                 temperature: float) -> MineralPhase:
        """根据条件确定稳定相"""
        if pressure < 13:
            return MineralPhase.OLIVINE_ALPHA
        elif pressure < 18:
            return MineralPhase.OLIVINE_BETA
        elif pressure < 23:
            return MineralPhase.OLIVINE_GAMMA
        else:
            return MineralPhase.PEROVSKITE_MG
    
    def calculate_seismic_profile(self,
                                 depths: np.ndarray) -> Dict[str, np.ndarray]:
        """计算地震波速剖面"""
        calculator = HighPressureCalculator(MineralDatabase())
        
        Vp_profile = []
        Vs_profile = []
        
        for depth in depths:
            pressure = depth * 0.03  # GPa
            temperature = 1600 if depth > 100 else 300 + depth * 10
            
            phase = self._get_phase_at_conditions(pressure, temperature)
            
            try:
                velocities = calculator.calculate_seismic_velocities(
                    phase, pressure, temperature
                )
                Vp_profile.append(velocities['Vp_km_s'])
                Vs_profile.append(velocities['Vs_km_s'])
            except:
                Vp_profile.append(8.0)
                Vs_profile.append(4.5)
        
        return {
            'depth_km': depths,
            'Vp_km_s': np.array(Vp_profile),
            'Vs_km_s': np.array(Vs_profile),
            'density': self.get_density_profile(depths)
        }


# 应用案例: 地球深部物质研究
def earth_deep_matter_example():
    """地球深部物质研究示例"""
    logger.info("=" * 60)
    logger.info("地球深部物质研究示例")
    logger.info("=" * 60)
    
    # 1. 初始化数据库
    db = MineralDatabase()
    
    # 2. 查询橄榄石性质
    forsterite = db.get_mineral(MineralPhase.OLIVINE_ALPHA)
    logger.info(f"\n镁橄榄石(α)性质:")
    logger.info(f"  密度: {forsterite.density} kg/m³")
    logger.info(f"  体积模量: {forsterite.bulk_modulus} GPa")
    logger.info(f"  剪切模量: {forsterite.shear_modulus} GPa")
    logger.info(f"  晶系: {forsterite.crystal_system.value}")
    
    # 3. 高压计算器
    hp_calc = HighPressureCalculator(db)
    
    # 状态方程
    pressures = np.linspace(0, 30, 50)
    eos = hp_calc.calculate_eos(MineralPhase.OLIVINE_ALPHA, pressures)
    
    logger.info(f"\n状态方程计算:")
    logger.info(f"  STP体积: {eos['V0']*1e6:.3f} cm³/mol")
    logger.info(f"  STP体积模量: {eos['B0']:.1f} GPa")
    logger.info(f"  30GPa时体积压缩: {(1 - eos['volume_m3_mol'][-1]/eos['V0'])*100:.1f}%")
    
    # 4. 高温高压弹性模量
    conditions = [
        (0, 300),    # STP
        (10, 1500),  # 上地幔
        (20, 1800),  # 过渡带
    ]
    
    logger.info(f"\n高温高压弹性模量:")
    for P, T in conditions:
        moduli = hp_calc.calculate_elastic_moduli_hpht(
            MineralPhase.OLIVINE_ALPHA, P, T
        )
        logger.info(f"  {P} GPa, {T} K: K={moduli['bulk_modulus_GPa']:.1f} GPa, "
                   f"G={moduli['shear_modulus_GPa']:.1f} GPa, "
                   f"ρ={moduli['density_kg_m3']:.0f} kg/m³")
    
    # 5. 地震波速计算
    logger.info(f"\n地震波速:")
    for P, T in conditions:
        velocities = hp_calc.calculate_seismic_velocities(
            MineralPhase.OLIVINE_ALPHA, P, T
        )
        logger.info(f"  {P} GPa, {T} K: Vp={velocities['Vp_km_s']:.2f} km/s, "
                   f"Vs={velocities['Vs_km_s']:.2f} km/s")
    
    # 6. 相变计算
    phase_calc = PhaseTransitionCalculator(db)
    
    logger.info(f"\n橄榄石-瓦兹利石相变:")
    for T in [1000, 1500, 2000]:
        P_trans = phase_calc.get_transition_pressure('alpha', 'beta', T)
        logger.info(f"  温度 {T} K 时相变压力: {P_trans:.1f} GPa")
    
    # 7. 构建相图
    phase_diagram = phase_calc.construct_phase_diagram(
        "Mg2SiO4",
        T_range=(300, 2500),
        P_range=(0, 30)
    )
    logger.info(f"\n相图包含边界: {list(phase_diagram['boundaries'].keys())}")
    
    # 8. 地幔成分和地震剖面
    mantle = MantleComposition()
    
    depths = np.linspace(0, 700, 100)
    density_profile = mantle.get_density_profile(depths)
    seismic_profile = mantle.calculate_seismic_profile(depths)
    
    logger.info(f"\n地幔密度剖面:")
    logger.info(f"  100 km: {density_profile[10]:.0f} kg/m³")
    logger.info(f"  400 km: {density_profile[57]:.0f} kg/m³")
    logger.info(f"  700 km: {density_profile[-1]:.0f} kg/m³")
    
    logger.info(f"\n地震波速剖面:")
    logger.info(f"  100 km: Vp={seismic_profile['Vp_km_s'][10]:.2f} km/s")
    logger.info(f"  400 km: Vp={seismic_profile['Vp_km_s'][57]:.2f} km/s")
    
    return {
        'eos': eos,
        'phase_diagram': phase_diagram,
        'seismic_profile': seismic_profile
    }


if __name__ == "__main__":
    earth_deep_matter_example()
