"""
SEI Growth Simulator
====================
锂离子电池SEI生长模拟器

模拟电极/电解质界面SEI层的形成和生长过程。
耦合电化学反应、离子传输和机械应力。

物理模型:
- 多相场描述 (电解质、电极、SEI组分)
- Butler-Volmer电化学反应
- 应力驱动的开裂
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
import logging

from ..core.electrochemical import ElectrochemicalPhaseField, ElectrochemicalConfig
from ..core.mechanochemistry import MechanoChemicalSolver

logger = logging.getLogger(__name__)


@dataclass
class SEIConfig(ElectrochemicalConfig):
    """
    SEI模拟配置
    
    Attributes:
        # SEI组分参数
        n_components: SEI组分数 (EC, Li2CO3, LiF, etc.)
        component_names: 组分名称列表
        component_properties: 各组分物性
        
        # 电化学反应
        reduction_potentials: 还原电位
        reaction_rates: 反应速率常数
        
        # 机械参数
        sei_modulus: SEI杨氏模量 (GPa)
        sei_fracture_toughness: 断裂韧性 (MPa·m^0.5)
        
        # 初始条件
        initial_sei_thickness: 初始SEI厚度 (nm)
        surface_roughness: 表面粗糙度
    """
    # SEI组分
    n_components: int = 3
    component_names: List[str] = field(default_factory=lambda: ['organic', 'Li2CO3', 'LiF'])
    
    # 组分体积分数
    component_fractions: Dict[str, float] = field(default_factory=lambda: {
        'organic': 0.3,
        'Li2CO3': 0.5,
        'LiF': 0.2
    })
    
    # 电化学参数
    reduction_potentials: Dict[str, float] = field(default_factory=lambda: {
        'organic': 0.8,  # V vs Li/Li+
        'Li2CO3': 1.0,
        'LiF': 1.2
    })
    
    reaction_rates: Dict[str, float] = field(default_factory=lambda: {
        'organic': 1e-3,  # m/s
        'Li2CO3': 5e-4,
        'LiF': 1e-4
    })
    
    # 机械参数
    sei_modulus: float = 10.0  # GPa (有机SEI较软)
    sei_fracture_toughness: float = 1.0  # MPa·m^0.5
    
    # 初始条件
    initial_sei_thickness: float = 5.0  # nm
    surface_roughness: float = 0.5  # nm
    
    # 模拟控制
    include_mechanical_failure: bool = True
    include_solvent_diffusion: bool = True
    
    def __post_init__(self):
        super().__post_init__()


class SEIGrowthSimulator(ElectrochemicalPhaseField):
    """
    SEI生长模拟器
    
    模拟锂离子电池SEI层的形成和演化。
    考虑多组分、电化学反应和机械失效。
    """
    
    def __init__(self, config: Optional[SEIConfig] = None):
        """
        初始化SEI模拟器
        
        Args:
            config: SEI配置
        """
        self.config = config or SEIConfig()
        super().__init__(self.config)
        
        # 初始化多相场
        self.component_phases = {}  # 相场变量 (电解质、SEI组分、电极)
        self._init_phase_fields()
        
        # SEI厚度追踪
        self.sei_thickness_history = []
        
        # 裂纹场
        self.damage = None
        if self.config.include_mechanical_failure:
            self.damage = np.zeros((self.config.nx, self.config.ny))
        
        logger.info(f"SEI simulator initialized")
        logger.info(f"Components: {self.config.component_names}")
    
    def _init_phase_fields(self):
        """初始化相场变量"""
        shape = (self.config.nx, self.config.ny)
        
        # 电解质相 (上部区域)
        self.component_phases['electrolyte'] = np.zeros(shape)
        self.component_phases['electrolyte'][:self.config.nx//3, :] = 1.0
        
        # SEI各组分
        for name in self.config.component_names:
            self.component_phases[name] = np.zeros(shape)
        
        # 初始SEI (中间薄层)
        sei_start = self.config.nx // 3
        sei_end = sei_start + int(self.config.initial_sei_thickness / self.config.dx)
        sei_end = min(sei_end, self.config.nx)
        
        for name in self.config.component_names:
            frac = self.config.component_fractions.get(name, 0.3)
            self.component_phases[name][sei_start:sei_end, :] = frac
        
        # 电极相 (下部区域)
        self.component_phases['electrode'] = np.zeros(shape)
        self.component_phases['electrode'][sei_end:, :] = 1.0
        
        # 添加到fields
        self.fields.update(self.component_phases)
    
    def initialize_fields(self, c0: Optional[np.ndarray] = None,
                         phi0: Optional[np.ndarray] = None,
                         custom_init: Optional[Dict] = None,
                         seed: Optional[int] = None):
        """
        初始化SEI场
        
        Args:
            c0: 初始Li浓度场
            phi0: 初始电势场
            custom_init: 自定义初始化参数
            seed: 随机种子
        """
        # 确保phi已初始化
        if not self.phi:
            self._init_phase_fields()
        
        # 调用父类初始化
        super().initialize_fields(c0, phi0, seed)
        
        # 应用自定义初始条件
        if custom_init:
            if 'sei_thickness' in custom_init:
                self._set_sei_thickness(custom_init['sei_thickness'])
            
            if 'surface_roughness' in custom_init:
                self._apply_surface_roughness(custom_init['surface_roughness'])
    
    def _set_sei_thickness(self, thickness: float):
        """设置SEI厚度"""
        # 重新初始化相场以匹配目标厚度
        shape = (self.config.nx, self.config.ny)
        
        # 清除现有SEI
        for name in self.config.component_names:
            self.component_phases[name][:] = 0
        
        # 设置新SEI
        sei_start = self.config.nx // 3
        sei_thickness_grid = int(thickness / self.config.dx)
        sei_end = min(sei_start + sei_thickness_grid, self.config.nx)
        
        for name in self.config.component_names:
            frac = self.config.component_fractions.get(name, 0.3)
            self.component_phases[name][sei_start:sei_end, :] = frac
    
    def _apply_surface_roughness(self, roughness: float):
        """应用表面粗糙度"""
        # 在SEI/电解质界面添加随机扰动
        sei_start = self.config.nx // 3
        
        # 生成粗糙界面
        ny = self.config.ny
        interface = np.zeros(ny)
        
        # 添加随机粗糙度
        noise = np.random.randn(ny) * roughness / self.config.dx
        
        # 平滑处理
        from scipy.ndimage import gaussian_filter1d
        interface = gaussian_filter1d(noise, sigma=3)
        
        # 应用界面形状
        for j in range(ny):
            shift = int(interface[j])
            # 根据界面位置调整相场
            pass  # 简化实现
    
    def _compute_sei_reaction_rates(self) -> Dict[str, np.ndarray]:
        """
        计算各组分的反应速率
        
        Returns:
            rates: 各组分反应速率字典
        """
        rates = {}
        
        # 计算过电位
        phi_eq = self._open_circuit_potential(self.c)
        eta_over = self.phi - phi_eq  # 过电位
        
        for comp_name in self.config.component_names:
            # 交换电流密度 (依赖于电解质浓度)
            j0 = self._exchange_current_density(self.c) * 0.1  # SEI反应较慢
            
            # Butler-Volmer动力学
            rate = self._butler_volmer(eta_over, j0)
            
            # 乘以该组分的反应速率常数
            k_reaction = self.config.reaction_rates.get(comp_name, 1e-3)
            rates[comp_name] = rate * k_reaction / self.config.reaction_rates['organic']
        
        return rates
    
    def _compute_mechanical_stress(self) -> np.ndarray:
        """
        计算SEI中的机械应力
        
        由于Li嵌入导致的体积变化产生应力
        
        Returns:
            stress: 应力场
        """
        if not self.config.include_mechanical_failure:
            return np.zeros((self.config.nx, self.config.ny))
        
        # 简化模型：应力正比于SEI厚度和Li浓度梯度
        stress = np.zeros((self.config.nx, self.config.ny))
        
        # SEI区域
        sei_mask = np.zeros((self.config.nx, self.config.ny))
        for name in self.config.component_names:
            sei_mask += self.component_phases[name]
        sei_mask = sei_mask > 0.5
        
        # 计算SEI厚度分布
        for j in range(self.config.ny):
            sei_indices = np.where(sei_mask[:, j])[0]
            if len(sei_indices) > 0:
                thickness = len(sei_indices) * self.config.dx
                # 应力 ∝ 厚度 * 应变
                strain = 0.1  # 假设应变
                stress[sei_indices, j] = self.config.sei_modulus * strain * thickness / 100
        
        return stress
    
    def _update_damage(self, stress: np.ndarray):
        """
        更新损伤场 (裂纹)
        
        Args:
            stress: 应力场
        """
        if self.damage is None:
            return
        
        # 裂纹准则: σ > K_IC / sqrt(π*a)
        threshold = self.config.sei_fracture_toughness  # 简化的阈值
        
        # 更新损伤
        new_damage = (np.abs(stress) > threshold).astype(float)
        
        # 损伤累积
        self.damage = np.maximum(self.damage, new_damage * 0.1)
        self.damage = np.clip(self.damage, 0, 1)
    
    def evolve_step(self) -> Dict:
        """
        执行SEI演化步骤
        
        Returns:
            info: 演化信息
        """
        # 1. 计算SEI反应速率
        reaction_rates = self._compute_sei_reaction_rates()
        
        # 2. 演化各组分相场
        for comp_name in self.config.component_names:
            # 相场演化 (Allen-Cahn类型)
            phi = self.component_phases[comp_name]
            rate = reaction_rates[comp_name]
            
            # 简化的演化方程
            dphi_dt = rate * (1 - phi) * self.component_phases['electrolyte'] - 0.01 * phi
            
            # 更新
            self.component_phases[comp_name] = np.clip(phi + self.config.dt * dphi_dt, 0, 1)
            self.fields[comp_name] = self.component_phases[comp_name]
        
        # 3. 更新电解质和电极相场
        total_sei = sum(self.component_phases[name] for name in self.config.component_names)
        self.component_phases['electrolyte'] = np.clip(1 - total_sei - self.component_phases['electrode'], 0, 1)
        self.component_phases['electrode'] = np.clip(1 - total_sei - self.component_phases['electrolyte'], 0, 1)
        
        self.fields['electrolyte'] = self.component_phases['electrolyte']
        self.fields['electrode'] = self.component_phases['electrode']
        
        # 4. 电化学演化 (父类)
        electro_info = super().evolve_step()
        
        # 5. 机械分析
        if self.config.include_mechanical_failure:
            stress = self._compute_mechanical_stress()
            self._update_damage(stress)
            self.fields['stress'] = stress
            self.fields['damage'] = self.damage
            electro_info['max_stress'] = float(np.abs(stress).max())
            electro_info['damage_fraction'] = float(self.damage.mean())
        
        # 6. 计算SEI厚度
        sei_thickness = self._compute_sei_thickness()
        self.sei_thickness_history.append(sei_thickness)
        electro_info['sei_thickness'] = sei_thickness
        
        return electro_info
    
    def _compute_sei_thickness(self) -> float:
        """
        计算平均SEI厚度
        
        Returns:
            thickness: SEI厚度 (nm)
        """
        total_sei = np.zeros((self.config.nx, self.config.ny))
        for name in self.config.component_names:
            total_sei += self.component_phases[name]
        
        # 找到SEI区域
        sei_mask = total_sei > 0.5
        
        if not np.any(sei_mask):
            return 0.0
        
        # 计算各位置的SEI厚度
        thicknesses = []
        for j in range(self.config.ny):
            sei_indices = np.where(sei_mask[:, j])[0]
            if len(sei_indices) > 0:
                thicknesses.append(len(sei_indices) * self.config.dx)
        
        avg_thickness = np.mean(thicknesses) if thicknesses else 0.0
        
        return avg_thickness
    
    def get_sei_properties(self) -> Dict:
        """
        获取SEI性质
        
        Returns:
            properties: SEI性质字典
        """
        # 各组分体积分数
        fractions = {}
        total_volume = 0
        
        for name in self.config.component_names:
            volume = np.sum(self.component_phases[name])
            fractions[name] = volume
            total_volume += volume
        
        # 归一化
        if total_volume > 0:
            fractions = {k: v/total_volume for k, v in fractions.items()}
        
        properties = {
            'thickness': self._compute_sei_thickness(),
            'volume_fractions': fractions,
            'growth_rate': self._compute_growth_rate(),
            'porosity': self._estimate_porosity(),
        }
        
        if self.config.include_mechanical_failure:
            properties['damage_fraction'] = float(self.damage.mean()) if self.damage is not None else 0
        
        return properties
    
    def _compute_growth_rate(self) -> float:
        """计算SEI生长速率"""
        if len(self.sei_thickness_history) < 2:
            return 0.0
        
        # 最近的变化速率
        dt = self.config.dt * self.config.save_interval
        recent_growth = (self.sei_thickness_history[-1] - 
                        self.sei_thickness_history[-5]) / (4 * dt) if len(self.sei_thickness_history) >= 5 else 0
        
        return recent_growth
    
    def _estimate_porosity(self) -> float:
        """估算SEI孔隙率"""
        # 简化：根据相场值估算
        total_sei = sum(self.component_phases[name] for name in self.config.component_names)
        
        # 低于阈值的区域认为是孔隙
        porosity = np.mean(total_sei < 0.3)
        
        return float(porosity)
    
    def get_impedance_contribution(self) -> float:
        """
        估算SEI对阻抗的贡献
        
        Returns:
            r_sei: SEI电阻 (Ω·m²)
        """
        thickness = self._compute_sei_thickness() * 1e-9  # nm -> m
        
        # 简化模型: R = d/σ
        sigma_sei = 1e-6  # S/m (SEI离子电导率)
        r_sei = thickness / sigma_sei
        
        return r_sei
