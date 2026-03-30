"""
Band Gap Inverse Design Module
==============================

带隙逆向设计模块 - 设计具有目标带隙的半导体/绝缘体材料。

应用案例：
- 太阳能电池材料优化 (Eg ≈ 1.0-1.5 eV)
- LED材料设计 (特定发射波长)
- 透明导电氧化物 (宽带隙 + 高导电性)
- 热电材料 (带隙与电导率平衡)
"""

import jax
import jax.numpy as jnp
from jax import grad, jit
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import numpy as np

from .core import (
    DesignTarget, DesignSpace, ParameterizedStructure,
    FractionalCoordinateStructure, ObjectiveFunction, InverseDesignOptimizer
)


@dataclass
class BandGapTarget:
    """带隙设计目标"""
    target_gap: float  # eV
    gap_type: str = 'indirect'  # 'direct' 或 'indirect'
    tolerance: float = 0.05  # eV
    
    # 额外约束
    max_absorption_edge: Optional[float] = None  # 最大吸收边
    min_carrier_mobility: Optional[float] = None  # 最小载流子迁移率
    stability_weight: float = 0.1  # 结构稳定性权重


class BandGapCalculator:
    """
    带隙计算器
    
    使用DFT计算材料的电子带隙
    """
    
    def __init__(self, dft_engine: Any):
        """
        Args:
            dft_engine: DFT计算引擎 (JAX-DFT或DFTK接口)
        """
        self.dft = dft_engine
        
        # 常用近似 (DFT低估带隙)
        self.scissor_correction = 0.5  # eV，剪刀算符修正
    
    def calculate(self,
                  positions: jnp.ndarray,
                  atomic_numbers: jnp.ndarray,
                  cell: jnp.ndarray) -> Dict[str, float]:
        """
        计算带隙
        
        Returns:
            {
                'band_gap': 带隙值 (eV),
                'direct_gap': 直接带隙 (eV),
                'indirect_gap': 间接带隙 (eV),
                'homo': 价带顶 (eV),
                'lumo': 导带底 (eV),
                'vbm_k': VBM的k点位置,
                'cbm_k': CBM的k点位置
            }
        """
        # 执行DFT计算获取能带
        # 这里使用简化的模型
        
        # 基于原子数和体积的启发式带隙估计
        n_electrons = jnp.sum(atomic_numbers)
        volume = jnp.abs(jnp.linalg.det(cell))
        density = n_electrons / volume
        
        # 简化的带隙模型 (仅用于演示)
        base_gap = 2.0 + 0.1 * jnp.sin(density * 10)
        
        # 添加随机扰动模拟不同结构
        gap_variation = jnp.std(positions) * 0.5
        
        indirect_gap = float(base_gap + gap_variation)
        direct_gap = indirect_gap + 0.2  # 直接带隙通常更大
        
        return {
            'band_gap': indirect_gap,
            'direct_gap': direct_gap,
            'indirect_gap': indirect_gap,
            'homo': -indirect_gap / 2,
            'lumo': indirect_gap / 2,
            'vbm_k': '(0,0,0)',
            'cbm_k': '(0.5,0,0)'
        }
    
    def calculate_with_scissor(self,
                               positions: jnp.ndarray,
                               atomic_numbers: jnp.ndarray,
                               cell: jnp.ndarray,
                               experimental_gap: Optional[float] = None) -> Dict[str, float]:
        """
        使用剪刀算符修正计算带隙
        
        Args:
            experimental_gap: 实验带隙值 (用于校准)
            
        Returns:
            修正后的带隙
        """
        result = self.calculate(positions, atomic_numbers, cell)
        
        if experimental_gap is not None:
            # 校准剪刀算符
            dft_gap = result['band_gap']
            scissor = experimental_gap - dft_gap
        else:
            scissor = self.scissor_correction
        
        result['band_gap_corrected'] = result['band_gap'] + scissor
        result['scissor'] = scissor
        
        return result
    
    def absorption_coefficient(self,
                               energy: jnp.ndarray,
                               band_gap: float,
                               direct_gap: float) -> jnp.ndarray:
        """
        计算吸收系数
        
        使用Tauc模型近似
        
        Args:
            energy: 光子能量数组 (eV)
            band_gap: 带隙 (eV)
            direct_gap: 直接带隙 (eV)
            
        Returns:
            吸收系数
        """
        # 直接跃迁: α ∝ (E - Eg)^0.5
        # 间接跃迁: α ∝ (E - Eg)^2
        
        alpha_direct = jnp.where(
            energy > direct_gap,
            (energy - direct_gap)**0.5,
            0.0
        )
        
        alpha_indirect = jnp.where(
            energy > band_gap,
            (energy - band_gap)**2,
            0.0
        )
        
        # 混合 (假设一定比例的直接/间接跃迁)
        return 1000 * (0.3 * alpha_direct + 0.7 * alpha_indirect)


class BandGapObjective:
    """
    带隙目标函数
    
    构建针对带隙优化的目标函数
    """
    
    def __init__(self,
                 band_gap_calculator: BandGapCalculator,
                 target: BandGapTarget):
        self.calculator = band_gap_calculator
        self.target = target
    
    def __call__(self,
                 positions: jnp.ndarray,
                 atomic_numbers: jnp.ndarray,
                 cell: jnp.ndarray) -> Dict[str, float]:
        """
        计算带隙相关的性质字典
        """
        result = self.calculator.calculate(positions, atomic_numbers, cell)
        return {
            'band_gap': result['band_gap'],
            'direct_gap': result['direct_gap'],
            'indirect_gap': result['indirect_gap']
        }
    
    def loss(self,
             positions: jnp.ndarray,
             atomic_numbers: jnp.ndarray,
             cell: jnp.ndarray) -> float:
        """
        计算带隙损失
        
        目标是最小化与目标带隙的偏差
        """
        props = self.calculator.calculate(positions, atomic_numbers, cell)
        
        if self.target.gap_type == 'direct':
            actual_gap = props['direct_gap']
        else:
            actual_gap = props['indirect_gap']
        
        # 带隙误差
        gap_error = (actual_gap - self.target.target_gap)**2 / (self.target.target_gap**2)
        
        # 类型约束 (如果需要直接带隙)
        type_penalty = 0.0
        if self.target.gap_type == 'direct':
            type_penalty = 0.1 * jnp.maximum(0, props['indirect_gap'] - props['direct_gap'])
        
        # 结构稳定性约束 (避免原子重叠)
        stability = self._stability_penalty(positions, cell)
        
        return gap_error + type_penalty + self.target.stability_weight * stability
    
    def _stability_penalty(self, positions: jnp.ndarray, cell: jnp.ndarray) -> float:
        """结构稳定性惩罚 (排斥原子重叠)"""
        # 计算所有原子对距离
        r_ij = positions[:, None, :] - positions[None, :, :]
        
        # 周期性边界条件
        frac = r_ij @ jnp.linalg.inv(cell)
        frac = frac - jnp.rint(frac)
        r_ij = frac @ cell
        
        distances = jnp.linalg.norm(r_ij, axis=2)
        
        # 避免自相互作用
        distances = jnp.where(distances < 1e-10, 1e10, distances)
        
        # 惩罚过近的原子对 (Lennard-Jones型)
        sigma = 2.0  # 特征长度 (Bohr)
        epsilon = 1.0
        
        lj = epsilon * ((sigma / distances)**12 - (sigma / distances)**6)
        
        # 只保留排斥部分 (距离小于平衡距离)
        penalty = jnp.sum(jnp.where(lj > 0, lj, 0))
        
        return penalty


class SolarCellOptimizer:
    """
    太阳能电池材料优化器
    
    优化太阳能电池吸收层的带隙和光学性质
    """
    
    # 理想带隙范围 (Shockley-Queisser极限)
    IDEAL_GAP_MIN = 1.0  # eV
    IDEAL_GAP_MAX = 1.5  # eV
    
    def __init__(self, band_gap_calculator: BandGapCalculator):
        self.calc = band_gap_calculator
    
    def detailed_balance_efficiency(self,
                                    band_gap: float,
                                    solar_spectrum: Optional[jnp.ndarray] = None,
                                    temperature: float = 300.0) -> Dict[str, float]:
        """
        计算Shockley-Queisser详细平衡效率
        
        Args:
            band_gap: 带隙 (eV)
            solar_spectrum: 太阳光谱
            temperature: 温度 (K)
            
        Returns:
            效率分析结果
        """
        # 简化计算
        # 理想效率近似: η = Eg/qVoc * (1 - T_c/T_s)
        # 这里使用简化公式
        
        kT = 8.617e-5 * temperature  # eV
        
        # 开路电压 (理想情况)
        voc = band_gap - 0.3  # 经验公式: Voc ≈ Eg - 0.3 eV
        
        # 短路电流 (与带隙以上的光谱积分相关)
        # 简化: Jsc ∝ 1/Eg
        jsc = 30.0 / band_gap  # mA/cm² (近似)
        
        # 填充因子 (理想情况)
        ff = 0.85
        
        # 效率
        efficiency = (voc * jsc * ff) / 100.0  # AM1.5G ~ 100 mW/cm²
        
        return {
            'voc': voc,  # V
            'jsc': jsc,  # mA/cm²
            'fill_factor': ff,
            'efficiency': efficiency,  # %
            'max_efficiency': min(33.0, efficiency)  # SQ极限
        }
    
    def optimize_for_spectrum(self,
                              initial_structures: List[ParameterizedStructure],
                              spectrum_type: str = 'AM1.5G') -> ParameterizedStructure:
        """
        针对特定光谱优化材料
        
        Args:
            initial_structures: 初始结构列表
            spectrum_type: 光谱类型 ('AM1.5G', 'AM0', 'indoor')
            
        Returns:
            优化后的结构
        """
        # 根据光谱类型选择目标带隙
        if spectrum_type == 'AM1.5G':
            target_gap = 1.34  # eV (单结最优)
        elif spectrum_type == 'AM0':
            target_gap = 1.30  # eV
        elif spectrum_type == 'indoor':
            target_gap = 1.8  # eV (室内光)
        else:
            target_gap = 1.34
        
        # 创建目标
        target = BandGapTarget(
            target_gap=target_gap,
            gap_type='direct',  # 太阳能电池偏好直接带隙
            tolerance=0.05
        )
        
        # 构建目标函数
        objective = BandGapObjective(self.calc, target)
        
        # 包装为通用目标函数
        def wrapped_objective(params, structure):
            structure.set_params(params)
            pos, nums, cell = structure.to_structure()
            return objective.loss(pos, nums, cell)
        
        # 优化
        optimizer = InverseDesignOptimizer(
            None,  # 占位
            optimizer_type='adam',
            learning_rate=0.01,
            max_iter=500
        )
        optimizer.objective = wrapped_objective
        
        result = optimizer.optimize(initial_structures[0])
        return result


class LEDMaterialDesigner:
    """
    LED材料设计师
    
    设计具有特定发射波长的LED材料
    """
    
    def __init__(self, band_gap_calculator: BandGapCalculator):
        self.calc = band_gap_calculator
    
    def wavelength_to_gap(self, wavelength_nm: float) -> float:
        """
        波长转换为带隙能量
        
        E(eV) = 1240 / λ(nm)
        """
        return 1240.0 / wavelength_nm
    
    def gap_to_wavelength(self, gap_ev: float) -> float:
        """
        带隙能量转换为波长
        
        λ(nm) = 1240 / E(eV)
        """
        return 1240.0 / gap_ev
    
    def design_for_color(self,
                         color: str,
                         initial_structure: ParameterizedStructure) -> ParameterizedStructure:
        """
        为特定颜色设计LED材料
        
        Args:
            color: 颜色名称 ('red', 'green', 'blue', 'white', 'UV')
            initial_structure: 初始结构
            
        Returns:
            优化后的结构
        """
        # 颜色到波长的映射
        color_wavelengths = {
            'red': 650,      # nm
            'orange': 600,
            'yellow': 580,
            'green': 530,
            'cyan': 490,
            'blue': 470,
            'violet': 400,
            'UV': 350,
            'white': 550  # 需要多色组合，这里简化为绿色
        }
        
        wavelength = color_wavelengths.get(color, 550)
        target_gap = self.wavelength_to_gap(wavelength)
        
        print(f"设计目标: {color} LED")
        print(f"目标波长: {wavelength} nm")
        print(f"目标带隙: {target_gap:.3f} eV")
        
        # 创建目标和优化
        target = BandGapTarget(
            target_gap=target_gap,
            gap_type='direct',  # LED需要直接带隙
            tolerance=0.02
        )
        
        objective = BandGapObjective(self.calc, target)
        
        def wrapped_objective(params, structure):
            structure.set_params(params)
            pos, nums, cell = structure.to_structure()
            return objective.loss(pos, nums, cell)
        
        optimizer = InverseDesignOptimizer(
            None,
            optimizer_type='adam',
            learning_rate=0.01,
            max_iter=500
        )
        optimizer.objective = wrapped_objective
        
        result = optimizer.optimize(initial_structure)
        
        # 验证结果
        final_pos, final_nums, final_cell = result.to_structure()
        final_props = self.calc.calculate(final_pos, final_nums, final_cell)
        final_wavelength = self.gap_to_wavelength(final_props['direct_gap'])
        
        print(f"\n优化结果:")
        print(f"最终带隙: {final_props['direct_gap']:.3f} eV")
        print(f"最终波长: {final_wavelength:.1f} nm")
        
        return result
    
    def rgb_led_set(self) -> Dict[str, Dict]:
        """
        设计RGB三色LED材料组合
        
        Returns:
            RGB材料规格
        """
        rgb_specs = {}
        
        for color, wavelength in [('red', 650), ('green', 530), ('blue', 470)]:
            gap = self.wavelength_to_gap(wavelength)
            rgb_specs[color] = {
                'wavelength_nm': wavelength,
                'band_gap_ev': gap,
                'optimal_structure': None  # 需要分别优化
            }
        
        return rgb_specs


class TransparentConductorOptimizer:
    """
    透明导电氧化物(TCO)优化器
    
    同时优化带隙(透明性)和电导率
    """
    
    def __init__(self, band_gap_calculator: BandGapCalculator):
        self.calc = band_gap_calculator
    
    def transparency_at_gap(self,
                           band_gap: float,
                           thickness_nm: float = 100.0) -> float:
        """
        计算给定带隙材料的透明度
        
        可见光范围: 1.6 - 3.1 eV (380-780 nm)
        
        Args:
            band_gap: 带隙 (eV)
            thickness_nm: 薄膜厚度
            
        Returns:
            可见光平均透明度 (0-1)
        """
        # 可见光范围
        vis_min, vis_max = 1.6, 3.1
        
        if band_gap > vis_max:
            # 宽带隙，完全透明
            return 1.0
        elif band_gap < vis_min:
            # 窄带隙，不透明
            return 0.0
        else:
            # 部分透明
            return (band_gap - vis_min) / (vis_max - vis_min)
    
    def conductivity_metric(self,
                          carrier_concentration: float,
                          mobility: float) -> float:
        """
        电导率指标
        
        σ = n * e * μ
        """
        # 归一化电导率 (相对于ITO)
        sigma = carrier_concentration * mobility
        sigma_ito = 1e21 * 30  # ITO参考值
        
        return sigma / sigma_ito
    
    def figure_of_merit(self,
                       band_gap: float,
                       carrier_concentration: float,
                       mobility: float,
                       thickness_nm: float = 100.0) -> float:
        """
        TCO品质因数
        
        FoM = T^10 / R_sheet
        其中 T 是透明度, R_sheet 是方块电阻
        """
        transparency = self.transparency_at_gap(band_gap, thickness_nm)
        
        # 方块电阻 (简化模型)
        sigma = carrier_concentration * mobility * 1.6e-19  # S/m
        R_sheet = 1.0 / (sigma * thickness_nm * 1e-9)  # Ohm/sq
        
        # 品质因数
        fom = (transparency ** 10) / R_sheet
        
        return fom
    
    def optimize_tco(self,
                     initial_structure: ParameterizedStructure,
                     min_transparency: float = 0.8) -> ParameterizedStructure:
        """
        优化TCO材料
        
        目标: 最大化电导率，同时保持透明度 > min_transparency
        
        Args:
            initial_structure: 初始结构
            min_transparency: 最小透明度要求
            
        Returns:
            优化后的结构
        """
        # 需要宽带隙 (>3.0 eV 以确保透明)
        target_gap = 3.2  # eV
        
        target = BandGapTarget(
            target_gap=target_gap,
            gap_type='direct',
            tolerance=0.2
        )
        
        objective = BandGapObjective(self.calc, target)
        
        def tco_objective(params, structure):
            structure.set_params(params)
            pos, nums, cell = structure.to_structure()
            
            # 基础带隙损失
            loss = objective.loss(pos, nums, cell)
            
            # 透明度约束
            props = self.calc.calculate(pos, nums, cell)
            transparency = self.transparency_at_gap(props['band_gap'])
            
            if transparency < min_transparency:
                # 惩罚低透明度
                loss += 10.0 * (min_transparency - transparency)**2
            
            return loss
        
        optimizer = InverseDesignOptimizer(
            None,
            optimizer_type='adam',
            learning_rate=0.01,
            max_iter=500
        )
        optimizer.objective = tco_objective
        
        return optimizer.optimize(initial_structure)


def example_bandgap_design():
    """带隙逆向设计示例"""
    print("=" * 60)
    print("带隙逆向设计示例")
    print("=" * 60)
    
    # 创建模拟的DFT引擎
    class MockDFTEngine:
        pass
    
    dft_engine = MockDFTEngine()
    bg_calc = BandGapCalculator(dft_engine)
    
    # 示例1: 设计太阳能电池材料
    print("\n【示例1: 太阳能电池材料设计】")
    solar_optimizer = SolarCellOptimizer(bg_calc)
    
    for spectrum in ['AM1.5G', 'indoor']:
        efficiency = solar_optimizer.detailed_balance_efficiency(1.34)
        print(f"\n光谱类型: {spectrum}")
        print(f"  理论效率: {efficiency['efficiency']:.1f}%")
        print(f"  开路电压: {efficiency['voc']:.2f} V")
        print(f"  短路电流: {efficiency['jsc']:.1f} mA/cm²")
    
    # 示例2: LED材料设计
    print("\n【示例2: LED材料设计】")
    led_designer = LEDMaterialDesigner(bg_calc)
    
    for color in ['red', 'green', 'blue']:
        wavelength = {'red': 650, 'green': 530, 'blue': 470}[color]
        gap = led_designer.wavelength_to_gap(wavelength)
        print(f"  {color}: λ={wavelength}nm, Eg={gap:.2f}eV")
    
    # 示例3: TCO优化
    print("\n【示例3: 透明导电氧化物】")
    tco_optimizer = TransparentConductorOptimizer(bg_calc)
    
    for gap in [2.5, 3.0, 3.5, 4.0]:
        T = tco_optimizer.transparency_at_gap(gap)
        print(f"  带隙 {gap:.1f} eV: 透明度 = {T:.1%}")
    
    # 示例4: 逆向优化
    print("\n【示例4: 逆向优化演示】")
    
    # 创建初始结构
    structure = FractionalCoordinateStructure(
        n_atoms=2,
        atomic_numbers=jnp.array([14, 8]),  # SiO-like
        initial_cell=jnp.eye(3) * 8.0,
        fix_cell=False
    )
    
    # 设计目标: 2.5 eV带隙
    target = BandGapTarget(target_gap=2.5, gap_type='indirect')
    objective = BandGapObjective(bg_calc, target)
    
    def wrapped_obj(params, struct):
        struct.set_params(params)
        pos, nums, cell = struct.to_structure()
        return objective.loss(pos, nums, cell)
    
    optimizer = InverseDesignOptimizer(
        None, 'adam', 0.05, 200
    )
    optimizer.objective = wrapped_obj
    
    print(f"初始参数: {structure.get_params()[:6]}")
    result = optimizer.optimize(structure)
    
    final_pos, final_nums, final_cell = result.to_structure()
    final_props = bg_calc.calculate(final_pos, final_nums, final_cell)
    
    print(f"优化后带隙: {final_props['band_gap']:.3f} eV")
    print(f"目标带隙: 2.5 eV")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    example_bandgap_design()
