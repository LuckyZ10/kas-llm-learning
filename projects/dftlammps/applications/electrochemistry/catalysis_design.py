"""
Electrochemical Applications Module - 电化学应用案例

本模块提供电化学计算的实际应用案例：
- HER/HOR催化剂设计与筛选
- OER/ORR催化剂设计
- 电池电极/电解质界面模拟
- CO2还原催化剂
- N2还原催化剂

Author: DFT-LAMMPS Team
Date: 2025
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import warnings
from enum import Enum


class ElectrochemicalReaction(Enum):
    """电化学反应类型"""
    HER = "hydrogen_evolution"
    HOR = "hydrogen_oxidation"
    OER = "oxygen_evolution"
    ORR = "oxygen_reduction"
    CO2RR = "co2_reduction"
    NRR = "nitrogen_reduction"


@dataclass
class CatalystCandidate:
    """催化剂候选"""
    name: str
    composition: str
    structure_type: str
    surface_energy: float = 0.0
    adsorption_energies: Dict[str, float] = field(default_factory=dict)
    conductivity: float = 1e6
    stability_score: float = 1.0
    synthesis_difficulty: float = 1.0
    cost_index: float = 1.0


# ============================================================
# HER/OER催化剂设计
# ============================================================

class HERCatalystDesigner:
    """HER催化剂设计器"""
    
    def __init__(self, temperature: float = 298.15):
        self.temperature = temperature
        self.kT = 8.617e-5 * temperature
        self.volcano_coeff_a = 3.0
        self.volcano_coeff_b = 0.5
        
    def calculate_exchange_current(self, dg_h: float) -> float:
        """计算交换电流密度"""
        log_i0 = self.volcano_coeff_a - self.volcano_coeff_b * (dg_h * 10)**2
        return 10**log_i0
    
    def calculate_overpotential_tafel(self, dg_h: float, current_density: float = 0.01) -> float:
        """使用Tafel方程计算过电位"""
        i0 = self.calculate_exchange_current(dg_h)
        b = 0.120  # V/decade
        eta = b * np.log10(current_density / i0)
        return max(eta, 0.0)
    
    def design_from_principles(self, metal_options: List[str]) -> List[CatalystCandidate]:
        """基于原理设计HER催化剂"""
        candidates = []
        for metal in metal_options:
            dg_h = self._estimate_dg_h_pure(metal)
            candidate = CatalystCandidate(
                name=f"{metal}(111)",
                composition=metal,
                structure_type="fcc(111)" if metal in ["Pt", "Pd", "Ni", "Cu", "Au"] else "hcp(0001)",
                adsorption_energies={"H*": dg_h},
            )
            candidates.append(candidate)
        return candidates
    
    def rank_candidates(self, candidates: List[CatalystCandidate], criteria: str = "overpotential"):
        """排序候选催化剂"""
        scored = []
        for cat in candidates:
            dg_h = cat.adsorption_energies.get("H*", 0.0)
            if criteria == "overpotential":
                score = -abs(dg_h)
            else:
                score = -abs(dg_h)
            scored.append((cat, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def _estimate_dg_h_pure(self, metal: str) -> float:
        """估算纯金属的H吸附能"""
        reference_values = {
            "Pt": -0.15, "Pd": -0.20, "Ni": -0.35, "Co": -0.30,
            "Fe": -0.50, "Cu": 0.15, "Au": 0.45, "Ag": 0.35,
            "Mo": -0.40, "W": -0.55,
        }
        return reference_values.get(metal, 0.0)


class OERCatalystDesigner:
    """OER催化剂设计器"""
    
    def __init__(self):
        self.ooh_oh_scaling_slope = 1.0
        self.ooh_oh_scaling_intercept = 3.2
        self.o_oh_scaling_slope = 2.0
        self.o_oh_scaling_intercept = 0.0
        
    def calculate_overpotential(self, dg_oh: float, dg_o: Optional[float] = None, 
                                dg_ooh: Optional[float] = None, use_scaling: bool = True) -> Dict:
        """计算OER过电位"""
        if dg_o is None and use_scaling:
            dg_o = self.o_oh_scaling_slope * dg_oh + self.o_oh_scaling_intercept
        if dg_ooh is None and use_scaling:
            dg_ooh = self.ooh_oh_scaling_slope * dg_oh + self.ooh_oh_scaling_intercept
        
        step1 = dg_oh
        step2 = dg_o - dg_oh
        step3 = dg_ooh - dg_o
        step4 = 4.92 - dg_ooh
        
        steps = [step1, step2, step3, step4]
        step_names = ["OH* formation", "O* formation", "OOH* formation", "O2 release"]
        max_step = max(steps)
        rds_idx = steps.index(max_step)
        overpotential = max_step - 1.23
        
        return {
            "overpotential": max(overpotential, 0.0),
            "limiting_potential": max_step,
            "steps": dict(zip(step_names, steps)),
            "rate_determining_step": step_names[rds_idx],
        }


# ============================================================
# 电池电极/电解质界面
# ============================================================

class BatteryInterfaceModel:
    """电池界面模型"""
    
    def __init__(self, electrode_material: str = "Li", electrolyte: str = "LiPF6_EC_DMC", 
                 temperature: float = 298.15):
        self.electrode = electrode_material
        self.electrolyte = electrolyte
        self.temperature = temperature
        self.sei_thickness = None
        
    def predict_lithium_plating(self, current_density: float, electrolyte_concentration: float,
                               diffusion_coefficient: float) -> Dict:
        """预测锂枝晶生长"""
        F = 96485
        c0 = electrolyte_concentration * 1e-3
        j = current_density * 1e-3
        
        sand_time = np.pi * diffusion_coefficient * (c0 * F)**2 / (4 * j**2)
        j_critical = np.sqrt(np.pi * diffusion_coefficient * (c0 * F)**2 / (4 * 3600))
        
        return {
            "sand_time": sand_time,
            "critical_current_density": j_critical * 1e3,
            "dendrite_risk": current_density > j_critical * 1e3,
        }


class SolidElectrolyteInterface:
    """固态电解质界面(SEI)模型"""
    
    def __init__(self):
        self.components = {
            "Li2CO3": {"conductivity": 1e-9, "thickness": 5.0},
            "LiF": {"conductivity": 1e-11, "thickness": 2.0},
            "Li2O": {"conductivity": 1e-8, "thickness": 1.0},
        }
        
    def calculate_effective_conductivity(self) -> float:
        """计算SEI有效离子电导率"""
        total_thickness = sum(c["thickness"] for c in self.components.values())
        effective_cond = total_thickness / sum(c["thickness"] / c["conductivity"] 
                                               for c in self.components.values())
        return effective_cond
    
    def simulate_growth(self, initial_thickness: float, time: float, 
                       current_density: float, growth_rate_constant: float = 1e-3) -> float:
        """模拟SEI生长"""
        final_thickness = np.sqrt(initial_thickness**2 + 2 * growth_rate_constant * time)
        return final_thickness


# ============================================================
# 使用示例
# ============================================================

def example_her_design():
    """HER催化剂设计示例"""
    print("=" * 60)
    print("HER催化剂设计示例")
    print("=" * 60)
    
    designer = HERCatalystDesigner(temperature=298.15)
    metals = ["Pt", "Pd", "Ni", "Co", "Mo", "W", "Cu", "Au"]
    candidates = designer.design_from_principles(metals)
    
    print(f"\n候选催化剂 ({len(candidates)}):")
    print("-" * 50)
    
    for cat in candidates:
        dg_h = cat.adsorption_energies["H*"]
        i0 = designer.calculate_exchange_current(dg_h)
        eta = designer.calculate_overpotential_tafel(dg_h, 0.01)
        print(f"{cat.name:12s} ΔG_H = {dg_h:+.2f} eV, log(i₀) = {np.log10(i0):.2f}, η = {eta:.3f} V")
    
    ranked = designer.rank_candidates(candidates, criteria="overpotential")
    print("\n最优HER催化剂 (Top 3):")
    print("-" * 50)
    for i, (cat, score) in enumerate(ranked[:3], 1):
        print(f"{i}. {cat.name}: 评分 = {score:.3f}")
    
    return designer


def example_oer_design():
    """OER催化剂设计示例"""
    print("\n" + "=" * 60)
    print("OER催化剂设计示例")
    print("=" * 60)
    
    designer = OERCatalystDesigner()
    
    # 模拟数据
    test_data = [
        ("IrO2", 0.8),
        ("RuO2", 0.7),
        ("Co3O4", 0.9),
        ("NiO", 1.0),
        ("MnO2", 1.1),
    ]
    
    print("\n氧化物OER性能:")
    print("-" * 50)
    for name, dg_oh in test_data:
        result = designer.calculate_overpotential(dg_oh)
        print(f"{name:12s} ΔG_OH = {dg_oh:.2f} eV, η = {result['overpotential']:.3f} V")
    
    return designer


def example_battery_interface():
    """电池界面示例"""
    print("\n" + "=" * 60)
    print("锂离子电池界面模拟示例")
    print("=" * 60)
    
    interface = BatteryInterfaceModel(electrode_material="Li", electrolyte="LiPF6_EC_DMC")
    sei = SolidElectrolyteInterface()
    
    print("\nSEI性质:")
    print("-" * 50)
    print(f"组分数量: {len(sei.components)}")
    print(f"有效离子电导率: {sei.calculate_effective_conductivity():.2e} S/cm")
    
    # SEI生长模拟
    print(f"\nSEI生长模拟:")
    print("-" * 50)
    initial_thickness = 5.0
    for t in [1, 10, 100, 1000]:
        final_thickness = sei.simulate_growth(initial_thickness, t, 1.0)
        print(f"  {t:4d} h: {final_thickness:.2f} nm")
    
    # 锂枝晶预测
    print(f"\n锂枝晶生长预测:")
    print("-" * 50)
    for j in [0.5, 1.0, 2.0, 5.0]:
        prediction = interface.predict_lithium_plating(j, 1.0, 1e-6)
        risk = "高" if prediction['dendrite_risk'] else "低"
        print(f"  j = {j:.1f} mA/cm²: Sand时间 = {prediction['sand_time']/3600:.2f} h, 风险 = {risk}")
    
    return interface


if __name__ == "__main__":
    print("=" * 70)
    print(" 电化学应用案例模块")
    print(" Electrochemical Applications Module")
    print("=" * 70)
    
    example_her_design()
    example_oer_design()
    example_battery_interface()
    
    print("\n" + "=" * 70)
    print(" 所有示例完成!")
    print("=" * 70)