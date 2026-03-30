"""
DFT-LAMMPS 合金设计套件
=======================
相图+力学性能+腐蚀

提供高熵合金设计的完整工作流。

Author: DFT-LAMMPS Team
Phase: 56 - Orchestration System
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..orchestration.workflow_composer import Workflow, WorkflowStep, WorkflowType
from ..orchestration.topic_template import ResearchTopic


logger = logging.getLogger("alloy_design_kit")


@dataclass
class AlloyComposition:
    """合金成分"""
    elements: Dict[str, float]  # 元素: 摩尔分数
    
    @property
    def num_elements(self) -> int:
        return len(self.elements)
    
    @property
    def is_hea(self) -> bool:
        """是否为高熵合金"""
        return self.num_elements >= 5 and all(
            0.05 <= frac <= 0.35 
            for frac in self.elements.values()
        )
    
    def to_string(self) -> str:
        """转换为字符串表示"""
        return "-".join(f"{el}{int(frac*100)}" for el, frac in sorted(self.elements.items()))


@dataclass
class PhaseDiagramData:
    """相图数据"""
    temperatures: List[float]
    phases: List[str]
    phase_fractions: Dict[str, List[float]]
    transition_temperatures: Dict[str, float]


@dataclass
class MechanicalProperties:
    """力学性能"""
    hardness: float               # HV 或 GPa
    yield_strength: float         # MPa
    ultimate_strength: float      # MPa
    elongation: float             # %
    
    # 弹性常数
    bulk_modulus: float           # GPa
    shear_modulus: float          # GPa
    young_modulus: float          # GPa
    poisson_ratio: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hardness": self.hardness,
            "yield_strength": self.yield_strength,
            "ultimate_strength": self.ultimate_strength,
            "elongation": self.elongation,
            "bulk_modulus": self.bulk_modulus,
            "shear_modulus": self.shear_modulus,
            "young_modulus": self.young_modulus,
            "poisson_ratio": self.poisson_ratio
        }


@dataclass
class CorrosionProperties:
    """腐蚀性能"""
    corrosion_rate: float         # mm/year
    pitting_potential: float      # V vs SCE
    passivation_ability: float    # 0-1
    corrosion_resistance_score: float  # 0-1


class AlloyDesignKit:
    """
    合金设计套件
    
    提供高熵合金设计的一站式解决方案
    
    Example:
        kit = AlloyDesignKit()
        
        composition = AlloyComposition({
            "Co": 0.2, "Cr": 0.2, "Fe": 0.2, "Mn": 0.2, "Ni": 0.2
        })
        
        # 运行完整设计流程
        results = kit.run_full_design(composition)
        
        # 或单独分析
        phase_diagram = kit.calculate_phase_diagram(composition)
        mechanical = kit.predict_mechanical_properties(composition)
    """
    
    def __init__(self):
        self._ce_model_cache: Dict[str, Any] = {}
    
    def run_full_design(
        self,
        composition: AlloyComposition,
        structure_type: str = "fcc",
        design_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        运行完整的合金设计流程
        
        Args:
            composition: 合金成分
            structure_type: 晶体结构类型 (fcc, bcc, hcp)
            design_options: 设计选项
        """
        options = design_options or {}
        results = {
            "composition": composition.to_string(),
            "is_hea": composition.is_hea,
            "structure_type": structure_type
        }
        
        logger.info(f"Starting alloy design for {composition.to_string()}")
        
        # 1. 生成SQS结构
        logger.info("Step 1: Generating SQS structures")
        sqs_structures = self.generate_sqs_structures(composition, structure_type)
        results['sqs_structures'] = sqs_structures
        
        # 2. 训练团簇展开模型
        if options.get('use_cluster_expansion', True):
            logger.info("Step 2: Training cluster expansion model")
            ce_model = self.train_cluster_expansion(composition, sqs_structures)
            results['cluster_expansion'] = ce_model
        
        # 3. 计算相图
        logger.info("Step 3: Calculating phase diagram")
        phase_diagram = self.calculate_phase_diagram(composition, structure_type)
        results['phase_diagram'] = phase_diagram
        
        # 4. 预测力学性能
        logger.info("Step 4: Predicting mechanical properties")
        mechanical = self.predict_mechanical_properties(composition, structure_type)
        results['mechanical_properties'] = mechanical
        
        # 5. 预测腐蚀性能
        if options.get('include_corrosion', True):
            logger.info("Step 5: Predicting corrosion resistance")
            corrosion = self.predict_corrosion_resistance(composition)
            results['corrosion_properties'] = corrosion
        
        # 6. 验证热力学稳定性
        logger.info("Step 6: Validating thermodynamic stability")
        stability = self.validate_thermodynamic_stability(composition, phase_diagram)
        results['thermodynamic_stability'] = stability
        
        # 7. 生成设计报告
        results['report'] = self._generate_design_report(composition, results)
        
        logger.info(f"Alloy design completed for {composition.to_string()}")
        return results
    
    def generate_sqs_structures(
        self,
        composition: AlloyComposition,
        structure_type: str,
        num_structures: int = 5
    ) -> List[Dict[str, Any]]:
        """
        生成特殊准随机结构(SQS)
        
        用于近似随机固溶体的周期性结构
        """
        logger.info(f"Generating SQS structures for {composition.to_string()}")
        
        workflow = Workflow(
            id=f"sqs_{composition.to_string()}",
            name=f"SQS generation for {composition.to_string()}",
            workflow_type=WorkflowType.SEQUENTIAL
        )
        
        workflow.steps.append(WorkflowStep(
            id="generate_sqs",
            name="Generate SQS",
            module_name="structure",
            capability_name="generate_sqs",
            inputs={
                "composition": composition.elements,
                "structure_type": structure_type,
                "num_structures": num_structures
            },
            outputs={"sqs_structures": "sqs_structures"}
        ))
        
        # 简化实现
        structures = []
        for i in range(num_structures):
            structures.append({
                "id": f"sqs_{i}",
                "size": 100,
                "composition": composition.elements,
                "structure_type": structure_type,
                "correlation_error": 0.02
            })
        
        return structures
    
    def train_cluster_expansion(
        self,
        composition: AlloyComposition,
        training_structures: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        训练团簇展开(CE)模型
        
        用于高效预测构型能量
        """
        logger.info(f"Training cluster expansion for {composition.to_string()}")
        
        workflow = Workflow(
            id=f"ce_{composition.to_string()}",
            name=f"Cluster expansion for {composition.to_string()}",
            workflow_type=WorkflowType.SEQUENTIAL
        )
        
        # 步骤1: 计算训练结构能量
        workflow.steps.append(WorkflowStep(
            id="compute_energies",
            name="Compute Training Energies",
            module_name="vasp",
            capability_name="calculate_energy",
            inputs={"structures": training_structures},
            outputs={"energies": "training_energies"}
        ))
        
        # 步骤2: 拟合CE
        workflow.steps.append(WorkflowStep(
            id="fit_ce",
            name="Fit Cluster Expansion",
            module_name="ce",
            capability_name="fit_cluster_expansion",
            inputs={
                "structures": training_structures,
                "energies": "$training_energies"
            },
            outputs={"ce_model": "ce_model"},
            depends_on=["compute_energies"]
        ))
        
        # 简化实现
        return {
            "num_clusters": 50,
            "cv_score": 0.001,  # 交叉验证分数
            "ecis": {},  # 有效团簇相互作用
            "fitted": True
        }
    
    def calculate_phase_diagram(
        self,
        composition: AlloyComposition,
        structure_type: str,
        temperature_range: Tuple[float, float] = (300, 1500)
    ) -> PhaseDiagramData:
        """
        计算相图
        
        使用蒙特卡洛模拟和团簇展开
        """
        logger.info(f"Calculating phase diagram for {composition.to_string()}")
        
        workflow = Workflow(
            id=f"phasediag_{composition.to_string()}",
            name=f"Phase diagram for {composition.to_string()}",
            workflow_type=WorkflowType.SEQUENTIAL
        )
        
        # 步骤: 蒙特卡洛模拟
        workflow.steps.append(WorkflowStep(
            id="mc_simulation",
            name="Monte Carlo Simulation",
            module_name="ce",
            capability_name="monte_carlo",
            inputs={
                "composition": composition.elements,
                "temperature_range": list(temperature_range)
            },
            outputs={"phase_diagram": "phase_diagram"}
        ))
        
        # 生成模拟的相图数据
        temperatures = list(range(int(temperature_range[0]), int(temperature_range[1]), 50))
        
        # 简化的相图数据
        phases = ["single_phase", "two_phase"]
        phase_fractions = {
            "single_phase": [1.0 if T > 800 else 0.0 for T in temperatures],
            "two_phase": [0.0 if T > 800 else 1.0 for T in temperatures]
        }
        
        transition_temperatures = {
            "solidus": 800,
            "liquidus": 1400
        }
        
        return PhaseDiagramData(
            temperatures=temperatures,
            phases=phases,
            phase_fractions=phase_fractions,
            transition_temperatures=transition_temperatures
        )
    
    def predict_mechanical_properties(
        self,
        composition: AlloyComposition,
        structure_type: str
    ) -> MechanicalProperties:
        """
        预测力学性能
        
        基于计算弹性常数和固溶强化模型
        """
        logger.info(f"Predicting mechanical properties for {composition.to_string()}")
        
        workflow = Workflow(
            id=f"mech_{composition.to_string()}",
            name=f"Mechanical properties of {composition.to_string()}",
            workflow_type=WorkflowType.SEQUENTIAL
        )
        
        # 步骤1: 计算弹性常数
        workflow.steps.append(WorkflowStep(
            id="elastic",
            name="Elastic Constants",
            module_name="vasp",
            capability_name="elastic_constants",
            inputs={"composition": composition.elements, "structure_type": structure_type},
            outputs={"elastic_tensor": "elastic_tensor", "moduli": "moduli"}
        ))
        
        # 步骤2: 预测硬度
        workflow.steps.append(WorkflowStep(
            id="hardness",
            name="Predict Hardness",
            module_name="analysis",
            capability_name="predict_hardness",
            inputs={"composition": composition.elements, "moduli": "$moduli"},
            outputs={"hardness": "hardness"},
            depends_on=["elastic"]
        ))
        
        # 基于组分平均估算（简化模型）
        # 实际应基于DFT计算
        
        # 元素贡献（示例值）
        element_contributions = {
            "Co": {"B": 180, "G": 75, "Hv": 300},
            "Cr": {"B": 160, "G": 115, "Hv": 400},
            "Fe": {"B": 170, "G": 82, "Hv": 250},
            "Mn": {"B": 120, "G": 45, "Hv": 200},
            "Ni": {"B": 180, "G": 76, "Hv": 280}
        }
        
        # 混合律计算
        bulk = 0
        shear = 0
        hardness = 0
        
        for elem, frac in composition.elements.items():
            if elem in element_contributions:
                bulk += frac * element_contributions[elem]["B"]
                shear += frac * element_contributions[elem]["G"]
                hardness += frac * element_contributions[elem]["Hv"]
        
        # 添加固溶强化贡献
        solid_solution_strengthening = self._calculate_sss(composition)
        hardness += solid_solution_strengthening
        
        # 计算其他模量
        young = 9 * bulk * shear / (3 * bulk + shear)
        poisson = (3 * bulk - 2 * shear) / (6 * bulk + 2 * shear)
        
        return MechanicalProperties(
            hardness=hardness,
            yield_strength=hardness * 3,  # 近似关系
            ultimate_strength=hardness * 4,
            elongation=20.0,
            bulk_modulus=bulk,
            shear_modulus=shear,
            young_modulus=young,
            poisson_ratio=poisson
        )
    
    def _calculate_sss(self, composition: AlloyComposition) -> float:
        """计算固溶强化贡献"""
        # 简化的Labusch-Nabarro模型
        delta_a = 0.0  # 晶格常数差异
        base_elem = max(composition.elements, key=composition.elements.get)
        
        # 这里简化处理
        return 50.0 * (composition.num_elements - 1)  # 每增加一种元素约50 HV贡献
    
    def predict_corrosion_resistance(
        self,
        composition: AlloyComposition
    ) -> CorrosionProperties:
        """
        预测耐腐蚀性
        
        基于元素的电化学特性
        """
        logger.info(f"Predicting corrosion resistance for {composition.to_string()}")
        
        # 元素的钝化能力（示例值，Cr是关键）
        passivation_scores = {
            "Cr": 1.0,
            "Ni": 0.8,
            "Co": 0.6,
            "Fe": 0.4,
            "Mn": 0.3,
            "Al": 0.9,
            "Ti": 0.95
        }
        
        # 计算加权钝化能力
        passivation = sum(
            composition.elements.get(elem, 0) * score
            for elem, score in passivation_scores.items()
        )
        
        # Cr含量对耐蚀性的关键影响
        cr_content = composition.elements.get("Cr", 0)
        cr_factor = min(1.0, cr_content / 0.12)  # 12% Cr为关键阈值
        
        corrosion_resistance = 0.5 * passivation + 0.5 * cr_factor
        
        return CorrosionProperties(
            corrosion_rate=0.01 * (1 - corrosion_resistance),  # mm/year
            pitting_potential=0.2 + 0.5 * corrosion_resistance,  # V
            passivation_ability=passivation,
            corrosion_resistance_score=corrosion_resistance
        )
    
    def validate_thermodynamic_stability(
        self,
        composition: AlloyComposition,
        phase_diagram: PhaseDiagramData
    ) -> Dict[str, Any]:
        """验证热力学稳定性"""
        
        # 检查室温下的单相区
        room_temp = 300
        single_phase_fraction = 0
        
        for T, frac in zip(phase_diagram.temperatures, phase_diagram.phase_fractions.get("single_phase", [])):
            if abs(T - room_temp) < 50:
                single_phase_fraction = frac
                break
        
        is_stable = single_phase_fraction > 0.9
        
        return {
            "is_single_phase_at_room_temp": is_stable,
            "single_phase_fraction": single_phase_fraction,
            "solidus_temperature": phase_diagram.transition_temperatures.get("solidus", 0),
            "recommendation": "Stable single-phase HEA" if is_stable else "May require heat treatment"
        }
    
    def design_new_alloy(
        self,
        target_properties: Dict[str, Any],
        available_elements: List[str],
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[AlloyComposition]:
        """
        设计新合金
        
        基于目标性质反向设计成分
        """
        logger.info("Designing new alloy compositions")
        
        # 简化的设计算法
        # 实际应使用更复杂的优化算法（如遗传算法、贝叶斯优化）
        
        candidates = []
        
        # 生成候选成分
        from itertools import combinations
        
        num_elements = constraints.get("num_elements", 5) if constraints else 5
        
        for elem_combo in combinations(available_elements, num_elements):
            # 等摩尔成分
            frac = 1.0 / num_elements
            composition = AlloyComposition({elem: frac for elem in elem_combo})
            
            # 预测性质
            mechanical = self.predict_mechanical_properties(composition, "fcc")
            corrosion = self.predict_corrosion_resistance(composition)
            
            # 检查是否满足目标
            passes = True
            if "min_hardness" in target_properties:
                if mechanical.hardness < target_properties["min_hardness"]:
                    passes = False
            
            if "min_corrosion_resistance" in target_properties:
                if corrosion.corrosion_resistance_score < target_properties["min_corrosion_resistance"]:
                    passes = False
            
            if passes:
                candidates.append(composition)
        
        # 按预测性能排序
        candidates.sort(
            key=lambda c: self._calculate_alloy_score(c, target_properties),
            reverse=True
        )
        
        return candidates[:10]  # 返回前10个
    
    def _calculate_alloy_score(
        self,
        composition: AlloyComposition,
        target_properties: Dict[str, Any]
    ) -> float:
        """计算合金评分"""
        score = 0.0
        
        mechanical = self.predict_mechanical_properties(composition, "fcc")
        corrosion = self.predict_corrosion_resistance(composition)
        
        if "min_hardness" in target_properties:
            score += min(1.0, mechanical.hardness / target_properties["min_hardness"])
        
        if "min_corrosion_resistance" in target_properties:
            score += corrosion.corrosion_resistance_score
        
        return score
    
    def one_click_alloy_design(
        self,
        elements: List[str],
        structure_type: str = "fcc",
        **kwargs
    ) -> Dict[str, Any]:
        """一键合金设计"""
        # 等摩尔成分
        frac = 1.0 / len(elements)
        composition = AlloyComposition({elem: frac for elem in elements})
        
        return self.run_full_design(composition, structure_type, kwargs)
    
    def _generate_design_report(
        self,
        composition: AlloyComposition,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成设计报告"""
        mechanical = results.get('mechanical_properties', MechanicalProperties(0, 0, 0, 0, 0, 0, 0, 0))
        corrosion = results.get('corrosion_properties', CorrosionProperties(0, 0, 0, 0))
        stability = results.get('thermodynamic_stability', {})
        
        return {
            "composition": composition.to_string(),
            "is_hea": composition.is_hea,
            "summary": {
                "predicted_hardness": mechanical.hardness,
                "corrosion_resistance": corrosion.corrosion_resistance_score,
                "thermodynamic_stability": stability.get("is_single_phase_at_room_temp", False)
            },
            "recommendations": self._generate_alloy_recommendations(composition, results)
        }
    
    def _generate_alloy_recommendations(
        self,
        composition: AlloyComposition,
        results: Dict[str, Any]
    ) -> List[str]:
        """生成合金设计建议"""
        recommendations = []
        
        mechanical = results.get('mechanical_properties')
        if mechanical and mechanical.hardness < 200:
            recommendations.append("Low hardness predicted. Consider adding more Cr or Mo for solid solution strengthening.")
        
        corrosion = results.get('corrosion_properties')
        if corrosion and corrosion.corrosion_resistance_score < 0.5:
            recommendations.append("Poor corrosion resistance. Increase Cr content to at least 12%.")
        
        stability = results.get('thermodynamic_stability', {})
        if not stability.get('is_single_phase_at_room_temp', True):
            recommendations.append("Multiphase region detected at room temperature. Consider annealing treatment or composition adjustment.")
        
        if composition.is_hea:
            recommendations.append("High-entropy alloy composition identified. Good configurational entropy for single-phase stability.")
        
        return recommendations


# 便捷函数
def quick_alloy_design(elements: List[str], **kwargs) -> Dict[str, Any]:
    """快速合金设计"""
    kit = AlloyDesignKit()
    return kit.one_click_alloy_design(elements, **kwargs)


def screen_hea_compositions(
    element_sets: List[List[str]],
    **kwargs
) -> List[Dict[str, Any]]:
    """筛选高熵合金成分"""
    kit = AlloyDesignKit()
    
    results = []
    for elements in element_sets:
        frac = 1.0 / len(elements)
        composition = AlloyComposition({elem: frac for elem in elements})
        
        if composition.is_hea:
            result = kit.run_full_design(composition, kwargs.get('structure_type', 'fcc'))
            results.append(result)
    
    return results