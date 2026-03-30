"""
自动合金设计-合成-测试模块
Autonomous Alloy Design-Synthesis-Testing

实现：
- 合金成分设计
- 相稳定性预测
- 力学性能优化
- 自动熔炼和测试
"""

import os
import json
import asyncio
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime
from enum import Enum
import numpy as np

from ..experiment.lab_automation import (
    AutonomousLab, SimulatedRobot, RobotInstruction, RobotCommand,
    SynthesisProtocol, SynthesisParameter
)
from ..experiment.synthesis_planning import (
    SynthesisPlanner, ChemicalCompound, ReactionType
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlloySystem(Enum):
    """合金体系"""
    HEA = "high_entropy_alloy"  # 高熵合金
    MEA = "medium_entropy_alloy"  # 中熵合金
    SUPERALLOY = "superalloy"
    TITANIUM_ALLOY = "titanium_alloy"
    ALUMINUM_ALLOY = "aluminum_allloy"
    STEEL = "steel"
    SHAPE_MEMORY = "shape_memory_alloy"
    MAGNETIC = "magnetic_alloy"


@dataclass
class AlloyComposition:
    """合金成分"""
    elements: Dict[str, float]  # 元素-含量(at%)
    
    @property
    def num_elements(self) -> int:
        return len(self.elements)
    
    @property
    def mixing_entropy(self) -> float:
        """计算混合熵 (J/mol/K)"""
        R = 8.314
        x = np.array(list(self.elements.values()))
        x = x / np.sum(x)  # 归一化
        return -R * np.sum(x * np.log(x + 1e-10))
    
    @property
    def is_hea(self) -> bool:
        """判断是否高熵合金 (ΔS > 1.5R)"""
        return self.mixing_entropy > 1.5 * 8.314
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "elements": self.elements,
            "num_elements": self.num_elements,
            "mixing_entropy": self.mixing_entropy,
            "is_hea": self.is_hea
        }


@dataclass
class PhasePrediction:
    """相预测结果"""
    phases: List[str]  # 预测相
    phase_fractions: Dict[str, float]
    stable_temperature_range: Tuple[float, float]
    
    # 特征参数
    valence_electron_concentration: float = 0.0
    atomic_size_difference: float = 0.0
    enthalpy_of_mixing: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MechanicalProperties:
    """力学性能"""
    yield_strength: float  # MPa
    ultimate_tensile_strength: float  # MPa
    elongation: float  # %
    hardness: float  # HV
    youngs_modulus: float  # GPa
    
    # 高温性能
    creep_resistance: Optional[float] = None
    fatigue_life: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AlloyCandidate:
    """合金候选"""
    name: str
    composition: AlloyComposition
    system: AlloySystem
    
    # 预测性质
    predicted_phase: Optional[PhasePrediction] = None
    predicted_properties: Optional[MechanicalProperties] = None
    predicted_density: float = 0.0  # g/cm³
    predicted_cost: float = 0.0  # USD/kg
    
    # 实验结果
    actual_properties: Optional[MechanicalProperties] = None
    microstructure: Optional[str] = None
    synthesis_success: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "composition": self.composition.to_dict(),
            "system": self.system.value,
            "predicted_density": self.predicted_density,
            "predicted_cost": self.predicted_cost,
            "synthesis_success": self.synthesis_success
        }


class AlloyDatabase:
    """合金数据库"""
    
    def __init__(self):
        self.element_properties = self._load_element_properties()
        self.phase_rules = self._load_phase_rules()
    
    def _load_element_properties(self) -> Dict[str, Dict[str, float]]:
        """加载元素性质"""
        return {
            "Al": {"radius": 1.43, "vec": 3.0, "Tm": 933, "cost": 2.0, "density": 2.70},
            "Ti": {"radius": 1.47, "vec": 4.0, "Tm": 1941, "cost": 35.0, "density": 4.51},
            "V": {"radius": 1.34, "vec": 5.0, "Tm": 2183, "cost": 300.0, "density": 6.11},
            "Cr": {"radius": 1.28, "vec": 6.0, "Tm": 2180, "cost": 10.0, "density": 7.19},
            "Mn": {"radius": 1.27, "vec": 7.0, "Tm": 1519, "cost": 2.0, "density": 7.43},
            "Fe": {"radius": 1.26, "vec": 8.0, "Tm": 1811, "cost": 0.5, "density": 7.87},
            "Co": {"radius": 1.25, "vec": 9.0, "Tm": 1768, "cost": 50.0, "density": 8.86},
            "Ni": {"radius": 1.24, "vec": 10.0, "Tm": 1728, "cost": 15.0, "density": 8.91},
            "Cu": {"radius": 1.28, "vec": 11.0, "Tm": 1358, "cost": 8.0, "density": 8.96},
            "Zn": {"radius": 1.34, "vec": 12.0, "Tm": 693, "cost": 3.0, "density": 7.14},
            "Zr": {"radius": 1.60, "vec": 4.0, "Tm": 2128, "cost": 80.0, "density": 6.52},
            "Nb": {"radius": 1.46, "vec": 5.0, "Tm": 2750, "cost": 180.0, "density": 8.57},
            "Mo": {"radius": 1.39, "vec": 6.0, "Tm": 2896, "cost": 80.0, "density": 10.28},
            "Hf": {"radius": 1.59, "vec": 4.0, "Tm": 2506, "cost": 900.0, "density": 13.31},
            "Ta": {"radius": 1.46, "vec": 5.0, "Tm": 3290, "cost": 300.0, "density": 16.65},
            "W": {"radius": 1.39, "vec": 6.0, "Tm": 3695, "cost": 100.0, "density": 19.25},
            "Re": {"radius": 1.37, "vec": 7.0, "Tm": 3459, "cost": 4000.0, "density": 21.02}
        }
    
    def _load_phase_rules(self) -> Dict[str, Any]:
        """加载相形成规则"""
        return {
            "vec_ss": (8.0, 11.5),  # 固溶体VEC范围
            "vec_bcc": (6.0, 7.5),
            "vec_fcc": (8.0, 10.0,
            "vec_hcp": (7.0, 8.0),
            "delta_critical": 0.066,  # 原子尺寸差临界值
            "omega_critical": 1.1  # 欧米参数临界值
        }
    
    def get_element_property(self, element: str, property: str) -> float:
        """获取元素性质"""
        return self.element_properties.get(element, {}).get(property, 0)


class AlloyDesigner:
    """合金设计师"""
    
    def __init__(self):
        self.database = AlloyDatabase()
        self.design_rules = self._init_design_rules()
    
    def _init_design_rules(self) -> Dict[AlloySystem, Dict[str, Any]]:
        """初始化设计规则"""
        return {
            AlloySystem.HEA: {
                "min_elements": 5,
                "entropy_threshold": 1.5 * 8.314,
                "composition_range": (5, 35),  # at%
                "preferred_elements": ["Co", "Cr", "Fe", "Mn", "Ni", "Al", "Cu"]
            },
            AlloySystem.SUPERALLOY: {
                "base_element": "Ni",
                "gamma_prime_formers": ["Al", "Ti", "Nb", "Ta"],
                "solid_solution_strengtheners": ["Co", "Cr", "Mo", "W", "Re"]
            },
            AlloySystem.TITANIUM_ALLOY: {
                "base_element": "Ti",
                "alpha_stabilizers": ["Al", "O", "N", "C"],
                "beta_stabilizers": ["V", "Mo", "Nb", "Ta", "Fe", "Mn", "Cr"],
                "neutral": ["Zr", "Sn", "Si"]
            }
        }
    
    def design_alloy(self, 
                    target_system: AlloySystem,
                    target_properties: Dict[str, float],
                    n_candidates: int = 10) -> List[AlloyCandidate]:
        """设计合金候选"""
        candidates = []
        
        rules = self.design_rules.get(target_system, {})
        
        for i in range(n_candidates):
            if target_system == AlloySystem.HEA:
                candidate = self._design_hea(rules, target_properties)
            elif target_system == AlloySystem.SUPERALLOY:
                candidate = self._design_superalloy(rules, target_properties)
            elif target_system == AlloySystem.TITANIUM_ALLOY:
                candidate = self._design_titanium_alloy(rules, target_properties)
            else:
                candidate = self._design_generic(target_system, target_properties)
            
            if candidate:
                candidate.name = f"{target_system.value}_{i+1:03d}"
                candidates.append(candidate)
        
        return candidates
    
    def _design_hea(self, rules: Dict, targets: Dict) -> Optional[AlloyCandidate]:
        """设计高熵合金"""
        elements = rules.get("preferred_elements", [])
        n_elem = np.random.randint(rules.get("min_elements", 5), 
                                   min(len(elements), 8))
        
        # 随机选择元素
        selected = np.random.choice(elements, n_elem, replace=False)
        
        # 等摩尔或随机配比
        if np.random.random() < 0.5:
            # 等摩尔
            composition = {elem: 100.0 / n_elem for elem in selected}
        else:
            # 随机配比，确保混合熵足够
            comp = np.random.dirichlet(np.ones(n_elem)) * 100
            composition = {elem: comp[i] for i, elem in enumerate(selected)}
        
        alloy_comp = AlloyComposition(elements=composition)
        
        # 检查是否满足HEA条件
        if not alloy_comp.is_hea:
            return None
        
        candidate = AlloyCandidate(
            name="",
            composition=alloy_comp,
            system=AlloySystem.HEA
        )
        
        # 预测相
        candidate.predicted_phase = self._predict_phase(alloy_comp)
        
        # 预测性能
        candidate.predicted_properties = self._predict_properties(alloy_comp)
        
        # 估算成本
        candidate.predicted_cost = self._estimate_cost(alloy_comp)
        candidate.predicted_density = self._estimate_density(alloy_comp)
        
        return candidate
    
    def _design_superalloy(self, rules: Dict, targets: Dict) -> AlloyCandidate:
        """设计高温合金"""
        base = rules.get("base_element", "Ni")
        
        composition = {base: 50.0}  # Ni基50%
        
        # 添加γ'形成元素
        gamma_formers = rules.get("gamma_prime_formers", [])
        for elem in np.random.choice(gamma_formers, 2, replace=False):
            composition[elem] = np.random.uniform(2, 8)
        
        # 添加固溶强化元素
        strengtheners = rules.get("solid_solution_strengtheners", [])
        for elem in np.random.choice(strengtheners, 
                                    min(3, len(strengtheners)), 
                                    replace=False):
            composition[elem] = np.random.uniform(5, 15)
        
        # 归一化
        total = sum(composition.values())
        composition = {k: v / total * 100 for k, v in composition.items()}
        
        alloy_comp = AlloyComposition(elements=composition)
        
        return AlloyCandidate(
            name="",
            composition=alloy_comp,
            system=AlloySystem.SUPERALLOY,
            predicted_properties=self._predict_properties(alloy_comp),
            predicted_cost=self._estimate_cost(alloy_comp),
            predicted_density=self._estimate_density(alloy_comp)
        )
    
    def _design_titanium_alloy(self, rules: Dict, targets: Dict) -> AlloyCandidate:
        """设计钛合金"""
        composition = {"Ti": 90.0}
        
        # 添加α稳定剂
        alpha_stab = rules.get("alpha_stabilizers", [])
        if alpha_stab:
            composition["Al"] = np.random.uniform(3, 8)
        
        # 添加β稳定剂
        beta_stab = rules.get("beta_stabilizers", [])
        if beta_stab and np.random.random() < 0.5:
            elem = np.random.choice(beta_stab)
            composition[elem] = np.random.uniform(2, 10)
        
        # 归一化
        total = sum(composition.values())
        composition = {k: v / total * 100 for k, v in composition.items()}
        
        alloy_comp = AlloyComposition(elements=composition)
        
        return AlloyCandidate(
            name="",
            composition=alloy_comp,
            system=AlloySystem.TITANIUM_ALLOY,
            predicted_properties=self._predict_properties(alloy_comp),
            predicted_cost=self._estimate_cost(alloy_comp),
            predicted_density=self._estimate_density(alloy_comp)
        )
    
    def _design_generic(self, system: AlloySystem, 
                       targets: Dict) -> AlloyCandidate:
        """通用合金设计"""
        # 随机选择3-5个元素
        available = list(self.database.element_properties.keys())
        n_elem = np.random.randint(3, 6)
        selected = np.random.choice(available, n_elem, replace=False)
        
        composition = {elem: 100.0 / n_elem for elem in selected}
        alloy_comp = AlloyComposition(elements=composition)
        
        return AlloyCandidate(
            name="",
            composition=alloy_comp,
            system=system,
            predicted_properties=self._predict_properties(alloy_comp),
            predicted_cost=self._estimate_cost(alloy_comp),
            predicted_density=self._estimate_density(alloy_comp)
        )
    
    def _predict_phase(self, composition: AlloyComposition) -> PhasePrediction:
        """预测合金相"""
        elements = list(composition.elements.keys())
        fractions = list(composition.elements.values())
        fractions = np.array(fractions) / sum(fractions)
        
        # 计算VEC
        vec = sum(
            self.database.get_element_property(e, "vec") * f
            for e, f in zip(elements, fractions)
        )
        
        # 计算原子尺寸差
        radii = [self.database.get_element_property(e, "radius") for e in elements]
        avg_radius = np.mean(radii)
        delta = np.sqrt(sum(f * (1 - r / avg_radius) ** 2 
                          for f, r in zip(fractions, radii)))
        
        # 相预测规则
        phases = []
        if delta < self.database.phase_rules["delta_critical"]:
            if 8 <= vec <= 10:
                phases.append("FCC")
            elif 6.5 <= vec < 7.5:
                phases.append("BCC")
            elif 7 <= vec < 8:
                phases.append("HCP")
        
        if not phases:
            phases.append("FCC+BCC")
        
        # 判断是否有金属间化合物
        if delta > 0.08:
            phases.append("intermetallic")
        
        return PhasePrediction(
            phases=phases,
            phase_fractions={p: 1.0 / len(phases) for p in phases},
            stable_temperature_range=(300, 1500),
            valence_electron_concentration=vec,
            atomic_size_difference=delta
        )
    
    def _predict_properties(self, composition: AlloyComposition) -> MechanicalProperties:
        """预测力学性能"""
        elements = composition.elements
        
        # 基于混合规则估算
        yield_strength = 0
        density = 0
        tm_avg = 0
        
        for elem, frac in elements.items():
            frac_dec = frac / 100
            yield_strength += self._estimate_element_strength(elem) * frac_dec
            density += self.database.get_element_property(elem, "density") * frac_dec
            tm_avg += self.database.get_element_property(elem, "Tm") * frac_dec
        
        # 固溶强化贡献
        ss_strengthening = len(elements) * 20  # MPa per element
        
        # 晶粒细化强化（假设5μm晶粒）
        hall_petch = 300  # MPa
        
        yield_strength += ss_strengthening + hall_petch
        
        # 加工硬化
        uts = yield_strength * 1.5
        
        # 塑性（高温合金通常较差）
        elongation = max(5, 30 - len(elements) * 3)
        
        # 硬度估算
        hardness = yield_strength * 0.3
        
        # 杨氏模量
        youngs = 200 - density * 5  # 粗略估算
        
        return MechanicalProperties(
            yield_strength=yield_strength,
            ultimate_tensile_strength=uts,
            elongation=elongation,
            hardness=hardness,
            youngs_modulus=youngs
        )
    
    def _estimate_element_strength(self, element: str) -> float:
        """估算元素的基础强度贡献"""
        strength_map = {
            "Fe": 250, "Ni": 200, "Co": 280, "Cr": 280,
            "Ti": 400, "Al": 100, "Cu": 150, "Mn": 350,
            "Mo": 500, "W": 600, "Nb": 450, "Ta": 500
        }
        return strength_map.get(element, 200)
    
    def _estimate_cost(self, composition: AlloyComposition) -> float:
        """估算合金成本"""
        total_cost = 0
        for elem, frac in composition.elements.items():
            cost = self.database.get_element_property(elem, "cost")
            total_cost += cost * frac / 100
        return total_cost
    
    def _estimate_density(self, composition: AlloyComposition) -> float:
        """估算密度"""
        density = 0
        for elem, frac in composition.elements.items():
            d = self.database.get_element_property(elem, "density")
            density += d * frac / 100
        return density


class RoboticAlloySynthesizer:
    """机器人合金合成系统"""
    
    def __init__(self):
        self.lab = AutonomousLab()
        self._setup_equipment()
    
    def _setup_equipment(self):
        """设置设备"""
        # 电弧熔炼炉
        self.lab.add_robot("melting_furnace", SimulatedRobot("ArcMelter"))
        
        # 真空系统
        self.lab.add_robot("vacuum_system", SimulatedRobot("VacuumPump"))
        
        # 机械臂（原料处理）
        self.lab.add_robot("manipulator", SimulatedRobot("MaterialHandler"))
    
    async def synthesize_alloy(self, candidate: AlloyCandidate) -> bool:
        """合成合金"""
        logger.info(f"Synthesizing alloy: {candidate.name}")
        logger.info(f"Composition: {candidate.composition.elements}")
        
        try:
            # 1. 称量和混合原料
            await self._prepare_charge(candidate.composition)
            
            # 2. 真空电弧熔炼
            await self._arc_melt()
            
            # 3. 翻转重熔（均匀化）
            for _ in range(4):
                await self._flip_and_remelt()
            
            # 4. 冷却
            await self._cool_down()
            
            candidate.synthesis_success = True
            return True
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            candidate.synthesis_success = False
            return False
    
    async def _prepare_charge(self, composition: AlloyComposition):
        """准备原料"""
        instructions = []
        
        for elem, at_percent in composition.elements.items():
            # 计算质量
            weight = self._calculate_element_weight(elem, at_percent)
            
            instructions.append(RobotInstruction(
                command=RobotCommand.PICK_UP,
                parameters={"material": elem, "weight": weight}
            ))
            
            instructions.append(RobotInstruction(
                command=RobotCommand.PLACE,
                parameters={"destination": "crucible"}
            ))
        
        for instruction in instructions:
            await self.lab.robots["manipulator"].execute_instruction(instruction)
    
    def _calculate_element_weight(self, element: str, at_percent: float) -> float:
        """计算元素质量"""
        atomic_weight = self.database.element_properties.get(element, {}).get("weight", 50)
        total_weight = 10.0  # 总质量10g
        return total_weight * at_percent / 100 * atomic_weight / 50
    
    async def _arc_melt(self):
        """电弧熔炼"""
        # 抽真空
        await self.lab.robots["vacuum_system"].execute_instruction(
            RobotInstruction(command=RobotCommand.DISPENSE, 
                           parameters={"vacuum_level": 1e-3})
        )
        
        # 充氩气
        await self.lab.robots["vacuum_system"].execute_instruction(
            RobotInstruction(command=RobotCommand.DISPENSE,
                           parameters={"gas": "Ar", "pressure": 0.5})
        )
        
        # 熔炼
        await self.lab.robots["melting_furnace"].execute_instruction(
            RobotInstruction(command=RobotCommand.HEAT,
                           parameters={"temperature": 2000, "time": 120})
        )
    
    async def _flip_and_remelt(self):
        """翻转重熔"""
        await self.lab.robots["manipulator"].execute_instruction(
            RobotInstruction(command=RobotCommand.MOVE_TO,
                           parameters={"action": "flip"})
        )
        
        await self._arc_melt()
    
    async def _cool_down(self):
        """冷却"""
        await self.lab.robots["melting_furnace"].execute_instruction(
            RobotInstruction(command=RobotCommand.COOL,
                           parameters={"rate": 100})
        )


class AlloyTester:
    """合金测试系统"""
    
    def __init__(self):
        self.lab = AutonomousLab()
    
    async def perform_tests(self, candidate: AlloyCandidate) -> MechanicalProperties:
        """执行测试"""
        logger.info(f"Testing alloy: {candidate.name}")
        
        # 模拟测试过程
        await asyncio.sleep(0.1)
        
        # 基于预测添加噪声
        predicted = candidate.predicted_properties
        
        noise_factor = np.random.normal(1.0, 0.1)
        
        actual = MechanicalProperties(
            yield_strength=predicted.yield_strength * noise_factor,
            ultimate_tensile_strength=predicted.ultimate_tensile_strength * noise_factor,
            elongation=predicted.elongation / noise_factor,  # 反向噪声
            hardness=predicted.hardness * noise_factor,
            youngs_modulus=predicted.youngs_modulus * noise_factor
        )
        
        candidate.actual_properties = actual
        return actual


class AutonomousAlloyDevelopment:
    """自驱动合金开发系统"""
    
    def __init__(self):
        self.designer = AlloyDesigner()
        self.synthesizer = RoboticAlloySynthesizer()
        self.tester = AlloyTester()
        
        self.candidates: List[AlloyCandidate] = []
        self.tested_alloys: List[AlloyCandidate] = []
    
    async def run_development(self,
                             target_system: AlloySystem,
                             target_yield_strength: float = 1000.0,
                             max_iterations: int = 10) -> List[AlloyCandidate]:
        """运行合金开发流程"""
        logger.info(f"Starting alloy development for {target_system.value}")
        logger.info(f"Target yield strength: {target_yield_strength} MPa")
        
        for iteration in range(max_iterations):
            logger.info(f"Iteration {iteration + 1}/{max_iterations}")
            
            # 1. 设计候选
            candidates = self.designer.design_alloy(
                target_system,
                {"yield_strength": target_yield_strength},
                n_candidates=5
            )
            
            # 2. 筛选（基于预测）
            promising = [
                c for c in candidates
                if c.predicted_properties and 
                c.predicted_properties.yield_strength >= target_yield_strength * 0.7
            ]
            
            # 3. 合成
            for candidate in promising:
                success = await self.synthesizer.synthesize_alloy(candidate)
                
                if success:
                    # 4. 测试
                    await self.tester.perform_tests(candidate)
                    self.tested_alloys.append(candidate)
            
            self.candidates.extend(candidates)
            
            # 检查是否达到目标
            successful = [
                a for a in self.tested_alloys
                if a.actual_properties and 
                a.actual_properties.yield_strength >= target_yield_strength
            ]
            
            if successful:
                logger.info(f"Target achieved! Found {len(successful)} alloys.")
                break
        
        return self._get_best_alloys()
    
    def _get_best_alloys(self, n: int = 5) -> List[AlloyCandidate]:
        """获取最优合金"""
        tested = [a for a in self.tested_alloys if a.actual_properties]
        
        if not tested:
            return []
        
        tested.sort(key=lambda a: a.actual_properties.yield_strength, reverse=True)
        return tested[:n]
    
    def generate_report(self) -> str:
        """生成开发报告"""
        report = ["=" * 60]
        report.append("自主合金开发报告")
        report.append("=" * 60)
        report.append("")
        report.append(f"设计候选数: {len(self.candidates)}")
        report.append(f"成功合成数: {sum(1 for c in self.candidates if c.synthesis_success)}")
        report.append(f"完成测试数: {len(self.tested_alloys)}")
        report.append("")
        
        best = self._get_best_alloys(5)
        if best:
            report.append("最优合金:")
            for i, alloy in enumerate(best, 1):
                props = alloy.actual_properties
                report.append(f"  {i}. {alloy.name} ({alloy.system.value})")
                report.append(f"     成分: {alloy.composition.elements}")
                report.append(f"     屈服强度: {props.yield_strength:.0f} MPa")
                report.append(f"     延伸率: {props.elongation:.1f}%")
                report.append(f"     密度: {alloy.predicted_density:.2f} g/cm³")
                report.append(f"     成本: ${alloy.predicted_cost:.2f}/kg")
        
        return "\n".join(report)


# ==================== 主入口函数 ====================

async def develop_alloy(
    system: str = "high_entropy_alloy",
    target_strength: float = 1000.0,
    max_iterations: int = 10
) -> List[AlloyCandidate]:
    """开发合金"""
    alloy_system = AlloySystem(system)
    
    development = AutonomousAlloyDevelopment()
    alloys = await development.run_development(
        alloy_system,
        target_strength,
        max_iterations
    )
    
    print(development.generate_report())
    
    return alloys


# 示例用法
if __name__ == "__main__":
    best_alloys = asyncio.run(develop_alloy(
        system="high_entropy_alloy",
        target_strength=1200.0,
        max_iterations=10
    ))
    
    print(f"\nDevelopment complete. Found {len(best_alloys)} high-performance alloys.")
