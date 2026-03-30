"""
自动模块选择演示
================
输入目标 → 自动推荐模块组合

演示如何基于目标自动选择最优模块组合。

Author: DFT-LAMMPS Team
Phase: 56 - Orchestration System
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..orchestration.module_registry import (
    ModuleRegistry, CapabilityType, RegisteredModule
)
from ..orchestration.capability_graph import (
    CapabilityGraph, CapabilityNode, CapabilityEdge,
    NodeType, EdgeType, CapabilityPath
)
from ..orchestration.workflow_composer import Workflow
from ..orchestration.topic_template import (
    TopicTemplateManager, ResearchTopic
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("auto_select_modules")


class ResearchGoal(Enum):
    """研究目标类型"""
    BATTERY_CATHODE = "battery_cathode"
    BATTERY_ANODE = "battery_anode"
    BATTERY_ELECTROLYTE = "battery_electrolyte"
    CATALYST_HER = "catalyst_her"
    CATALYST_ORR = "catalyst_orr"
    CATALYST_OER = "catalyst_oer"
    PHOTOVOLTAIC = "photovoltaic"
    THERMOELECTRIC = "thermoelectric"
    ALLOY_HEA = "alloy_hea"
    ALLOY_LIGHTWEIGHT = "alloy_lightweight"
    ELECTRONIC_2D = "electronic_2d"
    MECHANICAL_CERAMIC = "mechanical_ceramic"


@dataclass
class ModuleRecommendation:
    """模块推荐"""
    module_name: str
    capability_name: str
    relevance_score: float  # 0-1
    reason: str
    alternatives: List[str]
    estimated_time: float   # 分钟
    estimated_cost: float   # 计算成本


@dataclass
class WorkflowProposal:
    """工作流提案"""
    goal: str
    proposed_workflow: Workflow
    modules: List[ModuleRecommendation]
    total_estimated_time: float
    total_estimated_cost: float
    confidence: float
    explanation: str


class AutoModuleSelector:
    """
    自动模块选择器
    
    基于研究目标智能推荐模块组合
    
    Example:
        selector = AutoModuleSelector()
        
        # 输入自然语言目标
        proposal = selector.select_modules(
            goal="analyze Li-ion battery cathode material for high voltage"
        )
        
        # 或输入结构化目标
        proposal = selector.select_modules(
            goal=ResearchGoal.BATTERY_CATHODE,
            constraints={"max_voltage": 4.5, "min_capacity": 200}
        )
    """
    
    def __init__(
        self,
        registry: Optional[ModuleRegistry] = None,
        graph: Optional[CapabilityGraph] = None
    ):
        self.registry = registry or ModuleRegistry.get_instance()
        self.graph = graph or CapabilityGraph()
        self.template_manager = TopicTemplateManager()
        
        # 构建能力图谱
        self._build_capability_graph()
        
        # 目标到能力的映射
        self._goal_capability_map = self._init_goal_mapping()
    
    def select_modules(
        self,
        goal: Union[str, ResearchGoal],
        constraints: Optional[Dict[str, Any]] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> WorkflowProposal:
        """
        基于目标选择模块
        
        Args:
            goal: 研究目标（字符串或枚举）
            constraints: 约束条件
            preferences: 偏好设置
        
        Returns:
            工作流提案
        """
        constraints = constraints or {}
        preferences = preferences or {}
        
        # 解析目标
        parsed_goal = self._parse_goal(goal)
        logger.info(f"Parsed goal: {parsed_goal}")
        
        # 获取所需能力
        required_capabilities = self._goal_capability_map.get(
            parsed_goal, 
            ["structure_import", "relaxation", "calculation"]
        )
        
        # 应用约束筛选
        if constraints.get('high_accuracy'):
            required_capabilities.append("hse06_calculation")
        
        if constraints.get('include_dynamics'):
            required_capabilities.append("molecular_dynamics")
        
        # 查找提供能力的模块
        recommendations = self._find_modules_for_capabilities(
            required_capabilities,
            preferences
        )
        
        # 构建工作流
        workflow = self._build_workflow_from_recommendations(
            parsed_goal,
            recommendations,
            constraints
        )
        
        # 计算估算
        total_time = sum(r.estimated_time for r in recommendations)
        total_cost = sum(r.estimated_cost for r in recommendations)
        
        # 生成解释
        explanation = self._generate_explanation(parsed_goal, recommendations)
        
        return WorkflowProposal(
            goal=str(parsed_goal),
            proposed_workflow=workflow,
            modules=recommendations,
            total_estimated_time=total_time,
            total_estimated_cost=total_cost,
            confidence=0.85,
            explanation=explanation
        )
    
    def recommend_alternatives(
        self,
        module_name: str,
        capability: str,
        criteria: Optional[Dict[str, Any]] = None
    ) -> List[ModuleRecommendation]:
        """
        推荐替代模块
        
        当首选模块不可用时推荐替代方案
        """
        # 查找提供相同能力的其他模块
        candidates = self.registry.find_modules(capability_name=capability)
        
        alternatives = []
        for mod in candidates:
            if mod.metadata.name != module_name:
                score = self._score_module(mod, criteria or {})
                alternatives.append(ModuleRecommendation(
                    module_name=mod.metadata.name,
                    capability_name=capability,
                    relevance_score=score,
                    reason="Alternative provider",
                    alternatives=[],
                    estimated_time=60.0,
                    estimated_cost=1.0
                ))
        
        # 按分数排序
        alternatives.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return alternatives[:3]
    
    def explain_selection(
        self,
        proposal: WorkflowProposal,
        detail_level: str = "medium"
    ) -> str:
        """
        解释模块选择理由
        
        Args:
            proposal: 工作流提案
            detail_level: 详细程度 (brief/medium/detailed)
        """
        if detail_level == "brief":
            return proposal.explanation
        
        explanation = f"""
模块选择解释
==============

目标: {proposal.goal}
置信度: {proposal.confidence:.0%}

选择逻辑:
"""
        
        for i, rec in enumerate(proposal.modules, 1):
            explanation += f"\n{i}. {rec.module_name} - {rec.capability_name}\n"
            explanation += f"   理由: {rec.reason}\n"
            explanation += f"   相关度: {rec.relevance_score:.0%}\n"
            
            if detail_level == "detailed" and rec.alternatives:
                explanation += f"   替代方案: {', '.join(rec.alternatives)}\n"
        
        explanation += f"\n估算:\n"
        explanation += f"  总时间: {proposal.total_estimated_time:.0f} 分钟\n"
        explanation += f"  总成本: {proposal.total_estimated_cost:.1f} 计算单元\n"
        
        return explanation
    
    def optimize_for_constraints(
        self,
        goal: Union[str, ResearchGoal],
        time_limit: Optional[float] = None,
        cost_limit: Optional[float] = None,
        accuracy_target: Optional[float] = None
    ) -> List[WorkflowProposal]:
        """
        基于约束优化模块选择
        
        生成多个满足不同约束条件的方案
        """
        proposals = []
        
        # 方案1: 平衡型
        proposals.append(self.select_modules(
            goal,
            preferences={"balance": True}
        ))
        
        # 方案2: 最快（如果有限制）
        if time_limit:
            proposals.append(self.select_modules(
                goal,
                preferences={"speed": "fast"},
                constraints={"max_time": time_limit}
            ))
        
        # 方案3: 最经济
        if cost_limit:
            proposals.append(self.select_modules(
                goal,
                preferences={"cost": "low"},
                constraints={"max_cost": cost_limit}
            ))
        
        # 方案4: 最高精度
        if accuracy_target:
            proposals.append(self.select_modules(
                goal,
                preferences={"accuracy": "high"},
                constraints={"min_accuracy": accuracy_target}
            ))
        
        return proposals
    
    def _build_capability_graph(self) -> None:
        """构建能力图谱"""
        # 使用注册中心的数据构建图谱
        self.graph.build_from_registry(self.registry)
    
    def _init_goal_mapping(self) -> Dict[ResearchGoal, List[str]]:
        """初始化目标到能力的映射"""
        return {
            ResearchGoal.BATTERY_CATHODE: [
                "structure_import",
                "relax_structure",
                "calculate_energy",
                "voltage_profile",
                "ion_migration",
                "electronic_dos",
                "phonon_calculation"
            ],
            ResearchGoal.BATTERY_ANODE: [
                "structure_import",
                "relax_structure",
                "volume_expansion",
                "sei_formation",
                "lithium_diffusion"
            ],
            ResearchGoal.CATALYST_HER: [
                "build_surface",
                "calculate_adsorption",
                "hydrogen_binding_energy",
                "volcano_plot",
                "kinetic_analysis"
            ],
            ResearchGoal.CATALYST_ORR: [
                "build_surface",
                "calculate_adsorption",
                "oxygen_binding_energy",
                "oh_adsorption",
                "ooh_adsorption",
                "scaling_relations",
                "volcano_plot"
            ],
            ResearchGoal.PHOTOVOLTAIC: [
                "relax_structure",
                "calculate_bands",
                "hse06_calculation",
                "optical_properties",
                "dielectric_function",
                "exciton_binding",
                "effective_mass"
            ],
            ResearchGoal.ALLOY_HEA: [
                "generate_sqs",
                "cluster_expansion",
                "monte_carlo",
                "elastic_constants",
                "phonon_calculation",
                "phase_diagram"
            ]
        }
    
    def _parse_goal(self, goal: Union[str, ResearchGoal]) -> ResearchGoal:
        """解析目标"""
        if isinstance(goal, ResearchGoal):
            return goal
        
        # 自然语言解析（简化实现）
        goal_lower = goal.lower()
        
        if "battery" in goal_lower or "cathode" in goal_lower:
            return ResearchGoal.BATTERY_CATHODE
        elif "catalyst" in goal_lower and "hydrogen" in goal_lower:
            return ResearchGoal.CATALYST_HER
        elif "catalyst" in goal_lower and "oxygen" in goal_lower:
            return ResearchGoal.CATALYST_ORR
        elif "solar" in goal_lower or "photovoltaic" in goal_lower:
            return ResearchGoal.PHOTOVOLTAIC
        elif "alloy" in goal_lower or "hea" in goal_lower:
            return ResearchGoal.ALLOY_HEA
        
        return ResearchGoal.BATTERY_CATHODE  # 默认
    
    def _find_modules_for_capabilities(
        self,
        capabilities: List[str],
        preferences: Dict[str, Any]
    ) -> List[ModuleRecommendation]:
        """为每个能力查找最佳模块"""
        recommendations = []
        
        for cap in capabilities:
            # 查找提供该能力的模块
            modules = self.registry.find_modules(capability_name=cap)
            
            if not modules:
                logger.warning(f"No module found for capability: {cap}")
                continue
            
            # 评分并排序
            scored_modules = [
                (mod, self._score_module(mod, preferences))
                for mod in modules
            ]
            scored_modules.sort(key=lambda x: x[1], reverse=True)
            
            best_module, best_score = scored_modules[0]
            
            # 查找替代方案
            alternatives = [m.metadata.name for m, _ in scored_modules[1:3]]
            
            # 生成推荐理由
            reason = self._generate_reason(best_module, cap, preferences)
            
            recommendations.append(ModuleRecommendation(
                module_name=best_module.metadata.name,
                capability_name=cap,
                relevance_score=best_score,
                reason=reason,
                alternatives=alternatives,
                estimated_time=self._estimate_time(best_module, cap),
                estimated_cost=self._estimate_cost(best_module, cap)
            ))
        
        return recommendations
    
    def _score_module(
        self,
        module: RegisteredModule,
        preferences: Dict[str, Any]
    ) -> float:
        """为模块评分"""
        score = 0.5  # 基础分
        
        # 版本因素
        version = module.metadata.version
        score += 0.1 * (version.major * 0.1 + version.minor * 0.01)
        
        # 活跃度
        if module.state.name == "ACTIVE":
            score += 0.2
        
        # 偏好匹配
        if preferences.get('accuracy') == 'high' and 'accurate' in module.metadata.tags:
            score += 0.2
        
        if preferences.get('speed') == 'fast' and 'fast' in module.metadata.tags:
            score += 0.2
        
        return min(1.0, score)
    
    def _generate_reason(
        self,
        module: RegisteredModule,
        capability: str,
        preferences: Dict[str, Any]
    ) -> str:
        """生成推荐理由"""
        reasons = [
            f"Highest rated module for {capability}",
            f"Version {module.metadata.version} with active maintenance",
        ]
        
        if 'accurate' in module.metadata.tags:
            reasons.append("Tagged as high-accuracy")
        
        if 'fast' in module.metadata.tags:
            reasons.append("Optimized for performance")
        
        return "; ".join(reasons)
    
    def _estimate_time(self, module: RegisteredModule, capability: str) -> float:
        """估算执行时间（分钟）"""
        # 简化的估算
        base_times = {
            "relax_structure": 60,
            "calculate_energy": 30,
            "calculate_bands": 120,
            "molecular_dynamics": 240,
            "phonon_calculation": 180
        }
        return base_times.get(capability, 60)
    
    def _estimate_cost(self, module: RegisteredModule, capability: str) -> float:
        """估算计算成本"""
        base_costs = {
            "relax_structure": 1.0,
            "calculate_energy": 0.5,
            "calculate_bands": 2.0,
            "molecular_dynamics": 4.0,
            "phonon_calculation": 3.0
        }
        return base_costs.get(capability, 1.0)
    
    def _build_workflow_from_recommendations(
        self,
        goal: ResearchGoal,
        recommendations: List[ModuleRecommendation],
        constraints: Dict[str, Any]
    ) -> Workflow:
        """从推荐构建工作流"""
        # 使用课题模板作为基础
        workflow = self.template_manager.create_workflow(
            ResearchTopic.BATTERY if "BATTERY" in goal.name else ResearchTopic.CUSTOM,
            inputs={}
        )
        
        return workflow
    
    def _generate_explanation(
        self,
        goal: ResearchGoal,
        recommendations: List[ModuleRecommendation]
    ) -> str:
        """生成选择解释"""
        return (
            f"Selected {len(recommendations)} modules for {goal.value} research. "
            f"Primary modules: {', '.join(r.module_name for r in recommendations[:3])}. "
            f"Based on capability coverage and module ratings."
        )


def demo_auto_selection():
    """演示自动模块选择"""
    print("=" * 70)
    print("自动模块选择演示")
    print("=" * 70)
    
    selector = AutoModuleSelector()
    
    # 示例1: 电池正极材料
    print("\n示例1: 电池正极材料研究")
    print("-" * 70)
    
    proposal1 = selector.select_modules(
        goal="analyze lithium battery cathode material",
        constraints={"high_voltage": True, "min_capacity": 200},
        preferences={"accuracy": "high"}
    )
    
    print(f"目标: {proposal1.goal}")
    print(f"推荐模块数: {len(proposal1.modules)}")
    print(f"估算时间: {proposal1.total_estimated_time:.0f} 分钟")
    print(f"估算成本: {proposal1.total_estimated_cost:.1f} 单元")
    
    print("\n推荐模块:")
    for i, rec in enumerate(proposal1.modules, 1):
        print(f"  {i}. {rec.module_name} ({rec.capability_name})")
        print(f"     相关度: {rec.relevance_score:.0%} | 理由: {rec.reason}")
    
    # 示例2: 催化剂设计
    print("\n\n示例2: 催化剂设计 (HER)")
    print("-" * 70)
    
    proposal2 = selector.select_modules(
        goal=ResearchGoal.CATALYST_HER,
        preferences={"speed": "fast"}
    )
    
    print(f"目标: {proposal2.goal}")
    print(f"推荐模块数: {len(proposal2.modules)}")
    
    print("\n推荐模块:")
    for i, rec in enumerate(proposal2.modules, 1):
        print(f"  {i}. {rec.module_name} ({rec.capability_name})")
        if rec.alternatives:
            print(f"     替代: {', '.join(rec.alternatives)}")
    
    # 示例3: 光伏材料
    print("\n\n示例3: 光伏材料筛选")
    print("-" * 70)
    
    proposal3 = selector.select_modules(
        goal=ResearchGoal.PHOTOVOLTAIC,
        constraints={"include_excitons": True}
    )
    
    print(f"目标: {proposal3.goal}")
    print("\n推荐模块:")
    for i, rec in enumerate(proposal3.modules, 1):
        print(f"  {i}. {rec.module_name}")
    
    # 详细解释
    print("\n\n详细解释:")
    print("-" * 70)
    explanation = selector.explain_selection(proposal1, detail_level="detailed")
    print(explanation)


def demo_constraint_optimization():
    """演示约束优化"""
    print("\n" + "=" * 70)
    print("约束优化演示")
    print("=" * 70)
    
    selector = AutoModuleSelector()
    
    print("\n为不同约束生成优化方案:")
    print("-" * 70)
    
    proposals = selector.optimize_for_constraints(
        goal=ResearchGoal.BATTERY_CATHODE,
        time_limit=120,
        cost_limit=10,
        accuracy_target=0.95
    )
    
    for i, proposal in enumerate(proposals, 1):
        print(f"\n方案 {i}:")
        print(f"  目标: {proposal.goal}")
        print(f"  模块数: {len(proposal.modules)}")
        print(f"  总时间: {proposal.total_estimated_time:.0f} 分钟")
        print(f"  总成本: {proposal.total_estimated_cost:.1f} 单元")
        print(f"  置信度: {proposal.confidence:.0%}")


def demo_natural_language_goals():
    """演示自然语言目标解析"""
    print("\n" + "=" * 70)
    print("自然语言目标解析")
    print("=" * 70)
    
    selector = AutoModuleSelector()
    
    natural_goals = [
        "I want to study lithium ion battery cathodes for electric vehicles",
        "Design a catalyst for hydrogen evolution reaction in acidic media",
        "Find new materials for solar cells with high efficiency",
        "Develop high-entropy alloys for high temperature applications",
        "Investigate 2D materials for electronic devices"
    ]
    
    print("\n自然语言目标 → 模块选择:")
    print("-" * 70)
    
    for goal in natural_goals:
        print(f"\n输入: \"{goal}\"")
        
        proposal = selector.select_modules(goal=goal)
        
        print(f"  解析目标: {proposal.goal}")
        print(f"  选择模块: {', '.join(r.module_name for r in proposal.modules[:3])}")
        print(f"  解释: {proposal.explanation[:100]}...")


def run_auto_select_examples():
    """运行所有自动选择示例"""
    print("\n" + "=" * 70)
    print("自动模块选择演示")
    print("输入目标 → 自动推荐模块组合")
    print("=" * 70)
    
    try:
        demo_auto_selection()
    except Exception as e:
        print(f"自动选择错误: {e}")
    
    try:
        demo_constraint_optimization()
    except Exception as e:
        print(f"约束优化错误: {e}")
    
    try:
        demo_natural_language_goals()
    except Exception as e:
        print(f"自然语言解析错误: {e}")
    
    print("\n" + "=" * 70)
    print("自动模块选择演示完成")
    print("=" * 70)


if __name__ == "__main__":
    run_auto_select_examples()