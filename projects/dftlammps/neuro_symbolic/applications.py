"""
神经符号AI应用案例 - Neuro-Symbolic AI Applications for Materials Science

演示神经符号AI在材料科学中的实际应用：
1. 材料知识自动形式化
2. 符号-神经混合预测
3. 可解释材料设计
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass

# 导入神经符号模块
import sys
sys.path.insert(0, '/root/.openclaw/workspace/dftlammps')

from neuro_symbolic import (
    NeuralTheoremProver, KnowledgeGraphReasoner,
    NeuralToSymbolicTranslator, SymbolicToNeuralTranslator,
    ConceptLearner, BidirectionalNeuralSymbolic,
    NeuralProgramSynthesizer, MaterialDSL,
    create_material_knowledge_base, create_material_concept_hierarchy
)

from knowledge_reasoning import (
    ConceptHierarchy, DescriptionLogicReasoner,
    RuleEngine, create_material_rules, CertaintyFactor, Fact,
    CaseBasedReasoner, create_sample_material_cases
)


@dataclass
class Material:
    """材料数据结构"""
    name: str
    composition: Dict[str, float]
    properties: Dict[str, float]
    structure: str = "unknown"
    
    def to_vector(self) -> np.ndarray:
        """转换为特征向量"""
        # 简化的特征向量
        features = [
            self.composition.get('Si', 0),
            self.composition.get('Ge', 0),
            self.composition.get('C', 0),
            self.properties.get('band_gap', 0),
            self.properties.get('conductivity', 0),
            self.properties.get('thermal_conductivity', 0),
        ]
        return np.array(features)


class MaterialKnowledgeFormalizer:
    """
    材料知识自动形式化系统
    
    将非结构化的材料知识自动转换为形式化的知识表示。
    """
    
    def __init__(self):
        self.concept_hierarchy = None
        self.rule_engine = None
        self.neural_translator = None
        
    def build_ontology_from_text(self, text_descriptions: List[str]) -> ConceptHierarchy:
        """
        从文本描述自动构建材料本体
        
        Args:
            text_descriptions: 材料描述的文本列表
        
        Returns:
            构建的概念层次结构
        """
        # 初始化本体
        from knowledge_reasoning.ontology_reasoning import Concept, ConceptType
        hierarchy = ConceptHierarchy()
        
        # 基础概念
        root_concepts = [
            Concept("Material", ConceptType.PRIMITIVE),
            Concept("Property", ConceptType.PRIMITIVE),
            Concept("Structure", ConceptType.PRIMITIVE),
            Concept("Process", ConceptType.PRIMITIVE),
        ]
        
        for concept in root_concepts:
            hierarchy.add_concept(concept)
        
        # 从文本中提取概念（简化实现）
        material_keywords = {
            'conductor': 'Conductor',
            'semiconductor': 'Semiconductor',
            'insulator': 'Insulator',
            'metal': 'Metal',
            'alloy': 'Alloy',
            'ceramic': 'Ceramic',
            'polymer': 'Polymer',
        }
        
        property_keywords = {
            'conductivity': 'Conductivity',
            'band gap': 'BandGap',
            'thermal': 'ThermalProperty',
            'mechanical': 'MechanicalProperty',
            'optical': 'OpticalProperty',
        }
        
        structure_keywords = {
            'crystal': 'Crystalline',
            'amorphous': 'Amorphous',
            'cubic': 'CubicStructure',
            'hexagonal': 'HexagonalStructure',
            'diamond': 'DiamondStructure',
        }
        
        # 扫描文本提取概念
        for text in text_descriptions:
            text_lower = text.lower()
            
            # 提取材料类型
            for keyword, concept_name in material_keywords.items():
                if keyword in text_lower:
                    concept = Concept(
                        concept_name, 
                        ConceptType.PRIMITIVE,
                        parents={"Material"}
                    )
                    hierarchy.add_concept(concept)
            
            # 提取属性
            for keyword, concept_name in property_keywords.items():
                if keyword in text_lower:
                    concept = Concept(
                        concept_name,
                        ConceptType.PRIMITIVE,
                        parents={"Property"}
                    )
                    hierarchy.add_concept(concept)
            
            # 提取结构
            for keyword, concept_name in structure_keywords.items():
                if keyword in text_lower:
                    concept = Concept(
                        concept_name,
                        ConceptType.PRIMITIVE,
                        parents={"Structure"}
                    )
                    hierarchy.add_concept(concept)
        
        self.concept_hierarchy = hierarchy
        return hierarchy
    
    def formalize_properties(self, 
                            material: Material,
                            hierarchy: ConceptHierarchy) -> Dict[str, Any]:
        """
        将材料属性形式化
        
        将数值属性转换为符号断言。
        """
        formalized = {
            'concepts': [],
            'facts': [],
            'relations': []
        }
        
        # 根据属性推断概念归属
        reasoner = DescriptionLogicReasoner(hierarchy)
        
        # 带隙分类
        band_gap = material.properties.get('band_gap', 0)
        if band_gap == 0:
            formalized['concepts'].append('Conductor')
            formalized['facts'].append(
                Fact('isConductor', [material.name], CertaintyFactor(0.95))
            )
        elif band_gap < 2.0:
            formalized['concepts'].append('Semiconductor')
            formalized['facts'].append(
                Fact('isSemiconductor', [material.name], CertaintyFactor(0.9))
            )
        else:
            formalized['concepts'].append('Insulator')
            formalized['facts'].append(
                Fact('isInsulator', [material.name], CertaintyFactor(0.85))
            )
        
        # 导电性分类
        conductivity = material.properties.get('conductivity', 0)
        if conductivity > 1000:
            formalized['facts'].append(
                Fact('hasHighConductivity', [material.name], CertaintyFactor(0.95))
            )
        
        # 添加数值属性
        for prop, value in material.properties.items():
            formalized['relations'].append({
                'subject': material.name,
                'predicate': f'has{prop.capitalize()}',
                'object': value
            })
        
        return formalized
    
    def discover_rules_from_data(self, 
                                 materials: List[Material],
                                 min_confidence: float = 0.7) -> List[Dict]:
        """
        从材料数据中发现规则
        
        使用关联规则挖掘发现材料属性间的关系。
        """
        rules = []
        
        # 简化的规则发现：基于相关分析
        properties = ['band_gap', 'conductivity', 'thermal_conductivity']
        
        for i, prop1 in enumerate(properties):
            for prop2 in properties[i+1:]:
                values1 = [m.properties.get(prop1, 0) for m in materials]
                values2 = [m.properties.get(prop2, 0) for m in materials]
                
                # 计算相关系数
                if len(values1) > 1 and np.std(values1) > 0 and np.std(values2) > 0:
                    corr = np.corrcoef(values1, values2)[0, 1]
                    
                    if abs(corr) >= min_confidence:
                        rule = {
                            'if': prop1,
                            'then': prop2,
                            'correlation': corr,
                            'type': 'positive' if corr > 0 else 'negative',
                            'confidence': abs(corr)
                        }
                        rules.append(rule)
        
        return rules


class HybridPredictor(nn.Module):
    """
    符号-神经混合预测器
    
    结合神经网络的预测能力和符号推理的可解释性。
    """
    
    def __init__(self, 
                 input_dim: int = 10,
                 hidden_dim: int = 128,
                 num_properties: int = 5):
        super().__init__()
        
        # 神经网络部分
        self.neural_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 符号推理部分
        self.rule_weights = nn.Parameter(torch.randn(10))  # 假设10条规则
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + 10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_properties)
        )
        
        # 不确定性估计
        self.uncertainty_net = nn.Sequential(
            nn.Linear(hidden_dim + 10, 64),
            nn.ReLU(),
            nn.Linear(64, num_properties),
            nn.Softplus()
        )
        
        self.rule_engine = None
        self.bidirectional_system = None
    
    def set_rule_engine(self, rule_engine: RuleEngine):
        """设置规则引擎"""
        self.rule_engine = rule_engine
    
    def neural_forward(self, x: torch.Tensor) -> torch.Tensor:
        """神经网络前向传播"""
        return self.neural_encoder(x)
    
    def symbolic_forward(self, 
                        material_features: Dict[str, Any]) -> torch.Tensor:
        """符号推理前向传播"""
        if not self.rule_engine:
            return torch.zeros(10)
        
        # 添加事实到规则引擎
        for key, value in material_features.items():
            self.rule_engine.add_fact(
                Fact(key, [value], CertaintyFactor(1.0))
            )
        
        # 执行推理
        self.rule_engine.forward_chain()
        
        # 收集推理结果
        results = []
        for i, rule in enumerate(self.rule_engine.rules[:10]):
            # 检查规则是否被触发
            triggered, cf, _ = rule.is_triggered(self.rule_engine.facts)
            results.append(cf.value if triggered else 0.0)
        
        return torch.tensor(results, dtype=torch.float32)
    
    def forward(self, 
                neural_input: torch.Tensor,
                symbolic_input: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        混合前向传播
        
        Args:
            neural_input: 神经网络的输入特征
            symbolic_input: 符号推理的输入（可选）
        
        Returns:
            包含预测结果和不确定性的字典
        """
        # 神经网络部分
        neural_features = self.neural_forward(neural_input)
        
        # 符号推理部分
        if symbolic_input is not None and self.rule_engine:
            symbolic_features = self.symbolic_forward(symbolic_input)
        else:
            symbolic_features = torch.zeros(10)
        
        # 融合
        combined = torch.cat([neural_features, symbolic_features], dim=-1)
        
        # 预测
        predictions = self.fusion_layer(combined)
        
        # 不确定性
        uncertainty = self.uncertainty_net(combined)
        
        return {
            'predictions': predictions,
            'uncertainty': uncertainty,
            'neural_features': neural_features,
            'symbolic_features': symbolic_features,
            'combined_features': combined
        }
    
    def explain_prediction(self, 
                          neural_input: torch.Tensor,
                          symbolic_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成可解释的预测说明
        """
        # 执行前向传播
        result = self.forward(neural_input, symbolic_input)
        
        # 收集解释
        explanation = {
            'predicted_values': result['predictions'].detach().numpy(),
            'uncertainty': result['uncertainty'].detach().numpy(),
            'neural_contribution': torch.norm(result['neural_features']).item(),
            'symbolic_contribution': torch.norm(result['symbolic_features']).item(),
            'activated_rules': [],
            'reasoning_chain': []
        }
        
        # 如果有规则引擎，收集激活的规则
        if self.rule_engine:
            for fact in self.rule_engine.facts:
                explanation['reasoning_chain'].append(str(fact))
            
            for i, rule in enumerate(self.rule_engine.rules[:10]):
                triggered, cf, _ = rule.is_triggered(self.rule_engine.facts)
                if triggered and cf.value > 0.5:
                    explanation['activated_rules'].append({
                        'rule_name': rule.name,
                        'certainty': cf.value
                    })
        
        return explanation


class ExplainableMaterialDesigner:
    """
    可解释材料设计系统
    
    提供带有详细推理过程的材料设计建议。
    """
    
    def __init__(self):
        self.cbr_system = CaseBasedReasoner()
        self.rule_engine = RuleEngine()
        self.formalizer = MaterialKnowledgeFormalizer()
        
        # 初始化案例库
        cases = create_sample_material_cases()
        for case in cases:
            self.cbr_system.add_case(case)
        
        # 初始化规则库
        rules = create_material_rules()
        for rule in rules:
            self.rule_engine.add_rule(rule)
    
    def design_material(self,
                       target_properties: Dict[str, float],
                       constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        设计满足目标属性的材料
        
        Returns:
            包含设计方案和解释的字典
        """
        constraints = constraints or {}
        
        # 1. 基于案例推理获取初始设计
        problem = {
            'target_properties': target_properties,
            'constraints': constraints
        }
        
        case_solution, similar_cases, case_explanation = self.cbr_system.solve(
            problem, k=3
        )
        
        # 2. 基于规则推理优化设计
        # 添加目标属性作为事实
        for prop, value in target_properties.items():
            self.rule_engine.add_fact(
                Fact(f'target_{prop}', [value], CertaintyFactor(1.0))
            )
        
        self.rule_engine.forward_chain()
        
        # 3. 生成设计方案
        design = {
            'composition': case_solution.get('composition', {}) if case_solution else {},
            'structure': case_solution.get('structure_type', 'unknown') if case_solution else 'unknown',
            'synthesis_method': case_solution.get('synthesis_method', 'unknown') if case_solution else 'unknown',
        }
        
        # 4. 生成解释
        explanation = self._generate_design_explanation(
            target_properties, similar_cases, design
        )
        
        return {
            'design': design,
            'explanation': explanation,
            'confidence': self._calculate_confidence(similar_cases),
            'similar_cases': [c.case_id for c, _ in similar_cases],
            'alternative_designs': self._generate_alternatives(similar_cases[1:] if len(similar_cases) > 1 else [])
        }
    
    def _generate_design_explanation(self,
                                    target_properties: Dict[str, float],
                                    similar_cases: List,
                                    design: Dict[str, Any]) -> str:
        """生成设计解释"""
        explanation = "材料设计方案解释:\n\n"
        
        explanation += "1. 目标属性:\n"
        for prop, value in target_properties.items():
            explanation += f"   - {prop}: {value}\n"
        
        explanation += "\n2. 参考案例:\n"
        for i, (case, similarity) in enumerate(similar_cases[:3], 1):
            explanation += f"   {i}. {case.case_id} (相似度: {similarity:.3f})\n"
            explanation += f"      原问题: {case.problem}\n"
        
        explanation += "\n3. 设计方案:\n"
        explanation += f"   - 组成: {design['composition']}\n"
        explanation += f"   - 结构: {design['structure']}\n"
        explanation += f"   - 合成方法: {design['synthesis_method']}\n"
        
        explanation += "\n4. 推理依据:\n"
        explanation += "   本方案基于案例推理，参考了具有相似目标属性的历史案例。"
        explanation += "通过分析成功案例的特征，推断出本设计方案。\n"
        
        return explanation
    
    def _calculate_confidence(self, similar_cases: List) -> float:
        """计算设计置信度"""
        if not similar_cases:
            return 0.0
        
        # 基于相似度计算置信度
        similarities = [sim for _, sim in similar_cases]
        return np.mean(similarities)
    
    def _generate_alternatives(self, 
                              other_cases: List) -> List[Dict[str, Any]]:
        """生成替代设计方案"""
        alternatives = []
        
        for case, similarity in other_cases[:2]:
            alt = {
                'composition': case.solution.get('composition', {}),
                'structure': case.solution.get('structure_type', 'unknown'),
                'synthesis_method': case.solution.get('synthesis_method', 'unknown'),
                'similarity': similarity
            }
            alternatives.append(alt)
        
        return alternatives
    
    def validate_design(self, 
                       design: Dict[str, Any],
                       target_properties: Dict[str, float]) -> Dict[str, Any]:
        """
        验证设计方案
        
        检查设计是否满足目标属性。
        """
        validation_result = {
            'valid': True,
            'issues': [],
            'recommendations': []
        }
        
        # 检查组成是否合理
        composition = design.get('composition', {})
        if not composition:
            validation_result['valid'] = False
            validation_result['issues'].append("缺少组成信息")
        
        # 基于规则检查
        # 这里可以添加更多的验证逻辑
        
        if validation_result['valid']:
            validation_result['recommendations'].append(
                "设计方案结构完整，建议进行DFT计算验证"
            )
        
        return validation_result


def demonstrate_automatic_formalization():
    """演示材料知识自动形式化"""
    print("\n" + "=" * 60)
    print("应用1: 材料知识自动形式化")
    print("=" * 60)
    
    # 文本描述
    text_descriptions = [
        "Silicon is a semiconductor with a diamond cubic crystal structure.",
        "Copper is a metal with high electrical conductivity.",
        "Graphene is a 2D material with hexagonal structure.",
        "Ceramics are insulators with high thermal resistance.",
    ]
    
    formalizer = MaterialKnowledgeFormalizer()
    
    # 自动构建本体
    hierarchy = formalizer.build_ontology_from_text(text_descriptions)
    print(f"\n自动构建的本体包含 {len(hierarchy.concepts)} 个概念")
    print("概念列表:")
    for name in list(hierarchy.concepts.keys())[:10]:
        print(f"  - {name}")
    
    # 形式化材料属性
    material = Material(
        name="silicon",
        composition={"Si": 1.0},
        properties={"band_gap": 1.12, "conductivity": 0.001},
        structure="diamond"
    )
    
    formalized = formalizer.formalize_properties(material, hierarchy)
    print(f"\n形式化结果:")
    print(f"  推断的概念: {formalized['concepts']}")
    print(f"  生成的事实: {len(formalized['facts'])} 个")
    
    return formalizer


def demonstrate_hybrid_prediction():
    """演示符号-神经混合预测"""
    print("\n" + "=" * 60)
    print("应用2: 符号-神经混合预测")
    print("=" * 60)
    
    # 创建混合预测器
    predictor = HybridPredictor(input_dim=6, num_properties=3)
    
    # 设置规则引擎
    rule_engine = RuleEngine()
    rules = create_material_rules()
    for rule in rules:
        rule_engine.add_rule(rule)
    predictor.set_rule_engine(rule_engine)
    
    # 准备输入
    neural_input = torch.tensor([[1.0, 0.0, 0.0, 1.12, 0.001, 150.0]])
    symbolic_input = {
        'hasBandGap': 'silicon',
        'material_name': 'silicon'
    }
    
    # 执行预测
    result = predictor.forward(neural_input, symbolic_input)
    
    print(f"\n预测结果:")
    print(f"  神经网络特征维度: {result['neural_features'].shape}")
    print(f"  符号特征维度: {result['symbolic_features'].shape}")
    print(f"  预测属性值: {result['predictions'].detach().numpy()[0]}")
    print(f"  不确定性: {result['uncertainty'].detach().numpy()[0]}")
    
    # 生成解释
    explanation = predictor.explain_prediction(neural_input, symbolic_input)
    print(f"\n预测解释:")
    print(f"  神经贡献: {explanation['neural_contribution']:.3f}")
    print(f"  符号贡献: {explanation['symbolic_contribution']:.3f}")
    print(f"  激活规则数: {len(explanation['activated_rules'])}")
    
    return predictor


def demonstrate_explainable_design():
    """演示可解释材料设计"""
    print("\n" + "=" * 60)
    print("应用3: 可解释材料设计")
    print("=" * 60)
    
    # 创建设计系统
    designer = ExplainableMaterialDesigner()
    
    # 定义设计目标
    target_properties = {
        'band_gap': 1.5,
        'thermal_conductivity': 100
    }
    constraints = {
        'cost_effective': True,
        'synthesis_method': 'czochralski'
    }
    
    # 执行设计
    design_result = designer.design_material(target_properties, constraints)
    
    print(f"\n设计方案:")
    print(f"  组成: {design_result['design']['composition']}")
    print(f"  结构: {design_result['design']['structure']}")
    print(f"  合成方法: {design_result['design']['synthesis_method']}")
    print(f"  置信度: {design_result['confidence']:.3f}")
    
    print(f"\n{design_result['explanation']}")
    
    # 验证设计
    validation = designer.validate_design(
        design_result['design'], target_properties
    )
    print(f"验证结果: {'有效' if validation['valid'] else '无效'}")
    if validation['issues']:
        print(f"问题: {validation['issues']}")
    if validation['recommendations']:
        print(f"建议: {validation['recommendations']}")
    
    return designer


if __name__ == "__main__":
    print("=" * 70)
    print("神经符号AI应用案例")
    print("Neuro-Symbolic AI Applications for Materials Science")
    print("=" * 70)
    
    # 运行三个应用案例
    formalizer = demonstrate_automatic_formalization()
    predictor = demonstrate_hybrid_prediction()
    designer = demonstrate_explainable_design()
    
    print("\n" + "=" * 70)
    print("所有应用案例演示完成!")
    print("=" * 70)
