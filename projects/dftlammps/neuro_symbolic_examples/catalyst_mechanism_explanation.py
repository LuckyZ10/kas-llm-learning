"""
催化剂机理解释 - Catalyst Mechanism Explanation
神经-符号融合解释催化反应机理
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import warnings


class CatalystMechanismExplainer:
    """
    催化剂机理解释器
    结合神经网络和符号推理解释催化反应机理
    """
    
    def __init__(self):
        self.neural_perception = None
        self.symbolic_reasoner = None
        self.bridge = None
        self.xai = None
        self.causal_graph = None
        
        # 催化反应知识库
        self.catalytic_knowledge = self._build_catalytic_knowledge()
    
    def _build_catalytic_knowledge(self) -> Dict[str, Any]:
        """构建催化反应知识库"""
        knowledge = {
            "reaction_types": {
                "oxidation": {
                    "requires": ["active_site", "oxygen_source"],
                    "produces": ["oxidized_product"],
                    "influenced_by": ["temperature", "partial_pressure"]
                },
                "reduction": {
                    "requires": ["active_site", "hydrogen_source"],
                    "produces": ["reduced_product"],
                    "influenced_by": ["temperature", "partial_pressure", "pH"]
                },
                "acid_base": {
                    "requires": ["acid_site", "base_site"],
                    "produces": ["neutralized_product"],
                    "influenced_by": ["temperature", "concentration"]
                }
            },
            "active_site_types": {
                "metallic": {
                    "characteristics": ["electron_donating", "coordinatively_unsaturated"],
                    "typical_metals": ["Pt", "Pd", "Ni", "Cu", "Au"],
                    "optimal_properties": ["high_surface_area", "small_particle_size"]
                },
                "acidic": {
                    "characteristics": ["proton_donating", "lewis_acid"],
                    "typical_materials": ["zeolites", "alumina", "sulfated_zirconia"],
                    "optimal_properties": ["appropriate_acid_strength", "high_pore_volume"]
                },
                "basic": {
                    "characteristics": ["proton_accepting", "electron_donating"],
                    "typical_materials": ["MgO", "CaO", "hydrotalcites"],
                    "optimal_properties": ["basic_site_density", "surface_area"]
                }
            },
            "structure_activity_relationships": {
                "particle_size": {
                    "effect": "smaller_particles_have_more_under_coordinated_sites",
                    "optimal_range": "2-10_nm",
                    "trade_offs": ["stability_decreases_with_size"]
                },
                "surface_area": {
                    "effect": "higher_surface_area_provides_more_active_sites",
                    "optimal_range": "100-500_m2/g",
                    "trade_offs": ["pore_diffusion_limitations"]
                },
                "support_interaction": {
                    "effect": "strong_metal_support_interaction_can_enhance_activity",
                    "optimal_conditions": "moderate_interaction_strength",
                    "trade_offs": ["may_block_active_sites"]
                }
            }
        }
        return knowledge
    
    def initialize_modules(self):
        """初始化所有模块"""
        import sys
        sys.path.insert(0, '/root/.openclaw/workspace/dftlammps')
        
        from neuro_symbolic.neural_perception import (
            NeuralPerceptionSystem, FeatureConfig
        )
        from neuro_symbolic.symbolic_reasoning import SymbolicReasoner
        from neuro_symbolic.neural_symbolic_bridge import (
            NeuralSymbolicBridge, BridgeConfig
        )
        from neuro_symbolic.explainable_ai import ExplainableAI
        
        # 神经感知
        feature_config = FeatureConfig(
            input_dim=10,
            hidden_dims=[64, 128, 64],
            output_dim=32
        )
        self.neural_perception = NeuralPerceptionSystem(feature_config)
        
        # 符号推理
        self.symbolic_reasoner = SymbolicReasoner()
        self._populate_knowledge_base()
        
        # 神经-符号桥接
        bridge_config = BridgeConfig(
            neural_dim=32,
            symbol_dim=16,
            num_symbols=50
        )
        self.bridge = NeuralSymbolicBridge(bridge_config)
        
        # 可解释AI
        self.xai = None  # 需要模型后才能初始化
    
    def _populate_knowledge_base(self):
        """填充知识库"""
        from neuro_symbolic.symbolic_reasoning import Literal, Term, TermType, Rule
        
        # 添加催化反应规则
        x = Term("X", TermType.VARIABLE)
        y = Term("Y", TermType.VARIABLE)
        
        # 活性位点 -> 催化活性
        rule1 = Rule(
            head=Literal("has_catalytic_activity", [x]),
            body=[Literal("has_active_site", [x])]
        )
        self.symbolic_reasoner.add_rule(rule1)
        
        # 小颗粒 -> 高比表面积
        rule2 = Rule(
            head=Literal("has_high_surface_area", [x]),
            body=[Literal("has_small_particles", [x])]
        )
        self.symbolic_reasoner.add_rule(rule2)
        
        # 高比表面积 -> 更多活性位点
        rule3 = Rule(
            head=Literal("has_many_active_sites", [x]),
            body=[Literal("has_high_surface_area", [x])]
        )
        self.symbolic_reasoner.add_rule(rule3)
        
        # 添加具体实例
        catalysts = ["Pt_Al2O3", "Pd_C", "Ni_SiO2", "Cu_ZnO", "Au_TiO2"]
        for cat in catalysts:
            cat_term = Term(cat, TermType.CONSTANT)
            self.symbolic_reasoner.add_fact(Literal("is_catalyst", [cat_term]))
    
    def analyze_catalyst(
        self,
        catalyst_features: Dict[str, float],
        reaction_conditions: Dict[str, float],
        activity_measurement: float
    ) -> Dict[str, Any]:
        """
        分析催化剂
        
        Args:
            catalyst_features: 催化剂特征（如粒径、比表面积等）
            reaction_conditions: 反应条件
            activity_measurement: 活性测量值
        
        Returns:
            机理解释
        """
        results = {}
        
        # 1. 神经感知分析
        neural_analysis = self._neural_analysis(catalyst_features)
        results['neural_analysis'] = neural_analysis
        
        # 2. 符号推理分析
        symbolic_analysis = self._symbolic_analysis(catalyst_features)
        results['symbolic_analysis'] = symbolic_analysis
        
        # 3. 神经-符号融合解释
        fused_explanation = self._fuse_explanations(
            neural_analysis, symbolic_analysis, catalyst_features
        )
        results['fused_explanation'] = fused_explanation
        
        # 4. 生成机理解释
        mechanism = self._generate_mechanism_explanation(
            catalyst_features, activity_measurement
        )
        results['mechanism'] = mechanism
        
        return results
    
    def _neural_analysis(self, features: Dict[str, float]) -> Dict[str, Any]:
        """神经感知分析"""
        # 特征向量化
        feature_vector = np.array([
            features.get('particle_size_nm', 5),
            features.get('surface_area_m2g', 100),
            features.get('pore_volume_cm3g', 0.5),
            features.get('metal_loading_wt', 1.0),
            features.get('reduction_temperature_C', 300),
            features.get('acidity_mmolg', 0.1),
            features.get('basicity_mmolg', 0.05),
            features.get('crystallinity_percent', 50),
            features.get('defect_density_percent', 5),
            features.get('support_interaction_index', 0.5)
        ]).reshape(1, -1)
        
        # 提取特征
        extracted_features = self.neural_perception.extract_features(
            feature_vector, data_type="tabular"
        )
        
        # 检测模式
        patterns = self.neural_perception.detect_patterns(feature_vector)
        
        return {
            'extracted_features': extracted_features.detach().cpu().numpy() if hasattr(extracted_features, 'detach') else extracted_features,
            'patterns': patterns
        }
    
    def _symbolic_analysis(self, features: Dict[str, float]) -> Dict[str, Any]:
        """符号推理分析"""
        from neuro_symbolic.symbolic_reasoning import Literal, Term, TermType
        
        activated_rules = []
        inferences = []
        
        # 根据特征激活规则
        if features.get('particle_size_nm', 10) < 5:
            cat_term = Term("catalyst", TermType.CONSTANT)
            fact = Literal("has_small_particles", [cat_term])
            self.symbolic_reasoner.add_fact(fact)
            activated_rules.append("small_particles_rule")
            inferences.append("催化剂具有小颗粒，可能有高比表面积")
        
        if features.get('surface_area_m2g', 50) > 100:
            inferences.append("高比表面积提供更多活性位点")
        
        if features.get('acidity_mmolg', 0) > 0.2:
            inferences.append("较强的酸性位点适合酸催化反应")
        
        # 查询知识库
        query_results = self.symbolic_reasoner.query("has_catalytic_activity(X)")
        
        return {
            'activated_rules': activated_rules,
            'inferences': inferences,
            'query_results': query_results
        }
    
    def _fuse_explanations(
        self,
        neural_analysis: Dict,
        symbolic_analysis: Dict,
        features: Dict[str, float]
    ) -> Dict[str, Any]:
        """融合神经和符号解释"""
        # 神经层面的重要特征
        neural_importance = self.neural_perception.get_feature_importance(
            np.array([list(features.values())])
        )
        
        # 符号层面的推理链
        symbolic_chains = symbolic_analysis['inferences']
        
        # 对齐神经和符号概念
        fused_concepts = []
        
        feature_names = list(features.keys())
        top_neural_features = sorted(
            enumerate(neural_importance),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        for idx, importance in top_neural_features:
            feature_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            
            # 找到对应的符号概念
            for chain in symbolic_chains:
                if any(keyword in chain.lower() for keyword in feature_name.lower().split('_')):
                    fused_concepts.append({
                        'feature': feature_name,
                        'neural_importance': float(importance),
                        'symbolic_explanation': chain,
                        'confidence': 'high' if importance > 0.5 else 'medium'
                    })
                    break
        
        return {
            'fused_concepts': fused_concepts,
            'neural_importance': neural_importance.tolist(),
            'symbolic_chains': symbolic_chains
        }
    
    def _generate_mechanism_explanation(
        self,
        features: Dict[str, float],
        activity: float
    ) -> Dict[str, str]:
        """生成机理解释"""
        explanation = {
            'overview': '',
            'active_sites': '',
            'structure_activity': '',
            'recommendations': []
        }
        
        # 概述
        particle_size = features.get('particle_size_nm', 5)
        surface_area = features.get('surface_area_m2g', 100)
        
        explanation['overview'] = (
            f"该催化剂具有{particle_size:.1f}nm的颗粒大小和"
            f"{surface_area:.1f}m²/g的比表面积，"
            f"测得活性为{activity:.2f}。"
        )
        
        # 活性位点分析
        if particle_size < 3:
            explanation['active_sites'] = (
                "极小的颗粒尺寸（<3nm）产生了大量低配位活性位点，"
                "这些位点通常具有更高的反应活性。"
            )
        elif particle_size < 10:
            explanation['active_sites'] = (
                "适中的颗粒尺寸提供了良好的活性位点密度"
                "和稳定性平衡。"
            )
        else:
            explanation['active_sites'] = (
                "较大的颗粒尺寸限制了活性位点的数量，"
                "可能需要进一步优化。"
            )
        
        # 构效关系
        sar_parts = []
        if surface_area > 200:
            sar_parts.append("高比表面积提供了丰富的反应界面")
        if features.get('metal_loading_wt', 0) > 2:
            sar_parts.append("较高的金属负载量增加了活性组分")
        if features.get('defect_density_percent', 0) > 3:
            sar_parts.append("适当的缺陷密度创造了额外的活性位点")
        
        explanation['structure_activity'] = "；".join(sar_parts) + "。"
        
        # 优化建议
        if activity < 0.5:
            explanation['recommendations'].append("考虑进一步降低颗粒尺寸以提高活性")
        if surface_area < 100:
            explanation['recommendations'].append("优化合成条件以提高比表面积")
        if features.get('metal_loading_wt', 0) < 1:
            explanation['recommendations'].append("适当增加金属负载量")
        
        return explanation
    
    def compare_catalysts(
        self,
        catalysts_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """比较多个催化剂"""
        comparison = {
            'individual_analyses': [],
            'comparative_insights': [],
            'ranking': []
        }
        
        for catalyst_data in catalysts_data:
            analysis = self.analyze_catalyst(
                catalyst_data['features'],
                catalyst_data.get('conditions', {}),
                catalyst_data['activity']
            )
            comparison['individual_analyses'].append({
                'name': catalyst_data['name'],
                'analysis': analysis
            })
        
        # 比较分析
        activities = [(c['name'], c['activity']) for c in catalysts_data]
        ranking = sorted(activities, key=lambda x: x[1], reverse=True)
        comparison['ranking'] = ranking
        
        # 生成比较洞察
        best = ranking[0]
        worst = ranking[-1]
        comparison['comparative_insights'].append(
            f"{best[0]}展现出最高的催化活性({best[1]:.2f})，"
            f"比{worst[0]}高出{(best[1]-worst[1])/worst[1]*100:.1f}%"
        )
        
        return comparison
    
    def generate_report(
        self,
        catalyst_name: str,
        analysis_results: Dict[str, Any]
    ) -> str:
        """生成完整的解释报告"""
        report = []
        report.append("=" * 70)
        report.append(f"催化剂机理解释报告: {catalyst_name}")
        report.append("=" * 70)
        
        # 概述
        report.append("\n【概述】")
        report.append(analysis_results['mechanism']['overview'])
        
        # 活性位点分析
        report.append("\n【活性位点分析】")
        report.append(analysis_results['mechanism']['active_sites'])
        
        # 构效关系
        report.append("\n【结构-活性关系】")
        report.append(analysis_results['mechanism']['structure_activity'])
        
        # 融合解释
        report.append("\n【融合解释（神经+符号）】")
        for concept in analysis_results['fused_explanation']['fused_concepts']:
            report.append(f"- {concept['feature']}: {concept['symbolic_explanation']}")
            report.append(f"  神经重要性: {concept['neural_importance']:.3f}")
        
        # 优化建议
        report.append("\n【优化建议】")
        for rec in analysis_results['mechanism']['recommendations']:
            report.append(f"- {rec}")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)


def generate_sample_catalyst_data() -> List[Dict[str, Any]]:
    """生成示例催化剂数据"""
    catalysts = [
        {
            'name': 'Pt/Al2O3_A',
            'features': {
                'particle_size_nm': 2.5,
                'surface_area_m2g': 250,
                'pore_volume_cm3g': 0.8,
                'metal_loading_wt': 1.5,
                'reduction_temperature_C': 250,
                'acidity_mmolg': 0.15,
                'basicity_mmolg': 0.03,
                'crystallinity_percent': 60,
                'defect_density_percent': 8,
                'support_interaction_index': 0.7
            },
            'conditions': {'temperature': 200, 'pressure': 1.0},
            'activity': 0.85
        },
        {
            'name': 'Pd/C_B',
            'features': {
                'particle_size_nm': 4.0,
                'surface_area_m2g': 800,
                'pore_volume_cm3g': 1.2,
                'metal_loading_wt': 5.0,
                'reduction_temperature_C': 200,
                'acidity_mmolg': 0.05,
                'basicity_mmolg': 0.1,
                'crystallinity_percent': 40,
                'defect_density_percent': 5,
                'support_interaction_index': 0.3
            },
            'conditions': {'temperature': 180, 'pressure': 1.0},
            'activity': 0.72
        },
        {
            'name': 'Ni/SiO2_C',
            'features': {
                'particle_size_nm': 8.0,
                'surface_area_m2g': 120,
                'pore_volume_cm3g': 0.6,
                'metal_loading_wt': 10.0,
                'reduction_temperature_C': 400,
                'acidity_mmolg': 0.2,
                'basicity_mmolg': 0.08,
                'crystallinity_percent': 75,
                'defect_density_percent': 3,
                'support_interaction_index': 0.5
            },
            'conditions': {'temperature': 250, 'pressure': 2.0},
            'activity': 0.58
        }
    ]
    return catalysts


def run_catalyst_mechanism_demo():
    """运行催化剂机理解释演示"""
    print("=" * 70)
    print("催化剂机理解释演示 - 神经符号融合")
    print("=" * 70)
    
    # 创建解释器
    print("\n1. 初始化解释器...")
    explainer = CatalystMechanismExplainer()
    explainer.initialize_modules()
    print("   所有模块已初始化")
    
    # 生成示例数据
    print("\n2. 加载示例催化剂数据...")
    catalysts = generate_sample_catalyst_data()
    print(f"   加载了 {len(catalysts)} 个催化剂样本")
    
    # 分析每个催化剂
    print("\n3. 分析每个催化剂...")
    for catalyst in catalysts:
        print(f"\n   分析 {catalyst['name']}...")
        
        results = explainer.analyze_catalyst(
            catalyst['features'],
            catalyst['conditions'],
            catalyst['activity']
        )
        
        # 打印简要结果
        print(f"   - 检测到 {len(results['fused_explanation']['fused_concepts'])} 个融合概念")
        print(f"   - 符号推理生成 {len(results['symbolic_analysis']['inferences'])} 条推理链")
        
        # 生成报告
        report = explainer.generate_report(catalyst['name'], results)
        print("\n" + report)
        print("\n" + "-" * 50)
    
    # 比较分析
    print("\n4. 催化剂比较分析...")
    comparison = explainer.compare_catalysts(catalysts)
    
    print("\n   活性排名:")
    for i, (name, activity) in enumerate(comparison['ranking'], 1):
        print(f"     {i}. {name}: {activity:.2f}")
    
    print("\n   比较洞察:")
    for insight in comparison['comparative_insights']:
        print(f"     - {insight}")
    
    print("\n" + "=" * 70)
    print("演示完成!")
    print("=" * 70)


def demo():
    """演示函数"""
    return run_catalyst_mechanism_demo()


if __name__ == "__main__":
    demo()
