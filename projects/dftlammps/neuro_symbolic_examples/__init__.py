"""
神经符号融合案例示例 - Neuro-Symbolic Examples

实际应用案例展示:
- battery_causal_discovery: 电池性能因果图自动发现
- catalyst_mechanism_explanation: 催化剂机理解释
- material_property_predictor: 材料性质预测器
"""

from .battery_causal_discovery import (
    generate_battery_data,
    discover_battery_causal_graph,
    generate_insights,
    recommend_interventions,
    run_battery_causal_discovery
)

from .catalyst_mechanism_explanation import (
    CatalystMechanismExplainer,
    generate_sample_catalyst_data,
    run_catalyst_mechanism_demo
)

from .material_property_predictor import (
    MaterialPropertyPredictor,
    Material,
    PredictionExplanation,
    generate_sample_materials,
    run_material_property_demo
)

__all__ = [
    # Battery Causal Discovery
    'generate_battery_data',
    'discover_battery_causal_graph',
    'generate_insights',
    'recommend_interventions',
    'run_battery_causal_discovery',
    
    # Catalyst Mechanism
    'CatalystMechanismExplainer',
    'generate_sample_catalyst_data',
    'run_catalyst_mechanism_demo',
    
    # Material Property Predictor
    'MaterialPropertyPredictor',
    'Material',
    'PredictionExplanation',
    'generate_sample_materials',
    'run_material_property_demo',
]
