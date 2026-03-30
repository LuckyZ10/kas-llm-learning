"""
知识推理模块 - Knowledge Reasoning for Materials Science

提供基于本体的推理、规则引擎和案例推理功能，
为材料科学提供强大的知识表示和推理能力。

主要组件:
    - Ontology Reasoning: 描述逻辑推理与概念层次
    - Rule Engine: 产生式规则系统
    - Case-Based Reasoning: 案例推理与学习

作者: DFT-LAMMPS Team
"""

from .ontology_reasoning import (
    ConceptType, RoleType,
    Concept, Role, Individual,
    ConceptHierarchy,
    DescriptionLogicReasoner,
    NeuralOntologyReasoner,
    create_material_ontology,
    perform_ontology_reasoning_example,
)

from .rule_engine import (
    CertaintyFactor, Fact, Condition, Rule,
    ConflictResolutionStrategy, RuleEngine,
    NeuralRuleLearner,
    create_material_rules,
)

from .case_based_reasoning import (
    CaseStatus, MaterialCase,
    SimilarityMetric, CaseRetriever,
    NeuralCaseEncoder, CaseAdapter, CaseBasedReasoner,
    create_sample_material_cases,
)

__version__ = "1.0.0"
__all__ = [
    # Ontology Reasoning
    "ConceptType", "RoleType", "Concept", "Role", "Individual",
    "ConceptHierarchy", "DescriptionLogicReasoner", "NeuralOntologyReasoner",
    "create_material_ontology", "perform_ontology_reasoning_example",
    
    # Rule Engine
    "CertaintyFactor", "Fact", "Condition", "Rule",
    "ConflictResolutionStrategy", "RuleEngine", "NeuralRuleLearner",
    "create_material_rules",
    
    # Case-Based Reasoning
    "CaseStatus", "MaterialCase", "SimilarityMetric", "CaseRetriever",
    "NeuralCaseEncoder", "CaseAdapter", "CaseBasedReasoner",
    "create_sample_material_cases",
]
