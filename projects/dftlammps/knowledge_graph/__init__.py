"""
知识图谱模块初始化文件
=====================

此模块提供材料科学知识图谱功能，包括：
- 实体抽取 (kg_core.py)
- 知识图谱构建与管理 (kg_core.py)
- 本体初始化 (kg_init.py)
- 推理与查询 (kg_core.py)

Author: DFT-LAMMPS Team
Date: 2025
"""

from .kg_core import (
    KnowledgeGraph,
    Entity,
    EntityType,
    Relation,
    RelationType,
    EntityExtractor,
    LiteratureMiningPipeline,
    create_knowledge_graph,
    extract_from_literature
)

from .kg_init import (
    initialize_materials_ontology,
    create_default_knowledge_graph
)

__all__ = [
    # Core classes
    'KnowledgeGraph',
    'Entity',
    'EntityType',
    'Relation',
    'RelationType',
    'EntityExtractor',
    'LiteratureMiningPipeline',
    
    # Initialization
    'initialize_materials_ontology',
    'create_default_knowledge_graph',
    
    # Convenience functions
    'create_knowledge_graph',
    'extract_from_literature'
]

__version__ = "1.0.0"
