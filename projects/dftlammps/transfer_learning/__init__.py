"""
DFT-LAMMPS 跨领域迁移与元材料发现模块

该模块实现了知识在不同材料领域间的迁移，加速新领域材料发现

模块结构:
- transfer_learning/: 迁移学习核心算法
- cross_domain/: 跨领域应用
- universal_descriptors/: 通用描述符
- transfer_examples/: 应用示例

作者: DFT-LAMMPS Team
版本: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "DFT-LAMMPS Team"

# Transfer Learning
from .domain_adapter import (
    DomainAdapter,
    DANNAdapter,
    MMDAdapter,
    CORALAdapter,
    DomainAdaptationConfig
)

from .meta_material_learning import (
    CrossDomainMetaLearner,
    MaterialTaskSampler,
    MetaLearningConfig
)

from .knowledge_transfer import (
    KnowledgeTransferEngine,
    KnowledgeTransferConfig
)

from .domain_similarity import (
    DomainSimilarityAnalyzer,
    SimilarityConfig,
    DomainSimilarityMetrics
)

# Cross Domain
from .battery_to_catalyst import BatteryToCatalystTransfer
from .semiconductor_to_photovoltaic import SemiconductorToPVTransfer
from .metal_to_ceramic import MetalToCeramicTransfer
from .high_entropy_transfer import HighEntropyMaterialFramework

# Universal Descriptors
from .matminer_integration import MatminerFeatureExtractor
from .megnet_descriptor import MEGNetDescriptor
from .cgcnn_features import CGCNNDescriptor
from .universal_fingerprint import UniversalFingerprintGenerator

__all__ = [
    # Transfer Learning
    "DomainAdapter",
    "DANNAdapter",
    "MMDAdapter",
    "CORALAdapter",
    "DomainAdaptationConfig",
    "CrossDomainMetaLearner",
    "MaterialTaskSampler",
    "MetaLearningConfig",
    "KnowledgeTransferEngine",
    "KnowledgeTransferConfig",
    "DomainSimilarityAnalyzer",
    "SimilarityConfig",
    "DomainSimilarityMetrics",
    
    # Cross Domain
    "BatteryToCatalystTransfer",
    "SemiconductorToPVTransfer",
    "MetalToCeramicTransfer",
    "HighEntropyMaterialFramework",
    
    # Universal Descriptors
    "MatminerFeatureExtractor",
    "MEGNetDescriptor",
    "CGCNNDescriptor",
    "UniversalFingerprintGenerator",
]
