"""
DFT-LAMMPS 通用描述符模块

实现跨领域材料描述的通用描述符
"""

from .matminer_integration import MatminerFeatureExtractor, MatminerConfig
from .megnet_descriptor import MEGNetDescriptor, MEGNetConfig
from .cgcnn_features import CGCNNDescriptor, CGCNNConfig
from .universal_fingerprint import UniversalFingerprintGenerator, UniversalFingerprintConfig

__all__ = [
    "MatminerFeatureExtractor",
    "MatminerConfig",
    "MEGNetDescriptor",
    "MEGNetConfig",
    "CGCNNDescriptor",
    "CGCNNConfig",
    "UniversalFingerprintGenerator",
    "UniversalFingerprintConfig",
]
