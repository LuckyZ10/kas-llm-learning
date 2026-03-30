"""
Data Module
===========

Dataset and preprocessing utilities:

1. **CrystalDataset** - Base dataset for crystal structures
2. **MPDataset** - Materials Project dataset loader
3. **SyntheticCrystalDataset** - Synthetic data generator
4. **CrystalPreprocessor** - Data preprocessing
5. **WyckoffPositionEncoder** - Wyckoff position encoding
6. **SpaceGroupAnalyzer** - Symmetry analysis
"""

from .crystal_dataset import (
    CrystalDataset,
    MPDataset,
    SyntheticCrystalDataset,
    collate_crystal_batch
)

from .preprocessing import (
    CrystalPreprocessor,
    WyckoffPositionEncoder,
    SpaceGroupAnalyzer,
    structure_to_tensors,
    tensors_to_structure
)

__all__ = [
    "CrystalDataset",
    "MPDataset",
    "SyntheticCrystalDataset",
    "collate_crystal_batch",
    "CrystalPreprocessor",
    "WyckoffPositionEncoder",
    "SpaceGroupAnalyzer",
    "structure_to_tensors",
    "tensors_to_structure",
]
