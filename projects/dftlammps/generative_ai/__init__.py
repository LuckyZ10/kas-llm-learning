"""
DFT-LAMMPS Generative AI Module
================================

A comprehensive module for generative modeling of materials using:
- Diffusion Transformers (DiT/ADiT)
- Riemannian Flow Matching
- Consistency Models
- Conditional Generation for Inverse Design

Author: DFT-LAMMPS Integration Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "DFT-LAMMPS Integration Team"

# Core model classes
from .models.crystal_dit import CrystalDiT, ADiT
from .models.flow_matching import RiemannianFlowMatcher, CrystalFlow
from .models.consistency import ConsistencyCrystalModel
from .models.conditional import ConditionalDiffusion
from .models.joint_generator import JointMolecularCrystalGenerator

# Training
from .training.diffusion_trainer import DiffusionTrainer
from .training.flow_trainer import FlowMatchingTrainer
from .training.consistency_trainer import ConsistencyTrainer

# Data
from .data.crystal_dataset import CrystalDataset, MPDataset
from .data.preprocessing import CrystalPreprocessor

# Utils
from .utils.sampling import DiffusionSampler, FlowSampler
from .utils.evaluation import CrystalMetrics
from .utils.symmetry import WyckoffPositionEncoder, SpaceGroupConstraint

# Integration
from .integration.screening_integration import GenerativeScreening
from .integration.inverse_design import InverseDesignPipeline

# Pretrained models
from .pretrained.model_hub import PretrainedModelHub

__all__ = [
    # Models
    "CrystalDiT",
    "ADiT",
    "RiemannianFlowMatcher",
    "CrystalFlow",
    "ConsistencyCrystalModel",
    "ConditionalDiffusion",
    "JointMolecularCrystalGenerator",
    # Training
    "DiffusionTrainer",
    "FlowMatchingTrainer",
    "ConsistencyTrainer",
    # Data
    "CrystalDataset",
    "MPDataset",
    "CrystalPreprocessor",
    # Utils
    "DiffusionSampler",
    "FlowSampler",
    "CrystalMetrics",
    "WyckoffPositionEncoder",
    "SpaceGroupConstraint",
    # Integration
    "GenerativeScreening",
    "InverseDesignPipeline",
    # Pretrained
    "PretrainedModelHub",
]


def get_model(model_name: str, **kwargs):
    """
    Factory function to get a model by name.
    
    Args:
        model_name: Name of the model
        **kwargs: Model-specific arguments
        
    Returns:
        Model instance
    """
    models = {
        "crystal_dit": CrystalDiT,
        "adit": ADiT,
        "flow_matching": RiemannianFlowMatcher,
        "crystal_flow": CrystalFlow,
        "consistency": ConsistencyCrystalModel,
        "conditional": ConditionalDiffusion,
        "joint": JointMolecularCrystalGenerator,
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name.lower()](**kwargs)


def get_trainer(trainer_name: str, **kwargs):
    """
    Factory function to get a trainer by name.
    
    Args:
        trainer_name: Name of the trainer
        **kwargs: Trainer-specific arguments
        
    Returns:
        Trainer instance
    """
    trainers = {
        "diffusion": DiffusionTrainer,
        "flow_matching": FlowMatchingTrainer,
        "consistency": ConsistencyTrainer,
    }
    
    if trainer_name.lower() not in trainers:
        raise ValueError(f"Unknown trainer: {trainer_name}. Available: {list(trainers.keys())}")
    
    return trainers[trainer_name.lower()](**kwargs)
