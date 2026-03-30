"""
Pretrained Models Module
========================

Download and use pretrained generative models.

Example:
    >>> from dftlammps.generative_ai.pretrained import load_model
    >>> model = load_model("crystal_dit_base")
    >>> structures = model.generate(batch_size=10, num_atoms=20)
"""

from .model_hub import (
    PretrainedModelHub,
    download_model,
    load_model,
    MODEL_REGISTRY
)

__all__ = [
    "PretrainedModelHub",
    "download_model",
    "load_model",
    "MODEL_REGISTRY",
]
