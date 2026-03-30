"""Tests for ML potential module."""

import pytest
import numpy as np
from dataclasses import asdict

from dftlammps.core import NEPDataConfig, NEPModelConfig


class TestNEPDataConfig:
    """Test NEPDataConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = NEPDataConfig()
        assert config.energy_threshold == 50.0
        assert config.force_threshold == 50.0
        assert config.min_force == 0.001
        assert config.train_ratio == 0.9
        assert config.test_ratio == 0.1
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = NEPDataConfig(
            energy_threshold=100.0,
            train_ratio=0.8,
            test_ratio=0.2
        )
        assert config.energy_threshold == 100.0
        assert config.train_ratio == 0.8
        assert config.test_ratio == 0.2
    
    def test_type_map(self):
        """Test type map configuration."""
        type_map = ["Li", "S", "P"]
        config = NEPDataConfig(type_map=type_map)
        assert config.type_map == type_map


class TestNEPModelConfig:
    """Test NEPModelConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = NEPModelConfig()
        assert config.model_type == 0
        assert config.version == 4
        assert config.cutoff_radial == 6.0
        assert config.cutoff_angular == 4.0
        assert config.n_max_radial == 4
        assert config.n_max_angular == 4
        assert config.neuron == 50
        assert config.learning_rate == 0.001
    
    def test_model_types(self):
        """Test different model types."""
        for model_type in [0, 1, 2]:
            config = NEPModelConfig(model_type=model_type)
            assert config.model_type == model_type
    
    def test_nep_versions(self):
        """Test different NEP versions."""
        for version in [2, 3, 4]:
            config = NEPModelConfig(version=version)
            assert config.version == version
