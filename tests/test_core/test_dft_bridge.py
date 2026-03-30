"""Tests for DFT bridge module."""

import pytest
import numpy as np
from dataclasses import asdict

from dftlammps.core import (
    VASPParserConfig,
    ForceFieldConfig,
    LAMMPSInputConfig,
    VASPDataExtractor,
)


class TestVASPParserConfig:
    """Test VASPParserConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = VASPParserConfig()
        assert config.extract_energy is True
        assert config.extract_forces is True
        assert config.extract_stress is True
        assert config.extract_positions is True
        assert config.extract_velocities is False
        assert config.extract_magmom is False
        assert config.filter_unconverged is True
        assert config.energy_threshold == 100.0
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = VASPParserConfig(
            extract_energy=False,
            extract_velocities=True,
            energy_threshold=50.0
        )
        assert config.extract_energy is False
        assert config.extract_velocities is True
        assert config.energy_threshold == 50.0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = VASPParserConfig()
        config_dict = asdict(config)
        assert 'extract_energy' in config_dict
        assert 'extract_forces' in config_dict


class TestForceFieldConfig:
    """Test ForceFieldConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ForceFieldConfig()
        assert config.ff_type == "buckingham"
        assert config.cutoff == 6.0
        assert config.fit_method == "least_squares"
        assert config.regularization == 0.01
        assert config.max_iterations == 1000
    
    def test_custom_ff_type(self):
        """Test different force field types."""
        for ff_type in ["buckingham", "morse", "lj", "eam"]:
            config = ForceFieldConfig(ff_type=ff_type)
            assert config.ff_type == ff_type
    
    def test_elements_list(self):
        """Test elements list configuration."""
        elements = ["Li", "S", "P"]
        config = ForceFieldConfig(elements=elements)
        assert config.elements == elements


class TestLAMMPSInputConfig:
    """Test LAMMPSInputConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = LAMMPSInputConfig()
        assert config.units == "metal"
        assert config.atom_style == "atomic"
        assert config.boundary == "p p p"
        assert config.timestep == 1.0
        assert config.temperature == 300.0
        assert config.ensemble == "nvt"
    
    def test_ensemble_types(self):
        """Test different ensemble configurations."""
        for ensemble in ["nve", "nvt", "npt"]:
            config = LAMMPSInputConfig(ensemble=ensemble)
            assert config.ensemble == ensemble


class TestVASPDataExtractor:
    """Test VASPDataExtractor class."""
    
    def test_initialization(self):
        """Test extractor initialization."""
        config = VASPParserConfig()
        extractor = VASPDataExtractor(config=config)
        assert extractor.config == config
    
    def test_default_initialization(self):
        """Test extractor with default config."""
        extractor = VASPDataExtractor()
        assert extractor.config is not None
        assert isinstance(extractor.config, VASPParserConfig)
