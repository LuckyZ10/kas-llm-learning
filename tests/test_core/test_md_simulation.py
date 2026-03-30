"""Tests for MD simulation module."""

import pytest
import numpy as np

from dftlammps.core import MDConfig, MDSimulationRunner, MDTrajectoryAnalyzer, MDTrajectory


class TestMDConfig:
    """Test MDConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = MDConfig()
        assert config.ensemble == "nvt"
        assert config.temperature == 300.0
        assert config.timestep == 1.0
        assert config.nsteps == 100000
        assert config.nsteps_equil == 10000
        assert config.pair_style == "deepmd"
        assert config.nprocs == 4
    
    def test_ensemble_types(self):
        """Test different ensemble configurations."""
        for ensemble in ["nve", "nvt", "npt"]:
            config = MDConfig(ensemble=ensemble)
            assert config.ensemble == ensemble
    
    def test_temperature_range(self):
        """Test temperature configuration."""
        for temp in [100.0, 300.0, 500.0, 1000.0]:
            config = MDConfig(temperature=temp)
            assert config.temperature == temp


class TestMDTrajectory:
    """Test MDTrajectory dataclass."""
    
    def test_default_initialization(self):
        """Test default trajectory initialization."""
        traj = MDTrajectory()
        assert traj.positions == []
        assert traj.velocities == []
        assert traj.forces == []
        assert traj.energies == {}
        assert traj.temperatures == []
        assert traj.pressures == []
        assert traj.time == []
    
    def test_to_dict(self):
        """Test trajectory to dict conversion."""
        traj = MDTrajectory()
        traj.temperatures = [300.0, 301.0, 299.0]
        traj.time = [0.0, 1.0, 2.0]
        
        d = traj.to_dict()
        assert 'temperatures' in d
        assert 'time' in d
        assert d['temperatures'] == [300.0, 301.0, 299.0]


class TestMDSimulationRunner:
    """Test MDSimulationRunner class."""
    
    def test_initialization(self):
        """Test runner initialization."""
        config = MDConfig()
        runner = MDSimulationRunner(config)
        assert runner.config == config
    
    def test_default_initialization(self):
        """Test runner with default config."""
        runner = MDSimulationRunner()
        assert runner.config is not None
        assert isinstance(runner.config, MDConfig)


class TestMDTrajectoryAnalyzer:
    """Test MDTrajectoryAnalyzer class."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        traj = MDTrajectory()
        analyzer = MDTrajectoryAnalyzer(traj)
        assert analyzer.trajectory == traj
    
    def test_empty_trajectory_summary(self):
        """Test summary with empty trajectory."""
        traj = MDTrajectory()
        analyzer = MDTrajectoryAnalyzer(traj)
        summary = analyzer.get_summary()
        assert summary['n_frames'] == 0
        assert summary['simulation_time_ps'] == 0
    
    def test_summary_with_data(self):
        """Test summary with trajectory data."""
        traj = MDTrajectory()
        traj.positions = [np.random.randn(10, 3) for _ in range(5)]
        traj.time = [0.0, 1.0, 2.0, 3.0, 4.0]
        traj.temperatures = [300.0, 301.0, 299.0, 300.5, 298.5]
        
        analyzer = MDTrajectoryAnalyzer(traj)
        summary = analyzer.get_summary()
        
        assert summary['n_frames'] == 5
        assert summary['simulation_time_ps'] == 4.0
        assert abs(summary['average_temperature'] - 299.8) < 1.0
