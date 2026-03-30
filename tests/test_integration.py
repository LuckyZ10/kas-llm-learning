"""Integration tests for dftlammps package."""

import pytest
import numpy as np
from ase import Atoms

from dftlammps.core import (
    MDConfig,
    MDSimulationRunner,
    VASPParserConfig,
    VASPDataExtractor,
)
from dftlammps.hpc import ResourceRequest, JobSpec


@pytest.mark.integration
class TestCoreIntegration:
    """Integration tests for core modules."""
    
    def test_config_workflow(self):
        """Test configuration workflow."""
        # Create MD config
        md_config = MDConfig(
            ensemble="nvt",
            temperature=500.0,
            nsteps=1000
        )
        
        # Create runner
        runner = MDSimulationRunner(md_config)
        
        # Verify configuration chain
        assert runner.config.ensemble == "nvt"
        assert runner.config.temperature == 500.0
        assert runner.config.nsteps == 1000


@pytest.mark.integration
class TestHPCIntegration:
    """Integration tests for HPC modules."""
    
    def test_resource_to_job_workflow(self):
        """Test resource request to job spec workflow."""
        # Create resource request
        resources = ResourceRequest(
            num_nodes=4,
            num_cores_per_node=64,
            walltime_hours=72.0,
            partition="large"
        )
        
        # Create job spec with resources
        job = JobSpec(
            name="large_scale_md",
            executable="lmp",
            arguments=["-in", "in.lammps"],
            resources=resources
        )
        
        # Verify integration
        assert job.resources.num_nodes == 4
        assert job.resources.num_cores_per_node == 64
        assert "large" in job.resources.to_slurm()


@pytest.mark.integration
class TestDataFlow:
    """Integration tests for data flow between modules."""
    
    def test_atoms_structure_handling(self):
        """Test handling of ASE Atoms structures."""
        # Create structure
        atoms = Atoms(
            symbols=['Li', 'S', 'P'],
            positions=[[0, 0, 0], [2, 0, 0], [1, 1, 1]],
            cell=[5, 5, 5],
            pbc=True
        )
        
        # Verify structure properties
        assert len(atoms) == 3
        assert atoms.get_chemical_formula() == "LiPS"
        
        # Test positions array
        positions = atoms.get_positions()
        assert positions.shape == (3, 3)
