"""
Pytest Configuration and Shared Fixtures
========================================

This module provides shared fixtures for all test modules in the test suite.

Usage:
    Tests automatically have access to these fixtures without explicit import.
    
Example:
    def test_example(mock_atoms, tmp_path):
        atoms = mock_atoms
        output_dir = tmp_path / "output"
        # test code...
"""

import os
import sys
import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import modules under test
from core.dft.bridge import (
    VASPOUTCARParser, ForceFieldFitter, LAMMPSInputGenerator,
    ForceFieldConfig, LAMMPSInputConfig, VASPParserConfig
)
from core.ml.nep.pipeline import (
    NEPDataPreparer, NEPInputGenerator, NEPTrainer,
    NEPDataConfig, NEPModelConfig, NEPTrainingConfig
)
from platform.hpc.scheduler import (
    SlurmScheduler, PBSScheduler, LSFScheduler, LocalScheduler,
    JobSpec, ResourceRequest, JobInfo, JobStatus, SchedulerType
)


# =============================================================================
# Path Fixtures
# =============================================================================

@pytest.fixture
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def test_data_dir():
    """Return the test data directory (creates if needed)."""
    data_dir = Path(__file__).parent / "fixtures" / "test_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture
def mock_working_dir(tmp_path):
    """Create a temporary working directory for tests."""
    work_dir = tmp_path / "test_work"
    work_dir.mkdir()
    return work_dir


# =============================================================================
# ASE Atoms Fixtures
# =============================================================================

@pytest.fixture
def mock_atoms():
    """Create a simple ASE Atoms object for testing."""
    try:
        from ase import Atoms
        atoms = Atoms(
            symbols=['Li', 'Li', 'S', 'S', 'P', 'P'],
            positions=[
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [3.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [2.0, 1.0, 1.0]
            ],
            cell=[4.0, 4.0, 4.0],
            pbc=True
        )
        # Mock calculator results
        atoms.info['energy'] = -10.5
        return atoms
    except ImportError:
        pytest.skip("ASE not available")


@pytest.fixture
def mock_atoms_bulk():
    """Create a bulk crystal structure for testing."""
    try:
        from ase import Atoms
        from ase.build import bulk
        atoms = bulk('NaCl', 'rocksalt', a=5.64)
        return atoms
    except ImportError:
        # Fallback simple bulk
        from ase import Atoms
        atoms = Atoms(
            symbols=['Na', 'Cl', 'Na', 'Cl'],
            positions=[
                [0.0, 0.0, 0.0],
                [2.82, 0.0, 0.0],
                [0.0, 2.82, 0.0],
                [2.82, 2.82, 0.0]
            ],
            cell=[5.64, 5.64, 5.64],
            pbc=True
        )
        return atoms


@pytest.fixture
def mock_atoms_surface():
    """Create a surface structure for testing."""
    try:
        from ase.build import fcc100
        atoms = fcc100('Al', size=(2, 2, 4), vacuum=10.0)
        return atoms
    except ImportError:
        from ase import Atoms
        atoms = Atoms(
            symbols=['Al'] * 8,
            positions=np.random.rand(8, 3) * [8.0, 8.0, 10.0],
            cell=[8.0, 8.0, 20.0],
            pbc=[True, True, False]
        )
        return atoms


@pytest.fixture
def mock_trajectory():
    """Create a mock trajectory with multiple frames."""
    try:
        from ase import Atoms
        frames = []
        for i in range(10):
            atoms = Atoms(
                symbols=['Li', 'S'],
                positions=[
                    [0.1 * i, 0.0, 0.0],
                    [1.0 + 0.05 * i, 1.0, 0.0]
                ],
                cell=[3.0, 3.0, 3.0],
                pbc=True
            )
            atoms.info['energy'] = -5.0 + 0.1 * i
            atoms.arrays['forces'] = np.random.randn(2, 3) * 0.1
            frames.append(atoms)
        return frames
    except ImportError:
        pytest.skip("ASE not available")


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def mock_vasp_config():
    """Create a VASP parser configuration."""
    return VASPParserConfig(
        extract_energy=True,
        extract_forces=True,
        extract_stress=True,
        filter_unconverged=True,
        energy_threshold=100.0
    )


@pytest.fixture
def mock_ff_config():
    """Create a force field fitting configuration."""
    return ForceFieldConfig(
        ff_type="buckingham",
        elements=['Li', 'S', 'P'],
        cutoff=6.0,
        fit_method="least_squares",
        max_iterations=100
    )


@pytest.fixture
def mock_lammps_config():
    """Create a LAMMPS input configuration."""
    return LAMMPSInputConfig(
        units="metal",
        atom_style="atomic",
        boundary="p p p",
        pair_style="buck/coul/long",
        timestep=1.0,
        temperature=300.0,
        ensemble="nvt",
        nsteps=1000,
        thermo_interval=100,
        dump_interval=1000
    )


@pytest.fixture
def mock_nep_data_config():
    """Create a NEP data configuration."""
    return NEPDataConfig(
        energy_threshold=50.0,
        force_threshold=50.0,
        train_ratio=0.9,
        test_ratio=0.1,
        type_map=['Li', 'S', 'P']
    )


@pytest.fixture
def mock_nep_model_config():
    """Create a NEP model configuration."""
    return NEPModelConfig(
        type_list=['Li', 'S', 'P'],
        version=4,
        cutoff_radial=6.0,
        cutoff_angular=4.0,
        n_max_radial=4,
        n_max_angular=4,
        neuron=30,
        population_size=50,
        maximum_generation=10000
    )


@pytest.fixture
def mock_nep_training_config(mock_working_dir):
    """Create a NEP training configuration."""
    return NEPTrainingConfig(
        gpumd_path="/usr/local/gpumd",
        working_dir=str(mock_working_dir),
        use_gpu=False,
        restart=False
    )


@pytest.fixture
def mock_resource_request():
    """Create a resource request for HPC testing."""
    return ResourceRequest(
        num_nodes=1,
        num_cores_per_node=4,
        num_gpus=0,
        memory_gb=16,
        walltime_hours=1.0,
        queue="normal",
        exclusive=False
    )


@pytest.fixture
def mock_job_spec(mock_working_dir, mock_resource_request):
    """Create a job specification for HPC testing."""
    return JobSpec(
        name="test_job",
        working_dir=mock_working_dir,
        commands=["echo 'Hello World'", "sleep 1"],
        resources=mock_resource_request,
        stdout_file="job.out",
        stderr_file="job.err"
    )


# =============================================================================
# Mock Data Fixtures
# =============================================================================

@pytest.fixture
def mock_dft_frames():
    """Create mock DFT calculation frames."""
    frames = []
    for i in range(5):
        frame = {
            'index': i,
            'energy': -100.0 + i * 0.5,
            'energy_per_atom': -10.0 + i * 0.05,
            'forces': np.random.randn(6, 3) * 0.1,
            'max_force': 0.5,
            'rms_force': 0.2,
            'stress': np.array([0.1, 0.1, 0.1, 0.0, 0.0, 0.0]),
            'pressure': 0.5,
            'positions': np.random.randn(6, 3),
            'cell': np.eye(3) * 10.0,
            'pbc': [True, True, True],
            'symbols': ['Li', 'Li', 'S', 'S', 'P', 'P']
        }
        frames.append(frame)
    return frames


@pytest.fixture
def mock_buckingham_params():
    """Create mock Buckingham potential parameters."""
    return {
        'Li-S': {'A': 1000.0, 'rho': 0.3, 'C': 10.0},
        'Li-P': {'A': 800.0, 'rho': 0.25, 'C': 8.0},
        'S-P': {'A': 1200.0, 'rho': 0.35, 'C': 12.0}
    }


@pytest.fixture
def mock_morse_params():
    """Create mock Morse potential parameters."""
    return {
        'Li-S': {'D_e': 1.0, 'a': 1.5, 'r_e': 2.0},
        'Li-P': {'D_e': 0.8, 'a': 1.4, 'r_e': 2.1},
        'S-P': {'D_e': 1.2, 'a': 1.6, 'r_e': 1.9}
    }


@pytest.fixture
def mock_lj_params():
    """Create mock Lennard-Jones potential parameters."""
    return {
        'Li-S': {'epsilon': 0.1, 'sigma': 2.5},
        'Li-P': {'epsilon': 0.08, 'sigma': 2.7},
        'S-P': {'epsilon': 0.12, 'sigma': 2.3}
    }


@pytest.fixture
def mock_outcar_content():
    """Create mock VASP OUTCAR content."""
    return """
 vasp.5.4.4 18Apr17 complex 
  
 executed on           IFC15_xeon64 date 2024.01.15  10:30:45
 running on   32 total cores
distrk:  each k-point on   32 cores,    1 groups
distr:  one band on NCORES_PER_BAND=   1 cores,   32 groups


--------------------------------------------------------------------------------------------------------


 INCAR:
   ENCUT = 520.00 eV
   ISMEAR = 0; SIGMA = 0.050000
   ISPIN = 1
   MAGMOM = 6*0.0

 POTCAR:    PAW_PBE Li 17Jan2003                  
 POTCAR:    PAW_PBE S 17Jan2003                   
 POTCAR:    PAW_PBE P 17Jan2003                   



  energy-cutoff  :      520.00
  volume of cell :      216.00
      direct lattice vectors                 reciprocal lattice vectors
    10.000000000  0.000000000  0.000000000     0.100000000  0.000000000  0.000000000
     0.000000000 10.000000000  0.000000000     0.000000000  0.100000000  0.000000000
     0.000000000  0.000000000 10.000000000     0.000000000  0.000000000  0.100000000

  length of vectors
    10.000000000 10.000000000 10.000000000     0.100000000  0.100000000  0.100000000



 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
      0.00000      0.00000      0.00000        -0.001234    0.002345   -0.003456
      2.00000      0.00000      0.00000         0.001234   -0.002345    0.003456
      1.00000      1.00000      0.00000         0.002345   -0.001234    0.004567
      3.00000      1.00000      0.00000        -0.002345    0.001234   -0.004567
      1.00000      0.00000      1.00000         0.003456   -0.004567    0.001234
      2.00000      1.00000      1.00000        -0.003456    0.004567   -0.001234
 -----------------------------------------------------------------------------------
    total drift:                               -0.000012    0.000023   -0.000034


  free  energy   TOTEN  =      -100.12345678 eV

  energy  without entropy=     -100.00000000  energy(sigma->0) =     -100.06172839



--------------------------------------------------------------------------------------------------------



  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)
  ---------------------------------------------------
  free  energy   TOTEN  =      -100.12345678 eV

  energy  without entropy=     -100.00000000  energy(sigma->0) =     -100.06172839
  ---------------------------------------------------
  band energy                =     -50.12345678
  

--------------------------------------------------------------------------------------------------------


    Convergence achieved after    15 iterations

    """


@pytest.fixture
def mock_nep_xyz_content():
    """Create mock NEP XYZ file content."""
    return """6
Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" Properties=species:S:1:pos:R:3:forces:R:3 energy=-100.1234567890
 Li   0.00000000   0.00000000   0.00000000  -0.00123400   0.00234500  -0.00345600
 Li   2.00000000   0.00000000   0.00000000   0.00123400  -0.00234500   0.00345600
  S   1.00000000   1.00000000   0.00000000   0.00234500  -0.00123400   0.00456700
  S   3.00000000   1.00000000   0.00000000  -0.00234500   0.00123400  -0.00456700
  P   1.00000000   0.00000000   1.00000000   0.00345600  -0.00456700   0.00123400
  P   2.00000000   1.00000000   1.00000000  -0.00345600   0.00456700  -0.00123400
6
Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" Properties=species:S:1:pos:R:3:forces:R:3 energy=-99.6234567890
 Li   0.10000000   0.00000000   0.00000000  -0.00113400   0.00224500  -0.00335600
 Li   2.05000000   0.00000000   0.00000000   0.00133400  -0.00244500   0.00355600
  S   1.02000000   1.01000000   0.00000000   0.00244500  -0.00133400   0.00466700
  S   3.03000000   1.02000000   0.00000000  -0.00244500   0.00133400  -0.00466700
  P   1.01000000   0.01000000   1.01000000   0.00355600  -0.00466700   0.00133400
  P   2.02000000   1.01000000   1.01000000  -0.00355600   0.00466700  -0.00133400
"""


# =============================================================================
# Mock Object Fixtures
# =============================================================================

@pytest.fixture
def mock_calculator():
    """Create a mock ASE calculator."""
    calc = Mock()
    calc.get_potential_energy.return_value = -100.0
    calc.get_forces.return_value = np.random.randn(6, 3) * 0.1
    calc.get_stress.return_value = np.array([0.1, 0.1, 0.1, 0.0, 0.0, 0.0])
    return calc


@pytest.fixture
def mock_vasp_parser(mock_vasp_config):
    """Create a VASP parser instance."""
    return VASPOUTCARParser(mock_vasp_config)


@pytest.fixture
def mock_ff_fitter(mock_ff_config):
    """Create a force field fitter instance."""
    return ForceFieldFitter(mock_ff_config)


@pytest.fixture
def mock_lammps_generator(mock_lammps_config):
    """Create a LAMMPS input generator instance."""
    return LAMMPSInputGenerator(mock_lammps_config)


@pytest.fixture
def mock_nep_preparer(mock_nep_data_config):
    """Create a NEP data preparer instance."""
    return NEPDataPreparer(mock_nep_data_config)


@pytest.fixture
def mock_nep_input_generator(mock_nep_model_config):
    """Create a NEP input generator instance."""
    return NEPInputGenerator(mock_nep_model_config)


@pytest.fixture
def mock_local_scheduler():
    """Create a local scheduler for testing."""
    return LocalScheduler()


# =============================================================================
# Pytest Configuration Hooks
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "hpc: marks tests as HPC-related tests")
    config.addinivalue_line("markers", "dft: marks tests as DFT-related tests")
    config.addinivalue_line("markers", "ml: marks tests as machine learning tests")
    config.addinivalue_line("markers", "md: marks tests as molecular dynamics tests")
    config.addinivalue_line("markers", "benchmark: marks tests as performance benchmarks")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        
        # Add slow marker to certain tests
        if "benchmark" in item.nodeid or "performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.benchmark)


# =============================================================================
# Utility Functions
# =============================================================================

def create_mock_outcar(path: Path, energy: float = -100.0, n_atoms: int = 6):
    """Create a mock OUTCAR file for testing."""
    content = f"""
 vasp.5.4.4 18Apr17 complex 
 executed on           IFC15_xeon64 date 2024.01.15  10:30:45

 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
"""
    for i in range(n_atoms):
        pos = np.random.rand(3) * 10
        force = np.random.randn(3) * 0.1
        content += f"    {pos[0]:8.5f}    {pos[1]:8.5f}    {pos[2]:8.5f}       {force[0]:10.6f}   {force[1]:10.6f}   {force[2]:10.6f}\n"
    
    content += f""" -----------------------------------------------------------------------------------

  free  energy   TOTEN  =      {energy:.8f} eV

  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)
  ---------------------------------------------------
  free  energy   TOTEN  =      {energy:.8f} eV
  ---------------------------------------------------

    Convergence achieved after    15 iterations

"""
    path.write_text(content)
    return path


def create_mock_nep_xyz(path: Path, n_frames: int = 2, n_atoms: int = 6):
    """Create a mock NEP XYZ file for testing."""
    content = ""
    for frame in range(n_frames):
        content += f"{n_atoms}\n"
        energy = -100.0 + frame * 0.5
        content += f'Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" Properties=species:S:1:pos:R:3:forces:R:3 energy={energy:.10f}\n'
        
        for i in range(n_atoms // 3):
            content += " Li   0.00000000   0.00000000   0.00000000   0.00100000   0.00200000   0.00300000\n"
            content += "  S   1.00000000   1.00000000   0.00000000  -0.00100000  -0.00200000   0.00100000\n"
            content += "  P   2.00000000   0.00000000   1.00000000   0.00200000  -0.00100000  -0.00200000\n"
    
    path.write_text(content)
    return path


def create_mock_lammps_input(path: Path):
    """Create a mock LAMMPS input file for testing."""
    content = """# LAMMPS input file
units metal
atom_style atomic
boundary p p p
read_data structure.data
pair_style buck/coul/long 6.0
pair_coeff 1 1 1000.0 0.3 10.0
neighbor 2.0 bin
neigh_modify every 10 delay 0 check yes
timestep 0.001
velocity all create 300 12345
dump traj all custom 100 dump.lammpstrj id type x y z
thermo 100
run 1000
"""
    path.write_text(content)
    return path


# Make utilities available to tests
__all__ = [
    'create_mock_outcar',
    'create_mock_nep_xyz',
    'create_mock_lammps_input',
]