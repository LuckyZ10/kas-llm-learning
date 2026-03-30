"""Test suite for dftlammps package."""

import pytest
import numpy as np
from ase import Atoms

# Test fixtures

@pytest.fixture
def simple_atoms():
    """Create a simple ASE Atoms object for testing."""
    return Atoms(
        symbols=['Li', 'S'],
        positions=[[0, 0, 0], [2, 0, 0]],
        cell=[5, 5, 5],
        pbc=True
    )

@pytest.fixture
def li3ps4_structure():
    """Create a Li3PS4 structure for testing."""
    positions = [
        [0.0, 0.0, 0.0],      # P
        [1.5, 1.5, 1.5],      # S
        [1.5, -1.5, -1.5],    # S
        [-1.5, 1.5, -1.5],    # S
        [-1.5, -1.5, 1.5],    # S
        [2.5, 0.0, 0.0],      # Li
        [0.0, 2.5, 0.0],      # Li
        [0.0, 0.0, 2.5],      # Li
    ]
    return Atoms(
        symbols=['P', 'S', 'S', 'S', 'S', 'Li', 'Li', 'Li'],
        positions=positions,
        cell=[6, 6, 6],
        pbc=True
    )

@pytest.fixture
def mock_vasp_data():
    """Create mock VASP calculation data."""
    return {
        'energies': [-123.45, -124.56, -125.67],
        'forces': [
            np.random.randn(8, 3),
            np.random.randn(8, 3),
            np.random.randn(8, 3),
        ],
        'stress': [
            np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]),
            np.array([1.1, 2.1, 3.1, 0.15, 0.25, 0.35]),
            np.array([1.2, 2.2, 3.2, 0.2, 0.3, 0.4]),
        ]
    }

# Test configuration

def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "requires_lammps: marks tests requiring LAMMPS")
    config.addinivalue_line("markers", "requires_vasp: marks tests requiring VASP")
