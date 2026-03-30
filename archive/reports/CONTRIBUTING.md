# Contributing to DFT+LAMMPS

Thank you for your interest in contributing to the DFT+LAMMPS integration package! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Documentation](#documentation)
- [Release Process](#release-process)

---

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow:

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect differing viewpoints and experiences

## Getting Started

### Reporting Issues

If you find a bug or have a feature request:

1. **Search existing issues** to avoid duplicates
2. **Create a new issue** with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - System information (OS, Python version, package versions)
   - Minimal code example if applicable

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Documentation improvements
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed

---

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Installation

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/dftlammps.git
   cd dftlammps
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode:**
   ```bash
   pip install -e ".[dev,all]"
   ```

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

### Project Structure

```
dftlammps/
├── dftlammps/           # Main package
│   ├── __init__.py
│   ├── core/            # Core modules (DFT, ML, MD)
│   ├── hpc/             # HPC scheduling
│   ├── applications/    # Application cases
│   └── utils/           # Utilities
├── tests/               # Test suite
├── examples/            # Example scripts
├── docs/                # Documentation
├── setup.py
├── pyproject.toml
└── README.md
```

---

## Code Style

We follow PEP 8 with some modifications. Our style is enforced by `black` and `flake8`.

### Python Style Guide

#### Formatting

- **Line length**: 100 characters maximum
- **Formatter**: Use `black` for automatic formatting
  ```bash
  black dftlammps/ tests/
  ```

#### Import Ordering

Use the following import order:

```python
# 1. Standard library imports
import os
import sys
from pathlib import Path
from typing import Dict, List

# 2. Third-party imports
import numpy as np
import pandas as pd
from ase import Atoms
from pymatgen.core import Structure

# 3. Local imports
from dftlammps.core import DFTToLAMMPSBridge
from .utils import helper_function
```

#### Type Hints

Use type hints for function signatures:

```python
def calculate_energy(structure: Atoms, potential: str) -> float:
    """Calculate energy of structure.
    
    Args:
        structure: Atomic structure
        potential: Potential name
        
    Returns:
        Total energy in eV
    """
    pass
```

#### Docstrings

Use Google-style docstrings:

```python
def my_function(param1: int, param2: str) -> bool:
    """Short description.
    
    Longer description if needed. Can span multiple lines.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is negative
        
    Example:
        >>> result = my_function(5, "test")
        >>> print(result)
        True
    """
    if param1 < 0:
        raise ValueError("param1 must be non-negative")
    return True
```

### Code Quality Tools

Run all quality checks before committing:

```bash
# Format code
black dftlammps/ tests/

# Check style
flake8 dftlammps/ tests/

# Type checking
mypy dftlammps/

# Run tests
pytest tests/
```

---

## Testing

### Test Structure

Tests are located in the `tests/` directory:

```
tests/
├── __init__.py
├── conftest.py          # pytest fixtures
├── test_core/           # Core module tests
│   ├── __init__.py
│   ├── test_dft_bridge.py
│   ├── test_ml_potential.py
│   └── test_md_simulation.py
├── test_hpc/
│   └── test_scheduler.py
└── test_integration.py  # Integration tests
```

### Writing Tests

Use `pytest` for testing:

```python
import pytest
import numpy as np
from dftlammps.core import MDConfig, MDSimulationRunner

def test_md_config_defaults():
    """Test MDConfig default values."""
    config = MDConfig()
    assert config.temperature == 300.0
    assert config.timestep == 1.0
    assert config.ensemble == "nvt"

def test_md_runner_initialization():
    """Test MDSimulationRunner initialization."""
    config = MDConfig(temperature=500.0)
    runner = MDSimulationRunner(config)
    assert runner.config.temperature == 500.0

@pytest.mark.parametrize("ensemble", ["nve", "nvt", "npt"])
def test_md_ensembles(ensemble):
    """Test different MD ensembles."""
    config = MDConfig(ensemble=ensemble)
    assert config.ensemble == ensemble
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=dftlammps --cov-report=html

# Run specific test file
pytest tests/test_core/test_dft_bridge.py

# Run with verbose output
pytest -v

# Run only slow tests
pytest -m slow

# Run excluding slow tests
pytest -m "not slow"
```

### Test Coverage

Aim for at least 80% code coverage. Check coverage with:

```bash
pytest --cov=dftlammps --cov-report=term-missing
```

---

## Pull Request Process

### Before Submitting

1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Run all tests** and ensure they pass
4. **Update CHANGELOG.md** with your changes
5. **Ensure code is formatted** with `black`

### Submitting a PR

1. **Create a branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and commit:
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

3. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a Pull Request** on GitHub:
   - Fill out the PR template
   - Link related issues
   - Request review from maintainers

### PR Review Process

- All PRs require at least one review
- CI checks must pass
- Address review comments
- Squash commits if requested

### PR Title Format

Follow conventional commits format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, semicolons, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(core): add support for ABACUS DFT code
fix(hpc): resolve Slurm job dependency issue
docs(api): update docstrings for MD classes
test(ml): add unit tests for NEP training
```

---

## Commit Message Guidelines

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Examples

```
feat(core): add VASP6 compatibility

Add support for VASP6 OUTCAR format including
new XML-based output parsing.

Closes #123
```

```
fix(hpc): correct PBS resource allocation

Fix memory specification format for PBS Pro
scheduler versions >= 19.0.

Fixes #456
```

```
docs: update installation instructions

Add instructions for conda installation and
GPU support setup.
```

---

## Documentation

### API Documentation

API docs are generated from docstrings. Ensure all public functions have:

- Description
- Args with types
- Returns with type
- Raises (if applicable)
- Examples (if helpful)

### User Documentation

- Update `README.md` for user-facing changes
- Add examples to `examples/` directory
- Update `API_REFERENCE.md` for API changes

### Building Docs

```bash
cd docs/
make html
```

View at `docs/_build/html/index.html`

---

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- `MAJOR.MINOR.PATCH`
- MAJOR: Incompatible API changes
- MINOR: Backward-compatible functionality
- PATCH: Backward-compatible bug fixes

### Release Checklist

1. Update version in `dftlammps/__init__.py`
2. Update `CHANGELOG.md`
3. Create a git tag:
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin v1.0.0
   ```
4. Create release on GitHub
5. Publish to PyPI:
   ```bash
   python -m build
   twine upload dist/*
   ```

---

## Questions?

- **General questions**: Open a [Discussion](https://github.com/example/dftlammps/discussions)
- **Bug reports**: Open an [Issue](https://github.com/example/dftlammps/issues)
- **Security issues**: Email security@example.com

---

Thank you for contributing! 🎉
