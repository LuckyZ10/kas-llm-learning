#!/usr/bin/env python3
"""
Validation script for Physics-AI module.
Checks that the module can be imported and basic functionality works.
"""

import sys
import os

def check_imports():
    """Check that all modules can be imported."""
    print("=" * 60)
    print("Checking imports...")
    print("=" * 60)
    
    errors = []
    
    try:
        from dftlammps.physics_ai.core import PhysicsConstraintLayer, EnergyConservation, MomentumConservation, MassConservation
        print("✓ Core module")
    except Exception as e:
        errors.append(f"Core module: {e}")
        print(f"✗ Core module: {e}")
    
    try:
        from dftlammps.physics_ai.models import (
            PhysicsInformedNN, SIREN, AdaptiveWeightPINN,
            DeepONet, SeparablePhysicsInformedDeepONet, MultiOutputDeepONet,
            FourierNeuralOperator, PhysicsInformedFNO, MultiScaleFNO,
            PhysicsInformedGNN, MomentumConservingGNN, HamiltonianGNN
        )
        print("✓ Models module")
    except Exception as e:
        errors.append(f"Models module: {e}")
        print(f"✗ Models module: {e}")
    
    try:
        from dftlammps.physics_ai.validators import PhysicsLawValidator, ValidationResult, PhysicsTest
        print("✓ Validators module")
    except Exception as e:
        errors.append(f"Validators module: {e}")
        print(f"✗ Validators module: {e}")
    
    try:
        from dftlammps.physics_ai.symbolic import (
            SymbolicExpression, SymbolicRegressionEngine,
            PySRBackend, GPLearnBackend, AIFeynmanBackend
        )
        print("✓ Symbolic module")
    except Exception as e:
        errors.append(f"Symbolic module: {e}")
        print(f"✗ Symbolic module: {e}")
    
    try:
        from dftlammps.physics_ai.integration import MDPotentialFitter
        print("✓ Integration module")
    except Exception as e:
        errors.append(f"Integration module: {e}")
        print(f"✗ Integration module: {e}")
    
    return len(errors) == 0

def check_file_structure():
    """Check that all files are present."""
    print("\n" + "=" * 60)
    print("Checking file structure...")
    print("=" * 60)
    
    required_files = [
        '__init__.py',
        'core/__init__.py',
        'core/physics_layer.py',
        'core/conservation.py',
        'models/__init__.py',
        'models/pinns.py',
        'models/deeponet.py',
        'models/fno.py',
        'models/physics_gnn.py',
        'symbolic/__init__.py',
        'symbolic/regression_engine.py',
        'validators/__init__.py',
        'validators/physics_validator.py',
        'integration/__init__.py',
        'integration/md_potential.py',
        'tests/test_physics_ai.py',
        'README.md',
    ]
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    all_present = True
    for file in required_files:
        path = os.path.join(base_path, file)
        if os.path.exists(path):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING")
            all_present = False
    
    return all_present

def check_code_quality():
    """Check code quality metrics."""
    print("\n" + "=" * 60)
    print("Code quality metrics...")
    print("=" * 60)
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Count lines
    total_lines = 0
    python_files = []
    
    for root, dirs, files in os.walk(base_path):
        # Skip __pycache__
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                python_files.append(path)
                with open(path) as f:
                    total_lines += len(f.readlines())
    
    print(f"Total Python files: {len(python_files)}")
    print(f"Total lines of code: {total_lines:,}")
    print(f"Average lines per file: {total_lines // len(python_files)}")
    
    return True

def main():
    """Run all validation checks."""
    print("\n")
    print("=" * 60)
    print("PHYSICS-AI MODULE VALIDATION")
    print("=" * 60)
    print()
    
    results = []
    
    # Check imports
    try:
        results.append(("Imports", check_imports()))
    except Exception as e:
        print(f"Import check failed: {e}")
        results.append(("Imports", False))
    
    # Check file structure
    results.append(("File Structure", check_file_structure()))
    
    # Check code quality
    results.append(("Code Quality", check_code_quality()))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n✓ All validation checks passed!")
        return 0
    else:
        print("\n✗ Some validation checks failed!")
        return 1

if __name__ == '__main__':
    sys.exit(main())
