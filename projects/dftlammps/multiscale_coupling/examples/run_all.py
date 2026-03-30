#!/usr/bin/env python3
"""
Run all examples and generate report.
"""
import subprocess
import sys
import os
from pathlib import Path

EXAMPLES = [
    'ex1_qmmm_water.py',
    'ex2_ml_coarse_graining.py', 
    'ex3_gnn_force_field.py',
    'ex4_validation.py',
    'ex5_multiscale_gnn.py'
]

def run_example(example_file):
    """Run a single example."""
    print(f"\n{'='*60}")
    print(f"Running: {example_file}")
    print('='*60)
    
    example_path = Path(__file__).parent / example_file
    
    try:
        result = subprocess.run(
            [sys.executable, str(example_path)],
            capture_output=False,
            text=True,
            timeout=60
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {example_file}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    """Run all examples."""
    print("="*60)
    print("Running Multiscale Coupling Examples")
    print("="*60)
    
    results = {}
    for example in EXAMPLES:
        success = run_example(example)
        results[example] = success
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print('='*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for example, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {example}")
    
    print(f"\nTotal: {passed}/{total} examples passed")
    
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
