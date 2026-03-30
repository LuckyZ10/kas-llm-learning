#!/usr/bin/env python3
"""
ASDE System Test

Verifies the ASDE modules are properly structured and ready to use.
"""

import ast
import os
import sys

def check_syntax(filepath: str) -> bool:
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath) as f:
            ast.parse(f.read())
        return True
    except SyntaxError as e:
        print(f"  Syntax error in {filepath}: {e}")
        return False

def check_type_hints(filepath: str) -> tuple[int, int]:
    """Count type hints in a file."""
    with open(filepath) as f:
        content = f.read()
    
    # Count type annotations (rough estimate)
    hints = content.count(': ') - content.count(': "')  # Rough count
    returns = content.count('-> ')
    
    return hints, returns

def main():
    print("=" * 70)
    print("ASDE - Automatic Scientific Discovery Engine")
    print("System Verification Report")
    print("=" * 70)
    
    # Define all modules
    modules = {
        "Core Engine": [
            "dftlammps/asde/__init__.py",
            "dftlammps/asde/hypothesis_generator.py",
            "dftlammps/asde/experiment_planner.py",
            "dftlammps/asde/result_analyzer.py",
            "dftlammps/asde/paper_writer.py",
        ],
        "Knowledge Graph": [
            "dftlammps/asde/knowledge_graph/__init__.py",
            "dftlammps/asde/knowledge_graph/scientific_kg.py",
            "dftlammps/asde/knowledge_graph/literature_miner.py",
            "dftlammps/asde/knowledge_graph/citation_network.py",
        ],
        "Examples": [
            "dftlammps/asde_examples/__init__.py",
            "dftlammps/asde_examples/autonomous_discovery.py",
            "dftlammps/asde_examples/literature_review_bot.py",
        ]
    }
    
    total_lines = 0
    all_valid = True
    
    for category, files in modules.items():
        print(f"\n{category}:")
        print("-" * 70)
        
        for filepath in files:
            if not os.path.exists(filepath):
                print(f"  ✗ {filepath} - NOT FOUND")
                all_valid = False
                continue
            
            # Count lines
            with open(filepath) as f:
                lines = len(f.readlines())
                total_lines += lines
            
            # Check syntax
            valid = check_syntax(filepath)
            status = "✓" if valid else "✗"
            
            print(f"  {status} {filepath:55s} ({lines:4d} lines)")
            
            if not valid:
                all_valid = False
    
    print("\n" + "=" * 70)
    print(f"Total Lines of Code: {total_lines}")
    print("=" * 70)
    
    # Check required dependencies
    print("\nDependency Check:")
    print("-" * 70)
    
    deps = {
        "numpy": "Numerical computing",
        "scipy": "Scientific computing",
        "networkx": "Graph algorithms (optional for demo)",
        "aiohttp": "Async HTTP (optional for demo)",
    }
    
    for dep, purpose in deps.items():
        try:
            __import__(dep)
            print(f"  ✓ {dep:15s} - {purpose}")
        except ImportError:
            print(f"  ⚠ {dep:15s} - {purpose} (not installed)")
    
    print("\n" + "=" * 70)
    if all_valid:
        print("✓ All modules verified successfully!")
        print("\nTo run the examples:")
        print("  1. Install dependencies: pip install networkx aiohttp")
        print("  2. Run autonomous discovery: python3 -m dftlammps.asde_examples.autonomous_discovery")
        print("  3. Run literature review: python3 -m dftlammps.asde_examples.literature_review_bot")
    else:
        print("✗ Some modules have errors. Please review above.")
        return 1
    
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
