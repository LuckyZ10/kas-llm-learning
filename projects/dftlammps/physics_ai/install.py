#!/usr/bin/env python3
"""
Installation script for Physics-AI module.
"""

import subprocess
import sys

def install_requirements():
    """Install required packages."""
    packages = [
        'torch>=1.10.0',
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'matplotlib>=3.3.0',
        'sympy>=1.8',
        'scikit-learn>=0.24.0',
    ]
    
    print("Installing core dependencies...")
    for package in packages:
        print(f"  Installing {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    
    # Optional packages
    optional = [
        'pysr>=0.11.0',
        'gplearn>=0.4.0',
        'torch-geometric>=2.0.0',
    ]
    
    print("\nInstalling optional dependencies...")
    for package in optional:
        try:
            print(f"  Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        except:
            print(f"  Warning: Failed to install {package}")
    
    print("\nInstallation complete!")

if __name__ == '__main__':
    install_requirements()
