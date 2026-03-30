"""
Setup script for dftlammps-multiscale package.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dftlammps-multiscale",
    version="0.1.0",
    author="DFTLAMMPS Team",
    description="Multiscale coupling module for DFTLAMMPS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "ml": ["torch>=1.9.0", "scikit-learn>=0.24.0"],
        "ase": ["ase>=3.22.0"],
        "full": ["torch>=1.9.0", "scikit-learn>=0.24.0", "ase>=3.22.0"],
    },
)
