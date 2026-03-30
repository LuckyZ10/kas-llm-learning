"""
DFT+LAMMPS Integration Package
==============================
A comprehensive Python package for integrating DFT calculations 
with classical molecular dynamics simulations using LAMMPS.
"""

from setuptools import setup, find_packages
import os

# Read README
readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = "DFT+LAMMPS Integration Package"

# Read requirements
requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
if os.path.exists(requirements_path):
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = [
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'ase>=3.22.0',
        'pymatgen>=2022.0.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.4.0',
    ]

setup(
    name='dftlammps',
    version='1.0.0',
    author='DFT+LAMMPS Integration Team',
    author_email='dftlammps@example.com',
    description='Integration of DFT calculations with LAMMPS molecular dynamics',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/example/dftlammps',
    packages=find_packages(exclude=['tests', 'tests.*', 'examples', 'examples.*']),
    package_data={
        'dftlammps': ['*.yaml', '*.json', 'data/*'],
    },
    include_package_data=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=1.0.0',
        ],
        'deepmd': [
            'deepmd-kit>=2.0.0',
            'dpdata>=0.2.0',
        ],
        'screening': [
            'matminer>=0.7.0',
            'dscribe>=1.2.0',
            'mp-api>=0.30.0',
        ],
        'viz': [
            'ovito>=3.7.0',
            'nglview>=3.0.0',
            'plotly>=5.0.0',
        ],
        'all': [
            'deepmd-kit>=2.0.0',
            'dpdata>=0.2.0',
            'matminer>=0.7.0',
            'dscribe>=1.2.0',
            'mp-api>=0.30.0',
            'ovito>=3.7.0',
            'nglview>=3.0.0',
            'plotly>=5.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'dftlammps=dftlammps.cli:main',
            'dftlammps-workflow=dftlammps.cli.workflow:main',
            'dftlammps-screening=dftlammps.cli.screening:main',
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/example/dftlammps/issues',
        'Source': 'https://github.com/example/dftlammps',
        'Documentation': 'https://dftlammps.readthedocs.io',
    },
    zip_safe=False,
)
