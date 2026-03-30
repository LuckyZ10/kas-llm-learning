"""
DFTLammps Electron-Phonon Module
=================================

Calculation of electron-phonon coupling and related properties.

This module provides tools for:
- Electron-phonon coupling matrix elements
- Eliashberg spectral function and coupling constant λ
- Superconducting critical temperature Tc
- Electronic transport properties (resistivity, mobility)

Submodules:
-----------
epw_interface : Quantum ESPRESSO EPW interface
    - Wannier function generation
    - EPW calculation workflow
    - Fine q-point interpolation

elph_calculator : Electron-phonon coupling
    - Matrix element calculation
    - Eliashberg function α²F(ω)
    - Coupling constant λ
    - Tc calculation (McMillan, Allen-Dynes)

transport_eph : Electronic transport
    - Electrical resistivity ρ(T)
    - Electron mobility μ(T)
    - Seebeck coefficient S(T)
    - Wiedemann-Franz law

Example Usage:
--------------
    from dftlammps.electron_phonon import EPWInterface, EPWConfig
    from dftlammps.electron_phonon import ElectronPhononCalculator
    from dftlammps.electron_phonon import ElectronPhononTransport

    # EPW workflow
    epw_config = EPWConfig(prefix='my_material', n_wannier=8)
    epw = EPWInterface(epw_config)
    epw.run_full_workflow(structure, pseudopotentials)
    
    # Calculate coupling and Tc
    elph = ElectronPhononCalculator()
    results = elph.run_full_calculation(frequencies, g_kq_nu, energies_k, energies_kq)
    print(f"Tc = {results.tc_mcmillan:.2f} K")
    
    # Transport properties
    transport = ElectronPhononTransport()
    transport.run_full_transport_calculation(lambda_eph, omega_log, debye_temp)
    transport.plot_transport_properties()
"""

__version__ = "1.0.0"
__author__ = "DFTLammps Electron-Phonon Team"

from .epw_interface import (
    EPWInterface,
    EPWConfig,
    EPWResults,
    calculate_lambda_from_a2f
)

from .elph_calculator import (
    ElectronPhononCalculator,
    ElPhConfig,
    ElPhResults
)

from .transport_eph import (
    ElectronPhononTransport,
    TransportConfig,
    TransportResults
)

__all__ = [
    # EPW interface
    'EPWInterface',
    'EPWConfig',
    'EPWResults',
    'calculate_lambda_from_a2f',
    
    # E-ph calculator
    'ElectronPhononCalculator',
    'ElPhConfig',
    'ElPhResults',
    
    # Transport
    'ElectronPhononTransport',
    'TransportConfig',
    'TransportResults'
]
