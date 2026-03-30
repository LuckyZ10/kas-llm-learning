"""
DFTLammps Spectroscopy Module
==============================

Calculation of vibrational spectroscopic properties from phonon data.

This module provides tools for:
- Raman spectroscopy
- Infrared (IR) absorption spectroscopy
- Neutron scattering cross sections

Submodules:
-----------
raman_calculator : Raman spectroscopy
    - Raman tensor calculation from DFPT
    - Temperature-dependent Raman spectra
    - Polarization-resolved Raman

ir_calculator : Infrared spectroscopy
    - IR intensity from Born effective charges
    - Dielectric function
    - Absorption coefficient and reflectivity

neutron_scattering : Neutron scattering
    - S(Q, ω) calculation
    - Coherent and incoherent scattering
    - Powder diffraction patterns

Example Usage:
--------------
    from dftlammps.spectroscopy import RamanCalculator, RamanConfig
    from dftlammps.spectroscopy import IRCalculator, IRConfig
    from dftlammps.spectroscopy import NeutronScatteringCalculator

    # Raman calculation
    raman_config = RamanConfig(laser_wavelength=532.0)
    raman_calc = RamanCalculator(raman_config)
    raman_calc.calculate_raman_tensors_phonopy(phonopy, born_charges, epsilon_inf)
    raman_spectrum = raman_calc.calculate_spectrum()
    
    # IR calculation
    ir_config = IRConfig(gamma=5.0)
    ir_calc = IRCalculator(ir_config)
    ir_calc.calculate_ir_tensors(phonopy, born_charges)
    ir_spectrum = ir_calc.calculate_spectrum(epsilon_infinity=epsilon_inf)
    
    # Neutron scattering
    neutron_calc = NeutronScatteringCalculator()
    neutron_spectrum = neutron_calc.calculate_s_qw(phonopy, q_path)
"""

__version__ = "1.0.0"
__author__ = "DFTLammps Spectroscopy Team"

from .raman_calculator import (
    RamanCalculator,
    RamanConfig,
    RamanTensor,
    RamanSpectrum,
    RamanMode,
    calculate_raman_spectrum_workflow
)

from .ir_calculator import (
    IRCalculator,
    IRConfig,
    IRTensor,
    IRSpectrum,
    calculate_ir_spectrum_workflow
)

from .neutron_scattering import (
    NeutronScatteringCalculator,
    NeutronConfig,
    NeutronSpectrum,
    calculate_neutron_scattering_workflow,
    NEUTRON_SCATTERING_LENGTHS
)

__all__ = [
    # Raman
    'RamanCalculator',
    'RamanConfig',
    'RamanTensor',
    'RamanSpectrum',
    'RamanMode',
    'calculate_raman_spectrum_workflow',
    
    # IR
    'IRCalculator',
    'IRConfig',
    'IRTensor',
    'IRSpectrum',
    'calculate_ir_spectrum_workflow',
    
    # Neutron
    'NeutronScatteringCalculator',
    'NeutronConfig',
    'NeutronSpectrum',
    'calculate_neutron_scattering_workflow',
    'NEUTRON_SCATTERING_LENGTHS'
]
