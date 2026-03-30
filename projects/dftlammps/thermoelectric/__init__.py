"""
DFTLammps Thermoelectric Transport Module

This module provides tools for calculating thermoelectric properties
including Seebeck coefficient, electrical and thermal conductivity,
and the thermoelectric figure of merit ZT.

Example Usage:
    from dftlammps.thermoelectric import (
        SeebeckCalculator,
        TransportCoefficients,
        ZTOptimizer,
        ThermoelectricDevice
    )
    
    # Calculate Seebeck coefficient
    calc = SeebeckCalculator(temperatures=[300, 400, 500])
    temps, seebeck = calc.calculate_from_transmission(
        energies, transmission, fermi_level=0.0
    )
    
    # Calculate ZT
    coeffs = TransportCoefficients(
        sigma=conductivity,
        seebeck=seebeck_coeff,
        kappa_e=thermal_cond_elec,
        kappa_l=thermal_cond_latt,
        temperatures=temps
    )
    zt = coeffs.calculate_zt()
"""

from .thermoelectric import (
    # Data classes
    TransportCoefficients,
    
    # Calculators
    SeebeckCalculator,
    ConductivityCalculator,
    
    # Optimizers
    ZTOptimizer,
    
    # Device simulation
    ThermoelectricDevice,
)

__all__ = [
    'TransportCoefficients',
    'SeebeckCalculator',
    'ConductivityCalculator',
    'ZTOptimizer',
    'ThermoelectricDevice',
]

__version__ = "1.0.0"
