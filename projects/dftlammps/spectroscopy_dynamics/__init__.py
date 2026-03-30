"""
Spectroscopy Dynamics Module
============================

Time-resolved spectroscopy simulation and analysis tools.

Submodules:
- spectroscopy_dynamics: Core spectroscopy simulation classes

Classes:
- LaserPulse: Laser pulse definition and properties
- ElectronicTransition: Electronic transition representation
- UltrafastAbsorption: Transient absorption simulator
- TimeResolvedPhotoelectronSpectroscopy: TRPES simulator
- VibrationalCoherenceAnalysis: Coherence extraction tools
- SpectroscopyDynamicsWorkflow: Complete workflow

Author: dftlammps development team
"""

from .spectroscopy_dynamics import (
    LaserPulse,
    ElectronicTransition,
    UltrafastAbsorption,
    TimeResolvedPhotoelectronSpectroscopy,
    VibrationalCoherenceAnalysis,
    SpectroscopyDynamicsWorkflow,
)

__all__ = [
    'LaserPulse',
    'ElectronicTransition',
    'UltrafastAbsorption',
    'TimeResolvedPhotoelectronSpectroscopy',
    'VibrationalCoherenceAnalysis',
    'SpectroscopyDynamicsWorkflow',
]

__version__ = '0.1.0'
