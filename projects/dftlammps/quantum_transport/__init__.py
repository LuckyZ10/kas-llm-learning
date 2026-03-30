#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DFTLAMMPS Quantum Transport Module
==================================

This module provides comprehensive tools for quantum transport calculations
including NEGF formalism, interfaces to TranSIESTA and Kwant, and various
transport property calculations.

Submodules:
-----------
- negf_formalism : Core NEGF implementation
- transiesta_interface : Interface to TranSIESTA/SIESTA
- kwant_interface : Interface to Kwant tight-binding code

Example Usage:
--------------
>>> from dftlammps.quantum_transport import NEGFTransport, TransportParameters
>>> params = TransportParameters(temperature=300.0)
>>> negf = NEGFTransport(hamiltonian, params)
>>> negf.add_lead("left", lead_info)
>>> negf.add_lead("right", lead_info)
>>> T = LandauerButtiker(negf)
>>> transmission = T.calculate_transmission(energy, "left", "right")
"""

from .negf_formalism import (
    NEGFTransport,
    LandauerButtiker,
    TransportParameters,
    LeadInfo,
    SurfaceGreensFunction,
    SelfEnergyCalculator,
    RecursiveGreenFunction,
    create_tight_binding_chain,
    create_graphene_nanoribbon,
    HBAR,
    Q_E,
    K_B,
    G_0
)

from .transiesta_interface import (
    TranSIESTACalculator,
    TranSIESTAParameters,
    Electrode,
    TranSIESTAMode,
    TransmissionAnalyzer,
    IVCurveCalculator
)

from .kwant_interface import (
    LatticeBuilder,
    BallisticTransport,
    HallEffectCalculator,
    TopologicalInvariant,
    TightBindingParameters,
    LatticeType
)

__version__ = "1.0.0"
__author__ = "DFTLAMMPS Team"

__all__ = [
    # NEGF formalism
    'NEGFTransport',
    'LandauerButtiker',
    'TransportParameters',
    'LeadInfo',
    'SurfaceGreensFunction',
    'SelfEnergyCalculator',
    'RecursiveGreenFunction',
    'create_tight_binding_chain',
    'create_graphene_nanoribbon',
    
    # TranSIESTA
    'TranSIESTACalculator',
    'TranSIESTAParameters',
    'Electrode',
    'TranSIESTAMode',
    'TransmissionAnalyzer',
    'IVCurveCalculator',
    
    # Kwant
    'LatticeBuilder',
    'BallisticTransport',
    'HallEffectCalculator',
    'TopologicalInvariant',
    'TightBindingParameters',
    'LatticeType',
    
    # Constants
    'HBAR',
    'Q_E',
    'K_B',
    'G_0'
]
