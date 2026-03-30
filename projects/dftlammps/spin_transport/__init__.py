"""
DFTLammps Spin Transport Module

This module provides tools for spin-dependent transport and spintronics
including Magnetic Tunnel Junctions, Spin Transfer Torque, Spin Hall Effect,
and Non-local Spin Valves.

Example Usage:
    from dftlammps.spin_transport import (
        MagneticTunnelJunction, 
        MagneticLayer,
        SpinTransferTorque,
        SpinHallEffect
    )
    
    # Create MTJ
    free_layer = MagneticLayer(...)
    pinned_layer = MagneticLayer(...)
    mtj = MagneticTunnelJunction(free_layer, pinned_layer, barrier)
    
    # Calculate TMR
    tmr = mtj.calculate_tmr_ratio()
    
    # STT dynamics
    stt = SpinTransferTorque(free_layer)
    torque = stt.calculate_slonczewski_torque(current_density, m_pinned)
"""

from .spin_transport import (
    # Enums
    MagnetizationDirection,
    
    # Data classes
    MagneticLayer,
    TunnelBarrier,
    
    # Main classes
    MagneticTunnelJunction,
    SpinTransferTorque,
    SpinHallEffect,
    NonLocalSpinValve,
    SpinOrbitTorque,
)

__all__ = [
    'MagnetizationDirection',
    'MagneticLayer',
    'TunnelBarrier',
    'MagneticTunnelJunction',
    'SpinTransferTorque',
    'SpinHallEffect',
    'NonLocalSpinValve',
    'SpinOrbitTorque',
]

__version__ = "1.0.0"
