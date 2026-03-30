#!/usr/bin/env python3
"""
machine_limits/__init__.py - Machine limits module for extreme-scale computing

Provides optimizations for million-core parallelism, memory management,
and checkpoint/restart mechanisms.

Author: DFT-LAMMPS Team
Version: 1.0.0
"""

from .parallel_optimization import (
    MillionCoreOptimizer,
    CommunicationPattern,
    LoadBalancer,
    TopologyAwareMapping
)

from .memory_optimization import (
    MemoryManager,
    OutOfCoreArray,
    CompressionManager,
    MemoryPool
)

from .checkpoint_restart import (
    CheckpointManager,
    IncrementalCheckpoint,
    FaultToleranceManager
)

__all__ = [
    'MillionCoreOptimizer',
    'CommunicationPattern', 
    'LoadBalancer',
    'TopologyAwareMapping',
    'MemoryManager',
    'OutOfCoreArray',
    'CompressionManager',
    'MemoryPool',
    'CheckpointManager',
    'IncrementalCheckpoint',
    'FaultToleranceManager'
]
