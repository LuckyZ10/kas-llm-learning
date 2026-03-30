"""
量子信息模块 - DFT-LAMMPS量子计算扩展
=====================================

本模块提供量子计算和量子材料模拟功能：

子模块:
--------
- quantum_chemistry_qc: 量子化学量子计算
    - VQE/UCCSD分子基态计算
    - 量子相位估计(QPE)
    - 错误缓解技术

- quantum_materials: 量子材料模拟
    - Hubbard模型
    - 自旋系统(Heisenberg, Ising, XXZ)
    - 拓扑不变量计算

示例:
------
- examples/htc_superconductor.py: 高温超导模拟
- examples/magnetic_phase_diagram.py: 磁性相图
- examples/topological_qc.py: 拓扑量子计算

使用示例:
---------
>>> from dftlammps.quantum_info.quantum_chemistry_qc import VQE, UCCSD
>>> from dftlammps.quantum_info.quantum_materials import HubbardModel, Lattice

作者: DFT-Team
版本: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "DFT-Team"

from .quantum_chemistry_qc import (
    VQE,
    UCCSD,
    QuantumPhaseEstimation,
    FermionOperator,
    QubitOperator,
    MolecularOrbitals,
    ErrorMitigation,
    QuantumSimulator,
    molecular_hamiltonian_to_qubit,
    h2_molecule_hamiltonian,
    compute_ground_state_exact,
    run_vqe_example,
    run_qpe_example
)

from .quantum_materials import (
    Lattice,
    LatticeType,
    HubbardModel,
    SpinSystem,
    TopologicalInvariant,
    HaldaneModel,
    KitaevModel,
    hubbard_phase_diagram,
    spin_phase_diagram,
    run_hubbard_example,
    run_spin_example,
    run_topological_example
)

__all__ = [
    # 量子化学
    'VQE',
    'UCCSD', 
    'QuantumPhaseEstimation',
    'FermionOperator',
    'QubitOperator',
    'MolecularOrbitals',
    'ErrorMitigation',
    'QuantumSimulator',
    'molecular_hamiltonian_to_qubit',
    'h2_molecule_hamiltonian',
    'compute_ground_state_exact',
    
    # 量子材料
    'Lattice',
    'LatticeType',
    'HubbardModel',
    'SpinSystem',
    'TopologicalInvariant',
    'HaldaneModel',
    'KitaevModel',
    'hubbard_phase_diagram',
    'spin_phase_diagram',
]
