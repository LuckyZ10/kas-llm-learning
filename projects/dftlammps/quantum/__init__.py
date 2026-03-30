"""
DFT-LAMMPS 量子计算模块

提供量子-经典混合计算功能，包括：
- 量子电路接口（支持Qiskit、Cirq、PennyLane）
- VQE求解器（分子电子结构）
- 量子机器学习势
- 量子动力学与经典MD耦合
"""

from .quantum_interface import (
    QuantumInterface,
    QuantumCircuitBase,
    QuantumBackend,
    QuantumDevice,
    create_quantum_interface,
    build_ansatz_circuit,
    build_hartree_fock_circuit,
    build_uccsd_ansatz,
    QiskitCircuit,
    CirqCircuit,
    PennyLaneCircuit,
)

from .vqe_solver import (
    VQESolver,
    MolecularHamiltonian,
    VQECallback,
    run_vqe_for_molecule,
    compare_classical_vqe,
)

from .quantum_ml import (
    QuantumFeatureMap,
    QuantumKernel,
    QuantumNeuralNetwork,
    QuantumKernelRidge,
    QuantumGaussianProcess,
    QuantumPotentialEnergySurface,
)

from .quantum_dynamics import (
    QuantumRegion,
    ClassicalRegion,
    QuantumClassicalPartition,
    QMMCoupling,
    MechanicalEmbedding,
    ElectrostaticEmbedding,
    QuantumDynamics,
    HybridQMMD,
)

__version__ = "0.1.0"

__all__ = [
    # Quantum Interface
    "QuantumInterface",
    "QuantumCircuitBase", 
    "QuantumBackend",
    "QuantumDevice",
    "create_quantum_interface",
    "build_ansatz_circuit",
    "build_hartree_fock_circuit",
    "build_uccsd_ansatz",
    "QiskitCircuit",
    "CirqCircuit",
    "PennyLaneCircuit",
    
    # VQE
    "VQESolver",
    "MolecularHamiltonian",
    "VQECallback",
    "run_vqe_for_molecule",
    "compare_classical_vqe",
    
    # Quantum ML
    "QuantumFeatureMap",
    "QuantumKernel",
    "QuantumNeuralNetwork",
    "QuantumKernelRidge",
    "QuantumGaussianProcess",
    "QuantumPotentialEnergySurface",
    
    # Quantum Dynamics
    "QuantumRegion",
    "ClassicalRegion",
    "QuantumClassicalPartition",
    "QMMCoupling",
    "MechanicalEmbedding",
    "ElectrostaticEmbedding",
    "QuantumDynamics",
    "HybridQMMD",
]
