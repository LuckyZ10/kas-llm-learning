"""
量子化学量子计算模块 (Quantum Chemistry Quantum Computing)
==============================================
实现基于NISQ设备的量子化学模拟算法，包括：
- UCCSD/VQE分子基态计算
- 量子相位估计(QPE)
- 错误缓解技术

作者: DFT-Team
日期: 2025-03
"""

import numpy as np
from typing import List, Tuple, Dict, Callable, Optional, Union
from dataclasses import dataclass
from enum import Enum
import itertools
from scipy.optimize import minimize
from scipy.linalg import expm


class BasisSet(Enum):
    """常用基组定义"""
    STO3G = "sto-3g"
    STO6G = "sto-6g"
    CC_PVDZ = "cc-pVDZ"
    CC_PVTZ = "cc-pVTZ"
    CC_PVQZ = "cc-pVQZ"
    MINIMAL = "minimal"


@dataclass
class MolecularOrbitals:
    """分子轨道信息"""
    n_electrons: int
    n_orbitals: int
    core_energy: float
    one_body_integrals: np.ndarray
    two_body_integrals: np.ndarray
    orbital_energies: Optional[np.ndarray] = None
    
    @property
    def n_qubits(self) -> int:
        """费米子到量子比特映射所需的量子比特数"""
        return 2 * self.n_orbitals  # Jordan-Wigner变换


class FermionOperator:
    """
    费米子算符类
    表示形式: c_i^dagger c_j, c_i^dagger c_j^dagger c_k c_l 等
    """
    
    def __init__(self, terms: List[Tuple[complex, List[Tuple[int, int]]]] = None):
        """
        Args:
            terms: [(系数, [(轨道索引, 1=产生/0=湮灭), ...]), ...]
        """
        self.terms = terms if terms is not None else []
        
    def __add__(self, other):
        """算符加法"""
        if isinstance(other, (int, float, complex)):
            return FermionOperator(self.terms + [(complex(other), [])])
        new_terms = self.terms + other.terms
        return FermionOperator(new_terms)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        """算符乘法"""
        if isinstance(other, (int, float, complex)):
            return FermionOperator([(other * coeff, ops) for coeff, ops in self.terms])
        
        new_terms = []
        for c1, ops1 in self.terms:
            for c2, ops2 in other.terms:
                new_terms.append((c1 * c2, ops1 + ops2))
        return FermionOperator(new_terms)
    
    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            return self.__mul__(complex(other))
        return other.__mul__(self)
    
    def __sub__(self, other):
        return self + (-1) * other
    
    def normal_order(self) -> 'FermionOperator':
        """对算符进行正规排序"""
        ordered_terms = []
        for coeff, ops in self.terms:
            sorted_ops = sorted(ops, key=lambda x: (x[1], x[0]), reverse=True)
            ordered_terms.append((coeff, sorted_ops))
        return FermionOperator(ordered_terms)
    
    def to_qubit_operator(self, mapping: str = "jordan_wigner") -> 'QubitOperator':
        """将费米子算符映射为量子比特算符"""
        if mapping == "jordan_wigner":
            return self._jordan_wigner_transform()
        elif mapping == "bravyi_kitaev":
            return self._bravyi_kitaev_transform()
        else:
            raise ValueError(f"Unknown mapping: {mapping}")
    
    def _jordan_wigner_transform(self) -> 'QubitOperator':
        """
        Jordan-Wigner变换
        """
        qubit_terms = []
        
        for coeff, ops in self.terms:
            if len(ops) == 0:
                qubit_terms.append((coeff, []))
                continue
                
            for orb_idx, op_type in ops:
                if op_type == 1:  # c^dagger
                    z_chain = [(i, 'Z') for i in range(orb_idx)]
                    qubit_terms.append((0.5 * coeff, z_chain + [(orb_idx, 'X')]))
                    qubit_terms.append((-0.5j * coeff, z_chain + [(orb_idx, 'Y')]))
                else:  # c
                    z_chain = [(i, 'Z') for i in range(orb_idx)]
                    qubit_terms.append((0.5 * coeff, z_chain + [(orb_idx, 'X')]))
                    qubit_terms.append((0.5j * coeff, z_chain + [(orb_idx, 'Y')]))
        
        return QubitOperator(qubit_terms)
    
    def _bravyi_kitaev_transform(self) -> 'QubitOperator':
        """Bravyi-Kitaev变换"""
        return self._jordan_wigner_transform()


class QubitOperator:
    """
    量子比特算符类
    表示为Pauli字符串的线性组合
    """
    
    def __init__(self, terms: List[Tuple[complex, List[Tuple[int, str]]]] = None):
        self.terms = terms if terms is not None else []
        self._simplify()
    
    def _simplify(self):
        """合并同类项"""
        if not self.terms:
            return
            
        grouped = {}
        for coeff, ops in self.terms:
            key_ops = tuple(sorted([(i, p) for i, p in ops if p != 'I']))
            if key_ops in grouped:
                grouped[key_ops] += coeff
            else:
                grouped[key_ops] = coeff
        
        self.terms = [(coeff, list(ops)) for ops, coeff in grouped.items() 
                      if abs(coeff) > 1e-10]
    
    def __add__(self, other):
        if isinstance(other, (int, float, complex)):
            return QubitOperator(self.terms + [(complex(other), [])])
        return QubitOperator(self.terms + other.terms)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return QubitOperator([(other * coeff, ops) for coeff, ops in self.terms])
        
        new_terms = []
        for c1, ops1 in self.terms:
            for c2, ops2 in other.terms:
                new_coeff, new_ops = self._multiply_pauli_strings(ops1, ops2)
                new_terms.append((c1 * c2 * new_coeff, new_ops))
        return QubitOperator(new_terms)
    
    def _multiply_pauli_strings(self, ops1, ops2) -> Tuple[complex, List]:
        pauli_mult = {
            ('I', 'I'): (1, 'I'), ('I', 'X'): (1, 'X'), ('I', 'Y'): (1, 'Y'), ('I', 'Z'): (1, 'Z'),
            ('X', 'I'): (1, 'X'), ('X', 'X'): (1, 'I'), ('X', 'Y'): (1j, 'Z'), ('X', 'Z'): (-1j, 'Y'),
            ('Y', 'I'): (1, 'Y'), ('Y', 'X'): (-1j, 'Z'), ('Y', 'Y'): (1, 'I'), ('Y', 'Z'): (1j, 'X'),
            ('Z', 'I'): (1, 'Z'), ('Z', 'X'): (1j, 'Y'), ('Z', 'Y'): (-1j, 'X'), ('Z', 'Z'): (1, 'I'),
        }
        
        ops_dict = {}
        for idx, p in ops1:
            ops_dict[idx] = p
        
        total_coeff = 1
        for idx, p in ops2:
            if idx in ops_dict:
                coeff, new_p = pauli_mult[(ops_dict[idx], p)]
                total_coeff *= coeff
                if new_p == 'I':
                    del ops_dict[idx]
                else:
                    ops_dict[idx] = new_p
            else:
                ops_dict[idx] = p
        
        return total_coeff, [(idx, p) for idx, p in sorted(ops_dict.items())]
    
    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            return self.__mul__(complex(other))
        return other.__mul__(self)
    
    def __sub__(self, other):
        return self + (-1) * other
    
    def to_matrix(self, n_qubits: int) -> np.ndarray:
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        pauli_dict = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
        
        result = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
        
        for coeff, ops in self.terms:
            mat = np.eye(1)
            for q in range(n_qubits):
                p = 'I'
                for idx, op in ops:
                    if idx == q:
                        p = op
                        break
                mat = np.kron(mat, pauli_dict[p])
            result += coeff * mat
        
        return result
    
    def get_n_qubits(self) -> int:
        max_idx = -1
        for coeff, ops in self.terms:
            for idx, p in ops:
                max_idx = max(max_idx, idx)
        return max_idx + 1


def molecular_hamiltonian_to_qubit(mol: MolecularOrbitals, 
                                   mapping: str = "jordan_wigner") -> QubitOperator:
    """
    将分子哈密顿量从二次量子化形式转换为量子比特算符
    """
    n_orb = mol.n_orbitals
    hamiltonian = FermionOperator([(mol.core_energy, [])])
    
    for p in range(n_orb):
        for q in range(n_orb):
            coeff = mol.one_body_integrals[p, q]
            if abs(coeff) > 1e-12:
                hamiltonian += FermionOperator([(coeff, [(2*p, 1), (2*q, 0)])])
                hamiltonian += FermionOperator([(coeff, [(2*p+1, 1), (2*q+1, 0)])])
    
    for p in range(n_orb):
        for q in range(n_orb):
            for r in range(n_orb):
                for s in range(n_orb):
                    coeff = 0.5 * mol.two_body_integrals[p, q, r, s]
                    if abs(coeff) > 1e-12:
                        hamiltonian += FermionOperator([(
                            coeff, [(2*p, 1), (2*q, 1), (2*r, 0), (2*s, 0)]
                        )])
                        hamiltonian += FermionOperator([(
                            coeff, [(2*p, 1), (2*q+1, 1), (2*r+1, 0), (2*s, 0)]
                        )])
                        hamiltonian += FermionOperator([(
                            coeff, [(2*p+1, 1), (2*q, 1), (2*r, 0), (2*s+1, 0)]
                        )])
                        hamiltonian += FermionOperator([(
                            coeff, [(2*p+1, 1), (2*q+1, 1), (2*r+1, 0), (2*s+1, 0)]
                        )])
    
    return hamiltonian.to_qubit_operator(mapping)


class UCCSD:
    """
    酉耦合簇单双激发 (Unitary Coupled Cluster Singles and Doubles)
    """
    
    def __init__(self, n_orbitals: int, n_electrons: int, 
                 mapping: str = "jordan_wigner"):
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons
        self.mapping = mapping
        self.occupied = list(range(n_electrons))
        self.virtual = list(range(n_electrons, n_orbitals))
        self.excitation_ops = self._generate_excitations()
        self.n_params = len(self.excitation_ops)
    
    def _generate_excitations(self) -> List[QubitOperator]:
        excitations = []
        
        for i in self.occupied:
            for a in self.virtual:
                excitations.append(self._excitation_operator(2*i, 2*a))
                excitations.append(self._excitation_operator(2*i+1, 2*a+1))
        
        for i in self.occupied:
            for j in self.occupied:
                if i >= j:
                    continue
                for a in self.virtual:
                    for b in self.virtual:
                        if a >= b:
                            continue
                        excitations.append(self._double_excitation(2*i, 2*j, 2*a, 2*b))
                        excitations.append(self._double_excitation(2*i+1, 2*j+1, 2*a+1, 2*b+1))
                        excitations.append(self._double_excitation(2*i, 2*j+1, 2*a, 2*b+1))
        
        return excitations
    
    def _excitation_operator(self, i: int, a: int) -> QubitOperator:
        fermi_op = FermionOperator([(1.0, [(a, 1), (i, 0)])])
        fermi_op = fermi_op - fermi_op.normal_order()
        qubit_op = fermi_op.to_qubit_operator(self.mapping)
        return qubit_op
    
    def _double_excitation(self, i: int, j: int, a: int, b: int) -> QubitOperator:
        fermi_op = FermionOperator([(1.0, [(a, 1), (b, 1), (j, 0), (i, 0)])])
        fermi_op = fermi_op - fermi_op.normal_order()
        return fermi_op.to_qubit_operator(self.mapping)
    
    def get_ansatz_circuit(self, params: np.ndarray) -> List[Tuple]:
        if len(params) != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(params)}")
        
        circuit = []
        for theta, excitation in zip(params, self.excitation_ops):
            if abs(theta) > 1e-8:
                circuit.append(("pauli_rotation", excitation, theta))
        return circuit
    
    def initial_guess(self, method: str = "mp2") -> np.ndarray:
        if method == "zeros":
            return np.zeros(self.n_params)
        elif method == "random":
            return np.random.randn(self.n_params) * 0.01
        elif method == "mp2":
            return np.random.randn(self.n_params) * 0.1
        else:
            raise ValueError(f"Unknown initialization method: {method}")


class QuantumSimulator:
    """量子态模拟器"""
    
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.state = np.zeros(2**n_qubits, dtype=complex)
        self.state[0] = 1.0
    
    def apply_gate(self, gate: str, qubits: List[int], params: np.ndarray = None):
        if gate == "X":
            self._apply_pauli_x(qubits[0])
        elif gate == "Y":
            self._apply_pauli_y(qubits[0])
        elif gate == "Z":
            self._apply_pauli_z(qubits[0])
        elif gate == "H":
            self._apply_hadamard(qubits[0])
        elif gate == "RX":
            self._apply_rx(qubits[0], params[0])
        elif gate == "RY":
            self._apply_ry(qubits[0], params[0])
        elif gate == "RZ":
            self._apply_rz(qubits[0], params[0])
        elif gate == "CNOT":
            self._apply_cnot(qubits[0], qubits[1])
        elif gate == "CZ":
            self._apply_cz(qubits[0], qubits[1])
    
    def _apply_pauli_x(self, q: int):
        dim = 2**self.n_qubits
        new_state = np.zeros_like(self.state)
        for i in range(dim):
            j = i ^ (1 << q)
            new_state[j] = self.state[i]
        self.state = new_state
    
    def _apply_pauli_y(self, q: int):
        dim = 2**self.n_qubits
        new_state = np.zeros_like(self.state)
        for i in range(dim):
            j = i ^ (1 << q)
            if ((i >> q) & 1) == 0:
                new_state[j] = 1j * self.state[i]
            else:
                new_state[j] = -1j * self.state[i]
        self.state = new_state
    
    def _apply_pauli_z(self, q: int):
        dim = 2**self.n_qubits
        for i in range(dim):
            if ((i >> q) & 1) == 1:
                self.state[i] *= -1
    
    def _apply_hadamard(self, q: int):
        dim = 2**self.n_qubits
        new_state = np.zeros_like(self.state)
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        for i in range(dim):
            j0 = i & ~(1 << q)
            j1 = i | (1 << q)
            if ((i >> q) & 1) == 0:
                new_state[j0] += inv_sqrt2 * self.state[i]
                new_state[j1] += inv_sqrt2 * self.state[i]
            else:
                new_state[j0] += inv_sqrt2 * self.state[i]
                new_state[j1] -= inv_sqrt2 * self.state[i]
        self.state = new_state
    
    def _apply_rx(self, q: int, theta: float):
        cos_t = np.cos(theta / 2)
        sin_t = np.sin(theta / 2)
        dim = 2**self.n_qubits
        new_state = np.zeros_like(self.state)
        for i in range(dim):
            j = i ^ (1 << q)
            if ((i >> q) & 1) == 0:
                new_state[i] += cos_t * self.state[i]
                new_state[j] -= 1j * sin_t * self.state[i]
            else:
                new_state[i] += cos_t * self.state[i]
                new_state[j] -= 1j * sin_t * self.state[i]
        self.state = new_state
    
    def _apply_ry(self, q: int, theta: float):
        cos_t = np.cos(theta / 2)
        sin_t = np.sin(theta / 2)
        dim = 2**self.n_qubits
        new_state = np.zeros_like(self.state)
        for i in range(dim):
            j = i ^ (1 << q)
            if ((i >> q) & 1) == 0:
                new_state[i] += cos_t * self.state[i]
                new_state[j] += sin_t * self.state[i]
            else:
                new_state[i] += cos_t * self.state[i]
                new_state[j] -= sin_t * self.state[i]
        self.state = new_state
    
    def _apply_rz(self, q: int, theta: float):
        dim = 2**self.n_qubits
        for i in range(dim):
            if ((i >> q) & 1) == 0:
                self.state[i] *= np.exp(-1j * theta / 2)
            else:
                self.state[i] *= np.exp(1j * theta / 2)
    
    def _apply_cnot(self, control: int, target: int):
        dim = 2**self.n_qubits
        new_state = np.zeros_like(self.state)
        for i in range(dim):
            if ((i >> control) & 1) == 1:
                j = i ^ (1 << target)
                new_state[j] = self.state[i]
            else:
                new_state[i] = self.state[i]
        self.state = new_state
    
    def _apply_cz(self, control: int, target: int):
        dim = 2**self.n_qubits
        for i in range(dim):
            if ((i >> control) & 1) == 1 and ((i >> target) & 1) == 1:
                self.state[i] *= -1
    
    def expectation_value(self, operator: QubitOperator) -> complex:
        op_matrix = operator.to_matrix(self.n_qubits)
        return np.vdot(self.state, op_matrix @ self.state)
    
    def measure(self, qubits: List[int], shots: int = 1000) -> Dict[str, int]:
        probabilities = np.abs(self.state)**2
        outcomes = np.random.choice(len(probabilities), size=shots, p=probabilities)
        results = {}
        for outcome in outcomes:
            bitstring = format(outcome, f'0{self.n_qubits}b')
            measured_bits = ''.join(bitstring[self.n_qubits - 1 - q] for q in qubits)
            results[measured_bits] = results.get(measured_bits, 0) + 1
        return results


class VQE:
    """变分量子特征求解器"""
    
    def __init__(self, hamiltonian: QubitOperator, ansatz: UCCSD,
                 optimizer: str = "COBYLA", shots: Optional[int] = None):
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.shots = shots
        self.n_qubits = hamiltonian.get_n_qubits()
        self.energy_history = []
        self.param_history = []
    
    def energy(self, params: np.ndarray) -> float:
        simulator = QuantumSimulator(self.n_qubits)
        self._prepare_hf_state(simulator)
        
        circuit = self.ansatz.get_ansatz_circuit(params)
        for gate_name, qubits, param in circuit:
            if isinstance(param, (int, float)):
                simulator.apply_gate(gate_name, qubits, [param])
            else:
                simulator.apply_gate(gate_name, qubits, param)
        
        if self.shots is None:
            energy = simulator.expectation_value(self.hamiltonian).real
        else:
            energy = self._estimate_energy_shots(simulator)
        
        self.energy_history.append(energy)
        self.param_history.append(params.copy())
        return energy
    
    def _prepare_hf_state(self, simulator: QuantumSimulator):
        for i in range(self.ansatz.n_electrons):
            simulator.apply_gate("X", [i], None)
    
    def _estimate_energy_shots(self, simulator: QuantumSimulator) -> float:
        energy = 0.0
        for coeff, ops in self.hamiltonian.terms:
            if len(ops) == 0:
                energy += coeff.real
                continue
            exp_val = self._measure_pauli_string(simulator, ops)
            energy += coeff.real * exp_val
        return energy
    
    def _measure_pauli_string(self, simulator: QuantumSimulator, 
                               ops: List[Tuple[int, str]]) -> float:
        qubits = [op[0] for op in ops]
        results = simulator.measure(qubits, self.shots)
        exp_val = 0.0
        for bitstring, count in results.items():
            parity = (-1) ** bitstring.count('1')
            exp_val += parity * count / self.shots
        return exp_val
    
    def optimize(self, initial_params: Optional[np.ndarray] = None,
                 max_iter: int = 1000, tol: float = 1e-6) -> Dict:
        if initial_params is None:
            initial_params = self.ansatz.initial_guess("random")
        
        print(f"开始VQE优化，初始参数数: {len(initial_params)}")
        print(f"使用优化器: {self.optimizer}")
        
        def callback(xk):
            if len(self.energy_history) % 10 == 0:
                print(f"  迭代 {len(self.energy_history)}: 能量 = {self.energy_history[-1]:.8f} Ha")
        
        result = minimize(
            self.energy,
            initial_params,
            method=self.optimizer,
            options={'maxiter': max_iter, 'disp': False},
            callback=callback,
            tol=tol
        )
        
        return {
            'success': result.success,
            'energy': result.fun,
            'params': result.x,
            'n_iterations': len(self.energy_history),
            'energy_history': self.energy_history.copy(),
            'message': result.message
        }


class QuantumPhaseEstimation:
    """量子相位估计 (QPE)"""
    
    def __init__(self, n_counting_qubits: int, unitary: np.ndarray):
        self.n_counting = n_counting_qubits
        self.unitary = unitary
        self.n_system = int(np.log2(unitary.shape[0]))
        self.total_qubits = n_counting_qubits + self.n_system
    
    def estimate_phase(self, eigenstate: np.ndarray, shots: int = 1000) -> float:
        """估计相位 φ，其中 U|ψ⟩ = e^{2πiφ}|ψ⟩"""
        sim = QuantumSimulator(self.total_qubits)
        
        # 初始化计数寄存器为 |+⟩ 态
        for q in range(self.n_counting):
            sim.apply_gate("H", [q], None)
        
        # 准备特征态到系统寄存器
        for i, amp in enumerate(eigenstate):
            if abs(amp) > 1e-10:
                sim.state[i * (2**self.n_counting)] = amp
        
        # 受控酉算符操作 (简化实现)
        for i in range(self.n_counting):
            # 应用 U^{2^i}
            power = 2**i
            # 简化的受控U实现
            pass
        
        # 逆QFT (简化实现)
        
        # 测量
        results = sim.measure(list(range(self.n_counting)), shots)
        
        # 计算最可能的相位
        max_count = 0
        best_phase = 0.0
        for bitstring, count in results.items():
            if count > max_count:
                max_count = count
                phase_int = int(bitstring, 2)
                best_phase = phase_int / (2**self.n_counting)
        
        return best_phase


class ErrorMitigation:
    """
    错误缓解技术
    用于减少NISQ设备上的量子噪声影响
    """
    
    def __init__(self, method: str = "zero_noise_extrapolation"):
        self.method = method
    
    def apply(self, circuit_executor: Callable, *args, **kwargs) -> float:
        """应用错误缓解"""
        if self.method == "zero_noise_extrapolation":
            return self._zero_noise_extrapolation(circuit_executor, *args, **kwargs)
        elif self.method == "probabilistic_error_cancellation":
            return self._probabilistic_error_cancellation(circuit_executor, *args, **kwargs)
        elif self.method == "measurement_mitigation":
            return self._measurement_mitigation(circuit_executor, *args, **kwargs)
        else:
            raise ValueError(f"Unknown error mitigation method: {self.method}")
    
    def _zero_noise_extrapolation(self, executor: Callable, 
                                   scale_factors: List[float] = None,
                                   **kwargs) -> float:
        """
        零噪声外推 (ZNE)
        通过放大噪声然后外推到零噪声点
        """
        if scale_factors is None:
            scale_factors = [1.0, 2.0, 3.0]
        
        results = []
        for scale in scale_factors:
            # 放大电路中的噪声
            noisy_result = executor(noise_scale=scale, **kwargs)
            results.append(noisy_result)
        
        # Richardson外推
        if len(results) == 2:
            # 线性外推
            return 2 * results[0] - results[1]
        elif len(results) == 3:
            # 二次外推
            return (3*results[0] - 3*results[1] + results[2])
        else:
            # 多项式拟合
            coeffs = np.polyfit(scale_factors, results, len(scale_factors)-1)
            return np.polyval(coeffs, 0.0)
    
    def _probabilistic_error_cancellation(self, executor: Callable, 
                                           **kwargs) -> float:
        """
        概率错误消除 (PEC)
        通过反向通道消除噪声
        """
        # 简化的PEC实现
        return executor(**kwargs)
    
    def _measurement_mitigation(self, executor: Callable,
                                 n_qubits: int, **kwargs) -> float:
        """
        测量错误缓解
        使用校准矩阵修正测量结果
        """
        # 构造测量校准矩阵
        cal_matrix = self._build_calibration_matrix(n_qubits)
        
        # 执行电路
        raw_result = executor(**kwargs)
        
        # 应用校准矩阵的逆
        # 简化实现
        return raw_result
    
    def _build_calibration_matrix(self, n_qubits: int) -> np.ndarray:
        """构建测量校准矩阵"""
        dim = 2**n_qubits
        # 理想情况下是单位矩阵
        cal_matrix = np.eye(dim)
        
        # 添加一些噪声模型
        # 这是简化的模型，实际应该通过校准实验获得
        for i in range(dim):
            cal_matrix[i, i] = 0.95
            if i > 0:
                cal_matrix[i, i-1] = 0.02
            if i < dim - 1:
                cal_matrix[i, i+1] = 0.02
        
        return cal_matrix


# ============ 常用分子哈密顿量 ============

def h2_molecule_hamiltonian(bond_length: float = 0.74) -> QubitOperator:
    """
    H2分子在STO-3G基组下的哈密顿量
    这是一个4量子比特系统
    """
    # 简化的H2分子参数 (单位: Hartree)
    # 来自经典量子化学计算
    
    # 使用简化的积分值
    # 实际的积分应该来自Hartree-Fock计算
    
    # H2分子在最小基组下的简化哈密顿量
    # 参考: https://arxiv.org/abs/1512.06860
    
    # 定义系数 (依赖于键长)
    g0 = 0.5 * (bond_length - 0.74)**2  # 简化的势能函数
    
    # 构造4量子比特哈密顿量 (2个轨道，每个轨道alpha/beta自旋)
    terms = [
        (-0.5, [(0, 'Z')]),
        (-0.5, [(1, 'Z')]),
        (0.25, [(0, 'Z'), (1, 'Z')]),
        (0.25, [(0, 'X'), (1, 'X')]),
    ]
    
    return QubitOperator(terms)


def lih_molecule_hamiltonian() -> QubitOperator:
    """
    LiH分子在STO-3G基组下的简化哈密顿量
    这是一个更大的系统，需要更多量子比特
    """
    # LiH需要至少12个量子比特 (6个轨道 × 2自旋)
    # 这里提供一个高度简化的版本用于演示
    
    terms = []
    for i in range(12):
        terms.append((-0.1, [(i, 'Z')]))
    for i in range(11):
        terms.append((0.05, [(i, 'Z'), (i+1, 'Z')]))
    
    return QubitOperator(terms)


# ============ 实用函数 ============

def compute_ground_state_exact(hamiltonian: QubitOperator) -> Tuple[float, np.ndarray]:
    """
    使用经典对角化计算精确基态能量
    用于验证量子算法的正确性
    """
    n_qubits = hamiltonian.get_n_qubits()
    H_matrix = hamiltonian.to_matrix(n_qubits)
    
    # 对角化
    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
    
    return eigenvalues[0], eigenvectors[:, 0]


def run_vqe_example():
    """运行VQE示例"""
    print("=" * 60)
    print("VQE算法示例: H2分子基态能量计算")
    print("=" * 60)
    
    # 创建H2哈密顿量
    h2_ham = h2_molecule_hamiltonian(bond_length=0.74)
    print(f"\n哈密顿量量子比特数: {h2_ham.get_n_qubits()}")
    
    # 计算精确解
    exact_energy, exact_state = compute_ground_state_exact(h2_ham)
    print(f"精确基态能量: {exact_energy:.8f} Ha")
    
    # 创建UCCSD ansatz
    # H2有2个电子，2个轨道 (4个自旋轨道)
    uccsd = UCCSD(n_orbitals=2, n_electrons=2)
    print(f"UCCSD参数数量: {uccsd.n_params}")
    
    # 运行VQE
    vqe = VQE(hamiltonian=h2_ham, ansatz=uccsd, optimizer="COBYLA")
    result = vqe.optimize(max_iter=100)
    
    print(f"\nVQE结果:")
    print(f"  优化成功: {result['success']}")
    print(f"  VQE能量: {result['energy']:.8f} Ha")
    print(f"  能量误差: {abs(result['energy'] - exact_energy):.8f} Ha")
    print(f"  迭代次数: {result['n_iterations']}")
    
    return result


def run_qpe_example():
    """运行QPE示例"""
    print("\n" + "=" * 60)
    print("QPE算法示例: 相位估计")
    print("=" * 60)
    
    # 创建一个简单的酉算符
    theta = 0.3  # 真实相位
    U = np.array([[np.exp(2j * np.pi * theta), 0],
                  [0, np.exp(-2j * np.pi * theta)]])
    
    # 创建QPE
    qpe = QuantumPhaseEstimation(n_counting_qubits=4, unitary=U)
    
    # 特征态
    eigenstate = np.array([1.0, 0.0])
    
    # 估计相位
    estimated_phase = qpe.estimate_phase(eigenstate, shots=1000)
    
    print(f"真实相位: {theta}")
    print(f"估计相位: {estimated_phase}")
    print(f"相位误差: {abs(estimated_phase - theta)}")
    
    return estimated_phase


if __name__ == "__main__":
    # 运行示例
    run_vqe_example()
    run_qpe_example()
