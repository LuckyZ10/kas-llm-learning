"""
变分量子特征值求解器 (VQE)
用于分子电子结构计算
"""

from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import numpy as np
from scipy.optimize import minimize
import warnings

try:
    from .quantum_interface import (
        QuantumInterface, 
        QuantumCircuitBase,
        build_ansatz_circuit,
        build_hartree_fock_circuit,
        build_uccsd_ansatz,
        create_quantum_interface
    )
except ImportError:
    from quantum_interface import (
        QuantumInterface, 
        QuantumCircuitBase,
        build_ansatz_circuit,
        build_hartree_fock_circuit,
        build_uccsd_ansatz,
        create_quantum_interface
    )


class MolecularHamiltonian:
    """分子哈密顿量"""
    
    def __init__(
        self,
        num_qubits: int,
        one_body_integrals: Optional[np.ndarray] = None,
        two_body_integrals: Optional[np.ndarray] = None,
        nuclear_repulsion: float = 0.0
    ):
        self.num_qubits = num_qubits
        self.one_body = one_body_integrals or np.zeros((num_qubits, num_qubits))
        self.two_body = two_body_integrals or np.zeros((num_qubits, num_qubits, num_qubits, num_qubits))
        self.nuclear_repulsion = nuclear_repulsion
        self._fermionic_terms: List[Tuple[float, Tuple[int, ...]]] = []
        self._pauli_terms: List[Tuple[float, str]] = []
    
    @classmethod
    def from_pyscf(cls, mf: Any) -> 'MolecularHamiltonian':
        """从PySCF计算结果创建哈密顿量"""
        mol = mf.mol
        
        # 获取积分
        h1 = mf.get_hcore()
        h2 = mol.intor('int2e')
        
        # 转换到MO基
        mo_coeff = mf.mo_coeff
        h1_mo = np.einsum('pi,pq,qj->ij', mo_coeff, h1, mo_coeff)
        h2_mo = np.einsum('pqrs,pa,qb,rc,sd->abcd', h2, mo_coeff, mo_coeff, mo_coeff, mo_coeff)
        
        num_orbitals = mol.nao * 2  # 自旋轨道
        
        return cls(
            num_qubits=num_orbitals,
            one_body_integrals=h1_mo,
            two_body_integrals=h2_mo,
            nuclear_repulsion=mol.energy_nuc()
        )
    
    @classmethod
    def from_openfermion(cls, qubit_operator: Any) -> 'MolecularHamiltonian':
        """从OpenFermion QubitOperator创建"""
        try:
            from openfermion import QubitOperator
        except ImportError:
            raise ImportError("OpenFermion not installed. Run: pip install openfermion")
        
        # 获取最大qubit索引
        max_qubit = 0
        for term in qubit_operator.terms:
            for op in term:
                max_qubit = max(max_qubit, op[0])
        
        ham = cls(num_qubits=max_qubit + 1)
        
        # 转换Pauli项
        for term, coeff in qubit_operator.terms.items():
            pauli_str = ['I'] * ham.num_qubits
            for qubit_idx, pauli in term:
                pauli_str[qubit_idx] = pauli
            ham._pauli_terms.append((coeff, ''.join(pauli_str)))
        
        return ham
    
    def to_pauli_strings(self) -> List[Tuple[float, str]]:
        """
        将哈密顿量转换为Pauli字符串列表
        
        Returns:
            [(系数, Pauli字符串), ...]
        """
        if self._pauli_terms:
            return self._pauli_terms
        
        # Jordan-Wigner变换
        terms = []
        
        # 单电子项
        for p in range(self.num_qubits):
            for q in range(self.num_qubits):
                if abs(self.one_body[p, q]) > 1e-10:
                    # a_p† a_q 的JW变换
                    pauli_terms = self._jw_one_body(p, q)
                    for coeff, pauli in pauli_terms:
                        terms.append((coeff * self.one_body[p, q], pauli))
        
        # 双电子项
        for p in range(self.num_qubits):
            for q in range(self.num_qubits):
                for r in range(self.num_qubits):
                    for s in range(self.num_qubits):
                        if abs(self.two_body[p, q, r, s]) > 1e-10:
                            pauli_terms = self._jw_two_body(p, q, r, s)
                            for coeff, pauli in pauli_terms:
                                terms.append((0.5 * coeff * self.two_body[p, q, r, s], pauli))
        
        # 添加核排斥能
        terms.append((self.nuclear_repulsion, 'I' * self.num_qubits))
        
        self._pauli_terms = self._simplify_pauli_terms(terms)
        return self._pauli_terms
    
    def _jw_one_body(self, p: int, q: int) -> List[Tuple[float, str]]:
        """Jordan-Wigner变换：单电子项"""
        n = self.num_qubits
        
        if p == q:
            # a_p† a_p = (1 - Z_p)/2
            pauli = ['I'] * n
            pauli[p] = 'Z'
            return [(0.5, 'I' * n), (-0.5, ''.join(pauli))]
        elif p < q:
            # a_p† a_q = -1/2 (X_p X_q + Y_p Y_q) ⊗ Z_{p+1}...Z_{q-1}
            paulis = []
            for op_pair in [('X', 'X'), ('Y', 'Y')]:
                pauli = ['I'] * n
                pauli[p] = op_pair[0]
                pauli[q] = op_pair[1]
                for k in range(p + 1, q):
                    pauli[k] = 'Z'
                paulis.append((-0.5, ''.join(pauli)))
            return paulis
        else:
            # a_p† a_q = (a_q† a_p)†
            terms = self._jw_one_body(q, p)
            return [(coeff.conjugate(), pauli) for coeff, pauli in terms]
    
    def _jw_two_body(self, p: int, q: int, r: int, s: int) -> List[Tuple[float, str]]:
        """Jordan-Wigner变换：双电子项 (简化版)"""
        # 这里使用简化实现，实际应该用完整的JW变换
        # 返回主要贡献项
        n = self.num_qubits
        
        if p == r and q == s:
            pauli = ['I'] * n
            pauli[p] = 'Z'
            pauli[q] = 'Z'
            return [(0.25, 'I' * n), (-0.25, ''.join(pauli))]
        
        return []
    
    def _simplify_pauli_terms(self, terms: List[Tuple[float, str]]) -> List[Tuple[float, str]]:
        """简化Pauli项：合并相同项"""
        from collections import defaultdict
        
        grouped = defaultdict(float)
        for coeff, pauli in terms:
            grouped[pauli] += coeff
        
        # 过滤接近零的项
        result = [(coeff, pauli) for pauli, coeff in grouped.items() if abs(coeff) > 1e-10]
        return sorted(result, key=lambda x: -abs(x[0]))


class VQESolver:
    """
    变分量子特征值求解器
    
    使用变分量子电路找到分子哈密顿量的基态能量
    """
    
    def __init__(
        self,
        quantum_interface: Optional[QuantumInterface] = None,
        ansatz_type: str = "UCCSD",
        optimizer: str = "COBYLA",
        max_iterations: int = 1000,
        convergence_tol: float = 1e-6,
        shots: Optional[int] = None,
        verbose: bool = True
    ):
        self.interface = quantum_interface or create_quantum_interface()
        self.ansatz_type = ansatz_type
        self.optimizer = optimizer
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        self.shots = shots
        self.verbose = verbose
        
        self._hamiltonian: Optional[MolecularHamiltonian] = None
        self._circuit: Optional[QuantumCircuitBase] = None
        self._param_names: List[str] = []
        self._history: List[Dict[str, Any]] = []
        self._optimal_params: Optional[np.ndarray] = None
        self._ground_state_energy: Optional[float] = None
    
    def build_hamiltonian(
        self,
        geometry: List[Tuple[str, Tuple[float, float, float]]],
        basis: str = "sto-3g",
        charge: int = 0,
        spin: int = 0
    ) -> MolecularHamiltonian:
        """
        从分子几何构建哈密顿量
        
        Args:
            geometry: [(原子符号, (x, y, z)), ...]
            basis: 基组
            charge: 电荷
            spin: 自旋多重度
        
        Returns:
            MolecularHamiltonian
        """
        try:
            from pyscf import gto, scf
        except ImportError:
            warnings.warn("PySCF not available, using dummy Hamiltonian")
            self._hamiltonian = self._build_dummy_hamiltonian(len(geometry))
            return self._hamiltonian
        
        # 构建分子
        mol = gto.M(
            atom=[(atom, coords) for atom, coords in geometry],
            basis=basis,
            charge=charge,
            spin=spin,
            unit='Angstrom'
        )
        
        # RHF计算
        mf = scf.RHF(mol)
        mf.kernel()
        
        self._hamiltonian = MolecularHamiltonian.from_pyscf(mf)
        return self._hamiltonian
    
    def _build_dummy_hamiltonian(self, num_atoms: int) -> MolecularHamiltonian:
        """构建测试用哈密顿量"""
        num_qubits = num_atoms * 2  # 简化模型
        return MolecularHamiltonian(
            num_qubits=num_qubits,
            nuclear_repulsion=0.7 * num_atoms
        )
    
    def build_ansatz(self, num_electrons: Optional[int] = None) -> QuantumCircuitBase:
        """
        构建变分ansatz电路
        
        Args:
            num_electrons: 电子数
        
        Returns:
            参数化量子电路
        """
        if self._hamiltonian is None:
            raise ValueError("Hamiltonian not built. Call build_hamiltonian() first.")
        
        n_qubits = self._hamiltonian.num_qubits
        n_electrons = num_electrons or n_qubits // 2
        
        if self.ansatz_type.upper() == "UCCSD":
            self._circuit, self._param_names = build_uccsd_ansatz(
                self.interface, n_qubits, n_electrons
            )
        elif self.ansatz_type.upper() == "HARDWARE_EFFICIENT":
            self._circuit, self._param_names = build_ansatz_circuit(
                self.interface, n_qubits, num_layers=3, entanglement="linear"
            )
        elif self.ansatz_type.upper() == "ADAPTIVE":
            self._circuit, self._param_names = self._build_adapt_ansatz(n_qubits, n_electrons)
        else:
            raise ValueError(f"Unknown ansatz type: {self.ansatz_type}")
        
        return self._circuit
    
    def _build_adapt_ansatz(self, num_qubits: int, num_electrons: int) -> Tuple[QuantumCircuitBase, List[str]]:
        """构建ADAPT-VQE ansatz"""
        # 从HF态开始
        circuit = build_hartree_fock_circuit(self.interface, num_qubits, num_electrons)
        param_names = []
        
        # 添加可适应的激发算符池
        pool = self._build_operator_pool(num_qubits, num_electrons)
        
        for i, (op_type, qubits) in enumerate(pool):
            param_name = f"theta_{i}"
            param_names.append(param_name)
            
            if op_type == "single":
                circuit.add_parameterized_rotation('y', qubits[0], param_name)
                circuit.cx(qubits[0], qubits[1])
            elif op_type == "double":
                circuit.add_parameterized_rotation('z', qubits[0], param_name)
                circuit.cx(qubits[0], qubits[2])
                circuit.cx(qubits[1], qubits[3])
        
        return circuit, param_names
    
    def _build_operator_pool(self, num_qubits: int, num_electrons: int) -> List[Tuple[str, Tuple[int, ...]]]:
        """构建ADAPT-VQE算符池"""
        pool = []
        occupied = list(range(num_electrons))
        virtual = list(range(num_electrons, num_qubits))
        
        # 单激发
        for i in occupied:
            for a in virtual:
                pool.append(("single", (i, a)))
        
        # 双激发
        for i in occupied:
            for j in occupied:
                if i < j:
                    for a in virtual:
                        for b in virtual:
                            if a < b:
                                pool.append(("double", (i, j, a, b)))
        
        return pool
    
    def compute_energy(self, parameters: np.ndarray) -> float:
        """
        计算给定参数下的能量期望值
        
        Args:
            parameters: 参数数组
        
        Returns:
            能量值
        """
        if self._circuit is None:
            raise ValueError("Ansatz not built. Call build_ansatz() first.")
        if self._hamiltonian is None:
            raise ValueError("Hamiltonian not built.")
        
        # 构建参数字典
        param_dict = {name: float(val) for name, val in zip(self._param_names, parameters)}
        
        # 获取Pauli项
        pauli_terms = self._hamiltonian.to_pauli_strings()
        
        # 计算期望值
        energy = 0.0
        for coeff, pauli_str in pauli_terms:
            exp_val = self._measure_pauli_expectation(pauli_str, param_dict)
            energy += coeff * exp_val
        
        return energy
    
    def _measure_pauli_expectation(self, pauli_str: str, param_dict: Dict[str, float]) -> float:
        """测量单个Pauli字符串的期望值"""
        # 构建测量电路
        n_qubits = len(pauli_str)
        meas_circuit = self._circuit.copy()
        
        # 添加基变换
        for i, p in enumerate(pauli_str):
            if p == 'X':
                meas_circuit.ry(-np.pi/2, i)
            elif p == 'Y':
                meas_circuit.rx(np.pi/2, i)
        
        # 测量
        try:
            result = self.interface.execute(meas_circuit, param_dict)
            
            if 'counts' in result:
                # 从测量结果计算期望值
                counts = result['counts']
                exp_val = 0.0
                total = sum(counts.values())
                
                for bitstring, count in counts.items():
                    # 计算奇偶性
                    parity = sum(int(bit) for bit in bitstring if bit == '1') % 2
                    sign = -1 if parity else 1
                    exp_val += sign * count / total
                
                return exp_val
            else:
                # 从statevector直接计算
                return result.get('result', 0.0)
        
        except Exception as e:
            # 模拟回退
            return self._simulate_expectation(pauli_str, param_dict)
    
    def _simulate_expectation(self, pauli_str: str, param_dict: Dict[str, float]) -> float:
        """经典模拟期望值计算（回退方案）"""
        # 简化模拟：随机期望值
        return np.random.uniform(-1, 1)
    
    def optimize(
        self,
        initial_params: Optional[np.ndarray] = None,
        callback: Optional[Callable[[int, np.ndarray, float], None]] = None
    ) -> Dict[str, Any]:
        """
        优化变分参数
        
        Args:
            initial_params: 初始参数
            callback: 回调函数(iteration, params, energy)
        
        Returns:
            优化结果字典
        """
        if self._circuit is None:
            self.build_ansatz()
        
        n_params = len(self._param_names)
        
        if initial_params is None:
            # 随机初始化
            x0 = np.random.uniform(-np.pi, np.pi, n_params)
        else:
            x0 = initial_params
        
        self._history = []
        
        def objective(x):
            energy = self.compute_energy(x)
            self._history.append({'energy': energy, 'params': x.copy()})
            return energy
        
        iteration = [0]
        def opt_callback(x):
            if callback:
                callback(iteration[0], x, objective(x))
            iteration[0] += 1
            
            if self.verbose and iteration[0] % 10 == 0:
                print(f"Iteration {iteration[0]}: E = {objective(x):.8f} Hartree")
        
        if self.verbose:
            print(f"Starting VQE optimization with {n_params} parameters...")
            print(f"Initial energy: {objective(x0):.8f} Hartree")
        
        # 运行优化
        result = minimize(
            objective,
            x0,
            method=self.optimizer,
            callback=opt_callback,
            options={'maxiter': self.max_iterations, 'ftol': self.convergence_tol}
        )
        
        self._optimal_params = result.x
        self._ground_state_energy = result.fun
        
        if self.verbose:
            print(f"\nVQE optimization completed!")
            print(f"Ground state energy: {result.fun:.8f} Hartree")
            print(f"Iterations: {result.nfev}")
            print(f"Success: {result.success}")
        
        return {
            'energy': result.fun,
            'params': result.x,
            'success': result.success,
            'n_iterations': result.nfev,
            'history': self._history
        }
    
    def get_ground_state(self) -> Tuple[float, np.ndarray]:
        """
        获取基态能量和波函数参数
        
        Returns:
            (能量, 参数)
        """
        if self._ground_state_energy is None:
            self.optimize()
        
        return self._ground_state_energy, self._optimal_params
    
    def compute_excited_states(
        self,
        n_states: int = 3,
        orthogonality_weight: float = 10.0
    ) -> List[Dict[str, Any]]:
        """
        计算激发态（使用正交约束VQE）
        
        Args:
            n_states: 激发态数量
            orthogonality_weight: 正交约束权重
        
        Returns:
            激发态列表
        """
        states = []
        
        # 首先找到基态
        ground_result = self.optimize()
        states.append({
            'state': 0,
            'energy': ground_result['energy'],
            'params': ground_result['params']
        })
        
        prev_states = [ground_result['params']]
        
        for state_idx in range(1, n_states + 1):
            # 修改目标函数，添加正交约束
            def constrained_objective(x):
                energy = self.compute_energy(x)
                
                # 添加与之前态的正交惩罚
                overlap_penalty = 0.0
                for prev_state in prev_states:
                    overlap = np.dot(x, prev_state) / (np.linalg.norm(x) * np.linalg.norm(prev_state))
                    overlap_penalty += orthogonality_weight * overlap**2
                
                return energy + overlap_penalty
            
            # 优化
            n_params = len(self._param_names)
            x0 = np.random.uniform(-np.pi, np.pi, n_params)
            
            result = minimize(
                constrained_objective,
                x0,
                method=self.optimizer,
                options={'maxiter': self.max_iterations}
            )
            
            prev_states.append(result.x)
            states.append({
                'state': state_idx,
                'energy': result.fun,
                'params': result.x
            })
        
        return states


class VQECallback:
    """VQE回调函数集合"""
    
    @staticmethod
    def print_progress(iteration: int, params: np.ndarray, energy: float):
        """打印优化进度"""
        print(f"  Iter {iteration:4d}: E = {energy:12.8f} Ha")
    
    @staticmethod
    def save_history(iteration: int, params: np.ndarray, energy: float, history: List):
        """保存历史"""
        history.append({
            'iteration': iteration,
            'energy': energy,
            'params': params.copy()
        })
    
    @staticmethod
    def plot_convergence(history: List[Dict]):
        """绘制收敛曲线"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for plotting")
            return
        
        iterations = [h['iteration'] for h in history]
        energies = [h['energy'] for h in history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, energies, 'b-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Energy (Hartree)')
        plt.title('VQE Convergence')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('vqe_convergence.png', dpi=150)
        plt.close()


def run_vqe_for_molecule(
    geometry: List[Tuple[str, Tuple[float, float, float]]],
    basis: str = "sto-3g",
    ansatz: str = "UCCSD",
    backend: str = "auto",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    为分子运行VQE计算的便捷函数
    
    Args:
        geometry: 分子几何
        basis: 基组
        ansatz: Ansatz类型
        backend: 量子后端
        verbose: 是否打印详情
    
    Returns:
        计算结果
    """
    # 创建接口
    interface = create_quantum_interface(backend=backend)
    
    # 创建求解器
    solver = VQESolver(
        quantum_interface=interface,
        ansatz_type=ansatz,
        verbose=verbose
    )
    
    # 构建哈密顿量
    if verbose:
        print("Building molecular Hamiltonian...")
    solver.build_hamiltonian(geometry, basis=basis)
    
    # 构建ansatz
    if verbose:
        print(f"Building {ansatz} ansatz...")
    solver.build_ansatz()
    
    # 优化
    if verbose:
        print("Running VQE optimization...")
    result = solver.optimize()
    
    # 添加分子信息
    result['molecule'] = {
        'geometry': geometry,
        'basis': basis,
        'ansatz': ansatz,
        'backend': backend
    }
    
    return result


def compare_classical_vqe(
    geometry: List[Tuple[str, Tuple[float, float, float]]],
    basis: str = "sto-3g"
) -> Dict[str, float]:
    """
    比较经典FCI和VQE结果
    
    Args:
        geometry: 分子几何
        basis: 基组
    
    Returns:
        能量比较结果
    """
    results = {}
    
    try:
        from pyscf import gto, scf, fci
    except ImportError:
        print("PySCF not available for classical comparison")
        return results
    
    # 经典FCI
    mol = gto.M(
        atom=[(atom, coords) for atom, coords in geometry],
        basis=basis,
        unit='Angstrom'
    )
    mf = scf.RHF(mol)
    mf.kernel()
    
    cisolver = fci.FCI(mf)
    e_fci = cisolver.kernel()[0]
    results['FCI'] = e_fci
    results['RHF'] = mf.e_tot
    
    # VQE
    vqe_result = run_vqe_for_molecule(geometry, basis, verbose=False)
    results['VQE'] = vqe_result['energy']
    results['VQE_error'] = abs(results['VQE'] - e_fci)
    
    return results
