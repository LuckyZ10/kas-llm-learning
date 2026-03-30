"""
量子动力学与经典MD耦合
实现混合量子-经典分子动力学
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import numpy as np
from dataclasses import dataclass
import warnings

try:
    import scipy.linalg
    scipy_available = True
except ImportError:
    scipy_available = False
    scipy = None

try:
    from .quantum_interface import (
        QuantumInterface, 
        QuantumCircuitBase,
        create_quantum_interface,
        QuantumBackend
    )
    from .vqe_solver import VQESolver, MolecularHamiltonian
    from .quantum_ml import QuantumPotentialEnergySurface
except ImportError:
    from quantum_interface import (
        QuantumInterface, 
        QuantumCircuitBase,
        create_quantum_interface,
        QuantumBackend
    )
    from vqe_solver import VQESolver, MolecularHamiltonian
    from quantum_ml import QuantumPotentialEnergySurface


@dataclass
class QuantumRegion:
    """量子区域定义"""
    atom_indices: List[int]
    num_electrons: int
    basis: str = "sto-3g"
    description: str = ""


@dataclass
class ClassicalRegion:
    """经典区域定义"""
    atom_indices: List[int]
    force_field: str = "Lennard-Jones"
    parameters: Dict[str, float] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class QuantumClassicalPartition:
    """量子-经典分区管理"""
    
    def __init__(self):
        self.quantum_regions: List[QuantumRegion] = []
        self.classical_regions: List[ClassicalRegion] = []
        self.coupling_scheme: str = "mechanical"
    
    def add_quantum_region(
        self,
        atom_indices: List[int],
        num_electrons: int,
        basis: str = "sto-3g",
        description: str = ""
    ) -> 'QuantumClassicalPartition':
        """添加量子区域"""
        region = QuantumRegion(
            atom_indices=atom_indices,
            num_electrons=num_electrons,
            basis=basis,
            description=description
        )
        self.quantum_regions.append(region)
        return self
    
    def add_classical_region(
        self,
        atom_indices: List[int],
        force_field: str = "Lennard-Jones",
        parameters: Optional[Dict[str, float]] = None
    ) -> 'QuantumClassicalPartition':
        """添加经典区域"""
        region = ClassicalRegion(
            atom_indices=atom_indices,
            force_field=force_field,
            parameters=parameters or {}
        )
        self.classical_regions.append(region)
        return self
    
    def get_quantum_atoms(self) -> List[int]:
        """获取所有量子原子索引"""
        atoms = []
        for region in self.quantum_regions:
            atoms.extend(region.atom_indices)
        return sorted(list(set(atoms)))
    
    def get_classical_atoms(self) -> List[int]:
        """获取所有经典原子索引"""
        atoms = []
        for region in self.classical_regions:
            atoms.extend(region.atom_indices)
        return sorted(list(set(atoms)))


class QMMCoupling:
    """
    量子-分子力学耦合基类
    
    处理量子区域与经典区域之间的相互作用
    """
    
    def __init__(
        self,
        partition: QuantumClassicalPartition,
        interface: Optional[QuantumInterface] = None
    ):
        self.partition = partition
        self.interface = interface or create_quantum_interface()
        self.coupling_energy = 0.0
        self.coupling_forces: Dict[int, np.ndarray] = {}
    
    @abstractmethod
    def compute_coupling_energy(
        self,
        positions: np.ndarray,
        quantum_density: Optional[np.ndarray] = None
    ) -> float:
        """计算耦合能量"""
        pass
    
    @abstractmethod
    def compute_coupling_forces(
        self,
        positions: np.ndarray,
        quantum_density: Optional[np.ndarray] = None
    ) -> Dict[int, np.ndarray]:
        """计算耦合力"""
        pass


class MechanicalEmbedding(QMMCoupling):
    """
    机械嵌入方案
    经典区域电荷固定在量子区域的外部势中
    """
    
    def __init__(
        self,
        partition: QuantumClassicalPartition,
        interface: Optional[QuantumInterface] = None,
        classical_charges: Optional[Dict[int, float]] = None
    ):
        super().__init__(partition, interface)
        self.classical_charges = classical_charges or {}
        self.point_charges: List[Tuple[np.ndarray, float]] = []
    
    def set_classical_charges(self, charges: Dict[int, float]):
        """设置经典区域电荷"""
        self.classical_charges = charges
    
    def compute_coupling_energy(
        self,
        positions: np.ndarray,
        quantum_density: Optional[np.ndarray] = None
    ) -> float:
        """
        计算机械嵌入耦合能量
        
        E_coupling = Σ(经典电荷 * 量子电势)
        """
        energy = 0.0
        
        for atom_idx, charge in self.classical_charges.items():
            if atom_idx in self.partition.get_classical_atoms():
                # 获取该位置的量子电势
                pos = positions[atom_idx]
                potential = self._compute_quantum_potential_at_point(pos, quantum_density)
                energy += charge * potential
        
        self.coupling_energy = energy
        return energy
    
    def compute_coupling_forces(
        self,
        positions: np.ndarray,
        quantum_density: Optional[np.ndarray] = None
    ) -> Dict[int, np.ndarray]:
        """
        计算机械嵌入耦合力
        
        F = -∇E_coupling
        """
        forces = {}
        delta = 0.001  # 有限差分步长
        
        for atom_idx in self.partition.get_classical_atoms():
            force = np.zeros(3)
            
            for dim in range(3):
                pos_plus = positions.copy()
                pos_plus[atom_idx, dim] += delta
                e_plus = self.compute_coupling_energy(pos_plus, quantum_density)
                
                pos_minus = positions.copy()
                pos_minus[atom_idx, dim] -= delta
                e_minus = self.compute_coupling_energy(pos_minus, quantum_density)
                
                force[dim] = -(e_plus - e_minus) / (2 * delta)
            
            forces[atom_idx] = force
        
        self.coupling_forces = forces
        return forces
    
    def _compute_quantum_potential_at_point(
        self,
        point: np.ndarray,
        quantum_density: Optional[np.ndarray] = None
    ) -> float:
        """计算某点的量子电势"""
        # 简化实现：使用点电荷近似
        if quantum_density is None:
            return 0.0
        
        # 实际应该用量子密度积分计算
        return 0.0


class ElectrostaticEmbedding(QMMCoupling):
    """
    静电嵌入方案 (ONIOM-like)
    更精确的静电相互作用处理
    """
    
    def __init__(
        self,
        partition: QuantumClassicalPartition,
        interface: Optional[QuantumInterface] = None,
        cutoff: float = 10.0
    ):
        super().__init__(partition, interface)
        self.cutoff = cutoff  # 静电相互作用截断距离（Angstrom）
        self.mm_charges: np.ndarray = np.array([])
        self.mm_positions: np.ndarray = np.array([])
    
    def compute_coupling_energy(
        self,
        positions: np.ndarray,
        quantum_density: Optional[np.ndarray] = None
    ) -> float:
        """计算静电嵌入耦合能量"""
        energy = 0.0
        
        qm_atoms = self.partition.get_quantum_atoms()
        mm_atoms = self.partition.get_classical_atoms()
        
        for qm_idx in qm_atoms:
            qm_pos = positions[qm_idx]
            
            for mm_idx in mm_atoms:
                mm_pos = positions[mm_idx]
                distance = np.linalg.norm(qm_pos - mm_pos)
                
                if distance < self.cutoff and distance > 0.1:
                    # 获取MM原子电荷
                    mm_charge = self._get_mm_charge(mm_idx)
                    # 库仑相互作用
                    energy += mm_charge / distance  # 原子单位
        
        self.coupling_energy = energy
        return energy
    
    def compute_coupling_forces(
        self,
        positions: np.ndarray,
        quantum_density: Optional[np.ndarray] = None
    ) -> Dict[int, np.ndarray]:
        """计算静电嵌入耦合力"""
        forces = {}
        
        qm_atoms = self.partition.get_quantum_atoms()
        mm_atoms = self.partition.get_classical_atoms()
        
        # 初始化力
        for idx in qm_atoms + mm_atoms:
            forces[idx] = np.zeros(3)
        
        for qm_idx in qm_atoms:
            qm_pos = positions[qm_idx]
            
            for mm_idx in mm_atoms:
                mm_pos = positions[mm_idx]
                r_vec = qm_pos - mm_pos
                distance = np.linalg.norm(r_vec)
                
                if distance < self.cutoff and distance > 0.1:
                    mm_charge = self._get_mm_charge(mm_idx)
                    
                    # 库仑力 F = -∇(q/r) = q * r_vec / r^3
                    force_magnitude = mm_charge / (distance**3)
                    force = force_magnitude * r_vec
                    
                    forces[qm_idx] += force
                    forces[mm_idx] -= force
        
        self.coupling_forces = forces
        return forces
    
    def _get_mm_charge(self, atom_idx: int) -> float:
        """获取MM原子电荷"""
        # 简化实现：返回固定电荷
        # 实际应该从力场参数获取
        return 0.0


class QuantumDynamics:
    """
    量子动力学演化
    使用TDVP或类似方法演化量子态
    """
    
    def __init__(
        self,
        num_qubits: int,
        dt: float = 0.1,
        interface: Optional[QuantumInterface] = None
    ):
        self.num_qubits = num_qubits
        self.dt = dt  # 时间步长（fs）
        self.interface = interface or create_quantum_interface()
        
        self._current_state: Optional[np.ndarray] = None
        self._time = 0.0
        self._history: List[Dict] = []
    
    def initialize_state(self, state_vector: Optional[np.ndarray] = None):
        """初始化量子态"""
        if state_vector is None:
            # 默认初始化到基态|0...0>
            self._current_state = np.zeros(2**self.num_qubits)
            self._current_state[0] = 1.0
        else:
            self._current_state = state_vector / np.linalg.norm(state_vector)
    
    def evolve_trotter(
        self,
        hamiltonian_pauli: List[Tuple[float, str]],
        n_steps: int = 1
    ) -> np.ndarray:
        """
        使用Trotter-Suzuki分解演化
        
        Args:
            hamiltonian_pauli: [(系数, Pauli字符串), ...]
            n_steps: Trotter步数
        
        Returns:
            演化后的态
        """
        dt_small = self.dt / n_steps
        
        for _ in range(n_steps):
            for coeff, pauli in hamiltonian_pauli:
                # 应用exp(-i * coeff * dt * Pauli)
                self._apply_pauli_rotation(coeff * dt_small, pauli)
        
        self._time += self.dt
        self._history.append({
            'time': self._time,
            'state': self._current_state.copy()
        })
        
        return self._current_state
    
    def _apply_pauli_rotation(self, angle: float, pauli_str: str):
        """应用Pauli旋转门"""
        # 构建电路实现exp(-i * angle * Pauli_string)
        circuit = self.interface.create_circuit(self.num_qubits, "time_evolution")
        
        # 基变换
        for i, p in enumerate(pauli_str):
            if p == 'X':
                circuit.h(i)
            elif p == 'Y':
                circuit.rx(-np.pi/2, i)
        
        # CNOT链
        non_identity_indices = [i for i, p in enumerate(pauli_str) if p != 'I']
        if len(non_identity_indices) > 1:
            for i in range(len(non_identity_indices) - 1):
                circuit.cx(non_identity_indices[i], non_identity_indices[i + 1])
        
        # R_z旋转
        if non_identity_indices:
            circuit.rz(2 * angle, non_identity_indices[-1])
        
        # 反向CNOT链
        if len(non_identity_indices) > 1:
            for i in range(len(non_identity_indices) - 2, -1, -1):
                circuit.cx(non_identity_indices[i], non_identity_indices[i + 1])
        
        # 反向基变换
        for i, p in enumerate(pauli_str):
            if p == 'X':
                circuit.h(i)
            elif p == 'Y':
                circuit.rx(np.pi/2, i)
        
        # 执行（模拟）
        # 实际应该使用量子模拟器
        self._current_state = self._simulate_evolution(self._current_state, angle, pauli_str)
    
    def _simulate_evolution(
        self,
        state: np.ndarray,
        angle: float,
        pauli_str: str
    ) -> np.ndarray:
        """经典模拟演化"""
        # 构建Pauli算符矩阵
        n = len(state)
        n_qubits = int(np.log2(n))
        
        pauli_matrices = {
            'I': np.eye(2),
            'X': np.array([[0, 1], [1, 0]]),
            'Y': np.array([[0, -1j], [1j, 0]]),
            'Z': np.array([[1, 0], [0, -1]])
        }
        
        # 构建完整Pauli字符串矩阵
        P = np.eye(1)
        for p in pauli_str:
            P = np.kron(P, pauli_matrices[p])
        
        # 应用演化算符
        if scipy_available:
            U = scipy.linalg.expm(-1j * angle * P)
        else:
            # 使用泰勒展开近似
            U = np.eye(P.shape[0]) - 1j * angle * P
        return U @ state
    
    def get_expectation_values(
        self,
        observables: List[Tuple[float, str]]
    ) -> Dict[str, float]:
        """计算期望值"""
        results = {}
        
        for coeff, pauli in observables:
            exp_val = self._compute_expectation(pauli)
            results[pauli] = coeff * exp_val
        
        return results
    
    def _compute_expectation(self, pauli_str: str) -> float:
        """计算单个Pauli期望值"""
        n = len(self._current_state)
        n_qubits = int(np.log2(n))
        
        pauli_matrices = {
            'I': np.eye(2),
            'X': np.array([[0, 1], [1, 0]]),
            'Y': np.array([[0, -1j], [1j, 0]]),
            'Z': np.array([[1, 0], [0, -1]])
        }
        
        P = np.eye(1)
        for p in pauli_str:
            P = np.kron(P, pauli_matrices[p])
        
        return np.real(np.conj(self._current_state) @ P @ self._current_state)


class HybridQMMD:
    """
    混合量子-经典分子动力学
    主控制器类
    """
    
    def __init__(
        self,
        partition: QuantumClassicalPartition,
        quantum_interface: Optional[QuantumInterface] = None,
        coupling_scheme: str = "mechanical",
        temperature: float = 300.0,  # K
        dt: float = 0.5,  # fs
    ):
        self.partition = partition
        self.interface = quantum_interface or create_quantum_interface()
        self.temperature = temperature
        self.dt = dt
        
        # 初始化耦合方案
        if coupling_scheme == "mechanical":
            self.coupling = MechanicalEmbedding(partition, self.interface)
        elif coupling_scheme == "electrostatic":
            self.coupling = ElectrostaticEmbedding(partition, self.interface)
        else:
            raise ValueError(f"Unknown coupling scheme: {coupling_scheme}")
        
        # VQE求解器
        self.vqe_solver: Optional[VQESolver] = None
        self.quantum_dynamics: Optional[QuantumDynamics] = None
        
        # 当前状态
        self.positions: Optional[np.ndarray] = None
        self.velocities: Optional[np.ndarray] = None
        self.masses: Optional[np.ndarray] = None
        
        self._step = 0
        self._trajectory: List[Dict] = []
    
    def initialize_system(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
        velocities: Optional[np.ndarray] = None
    ):
        """
        初始化系统
        
        Args:
            positions: 原子位置 (n_atoms, 3)
            masses: 原子质量 (n_atoms,)
            velocities: 初始速度，如果为None则从温度采样
        """
        self.positions = positions.copy()
        self.masses = masses.copy()
        
        if velocities is None:
            # 从麦克斯韦-玻尔兹曼分布采样
            kB = 0.0019872041  # kcal/(mol·K)
            sigma = np.sqrt(kB * self.temperature / masses[:, None])  # 每个原子的速度标准差
            self.velocities = np.random.normal(0, sigma, positions.shape)
        else:
            self.velocities = velocities.copy()
        
        # 移除质心运动
        self._remove_center_of_mass_motion()
    
    def _remove_center_of_mass_motion(self):
        """移除质心平动"""
        total_mass = np.sum(self.masses)
        com_velocity = np.sum(self.masses[:, None] * self.velocities, axis=0) / total_mass
        self.velocities -= com_velocity
    
    def setup_quantum_solver(
        self,
        ansatz_type: str = "UCCSD",
        basis: str = "sto-3g"
    ):
        """设置量子求解器"""
        self.vqe_solver = VQESolver(
            quantum_interface=self.interface,
            ansatz_type=ansatz_type,
            verbose=False
        )
        
        # 为每个量子区域设置求解器
        for region in self.partition.quantum_regions:
            region.basis = basis
    
    def compute_forces(self) -> np.ndarray:
        """
        计算所有原子的力
        
        Returns:
            力数组 (n_atoms, 3)
        """
        n_atoms = len(self.positions)
        forces = np.zeros((n_atoms, 3))
        
        # 1. 量子区域力（来自VQE）
        for region in self.partition.quantum_regions:
            qm_positions = self.positions[region.atom_indices]
            qm_forces = self._compute_quantum_forces(region, qm_positions)
            
            for i, idx in enumerate(region.atom_indices):
                forces[idx] += qm_forces[i]
        
        # 2. 经典区域力（来自力场）
        for region in self.partition.classical_regions:
            mm_positions = self.positions[region.atom_indices]
            mm_forces = self._compute_classical_forces(region, mm_positions)
            
            for i, idx in enumerate(region.atom_indices):
                forces[idx] += mm_forces[i]
        
        # 3. 耦合力
        coupling_forces = self.coupling.compute_coupling_forces(self.positions)
        for idx, force in coupling_forces.items():
            forces[idx] += force
        
        return forces
    
    def _compute_quantum_forces(
        self,
        region: QuantumRegion,
        positions: np.ndarray
    ) -> np.ndarray:
        """计算量子区域的力"""
        # 构建几何
        symbols = ['H'] * len(positions)  # 简化，实际需要原子类型
        geometry = [(s, tuple(p)) for s, p in zip(symbols, positions)]
        
        # 使用VQE或有限差分计算力
        if self.vqe_solver is None:
            return self._finite_difference_forces(geometry, region)
        
        try:
            # 构建哈密顿量并优化
            self.vqe_solver.build_hamiltonian(geometry, basis=region.basis)
            self.vqe_solver.build_ansatz(region.num_electrons)
            result = self.vqe_solver.optimize()
            
            # 使用Hellmann-Feynman定理或有限差分计算力
            return self._finite_difference_forces(geometry, region)
        except Exception as e:
            warnings.warn(f"VQE failed: {e}, using fallback forces")
            return np.zeros_like(positions)
    
    def _finite_difference_forces(
        self,
        geometry: List,
        region: QuantumRegion,
        delta: float = 0.001
    ) -> np.ndarray:
        """使用有限差分计算量子力"""
        n_atoms = len(geometry)
        forces = np.zeros((n_atoms, 3))
        
        # 计算参考能量
        e_ref = self._compute_energy_at_geometry(geometry, region)
        
        for i in range(n_atoms):
            for dim in range(3):
                # 正向位移
                geom_plus = geometry.copy()
                pos = list(geom_plus[i][1])
                pos[dim] += delta
                geom_plus[i] = (geom_plus[i][0], tuple(pos))
                e_plus = self._compute_energy_at_geometry(geom_plus, region)
                
                # 负向位移
                geom_minus = geometry.copy()
                pos = list(geom_minus[i][1])
                pos[dim] -= delta
                geom_minus[i] = (geom_minus[i][0], tuple(pos))
                e_minus = self._compute_energy_at_geometry(geom_minus, region)
                
                forces[i, dim] = -(e_plus - e_minus) / (2 * delta)
        
        return forces
    
    def _compute_energy_at_geometry(
        self,
        geometry: List,
        region: QuantumRegion
    ) -> float:
        """计算特定几何下的能量"""
        if self.vqe_solver is None:
            # 使用简化模型
            return 0.0
        
        try:
            self.vqe_solver.build_hamiltonian(geometry, basis=region.basis)
            self.vqe_solver.build_ansatz(region.num_electrons)
            result = self.vqe_solver.optimize()
            return result['energy']
        except:
            return 0.0
    
    def _compute_classical_forces(
        self,
        region: ClassicalRegion,
        positions: np.ndarray
    ) -> np.ndarray:
        """计算经典区域的力"""
        n_atoms = len(positions)
        forces = np.zeros((n_atoms, 3))
        
        # 简化实现：Lennard-Jones势
        epsilon = region.parameters.get('epsilon', 0.1)
        sigma = region.parameters.get('sigma', 3.0)
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r_vec = positions[i] - positions[j]
                r = np.linalg.norm(r_vec)
                
                if r > 0.1:
                    # LJ力
                    sr6 = (sigma / r)**6
                    force_mag = 24 * epsilon * (2 * sr6**2 - sr6) / r
                    force = force_mag * r_vec / r
                    
                    forces[i] += force
                    forces[j] -= force
        
        return forces
    
    def step(self):
        """
        执行一个MD步（Velocity Verlet积分）
        """
        if self.positions is None or self.velocities is None:
            raise ValueError("System not initialized. Call initialize_system() first.")
        
        dt = self.dt
        masses = self.masses[:, None]
        
        # 1. 计算当前力
        forces = self.compute_forces()
        
        # 2. 更新位置：r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
        self.positions += self.velocities * dt + 0.5 * forces / masses * dt**2
        
        # 3. 计算新力
        forces_new = self.compute_forces()
        
        # 4. 更新速度：v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        self.velocities += 0.5 * (forces + forces_new) / masses * dt
        
        # 5. 更新步数
        self._step += 1
        
        # 6. 记录轨迹
        if self._step % 10 == 0:
            self._trajectory.append({
                'step': self._step,
                'positions': self.positions.copy(),
                'velocities': self.velocities.copy(),
                'forces': forces_new.copy(),
                'kinetic_energy': self._compute_kinetic_energy(),
                'potential_energy': self._compute_potential_energy()
            })
    
    def _compute_kinetic_energy(self) -> float:
        """计算动能"""
        return 0.5 * np.sum(self.masses[:, None] * self.velocities**2)
    
    def _compute_potential_energy(self) -> float:
        """计算势能"""
        # 简化和
        return 0.0
    
    def run(self, n_steps: int, progress_interval: int = 100):
        """
        运行MD模拟
        
        Args:
            n_steps: 步数
            progress_interval: 进度打印间隔
        """
        print(f"Starting hybrid QMMD simulation for {n_steps} steps...")
        print(f"Temperature: {self.temperature} K")
        print(f"Timestep: {self.dt} fs")
        print(f"Quantum regions: {len(self.partition.quantum_regions)}")
        print(f"Classical regions: {len(self.partition.classical_regions)}")
        
        for step in range(n_steps):
            self.step()
            
            if (step + 1) % progress_interval == 0:
                ke = self._compute_kinetic_energy()
                temp = 2 * ke / (3 * len(self.positions) * 0.001987)  # kcal/mol / kB
                print(f"Step {step + 1}/{n_steps}, T = {temp:.1f} K, KE = {ke:.3f} kcal/mol")
        
        print(f"Simulation completed!")
    
    def get_trajectory(self) -> List[Dict]:
        """获取轨迹"""
        return self._trajectory
    
    def save_trajectory(self, filename: str):
        """保存轨迹到文件"""
        import json
        
        # 将numpy数组转换为列表
        serializable_traj = []
        for frame in self._trajectory:
            serializable_traj.append({
                'step': int(frame['step']),
                'positions': frame['positions'].tolist(),
                'kinetic_energy': float(frame['kinetic_energy']),
                'potential_energy': float(frame['potential_energy'])
            })
        
        with open(filename, 'w') as f:
            json.dump(serializable_traj, f, indent=2)
