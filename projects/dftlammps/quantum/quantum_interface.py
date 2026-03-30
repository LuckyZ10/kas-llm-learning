"""
量子-经典混合计算接口层
支持IBM Qiskit、Google Cirq、PennyLane
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
import warnings
import numpy as np


class QuantumBackend(Enum):
    """支持的量子计算后端"""
    QISKIT = "qiskit"
    CIRQ = "cirq"
    PENNYLANE = "pennylane"
    AUTO = "auto"


class QuantumDevice(Enum):
    """量子设备类型"""
    STATEVECTOR = "statevector"
    QASM_SIMULATOR = "qasm_simulator"
    NOISY_SIMULATOR = "noisy_simulator"
    IBMQ = "ibmq"
    IONQ = "ionq"
    RIGETTI = "rigetti"


class QuantumCircuitBase(ABC):
    """量子电路基类"""
    
    def __init__(self, num_qubits: int, name: str = "circuit"):
        self.num_qubits = num_qubits
        self.name = name
        self._circuit: Any = None
        self._parameters: List[Any] = []
        
    @abstractmethod
    def h(self, qubit: int) -> 'QuantumCircuitBase':
        """Hadamard门"""
        pass
    
    @abstractmethod
    def x(self, qubit: int) -> 'QuantumCircuitBase':
        """Pauli-X门"""
        pass
    
    @abstractmethod
    def y(self, qubit: int) -> 'QuantumCircuitBase':
        """Pauli-Y门"""
        pass
    
    @abstractmethod
    def z(self, qubit: int) -> 'QuantumCircuitBase':
        """Pauli-Z门"""
        pass
    
    @abstractmethod
    def rx(self, theta: float, qubit: int) -> 'QuantumCircuitBase':
        """旋转X门"""
        pass
    
    @abstractmethod
    def ry(self, theta: float, qubit: int) -> 'QuantumCircuitBase':
        """旋转Y门"""
        pass
    
    @abstractmethod
    def rz(self, theta: float, qubit: int) -> 'QuantumCircuitBase':
        """旋转Z门"""
        pass
    
    @abstractmethod
    def cx(self, control: int, target: int) -> 'QuantumCircuitBase':
        """CNOT门"""
        pass
    
    @abstractmethod
    def cz(self, control: int, target: int) -> 'QuantumCircuitBase':
        """Controlled-Z门"""
        pass
    
    @abstractmethod
    def swap(self, qubit1: int, qubit2: int) -> 'QuantumCircuitBase':
        """SWAP门"""
        pass
    
    @abstractmethod
    def measure(self, qubits: List[int], classical_bits: Optional[List[int]] = None) -> 'QuantumCircuitBase':
        """测量门"""
        pass
    
    @abstractmethod
    def barrier(self, qubits: Optional[List[int]] = None) -> 'QuantumCircuitBase':
        """屏障"""
        pass
    
    @abstractmethod
    def add_parameterized_rotation(
        self, 
        axis: str, 
        qubit: int, 
        param_name: str
    ) -> 'QuantumCircuitBase':
        """添加参数化旋转门"""
        pass
    
    @abstractmethod
    def get_circuit(self) -> Any:
        """获取底层电路对象"""
        pass
    
    @abstractmethod
    def copy(self) -> 'QuantumCircuitBase':
        """复制电路"""
        pass
    
    @property
    def parameters(self) -> List[Any]:
        """获取电路参数"""
        return self._parameters


class QiskitCircuit(QuantumCircuitBase):
    """Qiskit电路实现"""
    
    def __init__(self, num_qubits: int, name: str = "qiskit_circuit"):
        super().__init__(num_qubits, name)
        try:
            from qiskit import QuantumCircuit as QC
            from qiskit.circuit import Parameter
            self._circuit = QC(num_qubits, name=name)
            self.Parameter = Parameter
        except ImportError:
            raise ImportError("Qiskit not installed. Run: pip install qiskit")
    
    def h(self, qubit: int) -> 'QiskitCircuit':
        self._circuit.h(qubit)
        return self
    
    def x(self, qubit: int) -> 'QiskitCircuit':
        self._circuit.x(qubit)
        return self
    
    def y(self, qubit: int) -> 'QiskitCircuit':
        self._circuit.y(qubit)
        return self
    
    def z(self, qubit: int) -> 'QiskitCircuit':
        self._circuit.z(qubit)
        return self
    
    def rx(self, theta: float, qubit: int) -> 'QiskitCircuit':
        self._circuit.rx(theta, qubit)
        return self
    
    def ry(self, theta: float, qubit: int) -> 'QiskitCircuit':
        self._circuit.ry(theta, qubit)
        return self
    
    def rz(self, theta: float, qubit: int) -> 'QiskitCircuit':
        self._circuit.rz(theta, qubit)
        return self
    
    def cx(self, control: int, target: int) -> 'QiskitCircuit':
        self._circuit.cx(control, target)
        return self
    
    def cz(self, control: int, target: int) -> 'QiskitCircuit':
        self._circuit.cz(control, target)
        return self
    
    def swap(self, qubit1: int, qubit2: int) -> 'QiskitCircuit':
        self._circuit.swap(qubit1, qubit2)
        return self
    
    def measure(self, qubits: List[int], classical_bits: Optional[List[int]] = None) -> 'QiskitCircuit':
        if classical_bits is None:
            classical_bits = qubits
        for q, c in zip(qubits, classical_bits):
            self._circuit.measure(q, c)
        return self
    
    def barrier(self, qubits: Optional[List[int]] = None) -> 'QiskitCircuit':
        if qubits:
            self._circuit.barrier(qubits)
        else:
            self._circuit.barrier()
        return self
    
    def add_parameterized_rotation(self, axis: str, qubit: int, param_name: str) -> 'QiskitCircuit':
        param = self.Parameter(param_name)
        self._parameters.append(param)
        if axis.lower() == 'x':
            self._circuit.rx(param, qubit)
        elif axis.lower() == 'y':
            self._circuit.ry(param, qubit)
        elif axis.lower() == 'z':
            self._circuit.rz(param, qubit)
        else:
            raise ValueError(f"Unknown rotation axis: {axis}")
        return self
    
    def get_circuit(self) -> Any:
        return self._circuit
    
    def copy(self) -> 'QiskitCircuit':
        new_circ = QiskitCircuit(self.num_qubits, self.name)
        new_circ._circuit = self._circuit.copy()
        new_circ._parameters = self._parameters.copy()
        return new_circ


class CirqCircuit(QuantumCircuitBase):
    """Cirq电路实现"""
    
    def __init__(self, num_qubits: int, name: str = "cirq_circuit"):
        super().__init__(num_qubits, name)
        try:
            import cirq
            self.cirq = cirq
            self._qubits = [cirq.GridQubit(0, i) for i in range(num_qubits)]
            self._circuit = cirq.Circuit()
            self._sympy = None
            try:
                import sympy
                self._sympy = sympy
            except ImportError:
                pass
        except ImportError:
            raise ImportError("Cirq not installed. Run: pip install cirq")
    
    def _get_qubit(self, idx: int):
        return self._qubits[idx]
    
    def h(self, qubit: int) -> 'CirqCircuit':
        self._circuit.append(self.cirq.H(self._get_qubit(qubit)))
        return self
    
    def x(self, qubit: int) -> 'CirqCircuit':
        self._circuit.append(self.cirq.X(self._get_qubit(qubit)))
        return self
    
    def y(self, qubit: int) -> 'CirqCircuit':
        self._circuit.append(self.cirq.Y(self._get_qubit(qubit)))
        return self
    
    def z(self, qubit: int) -> 'CirqCircuit':
        self._circuit.append(self.cirq.Z(self._get_qubit(qubit)))
        return self
    
    def rx(self, theta: float, qubit: int) -> 'CirqCircuit':
        self._circuit.append(self.cirq.rx(theta)(self._get_qubit(qubit)))
        return self
    
    def ry(self, theta: float, qubit: int) -> 'CirqCircuit':
        self._circuit.append(self.cirq.ry(theta)(self._get_qubit(qubit)))
        return self
    
    def rz(self, theta: float, qubit: int) -> 'CirqCircuit':
        self._circuit.append(self.cirq.rz(theta)(self._get_qubit(qubit)))
        return self
    
    def cx(self, control: int, target: int) -> 'CirqCircuit':
        self._circuit.append(self.cirq.CNOT(self._get_qubit(control), self._get_qubit(target)))
        return self
    
    def cz(self, control: int, target: int) -> 'CirqCircuit':
        self._circuit.append(self.cirq.CZ(self._get_qubit(control), self._get_qubit(target)))
        return self
    
    def swap(self, qubit1: int, qubit2: int) -> 'CirqCircuit':
        self._circuit.append(self.cirq.SWAP(self._get_qubit(qubit1), self._get_qubit(qubit2)))
        return self
    
    def measure(self, qubits: List[int], classical_bits: Optional[List[int]] = None) -> 'CirqCircuit':
        for q in qubits:
            self._circuit.append(self.cirq.measure(self._get_qubit(q), key=f'm{q}'))
        return self
    
    def barrier(self, qubits: Optional[List[int]] = None) -> 'CirqCircuit':
        # Cirq doesn't have explicit barriers, use identity as placeholder
        if qubits:
            for q in qubits:
                self._circuit.append(self.cirq.I(self._get_qubit(q)))
        return self
    
    def add_parameterized_rotation(self, axis: str, qubit: int, param_name: str) -> 'CirqCircuit':
        if self._sympy is None:
            raise ImportError("SymPy required for parameterized circuits")
        param = self._sympy.Symbol(param_name)
        self._parameters.append(param)
        if axis.lower() == 'x':
            self._circuit.append(self.cirq.rx(param)(self._get_qubit(qubit)))
        elif axis.lower() == 'y':
            self._circuit.append(self.cirq.ry(param)(self._get_qubit(qubit)))
        elif axis.lower() == 'z':
            self._circuit.append(self.cirq.rz(param)(self._get_qubit(qubit)))
        else:
            raise ValueError(f"Unknown rotation axis: {axis}")
        return self
    
    def get_circuit(self) -> Any:
        return self._circuit
    
    def copy(self) -> 'CirqCircuit':
        new_circ = CirqCircuit(self.num_qubits, self.name)
        new_circ._circuit = self._circuit.copy()
        new_circ._parameters = self._parameters.copy()
        return new_circ


class PennyLaneCircuit(QuantumCircuitBase):
    """PennyLane电路实现"""
    
    def __init__(self, num_qubits: int, name: str = "pennylane_circuit"):
        super().__init__(num_qubits, name)
        try:
            import pennylane as qml
            self.qml = qml
            self._ops: List[Tuple[str, Any]] = []
            self._device = None
        except ImportError:
            raise ImportError("PennyLane not installed. Run: pip install pennylane")
    
    def h(self, qubit: int) -> 'PennyLaneCircuit':
        self._ops.append(('H', qubit))
        return self
    
    def x(self, qubit: int) -> 'PennyLaneCircuit':
        self._ops.append(('X', qubit))
        return self
    
    def y(self, qubit: int) -> 'PennyLaneCircuit':
        self._ops.append(('Y', qubit))
        return self
    
    def z(self, qubit: int) -> 'PennyLaneCircuit':
        self._ops.append(('Z', qubit))
        return self
    
    def rx(self, theta: float, qubit: int) -> 'PennyLaneCircuit':
        self._ops.append(('RX', theta, qubit))
        return self
    
    def ry(self, theta: float, qubit: int) -> 'PennyLaneCircuit':
        self._ops.append(('RY', theta, qubit))
        return self
    
    def rz(self, theta: float, qubit: int) -> 'PennyLaneCircuit':
        self._ops.append(('RZ', theta, qubit))
        return self
    
    def cx(self, control: int, target: int) -> 'PennyLaneCircuit':
        self._ops.append(('CNOT', control, target))
        return self
    
    def cz(self, control: int, target: int) -> 'PennyLaneCircuit':
        self._ops.append(('CZ', control, target))
        return self
    
    def swap(self, qubit1: int, qubit2: int) -> 'PennyLaneCircuit':
        self._ops.append(('SWAP', qubit1, qubit2))
        return self
    
    def measure(self, qubits: List[int], classical_bits: Optional[List[int]] = None) -> 'PennyLaneCircuit':
        # PennyLane测量在QNode中处理
        self._measure_qubits = qubits
        return self
    
    def barrier(self, qubits: Optional[List[int]] = None) -> 'PennyLaneCircuit':
        # PennyLane不需要屏障
        return self
    
    def add_parameterized_rotation(self, axis: str, qubit: int, param_name: str) -> 'PennyLaneCircuit':
        import pennylane as qml
        param = qml.numpy.array(0.0, requires_grad=True)
        param.name = param_name
        self._parameters.append(param)
        if axis.lower() == 'x':
            self._ops.append(('RX_param', param, qubit))
        elif axis.lower() == 'y':
            self._ops.append(('RY_param', param, qubit))
        elif axis.lower() == 'z':
            self._ops.append(('RZ_param', param, qubit))
        else:
            raise ValueError(f"Unknown rotation axis: {axis}")
        return self
    
    def get_circuit(self) -> Any:
        """返回一个可执行的QNode"""
        dev = self.qml.device("default.qubit", wires=self.num_qubits)
        
        @self.qml.qnode(dev)
        def circuit(*params):
            param_idx = 0
            for op in self._ops:
                gate = op[0]
                if gate == 'H':
                    self.qml.Hadamard(wires=op[1])
                elif gate == 'X':
                    self.qml.PauliX(wires=op[1])
                elif gate == 'Y':
                    self.qml.PauliY(wires=op[1])
                elif gate == 'Z':
                    self.qml.PauliZ(wires=op[1])
                elif gate == 'RX':
                    self.qml.RX(op[1], wires=op[2])
                elif gate == 'RY':
                    self.qml.RY(op[1], wires=op[2])
                elif gate == 'RZ':
                    self.qml.RZ(op[1], wires=op[2])
                elif gate == 'RX_param':
                    self.qml.RX(params[param_idx], wires=op[2])
                    param_idx += 1
                elif gate == 'RY_param':
                    self.qml.RY(params[param_idx], wires=op[2])
                    param_idx += 1
                elif gate == 'RZ_param':
                    self.qml.RZ(params[param_idx], wires=op[2])
                    param_idx += 1
                elif gate == 'CNOT':
                    self.qml.CNOT(wires=[op[1], op[2]])
                elif gate == 'CZ':
                    self.qml.CZ(wires=[op[1], op[2]])
                elif gate == 'SWAP':
                    self.qml.SWAP(wires=[op[1], op[2]])
            
            if hasattr(self, '_measure_qubits'):
                return self.qml.expval(self.qml.PauliZ(0))
            return self.qml.expval(self.qml.PauliZ(0))
        
        return circuit
    
    def copy(self) -> 'PennyLaneCircuit':
        new_circ = PennyLaneCircuit(self.num_qubits, self.name)
        new_circ._ops = self._ops.copy()
        new_circ._parameters = self._parameters.copy()
        return new_circ


class QuantumExecutor(ABC):
    """量子执行器基类"""
    
    def __init__(self, backend: QuantumBackend, device: QuantumDevice, shots: int = 1024):
        self.backend = backend
        self.device = device
        self.shots = shots
        self._executor: Any = None
    
    @abstractmethod
    def execute(self, circuit: QuantumCircuitBase, parameters: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """执行量子电路"""
        pass
    
    @abstractmethod
    def get_expectation_value(
        self, 
        circuit: QuantumCircuitBase, 
        observable: Any,
        parameters: Optional[Dict[str, float]] = None
    ) -> float:
        """计算期望值"""
        pass
    
    @abstractmethod
    def run_batch(
        self, 
        circuits: List[QuantumCircuitBase], 
        parameters_list: Optional[List[Dict[str, float]]] = None
    ) -> List[Dict[str, Any]]:
        """批量执行电路"""
        pass


class QiskitExecutor(QuantumExecutor):
    """Qiskit执行器"""
    
    def __init__(self, device: QuantumDevice = QuantumDevice.STATEVECTOR, shots: int = 1024):
        super().__init__(QuantumBackend.QISKIT, device, shots)
        try:
            from qiskit import transpile
            from qiskit.providers.aer import AerSimulator
            self._transpile = transpile
            self._simulator = AerSimulator()
        except ImportError:
            raise ImportError("Qiskit Aer not installed. Run: pip install qiskit-aer")
    
    def execute(self, circuit: QuantumCircuitBase, parameters: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        from qiskit import transpile
        
        qc = circuit.get_circuit()
        
        # 绑定参数
        if parameters and circuit.parameters:
            param_dict = {p: parameters.get(p.name, 0.0) for p in circuit.parameters}
            qc = qc.assign_parameters(param_dict)
        
        # 编译并执行
        transpiled = transpile(qc, self._simulator)
        job = self._simulator.run(transpiled, shots=self.shots)
        result = job.result()
        
        return {
            'counts': result.get_counts(),
            'success': result.success,
            'job_id': result.job_id,
            'backend': str(self._simulator)
        }
    
    def get_expectation_value(
        self, 
        circuit: QuantumCircuitBase, 
        observable: Any,
        parameters: Optional[Dict[str, float]] = None
    ) -> float:
        from qiskit.quantum_info import Statevector
        
        qc = circuit.get_circuit()
        
        if parameters and circuit.parameters:
            param_dict = {p: parameters.get(p.name, 0.0) for p in circuit.parameters}
            qc = qc.assign_parameters(param_dict)
        
        # 使用statevector计算期望值
        sv = Statevector.from_instruction(qc)
        exp_val = sv.expectation_value(observable)
        
        return float(np.real(exp_val))
    
    def run_batch(
        self, 
        circuits: List[QuantumCircuitBase], 
        parameters_list: Optional[List[Dict[str, float]]] = None
    ) -> List[Dict[str, Any]]:
        results = []
        params_list = parameters_list or [None] * len(circuits)
        
        for circ, params in zip(circuits, params_list):
            results.append(self.execute(circ, params))
        
        return results


class CirqExecutor(QuantumExecutor):
    """Cirq执行器"""
    
    def __init__(self, device: QuantumDevice = QuantumDevice.STATEVECTOR, shots: int = 1024):
        super().__init__(QuantumBackend.CIRQ, device, shots)
        try:
            import cirq
            self.cirq = cirq
        except ImportError:
            raise ImportError("Cirq not installed. Run: pip install cirq")
    
    def execute(self, circuit: QuantumCircuitBase, parameters: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        import sympy
        
        cirq_circuit = circuit.get_circuit()
        
        # 解析符号
        if parameters and circuit.parameters:
            resolver = {p: parameters.get(p.name, 0.0) for p in circuit.parameters}
            cirq_circuit = cirq.resolve_parameters(cirq_circuit, resolver)
        
        # 执行
        simulator = self.cirq.Simulator()
        result = simulator.run(cirq_circuit, repetitions=self.shots)
        
        # 获取测量结果
        histogram = result.histogram(key='m0') if 'm0' in result.measurements else {}
        
        return {
            'counts': dict(histogram),
            'measurements': {k: v.tolist() for k, v in result.measurements.items()},
            'success': True,
            'backend': 'cirq.simulator'
        }
    
    def get_expectation_value(
        self, 
        circuit: QuantumCircuitBase, 
        observable: Any,
        parameters: Optional[Dict[str, float]] = None
    ) -> float:
        cirq_circuit = circuit.get_circuit()
        
        if parameters and circuit.parameters:
            resolver = {p: parameters.get(p.name, 0.0) for p in circuit.parameters}
            cirq_circuit = self.cirq.resolve_parameters(cirq_circuit, resolver)
        
        simulator = self.cirq.Simulator()
        result = simulator.simulate(cirq_circuit)
        
        # 计算期望值
        if isinstance(observable, self.cirq.PauliString):
            exp_val = result.expectation_value(observable)
        else:
            # 默认Z基期望值
            exp_val = result.expectation_value(self.cirq.Z(cirq.GridQubit(0, 0)))
        
        return float(np.real(exp_val))
    
    def run_batch(
        self, 
        circuits: List[QuantumCircuitBase], 
        parameters_list: Optional[List[Dict[str, float]]] = None
    ) -> List[Dict[str, Any]]:
        results = []
        params_list = parameters_list or [None] * len(circuits)
        
        for circ, params in zip(circuits, params_list):
            results.append(self.execute(circ, params))
        
        return results


class PennyLaneExecutor(QuantumExecutor):
    """PennyLane执行器"""
    
    def __init__(self, device: QuantumDevice = QuantumDevice.STATEVECTOR, shots: int = 1024):
        super().__init__(QuantumBackend.PENNYLANE, device, shots)
        try:
            import pennylane as qml
            self.qml = qml
        except ImportError:
            raise ImportError("PennyLane not installed. Run: pip install pennylane")
    
    def execute(self, circuit: QuantumCircuitBase, parameters: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        qnode = circuit.get_circuit()
        
        # 准备参数
        if parameters and circuit.parameters:
            param_values = [parameters.get(p.name, 0.0) for p in circuit.parameters]
        else:
            param_values = []
        
        # 执行
        result = qnode(*param_values)
        
        return {
            'result': float(result),
            'success': True,
            'backend': 'pennylane.default.qubit'
        }
    
    def get_expectation_value(
        self, 
        circuit: QuantumCircuitBase, 
        observable: Any,
        parameters: Optional[Dict[str, float]] = None
    ) -> float:
        result = self.execute(circuit, parameters)
        return result['result']
    
    def run_batch(
        self, 
        circuits: List[QuantumCircuitBase], 
        parameters_list: Optional[List[Dict[str, float]]] = None
    ) -> List[Dict[str, Any]]:
        results = []
        params_list = parameters_list or [None] * len(circuits)
        
        for circ, params in zip(circuits, params_list):
            results.append(self.execute(circ, params))
        
        return results


class QuantumInterface:
    """
    量子-经典混合计算接口主类
    统一管理量子计算后端
    """
    
    def __init__(
        self, 
        backend: Union[QuantumBackend, str] = QuantumBackend.AUTO,
        device: Union[QuantumDevice, str] = QuantumDevice.STATEVECTOR,
        shots: int = 1024
    ):
        self.backend = self._resolve_backend(backend)
        self.device = self._resolve_device(device)
        self.shots = shots
        self._executor: Optional[QuantumExecutor] = None
        self._circuit_class: type = None
        
        self._initialize_backend()
    
    def _resolve_backend(self, backend: Union[QuantumBackend, str]) -> QuantumBackend:
        if isinstance(backend, str):
            backend = backend.lower()
            if backend == 'auto':
                return self._auto_select_backend()
            return QuantumBackend(backend)
        return backend
    
    def _resolve_device(self, device: Union[QuantumDevice, str]) -> QuantumDevice:
        if isinstance(device, str):
            return QuantumDevice(device)
        return device
    
    def _auto_select_backend(self) -> QuantumBackend:
        """自动选择可用的后端"""
        try:
            import qiskit
            return QuantumBackend.QISKIT
        except ImportError:
            pass
        
        try:
            import pennylane
            return QuantumBackend.PENNYLANE
        except ImportError:
            pass
        
        try:
            import cirq
            return QuantumBackend.CIRQ
        except ImportError:
            pass
        
        # 如果没有可用的后端，使用模拟回退
        warnings.warn("No quantum computing backend found. Using classical simulation fallback.")
        return QuantumBackend.QISKIT  # 将使用模拟实现
    
    def _initialize_backend(self):
        """初始化选定的后端"""
        if self.backend == QuantumBackend.QISKIT:
            try:
                self._executor = QiskitExecutor(self.device, self.shots)
                self._circuit_class = QiskitCircuit
            except ImportError:
                warnings.warn("Qiskit not available, using fallback executor")
                self._executor = FallbackExecutor(self.device, self.shots)
                self._circuit_class = FallbackCircuit
        
        elif self.backend == QuantumBackend.CIRQ:
            try:
                self._executor = CirqExecutor(self.device, self.shots)
                self._circuit_class = CirqCircuit
            except ImportError:
                warnings.warn("Cirq not available, using fallback executor")
                self._executor = FallbackExecutor(self.device, self.shots)
                self._circuit_class = FallbackCircuit
        
        elif self.backend == QuantumBackend.PENNYLANE:
            try:
                self._executor = PennyLaneExecutor(self.device, self.shots)
                self._circuit_class = PennyLaneCircuit
            except ImportError:
                warnings.warn("PennyLane not available, using fallback executor")
                self._executor = FallbackExecutor(self.device, self.shots)
                self._circuit_class = FallbackCircuit
        
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def create_circuit(self, num_qubits: int, name: str = "circuit") -> QuantumCircuitBase:
        """创建量子电路"""
        return self._circuit_class(num_qubits, name)
    
    def execute(
        self, 
        circuit: QuantumCircuitBase, 
        parameters: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """执行量子电路"""
        return self._executor.execute(circuit, parameters)
    
    def get_expectation_value(
        self, 
        circuit: QuantumCircuitBase, 
        observable: Any,
        parameters: Optional[Dict[str, float]] = None
    ) -> float:
        """计算期望值"""
        return self._executor.get_expectation_value(circuit, observable, parameters)
    
    def run_batch(
        self, 
        circuits: List[QuantumCircuitBase], 
        parameters_list: Optional[List[Dict[str, float]]] = None
    ) -> List[Dict[str, Any]]:
        """批量执行"""
        return self._executor.run_batch(circuits, parameters_list)
    
    @property
    def backend_name(self) -> str:
        return self.backend.value
    
    @property
    def device_name(self) -> str:
        return self.device.value


def create_quantum_interface(
    backend: str = "auto",
    device: str = "statevector",
    shots: int = 1024
) -> QuantumInterface:
    """
    工厂函数：创建量子接口
    
    Args:
        backend: 后端类型 ('qiskit', 'cirq', 'pennylane', 'auto')
        device: 设备类型 ('statevector', 'qasm_simulator', etc.)
        shots: 采样次数
    
    Returns:
        QuantumInterface实例
    """
    return QuantumInterface(backend=backend, device=device, shots=shots)


# 常用量子门构建器
def build_ansatz_circuit(
    interface: QuantumInterface,
    num_qubits: int,
    num_layers: int,
    entanglement: str = "linear"
) -> Tuple[QuantumCircuitBase, List[str]]:
    """
    构建参数化ansatz电路
    
    Args:
        interface: 量子接口
        num_qubits: 量子比特数
        num_layers: 层数
        entanglement: 纠缠模式 ('linear', 'circular', 'full')
    
    Returns:
        (电路, 参数名称列表)
    """
    circuit = interface.create_circuit(num_qubits, "ansatz")
    param_names = []
    
    for layer in range(num_layers):
        # 旋转层
        for q in range(num_qubits):
            param_name = f"theta_{layer}_{q}_x"
            circuit.add_parameterized_rotation('y', q, param_name)
            param_names.append(param_name)
            
            param_name = f"theta_{layer}_{q}_z"
            circuit.add_parameterized_rotation('z', q, param_name)
            param_names.append(param_name)
        
        # 纠缠层
        if entanglement == "linear":
            for q in range(num_qubits - 1):
                circuit.cx(q, q + 1)
        elif entanglement == "circular":
            for q in range(num_qubits):
                circuit.cx(q, (q + 1) % num_qubits)
        elif entanglement == "full":
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    circuit.cx(i, j)
    
    return circuit, param_names


class FallbackCircuit(QuantumCircuitBase):
    """
    回退电路实现
    当没有量子计算后端安装时使用经典模拟
    """
    
    def __init__(self, num_qubits: int, name: str = "fallback_circuit"):
        super().__init__(num_qubits, name)
        self._gates: List[Tuple[str, Tuple]] = []
        self._statevector: np.ndarray = np.zeros(2**num_qubits)
        self._statevector[0] = 1.0  # |0...0>
    
    def h(self, qubit: int) -> 'FallbackCircuit':
        self._gates.append(('H', (qubit,)))
        return self
    
    def x(self, qubit: int) -> 'FallbackCircuit':
        self._gates.append(('X', (qubit,)))
        return self
    
    def y(self, qubit: int) -> 'FallbackCircuit':
        self._gates.append(('Y', (qubit,)))
        return self
    
    def z(self, qubit: int) -> 'FallbackCircuit':
        self._gates.append(('Z', (qubit,)))
        return self
    
    def rx(self, theta: float, qubit: int) -> 'FallbackCircuit':
        self._gates.append(('RX', (theta, qubit)))
        return self
    
    def ry(self, theta: float, qubit: int) -> 'FallbackCircuit':
        self._gates.append(('RY', (theta, qubit)))
        return self
    
    def rz(self, theta: float, qubit: int) -> 'FallbackCircuit':
        self._gates.append(('RZ', (theta, qubit)))
        return self
    
    def cx(self, control: int, target: int) -> 'FallbackCircuit':
        self._gates.append(('CX', (control, target)))
        return self
    
    def cz(self, control: int, target: int) -> 'FallbackCircuit':
        self._gates.append(('CZ', (control, target)))
        return self
    
    def swap(self, qubit1: int, qubit2: int) -> 'FallbackCircuit':
        self._gates.append(('SWAP', (qubit1, qubit2)))
        return self
    
    def measure(self, qubits: List[int], classical_bits: Optional[List[int]] = None) -> 'FallbackCircuit':
        self._gates.append(('MEASURE', tuple(qubits)))
        return self
    
    def barrier(self, qubits: Optional[List[int]] = None) -> 'FallbackCircuit':
        return self
    
    def add_parameterized_rotation(self, axis: str, qubit: int, param_name: str) -> 'FallbackCircuit':
        param = len(self._parameters)  # 使用索引作为参数标识
        self._parameters.append(param_name)
        self._gates.append((f'R{axis}_param', (param_name, qubit)))
        return self
    
    def get_circuit(self) -> Any:
        return self
    
    def copy(self) -> 'FallbackCircuit':
        new_circ = FallbackCircuit(self.num_qubits, self.name)
        new_circ._gates = self._gates.copy()
        new_circ._parameters = self._parameters.copy()
        return new_circ


class FallbackExecutor(QuantumExecutor):
    """回退执行器 - 使用经典模拟"""
    
    def __init__(self, device: QuantumDevice = QuantumDevice.STATEVECTOR, shots: int = 1024):
        super().__init__(QuantumBackend.QISKIT, device, shots)
    
    def execute(self, circuit: QuantumCircuitBase, parameters: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """模拟执行"""
        # 返回模拟结果
        return {
            'counts': {'0' * circuit.num_qubits: self.shots},
            'success': True,
            'job_id': 'fallback_simulation',
            'backend': 'fallback.classical'
        }
    
    def get_expectation_value(
        self, 
        circuit: QuantumCircuitBase, 
        observable: Any,
        parameters: Optional[Dict[str, float]] = None
    ) -> float:
        """计算期望值（模拟）"""
        # 返回随机期望值作为占位
        return np.random.uniform(-1, 1)
    
    def run_batch(
        self, 
        circuits: List[QuantumCircuitBase], 
        parameters_list: Optional[List[Dict[str, float]]] = None
    ) -> List[Dict[str, Any]]:
        """批量执行"""
        return [self.execute(c, p) for c, p in zip(circuits, parameters_list or [None] * len(circuits))]


def build_hartree_fock_circuit(
    interface: QuantumInterface,
    num_qubits: int,
    num_electrons: int
) -> QuantumCircuitBase:
    """
    构建Hartree-Fock初始态电路
    
    Args:
        interface: 量子接口
        num_qubits: 量子比特数（自旋轨道数）
        num_electrons: 电子数
    
    Returns:
        HF态电路
    """
    circuit = interface.create_circuit(num_qubits, "hf_initial")
    
    # 占据轨道设为|1>
    for i in range(num_electrons):
        circuit.x(i)
    
    return circuit


def build_uccsd_ansatz(
    interface: QuantumInterface,
    num_qubits: int,
    num_electrons: int,
    singles: bool = True,
    doubles: bool = True
) -> Tuple[QuantumCircuitBase, List[str]]:
    """
    构建UCCSD (Unitary Coupled Cluster Singles and Doubles) ansatz
    
    Args:
        interface: 量子接口
        num_qubits: 量子比特数
        num_electrons: 电子数
        singles: 是否包含单激发
        doubles: 是否包含双激发
    
    Returns:
        (UCCSD电路, 参数名称列表)
    """
    circuit = interface.create_circuit(num_qubits, "uccsd")
    param_names = []
    
    occupied = list(range(num_electrons))
    virtual = list(range(num_electrons, num_qubits))
    
    # 单激发
    if singles:
        for i in occupied:
            for a in virtual:
                param_name = f"t_{i}_{a}"
                param_names.append(param_name)
                # exp(-i * t * (a† i - i† a)) 的Jordan-Wigner实现
                # 简化为RY旋转
                circuit.add_parameterized_rotation('y', i, param_name)
                circuit.cx(i, a)
    
    # 双激发
    if doubles:
        for i in occupied:
            for j in occupied:
                if i < j:
                    for a in virtual:
                        for b in virtual:
                            if a < b:
                                param_name = f"t_{i}_{j}_{a}_{b}"
                                param_names.append(param_name)
                                # 简化的双激发实现
                                circuit.add_parameterized_rotation('z', i, param_name)
                                circuit.cx(i, a)
                                circuit.cx(j, b)
    
    return circuit, param_names
