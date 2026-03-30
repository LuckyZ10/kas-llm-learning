"""
QAOA Interface Module - Quantum Approximate Optimization Algorithm
量子近似优化算法接口模块

本模块实现了QAOA算法用于组合优化问题，包括：
- 组合优化问题映射到量子比特
- MaxCut、图着色、材料设计优化
- 多目标优化
- 与经典优化器混合

作者: Quantum Computing Team
版本: 1.0.0
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable, Union, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import warnings
from collections import defaultdict
import itertools

# 尝试导入量子计算库
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.primitives import Estimator, Sampler
    from qiskit_algorithms import QAOA as QiskitQAOA
    from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP, NFT
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_algorithms import NumPyMinimumEigensolver
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    warnings.warn("Qiskit not available. QAOA functionality will be limited.")

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("NetworkX not available. Graph operations limited.")


class OptimizationType(Enum):
    """优化问题类型"""
    MAXCUT = "maxcut"
    MAXCLIQUE = "maxclique"
    GRAPH_COLORING = "graph_coloring"
    TSP = "tsp"
    KNAPSACK = "knapsack"
    NUMBER_PARTITIONING = "number_partitioning"
    SET_PARTITIONING = "set_partitioning"
    BIN_PACKING = "bin_packing"
    MATERIAL_DESIGN = "material_design"
    CUSTOM = "custom"


class MixerType(Enum):
    """QAOA Mixer类型"""
    STANDARD = "standard"       # X-mixer
    XY = "xy"                   # XY-mixer (适用于约束优化)
    RING = "ring"               # 环形mixer
    PARITY = "parity"           # 奇偶mixer
    CUSTOM = "custom"           # 自定义mixer


@dataclass
class QAOAConfig:
    """QAOA配置参数"""
    # 算法参数
    p_layers: int = 3  # QAOA层数
    optimization_type: OptimizationType = OptimizationType.MAXCUT
    mixer_type: MixerType = MixerType.STANDARD
    
    # 优化器参数
    optimizer: str = "COBYLA"
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6
    
    # 初始参数策略
    init_params: str = "random"  # random, zeros, linear_ramp
    init_beta_range: Tuple[float, float] = (0.0, np.pi)
    init_gamma_range: Tuple[float, float] = (0.0, 2*np.pi)
    
    # 经典优化器
    use_warm_start: bool = False
    use_recursive_qaoa: bool = False
    
    # 量子后端
    backend: str = "statevector"
    shots: Optional[int] = None
    
    # 约束处理
    penalty_weight: float = 10.0  # 约束违反惩罚权重
    use_soft_constraints: bool = False
    
    # 多目标优化
    multi_objective: bool = False
    objective_weights: Optional[List[float]] = None
    
    # 采样参数
    n_samples: int = 1000  # 最终采样次数
    top_k_solutions: int = 10  # 返回前k个解


@dataclass
class OptimizationProblem:
    """优化问题数据结构"""
    problem_type: OptimizationType
    n_variables: int
    n_constraints: int = 0
    
    # 问题数据
    graph: Optional[Any] = None  # 图问题
    weights: Optional[np.ndarray] = None  # 权重
    costs: Optional[np.ndarray] = None  # 成本
    constraints: Optional[List[Callable]] = None
    
    # Ising/QUBO表示
    ising_h: Optional[np.ndarray] = None  # 线性项
    ising_j: Optional[np.ndarray] = None  # 二次项
    ising_offset: float = 0.0
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_ising(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """转换为Ising模型"""
        if self.ising_h is not None and self.ising_j is not None:
            return self.ising_h, self.ising_j, self.ising_offset
        
        # 从问题类型转换
        if self.problem_type == OptimizationType.MAXCUT and self.graph is not None:
            return self._graph_to_ising()
        
        raise NotImplementedError(f"Conversion for {self.problem_type} not implemented")
    
    def _graph_to_ising(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """图问题转换为Ising模型"""
        n = self.n_variables
        h = np.zeros(n)
        J = np.zeros((n, n))
        
        if NETWORKX_AVAILABLE and isinstance(self.graph, nx.Graph):
            for u, v, w in self.graph.edges(data=True):
                weight = w.get('weight', 1.0)
                J[u, v] = weight / 2
                J[v, u] = weight / 2
                h[u] += weight / 2
                h[v] += weight / 2
        
        offset = -np.sum([w.get('weight', 1.0) 
                         for _, _, w in self.graph.edges(data=True)]) / 4
        
        self.ising_h = h
        self.ising_j = J
        self.ising_offset = offset
        
        return h, J, offset
    
    def to_qubo(self) -> Tuple[np.ndarray, float]:
        """转换为QUBO矩阵"""
        h, J, offset = self.to_ising()
        n = len(h)
        
        # Ising到QUBO: s = 2x - 1
        Q = np.zeros((n, n))
        for i in range(n):
            Q[i, i] = 2 * h[i]
            for j in range(i+1, n):
                Q[i, j] = 4 * J[i, j]
                Q[i, i] -= 2 * J[i, j]
                Q[j, j] -= 2 * J[i, j]
        
        qubo_offset = offset + np.sum(h) + np.sum(J)
        
        return Q, qubo_offset


@dataclass
class QAOAResult:
    """QAOA计算结果"""
    success: bool = False
    optimal_value: Optional[float] = None
    optimal_solution: Optional[np.ndarray] = None
    optimal_params: Optional[np.ndarray] = None
    energy_history: List[float] = field(default_factory=list)
    expectation_history: List[float] = field(default_factory=list)
    
    # 统计信息
    n_iterations: int = 0
    n_function_evals: int = 0
    optimization_time: float = 0.0
    
    # 采样结果
    samples: List[Tuple[np.ndarray, float, int]] = field(default_factory=list)
    solution_distribution: Dict[str, float] = field(default_factory=dict)
    top_solutions: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    
    # 电路信息
    circuit_depth: Optional[int] = None
    n_qubits: Optional[int] = None
    n_cnots: Optional[int] = None
    
    # 近似比
    approximation_ratio: Optional[float] = None
    classical_bound: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "optimal_value": self.optimal_value,
            "optimal_solution": self.optimal_solution.tolist() if self.optimal_solution is not None else None,
            "n_iterations": self.n_iterations,
            "n_qubits": self.n_qubits,
            "approximation_ratio": self.approximation_ratio,
            "top_solutions_count": len(self.top_solutions)
        }


class QAOACircuitBuilder:
    """QAOA电路构建器"""
    
    def __init__(self, config: QAOAConfig):
        self.config = config
    
    def build_cost_hamiltonian_circuit(self, ising_j: np.ndarray, 
                                       ising_h: np.ndarray,
                                       gamma: float) -> QuantumCircuit:
        """构建成本哈密顿量电路"""
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        n_qubits = len(ising_h)
        qc = QuantumCircuit(n_qubits, name="U(C,gamma)")
        
        # 二次项: exp(-i gamma J_ij Z_i Z_j)
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                if abs(ising_j[i, j]) > 1e-10:
                    qc.cx(i, j)
                    qc.rz(2 * gamma * ising_j[i, j], j)
                    qc.cx(i, j)
        
        # 线性项: exp(-i gamma h_i Z_i)
        for i in range(n_qubits):
            if abs(ising_h[i]) > 1e-10:
                qc.rz(2 * gamma * ising_h[i], i)
        
        return qc
    
    def build_mixer_circuit(self, n_qubits: int, beta: float,
                           mixer_type: MixerType = MixerType.STANDARD) -> QuantumCircuit:
        """构建Mixer电路"""
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        qc = QuantumCircuit(n_qubits, name="U(B,beta)")
        
        if mixer_type == MixerType.STANDARD:
            # 标准X-mixer: exp(-i beta X_i)
            for i in range(n_qubits):
                qc.rx(2 * beta, i)
        
        elif mixer_type == MixerType.XY:
            # XY-mixer: exp(-i beta (X_i X_j + Y_i Y_j))
            for i in range(n_qubits - 1):
                qc.rx(np.pi/2, i)
                qc.rx(np.pi/2, i+1)
                qc.cx(i, i+1)
                qc.rz(2 * beta, i+1)
                qc.cx(i, i+1)
                qc.rx(-np.pi/2, i)
                qc.rx(-np.pi/2, i+1)
        
        elif mixer_type == MixerType.RING:
            # 环形mixer
            for i in range(n_qubits):
                j = (i + 1) % n_qubits
                qc.rx(beta, i)
                qc.cx(i, j)
                qc.rz(beta, j)
                qc.cx(i, j)
        
        return qc
    
    def build_full_circuit(self, ising_j: np.ndarray, ising_h: np.ndarray,
                          p: int, params: np.ndarray) -> QuantumCircuit:
        """构建完整QAOA电路"""
        n_qubits = len(ising_h)
        qc = QuantumCircuit(n_qubits)
        
        # 初始态: |+\rangle^{\otimes n}
        qc.h(range(n_qubits))
        
        # 参数分割
        betas = params[:p]
        gammas = params[p:]
        
        # p层QAOA
        for i in range(p):
            # 成本哈密顿量
            cost_gate = self.build_cost_hamiltonian_circuit(
                ising_j, ising_h, gammas[i]
            )
            qc.append(cost_gate, range(n_qubits))
            
            # Mixer
            mixer_gate = self.build_mixer_circuit(
                n_qubits, betas[i], self.config.mixer_type
            )
            qc.append(mixer_gate, range(n_qubits))
        
        return qc
    
    def build_parameterized_circuit(self, n_qubits: int, p: int) -> Tuple[QuantumCircuit, List[Parameter]]:
        """构建参数化QAOA电路"""
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        qc = QuantumCircuit(n_qubits)
        
        # 创建参数
        beta_params = [Parameter(f"β_{i}") for i in range(p)]
        gamma_params = [Parameter(f"γ_{i}") for i in range(p)]
        params = beta_params + gamma_params
        
        # 初始态
        qc.h(range(n_qubits))
        
        # p层QAOA (简化版)
        for i in range(p):
            # 成本哈密顿量 - 简化表示
            for j in range(n_qubits):
                qc.rz(gamma_params[i], j)
            for j in range(n_qubits - 1):
                qc.cx(j, j+1)
                qc.rz(gamma_params[i], j+1)
                qc.cx(j, j+1)
            
            # Mixer
            for j in range(n_qubits):
                qc.rx(beta_params[i], j)
        
        return qc, params


class ProblemGenerator:
    """优化问题生成器"""
    
    @staticmethod
    def maxcut(graph: Union[nx.Graph, np.ndarray]) -> OptimizationProblem:
        """生成MaxCut问题"""
        if NETWORKX_AVAILABLE and isinstance(graph, nx.Graph):
            n_nodes = graph.number_of_nodes()
        else:
            n_nodes = len(graph)
        
        return OptimizationProblem(
            problem_type=OptimizationType.MAXCUT,
            n_variables=n_nodes,
            n_constraints=0,
            graph=graph
        )
    
    @staticmethod
    def graph_coloring(graph: nx.Graph, n_colors: int) -> OptimizationProblem:
        """生成图着色问题"""
        n_nodes = graph.number_of_nodes()
        n_variables = n_nodes * n_colors  # one-hot编码
        
        return OptimizationProblem(
            problem_type=OptimizationType.GRAPH_COLORING,
            n_variables=n_variables,
            n_constraints=n_nodes,  # 每个节点必须有一个颜色
            graph=graph,
            metadata={"n_colors": n_colors, "n_nodes": n_nodes}
        )
    
    @staticmethod
    def tsp(distance_matrix: np.ndarray) -> OptimizationProblem:
        """生成旅行商问题"""
        n_cities = len(distance_matrix)
        n_variables = n_cities * n_cities  # 位置×城市
        
        return OptimizationProblem(
            problem_type=OptimizationType.TSP,
            n_variables=n_variables,
            n_constraints=2 * n_cities,  # 每个城市访问一次，每个位置一个城市
            costs=distance_matrix,
            metadata={"n_cities": n_cities}
        )
    
    @staticmethod
    def random_maxcut(n_nodes: int, edge_prob: float = 0.5,
                     seed: Optional[int] = None) -> OptimizationProblem:
        """生成随机MaxCut问题"""
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX required")
        
        if seed is not None:
            np.random.seed(seed)
        
        graph = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=seed)
        
        # 添加随机权重
        for u, v in graph.edges():
            graph[u][v]['weight'] = np.random.uniform(0.5, 2.0)
        
        return ProblemGenerator.maxcut(graph)
    
    @staticmethod
    def regular_graph_maxcut(n_nodes: int, degree: int,
                            seed: Optional[int] = None) -> OptimizationProblem:
        """生成正则图MaxCut问题"""
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX required")
        
        graph = nx.random_regular_graph(degree, n_nodes, seed=seed)
        
        for u, v in graph.edges():
            graph[u][v]['weight'] = 1.0
        
        return ProblemGenerator.maxcut(graph)


class MaterialDesignOptimizer:
    """材料设计优化器 - 使用QAOA进行材料发现"""
    
    def __init__(self, config: QAOAConfig):
        self.config = config
    
    def optimize_alloy_composition(self, elements: List[str],
                                   target_properties: Dict[str, float],
                                   constraints: Optional[Dict] = None) -> QAOAResult:
        """
        优化合金成分
        
        Parameters:
            elements: 可选元素列表
            target_properties: 目标性质 (如 {"band_gap": 1.5, "bulk_modulus": 150})
            constraints: 约束条件
            
        Returns:
            QAOA优化结果
        """
        n_elements = len(elements)
        
        # 构建性质预测模型 (简化版)
        def property_model(composition):
            # 占位符 - 实际应使用ML模型或DFT计算
            return np.random.random()
        
        # 构建QUBO
        Q = self._build_alloy_qubo(n_elements, target_properties)
        
        problem = OptimizationProblem(
            problem_type=OptimizationType.MATERIAL_DESIGN,
            n_variables=n_elements,
            metadata={"elements": elements, "targets": target_properties}
        )
        
        # 转换为Ising
        h, J, offset = problem.to_ising()
        
        # 运行QAOA
        qaoa = QAOAInterface(self.config)
        return qaoa.solve_ising(h, J, offset)
    
    def _build_alloy_qubo(self, n_elements: int, 
                         target_properties: Dict[str, float]) -> np.ndarray:
        """构建合金QUBO矩阵"""
        Q = np.zeros((n_elements, n_elements))
        
        # 对角项：单个元素的贡献
        for i in range(n_elements):
            Q[i, i] = np.random.uniform(-1, 1)  # 占位符
        
        # 非对角项：元素间相互作用
        for i in range(n_elements):
            for j in range(i+1, n_elements):
                Q[i, j] = np.random.uniform(-0.5, 0.5)
        
        return Q
    
    def optimize_defect_configuration(self, lattice_size: Tuple[int, int, int],
                                     defect_types: List[str],
                                     target_formation_energy: float) -> QAOAResult:
        """
        优化缺陷构型
        
        Parameters:
            lattice_size: 晶格大小 (nx, ny, nz)
            defect_types: 缺陷类型列表
            target_formation_energy: 目标形成能
            
        Returns:
            QAOA优化结果
        """
        n_sites = np.prod(lattice_size)
        n_types = len(defect_types)
        n_variables = n_sites * n_types
        
        # 构建问题
        problem = OptimizationProblem(
            problem_type=OptimizationType.MATERIAL_DESIGN,
            n_variables=n_variables,
            metadata={
                "lattice_size": lattice_size,
                "defect_types": defect_types,
                "target_energy": target_formation_energy
            }
        )
        
        h, J, offset = self._build_defect_ising(lattice_size, defect_types)
        
        qaoa = QAOAInterface(self.config)
        return qaoa.solve_ising(h, J, offset)
    
    def _build_defect_ising(self, lattice_size: Tuple[int, ...],
                           defect_types: List[str]) -> Tuple[np.ndarray, np.ndarray, float]:
        """构建缺陷Ising模型"""
        n_sites = np.prod(lattice_size)
        n_types = len(defect_types)
        n_vars = n_sites * n_types
        
        h = np.random.uniform(-1, 1, n_vars)
        J = np.zeros((n_vars, n_vars))
        
        # 添加最近邻相互作用
        # 简化实现
        
        offset = 0.0
        return h, J, offset


class QAOAInterface:
    """QAOA主接口类"""
    
    def __init__(self, config: Optional[QAOAConfig] = None):
        self.config = config or QAOAConfig()
        self.circuit_builder = QAOACircuitBuilder(self.config)
        self._setup_optimizer()
        self._setup_backend()
    
    def _setup_optimizer(self):
        """设置经典优化器"""
        optimizers = {
            "COBYLA": COBYLA if QISKIT_AVAILABLE else None,
            "L_BFGS_B": L_BFGS_B if QISKIT_AVAILABLE else None,
            "SLSQP": SLSQP if QISKIT_AVAILABLE else None,
            "NFT": NFT if QISKIT_AVAILABLE else None
        }
        
        opt_class = optimizers.get(self.config.optimizer, COBYLA if QISKIT_AVAILABLE else None)
        
        if opt_class:
            self.optimizer = opt_class(maxiter=self.config.max_iterations)
        else:
            from scipy.optimize import minimize
            self.optimizer = None
            self.scipy_optimizer = minimize
    
    def _setup_backend(self):
        """设置量子后端"""
        if QISKIT_AVAILABLE:
            from qiskit import Aer
            if self.config.backend == "statevector":
                self.backend = Aer.get_backend('statevector_simulator')
            else:
                self.backend = Aer.get_backend('qasm_simulator')
            self.estimator = Estimator()
        else:
            self.backend = None
            self.estimator = None
    
    def solve(self, problem: OptimizationProblem) -> QAOAResult:
        """
        求解优化问题
        
        Parameters:
            problem: 优化问题
            
        Returns:
            QAOA结果
        """
        # 转换为Ising模型
        h, J, offset = problem.to_ising()
        return self.solve_ising(h, J, offset)
    
    def solve_ising(self, h: np.ndarray, J: np.ndarray,
                   offset: float = 0.0) -> QAOAResult:
        """
        求解Ising模型
        
        Parameters:
            h: 线性项系数
            J: 二次项系数
            offset: 能量偏移
            
        Returns:
            QAOA结果
        """
        result = QAOAResult()
        n_qubits = len(h)
        result.n_qubits = n_qubits
        
        # 初始化参数
        p = self.config.p_layers
        init_params = self._initialize_parameters(p)
        
        # 定义能量函数
        def energy_fn(params):
            return self._compute_expectation(h, J, params)
        
        # 优化参数
        import time
        start_time = time.time()
        
        if self.optimizer and QISKIT_AVAILABLE:
            # 使用Qiskit优化器
            opt_result = self.optimizer.minimize(energy_fn, init_params)
            optimal_params = opt_result.x
            result.n_iterations = opt_result.nit
            result.n_function_evals = opt_result.nfev
        else:
            # 使用SciPy优化器
            opt_result = self.scipy_optimizer(
                energy_fn, init_params, method='COBYLA',
                options={'maxiter': self.config.max_iterations}
            )
            optimal_params = opt_result.x
            result.n_iterations = opt_result.get('nit', 0)
            result.n_function_evals = opt_result.get('nfev', 0)
        
        result.optimization_time = time.time() - start_time
        result.optimal_params = optimal_params
        
        # 采样最优解
        samples = self._sample_solutions(h, J, optimal_params)
        result.samples = samples
        
        # 提取最优解
        if samples:
            best_sample = max(samples, key=lambda x: x[1])
            result.optimal_solution = best_sample[0]
            result.optimal_value = -best_sample[1] + offset  # 转回原始问题
            result.success = True
        
        # 计算电路信息
        circuit = self.circuit_builder.build_full_circuit(
            J, h, p, optimal_params
        )
        result.circuit_depth = circuit.decompose().depth()
        result.n_cnots = circuit.decompose().num_nonlocal_gates()
        
        # 计算近似比
        classical_result = self._solve_classical(h, J, offset)
        if classical_result is not None:
            result.classical_bound = classical_result
            if result.optimal_value is not None:
                result.approximation_ratio = result.optimal_value / classical_result
        
        return result
    
    def _initialize_parameters(self, p: int) -> np.ndarray:
        """初始化QAOA参数"""
        init_type = self.config.init_params
        
        if init_type == "random":
            betas = np.random.uniform(
                self.config.init_beta_range[0],
                self.config.init_beta_range[1],
                p
            )
            gammas = np.random.uniform(
                self.config.init_gamma_range[0],
                self.config.init_gamma_range[1],
                p
            )
        
        elif init_type == "zeros":
            betas = np.zeros(p)
            gammas = np.zeros(p)
        
        elif init_type == "linear_ramp":
            # 线性递增策略
            betas = np.linspace(0.1, np.pi/4, p)
            gammas = np.linspace(0.1, np.pi/2, p)
        
        else:
            betas = np.zeros(p)
            gammas = np.zeros(p)
        
        return np.concatenate([betas, gammas])
    
    def _compute_expectation(self, h: np.ndarray, J: np.ndarray,
                            params: np.ndarray) -> float:
        """计算期望值 (能量)"""
        p = self.config.p_layers
        n_qubits = len(h)
        
        if QISKIT_AVAILABLE and self.estimator:
            # 使用Qiskit Estimator
            circuit = self.circuit_builder.build_full_circuit(
                J, h, p, params
            )
            
            # 构建哈密顿量
            from qiskit.quantum_info import SparsePauliOp
            
            pauli_list = []
            coeff_list = []
            
            # Z项
            for i in range(n_qubits):
                if abs(h[i]) > 1e-10:
                    paulis = ['I'] * n_qubits
                    paulis[i] = 'Z'
                    pauli_list.append(''.join(paulis))
                    coeff_list.append(h[i])
            
            # ZZ项
            for i in range(n_qubits):
                for j in range(i+1, n_qubits):
                    if abs(J[i, j]) > 1e-10:
                        paulis = ['I'] * n_qubits
                        paulis[i] = 'Z'
                        paulis[j] = 'Z'
                        pauli_list.append(''.join(paulis))
                        coeff_list.append(J[i, j])
            
            if pauli_list:
                hamiltonian = SparsePauliOp(pauli_list, coeff_list)
                job = self.estimator.run([circuit], [hamiltonian])
                energy = job.result().values[0]
            else:
                energy = 0.0
        
        else:
            # 使用状态向量模拟
            energy = self._statevector_expectation(h, J, params)
        
        return energy
    
    def _statevector_expectation(self, h: np.ndarray, J: np.ndarray,
                                 params: np.ndarray) -> float:
        """使用状态向量计算期望 (简化版)"""
        n_qubits = len(h)
        
        # 生成所有可能的比特串 (2^n个)
        n_states = 2 ** n_qubits
        
        # 简化计算 - 实际应该模拟量子电路
        energy = 0.0
        
        # 计算Ising能量
        for state in range(n_states):
            bits = [(state >> i) & 1 for i in range(n_qubits)]
            spins = [2*b - 1 for b in bits]
            
            # 计算该状态的能量
            state_energy = 0.0
            for i in range(n_qubits):
                state_energy += h[i] * spins[i]
            for i in range(n_qubits):
                for j in range(i+1, n_qubits):
                    state_energy += J[i, j] * spins[i] * spins[j]
            
            # 简化：假设均匀分布
            energy += state_energy / n_states
        
        return energy
    
    def _sample_solutions(self, h: np.ndarray, J: np.ndarray,
                         params: np.ndarray) -> List[Tuple[np.ndarray, float, int]]:
        """采样解"""
        n_qubits = len(h)
        p = self.config.p_layers
        n_samples = self.config.n_samples
        
        samples = []
        
        # 简化版：枚举低能量状态
        n_states = min(2 ** n_qubits, 1024)  # 限制枚举数量
        
        state_energies = []
        for state in range(n_states):
            bits = np.array([(state >> i) & 1 for i in range(n_qubits)])
            spins = 2 * bits - 1
            
            energy = 0.0
            for i in range(n_qubits):
                energy += h[i] * spins[i]
            for i in range(n_qubits):
                for j in range(i+1, n_qubits):
                    energy += J[i, j] * spins[i] * spins[j]
            
            state_energies.append((bits, -energy, 1))  # 存储比特、价值、计数
        
        # 按能量排序并选择前k个
        state_energies.sort(key=lambda x: x[1], reverse=True)
        samples = state_energies[:self.config.top_k_solutions]
        
        return samples
    
    def _solve_classical(self, h: np.ndarray, J: np.ndarray,
                        offset: float) -> Optional[float]:
        """经典求解 (用于计算近似比)"""
        n_qubits = len(h)
        
        if n_qubits <= 20:
            # 精确求解
            best_energy = float('inf')
            for state in range(2 ** n_qubits):
                spins = np.array([2*((state >> i) & 1) - 1 for i in range(n_qubits)])
                energy = offset
                for i in range(n_qubits):
                    energy += h[i] * spins[i]
                for i in range(n_qubits):
                    for j in range(i+1, n_qubits):
                        energy += J[i, j] * spins[i] * spins[j]
                best_energy = min(best_energy, energy)
            return best_energy
        
        return None
    
    def solve_maxcut(self, graph: Union[nx.Graph, np.ndarray]) -> QAOAResult:
        """
        求解MaxCut问题
        
        Parameters:
            graph: 图 (NetworkX图或邻接矩阵)
            
        Returns:
            QAOA结果
        """
        problem = ProblemGenerator.maxcut(graph)
        return self.solve(problem)
    
    def solve_tsp(self, distance_matrix: np.ndarray) -> QAOAResult:
        """
        求解TSP问题
        
        Parameters:
            distance_matrix: 城市间距离矩阵
            
        Returns:
            QAOA结果
        """
        problem = ProblemGenerator.tsp(distance_matrix)
        return self.solve(problem)
    
    def multi_objective_optimize(self, problems: List[OptimizationProblem],
                                weights: Optional[List[float]] = None) -> QAOAResult:
        """
        多目标优化
        
        Parameters:
            problems: 优化问题列表
            weights: 目标权重
            
        Returns:
            QAOA结果
        """
        if weights is None:
            weights = [1.0 / len(problems)] * len(problems)
        
        # 组合多个目标
        combined_h = np.zeros_like(problems[0].ising_h)
        combined_J = np.zeros_like(problems[0].ising_j)
        combined_offset = 0.0
        
        for problem, weight in zip(problems, weights):
            h, J, offset = problem.to_ising()
            combined_h += weight * h
            combined_J += weight * J
            combined_offset += weight * offset
        
        return self.solve_ising(combined_h, combined_J, combined_offset)


class SpinGlassSolver:
    """自旋玻璃基态求解器"""
    
    def __init__(self, config: Optional[QAOAConfig] = None):
        self.config = config or QAOAConfig()
        self.qaoa = QAOAInterface(self.config)
    
    def solve_2d_ising(self, size: Tuple[int, int],
                      coupling: float = 1.0,
                      field: float = 0.0,
                      boundaries: str = "periodic") -> QAOAResult:
        """
        求解2D Ising模型
        
        Parameters:
            size: 网格大小 (Lx, Ly)
            coupling: 耦合强度 J
            field: 外磁场 h
            boundaries: 边界条件 ("periodic", "open")
            
        Returns:
            QAOA结果
        """
        Lx, Ly = size
        n_spins = Lx * Ly
        
        # 构建哈密顿量
        h = np.full(n_spins, field)
        J = np.zeros((n_spins, n_spins))
        
        # 最近邻相互作用
        for i in range(Lx):
            for j in range(Ly):
                idx = i * Ly + j
                
                # x方向邻居
                i_next = (i + 1) % Lx if boundaries == "periodic" else i + 1
                if i_next < Lx:
                    idx_next = i_next * Ly + j
                    J[idx, idx_next] = coupling
                    J[idx_next, idx] = coupling
                
                # y方向邻居
                j_next = (j + 1) % Ly if boundaries == "periodic" else j + 1
                if j_next < Ly:
                    idx_next = i * Ly + j_next
                    J[idx, idx_next] = coupling
                    J[idx_next, idx] = coupling
        
        return self.qaoa.solve_ising(h, J)
    
    def solve_3d_heisenberg(self, size: Tuple[int, int, int],
                           coupling: float = 1.0) -> QAOAResult:
        """
        求解3D Heisenberg模型 (简化版)
        
        Parameters:
            size: 网格大小 (Lx, Ly, Lz)
            coupling: 交换耦合 J
            
        Returns:
            QAOA结果
        """
        Lx, Ly, Lz = size
        n_spins = Lx * Ly * Lz
        
        # 简化为Ising模型 (忽略XY和YZ分量)
        h = np.zeros(n_spins)
        J = np.zeros((n_spins, n_spins))
        
        def get_idx(x, y, z):
            return x * Ly * Lz + y * Lz + z
        
        for x in range(Lx):
            for y in range(Ly):
                for z in range(Lz):
                    idx = get_idx(x, y, z)
                    
                    # 最近邻
                    for dx, dy, dz in [(1,0,0), (0,1,0), (0,0,1)]:
                        nx, ny, nz = (x + dx) % Lx, (y + dy) % Ly, (z + dz) % Lz
                        nidx = get_idx(nx, ny, nz)
                        J[idx, nidx] = coupling
        
        return self.qaoa.solve_ising(h, J)
    
    def solve_sherrington_kirkpatrick(self, n_spins: int,
                                     seed: Optional[int] = None) -> QAOAResult:
        """
        求解Sherrington-Kirkpatrick (SK) 模型
        
        Parameters:
            n_spins: 自旋数
            seed: 随机种子
            
        Returns:
            QAOA结果
        """
        if seed is not None:
            np.random.seed(seed)
        
        h = np.zeros(n_spins)
        J = np.random.randn(n_spins, n_spins) / np.sqrt(n_spins)
        J = (J + J.T) / 2  # 对称化
        np.fill_diagonal(J, 0)
        
        return self.qaoa.solve_ising(h, J)
    
    def solve_viana_bray(self, n_spins: int, coordination: int = 3,
                        seed: Optional[int] = None) -> QAOAResult:
        """
        求解Viana-Bray模型 (有限连接自旋玻璃)
        
        Parameters:
            n_spins: 自旋数
            coordination: 配位数
            seed: 随机种子
            
        Returns:
            QAOA结果
        """
        if seed is not None:
            np.random.seed(seed)
        
        h = np.random.randn(n_spins) * 0.5
        J = np.zeros((n_spins, n_spins))
        
        # 随机连接
        for i in range(n_spins):
            for _ in range(coordination // 2):
                j = np.random.randint(0, n_spins)
                if i != j:
                    J[i, j] = np.random.randn()
                    J[j, i] = J[i, j]
        
        return self.qaoa.solve_ising(h, J)


def demo_maxcut():
    """MaxCut示例"""
    print("=" * 60)
    print("MaxCut Demo")
    print("=" * 60)
    
    if not NETWORKX_AVAILABLE:
        print("NetworkX not available")
        return
    
    # 创建简单图
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    
    config = QAOAConfig(p_layers=2, optimizer="COBYLA")
    qaoa = QAOAInterface(config)
    
    result = qaoa.solve_maxcut(graph)
    
    print(f"Optimal cut value: {result.optimal_value}")
    print(f"Optimal solution: {result.optimal_solution}")
    print(f"Number of qubits: {result.n_qubits}")
    print(f"Circuit depth: {result.circuit_depth}")
    print(f"Approximation ratio: {result.approximation_ratio}")


def demo_spin_glass():
    """自旋玻璃示例"""
    print("\n" + "=" * 60)
    print("Spin Glass Demo (2D Ising Model)")
    print("=" * 60)
    
    solver = SpinGlassSolver(QAOAConfig(p_layers=3))
    
    result = solver.solve_2d_ising(size=(3, 3), coupling=-1.0, field=0.5)
    
    print(f"Ground state energy: {result.optimal_value:.6f}")
    print(f"Optimal configuration: {result.optimal_solution}")
    print(f"Number of iterations: {result.n_iterations}")


def demo_material_design():
    """材料设计优化示例"""
    print("\n" + "=" * 60)
    print("Material Design Optimization Demo")
    print("=" * 60)
    
    config = QAOAConfig(p_layers=2)
    optimizer = MaterialDesignOptimizer(config)
    
    elements = ["Fe", "Ni", "Co", "Cr"]
    target_properties = {
        "band_gap": 0.0,  # 金属
        "bulk_modulus": 180.0,
        "magnetic_moment": 2.2
    }
    
    result = optimizer.optimize_alloy_composition(
        elements, target_properties
    )
    
    print(f"Optimal composition found")
    print(f"Objective value: {result.optimal_value:.4f}")
    print(f"Selected elements: {result.optimal_solution}")


if __name__ == "__main__":
    demo_maxcut()
    demo_spin_glass()
    demo_material_design()
