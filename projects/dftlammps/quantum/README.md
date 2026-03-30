# DFT-LAMMPS 量子计算模块

量子-经典混合计算框架，集成量子计算与经典DFT/MD模拟。

## 功能概述

本模块提供以下核心功能：

1. **量子电路接口** - 统一接口支持IBM Qiskit、Google Cirq、PennyLane
2. **VQE求解器** - 变分量子特征值求解器用于分子电子结构
3. **量子机器学习** - 量子核方法、量子神经网络用于势能面拟合
4. **量子动力学** - 量子动力学与经典MD耦合

## 安装

### 依赖项

```bash
# 必需
numpy>=1.20.0
scipy>=1.7.0

# 量子计算后端（至少安装一个）
pip install qiskit qiskit-aer          # IBM Qiskit
pip install cirq                        # Google Cirq
pip install pennylane                   # PennyLane

# 可选（用于完整功能）
pip install pyscf                       # 量子化学计算
pip install openfermion                 # 费米子-量子比特映射
pip install matplotlib                  # 绘图
pip install scikit-learn                # 机器学习工具
```

### 模块结构

```
dftlammps/quantum/
├── __init__.py              # 模块初始化
├── quantum_interface.py     # 量子-经典接口层 (~700行)
├── vqe_solver.py           # VQE求解器 (~550行)
├── quantum_ml.py           # 量子机器学习 (~600行)
└── quantum_dynamics.py     # 量子动力学耦合 (~580行)

dftlammps/quantum_examples/
├── quantum_chemistry_demo.py   # 量子化学演示 (~350行)
└── quantum_materials.py        # 量子材料模拟 (~520行)
```

## 快速开始

### 1. 量子电路接口

```python
from dftlammps.quantum import create_quantum_interface, QuantumBackend

# 创建量子接口（自动选择后端）
interface = create_quantum_interface(backend="auto")

# 或指定特定后端
interface = create_quantum_interface(backend="qiskit")

# 创建量子电路
circuit = interface.create_circuit(num_qubits=4, name="bell_state")
circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0, 1])

# 执行
result = interface.execute(circuit)
print(result['counts'])
```

### 2. VQE分子计算

```python
from dftlammps.quantum import run_vqe_for_molecule

# H2分子几何
geometry = [
    ('H', (0.0, 0.0, 0.0)),
    ('H', (0.0, 0.0, 0.74))
]

# 运行VQE
result = run_vqe_for_molecule(
    geometry=geometry,
    basis="sto-3g",
    ansatz="Hardware_Efficient",
    backend="auto"
)

print(f"基态能量: {result['energy']:.6f} Hartree")
```

### 3. 量子机器学习势

```python
from dftlammps.quantum import QuantumPotentialEnergySurface

# 创建量子PES模型
qpes = QuantumPotentialEnergySurface(
    model_type="qnn",  # 或 "kernel_ridge", "gaussian_process"
    num_qubits=6,
    num_layers=3
)

# 拟合数据
configurations = ...  # 分子构型数组
energies = ...        # 对应能量
qpes.fit(configurations, energies)

# 预测能量和力
energy = qpes.predict_energy(new_configuration)
forces = qpes.predict_force(new_configuration)
```

### 4. 混合QM/MM模拟

```python
from dftlammps.quantum import (
    HybridQMMD, QuantumClassicalPartition
)

# 定义量子-经典分区
partition = QuantumClassicalPartition()
partition.add_quantum_region(
    atom_indices=[0, 1],
    num_electrons=2,
    basis="sto-3g"
)
partition.add_classical_region(
    atom_indices=[2, 3, 4, 5],
    force_field="Lennard-Jones"
)

# 初始化混合MD
qmmd = HybridQMMD(
    partition=partition,
    temperature=300.0,
    dt=0.5
)

qmmd.initialize_system(positions, masses)
qmmd.run(n_steps=1000)
```

## 示例演示

### 量子化学演示

```bash
cd dftlammps/quantum_examples
python quantum_chemistry_demo.py
```

包含：
- H2、LiH分子VQE计算
- 势能面扫描
- 激发态计算
- 与经典FCI比较

### 量子材料模拟

```bash
python quantum_materials.py
```

包含：
- Hubbard模型模拟
- 海森堡自旋链
- 量子相变
- 量子自旋液体
- 量子ML材料预测
- 混合QM/MD模拟

## API文档

### QuantumInterface

主接口类，统一管理量子计算后端。

```python
interface = QuantumInterface(
    backend=QuantumBackend.QISKIT,  # 或 "auto", "cirq", "pennylane"
    device=QuantumDevice.STATEVECTOR,
    shots=1024
)
```

### VQESolver

变分量子特征值求解器。

```python
solver = VQESolver(
    quantum_interface=interface,
    ansatz_type="UCCSD",  # 或 "Hardware_Efficient", "Adaptive"
    optimizer="COBYLA",
    max_iterations=1000
)

# 构建哈密顿量
solver.build_hamiltonian(geometry, basis="sto-3g")

# 构建ansatz
solver.build_ansatz(num_electrons=2)

# 优化
result = solver.optimize()
```

### QuantumNeuralNetwork

量子神经网络用于势能面拟合。

```python
qnn = QuantumNeuralNetwork(
    num_qubits=6,
    num_layers=3,
    encoding_type="angle",
    ansatz_type="hardware_efficient"
)

# 训练
history = qnn.fit(X_train, y_train, learning_rate=0.01, epochs=100)

# 预测
predictions = qnn.predict(X_test)
```

### QuantumKernelRidge

量子核岭回归。

```python
qkr = QuantumKernelRidge(
    num_qubits=4,
    alpha=1.0,
    feature_map_type="angle",
    feature_map_reps=2
)

qkr.fit(X_train, y_train)
predictions = qkr.predict(X_test)
```

## 支持的量子后端

| 后端 | 状态 | 说明 |
|------|------|------|
| Qiskit | ✅ | IBM量子计算SDK |
| Cirq | ✅ | Google量子计算框架 |
| PennyLane | ✅ | 跨平台量子机器学习 |
| IBMQ | 🔄 | IBM量子硬件（需API密钥） |

## 性能注意事项

1. **模拟器 vs 真实量子设备**
   - 开发测试使用 `Statevector` 模拟器
   - 噪声模拟使用 `QASM Simulator`
   - 真实硬件需要相应云平台的API密钥

2. **计算成本**
   - VQE需要大量量子电路评估
   - 考虑使用梯度优化而非有限差分
   - ADAPT-VQE可减少电路深度

3. **内存需求**
   - 量子模拟器内存随量子比特数指数增长
   - 20+ qubits建议使用量子硬件或张量网络模拟

## 故障排除

### 导入错误

```python
# 如果PySCF不可用，使用简化哈密顿量
from dftlammps.quantum import MolecularHamiltonian

hamiltonian = MolecularHamiltonian(
    num_qubits=4,
    nuclear_repulsion=0.7
)
```

### 量子后端选择

```python
# 强制使用特定后端
interface = create_quantum_interface(backend="qiskit")

# 或回退到经典模拟
interface = create_quantum_interface(backend="auto")  # 自动选择可用后端
```

## 参考文献

1. Peruzzo et al. (2014) - VQE原始论文
2. Kandala et al. (2017) - 硬件高效ansatz
3. Havlíček et al. (2019) - 量子核方法
4. Mitarai et al. (2018) - 量子电路学习

## 许可

MIT License

## 贡献

欢迎提交Issue和Pull Request！
