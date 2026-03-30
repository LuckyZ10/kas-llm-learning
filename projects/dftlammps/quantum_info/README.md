# 量子计算与量子材料模拟模块

## 概述

本模块实现了量子计算在材料科学中的应用，包括NISQ算法、量子纠错和量子材料模拟。

## 模块结构

```
dftlammps/quantum_info/
├── __init__.py                    # 模块初始化
├── quantum_chemistry_qc.py        # 量子化学量子计算 (600+ 行)
├── quantum_materials.py           # 量子材料模拟 (800+ 行)
└── examples/
    ├── htc_superconductor.py      # 高温超导模拟 (400+ 行)
    ├── magnetic_phase_diagram.py  # 量子磁性相图 (500+ 行)
    └── topological_qc.py          # 拓扑量子计算 (500+ 行)

dftlammps/quantum_error/
├── __init__.py                    # 模块初始化
├── surface_code.py                # 表面码实现 (500+ 行)
└── (完整纠错模块在__init__.py中)   # (600+ 行)
```

## 主要功能

### 1. 量子化学量子计算 (`quantum_chemistry_qc.py`)

#### VQE (变分量子特征求解器)
```python
from dftlammps.quantum_info import VQE, UCCSD, h2_molecule_hamiltonian

# 创建H2分子哈密顿量
hamiltonian = h2_molecule_hamiltonian(bond_length=0.74)

# 创建UCCSD ansatz
uccsd = UCCSD(n_orbitals=2, n_electrons=2)

# 运行VQE
vqe = VQE(hamiltonian=hamiltonian, ansatz=uccsd)
result = vqe.optimize()
print(f"基态能量: {result['energy']:.8f} Ha")
```

#### 量子相位估计 (QPE)
```python
from dftlammps.quantum_info import QuantumPhaseEstimation

# 创建酉算符
U = np.array([[np.exp(2j*np.pi*0.3), 0],
              [0, np.exp(-2j*np.pi*0.3)]])

# 估计相位
qpe = QuantumPhaseEstimation(n_counting_qubits=4, unitary=U)
phase = qpe.estimate_phase(eigenstate=np.array([1.0, 0.0]))
```

#### 错误缓解
```python
from dftlammps.quantum_info import ErrorMitigation

# 零噪声外推
mitigation = ErrorMitigation(method="zero_noise_extrapolation")
corrected_energy = mitigation.apply(circuit_executor, 
                                     scale_factors=[1.0, 2.0, 3.0])
```

### 2. 量子材料模拟 (`quantum_materials.py`)

#### Hubbard模型
```python
from dftlammps.quantum_info import HubbardModel, Lattice

# 创建4x4方格晶格
lattice = Lattice.create_square(4, 4, periodic=True)

# 创建Hubbard模型
hubbard = HubbardModel(lattice, t=1.0, U=4.0, n_electrons=8)

# 计算基态
energy = hubbard.get_ground_state_energy()
print(f"基态能量: {energy:.6f}")

# 计算双占据概率
energies, states = hubbard.diagonalize(n_states=1)
double_occ = hubbard.compute_double_occupancy(states[:, 0])
```

#### 自旋系统
```python
from dftlammps.quantum_info import SpinSystem

# Heisenberg模型
heisenberg = SpinSystem(lattice, model_type="heisenberg", J=1.0)

# 计算基态
energies, states = heisenberg.diagonalize(n_states=1)

# 计算磁化强度
mag_z = heisenberg.compute_magnetization(states[:, 0], 'z')
mag_s = heisenberg.compute_staggered_magnetization(states[:, 0])
```

#### 拓扑不变量
```python
from dftlammps.quantum_info import HaldaneModel, TopologicalInvariant

# Haldane模型
haldane = HaldaneModel(t=1.0, t2=0.1, phi=np.pi/2, M=0.0)

# 计算Chern数
chern = haldane.compute_chern_number(nk=100)
print(f"Chern数: {chern:.0f}")
```

### 3. 量子纠错 (`quantum_error/`)

#### 表面码
```python
from dftlammps.quantum_error import SurfaceCode, LogicalQubit

# 创建d=3的表面码
code = SurfaceCode(distance=3)

# 估计逻辑错误率
result = code.get_logical_error_rate(physical_error_rate=0.001)
print(f"逻辑错误率: {result['logical_error_rate']:.6f}")

# 逻辑量子比特操作
lq = LogicalQubit(code)
lq.initialize("|+>")
lq.apply_logical_hadamard()
measurement = lq.measure_logical_z(shots=1000)
```

#### 表面码模拟
```python
from dftlammps.quantum_error import SurfaceCodeSimulation

# 创建模拟器
sim = SurfaceCodeSimulation(distance=5, physical_error_rate=0.01)

# 运行内存实验
result = sim.run_memory_experiment(n_cycles=1000)
print(f"逻辑错误率: {result['logical_error_rate']:.4f}")
print(f"逻辑量子比特寿命: {result['lifetime']:.1f} 周期")
```

## 应用示例

### 高温超导模拟
```python
from dftlammps.quantum_info.examples.htc_superconductor import HighTcSuperconductor

# 创建铜氧化物模型
htc = HighTcSuperconductor(
    lattice_size=(4, 4),
    t=1.0, U=8.0, doping=0.125
)

# 计算配对关联
pairing = htc.compute_pairing_correlation(r_max=3)

# 分析激发谱
spectrum = htc.analyze_excitation_spectrum()
```

### 量子磁性相图
```python
from dftlammps.quantum_info.examples.magnetic_phase_diagram import QuantumMagneticPhaseDiagram

# 计算Heisenberg相图
qmpd = QuantumMagneticPhaseDiagram(lattice, model_type="heisenberg")
phase_diagram = qmpd.compute_heisenberg_phase_diagram(J_range, h_range)

# 计算XXZ相图
xxz_diagram = qmpd.compute_xxz_phase_diagram(J_range, Delta_range)
```

### 拓扑量子计算
```python
from dftlammps.quantum_info.examples.topological_qc import TopologicalQubit, BraidingCircuit

# 创建拓扑量子比特
tq = TopologicalQubit()

# 应用编织门
tq.braid(0, 1)
tq.apply_hadamard()

# 测量
result = tq.measure()
```

## 物理模型

### 支持的模型

| 模型 | 描述 | 参数 |
|------|------|------|
| Hubbard | 强关联电子 | t, U, μ |
| Heisenberg | 各向同性自旋 | J, h |
| XXZ | 各向异性自旋 | J, Δ |
| Ising | 易轴自旋 | J, h |
| XY | 易面自旋 | J |
| t-J | 强关联极限 | t, J |
| Haldane | 拓扑能带 | t, t₂, φ, M |
| Kitaev | 量子自旋液体 | Jₓ, Jᵧ, Jᵥ |

### 量子纠错码

| 码类型 | 参数 | 容错阈值 |
|--------|------|----------|
| 表面码 | [[d²+(d-1)², 1, d]] | ~1% |
| Steane码 | [[7, 1, 3]] | ~0.1% |

## 技术细节

### 费米子到量子比特映射

- **Jordan-Wigner变换**: 简单但非局域
- **Bravyi-Kitaev变换**: 节省量子比特，局域性更好

### Ansatz类型

- **UCCSD**: 酉耦合簇单双激发
- **k-UpCCGSD**: 广义配对激发
- **ADAPT-VQE**: 自适应ansatz

### 解码算法

- **MWPM**: 最小权重完美匹配 (最优但慢)
- **Union-Find**: 快速近似解码
- **BP**: 置信传播 (适合LDPC码)

## 性能考虑

### 经典模拟限制
- Hubbard模型: ~16个格点 (32量子比特)
- 自旋系统: ~20个自旋
- 表面码: 码距d≤9

### NISQ设备考虑
- 电路深度: 受相干时间限制
- 错误率: ~0.1-1%
- 连接性: 最近邻最优

## 参考文献

1. Peruzzo et al., "A variational eigenvalue solver...", Nat. Commun. (2014)
2. Kitaev, "Fault-tolerant quantum computation...", Ann. Phys. (2003)
3. Fowler et al., "Surface codes...", Phys. Rev. A (2012)
4. Nayak et al., "Non-Abelian anyons...", Rev. Mod. Phys. (2008)

## 版本信息

- 版本: 1.0.0
- 作者: DFT-Team
- 日期: 2025-03
