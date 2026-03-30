# 量子纠错模块

## 概述

本模块实现量子纠错码，特别是表面码（Surface Code）及其相关算法。

## 主要组件

### 表面码 (`surface_code.py`)

Kitaev提出的拓扑量子纠错码，具有高容错阈值 (~1%)。

```python
from dftlammps.quantum_error import SurfaceCode, LogicalQubit

# 创建d=3的表面码
code = SurfaceCode(distance=3)

# 估计逻辑错误率
result = code.get_logical_error_rate(physical_error_rate=0.001)
print(f"逻辑错误率: {result['logical_error_rate']:.6f}")

# 逻辑量子比特操作
lq = LogicalQubit(code)
lq.initialize("|0_L>")
lq.apply_logical_hadamard()
measurement = lq.measure_logical_z(shots=1000)
```

### 解码器

#### MWPM解码器 (最小权重完美匹配)
- 使用blossom算法
- 最优解码但计算成本较高
- 适合离线纠错

#### Union-Find解码器
- 近乎线性时间复杂度
- 适合实时解码
- 性能接近MWPM

```python
from dftlammps.quantum_error import MWPM_Decoder, UnionFindDecoder
from dftlammps.quantum_error.surface_code import SurfaceCodeLattice

lattice = SurfaceCodeLattice(distance=5)
decoder = MWPM_Decoder(lattice)  # 或 UnionFindDecoder(lattice)
```

### 错误率估计

```python
from dftlammps.quantum_error import ErrorRateEstimator

estimator = ErrorRateEstimator(n_qubits=100)

# 估计电路错误率
circuit = [
    {'gate': 'H', 'qubits': [0], 'duration': 10},
    {'gate': 'CNOT', 'qubits': [0, 1], 'duration': 20},
]
error_rate = estimator.estimate_circuit_error(circuit)

# 预测逻辑量子比特寿命
lifetime = estimator.predict_logical_lifetime(
    error_correction_code=code,
    physical_error_rate=0.001
)
```

## 表面码结构

### 格点布局 (d=3)

```
Z - D - Z - D - Z
|       |       |
D   X   D   X   D
|       |       |
Z - D - Z - D - Z
|       |       |
D   X   D   X   D
|       |       |
Z - D - Z - D - Z
```

- **D** = 数据量子比特
- **X** = X型稳定子 (星形)
- **Z** = Z型稳定子 (面)

### 编码参数

| 码距 | 数据量子比特 | 辅助量子比特 | 总量子比特 |
|------|-------------|-------------|-----------|
| 3    | 13          | 12          | 25        |
| 5    | 41          | 40          | 81        |
| 7    | 85          | 84          | 169       |
| 9    | 145         | 144         | 289       |

## 纠错周期

```python
from dftlammps.quantum_error.surface_code import SurfaceCodeSimulation

# 创建模拟器
sim = SurfaceCodeSimulation(
    distance=5,
    physical_error_rate=0.01,
    measurement_error_rate=0.02
)

# 运行内存实验
result = sim.run_memory_experiment(n_cycles=1000)

print(f"逻辑错误率: {result['logical_error_rate']:.4f}")
print(f"逻辑量子比特寿命: {result['lifetime']:.1f} 周期")
```

纠错周期步骤：
1. **初始化**: 准备逻辑态
2. **空闲**: 等待期间可能发生错误
3. **综合征提取**: 测量所有稳定子
4. **解码**: 确定最可能的错误模式
5. **纠正**: 应用纠正操作
6. **验证**: 检查逻辑态完整性

## 容错阈值

表面码的容错阈值约为 **~1%**，这意味着：
- 物理错误率 < 1%: 逻辑错误率随码距指数降低
- 物理错误率 > 1%: 逻辑错误率随码距增加

```python
from dftlammps.quantum_error import compare_code_performance

# 比较不同码距的性能
results = compare_code_performance(
    distances=[3, 5, 7],
    error_rates=[0.001, 0.003, 0.005, 0.01]
)
```

## 支持的纠错码

### 表面码
- 拓扑码
- 最近邻相互作用
- 高容错阈值

### Steane码 [[7,1,3]]
- 最小的CSS码
- 能纠正所有单比特错误
- 需要长程相互作用

### 级联码
```python
from dftlammps.quantum_error import ConcatenatedCode, SteaneCode

# 级联Steane码和表面码
inner = SurfaceCode(distance=3)
outer = SteaneCode()
concatenated = ConcatenatedCode(inner, outer)
```

## 错误模型

### 支持的噪声类型
- **比特翻转错误** (X)
- **相位翻转错误** (Z)
- **联合错误** (Y = iXZ)
- **测量错误**
- **空闲错误**

### 错误率参数
```python
error_model = {
    'single_qubit_gate_error': 0.001,
    'two_qubit_gate_error': 0.01,
    'measurement_error': 0.005,
    'idle_error': 0.0001,
    'readout_error_0': 0.02,
    'readout_error_1': 0.02
}
```

## 性能优化

### 解码速度
- Union-Find: ~O(n) 复杂度
- MWPM: ~O(n³) 复杂度

### 内存使用
- 综合征历史存储
- 错误模式缓存

## 参考文献

1. Kitaev, "Fault-tolerant quantum computation by anyons", Ann. Phys. (2003)
2. Fowler et al., "Surface codes: Towards practical large-scale quantum computation", Phys. Rev. A (2012)
3. Dennis et al., "Topological quantum memory", J. Math. Phys. (2002)
4. Delfosse and Tillich, "A decoding algorithm for CSS codes using the X/Z correlations", arXiv:1401.6973
