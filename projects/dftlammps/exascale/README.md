# Exascale Computing Module

极限计算模块 - 百万原子DFT和极端条件分子动力学模拟

## 概述

本模块提供超大规模并行计算能力，支持百万原子体系的量子力学模拟和极端条件（高压、高温、高应变率）下的分子动力学模拟。

## 模块结构

```
dftlammps/exascale/
├── __init__.py          # 模块接口
├── exascale_dft.py      # 百万原子DFT (800+ lines)
├── extreme_md.py        # 极端条件MD (700+ lines)
└── applications.py      # 应用案例 (800+ lines)

dftlammps/machine_limits/
├── __init__.py                # 模块接口
├── parallel_optimization.py   # 百万核并行优化 (700+ lines)
├── memory_optimization.py     # 内存优化 (550+ lines)
└── checkpoint_restart.py      # 检查点/重启 (700+ lines)
```

## 功能特性

### 1. 百万原子DFT (`exascale_dft.py`)

#### 线性标度DFT方法
- **ONETEP风格**: 非正交广义Wannier函数(NGWFs)
- **CONQUEST方法**: 密度矩阵线性标度方法
- **局部化轨道**: 自适应截断和空间哈希

#### 区域分解并行
- **3D域分解**: 支持任意维度分解
- **幽灵原子交换**: 非阻塞MPI通信
- **动态负载均衡**: 自动检测和重新分配

#### GPU加速线性代数
- **cuBLAS/cuSOLVER**: GPU矩阵运算
- **混合精度**: FP32/FP64自动切换
- **稀疏矩阵**: GPU稀疏矩阵运算

### 2. 极端条件MD (`extreme_md.py`)

#### 冲击波模拟
- **活塞法**: 移动活塞产生冲击波
- **动量镜像法**: NEMD方法
- **双层法**: 冲击波对撞

#### 相变检测
- **Lindemann判据**: 熔化温度判定
- **RDF分析**: 径向分布函数
- **CNA分析**: 通用邻域分析
- **Voronoi分析**: 局部结构识别

#### Hugoniot状态
- Rankine-Hugoniot跳跃条件
- 冲击波速度追踪
- 粒子速度测量

### 3. 并行优化 (`machine_limits/parallel_optimization.py`)

#### 百万核优化
- **拓扑感知映射**: 网络拓扑优化
- **通信模式**: 多种MPI通信优化
- **持久请求**: 预分配MPI请求

#### 负载均衡
- **动态检测**: 实时负载监控
- **原子迁移**: 智能重分配
- **成本估计**: 迁移开销预测

### 4. 内存优化 (`machine_limits/memory_optimization.py`)

#### 核外数组
- **内存映射**: 大数组磁盘交换
- **分块访问**: 局部性优化
- **自动缓存**: LRU缓存策略

#### 压缩管理
- **多算法支持**: zlib, gzip, lz4, zstd, blosc
- **自动选择**: 基于数据特征
- **压缩比统计**: 性能监控

#### 内存池
- **预分配**: 减少分配开销
- **复用机制**: 碎片化管理
- **压力检测**: 自动溢出到磁盘

### 5. 检查点/重启 (`machine_limits/checkpoint_restart.py`)

#### 增量检查点
- **差异存储**: 仅保存变化
- **完整/增量混合**: 定期完整检查点
- **压缩存储**: 减少I/O开销

#### 异步I/O
- **后台写入**: 非阻塞检查点
- **队列管理**: 自动流控
- **错误处理**: 失败重试机制

#### 容错机制
- **心跳检测**: 进程健康监控
- **自动恢复**: 故障自动重启
- **状态重建**: 从检查点恢复

## 应用案例 (`applications.py`)

### 1. 行星内核模拟
- **地球核心**: Fe-Ni合金，360 GPa，6000 K
- **内核-外核边界**: 固液相变检测
- **温度梯度**: 径向温度分布

### 2. 核材料极端条件
- **UO2燃料**: 萤石结构
- **温度斜坡**: 模拟失冷事故
- **辐照损伤**: 位移级联模拟
- **缺陷分析**: 空位/间隙统计

### 3. 小行星撞击
- **超高速冲击**: 20 km/s撞击
- **冲击波传播**: 高压状态追踪
- **弹坑形成**: 形貌分析
- **Hugoniot曲线**: 状态方程提取

## 使用方法

### 基本DFT计算

```python
from dftlammps.exascale import ExascaleDFT, ExascaleDFTConfig, LinearScalingMethod

# 配置
config = ExascaleDFTConfig(
    method=LinearScalingMethod.ONETEP,
    localization_radius=8.0,
    domain_decomposition=(8, 8, 8),
    use_gpu=True
)

# 运行
dft = ExascaleDFT(config)
dft.initialize_system(positions, atomic_numbers, box)
energy = dft.run_scf()
forces = dft.get_forces()
```

### 冲击波模拟

```python
from dftlammps.exascale import ExtremeMD, ExtremeMDConfig, ShockMethod

config = ExtremeMDConfig(
    shock_method=ShockMethod.PISTON,
    piston_velocity=5.0,  # km/s
    target_pressure=100.0  # GPa
)

md = ExtremeMD(config)
md.initialize(positions, velocities, masses, box)
md.run()

# 获取Hugoniot状态
hugoniot = md.get_hugoniot_curve()
```

### 应用案例

```python
from dftlammps.exascale import PlanetaryCoreSimulator, PlanetaryCoreConfig

# 行星内核模拟
config = PlanetaryCoreConfig(
    n_atoms=1000000,
    inner_core_temp=6000.0,
    center_pressure=360.0
)

sim = PlanetaryCoreSimulator(config)
results = sim.run()
```

## 性能特征

| 功能 | 扩展性 | 内存效率 | GPU加速 |
|------|--------|----------|---------|
| 线性标度DFT | O(N) | 核外存储 | 是 |
| 域分解并行 | 百万核 | 分布式 | 部分 |
| 冲击波MD | O(N) | 压缩存储 | 否 |
| 检查点 | 异步I/O | 增量存储 | N/A |

## 依赖要求

### 必需
- numpy >= 1.20
- scipy >= 1.7

### 可选
- mpi4py (MPI并行)
- cupy (GPU加速)
- h5py (HDF5检查点)
- zarr (Zarr检查点)
- psutil (内存监控)

### GPU支持
- CUDA >= 11.0
- cuBLAS/cuSOLVER

### HPC环境
- MPI实现 (OpenMPI/MPICH)
- 高速互联网络 (InfiniBand)
- 并行文件系统 (Lustre/GPFS)

## 设计特点

1. **模块化设计**: 各组件可独立使用
2. **渐进式优化**: 从单机到百万核平滑扩展
3. **容错机制**: 自动故障检测和恢复
4. **性能监控**: 详细的统计和报告
5. **灵活配置**: 丰富的配置选项

## 参考文献

1. Skylaris, C.K., et al. (2005). Introducing ONETEP. JCP.
2. Bowler, D.R., et al. (2010). Methods in Electronic Structure Calculations.
3. Reed, E.J., et al. (2003). A initio molecular dynamics of shock waves.
4. Kadau, K., et al. (2004). Molecular dynamics simulations of shock waves.

## 版本历史

- v1.0.0: 初始版本，完整功能实现
