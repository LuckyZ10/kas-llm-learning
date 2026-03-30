# 增强采样方法 (Enhanced Sampling Methods)

## 简介

增强采样方法是分子动力学模拟中用于克服能垒、探索稀有事件和计算自由能面的重要技术。这些方法通过引入偏置势、提高温度或扩展系综等方式，加速对复杂能量景观的采样。

---

## 理论基础

### 自由能与反应坐标

自由能沿反应坐标 $s$ 的定义：

$$F(s) = -k_B T \ln P(s) + C$$

其中 $P(s)$ 是反应坐标的概率分布。

**反应坐标/集体变量 (CV)**:
- 距离、角度、二面角
- 配位数
- 路径变量
- 主成分分析 (PCA) 模式

### 稀有事件问题

能垒 $\Delta G^\ddagger \gg k_B T$ 时，跃迁时间 $\tau \propto e^{\Delta G^\ddagger/k_BT}$ 可能远超模拟时间尺度。

---

## Metadynamics

### 原理

Metadynamics通过向系统添加随时间累积的高斯偏置势，"填充"自由能极小值：

$$V_G(s, t) = \sum_{t'=\tau_G, 2\tau_G, ...}^{t'} h \exp\left(-\frac{(s-s(t'))^2}{2\sigma^2}\right)$$

在长时间极限下：

$$\lim_{t\to\infty} V_G(s, t) \approx -F(s)$$

### Well-Tempered Metadynamics (推荐)

偏置势高度随时间衰减：

$$h(t) = h_0 \exp\left(-\frac{V_G(s,t)}{k_B\Delta T}\right)$$

**优势**:
- 收敛更快
- 可计算自由能
- 边界效应更小

### VASP内置Metadynamics

**基本设置**:

```
# INCAR
IBRION = 0              # MD计算
MDALGO = 21             # 21=Nose-Hoover, 11=Andersen
TEBEG = 300             # 起始温度
POTIM = 0.5             # 时间步长 (fs)
NSW = 100000            # MD步数

# Metadynamics参数
HILLS_H = 0.05          # 高斯高度 (eV)
HILLS_W = 0.05          # 高斯宽度
HILLS_BIN = 100         # 沉积间隔 (步数)
```

**集体变量定义** (`ICONST`):

```
# ICONST文件格式
# 类型  STATUS  原子列表  参数

# 距离 CV
R 5 12 34             # STATUS=5表示Metadynamics CV
# 原子12和34之间的距离

# 角度 CV
A 5 12 34 56          # 原子12-34-56的角度

# 二面角 CV
T 5 12 34 56 78       # 原子12-34-56-78的二面角

# 配位数 CV
C 5 12 0.5 6.0        # 原子12的配位数，截止半径0.5-6.0 Å
```

**初始偏置势** (`PENALTYPOT`):

```
# 可选：提供初始偏置势
# 格式：CV值  偏置势值
0.0  0.0
0.1  0.05
0.2  0.10
...
```

**输出文件**:
- `REPORT`: CV值和偏置势历史
- `HILLSPOT`: 累积的高斯偏置势

**自由能重建**:

```python
import numpy as np
import matplotlib.pyplot as plt

# 从HILLSPOT读取偏置势
data = np.loadtxt('HILLSPOT')
cv = data[:, 0]
bias = data[:, 1]

# 自由能 ≈ -偏置势 (负号)
free_energy = -bias

plt.plot(cv, free_energy)
plt.xlabel('Reaction Coordinate (Å)')
plt.ylabel('Free Energy (eV)')
plt.savefig('free_energy.png')
```

### PLUMED + Quantum ESPRESSO

**安装PLUMED补丁**:

```bash
# 下载并安装PLUMED
wget https://github.com/plumed/plumed2/releases/download/v2.9.0/plumed-2.9.0.tgz
tar -xzf plumed-2.9.0.tgz
cd plumed-2.9.0
./configure --prefix=/path/to/plumed
make -j 4
make install

# 给QE打补丁
cd /path/to/qe-6.8
/path/to/plumed/bin/plumed patch --engine qespresso-6.8
./configure
make all
```

**PLUMED输入文件** (`plumed.dat`):

```plumed
# 定义集体变量
DISTANCE ATOMS=12,34 LABEL=d1

# Well-tempered metadynamics
METAD ARG=d1 SIGMA=0.05 HEIGHT=1.2 PACE=100 BIASFACTOR=10 FILE=HILLS

# 输出
PRINT ARG=d1 STRIDE=10 FILE=COLVAR
```

**QE输入文件**:

```fortran
&CONTROL
  calculation = 'md'
  prefix = 'metad'
  outdir = './tmp/'
  pseudo_dir = '../pseudos/'
  dt = 20.0               ! 时间步长 (Ry a.u.)
  nstep = 10000
/

&SYSTEM
  ibrav = 1
  celldm(1) = 20.0
  nat = 50
  ntyp = 3
  ecutwfc = 40
  nosym = .true.
/

&ELECTRONS
  conv_thr = 1.0d-8
/

&IONS
  tempw = 300.0           ! 目标温度 (K)
  ion_temperature = 'nose-hoover'
  fnosep = 50.0
/
```

**运行**:

```bash
export PLUMED_KERNEL=/path/to/plumed/lib/libplumedKernel.so
mpirun -np 16 pw.x -plumed plumed.dat -in pw.in > pw.out
```

**后处理**:

```bash
# 重建自由能面
plumed sum_hills --hills HILLS --outfile fes.dat

# 检查收敛性
plumed sum_hills --hills HILLS --stride 100 --mintozero
```

---

## 自适应偏置力 (ABF)

### 原理

ABF通过估计并抵消平均力，使系统在CV空间均匀扩散：

$$\mathbf{F}^{ABF}(\xi) = \langle \nabla_\xi V \rangle_\xi$$

自由能：

$$F(\xi) = -\int^\xi \mathbf{F}^{ABF}(\xi') \cdot d\xi'$$

### PySAGES实现

PySAGES是用于增强采样的Python库，支持多种后端：

```python
import pysages
from pysages.methods import ABF
from pysages.colvars import Distance

# 定义CV
cv = Distance([0, 1])  # 原子0和1之间的距离

# ABF方法
method = ABF(
    cvs=[cv],
    grid=pysages.Grid(lower=0.0, upper=5.0, shape=50),
    restraint_lower=0.1,
    restraint_upper=4.9,
)

# 运行模拟
result = pysages.run(method, generate_context, timesteps=100000)

# 提取自由能
free_energy = result.free_energy
```

---

## 伞形采样 (Umbrella Sampling)

### 原理

通过谐振约束将系统限制在CV空间的特定窗口：

$$V_{umb}(s) = \frac{1}{2}k(s - s_0)^2$$

### 实现

**VASP**:

```
# INCAR
IBRION = 0
MDALGO = 21
TEBEG = 300
POTIM = 0.5
NSW = 50000

# 约束 (通过ICONST)
```

**ICONST**:

```
R 7 12 34 2.5 50.0    # STATUS=7表示约束
# 约束原子12和34之间的距离为2.5 Å，力常数50 eV/Å^2
```

**PLUMED**:

```plumed
# 定义CV
DISTANCE ATOMS=12,34 LABEL=d1

# 谐振约束
RESTRAINT ARG=d1 AT=2.5 KAPPA=1000.0

# 输出
PRINT ARG=d1 STRIDE=10 FILE=COLVAR
```

**加权直方图分析 (WHAM)**:

```python
import numpy as np
from wham import WHAM

# 收集各窗口的直方图
histograms = []
for window in windows:
    hist, bins = np.histogram(window_data, bins=50)
    histograms.append(hist)

# WHAM分析
wham = WHAM(bins, histograms, kappa=1000.0, centers=window_centers)
free_energy = wham.run(tolerance=1e-6)
```

---

## 副本交换分子动力学 (REMD)

### 原理

多个副本在不同温度下并行运行，定期交换构型：

$$P_{acc}(i \leftrightarrow j) = \min\left(1, \exp\left[(\beta_i - \beta_j)(U_i - U_j)\right]\right)$$

### 实现

**i-PI**:

```xml
<simulation>
  <system>
    <motion mode='remd'>
      <remd>
        <nreplicas>8</nreplicas>
        <temperatures>[300, 350, 400, 450, 500, 550, 600, 650]</temperatures>
      </remd>
    </motion>
  </system>
</simulation>
```

**PLUMED**:

```plumed
# 副本交换Metadynamics
METAD ARG=d1 SIGMA=0.1 HEIGHT=1.2 PACE=500 FILE=HILLS

# 多个副本在不同温度运行
# 通过MPI通信交换
```

---

## 温度加速分子动力学 (TAMD/d-AFED)

### 原理

对CV赋予高温，加速CV空间的探索：

$$\tilde{F}(z) = -\frac{1}{\beta}\ln\langle\delta_\kappa(\theta(x) - z)\rangle$$

### 实现

```plumed
# 扩展拉格朗日量方法
METAD ARG=d1 SIGMA=0.1 HEIGHT=0.0 PACE=1000000
# 或使用专门的TAMD实现
```

---

## 方法对比

| 方法 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| Metadynamics | 探索未知路径 | 自动发现路径 | CV选择关键 |
| Well-tempered | 计算自由能 | 收敛快 | 参数调优 |
| ABF | 高维CV | 直接得平均力 | 初始探索慢 |
| Umbrella Sampling | 已知路径 | 精确控制 | 需要预定义窗口 |
| REMD | 复杂能景 | 无需CV | 计算成本高 |
| TAMD | 慢CV | 加速CV动力学 | 近似结果 |

---

## 最佳实践

### CV选择原则

1. **物理相关性**: CV应描述反应的本质
2. **区分性**: 能区分反应物、过渡态、产物
3. **正交性**: 多个CV应尽量正交
4. **可计算性**: 计算开销合理

### 收敛性检查

```python
def check_metad_convergence(hills_files):
    """检查metadynamics收敛性"""
    import numpy as np
    
    # 1. 系统是否在CV空间充分扩散
    colvar = np.loadtxt('COLVAR')
    cv_range = np.max(colvar[:, 1]) - np.min(colvar[:, 1])
    
    # 2. 自由能是否稳定
    free_energies = []
    for i in range(5):
        fes = calculate_fes(hills_files, upto=i*len(hills_files)//5)
        free_energies.append(fes)
    
    # 检查自由能差异
    for i in range(1, len(free_energies)):
        diff = np.max(np.abs(free_energies[i] - free_energies[i-1]))
        print(f"Step {i}: max difference = {diff:.3f} kJ/mol")
        if diff < 2.0:  # 2 kJ/mol阈值
            print("Likely converged")
```

### 误差估计

```bash
# 使用PLUMED的sum_hills进行块分析
plumed sum_hills --hills HILLS --stride 100 --mintozero

# 或使用独立轨迹进行交叉验证
```

---

## 参考资源

- PLUMED官网: https://www.plumed.org/
- PySAGES: https://pysages.readthedocs.io/
- Metadynamics综述: Laio & Parrinello, PNAS (2002)
- Well-tempered: Barducci et al., PRL (2008)
