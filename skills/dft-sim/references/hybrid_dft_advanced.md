# Hybrid-DFT 高级设置与参数优化

## 概述

杂化密度泛函（Hybrid DFT）混合了DFT交换相关能与精确的Hartree-Fock交换能，显著改善带隙、反应能垒和电子定域性的描述。本指南涵盖HSE、PBE0、SCAN0等主流杂化泛函的参数优化和高级设置。

---

## 理论基础

### 杂化泛函的一般形式

```
E_xc = α·E_x^HF + (1-α)·E_x^DFT + E_c^DFT
```

其中：
- **α**: HF交换混合比例（全局杂化）
- **E_x^HF**: Hartree-Fock交换能
- **E_x^DFT**: DFT交换能
- **E_c^DFT**: DFT相关能

### HSE (Heyd-Scuseria-Ernzerhof) 泛函

HSE将交换能分为短程(SR)和长程(LR)两部分：

```
E_xc^HSE = α·E_x^HF,SR(μ) + (1-α)·E_x^PBE,SR(μ) + E_x^PBE,LR(μ) + E_c^PBE
```

**关键参数**:
- **μ (screening parameter)**: 屏蔽参数，决定SR/LR分界，默认 0.2 Å⁻¹ (HSE06)
- **α (mixing parameter)**: HF混合比例，默认 0.25

### HSE03 vs HSE06

| 参数 | HSE03 | HSE06 |
|------|-------|-------|
| 交换分割 | 不同α | 相同α |
| 收敛性 | 较慢 | 更快 |
| 准确性 | 相当 | 推荐 |

### PBE0 与 SCAN0

**PBE0**: 全局杂化，α = 0.25
```
E_xc^PBE0 = 0.25·E_x^HF + 0.75·E_x^PBE + E_c^PBE
```

**SCAN0**: 基于meta-GGA的杂化
```
E_xc^SCAN0 = α·E_x^HF + (1-α)·E_x^SCAN + E_c^SCAN
```

---

## VASP 设置详解

### 基础 INCAR 设置

```bash
# 标准 HSE06
LHFCALC   = .TRUE.    # 开启杂化计算
HFSCREEN  = 0.2       # HSE06: μ = 0.2 Å⁻¹
ALGO      = Damped    # 或 All (初始), Normal/Damped (收敛)
TIME      = 0.4       # Damped算法混合参数
PRECFOCK  = Normal    # FFT网格精度: Fast/Normal/Accurate
NKRED     = 2         # Fock交换k点缩减因子

# 混合参数 (可选，默认0.25)
AEXX      = 0.25      # HF交换比例
AGGAX     = 0.75      # GGA交换比例 (1-AEXX)
AGGAC     = 1.00      # GGA相关比例
ALDAC     = 1.00      # LDA相关比例

# HSE03 (特殊设置)
# LHFCALC = .TRUE.
# HFSCREEN = 0.3      # HSE03使用μ=0.3
```

### PBE0 设置

```bash
LHFCALC   = .TRUE.
AEXX      = 0.25      # PBE0固定混合25%
AGGAX     = 0.75
AGGAC     = 1.00
ALDAC     = 1.00
ALGO      = Damped
PRECFOCK  = Normal
```

### HSEsol (固体优化版)

```bash
LHFCALC   = .TRUE.
HFSCREEN  = 0.2
AEXX      = 0.25
AGGAX     = 0.75
AGGAC     = 1.00
ALDAC     = 1.00
GGA       = PS        # PBEsol交换
ALGO      = Damped
```

### SCAN0 设置 (VASP 6.3+)

```bash
METAGGA   = SCAN      # 必须指定meta-GGA
LHFCALC   = .TRUE.
AEXX      = 0.25      # SCAN0混合参数
ALGO      = All       # SCAN需要All算法初始
PRECFOCK  = Normal
```

---

## Quantum ESPRESSO 设置

### input_gipaw 文件

```fortran
&inputgipaw
    job = 'hse'
    prefix = 'si'
    tmp_dir = './tmp/'
    verbosity = 'high'
/
```

### 使用 pw.x + Yambo

```bash
# 1. 标准DFT计算
pw.x -in scf.in > scf.out

# 2. Yambo读取并计算杂化修正
yambo -F hse.in -J hse_job
```

### Yambo HSE 输入示例

```
rim_cut                      # 库仑截断 (2D材料必需)
FFTGvecs=  50.0000     Ry    # FFT格点
% EXXRLvcs
  50.00000 |  50.00000 |  50.00000 |        Ry  # 交换格点数
%
% XfnQP_E
 0.250000 | 1.000000 | 1.000000 |      # GW/HSE准粒子能量修正
%
```

---

## 参数优化策略

### 1. 混合参数 α 优化

**物理意义**:
- 增大α → 更多HF交换 → 更大带隙
- 减小α → 更多DFT交换 → 更小带隙

**系统相关推荐值**:

| 材料类型 | 推荐 α | 说明 |
|---------|--------|------|
| 普通半导体 | 0.25 | HSE06默认值 |
| 宽禁带半导体 | 0.30-0.32 | GaN, ZnO等 |
| 强关联氧化物 | 0.20-0.25 | 避免过度局域化 |
| 有机半导体 | 0.50-0.65 | 减少自相互作用误差 |
| 2D材料 | 0.25-0.40 | 考虑介电屏蔽 |

**VASP调整**:
```bash
AEXX = 0.30    # 调整为30% HF交换
```

### 2. Screening 参数 μ 优化

**物理意义**:
- 增大μ → 更多SR交换 → 更快衰减
- 减小μ → 更多LR交换 → 更慢衰减

**HSE03 (μ=0.3)** 适合:
- 分子体系
- 需要更快收敛的情况

**HSE06 (μ=0.2)** 适合:
- 周期性固体
- 标准推荐

### 3. 收敛参数优化

#### FFT 网格 (PRECFOCK)

| 设置 | 精度 | 速度 | 适用场景 |
|------|------|------|---------|
| Fast | 低 | 最快 | 初步测试 |
| Normal | 中 | 中等 | 标准计算 |
| Accurate | 高 | 慢 | 发表级结果 |

```bash
PRECFOCK = Accurate   # 高精度要求
```

#### Fock 交换 k点缩减 (NKRED)

**原理**: 利用Fock交换的短程特性减少k点采样

```bash
NKRED = 2             # Fock矩阵使用减半的k点网格
NKREDX = 2            # X方向单独设置
NKREDY = 2
NKREDZ = 2
```

**注意**: 带隙计算时建议 NKRED=1（全k点）

### 4. 算法选择

| 算法 | 适用场景 | 特点 |
|------|---------|------|
| **All** | 初始计算, SCAN | 全局混合，最稳定 |
| **Damped** | 标准HSE/PBE0 | 二阶收敛，推荐 |
| **Normal** | 小体系 | 简单混合 |
| **Eigenval** | 能带计算 | 仅对角化 |

```bash
# 推荐流程
ALGO = All          # 第一步: 用ALGO=All获得收敛波函数
# 然后
ALGO = Damped       # 第二步: 用Damped精细收敛
TIME = 0.4          # 阻尼因子，可尝试0.2-0.5
```

---

## 特殊体系设置

### 2D 材料

**库仑截断** (避免层间虚假相互作用):

```bash
# VASP 6.3+
LHFCALC   = .TRUE.
HFSCREEN  = 0.2
LVHAR     = .TRUE.    # 输出Hartree势用于截断分析
# 需要确保真空层 > 15 Å

# 或使用外部截断 (VASP 6.4+)
LVCADER   = .TRUE.    # 库仑核截断
```

**k点设置**:
```bash
# 2D材料推荐
KPOINTS: Gamma centered
4 4 1
0 0 0
```

### 分子/团簇 (0D)

```bash
LHFCALC   = .TRUE.
HFSCREEN  = 0.3       # HSE03更适合分子
AEXX      = 0.25
# 使用大BOX，避免周期性镜像作用
```

### 金属体系

**挑战**: HF交换在金属中收敛困难

**解决方案**:
```bash
ALGO      = Damped
TIME      = 0.2       # 更小阻尼
NELMIN    = 6         # 最小电子步数
AMIX      = 0.1       # 更保守混合
BMIX      = 0.0001
```

---

## 能带结构计算流程

### 高效杂化能带计算方法

#### 方法1: Wannier 插值 (推荐)

```bash
# 1. 标准HSE自洽计算
LHFCALC = .TRUE.
HFSCREEN = 0.2
ALGO = Damped

# 2. 用Wannier90插值能带
LWANNIER90 = .TRUE.
```

#### 方法2: HSE06 + GGA 能带 (近似)

```bash
# Step 1: HSE自洽计算
LHFCALC = .TRUE.
ISTART = 0
ALGO = Damped

# Step 2: 非自洽能带 (节省60-70%时间)
ISTART = 1
ICHARG = 11           # 非自洽
LHFCALC = .FALSE.     # 关闭HSE，用GGA计算能带
# 或使用
LHFCALC = .TRUE.
ALGO = Eigenval       # 仅对角化
```

**精度评估**:
- HSE自洽: 最精确
- GGA非自洽: 带隙低估 ~0.2-0.5 eV
- 适合趋势分析

---

## 性能优化技巧

### 1. 并行设置

```bash
# VASP 标准HSE并行
mpirun -np 64 vasp_std

# 大体系推荐
NCORE = 4             # 每节点核心数
KPAR = 4              # k点并行
```

### 2. 分步计算策略

```bash
# Step 1: PBE快速预收敛 (节省50%时间)
LHFCALC = .FALSE.
ALGO = Fast
EDIFF = 1E-4

# Step 2: HSE从PBE波函数开始
LHFCALC = .TRUE.
ISTART = 1
ALGO = Damped
EDIFF = 1E-6
```

### 3. 内存优化

```bash
# 大体系减少内存
PRECFOCK = Fast       # 降低FFT网格
NBANDSHSE = 50        # 限制HSE空态数 (VASP 6.4+)
```

---

## 常见错误与解决

### 错误1: "Fock exchange not converged"

**原因**: Fock项振荡

**解决**:
```bash
ALGO = Damped
TIME = 0.3            # 减小阻尼
NELM = 200            # 增加最大迭代
AMIX = 0.05           # 更保守电荷混合
```

### 错误2: 能带顺序错乱

**原因**: 空态收敛不足

**解决**:
```bash
NBANDS = 1.5*N        # 增加空态数
NELMDL = -12          # 初始非自洽步
```

### 错误3: 2D材料带隙异常

**原因**: 真空层不足或截断问题

**解决**:
```bash
# 确保真空层 > 15 Å
# 检查LVHAR输出
# VASP 6.4+使用LVCADER
```

---

## 精度验证与基准

### 带隙基准测试 (VASP HSE06)

| 材料 | 实验带隙 (eV) | HSE06 (eV) | 误差 |
|------|--------------|-----------|------|
| Si | 1.17 | 1.24 | +0.07 |
| GaAs | 1.52 | 1.47 | -0.05 |
| ZnO | 3.44 | 2.78 | -0.66* |
| TiO₂ | 3.30 | 3.46 | +0.16 |

*ZnO需要GW或调整α值

### 计算成本对比

| 方法 | 相对CPU时间 | 内存需求 |
|------|------------|---------|
| PBE | 1x | 1x |
| PBE0 | 15-20x | 3x |
| HSE06 | 10-15x | 3x |
| HSE03 | 8-12x | 3x |
| B3LYP | 20-30x | 4x |

---

## 推荐工作流程

### 标准HSE能带计算流程

```bash
# 1. PBE结构优化
# INCAR.relax
ISIF = 3
EDIFFG = -0.01

# 2. PBE静态计算 (获得WAVECAR)
# INCAR.scf
ISTART = 0
LHFCALC = .FALSE.

# 3. HSE自洽计算
# INCAR.hse_scf
ISTART = 1
LHFCALC = .TRUE.
HFSCREEN = 0.2
ALGO = Damped
TIME = 0.4

# 4. HSE能带计算
# INCAR.hse_bands
ISTART = 1
ICHARG = 11
LHFCALC = .TRUE.
ALGO = Eigenval
KPOINTS = bands    # 高密度k点路径
```

---

## 参考资源

- [VASP HSE Wiki](https://www.vasp.at/wiki/index.php/HSE06)
- [HSE06原始论文](https://doi.org/10.1063/1.1564060)
- [SCAN0论文](https://doi.org/10.1063/1.4944918)
- [Yambo杂化泛函文档](https://www.yambo-code.eu/wiki/index.php/Techniques)
