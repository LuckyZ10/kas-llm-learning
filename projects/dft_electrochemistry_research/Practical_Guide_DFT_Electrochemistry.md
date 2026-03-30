# DFT电化学计算实践指南
## 基于2024-2025前沿方法的实操手册

---

## 第一部分：计算氢电极(CHE)模型详解

### 1.1 CHE模型基本公式

反应自由能计算：
```
ΔG = ΔE_DFT + ΔE_ZPE - TΔS + ΔG_U + ΔG_pH
```

各项含义：
- **ΔE_DFT**: DFT计算能量差
- **ΔE_ZPE**: 零点能校正 (通过频率计算获得)
- **TΔS**: 熵校正 (T = 298.15 K)
- **ΔG_U = -neU**: 电极电位校正 (n: 电子数, U: 电位 vs RHE)
- **ΔG_pH = k_B T × pH × ln(10)**: pH校正

### 1.2 关键近似

**质子-电子对假设** (CHE核心)：
```
H⁺ + e⁻ ⇌ ½ H₂(g)   ΔG = 0  (在 U = 0 V vs RHE, pH = 0)
```

**电位校正**：
- 每转移一个电子，能量变化 -eU
- 在电位 U 下，所有涉及H⁺/e⁻的步骤自由能变化 -eU

**pH校正**：
- 标准CHE使用RHE参考电极，自动包含pH效应
- 若使用SHE参考，需显式添加 ΔG_pH

### 1.3 典型计算流程

```python
# 概念性流程
1. 优化催化剂结构
   ├── 自旋极化计算
   ├── PBE/PBE+U 泛函
   ├── 力收敛标准: 0.01-0.02 eV/Å
   └── ENCUT: 400-500 eV

2. 吸附中间体优化
   ├── *H, *OH, *O, *OOH (HER/OER/ORR)
   ├── *CO, *COOH, *HCOO (CO2RR)
   ├── *N, *N2H, *NH (NRR)
   └── 频率计算获取ZPE

3. 自由能图构建
   ├── 计算各步骤ΔG
   ├── 确定电位决定步(PDS)
   └── 极限电位 U_L = -ΔG_max/e
```

### 1.4 VASP计算参数模板

**标准CHE计算INCAR**:
```fortran
# 基础设置
PREC = Normal
ENCUT = 450
EDIFF = 1E-5
EDIFFG = -0.02

# 电子结构
ISMEAR = 0
SIGMA = 0.1
ISPIN = 2

# 优化
IBRION = 2
ISIF = 2
NSW = 300

# 溶剂化 (可选 - VASPsol)
LSOL = .TRUE.
EB_K = 80.0        # 水介电常数
TAU = 0            # 表面张力参数
LAMBDA_D_K = 3.04  # Debye长度 (1M电解质)
```

**频率计算INCAR**:
```fortran
# 继承优化设置
ISTART = 1
ICHARG = 1

# 频率计算
IBRION = 5         # 有限差分法
NFREE = 2
POTIM = 0.015
NWRITE = 3
```

### 1.5 常见错误与注意事项

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 负频率 | 过渡态或未收敛 | 重新优化，检查力收敛 |
| ZPE异常大 | 吸附物不稳定 | 检查吸附构型 |
| 过电位与实验不符 | CHE近似局限 | 考虑显式溶剂或GC-DFT |
| pH效应错误 | 参考电极混淆 | 确认RHE vs SHE |

---

## 第二部分：恒电位方法实施

### 2.1 GC-DFT实施 (JDFTx)

**安装**:
```bash
git clone https://github.com/shankar1729/jdftx.git
cd jdftx && mkdir build && cd build
cmake .. -D EnableCUDA=yes  # GPU加速
make -j4
```

**输入文件模板**:
```bash
# 催化剂结构
coulomb-interaction Slab 001
coulomb-truncation-embed 0 0 0

# DFT设置
elec-cutoff 20 100
elec-ex-corr gga-PBE

# 溶剂化
fluid LinearPCM
pcm-variant GLSSA13
fluid-solvent H2O

# 恒电位 (GC-DFT核心)
target-mu -0.16  # 对应约0V vs SHE
dump End State
```

**电位-化学势转换**:
```
μ_e = -φ(SHE) - 4.43 eV

示例:
- U = 0 V vs SHE → μ_e = -4.43 eV
- U = -1.0 V vs SHE → μ_e = -3.43 eV
```

### 2.2 CIP-DFT实施 (GPAW)

**Python脚本模板**:
```python
from gpaw import GPAW, PW, SJM
from ase import Atoms
from ase.io import read

# 读取催化剂
atoms = read('catalyst.cif')

# SJM计算器 (CIP-DFT)
sj = {'target_potential': 3.84,  # vs 内势参考
      'pot_ref': 'CIP',
      'cip': {'autoinner': {'nlayers': 4},
              'mu_pzc': -4.44,   # PZC费米能级
              'phi_pzc': 4.44}}  # PZC内势

calc = GPAW(mode=PW(450),
            xc='PBE',
            sj=sj,
            kpts=(3, 3, 1))

atoms.calc = calc
energy = atoms.get_potential_energy()
inner_potential = calc.get_inner_potential(atoms)
```

### 2.3 VASP+VASPsol恒电位近似

**表面电荷扫描法**:
```python
# 通过改变NELECT模拟不同电位
# 需拟合电荷-电位关系

# 1. 中性计算获取PZC
# 2. 增减电子模拟充放电
# 3. 计算功函 vs 电子数关系
# 4. 插值获取目标电位构型
```

**参数设置**:
```fortran
# VASPsol设置
LSOL = .TRUE.
EB_K = 80.0
LAMBDA_D_K = 3.04

# 电荷调整
NELECT = [默认值] ± δn
```

---

## 第三部分：隐式溶剂化模型对比

### 3.1 模型选择指南

| 模型 | 适用场景 | 计算成本 | 精度 |
|------|----------|----------|------|
| VASPsol | 金属表面、快速筛选 | 低 | 中等 |
| CANDLE | 带电物种、极性溶质 | 中 | 高 |
| SaLSA | 非局域响应重要时 | 中-高 | 高 |
| JDFT | 固液界面精细结构 | 高 | 最高 |

### 3.2 VASPsol参数优化

**介电常数选择**:
```
水 (H2O):     EB_K = 78.4 (25°C)
乙腈:         EB_K = 36.6
甲醇:         EB_K = 32.7
DMF:          EB_K = 36.7
```

**Debye长度计算**:
```
λ_D = √(εk_B T / 2N_A e² I)

其中I为离子强度 (mol/L)
1M单价电解质: λ_D ≈ 3.04 Å
0.1M单价电解质: λ_D ≈ 9.6 Å
```

### 3.3 常见溶剂参数表

| 溶剂 | ε | λ_D (1M) | 适用模型 |
|------|---|----------|----------|
| 水 | 78.4 | 3.04 Å | VASPsol, CANDLE |
| 碳酸丙烯酯 | 64.9 | 3.36 Å | VASPsol |
| 离子液体 | 10-15 | 8.5 Å | NonlinearPCM |

---

## 第四部分：过渡态与动力学计算

### 4.1 NEB方法选择

| 方法 | 适用场景 | 特点 |
|------|----------|------|
| 标准NEB | 简单反应路径 | 需要初始猜测 |
| CI-NEB | 精确过渡态 | 爬坡图像收敛 |
| dNEB | 双端固定 | 适合表面扩散 |
| GC-NEB | 电化学反应 | 恒电位条件下 |

### 4.2 VASP CI-NEB实现

**INCAR设置**:
```fortran
# NEB设置
IMAGES = 7         # 中间图像数
SPRING = -5        # 弹簧常数
LCLIMB = .TRUE.    # 爬坡图像
ICHAIN = 0

# 优化
IBRION = 3
POTIM = 0

# 收敛
EDIFFG = -0.05
NELMIN = 4
```

**工作流**:
```bash
# 1. 准备目录结构
mkdir -p 00 01 02 03 04 05 06 07 08
# 00 - 初态, 08 - 末态

# 2. 插值生成中间图像
nebmake.pl POSCAR_initial POSCAR_final 7

# 3. 运行NEB
mpirun vasp_std

# 4. 分析结果
nebresults.pl
```

### 4.3 恒电位过渡态 (GC-NEB)

**JDFTx实现**:
```bash
# 在neb路径的每个点添加
target-mu -4.0  # 恒电位条件

# 或使用SJM在GPAW中
```

**关键注意事项**:
- 电位会影响过渡态位置
- 能垒随电位变化 (Butler-Volmer行为)
- 需要电位-电荷自洽

---

## 第五部分：单原子催化剂高通量筛选

### 5.1 自动化工作流

**Python脚本框架**:
```python
from ase.io import read, write
from ase.build import add_adsorbate
from ase.calculators.vasp import Vasp
import numpy as np

# 定义SAC体系
substrates = ['graphene', 'C2N', 'g-C3N4', 'MOF-5']
metals = ['Fe', 'Co', 'Ni', 'Cu', 'Mn', 'Cr', 'V', 'Ti']
adsorbates = {'CO2RR': ['*CO', '*COOH', '*HCOO'],
              'HER': ['*H'],
              'NRR': ['*N', '*NNH', '*NH']}

# 高通量计算循环
for sub in substrates:
    for metal in metals:
        # 构建SAC结构
        sac = build_sac(sub, metal)
        
        # 稳定性筛选 (形成能)
        E_form = calculate_formation_energy(sac)
        if E_form > 0.5:  # eV, 不稳定
            continue
        
        for ads in adsorbates['CO2RR']:
            # 吸附能计算
            E_ads = calculate_adsorption_energy(sac, ads)
            
            # 存储结果
            save_results(sub, metal, ads, E_ads)
```

### 5.2 描述符自动提取

**电子结构描述符**:
```python
def extract_descriptors(calc, atoms):
    """从DFT计算提取关键描述符"""
    
    # d带中心
    dos = calc.get_dos()
    d_band_center = calculate_d_band_center(dos, atom_index=0)
    
    # Bader电荷
    bader_charges = run_bader_analysis()
    
    # 磁矩
    magnetic_moments = atoms.get_magnetic_moments()
    
    # 配位数
    cn = calculate_coordination_number(atoms, index=0)
    
    return {
        'd_band_center': d_band_center,
        'bader_charge': bader_charges[0],
        'magnetic_moment': magnetic_moments[0],
        'coordination_number': cn
    }
```

### 5.3 机器学习集成

**活性预测模型**:
```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# 特征准备
X = np.array([[d_band, cn, electronegativity] 
              for d_band, cn, electronegativity in descriptors])
y = np.array(overpotentials)  # 目标变量

# 模型训练
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = GradientBoostingRegressor(n_estimators=100)
model.fit(X_train, y_train)

# 预测新催化剂
predicted_overpotential = model.predict(new_descriptor)
```

---

## 第六部分：机器学习势(MLIP)应用

### 6.1 MACE快速入门

**安装**:
```bash
pip install mace-torch
```

**训练数据准备**:
```python
from mace.calculators import mace_mp
from ase import Atoms
from ase.io import read

# 加载预训练模型
calc = mace_mp(model="medium", device="cuda")

# 或微调
configs = read('training_data.xyz', ':')
```

**分子动力学模拟**:
```python
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

# 设置计算器
atoms.calc = calc

# 初始速度
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# Langevin动力学
dyn = Langevin(atoms, timestep=1*units.fs, 
               temperature_K=300, friction=0.01)

# 运行
def print_energy():
    print(f"Energy: {atoms.get_potential_energy()}")

dyn.attach(print_energy, interval=10)
dyn.run(10000)  # 10 ps
```

### 6.2 主动学习工作流

```python
# 主动学习循环
for iteration in range(max_iter):
    # 1. 当前MLIP采样
    trajectories = run_md_sampling(calculator)
    
    # 2. 不确定性量化
    uncertain_configs = select_uncertain_structures(trajectories)
    
    # 3. DFT标记
    dft_energies = run_dft_calculations(uncertain_configs)
    
    # 4. 重新训练
    retrain_mlip(uncertain_configs, dft_energies)
    
    # 5. 收敛检查
    if uncertainty < threshold:
        break
```

---

## 附录：关键参考资料

### 软件文档
1. **JDFTx**: http://jdftx.org/
2. **VASPsol**: https://github.com/henniggroup/VASPsol
3. **GPAW+SJM**: https://gpaw.readthedocs.io/
4. **MACE**: https://github.com/ACEsuit/mace

### 关键论文
1. Nørskov et al., J. Phys. Chem. B 2004 (CHE模型)
2. Sundararaman et al., Chem Rev 2022 (溶剂化综述)
3. Melander et al., npj Comput Mater 2024 (CIP-DFT)
4. Shiota et al., 2024 (MACE-Osaka24)

### 基准数据集
1. **BEAST-DB**: 电化学GC-DFT计算数据库
2. **OC20/OC22**: Open Catalyst Project
3. **Materials Project**: 材料计算数据库

---

*文档版本: 2025.03  
最后更新: 2026-03-08*
