# DFT-Sim 标准操作程序 (SOP)

## 目录
1. [环境设置](#1-环境设置)
2. [VASP计算流程](#2-vasp计算流程)
3. [QE计算流程](#3-qe计算流程)
4. [后处理分析](#4-后处理分析)
5. [故障排查](#5-故障排查)

---

## 1. 环境设置

### 1.1 加载环境模块

**VASP:**
```bash
module purge
module load intel/2023.1
module load intel-mpi/2021.9
export VASP_HOME=/opt/vasp/6.5.0
export PATH=$VASP_HOME/bin:$PATH
```

**QE:**
```bash
module purge
module load gcc/11.2
module load openmpi/4.1.5
export QE_HOME=/opt/qe/7.4
export PATH=$QE_HOME/bin:$PATH
export ESPRESSO_PSEUDO=$QE_HOME/pseudo
```

### 1.2 目录结构模板
```
project/
├── 01_scf/           # 自洽计算
├── 02_relax/         # 结构优化
├── 03_bands/         # 能带计算
├── 04_dos/           # 态密度
├── 05_md/            # 分子动力学
├── scripts/          # 脚本
└── results/          # 结果汇总
```

---

## 2. VASP计算流程

### 2.1 基础自洽计算

```bash
# 创建目录
mkdir -p 01_scf && cd 01_scf

# 准备输入文件
# - POSCAR: 晶体结构
# - INCAR: 计算参数
# - KPOINTS: k点网格
# - POTCAR: 赝势文件

# 提交计算
mpirun -np 16 vasp_std

# 检查收敛
grep "reached required accuracy" OUTCAR
```

### 2.2 结构优化

```bash
mkdir -p 02_relax && cd 02_relax

# 复制自洽结果
cp ../01_scaf/POSCAR .
cp ../01_scf/POTCAR .

# 修改INCAR
# IBRION = 2
# ISIF = 3
# NSW = 100
# EDIFFG = -0.01

mpirun -np 16 vasp_std

# 检查结果
cp CONTCAR POSCAR_optimized
```

### 2.3 能带计算

```bash
mkdir -p 03_bands && cd 03_bands

# 步骤1: 自洽计算
cp ../02_relax/CONTCAR POSCAR
cp ../02_relax/WAVECAR .
# ISMEAR = -5
mpirun -np 16 vasp_std

# 步骤2: 非自洽能带计算
# ISTART = 1
# ICHARG = 11
# 修改KPOINTS为Line-mode
mpirun -np 16 vasp_std

# 后处理
python $DFT_SIM/scripts/vasp/plot_bands.py
```

### 2.4 MLFF训练

```bash
mkdir -p 05_mlff && cd 05_mlff

# 步骤1: 训练
# ML_MODE = train
# ML_CDOUB = 10
mpirun -np 16 vasp_std

# 步骤2: 检查质量
grep "ERR" ML_LOGFILE

# 步骤3: 生产运行
# ML_MODE = run
# cp ML_FFN ML_FF
mpirun -np 4 vasp_std
```

---

## 3. QE计算流程

### 3.1 基础SCF计算

```bash
mkdir -p 01_scf && cd 01_scf

# 准备pw.in输入文件
mpirun -np 16 pw.x -in pw.in > pw.out 2>&1

# 检查收敛
grep "convergence has been achieved" pw.out
```

### 3.2 结构优化

```bash
mkdir -p 02_relax && cd 02_relax

# calculation = 'relax'
# ion_dynamics = 'bfgs'

mpirun -np 16 pw.x -in relax.in > relax.out 2>&1
```

### 3.3 能带计算

```bash
mkdir -p 03_bands && cd 03_bands

# 步骤1: SCF
cp ../02_relax/*.save ./
mpirun -np 16 pw.x -in scf.in > scf.out 2>&1

# 步骤2: Bands
mpirun -np 16 pw.x -in bands.in > bands.out 2>&1

# 步骤3: 后处理
bands.x < bands_post.in
plotband.x < plotband.in
```

### 3.4 声子计算

```bash
mkdir -p 04_phonon && cd 04_phonon

# 步骤1: SCF
mpirun -np 16 pw.x -in scf.in > scf.out 2>&1

# 步骤2: Phonon
mpirun -np 16 ph.x -in ph.in > ph.out 2>&1

# 步骤3: 后处理
q2r.x < q2r.in
matdyn.x < matdyn.in
```

---

## 4. 后处理分析

### 4.1 提取能量和力

**VASP:**
```bash
grep "TOTEN" OUTCAR | tail -1
grep "TOTAL-FORCE" OUTCAR -A 20
```

**QE:**
```bash
grep "!" pw.out
grep "Total force" pw.out
```

### 4.2 能带绘图

```bash
# VASP
python $DFT_SIM/scripts/vasp/plot_bands.py --eigenval EIGENVAL --outcar OUTCAR

# QE
python $DFT_SIM/scripts/qe/plot_bands.py --input bands.dat.gnu
```

### 4.3 DOS绘图

```bash
# VASP
python $DFT_SIM/scripts/vasp/plot_dos.py --doscar DOSCAR

# QE
python $DFT_SIM/scripts/qe/plot_dos.py --dos pwscf.dos
```

### 4.4 批量分析

```bash
# VASP
python $DFT_SIM/scripts/vasp/compare_results.py --dir . --report report.txt

# QE
python $DFT_SIM/scripts/qe/analyze_results.py --dir . --csv summary.csv
```

---

## 5. 故障排查

### 5.1 不收敛问题

**VASP:**
- 调整ALGO (Normal比Fast更稳定)
- 降低mixing_beta
- 增加NELMIN
- 调整SIGMA

**QE:**
- 降低mixing_beta
- 改变mixing_mode
- 增加electron_maxstep
- 调整degauss

### 5.2 内存不足

- 增加MPI进程数
- 降低ENCUT/ecutwfc
- 减少k点
- 使用NCORE/kpar并行

### 5.3 常见错误速查

| 错误 | 可能原因 | 解决方案 |
|------|---------|---------|
| Segmentation fault | 栈内存不足 | ulimit -s unlimited |
| Out of memory | 内存不足 | 增加进程数/降低精度 |
| Convergence failed | 初始猜测差 | 调整mixing/算法 |
| Symmetry error | 结构问题 | ISYM=0/nosym=.true. |

---

## 附录

### A. 常用命令速查

```bash
# 查看计算状态
tail -f OSZICAR          # VASP
tail -f pw.out | grep "!" # QE

# 检查资源使用
top -u $USER
nvidia-smi               # GPU

# 批量提交
bash $DFT_SIM/scripts/vasp/batch_submit.sh
bash $DFT_SIM/scripts/qe/batch_run.sh
```

### B. 参考值

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| ENCUT | 1.3×ENMAX | 截断能 |
| KPOINTS | 0.2 Å⁻¹ | k点密度 |
| EDIFF | 1E-6 | 电子收敛 |
| EDIFFG | -0.01 | 力收敛 |

---

*文档版本: 1.0*
*更新日期: 2026-03-08*
