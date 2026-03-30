# DFT计算性能优化指南

本文档提供DFT计算的性能优化策略，涵盖并行计算、内存管理、IO优化等方面，帮助用户在有限资源下实现最高计算效率。

---

## 1. 并行策略

### 1.1 VASP并行参数

#### NCORE - 核心分组策略

```
NCORE = 4    # 每MPI进程使用的核数
```

**选择策略**:
| 总核数 | k点数 | 推荐NCORE | 说明 |
|--------|-------|-----------|------|
| 32 | 8 | 4 | 8组×4核 |
| 64 | 16 | 4 | 16组×4核 |
| 128 | 8 | 16 | 8组×16核 (共享内存) |
| 256 | 16 | 16 | 16组×16核 |

**优化建议**:
- NCORE应与节点核心数匹配或约数
- 共享内存节点可用较大NCORE (8-16)
- 跨节点时NCORE应整除每节点核心数

#### KPAR - k点并行

```
KPAR = 8     # k点并行组数
```

**设置原则**:
```
总核数 = KPAR × NCORE × NPAR
其中KPAR应整除k点总数
```

**示例配置**:
```
# 64核, 16个k点
KPAR = 8     # 8组并行处理k点
NCORE = 4    # 每组4核处理1个k点
NPAR = 2     # 可选，与NCORE互斥
```

#### NPAR - 能带并行 (VASP 5.x)

```
NPAR = 4     # 能带并行组数
# 注意: NPAR与NCORE互斥，不要同时设置
```

**选择建议**:
- k点少、能带多时: 使用NPAR
- k点多、能带少时: 使用KPAR+NCORE

### 1.2 QE并行参数

#### pw.x并行策略

```bash
# 混合并行 (MPI + OpenMP)
mpirun -np 64 pw.x -nk 4 -nt 4 -in pw.in

# 参数说明
-nk 4    # k点并行池数
-nt 4    # 每MPI线程的OpenMP线程数
```

**QE并行层次**:
1. **k点并行** (`-nk`): 最高效，线性扩展
2. **R/G并行**: 通过 `-nd` 控制
3. **能带并行** (`-nb`): 对角化并行
4. **OpenMP线程** (`-nt`): 节点内并行

**配置矩阵**:
| 总核 | k点 | -nk | -nb | -nt | 效率 |
|------|-----|-----|-----|-----|------|
| 64 | 16 | 16 | 1 | 4 | 95% |
| 64 | 8 | 8 | 2 | 4 | 90% |
| 128 | 16 | 16 | 2 | 4 | 92% |
| 256 | 32 | 32 | 1 | 8 | 88% |

### 1.3 并行效率诊断

**VASP计时分析**:
```bash
# OUTCAR末尾查找:
# Parallelisation info
# orbital    projec
# 1          2.35    
# ...

# 提取并行效率
grep "Elapsed time" OUTCAR
```

**QE性能分析**:
```bash
# 添加 -v 选项查看详细计时
mpirun -np 64 pw.x -nk 8 -v -in pw.in | tee pw.out

# 输出示例:
 #    k-point           seconds
 #    1                45.23
 #    ...
```

**Python分析脚本**:
```python
def analyze_parallel_efficiency(log_file):
    """
    分析并行效率
    """
    import re
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # 提取各k点计算时间
    times = re.findall(r'k-point\s+\d+\s+([\d.]+)', content)
    times = [float(t) for t in times]
    
    if times:
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        
        # 负载均衡度
        balance = min_time / max_time
        
        print(f"平均时间: {avg_time:.2f} s")
        print(f"最大时间: {max_time:.2f} s")
        print(f"最小时间: {min_time:.2f} s")
        print(f"负载均衡度: {balance:.2%}")
        
        if balance < 0.8:
            print("警告: 负载不均衡，考虑调整KPAR/nk")
    
    return balance
```

---

## 2. 内存优化

### 2.1 VASP内存控制

#### 关键参数

```
# INCAR内存相关设置
NELMIN = 4          # 最小电子步数
NELMDL = -5         # 初始非自洽步
LREAL = Auto        # 实空间投影 (大体系必需)
# LREAL = .FALSE.   # 小体系用倒空间更精确
```

#### WAVECAR/CHGCAR管理

```bash
# 1. 分阶段计算，避免存储大文件
# SCF → 保存CHGCAR → 后续计算读取

# 2. 使用gzip压缩
 gzip CHGCAR WAVECAR
# 使用时解压
 gunzip CHGCAR.gz

# 3. 删除不必要的文件
rm WAVECAR  # 如果能带计算不需要
```

#### 大体系策略 (>500原子)

```
# INCAR大体系设置
LREAL = Auto        # 节省内存
PREC = Normal       # 降低精度换取速度
LWAVE = .FALSE.     # 不保存波函数
LCHARG = .FALSE.    # 不保存电荷密度 (仅优化)
ISYM = 0            # 关闭对称性 (如果内存不足)
```

### 2.2 QE内存控制

#### 磁盘IO vs 内存

```fortran
&CONTROL
   disk_io = 'medium'   ! 'low', 'medium', 'high'
   wf_collect = .true.  ! 收集波函数到主节点
/
```

**选项对比**:
| 选项 | 内存 | 磁盘 | 适用场景 |
|------|------|------|----------|
| low | 低 | 高 | 大体系，小内存 |
| medium | 中 | 中 | 平衡配置 |
| high | 高 | 低 | 小体系，追求速度 |

#### 波函数存储优化

```bash
# 使用 -pool 分布波函数
mpirun -np 64 pw.x -nk 8 -in pw.in

# 清理临时文件
cd $TMPDIR
rm -rf ${JOB_NAME}.*
```

### 2.3 内存需求估算

**VASP估算公式**:
```
内存 (GB) ≈ N_electrons × NBANDS × N_kpoints × 16 / 1e9

示例:
- 200电子, 200能带, 10k点
- 内存 ≈ 200 × 200 × 10 × 16 / 1e9 = 6.4 GB
```

**QE估算公式**:
```
内存 (GB) ≈ N_PW × N_bands × N_kpoints × 16 / 1e9

示例:
- 10000平面波, 100能带, 10k点
- 内存 ≈ 10000 × 100 × 10 × 16 / 1e9 = 1.6 GB
```

**Python内存估算器**:
```python
def estimate_memory(n_atoms, n_electrons, encut, kpoints, n_bands=None):
    """
    估算DFT计算内存需求
    """
    import numpy as np
    
    # 估算平面波数
    # VASP: NPW ~ ENCUT^1.5 × Volume
    # 简化估算
    npw = int(100 * encut / 100 * (n_atoms / 10) ** 0.5)
    
    if n_bands is None:
        n_bands = int(n_electrons / 2 * 1.2)  # 20%空带
    
    n_k = np.prod(kpoints)
    
    # VASP内存 (GB)
    vasp_mem = n_electrons * n_bands * n_k * 16 / 1e9 * 2
    
    # QE内存 (GB)
    qe_mem = npw * n_bands * n_k * 16 / 1e9 * 1.5
    
    print("=== 内存估算 ===")
    print(f"体系: {n_atoms}原子, {n_electrons}电子")
    print(f"k点: {kpoints} ({n_k}个)")
    print(f"能带数: {n_bands}")
    print(f"VASP估算: {vasp_mem:.1f} GB")
    print(f"QE估算: {qe_mem:.1f} GB")
    
    return vasp_mem, qe_mem

# 使用示例
estimate_memory(
    n_atoms=100,
    n_electrons=800,
    encut=520,
    kpoints=[8, 8, 8]
)
```

---

## 3. IO优化

### 3.1 文件系统选择

| 存储类型 | 速度 | 容量 | 用途 |
|----------|------|------|------|
| NVMe SSD | 极高 | 小 | 临时文件, WAVECAR |
| 并行Lustre | 高 | 大 | 计算目录 |
| NFS | 中 | 大 | home目录 |
| 磁带/对象 | 低 | 极大 | 归档 |

**最佳实践**:
```bash
# 1. 在快速存储上运行
export VASP_RUNDIR=/scratch/$USER/$JOB_ID
mkdir -p $VASP_RUNDIR
cd $VASP_RUNDIR

# 2. 复制输入文件
cp $SLURM_SUBMIT_DIR/{INCAR,KPOINTS,POSCAR,POTCAR} .

# 3. 运行计算
mpirun vasp_std

# 4. 只复制必要输出回提交目录
cp {OUTCAR,OSZICAR,CONTCAR,vasprun.xml} $SLURM_SUBMIT_DIR/

# 5. 清理
rm -rf $VASP_RUNDIR
```

### 3.2 VASP IO优化

```
# INCAR IO设置
LWAVE = .FALSE.     # 不保存WAVECAR (节省IO)
LCHARG = .FALSE.    # 不保存CHGCAR
LORBIT = 0          # 最小化DOSCAR输出
NWRITE = 1          # 减少OUTCAR输出

# 仅需要时开启
LWAVE = .TRUE.      # 后续计算需要波函数
LCHARG = .TRUE.      # 需要电荷密度分析
```

### 3.3 临时文件管理

```bash
#!/bin/bash
# optimize_io.sh - IO优化脚本

# 设置临时目录到快速存储
export TMPDIR=/dev/shm/$USER  # 共享内存 (RAM disk)
# 或使用SSD
export TMPDIR=/ssd/$USER/$SLURM_JOB_ID

# 创建并进入临时目录
mkdir -p $TMPDIR
cd $TMPDIR

# 复制输入
cp $SLURM_SUBMIT_DIR/* .

# 运行计算
mpirun vasp_std

# 压缩并复制结果
tar czf results.tgz OUTCAR vasprun.xml
mv results.tgz $SLURM_SUBMIT_DIR/

# 清理
cd $SLURM_SUBMIT_DIR
rm -rf $TMPDIR
```

---

## 4. 算法优化

### 4.1 电子步收敛加速

**VASP混合算法**:
```
# 线性混合 (小体系)
ALGO = Normal
IMIX = 4
AMIX = 0.4
BMIX = 1.0

# Kerker混合 (大体系/金属)
ALGO = Fast
IMIX = 1
AMIX = 0.2
BMIX = 0.0001
AMIN = 0.1

# 金属体系推荐
ALGO = Fast
ISMEAR = 1
SIGMA = 0.2
NELMIN = 6
NELMDL = -12
```

**QE对角化选项**:
```fortran
&ELECTRONS
   diagonalization = 'david'   ! 'david' 或 'cg'
   mixing_mode = 'local-TF'    ! 'plain', 'TF', 'local-TF'
   mixing_beta = 0.7
   conv_thr = 1.0d-8
/
```

### 4.2 离子步优化

**VASP优化算法选择**:
```
# 共轭梯度 (推荐)
IBRION = 2
POTIM = 0.1

# RMM-DIIS (快速但可能不稳定)
IBRION = 1
EDIFFG = -0.01

# 阻尼分子动力学 (困难体系)
IBRION = 3
SMASS = 2.0

# 准牛顿法 (近平衡)
IBRION = 1
NFREE = 10
```

### 4.3 大体系策略

**分块计算**:
```python
def fragment_calculation(large_structure, fragment_size=50):
    """
    大体系分块计算策略
    """
    from ase import Atoms
    from ase.neighborlist import neighbor_list
    
    # 识别片段
    fragments = identify_fragments(large_structure, fragment_size)
    
    energies = []
    for frag in fragments:
        # 添加缓冲层
        buffered = add_buffer(frag, large_structure, radius=5.0)
        
        # 计算片段
        energy = calculate_fragment(buffered)
        energies.append(energy)
    
    # 总和
    total_energy = sum(energies)
    return total_energy
```

---

## 5. 快速检查清单

### 5.1 提交前检查

```bash
#!/bin/bash
# preflight_check.sh

echo "=== DFT计算预检 ==="

# 1. 检查输入文件完整性
for file in INCAR POSCAR KPOINTS POTCAR; do
    if [ ! -f $file ]; then
        echo "错误: 缺少 $file"
        exit 1
    fi
done
echo "✓ 输入文件完整"

# 2. 检查资源申请
if [ -z "$SLURM_NTASKS" ]; then
    echo "警告: 未检测到SLURM环境"
fi
echo "✓ 申请核数: $SLURM_NTASKS"

# 3. 估算内存需求
python -c "
import sys
n_atoms = int(open('POSCAR').readlines()[6].split()[0])
print(f'估计内存需求: {n_atoms * 0.1:.1f} GB')
"

# 4. 检查k点合理性
kpoints=$(head -4 KPOINTS | tail -1)
echo "✓ k点网格: $kpoints"

# 5. 检查ENCUT
encut=$(grep ENCUT INCAR | awk '{print $3}')
echo "✓ ENCUT: $encut eV"

echo "=== 预检完成 ==="
```

### 5.2 运行中监控

```bash
#!/bin/bash
# monitor_job.sh

JOB_ID=$1

echo "监控作业 $JOB_ID"

# 监控资源使用
while squeue -j $JOB_ID | grep -q $JOB_ID; do
    # CPU使用率
    sstat -j $JOB_ID --format=JobID,CPUUtil,AveRSS,MaxRSS
    
    # 检查收敛
    if [ -f OUTCAR ]; then
        energy=$(grep "energy  without" OUTCAR | tail -1 | awk '{print $4}')
        echo "当前能量: $energy eV"
    fi
    
    sleep 60
done
```

---

## 6. 性能基准

### 6.1 标准测试体系

**Si单胞 (8原子)**:
- k点: 8×8×8
- ENCUT: 520 eV
- 基准时间: 60s (64核)

**Si超胞 (216原子)**:
- k点: 2×2×2
- ENCUT: 520 eV
- 基准时间: 600s (256核)

### 6.2 扩展性测试

```python
def scaling_test():
    """
    并行扩展性测试
    """
    import subprocess
    import time
    
    cores_list = [16, 32, 64, 128, 256]
    times = []
    
    for cores in cores_list:
        start = time.time()
        
        # 提交作业
        cmd = f"sbatch --ntasks={cores} job_script.sh"
        subprocess.run(cmd, shell=True)
        
        # 等待完成
        wait_for_job()
        
        elapsed = time.time() - start
        times.append(elapsed)
        
        print(f"{cores} cores: {elapsed:.1f} s")
    
    # 计算效率
    base_time = times[0]
    for cores, t in zip(cores_list, times):
        efficiency = (base_time * cores_list[0] / cores) / t
        print(f"{cores} cores efficiency: {efficiency:.1%}")
```

---

## 参考

1. VASP Wiki: [Performance Tuning](https://www.vasp.at/wiki/index.php/Performance_tuning)
2. QE用户指南: [Parallelization](https://www.quantum-espresso.org/Doc/user_guide/node20.html)
3. NERSC: [Optimizing VASP](https://docs.nersc.gov/applications/vasp/)
