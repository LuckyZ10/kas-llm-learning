# VASP机器学习力场 (MLFF) 训练工作流

## 概述

VASP 6.4+ 引入了机器学习力场 (MLFF) 功能，可以：
- 训练神经网络势函数
- 加速分子动力学模拟 20-100 倍
- 保持接近DFT的精度

## 工作流程

### 步骤1: 准备训练数据

创建初始结构文件 `POSCAR`:
```
Si MLFF Training
1.0
5.43 0.00 0.00
0.00 5.43 0.00
0.00 0.00 5.43
Si
8
direct
0.000 0.000 0.000
0.250 0.250 0.000
0.250 0.000 0.250
0.000 0.250 0.250
0.500 0.500 0.000
0.750 0.750 0.000
0.750 0.500 0.250
0.500 0.750 0.250
```

### 步骤2: 初始训练运行

**INCAR**:
```
# 基础设置
SYSTEM = Si MLFF Training - Stage 1
ENCUT = 520
ISMEAR = 0
SIGMA = 0.1
EDIFF = 1E-5

# MLFF设置
ML_LMLFF = .TRUE.
ML_MODE = train         # 训练模式
ML_CDOUB = 10           # 数据倍增因子
ML_CTIFOR = 0.01        # 力误差阈值 (eV/Å)

# MD设置 (生成训练数据)
IBRION = 0              # MD模拟
NSW = 5000              # 步数
POTIM = 1.0             # 时间步长 (fs)
MDALGO = 2              # Nose-Hoover
TEBEG = 300             # 起始温度 (K)
TEEND = 1200            # 结束温度 (升温探索)
SMASS = 0.5

# 输出控制
ML_OUTBLOCK = 50        # 输出频率
```

**运行训练**:
```bash
mpirun -np 16 vasp_std
```

**输出文件**:
- `ML_ABN`: 训练数据库
- `ML_FFN`: 训练好的力场
- `ML_LOGFILE`: 训练日志

### 步骤3: 检查训练质量

查看训练日志:
```bash
# 检查误差统计
grep "ERR" ML_LOGFILE | tail -20

# 检查训练数据点数量
grep "NCONF" ML_LOGFILE | tail -5
```

**关键指标**:
- ERR: 力误差 (应 < 0.1 eV/Å)
- NCONF: 配置数量 (应 > 100)

### 步骤4: 迭代训练 (可选)

如果训练不足，继续训练:

```bash
# 复制训练数据
cp ML_ABN ML_AB
cp CONTCAR POSCAR       # 使用最终结构

# 修改INCAR继续训练
# ML_MODE = train (保持不变)
# 可以增加 NSW 或改变温度范围
```

### 步骤5: 力场重选 (可选)

优化力场参数:
```
# INCAR
ML_MODE = select        # 重选局部参考构型
ML_CDOUB = 10
NSW = 1                 # 只重选，不运行MD
```

```bash
mpirun -np 16 vasp_std
# 生成新的 ML_FFN
```

### 步骤6: 生产运行 (预测模式)

**准备文件**:
```bash
# 复制训练好的力场
cp ML_FFN ML_FF

# 准备新的POSCAR (可以是更大体系或不同条件)
```

**INCAR**:
```
# 基础设置
SYSTEM = Si MLFF Production
ENCUT = 400             # 可降低截断能 (MLFF不需要DFT精度)
ISMEAR = 0
SIGMA = 0.1

# MLFF预测模式
ML_LMLFF = .TRUE.
ML_MODE = run           # 纯预测模式
ML_FF = ML_FF           # 读取力场

# MD设置
IBRION = 0
NSW = 100000            # 长轨迹
POTIM = 1.0
MDALGO = 2
TEBEG = 300
TEEND = 300
SMASS = 0.5

# 输出
ML_IERR = 1             # 输出溢出因子 (误差估计)
```

**运行**:
```bash
mpirun -np 4 vasp_std   # MLFF可以用更少核
```

## 高级技巧

### 多元素体系

对于多元素体系 (如SiO2):
```
# INCAR
ML_MODE = train
ML_CDOUB = 20           # 增加数据量
ML_CTIFOR = 0.02        # 放宽阈值 (多元素更难)

# 确保所有元素都有足够训练数据
```

### 主动学习策略

```bash
#!/bin/bash
# 主动学习循环脚本

for i in {1..5}; do
    echo "Iteration $i"
    
    # 运行MLFF MD
    mpirun -np 16 vasp_std
    
    # 检查溢出因子
    max_spilling=$(grep "SPILLING" ML_LOGFILE | tail -1 | awk '{print $3}')
    
    if (( $(echo "$max_spilling > 0.5" | bc -l) )); then
        echo "High spilling detected, retraining..."
        cp ML_ABN ML_AB
        # 继续训练
    else
        echo "MLFF quality good, stopping."
        break
    fi
done
```

### 与LAMMPS联用

VASP 6.5+ 支持将MLFF导出到LAMMPS:

```bash
# 使用VASPml库编译LAMMPS
# 在LAMMPS中使用:
# pair_style vaspml
# pair_coeff * * ML_FFN Si
```

## 常见问题

### Q: 训练数据需要多少?
**A**: 通常100-1000个构型，取决于体系复杂度。

### Q: 如何验证MLFF精度?
**A**: 
1. 比较MLFF和DFT的力/能量
2. 检查径向分布函数
3. 比较声子谱

### Q: MLFF可以用于什么?
**A**: 
- 长时分子动力学
- 热导率计算
- 相变研究
- 缺陷扩散

## 性能对比

| 方法 | 速度 | 精度 | 适用场景 |
|------|------|------|---------|
| DFT MD | 1x | 基准 | 短轨迹,高精度 |
| MLFF MD | 20-100x | ~DFT | 长轨迹,统计采样 |
| 经典力场 | 1000x+ | 低 | 定性研究 |

## 参考

- VASP Wiki MLFF: https://www.vasp.at/wiki/ML_MODE
- Jinnouchi et al., PRB 100, 014105 (2019)
