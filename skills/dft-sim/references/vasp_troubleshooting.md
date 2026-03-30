# VASP常见错误排查指南

## 1. 编译错误

### 错误: "undefined reference to 'mpi_init_'"
**原因**: MPI库链接问题
**解决**:
```bash
# 检查MPI环境
which mpirun
mpif90 --showme

# 确保使用正确的MPI编译器
export MPIF90=/usr/lib/openmpi/bin/mpif90
make std
```

### 错误: "MKL not found"
**解决**:
```bash
# 加载Intel环境
source /opt/intel/oneapi/setvars.sh

# 或在makefile.include中硬编码路径
MKLROOT = /opt/intel/oneapi/mkl/latest
```

### 错误: "fft3dlib.o: No such file"
**解决**:
```bash
# 清理并重新编译
make veryclean
make std
```

## 2. 运行时错误

### 错误: "Error reading item 'VCA' from file INCAR"
**原因**: INCAR中有不识别的标签
**解决**: 检查INCAR拼写，删除或注释掉无效标签

### 错误: "VERY BAD NEWS! internal error in subroutine SGRCON"
**原因**: 对称性检测问题
**解决**:
```
# INCAR中添加
ISYM = 0    # 关闭对称性
# 或
SYMPREC = 1E-5   # 放宽对称性精度
```

### 错误: "ZBRENT: fatal error in bracketing"
**原因**: 电子步不收敛
**解决**:
```
# INCAR中调整
ALGO = Normal       # 使用更稳定的算法
NELMIN = 6          # 最小电子步数
MIXING = 0.4        # 降低混合参数
AMIX = 0.2
BMIX = 0.0001
```

### 错误: "EDDDAV: Call to ZHEGV failed"
**原因**: 矩阵对角化失败
**解决**:
```
# INCAR中
ALGO = Fast         # 改用RMM-DIIS
# 或增加NBANDS
NBANDS = 1.3 * 默认值
```

### 错误: "The distance between some ions is very small"
**原因**: 原子位置重叠
**解决**: 检查POSCAR，确保原子间距合理 (>0.5 Å)

## 3. 收敛问题

### SCF不收敛
**症状**: 能量振荡或不下降
**解决**:
```
# 方法1: 调整展宽
ISMEAR = 0
SIGMA = 0.05        # 减小SIGMA

# 方法2: 调整混合
AMIX = 0.2
BMIX = 0.0001
AMIX_MAG = 0.4
BMIX_MAG = 0.0001

# 方法3: 改变算法
ALGO = Normal       # 比Fast更稳定
# 或
ALGO = All          # 全对角化

# 方法4: 线性混合
IMIX = 4
```

### 离子弛豫不收敛
**症状**: 力无法降到收敛标准
**解决**:
```
# 增加步数
NSW = 200

# 调整步长
POTIM = 0.1         # 减小步长

# 改变算法
IBRION = 1          # 准牛顿法
# 或
IBRION = 2          # 共轭梯度

# 放宽收敛标准 (如果接近收敛)
EDIFFG = -0.05      # 默认-0.01可能太严格
```

## 4. 内存问题

### 错误: "Allocation failed"
**解决**:
```
# 降低并行度
NCORE = 2           # 减小NCORE

# 减少k点
# KPOINTS中使用更粗的网格

# 降低截断能
ENCUT = 400         # 临时降低
```

### 错误: "segmentation fault"
**可能原因**:
1. 栈内存不足
```bash
ulimit -s unlimited
```

2. 内存泄漏
```
# INCAR中
LWAVE = .FALSE.     # 不写入大文件
LCHARG = .FALSE.
```

## 5. MLFF相关问题

### 错误: "ML_FF file not found"
**解决**:
```bash
# 确保ML_FFN文件存在并复制为ML_FF
cp ML_FFN ML_FF
```

### 错误: "Spilling factor too large"
**原因**: MLFF精度不足
**解决**:
```
# 重新训练或增加训练数据
ML_MODE = train
ML_CDOUB = 20       # 增加数据量
```

## 6. 性能问题

### 计算速度过慢
**优化建议**:
```
# 1. 优化并行设置
NCORE = 4           # 每核芯数
KPAR = 4            # k点并行 (≤k点数)

# 2. 使用RMM-DIIS
ALGO = Fast

# 3. 降低收敛标准 (初步计算)
EDIFF = 1E-5

# 4. 使用WAVECAR重启
ISTART = 1
```

## 7. 输出文件问题

### OSZICAR为空或不更新
**原因**: 缓冲问题
**解决**:
```bash
# 使用无缓冲输出
mpirun -np 4 vasp_std | tee vasp.out
```

### WAVECAR损坏
**解决**:
```
# INCAR中
ISTART = 0          # 从头开始
# 或删除WAVECAR
rm WAVECAR
```

## 8. 特定计算类型问题

### 能带计算错误
```
# 确保先完成自洽计算
ISTART = 1
ICHARG = 11

# 检查KPOINTS格式 (Line-mode)
```

### HSE计算内存不足
```
# 减少NKRED
NKRED = 2
# 或使用更少的k点
```

### MD模拟能量不守恒
```
# 减小步长
POTIM = 0.5

# 检查初始结构是否优化充分
```

## 调试技巧

### 1. 使用--dry-run检查输入
```bash
vasp_std --dry-run
```

### 2. 逐步增加复杂度
- 先运行小体系测试
- 逐步增加k点、截断能

### 3. 查看详细输出
```
# INCAR中
NWRITE = 2          # 详细输出
```

### 4. 监控计算过程
```bash
# 实时查看能量
tail -f OSZICAR

# 查看资源使用
top -u $USER
```

## 参考资源

- VASP Wiki: https://www.vasp.at/wiki/
- VASP论坛: https://www.vasp.at/forum/
