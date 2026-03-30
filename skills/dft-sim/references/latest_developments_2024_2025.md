# 2024-2025年第一性原理计算最新进展

> **文档更新时间**: 2026-03-08 15:50
> **维护状态**: 持续更新中

## VASP 最新进展 (6.4 - 6.5)

### VASP 6.5.1 (2025年3月发布)
- **错误修复**: 针对6.5.0的多项bug修复
- **MLFF增强**: 机器学习力场相关模拟协议的改进
- **MLFF重拟合选项**: 提升性能的重拟合功能

### VASP 6.5.0 (2024年12月发布)

#### 1. 电子-声子耦合
- **零点能带重整化**: 自动计算温度依赖的能带结构
- **输运系数**: 基于线性化Boltzmann输运方程的计算
- **应用**: 热电材料、电子器件设计

```
# INCAR示例
SYSTEM = Electron-Phonon Coupling
ALGO = Normal
ISMEAR = -5
# 需要配合PHONON计算使用
```

#### 2. Python插件接口
- **功能**: 通过C++接口将VASP与Python连接
- **应用**: 自定义算法、机器学习集成、自动化工作流
- **要求**: Python 3.8+, pybind11

```python
# Python插件示例
# 可在VASP运行时修改内部参数
import vasp_plugins

class CustomPlugin:
    def __init__(self):
        pass
    
    def on_step(self, step, energy, forces):
        # 自定义每步操作
        pass
```

#### 3. Bethe-Salpeter方程 (BSE) 改进
- **Lanczos对角化** (IBSE=3): 更高效的BSE求解
- **GPU支持**: 时间演化BSE的GPU加速
- **适用**: 大体系光学性质计算

```
# INCAR
ALGO = BSE
IBSE = 3          # Lanczos方法
LHFCALC = .TRUE.
```

#### 4. 新的交换关联泛函
- **(r)MS-B86bl, (r)MS-PBEl, (r)MS-RPBEl**: 新的meta-GGA泛函
- **TASK和LAK泛函**: 由Timo Lebeda提供
- **无源交换关联B场** (LSFBXC)

```
# INCAR
METAGGA = MSB86BL   # 或 MSPBEL, MSRPBEL, TASK, LAK
```

#### 5. 库仑核截断 (Coulomb Kernel Truncation)
- **功能**: 开放边界条件
- **应用**: 
  - 偶极分子计算
  - 带电分子
  - 2D材料
  - 表面计算

```
# INCAR
KERNEL_TRUNCATION = .TRUE.
LTRUNCATE = .TRUE.
```

#### 6. 机器学习力场 (MLFF) 改进
- **溢出因子** (ML_IERR): 作为力场误差的估计
- **快速执行模式**: 可与误差估计结合使用
- **VASPml库**: C++库支持LAMMPS接口

```
# INCAR - MLFF快速预测
ML_LMLFF = .TRUE.
ML_MODE = run
ML_FF = ML_FFN
ML_IERR = 1         # 启用溢出因子估计
```

#### 7. 其他新功能
- **外部力** (EFOR): 可施加自定义外力
- **电子结构因子样条插值** (ESF_SPLINES): 加速RPA相关能量收敛
- **Müller-Plathe方法**: 热导率计算

### VASP 6.4.0-6.4.3 (2024年)

#### 主要改进

1. **MLFF快速预测模式**
   - 速度提升: 20-100倍
   - 改进的近邻列表算法 (Cell列表 + Verlet列表)
   - ML_MODE超级标签简化使用

2. **HDF5输出增强**
   - 部分电荷写入HDF5而非PARCHG文件
   - OSZICAR内容写入vaspout.h5
   - 时间演化BSE的介电函数写入HDF5
   - LSYNCH5: 运行时同步HDF5文件

3. **杂化泛函GPU加速**
   - OpenACC移植
   - ACFDTR算法GPU支持

4. **新的XC泛函组合**
   - XC和XC_C标签: 线性组合XC泛函
   - v1-sregTM, v2-sregTM等meta-GGA

5. **VASP-TRIQS接口改进**
   - 使用HDF5文件 (vaspgamma.h5)
   - DFT+DMFT计算更便捷

## Quantum ESPRESSO 最新进展 (7.3 - 7.5)

### QE 7.5 (2025年12月发布)

#### 1. 新的轨道分辨DFT+U方法 (-E)
- **功能**: 基于轨道分辨的DFT+U实现
- **优势**: 更精确的Hubbard校正
- **应用**: 强关联电子体系

#### 2. Wannier90-DFT+U接口
- **功能**: 使用Wannier函数作为Hubbard投影器
- **实现者**: I. Timrov等
- **意义**: 统一了cRPA和线性响应方法，提高U参数的可转移性
- **参考**: Carta et al., PRB (2024) - 展示KCuF₃和Sr₂FeO₄的应用

```fortran
&SYSTEM
  input_dft = 'vdw-df2',
  hubbard_projectors = 'wannier',
/
```

#### 3. 双化学势声子计算扩展
- **功能**: 线性响应声子计算支持两个化学势
- **应用**: 热电材料、非平衡体系
- **实现者**: G. Stenuit

#### 4. GPU加速持续优化
- CUDA核心模块性能提升
- OpenACC并行区域改进
- A100 GPU上保持3倍加速

### QE 7.4 (2024年发布)

#### 1. GPU全面支持
- **CUDA加速**: 核心计算模块CUDA化
- **OpenACC**: 并行区域OpenACC优化
- **性能提升**: 
  - A100 GPU上可达3倍加速
  - 成本降低约75%

```bash
# CMake配置示例
cmake .. \
  -DQE_ENABLE_CUDA=ON \
  -DQE_ENABLE_OPENACC=ON \
  -DNVFORTRAN_CUDA_CC=80 \
  -DQE_ENABLE_MPI_GPU_AWARE=ON
```

#### 2. 双化学势声子计算
- **功能**: 线性响应声子计算支持两个化学势
- **应用**: 热电材料、非平衡体系
- **实现**: 由G. Stenuit贡献

```fortran
&INPUTPH
  twochem = .true.
  el_ph_sigma = 0.01
  el_ph_nsigma = 10
/
```

#### 3. 对称性检测改进
- **功能**: 通过物种比较检测对称性
- **优势**: 更准确地识别复杂结构的空间群
- **应用**: 高通量计算、自动化工作流

#### 4. 性能优化
- **CPU版本**: 相比7.3提升10-20%
- **内存优化**: 大体系计算内存占用降低
- **并行扩展性**: 改进大规模并行效率

### QE 7.3 (2024年初发布)

#### 主要特性
1. **改进的CMake构建系统**
2. **新的赝势库支持**
3. **增强的DFT+U实现**
4. **改进的声子计算稳定性**

## 行业趋势与展望

### 1. Hubbard参数机器学习预测 (2024重大突破)

#### HubbardML - 自洽Hubbard校正零成本方案
- **作者**: Uhrin et al. (EPFL/MARVEL, 2024)
- **方法**: 使用等变神经网络预测自洽Hubbard U和V参数
- **输入**: 完整的在位占据矩阵（局部电荷密度）
- **精度**: U参数误差3%，V参数误差5%
- **优势**: 
  - 完全取代线性响应计算
  - 训练数据需求少
  - 对未见氧化态外推性好
- **arXiv**: 2406.02457

**意义**: 消除领域专业知识和大量计算资源的需求，实现广泛使用的自洽Hubbard参数

#### Koopmans谱泛函的机器学习加速
- **作者**: Linscott et al. (PSI, 2024)
- **方法**: 从DFT轨道密度预测屏蔽参数
- **精度**: 平均差异 < 20 meV
- **arXiv**: 2406.15205
- **应用**: 温度依赖的能谱性质预测

### 2. GPU计算成为主流

| 平台 | GPU支持状态 | 性能提升 |
|------|------------|---------|
| VASP | OpenACC部分支持 | 2-5倍 |
| QE 7.4+ | CUDA + OpenACC全面支持 | 3-5倍 |
| ABACUS | 原生GPU支持 | 5-10倍 |

**建议**: 新部署优先考虑GPU版本

### 3. 机器学习力场(MLFF)革命性进展

#### VASP MLFF应用突破 (2024-2025)
| 应用领域 | 成果 | 参考文献 |
|---------|------|----------|
| MOF材料 | 近DFT精度，速度提升100倍 | npj Comput Mater (2024) |
| 熔盐电解质 | 化学势计算，热力学性质 | Chem. Sci. (2025) |
| 钚氧化物 | 首次应用于锕系元素 | Mater. Today Commun. (2025) |
| 硫化物电解质 | 预训练模型，宽化学空间覆盖 | npj Comput Mater (2025) |

#### 通用机器学习势
- **MACE**: 等变消息传递网络，高精度
- **CHGNet**: 图神经网络势，MatGL框架
- **MatterSim**: Microsoft开发的通用势
- **DPA-2/M3GNet**: 大规模预训练模型

**关键发现**: 
- VASP MLFF训练仅需10-100个DFT参考结构
- 预测速度比DFT快100-1000倍
- 非平衡构型(nEQ)数据显著提升MLFF性能

### 4. 电声耦合与超导性研究进展

#### 二维超导体高通量筛选 (2025)
- **研究**: Mater. Horiz. 12, 3408 (2025)
- **方法**: DFPT + 机器学习
- **规模**: 筛选14万+二维化合物
- **发现**: 
  - 105个系统Tc > 5K
  - CuH₂, NbN, V₂NS₂等高温超导候选材料
  - 2D材料电声耦合强于3D对应物

#### 电声耦合计算方法进展
- **超导Tc预测**: Migdal-Eliashberg理论 + McMillan-Allen-Dynes公式
- **非绝热效应**: 考虑非绝热声子的修正
- **关联材料**: DFT+U/DMFT结合电声耦合

### 5. 激发态计算方法突破

#### GW-BSE激发态力计算 (2025年3月)
- **作者**: Alrahamneh et al. (Padova大学)
- **期刊**: Int. J. Mol. Sci. 26, 2306 (2025)
- **方法**: Hellmann-Feynman定理 + 有限差分
- **创新**: 单次BSE计算即可获得原子力
- **验证**: CO和CH₂O分子与量子化学方法结果一致
- **应用**: 复杂系统中的能级交叉问题

#### BSE方法改进
- **Lanczos对角化** (IBSE=3): 大体系光学性质计算
- **GPU加速**: 时间演化BSE的GPU支持
- **激子-声子耦合**: 激发态与声子相互作用

### 6. 自动化与高通量

- **AiiDA**: 工作流自动化，数据库集成
- **Materials Project**: 大规模高通量计算
- **NOMAD**: 数据共享与复现平台
- **Alexandria数据库**: 14万+二维材料结构

### 7. 混合精度计算

- **FP32/FP64混合**: 平衡精度与性能
- **低秩近似**: 加速电子结构计算
- **随机DFT**: 大体系近似方法

## 实践建议

### 对于新用户
1. **选择QE 7.4**: 开源免费，GPU支持好
2. **学习MLFF**: 掌握VASP 6.4+的机器学习力场
3. **关注自动化**: 使用AiiDA等工作流工具

### 对于有经验用户
1. **升级VASP 6.5**: 获取最新功能
2. **尝试GPU版本**: 特别是QE 7.4+
3. **集成Python**: 利用VASP 6.5的Python接口

### 对于机构部署
1. **硬件规划**: 考虑GPU集群
2. **软件栈**: 统一MPI/OpenMP/GPU环境
3. **培训**: 组织MLFF和GPU计算培训

## 参考文献

### 2025年
1. Alrahamneh et al., Int. J. Mol. Sci. 26, 2306 (2025) - GW-BSE激发态力计算
2. VASP 6.5.1 Release Notes (2025年3月)
3. Mater. Horiz. 12, 3408 (2025) - 二维超导体机器学习预测

### 2024年
4. VASP 6.5.0 Release Notes (2024年12月)
5. Quantum ESPRESSO 7.4/7.5 Release Notes (2024-2025)
6. Uhrin et al., arXiv:2406.02457 (2024) - HubbardML
7. Linscott et al., arXiv:2406.15205 (2024) - Koopmans谱泛函ML加速
8. Spiga et al., QE-GPU性能研究 (2024)
9. Jinnouchi et al., MLFF综述 (2024)
10. Oracle Cloud QE GPU Benchmark (2024)
11. npj Comput Mater (2024) - MOF的MLFF应用
12. Chem. Sci. (2025) - 熔盐电解质MLFF
13. npj Comput Mater (2025) - 硫化物电解质预训练模型
14. Maniar et al., PNAS (2025) - DFT自相互作用校正在过渡金属中的问题

---

*本文档由DFT-Sim技能库持续维护，最后更新: 2026-03-08*
