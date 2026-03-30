# DFT-Sim技能库维护日志

**维护时间**: 2026-03-08 15:49 - 16:30
**维护人员**: DFT-Sim子代理

---

## 【DFT-Sim维护】15:50 - 开始执行维护任务

检查现有文档结构，开始搜索2024-2025年最新DFT研究进展。

---

## 【DFT-Sim维护】15:55 - 发现VASP/QE重要版本更新

### VASP版本更新
- **VASP 6.5.0** (2024年12月发布)
- **VASP 6.5.1** (2025年3月发布) - bug修复和MLFF增强

### Quantum ESPRESSO版本更新
- **QE 7.5** (2025年12月发布) - 重要更新！
  - 新的轨道分辨DFT+U方法
  - Wannier90-DFT+U接口 (使用Wannier函数作为Hubbard投影器)
  - 双化学势声子计算

---

## 【DFT-Sim维护】16:00 - 发现机器学习DFT重大进展

### 1. HubbardML - 自洽Hubbard参数零成本方案
- **作者**: Uhrin et al. (EPFL/MARVEL)
- **arXiv**: 2406.02457 (2024)
- **突破**: 使用等变神经网络预测Hubbard U/V参数
- **精度**: U误差3%，V误差5%
- **意义**: 完全取代线性响应计算，实现广泛使用的自洽Hubbard校正

### 2. Koopmans谱泛函的ML加速
- **作者**: Linscott et al. (PSI)
- **arXiv**: 2406.15205 (2024)
- **突破**: 从DFT轨道密度预测屏蔽参数
- **精度**: 平均差异 < 20 meV

### 3. 通用ML势的PES软化问题 (Deng et al., 2025)
- **发现**: M3GNet/CHGNet/MACE-MP-0存在系统性势能面软化
- **表现**: 表面/缺陷能量低估、离子迁移势垒低估、声子频率低估
- **解决**: 通过少量(~100)OOD数据微调可高效修正
- **参考**: B. Deng et al., Nat. Commun. (2025)

### 4. 微调基础模型时的灾难性遗忘 (2025)
- **发现**: Fe系统微调研究表明
  - CHGNet和SevenNet-O: 学习率≤0.0001时遗忘轻微
  - MACE: 即使使用冻结层和数据重放，仍存在显著遗忘
- **建议**: 低学习率(≤0.0001)微调，谨慎选择架构

---

## 【DFT-Sim维护】16:05 - 发现电声耦合与超导性研究进展

### 二维超导体高通量筛选 (2025年2月)
- **期刊**: Mater. Horiz. 12, 3408 (2025)
- **方法**: DFPT + 机器学习
- **规模**: 筛选14万+二维化合物 (Alexandria数据库)
- **发现**:
  - 105个系统Tc > 5K
  - CuH₂, NbN, V₂NS₂等高温超导候选材料
  - 2D材料电声耦合强于3D对应物

### VASP 6.5电子-声子耦合新功能
- 零点能带重整化
- 输运系数计算
- 温度依赖的能带结构

---

## 【DFT-Sim维护】16:10 - 发现激发态计算方法突破

### GW-BSE激发态力计算 (2025年3月)
- **作者**: Alrahamneh et al. (Padova大学)
- **期刊**: Int. J. Mol. Sci. 26, 2306 (2025)
- **方法**: Hellmann-Feynman定理 + 有限差分
- **创新**: 单次BSE计算即可获得原子力
- **验证**: CO和CH₂O分子与量子化学方法结果一致

### 机器学习预测sTDA参数 (2024)
- **作者**: Wang et al. (北京师范大学)
- **突破**: ML预测sTDA方法的最优Fock交换混合参数
- **精度**: MAE < 0.004, R² > 0.96
- **应用**: 大规模分子激发态高通量筛选

### Ensemble TDDFT (2025)
- **作者**: D. C. Baker et al.
- **创新**: 结合Ensemble DFT和线性响应TDDFT
- **能力**: 访问基态和激发态之间的跃迁信息

---

## 【DFT-Sim维护】16:15 - 更新文档

已更新以下文档：
1. `references/latest_developments_2024_2025.md` - 添加QE 7.5、HubbardML、Koopmans泛函、PES软化等内容
2. `references/ml_assisted_dft.md` - 添加PES软化问题、微调策略、SevenNet-O等新模型

---

## 【DFT-Sim维护】16:20 - 发现更多MLFF应用进展

### MLFF在材料科学中的应用突破
| 应用领域 | 成果 | 参考文献 |
|---------|------|----------|
| MOF材料 | 近DFT精度，速度提升100倍 | npj Comput Mater (2024) |
| 熔盐电解质 | 化学势计算，热力学性质 | Chem. Sci. (2025) |
| 钚氧化物 | 首次应用于锕系元素 | Mater. Today Commun. (2025) |
| 硫化物电解质 | 预训练模型，宽化学空间覆盖 | npj Comput Mater (2025) |

### 新的通用ML势
- **SevenNet-O** (2024): 基于NequIP架构，支持大规模并行MD
- **MatterSim** (2024): Microsoft开发的通用材料势
- **MACE-MP-0**: Materials Project大规模训练模型

---

## 【DFT-Sim维护】16:25 - 搜索DFT+U和缺陷计算最新进展

### DFT+U最新进展
- Wannier函数作为Hubbard投影器 (Carta et al., 2024)
- cRPA与线性响应方法的统一框架
- 适用于DFT+U和DFT+DMFT的一致性U参数计算

### 自相互作用校正(SIC)研究 (2025年3月)
- **发现**: Perdew-Zunger SIC在过渡金属中s/d电子能量平衡问题
- **作者**: Maniar et al., PNAS (2025)
- **意义**: 为改进DFT过渡金属描述指明方向

---

## 【DFT-Sim维护】16:30 - 维护任务总结

### 完成的更新
1. ✅ 更新了最新进展文档 (latest_developments_2024_2025.md)
   - VASP 6.5.1信息
   - QE 7.5新功能
   - HubbardML和Koopmans泛函
   - MLFF应用突破
   - 电声耦合与超导性进展
   - 激发态计算方法

2. ✅ 更新了ML辅助DFT文档 (ml_assisted_dft.md)
   - PES软化问题与解决方案
   - 微调策略和灾难性遗忘
   - 新的通用ML势模型

### 新增重要参考文献 (2024-2025)
1. Alrahamneh et al., Int. J. Mol. Sci. 26, 2306 (2025) - GW-BSE激发态力
2. Uhrin et al., arXiv:2406.02457 (2024) - HubbardML
3. Linscott et al., arXiv:2406.15205 (2024) - Koopmans谱泛函
4. Deng et al., Nat. Commun. (2025) - PES软化
5. Mater. Horiz. 12, 3408 (2025) - 二维超导体筛选
6. Maniar et al., PNAS (2025) - SIC在过渡金属中的问题
7. Wang et al., Chem. Sci. (2025) - ML预测sTDA参数
8. npj Comput Mater (2024/2025) - MLFF在MOF和电解质中的应用

### 建议后续更新
1. 添加GW-BSE激发态力的实际计算示例
2. 创建HubbardML使用指南
3. 更新MLFF训练最佳实践 (考虑PES软化)
4. 添加电声耦合计算教程 (VASP 6.5+)

---

**维护完成时间**: 2026-03-08 16:30
**状态**: ✅ 完成
