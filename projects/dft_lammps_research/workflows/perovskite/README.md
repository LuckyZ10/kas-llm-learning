# 钙钛矿稳定性预测案例
# Perovskite Stability Prediction Case Study

## 目录
- [概述](#概述)
- [科学背景](#科学背景)
- [工作流程](#工作流程)
- [文件说明](#文件说明)
- [使用方法](#使用方法)
- [结果解读](#结果解读)
- [参考文献](#参考文献)

---

## 概述

本案例研究演示如何使用Goldschmidt容忍因子和DFT计算预测卤化物钙钛矿的稳定性。

### 目标
- **预测钙钛矿结构稳定性**
- **识别可合成的候选材料**
- **预测相变温度**
- **提供合成建议**

### 应用意义
钙钛矿太阳能电池效率已超过25%，但稳定性仍是商业化瓶颈。本案例帮助筛选稳定的高性能钙钛矿材料。

---

## 科学背景

### 钙钛矿结构

钙钛矿通式为ABX₃，其中:
- **A位**: 大阳离子 (Cs⁺, MA⁺, FA⁺等)
- **B位**: 金属阳离子 (Pb²⁺, Sn²⁺等)  
- **X位**: 卤素阴离子 (I⁻, Br⁻, Cl⁻)

### Goldschmidt容忍因子

$$t = \\frac{r_A + r_X}{\\sqrt{2}(r_B + r_X)}$$

其中 $r_A$, $r_B$, $r_X$ 是Shannon离子半径。

**结构预测规则:**

| t 范围 | 预测结构 | 说明 |
|--------|----------|------|
| 0.9 ≤ t ≤ 1.0 | 立方 | 理想钙钛矿 |
| 0.8 ≤ t < 0.9 | 正交/四方 | 畸变钙钛矿 |
| 0.71 ≤ t < 0.8 | 菱方 | 严重畸变 |
| t < 0.71 或 t > 1.1 | 非钙钛矿 | 不稳定 |

### 八面体因子

$$\\mu = \\frac{r_B}{r_X}$$

- 稳定条件: μ ≥ 0.442
- 保证[BX₆]八面体不坍塌

### 分解能

钙钛矿相对于竞争相的热力学稳定性:

$$E_{decomp} = E(ABX_3) - \\sum_i x_i E(competing\\_phase_i)$$

- E_decomp < 0: 热力学稳定
- E_decomp > 0.1 eV/atom: 可能分解

### 关键材料

| 材料 | 带隙 (eV) | 效率记录 | 稳定性 |
|------|-----------|----------|--------|
| MAPbI₃ | 1.57 | 25.2% | 一般 |
| FAPbI₃ | 1.48 | 25.7% | 良好 |
| CsPbI₃ | 1.73 | 21.0% | 中等 |
| CsPbBr₃ | 2.30 | 11.0% | 优秀 |
| Cs₂AgBiBr₆ | 2.20 | - | 优秀 (无毒) |

---

## 工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: 生成化学组成                                          │
│  - 组合A位、B位、X位元素                                        │
│  - 生成ABX₃化学式                                               │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: Goldschmidt容忍因子计算                               │
│  - 查询Shannon离子半径                                          │
│  - 计算 t = (r_A + r_X) / [√2 × (r_B + r_X)]                    │
│  - 计算八面体因子 μ = r_B / r_X                                 │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: 结构预测                                              │
│  - 根据t预测结构类型                                            │
│  - 立方 / 正交 / 四方 / 非钙钛矿                                │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 4: DFT分解能计算                                         │
│  - 计算钙钛矿总能量                                             │
│  - 计算竞争相能量                                               │
│  - E_decomp = E(ABX₃) - ΣE(竞争相)                              │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 5: 相变温度预测                                          │
│  - 基于结构类型和组成                                           │
│  - 估算高温/低温相变                                            │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 6: 稳定性分析与排名                                      │
│  - 综合评分 (容忍因子 + 分解能 + 相变温度)                      │
│  - 生成稳定性相图                                               │
│  - 合成建议                                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 文件说明

### 主脚本
```
case_perovskite.py                   # 主程序，包含完整工作流
```

### 配置文件
```
configs/
  perovskite_config.yaml             # 钙钛矿稳定性预测参数
```

### 数据文件
```
data/
  ionic_radii.csv                    # Shannon离子半径数据
  reference_data.yaml                # 实验参考数据
```

### Jupyter Notebook
```
notebooks/
  perovskite_stability.ipynb         # 交互式分析
```

---

## 使用方法

### 方法1: 命令行运行

```bash
cd /root/.openclaw/workspace/dft_lammps_research/applications/perovskite

# 基本运行
python case_perovskite.py

# 指定钙钛矿类型
python case_perovskite.py --type halide   # 卤化物钙钛矿
python case_perovskite.py --type oxide    # 氧化物钙钛矿

# 指定工作目录
python case_perovskite.py --work-dir ./my_results

# 使用自定义配置
python case_perovskite.py --config configs/perovskite_config.yaml
```

### 方法2: Jupyter Notebook交互

```bash
jupyter notebook notebooks/perovskite_stability.ipynb
```

### 方法3: Python API调用

```python
from case_perovskite import PerovskiteStabilityConfig, PerovskiteStabilityAnalyzer

# 创建配置
config = PerovskiteStabilityConfig(
    perovskite_type="halide",
    A_site_elements=["Cs", "MA", "FA"],
    B_site_elements=["Pb", "Sn"],
    X_site_elements=["I", "Br"],
    work_dir="./my_perovskite"
)

# 创建分析器
analyzer = PerovskiteStabilityAnalyzer(config)

# 运行完整工作流
df, results = analyzer.run_full_workflow()

# 查看最佳候选
best = df.iloc[0]
print(f"最佳候选: {best['Formula']}")
print(f"容忍因子: {best['Tolerance Factor']}")
```

---

## 结果解读

### 输出文件

```
perovskite_stability_results/
├── perovskite_stability_results.csv  # 筛选结果表格
├── perovskite_stability_report.txt   # 详细文字报告
├── perovskite_stability_analysis.png # 综合分析图表
├── fig1_stability_map.png            # 容忍因子相图 (高清)
├── fig2_stability_ranking.png        # 稳定性排名图 (高清)
└── *.png                             # 其他分析图表
```

### 结果表格字段

| 字段 | 说明 | 单位 |
|------|------|------|
| Formula | 化学式 | - |
| A-site | A位元素 | - |
| B-site | B位元素 | - |
| X-site | X位元素 | - |
| Tolerance Factor | Goldschmidt容忍因子 | - |
| Octahedral Factor | 八面体因子 | - |
| Predicted Structure | 预测结构 | - |
| Decomposition Energy | 分解能 | eV/atom |
| Phase Transition | 相变温度 | K |
| Overall Stability Score | 综合稳定性评分 | 0-1 |

### 稳定性相图解读

**主图: 容忍因子-八面体因子相图**
- **X轴**: 容忍因子 t
- **Y轴**: 八面体因子 μ
- **绿色区域**: 稳定钙钛矿 (0.8 < t < 1.0, μ > 0.442)
- **红色区域**: 非钙钛矿
- **黄色标记**: 已知钙钛矿位置

### 评价标准

| 等级 | 分解能范围 | 说明 |
|------|------------|------|
| ★★★★★ Excellent | < 0.02 eV/atom | 非常稳定，适合商业化 |
| ★★★★☆ Very Good | 0.02-0.05 | 稳定，适合研究 |
| ★★★☆☆ Good | 0.05-0.1 | 可合成，需注意条件 |
| ★★☆☆☆ Moderate | 0.1-0.2 | 稳定性较差 |
| ★☆☆☆☆ Poor | > 0.2 | 不推荐 |

---

## 计算参数建议

### 容忍因子计算
```yaml
ionic_radii_source: "Shannon_1976"
A_site_CN: 12  # 配位数
B_site_CN: 6
X_site_CN: 6
```

### DFT计算
```yaml
functional: PBE
encut: 520 eV
kpoints: 0.2 Å⁻¹
include_soc: true  # 对重元素重要
```

### 稳定性阈值
```yaml
max_decomposition_energy: 0.1 eV/atom
tolerance_factor_range: [0.8, 1.0]
octahedral_factor_min: 0.442
```

---

## 扩展与改进

### 可能的扩展

1. **混合阳离子/阴离子钙钛矿**
   - (MA/FA/Cs)Pb(I/Br)₃
   - 混合效应提高稳定性

2. **双钙钛矿**
   - Cs₂AgBiBr₆ (无毒)
   - A₂B⁺B³⁺X₆结构

3. **低维钙钛矿**
   - 2D Ruddlesden-Popper
   - 提高湿度稳定性

4. **机器学习预测**
   - 训练形成能预测模型
   - 加速高通量筛选

---

## 参考文献

1. Goldschmidt, V.M. (1926). "Die Gesetze der Krystallochemie." *Naturwissenschaften*, 14(21), 477-485.

2. Kieslich, G., et al. (2014). "Can the tolerance factor be used to predict new perovskite-based photovoltaic absorbers?" *Journal of Materials Chemistry A*, 2(36), 14996-15000.

3. Li, Z., et al. (2021). "High-throughput screening of perovskite materials using machine learning." *Advanced Materials*, 33(15), 2005033.

4. Filip, M.R., et al. (2016). "Computational screening of all-stoichiometric inorganic materials using the Goldschmidt tolerance factor." *Journal of Physical Chemistry C*, 120(30), 16606-16613.

5. Bartel, C.J., et al. (2019). "A critical examination of compound stability predictions from machine-learned formation energies." *npj Computational Materials*, 5(1), 1-11.

---

## 联系与支持

如有问题或建议，请参考主项目文档：
- 主README: `/root/.openclaw/workspace/dft_lammps_research/README.md`
- 集成指南: `/root/.openclaw/workspace/dft_lammps_research/integration_guide.md`

---

*Last Updated: 2026-03-09*
