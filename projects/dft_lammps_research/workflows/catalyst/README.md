# 电催化剂设计案例
# Electrocatalyst Design Case Study

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

本案例研究演示如何设计ORR/OER双功能电催化剂，用于燃料电池和电解水应用。

### 目标
- **ORR目标**: 过电位 η < 0.40 V
- **OER目标**: 过电位 η < 0.40 V
- **双功能目标**: 总过电位 η_total < 0.80 V

### 应用意义
电催化剂是燃料电池、金属-空气电池和电解水的核心材料，直接影响能源转换效率和成本。

---

## 科学背景

### 氧还原反应 (ORR)

**反应方程式:**
```
酸性: O₂ + 4H⁺ + 4e⁻ → 2H₂O     E° = 1.23 V vs RHE
碱性: O₂ + 2H₂O + 4e⁻ → 4OH⁻    E° = 1.23 V vs RHE
```

**反应机理:**
1. O₂ + * → O₂* (吸附)
2. O₂* + H⁺ + e⁻ → OOH* (质子-电子转移)
3. OOH* + H⁺ + e⁻ → O* + H₂O
4. O* + H⁺ + e⁻ → OH*
5. OH* + H⁺ + e⁻ → H₂O + * (脱附)

### 氧析出反应 (OER)

**反应方程式:**
```
酸性: 2H₂O → O₂ + 4H⁺ + 4e⁻     E° = 1.23 V vs RHE
碱性: 4OH⁻ → O₂ + 2H₂O + 4e⁻    E° = 1.23 V vs RHE
```

### Scaling Relations与火山曲线

催化活性与中间体吸附能之间存在"火山型"关系:
- 吸附太弱: 反应物难以活化 (高过电位)
- 吸附太强: 产物难以脱附 (高过电位)
- 最优值: 火山顶点

**关键Scaling Relations:**
```
ΔG_OOH = ΔG_OH + 3.2 ± 0.2 eV
```

### d带中心理论

Hammer-Nørskov d带中心模型:
- d带中心上移 → 吸附增强
- d带中心下移 → 吸附减弱
- 最佳催化活性: ε_d ≈ -2.0 eV

---

## 工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: 构建金属表面模型                                      │
│  - 创建(111), (100), (110)表面                                  │
│  - 4-6层金属原子 + 15Å真空层                                    │
│  - 固定底层2-3层原子                                            │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: DFT计算吸附能                                         │
│  - O*, OH*, OOH* 吸附构型优化                                   │
│  - 计算吸附自由能 (含ZPE和熵修正)                               │
│  - 参考态: H₂O, H₂, O₂                                          │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: Scaling Relation分析                                  │
│  - 验证ΔG_OOH vs ΔG_OH 线性关系                                 │
│  - 分析吸附能之间的关联性                                       │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 4: 过电位计算                                            │
│  - 计算各反应步自由能变化                                       │
│  - 确定决速步 (RDS)                                             │
│  - 计算过电位 η = |U - U₀|                                      │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 5: 火山图绘制与活性预测                                  │
│  - ORR火山图 (descriptor: ΔG_OH)                                │
│  - OER火山图 (descriptor: ΔG_O - ΔG_OH)                         │
│  - 双功能活性图                                                 │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 6: 催化剂推荐                                            │
│  - 识别ORR最优催化剂                                            │
│  - 识别OER最优催化剂                                            │
│  - 识别双功能催化剂                                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 文件说明

### 主脚本
```
case_catalyst.py                   # 主程序，包含完整工作流
```

### 配置文件
```
configs/
  catalyst_config.yaml             # 催化剂设计参数配置
```

### 数据文件
```
data/
  metal_properties.csv             # 金属物理化学性质
  reference_data.yaml              # 实验参考数据
```

### Jupyter Notebook
```
notebooks/
  catalyst_design.ipynb            # 交互式分析
```

---

## 使用方法

### 方法1: 命令行运行

```bash
cd /root/.openclaw/workspace/dft_lammps_research/applications/catalyst

# 基本运行
python case_catalyst.py

# 指定反应类型
python case_catalyst.py --reaction orr    # 仅ORR
python case_catalyst.py --reaction oer    # 仅OER
python case_catalyst.py --reaction both   # 双功能

# 指定工作目录
python case_catalyst.py --work-dir ./my_results

# 使用自定义配置
python case_catalyst.py --config configs/catalyst_config.yaml
```

### 方法2: Jupyter Notebook交互

```bash
jupyter notebook notebooks/catalyst_design.ipynb
```

### 方法3: Python API调用

```python
from case_catalyst import CatalystDesignConfig, ElectrocatalystDesigner

# 创建配置
config = CatalystDesignConfig(
    reaction="both",
    metals=["Pt", "Pd", "Au", "Ni", "Co", "Ir", "Ru"],
    surface_facets=["111", "100"],
    work_dir="./my_catalyst"
)

# 创建设计器
designer = ElectrocatalystDesigner(config)

# 运行完整工作流
df, results = designer.run_full_workflow()

# 查看最佳催化剂
best = df.loc[df['Total_η (V)'].idxmin()]
print(f"最佳双功能催化剂: {best['Surface']}")
```

---

## 结果解读

### 输出文件

```
catalyst_design_results/
├── catalyst_screening_results.csv  # 筛选结果表格
├── catalyst_design_report.txt      # 详细文字报告
├── volcano_analysis.png            # 火山图综合
├── fig1_ORR_volcano.png            # ORR火山图 (高清)
├── fig2_bifunctional_map.png       # 双功能活性图 (高清)
└── surfaces/                       # 表面结构文件
    ├── Pt_111.xyz
    ├── Pt_100.xyz
    └── ...
```

### 结果表格字段

| 字段 | 说明 | 单位 |
|------|------|------|
| Surface | 表面标识 | - |
| Metal | 金属元素 | - |
| Facet | 晶面指数 | - |
| dG_O* | O*吸附自由能 | eV |
| dG_OH* | OH*吸附自由能 | eV |
| dG_OOH* | OOH*吸附自由能 | eV |
| ORR_η | ORR过电位 | V |
| OER_η | OER过电位 | V |
| Total_η | 总过电位 | V |

### 火山图解读

**ORR火山图:**
- **X轴**: ΔG_OH (描述符)
- **Y轴**: ORR过电位
- **火山顶点**: ΔG_OH ≈ 0.0 eV (理论最优)
- **Pt(111)**: 位于火山顶点附近，实验验证的最佳催化剂

**OER火山图:**
- **X轴**: ΔG_O - ΔG_OH (描述符)
- **Y轴**: OER过电位
- **火山顶点**: ΔG_O - ΔG_OH ≈ 1.6 eV
- **RuO₂, IrO₂**: 位于火山顶点附近

### 推荐标准

| 催化剂类型 | 目标过电位 | 推荐材料 |
|------------|------------|----------|
| ORR催化剂 | η < 0.35 V | Pt(111), Pt₃Ni(111) |
| OER催化剂 | η < 0.35 V | RuO₂, IrO₂, Co₃O₄ |
| 双功能催化剂 | η_total < 0.80 V | Pt-Ir合金, 钙钛矿氧化物 |

---

## 计算参数建议

### DFT计算
```yaml
functional: PBE
encut: 500 eV
kpoints: [6, 6, 1]  # 表面计算
dipole_correction: true
# 吸附能收敛标准: < 0.01 eV
```

### 自由能修正
```yaml
# 零点能 (ZPE)
include_zpe: true

# 熵修正 (@ 298K)
include_entropy: true

# 溶剂化修正
solvation_correction:
  OH: 0.35 eV
  O: 0.05 eV
  OOH: 0.40 eV
```

---

## 扩展与改进

### 可能的扩展

1. **合金催化剂**
   - Pt-M (M = Ni, Co, Fe, Cu) 合金
   - 表面偏析效应
   - 配体效应与应变效应

2. **非贵金属催化剂**
   - Fe-N-C 催化剂
   - 单原子催化剂 (SACs)
   - 过渡金属氧化物

3. **进阶计算方法**
   - 显式溶剂化模型
   - 动力学蒙特卡罗 (kMC)
   - 机器学习势加速

---

## 参考文献

1. Nørskov, J.K., et al. (2004). "Origin of the overpotential for oxygen reduction at a fuel-cell cathode." *Journal of Physical Chemistry B*, 108(46), 17886-17892.

2. Stamenkovic, V.R., et al. (2007). "Improved oxygen reduction activity on Pt₃Ni(111) via increased surface site availability." *Science*, 315(5811), 493-497.

3. Man, I.C., et al. (2011). "Universality in oxygen evolution electrocatalysis on oxide surfaces." *ChemCatChem*, 3(7), 1159-1165.

4. Suntivich, J., et al. (2011). "A perovskite oxide optimized for oxygen evolution catalysis from molecular orbital principles." *Science*, 334(6061), 1383-1385.

5. Kulkarni, A., et al. (2018). "Cation-dependent design principles for electrochemical synthesis of ammonia using single-atom catalysts." *Catalysis Science & Technology*, 8(1), 114-122.

---

## 联系与支持

如有问题或建议，请参考主项目文档：
- 主README: `/root/.openclaw/workspace/dft_lammps_research/README.md`
- 集成指南: `/root/.openclaw/workspace/dft_lammps_research/integration_guide.md`

---

*Last Updated: 2026-03-09*
