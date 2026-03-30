# 应用案例总览
# Application Cases Overview

本目录包含三个完整的材料计算应用案例，展示DFT+ML+MD集成工作流在不同材料研究领域的应用。

---

## 案例列表

### 1. 固态电解质筛选案例 (`solid_electrolyte/`)

**目标**: 高通量筛选高Li离子电导率的硫化物固态电解质

**核心功能**:
- Materials Project数据获取
- DFT结构优化与能量计算
- 机器学习势训练
- 多温度MD模拟与扩散分析
- 离子电导率与活化能预测
- 候选材料排名与可视化

**输出**:
- 电导率-活化能散点图
- Arrhenius分析图
- 候选材料排行榜
- 发表质量分析图表

**适用场景**:
- 全固态锂电池电解质筛选
- 离子导体材料设计
- 扩散机制研究

---

### 2. 电催化剂设计案例 (`catalyst/`)

**目标**: ORR/OER双功能电催化剂设计与活性预测

**核心功能**:
- 金属表面模型构建
- DFT吸附能计算
- Scaling relation分析
- 火山图绘制
- 过电位计算与预测
- 双功能活性评估

**输出**:
- ORR/OER火山图
- 双功能活性图
- 催化剂推荐列表
- 吸附能关系分析

**适用场景**:
- 燃料电池催化剂设计
- 电解水催化剂优化
- 金属-空气电池研究

---

### 3. 钙钛矿稳定性案例 (`perovskite/`)

**目标**: 预测钙钛矿太阳能电池的稳定性与可合成性

**核心功能**:
- Goldschmidt容忍因子计算
- 八面体因子分析
- DFT分解能计算
- 相变温度预测
- 稳定性相图绘制
- 可合成性评分

**输出**:
- 容忍因子-八面体因子相图
- 稳定性热图
- 分解能分布分析
- 元素组合稳定性矩阵

**适用场景**:
- 钙钛矿太阳能电池材料筛选
- 新型钙钛矿发现
- 稳定性改进策略设计

---

## 快速开始

### 运行所有案例

```bash
cd /root/.openclaw/workspace/dft_lammps_research/applications

# 固态电解质筛选
python solid_electrolyte/case_solid_electrolyte.py --work-dir ./results/solid_electrolyte

# 电催化剂设计
python catalyst/case_catalyst.py --work-dir ./results/catalyst

# 钙钛矿稳定性预测
python perovskite/case_perovskite.py --work-dir ./results/perovskite
```

### 使用Jupyter Notebook

```bash
# 启动Jupyter
jupyter notebook

# 然后在浏览器中打开:
# - solid_electrolyte/notebooks/solid_electrolyte_analysis.ipynb
# - catalyst/notebooks/catalyst_design.ipynb
# - perovskite/notebooks/perovskite_stability.ipynb
```

---

## 目录结构

```
applications/
├── solid_electrolyte/              # 固态电解质筛选案例
│   ├── case_solid_electrolyte.py   # 主程序
│   ├── configs/
│   │   └── screening_config.yaml   # 配置文件
│   ├── data/
│   │   ├── demo_candidates.csv     # 示例数据
│   │   └── reference_data.yaml     # 参考数据
│   ├── notebooks/
│   │   └── solid_electrolyte_analysis.ipynb  # Jupyter Notebook
│   └── README.md                   # 详细说明
│
├── catalyst/                       # 电催化剂设计案例
│   ├── case_catalyst.py            # 主程序
│   ├── configs/
│   │   └── catalyst_config.yaml    # 配置文件
│   ├── data/
│   │   ├── metal_properties.csv    # 金属性质数据
│   │   └── reference_data.yaml     # 参考数据
│   ├── notebooks/
│   │   └── catalyst_design.ipynb   # Jupyter Notebook
│   └── README.md                   # 详细说明
│
└── perovskite/                     # 钙钛矿稳定性案例
    ├── case_perovskite.py          # 主程序
    ├── configs/
    │   └── perovskite_config.yaml  # 配置文件
    ├── data/
    │   ├── ionic_radii.csv         # 离子半径数据
    │   └── reference_data.yaml     # 参考数据
    ├── notebooks/
    │   └── perovskite_stability.ipynb  # Jupyter Notebook
    └── README.md                   # 详细说明
```

---

## 依赖要求

### Python包
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install pymatgen ase
pip install mp-api  # Materials Project API (可选)
```

### DFT软件 (可选，用于实际计算)
- VASP
- Quantum ESPRESSO
- ABACUS

### MD软件 (可选，用于实际计算)
- LAMMPS
- GROMACS

### ML势训练 (可选，用于实际计算)
- DeepMD-kit
- NEP (GPUMD)
- MACE

---

## 案例对比

| 特性 | 固态电解质 | 电催化剂 | 钙钛矿稳定性 |
|------|-----------|----------|--------------|
| **核心计算** | MD扩散分析 | DFT吸附能 | 容忍因子 |
| **时间尺度** | 皮秒-纳秒 | 静态计算 | 静态计算 |
| **关键输出** | 电导率 | 过电位 | 稳定性评分 |
| **应用场景** | 电池材料 | 催化材料 | 光伏材料 |
| **计算成本** | 高 | 中 | 低 |

---

## 扩展开发

### 添加新案例的步骤

1. **创建案例目录**
```bash
mkdir applications/my_case/{configs,data,notebooks}
```

2. **编写主程序** (`case_my_case.py`)
- 导入现有框架模块
- 实现核心分析类
- 提供命令行接口

3. **创建配置文件** (`configs/config.yaml`)
- 定义计算参数
- 设置筛选标准

4. **准备示例数据** (`data/`)
- 参考实验数据
- 演示数据集

5. **编写Jupyter Notebook** (`notebooks/`)
- 交互式分析流程
- 可视化展示

6. **编写README** (`README.md`)
- 科学背景介绍
- 使用说明
- 结果解读

---

## 常见问题

### Q: 案例运行需要真实的DFT计算吗？
**A**: 不需要。所有案例都包含模拟模式，可以快速演示完整流程。要进行实际计算，请使用 `--run-dft` 或 `--run-ml` 参数。

### Q: 如何修改筛选条件？
**A**: 编辑对应案例的 `configs/*.yaml` 配置文件，调整参数后重新运行。

### Q: 结果图表在哪里？
**A**: 所有输出文件保存在 `--work-dir` 指定的目录下，包括:
- `*.csv` - 数据表格
- `*.png` - 分析图表
- `*.txt` - 详细报告

### Q: 可以使用自己的数据吗？
**A**: 可以。在 `data/` 目录下替换示例数据文件，或在代码中直接传入自定义数据。

---

## 技术参考

### 核心框架模块
```
dft_lammps_research/
├── integrated_materials_workflow.py   # 集成工作流
├── dft_to_lammps_bridge.py            # DFT/LAMMPS桥接
├── nep_training_pipeline.py           # NEP训练
├── battery_screening_pipeline.py      # 电池筛选
└── ...
```

### 数据流
```
输入配置 → 结构获取 → DFT计算 → ML训练 → MD模拟 → 分析 → 可视化
```

---

## 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| 1.0.0 | 2026-03-09 | 初始发布，三个完整案例 |

---

## 联系我们

如有问题或建议，请通过以下方式联系：
- 查看主项目README
- 参考integration_guide.md
- 提交Issue (如适用)

---

*Last Updated: 2026-03-09*
