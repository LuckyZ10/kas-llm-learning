# DFT-LAMMPS 实验数据对接与验证模块

Phase 65: 实验数据对接与验证 - 计算结果与实验数据对比验证

## 功能概述

本模块提供完整的实验数据验证工作流，支持：

- 🔬 **实验数据导入**: CIF、POSCAR、JSON等多种格式
- 🌐 **数据库连接**: Materials Project、AFLOW、COD等高通量数据库
- 📊 **计算-实验对比**: 误差分析、不确定性量化
- 📈 **可视化报告**: Parity图、误差分布、统计报表
- 🔄 **反馈优化**: 基于对比结果自动优化计算参数

## 目录结构

```
experimental_validation/
├── __init__.py              # 模块入口
├── data_formats.py          # 数据格式处理 (~600行)
├── data_sources.py          # 数据源连接器 (~600行)
├── importers.py             # 数据导入器 (~500行)
├── comparison.py            # 对比分析 (~550行)
├── visualization.py         # 可视化 (~600行)
├── feedback.py              # 反馈优化 (~550行)
├── examples.py              # 使用示例 (~300行)
└── tests/
    └── test_validation.py   # 单元测试 (~350行)
```

**总计: ~3600行代码**

## 快速开始

### 1. 基础使用

```python
from dftlammps.experimental_validation import (
    read_structure, ExperimentalProperty, ExperimentalDataset,
    compare_property, validate_properties
)

# 读取实验结构
structure = read_structure('experiment.cif')

# 创建实验数据集
exp_dataset = ExperimentalDataset(
    structure=structure,
    properties=[
        ExperimentalProperty('band_gap', 3.2, 'eV', uncertainty=0.1)
    ]
)

# 对比计算值和实验值
result = compare_property(
    computed=3.5,
    experimental=3.2,
    uncertainty=0.1,
    name='band_gap'
)
print(f"误差: {result.percentage_error:.2f}%")
```

### 2. 批量验证

```python
import numpy as np

# 模拟数据
computed = [3.1, 3.2, 3.0, 3.3, 3.15]
experimental = [3.0, 3.15, 2.95, 3.25, 3.1]
uncertainties = [0.1, 0.1, 0.1, 0.1, 0.1]

# 生成验证报告
report = validate_properties(
    computed, experimental, 'band_gap', uncertainties
)

print(f"MAE: {report.statistics.mae:.4f}")
print(f"MAPE: {report.statistics.mape:.2f}%")
print(f"R²: {report.statistics.r2:.4f}")
print(f"通过验证: {report.is_validated}")
```

### 3. 数据库查询

```python
from dftlammps.experimental_validation import (
    create_mp_connector, SearchQuery
)

# 连接Materials Project
mp = create_mp_connector(api_key='your_api_key')

# 搜索
query = SearchQuery(
    elements=['Li', 'Fe', 'P', 'O'],
    has_structure=True
)

results = mp.search(query, limit=10)
for result in results:
    print(f"{result.entry_id}: {result.formula}")
```

### 4. 生成报告

```python
from dftlammps.experimental_validation import generate_full_report

# 生成HTML和Markdown报告
output_paths = generate_full_report(
    reports=[report1, report2],
    output_dir='./validation_reports',
    formats=['html', 'md']
)
```

### 5. 反馈优化

```python
from dftlammps.experimental_validation import create_feedback_loop

# 创建反馈循环
loop = create_feedback_loop()

# 当前参数
current_params = {'ENCUT': 400, 'KPOINTS': [4, 4, 4]}

# 运行优化循环
cycle = loop.run_cycle(report, current_params, 'dft')

# 获取建议
for rec in cycle.recommendations:
    print(f"目标: {rec.target.value}")
    for adj in rec.adjustments:
        print(f"  {adj.parameter}: {adj.current_value} -> {adj.suggested_value}")
```

## 支持的格式

### 结构格式
- **CIF**: 晶体学信息文件
- **POSCAR/VASP**: VASP输入格式
- **JSON**: 自定义JSON格式

### 数据源
- **Materials Project**: 高通量计算数据库
- **AFLOW**: 自动材料发现库
- **COD**: 开放晶体学数据库
- **本地文件**: CIF、CSV、Excel等

## 误差指标

| 指标 | 说明 | 阈值 |
|------|------|------|
| MAE | 平均绝对误差 | - |
| RMSE | 均方根误差 | - |
| MAPE | 平均绝对百分比误差 | <10%良好 |
| R² | 决定系数 | >0.7可接受 |
| Z-score | 与实验不确定度的偏差 | <2σ内 |

## 一致性等级

| 等级 | 误差范围 | 说明 |
|------|----------|------|
| Excellent | <1% | 优秀 |
| Good | 1-5% | 良好 |
| Acceptable | 5-10% | 可接受 |
| Poor | 10-20% | 较差 |
| Unacceptable | >20% | 不可接受 |

## 运行演示

```python
# 运行所有演示
from dftlammps.experimental_validation import run_all_demos
run_all_demos()

# 运行示例工作流
from dftlammps.experimental_validation import create_example_workflow
create_example_workflow()

# 运行测试
from dftlammps.experimental_validation.tests.test_validation import run_tests
run_tests()
```

## 运行示例脚本

```bash
cd /root/.openclaw/workspace
python -m dftlammps.experimental_validation.examples
```

## API 参考

### 核心类

- `CrystalStructure`: 晶体结构数据
- `ExperimentalProperty`: 实验属性
- `ExperimentalDataset`: 实验数据集
- `ComparisonResult`: 对比结果
- `ValidationReport`: 验证报告
- `FeedbackLoop`: 反馈优化循环

### 便捷函数

- `read_structure()`: 读取结构文件
- `compare_property()`: 对比单个属性
- `validate_properties()`: 批量验证
- `create_feedback_loop()`: 创建反馈循环
- `generate_full_report()`: 生成完整报告

## 配置

```python
from dftlammps.experimental_validation import ImportConfig, DatabaseConfig

# 导入配置
import_config = ImportConfig(
    validate_on_import=True,
    auto_convert_units=True
)

# 数据库配置
db_config = DatabaseConfig(
    api_key='your_api_key',
    cache_enabled=True,
    cache_ttl=86400
)
```

## 依赖

- numpy
- scipy
- requests
- matplotlib (可选，用于可视化)

## 作者

DFT-LAMMPS Team

## 版本

v1.0.0 - Phase 65
