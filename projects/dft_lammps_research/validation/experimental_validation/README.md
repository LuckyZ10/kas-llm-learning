# 实验数据对比与验证模块 (Experimental Validation Module)

## 概述

本模块提供了一套完整的工具，用于将计算模拟结果（如DFT、分子动力学）与实验测量数据进行对比验证。支持多种实验数据类型和格式，提供统计分析和可视化功能。

## 功能特性

### 1. 实验数据连接器 (Connectors)

#### XRD连接器
- 支持格式：CSV, TXT, XRDML, RAW, CIF, HDF5
- 功能：峰查找、晶面指数标定、背景扣除
- 数据库接口：ICSD, Materials Project

#### 电化学连接器
- 支持格式：CSV, MPR/MPT (Bio-Logic), NDT (NEWARE), HDF5
- 测试类型：恒流充放电(GCD)、循环伏安(CV)、电化学阻抗(EIS)
- 功能：容量计算、库伦效率、循环数据提取

#### 光谱连接器
- 支持类型：XPS, Raman, FTIR, UV-Vis, NMR, XAS/XANES
- 支持格式：CSV, VMS, SPC, DPT, JCAMP-DX
- 功能：基线校正、峰查找、波长校准

#### TEM连接器
- 支持格式：DM3/DM4, TIFF, SER/EMI, MRC, HDF5
- 数据类型：图像、衍射图样、EDS/EELS谱
- 功能：FFT分析、线扫描剖面

#### 数据库连接器
- Materials Project API
- ICSD (需要订阅)
- COD (Crystallography Open Database)

### 2. 对比分析工具 (Analyzers)

#### 结构分析器
- XRD图谱比较（Pearson相关系数、Rwp、GoF）
- 晶体结构匹配（RMS距离、晶格参数比较）
- 差分图谱生成

#### 性能分析器
- 充放电曲线对比
- 循环稳定性分析
- 倍率性能对比
- CV曲线峰匹配

#### 统计分析器
- 误差指标：MAE, RMSE, R², MAPE
- 假设检验：Kolmogorov-Smirnov, 卡方检验
- Bland-Altman一致性分析
- Bootstrap置信区间
- 异常值检测

#### 可视化器
- 对比图表（XRD、GCD、CV）
- 散点图和残差图
- Bland-Altman图
- 差异热图
- 循环稳定性图

### 3. 验证工作流 (Workflows)

#### 单系统验证
```python
from experimental_validation import ValidationWorkflow, ValidationConfig

config = ValidationConfig(
    data_type='xrd',
    output_dir='./results',
    generate_report=True,
    rmse_threshold=0.1,
    r2_threshold=0.9,
)

workflow = ValidationWorkflow(config)
workflow.load_experimental_data('exp_xrd.csv')
workflow.load_computational_data('sim_xrd.csv')
result = workflow.run_validation()
workflow.generate_report('report.html')
```

#### 批量验证
```python
from experimental_validation import BatchValidator, BatchValidationConfig

config = BatchValidationConfig(
    exp_data_dir='./experimental',
    sim_data_dir='./simulated',
    data_type='xrd',
    n_workers=4,
)

validator = BatchValidator(config)
results = validator.run_batch()
validator.generate_summary_plots()
```

### 4. 不确定性量化 (Uncertainty)

#### 误差传播
- 线性误差传播
- 蒙特卡洛误差传播
- 相关变量处理

#### 置信区间估计
- 均值置信区间（t分布）
- 比例置信区间（Wilson、Agresti-Coull）
- Bootstrap置信区间
- 预测区间
- 多重比较校正

#### 敏感性分析
- 局部敏感性（导数）
- Sobol全局敏感性分析
- Morris筛选法
- 龙卷风图
- 相关性分析

## 安装

### 依赖包

```bash
pip install numpy scipy pandas matplotlib
```

### 可选依赖

```bash
# 数据库和结构处理
pip install pymatgen

# HDF5支持
pip install h5py

# TEM数据处理
pip install ncempy mrcfile pillow

# 高级统计分析
pip install SALib

# YAML配置支持
pip install pyyaml
```

## 快速开始

### 示例1：XRD数据对比

```python
from experimental_validation import XRDConnector, XRDComparator

# 加载数据
connector = XRDConnector()
exp_data = connector.read('experimental.xrdml')
sim_data = connector.read('simulated.csv')

# 对比
comparator = XRDComparator()
results = comparator.compare(exp_data, sim_data)

print(f"R² = {results['r2']:.3f}")
print(f"RMSE = {results['rmse']:.4f}")
```

### 示例2：电化学数据对比

```python
from experimental_validation import ElectrochemicalConnector, ElectrochemicalComparator

# 加载充放电数据
connector = ElectrochemicalConnector()
exp_data = connector.read('experimental.mpr', test_type='gcd')
sim_data = connector.read('simulated.csv', test_type='gcd')

# 对比
comparator = ElectrochemicalComparator()
results = comparator.compare_gcd_curves(exp_data, sim_data)

print(f"Capacity error: {results['capacity_error_percent']:.2f}%")
```

### 示例3：统计分析

```python
from experimental_validation import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()

# 综合统计分析
results = analyzer.comprehensive_analysis(experimental, predicted)

# 误差指标
print(f"MAE: {results['errors']['mae']:.4f}")
print(f"R²: {results['errors']['r2']:.4f}")

# 异常值
print(f"Found {len(results['outliers'])} outliers")
```

### 示例4：不确定性量化

```python
from experimental_validation import ErrorPropagator, SensitivityAnalyzer

# 误差传播
propagator = ErrorPropagator()
results = propagator.monte_carlo_propagation(
    func=my_function,
    values=np.array([1.0, 2.0]),
    uncertainties=np.array([0.1, 0.2]),
)
print(f"Mean: {results['mean']:.3f} ± {results['std']:.3f}")

# 敏感性分析
analyzer = SensitivityAnalyzer()
sens = analyzer.local_sensitivity(my_function, values, ['param1', 'param2'])
print(f"Most sensitive: {sens['most_sensitive']}")
```

## 项目结构

```
experimental_validation/
├── __init__.py              # 模块入口
├── connectors/              # 数据连接器
│   ├── base_connector.py    # 基础连接器类
│   ├── xrd_connector.py     # XRD连接器
│   ├── electrochemical_connector.py
│   ├── spectroscopy_connector.py
│   ├── tem_connector.py
│   └── database_connector.py
├── analyzers/               # 分析工具
│   ├── structure_analyzer.py
│   ├── performance_analyzer.py
│   ├── statistical_analyzer.py
│   └── visualizer.py
├── workflows/               # 工作流
│   ├── validation_workflow.py
│   └── batch_validator.py
├── uncertainty/             # 不确定性量化
│   └── error_propagation.py
├── utils/                   # 工具函数
│   └── helpers.py
└── examples/                # 示例
    ├── xrd_validation_example.py
    ├── electrochemical_validation_example.py
    └── uncertainty_quantification_example.py
```

## 配置说明

### 验证工作流配置

```python
ValidationConfig(
    data_type='xrd',              # 数据类型
    output_dir='./results',       # 输出目录
    generate_report=True,         # 生成报告
    generate_plots=True,          # 生成图表
    report_format='html',         # 报告格式
    rmse_threshold=0.1,           # RMSE阈值
    r2_threshold=0.9,             # R²阈值
    confidence_level=0.95,        # 置信水平
)
```

### 批量验证配置

```python
BatchValidationConfig(
    exp_data_dir='./exp',         # 实验数据目录
    sim_data_dir='./sim',         # 模拟数据目录
    file_pattern='*.csv',         # 文件匹配模式
    n_workers=4,                  # 并行工作数
    individual_reports=True,      # 生成单个报告
    summary_report=True,          # 生成汇总报告
)
```

## API密钥配置

### Materials Project

设置环境变量：
```bash
export MP_API_KEY="your_api_key_here"
```

或在代码中配置：
```python
from experimental_validation import MaterialsProjectConnector

connector = MaterialsProjectConnector({
    'api_key': 'your_api_key_here'
})
```

### ICSD

```bash
export ICSD_USERNAME="your_username"
export ICSD_PASSWORD="your_password"
```

## 输出报告

### JSON报告
包含所有比较指标、统计分析和元数据

### HTML报告
交互式报告，包含图表和表格

### 文本报告
简洁的文本格式摘要

## 常见问题

### Q: 如何添加对新数据格式的支持？

A: 继承`BaseConnector`类并实现`read()`和`validate_format()`方法：

```python
from experimental_validation.connectors import BaseConnector, ExperimentalData

class MyConnector(BaseConnector):
    def read(self, filepath: str, **kwargs) -> ExperimentalData:
        # 实现数据读取逻辑
        pass
    
    def validate_format(self, filepath: str) -> bool:
        # 实现格式验证逻辑
        pass
```

### Q: 如何处理大型数据集？

A: 使用批量验证功能和HDF5格式：

```python
config = BatchValidationConfig(
    n_workers=8,  # 并行处理
)

# 数据保存为HDF5格式以提高性能
exp_data.save('data.h5', format='h5')
```

### Q: 如何自定义可视化？

A: 使用可视化器类或直接使用matplotlib：

```python
from experimental_validation import ValidationVisualizer

visualizer = ValidationVisualizer(config={
    'figure_size': (12, 8),
    'dpi': 200,
})

# 自定义图表
fig = visualizer.plot_xrd_comparison(exp_data, sim_data)
# 进一步自定义
fig.suptitle('My Custom Title')
fig.savefig('output.png')
```

## 参考资料

- [pymatgen文档](https://pymatgen.org/)
- [Materials Project API](https://api.materialsproject.org/)
- [ICSD数据库](https://icsd.fiz-karlsruhe.de/)
- [SALib敏感性分析](https://salib.readthedocs.io/)

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交issue或联系开发团队。
