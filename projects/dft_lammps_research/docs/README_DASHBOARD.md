# DFT+LAMMPS Workflow Monitoring Dashboard

一个功能完整的交互式Web仪表盘，用于实时监控和可视化DFT+LAMMPS计算工作流的各个方面。

## 功能特点

### 📊 实时监控视图

1. **ML训练监控**
   - 实时损失曲线（支持线性/对数刻度）
   - 能量、力、维里RMSE演化
   - 学习率调度可视化
   - 模型测试结果对比
   - KPI卡片显示当前训练状态

2. **MD轨迹可视化**
   - 3D原子结构可视化（支持帧动画）
   - 温度、能量、压强随时间变化
   - 扩散系数分析（MSD计算）
   - 轨迹文件选择和播放控制

3. **高通量筛选结果**
   - 散点图分析（可自定义X/Y轴）
   - 动态筛选和过滤
   - 排名表格（支持排序和分页）
   - 属性分布直方图
   - 相关性矩阵热图

4. **主动学习进度**
   - 迭代进度追踪
   - 模型偏差分布
   - 误差收敛曲线
   - 候选结构统计
   - 收敛状态指示

### 🔄 自动更新
- 可配置的自动刷新间隔
- 文件变更监控（可选）
- 数据缓存优化

### 📤 导出功能
- PDF报告生成（含图表和表格）
- CSV数据导出
- 图表图片导出

## 安装要求

### 必需依赖
```bash
pip install dash dash-bootstrap-components plotly pandas numpy pyyaml reportlab
```

### 可选依赖
```bash
# 3D结构可视化
pip install ase

# 文件监控
pip install watchdog

# 统计分析
pip install scipy
```

### 完整安装
```bash
pip install -r requirements_dashboard.txt
```

## 快速开始

### 1. 基本启动
```bash
python monitoring_dashboard.py
```
仪表盘将在 http://localhost:8050 启动

### 2. 使用配置文件
```bash
python monitoring_dashboard.py --config dashboard_config.yaml
```

### 3. 自定义端口和主机
```bash
python monitoring_dashboard.py --host 0.0.0.0 --port 8080
```

### 4. 调试模式
```bash
python monitoring_dashboard.py --debug
```

## 配置文件说明

`dashboard_config.yaml` 包含以下主要配置项：

### 数据路径
```yaml
work_dir: "./battery_screening"
dft_results_path: "./battery_screening/dft_results"
md_results_path: "./battery_screening/md_results"
models_path: "./battery_screening/models"
al_workflow_path: "./active_learning_workflow"
screening_db_path: "./screening_db"
```

### 刷新设置
```yaml
auto_refresh: true
refresh_interval: 5  # 秒
max_points: 10000    # 最大显示点数
time_window: 3600    # 时间窗口（秒）
```

### 阈值设置
```yaml
force_error_threshold: 0.05    # eV/Å
energy_error_threshold: 0.001  # eV/atom
```

### 服务器设置
```yaml
host: "0.0.0.0"
port: 8050
debug: false
```

## 数据格式要求

### ML训练数据
仪表盘自动读取DeePMD的 `lcurve.out` 文件：
```
# batch  lr       loss     energy_rmse  energy_rmse_traj  force_rmse  force_rmse_traj  virial_rmse  virial_rmse_traj
0        1.0e-3   2.5e+2   0.1          0.1               1.5         1.5              0.05         0.05
100      9.8e-4   2.3e+2   0.09         0.095             1.4         1.45             0.048        0.049
...
```

### MD模拟数据
从LAMMPS的 `log.lammps` 文件读取：
```
Step Temp PotEng KinEng TotEng Press Volume
0    300  -5000  100    -4900  1.0    1000
100  305  -4998  102    -4896  1.2    1002
...
```

轨迹文件支持 `.lammpstrj` 格式（需安装ASE）

### 高通量筛选数据
支持CSV格式或JSON格式：
```csv
structure_id,formula,ionic_conductivity,formation_energy,band_gap
mp-1234,Li2O,1.5e-4,-2.5,4.2
mp-5678,Li3PS4,2.3e-3,-3.1,3.8
```

### 主动学习数据
从工作目录的迭代子目录读取：
```
active_learning_workflow/
├── iter_000/
│   ├── candidates.traj
│   ├── exploration_stats.json
│   └── model_deviations.json
├── iter_001/
│   └── ...
```

## 使用指南

### ML训练监控
1. 进入"ML Training"标签页
2. 查看实时更新的KPI卡片
3. 切换损失曲线的线性/对数刻度
4. 观察RMSE是否低于阈值线

### MD轨迹可视化
1. 进入"MD Simulation"标签页
2. 从下拉菜单选择轨迹文件
3. 使用滑块浏览不同帧
4. 点击播放按钮查看动画
5. 选择特定原子类型进行MSD分析

### 高通量筛选
1. 进入"High-Throughput Screening"标签页
2. 选择X轴和Y轴属性
3. 设置筛选条件并应用
4. 查看排名表格中的候选材料
5. 分析属性分布和相关性

### 主动学习监控
1. 进入"Active Learning"标签页
2. 查看当前迭代和候选数量
3. 观察模型偏差分布的收敛趋势
4. 检查误差是否低于目标阈值

### 导出报告
1. 滚动到页面底部的导出区域
2. 点击"Export PDF Report"生成完整报告
3. 点击"Export CSV Data"导出筛选数据
4. 点击"Export Charts"保存图表图片

## 高级功能

### 自定义列映射
在配置文件中修改 `column_mappings` 可自定义显示名称：
```yaml
column_mappings:
  my_custom_property: "My Custom Property Name"
```

### 文件模式匹配
修改 `file_patterns` 可适配不同的文件命名：
```yaml
file_patterns:
  training_log: "**/my_custom_lcurve*.out"
  md_trajectory: "**/custom_dump*.lammpstrj"
```

### 性能优化
对于大数据集，可以：
1. 减少 `max_points` 限制显示点数
2. 增加 `refresh_interval` 降低刷新频率
3. 启用 `parallel_loading` 并行加载

## 故障排除

### 仪表盘无法启动
```bash
# 检查端口占用
lsof -i :8050

# 使用其他端口
python monitoring_dashboard.py --port 8080
```

### 数据不显示
1. 检查配置文件中的路径是否正确
2. 确认数据文件格式符合要求
3. 查看控制台日志获取错误信息
4. 启用debug模式获取详细信息

### 3D可视化不工作
```bash
# 安装ASE
pip install ase

# 确保轨迹文件格式正确
```

### 文件监控不工作
```bash
# 安装watchdog
pip install watchdog

# 检查文件权限
```

## API参考

### DataManager类
```python
from monitoring_dashboard import DataManager, DashboardConfig, load_config

config = load_config("dashboard_config.yaml")
dm = DataManager(config)

# 获取各类数据
training_df = dm.get_training_data()
md_df = dm.get_md_trajectory_data()
screening_df = dm.get_screening_results()
al_df = dm.get_active_learning_progress()
```

### 程序化导出
```python
from monitoring_dashboard import generate_pdf_report, DataManager, load_config

config = load_config("dashboard_config.yaml")
dm = DataManager(config)
generate_pdf_report("report.pdf", dm, config)
```

## 开发指南

### 添加新的图表
1. 在 `create_*_tab()` 函数中添加图表组件
2. 在 `register_callbacks()` 中添加回调函数
3. 在 `DataManager` 中添加数据读取方法

### 自定义样式
修改外部CSS文件或Dash组件的 `style` 属性。

## 版本历史

### v1.0.0 (2025-03-09)
- 初始版本
- 支持ML训练、MD模拟、筛选、主动学习四个模块
- 实现实时监控和导出功能

## 许可证
MIT License

## 联系方式
如有问题或建议，请通过GitHub Issues反馈。
