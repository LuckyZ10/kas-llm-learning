# Battery Screening Pipeline - 固态电解质高通量筛选管道

针对Li/Na离子导体的高通量计算筛选工作流，集成DFT、机器学习势和分子动力学进行材料性能预测。

## 功能特性

### 1. 自动结构获取
- 从Materials Project API自动查询候选结构
- 支持Li、Na、K等多种离子导体体系
- 自动生成替代变体结构（如Li→Na替代）

### 2. 特征工程
- **Matminer描述符**: 结构异质性、化学有序度、密度特征等
- **Dscribe SOAP**: 平滑重叠原子位置描述符，捕捉局部环境
- 组分特征: 元素属性、氧化态等

### 3. 多层级筛选策略
```
MP数据库查询 → 特征计算 → DFT初筛 → ML势训练 → ML-MD深度采样 → 综合排名
```

### 4. 性能预测
- 形成能 (DFT)
- 离子扩散系数 (ML-MD)
- 离子电导率 (Nernst-Einstein方程)
- 活化能 (Arrhenius拟合)
- 稳定性排名

## 安装依赖

```bash
# 核心依赖
pip install pymatgen mp-api ase matminer scikit-learn pandas numpy

# 可选: Dscribe SOAP描述符
pip install dscribe

# 可选: DeePMD-kit (ML势训练)
pip install deepmd-kit dpdata

# 可选: LAMMPS (MD模拟)
# 需要单独安装LAMMPS并编译Python接口
```

## 快速开始

### 1. 设置API密钥

```bash
export MP_API_KEY="your_materials_project_api_key"
```

### 2. 命令行使用

```bash
# 创建默认配置文件
python battery_screening_pipeline.py --create-config

# 运行Li离子导体筛选
python battery_screening_pipeline.py --target Li --max-candidates 50 --n-dft 10

# 使用自定义配置
python battery_screening_pipeline.py --config screening_config.yaml
```

### 3. Python API使用

```python
from battery_screening_pipeline import (
    BatteryScreeningPipeline,
    BatteryScreeningConfig,
    ScreeningCriteria
)

# 配置
config = BatteryScreeningConfig(
    target_ion="Li",
    use_ml_acceleration=True,
    md_temperatures=[300, 600, 900],
)

# 筛选标准
criteria = ScreeningCriteria(
    target_ion="Li",
    max_natoms=100,
    min_gap=2.0,  # 电子绝缘性
    max_ehull=0.1,  # 热力学稳定性
)

# 运行管道
pipeline = BatteryScreeningPipeline(config)
results = pipeline.run_full_pipeline(
    criteria=criteria,
    max_candidates=100,
    n_dft_screen=20
)

# 查看结果
print(results.head(10))
```

### 4. 运行示例

```bash
# 快速筛选示例（仅MP数据）
python screening_examples.py --example 1

# Li离子导体完整筛选
python screening_examples.py --example 2

# Na离子导体筛选
python screening_examples.py --example 3

# 特定化学系统分析
python screening_examples.py --example 4
```

## 工作流详解

### Step 1: 获取候选结构
- 从Materials Project查询含目标离子的化合物
- 筛选条件：带隙>2eV（电子绝缘）、能量凸包<0.1eV/atom（稳定）
- 生成离子替换变体（如Li→Na, K）

### Step 2: 特征工程
- **Matminer结构特征**: 密度、对称性、异质性指标
- **SOAP描述符**: 局部原子环境的几何描述
- **组分特征**: 元素电负性、离子半径等

### Step 3: DFT初筛
- 对前N个候选进行VASP结构优化
- 计算形成能，筛选热力学稳定结构
- 保存优化后的结构和能量数据

### Step 4: ML势训练
- 使用DeePMD-kit训练深度势能
- 训练数据：DFT计算的力和能量
- 模型架构：se_e2_a描述符 + 拟合网络

### Step 5: ML-MD深度采样
- 使用训练好的ML势运行LAMMPS MD
- NVT系综平衡化 + 生产模拟
- 计算扩散系数和离子电导率

### Step 6: 综合排名
- 多目标优化：电导率(50%) + 稳定性(30%) + 带隙(20%)
- 输出排名结果和详细分析报告

## 输出文件结构

```
battery_screening/
├── db/                          # 结构数据库
├── dft_results/                 # DFT计算结果
│   └── mp-XXXXX/
│       ├── CONTCAR
│       └── OUTCAR
├── md_results/                  # MD模拟结果
│   └── mp-XXXXX/
│       ├── dump.lammpstrj
│       └── md_analysis_report.json
├── models/                      # ML势模型
│   └── model.pb
├── features.csv                 # 结构特征数据
├── ranking_results_YYYYMMDD_HHMMSS.csv  # 排名结果
└── candidates_YYYYMMDD_HHMMSS.json      # 完整候选数据
```

## 输出结果说明

### CSV结果文件
| 列名 | 说明 |
|------|------|
| rank | 综合排名 |
| material_id | Materials Project ID |
| formula | 化学式 |
| ionic_conductivity_S_cm | 离子电导率 (S/cm) |
| dft_formation_energy_eV | 形成能 (eV/atom) |
| band_gap_eV | 带隙 (eV) |
| diffusion_coefficient_cm2s | 扩散系数 (cm²/s) |
| activation_energy_eV | 活化能 (eV) |
| score | 综合得分 |

## 高级配置

### 修改MD参数
```yaml
# screening_config.yaml
md_temperatures: [300, 500, 700, 900]  # 模拟温度
md_timestep: 1.0  # 时间步长 (fs)
md_nsteps_equil: 50000  # 平衡步数
md_nsteps_prod: 500000  # 生产步数
```

### 自定义筛选条件
```python
criteria = ScreeningCriteria(
    target_ion="Li",
    allowed_anions=["O", "S", "Se", "F"],  # 允许阴离子
    allowed_cations=["P", "Si", "Ge"],      # 允许阳离子
    max_natoms=80,
    min_gap=3.0,  # 宽禁带
    max_ehull=0.05,  # 高稳定性
)
```

### 禁用ML加速（纯DFT）
```python
config = BatteryScreeningConfig(
    use_ml_acceleration=False,  # 禁用ML加速
)
```

## 性能优化建议

1. **并行计算**
   - 设置 `max_parallel` 参数控制并行任务数
   - DFT计算可使用作业调度系统

2. **ML势训练**
   - 首次运行需要足够DFT数据训练ML势
   - 可以使用预训练模型加速后续筛选

3. **快速筛选模式**
   - 减少MD步数（测试用）
   - 使用SOAP相似性快速预筛选
   - 基于文献数据训练预测模型

## 参考文献

1. Materials Project: https://materialsproject.org
2. Matminer: Ward et al., npj Comput. Mater. 4, 65 (2018)
3. Dscribe: Himanen et al., Comput. Phys. Commun. 247, 106949 (2020)
4. DeePMD-kit: Wang et al., Comput. Phys. Commun. 228, 178-184 (2018)
5. Nernst-Einstein方程用于电导率计算

## 常见问题

### Q: 如何获取Materials Project API密钥?
A: 访问 https://materialsproject.org/api 注册并获取API密钥。

### Q: DFT计算需要什么样的计算资源?
A: 建议使用HPC集群，每个结构优化约需32核、4-8小时。

### Q: 没有LAMMPS/DeePMD可以使用吗?
A: 可以，设置 `use_ml_acceleration=False`，但只能获得DFT层面的结果。

### Q: 如何加速筛选过程?
A: 
1. 使用预训练ML势
2. 减少MD步数
3. 增加DFT并行度
4. 使用较小的候选集测试

## 许可证

MIT License

## 作者

高通量筛选专家 AI Assistant
