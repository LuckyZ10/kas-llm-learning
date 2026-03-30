# DFT-LAMMPS 计算-实验闭环与机器人接口模块

## 概述

本模块实现了一个完整的**自驱动实验室（Self-Driving Laboratory）**框架，支持计算与实验的自动闭环。通过结合机器学习、机器人自动化和先进表征技术，实现材料的自主发现与优化。

## 模块结构

```
dftlammps/
├── experiment/          # 实验自动化模块
│   ├── lab_automation.py       # 实验室自动化核心
│   └── synthesis_planning.py   # 合成规划
├── characterization/    # 表征接口模块
│   ├── xrd_analysis.py         # XRD数据分析
│   ├── sem_analysis.py         # SEM数据分析
│   ├── tem_analysis.py         # TEM数据分析
│   └── comparison.py           # 计算-实验对比
└── applications/        # 应用案例
    ├── autonomous_battery_discovery.py  # 自驱动电池材料发现
    ├── robotic_catalysis.py             # 机器人催化实验
    └── autonomous_alloy_design.py       # 自动合金设计
```

---

## 1. 实验自动化模块 (experiment/)

### 1.1 lab_automation.py - 实验室自动化

**功能特性：**
- 机器人接口抽象层 (RobotInterface)
- 支持多种机器人平台：
  - `SimulatedRobot`: 模拟机器人，用于开发和测试
  - `UR5Robot`: Universal Robots UR5 协作机械臂
  - `OT2Robot`: Opentrons OT-2 液体处理机器人
- 合成协议生成和执行
- 实验结果自动读取和存储
- 闭环优化循环
- 实验队列管理

**核心类：**

| 类名 | 功能 |
|------|------|
| `ExperimentStatus` | 实验状态枚举 |
| `RobotCommand` | 机器人命令类型 |
| `MaterialSpec` | 材料规格定义 |
| `SynthesisParameter` | 合成参数 |
| `RobotInstruction` | 机器人指令 |
| `ExperimentResult` | 实验结果 |
| `SynthesisProtocol` | 合成协议 |
| `ProtocolGenerator` | 协议生成器 |
| `ExperimentRunner` | 实验运行器 |
| `AutonomousLab` | 自驱动实验室主控 |
| `ExperimentQueue` | 实验队列管理 |

**支持的合成方法：**
- 固相合成 (Solid State)
- 溶胶-凝胶法 (Sol-Gel)
- 水热合成 (Hydrothermal)
- 共沉淀 (Co-precipitation)
- 固态电解质专用合成

**使用示例：**

```python
from dftlammps.experiment import create_lab, create_simulated_robot
from dftlammps.experiment import ProtocolGenerator, SynthesisParameter

# 创建实验室
lab = create_lab()
robot = create_simulated_robot("LabRobot1")
lab.add_robot("robot1", robot)

# 生成合成协议
generator = ProtocolGenerator()
protocol = generator.generate_protocol(
    "solid_state",
    {"Li": 3.0, "P": 1.0, "S": 4.0},
    SynthesisParameter(temperature=550, time=3600)
)

# 运行实验
result = await lab.run_experiment("robot1", protocol)
```

---

### 1.2 synthesis_planning.py - 合成规划

**功能特性：**
- 可合成性预测（知识库和机器学习两种模式）
- 合成路径规划
- 前驱体选择和优化
- 反应条件优化

**核心类：**

| 类名 | 功能 |
|------|------|
| `SynthesisDifficulty` | 合成难度等级 |
| `ReactionType` | 反应类型枚举 |
| `ChemicalCompound` | 化学化合物定义 |
| `Precursor` | 前驱体 |
| `SynthesisStep` | 合成步骤 |
| `SynthesisRoute` | 合成路线 |
| `SynthesisPredictor` | 合成预测器基类 |
| `KnowledgeBasedPredictor` | 知识库预测器 |
| `MLPredictor` | 机器学习预测器 |
| `SynthesisPlanner` | 合成规划器 |
| `PrecursorOptimizer` | 前驱体优化器 |

**使用示例：**

```python
from dftlammps.experiment import create_planner, predict_synthesis_feasibility

# 预测合成可行性
result = predict_synthesis_feasibility("Li3PS4")
print(f"可合成性: {result['is_synthesizable']}")
print(f"成功概率: {result['success_probability']:.2%}")

# 规划合成路线
planner = create_planner("knowledge")
routes = planner.plan_synthesis("Li3PS4", {"max_cost": 100})

for route in routes:
    print(f"路线: {len(route.steps)} 步")
    print(f"预期产率: {route.total_yield:.2%}")
    print(f"估计成本: ${route.estimated_cost:.2f}")
```

---

## 2. 表征接口模块 (characterization/)

### 2.1 xrd_analysis.py - XRD数据分析

**功能特性：**
- 多格式XRD数据解析 (XY, DAT, CSV, XRDML, RAW)
- 自动峰检测和拟合
- 物相识别（内置数据库）
- Rietveld精修接口（GSAS-II, FullProf）
- 计算-实验XRD对比

**核心类：**

| 类名 | 功能 |
|------|------|
| `XRDPeak` | XRD峰数据 |
| `XRDPattern` | XRD衍射图谱 |
| `PhaseIdentification` | 物相识别结果 |
| `XRDParser` | XRD数据解析器 |
| `PeakDetector` | 峰检测器 |
| `PhaseIdentifier` | 物相识别器 |
| `XRDAnalyzer` | XRD综合分析器 |
| `RietveldRefinement` | Rietveld精修接口 |

**使用示例：**

```python
from dftlammps.characterization import analyze_xrd, compare_xrd

# 分析XRD数据
result = analyze_xrd("sample.xy")
print(f"检测到 {result['statistics']['num_peaks']} 个峰")

# 识别物相
for phase in result['phases']:
    print(f"{phase['phase_name']}: {phase['match_score']:.2%} 匹配")

# 对比实验和计算XRD
comparison = compare_xrd("exp.xy", "calc.xy")
print(f"相关性: {comparison['correlation']:.4f}")
print(f"Rwp: {comparison['Rwp']:.4f}")
```

---

### 2.2 sem_analysis.py - SEM数据分析

**功能特性：**
- SEM图像加载（TIFF, DM3/DM4, 标准格式）
- 颗粒检测和统计
- 形貌学分析
- EDS能谱分析
- 粒径分布拟合

**核心类：**

| 类名 | 功能 |
|------|------|
| `Particle` | 颗粒数据 |
| `SEMImage` | SEM图像 |
| `MorphologyMetrics` | 形貌学指标 |
| `EDSData` | EDS能谱数据 |
| `SEMImageLoader` | 图像加载器 |
| `ParticleAnalyzer` | 颗粒分析器 |
| `EDSAnalyzer` | EDS分析器 |
| `SEMAnalyzer` | SEM综合分析器 |

**使用示例：**

```python
from dftlammps.characterization import analyze_sem, analyze_particles

# 分析SEM图像
result = analyze_sem("sample_sem.tif")
print(f"颗粒数: {result['particle_analysis']['num_particles']}")

# 形貌分析
morph = result['particle_analysis']['morphology']
print(f"平均粒径: {morph['mean_particle_size']:.2f} nm")
print(f"D50: {morph['size_distribution']['d50']:.2f} nm")
```

---

### 2.3 tem_analysis.py - TEM数据分析

**功能特性：**
- HRTEM图像分析
- 晶格条纹检测和标定
- SAED衍射图谱分析
- 晶带轴确定
- FFT分析

**核心类：**

| 类名 | 功能 |
|------|------|
| `LatticeFringe` | 晶格条纹 |
| `DiffractionSpot` | 衍射斑点 |
| `SAEDPattern` | SAED图谱 |
| `HRTEMImage` | HRTEM图像 |
| `CrystalInfo` | 晶体信息 |
| `TEMImageLoader` | TEM图像加载器 |
| `LatticeAnalyzer` | 晶格分析器 |
| `DiffractionAnalyzer` | 衍射分析器 |
| `TEMAnalyzer` | TEM综合分析器 |

**使用示例：**

```python
from dftlammps.characterization import analyze_tem_hrtem, analyze_tem_saed

# 分析HRTEM
result = analyze_tem_hrtem("sample_hrtem.dm4")
print(f"检测到 {result['num_fringes']} 组晶格条纹")

# 分析SAED
result = analyze_tem_saed("sample_saed.dm4")
print(f"检测到 {result['num_spots']} 个衍射斑点")
```

---

### 2.4 comparison.py - 计算-实验对比

**功能特性：**
- 多尺度数据对比框架
- 光谱数据对比（XRD, Raman, FTIR）
- 结构对比（RMSD, 晶格参数）
- 性质对比验证
- 反馈循环（模型改进建议）

**核心类：**

| 类名 | 功能 |
|------|------|
| `ComparisonMetrics` | 对比指标 |
| `StructureComparison` | 结构对比结果 |
| `PropertyComparison` | 性质对比结果 |
| `SpectrumComparator` | 光谱对比器 |
| `XRDComparator` | XRD对比器 |
| `StructureComparator` | 结构对比器 |
| `PropertyComparator` | 性质对比器 |
| `ImageComparator` | 图像对比器 |
| `FeedbackLoop` | 反馈循环 |
| `ComparisonManager` | 对比管理器 |

**使用示例：**

```python
from dftlammps.characterization import (
    compare_calculation_experiment,
    generate_validation_report
)

# 全面对比
exp_data = {
    "properties": {"band_gap": 3.2, "lattice_constant": 4.18},
    "xrd": {"two_theta": [...], "intensity": [...]}
}

calc_data = {
    "properties": {"band_gap": 3.15, "lattice_constant": 4.20},
    "xrd": {"two_theta": [...], "intensity": [...]}
}

result = compare_calculation_experiment(calc_data, exp_data)
print(f"综合评分: {result['overall_score']:.2%}")

# 生成报告
report = generate_validation_report(calc_data, exp_data)
print(report)
```

---

## 3. 应用案例 (applications/)

### 3.1 自驱动电池材料发现

**文件：** `autonomous_battery_discovery.py`

**功能：**
- 固态电解质材料自动发现
- 离子电导率优化
- 电化学稳定性评估
- 机器人合成-测试闭环

**支持的体系：**
- Garnet (Li₇La₃Zr₂O₁₂)
- NASICON (Li₁.₃Al₀.₃Ti₁.₇P₃O₁₂)
- Perovskite (Li₃ₓLa₂/₃₋ₓTiO₃)
- 硫化物 (Li₆PS₅Cl)
- Argyrodite
- LISICON

**使用示例：**

```python
from dftlammps.applications import run_battery_discovery

# 运行电池材料发现
best_materials = await run_battery_discovery(
    target_conductivity=1.0,  # mS/cm
    max_iterations=20,
    require_experiments=True
)

for material in best_materials:
    print(f"{material.formula}: {material.measured_ionic_conductivity:.3f} mS/cm")
```

---

### 3.2 机器人催化实验

**文件：** `robotic_catalysis.py`

**功能：**
- 催化剂库管理
- 反应条件自动优化
- 在线活性检测
- 反应动力学分析

**支持的反应类型：**
- 加氢 (Hydrogenation)
- 氧化 (Oxidation)
- 偶联 (Coupling, Cross-coupling)
- 羰基化 (Carbonylation)
- 烯烃复分解 (Metathesis)
- 电催化 (Electrocatalysis)
- 光催化 (Photocatalysis)

**使用示例：**

```python
from dftlammps.applications import run_catalysis_optimization

# 运行催化优化
result = await run_catalysis_optimization(
    reaction_type="hydrogenation",
    substrate="acetophenone",
    target_yield=0.95
)

print(f"最佳产率: {result.yield_:.1f}%")
print(f"TOF: {result.turnover_frequency:.1f} h⁻¹")
```

---

### 3.3 自动合金设计-合成-测试

**文件：** `autonomous_alloy_design.py`

**功能：**
- 合金成分设计（HEA, 高温合金, 钛合金等）
- 相稳定性预测（VEC规则）
- 力学性能预测
- 电弧熔炼自动合成
- 力学性能自动测试

**支持的合金体系：**
- 高熵合金 (HEA)
- 中熵合金 (MEA)
- 高温合金 (Superalloy)
- 钛合金
- 铝合金
- 钢铁
- 形状记忆合金
- 磁性合金

**使用示例：**

```python
from dftlammps.applications import develop_alloy

# 开发高熵合金
best_alloys = await develop_alloy(
    system="high_entropy_alloy",
    target_strength=1200.0,  # MPa
    max_iterations=10
)

for alloy in best_alloys:
    print(f"{alloy.name}: {alloy.actual_properties.yield_strength:.0f} MPa")
```

---

## 4. 架构设计

### 4.1 闭环流程

```
┌─────────────────────────────────────────────────────────────┐
│                    自驱动实验室闭环                          │
└─────────────────────────────────────────────────────────────┘

  ┌──────────┐     ┌──────────┐     ┌──────────┐
  │ 候选生成  │────▶│ 计算筛选  │────▶│ 合成规划  │
  └──────────┘     └──────────┘     └──────────┘
                                              │
                                              ▼
  ┌──────────┐     ┌──────────┐     ┌──────────┐
  │ 模型更新  │◀────│ 对比验证  │◀────│ 机器人合成 │
  └──────────┘     └──────────┘     └──────────┘
       ▲                                  │
       └──────────────────────────────────┘
              表征数据分析 (XRD/SEM/TEM)
```

### 4.2 模块关系

```
experiment/                characterization/           applications/
    │                            │                          │
    ├─ lab_automation.py ────────┼──────────────────────────┤
    │   - RobotInterface         │   - XRD analysis         │   - Battery discovery
    │   - AutonomousLab          │   - SEM analysis         │   - Catalysis
    │                            │   - TEM analysis         │   - Alloy design
    ├─ synthesis_planning.py ────┤   - Comparison           │
        - SynthesisPlanner       │       (Feedback loop)    │
        - PrecursorOptimizer     │                          │
```

---

## 5. 依赖要求

### 必需依赖
```
numpy
scipy
```

### 可选依赖
```
scikit-learn      # 机器学习预测
scikit-image      # 图像分析
opencv-python     # 计算机视觉
Pillow            # 图像处理
ncempy            # DM3/DM4文件
mrcfile           # MRC格式
networkx          # 反应图分析
aiosqlite         # 数据库
```

---

## 6. 快速开始

### 安装

```bash
# 克隆仓库
git clone <repository>
cd dftlammps

# 安装依赖
pip install numpy scipy

# 安装可选依赖（推荐）
pip install scikit-learn scikit-image opencv-python pillow
```

### 运行示例

```bash
# 测试电池材料发现
python -m dftlammps.applications.autonomous_battery_discovery

# 测试催化优化
python -m dftlammps.applications.robotic_catalysis

# 测试合金设计
python -m dftlammps.applications.autonomous_alloy_design
```

---

## 7. 扩展开发

### 添加新的机器人接口

```python
from dftlammps.experiment import RobotInterface, RobotInstruction

class MyRobot(RobotInterface):
    async def connect(self) -> bool:
        # 实现连接逻辑
        pass
    
    async def execute_instruction(self, instruction: RobotInstruction) -> bool:
        # 实现指令执行
        pass
```

### 添加新的表征方法

```python
from dftlammps.characterization import DataReader

class AFMReader(DataReader):
    async def read(self, source: str) -> Dict[str, Any]:
        # 实现AFM数据读取
        pass
```

### 添加新的应用案例

参考 `applications/` 目录下的现有案例，实现：
- 候选生成器
- 性质预测器
- 实验流程

---

## 8. 参考文献

1. MacLeod, B. P., et al. (2020). Self-driving laboratory for accelerated discovery of thin-film materials. *Science Advances*.
2. Burger, B., et al. (2020). A mobile robotic chemist. *Nature*.
3. Nikolaev, P., et al. (2016). Autonomy in materials research: a case study in carbon nanotube growth. *npj Computational Materials*.
4. Shatruk, M. (2019). The rise of high-entropy alloys. *Science*.
5. Goodenough, J. B., & Park, K. S. (2013). The Li-ion rechargeable battery: a perspective. *Journal of the American Chemical Society*.

---

## 9. 许可证

MIT License

---

## 10. 联系方式

DFT-LAMMPS Development Team

---

**版本：** v1.0  
**最后更新：** 2026-03-09  
**代码统计：** ~2500+ 行核心代码
