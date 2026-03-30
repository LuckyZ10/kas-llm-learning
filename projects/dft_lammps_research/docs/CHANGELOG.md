# 更新日志 (Changelog)

所有重要变更都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
并且本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

---

## [未发布]

### 计划中
- 支持 FHI-aims DFT代码
- 集成 ORCA量子化学软件
- 添加 ONETEP线性标度DFT支持
- 支持 CHGNet预训练模型
- 实现 M3GNet集成

---

## [1.0.0] - 2026-03-09

### 新增
- **核心工作流引擎**
  - 端到端DFT→ML势→MD工作流
  - 模块化阶段设计（获取→DFT→训练→MD→分析）
  - 自动依赖管理和阶段顺序控制
  - 全面的错误处理和重试机制

- **DFT集成**
  - VASP计算接口（结构优化、AIMD）
  - Quantum ESPRESSO支持
  - ABACUS初步支持
  - 自动k点生成和收敛测试

- **机器学习势**
  - DeepMD-kit完整训练流程
  - NEP (GPUMD) 势训练
  - MACE等变神经网络支持
  - 集成SNAP经典ML势

- **主动学习**
  - 探索-标注-重训练循环
  - 不确定性量化（模型集成）
  - 候选结构自动筛选
  - 收敛监控和早期停止

- **MD模拟**
  - LAMMPS集成（CPU/GPU）
  - 多温度点批量模拟
  - NVT/NPT/NVE系综支持
  - 轨迹分析和后处理

- **高通量筛选**
  - 批量结构处理
  - 并行作业提交
  - 结果数据库管理
  - 自动化报告生成

- **HPC集成**
  - Slurm调度器支持
  - PBS/Torque支持
  - SGE (Sun Grid Engine) 支持
  - 作业监控和资源管理

- **监控和可视化**
  - Dash-based监控仪表板
  - 实时训练曲线
  - MD轨迹可视化
  - 扩散分析和Arrhenius拟合

- **示例和文档**
  - 完整演示工作流（无需DFT/MD软件）
  - 7个详细教程
  - 5个应用案例（电池、催化剂等）
  - Docker完整环境

### 变更
- 重构项目结构，提高模块化程度
- 统一配置系统（dataclass-based）
- 改进日志记录和错误报告

### 修复
- 修复LAMMPS数据文件写入格式问题
- 修正扩散系数单位换算
- 修复多线程竞争条件

---

## [0.9.0] - 2026-02-15

### 新增
- 初步DFT到LAMMPS桥接功能
- NEP训练管道原型
- 基础HPC调度器
- 检查点管理器

### 变更
- 从单一脚本架构迁移到模块化设计
- 引入配置类替代字典配置

---

## [0.8.0] - 2026-01-20

### 新增
- 电池材料筛选管道原型
- Li3PS4工作流示例
- 基础监控仪表板

### 修复
- Materials Project API更新兼容性
- ASE版本兼容性问题

---

## [0.7.0] - 2025-12-10

### 新增
- 集成DeepMD-kit 2.2.x
- 多模型集成推理
- 扩散系数计算
- 离子电导率计算

### 变更
- 更新Pymatgen到2023.7+
- 升级ASE到3.22+

---

## [0.6.0] - 2025-11-01

### 新增
- 主动学习框架原型
- 模型偏差计算
- 候选结构选择策略

### 修复
- 内存泄漏问题
- 大体系轨迹处理

---

## [0.5.0] - 2025-09-15

### 新增
- LAMMPS GPU支持
- OVITO集成
- VMD脚本生成

### 变更
- 改进并行性能
- 优化大文件I/O

---

## [0.4.0] - 2025-08-01

### 新增
- 量子 espresso 支持
- 伪势自动下载
- 基础高通量框架

---

## [0.3.0] - 2025-06-20

### 新增
- VASP工作流模板
- 结构优化自动化
- 能带结构计算

### 变更
- 重构DFT模块
- 统一计算器接口

---

## [0.2.0] - 2025-05-10

### 新增
- Materials Project集成
- 结构获取自动化
- 基础分析工具

---

## [0.1.0] - 2025-04-01

### 新增
- 项目初始化
- 基础架构设计
- 概念验证代码

---

## 版本说明

### 版本号格式

版本号格式：`主版本号.次版本号.修订号`

- **主版本号**：不兼容的API修改
- **次版本号**：向下兼容的功能新增
- **修订号**：向下兼容的问题修正

### 标签说明

- `[未发布]` - 已合并到主分支但尚未发布的变更
- `[YANKED]` - 因严重问题被撤回的安全版本

---

## 迁移指南

### 从 0.9.x 升级到 1.0.0

#### 配置变更

**旧格式 (0.9.x)**:
```python
config = {
    "dft_code": "vasp",
    "encut": 520,
    "ml_framework": "deepmd"
}
```

**新格式 (1.0.0)**:
```python
from integrated_materials_workflow import (
    IntegratedWorkflowConfig,
    DFTStageConfig,
    MLPotentialConfig
)

config = IntegratedWorkflowConfig(
    dft_config=DFTStageConfig(code="vasp", encut=520),
    ml_config=MLPotentialConfig(framework="deepmd")
)
```

#### API变更

| 旧API | 新API |
|-------|-------|
| `run_workflow()` | `IntegratedMaterialsWorkflow.run()` |
| `train_deepmd_model()` | `MLTrainingStage.train_deepmd()` |
| `submit_slurm_job()` | `HPCScheduler.submit_job()` |

#### 环境要求

- Python >= 3.10（之前 >= 3.8）
- ASE >= 3.22（之前 >= 3.21）
- Pymatgen >= 2023.7（之前 >= 2022.0）

---

## 贡献者

感谢以下贡献者（按字母顺序）：

- 开发团队核心成员
- 社区测试者和反馈提供者

---

## 引用

如果您在研究中使用了本框架，请引用：

```bibtex
@software{dft_lammps_framework,
  author = {DFT+LAMMPS Framework Team},
  title = {DFT + LAMMPS Multi-Scale Materials Simulation Framework},
  year = {2026},
  version = {1.0.0},
  url = {https://github.com/yourusername/dft-lammps-framework}
}
```
