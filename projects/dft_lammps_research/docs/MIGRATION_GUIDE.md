# 代码整理迁移记录

## 迁移时间
2026-03-12

## 迁移目标
统一项目结构，提高代码可维护性和可读性。

## 目录结构变化

### 新的顶层结构
```
dft_lammps_research/
├── core/              # 核心计算引擎 (新增)
├── workflows/         # 应用工作流 (重组)
├── platform/          # 平台服务 (新增)
├── intelligence/      # AI/智能模块 (重组)
├── simulation/        # 高级模拟 (重组)
├── validation/        # 实验验证 (移动)
├── examples/          # 示例代码 (重组)
├── tests/             # 测试套件 (重组)
├── docs/              # 文档 (重组)
├── scripts/           # 工具脚本 (新增)
├── references/        # 参考文献 (保留)
└── tutorials/         # 教程 (保留)
```

## 文件移动清单

### 根目录文件迁移

| 原位置 | 新位置 | 说明 |
|--------|--------|------|
| `battery_screening_pipeline.py` | `workflows/battery/` | 电池筛选管道 |
| `checkpoint_manager.py` | `core/common/` | Checkpoint管理 |
| `dft_to_lammps_bridge.py` | `core/` | DFT-MD桥接 |
| `integrated_materials_workflow.py` | `core/` | 集成工作流 |
| `hpc_scheduler.py` | `platform/hpc/` | HPC调度器 |
| `monitoring_dashboard.py` | `platform/` | 监控仪表板 |
| `nep_training_pipeline.py` | `core/ml/` | NEP训练管道 |
| `parallel_optimizer.py` | `core/common/` | 并行优化器 |
| `screening_examples.py` | `workflows/` | 筛选示例 |
| `generate_demo_data.py` | `scripts/` | 数据生成脚本 |
| `Li3PS4_workflow_example.py` | `workflows/battery/` | 固态电解质示例 |

### 目录迁移

| 原目录 | 新目录 | 说明 |
|--------|--------|------|
| `code_templates/` | `core/templates/` | 代码模板 |
| `active_learning_v2/` | `intelligence/active_learning/v2/` | 主动学习v2 |
| `literature_survey/` | `intelligence/literature/` | 文献智能 |
| `experimental_validation/` | `validation/experimental_validation/` | 实验验证 |
| `phase_field/` | `simulation/phase_field/v1/` | 相场模拟 |
| `rl_optimizer/` | `simulation/rl/` | RL优化器 |
| `nep_training/` | `core/ml/nep/` | NEP训练 |
| `api_platform/` | `platform/api/` | API平台 |
| `webui_v2/` | `platform/web/v2/` | Web界面V2 |
| `applications/catalyst/` | `workflows/catalyst/` | 催化剂应用 |
| `applications/perovskite/` | `workflows/perovskite/` | 钙钛矿应用 |
| `applications/solid_electrolyte/` | `workflows/battery/solid_electrolyte/` | 固态电解质 |
| `benchmarks/` | `tests/benchmarks/` | 基准测试 |
| `validation_results/` | `validation/results/` | 验证结果 |
| `nep_checkpoints/` | `core/ml/nep/checkpoints/` | NEP检查点 |
| `nep_output/` | `core/ml/nep/output/` | NEP输出 |
| `docker/` | `platform/docker/` | Docker配置 |

### 示例目录重组

| 原目录 | 新目录 |
|--------|--------|
| `example_basic/` | `examples/basic/` |
| `example_advanced/` | `examples/advanced/` |
| `example_al/` | `examples/active_learning/` |
| `example_ensemble/` | `examples/advanced/ensemble/` |
| `example_monitoring/` | `examples/advanced/monitoring/` |

### 文档和配置迁移

| 原位置 | 新位置 |
|--------|--------|
| `*.md` (根目录) | `docs/` |
| `*.yaml`, `*.yml` | `scripts/` |
| `pyproject.toml` | `scripts/` |
| `pytest.ini` | `scripts/` |
| `requirements*.txt` | `scripts/` |
| `Makefile` | `scripts/` |
| `codecov.yml` | `scripts/` |
| `package.json` | `platform/web/` |

### 测试目录重组

原有 `tests/` 目录重组为：
```
tests/
├── conftest.py           # (保留)
├── __init__.py          # (保留)
├── utils.py             # (保留)
├── unit/                # (从 tests/unit/)
├── integration/         # (从 tests/integration/)
├── e2e/                 # (从 tests/e2e/)
├── performance/         # (从 tests/performance/)
├── regression/          # (从 tests/regression/)
└── benchmarks/          # (原 benchmarks/)
```

## 已删除/清理的目录

| 目录 | 原因 |
|------|------|
| `applications/` | 内容已迁移到 workflows/ |
| `ml_potentials/` | 空目录，功能合并到 core/ml/ |
| `high_throughput/` | 空目录，功能合并到 workflows/high_throughput/ |
| `workflows/` (原) | 空目录 |
| `.benchmarks/` | 缓存目录 |
| `.pytest_cache/` | 缓存目录 |

## 待手动检查事项

### 1. Python导入路径
以下文件可能需要更新 import 路径：
- 所有从根目录导入的模块
- 相对导入路径变更的文件

需要检查的导入模式：
```python
# 旧导入方式 (可能失效)
from battery_screening_pipeline import ...
from active_learning_v2 import ...

# 新导入方式
from workflows.battery.battery_screening_pipeline import ...
from intelligence.active_learning.v2 import ...
```

### 2. 配置文件路径
以下配置文件中的路径可能需要更新：
- `scripts/dashboard_config.yaml`
- `scripts/screening_config.yaml`
- `platform/docker/docker-compose.yml`

### 3. 测试配置
- `scripts/pytest.ini` 中的测试路径
- `tests/conftest.py` 中的fixture路径

### 4. 文档链接
- 所有 Markdown 文档中的相对链接
- README 中的目录引用

## 推荐的导入模式

### 绝对导入（推荐）
```python
# 从核心层导入
from core.dft.vasp import VaspCalculator
from core.ml.nep import NEPTrainer

# 从平台层导入
from platform.api.auth import APIKeyManager
from platform.hpc.scheduler import HPCScheduler

# 从智能层导入
from intelligence.active_learning import ActiveLearner
from intelligence.literature import LiteratureAnalyzer

# 从工作流导入
from workflows.battery import BatteryScreeningPipeline
```

### 相对导入（模块内部）
```python
# 在 core/ml/nep/core.py 中
from ..common import CheckpointManager
from ...platform.api import APIService
```

## 后续优化建议

### 1. 创建 __init__.py
为新目录添加适当的 `__init__.py`，简化导入：
```python
# core/__init__.py
from .dft import VaspCalculator, QECalculator
from .md import LammpsSimulator
from .ml import NEPTrainer, DeepMDTrainer
```

### 2. 添加 setup.py
创建项目级的 `setup.py` 或 `pyproject.toml`，支持：
```bash
pip install -e .
```

### 3. 统一配置管理
考虑使用统一的配置管理系统，如：
- Hydra
- OmegaConf
- Pydantic Settings

### 4. 完善 CI/CD
更新 `.github/workflows/` 中的路径配置。

### 5. 添加类型检查
配置 mypy 进行静态类型检查：
```ini
# mypy.ini
[mypy]
python_version = 3.9
warn_return_any = True
warn_unused_configs = True
```

## 文件统计

迁移前后对比：

| 指标 | 迁移前 | 迁移后 |
|------|--------|--------|
| Python文件数 | ~260 | ~260 |
| 目录数 | 混乱 | 清晰分层 |
| 根目录文件 | 20+ | 精简 |
| 空目录 | 多个 | 已清理 |

## 验证检查清单

- [ ] 所有 Python 文件可以正常导入
- [ ] 测试可以正常运行
- [ ] 示例代码可以执行
- [ ] 文档链接有效
- [ ] Docker 构建成功
- [ ] CI/CD 通过

## 回滚计划

如需回滚，可以使用以下命令恢复原始结构：
```bash
# 从 Git 恢复（如果有提交）
git checkout <commit-before-migration>

# 或者从备份恢复
# (迁移前已建议创建完整备份)
```
