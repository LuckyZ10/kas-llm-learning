# 自动化测试与质量保障框架
# Automated Testing & Quality Assurance Framework

## 框架概览

本框架为DFT-MD-ML多尺度材料计算平台提供全面的自动化测试和质量保障能力。

## 统计信息

- **测试代码总量**: ~25,000+ 行
- **测试文件数**: 15+
- **测试用例数**: 200+
- **CI/CD工作流**: 3个
- **测试类别**: 5大类

## 目录结构

```
dft_lammps_research/
├── tests/                          # 测试目录
│   ├── conftest.py                 # pytest配置和fixtures (500+行)
│   ├── utils.py                    # 测试工具库 (600+行)
│   ├── README.md                   # 测试文档
│   ├── unit/                       # 单元测试
│   │   └── test_core_modules.py    # 核心模块单元测试 (400+行)
│   ├── integration/                # 集成测试
│   │   └── test_integration.py     # 跨模块集成测试 (400+行)
│   ├── e2e/                        # 端到端测试
│   │   └── test_e2e_workflows.py   # 完整工作流测试 (500+行)
│   ├── regression/                 # 回归测试
│   │   ├── __init__.py             # 回归测试基础设施 (400+行)
│   │   ├── test_dft_regression.py  # DFT计算回归测试 (500+行)
│   │   ├── test_md_regression.py   # MD模拟回归测试 (600+行)
│   │   └── test_ml_regression.py   # ML势回归测试 (600+行)
│   └── performance/                # 性能测试
│       └── test_benchmarks.py      # 性能基准测试 (500+行)
│
├── .github/workflows/              # CI/CD配置
│   ├── ci-cd.yml                   # 主CI/CD流水线 (400+行)
│   ├── nightly-regression.yml      # 夜间回归测试 (200+行)
│   └── security-scan.yml           # 安全扫描 (100+行)
│
├── scripts/                        # 测试脚本
│   └── run_tests.py                # 测试运行脚本 (200+行)
│
├── pytest.ini                      # pytest主配置
├── pyproject.toml                  # 项目配置(Black/isort/mypy)
├── codecov.yml                     # Codecov覆盖率配置
├── Makefile                        # 测试命令快捷方式
└── requirements-test.txt           # 测试依赖
```

## 测试类别详解

### 1. 单元测试 (Unit Tests)

**位置**: `tests/unit/`

测试独立的代码单元，确保各个组件独立正确工作。

- **DFT解析器单元测试**: OUTCAR解析、能量/力提取、收敛检测
- **MD模拟器单元测试**: Verlet积分器、温度计算、PBC处理
- **ML势单元测试**: 能量/力预测、梯度计算、模型参数
- **HPC调度器单元测试**: 作业脚本生成、资源请求验证
- **工作流管理单元测试**: 依赖解析、状态转换、错误恢复

**运行**: `pytest -m unit` 或 `make test-unit`

### 2. 集成测试 (Integration Tests)

**位置**: `tests/integration/`

测试模块之间的交互和集成。

- DFT解析器 → 力场拟合器集成
- 力场拟合器 → LAMMPS生成器集成
- NEP数据准备 → NEP训练集成
- MD模拟 → 分析工具集成
- 工作流管理器 → 调度器集成

**运行**: `pytest -m integration` 或 `make test-integration`

### 3. 端到端测试 (E2E Tests)

**位置**: `tests/e2e/`

测试完整的用户工作流。

- DFT → ML → MD完整工作流
- 电池材料筛选工作流
- 主动学习工作流
- HPC作业提交到完成
- Web UI交互流程
- 错误恢复和检查点恢复

**运行**: `pytest -m e2e` 或 `make test-e2e`

### 4. 回归测试 (Regression Tests)

**位置**: `tests/regression/`

确保科学计算结果的准确性和可重复性。

#### DFT回归测试
- 能量计算一致性
- 力计算一致性
- 应力张量一致性
- SCF收敛性
- 跨平台数值稳定性

#### MD回归测试
- 轨迹可重复性
- 能量守恒
- 温度/压强控制稳定性
- 长时间稳定性

#### ML回归测试
- 预测一致性
- 数值稳定性
- 能量-力一致性
- 对称性守恒

**运行**: `pytest -m regression` 或 `make test-regression`

### 5. 性能测试 (Performance Tests)

**位置**: `tests/performance/`

检测性能退化。

- DFT解析性能
- MD模拟性能
- ML训练/推理性能
- 工作流执行性能
- 数据I/O性能
- 内存使用基准

**运行**: `pytest -m performance` 或 `make test-performance`

## CI/CD流水线

### 主CI/CD流水线 (ci-cd.yml)

**触发条件**:
- Push到main/master分支
- Pull Request
- 每日定时运行
- 手动触发

**阶段**:
1. **代码质量检查**: Black、isort、flake8、mypy、bandit
2. **单元测试**: 多Python版本(3.9, 3.10, 3.11)
3. **集成测试**: 带Redis/MongoDB服务
4. **回归测试**: 夜间运行
5. **性能测试**: 夜间运行
6. **端到端测试**: 主要分支
7. **文档构建**: MkDocs

### 夜间回归测试 (nightly-regression.yml)

**运行时间**: 每日03:00 UTC

包含:
- DFT计算一致性测试
- MD轨迹可重复性测试
- ML势稳定性测试
- 性能基准对比
- 邮件通知

### 安全扫描 (security-scan.yml)

**扫描内容**:
- 依赖漏洞扫描 (Safety, pip-audit)
- 代码安全扫描 (Bandit)
- 密钥泄露检测 (GitLeaks)
- CodeQL分析

## 代码质量工具

| 工具 | 用途 | 配置 |
|------|------|------|
| Black | 代码格式化 | pyproject.toml |
| isort | 导入排序 | pyproject.toml |
| flake8 | 代码检查 | pyproject.toml |
| mypy | 类型检查 | pyproject.toml |
| bandit | 安全扫描 | pyproject.toml |
| pytest-cov | 覆盖率 | pytest.ini |

## 测试标记体系

| 标记 | 描述 | 使用场景 |
|------|------|----------|
| `unit` | 单元测试 | 每次提交 |
| `integration` | 集成测试 | 每次提交 |
| `e2e` | 端到端测试 | 主要分支 |
| `regression` | 回归测试 | 夜间运行 |
| `performance` | 性能测试 | 夜间运行 |
| `slow` | 慢速测试 | 手动/夜间 |
| `dft` | DFT相关 | 按需 |
| `md` | MD相关 | 按需 |
| `ml` | ML相关 | 按需 |
| `hpc` | HPC相关 | 按需 |

## 关键特性

### 1. Fixtures系统

预定义的测试数据生成器:
- `mock_atoms`: 模拟原子结构
- `mock_trajectory`: 模拟轨迹
- `mock_dft_frames`: 模拟DFT计算结果
- `mock_vasp_config`: VASP配置
- `mock_nep_config`: NEP配置

### 2. 数值比较工具

`NumericalComparator` 类提供:
- `compare_scalars`: 标量比较
- `compare_arrays`: 数组比较
- `compare_energies`: 能量比较（DFT容差）
- `compare_forces`: 力比较
- `compute_trajectory_hash`: 轨迹哈希

### 3. 参考数据管理

`ReferenceDataManager` 管理:
- DFT参考数据
- MD参考数据
- ML参考数据

### 4. 性能监控

`PerformanceMonitor` 类提供:
- 执行时间测量
- 内存使用监控
- CPU使用率跟踪
- 报告生成

## 使用方法

### 快速开始

```bash
# 安装测试依赖
make install-test-deps

# 运行所有测试
make test

# 运行快速测试(跳过慢速)
make test-quick
```

### 特定测试类别

```bash
# 单元测试
make test-unit

# 集成测试
make test-integration

# 回归测试
make test-regression

# 性能测试
make test-performance

# 端到端测试
make test-e2e
```

### 代码质量

```bash
# 格式化代码
make format

# 检查代码
make lint

# 类型检查
make type-check

# 安全扫描
make security

# 完整CI流程
make ci
```

### 覆盖率报告

```bash
# HTML报告
make coverage-html

# XML报告
make coverage-xml

# 查看报告
open htmlcov/index.html
```

## 持续监控

### 覆盖率监控

- **目标覆盖率**: 70%
- **DFT模块**: 80%
- **ML模块**: 75%
- **HPC模块**: 70%

### 性能基准

- **基准保存**: `pytest --benchmark-save=baseline`
- **性能对比**: `pytest --benchmark-compare`
- **回归检测**: 性能下降>20%触发告警

### 告警机制

1. **测试失败**: GitHub Actions通知
2. **覆盖率下降**: Codecov评论
3. **安全漏洞**: 邮件通知
4. **性能退化**: PR评论

## 最佳实践

1. **保持测试独立**: 每个测试可独立运行
2. **使用Fixtures**: 复用测试数据
3. **适当标记**: 正确标记慢速测试
4. **清晰命名**: 测试名说明测试内容
5. **覆盖边界**: 测试正常和异常情况

## 故障排除

### 测试发现失败
```bash
# 检查测试文件命名 test_*.py
pytest --collect-only
```

### Fixtures找不到
```bash
# 确保conftest.py在正确位置
pytest --fixtures
```

### 慢速测试超时
```bash
# 增加超时时间
pytest --timeout=600
```

## 扩展指南

添加新测试:
1. 选择正确的测试目录
2. 添加适当的标记
3. 使用现有fixtures
4. 更新文档
5. 运行测试验证

## 贡献者信息

本测试框架由Phase 66开发，确保DFT-MD-ML平台的稳定性和可靠性。

---

**文档版本**: 1.0  
**最后更新**: 2024-03-11
