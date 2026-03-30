# 材料数据库与自动化工具研究报告
## 模块2: 高通量工作流自动化 (FireWorks + MongoDB)

**研究时间**: 2026-03-08 16:35+

---

## 1. FireWorks 工作流引擎

### 1.1 核心架构
FireWorks采用**中心化服务器模型**，由服务器管理工作流，workers执行作业。

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   LaunchPad     │◄────►│     Rocket      │◄────►│  FireWorker     │
│  (MongoDB)      │      │  (作业执行)      │      │  (计算节点)      │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

### 1.2 核心概念
| 术语 | 说明 |
|------|------|
| **FireServer** | MongoDB数据库，控制工作流，包含所有任务执行状态 |
| **FireTask** | 原子计算任务，可调用shell脚本或Python函数 |
| **FireWork** | 包含JSON规范的作业单元，包含FireTask数组和输入参数 |
| **Workflow** | FireWork集合，定义依赖关系 |
| **Rocket** | 从LaunchPad获取FireWork并执行的组件 |
| **LaunchPad** | MongoDB工作流控制中心 |

### 1.3 基础使用示例
```python
from fireworks import Firework, LaunchPad, ScriptTask
from fireworks.core.rocket_launcher import launch_rocket, rapidfire

# 配置LaunchPad
launchpad = LaunchPad(
    host="myhost", 
    port=27017, 
    name="fireworks_db",
    username="user",
    password="pass"
)

# 创建FireTask和Firework
firetask = ScriptTask.from_str('echo "Job launched!"')
firework = Firework(firetask, name="hello")

# 存储并启动
launchpad.add_wf(firework)
launch_rocket(launchpad)  # 单作业
# rapidfire(launchpad)    # 批量作业
```

### 1.4 SLURM队列适配器配置
```yaml
# my_qadapter.yaml
_fw_name: CommonAdapter
_fw_q_type: SLURM
rocket_launch: rlaunch multi 1
nodes: 1
cpus_per_task: 1
ntasks_per_node: 40
mem_per_cpu: 1000
walltime: '04:00:00'
queue: normal
account: <project_id>
job_name: vasp_job
pre_rocket: |
    module load vasp
    conda activate materials
post_rocket: null
```

---

## 2. Jobflow: 新一代工作流框架

### 2.1 设计目标
Jobflow是专为**高通量计算**设计的Python工作流包，采用**装饰器模式**定义作业。

### 2.2 核心特性
- **@job装饰器**: 将函数转换为延迟执行的Job对象
- **动态工作流**: 运行时确定作业图节点数量
- **多执行后端**: 支持本地执行、FireWorks、jobflow-remote
- **数据库无关**: 支持MongoDB、内存数据库、云存储

### 2.3 Jobflow vs FireWorks对比
| 特性 | FireWorks | Jobflow |
|------|-----------|---------|
| 定义方式 | YAML/JSON | Python装饰器 |
| 动态工作流 | 有限支持 | 完全支持 |
| 执行引擎 | 内置 | 可插拔(FireWorks/jobflow-remote) |
| 学习曲线 | 较陡 | 平缓 |
| 数据库要求 | 必须MongoDB | 内存/MongoDB可选 |

### 2.4 Jobflow基础示例
```python
from jobflow import job, Flow, run_locally

@job
def add(a, b):
    return a + b

@job  
def multiply(a, b):
    return a * b

# 创建作业
job1 = add(1, 2)
job2 = multiply(job1.output, 3)  # 引用前一个作业的输出

# 构建工作流
flow = Flow([job1, job2], output=job2.output)

# 本地执行
response = run_locally(flow)
```

### 2.5 与FireWorks集成
```python
from fireworks import LaunchPad
from jobflow.managers.fireworks import flow_to_workflow
from atomate2.vasp.flows.core import RelaxBandStructureMaker

# 创建atomate2工作流
flow = RelaxBandStructureMaker().make(structure)

# 转换为FireWorks工作流
wf = flow_to_workflow(flow)

# 提交到LaunchPad
lpad = LaunchPad.auto_load()
lpad.add_wf(wf)
```

---

## 3. Custodian: 错误处理与作业管理

### 3.1 核心功能
Custodian是**JIT(Just-In-Time)作业管理框架**，专注于：
- 错误检测与自动恢复
- 长时间运行作业的管理
- VASP/NwChem/QChem/CP2K等计算化学代码的错误处理

### 3.2 VASP错误处理Handler
| Handler | 功能 |
|---------|------|
| **VaspErrorHandler** | 处理常见VASP错误(EDDRMM, ZBRENT等) |
| **UnconvergedErrorHandler** | 检测未收敛计算并调整参数 |
| **WalltimeHandler** | 监控墙时，提前写STOPCAR |
| **IncorrectSmearingHandler** | 金属体系ISMEAR自动调整 |
| **LargeSigmaHandler** | SIGMA过大时自动减小 |
| **ScanMetalHandler** | SCAN泛函金属体系KSPACING调整 |
| **StoppedRunHandler** | 检查点功能，定期保存进度 |

### 3.3 Custodian使用示例
```python
from custodian.custodian import Custodian
from custodian.vasp.handlers import (
    VaspErrorHandler, 
    UnconvergedErrorHandler,
    WalltimeHandler
)
from custodian.vasp.jobs import VaspJob

# 配置错误处理器
handlers = [
    VaspErrorHandler(),
    UnconvergedErrorHandler(),
    WalltimeHandler(wall_time=36000, buffer_time=300)
]

# 双弛豫作业
jobs = VaspJob.double_relaxation_run(
    vasp_cmd=["mpirun", "-n", "64", "vasp_std"],
    auto_npar=True
)

# 运行(最大错误10次)
c = Custodian(handlers, jobs, max_errors=10)
c.run()
```

### 3.4 NEB过渡态作业
```python
from custodian.vasp.jobs import VaspNEBJob

neb_job = VaspNEBJob(
    vasp_cmd=["mpirun", "vasp_std"],
    output_file='neb_vasp.out',
    auto_gamma=True,
    half_kpts=True  # 加速收敛
)
```

---

## 4. 集成部署建议

### 4.1 推荐架构 (2025)
```
Jobflow (工作流定义)
    ↓
flow_to_workwork() (转换)
    ↓
FireWorks (作业调度)
    ↓
SLURM/PBS (集群队列)
    ↓
Custodian (错误处理)
    ↓
VASP/CP2K/QE (计算引擎)
    ↓
MongoDB (数据存储)
```

### 4.2 配置最佳实践
```yaml
# FW_config.yaml
CONFIG_FILE_DIR: /path/to/config
LAUNCHPAD_LOC: my_launchpad.yaml
QUEUEADAPTER_LOC: my_qadapter.yaml

# 性能优化
PING_TIME_SECS: 60
RESERVATION_TIME_SECS: 120
```

### 4.3 已知问题与解决
1. **多作业冲突**: `rlaunch multi`与Custodian的`killall`冲突
   - 解决: 单节点单作业或使用`rlaunch singleshot`
   
2. **重复提交**: rapidfire模式可能重复提交队列
   - 解决: 使用`-m`和`--nlaunches`限制

---

**模块2研究完成**
