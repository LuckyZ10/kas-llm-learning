# 架构设计文档

## 1. 设计原则

### 1.1 分层架构
平台采用清晰的分层架构，每层职责明确：
- **核心层 (Core)**: 底层计算引擎，与具体科学计算软件交互
- **模拟层 (Simulation)**: 高级模拟方法，扩展核心能力
- **平台层 (Platform)**: 服务和接口，提供用户交互能力
- **智能层 (Intelligence)**: AI/ML能力，增强自动化和智能化
- **应用层 (Workflows)**: 面向具体应用场景的完整工作流

### 1.2 模块化设计
- 每个模块独立可测试
- 模块间通过标准接口通信
- 支持热插拔和扩展

### 1.3 统一数据模型
所有模块使用统一的数据模型：
```python
class MaterialStructure:
    """材料结构统一模型"""
    atoms: Atoms              # ASE Atoms对象
    metadata: Dict            # 结构元数据
    calculations: List        # 关联计算任务
    properties: Dict          # 计算得到的性质
```

## 2. 模块详解

### 2.1 Core Layer

#### core/dft/
DFT计算模块，统一封装VASP、Quantum ESPRESSO等软件。

```
core/dft/
├── vasp/              # VASP接口
├── quantum_espresso/  # QE接口
├── calculators/       # 统一计算器接口
└── parsers/           # 输出解析器
```

#### core/md/
分子动力学模块，主要对接LAMMPS。

```
core/md/
├── lammps/            # LAMMPS接口
├── calculators/       # MD计算器封装
├── analyzers/         # 轨迹分析工具
└── potentials/        # 势函数管理
```

#### core/ml/
机器学习势模块。

```
core/ml/
├── nep/               # NEP势训练与推理
├── deepmd/            # DeepMD-kit接口
├── mace/              # MACE势支持
├── data/              # 数据集管理
└── training/          # 训练工作流
```

### 2.2 Simulation Layer

#### simulation/phase_field/
相场模拟模块，支持：
- Cahn-Hilliard方程
- Allen-Cahn方程
- 电化学相场
- 与DFT/MD耦合

#### simulation/quantum/
量子-经典混合计算：
- VQE求解器
- 量子机器学习势
- 量子动力学耦合

#### simulation/rl/
强化学习优化器：
- 材料成分优化
- 晶体结构搜索
- 多目标优化

### 2.3 Platform Layer

#### platform/api/
REST API平台：
- 任务管理API
- 数据查询API
- WebSocket实时通信
- 认证与授权

#### platform/web/
Web用户界面：
- 工作流编辑器
- 3D结构可视化
- 实时监控仪表板
- 报告生成

#### platform/hpc/
HPC集群连接器：
- SLURM/PBS/LSF/SGE支持
- 自动数据同步
- 故障恢复
- Checkpoint续算

### 2.4 Intelligence Layer

#### intelligence/active_learning/
主动学习系统：
- 不确定性量化
- 采样策略
- 模型选择

#### intelligence/literature/
文献智能分析：
- arXiv/PubMed抓取
- 主题建模 (BERTopic)
- 知识图谱构建
- 自动综述生成

#### intelligence/multi_agent/
多智能体协同：
- 理论家Agent
- 实验员Agent
- 审稿人Agent
- 协调员Agent

### 2.5 Workflows Layer

每个应用工作流包含：
```
workflows/<material_type>/
├── config/            # 配置文件
├── pipelines/         # 工作流管道
├── models/            # 数据模型
├── tasks/             # 具体任务
└── examples/          # 使用示例
```

## 3. 数据流

### 3.1 典型工作流数据流

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ 结构生成    │───▶│ DFT计算     │───▶│ 数据收集    │
│Structure Gen│    │   (VASP)    │    │ Data Collect│
└─────────────┘    └─────────────┘    └──────┬──────┘
                                              │
┌─────────────┐    ┌─────────────┐    ┌──────▼──────┐
│ MD模拟      │◀───│ ML势训练    │◀───│ 主动学习    │
│ MD Sim      │    │ML Potential │    │Active Learn │
└──────┬──────┘    └─────────────┘    └─────────────┘
       │
┌──────▼──────┐    ┌─────────────┐    ┌─────────────┐
│ 性质分析    │───▶│ 结果存储    │───▶│ 可视化      │
│  Analysis   │    │   Storage   │    │Visualization│
└─────────────┘    └─────────────┘    └─────────────┘
```

### 3.2 事件驱动架构

平台采用事件驱动架构，主要事件类型：
- `calculation.started`: 计算任务开始
- `calculation.completed`: 计算任务完成
- `calculation.failed`: 计算任务失败
- `data.updated`: 数据更新
- `model.trained`: 模型训练完成

## 4. 接口设计

### 4.1 核心接口

```python
# DFT计算器接口
class DFTCalculator(ABC):
    @abstractmethod
    def calculate(self, atoms: Atoms) -> CalculationResult:
        pass

# MD模拟器接口
class MDSimulator(ABC):
    @abstractmethod
    def run_md(self, atoms: Atoms, steps: int) -> Trajectory:
        pass

# ML势接口
class MLPotential(ABC):
    @abstractmethod
    def predict(self, atoms: Atoms) -> Tuple[np.ndarray, np.ndarray]:
        """预测能量和力"""
        pass
```

### 4.2 API端点

```
/api/v1/
├── /structures          # 结构管理
├── /calculations        # 计算任务
├── /workflows           # 工作流
├── /screening           # 高通量筛选
├── /models              # ML模型管理
├── /datasets            # 数据集
└── /monitoring          # 监控数据
```

## 5. 扩展性设计

### 5.1 添加新的DFT代码支持

1. 在 `core/dft/` 下创建新目录
2. 实现 `DFTCalculator` 接口
3. 添加配置解析器
4. 注册到计算器工厂

### 5.2 添加新的应用场景

1. 在 `workflows/` 下创建新目录
2. 定义工作流管道
3. 实现具体任务
4. 添加示例和文档

### 5.3 插件系统

```python
# 插件注册示例
from core.plugins import register_plugin

@register_plugin(name="custom_potential", type="potential")
class CustomPotential:
    def __init__(self, config):
        self.config = config
    
    def calculate(self, atoms):
        # 实现计算逻辑
        pass
```

## 6. 性能考虑

### 6.1 并行计算
- DFT计算：MPI并行
- MD模拟：OpenMP/GPU加速
- 高通量筛选：任务级并行

### 6.2 数据缓存
- 计算结果缓存
- 结构数据缓存
- 模型权重缓存

### 6.3 异步处理
- 使用Celery处理长时间任务
- WebSocket推送实时状态
- 数据库异步操作

## 7. 安全设计

### 7.1 认证授权
- OAuth2认证
- API Key管理
- 角色权限控制

### 7.2 数据安全
- 传输加密 (TLS)
- 存储加密
- 访问日志

## 8. 部署架构

### 8.1 单节点部署

```
┌─────────────────────────────────────┐
│           单节点部署                │
│  ┌─────────┐  ┌─────────┐          │
│  │ Web UI  │  │ API     │          │
│  └────┬────┘  └────┬────┘          │
│       └─────────────┘               │
│              │                      │
│       ┌──────▼──────┐               │
│       │   Redis     │               │
│       └──────┬──────┘               │
│              │                      │
│       ┌──────▼──────┐               │
│       │   Celery    │               │
│       │   Workers   │               │
│       └─────────────┘               │
└─────────────────────────────────────┘
```

### 8.2 分布式部署

```
┌─────────────────────────────────────────────┐
│              负载均衡器 (Nginx)              │
└──────────────┬──────────────────┬───────────┘
               │                  │
    ┌──────────▼────────┐  ┌──────▼──────┐
    │    API 服务器 1   │  │ API 服务器 2 │
    └──────────┬────────┘  └──────┬──────┘
               │                  │
    ┌──────────▼──────────────────▼──────┐
    │        Redis 集群                  │
    └──────────┬──────────────────┬──────┘
               │                  │
    ┌──────────▼────────┐  ┌──────▼──────┐
    │   Celery Worker   │  │Celery Worker│
    └───────────────────┘  └─────────────┘
```

## 9. 未来演进

### 9.1 短期 (3-6个月)
- 完善测试覆盖率
- 优化Web界面性能
- 添加更多示例

### 9.2 中期 (6-12个月)
- 量子计算集成
- 数字孪生平台
- 移动端支持

### 9.3 长期 (12个月+)
- 全自动科研Agent
- 跨领域知识迁移
- 社区生态建设
