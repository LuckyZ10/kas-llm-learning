# KAS (Klaw Agent Studio)

> 专业开发者的 CLI-first Agent 孵化平台
> 核心：代码吞食 → 能力提取 → Agent 进化

## 为什么不是 Dify/Coze？

| 特性 | Dify/Coze | Klaw Agent Studio |
|------|-----------|-------------------|
| 目标用户 | 业务人员 | 专业开发者 |
| 创建方式 | 拖拉拽组装 | 代码吞食提取 |
| 能力来源 | 预设组件 | 你的实战项目 |
| 交互方式 | Web UI | CLI 优先 |

## 核心概念

### 1. Ingestion（吞食）
```bash
kas ingest ./my-project --name "MyAgent"
```

分析你的代码项目，自动提取：
- 编码规范
- 测试模式
- 文档习惯
- 架构偏好

### 2. Fusion（合体）
```bash
kas fuse agent1 agent2 --strategy synthesis --name "SuperAgent"
```

多个 Agent 合并，可能产生涌现能力：
- **Union**: 并集，保留所有能力
- **Intersection**: 交集，找共同特质
- **Dominant**: 以主 Agent 为准
- **Synthesis**: 创造新能力

## 快速开始

### 安装
```bash
pip install klaw-agent-studio
```

### 配置
```bash
kas config set api_key <your-openai-key>
```

### 吞食你的项目
```bash
kas ingest ./my-awesome-project --name "AwesomeAgent"
```

### 与 Agent 对话
```bash
kas chat AwesomeAgent --interactive
```

### 合体 Agent
```bash
kas fuse BackendAgent TestAgent --name "FullStackAgent"
```

## CLI 命令

| 命令 | 描述 |
|------|------|
| `kas ingest <path>` | 吞食项目，提取能力 |
| `kas fuse <agents...>` | 合体多个 Agent |
| `kas chat <agent>` | 与 Agent 对话 |
| `kas evolve <agent>` | Agent 自主进化 |
| `kas list` | 列出所有 Agents |
| `kas inspect <agent>` | 查看 Agent 详情 |
| `kas validate <agent>` | 能力验证测试 |
| `kas stats [agent]` | 使用统计 |
| `kas knowledge` | 知识库管理 |
| `kas abtest` | A/B 测试 |
| `kas versions <agent>` | 版本管理 |
| `kas market publish/search/install` | Agent 市场 |
| `kas equip list/add/remove` | 装备管理 |
| `kas crew create/list/show` | 特种部队管理 |
| `kas crew start/dispatch` | 团队任务执行 |
| `kas workflow run` | 工作流执行 |
| `kas web` | 启动 Web 界面 |
| `kas dashboard` | 启动仪表盘 |

## 示例

### 创建 Code Reviewer Agent
```bash
# 从一个优秀的后端项目提取
git clone https://github.com/example/great-backend-project
cd great-backend-project

kas ingest . --name "CodeReviewer"

# 使用
kas chat CodeReviewer --message "Review this function: def foo(): pass"
```

### 合体两个 Agent
```bash
# 合体后端专家和测试专家
kas fuse BackendAgent TestAgent --strategy synthesis --name "TDDExpert"

# 查看涌现能力
kas inspect TDDExpert
```

### 创建特种部队
```bash
# 创建合同审查团队
kas crew create ContractReviewCrew --members "Alice,Bob,Carol"

# 注入 Agent 到沙盒
kas crew inject ContractReviewCrew LegalExpert

# 启动并执行任务
kas crew start ContractReviewCrew
kas crew dispatch ContractReviewCrew Alice "审查这份合同" --wait
```

### 使用安全沙箱
```python
from kas.core.security import create_secure_sandbox

# 创建安全沙箱 (自动过滤敏感信息、限制资源、网络控制)
sandbox = create_secure_sandbox(
    name="secure-agent",
    work_dir="./workspace",
    preset="strict"  # strict/default/relaxed
)

# 执行代码 (在 Docker 容器中隔离运行)
result = sandbox.execute(["python", "script.py"])
print(result.filtered_output)  # 自动脱敏的输出
```

## 架构

```
kas/
├── core/                    # 核心引擎
│   ├── models.py           # Agent 数据模型
│   ├── ingestion.py        # 吞食引擎
│   ├── fusion.py           # 合体引擎
│   ├── chat.py             # 对话引擎
│   ├── llm_learning.py     # 进化引擎
│   ├── equipment.py        # 装备系统 (MCP/Plugin)
│   ├── multimodal.py       # 多模态处理
│   ├── knowledge.py        # 知识库/RAG
│   ├── market.py           # Agent 市场
│   ├── validation.py       # 能力验证
│   ├── abtest.py           # A/B 测试
│   ├── versioning.py       # 版本管理
│   ├── workflow.py         # 工作流引擎
│   ├── crew_*.py           # 特种部队
│   ├── sandbox/            # OpenClaw 沙盒
│   │   ├── __init__.py     # SoulInjector
│   │   ├── sandbox.py      # OpenClawSandbox
│   │   ├── message_bus.py  # MessageBus
│   │   └── supervisor.py   # SandboxSupervisor
│   ├── security/           # 安全模块
│   │   ├── sensitive_filter.py  # 敏感信息过滤
│   │   ├── resource_quota.py    # 资源配额
│   │   ├── network_controller.py # 网络控制
│   │   ├── docker_sandbox.py    # Docker 隔离
│   │   └── secure_sandbox.py    # 集成安全沙箱
│   └── cluster/            # 分布式集群
│       ├── node.py         # 节点管理
│       ├── manager.py      # 集群管理
│       ├── scheduler.py    # 分布式调度
│       └── state.py        # 状态存储
├── cli/                     # CLI 工具
│   └── main.py             # 命令入口 (30+ 命令)
├── web/                     # Web 界面
│   └── app.py              # Flask 应用
├── dashboard/               # 仪表盘
├── tests/                   # 测试
└── docs/                    # 文档
    └── ARCHITECTURE.md     # 架构文档
```

## 设计原则

1. **简单优先**: 能用规则解决的不用 ML，能用 ML 解决的不用 DRL
2. **CLI First**: 开发者最爱的交互方式
3. **代码即知识**: 从实战代码提取能力，而非预设模板
4. **可组合**: Agent 可以合体，能力可以流通
5. **沙盒化优先**: 多 Agent 协作必须基于沙盒隔离

## 核心特性

| 特性 | 描述 |
|------|------|
| 🔥 **代码吞食** | 从项目代码自动提取 Agent 能力 |
| 🧬 **Agent 合体** | 多 Agent 融合，产生涌现能力 |
| 🎯 **特种部队** | 多 Agent 沙盒化协作 (Crew) |
| 🛡️ **安全沙箱** | 敏感过滤 + 资源限制 + Docker 隔离 |
| 🌐 **分布式集群** | 跨机器 Agent 协作 |
| 🔌 **装备系统** | MCP/Plugin 可扩展工具 |
| 📚 **知识库** | RAG 增强对话 |
| 🧪 **A/B 测试** | Agent 效果对比 |
| 📦 **版本管理** | Agent 版本控制与回滚 |
| 🏪 **Agent 市场** | 发布、搜索、下载 Agent |

## 快速开始

```bash
# 克隆
git clone https://github.com/kas-team/kas.git
cd kas

# 安装依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 本地运行
python -m kas.cli.main --help
```

## 路线图

- [x] 核心 CLI 命令 (ingest, fuse, chat, evolve)
- [x] Agent 市场和云端服务
- [x] 知识库/RAG 支持
- [x] 工作流引擎
- [x] A/B 测试
- [x] 版本管理
- [x] Web 界面
- [x] 装备系统
- [x] 多模态输入 (PDF, 图片, 代码)
- [x] OpenClaw 沙盒化
- [x] Agent 特种部队 (多 Agent 协作)
- [x] 分布式集群
- [x] 安全沙箱强化 (Docker 隔离)
- [ ] 团队协作 (多用户, RBAC) - 可选
- [ ] VS Code 插件
- [ ] MCP Server 接入

## 文档

- [架构文档](docs/ARCHITECTURE.md) - 详细架构设计
- [路线图](ROADMAP.md) - 开发进度和计划

## 开发

```bash
# 克隆
git clone https://github.com/LuckyZ10/kas-llm-learning.git
cd kas-llm-learning/kas

# 安装依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 本地运行
python -m kas.cli.main --help
```

## 依赖

```
click>=8.0.0         # CLI 框架
rich>=13.0.0         # 终端美化
pyyaml>=6.0          # YAML 解析
openai>=1.0.0        # OpenAI API
requests>=2.28.0     # HTTP 客户端
flask>=2.0.0         # Web 界面
chromadb>=0.4.0      # 向量数据库
psutil>=5.9.0        # 资源监控
docker>=6.0.0        # Docker SDK (可选)
```

## License

MIT
