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

## 架构

```
kas/
├── core/                    # 核心引擎
│   ├── models.py           # Agent 数据模型
│   ├── ingestion.py        # 吞食引擎
│   ├── fusion.py           # 合体引擎
│   ├── chat.py             # 对话引擎
│   ├── equipment.py        # 装备系统
│   ├── multimodal.py       # 多模态处理
│   ├── knowledge.py        # 知识库/RAG
│   ├── market.py           # Agent 市场
│   ├── validation.py       # 能力验证
│   ├── abtest.py           # A/B 测试
│   ├── versioning.py       # 版本管理
│   ├── workflow.py         # 工作流引擎
│   ├── crew_*.py           # 特种部队
│   ├── sandbox/            # OpenClaw 沙盒
│   └── cluster/            # 分布式集群
├── cli/                     # CLI 工具
│   └── main.py             # 命令入口
├── web/                     # Web 界面
│   └── app.py              # Flask 应用
├── dashboard/               # 仪表盘
└── tests/                   # 测试
```

## 设计原则

1. **简单优先**: 能用规则解决的不用ML，能用ML解决的不用DRL
2. **CLI First**: 开发者最爱的交互方式
3. **代码即知识**: 从实战代码提取能力，而非预设模板
4. **可组合**: Agent可以合体，能力可以流通

## 开发

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
- [ ] 安全沙箱强化 (Docker 隔离)
- [ ] 团队协作 (多用户, RBAC)
- [ ] VS Code 插件
- [ ] MCP Server 接入

## License

MIT
