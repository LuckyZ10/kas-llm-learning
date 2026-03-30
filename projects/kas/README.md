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
| `kas list` | 列出所有 Agents |
| `kas inspect <agent>` | 查看 Agent 详情 |

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

## 架构

```
kas/
├── core/                    # 核心引擎
│   ├── models.py           # Agent 数据模型
│   ├── ingestion.py        # 吞食引擎
│   ├── fusion.py           # 合体引擎
│   └── chat.py             # 对话引擎
├── cli/                     # CLI 工具
│   └── main.py             # 命令入口
└── examples/                # 示例
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

- [x] 核心 CLI 命令 (ingest, fuse, chat)
- [ ] Agent Registry (市场)
- [ ] 更多 LLM 支持
- [ ] VS Code 插件
- [ ] Web UI (轻量级)

## License

MIT
