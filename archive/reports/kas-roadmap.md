# Kimi Agent Studio - 重构与开源路线图

> 目标：面向专业开发者的 CLI-first Agent 孵化平台
> 核心：代码吞食 → 能力提取 → Agent 进化

---

## Phase 1: 代码整理（1-2 天）

### 1.1 目录重组

```
kimi-agent-studio/
├── README.md
├── LICENSE
├── package.json                 # root monorepo
├── pnpm-workspace.yaml
├── turbo.json
│
├── packages/
│   ├── core/                    # ⭐ 核心引擎（Python）
│   │   ├── ingestion/          # 吞食引擎
│   │   ├── fusion/             # 合体引擎
│   │   ├── capabilities/       # 能力系统
│   │   ├── llm_adapter/        # LLM 适配器
│   │   └── brainstorm/         # 头脑风暴
│   │
│   ├── cli/                     # ⭐ CLI 工具（Python）
│   │   ├── cmd/
│   │   │   ├── ingest.py       # `kas ingest`
│   │   │   ├── fuse.py         # `kas fuse`
│   │   │   ├── publish.py      # `kas publish`
│   │   │   └── registry.py     # `kas search/install`
│   │   └── main.py             # entry point
│   │
│   ├── web/                     # Web UI（可选，延后）
│   │   └── ...
│   │
│   └── registry-server/         # Agent 市场服务端
│       └── ...
│
├── examples/                    # 示例 Agent
│   ├── code-reviewer/          # 从代码提取的 Reviewer Agent
│   ├── test-writer/            # 从测试文件提取的 Agent
│   └── doc-generator/          # 从文档提取的 Agent
│
└── docs/
    ├── ingestion-guide.md      # 如何写好可被吞食的项目
    ├── fusion-patterns.md      # 合体模式最佳实践
    └── architecture.md         # 架构设计
```

### 1.2 具体移动命令

```bash
# 1. 创建新结构
mkdir -p packages/core packages/cli packages/web packages/registry-server
mkdir -p examples docs

# 2. 移动后端核心
mv backend/* packages/core/
rm -rf backend/

# 3. 移动前端
mv src/* packages/web/src/ 2>/dev/null || true
mv packages/backend/src/* packages/core/ 2>/dev/null || true

# 4. 清理重复文件
rm -rf packages/web/backend  # 如果存在
```

### 1.3 依赖整理

**packages/core/requirements.txt:**
```
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
openai>=1.0.0
httpx>=0.24.0
pyyaml>=6.0
python-dotenv>=1.0.0
```

**packages/cli/requirements.txt:**
```
click>=8.0.0
rich>=13.0.0          # 漂亮的 CLI 输出
requests>=2.31.0
pyyaml>=6.0
# 本地依赖
-e ../core
```

**packages/cli/setup.py:**
```python
from setuptools import setup, find_packages

setup(
    name="kas",                    # Kimi Agent Studio CLI
    version="0.1.0",
    packages=find_packages(),
    install_requires=[...],
    entry_points={
        'console_scripts': [
            'kas=kas.cli:main',
        ],
    },
)
```

---

## Phase 2: CLI 核心命令（3-5 天）

### 2.1 命令设计

```bash
# 吞食项目
kas ingest <path> [--name <name>] [--output <dir>]

# 查看已提取的能力
kas inspect <agent-path>

# 合体 Agent
kas fuse <agent1> <agent2> [--strategy union|intersect|dominant|synthesis] --name <new-name>

# 与 Agent 对话（快速测试）
kas chat <agent-path> [--interactive]

# 发布到市场（本地或远程）
kas publish <agent-path> [--registry <url>] [--public|--private]

# 发现 Agent
kas search <query> [--registry <url>]
kas install <agent-id> [--registry <url>]

# 配置
kas config set api_key <key>
kas config set default_model <model>
```

### 2.2 CLI 交互示例

**吞食项目：**
```bash
$ kas ingest ./my-backend-project --name "BackendHelper"

🔍 Analyzing project structure...
   Found: 15 Python files, 3 config files, 12 test files

🧠 Extracting capabilities...
   ✓ Code review capability
   ✓ Test generation capability  
   ✓ API documentation capability
   ✓ Error handling patterns

📦 Creating Agent: BackendHelper
   Location: ~/.kas/agents/backend-helper/
   Config: agent.yaml
   Prompt: system_prompt.txt
   Tools: tools/

🎉 Agent "BackendHelper" is ready!
   Try: kas chat backend-helper
```

**合体 Agent：**
```bash
$ kas fuse backend-helper test-expert --strategy synthesis --name "FullStackDev"

🔀 Fusing agents...
   Agent 1: BackendHelper (4 capabilities)
   Agent 2: TestExpert (3 capabilities)
   Strategy: synthesis

✨ Emergent capabilities detected:
   + TDD Workflow (new!)
   + Test-Driven Development guide

📦 Created: FullStackDev
   Total capabilities: 9 (7 inherited + 2 emergent)
```

### 2.3 核心代码结构

**packages/cli/kas/cli.py:**
```python
import click
from rich.console import Console
from rich.table import Table

console = Console()

@click.group()
def cli():
    """Kimi Agent Studio - 专业开发者 Agent 孵化平台"""
    pass

@cli.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--name', '-n', help='Agent 名称')
@click.option('--output', '-o', default='./.kas/agents', help='输出目录')
@click.option('--model', default='deepseek-chat', help='使用的 LLM 模型')
def ingest(path, name, output, model):
    """吞食项目，提取能力，孵化 Agent"""
    from kas.commands.ingest import ingest_project
    
    console.print(f"🔍 正在分析项目: {path}")
    
    result = ingest_project(path, name=name, model=model)
    
    # 漂亮的输出表格
    table = Table(title="提取的能力")
    table.add_column("能力", style="cyan")
    table.add_column("类型", style="green")
    table.add_column("置信度", style="yellow")
    
    for cap in result['capabilities']:
        table.add_row(cap['name'], cap['type'], f"{cap['confidence']}%")
    
    console.print(table)
    console.print(f"\n✅ Agent 已保存到: {result['output_path']}")

@cli.command()
@click.argument('agents', nargs=-1, required=True)
@click.option('--strategy', '-s', 
              type=click.Choice(['union', 'intersect', 'dominant', 'synthesis']),
              default='union',
              help='合体策略')
@click.option('--name', '-n', required=True, help='新 Agent 名称')
def fuse(agents, strategy, name):
    """合体多个 Agent"""
    from kas.commands.fuse import fuse_agents
    
    console.print(f"🔀 使用 [{strategy}] 策略合体 {len(agents)} 个 Agent...")
    
    result = fuse_agents(list(agents), strategy, name)
    
    if result.get('emergent_capabilities'):
        console.print("\n✨ 发现涌现能力:")
        for cap in result['emergent_capabilities']:
            console.print(f"   + {cap['name']}: {cap['description']}")

@cli.command()
@click.argument('agent_path')
@click.option('--interactive', '-i', is_flag=True, help='交互模式')
def chat(agent_path, interactive):
    """与 Agent 对话"""
    from kas.commands.chat import chat_with_agent
    
    if interactive:
        console.print(f"🤖 正在启动 Agent: {agent_path}")
        console.print("输入 'exit' 退出对话\n")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            response = chat_with_agent(agent_path, user_input)
            console.print(f"Agent: {response}\n")
    else:
        # 单次对话模式
        pass

if __name__ == '__main__':
    cli()
```

---

## Phase 3: 文档与开源准备（2-3 天）

### 3.1 README.md 结构

```markdown
# Kimi Agent Studio (kas)

> 专业开发者的 Agent 孵化平台
> 不要从零创建 Agent，直接吞掉你已有的能力

## 为什么不是 Dify/Coze？

| 特性 | Dify/Coze | Kimi Agent Studio |
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
- Union: 并集，保留所有能力
- Intersection: 交集，找共同特质
- Dominant: 以主 Agent 为准
- Synthesis: 创造新能力

## 快速开始

```bash
# 安装
pip install kas

# 配置 API Key
kas config set api_key <your-deepseek-key>

# 吞食你的项目
kas ingest ./my-awesome-project --name "AwesomeAgent"

# 开始对话
kas chat AwesomeAgent
```

## 架构

[链接到 ARCHITECTURE.md]

## 贡献

[Contributing Guide]

## License

MIT
```

### 3.2 必须写的文档

**docs/ingestion-guide.md:**
- 什么样的项目适合被吞食
- 如何提高提取质量
- 支持的文件类型

**docs/fusion-patterns.md:**
- 四种合体策略的适用场景
- 涌现能力的例子
- 最佳实践

**ARCHITECTURE.md:**
- 系统架构图
- 核心模块说明
- 数据流

### 3.3 开源 Checklist

- [ ] 选择 License（推荐 MIT）
- [ ] 添加 LICENSE 文件
- [ ] 添加 CONTRIBUTING.md
- [ ] 添加 CODE_OF_CONDUCT.md
- [ ] 清理代码中的敏感信息（API key、内网地址）
- [ ] 添加 GitHub Actions（CI/CD）
- [ ] 创建示例项目（examples/）

---

## Phase 4: 发布与迭代（持续）

### 4.1 首次发布

**版本号：** v0.1.0

**发布渠道：**
1. GitHub Release
2. PyPI: `pip install kas`
3. Homebrew（后期）

**发布内容：**
- 核心功能：ingest、fuse、chat
- 支持模型：DeepSeek、OpenAI 兼容接口
- 文档：Quick Start、Architecture、API Reference

### 4.2 社区运营

**技术博客：**
- 《不要用拖拉拽创建 Agent，直接吞掉你的代码》
- 《Agent 合体：当两个开发者 Agent 碰撞会产生什么？》
- 《从 Cursor rules 到 Agent：我的开发工作流进化》

**示例 Agent 仓库：**
- `kas-agent-examples` - 社区共享的 Agent
- 每个示例包含：源码项目 + 生成的 Agent

### 4.3 后续迭代

**v0.2.0:**
- [ ] 支持更多 ingestion 来源（GitHub URL、Cursor rules、Claude Code 历史）
- [ ] Agent Registry（在线市场）
- [ ] 能力评分系统

**v0.3.0:**
- [ ] Web UI（轻量级，仅用于展示）
- [ ] 团队协作功能
- [ ] VS Code 插件

---

## 执行 Checklist

### 本周必须完成

- [ ] 代码整理（Phase 1）
- [ ] CLI 核心命令框架（Phase 2.1）
- [ ] `kas ingest` 可用
- [ ] README 初稿

### 下周完成

- [ ] `kas fuse` 可用
- [ ] `kas chat` 可用
- [ ] 文档完善
- [ ] GitHub 仓库公开

### 下下周

- [ ] PyPI 发布
- [ ] 技术博客发布
- [ ] 社区推广

---

## 关键决策点

### 决策 1: 是否保留 Web UI？
**建议：** 保留但不作为重点
- CLI 是核心，Web UI 只是锦上添花
- 可以先注释掉 Web 相关代码，专注 CLI
- 后期再决定是否恢复

### 决策 2: 支持哪些 LLM？
**建议：** 优先 DeepSeek，保持 OpenAI 兼容
- 国内开发者易获取
- 成本低
- 通过 Adapter 模式易于扩展

### 决策 3: Agent 存储格式？
**建议：** 本地文件系统 + 可选远程 Registry
```
~/.kas/
├── agents/
│   ├── backend-helper/
│   │   ├── agent.yaml          # 配置
│   │   ├── system_prompt.txt   # 系统提示词
│   │   ├── capabilities/       # 能力定义
│   │   └── memory/             # 记忆（可选）
│   └── test-expert/
│       └── ...
└── config.yaml                 # 全局配置
```

---

## 总结

**核心策略：**
1. **CLI First** - 开发者最爱
2. **Ingestion Core** - 差异化卖点
3. **Open Source** - 建立生态
4. **Lean** - 砍掉一切非核心

**成功标准：**
- 开发者能用一条命令吞食项目
- 能在 5 分钟内创建一个可用的 Agent
- 有 3 个以上的真实用户案例

---

*最后更新：2026-03-16*
*下一步：开始 Phase 1 代码整理*
