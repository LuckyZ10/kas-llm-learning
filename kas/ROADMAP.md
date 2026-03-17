# KAS (Klaw Agent Studio) 完整路线图

> 版本: v1.1
> 最后更新: 2026-03-18
> 作者: Yilin.zhang

---

## 🎯 项目愿景

**KAS = 代码吸血鬼 + Agent 孵化器 + Agent 特种部队指挥系统**

终极目标: 从代码项目孵化 Agent，让多个 Agent 组成团队，装备各种工具，协作解决复杂问题。

---

## 📅 阶段规划

### Phase 1: 基础功能 ✅ (已完成)
**时间**: 2026-03-17 完成
**核心**: Agent 的创建、使用、进化

| 功能 | 状态 | 文件 |
|------|------|------|
| 代码吞食 | ✅ | `core/ingestion.py` |
| Agent 合体 | ✅ | `core/fusion.py` |
| 对话引擎 | ✅ | `core/chat.py` |
| LLM 增强学习 | ✅ | `core/llm_learning.py` |
| 配置管理 | ✅ | `core/config.py` |
| 版本管理 | ✅ | `core/versioning.py` |
| CLI 命令 | ✅ | `cli/main.py` |

---

### Phase 2: 生态建设 ✅ (已完成)
**时间**: 2026-03-17 完成
**核心**: Agent 市场和能力验证

| 功能 | 状态 | 文件 |
|------|------|------|
| Agent 市场 | ✅ | `core/market.py`, `core/cloud_market.py` |
| 能力验证 | ✅ | `core/validation.py` |
| 统计面板 | ✅ | `core/stats.py`, `dashboard/` |

---

### Phase 3: 智能化升级 ✅ (已完成)
**时间**: 2026-03-18 完成
**核心**: 知识库、工作流、A/B测试

| 功能 | 状态 | 文件 |
|------|------|------|
| 知识库/RAG | ✅ | `core/knowledge.py`, `core/rag_chat.py` |
| Agent 工作流 | ✅ | `core/workflow.py` |
| A/B 测试 | ✅ | `core/abtest.py` |

---

### Phase 4: Agent 特种部队 (进行中 - 重新设计)
**目标**: 从单个 Agent 升级为协作团队

#### 4.1 网页界面 MVP ✅
**状态**: 已完成
**文件**: `kas/web/app.py`, `kas/web/static/index.html`

**功能**:
- FastAPI 后端 + 原生 JS 前端
- Agent 列表/详情/对话
- 市场搜索/安装
- 统计数据可视化

**CLI**:
```bash
kas web [--host] [--port]  # 启动 Web 界面
```

---

#### 4.2 Agent 装备系统 🛠️ ⏳
**优先级**: 🔴 P0
**依赖**: Web 界面 ✅
**工作量**: 1 周

**核心概念**: 每个 Agent 可以装备各种"武器"

```yaml
# agent.yaml 扩展
name: "ResearchAgent"
equipment:
  - type: mcp
    name: "web_search"
    config: { engine: "duckduckgo" }
  - type: mcp
    name: "ocr"
    config: { language: "zh+en" }
  - type: plugin
    name: "pdf_parser"
  - type: plugin
    name: "code_executor"
    config: { sandbox: "docker" }
```

**内置装备清单**:
| 装备 | 功能 | 场景 |
|------|------|------|
| `web_search` | 联网搜索 | 获取最新信息 |
| `ocr` | 图片文字识别 | 处理扫描件 |
| `pdf_parser` | PDF 解析 | 文档分析 |
| `code_executor` | 代码执行(沙箱) | 验证代码 |
| `file_reader` | 文件读取 | 处理各种格式 |
| `image_analysis` | 图片分析 | 理解图像内容 |

**CLI**:
```bash
kas equip list                           # 列出可用装备
kas equip add MyAgent web_search         # 给 Agent 添加装备
kas equip remove MyAgent ocr             # 移除装备
kas equip show MyAgent                   # 查看已装备
```

**技术方案**:
```python
# core/equipment.py
class Equipment(ABC):
    @abstractmethod
    def use(self, params: dict) -> Any:
        pass

class MCPEquipment(Equipment):
    """MCP 协议装备"""
    def use(self, params):
        # 调用 MCP Server
        pass

class PluginEquipment(Equipment):
    """内置插件装备"""
    def use(self, params):
        # 调用本地插件
        pass

# Agent 使用装备
class Agent:
    def use_tool(self, name: str, params: dict):
        equip = self.get_equipment(name)
        return equip.use(params)
```

---

#### 4.3 多模态输入 📁 ⏳
**优先级**: 🔴 P0
**依赖**: Web 界面 ✅
**工作量**: 3 天

**功能**:
- 文件上传 (PDF, 图片, 代码文件)
- 图片 OCR 预处理
- PDF 文本提取
- 附件传递给 Agent

**CLI**:
```bash
kas chat MyAgent "分析这份合同" --attach contract.pdf --attach photo.jpg
```

**Web**:
```
用户上传文件 → 预处理器 → Agent 获取内容
              ↓
         PDF → pdf_parser
         JPG → ocr / image_analysis
         PY  → file_reader
```

---

#### 4.4 Agent 团队 (Crew) 👥 ⏳
**优先级**: 🔴 P0
**依赖**: 装备系统, 多模态
**工作量**: 1 周

**核心概念**: 多个 Agent 组成团队，分工协作

```yaml
# crew.yaml
name: "ContractReviewCrew"
description: "合同审查团队"

members:
  - name: "Alice"
    role: "coordinator"      # 协调员角色
    description: "法律背景，协调沟通"
    equipment: ["web_search", "file_reader"]
    
  - name: "Bob"
    role: "ocr_expert"
    description: "OCR 专家，处理扫描件"
    equipment: ["ocr", "image_analysis"]
    
  - name: "Carol"
    role: "legal_analyst"
    description: "法律分析师，提取关键信息"
    equipment: ["pdf_parser", "web_search"]
    
  - name: "David"
    role: "writer"
    description: "报告撰写员"
    equipment: ["file_reader"]

workflow:
  - step: 1
    agent: "Alice"
    task: "理解用户需求，确认文档类型"
    
  - step: 2
    agent: "Bob"
    task: "对图片/扫描件进行 OCR"
    condition: "if input.has_image"
    
  - step: 3
    agent: "Carol"
    task: "分析文档内容，提取关键信息"
    depends_on: [2]
    
  - step: 4
    agent: "David"
    task: "撰写分析报告"
    depends_on: [3]
    
  - step: 5
    agent: "Alice"
    task: "审核报告，向用户展示结果并确认"
    depends_on: [4]
```

**CLI**:
```bash
kas crew create ReviewTeam                    # 创建团队
kas crew add ReviewTeam MyAgent --role analyst # 添加成员
kas crew run ReviewTeam "分析合同" \
  --attach contract.pdf \
  --attach photo.jpg

kas crew chat ReviewTeam                      # 交互式团队对话
```

---

#### 4.5 协调员模式 (动态选举) 🎭 ⏳
**优先级**: 🔴 P0
**依赖**: Agent 团队
**工作量**: 1 周

**核心概念**: 协调员不是固定的，根据任务动态选举谁对外沟通

**交互示例**:
```
用户: 分析这份合同的风险点
附件: contract.pdf, photo1.jpg

[团队内部讨论]

Alice: "我来确认需求"
       → 临时协调员
       "我看到你上传了合同 PDF 和一张照片。
        让 [Bob] 处理图片，[Carol] 分析条款。
        需要我这样做吗？"

用户: 是的

[Bob 处理图片...]
Bob: "图片识别完成，这是签署页"
    → Bob 汇报进度

[Carol 分析条款...]
Carol: "第5条违约责任表述模糊，不确定具体含义"
       → Carol 成为临时协调员，询问用户
       "你能解释一下这条的意图吗？"

用户: 这是想表达违约方需赔偿...

Carol: [把解释同步给团队]

[David 撰写报告...]
David: "报告完成，关键风险点：1. XX 2. YY"
       → David 成为临时协调员
       "需要修改哪里吗？"

用户: 没问题

David: "最终报告已生成"
```

**选举逻辑**:
| 阶段 | 谁出面 | 原因 |
|------|--------|------|
| 需求确认 | 最理解业务的 Agent | 快速理解意图 |
| 执行阶段 | 当前负责的 Agent | 直接汇报进度 |
| 有问题时 | 遇到问题的 Agent | 第一时间澄清 |
| 结果汇总 | 负责交付的 Agent | 完整呈现结果 |

**技术方案**:
```python
# core/crew.py
class Crew:
    def elect_coordinator(self, context: dict) -> Agent:
        """根据上下文选举临时协调员"""
        # 策略1: 谁最相关选谁
        # 策略2: 轮流制
        # 策略3: 指定优先级
        pass
    
    def execute_with_coordination(self, task: str, attachments: list):
        """带协调的团队执行"""
        # 1. 选举初始协调员
        coordinator = self.elect_coordinator({'stage': 'init'})
        
        # 2. 协调员确认需求
        confirmed = coordinator.confirm_with_user(task)
        
        # 3. 分发任务给各成员
        for member in self.members:
            if member != coordinator:
                result = member.execute(confirmed)
                
                # 4. 需要询问时，该成员成为临时协调员
                if result.needs_clarification:
                    temp_coord = member
                    clarification = temp_coord.ask_user(result.question)
                    result.update(clarification)
        
        # 5. 最终汇总
        final_coord = self.elect_coordinator({'stage': 'final'})
        return final_coord.summarize_results()
```

---

### Phase 5: 企业级功能 (长期 - 2-3 月) ⏳
**目标**: 生产环境可用

#### 5.1 安全沙箱 🔒
- Docker 隔离代码执行
- 网络访问限制
- 敏感信息过滤
- 资源配额限制

#### 5.2 性能监控 📈
- 响应时间监控
- Token 消耗统计
- 错误率追踪
- 告警机制

#### 5.3 团队协作 👥
- 多用户支持
- 权限管理
- Agent 共享
- 团队知识库

---

## 🗺️ 新架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        KAS Core                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Agent      │  │   Agent      │  │   Agent      │     │
│  │  (Alice)     │  │   (Bob)      │  │  (Carol)     │     │
│  │              │  │              │  │              │     │
│  │ Equipment:   │  │ Equipment:   │  │ Equipment:   │     │
│  │ - web_search │  │ - ocr        │  │ - pdf_parser │     │
│  │ - file_reader│  │ - image      │  │ - web_search │     │
│  │              │  │              │  │              │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                 │              │
│         └─────────────────┼─────────────────┘              │
│                           │                                │
│                    ┌──────┴──────┐                        │
│                    │   Crew      │                        │
│                    │   团队      │                        │
│                    │             │                        │
│                    │ 动态选举    │                        │
│                    │ 协调员      │                        │
│                    └──────┬──────┘                        │
│                           │                                │
│         ┌─────────────────┼─────────────────┐             │
│         │                 │                 │              │
│    ┌────┴────┐      ┌────┴────┐      ┌────┴────┐         │
│    │   MCP   │      │ Plugin  │      │ Multimodal│        │
│    │ Server  │      │ System  │      │  Input    │        │
│    │         │      │         │      │           │        │
│    │ - web   │      │ - pdf   │      │ - PDF     │        │
│    │ - search│      │ - parser│      │ - Image   │        │
│    │ - ocr   │      │ - code  │      │ - File    │        │
│    └─────────┘      └─────────┘      └───────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │      User       │
                    │   (动态协调)    │
                    └─────────────────┘
```

---

## 📊 实施计划

### 当前状态
| 阶段 | 完成度 | 核心交付 |
|------|--------|----------|
| Phase 1-3 | ✅ 100% | 25个CLI命令，8868行代码 |
| Phase 4.1 | ✅ 100% | Web界面MVP |
| Phase 4.2-4.5 | ⏳ 0% | Agent特种部队 |

### 下一步 (Phase 4.2 开始)
| 周 | 任务 | 产出 |
|----|------|------|
| W1 | 装备系统 | MCP适配器 + 6个内置装备 |
| W2 | 多模态输入 | 文件上传 + 预处理 |
| W3 | Agent团队 | Crew定义 + 工作流编排 |
| W4 | 协调员模式 | 动态选举 + 用户确认流程 |

---

## 💡 关键设计决策

1. **轻量 Web**: 原生 JS 够用，不上 React
2. **MCP 优先**: 装备系统基于 MCP 协议，兼容生态
3. **动态协调**: 谁负责谁沟通，不固定协调员
4. **多模态**: 文件上传 + 预处理器链
5. **团队协作**: 声明式 Crew YAML + 运行时选举

---

**已推送**: `e62d6f7` feat: Web Market API + Phase 4 设计更新
