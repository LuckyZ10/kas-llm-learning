# KAS Phase 4 完整设计总结

## 🎯 核心愿景

**KAS = OpenClaw 特种部队指挥系统**

从"单个 Agent 工具"升级为"多个 Agent 协作团队"，每个 Agent 都是注入灵魂的 OpenClaw 实例，装备共享武器，拥有分层记忆，动态选举协调员，共同解决复杂问题。

---

## 🏗️ 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         KAS Phase 4                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                        Crew (团队)                           │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │   │
│  │  │  Alice  │  │   Bob   │  │  Carol  │  │  David  │        │   │
│  │  │协调员   │  │OCR专家  │  │法律分析 │  │撰写员   │        │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │   │
│  │       │            │            │            │              │   │
│  │       └────────────┴────────────┴────────────┘              │   │
│  │                    │                                         │   │
│  │                    ▼                                         │   │
│  │           ┌─────────────────┐                               │   │
│  │           │  Shared Memory  │  ← Crew 共同记忆               │   │
│  │           │  对话历史        │                               │   │
│  │           │  任务上下文      │                               │   │
│  │           │  共享知识库      │                               │   │
│  │           │  中间结果        │                               │   │
│  │           └────────┬────────┘                               │   │
│  │                    │                                         │   │
│  │       ┌────────────┼────────────┐                          │   │
│  │       ▼            ▼            ▼                          │   │
│  │  ┌────────┐   ┌────────┐   ┌────────┐                     │   │
│  │  │Alice   │   │ Bob    │   │ Carol  │                     │   │
│  │  │Memory  │   │ Memory │   │ Memory │  ← 个人记忆         │   │
│  │  │-相关提取│   │-相关提取│   │-相关提取│                     │   │
│  │  │-个人偏好│   │-OCR专长│   │-法律专长│                     │   │
│  │  └────────┘   └────────┘   └────────┘                     │   │
│  │                                                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Shared Equipment (共享装备池)              │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │   │
│  │  │web_search│  │   ocr    │  │pdf_parser│  │code_exec │    │   │
│  │  │ (MCP)    │  │  (MCP)   │  │ (Plugin) │  │ (Plugin) │    │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │   │
│  │                                                             │   │
│  │  任何 Agent 都可以调用，协调员根据任务分配合适的组合          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      Dynamic Coordinator                      │   │
│  │                     (动态协调员选举)                          │   │
│  │                                                             │   │
│  │  需求确认 → Alice 问用户                                    │   │
│  │  执行阶段 → Bob 汇报进度                                    │   │
│  │  有问题   → Carol 直接问                                    │   │
│  │  结果汇总 → David 展示                                      │   │
│  │                                                             │   │
│  │  谁负责当前环节，谁来沟通                                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │      User       │
                    │   (交互入口)    │
                    └─────────────────┘
```

---

## 📦 四大子系统

### 1. 装备系统 (Equipment System)

#### 核心概念
- **共享装备池**: Crew 级别的共享资源，所有 Agent 可用
- **MCP 协议**: 标准化外部工具接入
- **Plugin 系统**: 内置工具实现

#### 装备清单
| 装备 | 类型 | 功能 | 场景 |
|------|------|------|------|
| `web_search` | MCP | 联网搜索 | 获取最新信息 |
| `ocr` | MCP | 图片文字识别 | 处理扫描件 |
| `pdf_parser` | Plugin | PDF 解析 | 文档分析 |
| `code_executor` | Plugin | 代码执行 | 验证代码(沙箱) |
| `file_reader` | Plugin | 文件读取 | 处理各种格式 |
| `image_analysis` | MCP | 图片分析 | 理解图像内容 |

#### 配置方式
```yaml
# crew.yaml
name: "ContractReviewCrew"

# 共享装备池
shared_equipment:
  - name: "web_search"
    type: mcp
    config: { engine: "duckduckgo" }
    
  - name: "ocr"
    type: mcp
    config: { language: "zh+en" }
    
  - name: "pdf_parser"
    type: plugin
    
  - name: "code_executor"
    type: plugin
    config: { sandbox: "docker" }

members:
  - name: "Alice"
    role: "coordinator"
    # 不指定 equipment，使用全部共享装备
    
  - name: "Bob"
    role: "ocr_expert"
    preferred_equipment: ["ocr", "image_analysis"]
    
  - name: "Carol"
    role: "legal_analyst"
    preferred_equipment: ["pdf_parser", "web_search"]
```

#### 使用流程
```python
# 1. Agent 调用装备
result = agent.use_tool("ocr", {"image": "contract_scan.jpg"})

# 2. 协调员根据偏好分配
if task.needs_ocr:
    best_agent = crew.find_agent_by_preference("ocr")  # → Bob
    
# 3. 装备执行
# MCP 装备 → 调用 MCP Server
# Plugin 装备 → 调用本地实现
```

---

### 2. 记忆系统 (Memory System)

#### 三层架构
```
┌─────────────────────────────────────┐
│         Crew 共同记忆                │
│  - 当前任务上下文                    │
│  - 对话历史 (完整)                   │
│  - 共享知识 (RAG)                    │
│  - 中间结果                         │
└─────────────────────────────────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
┌────────┐  ┌────────┐
│ Alice  │  │  Bob   │
│ 记忆   │  │  记忆   │
│        │  │        │
│-从共同 │  │-从共同 │
│ 记忆中 │  │ 记忆中 │
│ 提取的 │  │ 提取的 │
│ 相关部分│  │ 相关部分│
│        │  │        │
│-个人   │  │-个人   │
│ 偏好   │  │ 专长   │
└────────┘  └────────┘
```

#### Crew 共同记忆
```python
class CrewMemory:
    """团队共享记忆"""
    
    def __init__(self, crew_name: str):
        self.conversation_history = []   # 完整对话历史
        self.task_context = {}           # 当前任务上下文
        self.shared_knowledge = ChromaDBVectorStore()  # 共享知识库
        self.intermediate_results = {}   # 中间结果缓存
    
    def add_message(self, agent_name: str, role: str, content: str):
        """添加对话记录"""
        self.conversation_history.append({
            "agent": agent_name,
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })
    
    def get_relevant_context(self, agent_name: str, query: str, k: int = 5) -> str:
        """为特定 Agent 提取相关上下文"""
        # 1. 从对话历史中检索相关部分
        relevant_history = self._retrieve_relevant_history(query, k)
        
        # 2. 从共享知识库检索
        relevant_knowledge = self.shared_knowledge.search(query, k)
        
        # 3. 该 Agent 之前的中间结果
        agent_results = self.intermediate_results.get(agent_name, [])
        
        return self._format_context(relevant_history, relevant_knowledge, agent_results)
    
    def store_intermediate_result(self, agent_name: str, task_id: str, result: Any):
        """存储中间结果供其他 Agent 使用"""
        if agent_name not in self.intermediate_results:
            self.intermediate_results[agent_name] = []
        self.intermediate_results[agent_name].append({
            "task_id": task_id,
            "result": result,
            "timestamp": datetime.now()
        })
```

#### Agent 个人记忆
```python
class AgentMemory:
    """Agent 个人记忆 - 从 Crew 记忆 + 个人偏好"""
    
    def __init__(self, agent_name: str, crew_memory: CrewMemory):
        self.agent_name = agent_name
        self.crew_memory = crew_memory
        self.personal_preferences = {}   # 个人偏好记忆
        self.specialty_knowledge = {}    # 专业领域知识
    
    def get_context_for_task(self, task: str) -> str:
        """构建 Agent 的完整上下文"""
        # 1. 从 Crew 共同记忆提取相关部分
        crew_context = self.crew_memory.get_relevant_context(
            self.agent_name, task, k=5
        )
        
        # 2. 个人偏好
        personal = f"\n[你的偏好和风格]\n{self.personal_preferences}"
        
        # 3. 专业背景
        specialty = f"\n[你的专业领域]\n{self.specialty_knowledge}"
        
        return f"{crew_context}\n{personal}\n{specialty}"
```

#### 记忆流转流程
```
1. 用户输入问题
   ↓
2. 存入 Crew 共同记忆
   conversation_history.append({user: "分析合同"})
   ↓
3. 协调员分配给 Bob
   ↓
4. Bob 从 Crew 记忆提取相关上下文
   crew_memory.get_relevant_context("Bob", "合同扫描件 OCR")
   → 用户上传了 contract_scan.pdf
   → 这是采购合同类型
   → 需要关注第3、5条
   ↓
5. Bob 完成 OCR，结果存入 Crew 记忆
   crew_memory.store_intermediate_result("Bob", "ocr_task", result)
   intermediate_results["Bob"] = [{ocr结果}]
   ↓
6. 协调员分配给 Carol
   ↓
7. Carol 从 Crew 记忆提取 (包含 Bob 的 OCR 结果)
   crew_memory.get_relevant_context("Carol", "分析合同条款")
   → Bob 已完成 OCR，文本内容: ...
   → 第3条是关于...，第5条是关于...
   ↓
8. Carol 完成分析，结果存入 Crew 记忆
   ...
```

---

### 3. 团队系统 (Crew System)

#### 核心概念
- **声明式定义**: YAML 定义团队成员和协作流程
- **角色分工**: 每个 Agent 有明确的角色和专长
- **工作流编排**: 支持依赖关系和条件分支

#### Crew 配置
```yaml
# crew.yaml
name: "ContractReviewCrew"
description: "合同审查团队"

# 共享装备池
shared_equipment:
  - name: "web_search"
    type: mcp
    config: { engine: "duckduckgo" }
  - name: "ocr"
    type: mcp
    config: { language: "zh+en" }
  - name: "pdf_parser"
    type: plugin

# 团队成员
members:
  - name: "Alice"
    role: "coordinator"
    description: "法律背景，协调沟通"
    preferred_equipment: ["web_search", "file_reader"]
    
  - name: "Bob"
    role: "ocr_expert"
    description: "OCR 专家，处理扫描件"
    preferred_equipment: ["ocr", "image_analysis"]
    
  - name: "Carol"
    role: "legal_analyst"
    description: "法律分析师，提取关键信息"
    preferred_equipment: ["pdf_parser", "web_search"]
    
  - name: "David"
    role: "writer"
    description: "报告撰写员"
    preferred_equipment: ["file_reader"]

# 工作流定义
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

#### CLI 命令
```bash
# Crew 管理
kas crew create ReviewTeam                    # 创建团队
kas crew add ReviewTeam MyAgent --role analyst # 添加成员
kas crew remove ReviewTeam Alice               # 移除成员
kas crew list                                  # 列出团队
kas crew show ReviewTeam                       # 查看团队详情

# 执行任务
kas crew run ReviewTeam "分析合同" \
  --attach contract.pdf \
  --attach photo.jpg

# 交互式对话
kas crew chat ReviewTeam
```

---

### 4. 协调员系统 (Coordinator System)

#### 核心概念
- **动态选举**: 协调员不固定，根据任务阶段动态选择
- **直接沟通**: 谁负责当前环节，谁来和用户沟通
- **问题即时澄清**: 遇到问题的 Agent 直接询问，不中转

#### 协调流程
```
用户: 分析这份合同的风险点
附件: contract.pdf, photo1.jpg

[团队内部讨论 - Alice, Bob, Carol, David]

Alice: "我来确认需求" 
       → 临时协调员 (需求确认阶段)
       "我看到你上传了合同 PDF 和一张照片。
        让 [Bob] 处理图片，[Carol] 分析条款。
        需要我这样做吗？"

用户: 是的

[执行阶段]

Bob: "图片识别完成，这是签署页"
     → Bob 汇报进度 (执行阶段)

Carol: "第5条违约责任表述模糊，不确定具体含义" 
       → Carol 成为临时协调员 (有问题时)
       "你能解释一下这条的意图吗？"

用户: 这是想表达违约方需赔偿实际损失...

Carol: [把解释同步给团队]

[结果汇总]

David: "报告完成，关键风险点：1. XX 2. YY"
       → David 成为临时协调员 (交付阶段)
       "需要修改哪里吗？"

用户: 没问题

David: "最终报告已生成"
```

#### 选举逻辑
| 阶段 | 谁出面 | 原因 |
|------|--------|------|
| 需求确认 | 最理解业务的 Agent | 快速理解意图 |
| 执行阶段 | 当前负责的 Agent | 直接汇报进度 |
| 有问题时 | 遇到问题的 Agent | 第一时间澄清 |
| 结果汇总 | 负责交付的 Agent | 完整呈现结果 |

#### 单 Agent 退化
```yaml
# 单 Agent Crew = OpenClaw
crew:
  name: "SoloAgent"
  members:
    - name: "Assistant"
      role: "general"
  
  # 退化行为:
  # - 没有团队讨论
  # - 没有协调员选举 (自己就是协调员)
  # - 直接使用 Agent 的 system_prompt
  # - 装备直接使用，不需要分配
  # - 记忆 = Crew记忆 = Agent记忆 (同一层)
```

```python
# 代码层面的退化
if len(crew.members) == 1:
    # 退化模式：跳过所有团队逻辑
    agent = crew.members[0]
    return agent.run(task)  # 直接执行
else:
    # 完整团队模式
    return crew.execute_with_coordination(task)
```

---

## 🔄 完整交互流程

### 场景: 合同审查

```
┌─────────┐
│  User   │
└────┬────┘
     │
     │ 输入: "分析合同风险"
     │ 附件: contract.pdf, photo1.jpg
     ▼
┌─────────────────────────────────────────┐
│           Crew 共同记忆                 │
│  conversation_history: [                │
│    {role: "user", content: "分析..."}   │
│  ]                                      │
│  attachments: [contract.pdf, photo1.jpg]│
└─────────────────────────────────────────┘
     │
     ▼
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│  Alice  │  │   Bob   │  │  Carol  │  │  David  │
│ 选举为  │  │         │  │         │  │         │
│ 协调员  │  │         │  │         │  │         │
└────┬────┘  └─────────┘  └─────────┘  └─────────┘
     │
     │ Alice: "让Bob处理图片，Carol分析条款，可以吗？"
     ▼
┌─────────┐
│  User   │
└────┬────┘
     │ 回答: "是的"
     ▼
┌─────────────────────────────────────────┐
│           Crew 共同记忆                 │
│  conversation_history: [                │
│    {role: "user", content: "分析..."},  │
│    {agent: "Alice", role: "assistant",  │
│     content: "让Bob处理...可以吗？"},    │
│    {role: "user", content: "是的"}      │
│  ]                                      │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────┐
│   Bob   │  ← 使用 ocr 装备
│ OCR专家 │
└────┬────┘
     │
     │ Bob 提取上下文:
     │ - 用户要分析合同
     │ - 有 photo1.jpg 需要 OCR
     │
     │ Bob.use_tool("ocr", {image: "photo1.jpg"})
     │ → 识别结果: "签署页，甲方: XX公司..."
     ▼
┌─────────────────────────────────────────┐
│           Crew 共同记忆                 │
│  intermediate_results["Bob"]: [{         │
│    task: "ocr",                         │
│    result: "签署页，甲方: XX公司..."      │
│  }]                                     │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────┐
│  Carol  │  ← 使用 pdf_parser + web_search
│法律分析 │
└────┬────┘
     │
     │ Carol 提取上下文:
     │ - 用户要分析合同
     │ - Bob 已完成 OCR: 签署页...
     │ - 需要分析 contract.pdf
     │
     │ Carol.use_tool("pdf_parser", {file: "contract.pdf"})
     │ → 文本内容: "第1条...第2条..."
     │
     │ Carol: "第5条违约责任表述模糊"
     │ 成为临时协调员询问用户
     ▼
┌─────────┐
│  User   │
└────┬────┘
     │ 解释: "这是想表达违约方需赔偿实际损失"
     ▼
┌─────────────────────────────────────────┐
│           Crew 共同记忆                 │
│  intermediate_results["Carol"]: [{       │
│    task: "analyze",                     │
│    result: "第5条问题...",               │
│    clarification: "违约方需赔偿..."       │
│  }]                                     │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────┐
│  David  │  ← 撰写报告
│ 撰写员  │
└────┬────┘
     │
     │ David 提取上下文:
     │ - Bob 的 OCR 结果
     │ - Carol 的分析结果
     │ - 用户解释的条款含义
     │
     │ David 生成报告
     │ 成为临时协调员展示结果
     ▼
┌─────────┐
│  User   │
└────┬────┘
     │ "没问题，报告很好"
     ▼
┌─────────────────────────────────────────┐
│           Crew 共同记忆                 │
│  conversation_history: [完整对话]       │
│  intermediate_results: [                │
│    Bob: {ocr结果},                      │
│    Carol: {分析结果},                   │
│    David: {最终报告}                    │
│  ]                                      │
│  final_report: "合同风险分析报告..."      │
└─────────────────────────────────────────┘
```

---

## 📝 与 OpenClaw 的对应关系

| OpenClaw | KAS Phase 4 | 说明 |
|----------|-------------|------|
| `SOUL.md` | `system_prompt` | Agent 灵魂定义 |
| `TOOLS.md` | `shared_equipment` | 可用工具列表 |
| `AGENTS.md` | `crew.yaml` | Agent 定义文件 |
| `Memory/` | `CrewMemory` | 对话历史存储 |
| `memory/*.md` | `AgentMemory` | 个人偏好/专长 |
| Skills | MCP/Plugin | 工具实现 |
| Session | `CrewSession` | 单次对话上下文 |
| Single Agent | `len(crew.members) == 1` | 单 Agent 退化 |

**关键区别**:
- OpenClaw: 单实例，帮用户做事
- KAS Crew: 多 Agent 协作，Agent 之间也协作

---

## 🚀 实施计划

### Phase 4.2: 装备系统 (1周)
- [ ] MCP 协议适配器
- [ ] Plugin 基类和注册机制
- [ ] 内置装备: web_search, ocr, pdf_parser, code_executor, file_reader
- [ ] EquipmentPool 管理
- [ ] CLI: `kas equip`

### Phase 4.3: 多模态输入 (3天)
- [ ] 文件上传 API
- [ ] 图片/PDF 预处理器
- [ ] 附件传递到 Agent
- [ ] CLI: `--attach` 参数

### Phase 4.4: Agent 团队 (1周)
- [ ] Crew 定义和加载
- [ ] 成员角色管理
- [ ] 工作流编排引擎
- [ ] 装备偏好匹配
- [ ] CLI: `kas crew`

### Phase 4.5: 协调员模式 (1周)
- [ ] CrewMemory 实现
- [ ] AgentMemory 实现
- [ ] 动态协调员选举
- [ ] 用户确认流程
- [ ] 单 Agent 退化逻辑

---

**总工作量**: 3.5 周
**核心文件**:
- `kas/core/equipment.py` - 装备系统
- `kas/core/crew.py` - 团队和协调员
- `kas/core/crew_memory.py` - 分层记忆
- `kas/cli/main.py` - 新增 CLI 命令

---

Author: Yilin.zhang | Date: 2026-03-18
