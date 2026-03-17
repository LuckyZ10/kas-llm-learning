# KAS Phase 4 重新设计

## 愿景
从"单个 Agent" 升级到 "Agent 特种部队"

## 核心概念

### 1. Agent 装备系统 (MCP/Plugin)
每个 Agent 可以装备不同的"武器":

```yaml
# agent.yaml
name: "ResearchAgent"
equipment:
  - type: mcp
    name: "web_search"
    config:
      engine: "duckduckgo"
  - type: mcp
    name: "ocr"
    config:
      language: "zh+en"
  - type: plugin
    name: "pdf_parser"
  - type: plugin
    name: "code_executor"
    config:
      sandbox: "docker"
```

内置装备:
- `web_search` - 联网搜索
- `ocr` - 图片文字识别
- `pdf_parser` - PDF 解析
- `code_executor` - 代码执行（沙箱）
- `file_reader` - 文件读取
- `image_analysis` - 图片分析

### 2. Agent 团队 (Crew)
多个 Agent 组成团队，分工协作:

```yaml
# crew.yaml
name: "DocumentAnalysisCrew"
description: "文档分析团队"

coordinator: "Alice"  # 协调员，对外沟通

members:
  - name: "Alice"
    role: "coordinator"
    description: "团队协调员，负责与用户确认需求"
    equipment: ["web_search", "file_reader"]
    
  - name: "Bob"
    role: "ocr_expert"
    description: "OCR 专家，处理图片和扫描件"
    equipment: ["ocr", "image_analysis"]
    
  - name: "Carol"
    role: "analyst"
    description: "分析师，提取关键信息"
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

### 3. 协调员模式 (动态选举)

**不是固定协调员**，而是根据任务动态选举：

```
用户: 分析这份合同的风险点
附件: contract.pdf, photo1.jpg

[团队内部讨论 - Alice, Bob, Carol, David]

Alice: "我来确认需求" → 临时协调员
       "我看到你上传了合同 PDF 和一张照片。
        让 [Bob] 处理图片，[Carol] 分析条款。
        需要我这样做吗？"

用户: 是的

[执行阶段 - Bob 负责图片，Carol 负责分析]

Carol: "发现第5条有问题，不确定具体含义" 
       → Carol 成为临时协调员，询问用户
       "第5条违约责任表述模糊，你能解释一下意图吗？"

用户: 这是想表达...

Carol: [把解释同步给团队]

David: "我写报告" → David 成为临时协调员
       "报告完成，关键风险点：1. XX 2. YY
        需要修改哪里吗？"
```

**选举逻辑**:
- 需求确认阶段 → 最理解业务的 Agent 出面
- 执行阶段 → 当前负责的 Agent 直接汇报进度
- 有问题时 → 遇到问题的 Agent 直接询问
- 结果汇总 → 负责最终交付的 Agent 出面

**优势**:
- 更自然，像真实团队讨论
- 减少信息中转损耗
- 每个 Agent 都有机会直接沟通
- 问题能第一时间得到澄清

### 4. 多模态输入
支持的问题格式:
```
用户: 分析这份合同的风险点
附件: contract.pdf, photo1.jpg

Coordinator: 我看到你上传了合同 PDF 和一张照片。
           让我分配 [Carol] 分析 PDF，[Bob] 识别照片内容。
           需要我这样做吗？

用户: 是的

Coordinator: ✅ [Carol] 完成 PDF 分析，发现 3 个条款需要关注...
           ✅ [Bob] 识别出照片是合同签署页...
           [David] 正在撰写风险报告...
           
           初步发现：第 5 条违约责任条款表述模糊，
           建议修改为 XX。你觉得这个方向对吗？

用户: 第 3 条也需要看看

Coordinator: 明白，让 [Carol] 重点分析第 3 条...
```

## 技术架构

```
KAS Core
├── Agent (装备系统)
│   ├── equipment[]  # MCP/Plugin 列表
│   ├── use_tool(name, params)  # 调用装备
│   └── can_handle(task)  # 检查能力匹配
│
├── Equipment (装备基类)
│   ├── MCPAdapter  # MCP 协议适配
│   └── PluginAdapter  # 内部插件适配
│
├── Crew (团队)
│   ├── members[]  # Agent 列表
│   ├── coordinator  # 协调员
│   └── execute(task, attachments)  # 团队执行
│
├── Multimodal Input
│   ├── ImageProcessor  # 图片处理
│   ├── PDFProcessor   # PDF 处理
│   └── FileProcessor  # 通用文件
│
└── Coordinator Engine
    ├── understand_intent()  # 理解意图
    ├── confirm_with_user()  # 确认需求
    ├── delegate_task()      # 任务分发
    ├── summarize_results()  # 结果汇总
    └── ask_clarification()  # 澄清问题
```

## CLI 设计

```bash
# 装备管理
kas equip list                    # 列出可用装备
kas equip add MyAgent web_search  # 给 Agent 添加装备
kas equip remove MyAgent ocr      # 移除装备

# 团队管理
kas crew create AnalysisTeam      # 创建团队
kas crew add AnalysisTeam MyAgent --role analyst
kas crew set-coordinator AnalysisTeam Alice
kas crew show AnalysisTeam        # 查看团队

# 多模态任务
kas crew run AnalysisTeam "分析这份合同" \
  --attach contract.pdf \
  --attach photo.jpg

# 或者交互式
kas crew chat AnalysisTeam
> 分析这份合同
[上传文件: contract.pdf]
Coordinator: 我看到你上传了合同 PDF。让我分配 [Carol] 分析，需要吗？
> 是的
...
```

## 实现优先级

Phase 4.2: 装备系统 (MCP + 内置插件)
- MCP 协议适配器
- 内置装备: web_search, ocr, pdf_parser
- Agent.use_tool() 接口

Phase 4.3: 多模态输入
- 文件上传处理
- 图片/PDF 预处理器
- 附件传递给 Agent

Phase 4.4: 协调员模式
- Coordinator 角色
- 任务分发逻辑
- 用户确认流程

Phase 4.5: Agent 团队 (Crew)
- Crew 定义和管理
- 成员角色分配
- 团队执行引擎

## 与现有功能的整合

- 复用 workflow.py 的任务编排
- 复用 knowledge.py 的 RAG 能力
- 复用 chat.py 的对话引擎
- 新增装备调用层

## 示例场景

**场景 1: 合同审查**
```
Crew: ContractReviewTeam
Members:
  - Alice (Coordinator): 法律背景，协调沟通
  - Bob (OCR Expert): 处理扫描件
  - Carol (Legal Analyst): 法律分析
  - David (Risk Assessor): 风险评估

Input: "审查这份采购合同的风险" + contract_scan.pdf
Output: 风险报告 + 修改建议
```

**场景 2: 代码审查**
```
Crew: CodeReviewTeam
Members:
  - Alice (Coordinator): 架构师，协调沟通
  - Bob (Security Expert): 安全检查
  - Carol (Performance Expert): 性能分析
  - David (Style Expert): 代码规范

Input: "review 这个 PR" + pr_diff.patch
Output: 审查意见汇总
```

**场景 3: 研究助理**
```
Crew: ResearchTeam
Members:
  - Alice (Coordinator): 研究方向把控
  - Bob (Web Searcher): 资料搜集
  - Carol (Summarizer): 信息整理
  - David (Writer): 撰写报告

Input: "调研一下 RAG 的最新进展" + paper1.pdf + paper2.pdf
Output: 调研报告
```
