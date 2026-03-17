# KAS (Klaw Agent Studio) ROADMAP v1.2

> 版本: v1.2
> 最后更新: 2026-03-18
> 作者: Yilin.zhang
> 状态: Phase 4 重构 - Agent 特种部队

---

## 🎯 项目愿景

**KAS = OpenClaw 特种部队指挥系统**

从"单个 Agent 工具"进化为"多个 Agent 协作团队"，每个 Agent 都是注入灵魂的 OpenClaw 实例，通过沙盒隔离实现真正的多 Agent 并行协作。

---

## 📅 已完成阶段

### Phase 1-3: 基础功能 ✅ (2026-03-17 完成)

| 阶段 | 功能 | 状态 |
|------|------|------|
| Phase 1 | 代码吞食、Agent合体、对话、进化 | ✅ |
| Phase 2 | Agent市场、能力验证、统计面板 | ✅ |
| Phase 3 | 知识库/RAG、工作流、A/B测试 | ✅ |

### Phase 4.1-4.2: 装备与界面 ✅ (2026-03-18 完成)

| 阶段 | 功能 | 核心文件 |
|------|------|----------|
| 4.1 | Web界面MVP | `kas/web/app.py` |
| 4.2 | 装备系统 | `kas/core/equipment.py` |

---

## 🚀 Phase 4: Agent 特种部队 (进行中)

**目标**: 实现真正的多 Agent 沙盒化协作

### Phase 4.3: 多模态输入 📁 ⏳
**时间**: 3天  
**优先级**: 🔴 P0  
**前置**: 4.2 装备系统 ✅

**为什么先做**: 装备系统(web_search, ocr, pdf_parser)需要处理文件，这是基础能力

**功能**:
- 文件上传 API (PDF, 图片, 代码文件)
- 图片 OCR 预处理
- PDF 文本提取
- 附件传递给 Agent/装备

**CLI**:
```bash
kas chat MyAgent "分析这份合同" --attach contract.pdf --attach photo.jpg
kas crew run Team "任务" --attach file1.pdf --attach file2.png
```

**核心实现**:
```python
# core/multimodal.py
class MultimodalProcessor:
    def process(self, file: UploadFile) -> ProcessedContent:
        # PDF → pdf_parser 装备
        # Image → ocr / image_analysis 装备
        # Code → file_reader 装备
        pass
```

---

### Phase 4.4: OpenClaw 沙盒化 🔮 ⏳
**时间**: 2周  
**优先级**: 🔴 P0  
**前置**: 无 (独立基础设施)

**核心目标**: 将 KAS Agent 转换为真正的 OpenClaw 沙盒实例

**架构**:
```
~/.kas/sandboxes/
└── ContractReviewCrew/
    ├── alice/              # Alice 的 OpenClaw 沙盒
    │   ├── SOUL.md         # 注入的灵魂 (system_prompt)
    │   ├── AGENTS.md       # 工作指南 (Crew协作规则)
    │   ├── TOOLS.md        # 装备配置
    │   ├── USER.md         # 用户信息 (共享链接)
    │   ├── MEMORY.md       # 个人长期记忆
    │   └── memory/         # 每日记忆
    │
    ├── bob/                # Bob 的 OpenClaw 沙盒
    ├── carol/              # Carol 的 OpenClaw 沙盒
    │
    ├── shared/             # 共享资源
    │   ├── crew_memory.json    # Crew 共同记忆
    │   └── message_bus/        # 消息总线
    │       ├── inbox/
    │       └── outbox/
    │
    └── supervisor.py       # 沙盒监督器
```

**子任务**:

| 模块 | 功能 | 时间 | 文件 |
|------|------|------|------|
| SoulInjector | KAS Agent → OpenClaw 配置转换 | 2天 | `core/sandbox/soul_injector.py` |
| MessageBus | 沙盒间文件队列通信 | 3天 | `core/sandbox/message_bus.py` |
| OpenClawSandbox | 沙盒包装器 | 3天 | `core/sandbox/sandbox.py` |
| SandboxSupervisor | 监督器+生命周期管理 | 4天 | `core/sandbox/supervisor.py` |
| 集成测试 | 单沙盒+多沙盒测试 | 2天 | `tests/test_sandbox.py` |

**灵魂注入示例**:
```python
# KAS Agent 定义
agent = Agent(
    name="Alice",
    system_prompt="你是一位法律专家...",
    equipment=["web_search", "pdf_parser"]
)

# 注入后生成 OpenClaw 配置
sandboxes/alice/
├── SOUL.md          # 身份+能力+系统提示词
├── AGENTS.md        # "你是KAS Crew成员，队友是Bob/Carol..."
├── TOOLS.md         # "可用装备: web_search, pdf_parser..."
└── memory/          # 个人记忆存储
```

**MessageBus 通信**:
```python
# Alice 分配任务给 Bob
message_bus.send(
    from_agent="alice",
    to_agent="bob",
    type="task",
    content={"task": "OCR识别", "image": "contract.jpg"}
)

# Bob 返回结果
message_bus.send(
    from_agent="bob",
    to_agent="alice",
    type="result",
    content={"text": "识别完成..."}
)

# Carol 有问题 → 通过协调员询问用户
message_bus.send(
    from_agent="carol",
    to_agent="alice",
    type="question",
    content={"question": "第5条模糊，能解释吗？"}
)
```

---

### Phase 4.5: Agent 特种部队 👥 ⏳
**时间**: 1周  
**优先级**: 🔴 P0  
**前置**: 4.4 OpenClaw 沙盒化 ✅

**核心目标**: 基于沙盒实现真正的多 Agent 协作团队

**功能**:

1. **Crew 定义**
```yaml
# crew.yaml
name: "ContractReviewCrew"
members:
  - name: "Alice"
    role: "coordinator"
    description: "法律专家，协调沟通"
    
  - name: "Bob"
    role: "ocr_expert"
    description: "OCR专家"
    
  - name: "Carol"
    role: "legal_analyst"
    description: "法律分析师"

workflow:
  - step: 1
    agent: "Alice"
    task: "确认需求"
    
  - step: 2
    agent: "Bob"
    task: "OCR识别"
    condition: "if input.has_image"
    
  - step: 3
    agent: "Carol"
    task: "分析条款"
    depends_on: [2]
```

2. **动态协调员**
```
用户: 分析合同
       ↓
Alice(确认需求) → "让Bob处理图片，Carol分析条款？"
       ↓
用户: 是的
       ↓
Bob(处理中) → "图片识别完成"
       ↓
Carol(发现问题) → "第5条模糊，能解释吗？" ← Carol临时协调
       ↓
用户解释
       ↓
Carol(继续分析) → 结果存入Crew记忆
       ↓
Alice(汇总展示) → "分析报告完成"
```

3. **CLI 命令**
```bash
# Crew 管理
kas crew create ReviewTeam                    # 创建团队
kas crew add ReviewTeam MyAgent --role analyst # 添加成员
kas crew remove ReviewTeam Alice               # 移除成员
kas crew list                                  # 列出团队
kas crew show ReviewTeam                       # 查看详情

# 执行任务
kas crew run ReviewTeam "分析合同" \
  --attach contract.pdf \
  --attach photo.jpg

# 交互式团队对话
kas crew chat ReviewTeam
```

4. **分层记忆**
```
Crew 共同记忆 (shared/crew_memory.json)
├── 对话历史 (完整)
├── 任务上下文
├── 中间结果缓存
└── 共享知识库 (RAG)
         │
    ┌────┴────┐
    ▼         ▼
Alice记忆   Bob记忆  (各沙盒 memory/)
├── 提取的   ├── 提取的
│   相关部分    相关部分
├── 个人     ├── 个人
│   偏好        专长
└── 专业     └── 专业
    背景          背景
```

---

## 🏗️ Phase 5: 企业级功能 (长期)

**目标**: 生产环境可用

### 5.1 安全沙箱强化 🔒
- Docker 隔离代码执行
- 网络访问限制 (代理模式)
- 敏感信息过滤
- 资源配额限制 (CPU/内存/执行时间)

### 5.2 性能监控 📈
- 响应时间监控
- Token 消耗统计
- 错误率追踪
- 告警机制 (Slack/邮件)

### 5.3 团队协作 👥
- 多用户支持
- 权限管理 (RBAC)
- Agent 共享机制
- 团队知识库

### 5.4 高级装备 🛠️
- 自定义 MCP Server 接入
- Plugin 市场
- 装备版本管理
- A/B 测试装备效果

---

## 📊 实施时间线

```
2026-03-18  ├─ 4.3 多模态输入 (3天)
            │   └─ 文件上传、预处理、装备集成
            │
2026-03-21  ├─ 4.4 OpenClaw 沙盒化 (2周)
            │   ├─ Week 1: SoulInjector + MessageBus
            │   └─ Week 2: SandboxSupervisor + 测试
            │
2026-04-04  ├─ 4.5 Agent 特种部队 (1周)
            │   ├─ Crew 定义和管理
            │   ├─ 动态协调员
            │   └─ 完整流程测试
            │
2026-04-11  └─ Phase 4 完成！
                总时间: 3.5 周
```

---

## 🎯 成功标准

### Phase 4.3
- [ ] 支持上传 PDF、图片、代码文件
- [ ] 文件自动路由到对应装备
- [ ] CLI `--attach` 参数可用

### Phase 4.4
- [ ] SoulInjector 成功转换 Agent 配置
- [ ] MessageBus 支持 4 种消息类型
- [ ] 单沙盒可独立运行
- [ ] 多沙盒可相互通信

### Phase 4.5
- [ ] Crew YAML 定义和加载
- [ ] 动态协调员选举可用
- [ ] 任务自动分发和执行
- [ ] 分层记忆系统工作正常
- [ ] 完整合同审查流程可运行

---

## 💡 关键设计决策

1. **沙盒化优先**: 没有沙盒的"Agent团队"只是工作流包装，必须先实现沙盒
2. **文件队列通信**: MessageBus 使用文件队列，比内存队列更可靠、可持久化
3. **OpenClaw 兼容**: 沙盒就是标准 OpenClaw 实例，可独立运行调试
4. **向后兼容**: 单 Agent 自动退化为 OpenClaw 模式

---

## 📈 当前进度

| 阶段 | 状态 | 完成度 |
|------|------|--------|
| Phase 1-3 | ✅ | 100% |
| Phase 4.1 Web | ✅ | 100% |
| Phase 4.2 装备 | ✅ | 100% |
| Phase 4.3 多模态 | ⏳ | 0% |
| Phase 4.4 沙盒化 | ⏳ | 0% (设计✅) |
| Phase 4.5 特种部队 | ⏳ | 0% (设计✅) |
| Phase 5 企业级 | ⏳ | 0% |

**下一步**: 开始 Phase 4.3 多模态输入

---

Author: Yilin.zhang | Date: 2026-03-18
