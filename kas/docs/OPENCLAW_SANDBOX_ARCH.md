# KAS + OpenClaw 多 Agent 沙盒架构设计

## 🎯 核心思想

**每个 OpenClaw 实例 = 一个 Agent 沙盒**

通过 "灵魂注入" 把 KAS Agent 配置转换为 OpenClaw 配置文件，实现真正的 Agent 沙盒化运行。

---

## 🏗️ 架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                     KAS Crew (特种部队指挥中心)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  OpenClaw    │  │  OpenClaw    │  │  OpenClaw    │              │
│  │   Sandbox    │  │   Sandbox    │  │   Sandbox    │              │
│  │   (Alice)    │  │    (Bob)     │  │   (Carol)    │              │
│  │              │  │              │  │              │              │
│  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │              │
│  │ │ SOUL.md  │ │  │ │ SOUL.md  │ │  │ │ SOUL.md  │ │              │
│  │ │(注入灵魂)│ │  │ │(注入灵魂)│ │  │ │(注入灵魂)│ │              │
│  │ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │              │
│  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │              │
│  │ │AGENTS.md │ │  │ │AGENTS.md │ │  │ │AGENTS.md │ │              │
│  │ │(工作指南)│ │  │ │(工作指南)│ │  │ │(工作指南)│ │              │
│  │ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │              │
│  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │              │
│  │ │TOOLS.md  │ │  │ │TOOLS.md  │ │  │ │TOOLS.md  │ │              │
│  │ │(装备配置)│ │  │ │(装备配置)│ │  │ │(装备配置)│ │              │
│  │ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │              │
│  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │              │
│  │ │Memory/   │ │  │ │Memory/   │ │  │ │Memory/   │ │              │
│  │ │(个人记忆)│ │  │ │(个人记忆)│ │  │ │(个人记忆)│ │              │
│  │ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
│         │                 │                 │                       │
│         └─────────────────┼─────────────────┘                       │
│                           │                                         │
│                    ┌──────┴──────┐                                 │
│                    │  Message    │                                 │
│                    │   Bus       │  ← 沙盒间通信总线                │
│                    │ (IPC/Queue) │                                 │
│                    └──────┬──────┘                                 │
│                           │                                         │
│         ┌─────────────────┼─────────────────┐                      │
│         ▼                 ▼                 ▼                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │  CrewMemory  │  │  Equipment   │  │ Coordinator  │             │
│  │  (共享记忆)  │  │    Pool      │  │   Engine     │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │      User       │
                    └─────────────────┘
```

---

## 🔮 灵魂注入机制

### 1. KAS Agent → OpenClaw 配置转换

```python
# KAS Agent 定义 (agent.yaml)
name: "Alice"
description: "法律专家，擅长合同审查"
system_prompt: |
  你是一位经验丰富的法律专家，专注于合同法领域...
capabilities:
  - contract_review
  - risk_assessment
equipment:
  - web_search
  - pdf_parser
```

```python
# 转换为 OpenClaw SOUL.md
class SoulInjector:
    """灵魂注入器 - 将 KAS Agent 转换为 OpenClaw 配置"""
    
    def inject(self, kas_agent: Agent, sandbox_path: Path):
        """注入灵魂到沙盒"""
        
        # 1. 生成 SOUL.md (核心灵魂)
        soul_content = f"""# SOUL.md - {kas_agent.name}

## 身份

**名字**: {kas_agent.name}
**emoji**: {kas_agent.emoji or '🤖'}
**本质**: {kas_agent.description}

## 核心能力

{self._format_capabilities(kas_agent.capabilities)}

## 系统提示词

```
{kas_agent.system_prompt}
```

## 工作原则

1. 专注于你的专业领域
2. 与其他 Agent 协作时保持专业态度
3. 遇到不确定的问题时主动询问
4. 通过 Message Bus 与其他 Agent 通信

## 记忆

你会记住：
- 对话中的关键决策
- 用户的偏好和要求
- 与其他 Agent 的协作历史
"""
        
        # 2. 生成 AGENTS.md (工作指南)
        agents_content = f"""# AGENTS.md - {kas_agent.name}

## 你是 KAS Crew 的成员

你属于 Crew: {{crew_name}}
你的角色: {{role}}

## 你的队友

{{team_members_list}}

## 协作规则

1. **通过 Message Bus 通信**
   - 不要直接读写其他 Agent 的文件
   - 使用 send_message() 发送消息
   - 使用 receive_message() 接收消息

2. **任务执行流程**
   - 从 Coordinator 接收任务
   - 执行你的专业能力
   - 通过 Message Bus 汇报结果
   - 等待下一步指示

3. **成为协调员时**
   - 直接与用户沟通
   - 代表团队询问确认
   - 将用户反馈传达给团队

## 安全边界

- 你只能访问自己的 Memory/ 目录
- 你可以使用分配的 Equipment
- 你不能执行 Crew Memory 以外的代码
"""
        
        # 3. 生成 TOOLS.md (装备配置)
        tools_content = f"""# TOOLS.md - {kas_agent.name}

## 可用工具

{self._format_equipment(kas_agent.equipment)}

## 队友信息

- 使用 Message Bus 与其他 Agent 通信
- Crew Memory 位置: {{crew_memory_path}}

## 注意

这些工具仅限当前沙盒使用，不会影响其他 Agent。
"""
        
        # 写入沙盒
        (sandbox_path / "SOUL.md").write_text(soul_content)
        (sandbox_path / "AGENTS.md").write_text(agents_content)
        (sandbox_path / "TOOLS.md").write_text(tools_content)
```

---

## 📦 沙盒结构

```
~/.kas/sandboxes/
└── ContractReviewCrew/
    ├── alice/                 # Alice 的 OpenClaw 沙盒
    │   ├── SOUL.md           # 注入的灵魂
    │   ├── AGENTS.md         # 工作指南
    │   ├── TOOLS.md          # 装备配置
    │   ├── USER.md           # 用户信息（共享）
    │   ├── MEMORY.md         # 个人长期记忆
    │   └── memory/           # 每日记忆
    │       ├── 2026-03-18.md
    │       └── ...
    │
    ├── bob/                   # Bob 的 OpenClaw 沙盒
    │   ├── SOUL.md
    │   ├── AGENTS.md
    │   ├── TOOLS.md
    │   ├── USER.md -> ../alice/USER.md  (共享)
    │   ├── MEMORY.md
    │   └── memory/
    │
    ├── carol/                 # Carol 的 OpenClaw 沙盒
    │   └── ...
    │
    ├── shared/                # 共享资源
    │   ├── crew_memory.json   # Crew 共同记忆
    │   ├── message_bus/       # 消息总线存储
    │   │   ├── inbox/         # 收件箱
    │   │   └── outbox/        # 发件箱
    │   └── equipment_pool/    # 装备池配置
    │
    └── supervisor.py          # 沙盒监督器
```

---

## 📡 沙盒间通信 (Message Bus)

### 1. 通信协议

```python
class MessageBus:
    """轻量级消息总线 - 沙盒间通信"""
    
    def __init__(self, crew_path: Path):
        self.inbox = crew_path / "shared" / "message_bus" / "inbox"
        self.outbox = crew_path / "shared" / "message_bus" / "outbox"
    
    def send(self, from_agent: str, to_agent: str, message_type: str, content: dict):
        """发送消息"""
        msg = {
            "id": str(uuid.uuid4()),
            "from": from_agent,
            "to": to_agent,  # "*" 表示广播
            "type": message_type,  # task, result, question, broadcast
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "reply_to": None
        }
        
        # 写入消息队列
        msg_file = self.inbox / f"{msg['id']}.json"
        msg_file.write_text(json.dumps(msg, indent=2))
    
    def receive(self, agent_name: str, block: bool = False, timeout: float = None) -> Optional[dict]:
        """接收消息"""
        # 查找发给该 Agent 的消息
        for msg_file in sorted(self.inbox.glob("*.json")):
            msg = json.loads(msg_file.read_text())
            if msg["to"] in [agent_name, "*"]:
                # 标记为已读
                msg["read_by"] = agent_name
                msg["read_at"] = datetime.now().isoformat()
                
                # 移动到已读
                read_file = self.outbox / msg_file.name
                read_file.write_text(json.dumps(msg, indent=2))
                msg_file.unlink()
                
                return msg
        
        return None
    
    def broadcast(self, from_agent: str, content: dict):
        """广播消息给所有 Agent"""
        self.send(from_agent, "*", "broadcast", content)
```

### 2. 消息类型

```python
class MessageTypes:
    TASK = "task"           # 分配任务
    RESULT = "result"       # 任务结果
    QUESTION = "question"   # 询问用户
    ANSWER = "answer"       # 用户回答
    BROADCAST = "broadcast" # 广播通知
    HANDOVER = "handover"   # 协调员交接
```

### 3. 使用示例

```python
# Alice (协调员) 分配任务给 Bob
message_bus.send(
    from_agent="alice",
    to_agent="bob",
    message_type="task",
    content={
        "task_id": "ocr_001",
        "description": "对 contract_scan.jpg 进行 OCR",
        "params": {"image": "shared/inputs/contract_scan.jpg"},
        "deadline": "2026-03-18T02:00:00"
    }
)

# Bob 接收任务
msg = message_bus.receive("bob", block=True, timeout=30)
if msg and msg["type"] == "task":
    result = execute_task(msg["content"])
    
    # 返回结果
    message_bus.send(
        from_agent="bob",
        to_agent="alice",
        message_type="result",
        content={
            "task_id": msg["content"]["task_id"],
            "result": result,
            "status": "success"
        }
    )

# Carol 有问题需要询问用户
message_bus.send(
    from_agent="carol",
    to_agent="alice",  # 通过协调员转达
    message_type="question",
    content={
        "question": "第5条违约责任表述模糊，能解释一下意图吗？",
        "context": "这是关于违约赔偿的条款"
    }
)

# Alice 将问题转达给用户，然后把回答传回
user_answer = ask_user(msg["content"]["question"])
message_bus.send(
    from_agent="alice",
    to_agent="carol",
    message_type="answer",
    content={"answer": user_answer}
)
```

---

## 🎭 沙盒监督器 (Supervisor)

### 职责

```python
class SandboxSupervisor:
    """沙盒监督器 - 管理所有 OpenClaw Agent 沙盒"""
    
    def __init__(self, crew_path: Path):
        self.crew_path = crew_path
        self.sandboxes: Dict[str, OpenClawSandbox] = {}
        self.message_bus = MessageBus(crew_path)
        self.coordinator = DynamicCoordinator()
    
    def create_sandbox(self, agent_config: AgentConfig) -> OpenClawSandbox:
        """创建新的 OpenClaw 沙盒"""
        sandbox_path = self.crew_path / agent_config.name
        sandbox_path.mkdir(parents=True, exist_ok=True)
        
        # 灵魂注入
        injector = SoulInjector()
        injector.inject(agent_config, sandbox_path)
        
        # 创建共享链接
        self._setup_shared_resources(sandbox_path)
        
        # 启动沙盒
        sandbox = OpenClawSandbox(
            path=sandbox_path,
            message_bus=self.message_bus,
            equipment_pool=self.equipment_pool
        )
        
        self.sandboxes[agent_config.name] = sandbox
        return sandbox
    
    def run_crew_task(self, task: str, attachments: List[Path]):
        """执行 Crew 任务"""
        # 1. 选举初始协调员
        coordinator = self.coordinator.elect_for_stage("init", self.sandboxes)
        
        # 2. 协调员确认需求
        confirmed = coordinator.confirm_task(task, attachments)
        
        # 3. 创建任务计划
        plan = self.coordinator.create_plan(confirmed, self.sandboxes)
        
        # 4. 按顺序执行
        for step in plan.steps:
            agent = self.sandboxes[step.agent_name]
            
            # 准备上下文 (从 Crew Memory + 个人记忆)
            context = self._prepare_context(agent, step)
            
            # 在沙盒中执行任务
            result = agent.execute(step.task, context)
            
            # 存储到 Crew Memory
            self.crew_memory.store_result(agent.name, step.id, result)
            
            # 如果需要询问用户
            if result.needs_clarification:
                # 该 Agent 成为临时协调员
                answer = agent.ask_user(result.question)
                result.update_with_answer(answer)
        
        # 5. 汇总结果
        final_coordinator = self.coordinator.elect_for_stage("final", self.sandboxes)
        return final_coordinator.summarize(self.crew_memory)
    
    def _setup_shared_resources(self, sandbox_path: Path):
        """设置共享资源链接"""
        shared_path = self.crew_path / "shared"
        
        # 共享 USER.md
        user_link = sandbox_path / "USER.md"
        if not user_link.exists():
            user_link.symlink_to(shared_path / "USER.md")
    
    def shutdown_all(self):
        """关闭所有沙盒"""
        for sandbox in self.sandboxes.values():
            sandbox.shutdown()
```

---

## 🔧 沙盒实现 (OpenClawSandbox)

```python
class OpenClawSandbox:
    """OpenClaw 沙盒包装器"""
    
    def __init__(self, path: Path, message_bus: MessageBus, equipment_pool: EquipmentPool):
        self.path = path
        self.name = path.name
        self.message_bus = message_bus
        self.equipment_pool = equipment_pool
        
        # 加载配置
        self.soul = self._load_soul()
        self.agent_config = self._load_agents_md()
        self.tools = self._load_tools()
        
        # 启动 OpenClaw 会话
        self.session = self._start_openclaw_session()
    
    def execute(self, task: str, context: dict) -> SandboxResult:
        """在沙盒中执行任务"""
        # 构建完整提示词
        prompt = self._build_prompt(task, context)
        
        # 调用 OpenClaw
        response = self.session.run(prompt)
        
        # 解析响应
        return self._parse_response(response)
    
    def send_message(self, to: str, msg_type: str, content: dict):
        """发送消息给其他 Agent"""
        self.message_bus.send(self.name, to, msg_type, content)
    
    def receive_message(self, block: bool = True, timeout: float = None) -> Optional[dict]:
        """接收消息"""
        return self.message_bus.receive(self.name, block, timeout)
    
    def use_equipment(self, equipment_name: str, params: dict) -> Any:
        """使用装备"""
        return self.equipment_pool.use(equipment_name, params)
    
    def ask_user(self, question: str) -> str:
        """询问用户 (通过协调员转发)"""
        self.send_message(
            to="coordinator",
            msg_type="question",
            content={"question": question, "from": self.name}
        )
        
        # 等待回答
        while True:
            msg = self.receive_message(block=True)
            if msg["type"] == "answer":
                return msg["content"]["answer"]
    
    def _build_prompt(self, task: str, context: dict) -> str:
        """构建完整提示词"""
        return f"""
{self.soul.system_prompt}

## 当前任务

{task}

## 上下文

{context}

## 可用工具

{self._format_tools()}

## 执行指令

请使用你的专业能力完成上述任务。
如果需要使用工具，请明确说明。
如果需要询问用户，请通过协调员转达。

## 输出格式

请按以下格式输出：

### 思考过程
[你的思考]

### 执行结果
[具体结果]

### 是否需要澄清
[如果需要询问用户，请写出问题]
"""
    
    def shutdown(self):
        """关闭沙盒"""
        self.session.close()
```

---

## 📋 实施步骤

### Phase A: 基础沙盒 (1周)
1. [ ] SoulInjector 实现
2. [ ] OpenClawSandbox 包装器
3. [ ] MessageBus 文件队列
4. [ ] 单沙盒测试

### Phase B: 多沙盒通信 (1周)
1. [ ] 消息协议设计
2. [ ] 沙盒间通信测试
3. [ ] 任务分发机制
4. [ ] 结果收集机制

### Phase C: Crew 集成 (1周)
1. [ ] SandboxSupervisor
2. [ ] 动态协调员
3. [ ] Crew Memory 同步
4. [ ] 完整流程测试

### Phase D: 优化 (3天)
1. [ ] 性能优化
2. [ ] 错误恢复
3. [ ] 沙盒隔离强化
4. [ ] 监控和日志

---

## 🔐 安全考虑

```python
class SandboxSecurity:
    """沙盒安全层"""
    
    def __init__(self, sandbox_path: Path):
        self.path = sandbox_path
    
    def enforce_isolation(self):
        """强制执行沙盒隔离"""
        # 1. 文件系统隔离
        # 沙盒只能访问自己的目录
        
        # 2. 网络隔离
        # 沙盒不能直接访问网络，必须通过 Equipment
        
        # 3. 资源限制
        # CPU、内存、执行时间限制
        
        # 4. 审计日志
        # 记录所有文件访问和网络请求
        pass
```

---

## 💡 关键技术决策

1. **文件队列 vs 内存队列**: 文件队列更可靠，支持持久化
2. **Symlink vs Copy**: Symlink 共享文件，Copy 保证隔离
3. **同步 vs 异步**: 任务执行异步，用户确认同步
4. **沙盒粒度**: 每个 Agent 一个完整 OpenClaw 实例

---

这个架构实现了真正的 Agent 沙盒化：
- 每个 Agent 是独立的 OpenClaw 实例
- 灵魂注入实现角色区分
- Message Bus 实现安全通信
- Supervisor 统一协调

Author: Yilin.zhang | Date: 2026-03-18
