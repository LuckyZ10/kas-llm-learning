# KAS (Klaw Agent Studio) ROADMAP v1.3

> 版本: v1.3
> 最后更新: 2026-03-23
> 作者: Yilin.zhang
> 状态: Phase 5 企业级功能开发中

---

## 🎯 项目愿景

**KAS = OpenClaw 特种部队指挥系统**

从"单个 Agent 工具"进化为"多个 Agent 协作团队"，每个 Agent 都是注入灵魂的 OpenClaw 实例，通过沙盒隔离实现真正的多 Agent 并行协作。

---

## 📅 已完成阶段

### Phase 1-3: 基础功能 ✅ (2026-03-17 完成)

| 阶段 | 功能 | 核心文件 | CLI 命令 |
|------|------|----------|----------|
| Phase 1 | 代码吞食 | `core/ingestion.py` | `kas ingest` |
| Phase 1 | Agent 合体 | `core/fusion.py` | `kas fuse` |
| Phase 1 | 对话系统 | `core/chat.py` | `kas chat` |
| Phase 1 | Agent 进化 | `core/llm_learning.py` | `kas evolve` |
| Phase 2 | Agent 市场 | `core/market.py` | `kas market publish/search/install` |
| Phase 2 | 能力验证 | `core/validation.py` | `kas validate` |
| Phase 2 | 统计面板 | `core/stats.py` | `kas stats` |
| Phase 3 | 知识库/RAG | `core/knowledge.py` | `kas knowledge` |
| Phase 3 | 工作流 | `core/workflow.py` | `kas workflow run` |
| Phase 3 | A/B 测试 | `core/abtest.py` | `kas abtest` |
| Phase 3 | 版本管理 | `core/versioning.py` | `kas versions/rollback/diff` |

### Phase 4.1-4.2: 装备与界面 ✅ (2026-03-18 完成)

| 阶段 | 功能 | 核心文件 | CLI 命令 |
|------|------|----------|----------|
| 4.1 | Web 界面 MVP | `web/app.py` | `kas web` / `kas dashboard` |
| 4.2 | 装备系统 | `core/equipment.py` | `kas equip list/add/remove` |

### Phase 4.3: 多模态输入 ✅ (2026-03-20 完成)

| 功能 | 核心文件 | 状态 |
|------|----------|------|
| 文件上传 API | `core/multimodal.py` | ✅ |
| PDF 文本提取 | PDFProcessor 类 | ✅ |
| 图片 OCR | ImageProcessor 类 | ✅ |
| 代码文件处理 | CodeProcessor 类 | ✅ |
| CLI --attach | `cli/main.py` | ✅ |

```bash
kas chat MyAgent "分析这份合同" --attach contract.pdf --attach photo.jpg
```

### Phase 4.4: OpenClaw 沙盒化 ✅ (2026-03-21 完成)

| 模块 | 功能 | 核心文件 | 状态 |
|------|------|----------|------|
| SoulInjector | KAS Agent → OpenClaw 配置转换 | `core/sandbox/__init__.py` | ✅ |
| MessageBus | 沙盒间文件队列通信 | `core/sandbox/message_bus.py` | ✅ |
| OpenClawSandbox | 沙盒包装器 | `core/sandbox/sandbox.py` | ✅ |
| SandboxSupervisor | 监督器+生命周期管理 | `core/sandbox/supervisor.py` | ✅ |

**架构**:
```
~/.kas/sandboxes/
└── ContractReviewCrew/
    ├── alice/              # Alice 的 OpenClaw 沙盒
    │   ├── SOUL.md         # 注入的灵魂 (system_prompt)
    │   ├── AGENTS.md       # 工作指南 (Crew协作规则)
    │   ├── TOOLS.md        # 装备配置
    │   ├── USER.md         # 用户信息
    │   ├── MEMORY.md       # 个人长期记忆
    │   └── memory/         # 每日记忆
    │
    ├── bob/                # Bob 的 OpenClaw 沙盒
    ├── carol/              # Carol 的 OpenClaw 沙盒
    │
    ├── shared/             # 共享资源
    │   ├── crew_memory.json    # Crew 共同记忆
    │   └── message_bus/        # 消息总线
    │
    └── supervisor.py       # 沙盒监督器
```

### Phase 4.5: Agent 特种部队 ✅ (2026-03-22 完成)

| 功能 | 核心文件 | CLI 命令 | 状态 |
|------|----------|----------|------|
| Crew 定义 | `core/crew_workflow.py` | `kas crew create` | ✅ |
| 动态协调员 | `core/sandbox/supervisor.py` | `kas crew elect/switch` | ✅ |
| 分层记忆 | `core/crew_memory.py` | - | ✅ |
| 沙盒注入 | `core/crew_demo.py` | `kas crew inject` | ✅ |
| 任务分发 | `cli/main.py` | `kas crew dispatch` | ✅ |

```bash
# Crew 管理
kas crew create ReviewTeam                    # 创建团队
kas crew list                                 # 列出团队
kas crew show ReviewTeam                      # 查看详情
kas crew inject ReviewTeam MyAgent            # 注入 Agent 到沙盒

# 执行任务
kas crew start ReviewTeam
kas crew dispatch ReviewTeam Alice "分析合同" --wait
```

### Phase 5.2: 分布式集群 ✅ (2026-03-23 完成)

| 模块 | 功能 | 核心文件 | 状态 |
|------|------|----------|------|
| 集群节点管理 | 节点注册、发现、心跳、Leader 选举 | `core/cluster/node.py` | ✅ |
| 集群管理器 | 负载均衡、故障检测和恢复 | `core/cluster/manager.py` | ✅ |
| 分布式调度器 | 任务分片和结果聚合 | `core/cluster/scheduler.py` | ✅ |
| 分布式状态存储 | 简化版 Raft 共识 | `core/cluster/state.py` | ✅ |
| 集成层 | 与通信协议集成 | `core/cluster/integration.py` | ✅ |

---

## 🚀 Phase 5: 企业级功能 (进行中)

**目标**: 生产环境可用

### 5.1 安全沙箱强化 🔒 ⏳
**优先级**: 🔴 P0

| 功能 | 状态 | 说明 |
|------|------|------|
| Docker 隔离 | ❌ | 代码执行沙箱隔离 |
| 网络访问限制 | ❌ | 代理模式控制外网访问 |
| 敏感信息过滤 | ❌ | API Key、密码等过滤 |
| 资源配额限制 | ❌ | CPU/内存/执行时间限制 |

### 5.3 团队协作 👥 ⏳
**优先级**: 🟡 P1

| 功能 | 状态 | 说明 |
|------|------|------|
| 多用户支持 | ❌ | 用户注册、登录 |
| 权限管理 (RBAC) | ❌ | 角色、权限控制 |
| Agent 共享机制 | ❌ | 团队内 Agent 共享 |
| 团队知识库 | ❌ | 共享 RAG 知识库 |

### 5.4 高级装备 🛠️ 🔄
**优先级**: 🟡 P1

| 功能 | 状态 | 说明 |
|------|------|------|
| 自定义 MCP Server | ❌ | 接入外部 MCP 服务 |
| Plugin 市场 | ❌ | 第三方插件市场 |
| 装备版本管理 | ❌ | 装备升级、回滚 |
| A/B 测试装备效果 | ✅ | 已有 abtest 模块 |

---

## 📊 当前进度

| 阶段 | 状态 | 完成度 |
|------|------|--------|
| Phase 1-3 基础功能 | ✅ | 100% |
| Phase 4.1 Web 界面 | ✅ | 100% |
| Phase 4.2 装备系统 | ✅ | 100% |
| Phase 4.3 多模态 | ✅ | 100% |
| Phase 4.4 沙盒化 | ✅ | 100% |
| Phase 4.5 特种部队 | ✅ | 100% |
| Phase 5.1 安全沙箱 | ⏳ | 0% |
| Phase 5.2 分布式集群 | ✅ | 100% |
| Phase 5.3 团队协作 | ⏳ | 0% |
| Phase 5.4 高级装备 | 🔄 | 25% |

**总体进度**: Phase 4 完整完成，Phase 5 进行中

**下一步**: Phase 5.1 安全沙箱强化

---

## 📈 代码统计

| 模块 | 文件 | 代码行数 |
|------|------|----------|
| CLI | `cli/main.py` | 2561 |
| 核心引擎 | `core/*.py` | ~9000 |
| 沙盒系统 | `core/sandbox/*.py` | ~1600 |
| 集群系统 | `core/cluster/*.py` | ~4000 |
| Web 界面 | `web/app.py` | ~460 |
| **总计** | - | **~18000** |

---

## 💡 关键设计决策

1. **沙盒化优先**: 没有沙盒的"Agent团队"只是工作流包装，必须先实现沙盒
2. **文件队列通信**: MessageBus 使用文件队列，比内存队列更可靠、可持久化
3. **OpenClaw 兼容**: 沙盒就是标准 OpenClaw 实例，可独立运行调试
4. **向后兼容**: 单 Agent 自动退化为 OpenClaw 模式
5. **分布式优先**: Phase 5.2 提前完成，为大规模部署做准备

---

## 🎯 成功标准

### Phase 5.1 (安全沙箱)
- [ ] Docker 容器隔离可用
- [ ] 网络访问白名单机制
- [ ] 敏感信息自动脱敏
- [ ] 资源限制可配置

### Phase 5.3 (团队协作)
- [ ] 多用户认证系统
- [ ] RBAC 权限控制
- [ ] Agent 共享和权限管理

### Phase 5.4 (高级装备)
- [ ] MCP Server 接入规范
- [ ] Plugin 打包和分发机制

---

## 📅 更新日志

### v1.3 (2026-03-23)
- ✅ Phase 4.3 多模态输入完成
- ✅ Phase 4.4 OpenClaw 沙盒化完成
- ✅ Phase 4.5 Agent 特种部队完成
- ✅ Phase 5.2 分布式集群完成
- 📝 ROADMAP 同步更新，反映实际进度
- 🔄 项目名称从 Kimi 更名为 Klaw

### v1.2 (2026-03-18)
- ✅ Phase 1-3 基础功能完成
- ✅ Phase 4.1 Web 界面完成
- ✅ Phase 4.2 装备系统完成

---

Author: Yilin.zhang | Date: 2026-03-23
