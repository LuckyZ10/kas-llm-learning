# KAS 测试计划

> 日期: 2026-03-24
> 状态: 待执行

---

## 测试环境准备

```bash
cd D:/vibe_Project/KAS/kas
pip install -e .
```

---

## 测试清单

### 1. 安装和配置
- [ ] `pip install -e .` - 安装项目
- [ ] `kas --help` - 查看帮助
- [ ] `kas config setup` - 配置向导
- [ ] `kas config status` - 查看配置状态

### 2. Agent 核心功能
- [ ] `kas ingest ./project --name TestAgent` - 代码吞食
- [ ] `kas list` - 查看 Agent 列表
- [ ] `kas inspect TestAgent` - 查看 Agent 详情
- [ ] `kas chat TestAgent` - Agent 对话

### 3. Agent 合体
- [ ] `kas fuse Agent1 Agent2 --name FusedAgent` - 合并 Agent
- [ ] `kas inspect FusedAgent` - 查看合体后能力

### 4. 特种部队 (Crew)
- [ ] `kas crew create TestCrew` - 创建团队
- [ ] `kas crew list` - 查看团队列表
- [ ] `kas crew show TestCrew` - 查看团队详情

### 5. 装备系统
- [ ] `kas equip list` - 查看装备列表
- [ ] `kas equip add web_search` - 添加装备
- [ ] `kas equip remove web_search` - 移除装备

### 6. 知识库
- [ ] `kas knowledge add ./docs` - 添加知识
- [ ] `kas knowledge list` - 查看知识库
- [ ] `kas knowledge search "query"` - 搜索知识

### 7. 工作流
- [ ] `kas workflow run workflow.yaml` - 执行工作流

### 8. Agent 市场
- [ ] `kas market search "keyword"` - 搜索 Agent
- [ ] `kas market publish TestAgent` - 发布 Agent

### 9. 安全模块
- [ ] 测试敏感信息过滤 (API Key、密码)
- [ ] 测试资源限制 (CPU、内存、超时)

### 10. 智谱 AI 集成
- [ ] 配置智谱 AI (需要确认正确的配置方式)
- [ ] 测试基本对话
- [ ] 测试代码生成

---

## 智谱 AI 配置 (待确认)

当前配置方式:
```bash
export ANTHROPIC_AUTH_TOKEN="your_api_key"
export ANTHROPIC_BASE_URL="https://open.bigmodel.cn/api/anthropic"
export API_TIMEOUT_MS="3000000"
```

**注意**: 需要确认这个配置是否正确。

---

## 项目信息

- 项目路径: `D:/vibe_Project/KAS`
- GitHub: `https://github.com/LuckyZ10/kas-llm-learning.git`
- 代码行数: ~25,000
- Python 文件: 63 个
- 测试状态: 19/19 通过

---

## 相关文档

- `kas/README.md` - 项目说明
- `kas/ROADMAP.md` - 开发路线图
- `kas/docs/ARCHITECTURE.md` - 架构文档
