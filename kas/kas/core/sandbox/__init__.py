"""
KAS 沙盒系统 - OpenClaw 集成
实现 KAS Agent 到 OpenClaw 沙盒的转换和运行
"""
import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class SandboxConfig:
    """沙盒配置"""
    name: str
    workspace_path: Path
    agent_name: str
    agent_version: str = "0.1.0"
    description: str = ""
    system_prompt: str = ""
    capabilities: List[Dict] = field(default_factory=list)
    equipment: List[str] = field(default_factory=list)
    memory_enabled: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class SoulInjector:
    """
    灵魂注入器 - 将 KAS Agent 转换为 OpenClaw 沙盒配置
    
    KAS Agent (agent.yaml) 
        ↓ SoulInjector
    OpenClaw 沙盒/
    ├── SOUL.md         # 灵魂 (system_prompt)
    ├── AGENTS.md       # 工作指南 (Crew协作)
    ├── TOOLS.md        # 装备配置
    ├── USER.md         # 用户信息
    └── memory/         # 记忆存储
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Args:
            base_path: 沙盒根目录，默认 ~/.kas/sandboxes/
        """
        if base_path is None:
            base_path = Path.home() / '.kas' / 'sandboxes'
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def inject(self, agent_config: Dict[str, Any], crew_context: Optional[Dict] = None) -> SandboxConfig:
        """
        注入灵魂，创建 OpenClaw 沙盒
        
        Args:
            agent_config: KAS Agent 配置 (从 agent.yaml 加载)
            crew_context: Crew 上下文信息 (可选)
        
        Returns:
            SandboxConfig: 沙盒配置
        """
        agent_name = agent_config['name']
        sandbox_name = crew_context.get('sandbox_name', agent_name) if crew_context else agent_name
        
        workspace_path = self.base_path / sandbox_name / agent_name
        workspace_path.mkdir(parents=True, exist_ok=True)
        
        config = SandboxConfig(
            name=sandbox_name,
            workspace_path=workspace_path,
            agent_name=agent_name,
            agent_version=agent_config.get('version', '0.1.0'),
            description=agent_config.get('description', ''),
            system_prompt=agent_config.get('system_prompt', ''),
            capabilities=agent_config.get('capabilities', []),
            equipment=agent_config.get('equipment', [])
        )
        
        # 创建 OpenClaw 配置文件
        self._create_soul_md(config)
        self._create_agents_md(config, crew_context)
        self._create_tools_md(config)
        self._create_user_md(config, crew_context)
        self._create_identity_md(config)
        self._init_memory(config)
        
        logger.info(f"Soul injected: {agent_name} -> {workspace_path}")
        return config
    
    def _create_soul_md(self, config: SandboxConfig):
        """创建 SOUL.md - Agent 的灵魂"""
        soul_content = f"""# SOUL.md - {config.agent_name}

## 身份

**名字**: {config.agent_name}
**emoji**: 🤖
**本质**: 从代码项目中提取能力孵化而成的 AI Agent

## 核心能力

"""
        # 添加能力描述
        for cap in config.capabilities:
            soul_content += f"- **{cap.get('name', 'Unknown')}**: {cap.get('description', '')}\n"
        
        soul_content += f"""
## 系统提示词

```
{config.system_prompt}
```

## 可用装备

"""
        for equip in config.equipment:
            soul_content += f"- {equip}\n"
        
        soul_content += f"""
## 记忆

你会记住：
- 用户的偏好和习惯
- 对话中的关键决策
- 你的错误和教训
- 用户的项目上下文

---

_这个 SOUL 是从 KAS Agent 「{config.agent_name}」注入的_
_你在 OpenClaw 中运行，但灵魂来自 KAS_
"""
        
        soul_path = config.workspace_path / "SOUL.md"
        soul_path.write_text(soul_content, encoding='utf-8')
    
    def _create_agents_md(self, config: SandboxConfig, crew_context: Optional[Dict]):
        """创建 AGENTS.md - Crew 工作指南"""
        content = f"""# AGENTS.md - {config.agent_name}

## 你是谁

你是 KAS Crew 的成员，专注于自己的专业领域。

## 工作原则

1. **保持专业**: 发挥你的核心能力
2. **协作沟通**: 通过 MessageBus 与队友协作
3. **持续学习**: 从每次任务中积累经验

## Crew 上下文

"""
        
        if crew_context:
            content += f"""
**Crew 名称**: {crew_context.get('crew_name', 'Unknown')}
**你的角色**: {crew_context.get('role', 'member')}

**队友**:
"""
            for member in crew_context.get('members', []):
                if member['name'] != config.agent_name:
                    content += f"- **{member['name']}**: {member.get('description', '')}\n"
            
            content += f"""
**协作方式**:
- 使用 `message_bus` 与其他 Agent 通信
- 任务分配通过 inbox 接收
- 结果通过 outbox 返回
"""
        else:
            content += "这是一个独立运行的 Agent。\n"
        
        content += """
## 文件组织

- SOUL.md - 你的灵魂和身份
- TOOLS.md - 可用装备
- USER.md - 用户信息
- MEMORY.md - 长期记忆
- memory/ - 每日记忆
"""
        
        agents_path = config.workspace_path / "AGENTS.md"
        agents_path.write_text(content, encoding='utf-8')
    
    def _create_tools_md(self, config: SandboxConfig):
        """创建 TOOLS.md - 装备配置"""
        content = f"""# TOOLS.md - {config.agent_name} 的装备

## 可用装备

"""
        # 装备池
        equipment_pool = {
            "web_search": "网络搜索",
            "file_reader": "文件读取",
            "code_analyzer": "代码分析",
            "git_analyzer": "Git 分析",
            "pdf_parser": "PDF 解析",
            "ocr": "OCR 文字识别",
            "image_analysis": "图片分析"
        }
        
        for equip in config.equipment:
            desc = equipment_pool.get(equip, "自定义装备")
            content += f"- **{equip}**: {desc}\n"
        
        content += """
## 使用方式

通过 message_bus 调用装备：
```
message_bus.use_equipment("web_search", {"query": "..."})
```

## MCP 装备

如有 MCP Server 配置，通过 equipment 系统加载
"""
        
        tools_path = config.workspace_path / "TOOLS.md"
        tools_path.write_text(content, encoding='utf-8')
    
    def _create_user_md(self, config: SandboxConfig, crew_context: Optional[Dict]):
        """创建 USER.md - 用户信息"""
        # 尝试从 Crew 上下文获取用户信息
        user_info = crew_context.get('user', {}) if crew_context else {}
        
        content = f"""# USER.md - 关于用户

## 基本信息

- **Name**: {user_info.get('name', 'Unknown')}
- **What to call them**: {user_info.get('preferred_name', 'you')}

## Context

_(What do they care about? What projects are they working on?)_

"""
        if user_info.get('projects'):
            content += f"""
## Projects

"""
            for project in user_info.get('projects', []):
                content += f"- {project}\n"
        
        content += """
---

The more you know, the better you can help.
"""
        
        user_path = config.workspace_path / "USER.md"
        user_path.write_text(content, encoding='utf-8')
    
    def _create_identity_md(self, config: SandboxConfig):
        """创建 IDENTITY.md"""
        content = f"""# IDENTITY.md - Who Am I?

- **Name**: {config.agent_name}
- **Creature**: KAS 孵化的 AI Agent
- **Vibe**: 专业、高效、协作

---

This isn't just metadata. It's the start of figuring out who you are.
"""
        
        identity_path = config.workspace_path / "IDENTITY.md"
        identity_path.write_text(content, encoding='utf-8')
    
    def _init_memory(self, config: SandboxConfig):
        """初始化记忆目录"""
        memory_dir = config.workspace_path / "memory"
        memory_dir.mkdir(exist_ok=True)
        
        # 创建空的 MEMORY.md
        memory_path = config.workspace_path / "MEMORY.md"
        if not memory_path.exists():
            memory_path.write_text("# 记忆档案\n\n_(Long-term memories)_\n", encoding='utf-8')
        
        # 创建今天的记忆文件
        today = datetime.now().strftime("%Y-%m-%d")
        today_path = memory_dir / f"{today}.md"
        if not today_path.exists():
            today_path.write_text(f"# {today}\n\n_(Today's activities)_\n", encoding='utf-8')
        
        # 创建 MESSAGE.md (用于启动问候)
        message_path = config.workspace_path / "MESSAGE.md"
        if not message_path.exists():
            message_path.write_text(f"欢迎使用 {config.agent_name}!\n", encoding='utf-8')
    
    def list_sandboxes(self) -> List[str]:
        """列出所有沙盒"""
        sandboxes = []
        if self.base_path.exists():
            for item in self.base_path.iterdir():
                if item.is_dir():
                    sandboxes.append(item.name)
        return sandboxes
    
    def get_sandbox_path(self, sandbox_name: str) -> Optional[Path]:
        """获取沙盒路径"""
        path = self.base_path / sandbox_name
        if path.exists():
            return path
        return None
    
    def delete_sandbox(self, sandbox_name: str) -> bool:
        """删除沙盒"""
        path = self.base_path / sandbox_name
        if path.exists():
            shutil.rmtree(path)
            logger.info(f"Sandbox deleted: {sandbox_name}")
            return True
        return False
