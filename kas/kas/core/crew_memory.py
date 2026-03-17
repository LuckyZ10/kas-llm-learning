"""
KAS Crew 分层记忆系统
Crew共享记忆 + Agent个人记忆 + 任务上下文
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class CrewMemory:
    """Crew 共享记忆"""
    conversations: List[Dict] = field(default_factory=list)
    tasks: List[Dict] = field(default_factory=list)
    shared_knowledge: Dict[str, Any] = field(default_factory=dict)
    agent_summaries: Dict[str, str] = field(default_factory=dict)
    
    def add_conversation(self, agent_name: str, role: str, content: str, 
                         message_type: str = "message"):
        """添加对话记录"""
        self.conversations.append({
            "agent": agent_name,
            "role": role,
            "content": content,
            "type": message_type,
            "timestamp": datetime.now().isoformat()
        })
        # 保持最近 100 条
        if len(self.conversations) > 100:
            self.conversations = self.conversations[-100:]
    
    def add_task(self, task_id: str, description: str, assigned_to: str,
                 status: str = "pending", result: Any = None):
        """添加任务记录"""
        self.tasks.append({
            "id": task_id,
            "description": description,
            "assigned_to": assigned_to,
            "status": status,
            "result": result,
            "created_at": datetime.now().isoformat(),
            "completed_at": None
        })
    
    def update_task(self, task_id: str, status: str, result: Any = None):
        """更新任务状态"""
        for task in self.tasks:
            if task["id"] == task_id:
                task["status"] = status
                task["result"] = result
                if status in ["completed", "failed"]:
                    task["completed_at"] = datetime.now().isoformat()
                break
    
    def set_knowledge(self, key: str, value: Any):
        """设置共享知识"""
        self.shared_knowledge[key] = {
            "value": value,
            "updated_at": datetime.now().isoformat()
        }
    
    def get_knowledge(self, key: str) -> Optional[Any]:
        """获取共享知识"""
        entry = self.shared_knowledge.get(key)
        return entry["value"] if entry else None
    
    def to_dict(self) -> Dict:
        return {
            "conversations": self.conversations,
            "tasks": self.tasks,
            "shared_knowledge": self.shared_knowledge,
            "agent_summaries": self.agent_summaries
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CrewMemory':
        return cls(
            conversations=data.get("conversations", []),
            tasks=data.get("tasks", []),
            shared_knowledge=data.get("shared_knowledge", {}),
            agent_summaries=data.get("agent_summaries", {})
        )


class LayeredMemory:
    """
    分层记忆系统
    
    三层架构:
    1. Crew 层 (共享) - 对话历史、任务记录、共享知识
    2. Agent 层 (个人) - 个人记忆、专业背景
    3. Task 层 (临时) - 当前任务上下文、中间结果
    
    目录结构:
    ~/.kas/sandboxes/{crew}/
    ├── shared/
    │   └── crew_memory.json    # Crew 层
    ├── {agent1}/
    │   ├── MEMORY.md           # Agent 层 (长期)
    │   └── memory/             # Agent 层 (日常)
    │       └── YYYY-MM-DD.md
    └── ...
    """
    
    def __init__(self, crew_path: Path, agent_name: Optional[str] = None):
        """
        Args:
            crew_path: Crew 根目录
            agent_name: Agent 名称 (可选，用于 Agent 层访问)
        """
        self.crew_path = Path(crew_path)
        self.agent_name = agent_name
        
        # Crew 层
        self.crew_memory_path = self.crew_path / "shared" / "crew_memory.json"
        self._crew_memory: Optional[CrewMemory] = None
        
        # Agent 层
        if agent_name:
            self.agent_path = self.crew_path / agent_name
            self.memory_path = self.agent_path / "MEMORY.md"
            self.daily_memory_dir = self.agent_path / "memory"
    
    # ==================== Crew 层 ====================
    
    @property
    def crew_memory(self) -> CrewMemory:
        """获取 Crew 记忆 (懒加载)"""
        if self._crew_memory is None:
            self._crew_memory = self._load_crew_memory()
        return self._crew_memory
    
    def _load_crew_memory(self) -> CrewMemory:
        """从文件加载 Crew 记忆"""
        if self.crew_memory_path.exists():
            try:
                with open(self.crew_memory_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return CrewMemory.from_dict(data)
            except Exception as e:
                logger.error(f"Failed to load crew memory: {e}")
        
        return CrewMemory()
    
    def save_crew_memory(self):
        """保存 Crew 记忆到文件"""
        try:
            with open(self.crew_memory_path, 'w', encoding='utf-8') as f:
                json.dump(self.crew_memory.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save crew memory: {e}")
    
    def add_to_crew_conversation(self, agent_name: str, role: str, content: str,
                                  message_type: str = "message"):
        """添加对话到 Crew 记忆"""
        self.crew_memory.add_conversation(agent_name, role, content, message_type)
        self.save_crew_memory()
    
    def add_to_crew_tasks(self, task_id: str, description: str, assigned_to: str):
        """添加任务到 Crew 记忆"""
        self.crew_memory.add_task(task_id, description, assigned_to)
        self.save_crew_memory()
    
    def update_crew_task(self, task_id: str, status: str, result: Any = None):
        """更新 Crew 任务状态"""
        self.crew_memory.update_task(task_id, status, result)
        self.save_crew_memory()
    
    def get_crew_conversation_history(self, limit: int = 10) -> List[Dict]:
        """获取 Crew 对话历史"""
        return self.crew_memory.conversations[-limit:]
    
    def get_crew_task_history(self) -> List[Dict]:
        """获取 Crew 任务历史"""
        return self.crew_memory.tasks
    
    def set_crew_knowledge(self, key: str, value: Any):
        """设置 Crew 共享知识"""
        self.crew_memory.set_knowledge(key, value)
        self.save_crew_memory()
    
    def get_crew_knowledge(self, key: str) -> Optional[Any]:
        """获取 Crew 共享知识"""
        return self.crew_memory.get_knowledge(key)
    
    def build_crew_context(self, for_agent: Optional[str] = None) -> str:
        """
        构建 Crew 上下文摘要
        
        用于 Agent 启动时加载，了解 Crew 当前状态
        """
        lines = ["# Crew 上下文\n"]
        
        # 最近对话
        if self.crew_memory.conversations:
            lines.append("\n## 最近对话\n")
            for conv in self.crew_memory.conversations[-5:]:
                lines.append(f"- **[{conv['agent']}]** {conv['role']}: {conv['content'][:100]}...\n")
        
        # 活跃任务
        active_tasks = [t for t in self.crew_memory.tasks if t["status"] in ["pending", "running"]]
        if active_tasks:
            lines.append("\n## 活跃任务\n")
            for task in active_tasks[-5:]:
                lines.append(f"- [{task['status']}] {task['description']} (→ {task['assigned_to']})\n")
        
        # 共享知识
        if self.crew_memory.shared_knowledge:
            lines.append("\n## 共享知识\n")
            for key in list(self.crew_memory.shared_knowledge.keys())[:5]:
                value = self.get_crew_knowledge(key)
                preview = str(value)[:100] if value else "None"
                lines.append(f"- **{key}**: {preview}...\n")
        
        return "".join(lines)
    
    # ==================== Agent 层 ====================
    
    def read_agent_memory(self) -> str:
        """读取 Agent 长期记忆"""
        if not self.agent_name or not self.memory_path.exists():
            return ""
        
        try:
            return self.memory_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to read agent memory: {e}")
            return ""
    
    def read_agent_daily_memory(self, date: Optional[str] = None) -> str:
        """读取 Agent 每日记忆"""
        if not self.agent_name:
            return ""
        
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        daily_path = self.daily_memory_dir / f"{date}.md"
        
        if daily_path.exists():
            try:
                return daily_path.read_text(encoding='utf-8')
            except Exception as e:
                logger.error(f"Failed to read daily memory: {e}")
        
        return ""
    
    def write_agent_memory(self, content: str, append: bool = False):
        """写入 Agent 长期记忆"""
        if not self.agent_name:
            return
        
        try:
            if append and self.memory_path.exists():
                existing = self.memory_path.read_text(encoding='utf-8')
                content = existing + "\n\n" + content
            
            self.memory_path.write_text(content, encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to write agent memory: {e}")
    
    def append_to_daily_memory(self, content: str):
        """追加到每日记忆"""
        if not self.agent_name:
            return
        
        date = datetime.now().strftime("%Y-%m-%d")
        daily_path = self.daily_memory_dir / f"{date}.md"
        
        try:
            self.daily_memory_dir.mkdir(parents=True, exist_ok=True)
            
            if daily_path.exists():
                existing = daily_path.read_text(encoding='utf-8')
                content = existing + "\n\n" + content
            else:
                content = f"# {date}\n\n" + content
            
            daily_path.write_text(content, encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to append to daily memory: {e}")
    
    def build_agent_context(self) -> str:
        """
        构建 Agent 个人上下文
        
        包括长期记忆和今日记忆
        """
        if not self.agent_name:
            return ""
        
        lines = [f"# {self.agent_name} 的个人记忆\n"]
        
        # 长期记忆（只取最近的部分）
        long_term = self.read_agent_memory()
        if long_term:
            lines.append("\n## 长期记忆\n")
            lines.append(long_term[:2000])  # 限制长度
            if len(long_term) > 2000:
                lines.append("\n... (更多记忆已省略)")
        
        # 今日记忆
        daily = self.read_agent_daily_memory()
        if daily:
            lines.append("\n## 今日记录\n")
            lines.append(daily)
        
        return "".join(lines)
    
    # ==================== Task 层 ====================
    
    def build_task_context(self, task_description: str, 
                          previous_results: Dict[str, Any] = None) -> str:
        """
        构建任务上下文
        
        包括当前任务描述和前置任务结果
        """
        lines = ["# 当前任务\n", f"\n{task_description}\n"]
        
        if previous_results:
            lines.append("\n## 前置任务结果\n")
            for task_id, result in previous_results.items():
                lines.append(f"\n### {task_id}\n")
                result_str = str(result)[:500] if result else "None"
                lines.append(f"```\n{result_str}\n```\n")
        
        return "".join(lines)
    
    # ==================== 综合上下文 ====================
    
    def build_full_context(self, task_description: str,
                          previous_results: Dict[str, Any] = None,
                          include_crew: bool = True,
                          include_agent: bool = True) -> str:
        """
        构建完整上下文
        
        按层次整合：
        1. Crew 层（可选）
        2. Agent 层（可选）
        3. Task 层（必须）
        """
        parts = []
        
        if include_crew:
            crew_context = self.build_crew_context()
            if crew_context:
                parts.append(crew_context)
        
        if include_agent and self.agent_name:
            agent_context = self.build_agent_context()
            if agent_context:
                parts.append(agent_context)
        
        task_context = self.build_task_context(task_description, previous_results)
        parts.append(task_context)
        
        return "\n---\n".join(parts)
