"""
SandboxSupervisor - 沙盒监督器
管理所有沙盒的生命周期，支持动态协调员选举
"""
import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
import threading
import time

from kas.core.sandbox import SoulInjector, SandboxConfig
from kas.core.sandbox.sandbox import OpenClawSandbox, SandboxStatus
from kas.core.sandbox.message_bus import MessageBus, MessageType

logger = logging.getLogger(__name__)


@dataclass
class CrewConfig:
    """Crew 配置"""
    name: str
    description: str = ""
    members: List[Dict] = field(default_factory=list)
    workflow: List[Dict] = field(default_factory=list)
    coordinator: Optional[str] = None  # 协调员 Agent 名称
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "members": self.members,
            "workflow": self.workflow,
            "coordinator": self.coordinator,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CrewConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SandboxSupervisor:
    """
    沙盒监督器
    
    职责:
    1. 管理所有 Crew 和沙盒
    2. 启动/停止沙盒
    3. 选举/切换协调员
    4. 监控沙盒健康状态
    5. 全局消息路由
    
    目录结构:
    ~/.kas/sandboxes/
    └── {crew_name}/
        ├── {agent1}/          # Agent 沙盒
        ├── {agent2}/
        ├── shared/            # 共享资源
        │   ├── crew_memory.json
        │   └── message_bus/
        │       ├── broadcast/
        │       └── ...
        └── crew_config.json   # Crew 配置
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Args:
            base_path: 沙盒根目录，默认 ~/.kas/sandboxes
        """
        if base_path is None:
            base_path = Path.home() / '.kas' / 'sandboxes'
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # SoulInjector 用于创建沙盒
        self.injector = SoulInjector(self.base_path)
        
        # 活跃的沙盒
        self.sandboxes: Dict[str, OpenClawSandbox] = {}  # {crew_name}/{agent_name} -> sandbox
        
        # Crew 配置缓存
        self._crews: Dict[str, CrewConfig] = {}
        
        # 监控线程
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False
    
    # ==================== Crew 管理 ====================
    
    def create_crew(self, config: CrewConfig) -> Path:
        """
        创建新的 Crew
        
        Args:
            config: Crew 配置
        
        Returns:
            Crew 路径
        """
        crew_path = self.base_path / config.name
        crew_path.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        config_path = crew_path / "crew_config.json"
        config_path.write_text(
            json.dumps(config.to_dict(), indent=2, ensure_ascii=False),
            encoding='utf-8'
        )
        
        # 创建共享目录
        shared_path = crew_path / "shared"
        (shared_path / "message_bus" / "broadcast").mkdir(parents=True, exist_ok=True)
        
        # 创建 Crew 记忆
        crew_memory = shared_path / "crew_memory.json"
        crew_memory.write_text(
            json.dumps({
                "conversations": [],
                "tasks": [],
                "shared_knowledge": {}
            }, indent=2),
            encoding='utf-8'
        )
        
        self._crews[config.name] = config
        
        logger.info(f"Crew created: {config.name} with {len(config.members)} members")
        return crew_path
    
    def load_crew(self, crew_name: str) -> Optional[CrewConfig]:
        """加载 Crew 配置"""
        if crew_name in self._crews:
            return self._crews[crew_name]
        
        config_path = self.base_path / crew_name / "crew_config.json"
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text(encoding='utf-8'))
                config = CrewConfig.from_dict(data)
                self._crews[crew_name] = config
                return config
            except Exception as e:
                logger.error(f"Failed to load crew {crew_name}: {e}")
        
        return None
    
    def list_crews(self) -> List[str]:
        """列出所有 Crew"""
        crews = []
        for item in self.base_path.iterdir():
            if item.is_dir() and (item / "crew_config.json").exists():
                crews.append(item.name)
        return crews
    
    def delete_crew(self, crew_name: str) -> bool:
        """删除 Crew"""
        # 先停止所有沙盒
        self.stop_crew(crew_name)
        
        crew_path = self.base_path / crew_name
        if crew_path.exists():
            import shutil
            shutil.rmtree(crew_path)
            self._crews.pop(crew_name, None)
            logger.info(f"Crew deleted: {crew_name}")
            return True
        return False
    
    # ==================== 沙盒管理 ====================
    
    def inject_agent(self, crew_name: str, agent_config: Dict[str, Any]) -> Optional[SandboxConfig]:
        """
        将 Agent 注入到 Crew
        
        Args:
            crew_name: Crew 名称
            agent_config: Agent 配置
        
        Returns:
            沙盒配置
        """
        crew = self.load_crew(crew_name)
        if not crew:
            logger.error(f"Crew not found: {crew_name}")
            return None
        
        agent_name = agent_config['name']
        
        # 构建 Crew 上下文
        crew_context = {
            "crew_name": crew_name,
            "sandbox_name": crew_name,
            "role": self._get_agent_role(crew, agent_name),
            "members": crew.members,
            "user": {}  # 可以从配置加载
        }
        
        # 使用 SoulInjector 创建沙盒
        sandbox_config = self.injector.inject(agent_config, crew_context)
        
        logger.info(f"Agent {agent_name} injected into crew {crew_name}")
        return sandbox_config
    
    def start_sandbox(self, crew_name: str, agent_name: str, use_mock: bool = False) -> bool:
        """启动沙盒"""
        sandbox_key = f"{crew_name}/{agent_name}"
        
        # 检查是否已运行
        if sandbox_key in self.sandboxes:
            sandbox = self.sandboxes[sandbox_key]
            if sandbox.is_running():
                logger.warning(f"Sandbox {sandbox_key} already running")
                return True
        
        # 加载配置
        crew_path = self.base_path / crew_name
        workspace_path = crew_path / agent_name
        
        if not workspace_path.exists():
            logger.error(f"Sandbox workspace not found: {workspace_path}")
            return False
        
        # 创建并启动沙盒
        sandbox = OpenClawSandbox(workspace_path, agent_name, crew_path)
        
        if sandbox.start(use_mock=use_mock):
            self.sandboxes[sandbox_key] = sandbox
            return True
        
        return False
    
    def stop_sandbox(self, crew_name: str, agent_name: str) -> bool:
        """停止沙盒"""
        sandbox_key = f"{crew_name}/{agent_name}"
        
        if sandbox_key in self.sandboxes:
            sandbox = self.sandboxes[sandbox_key]
            result = sandbox.stop()
            if result:
                del self.sandboxes[sandbox_key]
            return result
        
        return True
    
    def start_crew(self, crew_name: str, use_mock: bool = False) -> Dict[str, bool]:
        """启动 Crew 的所有沙盒"""
        crew = self.load_crew(crew_name)
        if not crew:
            return {}
        
        results = {}
        for member in crew.members:
            agent_name = member['name']
            results[agent_name] = self.start_sandbox(crew_name, agent_name, use_mock)
        
        # 如果没有指定协调员，进行选举
        if not crew.coordinator:
            self.elect_coordinator(crew_name)
        
        logger.info(f"Crew {crew_name} started: {sum(results.values())}/{len(results)} sandboxes")
        return results
    
    def stop_crew(self, crew_name: str) -> bool:
        """停止 Crew 的所有沙盒"""
        to_stop = [key for key in self.sandboxes if key.startswith(f"{crew_name}/")]
        
        for sandbox_key in to_stop:
            sandbox = self.sandboxes[sandbox_key]
            sandbox.stop()
            del self.sandboxes[sandbox_key]
        
        logger.info(f"Crew {crew_name} stopped")
        return True
    
    def get_sandbox_status(self, crew_name: str, agent_name: str) -> Optional[SandboxStatus]:
        """获取沙盒状态"""
        sandbox_key = f"{crew_name}/{agent_name}"
        
        if sandbox_key in self.sandboxes:
            return self.sandboxes[sandbox_key].status
        
        # 尝试从文件加载
        workspace_path = self.base_path / crew_name / agent_name
        if workspace_path.exists():
            sandbox = OpenClawSandbox(workspace_path, agent_name, self.base_path / crew_name)
            return sandbox.load_status()
        
        return None
    
    def get_crew_status(self, crew_name: str) -> Dict[str, Any]:
        """获取 Crew 整体状态"""
        crew = self.load_crew(crew_name)
        if not crew:
            return {}
        
        statuses = {}
        for member in crew.members:
            agent_name = member['name']
            status = self.get_sandbox_status(crew_name, agent_name)
            statuses[agent_name] = status.to_dict() if status else {"state": "unknown"}
        
        return {
            "crew_name": crew_name,
            "coordinator": crew.coordinator,
            "members": len(crew.members),
            "sandboxes": statuses
        }
    
    # ==================== 协调员管理 ====================
    
    def elect_coordinator(self, crew_name: str) -> Optional[str]:
        """
        选举协调员
        
        策略:
        1. 优先选择标记为 coordinator 角色的 Agent
        2. 如果没有，选择第一个运行的 Agent
        3. 如果都没有，随机选择一个
        """
        crew = self.load_crew(crew_name)
        if not crew:
            return None
        
        # 寻找 coordinator 角色的 Agent
        for member in crew.members:
            if member.get('role') == 'coordinator':
                crew.coordinator = member['name']
                self._save_crew_config(crew)
                logger.info(f"Coordinator elected: {crew.coordinator} (by role)")
                return crew.coordinator
        
        # 寻找运行的 Agent
        running_agents = []
        for member in crew.members:
            agent_name = member['name']
            status = self.get_sandbox_status(crew_name, agent_name)
            if status and status.state == "running":
                running_agents.append(agent_name)
        
        if running_agents:
            crew.coordinator = running_agents[0]
        else:
            # 随机选择
            crew.coordinator = random.choice(crew.members)['name']
        
        self._save_crew_config(crew)
        logger.info(f"Coordinator elected: {crew.coordinator}")
        return crew.coordinator
    
    def switch_coordinator(self, crew_name: str, new_coordinator: str) -> bool:
        """切换协调员"""
        crew = self.load_crew(crew_name)
        if not crew:
            return False
        
        # 检查新协调员是否存在
        member_names = [m['name'] for m in crew.members]
        if new_coordinator not in member_names:
            logger.error(f"Agent {new_coordinator} not in crew {crew_name}")
            return False
        
        old_coordinator = crew.coordinator
        crew.coordinator = new_coordinator
        self._save_crew_config(crew)
        
        # 广播协调员变更
        self._broadcast_to_crew(
            crew_name,
            MessageType.STATUS,
            {
                "event": "coordinator_changed",
                "from": old_coordinator,
                "to": new_coordinator
            }
        )
        
        logger.info(f"Coordinator switched: {old_coordinator} -> {new_coordinator}")
        return True
    
    # ==================== 任务分发 ====================
    
    def dispatch_task(self, crew_name: str, agent_name: str, task: str,
                     context: Dict[str, Any] = None, wait_result: bool = False,
                     timeout: float = 30.0) -> Optional[Any]:
        """
        分发任务给指定 Agent
        
        Args:
            crew_name: Crew 名称
            agent_name: Agent 名称
            task: 任务描述
            context: 任务上下文
            wait_result: 是否等待结果
            timeout: 等待超时
        
        Returns:
            如果 wait_result=True，返回结果；否则返回 None
        """
        sandbox_key = f"{crew_name}/{agent_name}"
        
        if sandbox_key not in self.sandboxes:
            # 尝试启动
            if not self.start_sandbox(crew_name, agent_name):
                logger.error(f"Failed to start sandbox: {sandbox_key}")
                return None
        
        sandbox = self.sandboxes[sandbox_key]
        msg_id = sandbox.send_task(task, context)
        
        if wait_result:
            return sandbox.receive_result(timeout=timeout)
        
        return msg_id
    
    def broadcast_task(self, crew_name: str, task: str, context: Dict[str, Any] = None):
        """广播任务给所有 Agent"""
        crew = self.load_crew(crew_name)
        if not crew:
            return
        
        for member in crew.members:
            agent_name = member['name']
            self.dispatch_task(crew_name, agent_name, task, context, wait_result=False)
    
    # ==================== 内部方法 ====================
    
    def _get_agent_role(self, crew: CrewConfig, agent_name: str) -> str:
        """获取 Agent 在 Crew 中的角色"""
        for member in crew.members:
            if member['name'] == agent_name:
                return member.get('role', 'member')
        return 'member'
    
    def _save_crew_config(self, crew: CrewConfig):
        """保存 Crew 配置"""
        config_path = self.base_path / crew.name / "crew_config.json"
        config_path.write_text(
            json.dumps(crew.to_dict(), indent=2, ensure_ascii=False),
            encoding='utf-8'
        )
        self._crews[crew.name] = crew
    
    def _broadcast_to_crew(self, crew_name: str, msg_type: MessageType, content: Dict):
        """向 Crew 广播消息"""
        crew = self.load_crew(crew_name)
        if not crew:
            return
        
        for member in crew.members:
            agent_name = member['name']
            sandbox_key = f"{crew_name}/{agent_name}"
            
            if sandbox_key in self.sandboxes:
                sandbox = self.sandboxes[sandbox_key]
                sandbox.send_message("broadcast", msg_type, content)
    
    def start_monitor(self, interval: float = 5.0):
        """启动监控线程"""
        if self._running:
            return
        
        self._running = True
        
        def monitor():
            while self._running:
                try:
                    # 检查沙盒健康状态
                    for sandbox_key, sandbox in list(self.sandboxes.items()):
                        if not sandbox.is_running():
                            logger.warning(f"Sandbox {sandbox_key} not running, attempting restart")
                            # 可以尝试自动重启
                    
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Monitor error: {e}")
        
        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()
        logger.info(f"Sandbox monitor started (interval: {interval}s)")
    
    def stop_monitor(self):
        """停止监控线程"""
        self._running = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
