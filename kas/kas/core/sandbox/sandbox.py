"""
OpenClawSandbox - 沙盒包装器
管理单个 Agent 的 OpenClaw 实例生命周期
"""
import os
import json
import subprocess
import signal
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import logging
import time

from kas.core.sandbox.message_bus import MessageBus, Message, MessageType

logger = logging.getLogger(__name__)


@dataclass
class SandboxStatus:
    """沙盒状态"""
    name: str
    agent_name: str
    pid: Optional[int] = None
    state: str = "stopped"  # stopped, starting, running, error
    start_time: Optional[str] = None
    last_activity: Optional[str] = None
    message_count: int = 0
    error_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "agent_name": self.agent_name,
            "pid": self.pid,
            "state": self.state,
            "start_time": self.start_time,
            "last_activity": self.last_activity,
            "message_count": self.message_count,
            "error_count": self.error_count
        }


class OpenClawSandbox:
    """
    OpenClaw 沙盒包装器
    
    封装一个 OpenClaw 实例，提供:
    - 启动/停止生命周期管理
    - MessageBus 集成
    - 状态监控
    - 日志收集
    """
    
    def __init__(self, workspace_path: Path, agent_name: str, crew_path: Path):
        """
        Args:
            workspace_path: 沙盒工作目录
            agent_name: Agent 名称
            crew_path: Crew 根目录
        """
        self.workspace_path = Path(workspace_path)
        self.agent_name = agent_name
        self.crew_path = Path(crew_path)
        self.name = workspace_path.parent.name  # sandbox_name
        
        self.process: Optional[subprocess.Popen] = None
        self.message_bus: Optional[MessageBus] = None
        self.status = SandboxStatus(name=self.name, agent_name=agent_name)
        
        # 状态文件
        self.status_file = self.workspace_path / ".sandbox_status.json"
        
        # 消息处理器
        self._task_handler: Optional[Callable] = None
        self._question_handler: Optional[Callable] = None
    
    def start(self, use_mock: bool = False) -> bool:
        """
        启动沙盒
        
        方式1: 如果 OpenClaw 支持 headless 模式，启动子进程
        方式2: 否则，使用模拟模式（KAS 内部运行）
        """
        if self.status.state == "running":
            logger.warning(f"Sandbox {self.name} already running")
            return True
        
        self.status.state = "starting"
        self._save_status()
        
        try:
            # 初始化 MessageBus
            self.message_bus = MessageBus(self.crew_path, self.agent_name)
            
            # 启动消息监听
            self.message_bus.start_listener(self._on_message)
            
            # 启动模拟模式（实际 OpenClaw 集成待后续实现）
            self._start_mock_mode() if use_mock else self._start_openclaw_mode()
            
            self.status.state = "running"
            self.status.start_time = datetime.now().isoformat()
            self.status.pid = os.getpid()  # 模拟模式下是当前进程
            self._save_status()
            
            logger.info(f"Sandbox started: {self.name} ({self.agent_name})")
            return True
            
        except Exception as e:
            self.status.state = "error"
            self.status.error_count += 1
            self._save_status()
            logger.error(f"Failed to start sandbox {self.name}: {e}")
            return False
    
    def stop(self) -> bool:
        """停止沙盒"""
        if self.status.state == "stopped":
            return True
        
        try:
            # 停止消息监听
            if self.message_bus:
                self.message_bus.stop_listener()
            
            # 终止进程
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            
            self.status.state = "stopped"
            self.status.pid = None
            self._save_status()
            
            logger.info(f"Sandbox stopped: {self.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop sandbox {self.name}: {e}")
            return False
    
    def is_running(self) -> bool:
        """检查是否运行中"""
        if self.process:
            return self.process.poll() is None
        return self.status.state == "running"
    
    def send_task(self, task: str, context: Dict[str, Any] = None) -> str:
        """发送任务给沙盒"""
        if not self.message_bus:
            raise RuntimeError("Sandbox not started")
        
        # 发送到沙盒的 inbox
        msg_id = self.message_bus.send_task(
            to_agent=self.agent_name,
            task=task,
            context=context
        )
        
        self.status.message_count += 1
        self.status.last_activity = datetime.now().isoformat()
        self._save_status()
        
        return msg_id
    
    def send_message(self, to_agent: str, msg_type: MessageType, content: Dict) -> str:
        """发送消息"""
        if not self.message_bus:
            raise RuntimeError("Sandbox not started")
        
        msg_id = self.message_bus.send(to_agent, msg_type, content)
        self.status.message_count += 1
        self._save_status()
        return msg_id
    
    def receive_result(self, timeout: float = 30.0) -> Optional[Dict]:
        """等待并接收结果"""
        if not self.message_bus:
            raise RuntimeError("Sandbox not started")
        
        start = time.time()
        while time.time() - start < timeout:
            messages = self.message_bus.receive_all(MessageType.RESULT)
            for msg in messages:
                if msg.to_agent == self.agent_name:
                    return msg.content.get("result")
            time.sleep(0.1)
        
        return None
    
    def on_task(self, handler: Callable[[str, Dict], Any]):
        """注册任务处理器"""
        self._task_handler = handler
    
    def on_question(self, handler: Callable[[str, Dict], str]):
        """注册问题处理器"""
        self._question_handler = handler
    
    def _on_message(self, msg: Message):
        """内部消息处理"""
        self.status.last_activity = datetime.now().isoformat()
        
        try:
            if msg.type == MessageType.TASK.value and self._task_handler:
                task = msg.content.get("task", "")
                context = msg.content.get("context", {})
                result = self._task_handler(task, context)
                
                # 发送结果
                self.message_bus.send_result(
                    to_agent=msg.from_agent,
                    result=result,
                    task_id=msg.id
                )
            
            elif msg.type == MessageType.QUESTION.value and self._question_handler:
                question = msg.content.get("question", "")
                context = msg.content.get("context", {})
                answer = self._question_handler(question, context)
                
                # 发送回答
                self.message_bus.answer_question(msg, answer)
            
            self.status.message_count += 1
            
        except Exception as e:
            logger.error(f"Message handling error in {self.name}: {e}")
            self.status.error_count += 1
            
            # 发送错误
            if self.message_bus:
                self.message_bus.send(
                    to_agent=msg.from_agent,
                    msg_type=MessageType.ERROR,
                    content={"error": str(e), "original_msg_id": msg.id}
                )
        
        finally:
            self._save_status()
    
    def _start_mock_mode(self):
        """启动模拟模式（当前实现）"""
        # 在模拟模式下，Agent 逻辑在 KAS 进程中运行
        # 通过 MessageBus 进行通信
        logger.info(f"Sandbox {self.name} started in mock mode")
    
    def _start_openclaw_mode(self):
        """启动真正的 OpenClaw 模式（待实现）"""
        # TODO: 当 OpenClaw 支持 headless 模式时实现
        # 启动子进程运行 OpenClaw
        # 通过 stdio 或 socket 通信
        logger.info(f"Sandbox {self.name} would start OpenClaw (not yet implemented)")
        self._start_mock_mode()  # 暂时回退到模拟模式
    
    def _save_status(self):
        """保存状态到文件"""
        try:
            self.status_file.write_text(
                json.dumps(self.status.to_dict(), indent=2),
                encoding='utf-8'
            )
        except Exception as e:
            logger.error(f"Failed to save status: {e}")
    
    def load_status(self) -> Optional[SandboxStatus]:
        """从文件加载状态"""
        if self.status_file.exists():
            try:
                data = json.loads(self.status_file.read_text(encoding='utf-8'))
                self.status = SandboxStatus(**data)
                return self.status
            except Exception as e:
                logger.error(f"Failed to load status: {e}")
        return None
    
    def get_logs(self, lines: int = 100) -> List[str]:
        """获取日志（待实现）"""
        # TODO: 从日志文件读取
        return []
