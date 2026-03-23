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
from kas.core.security.sensitive_filter import SensitiveInfoFilter, FilterConfig

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
        
        self._sensitive_filter = SensitiveInfoFilter(FilterConfig(
            mask_email=False,
            mask_phone=False
        ))
        
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
                
                if isinstance(result, str):
                    result = self._sensitive_filter.filter(result)
                elif isinstance(result, dict):
                    result = self._sensitive_filter.filter_dict(result)
                
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
                
                if isinstance(answer, str):
                    answer = self._sensitive_filter.filter(answer)
                
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
        logger.info(f"Sandbox {self.name} started in mock mode")
    
    def filter_output(self, text: str) -> str:
        """Filter sensitive information from output text"""
        return self._sensitive_filter.filter(text)
    
    def filter_log(self, text: str) -> str:
        """Filter sensitive information from log text"""
        return self._sensitive_filter.filter(text)
    
    def _start_openclaw_mode(self):
        """
        启动真正的 OpenClaw 沙盒
        
        由于 OpenClaw 是命令行工具，沙盒模式工作原理:
        1. 启动消息监听线程（监控 MessageBus 文件队列）
        2. 当收到 TASK 消息时，调用 openclaw agent 执行
        3. 将 OpenClaw 的输出通过 MessageBus 返回
        """
        try:
            # 检查 openclaw 是否可用
            result = subprocess.run(
                ["openclaw", "--version"],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("openclaw not available")
            
            # 准备 OpenClaw 工作目录（配置文件已存在）
            openclaw_workspace = self.workspace_path / ".openclaw"
            openclaw_workspace.mkdir(exist_ok=True)
            
            # 日志文件
            log_file = openclaw_workspace / "openclaw.log"
            self._log_file = log_file
            
            # 注册真实任务处理器（调用 openclaw agent）
            self.on_task(self._execute_with_openclaw)
            
            logger.info(f"Sandbox {self.name} started in OpenClaw mode (message-driven)")
            
        except FileNotFoundError:
            logger.error("openclaw command not found, falling back to mock mode")
            self._start_mock_mode()
        except Exception as e:
            logger.error(f"Failed to start OpenClaw mode: {e}, falling back to mock mode")
            self._start_mock_mode()
    
    def _execute_with_openclaw(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用 OpenClaw 执行任务
        
        调用: openclaw agent --local --message "task" --json
        """
        import tempfile
        import json
        
        try:
            # 构建系统提示词（从 SOUL.md）
            soul_content = ""
            soul_file = self.workspace_path / "SOUL.md"
            if soul_file.exists():
                soul_content = soul_file.read_text(encoding='utf-8')
            
            # 构建完整提示词
            full_prompt = f"""{soul_content}

## 当前任务
{task}

## 上下文
{json.dumps(context, indent=2, ensure_ascii=False)}
"""
            
            # 写入临时文件（避免命令行长度限制）
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(full_prompt)
                prompt_file = f.name
            
            try:
                # 调用 OpenClaw
                cmd = [
                    "openclaw", "agent",
                    "--local",           # 本地运行，不使用 Gateway
                    "--message", f"@{prompt_file}",  # 从文件读取消息
                    "--json",            # JSON 输出
                    "--timeout", "120",  # 2分钟超时
                ]
                
                # 执行命令
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=130,
                    cwd=str(self.workspace_path)
                )
                
                # 解析结果
                output = result.stdout
                error = result.stderr
                
                # 尝试解析 JSON
                try:
                    response = json.loads(output)
                    return {
                        "status": "success",
                        "output": self._sensitive_filter.filter(response.get("message", output)),
                        "exit_code": result.returncode
                    }
                except json.JSONDecodeError:
                    # 非 JSON 输出
                    return {
                        "status": "success" if result.returncode == 0 else "error",
                        "output": self._sensitive_filter.filter(output),
                        "error": self._sensitive_filter.filter(error) if error else None,
                        "exit_code": result.returncode
                    }
                
            finally:
                # 清理临时文件
                try:
                    os.unlink(prompt_file)
                except:
                    pass
                
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "error": "OpenClaw execution timeout (120s)"
            }
        except Exception as e:
            logger.error(f"OpenClaw execution error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
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
    
    def get_logs(self, lines: int = 100, filter_sensitive: bool = True) -> List[str]:
        """获取 OpenClaw 日志"""
        log_file = getattr(self, '_log_file', None)
        if not log_file or not log_file.exists():
            # 尝试默认路径
            log_file = self.workspace_path / ".openclaw" / "openclaw.log"
        
        if not log_file.exists():
            return []
        
        try:
            # 读取最后 N 行
            with open(log_file, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                result = all_lines[-lines:] if len(all_lines) > lines else all_lines
                
                if filter_sensitive:
                    result = [self._sensitive_filter.filter(line) for line in result]
                
                return result
        except Exception as e:
            logger.error(f"Failed to read logs: {e}")
            return []
