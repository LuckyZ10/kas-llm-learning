"""
MessageBus - 沙盒间文件队列通信系统
使用文件系统实现可靠的消息传递
"""
import os
import json
import time
import fcntl
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import threading
import logging

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """消息类型"""
    TASK = "task"           # 分配任务
    RESULT = "result"       # 返回结果
    QUESTION = "question"   # 询问问题
    ANSWER = "answer"       # 回答问题
    STATUS = "status"       # 状态更新
    ERROR = "error"         # 错误报告


@dataclass
class Message:
    """消息结构"""
    id: str
    type: str
    from_agent: str
    to_agent: str
    content: Dict[str, Any]
    timestamp: str
    priority: int = 0  # 0=normal, 1=high, -1=low
    reply_to: Optional[str] = None  # 回复的消息ID
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "content": self.content,
            "timestamp": self.timestamp,
            "priority": self.priority,
            "reply_to": self.reply_to
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        return cls(**data)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        return cls.from_dict(json.loads(json_str))


class MessageBus:
    """
    消息总线 - 基于文件队列的异步通信
    
    目录结构:
    ~/.kas/sandboxes/{crew_name}/shared/message_bus/
    ├── inbox/          # 本沙盒收件箱
    │   ├── {msg_id}.json
    │   └── ...
    ├── outbox/         # 本沙盒发件箱
    │   └── ...
    ├── processed/      # 已处理消息
    └── broadcast/      # 广播消息
    
    使用文件锁保证并发安全
    """
    
    def __init__(self, crew_path: Path, agent_name: str):
        """
        Args:
            crew_path: Crew 根目录
            agent_name: 本 Agent 名称
        """
        self.crew_path = Path(crew_path)
        self.agent_name = agent_name
        self.bus_path = self.crew_path / "shared" / "message_bus"
        
        # 创建目录
        self.inbox = self.bus_path / "inbox"
        self.outbox = self.bus_path / "outbox"
        self.processed = self.bus_path / "processed"
        self.broadcast = self.bus_path / "broadcast"
        
        for d in [self.inbox, self.outbox, self.processed, self.broadcast]:
            d.mkdir(parents=True, exist_ok=True)
        
        self._running = False
        self._handlers: Dict[str, Callable] = {}
        self._listener_thread: Optional[threading.Thread] = None
    
    def send(self, to_agent: str, msg_type: MessageType, content: Dict[str, Any],
             priority: int = 0, reply_to: Optional[str] = None) -> str:
        """
        发送消息
        
        Args:
            to_agent: 目标 Agent 名称 (或 "broadcast" 广播)
            msg_type: 消息类型
            content: 消息内容
            priority: 优先级 (0=normal, 1=high, -1=low)
            reply_to: 回复的消息ID
        
        Returns:
            消息ID
        """
        msg_id = f"{self.agent_name}_{int(time.time() * 1000)}_{os.urandom(4).hex()}"
        
        msg = Message(
            id=msg_id,
            type=msg_type.value,
            from_agent=self.agent_name,
            to_agent=to_agent,
            content=content,
            timestamp=datetime.now().isoformat(),
            priority=priority,
            reply_to=reply_to
        )
        
        # 确定消息位置
        if to_agent == "broadcast":
            msg_path = self.broadcast / f"{msg_id}.json"
        else:
            # 发送到对方 inbox
            target_inbox = self.crew_path / to_agent / "shared" / "message_bus" / "inbox"
            target_inbox.mkdir(parents=True, exist_ok=True)
            msg_path = target_inbox / f"{msg_id}.json"
        
        # 写入文件（带锁）
        self._write_message(msg_path, msg)
        
        logger.debug(f"Message sent: {msg_id} from {self.agent_name} to {to_agent}")
        return msg_id
    
    def receive(self, timeout: Optional[float] = None) -> Optional[Message]:
        """
        接收一条消息（阻塞或非阻塞）
        
        Args:
            timeout: 超时时间（秒），None 表示非阻塞
        
        Returns:
            Message 或 None
        """
        start_time = time.time()
        
        while True:
            # 按优先级排序获取消息
            messages = self._get_pending_messages()
            
            if messages:
                msg_path, msg = messages[0]
                
                # 移动到处理中
                processed_path = self.processed / f"{msg.id}.json"
                self._move_message(msg_path, processed_path)
                
                logger.debug(f"Message received: {msg.id}")
                return msg
            
            # 检查超时
            if timeout is not None:
                if time.time() - start_time >= timeout:
                    return None
                time.sleep(0.1)  # 短暂休眠
            else:
                return None
    
    def receive_all(self, msg_type: Optional[MessageType] = None) -> List[Message]:
        """接收所有待处理消息"""
        messages = self._get_pending_messages()
        result = []
        
        for msg_path, msg in messages:
            if msg_type is None or msg.type == msg_type.value:
                processed_path = self.processed / f"{msg.id}.json"
                self._move_message(msg_path, processed_path)
                result.append(msg)
        
        return result
    
    def peek(self) -> List[Message]:
        """查看待处理消息（不移动）"""
        return [msg for _, msg in self._get_pending_messages()]
    
    def reply(self, original_msg: Message, content: Dict[str, Any],
              msg_type: MessageType = MessageType.ANSWER) -> str:
        """
        回复消息
        
        Args:
            original_msg: 原消息
            content: 回复内容
            msg_type: 回复类型（默认 ANSWER）
        
        Returns:
            新消息ID
        """
        return self.send(
            to_agent=original_msg.from_agent,
            msg_type=msg_type,
            content=content,
            reply_to=original_msg.id
        )
    
    def send_task(self, to_agent: str, task: str, context: Dict[str, Any] = None) -> str:
        """发送任务"""
        content = {
            "task": task,
            "context": context or {}
        }
        return self.send(to_agent, MessageType.TASK, content, priority=1)
    
    def send_result(self, to_agent: str, result: Any, task_id: Optional[str] = None) -> str:
        """发送结果"""
        content = {
            "result": result,
            "task_id": task_id
        }
        return self.send(to_agent, MessageType.RESULT, content)
    
    def ask_question(self, to_agent: str, question: str, context: Dict[str, Any] = None) -> str:
        """询问问题"""
        content = {
            "question": question,
            "context": context or {}
        }
        return self.send(to_agent, MessageType.QUESTION, content, priority=1)
    
    def answer_question(self, original_question: Message, answer: str) -> str:
        """回答问题"""
        content = {
            "answer": answer,
            "original_question": original_question.content.get("question", "")
        }
        return self.reply(original_question, content, MessageType.ANSWER)
    
    def broadcast(self, msg_type: MessageType, content: Dict[str, Any]) -> str:
        """广播消息给所有 Agent"""
        return self.send("broadcast", msg_type, content)
    
    def start_listener(self, handler: Callable[[Message], None], poll_interval: float = 0.5):
        """
        启动消息监听器（后台线程）
        
        Args:
            handler: 消息处理函数
            poll_interval: 轮询间隔（秒）
        """
        if self._running:
            logger.warning("Listener already running")
            return
        
        self._running = True
        self._handlers["default"] = handler
        
        def listen():
            while self._running:
                try:
                    msg = self.receive(timeout=poll_interval)
                    if msg:
                        handler(msg)
                except Exception as e:
                    logger.error(f"Message handler error: {e}")
        
        self._listener_thread = threading.Thread(target=listen, daemon=True)
        self._listener_thread.start()
        logger.info(f"Message listener started for {self.agent_name}")
    
    def stop_listener(self):
        """停止消息监听器"""
        self._running = False
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=2.0)
        logger.info(f"Message listener stopped for {self.agent_name}")
    
    def _get_pending_messages(self) -> List[tuple]:
        """获取待处理消息，按优先级排序"""
        messages = []
        
        # 读取 inbox
        for msg_file in self.inbox.glob("*.json"):
            try:
                msg = Message.from_json(msg_file.read_text(encoding='utf-8'))
                messages.append((msg_file, msg))
            except Exception as e:
                logger.error(f"Failed to read message {msg_file}: {e}")
        
        # 读取广播（只读取未处理的）
        for msg_file in self.broadcast.glob("*.json"):
            try:
                msg = Message.from_json(msg_file.read_text(encoding='utf-8'))
                # 检查是否已处理
                processed_marker = self.processed / f"broadcast_{msg.id}"
                if not processed_marker.exists():
                    messages.append((msg_file, msg))
            except Exception as e:
                logger.error(f"Failed to read broadcast {msg_file}: {e}")
        
        # 按优先级排序（高优先级在前）
        messages.sort(key=lambda x: -x[1].priority)
        
        return messages
    
    def _write_message(self, path: Path, msg: Message):
        """写入消息文件（带锁）"""
        with open(path, 'w', encoding='utf-8') as f:
            # 使用文件锁
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(msg.to_json())
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    
    def _move_message(self, src: Path, dst: Path):
        """移动消息文件"""
        try:
            # 如果是广播消息，创建处理标记
            if "broadcast" in str(src):
                marker = self.processed / f"broadcast_{src.stem}"
                marker.touch()
            
            # 实际移动
            src.rename(dst)
        except Exception as e:
            logger.error(f"Failed to move message {src} -> {dst}: {e}")
    
    def get_stats(self) -> Dict[str, int]:
        """获取消息统计"""
        return {
            "inbox": len(list(self.inbox.glob("*.json"))),
            "outbox": len(list(self.outbox.glob("*.json"))),
            "processed": len(list(self.processed.glob("*.json"))),
            "broadcast": len(list(self.broadcast.glob("*.json")))
        }
    
    def cleanup_old_messages(self, days: int = 7):
        """清理旧消息"""
        cutoff = time.time() - (days * 24 * 3600)
        
        for directory in [self.processed, self.outbox]:
            for msg_file in directory.glob("*.json"):
                try:
                    if msg_file.stat().st_mtime < cutoff:
                        msg_file.unlink()
                        logger.debug(f"Cleaned up old message: {msg_file}")
                except Exception as e:
                    logger.error(f"Failed to cleanup {msg_file}: {e}")
