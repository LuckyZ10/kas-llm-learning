"""
KAS Agent Communication Protocol - 智能体通信协议
消息协议定义模块

提供标准化的Agent间消息传递机制，支持多种序列化格式和消息类型。
"""
import json
import uuid
import gzip
import struct
from enum import Enum, auto
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# 可选依赖：MessagePack
try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False
    logger.debug("msgpack not available, falling back to JSON")

# 可选依赖：加密
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.debug("cryptography not available, encryption disabled")


class MessageType(Enum):
    """消息类型枚举"""
    REQUEST = "request"       # 请求消息
    RESPONSE = "response"     # 响应消息
    EVENT = "event"           # 事件消息
    ERROR = "error"           # 错误消息
    HEARTBEAT = "heartbeat"   # 心跳消息
    ACK = "ack"               # 确认消息
    BROADCAST = "broadcast"   # 广播消息


class MessagePriority(Enum):
    """消息优先级枚举"""
    CRITICAL = 0    # 紧急 - 立即处理
    HIGH = 1        # 高优先级
    NORMAL = 2      # 普通优先级
    LOW = 3         # 低优先级
    BACKGROUND = 4  # 后台任务


class SerializationFormat(Enum):
    """序列化格式枚举"""
    JSON = "json"
    MESSAGEPACK = "msgpack"


@dataclass
class Message:
    """
    消息基类
    
    所有Agent间通信消息的基础结构，支持序列化/反序列化、
    压缩和加密。
    
    Attributes:
        id: 消息唯一标识符
        type: 消息类型
        sender: 发送者标识
        receiver: 接收者标识（None表示广播）
        payload: 消息负载数据
        timestamp: 消息创建时间戳
        priority: 消息优先级
        correlation_id: 关联ID（用于请求-响应配对）
        ttl: 生存时间（秒），None表示永久
        headers: 消息头信息
        compressed: 是否已压缩
        encrypted: 是否已加密
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.EVENT
    sender: str = ""
    receiver: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: Optional[str] = None
    ttl: Optional[int] = None
    headers: Dict[str, str] = field(default_factory=dict)
    compressed: bool = False
    encrypted: bool = False
    
    def __post_init__(self):
        """初始化后处理"""
        if isinstance(self.type, str):
            self.type = MessageType(self.type)
        if isinstance(self.priority, str):
            self.priority = MessagePriority[self.priority]
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "type": self.type.value,
            "sender": self.sender,
            "receiver": self.receiver,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.name,
            "correlation_id": self.correlation_id,
            "ttl": self.ttl,
            "headers": self.headers,
            "compressed": self.compressed,
            "encrypted": self.encrypted
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """从字典创建消息"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=MessageType(data.get("type", "event")),
            sender=data.get("sender", ""),
            receiver=data.get("receiver"),
            payload=data.get("payload", {}),
            timestamp=datetime.fromisoformat(data.get("timestamp")) if data.get("timestamp") else datetime.now(),
            priority=MessagePriority[data.get("priority", "NORMAL")],
            correlation_id=data.get("correlation_id"),
            ttl=data.get("ttl"),
            headers=data.get("headers", {}),
            compressed=data.get("compressed", False),
            encrypted=data.get("encrypted", False)
        )
    
    def serialize(self, format: SerializationFormat = SerializationFormat.JSON,
                  compress: bool = False, encryption_key: Optional[bytes] = None) -> bytes:
        """
        序列化消息
        
        Args:
            format: 序列化格式
            compress: 是否启用压缩
            encryption_key: 加密密钥（None表示不加密）
        
        Returns:
            序列化后的字节数据
        """
        # 基础序列化
        if format == SerializationFormat.MESSAGEPACK and MSGPACK_AVAILABLE:
            data = msgpack.packb(self.to_dict(), use_bin_type=True)
            format_byte = b'\x01'  # MessagePack标识
        else:
            data = json.dumps(self.to_dict(), ensure_ascii=False).encode('utf-8')
            format_byte = b'\x00'  # JSON标识
        
        # 压缩
        if compress:
            data = gzip.compress(data)
            self.compressed = True
        
        # 加密
        if encryption_key and CRYPTO_AVAILABLE:
            fernet = self._get_fernet(encryption_key)
            data = fernet.encrypt(data)
            self.encrypted = True
        elif encryption_key and not CRYPTO_AVAILABLE:
            logger.warning("Encryption requested but cryptography not available")
        
        # 添加头部：格式标识(1字节) + 压缩标识(1字节) + 加密标识(1字节) + 数据长度(4字节)
        header = format_byte
        header += b'\x01' if self.compressed else b'\x00'
        header += b'\x01' if self.encrypted else b'\x00'
        header += struct.pack('>I', len(data))
        
        return header + data
    
    @classmethod
    def deserialize(cls, data: bytes, encryption_key: Optional[bytes] = None) -> 'Message':
        """
        反序列化消息
        
        Args:
            data: 序列化后的字节数据
            encryption_key: 解密密钥（如果消息已加密）
        
        Returns:
            Message对象
        """
        if len(data) < 7:
            raise ValueError("Invalid message data: too short")
        
        # 解析头部
        format_byte = data[0:1]
        compressed = data[1:2] == b'\x01'
        encrypted = data[2:3] == b'\x01'
        data_len = struct.unpack('>I', data[3:7])[0]
        payload = data[7:7+data_len]
        
        # 解密
        if encrypted:
            if not CRYPTO_AVAILABLE:
                raise RuntimeError("Message is encrypted but cryptography not available")
            if not encryption_key:
                raise ValueError("Message is encrypted but no key provided")
            fernet = cls._get_fernet(encryption_key)
            payload = fernet.decrypt(payload)
        
        # 解压
        if compressed:
            payload = gzip.decompress(payload)
        
        # 反序列化
        if format_byte == b'\x01' and MSGPACK_AVAILABLE:
            msg_dict = msgpack.unpackb(payload, raw=False)
        else:
            msg_dict = json.loads(payload.decode('utf-8'))
        
        return cls.from_dict(msg_dict)
    
    @staticmethod
    def _get_fernet(key: bytes) -> 'Fernet':
        """从密钥创建Fernet实例"""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography not available")
        # 确保密钥是32字节base64编码
        if len(key) != 32:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'kas_protocol_salt',  # 固定salt（生产环境应使用随机salt并存储）
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(key))
        return Fernet(key)
    
    def create_response(self, payload: Dict[str, Any], 
                        msg_type: MessageType = MessageType.RESPONSE) -> 'Message':
        """
        创建响应消息
        
        Args:
            payload: 响应负载
            msg_type: 响应类型
        
        Returns:
            新的响应消息
        """
        return Message(
            type=msg_type,
            sender=self.receiver or "",  # 原接收者变为发送者
            receiver=self.sender,         # 原发送者变为接收者
            payload=payload,
            priority=self.priority,
            correlation_id=self.id,       # 关联到原消息
            headers={"in_reply_to": self.id, **self.headers}
        )
    
    def create_ack(self) -> 'Message':
        """创建确认消息"""
        return Message(
            type=MessageType.ACK,
            sender=self.receiver or "",
            receiver=self.sender,
            payload={"acknowledged_msg_id": self.id},
            priority=MessagePriority.HIGH,
            correlation_id=self.id
        )
    
    def create_error(self, error_code: str, error_message: str, 
                     details: Optional[Dict] = None) -> 'Message':
        """
        创建错误响应消息
        
        Args:
            error_code: 错误代码
            error_message: 错误信息
            details: 详细错误信息
        """
        return Message(
            type=MessageType.ERROR,
            sender=self.receiver or "",
            receiver=self.sender,
            payload={
                "error_code": error_code,
                "error_message": error_message,
                "details": details or {},
                "original_msg_id": self.id
            },
            priority=MessagePriority.HIGH,
            correlation_id=self.id
        )
    
    def is_expired(self) -> bool:
        """检查消息是否已过期"""
        if self.ttl is None:
            return False
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > self.ttl
    
    def __repr__(self) -> str:
        return (f"Message(id={self.id[:8]}..., type={self.type.value}, "
                f"sender={self.sender}, receiver={self.receiver}, "
                f"priority={self.priority.name})")


class MessageBuilder:
    """
    消息构建器
    
    提供流畅的API构建消息
    
    Example:
        msg = (MessageBuilder()
               .request()
               .from_agent("agent1")
               .to_agent("agent2")
               .with_payload({"action": "process", "data": "..."})
               .with_priority(MessagePriority.HIGH)
               .build())
    """
    
    def __init__(self):
        self._id = str(uuid.uuid4())
        self._type = MessageType.EVENT
        self._sender = ""
        self._receiver = None
        self._payload = {}
        self._priority = MessagePriority.NORMAL
        self._correlation_id = None
        self._ttl = None
        self._headers = {}
    
    def request(self) -> 'MessageBuilder':
        self._type = MessageType.REQUEST
        return self
    
    def response(self) -> 'MessageBuilder':
        self._type = MessageType.RESPONSE
        return self
    
    def event(self) -> 'MessageBuilder':
        self._type = MessageType.EVENT
        return self
    
    def error(self) -> 'MessageBuilder':
        self._type = MessageType.ERROR
        return self
    
    def heartbeat(self) -> 'MessageBuilder':
        self._type = MessageType.HEARTBEAT
        return self
    
    def from_agent(self, sender: str) -> 'MessageBuilder':
        self._sender = sender
        return self
    
    def to_agent(self, receiver: str) -> 'MessageBuilder':
        self._receiver = receiver
        return self
    
    def broadcast(self) -> 'MessageBuilder':
        self._receiver = None
        self._type = MessageType.BROADCAST
        return self
    
    def with_payload(self, payload: Dict[str, Any]) -> 'MessageBuilder':
        self._payload = payload
        return self
    
    def with_priority(self, priority: MessagePriority) -> 'MessageBuilder':
        self._priority = priority
        return self
    
    def with_correlation_id(self, correlation_id: str) -> 'MessageBuilder':
        self._correlation_id = correlation_id
        return self
    
    def with_ttl(self, ttl_seconds: int) -> 'MessageBuilder':
        self._ttl = ttl_seconds
        return self
    
    def with_header(self, key: str, value: str) -> 'MessageBuilder':
        self._headers[key] = value
        return self
    
    def build(self) -> Message:
        return Message(
            id=self._id,
            type=self._type,
            sender=self._sender,
            receiver=self._receiver,
            payload=self._payload,
            priority=self._priority,
            correlation_id=self._correlation_id,
            ttl=self._ttl,
            headers=self._headers
        )


# 便捷函数
def create_request(sender: str, receiver: str, payload: Dict[str, Any],
                   priority: MessagePriority = MessagePriority.NORMAL) -> Message:
    """快速创建请求消息"""
    return MessageBuilder().request().from_agent(sender).to_agent(receiver) \
        .with_payload(payload).with_priority(priority).build()


def create_response(request_msg: Message, payload: Dict[str, Any],
                    success: bool = True) -> Message:
    """快速创建响应消息"""
    return request_msg.create_response({
        "success": success,
        **payload
    })


def create_event(sender: str, event_type: str, payload: Dict[str, Any],
                 receiver: Optional[str] = None) -> Message:
    """快速创建事件消息"""
    return Message(
        type=MessageType.EVENT,
        sender=sender,
        receiver=receiver,
        payload={"event_type": event_type, **payload}
    )


def create_heartbeat(sender: str, status: Dict[str, Any] = None) -> Message:
    """快速创建心跳消息"""
    return Message(
        type=MessageType.HEARTBEAT,
        sender=sender,
        payload={"status": status or {}, "timestamp": datetime.now().isoformat()},
        priority=MessagePriority.LOW
    )
