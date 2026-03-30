"""
KAS Agent Communication Protocol
智能体通信协议

提供标准化的Agent间通信机制。
"""

# 消息协议
from kas.core.protocol.message import (
    Message,
    MessageType,
    MessagePriority,
    MessageBuilder,
    SerializationFormat,
)

# 传输层
from kas.core.protocol.transport import (
    Transport,
    LocalTransport,
    TCPTransport,
    WebSocketTransport,
)

# 路由
from kas.core.protocol.router import (
    MessageRouter,
    RoutingStrategy,
    Subscription,
    Route,
)

# 通信管理
from kas.core.protocol.manager import (
    CommunicationManager,
    TransportConfig,
    ConnectionState,
    PendingMessage,
)

__all__ = [
    # 消息
    "Message",
    "MessageType",
    "MessagePriority",
    "MessageBuilder",
    "SerializationFormat",
    # 传输层
    "Transport",
    "LocalTransport",
    "TCPTransport",
    "WebSocketTransport",
    # 路由
    "MessageRouter",
    "RoutingStrategy",
    "Subscription",
    "Route",
    # 通信管理
    "CommunicationManager",
    "TransportConfig",
    "ConnectionState",
    "PendingMessage",
]
