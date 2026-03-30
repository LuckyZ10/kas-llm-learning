#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collaborative Space Module for Metaverse Materials Laboratory
协作空间模块 - 元宇宙材料实验室

Provides virtual meeting rooms, shared whiteboards, and avatar systems for
collaborative materials research in the metaverse.

Author: XR Expert Team
Version: 1.0.0
"""

import numpy as np
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any, Union, Set
from enum import Enum, auto
import logging
from collections import defaultdict, deque
import time
import asyncio
import uuid
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserRole(Enum):
    """用户角色"""
    OWNER = "owner"
    ADMIN = "admin"
    RESEARCHER = "researcher"
    VISITOR = "visitor"
    GUEST = "guest"


class RoomType(Enum):
    """房间类型"""
    PRIVATE = "private"           # 私人房间
    TEAM = "team"                 # 团队房间
    PUBLIC = "public"             # 公共房间
    PRESENTATION = "presentation" # 演示房间
    CONFERENCE = "conference"     # 会议室
    CLASSROOM = "classroom"       # 教室
    LABORATORY = "laboratory"     # 实验室


class PresenceStatus(Enum):
    """在线状态"""
    ONLINE = "online"
    AWAY = "away"
    BUSY = "busy"
    DO_NOT_DISTURB = "dnd"
    OFFLINE = "offline"


@dataclass
class Vector3:
    """3D向量"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Vector3':
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))
    
    def distance_to(self, other: 'Vector3') -> float:
        return np.linalg.norm(self.to_array() - other.to_array())
    
    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)


@dataclass
class Quaternion:
    """四元数"""
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class AvatarAppearance:
    """化身外观"""
    # 基础属性
    skin_color: Tuple[float, float, float] = (0.8, 0.7, 0.6)
    height: float = 1.75  # 米
    build: str = "average"  # slim, average, athletic, heavy
    
    # 服装
    shirt_color: Tuple[float, float, float] = (0.3, 0.5, 0.8)
    pants_color: Tuple[float, float, float] = (0.2, 0.2, 0.3)
    shoes_color: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    
    # 配饰
    glasses: bool = False
    hat: Optional[str] = None
    lab_coat: bool = True  # 实验室外套
    safety_goggles: bool = False
    
    # 自定义模型
    custom_model_url: Optional[str] = None
    animations: List[str] = field(default_factory=list)


@dataclass
class AvatarState:
    """化身状态"""
    user_id: str
    position: Vector3 = field(default_factory=Vector3)
    rotation: Quaternion = field(default_factory=Quaternion)
    
    # 身体部位位置
    head_position: Vector3 = field(default_factory=Vector3)
    left_hand: Optional[Vector3] = None
    right_hand: Optional[Vector3] = None
    
    # 动画状态
    current_animation: str = "idle"
    animation_time: float = 0.0
    
    # 交互状态
    is_gesturing: bool = False
    gesture_type: Optional[str] = None
    held_object_id: Optional[str] = None
    
    # 语音状态
    is_speaking: bool = False
    voice_volume: float = 0.0
    
    # 外观
    appearance: AvatarAppearance = field(default_factory=AvatarAppearance)
    
    # 时间戳
    last_update: float = field(default_factory=time.time)


@dataclass
class User:
    """协作用户"""
    user_id: str
    display_name: str
    email: str = ""
    role: UserRole = UserRole.RESEARCHER
    
    # 在线状态
    presence: PresenceStatus = PresenceStatus.ONLINE
    status_message: str = ""
    
    # 化身
    avatar: AvatarState = field(default_factory=lambda: AvatarState(""))
    
    # 统计
    joined_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    
    # 权限
    permissions: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        self.avatar.user_id = self.user_id


@dataclass
class ChatMessage:
    """聊天消息"""
    message_id: str
    sender_id: str
    sender_name: str
    content: str
    timestamp: float = field(default_factory=time.time)
    message_type: str = "text"  # text, image, file, system
    reply_to: Optional[str] = None
    reactions: Dict[str, List[str]] = field(default_factory=dict)  # emoji -> user_ids


@dataclass
class WhiteboardStroke:
    """白板笔画"""
    stroke_id: str
    user_id: str
    points: List[Tuple[float, float]]  # [(x, y), ...]
    color: Tuple[int, int, int, int]  # RGBA
    stroke_width: float = 2.0
    tool: str = "pen"  # pen, marker, highlighter, eraser
    timestamp: float = field(default_factory=time.time)
    page_index: int = 0


@dataclass
class WhiteboardElement:
    """白板元素（文本、图形、图片）"""
    element_id: str
    user_id: str
    element_type: str  # text, rectangle, circle, line, arrow, image, latex
    position: Tuple[float, float]  # (x, y)
    
    # 内容
    content: str = ""  # 文本内容或LaTeX
    image_data: Optional[str] = None  # Base64图像
    
    # 样式
    fill_color: Optional[Tuple[int, int, int, int]] = None
    stroke_color: Tuple[int, int, int, int] = field(default_factory=lambda: (0, 0, 0, 255))
    stroke_width: float = 1.0
    font_size: int = 16
    font_family: str = "Arial"
    
    # 变换
    rotation: float = 0.0
    scale: Tuple[float, float] = (1.0, 1.0)
    
    # 元数据
    timestamp: float = field(default_factory=time.time)
    page_index: int = 0


@dataclass
class WhiteboardPage:
    """白板页面"""
    page_index: int
    background_color: Tuple[int, int, int] = (255, 255, 255)
    background_image: Optional[str] = None
    
    # 内容
    strokes: List[WhiteboardStroke] = field(default_factory=list)
    elements: List[WhiteboardElement] = field(default_factory=list)
    
    # 尺寸
    width: float = 1920.0
    height: float = 1080.0


class SharedWhiteboard:
    """共享白板"""
    
    def __init__(self, board_id: str, name: str = ""):
        self.board_id = board_id
        self.name = name or f"Whiteboard-{board_id}"
        
        # 页面
        self.pages: List[WhiteboardPage] = [WhiteboardPage(0)]
        self.current_page_index = 0
        
        # 历史记录（用于撤销/重做）
        self.history: deque = deque(maxlen=100)
        self.redo_stack: List[Any] = []
        
        # 参与者光标位置
        user_cursors: Dict[str, Tuple[float, float]] = {}
        
        # 事件回调
        self.on_stroke_added: Optional[Callable] = None
        self.on_element_added: Optional[Callable] = None
        self.on_page_changed: Optional[Callable] = None
        
    @property
    def current_page(self) -> WhiteboardPage:
        """获取当前页面"""
        return self.pages[self.current_page_index]
    
    def add_page(self) -> int:
        """添加新页面"""
        new_index = len(self.pages)
        self.pages.append(WhiteboardPage(new_index))
        return new_index
    
    def switch_page(self, page_index: int) -> bool:
        """切换页面"""
        if 0 <= page_index < len(self.pages):
            self.current_page_index = page_index
            if self.on_page_changed:
                self.on_page_changed(page_index)
            return True
        return False
    
    def add_stroke(self, user_id: str, points: List[Tuple[float, float]],
                  color: Tuple[int, int, int, int] = (0, 0, 0, 255),
                  stroke_width: float = 2.0, tool: str = "pen") -> str:
        """添加笔画"""
        stroke_id = f"stroke_{int(time.time() * 1000)}_{user_id}"
        
        stroke = WhiteboardStroke(
            stroke_id=stroke_id,
            user_id=user_id,
            points=points,
            color=color,
            stroke_width=stroke_width,
            tool=tool,
            page_index=self.current_page_index
        )
        
        self.current_page.strokes.append(stroke)
        self.history.append(('add_stroke', stroke_id))
        self.redo_stack.clear()
        
        if self.on_stroke_added:
            self.on_stroke_added(stroke)
        
        return stroke_id
    
    def add_element(self, user_id: str, element_type: str,
                   position: Tuple[float, float], **kwargs) -> str:
        """添加元素"""
        element_id = f"elem_{int(time.time() * 1000)}_{user_id}"
        
        element = WhiteboardElement(
            element_id=element_id,
            user_id=user_id,
            element_type=element_type,
            position=position,
            page_index=self.current_page_index,
            **kwargs
        )
        
        self.current_page.elements.append(element)
        self.history.append(('add_element', element_id))
        self.redo_stack.clear()
        
        if self.on_element_added:
            self.on_element_added(element)
        
        return element_id
    
    def add_text(self, user_id: str, text: str, position: Tuple[float, float],
                font_size: int = 16, color: Tuple[int, int, int] = (0, 0, 0)) -> str:
        """添加文本"""
        return self.add_element(
            user_id, "text", position,
            content=text,
            font_size=font_size,
            stroke_color=(*color, 255)
        )
    
    def add_latex(self, user_id: str, latex: str, position: Tuple[float, float],
                 font_size: int = 20) -> str:
        """添加LaTeX公式"""
        return self.add_element(
            user_id, "latex", position,
            content=latex,
            font_size=font_size
        )
    
    def add_structure_image(self, user_id: str, image_base64: str,
                           position: Tuple[float, float]) -> str:
        """添加结构图像"""
        return self.add_element(
            user_id, "image", position,
            image_data=image_base64
        )
    
    def delete_stroke(self, stroke_id: str) -> bool:
        """删除笔画"""
        page = self.current_page
        for i, stroke in enumerate(page.strokes):
            if stroke.stroke_id == stroke_id:
                removed = page.strokes.pop(i)
                self.history.append(('delete_stroke', removed))
                return True
        return False
    
    def clear_page(self) -> None:
        """清空当前页面"""
        page = self.current_page
        old_strokes = page.strokes.copy()
        old_elements = page.elements.copy()
        
        page.strokes.clear()
        page.elements.clear()
        
        self.history.append(('clear_page', old_strokes, old_elements))
    
    def undo(self) -> bool:
        """撤销"""
        if not self.history:
            return False
        
        action = self.history.pop()
        self.redo_stack.append(action)
        
        if action[0] == 'add_stroke':
            stroke_id = action[1]
            self.delete_stroke(stroke_id)
        elif action[0] == 'add_element':
            element_id = action[1]
            self._delete_element(element_id)
        
        return True
    
    def redo(self) -> bool:
        """重做"""
        if not self.redo_stack:
            return False
        
        action = self.redo_stack.pop()
        # 重新执行操作...
        
        return True
    
    def _delete_element(self, element_id: str) -> bool:
        """删除元素"""
        page = self.current_page
        for i, elem in enumerate(page.elements):
            if elem.element_id == element_id:
                page.elements.pop(i)
                return True
        return False
    
    def export_to_image(self, page_index: Optional[int] = None) -> bytes:
        """导出页面为图像"""
        # 使用PIL生成图像
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            page = self.pages[page_index] if page_index is not None else self.current_page
            img = Image.new('RGB', (int(page.width), int(page.height)), 
                          page.background_color)
            draw = ImageDraw.Draw(img)
            
            # 绘制所有笔画
            for stroke in page.strokes:
                if len(stroke.points) >= 2:
                    draw.line(stroke.points, fill=stroke.color[:3], 
                             width=int(stroke.stroke_width))
            
            # 绘制元素
            for elem in page.elements:
                if elem.element_type == "text":
                    try:
                        font = ImageFont.truetype("arial.ttf", elem.font_size)
                    except:
                        font = ImageFont.load_default()
                    draw.text(elem.position, elem.content, 
                             fill=elem.stroke_color[:3], font=font)
            
            # 转换为bytes
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            return buffer.getvalue()
            
        except ImportError:
            logger.warning("PIL not available for whiteboard export")
            return b""
    
    def export_to_pdf(self) -> bytes:
        """导出所有页面为PDF"""
        # 使用reportlab生成PDF
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import A4
            
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=A4)
            width, height = A4
            
            for i, page in enumerate(self.pages):
                # 简化的PDF导出
                c.drawString(100, height - 50, f"Page {i+1}: {self.name}")
                c.showPage()
            
            c.save()
            return buffer.getvalue()
            
        except ImportError:
            logger.warning("reportlab not available for PDF export")
            return b""
    
    def get_state(self) -> Dict[str, Any]:
        """获取白板状态"""
        return {
            'board_id': self.board_id,
            'name': self.name,
            'current_page': self.current_page_index,
            'total_pages': len(self.pages),
            'stroke_count': sum(len(p.strokes) for p in self.pages),
            'element_count': sum(len(p.elements) for p in self.pages)
        }


@dataclass
class VirtualMeetingRoom:
    """虚拟会议室"""
    room_id: str
    name: str
    room_type: RoomType = RoomType.CONFERENCE
    owner_id: str = ""
    
    # 参与者
    participants: Dict[str, User] = field(default_factory=dict)
    max_participants: int = 50
    
    # 白板
    whiteboards: Dict[str, SharedWhiteboard] = field(default_factory=dict)
    active_whiteboard_id: Optional[str] = None
    
    # 聊天
    chat_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # 媒体
    screen_sharer_id: Optional[str] = None
    presentation_url: Optional[str] = None
    
    # 设置
    settings: Dict[str, Any] = field(default_factory=lambda: {
        'mute_on_entry': False,
        'video_on_entry': True,
        'allow_recording': True,
        'waiting_room': False,
        'chat_enabled': True,
        'whiteboard_enabled': True,
    })
    
    # 时间
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    
    def __post_init__(self):
        # 创建默认白板
        default_board = SharedWhiteboard("default", "Main Whiteboard")
        self.whiteboards["default"] = default_board
        self.active_whiteboard_id = "default"
    
    @property
    def participant_count(self) -> int:
        """参与者数量"""
        return len(self.participants)
    
    @property
    def is_active(self) -> bool:
        """会议是否在进行中"""
        return self.started_at is not None and self.ended_at is None
    
    def join(self, user: User) -> bool:
        """用户加入会议"""
        if self.participant_count >= self.max_participants:
            return False
        
        if user.user_id in self.participants:
            return False
        
        self.participants[user.user_id] = user
        user.joined_at = time.time()
        
        # 发送系统消息
        self.send_system_message(f"{user.display_name} joined the room")
        
        logger.info(f"User {user.display_name} joined room {self.name}")
        return True
    
    def leave(self, user_id: str) -> bool:
        """用户离开会议"""
        if user_id not in self.participants:
            return False
        
        user = self.participants.pop(user_id)
        
        # 如果正在屏幕共享，停止共享
        if self.screen_sharer_id == user_id:
            self.screen_sharer_id = None
        
        self.send_system_message(f"{user.display_name} left the room")
        
        logger.info(f"User {user.display_name} left room {self.name}")
        return True
    
    def start_meeting(self) -> bool:
        """开始会议"""
        if self.is_active:
            return False
        
        self.started_at = time.time()
        self.send_system_message("Meeting started")
        return True
    
    def end_meeting(self) -> bool:
        """结束会议"""
        if not self.is_active:
            return False
        
        self.ended_at = time.time()
        self.send_system_message("Meeting ended")
        
        # 踢出所有参与者
        for user_id in list(self.participants.keys()):
            self.leave(user_id)
        
        return True
    
    def send_chat_message(self, user_id: str, content: str,
                         msg_type: str = "text", reply_to: Optional[str] = None) -> str:
        """发送聊天消息"""
        if user_id not in self.participants and msg_type != "system":
            return ""
        
        sender = self.participants.get(user_id)
        sender_name = sender.display_name if sender else "System"
        
        message = ChatMessage(
            message_id=f"msg_{int(time.time() * 1000)}",
            sender_id=user_id,
            sender_name=sender_name,
            content=content,
            message_type=msg_type,
            reply_to=reply_to
        )
        
        self.chat_history.append(message)
        return message.message_id
    
    def send_system_message(self, content: str) -> str:
        """发送系统消息"""
        return self.send_chat_message("system", content, "system")
    
    def create_whiteboard(self, name: str) -> str:
        """创建新白板"""
        board_id = f"board_{int(time.time() * 1000)}"
        board = SharedWhiteboard(board_id, name)
        self.whiteboards[board_id] = board
        return board_id
    
    def switch_whiteboard(self, board_id: str) -> bool:
        """切换白板"""
        if board_id in self.whiteboards:
            self.active_whiteboard_id = board_id
            return True
        return False
    
    def start_screen_share(self, user_id: str) -> bool:
        """开始屏幕共享"""
        if user_id not in self.participants:
            return False
        
        if self.screen_sharer_id and self.screen_sharer_id != user_id:
            return False
        
        self.screen_sharer_id = user_id
        user = self.participants[user_id]
        self.send_system_message(f"{user.display_name} started screen sharing")
        return True
    
    def stop_screen_share(self, user_id: str) -> bool:
        """停止屏幕共享"""
        if self.screen_sharer_id == user_id:
            self.screen_sharer_id = None
            user = self.participants.get(user_id)
            if user:
                self.send_system_message(f"{user.display_name} stopped screen sharing")
            return True
        return False
    
    def update_user_avatar(self, user_id: str, avatar_state: AvatarState) -> bool:
        """更新用户化身状态"""
        if user_id not in self.participants:
            return False
        
        user = self.participants[user_id]
        user.avatar = avatar_state
        user.last_activity = time.time()
        return True
    
    def get_room_state(self) -> Dict[str, Any]:
        """获取房间状态"""
        return {
            'room_id': self.room_id,
            'name': self.name,
            'type': self.room_type.value,
            'is_active': self.is_active,
            'participant_count': self.participant_count,
            'max_participants': self.max_participants,
            'participants': [
                {
                    'user_id': u.user_id,
                    'name': u.display_name,
                    'role': u.role.value,
                    'presence': u.presence.value,
                    'avatar_position': {
                        'x': u.avatar.position.x,
                        'y': u.avatar.position.y,
                        'z': u.avatar.position.z
                    } if u.avatar else None
                }
                for u in self.participants.values()
            ],
            'whiteboards': list(self.whiteboards.keys()),
            'active_whiteboard': self.active_whiteboard_id,
            'screen_sharer': self.screen_sharer_id,
            'settings': self.settings
        }


class CollaborativeSpace:
    """协作空间主类"""
    
    def __init__(self):
        self.rooms: Dict[str, VirtualMeetingRoom] = {}
        self.users: Dict[str, User] = {}
        self.user_presence: Dict[str, PresenceStatus] = {}
        
        # 事件处理
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # 自动清理
        self.inactive_threshold = 3600  # 1小时
        
    def create_user(self, user_id: str, display_name: str, 
                   email: str = "", role: UserRole = UserRole.RESEARCHER) -> User:
        """创建用户"""
        user = User(
            user_id=user_id,
            display_name=display_name,
            email=email,
            role=role
        )
        self.users[user_id] = user
        return user
    
    def create_room(self, name: str, room_type: RoomType = RoomType.TEAM,
                   owner_id: str = "", max_participants: int = 50) -> str:
        """创建会议室"""
        room_id = f"room_{str(uuid.uuid4())[:8]}"
        
        room = VirtualMeetingRoom(
            room_id=room_id,
            name=name,
            room_type=room_type,
            owner_id=owner_id,
            max_participants=max_participants
        )
        
        self.rooms[room_id] = room
        logger.info(f"Created room: {name} ({room_id})")
        return room_id
    
    def get_room(self, room_id: str) -> Optional[VirtualMeetingRoom]:
        """获取会议室"""
        return self.rooms.get(room_id)
    
    def join_room(self, room_id: str, user_id: str) -> bool:
        """加入会议室"""
        room = self.rooms.get(room_id)
        user = self.users.get(user_id)
        
        if not room or not user:
            return False
        
        return room.join(user)
    
    def leave_room(self, room_id: str, user_id: str) -> bool:
        """离开会议室"""
        room = self.rooms.get(room_id)
        if not room:
            return False
        
        return room.leave(user_id)
    
    def update_presence(self, user_id: str, status: PresenceStatus,
                       message: str = "") -> bool:
        """更新在线状态"""
        if user_id not in self.users:
            return False
        
        user = self.users[user_id]
        user.presence = status
        user.status_message = message
        
        self.user_presence[user_id] = status
        
        # 触发事件
        self._trigger_event('presence_changed', {
            'user_id': user_id,
            'status': status.value,
            'message': message
        })
        
        return True
    
    def get_online_users(self) -> List[User]:
        """获取在线用户列表"""
        return [
            u for u in self.users.values()
            if u.presence != PresenceStatus.OFFLINE
        ]
    
    def get_user_rooms(self, user_id: str) -> List[VirtualMeetingRoom]:
        """获取用户参与的房间"""
        return [
            room for room in self.rooms.values()
            if user_id in room.participants
        ]
    
    def on(self, event: str, handler: Callable) -> None:
        """注册事件处理器"""
        self.event_handlers[event].append(handler)
    
    def _trigger_event(self, event: str, data: Any) -> None:
        """触发事件"""
        for handler in self.event_handlers.get(event, []):
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
    
    def cleanup_inactive_rooms(self) -> int:
        """清理不活跃的房间"""
        current_time = time.time()
        to_remove = []
        
        for room_id, room in self.rooms.items():
            if not room.is_active and room.ended_at:
                if current_time - room.ended_at > self.inactive_threshold:
                    to_remove.append(room_id)
        
        for room_id in to_remove:
            del self.rooms[room_id]
        
        logger.info(f"Cleaned up {len(to_remove)} inactive rooms")
        return len(to_remove)
    
    def broadcast_to_room(self, room_id: str, message: Dict[str, Any],
                         exclude_user: Optional[str] = None) -> int:
        """向房间内所有用户广播消息"""
        room = self.rooms.get(room_id)
        if not room:
            return 0
        
        count = 0
        for user_id in room.participants:
            if user_id != exclude_user:
                # 实际实现中会通过网络发送
                count += 1
        
        return count
    
    def get_space_stats(self) -> Dict[str, Any]:
        """获取空间统计信息"""
        return {
            'total_users': len(self.users),
            'online_users': len(self.get_online_users()),
            'total_rooms': len(self.rooms),
            'active_rooms': sum(1 for r in self.rooms.values() if r.is_active),
            'total_participants': sum(r.participant_count for r in self.rooms.values()),
            'total_whiteboards': sum(len(r.whiteboards) for r in self.rooms.values())
        }


# 示例使用
def example_usage():
    """使用示例"""
    print("=" * 60)
    print("Collaborative Space Example - Metaverse Materials Lab")
    print("=" * 60)
    
    # 创建协作空间
    space = CollaborativeSpace()
    
    # 创建用户
    researcher1 = space.create_user(
        "user_001", "Dr. Zhang", "zhang@lab.com", UserRole.RESEARCHER
    )
    researcher2 = space.create_user(
        "user_002", "Dr. Li", "li@lab.com", UserRole.RESEARCHER
    )
    student = space.create_user(
        "user_003", "Wang Student", "wang@student.com", UserRole.VISITOR
    )
    
    print(f"\n✓ Created {len(space.users)} users")
    
    # 创建会议室
    room_id = space.create_room(
        "Crystal Structure Analysis",
        RoomType.LABORATORY,
        owner_id=researcher1.user_id,
        max_participants=20
    )
    print(f"✓ Created room: Crystal Structure Analysis ({room_id})")
    
    room = space.get_room(room_id)
    
    # 用户加入
    space.join_room(room_id, researcher1.user_id)
    space.join_room(room_id, researcher2.user_id)
    space.join_room(room_id, student.user_id)
    print(f"✓ {room.participant_count} users joined the room")
    
    # 开始会议
    room.start_meeting()
    print(f"✓ Meeting started")
    
    # 发送聊天消息
    msg1 = room.send_chat_message(
        researcher1.user_id,
        "Let's analyze the new Si crystal structure data."
    )
    msg2 = room.send_chat_message(
        researcher2.user_id,
        "I've uploaded the latest XRD results."
    )
    print(f"✓ Sent {len(room.chat_history)} chat messages")
    
    # 使用白板
    board = room.whiteboards["default"]
    
    # 绘制晶体结构
    board.add_stroke(
        researcher1.user_id,
        [(100, 100), (200, 150), (300, 100), (400, 150)],
        color=(0, 0, 255, 255),
        stroke_width=3
    )
    
    # 添加标注
    board.add_text(
        researcher1.user_id,
        "Unit Cell: a = 5.43 Å",
        (150, 200),
        font_size=20,
        color=(0, 0, 0)
    )
    
    # 添加LaTeX公式
    board.add_latex(
        researcher2.user_id,
        r"E = \\frac{\\hbar^2 k^2}{2m^*}",
        (150, 250),
        font_size=24
    )
    
    print(f"✓ Whiteboard has {len(board.current_page.strokes)} strokes and "
          f"{len(board.current_page.elements)} elements")
    
    # 创建新页面
    board.add_page()
    board.switch_page(1)
    print(f"✓ Created new page, total pages: {len(board.pages)}")
    
    # 更新化身位置
    researcher1.avatar.position = Vector3(0, 0, 0)
    researcher1.avatar.rotation = Quaternion(1, 0, 0, 0)
    researcher1.avatar.is_speaking = True
    
    room.update_user_avatar(researcher1.user_id, researcher1.avatar)
    print(f"✓ Updated avatar for {researcher1.display_name}")
    
    # 屏幕共享
    room.start_screen_share(researcher1.user_id)
    print(f"✓ {researcher1.display_name} started screen sharing")
    
    # 获取房间状态
    state = room.get_room_state()
    print(f"\n✓ Room state:")
    print(f"  - Active: {state['is_active']}")
    print(f"  - Participants: {state['participant_count']}")
    print(f"  - Whiteboards: {len(state['whiteboards'])}")
    
    # 更新在线状态
    space.update_presence(researcher2.user_id, PresenceStatus.BUSY, "In experiment")
    print(f"✓ Updated presence for {researcher2.display_name}: BUSY")
    
    # 获取统计信息
    stats = space.get_space_stats()
    print(f"\n✓ Space statistics:")
    print(f"  - Total users: {stats['total_users']}")
    print(f"  - Online users: {stats['online_users']}")
    print(f"  - Total rooms: {stats['total_rooms']}")
    print(f"  - Active rooms: {stats['active_rooms']}")
    
    # 白板导出
    board.switch_page(0)
    image_data = board.export_to_image()
    print(f"✓ Exported whiteboard to image ({len(image_data)} bytes)")
    
    # 用户离开
    space.leave_room(room_id, student.user_id)
    print(f"✓ {student.display_name} left the room")
    
    # 结束会议
    room.end_meeting()
    print(f"✓ Meeting ended")
    
    print("\n" + "=" * 60)
    print("Collaborative Space Example Complete")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()
