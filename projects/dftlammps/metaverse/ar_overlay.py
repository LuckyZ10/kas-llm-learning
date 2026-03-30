#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AR Overlay Module for Metaverse Materials Laboratory
AR叠加模块 - 元宇宙材料实验室

Provides real-time data overlay, remote expert guidance, and experimental assistance
through augmented reality interfaces.

Author: XR Expert Team
Version: 1.0.0
"""

import numpy as np
import cv2
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from enum import Enum, auto
import logging
from collections import deque
import time
import asyncio
from PIL import Image, ImageDraw, ImageFont
import io
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AROverlayType(Enum):
    """AR叠加类型"""
    TEXT = "text"
    ANNOTATION = "annotation"
    MEASUREMENT = "measurement"
    DATA_VISUALIZATION = "data_viz"
    WARNING = "warning"
    GUIDANCE = "guidance"
    MOLECULAR_STRUCTURE = "molecule"
    CHART = "chart"
    VIDEO_FEED = "video"
    MODEL_3D = "3d_model"


class ARAnchorType(Enum):
    """AR锚点类型"""
    WORLD = "world"           # 世界坐标固定
    SCREEN = "screen"         # 屏幕坐标固定
    OBJECT = "object"         # 附着在物体上
    FACE = "face"             # 附着在面部
    HAND = "hand"             # 附着在手部
    PLANE = "plane"           # 附着在检测到的平面


@dataclass
class ARVector2:
    """2D向量"""
    x: float = 0.0
    y: float = 0.0
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def distance_to(self, other: 'ARVector2') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class ARVector3:
    """3D向量用于AR空间"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'ARVector3':
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))


@dataclass
class ARBoundingBox:
    """AR边界框"""
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0
    label: str = ""
    
    @property
    def center(self) -> ARVector2:
        return ARVector2(self.x + self.width/2, self.y + self.height/2)
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def contains(self, point: ARVector2) -> bool:
        return (self.x <= point.x <= self.x + self.width and
                self.y <= point.y <= self.y + self.height)
    
    def intersects(self, other: 'ARBoundingBox') -> bool:
        return not (self.x + self.width < other.x or
                   other.x + other.width < self.x or
                   self.y + self.height < other.y or
                   other.y + other.height < self.y)


@dataclass
class ARElement:
    """AR元素基类"""
    element_id: str
    overlay_type: AROverlayType
    anchor_type: ARAnchorType
    
    # 位置信息
    position_2d: Optional[ARVector2] = None  # 屏幕坐标
    position_3d: Optional[ARVector3] = None  # 世界坐标
    
    # 视觉属性
    scale: float = 1.0
    rotation: float = 0.0  # 度
    opacity: float = 1.0
    visible: bool = True
    
    # 内容
    content: Dict[str, Any] = field(default_factory=dict)
    
    # 元数据
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    priority: int = 0
    
    # 交互
    is_interactive: bool = False
    on_tap: Optional[Callable] = None
    on_long_press: Optional[Callable] = None


@dataclass
class ARTextElement(ARElement):
    """AR文本元素"""
    text: str = ""
    font_size: int = 24
    font_color: Tuple[int, int, int, int] = (255, 255, 255, 255)
    background_color: Optional[Tuple[int, int, int, int]] = None
    font_family: str = "default"
    alignment: str = "left"  # left, center, right
    max_width: Optional[float] = None


@dataclass
class ARAnnotationElement(ARElement):
    """AR标注元素"""
    target_point: ARVector3 = field(default_factory=ARVector3)
    label_text: str = ""
    label_position: str = "above"  # above, below, left, right
    line_color: Tuple[int, int, int] = (0, 255, 255)
    line_width: int = 2
    arrow_size: float = 10.0
    show_distance: bool = False


@dataclass
class ARMeasurementElement(ARElement):
    """AR测量元素"""
    start_point: ARVector3 = field(default_factory=ARVector3)
    end_point: ARVector3 = field(default_factory=ARVector3)
    measurement_value: float = 0.0
    unit: str = "Å"
    precision: int = 3
    show_line: bool = True
    line_style: str = "solid"  # solid, dashed, dotted
    measurement_type: str = "distance"  # distance, angle, area, volume


@dataclass
class ARDataVizElement(ARElement):
    """AR数据可视化元素"""
    data: Dict[str, Any] = field(default_factory=dict)
    viz_type: str = "bar"  # bar, line, scatter, heatmap, contour
    width: float = 200.0
    height: float = 150.0
    color_map: str = "viridis"
    show_axes: bool = True
    show_legend: bool = True
    real_time_update: bool = True
    data_buffer: deque = field(default_factory=lambda: deque(maxlen=100))


@dataclass
class ARGuidanceElement(ARElement):
    """AR引导元素"""
    instruction_text: str = ""
    step_number: int = 0
    total_steps: int = 0
    highlight_region: Optional[ARBoundingBox] = None
    arrow_direction: Optional[str] = None  # up, down, left, right
    animation_type: str = "pulse"  # pulse, bounce, fade
    completion_required: bool = True


@dataclass
class ARTrackedObject:
    """AR追踪物体"""
    object_id: str
    object_type: str  # sample, equipment, marker, hand, face
    bounding_box: ARBoundingBox
    position_3d: Optional[ARVector3] = None
    rotation_3d: Optional[np.ndarray] = None
    confidence: float = 1.0
    
    # 识别信息
    recognized_label: str = ""
    recognition_confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 追踪历史
    position_history: deque = field(default_factory=lambda: deque(maxlen=30))
    last_seen: float = field(default_factory=time.time)
    
    def update_position(self, new_bbox: ARBoundingBox, new_pos_3d: Optional[ARVector3] = None):
        """更新位置"""
        self.bounding_box = new_bbox
        if new_pos_3d:
            self.position_3d = new_pos_3d
            self.position_history.append((time.time(), new_pos_3d))
        self.last_seen = time.time()


@dataclass
class ARRemoteExpertSession:
    """远程专家指导会话"""
    session_id: str
    expert_id: str
    expert_name: str
    local_user_id: str
    start_time: float = field(default_factory=time.time)
    
    # 通信状态
    is_audio_active: bool = True
    is_video_active: bool = True
    is_screen_share_active: bool = False
    
    # AR叠加元素
    expert_annotations: List[ARElement] = field(default_factory=list)
    pointer_position: Optional[ARVector2] = None
    expert_gaze: Optional[ARVector3] = None
    
    # 聊天消息
    messages: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_annotation(self, element: ARElement) -> str:
        """添加专家标注"""
        self.expert_annotations.append(element)
        return element.element_id
    
    def add_message(self, text: str, sender: str, msg_type: str = "text"):
        """添加消息"""
        self.messages.append({
            'text': text,
            'sender': sender,
            'type': msg_type,
            'timestamp': time.time()
        })


class ARSceneUnderstanding:
    """AR场景理解模块"""
    
    def __init__(self):
        self.detected_planes: List[Dict[str, Any]] = []
        self.tracked_objects: Dict[str, ARTrackedObject] = {}
        self.detected_markers: Dict[str, Dict[str, Any]] = {}
        self.scene_features: np.ndarray = np.array([])
        self.point_cloud: Optional[np.ndarray] = None
        
        # 检测器状态
        self.object_detector = None
        self.marker_detector = None
        self.plane_detector = None
        
    def initialize_detectors(self) -> bool:
        """初始化检测器"""
        try:
            # 尝试加载OpenCV对象检测
            # self.object_detector = cv2.dnn.readNet(...)
            logger.info("AR detectors initialized")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize some detectors: {e}")
            return False
    
    def detect_objects(self, frame: np.ndarray) -> List[ARBoundingBox]:
        """检测场景中的物体"""
        boxes = []
        
        # 模拟检测 - 实际使用时会调用深度学习模型
        # 这里使用简单的颜色分割来演示
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                box = ARBoundingBox(
                    x=float(x), y=float(y),
                    width=float(w), height=float(h),
                    confidence=0.8,
                    label=f"object_{i}"
                )
                boxes.append(box)
        
        return boxes
    
    def detect_markers(self, frame: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """检测AR标记（如ArUco标记）"""
        markers = {}
        
        try:
            # 使用ArUco标记检测
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            parameters = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            
            corners, ids, rejected = detector.detectMarkers(frame)
            
            if ids is not None:
                for i, marker_id in enumerate(ids.flatten()):
                    marker_corners = corners[i][0]
                    x = float(marker_corners[:, 0].min())
                    y = float(marker_corners[:, 1].min())
                    w = float(marker_corners[:, 0].max() - x)
                    h = float(marker_corners[:, 1].max() - y)
                    
                    markers[str(marker_id)] = {
                        'id': int(marker_id),
                        'bbox': ARBoundingBox(x, y, w, h),
                        'corners': marker_corners.tolist(),
                        'confidence': 0.95
                    }
        except Exception as e:
            logger.debug(f"Marker detection skipped: {e}")
        
        return markers
    
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """估计场景深度"""
        # 实际使用时会调用深度相机或深度估计模型
        # 这里返回模拟深度图
        h, w = frame.shape[:2]
        depth = np.ones((h, w), dtype=np.float32) * 2.0  # 默认2米
        
        # 模拟中心区域更近
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
        depth = 2.0 - np.clip(dist_from_center / max(h, w), 0, 1) * 1.5
        
        return depth
    
    def update_tracking(self, frame: np.ndarray) -> Dict[str, ARTrackedObject]:
        """更新物体追踪"""
        current_time = time.time()
        
        # 检测当前帧中的物体
        detected_boxes = self.detect_objects(frame)
        depth_map = self.estimate_depth(frame)
        
        # 更新或创建追踪对象
        for box in detected_boxes:
            # 查找匹配的现有对象
            matched = False
            for obj_id, tracked in self.tracked_objects.items():
                if tracked.bounding_box.intersects(box):
                    # 计算3D位置
                    cx, cy = int(box.center.x), int(box.center.y)
                    z = depth_map[min(cy, depth_map.shape[0]-1), min(cx, depth_map.shape[1]-1)]
                    pos_3d = ARVector3(box.center.x, box.center.y, z)
                    
                    tracked.update_position(box, pos_3d)
                    matched = True
                    break
            
            if not matched:
                # 创建新追踪对象
                new_id = f"obj_{int(current_time * 1000)}_{len(self.tracked_objects)}"
                cx, cy = int(box.center.x), int(box.center.y)
                z = depth_map[min(cy, depth_map.shape[0]-1), min(cx, depth_map.shape[1]-1)]
                
                new_obj = ARTrackedObject(
                    object_id=new_id,
                    object_type="unknown",
                    bounding_box=box,
                    position_3d=ARVector3(box.center.x, box.center.y, z)
                )
                self.tracked_objects[new_id] = new_obj
        
        # 移除过期的追踪对象
        expired = [
            oid for oid, obj in self.tracked_objects.items()
            if current_time - obj.last_seen > 5.0
        ]
        for oid in expired:
            del self.tracked_objects[oid]
        
        return self.tracked_objects


class ARDataOverlay:
    """AR数据叠加管理器"""
    
    def __init__(self, canvas_width: int = 1920, canvas_height: int = 1080):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.elements: Dict[str, ARElement] = {}
        self.element_counter = 0
        
        # 渲染配置
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.line_thickness = 2
        self.marker_size = 20
        
        # 性能优化
        self.render_cache: Dict[str, np.ndarray] = {}
        self.cache_ttl = 1.0  # 秒
        
    def create_text(self, text: str, position: ARVector2, 
                   font_size: int = 24, color: Tuple[int, int, int] = (255, 255, 255),
                   background: Optional[Tuple[int, int, int, int]] = None,
                   anchor: ARAnchorType = ARAnchorType.SCREEN) -> str:
        """创建文本叠加"""
        self.element_counter += 1
        element_id = f"text_{self.element_counter}"
        
        element = ARTextElement(
            element_id=element_id,
            overlay_type=AROverlayType.TEXT,
            anchor_type=anchor,
            position_2d=position,
            text=text,
            font_size=font_size,
            font_color=(*color, 255),
            background_color=background
        )
        
        self.elements[element_id] = element
        return element_id
    
    def create_annotation(self, target_3d: ARVector3, label: str,
                         label_pos: str = "above") -> str:
        """创建3D标注"""
        self.element_counter += 1
        element_id = f"annotation_{self.element_counter}"
        
        element = ARAnnotationElement(
            element_id=element_id,
            overlay_type=AROverlayType.ANNOTATION,
            anchor_type=ARAnchorType.WORLD,
            position_3d=target_3d,
            target_point=target_3d,
            label_text=label,
            label_position=label_pos
        )
        
        self.elements[element_id] = element
        return element_id
    
    def create_measurement(self, start: ARVector3, end: ARVector3,
                          unit: str = "Å") -> str:
        """创建测量标注"""
        self.element_counter += 1
        element_id = f"measurement_{self.element_counter}"
        
        # 计算距离
        distance = np.linalg.norm(end.to_array() - start.to_array())
        
        element = ARMeasurementElement(
            element_id=element_id,
            overlay_type=AROverlayType.MEASUREMENT,
            anchor_type=ARAnchorType.WORLD,
            start_point=start,
            end_point=end,
            measurement_value=distance,
            unit=unit
        )
        
        self.elements[element_id] = element
        return element_id
    
    def create_guidance(self, instruction: str, step: int = 0, 
                       total: int = 0, highlight: Optional[ARBoundingBox] = None) -> str:
        """创建操作引导"""
        self.element_counter += 1
        element_id = f"guidance_{self.element_counter}"
        
        element = ARGuidanceElement(
            element_id=element_id,
            overlay_type=AROverlayType.GUIDANCE,
            anchor_type=ARAnchorType.SCREEN,
            instruction_text=instruction,
            step_number=step,
            total_steps=total,
            highlight_region=highlight
        )
        
        self.elements[element_id] = element
        return element_id
    
    def create_data_viz(self, data: Dict[str, Any], position: ARVector2,
                       viz_type: str = "bar") -> str:
        """创建数据可视化"""
        self.element_counter += 1
        element_id = f"dataviz_{self.element_counter}"
        
        element = ARDataVizElement(
            element_id=element_id,
            overlay_type=AROverlayType.DATA_VISUALIZATION,
            anchor_type=ARAnchorType.SCREEN,
            position_2d=position,
            data=data,
            viz_type=viz_type
        )
        
        self.elements[element_id] = element
        return element_id
    
    def update_element(self, element_id: str, **kwargs) -> bool:
        """更新元素属性"""
        if element_id not in self.elements:
            return False
        
        element = self.elements[element_id]
        for key, value in kwargs.items():
            if hasattr(element, key):
                setattr(element, key, value)
        
        element.updated_at = time.time()
        return True
    
    def remove_element(self, element_id: str) -> bool:
        """移除元素"""
        if element_id in self.elements:
            del self.elements[element_id]
            return True
        return False
    
    def clear_elements(self, overlay_type: Optional[AROverlayType] = None):
        """清除元素"""
        if overlay_type is None:
            self.elements.clear()
        else:
            to_remove = [
                eid for eid, elem in self.elements.items()
                if elem.overlay_type == overlay_type
            ]
            for eid in to_remove:
                del self.elements[eid]
    
    def render_to_frame(self, frame: np.ndarray, 
                       camera_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """将所有元素渲染到视频帧"""
        output = frame.copy()
        
        # 按优先级排序
        sorted_elements = sorted(
            self.elements.values(),
            key=lambda e: e.priority
        )
        
        for element in sorted_elements:
            if not element.visible:
                continue
            
            if element.overlay_type == AROverlayType.TEXT:
                output = self._render_text(output, element)
            elif element.overlay_type == AROverlayType.ANNOTATION:
                output = self._render_annotation(output, element, camera_matrix)
            elif element.overlay_type == AROverlayType.MEASUREMENT:
                output = self._render_measurement(output, element, camera_matrix)
            elif element.overlay_type == AROverlayType.GUIDANCE:
                output = self._render_guidance(output, element)
            elif element.overlay_type == AROverlayType.DATA_VISUALIZATION:
                output = self._render_data_viz(output, element)
        
        return output
    
    def _render_text(self, frame: np.ndarray, element: ARTextElement) -> np.ndarray:
        """渲染文本元素"""
        if not element.position_2d:
            return frame
        
        x, y = int(element.position_2d.x), int(element.position_2d.y)
        
        # 计算文本大小
        (text_width, text_height), _ = cv2.getTextSize(
            element.text, self.font, element.font_size/30, self.line_thickness
        )
        
        # 绘制背景
        if element.background_color:
            bg_color = element.background_color[:3]
            alpha = element.background_color[3] / 255.0 if len(element.background_color) > 3 else 1.0
            
            overlay = frame.copy()
            cv2.rectangle(overlay, 
                         (x - 5, y - text_height - 5),
                         (x + text_width + 5, y + 5),
                         bg_color, -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # 绘制文本
        color = element.font_color[:3]
        cv2.putText(frame, element.text, (x, y), self.font,
                   element.font_size/30, color, self.line_thickness)
        
        return frame
    
    def _render_annotation(self, frame: np.ndarray, element: ARAnnotationElement,
                          camera_matrix: Optional[np.ndarray]) -> np.ndarray:
        """渲染3D标注"""
        # 3D到2D投影（简化版本）
        if camera_matrix is None:
            # 使用简单的透视投影
            point_3d = element.target_point
            x = int(point_3d.x + self.canvas_width/2)
            y = int(-point_3d.y + self.canvas_height/2)
        else:
            # 使用相机矩阵投影
            point_cam = camera_matrix @ np.append(element.target_point.to_array(), 1)
            x = int(point_cam[0] / point_cam[2])
            y = int(point_cam[1] / point_cam[2])
        
        # 绘制目标点
        cv2.circle(frame, (x, y), 5, element.line_color, -1)
        cv2.circle(frame, (x, y), 10, element.line_color, 2)
        
        # 绘制引线到标签
        label_offset = 50
        label_x = x + label_offset
        label_y = y - label_offset if element.label_position == "above" else y + label_offset
        
        cv2.line(frame, (x, y), (label_x, label_y), element.line_color, element.line_width)
        
        # 绘制标签背景
        (text_w, text_h), _ = cv2.getTextSize(
            element.label_text, self.font, 0.6, 1
        )
        cv2.rectangle(frame,
                     (label_x - 5, label_y - text_h - 5),
                     (label_x + text_w + 5, label_y + 5),
                     (0, 0, 0), -1)
        
        # 绘制标签文本
        cv2.putText(frame, element.label_text, (label_x, label_y),
                   self.font, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def _render_measurement(self, frame: np.ndarray, element: ARMeasurementElement,
                           camera_matrix: Optional[np.ndarray]) -> np.ndarray:
        """渲染测量标注"""
        # 投影起点和终点
        p1 = element.start_point
        p2 = element.end_point
        
        if camera_matrix is None:
            x1 = int(p1.x + self.canvas_width/2)
            y1 = int(-p1.y + self.canvas_height/2)
            x2 = int(p2.x + self.canvas_width/2)
            y2 = int(-p2.y + self.canvas_height/2)
        else:
            p1_cam = camera_matrix @ np.append(p1.to_array(), 1)
            p2_cam = camera_matrix @ np.append(p2.to_array(), 1)
            x1, y1 = int(p1_cam[0]/p1_cam[2]), int(p1_cam[1]/p1_cam[2])
            x2, y2 = int(p2_cam[0]/p2_cam[2]), int(p2_cam[1]/p2_cam[2])
        
        # 绘制测量线
        line_color = (0, 255, 0)  # 绿色
        cv2.line(frame, (x1, y1), (x2, y2), line_color, 2)
        
        # 绘制端点
        cv2.circle(frame, (x1, y1), 5, line_color, -1)
        cv2.circle(frame, (x2, y2), 5, line_color, -1)
        
        # 绘制测量值
        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
        value_text = f"{element.measurement_value:.{element.precision}f} {element.unit}"
        
        (text_w, text_h), _ = cv2.getTextSize(value_text, self.font, 0.6, 1)
        cv2.rectangle(frame,
                     (mid_x - text_w//2 - 5, mid_y - text_h - 5),
                     (mid_x + text_w//2 + 5, mid_y + 5),
                     (0, 0, 0), -1)
        cv2.putText(frame, value_text, (mid_x - text_w//2, mid_y),
                   self.font, 0.6, (0, 255, 0), 1)
        
        return frame
    
    def _render_guidance(self, frame: np.ndarray, element: ARGuidanceElement) -> np.ndarray:
        """渲染操作引导"""
        # 绘制步骤指示器
        if element.total_steps > 0:
            step_text = f"Step {element.step_number}/{element.total_steps}"
            cv2.putText(frame, step_text, (20, 40), self.font, 1, (0, 255, 255), 2)
        
        # 绘制指令文本
        # 自动换行
        words = element.instruction_text.split()
        lines = []
        current_line = ""
        max_width = self.canvas_width - 40
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            (w, _), _ = cv2.getTextSize(test_line, self.font, 0.8, 2)
            if w > max_width:
                lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        if current_line:
            lines.append(current_line)
        
        # 绘制背景面板
        panel_height = len(lines) * 35 + 30
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, self.canvas_height - panel_height - 10),
                     (self.canvas_width - 10, self.canvas_height - 10),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 绘制文本
        y = self.canvas_height - panel_height + 25
        for line in lines:
            cv2.putText(frame, line, (20, y), self.font, 0.8, (255, 255, 255), 2)
            y += 35
        
        # 高亮区域
        if element.highlight_region:
            bbox = element.highlight_region
            overlay = frame.copy()
            cv2.rectangle(overlay,
                         (int(bbox.x), int(bbox.y)),
                         (int(bbox.x + bbox.width), int(bbox.y + bbox.height)),
                         (0, 255, 255), 3)
            
            # 脉冲效果
            pulse = int((time.time() * 5) % 10)
            cv2.rectangle(overlay,
                         (int(bbox.x) - pulse, int(bbox.y) - pulse),
                         (int(bbox.x + bbox.width) + pulse, 
                          int(bbox.y + bbox.height) + pulse),
                         (0, 255, 255), 1)
            
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        return frame
    
    def _render_data_viz(self, frame: np.ndarray, element: ARDataVizElement) -> np.ndarray:
        """渲染数据可视化"""
        if not element.position_2d:
            return frame
        
        x = int(element.position_2d.x)
        y = int(element.position_2d.y)
        w = int(element.width)
        h = int(element.height)
        
        # 绘制背景
        cv2.rectangle(frame, (x, y), (x + w, y + h), (30, 30, 30), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)
        
        # 简化版条形图
        if element.viz_type == "bar" and element.data:
            values = element.data.get('values', [])
            labels = element.data.get('labels', [])
            max_val = max(values) if values else 1
            
            bar_width = w // (len(values) + 1) if values else w
            
            for i, (val, label) in enumerate(zip(values, labels)):
                bar_height = int((val / max_val) * (h - 50))
                bar_x = x + 20 + i * bar_width
                bar_y = y + h - 30 - bar_height
                
                # 渐变色
                color = (int(255 * (1 - val/max_val)), int(255 * val/max_val), 100)
                cv2.rectangle(frame, (bar_x, bar_y), 
                            (bar_x + bar_width - 5, y + h - 30), color, -1)
                
                # 标签
                cv2.putText(frame, str(label), (bar_x, y + h - 10),
                           self.font, 0.4, (200, 200, 200), 1)
        
        return frame


class ARExperimentAssistant:
    """AR实验助手"""
    
    def __init__(self):
        self.overlay = ARDataOverlay()
        self.scene_understanding = ARSceneUnderstanding()
        
        # 实验协议
        self.protocols: Dict[str, List[Dict[str, Any]]] = {}
        self.current_protocol: Optional[str] = None
        self.current_step: int = 0
        
        # 实时数据
        self.sensor_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # 安全监测
        self.safety_alerts: List[Dict[str, Any]] = []
        self.safety_enabled: bool = True
        
    def load_protocol(self, protocol_id: str, steps: List[Dict[str, Any]]) -> bool:
        """加载实验协议"""
        self.protocols[protocol_id] = steps
        logger.info(f"Loaded protocol '{protocol_id}' with {len(steps)} steps")
        return True
    
    def start_protocol(self, protocol_id: str) -> bool:
        """开始实验协议"""
        if protocol_id not in self.protocols:
            return False
        
        self.current_protocol = protocol_id
        self.current_step = 0
        
        protocol = self.protocols[protocol_id]
        first_step = protocol[0]
        
        self.overlay.create_guidance(
            first_step.get('instruction', 'Start experiment'),
            step=1,
            total=len(protocol)
        )
        
        logger.info(f"Started protocol: {protocol_id}")
        return True
    
    def next_step(self) -> bool:
        """进入下一步"""
        if not self.current_protocol:
            return False
        
        protocol = self.protocols[self.current_protocol]
        self.current_step += 1
        
        if self.current_step >= len(protocol):
            self.overlay.create_text(
                "Protocol Complete!",
                ARVector2(100, 100),
                font_size=36,
                color=(0, 255, 0)
            )
            self.current_protocol = None
            return True
        
        step = protocol[self.current_step]
        self.overlay.clear_elements(AROverlayType.GUIDANCE)
        self.overlay.create_guidance(
            step.get('instruction', 'Continue'),
            step=self.current_step + 1,
            total=len(protocol)
        )
        
        return True
    
    def update_sensor_data(self, sensor_type: str, value: float) -> None:
        """更新传感器数据"""
        self.sensor_data[sensor_type].append({
            'timestamp': time.time(),
            'value': value
        })
        
        # 检查安全阈值
        if self.safety_enabled:
            self._check_safety_thresholds(sensor_type, value)
    
    def _check_safety_thresholds(self, sensor_type: str, value: float) -> None:
        """检查安全阈值"""
        thresholds = {
            'temperature': {'max': 500, 'unit': '°C'},
            'pressure': {'max': 100, 'unit': 'bar'},
            'radiation': {'max': 0.1, 'unit': 'mSv/h'},
        }
        
        if sensor_type in thresholds:
            threshold = thresholds[sensor_type]
            if value > threshold['max']:
                alert = {
                    'type': sensor_type,
                    'value': value,
                    'threshold': threshold['max'],
                    'unit': threshold['unit'],
                    'timestamp': time.time()
                }
                self.safety_alerts.append(alert)
                
                # 创建警告叠加
                self.overlay.create_text(
                    f"⚠ WARNING: {sensor_type.upper()} {value}{threshold['unit']} "
                    f"exceeds {threshold['max']}{threshold['unit']}!",
                    ARVector2(50, 50),
                    font_size=28,
                    color=(255, 0, 0)
                )
                
                logger.warning(f"Safety alert: {alert}")
    
    def get_real_time_dashboard(self) -> Dict[str, Any]:
        """获取实时数据仪表板"""
        dashboard = {}
        
        for sensor_type, data in self.sensor_data.items():
            if data:
                values = [d['value'] for d in data]
                dashboard[sensor_type] = {
                    'current': values[-1],
                    'average': np.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'trend': values[-10:] if len(values) >= 10 else values
                }
        
        return dashboard
    
    def render_assistant_view(self, frame: np.ndarray) -> np.ndarray:
        """渲染助手视图"""
        # 更新场景理解
        self.scene_understanding.update_tracking(frame)
        
        # 添加传感器数据叠加
        dashboard = self.get_real_time_dashboard()
        y_offset = 100
        
        for sensor, data in dashboard.items():
            text = f"{sensor}: {data['current']:.2f} (avg: {data['average']:.2f})"
            self.overlay.create_text(
                text,
                ARVector2(self.overlay.canvas_width - 350, y_offset),
                font_size=18,
                color=(0, 255, 255),
                background=(0, 0, 0, 180)
            )
            y_offset += 30
        
        # 渲染所有叠加
        return self.overlay.render_to_frame(frame)


class ARRemoteExpert:
    """远程专家指导系统"""
    
    def __init__(self):
        self.overlay = ARDataOverlay()
        self.active_sessions: Dict[str, ARRemoteExpertSession] = {}
        self.pending_invitations: List[Dict[str, Any]] = []
        
        # 音视频通信
        self.audio_enabled = True
        self.video_enabled = True
        
        # 标注同步
        self.annotation_sync_interval = 0.1  # 100ms
        
    async def create_session(self, expert_id: str, expert_name: str,
                            local_user_id: str) -> str:
        """创建远程指导会话"""
        session_id = f"ar_session_{int(time.time() * 1000)}"
        
        session = ARRemoteExpertSession(
            session_id=session_id,
            expert_id=expert_id,
            expert_name=expert_name,
            local_user_id=local_user_id
        )
        
        self.active_sessions[session_id] = session
        logger.info(f"Created AR expert session: {session_id}")
        
        return session_id
    
    def end_session(self, session_id: str) -> bool:
        """结束会话"""
        if session_id in self.active_sessions:
            session = self.active_sessions.pop(session_id)
            logger.info(f"Ended AR expert session: {session_id}")
            return True
        return False
    
    def add_expert_annotation(self, session_id: str, annotation_type: str,
                             position: ARVector2, content: str) -> Optional[str]:
        """添加专家标注"""
        session = self.active_sessions.get(session_id)
        if not session:
            return None
        
        element_id = self.overlay.create_text(
            content,
            position,
            font_size=20,
            color=(255, 255, 0)  # 黄色表示专家标注
        )
        
        # 记录到会话
        if element_id in self.overlay.elements:
            session.add_annotation(self.overlay.elements[element_id])
        
        return element_id
    
    def update_expert_pointer(self, session_id: str, position: ARVector2) -> bool:
        """更新专家指针位置"""
        session = self.active_sessions.get(session_id)
        if not session:
            return False
        
        session.pointer_position = position
        return True
    
    def add_chat_message(self, session_id: str, text: str, 
                        sender: str) -> bool:
        """添加聊天消息"""
        session = self.active_sessions.get(session_id)
        if not session:
            return False
        
        session.add_message(text, sender)
        
        # 在AR中显示消息
        msg_text = f"{sender}: {text}"
        self.overlay.create_text(
            msg_text,
            ARVector2(50, self.overlay.canvas_height - 150),
            font_size=18,
            color=(200, 200, 200),
            background=(0, 0, 0, 150)
        )
        
        return True
    
    def render_expert_view(self, frame: np.ndarray, session_id: str) -> np.ndarray:
        """渲染专家指导视图"""
        session = self.active_sessions.get(session_id)
        if not session:
            return frame
        
        # 绘制专家指针
        if session.pointer_position:
            px, py = int(session.pointer_position.x), int(session.pointer_position.y)
            # 绘制激光指针效果
            cv2.circle(frame, (px, py), 10, (255, 0, 0), 2)
            cv2.line(frame, (px - 15, py), (px + 15, py), (255, 0, 0), 2)
            cv2.line(frame, (px, py - 15), (px, py + 15), (255, 0, 0), 2)
        
        # 绘制会话信息
        info_text = f"Expert: {session.expert_name} | Duration: {int(time.time() - session.start_time)}s"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 渲染所有叠加元素
        return self.overlay.render_to_frame(frame)


class AROverlayManager:
    """AR叠加管理器主类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.overlay = ARDataOverlay(
            canvas_width=self.config.get('width', 1920),
            canvas_height=self.config.get('height', 1080)
        )
        self.scene = ARSceneUnderstanding()
        self.experiment_assistant = ARExperimentAssistant()
        self.remote_expert = ARRemoteExpert()
        
        self.camera_matrix: Optional[np.ndarray] = None
        self.frame_count = 0
        self.fps = 0
        self.last_fps_update = time.time()
        
    def initialize(self) -> bool:
        """初始化AR系统"""
        logger.info("Initializing AR Overlay System...")
        
        # 初始化场景理解
        self.scene.initialize_detectors()
        
        # 加载相机标定参数
        self._load_camera_calibration()
        
        logger.info("AR Overlay System initialized")
        return True
    
    def _load_camera_calibration(self) -> None:
        """加载相机标定参数"""
        # 使用默认参数或从文件加载
        fx = fy = 1000  # 焦距
        cx = self.overlay.canvas_width / 2
        cy = self.overlay.canvas_height / 2
        
        self.camera_matrix = np.array([
            [fx, 0, cx, 0],
            [0, fy, cy, 0],
            [0, 0, 1, 0]
        ])
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """处理单帧图像"""
        self.frame_count += 1
        
        # 更新FPS
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_update = current_time
        
        # 场景理解
        tracked_objects = self.scene.update_tracking(frame)
        markers = self.scene.detect_markers(frame)
        
        # 渲染AR叠加
        output = self.overlay.render_to_frame(frame, self.camera_matrix)
        
        # 添加FPS显示
        cv2.putText(output, f"AR FPS: {self.fps}", 
                   (self.overlay.canvas_width - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return output
    
    def create_measurement(self, start_3d: ARVector3, end_3d: ARVector3) -> str:
        """创建3D测量"""
        return self.overlay.create_measurement(start_3d, end_3d)
    
    def create_annotation(self, position_3d: ARVector3, label: str) -> str:
        """创建3D标注"""
        return self.overlay.create_annotation(position_3d, label)
    
    def start_experiment_protocol(self, protocol_id: str) -> bool:
        """开始实验协议"""
        return self.experiment_assistant.start_protocol(protocol_id)
    
    async def start_remote_session(self, expert_id: str, expert_name: str,
                                   local_user_id: str) -> str:
        """开始远程会话"""
        return await self.remote_expert.create_session(
            expert_id, expert_name, local_user_id
        )


# 示例和测试
def example_usage():
    """使用示例"""
    print("=" * 60)
    print("AR Overlay Example - Metaverse Materials Laboratory")
    print("=" * 60)
    
    # 初始化AR管理器
    ar = AROverlayManager(config={'width': 1280, 'height': 720})
    ar.initialize()
    
    # 创建示例视频帧
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # 添加一些模拟内容到帧
    cv2.rectangle(frame, (100, 100), (300, 300), (100, 100, 100), -1)
    cv2.circle(frame, (900, 400), 50, (150, 150, 150), -1)
    
    # 测试文本叠加
    text_id = ar.overlay.create_text(
        "AR Overlay Active - Materials Lab",
        ARVector2(50, 50),
        font_size=28,
        color=(0, 255, 255),
        background=(0, 0, 0, 200)
    )
    print(f"✓ Created text overlay: {text_id}")
    
    # 测试3D标注
    anno_id = ar.create_annotation(
        ARVector3(200, 200, 1.5),
        "Sample Holder"
    )
    print(f"✓ Created 3D annotation: {anno_id}")
    
    # 测试测量
    meas_id = ar.create_measurement(
        ARVector3(100, 100, 1.0),
        ARVector3(300, 300, 1.0),
        "mm"
    )
    print(f"✓ Created measurement: {meas_id}")
    
    # 测试数据可视化
    viz_data = {
        'values': [23.5, 45.2, 67.8, 34.1, 89.3],
        'labels': ['A', 'B', 'C', 'D', 'E']
    }
    viz_id = ar.overlay.create_data_viz(
        viz_data,
        ARVector2(50, 400),
        "bar"
    )
    print(f"✓ Created data visualization: {viz_id}")
    
    # 测试实验协议
    protocol_steps = [
        {'instruction': '准备样品，确保表面清洁'},
        {'instruction': '将样品放置在测试台上'},
        {'instruction': '调整显微镜焦距'},
        {'instruction': '开始数据采集'},
    ]
    ar.experiment_assistant.load_protocol("sample_prep", protocol_steps)
    ar.start_experiment_protocol("sample_prep")
    print(f"✓ Started experiment protocol: sample_prep ({len(protocol_steps)} steps)")
    
    # 测试传感器数据更新
    ar.experiment_assistant.update_sensor_data("temperature", 25.5)
    ar.experiment_assistant.update_sensor_data("temperature", 26.0)
    ar.experiment_assistant.update_sensor_data("pressure", 1.02)
    print(f"✓ Updated sensor data")
    
    dashboard = ar.experiment_assistant.get_real_time_dashboard()
    print(f"  Dashboard: {dashboard}")
    
    # 处理帧
    output_frame = ar.process_frame(frame)
    print(f"✓ Processed frame with AR overlay")
    print(f"  Output shape: {output_frame.shape}")
    
    # 测试场景理解
    objects = ar.scene.detect_objects(frame)
    print(f"✓ Detected {len(objects)} objects in scene")
    
    print("\n" + "=" * 60)
    print("AR Overlay Example Complete")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()
