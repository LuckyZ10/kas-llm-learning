#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Immersive Visualization Module for Metaverse Materials Laboratory
沉浸式可视化模块 - 元宇宙材料实验室

Provides 3D volume rendering, spatiotemporal data animation, and haptic feedback
for materials science research.

Author: XR Expert Team
Version: 1.0.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from enum import Enum, auto
import logging
from collections import deque
import time
import asyncio
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VolumeRenderMode(Enum):
    """体渲染模式"""
    DIRECT_VOLUME = "direct"           # 直接体渲染
    MAXIMUM_INTENSITY = "mip"          # 最大强度投影
    AVERAGE_INTENSITY = "aip"          # 平均强度投影
    SURFACE_RENDERING = "surface"      # 表面渲染
    ISOSURFACE = "isosurface"          # 等值面
    SLICE = "slice"                    # 切片


class ColorMap(Enum):
    """颜色映射"""
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    TURBO = "turbo"
    JET = "jet"
    HOT = "hot"
    COOL = "cool"
    SEISMIC = "seismic"
    GREYS = "greys"
    CUSTOM = "custom"


class AnimationMode(Enum):
    """动画模式"""
    LOOP = "loop"                      # 循环
    PING_PONG = "ping_pong"            # 往返
    ONCE = "once"                      # 单次
    REVERSE = "reverse"                # 反向


@dataclass
class VolumeData:
    """体数据"""
    data: np.ndarray                    # 3D数组
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # 体素间距
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # 数据范围
    data_min: float = field(init=False)
    data_max: float = field(init=False)
    
    # 元数据
    unit: str = ""
    description: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        self.data_min = float(np.min(self.data))
        self.data_max = float(np.max(self.data))
        
    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape
    
    @property
    def physical_size(self) -> Tuple[float, float, float]:
        """物理尺寸"""
        return (
            self.shape[0] * self.spacing[0],
            self.shape[1] * self.spacing[1],
            self.shape[2] * self.spacing[2]
        )


@dataclass
class TransferFunction:
    """传递函数用于体渲染"""
    # 颜色控制点 (data_value, (r, g, b, a))
    color_points: List[Tuple[float, Tuple[float, float, float, float]]] = field(
        default_factory=list
    )
    
    # 标量值范围
    value_range: Tuple[float, float] = (0.0, 1.0)
    
    def __post_init__(self):
        if not self.color_points:
            # 默认传递函数
            self.color_points = [
                (0.0, (0.0, 0.0, 0.0, 0.0)),
                (0.5, (0.5, 0.5, 1.0, 0.3)),
                (1.0, (1.0, 0.0, 0.0, 1.0))
            ]
    
    def get_color(self, value: float) -> Tuple[float, float, float, float]:
        """获取标量值对应的颜色"""
        # 归一化
        normalized = (value - self.value_range[0]) / (
            self.value_range[1] - self.value_range[0]
        )
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # 查找插值
        points = sorted(self.color_points, key=lambda p: p[0])
        
        for i in range(len(points) - 1):
            p1, c1 = points[i]
            p2, c2 = points[i + 1]
            
            if p1 <= normalized <= p2:
                t = (normalized - p1) / (p2 - p1) if p2 != p1 else 0
                r = c1[0] + t * (c2[0] - c1[0])
                g = c1[1] + t * (c2[1] - c1[1])
                b = c1[2] + t * (c2[2] - c1[2])
                a = c1[3] + t * (c2[3] - c1[3])
                return (r, g, b, a)
        
        return points[-1][1] if points else (0.0, 0.0, 0.0, 0.0)
    
    def add_control_point(self, value: float, color: Tuple[float, float, float, float]):
        """添加控制点"""
        self.color_points.append((value, color))
        self.color_points.sort(key=lambda p: p[0])


@dataclass
class RenderSettings:
    """渲染设置"""
    render_mode: VolumeRenderMode = VolumeRenderMode.DIRECT_VOLUME
    color_map: ColorMap = ColorMap.VIRIDIS
    
    # 采样设置
    sample_rate: float = 1.0           # 采样率
    step_size: float = 0.5             # 步长
    max_steps: int = 500               # 最大步数
    
    # 光照
    lighting_enabled: bool = True
    ambient_light: float = 0.3
    diffuse_light: float = 0.7
    specular_light: float = 0.3
    light_direction: Tuple[float, float, float] = (0.5, 0.5, 1.0)
    
    # 效果
    shadows_enabled: bool = False
    fog_enabled: bool = False
    fog_density: float = 0.01
    
    # 输出
    image_width: int = 1920
    image_height: int = 1080
    background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class CameraSettings:
    """相机设置"""
    position: Tuple[float, float, float] = (0.0, 0.0, -5.0)
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    
    # 投影参数
    fov: float = 60.0                  # 视野角度
    near_plane: float = 0.1
    far_plane: float = 100.0
    
    # 运动
    auto_rotate: bool = False
    rotation_speed: float = 1.0        # 度/秒
    
    def get_view_matrix(self) -> np.ndarray:
        """获取视图矩阵"""
        pos = np.array(self.position)
        tgt = np.array(self.target)
        up = np.array(self.up)
        
        forward = tgt - pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        view = np.eye(4)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        view[:3, 3] = -np.array([
            np.dot(right, pos),
            np.dot(up, pos),
            np.dot(forward, pos)
        ])
        
        return view
    
    def get_projection_matrix(self) -> np.ndarray:
        """获取投影矩阵"""
        f = 1.0 / np.tan(np.radians(self.fov) / 2)
        aspect = 16.0 / 9.0
        
        proj = np.zeros((4, 4))
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (self.far_plane + self.near_plane) / (self.near_plane - self.far_plane)
        proj[2, 3] = (2 * self.far_plane * self.near_plane) / (self.near_plane - self.far_plane)
        proj[3, 2] = -1
        
        return proj


class VolumeRenderer:
    """体渲染器"""
    
    def __init__(self):
        self.volume_data: Optional[VolumeData] = None
        self.transfer_function = TransferFunction()
        self.render_settings = RenderSettings()
        self.camera = CameraSettings()
        
        # 渲染缓存
        self.render_cache: Dict[str, np.ndarray] = {}
        self.cache_enabled: bool = True
        
    def load_volume(self, data: np.ndarray, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                   origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                   unit: str = "") -> VolumeData:
        """加载体数据"""
        self.volume_data = VolumeData(
            data=data,
            spacing=spacing,
            origin=origin,
            unit=unit
        )
        
        # 更新传递函数范围
        self.transfer_function.value_range = (
            self.volume_data.data_min,
            self.volume_data.data_max
        )
        
        # 清空缓存
        self.render_cache.clear()
        
        logger.info(f"Loaded volume: shape={data.shape}, range=[{self.volume_data.data_min:.4f}, "
                   f"{self.volume_data.data_max:.4f}]")
        
        return self.volume_data
    
    def load_from_density_field(self, density_field: np.ndarray,
                                normalize: bool = True) -> VolumeData:
        """从密度场加载"""
        data = density_field.copy()
        
        if normalize:
            data_min, data_max = np.min(data), np.max(data)
            if data_max > data_min:
                data = (data - data_min) / (data_max - data_min)
        
        return self.load_volume(data)
    
    def create_transfer_function(self, preset: str = "default") -> TransferFunction:
        """创建预设传递函数"""
        tf = TransferFunction()
        
        if preset == "electron_density":
            tf.color_points = [
                (0.0, (0.0, 0.0, 0.0, 0.0)),
                (0.1, (0.0, 0.0, 0.5, 0.1)),
                (0.3, (0.0, 0.5, 1.0, 0.3)),
                (0.6, (0.5, 1.0, 0.5, 0.6)),
                (1.0, (1.0, 0.0, 0.0, 1.0))
            ]
        elif preset == "temperature":
            tf.color_points = [
                (0.0, (0.0, 0.0, 1.0, 0.1)),
                (0.3, (0.0, 1.0, 1.0, 0.3)),
                (0.5, (0.0, 1.0, 0.0, 0.5)),
                (0.7, (1.0, 1.0, 0.0, 0.7)),
                (1.0, (1.0, 0.0, 0.0, 1.0))
            ]
        elif preset == "stress":
            tf.color_points = [
                (0.0, (0.0, 0.0, 0.0, 0.0)),
                (0.5, (0.0, 1.0, 0.0, 0.5)),
                (0.8, (1.0, 1.0, 0.0, 0.8)),
                (1.0, (1.0, 0.0, 0.0, 1.0))
            ]
        
        self.transfer_function = tf
        return tf
    
    def render_slice(self, axis: int, position: float) -> np.ndarray:
        """渲染切片"""
        if self.volume_data is None:
            return np.zeros((512, 512, 3), dtype=np.uint8)
        
        # 获取切片索引
        idx = int(position * self.volume_data.shape[axis])
        idx = np.clip(idx, 0, self.volume_data.shape[axis] - 1)
        
        # 提取切片
        if axis == 0:
            slice_data = self.volume_data.data[idx, :, :]
        elif axis == 1:
            slice_data = self.volume_data.data[:, idx, :]
        else:
            slice_data = self.volume_data.data[:, :, idx]
        
        # 应用传递函数
        colored_slice = np.zeros((*slice_data.shape, 4), dtype=np.float32)
        
        for i in range(slice_data.shape[0]):
            for j in range(slice_data.shape[1]):
                color = self.transfer_function.get_color(slice_data[i, j])
                colored_slice[i, j] = color
        
        # 转换为RGB
        rgb = (colored_slice[:, :, :3] * 255).astype(np.uint8)
        
        return rgb
    
    def render_mip(self, axis: int = 2) -> np.ndarray:
        """最大强度投影"""
        if self.volume_data is None:
            return np.zeros((512, 512, 3), dtype=np.uint8)
        
        # 沿指定轴计算最大强度
        mip = np.max(self.volume_data.data, axis=axis)
        
        # 归一化
        mip_normalized = (mip - self.volume_data.data_min) / (
            self.volume_data.data_max - self.volume_data.data_min
        )
        
        # 应用颜色映射
        colored = self._apply_colormap(mip_normalized)
        
        return (colored * 255).astype(np.uint8)
    
    def _apply_colormap(self, data: np.ndarray, colormap: ColorMap = ColorMap.VIRIDIS) -> np.ndarray:
        """应用颜色映射"""
        # 简化的颜色映射实现
        h, w = data.shape
        colored = np.zeros((h, w, 3), dtype=np.float32)
        
        if colormap == ColorMap.VIRIDIS:
            # 简化的viridis近似
            colored[:, :, 0] = np.clip(0.8 * data + 0.2 * np.sin(data * np.pi), 0, 1)
            colored[:, :, 1] = data
            colored[:, :, 2] = np.clip(1.0 - data * 0.8, 0, 1)
        elif colormap == ColorMap.HOT:
            colored[:, :, 0] = np.clip(data * 3, 0, 1)
            colored[:, :, 1] = np.clip((data - 0.33) * 3, 0, 1)
            colored[:, :, 2] = np.clip((data - 0.66) * 3, 0, 1)
        else:
            # 灰度
            colored[:, :, :] = data[:, :, np.newaxis]
        
        return colored
    
    def extract_isosurface(self, threshold: float) -> Dict[str, Any]:
        """提取等值面"""
        if self.volume_data is None:
            return {}
        
        # 使用marching cubes算法
        try:
            from skimage import measure
            
            verts, faces, normals, values = measure.marching_cubes(
                self.volume_data.data,
                level=threshold,
                spacing=self.volume_data.spacing
            )
            
            # 计算顶点颜色
            vertex_colors = []
            for vert in verts:
                # 从标量值获取颜色
                value = self._interpolate_value(vert)
                color = self.transfer_function.get_color(value)
                vertex_colors.append(color[:3])
            
            return {
                'vertices': verts.tolist(),
                'faces': faces.tolist(),
                'normals': normals.tolist(),
                'colors': vertex_colors,
                'vertex_count': len(verts),
                'face_count': len(faces)
            }
            
        except ImportError:
            logger.warning("scikit-image not available for isosurface extraction")
            return {
                'error': 'scikit-image required for isosurface extraction'
            }
    
    def _interpolate_value(self, position: np.ndarray) -> float:
        """在指定位置插值获取标量值"""
        if self.volume_data is None:
            return 0.0
        
        # 简单的三线性插值
        x, y, z = position
        x_idx = int(x / self.volume_data.spacing[0])
        y_idx = int(y / self.volume_data.spacing[1])
        z_idx = int(z / self.volume_data.spacing[2])
        
        x_idx = np.clip(x_idx, 0, self.volume_data.shape[0] - 1)
        y_idx = np.clip(y_idx, 0, self.volume_data.shape[1] - 1)
        z_idx = np.clip(z_idx, 0, self.volume_data.shape[2] - 1)
        
        return float(self.volume_data.data[x_idx, y_idx, z_idx])
    
    def compute_gradient(self) -> np.ndarray:
        """计算梯度场"""
        if self.volume_data is None:
            return np.array([])
        
        # 计算三个方向的梯度
        gradient = np.gradient(self.volume_data.data)
        
        # 合并为向量场
        grad_magnitude = np.sqrt(
            gradient[0]**2 + gradient[1]**2 + gradient[2]**2
        )
        
        return grad_magnitude
    
    def apply_filter(self, filter_type: str, **params) -> np.ndarray:
        """应用滤波器"""
        if self.volume_data is None:
            return np.array([])
        
        data = self.volume_data.data.copy()
        
        if filter_type == "gaussian":
            sigma = params.get('sigma', 1.0)
            filtered = ndimage.gaussian_filter(data, sigma=sigma)
        elif filter_type == "median":
            size = params.get('size', 3)
            filtered = ndimage.median_filter(data, size=size)
        elif filter_type == "sobel":
            filtered = ndimage.sobel(data)
        elif filter_type == "laplace":
            filtered = ndimage.laplace(data)
        else:
            filtered = data
        
        return filtered


@dataclass
class AnimationFrame:
    """动画帧"""
    frame_index: int
    timestamp: float
    volume_data: Optional[VolumeData] = None
    camera_position: Optional[Tuple[float, float, float]] = None
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)


class SpatiotemporalAnimator:
    """时空数据动画器"""
    
    def __init__(self):
        self.frames: List[AnimationFrame] = []
        self.current_frame: int = 0
        self.is_playing: bool = False
        self.playback_speed: float = 1.0
        self.animation_mode: AnimationMode = AnimationMode.LOOP
        
        # 时间控制
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.time_step: float = 1.0
        
        # 回调
        self.on_frame_change: Optional[Callable] = None
        self.on_playback_end: Optional[Callable] = None
        
        # 渲染设置
        self.interpolate_frames: bool = True
        self.interpolation_method: str = "linear"
        
    def add_frame(self, volume_data: VolumeData, timestamp: Optional[float] = None,
                 **metadata) -> int:
        """添加帧"""
        frame_idx = len(self.frames)
        ts = timestamp if timestamp is not None else time.time()
        
        frame = AnimationFrame(
            frame_index=frame_idx,
            timestamp=ts,
            volume_data=volume_data,
            metadata=metadata
        )
        
        self.frames.append(frame)
        
        # 更新时间范围
        if frame_idx == 0:
            self.start_time = ts
        self.end_time = ts
        
        return frame_idx
    
    def load_timeseries(self, volumes: List[VolumeData], 
                       timestamps: Optional[List[float]] = None) -> None:
        """加载时间序列"""
        self.frames.clear()
        
        for i, vol in enumerate(volumes):
            ts = timestamps[i] if timestamps else i * self.time_step
            self.add_frame(vol, ts)
        
        logger.info(f"Loaded {len(volumes)} frames")
    
    def play(self) -> None:
        """播放动画"""
        self.is_playing = True
        logger.info("Animation started")
    
    def pause(self) -> None:
        """暂停动画"""
        self.is_playing = False
        logger.info("Animation paused")
    
    def stop(self) -> None:
        """停止动画"""
        self.is_playing = False
        self.current_frame = 0
        logger.info("Animation stopped")
    
    def seek_to_frame(self, frame_index: int) -> bool:
        """跳转到指定帧"""
        if 0 <= frame_index < len(self.frames):
            self.current_frame = frame_index
            if self.on_frame_change:
                self.on_frame_change(self.frames[frame_index])
            return True
        return False
    
    def seek_to_time(self, timestamp: float) -> bool:
        """跳转到指定时间"""
        # 找到最接近的帧
        closest_idx = 0
        min_diff = float('inf')
        
        for i, frame in enumerate(self.frames):
            diff = abs(frame.timestamp - timestamp)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
        
        return self.seek_to_frame(closest_idx)
    
    def update(self, delta_time: float) -> Optional[AnimationFrame]:
        """更新动画（每帧调用）"""
        if not self.is_playing or not self.frames:
            return None
        
        # 计算下一帧
        direction = -1 if self.animation_mode == AnimationMode.REVERSE else 1
        next_frame = self.current_frame + direction * self.playback_speed * delta_time * 30
        
        if next_frame >= len(self.frames):
            if self.animation_mode == AnimationMode.LOOP:
                next_frame = 0
            elif self.animation_mode == AnimationMode.PING_PONG:
                next_frame = len(self.frames) - 2
                self.playback_speed = -abs(self.playback_speed)
            elif self.animation_mode == AnimationMode.ONCE:
                self.stop()
                if self.on_playback_end:
                    self.on_playback_end()
                return None
        elif next_frame < 0:
            if self.animation_mode == AnimationMode.PING_PONG:
                next_frame = 1
                self.playback_speed = abs(self.playback_speed)
            else:
                next_frame = len(self.frames) - 1
        
        self.current_frame = int(next_frame)
        
        # 获取帧（支持插值）
        frame = self._get_frame_at(self.current_frame)
        
        if self.on_frame_change:
            self.on_frame_change(frame)
        
        return frame
    
    def _get_frame_at(self, frame_index: float) -> AnimationFrame:
        """获取指定位置的帧（支持插值）"""
        if not self.interpolate_frames:
            return self.frames[int(frame_index)]
        
        # 线性插值
        idx1 = int(frame_index)
        idx2 = min(idx1 + 1, len(self.frames) - 1)
        t = frame_index - idx1
        
        frame1 = self.frames[idx1]
        frame2 = self.frames[idx2]
        
        # 插值时间戳
        ts = frame1.timestamp + t * (frame2.timestamp - frame1.timestamp)
        
        # 插值体积数据
        if frame1.volume_data is not None and frame2.volume_data is not None:
            data1 = frame1.volume_data.data
            data2 = frame2.volume_data.data
            interp_data = data1 * (1 - t) + data2 * t
            
            volume = VolumeData(
                data=interp_data,
                spacing=frame1.volume_data.spacing,
                origin=frame1.volume_data.origin
            )
        else:
            volume = frame1.volume_data
        
        return AnimationFrame(
            frame_index=frame_index,
            timestamp=ts,
            volume_data=volume,
            metadata=frame1.metadata
        )
    
    def export_video(self, output_path: str, fps: int = 30,
                    quality: str = "high") -> bool:
        """导出视频"""
        try:
            import cv2
            
            if not self.frames:
                return False
            
            # 确定视频尺寸
            first_frame = self.frames[0].volume_data
            if first_frame is None:
                return False
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (1920, 1080))
            
            # 渲染每一帧
            renderer = VolumeRenderer()
            
            for frame in self.frames:
                if frame.volume_data:
                    renderer.load_volume(
                        frame.volume_data.data,
                        frame.volume_data.spacing
                    )
                    # 渲染MIP视图
                    image = renderer.render_mip()
                    # 调整尺寸
                    image_resized = cv2.resize(image, (1920, 1080))
                    writer.write(image_resized)
            
            writer.release()
            logger.info(f"Exported video to {output_path}")
            return True
            
        except ImportError:
            logger.error("OpenCV required for video export")
            return False
    
    def create_timeline_visualization(self) -> Dict[str, Any]:
        """创建时间线可视化"""
        timeline = {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.end_time - self.start_time,
            'frame_count': len(self.frames),
            'events': []
        }
        
        # 收集所有事件标记
        for frame in self.frames:
            if 'events' in frame.metadata:
                for event in frame.metadata['events']:
                    timeline['events'].append({
                        'timestamp': frame.timestamp,
                        'description': event
                    })
        
        return timeline


@dataclass
class HapticEvent:
    """触觉事件"""
    event_type: str  # "vibration", "force", "texture", "temperature"
    intensity: float  # 0-1
    duration: float   # 秒
    frequency: float = 0.0  # Hz (用于振动)
    
    # 空间信息
    position: Optional[Tuple[float, float, float]] = None
    direction: Optional[Tuple[float, float, float]] = None
    
    # 模式
    pattern: str = "continuous"  # continuous, pulse, wave
    
    # 时间戳
    timestamp: float = field(default_factory=time.time)


class HapticFeedbackSystem:
    """触觉反馈系统"""
    
    def __init__(self):
        self.device_connected: bool = False
        self.device_type: str = "none"  # "gamepad", "vr_controller", "haptic_glove"
        
        # 触觉缓冲区
        self.event_queue: deque = deque(maxlen=100)
        
        # 当前状态
        self.current_intensity: float = 0.0
        self.is_active: bool = False
        
        # 材料触感数据库
        self.material_textures: Dict[str, Dict[str, Any]] = {
            "metal_smooth": {
                "roughness": 0.1,
                "hardness": 0.9,
                "temperature": 0.2,
                "vibration_pattern": "smooth"
            },
            "metal_rough": {
                "roughness": 0.7,
                "hardness": 0.9,
                "temperature": 0.2,
                "vibration_pattern": "coarse"
            },
            "crystal": {
                "roughness": 0.0,
                "hardness": 1.0,
                "temperature": -0.1,
                "vibration_pattern": "sharp"
            },
            "polymer": {
                "roughness": 0.3,
                "hardness": 0.4,
                "temperature": 0.0,
                "vibration_pattern": "soft"
            },
            "ceramic": {
                "roughness": 0.2,
                "hardness": 0.95,
                "temperature": -0.05,
                "vibration_pattern": "rigid"
            }
        }
        
    def connect_device(self, device_type: str) -> bool:
        """连接触觉设备"""
        supported_devices = ["gamepad", "vr_controller", "haptic_glove", "force_feedback"]
        
        if device_type in supported_devices:
            self.device_type = device_type
            self.device_connected = True
            logger.info(f"Connected haptic device: {device_type}")
            return True
        
        logger.error(f"Unsupported haptic device: {device_type}")
        return False
    
    def disconnect(self) -> None:
        """断开设备"""
        self.device_connected = False
        self.device_type = "none"
        logger.info("Haptic device disconnected")
    
    def trigger_vibration(self, intensity: float, duration: float,
                         frequency: float = 50.0, pattern: str = "continuous") -> None:
        """触发振动反馈"""
        if not self.device_connected:
            return
        
        event = HapticEvent(
            event_type="vibration",
            intensity=np.clip(intensity, 0.0, 1.0),
            duration=duration,
            frequency=frequency,
            pattern=pattern
        )
        
        self.event_queue.append(event)
        self._send_to_device(event)
    
    def trigger_force_feedback(self, force_vector: Tuple[float, float, float],
                              duration: float) -> None:
        """触发力反馈"""
        if not self.device_connected:
            return
        
        intensity = np.linalg.norm(force_vector)
        
        event = HapticEvent(
            event_type="force",
            intensity=np.clip(intensity, 0.0, 1.0),
            duration=duration,
            direction=force_vector,
            pattern="continuous"
        )
        
        self.event_queue.append(event)
        self._send_to_device(event)
    
    def simulate_material_touch(self, material_type: str, 
                               pressure: float = 0.5) -> None:
        """模拟材料触感"""
        if not self.device_connected:
            return
        
        if material_type not in self.material_textures:
            logger.warning(f"Unknown material: {material_type}")
            return
        
        texture = self.material_textures[material_type]
        
        # 根据材料属性生成触觉反馈
        intensity = pressure * texture["hardness"]
        
        # 振动频率基于粗糙度
        base_freq = 50
        frequency = base_freq + texture["roughness"] * 200
        
        # 温度感知（可选）
        temp_effect = texture.get("temperature", 0)
        
        self.trigger_vibration(
            intensity=intensity,
            duration=0.1,
            frequency=frequency,
            pattern=texture["vibration_pattern"]
        )
        
        logger.debug(f"Simulated {material_type} touch: intensity={intensity:.2f}")
    
    def simulate_crystal_structure(self, structure_type: str,
                                  probe_position: np.ndarray) -> None:
        """模拟晶体结构触感"""
        if not self.device_connected:
            return
        
        # 根据结构类型生成不同的触觉反馈
        structure_patterns = {
            "cubic": {"symmetry": 4, "base_freq": 60},
            "hexagonal": {"symmetry": 6, "base_freq": 55},
            "tetragonal": {"symmetry": 4, "base_freq": 65},
            "orthorhombic": {"symmetry": 2, "base_freq": 70},
            "triclinic": {"symmetry": 1, "base_freq": 80}
        }
        
        pattern = structure_patterns.get(structure_type, structure_patterns["cubic"])
        
        # 周期性振动模拟晶体对称性
        for i in range(pattern["symmetry"]):
            self.trigger_vibration(
                intensity=0.3,
                duration=0.05,
                frequency=pattern["base_freq"] + i * 10,
                pattern="pulse"
            )
    
    def _send_to_device(self, event: HapticEvent) -> None:
        """发送事件到设备（模拟）"""
        # 实际实现中会调用设备API
        logger.debug(f"Haptic event: {event.event_type} intensity={event.intensity:.2f} "
                    f"duration={event.duration:.2f}s")
    
    def get_device_status(self) -> Dict[str, Any]:
        """获取设备状态"""
        return {
            'connected': self.device_connected,
            'device_type': self.device_type,
            'queue_size': len(self.event_queue),
            'current_intensity': self.current_intensity
        }


class ImmersiveVizManager:
    """沉浸式可视化管理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 子系统
        self.volume_renderer = VolumeRenderer()
        self.animator = SpatiotemporalAnimator()
        self.haptics = HapticFeedbackSystem()
        
        # 场景状态
        self.active_volumes: Dict[str, VolumeData] = {}
        self.render_callbacks: List[Callable] = []
        
        # 性能设置
        self.target_fps: int = self.config.get('target_fps', 60)
        self.quality_level: str = self.config.get('quality', 'high')
        
    def load_density_volume(self, data: np.ndarray, name: str = "",
                           spacing: Tuple[float, float, float] = (0.1, 0.1, 0.1)) -> VolumeData:
        """加载密度体数据"""
        volume = self.volume_renderer.load_volume(data, spacing)
        volume.description = name
        self.active_volumes[name] = volume
        return volume
    
    def load_md_trajectory(self, trajectory_data: List[np.ndarray],
                          timestamps: Optional[List[float]] = None) -> None:
        """加载分子动力学轨迹"""
        volumes = []
        for i, frame_data in enumerate(trajectory_data):
            vol = VolumeData(
                data=frame_data,
                spacing=(0.1, 0.1, 0.1),
                description=f"MD Frame {i}"
            )
            volumes.append(vol)
        
        self.animator.load_timeseries(volumes, timestamps)
        logger.info(f"Loaded MD trajectory with {len(volumes)} frames")
    
    def setup_crystal_visualization(self, charge_density: np.ndarray,
                                   fermi_surface: Optional[np.ndarray] = None) -> None:
        """设置晶体可视化"""
        # 加载电荷密度
        self.volume_renderer.load_volume(charge_density)
        
        # 设置电子密度传递函数
        self.volume_renderer.create_transfer_function("electron_density")
        
        # 如果有费米面数据
        if fermi_surface is not None:
            # 设置等值面渲染
            self.volume_renderer.render_settings.render_mode = VolumeRenderMode.ISOSURFACE
        
        logger.info("Crystal visualization setup complete")
    
    def play_animation(self) -> None:
        """播放动画"""
        self.animator.play()
    
    def pause_animation(self) -> None:
        """暂停动画"""
        self.animator.pause()
    
    def update(self, delta_time: float) -> Optional[np.ndarray]:
        """更新可视化（每帧调用）"""
        # 更新动画
        frame = self.animator.update(delta_time)
        
        if frame and frame.volume_data:
            # 渲染当前帧
            self.volume_renderer.load_volume(
                frame.volume_data.data,
                frame.volume_data.spacing
            )
            return self.volume_renderer.render_mip()
        
        return None
    
    def connect_haptics(self, device_type: str) -> bool:
        """连接触觉设备"""
        return self.haptics.connect_device(device_type)
    
    def trigger_material_touch(self, material_type: str) -> None:
        """触发材料触感"""
        self.haptics.simulate_material_touch(material_type)
    
    def export_frame(self, output_path: str, frame_type: str = "mip") -> bool:
        """导出当前帧"""
        try:
            from PIL import Image
            
            if frame_type == "mip":
                image = self.volume_renderer.render_mip()
            elif frame_type == "slice":
                image = self.volume_renderer.render_slice(2, 0.5)
            else:
                return False
            
            img = Image.fromarray(image)
            img.save(output_path)
            logger.info(f"Exported frame to {output_path}")
            return True
            
        except ImportError:
            logger.error("PIL required for image export")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'volumes_loaded': len(self.active_volumes),
            'animation_frames': len(self.animator.frames),
            'animation_playing': self.animator.is_playing,
            'current_frame': self.animator.current_frame,
            'haptics_connected': self.haptics.device_connected,
            'haptics_device': self.haptics.device_type
        }


# 示例使用
def example_usage():
    """使用示例"""
    print("=" * 60)
    print("Immersive Visualization Example - Materials Lab")
    print("=" * 60)
    
    # 创建可视化管理器
    viz = ImmersiveVizManager()
    
    # 创建示例体数据（模拟电子密度）
    size = 64
    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    z = np.linspace(-3, 3, size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # 创建高斯分布的电子密度
    electron_density = np.exp(-(X**2 + Y**2 + Z**2) / 2)
    # 添加一些特征（模拟原子）
    for center in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]:
        electron_density += 0.5 * np.exp(-((X-center[0])**2 + 
                                           (Y-center[1])**2 + 
                                           (Z-center[2])**2) / 0.5)
    
    # 加载体数据
    volume = viz.load_density_volume(
        electron_density,
        name="Si_Electron_Density",
        spacing=(0.1, 0.1, 0.1)
    )
    print(f"\n✓ Loaded volume data:")
    print(f"  Shape: {volume.shape}")
    print(f"  Range: [{volume.data_min:.4f}, {volume.data_max:.4f}]")
    print(f"  Physical size: {volume.physical_size}")
    
    # 设置传递函数
    viz.volume_renderer.create_transfer_function("electron_density")
    print(f"✓ Created electron density transfer function")
    
    # 渲染切片
    slice_xy = viz.volume_renderer.render_slice(axis=2, position=0.5)
    print(f"✓ Rendered XY slice: shape={slice_xy.shape}")
    
    # 渲染MIP
    mip = viz.volume_renderer.render_mip(axis=2)
    print(f"✓ Rendered MIP: shape={mip.shape}")
    
    # 计算梯度
    gradient = viz.volume_renderer.compute_gradient()
    print(f"✓ Computed gradient field: shape={gradient.shape}")
    
    # 提取等值面
    isosurface = viz.volume_renderer.extract_isosurface(threshold=0.5)
    if 'vertices' in isosurface:
        print(f"✓ Extracted isosurface:")
        print(f"  Vertices: {isosurface['vertex_count']}")
        print(f"  Faces: {isosurface['face_count']}")
    
    # 创建时间序列动画
    print(f"\n✓ Creating time series animation...")
    trajectory_frames = []
    for t in np.linspace(0, 2*np.pi, 30):
        # 创建随时间变化的密度场
        dynamic_density = electron_density * (1 + 0.3 * np.sin(t + X + Y))
        trajectory_frames.append(dynamic_density)
    
    viz.load_md_trajectory(trajectory_frames)
    print(f"  Loaded {len(trajectory_frames)} frames")
    
    # 播放动画
    viz.play_animation()
    print(f"✓ Animation playing")
    
    # 模拟更新
    for i in range(5):
        result = viz.update(0.033)  # 30fps
        if result is not None:
            print(f"  Frame {viz.animator.current_frame} rendered")
    
    # 创建时间线
    timeline = viz.animator.create_timeline_visualization()
    print(f"✓ Animation timeline: {timeline['duration']:.2f}s, "
          f"{timeline['frame_count']} frames")
    
    # 触觉反馈
    print(f"\n✓ Haptic feedback system:")
    viz.connect_haptics("vr_controller")
    
    # 模拟材料触感
    for material in ["metal_smooth", "crystal", "ceramic"]:
        viz.trigger_material_touch(material)
        print(f"  Simulated {material} touch")
    
    # 模拟晶体结构触感
    viz.haptics.simulate_crystal_structure("cubic", np.array([0, 0, 0]))
    print(f"  Simulated cubic crystal structure haptics")
    
    # 系统状态
    status = viz.get_system_status()
    print(f"\n✓ System status:")
    print(f"  Volumes: {status['volumes_loaded']}")
    print(f"  Animation: {status['animation_frames']} frames, "
          f"playing={status['animation_playing']}")
    print(f"  Haptics: {status['haptics_connected']} ({status['haptics_device']})")
    
    print("\n" + "=" * 60)
    print("Immersive Visualization Example Complete")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()
