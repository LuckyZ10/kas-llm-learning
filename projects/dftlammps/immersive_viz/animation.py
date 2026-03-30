#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatiotemporal Animation Module
时空数据动画模块

Provides advanced animation capabilities for materials simulation data visualization.

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
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterpolationMethod(Enum):
    """插值方法"""
    LINEAR = "linear"
    CUBIC = "cubic"
    SPLINE = "spline"
    NEAREST = "nearest"
    GAUSSIAN = "gaussian"


class TimeWarpMode(Enum):
    """时间扭曲模式"""
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"


@dataclass
class Keyframe:
    """关键帧"""
    time: float  # 归一化时间 0-1
    value: Any   # 任意值（位置、颜色、标量等）
    
    # 插值控制
    in_tangent: Optional[np.ndarray] = None
    out_tangent: Optional[np.ndarray] = None
    
    # 缓动
    ease_in: float = 0.0
    ease_out: float = 0.0
    
    # 元数据
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnimationTrack:
    """动画轨道"""
    track_id: str
    track_name: str
    target_property: str  # 目标属性路径，如 "atom.position.x"
    
    # 关键帧
    keyframes: List[Keyframe] = field(default_factory=list)
    
    # 插值设置
    interpolation: InterpolationMethod = InterpolationMethod.LINEAR
    time_warp: TimeWarpMode = TimeWarpMode.LINEAR
    
    # 变换函数
    pre_process: Optional[Callable[[Any], Any]] = None
    post_process: Optional[Callable[[Any], Any]] = None
    
    def sort_keyframes(self) -> None:
        """按时间排序关键帧"""
        self.keyframes.sort(key=lambda k: k.time)
    
    def add_keyframe(self, time: float, value: Any, **kwargs) -> None:
        """添加关键帧"""
        keyframe = Keyframe(time=time, value=value, **kwargs)
        self.keyframes.append(keyframe)
        self.sort_keyframes()
    
    def evaluate(self, t: float) -> Any:
        """在时间点t评估轨道值"""
        if not self.keyframes:
            return None
        
        # 应用时间扭曲
        t = self._apply_time_warp(t)
        
        # 查找关键帧区间
        if t <= self.keyframes[0].time:
            return self.keyframes[0].value
        if t >= self.keyframes[-1].time:
            return self.keyframes[-1].value
        
        # 找到相邻关键帧
        for i in range(len(self.keyframes) - 1):
            k1, k2 = self.keyframes[i], self.keyframes[i + 1]
            if k1.time <= t <= k2.time:
                # 计算插值参数
                local_t = (t - k1.time) / (k2.time - k1.time) if k2.time != k1.time else 0
                return self._interpolate(k1.value, k2.value, local_t)
        
        return self.keyframes[-1].value
    
    def _apply_time_warp(self, t: float) -> float:
        """应用时间扭曲"""
        if self.time_warp == TimeWarpMode.LINEAR:
            return t
        elif self.time_warp == TimeWarpMode.EASE_IN:
            return t * t
        elif self.time_warp == TimeWarpMode.EASE_OUT:
            return 1 - (1 - t) * (1 - t)
        elif self.time_warp == TimeWarpMode.EASE_IN_OUT:
            if t < 0.5:
                return 2 * t * t
            else:
                return 1 - (-2 * t + 2)**2 / 2
        elif self.time_warp == TimeWarpMode.EXPONENTIAL:
            return (np.exp(t) - 1) / (np.e - 1)
        elif self.time_warp == TimeWarpMode.LOGARITHMIC:
            return np.log(1 + t * (np.e - 1))
        return t
    
    def _interpolate(self, v1: Any, v2: Any, t: float) -> Any:
        """在两个值之间插值"""
        # 预处理
        if self.pre_process:
            v1 = self.pre_process(v1)
            v2 = self.pre_process(v2)
        
        # 根据类型插值
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            result = v1 + t * (v2 - v1)
        elif isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
            result = self._interpolate_arrays(v1, v2, t)
        elif isinstance(v1, (tuple, list)) and isinstance(v2, (tuple, list)):
            result = tuple(v1[i] + t * (v2[i] - v1[i]) for i in range(len(v1)))
        else:
            result = v1 if t < 0.5 else v2
        
        # 后处理
        if self.post_process:
            result = self.post_process(result)
        
        return result
    
    def _interpolate_arrays(self, arr1: np.ndarray, arr2: np.ndarray, 
                           t: float) -> np.ndarray:
        """数组插值"""
        if self.interpolation == InterpolationMethod.LINEAR:
            return arr1 + t * (arr2 - arr1)
        elif self.interpolation == InterpolationMethod.CUBIC:
            # Catmull-Rom样条插值
            t2 = t * t
            t3 = t2 * t
            return arr1 * (2*t3 - 3*t2 + 1) + arr2 * (-2*t3 + 3*t2)
        elif self.interpolation == InterpolationMethod.GAUSSIAN:
            # 高斯混合
            weight = np.exp(-((t - 0.5)**2) / 0.1)
            return arr1 * (1 - weight) + arr2 * weight
        else:
            return arr1 + t * (arr2 - arr1)


@dataclass
class AnimationLayer:
    """动画层（用于分层动画）"""
    layer_id: str
    layer_name: str
    tracks: Dict[str, AnimationTrack] = field(default_factory=dict)
    
    # 层设置
    weight: float = 1.0  # 混合权重
    is_additive: bool = False
    is_enabled: bool = True
    
    # 时间偏移和缩放
    time_offset: float = 0.0
    time_scale: float = 1.0
    
    def add_track(self, track: AnimationTrack) -> None:
        """添加轨道"""
        self.tracks[track.track_id] = track
    
    def evaluate(self, time: float) -> Dict[str, Any]:
        """评估层中的所有轨道"""
        if not self.is_enabled:
            return {}
        
        # 应用时间变换
        adjusted_time = (time + self.time_offset) * self.time_scale
        
        results = {}
        for track_id, track in self.tracks.items():
            results[track_id] = track.evaluate(adjusted_time)
        
        return results


class SpatiotemporalAnimator:
    """增强版时空动画器"""
    
    def __init__(self):
        self.layers: Dict[str, AnimationLayer] = {}
        self.global_time: float = 0.0
        self.duration: float = 10.0  # 默认10秒
        self.is_playing: bool = False
        self.playback_speed: float = 1.0
        
        # 事件标记
        self.markers: List[Dict[str, Any]] = []
        
        # 回调
        self.on_marker_reached: Optional[Callable] = None
        self.on_animation_complete: Optional[Callable] = None
        
        # 录制
        self.is_recording: bool = False
        self.recorded_frames: List[Dict[str, Any]] = []
        
    def create_layer(self, layer_name: str, layer_id: Optional[str] = None) -> AnimationLayer:
        """创建动画层"""
        lid = layer_id or f"layer_{len(self.layers)}"
        layer = AnimationLayer(layer_id=lid, layer_name=layer_name)
        self.layers[lid] = layer
        return layer
    
    def add_marker(self, time: float, label: str, 
                  callback: Optional[Callable] = None) -> None:
        """添加时间标记"""
        self.markers.append({
            'time': time,
            'label': label,
            'callback': callback,
            'triggered': False
        })
        self.markers.sort(key=lambda m: m['time'])
    
    def update(self, delta_time: float) -> Dict[str, Any]:
        """更新动画"""
        if not self.is_playing:
            return {}
        
        # 更新时间
        self.global_time += delta_time * self.playback_speed
        
        # 检查循环
        if self.global_time > self.duration:
            self.global_time = 0.0
            # 重置标记
            for marker in self.markers:
                marker['triggered'] = False
            
            if self.on_animation_complete:
                self.on_animation_complete()
        
        # 检查标记
        self._check_markers()
        
        # 评估所有层
        results = self._evaluate_layers()
        
        # 录制
        if self.is_recording:
            self.recorded_frames.append({
                'time': self.global_time,
                'values': results
            })
        
        return results
    
    def _check_markers(self) -> None:
        """检查并触发时间标记"""
        for marker in self.markers:
            if not marker['triggered'] and self.global_time >= marker['time']:
                marker['triggered'] = True
                
                if marker['callback']:
                    marker['callback'](marker)
                
                if self.on_marker_reached:
                    self.on_marker_reached(marker)
    
    def _evaluate_layers(self) -> Dict[str, Any]:
        """评估所有层并混合结果"""
        all_results = {}
        
        for layer in self.layers.values():
            layer_results = layer.evaluate(self.global_time)
            
            # 应用层权重
            if layer.weight != 1.0:
                layer_results = self._apply_weight(layer_results, layer.weight)
            
            # 合并结果
            if layer.is_additive:
                all_results = self._additive_merge(all_results, layer_results)
            else:
                all_results.update(layer_results)
        
        return all_results
    
    def _apply_weight(self, results: Dict[str, Any], weight: float) -> Dict[str, Any]:
        """应用权重"""
        weighted = {}
        for key, value in results.items():
            if isinstance(value, (int, float)):
                weighted[key] = value * weight
            elif isinstance(value, np.ndarray):
                weighted[key] = value * weight
            else:
                weighted[key] = value
        return weighted
    
    def _additive_merge(self, base: Dict[str, Any], additive: Dict[str, Any]) -> Dict[str, Any]:
        """加法合并"""
        merged = base.copy()
        for key, value in additive.items():
            if key in merged:
                if isinstance(merged[key], np.ndarray) and isinstance(value, np.ndarray):
                    merged[key] = merged[key] + value
                elif isinstance(merged[key], (int, float)) and isinstance(value, (int, float)):
                    merged[key] = merged[key] + value
            else:
                merged[key] = value
        return merged
    
    def play(self) -> None:
        """播放"""
        self.is_playing = True
        logger.info("Animation started")
    
    def pause(self) -> None:
        """暂停"""
        self.is_playing = False
        logger.info("Animation paused")
    
    def stop(self) -> None:
        """停止"""
        self.is_playing = False
        self.global_time = 0.0
        logger.info("Animation stopped")
    
    def seek(self, time: float) -> None:
        """跳转到指定时间"""
        self.global_time = np.clip(time, 0, self.duration)
        # 重置标记状态
        for marker in self.markers:
            marker['triggered'] = marker['time'] < self.global_time
    
    def start_recording(self) -> None:
        """开始录制"""
        self.is_recording = True
        self.recorded_frames.clear()
        logger.info("Started recording animation")
    
    def stop_recording(self) -> List[Dict[str, Any]]:
        """停止录制并返回录制的帧"""
        self.is_recording = False
        logger.info(f"Stopped recording, captured {len(self.recorded_frames)} frames")
        return self.recorded_frames.copy()
    
    def export_to_json(self, filepath: str) -> bool:
        """导出动画到JSON"""
        try:
            data = {
                'duration': self.duration,
                'layers': {},
                'markers': [
                    {'time': m['time'], 'label': m['label']}
                    for m in self.markers
                ]
            }
            
            for lid, layer in self.layers.items():
                layer_data = {
                    'name': layer.layer_name,
                    'weight': layer.weight,
                    'tracks': {}
                }
                for tid, track in layer.tracks.items():
                    layer_data['tracks'][tid] = {
                        'target': track.target_property,
                        'keyframes': [
                            {'time': k.time, 'value': self._serialize_value(k.value)}
                            for k in track.keyframes
                        ]
                    }
                data['layers'][lid] = layer_data
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Exported animation to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export animation: {e}")
            return False
    
    def _serialize_value(self, value: Any) -> Any:
        """序列化值"""
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value


# 使用示例
def example_animation():
    """动画示例"""
    print("=" * 60)
    print("Spatiotemporal Animation Example")
    print("=" * 60)
    
    animator = SpatiotemporalAnimator()
    animator.duration = 5.0
    
    # 创建基础层
    base_layer = animator.create_layer("Base Animation")
    
    # 创建位置轨道
    position_track = AnimationTrack(
        track_id="pos_track",
        track_name="Atom Position",
        target_property="atom.position",
        interpolation=InterpolationMethod.CUBIC,
        time_warp=TimeWarpMode.EASE_IN_OUT
    )
    
    # 添加位置关键帧
    position_track.add_keyframe(0.0, np.array([0.0, 0.0, 0.0]))
    position_track.add_keyframe(0.25, np.array([1.0, 0.5, 0.0]))
    position_track.add_keyframe(0.5, np.array([2.0, 0.0, 0.5]))
    position_track.add_keyframe(0.75, np.array([1.0, -0.5, 0.0]))
    position_track.add_keyframe(1.0, np.array([0.0, 0.0, 0.0]))
    
    base_layer.add_track(position_track)
    
    # 创建缩放轨道
    scale_track = AnimationTrack(
        track_id="scale_track",
        track_name="Atom Scale",
        target_property="atom.scale"
    )
    scale_track.add_keyframe(0.0, 1.0)
    scale_track.add_keyframe(0.5, 1.5)
    scale_track.add_keyframe(1.0, 1.0)
    
    base_layer.add_track(scale_track)
    
    # 添加时间标记
    def on_phase_change(marker):
        print(f"  [Marker] Reached: {marker['label']}")
    
    animator.add_marker(1.25, "Phase 1 Complete", on_phase_change)
    animator.add_marker(2.5, "Phase 2 Complete", on_phase_change)
    animator.add_marker(3.75, "Phase 3 Complete", on_phase_change)
    
    # 播放动画
    animator.play()
    
    # 模拟播放
    dt = 0.1
    print(f"\n✓ Playing animation (duration: {animator.duration}s)")
    
    for i in range(60):
        results = animator.update(dt)
        if i % 10 == 0:
            pos = results.get('pos_track', np.array([0, 0, 0]))
            scale = results.get('scale_track', 1.0)
            print(f"  t={animator.global_time:.2f}s: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
                  f"scale={scale:.2f}")
    
    animator.stop()
    
    print(f"\n✓ Animation complete")
    print(f"  Total layers: {len(animator.layers)}")
    print(f"  Total markers: {len(animator.markers)}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    example_animation()
