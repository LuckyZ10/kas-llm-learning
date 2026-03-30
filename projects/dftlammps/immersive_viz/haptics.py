#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Haptic Feedback Module - Extended
触觉反馈扩展模块

Advanced haptic feedback for materials research and VR interaction.

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HapticDeviceType(Enum):
    """触觉设备类型"""
    VR_CONTROLLER = "vr_controller"
    HAPTIC_GLOVE = "haptic_glove"
    FORCE_FEEDBACK_ARM = "force_feedback_arm"
    VIBRATION_VEST = "vibration_vest"
    ULTRASONIC_ARRAY = "ultrasonic_array"
    TEMPERATURE_PAD = "temperature_pad"


class HapticPattern(Enum):
    """触觉模式"""
    CONTINUOUS = "continuous"
    PULSE = "pulse"
    WAVE = "wave"
    BURST = "burst"
    RAMP_UP = "ramp_up"
    RAMP_DOWN = "ramp_down"
    HEARTBEAT = "heartbeat"
    IMPACT = "impact"
    FRICTION = "friction"
    TEXTURE = "texture"


@dataclass
class HapticChannel:
    """触觉通道"""
    channel_id: str
    channel_type: str  # "vibration", "force", "temperature", "electrostimulation"
    
    # 当前状态
    current_intensity: float = 0.0
    current_frequency: float = 0.0
    is_active: bool = False
    
    # 配置
    min_intensity: float = 0.0
    max_intensity: float = 1.0
    frequency_range: Tuple[float, float] = (0, 1000)
    
    # 位置信息（用于空间触觉）
    position: Optional[Tuple[float, float, float]] = None
    direction: Optional[Tuple[float, float, float]] = None


@dataclass
class MaterialHapticProfile:
    """材料触觉配置"""
    material_name: str
    
    # 表面特性
    roughness: float = 0.0  # 0-1
    hardness: float = 0.5   # 0-1
    smoothness: float = 0.5  # 0-1
    
    # 触觉参数
    base_frequency: float = 50.0  # Hz
    frequency_variation: float = 20.0
    intensity_base: float = 0.3
    intensity_variation: float = 0.2
    
    # 纹理模式
    texture_pattern: str = "uniform"  # uniform, periodic, random, grid
    texture_scale: float = 1.0
    
    # 温度特性
    temperature_offset: float = 0.0  # -1 (cold) to 1 (hot)
    thermal_conductivity: float = 0.5
    
    # 声音关联（可选）
    associated_sound: Optional[str] = None


@dataclass
class ForceFeedbackConfig:
    """力反馈配置"""
    stiffness: float = 1000.0  # N/m
    damping: float = 10.0      # Ns/m
    max_force: float = 10.0    # N
    
    # 边界
    workspace_radius: float = 0.15  # m
    safety_limit_enabled: bool = True
    
    # 效果
    surface_hardness: float = 0.8
    friction_coefficient: float = 0.3
    texture_amplitude: float = 0.5


class AdvancedHapticSystem:
    """高级触觉反馈系统"""
    
    def __init__(self):
        self.connected_devices: Dict[HapticDeviceType, bool] = {}
        self.channels: Dict[str, HapticChannel] = {}
        self.active_effects: List[Dict[str, Any]] = []
        
        # 材料配置库
        self.material_profiles: Dict[str, MaterialHapticProfile] = {}
        self._init_default_materials()
        
        # 力反馈配置
        self.force_feedback = ForceFeedbackConfig()
        
        # 空间映射
        self.spatial_grid_resolution: float = 0.01  # 1cm
        self.spatial_haptic_field: Optional[np.ndarray] = None
        
        # 触觉历史
        self.haptic_history: deque = deque(maxlen=1000)
        
        # 渲染状态
        self.render_thread_active: bool = False
        self.effect_queue: asyncio.Queue = asyncio.Queue()
        
    def _init_default_materials(self) -> None:
        """初始化默认材料配置"""
        materials = {
            "polished_metal": MaterialHapticProfile(
                material_name="Polished Metal",
                roughness=0.05,
                hardness=0.95,
                smoothness=0.95,
                base_frequency=80,
                intensity_base=0.4,
                temperature_offset=-0.2
            ),
            "rough_metal": MaterialHapticProfile(
                material_name="Rough Metal",
                roughness=0.7,
                hardness=0.95,
                smoothness=0.2,
                base_frequency=120,
                frequency_variation=50,
                texture_pattern="random"
            ),
            "single_crystal": MaterialHapticProfile(
                material_name="Single Crystal",
                roughness=0.0,
                hardness=1.0,
                smoothness=1.0,
                base_frequency=200,
                texture_pattern="periodic",
                temperature_offset=-0.3
            ),
            "polycrystal": MaterialHapticProfile(
                material_name="Polycrystalline",
                roughness=0.3,
                hardness=0.85,
                base_frequency=100,
                texture_pattern="grid"
            ),
            "ceramic": MaterialHapticProfile(
                material_name="Ceramic",
                roughness=0.2,
                hardness=0.95,
                base_frequency=150,
                temperature_offset=-0.1
            ),
            "polymer": MaterialHapticProfile(
                material_name="Polymer",
                roughness=0.2,
                hardness=0.4,
                smoothness=0.6,
                base_frequency=40,
                texture_pattern="uniform"
            ),
            "rubber": MaterialHapticProfile(
                material_name="Rubber",
                roughness=0.1,
                hardness=0.2,
                smoothness=0.7,
                base_frequency=30,
                intensity_base=0.2
            ),
            "wood": MaterialHapticProfile(
                material_name="Wood",
                roughness=0.4,
                hardness=0.5,
                base_frequency=60,
                texture_pattern="periodic",
                texture_scale=2.0
            ),
            "glass": MaterialHapticProfile(
                material_name="Glass",
                roughness=0.0,
                hardness=0.9,
                smoothness=1.0,
                base_frequency=180,
                temperature_offset=-0.15
            ),
            "graphene": MaterialHapticProfile(
                material_name="Graphene",
                roughness=0.0,
                hardness=1.0,
                smoothness=1.0,
                base_frequency=300,
                texture_pattern="grid",
                texture_scale=0.5
            ),
            "nanotube": MaterialHapticProfile(
                material_name="Carbon Nanotube",
                roughness=0.1,
                hardness=1.0,
                base_frequency=250,
                texture_pattern="periodic",
                texture_scale=0.3
            ),
            "quantum_dot": MaterialHapticProfile(
                material_name="Quantum Dot",
                roughness=0.0,
                hardness=0.8,
                base_frequency=400,
                texture_pattern="uniform",
                intensity_variation=0.5
            )
        }
        
        self.material_profiles.update(materials)
    
    def connect_device(self, device_type: HapticDeviceType,
                      config: Optional[Dict[str, Any]] = None) -> bool:
        """连接触觉设备"""
        try:
            # 模拟设备连接
            self.connected_devices[device_type] = True
            
            # 初始化设备通道
            if device_type == HapticDeviceType.VR_CONTROLLER:
                self._init_vr_controller_channels()
            elif device_type == HapticDeviceType.HAPTIC_GLOVE:
                self._init_haptic_glove_channels()
            elif device_type == HapticDeviceType.FORCE_FEEDBACK_ARM:
                self._init_force_feedback_channels()
            
            logger.info(f"Connected {device_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect {device_type.value}: {e}")
            return False
    
    def _init_vr_controller_channels(self) -> None:
        """初始化VR控制器通道"""
        for hand in ['left', 'right']:
            for i in range(3):  # 多个振动点
                channel_id = f"vr_{hand}_vib_{i}"
                self.channels[channel_id] = HapticChannel(
                    channel_id=channel_id,
                    channel_type="vibration",
                    position=(0, 0, 0) if hand == 'left' else (0.5, 0, 0)
                )
    
    def _init_haptic_glove_channels(self) -> None:
        """初始化触觉手套通道"""
        fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
        segments = ['proximal', 'intermediate', 'distal']
        
        for hand in ['left', 'right']:
            for finger in fingers:
                for segment in segments:
                    channel_id = f"glove_{hand}_{finger}_{segment}"
                    self.channels[channel_id] = HapticChannel(
                        channel_id=channel_id,
                        channel_type="vibration",
                        frequency_range=(10, 300)
                    )
    
    def _init_force_feedback_channels(self) -> None:
        """初始化力反馈通道"""
        for axis in ['x', 'y', 'z']:
            channel_id = f"ffb_axis_{axis}"
            self.channels[channel_id] = HapticChannel(
                channel_id=channel_id,
                channel_type="force",
                min_intensity=-1.0,
                max_intensity=1.0
            )
    
    def trigger_material_sensation(self, material_name: str,
                                  contact_force: float = 0.5,
                                  sliding_velocity: float = 0.0,
                                  channels: Optional[List[str]] = None) -> None:
        """触发材料触感"""
        if material_name not in self.material_profiles:
            logger.warning(f"Unknown material: {material_name}")
            return
        
        profile = self.material_profiles[material_name]
        
        # 计算基础振动参数
        base_intensity = profile.intensity_base * contact_force
        base_freq = profile.base_frequency
        
        # 根据滑动速度调制
        if sliding_velocity > 0:
            freq_mod = profile.frequency_variation * sliding_velocity
            base_freq += freq_mod
        
        # 根据粗糙度添加随机变化
        if profile.roughness > 0 and sliding_velocity > 0:
            noise = np.random.normal(0, profile.roughness * 0.2)
            base_intensity = np.clip(base_intensity + noise, 0, 1)
        
        # 发送到通道
        target_channels = channels or [c for c in self.channels.keys() 
                                       if self.channels[c].channel_type == "vibration"]
        
        for channel_id in target_channels[:2]:  # 限制为2个主要通道
            self._send_haptic_command(
                channel_id=channel_id,
                intensity=base_intensity,
                frequency=base_freq,
                pattern=HapticPattern.TEXTURE,
                duration=0.1
            )
        
        # 记录历史
        self.haptic_history.append({
            'timestamp': time.time(),
            'material': material_name,
            'force': contact_force,
            'velocity': sliding_velocity
        })
    
    def simulate_crystal_lattice(self, lattice_type: str, lattice_constant: float,
                                probe_position: np.ndarray) -> None:
        """模拟晶格触感"""
        lattice_patterns = {
            'simple_cubic': {'symmetry': 4, 'frequency_base': 100, 'harmonics': [1, 2]},
            'bcc': {'symmetry': 8, 'frequency_base': 120, 'harmonics': [1, 1.5, 2]},
            'fcc': {'symmetry': 12, 'frequency_base': 140, 'harmonics': [1, 1.41, 2]},
            'diamond': {'symmetry': 4, 'frequency_base': 160, 'harmonics': [1, 1.73, 2]},
            'hexagonal': {'symmetry': 6, 'frequency_base': 110, 'harmonics': [1, 1.15, 2]}
        }
        
        pattern = lattice_patterns.get(lattice_type, lattice_patterns['simple_cubic'])
        
        # 计算在晶格中的相位位置
        phase = (probe_position / lattice_constant) % 1.0
        phase_sum = np.sum(phase)
        
        # 生成谐波序列
        for i, harmonic in enumerate(pattern['harmonics']):
            freq = pattern['frequency_base'] * harmonic
            intensity = 0.3 / (i + 1)  # 衰减
            
            # 根据相位调制强度
            intensity *= (1 + np.sin(phase_sum * 2 * np.pi * pattern['symmetry'])) / 2
            
            self._send_haptic_command(
                channel_id=list(self.channels.keys())[0],
                intensity=intensity,
                frequency=freq,
                pattern=HapticPattern.PULSE,
                duration=0.05
            )
    
    def simulate_bond_breaking(self, bond_energy: float, 
                              bond_type: str = "covalent") -> None:
        """模拟化学键断裂"""
        # 根据键类型确定参数
        bond_params = {
            "ionic": {"base_freq": 200, "decay": 0.3, "intensity": 0.8},
            "covalent": {"base_freq": 300, "decay": 0.5, "intensity": 0.9},
            "metallic": {"base_freq": 150, "decay": 0.4, "intensity": 0.7},
            "hydrogen": {"base_freq": 400, "decay": 0.2, "intensity": 0.5},
            "van_der_waals": {"base_freq": 50, "decay": 0.1, "intensity": 0.3}
        }
        
        params = bond_params.get(bond_type, bond_params["covalent"])
        
        # 生成断裂效果（快速频率下降）
        for i in range(10):
            freq = params["base_freq"] * (1 - i * params["decay"] / 10)
            intensity = params["intensity"] * (1 - i / 10)
            
            self._send_haptic_command(
                channel_id=list(self.channels.keys())[0],
                intensity=intensity,
                frequency=freq,
                pattern=HapticPattern.BURST,
                duration=0.02
            )
    
    def simulate_phase_transition(self, from_phase: str, to_phase: str,
                                 temperature: float) -> None:
        """模拟相变"""
        # 创建渐变效果
        steps = 20
        for i in range(steps):
            progress = i / steps
            
            # 频率从低频向高频或反之变化
            if from_phase in ["solid", "crystal"] and to_phase in ["liquid", "melt"]:
                freq = 200 - progress * 150  # 固到液：高频到低频
            else:
                freq = 50 + progress * 150   # 液到固：低频到高频
            
            intensity = 0.5 + 0.3 * np.sin(progress * np.pi)
            
            self._send_haptic_command(
                channel_id=list(self.channels.keys())[0],
                intensity=intensity,
                frequency=freq,
                pattern=HapticPattern.CONTINUOUS,
                duration=0.05
            )
    
    def apply_force_feedback(self, force_vector: np.ndarray,
                            position: np.ndarray) -> bool:
        """应用力反馈"""
        if HapticDeviceType.FORCE_FEEDBACK_ARM not in self.connected_devices:
            return False
        
        # 限制力的大小
        force_magnitude = np.linalg.norm(force_vector)
        if force_magnitude > self.force_feedback.max_force:
            force_vector = force_vector / force_magnitude * self.force_feedback.max_force
        
        # 发送到各轴
        for i, axis in enumerate(['x', 'y', 'z']):
            channel_id = f"ffb_axis_{axis}"
            if channel_id in self.channels:
                self.channels[channel_id].current_intensity = force_vector[i]
        
        return True
    
    def _send_haptic_command(self, channel_id: str, intensity: float,
                            frequency: float, pattern: HapticPattern,
                            duration: float) -> None:
        """发送触觉命令到设备（模拟）"""
        if channel_id not in self.channels:
            return
        
        channel = self.channels[channel_id]
        channel.current_intensity = intensity
        channel.current_frequency = frequency
        channel.is_active = True
        
        # 模拟异步执行
        logger.debug(f"Haptic: {channel_id} - {pattern.value} @ {frequency:.0f}Hz, "
                    f"intensity={intensity:.2f}, duration={duration:.2f}s")
    
    def create_spatial_field(self, field_data: np.ndarray,
                            bounds: Tuple[Tuple[float, float, float], 
                                        Tuple[float, float, float]]) -> None:
        """创建空间触觉场"""
        self.spatial_haptic_field = field_data
        logger.info(f"Created spatial haptic field: shape={field_data.shape}")
    
    def sample_spatial_field(self, position: np.ndarray) -> Dict[str, float]:
        """采样空间触觉场"""
        if self.spatial_haptic_field is None:
            return {}
        
        # 三线性插值采样
        # 简化版本
        return {
            'intensity': 0.5,
            'frequency': 100.0,
            'direction': (0.0, 0.0, 1.0)
        }
    
    def get_haptic_history(self, duration: float = 10.0) -> List[Dict[str, Any]]:
        """获取触觉历史"""
        current_time = time.time()
        return [
            h for h in self.haptic_history
            if current_time - h['timestamp'] <= duration
        ]
    
    def export_profile(self, material_name: str) -> Optional[Dict[str, Any]]:
        """导出材料配置文件"""
        profile = self.material_profiles.get(material_name)
        if not profile:
            return None
        
        return {
            'name': profile.material_name,
            'roughness': profile.roughness,
            'hardness': profile.hardness,
            'smoothness': profile.smoothness,
            'base_frequency': profile.base_frequency,
            'texture_pattern': profile.texture_pattern
        }


# 使用示例
def example_haptics():
    """触觉示例"""
    print("=" * 60)
    print("Advanced Haptic System Example")
    print("=" * 60)
    
    haptics = AdvancedHapticSystem()
    
    # 连接设备
    haptics.connect_device(HapticDeviceType.VR_CONTROLLER)
    print(f"✓ Connected VR controller")
    print(f"  Channels: {len(haptics.channels)}")
    
    # 展示材料配置
    print(f"\n✓ Available material profiles:")
    for name, profile in haptics.material_profiles.items():
        print(f"  - {profile.material_name}: "
              f"roughness={profile.roughness:.2f}, "
              f"hardness={profile.hardness:.2f}")
    
    # 模拟材料触感
    print(f"\n✓ Simulating material sensations:")
    for material in ['polished_metal', 'single_crystal', 'rubber', 'wood']:
        haptics.trigger_material_sensation(material, contact_force=0.6)
        print(f"  Triggered {material} sensation")
    
    # 模拟晶格
    print(f"\n✓ Simulating crystal lattices:")
    for lattice in ['simple_cubic', 'fcc', 'diamond']:
        haptics.simulate_crystal_lattice(
            lattice, 
            lattice_constant=5.43,
            probe_position=np.array([1.0, 0.5, 0.0])
        )
        print(f"  Simulated {lattice} lattice")
    
    # 模拟键断裂
    print(f"\n✓ Simulating bond breaking:")
    for bond in ['covalent', 'ionic', 'hydrogen']:
        haptics.simulate_bond_breaking(bond_energy=4.5, bond_type=bond)
        print(f"  Simulated {bond} bond breaking")
    
    # 模拟相变
    print(f"\n✓ Simulating phase transitions:")
    haptics.simulate_phase_transition("solid", "liquid", temperature=1000)
    print(f"  Simulated solid to liquid transition")
    
    # 力反馈
    print(f"\n✓ Force feedback:")
    haptics.apply_force_feedback(
        np.array([2.0, 0.0, 1.0]),
        np.array([0.1, 0.0, 0.0])
    )
    print(f"  Applied force vector (2.0, 0.0, 1.0) N")
    
    # 导出配置文件
    print(f"\n✓ Exporting profiles:")
    profile = haptics.export_profile("graphene")
    print(f"  Graphene profile: {profile}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    example_haptics()
