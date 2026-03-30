#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR Interface Module for Metaverse Materials Laboratory
VR接口模块 - 元宇宙材料实验室

Provides immersive VR visualization, gesture interaction, and collaborative virtual spaces
for materials science research and education.

Author: XR Expert Team
Version: 1.0.0
"""

import numpy as np
import json
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from enum import Enum, auto
import logging
from collections import defaultdict
import time
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VRRenderMode(Enum):
    """VR渲染模式"""
    WIREFRAME = auto()
    SURFACE = auto()
    VOLUMETRIC = auto()
    BALL_AND_STICK = auto()
    SPACE_FILLING = auto()
    CRYSTAL_LATTICE = auto()
    ELECTRON_DENSITY = auto()


class GestureType(Enum):
    """手势类型"""
    GRAB = "grab"
    PINCH = "pinch"
    POINT = "point"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    CIRCLE = "circle"
    PALM_OPEN = "palm_open"
    FIST = "fist"
    ROTATE = "rotate"
    SCALE = "scale"


@dataclass
class VRVector3:
    """3D向量"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'VRVector3':
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))
    
    def distance_to(self, other: 'VRVector3') -> float:
        return np.linalg.norm(self.to_array() - other.to_array())
    
    def __add__(self, other: 'VRVector3') -> 'VRVector3':
        return VRVector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __mul__(self, scalar: float) -> 'VRVector3':
        return VRVector3(self.x * scalar, self.y * scalar, self.z * scalar)


@dataclass
class VRQuaternion:
    """四元数用于VR旋转"""
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_rotation_matrix(self) -> np.ndarray:
        """转换为旋转矩阵"""
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
    
    @classmethod
    def from_euler(cls, pitch: float, yaw: float, roll: float) -> 'VRQuaternion':
        """从欧拉角创建四元数"""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        return cls(
            w=cr * cp * cy + sr * sp * sy,
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy
        )


@dataclass
class VRTransform:
    """VR变换组件"""
    position: VRVector3 = field(default_factory=VRVector3)
    rotation: VRQuaternion = field(default_factory=VRQuaternion)
    scale: VRVector3 = field(default_factory=lambda: VRVector3(1.0, 1.0, 1.0))
    
    def get_matrix(self) -> np.ndarray:
        """获取4x4变换矩阵"""
        rotation_matrix = self.rotation.to_rotation_matrix()
        matrix = np.eye(4)
        matrix[:3, :3] = rotation_matrix * self.scale.to_array()
        matrix[:3, 3] = self.position.to_array()
        return matrix


@dataclass
class AtomVRData:
    """原子VR数据"""
    element: str
    position: VRVector3
    radius: float
    color: Tuple[float, float, float, float]  # RGBA
    atomic_number: int
    bonds: List[int] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # VR特定属性
    glow_intensity: float = 0.0
    emission_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    is_highlighted: bool = False
    label_visible: bool = True


@dataclass
class VRCrystalStructure:
    """VR晶体结构"""
    name: str
    lattice_constants: Tuple[float, float, float, float, float, float]  # a, b, c, alpha, beta, gamma
    atoms: List[AtomVRData] = field(default_factory=list)
    unit_cell_vertices: List[VRVector3] = field(default_factory=list)
    miller_planes: List[Dict[str, Any]] = field(default_factory=list)
    
    # VR渲染属性
    render_mode: VRRenderMode = VRRenderMode.BALL_AND_STICK
    show_unit_cell: bool = True
    show_axes: bool = True
    transparency: float = 0.0


@dataclass
class GestureEvent:
    """手势事件"""
    gesture_type: GestureType
    hand: str  # 'left' or 'right'
    position: VRVector3
    direction: VRVector3
    confidence: float
    timestamp: float = field(default_factory=time.time)
    velocity: VRVector3 = field(default_factory=VRVector3)
    is_holding: bool = False
    held_object_id: Optional[str] = None


@dataclass
class VRUser:
    """VR用户"""
    user_id: str
    display_name: str
    head_position: VRVector3 = field(default_factory=VRVector3)
    head_rotation: VRQuaternion = field(default_factory=VRQuaternion)
    left_hand: Optional[VRTransform] = None
    right_hand: Optional[VRTransform] = None
    avatar_color: Tuple[float, float, float] = (0.3, 0.6, 1.0)
    is_speaking: bool = False
    voice_volume: float = 0.0
    last_active: float = field(default_factory=time.time)


class VRStructureVisualizer:
    """VR结构可视化器"""
    
    # 元素颜色映射 (CPK coloring)
    ELEMENT_COLORS = {
        'H': (1.0, 1.0, 1.0, 1.0),      # 白色
        'C': (0.3, 0.3, 0.3, 1.0),      # 深灰色
        'N': (0.0, 0.0, 1.0, 1.0),      # 蓝色
        'O': (1.0, 0.0, 0.0, 1.0),      # 红色
        'F': (0.0, 1.0, 0.0, 1.0),      # 绿色
        'P': (1.0, 0.65, 0.0, 1.0),     # 橙色
        'S': (1.0, 1.0, 0.0, 1.0),      # 黄色
        'Cl': (0.0, 1.0, 0.0, 1.0),     # 绿色
        'Fe': (1.0, 0.6, 0.0, 1.0),     # 橙棕色
        'Cu': (0.8, 0.5, 0.2, 1.0),     # 铜色
        'Si': (1.0, 0.6, 0.9, 1.0),     # 粉色
        'Au': (1.0, 0.84, 0.0, 1.0),    # 金色
        'Ag': (0.75, 0.75, 0.75, 1.0),  # 银色
        'Al': (0.7, 0.7, 0.8, 1.0),     # 浅灰色
        'Ti': (0.6, 0.6, 0.6, 1.0),     # 灰色
        'Na': (0.7, 0.4, 0.9, 1.0),     # 紫色
        'K': (0.5, 0.2, 0.7, 1.0),      # 深紫色
        'Ca': (0.4, 0.6, 0.4, 1.0),     # 深绿色
        'Mg': (0.5, 0.7, 0.5, 1.0),     # 浅绿色
    }
    
    # 元素范德华半径 (Å)
    VDW_RADII = {
        'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'F': 1.47,
        'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Fe': 2.0, 'Cu': 1.9,
        'Si': 2.1, 'Au': 1.66, 'Ag': 1.72, 'Al': 1.84, 'Ti': 1.87,
        'Na': 2.27, 'K': 2.75, 'Ca': 2.0, 'Mg': 1.73,
    }
    
    # 共价半径 (Å)
    COVALENT_RADII = {
        'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
        'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Fe': 1.32, 'Cu': 1.32,
        'Si': 1.11, 'Au': 1.21, 'Ag': 1.34, 'Al': 1.21, 'Ti': 1.36,
        'Na': 1.66, 'K': 1.96, 'Ca': 1.76, 'Mg': 1.41,
    }
    
    def __init__(self, render_scale: float = 1.0):
        self.render_scale = render_scale
        self.structures: Dict[str, VRCrystalStructure] = {}
        self.selected_structure_id: Optional[str] = None
        self.clip_planes: List[Tuple[VRVector3, VRVector3]] = []  # (point, normal)
        self.lod_level: int = 2  # Level of detail
        self.render_callbacks: List[Callable] = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def add_structure(self, structure_id: str, structure: VRCrystalStructure) -> None:
        """添加晶体结构"""
        self.structures[structure_id] = structure
        logger.info(f"Added VR structure: {structure_id} with {len(structure.atoms)} atoms")
        
    def remove_structure(self, structure_id: str) -> bool:
        """移除晶体结构"""
        if structure_id in self.structures:
            del self.structures[structure_id]
            return True
        return False
    
    def get_structure(self, structure_id: str) -> Optional[VRCrystalStructure]:
        """获取晶体结构"""
        return self.structures.get(structure_id)
    
    def create_from_ase(self, atoms, structure_id: str, name: str = "") -> VRCrystalStructure:
        """从ASE Atoms对象创建VR结构"""
        from ase import Atoms
        
        vr_atoms = []
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        
        for i, (symbol, pos) in enumerate(zip(symbols, positions)):
            color = self.ELEMENT_COLORS.get(symbol, (0.5, 0.5, 0.5, 1.0))
            radius = self.VDW_RADII.get(symbol, 1.5) * self.render_scale
            
            atom_data = AtomVRData(
                element=symbol,
                position=VRVector3(pos[0], pos[1], pos[2]),
                radius=radius,
                color=color,
                atomic_number=self._get_atomic_number(symbol),
                properties={'index': i}
            )
            vr_atoms.append(atom_data)
        
        # 计算化学键
        self._calculate_bonds(vr_atoms, atoms)
        
        # 获取晶格参数
        cell = atoms.get_cell()
        if cell is not None and len(cell) == 3:
            a, b, c = np.linalg.norm(cell[0]), np.linalg.norm(cell[1]), np.linalg.norm(cell[2])
            alpha = np.arccos(np.dot(cell[1], cell[2]) / (b * c)) * 180 / np.pi
            beta = np.arccos(np.dot(cell[0], cell[2]) / (a * c)) * 180 / np.pi
            gamma = np.arccos(np.dot(cell[0], cell[1]) / (a * b)) * 180 / np.pi
            lattice = (a, b, c, alpha, beta, gamma)
            
            # 计算晶胞顶点
            vertices = self._calculate_unit_cell_vertices(cell)
        else:
            lattice = (10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
            vertices = []
        
        structure = VRCrystalStructure(
            name=name or structure_id,
            lattice_constants=lattice,
            atoms=vr_atoms,
            unit_cell_vertices=vertices
        )
        
        self.add_structure(structure_id, structure)
        return structure
    
    def _get_atomic_number(self, symbol: str) -> int:
        """获取原子序数"""
        periodic_table = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
            'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
            'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
            'Br': 35, 'Kr': 36,
        }
        return periodic_table.get(symbol, 0)
    
    def _calculate_bonds(self, vr_atoms: List[AtomVRData], atoms) -> None:
        """计算原子间的化学键"""
        positions = np.array([a.position.to_array() for a in vr_atoms])
        
        for i, atom_i in enumerate(vr_atoms):
            radius_i = self.COVALENT_RADII.get(atom_i.element, 1.5)
            for j, atom_j in enumerate(vr_atoms[i+1:], start=i+1):
                radius_j = self.COVALENT_RADII.get(atom_j.element, 1.5)
                max_bond_length = (radius_i + radius_j) * 1.2  # 允许20%容差
                
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance < max_bond_length and distance > 0.5:
                    atom_i.bonds.append(j)
                    atom_j.bonds.append(i)
    
    def _calculate_unit_cell_vertices(self, cell: np.ndarray) -> List[VRVector3]:
        """计算晶胞顶点"""
        vertices = []
        for i in [0, 1]:
            for j in [0, 1]:
                for k in [0, 1]:
                    vertex = i * cell[0] + j * cell[1] + k * cell[2]
                    vertices.append(VRVector3.from_array(vertex))
        return vertices
    
    def set_render_mode(self, structure_id: str, mode: VRRenderMode) -> bool:
        """设置渲染模式"""
        if structure_id in self.structures:
            self.structures[structure_id].render_mode = mode
            return True
        return False
    
    def highlight_atoms(self, structure_id: str, atom_indices: List[int], 
                       color: Optional[Tuple[float, float, float]] = None) -> bool:
        """高亮指定原子"""
        structure = self.structures.get(structure_id)
        if not structure:
            return False
        
        highlight_color = color or (1.0, 1.0, 0.0)  # 默认黄色
        
        for idx in atom_indices:
            if 0 <= idx < len(structure.atoms):
                atom = structure.atoms[idx]
                atom.is_highlighted = True
                atom.emission_color = highlight_color
                atom.glow_intensity = 0.5
        
        return True
    
    def clear_highlight(self, structure_id: str) -> bool:
        """清除高亮"""
        structure = self.structures.get(structure_id)
        if not structure:
            return False
        
        for atom in structure.atoms:
            atom.is_highlighted = False
            atom.emission_color = (0.0, 0.0, 0.0)
            atom.glow_intensity = 0.0
        
        return True
    
    def slice_structure(self, structure_id: str, plane_point: VRVector3, 
                       plane_normal: VRVector3) -> Dict[str, Any]:
        """切片分析结构"""
        structure = self.structures.get(structure_id)
        if not structure:
            return {}
        
        normal = plane_normal.to_array()
        normal = normal / np.linalg.norm(normal)
        point = plane_point.to_array()
        
        atoms_above = []
        atoms_below = []
        atoms_on_plane = []
        
        for i, atom in enumerate(structure.atoms):
            atom_pos = atom.position.to_array()
            distance = np.dot(atom_pos - point, normal)
            
            tolerance = 0.1  # Å
            if abs(distance) < tolerance:
                atoms_on_plane.append(i)
            elif distance > 0:
                atoms_above.append(i)
            else:
                atoms_below.append(i)
        
        return {
            'atoms_above_plane': atoms_above,
            'atoms_below_plane': atoms_below,
            'atoms_on_plane': atoms_on_plane,
            'plane_point': plane_point,
            'plane_normal': plane_normal
        }
    
    def get_render_data(self, structure_id: str) -> Dict[str, Any]:
        """获取渲染数据（用于WebGL/Unity/Unreal）"""
        structure = self.structures.get(structure_id)
        if not structure:
            return {}
        
        atom_positions = []
        atom_colors = []
        atom_radii = []
        atom_elements = []
        bonds = []
        
        for atom in structure.atoms:
            atom_positions.extend([atom.position.x, atom.position.y, atom.position.z])
            atom_colors.extend(atom.color[:3])
            atom_radii.append(atom.radius)
            atom_elements.append(atom.element)
        
        # 构建键数据
        for i, atom in enumerate(structure.atoms):
            for j in atom.bonds:
                if i < j:  # 避免重复
                    bonds.extend([i, j])
        
        return {
            'positions': atom_positions,
            'colors': atom_colors,
            'radii': atom_radii,
            'elements': atom_elements,
            'bonds': bonds,
            'lattice_constants': structure.lattice_constants,
            'unit_cell_vertices': [(v.x, v.y, v.z) for v in structure.unit_cell_vertices],
            'render_mode': structure.render_mode.name,
            'num_atoms': len(structure.atoms)
        }
    
    def export_to_gltf(self, structure_id: str, filepath: str) -> bool:
        """导出为glTF格式（用于WebXR）"""
        # glTF导出实现（简化版本）
        render_data = self.get_render_data(structure_id)
        if not render_data:
            return False
        
        gltf_data = {
            "asset": {"version": "2.0", "generator": "DFTLammps VR"},
            "scene": 0,
            "scenes": [{"nodes": list(range(render_data['num_atoms']))}],
            "nodes": [],
            "meshes": [],
            "materials": [],
            "accessors": [],
            "bufferViews": [],
            "buffers": []
        }
        
        # 为每个原子创建一个节点
        for i in range(render_data['num_atoms']):
            pos = render_data['positions'][i*3:(i+1)*3]
            color = render_data['colors'][i*3:(i+1)*3]
            radius = render_data['radii'][i]
            
            gltf_data["nodes"].append({
                "mesh": 0,
                "translation": pos,
                "scale": [radius, radius, radius],
                "extras": {"element": render_data['elements'][i]}
            })
        
        with open(filepath, 'w') as f:
            json.dump(gltf_data, f, indent=2)
        
        logger.info(f"Exported VR structure to {filepath}")
        return True


class GestureRecognizer:
    """手势识别器"""
    
    def __init__(self):
        self.gesture_history: List[GestureEvent] = []
        self.history_limit = 30
        self.active_gestures: Dict[str, GestureType] = {}  # hand -> gesture
        self.callbacks: Dict[GestureType, List[Callable]] = defaultdict(list)
        self.confidence_threshold = 0.7
        
    def register_callback(self, gesture: GestureType, callback: Callable) -> None:
        """注册手势回调"""
        self.callbacks[gesture].append(callback)
        
    def unregister_callback(self, gesture: GestureType, callback: Callable) -> None:
        """注销手势回调"""
        if callback in self.callbacks[gesture]:
            self.callbacks[gesture].remove(callback)
    
    def process_hand_data(self, hand_data: Dict[str, Any]) -> Optional[GestureEvent]:
        """处理手部追踪数据"""
        # 模拟手部数据处理
        hand = hand_data.get('hand', 'right')
        finger_positions = hand_data.get('finger_positions', [])
        palm_position = hand_data.get('palm_position', [0, 0, 0])
        
        gesture_type = self._recognize_gesture(finger_positions, hand_data)
        confidence = self._calculate_confidence(finger_positions)
        
        if confidence < self.confidence_threshold:
            return None
        
        event = GestureEvent(
            gesture_type=gesture_type,
            hand=hand,
            position=VRVector3(*palm_position),
            direction=VRVector3(0, 0, -1),  # 默认朝向
            confidence=confidence,
            velocity=VRVector3(*hand_data.get('velocity', [0, 0, 0]))
        )
        
        self._update_history(event)
        self._trigger_callbacks(event)
        
        return event
    
    def _recognize_gesture(self, finger_positions: List[List[float]], 
                          hand_data: Dict) -> GestureType:
        """识别手势类型"""
        # 简化版手势识别逻辑
        fingers_extended = hand_data.get('fingers_extended', [False] * 5)
        pinch_distance = hand_data.get('pinch_distance', float('inf'))
        
        if pinch_distance < 0.02:  # 2cm
            return GestureType.PINCH
        
        extended_count = sum(fingers_extended)
        
        if extended_count == 0:
            return GestureType.FIST
        elif extended_count == 5:
            return GestureType.PALM_OPEN
        elif extended_count == 1 and fingers_extended[1]:  # 食指
            return GestureType.POINT
        elif extended_count == 2 and fingers_extended[0] and fingers_extended[1]:
            return GestureType.GRAB
        
        # 检测滑动手势
        velocity = hand_data.get('velocity', [0, 0, 0])
        if abs(velocity[0]) > abs(velocity[1]) and abs(velocity[0]) > 0.1:
            return GestureType.SWIPE_RIGHT if velocity[0] > 0 else GestureType.SWIPE_LEFT
        elif abs(velocity[1]) > abs(velocity[0]) and abs(velocity[1]) > 0.1:
            return GestureType.SWIPE_UP if velocity[1] > 0 else GestureType.SWIPE_DOWN
        
        return GestureType.PALM_OPEN
    
    def _calculate_confidence(self, finger_positions: List[List[float]]) -> float:
        """计算手势识别置信度"""
        if not finger_positions:
            return 0.0
        return min(1.0, len(finger_positions) / 5.0 + 0.5)
    
    def _update_history(self, event: GestureEvent) -> None:
        """更新手势历史"""
        self.gesture_history.append(event)
        if len(self.gesture_history) > self.history_limit:
            self.gesture_history.pop(0)
        
        self.active_gestures[event.hand] = event.gesture_type
    
    def _trigger_callbacks(self, event: GestureEvent) -> None:
        """触发回调"""
        for callback in self.callbacks[event.gesture_type]:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Gesture callback error: {e}")
    
    def get_gesture_velocity(self, hand: str, window: int = 5) -> VRVector3:
        """获取手势速度"""
        hand_events = [e for e in self.gesture_history if e.hand == hand][-window:]
        if len(hand_events) < 2:
            return VRVector3()
        
        total_velocity = VRVector3()
        for i in range(1, len(hand_events)):
            dt = hand_events[i].timestamp - hand_events[i-1].timestamp
            if dt > 0:
                dx = hand_events[i].position.x - hand_events[i-1].position.x
                dy = hand_events[i].position.y - hand_events[i-1].position.y
                dz = hand_events[i].position.z - hand_events[i-1].position.z
                total_velocity = total_velocity + VRVector3(dx/dt, dy/dt, dz/dt)
        
        n = len(hand_events) - 1
        return VRVector3(total_velocity.x/n, total_velocity.y/n, total_velocity.z/n)


class VRCollaborativeSpace:
    """VR协作空间"""
    
    def __init__(self, space_id: str, name: str = ""):
        self.space_id = space_id
        self.name = name or space_id
        self.users: Dict[str, VRUser] = {}
        self.objects: Dict[str, Dict[str, Any]] = {}  # 空间中的交互对象
        self.shared_structures: Dict[str, str] = {}  # structure_id -> owner_id
        self.annotations: List[Dict[str, Any]] = []
        self.voice_channels: Dict[str, List[str]] = defaultdict(list)  # channel -> user_ids
        self.permissions: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.created_at = time.time()
        self.last_activity = time.time()
        
    def add_user(self, user: VRUser) -> bool:
        """添加用户到空间"""
        if user.user_id in self.users:
            return False
        
        self.users[user.user_id] = user
        logger.info(f"User {user.display_name} joined VR space {self.space_id}")
        return True
    
    def remove_user(self, user_id: str) -> bool:
        """从空间移除用户"""
        if user_id in self.users:
            user = self.users.pop(user_id)
            logger.info(f"User {user.display_name} left VR space {self.space_id}")
            return True
        return False
    
    def update_user_position(self, user_id: str, head_pos: VRVector3, 
                            head_rot: VRQuaternion) -> bool:
        """更新用户位置"""
        if user_id not in self.users:
            return False
        
        user = self.users[user_id]
        user.head_position = head_pos
        user.head_rotation = head_rot
        user.last_active = time.time()
        self.last_activity = time.time()
        return True
    
    def update_user_hands(self, user_id: str, left_hand: Optional[VRTransform],
                         right_hand: Optional[VRTransform]) -> bool:
        """更新用户手部位置"""
        if user_id not in self.users:
            return False
        
        user = self.users[user_id]
        user.left_hand = left_hand
        user.right_hand = right_hand
        user.last_active = time.time()
        return True
    
    def add_shared_structure(self, structure_id: str, owner_id: str) -> bool:
        """添加共享结构"""
        if owner_id not in self.users:
            return False
        
        self.shared_structures[structure_id] = owner_id
        logger.info(f"Structure {structure_id} shared by {owner_id}")
        return True
    
    def add_annotation(self, user_id: str, position: VRVector3, 
                      text: str, annotation_type: str = "text") -> str:
        """添加空间标注"""
        annotation_id = f"annotation_{int(time.time() * 1000)}"
        
        annotation = {
            'id': annotation_id,
            'user_id': user_id,
            'position': {'x': position.x, 'y': position.y, 'z': position.z},
            'text': text,
            'type': annotation_type,
            'created_at': time.time(),
            'color': self.users.get(user_id, VRUser("", "")).avatar_color
        }
        
        self.annotations.append(annotation)
        return annotation_id
    
    def get_nearby_users(self, position: VRVector3, radius: float) -> List[VRUser]:
        """获取附近的用户"""
        nearby = []
        for user in self.users.values():
            distance = user.head_position.distance_to(position)
            if distance <= radius:
                nearby.append(user)
        return nearby
    
    def get_space_state(self) -> Dict[str, Any]:
        """获取空间状态（用于同步）"""
        return {
            'space_id': self.space_id,
            'name': self.name,
            'users': {
                uid: {
                    'display_name': u.display_name,
                    'position': {'x': u.head_position.x, 'y': u.head_position.y, 'z': u.head_position.z},
                    'is_speaking': u.is_speaking,
                    'avatar_color': u.avatar_color
                }
                for uid, u in self.users.items()
            },
            'shared_structures': self.shared_structures,
            'annotations': self.annotations,
            'user_count': len(self.users)
        }


class VRInterface:
    """主VR接口类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.visualizer = VRStructureVisualizer(
            render_scale=self.config.get('render_scale', 1.0)
        )
        self.gesture_recognizer = GestureRecognizer()
        self.collaborative_spaces: Dict[str, VRCollaborativeSpace] = {}
        self.active_space_id: Optional[str] = None
        self.current_user: Optional[VRUser] = None
        
        # VR硬件接口
        self.headset_connected = False
        self.controllers: Dict[str, Any] = {}
        self.tracking_rate = self.config.get('tracking_rate', 90)  # Hz
        
        # 事件循环
        self.running = False
        self.event_queue: asyncio.Queue = asyncio.Queue()
        
        # 注册默认手势回调
        self._setup_default_gestures()
        
    def _setup_default_gestures(self) -> None:
        """设置默认手势处理"""
        self.gesture_recognizer.register_callback(
            GestureType.GRAB, self._on_grab_gesture
        )
        self.gesture_recognizer.register_callback(
            GestureType.PINCH, self._on_pinch_gesture
        )
        self.gesture_recognizer.register_callback(
            GestureType.POINT, self._on_point_gesture
        )
    
    def _on_grab_gesture(self, event: GestureEvent) -> None:
        """抓取手势处理"""
        logger.debug(f"Grab gesture detected: {event.hand}")
        # 实现抓取逻辑
        
    def _on_pinch_gesture(self, event: GestureEvent) -> None:
        """捏合手势处理"""
        logger.debug(f"Pinch gesture detected: {event.hand}")
        # 实现缩放/选择逻辑
        
    def _on_point_gesture(self, event: GestureEvent) -> None:
        """指向手势处理"""
        logger.debug(f"Point gesture detected: {event.hand}")
        # 实现指向/选择逻辑
    
    async def initialize(self) -> bool:
        """初始化VR系统"""
        logger.info("Initializing VR Interface...")
        
        # 模拟VR硬件初始化
        self.headset_connected = True
        self.controllers = {'left': {}, 'right': {}}
        
        logger.info("VR Interface initialized successfully")
        return True
    
    async def shutdown(self) -> None:
        """关闭VR系统"""
        logger.info("Shutting down VR Interface...")
        self.running = False
        
        # 离开所有协作空间
        if self.active_space_id and self.current_user:
            await self.leave_collaborative_space()
        
        self.headset_connected = False
        logger.info("VR Interface shut down")
    
    def create_collaborative_space(self, name: str) -> str:
        """创建协作空间"""
        space_id = f"vr_space_{int(time.time() * 1000)}"
        space = VRCollaborativeSpace(space_id, name)
        self.collaborative_spaces[space_id] = space
        logger.info(f"Created VR collaborative space: {name} ({space_id})")
        return space_id
    
    async def join_collaborative_space(self, space_id: str, user: VRUser) -> bool:
        """加入协作空间"""
        if space_id not in self.collaborative_spaces:
            logger.error(f"VR space not found: {space_id}")
            return False
        
        space = self.collaborative_spaces[space_id]
        if space.add_user(user):
            self.active_space_id = space_id
            self.current_user = user
            
            # 开始同步循环
            asyncio.create_task(self._sync_loop(space_id))
            return True
        return False
    
    async def leave_collaborative_space(self) -> bool:
        """离开协作空间"""
        if not self.active_space_id or not self.current_user:
            return False
        
        space = self.collaborative_spaces.get(self.active_space_id)
        if space:
            space.remove_user(self.current_user.user_id)
        
        self.active_space_id = None
        return True
    
    async def _sync_loop(self, space_id: str) -> None:
        """同步循环"""
        while self.running and self.active_space_id == space_id:
            space = self.collaborative_spaces.get(space_id)
            if not space:
                break
            
            # 发送当前用户状态
            if self.current_user:
                space.update_user_position(
                    self.current_user.user_id,
                    self.current_user.head_position,
                    self.current_user.head_rotation
                )
            
            await asyncio.sleep(1.0 / self.tracking_rate)
    
    def update_head_pose(self, position: VRVector3, rotation: VRQuaternion) -> None:
        """更新头部姿态"""
        if self.current_user:
            self.current_user.head_position = position
            self.current_user.head_rotation = rotation
    
    def process_controller_input(self, controller_id: str, 
                                 hand_data: Dict[str, Any]) -> Optional[GestureEvent]:
        """处理控制器输入"""
        return self.gesture_recognizer.process_hand_data(hand_data)
    
    def get_render_commands(self) -> List[Dict[str, Any]]:
        """获取渲染命令"""
        commands = []
        
        # 添加所有结构到渲染队列
        for structure_id, structure in self.visualizer.structures.items():
            render_data = self.visualizer.get_render_data(structure_id)
            commands.append({
                'type': 'render_structure',
                'structure_id': structure_id,
                'data': render_data,
                'transform': VRTransform().get_matrix().tolist()
            })
        
        # 添加其他用户化身
        if self.active_space_id:
            space = self.collaborative_spaces[self.active_space_id]
            for user_id, user in space.users.items():
                if user_id != self.current_user.user_id:
                    commands.append({
                        'type': 'render_avatar',
                        'user_id': user_id,
                        'position': {
                            'x': user.head_position.x,
                            'y': user.head_position.y,
                            'z': user.head_position.z
                        },
                        'color': user.avatar_color,
                        'is_speaking': user.is_speaking
                    })
        
        return commands
    
    def teleport(self, position: VRVector3) -> None:
        """传送功能"""
        if self.current_user:
            self.current_user.head_position = position
            logger.info(f"Teleported to ({position.x}, {position.y}, {position.z})")
    
    def scale_world(self, scale_factor: float) -> None:
        """缩放世界"""
        self.visualizer.render_scale *= scale_factor
        logger.info(f"World scale changed to {self.visualizer.render_scale}")


# 使用示例和测试
def example_usage():
    """使用示例"""
    print("=" * 60)
    print("VR Interface Example - Metaverse Materials Laboratory")
    print("=" * 60)
    
    # 创建VR接口
    vr = VRInterface(config={'render_scale': 1.0, 'tracking_rate': 90})
    
    # 创建示例晶体结构 (Si单胞)
    try:
        from ase import Atoms
        from ase.lattice.cubic import Diamond
        
        si = Diamond(symbol='Si', latticeconstant=5.43)
        structure = vr.visualizer.create_from_ase(si, "si_diamond", "Silicon Diamond")
        
        print(f"\n✓ Created VR structure: {structure.name}")
        print(f"  - Atoms: {len(structure.atoms)}")
        print(f"  - Lattice: a={structure.lattice_constants[0]:.3f} Å")
        print(f"  - Render mode: {structure.render_mode.name}")
        
        # 高亮中心原子
        center_idx = len(structure.atoms) // 2
        vr.visualizer.highlight_atoms("si_diamond", [center_idx], (0.0, 1.0, 0.0))
        print(f"  - Highlighted atom {center_idx}")
        
        # 切片分析
        slice_result = vr.visualizer.slice_structure(
            "si_diamond",
            VRVector3(0, 0, 0),
            VRVector3(0, 0, 1)
        )
        print(f"\n✓ Sliced structure at Z=0 plane:")
        print(f"  - Atoms above: {len(slice_result['atoms_above_plane'])}")
        print(f"  - Atoms below: {len(slice_result['atoms_below_plane'])}")
        print(f"  - Atoms on plane: {len(slice_result['atoms_on_plane'])}")
        
    except ImportError:
        print("\n⚠ ASE not available, using mock data")
        # 创建简单的测试结构
        atoms = [
            AtomVRData('Si', VRVector3(0, 0, 0), 1.1, (0.5, 0.5, 0.5, 1.0), 14),
            AtomVRData('Si', VRVector3(2.7, 2.7, 0), 1.1, (0.5, 0.5, 0.5, 1.0), 14),
            AtomVRData('Si', VRVector3(2.7, 0, 2.7), 1.1, (0.5, 0.5, 0.5, 1.0), 14),
            AtomVRData('Si', VRVector3(0, 2.7, 2.7), 1.1, (0.5, 0.5, 0.5, 1.0), 14),
        ]
        structure = VRCrystalStructure(
            name="Mock Silicon",
            lattice_constants=(5.43, 5.43, 5.43, 90, 90, 90),
            atoms=atoms
        )
        vr.visualizer.add_structure("mock_si", structure)
        print(f"  - Mock structure with {len(atoms)} atoms created")
    
    # 创建协作空间
    space_id = vr.create_collaborative_space("Materials Research Lab")
    print(f"\n✓ Created collaborative VR space: {space_id}")
    
    # 模拟用户加入
    user = VRUser(
        user_id="user_001",
        display_name="Researcher_A",
        avatar_color=(0.2, 0.7, 0.9)
    )
    
    # 注册手势回调示例
    def on_grab(event: GestureEvent):
        print(f"  [Gesture] {event.hand} hand grabbed at ({event.position.x:.2f}, "
              f"{event.position.y:.2f}, {event.position.z:.2f})")
    
    vr.gesture_recognizer.register_callback(GestureType.GRAB, on_grab)
    
    # 模拟手势事件
    mock_hand_data = {
        'hand': 'right',
        'finger_positions': [[0, 0, 0]] * 5,
        'palm_position': [1.0, 1.5, -0.5],
        'fingers_extended': [True, True, False, False, False],
        'pinch_distance': 0.1,
        'velocity': [0.05, 0, 0]
    }
    
    event = vr.gesture_recognizer.process_hand_data(mock_hand_data)
    if event:
        print(f"\n✓ Gesture recognized: {event.gesture_type.value}")
    
    # 获取渲染数据
    render_data = vr.visualizer.get_render_data("mock_si" if "mock_si" in vr.visualizer.structures else "si_diamond")
    print(f"\n✓ Render data prepared:")
    if render_data:
        print(f"  - Positions: {len(render_data.get('positions', []))//3} atoms")
        print(f"  - Bonds: {len(render_data.get('bonds', []))//2} bonds")
    
    print("\n" + "=" * 60)
    print("VR Interface Example Complete")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()
