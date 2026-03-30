"""
三维结构演化渲染 (3D Structure Evolution Renderer)

实现材料系统的三维可视化渲染，支持原子结构、场分布和演化过程。
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, Generic, List, Optional, Protocol, Set,
    Tuple, TypeVar, Union, Iterator
)
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


# 可选依赖
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


try:
    from ..digital_twin.twin_core import StateVector
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from digital_twin.twin_core import StateVector


@dataclass
class Atom:
    """原子表示"""
    element: str
    position: NDArray[np.float64]
    velocity: Optional[NDArray[np.float64]] = None
    force: Optional[NDArray[np.float64]] = None
    charge: float = 0.0
    radius: float = 1.0
    color: str = "gray"
    
    def __post_init__(self):
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.float64)


@dataclass
class Bond:
    """化学键表示"""
    atom1_idx: int
    atom2_idx: int
    bond_type: str = "single"  # single, double, triple
    length: float = 0.0


@dataclass
class Cell:
    """晶胞表示"""
    a: NDArray[np.float64]
    b: NDArray[np.float64]
    c: NDArray[np.float64]
    origin: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    
    @property
    def volume(self) -> float:
        """计算晶胞体积"""
        return float(np.abs(np.dot(self.a, np.cross(self.b, self.c))))
    
    @property
    def lengths(self) -> Tuple[float, float, float]:
        """晶胞边长"""
        return (float(np.linalg.norm(self.a)),
                float(np.linalg.norm(self.b)),
                float(np.linalg.norm(self.c)))
    
    @property
    def angles(self) -> Tuple[float, float, float]:
        """晶胞角度"""
        a_mag, b_mag, c_mag = self.lengths
        
        alpha = np.arccos(np.dot(self.b, self.c) / (b_mag * c_mag)) * 180 / np.pi
        beta = np.arccos(np.dot(self.a, self.c) / (a_mag * c_mag)) * 180 / np.pi
        gamma = np.arccos(np.dot(self.a, self.b) / (a_mag * b_mag)) * 180 / np.pi
        
        return (float(alpha), float(beta), float(gamma))


@dataclass
class FieldGrid:
    """场数据网格"""
    data: NDArray[np.float64]  # 3D数组
    origin: NDArray[np.float64]
    spacing: Tuple[float, float, float]
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape


class ColorMap:
    """颜色映射"""
    
    # 元素颜色表 (CPK配色)
    ELEMENT_COLORS = {
        'H': '#FFFFFF',  # 白色
        'C': '#909090',  # 灰色
        'N': '#3050F8',  # 蓝色
        'O': '#FF0D0D',  # 红色
        'F': '#90E050',  # 绿色
        'P': '#FF8000',  # 橙色
        'S': '#FFFF30',  # 黄色
        'Cl': '#1FF01F', # 绿色
        'Fe': '#E06633', # 铁锈色
        'Cu': '#C78033', # 铜色
        'Zn': '#7D80B0', # 锌色
        'Au': '#FFD123', # 金色
        'Si': '#F0C8A0', # 硅色
        'Al': '#A6A6A6', # 铝色
        'Li': '#CC80FF', # 紫色
        'Na': '#AB5CF2', # 钠色
        'K': '#8F40D4',  # 钾色
        'Ca': '#3DFF00', # 钙色
        'Mg': '#8AFF00', # 镁色
    }
    
    # 元素范德华半径 (Å)
    ELEMENT_RADII = {
        'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52,
        'F': 1.47, 'P': 1.80, 'S': 1.80, 'Cl': 1.75,
        'Fe': 2.0, 'Cu': 1.9, 'Zn': 1.9, 'Au': 1.9,
        'Si': 2.1, 'Al': 2.0, 'Li': 1.8, 'Na': 2.3,
        'K': 2.8, 'Ca': 2.4, 'Mg': 1.7,
    }
    
    @classmethod
    def get_element_color(cls, element: str) -> str:
        """获取元素颜色"""
        return cls.ELEMENT_COLORS.get(element, '#808080')  # 默认灰色
    
    @classmethod
    def get_element_radius(cls, element: str) -> float:
        """获取元素半径"""
        return cls.ELEMENT_RADII.get(element, 1.5)  # 默认1.5Å
    
    @classmethod
    def value_to_color(cls, value: float, vmin: float, vmax: float, 
                       cmap: str = 'viridis') -> Tuple[int, int, int]:
        """将数值映射到颜色"""
        # 归一化
        norm_val = (value - vmin) / (vmax - vmin + 1e-10)
        norm_val = np.clip(norm_val, 0, 1)
        
        # 简化的颜色映射
        if cmap == 'viridis':
            r = int(68 + (59 - 68) * norm_val + (253 - 59) * norm_val**2)
            g = int(1 + (183 - 1) * norm_val + (231 - 183) * norm_val**2)
            b = int(84 + (140 - 84) * norm_val + (37 - 140) * norm_val**2)
        elif cmap == 'jet':
            r = int(255 * (1 if norm_val > 0.5 else 2 * norm_val))
            g = int(255 * (2 * norm_val if norm_val < 0.5 else 2 * (1 - norm_val)))
            b = int(255 * (1 - 2 * norm_val if norm_val > 0.5 else 1))
        elif cmap == 'hot':
            r = int(255 * min(1, norm_val * 3))
            g = int(255 * max(0, min(1, (norm_val - 0.33) * 3)))
            b = int(255 * max(0, min(1, (norm_val - 0.66) * 3)))
        else:
            gray = int(255 * norm_val)
            r = g = b = gray
        
        return (r, g, b)


class Structure3D:
    """
    3D结构表示
    
    管理原子、化学键和晶胞信息
    """
    
    def __init__(self):
        self.atoms: List[Atom] = []
        self.bonds: List[Bond] = []
        self.cell: Optional[Cell] = None
        self.fields: Dict[str, FieldGrid] = {}
        self.metadata: Dict[str, Any] = {}
        
    def add_atom(self, atom: Atom) -> int:
        """添加原子，返回索引"""
        idx = len(self.atoms)
        self.atoms.append(atom)
        return idx
    
    def add_bond(self, atom1_idx: int, atom2_idx: int, 
                bond_type: str = "single") -> None:
        """添加化学键"""
        if 0 <= atom1_idx < len(self.atoms) and 0 <= atom2_idx < len(self.atoms):
            atom1 = self.atoms[atom1_idx]
            atom2 = self.atoms[atom2_idx]
            length = np.linalg.norm(atom1.position - atom2.position)
            
            self.bonds.append(Bond(atom1_idx, atom2_idx, bond_type, length))
    
    def set_cell(self, cell: Cell) -> None:
        """设置晶胞"""
        self.cell = cell
    
    def add_field(self, name: str, field: FieldGrid) -> None:
        """添加场数据"""
        self.fields[name] = field
    
    def auto_bonds(self, max_bond_length: float = 2.0) -> None:
        """自动检测化学键"""
        self.bonds.clear()
        
        for i in range(len(self.atoms)):
            for j in range(i + 1, len(self.atoms)):
                dist = np.linalg.norm(self.atoms[i].position - self.atoms[j].position)
                if dist < max_bond_length:
                    self.add_bond(i, j)
    
    def center_of_mass(self) -> NDArray[np.float64]:
        """计算质心"""
        if not self.atoms:
            return np.zeros(3)
        
        positions = np.array([a.position for a in self.atoms])
        return np.mean(positions, axis=0)
    
    def bounding_box(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """计算边界框"""
        if not self.atoms:
            return np.zeros(3), np.ones(3)
        
        positions = np.array([a.position for a in self.atoms])
        return np.min(positions, axis=0), np.max(positions, axis=0)
    
    def from_state_vector(self, state: StateVector, 
                         n_atoms: int,
                         elements: Optional[List[str]] = None) -> None:
        """从StateVector解析结构"""
        data = state.data
        
        # 假设state包含 [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, ...]
        for i in range(min(n_atoms, len(data) // 6)):
            pos = data[i*6:i*6+3]
            vel = data[i*6+3:i*6+6]
            
            element = elements[i] if elements and i < len(elements) else 'X'
            
            atom = Atom(
                element=element,
                position=pos,
                velocity=vel,
                color=ColorMap.get_element_color(element),
                radius=ColorMap.get_element_radius(element)
            )
            self.add_atom(atom)


class Renderer3D(ABC):
    """3D渲染器基类"""
    
    @abstractmethod
    def render(self, structure: Structure3D, **kwargs) -> Any:
        """渲染结构"""
        pass


class MatplotlibRenderer3D(Renderer3D):
    """
    基于Matplotlib的3D渲染器
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8), dpi: int = 100):
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib not available")
        
        self.figsize = figsize
        self.dpi = dpi
        
    def render(self, structure: Structure3D, 
               show_cell: bool = True,
               show_bonds: bool = True,
               show_forces: bool = False,
               view: Tuple[float, float] = (30, 45),  # (elev, azim)
               **kwargs) -> plt.Figure:
        """渲染3D结构"""
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制晶胞
        if show_cell and structure.cell:
            self._draw_cell(ax, structure.cell)
        
        # 绘制化学键
        if show_bonds:
            self._draw_bonds(ax, structure)
        
        # 绘制原子
        self._draw_atoms(ax, structure)
        
        # 绘制力向量
        if show_forces:
            self._draw_forces(ax, structure)
        
        # 设置视角
        ax.view_init(elev=view[0], azim=view[1])
        
        # 设置等比例
        self._set_equal_aspect(ax, structure)
        
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(f'Structure: {len(structure.atoms)} atoms')
        
        return fig
    
    def _draw_atoms(self, ax: Axes3D, structure: Structure3D) -> None:
        """绘制原子"""
        if not structure.atoms:
            return
        
        positions = np.array([a.position for a in structure.atoms])
        colors = [a.color for a in structure.atoms]
        sizes = [a.radius * 100 for a in structure.atoms]
        
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  c=colors, s=sizes, alpha=0.8, edgecolors='black', linewidths=0.5)
        
        # 添加原子标签
        for i, atom in enumerate(structure.atoms):
            ax.text(atom.position[0], atom.position[1], atom.position[2],
                   f'  {atom.element}{i+1}', fontsize=8)
    
    def _draw_bonds(self, ax: Axes3D, structure: Structure3D) -> None:
        """绘制化学键"""
        for bond in structure.bonds:
            atom1 = structure.atoms[bond.atom1_idx]
            atom2 = structure.bonds[bond.atom2_idx] if bond.atom2_idx < len(structure.atoms) else None
            if atom2 is not None:
                atom2 = structure.atoms[bond.atom2_idx]
            else:
                continue
            
            x = [atom1.position[0], atom2.position[0]]
            y = [atom1.position[1], atom2.position[1]]
            z = [atom1.position[2], atom2.position[2]]
            
            ax.plot(x, y, z, 'gray', linewidth=2, alpha=0.6)
    
    def _draw_cell(self, ax: Axes3D, cell: Cell) -> None:
        """绘制晶胞"""
        o = cell.origin
        a = cell.a
        b = cell.b
        c = cell.c
        
        # 晶胞顶点
        vertices = [
            o, o + a, o + a + b, o + b,
            o + c, o + a + c, o + a + b + c, o + b + c
        ]
        
        # 晶胞边
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
            [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
            [0, 4], [1, 5], [2, 6], [3, 7]   # 侧棱
        ]
        
        for edge in edges:
            x = [vertices[edge[0]][0], vertices[edge[1]][0]]
            y = [vertices[edge[0]][1], vertices[edge[1]][1]]
            z = [vertices[edge[0]][2], vertices[edge[1]][2]]
            ax.plot(x, y, z, 'b-', linewidth=1, alpha=0.5)
    
    def _draw_forces(self, ax: Axes3D, structure: Structure3D) -> None:
        """绘制力向量"""
        for atom in structure.atoms:
            if atom.force is not None:
                scale = 0.5  # 力向量缩放因子
                ax.quiver(atom.position[0], atom.position[1], atom.position[2],
                         atom.force[0] * scale, atom.force[1] * scale, atom.force[2] * scale,
                         color='red', arrow_length_ratio=0.3)
    
    def _set_equal_aspect(self, ax: Axes3D, structure: Structure3D) -> None:
        """设置等比例坐标轴"""
        min_pos, max_pos = structure.bounding_box()
        center = (min_pos + max_pos) / 2
        max_range = np.max(max_pos - min_pos) / 2
        
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
    
    def render_trajectory(self, structures: List[Structure3D], 
                         n_frames: int = 10,
                         **kwargs) -> List[plt.Figure]:
        """渲染轨迹动画帧"""
        figures = []
        
        step = max(1, len(structures) // n_frames)
        selected = structures[::step][:n_frames]
        
        for i, struct in enumerate(selected):
            fig = self.render(struct, **kwargs)
            fig.suptitle(f'Frame {i+1}/{len(selected)}')
            figures.append(fig)
        
        return figures
    
    def save(self, fig: plt.Figure, filepath: str) -> None:
        """保存渲染结果"""
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"3D render saved to {filepath}")


class VolumeRenderer:
    """
    体积渲染器
    
    渲染3D场数据 (电子密度、温度场等)
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib not available")
        
        self.figsize = figsize
    
    def render_slice(self, field: FieldGrid, 
                    axis: str = 'z',
                    slice_index: Optional[int] = None,
                    cmap: str = 'viridis') -> plt.Figure:
        """渲染场数据的切片"""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        data = field.data
        nx, ny, nz = data.shape
        
        # 默认取中间切片
        if slice_index is None:
            slice_index = nz // 2
        
        # XY平面 (Z切片)
        ax = axes[0, 0]
        slice_xy = data[:, :, slice_index]
        im = ax.imshow(slice_xy.T, origin='lower', cmap=cmap, 
                      extent=[0, nx*field.spacing[0], 0, ny*field.spacing[1]])
        ax.set_title(f'XY Plane (z={slice_index})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax)
        
        # XZ平面 (Y切片)
        ax = axes[0, 1]
        slice_xz = data[:, ny//2, :]
        im = ax.imshow(slice_xz.T, origin='lower', cmap=cmap,
                      extent=[0, nx*field.spacing[0], 0, nz*field.spacing[2]])
        ax.set_title('XZ Plane')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        plt.colorbar(im, ax=ax)
        
        # YZ平面 (X切片)
        ax = axes[1, 0]
        slice_yz = data[nx//2, :, :]
        im = ax.imshow(slice_yz.T, origin='lower', cmap=cmap,
                      extent=[0, ny*field.spacing[1], 0, nz*field.spacing[2]])
        ax.set_title('YZ Plane')
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        plt.colorbar(im, ax=ax)
        
        # 统计信息
        ax = axes[1, 1]
        ax.hist(data.flatten(), bins=50, alpha=0.7, color='blue')
        ax.set_title('Value Distribution')
        ax.set_xlabel('Field Value')
        ax.set_ylabel('Frequency')
        
        plt.tight_layout()
        return fig
    
    def render_isosurface(self, field: FieldGrid, 
                         isovalues: List[float],
                         colors: Optional[List[str]] = None) -> plt.Figure:
        """渲染等值面"""
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        data = field.data
        nx, ny, nz = data.shape
        
        # 创建坐标网格
        x = np.linspace(0, nx*field.spacing[0], nx)
        y = np.linspace(0, ny*field.spacing[1], ny)
        z = np.linspace(0, nz*field.spacing[2], nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        if colors is None:
            colors = plt.cm.viridis(np.linspace(0, 1, len(isovalues)))
        
        # 绘制等值面 (使用contour3D的近似)
        for isoval, color in zip(isovalues, colors):
            # 找到近似等值面点
            mask = np.abs(data - isoval) < 0.05 * (data.max() - data.min())
            if np.any(mask):
                points = np.argwhere(mask)
                ax.scatter(X[mask], Y[mask], Z[mask], 
                          c=[color], s=1, alpha=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Isosurfaces')
        
        return fig


class EvolutionAnimator:
    """
    演化动画生成器
    
    从时间序列的Structure3D生成动画
    """
    
    def __init__(self, renderer: Renderer3D):
        self.renderer = renderer
        self.frames: List[Any] = []
    
    def add_frame(self, structure: Structure3D) -> None:
        """添加帧"""
        fig = self.renderer.render(structure)
        self.frames.append(fig)
    
    def save_animation(self, output_dir: str, prefix: str = "frame") -> List[str]:
        """保存动画帧"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        for i, fig in enumerate(self.frames):
            filepath = output_path / f"{prefix}_{i:04d}.png"
            
            if isinstance(self.renderer, MatplotlibRenderer3D):
                self.renderer.save(fig, str(filepath))
            
            saved_files.append(str(filepath))
        
        return saved_files
    
    def create_gif(self, output_path: str, duration: float = 0.5) -> None:
        """创建GIF动画"""
        if not PIL_AVAILABLE:
            print("PIL not available, cannot create GIF")
            return
        
        if not self.frames:
            print("No frames to animate")
            return
        
        # 保存临时帧
        temp_dir = Path("/tmp/anim_frames")
        temp_dir.mkdir(exist_ok=True)
        
        frame_files = self.save_animation(str(temp_dir))
        
        # 读取并创建GIF
        images = [Image.open(f) for f in frame_files]
        
        if images:
            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                duration=int(duration * 1000),
                loop=0
            )
            print(f"Animation saved to {output_path}")
        
        # 清理临时文件
        for f in frame_files:
            Path(f).unlink()


def demo():
    """演示3D渲染功能"""
    print("=" * 60)
    print("三维结构演化渲染演示")
    print("=" * 60)
    
    if not MATPLOTLIB_AVAILABLE:
        print("Error: Matplotlib not available")
        return
    
    # 创建测试结构 (水分子)
    print("\n1. 创建分子结构")
    structure = Structure3D()
    
    # 水分子坐标 (Å)
    structure.add_atom(Atom('O', [0.0, 0.0, 0.0], color=ColorMap.get_element_color('O'), radius=1.5))
    structure.add_atom(Atom('H', [0.757, 0.586, 0.0], color=ColorMap.get_element_color('H'), radius=1.0))
    structure.add_atom(Atom('H', [-0.757, 0.586, 0.0], color=ColorMap.get_element_color('H'), radius=1.0))
    
    structure.add_bond(0, 1)
    structure.add_bond(0, 2)
    
    print(f"   原子数: {len(structure.atoms)}")
    print(f"   化学键数: {len(structure.bonds)}")
    
    # 渲染
    print("\n2. 渲染分子结构")
    renderer = MatplotlibRenderer3D(figsize=(10, 8), dpi=100)
    fig = renderer.render(structure, show_bonds=True)
    renderer.save(fig, '/tmp/molecule_demo.png')
    
    # 创建晶胞
    print("\n3. 创建晶体结构")
    crystal = Structure3D()
    
    # 简单立方晶格
    a = 3.0  # 晶格常数
    for i in range(2):
        for j in range(2):
            for k in range(2):
                pos = [i*a, j*a, k*a]
                crystal.add_atom(Atom('Cu', pos, color=ColorMap.get_element_color('Cu'), radius=1.5))
    
    cell = Cell(
        a=np.array([2*a, 0, 0]),
        b=np.array([0, 2*a, 0]),
        c=np.array([0, 0, 2*a])
    )
    crystal.set_cell(cell)
    crystal.auto_bonds(max_bond_length=3.5)
    
    print(f"   原子数: {len(crystal.atoms)}")
    print(f"   晶胞体积: {cell.volume:.2f} Å³")
    print(f"   晶胞边长: {cell.lengths}")
    
    fig2 = renderer.render(crystal, show_cell=True)
    renderer.save(fig2, '/tmp/crystal_demo.png')
    
    # 体积渲染
    print("\n4. 体积渲染 (场数据)")
    volume_renderer = VolumeRenderer(figsize=(12, 10))
    
    # 创建测试场数据 (高斯分布)
    nx, ny, nz = 50, 50, 50
    x = np.linspace(-3, 3, nx)
    y = np.linspace(-3, 3, ny)
    z = np.linspace(-3, 3, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    field_data = np.exp(-(X**2 + Y**2 + Z**2)) + 0.5 * np.exp(-((X-1)**2 + (Y-1)**2 + (Z-1)**2))
    
    field = FieldGrid(
        data=field_data,
        origin=np.zeros(3),
        spacing=(0.1, 0.1, 0.1)
    )
    
    fig3 = volume_renderer.render_slice(field, cmap='hot')
    plt.savefig('/tmp/field_slice_demo.png', dpi=100, bbox_inches='tight')
    print("   场切片渲染已保存")
    
    # 轨迹动画
    print("\n5. 生成轨迹动画帧")
    trajectory = []
    n_frames = 5
    
    for i in range(n_frames):
        frame = Structure3D()
        angle = 2 * np.pi * i / n_frames
        
        # 旋转的水分子
        frame.add_atom(Atom('O', [0.0, 0.0, 0.0]))
        frame.add_atom(Atom('H', [0.757*np.cos(angle), 0.586, 0.757*np.sin(angle)]))
        frame.add_atom(Atom('H', [-0.757*np.cos(angle), 0.586, -0.757*np.sin(angle)]))
        
        trajectory.append(frame)
    
    anim = EvolutionAnimator(renderer)
    for frame in trajectory:
        anim.add_frame(frame)
    
    frame_files = anim.save_animation('/tmp/anim_frames', prefix='water')
    print(f"   生成了 {len(frame_files)} 帧动画")
    
    if PIL_AVAILABLE:
        anim.create_gif('/tmp/water_rotation.gif', duration=0.5)
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
    
    return renderer


if __name__ == "__main__":
    demo()
