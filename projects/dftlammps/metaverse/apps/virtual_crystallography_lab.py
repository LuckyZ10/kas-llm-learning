#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Virtual Crystallography Laboratory Application
虚拟晶体学实验室应用

An immersive VR/AR application for crystal structure analysis and visualization.

Author: XR Expert Team
Version: 1.0.0
"""

import numpy as np
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from dftlammps.metaverse.vr_interface import (
    VRInterface, VRStructureVisualizer, VRVector3, VRQuaternion,
    VRRenderMode, VRCrystalStructure, VRUser
)
from dftlammps.metaverse.ar_overlay import AROverlayManager, ARVector3 as ARV3
from dftlammps.immersive_viz.volume_renderer import (
    VolumeRenderer, VolumeRenderMode as VRM, ColorMap
)
from dftlammps.immersive_viz.haptics import AdvancedHapticSystem, HapticDeviceType

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VirtualCrystallographyLab:
    """虚拟晶体学实验室"""
    
    def __init__(self):
        self.vr = VRInterface(config={'render_scale': 1.0, 'tracking_rate': 90})
        self.ar = AROverlayManager(config={'width': 1920, 'height': 1080})
        self.volume_renderer = VolumeRenderer()
        self.haptics = AdvancedHapticSystem()
        
        # 实验状态
        self.current_sample: str = ""
        self.active_analysis: str = ""
        self.measurements: list = []
        
        # 晶体数据库
        self.crystal_database: dict = {}
        self._init_crystal_database()
        
    def _init_crystal_database(self) -> None:
        """初始化晶体数据库"""
        self.crystal_database = {
            "si_diamond": {
                "name": "Silicon (Diamond)",
                "formula": "Si",
                "space_group": "Fd-3m",
                "lattice_constant": 5.43,
                "color": (0.5, 0.5, 0.5),
                "description": "Silicon with diamond cubic structure"
            },
            "gaas_zincblende": {
                "name": "Gallium Arsenide",
                "formula": "GaAs",
                "space_group": "F-43m",
                "lattice_constant": 5.65,
                "color": (0.8, 0.7, 0.3),
                "description": "III-V Semiconductor"
            },
            "nacl_rocksalt": {
                "name": "Sodium Chloride",
                "formula": "NaCl",
                "space_group": "Fm-3m",
                "lattice_constant": 5.64,
                "color": (0.9, 0.9, 0.95),
                "description": "Ionic crystal with rocksalt structure"
            },
            "cao_rocksalt": {
                "name": "Calcium Oxide",
                "formula": "CaO",
                "space_group": "Fm-3m",
                "lattice_constant": 4.81,
                "color": (0.95, 0.9, 0.8),
                "description": "Refractory ceramic material"
            },
            "tio2_rutile": {
                "name": "Titanium Dioxide (Rutile)",
                "formula": "TiO2",
                "space_group": "P42/mnm",
                "lattice_constants": (4.59, 4.59, 2.96),
                "color": (0.9, 0.9, 0.9),
                "description": "Photocatalytic material"
            },
            "zno_wurtzite": {
                "name": "Zinc Oxide",
                "formula": "ZnO",
                "space_group": "P63mc",
                "lattice_constants": (3.25, 3.25, 5.21),
                "color": (0.95, 0.95, 1.0),
                "description": "Piezoelectric semiconductor"
            },
            "al2o3_corundum": {
                "name": "Aluminum Oxide (Corundum)",
                "formula": "Al2O3",
                "space_group": "R-3c",
                "lattice_constants": (4.76, 4.76, 12.99),
                "color": (0.9, 0.85, 0.9),
                "description": "Hard ceramic material"
            }
        }
    
    async def initialize(self) -> bool:
        """初始化实验室"""
        logger.info("Initializing Virtual Crystallography Laboratory...")
        
        # 初始化VR
        await self.vr.initialize()
        
        # 初始化AR
        self.ar.initialize()
        
        # 连接触觉设备
        self.haptics.connect_device(HapticDeviceType.VR_CONTROLLER)
        
        logger.info("Virtual Crystallography Laboratory initialized")
        return True
    
    def load_crystal_structure(self, structure_id: str, 
                              custom_atoms=None) -> bool:
        """加载晶体结构"""
        try:
            from ase.lattice.cubic import Diamond, SimpleCubic, BodyCenteredCubic, FaceCenteredCubic
            from ase.lattice.compounds import NaCl, CsCl, ZincBlende
            from ase.lattice.hexagonal import Graphite, Wurtzite
            
            crystal_info = self.crystal_database.get(structure_id, {})
            
            # 创建ASE结构
            if structure_id == "si_diamond":
                atoms = Diamond(symbol='Si', latticeconstant=5.43)
            elif structure_id == "gaas_zincblende":
                atoms = ZincBlende('Ga', 'As', latticeconstant=5.65)
            elif structure_id == "nacl_rocksalt":
                atoms = NaCl(['Na', 'Cl'], latticeconstant=5.64)
            elif structure_id == "cao_rocksalt":
                atoms = NaCl(['Ca', 'O'], latticeconstant=4.81)
            else:
                # 默认使用简单立方
                atoms = SimpleCubic('Si', latticeconstant=5.0)
            
            # 创建VR结构
            structure = self.vr.visualizer.create_from_ase(
                atoms, structure_id, crystal_info.get("name", structure_id)
            )
            
            # 设置渲染模式
            self.vr.visualizer.set_render_mode(structure_id, VRRenderMode.BALL_AND_STICK)
            
            self.current_sample = structure_id
            
            logger.info(f"Loaded crystal structure: {structure_id}")
            return True
            
        except ImportError:
            logger.warning("ASE not available, using mock data")
            # 创建模拟结构
            mock_atoms = self._create_mock_structure(structure_id)
            self.vr.visualizer.add_structure(structure_id, mock_atoms)
            self.current_sample = structure_id
            return True
        
        except Exception as e:
            logger.error(f"Failed to load structure: {e}")
            return False
    
    def _create_mock_structure(self, structure_id: str):
        """创建模拟结构"""
        from dftlammps.metaverse.vr_interface import AtomVRData, VRCrystalStructure
        
        # 创建一个简单的2x2x2原子网格
        atoms = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    pos = VRVector3(i * 2.5, j * 2.5, k * 2.5)
                    atom = AtomVRData(
                        element='Si',
                        position=pos,
                        radius=1.1,
                        color=(0.5, 0.5, 0.5, 1.0),
                        atomic_number=14
                    )
                    atoms.append(atom)
        
        return VRCrystalStructure(
            name=structure_id,
            lattice_constants=(5.0, 5.0, 5.0, 90, 90, 90),
            atoms=atoms
        )
    
    def analyze_miller_planes(self, h: int, k: int, l: int) -> dict:
        """分析密勒平面"""
        if not self.current_sample:
            return {}
        
        structure = self.vr.visualizer.get_structure(self.current_sample)
        if not structure:
            return {}
        
        # 计算平面方程
        # 对于立方晶系，平面方程为: hx + ky + lz = d
        a = structure.lattice_constants[0]
        d_hkl = a / np.sqrt(h**2 + k**2 + l**2)
        
        # 高亮相关原子
        # 简化版本：高亮靠近平面的原子
        highlighted = []
        for i, atom in enumerate(structure.atoms):
            # 计算到平面的距离
            pos = atom.position
            distance = abs(h * pos.x + k * pos.y + l * pos.z - d_hkl) / np.sqrt(h**2 + k**2 + l**2)
            if distance < 1.0:  # 阈值
                highlighted.append(i)
        
        # 应用高亮
        self.vr.visualizer.highlight_atoms(
            self.current_sample, highlighted, (1.0, 0.5, 0.0)
        )
        
        # 添加AR标注
        self.ar.create_annotation(
            ARV3(h * 2, k * 2, l * 2),
            f"({h}{k}{l}) plane, d={d_hkl:.3f}Å"
        )
        
        analysis_result = {
            "miller_indices": (h, k, l),
            "interplanar_spacing": d_hkl,
            "atoms_highlighted": len(highlighted),
            "plane_equation": f"{h}x + {k}y + {l}z = {d_hkl:.3f}"
        }
        
        self.measurements.append(analysis_result)
        
        # 触觉反馈
        self.haptics.simulate_crystal_lattice("simple_cubic", 5.43, 
                                             np.array([h, k, l]))
        
        return analysis_result
    
    def measure_bond_lengths(self) -> list:
        """测量键长"""
        if not self.current_sample:
            return []
        
        structure = self.vr.visualizer.get_structure(self.current_sample)
        if not structure:
            return []
        
        bonds = []
        for i, atom in enumerate(structure.atoms):
            for j in atom.bonds:
                if i < j:
                    other = structure.atoms[j]
                    distance = atom.position.distance_to(other.position)
                    bonds.append({
                        "atom1": i,
                        "atom2": j,
                        "elements": (atom.element, other.element),
                        "distance": distance
                    })
        
        # 添加AR测量
        if bonds:
            avg_bond = np.mean([b["distance"] for b in bonds])
            self.ar.overlay.create_text(
                f"Average bond length: {avg_bond:.3f} Å",
                type('ARV2', (), {'x': 50, 'y': 100})(),
                font_size=24,
                color=(0, 255, 255)
            )
        
        return bonds
    
    def visualize_charge_density(self, charge_data: np.ndarray) -> bool:
        """可视化电荷密度"""
        try:
            self.volume_renderer.load_volume(charge_data)
            self.volume_renderer.create_transfer_function("electron_density")
            
            # 设置渲染参数
            self.volume_renderer.render_settings.render_mode = VRM.DIRECT_VOLUME
            self.volume_renderer.render_settings.color_map = ColorMap.VIRIDIS
            
            logger.info("Charge density visualization loaded")
            return True
            
        except Exception as e:
            logger.error(f"Failed to visualize charge density: {e}")
            return False
    
    def perform_xrd_simulation(self, wavelength: float = 1.54) -> dict:
        """执行XRD模拟"""
        if not self.current_sample:
            return {}
        
        # 简化的XRD模拟
        # 实际实现需要完整的晶体学计算
        two_theta_peaks = []
        intensities = []
        
        # 模拟几个峰
        for i in range(5):
            two_theta = 20 + i * 15 + np.random.normal(0, 0.5)
            intensity = np.random.exponential(1000)
            two_theta_peaks.append(two_theta)
            intensities.append(intensity)
        
        result = {
            "wavelength": wavelength,
            "two_theta_peaks": two_theta_peaks,
            "intensities": intensities,
            "sample": self.current_sample
        }
        
        # 在AR中显示结果
        self.ar.overlay.create_data_viz(
            {
                'values': intensities,
                'labels': [f"{t:.1f}°" for t in two_theta_peaks]
            },
            type('ARV2', (), {'x': 50, 'y': 300})(),
            "bar"
        )
        
        return result
    
    def export_session(self, filepath: str) -> bool:
        """导出实验会话"""
        session_data = {
            "sample": self.current_sample,
            "analysis_type": self.active_analysis,
            "measurements": self.measurements,
            "timestamp": logging.time.time() if hasattr(logging, 'time') else 0
        }
        
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=2)
            logger.info(f"Session exported to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to export session: {e}")
            return False
    
    def get_lab_status(self) -> dict:
        """获取实验室状态"""
        return {
            "current_sample": self.current_sample,
            "loaded_structures": list(self.vr.visualizer.structures.keys()),
            "analysis_active": self.active_analysis != "",
            "measurements_count": len(self.measurements),
            "vr_ready": self.vr.headset_connected,
            "ar_ready": True,
            "haptics_ready": len(self.haptics.connected_devices) > 0
        }


def run_crystallography_lab_demo():
    """运行晶体学实验室演示"""
    print("=" * 70)
    print("  Virtual Crystallography Laboratory")
    print("  虚拟晶体学实验室 - 沉浸式材料分析系统")
    print("=" * 70)
    
    lab = VirtualCrystallographyLab()
    
    # 初始化
    import asyncio
    asyncio.run(lab.initialize())
    print("\n✓ Laboratory initialized")
    
    # 加载晶体结构
    print("\n[1] Loading Crystal Structures...")
    structures = ["si_diamond", "gaas_zincblende", "nacl_rocksalt"]
    for struct_id in structures:
        success = lab.load_crystal_structure(struct_id)
        status = "✓" if success else "✗"
        info = lab.crystal_database.get(struct_id, {})
        print(f"  {status} {info.get('name', struct_id)} ({info.get('formula', 'N/A')})")
    
    # 密勒平面分析
    print("\n[2] Analyzing Miller Planes...")
    lab.current_sample = "si_diamond"
    planes = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
    for h, k, l in planes:
        result = lab.analyze_miller_planes(h, k, l)
        print(f"  ✓ ({h}{k}{l}) plane: d-spacing = {result.get('interplanar_spacing', 0):.3f} Å, "
              f"{result.get('atoms_highlighted', 0)} atoms highlighted")
    
    # 键长测量
    print("\n[3] Measuring Bond Lengths...")
    bonds = lab.measure_bond_lengths()
    if bonds:
        avg_length = np.mean([b['distance'] for b in bonds])
        print(f"  ✓ Found {len(bonds)} bonds, average length: {avg_length:.3f} Å")
    
    # XRD模拟
    print("\n[4] XRD Simulation...")
    xrd_result = lab.perform_xrd_simulation(wavelength=1.54)
    print(f"  ✓ XRD pattern simulated with Cu Kα radiation (λ = 1.54 Å)")
    print(f"    Peaks at 2θ: {', '.join([f'{t:.1f}°' for t in xrd_result['two_theta_peaks']])}")
    
    # 电荷密度可视化
    print("\n[5] Charge Density Visualization...")
    # 创建模拟电荷密度数据
    size = 32
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    z = np.linspace(-2, 2, size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    charge_density = np.exp(-(X**2 + Y**2 + Z**2)) + \
                     0.5 * np.exp(-((X-1)**2 + Y**2 + Z**2))
    
    success = lab.visualize_charge_density(charge_density)
    print(f"  {'✓' if success else '✗'} Charge density volume rendered")
    
    # 实验室状态
    print("\n[6] Laboratory Status...")
    status = lab.get_lab_status()
    print(f"  Current Sample: {status['current_sample']}")
    print(f"  Loaded Structures: {len(status['loaded_structures'])}")
    print(f"  VR System: {'Ready' if status['vr_ready'] else 'Not Ready'}")
    print(f"  Haptics: {'Ready' if status['haptics_ready'] else 'Not Ready'}")
    
    print("\n" + "=" * 70)
    print("  Demo Complete - Virtual Crystallography Laboratory")
    print("=" * 70)


if __name__ == "__main__":
    run_crystallography_lab_demo()
