#!/usr/bin/env python3
"""
主动学习工作流脚本 (Active Learning Workflow)
实现 DP-GEN 风格的 Explore-Label-Retrain 循环

功能特点:
1. 集成 DeePMD-kit API，支持自动化训练
2. 模型偏差(model deviation)不确定性量化
3. 多种探索策略: 温度扰动、压力扰动、结构变形、AIMD采样
4. 模型压缩支持 (DeepPot-SE Compress)
5. 收敛判断和自适应阈值调整

作者: ML Potential Training Expert
日期: 2025-03-09
"""

import os
import sys
import json
import glob
import time
import shutil
import logging
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import queue

# ASE
from ase import Atoms
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS, FIRE
from ase import units

# Dpdata
import dpdata

# DeePMD Python API
try:
    from deepmd.infer import DeepPot
    from deepmd.infer.model_devi import calc_model_devi_v2
    DEEPMD_AVAILABLE = True
except ImportError:
    DEEPMD_AVAILABLE = False
    logging.warning("DeepMD-kit Python API not available. Some features may be limited.")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# 配置类
# ==============================================================================

@dataclass
class UncertaintyConfig:
    """不确定性量化配置"""
    # 模型偏差阈值 (eV/Å)
    f_trust_lo: float = 0.05  # 力偏差下限
    f_trust_hi: float = 0.15  # 力偏差上限
    e_trust_lo: float = 0.05  # 能量偏差下限 (eV/atom)
    e_trust_hi: float = 0.15  # 能量偏差上限 (eV/atom)
    v_trust_lo: float = 0.05  # 维里偏差下限
    v_trust_hi: float = 0.15  # 维里偏差上限
    
    # 自适应阈值调整
    adaptive_threshold: bool = True
    target_candidate_ratio: float = 0.1  # 目标候选结构比例
    adjustment_factor: float = 0.9  # 阈值调整因子
    
    # 模型集成
    num_models: int = 4  # 集成模型数量
    model_devi_method: str = "max"  # max, avg, std


@dataclass  
class ExplorationConfig:
    """探索策略配置"""
    # 温度扰动 (K)
    temperature_range: Tuple[float, float] = (300, 2000)
    temperature_schedule: str = "linear"  # linear, exponential, adaptive
    
    # 压力扰动 (GPa)
    pressure_range: Tuple[float, float] = (-5.0, 50.0)
    pressure_schedule: str = "linear"
    
    # 结构变形
    strain_range: Tuple[float, float] = (-0.15, 0.15)
    deformation_modes: List[str] = field(default_factory=lambda: ['uniaxial', 'biaxial', 'shear'])
    
    # AIMD采样
    aimd_time_step: float = 1.0  # fs
    aimd_nsteps: int = 10000
    aimd_ensemble: str = "NVT"  # NVT, NPT, NVE
    
    # 混合探索参数
    md_steps: int = 50000
    md_sample_freq: int = 100
    max_structures_per_iter: int = 200
    
    # 特殊探索策略
    enable_surface_sampling: bool = True
    enable_defect_sampling: bool = True
    enable_phase_transition: bool = True


@dataclass
class DeepMDConfig:
    """DeePMD训练配置"""
    # 模型架构
    descriptor_type: str = "se_e2_a"  # se_e2_a, se_e3, dpa1, dpa2
    rcut: float = 6.0
    rcut_smth: float = 0.5
    sel: List[int] = field(default_factory=lambda: [50, 50, 50])
    neuron: List[int] = field(default_factory=lambda: [25, 50, 100])
    axis_neuron: int = 16
    fitting_neuron: List[int] = field(default_factory=lambda: [240, 240, 240])
    
    # 训练参数
    start_lr: float = 0.001
    stop_lr: float = 3.51e-8
    decay_steps: int = 5000
    numb_steps: int = 1000000
    batch_size: str = "auto"
    
    # 损失函数权重 (自适应策略)
    start_pref_e: float = 0.02
    limit_pref_e: float = 1.0
    start_pref_f: float = 1000.0
    limit_pref_f: float = 1.0
    start_pref_v: float = 0.01
    limit_pref_v: float = 1.0
    
    # 系统设置
    type_map: List[str] = field(default_factory=lambda: ["H", "C", "N", "O"])
    seed: int = 1
    
    # 路径
    training_data: str = "./data/training"
    validation_data: str = "./data/validation"
    output_dir: str = "./model"
    
    # 压缩设置
    enable_compression: bool = True
    compression_step: int = 10000
    extrapolate: float = 5.0
    check_frequency: int = -1


@dataclass
class ActiveLearningConfig:
    """主动学习工作流配置"""
    # 迭代控制
    max_iterations: int = 20
    convergence_patience: int = 3  # 收敛耐心值
    min_iteration: int = 5  # 最小迭代次数
    
    # 收敛标准
    convergence_criteria: Dict[str, float] = field(default_factory=lambda: {
        'max_force_error': 0.05,  # eV/Å
        'max_energy_error': 0.001,  # eV/atom
        'max_virial_error': 0.01,  # eV
        'candidate_ratio_threshold': 0.05  # 候选结构比例阈值
    })
    
    # 并行设置
    max_workers: int = 4
    parallel_exploration: bool = True
    
    # 工作目录
    work_dir: str = "./active_learning_workflow"
    iteration_prefix: str = "iter"
    
    # 嵌套配置
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    deepmd: DeepMDConfig = field(default_factory=DeepMDConfig)


# ==============================================================================
# 不确定性量化模块
# ==============================================================================

class UncertaintyQuantifier:
    """
    不确定性量化器
    使用模型集成(model ensemble)计算预测不确定性
    """
    
    def __init__(self, model_paths: List[str], config: UncertaintyConfig):
        self.model_paths = model_paths
        self.config = config
        self.models = []
        self._load_models()
        
    def _load_models(self):
        """加载所有模型"""
        if not DEEPMD_AVAILABLE:
            logger.warning("DeepPot not available, using dummy models")
            return
            
        for path in self.model_paths:
            if os.path.exists(path):
                try:
                    model = DeepPot(path)
                    self.models.append(model)
                    logger.info(f"Loaded model: {path}")
                except Exception as e:
                    logger.error(f"Failed to load model {path}: {e}")
        
        if len(self.models) < 2:
            logger.warning(f"Only {len(self.models)} model(s) loaded. Model deviation requires at least 2 models.")
    
    def compute_model_deviation(self, atoms: Atoms) -> Dict[str, float]:
        """
        计算模型偏差 (Model Deviation)
        
        对于力: ε_F,max = max_i sqrt(⟨||F_i||²⟩ - ⟨||F_i||⟩²)
        对于能量: ε_E = sqrt(⟨E²⟩ - ⟨E⟩²) / N_atoms
        对于维里: ε_V = sqrt(⟨||V||²⟩ - ⟨||V||⟩²) / N_atoms
        
        Returns:
            Dict with keys: 'forces', 'energy', 'virial', 'max_force_devi'
        """
        if len(self.models) < 2:
            return {'forces': 0.0, 'energy': 0.0, 'virial': 0.0, 'max_force_devi': 0.0}
        
        # 准备输入
        coord = atoms.get_positions().reshape(1, -1)
        cell = atoms.get_cell().array.reshape(1, -1)
        atype = np.array([self._get_type_index(s) for s in atoms.get_chemical_symbols()])
        
        # 收集所有模型的预测
        energies = []
        forces_list = []
        virials = []
        
        for model in self.models:
            try:
                e, f, v = model.eval(coord, cell, atype)
                energies.append(e[0][0])
                forces_list.append(f[0])
                virials.append(v[0][0] if v is not None else 0.0)
            except Exception as e:
                logger.error(f"Model evaluation failed: {e}")
                continue
        
        if len(energies) < 2:
            return {'forces': 0.0, 'energy': 0.0, 'virial': 0.0, 'max_force_devi': 0.0}
        
        # 计算偏差
        energies = np.array(energies)
        forces_array = np.array(forces_list)  # shape: (n_models, n_atoms, 3)
        virials = np.array(virials)
        n_atoms = len(atoms)
        
        # 力偏差 (按原子计算标准差)
        force_stds = np.std(forces_array, axis=0)  # (n_atoms, 3)
        force_devi_per_atom = np.linalg.norm(force_stds, axis=1)  # (n_atoms,)
        max_force_devi = np.max(force_devi_per_atom)
        
        # 能量偏差
        energy_devi = np.std(energies) / n_atoms
        
        # 维里偏差
        virial_devi = np.std(virials) / n_atoms if len(virials) > 1 else 0.0
        
        return {
            'forces': max_force_devi,
            'energy': energy_devi,
            'virial': virial_devi,
            'max_force_devi': max_force_devi,
            'force_devi_per_atom': force_devi_per_atom,
            'all_forces': forces_array,
            'all_energies': energies
        }
    
    def _get_type_index(self, symbol: str) -> int:
        """获取元素类型索引"""
        # 从第一个模型获取type_map
        if self.models and hasattr(self.models[0], 'get_type_map'):
            type_map = self.models[0].get_type_map()
            if symbol in type_map:
                return type_map.index(symbol)
        # 默认映射
        default_map = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'Li': 0, 'Na': 1, 'K': 2}
        return default_map.get(symbol, 0)
    
    def select_candidates(self, 
                         structures: List[Atoms],
                         f_trust_lo: float = None,
                         f_trust_hi: float = None) -> Tuple[List[Atoms], Dict]:
        """
        选择候选结构进行DFT计算
        
        选择标准: θ_lo ≤ ε_F,max < θ_hi
        即模型不确定性在中等范围的结构
        
        Args:
            structures: 待筛选的结构列表
            f_trust_lo: 力偏差下限
            f_trust_hi: 力偏差上限
            
        Returns:
            candidates: 候选结构列表
            stats: 统计信息
        """
        f_trust_lo = f_trust_lo or self.config.f_trust_lo
        f_trust_hi = f_trust_hi or self.config.f_trust_hi
        
        candidates = []
        stats = {
            'total': len(structures),
            'accurate': 0,  # ε < θ_lo
            'candidate': 0,  # θ_lo ≤ ε < θ_hi
            'failed': 0,    # ε ≥ θ_hi
            'deviations': []
        }
        
        for atoms in structures:
            devi = self.compute_model_deviation(atoms)
            max_devi = devi['max_force_devi']
            stats['deviations'].append(max_devi)
            
            if max_devi < f_trust_lo:
                stats['accurate'] += 1
            elif max_devi >= f_trust_hi:
                stats['failed'] += 1
            else:
                stats['candidate'] += 1
                candidates.append(atoms)
        
        # 如果候选太少，适当放宽阈值
        if self.config.adaptive_threshold and stats['candidate'] < stats['total'] * 0.05:
            logger.warning(f"Too few candidates ({stats['candidate']}), adapting thresholds")
            f_trust_lo *= self.config.adjustment_factor
            return self.select_candidates(structures, f_trust_lo, f_trust_hi)
        
        return candidates, stats
    
    def adjust_thresholds(self, candidate_ratio: float) -> None:
        """
        根据候选结构比例自适应调整阈值
        
        Args:
            candidate_ratio: 候选结构比例
        """
        if not self.config.adaptive_threshold:
            return
            
        target = self.config.target_candidate_ratio
        
        if candidate_ratio > target * 1.5:
            # 候选太多，提高阈值
            self.config.f_trust_lo *= 1.1
            self.config.f_trust_hi *= 1.1
            logger.info(f"Adjusted thresholds up: lo={self.config.f_trust_lo:.3f}, hi={self.config.f_trust_hi:.3f}")
        elif candidate_ratio < target * 0.5:
            # 候选太少，降低阈值
            self.config.f_trust_lo *= 0.9
            self.config.f_trust_hi = max(self.config.f_trust_hi * 0.95, self.config.f_trust_lo + 0.05)
            logger.info(f"Adjusted thresholds down: lo={self.config.f_trust_lo:.3f}, hi={self.config.f_trust_hi:.3f}")


# ==============================================================================
# 探索策略模块
# ==============================================================================

class StructureExplorer:
    """
    结构探索器
    实现多种探索策略：温度扰动、压力扰动、结构变形、AIMD采样
    """
    
    def __init__(self, config: ExplorationConfig, calculator=None):
        self.config = config
        self.calculator = calculator  # ML势计算器
        
    def set_calculator(self, calculator):
        """设置计算器"""
        self.calculator = calculator
    
    # -------------------------------------------------------------------------
    # 1. 温度扰动探索
    # -------------------------------------------------------------------------
    def explore_temperature_perturbation(self,
                                        base_structure: Atoms,
                                        temperatures: List[float] = None,
                                        n_steps: int = None,
                                        n_parallel: int = 4) -> List[Atoms]:
        """
        温度扰动探索
        在不同温度下运行MD采样
        
        Args:
            base_structure: 基础结构
            temperatures: 温度列表，默认使用配置中的范围
            n_steps: MD步数
            n_parallel: 并行任务数
            
        Returns:
            采样得到的结构列表
        """
        if temperatures is None:
            t_min, t_max = self.config.temperature_range
            temperatures = np.linspace(t_min, t_max, 8).tolist()
        
        n_steps = n_steps or self.config.md_steps
        structures = []
        
        logger.info(f"Exploring with temperature perturbation: {len(temperatures)} temperatures")
        
        def run_md_at_temp(temp):
            atoms = base_structure.copy()
            atoms.calc = self.calculator
            
            # 初始化速度
            MaxwellBoltzmannDistribution(atoms, temperature_K=temp)
            
            # NVT MD
            dyn = Langevin(atoms, timestep=self.config.aimd_time_step*units.fs, 
                          temperature_K=temp, friction=0.01)
            
            local_structures = []
            for i in range(0, n_steps, self.config.md_sample_freq):
                dyn.run(self.config.md_sample_freq)
                local_structures.append(atoms.copy())
            
            return local_structures
        
        # 并行执行
        if self.config.parallel_exploration and n_parallel > 1:
            with ProcessPoolExecutor(max_workers=n_parallel) as executor:
                futures = [executor.submit(run_md_at_temp, t) for t in temperatures]
                for future in as_completed(futures):
                    structures.extend(future.result())
        else:
            for temp in temperatures:
                structures.extend(run_md_at_temp(temp))
        
        logger.info(f"Temperature exploration collected {len(structures)} structures")
        return structures
    
    # -------------------------------------------------------------------------
    # 2. 压力扰动探索
    # -------------------------------------------------------------------------
    def explore_pressure_perturbation(self,
                                     base_structure: Atoms,
                                     pressures: List[float] = None,
                                     temperatures: List[float] = None) -> List[Atoms]:
        """
        压力扰动探索
        在不同压力下运行NPT MD
        
        Args:
            base_structure: 基础结构
            pressures: 压力列表 (GPa)
            temperatures: 温度列表 (K)
            
        Returns:
            采样得到的结构列表
        """
        if pressures is None:
            p_min, p_max = self.config.pressure_range
            pressures = np.linspace(p_min, p_max, 6).tolist()
        
        if temperatures is None:
            t_min, t_max = self.config.temperature_range
            temperatures = [300, 600, 900, 1200]
        
        structures = []
        logger.info(f"Exploring with pressure perturbation: {len(pressures)} pressures x {len(temperatures)} temperatures")
        
        for temp in temperatures:
            for press in pressures:
                atoms = base_structure.copy()
                atoms.calc = self.calculator
                
                # 初始化速度
                MaxwellBoltzmannDistribution(atoms, temperature_K=temp)
                
                # NPT MD (Berendsen)
                dyn = NPTBerendsen(atoms, 
                                  timestep=self.config.aimd_time_step*units.fs,
                                  temperature_K=temp,
                                  pressure_au=press * units.GPa,
                                  taut=100*units.fs, 
                                  taup=1000*units.fs,
                                  compressibility_au=4.57e-5/units.bar)
                
                # 平衡
                dyn.run(5000)
                
                # 采样
                for i in range(0, self.config.md_steps, self.config.md_sample_freq):
                    dyn.run(self.config.md_sample_freq)
                    structures.append(atoms.copy())
        
        logger.info(f"Pressure exploration collected {len(structures)} structures")
        return structures
    
    # -------------------------------------------------------------------------
    # 3. 结构变形探索
    # -------------------------------------------------------------------------
    def explore_structure_deformation(self,
                                     base_structure: Atoms,
                                     strain_values: List[float] = None,
                                     modes: List[str] = None) -> List[Atoms]:
        """
        结构变形探索
        应用各种应变模式
        
        Args:
            base_structure: 基础结构
            strain_values: 应变值列表
            modes: 变形模式列表 ['uniaxial', 'biaxial', 'shear', 'volumetric']
            
        Returns:
            变形后的结构列表
        """
        if strain_values is None:
            s_min, s_max = self.config.strain_range
            strain_values = np.linspace(s_min, s_max, 15).tolist()
        
        modes = modes or self.config.deformation_modes
        structures = []
        
        logger.info(f"Exploring with structure deformation: {len(strain_values)} strains x {len(modes)} modes")
        
        cell = base_structure.get_cell()
        
        for strain in strain_values:
            for mode in modes:
                atoms = base_structure.copy()
                
                if mode == 'uniaxial':
                    # 单轴应变 (x方向)
                    deformation = np.eye(3)
                    deformation[0, 0] = 1 + strain
                    
                elif mode == 'biaxial':
                    # 双轴应变 (xy平面)
                    deformation = np.eye(3)
                    deformation[0, 0] = 1 + strain
                    deformation[1, 1] = 1 + strain
                    
                elif mode == 'shear':
                    # 剪切应变
                    deformation = np.eye(3)
                    deformation[0, 1] = strain
                    
                elif mode == 'volumetric':
                    # 体积应变
                    f = (1 + strain) ** (1/3)
                    deformation = np.eye(3) * f
                    
                else:
                    continue
                
                # 应用变形
                new_cell = cell @ deformation.T
                atoms.set_cell(new_cell, scale_atoms=True)
                
                # 可选: 局部弛豫
                if self.calculator is not None:
                    atoms.calc = self.calculator
                    try:
                        optimizer = FIRE(atoms, logfile=None)
                        optimizer.run(fmax=0.5, steps=100)
                    except:
                        pass
                
                structures.append(atoms)
        
        logger.info(f"Structure deformation collected {len(structures)} structures")
        return structures
    
    # -------------------------------------------------------------------------
    # 4. AIMD风格采样
    # -------------------------------------------------------------------------
    def explore_aimd_sampling(self,
                             base_structure: Atoms,
                             temperature: float = None,
                             n_steps: int = None,
                             ensemble: str = None) -> List[Atoms]:
        """
        AIMD风格采样
        模拟从头算分子动力学的采样过程
        
        Args:
            base_structure: 基础结构
            temperature: 采样温度
            n_steps: 总步数
            ensemble: 系综类型
            
        Returns:
            采样结构列表
        """
        temperature = temperature or 900  # 默认900K
        n_steps = n_steps or self.config.aimd_nsteps
        ensemble = ensemble or self.config.aimd_ensemble
        
        logger.info(f"AIMD sampling: T={temperature}K, steps={n_steps}, ensemble={ensemble}")
        
        atoms = base_structure.copy()
        atoms.calc = self.calculator
        
        # 初始化速度
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
        
        # 选择系综
        if ensemble == "NVT":
            dyn = Langevin(atoms, timestep=self.config.aimd_time_step*units.fs,
                          temperature_K=temperature, friction=0.01)
        elif ensemble == "NPT":
            dyn = NPTBerendsen(atoms, timestep=self.config.aimd_time_step*units.fs,
                              temperature_K=temperature,
                              pressure_au=1.0*units.bar,
                              taut=100*units.fs, taup=1000*units.fs,
                              compressibility_au=4.57e-5/units.bar)
        else:  # NVE
            from ase.md.verlet import VelocityVerlet
            dyn = VelocityVerlet(atoms, timestep=self.config.aimd_time_step*units.fs)
        
        structures = []
        
        # 平衡阶段
        dyn.run(1000)
        
        # 生产阶段采样
        for i in range(0, n_steps, self.config.md_sample_freq):
            dyn.run(self.config.md_sample_freq)
            structures.append(atoms.copy())
        
        logger.info(f"AIMD sampling collected {len(structures)} structures")
        return structures
    
    # -------------------------------------------------------------------------
    # 5. 表面采样
    # -------------------------------------------------------------------------
    def explore_surface_structures(self,
                                  bulk_structure: Atoms,
                                  miller_indices: List[Tuple[int, int, int]] = None,
                                  vacuum: float = 15.0,
                                  n_layers: int = 6) -> List[Atoms]:
        """
        表面结构探索
        生成不同晶面的表面结构
        
        Args:
            bulk_structure: 体相结构
            miller_indices: 米勒指数列表
            vacuum: 真空层厚度 (Å)
            n_layers: 原子层数
            
        Returns:
            表面结构列表
        """
        try:
            from ase.build import surface
        except ImportError:
            logger.warning("ase.build.surface not available")
            return []
        
        if miller_indices is None:
            miller_indices = [(1,0,0), (1,1,0), (1,1,1), (2,1,0)]
        
        structures = []
        
        for hkl in miller_indices:
            try:
                # 构建表面
                slab = surface(bulk_structure, hkl, n_layers, vacuum)
                structures.append(slab)
                
                # 添加表面弛豫结构
                for strain in [-0.05, 0.0, 0.05]:
                    slab_strained = slab.copy()
                    cell = slab_strained.get_cell()
                    cell[2, 2] *= (1 + strain)  # 修改真空层方向
                    slab_strained.set_cell(cell, scale_atoms=False)
                    structures.append(slab_strained)
                    
            except Exception as e:
                logger.warning(f"Failed to create surface {hkl}: {e}")
        
        logger.info(f"Surface exploration collected {len(structures)} structures")
        return structures
    
    # -------------------------------------------------------------------------
    # 6. 缺陷结构采样
    # -------------------------------------------------------------------------
    def explore_defect_structures(self,
                                 base_structure: Atoms,
                                 defect_types: List[str] = None) -> List[Atoms]:
        """
        缺陷结构探索
        生成空位、间隙等缺陷结构
        
        Args:
            base_structure: 基础结构
            defect_types: 缺陷类型 ['vacancy', 'interstitial', 'substitution']
            
        Returns:
            缺陷结构列表
        """
        defect_types = defect_types or ['vacancy', 'interstitial']
        structures = []
        
        positions = base_structure.get_positions()
        symbols = base_structure.get_chemical_symbols()
        
        for defect_type in defect_types:
            if defect_type == 'vacancy':
                # 生成空位
                n_atoms = len(base_structure)
                vacancy_indices = np.random.choice(n_atoms, min(5, n_atoms//10), replace=False)
                
                for idx in vacancy_indices:
                    atoms = base_structure.copy()
                    del atoms[idx]
                    structures.append(atoms)
                    
            elif defect_type == 'interstitial':
                # 生成间隙原子
                for i in range(5):
                    atoms = base_structure.copy()
                    # 在随机位置添加一个原子
                    new_pos = positions.mean(axis=0) + np.random.randn(3) * 0.5
                    atoms.append(atoms[0].symbol)  # 添加与第一个原子相同类型的原子
                    atoms.positions[-1] = new_pos
                    structures.append(atoms)
        
        logger.info(f"Defect exploration collected {len(structures)} structures")
        return structures
    
    # -------------------------------------------------------------------------
    # 综合探索
    # -------------------------------------------------------------------------
    def comprehensive_exploration(self, base_structures: List[Atoms]) -> List[Atoms]:
        """
        综合探索策略
        结合多种探索方法
        
        Args:
            base_structures: 基础结构列表
            
        Returns:
            所有探索得到的结构
        """
        all_structures = []
        
        for base in base_structures:
            # 1. 温度探索
            if self.config.md_steps > 0:
                temp_structs = self.explore_temperature_perturbation(base)
                all_structures.extend(temp_structs)
            
            # 2. 结构变形
            deform_structs = self.explore_structure_deformation(base)
            all_structures.extend(deform_structs)
            
            # 3. AIMD采样
            aimd_structs = self.explore_aimd_sampling(base)
            all_structures.extend(aimd_structs)
            
            # 4. 表面采样
            if self.config.enable_surface_sampling:
                try:
                    surf_structs = self.explore_surface_structures(base)
                    all_structures.extend(surf_structs)
                except:
                    pass
            
            # 5. 缺陷采样
            if self.config.enable_defect_sampling:
                defect_structs = self.explore_defect_structures(base)
                all_structures.extend(defect_structs)
        
        # 去重 (基于能量和结构的简单去重)
        all_structures = self._deduplicate_structures(all_structures)
        
        # 限制数量
        max_structs = self.config.max_structures_per_iter
        if len(all_structures) > max_structs:
            indices = np.random.choice(len(all_structures), max_structs, replace=False)
            all_structures = [all_structures[i] for i in indices]
        
        logger.info(f"Comprehensive exploration: total {len(all_structures)} unique structures")
        return all_structures
    
    def _deduplicate_structures(self, structures: List[Atoms], 
                               energy_tol: float = 0.01) -> List[Atoms]:
        """简单的结构去重"""
        if len(structures) <= 1:
            return structures
            
        unique = [structures[0]]
        
        for struct in structures[1:]:
            is_duplicate = False
            
            for u in unique:
                # 简单比较：原子数和元素组成
                if len(struct) != len(u):
                    continue
                if set(struct.get_chemical_symbols()) != set(u.get_chemical_symbols()):
                    continue
                # 可以添加更多比较逻辑
                is_duplicate = True
                break
            
            if not is_duplicate:
                unique.append(struct)
        
        return unique


# ==============================================================================
# DeePMD 训练器
# ==============================================================================

class DeepMDTrainer:
    """
    DeePMD-kit 训练管理器
    支持模型训练和压缩
    """
    
    def __init__(self, config: DeepMDConfig):
        self.config = config
        self.iteration = 0
        
    def generate_input(self, output_file: str = "input.json") -> str:
        """生成DeePMD输入文件"""
        
        input_dict = {
            "model": {
                "type_map": self.config.type_map,
                "descriptor": {
                    "type": self.config.descriptor_type,
                    "rcut": self.config.rcut,
                    "rcut_smth": self.config.rcut_smth,
                    "sel": self.config.sel,
                    "neuron": self.config.neuron,
                    "resnet_dt": False,
                    "axis_neuron": self.config.axis_neuron,
                    "seed": self.config.seed,
                    "type_one_side": True
                },
                "fitting_net": {
                    "neuron": self.config.fitting_neuron,
                    "resnet_dt": True,
                    "seed": self.config.seed
                }
            },
            "learning_rate": {
                "type": "exp",
                "decay_steps": self.config.decay_steps,
                "start_lr": self.config.start_lr,
                "stop_lr": self.config.stop_lr
            },
            "loss": {
                "type": "ener",
                "start_pref_e": self.config.start_pref_e,
                "limit_pref_e": self.config.limit_pref_e,
                "start_pref_f": self.config.start_pref_f,
                "limit_pref_f": self.config.limit_pref_f,
                "start_pref_v": self.config.start_pref_v,
                "limit_pref_v": self.config.limit_pref_v
            },
            "training": {
                "training_data": {
                    "systems": [self.config.training_data],
                    "batch_size": self.config.batch_size
                },
                "validation_data": {
                    "systems": [self.config.validation_data],
                    "batch_size": self.config.batch_size
                },
                "numb_steps": self.config.numb_steps,
                "seed": self.config.seed + 9,
                "disp_file": "lcurve.out",
                "disp_freq": 1000,
                "save_freq": 10000,
                "max_ckpt_keep": 5
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(input_dict, f, indent=2)
        
        logger.info(f"Generated DeePMD input: {output_file}")
        return output_file
    
    def train(self, 
             input_file: str = "input.json",
             restart: bool = False,
             init_model: str = None) -> str:
        """
        执行训练
        
        Args:
            input_file: 输入文件路径
            restart: 是否从checkpoint重启
            init_model: 初始化模型路径
            
        Returns:
            模型目录路径
        """
        logger.info("Starting DeePMD training...")
        
        cmd = ["dp", "train", input_file]
        
        if restart:
            cmd.append("--restart")
        elif init_model:
            cmd.extend(["--init-model", init_model])
        
        try:
            subprocess.run(cmd, check=True, capture_output=False)
            logger.info("Training completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e}")
            raise
        
        return os.path.dirname(input_file) or "."
    
    def train_ensemble(self, 
                      input_file: str = "input.json",
                      n_models: int = 4) -> List[str]:
        """
        训练模型集成
        使用不同随机种子训练多个模型
        
        Args:
            input_file: 输入文件模板
            n_models: 模型数量
            
        Returns:
            各模型目录列表
        """
        model_dirs = []
        base_seed = self.config.seed
        
        for i in range(n_models):
            model_dir = f"model_{i}"
            os.makedirs(model_dir, exist_ok=True)
            
            # 修改种子
            self.config.seed = base_seed + i * 100
            
            # 生成输入文件
            model_input = os.path.join(model_dir, "input.json")
            self.generate_input(model_input)
            
            # 训练
            self.train(model_input)
            model_dirs.append(model_dir)
        
        # 恢复原始种子
        self.config.seed = base_seed
        
        return model_dirs
    
    def freeze_model(self, 
                    model_dir: str = ".",
                    output_name: str = "graph.pb") -> str:
        """
        冻结模型
        
        Args:
            model_dir: 模型目录
            output_name: 输出文件名
            
        Returns:
            冻结模型路径
        """
        logger.info(f"Freezing model in {model_dir}...")
        
        output_path = os.path.join(model_dir, output_name)
        cmd = ["dp", "freeze", "-o", output_name]
        
        subprocess.run(cmd, cwd=model_dir, check=True)
        
        frozen_path = os.path.join(model_dir, output_name)
        logger.info(f"Model frozen: {frozen_path}")
        
        return frozen_path
    
    def compress_model(self,
                      model_path: str,
                      output_path: str = None,
                      step: float = None,
                      extrapolate: float = None) -> str:
        """
        压缩模型
        
        模型压缩使用表格化推理，可加速10倍以上，减少内存消耗20倍
        
        Args:
            model_path: 原始模型路径
            output_path: 输出路径
            step: 表格步长
            extrapolate: 外推参数
            
        Returns:
            压缩模型路径
        """
        if not self.config.enable_compression:
            logger.info("Model compression disabled")
            return model_path
        
        if output_path is None:
            output_path = model_path.replace(".pb", "-compress.pb")
            if output_path == model_path:
                output_path = model_path + "-compress"
        
        logger.info(f"Compressing model: {model_path} -> {output_path}")
        
        cmd = ["dp", "compress", "-i", model_path, "-o", output_path]
        
        if step is not None:
            cmd.extend(["-s", str(step)])
        if extrapolate is not None:
            cmd.extend(["-e", str(extrapolate)])
        
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Model compressed: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Compression failed: {e}")
            return model_path
    
    def test_model(self,
                  model_path: str,
                  test_data: str,
                  output_dir: str = "test_results") -> Dict:
        """
        测试模型
        
        Args:
            model_path: 模型路径
            test_data: 测试数据路径
            output_dir: 输出目录
            
        Returns:
            测试结果字典
        """
        logger.info(f"Testing model: {model_path}")
        
        cmd = ["dp", "test", "-m", model_path, "-s", test_data, "-d", output_dir]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 解析结果
        results = self._parse_test_output(result.stdout)
        results['model'] = model_path
        results['test_data'] = test_data
        
        return results
    
    def _parse_test_output(self, output: str) -> Dict:
        """解析dp test输出"""
        results = {}
        
        for line in output.split('\n'):
            if 'Energy RMSE' in line and 'Natoms' not in line:
                try:
                    results['energy_rmse'] = float(line.split(':')[1].split()[0])
                except:
                    pass
            elif 'Force RMSE' in line:
                try:
                    results['force_rmse'] = float(line.split(':')[1].split()[0])
                except:
                    pass
            elif 'Virial RMSE' in line and 'Natoms' not in line:
                try:
                    results['virial_rmse'] = float(line.split(':')[1].split()[0])
                except:
                    pass
        
        return results


# ==============================================================================
# DFT 标签器接口
# ==============================================================================

class DFTLabeler:
    """
    DFT标签器
    用于对候选结构进行DFT单点计算
    """
    
    def __init__(self, 
                 calculator_type: str = "vasp",
                 work_dir: str = "./dft_calculations",
                 max_workers: int = 4):
        self.calculator_type = calculator_type
        self.work_dir = work_dir
        self.max_workers = max_workers
        os.makedirs(work_dir, exist_ok=True)
    
    def label_structures(self,
                        structures: List[Atoms],
                        iteration: int = 0) -> str:
        """
        对结构列表进行DFT标注
        
        Args:
            structures: 待标注结构
            iteration: 当前迭代数
            
        Returns:
            标注数据目录
        """
        iter_dir = os.path.join(self.work_dir, f"iter_{iteration:03d}")
        os.makedirs(iter_dir, exist_ok=True)
        
        logger.info(f"Labeling {len(structures)} structures with DFT...")
        
        for i, atoms in enumerate(structures):
            struct_dir = os.path.join(iter_dir, f"struct_{i:04d}")
            self._submit_dft_calculation(atoms, struct_dir)
        
        # 等待计算完成并收集结果
        labeled_data_dir = os.path.join(iter_dir, "labeled_data")
        self._collect_results(iter_dir, labeled_data_dir)
        
        return labeled_data_dir
    
    def _submit_dft_calculation(self, atoms: Atoms, work_dir: str):
        """提交单个DFT计算"""
        os.makedirs(work_dir, exist_ok=True)
        
        # 保存结构
        write(os.path.join(work_dir, "POSCAR"), atoms)
        
        # 这里应该调用实际的DFT计算
        # 示例：生成输入文件
        if self.calculator_type == "vasp":
            self._generate_vasp_input(work_dir)
        elif self.calculator_type == "qe":
            self._generate_qe_input(work_dir, atoms)
        elif self.calculator_type == "abacus":
            self._generate_abacus_input(work_dir, atoms)
    
    def _generate_vasp_input(self, work_dir: str):
        """生成VASP输入文件"""
        incar_content = """
SYSTEM = DFT Calculation for Active Learning
ISTART = 0
ICHARG = 2
ENCUT = 500
ISMEAR = 0
SIGMA = 0.05
EDIFF = 1E-6
EDIFFG = -0.01
NSW = 0
IBRION = -1
"""
        with open(os.path.join(work_dir, "INCAR"), 'w') as f:
            f.write(incar_content)
    
    def _generate_qe_input(self, work_dir: str, atoms: Atoms):
        """生成Quantum ESPRESSO输入文件"""
        # 简化版本
        pass
    
    def _generate_abacus_input(self, work_dir: str, atoms: Atoms):
        """生成ABACUS输入文件"""
        # 简化版本
        pass
    
    def _collect_results(self, iter_dir: str, output_dir: str):
        """收集DFT计算结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 使用dpdata收集结果
        # 这里简化处理，实际应解析VASP/ABACUS输出
        logger.info(f"Results would be collected in {output_dir}")


# ==============================================================================
# 主动学习工作流
# ==============================================================================

class ActiveLearningWorkflow:
    """
    主动学习工作流主类
    实现完整的 Explore -> Label -> Retrain 循环
    """
    
    def __init__(self, config: ActiveLearningConfig):
        self.config = config
        self.iteration = 0
        self.convergence_history = []
        self.best_model = None
        
        # 创建工作目录
        os.makedirs(config.work_dir, exist_ok=True)
        
        # 初始化组件
        self.trainer = DeepMDTrainer(config.deepmd)
        self.explorer = StructureExplorer(config.exploration)
        self.labeler = DFTLabeler(work_dir=os.path.join(config.work_dir, "dft"))
        self.quantifier = None  # 将在训练后初始化
        
        # 设置日志
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志记录"""
        log_file = os.path.join(self.config.work_dir, "workflow.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    def initialize(self, 
                  initial_structures: List[Atoms],
                  initial_data_dir: str = None):
        """
        初始化主动学习工作流
        
        Args:
            initial_structures: 初始结构列表
            initial_data_dir: 初始训练数据目录（如果有）
        """
        logger.info("=" * 60)
        logger.info("Initializing Active Learning Workflow")
        logger.info("=" * 60)
        
        # 保存初始结构
        init_structs_file = os.path.join(self.config.work_dir, "initial_structures.traj")
        write(init_structs_file, initial_structures)
        
        # 如果有初始数据，直接开始训练
        if initial_data_dir and os.path.exists(initial_data_dir):
            logger.info(f"Using initial data: {initial_data_dir}")
            self.config.deepmd.training_data = initial_data_dir
        else:
            # 需要对初始结构进行DFT计算
            logger.info("Labeling initial structures with DFT...")
            labeled_data = self.labeler.label_structures(initial_structures, iteration=0)
            self.config.deepmd.training_data = labeled_data
        
        # 初始训练
        logger.info("Performing initial training...")
        self._train_iteration(init_iter=True)
        
        logger.info("Initialization completed!")
    
    def _train_iteration(self, init_iter: bool = False) -> List[str]:
        """
        执行一次训练迭代
        
        Args:
            init_iter: 是否为初始迭代
            
        Returns:
            模型路径列表
        """
        iter_dir = os.path.join(self.config.work_dir, 
                               f"{self.config.iteration_prefix}_{self.iteration:03d}")
        os.makedirs(iter_dir, exist_ok=True)
        
        # 生成输入文件
        input_file = os.path.join(iter_dir, "input.json")
        self.trainer.generate_input(input_file)
        
        # 训练模型集成
        model_dirs = self.trainer.train_ensemble(
            input_file, 
            n_models=self.config.uncertainty.num_models
        )
        
        # 冻结所有模型
        model_paths = []
        for i, model_dir in enumerate(model_dirs):
            frozen_path = self.trainer.freeze_model(
                model_dir, 
                output_name=f"graph_{i}.pb"
            )
            model_paths.append(frozen_path)
        
        # 压缩模型（可选）
        compressed_paths = []
        for path in model_paths:
            compressed = self.trainer.compress_model(path)
            compressed_paths.append(compressed)
        
        # 更新不确定性量化器
        self.quantifier = UncertaintyQuantifier(
            compressed_paths,
            self.config.uncertainty
        )
        
        # 更新探索器的计算器
        if compressed_paths:
            from deepmd.calculator import DP
            self.explorer.set_calculator(DP(model=compressed_paths[0]))
        
        return compressed_paths
    
    def explore(self) -> Tuple[List[Atoms], Dict]:
        """
        探索阶段：发现不确定性高的结构
        
        Returns:
            (候选结构列表, 统计信息)
        """
        logger.info(f"\n--- Exploration Phase (Iteration {self.iteration}) ---")
        
        # 加载当前最佳模型进行探索
        if self.quantifier is None:
            raise RuntimeError("Models not trained yet. Call initialize() first.")
        
        # 获取基础结构 (可以从上一轮采样或初始结构)
        base_structures = self._get_base_structures()
        
        # 综合探索
        all_structures = self.explorer.comprehensive_exploration(base_structures)
        
        # 使用模型偏差选择候选结构
        candidates, stats = self.quantifier.select_candidates(all_structures)
        
        logger.info(f"Exploration stats: {stats}")
        
        # 保存探索结果
        self._save_exploration_results(all_structures, candidates)
        
        return candidates, stats
    
    def _get_base_structures(self) -> List[Atoms]:
        """获取用于探索的基础结构"""
        # 简化版本：从初始结构或上一轮的结构中选择
        init_file = os.path.join(self.config.work_dir, "initial_structures.traj")
        if os.path.exists(init_file):
            return read(init_file, index=':')
        return []
    
    def _save_exploration_results(self, all_structs: List[Atoms], 
                                 candidates: List[Atoms]):
        """保存探索结果"""
        iter_dir = os.path.join(self.config.work_dir,
                               f"{self.config.iteration_prefix}_{self.iteration:03d}")
        
        write(os.path.join(iter_dir, "all_explored.traj"), all_structs)
        write(os.path.join(iter_dir, "candidates.traj"), candidates)
    
    def label(self, candidates: List[Atoms]) -> str:
        """
        标注阶段：对候选结构进行DFT计算
        
        Args:
            candidates: 候选结构列表
            
        Returns:
            标注数据目录
        """
        logger.info(f"\n--- Labeling Phase (Iteration {self.iteration}) ---")
        
        # 限制每轮标注数量
        max_label = self.config.exploration.max_structures_per_iter // 2
        if len(candidates) > max_label:
            # 按不确定性排序，选择最不确定的
            deviations = []
            for atoms in candidates:
                devi = self.quantifier.compute_model_deviation(atoms)
                deviations.append(devi['max_force_devi'])
            
            indices = np.argsort(deviations)[-max_label:]
            candidates = [candidates[i] for i in indices]
        
        labeled_data = self.labeler.label_structures(candidates, self.iteration)
        
        return labeled_data
    
    def retrain(self, new_data_dir: str) -> List[str]:
        """
        重新训练阶段
        
        Args:
            new_data_dir: 新标注的数据目录
            
        Returns:
            新模型路径列表
        """
        logger.info(f"\n--- Retraining Phase (Iteration {self.iteration}) ---")
        
        # 合并数据
        self._merge_training_data(new_data_dir)
        
        # 增量训练
        model_paths = self._train_iteration()
        
        # 测试模型
        self._evaluate_models(model_paths)
        
        return model_paths
    
    def _merge_training_data(self, new_data_dir: str):
        """合并新旧训练数据"""
        # 使用dpdata合并
        # 简化版本
        logger.info(f"Merging new data from {new_data_dir}")
    
    def _evaluate_models(self, model_paths: List[str]):
        """评估模型性能"""
        results = []
        for path in model_paths:
            if os.path.exists(self.config.deepmd.validation_data):
                result = self.trainer.test_model(
                    path, 
                    self.config.deepmd.validation_data
                )
                results.append(result)
        
        if results:
            avg_force_rmse = np.mean([r.get('force_rmse', 0) for r in results])
            avg_energy_rmse = np.mean([r.get('energy_rmse', 0) for r in results])
            
            logger.info(f"Average Force RMSE: {avg_force_rmse:.4f} eV/Å")
            logger.info(f"Average Energy RMSE: {avg_energy_rmse:.6f} eV/atom")
            
            self.convergence_history.append({
                'iteration': self.iteration,
                'force_rmse': avg_force_rmse,
                'energy_rmse': avg_energy_rmse
            })
    
    def check_convergence(self, stats: Dict) -> bool:
        """
        检查收敛性
        
        收敛标准:
        1. 力误差 < threshold
        2. 能量误差 < threshold
        3. 候选结构比例 < threshold
        4. 连续多轮满足上述条件
        
        Args:
            stats: 探索阶段统计信息
            
        Returns:
            是否收敛
        """
        if len(self.convergence_history) < self.config.min_iteration:
            return False
        
        criteria = self.config.convergence_criteria
        
        # 检查最新结果
        if self.convergence_history:
            latest = self.convergence_history[-1]
            
            force_converged = latest.get('force_rmse', float('inf')) < criteria['max_force_error']
            energy_converged = latest.get('energy_rmse', float('inf')) < criteria['max_energy_error']
            
            # 候选比例收敛
            candidate_ratio = stats.get('candidate', 0) / max(stats.get('total', 1), 1)
            ratio_converged = candidate_ratio < criteria['candidate_ratio_threshold']
            
            if force_converged and energy_converged and ratio_converged:
                # 检查是否连续多轮收敛
                if len(self.convergence_history) >= self.config.convergence_patience:
                    recent = self.convergence_history[-self.config.convergence_patience:]
                    all_converged = all(
                        r.get('force_rmse', float('inf')) < criteria['max_force_error']
                        for r in recent
                    )
                    if all_converged:
                        logger.info("Convergence achieved!")
                        return True
        
        return False
    
    def run_iteration(self) -> Dict:
        """
        运行一个完整的主动学习迭代
        
        Returns:
            迭代结果字典
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Active Learning Iteration {self.iteration}")
        logger.info(f"{'='*60}")
        
        # 1. 探索
        candidates, stats = self.explore()
        
        if len(candidates) == 0:
            logger.info("No uncertain structures found. Converged!")
            return {'converged': True, 'reason': 'no_candidates'}
        
        # 2. 标注
        labeled_data = self.label(candidates)
        
        # 3. 重新训练
        model_paths = self.retrain(labeled_data)
        
        # 4. 检查收敛
        converged = self.check_convergence(stats)
        
        # 5. 自适应调整
        if self.config.uncertainty.adaptive_threshold:
            candidate_ratio = stats['candidate'] / max(stats['total'], 1)
            self.quantifier.adjust_thresholds(candidate_ratio)
        
        self.iteration += 1
        
        return {
            'converged': converged,
            'iteration': self.iteration,
            'n_candidates': len(candidates),
            'stats': stats,
            'models': model_paths
        }
    
    def run(self, max_iterations: int = None) -> str:
        """
        运行完整的主动学习工作流
        
        Args:
            max_iterations: 最大迭代次数
            
        Returns:
            最终模型路径
        """
        max_iterations = max_iterations or self.config.max_iterations
        
        logger.info("\n" + "="*60)
        logger.info("Starting Active Learning Workflow")
        logger.info("="*60)
        
        for i in range(max_iterations):
            result = self.run_iteration()
            
            if result['converged']:
                logger.info(f"\nConvergence achieved at iteration {self.iteration}!")
                break
        
        # 保存最终模型
        final_model = self._finalize()
        
        return final_model
    
    def _finalize(self) -> str:
        """完成工作流，保存最终模型"""
        logger.info("\nFinalizing workflow...")
        
        # 选择最佳模型 (这里简化处理，选择最后一个)
        if self.quantifier and self.quantifier.models:
            final_model = self.quantifier.model_paths[0]
            final_path = os.path.join(self.config.work_dir, "final_model.pb")
            shutil.copy(final_model, final_path)
            
            # 保存配置
            config_path = os.path.join(self.config.work_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(asdict(self.config), f, indent=2)
            
            logger.info(f"Final model saved: {final_path}")
            return final_path
        
        return None
    
    def generate_report(self) -> str:
        """
        生成训练报告
        
        Returns:
            报告文件路径
        """
        report_path = os.path.join(self.config.work_dir, "training_report.md")
        
        with open(report_path, 'w') as f:
            f.write("# 主动学习训练报告\n\n")
            f.write(f"## 工作流配置\n")
            f.write(f"- 最大迭代次数: {self.config.max_iterations}\n")
            f.write(f"- 模型数量: {self.config.uncertainty.num_models}\n")
            f.write(f"- 力偏差阈值: [{self.config.uncertainty.f_trust_lo}, {self.config.uncertainty.f_trust_hi}]\n")
            f.write(f"\n## 收敛历史\n\n")
            f.write("| 迭代 | 力误差 (eV/Å) | 能量误差 (eV/atom) |\n")
            f.write("|------|---------------|-------------------|\n")
            
            for h in self.convergence_history:
                f.write(f"| {h['iteration']} | {h.get('force_rmse', 'N/A'):.4f} | "
                       f"{h.get('energy_rmse', 'N/A'):.6f} |\n")
        
        logger.info(f"Report saved: {report_path}")
        return report_path


# ==============================================================================
# 工具函数
# ==============================================================================

def create_workflow_from_config(config_file: str) -> ActiveLearningWorkflow:
    """
    从配置文件创建工作流
    
    Args:
        config_file: JSON配置文件路径
        
    Returns:
        ActiveLearningWorkflow实例
    """
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    
    # 解析配置
    uncertainty_config = UncertaintyConfig(**config_dict.get('uncertainty', {}))
    exploration_config = ExplorationConfig(**config_dict.get('exploration', {}))
    deepmd_config = DeepMDConfig(**config_dict.get('deepmd', {}))
    
    al_config = ActiveLearningConfig(
        max_iterations=config_dict.get('max_iterations', 20),
        convergence_criteria=config_dict.get('convergence_criteria', {}),
        work_dir=config_dict.get('work_dir', './active_learning'),
        uncertainty=uncertainty_config,
        exploration=exploration_config,
        deepmd=deepmd_config
    )
    
    return ActiveLearningWorkflow(al_config)


def quick_start_example():
    """快速开始示例"""
    # 创建配置
    config = ActiveLearningConfig(
        max_iterations=10,
        work_dir="./al_workflow_demo",
        uncertainty=UncertaintyConfig(
            f_trust_lo=0.05,
            f_trust_hi=0.15,
            num_models=4
        ),
        exploration=ExplorationConfig(
            temperature_range=(300, 1500),
            max_structures_per_iter=100
        ),
        deepmd=DeepMDConfig(
            type_map=["H", "O"],
            numb_steps=100000,
            enable_compression=True
        )
    )
    
    # 创建工作流
    workflow = ActiveLearningWorkflow(config)
    
    # 准备初始结构 (这里用简单的水分子作为示例)
    from ase.build import molecule
    water = molecule('H2O')
    water.set_cell([10, 10, 10])
    water.center()
    
    # 初始化并运行
    workflow.initialize([water])
    final_model = workflow.run()
    
    # 生成报告
    report = workflow.generate_report()
    
    print(f"\nWorkflow completed!")
    print(f"Final model: {final_model}")
    print(f"Report: {report}")


# ==============================================================================
# 主程序
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Active Learning Workflow for ML Potentials")
    parser.add_argument("--config", type=str, help="Configuration file (JSON)")
    parser.add_argument("--init-struct", type=str, help="Initial structures file")
    parser.add_argument("--work-dir", type=str, default="./active_learning", 
                       help="Working directory")
    parser.add_argument("--max-iter", type=int, default=10, help="Maximum iterations")
    parser.add_argument("--demo", action="store_true", help="Run demo example")
    
    args = parser.parse_args()
    
    if args.demo:
        quick_start_example()
    elif args.config:
        workflow = create_workflow_from_config(args.config)
        
        if args.init_struct:
            initial_structs = read(args.init_struct, index=':')
            workflow.initialize(initial_structs)
        
        workflow.run(max_iterations=args.max_iter)
        workflow.generate_report()
    else:
        parser.print_help()
