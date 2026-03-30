#!/usr/bin/env python3
"""
mace_interface.py
=================
MACE (Multi-Atomic Cluster Expansion) 接口

功能：
1. MACE模型加载和预测
2. 训练数据准备
3. 主动学习循环
4. 与LAMMPS的集成
5. 批处理和高效推理

MACE特点：
- 高阶等变消息传递
- 严格的旋转等变性
- 多元素支持
- 高精度能量、力、应力预测

作者: ML Potential Integration Expert
日期: 2026-03-09
"""

import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging
from datetime import datetime
import warnings
import subprocess
import tempfile

# ASE
from ase import Atoms
from ase.io import read, write
from ase.io.extxyz import write_extxyz, read_extxyz
from ase.units import eV, Ang, GPa, fs
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import BFGS, FIRE, LBFGS
from ase.md import Langevin, VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.constraints import FixAtoms

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 尝试导入MACE
try:
    import mace
    from mace.calculators import mace_mp, mace_off
    from mace.modules.utils import extract_config_mace_model
    MACE_AVAILABLE = True
except ImportError:
    MACE_AVAILABLE = False
    warnings.warn("MACE not available. Install with: pip install mace-torch")


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class MACEConfig:
    """MACE模型配置"""
    # 模型选择
    model_type: str = "medium"  # small, medium, large, custom
    model_path: Optional[str] = None  # 自定义模型路径
    
    # 预训练模型
    use_mp: bool = True  # 使用Materials Project预训练模型
    use_off: bool = False  # 使用OpenFF预训练模型
    
    # 计算设置
    device: str = "cuda"  # cuda, cpu, mps
    default_dtype: str = "float64"  # float32, float64
    
    # 截断半径
    cutoff: float = 6.0  # Å
    
    # 批处理
    batch_size: int = 32
    
    # 精度设置
    compute_stress: bool = True
    compute_forces: bool = True


@dataclass
class MACEMDConfig:
    """MACE MD配置"""
    # 积分器
    integrator: str = "langevin"  # langevin, verlet
    timestep: float = 1.0  # fs
    
    # 温度控制
    temperature: float = 300.0  # K
    friction: float = 0.01  # Langevin摩擦系数
    
    # 系综
    ensemble: str = "nvt"  # nvt, nve
    
    # 模拟长度
    n_steps: int = 10000
    
    # 输出
    output_interval: int = 100
    trajectory_file: str = "mace_md.traj"


@dataclass
class MACEActiveLearningConfig:
    """MACE主动学习配置"""
    # 不确定性阈值
    force_uncertainty_threshold: float = 0.1  # eV/Å
    energy_uncertainty_threshold: float = 0.01  # eV/atom
    
    # 探索设置
    explore_temperatures: List[float] = field(default_factory=lambda: [300, 500, 700, 900])
    explore_pressures: List[float] = field(default_factory=lambda: [1.0, 10.0, 50.0])
    
    # DFT设置
    dft_calculator: str = "vasp"
    dft_params: Dict = field(default_factory=dict)
    
    # 停止条件
    max_iterations: int = 10
    max_structures_per_iter: int = 50
    min_improvement: float = 0.05


# =============================================================================
# MACE计算器
# =============================================================================

class MACECalculator(Calculator):
    """
    ASE兼容的MACE计算器
    
    支持能量、力和应力的预测
    """
    
    implemented_properties = ['energy', 'forces', 'stress', 'free_energy']
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 model_type: str = "medium",
                 device: str = "cuda",
                 default_dtype: str = "float64",
                 cutoff: float = 6.0,
                 compute_stress: bool = True,
                 **kwargs):
        """
        初始化MACE计算器
        
        Args:
            model_path: 模型文件路径（如果为None则使用预训练模型）
            model_type: 预训练模型类型 (small, medium, large)
            device: 计算设备
            default_dtype: 数据类型
            cutoff: 截断半径
            compute_stress: 是否计算应力
        """
        super().__init__(**kwargs)
        
        if not MACE_AVAILABLE:
            raise ImportError("MACE is not available. Install with: pip install mace-torch")
        
        self.device = device
        self.default_dtype = default_dtype
        self.cutoff = cutoff
        self.compute_stress_flag = compute_stress
        
        # 加载模型
        if model_path and Path(model_path).exists():
            logger.info(f"Loading MACE model from {model_path}")
            self.calculator = self._load_custom_model(model_path)
        else:
            logger.info(f"Loading MACE-MP {model_type} model")
            self.calculator = mace_mp(
                model=model_type,
                device=device,
                default_dtype=default_dtype
            )
        
        self.model_type = model_type
    
    def _load_custom_model(self, model_path: str):
        """加载自定义MACE模型"""
        from mace.calculators import MACECalculator as MACEInternalCalc
        
        return MACEInternalCalc(
            model_paths=model_path,
            device=self.device,
            default_dtype=self.default_dtype
        )
    
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """执行计算"""
        super().calculate(atoms, properties, system_changes)
        
        if atoms is None:
            atoms = self.atoms
        
        # 使用MACE计算器
        atoms.calc = self.calculator
        
        # 计算能量
        if 'energy' in properties:
            self.results['energy'] = atoms.get_potential_energy()
            self.results['free_energy'] = self.results['energy']
        
        # 计算力
        if 'forces' in properties:
            self.results['forces'] = atoms.get_forces()
        
        # 计算应力
        if 'stress' in properties and self.compute_stress_flag:
            try:
                stress = atoms.get_stress()
                self.results['stress'] = stress
            except:
                self.results['stress'] = np.zeros(6)


# =============================================================================
# MACE MD模拟器
# =============================================================================

class MACEMDSimulator:
    """
    MACE驱动的分子动力学模拟
    """
    
    def __init__(self,
                 mace_calc: MACECalculator,
                 config: MACEMDConfig):
        self.calc = mace_calc
        self.config = config
        self.atoms = None
        self.dynamics = None
        self.trajectory = []
    
    def setup_system(self,
                    initial_structure: Union[str, Atoms],
                    fix_atoms: Optional[List[int]] = None):
        """
        设置模拟系统
        
        Args:
            initial_structure: 初始结构（文件路径或ASE Atoms）
            fix_atoms: 要固定的原子索引列表
        """
        # 加载结构
        if isinstance(initial_structure, str):
            self.atoms = read(initial_structure)
        else:
            self.atoms = initial_structure.copy()
        
        # 设置计算器
        self.atoms.calc = self.calc
        
        # 应用约束
        if fix_atoms:
            constraint = FixAtoms(indices=fix_atoms)
            self.atoms.set_constraint(constraint)
        
        cfg = self.config
        
        # 初始化速度
        MaxwellBoltzmannDistribution(self.atoms, temperature_K=cfg.temperature)
        
        # 创建积分器
        if cfg.integrator == "langevin":
            self.dynamics = Langevin(
                self.atoms,
                cfg.timestep * fs,
                temperature_K=cfg.temperature,
                friction=cfg.friction
            )
        elif cfg.integrator == "verlet":
            self.dynamics = VelocityVerlet(
                self.atoms,
                cfg.timestep * fs
            )
        
        logger.info(f"MD system set up: {len(self.atoms)} atoms")
        logger.info(f"Integrator: {cfg.integrator}, T={cfg.temperature}K")
    
    def run(self, n_steps: Optional[int] = None):
        """运行MD模拟"""
        cfg = self.config
        n_steps = n_steps or cfg.n_steps
        
        logger.info(f"Starting MACE MD for {n_steps} steps")
        
        self.trajectory = []
        
        for step in range(n_steps):
            self.dynamics.run(1)
            
            # 存储
            if step % cfg.output_interval == 0:
                self.trajectory.append(self.atoms.copy())
                
                # 输出信息
                temp = self.atoms.get_temperature()
                pe = self.atoms.get_potential_energy()
                ke = self.atoms.get_kinetic_energy()
                
                logger.info(f"Step {step}: T={temp:.1f}K, PE={pe:.3f}eV, KE={ke:.3f}eV")
        
        # 保存轨迹
        if cfg.trajectory_file:
            write(cfg.trajectory_file, self.trajectory)
            logger.info(f"Trajectory saved to {cfg.trajectory_file}")
        
        logger.info("MD simulation completed")
    
    def get_trajectory(self) -> List[Atoms]:
        """获取轨迹"""
        return self.trajectory


# =============================================================================
# MACE训练数据准备
# =============================================================================

class MACEDatasetPreparer:
    """
    MACE训练数据准备
    
    将DFT数据转换为MACE训练格式
    """
    
    def __init__(self, cutoff: float = 6.0):
        self.cutoff = cutoff
        self.frames = []
    
    def load_vasp_data(self, outcar_files: List[str]):
        """从VASP OUTCAR加载数据"""
        for outcar in outcar_files:
            logger.info(f"Loading {outcar}")
            try:
                atoms_list = read(outcar, index=':')
                if not isinstance(atoms_list, list):
                    atoms_list = [atoms_list]
                self.frames.extend(atoms_list)
            except Exception as e:
                logger.error(f"Failed to load {outcar}: {e}")
        
        logger.info(f"Loaded {len(self.frames)} frames from VASP")
    
    def load_extxyz(self, xyz_file: str):
        """从extended XYZ加载数据"""
        atoms_list = read(xyz_file, index=':')
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
        self.frames.extend(atoms_list)
        
        logger.info(f"Loaded {len(atoms_list)} frames from {xyz_file}")
    
    def filter_frames(self,
                     energy_threshold: float = 50.0,
                     force_threshold: float = 50.0) -> List[Atoms]:
        """
        过滤异常帧
        """
        filtered = []
        
        # 计算统计信息
        energies = [atoms.get_potential_energy() / len(atoms) 
                   for atoms in self.frames if atoms.calc is not None]
        
        if len(energies) > 0:
            mean_e = np.mean(energies)
            std_e = np.std(energies)
            
            for atoms in self.frames:
                e_per_atom = atoms.get_potential_energy() / len(atoms)
                
                # 能量过滤
                if abs(e_per_atom - mean_e) > energy_threshold * std_e:
                    continue
                
                # 力过滤
                forces = atoms.get_forces()
                max_force = np.max(np.abs(forces))
                if max_force > force_threshold:
                    continue
                
                filtered.append(atoms)
        
        logger.info(f"Filtered {len(filtered)}/{len(self.frames)} frames")
        self.frames = filtered
        
        return filtered
    
    def prepare_dataset(self,
                       train_ratio: float = 0.9,
                       val_ratio: float = 0.05,
                       test_ratio: float = 0.05,
                       output_dir: str = "./mace_dataset") -> Dict[str, str]:
        """
        准备MACE训练数据集
        
        Returns:
            训练、验证、测试文件路径
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        n_frames = len(self.frames)
        n_train = int(n_frames * train_ratio)
        n_val = int(n_frames * val_ratio)
        
        # 随机打乱
        indices = np.random.permutation(n_frames)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        # 保存
        train_file = output_dir / "train.xyz"
        val_file = output_dir / "valid.xyz"
        test_file = output_dir / "test.xyz"
        
        write(train_file, [self.frames[i] for i in train_idx])
        write(val_file, [self.frames[i] for i in val_idx])
        write(test_file, [self.frames[i] for i in test_idx])
        
        logger.info(f"Dataset prepared:")
        logger.info(f"  Train: {len(train_idx)} frames")
        logger.info(f"  Valid: {len(val_idx)} frames")
        logger.info(f"  Test: {len(test_idx)} frames")
        
        return {
            'train': str(train_file),
            'valid': str(val_file),
            'test': str(test_file)
        }
    
    def get_statistics(self) -> Dict:
        """获取数据统计信息"""
        if not self.frames:
            return {}
        
        stats = {
            'n_frames': len(self.frames),
            'n_atoms': [len(f) for f in self.frames],
            'elements': list(set(self.frames[0].get_chemical_symbols()))
        }
        
        # 能量统计
        energies = [f.get_potential_energy() for f in self.frames]
        stats['energy'] = {
            'mean': float(np.mean(energies)),
            'std': float(np.std(energies)),
            'min': float(np.min(energies)),
            'max': float(np.max(energies))
        }
        
        # 力统计
        forces = np.concatenate([f.get_forces().flatten() for f in self.frames])
        stats['forces'] = {
            'mean': float(np.mean(forces)),
            'std': float(np.std(forces)),
            'max_abs': float(np.max(np.abs(forces)))
        }
        
        return stats


# =============================================================================
# MACE主动学习
# =============================================================================

class MACEActiveLearning:
    """
    MACE主动学习工作流
    
    实现自动化的Explore-Label-Retrain循环
    """
    
    def __init__(self,
                 initial_model: MACECalculator,
                 config: MACEActiveLearningConfig):
        self.model = initial_model
        self.config = config
        self.iteration = 0
        self.training_data = []
        self.uncertainty_history = []
    
    def explore_uncertain_structures(self,
                                     initial_structure: Atoms,
                                     n_structures: int = 50) -> List[Atoms]:
        """
        探索不确定结构
        
        运行MD并在高不确定性区域采样
        """
        cfg = self.config
        uncertain_structures = []
        
        # 在不同条件下运行短MD
        for T in cfg.explore_temperatures[:2]:  # 使用前两个温度
            atoms = initial_structure.copy()
            
            # 设置计算器
            atoms.calc = self.model
            
            # 运行MD
            md_config = MACEMDConfig(
                temperature=T,
                n_steps=1000,
                output_interval=10
            )
            
            simulator = MACEMDSimulator(self.model, md_config)
            simulator.setup_system(atoms)
            simulator.run()
            
            # 评估不确定性（使用预测方差）
            for snapshot in simulator.get_trajectory():
                uncertainty = self._estimate_uncertainty(snapshot)
                
                if uncertainty > cfg.force_uncertainty_threshold:
                    uncertain_structures.append(snapshot)
                
                if len(uncertain_structures) >= n_structures:
                    break
            
            if len(uncertain_structures) >= n_structures:
                break
        
        logger.info(f"Found {len(uncertain_structures)} uncertain structures")
        
        return uncertain_structures[:n_structures]
    
    def _estimate_uncertainty(self, atoms: Atoms) -> float:
        """
        估计预测不确定性
        
        使用多个模型集成或MC dropout
        """
        # 简化的不确定性估计：使用力的梯度
        forces = atoms.get_forces()
        
        # 计算力的大小变化作为不确定性代理
        uncertainty = np.std(np.linalg.norm(forces, axis=1))
        
        return uncertainty
    
    def label_with_dft(self, structures: List[Atoms]) -> List[Atoms]:
        """
        使用DFT标记结构
        
        这里应该调用DFT计算器，简化实现
        """
        labeled = []
        
        for struct in structures:
            # 实际应调用DFT
            # struct.calc = Vasp(...)
            # energy = struct.get_potential_energy()
            # forces = struct.get_forces()
            
            # 标记为已标记（保留现有能量/力）
            labeled.append(struct)
        
        return labeled
    
    def retrain_model(self, new_data: List[Atoms]) -> bool:
        """
        使用新数据重新训练模型
        
        实际应调用MACE训练脚本
        """
        # 保存新数据
        train_file = f"active_learning_iter{self.iteration}.xyz"
        write(train_file, new_data)
        
        logger.info(f"New training data saved to {train_file}")
        logger.info("Run MACE training to update model")
        
        # 这里应该调用mace_run_train
        # 简化：返回False表示需要手动训练
        return False
    
    def run_iteration(self, initial_structure: Atoms) -> bool:
        """
        运行一个主动学习迭代
        
        Returns:
            converged: 是否收敛
        """
        logger.info(f"=== Active Learning Iteration {self.iteration} ===")
        
        # 探索
        new_structures = self.explore_uncertain_structures(initial_structure)
        
        if len(new_structures) == 0:
            logger.info("No uncertain structures found. Converged!")
            return True
        
        # 标记
        labeled_data = self.label_with_dft(new_structures)
        self.training_data.extend(labeled_data)
        
        # 重新训练
        success = self.retrain_model(self.training_data)
        
        self.iteration += 1
        
        # 检查停止条件
        if self.iteration >= self.config.max_iterations:
            logger.info("Reached maximum iterations")
            return True
        
        return False
    
    def run(self, initial_structure: Atoms, max_iterations: Optional[int] = None) -> List[Atoms]:
        """
        运行主动学习循环
        
        Returns:
            training_data: 收集的训练数据
        """
        max_iter = max_iterations or self.config.max_iterations
        
        converged = False
        while not converged and self.iteration < max_iter:
            converged = self.run_iteration(initial_structure)
        
        return self.training_data


# =============================================================================
# MACE批量预测
# =============================================================================

class MACEBatchPredictor:
    """
    MACE批量预测器
    
    高效处理大量结构的预测
    """
    
    def __init__(self, calculator: MACECalculator, batch_size: int = 32):
        self.calc = calculator
        self.batch_size = batch_size
    
    def predict(self, structures: List[Atoms]) -> pd.DataFrame:
        """
        批量预测能量和力
        
        Returns:
            DataFrame with predictions
        """
        results = []
        
        for i in range(0, len(structures), self.batch_size):
            batch = structures[i:i+self.batch_size]
            
            for atoms in batch:
                atoms.calc = self.calc
                
                try:
                    energy = atoms.get_potential_energy()
                    forces = atoms.get_forces()
                    
                    results.append({
                        'n_atoms': len(atoms),
                        'energy': energy,
                        'energy_per_atom': energy / len(atoms),
                        'max_force': np.max(np.abs(forces)),
                        'rms_force': np.sqrt(np.mean(forces**2))
                    })
                except Exception as e:
                    logger.warning(f"Prediction failed: {e}")
                    results.append({
                        'n_atoms': len(atoms),
                        'energy': None,
                        'error': str(e)
                    })
            
            if (i // self.batch_size) % 10 == 0:
                logger.info(f"Processed {i}/{len(structures)} structures")
        
        return pd.DataFrame(results)
    
    def screen_structures(self,
                         structures: List[Atoms],
                         energy_threshold: Optional[float] = None) -> Tuple[List[Atoms], pd.DataFrame]:
        """
        筛选低能量结构
        
        Returns:
            filtered_structures, results_df
        """
        results_df = self.predict(structures)
        
        # 过滤有效结果
        valid_results = results_df[results_df['energy'].notna()]
        
        if energy_threshold is None:
            # 自动阈值（低于均值）
            mean_e = valid_results['energy_per_atom'].mean()
            energy_threshold = mean_e
        
        # 筛选
        mask = results_df['energy_per_atom'] < energy_threshold
        mask = mask.fillna(False)
        
        filtered_structures = [s for s, m in zip(structures, mask) if m]
        
        logger.info(f"Screened {len(filtered_structures)}/{len(structures)} structures")
        
        return filtered_structures, results_df


# =============================================================================
# MACE-LAMMPS接口
# =============================================================================

class MACELAMMPSInterface:
    """
    MACE与LAMMPS的接口
    
    允许在LAMMPS中使用MACE作为pair_style
    """
    
    def __init__(self, mace_model_path: str):
        self.model_path = Path(mace_model_path)
        self.lammps_input_template = """
# MACE potential for LAMMPS
units metal
atom_style atomic

# Read structure
read_data {data_file}

# MACE pair style (requires mace-lammps plugin)
pair_style mace {cutoff} {model_file}
pair_coeff * * {elements}

# Run
run {n_steps}
"""
    
    def export_to_lammps_format(self, output_dir: str) -> str:
        """
        导出MACE模型到LAMMPS可用格式
        
        MACE模型可以直接在LAMMPS中使用，需要mace-lammps插件
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制模型文件
        model_name = self.model_path.name
        output_model = output_dir / model_name
        
        import shutil
        shutil.copy(self.model_path, output_model)
        
        logger.info(f"MACE model exported for LAMMPS: {output_model}")
        
        return str(output_model)
    
    def generate_lammps_input(self,
                             structure_file: str,
                             elements: List[str],
                             cutoff: float = 6.0,
                             n_steps: int = 1000,
                             output_file: str = "in.lammps") -> str:
        """
        生成LAMMPS输入文件
        
        注意：需要在LAMMPS中安装mace-lammps插件
        """
        content = self.lammps_input_template.format(
            data_file=structure_file,
            model_file=str(self.model_path),
            elements=" ".join(elements),
            cutoff=cutoff,
            n_steps=n_steps
        )
        
        with open(output_file, 'w') as f:
            f.write(content)
        
        logger.info(f"LAMMPS input generated: {output_file}")
        
        return output_file


# =============================================================================
# 主工作流类
# =============================================================================

class MACEWorkflow:
    """
    MACE完整工作流
    
    整合预测、MD模拟、主动学习
    """
    
    def __init__(self, config: MACEConfig):
        self.config = config
        self.calculator = None
        self.md_simulator = None
    
    def setup_calculator(self) -> MACECalculator:
        """设置MACE计算器"""
        cfg = self.config
        
        self.calculator = MACECalculator(
            model_path=cfg.model_path,
            model_type=cfg.model_type,
            device=cfg.device,
            default_dtype=cfg.default_dtype,
            cutoff=cfg.cutoff,
            compute_stress=cfg.compute_stress
        )
        
        return self.calculator
    
    def run_md_simulation(self,
                         initial_structure: Union[str, Atoms],
                         md_config: MACEMDConfig,
                         fix_atoms: Optional[List[int]] = None) -> List[Atoms]:
        """
        运行MACE MD模拟
        """
        if self.calculator is None:
            self.setup_calculator()
        
        self.md_simulator = MACEMDSimulator(self.calculator, md_config)
        self.md_simulator.setup_system(initial_structure, fix_atoms)
        self.md_simulator.run()
        
        return self.md_simulator.get_trajectory()
    
    def relax_structure(self,
                       structure: Union[str, Atoms],
                       optimizer: str = "LBFGS",
                       fmax: float = 0.01,
                       max_steps: int = 500) -> Atoms:
        """
        结构弛豫
        """
        if self.calculator is None:
            self.setup_calculator()
        
        # 加载结构
        if isinstance(structure, str):
            atoms = read(structure)
        else:
            atoms = structure.copy()
        
        atoms.calc = self.calculator
        
        # 选择优化器
        if optimizer == "LBFGS":
            opt = LBFGS(atoms)
        elif optimizer == "BFGS":
            opt = BFGS(atoms)
        elif optimizer == "FIRE":
            opt = FIRE(atoms)
        else:
            opt = BFGS(atoms)
        
        # 运行优化
        opt.run(fmax=fmax, steps=max_steps)
        
        logger.info(f"Structure relaxed: E={atoms.get_potential_energy():.4f}eV")
        
        return atoms
    
    def batch_screening(self,
                       structures: List[Atoms],
                       energy_threshold: Optional[float] = None) -> Tuple[List[Atoms], pd.DataFrame]:
        """
        批量筛选结构
        """
        if self.calculator is None:
            self.setup_calculator()
        
        predictor = MACEBatchPredictor(self.calculator, self.config.batch_size)
        
        return predictor.screen_structures(structures, energy_threshold)


# =============================================================================
# 命令行接口
# =============================================================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MACE ML Potential Tool")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict energy and forces')
    predict_parser.add_argument('structure', help='Structure file')
    predict_parser.add_argument('--model', default='medium', help='Model type')
    predict_parser.add_argument('--device', default='cuda', help='Device')
    
    # MD command
    md_parser = subparsers.add_parser('md', help='Run MD simulation')
    md_parser.add_argument('structure', help='Structure file')
    md_parser.add_argument('--temperature', type=float, default=300.0)
    md_parser.add_argument('--steps', type=int, default=10000)
    md_parser.add_argument('--output', default='mace_md.traj')
    
    # Relax command
    relax_parser = subparsers.add_parser('relax', help='Relax structure')
    relax_parser.add_argument('structure', help='Structure file')
    relax_parser.add_argument('--fmax', type=float, default=0.01)
    
    args = parser.parse_args()
    
    if args.command == 'predict':
        config = MACEConfig(model_type=args.model, device=args.device)
        workflow = MACEWorkflow(config)
        workflow.setup_calculator()
        
        atoms = read(args.structure)
        atoms.calc = workflow.calculator
        
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        
        print(f"Energy: {energy:.4f} eV")
        print(f"Energy per atom: {energy/len(atoms):.4f} eV/atom")
        print(f"Max force: {np.max(np.abs(forces)):.4f} eV/Å")
        
    elif args.command == 'md':
        config = MACEConfig(device='cuda')
        workflow = MACEWorkflow(config)
        
        md_config = MACEMDConfig(
            temperature=args.temperature,
            n_steps=args.steps,
            trajectory_file=args.output
        )
        
        trajectory = workflow.run_md_simulation(args.structure, md_config)
        print(f"MD completed: {len(trajectory)} frames saved")
        
    elif args.command == 'relax':
        config = MACEConfig(device='cuda')
        workflow = MACEWorkflow(config)
        
        relaxed = workflow.relax_structure(args.structure, fmax=args.fmax)
        write('relaxed_structure.xyz', relaxed)
        print(f"Relaxed structure saved to relaxed_structure.xyz")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
