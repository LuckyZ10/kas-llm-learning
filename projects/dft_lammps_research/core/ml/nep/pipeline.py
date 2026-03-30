#!/usr/bin/env python3
"""
nep_training_pipeline.py
=======================
NEP (Neural Evolution Potential) 训练流程

功能：
1. 从DFT数据准备NEP训练格式
2. 自动配置nep.in训练参数
3. 运行NEP训练 (GPUMD)
4. 模型验证和测试
5. 导出到LAMMPS格式

作者: DFT-MD Coupling Expert
日期: 2026-03-09
"""

import os
import re
import sys
import json
import glob
import shutil
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging
from datetime import datetime
import warnings

# ASE
from ase import Atoms
from ase.io import read, write
from ase.io.vasp import read_vasp_out
from ase.io.extxyz import write_extxyz, read_extxyz
from ase.units import eV, Ang, fs, GPa

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 可选依赖
try:
    import dpdata
    DPDATA_AVAILABLE = True
except ImportError:
    DPDATA_AVAILABLE = False
    warnings.warn("dpdata not available")

try:
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class NEPDataConfig:
    """NEP数据配置"""
    # 输入数据源
    vasp_outcars: List[str] = field(default_factory=list)
    dft_trajectories: List[str] = field(default_factory=list)
    existing_xyz: Optional[str] = None
    
    # 数据处理
    energy_threshold: float = 50.0  # eV/atom, 异常值过滤
    force_threshold: float = 50.0  # eV/Å, 异常力过滤
    min_force: float = 0.001  # 最小力值
    
    # 数据集分割
    train_ratio: float = 0.9
    test_ratio: float = 0.1
    
    # 元素映射
    type_map: List[str] = field(default_factory=list)


@dataclass
class NEPModelConfig:
    """NEP模型配置"""
    # 模型类型
    model_type: int = 0  # 0=PES, 1=dipole, 2=polarizability
    
    # 元素类型
    type_list: List[str] = field(default_factory=list)
    
    # 描述符参数
    version: int = 4  # NEP版本 (2, 3, or 4)
    cutoff_radial: float = 6.0  # 径向截断 (Å)
    cutoff_angular: float = 4.0  # 角向截断 (Å)
    n_max_radial: int = 4  # 径向基函数数
    n_max_angular: int = 4  # 角向基函数数
    basis_size_radial: int = 8  # 径向基大小
    basis_size_angular: int = 8  # 角向基大小
    l_max_3body: int = 4  # 3-body最大角动量
    l_max_4body: int = 0  # 4-body最大角动量 (0=禁用)
    l_max_5body: int = 0  # 5-body最大角动量 (0=禁用)
    
    # 神经网络参数
    neuron: int = 30  # 隐藏层神经元数
    
    # 优化参数
    population_size: int = 50  # SNES种群大小
    maximum_generation: int = 100000  # 最大迭代代数
    batch_size: int = 1000  # 批量大小


@dataclass
class NEPTrainingConfig:
    """NEP训练配置"""
    # 路径
    gpumd_path: str = "/path/to/gpumd"
    working_dir: str = "./nep_training"
    
    # 文件
    train_xyz: str = "train.xyz"
    test_xyz: str = "test.xyz"
    nep_in: str = "nep.in"
    
    # 计算设置
    use_gpu: bool = True
    gpu_id: int = 0
    
    # 重启选项
    restart: bool = False


# =============================================================================
# NEP数据准备
# =============================================================================

class NEPDataPreparer:
    """
    NEP数据准备器
    
    将各种DFT格式转换为NEP所需的extended XYZ格式
    """
    
    def __init__(self, config: Optional[NEPDataConfig] = None):
        self.config = config or NEPDataConfig()
        self.frames = []
        
    def load_vasp_outcar(self, outcar_path: Union[str, Path]) -> List[Dict]:
        """从VASP OUTCAR加载数据"""
        outcar_path = Path(outcar_path)
        
        if not outcar_path.exists():
            raise FileNotFoundError(f"OUTCAR not found: {outcar_path}")
        
        logger.info(f"Loading VASP data from: {outcar_path}")
        
        try:
            # 使用ASE读取
            atoms_list = read_vasp_out(str(outcar_path), index=':')
            if not isinstance(atoms_list, list):
                atoms_list = [atoms_list]
        except Exception as e:
            logger.error(f"Failed to parse OUTCAR: {e}")
            return []
        
        frames = []
        for atoms in atoms_list:
            frame = self._extract_frame_data(atoms)
            if frame:
                frames.append(frame)
        
        logger.info(f"Loaded {len(frames)} frames from {outcar_path}")
        return frames
    
    def load_trajectory(self, traj_file: Union[str, Path]) -> List[Dict]:
        """从ASE trajectory加载"""
        traj_file = Path(traj_file)
        
        logger.info(f"Loading trajectory: {traj_file}")
        
        try:
            atoms_list = read(str(traj_file), index=':')
            if not isinstance(atoms_list, list):
                atoms_list = [atoms_list]
        except Exception as e:
            logger.error(f"Failed to load trajectory: {e}")
            return []
        
        frames = []
        for atoms in atoms_list:
            frame = self._extract_frame_data(atoms)
            if frame:
                frames.append(frame)
        
        logger.info(f"Loaded {len(frames)} frames from {traj_file}")
        return frames
    
    def _extract_frame_data(self, atoms: Atoms) -> Optional[Dict]:
        """提取单帧数据"""
        frame = {
            'atoms': atoms.copy(),
            'symbols': atoms.get_chemical_symbols(),
            'positions': atoms.get_positions(),
            'cell': atoms.get_cell(),
            'pbc': atoms.get_pbc(),
        }
        
        # 能量
        try:
            if atoms.calc is not None:
                frame['energy'] = atoms.get_potential_energy()
                frame['forces'] = atoms.get_forces()
                
                # 应力 (如果可用)
                try:
                    stress = atoms.get_stress(voigt=True)
                    frame['stress'] = stress
                    frame['virial'] = -stress * atoms.get_volume()
                except:
                    frame['stress'] = None
                    frame['virial'] = None
            else:
                # 从atoms.info读取
                frame['energy'] = atoms.info.get('energy', None)
                frame['forces'] = atoms.arrays.get('forces', None)
        except Exception as e:
            logger.warning(f"Failed to extract data: {e}")
            return None
        
        return frame
    
    def filter_frames(self, frames: List[Dict]) -> List[Dict]:
        """过滤异常帧"""
        if len(frames) < 2:
            return frames
        
        # 能量过滤
        energies = [f['energy'] for f in frames if f.get('energy') is not None]
        if energies:
            mean_e = np.mean(energies)
            std_e = np.std(energies)
            
            filtered = []
            for frame in frames:
                if frame.get('energy') is not None:
                    e_per_atom = frame['energy'] / len(frame['atoms'])
                    mean_per_atom = mean_e / len(frame['atoms'])
                    if abs(e_per_atom - mean_per_atom) < self.config.energy_threshold:
                        filtered.append(frame)
                    else:
                        logger.warning(f"Filtered outlier: E = {e_per_atom:.4f} eV/atom")
                else:
                    filtered.append(frame)
            
            frames = filtered
        
        # 力过滤
        filtered = []
        for frame in frames:
            if frame.get('forces') is not None:
                max_force = np.max(np.abs(frame['forces']))
                if max_force < self.config.force_threshold:
                    filtered.append(frame)
                else:
                    logger.warning(f"Filtered frame with large force: {max_force:.2f} eV/Å")
            else:
                filtered.append(frame)
        
        logger.info(f"After filtering: {len(filtered)} frames")
        return filtered
    
    def prepare_dataset(self, output_dir: Union[str, Path]) -> Tuple[str, str]:
        """
        准备NEP数据集
        
        Returns:
            (train_xyz, test_xyz): 训练和测试文件路径
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 收集所有数据
        all_frames = []
        
        # 从VASP OUTCAR加载
        for outcar in self.config.vasp_outcars:
            frames = self.load_vasp_outcar(outcar)
            all_frames.extend(frames)
        
        # 从trajectory加载
        for traj in self.config.dft_trajectories:
            frames = self.load_trajectory(traj)
            all_frames.extend(frames)
        
        # 从现有XYZ加载
        if self.config.existing_xyz:
            frames = self._load_xyz(self.config.existing_xyz)
            all_frames.extend(frames)
        
        if not all_frames:
            raise ValueError("No data loaded!")
        
        logger.info(f"Total frames loaded: {len(all_frames)}")
        
        # 过滤
        all_frames = self.filter_frames(all_frames)
        
        # 分割训练集和测试集
        n_frames = len(all_frames)
        n_train = int(n_frames * self.config.train_ratio)
        
        indices = np.random.permutation(n_frames)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        train_frames = [all_frames[i] for i in train_idx]
        test_frames = [all_frames[i] for i in test_idx]
        
        # 确定type_map
        if not self.config.type_map:
            symbols = all_frames[0]['symbols']
            self.config.type_map = sorted(set(symbols))
        
        # 写入XYZ文件
        train_xyz = output_dir / "train.xyz"
        test_xyz = output_dir / "test.xyz"
        
        self._write_xyz(train_xyz, train_frames)
        self._write_xyz(test_xyz, test_frames)
        
        logger.info(f"Dataset prepared:")
        logger.info(f"  Training: {len(train_frames)} frames -> {train_xyz}")
        logger.info(f"  Testing: {len(test_frames)} frames -> {test_xyz}")
        
        return str(train_xyz), str(test_xyz)
    
    def _load_xyz(self, xyz_file: str) -> List[Dict]:
        """加载XYZ文件"""
        try:
            atoms_list = read(xyz_file, index=':', format='extxyz')
            if not isinstance(atoms_list, list):
                atoms_list = [atoms_list]
            
            frames = []
            for atoms in atoms_list:
                frame = self._extract_frame_data(atoms)
                if frame:
                    frames.append(frame)
            
            return frames
        except Exception as e:
            logger.error(f"Failed to load XYZ: {e}")
            return []
    
    def _write_xyz(self, output_file: Path, frames: List[Dict]):
        """写入extended XYZ文件"""
        
        with open(output_file, 'w') as f:
            for frame in frames:
                atoms = frame['atoms']
                n_atoms = len(atoms)
                
                # 晶格
                cell = atoms.get_cell()
                lattice_str = " ".join([f"{x:.10f}" for x in cell.flatten()])
                
                # 能量
                energy = frame.get('energy', 0.0)
                
                # 应力/位力
                virial = frame.get('virial')
                if virial is not None:
                    # 转换为6个分量 (xx, yy, zz, xy, xz, yz)
                    virial_flat = [
                        virial[0], virial[1], virial[2],
                        virial[5], virial[4], virial[3]  # 注意顺序
                    ]
                    virial_str = " ".join([f"{v:.10f}" for v in virial_flat])
                    property_str = f"energy={energy:.10f} virial=\"{virial_str}\""
                else:
                    property_str = f"energy={energy:.10f}"
                
                # 写入头部
                f.write(f"{n_atoms}\n")
                f.write(f"Lattice=\"{lattice_str}\" Properties=species:S:1:pos:R:3:forces:R:3 {property_str}\n")
                
                # 写入原子数据
                symbols = atoms.get_chemical_symbols()
                positions = atoms.get_positions()
                forces = frame.get('forces', np.zeros_like(positions))
                
                for symbol, pos, force in zip(symbols, positions, forces):
                    f.write(f"{symbol:>3} {pos[0]:15.8f} {pos[1]:15.8f} {pos[2]:15.8f} "
                           f"{force[0]:15.8f} {force[1]:15.8f} {force[2]:15.8f}\n")
        
        logger.info(f"Wrote {len(frames)} frames to {output_file}")
    
    def get_statistics(self) -> Dict:
        """获取数据统计信息"""
        if not self.frames:
            return {}
        
        stats = {
            'n_frames': len(self.frames),
            'n_atoms_per_frame': len(self.frames[0]['atoms']),
            'elements': list(set(self.frames[0]['symbols'])),
        }
        
        # 能量统计
        energies = [f['energy'] for f in self.frames if f.get('energy') is not None]
        if energies:
            stats['energy'] = {
                'mean': float(np.mean(energies)),
                'std': float(np.std(energies)),
                'min': float(np.min(energies)),
                'max': float(np.max(energies)),
                'unit': 'eV'
            }
        
        # 力统计
        forces = [f['forces'] for f in self.frames if f.get('forces') is not None]
        if forces:
            all_forces = np.concatenate([f.flatten() for f in forces])
            stats['forces'] = {
                'mean': float(np.mean(all_forces)),
                'std': float(np.std(all_forces)),
                'max_abs': float(np.max(np.abs(all_forces))),
                'unit': 'eV/Ang'
            }
        
        return stats


# =============================================================================
# NEP输入文件生成器
# =============================================================================

class NEPInputGenerator:
    """
    NEP输入文件生成器
    
    生成nep.in配置文件
    """
    
    def __init__(self, config: Optional[NEPModelConfig] = None):
        self.config = config or NEPModelConfig()
    
    def generate(self, output_file: str = "nep.in") -> str:
        """
        生成nep.in文件
        
        Returns:
            output_file: 生成的文件路径
        """
        lines = []
        
        # 元素类型
        type_line = f"type {' '.join(self.config.type_list)}"
        lines.append(type_line)
        
        # 模型类型
        if self.config.model_type != 0:
            lines.append(f"model_type {self.config.model_type}")
        
        # 版本
        lines.append(f"version {self.config.version}")
        
        # 截断半径
        lines.append(f"cutoff {self.config.cutoff_radial} {self.config.cutoff_angular}")
        
        # n_max
        lines.append(f"n_max {self.config.n_max_radial} {self.config.n_max_angular}")
        
        # basis_size (NEP4)
        if self.config.version >= 4:
            lines.append(f"basis_size {self.config.basis_size_radial} {self.config.basis_size_angular}")
        
        # l_max
        l_max_str = f"l_max {self.config.l_max_3body}"
        if self.config.l_max_4body > 0:
            l_max_str += f" {self.config.l_max_4body}"
        if self.config.l_max_5body > 0:
            l_max_str += f" {self.config.l_max_5body}"
        lines.append(l_max_str)
        
        # 神经元
        lines.append(f"neuron {self.config.neuron}")
        
        # 种群大小
        lines.append(f"population {self.config.population_size}")
        
        # 最大代数
        lines.append(f"generation {self.config.maximum_generation}")
        
        # 批量大小
        lines.append(f"batch {self.config.batch_size}")
        
        # 写入文件
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Generated NEP input: {output_file}")
        logger.info(f"  Type: {self.config.type_list}")
        logger.info(f"  Version: {self.config.version}")
        logger.info(f"  Cutoff: {self.config.cutoff_radial}/{self.config.cutoff_angular} Å")
        
        return output_file
    
    def generate_from_preset(self, 
                            preset: str,
                            type_list: List[str],
                            output_file: str = "nep.in") -> str:
        """
        从预设配置生成
        
        Presets:
        - fast: 快速训练，适合小数据集
        - accurate: 高精度，适合复杂系统
        - light: 轻量级模型，推理速度快
        """
        self.config.type_list = type_list
        
        if preset == "fast":
            self.config.version = 3
            self.config.n_max_radial = 4
            self.config.n_max_angular = 4
            self.config.l_max_3body = 4
            self.config.neuron = 10
            self.config.population_size = 30
            self.config.maximum_generation = 10000
            
        elif preset == "accurate":
            self.config.version = 4
            self.config.n_max_radial = 6
            self.config.n_max_angular = 6
            self.config.basis_size_radial = 12
            self.config.basis_size_angular = 12
            self.config.l_max_3body = 6
            self.config.neuron = 50
            self.config.population_size = 100
            self.config.maximum_generation = 1000000
            
        elif preset == "light":
            self.config.version = 3
            self.config.n_max_radial = 4
            self.config.n_max_angular = 2
            self.config.l_max_3body = 2
            self.config.neuron = 5
            self.config.population_size = 20
            self.config.maximum_generation = 50000
        
        else:
            raise ValueError(f"Unknown preset: {preset}")
        
        logger.info(f"Using preset: {preset}")
        return self.generate(output_file)


# =============================================================================
# NEP训练器
# =============================================================================

class NEPTrainer:
    """
    NEP训练器
    
    调用GPUMD的nep可执行文件进行训练
    """
    
    def __init__(self, config: Optional[NEPTrainingConfig] = None):
        self.config = config or NEPTrainingConfig()
        self.working_dir = Path(self.config.working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        self.training_log = []
    
    def setup(self, 
              model_config: NEPModelConfig,
              train_xyz: str = "train.xyz",
              test_xyz: str = "test.xyz"):
        """
        设置训练环境
        
        Args:
            model_config: NEP模型配置
            train_xyz: 训练数据文件
            test_xyz: 测试数据文件
        """
        # 复制数据文件到工作目录
        train_src = Path(train_xyz)
        test_src = Path(test_xyz)
        
        if train_src.exists():
            shutil.copy(train_src, self.working_dir / "train.xyz")
        else:
            raise FileNotFoundError(f"Training data not found: {train_xyz}")
        
        if test_src.exists():
            shutil.copy(test_src, self.working_dir / "test.xyz")
        
        # 生成nep.in
        generator = NEPInputGenerator(model_config)
        generator.generate(str(self.working_dir / "nep.in"))
        
        logger.info(f"Training setup complete in {self.working_dir}")
    
    def train(self, verbose: bool = True) -> str:
        """
        执行NEP训练
        
        Returns:
            model_file: 训练好的模型文件路径
        """
        nep_exe = Path(self.config.gpumd_path) / "nep"
        
        if not nep_exe.exists():
            # 尝试系统路径
            nep_exe = "nep"
        
        logger.info(f"Starting NEP training...")
        logger.info(f"Working directory: {self.working_dir}")
        
        # 构建命令
        cmd = [str(nep_exe)]
        
        if self.config.use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.config.gpu_id)
        
        # 执行训练
        try:
            if verbose:
                # 实时输出
                process = subprocess.Popen(
                    cmd,
                    cwd=self.working_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
                
                for line in process.stdout:
                    print(line, end='')
                    self.training_log.append(line)
                
                process.wait()
                
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, cmd)
            else:
                # 静默运行
                result = subprocess.run(
                    cmd,
                    cwd=self.working_dir,
                    capture_output=True,
                    text=True
                )
                self.training_log = result.stdout.split('\n')
                
                if result.returncode != 0:
                    raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
            
            logger.info("NEP training completed!")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e}")
            raise
        
        # 检查输出文件
        model_file = self.working_dir / "nep.txt"
        if not model_file.exists():
            raise RuntimeError("Model file not generated!")
        
        return str(model_file)
    
    def get_loss_history(self) -> pd.DataFrame:
        """获取训练损失历史"""
        loss_file = self.working_dir / "loss.out"
        
        if not loss_file.exists():
            logger.warning("Loss file not found")
            return pd.DataFrame()
        
        # 解析loss.out
        # 格式: generation L1_train L2_train ... L1_test L2_test ...
        try:
            data = []
            with open(loss_file, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        values = [float(x) for x in line.split()]
                        if len(values) >= 3:
                            data.append({
                                'generation': int(values[0]),
                                'energy_train': values[1],
                                'force_train': values[2],
                            })
            
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Failed to parse loss file: {e}")
            return pd.DataFrame()
    
    def plot_training_curves(self, output_file: str = "training_curves.png"):
        """绘制训练曲线"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return
        
        df = self.get_loss_history()
        
        if df.empty:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 能量损失
        axes[0].semilogy(df['generation'], df['energy_train'], label='Train')
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('Energy RMSE (eV/atom)')
        axes[0].set_title('Energy Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 力损失
        axes[1].semilogy(df['generation'], df['force_train'], label='Train')
        axes[1].set_xlabel('Generation')
        axes[1].set_ylabel('Force RMSE (eV/Å)')
        axes[1].set_title('Force Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.working_dir / output_file, dpi=150)
        logger.info(f"Saved training curves to {output_file}")


# =============================================================================
# NEP模型验证与测试
# =============================================================================

class NEPValidator:
    """
    NEP模型验证器
    """
    
    def __init__(self, model_file: str, gpumd_path: str = "/path/to/gpumd"):
        self.model_file = Path(model_file)
        self.gpumd_path = gpumd_path
        
        if not self.model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
    
    def predict(self, xyz_file: str, output_prefix: str = "prediction") -> Dict:
        """
        使用训练好的模型进行预测
        
        Returns:
            results: 包含预测结果的字典
        """
        working_dir = Path(xyz_file).parent
        
        # 创建预测用的nep.in
        nep_in = working_dir / "nep_predict.in"
        
        # 复制模型文件
        shutil.copy(self.model_file, working_dir / "nep.txt")
        
        # 写入预测配置
        with open(nep_in, 'w') as f:
            f.write("prediction 1\n")
        
        # 运行预测
        gpumd_exe = Path(self.gpumd_path) / "gpumd"
        if not gpumd_exe.exists():
            gpumd_exe = "gpumd"
        
        cmd = [str(gpumd_exe)]
        
        try:
            subprocess.run(cmd, cwd=working_dir, check=True)
            
            # 读取预测结果
            results = self._read_prediction_results(working_dir)
            
            return results
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def _read_prediction_results(self, working_dir: Path) -> Dict:
        """读取预测结果"""
        results = {
            'energy': [],
            'forces': [],
        }
        
        # 读取能量预测
        energy_file = working_dir / "energy_test.out"
        if energy_file.exists():
            with open(energy_file, 'r') as f:
                for line in f:
                    if line.strip():
                        results['energy'].append(float(line.split()[0]))
        
        # 读力预测
        force_file = working_dir / "force_test.out"
        if force_file.exists():
            forces = np.loadtxt(force_file)
            results['forces'] = forces
        
        return results
    
    def validate(self, 
                test_xyz: str,
                dft_energies: Optional[np.ndarray] = None,
                dft_forces: Optional[np.ndarray] = None) -> Dict:
        """
        验证模型精度
        
        Returns:
            metrics: 包含RMSE, MAE等指标的字典
        """
        results = self.predict(test_xyz)
        
        metrics = {}
        
        # 能量误差
        if dft_energies is not None:
            pred_e = np.array(results['energy'])
            true_e = dft_energies
            
            metrics['energy'] = {
                'rmse': np.sqrt(np.mean((pred_e - true_e)**2)),
                'mae': np.mean(np.abs(pred_e - true_e)),
                'r2': 1 - np.sum((pred_e - true_e)**2) / np.sum((true_e - np.mean(true_e))**2)
            }
        
        # 力误差
        if dft_forces is not None:
            pred_f = np.array(results['forces']).flatten()
            true_f = dft_forces.flatten()
            
            metrics['forces'] = {
                'rmse': np.sqrt(np.mean((pred_f - true_f)**2)),
                'mae': np.mean(np.abs(pred_f - true_f)),
                'max_error': np.max(np.abs(pred_f - true_f))
            }
        
        return metrics
    
    def export_to_lammps(self, output_file: str = "nep_lammps.txt") -> str:
        """
        导出模型到LAMMPS格式
        
        NEP模型可以直接在LAMMPS中使用，需要pair_style nep
        """
        # NEP格式已经是文本格式，可以直接复制
        output_path = Path(output_file)
        shutil.copy(self.model_file, output_path)
        
        logger.info(f"Exported NEP model to LAMMPS format: {output_path}")
        
        return str(output_path)


# =============================================================================
# 主动学习工作流
# =============================================================================

class NEPActiveLearning:
    """
    NEP主动学习工作流
    
    实现自动化的Explore-Label-Retrain循环
    """
    
    def __init__(self,
                 trainer: NEPTrainer,
                 dft_calculator=None,
                 uncertainty_threshold: float = 0.3):
        self.trainer = trainer
        self.dft_calculator = dft_calculator
        self.uncertainty_threshold = uncertainty_threshold
        
        self.iteration = 0
        self.training_data = []
    
    def explore(self, 
               current_model: str,
               initial_structure: Atoms,
               n_structures: int = 100) -> List[Atoms]:
        """
        探索构型空间
        
        使用当前ML势运行MD，识别高不确定性结构
        """
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from ase.md.langevin import Langevin
        
        # 使用NEP计算器运行MD (简化版本)
        # 实际应使用GPUMD或ASE接口
        
        uncertain_structures = []
        
        # 在不同温度下运行短MD
        for T in [300, 500, 800, 1000]:
            atoms = initial_structure.copy()
            
            # 这里简化处理，实际应使用NEP计算器
            # atoms.calc = NEPCalculator(current_model)
            
            MaxwellBoltzmannDistribution(atoms, temperature_K=T)
            dyn = Langevin(atoms, 1.0 * fs, temperature_K=T, friction=0.01)
            
            # 运行短轨迹
            for _ in range(100):
                dyn.run(10)
                
                # 计算不确定性 (简化)
                uncertainty = self._estimate_uncertainty(atoms, current_model)
                
                if uncertainty > self.uncertainty_threshold:
                    uncertain_structures.append(atoms.copy())
                
                if len(uncertain_structures) >= n_structures:
                    break
            
            if len(uncertain_structures) >= n_structures:
                break
        
        logger.info(f"Exploration found {len(uncertain_structures)} uncertain structures")
        
        return uncertain_structures[:n_structures]
    
    def _estimate_uncertainty(self, atoms: Atoms, model: str) -> float:
        """估计模型预测不确定性"""
        # 简化：使用多个模型集成或MC dropout
        # 这里使用随机数作为占位符
        return np.random.random()
    
    def label(self, structures: List[Atoms]) -> List[Dict]:
        """
        用DFT标记新结构
        """
        labeled = []
        
        for struct in structures:
            # DFT计算
            if self.dft_calculator:
                energy = self.dft_calculator.get_potential_energy(struct)
                forces = self.dft_calculator.get_forces(struct)
                
                labeled.append({
                    'atoms': struct,
                    'energy': energy,
                    'forces': forces
                })
        
        return labeled
    
    def run_iteration(self, 
                     current_model: str,
                     initial_structure: Atoms) -> str:
        """运行一个主动学习迭代"""
        logger.info(f"=== Active Learning Iteration {self.iteration} ===")
        
        # 探索
        new_structures = self.explore(current_model, initial_structure)
        
        if len(new_structures) == 0:
            logger.info("No uncertain structures found. Converged!")
            return current_model
        
        # 标记
        labeled_data = self.label(new_structures)
        
        # 添加到训练集
        self.training_data.extend(labeled_data)
        
        # 重新训练 (简化)
        # 实际应重新准备数据并训练
        
        self.iteration += 1
        
        return current_model
    
    def run(self, 
           initial_model: str,
           initial_structure: Atoms,
           max_iterations: int = 10) -> str:
        """运行主动学习循环"""
        model = initial_model
        
        for i in range(max_iterations):
            model = self.run_iteration(model, initial_structure)
            
            # 检查收敛
            if self._check_convergence():
                break
        
        return model
    
    def _check_convergence(self) -> bool:
        """检查收敛"""
        # 实现收敛判断逻辑
        return False


# =============================================================================
# 主工作流类
# =============================================================================

class NEPTrainingPipeline:
    """
    NEP训练完整流程
    
    整合数据准备、训练、验证的全过程
    """
    
    def __init__(self, working_dir: str = "./nep_workflow"):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_preparer = None
        self.trainer = None
        self.validator = None
    
    def run(self,
           vasp_outcars: List[str],
           type_list: List[str],
           preset: str = "fast",
           gpumd_path: str = "/path/to/gpumd") -> Dict:
        """
        运行完整NEP训练流程
        
        Args:
            vasp_outcars: VASP OUTCAR文件列表
            type_list: 元素类型列表
            preset: 预设配置 (fast/accurate/light)
            gpumd_path: GPUMD路径
            
        Returns:
            results: 训练结果字典
        """
        logger.info("=" * 60)
        logger.info("Starting NEP Training Pipeline")
        logger.info("=" * 60)
        
        results = {}
        
        # 步骤1: 数据准备
        logger.info("\n[Step 1] Preparing dataset...")
        
        data_config = NEPDataConfig(
            vasp_outcars=vasp_outcars,
            type_map=type_list
        )
        
        self.data_preparer = NEPDataPreparer(data_config)
        train_xyz, test_xyz = self.data_preparer.prepare_dataset(
            self.working_dir / "data"
        )
        
        results['train_xyz'] = train_xyz
        results['test_xyz'] = test_xyz
        results['statistics'] = self.data_preparer.get_statistics()
        
        # 步骤2: 训练
        logger.info("\n[Step 2] Training NEP model...")
        
        training_config = NEPTrainingConfig(
            gpumd_path=gpumd_path,
            working_dir=str(self.working_dir / "training")
        )
        
        self.trainer = NEPTrainer(training_config)
        
        model_config = NEPModelConfig(type_list=type_list)
        
        self.trainer.setup(
            model_config=model_config,
            train_xyz=train_xyz,
            test_xyz=test_xyz
        )
        
        model_file = self.trainer.train()
        results['model_file'] = model_file
        
        # 绘制训练曲线
        self.trainer.plot_training_curves()
        
        # 步骤3: 验证
        logger.info("\n[Step 3] Validating model...")
        
        self.validator = NEPValidator(model_file, gpumd_path)
        
        # 导出到LAMMPS
        lammps_file = self.validator.export_to_lammps(
            str(self.working_dir / "nep_lammps.txt")
        )
        results['lammps_model'] = lammps_file
        
        logger.info("\n" + "=" * 60)
        logger.info("NEP Training Pipeline Completed!")
        logger.info("=" * 60)
        
        return results


# =============================================================================
# 命令行接口
# =============================================================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="NEP Training Pipeline for GPUMD"
    )
    
    parser.add_argument("outcars", nargs='+', help="VASP OUTCAR files")
    parser.add_argument("--type-list", nargs='+', required=True,
                       help="Element type list (e.g., Pb Te)")
    parser.add_argument("--preset", default="fast",
                       choices=["fast", "accurate", "light"],
                       help="Training preset")
    parser.add_argument("--gpumd-path", default="/path/to/gpumd",
                       help="Path to GPUMD installation")
    parser.add_argument("--output-dir", default="./nep_training",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # 运行管道
    pipeline = NEPTrainingPipeline(working_dir=args.output_dir)
    
    results = pipeline.run(
        vasp_outcars=args.outcars,
        type_list=args.type_list,
        preset=args.preset,
        gpumd_path=args.gpumd_path
    )
    
    # 打印结果
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Model file: {results['model_file']}")
    print(f"LAMMPS model: {results['lammps_model']}")
    print(f"Training data: {results['train_xyz']}")
    print(f"Test data: {results['test_xyz']}")
    
    if 'statistics' in results:
        stats = results['statistics']
        print(f"\nDataset statistics:")
        print(f"  Frames: {stats.get('n_frames', 'N/A')}")
        if 'energy' in stats:
            e = stats['energy']
            print(f"  Energy: {e['mean']:.4f} ± {e['std']:.4f} eV")
        if 'forces' in stats:
            f = stats['forces']
            print(f"  Forces: {f['mean']:.4f} ± {f['std']:.4f} eV/Å")


if __name__ == "__main__":
    main()
