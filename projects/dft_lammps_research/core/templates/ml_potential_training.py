#!/usr/bin/env python3
"""
机器学习势训练工作流
支持DeePMD-kit和主动学习
"""

import os
import json
import glob
import numpy as np
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging
import shutil

# ASE
from ase import Atoms
from ase.io import read, write
from ase.io.vasp import read_vasp_out
from ase.calculators.calculator import Calculator

# Dpdata
import dpdata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DeepMDConfig:
    """DeePMD训练配置"""
    # 模型架构
    descriptor_type: str = "se_e2_a"  # se_e2_a, se_e3, dpa1, dpa2
    rcut: float = 6.0
    rcut_smth: float = 0.5
    sel: List[int] = None  # 每种类型的邻居数
    neuron: List[int] = None  # 描述符网络结构
    axis_neuron: int = 16
    fitting_neuron: List[int] = None  # 拟合网络结构
    
    # 训练参数
    start_lr: float = 0.001
    stop_lr: float = 3.51e-8
    decay_steps: int = 5000
    numb_steps: int = 1000000
    batch_size: str = "auto"
    
    # 损失函数权重
    start_pref_e: float = 0.02
    limit_pref_e: float = 1.0
    start_pref_f: float = 1000.0
    limit_pref_f: float = 1.0
    start_pref_v: float = 0.01
    limit_pref_v: float = 1.0
    
    # 系统设置
    type_map: List[str] = None
    seed: int = 1
    
    # 路径
    training_data: str = "./data/training"
    validation_data: str = "./data/validation"
    output_dir: str = "./model"
    
    def __post_init__(self):
        if self.sel is None:
            self.sel = [50, 50, 50]
        if self.neuron is None:
            self.neuron = [25, 50, 100]
        if self.fitting_neuron is None:
            self.fitting_neuron = [240, 240, 240]
        if self.type_map is None:
            self.type_map = ["H", "C", "N", "O"]  # 默认示例


class DataPreprocessor:
    """训练数据预处理"""
    
    def __init__(self, type_map: List[str]):
        self.type_map = type_map
        
    def convert_vasp_to_deepmd(self, 
                               vasp_dirs: List[str],
                               output_dir: str,
                               train_ratio: float = 0.9) -> Tuple[str, str]:
        """
        将VASP输出转换为DeePMD格式
        
        Args:
            vasp_dirs: VASP计算目录列表
            output_dir: 输出目录
            train_ratio: 训练集比例
        """
        logger.info("Converting VASP data to DeePMD format...")
        
        all_systems = []
        
        for vasp_dir in vasp_dirs:
            # 读取VASP输出
            outcar_path = Path(vasp_dir) / "OUTCAR"
            if not outcar_path.exists():
                logger.warning(f"OUTCAR not found in {vasp_dir}")
                continue
            
            try:
                # 使用dpdata读取
                system = dpdata.LabeledSystem(outcar_path, fmt='vasp/outcar')
                all_systems.append(system)
                logger.info(f"Loaded {len(system)} frames from {vasp_dir}")
            except Exception as e:
                logger.error(f"Failed to load {vasp_dir}: {e}")
        
        if not all_systems:
            raise ValueError("No valid VASP data found")
        
        # 合并所有系统
        multi_systems = dpdata.MultiSystems(*all_systems)
        
        # 分割训练集和验证集
        train_dir = Path(output_dir) / "training"
        valid_dir = Path(output_dir) / "validation"
        
        train_dir.mkdir(parents=True, exist_ok=True)
        valid_dir.mkdir(parents=True, exist_ok=True)
        
        for name, system in multi_systems.systems.items():
            n_frames = len(system)
            n_train = int(n_frames * train_ratio)
            
            # 随机打乱
            indices = np.random.permutation(n_frames)
            train_idx = indices[:n_train]
            valid_idx = indices[n_train:]
            
            # 保存训练集
            train_system = system.sub_system(train_idx)
            train_system.to_deepmd_npy(train_dir / name)
            
            # 保存验证集
            valid_system = system.sub_system(valid_idx)
            valid_system.to_deepmd_npy(valid_dir / name)
            
            logger.info(f"{name}: {n_train} train, {len(valid_idx)} valid frames")
        
        return str(train_dir), str(valid_dir)
    
    def convert_aimd_to_deepmd(self,
                               trajectory_file: str,
                               output_dir: str,
                               energy_threshold: float = 100.0) -> str:
        """
        从AIMD轨迹生成训练数据
        
        Args:
            trajectory_file: ASE trajectory文件
            output_dir: 输出目录
            energy_threshold: 能量阈值，过滤异常值 (eV/atom)
        """
        logger.info(f"Processing trajectory: {trajectory_file}")
        
        # 读取轨迹
        frames = read(trajectory_file, index=':')
        
        # 过滤异常能量
        energies = [atoms.get_potential_energy() / len(atoms) for atoms in frames]
        mean_e = np.mean(energies)
        std_e = np.std(energies)
        
        filtered_frames = []
        for atoms, e in zip(frames, energies):
            if abs(e - mean_e) < energy_threshold * std_e:
                filtered_frames.append(atoms)
        
        logger.info(f"Filtered {len(frames) - len(filtered_frames)} outliers")
        
        # 转换为dpdata格式
        coords = []
        cells = []
        energies = []
        forces = []
        
        for atoms in filtered_frames:
            coords.append(atoms.get_positions())
            cells.append(atoms.get_cell().array)
            energies.append(atoms.get_potential_energy())
            forces.append(atoms.get_forces())
        
        # 创建系统
        system = dpdata.LabeledSystem()
        system['atom_names'] = self.type_map
        system['atom_numbs'] = [list(atoms.get_chemical_symbols()).count(t) 
                                for t in self.type_map]
        system['atom_types'] = np.array([self.type_map.index(s) 
                                         for s in atoms.get_chemical_symbols()])
        system['coords'] = np.array(coords)
        system['cells'] = np.array(cells)
        system['energies'] = np.array(energies)
        system['forces'] = np.array(forces)
        system['orig'] = np.zeros(3)
        system['nopbc'] = False
        
        # 保存
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        system.to_deepmd_npy(output_path / "aimd_data")
        
        logger.info(f"Saved {len(filtered_frames)} frames to {output_path}")
        
        return str(output_path)
    
    def normalize_data(self, data_dir: str, output_dir: str) -> Dict[str, float]:
        """数据归一化"""
        # 读取所有能量
        energies = []
        for npy_file in Path(data_dir).rglob("energy.npy"):
            e = np.load(npy_file)
            energies.extend(e.flatten())
        
        energies = np.array(energies)
        
        # 计算统计量
        stats = {
            'mean': float(np.mean(energies)),
            'std': float(np.std(energies)),
            'min': float(np.min(energies)),
            'max': float(np.max(energies))
        }
        
        logger.info(f"Energy statistics: {stats}")
        
        return stats


class DeepMDTrainer:
    """DeePMD训练器"""
    
    def __init__(self, config: DeepMDConfig):
        self.config = config
        self.input_file = "input.json"
        
    def generate_input(self) -> str:
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
                "save_freq": 10000,
                "max_ckpt_keep": 5
            }
        }
        
        # 保存输入文件
        with open(self.input_file, 'w') as f:
            json.dump(input_dict, f, indent=2)
        
        logger.info(f"Generated input file: {self.input_file}")
        
        return self.input_file
    
    def train(self, restart: bool = False) -> str:
        """执行训练"""
        logger.info("Starting DeePMD training...")
        
        # 生成输入文件
        self.generate_input()
        
        # 构建命令
        cmd = ["dp", "train", self.input_file]
        if restart:
            cmd.append("--restart")
        
        # 执行训练
        try:
            subprocess.run(cmd, check=True)
            logger.info("Training completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e}")
            raise
        
        # 返回模型路径
        model_path = Path(self.config.output_dir) / "model.pb"
        return str(model_path)
    
    def freeze_model(self, model_dir: str = ".") -> str:
        """冻结模型"""
        logger.info("Freezing model...")
        
        cmd = ["dp", "freeze", "-o", "model.pb"]
        subprocess.run(cmd, cwd=model_dir, check=True)
        
        frozen_model = Path(model_dir) / "model.pb"
        logger.info(f"Model frozen: {frozen_model}")
        
        return str(frozen_model)
    
    def compress_model(self, model_path: str) -> str:
        """压缩模型"""
        logger.info("Compressing model...")
        
        compressed_path = model_path.replace(".pb", "-compress.pb")
        cmd = ["dp", "compress", "-i", model_path, "-o", compressed_path]
        subprocess.run(cmd, check=True)
        
        logger.info(f"Model compressed: {compressed_path}")
        
        return compressed_path
    
    def test_model(self, model_path: str, test_data: str) -> Dict:
        """测试模型"""
        logger.info("Testing model...")
        
        cmd = ["dp", "test", "-m", model_path, "-s", test_data, "-d", "test_results"]
        subprocess.run(cmd, check=True)
        
        # 读取测试结果
        # 这里简化处理，实际应解析详细输出
        results = {
            'model': model_path,
            'test_data': test_data,
            'status': 'completed'
        }
        
        return results


class ActiveLearningWorkflow:
    """
    主动学习工作流
    实现 Explore -> Label (DFT) -> Retrain 循环
    """
    
    def __init__(self,
                 initial_model: str,
                 explorer,
                 dft_calculator,
                 trainer: DeepMDTrainer,
                 uncertainty_threshold: float = 0.3):
        self.model = initial_model
        self.explorer = explorer  # MD explorer
        self.dft_calculator = dft_calculator
        self.trainer = trainer
        self.uncertainty_threshold = uncertainty_threshold
        self.iteration = 0
        
    def explore(self, n_structures: int = 100) -> List[Atoms]:
        """
        探索构型空间
        使用ML势运行MD，识别不确定性高的结构
        """
        logger.info(f"Exploring configuration space...")
        
        # 使用当前ML势运行探索MD
        trajectory = self.explorer.run_exploration(
            self.model,
            n_steps=100000,
            temperature_range=(300, 1000)
        )
        
        # 选择不确定性高的结构
        uncertain_structures = []
        for atoms in trajectory[::10]:  # 每10帧取一个
            uncertainty = self._compute_uncertainty(atoms)
            if uncertainty > self.uncertainty_threshold:
                uncertain_structures.append(atoms)
        
        # 限制数量
        if len(uncertain_structures) > n_structures:
            indices = np.random.choice(len(uncertain_structures), n_structures, replace=False)
            uncertain_structures = [uncertain_structures[i] for i in indices]
        
        logger.info(f"Found {len(uncertain_structures)} uncertain structures")
        
        return uncertain_structures
    
    def _compute_uncertainty(self, atoms: Atoms) -> float:
        """计算模型预测不确定性"""
        # 简化版本：使用模型集成或MC dropout
        # 实际应调用DeePMD的预测方差
        return np.random.random()  # 占位符
    
    def label(self, structures: List[Atoms]) -> str:
        """
        用DFT计算新结构的能量和力
        """
        logger.info(f"Labeling {len(structures)} structures with DFT...")
        
        labeled_data_dir = f"./iteration_{self.iteration}/labeled_data"
        Path(labeled_data_dir).mkdir(parents=True, exist_ok=True)
        
        for i, atoms in enumerate(structures):
            # DFT单点计算
            result = self.dft_calculator.single_point(atoms)
            
            # 保存结果
            atoms.calc = None
            atoms.info['energy'] = result['energy']
            atoms.arrays['forces'] = result['forces']
            
            write(f"{labeled_data_dir}/structure_{i}.extxyz", atoms)
        
        return labeled_data_dir
    
    def retrain(self, new_data_dir: str) -> str:
        """重新训练模型"""
        logger.info("Retraining model with new data...")
        
        # 合并新旧数据
        preprocessor = DataPreprocessor(self.trainer.config.type_map)
        preprocessor.convert_aimd_to_deepmd(
            f"{new_data_dir}/combined.traj",
            f"./iteration_{self.iteration}/training_data"
        )
        
        # 更新配置
        self.trainer.config.training_data = f"./iteration_{self.iteration}/training_data"
        
        # 训练
        new_model = self.trainer.train(restart=True)
        
        return new_model
    
    def run_iteration(self) -> str:
        """运行一个主动学习迭代"""
        logger.info(f"=== Active Learning Iteration {self.iteration} ===")
        
        # 探索
        new_structures = self.explore()
        
        if len(new_structures) == 0:
            logger.info("No uncertain structures found. Converged!")
            return self.model
        
        # 标记
        labeled_data = self.label(new_structures)
        
        # 重新训练
        new_model = self.retrain(labeled_data)
        self.model = new_model
        
        self.iteration += 1
        
        return new_model
    
    def run(self, max_iterations: int = 10) -> str:
        """运行主动学习循环"""
        for i in range(max_iterations):
            model = self.run_iteration()
            
            # 检查收敛
            if self._check_convergence():
                logger.info("Convergence achieved!")
                break
        
        return model
    
    def _check_convergence(self) -> bool:
        """检查收敛性"""
        # 实现收敛性检查逻辑
        return False


class DPGenInterface:
    """
    DP-GEN接口
    用于自动化主动学习
    """
    
    def __init__(self, config_file: str):
        self.config_file = config_file
        
    def generate_param_file(self, output: str = "param.json"):
        """生成DP-GEN参数文件"""
        param = {
            "type_map": ["H", "C", "O"],
            "mass_map": [1.008, 12.011, 15.999],
            "init_data_prefix": "./data",
            "init_data_sys": ["training"],
            "sys_configs": [
                ["./structures/config_1"],
                ["./structures/config_2"]
            ],
            "numb_models": 4,
            "default_training_param": {
                "model": {
                    "type_map": ["H", "C", "O"],
                    "descriptor": {
                        "type": "se_e2_a",
                        "rcut": 6.0,
                        "rcut_smth": 0.5,
                        "sel": [46, 46, 46],
                        "neuron": [25, 50, 100],
                        "resnet_dt": False,
                        "axis_neuron": 16,
                        "seed": 1
                    },
                    "fitting_net": {
                        "neuron": [240, 240, 240],
                        "resnet_dt": True,
                        "seed": 1
                    }
                },
                "learning_rate": {
                    "type": "exp",
                    "start_lr": 0.001,
                    "stop_lr": 3.51e-8,
                    "decay_steps": 5000
                },
                "loss": {
                    "start_pref_e": 0.02,
                    "limit_pref_e": 1,
                    "start_pref_f": 1000,
                    "limit_pref_f": 1,
                    "start_pref_v": 0.01,
                    "limit_pref_v": 1
                },
                "training": {
                    "stop_batch": 1000000,
                    "batch_size": "auto",
                    "disp_file": "lcurve.out",
                    "save_freq": 10000
                }
            },
            "model_devi_dt": 0.002,
            "model_devi_skip": 0,
            "model_devi_f_trust_lo": 0.05,
            "model_devi_f_trust_hi": 0.15,
            "model_devi_e_trust_lo": 0.05,
            "model_devi_e_trust_hi": 0.15,
            "model_devi_jobs": [
                {"sys_idx": [0], "temps": [50, 100], "press": [1.0], "trj_freq": 10, "nsteps": 1000}
            ],
            "fp_style": "vasp",
            "fp_task_max": 20,
            "fp_task_min": 5,
            "fp_pp_path": "./pp",
            "fp_pp_files": {"H": "H.pbe", "C": "C.pbe", "O": "O.pbe"},
            "fp_incar": "./INCAR"
        }
        
        with open(output, 'w') as f:
            json.dump(param, f, indent=2)
        
        logger.info(f"Generated DP-GEN param file: {output}")
    
    def run(self):
        """运行DP-GEN"""
        cmd = ["dpgen", "run", self.config_file, "machine.json"]
        subprocess.run(cmd, check=True)


def main():
    """示例用法"""
    
    # 配置
    config = DeepMDConfig(
        type_map=["Li", "C", "O"],
        descriptor_type="se_e2_a",
        rcut=6.0,
        sel=[50, 50, 50],
        numb_steps=1000000
    )
    
    # 数据预处理
    preprocessor = DataPreprocessor(config.type_map)
    
    # 转换VASP数据
    # train_dir, valid_dir = preprocessor.convert_vasp_to_deepmd(
    #     vasp_dirs=["./vasp_run1", "./vasp_run2"],
    #     output_dir="./data"
    # )
    
    # 训练
    trainer = DeepMDTrainer(config)
    # model_path = trainer.train()
    
    # 冻结和压缩
    # frozen = trainer.freeze_model()
    # compressed = trainer.compress_model(frozen)
    
    print("ML Potential Training template ready!")


if __name__ == "__main__":
    main()
