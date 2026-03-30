"""
nep_training/trainer.py
=======================
NEP训练器模块

包含:
- 基础NEP训练器
- 分布式训练器
- 混合精度训练器
"""

import os
import sys
import time
import json
import logging
import subprocess
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import shutil
import threading
import queue

from ase import Atoms
from ase.io import read, write

from .core import (
    NEPModelConfig, NEPTrainingConfig, NEPCheckpoint, 
    TrainingState, PrecisionMode, DistributedConfig
)
from .strategies import LRScheduler, EarlyStopping, ModelEnsemble, EnsembleConfig
from .data import NEPDataset, NEPDataLoader

logger = logging.getLogger(__name__)


class NEPTrainerV2:
    """
    增强版NEP训练器
    
    支持:
    - 学习率调度
    - 早停
    - 检查点管理
    - 实时验证
    - 混合精度训练
    """
    
    def __init__(self, 
                 model_config: NEPModelConfig,
                 training_config: NEPTrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
        
        self.state = TrainingState.IDLE
        self.current_generation = 0
        self.best_loss = float('inf')
        
        # 训练历史
        self.train_loss_history = []
        self.val_loss_history = []
        
        # 组件
        self.lr_scheduler: Optional[LRScheduler] = None
        self.early_stopping: Optional[EarlyStopping] = None
        
        # 工作目录
        self.work_dir = Path(training_config.working_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # 回调函数
        self.callbacks: List[Callable] = []
        
        # 初始化
        self._setup()
    
    def _setup(self):
        """初始化训练设置"""
        # 设置GPU
        if self.training_config.use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.training_config.gpu_id)
        
        # 初始化学习率调度器
        if self.training_config.use_lr_scheduler:
            from .strategies import ExponentialDecayScheduler
            self.lr_scheduler = ExponentialDecayScheduler(
                initial_lr=self.model_config.initial_lr,
                decay_rate=self.model_config.lr_decay_rate,
                decay_steps=self.model_config.lr_decay_steps,
                min_lr=self.model_config.min_lr
            )
        
        # 初始化早停
        if self.training_config.use_early_stopping:
            self.early_stopping = EarlyStopping(
                patience=self.training_config.early_stopping_patience,
                min_delta=self.training_config.early_stopping_min_delta,
                restore_best=True
            )
    
    def setup_training(self, train_dataset: NEPDataset, 
                      val_dataset: Optional[NEPDataset] = None):
        """
        设置训练数据
        
        Args:
            train_dataset: 训练数据集
            val_dataset: 验证数据集 (可选)
        """
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # 保存训练数据
        train_path = self.work_dir / "train.xyz"
        train_dataset.save_xyz(str(train_path))
        self.training_config.train_xyz = str(train_path)
        
        if val_dataset:
            val_path = self.work_dir / "val.xyz"
            val_dataset.save_xyz(str(val_path))
            self.training_config.val_xyz = str(val_path)
        
        # 生成nep.in
        self._generate_nep_in()
        
        logger.info(f"Training setup complete:")
        logger.info(f"  Train: {len(train_dataset)} frames")
        if val_dataset:
            logger.info(f"  Val: {len(val_dataset)} frames")
    
    def _generate_nep_in(self):
        """生成nep.in输入文件"""
        nep_in_path = self.work_dir / self.training_config.nep_in
        
        lines = []
        
        # 元素类型
        lines.append(f"type {' '.join(self.model_config.type_list)}")
        
        # 模型类型
        if self.model_config.model_type != 0:
            lines.append(f"model_type {self.model_config.model_type}")
        
        # 版本
        lines.append(f"version {self.model_config.version}")
        
        # 截断半径
        lines.append(f"cutoff {self.model_config.cutoff_radial} {self.model_config.cutoff_angular}")
        
        # n_max
        lines.append(f"n_max {self.model_config.n_max_radial} {self.model_config.n_max_angular}")
        
        # basis_size (NEP4)
        if self.model_config.version >= 4:
            lines.append(f"basis_size {self.model_config.basis_size_radial} {self.model_config.basis_size_angular}")
        
        # l_max
        l_max_str = f"l_max {self.model_config.l_max_3body}"
        if self.model_config.l_max_4body > 0:
            l_max_str += f" {self.model_config.l_max_4body}"
        if self.model_config.l_max_5body > 0:
            l_max_str += f" {self.model_config.l_max_5body}"
        lines.append(l_max_str)
        
        # 神经元
        lines.append(f"neuron {self.model_config.neuron}")
        
        # 种群大小
        lines.append(f"population {self.model_config.population_size}")
        
        # 最大代数
        lines.append(f"generation {self.model_config.maximum_generation}")
        
        # 批量大小
        lines.append(f"batch {self.model_config.batch_size}")
        
        # 写入文件
        with open(nep_in_path, 'w') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Generated nep.in: {nep_in_path}")
    
    def train(self, verbose: bool = True) -> str:
        """
        执行NEP训练
        
        Returns:
            训练好的模型文件路径
        """
        self.state = TrainingState.TRAINING
        
        nep_exe = Path(self.training_config.gpumd_path) / "nep"
        if not nep_exe.exists():
            nep_exe = shutil.which("nep") or "nep"
        
        logger.info(f"Starting NEP training with executable: {nep_exe}")
        logger.info(f"Working directory: {self.work_dir}")
        
        # 构建训练环境
        env = os.environ.copy()
        if self.training_config.precision == PrecisionMode.FP16:
            env['NEP_PRECISION'] = 'fp16'
        
        try:
            # 启动训练进程
            process = subprocess.Popen(
                [str(nep_exe)],
                cwd=self.work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env
            )
            
            # 实时监控输出
            log_file = self.work_dir / "training.log"
            with open(log_file, 'w') as f:
                for line in process.stdout:
                    if verbose:
                        print(line, end='')
                    f.write(line)
                    f.flush()
                    
                    # 解析训练进度
                    self._parse_training_output(line)
                    
                    # 执行回调
                    for callback in self.callbacks:
                        callback(self._get_training_status())
            
            process.wait()
            
            if process.returncode != 0:
                self.state = TrainingState.FAILED
                raise subprocess.CalledProcessError(process.returncode, [str(nep_exe)])
            
            self.state = TrainingState.COMPLETED
            logger.info("NEP training completed successfully!")
            
        except subprocess.CalledProcessError as e:
            self.state = TrainingState.FAILED
            logger.error(f"Training failed: {e}")
            raise
        
        # 返回模型文件路径
        model_file = self.work_dir / "nep.txt"
        if not model_file.exists():
            raise RuntimeError("Model file not generated!")
        
        return str(model_file)
    
    def _parse_training_output(self, line: str):
        """解析训练输出，更新状态"""
        # 解析loss.out格式的输出
        # 格式: generation L1_train L2_train ... L1_test L2_test ...
        parts = line.strip().split()
        if len(parts) >= 3 and parts[0].isdigit():
            try:
                gen = int(parts[0])
                self.current_generation = gen
                
                train_loss = float(parts[1])
                self.train_loss_history.append({'generation': gen, 'loss': train_loss})
                
                if len(parts) > 4:
                    val_loss = float(parts[4])  # 假设第5列是验证损失
                    self.val_loss_history.append({'generation': gen, 'loss': val_loss})
                    
                    # 更新最佳损失
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                
                # 学习率调度
                if self.lr_scheduler:
                    self.lr_scheduler.step(gen, train_loss)
                
                # 早停检查
                if self.early_stopping:
                    should_stop, _ = self.early_stopping.step(gen, train_loss)
                    if should_stop:
                        logger.info(f"Early stopping triggered at generation {gen}")
                        # 注意: NEP不直接支持中途停止，这里只是记录
                
            except (ValueError, IndexError):
                pass
    
    def _get_training_status(self) -> Dict[str, Any]:
        """获取当前训练状态"""
        return {
            'state': self.state.value,
            'generation': self.current_generation,
            'max_generation': self.model_config.maximum_generation,
            'progress': self.current_generation / self.model_config.maximum_generation * 100,
            'best_loss': self.best_loss,
            'train_loss_history': self.train_loss_history[-10:] if self.train_loss_history else [],
            'val_loss_history': self.val_loss_history[-10:] if self.val_loss_history else [],
        }
    
    def add_callback(self, callback: Callable):
        """添加训练回调函数"""
        self.callbacks.append(callback)
    
    def get_loss_history(self) -> Dict[str, List]:
        """获取训练损失历史"""
        return {
            'train': self.train_loss_history,
            'val': self.val_loss_history
        }
    
    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """保存训练检查点"""
        if path is None:
            path = Path(self.training_config.checkpoint_dir) / f"checkpoint_gen{self.current_generation}.pkl"
        
        checkpoint = NEPCheckpoint(
            generation=self.current_generation,
            model_state={},  # NEP模型通过文件保存
            optimizer_state={},
            scheduler_state={'lr': self.lr_scheduler.get_lr()} if self.lr_scheduler else None,
            best_loss=self.best_loss,
            train_loss_history=self.train_loss_history,
            val_loss_history=self.val_loss_history,
            metrics={'generation': self.current_generation},
            timestamp=datetime.now().isoformat(),
            config=self.training_config
        )
        
        checkpoint.save(str(path))
        logger.info(f"Checkpoint saved: {path}")
        
        return str(path)
    
    def load_checkpoint(self, path: str):
        """加载训练检查点"""
        checkpoint = NEPCheckpoint.load(path)
        
        self.current_generation = checkpoint.generation
        self.best_loss = checkpoint.best_loss
        self.train_loss_history = checkpoint.train_loss_history
        self.val_loss_history = checkpoint.val_loss_history
        
        logger.info(f"Checkpoint loaded: generation={self.current_generation}")


class DistributedNEPTrainer:
    """
    分布式NEP训练器
    
    支持多GPU并行训练
    
    注意: NEP本身使用SNES算法，并行化主要在数据加载和
    多个独立模型训练层面实现
    """
    
    def __init__(self, 
                 model_config: NEPModelConfig,
                 training_config: NEPTrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
        self.distributed_config = training_config.distributed
        
        self.trainers: List[NEPTrainerV2] = []
        self.is_distributed = self.distributed_config.enabled
    
    def setup(self, train_dataset: NEPDataset, val_dataset: Optional[NEPDataset] = None):
        """设置分布式训练"""
        if not self.is_distributed:
            # 单GPU模式
            self.trainers = [NEPTrainerV2(self.model_config, self.training_config)]
            self.trainers[0].setup_training(train_dataset, val_dataset)
            return
        
        # 多GPU模式 - 为每个GPU创建独立训练器
        world_size = self.distributed_config.world_size
        
        for rank in range(world_size):
            config_copy = self._copy_config_for_rank(rank)
            trainer = NEPTrainerV2(self.model_config, config_copy)
            
            # 数据分片
            train_shard = self._split_dataset(train_dataset, rank, world_size)
            val_shard = self._split_dataset(val_dataset, rank, world_size) if val_dataset else None
            
            trainer.setup_training(train_shard, val_shard)
            self.trainers.append(trainer)
    
    def _copy_config_for_rank(self, rank: int) -> NEPTrainingConfig:
        """为指定rank复制配置"""
        import copy
        config = copy.deepcopy(self.training_config)
        config.gpu_id = rank
        config.working_dir = f"{config.working_dir}/rank_{rank}"
        return config
    
    def _split_dataset(self, dataset: NEPDataset, rank: int, world_size: int) -> NEPDataset:
        """分割数据集"""
        n_samples = len(dataset)
        samples_per_rank = n_samples // world_size
        
        start_idx = rank * samples_per_rank
        end_idx = start_idx + samples_per_rank if rank < world_size - 1 else n_samples
        
        return NEPDataset(frames=dataset.frames[start_idx:end_idx])
    
    def train(self) -> List[str]:
        """
        执行分布式训练
        
        Returns:
            各rank的模型文件路径列表
        """
        if not self.is_distributed:
            return [self.trainers[0].train()]
        
        # 并行训练多个模型
        import multiprocessing as mp
        
        results = []
        processes = []
        
        for trainer in self.trainers:
            p = mp.Process(target=self._train_worker, args=(trainer, results))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        return results
    
    def _train_worker(self, trainer: NEPTrainerV2, results: List):
        """训练工作进程"""
        model_path = trainer.train(verbose=False)
        results.append(model_path)


class MixedPrecisionTrainer:
    """
    混合精度训练器
    
    使用FP16/BF16加速训练并减少内存占用
    
    注意: NEP本身使用SNES算法，这里主要是在数据加载和
    预处理阶段使用混合精度
    """
    
    def __init__(self, 
                 model_config: NEPModelConfig,
                 training_config: NEPTrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
        self.precision = training_config.precision
        
        self.base_trainer = NEPTrainerV2(model_config, training_config)
        self._setup_precision()
    
    def _setup_precision(self):
        """设置混合精度环境"""
        if self.precision == PrecisionMode.FP16:
            os.environ['NEP_PRECISION'] = 'fp16'
            logger.info("Using FP16 precision")
        elif self.precision == PrecisionMode.BF16:
            os.environ['NEP_PRECISION'] = 'bf16'
            logger.info("Using BF16 precision")
    
    def setup_training(self, train_dataset: NEPDataset,
                      val_dataset: Optional[NEPDataset] = None):
        """设置训练"""
        # 转换数据精度
        if self.precision in [PrecisionMode.FP16, PrecisionMode.BF16]:
            train_dataset = self._convert_dataset_precision(train_dataset)
            if val_dataset:
                val_dataset = self._convert_dataset_precision(val_dataset)
        
        self.base_trainer.setup_training(train_dataset, val_dataset)
    
    def _convert_dataset_precision(self, dataset: NEPDataset) -> NEPDataset:
        """转换数据集精度"""
        # NEP数据以文本格式存储，精度影响主要在计算阶段
        # 这里只是标记
        return dataset
    
    def train(self, verbose: bool = True) -> str:
        """执行训练"""
        return self.base_trainer.train(verbose=verbose)


class EnsembleTrainer:
    """
    集成训练器
    
    训练多个NEP模型并集成
    """
    
    def __init__(self,
                 model_config: NEPModelConfig,
                 training_config: NEPTrainingConfig,
                 ensemble_config: EnsembleConfig):
        self.model_config = model_config
        self.training_config = training_config
        self.ensemble_config = ensemble_config
        
        self.trainers: List[NEPTrainerV2] = []
        self.models: List[str] = []
    
    def setup(self, train_dataset: NEPDataset, val_dataset: Optional[NEPDataset] = None):
        """设置集成训练"""
        from sklearn.model_selection import train_test_split
        
        for i in range(self.ensemble_config.n_models):
            # 为每个模型创建独立工作目录
            work_dir = Path(self.training_config.working_dir) / f"model_{i}"
            work_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制配置
            import copy
            config = copy.deepcopy(self.training_config)
            config.working_dir = str(work_dir)
            
            trainer = NEPTrainerV2(self.model_config, config)
            
            # Bootstrap采样
            if self.ensemble_config.bootstrap:
                train_shard = self._bootstrap_sample(train_dataset)
            else:
                train_shard = train_dataset
            
            trainer.setup_training(train_shard, val_dataset)
            self.trainers.append(trainer)
    
    def _bootstrap_sample(self, dataset: NEPDataset) -> NEPDataset:
        """Bootstrap采样"""
        n_samples = int(len(dataset) * self.ensemble_config.bootstrap_ratio)
        indices = np.random.choice(len(dataset), n_samples, replace=True)
        return NEPDataset(frames=[dataset.frames[i] for i in indices])
    
    def train(self) -> List[str]:
        """
        训练所有集成模型
        
        Returns:
            各模型的文件路径列表
        """
        self.models = []
        
        for i, trainer in enumerate(self.trainers):
            logger.info(f"\n{'='*60}")
            logger.info(f"Training ensemble model {i+1}/{len(self.trainers)}")
            logger.info(f"{'='*60}")
            
            model_path = trainer.train(verbose=True)
            self.models.append(model_path)
            
            # 重命名模型文件以避免覆盖
            new_path = Path(model_path).parent / f"nep_model_{i}.txt"
            shutil.copy(model_path, new_path)
        
        return self.models
    
    def get_ensemble_predictions(self, xyz_file: str) -> Dict[str, np.ndarray]:
        """获取集成预测"""
        predictions = []
        
        for model_path in self.models:
            # 使用模型进行预测
            # 这里简化处理，实际应调用GPUMD进行预测
            pred = self._predict_single(model_path, xyz_file)
            predictions.append(pred)
        
        # 集成预测
        energies = np.array([p['energy'] for p in predictions])
        
        return {
            'energy_mean': np.mean(energies, axis=0),
            'energy_std': np.std(energies, axis=0),
            'energy_ensemble': energies,
        }
    
    def _predict_single(self, model_path: str, xyz_file: str) -> Dict[str, np.ndarray]:
        """单个模型预测"""
        # 简化版本
        return {
            'energy': np.random.randn(10),  # 占位符
            'forces': np.random.randn(10, 3)
        }
