"""
nep_training/strategies.py
==========================
高级训练策略模块

包含:
- 学习率调度器
- 早停机制
- 模型集成
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class LRScheduler:
    """
    学习率调度器基类
    
    NEP使用SNES (Separable Natural Evolution Strategy) 优化，
    学习率概念与常规梯度下降略有不同，这里实现通用的学习率衰减策略
    """
    
    def __init__(self, initial_lr: float = 0.1):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.step_count = 0
    
    def step(self, generation: int, current_loss: float) -> float:
        """更新学习率"""
        self.step_count = generation
        self.current_lr = self._compute_lr(generation, current_loss)
        return self.current_lr
    
    def _compute_lr(self, generation: int, current_loss: float) -> float:
        """计算当前学习率 - 子类实现"""
        raise NotImplementedError
    
    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.current_lr
    
    def reset(self):
        """重置调度器"""
        self.current_lr = self.initial_lr
        self.step_count = 0


class ExponentialDecayScheduler(LRScheduler):
    """
    指数衰减学习率调度器
    
    lr = initial_lr * decay_rate^(step / decay_steps)
    """
    
    def __init__(self, initial_lr: float = 0.1, decay_rate: float = 0.95, 
                 decay_steps: int = 1000, min_lr: float = 1e-6):
        super().__init__(initial_lr)
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.min_lr = min_lr
    
    def _compute_lr(self, generation: int, current_loss: float) -> float:
        lr = self.initial_lr * (self.decay_rate ** (generation / self.decay_steps))
        return max(lr, self.min_lr)


class CosineAnnealingScheduler(LRScheduler):
    """
    余弦退火学习率调度器
    
    lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * step / T_max))
    """
    
    def __init__(self, initial_lr: float = 0.1, min_lr: float = 1e-6,
                 T_max: int = 10000, warmup_steps: int = 0):
        super().__init__(initial_lr)
        self.min_lr = min_lr
        self.T_max = T_max
        self.warmup_steps = warmup_steps
    
    def _compute_lr(self, generation: int, current_loss: float) -> float:
        if generation < self.warmup_steps:
            # 线性warmup
            return self.initial_lr * (generation / self.warmup_steps)
        
        step = generation - self.warmup_steps
        cosine_decay = 0.5 * (1 + np.cos(np.pi * step / self.T_max))
        lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
        return max(lr, self.min_lr)


class PlateauScheduler(LRScheduler):
    """
    基于平台的学习率调度器
    
    当验证损失不再改善时降低学习率
    """
    
    def __init__(self, initial_lr: float = 0.1, factor: float = 0.5,
                 patience: int = 10, min_lr: float = 1e-6):
        super().__init__(initial_lr)
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.wait = 0
    
    def _compute_lr(self, generation: int, current_loss: float) -> float:
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.wait = 0
            new_lr = self.current_lr * self.factor
            logger.info(f"Reducing learning rate: {self.current_lr:.6f} -> {new_lr:.6f}")
            return max(new_lr, self.min_lr)
        
        return self.current_lr


class WarmupScheduler(LRScheduler):
    """
    Warmup学习率调度器
    
    支持linear和exponential warmup
    """
    
    def __init__(self, base_scheduler: LRScheduler, warmup_steps: int = 1000,
                 warmup_mode: str = "linear"):
        super().__init__(base_scheduler.initial_lr)
        self.base_scheduler = base_scheduler
        self.warmup_steps = warmup_steps
        self.warmup_mode = warmup_mode
        self.final_initial_lr = base_scheduler.initial_lr
        self.base_scheduler.initial_lr = self.final_initial_lr
    
    def _compute_lr(self, generation: int, current_loss: float) -> float:
        if generation < self.warmup_steps:
            if self.warmup_mode == "linear":
                return self.final_initial_lr * (generation / self.warmup_steps)
            elif self.warmup_mode == "exponential":
                progress = generation / self.warmup_steps
                return self.final_initial_lr * (0.1 ** (1 - progress))
            else:
                raise ValueError(f"Unknown warmup mode: {self.warmup_mode}")
        
        return self.base_scheduler._compute_lr(generation, current_loss)
    
    def step(self, generation: int, current_loss: float) -> float:
        self.step_count = generation
        self.current_lr = self._compute_lr(generation, current_loss)
        # 同时更新base scheduler
        self.base_scheduler.step_count = generation
        self.base_scheduler.current_lr = self.current_lr
        return self.current_lr


class EarlyStopping:
    """
    早停机制
    
    监控验证损失，当连续patience轮没有改善时停止训练
    """
    
    def __init__(self, patience: int = 20, min_delta: float = 1e-5,
                 mode: str = "min", restore_best: bool = True):
        """
        Args:
            patience: 容忍轮数
            min_delta: 最小改善阈值
            mode: "min" (最小化) 或 "max" (最大化)
            restore_best: 是否恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.best_generation = 0
        self.wait = 0
        self.should_stop = False
        self.best_model_state = None
        
        self.history = []
    
    def step(self, generation: int, value: float, model_state: Optional[Dict] = None):
        """
        更新早停状态
        
        Returns:
            should_stop: 是否应该停止
            is_improved: 是否有改善
        """
        self.history.append({'generation': generation, 'value': value})
        
        is_improved = self._is_improved(value)
        
        if is_improved:
            self.best_value = value
            self.best_generation = generation
            self.wait = 0
            if model_state is not None and self.restore_best:
                self.best_model_state = model_state.copy()
            logger.info(f"New best value: {value:.6f} at generation {generation}")
        else:
            self.wait += 1
            logger.debug(f"No improvement for {self.wait} generations (best: {self.best_value:.6f})")
        
        if self.wait >= self.patience:
            self.should_stop = True
            logger.info(f"Early stopping triggered at generation {generation}")
        
        return self.should_stop, is_improved
    
    def _is_improved(self, value: float) -> bool:
        """检查是否有改善"""
        if self.mode == "min":
            return value < (self.best_value - self.min_delta)
        else:
            return value > (self.best_value + self.min_delta)
    
    def get_best_state(self) -> Optional[Dict]:
        """获取最佳模型状态"""
        return self.best_model_state
    
    def reset(self):
        """重置早停状态"""
        self.best_value = float('inf') if self.mode == "min" else float('-inf')
        self.best_generation = 0
        self.wait = 0
        self.should_stop = False
        self.best_model_state = None
        self.history = []


@dataclass
class EnsembleConfig:
    """模型集成配置"""
    n_models: int = 4
    bootstrap: bool = True  # 使用bootstrap采样
    bootstrap_ratio: float = 0.8
    seeds: Optional[List[int]] = None
    aggregation_method: str = "mean"  # mean, median, weighted
    
    def __post_init__(self):
        if self.seeds is None:
            self.seeds = list(range(42, 42 + self.n_models))


class ModelEnsemble:
    """
    NEP模型集成
    
    训练多个NEP模型并集成预测，提高可靠性和不确定性估计
    """
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.models = []  # 存储模型路径
        self.training_histories = []
        self.weights = None  # 加权平均权重
    
    def prepare_datasets(self, train_xyz: str, output_dir: str) -> List[Tuple[str, str]]:
        """
        准备集成训练数据集
        
        Returns:
            每个模型的(train_xyz, val_xyz)路径列表
        """
        from sklearn.model_selection import train_test_split
        import shutil
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 读取完整训练集
        # 这里简化处理，实际应使用NEPDataLoader
        
        datasets = []
        for i in range(self.config.n_models):
            model_dir = output_dir / f"model_{i}"
            model_dir.mkdir(exist_ok=True)
            
            if self.config.bootstrap:
                # Bootstrap采样
                # 实际应使用NEPDataLoader进行采样
                train_out = model_dir / "train.xyz"
                val_out = model_dir / "val.xyz"
                
                # 复制并标记
                shutil.copy(train_xyz, train_out)
                # 创建验证集 (可以通过随机分割)
                
                datasets.append((str(train_out), str(val_out)))
            else:
                # 使用不同的随机种子分割
                train_out = model_dir / "train.xyz"
                val_out = model_dir / "val.xyz"
                
                shutil.copy(train_xyz, train_out)
                datasets.append((str(train_out), str(val_out)))
        
        return datasets
    
    def aggregate_predictions(self, predictions: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        集成多个模型的预测结果
        
        Args:
            predictions: 每个模型的预测结果列表
            
        Returns:
            集成后的预测结果和不确定性估计
        """
        if not predictions:
            raise ValueError("No predictions to aggregate")
        
        # 提取各模型的预测
        energies = np.array([p['energy'] for p in predictions])  # (n_models, n_structures)
        forces_list = [p['forces'] for p in predictions]
        
        # 计算均值 (集成预测)
        if self.config.aggregation_method == "mean":
            energy_mean = np.mean(energies, axis=0)
        elif self.config.aggregation_method == "median":
            energy_mean = np.median(energies, axis=0)
        elif self.config.aggregation_method == "weighted" and self.weights is not None:
            energy_mean = np.average(energies, axis=0, weights=self.weights)
        else:
            energy_mean = np.mean(energies, axis=0)
        
        # 计算不确定性 (标准差)
        energy_std = np.std(energies, axis=0)
        
        # 力集成
        force_mean = np.mean(forces_list, axis=0)
        force_std = np.std(forces_list, axis=0)
        
        return {
            'energy': energy_mean,
            'energy_std': energy_std,
            'energy_ensemble': energies,  # 保留所有模型预测
            'forces': force_mean,
            'forces_std': force_std,
            'forces_ensemble': forces_list,
            'uncertainty': energy_std,  # 主要不确定性指标
        }
    
    def compute_uncertainty(self, predictions: List[Dict[str, np.ndarray]]) -> np.ndarray:
        """
        计算预测不确定性
        
        使用模型间预测的方差作为不确定性估计
        """
        energies = np.array([p['energy'] for p in predictions])
        return np.std(energies, axis=0)
    
    def save_metadata(self, output_path: str):
        """保存集成元数据"""
        metadata = {
            'config': {
                'n_models': self.config.n_models,
                'bootstrap': self.config.bootstrap,
                'seeds': self.config.seeds,
                'aggregation_method': self.config.aggregation_method,
            },
            'models': self.models,
            'training_histories': self.training_histories,
        }
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_metadata(self, metadata_path: str):
        """加载集成元数据"""
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.config = EnsembleConfig(**metadata['config'])
        self.models = metadata['models']
        self.training_histories = metadata.get('training_histories', [])


class AdaptiveStrategy:
    """
    自适应训练策略
    
    根据训练进展动态调整超参数
    """
    
    def __init__(self):
        self.generation = 0
        self.loss_history = []
        self.strategy_history = []
    
    def update(self, generation: int, train_loss: float, val_loss: float) -> Dict[str, Any]:
        """
        更新策略状态并返回调整建议
        
        Returns:
            调整建议字典
        """
        self.generation = generation
        self.loss_history.append({
            'generation': generation,
            'train': train_loss,
            'val': val_loss
        })
        
        suggestions = {}
        
        # 检测过拟合
        if len(self.loss_history) >= 20:
            recent_train = [h['train'] for h in self.loss_history[-20:]]
            recent_val = [h['val'] for h in self.loss_history[-20:]]
            
            train_trend = np.polyfit(range(20), recent_train, 1)[0]
            val_trend = np.polyfit(range(20), recent_val, 1)[0]
            
            # 如果训练损失下降但验证损失上升或停滞，可能是过拟合
            if train_trend < 0 and val_trend > 0:
                suggestions['detected_overfitting'] = True
                suggestions['recommendations'] = [
                    "Consider reducing model complexity",
                    "Increase regularization",
                    "Reduce learning rate",
                    "Collect more diverse training data"
                ]
        
        # 检测训练停滞
        if len(self.loss_history) >= 50:
            recent_losses = [h['train'] for h in self.loss_history[-50:]]
            if np.std(recent_losses) / np.mean(recent_losses) < 0.01:
                suggestions['detected_stagnation'] = True
                suggestions['recommendations'] = [
                    "Learning rate might be too low",
                    "Consider increasing batch size",
                    "Model may have reached capacity"
                ]
        
        self.strategy_history.append({
            'generation': generation,
            'suggestions': suggestions
        })
        
        return suggestions
