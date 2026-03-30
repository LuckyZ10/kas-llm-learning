"""
Delta Learning (Δ-Learning) Framework
Δ-学习框架

实现从低精度DFT到高精度方法(如CCSD(T)/QMC)的机器学习修正。
支持能量、力、应力的多目标修正，以及转移学习策略。

核心思想:
E_high = E_low + ΔE_ML
F_high = F_low + ΔF_ML
σ_high = σ_low + Δσ_ML

其中ΔE_ML, ΔF_ML, Δσ_ML由神经网络学习

作者: DFT-LAMMPS Team
日期: 2026-03-09
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
import logging
from collections import defaultdict
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class DeltaLearningConfig:
    """Δ-学习配置"""
    # 模型架构
    descriptor_type: str = 'soap'  # soap, ace, deepmd, custom
    descriptor_dim: int = 100
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 128, 64])
    activation: str = 'silu'
    use_batch_norm: bool = True
    dropout_rate: float = 0.1
    
    # 训练目标
    predict_energy: bool = True
    predict_forces: bool = True
    predict_stress: bool = False
    predict_virial: bool = False
    
    # 损失函数权重
    energy_weight: float = 1.0
    force_weight: float = 50.0
    stress_weight: float = 1.0
    virial_weight: float = 1.0
    smoothness_weight: float = 0.01
    
    # 训练参数
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 1000
    early_stopping_patience: int = 100
    learning_rate_decay: float = 0.5
    learning_rate_patience: int = 20
    
    # 优化器
    optimizer: str = 'adam'  # adam, adamw, sgd
    scheduler: str = 'plateau'  # plateau, cosine, step
    
    # 数值稳定性
    energy_scale: float = 1.0  # eV -> 目标单位
    length_scale: float = 1.0  # Angstrom -> 目标单位
    force_scale: float = None  # 自动计算
    
    # 转移学习
    use_transfer_learning: bool = False
    pretrained_model_path: Optional[str] = None
    freeze_layers: List[int] = field(default_factory=list)
    fine_tune_lr_ratio: float = 0.1


class SOAPDescriptor(nn.Module):
    """
    SOAP (Smooth Overlap of Atomic Positions) 描述符
    用于原子环境的平滑描述
    """
    
    def __init__(self, rcut: float = 6.0, nmax: int = 8, lmax: int = 6,
                 sigma: float = 0.5, periodic: bool = True):
        super().__init__()
        self.rcut = rcut
        self.nmax = nmax
        self.lmax = lmax
        self.sigma = sigma
        self.periodic = periodic
        
        # 预计算GTO径向基
        self.register_buffer('radial_grid', torch.linspace(0, rcut, 100))
    
    def forward(self, positions: torch.Tensor, atom_types: torch.Tensor,
                cell: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算SOAP描述符
        
        Args:
            positions: 原子位置 [n_atoms, 3]
            atom_types: 原子类型 [n_atoms]
            cell: 晶胞矩阵 [3, 3] (周期性体系)
            
        Returns:
            SOAP描述符 [n_atoms, descriptor_dim]
        """
        n_atoms = positions.shape[0]
        
        # 计算邻居列表 (简化实现)
        neighbors, distances = self._compute_neighbors(positions, cell)
        
        # 计算SOAP描述符
        descriptors = []
        for i in range(n_atoms):
            desc_i = self._compute_single_soap(
                positions[i], positions[neighbors[i]],
                distances[i], atom_types[neighbors[i]]
            )
            descriptors.append(desc_i)
        
        return torch.stack(descriptors)
    
    def _compute_neighbors(self, positions: torch.Tensor,
                          cell: Optional[torch.Tensor]) -> Tuple[List[List[int]], List[List[float]]]:
        """计算邻居列表"""
        n_atoms = positions.shape[0]
        
        # 计算距离矩阵
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # [n, n, 3]
        
        if self.periodic and cell is not None:
            # 应用最小图像约定
            inv_cell = torch.inverse(cell)
            fractional = torch.matmul(diff, inv_cell)
            fractional = fractional - torch.round(fractional)
            diff = torch.matmul(fractional, cell)
        
        distances = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-10)
        
        # 找出邻居
        neighbors = []
        distances_list = []
        for i in range(n_atoms):
            mask = (distances[i] < self.rcut) & (distances[i] > 0.1)
            neighbor_idx = torch.where(mask)[0].tolist()
            neighbor_dist = distances[i, mask].tolist()
            neighbors.append(neighbor_idx)
            distances_list.append(neighbor_dist)
        
        return neighbors, distances_list
    
    def _compute_single_soap(self, center_pos: torch.Tensor,
                             neighbor_pos: torch.Tensor,
                             distances: List[float],
                             neighbor_types: torch.Tensor) -> torch.Tensor:
        """计算单个原子的SOAP描述符"""
        # 简化的SOAP实现
        # 实际实现需要完整的球谐函数展开
        
        n_neighbors = len(distances)
        if n_neighbors == 0:
            return torch.zeros(self.nmax * (self.lmax + 1))
        
        # 径向基函数 (GTO)
        distances_t = torch.tensor(distances)
        radial_parts = []
        for n in range(self.nmax):
            alpha = 0.5 / self.sigma**2
            gto = (distances_t / self.rcut)**n * torch.exp(-alpha * distances_t**2)
            radial_parts.append(gto)
        
        radial_features = torch.cat(radial_parts)
        
        # 简化的角向部分 (实际需要球谐函数)
        angular_features = torch.zeros(self.lmax + 1)
        
        # 组合特征
        descriptor = torch.cat([radial_features, angular_features])
        
        return descriptor


class ACEDescriptor(nn.Module):
    """
    ACE (Atomic Cluster Expansion) 描述符
    高阶多体描述符
    """
    
    def __init__(self, rcut: float = 6.0, nmax: int = 10, lmax: int = 6,
                 nu_max: int = 3):
        super().__init__()
        self.rcut = rcut
        self.nmax = nmax
        self.lmax = lmax
        self.nu_max = nu_max  # 最大相关阶数
    
    def forward(self, positions: torch.Tensor, atom_types: torch.Tensor,
                cell: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算ACE描述符"""
        # 简化实现
        # 实际ACE需要复杂的对称性适配基组
        n_atoms = positions.shape[0]
        descriptor_dim = self.nmax * (self.lmax + 1) * self.nu_max
        return torch.randn(n_atoms, descriptor_dim)  # 占位


class DeltaLearningModel(nn.Module):
    """
    Δ-学习神经网络模型
    """
    
    def __init__(self, config: DeltaLearningConfig, descriptor_dim: int):
        super().__init__()
        self.config = config
        
        # 特征网络 (共享)
        self.feature_network = self._build_feature_network(descriptor_dim)
        
        # 输出头
        if config.predict_energy:
            self.energy_head = self._build_output_head(1)
        
        if config.predict_forces:
            self.force_head = self._build_output_head(3)
        
        if config.predict_stress:
            self.stress_head = self._build_output_head(6)  # 对称应力张量
        
        self._init_weights()
    
    def _build_feature_network(self, input_dim: int) -> nn.Module:
        """构建特征网络"""
        layers = []
        dims = [input_dim] + self.config.hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            
            if self.config.use_batch_norm:
                layers.append(nn.BatchNorm1d(dims[i+1]))
            
            layers.append(self._get_activation())
            
            if self.config.dropout_rate > 0:
                layers.append(nn.Dropout(self.config.dropout_rate))
        
        return nn.Sequential(*layers)
    
    def _build_output_head(self, output_dim: int) -> nn.Module:
        """构建输出头"""
        return nn.Sequential(
            nn.Linear(self.config.hidden_dims[-1], self.config.hidden_dims[-1] // 2),
            self._get_activation(),
            nn.Linear(self.config.hidden_dims[-1] // 2, output_dim)
        )
    
    def _get_activation(self):
        """获取激活函数"""
        activations = {
            'relu': nn.ReLU(),
            'silu': nn.SiLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
            'elu': nn.ELU(),
        }
        return activations.get(self.config.activation, nn.SiLU())
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, descriptors: torch.Tensor, 
                compute_forces: bool = False,
                positions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            descriptors: 原子描述符 [n_atoms, descriptor_dim]
            compute_forces: 是否计算力
            positions: 原子位置 (用于力计算)
            
        Returns:
            {
                'energy_delta': 能量修正,
                'forces_delta': 力修正 (如果计算),
                'stress_delta': 应力修正 (如果计算)
            }
        """
        # 特征提取
        features = self.feature_network(descriptors)
        
        # 聚合特征 (sum pooling)
        aggregated = torch.sum(features, dim=0, keepdim=True)
        
        result = {}
        
        # 能量修正
        if self.config.predict_energy:
            energy_delta = self.energy_head(aggregated).squeeze()
            result['energy_delta'] = energy_delta
        
        # 力修正
        if compute_forces and self.config.predict_forces and positions is not None:
            positions.requires_grad_(True)
            # 力的计算需要通过位置的梯度
            # 这里简化处理
            forces_delta = torch.zeros(positions.shape)
            result['forces_delta'] = forces_delta
        
        # 应力修正
        if self.config.predict_stress:
            stress_delta = self.stress_head(aggregated).squeeze()
            result['stress_delta'] = stress_delta
        
        return result


class DeltaLearningLoss(nn.Module):
    """Δ-学习损失函数"""
    
    def __init__(self, config: DeltaLearningConfig):
        super().__init__()
        self.config = config
    
    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算总损失
        
        Args:
            predictions: 模型预测
            targets: 目标值
            
        Returns:
            总损失, 各分项损失字典
        """
        total_loss = 0.0
        loss_dict = {}
        
        # 能量损失
        if 'energy_delta' in predictions and 'energy_delta' in targets:
            energy_loss = F.mse_loss(
                predictions['energy_delta'], 
                targets['energy_delta']
            )
            total_loss += self.config.energy_weight * energy_loss
            loss_dict['energy'] = energy_loss.item()
        
        # 力损失
        if 'forces_delta' in predictions and 'forces_delta' in targets:
            force_loss = F.mse_loss(
                predictions['forces_delta'],
                targets['forces_delta']
            )
            total_loss += self.config.force_weight * force_loss
            loss_dict['forces'] = force_loss.item()
        
        # 应力损失
        if 'stress_delta' in predictions and 'stress_delta' in targets:
            stress_loss = F.mse_loss(
                predictions['stress_delta'],
                targets['stress_delta']
            )
            total_loss += self.config.stress_weight * stress_loss
            loss_dict['stress'] = stress_loss.item()
        
        # 平滑性正则化
        if self.config.smoothness_weight > 0:
            smoothness_loss = self._compute_smoothness(predictions)
            total_loss += self.config.smoothness_weight * smoothness_loss
            loss_dict['smoothness'] = smoothness_loss.item()
        
        return total_loss, loss_dict
    
    def _compute_smoothness(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算平滑性正则化"""
        # 鼓励小的修正值 (物理上修正应该相对较小)
        smoothness = 0.0
        for key, value in predictions.items():
            smoothness += torch.mean(value**2)
        return smoothness


class DeltaLearningDataset(Dataset):
    """Δ-学习数据集"""
    
    def __init__(self, structures: List[Dict], descriptor_calculator):
        """
        Args:
            structures: 结构列表，每个结构包含:
                {
                    'positions': 原子位置,
                    'atom_types': 原子类型,
                    'cell': 晶胞 (可选),
                    'energy_low': 低精度能量,
                    'energy_high': 高精度能量,
                    'forces_low': 低精度力,
                    'forces_high': 高精度力,
                    ...
                }
            descriptor_calculator: 描述符计算器
        """
        self.structures = structures
        self.descriptor_calculator = descriptor_calculator
    
    def __len__(self) -> int:
        return len(self.structures)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        struct = self.structures[idx]
        
        # 计算描述符
        positions = torch.from_numpy(struct['positions']).float()
        atom_types = torch.from_numpy(struct['atom_types']).long()
        cell = None
        if 'cell' in struct:
            cell = torch.from_numpy(struct['cell']).float()
        
        descriptors = self.descriptor_calculator(positions, atom_types, cell)
        
        # 构建样本
        sample = {
            'descriptors': descriptors,
            'positions': positions,
            'atom_types': atom_types,
        }
        
        # Δ-学习标签
        if 'energy_high' in struct and 'energy_low' in struct:
            sample['energy_delta'] = torch.tensor(
                struct['energy_high'] - struct['energy_low']
            ).float()
        
        if 'forces_high' in struct and 'forces_low' in struct:
            sample['forces_delta'] = torch.from_numpy(
                struct['forces_high'] - struct['forces_low']
            ).float()
        
        if 'stress_high' in struct and 'stress_low' in struct:
            sample['stress_delta'] = torch.from_numpy(
                struct['stress_high'] - struct['stress_low']
            ).float()
        
        return sample


class DeltaLearningTrainer:
    """Δ-学习训练器"""
    
    def __init__(self, model: DeltaLearningModel, config: DeltaLearningConfig,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.loss_fn = DeltaLearningLoss(config)
        
        # 优化器
        if config.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=config.weight_decay
            )
        
        # 学习率调度器
        if config.scheduler == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=config.learning_rate_decay,
                patience=config.learning_rate_patience
            )
        elif config.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.max_epochs
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.learning_rate_patience,
                gamma=config.learning_rate_decay
            )
        
        self.history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_losses = defaultdict(list)
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            # 处理batch数据
            # 注意：这里需要处理变长序列
            descriptors = batch['descriptors'][0].to(self.device)
            
            # 前向传播
            predictions = self.model(descriptors)
            
            # 准备目标
            targets = {}
            if 'energy_delta' in batch:
                targets['energy_delta'] = batch['energy_delta'][0].to(self.device)
            if 'forces_delta' in batch:
                targets['forces_delta'] = batch['forces_delta'][0].to(self.device)
            
            # 计算损失
            loss, loss_dict = self.loss_fn(predictions, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            for k, v in loss_dict.items():
                epoch_losses[k].append(v)
            epoch_losses['total'].append(loss.item())
        
        return {k: np.mean(v) for k, v in epoch_losses.items()}
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        val_losses = defaultdict(list)
        
        with torch.no_grad():
            for batch in dataloader:
                descriptors = batch['descriptors'][0].to(self.device)
                predictions = self.model(descriptors)
                
                targets = {}
                if 'energy_delta' in batch:
                    targets['energy_delta'] = batch['energy_delta'][0].to(self.device)
                if 'forces_delta' in batch:
                    targets['forces_delta'] = batch['forces_delta'][0].to(self.device)
                
                loss, loss_dict = self.loss_fn(predictions, targets)
                
                for k, v in loss_dict.items():
                    val_losses[k].append(v)
                val_losses['total'].append(loss.item())
        
        return {k: np.mean(v) for k, v in val_losses.items()}
    
    def train(self, train_dataset: DeltaLearningDataset,
              val_dataset: Optional[DeltaLearningDataset] = None,
              epochs: int = None) -> Dict:
        """完整训练流程"""
        epochs = epochs or self.config.max_epochs
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,  # 每次处理一个结构
            shuffle=True
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, batch_size=1)
        
        logger.info(f"开始Δ-学习训练，共{epochs}个epoch")
        
        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            
            for k, v in train_metrics.items():
                self.history[f'train_{k}'].append(v)
            
            # 验证
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                
                for k, v in val_metrics.items():
                    self.history[f'val_{k}'].append(v)
                
                val_loss = val_metrics['total']
                self.scheduler.step(val_loss)
                
                # 早停检查
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint('best_delta_model.pt')
                else:
                    self.patience_counter += 1
                
                if epoch % 10 == 0:
                    logger.info(
                        f"Epoch {epoch}: Train={train_metrics['total']:.6f}, "
                        f"Val={val_loss:.6f}"
                    )
                
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"早停触发于epoch {epoch}")
                    break
            else:
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train={train_metrics['total']:.6f}")
        
        return dict(self.history)
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': dict(self.history)
        }, path)
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = defaultdict(list, checkpoint.get('history', {}))


class DeltaLearningInterface:
    """
    Δ-学习主接口类
    """
    
    def __init__(self, config: Optional[DeltaLearningConfig] = None,
                 descriptor_type: str = 'soap'):
        self.config = config or DeltaLearningConfig()
        self.descriptor_type = descriptor_type
        
        # 初始化描述符计算器
        if descriptor_type == 'soap':
            self.descriptor_calculator = SOAPDescriptor()
        elif descriptor_type == 'ace':
            self.descriptor_calculator = ACEDescriptor()
        else:
            raise ValueError(f"未知的描述符类型: {descriptor_type}")
        
        # 初始化模型
        dummy_desc = self.descriptor_calculator(
            torch.randn(5, 3),
            torch.randint(0, 3, (5,))
        )
        descriptor_dim = dummy_desc.shape[1]
        
        self.model = DeltaLearningModel(self.config, descriptor_dim)
        
        # 转移学习设置
        if self.config.use_transfer_learning and self.config.pretrained_model_path:
            self._load_pretrained_model()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        self.trainer = None
    
    def _load_pretrained_model(self):
        """加载预训练模型进行转移学习"""
        checkpoint = torch.load(
            self.config.pretrained_model_path,
            map_location=self.device
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 冻结指定层
        for layer_idx in self.config.freeze_layers:
            for param in self.model.feature_network[layer_idx].parameters():
                param.requires_grad = False
        
        logger.info(f"已加载预训练模型: {self.config.pretrained_model_path}")
    
    def fit(self, structures: List[Dict],
            validation_split: float = 0.1) -> Dict:
        """
        训练模型
        
        Args:
            structures: 训练结构列表
            validation_split: 验证集比例
            
        Returns:
            训练历史
        """
        # 分割数据集
        n_total = len(structures)
        n_val = int(n_total * validation_split)
        n_train = n_total - n_val
        
        indices = np.random.permutation(n_total)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        train_structures = [structures[i] for i in train_indices]
        val_structures = [structures[i] for i in val_indices]
        
        train_dataset = DeltaLearningDataset(
            train_structures,
            self.descriptor_calculator
        )
        val_dataset = DeltaLearningDataset(
            val_structures,
            self.descriptor_calculator
        )
        
        # 训练
        self.trainer = DeltaLearningTrainer(self.model, self.config)
        history = self.trainer.train(train_dataset, val_dataset)
        
        return history
    
    def predict(self, structure: Dict) -> Dict[str, np.ndarray]:
        """
        预测修正值
        
        Args:
            structure: 结构字典
            
        Returns:
            {
                'energy_delta': 能量修正,
                'forces_delta': 力修正,
                'stress_delta': 应力修正
            }
        """
        self.model.eval()
        
        with torch.no_grad():
            positions = torch.from_numpy(structure['positions']).float()
            atom_types = torch.from_numpy(structure['atom_types']).long()
            cell = None
            if 'cell' in structure:
                cell = torch.from_numpy(structure['cell']).float()
            
            descriptors = self.descriptor_calculator(positions, atom_types, cell)
            descriptors = descriptors.to(self.device)
            
            predictions = self.model(descriptors)
            
            result = {}
            for key, value in predictions.items():
                result[key] = value.cpu().numpy()
            
            return result
    
    def correct_energy(self, structure: Dict, low_accuracy_energy: float) -> float:
        """
        修正能量
        
        Args:
            structure: 结构
            low_accuracy_energy: 低精度能量
            
        Returns:
            高精度能量估计
        """
        delta = self.predict(structure)
        energy_delta = delta.get('energy_delta', 0.0)
        
        if isinstance(energy_delta, np.ndarray):
            energy_delta = float(energy_delta)
        
        return low_accuracy_energy + energy_delta
    
    def correct_forces(self, structure: Dict, low_accuracy_forces: np.ndarray) -> np.ndarray:
        """修正力"""
        delta = self.predict(structure)
        forces_delta = delta.get('forces_delta', np.zeros_like(low_accuracy_forces))
        return low_accuracy_forces + forces_delta
    
    def save(self, path: str):
        """保存模型"""
        if self.trainer is None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'descriptor_type': self.descriptor_type
            }, path)
        else:
            self.trainer.save_checkpoint(path)
        logger.info(f"模型保存至: {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"模型加载自: {path}")


def create_delta_learning_pipeline(
    low_level_dft: str = 'pbe',
    high_level_method: str = 'ccsd_t',
    descriptor_type: str = 'soap'
) -> DeltaLearningInterface:
    """
    创建Δ-学习流程
    
    Args:
        low_level_dft: 低精度DFT方法 (pbe, lda, blyp等)
        high_level_method: 高精度方法 (ccsd_t, qmc, mp2等)
        descriptor_type: 描述符类型
        
    Returns:
        DeltaLearningInterface实例
    """
    config = DeltaLearningConfig()
    
    # 根据高低精度方法调整权重
    if high_level_method in ['ccsd_t', 'qmc']:
        config.energy_weight = 1.0
        config.force_weight = 100.0  # 高精度方法力更可靠
    
    interface = DeltaLearningInterface(config, descriptor_type)
    
    logger.info(
        f"创建Δ-学习流程: {low_level_dft} → {high_level_method} "
        f"(描述符: {descriptor_type})"
    )
    
    return interface


def transfer_learning_delta_model(
    pretrained_path: str,
    new_config: DeltaLearningConfig,
    freeze_layers: List[int] = [0, 1]
) -> DeltaLearningInterface:
    """
    使用转移学习创建新的Δ-学习模型
    
    Args:
        pretrained_path: 预训练模型路径
        new_config: 新配置
        freeze_layers: 要冻结的层索引
        
    Returns:
        新的DeltaLearningInterface实例
    """
    new_config.use_transfer_learning = True
    new_config.pretrained_model_path = pretrained_path
    new_config.freeze_layers = freeze_layers
    
    interface = DeltaLearningInterface(new_config)
    
    logger.info(f"创建转移学习模型，冻结层: {freeze_layers}")
    
    return interface


# 导出
__all__ = [
    'DeltaLearningConfig',
    'SOAPDescriptor',
    'ACEDescriptor',
    'DeltaLearningModel',
    'DeltaLearningLoss',
    'DeltaLearningDataset',
    'DeltaLearningTrainer',
    'DeltaLearningInterface',
    'create_delta_learning_pipeline',
    'transfer_learning_delta_model',
]
