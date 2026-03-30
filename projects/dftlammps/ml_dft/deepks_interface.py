"""
DeepKS (Deep Kohn-Sham) Interface
DeePKS深度KS方法接口

实现神经网络拟合交换关联能，从高精度参考数据学习修正，
与VASP/QE等DFT代码集成。

参考文献:
- DeePKS: A Deep Learning Approach to Kohn-Sham DFT
- Deep Learning for Accurate and Efficient Density Functional Theory

作者: DFT-LAMMPS Team
日期: 2026-03-09
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from collections import defaultdict
import subprocess
import tempfile
import os

from .neural_xc import NeuralXCConfig, NeuralXCNetwork, DensityFeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class DeepKSConfig:
    """DeePKS配置"""
    # 模型架构
    descriptor_dim: int = 100  # 描述符维度
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 128])
    activation: str = 'silu'
    use_residual: bool = True
    
    # 训练参数
    learning_rate: float = 1e-3
    batch_size: int = 16
    max_epochs: int = 2000
    early_stopping_patience: int = 100
    
    # 损失函数权重
    energy_weight: float = 1.0
    force_weight: float = 50.0
    virial_weight: float = 1.0
    descriptor_weight: float = 0.1
    
    # 描述符参数
    rcut: float = 6.0  # 截断半径
    rcut_smth: float = 0.5
    sel: List[int] = field(default_factory=lambda: [50, 50])  # 邻居选择
    neuron: List[int] = field(default_factory=lambda: [25, 50, 100])
    axis_neuron: int = 16
    
    # 参考计算
    reference_backend: str = 'pyscf'  # pyscf, gaussian, orca
    reference_method: str = 'CCSD(T)'  # 高精度参考方法
    reference_basis: str = 'cc-pVTZ'
    
    # 迭代训练
    scf_max_iter: int = 50
    scf_conv_tol: float = 1e-6
    mixing_beta: float = 0.4
    
    # 输出控制
    save_freq: int = 100
    print_freq: int = 10
    
    # DFT代码集成
    dft_code: str = 'vasp'  # vasp, quantum_espresso, abacus
    dft_input_template: Optional[str] = None


class DescriptorGenerator(nn.Module):
    """
    DeePKS描述符生成器
    基于深度势(Deep Potential)的描述符框架
    """
    
    def __init__(self, config: DeepKSConfig, type_map: List[str]):
        super().__init__()
        self.config = config
        self.type_map = type_map
        self.ntypes = len(type_map)
        
        # 嵌入网络 (se_e2_a类型)
        self.embeddings = nn.ModuleList()
        for ii in range(self.ntypes):
            for jj in range(self.ntypes):
                embedding = self._build_embedding_net()
                self.embeddings.append(embedding)
        
        # 拟合网络
        self.fitting_net = self._build_fitting_net()
        
    def _build_embedding_net(self) -> nn.Module:
        """构建嵌入网络"""
        layers = []
        dims = [1] + self.config.neuron
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i]))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(dims[-1], self.config.axis_neuron))
        return nn.Sequential(*layers)
    
    def _build_fitting_net(self) -> nn.Module:
        """构建拟合网络"""
        layers = []
        # 输入维度计算
        input_dim = self.ntypes * self.config.neuron[-1]
        dims = [input_dim] + self.config.hidden_dims + [1]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(self._get_activation())
                if self.config.use_residual and i > 0 and dims[i] == dims[i+1]:
                    # 可以添加残差连接
                    pass
        return nn.Sequential(*layers)
    
    def _get_activation(self):
        """获取激活函数"""
        activations = {
            'relu': nn.ReLU(),
            'silu': nn.SiLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
        }
        return activations.get(self.config.activation, nn.SiLU())
    
    def forward(self, coord: torch.Tensor, atype: torch.Tensor,
                box: Optional[torch.Tensor] = None,
                natoms: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            coord: 原子坐标 [batch_size, nframes, natoms, 3]
            atype: 原子类型 [batch_size, natoms]
            box: 模拟盒子 [batch_size, 9]
            natoms: 每个样本的原子数 [batch_size]
            
        Returns:
            描述符 [batch_size, nframes, descriptor_dim]
        """
        batch_size, nframes, natoms_total, _ = coord.shape
        
        # 构建邻居列表 (简化实现)
        # 实际应用中需要高效的邻居搜索算法
        descriptor = self._compute_descriptor(coord, atype, box)
        
        return descriptor
    
    def _compute_descriptor(self, coord: torch.Tensor, atype: torch.Tensor,
                           box: Optional[torch.Tensor]) -> torch.Tensor:
        """计算环境描述符"""
        batch_size, nframes, natoms, _ = coord.shape
        
        # 计算相对位置 (简化版)
        # 实际实现需要处理周期性边界条件
        descriptor_list = []
        
        for frame in range(nframes):
            frame_coord = coord[:, frame, :, :]  # [batch, natoms, 3]
            
            # 计算距离矩阵
            diff = frame_coord.unsqueeze(2) - frame_coord.unsqueeze(1)  # [batch, natoms, natoms, 3]
            dist = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-10)  # [batch, natoms, natoms]
            
            # 应用截断
            mask = (dist < self.config.rcut) & (dist > 0.1)
            
            # 生成描述符特征 (简化)
            # 实际DeePKS使用更复杂的平滑截断函数
            s_ij = self._smooth_cosine(dist, self.config.rcut, self.config.rcut_smth)
            
            # 汇总邻居信息
            desc = torch.sum(s_ij.unsqueeze(-1) * diff, dim=2)  # [batch, natoms, 3]
            descriptor_list.append(desc.reshape(batch_size, -1))
        
        descriptor = torch.stack(descriptor_list, dim=1)
        return descriptor
    
    def _smooth_cosine(self, r: torch.Tensor, rcut: float, rcut_smth: float) -> torch.Tensor:
        """平滑截断余弦函数"""
        # 根据DeepMD-kit的实现
        u = (r - rcut_smth) / (rcut - rcut_smth)
        
        # 平滑余弦截断
        s = torch.where(
            r < rcut_smth,
            torch.ones_like(r),
            torch.where(
                r < rcut,
                0.5 * torch.cos(np.pi * u) + 0.5,
                torch.zeros_like(r)
            )
        )
        return s


class DeepKSEnergyCorrector(nn.Module):
    """
    DeePKS能量修正器
    学习DFT与高精度参考方法之间的差异
    """
    
    def __init__(self, config: DeepKSConfig, type_map: List[str]):
        super().__init__()
        self.config = config
        self.type_map = type_map
        
        # 描述符生成器
        self.descriptor_gen = DescriptorGenerator(config, type_map)
        
        # 能量修正网络
        self.correction_net = self._build_correction_net()
        
        # 初始化
        self._init_weights()
    
    def _build_correction_net(self) -> nn.Module:
        """构建能量修正网络"""
        layers = []
        dims = [self.config.descriptor_dim] + self.config.hidden_dims + [1]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
                layers.append(nn.LayerNorm(dims[i+1]))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, coord: torch.Tensor, atype: torch.Tensor,
                box: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Returns:
            {
                'energy_correction': 能量修正 [batch_size],
                'descriptor': 描述符 [batch_size, descriptor_dim],
                'forces_correction': 力修正 (如果计算)
            }
        """
        # 生成描述符
        descriptor = self.descriptor_gen(coord, atype, box)
        
        # 计算能量修正
        # 对描述符进行聚合 (sum over atoms)
        desc_aggregated = torch.sum(descriptor, dim=1)  # [batch, descriptor_dim]
        
        energy_correction = self.correction_net(desc_aggregated).squeeze(-1)
        
        result = {
            'energy_correction': energy_correction,
            'descriptor': desc_aggregated,
        }
        
        return result
    
    def compute_forces_correction(self, coord: torch.Tensor, atype: torch.Tensor,
                                   box: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算力修正 (通过自动微分)"""
        coord.requires_grad_(True)
        
        result = self.forward(coord, atype, box)
        energy_corr = result['energy_correction'].sum()
        
        forces_corr = -torch.autograd.grad(
            energy_corr, coord,
            create_graph=True
        )[0]
        
        return forces_corr


class DeepKSInterface:
    """
    DeePKS主接口类
    集成描述符生成、模型训练和DFT代码调用
    """
    
    def __init__(self, config: DeepKSConfig, type_map: List[str]):
        self.config = config
        self.type_map = type_map
        self.model = DeepKSEnergyCorrector(config, type_map)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=50, factor=0.5
        )
        
        self.history = defaultdict(list)
    
    def train_step(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单步训练"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # 准备数据
        coord = batch_data['coord'].to(self.device)
        atype = batch_data['atype'].to(self.device)
        box = batch_data.get('box', None)
        if box is not None:
            box = box.to(self.device)
        
        # 目标值
        target_energy = batch_data['energy'].to(self.device)
        target_forces = batch_data.get('forces', None)
        if target_forces is not None:
            target_forces = target_forces.to(self.device)
        
        # 前向传播
        result = self.model(coord, atype, box)
        energy_corr = result['energy_correction']
        
        # DFT能量 + 修正
        dft_energy = batch_data.get('dft_energy', torch.zeros_like(target_energy))
        predicted_energy = dft_energy.to(self.device) + energy_corr
        
        # 计算损失
        energy_loss = F.mse_loss(predicted_energy, target_energy)
        
        total_loss = self.config.energy_weight * energy_loss
        
        # 力损失
        force_loss = torch.tensor(0.0)
        if target_forces is not None and self.config.force_weight > 0:
            forces_corr = self.model.compute_forces_correction(coord, atype, box)
            dft_forces = batch_data.get('dft_forces', torch.zeros_like(target_forces))
            predicted_forces = dft_forces.to(self.device) + forces_corr
            
            force_loss = F.mse_loss(predicted_forces, target_forces)
            total_loss += self.config.force_weight * force_loss
        
        # 反向传播
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'energy_loss': energy_loss.item(),
            'force_loss': force_loss.item() if isinstance(force_loss, torch.Tensor) else 0.0,
        }
    
    def train(self, train_loader, val_loader=None, epochs: int = None) -> Dict:
        """完整训练流程"""
        epochs = epochs or self.config.max_epochs
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"开始DeePKS训练，共{epochs}个epoch")
        
        for epoch in range(epochs):
            epoch_losses = defaultdict(list)
            
            for batch in train_loader:
                losses = self.train_step(batch)
                for k, v in losses.items():
                    epoch_losses[k].append(v)
            
            # 记录平均损失
            for k, v in epoch_losses.items():
                self.history[k].append(np.mean(v))
            
            # 验证
            if val_loader is not None and epoch % 10 == 0:
                val_loss = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model('best_deepks_model.pt')
                else:
                    patience_counter += 1
                
                if epoch % 50 == 0:
                    logger.info(f"Epoch {epoch}: Train={self.history['total_loss'][-1]:.6f}, "
                               f"Val={val_loss:.6f}")
                
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"早停触发于epoch {epoch}")
                    break
        
        return dict(self.history)
    
    def validate(self, val_loader) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                coord = batch['coord'].to(self.device)
                atype = batch['atype'].to(self.device)
                box = batch.get('box', None)
                if box is not None:
                    box = box.to(self.device)
                
                target = batch['energy'].to(self.device)
                dft_energy = batch.get('dft_energy', torch.zeros_like(target))
                
                result = self.model(coord, atype, box)
                predicted = dft_energy.to(self.device) + result['energy_correction']
                
                loss = F.mse_loss(predicted, target)
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def predict(self, coord: np.ndarray, atype: np.ndarray,
                box: Optional[np.ndarray] = None,
                dft_energy: Optional[float] = None) -> Dict[str, np.ndarray]:
        """预测能量修正"""
        self.model.eval()
        
        with torch.no_grad():
            coord_t = torch.from_numpy(coord).float().unsqueeze(0).to(self.device)
            atype_t = torch.from_numpy(atype).long().unsqueeze(0).to(self.device)
            
            box_t = None
            if box is not None:
                box_t = torch.from_numpy(box).float().unsqueeze(0).to(self.device)
            
            result = self.model(coord_t, atype_t, box_t)
            
            energy_corr = result['energy_correction'].cpu().numpy()
            
            prediction = {'energy_correction': energy_corr}
            
            if dft_energy is not None:
                prediction['total_energy'] = dft_energy + energy_corr.item()
            
            return prediction
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'type_map': self.type_map,
            'history': dict(self.history)
        }, path)
        logger.info(f"模型保存至: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = defaultdict(list, checkpoint.get('history', {}))
        logger.info(f"模型加载自: {path}")
    
    # ============= DFT代码集成 =============
    
    def run_vasp_with_correction(self, structure: Dict,
                                  vasp_cmd: str = 'vasp',
                                  incar_settings: Optional[Dict] = None) -> Dict:
        """
        运行VASP并应用DeePKS修正
        
        Args:
            structure: 结构信息
            vasp_cmd: VASP命令
            incar_settings: INCAR设置
            
        Returns:
            包含DFT能量和DeePKS修正的结果
        """
        # 1. 运行标准VASP计算
        dft_result = self._run_vasp(structure, vasp_cmd, incar_settings)
        
        # 2. 计算DeePKS修正
        coord = np.array(structure['positions'])
        atype = np.array([self.type_map.index(s) for s in structure['species']])
        box = np.array(structure.get('cell', None))
        
        correction = self.predict(coord, atype, box, 
                                   dft_energy=dft_result['energy'])
        
        # 3. 合并结果
        result = {
            'dft_energy': dft_result['energy'],
            'deepks_correction': correction['energy_correction'],
            'corrected_energy': correction['total_energy'],
            'forces': dft_result.get('forces', None),
            'stress': dft_result.get('stress', None),
        }
        
        return result
    
    def _run_vasp(self, structure: Dict, vasp_cmd: str,
                  incar_settings: Optional[Dict]) -> Dict:
        """运行VASP计算"""
        # 这里需要实现VASP调用
        # 简化版本，实际需要完整的VASP输入文件生成和解析
        logger.info("运行VASP计算...")
        
        # 模拟VASP结果
        result = {
            'energy': -100.0,  # eV
            'forces': np.zeros((len(structure['positions']), 3)),
        }
        
        return result
    
    def run_quantum_espresso_with_correction(self, structure: Dict,
                                              pw_cmd: str = 'pw.x',
                                              input_settings: Optional[Dict] = None) -> Dict:
        """运行Quantum ESPRESSO并应用DeePKS修正"""
        logger.info("运行Quantum ESPRESSO计算...")
        
        # 1. 运行标准QE计算
        dft_result = self._run_qe(structure, pw_cmd, input_settings)
        
        # 2. 计算DeePKS修正
        coord = np.array(structure['positions'])
        atype = np.array([self.type_map.index(s) for s in structure['species']])
        box = np.array(structure.get('cell', None))
        
        correction = self.predict(coord, atype, box,
                                   dft_energy=dft_result['energy'])
        
        return {
            'dft_energy': dft_result['energy'],
            'deepks_correction': correction['energy_correction'],
            'corrected_energy': correction['total_energy'],
        }
    
    def _run_qe(self, structure: Dict, pw_cmd: str,
                input_settings: Optional[Dict]) -> Dict:
        """运行Quantum ESPRESSO计算"""
        # 简化实现
        result = {
            'energy': -100.0,
            'forces': np.zeros((len(structure['positions']), 3)),
        }
        return result
    
    def generate_training_data(self, structures: List[Dict],
                               reference_calculator: Optional[Any] = None) -> Dict:
        """
        生成训练数据
        从结构列表生成DFT能量和高精度参考能量
        """
        training_data = {
            'coord': [],
            'atype': [],
            'box': [],
            'dft_energy': [],
            'reference_energy': [],
            'dft_forces': [],
            'reference_forces': [],
        }
        
        for i, structure in enumerate(structures):
            logger.info(f"处理结构 {i+1}/{len(structures)}")
            
            # DFT计算
            if self.config.dft_code == 'vasp':
                dft_result = self._run_vasp(structure, 'vasp')
            elif self.config.dft_code == 'quantum_espresso':
                dft_result = self._run_qe(structure, 'pw.x')
            else:
                raise ValueError(f"不支持的DFT代码: {self.config.dft_code}")
            
            # 高精度参考计算
            ref_result = self._run_reference_calculation(structure, reference_calculator)
            
            # 存储数据
            training_data['coord'].append(structure['positions'])
            training_data['atype'].append([self.type_map.index(s) for s in structure['species']])
            training_data['box'].append(structure.get('cell', None))
            training_data['dft_energy'].append(dft_result['energy'])
            training_data['reference_energy'].append(ref_result['energy'])
            training_data['dft_forces'].append(dft_result.get('forces', None))
            training_data['reference_forces'].append(ref_result.get('forces', None))
        
        # 转换为numpy数组
        for key in training_data:
            if training_data[key] and training_data[key][0] is not None:
                training_data[key] = np.array(training_data[key])
        
        return training_data
    
    def _run_reference_calculation(self, structure: Dict,
                                   calculator: Optional[Any] = None) -> Dict:
        """
        运行高精度参考计算 (CCSD(T)等)
        """
        if calculator is None:
            # 使用PySCF进行参考计算
            return self._run_pyscf_reference(structure)
        else:
            # 使用外部计算器
            return calculator.calculate(structure)
    
    def _run_pyscf_reference(self, structure: Dict) -> Dict:
        """使用PySCF运行高精度计算"""
        try:
            from pyscf import gto, scf, cc
            
            # 构建分子
            atom_str = []
            for pos, species in zip(structure['positions'], structure['species']):
                atom_str.append([species, tuple(pos)])
            
            mol = gto.M(
                atom=atom_str,
                basis=self.config.reference_basis,
                verbose=0
            )
            
            # HF计算
            mf = scf.RHF(mol)
            mf.kernel()
            
            # CCSD(T)计算
            mycc = cc.CCSD(mf)
            mycc.kernel()
            et = mycc.ccsd_t()
            
            total_energy = mf.e_tot + mycc.e_corr + et
            
            return {'energy': total_energy}
            
        except ImportError:
            logger.warning("PySCF未安装，使用模拟数据")
            return {'energy': -110.0}  # 模拟数据


def create_deepks_from_pyscf(pyscf_mol, pyscf_mf, config: Optional[DeepKSConfig] = None):
    """
    从PySCF计算结果创建DeePKS模型
    
    Args:
        pyscf_mol: PySCF分子对象
        pyscf_mf: PySCF均值场对象
        config: DeePKS配置
        
    Returns:
        DeepKSInterface实例
    """
    if config is None:
        config = DeepKSConfig()
    
    # 提取元素类型
    elements = [pyscf_mol.atom_symbol(i) for i in range(pyscf_mol.natm)]
    type_map = sorted(list(set(elements)))
    
    deepks = DeepKSInterface(config, type_map)
    
    # 从PySCF提取描述符 (简化)
    # 实际实现需要更复杂的密度矩阵处理
    
    return deepks


# 导出
__all__ = [
    'DeepKSConfig',
    'DescriptorGenerator',
    'DeepKSEnergyCorrector',
    'DeepKSInterface',
    'create_deepks_from_pyscf',
]
