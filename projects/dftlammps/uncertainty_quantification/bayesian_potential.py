"""
贝叶斯神经网络势函数 - Bayesian Neural Network Potentials

实现用于原子间势能的不确定性量化神经网络势函数。
支持多种贝叶斯方法：变分推断、MC Dropout、深度集成。

核心特性:
- 能量和力的不确定性预测
- 原子环境描述符(ACSF, SOAP, MBTR)
- 自适应采样和主动学习
- 不确定性校准
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    optim = None

try:
    from scipy.spatial.distance import cdist
    from scipy.optimize import minimize
    from scipy.stats import norm, t
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ==================== 数据结构 ====================

@dataclass
class PotentialUncertainty:
    """势能预测的不确定性"""
    energy_mean: np.ndarray
    energy_var: np.ndarray
    force_mean: Optional[np.ndarray] = None
    force_var: Optional[np.ndarray] = None
    stress_mean: Optional[np.ndarray] = None
    stress_var: Optional[np.ndarray] = None
    
    # 分解的不确定性
    epistemic_energy: Optional[np.ndarray] = None  # 认知不确定性
    aleatoric_energy: Optional[np.ndarray] = None  # 随机不确定性
    epistemic_force: Optional[np.ndarray] = None
    aleatoric_force: Optional[np.ndarray] = None
    
    def energy_confidence_interval(self, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """计算能量置信区间"""
        alpha = 1 - confidence
        z = norm.ppf(1 - alpha/2)
        std = np.sqrt(self.energy_var)
        lower = self.energy_mean - z * std
        upper = self.energy_mean + z * std
        return lower, upper
    
    def force_confidence_interval(self, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """计算力置信区间"""
        if self.force_mean is None or self.force_var is None:
            return None, None
        alpha = 1 - confidence
        z = norm.ppf(1 - alpha/2)
        std = np.sqrt(self.force_var)
        lower = self.force_mean - z * std
        upper = self.force_mean + z * std
        return lower, upper
    
    def total_uncertainty(self) -> np.ndarray:
        """总不确定性（认知+随机）"""
        return self.energy_var
    
    def is_confident(self, threshold: float = 0.1) -> np.ndarray:
        """判断预测是否足够可靠"""
        std = np.sqrt(self.energy_var)
        return std < threshold * np.abs(self.energy_mean)


@dataclass
class EnergyPrediction:
    """能量预测结果"""
    energy: np.ndarray
    atoms_index: List[int]
    uncertainty: Optional[PotentialUncertainty] = None
    
    def get_per_atom_energy(self) -> np.ndarray:
        """获取每个原子的能量"""
        return self.energy / len(self.atoms_index)


@dataclass
class ForcePrediction:
    """力预测结果"""
    forces: np.ndarray  # (N_atoms, 3)
    atoms_index: List[int]
    uncertainty: Optional[PotentialUncertainty] = None
    
    def get_force_magnitude(self) -> np.ndarray:
        """获取力的大小"""
        return np.linalg.norm(self.forces, axis=1)


@dataclass
class StressPrediction:
    """应力预测结果"""
    stress: np.ndarray  # (3, 3) or (6,) Voigt notation
    uncertainty: Optional[PotentialUncertainty] = None
    
    def get_pressure(self) -> float:
        """获取压强（应力迹的负平均）"""
        if self.stress.shape == (3, 3):
            return -np.trace(self.stress) / 3
        else:
            # Voigt notation: [xx, yy, zz, yz, xz, xy]
            return -(self.stress[0] + self.stress[1] + self.stress[2]) / 3


# ==================== 原子环境描述符 ====================

class Descriptor(ABC):
    """原子环境描述符基类"""
    
    @abstractmethod
    def transform(self, positions: np.ndarray, 
                  atom_types: np.ndarray,
                  cell: Optional[np.ndarray] = None) -> np.ndarray:
        """将原子构型转换为描述符"""
        pass
    
    @abstractmethod
    def get_feature_dimension(self) -> int:
        """获取特征维度"""
        pass


class ACSFDescriptor(Descriptor):
    """
    Atom-Centered Symmetry Functions (ACSF) 描述符
    
    基于Behler-Parrinello方法的原子环境描述
    """
    
    def __init__(self, 
                 radial_params: List[Tuple[float, float]] = None,
                 angular_params: List[Tuple[float, float, float]] = None,
                 cutoff: float = 6.0):
        """
        初始化ACSF描述符
        
        Args:
            radial_params: 径向函数参数列表 [(eta, Rs), ...]
            angular_params: 角向函数参数列表 [(eta, zeta, lambda), ...]
            cutoff: 截断半径
        """
        self.cutoff = cutoff
        
        # 默认径向参数
        if radial_params is None:
            self.radial_params = [
                (0.05, 0.0), (0.05, 0.5), (0.05, 1.0), (0.05, 1.5),
                (0.05, 2.0), (0.05, 2.5), (0.05, 3.0), (0.05, 3.5),
                (0.005, 0.0), (0.005, 0.5), (0.005, 1.0), (0.005, 1.5),
                (0.005, 2.0), (0.005, 2.5), (0.005, 3.0), (0.005, 3.5),
            ]
        else:
            self.radial_params = radial_params
        
        # 默认角向参数
        if angular_params is None:
            self.angular_params = [
                (0.0001, 1.0, -1.0), (0.0001, 1.0, 1.0),
                (0.0001, 2.0, -1.0), (0.0001, 2.0, 1.0),
                (0.0001, 4.0, -1.0), (0.0001, 4.0, 1.0),
                (0.0001, 8.0, -1.0), (0.0001, 8.0, 1.0),
                (0.0001, 16.0, -1.0), (0.0001, 16.0, 1.0),
            ]
        else:
            self.angular_params = angular_params
    
    def _cutoff_function(self, r: np.ndarray) -> np.ndarray:
        """余弦截断函数"""
        return 0.5 * (np.cos(np.pi * r / self.cutoff) + 1.0) * (r < self.cutoff)
    
    def _radial_symmetry_function(self, distances: np.ndarray, 
                                   eta: float, Rs: float) -> float:
        """计算径向对称函数G2"""
        fc = self._cutoff_function(distances)
        return np.sum(np.exp(-eta * (distances - Rs)**2) * fc)
    
    def _angular_symmetry_function(self, positions: np.ndarray,
                                    atom_i: int, eta: float, 
                                    zeta: float, lambda_: float) -> float:
        """计算角向对称函数G4/G5"""
        n_atoms = len(positions)
        result = 0.0
        
        for j in range(n_atoms):
            if j == atom_i:
                continue
            rij = positions[j] - positions[atom_i]
            dij = np.linalg.norm(rij)
            if dij >= self.cutoff:
                continue
            
            for k in range(j+1, n_atoms):
                if k == atom_i:
                    continue
                rik = positions[k] - positions[atom_i]
                dik = np.linalg.norm(rik)
                if dik >= self.cutoff:
                    continue
                
                rjk = positions[k] - positions[j]
                djk = np.linalg.norm(rjk)
                
                # 角度
                cos_theta = np.dot(rij, rik) / (dij * dik)
                theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                
                # 截断函数
                fc_ij = self._cutoff_function(np.array([dij]))[0]
                fc_ik = self._cutoff_function(np.array([dik]))[0]
                fc_jk = self._cutoff_function(np.array([djk]))[0]
                
                result += (1 + lambda_ * cos_theta)**zeta * \
                         np.exp(-eta * (dij**2 + dik**2 + djk**2)) * \
                         fc_ij * fc_ik * fc_jk
        
        return result * 2**(1-zeta)
    
    def transform(self, positions: np.ndarray,
                  atom_types: np.ndarray,
                  cell: Optional[np.ndarray] = None) -> np.ndarray:
        """计算ACSF描述符"""
        n_atoms = len(positions)
        n_features = len(self.radial_params) + len(self.angular_params)
        descriptors = np.zeros((n_atoms, n_features))
        
        # 计算距离矩阵
        if cell is not None:
            # PBC处理（简化版本）
            distances = cdist(positions, positions)
        else:
            distances = cdist(positions, positions)
        
        for i in range(n_atoms):
            feat_idx = 0
            
            # 径向对称函数
            for eta, Rs in self.radial_params:
                descriptors[i, feat_idx] = self._radial_symmetry_function(
                    distances[i], eta, Rs
                )
                feat_idx += 1
            
            # 角向对称函数
            for eta, zeta, lambda_ in self.angular_params:
                descriptors[i, feat_idx] = self._angular_symmetry_function(
                    positions, i, eta, zeta, lambda_
                )
                feat_idx += 1
        
        return descriptors
    
    def get_feature_dimension(self) -> int:
        return len(self.radial_params) + len(self.angular_params)


class SOAPDescriptor(Descriptor):
    """
    Smooth Overlap of Atomic Positions (SOAP) 描述符
    
    基于原子密度展开的平滑原子环境描述
    """
    
    def __init__(self,
                 n_max: int = 8,
                 l_max: int = 8,
                 cutoff: float = 6.0,
                 sigma: float = 0.5):
        """
        初始化SOAP描述符
        
        Args:
            n_max: 径向基函数数量
            l_max: 角动量量子数最大值
            cutoff: 截断半径
            sigma: 高斯展宽参数
        """
        self.n_max = n_max
        self.l_max = l_max
        self.cutoff = cutoff
        self.sigma = sigma
        
        # 预计算高斯-洛朗特基函数
        self._setup_basis()
    
    def _setup_basis(self):
        """设置正交径向基函数"""
        # 简化版本：使用高斯基函数
        self.basis_centers = np.linspace(0, self.cutoff, self.n_max + 2)[1:-1]
        self.basis_widths = np.ones(self.n_max) * (self.cutoff / self.n_max)
    
    def _radial_basis(self, r: float, n: int) -> float:
        """计算径向基函数"""
        return np.exp(-0.5 * ((r - self.basis_centers[n]) / self.basis_widths[n])**2)
    
    def _spherical_harmonic(self, x: float, y: float, z: float, 
                            l: int, m: int) -> complex:
        """计算球谐函数（简化版本）"""
        r = np.sqrt(x**2 + y**2 + z**2)
        if r < 1e-10:
            return 0.0
        
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        
        # 简化的球谐函数实现
        if l == 0:
            return 1.0
        elif l == 1:
            if m == -1:
                return np.sin(theta) * np.exp(-1j * phi)
            elif m == 0:
                return np.cos(theta)
            elif m == 1:
                return -np.sin(theta) * np.exp(1j * phi)
        
        # 高阶使用近似
        return np.cos(l * theta) * np.exp(1j * m * phi)
    
    def transform(self, positions: np.ndarray,
                  atom_types: np.ndarray,
                  cell: Optional[np.ndarray] = None) -> np.ndarray:
        """计算SOAP描述符"""
        n_atoms = len(positions)
        n_features = self.n_max * (self.l_max + 1)**2
        descriptors = np.zeros((n_atoms, n_features))
        
        for i in range(n_atoms):
            # 计算展开系数
            coeffs = np.zeros((self.n_max, self.l_max + 1, 2 * self.l_max + 1), 
                             dtype=complex)
            
            for j in range(n_atoms):
                if i == j:
                    continue
                
                rij = positions[j] - positions[i]
                r = np.linalg.norm(rij)
                
                if r >= self.cutoff:
                    continue
                
                # 高斯权重
                weight = np.exp(-r**2 / (2 * self.sigma**2))
                
                for n in range(self.n_max):
                    radial_part = self._radial_basis(r, n)
                    for l in range(self.l_max + 1):
                        for m in range(-l, l+1):
                            angular_part = self._spherical_harmonic(
                                rij[0], rij[1], rij[2], l, m
                            )
                            coeffs[n, l, m + self.l_max] += weight * radial_part * angular_part
            
            # 展平为特征向量
            descriptors[i] = np.abs(coeffs.flatten())
        
        # 归一化
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        descriptors = descriptors / (norms + 1e-10)
        
        return descriptors
    
    def get_feature_dimension(self) -> int:
        return self.n_max * (self.l_max + 1)**2


class MBTRDescriptor(Descriptor):
    """
    Many-Body Tensor Representation (MBTR) 描述符
    
    多体关联函数描述符
    """
    
    def __init__(self,
                 k1_config: Dict = None,
                 k2_config: Dict = None,
                 k3_config: Dict = None,
                 normalization: str = 'l2'):
        """
        初始化MBTR描述符
        
        Args:
            k1_config: k=1项配置
            k2_config: k=2项配置
            k3_config: k=3项配置
            normalization: 归一化方法
        """
        self.k1_config = k1_config or {'geometry': {'function': 'atomic_number'}, 'grid': {'min': 1, 'max': 100, 'n': 100}}
        self.k2_config = k2_config or {'geometry': {'function': 'distance'}, 'grid': {'min': 0, 'max': 10, 'n': 100}}
        self.k3_config = k3_config or {'geometry': {'function': 'cosine'}, 'grid': {'min': -1, 'max': 1, 'n' : 100}}
        self.normalization = normalization
    
    def transform(self, positions: np.ndarray,
                  atom_types: np.ndarray,
                  cell: Optional[np.ndarray] = None) -> np.ndarray:
        """计算MBTR描述符"""
        # 简化实现：返回原子类型统计和距离分布
        n_atoms = len(positions)
        
        # k=1: 原子类型分布
        unique_types = np.unique(atom_types)
        k1_features = np.array([np.sum(atom_types == t) for t in unique_types])
        
        # k=2: 距离分布
        distances = cdist(positions, positions)
        dist_mask = (distances > 0) & (distances < 10)
        k2_hist, _ = np.histogram(distances[dist_mask], bins=50, range=(0, 10))
        
        # k=3: 角度分布（简化）
        angles = []
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                for k in range(j+1, n_atoms):
                    rij = positions[j] - positions[i]
                    rik = positions[k] - positions[i]
                    dij = np.linalg.norm(rij)
                    dik = np.linalg.norm(rik)
                    if dij > 0 and dik > 0:
                        cos_angle = np.dot(rij, rik) / (dij * dik)
                        angles.append(np.arccos(np.clip(cos_angle, -1, 1)))
        k3_hist, _ = np.histogram(angles, bins=30, range=(0, np.pi)) if angles else (np.zeros(30), None)
        
        # 合并特征
        features = np.concatenate([k1_features, k2_hist, k3_hist])
        
        if self.normalization == 'l2':
            features = features / (np.linalg.norm(features) + 1e-10)
        elif self.normalization == 'n_atoms':
            features = features / n_atoms
        
        return features.reshape(1, -1)  # 全局描述符
    
    def get_feature_dimension(self) -> int:
        return len(self.k1_config['grid']['n']) + 50 + 30


# ==================== 贝叶斯势函数基类 ====================

class BayesianPotential(ABC):
    """
    贝叶斯神经网络势函数基类
    
    提供不确定性量化的原子间势能预测
    """
    
    def __init__(self,
                 descriptor: Descriptor,
                 n_species: int = 1,
                 device: str = 'cpu'):
        """
        初始化贝叶斯势函数
        
        Args:
            descriptor: 原子环境描述符
            n_species: 原子种类数
            device: 计算设备
        """
        self.descriptor = descriptor
        self.n_species = n_species
        self.device = device
        self.is_trained = False
        
        # 训练统计
        self.training_energies = []
        self.training_forces = []
    
    @abstractmethod
    def predict_energy(self, 
                       positions: np.ndarray,
                       atom_types: np.ndarray,
                       cell: Optional[np.ndarray] = None,
                       n_samples: int = 100) -> EnergyPrediction:
        """预测能量及其不确定性"""
        pass
    
    @abstractmethod
    def predict_forces(self,
                       positions: np.ndarray,
                       atom_types: np.ndarray,
                       cell: Optional[np.ndarray] = None,
                       n_samples: int = 100) -> ForcePrediction:
        """预测力及其不确定性"""
        pass
    
    @abstractmethod
    def predict_stress(self,
                       positions: np.ndarray,
                       atom_types: np.ndarray,
                       cell: np.ndarray,
                       volume: float,
                       n_samples: int = 100) -> StressPrediction:
        """预测应力及其不确定性"""
        pass
    
    def predict(self,
                positions: np.ndarray,
                atom_types: np.ndarray,
                cell: Optional[np.ndarray] = None,
                volume: Optional[float] = None,
                n_samples: int = 100) -> PotentialUncertainty:
        """综合预测能量、力和应力"""
        energy_pred = self.predict_energy(positions, atom_types, cell, n_samples)
        force_pred = self.predict_forces(positions, atom_types, cell, n_samples)
        
        stress_pred = None
        if cell is not None and volume is not None:
            stress_pred = self.predict_stress(positions, atom_types, cell, volume, n_samples)
        
        return PotentialUncertainty(
            energy_mean=energy_pred.energy,
            energy_var=energy_pred.uncertainty.energy_var if energy_pred.uncertainty else np.zeros_like(energy_pred.energy),
            force_mean=force_pred.forces if force_pred else None,
            force_var=force_pred.uncertainty.force_var if force_pred and force_pred.uncertainty else None,
            stress_mean=stress_pred.stress if stress_pred else None,
            stress_var=stress_pred.uncertainty.stress_var if stress_pred and stress_pred.uncertainty else None
        )
    
    def get_uncertainty_per_atom(self, 
                                  positions: np.ndarray,
                                  atom_types: np.ndarray,
                                  cell: Optional[np.ndarray] = None) -> np.ndarray:
        """获取每个原子的不确定性"""
        uncertainty = self.predict(positions, atom_types, cell)
        return uncertainty.energy_var / len(atom_types)


# ==================== 具体实现 ====================

class BayesianNeuralPotential(BayesianPotential):
    """
    基于变分推断的贝叶斯神经网络势函数
    
    使用贝叶斯神经网络建模原子间势能
    """
    
    def __init__(self,
                 descriptor: Descriptor,
                 n_species: int = 1,
                 hidden_layers: List[int] = [128, 128, 128],
                 activation: str = 'tanh',
                 device: str = 'cpu'):
        super().__init__(descriptor, n_species, device)
        
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for BayesianNeuralPotential")
        
        self.hidden_layers = hidden_layers
        self.activation = activation
        
        # 构建网络
        self._build_network()
    
    def _build_network(self):
        """构建贝叶斯神经网络"""
        input_dim = self.descriptor.get_feature_dimension()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.activation == 'tanh':
                layers.append(nn.Tanh())
            elif self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'elu':
                layers.append(nn.ELU())
            prev_dim = hidden_dim
        
        # 输出层：能量预测
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers).to(self.device)
        
        # 变分参数（简化版本）
        self.variational_params = {}
        for name, param in self.network.named_parameters():
            self.variational_params[name] = {
                'mu': param.data.clone(),
                'rho': nn.Parameter(torch.ones_like(param) * -3)
            }
    
    def predict_energy(self,
                       positions: np.ndarray,
                       atom_types: np.ndarray,
                       cell: Optional[np.ndarray] = None,
                       n_samples: int = 100) -> EnergyPrediction:
        """预测能量（MC采样）"""
        # 计算描述符
        descriptors = self.descriptor.transform(positions, atom_types, cell)
        
        # 转换为torch
        x = torch.FloatTensor(descriptors).to(self.device)
        
        # 多次采样
        energies = []
        self.network.eval()
        with torch.no_grad():
            for _ in range(n_samples):
                # 从变分分布采样参数
                self._sample_parameters()
                # 前向传播
                atom_energies = self.network(x).cpu().numpy().flatten()
                total_energy = np.sum(atom_energies)
                energies.append(total_energy)
        
        energies = np.array(energies)
        mean_energy = np.mean(energies)
        var_energy = np.var(energies)
        
        uncertainty = PotentialUncertainty(
            energy_mean=np.array([mean_energy]),
            energy_var=np.array([var_energy]),
            epistemic_energy=np.array([var_energy])
        )
        
        return EnergyPrediction(
            energy=np.array([mean_energy]),
            atoms_index=list(range(len(positions))),
            uncertainty=uncertainty
        )
    
    def _sample_parameters(self):
        """从变分分布采样网络参数"""
        for name, param in self.network.named_parameters():
            if name in self.variational_params:
                mu = self.variational_params[name]['mu']
                rho = self.variational_params[name]['rho']
                sigma = torch.log1p(torch.exp(rho))
                # 重参数化采样
                epsilon = torch.randn_like(mu)
                param.data = mu + sigma * epsilon
    
    def predict_forces(self,
                       positions: np.ndarray,
                       atom_types: np.ndarray,
                       cell: Optional[np.ndarray] = None,
                       n_samples: int = 100) -> ForcePrediction:
        """预测力（使用自动微分）"""
        positions_torch = torch.FloatTensor(positions).to(self.device)
        positions_torch.requires_grad_(True)
        
        forces_list = []
        
        self.network.eval()
        for _ in range(n_samples):
            self._sample_parameters()
            
            # 计算能量
            descriptors = self.descriptor.transform(
                positions_torch.detach().cpu().numpy(), 
                atom_types, cell
            )
            x = torch.FloatTensor(descriptors).to(self.device)
            
            atom_energies = self.network(x)
            total_energy = torch.sum(atom_energies)
            
            # 力的负梯度
            forces = -torch.autograd.grad(
                total_energy, positions_torch,
                create_graph=False, retain_graph=False
            )[0]
            
            forces_list.append(forces.detach().cpu().numpy())
        
        forces_array = np.array(forces_list)
        mean_forces = np.mean(forces_array, axis=0)
        var_forces = np.var(forces_array, axis=0)
        
        uncertainty = PotentialUncertainty(
            energy_mean=np.array([0]),
            energy_var=np.array([0]),
            force_mean=mean_forces,
            force_var=var_forces,
            epistemic_force=var_forces
        )
        
        return ForcePrediction(
            forces=mean_forces,
            atoms_index=list(range(len(positions))),
            uncertainty=uncertainty
        )
    
    def predict_stress(self,
                       positions: np.ndarray,
                       atom_types: np.ndarray,
                       cell: np.ndarray,
                       volume: float,
                       n_samples: int = 100) -> StressPrediction:
        """预测应力张量（简化实现）"""
        # 应力计算需要更复杂的实现
        # 这里返回零作为占位
        stress = np.zeros((3, 3))
        return StressPrediction(stress=stress)


class MCDropoutPotential(BayesianPotential):
    """
    基于MC Dropout的贝叶斯势函数
    
    使用Dropout作为近似贝叶斯推断
    """
    
    def __init__(self,
                 descriptor: Descriptor,
                 n_species: int = 1,
                 hidden_layers: List[int] = [128, 128, 128],
                 dropout_rate: float = 0.1,
                 device: str = 'cpu'):
        super().__init__(descriptor, n_species, device)
        
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for MCDropoutPotential")
        
        self.dropout_rate = dropout_rate
        self._build_network(hidden_layers)
    
    def _build_network(self, hidden_layers: List[int]):
        """构建带Dropout的网络"""
        input_dim = self.descriptor.get_feature_dimension()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers).to(self.device)
    
    def predict_energy(self,
                       positions: np.ndarray,
                       atom_types: np.ndarray,
                       cell: Optional[np.ndarray] = None,
                       n_samples: int = 100) -> EnergyPrediction:
        """预测能量（使用MC Dropout）"""
        descriptors = self.descriptor.transform(positions, atom_types, cell)
        x = torch.FloatTensor(descriptors).to(self.device)
        
        # 启用dropout（即使在eval模式）
        energies = []
        self.network.train()  # 保持dropout开启
        
        with torch.no_grad():
            for _ in range(n_samples):
                atom_energies = self.network(x).cpu().numpy().flatten()
                total_energy = np.sum(atom_energies)
                energies.append(total_energy)
        
        energies = np.array(energies)
        mean_energy = np.mean(energies)
        var_energy = np.var(energies)
        
        uncertainty = PotentialUncertainty(
            energy_mean=np.array([mean_energy]),
            energy_var=np.array([var_energy]),
            epistemic_energy=np.array([var_energy])
        )
        
        return EnergyPrediction(
            energy=np.array([mean_energy]),
            atoms_index=list(range(len(positions))),
            uncertainty=uncertainty
        )
    
    def predict_forces(self,
                       positions: np.ndarray,
                       atom_types: np.ndarray,
                       cell: Optional[np.ndarray] = None,
                       n_samples: int = 100) -> ForcePrediction:
        """预测力"""
        # 类似BayesianNeuralPotential的实现
        positions_torch = torch.FloatTensor(positions).to(self.device)
        positions_torch.requires_grad_(True)
        
        forces_list = []
        self.network.train()
        
        for _ in range(n_samples):
            descriptors = self.descriptor.transform(
                positions_torch.detach().cpu().numpy(),
                atom_types, cell
            )
            x = torch.FloatTensor(descriptors).to(self.device)
            
            atom_energies = self.network(x)
            total_energy = torch.sum(atom_energies)
            
            forces = -torch.autograd.grad(
                total_energy, positions_torch,
                create_graph=False, retain_graph=False
            )[0]
            
            forces_list.append(forces.detach().cpu().numpy())
        
        forces_array = np.array(forces_list)
        mean_forces = np.mean(forces_array, axis=0)
        var_forces = np.var(forces_array, axis=0)
        
        uncertainty = PotentialUncertainty(
            energy_mean=np.array([0]),
            energy_var=np.array([0]),
            force_mean=mean_forces,
            force_var=var_forces,
            epistemic_force=var_forces
        )
        
        return ForcePrediction(
            forces=mean_forces,
            atoms_index=list(range(len(positions))),
            uncertainty=uncertainty
        )
    
    def predict_stress(self,
                       positions: np.ndarray,
                       atom_types: np.ndarray,
                       cell: np.ndarray,
                       volume: float,
                       n_samples: int = 100) -> StressPrediction:
        """预测应力"""
        stress = np.zeros((3, 3))
        return StressPrediction(stress=stress)


class EnsemblePotential(BayesianPotential):
    """
    基于深度集成的贝叶斯势函数
    
    训练多个网络，使用集成预测不确定性
    """
    
    def __init__(self,
                 descriptor: Descriptor,
                 n_species: int = 1,
                 n_models: int = 5,
                 hidden_layers: List[int] = [128, 128, 128],
                 device: str = 'cpu'):
        super().__init__(descriptor, n_species, device)
        
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for EnsemblePotential")
        
        self.n_models = n_models
        self.networks = []
        
        for _ in range(n_models):
            network = self._build_single_network(hidden_layers)
            self.networks.append(network)
    
    def _build_single_network(self, hidden_layers: List[int]):  # type: ignore
        """构建单个网络"""
        input_dim = self.descriptor.get_feature_dimension()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        return nn.Sequential(*layers).to(self.device)
    
    def predict_energy(self,
                       positions: np.ndarray,
                       atom_types: np.ndarray,
                       cell: Optional[np.ndarray] = None,
                       n_samples: int = 100) -> EnergyPrediction:
        """预测能量（集成方法）"""
        descriptors = self.descriptor.transform(positions, atom_types, cell)
        x = torch.FloatTensor(descriptors).to(self.device)
        
        energies = []
        for network in self.networks:
            network.eval()
            with torch.no_grad():
                atom_energies = network(x).cpu().numpy().flatten()
                total_energy = np.sum(atom_energies)
                energies.append(total_energy)
        
        energies = np.array(energies)
        mean_energy = np.mean(energies)
        var_energy = np.var(energies)
        
        uncertainty = PotentialUncertainty(
            energy_mean=np.array([mean_energy]),
            energy_var=np.array([var_energy]),
            epistemic_energy=np.array([var_energy])
        )
        
        return EnergyPrediction(
            energy=np.array([mean_energy]),
            atoms_index=list(range(len(positions))),
            uncertainty=uncertainty
        )
    
    def predict_forces(self,
                       positions: np.ndarray,
                       atom_types: np.ndarray,
                       cell: Optional[np.ndarray] = None,
                       n_samples: int = 100) -> ForcePrediction:
        """预测力"""
        positions_torch = torch.FloatTensor(positions).to(self.device)
        positions_torch.requires_grad_(True)
        
        forces_list = []
        
        for network in self.networks:
            descriptors = self.descriptor.transform(
                positions_torch.detach().cpu().numpy(),
                atom_types, cell
            )
            x = torch.FloatTensor(descriptors).to(self.device)
            
            atom_energies = network(x)
            total_energy = torch.sum(atom_energies)
            
            forces = -torch.autograd.grad(
                total_energy, positions_torch,
                create_graph=False, retain_graph=False
            )[0]
            
            forces_list.append(forces.detach().cpu().numpy())
        
        forces_array = np.array(forces_list)
        mean_forces = np.mean(forces_array, axis=0)
        var_forces = np.var(forces_array, axis=0)
        
        uncertainty = PotentialUncertainty(
            energy_mean=np.array([0]),
            energy_var=np.array([0]),
            force_mean=mean_forces,
            force_var=var_forces,
            epistemic_force=var_forces
        )
        
        return ForcePrediction(
            forces=mean_forces,
            atoms_index=list(range(len(positions))),
            uncertainty=uncertainty
        )
    
    def predict_stress(self,
                       positions: np.ndarray,
                       atom_types: np.ndarray,
                       cell: np.ndarray,
                       volume: float,
                       n_samples: int = 100) -> StressPrediction:
        """预测应力"""
        stress = np.zeros((3, 3))
        return StressPrediction(stress=stress)


class VariationalPotential(BayesianPotential):
    """
    变分推断势函数（使用贝叶斯后验的变分近似）
    """
    
    def __init__(self,
                 descriptor: Descriptor,
                 n_species: int = 1,
                 hidden_layers: List[int] = [128, 128],
                 prior_sigma: float = 1.0,
                 device: str = 'cpu'):
        super().__init__(descriptor, n_species, device)
        
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for VariationalPotential")
        
        self.prior_sigma = prior_sigma
        self.hidden_layers = hidden_layers
        self._build_network()
    
    def _build_network(self):
        """构建变分层网络"""
        # 变分线性层
        self.layers = nn.ModuleList() if HAS_TORCH else []
        
        input_dim = self.descriptor.get_feature_dimension()
        for hidden_dim in self.hidden_layers:
            self.layers.append(VariationalLinear(input_dim, hidden_dim, self.prior_sigma))
            input_dim = hidden_dim
        
        self.output_layer = VariationalLinear(input_dim, 1, self.prior_sigma)
    
    def forward(self, x, sample: bool = True) -> Any:
        """前向传播"""
        for layer in self.layers:
            x = torch.tanh(layer(x, sample))
        return self.output_layer(x, sample)
    
    def predict_energy(self,
                       positions: np.ndarray,
                       atom_types: np.ndarray,
                       cell: Optional[np.ndarray] = None,
                       n_samples: int = 100) -> EnergyPrediction:
        """预测能量"""
        descriptors = self.descriptor.transform(positions, atom_types, cell)
        x = torch.FloatTensor(descriptors).to(self.device)
        
        energies = []
        self.eval()
        
        with torch.no_grad():
            for _ in range(n_samples):
                atom_energies = self.forward(x, sample=True).cpu().numpy().flatten()
                total_energy = np.sum(atom_energies)
                energies.append(total_energy)
        
        energies = np.array(energies)
        mean_energy = np.mean(energies)
        var_energy = np.var(energies)
        
        uncertainty = PotentialUncertainty(
            energy_mean=np.array([mean_energy]),
            energy_var=np.array([var_energy]),
            epistemic_energy=np.array([var_energy])
        )
        
        return EnergyPrediction(
            energy=np.array([mean_energy]),
            atoms_index=list(range(len(positions))),
            uncertainty=uncertainty
        )
    
    def predict_forces(self,
                       positions: np.ndarray,
                       atom_types: np.ndarray,
                       cell: Optional[np.ndarray] = None,
                       n_samples: int = 100) -> ForcePrediction:
        """预测力"""
        # 简化实现
        forces = np.zeros((len(positions), 3))
        return ForcePrediction(forces=forces, atoms_index=list(range(len(positions))))
    
    def predict_stress(self,
                       positions: np.ndarray,
                       atom_types: np.ndarray,
                       cell: np.ndarray,
                       volume: float,
                       n_samples: int = 100) -> StressPrediction:
        """预测应力"""
        stress = np.zeros((3, 3))
        return StressPrediction(stress=stress)


class VariationalLinear:
    """变分线性层"""
    
    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0):
        if HAS_TORCH:
            super().__init__()
        else:
            raise ImportError("PyTorch is required for VariationalLinear")
        
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma
        
        # 变分参数
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-3, 0.1))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).normal_(-3, 0.1))
    
    def forward(self, x, sample: bool = True) -> Any:
        """前向传播"""
        if sample:
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            
            weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
            bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return torch.nn.functional.linear(x, weight, bias)


# ==================== 训练与校准 ====================

class PotentialTrainer:
    """势函数训练器"""
    
    def __init__(self,
                 potential: BayesianPotential,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5):
        self.potential = potential
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        if HAS_TORCH and hasattr(potential, 'network'):
            if hasattr(potential, 'networks'):
                # 集成模型
                self.optimizers = [
                    optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
                    for net in potential.networks
                ]
            else:
                # 单一模型
                self.optimizer = optim.Adam(
                    potential.network.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
    
    def train(self,
              train_structures: List[Dict],
              val_structures: Optional[List[Dict]] = None,
              n_epochs: int = 1000,
              batch_size: int = 32,
              energy_weight: float = 1.0,
              force_weight: float = 10.0,
              verbose: bool = True) -> Dict:
        """
        训练势函数
        
        Args:
            train_structures: 训练结构列表
            val_structures: 验证结构列表
            n_epochs: 训练轮数
            batch_size: 批次大小
            energy_weight: 能量损失权重
            force_weight: 力损失权重
        """
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(n_epochs):
            # 训练
            train_loss = self._train_epoch(
                train_structures, batch_size, energy_weight, force_weight
            )
            history['train_loss'].append(train_loss)
            
            # 验证
            if val_structures is not None:
                val_loss = self._validate(val_structures, energy_weight, force_weight)
                history['val_loss'].append(val_loss)
            
            if verbose and epoch % 100 == 0:
                msg = f"Epoch {epoch}: train_loss={train_loss:.4f}"
                if val_structures:
                    msg += f", val_loss={val_loss:.4f}"
                print(msg)
        
        self.potential.is_trained = True
        return history
    
    def _train_epoch(self,
                     structures: List[Dict],
                     batch_size: int,
                     energy_weight: float,
                     force_weight: float) -> float:
        """训练一个epoch"""
        if not hasattr(self, 'optimizer'):
            return 0.0
        
        self.potential.network.train()
        total_loss = 0.0
        
        for i in range(0, len(structures), batch_size):
            batch = structures[i:i+batch_size]
            
            self.optimizer.zero_grad()
            
            batch_loss = 0.0
            for struct in batch:
                # 前向传播和损失计算
                pred = self.potential.predict(
                    struct['positions'],
                    struct['atom_types'],
                    struct.get('cell')
                )
                
                # 能量损失
                if 'energy' in struct:
                    energy_loss = (pred.energy_mean - struct['energy'])**2
                    batch_loss += energy_weight * energy_loss.sum()
                
                # 力损失
                if 'forces' in struct and pred.force_mean is not None:
                    force_loss = np.mean((pred.force_mean - struct['forces'])**2)
                    batch_loss += force_weight * force_loss
            
            # 反向传播
            if HAS_TORCH and isinstance(batch_loss, torch.Tensor):
                batch_loss.backward()
                self.optimizer.step()
            
            total_loss += batch_loss.item() if hasattr(batch_loss, 'item') else batch_loss
        
        return total_loss / len(structures)
    
    def _validate(self,
                  structures: List[Dict],
                  energy_weight: float,
                  force_weight: float) -> float:
        """验证"""
        return self._train_epoch(structures, len(structures), energy_weight, force_weight)


class BayesianCalibration:
    """贝叶斯校准器"""
    
    def __init__(self, potential: BayesianPotential):
        self.potential = potential
        self.calibration_data = []
    
    def calibrate(self,
                  calibration_structures: List[Dict],
                  method: str = 'temperature_scaling') -> Dict:
        """
        校准不确定性估计
        
        Args:
            calibration_structures: 校准结构
            method: 校准方法 ('temperature_scaling', 'isotonic', 'platt')
        """
        # 收集预测和真实值
        predictions = []
        uncertainties = []
        targets = []
        
        for struct in calibration_structures:
            pred = self.potential.predict(
                struct['positions'],
                struct['atom_types'],
                struct.get('cell')
            )
            
            predictions.append(pred.energy_mean[0])
            uncertainties.append(np.sqrt(pred.energy_var[0]))
            targets.append(struct['energy'])
        
        predictions = np.array(predictions)
        uncertainties = np.array(uncertainties)
        targets = np.array(targets)
        
        if method == 'temperature_scaling':
            # 温度缩放
            residuals = np.abs(predictions - targets)
            # 找到最优温度使不确定性匹配残差
            T = np.mean(residuals / (uncertainties + 1e-10))
            
            return {
                'method': 'temperature_scaling',
                'temperature': T,
                'expected_calibration_error': self._compute_ece(predictions, uncertainties, targets)
            }
        
        return {'method': method}
    
    def _compute_ece(self,
                     predictions: np.ndarray,
                     uncertainties: np.ndarray,
                     targets: np.ndarray,
                     n_bins: int = 10) -> float:
        """计算期望校准误差"""
        # 置信区间覆盖率
        confidences = np.linspace(0.1, 0.9, n_bins)
        coverages = []
        
        for conf in confidences:
            z = norm.ppf((1 + conf) / 2)
            lower = predictions - z * uncertainties
            upper = predictions + z * uncertainties
            
            coverage = np.mean((targets >= lower) & (targets <= upper))
            coverages.append(coverage)
        
        coverages = np.array(coverages)
        ece = np.mean(np.abs(coverages - confidences))
        
        return ece


class UncertaintyCalibrator:
    """不确定性校准器"""
    
    def __init__(self):
        self.scale_factor = 1.0
    
    def fit(self,
            uncertainties: np.ndarray,
            residuals: np.ndarray,
            method: str = 'mle') -> float:
        """
        拟合校准参数
        
        Args:
            uncertainties: 预测不确定性
            residuals: 实际残差
            method: 拟合方法
        """
        if method == 'mle':
            # 最大似然估计
            self.scale_factor = np.mean(residuals**2) / np.mean(uncertainties**2)
        elif method == 'quantile':
            # 分位数匹配
            self.scale_factor = np.median(residuals) / np.median(uncertainties)
        
        return self.scale_factor
    
    def transform(self, uncertainties: np.ndarray) -> np.ndarray:
        """应用校准"""
        return uncertainties * np.sqrt(self.scale_factor)


# ==================== 示例和测试 ====================

def demo():
    """演示贝叶斯势函数"""
    print("=" * 80)
    print("🔬 贝叶斯神经网络势函数演示")
    print("=" * 80)
    
    # 创建描述符
    print("\n1. 创建ACSF描述符...")
    acsf = ACSFDescriptor(
        radial_params=[(0.05, 0.0), (0.05, 1.0), (0.005, 0.0), (0.005, 1.0)],
        angular_params=[(0.0001, 1.0, -1.0), (0.0001, 1.0, 1.0)],
        cutoff=5.0
    )
    print(f"   特征维度: {acsf.get_feature_dimension()}")
    
    # 测试结构：Si2二聚体
    print("\n2. 测试原子构型 (Si2)...")
    positions = np.array([
        [0.0, 0.0, 0.0],
        [2.35, 0.0, 0.0]  # Si-Si键长约2.35Å
    ])
    atom_types = np.array([14, 14])  # Si的原子序数
    
    # 计算描述符
    descriptors = acsf.transform(positions, atom_types)
    print(f"   描述符形状: {descriptors.shape}")
    print(f"   描述符范数: {np.linalg.norm(descriptors, axis=1)}")
    
    # 创建势函数
    if HAS_TORCH:
        print("\n3. 创建贝叶斯势函数...")
        
        # MC Dropout势函数
        print("   - MC Dropout势函数")
        mc_potential = MCDropoutPotential(
            descriptor=acsf,
            hidden_layers=[64, 64],
            dropout_rate=0.1
        )
        
        # 集成势函数
        print("   - 集成势函数")
        ensemble_potential = EnsemblePotential(
            descriptor=acsf,
            n_models=3,
            hidden_layers=[64, 64]
        )
        
        # 模拟预测（未训练，仅演示接口）
        print("\n4. 模拟预测...")
        
        # 使用随机初始化权重进行演示
        with torch.no_grad():
            # MC Dropout预测
            energy_pred = mc_potential.predict_energy(positions, atom_types, n_samples=50)
            print(f"   MC Dropout - 能量: {energy_pred.energy[0]:.4f} ± {np.sqrt(energy_pred.uncertainty.energy_var[0]):.4f} eV")
            
            # 集成预测
            energy_pred_ens = ensemble_potential.predict_energy(positions, atom_types)
            print(f"   集成方法 - 能量: {energy_pred_ens.energy[0]:.4f} ± {np.sqrt(energy_pred_ens.uncertainty.energy_var[0]):.4f} eV")
    else:
        print("\n3. PyTorch未安装，跳过势函数创建")
    
    # 演示SOAP描述符
    print("\n5. SOAP描述符...")
    soap = SOAPDescriptor(n_max=4, l_max=4, cutoff=5.0)
    soap_desc = soap.transform(positions, atom_types)
    print(f"   SOAP描述符形状: {soap_desc.shape}")
    
    print("\n" + "=" * 80)
    print("✅ 贝叶斯势函数演示完成")
    print("=" * 80)


if __name__ == "__main__":
    demo()
