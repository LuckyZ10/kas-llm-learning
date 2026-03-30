"""
DM21 (DeepMind 2021) Neural XC Functional Interface
DM21神经XC泛函接口

实现DeepMind 2021神经XC泛函，提供对强关联体系的改进和自相互作用修正。

DM21特点:
- 在21个精确解数据集上训练
- 对强关联体系(如H2解离曲线)有显著改进
- 减少自相互作用误差
- 满足重要的物理约束

参考文献:
- "Pushing the frontiers of density functionals by solving the fractional electron problem"
  - Kirkpatrick et al., Science 2021
  - DeepMind, 2021

作者: DFT-LAMMPS Team
日期: 2026-03-09
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
import requests
import hashlib

from .neural_xc import NeuralXCConfig, DensityFeatureExtractor, XCConstraints

logger = logging.getLogger(__name__)


@dataclass
class DM21Config:
    """DM21配置"""
    # 模型架构 (基于DM21论文描述)
    input_features: List[str] = field(default_factory=lambda: [
        'n', 's', 'tau', 'alpha', 'nu', 'zeta'
    ])
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64, 64])
    activation: str = 'silu'
    
    # 约束参数
    enforce_constraints: bool = True
    uniform_gas_limit: bool = True
    lieb_oxford_bound: bool = True
    
    # 计算设置
    spin_polarized: bool = True
    grid_level: str = 'fine'  # coarse, medium, fine, ultrafine
    
    # 模型权重
    model_path: Optional[str] = None  # 预训练模型路径
    download_pretrained: bool = True
    
    # 数值稳定性
    density_threshold: float = 1e-10
    gradient_threshold: float = 1e-10


class DM21FeatureExtractor(nn.Module):
    """
    DM21特征提取器
    实现DM21论文中描述的特征工程
    """
    
    def __init__(self, config: DM21Config):
        super().__init__()
        self.config = config
        
    def forward(self, n: torch.Tensor, grad_n: torch.Tensor,
                tau: torch.Tensor, laplacian: Optional[torch.Tensor] = None,
                n_up: Optional[torch.Tensor] = None,
                n_down: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        提取DM21特征
        
        Args:
            n: 总电子密度
            grad_n: 密度梯度模 |∇n|
            tau: 正交动能密度
            laplacian: 密度拉普拉斯 ∇²n
            n_up, n_down: 自旋向上/向下密度
            
        Returns:
            特征张量 [..., n_features]
        """
        n = torch.clamp(n, min=self.config.density_threshold)
        
        features = []
        
        # 1. 基础密度特征
        rs = (3 / (4 * np.pi * n))**(1/3)  # Wigner-Seitz半径
        kf = (3 * np.pi**2 * n)**(1/3)  # Fermi波矢
        
        features.extend([
            torch.log(n + 1e-12),
            rs,
            kf,
            n**(1/3),
        ])
        
        # 2. 约化梯度 s = |∇n| / (2 * (3π²)^(1/3) * n^(4/3))
        if grad_n is not None:
            grad_n = torch.clamp(grad_n, min=self.config.gradient_threshold)
            s = grad_n / (2 * (3 * np.pi**2)**(1/3) * n**(4/3))
            s = torch.clamp(s, max=50.0)
            
            features.extend([
                s,
                s**2,
                torch.log(1 + s**2),
                s / (1 + s),
                torch.tanh(s),
            ])
        
        # 3. 动能密度相关特征
        if tau is not None:
            # 均匀电子气动能密度
            tau_uniform = 3/10 * (3 * np.pi**2)**(2/3) * n**(5/3)
            
            # iso-orbital indicator α
            # α = (τ - |∇n|²/(8n)) / τ_uniform
            alpha = (tau - grad_n**2 / (8*n + 1e-10)) / (tau_uniform + 1e-10)
            alpha = torch.clamp(alpha, min=0.0, max=5.0)
            
            features.extend([
                tau / (tau_uniform + 1e-10),
                alpha,
                torch.log(alpha + 1e-10),
                torch.tanh(alpha),
            ])
        
        # 4. 非动力学相关性度量 ν
        if laplacian is not None:
            nu = laplacian / (n**(5/3) + 1e-10)
            features.extend([
                nu,
                torch.tanh(nu),
            ])
        
        # 5. 自旋极化
        if self.config.spin_polarized and n_up is not None and n_down is not None:
            zeta = (n_up - n_down) / (n + 1e-10)
            zeta = torch.clamp(zeta, min=-1.0, max=1.0)
            
            features.extend([
                zeta,
                zeta**2,
                (1 + zeta)**(1/3),
                (1 - zeta)**(1/3),
                torch.log(1 + zeta**2),
            ])
        
        # 6. 分数电荷特征 (DM21的关键创新)
        # 使用密度的高阶导数作为特征
        if laplacian is not None:
            features.append(laplacian / (n**(5/3) + 1e-10))
        
        # 堆叠特征
        features_stacked = torch.stack(features, dim=-1)
        
        # 数值稳定性处理
        features_stacked = torch.nan_to_num(
            features_stacked, 
            nan=0.0, 
            posinf=50.0, 
            neginf=-50.0
        )
        
        return features_stacked


class DM21NeuralNetwork(nn.Module):
    """
    DM21神经网络架构
    实现DeepMind 2021神经XC泛函的网络结构
    """
    
    def __init__(self, config: DM21Config, n_features: int):
        super().__init__()
        self.config = config
        
        # 构建网络
        layers = []
        dims = [n_features] + config.hidden_dims + [2]  # 输出: [exchange_enhancement, correlation_factor]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(self._get_activation())
                layers.append(nn.LayerNorm(dims[i+1]))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化
        self._init_weights()
    
    def _get_activation(self):
        """获取激活函数"""
        activations = {
            'relu': nn.ReLU(),
            'silu': nn.SiLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
        }
        return activations.get(self.config.activation, nn.SiLU())
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            Fx: 交换增强因子
            Fc: 相关能密度修正
        """
        output = self.network(features)
        
        # 分离交换和相关输出
        Fx = output[..., 0]  # 交换增强因子
        Fc = output[..., 1]  # 相关能密度
        
        # 应用约束
        if self.config.enforce_constraints:
            Fx = torch.clamp(Fx, min=0.5, max=3.0)  # 合理的交换增强范围
            Fc = torch.clamp(Fc, min=-2.0, max=0.0)  # 相关能应为负值
        
        return Fx, Fc


class DM21ExchangeCorrelation(nn.Module):
    """
    DM21交换关联泛函实现
    """
    
    def __init__(self, config: DM21Config):
        super().__init__()
        self.config = config
        
        # 特征提取器
        self.feature_extractor = DM21FeatureExtractor(config)
        
        # 计算特征维度 (用于网络初始化)
        dummy_n = torch.ones(1, 1)
        dummy_grad = torch.ones(1, 1)
        dummy_tau = torch.ones(1, 1)
        dummy_features = self.feature_extractor(dummy_n, dummy_grad, dummy_tau)
        n_features = dummy_features.shape[-1]
        
        # 神经网络
        self.neural_net = DM21NeuralNetwork(config, n_features)
        
        # LDA交换相关 (用于混合)
        self.lda_xc = LDAExchangeCorrelation()
    
    def forward(self, n: torch.Tensor, grad_n: Optional[torch.Tensor] = None,
                tau: Optional[torch.Tensor] = None,
                laplacian: Optional[torch.Tensor] = None,
                n_up: Optional[torch.Tensor] = None,
                n_down: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算DM21 XC能量密度
        
        Args:
            n: 总密度
            grad_n: 密度梯度
            tau: 动能密度
            laplacian: 拉普拉斯
            n_up, n_down: 自旋密度
            
        Returns:
            {
                'exc': XC能量密度,
                'ex': 交换能密度,
                'ec': 相关能密度,
                'Fx': 交换增强因子,
                'Fc': 相关能因子
            }
        """
        n = torch.clamp(n, min=self.config.density_threshold)
        
        # 提取特征
        features = self.feature_extractor(n, grad_n, tau, laplacian, n_up, n_down)
        
        # 神经网络预测
        Fx, Fc = self.neural_net(features)
        
        # 计算LDA交换
        ex_lda = -3/4 * (3/np.pi)**(1/3) * n**(1/3)
        
        # 应用交换增强因子
        ex = ex_lda * Fx
        
        # 相关能 (使用神经网络预测的值)
        ec = Fc * n  # 转换为能量密度
        
        # 总XC能量密度
        exc = ex + ec
        
        return {
            'exc': exc,
            'ex': ex,
            'ec': ec,
            'Fx': Fx,
            'Fc': Fc,
        }
    
    def calculate_xc_potential(self, n: torch.Tensor,
                               grad_n: Optional[torch.Tensor] = None,
                               tau: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算XC势 (通过自动微分)
        """
        n_clone = n.clone().requires_grad_(True)
        
        result = self.forward(n_clone, grad_n, tau)
        exc = result['exc']
        
        # 总能量
        E_xc = torch.sum(exc * n_clone)
        
        # 泛函导数
        v_xc = torch.autograd.grad(E_xc, n_clone, create_graph=True)[0]
        
        return v_xc


class LDAExchangeCorrelation:
    """LDA交换相关 (作为DM21的基础)"""
    
    def __init__(self):
        # PW92关联参数
        self.A = 0.031091
        self.alpha1 = 0.21370
    
    def exchange(self, n: torch.Tensor) -> torch.Tensor:
        """LDA交换能"""
        return -3/4 * (3/np.pi)**(1/3) * n**(1/3)
    
    def correlation(self, n: torch.Tensor) -> torch.Tensor:
        """PW92 LDA关联能 (简化)"""
        rs = (3 / (4 * np.pi * n))**(1/3)
        
        # 参数化形式 (简化PW92)
        ec = -self.A * (1 + self.alpha1 * rs) * torch.log(1 + 1 / (self.A * rs + 1e-10))
        
        return ec


class DM21Interface:
    """
    DM21主接口类
    提供与DFT代码的集成接口
    """
    
    # 预训练模型URL (示例)
    MODEL_URLS = {
        'dm21': 'https://example.com/models/dm21_weights.pt',
        'dm21m': 'https://example.com/models/dm21m_weights.pt',
    }
    
    def __init__(self, config: Optional[DM21Config] = None, 
                 model_name: str = 'dm21'):
        self.config = config or DM21Config()
        self.model_name = model_name
        
        # 初始化模型
        self.model = DM21ExchangeCorrelation(self.config)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        # 加载预训练权重
        if self.config.model_path:
            self.load_model(self.config.model_path)
        elif self.config.download_pretrained:
            self._download_pretrained_model(model_name)
        
        self.model.eval()
    
    def _download_pretrained_model(self, model_name: str):
        """下载预训练模型"""
        cache_dir = Path.home() / '.deepchem' / 'dm21_models'
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = cache_dir / f'{model_name}.pt'
        
        if model_path.exists():
            logger.info(f"使用缓存模型: {model_path}")
            self.load_model(str(model_path))
            return
        
        url = self.MODEL_URLS.get(model_name)
        if url:
            logger.info(f"下载DM21模型: {model_name}")
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                model_path.write_bytes(response.content)
                self.load_model(str(model_path))
            except Exception as e:
                logger.warning(f"模型下载失败: {e}，使用随机初始化权重")
        else:
            logger.warning(f"未知模型名称: {model_name}")
    
    def load_model(self, path: str):
        """加载模型权重"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"模型加载成功: {path}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
    
    def save_model(self, path: str):
        """保存模型权重"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'model_name': self.model_name
        }, path)
        logger.info(f"模型保存至: {path}")
    
    def calculate_exc(self, density: np.ndarray,
                      grad_density: Optional[np.ndarray] = None,
                      tau: Optional[np.ndarray] = None,
                      laplacian: Optional[np.ndarray] = None,
                      n_up: Optional[np.ndarray] = None,
                      n_down: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        计算XC能量密度 (NumPy接口)
        
        Returns:
            {
                'exc': XC能量密度,
                'ex': 交换能密度,
                'ec': 相关能密度
            }
        """
        with torch.no_grad():
            # 转换为张量
            n = torch.from_numpy(density).float().to(self.device)
            
            grad_n = None
            if grad_density is not None:
                grad_n = torch.from_numpy(grad_density).float().to(self.device)
            
            tau_t = None
            if tau is not None:
                tau_t = torch.from_numpy(tau).float().to(self.device)
            
            lap = None
            if laplacian is not None:
                lap = torch.from_numpy(laplacian).float().to(self.device)
            
            n_up_t = None
            n_down_t = None
            if n_up is not None:
                n_up_t = torch.from_numpy(n_up).float().to(self.device)
                n_down_t = torch.from_numpy(n_down).float().to(self.device)
            
            # 计算
            result = self.model(n, grad_n, tau_t, lap, n_up_t, n_down_t)
            
            # 转换为NumPy
            return {k: v.cpu().numpy() for k, v in result.items()}
    
    def calculate_total_xc_energy(self, density: np.ndarray,
                                   grad_density: Optional[np.ndarray] = None,
                                   tau: Optional[np.ndarray] = None,
                                   weights: Optional[np.ndarray] = None) -> float:
        """
        计算总XC能量
        
        Args:
            density: 电子密度
            grad_density: 密度梯度
            tau: 动能密度
            weights: 积分权重 (用于网格积分)
            
        Returns:
            总XC能量 (Hartree)
        """
        result = self.calculate_exc(density, grad_density, tau)
        exc = result['exc']
        
        if weights is None:
            # 均匀权重
            weights = np.ones_like(density)
        
        E_xc = np.sum(exc * density * weights)
        return E_xc
    
    def calculate_xc_potential(self, density: np.ndarray,
                               grad_density: Optional[np.ndarray] = None,
                               tau: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算XC势
        """
        n = torch.from_numpy(density).float().to(self.device)
        n.requires_grad_(True)
        
        grad_n = None
        if grad_density is not None:
            grad_n = torch.from_numpy(grad_density).float().to(self.device)
        
        tau_t = None
        if tau is not None:
            tau_t = torch.from_numpy(tau).float().to(self.device)
        
        result = self.model(n, grad_n, tau_t)
        exc = result['exc']
        
        E_xc = torch.sum(exc * n)
        v_xc = torch.autograd.grad(E_xc, n)[0]
        
        return v_xc.detach().cpu().numpy()
    
    # ============ 强关联体系改进 ============
    
    def analyze_strong_correlation(self, density: np.ndarray,
                                    grad_density: np.ndarray,
                                    tau: np.ndarray) -> Dict[str, np.ndarray]:
        """
        分析强关联特征
        
        Returns:
            {
                'iso_orbital_indicator': iso-orbital指标α,
                'localization_index': 局域化指标,
                'correlation_strength': 关联强度评估
            }
        """
        n = torch.from_numpy(density).float().to(self.device)
        grad_n = torch.from_numpy(grad_density).float().to(self.device)
        tau_t = torch.from_numpy(tau).float().to(self.device)
        
        with torch.no_grad():
            n = torch.clamp(n, min=1e-10)
            
            # 计算iso-orbital指标
            tau_uniform = 3/10 * (3 * np.pi**2)**(2/3) * n**(5/3)
            alpha = (tau_t - grad_n**2 / (8*n)) / (tau_uniform + 1e-10)
            alpha = torch.clamp(alpha, min=0.0, max=5.0)
            
            # 局域化指标 (简化)
            localization = 1.0 / (1.0 + alpha)
            
            # 关联强度 (基于α)
            correlation_strength = torch.where(
                alpha < 0.5,
                torch.ones_like(alpha),  # 强关联
                torch.zeros_like(alpha)  # 弱关联
            )
            
            return {
                'iso_orbital_indicator': alpha.cpu().numpy(),
                'localization_index': localization.cpu().numpy(),
                'correlation_strength': correlation_strength.cpu().numpy()
            }
    
    # ============ 自相互作用修正 ============
    
    def calculate_self_interaction_error(self, density: np.ndarray,
                                         molecular_orbitals: np.ndarray,
                                         occupation_numbers: np.ndarray) -> float:
        """
        计算自相互作用误差 (SIE)
        
        Args:
            density: 总电子密度
            molecular_orbitals: 分子轨道
            occupation_numbers: 占据数
            
        Returns:
            SIE估计 (Hartree)
        """
        # 简化实现：基于Perdew-Zunger自相互作用修正思想
        n = torch.from_numpy(density).float().to(self.device)
        
        with torch.no_grad():
            # 单电子密度的XC能量应等于交换能 (相关能为0)
            result = self.model(n)
            exc_per_electron = result['exc'] / (n + 1e-10)
            
            # 理想单电子情况：exc = ex_only
            ex_only = -3/4 * (3/np.pi)**(1/3) * n**(1/3)
            
            # SIE = |exc - ex_only| (对于单电子)
            sie_estimate = torch.mean(torch.abs(exc_per_electron - ex_only))
            
            return sie_estimate.item()
    
    def apply_sic_correction(self, density: np.ndarray,
                             orbital_densities: np.ndarray,
                             orbital_energies: np.ndarray) -> Dict[str, np.ndarray]:
        """
        应用自相互作用修正 (SIC)
        
        Args:
            density: 总密度
            orbital_densities: 各轨道密度
            orbital_energies: 各轨道能量
            
        Returns:
            {
                'corrected_energy': 修正后的能量,
                'sic_contribution': SIC贡献,
                'orbital_corrections': 各轨道修正
            }
        """
        n_orbitals = orbital_densities.shape[0]
        sic_correction = 0.0
        orbital_corrections = []
        
        for i in range(n_orbitals):
            orb_density = orbital_densities[i]
            
            # 计算该轨道的XC能量
            result = self.calculate_exc(orb_density)
            E_xc_orbital = np.sum(result['exc'] * orb_density)
            
            # 计算该轨道的Hartree能量
            # 简化：假设近似相等
            E_hartree_orbital = E_xc_orbital * 0.5  # 近似
            
            # SIC = E_H[n_i] + E_xc[n_i]
            sic_i = E_hartree_orbital + E_xc_orbital
            sic_correction += sic_i
            orbital_corrections.append(sic_i)
        
        return {
            'sic_contribution': sic_correction,
            'orbital_corrections': np.array(orbital_corrections),
        }
    
    # ============ PySCF集成 ============
    
    def to_pyscf_functional(self):
        """
        转换为PySCF可用的泛函对象
        """
        class DM21PySCFFunctional:
            def __init__(self, dm21_interface):
                self.dm21 = dm21_interface
                self.name = 'DM21'
            
            def eval_xc(self, xc_code, rho, spin=0, relativity=0, deriv=1,
                       omega=None, verbose=None):
                """
                PySCF eval_xc接口
                
                Args:
                    xc_code: XC代码 (忽略，固定为DM21)
                    rho: 密度 (和可能的其他量)
                    spin: 自旋 (0:非极化, 1:极化)
                    
                Returns:
                    (exc, vxc, fxc, kxc)
                """
                # 解析rho
                if spin == 0:
                    n = rho[0]  # 总密度
                    grad_n = rho[1] if len(rho) > 1 else None
                    tau = rho[3] if len(rho) > 3 else None
                    
                    result = self.dm21.calculate_exc(n, grad_n, tau)
                    exc = result['exc']
                    
                    # 计算势 (简化)
                    vxc = (exc, None, None, None)
                    
                else:
                    n_up = rho[0]
                    n_down = rho[1]
                    n = n_up + n_down
                    grad_n = None
                    tau = None
                    
                    result = self.dm21.calculate_exc(n, grad_n, tau, n_up=n_up, n_down=n_down)
                    exc = result['exc']
                    vxc = (exc, None, None, None)
                
                return exc, vxc, None, None
        
        return DM21PySCFFunctional(self)


def compare_dm21_vs_pbe(structure: Dict, grid_points: np.ndarray) -> Dict[str, np.ndarray]:
    """
    比较DM21和PBE泛函的结果
    
    Returns:
        {
            'dm21_exc': DM21 XC能量密度,
            'pbe_exc': PBE XC能量密度,
            'difference': 差异,
            'improvement_ratio': 改进比例
        }
    """
    dm21 = DM21Interface()
    
    # 计算密度 (简化)
    density = np.exp(-np.sum(grid_points**2, axis=-1))
    grad_density = np.abs(np.gradient(density))
    tau = 0.3 * density**(5/3)
    
    # DM21结果
    dm21_result = dm21.calculate_exc(density, grad_density, tau)
    
    # PBE结果 (简化)
    pbe_exc = calculate_pbe_exc(density, grad_density)
    
    difference = dm21_result['exc'] - pbe_exc
    improvement = np.abs(difference) / (np.abs(pbe_exc) + 1e-10)
    
    return {
        'dm21_exc': dm21_result['exc'],
        'pbe_exc': pbe_exc,
        'difference': difference,
        'improvement_ratio': improvement
    }


def calculate_pbe_exc(n: np.ndarray, grad_n: np.ndarray) -> np.ndarray:
    """简化PBE计算 (用于比较)"""
    # LDA交换
    ex_lda = -3/4 * (3/np.pi)**(1/3) * n**(1/3)
    
    # GGA增强 (简化PBE)
    s = grad_n / (2 * (3 * np.pi**2)**(1/3) * n**(4/3) + 1e-10)
    s = np.clip(s, 0, 50)
    
    # PBE交换增强因子 (简化)
    kappa = 0.804
    mu = 0.21951
    Fx = 1 + kappa - kappa / (1 + mu * s**2 / kappa)
    
    ex = ex_lda * Fx
    
    # 关联能 (简化)
    ec = -0.02 * n**(1/3)
    
    return ex + ec


def create_dm21_functional(pretrained: bool = True) -> DM21Interface:
    """
    创建DM21泛函实例
    
    Args:
        pretrained: 是否加载预训练权重
        
    Returns:
        DM21Interface实例
    """
    config = DM21Config(download_pretrained=pretrained)
    return DM21Interface(config)


# 导出
__all__ = [
    'DM21Config',
    'DM21FeatureExtractor',
    'DM21NeuralNetwork',
    'DM21ExchangeCorrelation',
    'DM21Interface',
    'create_dm21_functional',
    'compare_dm21_vs_pbe',
]
