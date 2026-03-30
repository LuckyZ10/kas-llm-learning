"""
半导体→光伏材料迁移 (Semiconductor to Photovoltaic Transfer)
实现从半导体材料到光伏材料的跨领域知识迁移

作者: DFT-LAMMPS Team
版本: 1.0.0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class SemiPVConfig:
    """半导体-光伏迁移配置"""
    semi_feature_dim: int = 128
    pv_feature_dim: int = 128
    shared_dim: int = 64
    
    learning_rate: float = 1e-3
    num_epochs: int = 100
    batch_size: int = 32
    
    # 多任务学习权重
    bandgap_weight: float = 0.4
    absorption_weight: float = 0.3
    efficiency_weight: float = 0.3
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SemiconductorFeatureExtractor(nn.Module):
    """半导体特征提取器"""
    
    def __init__(self, input_dim: int = 128, output_dim: int = 64):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, 192),
            nn.ReLU(),
            nn.BatchNorm1d(192),
            nn.Dropout(0.2),
            
            nn.Linear(192, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_layers(x)


class PVFeatureExtractor(nn.Module):
    """光伏特征提取器"""
    
    def __init__(self, input_dim: int = 128, output_dim: int = 64):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, 192),
            nn.ReLU(),
            nn.BatchNorm1d(192),
            nn.Dropout(0.2),
            
            nn.Linear(192, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_layers(x)


class BandGapPredictor(nn.Module):
    """带隙预测器 - 共享"""
    
    def __init__(self, input_dim: int = 64):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(x)


class AbsorptionPredictor(nn.Module):
    """光吸收预测器 - 光伏专用"""
    
    def __init__(self, input_dim: int = 64):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(x)


class EfficiencyPredictor(nn.Module):
    """转换效率预测器 - 光伏专用"""
    
    def __init__(self, input_dim: int = 64):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 效率在0-1之间
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(x) * 30  # 放大到0-30%


class SemiconductorToPVTransfer:
    """半导体到光伏知识迁移器"""
    
    def __init__(self, config: SemiPVConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 特征提取器
        self.semi_extractor = SemiconductorFeatureExtractor(
            config.semi_feature_dim,
            config.shared_dim
        ).to(self.device)
        
        self.pv_extractor = PVFeatureExtractor(
            config.pv_feature_dim,
            config.shared_dim
        ).to(self.device)
        
        # 预测器
        self.bandgap_predictor = BandGapPredictor(
            config.shared_dim
        ).to(self.device)
        
        self.absorption_predictor = AbsorptionPredictor(
            config.shared_dim
        ).to(self.device)
        
        self.efficiency_predictor = EfficiencyPredictor(
            config.shared_dim
        ).to(self.device)
        
        # 优化器
        params = (
            list(self.semi_extractor.parameters()) +
            list(self.bandgap_predictor.parameters())
        )
        self.semi_optimizer = torch.optim.Adam(params, lr=config.learning_rate)
        
        self.pv_optimizer = torch.optim.Adam(
            list(self.pv_extractor.parameters()) +
            list(self.absorption_predictor.parameters()) +
            list(self.efficiency_predictor.parameters()),
            lr=config.learning_rate
        )
        
        self.criterion = nn.MSELoss()
    
    def train_semiconductor_phase(
        self,
        train_loader: torch.utils.data.DataLoader
    ) -> Dict[str, List[float]]:
        """
        半导体训练阶段
        
        学习带隙预测
        """
        print("Training semiconductor phase...")
        
        history = {"loss": [], "bandgap_mae": []}
        
        for epoch in range(self.config.num_epochs):
            self.semi_extractor.train()
            self.bandgap_predictor.train()
            
            epoch_loss = 0.0
            epoch_mae = 0.0
            num_batches = 0
            
            for features, bandgaps in train_loader:
                features = features.to(self.device)
                bandgaps = bandgaps.to(self.device)
                
                # 前向传播
                embeddings = self.semi_extractor(features)
                predictions = self.bandgap_predictor(embeddings)
                
                # 损失
                loss = self.criterion(predictions.squeeze(), bandgaps)
                
                # 反向传播
                self.semi_optimizer.zero_grad()
                loss.backward()
                self.semi_optimizer.step()
                
                epoch_loss += loss.item()
                epoch_mae += torch.abs(predictions.squeeze() - bandgaps).mean().item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            avg_mae = epoch_mae / num_batches
            
            history["loss"].append(avg_loss)
            history["bandgap_mae"].append(avg_mae)
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, MAE={avg_mae:.4f} eV")
        
        return history
    
    def transfer_to_pv(
        self,
        pv_train_loader: torch.utils.data.DataLoader,
        transfer_strategy: str = "feature_extraction"
    ) -> Dict[str, List[float]]:
        """
        迁移到光伏领域
        
        Args:
            transfer_strategy: feature_extraction, fine_tuning, or multi_task
        """
        print(f"Transferring to PV with strategy: {transfer_strategy}...")
        
        # 根据策略冻结参数
        if transfer_strategy == "feature_extraction":
            for param in self.semi_extractor.parameters():
                param.requires_grad = False
            for param in self.bandgap_predictor.parameters():
                param.requires_grad = False
        
        # 复制半导体编码器权重到光伏编码器
        self._copy_encoder_weights()
        
        history = {"total_loss": [], "efficiency_mae": []}
        
        for epoch in range(self.config.num_epochs):
            self.pv_extractor.train()
            self.absorption_predictor.train()
            self.efficiency_predictor.train()
            
            if transfer_strategy in ["fine_tuning", "multi_task"]:
                self.bandgap_predictor.train()
            
            epoch_loss = 0.0
            epoch_eff_mae = 0.0
            num_batches = 0
            
            for features, bandgaps, absorption, efficiency in pv_train_loader:
                features = features.to(self.device)
                bandgaps = bandgaps.to(self.device)
                absorption = absorption.to(self.device)
                efficiency = efficiency.to(self.device)
                
                # 特征提取
                pv_embeddings = self.pv_extractor(features)
                
                # 预测
                if transfer_strategy == "multi_task":
                    # 同时预测带隙
                    bg_pred = self.bandgap_predictor(pv_embeddings)
                    bg_loss = self.criterion(bg_pred.squeeze(), bandgaps)
                else:
                    bg_loss = 0
                
                abs_pred = self.absorption_predictor(pv_embeddings)
                eff_pred = self.efficiency_predictor(pv_embeddings)
                
                abs_loss = self.criterion(abs_pred.squeeze(), absorption)
                eff_loss = self.criterion(eff_pred.squeeze(), efficiency)
                
                # 总损失
                total_loss = (
                    self.config.bandgap_weight * bg_loss +
                    self.config.absorption_weight * abs_loss +
                    self.config.efficiency_weight * eff_loss
                )
                
                # 反向传播
                self.pv_optimizer.zero_grad()
                if transfer_strategy == "multi_task":
                    self.semi_optimizer.zero_grad()
                
                total_loss.backward()
                
                self.pv_optimizer.step()
                if transfer_strategy == "multi_task":
                    self.semi_optimizer.step()
                
                epoch_loss += total_loss.item()
                epoch_eff_mae += torch.abs(eff_pred.squeeze() - efficiency).mean().item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            avg_eff_mae = epoch_eff_mae / num_batches
            
            history["total_loss"].append(avg_loss)
            history["efficiency_mae"].append(avg_eff_mae)
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, Eff MAE={avg_eff_mae:.2f}%")
        
        return history
    
    def _copy_encoder_weights(self):
        """复制编码器权重"""
        semi_state = self.semi_extractor.state_dict()
        pv_state = self.pv_extractor.state_dict()
        
        # 复制匹配的层
        for key in semi_state:
            if key in pv_state and semi_state[key].shape == pv_state[key].shape:
                pv_state[key] = semi_state[key].clone()
        
        self.pv_extractor.load_state_dict(pv_state)
        print("Encoder weights copied from semiconductor to PV")
    
    def predict_pv_properties(
        self,
        features: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        预测光伏材料性质
        """
        self.pv_extractor.eval()
        self.bandgap_predictor.eval()
        self.absorption_predictor.eval()
        self.efficiency_predictor.eval()
        
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            embeddings = self.pv_extractor(features_tensor)
            
            bandgap = self.bandgap_predictor(embeddings).cpu().numpy()
            absorption = self.absorption_predictor(embeddings).cpu().numpy()
            efficiency = self.efficiency_predictor(embeddings).cpu().numpy()
        
        return {
            "bandgap": bandgap.squeeze(),
            "absorption": absorption.squeeze(),
            "efficiency": efficiency.squeeze()
        }
    
    def evaluate_transfer_quality(
        self,
        test_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        评估迁移质量
        """
        self.pv_extractor.eval()
        self.bandgap_predictor.eval()
        self.absorption_predictor.eval()
        self.efficiency_predictor.eval()
        
        metrics = {
            "bandgap_mae": 0.0,
            "absorption_mae": 0.0,
            "efficiency_mae": 0.0,
            "count": 0
        }
        
        with torch.no_grad():
            for features, bandgaps, absorption, efficiency in test_loader:
                features = features.to(self.device)
                
                embeddings = self.pv_extractor(features)
                
                bg_pred = self.bandgap_predictor(embeddings)
                abs_pred = self.absorption_predictor(embeddings)
                eff_pred = self.efficiency_predictor(embeddings)
                
                metrics["bandgap_mae"] += torch.abs(
                    bg_pred.squeeze().cpu() - bandgaps
                ).sum().item()
                metrics["absorption_mae"] += torch.abs(
                    abs_pred.squeeze().cpu() - absorption
                ).sum().item()
                metrics["efficiency_mae"] += torch.abs(
                    eff_pred.squeeze().cpu() - efficiency
                ).sum().item()
                metrics["count"] += len(features)
        
        for key in ["bandgap_mae", "absorption_mae", "efficiency_mae"]:
            metrics[key] /= metrics["count"]
        
        del metrics["count"]
        
        return metrics
    
    def analyze_bandgap_efficiency_correlation(
        self,
        test_features: np.ndarray,
        test_efficiency: np.ndarray
    ) -> Dict:
        """
        分析带隙-效率关系
        
        探索从半导体学到的带隙知识如何影响光伏效率预测
        """
        predictions = self.predict_pv_properties(test_features)
        
        bandgaps = predictions["bandgap"]
        efficiencies = predictions["efficiency"]
        
        # 计算相关性
        correlation = np.corrcoef(bandgaps, efficiencies)[0, 1]
        
        # 最优带隙范围 (Shockley-Queisser极限约1.34 eV)
        optimal_range = (1.0, 1.7)
        in_range_mask = (bandgaps >= optimal_range[0]) & (bandgaps <= optimal_range[1])
        
        analysis = {
            "bandgap_efficiency_correlation": correlation,
            "predicted_bandgap_range": (float(bandgaps.min()), float(bandgaps.max())),
            "predicted_efficiency_range": (float(efficiencies.min()), float(efficiencies.max())),
            "optimal_bandgap_fraction": float(in_range_mask.mean()),
            "mean_efficiency_in_range": float(efficiencies[in_range_mask].mean()) if in_range_mask.any() else 0.0
        }
        
        return analysis


def generate_semiconductor_data(
    n_samples: int = 1000,
    feature_dim: int = 128
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成半导体数据
    
    特征：晶体结构、电子结构参数
    目标：带隙 (eV)
    """
    np.random.seed(42)
    
    features = np.random.randn(n_samples, feature_dim)
    
    # 影响带隙的特征
    features[:, 0] = np.random.uniform(2.0, 15.0, n_samples)  # 晶格常数
    features[:, 1] = np.random.uniform(1, 7, n_samples)       # 配位数
    features[:, 2] = np.random.uniform(-3, 3, n_samples)      # 电负性差
    
    # 带隙 = f(结构, 组成)
    bandgap = (
        0.5 + 0.1 * features[:, 0] + 0.2 * features[:, 1] +
        0.3 * np.abs(features[:, 2]) + 0.5 * np.random.randn(n_samples)
    )
    bandgap = np.clip(bandgap, 0.1, 6.0)
    
    return features, bandgap


def generate_pv_data(
    n_samples: int = 500,
    feature_dim: int = 128
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成光伏材料数据
    
    特征：光吸收相关特征
    目标：带隙、吸收系数、转换效率
    """
    np.random.seed(43)
    
    features = np.random.randn(n_samples, feature_dim)
    
    # 光伏材料特征
    features[:, 0] = np.random.uniform(2.0, 15.0, n_samples)  # 晶格常数
    features[:, 1] = np.random.uniform(100, 1000, n_samples)  # 载流子迁移率
    features[:, 2] = np.random.uniform(1e14, 1e18, n_samples)  # 载流子浓度
    
    # 带隙 (与半导体类似)
    bandgap = (
        0.5 + 0.1 * features[:, 0] +
        0.3 * np.random.randn(n_samples)
    )
    bandgap = np.clip(bandgap, 0.3, 3.5)
    
    # 吸收系数 (与带隙相关，带隙越小吸收越强)
    absorption = (
        1e5 + 5e4 * (3.5 - bandgap) + 1e4 * np.random.randn(n_samples)
    )
    absorption = np.clip(absorption, 1e4, 2e5)
    
    # 转换效率 (Shockley-Queisser极限相关)
    optimal_bg = 1.34
    bg_penalty = -2.0 * (bandgap - optimal_bg) ** 2
    mobility_bonus = 0.01 * np.log10(features[:, 1])
    
    efficiency = (
        20 + bg_penalty + mobility_bonus + 2 * np.random.randn(n_samples)
    )
    efficiency = np.clip(efficiency, 5, 30)
    
    return features, bandgap, absorption, efficiency


# 演示代码
if __name__ == "__main__":
    print("=" * 70)
    print("半导体→光伏材料迁移演示 (Semiconductor to PV Transfer Demo)")
    print("=" * 70)
    
    # 生成数据
    print("\nGenerating semiconductor and PV data...")
    
    semi_features, semi_bandgaps = generate_semiconductor_data(n_samples=1000)
    pv_features, pv_bandgaps, pv_absorption, pv_efficiency = generate_pv_data(n_samples=500)
    
    print(f"Semiconductor data: {semi_features.shape}")
    print(f"PV data: {pv_features.shape}")
    
    # 创建数据加载器
    semi_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(semi_features),
        torch.FloatTensor(semi_bandgaps)
    )
    semi_loader = torch.utils.data.DataLoader(
        semi_dataset,
        batch_size=32,
        shuffle=True
    )
    
    pv_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(pv_features),
        torch.FloatTensor(pv_bandgaps),
        torch.FloatTensor(pv_absorption),
        torch.FloatTensor(pv_efficiency)
    )
    pv_loader = torch.utils.data.DataLoader(
        pv_dataset,
        batch_size=32,
        shuffle=True
    )
    
    # 创建迁移模型
    config = SemiPVConfig()
    transfer_model = SemiconductorToPVTransfer(config)
    
    # 第一阶段：半导体训练
    print("\n" + "=" * 70)
    print("Phase 1: Training on Semiconductor Data")
    print("=" * 70)
    
    history1 = transfer_model.train_semiconductor_phase(semi_loader)
    
    # 第二阶段：迁移到光伏
    print("\n" + "=" * 70)
    print("Phase 2: Transfer to PV (Multi-Task Learning)")
    print("=" * 70)
    
    history2 = transfer_model.transfer_to_pv(
        pv_loader,
        transfer_strategy="multi_task"
    )
    
    # 测试预测
    print("\n" + "=" * 70)
    print("Testing PV Property Prediction")
    print("=" * 70)
    
    test_features = pv_features[:100]
    predictions = transfer_model.predict_pv_properties(test_features)
    
    print(f"\nPredicted Bandgap: mean={predictions['bandgap'].mean():.3f} eV")
    print(f"Actual Bandgap: mean={pv_bandgaps[:100].mean():.3f} eV")
    print(f"\nPredicted Absorption: mean={predictions['absorption'].mean():.2e}")
    print(f"Actual Absorption: mean={pv_absorption[:100].mean():.2e}")
    print(f"\nPredicted Efficiency: mean={predictions['efficiency'].mean():.2f}%")
    print(f"Actual Efficiency: mean={pv_efficiency[:100].mean():.2f}%")
    
    # 分析带隙-效率关系
    print("\n" + "=" * 70)
    print("Analyzing Bandgap-Efficiency Relationship")
    print("=" * 70)
    
    analysis = transfer_model.analyze_bandgap_efficiency_correlation(
        pv_features[:200],
        pv_efficiency[:200]
    )
    
    for key, value in analysis.items():
        if isinstance(value, tuple):
            print(f"  {key}: ({value[0]:.3f}, {value[1]:.3f})")
        else:
            print(f"  {key}: {value:.4f}")
    
    # 评估迁移质量
    print("\n" + "=" * 70)
    print("Transfer Quality Evaluation")
    print("=" * 70)
    
    metrics = transfer_model.evaluate_transfer_quality(pv_loader)
    
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n" + "=" * 70)
    print("Semiconductor to PV Transfer Demo Complete!")
    print("=" * 70)
