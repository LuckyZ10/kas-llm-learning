"""
电池→催化剂知识迁移 (Battery to Catalyst Transfer)
实现从电池材料到催化剂材料的跨领域知识迁移

作者: DFT-LAMMPS Team
版本: 1.0.0
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
from collections import defaultdict


@dataclass
class BatteryCatalystConfig:
    """电池-催化剂迁移配置"""
    # 特征维度
    battery_feature_dim: int = 128
    catalyst_feature_dim: int = 128
    shared_dim: int = 64
    
    # 训练配置
    learning_rate: float = 1e-3
    num_epochs: int = 100
    batch_size: int = 32
    
    # 迁移配置
    freeze_encoder_epochs: int = 20
    adaptation_lambda: float = 0.5
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class BatteryFeatureEncoder(nn.Module):
    """电池材料特征编码器"""
    
    def __init__(self, input_dim: int = 128, output_dim: int = 64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CatalystFeatureEncoder(nn.Module):
    """催化剂材料特征编码器"""
    
    def __init__(self, input_dim: int = 128, output_dim: int = 64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SharedPropertyPredictor(nn.Module):
    """共享性质预测器"""
    
    def __init__(self, input_dim: int = 64, num_properties: int = 1):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_properties)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(x)


class DomainDiscriminator(nn.Module):
    """域判别器 - 用于域适应"""
    
    def __init__(self, input_dim: int = 64):
        super().__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)


class BatteryToCatalystTransfer:
    """电池到催化剂知识迁移器"""
    
    def __init__(self, config: BatteryCatalystConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 编码器
        self.battery_encoder = BatteryFeatureEncoder(
            config.battery_feature_dim,
            config.shared_dim
        ).to(self.device)
        
        self.catalyst_encoder = CatalystFeatureEncoder(
            config.catalyst_feature_dim,
            config.shared_dim
        ).to(self.device)
        
        # 性质预测器
        self.energy_predictor = SharedPropertyPredictor(
            config.shared_dim, 1
        ).to(self.device)  # 预测能量/电位
        
        self.stability_predictor = SharedPropertyPredictor(
            config.shared_dim, 1
        ).to(self.device)  # 预测稳定性
        
        # 域判别器
        self.domain_discriminator = DomainDiscriminator(
            config.shared_dim
        ).to(self.device)
        
        # 优化器
        self.encoder_optimizer = torch.optim.Adam(
            list(self.battery_encoder.parameters()) +
            list(self.catalyst_encoder.parameters()),
            lr=config.learning_rate
        )
        
        self.predictor_optimizer = torch.optim.Adam(
            list(self.energy_predictor.parameters()) +
            list(self.stability_predictor.parameters()),
            lr=config.learning_rate
        )
        
        self.discriminator_optimizer = torch.optim.Adam(
            self.domain_discriminator.parameters(),
            lr=config.learning_rate
        )
        
        self.criterion_mse = nn.MSELoss()
        self.criterion_bce = nn.BCELoss()
    
    def pretrain_on_battery(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None
    ) -> Dict[str, List[float]]:
        """
        在电池数据上预训练
        
        学习电池材料的特征表示和性质预测
        """
        print("Pre-training on battery data...")
        
        history = {"train_loss": [], "val_loss": []}
        
        for epoch in range(self.config.num_epochs // 2):
            self.battery_encoder.train()
            self.energy_predictor.train()
            self.stability_predictor.train()
            
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_features, batch_energy, batch_stability in train_loader:
                batch_features = batch_features.to(self.device)
                batch_energy = batch_energy.to(self.device)
                batch_stability = batch_stability.to(self.device)
                
                # 编码
                encoded = self.battery_encoder(batch_features)
                
                # 预测
                energy_pred = self.energy_predictor(encoded)
                stability_pred = self.stability_predictor(encoded)
                
                # 损失
                loss = (
                    self.criterion_mse(energy_pred.squeeze(), batch_energy) +
                    self.criterion_mse(stability_pred.squeeze(), batch_stability)
                ) / 2
                
                # 反向传播
                self.encoder_optimizer.zero_grad()
                self.predictor_optimizer.zero_grad()
                loss.backward()
                self.encoder_optimizer.step()
                self.predictor_optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            history["train_loss"].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        return history
    
    def adapt_to_catalyst(
        self,
        catalyst_loader: torch.utils.data.DataLoader,
        use_adversarial: bool = True
    ) -> Dict[str, List[float]]:
        """
        适应到催化剂领域
        
        使用对抗性域适应或微调
        """
        print("Adapting to catalyst domain...")
        
        history = {"total_loss": [], "pred_loss": [], "domain_loss": []}
        
        for epoch in range(self.config.num_epochs):
            self.catalyst_encoder.train()
            self.energy_predictor.train()
            self.stability_predictor.train()
            
            if use_adversarial:
                self.domain_discriminator.train()
            
            epoch_total = 0.0
            epoch_pred = 0.0
            epoch_domain = 0.0
            num_batches = 0
            
            for batch_features, batch_energy, batch_stability in catalyst_loader:
                batch_features = batch_features.to(self.device)
                batch_energy = batch_energy.to(self.device)
                batch_stability = batch_stability.to(self.device)
                
                batch_size = batch_features.size(0)
                
                # 编码
                catalyst_encoded = self.catalyst_encoder(batch_features)
                
                # 预测损失
                energy_pred = self.energy_predictor(catalyst_encoded)
                stability_pred = self.stability_predictor(catalyst_encoded)
                
                pred_loss = (
                    self.criterion_mse(energy_pred.squeeze(), batch_energy) +
                    self.criterion_mse(stability_pred.squeeze(), batch_stability)
                ) / 2
                
                # 域适应
                domain_loss = 0.0
                if use_adversarial and epoch >= self.config.freeze_encoder_epochs:
                    # 生成域标签
                    domain_labels = torch.ones(batch_size, 1).to(self.device)
                    
                    # 判别
                    domain_pred = self.domain_discriminator(catalyst_encoded)
                    domain_loss = self.criterion_bce(domain_pred, domain_labels)
                    
                    # 生成器损失 (骗过判别器)
                    fool_loss = self.criterion_bce(
                        domain_pred,
                        torch.zeros(batch_size, 1).to(self.device)
                    )
                    
                    total_loss = pred_loss + self.config.adaptation_lambda * fool_loss
                else:
                    total_loss = pred_loss
                
                # 反向传播
                self.encoder_optimizer.zero_grad()
                self.predictor_optimizer.zero_grad()
                total_loss.backward()
                self.encoder_optimizer.step()
                self.predictor_optimizer.step()
                
                # 更新判别器
                if use_adversarial and epoch >= self.config.freeze_encoder_epochs:
                    with torch.no_grad():
                        battery_samples = torch.randn(
                            batch_size,
                            self.config.battery_feature_dim
                        ).to(self.device)
                        battery_encoded = self.battery_encoder(battery_samples)
                        catalyst_encoded = self.catalyst_encoder(batch_features)
                    
                    # 判别
                    battery_domain = torch.zeros(batch_size, 1).to(self.device)
                    catalyst_domain = torch.ones(batch_size, 1).to(self.device)
                    
                    battery_pred = self.domain_discriminator(battery_encoded)
                    catalyst_pred = self.domain_discriminator(catalyst_encoded)
                    
                    disc_loss = (
                        self.criterion_bce(battery_pred, battery_domain) +
                        self.criterion_bce(catalyst_pred, catalyst_domain)
                    ) / 2
                    
                    self.discriminator_optimizer.zero_grad()
                    disc_loss.backward()
                    self.discriminator_optimizer.step()
                    
                    epoch_domain += disc_loss.item()
                
                epoch_total += total_loss.item()
                epoch_pred += pred_loss.item()
                num_batches += 1
            
            history["total_loss"].append(epoch_total / num_batches)
            history["pred_loss"].append(epoch_pred / num_batches)
            history["domain_loss"].append(epoch_domain / num_batches)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Total Loss = {epoch_total/num_batches:.4f}, "
                      f"Pred Loss = {epoch_pred/num_batches:.4f}")
        
        return history
    
    def predict_catalyst_properties(
        self,
        catalyst_features: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        预测催化剂性质
        
        利用从电池材料学到的知识
        """
        self.catalyst_encoder.eval()
        self.energy_predictor.eval()
        self.stability_predictor.eval()
        
        features = torch.FloatTensor(catalyst_features).to(self.device)
        
        with torch.no_grad():
            encoded = self.catalyst_encoder(features)
            
            energy = self.energy_predictor(encoded).cpu().numpy()
            stability = self.stability_predictor(encoded).cpu().numpy()
        
        return {
            "energy": energy.squeeze(),
            "stability": stability.squeeze(),
            "embedding": encoded.cpu().numpy()
        }
    
    def analyze_learned_relationships(self) -> Dict:
        """
        分析学到的电池-催化剂关系
        
        探索两个领域之间的知识联系
        """
        analysis = {
            "shared_knowledge": {},
            "domain_specific": {},
            "transfer_effectiveness": {}
        }
        
        # 分析编码器权重
        battery_weights = self.battery_encoder.encoder[-2].weight.data.cpu().numpy()
        catalyst_weights = self.catalyst_encoder.encoder[-2].weight.data.cpu().numpy()
        
        # 计算权重相似度
        weight_corr = np.corrcoef(
            battery_weights.flatten(),
            catalyst_weights.flatten()
        )[0, 1]
        
        analysis["shared_knowledge"]["weight_correlation"] = weight_corr
        
        # 分析预测器
        energy_weights = self.energy_predictor.predictor[0].weight.data.cpu().numpy()
        stability_weights = self.stability_predictor.predictor[0].weight.data.cpu().numpy()
        
        predictor_corr = np.corrcoef(
            energy_weights.flatten(),
            stability_weights.flatten()
        )[0, 1]
        
        analysis["shared_knowledge"]["predictor_correlation"] = predictor_corr
        
        return analysis
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            "battery_encoder": self.battery_encoder.state_dict(),
            "catalyst_encoder": self.catalyst_encoder.state_dict(),
            "energy_predictor": self.energy_predictor.state_dict(),
            "stability_predictor": self.stability_predictor.state_dict(),
            "domain_discriminator": self.domain_discriminator.state_dict()
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.battery_encoder.load_state_dict(checkpoint["battery_encoder"])
        self.catalyst_encoder.load_state_dict(checkpoint["catalyst_encoder"])
        self.energy_predictor.load_state_dict(checkpoint["energy_predictor"])
        self.stability_predictor.load_state_dict(checkpoint["stability_predictor"])
        self.domain_discriminator.load_state_dict(checkpoint["domain_discriminator"])
        
        print(f"Model loaded from {path}")


def generate_battery_data(
    n_samples: int = 1000,
    feature_dim: int = 128
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成模拟电池材料数据
    
    特征：成分、结构、电化学性质
    目标：工作电压、循环稳定性
    """
    np.random.seed(42)
    
    # 特征
    features = np.random.randn(n_samples, feature_dim)
    
    # 添加电池材料的特征模式
    # 离子半径影响
    features[:, 0] = np.random.uniform(0.5, 2.0, n_samples)
    # 电负性影响
    features[:, 1] = np.random.uniform(0.5, 4.0, n_samples)
    # 晶格参数影响
    features[:, 2] = np.random.uniform(3.0, 15.0, n_samples)
    
    # 目标：工作电压 (V)
    voltage = (
        2.5 + 0.5 * features[:, 1] +  # 电负性贡献
        0.2 * features[:, 0] +         # 离子半径贡献
        0.1 * np.random.randn(n_samples)
    )
    voltage = np.clip(voltage, 1.0, 5.0)
    
    # 目标：循环稳定性 (%)
    stability = (
        80 + 5 * features[:, 0] - 2 * np.abs(features[:, 1] - 2.5) +
        5 * np.random.randn(n_samples)
    )
    stability = np.clip(stability, 50, 100)
    
    return features, voltage, stability


def generate_catalyst_data(
    n_samples: int = 500,
    feature_dim: int = 128
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成模拟催化剂材料数据
    
    特征：成分、表面结构、电子结构
    目标：反应能垒、催化稳定性
    """
    np.random.seed(43)
    
    # 特征
    features = np.random.randn(n_samples, feature_dim)
    
    # 催化剂特征模式
    # d带中心
    features[:, 0] = np.random.uniform(-3.0, -1.0, n_samples)
    # 配位数
    features[:, 1] = np.random.uniform(3, 12, n_samples)
    # 表面积
    features[:, 2] = np.random.uniform(10, 500, n_samples)
    
    # 目标：反应能垒 (eV) - 与电池能量相关
    barrier = (
        1.5 - 0.3 * features[:, 0] +  # d带中心贡献
        0.1 * features[:, 1] +         # 配位数贡献
        0.2 * np.random.randn(n_samples)
    )
    barrier = np.clip(barrier, 0.3, 3.0)
    
    # 目标：催化稳定性 - 与电池稳定性类似
    stability = (
        75 + 3 * features[:, 1] - 5 * np.abs(features[:, 0] + 2.0) +
        5 * np.random.randn(n_samples)
    )
    stability = np.clip(stability, 40, 100)
    
    return features, barrier, stability


# 演示代码
if __name__ == "__main__":
    print("=" * 70)
    print("电池→催化剂知识迁移演示 (Battery to Catalyst Transfer Demo)")
    print("=" * 70)
    
    # 生成数据
    print("\nGenerating battery and catalyst data...")
    
    battery_features, battery_voltage, battery_stability = generate_battery_data(
        n_samples=1000
    )
    
    catalyst_features, catalyst_barrier, catalyst_stability = generate_catalyst_data(
        n_samples=500
    )
    
    print(f"Battery data: {battery_features.shape}")
    print(f"Catalyst data: {catalyst_features.shape}")
    
    # 创建数据加载器
    battery_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(battery_features),
        torch.FloatTensor(battery_voltage),
        torch.FloatTensor(battery_stability)
    )
    battery_loader = torch.utils.data.DataLoader(
        battery_dataset,
        batch_size=32,
        shuffle=True
    )
    
    catalyst_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(catalyst_features),
        torch.FloatTensor(catalyst_barrier),
        torch.FloatTensor(catalyst_stability)
    )
    catalyst_loader = torch.utils.data.DataLoader(
        catalyst_dataset,
        batch_size=32,
        shuffle=True
    )
    
    # 创建迁移模型
    config = BatteryCatalystConfig()
    transfer_model = BatteryToCatalystTransfer(config)
    
    # 第一阶段：在电池数据上预训练
    print("\n" + "=" * 70)
    print("Phase 1: Pre-training on battery data")
    print("=" * 70)
    
    history1 = transfer_model.pretrain_on_battery(battery_loader)
    
    # 第二阶段：适应到催化剂
    print("\n" + "=" * 70)
    print("Phase 2: Adapting to catalyst domain")
    print("=" * 70)
    
    history2 = transfer_model.adapt_to_catalyst(
        catalyst_loader,
        use_adversarial=True
    )
    
    # 测试预测
    print("\n" + "=" * 70)
    print("Testing Catalyst Property Prediction")
    print("=" * 70)
    
    test_features = catalyst_features[:100]
    predictions = transfer_model.predict_catalyst_properties(test_features)
    
    print(f"\nPredicted Energy Barrier: mean={predictions['energy'].mean():.3f} eV")
    print(f"Actual Energy Barrier: mean={catalyst_barrier[:100].mean():.3f} eV")
    print(f"\nPredicted Stability: mean={predictions['stability'].mean():.2f}%")
    print(f"Actual Stability: mean={catalyst_stability[:100].mean():.2f}%")
    
    # 分析学到的关系
    print("\n" + "=" * 70)
    print("Analyzing Learned Relationships")
    print("=" * 70)
    
    analysis = transfer_model.analyze_learned_relationships()
    
    for category, metrics in analysis.items():
        print(f"\n{category}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # 计算迁移效果
    print("\n" + "=" * 70)
    print("Transfer Effectiveness")
    print("=" * 70)
    
    # 计算预测误差
    energy_mae = np.mean(np.abs(predictions['energy'] - catalyst_barrier[:100]))
    stability_mae = np.mean(np.abs(predictions['stability'] - catalyst_stability[:100]))
    
    print(f"\nEnergy Barrier MAE: {energy_mae:.4f} eV")
    print(f"Stability MAE: {stability_mae:.2f}%")
    
    print("\n" + "=" * 70)
    print("Battery to Catalyst Transfer Demo Complete!")
    print("=" * 70)
