"""
金属→陶瓷迁移策略 (Metal to Ceramic Transfer)
实现从金属材料到陶瓷材料的知识迁移

作者: DFT-LAMMPS Team
版本: 1.0.0
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class MetalCeramicConfig:
    """金属-陶瓷迁移配置"""
    metal_feature_dim: int = 128
    ceramic_feature_dim: int = 128
    shared_dim: int = 64
    
    learning_rate: float = 1e-3
    num_epochs: int = 100
    batch_size: int = 32
    
    # 迁移配置
    contrastive_temperature: float = 0.07
    consistency_weight: float = 0.5
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MetalEncoder(nn.Module):
    """金属编码器"""
    
    def __init__(self, input_dim: int = 128, output_dim: int = 64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 192),
            nn.ReLU(),
            nn.BatchNorm1d(192),
            
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CeramicEncoder(nn.Module):
    """陶瓷编码器"""
    
    def __init__(self, input_dim: int = 128, output_dim: int = 64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 192),
            nn.ReLU(),
            nn.BatchNorm1d(192),
            
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MechanicalPropertyPredictor(nn.Module):
    """力学性质预测器 - 共享"""
    
    def __init__(self, input_dim: int = 64):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.BatchNorm1d(96),
            nn.Dropout(0.2),
            
            nn.Linear(96, 2)  # 硬度和韧性
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(x)


class ThermalPropertyPredictor(nn.Module):
    """热学性质预测器 - 陶瓷专用"""
    
    def __init__(self, input_dim: int = 64):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 热导率和热膨胀系数
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(x)


class NT_XentLoss(nn.Module):
    """归一化温度标度交叉熵损失 (对比学习)"""
    
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor
    ) -> torch.Tensor:
        """
        计算对比损失
        
        Args:
            z_i, z_j: 两组特征表示
        """
        batch_size = z_i.size(0)
        
        # 归一化
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)
        
        # 计算相似度矩阵
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.mm(representations, representations.t())
        
        # 掩码去除自身相似度
        mask = torch.eye(2 * batch_size, device=z_i.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
        
        # 正样本对 (i, i+batch_size) 和 (i+batch_size, i)
        positives = torch.cat([
            similarity_matrix[range(batch_size), range(batch_size, 2*batch_size)].unsqueeze(1),
            similarity_matrix[range(batch_size, 2*batch_size), range(batch_size)].unsqueeze(1)
        ], dim=0)
        
        # 负样本
        negatives = similarity_matrix[range(2*batch_size), :]
        
        # 计算logits
        logits = torch.cat([positives, negatives], dim=1) / self.temperature
        
        # 标签 (正样本在位置0)
        labels = torch.zeros(2 * batch_size, device=z_i.device).long()
        
        loss = nn.functional.cross_entropy(logits, labels)
        
        return loss


class MetalToCeramicTransfer:
    """金属到陶瓷知识迁移器"""
    
    def __init__(self, config: MetalCeramicConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 编码器
        self.metal_encoder = MetalEncoder(
            config.metal_feature_dim,
            config.shared_dim
        ).to(self.device)
        
        self.ceramic_encoder = CeramicEncoder(
            config.ceramic_feature_dim,
            config.shared_dim
        ).to(self.device)
        
        # 预测器
        self.mechanical_predictor = MechanicalPropertyPredictor(
            config.shared_dim
        ).to(self.device)
        
        self.thermal_predictor = ThermalPropertyPredictor(
            config.shared_dim
        ).to(self.device)
        
        # 优化器
        self.encoder_optimizer = torch.optim.Adam(
            list(self.metal_encoder.parameters()) +
            list(self.ceramic_encoder.parameters()),
            lr=config.learning_rate
        )
        
        self.predictor_optimizer = torch.optim.Adam(
            list(self.mechanical_predictor.parameters()) +
            list(self.thermal_predictor.parameters()),
            lr=config.learning_rate
        )
        
        self.contrastive_loss = NT_XentLoss(config.contrastive_temperature)
        self.mse_loss = nn.MSELoss()
    
    def train_metal_baseline(
        self,
        metal_loader: torch.utils.data.DataLoader
    ) -> Dict[str, List[float]]:
        """
        在金属数据上训练基线模型
        
        学习力学性质预测
        """
        print("Training metal baseline...")
        
        history = {"loss": [], "hardness_mae": [], "toughness_mae": []}
        
        for epoch in range(self.config.num_epochs):
            self.metal_encoder.train()
            self.mechanical_predictor.train()
            
            epoch_loss = 0.0
            epoch_hardness_mae = 0.0
            epoch_toughness_mae = 0.0
            num_batches = 0
            
            for features, hardness, toughness in metal_loader:
                features = features.to(self.device)
                hardness = hardness.to(self.device)
                toughness = toughness.to(self.device)
                
                # 编码
                embeddings = self.metal_encoder(features)
                
                # 预测
                predictions = self.mechanical_predictor(embeddings)
                hardness_pred = predictions[:, 0]
                toughness_pred = predictions[:, 1]
                
                # 损失
                loss = (
                    self.mse_loss(hardness_pred, hardness) +
                    self.mse_loss(toughness_pred, toughness)
                ) / 2
                
                # 反向传播
                self.encoder_optimizer.zero_grad()
                self.predictor_optimizer.zero_grad()
                loss.backward()
                self.encoder_optimizer.step()
                self.predictor_optimizer.step()
                
                epoch_loss += loss.item()
                epoch_hardness_mae += torch.abs(hardness_pred - hardness).mean().item()
                epoch_toughness_mae += torch.abs(toughness_pred - toughness).mean().item()
                num_batches += 1
            
            history["loss"].append(epoch_loss / num_batches)
            history["hardness_mae"].append(epoch_hardness_mae / num_batches)
            history["toughness_mae"].append(epoch_toughness_mae / num_batches)
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: Loss={epoch_loss/num_batches:.4f}")
        
        return history
    
    def contrastive_alignment(
        self,
        metal_loader: torch.utils.data.DataLoader,
        ceramic_loader: torch.utils.data.DataLoader,
        num_epochs: int = 50
    ) -> Dict[str, List[float]]:
        """
        对比学习对齐
        
        通过对比学习对齐金属和陶瓷的表示空间
        """
        print("Performing contrastive alignment...")
        
        history = {"contrastive_loss": []}
        
        for epoch in range(num_epochs):
            self.metal_encoder.train()
            self.ceramic_encoder.train()
            
            epoch_loss = 0.0
            num_batches = 0
            
            metal_iter = iter(metal_loader)
            ceramic_iter = iter(ceramic_loader)
            
            for _ in range(min(len(metal_loader), len(ceramic_loader))):
                try:
                    metal_batch = next(metal_iter)[0].to(self.device)
                    ceramic_batch = next(ceramic_iter)[0].to(self.device)
                    
                    # 确保批次大小相同
                    min_batch = min(metal_batch.size(0), ceramic_batch.size(0))
                    metal_batch = metal_batch[:min_batch]
                    ceramic_batch = ceramic_batch[:min_batch]
                    
                    # 编码
                    metal_embeddings = self.metal_encoder(metal_batch)
                    ceramic_embeddings = self.ceramic_encoder(ceramic_batch)
                    
                    # 对比损失
                    loss = self.contrastive_loss(metal_embeddings, ceramic_embeddings)
                    
                    # 反向传播
                    self.encoder_optimizer.zero_grad()
                    loss.backward()
                    self.encoder_optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                except StopIteration:
                    break
            
            avg_loss = epoch_loss / max(num_batches, 1)
            history["contrastive_loss"].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Contrastive Loss={avg_loss:.4f}")
        
        return history
    
    def finetune_on_ceramic(
        self,
        ceramic_loader: torch.utils.data.DataLoader,
        freeze_encoder: bool = False
    ) -> Dict[str, List[float]]:
        """
        在陶瓷数据上微调
        
        利用从金属学到的力学知识预测陶瓷性质
        """
        print("Fine-tuning on ceramic data...")
        
        # 复制金属编码器权重到陶瓷编码器
        self._align_encoders()
        
        if freeze_encoder:
            for param in self.ceramic_encoder.parameters():
                param.requires_grad = False
        
        history = {
            "total_loss": [],
            "mechanical_loss": [],
            "thermal_loss": []
        }
        
        for epoch in range(self.config.num_epochs):
            self.ceramic_encoder.train()
            self.mechanical_predictor.train()
            self.thermal_predictor.train()
            
            epoch_total = 0.0
            epoch_mech = 0.0
            epoch_thermal = 0.0
            num_batches = 0
            
            for (features, hardness, toughness,
                 thermal_cond, thermal_exp) in ceramic_loader:
                features = features.to(self.device)
                hardness = hardness.to(self.device)
                toughness = toughness.to(self.device)
                thermal_cond = thermal_cond.to(self.device)
                thermal_exp = thermal_exp.to(self.device)
                
                # 编码
                embeddings = self.ceramic_encoder(features)
                
                # 力学预测 (从金属迁移)
                mech_pred = self.mechanical_predictor(embeddings)
                hardness_pred = mech_pred[:, 0]
                toughness_pred = mech_pred[:, 1]
                
                mech_loss = (
                    self.mse_loss(hardness_pred, hardness) +
                    self.mse_loss(toughness_pred, toughness)
                ) / 2
                
                # 热学预测 (陶瓷特有)
                thermal_pred = self.thermal_predictor(embeddings)
                cond_pred = thermal_pred[:, 0]
                exp_pred = thermal_pred[:, 1]
                
                thermal_loss = (
                    self.mse_loss(cond_pred, thermal_cond) +
                    self.mse_loss(exp_pred, thermal_exp)
                ) / 2
                
                # 总损失
                total_loss = mech_loss + self.config.consistency_weight * thermal_loss
                
                # 反向传播
                self.encoder_optimizer.zero_grad()
                self.predictor_optimizer.zero_grad()
                total_loss.backward()
                self.encoder_optimizer.step()
                self.predictor_optimizer.step()
                
                epoch_total += total_loss.item()
                epoch_mech += mech_loss.item()
                epoch_thermal += thermal_loss.item()
                num_batches += 1
            
            history["total_loss"].append(epoch_total / num_batches)
            history["mechanical_loss"].append(epoch_mech / num_batches)
            history["thermal_loss"].append(epoch_thermal / num_batches)
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: Total Loss={epoch_total/num_batches:.4f}")
        
        return history
    
    def _align_encoders(self):
        """对齐编码器权重"""
        metal_state = self.metal_encoder.state_dict()
        ceramic_state = self.ceramic_encoder.state_dict()
        
        # 部分权重初始化
        for key in metal_state:
            if key in ceramic_state and "encoder" in key:
                if metal_state[key].shape == ceramic_state[key].shape:
                    ceramic_state[key] = metal_state[key].clone()
        
        self.ceramic_encoder.load_state_dict(ceramic_state)
        print("Encoder weights aligned")
    
    def predict_ceramic_properties(
        self,
        features: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        预测陶瓷性质
        """
        self.ceramic_encoder.eval()
        self.mechanical_predictor.eval()
        self.thermal_predictor.eval()
        
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            embeddings = self.ceramic_encoder(features_tensor)
            
            mech_pred = self.mechanical_predictor(embeddings)
            thermal_pred = self.thermal_predictor(embeddings)
        
        return {
            "hardness": mech_pred[:, 0].cpu().numpy(),
            "toughness": mech_pred[:, 1].cpu().numpy(),
            "thermal_conductivity": thermal_pred[:, 0].cpu().numpy(),
            "thermal_expansion": thermal_pred[:, 1].cpu().numpy(),
            "embedding": embeddings.cpu().numpy()
        }
    
    def evaluate_transfer(self, test_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """评估迁移效果"""
        self.ceramic_encoder.eval()
        self.mechanical_predictor.eval()
        self.thermal_predictor.eval()
        
        metrics = {
            "hardness_mae": 0.0,
            "toughness_mae": 0.0,
            "thermal_cond_mae": 0.0,
            "thermal_exp_mae": 0.0,
            "count": 0
        }
        
        with torch.no_grad():
            for (features, hardness, toughness,
                 thermal_cond, thermal_exp) in test_loader:
                features = features.to(self.device)
                
                embeddings = self.ceramic_encoder(features)
                
                mech_pred = self.mechanical_predictor(embeddings)
                thermal_pred = self.thermal_predictor(embeddings)
                
                metrics["hardness_mae"] += torch.abs(
                    mech_pred[:, 0].cpu() - hardness
                ).sum().item()
                metrics["toughness_mae"] += torch.abs(
                    mech_pred[:, 1].cpu() - toughness
                ).sum().item()
                metrics["thermal_cond_mae"] += torch.abs(
                    thermal_pred[:, 0].cpu() - thermal_cond
                ).sum().item()
                metrics["thermal_exp_mae"] += torch.abs(
                    thermal_pred[:, 1].cpu() - thermal_exp
                ).sum().item()
                metrics["count"] += len(features)
        
        for key in ["hardness_mae", "toughness_mae",
                    "thermal_cond_mae", "thermal_exp_mae"]:
            metrics[key] /= metrics["count"]
        
        del metrics["count"]
        
        return metrics


def generate_metal_data(
    n_samples: int = 1000,
    feature_dim: int = 128
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成金属数据
    
    特征：晶体结构、电子结构
    目标：硬度、韧性
    """
    np.random.seed(42)
    
    features = np.random.randn(n_samples, feature_dim)
    
    # 金属特征
    features[:, 0] = np.random.uniform(2.0, 6.0, n_samples)   # 电导率
    features[:, 1] = np.random.uniform(0.5, 3.0, n_samples)   # 费米能级
    features[:, 2] = np.random.uniform(0.5, 2.0, n_samples)   # 德拜温度
    
    # 硬度 (GPa) - 与德拜温度和电导率相关
    hardness = (
        0.5 + 2.0 * features[:, 2] + 1.5 * features[:, 0] +
        2 * np.random.randn(n_samples)
    )
    hardness = np.clip(hardness, 0.5, 20.0)
    
    # 韧性 (MPa·m^0.5) - 通常与硬度呈反比
    toughness = (
        100 - 3 * hardness + 10 * features[:, 1] +
        10 * np.random.randn(n_samples)
    )
    toughness = np.clip(toughness, 10, 150)
    
    return features, hardness, toughness


def generate_ceramic_data(
    n_samples: int = 500,
    feature_dim: int = 128
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成陶瓷数据
    
    特征：晶格结构、离子性
    目标：硬度、韧性、热导率、热膨胀系数
    """
    np.random.seed(43)
    
    features = np.random.randn(n_samples, feature_dim)
    
    # 陶瓷特征
    features[:, 0] = np.random.uniform(5.0, 20.0, n_samples)  # 离子性
    features[:, 1] = np.random.uniform(2.0, 15.0, n_samples)  # 晶格能
    features[:, 2] = np.random.uniform(100, 2000, n_samples)  # 德拜温度
    
    # 硬度 (GPa) - 陶瓷通常比金属硬
    hardness = (
        5.0 + 0.5 * features[:, 1] + 0.01 * features[:, 2] +
        5 * np.random.randn(n_samples)
    )
    hardness = np.clip(hardness, 5.0, 50.0)
    
    # 韧性 (MPa·m^0.5) - 陶瓷通常比金属脆
    toughness = (
        50 - hardness + 2 * features[:, 0] +
        10 * np.random.randn(n_samples)
    )
    toughness = np.clip(toughness, 1, 100)
    
    # 热导率 (W/m·K)
    thermal_cond = (
        50 + 0.1 * features[:, 2] - 0.5 * features[:, 0] +
        20 * np.random.randn(n_samples)
    )
    thermal_cond = np.clip(thermal_cond, 1, 500)
    
    # 热膨胀系数 (10^-6 /K)
    thermal_exp = (
        15 - 0.005 * features[:, 2] + 0.1 * features[:, 0] +
        2 * np.random.randn(n_samples)
    )
    thermal_exp = np.clip(thermal_exp, 0.5, 20)
    
    return features, hardness, toughness, thermal_cond, thermal_exp


# 演示代码
if __name__ == "__main__":
    print("=" * 70)
    print("金属→陶瓷迁移策略演示 (Metal to Ceramic Transfer Demo)")
    print("=" * 70)
    
    # 生成数据
    print("\nGenerating metal and ceramic data...")
    
    metal_features, metal_hardness, metal_toughness = generate_metal_data(
        n_samples=1000
    )
    
    ceramic_features, ceramic_hardness, ceramic_toughness, \
    ceramic_thermal_cond, ceramic_thermal_exp = generate_ceramic_data(
        n_samples=500
    )
    
    print(f"Metal data: {metal_features.shape}")
    print(f"Ceramic data: {ceramic_features.shape}")
    
    # 创建数据加载器
    metal_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(metal_features),
        torch.FloatTensor(metal_hardness),
        torch.FloatTensor(metal_toughness)
    )
    metal_loader = torch.utils.data.DataLoader(
        metal_dataset,
        batch_size=32,
        shuffle=True
    )
    
    ceramic_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(ceramic_features),
        torch.FloatTensor(ceramic_hardness),
        torch.FloatTensor(ceramic_toughness),
        torch.FloatTensor(ceramic_thermal_cond),
        torch.FloatTensor(ceramic_thermal_exp)
    )
    ceramic_loader = torch.utils.data.DataLoader(
        ceramic_dataset,
        batch_size=32,
        shuffle=True
    )
    
    # 创建迁移模型
    config = MetalCeramicConfig()
    transfer_model = MetalToCeramicTransfer(config)
    
    # 第一阶段：金属基线训练
    print("\n" + "=" * 70)
    print("Phase 1: Training Metal Baseline")
    print("=" * 70)
    
    history1 = transfer_model.train_metal_baseline(metal_loader)
    
    # 第二阶段：对比对齐
    print("\n" + "=" * 70)
    print("Phase 2: Contrastive Alignment")
    print("=" * 70)
    
    history2 = transfer_model.contrastive_alignment(
        metal_loader,
        ceramic_loader,
        num_epochs=30
    )
    
    # 第三阶段：陶瓷微调
    print("\n" + "=" * 70)
    print("Phase 3: Fine-tuning on Ceramic Data")
    print("=" * 70)
    
    history3 = transfer_model.finetune_on_ceramic(
        ceramic_loader,
        freeze_encoder=False
    )
    
    # 测试
    print("\n" + "=" * 70)
    print("Testing Ceramic Property Prediction")
    print("=" * 70)
    
    test_features = ceramic_features[:100]
    predictions = transfer_model.predict_ceramic_properties(test_features)
    
    print(f"\nHardness - Predicted: {predictions['hardness'].mean():.2f} GPa, "
          f"Actual: {ceramic_hardness[:100].mean():.2f} GPa")
    print(f"Toughness - Predicted: {predictions['toughness'].mean():.2f} MPa·m^0.5, "
          f"Actual: {ceramic_toughness[:100].mean():.2f} MPa·m^0.5")
    print(f"Thermal Conductivity - Predicted: {predictions['thermal_conductivity'].mean():.2f} W/m·K, "
          f"Actual: {ceramic_thermal_cond[:100].mean():.2f} W/m·K")
    print(f"Thermal Expansion - Predicted: {predictions['thermal_expansion'].mean():.2f} 10^-6/K, "
          f"Actual: {ceramic_thermal_exp[:100].mean():.2f} 10^-6/K")
    
    # 评估
    print("\n" + "=" * 70)
    print("Transfer Evaluation")
    print("=" * 70)
    
    metrics = transfer_model.evaluate_transfer(ceramic_loader)
    
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n" + "=" * 70)
    print("Metal to Ceramic Transfer Demo Complete!")
    print("=" * 70)
