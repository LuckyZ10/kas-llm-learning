"""
域适配器 (Domain Adapter)
实现特征对齐、分布匹配等域适应技术

作者: DFT-LAMMPS Team
版本: 1.0.0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


@dataclass
class DomainAdaptationConfig:
    """域适应配置"""
    method: str = "dann"  # dann, mmd, coral, jdot, wdann
    input_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 64
    lambda_adapt: float = 1.0  # 适应损失权重
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # MMD specific
    mmd_kernel: str = "rbf"  # rbf, linear, polynomial
    mmd_bandwidth: float = 1.0
    
    # CORAL specific
    coral_lambda: float = 1.0
    
    # JDOT specific
    jdot_alpha: float = 1.0
    jdot_metric: str = "euclidean"


class GradientReversalLayer(torch.autograd.Function):
    """梯度反转层 (GRL) - 用于DANN"""
    
    @staticmethod
    def forward(ctx, x, lambda_p):
        ctx.lambda_p = lambda_p
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_p * grad_output, None


class DomainDiscriminator(nn.Module):
    """域判别器 - 用于对抗性域适应"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            in_dim = hidden_dim
            hidden_dim = max(hidden_dim // 2, 64)
        
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FeatureExtractor(nn.Module):
    """特征提取器"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 64,
        num_layers: int = 3
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class LabelPredictor(nn.Module):
    """标签预测器"""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MMDLoss(nn.Module):
    """最大均值差异损失"""
    
    def __init__(self, kernel: str = "rbf", bandwidth: float = 1.0):
        super().__init__()
        self.kernel = kernel
        self.bandwidth = bandwidth
    
    def rbf_kernel(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        bandwidth: float = None
    ) -> torch.Tensor:
        """RBF核函数"""
        if bandwidth is None:
            bandwidth = self.bandwidth
        
        XX = torch.sum(X ** 2, dim=1, keepdim=True)
        YY = torch.sum(Y ** 2, dim=1, keepdim=True)
        XY = torch.mm(X, Y.t())
        
        distances = XX + YY.t() - 2 * XY
        return torch.exp(-distances / (2 * bandwidth ** 2))
    
    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """计算MMD损失"""
        n_source = source_features.size(0)
        n_target = target_features.size(0)
        
        if self.kernel == "rbf":
            K_ss = self.rbf_kernel(source_features, source_features)
            K_tt = self.rbf_kernel(target_features, target_features)
            K_st = self.rbf_kernel(source_features, target_features)
            
            loss = (K_ss.sum() - K_ss.trace()) / (n_source * (n_source - 1))
            loss += (K_tt.sum() - K_tt.trace()) / (n_target * (n_target - 1))
            loss -= 2 * K_st.sum() / (n_source * n_target)
        else:
            # Linear kernel
            source_mean = source_features.mean(dim=0)
            target_mean = target_features.mean(dim=0)
            loss = torch.sum((source_mean - target_mean) ** 2)
        
        return loss


class CORALLoss(nn.Module):
    """CORAL - Correlation Alignment Loss"""
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """计算CORAL损失"""
        n_source = source_features.size(0)
        n_target = target_features.size(0)
        
        # 计算协方差矩阵
        source_cov = torch.mm(source_features.t(), source_features) / n_source
        target_cov = torch.mm(target_features.t(), target_features) / n_target
        
        # Frobenius范数
        loss = torch.sum((source_cov - target_cov) ** 2)
        
        return loss / (4 * source_features.size(1) ** 2)


class WassersteinLoss(nn.Module):
    """Wasserstein距离损失 - 用于WDANN"""
    
    def __init__(self, clip_value: float = 0.01):
        super().__init__()
        self.clip_value = clip_value
    
    def forward(
        self,
        source_logits: torch.Tensor,
        target_logits: torch.Tensor
    ) -> torch.Tensor:
        """计算Wasserstein距离"""
        return -torch.mean(source_logits) + torch.mean(target_logits)


class DomainAdapter(ABC):
    """域适配器基类"""
    
    def __init__(self, config: DomainAdaptationConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "adaptation_loss": [],
            "domain_acc": []
        }
    
    @abstractmethod
    def fit(
        self,
        source_features: np.ndarray,
        source_labels: np.ndarray,
        target_features: np.ndarray,
        target_labels: Optional[np.ndarray] = None
    ) -> Dict:
        """训练域适配器"""
        pass
    
    @abstractmethod
    def transform(self, features: np.ndarray) -> np.ndarray:
        """特征转换"""
        pass
    
    def fit_transform(
        self,
        source_features: np.ndarray,
        source_labels: np.ndarray,
        target_features: np.ndarray,
        target_labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """训练并转换"""
        self.fit(source_features, source_labels, target_features, target_labels)
        return (
            self.transform(source_features),
            self.transform(target_features)
        )


class DANNAdapter(DomainAdapter):
    """Domain-Adversarial Neural Network (DANN) 适配器"""
    
    def __init__(self, config: DomainAdaptationConfig, num_classes: int = 2):
        super().__init__(config)
        self.num_classes = num_classes
        
        self.feature_extractor = FeatureExtractor(
            config.input_dim,
            config.hidden_dim,
            config.output_dim
        ).to(self.device)
        
        self.label_predictor = LabelPredictor(
            config.output_dim,
            num_classes
        ).to(self.device)
        
        self.domain_discriminator = DomainDiscriminator(
            config.output_dim
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) +
            list(self.label_predictor.parameters()) +
            list(self.domain_discriminator.parameters()),
            lr=config.learning_rate
        )
        
        self.class_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.BCELoss()
    
    def _train_epoch(
        self,
        source_loader: torch.utils.data.DataLoader,
        target_loader: torch.utils.data.DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """训练一个epoch"""
        self.feature_extractor.train()
        self.label_predictor.train()
        self.domain_discriminator.train()
        
        total_loss = 0.0
        total_class_loss = 0.0
        total_domain_loss = 0.0
        total_domain_acc = 0.0
        num_batches = 0
        
        # 进度lambda
        p = float(epoch) / self.config.num_epochs
        lambda_p = 2. / (1. + np.exp(-10 * p)) - 1
        
        # 合并迭代
        target_iter = iter(target_loader)
        
        for source_data, source_labels in source_loader:
            try:
                target_data, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_data, _ = next(target_iter)
            
            source_data = source_data.to(self.device)
            source_labels = source_labels.to(self.device)
            target_data = target_data.to(self.device)
            
            batch_size = source_data.size(0)
            
            # 域标签
            source_domain = torch.zeros(batch_size, 1).to(self.device)
            target_domain = torch.ones(target_data.size(0), 1).to(self.device)
            
            # 特征提取
            source_features = self.feature_extractor(source_data)
            target_features = self.feature_extractor(target_data)
            
            # 标签预测
            source_preds = self.label_predictor(source_features)
            class_loss = self.class_criterion(source_preds, source_labels)
            
            # 域判别 (带梯度反转)
            reversed_source = GradientReversalLayer.apply(source_features, lambda_p)
            reversed_target = GradientReversalLayer.apply(target_features, lambda_p)
            
            source_domain_pred = self.domain_discriminator(reversed_source)
            target_domain_pred = self.domain_discriminator(reversed_target)
            
            domain_loss = (
                self.domain_criterion(source_domain_pred, source_domain) +
                self.domain_criterion(target_domain_pred, target_domain)
            ) / 2
            
            # 总损失
            loss = class_loss + self.config.lambda_adapt * domain_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_class_loss += class_loss.item()
            total_domain_loss += domain_loss.item()
            
            # 域判别准确率
            source_pred_domain = (source_domain_pred > 0.5).float()
            target_pred_domain = (target_domain_pred > 0.5).float()
            domain_acc = (
                (source_pred_domain == source_domain).float().mean() +
                (target_pred_domain == target_domain).float().mean()
            ) / 2
            total_domain_acc += domain_acc.item()
            
            num_batches += 1
        
        return {
            "loss": total_loss / num_batches,
            "class_loss": total_class_loss / num_batches,
            "domain_loss": total_domain_loss / num_batches,
            "domain_acc": total_domain_acc / num_batches
        }
    
    def fit(
        self,
        source_features: np.ndarray,
        source_labels: np.ndarray,
        target_features: np.ndarray,
        target_labels: Optional[np.ndarray] = None
    ) -> Dict:
        """训练DANN"""
        # 创建数据加载器
        source_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(source_features),
            torch.LongTensor(source_labels)
        )
        source_loader = torch.utils.data.DataLoader(
            source_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        target_labels_dummy = torch.zeros(len(target_features))
        target_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(target_features),
            target_labels_dummy
        )
        target_loader = torch.utils.data.DataLoader(
            target_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        for epoch in range(self.config.num_epochs):
            metrics = self._train_epoch(source_loader, target_loader, epoch)
            
            self.history["train_loss"].append(metrics["loss"])
            self.history["adaptation_loss"].append(metrics["domain_loss"])
            self.history["domain_acc"].append(metrics["domain_acc"])
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                      f"Loss={metrics['loss']:.4f}, "
                      f"Class Loss={metrics['class_loss']:.4f}, "
                      f"Domain Loss={metrics['domain_loss']:.4f}, "
                      f"Domain Acc={metrics['domain_acc']:.4f}")
        
        return self.history
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """特征转换"""
        self.feature_extractor.eval()
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)
            transformed = self.feature_extractor(features_tensor)
        
        return transformed.cpu().numpy()
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """标签预测"""
        self.feature_extractor.eval()
        self.label_predictor.eval()
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)
            extracted = self.feature_extractor(features_tensor)
            logits = self.label_predictor(extracted)
            preds = torch.argmax(logits, dim=1)
        
        return preds.cpu().numpy()


class MMDAdapter(DomainAdapter):
    """MMD-based 域适配器"""
    
    def __init__(self, config: DomainAdaptationConfig, num_classes: int = 2):
        super().__init__(config)
        self.num_classes = num_classes
        
        self.feature_extractor = FeatureExtractor(
            config.input_dim,
            config.hidden_dim,
            config.output_dim
        ).to(self.device)
        
        self.label_predictor = LabelPredictor(
            config.output_dim,
            num_classes
        ).to(self.device)
        
        self.mmd_loss = MMDLoss(
            kernel=config.mmd_kernel,
            bandwidth=config.mmd_bandwidth
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) +
            list(self.label_predictor.parameters()),
            lr=config.learning_rate
        )
        
        self.class_criterion = nn.CrossEntropyLoss()
    
    def fit(
        self,
        source_features: np.ndarray,
        source_labels: np.ndarray,
        target_features: np.ndarray,
        target_labels: Optional[np.ndarray] = None
    ) -> Dict:
        """训练MMD适配器"""
        source_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(source_features),
            torch.LongTensor(source_labels)
        )
        source_loader = torch.utils.data.DataLoader(
            source_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        target_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(target_features),
            torch.zeros(len(target_features))  # dummy
        )
        target_loader = torch.utils.data.DataLoader(
            target_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        for epoch in range(self.config.num_epochs):
            self.feature_extractor.train()
            self.label_predictor.train()
            
            total_loss = 0.0
            total_class_loss = 0.0
            total_mmd_loss = 0.0
            num_batches = 0
            
            target_iter = iter(target_loader)
            
            for source_data, source_labels in source_loader:
                try:
                    target_data, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    target_data, _ = next(target_iter)
                
                source_data = source_data.to(self.device)
                source_labels = source_labels.to(self.device)
                target_data = target_data.to(self.device)
                
                # 特征提取
                source_feats = self.feature_extractor(source_data)
                target_feats = self.feature_extractor(target_data)
                
                # 标签预测
                source_preds = self.label_predictor(source_feats)
                class_loss = self.class_criterion(source_preds, source_labels)
                
                # MMD损失
                mmd_loss = self.mmd_loss(source_feats, target_feats)
                
                # 总损失
                loss = class_loss + self.config.lambda_adapt * mmd_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_class_loss += class_loss.item()
                total_mmd_loss += mmd_loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            avg_mmd = total_mmd_loss / num_batches
            
            self.history["train_loss"].append(avg_loss)
            self.history["adaptation_loss"].append(avg_mmd)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                      f"Loss={avg_loss:.4f}, MMD Loss={avg_mmd:.4f}")
        
        return self.history
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """特征转换"""
        self.feature_extractor.eval()
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)
            transformed = self.feature_extractor(features_tensor)
        
        return transformed.cpu().numpy()
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """标签预测"""
        self.feature_extractor.eval()
        self.label_predictor.eval()
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)
            extracted = self.feature_extractor(features_tensor)
            logits = self.label_predictor(extracted)
            preds = torch.argmax(logits, dim=1)
        
        return preds.cpu().numpy()


class CORALAdapter(DomainAdapter):
    """CORAL - Correlation Alignment 适配器"""
    
    def __init__(self, config: DomainAdaptationConfig, num_classes: int = 2):
        super().__init__(config)
        self.num_classes = num_classes
        
        self.feature_extractor = FeatureExtractor(
            config.input_dim,
            config.hidden_dim,
            config.output_dim
        ).to(self.device)
        
        self.label_predictor = LabelPredictor(
            config.output_dim,
            num_classes
        ).to(self.device)
        
        self.coral_loss = CORALLoss().to(self.device)
        
        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) +
            list(self.label_predictor.parameters()),
            lr=config.learning_rate
        )
        
        self.class_criterion = nn.CrossEntropyLoss()
    
    def fit(
        self,
        source_features: np.ndarray,
        source_labels: np.ndarray,
        target_features: np.ndarray,
        target_labels: Optional[np.ndarray] = None
    ) -> Dict:
        """训练CORAL适配器"""
        source_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(source_features),
            torch.LongTensor(source_labels)
        )
        source_loader = torch.utils.data.DataLoader(
            source_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        target_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(target_features),
            torch.zeros(len(target_features))
        )
        target_loader = torch.utils.data.DataLoader(
            target_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        for epoch in range(self.config.num_epochs):
            self.feature_extractor.train()
            self.label_predictor.train()
            
            total_loss = 0.0
            total_class_loss = 0.0
            total_coral_loss = 0.0
            num_batches = 0
            
            target_iter = iter(target_loader)
            
            for source_data, source_labels in source_loader:
                try:
                    target_data, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    target_data, _ = next(target_iter)
                
                source_data = source_data.to(self.device)
                source_labels = source_labels.to(self.device)
                target_data = target_data.to(self.device)
                
                source_feats = self.feature_extractor(source_data)
                target_feats = self.feature_extractor(target_data)
                
                source_preds = self.label_predictor(source_feats)
                class_loss = self.class_criterion(source_preds, source_labels)
                
                coral_loss = self.coral_loss(source_feats, target_feats)
                
                loss = class_loss + self.config.coral_lambda * coral_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_class_loss += class_loss.item()
                total_coral_loss += coral_loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            avg_coral = total_coral_loss / num_batches
            
            self.history["train_loss"].append(avg_loss)
            self.history["adaptation_loss"].append(avg_coral)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                      f"Loss={avg_loss:.4f}, CORAL Loss={avg_coral:.4f}")
        
        return self.history
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """特征转换"""
        self.feature_extractor.eval()
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)
            transformed = self.feature_extractor(features_tensor)
        
        return transformed.cpu().numpy()
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """标签预测"""
        self.feature_extractor.eval()
        self.label_predictor.eval()
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)
            extracted = self.feature_extractor(features_tensor)
            logits = self.label_predictor(extracted)
            preds = torch.argmax(logits, dim=1)
        
        return preds.cpu().numpy()


class DeepCORAL(CORALAdapter):
    """Deep CORAL - CORAL的深度版本"""
    
    def __init__(self, config: DomainAdaptationConfig, num_classes: int = 2):
        super().__init__(config, num_classes)
        self.name = "DeepCORAL"


class SubspaceAlignment:
    """子空间对齐 - 基于PCA的域适应"""
    
    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.source_pca = None
        self.target_pca = None
        self.alignment_matrix = None
    
    def fit(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray
    ) -> "SubspaceAlignment":
        """拟合子空间对齐"""
        # PCA降维
        self.source_pca = PCA(n_components=self.n_components)
        self.target_pca = PCA(n_components=self.n_components)
        
        source_subspace = self.source_pca.fit_transform(source_features)
        target_subspace = self.target_pca.fit_transform(target_features)
        
        # 计算对齐矩阵
        source_basis = self.source_pca.components_.T
        target_basis = self.target_pca.components_.T
        
        self.alignment_matrix = np.dot(source_basis.T, target_basis)
        
        return self
    
    def transform(self, features: np.ndarray, domain: str = "source") -> np.ndarray:
        """转换特征"""
        if domain == "source":
            return self.source_pca.transform(features)
        else:
            return self.target_pca.transform(features)
    
    def align_target_to_source(
        self,
        target_features: np.ndarray
    ) -> np.ndarray:
        """将目标域对齐到源域"""
        target_subspace = self.target_pca.transform(target_features)
        aligned = np.dot(target_subspace, self.alignment_matrix.T)
        return aligned


class JointDistributionOptimalTransport:
    """JDOT - 联合分布最优传输"""
    
    def __init__(
        self,
        alpha: float = 1.0,
        metric: str = "euclidean",
        max_iter: int = 100
    ):
        self.alpha = alpha
        self.metric = metric
        self.max_iter = max_iter
        self.transport_plan = None
    
    def compute_cost_matrix(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
        source_labels: np.ndarray,
        target_labels: np.ndarray
    ) -> np.ndarray:
        """计算联合分布成本矩阵"""
        from scipy.spatial.distance import cdist
        
        # 特征距离
        feat_cost = cdist(source_features, target_features, metric=self.metric)
        
        # 标签距离
        label_cost = cdist(
            source_labels.reshape(-1, 1),
            target_labels.reshape(-1, 1),
            metric=self.metric
        )
        
        # 联合成本
        return feat_cost + self.alpha * label_cost
    
    def sinkhorn_knopp(
        self,
        cost_matrix: np.ndarray,
        reg: float = 0.1,
        numItermax: int = 1000
    ) -> np.ndarray:
        """Sinkhorn-Knopp算法求解最优传输"""
        n_source, n_target = cost_matrix.shape
        
        # 初始化
        K = np.exp(-cost_matrix / reg)
        u = np.ones(n_source) / n_source
        v = np.ones(n_target) / n_target
        
        for _ in range(numItermax):
            u_prev = u
            u = 1.0 / (K @ v)
            v = 1.0 / (K.T @ u)
            
            if np.max(np.abs(u - u_prev)) < 1e-8:
                break
        
        transport_plan = np.diag(u) @ K @ np.diag(v)
        return transport_plan
    
    def fit_transform(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
        source_labels: np.ndarray,
        target_labels: np.ndarray
    ) -> np.ndarray:
        """拟合并转换"""
        cost_matrix = self.compute_cost_matrix(
            source_features, target_features,
            source_labels, target_labels
        )
        
        self.transport_plan = self.sinkhorn_knopp(cost_matrix)
        
        # 使用传输计划进行特征转换
        transported = self.transport_plan @ target_features
        
        return transported


class DomainAdaptationFactory:
    """域适应工厂"""
    
    @staticmethod
    def create_adapter(
        method: str,
        config: DomainAdaptationConfig,
        num_classes: int = 2
    ) -> DomainAdapter:
        """创建域适配器"""
        adapters = {
            "dann": DANNAdapter,
            "mmd": MMDAdapter,
            "coral": CORALAdapter,
            "deep_coral": DeepCORAL
        }
        
        if method not in adapters:
            raise ValueError(f"Unknown method: {method}. "
                           f"Available: {list(adapters.keys())}")
        
        return adapters[method](config, num_classes)


def visualize_domain_alignment(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_labels: Optional[np.ndarray] = None,
    target_labels: Optional[np.ndarray] = None,
    method: str = "tsne",
    title: str = "Domain Alignment Visualization"
) -> plt.Figure:
    """可视化域对齐效果"""
    # 合并特征
    all_features = np.vstack([source_features, target_features])
    n_source = len(source_features)
    
    # 域标签
    domain_labels = np.array([0] * n_source + [1] * len(target_features))
    
    # 降维
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)
    
    embedded = reducer.fit_transform(all_features)
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 按域着色
    ax = axes[0]
    ax.scatter(
        embedded[:n_source, 0],
        embedded[:n_source, 1],
        c="blue",
        alpha=0.6,
        label="Source",
        s=30
    )
    ax.scatter(
        embedded[n_source:, 0],
        embedded[n_source:, 1],
        c="red",
        alpha=0.6,
        label="Target",
        s=30
    )
    ax.set_title("By Domain")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 按类别着色
    ax = axes[1]
    if source_labels is not None and target_labels is not None:
        all_labels = np.concatenate([source_labels, target_labels])
        scatter = ax.scatter(
            embedded[:, 0],
            embedded[:, 1],
            c=all_labels,
            cmap="tab10",
            alpha=0.6,
            s=30
        )
        ax.set_title("By Class")
        plt.colorbar(scatter, ax=ax)
    else:
        ax.text(0.5, 0.5, "No labels provided",
                ha="center", va="center")
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    return fig


def compute_domain_discrepancy(
    source_features: np.ndarray,
    target_features: np.ndarray,
    metric: str = "mmd"
) -> float:
    """计算域间差异度量"""
    if metric == "mmd":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mmd = MMDLoss().to(device)
        
        source_tensor = torch.FloatTensor(source_features).to(device)
        target_tensor = torch.FloatTensor(target_features).to(device)
        
        with torch.no_grad():
            discrepancy = mmd(source_tensor, target_tensor)
        
        return discrepancy.item()
    
    elif metric == "coral":
        coral = CORALLoss()
        
        source_tensor = torch.FloatTensor(source_features)
        target_tensor = torch.FloatTensor(target_features)
        
        with torch.no_grad():
            discrepancy = coral(source_tensor, target_tensor)
        
        return discrepancy.item()
    
    elif metric == "mean_shift":
        source_mean = source_features.mean(axis=0)
        target_mean = target_features.mean(axis=0)
        return np.linalg.norm(source_mean - target_mean)
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def progressive_domain_adaptation(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
    intermediate_domains: List[np.ndarray],
    config: DomainAdaptationConfig,
    num_classes: int = 2
) -> np.ndarray:
    """渐进式域适应 - 通过中间域逐步迁移"""
    print("Starting Progressive Domain Adaptation...")
    
    adapter = MMDAdapter(config, num_classes)
    
    # 当前域
    current_features = source_features.copy()
    current_labels = source_labels.copy()
    
    # 依次通过中间域
    for i, intermediate in enumerate(intermediate_domains):
        print(f"Adapting through intermediate domain {i+1}/{len(intermediate_domains)}...")
        
        # 适应到中间域
        adapter.fit(current_features, current_labels, intermediate)
        current_features = adapter.transform(current_features)
        
        # 更新中间域特征
        intermediate_domains[i] = adapter.transform(intermediate)
    
    # 最终适应到目标域
    print("Final adaptation to target domain...")
    adapter.fit(current_features, current_labels, target_features)
    
    return adapter.transform(target_features)


# 演示代码
if __name__ == "__main__":
    print("=" * 60)
    print("域适配器演示 (Domain Adapter Demo)")
    print("=" * 60)
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 500
    n_features = 100
    
    # 源域数据
    source_features = np.random.randn(n_samples, n_features)
    source_labels = np.random.randint(0, 3, n_samples)
    
    # 目标域数据 (添加分布偏移)
    target_features = np.random.randn(n_samples, n_features) + 2.0
    target_labels = np.random.randint(0, 3, n_samples)
    
    print(f"\nSource: {source_features.shape}, Target: {target_features.shape}")
    
    # 计算适应前的域差异
    pre_mmd = compute_domain_discrepancy(
        source_features, target_features, metric="mmd"
    )
    print(f"\nPre-adaptation MMD: {pre_mmd:.4f}")
    
    # 创建配置
    config = DomainAdaptationConfig(
        method="mmd",
        input_dim=n_features,
        hidden_dim=128,
        output_dim=32,
        num_epochs=20,
        lambda_adapt=1.0
    )
    
    # 训练MMD适配器
    print("\nTraining MMD Adapter...")
    adapter = MMDAdapter(config, num_classes=3)
    history = adapter.fit(source_features, source_labels, target_features)
    
    # 转换特征
    source_transformed = adapter.transform(source_features)
    target_transformed = adapter.transform(target_features)
    
    # 计算适应后的域差异
    post_mmd = compute_domain_discrepancy(
        source_transformed, target_transformed, metric="mmd"
    )
    print(f"\nPost-adaptation MMD: {post_mmd:.4f}")
    print(f"MMD Reduction: {(pre_mmd - post_mmd) / pre_mmd * 100:.2f}%")
    
    # 测试DANN
    print("\n" + "=" * 60)
    print("Testing DANN Adapter...")
    config.method = "dann"
    dann_adapter = DANNAdapter(config, num_classes=3)
    dann_history = dann_adapter.fit(
        source_features, source_labels, target_features
    )
    
    # 可视化
    print("\nGenerating visualization...")
    fig = visualize_domain_alignment(
        source_transformed,
        target_transformed,
        source_labels,
        target_labels,
        title="MMD Adaptation Result"
    )
    plt.savefig("domain_adaptation_result.png", dpi=150, bbox_inches="tight")
    print("Saved to domain_adaptation_result.png")
    
    print("\n" + "=" * 60)
    print("Domain Adapter Demo Complete!")
    print("=" * 60)
