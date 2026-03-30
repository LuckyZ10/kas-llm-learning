"""
跨领域通用指纹生成 (Universal Fingerprint)
生成适用于多个材料领域的通用描述符指纹

作者: DFT-LAMMPS Team
版本: 1.0.0
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from collections import defaultdict
import warnings


@dataclass
class UniversalFingerprintConfig:
    """通用指纹配置"""
    # 输入特征维度
    composition_dim: int = 100
    structure_dim: int = 64
    property_dim: int = 32
    
    # 指纹维度
    fingerprint_dim: int = 128
    
    # 网络配置
    n_encoder_layers: int = 3
    n_fusion_layers: int = 2
    hidden_dim: int = 256
    
    # 域无关配置
    domain_adaptation: bool = True
    gradient_reversal_lambda: float = 1.0
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class CompositionEncoder(nn.Module):
    """成分编码器"""
    
    def __init__(self, input_dim: int = 100, output_dim: int = 128):
        super().__init__()
        
        self.encoder = nn.Sequential(
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
        return self.encoder(x)


class StructureEncoder(nn.Module):
    """结构编码器"""
    
    def __init__(self, input_dim: int = 64, output_dim: int = 128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.ReLU(),
            nn.BatchNorm1d(192),
            nn.Dropout(0.2),
            
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class PropertyEncoder(nn.Module):
    """性质编码器"""
    
    def __init__(self, input_dim: int = 32, output_dim: int = 64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class FeatureFusion(nn.Module):
    """特征融合模块"""
    
    def __init__(
        self,
        composition_dim: int = 128,
        structure_dim: int = 128,
        property_dim: int = 64,
        output_dim: int = 128
    ):
        super().__init__()
        
        total_dim = composition_dim + structure_dim + property_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, output_dim),
            nn.ReLU()
        )
        
        # 注意力权重
        self.attention = nn.Sequential(
            nn.Linear(total_dim, 3),
            nn.Softmax(dim=1)
        )
    
    def forward(
        self,
        composition_feat: torch.Tensor,
        structure_feat: torch.Tensor,
        property_feat: torch.Tensor
    ) -> torch.Tensor:
        """融合多种特征"""
        # 拼接
        combined = torch.cat([
            composition_feat,
            structure_feat,
            property_feat
        ], dim=-1)
        
        # 计算注意力权重
        attn_weights = self.attention(combined)
        
        # 加权融合
        weighted = (
            attn_weights[:, 0:1] * composition_feat +
            attn_weights[:, 1:2] * structure_feat +
            attn_weights[:, 2:3] * property_feat
        )
        
        # 最终融合
        output = self.fusion(combined)
        
        return output + weighted


class DomainClassifier(nn.Module):
    """域分类器 - 用于域适应"""
    
    def __init__(self, input_dim: int = 128, num_domains: int = 5):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_domains)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class GradientReversalLayer(torch.autograd.Function):
    """梯度反转层"""
    
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


class UniversalFingerprintGenerator:
    """通用指纹生成器"""
    
    def __init__(self, config: UniversalFingerprintConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 编码器
        self.composition_encoder = CompositionEncoder(
            config.composition_dim,
            128
        ).to(self.device)
        
        self.structure_encoder = StructureEncoder(
            config.structure_dim,
            128
        ).to(self.device)
        
        self.property_encoder = PropertyEncoder(
            config.property_dim,
            64
        ).to(self.device)
        
        # 融合
        self.feature_fusion = FeatureFusion(
            composition_dim=128,
            structure_dim=128,
            property_dim=64,
            output_dim=config.fingerprint_dim
        ).to(self.device)
        
        # 域分类器
        self.domain_classifier = DomainClassifier(
            config.fingerprint_dim,
            num_domains=5
        ).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            list(self.composition_encoder.parameters()) +
            list(self.structure_encoder.parameters()) +
            list(self.property_encoder.parameters()) +
            list(self.feature_fusion.parameters()),
            lr=1e-3
        )
        
        self.domain_optimizer = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=1e-3
        )
    
    def generate_fingerprint(
        self,
        composition: np.ndarray,
        structure: np.ndarray,
        properties: np.ndarray
    ) -> np.ndarray:
        """
        生成通用指纹
        
        Args:
            composition: 成分特征
            structure: 结构特征
            properties: 性质特征
        
        Returns:
            指纹向量
        """
        self.composition_encoder.eval()
        self.structure_encoder.eval()
        self.property_encoder.eval()
        self.feature_fusion.eval()
        
        composition_t = torch.FloatTensor(composition).to(self.device)
        structure_t = torch.FloatTensor(structure).to(self.device)
        properties_t = torch.FloatTensor(properties).to(self.device)
        
        with torch.no_grad():
            comp_feat = self.composition_encoder(composition_t)
            struct_feat = self.structure_encoder(structure_t)
            prop_feat = self.property_encoder(properties_t)
            
            fingerprint = self.feature_fusion(comp_feat, struct_feat, prop_feat)
        
        return fingerprint.cpu().numpy()
    
    def train_unsupervised(
        self,
        data_by_domain: Dict[str, List[Dict]],
        num_epochs: int = 100
    ) -> Dict[str, List[float]]:
        """
        无监督训练
        
        使用域适应使指纹域无关
        """
        print("Training universal fingerprint generator...")
        
        history = {"reconstruction_loss": [], "domain_loss": []}
        
        domain_to_idx = {domain: idx for idx, domain in enumerate(data_by_domain.keys())}
        
        for epoch in range(num_epochs):
            epoch_recon_loss = 0.0
            epoch_domain_loss = 0.0
            num_batches = 0
            
            for domain, samples in data_by_domain.items():
                for sample in samples:
                    # 提取特征
                    composition = torch.FloatTensor(sample["composition"]).to(self.device)
                    structure = torch.FloatTensor(sample["structure"]).to(self.device)
                    properties = torch.FloatTensor(sample["properties"]).to(self.device)
                    
                    # 编码
                    comp_feat = self.composition_encoder(composition)
                    struct_feat = self.structure_encoder(structure)
                    prop_feat = self.property_encoder(properties)
                    
                    # 融合
                    fingerprint = self.feature_fusion(comp_feat, struct_feat, prop_feat)
                    
                    # 重构损失 (自编码器)
                    recon_comp = nn.Linear(self.config.fingerprint_dim, len(composition)).to(self.device)(fingerprint)
                    recon_loss = nn.MSELoss()(recon_comp, composition)
                    
                    # 域分类损失
                    if self.config.domain_adaptation:
                        reversed_fp = GradientReversalLayer.apply(
                            fingerprint,
                            self.config.gradient_reversal_lambda
                        )
                        domain_pred = self.domain_classifier(reversed_fp)
                        domain_label = torch.LongTensor([domain_to_idx[domain]]).to(self.device)
                        domain_loss = nn.CrossEntropyLoss()(domain_pred.unsqueeze(0), domain_label)
                    else:
                        domain_loss = torch.tensor(0.0)
                    
                    # 总损失
                    total_loss = recon_loss + 0.1 * domain_loss
                    
                    # 反向传播
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()
                    
                    epoch_recon_loss += recon_loss.item()
                    epoch_domain_loss += domain_loss.item() if isinstance(domain_loss, torch.Tensor) else 0
                    num_batches += 1
            
            history["reconstruction_loss"].append(epoch_recon_loss / num_batches)
            history["domain_loss"].append(epoch_domain_loss / num_batches)
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: Recon Loss = {history['reconstruction_loss'][-1]:.4f}")
        
        return history
    
    def compute_fingerprint_similarity(
        self,
        fp1: np.ndarray,
        fp2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """
        计算指纹相似度
        """
        if metric == "cosine":
            return np.dot(fp1, fp2) / (np.linalg.norm(fp1) * np.linalg.norm(fp2))
        elif metric == "euclidean":
            return -np.linalg.norm(fp1 - fp2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def find_similar_materials(
        self,
        query_fingerprint: np.ndarray,
        material_fingerprints: Dict[str, np.ndarray],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        查找相似材料
        """
        similarities = []
        
        for name, fp in material_fingerprints.items():
            sim = self.compute_fingerprint_similarity(query_fingerprint, fp)
            similarities.append((name, sim))
        
        # 排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def transfer_to_new_domain(
        self,
        new_domain_data: List[Dict],
        num_epochs: int = 50
    ) -> None:
        """
        迁移到新领域
        """
        print(f"Transferring to new domain with {len(new_domain_data)} samples...")
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for sample in new_domain_data:
                composition = torch.FloatTensor(sample["composition"]).to(self.device)
                structure = torch.FloatTensor(sample["structure"]).to(self.device)
                properties = torch.FloatTensor(sample["properties"]).to(self.device)
                
                comp_feat = self.composition_encoder(composition)
                struct_feat = self.structure_encoder(structure)
                prop_feat = self.property_encoder(properties)
                
                fingerprint = self.feature_fusion(comp_feat, struct_feat, prop_feat)
                
                # 重构损失
                recon_comp = nn.Linear(self.config.fingerprint_dim, len(composition)).to(self.device)(fingerprint)
                loss = nn.MSELoss()(recon_comp, composition)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Loss = {epoch_loss / len(new_domain_data):.4f}")


def generate_sample_material(domain: str) -> Dict[str, np.ndarray]:
    """生成示例材料数据"""
    np.random.seed(hash(domain) % 2**32)
    
    return {
        "composition": np.random.rand(100),
        "structure": np.random.rand(64),
        "properties": np.random.rand(32),
        "domain": domain
    }


# 演示代码
if __name__ == "__main__":
    print("=" * 60)
    print("Universal Fingerprint Generator Demo")
    print("=" * 60)
    
    # 生成多领域数据
    domains = ["battery", "catalyst", "semiconductor", "metal", "ceramic"]
    data_by_domain = {}
    
    for domain in domains:
        data_by_domain[domain] = [generate_sample_material(domain) for _ in range(50)]
    
    print(f"\nGenerated data for {len(domains)} domains")
    
    # 创建指纹生成器
    config = UniversalFingerprintConfig()
    generator = UniversalFingerprintGenerator(config)
    
    # 训练
    print("\n" + "=" * 60)
    print("Training Universal Fingerprint Generator")
    print("=" * 60)
    
    history = generator.train_unsupervised(data_by_domain, num_epochs=50)
    
    # 生成指纹
    print("\n" + "=" * 60)
    print("Generating Fingerprints")
    print("=" * 60)
    
    fingerprints = {}
    for domain in domains:
        sample = data_by_domain[domain][0]
        fp = generator.generate_fingerprint(
            sample["composition"],
            sample["structure"],
            sample["properties"]
        )
        fingerprints[domain] = fp
        print(f"{domain}: fingerprint shape = {fp.shape}")
    
    # 计算跨领域相似度
    print("\n" + "=" * 60)
    print("Cross-Domain Similarity Matrix")
    print("=" * 60)
    
    print("\nCosine similarity between domains:")
    print(" " * 15, end="")
    for d in domains:
        print(f"{d[:6]:8}", end="")
    print()
    
    for d1 in domains:
        print(f"{d1:14}", end="")
        for d2 in domains:
            sim = generator.compute_fingerprint_similarity(
                fingerprints[d1],
                fingerprints[d2]
            )
            print(f"{sim:8.3f}", end="")
        print()
    
    # 查找相似材料
    print("\n" + "=" * 60)
    print("Similar Material Search")
    print("=" * 60)
    
    query = fingerprints["battery"]
    similar = generator.find_similar_materials(query, fingerprints, top_k=3)
    
    print("\nMaterials most similar to 'battery':")
    for name, sim in similar:
        print(f"  {name}: {sim:.4f}")
    
    print("\n" + "=" * 60)
    print("Universal Fingerprint Demo Complete!")
    print("=" * 60)
