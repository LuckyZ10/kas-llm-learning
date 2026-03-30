"""
零样本性质预测 (Zero-Shot Prediction)
实现无需目标领域训练数据的性质预测

作者: DFT-LAMMPS Team
版本: 1.0.0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from scipy.spatial.distance import cdist


@dataclass
class ZeroShotConfig:
    """零样本预测配置"""
    feature_dim: int = 128
    embedding_dim: int = 256
    semantic_dim: int = 64
    
    # 相似度度量
    similarity_metric: str = "cosine"  # cosine, euclidean
    temperature: float = 0.1
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MaterialEncoder(nn.Module):
    """材料编码器 - 提取视觉/特征嵌入"""
    
    def __init__(self, input_dim: int = 128, output_dim: int = 256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(x), p=2, dim=1)


class SemanticEncoder(nn.Module):
    """语义编码器 - 提取属性描述嵌入"""
    
    def __init__(self, input_dim: int = 64, output_dim: int = 256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 192),
            nn.ReLU(),
            nn.BatchNorm1d(192),
            nn.Dropout(0.2),
            nn.Linear(192, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(x), p=2, dim=1)


class ZeroShotMaterialPredictor:
    """零样本材料性质预测器"""
    
    def __init__(self, config: ZeroShotConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        self.material_encoder = MaterialEncoder(
            config.feature_dim,
            config.embedding_dim
        ).to(self.device)
        
        self.semantic_encoder = SemanticEncoder(
            config.semantic_dim,
            config.embedding_dim
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            list(self.material_encoder.parameters()) +
            list(self.semantic_encoder.parameters()),
            lr=1e-3
        )
    
    def compute_similarity(
        self,
        material_embeddings: torch.Tensor,
        semantic_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        计算材料与语义描述的相似度
        """
        if self.config.similarity_metric == "cosine":
            # 余弦相似度
            similarity = torch.mm(material_embeddings, semantic_embeddings.t())
            similarity = similarity / self.config.temperature
        else:  # euclidean
            # 负欧氏距离
            similarity = -torch.cdist(
                material_embeddings,
                semantic_embeddings
            ) / self.config.temperature
        
        return similarity
    
    def train_on_seen_classes(
        self,
        material_features: np.ndarray,
        semantic_descriptions: np.ndarray,
        labels: np.ndarray,
        num_epochs: int = 100
    ) -> Dict[str, List[float]]:
        """
        在已知类别上训练
        
        学习材料特征与语义描述的关联
        """
        print("Training on seen classes...")
        
        history = {"loss": [], "accuracy": []}
        
        material_tensor = torch.FloatTensor(material_features).to(self.device)
        semantic_tensor = torch.FloatTensor(semantic_descriptions).to(self.device)
        labels_tensor = torch.LongTensor(labels).to(self.device)
        
        for epoch in range(num_epochs):
            self.material_encoder.train()
            self.semantic_encoder.train()
            
            # 编码
            material_emb = self.material_encoder(material_tensor)
            semantic_emb = self.semantic_encoder(semantic_tensor)
            
            # 计算相似度
            similarity = self.compute_similarity(material_emb, semantic_emb)
            
            # 损失 (多标签分类)
            loss = F.cross_entropy(similarity, labels_tensor)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 计算准确率
            preds = torch.argmax(similarity, dim=1)
            acc = (preds == labels_tensor).float().mean().item()
            
            history["loss"].append(loss.item())
            history["accuracy"].append(acc)
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc:.4f}")
        
        return history
    
    def predict_unseen_classes(
        self,
        material_features: np.ndarray,
        unseen_semantic_descriptions: np.ndarray
    ) -> np.ndarray:
        """
        预测未见过的类别
        
        利用语义描述进行零样本预测
        """
        self.material_encoder.eval()
        self.semantic_encoder.eval()
        
        material_tensor = torch.FloatTensor(material_features).to(self.device)
        semantic_tensor = torch.FloatTensor(unseen_semantic_descriptions).to(self.device)
        
        with torch.no_grad():
            material_emb = self.material_encoder(material_tensor)
            semantic_emb = self.semantic_encoder(semantic_tensor)
            
            similarity = self.compute_similarity(material_emb, semantic_emb)
            
            preds = torch.argmax(similarity, dim=1)
        
        return preds.cpu().numpy()
    
    def predict_properties_zero_shot(
        self,
        material_features: np.ndarray,
        property_descriptions: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        零样本性质预测
        
        通过性质描述预测材料性质
        """
        predictions = {}
        
        for prop_name, prop_description in property_descriptions.items():
            # 将性质描述作为语义输入
            semantic_input = prop_description.reshape(1, -1)
            
            # 预测
            pred = self.predict_unseen_classes(
                material_features,
                semantic_input
            )
            
            predictions[prop_name] = pred
        
        return predictions


class AttributeBasedZeroShot:
    """基于属性的零样本学习"""
    
    def __init__(self, config: ZeroShotConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        self.attribute_predictor = nn.Sequential(
            nn.Linear(config.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, config.semantic_dim),
            nn.Sigmoid()
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.attribute_predictor.parameters(),
            lr=1e-3
        )
    
    def train_attribute_predictor(
        self,
        features: np.ndarray,
        attributes: np.ndarray,
        num_epochs: int = 100
    ) -> Dict[str, List[float]]:
        """
        训练属性预测器
        
        学习从材料特征预测属性
        """
        print("Training attribute predictor...")
        
        history = {"loss": []}
        
        features_tensor = torch.FloatTensor(features).to(self.device)
        attributes_tensor = torch.FloatTensor(attributes).to(self.device)
        
        criterion = nn.BCELoss()
        
        for epoch in range(num_epochs):
            self.attribute_predictor.train()
            
            # 预测属性
            pred_attributes = self.attribute_predictor(features_tensor)
            
            # 损失
            loss = criterion(pred_attributes, attributes_tensor)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            history["loss"].append(loss.item())
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}")
        
        return history
    
    def predict_with_attributes(
        self,
        features: np.ndarray,
        class_attributes: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        使用属性进行预测
        """
        self.attribute_predictor.eval()
        
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            # 预测属性
            pred_attributes = self.attribute_predictor(features_tensor)
            pred_attributes_np = pred_attributes.cpu().numpy()
        
        # 与类别属性比较
        predictions = {}
        
        for class_name, class_attr in class_attributes.items():
            # 计算相似度
            similarity = 1 - cdist(
                pred_attributes_np,
                class_attr.reshape(1, -1),
                metric="cosine"
            ).flatten()
            
            predictions[class_name] = similarity
        
        return predictions


def generate_semantic_descriptions(
    class_names: List[str],
    semantic_dim: int = 64
) -> Dict[str, np.ndarray]:
    """
    生成类别语义描述
    
    在实际应用中，这可以是词嵌入或人工定义的属性
    """
    np.random.seed(42)
    
    descriptions = {}
    
    for name in class_names:
        # 为每个类别生成一个随机但确定的描述
        np.random.seed(hash(name) % 2**32)
        descriptions[name] = np.random.randn(semantic_dim)
        descriptions[name] = descriptions[name] / np.linalg.norm(descriptions[name])
    
    return descriptions


def generate_material_data_with_semantics(
    n_samples: int = 1000,
    seen_classes: List[str] = None,
    unseen_classes: List[str] = None,
    feature_dim: int = 128
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """生成带有语义描述的材料数据"""
    if seen_classes is None:
        seen_classes = ["conductor", "semiconductor", "insulator"]
    if unseen_classes is None:
        unseen_classes = ["superconductor", "topological", "ferroelectric"]
    
    all_classes = seen_classes + unseen_classes
    
    # 生成语义描述
    semantic_descriptions = generate_semantic_descriptions(all_classes)
    
    # 生成训练数据 (仅seen classes)
    train_data = {
        "features": [],
        "labels": [],
        "semantic": []
    }
    
    for i, cls in enumerate(seen_classes):
        n_cls_samples = n_samples // len(seen_classes)
        
        # 生成特征
        features = np.random.randn(n_cls_samples, feature_dim)
        features[:, 0] += i * 0.5  # 添加类别特定偏移
        
        train_data["features"].append(features)
        train_data["labels"].extend([i] * n_cls_samples)
        train_data["semantic"].extend([semantic_descriptions[cls]] * n_cls_samples)
    
    train_data["features"] = np.vstack(train_data["features"])
    train_data["labels"] = np.array(train_data["labels"])
    train_data["semantic"] = np.array(train_data["semantic"])
    
    # 生成测试数据 (unseen classes)
    test_data = {
        "features": [],
        "labels": [],
        "semantic": []
    }
    
    for i, cls in enumerate(unseen_classes):
        n_cls_samples = 100
        
        features = np.random.randn(n_cls_samples, feature_dim)
        features[:, 0] += (i + len(seen_classes)) * 0.5
        
        test_data["features"].append(features)
        test_data["labels"].extend([i] * n_cls_samples)
        test_data["semantic"].extend([semantic_descriptions[cls]] * n_cls_samples)
    
    test_data["features"] = np.vstack(test_data["features"])
    test_data["labels"] = np.array(test_data["labels"])
    test_data["semantic"] = np.array(test_data["semantic"])
    
    return train_data, test_data, semantic_descriptions


# 演示代码
if __name__ == "__main__":
    print("=" * 70)
    print("零样本性质预测演示 (Zero-Shot Prediction Demo)")
    print("=" * 70)
    
    # 定义类别
    seen_classes = ["conductor", "semiconductor", "insulator"]
    unseen_classes = ["superconductor", "topological", "ferroelectric"]
    
    print(f"\nSeen classes: {seen_classes}")
    print(f"Unseen classes: {unseen_classes}")
    
    # 生成数据
    train_data, test_data, semantic_desc = generate_material_data_with_semantics(
        n_samples=900,
        seen_classes=seen_classes,
        unseen_classes=unseen_classes
    )
    
    print(f"\nTraining samples: {len(train_data['features'])}")
    print(f"Test samples: {len(test_data['features'])}")
    
    # 配置
    config = ZeroShotConfig()
    
    # 创建零样本预测器
    predictor = ZeroShotMaterialPredictor(config)
    
    # 训练
    print("\n" + "=" * 70)
    print("Training Zero-Shot Model")
    print("=" * 70)
    
    history = predictor.train_on_seen_classes(
        train_data["features"],
        train_data["semantic"],
        train_data["labels"],
        num_epochs=50
    )
    
    # 零样本预测
    print("\n" + "=" * 70)
    print("Zero-Shot Prediction on Unseen Classes")
    print("=" * 70)
    
    # 获取未见类别的语义描述
    unseen_semantic = np.array([semantic_desc[cls] for cls in unseen_classes])
    
    predictions = predictor.predict_unseen_classes(
        test_data["features"],
        unseen_semantic
    )
    
    # 评估
    accuracy = (predictions == test_data["labels"]).mean()
    print(f"\nZero-shot accuracy: {accuracy:.4f}")
    
    # 打印各类别准确率
    for i, cls in enumerate(unseen_classes):
        mask = test_data["labels"] == i
        cls_acc = (predictions[mask] == i).mean()
        print(f"  {cls}: {cls_acc:.4f}")
    
    # 属性基零样本学习
    print("\n" + "=" * 70)
    print("Attribute-Based Zero-Shot Learning")
    print("=" * 70)
    
    attr_predictor = AttributeBasedZeroShot(config)
    
    # 生成属性 (简化：随机属性)
    n_attributes = 10
    train_attributes = np.random.rand(len(train_data["features"]), n_attributes)
    
    # 训练
    attr_history = attr_predictor.train_attribute_predictor(
        train_data["features"],
        train_attributes,
        num_epochs=50
    )
    
    # 定义类别属性
    class_attributes = {
        cls: np.random.rand(n_attributes)
        for cls in unseen_classes
    }
    
    # 预测
    attr_predictions = attr_predictor.predict_with_attributes(
        test_data["features"][:50],
        class_attributes
    )
    
    print("\nAttribute-based predictions (sample similarities):")
    for cls in list(class_attributes.keys())[:2]:
        print(f"  {cls}: mean similarity = {np.mean(attr_predictions[cls]):.4f}")
    
    print("\n" + "=" * 70)
    print("Zero-Shot Prediction Demo Complete!")
    print("=" * 70)
