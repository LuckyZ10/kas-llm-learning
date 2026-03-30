"""
GNN材料表示学习应用案例
展示零样本预测、跨域迁移学习和大规模筛选加速
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch, Dataset
from torch_geometric.loader import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import json
from pathlib import Path
import warnings
from tqdm import tqdm


# 导入我们创建的GNN模块
import sys
sys.path.append('/root/.openclaw/workspace/dftlammps')

from gnn_advanced.pretrained_models import (
    PretrainedGNN, PretrainConfig, PretrainingTrainer,
    PropertyPredictionHead, FewShotLearner
)
from gnn_advanced.equivariant_gnn import (
    EquivariantGNN, EquivariantConfig
)
from gnn_advanced.hierarchical_gnn import (
    HierarchicalGNN, HierarchicalConfig
)
from foundation_models.material_foundation import (
    MaterialFoundationModel, FoundationModelConfig,
    MultiTaskTrainer, ZeroShotPredictor, PrototypicalNetwork
)


@dataclass
class ApplicationConfig:
    """应用配置"""
    
    # 数据设置
    batch_size: int = 32
    num_workers: int = 4
    
    # 训练设置
    learning_rate: float = 1e-4
    num_epochs: int = 100
    
    # 零样本设置
    similarity_threshold: float = 0.7
    
    # 跨域迁移
    source_domain: str = 'bulk_crystals'
    target_domain: str = 'surfaces'
    
    # 大规模筛选
    screening_batch_size: int = 1000
    top_k: int = 100


# ============== 零样本材料性质预测 ==============

class ZeroShotMaterialDiscovery:
    """
    零样本材料发现
    利用预训练模型的泛化能力预测未见过材料的性质
    """
    
    def __init__(self, foundation_model: MaterialFoundationModel,
                 config: ApplicationConfig = None):
        self.model = foundation_model
        self.config = config or ApplicationConfig()
        self.model.eval()
        
        # 性质数据库
        self.property_database = {}
        
        # 相似性模型
        self.similarity_encoder = None
    
    def build_property_database(self, materials: List[Data],
                                properties: Dict[str, List[float]]):
        """
        构建性质数据库
        
        Args:
            materials: 已知材料列表
            properties: 属性字典 {属性名: [值列表]}
        """
        print("构建性质数据库...")
        
        with torch.no_grad():
            # 计算所有材料的嵌入
            embeddings = []
            for material in tqdm(materials, desc="Encoding materials"):
                emb = self.model.get_embeddings(material, pooling='contrastive')
                embeddings.append(emb)
            
            embeddings = torch.cat(embeddings, dim=0)
        
        # 存储
        self.property_database = {
            'embeddings': embeddings,
            'materials': materials,
            'properties': properties
        }
        
        print(f"数据库包含 {len(materials)} 种材料")
    
    def predict_zero_shot(self, new_material: Data, 
                          property_name: str,
                          k: int = 5) -> Tuple[float, float, List[int]]:
        """
        零样本预测新材料性质
        
        Args:
            new_material: 新材料
            property_name: 要预测的属性名
            k: 最近邻数量
        
        Returns:
            (预测值, 置信度, 相似材料索引)
        """
        if property_name not in self.property_database.get('properties', {}):
            raise ValueError(f"未知属性: {property_name}")
        
        with torch.no_grad():
            # 编码新材料
            new_emb = self.model.get_embeddings(new_material, pooling='contrastive')
            
            # 计算与数据库中材料的相似度
            similarities = F.cosine_similarity(
                new_emb, 
                self.property_database['embeddings']
            )
            
            # 获取k近邻
            top_k_values, top_k_indices = torch.topk(similarities, k)
            
            # 加权预测
            weights = F.softmax(top_k_values * 10, dim=0)
            neighbor_properties = torch.tensor([
                self.property_database['properties'][property_name][i]
                for i in top_k_indices.tolist()
            ], device=new_emb.device)
            
            prediction = (weights * neighbor_properties).sum().item()
            
            # 置信度基于相似度
            confidence = top_k_values.mean().item()
        
        return prediction, confidence, top_k_indices.tolist()
    
    def screen_materials(self, candidates: List[Data],
                        target_property: str,
                        target_range: Tuple[float, float],
                        batch_size: int = None) -> List[Tuple[int, float, float]]:
        """
        大规模材料筛选
        
        Args:
            candidates: 候选材料列表
            target_property: 目标属性
            target_range: 目标范围 (min, max)
            batch_size: 批处理大小
        
        Returns:
            筛选结果列表 [(索引, 预测值, 置信度), ...]
        """
        batch_size = batch_size or self.config.screening_batch_size
        
        results = []
        
        print(f"筛选 {len(candidates)} 种候选材料...")
        
        for i in tqdm(range(0, len(candidates), batch_size), desc="Screening"):
            batch = candidates[i:i + batch_size]
            
            for j, material in enumerate(batch):
                try:
                    pred, conf, _ = self.predict_zero_shot(
                        material, target_property, k=5
                    )
                    
                    # 检查是否在目标范围内
                    if target_range[0] <= pred <= target_range[1]:
                        results.append((i + j, pred, conf))
                except Exception as e:
                    continue
        
        # 按置信度排序
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results[:self.config.top_k]
    
    def predict_multiple_properties(self, material: Data,
                                     property_names: List[str]) -> Dict[str, Tuple[float, float]]:
        """
        同时预测多种属性
        
        Returns:
            {属性名: (预测值, 置信度)}
        """
        predictions = {}
        
        for prop_name in property_names:
            try:
                pred, conf, _ = self.predict_zero_shot(material, prop_name)
                predictions[prop_name] = (pred, conf)
            except Exception as e:
                predictions[prop_name] = (None, 0.0)
        
        return predictions


# ============== 跨域迁移学习 ==============

class CrossDomainTransfer:
    """
    跨域迁移学习
    将在源域训练的模型迁移到目标域
    """
    
    def __init__(self, source_model: nn.Module, config: ApplicationConfig = None):
        self.source_model = source_model
        self.config = config or ApplicationConfig()
        self.target_model = None
    
    def analyze_domain_gap(self, source_data: List[Data],
                          target_data: List[Data]) -> Dict[str, float]:
        """
        分析源域和目标域之间的差异
        
        Returns:
            域差异指标
        """
        print("分析域差异...")
        
        with torch.no_grad():
            # 获取源域嵌入
            source_embs = []
            for data in source_data[:100]:  # 采样
                emb = self.source_model.get_embeddings(data)
                source_embs.append(emb)
            source_embs = torch.cat(source_embs, dim=0)
            
            # 获取目标域嵌入
            target_embs = []
            for data in target_data[:100]:
                emb = self.source_model.get_embeddings(data)
                target_embs.append(emb)
            target_embs = torch.cat(target_embs, dim=0)
        
        # 计算域差异（MMD - Maximum Mean Discrepancy）
        mmd = self._compute_mmd(source_embs, target_embs)
        
        # 计算分布差异
        source_mean = source_embs.mean(dim=0)
        target_mean = target_embs.mean(dim=0)
        mean_diff = torch.norm(source_mean - target_mean).item()
        
        source_std = source_embs.std(dim=0).mean().item()
        target_std = target_embs.std(dim=0).mean().item()
        
        return {
            'mmd': mmd,
            'mean_difference': mean_diff,
            'source_std': source_std,
            'target_std': target_std,
            'std_ratio': target_std / (source_std + 1e-8)
        }
    
    def _compute_mmd(self, x: torch.Tensor, y: torch.Tensor, 
                     kernel: str = 'rbf') -> float:
        """计算最大均值差异"""
        n = x.shape[0]
        m = y.shape[0]
        
        if kernel == 'rbf':
            # RBF核
            xx = torch.exp(-torch.cdist(x, x) ** 2 / (2 * x.shape[1]))
            yy = torch.exp(-torch.cdist(y, y) ** 2 / (2 * y.shape[1]))
            xy = torch.exp(-torch.cdist(x, y) ** 2 / (2 * x.shape[1]))
        else:
            # 线性核
            xx = torch.matmul(x, x.T)
            yy = torch.matmul(y, y.T)
            xy = torch.matmul(x, y.T)
        
        mmd = xx.sum() / (n * n) + yy.sum() / (m * m) - 2 * xy.sum() / (n * m)
        
        return mmd.item()
    
    def transfer_with_adaptation(self, target_data: List[Data],
                                  target_labels: torch.Tensor,
                                  adaptation_strategy: str = 'fine_tune',
                                  num_epochs: int = 50) -> nn.Module:
        """
        执行域适应
        
        Args:
            target_data: 目标域数据
            target_labels: 目标域标签
            adaptation_strategy: 'fine_tune', 'feature_extraction', 'adapter'
            num_epochs: 训练轮数
        
        Returns:
            适应后的模型
        """
        print(f"执行域适应: {adaptation_strategy}")
        
        # 克隆源模型
        import copy
        self.target_model = copy.deepcopy(self.source_model)
        
        if adaptation_strategy == 'fine_tune':
            # 全模型微调
            for param in self.target_model.parameters():
                param.requires_grad = True
            
            optimizer = torch.optim.AdamW(
                self.target_model.parameters(),
                lr=self.config.learning_rate / 10  # 较小学习率
            )
        
        elif adaptation_strategy == 'feature_extraction':
            # 只训练输出层
            for param in self.target_model.parameters():
                param.requires_grad = False
            
            # 解冻输出头
            for param in self.target_model.energy_head.parameters():
                param.requires_grad = True
            
            optimizer = torch.optim.AdamW(
                self.target_model.energy_head.parameters(),
                lr=self.config.learning_rate
            )
        
        elif adaptation_strategy == 'adapter':
            # 添加适配器层
            self._add_adapters()
            
            # 只训练适配器
            optimizer = torch.optim.AdamW(
                self._get_adapter_params(),
                lr=self.config.learning_rate
            )
        
        # 训练
        self.target_model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            for data, label in zip(target_data, target_labels):
                optimizer.zero_grad()
                
                output = self.target_model(data)
                loss = F.mse_loss(output['energy'], label.unsqueeze(0))
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(target_data):.4f}")
        
        return self.target_model
    
    def _add_adapters(self):
        """添加适配器层（类似BERT的adapter）"""
        # 在每个Transformer层后添加轻量级适配器
        for layer in self.target_model.layers:
            layer.adapter = nn.Sequential(
                nn.Linear(layer.self_attn.d_model, 64),
                nn.ReLU(),
                nn.Linear(64, layer.self_attn.d_model)
            )
    
    def _get_adapter_params(self):
        """获取适配器参数"""
        params = []
        for layer in self.target_model.layers:
            if hasattr(layer, 'adapter'):
                params.extend(layer.adapter.parameters())
        return params
    
    def evaluate_transfer(self, test_data: List[Data],
                         test_labels: torch.Tensor) -> Dict[str, float]:
        """评估迁移效果"""
        if self.target_model is None:
            raise ValueError("请先执行域适应")
        
        self.target_model.eval()
        
        predictions = []
        with torch.no_grad():
            for data in test_data:
                output = self.target_model(data)
                predictions.append(output['energy'].item())
        
        predictions = torch.tensor(predictions)
        
        mae = F.l1_loss(predictions, test_labels).item()
        mse = F.mse_loss(predictions, test_labels).item()
        rmse = np.sqrt(mse)
        
        # R²分数
        ss_res = ((predictions - test_labels) ** 2).sum().item()
        ss_tot = ((test_labels - test_labels.mean()) ** 2).sum().item()
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }


# ============== 大规模筛选加速 ==============

class LargeScaleScreening:
    """
    大规模材料筛选加速
    利用GNN快速筛选大规模候选材料
    """
    
    def __init__(self, model: nn.Module, config: ApplicationConfig = None):
        self.model = model
        self.config = config or ApplicationConfig()
        self.model.eval()
        
        # 筛选历史
        self.screening_history = []
    
    def create_candidate_space(self, composition_space: Dict[str, List],
                               structure_generator: Callable) -> List[Data]:
        """
        创建候选材料空间
        
        Args:
            composition_space: 成分空间 {元素: [可能的比例]}
            structure_generator: 结构生成函数
        
        Returns:
            候选材料列表
        """
        print("生成候选材料空间...")
        
        candidates = []
        
        # 生成组合
        from itertools import product
        
        elements = list(composition_space.keys())
        compositions = list(product(*composition_space.values()))
        
        print(f"探索 {len(compositions)} 种组合...")
        
        for comp in compositions:
            composition = dict(zip(elements, comp))
            
            try:
                # 生成结构
                structure = structure_generator(composition)
                candidates.append(structure)
            except Exception as e:
                continue
        
        print(f"生成 {len(candidates)} 个候选结构")
        
        return candidates
    
    def parallel_screening(self, candidates: List[Data],
                          target_property: str,
                          target_range: Tuple[float, float],
                          num_gpus: int = 1) -> List[Tuple[int, float]]:
        """
        并行筛选
        
        Args:
            candidates: 候选材料
            target_property: 目标属性
            target_range: 目标范围
            num_gpus: GPU数量
        
        Returns:
            筛选结果
        """
        print(f"并行筛选 {len(candidates)} 个候选...")
        
        results = []
        
        # 分批处理
        batch_size = self.config.screening_batch_size
        
        with torch.no_grad():
            for i in tqdm(range(0, len(candidates), batch_size)):
                batch = candidates[i:i + batch_size]
                
                # 批量预测
                predictions = []
                for data in batch:
                    try:
                        output = self.model(data)
                        
                        # 获取目标属性
                        if target_property == 'energy':
                            pred = output['energy'].item()
                        elif target_property in output:
                            pred = output[target_property].item()
                        else:
                            continue
                        
                        predictions.append(pred)
                    except Exception as e:
                        continue
                
                # 筛选
                for j, pred in enumerate(predictions):
                    if target_range[0] <= pred <= target_range[1]:
                        results.append((i + j, pred))
        
        # 按预测值排序
        results.sort(key=lambda x: abs(x[1] - sum(target_range) / 2))
        
        return results[:self.config.top_k]
    
    def active_learning_screening(self, candidates: List[Data],
                                   oracle: Callable,
                                   num_iterations: int = 10,
                                   samples_per_iter: int = 10) -> List[Tuple[Data, float]]:
        """
        主动学习筛选
        
        Args:
            candidates: 候选材料
            oracle: 真实标签获取函数（如DFT计算）
            num_iterations: 迭代次数
            samples_per_iter: 每轮选择的样本数
        
        Returns:
            标记样本列表 [(data, label), ...]
        """
        print("开始主动学习筛选...")
        
        labeled_data = []
        remaining_indices = list(range(len(candidates)))
        
        # 初始随机选择
        initial_samples = np.random.choice(
            remaining_indices, 
            min(samples_per_iter, len(remaining_indices)),
            replace=False
        )
        
        for idx in initial_samples:
            label = oracle(candidates[idx])
            labeled_data.append((candidates[idx], label))
            remaining_indices.remove(idx)
        
        # 迭代主动学习
        for iteration in range(num_iterations):
            print(f"\n迭代 {iteration + 1}/{num_iterations}")
            
            # 在标记数据上微调模型
            self._fine_tune_on_labeled(labeled_data)
            
            # 计算剩余候选的不确定性
            uncertainties = []
            for idx in remaining_indices[:100]:  # 采样计算
                uncertainty = self._compute_uncertainty(candidates[idx])
                uncertainties.append((idx, uncertainty))
            
            # 选择最不确定的样本
            uncertainties.sort(key=lambda x: x[1], reverse=True)
            selected = [idx for idx, _ in uncertainties[:samples_per_iter]]
            
            # 查询oracle
            for idx in selected:
                if idx in remaining_indices:
                    label = oracle(candidates[idx])
                    labeled_data.append((candidates[idx], label))
                    remaining_indices.remove(idx)
            
            print(f"  已标记 {len(labeled_data)} 个样本")
        
        return labeled_data
    
    def _fine_tune_on_labeled(self, labeled_data: List[Tuple[Data, float]]):
        """在标记数据上快速微调"""
        self.model.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        
        for _ in range(5):  # 快速微调
            total_loss = 0
            for data, label in labeled_data:
                optimizer.zero_grad()
                
                output = self.model(data)
                loss = F.mse_loss(output['energy'], torch.tensor([label]))
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
    
    def _compute_uncertainty(self, data: Data) -> float:
        """计算预测不确定性（使用dropout）"""
        self.model.train()  # 启用dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(10):  # MC dropout
                output = self.model(data)
                predictions.append(output['energy'].item())
        
        # 不确定性 = 预测方差
        uncertainty = np.var(predictions)
        
        self.model.eval()
        
        return uncertainty
    
    def generate_screening_report(self, results: List[Tuple[int, float]],
                                   candidates: List[Data]) -> Dict:
        """生成筛选报告"""
        report = {
            'total_candidates': len(candidates),
            'selected_count': len(results),
            'selection_rate': len(results) / len(candidates),
            'predictions': {
                'mean': np.mean([r[1] for r in results]),
                'std': np.std([r[1] for r in results]),
                'min': min(r[1] for r in results),
                'max': max(r[1] for r in results)
            },
            'top_candidates': []
        }
        
        # 详细分析前10个
        for idx, pred in results[:10]:
            data = candidates[idx]
            unique_elements = torch.unique(data.atomic_numbers).tolist()
            
            report['top_candidates'].append({
                'index': idx,
                'prediction': pred,
                'elements': unique_elements,
                'num_atoms': data.atomic_numbers.shape[0]
            })
        
        return report


# ============== 应用示例 ==============

def example_zero_shot_prediction():
    """零样本预测示例"""
    
    print("=" * 70)
    print("零样本材料性质预测示例")
    print("=" * 70)
    
    # 创建基础模型
    config = FoundationModelConfig(hidden_dim=128, num_layers=4)
    model = MaterialFoundationModel(config)
    
    # 创建零样本预测器
    predictor = ZeroShotMaterialDiscovery(model)
    
    # 生成模拟数据库
    print("\n1. 构建性质数据库...")
    database_size = 100
    database_materials = []
    database_properties = {
        'bandgap': [],
        'bulk_modulus': [],
        'formation_energy': []
    }
    
    for i in range(database_size):
        num_atoms = np.random.randint(10, 30)
        data = Data(
            atomic_numbers=torch.randint(1, 50, (num_atoms,)),
            pos=torch.randn(num_atoms, 3) * 8,
            batch=torch.zeros(num_atoms, dtype=torch.long)
        )
        database_materials.append(data)
        
        # 模拟属性值
        database_properties['bandgap'].append(np.random.uniform(0, 5))
        database_properties['bulk_modulus'].append(np.random.uniform(50, 200))
        database_properties['formation_energy'].append(np.random.uniform(-5, 0))
    
    predictor.build_property_database(database_materials, database_properties)
    
    # 测试零样本预测
    print("\n2. 零样本预测新材料...")
    test_material = Data(
        atomic_numbers=torch.randint(1, 50, (20,)),
        pos=torch.randn(20, 3) * 8,
        batch=torch.zeros(20, dtype=torch.long)
    )
    
    for prop in ['bandgap', 'bulk_modulus', 'formation_energy']:
        pred, conf, neighbors = predictor.predict_zero_shot(
            test_material, prop, k=5
        )
        print(f"  {prop}: {pred:.3f} (置信度: {conf:.3f})")
    
    print("\n3. 多属性同时预测...")
    multi_pred = predictor.predict_multiple_properties(
        test_material,
        ['bandgap', 'bulk_modulus', 'formation_energy']
    )
    for prop, (val, conf) in multi_pred.items():
        print(f"  {prop}: {val:.3f} (置信度: {conf:.3f})")
    
    print("\n" + "=" * 70)
    print("零样本预测示例完成！")
    print("=" * 70)
    
    return predictor


def example_cross_domain_transfer():
    """跨域迁移学习示例"""
    
    print("\n" + "=" * 70)
    print("跨域迁移学习示例")
    print("=" * 70)
    
    # 创建源域模型（假设在体材料上预训练）
    config = FoundationModelConfig(hidden_dim=128, num_layers=4)
    source_model = MaterialFoundationModel(config)
    
    # 创建迁移学习器
    transfer = CrossDomainTransfer(source_model)
    
    # 生成模拟数据
    print("\n1. 生成源域和目标域数据...")
    
    # 源域：体材料
    source_data = []
    for _ in range(50):
        num_atoms = np.random.randint(10, 30)
        data = Data(
            atomic_numbers=torch.randint(1, 50, (num_atoms,)),
            pos=torch.randn(num_atoms, 3) * 8,
            batch=torch.zeros(num_atoms, dtype=torch.long)
        )
        source_data.append(data)
    
    # 目标域：表面
    target_data = []
    for _ in range(20):
        num_atoms = np.random.randint(15, 40)  # 表面通常原子更多
        data = Data(
            atomic_numbers=torch.randint(1, 50, (num_atoms,)),
            pos=torch.randn(num_atoms, 3) * 10,
            batch=torch.zeros(num_atoms, dtype=torch.long)
        )
        target_data.append(data)
    
    # 分析域差异
    print("\n2. 分析域差异...")
    gap_metrics = transfer.analyze_domain_gap(source_data, target_data)
    print(f"  MMD: {gap_metrics['mmd']:.4f}")
    print(f"  均值差异: {gap_metrics['mean_difference']:.4f}")
    print(f"  源域标准差: {gap_metrics['source_std']:.4f}")
    print(f"  目标域标准差: {gap_metrics['target_std']:.4f}")
    
    # 执行域适应
    print("\n3. 执行域适应...")
    target_labels = torch.randn(20)  # 模拟标签
    adapted_model = transfer.transfer_with_adaptation(
        target_data[:15],
        target_labels[:15],
        adaptation_strategy='feature_extraction',
        num_epochs=20
    )
    
    # 评估
    print("\n4. 评估迁移效果...")
    metrics = transfer.evaluate_transfer(target_data[15:], target_labels[15:])
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")
    
    print("\n" + "=" * 70)
    print("跨域迁移学习示例完成！")
    print("=" * 70)
    
    return transfer


def example_large_scale_screening():
    """大规模筛选加速示例"""
    
    print("\n" + "=" * 70)
    print("大规模材料筛选加速示例")
    print("=" * 70)
    
    # 创建模型
    config = FoundationModelConfig(hidden_dim=128, num_layers=4)
    model = MaterialFoundationModel(config)
    
    # 创建筛选器
    screener = LargeScaleScreening(model)
    
    # 生成候选材料
    print("\n1. 生成候选材料空间...")
    candidates = []
    for i in range(500):  # 500个候选
        num_atoms = np.random.randint(10, 40)
        data = Data(
            atomic_numbers=torch.randint(1, 80, (num_atoms,)),
            pos=torch.randn(num_atoms, 3) * 10,
            batch=torch.zeros(num_atoms, dtype=torch.long)
        )
        candidates.append(data)
    
    print(f"  生成 {len(candidates)} 个候选材料")
    
    # 并行筛选
    print("\n2. 并行筛选...")
    results = screener.parallel_screening(
        candidates,
        target_property='energy',
        target_range=(-5, -2),
        num_gpus=1
    )
    
    print(f"  筛选出 {len(results)} 个候选")
    print(f"  前5个预测值: {[r[1] for r in results[:5]]}")
    
    # 生成报告
    print("\n3. 生成筛选报告...")
    report = screener.generate_screening_report(results, candidates)
    print(f"  总候选数: {report['total_candidates']}")
    print(f"  选择率: {report['selection_rate']:.2%}")
    print(f"  预测值范围: [{report['predictions']['min']:.3f}, {report['predictions']['max']:.3f}]")
    
    # 主动学习示例（模拟）
    print("\n4. 主动学习筛选（模拟）...")
    
    def mock_oracle(data):
        # 模拟昂贵的DFT计算
        return np.random.uniform(-6, -1)
    
    labeled = screener.active_learning_screening(
        candidates[:100],
        mock_oracle,
        num_iterations=3,
        samples_per_iter=5
    )
    
    print(f"  主动学习标记了 {len(labeled)} 个样本")
    
    print("\n" + "=" * 70)
    print("大规模筛选加速示例完成！")
    print("=" * 70)
    
    return screener


# ============== 主入口 ==============

if __name__ == "__main__":
    print("=" * 70)
    print("GNN材料表示学习应用案例")
    print("=" * 70)
    
    # 运行所有示例
    example_zero_shot_prediction()
    example_cross_domain_transfer()
    example_large_scale_screening()
    
    print("\n" + "=" * 70)
    print("所有示例完成！")
    print("=" * 70)
