"""
高熵材料通用发现框架 (High Entropy Materials Transfer)
实现适用于多种高熵材料体系的通用发现框架

作者: DFT-LAMMPS Team
版本: 1.0.0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class HighEntropyConfig:
    """高熵材料配置"""
    composition_dim: int = 20  # 最大元素数
    structure_dim: int = 64
    property_dim: int = 32
    
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    
    # 熵正则化
    entropy_weight: float = 0.1
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class CompositionEncoder(nn.Module):
    """成分编码器"""
    
    def __init__(self, input_dim: int = 20, output_dim: int = 64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.BatchNorm1d(96),
            nn.Dropout(0.2),
            
            nn.Linear(96, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class StructureEncoder(nn.Module):
    """结构编码器"""
    
    def __init__(self, input_dim: int = 64, output_dim: int = 64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.BatchNorm1d(96),
            nn.Dropout(0.2),
            
            nn.Linear(96, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class PhasePredictor(nn.Module):
    """相稳定性预测器"""
    
    def __init__(self, input_dim: int = 128, num_phases: int = 5):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_phases)  # 多类别分类
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(x)


class PropertyPredictor(nn.Module):
    """性质预测器"""
    
    def __init__(self, input_dim: int = 128, num_properties: int = 3):
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


class EntropyCalculator:
    """混合熵计算器"""
    
    @staticmethod
    def configurational_entropy(composition: torch.Tensor) -> torch.Tensor:
        """
        计算构型熵
        
        S = -R * Σ(xi * ln(xi))
        """
        # 避免log(0)
        composition = torch.clamp(composition, min=1e-10)
        
        # 归一化
        composition = composition / composition.sum(dim=-1, keepdim=True)
        
        # 计算熵
        entropy = -torch.sum(composition * torch.log(composition), dim=-1)
        
        return entropy
    
    @staticmethod
    def mixing_enthalpy_estimate(composition: torch.Tensor) -> torch.Tensor:
        """
        估计混合焓
        
        简化模型，实际应该使用元素对相互作用参数
        """
        # 基于成分差异的简化估计
        mean_comp = composition.mean(dim=-1, keepdim=True)
        variance = torch.sum((composition - mean_comp) ** 2, dim=-1)
        
        # 差异越大，混合焓越高
        return variance * 10  # 缩放因子
    
    @staticmethod
    def omega_parameter(
        entropy: torch.Tensor,
        enthalpy: torch.Tensor
    ) -> torch.Tensor:
        """
        计算Ω参数
        
        Ω = Tm * ΔS_mix / |ΔH_mix|
        
        Ω > 1.1 通常表示高熵相稳定
        """
        # 假设熔点约1500K
        Tm = 1500
        
        omega = Tm * entropy / (torch.abs(enthalpy) + 1e-10)
        
        return omega


class HighEntropyMaterialFramework:
    """高熵材料通用发现框架"""
    
    def __init__(self, config: HighEntropyConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 编码器
        self.composition_encoder = CompositionEncoder(
            config.composition_dim,
            64
        ).to(self.device)
        
        self.structure_encoder = StructureEncoder(
            config.structure_dim,
            64
        ).to(self.device)
        
        # 预测器
        self.phase_predictor = PhasePredictor(128, 5).to(self.device)
        self.property_predictor = PropertyPredictor(128, 3).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            list(self.composition_encoder.parameters()) +
            list(self.structure_encoder.parameters()) +
            list(self.phase_predictor.parameters()) +
            list(self.property_predictor.parameters()),
            lr=config.learning_rate
        )
        
        self.entropy_calc = EntropyCalculator()
    
    def encode_material(
        self,
        composition: torch.Tensor,
        structure: torch.Tensor
    ) -> torch.Tensor:
        """
        编码材料表示
        
        结合成分和结构信息
        """
        comp_embedding = self.composition_encoder(composition)
        struct_embedding = self.structure_encoder(structure)
        
        # 拼接
        combined = torch.cat([comp_embedding, struct_embedding], dim=-1)
        
        return combined
    
    def predict_phase_stability(
        self,
        composition: torch.Tensor,
        structure: torch.Tensor
    ) -> torch.Tensor:
        """预测相稳定性"""
        embedding = self.encode_material(composition, structure)
        logits = self.phase_predictor(embedding)
        return logits
    
    def predict_properties(
        self,
        composition: torch.Tensor,
        structure: torch.Tensor
    ) -> torch.Tensor:
        """预测材料性质"""
        embedding = self.encode_material(composition, structure)
        properties = self.property_predictor(embedding)
        return properties
    
    def compute_entropy_regularization(
        self,
        composition: torch.Tensor
    ) -> torch.Tensor:
        """计算熵正则化损失"""
        # 构型熵
        config_entropy = self.entropy_calc.configurational_entropy(composition)
        
        # 鼓励高熵 (最大化熵)
        entropy_loss = -config_entropy.mean()
        
        return entropy_loss
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """训练步骤"""
        self.composition_encoder.train()
        self.structure_encoder.train()
        self.phase_predictor.train()
        self.property_predictor.train()
        
        # 提取数据
        composition = batch["composition"].to(self.device)
        structure = batch["structure"].to(self.device)
        phase_labels = batch["phase"].to(self.device)
        properties = batch["properties"].to(self.device)
        
        # 前向传播
        embedding = self.encode_material(composition, structure)
        
        phase_logits = self.phase_predictor(embedding)
        pred_properties = self.property_predictor(embedding)
        
        # 损失
        phase_loss = F.cross_entropy(phase_logits, phase_labels)
        property_loss = F.mse_loss(pred_properties, properties)
        
        # 熵正则化
        entropy_reg = self.compute_entropy_regularization(composition)
        
        total_loss = (
            phase_loss +
            property_loss +
            self.config.entropy_weight * entropy_reg
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "phase_loss": phase_loss.item(),
            "property_loss": property_loss.item(),
            "entropy_reg": entropy_reg.item()
        }
    
    def design_high_entropy_alloy(
        self,
        target_properties: Dict[str, float],
        candidate_elements: List[str],
        num_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        设计高熵合金
        
        基于目标性质优化成分
        """
        # 初始化随机成分
        composition = torch.rand(1, self.config.composition_dim).to(self.device)
        composition = composition / composition.sum()
        
        # 结构特征 (假设)
        structure = torch.randn(1, self.config.structure_dim).to(self.device)
        
        # 优化成分
        composition.requires_grad = True
        optimizer = torch.optim.Adam([composition], lr=0.01)
        
        history = []
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # 归一化
            comp_normalized = F.softmax(composition, dim=-1)
            
            # 预测
            pred_props = self.predict_properties(comp_normalized, structure)
            
            # 计算与目标的差距
            target_tensor = torch.tensor([
                target_properties.get("hardness", 5.0),
                target_properties.get("corrosion_resistance", 0.5),
                target_properties.get("thermal_stability", 1000.0)
            ]).to(self.device)
            
            loss = F.mse_loss(pred_props.squeeze(), target_tensor)
            
            # 添加高熵约束
            entropy = self.entropy_calc.configurational_entropy(comp_normalized)
            entropy_penalty = torch.relu(1.0 - entropy)  # 鼓励熵 > 1.0
            
            total_loss = loss + 0.1 * entropy_penalty
            
            total_loss.backward()
            optimizer.step()
            
            if iteration % 20 == 0:
                history.append({
                    "iteration": iteration,
                    "loss": loss.item(),
                    "entropy": entropy.item()
                })
        
        # 最终结果
        final_composition = F.softmax(composition, dim=-1).detach().cpu().numpy()
        
        return {
            "composition": final_composition.squeeze(),
            "entropy": entropy.item(),
            "target_properties": target_properties,
            "optimization_history": history
        }
    
    def screen_compositions(
        self,
        compositions: np.ndarray,
        structures: np.ndarray,
        criteria: Dict[str, Tuple[float, float]]
    ) -> List[Dict[str, Any]]:
        """
        筛选高熵材料候选
        
        Args:
            compositions: 成分数组 [N, composition_dim]
            structures: 结构数组 [N, structure_dim]
            criteria: 筛选条件，如 {"entropy": (1.0, 5.0)}
        
        Returns:
            符合条件的候选列表
        """
        self.composition_encoder.eval()
        self.structure_encoder.eval()
        self.property_predictor.eval()
        
        compositions_tensor = torch.FloatTensor(compositions).to(self.device)
        structures_tensor = torch.FloatTensor(structures).to(self.device)
        
        candidates = []
        
        with torch.no_grad():
            for i in range(len(compositions)):
                comp = compositions_tensor[i:i+1]
                struct = structures_tensor[i:i+1]
                
                # 计算熵
                entropy = self.entropy_calc.configurational_entropy(comp)
                
                # 预测性质
                props = self.predict_properties(comp, struct)
                
                # 检查条件
                passes_criteria = True
                
                if "entropy" in criteria:
                    min_e, max_e = criteria["entropy"]
                    if not (min_e <= entropy.item() <= max_e):
                        passes_criteria = False
                
                if "hardness" in criteria:
                    min_h, max_h = criteria["hardness"]
                    hardness = props[0, 0].item()
                    if not (min_h <= hardness <= max_h):
                        passes_criteria = False
                
                if passes_criteria:
                    candidates.append({
                        "index": i,
                        "composition": compositions[i],
                        "entropy": entropy.item(),
                        "predicted_properties": {
                            "hardness": props[0, 0].item(),
                            "corrosion_resistance": props[0, 1].item(),
                            "thermal_stability": props[0, 2].item()
                        }
                    })
        
        # 按熵排序
        candidates.sort(key=lambda x: x["entropy"], reverse=True)
        
        return candidates
    
    def transfer_to_new_system(
        self,
        new_system_data: torch.utils.data.DataLoader,
        num_epochs: int = 50
    ) -> Dict[str, List[float]]:
        """
        迁移到新体系
        
        如从合金迁移到氧化物
        """
        print(f"Transferring to new system with {num_epochs} epochs...")
        
        history = defaultdict(list)
        
        for epoch in range(num_epochs):
            epoch_metrics = defaultdict(float)
            num_batches = 0
            
            for batch in new_system_data:
                metrics = self.train_step(batch)
                
                for key, value in metrics.items():
                    epoch_metrics[key] += value
                num_batches += 1
            
            for key in epoch_metrics:
                avg_value = epoch_metrics[key] / num_batches
                history[key].append(avg_value)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Loss={history['total_loss'][-1]:.4f}")
        
        return dict(history)


def generate_high_entropy_data(
    n_samples: int = 1000,
    n_elements: int = 5,
    composition_dim: int = 20
) -> Dict[str, np.ndarray]:
    """
    生成高熵材料数据
    """
    np.random.seed(42)
    
    data = {
        "composition": [],
        "structure": [],
        "phase": [],
        "properties": []
    }
    
    for _ in range(n_samples):
        # 随机选择n_elements个元素
        selected = np.random.choice(composition_dim, n_elements, replace=False)
        
        # 随机成分 (等摩尔或接近等摩尔)
        composition = np.zeros(composition_dim)
        if np.random.rand() > 0.3:  # 70% 等摩尔
            composition[selected] = 1.0 / n_elements
        else:  # 30% 随机偏差
            weights = np.random.dirichlet(np.ones(n_elements))
            composition[selected] = weights
        
        # 结构特征
        structure = np.random.randn(64)
        structure[0] = np.random.uniform(3.0, 6.0)  # 晶格常数
        structure[1] = np.random.uniform(0, 230)    # 空间群
        structure[2] = np.random.uniform(100, 2000) # 德拜温度
        
        # 相标签 (0: 单相固溶体, 1: 双相, 2: 金属间化合物, 3: 非晶, 4: 未知)
        entropy = -np.sum(composition[composition > 0] *
                         np.log(composition[composition > 0] + 1e-10))
        
        if entropy > 1.5:
            phase = 0  # 高熵相
        elif entropy > 1.0:
            phase = 1  # 双相
        elif entropy > 0.5:
            phase = 2  # 金属间化合物
        else:
            phase = 3  # 其他
        
        # 性质 [硬度, 耐蚀性, 热稳定性]
        hardness = 3.0 + 0.5 * entropy + np.random.randn()
        corrosion = 0.3 + 0.1 * entropy + 0.1 * np.random.randn()
        thermal = 500 + 200 * entropy + 100 * np.random.randn()
        
        properties = [hardness, corrosion, thermal]
        
        data["composition"].append(composition)
        data["structure"].append(structure)
        data["phase"].append(phase)
        data["properties"].append(properties)
    
    return {
        k: np.array(v) for k, v in data.items()
    }


# 演示代码
if __name__ == "__main__":
    print("=" * 70)
    print("高熵材料通用发现框架演示 (High Entropy Materials Demo)")
    print("=" * 70)
    
    # 生成数据
    print("\nGenerating high entropy material data...")
    
    data = generate_high_entropy_data(n_samples=1000)
    
    print(f"Data shapes:")
    for key, value in data.items():
        print(f"  {key}: {value.shape}")
    
    # 创建数据加载器
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(data["composition"]),
        torch.FloatTensor(data["structure"]),
        torch.LongTensor(data["phase"]),
        torch.FloatTensor(data["properties"])
    )
    
    def collate_fn(batch):
        return {
            "composition": torch.stack([item[0] for item in batch]),
            "structure": torch.stack([item[1] for item in batch]),
            "phase": torch.stack([item[2] for item in batch]),
            "properties": torch.stack([item[3] for item in batch])
        }
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # 创建框架
    config = HighEntropyConfig()
    framework = HighEntropyMaterialFramework(config)
    
    # 训练
    print("\n" + "=" * 70)
    print("Training Framework")
    print("=" * 70)
    
    for epoch in range(50):
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        for batch in dataloader:
            metrics = framework.train_step(batch)
            
            for key, value in metrics.items():
                epoch_metrics[key] += value
            num_batches += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}:")
            for key in ["total_loss", "phase_loss", "property_loss"]:
                print(f"  {key}: {epoch_metrics[key]/num_batches:.4f}")
    
    # 材料设计
    print("\n" + "=" * 70)
    print("Designing New High Entropy Alloy")
    print("=" * 70)
    
    target_props = {
        "hardness": 8.0,
        "corrosion_resistance": 0.7,
        "thermal_stability": 1500.0
    }
    
    design_result = framework.design_high_entropy_alloy(
        target_props,
        candidate_elements=["Fe", "Cr", "Ni", "Co", "Mn", "Al", "Ti", "Cu"],
        num_iterations=100
    )
    
    print(f"\nDesigned composition entropy: {design_result['entropy']:.3f}")
    print(f"Top 5 elements by composition:")
    
    composition = design_result["composition"]
    top_indices = np.argsort(composition)[-5:][::-1]
    for idx in top_indices:
        if composition[idx] > 0.01:
            print(f"  Element {idx}: {composition[idx]*100:.2f}%")
    
    # 筛选候选
    print("\n" + "=" * 70)
    print("Screening High Entropy Candidates")
    print("=" * 70)
    
    criteria = {
        "entropy": (1.0, 3.0),
        "hardness": (5.0, 15.0)
    }
    
    candidates = framework.screen_compositions(
        data["composition"][:100],
        data["structure"][:100],
        criteria
    )
    
    print(f"Found {len(candidates)} candidates")
    
    if candidates:
        print("\nTop 3 candidates:")
        for i, cand in enumerate(candidates[:3], 1):
            print(f"  {i}. Entropy: {cand['entropy']:.3f}, "
                  f"Hardness: {cand['predicted_properties']['hardness']:.2f}")
    
    print("\n" + "=" * 70)
    print("High Entropy Materials Framework Demo Complete!")
    print("=" * 70)
