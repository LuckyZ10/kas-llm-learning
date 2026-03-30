"""
MEGNet图神经网络描述符 (MEGNet Descriptor)
实现MEGNet (MatErials Graph Network) 材料图神经网络描述符

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
class MEGNetConfig:
    """MEGNet配置"""
    n_atom_features: int = 64
    n_bond_features: int = 64
    n_global_features: int = 64
    
    n_conv_layers: int = 3
    n_hidden_layers: int = 2
    hidden_dim: int = 128
    
    learning_rate: float = 1e-3
    num_epochs: int = 100
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MEGNetLayer(nn.Module):
    """MEGNet图卷积层"""
    
    def __init__(
        self,
        n_atom_features: int = 64,
        n_bond_features: int = 64,
        n_global_features: int = 64
    ):
        super().__init__()
        
        # 原子更新
        self.atom_update = nn.Sequential(
            nn.Linear(n_atom_features * 2 + n_bond_features + n_global_features, 128),
            nn.Softplus(),
            nn.Linear(128, n_atom_features)
        )
        
        # 键更新
        self.bond_update = nn.Sequential(
            nn.Linear(n_atom_features * 2 + n_bond_features + n_global_features, 128),
            nn.Softplus(),
            nn.Linear(128, n_bond_features)
        )
        
        # 全局更新
        self.global_update = nn.Sequential(
            nn.Linear(n_atom_features + n_bond_features + n_global_features, 128),
            nn.Softplus(),
            nn.Linear(128, n_global_features)
        )
    
    def forward(
        self,
        atom_features: torch.Tensor,
        bond_features: torch.Tensor,
        global_features: torch.Tensor,
        atom_bond_indices: torch.Tensor,
        bond_atom_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            atom_features: [N_atoms, n_atom_features]
            bond_features: [N_bonds, n_bond_features]
            global_features: [N_graphs, n_global_features]
            atom_bond_indices: [N_bonds, 2] - 每个键连接的原子索引
            bond_atom_indices: [N_atoms, max_degree] - 每个原子连接的键索引
        
        Returns:
            更新后的特征
        """
        # 更新键特征
        updated_bonds = self._update_bonds(
            atom_features, bond_features, global_features,
            atom_bond_indices
        )
        
        # 更新原子特征
        updated_atoms = self._update_atoms(
            atom_features, updated_bonds, global_features,
            bond_atom_indices
        )
        
        # 更新全局特征
        updated_global = self._update_global(
            updated_atoms, updated_bonds, global_features
        )
        
        return updated_atoms, updated_bonds, updated_global
    
    def _update_bonds(
        self,
        atom_features: torch.Tensor,
        bond_features: torch.Tensor,
        global_features: torch.Tensor,
        atom_bond_indices: torch.Tensor
    ) -> torch.Tensor:
        """更新键特征"""
        n_bonds = bond_features.size(0)
        
        # 获取连接的原子特征
        atom_i = atom_features[atom_bond_indices[:, 0]]
        atom_j = atom_features[atom_bond_indices[:, 1]]
        
        # 扩展全局特征
        global_expanded = global_features.repeat(n_bonds, 1)
        
        # 拼接特征
        combined = torch.cat([atom_i, atom_j, bond_features, global_expanded], dim=-1)
        
        # 更新
        updated = self.bond_update(combined)
        
        return updated + bond_features  # 残差连接
    
    def _update_atoms(
        self,
        atom_features: torch.Tensor,
        bond_features: torch.Tensor,
        global_features: torch.Tensor,
        bond_atom_indices: torch.Tensor
    ) -> torch.Tensor:
        """更新原子特征"""
        n_atoms = atom_features.size(0)
        n_atom_feat = atom_features.size(1)
        
        # 聚合邻居键特征
        aggregated_bonds = torch.zeros(n_atoms, n_atom_feat, device=atom_features.device)
        
        for i in range(n_atoms):
            neighbor_bonds = bond_atom_indices[i]
            valid_mask = neighbor_bonds >= 0
            
            if valid_mask.any():
                valid_bonds = neighbor_bonds[valid_mask]
                bond_feats = bond_features[valid_bonds]
                aggregated_bonds[i] = bond_feats.mean(dim=0)
        
        # 扩展全局特征
        global_expanded = global_features.repeat(n_atoms, 1)
        
        # 拼接
        combined = torch.cat([
            atom_features,
            aggregated_bonds,
            atom_features,  # 自身特征
            global_expanded
        ], dim=-1)
        
        # 更新
        updated = self.atom_update(combined)
        
        return updated + atom_features  # 残差连接
    
    def _update_global(
        self,
        atom_features: torch.Tensor,
        bond_features: torch.Tensor,
        global_features: torch.Tensor
    ) -> torch.Tensor:
        """更新全局特征"""
        # 聚合所有原子和键特征
        atom_pooled = atom_features.mean(dim=0, keepdim=True)
        bond_pooled = bond_features.mean(dim=0, keepdim=True)
        
        # 拼接
        combined = torch.cat([
            atom_pooled,
            bond_pooled,
            global_features
        ], dim=-1)
        
        # 更新
        updated = self.global_update(combined)
        
        return updated + global_features  # 残差连接


class MEGNet(nn.Module):
    """MEGNet模型"""
    
    def __init__(self, config: MEGNetConfig):
        super().__init__()
        
        self.config = config
        
        # 嵌入层
        self.atom_embedding = nn.Linear(100, config.n_atom_features)  # 100 = max atomic number
        self.bond_embedding = nn.Linear(10, config.n_bond_features)   # 10 = bond feature dim
        self.global_embedding = nn.Linear(10, config.n_global_features)
        
        # 图卷积层
        self.conv_layers = nn.ModuleList([
            MEGNetLayer(
                config.n_atom_features,
                config.n_bond_features,
                config.n_global_features
            )
            for _ in range(config.n_conv_layers)
        ])
        
        # 读出层
        self.readout = nn.Sequential(
            nn.Linear(config.n_global_features, config.hidden_dim),
            nn.Softplus(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.Softplus(),
            nn.Linear(config.hidden_dim // 2, 1)
        )
    
    def forward(
        self,
        atom_types: torch.Tensor,
        bond_features: torch.Tensor,
        global_features: torch.Tensor,
        atom_bond_indices: torch.Tensor,
        bond_atom_indices: torch.Tensor
    ) -> torch.Tensor:
        """前向传播"""
        # 嵌入
        atom_features = self.atom_embedding(
            F.one_hot(atom_types, num_classes=100).float()
        )
        bond_features = self.bond_embedding(bond_features)
        global_features = self.global_embedding(global_features)
        
        # 图卷积
        for conv_layer in self.conv_layers:
            atom_features, bond_features, global_features = conv_layer(
                atom_features,
                bond_features,
                global_features,
                atom_bond_indices,
                bond_atom_indices
            )
        
        # 读出
        output = self.readout(global_features)
        
        return output


class MEGNetDescriptor:
    """MEGNet描述符提取器"""
    
    def __init__(self, config: MEGNetConfig = None):
        self.config = config or MEGNetConfig()
        self.model = MEGNet(self.config).to(self.config.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
    
    def structure_to_graph(
        self,
        structure: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """将结构转换为图表示"""
        # 提取原子类型
        atom_types = []
        for site in structure.get("sites", []):
            species = site.get("species", [{}])[0].get("element", "H")
            atomic_num = self._element_to_atomic_number(species)
            atom_types.append(atomic_num)
        
        atom_types = torch.LongTensor(atom_types)
        n_atoms = len(atom_types)
        
        # 构建键连接
        bonds = []
        bond_features = []
        
        coords = np.array([site.get("xyz", [0, 0, 0])
                          for site in structure.get("sites", [])])
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < 3.0:  # 3 Å 截断
                    bonds.append([i, j])
                    bond_features.append([
                        dist,
                        1.0 / (dist + 0.1),
                        np.exp(-dist),
                        0, 0, 0, 0, 0, 0, 0
                    ])
        
        if not bonds:
            bonds = [[0, 0]]
            bond_features = [[0.0] * 10]
        
        bonds = torch.LongTensor(bonds)
        bond_features = torch.FloatTensor(bond_features)[:, :10]
        
        # 构建索引
        max_degree = max(
            sum(1 for b in bonds if i in b) for i in range(n_atoms)
        ) if len(bonds) > 0 else 1
        
        bond_atom_indices = torch.full((n_atoms, max_degree), -1, dtype=torch.long)
        
        for i in range(n_atoms):
            idx = 0
            for bond_idx, bond in enumerate(bonds):
                if i in bond:
                    if idx < max_degree:
                        bond_atom_indices[i, idx] = bond_idx
                        idx += 1
        
        # 全局特征
        global_features = torch.FloatTensor([
            [structure.get("density", 0.0),
             len(atom_types),
             0, 0, 0, 0, 0, 0, 0, 0]
        ])
        
        return {
            "atom_types": atom_types,
            "bond_features": bond_features,
            "global_features": global_features,
            "atom_bond_indices": bonds,
            "bond_atom_indices": bond_atom_indices
        }
    
    def _element_to_atomic_number(self, element: str) -> int:
        """元素符号转原子序数"""
        element_map = {
            "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
            "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
            "S": 16, "Cl": 17, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23,
            "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30
        }
        return element_map.get(element, 1)
    
    def extract_descriptor(
        self,
        structure: Dict[str, Any]
    ) -> np.ndarray:
        """提取MEGNet描述符"""
        self.model.eval()
        
        graph_data = self.structure_to_graph(structure)
        
        # 将数据移到设备
        graph_data = {k: v.to(self.config.device) for k, v in graph_data.items()}
        
        with torch.no_grad():
            # 获取最后一层全局特征
            atom_features = self.model.atom_embedding(
                F.one_hot(graph_data["atom_types"], num_classes=100).float()
            )
            bond_features = self.model.bond_embedding(graph_data["bond_features"])
            global_features = self.model.global_embedding(graph_data["global_features"])
            
            for conv_layer in self.model.conv_layers:
                atom_features, bond_features, global_features = conv_layer(
                    atom_features,
                    bond_features,
                    global_features,
                    graph_data["atom_bond_indices"],
                    graph_data["bond_atom_indices"]
                )
        
        # 返回全局特征作为描述符
        return global_features.cpu().numpy().flatten()
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader
    ) -> Dict[str, List[float]]:
        """训练模型"""
        self.model.train()
        criterion = nn.MSELoss()
        
        history = {"loss": []}
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                # 解包批次
                graph_data, targets = batch
                
                # 前向传播
                predictions = self.model(**graph_data)
                loss = criterion(predictions.squeeze(), targets)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            history["loss"].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        return history


# 演示代码
if __name__ == "__main__":
    print("=" * 60)
    print("MEGNet Descriptor Demo")
    print("=" * 60)
    
    # 创建模拟结构
    structure = {
        "sites": [
            {"species": [{"element": "Si"}], "xyz": [0, 0, 0]},
            {"species": [{"element": "O"}], "xyz": [1.5, 0, 0]},
            {"species": [{"element": "O"}], "xyz": [0, 1.5, 0]},
            {"species": [{"element": "O"}], "xyz": [0, 0, 1.5]},
        ],
        "density": 2.65
    }
    
    # 创建描述符提取器
    config = MEGNetConfig()
    descriptor = MEGNetDescriptor(config)
    
    # 提取描述符
    print("\nExtracting MEGNet descriptor...")
    megnet_desc = descriptor.extract_descriptor(structure)
    print(f"Descriptor shape: {megnet_desc.shape}")
    print(f"Descriptor mean: {megnet_desc.mean():.4f}")
    print(f"Descriptor std: {megnet_desc.std():.4f}")
    
    print("\n" + "=" * 60)
    print("MEGNet Demo Complete!")
    print("=" * 60)
