"""
CGCNN晶体图特征 (CGCNN Features)
实现CGCNN (Crystal Graph Convolutional Neural Networks) 晶体图特征

作者: DFT-LAMMPS Team
版本: 1.0.0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass


@dataclass
class CGCNNConfig:
    """CGCNN配置"""
    orig_atom_fea_len: int = 92
    nbr_fea_len: int = 41
    
    n_conv: int = 3
    atom_fea_len: int = 64
    h_fea_len: int = 128
    n_h: int = 1
    
    learning_rate: float = 1e-3
    num_epochs: int = 100
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ConvLayer(nn.Module):
    """CGCNN卷积层"""
    
    def __init__(
        self,
        atom_fea_len: int = 64,
        nbr_fea_len: int = 41
    ):
        super().__init__()
        
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        
        self.fc_full = nn.Linear(
            2 * atom_fea_len + nbr_fea_len,
            2 * atom_fea_len
        )
        
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * atom_fea_len)
        self.bn2 = nn.BatchNorm1d(atom_fea_len)
        self.softplus2 = nn.Softplus()
    
    def forward(
        self,
        atom_in_fea: torch.Tensor,
        nbr_fea: torch.Tensor,
        nbr_fea_idx: torch.Tensor
    ) -> torch.Tensor:
        """前向传播"""
        N_atoms = atom_in_fea.shape[0]
        
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        
        atom_in_fea_expanded = atom_in_fea.unsqueeze(1).expand(
            N_atoms, atom_nbr_fea.shape[1], self.atom_fea_len
        )
        
        total_nbr_fea = torch.cat(
            [atom_in_fea_expanded, atom_nbr_fea, nbr_fea],
            dim=2
        )
        
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(
            total_gated_fea.view(-1, self.atom_fea_len * 2)
        ).view(N_atoms, -1, self.atom_fea_len * 2)
        
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        
        return out


class CGCNN(nn.Module):
    """CGCNN模型"""
    
    def __init__(self, config: CGCNNConfig):
        super().__init__()
        
        self.config = config
        
        self.embedding = nn.Linear(
            config.orig_atom_fea_len,
            config.atom_fea_len
        )
        
        self.convs = nn.ModuleList([
            ConvLayer(
                atom_fea_len=config.atom_fea_len,
                nbr_fea_len=config.nbr_fea_len
            )
            for _ in range(config.n_conv)
        ])
        
        self.conv_to_fc = nn.Linear(config.atom_fea_len, config.h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        
        if config.n_h > 1:
            self.fcs = nn.ModuleList([
                nn.Linear(config.h_fea_len, config.h_fea_len)
                for _ in range(config.n_h - 1)
            ])
            self.softpluses = nn.ModuleList([
                nn.Softplus()
                for _ in range(config.n_h - 1)
            ])
        
        self.fc_out = nn.Linear(config.h_fea_len, 1)
    
    def forward(
        self,
        atom_fea: torch.Tensor,
        nbr_fea: torch.Tensor,
        nbr_fea_idx: torch.Tensor,
        crystal_atom_idx: List[torch.Tensor]
    ) -> torch.Tensor:
        """前向传播"""
        atom_fea = self.embedding(atom_fea)
        
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        
        crys_fea = self.conv_to_fc(crys_fea)
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        
        if self.config.n_h > 1:
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        
        out = self.fc_out(crys_fea)
        
        return out
    
    def pooling(
        self,
        atom_fea: torch.Tensor,
        crystal_atom_idx: List[torch.Tensor]
    ) -> torch.Tensor:
        """晶体级别的池化"""
        crys_fea = []
        for idx_map in crystal_atom_idx:
            pooled = torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
            crys_fea.append(pooled)
        
        return torch.cat(crys_fea, dim=0)


class CGCNNDescriptor:
    """CGCNN描述符提取器"""
    
    def __init__(self, config: CGCNNConfig = None):
        self.config = config or CGCNNConfig()
        self.model = CGCNN(self.config).to(self.config.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
    
    def structure_to_crystal_graph(
        self,
        structure: Dict[str, Any],
        radius: float = 8.0,
        max_num_nbr: int = 12
    ) -> Dict[str, torch.Tensor]:
        """将结构转换为晶体图"""
        atom_fea = []
        coords = []
        
        for site in structure.get("sites", []):
            species = site.get("species", [{}])[0].get("element", "H")
            atom_feat = self._get_atom_features(species)
            atom_fea.append(atom_feat)
            coords.append(site.get("abc", [0, 0, 0]))
        
        atom_fea = np.array(atom_fea)
        coords = np.array(coords)
        n_atoms = len(atom_fea)
        
        nbr_fea_idx = []
        nbr_fea = []
        
        for i in range(n_atoms):
            dists = []
            idxs = []
            
            for j in range(n_atoms):
                if i != j:
                    dist = np.linalg.norm(coords[i] - coords[j])
                    dists.append(dist)
                    idxs.append(j)
            
            dists = np.array(dists)
            idxs = np.array(idxs)
            sorted_idx = np.argsort(dists)
            
            in_radius = dists[sorted_idx] < radius
            selected_idx = sorted_idx[in_radius][:max_num_nbr]
            
            selected_nbr_idx = idxs[selected_idx]
            selected_nbr_dist = dists[selected_idx]
            
            nbr_fea_expanded = self._gaussian_expansion(selected_nbr_dist)
            
            while len(selected_nbr_idx) < max_num_nbr:
                selected_nbr_idx = np.append(selected_nbr_idx, 0)
                nbr_fea_expanded = np.vstack([
                    nbr_fea_expanded,
                    np.zeros(self.config.nbr_fea_len)
                ])
            
            nbr_fea_idx.append(selected_nbr_idx[:max_num_nbr])
            nbr_fea.append(nbr_fea_expanded[:max_num_nbr])
        
        nbr_fea_idx = np.array(nbr_fea_idx)
        nbr_fea = np.array(nbr_fea)
        
        return {
            "atom_fea": torch.FloatTensor(atom_fea),
            "nbr_fea": torch.FloatTensor(nbr_fea),
            "nbr_fea_idx": torch.LongTensor(nbr_fea_idx)
        }
    
    def _get_atom_features(self, element: str) -> np.ndarray:
        """获取原子特征"""
        atomic_num = self._element_to_atomic_number(element)
        
        features = np.zeros(92)
        features[atomic_num - 1] = 1.0
        features[90] = atomic_num / 100.0
        features[91] = np.log(atomic_num + 1) / 5.0
        
        return features
    
    def _element_to_atomic_number(self, element: str) -> int:
        """元素符号转原子序数"""
        element_map = {
            "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
            "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
            "S": 16, "Cl": 17, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23,
            "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30
        }
        return element_map.get(element, 1)
    
    def _gaussian_expansion(
        self,
        distances: np.ndarray,
        centers: np.ndarray = None,
        width: float = 0.5
    ) -> np.ndarray:
        """高斯扩展距离"""
        if centers is None:
            centers = np.linspace(0, 8, self.config.nbr_fea_len)
        
        return np.exp(-0.5 * ((distances[:, None] - centers) / width) ** 2)
    
    def extract_descriptor(
        self,
        structure: Dict[str, Any]
    ) -> np.ndarray:
        """提取CGCNN描述符"""
        self.model.eval()
        
        graph_data = self.structure_to_crystal_graph(structure)
        graph_data = {k: v.to(self.config.device) for k, v in graph_data.items()}
        
        crystal_atom_idx = [torch.arange(len(graph_data["atom_fea"]))]
        
        with torch.no_grad():
            atom_fea = self.model.embedding(graph_data["atom_fea"])
            
            for conv in self.model.convs:
                atom_fea = conv(
                    atom_fea,
                    graph_data["nbr_fea"],
                    graph_data["nbr_fea_idx"]
                )
            
            crys_fea = self.model.pooling(atom_fea, crystal_atom_idx)
            descriptor = crys_fea.cpu().numpy().flatten()
        
        return descriptor
    
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
                graph_data, targets = batch
                
                batch_size = len(targets)
                n_atoms = graph_data["atom_fea"].shape[0]
                crystal_atom_idx = [torch.arange(n_atoms)]
                
                predictions = self.model(
                    graph_data["atom_fea"],
                    graph_data["nbr_fea"],
                    graph_data["nbr_fea_idx"],
                    crystal_atom_idx
                )
                
                loss = criterion(predictions.squeeze(), targets)
                
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
    print("CGCNN Crystal Graph Features Demo")
    print("=" * 60)
    
    structure = {
        "sites": [
            {"species": [{"element": "Si"}], "abc": [0, 0, 0]},
            {"species": [{"element": "Si"}], "abc": [0.25, 0.25, 0.25]},
            {"species": [{"element": "O"}], "abc": [0.5, 0.5, 0.5]},
            {"species": [{"element": "O"}], "abc": [0.75, 0.75, 0.75]},
        ]
    }
    
    config = CGCNNConfig()
    descriptor = CGCNNDescriptor(config)
    
    print("\nExtracting CGCNN descriptor...")
    cgcnn_desc = descriptor.extract_descriptor(structure)
    print(f"Descriptor shape: {cgcnn_desc.shape}")
    print(f"Descriptor mean: {cgcnn_desc.mean():.4f}")
    print(f"Descriptor std: {cgcnn_desc.std():.4f}")
    
    print("\n" + "=" * 60)
    print("CGCNN Demo Complete!")
    print("=" * 60)
