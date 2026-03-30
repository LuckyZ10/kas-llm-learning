"""
alignn_wrapper.py
ALIGNN (Atomistic Line Graph Neural Network) 原子线图神经网络

ALIGNN是2021年提出的用于材料性质预测的强大图神经网络,
使用线图(line graph)捕获原子间相互作用的角度信息,
在Materials Project数据集上达到SOTA性能。

References:
- Choudhary et al. (2021) "Unified graph neural network force-field"
- Choudhary et al. (2022) "Atomistic Line Graph Neural Network"
- 2024进展: ALIGNN用于快速材料筛选和高通量计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Set
from dataclasses import dataclass
import numpy as np


@dataclass
class GraphData:
    """图数据结构"""
    node_features: torch.Tensor      # [n_nodes, node_feat_dim]
    edge_index: torch.Tensor         # [2, n_edges] 主图边
    edge_features: torch.Tensor      # [n_edges, edge_feat_dim]
    line_edge_index: torch.Tensor    # [2, n_line_edges] 线图边
    line_edge_features: torch.Tensor # [n_line_edges, line_feat_dim]
    batch: Optional[torch.Tensor] = None
    target: Optional[torch.Tensor] = None


class AtomEmbedding(nn.Module):
    """原子嵌入层"""
    
    def __init__(
        self,
        num_elements: int = 100,
        embedding_dim: int = 128,
        scheme: str = "cgcnn"  # "random", "cgcnn", "onehot"
    ):
        super().__init__()
        self.scheme = scheme
        
        if scheme == "random":
            self.embedding = nn.Embedding(num_elements, embedding_dim)
        elif scheme == "cgcnn":
            # CGCNN风格的元素属性嵌入
            self.embedding = nn.Embedding(num_elements, embedding_dim)
            # 可添加元素属性特征 (电负性、原子半径等)
        elif scheme == "onehot":
            self.register_buffer('eye', torch.eye(num_elements))
            self.proj = nn.Linear(num_elements, embedding_dim)
    
    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        if self.scheme == "onehot":
            onehot = self.eye[atomic_numbers - 1]  # 原子序数从1开始
            return self.proj(onehot)
        return self.embedding(atomic_numbers - 1)


class EdgeEmbedding(nn.Module):
    """边嵌入层 - 基于距离"""
    
    def __init__(
        self,
        num_rbf: int = 50,
        cutoff: float = 8.0,
        embedding_dim: int = 128
    ):
        super().__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        
        # 高斯径向基
        centers = torch.linspace(0, cutoff, num_rbf)
        self.register_buffer('centers', centers)
        
        widths = torch.ones(num_rbf) * (cutoff / num_rbf)
        self.register_buffer('widths', widths)
        
        # RBF到嵌入的投影
        self.rbf_proj = nn.Sequential(
            nn.Linear(num_rbf, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        计算边嵌入
        
        Args:
            distances: [n_edges] 边距离
        """
        # RBF特征
        rbf = torch.exp(-((distances.unsqueeze(-1) - self.centers) ** 2) / 
                       (2 * self.widths ** 2))
        
        # 截止函数
        cutoff_val = 0.5 * (torch.cos(distances * np.pi / self.cutoff) + 1)
        cutoff_val = cutoff_val * (distances < self.cutoff).float()
        
        rbf = rbf * cutoff_val.unsqueeze(-1)
        
        return self.rbf_proj(rbf)


class ALIGNNLayer(nn.Module):
    """
    ALIGNN层 - 同时更新主图和线图
    
    这是ALIGNN的核心创新:
    1. 线图边捕获键角信息 (三线关系)
    2. 主图和线图交替更新
    """
    
    def __init__(
        self,
        node_dim: int = 128,
        edge_dim: int = 128,
        line_edge_dim: int = 128,
        num_heads: int = 4
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        # 主图边更新 (考虑相连线图边)
        self.edge_update = nn.Sequential(
            nn.Linear(edge_dim * 2 + line_edge_dim, edge_dim * 2),
            nn.SiLU(),
            nn.Linear(edge_dim * 2, edge_dim)
        )
        
        # 线图边更新
        self.line_edge_update = nn.Sequential(
            nn.Linear(edge_dim * 2, line_edge_dim),
            nn.SiLU(),
            nn.Linear(line_edge_dim, line_edge_dim)
        )
        
        # 节点更新
        self.node_update = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim * 2),
            nn.SiLU(),
            nn.Linear(node_dim * 2, node_dim)
        )
        
        # 注意力权重计算
        self.attention = nn.MultiheadAttention(node_dim, num_heads, batch_first=True)
        
        # 层归一化
        self.node_norm = nn.LayerNorm(node_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)
        self.line_norm = nn.LayerNorm(line_edge_dim)
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        line_edge_index: torch.Tensor,
        line_edge_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ALIGNN层前向传播
        
        Args:
            node_features: [n_nodes, node_dim]
            edge_index: [2, n_edges] 主图边
            edge_features: [n_edges, edge_dim]
            line_edge_index: [2, n_line_edges] 线图边
            line_edge_features: [n_line_edges, line_edge_dim]
        """
        # 1. 更新线图边
        if line_edge_index.shape[1] > 0:
            line_src, line_dst = line_edge_index
            
            # 线图边连接两条主图边
            # line_edge_features[i] 对应 edge_features[line_src[i]] 和 edge_features[line_dst[i]]
            line_input = torch.cat([
                edge_features[line_src],
                edge_features[line_dst]
            ], dim=-1)
            
            line_update = self.line_edge_update(line_input)
            line_edge_features = self.line_norm(line_edge_features + line_update)
        
        # 2. 更新主图边 (考虑相连线图边)
        src, dst = edge_index
        
        # 构建线图邻接矩阵用于边更新
        # 简化: 使用平均池化
        if line_edge_index.shape[1] > 0:
            line_aggr = torch.zeros_like(edge_features)
            line_aggr.index_add_(0, line_edge_index[1], line_edge_features)
            
            # 统计每个主图边相连的线图边数量
            count = torch.zeros(edge_features.shape[0], device=edge_features.device)
            count.scatter_add_(0, line_edge_index[1], torch.ones(line_edge_index.shape[1], device=edge_features.device))
            count = count.clamp(min=1)
            line_aggr = line_aggr / count.unsqueeze(-1)
        else:
            line_aggr = torch.zeros_like(edge_features)
        
        edge_input = torch.cat([edge_features, edge_features, line_aggr], dim=-1)
        edge_update = self.edge_update(edge_input)
        edge_features = self.edge_norm(edge_features + edge_update)
        
        # 3. 更新节点
        # 聚合边消息
        node_aggr = torch.zeros_like(node_features)
        node_aggr.index_add_(0, dst, edge_features)
        
        # 统计入度
        degree = torch.zeros(node_features.shape[0], device=node_features.device)
        degree.scatter_add_(0, dst, torch.ones(edge_features.shape[0], device=edge_features.device))
        degree = degree.clamp(min=1)
        node_aggr = node_aggr / degree.unsqueeze(-1)
        
        node_input = torch.cat([node_features, node_aggr], dim=-1)
        node_update = self.node_update(node_input)
        node_features = self.node_norm(node_features + node_update)
        
        # 4. 自注意力 refinement
        node_features_unsq = node_features.unsqueeze(0)
        attended, _ = self.attention(
            node_features_unsq, node_features_unsq, node_features_unsq
        )
        node_features = attended.squeeze(0)
        
        return node_features, edge_features, line_edge_features


class ALIGNN(nn.Module):
    """
    ALIGNN模型 - 完整实现
    
    用于材料性质预测的端到端模型
    """
    
    def __init__(
        self,
        num_elements: int = 100,
        node_dim: int = 128,
        edge_dim: int = 128,
        line_edge_dim: int = 128,
        num_layers: int = 4,
        num_rbf: int = 50,
        cutoff: float = 8.0,
        output_dim: int = 1,
        task_type: str = "regression"  # "regression", "classification", "multitask"
    ):
        super().__init__()
        self.num_layers = num_layers
        self.task_type = task_type
        self.output_dim = output_dim
        
        # 嵌入层
        self.atom_embedding = AtomEmbedding(num_elements, node_dim, scheme="cgcnn")
        self.edge_embedding = EdgeEmbedding(num_rbf, cutoff, edge_dim)
        self.line_edge_embedding = nn.Linear(1, line_edge_dim)  # 线图边特征简化
        
        # ALIGNN层
        self.alignn_layers = nn.ModuleList([
            ALIGNNLayer(node_dim, edge_dim, line_edge_dim)
            for _ in range(num_layers)
        ])
        
        # 读出层
        self.readout = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim // 2),
            nn.SiLU(),
            nn.Linear(node_dim // 2, output_dim)
        )
        
        # 多头任务支持
        if task_type == "multitask":
            self.task_heads = nn.ModuleDict({
                'formation_energy': nn.Linear(node_dim // 2, 1),
                'band_gap': nn.Linear(node_dim // 2, 1),
                'bulk_modulus': nn.Linear(node_dim // 2, 1),
                'shear_modulus': nn.Linear(node_dim // 2, 1),
            })
    
    def build_line_graph(
        self,
        edge_index: torch.Tensor,
        positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建线图
        
        线图边连接共享一个顶点的两条主图边
        捕获键角信息
        
        Returns:
            line_edge_index: [2, n_line_edges]
            line_edge_features: [n_line_edges, 1] (角度)
        """
        n_edges = edge_index.shape[1]
        device = edge_index.device
        
        # 构建边的邻接关系
        # 线图边 (e_i, e_j) 存在当两条边共享一个顶点
        line_edges = []
        angles = []
        
        for i in range(n_edges):
            for j in range(i + 1, n_edges):
                # 检查是否共享顶点
                shared = set(edge_index[:, i].tolist()) & set(edge_index[:, j].tolist())
                
                if len(shared) > 0:
                    # 找到形成角的三条边
                    # 简化: 记录所有共享顶点的边对
                    line_edges.append([i, j])
                    
                    # 计算角度 (简化版)
                    if positions is not None:
                        src_i, dst_i = edge_index[:, i]
                        src_j, dst_j = edge_index[:, j]
                        
                        # 找到中心原子和两个邻居
                        center = list(shared)[0]
                        neighbor_i = dst_i if src_i == center else src_i
                        neighbor_j = dst_j if src_j == center else src_j
                        
                        v1 = positions[neighbor_i] - positions[center]
                        v2 = positions[neighbor_j] - positions[center]
                        
                        cos_angle = F.cosine_similarity(
                            v1.unsqueeze(0), v2.unsqueeze(0)
                        )
                        angle = torch.acos(cos_angle.clamp(-1, 1))
                        angles.append(angle.item())
                    else:
                        angles.append(0.0)
        
        if len(line_edges) == 0:
            line_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            line_edge_features = torch.zeros((0, 1), device=device)
        else:
            line_edge_index = torch.tensor(line_edges, dtype=torch.long, device=device).t()
            # 对称化
            line_edge_index_sym = torch.cat([
                line_edge_index,
                torch.stack([line_edge_index[1], line_edge_index[0]], dim=0)
            ], dim=1)
            
            angles_sym = angles + angles
            line_edge_features = torch.tensor(angles_sym, device=device).unsqueeze(-1)
            line_edge_index = line_edge_index_sym
        
        return line_edge_index, line_edge_features
    
    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        ALIGNN前向传播
        
        Args:
            atomic_numbers: [n_nodes] 原子序数
            positions: [n_nodes, 3] 位置
            edge_index: [2, n_edges] 边
            batch: [n_nodes] 批次索引
        """
        # 计算边距离
        src, dst = edge_index
        edge_vectors = positions[dst] - positions[src]
        edge_distances = torch.norm(edge_vectors, dim=-1)
        
        # 初始嵌入
        node_features = self.atom_embedding(atomic_numbers)
        edge_features = self.edge_embedding(edge_distances)
        
        # 构建线图
        line_edge_index, line_edge_features = self.build_line_graph(
            edge_index, positions
        )
        line_edge_features = self.line_edge_embedding(line_edge_features)
        
        # ALIGNN消息传递
        for layer in self.alignn_layers:
            node_features, edge_features, line_edge_features = layer(
                node_features, edge_index, edge_features,
                line_edge_index, line_edge_features
            )
        
        # 全局池化
        if batch is None:
            graph_features = node_features.mean(dim=0, keepdim=True)
        else:
            # 按批次聚合
            num_graphs = batch.max().item() + 1
            graph_features = torch.zeros(
                num_graphs, node_features.shape[1],
                device=node_features.device
            )
            graph_features.index_add_(0, batch, node_features)
            
            # 平均
            counts = torch.zeros(num_graphs, device=node_features.device)
            counts.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
            counts = counts.clamp(min=1).unsqueeze(-1)
            graph_features = graph_features / counts
        
        # 输出
        output = self.readout(graph_features)
        
        result = {'output': output}
        
        # 多任务输出
        if self.task_type == "multitask":
            for task_name, head in self.task_heads.items():
                result[task_name] = head(graph_features)
        
        return result
    
    def predict_properties(
        self,
        structures: List[Dict]
    ) -> torch.Tensor:
        """
        批量预测材料性质
        
        Args:
            structures: 结构列表,每个包含原子数、位置等
        """
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for struct in structures:
                output = self.forward(
                    atomic_numbers=struct['atomic_numbers'],
                    positions=struct['positions'],
                    edge_index=struct['edge_index']
                )
                predictions.append(output['output'])
        
        return torch.cat(predictions, dim=0)


class ALIGNNForceField(ALIGNN):
    """
    ALIGNN力场 - 预测能量和力
    
    扩展ALIGNN用于分子动力学模拟
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, output_dim=1, **kwargs)
        
        # 力预测头
        self.force_head = nn.Sequential(
            nn.Linear(self.readout[0].in_features, 128),
            nn.SiLU(),
            nn.Linear(128, 3)
        )
    
    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """预测能量和力"""
        # 启用梯度计算用于力预测
        positions_grad = positions.requires_grad_(True)
        
        # 父类前向
        result = super().forward(atomic_numbers, positions_grad, edge_index, batch)
        
        energy = result['output'].sum()
        
        # 计算力 (-dE/dr)
        forces = -torch.autograd.grad(
            energy, positions_grad,
            create_graph=self.training
        )[0]
        
        return {
            'energy': energy,
            'forces': forces,
            'atomic_energies': result['output']
        }


def train_alignn(
    model: ALIGNN,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int = 100,
    lr: float = 1e-3,
    device: str = 'cuda'
) -> Dict[str, List[float]]:
    """
    训练ALIGNN模型
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            optimizer.zero_grad()
            output = model(
                batch['atomic_numbers'],
                batch['positions'],
                batch['edge_index'],
                batch.get('batch')
            )
            
            loss = criterion(output['output'], batch['target'])
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # 验证
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                output = model(
                    batch['atomic_numbers'],
                    batch['positions'],
                    batch['edge_index'],
                    batch.get('batch')
                )
                
                loss = criterion(output['output'], batch['target'])
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        
        scheduler.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    return history


if __name__ == "__main__":
    print("=" * 60)
    print("ALIGNN Demo - Atomistic Line Graph Neural Network")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 创建模型
    print("\n1. Creating ALIGNN model")
    model = ALIGNN(
        num_elements=20,
        node_dim=64,
        edge_dim=64,
        line_edge_dim=32,
        num_layers=3,
        output_dim=1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建测试数据
    print("\n2. Creating test structure")
    
    # 简单的Li2O结构
    atomic_numbers = torch.tensor([3, 3, 8], device=device)  # Li, Li, O
    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.25, 0.25, 0.25]
    ], device=device)
    
    # 构建边 (简化: 全连接)
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 2],
        [1, 2, 0, 2, 0, 1]
    ], device=device)
    
    # 前向传播
    print("\n3. Running forward pass")
    model.eval()
    with torch.no_grad():
        output = model(atomic_numbers, positions, edge_index)
    
    print(f"Predicted property: {output['output'].item():.4f}")
    
    # 构建线图
    print("\n4. Building line graph")
    line_edge_index, line_edge_features = model.build_line_graph(
        edge_index, positions
    )
    print(f"Number of line edges: {line_edge_index.shape[1]}")
    print(f"Line graph captures bond angles between edges")
    
    # 力场测试
    print("\n5. Testing ALIGNN Force Field")
    ff_model = ALIGNNForceField(
        num_elements=20,
        node_dim=64,
        edge_dim=64,
        num_layers=2
    ).to(device)
    
    ff_output = ff_model(atomic_numbers, positions, edge_index)
    print(f"Energy: {ff_output['energy'].item():.4f} eV")
    print(f"Forces shape: {ff_output['forces'].shape}")
    
    print("\n" + "=" * 60)
    print("ALIGNN Demo completed!")
    print("Key features:")
    print("- Line graph captures 3-body interactions")
    print("- Superior performance on materials property prediction")
    print("- Supports both scalar properties and forces")
    print("=" * 60)
