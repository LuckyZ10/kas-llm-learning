"""
diffusion_materials.py
生成式AI材料设计 - 扩散模型生成晶体结构

实现了CDVAE (Crystal Diffusion Variational Autoencoder) 和 DiffCSP (Diffusion for Crystal Structure Prediction)
等前沿扩散模型用于晶体结构生成。

References:
- Xie et al. (2022) "Crystal Diffusion Variational Autoencoder for Periodic Material Generation"
- Jiao et al. (2023) "Crystal Structure Prediction by Jointly Modeling Spatial and Periodic Invariances"
- 2024-2025进展: 流匹配模型(Flow Matching)已在晶体生成中超越传统扩散模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import warnings
from collections import defaultdict


@dataclass
class CrystalFeatures:
    """晶体特征表示"""
    frac_coords: torch.Tensor      # 分数坐标 [n_atoms, 3]
    atom_types: torch.Tensor       # 原子类型 [n_atoms]
    lengths: torch.Tensor          # 晶胞边长 [3]
    angles: torch.Tensor           # 晶胞角度 [3]
    num_atoms: int
    
    def to_dict(self) -> Dict:
        return {
            'frac_coords': self.frac_coords,
            'atom_types': self.atom_types,
            'lengths': self.lengths,
            'angles': self.angles,
            'num_atoms': self.num_atoms
        }


class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间嵌入 - 用于扩散时间步"""
    
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class EquivariantGraphConv(nn.Module):
    """
    E(3)等变图卷积层
    处理晶体结构的平移和旋转等变性
    """
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        # 边特征网络
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # 节点更新网络
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
        # 坐标更新网络 (E(3)等变)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(
        self, 
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_feat: [n_nodes, node_dim]
            edge_index: [2, n_edges]
            edge_attr: [n_edges, edge_dim]
            coords: [n_nodes, 3]
        Returns:
            updated_node_feat, updated_coords
        """
        src, dst = edge_index
        
        # 边特征
        edge_feat = torch.cat([node_feat[src], node_feat[dst], edge_attr], dim=-1)
        edge_message = self.edge_mlp(edge_feat)
        
        # 聚合到节点
        aggr_message = torch.zeros_like(node_feat)
        aggr_message.index_add_(0, dst, edge_message)
        
        # 更新节点特征
        node_input = torch.cat([node_feat, aggr_message], dim=-1)
        node_update = self.node_mlp(node_input)
        updated_nodes = node_feat + node_update
        
        # 等变坐标更新
        coord_weights = self.coord_mlp(edge_message)
        coord_diff = coords[src] - coords[dst]
        coord_diff = coord_diff / (torch.norm(coord_diff, dim=-1, keepdim=True) + 1e-8)
        
        coord_update = torch.zeros_like(coords)
        weighted_diff = coord_weights * coord_diff
        coord_update.index_add_(0, dst, weighted_diff)
        
        updated_coords = coords + coord_update
        
        return updated_nodes, updated_coords


class PeriodicGraphNeuralNetwork(nn.Module):
    """
    周期性图神经网络 - 处理晶体结构的周期性边界条件
    """
    
    def __init__(
        self,
        node_dim: int = 128,
        edge_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 4,
        max_neighbors: int = 24,
        cutoff: float = 8.0
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.max_neighbors = max_neighbors
        self.cutoff = cutoff
        
        # 原子类型嵌入
        self.atom_embedding = nn.Embedding(100, node_dim)
        
        # 时间嵌入投影
        self.time_proj = nn.Linear(128, node_dim)
        
        # 图卷积层
        self.conv_layers = nn.ModuleList([
            EquivariantGraphConv(node_dim, edge_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # 边特征提取 (距离 + 径向基)
        self.rbf = RadialBasisFunction(cutoff, edge_dim)
        
    def build_periodic_graph(
        self,
        frac_coords: torch.Tensor,
        lengths: torch.Tensor,
        angles: torch.Tensor,
        num_atoms: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        构建周期性邻居图
        
        Returns:
            edge_index: [2, n_edges]
            edge_attr: [n_edges, edge_dim]
            cart_coords: [n_atoms, 3] 笛卡尔坐标
        """
        # 分数坐标转笛卡尔坐标
        lattice = self._build_lattice(lengths, angles)
        cart_coords = frac_coords @ lattice
        
        # 考虑周期性镜像找到邻居
        edge_list = []
        edge_attr_list = []
        
        # 考虑邻近晶胞 (3x3x3 = 27个镜像)
        shifts = torch.tensor([
            [i, j, k] for i in [-1, 0, 1] 
            for j in [-1, 0, 1] 
            for k in [-1, 0, 1]
        ], device=frac_coords.device, dtype=frac_coords.dtype)
        
        for i in range(num_atoms):
            # 计算到所有镜像的距离
            neighbors = []
            for shift in shifts:
                shifted_coords = cart_coords + shift @ lattice
                dists = torch.norm(shifted_coords - cart_coords[i:i+1], dim=-1)
                
                mask = (dists < self.cutoff) & (dists > 0.01)
                valid_indices = torch.where(mask)[0]
                valid_dists = dists[mask]
                
                for idx, dist in zip(valid_indices, valid_dists):
                    neighbors.append((idx.item(), dist.item()))
            
            # 选择最近的邻居
            neighbors.sort(key=lambda x: x[1])
            neighbors = neighbors[:self.max_neighbors]
            
            for idx, dist in neighbors:
                edge_list.append([i, idx])
                edge_attr_list.append(dist)
        
        if len(edge_list) == 0:
            # 处理没有邻居的情况
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=frac_coords.device)
            edge_attr = torch.zeros((0, self.edge_dim), device=frac_coords.device)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long, device=frac_coords.device).t()
            edge_dist = torch.tensor(edge_attr_list, device=frac_coords.device)
            edge_attr = self.rbf(edge_dist)
        
        return edge_index, edge_attr, cart_coords
    
    def _build_lattice(self, lengths: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """从晶胞参数构建晶格矩阵"""
        a, b, c = lengths
        alpha, beta, gamma = angles * np.pi / 180
        
        lx = a
        xy = b * torch.cos(gamma)
        xz = c * torch.cos(beta)
        ly = b * torch.sin(gamma)
        yz = (b * c * torch.cos(alpha) - xy * xz) / ly
        lz = torch.sqrt(c**2 - xz**2 - yz**2)
        
        lattice = torch.tensor([
            [lx, 0, 0],
            [xy, ly, 0],
            [xz, yz, lz]
        ], device=lengths.device)
        
        return lattice
    
    def forward(
        self,
        atom_types: torch.Tensor,
        frac_coords: torch.Tensor,
        lengths: torch.Tensor,
        angles: torch.Tensor,
        timesteps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            node_pred: [n_atoms, node_dim] 原子类型预测
            coord_pred: [n_atoms, 3] 坐标预测
        """
        num_atoms = atom_types.shape[0]
        
        # 构建图
        edge_index, edge_attr, cart_coords = self.build_periodic_graph(
            frac_coords, lengths, angles, num_atoms
        )
        
        # 初始节点特征
        node_feat = self.atom_embedding(atom_types)
        
        # 添加时间嵌入
        time_emb = self.time_proj(timesteps)
        node_feat = node_feat + time_emb
        
        # 图卷积传播
        coords = cart_coords.clone()
        for conv in self.conv_layers:
            if edge_index.shape[1] > 0:
                node_feat, coords = conv(node_feat, edge_index, edge_attr, coords)
        
        return node_feat, coords


class RadialBasisFunction(nn.Module):
    """径向基函数用于边特征"""
    
    def __init__(self, cutoff: float, num_rbf: int):
        super().__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        
        # 高斯中心
        centers = torch.linspace(0, cutoff, num_rbf)
        self.register_buffer('centers', centers)
        
        # 宽度参数
        widths = torch.ones(num_rbf) * (cutoff / num_rbf)
        self.register_buffer('widths', widths)
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """计算RBF特征"""
        distances = distances.unsqueeze(-1)
        rbf = torch.exp(-((distances - self.centers) ** 2) / (2 * self.widths ** 2))
        # 截止函数
        cutoff_val = 0.5 * (torch.cos(distances * np.pi / self.cutoff) + 1)
        cutoff_val = cutoff_val * (distances < self.cutoff).float()
        return rbf * cutoff_val


class CDVAE(nn.Module):
    """
    CDVAE: Crystal Diffusion Variational Autoencoder
    
    用于生成晶体结构的条件扩散模型
    2022年发表于Nature Communications, 开创了晶体生成的新范式
    """
    
    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 256,
        num_layers: int = 4,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_timesteps = num_timesteps
        
        # 扩散调度
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        
        # 编码器
        self.encoder = PeriodicGraphNeuralNetwork(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        # 变分推断
        self.fc_mu = nn.Linear(node_dim, latent_dim)
        self.fc_logvar = nn.Linear(node_dim, latent_dim)
        
        # 属性预测器 (条件生成)
        self.property_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3)  # 预测: 带隙、能量、稳定性
        )
        
        # 解码器 (扩散模型)
        self.time_embedding = SinusoidalTimeEmbedding(128)
        self.decoder = PeriodicGraphNeuralNetwork(
            node_dim=node_dim + latent_dim,  # 拼接条件
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        # 输出头
        self.coord_pred = nn.Linear(node_dim, 3)
        self.type_pred = nn.Linear(node_dim, 100)  # 100种元素
        self.lattice_pred = nn.Linear(latent_dim, 6)  # 3长度 + 3角度
        
    def encode(
        self,
        atom_types: torch.Tensor,
        frac_coords: torch.Tensor,
        lengths: torch.Tensor,
        angles: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """编码器 - 将晶体映射到潜在空间"""
        batch_size = 1  # 简化处理单个结构
        
        # 时间步为0 (编码时无噪声)
        t = torch.zeros(batch_size, device=atom_types.device)
        
        node_feat, _ = self.encoder(atom_types, frac_coords, lengths, angles, t)
        
        # 全局池化
        graph_feat = node_feat.mean(dim=0)
        
        mu = self.fc_mu(graph_feat)
        logvar = self.fc_logvar(graph_feat)
        
        # 重参数化
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar
    
    def decode(
        self,
        z: torch.Tensor,
        num_atoms: int,
        atom_types_init: Optional[torch.Tensor] = None,
        frac_coords_init: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0
    ) -> CrystalFeatures:
        """解码器 - 从潜在向量生成晶体"""
        device = z.device
        
        # 预测晶胞参数
        lattice_pred = self.lattice_pred(z)
        lengths = torch.sigmoid(lattice_pred[:3]) * 20 + 2  # 2-22 Å
        angles = torch.sigmoid(lattice_pred[3:]) * 60 + 60   # 60-120°
        
        # 初始化原子类型和坐标
        if atom_types_init is None:
            atom_types = torch.randint(0, 100, (num_atoms,), device=device)
        else:
            atom_types = atom_types_init
            
        if frac_coords_init is None:
            frac_coords = torch.rand(num_atoms, 3, device=device)
        else:
            frac_coords = frac_coords_init
        
        # 扩散去噪过程
        for t in reversed(range(self.num_timesteps)):
            timestep = torch.tensor([t], device=device, dtype=torch.float32)
            time_emb = self.time_embedding(timestep)
            
            # 条件向量
            z_expanded = z.unsqueeze(0).expand(num_atoms, -1)
            
            # 预测噪声
            node_feat, pred_coords = self.decoder(
                atom_types, frac_coords, lengths, angles, time_emb
            )
            
            # 去噪更新
            if t > 0:
                noise = torch.randn_like(frac_coords)
                alpha_t = self.alphas[t]
                alpha_cumprod_t = self.alphas_cumprod[t]
                beta_t = self.betas[t]
                
                # 简化DDPM采样
                frac_coords = (frac_coords - beta_t / torch.sqrt(1 - alpha_cumprod_t) * 
                              (frac_coords - pred_coords)) / torch.sqrt(alpha_t)
                frac_coords = frac_coords + torch.sqrt(beta_t) * noise
                
                # 保持在[0,1]范围内
                frac_coords = torch.clamp(frac_coords, 0, 1)
        
        # 最终预测原子类型
        _, final_coords = self.decoder(atom_types, frac_coords, lengths, angles, 
                                       self.time_embedding(torch.zeros(1, device=device)))
        node_feat, _ = self.decoder(atom_types, frac_coords, lengths, angles,
                                    self.time_embedding(torch.zeros(1, device=device)))
        type_logits = self.type_pred(node_feat)
        atom_types = torch.argmax(type_logits, dim=-1)
        
        return CrystalFeatures(
            frac_coords=frac_coords,
            atom_types=atom_types,
            lengths=lengths,
            angles=angles,
            num_atoms=num_atoms
        )
    
    def forward_diffusion(
        self,
        frac_coords: torch.Tensor,
        atom_types: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向扩散过程 - 添加噪声
        """
        noise_coords = torch.randn_like(frac_coords)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        noisy_coords = sqrt_alpha * frac_coords + sqrt_one_minus_alpha * noise_coords
        
        return noisy_coords, noise_coords, atom_types
    
    def compute_loss(
        self,
        crystal: CrystalFeatures,
        properties: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """计算训练损失"""
        # 编码
        z, mu, logvar = self.encode(
            crystal.atom_types,
            crystal.frac_coords,
            crystal.lengths,
            crystal.angles
        )
        
        # KL散度
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 属性预测损失 (如果有标签)
        prop_loss = torch.tensor(0.0)
        if properties is not None:
            pred_props = self.property_predictor(z)
            prop_loss = F.mse_loss(pred_props, properties)
        
        # 扩散重建损失
        t = torch.randint(0, self.num_timesteps, (1,), device=z.device)
        noisy_coords, noise_target, _ = self.forward_diffusion(
            crystal.frac_coords.unsqueeze(0),
            crystal.atom_types.unsqueeze(0),
            t
        )
        
        time_emb = self.time_embedding(t.float())
        z_expanded = z.unsqueeze(0).expand(crystal.num_atoms, -1)
        
        node_feat, pred_coords = self.decoder(
            crystal.atom_types,
            noisy_coords.squeeze(0),
            crystal.lengths,
            crystal.angles,
            time_emb
        )
        
        diffusion_loss = F.mse_loss(pred_coords, crystal.frac_coords)
        
        total_loss = kl_loss + diffusion_loss + 0.1 * prop_loss
        
        return {
            'total_loss': total_loss,
            'kl_loss': kl_loss,
            'diffusion_loss': diffusion_loss,
            'prop_loss': prop_loss
        }


class DiffCSP(nn.Module):
    """
    DiffCSP: Diffusion for Crystal Structure Prediction
    
    2023年ICML发表, 将扩散模型用于晶体结构预测任务
    在给定组分和条件下的结构生成
    """
    
    def __init__(
        self,
        composition_dim: int = 64,
        node_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_timesteps: int = 1000
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        # 组分编码
        self.composition_encoder = nn.Sequential(
            nn.Linear(100, hidden_dim),  # 100种元素
            nn.SiLU(),
            nn.Linear(hidden_dim, composition_dim)
        )
        
        # 分数坐标扩散
        self.coord_diffusion = CoordinateDiffusion(
            node_dim, hidden_dim, num_layers, num_timesteps
        )
        
        # 晶格参数扩散
        self.lattice_diffusion = LatticeDiffusion(
            composition_dim, hidden_dim, num_timesteps
        )
        
        # 原子类型扩散
        self.type_diffusion = TypeDiffusion(
            node_dim, 100, num_timesteps
        )
        
    def forward(
        self,
        composition: Dict[str, float],  # 如 {'Li': 1, 'Fe': 1, 'O': 2}
        num_atoms: int,
        conditions: Optional[Dict] = None
    ) -> CrystalFeatures:
        """生成给定组分的晶体结构"""
        device = next(self.parameters()).device
        
        # 编码组分
        comp_vec = self._encode_composition(composition, device)
        
        # 生成晶格
        lengths, angles = self.lattice_diffusion.generate(comp_vec)
        
        # 生成原子类型分布
        type_probs = self.type_diffusion.generate(comp_vec, num_atoms)
        atom_types = torch.multinomial(type_probs, 1).squeeze(-1)
        
        # 生成分数坐标
        frac_coords = self.coord_diffusion.generate(
            atom_types, lengths, angles, comp_vec, num_atoms
        )
        
        return CrystalFeatures(
            frac_coords=frac_coords,
            atom_types=atom_types,
            lengths=lengths,
            angles=angles,
            num_atoms=num_atoms
        )
    
    def _encode_composition(
        self,
        composition: Dict[str, float],
        device: torch.device
    ) -> torch.Tensor:
        """编码化学组分为向量"""
        # 元素到原子序数映射 (简化)
        elem_to_z = self._get_element_mapping()
        
        comp_vec = torch.zeros(100, device=device)
        total = sum(composition.values())
        
        for elem, count in composition.items():
            if elem in elem_to_z:
                z = elem_to_z[elem]
                comp_vec[z] = count / total
        
        return self.composition_encoder(comp_vec)
    
    def _get_element_mapping(self) -> Dict[str, int]:
        """获取元素到原子序数的映射"""
        elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                   'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                   'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
        return {elem: i for i, elem in enumerate(elements)}


class CoordinateDiffusion(nn.Module):
    """分数坐标扩散模块"""
    
    def __init__(self, node_dim: int, hidden_dim: int, num_layers: int, num_timesteps: int):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        betas = torch.linspace(1e-4, 0.02, num_timesteps)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1 - betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(1 - betas, dim=0))
        
        self.score_network = PeriodicGraphNeuralNetwork(
            node_dim=node_dim + 64,  # +composition
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
    def generate(
        self,
        atom_types: torch.Tensor,
        lengths: torch.Tensor,
        angles: torch.Tensor,
        composition: torch.Tensor,
        num_atoms: int
    ) -> torch.Tensor:
        """生成坐标"""
        device = atom_types.device
        coords = torch.rand(num_atoms, 3, device=device)
        
        for t in reversed(range(self.num_timesteps)):
            timestep = torch.tensor([t], device=device, dtype=torch.float32)
            score = self.score_network(atom_types, coords, lengths, angles, timestep)
            
            if t > 0:
                noise = torch.randn_like(coords)
                coords = coords + self.betas[t] * score + torch.sqrt(self.betas[t]) * noise
            coords = torch.clamp(coords, 0, 1)
        
        return coords


class LatticeDiffusion(nn.Module):
    """晶格参数扩散模块"""
    
    def __init__(self, comp_dim: int, hidden_dim: int, num_timesteps: int):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        self.network = nn.Sequential(
            nn.Linear(comp_dim + 1, hidden_dim),  # +timestep
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6)  # 3 lengths + 3 angles
        )
        
    def generate(self, composition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成晶格参数"""
        device = composition.device
        params = torch.tensor([10.0, 10.0, 10.0, 90.0, 90.0, 90.0], device=device)
        
        for t in range(self.num_timesteps):
            timestep = torch.tensor([t / self.num_timesteps], device=device)
            input_vec = torch.cat([composition, timestep])
            delta = self.network(input_vec)
            params = params + delta * 0.01
        
        lengths = torch.clamp(params[:3], 2, 50)
        angles = torch.clamp(params[3:], 60, 120)
        
        return lengths, angles


class TypeDiffusion(nn.Module):
    """原子类型扩散模块"""
    
    def __init__(self, node_dim: int, num_types: int, num_timesteps: int):
        super().__init__()
        self.network = nn.Linear(node_dim, num_types)
        
    def generate(
        self,
        composition: torch.Tensor,
        num_atoms: int
    ) -> torch.Tensor:
        """生成原子类型概率"""
        logits = self.network(composition)
        probs = F.softmax(logits, dim=-1)
        return probs.unsqueeze(0).expand(num_atoms, -1)


class CrystalGenerator:
    """
    晶体生成器 - 高级API接口
    """
    
    def __init__(
        self,
        model_type: str = 'cdvae',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.model_type = model_type
        
        if model_type == 'cdvae':
            self.model = CDVAE().to(device)
        elif model_type == 'diffcsp':
            self.model = DiffCSP().to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def generate(
        self,
        num_structures: int = 10,
        num_atoms_range: Tuple[int, int] = (5, 50),
        target_property: Optional[Dict[str, float]] = None,
        composition: Optional[Dict[str, float]] = None
    ) -> List[CrystalFeatures]:
        """
        生成晶体结构
        
        Args:
            num_structures: 生成结构数量
            num_atoms_range: 原子数范围
            target_property: 目标属性条件 (如 {'band_gap': 1.5})
            composition: 目标组分 (DiffCSP模式)
        """
        self.model.eval()
        structures = []
        
        with torch.no_grad():
            for _ in range(num_structures):
                num_atoms = np.random.randint(*num_atoms_range)
                
                if self.model_type == 'cdvae':
                    # 从标准正态采样潜在向量
                    z = torch.randn(self.model.latent_dim, device=self.device)
                    struct = self.model.decode(z, num_atoms)
                    
                elif self.model_type == 'diffcsp':
                    struct = self.model.forward(composition or {'Si': 1}, num_atoms)
                
                structures.append(struct)
        
        return structures
    
    def train_step(
        self,
        crystal_batch: List[CrystalFeatures],
        properties: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """单步训练"""
        self.model.train()
        
        # 简化: 处理单个结构
        crystal = crystal_batch[0]
        
        losses = self.model.compute_loss(crystal, properties)
        
        return {k: v.item() for k, v in losses.items()}
    
    def optimize_for_property(
        self,
        target_property: str,
        target_value: float,
        num_steps: int = 100,
        lr: float = 0.01
    ) -> CrystalFeatures:
        """
        针对特定属性优化生成结构
        
        使用潜在空间优化找到满足目标属性的结构
        """
        z = torch.randn(self.model.latent_dim, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([z], lr=lr)
        
        property_idx = {'band_gap': 0, 'energy': 1, 'stability': 2}.get(target_property, 0)
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # 预测属性
            pred_props = self.model.property_predictor(z)
            loss = (pred_props[property_idx] - target_value) ** 2
            
            loss.backward()
            optimizer.step()
            
            if step % 20 == 0:
                print(f"Step {step}: Loss = {loss.item():.4f}, "
                      f"Pred = {pred_props[property_idx].item():.4f}")
        
        # 生成最终结构
        with torch.no_grad():
            struct = self.model.decode(z, num_atoms=20)
        
        return struct


# 辅助函数和工具

def load_pretrained_cdvae(checkpoint_path: str) -> CDVAE:
    """加载预训练CDVAE模型"""
    model = CDVAE()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def evaluate_structure_quality(
    generated: CrystalFeatures,
    reference: Optional[CrystalFeatures] = None
) -> Dict[str, float]:
    """评估生成结构的质量指标"""
    metrics = {}
    
    # 合理性检查
    metrics['valid_coords'] = bool(
        torch.all((generated.frac_coords >= 0) & (generated.frac_coords <= 1))
    )
    
    metrics['valid_lengths'] = bool(
        torch.all((generated.lengths >= 2) & (generated.lengths <= 50))
    )
    
    metrics['valid_angles'] = bool(
        torch.all((generated.angles >= 60) & (generated.angles <= 120))
    )
    
    # 原子间距检查
    lattice = torch.eye(3) * generated.lengths
    cart_coords = generated.frac_coords @ lattice
    
    # 计算最小原子间距
    dists = torch.cdist(cart_coords, cart_coords)
    dists.fill_diagonal_(float('inf'))
    min_dist = dists.min().item()
    metrics['min_atomic_distance'] = min_dist
    metrics['no_overlap'] = min_dist > 0.5  # 0.5 Å阈值
    
    # 如果有参考结构,计算RMSD
    if reference is not None:
        rmsd = torch.norm(
            generated.frac_coords - reference.frac_coords, dim=-1
        ).mean().item()
        metrics['rmsd'] = rmsd
    
    return metrics


def batch_generate_diverse_structures(
    generator: CrystalGenerator,
    num_batches: int = 10,
    diversity_weight: float = 0.5
) -> List[CrystalFeatures]:
    """
    批量生成多样化结构
    
    使用多样性促进采样避免生成相似结构
    """
    all_structures = []
    
    for batch in range(num_batches):
        # 生成结构
        structs = generator.generate(num_structures=5)
        
        # 多样性过滤 (简化实现)
        for s in structs:
            is_diverse = True
            for existing in all_structures:
                sim = compute_structure_similarity(s, existing)
                if sim > 0.9:  # 相似度阈值
                    is_diverse = False
                    break
            
            if is_diverse:
                all_structures.append(s)
    
    return all_structures


def compute_structure_similarity(
    s1: CrystalFeatures,
    s2: CrystalFeatures
) -> float:
    """计算两个结构的相似度 (简化版)"""
    # 基于组分相似度
    types1 = set(s1.atom_types.tolist())
    types2 = set(s2.atom_types.tolist())
    
    if len(types1.union(types2)) == 0:
        return 1.0
    
    jaccard = len(types1.intersection(types2)) / len(types1.union(types2))
    return jaccard


if __name__ == "__main__":
    # 演示: CDVAE晶体生成
    print("=" * 60)
    print("CDVAE Crystal Generation Demo")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 初始化生成器
    generator = CrystalGenerator(model_type='cdvae', device=device)
    print(f"Model: {generator.model_type}")
    print(f"Parameters: {sum(p.numel() for p in generator.model.parameters()):,}")
    
    # 生成结构
    print("\nGenerating 5 crystal structures...")
    structures = generator.generate(num_structures=5, num_atoms_range=(5, 20))
    
    for i, struct in enumerate(structures):
        print(f"\nStructure {i+1}:")
        print(f"  Atoms: {struct.num_atoms}")
        print(f"  Cell lengths: {struct.lengths.detach().cpu().numpy()}")
        print(f"  Cell angles: {struct.angles.detach().cpu().numpy()}")
        
        # 评估质量
        metrics = evaluate_structure_quality(struct)
        print(f"  Quality: {metrics}")
    
    # 属性优化示例
    print("\n" + "=" * 60)
    print("Property Optimization Demo (Target: band_gap = 1.5 eV)")
    print("=" * 60)
    
    optimized = generator.optimize_for_property('band_gap', 1.5, num_steps=50)
    print(f"Optimized structure generated with {optimized.num_atoms} atoms")
    
    print("\nDemo completed successfully!")
