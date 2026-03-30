"""
flow_matching.py
流匹配模型 - 比扩散模型更快的生成方法

流匹配(Flow Matching)是2022-2023年提出的生成模型新范式,
已在晶体生成、分子生成等任务中展现出比扩散模型更快的收敛速度和更好的生成质量。

References:
- Lipman et al. (2023) "Flow Matching for Generative Modeling"
- Liu et al. (2022) "Flow Straight and Fast: Learning to Generate and Transfer Data"
- 2024进展: Riemannian Flow Matching用于晶体生成 (在球面和SO(3)流形上)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment


@dataclass
class FlowSample:
    """流样本 - 包含起点、终点和时间"""
    x0: torch.Tensor  # 初始噪声/简单分布
    x1: torch.Tensor  # 目标数据
    t: torch.Tensor   # 时间 [0, 1]
    xt: torch.Tensor  # 中间点
    ut: torch.Tensor  # 目标速度


class RiemannianFlowMatching(nn.Module):
    """
    黎曼流形上的流匹配
    
    处理晶体结构中的约束流形:
    - 分数坐标: 环面T^3 (周期性边界)
    - 晶格矩阵: SPD流形 (对称正定矩阵)
    - 原子类型: 单纯形/分类分布
    """
    
    def __init__(
        self,
        manifold_type: str = 'torus',  # 'torus', 'spd', 'simplex'
        ambient_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 4
    ):
        super().__init__()
        self.manifold_type = manifold_type
        self.ambient_dim = ambient_dim
        
        # 速度场网络
        self.velocity_net = VelocityNetwork(
            ambient_dim=ambient_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
    def geodesic_interpolant(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        计算测地线插值 xt = geodesic(x0, x1, t)
        
        不同流形有不同插值方式
        """
        if self.manifold_type == 'torus':
            # 环面: 最短路径插值 (考虑周期性)
            return self._torus_geodesic(x0, x1, t)
        elif self.manifold_type == 'spd':
            # SPD流形: 对数欧氏度量
            return self._spd_geodesic(x0, x1, t)
        elif self.manifold_type == 'simplex':
            # 单纯形: 测地线保持归一化
            return self._simplex_geodesic(x0, x1, t)
        else:
            # 欧氏空间
            return (1 - t) * x0 + t * x1
    
    def _torus_geodesic(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """环面测地线 - 最短周期路径"""
        # 计算最短差值 (考虑周期性)
        delta = x1 - x0
        delta = (delta + 0.5) % 1.0 - 0.5  # 映射到[-0.5, 0.5]
        
        # 插值
        xt = (x0 + t * delta) % 1.0
        return xt
    
    def _spd_geodesic(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """SPD流形测地线"""
        # x0, x1是晶格向量矩阵 [3, 3]
        # 使用矩阵对数插值
        
        # 简化: 使用Cholesky分解
        L0 = torch.linalg.cholesky(x0 @ x0.T + torch.eye(3, device=x0.device) * 1e-6)
        L1 = torch.linalg.cholesky(x1 @ x1.T + torch.eye(3, device=x1.device) * 1e-6)
        
        # 在对数空间插值
        Lt = (1 - t) * L0 + t * L1
        xt = Lt @ Lt.T
        
        return xt
    
    def _simplex_geodesic(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """单纯形测地线 - 保持概率归一化"""
        # 使用对数-线性插值
        log_x0 = torch.log(x0 + 1e-10)
        log_x1 = torch.log(x1 + 1e-10)
        
        log_xt = (1 - t) * log_x0 + t * log_x1
        xt = F.softmax(log_xt, dim=-1)
        
        return xt
    
    def compute_target_velocity(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        计算目标速度 ut = d/dt geodesic(x0, x1, t)
        
        这是流匹配的核心 - 回归这个速度
        """
        if self.manifold_type == 'torus':
            # 环面上速度是常数
            delta = x1 - x0
            delta = (delta + 0.5) % 1.0 - 0.5
            return delta
        elif self.manifold_type == 'simplex':
            # 单纯形上需要投影到切空间
            log_x0 = torch.log(x0 + 1e-10)
            log_x1 = torch.log(x1 + 1e-10)
            velocity = log_x1 - log_x0
            # 投影到切空间: sum(v * x) = 0
            velocity = velocity - (velocity * xt).sum(dim=-1, keepdim=True)
            return velocity
        else:
            # 欧氏空间
            return x1 - x0
    
    def sample_flow_pair(
        self,
        batch_size: int,
        data_sampler: Callable[[], torch.Tensor]
    ) -> FlowSample:
        """
        采样流对 (x0, x1, t) -> (xt, ut)
        
        这是流匹配训练的关键步骤
        """
        device = next(self.parameters()).device
        
        # 采样数据点
        x1 = data_sampler()
        if isinstance(x1, tuple):
            x1 = x1[0]
        
        # 采样先验噪声
        if self.manifold_type == 'torus':
            x0 = torch.rand_like(x1)
        elif self.manifold_type == 'simplex':
            # Dirichlet先验
            x0 = torch.distributions.Dirichlet(torch.ones_like(x1)).sample()
        else:
            x0 = torch.randn_like(x1)
        
        # 采样时间
        t = torch.rand(batch_size, device=device)
        
        # 计算插值点
        xt = self.geodesic_interpolant(x0, x1, t.unsqueeze(-1))
        
        # 计算目标速度
        ut = self.compute_target_velocity(x0, x1, xt, t.unsqueeze(-1))
        
        return FlowSample(x0=x0, x1=x1, t=t, xt=xt, ut=ut)
    
    def forward(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """预测速度场 v_theta(xt, t)"""
        return self.velocity_net(xt, t, context)
    
    def compute_loss(
        self,
        flow_sample: FlowSample
    ) -> torch.Tensor:
        """
        计算流匹配损失
        
        L = || v_theta(xt, t) - ut ||^2
        """
        pred_v = self.forward(flow_sample.xt, flow_sample.t)
        target_v = flow_sample.ut
        
        loss = F.mse_loss(pred_v, target_v)
        return loss
    
    def integrate(
        self,
        x0: torch.Tensor,
        num_steps: int = 50,
        method: str = 'euler'
    ) -> torch.Tensor:
        """
        数值积分从x0到x1
        
        dx/dt = v_theta(xt, t)
        """
        xt = x0.clone()
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.tensor([i * dt], device=x0.device)
            vt = self.forward(xt, t)
            
            if method == 'euler':
                xt = xt + dt * vt
            elif method == 'rk4':
                # 简化RK4
                k1 = vt
                k2 = self.forward(xt + 0.5 * dt * k1, t + 0.5 * dt)
                k3 = self.forward(xt + 0.5 * dt * k2, t + 0.5 * dt)
                k4 = self.forward(xt + dt * k3, t + dt)
                xt = xt + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            
            # 投影回流形
            xt = self.project_to_manifold(xt)
        
        return xt
    
    def project_to_manifold(self, x: torch.Tensor) -> torch.Tensor:
        """投影回流形约束"""
        if self.manifold_type == 'torus':
            return x % 1.0
        elif self.manifold_type == 'simplex':
            x = torch.clamp(x, 1e-10, 1.0)
            return x / x.sum(dim=-1, keepdim=True)
        return x


class VelocityNetwork(nn.Module):
    """速度场网络 - 预测v_theta(x, t)"""
    
    def __init__(
        self,
        ambient_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 4,
        condition_dim: int = 0
    ):
        super().__init__()
        
        # 时间嵌入
        self.time_embed = SinusoidalEmbedding(128)
        
        # 输入层
        input_dim = ambient_dim + 128 + condition_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 隐藏层
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output = nn.Linear(hidden_dim, ambient_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, dim] 空间坐标
            t: [batch] 时间
            context: [batch, cond_dim] 条件信息
        """
        # 时间嵌入
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        t_emb = self.time_embed(t)
        
        # 拼接输入
        inputs = [x, t_emb]
        if context is not None:
            inputs.append(context)
        h = torch.cat(inputs, dim=-1)
        
        h = self.input_proj(h)
        
        # 残差连接
        for layer in self.layers:
            h = h + layer(h)
        
        return self.output(h)


class SinusoidalEmbedding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class CrystalFlowMatching(nn.Module):
    """
    晶体流匹配生成器
    
    同时处理:
    1. 分数坐标 (环面T^3)
    2. 晶格参数 (欧氏空间)
    3. 原子类型 (单纯形)
    """
    
    def __init__(
        self,
        coord_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_atom_types: int = 100
    ):
        super().__init__()
        
        # 坐标流 (环面)
        self.coord_flow = RiemannianFlowMatching(
            manifold_type='torus',
            ambient_dim=coord_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        # 晶格流 (欧氏空间)
        self.lattice_flow = RiemannianFlowMatching(
            manifold_type='euclidean',
            ambient_dim=6,  # 3 lengths + 3 angles
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        # 原子类型流 (单纯形)
        self.type_flow = RiemannianFlowMatching(
            manifold_type='simplex',
            ambient_dim=num_atom_types,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        # 条件编码器 (用于属性引导生成)
        self.condition_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),  # band_gap, energy, stability
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(
        self,
        coords: torch.Tensor,
        lattice: torch.Tensor,
        atom_types: torch.Tensor,
        t: torch.Tensor,
        conditions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        预测速度场
        
        Returns:
            coord_v, lattice_v, type_v
        """
        # 编码条件
        context = None
        if conditions is not None:
            context = self.condition_encoder(conditions)
        
        coord_v = self.coord_flow(coords, t, context)
        lattice_v = self.lattice_flow(lattice, t, context)
        type_v = self.type_flow(atom_types, t, context)
        
        return coord_v, lattice_v, type_v
    
    def sample(self, num_atoms: int, num_steps: int = 50) -> Dict[str, torch.Tensor]:
        """采样生成晶体结构"""
        device = next(self.parameters()).device
        
        # 从先验采样
        x0_coords = torch.rand(num_atoms, 3, device=device)
        x0_lattice = torch.randn(6, device=device)
        x0_types = torch.distributions.Dirichlet(torch.ones(100, device=device)).sample((num_atoms,))
        
        # 积分生成
        coords = self.coord_flow.integrate(x0_coords, num_steps)
        lattice = self.lattice_flow.integrate(x0_lattice, num_steps)
        types = self.type_flow.integrate(x0_types, num_steps)
        
        # 解析晶格参数
        lengths = torch.sigmoid(lattice[:3]) * 20 + 2  # 2-22 Å
        angles = torch.sigmoid(lattice[3:]) * 60 + 60   # 60-120°
        
        # 原子类型
        atom_types = torch.argmax(types, dim=-1)
        
        return {
            'frac_coords': coords,
            'lengths': lengths,
            'angles': angles,
            'atom_types': atom_types
        }


class ConditionalFlowMatching(nn.Module):
    """
    条件流匹配 - 支持属性引导生成
    
    实现Classifier-Free Guidance (CFG)用于可控生成
    """
    
    def __init__(
        self,
        base_flow: RiemannianFlowMatching,
        condition_dim: int = 64,
        dropout_prob: float = 0.1
    ):
        super().__init__()
        self.base_flow = base_flow
        self.condition_dim = condition_dim
        self.dropout_prob = dropout_prob
        
        # 条件投影
        self.condition_proj = nn.Linear(condition_dim, condition_dim)
        
    def forward(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """
        条件前向传播, 支持Classifier-Free Guidance
        
        v_cfg = v_uncond + scale * (v_cond - v_uncond)
        """
        if condition is None or guidance_scale == 0:
            # 无条件生成
            return self.base_flow(xt, t, None)
        
        # 条件嵌入
        cond_emb = self.condition_proj(condition)
        
        # 条件预测
        v_cond = self.base_flow(xt, t, cond_emb)
        
        if guidance_scale == 1.0:
            return v_cond
        
        # 无条件预测
        v_uncond = self.base_flow(xt, t, None)
        
        # CFG组合
        v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)
        
        return v_cfg
    
    def training_forward(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """训练前向 - 随机丢弃条件实现CFG"""
        if torch.rand(1).item() < self.dropout_prob:
            # 无条件训练
            return self.base_flow(xt, t, None)
        else:
            cond_emb = self.condition_proj(condition)
            return self.base_flow(xt, t, cond_emb)


class OTFlowMatching(nn.Module):
    """
    最优输运流匹配 (Optimal Transport Flow Matching)
    
    使用最优输运耦合代替独立耦合, 使轨迹更直, 采样更快
    """
    
    def __init__(self, base_flow: RiemannianFlowMatching):
        super().__init__()
        self.base_flow = base_flow
        
    def sample_coupled_pair(
        self,
        x0_batch: torch.Tensor,
        x1_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样最优输运耦合对
        
        使用Sinkhorn算法或近似最近邻匹配
        """
        batch_size = x0_batch.shape[0]
        
        # 计算成本矩阵 (简化: 欧氏距离)
        cost_matrix = torch.cdist(x0_batch, x1_batch, p=2)
        
        # 使用匈牙利算法找到最优匹配
        cost_np = cost_matrix.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)
        
        # 重排x1以匹配x0
        x1_matched = x1_batch[col_ind]
        
        return x0_batch, x1_matched
    
    def compute_straightness(self, trajectories: List[torch.Tensor]) -> float:
        """
        计算轨迹直线度
        
        直线度 = ||x1 - x0|| / ∫||dx/dt|| dt
        值越接近1, 轨迹越直
        """
        straightness_scores = []
        
        for traj in trajectories:
            x0, x1 = traj[0], traj[-1]
            direct_dist = torch.norm(x1 - x0)
            
            # 路径长度
            path_length = sum(
                torch.norm(traj[i+1] - traj[i])
                for i in range(len(traj) - 1)
            )
            
            straightness = direct_dist / (path_length + 1e-8)
            straightness_scores.append(straightness.item())
        
        return np.mean(straightness_scores)


class MultiFidelityFlowMatching(nn.Module):
    """
    多保真度流匹配
    
    结合不同精度数据 (DFT + 力场) 训练流模型
    2024年最新进展: 用于加速材料发现
    """
    
    def __init__(
        self,
        base_flow: RiemannianFlowMatching,
        num_fidelities: int = 2
    ):
        super().__init__()
        self.base_flow = base_flow
        self.num_fidelities = num_fidelities
        
        # 保真度嵌入
        self.fidelity_embed = nn.Embedding(num_fidelities, 32)
        
        # 每个保真度的输出头
        self.fidelity_heads = nn.ModuleList([
            nn.Linear(256, 256) for _ in range(num_fidelities)
        ])
        
    def forward(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        fidelity: int = 0
    ) -> torch.Tensor:
        """前向传播, 考虑保真度"""
        # 基础速度
        v_base = self.base_flow(xt, t)
        
        # 保真度调制
        fid_emb = self.fidelity_embed(torch.tensor(fidelity, device=xt.device))
        
        # 应用保真度特定的调整
        adjustment = self.fidelity_heads[fidelity](fid_emb.unsqueeze(0))
        v_adjusted = v_base + 0.1 * adjustment
        
        return v_adjusted
    
    def knowledge_distillation_loss(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        知识蒸馏损失 - 使低保真度预测接近高保真度
        """
        v_high = self.forward(x, t, fidelity=self.num_fidelities - 1)
        v_low = self.forward(x, t, fidelity=0)
        
        loss = F.mse_loss(v_low, v_high.detach())
        return loss


def compare_flow_vs_diffusion(
    flow_model: RiemannianFlowMatching,
    diffusion_model: nn.Module,
    test_samples: int = 100,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    对比流匹配与扩散模型的性能
    """
    results = {}
    
    # 生成速度对比
    import time
    
    # Flow Matching生成
    x0 = torch.randn(test_samples, 3, device=device)
    
    start = time.time()
    with torch.no_grad():
        _ = flow_model.integrate(x0, num_steps=50)
    flow_time = time.time() - start
    
    # 扩散模型生成 (假设50步)
    start = time.time()
    with torch.no_grad():
        for _ in range(50):
            _ = diffusion_model(x0, torch.zeros(1, device=device))
    diffusion_time = time.time() - start
    
    results['flow_time'] = flow_time
    results['diffusion_time'] = diffusion_time
    results['speedup'] = diffusion_time / flow_time
    
    # 采样质量对比 (使用MMD等度量)
    # ...
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Flow Matching for Crystal Generation Demo")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 创建流匹配模型
    print("\n1. Testing Riemannian Flow Matching on Torus")
    flow = RiemannianFlowMatching(
        manifold_type='torus',
        ambient_dim=3,
        hidden_dim=128,
        num_layers=3
    ).to(device)
    
    # 测试测地线
    x0 = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    x1 = torch.tensor([[0.9, 0.1, 0.5]], device=device)
    t = torch.tensor([[0.5]], device=device)
    
    xt = flow.geodesic_interpolant(x0, x1, t)
    print(f"Geodesic interpolation: {x0.cpu().numpy()} -> {xt.cpu().numpy()} -> {x1.cpu().numpy()}")
    
    # 测试训练
    print("\n2. Training simulation")
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
    
    for step in range(100):
        # 模拟数据采样
        def data_sampler():
            return torch.rand(16, 3, device=device)
        
        sample = flow.sample_flow_pair(16, data_sampler)
        loss = flow.compute_loss(sample)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f"Step {step}: Loss = {loss.item():.6f}")
    
    # 测试生成
    print("\n3. Generation test")
    x0 = torch.rand(5, 3, device=device)
    x1_pred = flow.integrate(x0, num_steps=50)
    print(f"Generated {x1_pred.shape[0]} samples")
    print(f"Sample range: [{x1_pred.min():.3f}, {x1_pred.max():.3f}]")
    
    # 晶体流匹配
    print("\n4. Crystal Flow Matching")
    crystal_flow = CrystalFlowMatching(
        coord_dim=3,
        hidden_dim=128,
        num_layers=3
    ).to(device)
    
    result = crystal_flow.sample(num_atoms=10, num_steps=20)
    print(f"Generated crystal with {result['atom_types'].shape[0]} atoms")
    print(f"Cell lengths: {result['lengths'].cpu().numpy()}")
    print(f"Cell angles: {result['angles'].cpu().numpy()}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("Flow Matching advantages:")
    print("- Straight trajectories (faster sampling)")
    print("- No need for noise scheduling")
    print("- Better theoretical properties")
    print("=" * 60)
