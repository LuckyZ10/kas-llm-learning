"""
neural_operators.py
神经算子 (Neural Operators)

神经算子学习函数空间之间的映射, 用于多尺度材料模拟。
包括FNO (Fourier Neural Operator) 和 DeepONet等。

References:
- Li et al. (2021) "Fourier Neural Operator for Parametric Partial Differential Equations"
- Lu et al. (2021) "Learning nonlinear operators via DeepONet"
- 2024进展: 神经算子用于材料多尺度建模和晶体塑性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
import numpy as np
import math


@dataclass
class OperatorData:
    """算子学习数据结构"""
    input_function: torch.Tensor   # 输入函数 (如初始条件、材料参数场)
    output_function: torch.Tensor  # 输出函数 (如解场、应力场)
    coordinates: torch.Tensor      # 空间坐标
    parameters: Optional[Dict] = None  # 额外参数


class SpectralConv2d(nn.Module):
    """
    2D谱卷积层 - FNO核心组件
    
    在傅里叶空间进行卷积
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int = 12,
        modes2: int = 12
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # 傅里叶模式数 (x方向)
        self.modes2 = modes2  # 傅里叶模式数 (y方向)
        
        # 可学习的复数权重
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, 2)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, 2)
        )
    
    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        复数矩阵乘法
        
        (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        """
        # 将复数分解为实部和虚部
        input_real, input_imag = input[..., 0], input[..., 1]
        weights_real, weights_imag = weights[..., 0], weights[..., 1]
        
        # 复数乘法: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        output_real = (torch.einsum("bixy,ioxy->boxy", input_real, weights_real) -
                      torch.einsum("bixy,ioxy->boxy", input_imag, weights_imag))
        output_imag = (torch.einsum("bixy,ioxy->boxy", input_real, weights_imag) +
                      torch.einsum("bixy,ioxy->boxy", input_imag, weights_real))
        
        return torch.stack([output_real, output_imag], dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch, channels, height, width]
        """
        batchsize = x.shape[0]
        
        # FFT到傅里叶空间
        x_ft = torch.fft.rfft2(x)
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)
        
        # 初始化输出
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, 2,
            device=x.device, dtype=torch.float32
        )
        
        # 低频率模式相乘
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )
        
        # 回到物理空间
        out_ft_complex = torch.complex(out_ft[..., 0], out_ft[..., 1])
        x = torch.fft.irfft2(out_ft_complex, s=(x.size(-2), x.size(-1)))
        
        return x


class SpectralConv3d(nn.Module):
    """3D谱卷积层"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int = 8,
        modes2: int = 8,
        modes3: int = 8
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, 2)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, 2)
        )
    
    def compl_mul3d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """3D复数乘法"""
        input_real, input_imag = input[..., 0], input[..., 1]
        weights_real, weights_imag = weights[..., 0], weights[..., 1]
        
        output_real = (torch.einsum("bixyz,ioxyz->boxyz", input_real, weights_real) -
                      torch.einsum("bixyz,ioxyz->boxyz", input_imag, weights_imag))
        output_imag = (torch.einsum("bixyz,ioxyz->boxyz", input_real, weights_imag) +
                      torch.einsum("bixyz,ioxyz->boxyz", input_imag, weights_real))
        
        return torch.stack([output_real, output_imag], dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)
        
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1, 2,
            device=x.device
        )
        
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = self.compl_mul3d(
            x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1
        )
        
        out_ft_complex = torch.complex(out_ft[..., 0], out_ft[..., 1])
        x = torch.fft.irfftn(out_ft_complex, s=(x.size(-3), x.size(-2), x.size(-1)))
        
        return x


class FNO2d(nn.Module):
    """
    2D傅里叶神经算子
    
    用于学习2D场到场的映射
    应用: 多孔介质流动、复合材料应力分析
    """
    
    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        in_channels: int = 1,
        out_channels: int = 1,
        n_layers: int = 4
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        
        # 输入投影
        self.fc0 = nn.Linear(in_channels + 2, width)  # +2 for positional encoding
        
        # FNO层
        self.convs = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2)
            for _ in range(n_layers)
        ])
        
        self.ws = nn.ModuleList([
            nn.Conv2d(width, width, 1)
            for _ in range(n_layers)
        ])
        
        # 输出投影
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch, in_channels, height, width]
        Returns:
            [batch, out_channels, height, width]
        """
        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]
        
        # 添加位置编码
        gridx = torch.linspace(0, 1, size_x, device=x.device).reshape(1, 1, size_x, 1).repeat(batchsize, 1, 1, size_y)
        gridy = torch.linspace(0, 1, size_y, device=x.device).reshape(1, 1, 1, size_y).repeat(batchsize, 1, size_x, 1)
        
        x = torch.cat([x, gridx, gridy], dim=1)  # [batch, in_channels+2, H, W]
        x = x.permute(0, 2, 3, 1)  # [batch, H, W, in_channels+2]
        
        # 输入投影
        x = self.fc0(x)  # [batch, H, W, width]
        x = x.permute(0, 3, 1, 2)  # [batch, width, H, W]
        
        # FNO层
        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)
        
        # 输出投影
        x = x.permute(0, 2, 3, 1)  # [batch, H, W, width]
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)  # [batch, out_channels, H, W]
        
        return x


class FNO3d(nn.Module):
    """3D傅里叶神经算子"""
    
    def __init__(
        self,
        modes1: int = 8,
        modes2: int = 8,
        modes3: int = 8,
        width: int = 20,
        in_channels: int = 1,
        out_channels: int = 1,
        n_layers: int = 4
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        
        self.fc0 = nn.Linear(in_channels + 3, width)
        
        self.convs = nn.ModuleList([
            SpectralConv3d(width, width, modes1, modes2, modes3)
            for _ in range(n_layers)
        ])
        
        self.ws = nn.ModuleList([
            nn.Conv3d(width, width, 1)
            for _ in range(n_layers)
        ])
        
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, in_channels, D, H, W]
        """
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[2], x.shape[3], x.shape[4]
        
        # 位置编码
        gridx = torch.linspace(0, 1, size_x, device=x.device).reshape(1, 1, size_x, 1, 1).repeat(batchsize, 1, 1, size_y, size_z)
        gridy = torch.linspace(0, 1, size_y, device=x.device).reshape(1, 1, 1, size_y, 1).repeat(batchsize, 1, size_x, 1, size_z)
        gridz = torch.linspace(0, 1, size_z, device=x.device).reshape(1, 1, 1, 1, size_z).repeat(batchsize, 1, size_x, size_y, 1)
        
        x = torch.cat([x, gridx, gridy, gridz], dim=1)
        x = x.permute(0, 2, 3, 4, 1)
        
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        
        for i in range(len(self.convs)):
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2
            if i < len(self.convs) - 1:
                x = F.gelu(x)
        
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 4, 1, 2, 3)
        
        return x


class BranchNet(nn.Module):
    """DeepONet分支网络"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 4
    ):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrunkNet(nn.Module):
    """DeepONet主干网络"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 4
    ):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)


class DeepONet(nn.Module):
    """
    DeepONet神经算子
    
    学习非线性算子的通用框架
    G(u)(y) = branch(u) · trunk(y)
    
    应用: 材料响应预测、多尺度建模
    """
    
    def __init__(
        self,
        input_function_dim: int,
        evaluation_point_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_branch_layers: int = 4,
        num_trunk_layers: int = 4
    ):
        super().__init__()
        
        self.output_dim = output_dim
        
        # 分支网络: 编码输入函数
        self.branch = BranchNet(
            input_function_dim, hidden_dim, hidden_dim, num_branch_layers
        )
        
        # 主干网络: 编码评估点
        self.trunk = TrunkNet(
            evaluation_point_dim, hidden_dim, hidden_dim, num_trunk_layers
        )
        
        # 偏置项
        self.bias = nn.Parameter(torch.zeros(output_dim))
    
    def forward(
        self,
        input_function: torch.Tensor,
        evaluation_points: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_function: [batch, input_function_dim] 输入函数 (如材料参数)
            evaluation_points: [n_points, evaluation_point_dim] 评估点坐标
        Returns:
            [batch, n_points, output_dim]
        """
        # 分支网络编码
        branch_out = self.branch(input_function)  # [batch, hidden_dim]
        
        # 主干网络编码
        trunk_out = self.trunk(evaluation_points)  # [n_points, hidden_dim]
        
        # 点积: [batch, hidden] @ [hidden, n_points] = [batch, n_points]
        output = torch.einsum('bh,nh->bn', branch_out, trunk_out)
        
        # 添加偏置
        output = output + self.bias
        
        return output.unsqueeze(-1)  # [batch, n_points, output_dim]


class GNO(nn.Module):
    """
    图神经算子 (Graph Neural Operator)
    
    结合GNN和神经算子, 处理非结构化网格
    """
    
    def __init__(
        self,
        node_features: int = 3,
        edge_features: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 4
    ):
        super().__init__()
        
        self.node_encoder = nn.Linear(node_features, hidden_dim)
        self.edge_encoder = nn.Linear(edge_features, hidden_dim)
        
        self.layers = nn.ModuleList([
            GNOLayer(hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.decoder = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        node_pos: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            node_pos: [n_nodes, 3] 节点位置
            node_features: [n_nodes, node_feat_dim] 节点特征
            edge_index: [2, n_edges] 边索引
        """
        # 计算边特征
        src, dst = edge_index
        edge_vectors = node_pos[dst] - node_pos[src]
        edge_lengths = torch.norm(edge_vectors, dim=-1, keepdim=True)
        
        # 编码
        h = self.node_encoder(node_features)
        e = self.edge_encoder(edge_vectors)
        
        # 消息传递
        for layer in self.layers:
            h = layer(h, e, edge_index)
        
        return self.decoder(h)


class GNOLayer(nn.Module):
    """GNO层"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        src, dst = edge_index
        
        # 边更新
        edge_input = torch.cat([
            node_features[src],
            node_features[dst],
            edge_features
        ], dim=-1)
        edge_messages = self.edge_mlp(edge_input)
        
        # 聚合
        aggregated = torch.zeros_like(node_features)
        aggregated.index_add_(0, dst, edge_messages)
        
        # 节点更新
        node_input = torch.cat([node_features, aggregated], dim=-1)
        return self.node_mlp(node_input)


class MultiscaleFNO(nn.Module):
    """
    多尺度FNO
    
    使用不同尺度的傅里叶模式捕获多尺度特征
    用于材料多尺度建模
    """
    
    def __init__(
        self,
        modes_list: List[Tuple[int, int]] = [(4, 4), (8, 8), (16, 16)],
        width: int = 32,
        in_channels: int = 1,
        out_channels: int = 1
    ):
        super().__init__()
        
        self.scales = nn.ModuleList([
            FNO2d(modes[0], modes[1], width // len(modes_list), in_channels, width // len(modes_list))
            for modes in modes_list
        ])
        
        self.fusion = nn.Sequential(
            nn.Conv2d(width, width, 1),
            nn.GELU(),
            nn.Conv2d(width, out_channels, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """多尺度前向传播"""
        outputs = []
        
        for scale_net in self.scales:
            out = scale_net(x)
            outputs.append(out)
        
        # 融合不同尺度
        fused = torch.cat(outputs, dim=1)
        return self.fusion(fused)


def train_neural_operator(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int = 100,
    lr: float = 1e-3,
    device: str = 'cuda'
) -> Dict[str, List[float]]:
    """训练神经算子"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            x = batch['input'].to(device)
            y = batch['output'].to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['input'].to(device)
                y = batch['output'].to(device)
                
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train = {train_loss:.6f}, Val = {val_loss:.6f}")
    
    return history


if __name__ == "__main__":
    print("=" * 60)
    print("Neural Operators Demo")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 1. FNO2d测试
    print("\n1. FNO2d - Fourier Neural Operator")
    print("-" * 40)
    
    fno2d = FNO2d(
        modes1=12,
        modes2=12,
        width=32,
        in_channels=1,
        out_channels=1,
        n_layers=4
    ).to(device)
    
    # 测试输入 (如材料参数场)
    x = torch.randn(2, 1, 64, 64, device=device)
    
    with torch.no_grad():
        y = fno2d(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in fno2d.parameters()):,}")
    
    # 2. DeepONet测试
    print("\n2. DeepONet")
    print("-" * 40)
    
    deeponet = DeepONet(
        input_function_dim=100,
        evaluation_point_dim=2,
        hidden_dim=128,
        output_dim=1
    ).to(device)
    
    # 输入函数 (如初始条件)
    u = torch.randn(4, 100, device=device)
    # 评估点 (空间坐标)
    y = torch.rand(50, 2, device=device)
    
    with torch.no_grad():
        output = deeponet(u, y)
    
    print(f"Input function shape: {u.shape}")
    print(f"Evaluation points shape: {y.shape}")
    print(f"Output shape: {output.shape}")
    
    # 3. 多尺度FNO测试
    print("\n3. Multiscale FNO")
    print("-" * 40)
    
    ms_fno = MultiscaleFNO(
        modes_list=[(4, 4), (8, 8), (16, 16)],
        width=64,
        in_channels=1,
        out_channels=1
    ).to(device)
    
    x_ms = torch.randn(2, 1, 64, 64, device=device)
    with torch.no_grad():
        y_ms = ms_fno(x_ms)
    
    print(f"Input shape: {x_ms.shape}")
    print(f"Output shape: {y_ms.shape}")
    
    # 4. 速度对比
    print("\n4. Performance Comparison")
    print("-" * 40)
    
    import time
    
    # FNO推理时间
    x_test = torch.randn(8, 1, 128, 128, device=device)
    
    fno_large = FNO2d(modes1=16, modes2=16, width=64).to(device)
    fno_large.eval()
    
    # 预热
    with torch.no_grad():
        _ = fno_large(x_test)
    
    # 计时
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    
    with torch.no_grad():
        for _ in range(10):
            _ = fno_large(x_test)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    fno_time = (time.time() - start) / 10
    
    print(f"FNO inference time: {fno_time*1000:.2f} ms")
    print(f"Throughput: {x_test.shape[0] / fno_time:.1f} samples/sec")
    
    print("\n" + "=" * 60)
    print("Neural Operators Demo completed!")
    print("Key features:")
    print("- Fourier Neural Operator (FNO) for fast inference")
    print("- DeepONet for operator learning")
    print("- Multiscale modeling capabilities")
    print("- Resolution-invariant predictions")
    print("=" * 60)
