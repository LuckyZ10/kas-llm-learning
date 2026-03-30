"""
Fourier Neural Operator (FNO) Implementation

FNO learns solution operators in Fourier space, enabling
resolution-invariant solutions for PDEs.

Reference:
    Li et al., "Fourier Neural Operator for Parametric Partial 
    Differential Equations", ICLR 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import numpy as np
import math


class SpectralConv2d(nn.Module):
    """
    2D Spectral convolution layer in Fourier space.
    
    Performs convolution in Fourier space:
    (K * u)(x) = F^-1(R · F(u))
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int
    ):
        """
        Initialize spectral convolution.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes1: Number of Fourier modes in first dimension
            modes2: Number of Fourier modes in second dimension
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        # Complex weights for Fourier modes
        self.scale = 1.0 / (in_channels * out_channels)
        
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, 2)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, 2)
        )
        
    def compl_mul2d(
        self,
        input: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Complex multiplication in Fourier space.
        
        Args:
            input: [batch, in_channels, x, y, 2] (real and imaginary)
            weights: [in_channels, out_channels, x, y, 2]
            
        Returns:
            Output [batch, out_channels, x, y, 2]
        """
        # input: (batch, in_channel, x, y, 2)
        # weights: (in_channel, out_channel, x, y, 2)
        
        # Complex multiplication
        # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        
        # Real part
        real = (
            torch.einsum('bixy,ioxy->boxy', input[..., 0], weights[..., 0]) -
            torch.einsum('bixy,ioxy->boxy', input[..., 1], weights[..., 1])
        )
        
        # Imaginary part
        imag = (
            torch.einsum('bixy,ioxy->boxy', input[..., 0], weights[..., 1]) +
            torch.einsum('bixy,ioxy->boxy', input[..., 1], weights[..., 0])
        )
        
        return torch.stack([real, imag], dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [batch, in_channels, size_x, size_y]
            
        Returns:
            Output [batch, out_channels, size_x, size_y]
        """
        batch_size = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]
        
        # FFT
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        
        # Initialize output in Fourier space
        out_ft = torch.zeros(
            batch_size, self.out_channels, size_x, size_y // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        # Multiply relevant Fourier modes
        # Lower modes
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            'bixy,ioxy->boxy',
            x_ft[:, :, :self.modes1, :self.modes2],
            torch.view_as_complex(self.weights1)
        )
        
        # Upper modes
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            'bixy,ioxy->boxy',
            x_ft[:, :, -self.modes1:, :self.modes2],
            torch.view_as_complex(self.weights2)
        )
        
        # Inverse FFT
        x = torch.fft.irfft2(out_ft, s=(size_x, size_y), dim=(-2, -1))
        
        return x


class SpectralConv3d(nn.Module):
    """3D Spectral convolution layer."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        
        self.scale = 1.0 / (in_channels * out_channels)
        
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, 2)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, 2)
        )
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, 2)
        )
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, 2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = x.shape[0]
        size_x, size_y, size_z = x.shape[2], x.shape[3], x.shape[4]
        
        # FFT
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        
        # Initialize output
        out_ft = torch.zeros(
            batch_size, self.out_channels, size_x, size_y, size_z // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        # Multiply Fourier modes
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = torch.einsum(
            'bixyz,ioxyz->boxyz',
            x_ft[:, :, :self.modes1, :self.modes2, :self.modes3],
            torch.view_as_complex(self.weights1)
        )
        
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = torch.einsum(
            'bixyz,ioxyz->boxyz',
            x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3],
            torch.view_as_complex(self.weights2)
        )
        
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = torch.einsum(
            'bixyz,ioxyz->boxyz',
            x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3],
            torch.view_as_complex(self.weights3)
        )
        
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = torch.einsum(
            'bixyz,ioxyz->boxyz',
            x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3],
            torch.view_as_complex(self.weights4)
        )
        
        # Inverse FFT
        x = torch.fft.irfftn(out_ft, s=(size_x, size_y, size_z), dim=(-3, -2, -1))
        
        return x


class SpectralConv1d(nn.Module):
    """1D Spectral convolution layer."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        self.scale = 1.0 / (in_channels * out_channels)
        
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, 2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = x.shape[0]
        size = x.shape[2]
        
        # FFT
        x_ft = torch.fft.rfft(x, dim=-1)
        
        # Initialize output
        out_ft = torch.zeros(
            batch_size, self.out_channels, size // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        # Multiply Fourier modes
        out_ft[:, :, :self.modes] = torch.einsum(
            'bix,iox->box',
            x_ft[:, :, :self.modes],
            torch.view_as_complex(self.weights)
        )
        
        # Inverse FFT
        x = torch.fft.irfft(out_ft, n=size, dim=-1)
        
        return x


class FourierNeuralOperator(nn.Module):
    """
    Fourier Neural Operator for learning solution operators.
    
    Architecture:
    1. Lift input to higher dimension (P)
    2. Apply L layers of Fourier integral operators (K)
    3. Project back to output dimension (Q)
    
    G = Q ∘ K_L ∘ ... ∘ K_1 ∘ P
    """
    
    def __init__(
        self,
        modes: Union[int, Tuple[int, ...]],
        width: int,
        in_channels: int = 1,
        out_channels: int = 1,
        n_layers: int = 4,
        dim: int = 2,
        use_batch_norm: bool = False,
        activation: str = 'gelu'
    ):
        """
        Initialize FNO.
        
        Args:
            modes: Number of Fourier modes (int for 1D, tuple for 2D/3D)
            width: Latent space dimension
            in_channels: Input channels
            out_channels: Output channels
            n_layers: Number of Fourier layers
            dim: Spatial dimension (1, 2, or 3)
            use_batch_norm: Use batch normalization
            activation: Activation function
        """
        super().__init__()
        
        self.dim = dim
        self.width = width
        self.n_layers = n_layers
        
        # Handle modes
        if isinstance(modes, int):
            if dim == 1:
                self.modes = modes
            else:
                self.modes = (modes,) * dim
        else:
            self.modes = modes
        
        # Lifting layer
        self.fc0 = nn.Linear(in_channels + dim, width)
        
        # Fourier layers
        self.fourier_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if use_batch_norm else None
        
        for _ in range(n_layers):
            if dim == 1:
                conv = SpectralConv1d(width, width, self.modes)
            elif dim == 2:
                conv = SpectralConv2d(width, width, self.modes[0], self.modes[1])
            elif dim == 3:
                conv = SpectralConv3d(
                    width, width,
                    self.modes[0], self.modes[1], self.modes[2]
                )
            
            self.fourier_layers.append(conv)
            
            # Skip connection (local convolution)
            if dim == 1:
                self.w_layers.append(nn.Conv1d(width, width, 1))
            elif dim == 2:
                self.w_layers.append(nn.Conv2d(width, width, 1))
            elif dim == 3:
                self.w_layers.append(nn.Conv3d(width, width, 1))
            
            # Batch normalization
            if use_batch_norm:
                if dim == 1:
                    self.bn_layers.append(nn.BatchNorm1d(width))
                elif dim == 2:
                    self.bn_layers.append(nn.BatchNorm2d(width))
                elif dim == 3:
                    self.bn_layers.append(nn.BatchNorm3d(width))
        
        # Projection layer
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)
        
        # Activation
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch, in_channels, *spatial_dims]
            
        Returns:
            Output [batch, out_channels, *spatial_dims]
        """
        # Get grid coordinates
        grid = self.get_grid(x.shape, x.device)
        
        # Concatenate input with grid coordinates
        x = torch.cat([x, grid], dim=1)  # [batch, in_channels + dim, *spatial]
        
        # Move channels to last dimension for linear layer
        if self.dim == 1:
            x = x.permute(0, 2, 1)  # [batch, size, channels]
        elif self.dim == 2:
            x = x.permute(0, 2, 3, 1)  # [batch, size_x, size_y, channels]
        elif self.dim == 3:
            x = x.permute(0, 2, 3, 4, 1)  # [batch, size_x, size_y, size_z, channels]
        
        # Lifting
        x = self.fc0(x)  # [batch, *spatial, width]
        
        # Move channels back for convolutions
        if self.dim == 1:
            x = x.permute(0, 2, 1)  # [batch, width, size]
        elif self.dim == 2:
            x = x.permute(0, 3, 1, 2)  # [batch, width, size_x, size_y]
        elif self.dim == 3:
            x = x.permute(0, 4, 1, 2, 3)  # [batch, width, size_x, size_y, size_z]
        
        # Fourier layers
        for i, (conv, w) in enumerate(zip(self.fourier_layers, self.w_layers)):
            x1 = conv(x)
            x2 = w(x)
            x = x1 + x2
            
            if self.bn_layers is not None:
                x = self.bn_layers[i](x)
            
            if i < self.n_layers - 1:
                x = self.activation(x)
        
        # Move channels to last dimension
        if self.dim == 1:
            x = x.permute(0, 2, 1)
        elif self.dim == 2:
            x = x.permute(0, 2, 3, 1)
        elif self.dim == 3:
            x = x.permute(0, 2, 3, 4, 1)
        
        # Projection
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        
        return x
    
    def get_grid(
        self,
        shape: Tuple[int, ...],
        device: torch.device
    ) -> torch.Tensor:
        """
        Generate grid coordinates.
        
        Args:
            shape: Tensor shape [batch, channels, *spatial]
            device: Device
            
        Returns:
            Grid coordinates [batch, dim, *spatial]
        """
        batch_size = shape[0]
        
        if self.dim == 1:
            size = shape[2]
            grid = torch.linspace(0, 1, size, device=device).reshape(1, 1, size)
            grid = grid.repeat(batch_size, 1, 1)
            
        elif self.dim == 2:
            size_x, size_y = shape[2], shape[3]
            grid_x = torch.linspace(0, 1, size_x, device=device).reshape(1, 1, size_x, 1)
            grid_y = torch.linspace(0, 1, size_y, device=device).reshape(1, 1, 1, size_y)
            
            grid_x = grid_x.repeat(batch_size, 1, 1, size_y)
            grid_y = grid_y.repeat(batch_size, 1, size_x, 1)
            
            grid = torch.cat([grid_x, grid_y], dim=1)
            
        elif self.dim == 3:
            size_x, size_y, size_z = shape[2], shape[3], shape[4]
            grid_x = torch.linspace(0, 1, size_x, device=device).reshape(1, 1, size_x, 1, 1)
            grid_y = torch.linspace(0, 1, size_y, device=device).reshape(1, 1, 1, size_y, 1)
            grid_z = torch.linspace(0, 1, size_z, device=device).reshape(1, 1, 1, 1, size_z)
            
            grid_x = grid_x.repeat(batch_size, 1, 1, size_y, size_z)
            grid_y = grid_y.repeat(batch_size, 1, size_x, 1, size_z)
            grid_z = grid_z.repeat(batch_size, 1, size_x, size_y, 1)
            
            grid = torch.cat([grid_x, grid_y, grid_z], dim=1)
        
        return grid


class PhysicsInformedFNO(FourierNeuralOperator):
    """
    Physics-Informed Fourier Neural Operator.
    
    Adds physics-based regularization to FNO training.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pde_fn = None
        self.physics_weight = 0.0
    
    def set_physics_loss(self, pde_fn: callable, weight: float = 1.0):
        """Set physics loss function."""
        self.pde_fn = pde_fn
        self.physics_weight = weight
    
    def compute_physics_loss(self, x: torch.Tensor, **pde_kwargs) -> torch.Tensor:
        """Compute physics loss."""
        if self.pde_fn is None:
            return torch.tensor(0.0)
        
        x = x.requires_grad_(True)
        u = self.forward(x)
        
        # Compute PDE residual
        residual = self.pde_fn(u, x, **pde_kwargs)
        
        return torch.mean(residual ** 2)
    
    def compute_loss(
        self,
        x: torch.Tensor,
        y_target: torch.Tensor,
        **pde_kwargs
    ) -> dict:
        """Compute total loss."""
        # Data loss
        y_pred = self.forward(x)
        data_loss = torch.mean((y_pred - y_target) ** 2)
        
        losses = {'data': data_loss}
        
        # Physics loss
        if self.pde_fn is not None:
            physics_loss = self.compute_physics_loss(x, **pde_kwargs)
            losses['physics'] = physics_loss
            losses['total'] = data_loss + self.physics_weight * physics_loss
        else:
            losses['total'] = data_loss
        
        return losses


class MultiScaleFNO(nn.Module):
    """
    Multi-scale Fourier Neural Operator.
    
    Combines multiple FNOs at different resolutions.
    """
    
    def __init__(
        self,
        modes_list: List[Tuple[int, ...]],
        width: int,
        in_channels: int = 1,
        out_channels: int = 1,
        dim: int = 2
    ):
        super().__init__()
        
        self.fno_layers = nn.ModuleList([
            FourierNeuralOperator(
                modes=modes,
                width=width,
                in_channels=in_channels,
                out_channels=width,
                dim=dim
            )
            for modes in modes_list
        ])
        
        # Fusion layer
        fusion_channels = width * len(modes_list)
        if dim == 2:
            self.fusion = nn.Conv2d(fusion_channels, out_channels, 1)
        elif dim == 1:
            self.fusion = nn.Conv1d(fusion_channels, out_channels, 1)
        elif dim == 3:
            self.fusion = nn.Conv3d(fusion_channels, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        outputs = []
        for fno in self.fno_layers:
            outputs.append(fno(x))
        
        # Concatenate multi-scale features
        x = torch.cat(outputs, dim=1)
        
        # Fusion
        x = self.fusion(x)
        
        return x


class AdaptiveFNO(FourierNeuralOperator):
    """
    FNO with adaptive mode selection.
    
    Dynamically selects important Fourier modes during training.
    """
    
    def __init__(self, *args, sparsity_threshold: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparsity_threshold = sparsity_threshold
        self.mode_importance = None
    
    def update_mode_selection(self, validation_data: torch.Tensor):
        """Update mode selection based on importance."""
        with torch.no_grad():
            # Forward pass to compute mode importance
            x = validation_data
            x_ft = torch.fft.rfftn(x, dim=list(range(2, x.dim())))
            
            # Compute magnitude of Fourier coefficients
            magnitude = torch.abs(x_ft).mean(dim=(0, 1))
            
            # Normalize
            magnitude = magnitude / magnitude.max()
            
            # Select important modes
            self.mode_importance = magnitude
            self.selected_modes = magnitude > self.sparsity_threshold
