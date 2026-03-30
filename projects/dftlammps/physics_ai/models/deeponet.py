"""
Deep Operator Network (DeepONet) Implementation

DeepONet learns nonlinear operators between infinite-dimensional
function spaces using branch and trunk networks.

Reference:
    Lu et al., "Learning nonlinear operators via DeepONet based on 
    the universal approximation theorem of operators", Nature MI 2021
"""

import torch
import torch.nn as nn
from typing import List, Optional, Callable, Dict, Tuple
import numpy as np


class DeepONet(nn.Module):
    """
    Deep Operator Network for learning solution operators of PDEs.
    
    Architecture:
    - Branch net: Encodes input function at sensor points
    - Trunk net: Evaluates output function at query locations
    - Output: <branch_output, trunk_output> + bias
    
    G(u)(y) = sum(b_i * t_i) + b_0
    
    where b = branch_net({u(x_1), ..., u(x_m)})
    and   t = trunk_net(y)
    """
    
    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int,
        output_dim: int = 1,
        branch_hidden_dims: List[int] = [100, 100, 100],
        trunk_hidden_dims: List[int] = [100, 100, 100],
        activation: str = 'relu',
        use_bias: bool = True,
        share_trunk: bool = False
    ):
        """
        Initialize DeepONet.
        
        Args:
            branch_input_dim: Number of sensor points (m)
            trunk_input_dim: Dimension of evaluation coordinates (y)
            output_dim: Number of output functions
            branch_hidden_dims: Hidden dimensions for branch network
            trunk_hidden_dims: Hidden dimensions for trunk network
            activation: Activation function
            use_bias: Whether to include bias term
            share_trunk: Whether trunk is shared across output dimensions
        """
        super().__init__()
        
        self.branch_input_dim = branch_input_dim
        self.trunk_input_dim = trunk_input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.share_trunk = share_trunk
        
        # Build branch network (encodes input function)
        self.branch_net = self._build_mlp(
            branch_input_dim,
            branch_hidden_dims[-1],
            branch_hidden_dims[:-1],
            activation
        )
        
        # Build trunk network(s) (evaluates output function)
        if share_trunk:
            self.trunk_nets = nn.ModuleList([
                self._build_mlp(
                    trunk_input_dim,
                    branch_hidden_dims[-1],
                    trunk_hidden_dims,
                    activation
                )
                for _ in range(output_dim)
            ])
        else:
            self.trunk_net = self._build_mlp(
                trunk_input_dim,
                branch_hidden_dims[-1],
                trunk_hidden_dims,
                activation
            )
        
        # Bias terms
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        
        # Physics-informed mode
        self.pde_fn = None
        self.physics_weight = 0.0
        
    def _build_mlp(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: str
    ) -> nn.Module:
        """Build MLP."""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'sin':
                layers.append(nn.Sin())
            
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def forward(
        self,
        u: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            u: Input function values at sensor points [batch, branch_input_dim]
            y: Evaluation coordinates [batch, trunk_input_dim] or [n_points, trunk_input_dim]
            
        Returns:
            Output function values [batch, output_dim] or [batch, n_points, output_dim]
        """
        # Branch network output
        branch_out = self.branch_net(u)  # [batch, hidden_dim]
        
        # Handle different input shapes for y
        if y.dim() == 2 and y.shape[0] == u.shape[0]:
            # Same batch: [batch, trunk_input_dim]
            if self.share_trunk:
                outputs = []
                for i, trunk_net in enumerate(self.trunk_nets):
                    trunk_out = trunk_net(y)  # [batch, hidden_dim]
                    # Dot product
                    output = torch.sum(branch_out * trunk_out, dim=1, keepdim=True)
                    if self.use_bias:
                        output = output + self.bias[i]
                    outputs.append(output)
                return torch.cat(outputs, dim=1)  # [batch, output_dim]
            else:
                trunk_out = self.trunk_net(y)  # [batch, hidden_dim]
                # Dot product
                output = torch.sum(branch_out * trunk_out, dim=1, keepdim=True)
                if self.output_dim > 1:
                    # Repeat for multiple outputs
                    output = output.expand(-1, self.output_dim)
                if self.use_bias:
                    output = output + self.bias
                return output
        else:
            # Different points: y is [n_points, trunk_input_dim]
            n_points = y.shape[0]
            batch_size = u.shape[0]
            
            # Expand for broadcasting
            branch_out_expanded = branch_out.unsqueeze(1)  # [batch, 1, hidden_dim]
            
            if self.share_trunk:
                outputs = []
                for i, trunk_net in enumerate(self.trunk_nets):
                    trunk_out = trunk_net(y)  # [n_points, hidden_dim]
                    trunk_out_expanded = trunk_out.unsqueeze(0)  # [1, n_points, hidden_dim]
                    
                    # Dot product
                    output = torch.sum(
                        branch_out_expanded * trunk_out_expanded, 
                        dim=2
                    )  # [batch, n_points]
                    
                    if self.use_bias:
                        output = output + self.bias[i]
                    outputs.append(output.unsqueeze(-1))
                
                return torch.cat(outputs, dim=-1)  # [batch, n_points, output_dim]
            else:
                trunk_out = self.trunk_net(y)  # [n_points, hidden_dim]
                trunk_out_expanded = trunk_out.unsqueeze(0)  # [1, n_points, hidden_dim]
                
                # Dot product
                output = torch.sum(
                    branch_out_expanded * trunk_out_expanded,
                    dim=2
                )  # [batch, n_points]
                
                if self.output_dim > 1:
                    output = output.unsqueeze(-1).expand(-1, -1, self.output_dim)
                else:
                    output = output.unsqueeze(-1)
                
                if self.use_bias:
                    if self.output_dim > 1:
                        output = output + self.bias.unsqueeze(0).unsqueeze(0)
                    else:
                        output = output + self.bias[0]
                
                return output
    
    def set_physics_loss(
        self,
        pde_fn: Callable,
        weight: float = 1.0
    ):
        """
        Enable physics-informed training.
        
        Args:
            pde_fn: Function computing PDE residual
            weight: Weight for physics loss
        """
        self.pde_fn = pde_fn
        self.physics_weight = weight
    
    def compute_physics_loss(
        self,
        u: torch.Tensor,
        y_collocation: torch.Tensor,
        **pde_kwargs
    ) -> torch.Tensor:
        """
        Compute physics-informed loss.
        
        Args:
            u: Input function
            y_collocation: Collocation points
            
        Returns:
            Physics loss
        """
        if self.pde_fn is None:
            return torch.tensor(0.0)
        
        y_collocation = y_collocation.requires_grad_(True)
        
        # Forward pass
        G_u = self.forward(u, y_collocation)
        
        # Compute PDE residual
        residual = self.pde_fn(G_u, y_collocation, **pde_kwargs)
        
        return torch.mean(residual ** 2)
    
    def compute_loss(
        self,
        u: torch.Tensor,
        y: torch.Tensor,
        G_u_target: torch.Tensor,
        y_collocation: Optional[torch.Tensor] = None,
        **pde_kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.
        
        Args:
            u: Input functions
            y: Evaluation points
            G_u_target: Target output functions
            y_collocation: Collocation points for physics loss
            
        Returns:
            Dictionary of losses
        """
        # Data loss
        G_u_pred = self.forward(u, y)
        data_loss = torch.mean((G_u_pred - G_u_target) ** 2)
        
        losses = {'data': data_loss}
        
        # Physics loss
        if y_collocation is not None and self.pde_fn is not None:
            physics_loss = self.compute_physics_loss(u, y_collocation, **pde_kwargs)
            losses['physics'] = physics_loss
            losses['total'] = data_loss + self.physics_weight * physics_loss
        else:
            losses['total'] = data_loss
        
        return losses


class SeparablePhysicsInformedDeepONet(nn.Module):
    """
    Separable Physics-Informed DeepONet (Sep-PI-DeepONet).
    
    Uses separation of variables to reduce computational cost.
    
    Reference:
        "Separable physics-informed DeepONet: Breaking the curse of 
        dimensionality in physics-informed machine learning"
    """
    
    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int,
        rank: int = 10,
        branch_hidden_dims: List[int] = [50, 50],
        trunk_hidden_dims: List[int] = [50, 50],
        activation: str = 'tanh'
    ):
        """
        Initialize separable DeepONet.
        
        Args:
            branch_input_dim: Sensor points dimension
            trunk_input_dim: Coordinate dimension (spatial/temporal)
            rank: Rank of tensor decomposition
            branch_hidden_dims: Branch network hidden dimensions
            trunk_hidden_dims: Trunk network hidden dimensions
            activation: Activation function
        """
        super().__init__()
        
        self.trunk_input_dim = trunk_input_dim
        self.rank = rank
        
        # Branch network
        self.branch_net = self._build_mlp(
            branch_input_dim,
            rank * trunk_input_dim,
            branch_hidden_dims,
            activation
        )
        
        # Separate trunk network for each dimension
        self.trunk_nets = nn.ModuleList([
            self._build_mlp(1, rank, trunk_hidden_dims, activation)
            for _ in range(trunk_input_dim)
        ])
        
    def _build_mlp(self, input_dim, output_dim, hidden_dims, activation):
        """Build MLP."""
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)
    
    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with separable structure.
        
        Args:
            u: [batch, branch_input_dim]
            y: [batch, trunk_input_dim]
            
        Returns:
            Output [batch, 1]
        """
        # Branch output (coefficients)
        branch_out = self.branch_net(u)  # [batch, rank * trunk_input_dim]
        branch_out = branch_out.view(-1, self.rank, self.trunk_input_dim)
        
        # Trunk outputs for each dimension
        trunk_outputs = []
        for i in range(self.trunk_input_dim):
            y_i = y[:, i:i+1]  # [batch, 1]
            trunk_out = self.trunk_nets[i](y_i)  # [batch, rank]
            trunk_outputs.append(trunk_out)
        
        # Combine using tensor product
        output = torch.zeros(u.shape[0], 1, device=u.device)
        for r in range(self.rank):
            product = torch.ones(u.shape[0], device=u.device)
            for i in range(self.trunk_input_dim):
                product = product * (branch_out[:, r, i] * trunk_outputs[i][:, r])
            output = output + product.unsqueeze(-1)
        
        return output


class MultiOutputDeepONet(nn.Module):
    """
    DeepONet with multiple outputs (e.g., for vector fields).
    """
    
    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int,
        output_dim: int,
        branch_hidden_dims: List[int] = [100, 100],
        trunk_hidden_dims: List[int] = [100, 100],
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.output_dim = output_dim
        
        # Shared branch network
        self.branch_net = self._build_mlp(
            branch_input_dim,
            trunk_hidden_dims[-1],
            branch_hidden_dims,
            activation
        )
        
        # Separate trunk network for each output
        self.trunk_nets = nn.ModuleList([
            self._build_mlp(
                trunk_input_dim,
                trunk_hidden_dims[-1],
                trunk_hidden_dims,
                activation
            )
            for _ in range(output_dim)
        ])
        
        self.biases = nn.Parameter(torch.zeros(output_dim))
    
    def _build_mlp(self, input_dim, output_dim, hidden_dims, activation):
        """Build MLP."""
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)
    
    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            u: [batch, branch_input_dim]
            y: [n_points, trunk_input_dim]
            
        Returns:
            Output [batch, n_points, output_dim]
        """
        branch_out = self.branch_net(u)  # [batch, hidden_dim]
        
        n_points = y.shape[0]
        batch_size = u.shape[0]
        
        outputs = []
        for i in range(self.output_dim):
            trunk_out = self.trunk_nets[i](y)  # [n_points, hidden_dim]
            
            # Outer product
            output = torch.matmul(branch_out, trunk_out.T)  # [batch, n_points]
            output = output + self.biases[i]
            outputs.append(output.unsqueeze(-1))
        
        return torch.cat(outputs, dim=-1)  # [batch, n_points, output_dim]


class AttentionDeepONet(nn.Module):
    """
    DeepONet with attention mechanism.
    
    Uses cross-attention between branch and trunk representations.
    """
    
    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        output_dim: int = 1
    ):
        super().__init__()
        
        # Embed inputs
        self.branch_embed = nn.Linear(branch_input_dim, hidden_dim)
        self.trunk_embed = nn.Linear(trunk_input_dim, hidden_dim)
        
        # Transformer encoder for branch
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.branch_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, n_heads, batch_first=True
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention.
        
        Args:
            u: [batch, branch_input_dim]
            y: [n_points, trunk_input_dim]
            
        Returns:
            Output [batch, n_points, output_dim]
        """
        # Embed and encode branch input
        branch_emb = self.branch_embed(u).unsqueeze(1)  # [batch, 1, hidden_dim]
        branch_encoded = self.branch_encoder(branch_emb)  # [batch, 1, hidden_dim]
        
        # Embed trunk input
        trunk_emb = self.trunk_embed(y).unsqueeze(0)  # [1, n_points, hidden_dim]
        trunk_expanded = trunk_emb.expand(u.shape[0], -1, -1)  # [batch, n_points, hidden_dim]
        
        # Cross-attention
        attn_output, _ = self.cross_attention(
            trunk_expanded, branch_encoded, branch_encoded
        )  # [batch, n_points, hidden_dim]
        
        # Output
        output = self.output_head(attn_output)  # [batch, n_points, output_dim]
        
        return output
