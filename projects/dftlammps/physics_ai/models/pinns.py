"""
Physics-Informed Neural Networks (PINNs) implementation.

PINNs embed physical laws (PDEs) into neural network training through
physics-based residual losses.
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
from typing import Dict, List, Callable, Optional, Tuple, Union
import numpy as np


class PhysicsInformedNN(nn.Module):
    """
    Physics-Informed Neural Network for solving PDEs.
    
    Combines data-driven and physics-driven losses:
    L_total = L_data + λ * L_physics
    
    where L_physics = ||f(u_pred)||^2 (PDE residual)
    
    Reference:
        Raissi et al., "Physics-informed neural networks: A deep learning 
        framework for solving forward and inverse problems involving nonlinear 
        partial differential equations", JCP 2019
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        pde_fn: Callable,
        activation: str = 'tanh',
        use_fourier_features: bool = False,
        fourier_scale: float = 1.0,
        dropout: float = 0.0
    ):
        """
        Initialize PINN.
        
        Args:
            input_dim: Input dimension (spatial/temporal coordinates)
            output_dim: Output dimension (solution values)
            hidden_dims: Hidden layer dimensions
            pde_fn: Function defining PDE residual
            activation: Activation function ('tanh', 'sin', 'relu', 'swish')
            use_fourier_features: Use Fourier feature embeddings
            fourier_scale: Scale for Fourier features
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pde_fn = pde_fn
        self.use_fourier_features = use_fourier_features
        
        # Fourier feature embeddings
        if use_fourier_features:
            self.fourier_proj = nn.Linear(input_dim, 128)
            nn.init.normal_(self.fourier_proj.weight, 0, fourier_scale)
            nn.init.zeros_(self.fourier_proj.bias)
            network_input = 256  # 2 * 128 for sin and cos
        else:
            network_input = input_dim
            
        # Build network
        layers = []
        prev_dim = network_input
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Activation
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sin':
                layers.append(SineActivation())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'swish':
                layers.append(Swish())
            elif activation == 'siren':
                layers.append(SIRENActivation())
                
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
                
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        # Loss tracking
        self.loss_history = {'data': [], 'pde': [], 'bc': [], 'total': []}
        
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def fourier_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier feature embedding.
        
        γ(x) = [sin(2πBx), cos(2πBx)]
        """
        proj = 2 * np.pi * self.fourier_proj(x)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input coordinates [batch, input_dim]
            
        Returns:
            Predictions [batch, output_dim]
        """
        if self.use_fourier_features:
            x = self.fourier_embedding(x)
        return self.network(x)
    
    def compute_derivatives(
        self,
        u: torch.Tensor,
        x: torch.Tensor,
        order: int = 2
    ) -> Dict[str, torch.Tensor]:
        """
        Compute derivatives of u w.r.t. x.
        
        Args:
            u: Network output [batch, output_dim]
            x: Input coordinates [batch, input_dim]
            order: Maximum derivative order
            
        Returns:
            Dictionary of derivatives
        """
        derivatives = {}
        
        # First-order derivatives
        grads = []
        for i in range(u.shape[1]):
            grad = autograd.grad(
                outputs=u[:, i:i+1],
                inputs=x,
                grad_outputs=torch.ones_like(u[:, i:i+1]),
                create_graph=True,
                retain_graph=True
            )[0]
            grads.append(grad)
            
        derivatives['first'] = torch.stack(grads, dim=1)  # [batch, output_dim, input_dim]
        
        # Second-order derivatives if needed
        if order >= 2:
            hessians = []
            for i in range(u.shape[1]):
                hessian = []
                for j in range(x.shape[1]):
                    grad2 = autograd.grad(
                        outputs=grads[i][:, j],
                        inputs=x,
                        grad_outputs=torch.ones_like(grads[i][:, j]),
                        create_graph=True,
                        retain_graph=True
                    )[0]
                    hessian.append(grad2)
                hessians.append(torch.stack(hessian, dim=1))
            derivatives['second'] = torch.stack(hessians, dim=1)  # [batch, output_dim, input_dim, input_dim]
        
        return derivatives
    
    def compute_pde_residual(
        self,
        x: torch.Tensor,
        **pde_kwargs
    ) -> torch.Tensor:
        """
        Compute PDE residual.
        
        Args:
            x: Collocation points [batch, input_dim]
            **pde_kwargs: Additional arguments for PDE function
            
        Returns:
            PDE residual [batch]
        """
        x = x.requires_grad_(True)
        u = self.forward(x)
        
        derivatives = self.compute_derivatives(u, x)
        
        residual = self.pde_fn(x, u, derivatives, **pde_kwargs)
        
        return residual
    
    def compute_loss(
        self,
        x_data: Optional[torch.Tensor] = None,
        u_data: Optional[torch.Tensor] = None,
        x_collocation: Optional[torch.Tensor] = None,
        x_bc: Optional[torch.Tensor] = None,
        u_bc: Optional[torch.Tensor] = None,
        x_ic: Optional[torch.Tensor] = None,
        u_ic: Optional[torch.Tensor] = None,
        lambda_pde: float = 1.0,
        lambda_bc: float = 1.0,
        lambda_ic: float = 1.0,
        **pde_kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total PINN loss.
        
        L_total = L_data + λ_pde * L_pde + λ_bc * L_bc + λ_ic * L_ic
        
        Args:
            x_data: Data points coordinates
            u_data: Data values
            x_collocation: Collocation points for PDE residual
            x_bc, u_bc: Boundary condition points and values
            x_ic, u_ic: Initial condition points and values
            lambda_*: Loss weights
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=x_data.device if x_data is not None else 'cpu')
        
        # Data loss
        if x_data is not None and u_data is not None:
            u_pred = self.forward(x_data)
            losses['data'] = torch.mean((u_pred - u_data) ** 2)
            total_loss = total_loss + losses['data']
        else:
            losses['data'] = torch.tensor(0.0)
        
        # PDE residual loss
        if x_collocation is not None:
            residual = self.compute_pde_residual(x_collocation, **pde_kwargs)
            losses['pde'] = torch.mean(residual ** 2)
            total_loss = total_loss + lambda_pde * losses['pde']
        else:
            losses['pde'] = torch.tensor(0.0)
        
        # Boundary condition loss
        if x_bc is not None and u_bc is not None:
            u_bc_pred = self.forward(x_bc)
            losses['bc'] = torch.mean((u_bc_pred - u_bc) ** 2)
            total_loss = total_loss + lambda_bc * losses['bc']
        else:
            losses['bc'] = torch.tensor(0.0)
        
        # Initial condition loss
        if x_ic is not None and u_ic is not None:
            u_ic_pred = self.forward(x_ic)
            losses['ic'] = torch.mean((u_ic_pred - u_ic) ** 2)
            total_loss = total_loss + lambda_ic * losses['ic']
        else:
            losses['ic'] = torch.tensor(0.0)
        
        losses['total'] = total_loss
        
        # Update history
        for key, value in losses.items():
            self.loss_history[key].append(value.item())
        
        return losses
    
    def sample_collocation_points(
        self,
        n_points: int,
        bounds: List[Tuple[float, float]],
        method: str = 'uniform'
    ) -> torch.Tensor:
        """
        Sample collocation points.
        
        Args:
            n_points: Number of points to sample
            bounds: List of (min, max) for each dimension
            method: Sampling method ('uniform', 'lhs', 'sobol')
            
        Returns:
            Sampled points [n_points, input_dim]
        """
        input_dim = len(bounds)
        
        if method == 'uniform':
            points = torch.rand(n_points, input_dim)
            for i, (min_val, max_val) in enumerate(bounds):
                points[:, i] = min_val + (max_val - min_val) * points[:, i]
                
        elif method == 'lhs':
            # Latin Hypercube Sampling
            from torch.quasirandom import SobolEngine
            sobol = SobolEngine(dimension=input_dim, scramble=True)
            points = sobol.draw(n_points)
            for i, (min_val, max_val) in enumerate(bounds):
                points[:, i] = min_val + (max_val - min_val) * points[:, i]
                
        elif method == 'sobol':
            from torch.quasirandom import SobolEngine
            sobol = SobolEngine(dimension=input_dim, scramble=True)
            points = sobol.draw(n_points)
            for i, (min_val, max_val) in enumerate(bounds):
                points[:, i] = min_val + (max_val - min_val) * points[:, i]
        
        return points


class SineActivation(nn.Module):
    """Sine activation function (for SIREN networks)."""
    
    def __init__(self, omega_0: float = 1.0):
        super().__init__()
        self.omega_0 = omega_0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)


class Swish(nn.Module):
    """Swish activation: x * sigmoid(x)."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SIRENActivation(nn.Module):
    """
    SIREN (Sinusoidal Representation Networks) activation.
    
    Reference:
        Sitzmann et al., "Implicit Neural Representations with 
        Periodic Activation Functions", NeurIPS 2020
    """
    
    def __init__(self, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)


class SIREN(nn.Module):
    """
    SIREN network with special initialization.
    
    Particularly effective for representing complex signals
    and their derivatives.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        omega_0: float = 30.0
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # First layer
        layers.append(nn.Linear(prev_dim, hidden_dims[0]))
        layers.append(SIRENActivation(omega_0))
        prev_dim = hidden_dims[0]
        
        # Hidden layers
        for hidden_dim in hidden_dims[1:]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(SIRENActivation(omega_0))
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # SIREN initialization
        self._siren_init(omega_0)
        
    def _siren_init(self, omega_0: float):
        """SIREN weight initialization."""
        with torch.no_grad():
            for i, layer in enumerate(self.network):
                if isinstance(layer, nn.Linear):
                    if i == 0:
                        # First layer
                        bound = 1.0 / layer.weight.shape[1]
                    else:
                        # Subsequent layers
                        bound = np.sqrt(6.0 / layer.weight.shape[1]) / omega_0
                    layer.weight.uniform_(-bound, bound)
                    if layer.bias is not None:
                        layer.bias.uniform_(-bound, bound)
                        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class AdaptiveWeightPINN(PhysicsInformedNN):
    """
    PINN with adaptive weighting for loss terms.
    
    Implements the strategy from:
    "Self-adaptive physics-informed neural networks" by McClenny & Braga-Neto
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Learnable weights for loss terms
        self.log_lambda_pde = nn.Parameter(torch.tensor(0.0))
        self.log_lambda_bc = nn.Parameter(torch.tensor(0.0))
        self.log_lambda_data = nn.Parameter(torch.tensor(0.0))
        
    def compute_loss(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Compute loss with adaptive weighting."""
        lambda_pde = torch.exp(self.log_lambda_pde)
        lambda_bc = torch.exp(self.log_lambda_bc)
        lambda_data = torch.exp(self.log_lambda_data)
        
        kwargs['lambda_pde'] = lambda_pde.item()
        kwargs['lambda_bc'] = lambda_bc.item()
        
        losses = super().compute_loss(*args, **kwargs)
        
        # Update total loss with adaptive weights
        losses['total'] = (
            lambda_data * losses.get('data', 0) +
            lambda_pde * losses.get('pde', 0) +
            lambda_bc * losses.get('bc', 0)
        )
        
        losses['lambda_pde'] = lambda_pde
        losses['lambda_bc'] = lambda_bc
        losses['lambda_data'] = lambda_data
        
        return losses


class DomainDecompositionPINN(nn.Module):
    """
    Domain decomposition PINN for handling complex geometries.
    
    Divides domain into subdomains, each with its own network.
    """
    
    def __init__(
        self,
        n_subdomains: int,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        pde_fn: Callable
    ):
        super().__init__()
        
        self.n_subdomains = n_subdomains
        
        # Create network for each subdomain
        self.subdomain_networks = nn.ModuleList([
            PhysicsInformedNN(input_dim, output_dim, hidden_dims, pde_fn)
            for _ in range(n_subdomains)
        ])
        
        # Subdomain assignment function
        self.subdomain_fn = None
        
    def set_subdomain_function(self, fn: Callable):
        """Set function to assign points to subdomains."""
        self.subdomain_fn = fn
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with subdomain routing."""
        if self.subdomain_fn is None:
            raise ValueError("Subdomain function not set")
        
        subdomain_ids = self.subdomain_fn(x)
        
        outputs = torch.zeros(x.shape[0], self.subdomain_networks[0].output_dim)
        
        for i in range(self.n_subdomains):
            mask = subdomain_ids == i
            if mask.any():
                outputs[mask] = self.subdomain_networks[i](x[mask])
        
        return outputs


# Common PDE formulations
def burgers_pde(
    x: torch.Tensor,
    u: torch.Tensor,
    derivatives: Dict[str, torch.Tensor],
    nu: float = 0.01
) -> torch.Tensor:
    """
    Burgers equation: u_t + u*u_x = nu*u_xx
    
    Assumes x = [t, x]
    """
    u_x = derivatives['first'][:, 0, :]  # [batch, 2]
    u_t = u_x[:, 0]
    u_x_spatial = u_x[:, 1]
    
    u_xx = derivatives['second'][:, 0, 1, 1]
    
    residual = u_t + u[:, 0] * u_x_spatial - nu * u_xx
    return residual


def navier_stokes_pde(
    x: torch.Tensor,
    u: torch.Tensor,
    derivatives: Dict[str, torch.Tensor],
    nu: float = 0.01,
    rho: float = 1.0
) -> torch.Tensor:
    """
    Incompressible Navier-Stokes in 2D.
    
    u = [u, v, p]
    x = [t, x, y]
    """
    # Velocity derivatives
    u_t = derivatives['first'][:, 0, 0]
    u_x = derivatives['first'][:, 0, 1]
    u_y = derivatives['first'][:, 0, 2]
    
    v_t = derivatives['first'][:, 1, 0]
    v_x = derivatives['first'][:, 1, 1]
    v_y = derivatives['first'][:, 1, 2]
    
    p_x = derivatives['first'][:, 2, 1]
    p_y = derivatives['first'][:, 2, 2]
    
    # Second derivatives
    u_xx = derivatives['second'][:, 0, 1, 1]
    u_yy = derivatives['second'][:, 0, 2, 2]
    v_xx = derivatives['second'][:, 1, 1, 1]
    v_yy = derivatives['second'][:, 1, 2, 2]
    
    # Momentum equations
    residual_u = u_t + u[:, 0] * u_x + u[:, 1] * u_y + p_x / rho - nu * (u_xx + u_yy)
    residual_v = v_t + u[:, 0] * v_x + u[:, 1] * v_y + p_y / rho - nu * (v_xx + v_yy)
    
    # Continuity equation
    residual_cont = u_x + v_y
    
    return residual_u.pow(2) + residual_v.pow(2) + residual_cont.pow(2)


def heat_equation_pde(
    x: torch.Tensor,
    u: torch.Tensor,
    derivatives: Dict[str, torch.Tensor],
    alpha: float = 1.0
) -> torch.Tensor:
    """
    Heat equation: u_t = alpha * (u_xx + u_yy)
    
    Assumes x = [t, x, y]
    """
    u_t = derivatives['first'][:, 0, 0]
    u_xx = derivatives['second'][:, 0, 1, 1]
    u_yy = derivatives['second'][:, 0, 2, 2]
    
    residual = u_t - alpha * (u_xx + u_yy)
    return residual


def schrodinger_pde(
    x: torch.Tensor,
    u: torch.Tensor,
    derivatives: Dict[str, torch.Tensor],
    hbar: float = 1.0,
    m: float = 1.0,
    V: float = 0.0
) -> torch.Tensor:
    """
    Time-dependent Schrödinger equation.
    
    i*hbar*psi_t = -hbar^2/(2m) * psi_xx + V*psi
    
    u = [psi_real, psi_imag]
    """
    psi_real = u[:, 0]
    psi_imag = u[:, 1]
    
    # Time derivatives
    psi_real_t = derivatives['first'][:, 0, 0]
    psi_imag_t = derivatives['first'][:, 1, 0]
    
    # Spatial derivatives
    psi_real_xx = derivatives['second'][:, 0, 1, 1]
    psi_imag_xx = derivatives['second'][:, 1, 1, 1]
    
    # Real part: -hbar*psi_imag_t = -hbar^2/(2m)*psi_real_xx + V*psi_real
    residual_real = -hbar * psi_imag_t + (hbar**2 / (2*m)) * psi_real_xx - V * psi_real
    
    # Imaginary part: hbar*psi_real_t = -hbar^2/(2m)*psi_imag_xx + V*psi_imag
    residual_imag = hbar * psi_real_t + (hbar**2 / (2*m)) * psi_imag_xx - V * psi_imag
    
    return residual_real.pow(2) + residual_imag.pow(2)


def poisson_pde(
    x: torch.Tensor,
    u: torch.Tensor,
    derivatives: Dict[str, torch.Tensor],
    f: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Poisson equation: nabla^2 u = f
    
    For 2D: u_xx + u_yy = f
    """
    u_xx = derivatives['second'][:, 0, 0, 0]
    u_yy = derivatives['second'][:, 0, 1, 1]
    
    laplacian = u_xx + u_yy
    
    if f is None:
        f = torch.zeros_like(laplacian)
    
    residual = laplacian - f
    return residual
