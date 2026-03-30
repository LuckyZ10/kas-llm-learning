"""
Physics Constraint Layer - Pluggable physics constraints for any neural network.

This module implements a flexible physics constraint layer that can be 
inserted into any neural network architecture to enforce physical laws
during training and inference.
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
from typing import Dict, List, Callable, Optional, Union, Tuple
import numpy as np


class PhysicsConstraintLayer(nn.Module):
    """
    A pluggable physics constraint layer that can be inserted into any neural network.
    
    Features:
    - Automatic differentiation for computing derivatives
    - Soft and hard constraint enforcement
    - Multi-physics constraint composition
    - Adaptive constraint weighting
    
    Example:
        >>> layer = PhysicsConstraintLayer(
        ...     constraints=['energy', 'momentum'],
        ...     enforcement='soft',
        ...     weight=0.1
        ... )
    """
    
    def __init__(
        self,
        constraints: List[str],
        enforcement: str = 'soft',
        weight: float = 1.0,
        adaptive_weighting: bool = False,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize physics constraint layer.
        
        Args:
            constraints: List of constraint types ('energy', 'momentum', 'mass', 'custom')
            enforcement: 'soft' (loss penalty) or 'hard' (projection)
            weight: Base weight for constraint loss
            adaptive_weighting: Whether to use adaptive weighting
            device: Computing device
        """
        super().__init__()
        self.constraint_types = constraints
        self.enforcement = enforcement
        self.base_weight = weight
        self.adaptive_weighting = adaptive_weighting
        self.device = device
        
        # Constraint weights (adaptive)
        self.constraint_weights = nn.ParameterDict({
            c: nn.Parameter(torch.tensor(weight)) 
            for c in constraints
        })
        
        # Custom constraint functions
        self.custom_constraints: Dict[str, Callable] = {}
        
        # Constraint history for adaptive weighting
        self.constraint_history = {c: [] for c in constraints}
        
    def add_custom_constraint(
        self, 
        name: str, 
        constraint_fn: Callable,
        weight: float = 1.0
    ):
        """Add a custom physics constraint function."""
        self.custom_constraints[name] = constraint_fn
        self.constraint_weights[name] = nn.Parameter(torch.tensor(weight))
        self.constraint_history[name] = []
        
    def compute_derivatives(
        self,
        output: torch.Tensor,
        input_var: torch.Tensor,
        order: int = 1
    ) -> List[torch.Tensor]:
        """
        Compute derivatives using automatic differentiation.
        
        Args:
            output: Network output
            input_var: Input variable to differentiate w.r.t.
            order: Order of derivative (1 or 2)
            
        Returns:
            List of derivative tensors
        """
        derivatives = []
        
        # First-order derivatives
        grads = autograd.grad(
            outputs=output,
            inputs=input_var,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True
        )[0]
        derivatives.append(grads)
        
        # Second-order derivatives if requested
        if order >= 2:
            hessian = torch.zeros(
                output.shape[0], 
                input_var.shape[1], 
                input_var.shape[1],
                device=self.device
            )
            for i in range(input_var.shape[1]):
                grad2 = autograd.grad(
                    outputs=grads[:, i],
                    inputs=input_var,
                    grad_outputs=torch.ones_like(grads[:, i]),
                    create_graph=True,
                    retain_graph=True
                )[0]
                hessian[:, i, :] = grad2
            derivatives.append(hessian)
            
        return derivatives
    
    def energy_conservation_loss(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        masses: torch.Tensor,
        potential_fn: Callable,
        dt: float = 0.001
    ) -> torch.Tensor:
        """
        Compute energy conservation constraint loss.
        
        For Hamiltonian systems: dH/dt = 0
        
        Args:
            positions: Particle positions [batch, n_particles, 3]
            velocities: Particle velocities [batch, n_particles, 3]
            masses: Particle masses [batch, n_particles]
            potential_fn: Function to compute potential energy
            dt: Time step
            
        Returns:
            Energy conservation loss
        """
        # Compute kinetic energy
        kinetic_energy = 0.5 * torch.sum(
            masses.unsqueeze(-1) * velocities.pow(2), 
            dim=(-2, -1)
        )
        
        # Compute potential energy
        potential_energy = potential_fn(positions)
        
        # Total energy
        total_energy = kinetic_energy + potential_energy
        
        # Energy should be conserved (variance should be minimal)
        energy_variation = torch.var(total_energy)
        
        return energy_variation
    
    def momentum_conservation_loss(
        self,
        velocities: torch.Tensor,
        masses: torch.Tensor,
        forces: Optional[torch.Tensor] = None,
        external_forces: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute linear and angular momentum conservation loss.
        
        For isolated systems: dP/dt = 0, dL/dt = 0
        
        Args:
            velocities: Particle velocities [batch, n_particles, 3]
            masses: Particle masses [batch, n_particles]
            forces: Internal forces [batch, n_particles, 3]
            external_forces: External forces [batch, n_particles, 3]
            
        Returns:
            Momentum conservation loss
        """
        # Linear momentum
        linear_momentum = torch.sum(
            masses.unsqueeze(-1) * velocities, 
            dim=1
        )
        
        # For isolated systems, linear momentum should be constant
        momentum_loss = torch.var(linear_momentum, dim=0).mean()
        
        # Angular momentum (if positions provided)
        # L = sum(r x p)
        # This would require positions as input
        
        return momentum_loss
    
    def mass_conservation_loss(
        self,
        densities: torch.Tensor,
        velocities: torch.Tensor,
        dt: float = 0.001
    ) -> torch.Tensor:
        """
        Compute mass conservation (continuity equation) loss.
        
        ∂ρ/∂t + ∇·(ρv) = 0
        
        Args:
            densities: Mass densities [batch, n_points]
            velocities: Velocities [batch, n_points, 3]
            dt: Time step
            
        Returns:
            Mass conservation loss
        """
        # Time derivative of density
        density_dt = (densities[1:] - densities[:-1]) / dt
        
        # Spatial divergence (simplified, assumes regular grid)
        # In practice, use finite differences or spectral methods
        divergence = torch.sum(
            velocities[1:, :, :] - velocities[:-1, :, :], 
            dim=-1
        ) / dt
        
        # Continuity equation residual
        continuity_residual = density_dt + divergence
        
        return torch.mean(continuity_residual.pow(2))
    
    def pde_residual_loss(
        self,
        predictions: torch.Tensor,
        coordinates: torch.Tensor,
        pde_fn: Callable,
        pde_type: str = 'generic'
    ) -> torch.Tensor:
        """
        Compute generic PDE residual loss for PINNs.
        
        Args:
            predictions: Network predictions
            coordinates: Spatial/temporal coordinates
            pde_fn: Function defining the PDE residual
            pde_type: Type of PDE
            
        Returns:
            PDE residual loss
        """
        # Compute required derivatives
        grads = self.compute_derivatives(predictions, coordinates, order=2)
        
        # Evaluate PDE residual
        residual = pde_fn(predictions, coordinates, grads)
        
        return torch.mean(residual.pow(2))
    
    def forward(
        self,
        network_output: torch.Tensor,
        physics_variables: Dict[str, torch.Tensor],
        custom_fn: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply physics constraints to network output.
        
        Args:
            network_output: Output from neural network
            physics_variables: Dictionary of physics variables
            custom_fn: Optional custom constraint function
            
        Returns:
            Tuple of (constrained_output, constraint_losses)
        """
        constraint_losses = {}
        
        # Apply each constraint
        for constraint_type in self.constraint_types:
            if constraint_type == 'energy':
                loss = self.energy_conservation_loss(
                    physics_variables.get('positions'),
                    physics_variables.get('velocities'),
                    physics_variables.get('masses'),
                    physics_variables.get('potential_fn', lambda x: torch.zeros(x.shape[0])),
                    physics_variables.get('dt', 0.001)
                )
                
            elif constraint_type == 'momentum':
                loss = self.momentum_conservation_loss(
                    physics_variables.get('velocities'),
                    physics_variables.get('masses'),
                    physics_variables.get('forces'),
                    physics_variables.get('external_forces')
                )
                
            elif constraint_type == 'mass':
                loss = self.mass_conservation_loss(
                    physics_variables.get('densities'),
                    physics_variables.get('velocities'),
                    physics_variables.get('dt', 0.001)
                )
                
            elif constraint_type == 'pde':
                loss = self.pde_residual_loss(
                    network_output,
                    physics_variables.get('coordinates'),
                    physics_variables.get('pde_fn'),
                    physics_variables.get('pde_type', 'generic')
                )
                
            elif constraint_type == 'custom' and custom_fn is not None:
                loss = custom_fn(network_output, physics_variables)
                
            elif constraint_type in self.custom_constraints:
                loss = self.custom_constraints[constraint_type](
                    network_output, physics_variables
                )
            else:
                continue
                
            constraint_losses[constraint_type] = loss
            
            # Update history for adaptive weighting
            self.constraint_history[constraint_type].append(loss.item())
        
        # Compute weighted total constraint loss
        total_loss = torch.tensor(0.0, device=self.device)
        for name, loss in constraint_losses.items():
            weight = self.constraint_weights[name]
            total_loss = total_loss + weight * loss
            
        # Apply constraints to output (hard enforcement)
        if self.enforcement == 'hard':
            # Project output onto constraint manifold
            constrained_output = self._project_onto_manifold(
                network_output, constraint_losses
            )
        else:
            constrained_output = network_output
            
        return constrained_output, constraint_losses
    
    def _project_onto_manifold(
        self, 
        output: torch.Tensor, 
        constraints: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Project output onto constraint manifold (hard enforcement).
        
        This is a simplified projection. More sophisticated methods
        like optimization-based projection can be implemented.
        """
        # Simple gradient-based projection
        projected = output.clone()
        
        for name, constraint in constraints.items():
            # Gradient of constraint w.r.t. output
            grad = autograd.grad(
                constraint, output, 
                create_graph=True
            )[0]
            
            # Project orthogonal to constraint gradient
            projected = projected - constraint * grad / (grad.norm() + 1e-8)
            
        return projected
    
    def adapt_weights(self, window_size: int = 10):
        """
        Adapt constraint weights based on training history.
        
        Uses the strategy from "Self-adaptive physics-informed neural networks"
        to balance different constraint terms.
        """
        if not self.adaptive_weighting:
            return
            
        for name, history in self.constraint_history.items():
            if len(history) >= window_size:
                recent_mean = np.mean(history[-window_size:])
                
                # Increase weight if constraint is not satisfied
                if recent_mean > 0.01:
                    with torch.no_grad():
                        self.constraint_weights[name].data *= 1.1
                # Decrease weight if constraint is well satisfied
                elif recent_mean < 0.001:
                    with torch.no_grad():
                        self.constraint_weights[name].data *= 0.9
                        
    def get_constraint_summary(self) -> Dict[str, float]:
        """Get summary statistics for all constraints."""
        summary = {}
        for name, history in self.constraint_history.items():
            if history:
                summary[name] = {
                    'current': history[-1],
                    'mean': np.mean(history),
                    'std': np.std(history),
                    'weight': self.constraint_weights[name].item()
                }
        return summary
