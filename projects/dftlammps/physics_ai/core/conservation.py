"""
Conservation Laws Module

Implementation of fundamental physics conservation laws:
- Energy Conservation
- Momentum Conservation  
- Mass Conservation
- Angular Momentum Conservation
"""

import torch
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from abc import ABC, abstractmethod


class ConservationLaw(ABC):
    """Abstract base class for conservation laws."""
    
    @abstractmethod
    def compute_violation(
        self,
        state: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """Compute violation of conservation law."""
        pass
    
    @abstractmethod
    def compute_invariant(
        self,
        state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute the conserved quantity."""
        pass
    
    @abstractmethod
    def project_to_manifold(
        self,
        state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Project state onto conservation manifold."""
        pass


class EnergyConservation(ConservationLaw):
    """
    Energy conservation for Hamiltonian systems.
    
    For conservative systems: dE/dt = 0
    
    Total Energy E = T + V where:
    - T = kinetic energy
    - V = potential energy
    """
    
    def __init__(
        self,
        potential_fn: Optional[Callable] = None,
        include_dissipation: bool = False,
        dissipation_fn: Optional[Callable] = None
    ):
        """
        Initialize energy conservation law.
        
        Args:
            potential_fn: Function to compute potential energy
            include_dissipation: Whether to include dissipative effects
            dissipation_fn: Function to compute energy dissipation
        """
        self.potential_fn = potential_fn
        self.include_dissipation = include_dissipation
        self.dissipation_fn = dissipation_fn
        
    def compute_kinetic_energy(
        self,
        velocities: torch.Tensor,
        masses: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute kinetic energy.
        
        Args:
            velocities: [batch, n_particles, 3]
            masses: [batch, n_particles]
            
        Returns:
            Kinetic energy [batch]
        """
        # T = 0.5 * sum(m * v^2)
        kinetic = 0.5 * torch.sum(
            masses.unsqueeze(-1) * velocities.pow(2),
            dim=(-2, -1)
        )
        return kinetic
    
    def compute_potential_energy(
        self,
        positions: torch.Tensor,
        potential_fn: Optional[Callable] = None
    ) -> torch.Tensor:
        """
        Compute potential energy.
        
        Args:
            positions: [batch, n_particles, 3]
            potential_fn: Override potential function
            
        Returns:
            Potential energy [batch]
        """
        fn = potential_fn or self.potential_fn
        if fn is None:
            raise ValueError("Potential function not provided")
        return fn(positions)
    
    def compute_invariant(
        self,
        state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute total energy.
        
        Args:
            state: Dictionary containing 'positions', 'velocities', 'masses'
            
        Returns:
            Total energy [batch]
        """
        kinetic = self.compute_kinetic_energy(
            state['velocities'], 
            state['masses']
        )
        potential = self.compute_potential_energy(
            state['positions']
        )
        return kinetic + potential
    
    def compute_violation(
        self,
        state: Dict[str, torch.Tensor],
        state_prev: Optional[Dict[str, torch.Tensor]] = None,
        dt: float = 0.001,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute energy conservation violation.
        
        For conservative systems: |E(t) - E(0)| should be minimal.
        
        Args:
            state: Current state
            state_prev: Previous state (for computing dE/dt)
            dt: Time step
            
        Returns:
            Energy violation (scalar loss)
        """
        energy = self.compute_invariant(state)
        
        if state_prev is not None:
            # dE/dt = 0 for conservative systems
            energy_prev = self.compute_invariant(state_prev)
            dEdt = (energy - energy_prev) / dt
            violation = dEdt.pow(2).mean()
            
            if self.include_dissipation and self.dissipation_fn is not None:
                # Account for dissipation: dE/dt = -dissipation
                dissipation = self.dissipation_fn(state)
                violation = (dEdt + dissipation).pow(2).mean()
        else:
            # Use variance across batch
            violation = torch.var(energy)
            
        return violation
    
    def project_to_manifold(
        self,
        state: Dict[str, torch.Tensor],
        reference_energy: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Project state onto energy conservation manifold.
        
        This adjusts velocities to match the reference energy.
        
        Args:
            state: Current state
            reference_energy: Target energy (if None, use mean)
            
        Returns:
            Projected state
        """
        current_energy = self.compute_invariant(state)
        
        if reference_energy is None:
            reference_energy = current_energy.mean()
            
        # Scale velocities to match energy
        ratio = (reference_energy / (current_energy + 1e-8)).sqrt()
        
        projected_state = state.copy()
        projected_state['velocities'] = state['velocities'] * ratio.unsqueeze(-1).unsqueeze(-1)
        
        return projected_state


class MomentumConservation(ConservationLaw):
    """
    Linear and angular momentum conservation.
    
    For isolated systems:
    - dP/dt = 0 (linear momentum)
    - dL/dt = 0 (angular momentum)
    """
    
    def __init__(
        self,
        conserve_linear: bool = True,
        conserve_angular: bool = True,
        include_external_forces: bool = False
    ):
        """
        Initialize momentum conservation.
        
        Args:
            conserve_linear: Whether to conserve linear momentum
            conserve_angular: Whether to conserve angular momentum
            include_external_forces: Account for external forces
        """
        self.conserve_linear = conserve_linear
        self.conserve_angular = conserve_angular
        self.include_external_forces = include_external_forces
        
    def compute_linear_momentum(
        self,
        velocities: torch.Tensor,
        masses: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute total linear momentum.
        
        P = sum(m_i * v_i)
        
        Args:
            velocities: [batch, n_particles, 3]
            masses: [batch, n_particles]
            
        Returns:
            Linear momentum [batch, 3]
        """
        momentum = torch.sum(
            masses.unsqueeze(-1) * velocities,
            dim=1
        )
        return momentum
    
    def compute_angular_momentum(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        masses: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute total angular momentum.
        
        L = sum(r_i x p_i) = sum(m_i * r_i x v_i)
        
        Args:
            positions: [batch, n_particles, 3]
            velocities: [batch, n_particles, 3]
            masses: [batch, n_particles]
            
        Returns:
            Angular momentum [batch, 3]
        """
        # Linear momentum of each particle
        p = masses.unsqueeze(-1) * velocities  # [batch, n, 3]
        
        # Cross product r x p
        angular_momentum = torch.cross(positions, p, dim=-1)  # [batch, n, 3]
        
        # Sum over particles
        total_L = torch.sum(angular_momentum, dim=1)  # [batch, 3]
        
        return total_L
    
    def compute_invariant(
        self,
        state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute momentum invariants.
        
        Returns:
            Dictionary with 'linear' and 'angular' momentum
        """
        invariants = {}
        
        if self.conserve_linear:
            invariants['linear'] = self.compute_linear_momentum(
                state['velocities'],
                state['masses']
            )
            
        if self.conserve_angular:
            invariants['angular'] = self.compute_angular_momentum(
                state['positions'],
                state['velocities'],
                state['masses']
            )
            
        return invariants
    
    def compute_violation(
        self,
        state: Dict[str, torch.Tensor],
        state_prev: Optional[Dict[str, torch.Tensor]] = None,
        dt: float = 0.001,
        external_forces: Optional[torch.Tensor] = None,
        external_torques: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute momentum conservation violation.
        
        Args:
            state: Current state
            state_prev: Previous state
            dt: Time step
            external_forces: External forces [batch, 3]
            external_torques: External torques [batch, 3]
            
        Returns:
            Momentum violation loss
        """
        violation = torch.tensor(0.0)
        
        # Linear momentum
        if self.conserve_linear:
            linear_momentum = self.compute_linear_momentum(
                state['velocities'],
                state['masses']
            )
            
            if state_prev is not None:
                linear_prev = self.compute_linear_momentum(
                    state_prev['velocities'],
                    state_prev['masses']
                )
                dPdt = (linear_momentum - linear_prev) / dt
                
                if self.include_external_forces and external_forces is not None:
                    # dP/dt = F_ext
                    violation = violation + (dPdt - external_forces).pow(2).mean()
                else:
                    # dP/dt = 0
                    violation = violation + dPdt.pow(2).mean()
            else:
                # Variance across batch
                violation = violation + torch.var(linear_momentum, dim=0).sum()
        
        # Angular momentum
        if self.conserve_angular:
            angular_momentum = self.compute_angular_momentum(
                state['positions'],
                state['velocities'],
                state['masses']
            )
            
            if state_prev is not None:
                angular_prev = self.compute_angular_momentum(
                    state_prev['positions'],
                    state_prev['velocities'],
                    state_prev['masses']
                )
                dLdt = (angular_momentum - angular_prev) / dt
                
                if self.include_external_forces and external_torques is not None:
                    # dL/dt = tau_ext
                    violation = violation + (dLdt - external_torques).pow(2).mean()
                else:
                    # dL/dt = 0
                    violation = violation + dLdt.pow(2).mean()
            else:
                # Variance across batch
                violation = violation + torch.var(angular_momentum, dim=0).sum()
        
        return violation
    
    def project_to_manifold(
        self,
        state: Dict[str, torch.Tensor],
        reference_linear: Optional[torch.Tensor] = None,
        reference_angular: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Project state onto momentum conservation manifold.
        
        Adjusts center of mass velocity and removes rotational components.
        
        Args:
            state: Current state
            reference_linear: Target linear momentum
            reference_angular: Target angular momentum
            
        Returns:
            Projected state
        """
        projected_state = state.copy()
        
        # Center of mass correction
        if self.conserve_linear:
            current_linear = self.compute_linear_momentum(
                state['velocities'],
                state['masses']
            )
            
            if reference_linear is None:
                reference_linear = torch.zeros_like(current_linear)
            
            # Adjust velocities to match reference momentum
            total_mass = state['masses'].sum(dim=1, keepdim=True)  # [batch, 1]
            velocity_correction = (
                (reference_linear - current_linear) / total_mass.unsqueeze(-1)
            )  # [batch, 3]
            
            projected_state['velocities'] = (
                state['velocities'] + velocity_correction.unsqueeze(1)
            )
        
        # Angular momentum correction (simplified)
        if self.conserve_angular:
            # This requires more sophisticated treatment
            # involving removal of rotational velocity components
            pass
        
        return projected_state


class MassConservation(ConservationLaw):
    """
    Mass conservation (continuity equation).
    
    For compressible/incompressible flow:
    ∂ρ/∂t + ∇·(ρv) = 0
    """
    
    def __init__(
        self,
        discretization: str = 'finite_difference',
        grid_spacing: Optional[float] = None
    ):
        """
        Initialize mass conservation.
        
        Args:
            discretization: Discretization scheme ('finite_difference', 'spectral')
            grid_spacing: Grid spacing for finite differences
        """
        self.discretization = discretization
        self.grid_spacing = grid_spacing or 1.0
        
    def compute_divergence(
        self,
        field: torch.Tensor,
        spacing: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute divergence of a vector field.
        
        Args:
            field: [batch, n_points, 3] or [batch, nx, ny, nz, 3]
            spacing: Grid spacing
            
        Returns:
            Divergence [batch, n_points] or [batch, nx, ny, nz]
        """
        h = spacing or self.grid_spacing
        
        if field.dim() == 3:
            # Unstructured: use finite differences
            # This is a simplified implementation
            divergence = torch.zeros(field.shape[0], field.shape[1])
            
            # Central differences for each component
            for i in range(3):
                grad = torch.gradient(field[..., i], spacing=h, dim=1)[0]
                divergence = divergence + grad
                
        elif field.dim() == 5:
            # Structured 3D grid
            # Use PyTorch gradient function
            div_x = torch.gradient(field[..., 0], spacing=h, dim=1)[0]
            div_y = torch.gradient(field[..., 1], spacing=h, dim=2)[0]
            div_z = torch.gradient(field[..., 2], spacing=h, dim=3)[0]
            divergence = div_x + div_y + div_z
            
        return divergence
    
    def compute_invariant(
        self,
        state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute total mass.
        
        Args:
            state: Dictionary containing 'densities' and optionally 'volumes'
            
        Returns:
            Total mass [batch]
        """
        densities = state['densities']
        
        if 'volumes' in state:
            volumes = state['volumes']
            total_mass = torch.sum(densities * volumes, dim=-1)
        else:
            # Assume uniform grid
            total_mass = torch.sum(densities, dim=-1)
            
        return total_mass
    
    def compute_violation(
        self,
        state: Dict[str, torch.Tensor],
        state_prev: Optional[Dict[str, torch.Tensor]] = None,
        dt: float = 0.001,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute mass conservation violation.
        
        Args:
            state: Current state
            state_prev: Previous state
            dt: Time step
            
        Returns:
            Mass conservation violation
        """
        densities = state['densities']
        velocities = state['velocities']
        
        if state_prev is not None:
            # ∂ρ/∂t + ∇·(ρv) = 0
            densities_prev = state_prev['densities']
            d_rho_dt = (densities - densities_prev) / dt
        else:
            # Use variance as proxy
            return torch.var(densities, dim=-1).mean()
        
        # Compute ∇·(ρv)
        rho_v = densities.unsqueeze(-1) * velocities  # [batch, n, 3]
        divergence = self.compute_divergence(rho_v)
        
        # Continuity residual
        residual = d_rho_dt + divergence
        
        return residual.pow(2).mean()
    
    def project_to_manifold(
        self,
        state: Dict[str, torch.Tensor],
        reference_mass: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Project density field onto mass conservation manifold.
        
        Args:
            state: Current state
            reference_mass: Target total mass
            
        Returns:
            Projected state
        """
        current_mass = self.compute_invariant(state)
        
        if reference_mass is None:
            reference_mass = current_mass.mean()
        
        # Scale densities to match reference mass
        ratio = reference_mass / (current_mass + 1e-8)
        
        projected_state = state.copy()
        projected_state['densities'] = state['densities'] * ratio.unsqueeze(-1)
        
        return projected_state


class SymplecticConservation(ConservationLaw):
    """
    Symplectic structure conservation for Hamiltonian systems.
    
    Preserves the symplectic 2-form in phase space.
    """
    
    def __init__(self, n_dof: int):
        """
        Initialize symplectic conservation.
        
        Args:
            n_dof: Number of degrees of freedom
        """
        self.n_dof = n_dof
        
        # Symplectic matrix J = [[0, I], [-I, 0]]
        self.J = self._create_symplectic_matrix()
        
    def _create_symplectic_matrix(self) -> torch.Tensor:
        """Create symplectic matrix."""
        n = self.n_dof
        J = torch.zeros(2 * n, 2 * n)
        J[:n, n:] = torch.eye(n)
        J[n:, :n] = -torch.eye(n)
        return J
    
    def compute_invariant(
        self,
        state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute symplectic 2-form.
        
        ω = dq ∧ dp
        """
        # This is a simplified version
        # Full implementation requires differential forms
        q = state.get('positions')
        p = state.get('momenta')
        
        if q is None or p is None:
            return torch.tensor(0.0)
        
        # Simplified symplectic form
        omega = torch.sum(q * p, dim=-1)
        return omega
    
    def compute_violation(
        self,
        state: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """
        Compute symplectic violation.
        
        For symplectic integrators, this should be near machine precision.
        """
        # Simplified check: dω/dt = 0
        omega = self.compute_invariant(state)
        return torch.var(omega)
    
    def project_to_manifold(
        self,
        state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Project onto symplectic manifold.
        
        This typically requires symplectic integration schemes.
        """
        # Placeholder for symplectic projection
        return state


class ConstraintComposition:
    """
    Compose multiple conservation laws.
    
    Allows simultaneous enforcement of multiple physics constraints.
    """
    
    def __init__(self, constraints: List[ConservationLaw]):
        """
        Initialize constraint composition.
        
        Args:
            constraints: List of conservation laws
        """
        self.constraints = constraints
        
    def compute_total_violation(
        self,
        state: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total violation across all constraints.
        
        Args:
            state: Current state
            weights: Optional weighting for each constraint
            
        Returns:
            (total_violation, individual_violations)
        """
        total = torch.tensor(0.0)
        individual = {}
        
        for i, constraint in enumerate(self.constraints):
            name = f"constraint_{i}"
            violation = constraint.compute_violation(state, **kwargs)
            individual[name] = violation
            
            weight = weights.get(name, 1.0) if weights else 1.0
            total = total + weight * violation
            
        return total, individual
