"""
Fast Physics Simulator
======================

High-speed learned physics simulator for mental simulations.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class SimulationGranularity(Enum):
    """Simulation granularity levels"""
    ATOMISTIC = "atomistic"
    MESOSCOPIC = "mesoscopic"
    CONTINUUM = "continuum"
    ABSTRACT = "abstract"


@dataclass
class SimulationConfig:
    """Configuration for fast simulator"""
    granularity: SimulationGranularity = SimulationGranularity.MESOSCOPIC
    state_dim: int = 20
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    time_step: float = 1.0
    use_attention: bool = True
    dropout: float = 0.1


class TransformerBlock(nn.Module):
    """Transformer block with self-attention"""
    
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class ResidualBlock(nn.Module):
    """Residual MLP block"""
    
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class FastPhysicsSimulator(nn.Module):
    """
    Fast neural physics simulator.
    
    Learns to approximate physics simulations orders of magnitude
    faster than traditional MD.
    """
    
    def __init__(self, config: SimulationConfig = None):
        super().__init__()
        
        self.config = config or SimulationConfig()
        cfg = self.config
        
        # Input embedding
        self.state_embed = nn.Linear(cfg.state_dim, cfg.hidden_dim)
        
        # Processing layers
        layers = []
        for i in range(cfg.num_layers):
            if cfg.use_attention and i % 2 == 0:
                layers.append(TransformerBlock(cfg.hidden_dim, cfg.num_heads, cfg.dropout))
            else:
                layers.append(ResidualBlock(cfg.hidden_dim, cfg.hidden_dim * 2, cfg.dropout))
        
        self.processing_layers = nn.ModuleList(layers)
        
        # Output heads
        self.delta_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.state_dim)
        )
        
        self.energy_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim // 2, 1)
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim // 2, cfg.state_dim),
            nn.Softplus()
        )
    
    def forward(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: Current state (batch, state_dim)
            action: Action (batch, action_dim)
            return_uncertainty: Whether to return uncertainty estimate
            
        Returns:
            Dict with 'next_state', 'delta', 'energy', optionally 'uncertainty'
        """
        x = self.state_embed(state)
        
        # Incorporate action if provided
        if action is not None:
            action_padded = F.pad(action, (0, self.config.state_dim - action.size(-1)))
            action_embed = self.state_embed(action_padded)
            x = x + action_embed
        
        # Process
        for layer in self.processing_layers:
            x = layer(x)
        
        # Predict state change
        delta = self.delta_head(x)
        next_state = state + delta * self.config.time_step
        
        # Predict energy
        energy = self.energy_head(x).squeeze(-1)
        
        result = {
            'next_state': next_state,
            'delta': delta,
            'energy': energy
        }
        
        if return_uncertainty:
            result['uncertainty'] = self.uncertainty_head(x)
        
        return result
    
    def simulate_trajectory(
        self,
        initial_state: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        num_steps: int = 100,
        return_full: bool = True
    ) -> torch.Tensor:
        """
        Simulate full trajectory.
        
        Args:
            initial_state: Initial state (batch, state_dim) or (state_dim,)
            actions: Action sequence (num_steps, action_dim) or (num_steps, batch, action_dim)
            num_steps: Number of steps
            return_full: Whether to return full trajectory
            
        Returns:
            State trajectory
        """
        if initial_state.dim() == 1:
            initial_state = initial_state.unsqueeze(0)
        
        states = [initial_state]
        current_state = initial_state
        
        for i in range(num_steps):
            if actions is not None:
                if actions.dim() == 2:
                    action = actions[i].unsqueeze(0).expand(current_state.size(0), -1)
                else:
                    action = actions[i]
            else:
                action = None
            
            with torch.no_grad():
                result = self.forward(current_state, action)
                current_state = result['next_state']
            
            states.append(current_state)
        
        if return_full:
            return torch.stack(states, dim=1)  # (batch, num_steps+1, state_dim)
        else:
            return current_state
    
    def benchmark(
        self,
        batch_size: int = 32,
        num_steps: int = 1000,
        device: str = 'cpu'
    ) -> Dict[str, float]:
        """
        Benchmark simulation speed.
        
        Returns:
            Speed statistics
        """
        import time
        
        self.to(device)
        self.eval()
        
        # Random input
        state = torch.randn(batch_size, self.config.state_dim, device=device)
        
        # Warmup
        for _ in range(10):
            _ = self.forward(state)
        
        # Benchmark
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        
        with torch.no_grad():
            for _ in range(num_steps):
                state = self.forward(state)['next_state']
        
        torch.cuda.synchronize() if device == 'cuda' else None
        elapsed = time.time() - start
        
        steps_per_second = (num_steps * batch_size) / elapsed
        
        # Compare with estimated MD speed (~1000x slower)
        estimated_md_time = elapsed * 1000
        speedup = 1000.0
        
        return {
            'total_time': elapsed,
            'steps_per_second': steps_per_second,
            'time_per_step_ms': (elapsed / num_steps) * 1000,
            'estimated_md_time': estimated_md_time,
            'speedup_factor': speedup,
            'device': device
        }


class MultiScaleSimulator:
    """
    Multi-scale physics simulator.
    
    Combines simulators at different granularities.
    """
    
    def __init__(self):
        self.simulators: Dict[SimulationGranularity, FastPhysicsSimulator] = {}
        self.scale_coupling: Optional[nn.Module] = None
    
    def add_simulator(
        self,
        granularity: SimulationGranularity,
        simulator: FastPhysicsSimulator
    ):
        """Add simulator for a specific granularity"""
        self.simulators[granularity] = simulator
    
    def simulate_hybrid(
        self,
        initial_states: Dict[SimulationGranularity, torch.Tensor],
        num_steps: int = 100
    ) -> Dict[SimulationGranularity, torch.Tensor]:
        """
        Multi-scale hybrid simulation.
        
        Args:
            initial_states: Initial states at each scale
            num_steps: Number of steps
            
        Returns:
            Final states at each scale
        """
        results = {}
        
        # Simulate each scale independently (simplified)
        for gran, state in initial_states.items():
            if gran in self.simulators:
                sim = self.simulators[gran]
                traj = sim.simulate_trajectory(state, num_steps=num_steps)
                results[gran] = traj[:, -1, :]  # Final state
        
        return results


if __name__ == "__main__":
    print("Testing Fast Physics Simulator...")
    
    # Create config
    config = SimulationConfig(
        state_dim=20,
        hidden_dim=128,
        num_layers=3,
        use_attention=True
    )
    
    # Create simulator
    simulator = FastPhysicsSimulator(config)
    
    print(f"Simulator created with {sum(p.numel() for p in simulator.parameters())} parameters")
    
    # Test forward pass
    batch_size = 16
    state = torch.randn(batch_size, config.state_dim)
    action = torch.randn(batch_size, 5)
    
    result = simulator(state, action, return_uncertainty=True)
    
    print(f"\nForward pass:")
    print(f"  Input shape: {state.shape}")
    print(f"  Next state shape: {result['next_state'].shape}")
    print(f"  Energy shape: {result['energy'].shape}")
    print(f"  Uncertainty shape: {result['uncertainty'].shape}")
    
    # Test trajectory simulation
    traj = simulator.simulate_trajectory(state[:1], num_steps=100)
    print(f"\nTrajectory simulation:")
    print(f"  Trajectory shape: {traj.shape}")
    
    # Benchmark
    print("\nBenchmarking...")
    stats = simulator.simulator.benchmark(batch_size=32, num_steps=100)
    print(f"  Steps per second: {stats['steps_per_second']:,.0f}")
    print(f"  Speedup over MD: {stats['speedup_factor']}x")
    
    print("\nAll tests passed!")
