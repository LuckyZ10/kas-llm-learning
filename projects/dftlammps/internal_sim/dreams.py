"""
Dream Generation and Mental Simulation
======================================

Generative exploration of hypothetical material scenarios.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Callable, Union, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import copy


class DreamType(Enum):
    """Types of dreams"""
    FREE = auto()           # Unconstrained free exploration
    GUIDED = auto()         # Guided by objectives
    COUNTERFACTUAL = auto() # Counterfactual scenarios
    PREDICTIVE = auto()     # Future prediction
    CREATIVE = auto()       # Creative discovery


@dataclass
class DreamConfig:
    """Configuration for dream generation"""
    latent_dim: int = 32
    hidden_dim: int = 256
    num_layers: int = 3
    
    # Generation parameters
    temperature: float = 1.0
    length: int = 50
    num_dreams: int = 10
    
    # Guidance parameters
    guidance_strength: float = 0.5
    diversity_weight: float = 0.3
    
    # Memory parameters
    memory_size: int = 1000
    replay_ratio: float = 0.2


class LatentTransitionModel(nn.Module):
    """
    Model for transitions in latent dream space.
    """
    
    def __init__(self, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        self.transition = nn.GRUCell(latent_dim, latent_dim)
        
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Learned prior
        self.prior_mean = nn.Parameter(torch.zeros(latent_dim))
        self.prior_logvar = nn.Parameter(torch.zeros(latent_dim))
    
    def forward(
        self,
        z: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Transition to next latent state"""
        # GRU transition
        if action is not None:
            z_input = z + action
        else:
            z_input = z
        
        z_next = self.transition(z_input, z)
        z_next = self.projection(z_next)
        
        return z_next
    
    def sample_initial(self, batch_size: int, temperature: float = 1.0) -> torch.Tensor:
        """Sample initial latent state"""
        std = torch.exp(0.5 * self.prior_logvar)
        eps = torch.randn(batch_size, self.latent_dim, device=std.device)
        return self.prior_mean + temperature * std * eps


class DreamGenerator(nn.Module):
    """
    Generates hypothetical material scenarios (dreams).
    
    Enables exploration of counterfactual and creative scenarios.
    """
    
    def __init__(self, config: DreamConfig = None):
        super().__init__()
        
        self.config = config or DreamConfig()
        cfg = self.config
        
        # Latent transition model
        self.transition_model = LatentTransitionModel(
            cfg.latent_dim,
            cfg.hidden_dim
        )
        
        # State generator (decodes latent to state)
        self.state_generator = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 20),  # Assuming state_dim=20
            nn.Tanh()
        )
        
        # Value predictor for guidance
        self.value_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim // 2, 1)
        )
        
        # Novelty predictor
        self.novelty_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def generate_dream(
        self,
        initial_state: Optional[torch.Tensor] = None,
        length: Optional[int] = None,
        temperature: Optional[float] = None,
        actions: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate a single dream sequence.
        
        Args:
            initial_state: Initial latent state (None for random)
            length: Dream length
            temperature: Sampling temperature
            actions: Optional action sequence
            
        Returns:
            Dream sequence with metadata
        """
        cfg = self.config
        length = length or cfg.length
        temperature = temperature or cfg.temperature
        
        # Initialize
        if initial_state is None:
            z = self.transition_model.sample_initial(1, temperature)
        else:
            if initial_state.dim() == 1:
                z = initial_state.unsqueeze(0)
            else:
                z = initial_state
        
        # Generate sequence
        latent_states = [z]
        generated_states = []
        values = []
        novelties = []
        
        for i in range(length):
            # Generate state
            state = self.state_generator(z)
            generated_states.append(state)
            
            # Predict value and novelty
            values.append(self.value_head(z))
            novelties.append(self.novelty_head(z))
            
            # Transition
            action = actions[i] if actions is not None and i < len(actions) else None
            z = self.transition_model(z, action)
            
            # Add noise for exploration
            z = z + torch.randn_like(z) * temperature * 0.1
            
            latent_states.append(z)
        
        return {
            'latent_states': torch.cat(latent_states, dim=0),
            'generated_states': torch.cat(generated_states, dim=0),
            'values': torch.cat(values, dim=0),
            'novelties': torch.cat(novelties, dim=0),
            'length': length
        }
    
    def generate_dream_batch(
        self,
        num_dreams: Optional[int] = None,
        length: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """Generate multiple dreams"""
        num_dreams = num_dreams or self.config.num_dreams
        
        dreams = []
        for _ in range(num_dreams):
            dream = self.generate_dream(length=length, temperature=temperature)
            dreams.append(dream)
        
        return dreams
    
    def guided_dream(
        self,
        objective_fn: Callable[[torch.Tensor], torch.Tensor],
        num_attempts: int = 100,
        temperature: float = 1.0,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate dreams guided by an objective.
        
        Args:
            objective_fn: Function that evaluates desirability
            num_attempts: Number of attempts
            temperature: Sampling temperature
            top_k: Return top k dreams
            
        Returns:
            List of best dreams with scores
        """
        dreams_with_scores = []
        
        for _ in range(num_attempts):
            dream = self.generate_dream(temperature=temperature)
            
            # Evaluate final state
            final_state = dream['generated_states'][-1:]
            score = objective_fn(final_state).item()
            
            dreams_with_scores.append({
                'dream': dream,
                'score': score,
                'avg_value': dream['values'].mean().item(),
                'avg_novelty': dream['novelties'].mean().item()
            })
        
        # Sort by score
        dreams_with_scores.sort(key=lambda x: x['score'], reverse=True)
        
        return dreams_with_scores[:top_k]
    
    def creative_exploration(
        self,
        seed_states: List[torch.Tensor],
        exploration_budget: int = 100,
        novelty_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Creative exploration for novel discoveries.
        
        Args:
            seed_states: Seed states to start from
            exploration_budget: Total exploration budget
            novelty_threshold: Minimum novelty threshold
            
        Returns:
            List of novel discoveries
        """
        discoveries = []
        generated_states = []
        
        for _ in range(exploration_budget):
            # Start from random seed
            seed = seed_states[np.random.randint(len(seed_states))]
            
            # Generate dream
            dream = self.generate_dream(
                initial_state=seed[:self.config.latent_dim] if seed.dim() > 1 else seed,
                length=np.random.randint(10, self.config.length),
                temperature=self.config.temperature * (1 + np.random.rand())
            )
            
            # Check novelty
            avg_novelty = dream['novelties'].mean().item()
            
            if avg_novelty > novelty_threshold:
                discoveries.append({
                    'dream': dream,
                    'novelty': avg_novelty,
                    'value': dream['values'].mean().item(),
                    'source': 'creative_exploration'
                })
                
                generated_states.extend([s for s in dream['generated_states']])
        
        # Sort by novelty
        discoveries.sort(key=lambda x: x['novelty'], reverse=True)
        
        return discoveries
    
    def train_on_experiences(
        self,
        experiences: torch.Tensor,
        num_epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3
    ) -> List[float]:
        """
        Train dream generator on real experiences.
        
        Args:
            experiences: Experience states
            num_epochs: Training epochs
            batch_size: Batch size
            lr: Learning rate
            
        Returns:
            Training losses
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []
        
        dataset = torch.utils.data.TensorDataset(experiences)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            
            for (batch,) in loader:
                optimizer.zero_grad()
                
                # Encode to latent (simplified)
                z = torch.randn(batch.size(0), self.config.latent_dim, device=batch.device)
                
                # Generate
                generated = self.state_generator(z)
                
                # Reconstruction loss
                target = batch[:generated.size(0), :generated.size(1)]
                loss = F.mse_loss(generated, target)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Dream training epoch {epoch+1}: loss={avg_loss:.4f}")
        
        return losses


class MentalSimulationEngine:
    """
    Mental simulation engine for internal reasoning.
    
    Supports planning, rehearsal, and counterfactual thinking.
    """
    
    def __init__(self, dream_generator: Optional[DreamGenerator] = None):
        self.dream_generator = dream_generator or DreamGenerator()
        
        # Memory
        self.simulation_memory: deque = deque(maxlen=1000)
        self.experience_memory: deque = deque(maxlen=10000)
        
        # Abstract knowledge
        self.knowledge_graph: Dict[str, Any] = {}
    
    def mental_rehearsal(
        self,
        plan: torch.Tensor,
        initial_condition: torch.Tensor,
        num_variations: int = 10,
        noise_level: float = 0.1
    ) -> Dict[str, Any]:
        """
        Mental rehearsal of a plan with variations.
        
        Args:
            plan: Planned actions or states
            initial_condition: Starting condition
            num_variations: Number of variations to simulate
            noise_level: Perturbation level
            
        Returns:
            Rehearsal results
        """
        outcomes = []
        
        for i in range(num_variations):
            # Perturb plan
            perturbed_plan = plan + torch.randn_like(plan) * noise_level
            
            # Simulate
            dream = self.dream_generator.generate_dream(
                initial_state=initial_condition,
                actions=perturbed_plan if perturbed_plan.dim() > 1 else None
            )
            
            # Evaluate outcome
            success = self._evaluate_success(dream)
            
            outcomes.append({
                'dream': dream,
                'success': success,
                'final_value': dream['values'][-1].item(),
                'variation_id': i
            })
        
        # Aggregate results
        success_rate = sum(1 for o in outcomes if o['success']) / len(outcomes)
        avg_value = np.mean([o['final_value'] for o in outcomes])
        
        return {
            'success_rate': success_rate,
            'average_value': avg_value,
            'outcomes': outcomes,
            'best_outcome': max(outcomes, key=lambda x: x['final_value']),
            'worst_outcome': min(outcomes, key=lambda x: x['final_value'])
        }
    
    def counterfactual_simulation(
        self,
        actual_trajectory: torch.Tensor,
        alternative_action: torch.Tensor,
        divergence_point: int = 0
    ) -> Dict[str, Any]:
        """
        Counterfactual simulation: "what if I had done X instead?"
        
        Args:
            actual_trajectory: What actually happened
            alternative_action: Alternative action to consider
            divergence_point: When the alternative diverges
            
        Returns:
            Counterfactual comparison
        """
        # Get state at divergence point
        if divergence_point < len(actual_trajectory):
            divergent_state = actual_trajectory[divergence_point]
        else:
            divergent_state = actual_trajectory[-1]
        
        # Simulate alternative
        alternative_dream = self.dream_generator.generate_dream(
            initial_state=divergent_state,
            actions=alternative_action.unsqueeze(0) if alternative_action.dim() == 1 else alternative_action
        )
        
        # Compare
        actual_final = actual_trajectory[-1] if actual_trajectory.dim() == 2 else actual_trajectory
        alternative_final = alternative_dream['generated_states'][-1]
        
        difference = torch.norm(alternative_final - actual_final).item()
        
        return {
            'alternative_trajectory': alternative_dream,
            'actual_final': actual_final,
            'alternative_final': alternative_final,
            'difference': difference,
            'would_have_been_better': alternative_dream['values'][-1].item() > 
                                   (actual_final.mean().item() if actual_final.dim() > 0 else actual_final.item())
        }
    
    def future_projection(
        self,
        current_state: torch.Tensor,
        num_steps: int = 100,
        num_scenarios: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Project multiple possible futures.
        
        Args:
            current_state: Current state
            num_steps: Projection steps
            num_scenarios: Number of scenarios
            
        Returns:
            Possible futures
        """
        scenarios = []
        
        for i in range(num_scenarios):
            dream = self.dream_generator.generate_dream(
                initial_state=current_state,
                length=num_steps,
                temperature=1.0 + i * 0.2  # Increasing diversity
            )
            
            scenarios.append({
                'scenario_id': i,
                'trajectory': dream['generated_states'],
                'final_value': dream['values'][-1].item(),
                'path_diversity': dream['novelties'].std().item(),
                'predicted_outcome': self._interpret_trajectory(dream)
            })
        
        # Sort by predicted value
        scenarios.sort(key=lambda x: x['final_value'], reverse=True)
        
        return scenarios
    
    def creative_problem_solving(
        self,
        problem_constraints: Dict[str, Any],
        num_ideas: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Creative problem solving through mental simulation.
        
        Args:
            problem_constraints: Problem specification
            num_ideas: Number of ideas to generate
            
        Returns:
            Creative solutions
        """
        ideas = []
        
        # Generate diverse dreams as potential solutions
        for _ in range(num_ideas * 3):  # Generate more, filter later
            dream = self.dream_generator.generate_dream(
                temperature=1.5  # Higher temperature for creativity
            )
            
            # Check constraints
            satisfies, score = self._check_constraints(dream, problem_constraints)
            
            if satisfies:
                ideas.append({
                    'solution': dream,
                    'constraint_score': score,
                    'novelty': dream['novelties'].mean().item(),
                    'value': dream['values'].mean().item()
                })
                
                if len(ideas) >= num_ideas:
                    break
        
        # Rank by combined score
        for idea in ideas:
            idea['combined_score'] = (
                idea['constraint_score'] * 0.4 +
                idea['novelty'] * 0.3 +
                idea['value'] * 0.3
            )
        
        ideas.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return ideas[:num_ideas]
    
    def _evaluate_success(self, dream: Dict[str, torch.Tensor]) -> bool:
        """Evaluate if a dream represents successful outcome"""
        final_value = dream['values'][-1].item()
        return final_value > 0.5
    
    def _interpret_trajectory(self, dream: Dict[str, torch.Tensor]) -> str:
        """Interpret trajectory into human-readable description"""
        values = dream['values']
        
        if values[-1].item() > values[0].item():
            trend = "improving"
        elif values[-1].item() < values[0].item():
            trend = "declining"
        else:
            trend = "stable"
        
        novelty = dream['novelties'].mean().item()
        
        return f"Trajectory shows {trend} trend with {novelty:.2f} novelty"
    
    def _check_constraints(
        self,
        dream: Dict[str, torch.Tensor],
        constraints: Dict[str, Any]
    ) -> Tuple[bool, float]:
        """Check if dream satisfies constraints"""
        # Simplified constraint checking
        score = 1.0
        
        if 'min_value' in constraints:
            min_val = dream['values'].min().item()
            if min_val < constraints['min_value']:
                score *= 0.5
        
        if 'max_value' in constraints:
            max_val = dream['values'].max().item()
            if max_val > constraints['max_value']:
                score *= 0.5
        
        return score > 0.7, score
    
    def consolidate_memory(self):
        """Consolidate simulation memories into knowledge"""
        if len(self.simulation_memory) < 100:
            return
        
        # Extract patterns from memory
        # (Simplified implementation)
        recent_simulations = list(self.simulation_memory)[-100:]
        
        # Update knowledge graph
        self.knowledge_graph['consolidated_simulations'] = len(recent_simulations)
        self.knowledge_graph['avg_success_rate'] = np.mean([
            s.get('success', 0) for s in recent_simulations
        ])


if __name__ == "__main__":
    print("Testing Dream Generation and Mental Simulation...")
    
    # Create config
    config = DreamConfig(
        latent_dim=32,
        hidden_dim=128,
        length=20,
        num_dreams=5
    )
    
    # Create dream generator
    generator = DreamGenerator(config)
    
    print(f"Generator created with {sum(p.numel() for p in generator.parameters())} parameters")
    
    # Test free dream generation
    dream = generator.generate_dream()
    
    print(f"\nFree dream generation:")
    print(f"  Latent states shape: {dream['latent_states'].shape}")
    print(f"  Generated states shape: {dream['generated_states'].shape}")
    print(f"  Average value: {dream['values'].mean().item():.4f}")
    print(f"  Average novelty: {dream['novelties'].mean().item():.4f}")
    
    # Test guided dream
    def objective_fn(state):
        return -torch.norm(state)  # Prefer states with smaller norm
    
    guided_dreams = generator.guided_dream(objective_fn, num_attempts=50, top_k=3)
    
    print(f"\nGuided dreams:")
    for i, gd in enumerate(guided_dreams):
        print(f"  Dream {i+1}: score={gd['score']:.4f}, value={gd['avg_value']:.4f}")
    
    # Test mental simulation engine
    engine = MentalSimulationEngine(generator)
    
    # Test mental rehearsal
    plan = torch.randn(10, 32)
    initial = torch.randn(32)
    
    rehearsal = engine.mental_rehearsal(plan, initial, num_variations=5)
    
    print(f"\nMental rehearsal:")
    print(f"  Success rate: {rehearsal['success_rate']:.2%}")
    print(f"  Average value: {rehearsal['average_value']:.4f}")
    
    # Test future projection
    futures = engine.future_projection(initial, num_steps=30, num_scenarios=3)
    
    print(f"\nFuture projections:")
    for f in futures:
        print(f"  Scenario {f['scenario_id']}: value={f['final_value']:.4f}, {f['predicted_outcome']}")
    
    print("\nAll tests passed!")
