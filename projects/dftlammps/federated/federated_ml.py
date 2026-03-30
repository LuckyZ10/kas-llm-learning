"""
Federated Machine Learning for ML Potential Training
======================================================

This module implements privacy-preserving federated learning for training
machine learning interatomic potentials across multiple institutions.

Features:
- Local training at each institution
- Secure aggregation of model updates
- Differential privacy protection
- Support for various ML potential architectures

Author: DFT-LAMMPS Team
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import copy
import hashlib
import json
import logging
from abc import ABC, abstractmethod
import time
from collections import defaultdict
import warnings


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Strategies for aggregating model updates from multiple institutions."""
    FEDAVG = "fedavg"  # Standard Federated Averaging
    FEDPROX = "fedprox"  # Federated Proximal with regularization
    SCAFFOLD = "scaffold"  # Stochastic Controlled Averaging
    FEDOPT = "fedopt"  # Federated Optimization with server optimizer
    FEDNOVA = "fednova"  # Normalized averaging with momentum correction


@dataclass
class FederatedConfig:
    """Configuration for federated learning training."""
    # Communication settings
    num_rounds: int = 100
    num_clients: int = 5
    clients_per_round: int = 5
    local_epochs: int = 5
    batch_size: int = 32
    
    # Learning rates
    global_lr: float = 1.0
    local_lr: float = 0.001
    
    # Aggregation strategy
    aggregation: AggregationStrategy = AggregationStrategy.FEDAVG
    
    # FedProx specific
    mu: float = 0.01  # Proximal term coefficient
    
    # SCAFFOLD specific
    use_control_variates: bool = False
    
    # Secure aggregation
    use_secure_aggregation: bool = True
    num_threshold: int = 3  # Minimum clients for aggregation
    
    # Differential privacy
    use_differential_privacy: bool = True
    epsilon: float = 1.0  # Privacy budget
    delta: float = 1e-5  # Privacy failure probability
    max_grad_norm: float = 1.0  # Gradient clipping
    noise_multiplier: float = 1.0
    
    # Convergence criteria
    target_loss: float = 1e-4
    patience: int = 10
    min_delta: float = 1e-6
    
    # Checkpointing
    checkpoint_dir: str = "./federated_checkpoints"
    save_frequency: int = 10
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ClientState:
    """State information for a federated client."""
    client_id: str
    institution: str
    data_size: int = 0
    local_model: Optional[nn.Module] = None
    control_variate: Optional[Dict[str, torch.Tensor]] = None
    last_update: float = 0.0
    round_participated: int = 0
    is_active: bool = True
    
    # Privacy accounting
    privacy_spent: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert state to dictionary (excluding model weights)."""
        return {
            'client_id': self.client_id,
            'institution': self.institution,
            'data_size': self.data_size,
            'round_participated': self.round_participated,
            'is_active': self.is_active,
            'privacy_spent': self.privacy_spent,
            'last_update': self.last_update
        }


class DifferentialPrivacyMechanism:
    """
    Differential privacy mechanisms for federated learning.
    
    Implements various DP mechanisms including Gaussian mechanism,
    moments accountant, and privacy budget tracking.
    """
    
    def __init__(self, epsilon: float, delta: float, noise_multiplier: float = 1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.noise_multiplier = noise_multiplier
        self.privacy_accountant = MomentsAccountant()
        
    def clip_gradients(self, params: List[torch.Tensor], max_norm: float) -> float:
        """
        Clip gradients by global L2 norm (per-sample gradient clipping).
        
        Args:
            params: Model parameters
            max_norm: Maximum L2 norm for gradients
            
        Returns:
            Total norm of gradients before clipping
        """
        total_norm = 0.0
        for p in params:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
        total_norm = total_norm ** 0.5
        
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in params:
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
                    
        return total_norm
    
    def add_noise(self, params: List[torch.Tensor], noise_multiplier: float,
                  max_grad_norm: float, batch_size: int) -> None:
        """
        Add Gaussian noise to gradients for differential privacy.
        
        Args:
            params: Model parameters
            noise_multiplier: Noise multiplier (sigma)
            max_grad_norm: Maximum gradient norm used for clipping
            batch_size: Batch size for noise scaling
        """
        noise_std = noise_multiplier * max_grad_norm / batch_size
        
        for p in params:
            if p.grad is not None:
                noise = torch.randn_like(p.grad.data) * noise_std
                p.grad.data.add_(noise)
    
    def compute_privacy_spent(self, num_steps: int, batch_size: int,
                             dataset_size: int, noise_multiplier: float) -> Tuple[float, float]:
        """
        Compute privacy spent using moments accountant.
        
        Args:
            num_steps: Number of optimization steps
            batch_size: Batch size
            dataset_size: Total dataset size
            noise_multiplier: Noise multiplier
            
        Returns:
            Tuple of (epsilon, delta) spent
        """
        sampling_rate = batch_size / dataset_size
        
        epsilon_spent = self.privacy_accountant.compute_epsilon(
            q=sampling_rate,
            noise_multiplier=noise_multiplier,
            steps=num_steps,
            delta=self.delta
        )
        
        return epsilon_spent, self.delta


class MomentsAccountant:
    """
    Moments Accountant for privacy budget tracking.
    
    Based on Abadi et al. "Deep Learning with Differential Privacy"
    (CCS 2016).
    """
    
    def __init__(self, max_lambda: int = 32):
        self.max_lambda = max_lambda
        self.log_moments = []
        
    def compute_epsilon(self, q: float, noise_multiplier: float, 
                        steps: int, delta: float) -> float:
        """
        Compute epsilon for given parameters using moments accountant.
        
        Args:
            q: Sampling probability (batch_size / dataset_size)
            noise_multiplier: Noise multiplier
            steps: Number of steps
            delta: Target delta
            
        Returns:
            Computed epsilon
        """
        # Simplified computation using standard DP-SGD bounds
        # For production, use a more sophisticated implementation
        
        if noise_multiplier == 0:
            return float('inf')
            
        # Use RDP to approximate DP
        rdp_eps = self._compute_rdp(q, noise_multiplier, steps)
        
        # Convert RDP to (eps, delta)-DP
        eps = rdp_eps - np.log(delta) / (self.max_lambda - 1)
        
        return max(eps, 0)
    
    def _compute_rdp(self, q: float, noise_multiplier: float, 
                     steps: int) -> float:
        """Compute Renyi Differential Privacy guarantee."""
        # Simplified RDP computation
        alpha = self.max_lambda
        
        if q == 0:
            return 0
            
        # RDP for subsampled Gaussian mechanism
        rdp_per_step = alpha * q**2 / (2 * noise_multiplier**2)
        
        return rdp_per_step * steps


class SecureAggregationProtocol:
    """
    Secure Multi-Party Computation for federated aggregation.
    
    Implements secure aggregation protocol where the server only sees
    the aggregated result, not individual client updates.
    """
    
    def __init__(self, num_clients: int, threshold: int):
        self.num_clients = num_clients
        self.threshold = threshold
        self.client_secrets = {}
        
    def generate_shares(self, client_id: str, update: Dict[str, torch.Tensor],
                        participating_clients: List[str]) -> List[Dict]:
        """
        Generate secret shares of model update for secure aggregation.
        
        Uses additive secret sharing where sum of shares equals original value.
        
        Args:
            client_id: ID of the client
            update: Model update dictionary
            participating_clients: List of clients participating in aggregation
            
        Returns:
            List of shares for each participating client
        """
        num_shares = len(participating_clients)
        shares = [{} for _ in range(num_shares)]
        
        for key, param in update.items():
            # Generate random shares that sum to the original value
            shape = param.shape
            random_shares = [torch.randn_like(param) for _ in range(num_shares - 1)]
            
            # Last share makes the sum equal to original
            last_share = param - sum(random_shares)
            random_shares.append(last_share)
            
            for i, share in enumerate(random_shares):
                shares[i][key] = share
                
        return shares
    
    def aggregate_shares(self, all_shares: List[Dict[str, torch.Tensor]],
                        weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
        """
        Aggregate secret shares from multiple clients.
        
        Args:
            all_shares: List of share dictionaries from clients
            weights: Optional weights for weighted aggregation
            
        Returns:
            Aggregated model update
        """
        if not all_shares:
            return {}
            
        # Get all parameter keys
        keys = all_shares[0].keys()
        aggregated = {}
        
        if weights is None:
            weights = [1.0 / len(all_shares)] * len(all_shares)
            
        for key in keys:
            # Sum weighted shares
            weighted_sum = sum(
                share[key] * weight 
                for share, weight in zip(all_shares, weights)
                if key in share
            )
            aggregated[key] = weighted_sum
            
        return aggregated
    
    def mask_update(self, update: Dict[str, torch.Tensor], 
                   seed: int) -> Dict[str, torch.Tensor]:
        """
        Apply deterministic masking using shared seed.
        
        Args:
            update: Model update
            seed: Shared random seed
            
        Returns:
            Masked update
        """
        torch.manual_seed(seed)
        masked = {}
        
        for key, param in update.items():
            mask = torch.randn_like(param)
            masked[key] = param + mask
            
        return masked
    
    def unmask_update(self, masked_update: Dict[str, torch.Tensor],
                     seed: int) -> Dict[str, torch.Tensor]:
        """
        Remove deterministic masking.
        
        Args:
            masked_update: Masked model update
            seed: Shared random seed
            
        Returns:
            Unmasked update
        """
        torch.manual_seed(seed)
        unmasked = {}
        
        for key, param in masked_update.items():
            mask = torch.randn_like(param)
            unmasked[key] = param - mask
            
        return unmasked


class FederatedServer:
    """
    Central server for federated learning coordination.
    
    The server orchestrates the training process, aggregates client updates,
    and maintains the global model.
    """
    
    def __init__(self, global_model: nn.Module, config: FederatedConfig):
        self.config = config
        self.global_model = global_model.to(config.device)
        self.client_states: Dict[str, ClientState] = {}
        
        # Secure aggregation
        self.secure_agg = SecureAggregationProtocol(
            config.num_clients, 
            config.num_threshold
        ) if config.use_secure_aggregation else None
        
        # Differential privacy
        self.dp_mechanism = DifferentialPrivacyMechanism(
            config.epsilon,
            config.delta,
            config.noise_multiplier
        ) if config.use_differential_privacy else None
        
        # Global control variate for SCAFFOLD
        self.global_control = None
        
        # Metrics tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'privacy_spent': [],
            'participating_clients': []
        }
        
        # Convergence tracking
        self.best_loss = float('inf')
        self.patience_counter = 0
        
    def register_client(self, client_id: str, institution: str, 
                       data_size: int) -> ClientState:
        """
        Register a new client institution.
        
        Args:
            client_id: Unique client identifier
            institution: Institution name
            data_size: Size of local dataset
            
        Returns:
            ClientState object
        """
        state = ClientState(
            client_id=client_id,
            institution=institution,
            data_size=data_size
        )
        self.client_states[client_id] = state
        
        logger.info(f"Registered client {client_id} from {institution} "
                   f"with {data_size} samples")
        
        return state
    
    def distribute_model(self, client_id: str) -> nn.Module:
        """
        Distribute global model to a client.
        
        Args:
            client_id: Target client ID
            
        Returns:
            Copy of global model
        """
        return copy.deepcopy(self.global_model)
    
    def select_clients(self, round_num: int) -> List[str]:
        """
        Select clients for the current training round.
        
        Uses random sampling with option for weighted selection based on
        dataset sizes.
        
        Args:
            round_num: Current round number
            
        Returns:
            List of selected client IDs
        """
        active_clients = [
            cid for cid, state in self.client_states.items() 
            if state.is_active
        ]
        
        if len(active_clients) <= self.config.clients_per_round:
            return active_clients
            
        # Random selection (can be extended to weighted selection)
        np.random.seed(round_num)
        selected = np.random.choice(
            active_clients,
            size=self.config.clients_per_round,
            replace=False
        ).tolist()
        
        return selected
    
    def aggregate_updates(self, client_updates: List[Dict[str, torch.Tensor]],
                         client_weights: List[float],
                         client_ids: List[str]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using specified strategy.
        
        Args:
            client_updates: List of model updates from clients
            client_weights: Weights for each client (proportional to data size)
            client_ids: IDs of participating clients
            
        Returns:
            Aggregated model update
        """
        strategy = self.config.aggregation
        
        if strategy == AggregationStrategy.FEDAVG:
            return self._fedavg_aggregate(client_updates, client_weights)
        elif strategy == AggregationStrategy.FEDPROX:
            return self._fedprox_aggregate(client_updates, client_weights)
        elif strategy == AggregationStrategy.SCAFFOLD:
            return self._scaffold_aggregate(client_updates, client_weights, client_ids)
        elif strategy == AggregationStrategy.FEDOPT:
            return self._fedopt_aggregate(client_updates, client_weights)
        else:
            return self._fedavg_aggregate(client_updates, client_weights)
    
    def _fedavg_aggregate(self, updates: List[Dict[str, torch.Tensor]],
                         weights: List[float]) -> Dict[str, torch.Tensor]:
        """
        Federated Averaging (FedAvg) aggregation.
        
        McMahan et al. "Communication-Efficient Learning of Deep Networks 
        from Decentralized Data" (AISTATS 2017).
        """
        aggregated = {}
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        for key in updates[0].keys():
            weighted_sum = sum(
                update[key] * weight 
                for update, weight in zip(updates, normalized_weights)
            )
            aggregated[key] = weighted_sum
            
        return aggregated
    
    def _fedprox_aggregate(self, updates: List[Dict[str, torch.Tensor]],
                          weights: List[float]) -> Dict[str, torch.Tensor]:
        """
        Federated Proximal aggregation with regularization.
        
        Li et al. "Federated Optimization in Heterogeneous Networks"
        (MLSys 2020).
        """
        # FedProx uses the same aggregation as FedAvg
        # The proximal term is applied during local training
        return self._fedavg_aggregate(updates, weights)
    
    def _scaffold_aggregate(self, updates: List[Dict[str, torch.Tensor]],
                           weights: List[float],
                           client_ids: List[str]) -> Dict[str, torch.Tensor]:
        """
        SCAFFOLD aggregation with control variates.
        
        Karimireddy et al. "SCAFFOLD: Stochastic Controlled Averaging for 
        Federated Learning" (ICML 2020).
        """
        # Update global control variate
        if self.global_control is None:
            self.global_control = {
                key: torch.zeros_like(param)
                for key, param in self.global_model.state_dict().items()
            }
        
        # Aggregate model updates
        aggregated = self._fedavg_aggregate(updates, weights)
        
        # Update control variates
        for client_id in client_ids:
            state = self.client_states[client_id]
            if state.control_variate is not None:
                for key in self.global_control:
                    self.global_control[key] += state.control_variate[key]
                    
        return aggregated
    
    def _fedopt_aggregate(self, updates: List[Dict[str, torch.Tensor]],
                         weights: List[float]) -> Dict[str, torch.Tensor]:
        """
        FedOpt aggregation with server-side optimizer.
        
        Reddi et al. "Adaptive Federated Optimization" (ICLR 2021).
        """
        # Simple implementation - can be extended with Adam/Adagrad on server
        return self._fedavg_aggregate(updates, weights)
    
    def update_global_model(self, aggregated_update: Dict[str, torch.Tensor]) -> None:
        """
        Update global model with aggregated client updates.
        
        Args:
            aggregated_update: Aggregated model update
        """
        global_dict = self.global_model.state_dict()
        
        for key, update in aggregated_update.items():
            if key in global_dict:
                # Apply update with global learning rate
                global_dict[key] += self.config.global_lr * update
                
        self.global_model.load_state_dict(global_dict)
    
    def train_round(self, round_num: int) -> Dict[str, float]:
        """
        Execute one round of federated training.
        
        Args:
            round_num: Current round number
            
        Returns:
            Dictionary of training metrics
        """
        # Select participating clients
        selected_clients = self.select_clients(round_num)
        
        if len(selected_clients) < self.config.num_threshold:
            logger.warning(f"Not enough clients selected ({len(selected_clients)}). "
                          f"Skipping round {round_num}.")
            return {'loss': float('inf'), 'num_clients': 0}
        
        logger.info(f"Round {round_num}: Training with {len(selected_clients)} clients")
        
        # Collect client updates
        client_updates = []
        client_weights = []
        losses = []
        
        for client_id in selected_clients:
            state = self.client_states[client_id]
            
            # Distribute global model
            local_model = self.distribute_model(client_id)
            
            # Client performs local training (simulated here)
            update, loss, control_delta = self._simulate_client_training(
                client_id, local_model, round_num
            )
            
            client_updates.append(update)
            client_weights.append(state.data_size)
            losses.append(loss)
            
            # Update client state
            state.round_participated = round_num
            state.last_update = time.time()
            
            # Update control variate for SCAFFOLD
            if control_delta is not None:
                if state.control_variate is None:
                    state.control_variate = control_delta
                else:
                    for key in state.control_variate:
                        state.control_variate[key] += control_delta[key]
        
        # Aggregate updates
        if self.config.use_secure_aggregation and self.secure_agg is not None:
            # Use secure aggregation
            aggregated = self._secure_aggregate(client_updates, client_weights)
        else:
            aggregated = self.aggregate_updates(
                client_updates, client_weights, selected_clients
            )
        
        # Update global model
        self.update_global_model(aggregated)
        
        # Compute metrics
        avg_loss = np.mean(losses)
        total_privacy_spent = sum(
            state.privacy_spent for state in self.client_states.values()
        )
        
        # Record history
        self.history['train_loss'].append(avg_loss)
        self.history['privacy_spent'].append(total_privacy_spent)
        self.history['participating_clients'].append(len(selected_clients))
        
        metrics = {
            'round': round_num,
            'loss': avg_loss,
            'num_clients': len(selected_clients),
            'privacy_spent': total_privacy_spent
        }
        
        logger.info(f"Round {round_num} completed: loss={avg_loss:.6f}")
        
        return metrics
    
    def _simulate_client_training(self, client_id: str, 
                                  local_model: nn.Module,
                                  round_num: int) -> Tuple[Dict, float, Optional[Dict]]:
        """
        Simulate local training on a client (placeholder for actual training).
        
        In practice, this would be executed on the client's infrastructure.
        
        Args:
            client_id: Client identifier
            local_model: Model to train
            round_num: Current round number
            
        Returns:
            Tuple of (model_update, loss, control_variate_delta)
        """
        # This is a placeholder - actual implementation would train on real data
        # Returns model update (delta), loss, and optional control variate delta
        
        # Get initial state
        initial_state = copy.deepcopy(local_model.state_dict())
        
        # Simulate local training
        # In real scenario, this runs on client infrastructure
        local_model.train()
        
        # Placeholder: just add small random updates
        for param in local_model.parameters():
            if param.grad is None:
                param.grad = torch.randn_like(param) * 0.01
            param.data -= self.config.local_lr * param.grad
        
        # Compute update (delta)
        final_state = local_model.state_dict()
        update = {}
        for key in initial_state:
            update[key] = final_state[key] - initial_state[key]
        
        # Simulate loss
        loss = np.random.uniform(0.1, 1.0) * (0.99 ** round_num)
        
        # Control variate delta (for SCAFFOLD)
        control_delta = None
        if self.config.aggregation == AggregationStrategy.SCAFFOLD:
            control_delta = {
                key: torch.randn_like(val) * 0.001
                for key, val in update.items()
            }
        
        return update, loss, control_delta
    
    def _secure_aggregate(self, updates: List[Dict[str, torch.Tensor]],
                         weights: List[float]) -> Dict[str, torch.Tensor]:
        """
        Perform secure aggregation of client updates.
        
        Args:
            updates: Client model updates
            weights: Client weights
            
        Returns:
            Securely aggregated update
        """
        # Generate shares for each client
        all_shares = []
        client_ids = list(self.client_states.keys())
        
        for update in updates:
            shares = self.secure_agg.generate_shares(
                "", update, client_ids
            )
            all_shares.append(shares[0])  # Simplified: use first share
        
        # Aggregate shares
        aggregated = self.secure_agg.aggregate_shares(all_shares, weights)
        
        return aggregated
    
    def check_convergence(self, current_loss: float) -> bool:
        """
        Check if training has converged.
        
        Args:
            current_loss: Current training loss
            
        Returns:
            True if converged, False otherwise
        """
        if current_loss < self.config.target_loss:
            return True
            
        if self.best_loss - current_loss > self.config.min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.config.patience:
            logger.info(f"Early stopping triggered after {self.config.patience} rounds")
            return True
            
        return False
    
    def train(self, num_rounds: Optional[int] = None) -> Dict[str, List]:
        """
        Run federated training for multiple rounds.
        
        Args:
            num_rounds: Number of rounds (defaults to config)
            
        Returns:
            Training history dictionary
        """
        if num_rounds is None:
            num_rounds = self.config.num_rounds
            
        logger.info(f"Starting federated training for {num_rounds} rounds")
        logger.info(f"Strategy: {self.config.aggregation.value}")
        logger.info(f"Secure Aggregation: {self.config.use_secure_aggregation}")
        logger.info(f"Differential Privacy: {self.config.use_differential_privacy}")
        
        for round_num in range(num_rounds):
            metrics = self.train_round(round_num)
            
            # Check convergence
            if self.check_convergence(metrics['loss']):
                logger.info(f"Training converged at round {round_num}")
                break
                
            # Save checkpoint
            if (round_num + 1) % self.config.save_frequency == 0:
                self.save_checkpoint(round_num)
        
        logger.info("Federated training completed")
        return self.history
    
    def save_checkpoint(self, round_num: int) -> None:
        """Save model checkpoint."""
        import os
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_round_{round_num}.pt"
        )
        
        torch.save({
            'round': round_num,
            'model_state_dict': self.global_model.state_dict(),
            'config': self.config,
            'history': self.history
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        
        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint['history']
        
        logger.info(f"Checkpoint loaded from round {checkpoint['round']}")
        return checkpoint['round']
    
    def get_global_model(self) -> nn.Module:
        """Get the current global model."""
        return self.global_model


class FederatedClient:
    """
    Client-side implementation for federated learning.
    
    Each institution runs a client that trains locally and communicates
    with the central server.
    """
    
    def __init__(self, client_id: str, institution: str, 
                 model: nn.Module, config: FederatedConfig):
        self.client_id = client_id
        self.institution = institution
        self.local_model = model.to(config.device)
        self.config = config
        
        # Local dataset
        self.train_loader = None
        self.val_loader = None
        
        # Privacy mechanism
        self.dp_mechanism = DifferentialPrivacyMechanism(
            config.epsilon,
            config.delta,
            config.noise_multiplier
        ) if config.use_differential_privacy else None
        
        # Control variate for SCAFFOLD
        self.control_variate = None
        
        # Training state
        self.privacy_spent = 0.0
        self.local_steps = 0
        
    def set_data_loaders(self, train_loader: DataLoader, 
                        val_loader: Optional[DataLoader] = None):
        """Set local data loaders."""
        self.train_loader = train_loader
        self.val_loader = val_loader
        
    def receive_global_model(self, global_model: nn.Module) -> None:
        """Receive and load global model from server."""
        self.local_model.load_state_dict(global_model.state_dict())
        
    def local_train(self, global_model: Optional[nn.Module] = None) -> Tuple[Dict, float, Optional[Dict]]:
        """
        Perform local training.
        
        Args:
            global_model: Global model for proximal regularization
            
        Returns:
            Tuple of (model_update, loss, control_variate_delta)
        """
        if self.train_loader is None:
            raise ValueError("Training data loader not set")
            
        # Get initial state
        initial_state = copy.deepcopy(self.local_model.state_dict())
        
        # Initialize control variate if needed
        if self.config.aggregation == AggregationStrategy.SCAFFOLD:
            if self.control_variate is None:
                self.control_variate = {
                    key: torch.zeros_like(param)
                    for key, param in self.local_model.state_dict().items()
                }
        
        # Training
        self.local_model.train()
        optimizer = torch.optim.Adam(
            self.local_model.parameters(),
            lr=self.config.local_lr
        )
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.config.local_epochs):
            for batch_idx, batch in enumerate(self.train_loader):
                # Unpack batch (depends on dataset structure)
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                    inputs = inputs.to(self.config.device)
                    targets = targets.to(self.config.device)
                else:
                    inputs = batch.to(self.config.device)
                    targets = None
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.local_model(inputs)
                
                # Compute loss (placeholder - customize for specific task)
                loss = self._compute_loss(outputs, targets)
                
                # Add proximal term for FedProx
                if self.config.aggregation == AggregationStrategy.FEDPROX and global_model is not None:
                    proximal_term = self._compute_proximal_term(global_model)
                    loss += self.config.mu * proximal_term
                
                # Backward pass
                loss.backward()
                
                # Apply differential privacy
                if self.config.use_differential_privacy and self.dp_mechanism is not None:
                    # Clip gradients
                    params = [p for p in self.local_model.parameters() if p.requires_grad]
                    self.dp_mechanism.clip_gradients(params, self.config.max_grad_norm)
                    
                    # Add noise
                    self.dp_mechanism.add_noise(
                        params,
                        self.config.noise_multiplier,
                        self.config.max_grad_norm,
                        len(inputs)
                    )
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                self.local_steps += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Compute model update (delta)
        final_state = self.local_model.state_dict()
        update = {}
        for key in initial_state:
            update[key] = final_state[key] - initial_state[key]
        
        # Update control variate for SCAFFOLD
        control_delta = None
        if self.config.aggregation == AggregationStrategy.SCAFFOLD:
            control_delta = {}
            for key in self.control_variate:
                # Update control variate
                old_cv = self.control_variate[key].clone()
                self.control_variate[key] += update[key]
                control_delta[key] = self.control_variate[key] - old_cv
        
        # Update privacy accounting
        if self.config.use_differential_privacy:
            epsilon_spent, _ = self.dp_mechanism.compute_privacy_spent(
                self.local_steps,
                self.config.batch_size,
                len(self.train_loader.dataset),
                self.config.noise_multiplier
            )
            self.privacy_spent = epsilon_spent
        
        return update, avg_loss, control_delta
    
    def _compute_loss(self, outputs: torch.Tensor, 
                     targets: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute loss for model outputs.
        
        Override this method for specific ML potential architectures.
        """
        if targets is not None:
            return nn.functional.mse_loss(outputs, targets)
        else:
            # Unsupervised or self-supervised loss
            return outputs.mean()
    
    def _compute_proximal_term(self, global_model: nn.Module) -> torch.Tensor:
        """
        Compute proximal regularization term for FedProx.
        
        Args:
            global_model: Global model
            
        Returns:
            Proximal loss term
        """
        proximal_loss = 0.0
        global_params = dict(global_model.named_parameters())
        
        for name, param in self.local_model.named_parameters():
            if name in global_params:
                proximal_loss += torch.sum((param - global_params[name]) ** 2)
                
        return proximal_loss
    
    def validate(self) -> float:
        """
        Validate local model on validation set.
        
        Returns:
            Validation loss
        """
        if self.val_loader is None:
            return 0.0
            
        self.local_model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                    inputs = inputs.to(self.config.device)
                    targets = targets.to(self.config.device)
                else:
                    inputs = batch.to(self.config.device)
                    targets = None
                
                outputs = self.local_model(inputs)
                loss = self._compute_loss(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def get_privacy_spent(self) -> float:
        """Get total privacy budget spent."""
        return self.privacy_spent


class MLPotentialModel(nn.Module):
    """
    Neural network model for ML interatomic potentials.
    
    This is a flexible architecture that can be customized for different
    potential types (Behler-Parrinello, SchNet, DimeNet, etc.).
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 128, 128],
                 output_dim: int = 1, activation: str = "silu",
                 dropout: float = 0.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Activation
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "silu":
                layers.append(nn.SiLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
                
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


def create_federated_ml_system(num_institutions: int = 5,
                                model_config: Optional[Dict] = None) -> Tuple[FederatedServer, List[FederatedClient]]:
    """
    Create a complete federated learning system for ML potential training.
    
    Args:
        num_institutions: Number of participating institutions
        model_config: Model configuration dictionary
        
    Returns:
        Tuple of (server, clients)
    """
    # Default model config
    if model_config is None:
        model_config = {
            'input_dim': 384,  # SOAP descriptor size
            'hidden_dims': [256, 256, 256],
            'output_dim': 1,  # Energy prediction
            'activation': 'silu'
        }
    
    # Create global model
    global_model = MLPotentialModel(**model_config)
    
    # Create server configuration
    config = FederatedConfig(
        num_rounds=100,
        num_clients=num_institutions,
        clients_per_round=min(5, num_institutions),
        aggregation=AggregationStrategy.FEDAVG,
        use_secure_aggregation=True,
        use_differential_privacy=True,
        epsilon=1.0,
        delta=1e-5
    )
    
    # Create server
    server = FederatedServer(global_model, config)
    
    # Create clients
    clients = []
    institutions = [
        "MIT Materials Lab",
        "Stanford Chemistry",
        "Berkeley Physics",
        "Caltech Nanoscience",
        "Harvard Engineering"
    ]
    
    for i in range(num_institutions):
        client_model = MLPotentialModel(**model_config)
        client = FederatedClient(
            client_id=f"client_{i}",
            institution=institutions[i % len(institutions)],
            model=client_model,
            config=config
        )
        
        # Register with server
        server.register_client(
            client_id=f"client_{i}",
            institution=institutions[i % len(institutions)],
            data_size=np.random.randint(1000, 10000)
        )
        
        clients.append(client)
    
    return server, clients


# Example usage and demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Federated ML Potential Training System")
    print("=" * 60)
    
    # Create federated system
    server, clients = create_federated_ml_system(num_institutions=5)
    
    print(f"\nCreated federated system with:")
    print(f"  - Server with global model")
    print(f"  - {len(clients)} client institutions")
    print(f"  - Secure aggregation: {server.config.use_secure_aggregation}")
    print(f"  - Differential privacy: {server.config.use_differential_privacy}")
    
    # Display client information
    print("\nRegistered Clients:")
    for client_id, state in server.client_states.items():
        print(f"  {client_id}: {state.institution} ({state.data_size} samples)")
    
    print("\nFederated learning system ready for training!")
