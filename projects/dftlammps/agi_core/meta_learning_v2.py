"""
Meta-Learning V2: Learning to Learn for Materials Intelligence

This module implements advanced meta-learning capabilities for rapid adaptation
across different materials science tasks, enabling the system to:
- Learn from few examples (few-shot learning)
- Transfer knowledge across domains
- Rapidly adapt to new material types and properties

Author: AGI Materials Intelligence System
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import json
import pickle
from pathlib import Path
import logging
from abc import ABC, abstractmethod
import copy
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TaskConfig:
    """Configuration for a specific learning task."""
    task_name: str
    input_dim: int
    output_dim: int
    task_type: str = "regression"  # regression, classification, generation
    support_size: int = 10  # Number of support examples
    query_size: int = 20    # Number of query examples
    meta_lr: float = 0.001
    inner_lr: float = 0.01
    inner_steps: int = 5
    meta_batch_size: int = 4
    

@dataclass
class MetaLearningState:
    """State tracking for meta-learning progress."""
    episode: int = 0
    total_tasks_seen: int = 0
    adaptation_history: List[Dict] = field(default_factory=list)
    task_embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    performance_log: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    

class TaskDistribution:
    """Manages distribution of tasks for meta-training."""
    
    def __init__(self, task_configs: List[TaskConfig], seed: int = 42):
        self.task_configs = task_configs
        self.rng = np.random.RandomState(seed)
        self.task_statistics = defaultdict(lambda: {'count': 0, 'avg_loss': 0.0})
        
    def sample_task(self, n_samples: int = 1) -> List[TaskConfig]:
        """Sample task(s) from the distribution."""
        indices = self.rng.choice(len(self.task_configs), size=n_samples, replace=True)
        return [self.task_configs[i] for i in indices]
    
    def sample_support_query(self, task_config: TaskConfig, 
                            support_data: np.ndarray = None,
                            query_data: np.ndarray = None) -> Tuple[Tuple, Tuple]:
        """
        Sample support and query sets for a task.
        Returns: ((support_x, support_y), (query_x, query_y))
        """
        if support_data is None:
            # Generate synthetic data for demonstration
            support_x = self.rng.randn(task_config.support_size, task_config.input_dim)
            support_y = self._generate_labels(support_x, task_config)
        else:
            support_x, support_y = support_data
            
        if query_data is None:
            query_x = self.rng.randn(task_config.query_size, task_config.input_dim)
            query_y = self._generate_labels(query_x, task_config)
        else:
            query_x, query_y = query_data
            
        return (support_x, support_y), (query_x, query_y)
    
    def _generate_labels(self, x: np.ndarray, task_config: TaskConfig) -> np.ndarray:
        """Generate labels based on task type."""
        if task_config.task_type == "regression":
            # Simple linear relationship with noise
            w = self.rng.randn(task_config.input_dim, task_config.output_dim)
            y = x @ w + 0.1 * self.rng.randn(len(x), task_config.output_dim)
            return y
        elif task_config.task_type == "classification":
            # Binary classification
            w = self.rng.randn(task_config.input_dim)
            logits = x @ w
            y = (logits > 0).astype(np.int64)
            return y
        else:
            return self.rng.randn(len(x), task_config.output_dim)
    
    def update_task_stats(self, task_name: str, loss: float):
        """Update statistics for a task."""
        stats = self.task_statistics[task_name]
        stats['count'] += 1
        stats['avg_loss'] = (stats['avg_loss'] * (stats['count'] - 1) + loss) / stats['count']


class AdaptiveFeatureExtractor(nn.Module):
    """
    Feature extractor that adapts its architecture based on task characteristics.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 128], 
                 adaptation_rate: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.adaptation_rate = adaptation_rate
        
        # Base feature extraction layers
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim
        self.base_network = nn.Sequential(*layers)
        
        # Task-specific adaptation parameters (modulatory)
        self.task_gates = nn.ModuleList([
            nn.Linear(prev_dim, dim) for dim in hidden_dims
        ])
        
        # Feature importance estimator
        self.importance_estimator = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Linear(prev_dim // 2, prev_dim),
            nn.Sigmoid()
        )
        
        self.output_dim = prev_dim
        
    def forward(self, x: torch.Tensor, task_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional task-specific modulation."""
        features = self.base_network(x)
        
        # Apply task-specific modulation if provided
        if task_embedding is not None:
            # Compute feature importance
            importance = self.importance_estimator(features)
            features = features * importance
            
        return features
    
    def adapt_to_task(self, support_x: torch.Tensor, support_y: torch.Tensor,
                     n_steps: int = 5, lr: float = 0.01) -> Dict[str, torch.Tensor]:
        """
        Fast adaptation to a new task using support set.
        Returns adapted parameters.
        """
        # Create a copy of current parameters for adaptation
        adapted_params = {name: param.clone() 
                         for name, param in self.named_parameters()}
        
        # Perform gradient steps on support set
        for step in range(n_steps):
            self.zero_grad()
            features = self.forward(support_x)
            # Simple adaptation loss (can be customized)
            loss = F.mse_loss(features.mean(dim=1), support_y.float().mean(dim=1))
            loss.backward()
            
            # Manual parameter update
            with torch.no_grad():
                for name, param in self.named_parameters():
                    if param.grad is not None:
                        adapted_params[name] = adapted_params[name] - lr * param.grad
        
        return adapted_params


class TaskEmbeddingNetwork(nn.Module):
    """
    Learns to embed tasks into a continuous space for task similarity
    and transfer learning.
    """
    
    def __init__(self, feature_dim: int, embedding_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        
        # Task embedding encoder
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),  # Concatenate mean and std
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        
        # Task relationship predictor
        self.relationship_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def compute_task_embedding(self, support_x: torch.Tensor, 
                               support_y: torch.Tensor) -> torch.Tensor:
        """Compute embedding for a task from its support set."""
        # Compute statistics of the task
        x_mean = support_x.mean(dim=0)
        x_std = support_x.std(dim=0)
        y_mean = support_y.mean(dim=0) if len(support_y.shape) > 1 else support_y.float().mean()
        y_std = support_y.std(dim=0) if len(support_y.shape) > 1 else support_y.float().std()
        
        # Concatenate statistics
        stats = torch.cat([x_mean, x_std, y_mean.unsqueeze(0) if y_mean.dim() == 0 else y_mean,
                          y_std.unsqueeze(0) if y_std.dim() == 0 else y_std])
        
        # Pad or truncate to expected size
        expected_size = self.feature_dim * 2
        if stats.shape[0] < expected_size:
            stats = F.pad(stats, (0, expected_size - stats.shape[0]))
        else:
            stats = stats[:expected_size]
            
        embedding = self.encoder(stats)
        return embedding
    
    def predict_task_similarity(self, task1_embedding: torch.Tensor, 
                                task2_embedding: torch.Tensor) -> torch.Tensor:
        """Predict similarity between two tasks."""
        combined = torch.cat([task1_embedding, task2_embedding])
        return self.relationship_predictor(combined)


class MetaLearnerV2(nn.Module):
    """
    Advanced meta-learner implementing Model-Agnostic Meta-Learning (MAML)
    with enhancements for materials science tasks.
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 hidden_dims: List[int] = [128, 128, 128],
                 task_embedding_dim: int = 64,
                 meta_lr: float = 0.001,
                 inner_lr: float = 0.01,
                 inner_steps: int = 5):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        
        # Feature extractor
        self.feature_extractor = AdaptiveFeatureExtractor(
            input_dim, hidden_dims
        )
        
        # Task embedding network
        self.task_embedder = TaskEmbeddingNetwork(
            self.feature_extractor.output_dim, 
            task_embedding_dim
        )
        
        # Meta-model (base learner)
        self.meta_model = nn.Sequential(
            nn.Linear(self.feature_extractor.output_dim + task_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
        # Meta-optimizer
        self.meta_optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=meta_lr,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.meta_optimizer, 
            T_max=1000
        )
        
        # State tracking
        self.state = MetaLearningState()
        self.task_memory = {}  # Store learned task information
        
    def inner_loop(self, support_x: torch.Tensor, support_y: torch.Tensor,
                   task_config: TaskConfig) -> Dict[str, torch.Tensor]:
        """
        Perform inner loop adaptation on support set.
        Returns adapted parameters.
        """
        # Clone current parameters
        adapted_params = {}
        for name, param in self.named_parameters():
            adapted_params[name] = param.clone()
        
        # Compute task embedding
        task_embedding = self.task_embedder.compute_task_embedding(support_x, support_y)
        
        # Inner loop updates
        for step in range(task_config.inner_steps):
            # Forward pass with current adapted parameters
            features = self.feature_extractor(support_x)
            features_with_task = torch.cat([features, task_embedding.unsqueeze(0).expand(len(features), -1)], dim=1)
            predictions = self.meta_model(features_with_task)
            
            # Compute loss
            if task_config.task_type == "regression":
                loss = F.mse_loss(predictions, support_y)
            elif task_config.task_type == "classification":
                loss = F.cross_entropy(predictions, support_y.long())
            else:
                loss = F.mse_loss(predictions, support_y)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)
            
            # Update adapted parameters
            for (name, param), grad in zip(self.named_parameters(), grads):
                adapted_params[name] = adapted_params[name] - task_config.inner_lr * grad
                
        return adapted_params, task_embedding
    
    def forward_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor],
                           task_embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass using provided parameters."""
        # Manual forward pass with custom parameters
        features = self.feature_extractor(x)
        features_with_task = torch.cat([features, task_embedding.unsqueeze(0).expand(len(features), -1)], dim=1)
        
        # Apply meta_model layers manually
        x = F.linear(features_with_task, params['meta_model.0.weight'], params['meta_model.0.bias'])
        x = F.relu(x)
        x = F.linear(x, params['meta_model.2.weight'], params['meta_model.2.bias'])
        x = F.relu(x)
        x = F.linear(x, params['meta_model.4.weight'], params['meta_model.4.bias'])
        
        return x
    
    def meta_train_step(self, task_batch: List[Tuple]) -> Dict[str, float]:
        """
        Perform one meta-training step on a batch of tasks.
        Each task is a tuple of ((support_x, support_y), (query_x, query_y), task_config)
        """
        meta_loss = 0.0
        task_losses = []
        
        self.meta_optimizer.zero_grad()
        
        for (support_x, support_y), (query_x, query_y), task_config in task_batch:
            # Convert to tensors
            support_x = torch.FloatTensor(support_x)
            support_y = torch.FloatTensor(support_y)
            query_x = torch.FloatTensor(query_x)
            query_y = torch.FloatTensor(query_y)
            
            # Inner loop adaptation
            adapted_params, task_embedding = self.inner_loop(support_x, support_y, task_config)
            
            # Evaluate on query set
            predictions = self.forward_with_params(query_x, adapted_params, task_embedding)
            
            # Compute task loss
            if task_config.task_type == "regression":
                task_loss = F.mse_loss(predictions, query_y)
            elif task_config.task_type == "classification":
                task_loss = F.cross_entropy(predictions, query_y.long())
            else:
                task_loss = F.mse_loss(predictions, query_y)
                
            meta_loss += task_loss
            task_losses.append(task_loss.item())
            
            # Store task embedding
            self.state.task_embeddings[task_config.task_name] = task_embedding.detach().numpy()
        
        # Meta-update
        meta_loss = meta_loss / len(task_batch)
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.meta_optimizer.step()
        self.scheduler.step()
        
        # Update state
        self.state.episode += 1
        self.state.total_tasks_seen += len(task_batch)
        self.state.performance_log['meta_loss'].append(meta_loss.item())
        
        return {
            'meta_loss': meta_loss.item(),
            'avg_task_loss': np.mean(task_losses),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def adapt_to_new_task(self, support_x: np.ndarray, support_y: np.ndarray,
                         task_config: TaskConfig) -> Dict[str, Any]:
        """
        Fast adaptation to a completely new task.
        Returns adapted model state and performance metrics.
        """
        self.eval()
        with torch.no_grad():
            support_x_t = torch.FloatTensor(support_x)
            support_y_t = torch.FloatTensor(support_y)
            
            # Compute task embedding
            task_embedding = self.task_embedder.compute_task_embedding(support_x_t, support_y_t)
            
            # Find similar tasks
            similar_tasks = self._find_similar_tasks(task_embedding)
            
            # Perform adaptation
            adapted_state = self._fast_adapt(support_x_t, support_y_t, task_config)
            
        return {
            'task_embedding': task_embedding.numpy(),
            'similar_tasks': similar_tasks,
            'adapted_state': adapted_state,
            'adaptation_steps': task_config.inner_steps
        }
    
    def _find_similar_tasks(self, task_embedding: torch.Tensor, top_k: int = 3) -> List[Tuple[str, float]]:
        """Find most similar tasks based on task embeddings."""
        similarities = []
        for task_name, embedding in self.state.task_embeddings.items():
            emb_tensor = torch.FloatTensor(embedding)
            similarity = F.cosine_similarity(
                task_embedding.unsqueeze(0),
                emb_tensor.unsqueeze(0)
            ).item()
            similarities.append((task_name, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _fast_adapt(self, support_x: torch.Tensor, support_y: torch.Tensor,
                   task_config: TaskConfig) -> Dict[str, torch.Tensor]:
        """Perform fast adaptation to a new task."""
        adapted_params, _ = self.inner_loop(support_x, support_y, task_config)
        return adapted_params
    
    def predict(self, x: np.ndarray, task_embedding: Optional[np.ndarray] = None) -> np.ndarray:
        """Make predictions on new data."""
        self.eval()
        with torch.no_grad():
            x_t = torch.FloatTensor(x)
            if task_embedding is not None:
                task_emb_t = torch.FloatTensor(task_embedding)
                features = self.feature_extractor(x_t)
                features_with_task = torch.cat([features, task_emb_t.unsqueeze(0).expand(len(features), -1)], dim=1)
                predictions = self.meta_model(features_with_task)
            else:
                features = self.feature_extractor(x_t)
                predictions = self.meta_model(torch.cat([features, torch.zeros(len(features), self.task_embedder.embedding_dim)], dim=1))
            return predictions.numpy()
    
    def save(self, path: str):
        """Save meta-learner state."""
        state = {
            'model_state': self.state_dict(),
            'meta_optimizer': self.meta_optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'meta_learning_state': self.state,
            'task_memory': self.task_memory
        }
        torch.save(state, path)
        logger.info(f"Meta-learner saved to {path}")
    
    def load(self, path: str):
        """Load meta-learner state."""
        state = torch.load(path, map_location='cpu')
        self.load_state_dict(state['model_state'])
        self.meta_optimizer.load_state_dict(state['meta_optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.state = state['meta_learning_state']
        self.task_memory = state.get('task_memory', {})
        logger.info(f"Meta-learner loaded from {path}")


class CrossDomainTransfer:
    """
    Handles transfer learning across different domains in materials science.
    """
    
    def __init__(self, meta_learner: MetaLearnerV2):
        self.meta_learner = meta_learner
        self.domain_adaptation_layers = nn.ModuleDict()
        self.transfer_history = []
        
    def add_domain_adapter(self, source_domain: str, target_domain: str,
                          adapter_dim: int = 64):
        """Add domain adaptation layer between two domains."""
        key = f"{source_domain}_to_{target_domain}"
        self.domain_adaptation_layers[key] = nn.Sequential(
            nn.Linear(self.meta_learner.feature_extractor.output_dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, self.meta_learner.feature_extractor.output_dim)
        )
    
    def transfer_knowledge(self, source_task: str, target_task: str,
                          source_data: np.ndarray, target_support: Tuple) -> Dict[str, Any]:
        """
        Transfer knowledge from source task to target task.
        """
        source_x, source_y = source_data
        target_support_x, target_support_y = target_support
        
        # Get source task embedding
        with torch.no_grad():
            source_emb = self.meta_learner.task_embedder.compute_task_embedding(
                torch.FloatTensor(source_x),
                torch.FloatTensor(source_y)
            )
        
        # Check for domain adapter
        adapter_key = f"{source_task}_to_{target_task}"
        if adapter_key in self.domain_adaptation_layers:
            # Apply domain adaptation
            adapted_features = self.domain_adaptation_layers[adapter_key](source_emb)
        else:
            adapted_features = source_emb
        
        # Perform adaptation on target task
        adaptation_result = self.meta_learner.adapt_to_new_task(
            target_support_x, target_support_y,
            TaskConfig(task_name=target_task, 
                      input_dim=self.meta_learner.input_dim,
                      output_dim=self.meta_learner.output_dim)
        )
        
        # Log transfer
        self.transfer_history.append({
            'source': source_task,
            'target': target_task,
            'timestamp': datetime.now().isoformat(),
            'similarity': self.meta_learner.task_embedder.predict_task_similarity(
                source_emb,
                torch.FloatTensor(adaptation_result['task_embedding'])
            ).item()
        })
        
        return adaptation_result
    
    def get_transfer_matrix(self) -> np.ndarray:
        """Get transfer effectiveness matrix between all tasks."""
        task_names = list(self.meta_learner.state.task_embeddings.keys())
        n_tasks = len(task_names)
        matrix = np.zeros((n_tasks, n_tasks))
        
        for i, task1 in enumerate(task_names):
            for j, task2 in enumerate(task_names):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    emb1 = torch.FloatTensor(self.meta_learner.state.task_embeddings[task1])
                    emb2 = torch.FloatTensor(self.meta_learner.state.task_embeddings[task2])
                    similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
                    matrix[i, j] = similarity.item()
        
        return matrix, task_names


class FewShotLearner:
    """
    Specialized few-shot learning for materials discovery.
    """
    
    def __init__(self, meta_learner: MetaLearnerV2, n_way: int = 5, k_shot: int = 5):
        self.meta_learner = meta_learner
        self.n_way = n_way  # Number of classes
        self.k_shot = k_shot  # Shots per class
        self.prototype_cache = {}
        
    def compute_prototypes(self, support_x: np.ndarray, support_y: np.ndarray) -> Dict[int, np.ndarray]:
        """Compute class prototypes from support set."""
        self.meta_learner.eval()
        with torch.no_grad():
            features = self.meta_learner.feature_extractor(torch.FloatTensor(support_x))
            features_np = features.numpy()
        
        prototypes = {}
        for class_id in np.unique(support_y):
            mask = support_y == class_id
            prototypes[class_id] = features_np[mask].mean(axis=0)
        
        return prototypes
    
    def prototype_prediction(self, query_x: np.ndarray, 
                            prototypes: Dict[int, np.ndarray]) -> np.ndarray:
        """Make predictions using prototype-based classification."""
        self.meta_learner.eval()
        with torch.no_grad():
            query_features = self.meta_learner.feature_extractor(
                torch.FloatTensor(query_x)
            ).numpy()
        
        predictions = []
        for qf in query_features:
            # Compute distances to all prototypes
            distances = {
                class_id: np.linalg.norm(qf - proto)
                for class_id, proto in prototypes.items()
            }
            # Predict closest prototype
            pred_class = min(distances, key=distances.get)
            predictions.append(pred_class)
        
        return np.array(predictions)
    
    def evaluate_few_shot(self, support_set: Tuple, query_set: Tuple) -> Dict[str, float]:
        """Evaluate few-shot learning performance."""
        support_x, support_y = support_set
        query_x, query_y = query_set
        
        # Compute prototypes
        prototypes = self.compute_prototypes(support_x, support_y)
        
        # Make predictions
        predictions = self.prototype_prediction(query_x, prototypes)
        
        # Calculate metrics
        accuracy = (predictions == query_y).mean()
        
        return {
            'accuracy': accuracy,
            'n_classes': len(prototypes),
            'support_size': len(support_x),
            'query_size': len(query_x)
        }


class MetaLearningPipeline:
    """
    Complete meta-learning pipeline for materials science.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.meta_learner = None
        self.task_distribution = None
        self.cross_domain_transfer = None
        self.few_shot_learner = None
        
    def initialize(self, task_configs: List[TaskConfig], 
                  input_dim: int, output_dim: int):
        """Initialize the meta-learning pipeline."""
        # Create task distribution
        self.task_distribution = TaskDistribution(task_configs)
        
        # Create meta-learner
        self.meta_learner = MetaLearnerV2(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=self.config.get('hidden_dims', [128, 128, 128]),
            meta_lr=self.config.get('meta_lr', 0.001),
            inner_lr=self.config.get('inner_lr', 0.01),
            inner_steps=self.config.get('inner_steps', 5)
        )
        
        # Create cross-domain transfer module
        self.cross_domain_transfer = CrossDomainTransfer(self.meta_learner)
        
        # Create few-shot learner
        self.few_shot_learner = FewShotLearner(
            self.meta_learner,
            n_way=self.config.get('n_way', 5),
            k_shot=self.config.get('k_shot', 5)
        )
        
        logger.info("Meta-learning pipeline initialized")
        
    def train(self, n_episodes: int = 1000, tasks_per_episode: int = 4,
             eval_interval: int = 100) -> Dict[str, List]:
        """Train the meta-learner."""
        training_history = {
            'episodes': [],
            'meta_losses': [],
            'avg_task_losses': [],
            'evaluations': []
        }
        
        for episode in range(n_episodes):
            # Sample batch of tasks
            task_batch = []
            for _ in range(tasks_per_episode):
                task_config = self.task_distribution.sample_task(1)[0]
                support, query = self.task_distribution.sample_support_query(task_config)
                task_batch.append((support, query, task_config))
            
            # Meta-training step
            metrics = self.meta_learner.meta_train_step(task_batch)
            
            # Log progress
            if episode % 10 == 0:
                logger.info(f"Episode {episode}: meta_loss={metrics['meta_loss']:.4f}")
            
            training_history['episodes'].append(episode)
            training_history['meta_losses'].append(metrics['meta_loss'])
            training_history['avg_task_losses'].append(metrics['avg_task_loss'])
            
            # Evaluation
            if episode % eval_interval == 0 and episode > 0:
                eval_metrics = self.evaluate()
                training_history['evaluations'].append(eval_metrics)
                logger.info(f"Evaluation at episode {episode}: {eval_metrics}")
        
        return training_history
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate meta-learning performance."""
        # Sample test tasks
        test_tasks = self.task_distribution.sample_task(10)
        
        adaptation_losses = []
        for task_config in test_tasks:
            support, query = self.task_distribution.sample_support_query(task_config)
            
            # Adapt to task
            result = self.meta_learner.adapt_to_new_task(
                support[0], support[1], task_config
            )
            
            # Evaluate on query
            predictions = self.meta_learner.predict(query[0], result['task_embedding'])
            loss = np.mean((predictions - query[1]) ** 2)
            adaptation_losses.append(loss)
        
        return {
            'avg_adaptation_loss': np.mean(adaptation_losses),
            'std_adaptation_loss': np.std(adaptation_losses),
            'tasks_evaluated': len(test_tasks)
        }
    
    def quick_adapt(self, support_x: np.ndarray, support_y: np.ndarray,
                   task_name: str = "new_task") -> Dict[str, Any]:
        """Quick adaptation interface for new tasks."""
        task_config = TaskConfig(
            task_name=task_name,
            input_dim=support_x.shape[1],
            output_dim=1 if len(support_y.shape) == 1 else support_y.shape[1]
        )
        
        return self.meta_learner.adapt_to_new_task(support_x, support_y, task_config)
    
    def save(self, directory: str):
        """Save the entire pipeline."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save meta-learner
        self.meta_learner.save(path / "meta_learner.pt")
        
        # Save config
        with open(path / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save task distribution info
        task_info = {
            'task_statistics': dict(self.task_distribution.task_statistics)
        }
        with open(path / "task_info.json", 'w') as f:
            json.dump(task_info, f, indent=2, default=str)
        
        logger.info(f"Pipeline saved to {directory}")
    
    def load(self, directory: str):
        """Load the entire pipeline."""
        path = Path(directory)
        
        # Load config
        with open(path / "config.json", 'r') as f:
            self.config = json.load(f)
        
        # Load meta-learner
        if self.meta_learner is None:
            # Need to reinitialize with correct dimensions
            logger.info("Please reinitialize pipeline with correct dimensions before loading")
            return
        
        self.meta_learner.load(path / "meta_learner.pt")
        logger.info(f"Pipeline loaded from {directory}")


# Example usage and utilities
def create_materials_tasks() -> List[TaskConfig]:
    """Create example task configurations for materials science."""
    tasks = [
        # Crystal structure prediction tasks
        TaskConfig("crystal_structure", input_dim=100, output_dim=5, 
                  task_type="classification", support_size=20),
        # Property prediction tasks
        TaskConfig("bandgap_prediction", input_dim=100, output_dim=1,
                  task_type="regression", support_size=15),
        TaskConfig("elastic_modulus", input_dim=100, output_dim=3,
                  task_type="regression", support_size=10),
        TaskConfig("thermal_conductivity", input_dim=100, output_dim=1,
                  task_type="regression", support_size=12),
        # Phase stability tasks
        TaskConfig("phase_stability", input_dim=100, output_dim=1,
                  task_type="regression", support_size=18),
        # Defect formation energy
        TaskConfig("defect_energy", input_dim=100, output_dim=1,
                  task_type="regression", support_size=8),
    ]
    return tasks


if __name__ == "__main__":
    # Demo usage
    print("Meta-Learning V2 for Materials Intelligence")
    print("=" * 50)
    
    # Create tasks
    tasks = create_materials_tasks()
    print(f"Created {len(tasks)} material science tasks")
    
    # Initialize pipeline
    pipeline = MetaLearningPipeline({
        'hidden_dims': [128, 128],
        'meta_lr': 0.001,
        'inner_lr': 0.01,
        'inner_steps': 5
    })
    
    pipeline.initialize(tasks, input_dim=100, output_dim=1)
    print("Pipeline initialized successfully")
    
    # Quick adaptation demo
    support_x = np.random.randn(10, 100)
    support_y = np.random.randn(10, 1)
    
    result = pipeline.quick_adapt(support_x, support_y, "demo_task")
    print(f"Adaptation complete. Task embedding shape: {result['task_embedding'].shape}")
    print(f"Similar tasks found: {[t[0] for t in result['similar_tasks']]}")
