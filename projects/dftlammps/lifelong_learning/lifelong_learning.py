"""
Lifelong Learning Module: Continuous Learning without Forgetting

This module implements lifelong learning capabilities:
- Catastrophic forgetting prevention
- Knowledge accumulation over time
- Skill composition and transfer

Author: AGI Materials Intelligence System
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import pickle
from pathlib import Path
import logging
from datetime import datetime
from copy import deepcopy
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """A single learning experience."""
    task_id: str
    input_data: np.ndarray
    output_data: np.ndarray
    task_type: str
    timestamp: str
    importance: float = 1.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class Skill:
    """Represents a learned skill."""
    skill_id: str
    name: str
    description: str
    task_type: str
    model_state: Dict
    performance_metrics: Dict
    learned_at: str
    last_used: str
    usage_count: int = 0
    parent_skills: List[str] = field(default_factory=list)
    child_skills: List[str] = field(default_factory=list)
    

class ExperienceReplayBuffer:
    """
    Stores and manages past experiences for rehearsal-based learning.
    """
    
    def __init__(self, capacity: int = 10000, strategy: str = "reservoir"):
        self.capacity = capacity
        self.strategy = strategy  # reservoir, ring_buffer, importance_sampling
        self.buffer = []
        self.experience_counts = defaultdict(int)
        self.importance_scores = defaultdict(float)
        
    def add_experience(self, experience: Experience):
        """Add an experience to the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            if self.strategy == "reservoir":
                # Reservoir sampling
                idx = np.random.randint(0, len(self.buffer))
                self.buffer[idx] = experience
            elif self.strategy == "importance_sampling":
                # Replace least important experience
                min_idx = min(range(len(self.buffer)), 
                             key=lambda i: self.buffer[i].importance)
                self.buffer[min_idx] = experience
            else:
                # Ring buffer - replace oldest
                self.buffer.pop(0)
                self.buffer.append(experience)
        
        self.experience_counts[experience.task_id] += 1
        self.importance_scores[experience.task_id] += experience.importance
    
    def sample_experiences(self, n_samples: int, 
                          task_ids: List[str] = None) -> List[Experience]:
        """Sample experiences from buffer."""
        if not self.buffer:
            return []
        
        candidates = self.buffer
        if task_ids:
            candidates = [e for e in self.buffer if e.task_id in task_ids]
        
        if not candidates:
            return []
        
        n_samples = min(n_samples, len(candidates))
        
        if self.strategy == "importance_sampling":
            # Sample based on importance
            weights = [e.importance for e in candidates]
            weights = np.array(weights) / sum(weights)
            indices = np.random.choice(len(candidates), size=n_samples, 
                                     replace=False, p=weights)
        else:
            indices = np.random.choice(len(candidates), size=n_samples, replace=False)
        
        return [candidates[i] for i in indices]
    
    def get_task_distribution(self) -> Dict[str, int]:
        """Get distribution of tasks in buffer."""
        return dict(self.experience_counts)
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get statistics about the buffer."""
        if not self.buffer:
            return {'size': 0, 'utilization': 0}
        
        task_types = defaultdict(int)
        for exp in self.buffer:
            task_types[exp.task_type] += 1
        
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity,
            'task_distribution': dict(task_types),
            'unique_tasks': len(self.experience_counts)
        }


class ElasticWeightConsolidation:
    """
    Implements EWC (Kirkpatrick et al., 2017) for preventing catastrophic forgetting.
    """
    
    def __init__(self, model: nn.Module, importance: float = 1000.0):
        self.model = model
        self.importance = importance
        self.fisher_information = {}
        self.optimal_params = {}
        self.task_count = 0
        
    def compute_fisher_information(self, dataloader: DataLoader,
                                   num_samples: int = 200):
        """
        Compute Fisher Information Matrix for current task.
        """
        self.model.eval()
        fisher = {name: torch.zeros_like(param) 
                 for name, param in self.model.named_parameters()}
        
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= num_samples:
                break
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = F.mse_loss(outputs, targets)
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2
        
        # Normalize
        for name in fisher:
            fisher[name] /= num_samples
        
        # Store Fisher information
        self.fisher_information[self.task_count] = fisher
        
        # Store optimal parameters
        self.optimal_params[self.task_count] = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
        
        self.task_count += 1
        
    def compute_ewc_loss(self) -> torch.Tensor:
        """
        Compute EWC regularization loss.
        """
        if self.task_count == 0:
            return torch.tensor(0.0)
        
        loss = torch.tensor(0.0)
        
        for task_id in range(self.task_count):
            for name, param in self.model.named_parameters():
                if name in self.fisher_information[task_id]:
                    fisher = self.fisher_information[task_id][name]
                    optimal = self.optimal_params[task_id][name]
                    loss += (fisher * (param - optimal) ** 2).sum()
        
        return self.importance * loss
    
    def get_task_importance(self, task_id: int) -> Dict[str, torch.Tensor]:
        """Get importance of each parameter for a task."""
        if task_id in self.fisher_information:
            return {
                name: fisher.sum().item()
                for name, fisher in self.fisher_information[task_id].items()
            }
        return {}


class ProgressiveNeuralNetwork:
    """
    Implements Progressive Neural Networks (Rusu et al., 2016) for skill transfer.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.columns = nn.ModuleList()  # One column per task
        self.adapters = nn.ModuleList()  # Adapters from previous columns
        
        self.task_columns = {}  # Map task_id to column index
        
    def add_column(self, task_id: str):
        """Add a new column for a new task."""
        col_idx = len(self.columns)
        self.task_columns[task_id] = col_idx
        
        # Create new column
        column = nn.ModuleDict({
            'layer1': nn.Linear(self.input_dim, self.hidden_dim),
            'layer2': nn.Linear(self.hidden_dim, self.hidden_dim),
            'output': nn.Linear(self.hidden_dim, self.output_dim)
        })
        self.columns.append(column)
        
        # Create adapters from all previous columns
        if col_idx > 0:
            adapters_for_col = nn.ModuleList([
                nn.Linear(self.hidden_dim, self.hidden_dim)
                for _ in range(col_idx)
            ])
            self.adapters.append(adapters_for_col)
        else:
            self.adapters.append(nn.ModuleList())
    
    def forward(self, x: torch.Tensor, task_id: str) -> torch.Tensor:
        """Forward pass through appropriate column."""
        if task_id not in self.task_columns:
            raise ValueError(f"Unknown task: {task_id}")
        
        col_idx = self.task_columns[task_id]
        column = self.columns[col_idx]
        
        # Base forward pass
        h1 = F.relu(column['layer1'](x))
        
        # Add lateral connections from previous columns
        if col_idx > 0:
            adapters = self.adapters[col_idx]
            for prev_col_idx in range(col_idx):
                prev_column = self.columns[prev_col_idx]
                prev_h1 = F.relu(prev_column['layer1'](x))
                h1 = h1 + F.relu(adapters[prev_col_idx](prev_h1))
        
        h2 = F.relu(column['layer2'](h1))
        output = column['output'](h2)
        
        return output
    
    def freeze_column(self, col_idx: int):
        """Freeze a column to prevent forgetting."""
        for param in self.columns[col_idx].parameters():
            param.requires_grad = False
        
        if col_idx < len(self.adapters):
            for adapter in self.adapters[col_idx]:
                for param in adapter.parameters():
                    param.requires_grad = False
    
    def get_parameters_for_task(self, task_id: str):
        """Get trainable parameters for a specific task."""
        if task_id not in self.task_columns:
            return []
        
        col_idx = self.task_columns[task_id]
        params = []
        
        # Add current column parameters
        for param in self.columns[col_idx].parameters():
            if param.requires_grad:
                params.append(param)
        
        # Add adapter parameters
        if col_idx < len(self.adapters):
            for adapter in self.adapters[col_idx]:
                for param in adapter.parameters():
                    if param.requires_grad:
                        params.append(param)
        
        return params


class KnowledgeGraph:
    """
    Graph-based knowledge storage for accumulating and connecting concepts.
    """
    
    def __init__(self):
        self.nodes = {}  # concept_id -> concept_data
        self.edges = defaultdict(list)  # concept_id -> [(target_id, relation_type, strength)]
        self.concept_embeddings = {}
        
    def add_concept(self, concept_id: str, concept_data: Dict,
                   embedding: Optional[np.ndarray] = None):
        """Add a concept to the knowledge graph."""
        self.nodes[concept_id] = {
            'data': concept_data,
            'added_at': datetime.now().isoformat(),
            'access_count': 0
        }
        
        if embedding is not None:
            self.concept_embeddings[concept_id] = embedding
    
    def add_relation(self, source_id: str, target_id: str,
                    relation_type: str, strength: float = 1.0):
        """Add a relation between concepts."""
        if source_id in self.nodes and target_id in self.nodes:
            self.edges[source_id].append((target_id, relation_type, strength))
            
            # Make bidirectional for symmetric relations
            if relation_type in ['similar', 'related', 'correlated']:
                self.edges[target_id].append((source_id, relation_type, strength))
    
    def find_related_concepts(self, concept_id: str,
                             relation_filter: str = None,
                             min_strength: float = 0.5) -> List[Tuple[str, str, float]]:
        """Find concepts related to a given concept."""
        if concept_id not in self.edges:
            return []
        
        related = []
        for target_id, rel_type, strength in self.edges[concept_id]:
            if strength >= min_strength:
                if relation_filter is None or rel_type == relation_filter:
                    related.append((target_id, rel_type, strength))
        
        return sorted(related, key=lambda x: x[2], reverse=True)
    
    def find_path(self, source_id: str, target_id: str,
                 max_depth: int = 5) -> Optional[List[Tuple[str, str]]]:
        """Find a path between two concepts using BFS."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
        
        visited = {source_id}
        queue = [(source_id, [])]
        
        while queue and max_depth > 0:
            current, path = queue.pop(0)
            
            if current == target_id:
                return path
            
            for next_id, rel_type, strength in self.edges.get(current, []):
                if next_id not in visited and strength > 0.3:
                    visited.add(next_id)
                    queue.append((next_id, path + [(current, rel_type, next_id)]))
            
            max_depth -= 1
        
        return None
    
    def merge_similar_concepts(self, similarity_threshold: float = 0.9):
        """Merge highly similar concepts based on embeddings."""
        concepts = list(self.concept_embeddings.keys())
        
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i+1:]:
                if c1 in self.nodes and c2 in self.nodes:
                    emb1 = self.concept_embeddings[c1]
                    emb2 = self.concept_embeddings[c2]
                    
                    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    
                    if similarity > similarity_threshold:
                        self._merge_concepts(c1, c2)
    
    def _merge_concepts(self, keep_id: str, merge_id: str):
        """Merge merge_id into keep_id."""
        # Transfer edges
        for target_id, rel_type, strength in self.edges.get(merge_id, []):
            self.add_relation(keep_id, target_id, rel_type, strength)
        
        # Update node data
        self.nodes[keep_id]['data'].update(self.nodes[merge_id]['data'])
        
        # Remove merged concept
        del self.nodes[merge_id]
        if merge_id in self.concept_embeddings:
            del self.concept_embeddings[merge_id]
    
    def query(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Query the knowledge graph using embedding similarity."""
        similarities = []
        
        for concept_id, embedding in self.concept_embeddings.items():
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((concept_id, similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        total_edges = sum(len(edges) for edges in self.edges.values())
        
        relation_types = defaultdict(int)
        for edges in self.edges.values():
            for _, rel_type, _ in edges:
                relation_types[rel_type] += 1
        
        return {
            'num_concepts': len(self.nodes),
            'num_edges': total_edges,
            'relation_types': dict(relation_types),
            'concepts_with_embeddings': len(self.concept_embeddings)
        }


class SkillComposer:
    """
    Composes simple skills into complex capabilities.
    """
    
    def __init__(self):
        self.skills = {}  # skill_id -> Skill
        self.composition_rules = []
        self.composition_history = []
        
    def register_skill(self, skill: Skill):
        """Register a new skill."""
        self.skills[skill.skill_id] = skill
        logger.info(f"Registered skill: {skill.name}")
    
    def compose_skills(self, skill_ids: List[str],
                      composition_type: str = "sequential") -> Optional[Skill]:
        """
        Compose multiple skills into a new skill.
        
        Args:
            skill_ids: List of skill IDs to compose
            composition_type: sequential, parallel, or hierarchical
        """
        if not all(sid in self.skills for sid in skill_ids):
            logger.error("Not all skills found")
            return None
        
        skills = [self.skills[sid] for sid in skill_ids]
        
        if composition_type == "sequential":
            return self._compose_sequential(skills)
        elif composition_type == "parallel":
            return self._compose_parallel(skills)
        elif composition_type == "hierarchical":
            return self._compose_hierarchical(skills)
        else:
            logger.error(f"Unknown composition type: {composition_type}")
            return None
    
    def _compose_sequential(self, skills: List[Skill]) -> Skill:
        """Compose skills sequentially (output of one is input to next)."""
        composed_id = f"composed_seq_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create composed model state
        composed_state = {
            'type': 'sequential',
            'skill_chain': [s.skill_id for s in skills],
            'states': [s.model_state for s in skills]
        }
        
        composed_skill = Skill(
            skill_id=composed_id,
            name=f"Sequential composition of {', '.join(s.name for s in skills)}",
            description=f"Sequential execution of {len(skills)} skills",
            task_type="composed",
            model_state=composed_state,
            performance_metrics=self._aggregate_metrics(skills),
            learned_at=datetime.now().isoformat(),
            last_used=datetime.now().isoformat(),
            parent_skills=[s.skill_id for s in skills]
        )
        
        self.composition_history.append({
            'type': 'sequential',
            'composed_id': composed_id,
            'components': [s.skill_id for s in skills],
            'timestamp': datetime.now().isoformat()
        })
        
        self.skills[composed_id] = composed_skill
        return composed_skill
    
    def _compose_parallel(self, skills: List[Skill]) -> Skill:
        """Compose skills in parallel (ensemble)."""
        composed_id = f"composed_par_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        composed_state = {
            'type': 'parallel',
            'skill_ensemble': [s.skill_id for s in skills],
            'states': [s.model_state for s in skills],
            'aggregation': 'mean'  # Could be learned
        }
        
        composed_skill = Skill(
            skill_id=composed_id,
            name=f"Parallel ensemble of {', '.join(s.name for s in skills)}",
            description=f"Parallel execution of {len(skills)} skills with aggregation",
            task_type="composed",
            model_state=composed_state,
            performance_metrics=self._aggregate_metrics(skills),
            learned_at=datetime.now().isoformat(),
            last_used=datetime.now().isoformat(),
            parent_skills=[s.skill_id for s in skills]
        )
        
        self.skills[composed_id] = composed_skill
        return composed_skill
    
    def _compose_hierarchical(self, skills: List[Skill]) -> Skill:
        """Compose skills hierarchically (some control others)."""
        composed_id = f"composed_hier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # First skill is the controller, others are executors
        controller = skills[0]
        executors = skills[1:]
        
        composed_state = {
            'type': 'hierarchical',
            'controller': controller.skill_id,
            'executors': [s.skill_id for s in executors],
            'states': [s.model_state for s in skills]
        }
        
        composed_skill = Skill(
            skill_id=composed_id,
            name=f"Hierarchical control: {controller.name} + executors",
            description=f"Hierarchical composition with {len(executors)} executors",
            task_type="composed",
            model_state=composed_state,
            performance_metrics=self._aggregate_metrics(skills),
            learned_at=datetime.now().isoformat(),
            last_used=datetime.now().isoformat(),
            parent_skills=[s.skill_id for s in skills]
        )
        
        self.skills[composed_id] = composed_skill
        return composed_skill
    
    def _aggregate_metrics(self, skills: List[Skill]) -> Dict:
        """Aggregate performance metrics from multiple skills."""
        if not skills:
            return {}
        
        # Average metrics
        all_metrics = defaultdict(list)
        for skill in skills:
            for key, value in skill.performance_metrics.items():
                if isinstance(value, (int, float)):
                    all_metrics[key].append(value)
        
        return {
            key: np.mean(values) for key, values in all_metrics.items()
        }
    
    def find_composable_skills(self, target_task: str) -> List[List[str]]:
        """Find combinations of skills that could solve a target task."""
        # Simple matching based on task types
        candidates = [
            sid for sid, skill in self.skills.items()
            if target_task in skill.task_type or skill.task_type in target_task
        ]
        
        # Return combinations of up to 3 skills
        compositions = []
        for r in range(1, min(4, len(candidates) + 1)):
            compositions.extend(list(itertools.combinations(candidates, r)))
        
        return [list(comp) for comp in compositions]
    
    def get_skill_tree(self, skill_id: str) -> Dict[str, Any]:
        """Get the composition tree for a skill."""
        if skill_id not in self.skills:
            return {}
        
        skill = self.skills[skill_id]
        tree = {
            'skill_id': skill_id,
            'name': skill.name,
            'task_type': skill.task_type,
            'parent_skills': []
        }
        
        for parent_id in skill.parent_skills:
            tree['parent_skills'].append(self.get_skill_tree(parent_id))
        
        return tree


class LifelongLearningSystem:
    """
    Main lifelong learning system integrating all components.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Forgetting prevention
        self.use_ewc = self.config.get('use_ewc', True)
        self.use_progressive = self.config.get('use_progressive', True)
        self.use_replay = self.config.get('use_replay', True)
        
        # Components
        self.replay_buffer = ExperienceReplayBuffer(
            capacity=self.config.get('replay_capacity', 10000),
            strategy=self.config.get('replay_strategy', 'importance_sampling')
        )
        
        self.knowledge_graph = KnowledgeGraph()
        self.skill_composer = SkillComposer()
        
        # Current model and EWC
        self.current_model = None
        self.ewc = None
        self.progressive_net = None
        
        # Task tracking
        self.learned_tasks = {}
        self.current_task_id = None
        self.task_sequence = []
        
        # Metrics
        self.learning_history = []
        
    def initialize_model(self, model: nn.Module):
        """Initialize the learning model."""
        self.current_model = model
        
        if self.use_ewc:
            self.ewc = ElasticWeightConsolidation(
                model, 
                importance=self.config.get('ewc_importance', 1000.0)
            )
    
    def learn_task(self, task_id: str,
                  train_data: Tuple[np.ndarray, np.ndarray],
                  task_type: str = "regression",
                  epochs: int = 100,
                  batch_size: int = 32) -> Dict[str, Any]:
        """
        Learn a new task while preventing forgetting of previous tasks.
        """
        logger.info(f"Learning task: {task_id}")
        
        self.current_task_id = task_id
        self.task_sequence.append(task_id)
        
        X_train, y_train = train_data
        
        # Convert to tensors
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.current_model.parameters(),
            lr=self.config.get('learning_rate', 0.001)
        )
        
        # Training loop
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                predictions = self.current_model(batch_x)
                task_loss = F.mse_loss(predictions, batch_y)
                
                # Add EWC regularization
                if self.use_ewc and self.ewc:
                    ewc_loss = self.ewc.compute_ewc_loss()
                    total_loss = task_loss + ewc_loss
                else:
                    total_loss = task_loss
                
                # Add replay loss
                if self.use_replay and len(self.replay_buffer.buffer) > 0:
                    replay_samples = self.replay_buffer.sample_experiences(min(32, len(self.replay_buffer.buffer)))
                    if replay_samples:
                        replay_loss = self._compute_replay_loss(replay_samples)
                        total_loss = total_loss + 0.5 * replay_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += task_loss.item()
            
            losses.append(epoch_loss / len(train_loader))
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: loss = {losses[-1]:.4f}")
        
        # Store experiences
        for i in range(len(X_train)):
            exp = Experience(
                task_id=task_id,
                input_data=X_train[i],
                output_data=y_train[i],
                task_type=task_type,
                timestamp=datetime.now().isoformat(),
                importance=1.0  # Could be based on loss
            )
            self.replay_buffer.add_experience(exp)
        
        # Compute Fisher information for EWC
        if self.use_ewc and self.ewc:
            self.ewc.compute_fisher_information(train_loader)
        
        # Store task info
        self.learned_tasks[task_id] = {
            'task_type': task_type,
            'train_samples': len(X_train),
            'final_loss': losses[-1],
            'learned_at': datetime.now().isoformat()
        }
        
        # Create skill
        skill = Skill(
            skill_id=f"skill_{task_id}",
            name=task_id,
            description=f"Learned skill for {task_id}",
            task_type=task_type,
            model_state={k: v.clone() for k, v in self.current_model.state_dict().items()},
            performance_metrics={'final_loss': losses[-1]},
            learned_at=datetime.now().isoformat(),
            last_used=datetime.now().isoformat()
        )
        self.skill_composer.register_skill(skill)
        
        # Log learning
        self.learning_history.append({
            'task_id': task_id,
            'task_type': task_type,
            'epochs': epochs,
            'final_loss': losses[-1],
            'replay_buffer_size': len(self.replay_buffer.buffer)
        })
        
        return {
            'task_id': task_id,
            'final_loss': losses[-1],
            'epochs_trained': epochs,
            'replay_buffer_utilization': self.replay_buffer.get_buffer_stats()['utilization']
        }
    
    def _compute_replay_loss(self, experiences: List[Experience]) -> torch.Tensor:
        """Compute loss on replay experiences."""
        if not experiences:
            return torch.tensor(0.0)
        
        total_loss = 0.0
        for exp in experiences:
            x = torch.FloatTensor(exp.input_data).unsqueeze(0)
            y = torch.FloatTensor(exp.output_data).unsqueeze(0)
            
            pred = self.current_model(x)
            loss = F.mse_loss(pred, y)
            total_loss += loss
        
        return total_loss / len(experiences)
    
    def evaluate_on_all_tasks(self, task_data: Dict[str, Tuple]) -> Dict[str, float]:
        """Evaluate performance on all learned tasks."""
        self.current_model.eval()
        
        results = {}
        for task_id, (X_test, y_test) in task_data.items():
            with torch.no_grad():
                X_t = torch.FloatTensor(X_test)
                y_t = torch.FloatTensor(y_test)
                
                predictions = self.current_model(X_t)
                loss = F.mse_loss(predictions, y_t).item()
                
                results[task_id] = loss
        
        return results
    
    def add_knowledge(self, concept_id: str, concept_data: Dict,
                     embedding: np.ndarray = None):
        """Add knowledge to the knowledge graph."""
        self.knowledge_graph.add_concept(concept_id, concept_data, embedding)
        
        # Auto-link to related concepts
        if embedding is not None:
            related = self.knowledge_graph.query(embedding, top_k=3)
            for related_id, similarity in related:
                if related_id != concept_id and similarity > 0.7:
                    self.knowledge_graph.add_relation(
                        concept_id, related_id, 'similar', similarity
                    )
    
    def compose_new_skill(self, skill_ids: List[str],
                         composition_type: str = "sequential") -> Optional[Skill]:
        """Compose existing skills into a new skill."""
        return self.skill_composer.compose_skills(skill_ids, composition_type)
    
    def get_lifelong_stats(self) -> Dict[str, Any]:
        """Get statistics about lifelong learning."""
        return {
            'total_tasks_learned': len(self.learned_tasks),
            'task_sequence': self.task_sequence,
            'replay_buffer_stats': self.replay_buffer.get_buffer_stats(),
            'knowledge_graph_stats': self.knowledge_graph.get_stats(),
            'registered_skills': len(self.skill_composer.skills),
            'learning_history': self.learning_history
        }
    
    def save(self, path: str):
        """Save lifelong learning system state."""
        state = {
            'config': self.config,
            'learned_tasks': self.learned_tasks,
            'task_sequence': self.task_sequence,
            'learning_history': self.learning_history,
            'replay_buffer': self.replay_buffer,
            'knowledge_graph_nodes': self.knowledge_graph.nodes,
            'knowledge_graph_edges': dict(self.knowledge_graph.edges),
            'skills': self.skill_composer.skills
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Lifelong learning system saved to {path}")
    
    def load(self, path: str):
        """Load lifelong learning system state."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.config = state['config']
        self.learned_tasks = state['learned_tasks']
        self.task_sequence = state['task_sequence']
        self.learning_history = state['learning_history']
        self.replay_buffer = state['replay_buffer']
        self.knowledge_graph.nodes = state['knowledge_graph_nodes']
        self.knowledge_graph.edges = defaultdict(list, state['knowledge_graph_edges'])
        self.skill_composer.skills = state['skills']
        
        logger.info(f"Lifelong learning system loaded from {path}")


def demonstrate_lifelong_learning():
    """Demonstrate lifelong learning capabilities."""
    print("=" * 60)
    print("Lifelong Learning System Demonstration")
    print("=" * 60)
    
    # Create simple model
    class SimpleModel(nn.Module):
        def __init__(self, input_dim=10, hidden_dim=64, output_dim=1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        
        def forward(self, x):
            return self.net(x)
    
    # Initialize system
    system = LifelongLearningSystem({
        'use_ewc': True,
        'use_replay': True,
        'replay_capacity': 1000,
        'ewc_importance': 1000.0,
        'learning_rate': 0.001
    })
    
    model = SimpleModel(input_dim=5, hidden_dim=32, output_dim=1)
    system.initialize_model(model)
    
    print("\nLearning multiple tasks sequentially...")
    
    # Task 1: Predict band gap
    np.random.seed(42)
    X1 = np.random.randn(100, 5)
    y1 = X1[:, 0] * 0.5 + X1[:, 1] * 0.3 + np.random.randn(100) * 0.1
    y1 = y1.reshape(-1, 1)
    
    result1 = system.learn_task("band_gap_prediction", (X1, y1), "regression", epochs=50)
    print(f"\nTask 1 (band_gap_prediction): loss = {result1['final_loss']:.4f}")
    
    # Task 2: Predict elastic modulus
    X2 = np.random.randn(100, 5)
    y2 = X2[:, 2] * 0.7 + X2[:, 3] * 0.2 + np.random.randn(100) * 0.1
    y2 = y2.reshape(-1, 1)
    
    result2 = system.learn_task("elastic_modulus", (X2, y2), "regression", epochs=50)
    print(f"Task 2 (elastic_modulus): loss = {result2['final_loss']:.4f}")
    
    # Task 3: Predict thermal conductivity
    X3 = np.random.randn(100, 5)
    y3 = X3[:, 1] * 0.4 + X3[:, 4] * 0.5 + np.random.randn(100) * 0.1
    y3 = y3.reshape(-1, 1)
    
    result3 = system.learn_task("thermal_conductivity", (X3, y3), "regression", epochs=50)
    print(f"Task 3 (thermal_conductivity): loss = {result3['final_loss']:.4f}")
    
    # Evaluate on all tasks
    print("\nEvaluating on all tasks...")
    test_data = {
        'band_gap_prediction': (X1[:20], y1[:20]),
        'elastic_modulus': (X2[:20], y2[:20]),
        'thermal_conductivity': (X3[:20], y3[:20])
    }
    
    results = system.evaluate_on_all_tasks(test_data)
    print("\nTest losses:")
    for task_id, loss in results.items():
        print(f"  {task_id}: {loss:.4f}")
    
    # Add knowledge
    print("\nAdding knowledge to knowledge graph...")
    system.add_knowledge(
        "semiconductors",
        {'description': 'Materials with band gaps between 0.1 and 4 eV'},
        embedding=np.random.randn(64)
    )
    system.add_knowledge(
        "metals",
        {'description': 'Materials with zero band gap'},
        embedding=np.random.randn(64)
    )
    system.add_knowledge(
        "insulators",
        {'description': 'Materials with large band gaps > 4 eV'},
        embedding=np.random.randn(64)
    )
    
    # Add relations
    system.knowledge_graph.add_relation("semiconductors", "metals", "related", 0.6)
    system.knowledge_graph.add_relation("semiconductors", "insulators", "related", 0.7)
    
    # Compose skills
    print("\nComposing skills...")
    composed = system.compose_new_skill(
        ["skill_band_gap_prediction", "skill_elastic_modulus"],
        composition_type="sequential"
    )
    if composed:
        print(f"Created composed skill: {composed.name}")
    
    # Get stats
    stats = system.get_lifelong_stats()
    print(f"\n{'='*60}")
    print("LIFELONG LEARNING STATISTICS")
    print(f"{'='*60}")
    print(f"Total tasks learned: {stats['total_tasks_learned']}")
    print(f"Replay buffer size: {stats['replay_buffer_stats']['size']}")
    print(f"Knowledge graph concepts: {stats['knowledge_graph_stats']['num_concepts']}")
    print(f"Registered skills: {stats['registered_skills']}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    
    return system


if __name__ == "__main__":
    demonstrate_lifelong_learning()
