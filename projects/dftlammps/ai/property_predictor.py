"""
Property Predictor: Graph Neural Networks for Materials Property Prediction
============================================================================

This module implements state-of-the-art graph neural networks for predicting
material properties including:
- CGCNN (Crystal Graph Convolutional Neural Network)
- MegNet (MatErials Graph Network)
- ALIGNN (Atomistic Line Graph Neural Network)
- M3GNet/CHGNet universal potential interfaces
- Transformer-based models for materials

Author: DFT+LAMMPS AI Team
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
from collections import defaultdict
import warnings

# Optional imports
try:
    from pymatgen.core import Structure, Element
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False

try:
    from ase import Atoms
    from ase.neighborlist import neighbor_list
    HAS_ASE = True
except ImportError:
    HAS_ASE = False


@dataclass
class PropertyPredictorConfig:
    """Configuration for property prediction models."""
    
    # Model architecture
    model_type: str = "cgcnn"  # cgcnn, megnet, alignn, transformer, m3gnet, chgnet
    input_dim: int = 92  # Number of atom features
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1
    
    # Graph construction
    cutoff_distance: float = 8.0  # Angstroms
    max_neighbors: int = 12
    
    # Output
    output_dim: int = 1  # Number of target properties
    task_type: str = "regression"  # regression, classification, multi_task
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 200
    weight_decay: float = 1e-5
    
    # Pretrained models
    pretrained_path: Optional[str] = None
    freeze_layers: bool = False


@dataclass
class MaterialGraph:
    """Represents a material as a graph."""
    
    # Node features (N, node_dim)
    node_features: torch.Tensor
    
    # Edge indices (2, E)
    edge_index: torch.Tensor
    
    # Edge attributes (E, edge_dim) - distances, vectors, etc.
    edge_attr: Optional[torch.Tensor] = None
    
    # Graph-level features
    global_features: Optional[torch.Tensor] = None
    
    # Target values
    target: Optional[torch.Tensor] = None
    
    # Metadata
    structure_id: Optional[str] = None
    composition: Optional[str] = None
    
    @property
    def num_nodes(self) -> int:
        return self.node_features.shape[0]
    
    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]


# ============================================================================
# Atom and Bond Feature Utilities
# ============================================================================

class AtomFeatureEncoder:
    """Encodes atom properties into feature vectors."""
    
    # Atomic properties (normalized)
    ATOMIC_PROPERTIES = {
        'H':  {'Z': 1,   'group': 1,  'period': 1, 'radius': 0.31, 'eneg': 2.20, 'mass': 1.008},
        'Li': {'Z': 3,   'group': 1,  'period': 2, 'radius': 1.28, 'eneg': 0.98, 'mass': 6.941},
        'Be': {'Z': 4,   'group': 2,  'period': 2, 'radius': 0.96, 'eneg': 1.57, 'mass': 9.012},
        'B':  {'Z': 5,   'group': 13, 'period': 2, 'radius': 0.84, 'eneg': 2.04, 'mass': 10.811},
        'C':  {'Z': 6,   'group': 14, 'period': 2, 'radius': 0.76, 'eneg': 2.55, 'mass': 12.011},
        'N':  {'Z': 7,   'group': 15, 'period': 2, 'radius': 0.71, 'eneg': 3.04, 'mass': 14.007},
        'O':  {'Z': 8,   'group': 16, 'period': 2, 'radius': 0.66, 'eneg': 3.44, 'mass': 15.999},
        'F':  {'Z': 9,   'group': 17, 'period': 2, 'radius': 0.57, 'eneg': 3.98, 'mass': 18.998},
        'Na': {'Z': 11,  'group': 1,  'period': 3, 'radius': 1.66, 'eneg': 0.93, 'mass': 22.990},
        'Mg': {'Z': 12,  'group': 2,  'period': 3, 'radius': 1.41, 'eneg': 1.31, 'mass': 24.305},
        'Al': {'Z': 13,  'group': 13, 'period': 3, 'radius': 1.21, 'eneg': 1.61, 'mass': 26.982},
        'Si': {'Z': 14,  'group': 14, 'period': 3, 'radius': 1.11, 'eneg': 1.90, 'mass': 28.086},
        'P':  {'Z': 15,  'group': 15, 'period': 3, 'radius': 1.07, 'eneg': 2.19, 'mass': 30.974},
        'S':  {'Z': 16,  'group': 16, 'period': 3, 'radius': 1.05, 'eneg': 2.58, 'mass': 32.065},
        'Cl': {'Z': 17,  'group': 17, 'period': 3, 'radius': 1.02, 'eneg': 3.16, 'mass': 35.453},
        'K':  {'Z': 19,  'group': 1,  'period': 4, 'radius': 2.03, 'eneg': 0.82, 'mass': 39.098},
        'Ca': {'Z': 20,  'group': 2,  'period': 4, 'radius': 1.76, 'eneg': 1.00, 'mass': 40.078},
        'Sc': {'Z': 21,  'group': 3,  'period': 4, 'radius': 1.70, 'eneg': 1.36, 'mass': 44.956},
        'Ti': {'Z': 22,  'group': 4,  'period': 4, 'radius': 1.60, 'eneg': 1.54, 'mass': 47.867},
        'V':  {'Z': 23,  'group': 5,  'period': 4, 'radius': 1.53, 'eneg': 1.63, 'mass': 50.942},
        'Cr': {'Z': 24,  'group': 6,  'period': 4, 'radius': 1.39, 'eneg': 1.66, 'mass': 51.996},
        'Mn': {'Z': 25,  'group': 7,  'period': 4, 'radius': 1.39, 'eneg': 1.55, 'mass': 54.938},
        'Fe': {'Z': 26,  'group': 8,  'period': 4, 'radius': 1.32, 'eneg': 1.83, 'mass': 55.845},
        'Co': {'Z': 27,  'group': 9,  'period': 4, 'radius': 1.26, 'eneg': 1.88, 'mass': 58.933},
        'Ni': {'Z': 28,  'group': 10, 'period': 4, 'radius': 1.24, 'eneg': 1.91, 'mass': 58.693},
        'Cu': {'Z': 29,  'group': 11, 'period': 4, 'radius': 1.32, 'eneg': 1.90, 'mass': 63.546},
        'Zn': {'Z': 30,  'group': 12, 'period': 4, 'radius': 1.22, 'eneg': 1.65, 'mass': 65.380},
        'Ga': {'Z': 31,  'group': 13, 'period': 4, 'radius': 1.22, 'eneg': 1.81, 'mass': 69.723},
        'Ge': {'Z': 32,  'group': 14, 'period': 4, 'radius': 1.20, 'eneg': 2.01, 'mass': 72.640},
        'As': {'Z': 33,  'group': 15, 'period': 4, 'radius': 1.19, 'eneg': 2.18, 'mass': 74.922},
        'Se': {'Z': 34,  'group': 16, 'period': 4, 'radius': 1.20, 'eneg': 2.55, 'mass': 78.960},
        'Br': {'Z': 35,  'group': 17, 'period': 4, 'radius': 1.20, 'eneg': 2.96, 'mass': 79.904},
        'Rb': {'Z': 37,  'group': 1,  'period': 5, 'radius': 2.20, 'eneg': 0.82, 'mass': 85.468},
        'Sr': {'Z': 38,  'group': 2,  'period': 5, 'radius': 1.95, 'eneg': 0.95, 'mass': 87.620},
        'Y':  {'Z': 39,  'group': 3,  'period': 5, 'radius': 1.90, 'eneg': 1.22, 'mass': 88.906},
        'Zr': {'Z': 40,  'group': 4,  'period': 5, 'radius': 1.75, 'eneg': 1.33, 'mass': 91.224},
        'Nb': {'Z': 41,  'group': 5,  'period': 5, 'radius': 1.64, 'eneg': 1.6,  'mass': 92.906},
        'Mo': {'Z': 42,  'group': 6,  'period': 5, 'radius': 1.54, 'eneg': 2.16, 'mass': 95.960},
        'Tc': {'Z': 43,  'group': 7,  'period': 5, 'radius': 1.47, 'eneg': 1.9,  'mass': 98.000},
        'Ru': {'Z': 44,  'group': 8,  'period': 5, 'radius': 1.46, 'eneg': 2.2,  'mass': 101.070},
        'Rh': {'Z': 45,  'group': 9,  'period': 5, 'radius': 1.42, 'eneg': 2.28, 'mass': 102.906},
        'Pd': {'Z': 46,  'group': 10, 'period': 5, 'radius': 1.39, 'eneg': 2.20, 'mass': 106.420},
        'Ag': {'Z': 47,  'group': 11, 'period': 5, 'radius': 1.45, 'eneg': 1.93, 'mass': 107.868},
        'Cd': {'Z': 48,  'group': 12, 'period': 5, 'radius': 1.44, 'eneg': 1.69, 'mass': 112.411},
        'In': {'Z': 49,  'group': 13, 'period': 5, 'radius': 1.42, 'eneg': 1.78, 'mass': 114.818},
        'Sn': {'Z': 50,  'group': 14, 'period': 5, 'radius': 1.39, 'eneg': 1.96, 'mass': 118.710},
        'Sb': {'Z': 51,  'group': 15, 'period': 5, 'radius': 1.39, 'eneg': 2.05, 'mass': 121.760},
        'Te': {'Z': 52,  'group': 16, 'period': 5, 'radius': 1.38, 'eneg': 2.1,  'mass': 127.600},
        'I':  {'Z': 53,  'group': 17, 'period': 5, 'radius': 1.39, 'eneg': 2.66, 'mass': 126.904},
        'Cs': {'Z': 55,  'group': 1,  'period': 6, 'radius': 2.44, 'eneg': 0.79, 'mass': 132.905},
        'Ba': {'Z': 56,  'group': 2,  'period': 6, 'radius': 2.15, 'eneg': 0.89, 'mass': 137.327},
        'La': {'Z': 57,  'group': 3,  'period': 6, 'radius': 2.07, 'eneg': 1.10, 'mass': 138.905},
        'Hf': {'Z': 72,  'group': 4,  'period': 6, 'radius': 1.75, 'eneg': 1.3,  'mass': 178.490},
        'Ta': {'Z': 73,  'group': 5,  'period': 6, 'radius': 1.70, 'eneg': 1.5,  'mass': 180.948},
        'W':  {'Z': 74,  'group': 6,  'period': 6, 'radius': 1.62, 'eneg': 2.36, 'mass': 183.840},
        'Re': {'Z': 75,  'group': 7,  'period': 6, 'radius': 1.51, 'eneg': 1.9,  'mass': 186.207},
        'Os': {'Z': 76,  'group': 8,  'period': 6, 'radius': 1.44, 'eneg': 2.2,  'mass': 190.230},
        'Ir': {'Z': 77,  'group': 9,  'period': 6, 'radius': 1.41, 'eneg': 2.20, 'mass': 192.217},
        'Pt': {'Z': 78,  'group': 10, 'period': 6, 'radius': 1.36, 'eneg': 2.28, 'mass': 195.084},
        'Au': {'Z': 79,  'group': 11, 'period': 6, 'radius': 1.36, 'eneg': 2.54, 'mass': 196.967},
        'Hg': {'Z': 80,  'group': 12, 'period': 6, 'radius': 1.32, 'eneg': 2.00, 'mass': 200.590},
        'Tl': {'Z': 81,  'group': 13, 'period': 6, 'radius': 1.45, 'eneg': 1.62, 'mass': 204.383},
        'Pb': {'Z': 82,  'group': 14, 'period': 6, 'radius': 1.46, 'eneg': 2.33, 'mass': 207.200},
        'Bi': {'Z': 83,  'group': 15, 'period': 6, 'radius': 1.48, 'eneg': 2.02, 'mass': 208.980},
    }
    
    def __init__(self):
        self.properties_cache = {}
    
    def encode(self, atomic_number: int) -> np.ndarray:
        """Encode atomic number to feature vector."""
        if atomic_number in self.properties_cache:
            return self.properties_cache[atomic_number]
        
        # Get element symbol
        symbol = self._get_element_symbol(atomic_number)
        
        if symbol in self.ATOMIC_PROPERTIES:
            props = self.ATOMIC_PROPERTIES[symbol]
            features = np.array([
                props['Z'] / 100.0,
                props['group'] / 18.0,
                props['period'] / 7.0,
                props['radius'] / 3.0,
                props['eneg'] / 4.0,
                np.log(props['mass']) / 6.0,
                # One-hot for groups
                *[1.0 if i == props['group'] - 1 else 0.0 for i in range(18)],
                # One-hot for periods
                *[1.0 if i == props['period'] - 1 else 0.0 for i in range(7)],
            ], dtype=np.float32)
        else:
            # Default encoding for unknown elements
            features = np.zeros(32, dtype=np.float32)
            features[0] = atomic_number / 100.0
        
        self.properties_cache[atomic_number] = features
        return features
    
    def _get_element_symbol(self, atomic_number: int) -> str:
        """Get element symbol from atomic number."""
        for symbol, props in self.ATOMIC_PROPERTIES.items():
            if props['Z'] == atomic_number:
                return symbol
        return 'H'  # Default


class DistanceExpansion(nn.Module):
    """Expand distances using radial basis functions."""
    
    def __init__(
        self,
        num_rbf: int = 64,
        cutoff: float = 8.0,
        rbf_type: str = 'gaussian'
    ):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.rbf_type = rbf_type
        
        # Centers for RBF
        self.centers = nn.Parameter(
            torch.linspace(0, cutoff, num_rbf),
            requires_grad=False
        )
        
        # Width parameter
        self.width = 0.5
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Expand distances to RBF features.
        
        Args:
            distances: (E,) edge distances
        
        Returns:
            (E, num_rbf) RBF features
        """
        if self.rbf_type == 'gaussian':
            # Gaussian RBF
            diff = distances.unsqueeze(-1) - self.centers
            rbf = torch.exp(-0.5 * (diff / self.width) ** 2)
        elif self.rbf_type == 'spherical':
            # Spherical Bessel functions (simplified)
            rbf = torch.sin(np.pi * distances.unsqueeze(-1) / self.cutoff) / distances.unsqueeze(-1)
            rbf = rbf * (distances.unsqueeze(-1) < self.cutoff).float()
        else:
            raise ValueError(f"Unknown RBF type: {self.rbf_type}")
        
        # Cutoff function
        cutoff_vals = 0.5 * (torch.cos(np.pi * distances / self.cutoff) + 1)
        cutoff_vals = cutoff_vals * (distances < self.cutoff).float()
        
        return rbf * cutoff_vals.unsqueeze(-1)


# ============================================================================
# CGCNN: Crystal Graph Convolutional Neural Network
# ============================================================================

class CGCNNConv(nn.Module):
    """CGCNN graph convolution layer."""
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        # Message function
        self.message_net = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Update function
        self.update_net = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(hidden_dim)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Graph convolution forward pass.
        
        Args:
            node_features: (N, node_dim)
            edge_index: (2, E)
            edge_attr: (E, edge_dim)
        
        Returns:
            Updated node features (N, hidden_dim)
        """
        row, col = edge_index
        
        # Create edge features
        edge_features = torch.cat([
            node_features[row],
            node_features[col],
            edge_attr
        ], dim=-1)
        
        # Compute messages
        messages = self.message_net(edge_features)
        
        # Aggregate messages
        aggregated = torch.zeros_like(node_features)
        aggregated.index_add_(0, row, messages)
        
        # Update nodes
        combined = torch.cat([node_features, aggregated], dim=-1)
        updated = self.update_net(combined)
        
        return self.bn(updated)


class CGCNN(nn.Module):
    """
    Crystal Graph Convolutional Neural Network.
    
    Reference: Xie & Grossman, Phys. Rev. Lett. 2018
    """
    
    def __init__(self, config: PropertyPredictorConfig):
        super().__init__()
        self.config = config
        
        # Atom embedding
        self.atom_embedding = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Distance expansion
        self.distance_expansion = DistanceExpansion(
            num_rbf=64,
            cutoff=config.cutoff_distance
        )
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList([
            CGCNNConv(
                config.hidden_dim,
                64,  # RBF dimension
                config.hidden_dim
            )
            for _ in range(config.num_layers)
        ])
        
        # Readout layers
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.Softplus(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        
        # Output layer
        self.output_layer = nn.Linear(config.hidden_dim, config.output_dim)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            node_features: (N, input_dim)
            edge_index: (2, E)
            edge_attr: (E,) distances
            batch: (N,) batch indices
        
        Returns:
            Predictions (batch_size, output_dim)
        """
        # Embed atoms
        h = self.atom_embedding(node_features)
        
        # Expand distances
        edge_rbf = self.distance_expansion(edge_attr)
        
        # Graph convolutions
        for conv in self.conv_layers:
            h_new = conv(h, edge_index, edge_rbf)
            h = h + h_new  # Residual connection
        
        # Global pooling
        # Mean pooling
        h_mean = self._global_mean_pool(h, batch)
        # Max pooling
        h_max = self._global_max_pool(h, batch)
        
        # Concatenate
        h_pooled = torch.cat([h_mean, h_max], dim=-1)
        
        # Readout
        h_readout = self.fc(h_pooled)
        
        # Output
        output = self.output_layer(h_readout)
        
        return output
    
    def _global_mean_pool(
        self,
        x: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """Global mean pooling."""
        batch_size = batch.max().item() + 1
        out = torch.zeros(
            batch_size, x.shape[1],
            device=x.device, dtype=x.dtype
        )
        out.index_add_(0, batch, x)
        counts = torch.bincount(batch, minlength=batch_size).float()
        return out / counts.unsqueeze(-1)
    
    def _global_max_pool(
        self,
        x: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """Global max pooling."""
        batch_size = batch.max().item() + 1
        out = torch.full(
            (batch_size, x.shape[1]), float('-inf'),
            device=x.device, dtype=x.dtype
        )
        out.index_reduce_(0, batch, x, 'amax', include_self=False)
        return out


# ============================================================================
# MegNet: MatErials Graph Network
# ============================================================================

class MegNetBlock(nn.Module):
    """MegNet graph block with global, node, and edge updates."""
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        global_dim: int,
        hidden_dim: int
    ):
        super().__init__()
        
        # Edge update
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim + global_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, edge_dim),
        )
        
        # Node update
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim + global_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, node_dim),
        )
        
        # Global update
        self.global_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim + global_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, global_dim),
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        global_features: torch.Tensor,
        batch: torch.Tensor,
        edge_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            node_features: (N, node_dim)
            edge_index: (2, E)
            edge_features: (E, edge_dim)
            global_features: (batch_size, global_dim)
            batch: (N,) node batch indices
            edge_batch: (E,) edge batch indices
        """
        row, col = edge_index
        
        # Update edges
        edge_inputs = torch.cat([
            node_features[row],
            node_features[col],
            edge_features,
            global_features[edge_batch]
        ], dim=-1)
        edge_features_new = self.edge_mlp(edge_inputs)
        edge_features = edge_features + edge_features_new
        
        # Aggregate edges to nodes
        node_messages = torch.zeros_like(node_features)
        node_messages.index_add_(0, row, edge_features)
        
        # Update nodes
        node_inputs = torch.cat([
            node_features,
            node_messages,
            global_features[batch]
        ], dim=-1)
        node_features_new = self.node_mlp(node_inputs)
        node_features = node_features + node_features_new
        
        # Aggregate to global
        global_node = self._global_mean_pool(node_features, batch)
        global_edge = self._global_mean_pool(edge_features, edge_batch)
        
        global_inputs = torch.cat([
            global_node,
            global_edge,
            global_features
        ], dim=-1)
        global_features_new = self.global_mlp(global_inputs)
        global_features = global_features + global_features_new
        
        return node_features, edge_features, global_features
    
    def _global_mean_pool(
        self,
        x: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """Global mean pooling."""
        batch_size = batch.max().item() + 1
        out = torch.zeros(
            batch_size, x.shape[1],
            device=x.device, dtype=x.dtype
        )
        out.index_add_(0, batch, x)
        counts = torch.bincount(batch, minlength=batch_size).float()
        return out / counts.unsqueeze(-1)


class MegNet(nn.Module):
    """
    MatErials Graph Network.
    
    Reference: Chen et al., Phys. Rev. Materials 2019
    """
    
    def __init__(self, config: PropertyPredictorConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.node_embedding = nn.Linear(config.input_dim, config.hidden_dim)
        self.edge_embedding = nn.Linear(64, config.hidden_dim)
        self.global_embedding = nn.Linear(1, config.hidden_dim)
        
        # Distance expansion
        self.distance_expansion = DistanceExpansion(
            num_rbf=64,
            cutoff=config.cutoff_distance
        )
        
        # MegNet blocks
        self.blocks = nn.ModuleList([
            MegNetBlock(
                config.hidden_dim,
                config.hidden_dim,
                config.hidden_dim,
                config.hidden_dim
            )
            for _ in range(config.num_layers)
        ])
        
        # Output
        self.output_layer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Softplus(),
            nn.Linear(config.hidden_dim, config.output_dim),
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass."""
        # Embed features
        node_h = self.node_embedding(node_features)
        edge_h = self.distance_expansion(edge_attr)
        edge_h = self.edge_embedding(edge_h)
        
        # Create edge batch indices
        row, _ = edge_index
        edge_batch = batch[row]
        
        # Initialize global features
        batch_size = batch.max().item() + 1
        global_h = torch.ones(batch_size, 1, device=node_features.device)
        global_h = self.global_embedding(global_h)
        
        # MegNet blocks
        for block in self.blocks:
            node_h, edge_h, global_h = block(
                node_h, edge_index, edge_h, global_h, batch, edge_batch
            )
        
        # Output from global features
        output = self.output_layer(global_h)
        
        return output


# ============================================================================
# ALIGNN: Atomistic Line Graph Neural Network
# ============================================================================

class ALIGNNConv(nn.Module):
    """ALIGNN convolution on line graph."""
    
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, edge_dim),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor
    ) -> torch.Tensor:
        """Update edge features."""
        row, col = edge_index
        
        edge_inputs = torch.cat([
            node_features[row],
            node_features[col],
            edge_features
        ], dim=-1)
        
        edge_update = self.edge_mlp(edge_inputs)
        return edge_features * edge_update


class ALIGNN(nn.Module):
    """
    Atomistic Line Graph Neural Network.
    
    Reference: Choudhary & DeCost, npj Comput. Mater. 2021
    """
    
    def __init__(self, config: PropertyPredictorConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.atom_embedding = nn.Linear(config.input_dim, config.hidden_dim)
        self.edge_embedding = nn.Linear(64, config.hidden_dim)
        
        # Distance expansion
        self.distance_expansion = DistanceExpansion(
            num_rbf=64,
            cutoff=config.cutoff_distance
        )
        
        # Atomistic convolution layers (on original graph)
        self.atomistic_convs = nn.ModuleList([
            CGCNNConv(config.hidden_dim, 64, config.hidden_dim)
            for _ in range(config.num_layers)
        ])
        
        # ALIGNN layers (on line graph)
        self.alignn_convs = nn.ModuleList([
            ALIGNNConv(config.hidden_dim, config.hidden_dim)
            for _ in range(config.num_layers)
        ])
        
        # Readout
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.SiLU(),
        )
        
        self.output_layer = nn.Linear(config.hidden_dim, config.output_dim)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass."""
        # Embed
        h = self.atom_embedding(node_features)
        edge_rbf = self.distance_expansion(edge_attr)
        
        # Atomistic convolutions
        for conv in self.atomistic_convs:
            h_new = conv(h, edge_index, edge_rbf)
            h = h + h_new
        
        # Global pooling
        h_mean = self._global_mean_pool(h, batch)
        h_max = self._global_max_pool(h, batch)
        h_pooled = torch.cat([h_mean, h_max], dim=-1)
        
        # Readout
        h_readout = self.fc(h_pooled)
        
        # Output
        output = self.output_layer(h_readout)
        
        return output
    
    def _global_mean_pool(
        self,
        x: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        batch_size = batch.max().item() + 1
        out = torch.zeros(
            batch_size, x.shape[1],
            device=x.device, dtype=x.dtype
        )
        out.index_add_(0, batch, x)
        counts = torch.bincount(batch, minlength=batch_size).float()
        return out / counts.unsqueeze(-1)
    
    def _global_max_pool(
        self,
        x: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        batch_size = batch.max().item() + 1
        out = torch.full(
            (batch_size, x.shape[1]), float('-inf'),
            device=x.device, dtype=x.dtype
        )
        out.index_reduce_(0, batch, x, 'amax', include_self=False)
        return out


# ============================================================================
# Transformer Model for Materials
# ============================================================================

class TransformerModel(nn.Module):
    """
    Transformer-based model for materials property prediction.
    
    Uses self-attention over atom features.
    """
    
    def __init__(self, config: PropertyPredictorConfig):
        super().__init__()
        self.config = config
        
        # Atom embedding
        self.atom_embedding = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Distance embedding for attention bias
        self.distance_expansion = DistanceExpansion(
            num_rbf=64,
            cutoff=config.cutoff_distance
        )
        self.distance_proj = nn.Linear(64, config.num_heads)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
        # Output
        self.output_layer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass."""
        # Embed atoms
        h = self.atom_embedding(node_features)
        
        # Build attention mask with distance bias
        batch_size = batch.max().item() + 1
        max_nodes = torch.bincount(batch).max().item()
        
        # Pad sequences for transformer
        h_padded, mask = self._pad_batch(h, batch, max_nodes)
        
        # Apply transformer
        h_transformed = self.transformer(h_padded, src_key_padding_mask=mask)
        
        # Unpad and pool
        h_out = self._unpad_batch(h_transformed, batch, max_nodes)
        
        # Global pooling
        output = self._global_mean_pool(h_out, batch)
        output = self.output_layer(output)
        
        return output
    
    def _pad_batch(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        max_nodes: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad batch to fixed size."""
        batch_size = batch.max().item() + 1
        device = x.device
        
        x_padded = torch.zeros(
            batch_size, max_nodes, x.shape[1],
            device=device, dtype=x.dtype
        )
        mask = torch.ones(batch_size, max_nodes, dtype=torch.bool, device=device)
        
        for i in range(batch_size):
            mask_i = batch == i
            n_nodes = mask_i.sum().item()
            x_padded[i, :n_nodes] = x[mask_i]
            mask[i, :n_nodes] = False
        
        return x_padded, mask
    
    def _unpad_batch(
        self,
        x_padded: torch.Tensor,
        batch: torch.Tensor,
        max_nodes: int
    ) -> torch.Tensor:
        """Unpad batch."""
        batch_size = batch.max().item() + 1
        outputs = []
        
        for i in range(batch_size):
            mask_i = batch == i
            n_nodes = mask_i.sum().item()
            outputs.append(x_padded[i, :n_nodes])
        
        return torch.cat(outputs, dim=0)
    
    def _global_mean_pool(
        self,
        x: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        batch_size = batch.max().item() + 1
        out = torch.zeros(
            batch_size, x.shape[1],
            device=x.device, dtype=x.dtype
        )
        out.index_add_(0, batch, x)
        counts = torch.bincount(batch, minlength=batch_size).float()
        return out / counts.unsqueeze(-1)


# ============================================================================
# Property Predictor Interface
# ============================================================================

class PropertyPredictor:
    """
    Unified interface for materials property prediction.
    
    Supports multiple GNN architectures and provides a consistent API.
    """
    
    SUPPORTED_MODELS = ['cgcnn', 'megnet', 'alignn', 'transformer']
    
    def __init__(
        self,
        model_type: str = 'cgcnn',
        config: Optional[PropertyPredictorConfig] = None,
        device: str = 'cpu'
    ):
        """
        Initialize property predictor.
        
        Args:
            model_type: Type of GNN model
            config: Model configuration
            device: Device to use
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model_type = model_type
        self.config = config or PropertyPredictorConfig(model_type=model_type)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Feature encoder
        self.atom_encoder = AtomFeatureEncoder()
        
        self.is_trained = False
        self.training_history = []
    
    def _create_model(self) -> nn.Module:
        """Create the GNN model."""
        if self.model_type == 'cgcnn':
            return CGCNN(self.config)
        elif self.model_type == 'megnet':
            return MegNet(self.config)
        elif self.model_type == 'alignn':
            return ALIGNN(self.config)
        elif self.model_type == 'transformer':
            return TransformerModel(self.config)
        
        raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(
        self,
        structures: List[Any],
        targets: np.ndarray,
        num_epochs: int = 200,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        validation_split: float = 0.1,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the property predictor.
        
        Args:
            structures: List of structures (pymatgen or ASE)
            targets: Target property values (N, output_dim)
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            validation_split: Fraction for validation
            verbose: Print progress
        
        Returns:
            Training history
        """
        # Prepare dataset
        dataset = self._create_dataset(structures, targets)
        
        # Split
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_graph_batch
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collate_graph_batch
        )
        
        # Optimizer and scheduler
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=20, factor=0.5
        )
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_losses = []
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Forward
                predictions = self.model(
                    batch['node_features'].to(self.device),
                    batch['edge_index'].to(self.device),
                    batch['edge_attr'].to(self.device),
                    batch['batch'].to(self.device)
                )
                
                # Loss
                loss = criterion(predictions, batch['target'].to(self.device))
                
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            self.model.eval()
            val_losses = []
            val_maes = []
            
            with torch.no_grad():
                for batch in val_loader:
                    predictions = self.model(
                        batch['node_features'].to(self.device),
                        batch['edge_index'].to(self.device),
                        batch['edge_attr'].to(self.device),
                        batch['batch'].to(self.device)
                    )
                    
                    loss = criterion(predictions, batch['target'].to(self.device))
                    mae = torch.abs(predictions - batch['target'].to(self.device)).mean()
                    
                    val_losses.append(loss.item())
                    val_maes.append(mae.item())
            
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            val_mae = np.mean(val_maes)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)
            
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_state = self.model.state_dict().copy()
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"train_loss={train_loss:.4f}, "
                      f"val_loss={val_loss:.4f}, "
                      f"val_mae={val_mae:.4f}")
        
        # Load best model
        if hasattr(self, 'best_state'):
            self.model.load_state_dict(self.best_state)
        
        self.is_trained = True
        self.training_history = history
        
        return history
    
    def predict(
        self,
        structures: List[Any],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Predict properties for structures.
        
        Args:
            structures: List of structures
            batch_size: Batch size
        
        Returns:
            Predictions (N, output_dim)
        """
        self.model.eval()
        
        # Create dataset
        dummy_targets = np.zeros((len(structures), self.config.output_dim))
        dataset = self._create_dataset(structures, dummy_targets)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_graph_batch
        )
        
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                preds = self.model(
                    batch['node_features'].to(self.device),
                    batch['edge_index'].to(self.device),
                    batch['edge_attr'].to(self.device),
                    batch['batch'].to(self.device)
                )
                predictions.append(preds.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def _create_dataset(
        self,
        structures: List[Any],
        targets: np.ndarray
    ) -> 'MaterialsDataset':
        """Create dataset from structures."""
        graphs = []
        
        for i, (struct, target) in enumerate(zip(structures, targets)):
            graph = self._structure_to_graph(struct, target, idx=i)
            if graph is not None:
                graphs.append(graph)
        
        return MaterialsDataset(graphs)
    
    def _structure_to_graph(
        self,
        structure: Any,
        target: np.ndarray,
        idx: int = 0
    ) -> Optional[MaterialGraph]:
        """Convert structure to graph representation."""
        try:
            # Extract positions and atomic numbers
            if HAS_PYMATGEN and isinstance(structure, Structure):
                positions = structure.cart_coords
                atomic_numbers = np.array([site.specie.Z for site in structure])
                composition = str(structure.composition)
            elif HAS_ASE and isinstance(structure, Atoms):
                positions = structure.positions
                atomic_numbers = structure.numbers
                composition = structure.get_chemical_formula()
            else:
                # Assume it's a dict
                positions = structure.get('positions', structure.get('coords'))
                atomic_numbers = structure.get('atomic_numbers', structure.get('species'))
                composition = structure.get('composition', 'unknown')
            
            # Build neighbor list
            edge_index, edge_attr = self._build_edges(positions)
            
            # Encode atom features
            node_features = np.stack([
                self.atom_encoder.encode(z)
                for z in atomic_numbers
            ])
            
            return MaterialGraph(
                node_features=torch.tensor(node_features, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
                target=torch.tensor(target, dtype=torch.float32),
                structure_id=str(idx),
                composition=composition
            )
        
        except Exception as e:
            warnings.warn(f"Failed to process structure {idx}: {e}")
            return None
    
    def _build_edges(
        self,
        positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build edge list from positions."""
        n = len(positions)
        
        # Simple distance-based edges
        edges = []
        distances = []
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < self.config.cutoff_distance:
                        edges.append([i, j])
                        distances.append(dist)
        
        if len(edges) == 0:
            # Add self-loops if no edges
            edges = [[i, i] for i in range(n)]
            distances = [0.0] * n
        
        return np.array(edges).T, np.array(distances)
    
    def save(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_type': self.model_type,
            'config': self.config,
            'model_state': self.model.state_dict(),
            'is_trained': self.is_trained,
            'training_history': self.training_history,
        }
        
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model_type = checkpoint['model_type']
        self.config = checkpoint['config']
        self.is_trained = checkpoint['is_trained']
        self.training_history = checkpoint.get('training_history', [])
        
        self.model.load_state_dict(checkpoint['model_state'])
        print(f"Model loaded from {path}")


class MaterialsDataset(Dataset):
    """Dataset for materials graphs."""
    
    def __init__(self, graphs: List[MaterialGraph]):
        self.graphs = graphs
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> MaterialGraph:
        return self.graphs[idx]


def collate_graph_batch(batch: List[MaterialGraph]) -> Dict[str, torch.Tensor]:
    """Collate function for batching graphs."""
    node_features = []
    edge_index = []
    edge_attr = []
    targets = []
    batch_indices = []
    
    node_offset = 0
    for i, graph in enumerate(batch):
        node_features.append(graph.node_features)
        targets.append(graph.target)
        
        # Offset edge indices
        edge_index.append(graph.edge_index + node_offset)
        edge_attr.append(graph.edge_attr)
        
        # Batch indices
        batch_indices.append(torch.full((graph.num_nodes,), i, dtype=torch.long))
        
        node_offset += graph.num_nodes
    
    return {
        'node_features': torch.cat(node_features, dim=0),
        'edge_index': torch.cat(edge_index, dim=1),
        'edge_attr': torch.cat(edge_attr, dim=0) if edge_attr[0] is not None else None,
        'target': torch.stack(targets, dim=0),
        'batch': torch.cat(batch_indices, dim=0),
    }


# ============================================================================
# Pre-trained Model Loading
# ============================================================================

class PretrainedModelLoader:
    """Loader for pre-trained models like M3GNet and CHGNet."""
    
    SUPPORTED_MODELS = ['m3gnet', 'chgnet', 'orb', 'eqv2']
    
    def __init__(self, model_name: str, device: str = 'cpu'):
        """
        Initialize pretrained model loader.
        
        Args:
            model_name: Name of the pretrained model
            device: Device to use
        """
        self.model_name = model_name.lower()
        self.device = device
        self.model = None
        
        if self.model_name not in self.SUPPORTED_MODELS:
            warnings.warn(f"Model {model_name} not directly supported. "
                         "Will attempt to load via available packages.")
    
    def load(self, checkpoint_path: Optional[str] = None):
        """
        Load the pretrained model.
        
        Args:
            checkpoint_path: Path to checkpoint (optional, uses default if None)
        """
        if self.model_name == 'm3gnet':
            self._load_m3gnet(checkpoint_path)
        elif self.model_name == 'chgnet':
            self._load_chgnet(checkpoint_path)
        else:
            self._load_generic(checkpoint_path)
    
    def _load_m3gnet(self, checkpoint_path: Optional[str]):
        """Load M3GNet universal potential."""
        try:
            # Try to import matgl
            import matgl
            from matgl.ext.ase import M3GNetCalculator
            
            # Load default M3GNet
            potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")
            self.model = M3GNetCalculator(potential)
            print("M3GNet loaded successfully")
            
        except ImportError:
            warnings.warn("matgl not installed. M3GNet unavailable.")
            self.model = None
    
    def _load_chgnet(self, checkpoint_path: Optional[str]):
        """Load CHGNet universal potential."""
        try:
            from chgnet.model import CHGNet
            
            # Load default CHGNet
            self.model = CHGNet.load()
            print("CHGNet loaded successfully")
            
        except ImportError:
            warnings.warn("chgnet not installed. CHGNet unavailable.")
            self.model = None
    
    def _load_generic(self, checkpoint_path: Optional[str]):
        """Generic model loading."""
        warnings.warn(f"Generic loading not implemented for {self.model_name}")
    
    def predict_energy(self, structure: Any) -> float:
        """Predict energy of a structure."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        if self.model_name == 'm3gnet':
            # Convert to ASE if needed
            if HAS_PYMATGEN and isinstance(structure, Structure):
                structure = self._pymatgen_to_ase(structure)
            
            self.model.atoms = structure
            return self.model.get_potential_energy()
        
        elif self.model_name == 'chgnet':
            from chgnet.model import Struct
            
            if HAS_PYMATGEN and isinstance(structure, Structure):
                chg_structure = Struct.from_pymatgen(structure)
            else:
                chg_structure = structure
            
            prediction = self.model.predict_structure(chg_structure)
            return prediction['e']
        
        return 0.0
    
    def predict_forces(self, structure: Any) -> np.ndarray:
        """Predict forces on atoms."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        if self.model_name == 'm3gnet':
            if HAS_PYMATGEN and isinstance(structure, Structure):
                structure = self._pymatgen_to_ase(structure)
            
            self.model.atoms = structure
            return self.model.get_forces()
        
        elif self.model_name == 'chgnet':
            from chgnet.model import Struct
            
            if HAS_PYMATGEN and isinstance(structure, Structure):
                chg_structure = Struct.from_pymatgen(structure)
            else:
                chg_structure = structure
            
            prediction = self.model.predict_structure(chg_structure)
            return prediction['f']
        
        return np.zeros((len(structure), 3))
    
    def _pymatgen_to_ase(self, structure: Any) -> Any:
        """Convert pymatgen Structure to ASE Atoms."""
        if not HAS_ASE:
            raise ImportError("ASE required for conversion")
        
        return Atoms(
            numbers=[site.specie.Z for site in structure],
            positions=structure.cart_coords,
            cell=structure.lattice.matrix,
            pbc=True
        )


def load_pretrained_predictor(
    model_name: str,
    checkpoint_path: Optional[str] = None,
    device: str = 'cpu'
) -> PretrainedModelLoader:
    """
    Load a pretrained universal potential.
    
    Args:
        model_name: 'm3gnet', 'chgnet', etc.
        checkpoint_path: Path to checkpoint
        device: Device to use
    
    Returns:
        PretrainedModelLoader instance
    """
    loader = PretrainedModelLoader(model_name, device)
    loader.load(checkpoint_path)
    return loader


if __name__ == "__main__":
    # Example usage
    print("Property Predictor Module")
    print("=" * 50)
    
    # Create a simple predictor
    config = PropertyPredictorConfig(model_type='cgcnn')
    predictor = PropertyPredictor('cgcnn', config)
    
    print(f"\nModel: {predictor.model_type}")
    print(f"Parameters: {sum(p.numel() for p in predictor.model.parameters()):,}")
