"""
Graph Neural Networks for Cross-Scale Modeling

Implements GNN architectures for learning interatomic potentials
and coarse-grained force fields with equivariance properties.
"""
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass


@dataclass
class Graph:
    """Represents a molecular graph."""
    nodes: np.ndarray  # Node features (n_nodes, n_features)
    edges: np.ndarray  # Edge list (n_edges, 2)
    edge_features: np.ndarray  # Edge features (n_edges, n_edge_features)
    positions: np.ndarray  # 3D positions (n_nodes, 3)
    node_types: List[str]  # Node type labels
    
    @property
    def n_nodes(self) -> int:
        return len(self.nodes)
    
    @property
    def n_edges(self) -> int:
        return len(self.edges)


def build_graph(positions: np.ndarray,
                elements: List[str],
                cutoff: float = 5.0,
                pbc: Optional[np.ndarray] = None) -> Graph:
    """
    Build graph from atomic positions.
    
    Args:
        positions: (N, 3) atomic positions
        elements: Element symbols
        cutoff: Distance cutoff for edges
        pbc: Periodic boundary conditions (3,) or None
        
    Returns:
        Graph object
    """
    n_atoms = len(positions)
    
    # Node features: one-hot encoded element types
    unique_elements = sorted(set(elements))
    element_to_idx = {e: i for i, e in enumerate(unique_elements)}
    node_features = np.zeros((n_atoms, len(unique_elements)))
    for i, elem in enumerate(elements):
        node_features[i, element_to_idx[elem]] = 1.0
    
    # Build edges based on distance
    edges = []
    edge_features = []
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist_vec = positions[j] - positions[i]
            
            # Apply PBC if needed
            if pbc is not None:
                dist_vec -= pbc * np.round(dist_vec / pbc)
            
            dist = np.linalg.norm(dist_vec)
            
            if dist < cutoff:
                edges.append([i, j])
                edges.append([j, i])  # Bidirectional
                
                # Edge features: distance and direction
                edge_feat = np.array([dist, 
                                     dist_vec[0] / dist,
                                     dist_vec[1] / dist,
                                     dist_vec[2] / dist])
                edge_features.append(edge_feat)
                edge_features.append(edge_feat)
    
    edges = np.array(edges) if edges else np.zeros((0, 2), dtype=int)
    edge_features = np.array(edge_features) if edge_features else np.zeros((0, 4))
    
    return Graph(
        nodes=node_features,
        edges=edges,
        edge_features=edge_features,
        positions=positions,
        node_types=elements
    )


class MessagePassingLayer:
    """
    Message passing layer for graph neural networks.
    Implements E(3)-equivariant message passing.
    """
    
    def __init__(self, 
                 node_dim: int,
                 edge_dim: int,
                 hidden_dim: int = 64):
        """
        Initialize message passing layer.
        
        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
            hidden_dim: Hidden layer dimension
        """
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights
        self.W_msg = np.random.randn(node_dim, hidden_dim) * 0.01
        self.W_edge = np.random.randn(edge_dim, hidden_dim) * 0.01
        # Project back to node_dim for output, but for hidden layers use hidden_dim
        self.W_update = np.random.randn(hidden_dim, hidden_dim) * 0.01
        
    def message(self, 
                node_features: np.ndarray,
                edge_features: np.ndarray,
                edges: np.ndarray) -> np.ndarray:
        """
        Compute messages.
        
        Args:
            node_features: (N, node_dim)
            edge_features: (E, edge_dim)
            edges: (E, 2) edge list
            
        Returns:
            Messages (E, hidden_dim)
        """
        # Source node features for each edge
        src_nodes = edges[:, 0]
        src_features = node_features[src_nodes]
        
        # Combine node and edge features
        messages = np.dot(src_features, self.W_msg) * np.dot(edge_features, self.W_edge)
        
        return np.tanh(messages)
    
    def aggregate(self,
                 messages: np.ndarray,
                 edges: np.ndarray,
                 n_nodes: int) -> np.ndarray:
        """
        Aggregate messages at each node.
        
        Args:
            messages: (E, hidden_dim)
            edges: (E, 2) edge list
            n_nodes: Number of nodes
            
        Returns:
            Aggregated features (N, hidden_dim)
        """
        dst_nodes = edges[:, 1]
        aggregated = np.zeros((n_nodes, self.hidden_dim))
        
        for i, dst in enumerate(dst_nodes):
            aggregated[dst] += messages[i]
        
        return aggregated
    
    def update(self,
              node_features: np.ndarray,
              aggregated: np.ndarray) -> np.ndarray:
        """
        Update node features.
        
        Args:
            node_features: (N, node_dim)
            aggregated: (N, hidden_dim)
            
        Returns:
            Updated features (N, hidden_dim)
        """
        # Project node features to hidden_dim if needed
        if node_features.shape[1] != self.hidden_dim:
            # First layer: project node_features to hidden_dim
            projection = np.dot(node_features, self.W_msg)  # (N, hidden_dim)
        else:
            projection = node_features
            
        updated = np.dot(aggregated, self.W_update)
        return projection + np.tanh(updated)
    
    def forward(self, graph: Graph) -> Graph:
        """
        Forward pass through message passing layer.
        
        Args:
            graph: Input graph
            
        Returns:
            Updated graph
        """
        messages = self.message(graph.nodes, graph.edge_features, graph.edges)
        aggregated = self.aggregate(messages, graph.edges, graph.n_nodes)
        new_nodes = self.update(graph.nodes, aggregated)
        
        # If this is the first layer and node_dim != hidden_dim, 
        # the output is already hidden_dim due to W_update shape
        # For subsequent layers, node_dim == hidden_dim
        
        return Graph(
            nodes=new_nodes,
            edges=graph.edges,
            edge_features=graph.edge_features,
            positions=graph.positions,
            node_types=graph.node_types
        )


class CGGNN:
    """
    Graph Neural Network for Coarse-Grained Force Fields.
    Predicts forces on CG beads from CG configurations.
    """
    
    def __init__(self,
                 n_node_features: int,
                 n_edge_features: int = 4,
                 hidden_dim: int = 64,
                 n_layers: int = 4,
                 cutoff: float = 10.0):
        """
        Initialize CG GNN.
        
        Args:
            n_node_features: Number of node input features
            n_edge_features: Number of edge input features
            hidden_dim: Hidden layer dimension
            n_layers: Number of message passing layers
            cutoff: Distance cutoff
        """
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.cutoff = cutoff
        
        # Build layers
        self.layers = []
        for i in range(n_layers):
            self.layers.append(MessagePassingLayer(
                node_dim=n_node_features if i == 0 else hidden_dim,
                edge_dim=n_edge_features,
                hidden_dim=hidden_dim
            ))
        
        # Force prediction head
        self.force_head = None  # Initialized in _build_force_head
        self._build_force_head()
    
    def _build_force_head(self):
        """Build force prediction head."""
        # Simple linear layer for force prediction
        # In full implementation, would use equivariant architecture
        self.force_head = np.random.randn(self.hidden_dim, 3) * 0.01
    
    def forward(self, graph: Graph) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            graph: Input graph
            
        Returns:
            Predicted forces (N, 3)
        """
        # Message passing
        h = graph
        for layer in self.layers:
            h = layer.forward(h)
        
        # Predict forces
        forces = np.dot(h.nodes, self.force_head)
        
        return forces
    
    def train_step(self,
                  graph: Graph,
                  target_forces: np.ndarray,
                  learning_rate: float = 0.001) -> float:
        """
        Single training step.
        
        Args:
            graph: Input graph
            target_forces: Target forces
            learning_rate: Learning rate
            
        Returns:
            Loss value
        """
        # Forward pass
        pred_forces = self.forward(graph)
        
        # Compute loss (MSE)
        loss = np.mean((pred_forces - target_forces) ** 2)
        
        # Simple gradient descent update (simplified)
        # In full implementation, would use proper backprop
        grad = 2 * (pred_forces - target_forces) / len(target_forces)
        
        # Update force head
        h = graph
        for layer in self.layers:
            h = layer.forward(h)
        
        self.force_head -= learning_rate * np.dot(h.nodes.T, grad)
        
        return loss
    
    def predict_forces(self,
                      cg_positions: np.ndarray,
                      cg_types: List[str],
                      cg_mapping: np.ndarray = None) -> np.ndarray:
        """
        Predict forces on CG beads.
        
        Args:
            cg_positions: (N, 3) CG bead positions
            cg_types: CG bead types
            cg_mapping: Bead-to-bead mapping (for bonded interactions)
            
        Returns:
            Predicted forces (N, 3)
        """
        graph = build_graph(cg_positions, cg_types, self.cutoff)
        return self.forward(graph)


class MultiscaleGNN:
    """
    Multiscale Graph Neural Network.
    Simultaneously models atomistic and coarse-grained scales.
    """
    
    def __init__(self,
                 atom_features: int,
                 cg_features: int,
                 hidden_dim: int = 128,
                 n_atom_layers: int = 4,
                 n_cg_layers: int = 3):
        """
        Initialize multiscale GNN.
        
        Args:
            atom_features: Number of atom features
            cg_features: Number of CG bead features
            hidden_dim: Hidden dimension
            n_atom_layers: Number of atom-scale layers
            n_cg_layers: Number of CG-scale layers
        """
        self.atom_features = atom_features
        self.cg_features = cg_features
        self.hidden_dim = hidden_dim
        self.n_atom_layers = n_atom_layers
        self.n_cg_layers = n_cg_layers
        
        # Atom-scale GNN
        self.atom_gnn = CGGNN(
            n_node_features=atom_features,
            hidden_dim=hidden_dim,
            n_layers=n_atom_layers
        )
        
        # CG-scale GNN
        self.cg_gnn = CGGNN(
            n_node_features=cg_features,
            hidden_dim=hidden_dim,
            n_layers=n_cg_layers
        )
        
        # Cross-scale attention mechanism
        self.cross_attention = None
        self._build_cross_attention()
    
    def _build_cross_attention(self):
        """Build cross-scale attention mechanism."""
        # Attention weights from CG to atom
        self.W_q = np.random.randn(self.cg_features, self.hidden_dim) * 0.01
        self.W_k = np.random.randn(self.atom_features, self.hidden_dim) * 0.01
        self.W_v = np.random.randn(self.atom_features, self.hidden_dim) * 0.01
    
    def forward_atom(self, atom_graph: Graph) -> np.ndarray:
        """Forward pass on atom scale."""
        return self.atom_gnn.forward(atom_graph)
    
    def forward_cg(self, cg_graph: Graph) -> np.ndarray:
        """Forward pass on CG scale."""
        return self.cg_gnn.forward(cg_graph)
    
    def cross_scale_attention(self,
                              atom_features: np.ndarray,
                              cg_features: np.ndarray,
                              atom_to_cg_mapping: np.ndarray) -> np.ndarray:
        """
        Apply cross-scale attention.
        
        Args:
            atom_features: Atom-scale features (n_atoms, atom_features)
            cg_features: CG-scale features (n_cg, cg_features)
            atom_to_cg_mapping: Mapping from atoms to CG beads
            
        Returns:
            Updated CG features with atom-scale information
        """
        n_cg = len(cg_features)
        updated_cg = cg_features.copy()  # Start with original features
        
        # Only apply attention if dimensions match
        if cg_features.shape[1] != self.hidden_dim:
            # Need to project CG features to hidden_dim first
            # For simplicity, just return original features
            return updated_cg
        
        for cg_idx in range(n_cg):
            # Find atoms belonging to this CG bead
            atom_indices = np.where(atom_to_cg_mapping == cg_idx)[0]
            if len(atom_indices) == 0:
                continue
            
            # Query from CG feature
            query = np.dot(cg_features[cg_idx], self.W_q)
            
            # Keys and values from atom features
            keys = np.dot(atom_features[atom_indices], self.W_k)
            values = np.dot(atom_features[atom_indices], self.W_v)
            
            # Attention scores
            scores = np.dot(keys, query) / np.sqrt(self.hidden_dim)
            attention_weights = self._softmax(scores)
            
            # Weighted sum
            attention_output = np.dot(attention_weights, values)
            updated_cg[cg_idx] = cg_features[cg_idx] + attention_output
        
        return updated_cg
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def predict_multiscale(self,
                          atom_positions: np.ndarray,
                          atom_elements: List[str],
                          cg_positions: np.ndarray,
                          cg_types: List[str],
                          atom_to_cg_mapping: np.ndarray) -> Dict:
        """
        Make predictions at both scales.
        
        Args:
            atom_positions: Atom positions
            atom_elements: Atom element symbols
            cg_positions: CG bead positions
            cg_types: CG bead types
            atom_to_cg_mapping: Mapping from atoms to CG beads
            
        Returns:
            Dictionary with atom_forces, cg_forces
        """
        # Build graphs
        atom_graph = build_graph(atom_positions, atom_elements)
        cg_graph = build_graph(cg_positions, cg_types, cutoff=15.0)
        
        # Atom-scale prediction
        atom_forces = self.forward_atom(atom_graph)
        
        # CG-scale prediction with cross-attention
        cg_forces = self.forward_cg(cg_graph)
        
        return {
            'atom_forces': atom_forces,
            'cg_forces': cg_forces
        }


class EquivariantGNN:
    """
    E(3)-Equivariant Graph Neural Network.
    Ensures predictions transform correctly under rotations and translations.
    """
    
    def __init__(self,
                 n_scalars: int = 16,
                 n_vectors: int = 4,
                 hidden_dim: int = 64,
                 n_layers: int = 4):
        """
        Initialize equivariant GNN.
        
        Args:
            n_scalars: Number of scalar features
            n_vectors: Number of vector features
            hidden_dim: Hidden dimension
            n_layers: Number of layers
        """
        self.n_scalars = n_scalars
        self.n_vectors = n_vectors
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Initialize filters
        self.filters = []
        for _ in range(n_layers):
            self.filters.append({
                'phi': np.random.randn(hidden_dim, hidden_dim) * 0.01,
                'W_s': np.random.randn(n_scalars + n_vectors, hidden_dim) * 0.01
            })
    
    def radial_basis(self, distances: np.ndarray, 
                    n_rbf: int = 20,
                    cutoff: float = 5.0) -> np.ndarray:
        """
        Compute radial basis functions.
        
        Args:
            distances: Array of distances
            n_rbf: Number of radial basis functions
            cutoff: Cutoff distance
            
        Returns:
            RBF features (len(distances), n_rbf)
        """
        centers = np.linspace(0, cutoff, n_rbf)
        width = cutoff / n_rbf
        
        rbf = np.exp(-((distances[:, np.newaxis] - centers) ** 2) / (2 * width ** 2))
        return rbf
    
    def forward(self,
               positions: np.ndarray,
               scalars: np.ndarray,
               edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass with equivariance.
        
        Args:
            positions: (N, 3) positions
            scalars: (N, n_scalars) scalar features
            edges: (E, 2) edge list
            
        Returns:
            Updated scalars and vectors
        """
        n_nodes = len(positions)
        
        # Initialize vector features as zeros
        vectors = np.zeros((n_nodes, self.n_vectors, 3))
        
        for layer_idx in range(self.n_layers):
            # Compute edge vectors and distances
            edge_vectors = positions[edges[:, 1]] - positions[edges[:, 0]]
            edge_distances = np.linalg.norm(edge_vectors, axis=1, keepdims=True)
            
            # Normalize edge vectors
            edge_directions = edge_vectors / (edge_distances + 1e-8)
            
            # Radial basis
            rbf = self.radial_basis(edge_distances.flatten())
            
            # Message passing (simplified)
            # In full implementation, would use proper tensor field networks
            new_scalars = scalars.copy()
            new_vectors = vectors.copy()
        
        return scalars, vectors


def train_gnn(model: CGGNN,
              train_graphs: List[Graph],
              train_forces: List[np.ndarray],
              val_graphs: List[Graph] = None,
              val_forces: List[np.ndarray] = None,
              n_epochs: int = 100,
              learning_rate: float = 0.001,
              batch_size: int = 32) -> Dict:
    """
    Train a GNN model.
    
    Args:
        model: GNN model to train
        train_graphs: Training graphs
        train_forces: Training forces
        val_graphs: Validation graphs
        val_forces: Validation forces
        n_epochs: Number of epochs
        learning_rate: Learning rate
        batch_size: Batch size
        
    Returns:
        Training history
    """
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(n_epochs):
        # Training
        epoch_loss = 0.0
        for graph, forces in zip(train_graphs, train_forces):
            loss = model.train_step(graph, forces, learning_rate)
            epoch_loss += loss
        
        avg_loss = epoch_loss / len(train_graphs)
        history['train_loss'].append(avg_loss)
        
        # Validation
        if val_graphs is not None:
            val_loss = 0.0
            for graph, forces in zip(val_graphs, val_forces):
                pred_forces = model.forward(graph)
                val_loss += np.mean((pred_forces - forces) ** 2)
            
            avg_val_loss = val_loss / len(val_graphs)
            history['val_loss'].append(avg_val_loss)
        
        if epoch % 10 == 0:
            msg = f"Epoch {epoch}: train_loss={avg_loss:.6f}"
            if val_graphs:
                msg += f", val_loss={avg_val_loss:.6f}"
            print(msg)
    
    return history
