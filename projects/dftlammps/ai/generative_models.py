"""
Generative Models for Crystal Structure Generation
===================================================

This module implements state-of-the-art generative models for
crystal structure generation including CDVAE, DiffCSP, and MatterGen.

Supported Models:
- CDVAE: Crystal Diffusion Variational Autoencoder
- DiffCSP: Diffusion Model for Crystal Structure Prediction
- MatterGen: Generative model for materials design
- GSchNet: Generative SchNet for crystal structures

Author: DFT+LAMMPS AI Team
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
from collections import defaultdict
import warnings

# Optional imports for advanced features
try:
    from pymatgen.core import Structure, Lattice, Element, Composition
    from pymatgen.io.cif import CifWriter
    from pymatgen.analysis.structure_matcher import StructureMatcher
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False
    warnings.warn("pymatgen not available. Structure operations will be limited.")

try:
    from ase import Atoms
    from ase.io import write, read
    from ase.build import bulk
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    warnings.warn("ASE not available. Atomic operations will be limited.")


@dataclass
class GenerativeModelConfig:
    """Configuration for generative models."""
    model_type: str = "cdvae"  # cdvae, diffcsp, mattergen, gschnet
    latent_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    max_atoms: int = 100
    cutoff_distance: float = 8.0
    num_species: int = 100
    use_fractional_coords: bool = True
    symmetry_threshold: float = 0.01
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 1000
    
    # Generation parameters
    num_structures: int = 100
    temperature: float = 1.0
    guidance_scale: float = 1.0
    
    # Paths
    checkpoint_dir: str = "./checkpoints"
    pretrained_model_path: Optional[str] = None


@dataclass
class CrystalStructure:
    """Represents a crystal structure."""
    lattice: np.ndarray  # 3x3 lattice vectors
    frac_coords: np.ndarray  # Fractional coordinates (N, 3)
    atomic_numbers: np.ndarray  # Atomic numbers (N,)
    composition: Optional[str] = None
    space_group: Optional[int] = None
    energy: Optional[float] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_pymatgen(self) -> Optional[Any]:
        """Convert to pymatgen Structure."""
        if not HAS_PYMATGEN:
            return None
        return Structure(
            Lattice(self.lattice),
            self.atomic_numbers.tolist(),
            self.frac_coords
        )
    
    def to_ase(self) -> Optional[Any]:
        """Convert to ASE Atoms."""
        if not HAS_ASE:
            return None
        return Atoms(
            numbers=self.atomic_numbers,
            scaled_positions=self.frac_coords,
            cell=self.lattice,
            pbc=True
        )
    
    @classmethod
    def from_pymatgen(cls, structure: Any) -> "CrystalStructure":
        """Create from pymatgen Structure."""
        return cls(
            lattice=structure.lattice.matrix,
            frac_coords=structure.frac_coords,
            atomic_numbers=np.array([s.Z for s in structure.species]),
            composition=str(structure.composition),
            space_group=structure.get_space_group_info()[1] if HAS_PYMATGEN else None
        )
    
    @classmethod
    def from_ase(cls, atoms: Any) -> "CrystalStructure":
        """Create from ASE Atoms."""
        return cls(
            lattice=atoms.cell.array,
            frac_coords=atoms.get_scaled_positions(),
            atomic_numbers=atoms.numbers,
            composition=atoms.get_chemical_formula()
        )


# ============================================================================
# CDVAE: Crystal Diffusion Variational Autoencoder
# ============================================================================

class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for time steps."""
    
    def __init__(self, embed_dim: int, scale: float = 30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class EquivariantGraphConv(nn.Module):
    """E(n)-Equivariant Graph Convolution Layer."""
    
    def __init__(self, hidden_dim: int, edge_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Edge network
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Node network
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Coordinate network
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: (N, hidden_dim)
            edge_index: (2, E)
            edge_attr: (E, edge_dim)
            coords: (N, 3)
        Returns:
            Updated node features and coordinates
        """
        row, col = edge_index
        
        # Compute edge features
        edge_features = torch.cat([
            node_features[row],
            node_features[col],
            edge_attr
        ], dim=-1)
        
        # Edge messages
        edge_messages = self.edge_mlp(edge_features)
        
        # Coordinate updates (equivariant)
        coord_diff = coords[row] - coords[col]
        coord_dist = torch.norm(coord_diff, dim=-1, keepdim=True) + 1e-8
        coord_messages = self.coord_mlp(edge_messages) * coord_diff / coord_dist
        
        # Aggregate for coordinate update
        coord_update = torch.zeros_like(coords)
        coord_update.index_add_(0, row, coord_messages)
        
        # Aggregate edge messages for nodes
        node_messages = torch.zeros_like(node_features)
        node_messages.index_add_(0, row, edge_messages)
        
        # Update nodes
        node_features_new = self.node_mlp(
            torch.cat([node_features, node_messages], dim=-1)
        )
        
        return node_features_new, coords + coord_update


class CDVAEDiffusion(nn.Module):
    """Diffusion model for crystal structure generation (CDVAE)."""
    
    def __init__(self, config: GenerativeModelConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
        )
        
        # Atom type embedding
        self.atom_embed = nn.Embedding(config.num_species, config.hidden_dim)
        
        # Edge embedding (distances)
        self.edge_embed = nn.Sequential(
            GaussianFourierProjection(64),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList([
            EquivariantGraphConv(config.hidden_dim, 64)
            for _ in range(config.num_layers)
        ])
        
        # Output heads
        self.coord_pred = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 3),
        )
        
        self.type_pred = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.num_species),
        )
    
    def build_edges(
        self,
        coords: torch.Tensor,
        batch: torch.Tensor,
        cutoff: float = 8.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build edge index based on distance cutoff."""
        # Simple O(N^2) implementation for clarity
        # In practice, use neighbor lists
        edge_index = []
        edge_attr = []
        
        unique_batches = torch.unique(batch)
        for b in unique_batches:
            mask = batch == b
            coords_batch = coords[mask]
            n = coords_batch.shape[0]
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        dist = torch.norm(coords_batch[i] - coords_batch[j])
                        if dist < cutoff:
                            edge_index.append([i, j])
                            edge_attr.append(dist.item())
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32).unsqueeze(-1)
        
        return edge_index, edge_attr
    
    def forward(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion step.
        
        Args:
            coords: (N, 3) coordinates
            atom_types: (N,) atom type indices
            t: (batch_size,) time steps
            batch: (N,) batch indices
        
        Returns:
            coord_noise: (N, 3) predicted coordinate noise
            type_logits: (N, num_species) predicted atom type logits
        """
        # Time embedding
        t_embed = self.time_embed(t.float())
        t_embed = t_embed[batch]  # Expand to node level
        
        # Atom embedding
        node_features = self.atom_embed(atom_types) + t_embed
        
        # Build edges
        edge_index, edge_attr = self.build_edges(coords, batch, self.config.cutoff_distance)
        edge_attr = self.edge_embed(edge_attr.to(coords.device))
        
        # Graph convolutions
        for conv in self.conv_layers:
            node_features, coords = conv(node_features, edge_index, edge_attr, coords)
        
        # Predict noise
        coord_noise = self.coord_pred(node_features)
        type_logits = self.type_pred(node_features)
        
        return coord_noise, type_logits


class CDVAEEncoder(nn.Module):
    """Encoder for CDVAE - compresses crystal to latent space."""
    
    def __init__(self, config: GenerativeModelConfig):
        super().__init__()
        self.config = config
        
        self.atom_embed = nn.Embedding(config.num_species, config.hidden_dim)
        
        self.edge_embed = nn.Sequential(
            GaussianFourierProjection(64),
            nn.Linear(64, 64),
        )
        
        self.conv_layers = nn.ModuleList([
            EquivariantGraphConv(config.hidden_dim, 64)
            for _ in range(config.num_layers)
        ])
        
        # Pooling to get graph-level representation
        self.graph_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.latent_dim * 2),  # mu and logvar
        )
    
    def forward(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode structure to latent distribution."""
        node_features = self.atom_embed(atom_types)
        
        # Build edges
        edge_index, edge_attr = self.build_edges(coords, batch)
        edge_attr = self.edge_embed(edge_attr.to(coords.device))
        
        # Graph convolutions
        for conv in self.conv_layers:
            node_features, coords = conv(node_features, edge_index, edge_attr, coords)
        
        # Global pooling
        graph_features = []
        for b in torch.unique(batch):
            mask = batch == b
            graph_features.append(node_features[mask].mean(dim=0))
        graph_features = torch.stack(graph_features)
        
        # Get latent parameters
        latent_params = self.graph_mlp(graph_features)
        mu, logvar = latent_params.chunk(2, dim=-1)
        
        return mu, logvar
    
    def build_edges(self, coords, batch, cutoff=8.0):
        """Build edge index (simplified)."""
        edge_index = []
        edge_attr = []
        
        unique_batches = torch.unique(batch)
        offset = 0
        for b in unique_batches:
            mask = batch == b
            coords_batch = coords[mask]
            n = coords_batch.shape[0]
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        dist = torch.norm(coords_batch[i] - coords_batch[j])
                        if dist < cutoff:
                            edge_index.append([offset + i, offset + j])
                            edge_attr.append(dist.item())
            offset += n
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().to(coords.device)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32).unsqueeze(-1)
        
        return edge_index, edge_attr


class CDVAE(nn.Module):
    """
    Crystal Diffusion Variational Autoencoder.
    
    Combines an encoder that compresses crystals to a latent space
    with a diffusion decoder that generates new structures.
    """
    
    def __init__(self, config: GenerativeModelConfig):
        super().__init__()
        self.config = config
        self.encoder = CDVAEEncoder(config)
        self.decoder = CDVAEDiffusion(config)
        
        # Property prediction head (optional)
        self.property_head = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        
        # Diffusion parameters
        self.num_diffusion_steps = 1000
        self.beta = torch.linspace(1e-4, 0.02, self.num_diffusion_steps)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
    
    def encode(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode crystal to latent space."""
        return self.encoder(coords, atom_types, batch)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(
        self,
        z: torch.Tensor,
        num_atoms: List[int],
        num_steps: int = 100
    ) -> List[CrystalStructure]:
        """
        Decode latent vectors to crystal structures via diffusion.
        
        Args:
            z: (batch_size, latent_dim) latent vectors
            num_atoms: List of number of atoms for each structure
            num_steps: Number of diffusion steps
        
        Returns:
            List of generated CrystalStructure objects
        """
        device = z.device
        batch_size = z.shape[0]
        
        structures = []
        
        for i in range(batch_size):
            n_atoms = num_atoms[i]
            
            # Initialize random coordinates and atom types
            coords = torch.randn(n_atoms, 3, device=device) * 10
            atom_types = torch.randint(0, self.config.num_species, (n_atoms,), device=device)
            batch = torch.zeros(n_atoms, dtype=torch.long, device=device)
            
            # Expand latent vector to node level
            z_expanded = z[i:i+1].expand(n_atoms, -1)
            
            # Reverse diffusion process
            for t in reversed(range(num_steps)):
                t_tensor = torch.full((1,), t, device=device)
                
                # Predict noise
                coord_noise, type_logits = self.decoder(coords, atom_types, t_tensor, batch)
                
                # Denoise
                alpha_t = self.alpha_bar[t]
                alpha_t_prev = self.alpha_bar[t-1] if t > 0 else torch.tensor(1.0)
                
                beta_t = self.beta[t]
                alpha_t_val = self.alpha[t]
                
                # Update coordinates
                coords = (coords - beta_t / torch.sqrt(1 - alpha_t) * coord_noise) / torch.sqrt(alpha_t_val)
                
                if t > 0:
                    noise = torch.randn_like(coords)
                    sigma_t = torch.sqrt(beta_t)
                    coords = coords + sigma_t * noise
                
                # Update atom types (discrete diffusion)
                if t % 10 == 0:  # Update types periodically
                    atom_types = torch.argmax(type_logits, dim=-1)
            
            # Create structure
            structure = CrystalStructure(
                lattice=np.eye(3) * 10,  # Default lattice
                frac_coords=coords.detach().cpu().numpy(),
                atomic_numbers=atom_types.detach().cpu().numpy() + 1,
            )
            structures.append(structure)
        
        return structures
    
    def forward(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        batch: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training."""
        # Encode
        mu, logvar = self.encode(coords, atom_types, batch)
        z = self.reparameterize(mu, logvar)
        
        # Diffusion training (simplified)
        # Sample random timestep
        t = torch.randint(0, self.num_diffusion_steps, (batch.max().item() + 1,), device=coords.device)
        t_expanded = t[batch]
        
        # Add noise
        noise = torch.randn_like(coords)
        alpha_bar_t = self.alpha_bar.to(coords.device)[t_expanded][:, None]
        noisy_coords = torch.sqrt(alpha_bar_t) * coords + torch.sqrt(1 - alpha_bar_t) * noise
        
        # Predict noise
        pred_noise, type_logits = self.decoder(noisy_coords, atom_types, t_expanded, batch)
        
        # Property prediction
        graph_z = z.mean(dim=0, keepdim=True)  # Simplified pooling
        pred_property = self.property_head(graph_z)
        
        return {
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'pred_noise': pred_noise,
            'target_noise': noise,
            'type_logits': type_logits,
            'pred_property': pred_noise,
        }
    
    def generate(
        self,
        num_structures: int = 10,
        target_composition: Optional[str] = None,
        num_atoms_range: Tuple[int, int] = (10, 50),
        device: str = 'cpu'
    ) -> List[CrystalStructure]:
        """Generate new crystal structures."""
        z = torch.randn(num_structures, self.config.latent_dim, device=device)
        
        # Sample number of atoms
        num_atoms = [
            np.random.randint(num_atoms_range[0], num_atoms_range[1])
            for _ in range(num_structures)
        ]
        
        return self.decode(z, num_atoms)


# ============================================================================
# DiffCSP: Diffusion Model for Crystal Structure Prediction
# ============================================================================

class DiffCSP(nn.Module):
    """
    Diffusion Model for Crystal Structure Prediction.
    
    Generates crystal structures conditioned on composition.
    """
    
    def __init__(self, config: GenerativeModelConfig):
        super().__init__()
        self.config = config
        
        # Composition encoder
        self.composition_encoder = nn.Sequential(
            nn.Linear(config.num_species, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        
        # Diffusion model (similar to CDVAE decoder)
        self.diffusion = CDVAEDiffusion(config)
        
        # Lattice predictor
        self.lattice_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 6),  # 6 lattice parameters
        )
    
    def composition_to_features(self, composition: torch.Tensor) -> torch.Tensor:
        """Convert composition vector to features."""
        return self.composition_encoder(composition)
    
    def forward(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        composition: torch.Tensor,
        batch: torch.Tensor,
        t: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with composition conditioning."""
        # Encode composition
        comp_features = self.composition_encoder(composition)
        
        # Diffusion with conditioning
        coord_noise, type_logits = self.diffusion(coords, atom_types, t, batch)
        
        # Predict lattice
        lattice_params = self.lattice_encoder(comp_features)
        
        return {
            'coord_noise': coord_noise,
            'type_logits': type_logits,
            'lattice_params': lattice_params,
        }
    
    def generate(
        self,
        composition: Union[str, Dict[str, float]],
        num_structures: int = 10,
        device: str = 'cpu'
    ) -> List[CrystalStructure]:
        """
        Generate structures for a given composition.
        
        Args:
            composition: Chemical formula (e.g., "Li3PS4") or dict of fractions
            num_structures: Number of structures to generate
            device: Device to use
        """
        # Parse composition
        if isinstance(composition, str):
            comp_dict = self._parse_composition(composition)
        else:
            comp_dict = composition
        
        # Create composition vector
        comp_vector = torch.zeros(self.config.num_species, device=device)
        for elem, frac in comp_dict.items():
            z = self._element_to_z(elem)
            if z < self.config.num_species:
                comp_vector[z] = frac
        
        # Generate via diffusion
        # This is a simplified version - full implementation would use
        # the composition-conditioned diffusion process
        
        structures = []
        for _ in range(num_structures):
            n_atoms = sum(int(v * 20) for v in comp_dict.values())  # Estimate
            n_atoms = max(10, min(n_atoms, self.config.max_atoms))
            
            structure = self._generate_single(comp_vector, n_atoms, device)
            structures.append(structure)
        
        return structures
    
    def _parse_composition(self, formula: str) -> Dict[str, float]:
        """Parse chemical formula to element fractions."""
        # Simplified parsing - would use proper formula parser in production
        import re
        pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
        matches = re.findall(pattern, formula)
        
        result = {}
        for elem, count in matches:
            result[elem] = float(count) if count else 1.0
        
        # Normalize
        total = sum(result.values())
        return {k: v / total for k, v in result.items()}
    
    def _element_to_z(self, element: str) -> int:
        """Convert element symbol to atomic number."""
        try:
            if HAS_PYMATGEN:
                return Element(element).Z - 1  # 0-indexed
            else:
                # Basic element mapping
                elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                           'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca']
                return elements.index(element) if element in elements else 0
        except:
            return 0
    
    def _generate_single(
        self,
        comp_vector: torch.Tensor,
        n_atoms: int,
        device: str
    ) -> CrystalStructure:
        """Generate a single structure."""
        # Simplified generation - would use full diffusion in production
        coords = torch.randn(n_atoms, 3, device=device) * 5
        
        # Sample atom types based on composition
        atom_types = torch.multinomial(
            comp_vector.unsqueeze(0).expand(n_atoms, -1),
            1
        ).squeeze(-1)
        
        return CrystalStructure(
            lattice=np.eye(3) * 10,
            frac_coords=coords.cpu().numpy(),
            atomic_numbers=atom_types.cpu().numpy() + 1,
            composition=str(comp_vector.cpu().numpy())
        )


# ============================================================================
# MatterGen: Generative Model for Materials Design
# ============================================================================

class MatterGen(nn.Module):
    """
    MatterGen: A generative model for materials design.
    
    Uses a transformer-based architecture with geometric priors.
    """
    
    def __init__(self, config: GenerativeModelConfig):
        super().__init__()
        self.config = config
        
        # Atom and position embeddings
        self.atom_embed = nn.Embedding(config.num_species, config.hidden_dim)
        self.pos_embed = nn.Linear(3, config.hidden_dim)
        
        # Transformer encoder
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
        
        # Output heads
        self.coord_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 3),
        )
        
        self.type_head = nn.Linear(config.hidden_dim, config.num_species)
        self.lattice_head = nn.Linear(config.hidden_dim, 6)
    
    def forward(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            coords: (batch, n_atoms, 3) coordinates
            atom_types: (batch, n_atoms) atom types
            mask: (batch, n_atoms) attention mask
        """
        # Embeddings
        atom_features = self.atom_embed(atom_types)
        pos_features = self.pos_embed(coords)
        
        # Combine
        features = atom_features + pos_features
        
        # Transformer
        features = self.transformer(features, src_key_padding_mask=mask)
        
        # Predictions
        coord_update = self.coord_head(features)
        type_logits = self.type_head(features)
        
        # Lattice prediction (pooled)
        pooled = features.mean(dim=1)
        lattice_params = self.lattice_head(pooled)
        
        return {
            'coord_update': coord_update,
            'type_logits': type_logits,
            'lattice_params': lattice_params,
            'features': features,
        }
    
    def generate(
        self,
        num_structures: int = 10,
        num_atoms: int = 20,
        num_steps: int = 50,
        device: str = 'cpu'
    ) -> List[CrystalStructure]:
        """Generate structures using iterative refinement."""
        structures = []
        
        for _ in range(num_structures):
            # Initialize random structure
            coords = torch.randn(1, num_atoms, 3, device=device) * 5
            atom_types = torch.randint(0, self.config.num_species, (1, num_atoms), device=device)
            
            # Iterative refinement
            for _ in range(num_steps):
                output = self.forward(coords, atom_types)
                coords = coords + 0.1 * output['coord_update']
                atom_types = torch.argmax(output['type_logits'], dim=-1)
            
            # Build lattice from predicted parameters
            lattice_params = output['lattice_params'][0].cpu().detach().numpy()
            # Convert 6 parameters to 3x3 matrix (simplified)
            lattice = np.diag(np.abs(lattice_params[:3]) + 3)
            
            structure = CrystalStructure(
                lattice=lattice,
                frac_coords=coords[0].cpu().detach().numpy(),
                atomic_numbers=atom_types[0].cpu().detach().numpy() + 1,
            )
            structures.append(structure)
        
        return structures


# ============================================================================
# Structure Generator Interface
# ============================================================================

class StructureGenerator:
    """
    Unified interface for crystal structure generation.
    
    Supports multiple generative models and provides a consistent API.
    """
    
    SUPPORTED_MODELS = ['cdvae', 'diffcsp', 'mattergen', 'random']
    
    def __init__(
        self,
        model_type: str = 'cdvae',
        config: Optional[GenerativeModelConfig] = None,
        device: str = 'cpu'
    ):
        """
        Initialize structure generator.
        
        Args:
            model_type: Type of generative model
            config: Model configuration
            device: Device to use ('cpu' or 'cuda')
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Supported: {self.SUPPORTED_MODELS}")
        
        self.model_type = model_type
        self.config = config or GenerativeModelConfig(model_type=model_type)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = self._create_model()
        if self.model is not None:
            self.model.to(self.device)
        
        self.is_trained = False
    
    def _create_model(self) -> Optional[nn.Module]:
        """Create the generative model."""
        if self.model_type == 'cdvae':
            return CDVAE(self.config)
        elif self.model_type == 'diffcsp':
            return DiffCSP(self.config)
        elif self.model_type == 'mattergen':
            return MatterGen(self.config)
        elif self.model_type == 'random':
            return None  # Random generation doesn't need a model
        
        return None
    
    def train(
        self,
        structures: List[CrystalStructure],
        num_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        validation_split: float = 0.1,
    ) -> Dict[str, List[float]]:
        """
        Train the generative model.
        
        Args:
            structures: Training structures
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            validation_split: Fraction of data for validation
        
        Returns:
            Training history dictionary
        """
        if self.model is None:
            raise ValueError("No model to train (random generator)")
        
        # Prepare dataset
        dataset = CrystalDataset(structures, self.config)
        
        # Split train/val
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_losses = []
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(
                    batch['coords'].to(self.device),
                    batch['atom_types'].to(self.device),
                    batch['batch'].to(self.device)
                )
                
                # Compute loss
                loss = self._compute_loss(outputs, batch)
                
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            self.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    outputs = self.model(
                        batch['coords'].to(self.device),
                        batch['atom_types'].to(self.device),
                        batch['batch'].to(self.device)
                    )
                    loss = self._compute_loss(outputs, batch)
                    val_losses.append(loss.item())
            
            history['train_loss'].append(np.mean(train_losses))
            history['val_loss'].append(np.mean(val_losses))
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"train_loss={history['train_loss'][-1]:.4f}, "
                      f"val_loss={history['val_loss'][-1]:.4f}")
        
        self.is_trained = True
        return history
    
    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute training loss."""
        loss = 0.0
        
        # Noise prediction loss
        if 'pred_noise' in outputs and 'target_noise' in outputs:
            loss += F.mse_loss(outputs['pred_noise'], outputs['target_noise'])
        
        # KL divergence for VAE
        if 'mu' in outputs and 'logvar' in outputs:
            kl_loss = -0.5 * torch.sum(
                1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp()
            )
            loss += kl_loss / outputs['mu'].shape[0]
        
        return loss
    
    def generate(
        self,
        num_structures: int = 10,
        target_composition: Optional[str] = None,
        num_atoms_range: Tuple[int, int] = (10, 50),
        temperature: float = 1.0,
    ) -> List[CrystalStructure]:
        """
        Generate new crystal structures.
        
        Args:
            num_structures: Number of structures to generate
            target_composition: Target chemical formula (optional)
            num_atoms_range: Range of number of atoms
            temperature: Sampling temperature
        
        Returns:
            List of generated structures
        """
        if self.model_type == 'random':
            return self._generate_random(num_structures, num_atoms_range)
        
        if not self.is_trained:
            warnings.warn("Model not trained. Using random initialization.")
        
        self.model.eval()
        
        with torch.no_grad():
            if self.model_type == 'cdvae':
                return self.model.generate(
                    num_structures=num_structures,
                    target_composition=target_composition,
                    num_atoms_range=num_atoms_range,
                    device=self.device
                )
            elif self.model_type == 'diffcsp':
                return self.model.generate(
                    composition=target_composition or "SiO2",
                    num_structures=num_structures,
                    device=self.device
                )
            elif self.model_type == 'mattergen':
                return self.model.generate(
                    num_structures=num_structures,
                    num_atoms=num_atoms_range[0],
                    device=self.device
                )
        
        return []
    
    def _generate_random(
        self,
        num_structures: int,
        num_atoms_range: Tuple[int, int]
    ) -> List[CrystalStructure]:
        """Generate random structures for baseline comparison."""
        structures = []
        
        for _ in range(num_structures):
            n_atoms = np.random.randint(num_atoms_range[0], num_atoms_range[1])
            
            # Random lattice
            lattice = np.random.rand(3, 3) * 10 + np.eye(3) * 5
            
            # Random fractional coordinates
            frac_coords = np.random.rand(n_atoms, 3)
            
            # Random atom types (common elements)
            common_elements = [1, 3, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 26, 29, 30]
            atomic_numbers = np.random.choice(common_elements, n_atoms)
            
            structure = CrystalStructure(
                lattice=lattice,
                frac_coords=frac_coords,
                atomic_numbers=atomic_numbers,
            )
            structures.append(structure)
        
        return structures
    
    def save(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_type': self.model_type,
            'config': self.config,
            'model_state': self.model.state_dict() if self.model else None,
            'is_trained': self.is_trained,
        }
        
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model_type = checkpoint['model_type']
        self.config = checkpoint['config']
        self.is_trained = checkpoint['is_trained']
        
        if self.model and checkpoint['model_state']:
            self.model.load_state_dict(checkpoint['model_state'])
        
        print(f"Model loaded from {path}")


class CrystalDataset(Dataset):
    """Dataset for crystal structures."""
    
    def __init__(
        self,
        structures: List[CrystalStructure],
        config: GenerativeModelConfig
    ):
        self.structures = structures
        self.config = config
    
    def __len__(self) -> int:
        return len(self.structures)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        struct = self.structures[idx]
        
        coords = torch.tensor(struct.frac_coords, dtype=torch.float32)
        atom_types = torch.tensor(struct.atomic_numbers, dtype=torch.long) - 1  # 0-indexed
        
        return {
            'coords': coords,
            'atom_types': atom_types,
            'batch': torch.zeros(coords.shape[0], dtype=torch.long),
        }


def collate_crystal_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for batching crystal structures."""
    coords = []
    atom_types = []
    batch_indices = []
    
    for i, item in enumerate(batch):
        coords.append(item['coords'])
        atom_types.append(item['atom_types'])
        batch_indices.append(torch.full((item['coords'].shape[0],), i, dtype=torch.long))
    
    return {
        'coords': torch.cat(coords, dim=0),
        'atom_types': torch.cat(atom_types, dim=0),
        'batch': torch.cat(batch_indices, dim=0),
    }


# ============================================================================
# Utility Functions
# ============================================================================

def load_pretrained_model(
    model_name: str,
    checkpoint_path: Optional[str] = None,
    device: str = 'cpu'
) -> StructureGenerator:
    """
    Load a pre-trained generative model.
    
    Args:
        model_name: Name of the model ('cdvae', 'diffcsp', 'mattergen')
        checkpoint_path: Path to checkpoint (optional)
        device: Device to use
    
    Returns:
        StructureGenerator instance
    """
    config = GenerativeModelConfig(model_type=model_name)
    generator = StructureGenerator(model_name, config, device)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        generator.load(checkpoint_path)
    
    return generator


def generate_structures_for_screening(
    composition_space: List[str],
    num_per_composition: int = 10,
    model_type: str = 'cdvae'
) -> Dict[str, List[CrystalStructure]]:
    """
    Generate structures for a set of compositions for screening.
    
    Args:
        composition_space: List of chemical formulas
        num_per_composition: Number of structures per composition
        model_type: Generative model to use
    
    Returns:
        Dictionary mapping compositions to generated structures
    """
    generator = StructureGenerator(model_type=model_type)
    
    results = {}
    for composition in composition_space:
        print(f"Generating structures for {composition}...")
        structures = generator.generate(
            num_structures=num_per_composition,
            target_composition=composition
        )
        results[composition] = structures
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Generative Models Module")
    print("=" * 50)
    
    # Create a simple generator
    config = GenerativeModelConfig(model_type='cdvae', num_structures=5)
    generator = StructureGenerator('cdvae', config)
    
    # Generate some random structures
    print("\nGenerating random structures for baseline...")
    structures = generator.generate(num_structures=5)
    
    for i, struct in enumerate(structures):
        print(f"\nStructure {i+1}:")
        print(f"  Atoms: {len(struct.atomic_numbers)}")
        print(f"  Composition: {struct.composition}")
        print(f"  Lattice shape: {struct.lattice.shape}")
