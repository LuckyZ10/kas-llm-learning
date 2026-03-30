"""
Abstract Representation Learning
================================

Hierarchical abstract representation of material knowledge.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from enum import Enum, auto
import warnings


class RepresentationLevel(Enum):
    """Levels of abstraction"""
    RAW = auto()        # Raw data
    FEATURE = auto()    # Learned features
    CONCEPT = auto()    # Conceptual representation
    SCHEMA = auto()     # Schema-level abstraction
    META = auto()       # Meta-knowledge


@dataclass
class RepresentationConfig:
    """Configuration for representation learning"""
    input_dim: int = 20
    abstract_dim: int = 32
    num_levels: int = 4
    use_vq: bool = True  # Vector quantization
    vq_num_embeddings: int = 512
    commitment_cost: float = 0.25
    dropout: float = 0.1


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer for discrete representations.
    
    Based on VQ-VAE.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Quantize continuous vectors.
        
        Args:
            z: Continuous vectors (..., embedding_dim)
            
        Returns:
            Quantized vectors and loss info
        """
        # Flatten
        flat_z = z.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (
            torch.sum(flat_z**2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight**2, dim=1) -
            2 * torch.matmul(flat_z, self.embeddings.weight.t())
        )
        
        # Find nearest embeddings
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embeddings(encoding_indices)
        
        # Reshape
        quantized = quantized.view(z.shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        
        info = {
            'vq_loss': loss,
            'encoding_indices': encoding_indices.view(z.shape[:-1]),
            'perplexity': self._compute_perplexity(encoding_indices)
        }
        
        return quantized, info
    
    def _compute_perplexity(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute codebook usage perplexity"""
        encodings = F.one_hot(indices, self.num_embeddings).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity
    
    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """Get codebook entries for indices"""
        return self.embeddings(indices)


class HierarchicalEncoder(nn.Module):
    """Hierarchical encoder for multi-level abstraction"""
    
    def __init__(self, config: RepresentationConfig):
        super().__init__()
        
        self.config = config
        self.encoders = nn.ModuleList()
        self.level_dims = []
        
        prev_dim = config.input_dim
        
        for i in range(config.num_levels):
            # Decreasing dimensions for higher abstraction
            dim = config.abstract_dim * (2 ** (config.num_levels - 1 - i))
            self.level_dims.append(dim)
            
            self.encoders.append(nn.Sequential(
                nn.Linear(prev_dim, dim * 2),
                nn.LayerNorm(dim * 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(dim * 2, dim),
                nn.LayerNorm(dim)
            ))
            
            prev_dim = dim
        
        # Vector quantizers for discrete representations
        if config.use_vq:
            self.quantizers = nn.ModuleList([
                VectorQuantizer(
                    config.vq_num_embeddings,
                    dim,
                    config.commitment_cost
                )
                for dim in self.level_dims
            ])
        else:
            self.quantizers = None
    
    def forward(
        self,
        x: torch.Tensor,
        target_level: Optional[int] = None,
        use_quantization: bool = True
    ) -> Union[torch.Tensor, List[torch.Tensor], Dict]:
        """
        Encode to hierarchical representations.
        
        Args:
            x: Input tensor
            target_level: Target abstraction level (None for all)
            use_quantization: Whether to use VQ
            
        Returns:
            Encoded representation(s)
        """
        representations = []
        vq_losses = []
        
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            
            # Apply quantization
            if use_quantization and self.quantizers is not None:
                x, vq_info = self.quantizers[i](x)
                vq_losses.append(vq_info['vq_loss'])
            
            representations.append(x)
            
            if target_level == i:
                return {
                    'representation': x,
                    'vq_loss': sum(vq_losses) if vq_losses else torch.tensor(0.0),
                    'level': i
                }
        
        if target_level is None:
            return {
                'representations': representations,
                'most_abstract': representations[-1],
                'vq_loss': sum(vq_losses) if vq_losses else torch.tensor(0.0),
                'all_levels': representations
            }
        
        return representations[-1]


class HierarchicalDecoder(nn.Module):
    """Hierarchical decoder for reconstruction"""
    
    def __init__(self, config: RepresentationConfig):
        super().__init__()
        
        self.config = config
        self.decoders = nn.ModuleList()
        
        for i in range(config.num_levels):
            dim = config.abstract_dim * (2 ** i)
            
            if i == config.num_levels - 1:
                next_dim = config.input_dim
            else:
                next_dim = config.abstract_dim * (2 ** (i + 1))
            
            self.decoders.append(nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.LayerNorm(dim * 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(dim * 2, next_dim),
                nn.LayerNorm(next_dim)
            ))
    
    def forward(
        self,
        z: torch.Tensor,
        start_level: int = 0,
        target_level: Optional[int] = None
    ) -> torch.Tensor:
        """
        Decode from representation.
        
        Args:
            z: Latent representation
            start_level: Starting abstraction level
            target_level: Target reconstruction level
            
        Returns:
            Reconstructed input
        """
        x = z
        
        end_level = target_level if target_level is not None else self.config.num_levels
        
        for i in range(start_level, min(end_level, self.config.num_levels)):
            x = self.decoders[i](x)
        
        return x


class AbstractRepresentationLearner(nn.Module):
    """
    Hierarchical abstract representation learner.
    
    Learns multi-level compressed representations of material states.
    """
    
    def __init__(self, config: Optional[RepresentationConfig] = None):
        super().__init__()
        
        self.config = config or RepresentationConfig()
        
        self.encoder = HierarchicalEncoder(self.config)
        self.decoder = HierarchicalDecoder(self.config)
        
        # Learned prior for each level
        self.priors = nn.ModuleList([
            nn.Linear(dim, dim * 2) for dim in self.encoder.level_dims
        ])
    
    def encode(
        self,
        x: torch.Tensor,
        level: Optional[int] = None,
        use_quantization: bool = True
    ) -> Union[torch.Tensor, List[torch.Tensor], Dict]:
        """
        Encode input to abstract representation.
        
        Args:
            x: Input tensor (batch, input_dim)
            level: Target abstraction level (None for all)
            use_quantization: Whether to use VQ
            
        Returns:
            Encoded representation
        """
        return self.encoder(x, level, use_quantization)
    
    def decode(
        self,
        z: torch.Tensor,
        level: int = 0
    ) -> torch.Tensor:
        """
        Decode from abstract representation.
        
        Args:
            z: Abstract representation
            level: Current abstraction level
            
        Returns:
            Reconstructed input
        """
        return self.decoder(z, start_level=level)
    
    def forward(
        self,
        x: torch.Tensor,
        use_quantization: bool = True
    ) -> Dict[str, Any]:
        """
        Full forward pass.
        
        Returns:
            Dict with reconstruction and representations
        """
        # Encode
        encode_result = self.encode(x, use_quantization=use_quantization)
        
        representations = encode_result.get('all_levels', [])
        most_abstract = encode_result.get('most_abstract')
        vq_loss = encode_result.get('vq_loss', torch.tensor(0.0))
        
        # Decode from most abstract
        if most_abstract is not None:
            reconstruction = self.decode(most_abstract, level=0)
        else:
            reconstruction = x
        
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, x)
        
        return {
            'reconstruction': reconstruction,
            'representations': representations,
            'most_abstract': most_abstract,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'total_loss': recon_loss + vq_loss
        }
    
    def compress(self, x: torch.Tensor, level: int = -1) -> torch.Tensor:
        """Compress input to specified abstraction level"""
        with torch.no_grad():
            result = self.encode(x, level=level, use_quantization=True)
            return result['representation'] if isinstance(result, dict) else result
    
    def decompress(self, z: torch.Tensor, level: int = 0) -> torch.Tensor:
        """Decompress from abstract representation"""
        with torch.no_grad():
            return self.decode(z, level=level)
    
    def get_compression_ratio(self, level: int = -1) -> float:
        """Get compression ratio for specified level"""
        if level == -1:
            compressed_dim = self.config.abstract_dim
        else:
            compressed_dim = self.config.abstract_dim * (2 ** (self.config.num_levels - 1 - level))
        
        return self.config.input_dim / compressed_dim
    
    def interpolate(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        num_steps: int = 10
    ) -> torch.Tensor:
        """
        Interpolate between two representations.
        
        Args:
            z1: First representation
            z2: Second representation
            num_steps: Number of interpolation steps
            
        Returns:
            Interpolated representations
        """
        alphas = torch.linspace(0, 1, num_steps, device=z1.device).view(-1, 1)
        z1_expanded = z1.unsqueeze(0)
        z2_expanded = z2.unsqueeze(0)
        
        interpolated = (1 - alphas) * z1_expanded + alphas * z2_expanded
        return interpolated
    
    def traverse_latent_space(
        self,
        base_z: torch.Tensor,
        dim_indices: List[int],
        num_steps: int = 10,
        traversal_range: Tuple[float, float] = (-3, 3)
    ) -> torch.Tensor:
        """
        Traverse latent space along specified dimensions.
        
        Args:
            base_z: Base latent vector
            dim_indices: Dimensions to traverse
            num_steps: Number of steps per dimension
            traversal_range: Range to traverse
            
        Returns:
            Traversed representations
        """
        traversed = []
        
        for dim_idx in dim_indices:
            values = torch.linspace(
                traversal_range[0],
                traversal_range[1],
                num_steps,
                device=base_z.device
            )
            
            for val in values:
                z_modified = base_z.clone()
                z_modified[dim_idx] = val
                traversed.append(z_modified)
        
        return torch.stack(traversed)


class ConceptLibrary:
    """
    Library of learned concepts from abstract representations.
    """
    
    def __init__(self, concept_dim: int = 32, max_concepts: int = 1000):
        self.concept_dim = concept_dim
        self.max_concepts = max_concepts
        
        self.concepts: Dict[str, torch.Tensor] = {}
        self.concept_stats: Dict[str, Dict] = {}
    
    def add_concept(
        self,
        name: str,
        representation: torch.Tensor,
        metadata: Optional[Dict] = None
    ):
        """Add a concept to the library"""
        if len(self.concepts) >= self.max_concepts:
            warnings.warn("Concept library full, removing oldest concept")
            oldest = next(iter(self.concepts))
            del self.concepts[oldest]
            del self.concept_stats[oldest]
        
        self.concepts[name] = representation.detach().cpu()
        self.concept_stats[name] = {
            'count': 1,
            'metadata': metadata or {}
        }
    
    def find_similar(
        self,
        query: torch.Tensor,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find similar concepts"""
        if not self.concepts:
            return []
        
        query = query.detach().cpu()
        
        similarities = []
        for name, concept in self.concepts.items():
            sim = F.cosine_similarity(query.unsqueeze(0), concept.unsqueeze(0))
            similarities.append((name, sim.item()))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_concept_vector(self, name: str) -> Optional[torch.Tensor]:
        """Get concept vector by name"""
        return self.concepts.get(name)
    
    def merge_concepts(
        self,
        concept_names: List[str],
        new_name: str,
        merge_type: str = 'mean'
    ):
        """Merge multiple concepts"""
        vectors = [self.concepts[name] for name in concept_names if name in self.concepts]
        
        if not vectors:
            return
        
        if merge_type == 'mean':
            merged = torch.stack(vectors).mean(dim=0)
        elif merge_type == 'max':
            merged = torch.stack(vectors).max(dim=0)[0]
        else:
            raise ValueError(f"Unknown merge type: {merge_type}")
        
        self.add_concept(new_name, merged, {
            'merged_from': concept_names,
            'merge_type': merge_type
        })


if __name__ == "__main__":
    print("Testing Abstract Representation Learning...")
    
    # Create config
    config = RepresentationConfig(
        input_dim=20,
        abstract_dim=16,
        num_levels=3,
        use_vq=True
    )
    
    # Create learner
    learner = AbstractRepresentationLearner(config)
    
    print(f"Learner created with {sum(p.numel() for p in learner.parameters())} parameters")
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, config.input_dim)
    
    result = learner(x, use_quantization=True)
    
    print(f"\nForward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Reconstruction shape: {result['reconstruction'].shape}")
    print(f"  Num representation levels: {len(result['representations'])}")
    print(f"  Most abstract shape: {result['most_abstract'].shape}")
    print(f"  Reconstruction loss: {result['recon_loss'].item():.4f}")
    print(f"  VQ loss: {result['vq_loss'].item():.4f}")
    
    # Test compression
    compressed = learner.compress(x, level=-1)
    print(f"\nCompression:")
    print(f"  Original: {x.shape}")
    print(f"  Compressed: {compressed.shape}")
    print(f"  Compression ratio: {learner.get_compression_ratio():.2f}x")
    
    # Test interpolation
    z1 = learner.compress(x[:1], level=-1)
    z2 = learner.compress(x[1:2], level=-1)
    
    interpolated = learner.interpolate(z1.squeeze(), z2.squeeze(), num_steps=5)
    print(f"\nInterpolation:")
    print(f"  Interpolated shape: {interpolated.shape}")
    
    # Test concept library
    library = ConceptLibrary(concept_dim=config.abstract_dim)
    
    for i in range(5):
        library.add_concept(f"concept_{i}", compressed[i])
    
    similar = library.find_similar(compressed[0], top_k=3)
    print(f"\nConcept library:")
    print(f"  Similar concepts to concept_0: {similar}")
    
    print("\nAll tests passed!")
