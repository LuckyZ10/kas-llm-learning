#!/usr/bin/env python3
"""
Advanced Structural Analysis
============================

Structural characterization methods:
- Ring statistics
- Voronoi analysis
- Bond-orientational order parameters
- Common neighbor analysis

References:
- Honeycutt-Andersen pair analysis
- Steinhardt bond-orientational order
- Voronoi polyhedra analysis
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
import logging
from scipy.spatial import Voronoi, ConvexHull, Delaunay
from scipy.spatial.distance import cdist
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StructuralAnalysisConfig:
    """Configuration for structural analysis.
    
    Attributes:
        cutoff: Neighbor cutoff distance (Å)
        max_ring_size: Maximum ring size to search
        voronoi_tolerance: Tolerance for Voronoi face area
        q_lmax: Maximum l for bond-orientational order
    """
    cutoff: float = 3.5
    max_ring_size: int = 12
    voronoi_tolerance: float = 0.01
    q_lmax: int = 6
    cn_method: str = "cutoff"  # or 'voronoi'


class RingStatistics:
    """Compute ring statistics in disordered systems."""
    
    def __init__(self, config: StructuralAnalysisConfig):
        self.config = config
    
    def build_bond_network(self, positions: np.ndarray,
                          symbols: Optional[List[str]] = None,
                          box: Optional[np.ndarray] = None) -> Dict[int, List[int]]:
        """Build bond network from positions."""
        n_atoms = len(positions)
        bonds = {i: [] for i in range(n_atoms)}
        
        # Compute distance matrix
        if box is not None:
            # Minimum image convention
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            diff -= box * np.round(diff / box)
            dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
        else:
            dist_matrix = cdist(positions, positions)
        
        # Build bonds
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if dist_matrix[i, j] < self.config.cutoff:
                    bonds[i].append(j)
                    bonds[j].append(i)
        
        return bonds
    
    def find_primitive_rings(self, bonds: Dict[int, List[int]],
                            start_atom: int,
                            max_depth: Optional[int] = None) -> List[List[int]]:
        """Find primitive rings (shortest paths) starting from atom.
        
        Uses modified breadth-first search to find shortest cycles.
        """
        if max_depth is None:
            max_depth = self.config.max_ring_size
        
        rings = []
        visited = set()
        
        # BFS
        queue = deque([(start_atom, [start_atom])])
        
        while queue:
            current, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            for neighbor in bonds[current]:
                if neighbor == path[0] and len(path) >= 3:
                    # Found a ring
                    ring = path[:]
                    # Normalize ring representation
                    ring = self._normalize_ring(ring)
                    if ring not in rings:
                        rings.append(ring)
                elif neighbor not in path and len(path) < max_depth:
                    queue.append((neighbor, path + [neighbor]))
        
        return rings
    
    def _normalize_ring(self, ring: List[int]) -> Tuple[int, ...]:
        """Normalize ring to canonical form."""
        # Rotate to start with smallest index
        min_idx = ring.index(min(ring))
        rotated = ring[min_idx:] + ring[:min_idx]
        
        # Try both directions
        reversed_ring = list(reversed(rotated))
        reversed_rotated = reversed_ring[reversed_ring.index(min(reversed_ring)):] + \
                          reversed_ring[:reversed_ring.index(min(reversed_ring))]
        
        # Return lexicographically smaller
        if tuple(rotated) < tuple(reversed_rotated):
            return tuple(rotated)
        return tuple(reversed_rotated)
    
    def compute_ring_statistics(self, positions: np.ndarray,
                               box: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Compute complete ring statistics."""
        bonds = self.build_bond_network(positions, box=box)
        
        all_rings = []
        ring_sizes = Counter()
        
        # Find rings from each atom
        n_atoms = len(positions)
        for i in range(min(n_atoms, 100)):  # Sample subset for efficiency
            rings = self.find_primitive_rings(bonds, i)
            for ring in rings:
                if ring not in all_rings:
                    all_rings.append(ring)
                    ring_sizes[len(ring)] += 1
        
        # Compute statistics
        sizes = list(ring_sizes.keys())
        counts = [ring_sizes[s] for s in sizes]
        
        if sizes:
            mean_size = np.average(sizes, weights=counts)
            std_size = np.sqrt(np.average((np.array(sizes) - mean_size) ** 2, weights=counts))
        else:
            mean_size = std_size = 0
        
        return {
            'ring_sizes': dict(ring_sizes),
            'mean_size': mean_size,
            'std_size': std_size,
            'total_rings': len(all_rings),
            'rings': all_rings[:100]  # Limit stored rings
        }
    
    def compute_ring_order_parameter(self, ring: List[int],
                                    positions: np.ndarray,
                                    box: Optional[np.ndarray] = None) -> float:
        """Compute planarity of a ring (0 = planar, 1 = non-planar)."""
        if len(ring) < 3:
            return 0.0
        
        ring_positions = positions[list(ring)]
        
        # Center positions
        center = np.mean(ring_positions, axis=0)
        centered = ring_positions - center
        
        # Compute covariance matrix
        cov = np.dot(centered.T, centered) / len(ring)
        
        # Eigenvalues
        eigenvalues = np.sort(np.linalg.eigvalsh(cov))
        
        # Planarity: ratio of smallest to largest eigenvalue
        if eigenvalues[-1] > 1e-10:
            planarity = eigenvalues[0] / eigenvalues[-1]
        else:
            planarity = 0
        
        return planarity


class VoronoiAnalysis:
    """Voronoi tessellation analysis."""
    
    def __init__(self, config: StructuralAnalysisConfig):
        self.config = config
    
    def compute_voronoi(self, positions: np.ndarray,
                       box: Optional[np.ndarray] = None) -> Voronoi:
        """Compute Voronoi tessellation."""
        if box is not None:
            # For periodic systems, replicate positions
            positions = self._replicate_for_pbc(positions, box)
        
        vor = Voronoi(positions)
        
        return vor
    
    def _replicate_for_pbc(self, positions: np.ndarray,
                          box: np.ndarray) -> np.ndarray:
        """Replicate positions for PBC-aware Voronoi."""
        # Add periodic images
        replicated = [positions]
        
        # In 3D, add 26 periodic images
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    shift = np.array([dx, dy, dz]) * box
                    replicated.append(positions + shift)
        
        return np.vstack(replicated)
    
    def compute_voronoi_indices(self, vor: Voronoi,
                               n_original: int) -> List[Tuple[int, ...]]:
        """Compute Voronoi indices ⟨n₃, n₄, n₅, n₆, ...⟩.
        
        nᵢ is the number of faces with i edges.
        """
        indices = []
        
        for i in range(n_original):
            region = vor.regions[vor.point_region[i]]
            
            if -1 in region or len(region) == 0:
                indices.append(None)
                continue
            
            # Count face edges
            face_edges = defaultdict(int)
            
            for ridge in vor.ridge_dict.values():
                if i in ridge:
                    face_vertices = vor.vertices[vor.ridge_dict[ridge]]
                    n_vertices = len(face_vertices)
                    if 3 <= n_vertices <= 10:
                        face_edges[n_vertices] += 1
            
            # Create index tuple
            index = tuple([face_edges.get(n, 0) for n in range(3, 10)])
            indices.append(index)
        
        return indices
    
    def analyze_voronoi_statistics(self, positions: np.ndarray,
                                  box: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Complete Voronoi analysis."""
        vor = self.compute_voronoi(positions, box)
        
        indices = self.compute_voronoi_indices(vor, len(positions))
        
        # Count index frequencies
        valid_indices = [idx for idx in indices if idx is not None]
        index_counts = Counter(valid_indices)
        
        # Common indices in metallic glasses
        common_indices = {
            (0, 0, 12, 0, 0, 0, 0): 'icosahedron',
            (0, 3, 6, 0, 0, 0, 0): 'trigonal prism',
            (0, 2, 8, 0, 0, 0, 0): 'Archimedean octahedron',
        }
        
        # Compute average number of faces (coordination)
        n_faces = [sum(idx) if idx else 0 for idx in indices]
        
        return {
            'indices': indices,
            'index_frequencies': dict(index_counts),
            'mean_coordination': np.mean(n_faces),
            'std_coordination': np.std(n_faces),
            'fraction_icosahedral': index_counts.get((0, 0, 12, 0, 0, 0, 0), 0) / len(valid_indices) if valid_indices else 0
        }
    
    def compute_free_volume(self, vor: Voronoi,
                           positions: np.ndarray,
                           atomic_radii: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute free volume per atom."""
        volumes = np.zeros(len(positions))
        
        for i in range(len(positions)):
            region = vor.regions[vor.point_region[i]]
            
            if -1 in region or len(region) == 0:
                continue
            
            # Compute Voronoi cell volume
            vertices = vor.vertices[region]
            if len(vertices) >= 4:
                hull = ConvexHull(vertices)
                volumes[i] = hull.volume
        
        # Subtract atomic volume if radii provided
        if atomic_radii is not None:
            atomic_volumes = 4/3 * np.pi * atomic_radii ** 3
            volumes -= atomic_volumes
        
        return volumes


class BondOrientationalOrder:
    """Steinhardt bond-orientational order parameters."""
    
    def __init__(self, config: StructuralAnalysisConfig):
        self.config = config
        self.l_values = list(range(1, config.q_lmax + 1))
    
    def compute_qlm(self, positions: np.ndarray, i: int,
                   neighbors: List[int], l: int) -> np.ndarray:
        """Compute spherical harmonics qₗₘ for atom i."""
        from scipy.special import sph_harm
        
        n_neighbors = len(neighbors)
        if n_neighbors == 0:
            return np.zeros(2 * l + 1, dtype=complex)
        
        qlm = np.zeros(2 * l + 1, dtype=complex)
        
        for j in neighbors:
            # Vector from i to j
            r = positions[j] - positions[i]
            r_norm = np.linalg.norm(r)
            
            if r_norm < 1e-10:
                continue
            
            # Spherical coordinates
            theta = np.arccos(np.clip(r[2] / r_norm, -1, 1))
            phi = np.arctan2(r[1], r[0])
            
            for m in range(-l, l + 1):
                qlm[m + l] += sph_harm(m, l, phi, theta)
        
        return qlm / n_neighbors
    
    def compute_ql(self, positions: np.ndarray,
                  bonds: Dict[int, List[int]],
                  l: int) -> np.ndarray:
        """Compute qₗ parameter for all atoms."""
        n_atoms = len(positions)
        ql_values = np.zeros(n_atoms)
        
        for i in range(n_atoms):
            qlm = self.compute_qlm(positions, i, bonds.get(i, []), l)
            
            # qₗ = sqrt(4π/(2l+1) Σₘ |qₗₘ|²)
            ql_values[i] = np.sqrt(4 * np.pi / (2 * l + 1) * np.sum(np.abs(qlm) ** 2))
        
        return ql_values
    
    def compute_local_ql(self, positions: np.ndarray,
                        bonds: Dict[int, List[int]],
                        l: int) -> np.ndarray:
        """Compute locally-averaged qₗ (QL)."""
        ql = self.compute_ql(positions, bonds, l)
        
        n_atoms = len(positions)
        Ql = np.zeros(n_atoms)
        
        for i in range(n_atoms):
            neighbors = bonds.get(i, [])
            
            # Average over neighbors and self
            values = [ql[i]] + [ql[j] for j in neighbors if j < n_atoms]
            Ql[i] = np.mean(values)
        
        return Ql
    
    def compute_wl(self, positions: np.ndarray,
                  bonds: Dict[int, List[int]],
                  l: int) -> np.ndarray:
        """Compute third-order invariants wₗ."""
        from scipy.special import wigner_3j
        
        n_atoms = len(positions)
        wl_values = np.zeros(n_atoms)
        
        for i in range(n_atoms):
            qlm = self.compute_qlm(positions, i, bonds.get(i, []), l)
            
            # Compute wigner 3j symbols sum
            w = 0.0
            for m1 in range(-l, l + 1):
                for m2 in range(-l, l + 1):
                    m3 = -m1 - m2
                    if abs(m3) <= l:
                        w += qlm[m1 + l] * qlm[m2 + l] * qlm[m3 + l] * \
                             float(wigner_3j(l, l, l, m1, m2, m3))
            
            # Normalize
            q_norm = np.sum(np.abs(qlm) ** 2)
            if q_norm > 1e-10:
                wl_values[i] = w / (q_norm ** (3/2))
        
        return wl_values
    
    def analyze_structural_order(self, positions: np.ndarray,
                                box: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Complete bond-orientational order analysis."""
        # Build bonds
        ring_stats = RingStatistics(self.config)
        bonds = ring_stats.build_bond_network(positions, box=box)
        
        results = {}
        
        for l in self.l_values:
            ql = self.compute_ql(positions, bonds, l)
            Ql = self.compute_local_ql(positions, bonds, l)
            
            results[f'q{l}'] = {
                'mean': np.mean(ql),
                'std': np.std(ql),
                'local_mean': np.mean(Ql)
            }
        
        # q₄-q₆ map for structure identification
        q4 = results['q4']['mean']
        q6 = results['q6']['mean']
        
        # Identify structure
        if q6 > 0.5 and q4 < 0.3:
            structure = 'fcc-like'
        elif q6 > 0.4 and q4 > 0.2:
            structure = 'bcc-like'
        elif q6 > 0.35 and q4 < 0.2:
            structure = 'icosahedral'
        else:
            structure = 'disordered'
        
        results['structure_type'] = structure
        
        return results


class CommonNeighborAnalysis:
    """Honeycutt-Andersen common neighbor analysis."""
    
    def __init__(self, config: StructuralAnalysisConfig):
        self.config = config
    
    def classify_bond(self, positions: np.ndarray, i: int, j: int,
                     bonds: Dict[int, List[int]],
                     box: Optional[np.ndarray] = None) -> Tuple[int, int, int]:
        """Classify bond between atoms i and j using HA indices.
        
        Returns (n, m, l) where:
        - n: number of common neighbors
        - m: number of bonds among common neighbors
        - l: longest continuous chain of bonded common neighbors
        """
        # Common neighbors
        common = set(bonds[i]) & set(bonds[j])
        n = len(common)
        
        if n == 0:
            return (0, 0, 0)
        
        # Count bonds among common neighbors
        common_list = list(common)
        m = 0
        for a in range(n):
            for b in range(a + 1, n):
                if common_list[b] in bonds[common_list[a]]:
                    m += 1
        
        # Find longest chain (simplified)
        l = self._find_longest_chain(common_list, bonds)
        
        return (n, m, l)
    
    def _find_longest_chain(self, atoms: List[int],
                           bonds: Dict[int, List[int]]) -> int:
        """Find longest chain of bonded atoms."""
        if not atoms:
            return 0
        
        # Build subgraph
        max_chain = 1
        
        for start in atoms:
            visited = {start}
            queue = deque([(start, 1)])
            
            while queue:
                current, length = queue.popleft()
                max_chain = max(max_chain, length)
                
                for neighbor in bonds[current]:
                    if neighbor in atoms and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, length + 1))
        
        return max_chain
    
    def analyze_structure(self, positions: np.ndarray,
                         box: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Complete CNA analysis."""
        ring_stats = RingStatistics(self.config)
        bonds = ring_stats.build_bond_network(positions, box=box)
        
        n_atoms = len(positions)
        bond_types = Counter()
        atom_structures = []
        
        for i in range(n_atoms):
            local_types = Counter()
            
            for j in bonds[i]:
                if j > i:  # Avoid double counting
                    ha_index = self.classify_bond(positions, i, j, bonds, box)
                    bond_types[ha_index] += 1
                    local_types[ha_index] += 1
            
            # Identify local structure
            if local_types[(4, 2, 1)] >= 6:
                atom_structures.append('fcc')
            elif local_types[(4, 4, 4)] >= 6:
                atom_structures.append('hcp')
            elif local_types[(6, 6, 6)] >= 8:
                atom_structures.append('bcc')
            elif local_types[(5, 5, 5)] >= 5:
                atom_structures.append('icosahedral')
            else:
                atom_structures.append('other')
        
        return {
            'bond_types': dict(bond_types),
            'atom_structures': atom_structures,
            'fraction_fcc': atom_structures.count('fcc') / n_atoms,
            'fraction_hcp': atom_structures.count('hcp') / n_atoms,
            'fraction_bcc': atom_structures.count('bcc') / n_atoms,
            'fraction_ico': atom_structures.count('icosahedral') / n_atoms
        }


# Import for ring search
from collections import deque


class StructuralAnalyzer:
    """Unified structural analysis interface."""
    
    def __init__(self, config: Optional[StructuralAnalysisConfig] = None):
        self.config = config or StructuralAnalysisConfig()
        self.ring_analyzer = RingStatistics(self.config)
        self.voronoi_analyzer = VoronoiAnalysis(self.config)
        self.boo_analyzer = BondOrientationalOrder(self.config)
        self.cna_analyzer = CommonNeighborAnalysis(self.config)
    
    def full_analysis(self, positions: np.ndarray,
                     symbols: Optional[List[str]] = None,
                     box: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Perform complete structural analysis."""
        results = {}
        
        # Ring statistics
        results['rings'] = self.ring_analyzer.compute_ring_statistics(positions, box)
        
        # Voronoi analysis
        results['voronoi'] = self.voronoi_analyzer.analyze_voronoi_statistics(positions, box)
        
        # Bond-orientational order
        results['boo'] = self.boo_analyzer.analyze_structural_order(positions, box)
        
        # Common neighbor analysis
        results['cna'] = self.cna_analyzer.analyze_structure(positions, box)
        
        return results
    
    def save_results(self, results: Dict, output_file: str):
        """Save analysis results."""
        # Convert to serializable format
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, tuple):
                return list(obj)
            return obj
        
        with open(output_file, 'w') as f:
            json.dump(convert(results), f, indent=2)
        
        logger.info(f"Structural analysis saved to {output_file}")


# Export public API
__all__ = [
    'StructuralAnalysisConfig',
    'RingStatistics',
    'VoronoiAnalysis',
    'BondOrientationalOrder',
    'CommonNeighborAnalysis',
    'StructuralAnalyzer'
]
