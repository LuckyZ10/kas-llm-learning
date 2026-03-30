"""
Crystal Metrics Module
======================

Evaluation metrics for generated crystal structures:
- Validity (chemical/structural)
- Uniqueness
- Novelty
- Match rate for CSP
- DFT validation proxy
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from pymatgen.core import Structure, Composition
from pymatgen.analysis.structure_matcher import StructureMatcher
from collections import defaultdict


class CrystalMetrics:
    """
    Evaluation metrics for crystal generation.
    """
    
    def __init__(
        self,
        structure_matcher: Optional[StructureMatcher] = None,
        validity_thresholds: Optional[Dict] = None
    ):
        self.structure_matcher = structure_matcher or StructureMatcher(
            ltol=0.3,
            stol=0.5,
            angle_tol=10
        )
        
        self.validity_thresholds = validity_thresholds or {
            "min_distance": 0.5,  # Angstrom
            "max_distance": 5.0,
            "max_lattice_ratio": 10.0
        }
    
    def compute_all(
        self,
        generated: List[Structure],
        reference: Optional[List[Structure]] = None
    ) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            generated: List of generated structures
            reference: Optional list of reference structures
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Validity
        validity = self.compute_validity(generated)
        metrics.update(validity)
        
        # Uniqueness
        metrics["uniqueness"] = self.compute_uniqueness(generated)
        
        # Novelty (if reference provided)
        if reference is not None:
            metrics["novelty"] = self.compute_novelty(generated, reference)
        
        # Coverage (if reference provided)
        if reference is not None:
            coverage = self.compute_coverage(generated, reference)
            metrics.update(coverage)
        
        return metrics
    
    def compute_validity(self, structures: List[Structure]) -> Dict[str, float]:
        """
        Compute validity metrics.
        
        Checks:
        - No overlapping atoms
        - Reasonable bond distances
        - Valid lattice parameters
        """
        valid_count = 0
        valid_crystal_count = 0
        
        for struct in structures:
            is_valid = self._check_validity(struct)
            if is_valid:
                valid_count += 1
                
            # Check crystal validity (can be relaxed)
            is_valid_crystal = self._check_crystal_validity(struct)
            if is_valid_crystal:
                valid_crystal_count += 1
        
        total = len(structures)
        
        return {
            "validity_strict": valid_count / total if total > 0 else 0,
            "validity_relaxed": valid_crystal_count / total if total > 0 else 0
        }
    
    def _check_validity(self, structure: Structure) -> bool:
        """Check if structure is chemically valid."""
        try:
            # Check for overlapping atoms
            for i, site1 in enumerate(structure):
                for j, site2 in enumerate(structure):
                    if i < j:
                        dist = structure.get_distance(i, j)
                        if dist < self.validity_thresholds["min_distance"]:
                            return False
            
            # Check lattice parameters
            lattice = structure.lattice
            if lattice.a <= 0 or lattice.b <= 0 or lattice.c <= 0:
                return False
            
            max_lattice = max(lattice.a, lattice.b, lattice.c)
            min_lattice = min(lattice.a, lattice.b, lattice.c)
            if max_lattice / min_lattice > self.validity_thresholds["max_lattice_ratio"]:
                return False
            
            return True
        except Exception:
            return False
    
    def _check_crystal_validity(self, structure: Structure) -> bool:
        """Relaxed validity check for crystals."""
        try:
            # Just check basic properties
            if len(structure) == 0:
                return False
            
            lattice = structure.lattice
            if lattice.volume <= 0:
                return False
            
            return True
        except Exception:
            return False
    
    def compute_uniqueness(self, structures: List[Structure]) -> float:
        """
        Compute uniqueness - fraction of unique structures.
        """
        if len(structures) <= 1:
            return 1.0
        
        unique_structures = []
        
        for struct in structures:
            is_unique = True
            for unique_struct in unique_structures:
                try:
                    if self.structure_matcher.fit(struct, unique_struct):
                        is_unique = False
                        break
                except Exception:
                    continue
            
            if is_unique:
                unique_structures.append(struct)
        
        return len(unique_structures) / len(structures)
    
    def compute_novelty(
        self,
        generated: List[Structure],
        reference: List[Structure]
    ) -> float:
        """
        Compute novelty - fraction of generated structures not in reference.
        """
        if len(generated) == 0:
            return 0.0
        
        novel_count = 0
        
        for gen_struct in generated:
            is_novel = True
            for ref_struct in reference:
                try:
                    if self.structure_matcher.fit(gen_struct, ref_struct):
                        is_novel = False
                        break
                except Exception:
                    continue
            
            if is_novel:
                novel_count += 1
        
        return novel_count / len(generated)
    
    def compute_coverage(
        self,
        generated: List[Structure],
        reference: List[Structure],
        k: int = 1
    ) -> Dict[str, float]:
        """
        Compute coverage metrics.
        
        Args:
            generated: Generated structures
            reference: Reference structures to cover
            k: Number of generated samples per reference
            
        Returns:
            Coverage metrics
        """
        if len(reference) == 0:
            return {"match_rate": 0.0, "rmse": 0.0}
        
        matched = 0
        rmse_list = []
        
        for ref_struct in reference:
            best_rmse = float('inf')
            is_matched = False
            
            for gen_struct in generated:
                try:
                    if self.structure_matcher.fit(ref_struct, gen_struct):
                        is_matched = True
                        # Compute RMSE
                        rms_dist = self.structure_matcher.get_rms_dist(ref_struct, gen_struct)
                        if rms_dist:
                            best_rmse = min(best_rmse, rms_dist[0])
                except Exception:
                    continue
            
            if is_matched:
                matched += 1
                if best_rmse != float('inf'):
                    rmse_list.append(best_rmse)
        
        match_rate = matched / len(reference)
        avg_rmse = np.mean(rmse_list) if rmse_list else 0.0
        
        return {
            "match_rate": match_rate,
            "rmse": avg_rmse
        }
    
    def compute_property_statistics(
        self,
        structures: List[Structure],
        property_fn: callable
    ) -> Dict[str, float]:
        """
        Compute statistics of a property across structures.
        
        Args:
            structures: List of structures
            property_fn: Function that takes a Structure and returns a float
            
        Returns:
            Statistics dictionary
        """
        values = []
        for struct in structures:
            try:
                val = property_fn(struct)
                if val is not None:
                    values.append(val)
            except Exception:
                continue
        
        if len(values) == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values)
        }


def compute_frechet_distance(
    real_features: np.ndarray,
    gen_features: np.ndarray
) -> float:
    """
    Compute Fréchet distance between real and generated feature distributions.
    
    Similar to FID (Fréchet Inception Distance) used in image generation.
    """
    # Compute mean and covariance
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False)
    
    # Compute Fréchet distance
    diff = mu_real - mu_gen
    
    # Product might be almost singular
    covmean, _ = sqrtm(sigma_real @ sigma_gen, disp=False)
    
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_real.shape[0]) * 1e-6
        covmean = sqrtm((sigma_real + offset) @ (sigma_gen + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fd = np.sum(diff ** 2) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    
    return float(fd)


def sqrtm(matrix: np.ndarray) -> np.ndarray:
    """Compute matrix square root."""
    from scipy import linalg
    return linalg.sqrtm(matrix)
