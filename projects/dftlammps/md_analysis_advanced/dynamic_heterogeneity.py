#!/usr/bin/env python3
"""
Dynamic Heterogeneity Analysis
==============================

Analysis of spatial and temporal fluctuations in dynamics,
relevant for glass transition and supercooled liquids.

Methods:
- Non-Gaussian parameter
- Dynamic susceptibility
- Four-point correlation functions
- Spatial clustering of mobile particles

References:
- Ediger (2000) - Spatially heterogeneous dynamics
- Kob et al. - Non-Gaussian behavior in supercooled liquids
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging
from scipy import stats, signal, ndimage
from scipy.spatial import cKDTree, Voronoi
from sklearn.cluster import DBSCAN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DynamicHeterogeneityConfig:
    """Configuration for dynamic heterogeneity analysis.
    
    Attributes:
        tau_alpha: Structural relaxation time (MD steps or time)
        tau_min: Minimum lag time for analysis
        tau_max: Maximum lag time for analysis
        n_tau: Number of lag time points
        mobile_threshold: Threshold for mobile particle identification
        n_clusters_min: Minimum clusters for analysis
    """
    tau_alpha: Optional[float] = None
    tau_min: float = 1.0
    tau_max: float = 1000.0
    n_tau: int = 50
    mobile_threshold: float = 0.5  # in units of rms displacement
    n_clusters_min: int = 10
    chunk_size: int = 1000


class DynamicHeterogeneityAnalyzer:
    """Analyze dynamic heterogeneity in glass-forming systems."""
    
    def __init__(self, config: DynamicHeterogeneityConfig):
        self.config = config
        self.tau_values: np.ndarray = np.logspace(
            np.log10(config.tau_min),
            np.log10(config.tau_max),
            config.n_tau
        )
    
    def compute_displacements(self, positions_t0: np.ndarray,
                             positions_t: np.ndarray,
                             box: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute particle displacements with PBC handling."""
        dr = positions_t - positions_t0
        
        if box is not None:
            # Apply minimum image convention
            dr -= box * np.round(dr / box)
        
        return dr
    
    def compute_msd(self, trajectory: np.ndarray,
                   tau_values: Optional[np.ndarray] = None,
                   box: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean squared displacement.
        
        Args:
            trajectory: (n_frames, n_atoms, 3) array of positions
            tau_values: Lag times to compute MSD
            box: Box dimensions for PBC
        
        Returns:
            (tau_values, msd_values)
        """
        if tau_values is None:
            tau_values = self.tau_values
        
        n_frames = len(trajectory)
        msd_values = []
        
        for tau in tau_values:
            tau_idx = int(tau)
            if tau_idx >= n_frames:
                msd_values.append(np.nan)
                continue
            
            displacements = []
            for t0 in range(0, n_frames - tau_idx, max(1, tau_idx // 10)):
                dr = self.compute_displacements(
                    trajectory[t0], trajectory[t0 + tau_idx], box
                )
                displacements.extend(np.sum(dr ** 2, axis=1))
            
            msd_values.append(np.mean(displacements))
        
        return tau_values, np.array(msd_values)
    
    def compute_non_gaussian_parameter(self, trajectory: np.ndarray,
                                      tau: float,
                                      box: Optional[np.ndarray] = None) -> float:
        """Compute non-Gaussian parameter α₂(t).
        
        α₂(t) = (3⟨r⁴⟩) / (5⟨r²⟩²) - 1
        
        Returns 0 for Gaussian dynamics, positive for heterogeneous dynamics.
        """
        n_frames = len(trajectory)
        tau_idx = int(tau)
        
        if tau_idx >= n_frames:
            return np.nan
        
        r2_values = []
        r4_values = []
        
        for t0 in range(0, n_frames - tau_idx, max(1, tau_idx // 10)):
            dr = self.compute_displacements(
                trajectory[t0], trajectory[t0 + tau_idx], box
            )
            r2 = np.sum(dr ** 2, axis=1)
            r2_values.extend(r2)
            r4_values.extend(r2 ** 2)
        
        r2_mean = np.mean(r2_values)
        r4_mean = np.mean(r4_values)
        
        if r2_mean ** 2 < 1e-20:
            return 0.0
        
        alpha2 = (3 * r4_mean) / (5 * r2_mean ** 2) - 1
        
        return alpha2
    
    def compute_non_gaussian_parameter_series(self, trajectory: np.ndarray,
                                             tau_values: Optional[np.ndarray] = None,
                                             box: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute α₂(t) for a series of lag times."""
        if tau_values is None:
            tau_values = self.tau_values
        
        alpha2_values = []
        
        for tau in tau_values:
            alpha2 = self.compute_non_gaussian_parameter(trajectory, tau, box)
            alpha2_values.append(alpha2)
        
        return tau_values, np.array(alpha2_values)
    
    def compute_dynamic_susceptibility(self, trajectory: np.ndarray,
                                      q: float = 2.672,  # First peak of LJ structure factor
                                      tau_values: Optional[np.ndarray] = None,
                                      box: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute four-point dynamic susceptibility χ₄(t).
        
        χ₄(t) = N[⟨W²(t)⟩ - ⟨W(t)⟩²]
        where W(t) is the self-overlap function.
        """
        if tau_values is None:
            tau_values = self.tau_values
        
        n_frames = len(trajectory)
        n_atoms = trajectory.shape[1]
        chi4_values = []
        
        # Define overlap function cutoff
        a = 0.3  # Typically 0.3 * average interparticle spacing
        
        for tau in tau_values:
            tau_idx = int(tau)
            if tau_idx >= n_frames:
                chi4_values.append(np.nan)
                continue
            
            w_values = []
            
            for t0 in range(0, n_frames - tau_idx, max(1, tau_idx // 10)):
                dr = self.compute_displacements(
                    trajectory[t0], trajectory[t0 + tau_idx], box
                )
                r = np.sqrt(np.sum(dr ** 2, axis=1))
                
                # Heaviside overlap function
                w = np.mean(r < a)
                w_values.append(w)
            
            w_mean = np.mean(w_values)
            w_var = np.var(w_values)
            
            chi4 = n_atoms * w_var / (w_mean ** 2) if w_mean > 0 else 0
            chi4_values.append(chi4)
        
        return tau_values, np.array(chi4_values)
    
    def compute_self_overlap(self, trajectory: np.ndarray,
                            tau: float,
                            cutoff: float = 0.3,
                            box: Optional[np.ndarray] = None) -> float:
        """Compute self-overlap function Q(t)."""
        n_frames = len(trajectory)
        tau_idx = int(tau)
        
        if tau_idx >= n_frames:
            return np.nan
        
        overlaps = []
        
        for t0 in range(0, n_frames - tau_idx, max(1, tau_idx // 10)):
            dr = self.compute_displacements(
                trajectory[t0], trajectory[t0 + tau_idx], box
            )
            r = np.sqrt(np.sum(dr ** 2, axis=1))
            
            # Average overlap
            overlap = np.mean(r < cutoff)
            overlaps.append(overlap)
        
        return np.mean(overlaps)
    
    def identify_mobile_particles(self, trajectory: np.ndarray,
                                 tau: float,
                                 box: Optional[np.ndarray] = None,
                                 threshold: Optional[float] = None) -> np.ndarray:
        """Identify mobile particles at given lag time.
        
        Returns:
            Boolean array indicating mobile particles
        """
        if threshold is None:
            threshold = self.config.mobile_threshold
        
        n_frames = len(trajectory)
        tau_idx = int(tau)
        
        if tau_idx >= n_frames:
            return np.array([])
        
        # Use last frame pair for identification
        dr = self.compute_displacements(
            trajectory[0], trajectory[tau_idx], box
        )
        r = np.sqrt(np.sum(dr ** 2, axis=1))
        
        # Threshold based on mean displacement
        mean_disp = np.mean(r)
        mobile = r > threshold * mean_disp
        
        return mobile
    
    def cluster_mobile_particles(self, positions: np.ndarray,
                                mobile_mask: np.ndarray,
                                eps: float = 3.0,
                                min_samples: int = 3) -> np.ndarray:
        """Cluster mobile particles using DBSCAN."""
        mobile_positions = positions[mobile_mask]
        
        if len(mobile_positions) < min_samples:
            return np.full(len(positions), -1)
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(mobile_positions)
        
        # Map back to full array
        labels = np.full(len(positions), -1)
        labels[mobile_mask] = clustering.labels_
        
        return labels
    
    def compute_cluster_size_distribution(self, labels: np.ndarray) -> Dict[str, Any]:
        """Analyze cluster size distribution."""
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise
        
        if not unique_labels:
            return {'mean_size': 0, 'max_size': 0, 'sizes': []}
        
        sizes = [np.sum(labels == label) for label in unique_labels]
        
        return {
            'mean_size': np.mean(sizes),
            'max_size': np.max(sizes),
            'std_size': np.std(sizes),
            'n_clusters': len(sizes),
            'sizes': sizes
        }
    
    def compute_dynamic_correlation_length(self, trajectory: np.ndarray,
                                          tau_values: Optional[np.ndarray] = None,
                                          box: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate dynamic correlation length from cluster analysis."""
        if tau_values is None:
            tau_values = self.tau_values
        
        xi_values = []
        
        for tau in tau_values:
            mobile = self.identify_mobile_particles(trajectory, tau, box)
            
            if np.sum(mobile) < self.config.n_clusters_min:
                xi_values.append(np.nan)
                continue
            
            # Use last frame for positions
            positions = trajectory[-1]
            labels = self.cluster_mobile_particles(positions, mobile)
            
            cluster_stats = self.compute_cluster_size_distribution(labels)
            
            # Estimate correlation length from maximum cluster size
            # ξ ~ R_max where 4/3 π R³ ~ cluster_size
            if cluster_stats['max_size'] > 0:
                xi = (3 * cluster_stats['max_size'] / (4 * np.pi)) ** (1/3)
            else:
                xi = 0
            
            xi_values.append(xi)
        
        return tau_values, np.array(xi_values)
    
    def compute_van_hove_function(self, trajectory: np.ndarray,
                                 tau: float,
                                 r_bins: int = 100,
                                 r_max: Optional[float] = None,
                                 box: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute van Hove correlation function G(r, t).
        
        G_s(r, t): Self part (displacements of same particle)
        G_d(r, t): Distinct part (correlations between different particles)
        """
        n_frames = len(trajectory)
        tau_idx = int(tau)
        
        if tau_idx >= n_frames:
            return np.array([]), np.array([])
        
        # Compute displacements
        dr = self.compute_displacements(
            trajectory[0], trajectory[tau_idx], box
        )
        r_displacements = np.sqrt(np.sum(dr ** 2, axis=1))
        
        # Histogram
        if r_max is None:
            r_max = np.percentile(r_displacements, 95) * 2
        
        bins = np.linspace(0, r_max, r_bins)
        hist, bin_edges = np.histogram(r_displacements, bins=bins)
        
        # Normalize
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = bin_edges[1:] - bin_edges[:-1]
        
        # Volume normalization for spherical shells
        volumes = 4 * np.pi * bin_centers ** 2 * bin_widths
        g_s = hist / (volumes * len(dr))
        
        return bin_centers, g_s
    
    def compute_stokes_einstein_ratio(self, msd: np.ndarray,
                                     tau: np.ndarray,
                                     temperature: float,
                                     viscosity: float) -> float:
        """Compute deviation from Stokes-Einstein relation."""
        # D = k_B T / (6 π η r)
        # For a typical particle size, compute expected D
        kB = 8.617e-5  # eV/K
        
        # From MSD = 6Dt, get D
        # Use long-time diffusion
        if len(msd) > 10:
            D_measured = msd[-1] / (6 * tau[-1])
        else:
            D_measured = 0
        
        # Stokes-Einstein prediction (assuming unit particle size)
        D_se = kB * temperature / (6 * np.pi * viscosity)
        
        return D_measured / D_se if D_se > 0 else 0
    
    def analyze_dynamics(self, trajectory: np.ndarray,
                        box: Optional[np.ndarray] = None,
                        temperature: Optional[float] = None) -> Dict:
        """Complete dynamic heterogeneity analysis."""
        results = {}
        
        # MSD
        tau, msd = self.compute_msd(trajectory, box=box)
        results['msd'] = {'tau': tau, 'msd': msd}
        
        # Non-Gaussian parameter
        tau, alpha2 = self.compute_non_gaussian_parameter_series(trajectory, box=box)
        results['non_gaussian'] = {'tau': tau, 'alpha2': alpha2}
        
        # Dynamic susceptibility
        tau, chi4 = self.compute_dynamic_susceptibility(trajectory, box=box)
        results['chi4'] = {'tau': tau, 'chi4': chi4}
        
        # Correlation length
        tau, xi = self.compute_dynamic_correlation_length(trajectory, box=box)
        results['correlation_length'] = {'tau': tau, 'xi': xi}
        
        # Find tau* where χ₄ is maximum (dynamic heterogeneity timescale)
        if len(chi4) > 0 and not np.all(np.isnan(chi4)):
            chi4_max_idx = np.nanargmax(chi4)
            results['tau_star'] = tau[chi4_max_idx]
            results['chi4_max'] = chi4[chi4_max_idx]
        
        # Mobile particle analysis at tau*
        if 'tau_star' in results:
            mobile = self.identify_mobile_particles(
                trajectory, results['tau_star'], box
            )
            positions = trajectory[-1]
            labels = self.cluster_mobile_particles(positions, mobile)
            
            results['mobile_particles'] = {
                'fraction': np.mean(mobile),
                'cluster_stats': self.compute_cluster_size_distribution(labels)
            }
        
        return results
    
    def save_results(self, results: Dict, output_file: str):
        """Save analysis results to file."""
        # Convert numpy arrays to lists for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        
        with open(output_file, 'w') as f:
            json.dump(convert(results), f, indent=2)
        
        logger.info(f"Results saved to {output_file}")


# Export public API
__all__ = [
    'DynamicHeterogeneityConfig',
    'DynamicHeterogeneityAnalyzer'
]
