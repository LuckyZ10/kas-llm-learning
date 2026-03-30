"""
Advanced utilities for multiscale simulations.
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class SimulationData:
    """Container for simulation data at a single scale."""
    positions: np.ndarray
    velocities: Optional[np.ndarray] = None
    forces: Optional[np.ndarray] = None
    energies: Optional[np.ndarray] = None
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    time: Optional[np.ndarray] = None
    
    @property
    def n_frames(self) -> int:
        return len(self.positions)
    
    @property
    def n_atoms(self) -> int:
        return self.positions.shape[1]


class TrajectoryInterpolator:
    """Interpolate between different time resolutions."""
    
    def __init__(self, method: str = 'cubic'):
        self.method = method
    
    def interpolate(self,
                   times_low: np.ndarray,
                   data_low: np.ndarray,
                   times_high: np.ndarray) -> np.ndarray:
        """
        Interpolate from low to high time resolution.
        
        Args:
            times_low: Low-resolution time points
            data_low: Low-resolution data
            times_high: High-resolution time points
            
        Returns:
            Interpolated data at high resolution
        """
        from scipy import interpolate
        
        if self.method == 'linear':
            kind = 'linear'
        elif self.method == 'cubic':
            kind = 'cubic'
        else:
            kind = 'linear'
        
        # Handle multi-dimensional data
        if data_low.ndim == 3:  # (n_frames, n_atoms, 3)
            n_atoms = data_low.shape[1]
            result = np.zeros((len(times_high), n_atoms, 3))
            for i in range(n_atoms):
                for j in range(3):
                    f = interpolate.interp1d(
                        times_low, data_low[:, i, j],
                        kind=kind, fill_value='extrapolate'
                    )
                    result[:, i, j] = f(times_high)
            return result
        else:
            f = interpolate.interp1d(
                times_low, data_low,
                kind=kind, fill_value='extrapolate'
            )
            return f(times_high)


class AdaptiveResolution:
    """
    Adaptive resolution scheme for multiscale simulations.
    Dynamically switches between resolutions based on local environment.
    """
    
    def __init__(self,
                 qm_radius: float = 5.0,
                 buffer_radius: float = 2.0,
                 switch_function: str = 'cosine'):
        """
        Initialize adaptive resolution.
        
        Args:
            qm_radius: Inner QM region radius
            buffer_radius: Buffer zone thickness
            switch_function: Switching function type
        """
        self.qm_radius = qm_radius
        self.buffer_radius = buffer_radius
        self.switch_function = switch_function
        
    def calculate_weights(self,
                         positions: np.ndarray,
                         qm_center: np.ndarray) -> np.ndarray:
        """
        Calculate switching weights for each atom.
        
        Args:
            positions: Atomic positions
            qm_center: Center of QM region
            
        Returns:
            Weight array (0=MM, 1=QM, intermediate in buffer)
        """
        distances = np.linalg.norm(positions - qm_center, axis=1)
        
        weights = np.zeros(len(positions))
        
        for i, d in enumerate(distances):
            if d < self.qm_radius:
                weights[i] = 1.0
            elif d < self.qm_radius + self.buffer_radius:
                # Buffer zone - interpolate
                s = (d - self.qm_radius) / self.buffer_radius
                if self.switch_function == 'cosine':
                    weights[i] = 0.5 * (1 + np.cos(np.pi * s))
                elif self.switch_function == 'linear':
                    weights[i] = 1 - s
                else:
                    weights[i] = 1 - s
            else:
                weights[i] = 0.0
        
        return weights


class ReactionCoordinateMonitor:
    """Monitor reaction coordinates during simulation."""
    
    def __init__(self):
        self.coordinates = {}
    
    def add_distance(self, name: str, atom_i: int, atom_j: int):
        """Add distance coordinate."""
        self.coordinates[name] = ('distance', atom_i, atom_j)
    
    def add_angle(self, name: str, atom_i: int, atom_j: int, atom_k: int):
        """Add angle coordinate."""
        self.coordinates[name] = ('angle', atom_i, atom_j, atom_k)
    
    def add_dihedral(self, name: str, 
                    atom_i: int, atom_j: int, atom_k: int, atom_l: int):
        """Add dihedral coordinate."""
        self.coordinates[name] = ('dihedral', atom_i, atom_j, atom_k, atom_l)
    
    def compute(self, positions: np.ndarray) -> Dict[str, float]:
        """
        Compute all reaction coordinates.
        
        Args:
            positions: Atomic positions
            
        Returns:
            Dictionary of coordinate values
        """
        results = {}
        
        for name, coord_def in self.coordinates.items():
            coord_type = coord_def[0]
            
            if coord_type == 'distance':
                _, i, j = coord_def
                r = positions[j] - positions[i]
                results[name] = np.linalg.norm(r)
                
            elif coord_type == 'angle':
                _, i, j, k = coord_def
                r1 = positions[i] - positions[j]
                r2 = positions[k] - positions[j]
                cos_angle = np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
                results[name] = np.arccos(np.clip(cos_angle, -1, 1))
                
            elif coord_type == 'dihedral':
                _, i, j, k, l = coord_def
                # Simplified dihedral calculation
                b1 = positions[j] - positions[i]
                b2 = positions[k] - positions[j]
                b3 = positions[l] - positions[k]
                
                n1 = np.cross(b1, b2)
                n2 = np.cross(b2, b3)
                
                m1 = np.cross(n1, b2 / np.linalg.norm(b2))
                
                x = np.dot(n1, n2)
                y = np.dot(m1, n2)
                
                results[name] = np.arctan2(y, x)
        
        return results


class DataExporter:
    """Export simulation data to various formats."""
    
    @staticmethod
    def to_xyz(positions: np.ndarray,
              elements: List[str],
              filename: str,
              comment: str = ""):
        """Export to XYZ format."""
        with open(filename, 'w') as f:
            for frame in positions:
                f.write(f"{len(elements)}\n")
                f.write(f"{comment}\n")
                for elem, pos in zip(elements, frame):
                    f.write(f"{elem:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")
    
    @staticmethod
    def to_lammps_dump(positions: np.ndarray,
                      filename: str,
                      timestep: int = 0):
        """Export to LAMMPS dump format."""
        with open(filename, 'w') as f:
            for i_frame, frame in enumerate(positions):
                f.write("ITEM: TIMESTEP\n")
                f.write(f"{timestep + i_frame}\n")
                f.write("ITEM: NUMBER OF ATOMS\n")
                f.write(f"{len(frame)}\n")
                f.write("ITEM: BOX BOUNDS pp pp pp\n")
                f.write("-50.0 50.0\n")
                f.write("-50.0 50.0\n")
                f.write("-50.0 50.0\n")
                f.write("ITEM: ATOMS id type x y z\n")
                for i, pos in enumerate(frame):
                    f.write(f"{i+1} 1 {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
    
    @staticmethod
    def to_numpy_archive(data: Dict, filename: str):
        """Export to NumPy archive."""
        np.savez(filename, **data)


class PerformanceProfiler:
    """Profile performance of multiscale simulations."""
    
    def __init__(self):
        self.timings = {}
        self.memory = {}
    
    def record_time(self, label: str, time_seconds: float):
        """Record timing."""
        if label not in self.timings:
            self.timings[label] = []
        self.timings[label].append(time_seconds)
    
    def record_memory(self, label: str, memory_mb: float):
        """Record memory usage.""""
        if label not in self.memory:
            self.memory[label] = []
        self.memory[label].append(memory_mb)
    
    def get_summary(self) -> Dict:
        """Get performance summary."""
        summary = {}
        
        for label, times in self.timings.items():
            summary[label] = {
                'total': sum(times),
                'mean': np.mean(times),
                'std': np.std(times),
                'min': min(times),
                'max': max(times)
            }
        
        return summary
    
    def print_report(self):
        """Print performance report."""
        print("=" * 60)
        print("Performance Profile")
        print("=" * 60)
        
        for label, stats in self.get_summary().items():
            print(f"\n{label}:")
            print(f"  Total: {stats['total']:.3f} s")
            print(f"  Mean:  {stats['mean']:.3f} s")
            print(f"  Std:   {stats['std']:.3f} s")
            print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}] s")


def compute_rdf(positions: np.ndarray,
               box_size: float,
               n_bins: int = 100,
               r_max: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute radial distribution function.
    
    Args:
        positions: Atomic positions (n_frames, n_atoms, 3)
        box_size: Simulation box size
        n_bins: Number of bins
        r_max: Maximum distance
        
    Returns:
        r_bins, g(r)
    """
    if r_max is None:
        r_max = box_size / 2
    
    dr = r_max / n_bins
    r_bins = np.linspace(dr/2, r_max - dr/2, n_bins)
    
    g_r = np.zeros(n_bins)
    n_frames = len(positions)
    n_atoms = len(positions[0])
    
    for frame in positions:
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r_ij = frame[j] - frame[i]
                # Apply PBC
                r_ij -= box_size * np.round(r_ij / box_size)
                r = np.linalg.norm(r_ij)
                
                if r < r_max:
                    bin_idx = int(r / dr)
                    if bin_idx < n_bins:
                        g_r[bin_idx] += 2  # Count both i-j and j-i
    
    # Normalize
    rho = n_atoms / (box_size ** 3)
    for i, r in enumerate(r_bins):
        shell_volume = 4 * np.pi * r**2 * dr
        g_r[i] /= n_frames * n_atoms * shell_volume * rho
    
    return r_bins, g_r


def compute_msd(positions: np.ndarray,
               timestep: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean squared displacement.
    
    Args:
        positions: Atomic positions (n_frames, n_atoms, 3)
        timestep: Time between frames
        
    Returns:
        times, MSD(t)
    """
    n_frames = len(positions)
    max_lag = n_frames // 2
    
    times = np.arange(max_lag) * timestep
    msd = np.zeros(max_lag)
    
    for lag in range(1, max_lag):
        displacements = positions[lag:] - positions[:-lag]
        msd[lag] = np.mean(displacements ** 2)
    
    return times, msd
