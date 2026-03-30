#!/usr/bin/env python3
"""
optimized_md_analysis.py
========================
优化的MD分析模块

优化内容:
1. Numba加速的RDF/MSD计算
2. 内存高效的轨迹处理
3. 批量距离计算
4. 并行分析支持
5. JIT编译的关键循环

作者: Performance Optimization Expert
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import logging

# 尝试导入Numba
try:
    from numba import jit, njit, prange
    from numba.typed import List as NumbaList
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    jit = njit
    prange = range

logger = logging.getLogger(__name__)


# =============================================================================
# Numba加速的核心计算函数
# =============================================================================

@njit(cache=True, parallel=True)
def calculate_rdf_numba(positions, box, r_max, n_bins, type_i, type_j, atom_types):
    """
    使用Numba并行计算RDF
    
    Args:
        positions: 原子位置 (N, 3)
        box: 盒子尺寸 (3,)
        r_max: 最大距离
        n_bins: bin数量
        type_i, type_j: 原子类型筛选
        atom_types: 每个原子的类型
    
    Returns:
        hist: RDF直方图
    """
    n_atoms = len(positions)
    hist = np.zeros(n_bins, dtype=np.int64)
    bin_width = r_max / n_bins
    
    # 使用并行循环
    for i in prange(n_atoms):
        if atom_types[i] != type_i and type_i >= 0:
            continue
        
        for j in range(i+1, n_atoms):
            if atom_types[j] != type_j and type_j >= 0:
                continue
            
            # 计算距离（考虑周期性边界条件）
            dist_sq = 0.0
            for k in range(3):
                diff = positions[i, k] - positions[j, k]
                # PBC处理
                if abs(diff) > box[k] * 0.5:
                    diff -= box[k] * np.sign(diff)
                dist_sq += diff * diff
            
            dist = np.sqrt(dist_sq)
            
            if dist < r_max:
                bin_idx = int(dist / bin_width)
                if bin_idx < n_bins:
                    hist[bin_idx] += 1
    
    return hist


@njit(cache=True, parallel=True)
def calculate_msd_numba(trajectory, dt_max, n_selected):
    """
    计算均方位移 (MSD)
    
    Args:
        trajectory: 轨迹 (n_frames, n_atoms, 3)
        dt_max: 最大时间间隔
        n_selected: 选择的原子数
    
    Returns:
        msd: MSD数组
    """
    n_frames = len(trajectory)
    msd = np.zeros(min(dt_max, n_frames))
    counts = np.zeros(min(dt_max, n_frames))
    
    max_dt = min(dt_max, n_frames - 1)
    
    for dt in prange(1, max_dt + 1):
        sq_disp_sum = 0.0
        count = 0
        
        for t0 in range(n_frames - dt):
            t1 = t0 + dt
            
            for i in range(n_selected):
                disp_sq = 0.0
                for k in range(3):
                    diff = trajectory[t1, i, k] - trajectory[t0, i, k]
                    disp_sq += diff * diff
                
                sq_disp_sum += disp_sq
                count += 1
        
        if count > 0:
            msd[dt] = sq_disp_sum / count
            counts[dt] = count
    
    return msd


@njit(cache=True, parallel=True)
def calculate_distance_matrix_numba(positions, box):
    """
    计算距离矩阵（考虑PBC）
    """
    n = len(positions)
    dist_matrix = np.zeros((n, n))
    
    for i in prange(n):
        for j in range(i+1, n):
            dist_sq = 0.0
            for k in range(3):
                diff = positions[i, k] - positions[j, k]
                # PBC
                half_box = box[k] * 0.5
                if diff > half_box:
                    diff -= box[k]
                elif diff < -half_box:
                    diff += box[k]
                dist_sq += diff * diff
            
            dist = np.sqrt(dist_sq)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    return dist_matrix


@njit(cache=True)
def apply_pbc_drifts_numba(positions, box):
    """
    应用周期性边界条件修正漂移
    """
    corrected = positions.copy()
    n_frames, n_atoms, _ = positions.shape
    
    for t in range(1, n_frames):
        for i in range(n_atoms):
            for k in range(3):
                drift = corrected[t, i, k] - corrected[t-1, i, k]
                if drift > box[k] * 0.5:
                    corrected[t, i, k] -= box[k]
                elif drift < -box[k] * 0.5:
                    corrected[t, i, k] += box[k]
    
    return corrected


@njit(cache=True, parallel=True)
def calculate_structure_factor_numba(positions, box, q_points):
    """
    计算结构因子
    """
    n_atoms = len(positions)
    n_q = len(q_points)
    sq = np.zeros(n_q)
    
    for q_idx in prange(n_q):
        q = q_points[q_idx]
        
        real_sum = 0.0
        imag_sum = 0.0
        
        for i in range(n_atoms):
            phase = 0.0
            for k in range(3):
                phase += q[k] * positions[i, k]
            
            real_sum += np.cos(phase)
            imag_sum += np.sin(phase)
        
        sq[q_idx] = (real_sum**2 + imag_sum**2) / n_atoms
    
    return sq


# =============================================================================
# 优化的分析类
# =============================================================================

@dataclass
class AnalysisConfig:
    """分析配置"""
    use_numba: bool = True
    n_workers: int = 4
    chunk_size: int = 100
    memory_limit_gb: float = 4.0


class OptimizedTrajectoryAnalyzer:
    """优化的轨迹分析器"""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.trajectory = None
        self.box = None
        self.atom_types = None
    
    def load_trajectory(self, trajectory_file: str, 
                       format: str = 'lammps-dump') -> 'OptimizedTrajectoryAnalyzer':
        """
        加载轨迹文件
        """
        logger.info(f"Loading trajectory: {trajectory_file}")
        
        if format == 'lammps-dump':
            self.trajectory, self.box, self.atom_types = self._load_lammps_dump(trajectory_file)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Loaded {len(self.trajectory)} frames, {len(self.trajectory[0])} atoms")
        return self
    
    def _load_lammps_dump(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """加载LAMMPS dump文件"""
        frames = []
        box = np.array([10.0, 10.0, 10.0])  # 默认值
        atom_types_list = []
        
        with open(filepath, 'r') as f:
            frame_data = []
            for line in f:
                if "ITEM: TIMESTEP" in line:
                    if frame_data:
                        positions, types = self._parse_lammps_frame(frame_data)
                        frames.append(positions)
                        if len(atom_types_list) == 0:
                            atom_types_list = types
                        frame_data = []
                frame_data.append(line)
            
            # 最后一帧
            if frame_data:
                positions, types = self._parse_lammps_frame(frame_data)
                frames.append(positions)
                if len(atom_types_list) == 0:
                    atom_types_list = types
        
        trajectory = np.array(frames)
        atom_types = np.array(atom_types_list)
        
        return trajectory, box, atom_types
    
    def _parse_lammps_frame(self, lines: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """解析LAMMPS帧"""
        positions = []
        types = []
        
        in_atoms = False
        for line in lines:
            if line.startswith("ITEM: ATOMS"):
                in_atoms = True
                continue
            
            if in_atoms:
                parts = line.split()
                if len(parts) >= 5:
                    types.append(int(parts[1]))
                    positions.append([float(parts[2]), float(parts[3]), float(parts[4])])
        
        return np.array(positions), np.array(types)
    
    def compute_rdf(self, 
                   r_max: float = 10.0, 
                   n_bins: int = 100,
                   type_i: int = -1, 
                   type_j: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算径向分布函数 (RDF)
        """
        if self.trajectory is None:
            raise ValueError("No trajectory loaded")
        
        logger.info(f"Computing RDF: r_max={r_max}, n_bins={n_bins}")
        
        # 使用第一帧的位置（或平均）
        positions = self.trajectory[0]
        
        if self.config.use_numba and NUMBA_AVAILABLE:
            hist = calculate_rdf_numba(
                positions, self.box, r_max, n_bins,
                type_i, type_j, self.atom_types
            )
        else:
            hist = self._compute_rdf_python(positions, r_max, n_bins, type_i, type_j)
        
        # 归一化
        bin_edges = np.linspace(0, r_max, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # 体积归一化
        volume = np.prod(self.box)
        n_atoms = len(positions)
        density = n_atoms / volume
        
        shell_volumes = 4/3 * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
        normalization = density * shell_volumes * n_atoms
        
        g_r = hist / normalization
        
        return bin_centers, g_r
    
    def _compute_rdf_python(self, positions, r_max, n_bins, type_i, type_j):
        """Python版本的RDF计算"""
        n_atoms = len(positions)
        hist = np.zeros(n_bins)
        bin_width = r_max / n_bins
        
        for i in range(n_atoms):
            if type_i >= 0 and self.atom_types[i] != type_i:
                continue
            
            for j in range(i+1, n_atoms):
                if type_j >= 0 and self.atom_types[j] != type_j:
                    continue
                
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < r_max:
                    bin_idx = int(dist / bin_width)
                    if bin_idx < n_bins:
                        hist[bin_idx] += 1
        
        return hist
    
    def compute_msd(self, 
                   atom_indices: Optional[List[int]] = None,
                   dt_max: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算均方位移 (MSD)
        """
        if self.trajectory is None:
            raise ValueError("No trajectory loaded")
        
        n_frames, n_atoms, _ = self.trajectory.shape
        
        # 选择原子
        if atom_indices is None:
            # 选择类型1的原子或前20个
            atom_indices = list(range(min(20, n_atoms)))
        
        selected_trajectory = self.trajectory[:, atom_indices, :]
        
        # 应用PBC漂移修正
        if self.config.use_numba and NUMBA_AVAILABLE:
            corrected_traj = apply_pbc_drifts_numba(selected_trajectory, self.box)
        else:
            corrected_traj = selected_trajectory
        
        dt_max = dt_max or n_frames // 2
        n_selected = len(atom_indices)
        
        if self.config.use_numba and NUMBA_AVAILABLE:
            msd = calculate_msd_numba(corrected_traj, dt_max, n_selected)
        else:
            msd = self._compute_msd_python(corrected_traj, dt_max, n_selected)
        
        dt = np.arange(len(msd))
        
        return dt, msd
    
    def _compute_msd_python(self, trajectory, dt_max, n_selected):
        """Python版本的MSD计算"""
        n_frames = len(trajectory)
        msd = np.zeros(min(dt_max, n_frames))
        
        for dt in range(1, min(dt_max, n_frames)):
            sq_disp_sum = 0.0
            count = 0
            
            for t0 in range(n_frames - dt):
                t1 = t0 + dt
                
                for i in range(n_selected):
                    disp = trajectory[t1, i] - trajectory[t0, i]
                    sq_disp_sum += np.sum(disp**2)
                    count += 1
            
            if count > 0:
                msd[dt] = sq_disp_sum / count
        
        return msd
    
    def compute_diffusion_coefficient(self, 
                                     dt: np.ndarray, 
                                     msd: np.ndarray,
                                     fit_start: float = 0.2,
                                     fit_end: float = 0.8) -> float:
        """
        从MSD计算扩散系数
        """
        # 使用线性拟合
        n_points = len(msd)
        start_idx = int(n_points * fit_start)
        end_idx = int(n_points * fit_end)
        
        if end_idx <= start_idx:
            return 0.0
        
        # 线性拟合
        slope, intercept = np.polyfit(dt[start_idx:end_idx], msd[start_idx:end_idx], 1)
        
        # D = slope / (2 * dimension)
        D = slope / 6.0  # 3D
        
        return D
    
    def compute_vanhove(self, 
                       time_lag: int = 100,
                       r_max: float = 10.0,
                       n_bins: int = 100) -> np.ndarray:
        """
        计算Van Hove关联函数
        """
        if self.trajectory is None:
            raise ValueError("No trajectory loaded")
        
        n_frames, n_atoms, _ = self.trajectory.shape
        
        if time_lag >= n_frames:
            time_lag = n_frames - 1
        
        # 计算位移
        displacements = self.trajectory[time_lag:] - self.trajectory[:-time_lag]
        
        # 直方图
        dists = np.sqrt(np.sum(displacements**2, axis=2)).flatten()
        
        hist, bin_edges = np.histogram(dists, bins=n_bins, range=(0, r_max))
        
        return hist


# =============================================================================
# 批量分析器
# =============================================================================

class BatchTrajectoryAnalyzer:
    """批量轨迹分析器"""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
    
    def analyze_batch(self, 
                     trajectory_files: List[str],
                     analysis_funcs: Dict[str, Callable]) -> Dict[str, List]:
        """
        批量分析多个轨迹文件
        """
        from concurrent.futures import ProcessPoolExecutor
        
        results = defaultdict(list)
        
        with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
            future_to_file = {
                executor.submit(self._analyze_single, f, analysis_funcs): f
                for f in trajectory_files
            }
            
            for future in future_to_file:
                filepath = future_to_file[future]
                try:
                    result = future.result()
                    for key, value in result.items():
                        results[key].append(value)
                except Exception as e:
                    logger.error(f"Failed to analyze {filepath}: {e}")
        
        return dict(results)
    
    def _analyze_single(self, filepath: str, analysis_funcs: Dict) -> Dict:
        """分析单个轨迹"""
        analyzer = OptimizedTrajectoryAnalyzer(self.config)
        analyzer.load_trajectory(filepath)
        
        results = {}
        for name, func in analysis_funcs.items():
            try:
                results[name] = func(analyzer)
            except Exception as e:
                logger.error(f"Analysis {name} failed: {e}")
                results[name] = None
        
        return results


# =============================================================================
# 性能比较工具
# =============================================================================

def benchmark_analysis(trajectory_file: str) -> Dict:
    """比较优化前后的分析性能"""
    import time
    
    results = {}
    
    # 使用Numba
    logger.info("Testing with Numba...")
    analyzer_numba = OptimizedTrajectoryAnalyzer(AnalysisConfig(use_numba=True))
    analyzer_numba.load_trajectory(trajectory_file)
    
    start = time.time()
    r_numba, g_numba = analyzer_numba.compute_rdf()
    rdf_time_numba = time.time() - start
    
    start = time.time()
    dt, msd = analyzer_numba.compute_msd()
    msd_time_numba = time.time() - start
    
    results['with_numba'] = {
        'rdf_time': rdf_time_numba,
        'msd_time': msd_time_numba
    }
    
    # 不使用Numba
    logger.info("Testing without Numba...")
    analyzer_python = OptimizedTrajectoryAnalyzer(AnalysisConfig(use_numba=False))
    analyzer_python.load_trajectory(trajectory_file)
    
    start = time.time()
    r_python, g_python = analyzer_python.compute_rdf()
    rdf_time_python = time.time() - start
    
    start = time.time()
    dt, msd = analyzer_python.compute_msd()
    msd_time_python = time.time() - start
    
    results['without_numba'] = {
        'rdf_time': rdf_time_python,
        'msd_time': msd_time_python
    }
    
    # 计算加速比
    if rdf_time_python > 0:
        results['rdf_speedup'] = rdf_time_python / rdf_time_numba
    if msd_time_python > 0:
        results['msd_speedup'] = msd_time_python / msd_time_numba
    
    logger.info(f"RDF speedup: {results.get('rdf_speedup', 0):.2f}x")
    logger.info(f"MSD speedup: {results.get('msd_speedup', 0):.2f}x")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Optimized MD Analysis module loaded")
    print(f"Numba available: {NUMBA_AVAILABLE}")
