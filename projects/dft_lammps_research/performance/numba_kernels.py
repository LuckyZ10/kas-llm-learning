#!/usr/bin/env python3
"""
numba_kernels.py
================
Numba加速核模块

提供高性能计算核心：
- 距离矩阵计算
- RDF/MSD计算
- 邻居列表构建
- 能量/力计算

作者: Performance Optimization Expert
日期: 2026-03-22
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Numba导入处理
try:
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
except Exception as e:
    NUMBA_AVAILABLE = False
    logger.warning(f"Numba import failed: {e}. Kernels will run in pure Python mode.")
    
    # 创建虚拟装饰器
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    jit = njit
    prange = range


# =============================================================================
# 距离计算
# =============================================================================

def calculate_distance_matrix(positions: np.ndarray, 
                              box: np.ndarray,
                              cutoff: Optional[float] = None) -> np.ndarray:
    """
    计算距离矩阵
    
    Args:
        positions: 原子位置 (N, 3)
        box: 盒子尺寸 (3,)
        cutoff: 截断距离（可选）
    
    Returns:
        距离矩阵 (N, N)
    """
    return _calculate_distance_matrix_impl(positions, box, cutoff)


def _calculate_distance_matrix_impl(positions: np.ndarray, 
                                    box: np.ndarray,
                                    cutoff: Optional[float] = None) -> np.ndarray:
    """距离矩阵实现"""
    n_atoms = positions.shape[0]
    dist_matrix = np.zeros((n_atoms, n_atoms), dtype=np.float64)
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist_sq = 0.0
            for k in range(3):
                diff = positions[i, k] - positions[j, k]
                # PBC处理
                if box[k] > 0:
                    if diff > box[k] * 0.5:
                        diff -= box[k]
                    elif diff < -box[k] * 0.5:
                        diff += box[k]
                dist_sq += diff * diff
            
            dist = np.sqrt(dist_sq)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    return dist_matrix


if NUMBA_AVAILABLE:
    @njit(cache=True, parallel=True)
    def _calculate_distance_matrix_numba(positions: np.ndarray, 
                                          box: np.ndarray,
                                          cutoff: Optional[float] = None) -> np.ndarray:
        """Numba加速版本"""
        n_atoms = positions.shape[0]
        dist_matrix = np.zeros((n_atoms, n_atoms), dtype=np.float64)
        
        for i in prange(n_atoms):
            for j in range(i + 1, n_atoms):
                dist_sq = 0.0
                for k in range(3):
                    diff = positions[i, k] - positions[j, k]
                    if box[k] > 0:
                        if diff > box[k] * 0.5:
                            diff -= box[k]
                        elif diff < -box[k] * 0.5:
                            diff += box[k]
                    dist_sq += diff * diff
                
                dist = np.sqrt(dist_sq)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        return dist_matrix
    
    # 使用Numba版本
    calculate_distance_matrix = _calculate_distance_matrix_numba


# =============================================================================
# RDF计算
# =============================================================================

def calculate_rdf_parallel(positions: np.ndarray,
                           box: np.ndarray,
                           r_max: float,
                           n_bins: int,
                           atom_types: Optional[np.ndarray] = None,
                           type_i: int = -1,
                           type_j: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算径向分布函数（RDF）
    
    Args:
        positions: 原子位置 (N, 3)
        box: 盒子尺寸 (3,)
        r_max: 最大距离
        n_bins: bin数量
        atom_types: 原子类型数组 (N,)
        type_i, type_j: 要计算的原子类型（-1表示所有）
    
    Returns:
        r_bins, g(r)
    """
    return _calculate_rdf_impl(positions, box, r_max, n_bins, atom_types, type_i, type_j)


def _calculate_rdf_impl(positions: np.ndarray,
                        box: np.ndarray,
                        r_max: float,
                        n_bins: int,
                        atom_types: Optional[np.ndarray] = None,
                        type_i: int = -1,
                        type_j: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """RDF实现"""
    n_atoms = positions.shape[0]
    hist = np.zeros(n_bins, dtype=np.int64)
    bin_width = r_max / n_bins
    
    for i in range(n_atoms):
        if atom_types is not None and type_i >= 0:
            if atom_types[i] != type_i:
                continue
        
        for j in range(n_atoms):
            if i == j:
                continue
            
            if atom_types is not None and type_j >= 0:
                if atom_types[j] != type_j:
                    continue
            
            dist_sq = 0.0
            for k in range(3):
                diff = positions[i, k] - positions[j, k]
                if box[k] > 0:
                    if diff > box[k] * 0.5:
                        diff -= box[k]
                    elif diff < -box[k] * 0.5:
                        diff += box[k]
                dist_sq += diff * diff
            
            dist = np.sqrt(dist_sq)
            
            if dist < r_max:
                bin_idx = int(dist / bin_width)
                if bin_idx < n_bins:
                    hist[bin_idx] += 1
    
    # 计算r值
    r_bins = np.arange(n_bins) * bin_width + bin_width * 0.5
    
    # 归一化
    volume = box[0] * box[1] * box[2]
    density = n_atoms / volume
    
    for i in range(n_bins):
        r_inner = i * bin_width
        r_outer = (i + 1) * bin_width
        shell_vol = (4.0 / 3.0) * np.pi * (r_outer**3 - r_inner**3)
        n_ideal = shell_vol * density
        if n_ideal > 0:
            hist[i] = hist[i] / (n_ideal * n_atoms)
    
    return r_bins, hist


if NUMBA_AVAILABLE:
    @njit(cache=True, parallel=True)
    def _calculate_rdf_numba(positions: np.ndarray,
                             box: np.ndarray,
                             r_max: float,
                             n_bins: int,
                             atom_types: Optional[np.ndarray] = None,
                             type_i: int = -1,
                             type_j: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        """Numba加速版本"""
        n_atoms = positions.shape[0]
        hist = np.zeros(n_bins, dtype=np.int64)
        bin_width = r_max / n_bins
        
        local_hists = np.zeros((n_atoms, n_bins), dtype=np.int64)
        
        for i in prange(n_atoms):
            if atom_types is not None and type_i >= 0:
                if atom_types[i] != type_i:
                    continue
            
            for j in range(n_atoms):
                if i == j:
                    continue
                
                if atom_types is not None and type_j >= 0:
                    if atom_types[j] != type_j:
                        continue
                
                dist_sq = 0.0
                for k in range(3):
                    diff = positions[i, k] - positions[j, k]
                    if box[k] > 0:
                        if diff > box[k] * 0.5:
                            diff -= box[k]
                        elif diff < -box[k] * 0.5:
                            diff += box[k]
                    dist_sq += diff * diff
                
                dist = np.sqrt(dist_sq)
                
                if dist < r_max:
                    bin_idx = int(dist / bin_width)
                    if bin_idx < n_bins:
                        local_hists[i, bin_idx] += 1
        
        for i in range(n_atoms):
            for b in range(n_bins):
                hist[b] += local_hists[i, b]
        
        r_bins = np.arange(n_bins) * bin_width + bin_width * 0.5
        
        volume = box[0] * box[1] * box[2]
        density = n_atoms / volume
        
        for i in range(n_bins):
            r_inner = i * bin_width
            r_outer = (i + 1) * bin_width
            shell_vol = (4.0 / 3.0) * np.pi * (r_outer**3 - r_inner**3)
            n_ideal = shell_vol * density
            if n_ideal > 0:
                hist[i] = hist[i] / (n_ideal * n_atoms)
        
        return r_bins, hist
    
    calculate_rdf_parallel = _calculate_rdf_numba


# =============================================================================
# MSD计算
# =============================================================================

def calculate_msd_parallel(trajectory: np.ndarray,
                           box: np.ndarray,
                           max_lag: Optional[int] = None) -> np.ndarray:
    """
    计算均方位移（MSD）
    
    Args:
        trajectory: 轨迹数据 (n_frames, n_atoms, 3)
        box: 盒子尺寸 (3,)
        max_lag: 最大滞后时间（帧数）
    
    Returns:
        MSD数组 (max_lag,)
    """
    return _calculate_msd_impl(trajectory, box, max_lag)


def _calculate_msd_impl(trajectory: np.ndarray,
                        box: np.ndarray,
                        max_lag: Optional[int] = None) -> np.ndarray:
    """MSD实现"""
    n_frames = trajectory.shape[0]
    n_atoms = trajectory.shape[1]
    
    if max_lag is None:
        max_lag = n_frames // 2
    
    msd = np.zeros(max_lag, dtype=np.float64)
    counts = np.zeros(max_lag, dtype=np.int64)
    
    for lag in range(1, max_lag):
        for t in range(n_frames - lag):
            for i in range(n_atoms):
                dist_sq = 0.0
                for k in range(3):
                    diff = trajectory[t + lag, i, k] - trajectory[t, i, k]
                    if box[k] > 0:
                        while diff > box[k] * 0.5:
                            diff -= box[k]
                        while diff < -box[k] * 0.5:
                            diff += box[k]
                    dist_sq += diff * diff
                
                msd[lag] += dist_sq
                counts[lag] += 1
        
        if counts[lag] > 0:
            msd[lag] /= counts[lag]
    
    return msd


if NUMBA_AVAILABLE:
    @njit(cache=True, parallel=True)
    def _calculate_msd_numba(trajectory: np.ndarray,
                             box: np.ndarray,
                             max_lag: Optional[int] = None) -> np.ndarray:
        """Numba加速版本"""
        n_frames = trajectory.shape[0]
        n_atoms = trajectory.shape[1]
        
        if max_lag is None:
            max_lag = n_frames // 2
        
        msd = np.zeros(max_lag, dtype=np.float64)
        counts = np.zeros(max_lag, dtype=np.int64)
        
        for lag in prange(1, max_lag):
            for t in range(n_frames - lag):
                for i in range(n_atoms):
                    dist_sq = 0.0
                    for k in range(3):
                        diff = trajectory[t + lag, i, k] - trajectory[t, i, k]
                        if box[k] > 0:
                            while diff > box[k] * 0.5:
                                diff -= box[k]
                            while diff < -box[k] * 0.5:
                                diff += box[k]
                        dist_sq += diff * diff
                    
                    msd[lag] += dist_sq
                    counts[lag] += 1
            
            if counts[lag] > 0:
                msd[lag] /= counts[lag]
        
        return msd
    
    calculate_msd_parallel = _calculate_msd_numba


# =============================================================================
# 邻居列表
# =============================================================================

def build_neighbor_list(positions: np.ndarray,
                        box: np.ndarray,
                        cutoff: float,
                        max_neighbors: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建邻居列表
    
    Args:
        positions: 原子位置 (N, 3)
        box: 盒子尺寸 (3,)
        cutoff: 截断距离
        max_neighbors: 最大邻居数
    
    Returns:
        neighbors数组 (N, max_neighbors), 邻居数数组 (N,)
    """
    return _build_neighbor_list_impl(positions, box, cutoff, max_neighbors)


def _build_neighbor_list_impl(positions: np.ndarray,
                              box: np.ndarray,
                              cutoff: float,
                              max_neighbors: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """邻居列表实现"""
    n_atoms = positions.shape[0]
    neighbors = np.full((n_atoms, max_neighbors), -1, dtype=np.int32)
    n_neighbors = np.zeros(n_atoms, dtype=np.int32)
    
    cutoff_sq = cutoff ** 2
    
    for i in range(n_atoms):
        count = 0
        for j in range(n_atoms):
            if i == j:
                continue
            
            dist_sq = 0.0
            for k in range(3):
                diff = positions[i, k] - positions[j, k]
                if box[k] > 0:
                    if diff > box[k] * 0.5:
                        diff -= box[k]
                    elif diff < -box[k] * 0.5:
                        diff += box[k]
                dist_sq += diff * diff
            
            if dist_sq < cutoff_sq and count < max_neighbors:
                neighbors[i, count] = j
                count += 1
        
        n_neighbors[i] = count
    
    return neighbors, n_neighbors


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _build_neighbor_list_numba(positions: np.ndarray,
                                   box: np.ndarray,
                                   cutoff: float,
                                   max_neighbors: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Numba加速版本"""
        n_atoms = positions.shape[0]
        neighbors = np.full((n_atoms, max_neighbors), -1, dtype=np.int32)
        n_neighbors = np.zeros(n_atoms, dtype=np.int32)
        
        cutoff_sq = cutoff ** 2
        
        for i in range(n_atoms):
            count = 0
            for j in range(n_atoms):
                if i == j:
                    continue
                
                dist_sq = 0.0
                for k in range(3):
                    diff = positions[i, k] - positions[j, k]
                    if box[k] > 0:
                        if diff > box[k] * 0.5:
                            diff -= box[k]
                        elif diff < -box[k] * 0.5:
                            diff += box[k]
                    dist_sq += diff * diff
                
                if dist_sq < cutoff_sq and count < max_neighbors:
                    neighbors[i, count] = j
                    count += 1
            
            n_neighbors[i] = count
        
        return neighbors, n_neighbors
    
    build_neighbor_list = _build_neighbor_list_numba


# =============================================================================
# 能量和力计算
# =============================================================================

def calculate_lennard_jones_energy(positions: np.ndarray,
                                   box: np.ndarray,
                                   sigma: float = 1.0,
                                   epsilon: float = 1.0,
                                   cutoff: float = 2.5) -> float:
    """
    计算Lennard-Jones势能
    
    Args:
        positions: 原子位置 (N, 3)
        box: 盒子尺寸 (3,)
        sigma: LJ参数
        epsilon: LJ参数
        cutoff: 截断距离
    
    Returns:
        总能量
    """
    return _calculate_lj_energy_impl(positions, box, sigma, epsilon, cutoff)


def _calculate_lj_energy_impl(positions: np.ndarray,
                              box: np.ndarray,
                              sigma: float,
                              epsilon: float,
                              cutoff: float) -> float:
    """LJ能量实现"""
    n_atoms = positions.shape[0]
    cutoff_sq = cutoff ** 2
    energy = 0.0
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist_sq = 0.0
            for k in range(3):
                diff = positions[i, k] - positions[j, k]
                if box[k] > 0:
                    if diff > box[k] * 0.5:
                        diff -= box[k]
                    elif diff < -box[k] * 0.5:
                        diff += box[k]
                dist_sq += diff * diff
            
            if dist_sq < cutoff_sq and dist_sq > 0:
                r = np.sqrt(dist_sq)
                sr6 = (sigma / r) ** 6
                sr12 = sr6 ** 2
                energy += 4 * epsilon * (sr12 - sr6)
    
    return energy


if NUMBA_AVAILABLE:
    @njit(cache=True, parallel=True)
    def _calculate_lj_energy_numba(positions: np.ndarray,
                                   box: np.ndarray,
                                   sigma: float,
                                   epsilon: float,
                                   cutoff: float) -> float:
        """Numba加速版本"""
        n_atoms = positions.shape[0]
        cutoff_sq = cutoff ** 2
        
        local_energies = np.zeros(n_atoms, dtype=np.float64)
        
        for i in prange(n_atoms):
            for j in range(i + 1, n_atoms):
                dist_sq = 0.0
                for k in range(3):
                    diff = positions[i, k] - positions[j, k]
                    if box[k] > 0:
                        if diff > box[k] * 0.5:
                            diff -= box[k]
                        elif diff < -box[k] * 0.5:
                            diff += box[k]
                    dist_sq += diff * diff
                
                if dist_sq < cutoff_sq and dist_sq > 0:
                    r = np.sqrt(dist_sq)
                    sr6 = (sigma / r) ** 6
                    sr12 = sr6 ** 2
                    local_energies[i] += 4 * epsilon * (sr12 - sr6)
        
        return local_energies.sum()
    
    calculate_lennard_jones_energy = _calculate_lj_energy_numba


def calculate_lennard_jones_forces(positions: np.ndarray,
                                   box: np.ndarray,
                                   sigma: float = 1.0,
                                   epsilon: float = 1.0,
                                   cutoff: float = 2.5) -> np.ndarray:
    """
    计算Lennard-Jones力
    
    Args:
        positions: 原子位置 (N, 3)
        box: 盒子尺寸 (3,)
        sigma: LJ参数
        epsilon: LJ参数
        cutoff: 截断距离
    
    Returns:
        力数组 (N, 3)
    """
    return _calculate_lj_forces_impl(positions, box, sigma, epsilon, cutoff)


def _calculate_lj_forces_impl(positions: np.ndarray,
                              box: np.ndarray,
                              sigma: float,
                              epsilon: float,
                              cutoff: float) -> np.ndarray:
    """LJ力实现"""
    n_atoms = positions.shape[0]
    forces = np.zeros((n_atoms, 3), dtype=np.float64)
    cutoff_sq = cutoff ** 2
    
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                continue
            
            dist_sq = 0.0
            diff_vec = np.zeros(3, dtype=np.float64)
            
            for k in range(3):
                diff = positions[i, k] - positions[j, k]
                if box[k] > 0:
                    if diff > box[k] * 0.5:
                        diff -= box[k]
                    elif diff < -box[k] * 0.5:
                        diff += box[k]
                diff_vec[k] = diff
                dist_sq += diff * diff
            
            if dist_sq < cutoff_sq and dist_sq > 0:
                r = np.sqrt(dist_sq)
                sr = sigma / r
                sr6 = sr ** 6
                sr12 = sr6 ** 2
                
                f_mag = 24 * epsilon * (2 * sr12 - sr6) / r
                
                for k in range(3):
                    forces[i, k] += f_mag * diff_vec[k] / r
    
    return forces


if NUMBA_AVAILABLE:
    @njit(cache=True, parallel=True)
    def _calculate_lj_forces_numba(positions: np.ndarray,
                                   box: np.ndarray,
                                   sigma: float,
                                   epsilon: float,
                                   cutoff: float) -> np.ndarray:
        """Numba加速版本"""
        n_atoms = positions.shape[0]
        forces = np.zeros((n_atoms, 3), dtype=np.float64)
        cutoff_sq = cutoff ** 2
        
        for i in prange(n_atoms):
            for j in range(n_atoms):
                if i == j:
                    continue
                
                dist_sq = 0.0
                diff_vec = np.zeros(3, dtype=np.float64)
                
                for k in range(3):
                    diff = positions[i, k] - positions[j, k]
                    if box[k] > 0:
                        if diff > box[k] * 0.5:
                            diff -= box[k]
                        elif diff < -box[k] * 0.5:
                            diff += box[k]
                    diff_vec[k] = diff
                    dist_sq += diff * diff
                
                if dist_sq < cutoff_sq and dist_sq > 0:
                    r = np.sqrt(dist_sq)
                    sr = sigma / r
                    sr6 = sr ** 6
                    sr12 = sr6 ** 2
                    
                    f_mag = 24 * epsilon * (2 * sr12 - sr6) / r
                    
                    for k in range(3):
                        forces[i, k] += f_mag * diff_vec[k] / r
        
        return forces
    
    calculate_lennard_jones_forces = _calculate_lj_forces_numba


# =============================================================================
# 导出检查
# =============================================================================

def check_numba_availability() -> bool:
    """检查Numba是否可用"""
    return NUMBA_AVAILABLE


def get_kernel_info() -> dict:
    """获取内核信息"""
    return {
        "numba_available": NUMBA_AVAILABLE,
        "kernels": [
            "calculate_distance_matrix",
            "calculate_rdf_parallel",
            "calculate_msd_parallel",
            "build_neighbor_list",
            "calculate_lennard_jones_energy",
            "calculate_lennard_jones_forces",
        ]
    }
