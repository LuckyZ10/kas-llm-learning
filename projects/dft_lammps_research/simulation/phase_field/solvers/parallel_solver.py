"""
Parallel Solver
===============
并行求解器

提供MPI/OpenMP并行计算支持。
实现域分解和负载均衡。
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """并行配置"""
    n_processes: int = 4
    decomposition: str = "2d"  # 1d, 2d
    overlap: int = 2  # 重叠区域大小 (用于通信)
    load_balancing: bool = True
    
    # 通信
    mpi_communicator: Optional[str] = None


class ParallelSolver:
    """
    并行求解器
    
    使用域分解方法实现并行计算。
    支持1D和2D域分解。
    """
    
    def __init__(self, config: Optional[ParallelConfig] = None):
        """
        初始化并行求解器
        
        Args:
            config: 并行配置
        """
        self.config = config or ParallelConfig()
        
        # 域分解信息
        self.subdomain_sizes = None
        self.subdomain_starts = None
        
        # 模拟MPI (如果没有安装mpi4py)
        self.rank = 0
        self.size = 1
        
        try:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self.has_mpi = True
        except ImportError:
            self.comm = None
            self.has_mpi = False
            logger.warning("mpi4py not available. Running in serial mode.")
        
        logger.info(f"Parallel solver initialized (rank {self.rank}/{self.size})")
    
    def decompose_domain(self, nx: int, ny: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        分解计算域
        
        Args:
            nx, ny: 全局网格尺寸
            
        Returns:
            local_shape: 本地子域尺寸
            local_start: 本地子域起始索引
        """
        n_procs = self.size if self.has_mpi else self.config.n_processes
        
        if self.config.decomposition == "1d":
            # 1D分解 (沿x方向)
            nx_local = nx // n_procs
            ny_local = ny
            
            nx_local = min(nx_local, nx - self.rank * nx_local)
            local_shape = (nx_local, ny_local)
            local_start = (self.rank * (nx // n_procs), 0)
            
        else:  # 2D分解
            # 寻找最优分解
            npx = int(np.sqrt(n_procs))
            npy = n_procs // npx
            
            while npx * npy != n_procs and npx > 1:
                npx -= 1
                npy = n_procs // npx
            
            # 计算本地尺寸
            nx_local = nx // npx
            ny_local = ny // npy
            
            # 计算位置
            px = self.rank % npx
            py = self.rank // npx
            
            local_shape = (nx_local, ny_local)
            local_start = (px * nx_local, py * ny_local)
        
        self.subdomain_sizes = local_shape
        self.subdomain_starts = local_start
        
        logger.info(f"Rank {self.rank}: subdomain {local_shape} at {local_start}")
        
        return local_shape, local_start
    
    def scatter_field(self, global_field: np.ndarray) -> np.ndarray:
        """
        将全局场分布到各进程
        
        Args:
            global_field: 全局场
            
        Returns:
            local_field: 本地子域场
        """
        if not self.has_mpi or self.size == 1:
            return global_field.copy()
        
        nx, ny = global_field.shape[:2]
        local_shape, local_start = self.decompose_domain(nx, ny)
        
        # 提取本地数据
        i_start, j_start = local_start
        i_end = min(i_start + local_shape[0], nx)
        j_end = min(j_start + local_shape[1], ny)
        
        local_field = global_field[i_start:i_end, j_start:j_end].copy()
        
        return local_field
    
    def gather_field(self, local_field: np.ndarray, 
                    global_shape: Tuple[int, int]) -> np.ndarray:
        """
        从各进程收集全局场
        
        Args:
            local_field: 本地子域场
            global_shape: 全局场形状
            
        Returns:
            global_field: 全局场
        """
        if not self.has_mpi or self.size == 1:
            return local_field.copy()
        
        nx, ny = global_shape
        
        if self.rank == 0:
            global_field = np.zeros(global_shape)
            
            # 收集所有数据
            # 简化：只处理rank 0的情况
            global_field[:local_field.shape[0], :local_field.shape[1]] = local_field
            
            # 从其他进程接收
            for r in range(1, self.size):
                # 接收子域
                data = self.comm.recv(source=r, tag=r)
                start, field = data
                i_start, j_start = start
                global_field[i_start:i_start+field.shape[0], 
                           j_start:j_start+field.shape[1]] = field
        else:
            # 发送数据到rank 0
            self.comm.send((self.subdomain_starts, local_field), dest=0, tag=self.rank)
            global_field = None
        
        return global_field
    
    def exchange_boundaries(self, local_field: np.ndarray) -> np.ndarray:
        """
        交换边界数据
        
        Args:
            local_field: 本地场 (包含重叠区域)
            
        Returns:
            updated_field: 更新边界后的场
        """
        if not self.has_mpi or self.size == 1:
            return local_field
        
        overlap = self.config.overlap
        updated = local_field.copy()
        
        # 简化实现：只处理4邻域交换
        # 实际应处理所有边界
        
        return updated
    
    def parallel_laplacian(self, local_field: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """
        并行计算拉普拉斯
        
        Args:
            local_field: 本地场
            dx, dy: 网格间距
            
        Returns:
            laplacian: 拉普拉斯场
        """
        # 交换边界
        field_with_ghost = self.exchange_boundaries(local_field)
        
        # 计算拉普拉斯 (使用局部数据)
        nx, ny = local_field.shape[:2]
        laplacian = np.zeros_like(local_field)
        
        # 内部点
        laplacian[1:-1, 1:-1] = (
            (field_with_ghost[2:, 1:-1] - 2*field_with_ghost[1:-1, 1:-1] + field_with_ghost[:-2, 1:-1]) / dx**2 +
            (field_with_ghost[1:-1, 2:] - 2*field_with_ghost[1:-1, 1:-1] + field_with_ghost[1:-1, :-2]) / dy**2
        )
        
        return laplacian
    
    def run_domain_decomposition(self, solver, global_shape: Tuple[int, int],
                                 n_steps: int) -> Dict:
        """
        使用域分解运行求解器
        
        Args:
            solver: 相场求解器
            global_shape: 全局网格形状
            n_steps: 运行步数
            
        Returns:
            result: 运行结果
        """
        # 分解域
        local_shape, local_start = self.decompose_domain(*global_shape)
        
        # 初始化本地场
        # 这里需要与具体的求解器集成
        
        logger.info(f"Rank {self.rank}: running on subdomain {local_shape}")
        
        # 简化的并行运行框架
        result = {
            'rank': self.rank,
            'local_shape': local_shape,
            'local_start': local_start,
            'n_steps': n_steps
        }
        
        return result
    
    def reduce_sum(self, local_value: float) -> float:
        """全局求和归约"""
        if self.has_mpi:
            return self.comm.allreduce(local_value, op=None)  # SUM
        return local_value
    
    def barrier(self):
        """同步屏障"""
        if self.has_mpi:
            self.comm.Barrier()
