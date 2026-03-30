"""
GPU Accelerated Solver
======================
GPU加速求解器 (CuPy/Numba)

利用NVIDIA GPU加速相场方程求解。
支持FFT-based谱方法和有限差分。
"""

import numpy as np
from typing import Dict, Optional, Tuple, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# 尝试导入CuPy
try:
    import cupy as cp
    from cupy.cuda import Device
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logger.warning("CuPy not available. GPU solver will use CPU fallback.")

# 尝试导入Numba CUDA
try:
    from numba import cuda
    NUMBA_CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    NUMBA_CUDA_AVAILABLE = False


@dataclass
class GPUConfig:
    """GPU求解器配置"""
    device_id: int = 0  # GPU设备ID
    use_cupy: bool = True  # 使用CuPy (否则用Numba)
    use_float32: bool = False  # 使用单精度
    memory_pool: bool = True  # 使用内存池
    
    # 性能参数
    block_size: Tuple[int, ...] = (16, 16)  # CUDA线程块大小
    fft_cache: bool = True  # 缓存FFT计划


class GPUSolver:
    """
    GPU加速相场求解器
    
    使用CuPy或Numba CUDA加速计算。
    对于大规模问题(>512³网格)提供显著加速。
    """
    
    def __init__(self, config: Optional[GPUConfig] = None):
        """
        初始化GPU求解器
        
        Args:
            config: GPU配置
        """
        self.config = config or GPUConfig()
        
        # 检查GPU可用性
        self.gpu_available = self._check_gpu()
        
        if not self.gpu_available:
            logger.warning("GPU not available. Using CPU fallback.")
            self.xp = np
        else:
            logger.info(f"GPU solver initialized on device {self.config.device_id}")
            if self.config.use_cupy and CUPY_AVAILABLE:
                self.xp = cp
                # 设置设备
                cp.cuda.Device(self.config.device_id).use()
                
                # 内存池
                if self.config.memory_pool:
                    self._init_memory_pool()
            else:
                self.xp = np
                logger.info("Using Numba CUDA kernels")
        
        # FFT缓存
        self._fft_plan_cache = {}
    
    def _check_gpu(self) -> bool:
        """检查GPU是否可用"""
        if self.config.use_cupy and CUPY_AVAILABLE:
            try:
                cp.cuda.runtime.getDeviceCount()
                return True
            except:
                return False
        elif NUMBA_CUDA_AVAILABLE:
            return cuda.is_available()
        return False
    
    def _init_memory_pool(self):
        """初始化CuPy内存池"""
        if CUPY_AVAILABLE:
            # 使用内存池管理显存
            pool = cp.cuda.MemoryPool()
            cp.cuda.set_allocator(pool.malloc)
    
    def to_gpu(self, array: np.ndarray) -> 'cp.ndarray':
        """
        将数组转移到GPU
        
        Args:
            array: NumPy数组
            
        Returns:
            gpu_array: GPU数组
        """
        if self.gpu_available and CUPY_AVAILABLE:
            dtype = cp.float32 if self.config.use_float32 else cp.float64
            return cp.asarray(array, dtype=dtype)
        return array
    
    def to_cpu(self, array) -> np.ndarray:
        """
        将数组转移到CPU
        
        Args:
            array: GPU数组
            
        Returns:
            cpu_array: NumPy数组
        """
        if self.gpu_available and CUPY_AVAILABLE and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return array
    
    def fft(self, field, axes=None) -> 'cp.ndarray':
        """
        GPU加速FFT
        
        Args:
            field: 输入场
            axes: 变换轴
            
        Returns:
            field_fft: 傅里叶变换结果
        """
        if self.gpu_available and CUPY_AVAILABLE:
            return cp.fft.fftn(field, axes=axes)
        else:
            return np.fft.fftn(field, axes=axes)
    
    def ifft(self, field_fft, axes=None) -> 'cp.ndarray':
        """GPU加速逆FFT"""
        if self.gpu_available and CUPY_AVAILABLE:
            return cp.fft.ifftn(field_fft, axes=axes)
        else:
            return np.fft.ifftn(field_fft, axes=axes)
    
    def laplacian_spectral(self, field: np.ndarray, 
                          k_squared: np.ndarray) -> np.ndarray:
        """
        谱方法计算拉普拉斯算子
        
        ∇²u = IFFT(-k² * FFT(u))
        
        Args:
            field: 输入场 (CPU或GPU)
            k_squared: 波矢平方 (CPU或GPU)
            
        Returns:
            laplacian: 拉普拉斯场
        """
        # 确保数据在正确的设备上
        if self.gpu_available and CUPY_AVAILABLE:
            field_gpu = self.to_gpu(field) if isinstance(field, np.ndarray) else field
            k2_gpu = self.to_gpu(k_squared) if isinstance(k_squared, np.ndarray) else k_squared
            
            # GPU FFT
            field_fft = cp.fft.fftn(field_gpu)
            laplacian_fft = -k2_gpu * field_fft
            laplacian = cp.fft.ifftn(laplacian_fft).real
            
            return self.to_cpu(laplacian)
        else:
            # CPU回退
            field_fft = np.fft.fftn(field)
            laplacian_fft = -k_squared * field_fft
            laplacian = np.fft.ifftn(laplacian_fft).real
            return laplacian
    
    def gradient_spectral(self, field: np.ndarray,
                         kx: np.ndarray, ky: np.ndarray,
                         kz: Optional[np.ndarray] = None) -> Tuple:
        """
        谱方法计算梯度
        
        ∂u/∂x = IFFT(ikx * FFT(u))
        """
        if self.gpu_available and CUPY_AVAILABLE:
            field_gpu = self.to_gpu(field)
            kx_gpu = self.to_gpu(kx)
            ky_gpu = self.to_gpu(ky)
            
            field_fft = cp.fft.fftn(field_gpu)
            
            grad_x = cp.fft.ifftn(1j * kx_gpu * field_fft).real
            grad_y = cp.fft.ifftn(1j * ky_gpu * field_fft).real
            
            if kz is not None:
                kz_gpu = self.to_gpu(kz)
                grad_z = cp.fft.ifftn(1j * kz_gpu * field_fft).real
                return (self.to_cpu(grad_x), self.to_cpu(grad_y), self.to_cpu(grad_z))
            
            return (self.to_cpu(grad_x), self.to_cpu(grad_y))
        else:
            # CPU回退
            field_fft = np.fft.fftn(field)
            grad_x = np.fft.ifftn(1j * kx * field_fft).real
            grad_y = np.fft.ifftn(1j * ky * field_fft).real
            
            if kz is not None:
                grad_z = np.fft.ifftn(1j * kz * field_fft).real
                return (grad_x, grad_y, grad_z)
            
            return (grad_x, grad_y)
    
    def cahn_hilliard_step(self, c: np.ndarray, mu: np.ndarray,
                          M: float, dt: float, k_squared: np.ndarray) -> np.ndarray:
        """
        GPU加速Cahn-Hilliard演化步骤
        
        Args:
            c: 浓度场
            mu: 化学势场
            M: 迁移率
            dt: 时间步长
            k_squared: 波矢平方
            
        Returns:
            c_new: 新浓度场
        """
        if self.gpu_available and CUPY_AVAILABLE:
            c_gpu = self.to_gpu(c)
            mu_gpu = self.to_gpu(mu)
            k2_gpu = self.to_gpu(k_squared)
            
            # FFT
            c_fft = cp.fft.fftn(c_gpu)
            mu_fft = cp.fft.fftn(mu_gpu)
            
            # 谱方法演化: ĉ^{n+1} = ĉ^n - dt * M * k² * μ̂
            c_new_fft = c_fft - dt * M * k2_gpu * mu_fft
            c_new = cp.fft.ifftn(c_new_fft).real
            
            return self.to_cpu(c_new)
        else:
            # CPU回退
            c_fft = np.fft.fftn(c)
            mu_fft = np.fft.fftn(mu)
            c_new_fft = c_fft - dt * M * k_squared * mu_fft
            c_new = np.fft.ifftn(c_new_fft).real
            return c_new
    
    def allen_cahn_step(self, eta: np.ndarray, df_deta: np.ndarray,
                       L: float, dt: float, k_squared: np.ndarray,
                       kappa: float) -> np.ndarray:
        """
        GPU加速Allen-Cahn演化步骤
        
        η^{n+1} = η^n - dt * L * (f'(η) + κk²η)
        """
        if self.gpu_available and CUPY_AVAILABLE:
            eta_gpu = self.to_gpu(eta)
            df_gpu = self.to_gpu(df_deta)
            k2_gpu = self.to_gpu(k_squared)
            
            eta_fft = cp.fft.fftn(eta_gpu)
            df_fft = cp.fft.fftn(df_gpu)
            
            # 谱方法
            eta_new_fft = eta_fft - dt * L * (df_fft + kappa * k2_gpu * eta_fft)
            eta_new = cp.fft.ifftn(eta_new_fft).real
            
            return self.to_cpu(eta_new)
        else:
            eta_fft = np.fft.fftn(eta)
            df_fft = np.fft.fftn(df_deta)
            eta_new_fft = eta_fft - dt * L * (df_fft + kappa * k_squared * eta_fft)
            eta_new = np.fft.ifftn(eta_new_fft).real
            return eta_new
    
    def batch_process(self, fields: Dict[str, np.ndarray],
                     operation: Callable) -> Dict[str, np.ndarray]:
        """
        批量GPU处理多个场
        
        Args:
            fields: 场变量字典
            operation: 处理函数
            
        Returns:
            results: 处理结果
        """
        if self.gpu_available and CUPY_AVAILABLE:
            # 批量转移到GPU
            fields_gpu = {k: self.to_gpu(v) for k, v in fields.items()}
            
            # 执行操作
            results_gpu = operation(fields_gpu)
            
            # 批量转移回CPU
            results = {k: self.to_cpu(v) for k, v in results_gpu.items()}
            return results
        else:
            # CPU回退
            return operation(fields)
    
    def get_memory_info(self) -> Dict:
        """获取GPU内存信息"""
        if CUPY_AVAILABLE:
            mem_info = cp.cuda.Device(self.config.device_id).mem_info
            return {
                'free_bytes': int(mem_info[0]),
                'total_bytes': int(mem_info[1]),
                'used_bytes': int(mem_info[1] - mem_info[0]),
            }
        return {'error': 'CuPy not available'}
    
    def clear_cache(self):
        """清除GPU缓存"""
        if CUPY_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()


# Numba CUDA内核 (用于自定义操作)
if NUMBA_CUDA_AVAILABLE:
    @cuda.jit
    def _laplacian_kernel_2d(field, laplacian, dx, dy):
        """2D拉普拉斯CUDA内核"""
        i, j = cuda.grid(2)
        nx, ny = field.shape
        
        if 0 < i < nx - 1 and 0 < j < ny - 1:
            laplacian[i, j] = (
                (field[i+1, j] - 2*field[i, j] + field[i-1, j]) / dx**2 +
                (field[i, j+1] - 2*field[i, j] + field[i, j-1]) / dy**2
            )
    
    @cuda.jit
    def _gradient_kernel_2d(field, grad_x, grad_y, dx, dy):
        """2D梯度CUDA内核"""
        i, j = cuda.grid(2)
        nx, ny = field.shape
        
        if 0 < i < nx - 1 and 0 < j < ny - 1:
            grad_x[i, j] = (field[i+1, j] - field[i-1, j]) / (2 * dx)
            grad_y[i, j] = (field[i, j+1] - field[i, j-1]) / (2 * dy)
