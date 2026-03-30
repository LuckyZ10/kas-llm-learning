"""
Base Phase Field Model
======================
相场基础模型类

提供所有相场模型的公共基类和接口定义。
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BoundaryConditionType(Enum):
    """边界条件类型"""
    PERIODIC = "periodic"
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    MIXED = "mixed"
    ROBIN = "robin"


@dataclass
class PhaseFieldConfig:
    """相场模型基础配置"""
    # 网格参数
    nx: int = 128  # x方向网格数
    ny: int = 128  # y方向网格数
    nz: int = 1    # z方向网格数 (2D时设为1)
    dx: float = 1.0  # 网格间距 (nm)
    dy: float = 1.0
    dz: float = 1.0
    
    # 时间参数
    dt: float = 0.001  # 时间步长
    total_steps: int = 10000  # 总步数
    save_interval: int = 100  # 保存间隔
    
    # 边界条件
    bc_type: BoundaryConditionType = BoundaryConditionType.PERIODIC
    bc_values: Dict[str, Any] = field(default_factory=dict)
    
    # 数值参数
    tolerance: float = 1e-6  # 收敛容差
    max_iter: int = 1000  # 最大迭代次数
    
    # 输出参数
    output_dir: str = "./phase_field_output"
    output_format: str = "hdf5"  # hdf5, vtk, npy
    
    def __post_init__(self):
        """配置验证"""
        if self.nx <= 0 or self.ny <= 0 or self.nz <= 0:
            raise ValueError("Grid dimensions must be positive")
        if self.dx <= 0 or self.dt <= 0:
            raise ValueError("dx and dt must be positive")


class BasePhaseFieldModel(ABC):
    """
    相场模型基类
    
    所有具体相场模型都应继承此类并实现抽象方法。
    """
    
    def __init__(self, config: PhaseFieldConfig):
        """
        初始化相场模型
        
        Args:
            config: 相场配置对象
        """
        self.config = config
        self.ndim = 3 if config.nz > 1 else 2
        
        # 初始化场变量
        self.fields = {}
        self.time = 0.0
        self.step = 0
        
        # 历史数据
        self.history = {
            'time': [],
            'energy': [],
            'fields': []
        }
        
        # 边界条件处理
        self.bc_handler = BoundaryConditionHandler(config)
        
        logger.info(f"Initialized {self.__class__.__name__} ({self.ndim}D)")
        logger.info(f"Grid: {config.nx}x{config.ny}x{config.nz}, dx={config.dx}")
    
    @abstractmethod
    def initialize_fields(self, **kwargs):
        """
        初始化场变量
        
        必须在子类中实现
        """
        pass
    
    @abstractmethod
    def compute_energy(self) -> float:
        """
        计算系统总自由能
        
        Returns:
            energy: 系统自由能
        """
        pass
    
    @abstractmethod
    def compute_chemical_potential(self, field_name: str) -> np.ndarray:
        """
        计算化学势
        
        Args:
            field_name: 场变量名称
            
        Returns:
            mu: 化学势场
        """
        pass
    
    @abstractmethod
    def evolve_step(self) -> Dict[str, Any]:
        """
        执行单步演化
        
        Returns:
            info: 包含演化信息的字典
        """
        pass
    
    def run(self, n_steps: Optional[int] = None, 
            callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        运行模拟
        
        Args:
            n_steps: 运行步数 (默认使用config中的total_steps)
            callback: 每步回调函数
            
        Returns:
            results: 运行结果统计
        """
        n_steps = n_steps or self.config.total_steps
        
        logger.info(f"Starting simulation for {n_steps} steps")
        
        for i in range(n_steps):
            # 执行演化步骤
            info = self.evolve_step()
            
            # 更新时间
            self.step += 1
            self.time += self.config.dt
            
            # 保存历史
            if self.step % self.config.save_interval == 0:
                self._save_history()
            
            # 回调函数
            if callback:
                callback(self.step, self.time, info)
            
            # 收敛检查
            if info.get('converged', False):
                logger.info(f"Converged at step {self.step}")
                break
        
        results = {
            'total_steps': self.step,
            'final_time': self.time,
            'final_energy': self.compute_energy(),
            'converged': info.get('converged', False)
        }
        
        logger.info(f"Simulation completed: {results}")
        return results
    
    def get_field(self, name: str) -> np.ndarray:
        """获取场变量"""
        if name not in self.fields:
            raise KeyError(f"Field '{name}' not found. Available: {list(self.fields.keys())}")
        return self.fields[name].copy()
    
    def set_field(self, name: str, value: np.ndarray):
        """设置场变量"""
        self.fields[name] = value.copy()
    
    def _save_history(self):
        """保存当前状态到历史"""
        self.history['time'].append(self.time)
        self.history['energy'].append(self.compute_energy())
        self.history['fields'].append({
            name: field.copy() for name, field in self.fields.items()
        })
    
    def compute_gradient(self, field: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        计算场梯度
        
        Args:
            field: 输入场
            
        Returns:
            gradients: 梯度分量元组
        """
        if self.ndim == 2:
            grad_x = np.gradient(field, self.config.dx, axis=0)
            grad_y = np.gradient(field, self.config.dy, axis=1)
            return (grad_x, grad_y)
        else:
            grad_x = np.gradient(field, self.config.dx, axis=0)
            grad_y = np.gradient(field, self.config.dy, axis=1)
            grad_z = np.gradient(field, self.config.dz, axis=2)
            return (grad_x, grad_y, grad_z)
    
    def compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        计算拉普拉斯算子
        
        Args:
            field: 输入场
            
        Returns:
            laplacian: 拉普拉斯场
        """
        if self.ndim == 2:
            laplacian = (
                np.gradient(np.gradient(field, self.config.dx, axis=0), 
                           self.config.dx, axis=0) +
                np.gradient(np.gradient(field, self.config.dy, axis=1), 
                           self.config.dy, axis=1)
            )
        else:
            laplacian = (
                np.gradient(np.gradient(field, self.config.dx, axis=0), 
                           self.config.dx, axis=0) +
                np.gradient(np.gradient(field, self.config.dy, axis=1), 
                           self.config.dy, axis=1) +
                np.gradient(np.gradient(field, self.config.dz, axis=2), 
                           self.config.dz, axis=2)
            )
        return laplacian
    
    def compute_interface_width(self, field: np.ndarray, 
                                threshold: float = 0.1) -> float:
        """
        计算界面宽度
        
        Args:
            field: 序参量场
            threshold: 界面阈值
            
        Returns:
            width: 界面宽度 (nm)
        """
        # 找到界面区域 (0.1 < phi < 0.9)
        interface_mask = (field > threshold) & (field < 1 - threshold)
        
        if not np.any(interface_mask):
            return 0.0
        
        # 计算界面区域的最大距离
        coords = np.argwhere(interface_mask)
        if len(coords) > 1:
            max_dist = np.max(np.linalg.norm(coords - coords.mean(axis=0), axis=1))
            width = max_dist * self.config.dx
        else:
            width = 0.0
        
        return width
    
    def export_to_vtk(self, filename: str):
        """
        导出到VTK格式用于可视化
        
        Args:
            filename: 输出文件名
        """
        try:
            import pyevtk.hl as evtk
            
            x = np.arange(self.config.nx) * self.config.dx
            y = np.arange(self.config.ny) * self.config.dy
            z = np.arange(self.config.nz) * self.config.dz
            
            cell_data = {}
            for name, field in self.fields.items():
                if self.ndim == 2:
                    cell_data[name] = field[:, :, np.newaxis]
                else:
                    cell_data[name] = field
            
            evtk.gridToVTK(filename, x, y, z, cellData=cell_data)
            logger.info(f"Exported to VTK: {filename}")
            
        except ImportError:
            logger.warning("pyevtk not available, VTK export skipped")
            # 保存为numpy格式作为回退
            np.savez(f"{filename}.npz", **self.fields)


class BoundaryConditionHandler:
    """边界条件处理器"""
    
    def __init__(self, config: PhaseFieldConfig):
        self.config = config
        self.bc_type = config.bc_type
    
    def apply(self, field: np.ndarray) -> np.ndarray:
        """
        应用边界条件
        
        Args:
            field: 输入场
            
        Returns:
            field: 应用边界条件后的场
        """
        if self.bc_type == BoundaryConditionType.PERIODIC:
            return self._apply_periodic(field)
        elif self.bc_type == BoundaryConditionType.DIRICHLET:
            return self._apply_dirichlet(field)
        elif self.bc_type == BoundaryConditionType.NEUMANN:
            return self._apply_neumann(field)
        else:
            return field
    
    def _apply_periodic(self, field: np.ndarray) -> np.ndarray:
        """应用周期性边界条件"""
        # NumPy的gradient函数自动处理周期性边界
        return field
    
    def _apply_dirichlet(self, field: np.ndarray) -> np.ndarray:
        """应用Dirichlet边界条件"""
        values = self.config.bc_values.get('dirichlet', 0.0)
        
        if self.ndim == 2:
            field[0, :] = values
            field[-1, :] = values
            field[:, 0] = values
            field[:, -1] = values
        else:
            field[0, :, :] = values
            field[-1, :, :] = values
            field[:, 0, :] = values
            field[:, -1, :] = values
            field[:, :, 0] = values
            field[:, :, -1] = values
        
        return field
    
    def _apply_neumann(self, field: np.ndarray) -> np.ndarray:
        """应用Neumann边界条件 (零通量)"""
        if self.ndim == 2:
            field[0, :] = field[1, :]
            field[-1, :] = field[-2, :]
            field[:, 0] = field[:, 1]
            field[:, -1] = field[:, -2]
        else:
            field[0, :, :] = field[1, :, :]
            field[-1, :, :] = field[-2, :, :]
            field[:, 0, :] = field[:, 1, :]
            field[:, -1, :] = field[:, -2, :]
            field[:, :, 0] = field[:, :, 1]
            field[:, :, -1] = field[:, :, -2]
        
        return field
    
    @property
    def ndim(self) -> int:
        return 3 if self.config.nz > 1 else 2
