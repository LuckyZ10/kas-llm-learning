"""
Adaptive Mesh Refinement
========================
自适应网格细化

根据解的特征自动调整网格分辨率。
在界面区域使用高分辨率，体相区域使用低分辨率。
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AMRConfig:
    """自适应网格配置"""
    min_level: int = 0  # 最粗级别
    max_level: int = 4  # 最细级别
    refinement_threshold: float = 0.1
    coarsening_threshold: float = 0.01
    
    # 细化策略
    refinement_strategy: str = "gradient"  # gradient, error, interface
    
    # 负载均衡
    load_balancing: bool = True
    
    # 插值
    interpolation_order: int = 2


class AdaptiveMesh:
    """
    自适应网格
    
    使用四叉树/八叉树结构实现自适应网格细化。
    """
    
    def __init__(self, config: Optional[AMRConfig] = None):
        """
        初始化自适应网格
        
        Args:
            config: 自适应网格配置
        """
        self.config = config or AMRConfig()
        
        # 网格层级
        self.levels = {}
        self.max_level = 0
        
        # 网格单元
        self.cells = []
        
        logger.info(f"Adaptive mesh initialized")
        logger.info(f"Max refinement level: {self.config.max_level}")
    
    def create_base_mesh(self, nx: int, ny: int, 
                        xlim: Tuple[float, float] = (0, 1),
                        ylim: Tuple[float, float] = (0, 1)):
        """
        创建基础网格
        
        Args:
            nx, ny: 基础网格尺寸
            xlim, ylim: 边界范围
        """
        self.base_nx = nx
        self.base_ny = ny
        self.xlim = xlim
        self.ylim = ylim
        
        # 创建基础单元
        dx = (xlim[1] - xlim[0]) / nx
        dy = (ylim[1] - ylim[0]) / ny
        
        self.cells = []
        for i in range(nx):
            for j in range(ny):
                cell = {
                    'i': i,
                    'j': j,
                    'level': 0,
                    'x': xlim[0] + (i + 0.5) * dx,
                    'y': ylim[0] + (j + 0.5) * dy,
                    'dx': dx,
                    'dy': dy,
                    'children': [],
                    'parent': None
                }
                self.cells.append(cell)
        
        self.levels[0] = self.cells.copy()
        
        logger.info(f"Created base mesh: {nx}x{ny} cells")
    
    def refine_cell(self, cell_index: int):
        """
        细化单元
        
        Args:
            cell_index: 单元索引
        """
        cell = self.cells[cell_index]
        
        if cell['level'] >= self.config.max_level:
            return
        
        # 创建4个子单元
        dx_new = cell['dx'] / 2
        dy_new = cell['dy'] / 2
        level_new = cell['level'] + 1
        
        children = []
        for ii in range(2):
            for jj in range(2):
                child = {
                    'i': cell['i'] * 2 + ii,
                    'j': cell['j'] * 2 + jj,
                    'level': level_new,
                    'x': cell['x'] + (ii - 0.5) * dx_new,
                    'y': cell['y'] + (jj - 0.5) * dy_new,
                    'dx': dx_new,
                    'dy': dy_new,
                    'children': [],
                    'parent': cell_index
                }
                children.append(len(self.cells))
                self.cells.append(child)
        
        cell['children'] = children
        self.max_level = max(self.max_level, level_new)
        
        # 添加到层级
        if level_new not in self.levels:
            self.levels[level_new] = []
        self.levels[level_new].extend(children)
    
    def coarsen_cell(self, cell_index: int):
        """
        粗化单元
        
        Args:
            cell_index: 单元索引
        """
        cell = self.cells[cell_index]
        
        if not cell['children']:
            return
        
        # 移除子单元
        for child_idx in cell['children']:
            self.cells[child_idx] = None
        
        cell['children'] = []
    
    def adapt(self, field_values: Dict[int, float]):
        """
        根据场值调整网格
        
        Args:
            field_values: 单元索引到场值的字典
        """
        # 标记需要细化的单元
        refine_list = []
        coarsen_list = []
        
        for cell_idx, value in field_values.items():
            if cell_idx >= len(self.cells) or self.cells[cell_idx] is None:
                continue
            
            cell = self.cells[cell_idx]
            
            # 基于梯度估计
            if cell['level'] < self.config.max_level:
                # 检查是否需要细化
                if abs(value) > self.config.refinement_threshold:
                    refine_list.append(cell_idx)
            
            # 检查是否需要粗化
            if cell['level'] > self.config.min_level:
                if abs(value) < self.config.coarsening_threshold:
                    coarsen_list.append(cell_idx)
        
        # 执行细化
        for cell_idx in refine_list:
            self.refine_cell(cell_idx)
        
        # 执行粗化
        for cell_idx in coarsen_list:
            self.coarsen_cell(cell_idx)
        
        logger.info(f"Adapted mesh: {len(refine_list)} refined, {len(coarsen_list)} coarsened")
    
    def interpolate_to_finest(self, field_on_cells: Dict[int, float]) -> np.ndarray:
        """
        将所有级别的场插值到最细网格
        
        Args:
            field_on_cells: 单元场值
            
        Returns:
            fine_grid_field: 最细网格上的场
        """
        # 计算最细网格尺寸
        finest_nx = self.base_nx * (2 ** self.max_level)
        finest_ny = self.base_ny * (2 ** self.max_level)
        
        fine_grid = np.zeros((finest_nx, finest_ny))
        
        # 插值 (简化)
        for cell_idx, value in field_on_cells.items():
            if cell_idx >= len(self.cells) or self.cells[cell_idx] is None:
                continue
            
            cell = self.cells[cell_idx]
            level = cell['level']
            
            # 计算在最细网格上的范围
            factor = 2 ** (self.max_level - level)
            i_start = cell['i'] * factor
            i_end = (cell['i'] + 1) * factor
            j_start = cell['j'] * factor
            j_end = (cell['j'] + 1) * factor
            
            fine_grid[i_start:i_end, j_start:j_end] = value
        
        return fine_grid
    
    def get_interface_cells(self, phase_field: np.ndarray, 
                           threshold: float = 0.1) -> List[int]:
        """
        识别界面区域单元
        
        Args:
            phase_field: 相场值数组
            threshold: 界面阈值
            
        Returns:
            interface_cells: 界面单元索引列表
        """
        interface_cells = []
        
        for cell_idx, cell in enumerate(self.cells):
            if cell is None or cell['children']:
                continue
            
            # 计算单元内的场变化
            # 简化：使用单元中心值
            if cell_idx < len(phase_field):
                value = phase_field[cell_idx]
                
                # 如果场值在界面区域
                if threshold < abs(value) < 1 - threshold:
                    interface_cells.append(cell_idx)
        
        return interface_cells
    
    def estimate_error(self, field_values: Dict[int, float]) -> Dict[int, float]:
        """
        估计误差分布
        
        Args:
            field_values: 场值
            
        Returns:
            error_estimate: 误差估计
        """
        error = {}
        
        for cell_idx, value in field_values.items():
            if cell_idx >= len(self.cells) or self.cells[cell_idx] is None:
                continue
            
            cell = self.cells[cell_idx]
            
            # 简化：使用梯度作为误差估计
            # 实际应使用更复杂的后验误差估计
            
            # 计算邻近单元差异
            neighbors = self._get_neighbors(cell_idx)
            if neighbors:
                neighbor_values = [field_values.get(n, value) for n in neighbors]
                gradient = max(abs(v - value) for v in neighbor_values)
                error[cell_idx] = gradient
            else:
                error[cell_idx] = 0
        
        return error
    
    def _get_neighbors(self, cell_index: int) -> List[int]:
        """获取单元邻居"""
        # 简化实现
        return []
    
    def get_mesh_info(self) -> Dict:
        """获取网格统计信息"""
        n_cells_total = sum(1 for c in self.cells if c is not None)
        n_cells_active = sum(1 for c in self.cells 
                            if c is not None and not c['children'])
        
        return {
            'total_cells': n_cells_total,
            'active_cells': n_cells_active,
            'max_level': self.max_level,
            'base_grid': (self.base_nx, self.base_ny),
            'equivalent_uniform': (self.base_nx * 2**self.max_level,
                                   self.base_ny * 2**self.max_level)
        }
