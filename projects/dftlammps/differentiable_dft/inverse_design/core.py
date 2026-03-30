"""
Inverse Design Core Module
===========================

逆向设计核心模块，提供基于自动微分的材料逆向设计功能。

核心功能：
- 目标导向的结构优化
- 性质驱动的材料发现
- 参数化结构生成
- 多目标优化框架
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
from jax.experimental import optimizers
from functools import partial
from typing import Callable, Dict, List, Tuple, Optional, Any, Union, Protocol
from dataclasses import dataclass, field
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class DesignTarget:
    """设计目标定义"""
    target_type: str  # 'band_gap', 'ion_conductivity', 'bulk_modulus', etc.
    target_value: Union[float, jnp.ndarray]
    tolerance: float = 0.01
    weight: float = 1.0
    constraints: Optional[Dict] = None


@dataclass
class DesignSpace:
    """设计空间定义"""
    n_atoms: int
    atomic_species: List[str]
    composition_range: Dict[str, Tuple[float, float]]  # 组分范围
    cell_bounds: Tuple[float, float]  # 晶胞尺寸范围
    position_bounds: float = 0.5  # 原子位置变化范围


class ParameterizedStructure(ABC):
    """
    参数化结构基类
    
    将原子结构表示为可优化的连续参数
    """
    
    @abstractmethod
    def get_params(self) -> jnp.ndarray:
        """获取当前参数"""
        pass
    
    @abstractmethod
    def set_params(self, params: jnp.ndarray):
        """设置参数"""
        pass
    
    @abstractmethod
    def to_structure(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        转换为结构数据
        
        Returns:
            (positions, atomic_numbers, cell)
        """
        pass
    
    @abstractmethod
    def get_param_bounds(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        获取参数边界
        
        Returns:
            (lower_bounds, upper_bounds)
        """
        pass


class FractionalCoordinateStructure(ParameterizedStructure):
    """
    分数坐标参数化结构
    
    使用分数坐标和晶胞参数表示晶体结构
    """
    
    def __init__(self, 
                 n_atoms: int,
                 atomic_numbers: jnp.ndarray,
                 initial_cell: jnp.ndarray,
                 fix_cell: bool = False):
        """
        Args:
            n_atoms: 原子数
            atomic_numbers: 原子序数 (N,)
            initial_cell: 初始晶胞 (3, 3)
            fix_cell: 是否固定晶胞
        """
        self.n_atoms = n_atoms
        self.atomic_numbers = atomic_numbers
        self.initial_cell = initial_cell
        self.fix_cell = fix_cell
        
        # 初始化分数坐标 (随机)
        self.fractional_coords = jnp.random.rand(n_atoms, 3)
        
        # 晶胞参数 (6个独立参数: a, b, c, α, β, γ)
        if not fix_cell:
            # 从初始晶胞提取晶格参数
            a, b, c, alpha, beta, gamma = self._cell_to_params(initial_cell)
            self.cell_params = jnp.array([a, b, c, alpha, beta, gamma])
        else:
            self.cell_params = None
    
    def _cell_to_params(self, cell: jnp.ndarray) -> Tuple[float, float, float, float, float, float]:
        """晶胞矩阵到晶格参数"""
        a = jnp.linalg.norm(cell[0])
        b = jnp.linalg.norm(cell[1])
        c = jnp.linalg.norm(cell[2])
        
        alpha = jnp.arccos(jnp.dot(cell[1], cell[2]) / (b * c))
        beta = jnp.arccos(jnp.dot(cell[0], cell[2]) / (a * c))
        gamma = jnp.arccos(jnp.dot(cell[0], cell[1]) / (a * b))
        
        return float(a), float(b), float(c), float(alpha), float(beta), float(gamma)
    
    def _params_to_cell(self, params: jnp.ndarray) -> jnp.ndarray:
        """晶格参数到晶胞矩阵"""
        a, b, c, alpha, beta, gamma = params
        
        # 构建晶胞矩阵
        cell = jnp.zeros((3, 3))
        cell = cell.at[0, 0].set(a)
        cell = cell.at[0, 1].set(b * jnp.cos(gamma))
        cell = cell.at[0, 2].set(c * jnp.cos(beta))
        cell = cell.at[1, 1].set(b * jnp.sin(gamma))
        cell = cell.at[1, 2].set(c * (jnp.cos(alpha) - jnp.cos(beta) * jnp.cos(gamma)) / jnp.sin(gamma))
        cell = cell.at[2, 2].set(jnp.sqrt(c**2 - cell[0, 2]**2 - cell[1, 2]**2))
        
        return cell
    
    def get_params(self) -> jnp.ndarray:
        """获取参数向量"""
        coords_flat = self.fractional_coords.flatten()
        if self.cell_params is not None:
            return jnp.concatenate([coords_flat, self.cell_params])
        return coords_flat
    
    def set_params(self, params: jnp.ndarray):
        """设置参数"""
        if self.cell_params is not None:
            self.fractional_coords = params[:3*self.n_atoms].reshape(self.n_atoms, 3)
            self.cell_params = params[3*self.n_atoms:]
        else:
            self.fractional_coords = params.reshape(self.n_atoms, 3)
        
        # 确保分数坐标在[0, 1]范围内
        self.fractional_coords = self.fractional_coords % 1.0
    
    def to_structure(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """转换为笛卡尔坐标结构"""
        if self.cell_params is not None:
            cell = self._params_to_cell(self.cell_params)
        else:
            cell = self.initial_cell
        
        positions = self.fractional_coords @ cell
        return positions, self.atomic_numbers, cell
    
    def get_param_bounds(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """获取参数边界"""
        n_params = 3 * self.n_atoms + (0 if self.fix_cell else 6)
        
        lower = jnp.zeros(n_params)
        upper = jnp.ones(n_params)
        
        if not self.fix_cell:
            # 晶胞参数边界
            # a, b, c > 0
            lower = lower.at[3*self.n_atoms:3*self.n_atoms+3].set(2.0)  # 最小2 Bohr
            upper = upper.at[3*self.n_atoms:3*self.n_atoms+3].set(50.0)  # 最大50 Bohr
            # 角度在 (0, π) 之间
            lower = lower.at[3*self.n_atoms+3:].set(0.1)
            upper = upper.at[3*self.n_atoms+3:].set(jnp.pi - 0.1)
        
        return lower, upper


class WyckoffPositionStructure(ParameterizedStructure):
    """
    Wyckoff位置参数化结构
    
    基于晶体学Wyckoff位置的参数化，大幅减少自由度
    """
    
    def __init__(self,
                 space_group: int,
                 wyckoff_sites: List[Dict],
                 species: List[str],
                 lattice_type: str = 'cubic'):
        """
        Args:
            space_group: 空间群号 (1-230)
            wyckoff_sites: Wyckoff位置列表 [{'letter': 'a', 'species': 'Si', 'free_params': [...]}, ...]
            species: 元素种类
            lattice_type: 晶格类型
        """
        self.space_group = space_group
        self.wyckoff_sites = wyckoff_sites
        self.species = species
        self.lattice_type = lattice_type
        
        # 提取自由参数
        self.free_params = self._extract_free_params()
    
    def _extract_free_params(self) -> jnp.ndarray:
        """提取Wyckoff位置的自由参数"""
        params = []
        for site in self.wyckoff_sites:
            if 'free_params' in site:
                params.extend(site['free_params'])
        return jnp.array(params)
    
    def get_params(self) -> jnp.ndarray:
        """获取自由参数"""
        return self.free_params
    
    def set_params(self, params: jnp.ndarray):
        """设置自由参数"""
        self.free_params = params
    
    def to_structure(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """生成完整结构 (需要根据空间群展开Wyckoff位置)"""
        # 这里需要实现空间群操作来展开Wyckoff位置
        # 简化版本：直接返回占位符
        n_atoms = len(self.wyckoff_sites)
        positions = jnp.random.rand(n_atoms, 3)  # 简化
        atomic_numbers = jnp.array([14] * n_atoms)  # 假设都是Si
        cell = jnp.eye(3) * 10.0
        return positions, atomic_numbers, cell
    
    def get_param_bounds(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """参数边界"""
        n = len(self.free_params)
        return jnp.zeros(n), jnp.ones(n)


class PropertyCalculator(Protocol):
    """性质计算协议"""
    
    def compute(self, positions: jnp.ndarray, 
                atomic_numbers: jnp.ndarray,
                cell: jnp.ndarray) -> Dict[str, float]:
        ...


class ObjectiveFunction:
    """
    目标函数构建器
    
    将设计目标转换为可微分的目标函数
    """
    
    def __init__(self, 
                 property_calculator: Callable,
                 targets: List[DesignTarget],
                 regularization: Optional[Dict] = None):
        """
        Args:
            property_calculator: 性质计算函数
            targets: 设计目标列表
            regularization: 正则化参数
        """
        self.calc = property_calculator
        self.targets = targets
        self.reg = regularization or {}
    
    def __call__(self, params: jnp.ndarray,
                 structure: ParameterizedStructure) -> float:
        """
        计算目标函数值
        
        Args:
            params: 结构参数
            structure: 参数化结构对象
            
        Returns:
            目标函数值
        """
        # 更新结构
        structure.set_params(params)
        positions, atomic_numbers, cell = structure.to_structure()
        
        # 计算性质
        properties = self.calc(positions, atomic_numbers, cell)
        
        # 计算损失
        loss = 0.0
        for target in self.targets:
            if target.target_type in properties:
                value = properties[target.target_type]
                target_val = target.target_value
                
                # 相对误差
                diff = (value - target_val) / (abs(target_val) + 1e-10)
                
                # Huber损失
                delta = target.tolerance
                if abs(diff) < delta:
                    target_loss = 0.5 * diff**2
                else:
                    target_loss = delta * (abs(diff) - 0.5 * delta)
                
                loss += target.weight * target_loss
        
        # 添加正则化
        if 'position_smoothness' in self.reg:
            # 鼓励原子均匀分布
            loss += self.reg['position_smoothness'] * self._position_smoothness(positions, cell)
        
        if 'cell_volume' in self.reg:
            # 鼓励合理的晶胞体积
            volume = jnp.abs(jnp.linalg.det(cell))
            target_vol = self.reg['cell_volume']
            loss += 0.01 * ((volume - target_vol) / target_vol)**2
        
        return loss
    
    def _position_smoothness(self, positions: jnp.ndarray, 
                             cell: jnp.ndarray) -> float:
        """位置平滑度正则化 (避免原子重叠)"""
        # 计算原子间距离
        r_ij = positions[:, None, :] - positions[None, :, :]
        
        # 应用周期性边界条件
        frac = r_ij @ jnp.linalg.inv(cell)
        frac = frac - jnp.rint(frac)
        r_ij = frac @ cell
        
        distances = jnp.linalg.norm(r_ij, axis=2)
        
        # 避免除以零
        distances = jnp.where(distances < 1e-10, 1e10, distances)
        
        # 鼓励大距离 (斥力项)
        smoothness = jnp.sum(1.0 / distances**4)
        
        return smoothness
    
    def gradient(self, params: jnp.ndarray,
                 structure: ParameterizedStructure) -> jnp.ndarray:
        """
        计算目标函数梯度
        
        Args:
            params: 参数
            structure: 结构
            
        Returns:
            梯度
        """
        grad_fn = grad(lambda p: self(p, structure))
        return grad_fn(params)


class InverseDesignOptimizer:
    """
    逆向设计优化器
    
    使用梯度优化方法寻找满足目标性质的结构
    """
    
    def __init__(self,
                 objective: ObjectiveFunction,
                 optimizer_type: str = 'adam',
                 learning_rate: float = 0.01,
                 max_iter: int = 1000):
        """
        Args:
            objective: 目标函数
            optimizer_type: 优化器类型 ('adam', 'sgd', 'lbfgs')
            learning_rate: 学习率
            max_iter: 最大迭代次数
        """
        self.objective = objective
        self.optimizer_type = optimizer_type
        self.lr = learning_rate
        self.max_iter = max_iter
        
        self.history = []
    
    def optimize(self,
                 initial_structure: ParameterizedStructure,
                 callback: Optional[Callable] = None) -> ParameterizedStructure:
        """
        执行优化
        
        Args:
            initial_structure: 初始结构
            callback: 回调函数(step, structure, loss)
            
        Returns:
            优化后的结构
        """
        # 获取初始参数
        params = initial_structure.get_params()
        lower_bounds, upper_bounds = initial_structure.get_param_bounds()
        
        # 设置优化器
        if self.optimizer_type == 'adam':
            opt_init, opt_update, get_params = optimizers.adam(self.lr)
        elif self.optimizer_type == 'sgd':
            opt_init, opt_update, get_params = optimizers.sgd(self.lr)
        elif self.optimizer_type == 'momentum':
            opt_init, opt_update, get_params = optimizers.momentum(self.lr, mass=0.9)
        else:
            raise ValueError(f"不支持的优化器: {self.optimizer_type}")
        
        opt_state = opt_init(params)
        
        @jit
        def step(i, opt_state):
            p = get_params(opt_state)
            loss, g = value_and_grad(lambda x: self.objective(x, initial_structure))(p)
            return opt_update(i, g, opt_state), loss
        
        # 优化循环
        print(f"开始优化: {self.optimizer_type}, lr={self.lr}")
        for i in range(self.max_iter):
            opt_state, loss = step(i, opt_state)
            params = get_params(opt_state)
            
            # 应用边界约束
            params = jnp.clip(params, lower_bounds, upper_bounds)
            opt_state = opt_init(params)  # 重新初始化以保持约束
            
            self.history.append({'step': i, 'loss': float(loss)})
            
            if i % 50 == 0:
                print(f"Step {i}: loss = {loss:.6f}")
            
            if callback:
                initial_structure.set_params(params)
                callback(i, initial_structure, float(loss))
            
            # 收敛检查
            if i > 10 and abs(self.history[-1]['loss'] - self.history[-10]['loss']) < 1e-6:
                print(f"收敛于第 {i} 步")
                break
        
        # 设置最终参数
        initial_structure.set_params(get_params(opt_state))
        return initial_structure
    
    def optimize_with_line_search(self,
                                   initial_structure: ParameterizedStructure) -> ParameterizedStructure:
        """
        使用线搜索的优化 (更稳定但较慢)
        """
        params = initial_structure.get_params()
        lower_bounds, upper_bounds = initial_structure.get_param_bounds()
        
        for i in range(self.max_iter):
            # 计算梯度和损失
            loss, g = value_and_grad(lambda p: self.objective(p, initial_structure))(params)
            
            # 简单线搜索
            alpha = self.lr
            for _ in range(10):  # 最多尝试10次
                new_params = params - alpha * g
                new_params = jnp.clip(new_params, lower_bounds, upper_bounds)
                new_loss = self.objective(new_params, initial_structure)
                
                if new_loss < loss:
                    break
                alpha *= 0.5
            
            params = new_params
            initial_structure.set_params(params)
            
            self.history.append({'step': i, 'loss': float(loss)})
            
            if i % 50 == 0:
                print(f"Step {i}: loss = {loss:.6f}, alpha = {alpha:.6f}")
            
            # 收敛检查
            if jnp.linalg.norm(g) < 1e-5:
                print(f"梯度收敛于第 {i} 步")
                break
        
        return initial_structure


class MultiObjectiveOptimizer:
    """
    多目标优化器
    
    处理多个相互冲突的设计目标
    """
    
    def __init__(self, 
                 objectives: List[ObjectiveFunction],
                 weights: Optional[jnp.ndarray] = None):
        """
        Args:
            objectives: 目标函数列表
            weights: 权重数组
        """
        self.objectives = objectives
        self.weights = weights or jnp.ones(len(objectives)) / len(objectives)
    
    def pareto_front(self,
                     initial_structures: List[ParameterizedStructure],
                     n_points: int = 10) -> List[Tuple[ParameterizedStructure, jnp.ndarray]]:
        """
        计算Pareto前沿
        
        使用加权和方法扫描不同权重组合
        
        Args:
            initial_structures: 初始结构列表
            n_points: Pareto前沿上的点数
            
        Returns:
            [(结构, 目标值数组), ...]
        """
        pareto_set = []
        
        # 扫描权重空间
        for i in range(n_points):
            # 改变权重
            w1 = i / (n_points - 1)
            weights = jnp.array([w1, 1 - w1])
            
            # 构建组合目标
            def combined_objective(params, structure):
                total = 0.0
                for w, obj in zip(weights, self.objectives):
                    total += w * obj(params, structure)
                return total
            
            # 优化
            optimizer = InverseDesignOptimizer(
                lambda: combined_objective,  # 占位
                optimizer_type='adam',
                learning_rate=0.01
            )
            optimizer.objective = combined_objective
            
            result = optimizer.optimize(initial_structures[0])
            
            # 计算各目标值
            obj_values = jnp.array([
                obj(result.get_params(), result) 
                for obj in self.objectives
            ])
            
            pareto_set.append((result, obj_values))
        
        return pareto_set


def example_inverse_design():
    """逆向设计示例"""
    print("=" * 60)
    print("逆向设计示例")
    print("=" * 60)
    
    # 创建参数化结构
    structure = FractionalCoordinateStructure(
        n_atoms=2,
        atomic_numbers=jnp.array([14, 14]),
        initial_cell=jnp.eye(3) * 10.0,
        fix_cell=False
    )
    
    print(f"初始参数: {structure.get_params()}")
    
    # 定义简化的性质计算函数
    def mock_calculator(positions, atomic_numbers, cell):
        # 模拟带隙计算 (简化)
        volume = jnp.abs(jnp.linalg.det(cell))
        band_gap = 1.0 + 0.1 * volume  # 虚拟关系
        return {'band_gap': band_gap, 'volume': volume}
    
    # 定义设计目标
    targets = [
        DesignTarget(
            target_type='band_gap',
            target_value=2.0,  # 目标带隙 2.0 eV
            tolerance=0.1,
            weight=1.0
        )
    ]
    
    # 创建目标函数
    objective = ObjectiveFunction(mock_calculator, targets)
    
    # 创建优化器
    optimizer = InverseDesignOptimizer(
        objective,
        optimizer_type='adam',
        learning_rate=0.1,
        max_iter=200
    )
    
    # 执行优化
    print("\n开始优化...")
    result = optimizer.optimize(structure)
    
    print(f"\n优化完成!")
    print(f"最终参数: {result.get_params()}")
    
    final_props = mock_calculator(*result.to_structure())
    print(f"最终性质: {final_props}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    example_inverse_design()
