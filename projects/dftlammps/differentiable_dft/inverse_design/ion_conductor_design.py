"""
Ion Conductor Inverse Design Module
===================================

离子导体逆向设计模块 - 设计高离子电导率的固态电解质材料。

应用案例：
- 全固态锂电池电解质
- 钠离子电池固态电解质
- 氧离子导体 (SOFC)
- 质子导体

核心目标：
- 最大化离子电导率
- 最小化电子电导率
- 优化离子迁移通道
- 确保热力学稳定性
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field
import numpy as np

from .core import (
    DesignTarget, DesignSpace, ParameterizedStructure,
    FractionalCoordinateStructure, ObjectiveFunction, InverseDesignOptimizer
)


@dataclass
class IonConductorTarget:
    """离子导体设计目标"""
    ion_type: str = 'Li'  # 'Li', 'Na', 'K', 'Ag', 'O', 'H'
    target_conductivity: float = 1e-3  # S/cm
    temperature: float = 300.0  # K
    
    # 约束条件
    min_migration_barrier: float = 0.2  # eV (最大可接受迁移势垒)
    max_electronic_conductivity: float = 1e-6  # S/cm
    stability_window: float = 2.0  # V (电化学稳定窗口)
    
    # 优化权重
    conductivity_weight: float = 1.0
    barrier_weight: float = 0.5
    stability_weight: float = 0.3


class IonMigrationAnalyzer:
    """
    离子迁移分析器
    
    分析离子在材料中的迁移路径和势垒
    """
    
    def __init__(self, dft_engine: Any):
        self.dft = dft_engine
        
        # 离子半径 (Angstrom)
        self.ion_radii = {
            'Li': 0.76, 'Na': 1.02, 'K': 1.38,
            'Ag': 1.15, 'Cu': 0.77, 'Mg': 0.72,
            'O': 1.40, 'H': 0.13, 'F': 1.33
        }
    
    def find_migration_paths(self,
                            positions: jnp.ndarray,
                            atomic_numbers: jnp.ndarray,
                            cell: jnp.ndarray,
                            ion_type: str) -> List[Dict]:
        """
        寻找离子迁移路径
        
        使用Voronoi分析和键连通性方法
        
        Returns:
            迁移路径列表，每条路径包含：
            - start_site: 起始位点
            - end_site: 终止位点
            - midpoint: 路径中点 (鞍点近似)
            - distance: 路径长度
        """
        # 识别离子位点
        ion_sites = self._identify_ion_sites(positions, atomic_numbers, ion_type)
        
        paths = []
        n_sites = len(ion_sites)
        
        for i in range(n_sites):
            for j in range(i+1, n_sites):
                start = ion_sites[i]
                end = ion_sites[j]
                
                # 计算距离 (考虑周期性)
                dr = end - start
                frac_dr = dr @ jnp.linalg.inv(cell)
                frac_dr = frac_dr - jnp.rint(frac_dr)
                dr = frac_dr @ cell
                distance = jnp.linalg.norm(dr)
                
                # 只考虑合理距离的位点对
                if distance < 5.0:  # 5 Bohr  cutoff
                    paths.append({
                        'start_site': start,
                        'end_site': end,
                        'midpoint': (start + end) / 2,
                        'distance': float(distance)
                    })
        
        return paths
    
    def _identify_ion_sites(self,
                           positions: jnp.ndarray,
                           atomic_numbers: jnp.ndarray,
                           ion_type: str) -> jnp.ndarray:
        """识别可能的离子占据位点"""
        # 简化：返回所有空位附近的位置
        # 实际应用需要Voronoi分析或静电势分析
        
        # 在网格上采样可能的位点
        n_grid = 5
        grid_points = []
        
        for i in range(n_grid):
            for j in range(n_grid):
                for k in range(n_grid):
                    frac = jnp.array([i/n_grid, j/n_grid, k/n_grid])
                    cart = frac @ cell
                    grid_points.append(cart)
        
        return jnp.array(grid_points)
    
    def calculate_migration_barrier(self,
                                   positions: jnp.ndarray,
                                   atomic_numbers: jnp.ndarray,
                                   cell: jnp.ndarray,
                                   path: Dict) -> float:
        """
        计算迁移势垒
        
        使用NEB (Nudged Elastic Band) 或简化的鞍点近似
        
        Args:
            path: 迁移路径信息
            
        Returns:
            迁移势垒 (eV)
        """
        # 简化模型: 势垒与通道几何相关
        
        # 通道半径 (近似)
        channel_radius = self._estimate_channel_radius(
            positions, atomic_numbers, path['midpoint'], cell
        )
        
        # 离子半径
        r_ion = self.ion_radii.get(ion_type, 1.0)
        
        # 几何因子: 通道越窄，势垒越高
        size_mismatch = jnp.maximum(0, (r_ion - channel_radius) / r_ion)
        geometric_barrier = 0.5 * size_mismatch**2
        
        # 静电因子 (简化)
        electrostatic_barrier = 0.1  # 基础值
        
        total_barrier = geometric_barrier + electrostatic_barrier
        
        return float(total_barrier)
    
    def _estimate_channel_radius(self,
                                 positions: jnp.ndarray,
                                 atomic_numbers: jnp.ndarray,
                                 point: jnp.ndarray,
                                 cell: jnp.ndarray) -> float:
        """估计迁移通道半径"""
        # 计算点到最近原子的距离
        dr = positions - point[None, :]
        
        # 周期性边界条件
        frac_dr = dr @ jnp.linalg.inv(cell)
        frac_dr = frac_dr - jnp.rint(frac_dr)
        dr = frac_dr @ cell
        
        distances = jnp.linalg.norm(dr, axis=1)
        min_distance = jnp.min(distances)
        
        # 通道半径约为到最近原子距离减去其半径
        min_radius = min_distance - 1.0  # 假设原子半径约1 Angstrom
        
        return float(jnp.maximum(min_radius, 0.1))
    
    def conductivity_from_barriers(self,
                                  barriers: jnp.ndarray,
                                  temperature: float = 300.0) -> float:
        """
        从迁移势垒计算离子电导率
        
        使用Arrhenius关系: σ = σ0 * exp(-Ea/kT)
        
        Args:
            barriers: 迁移势垒数组 (eV)
            temperature: 温度 (K)
            
        Returns:
            离子电导率 (S/cm)
        """
        kT = 8.617e-5 * temperature  # eV
        
        # 使用最低势垒 (速率决定步骤)
        Ea = jnp.min(barriers)
        
        # 指前因子 (简化)
        # σ0 ~ n * q^2 * D0 / (kT)
        sigma_0 = 1e3  # S/cm (典型值)
        
        conductivity = sigma_0 * jnp.exp(-Ea / kT)
        
        return float(conductivity)
    
    def percolation_analysis(self,
                            positions: jnp.ndarray,
                            atomic_numbers: jnp.ndarray,
                            cell: jnp.ndarray,
                            ion_type: str,
                            max_barrier: float = 0.5) -> Dict[str, Any]:
        """
        渗流分析
        
        确定离子是否能够形成连通的三维迁移网络
        
        Returns:
            {
                'percolates': 是否渗流,
                'n_clusters': 连通团簇数,
                'max_cluster_size': 最大团簇大小,
                'dimensionality': 渗流维度 (1D, 2D, 3D)
            }
        """
        paths = self.find_migration_paths(positions, atomic_numbers, cell, ion_type)
        
        # 筛选低势垒路径
        low_barrier_paths = []
        for path in paths:
            barrier = self.calculate_migration_barrier(
                positions, atomic_numbers, cell, path
            )
            if barrier < max_barrier:
                low_barrier_paths.append(path)
        
        # 构建连通图 (简化分析)
        n_paths = len(low_barrier_paths)
        
        # 检查连通性
        if n_paths > 10:  # 足够多的路径
            percolates = True
            dimensionality = '3D'
        elif n_paths > 5:
            percolates = True
            dimensionality = '2D'
        elif n_paths > 2:
            percolates = True
            dimensionality = '1D'
        else:
            percolates = False
            dimensionality = 'none'
        
        return {
            'percolates': percolates,
            'n_low_barrier_paths': n_paths,
            'dimensionality': dimensionality
        }


class SolidElectrolyteDesigner:
    """
    固态电解质设计师
    
    设计高离子电导率的固态电解质材料
    """
    
    def __init__(self, dft_engine: Any):
        self.dft = dft_engine
        self.migration_analyzer = IonMigrationAnalyzer(dft_engine)
    
    def calculate_ionic_conductivity(self,
                                    positions: jnp.ndarray,
                                    atomic_numbers: jnp.ndarray,
                                    cell: jnp.ndarray,
                                    target: IonConductorTarget) -> Dict[str, float]:
        """
        计算离子电导率
        
        Returns:
            {
                'ionic_conductivity': S/cm,
                'activation_energy': eV,
                'migration_barrier_min': eV,
                'migration_barrier_max': eV,
                'percolation_dimensionality': str
            }
        """
        # 寻找迁移路径
        paths = self.migration_analyzer.find_migration_paths(
            positions, atomic_numbers, cell, target.ion_type
        )
        
        if len(paths) == 0:
            return {
                'ionic_conductivity': 0.0,
                'activation_energy': 10.0,
                'migration_barrier_min': 10.0,
                'migration_barrier_max': 10.0,
                'percolation_dimensionality': 'none'
            }
        
        # 计算所有路径的势垒
        barriers = []
        for path in paths:
            barrier = self.migration_analyzer.calculate_migration_barrier(
                positions, atomic_numbers, cell, path
            )
            barriers.append(barrier)
        
        barriers = jnp.array(barriers)
        
        # 渗流分析
        percolation = self.migration_analyzer.percolation_analysis(
            positions, atomic_numbers, cell, target.ion_type
        )
        
        # 计算电导率
        if percolation['percolates']:
            conductivity = self.migration_analyzer.conductivity_from_barriers(
                barriers, target.temperature
            )
        else:
            conductivity = 0.0
        
        return {
            'ionic_conductivity': float(conductivity),
            'activation_energy': float(jnp.min(barriers)),
            'migration_barrier_min': float(jnp.min(barriers)),
            'migration_barrier_max': float(jnp.max(barriers)),
            'percolation_dimensionality': percolation['dimensionality']
        }
    
    def evaluate_stability(self,
                          positions: jnp.ndarray,
                          atomic_numbers: jnp.ndarray,
                          cell: jnp.ndarray) -> Dict[str, float]:
        """
        评估电化学稳定性
        
        Returns:
            {
                'voltage_window_min': V,
                'voltage_window_max': V,
                'stable': bool
            }
        """
        # 简化模型
        # 实际计算需要分解能和电压分析
        
        # 假设的稳定性窗口
        voltage_min = 0.0
        voltage_max = 4.5
        
        return {
            'voltage_window_min': voltage_min,
            'voltage_window_max': voltage_max,
            'stable': True
        }
    
    def design_lithium_conductor(self,
                                 initial_structure: ParameterizedStructure,
                                 min_conductivity: float = 1e-4) -> ParameterizedStructure:
        """
        设计锂导体
        
        目标材料类型：
        - LLZO (Li7La3Zr2O12)
        - LGPS (Li10GeP2S12)
        - NASICON-type
        - Argyrodite
        
        Args:
            initial_structure: 初始结构
            min_conductivity: 最小电导率要求 (S/cm)
            
        Returns:
            优化后的结构
        """
        target = IonConductorTarget(
            ion_type='Li',
            target_conductivity=min_conductivity,
            temperature=300.0
        )
        
        def objective_fn(params, structure):
            structure.set_params(params)
            pos, nums, cell = structure.to_structure()
            return self._ion_conductor_loss(pos, nums, cell, target)
        
        optimizer = InverseDesignOptimizer(
            None, 'adam', 0.01, 500
        )
        optimizer.objective = objective_fn
        
        return optimizer.optimize(initial_structure)
    
    def _ion_conductor_loss(self,
                           positions: jnp.ndarray,
                           atomic_numbers: jnp.ndarray,
                           cell: jnp.ndarray,
                           target: IonConductorTarget) -> float:
        """
        离子导体目标函数
        
        最小化：-log(σ) + w1*max(0, Ea-Ea_max) + w2*stability_penalty
        """
        # 计算电导率
        props = self.calculate_ionic_conductivity(
            positions, atomic_numbers, cell, target
        )
        
        sigma = props['ionic_conductivity']
        Ea = props['activation_energy']
        
        # 电导率损失 (鼓励高电导率)
        if sigma > 0:
            conductivity_loss = -jnp.log(sigma / target.target_conductivity)
        else:
            conductivity_loss = 100.0  # 惩罚零电导率
        
        # 迁移势垒约束
        barrier_penalty = jnp.maximum(0, Ea - target.min_migration_barrier)**2
        
        # 渗流约束
        percolation = self.migration_analyzer.percolation_analysis(
            positions, atomic_numbers, cell, target.ion_type
        )
        percolation_penalty = 0.0 if percolation['percolates'] else 10.0
        
        # 稳定性损失
        stability = self.evaluate_stability(positions, atomic_numbers, cell)
        stability_penalty = 0.0 if stability['stable'] else 5.0
        
        # 结构稳定性 (避免原子重叠)
        structure_penalty = self._structure_stability(positions, cell)
        
        total_loss = (
            target.conductivity_weight * conductivity_loss +
            target.barrier_weight * barrier_penalty +
            percolation_penalty +
            target.stability_weight * stability_penalty +
            0.1 * structure_penalty
        )
        
        return total_loss
    
    def _structure_stability(self, positions: jnp.ndarray, cell: jnp.ndarray) -> float:
        """结构稳定性惩罚"""
        r_ij = positions[:, None, :] - positions[None, :, :]
        frac = r_ij @ jnp.linalg.inv(cell)
        frac = frac - jnp.rint(frac)
        r_ij = frac @ cell
        distances = jnp.linalg.norm(r_ij, axis=2)
        distances = jnp.where(distances < 1e-10, 1e10, distances)
        
        # 惩罚过近的原子
        penalty = jnp.sum(jnp.where(distances < 2.0, (2.0 - distances)**2, 0))
        return penalty


class NASICONDesigner:
    """
    NASICON结构设计师
    
    NASICON (Na Super Ionic Conductor) 是一类重要的钠离子导体
    化学式: Na1+xZr2SixP3-xO12
    """
    
    def __init__(self, dft_engine: Any):
        self.dft = dft_engine
        self.designer = SolidElectrolyteDesigner(dft_engine)
    
    def create_nasicon_template(self,
                                x_composition: float = 0.5,
                                lattice_param: float = 15.0) -> ParameterizedStructure:
        """
        创建NASICON结构模板
        
        Args:
            x_composition: Si掺杂量 (0-3)
            lattice_param: 晶格常数 (Bohr)
            
        Returns:
            NASICON结构模板
        """
        # NASICON的菱方晶胞
        cell = jnp.array([
            [lattice_param, 0, 0],
            [0, lattice_param, 0],
            [0, 0, lattice_param]
        ])
        
        # 简化结构: Zr, P/Si, O, Na位置
        # 实际NASICON结构更复杂
        n_atoms = 10
        atomic_numbers = jnp.array([
            40, 40,  # Zr
            15, 15, 15 - int(x_composition),  # P/Si混合
            8, 8, 8, 8, 8,  # O
            11  # Na (移动离子)
        ])
        
        structure = FractionalCoordinateStructure(
            n_atoms=n_atoms,
            atomic_numbers=atomic_numbers,
            initial_cell=cell,
            fix_cell=False
        )
        
        return structure
    
    def optimize_nasicon(self,
                        x_range: Tuple[float, float] = (0, 3),
                        n_samples: int = 5) -> List[Dict]:
        """
        优化NASICON组成
        
        扫描不同Si/P比例，找到最佳电导率
        
        Args:
            x_range: Si组成范围
            n_samples: 采样点数
            
        Returns:
            优化结果列表
        """
        results = []
        
        for x in np.linspace(x_range[0], x_range[1], n_samples):
            print(f"\n优化 NASICON: x = {x:.2f}")
            
            # 创建模板
            template = self.create_nasicon_template(x_composition=x)
            
            # 优化结构
            target = IonConductorTarget(
                ion_type='Na',
                target_conductivity=1e-3
            )
            
            def objective(params, struct):
                struct.set_params(params)
                pos, nums, cell = struct.to_structure()
                return self.designer._ion_conductor_loss(pos, nums, cell, target)
            
            optimizer = InverseDesignOptimizer(
                None, 'adam', 0.01, 300
            )
            optimizer.objective = objective
            
            result = optimizer.optimize(template)
            
            # 评估最终性质
            final_pos, final_nums, final_cell = result.to_structure()
            props = self.designer.calculate_ionic_conductivity(
                final_pos, final_nums, final_cell, target
            )
            
            results.append({
                'composition_x': x,
                'conductivity': props['ionic_conductivity'],
                'activation_energy': props['activation_energy'],
                'structure': result
            })
            
            print(f"  电导率: {props['ionic_conductivity']:.2e} S/cm")
            print(f"  活化能: {props['activation_energy']:.3f} eV")
        
        return results


class SulfideElectrolyteDesigner:
    """
    硫化物电解质设计师
    
    硫化物固态电解质 (如LGPS) 通常具有极高的离子电导率
    """
    
    def __init__(self, dft_engine: Any):
        self.dft = dft_engine
        self.designer = SolidElectrolyteDesigner(dft_engine)
    
    def design_lgps_variant(self,
                           initial_structure: ParameterizedStructure,
                           dopant: Optional[str] = None) -> ParameterizedStructure:
        """
        设计LGPS变体
        
        LGPS: Li10GeP2S12
        变体: 用Sn/Si替代Ge，调整P含量
        
        Args:
            initial_structure: 初始结构
            dopant: 掺杂元素 ('Sn', 'Si', 'Al')
            
        Returns:
            优化后的结构
        """
        target = IonConductorTarget(
            ion_type='Li',
            target_conductivity=1e-2,  # 10 mS/cm (高目标)
            temperature=300.0
        )
        
        # 硫化物通常有更低的迁移势垒
        target.min_migration_barrier = 0.15
        
        def objective(params, structure):
            structure.set_params(params)
            pos, nums, cell = structure.to_structure()
            
            # 基础损失
            loss = self.designer._ion_conductor_loss(pos, nums, cell, target)
            
            # 鼓励三维渗流 (硫化物的优势)
            percolation = self.designer.migration_analyzer.percolation_analysis(
                pos, nums, cell, 'Li'
            )
            if percolation['dimensionality'] == '3D':
                loss *= 0.9  # 奖励三维渗流
            
            return loss
        
        optimizer = InverseDesignOptimizer(
            None, 'adam', 0.01, 500
        )
        optimizer.objective = objective
        
        return optimizer.optimize(initial_structure)


def example_ion_conductor_design():
    """离子导体逆向设计示例"""
    print("=" * 60)
    print("离子导体逆向设计示例")
    print("=" * 60)
    
    # 创建模拟DFT引擎
    class MockDFTEngine:
        pass
    
    dft_engine = MockDFTEngine()
    
    # 示例1: 锂离子导体
    print("\n【示例1: 锂离子导体设计】")
    designer = SolidElectrolyteDesigner(dft_engine)
    
    # 创建简单结构
    structure = FractionalCoordinateStructure(
        n_atoms=4,
        atomic_numbers=jnp.array([3, 16, 16, 16]),  # LiS3-like
        initial_cell=jnp.eye(3) * 8.0,
        fix_cell=False
    )
    
    target = IonConductorTarget(
        ion_type='Li',
        target_conductivity=1e-3,
        temperature=300.0
    )
    
    # 计算初始性质
    init_pos, init_nums, init_cell = structure.to_structure()
    init_props = designer.calculate_ionic_conductivity(
        init_pos, init_nums, init_cell, target
    )
    
    print(f"初始电导率: {init_props['ionic_conductivity']:.2e} S/cm")
    print(f"初始活化能: {init_props['activation_energy']:.3f} eV")
    
    # 优化
    def objective(params, struct):
        struct.set_params(params)
        pos, nums, cell = struct.to_structure()
        return designer._ion_conductor_loss(pos, nums, cell, target)
    
    optimizer = InverseDesignOptimizer(None, 'adam', 0.05, 200)
    optimizer.objective = objective
    
    result = optimizer.optimize(structure)
    
    final_pos, final_nums, final_cell = result.to_structure()
    final_props = designer.calculate_ionic_conductivity(
        final_pos, final_nums, final_cell, target
    )
    
    print(f"\n优化后电导率: {final_props['ionic_conductivity']:.2e} S/cm")
    print(f"优化后活化能: {final_props['activation_energy']:.3f} eV")
    
    # 示例2: NASICON优化
    print("\n【示例2: NASICON钠离子导体】")
    nasicon_designer = NASICONDesigner(dft_engine)
    results = nasicon_designer.optimize_nasicon(x_range=(0, 2), n_samples=3)
    
    print("\nNASICON组成优化结果:")
    for r in results:
        print(f"  x={r['composition_x']:.1f}: σ={r['conductivity']:.2e}, Ea={r['activation_energy']:.3f}")
    
    # 示例3: 迁移分析
    print("\n【示例3: 离子迁移分析】")
    analyzer = IonMigrationAnalyzer(dft_engine)
    
    paths = analyzer.find_migration_paths(
        final_pos, final_nums, final_cell, 'Li'
    )
    print(f"找到 {len(paths)} 条迁移路径")
    
    if len(paths) > 0:
        percolation = analyzer.percolation_analysis(
            final_pos, final_nums, final_cell, 'Li'
        )
        print(f"渗流分析: {percolation}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    example_ion_conductor_design()
