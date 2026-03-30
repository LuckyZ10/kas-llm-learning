"""
MD Coupling Module
==================
分子动力学耦合模块

实现相场与MD模拟的耦合:
1. 从MD获取动力学参数 (扩散系数、关联函数)
2. 从MD获取迁移势垒
3. 传递原子构型到相场
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

try:
    from ase import Atoms
    from ase.io import read
    from ase.io.lammpsrun import read_lammps_dump_text
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False


@dataclass
class MDCouplingConfig:
    """MD耦合配置"""
    # MD引擎
    md_engine: str = "lammps"  # lammps, gromacs, openmm
    
    # 分析参数
    compute_msd: bool = True  # 计算均方位移
    compute_vacf: bool = True  # 计算速度自关联函数
    compute_rdf: bool = True  # 计算径向分布函数
    
    # 输运性质
    compute_diffusion: bool = True
    compute_viscosity: bool = False
    compute_thermal_conductivity: bool = False
    
    # 相变分析
    detect_phase_transition: bool = True
    melting_point_search: bool = False
    
    # 轨迹处理
    trajectory_interval: int = 100
    equilibration_steps: int = 10000


class MDCoupling:
    """
    MD耦合类
    
    管理相场模型与MD模拟之间的数据交换。
    """
    
    def __init__(self, config: Optional[MDCouplingConfig] = None):
        """
        初始化MD耦合
        
        Args:
            config: MD耦合配置
        """
        self.config = config or MDCouplingConfig()
        
        # 存储MD结果
        self.md_results = {}
        self.transport_params = {}
        self.structure_params = {}
        
        logger.info(f"MD coupling initialized ({self.config.md_engine})")
    
    def extract_from_trajectory(self, trajectory_file: str,
                                topology_file: Optional[str] = None) -> Dict:
        """
        从MD轨迹提取参数
        
        Args:
            trajectory_file: 轨迹文件路径
            topology_file: 拓扑文件路径 (可选)
            
        Returns:
            params: 提取的参数
        """
        params = {}
        
        if not ASE_AVAILABLE:
            logger.warning("ASE not available. Cannot parse trajectory.")
            return params
        
        try:
            # 读取LAMMPS轨迹
            if trajectory_file.endswith('.lammpstrj') or \
               trajectory_file.endswith('.dump'):
                with open(trajectory_file, 'r') as f:
                    atoms_list = read_lammps_dump_text(f, index=':')
                
                logger.info(f"Loaded {len(atoms_list)} frames from trajectory")
                
                # 分析轨迹
                if self.config.compute_msd:
                    msd_data = self._compute_msd(atoms_list)
                    params['msd'] = msd_data
                
                if self.config.compute_diffusion:
                    D = self._estimate_diffusion_coefficient(atoms_list)
                    params['diffusion_coefficient'] = D
                
                if self.config.compute_vacf:
                    vacf_data = self._compute_vacf(atoms_list)
                    params['vacf'] = vacf_data
                
                # 结构分析
                structure_data = self._analyze_structure(atoms_list)
                params['structure'] = structure_data
                
        except Exception as e:
            logger.error(f"Error extracting MD data: {e}")
        
        self.md_results = params
        return params
    
    def _compute_msd(self, atoms_list: List) -> Dict:
        """
        计算均方位移 (MSD)
        
        MSD(t) = <|r(t) - r(0)|²>
        
        Args:
            atoms_list: ASE Atoms对象列表
            
        Returns:
            msd_data: MSD数据
        """
        n_frames = len(atoms_list)
        
        # 获取参考位置 (第一帧)
        ref_positions = atoms_list[0].get_positions()
        
        # 计算MSD
        msd_values = []
        times = []
        
        dt = 1.0  # 假设时间步 (需要从输入获取)
        
        for i, atoms in enumerate(atoms_list):
            positions = atoms.get_positions()
            
            # 计算位移 (考虑周期性边界)
            displacement = positions - ref_positions
            
            # 均方位移
            msd = np.mean(np.sum(displacement**2, axis=1))
            msd_values.append(msd)
            times.append(i * dt)
        
        msd_data = {
            'times': np.array(times),
            'msd': np.array(msd_values)
        }
        
        return msd_data
    
    def _estimate_diffusion_coefficient(self, atoms_list: List) -> Dict:
        """
        从MSD估计扩散系数
        
        D = MSD / (6t) for 3D
        
        Args:
            atoms_list: ASE Atoms对象列表
            
        Returns:
            D_data: 扩散系数数据
        """
        msd_data = self._compute_msd(atoms_list)
        times = msd_data['times']
        msd = msd_data['msd']
        
        # 线性拟合MSD-t曲线
        # 使用后半部分数据 (扩散区域)
        start_idx = len(times) // 3
        
        times_fit = times[start_idx:]
        msd_fit = msd[start_idx:]
        
        # 线性拟合
        slope, intercept = np.polyfit(times_fit, msd_fit, deg=1)
        
        # D = slope / (2*d) where d is dimensionality
        ndim = 3  # 假设3D
        D = slope / (2 * ndim)
        
        # 单位转换 (Å²/fs -> m²/s)
        D_m2s = D * 1e-20 / 1e-15  # Å²/fs -> m²/s
        
        D_data = {
            'value': D_m2s,
            'slope': slope,
            'fit_quality': np.corrcoef(times_fit, msd_fit)[0, 1]**2,
            'unit': 'm²/s'
        }
        
        self.transport_params['diffusion_coefficient'] = D_data
        
        logger.info(f"Estimated D = {D_m2s:.2e} m²/s (R²={D_data['fit_quality']:.4f})")
        
        return D_data
    
    def _compute_vacf(self, atoms_list: List, max_lag: Optional[int] = None) -> Dict:
        """
        计算速度自关联函数 (VACF)
        
        VACF(t) = <v(t)·v(0)>
        
        Args:
            atoms_list: ASE Atoms对象列表
            max_lag: 最大滞后步数
            
        Returns:
            vacf_data: VACF数据
        """
        n_frames = len(atoms_list)
        max_lag = max_lag or n_frames // 4
        
        # 提取速度
        velocities = np.array([atoms.get_velocities() for atoms in atoms_list])
        n_atoms = velocities.shape[1]
        
        # 计算VACF
        vacf = np.zeros(max_lag)
        
        for lag in range(max_lag):
            corr = 0.0
            count = 0
            for t in range(n_frames - lag):
                v_t = velocities[t]
                v_t_lag = velocities[t + lag]
                corr += np.sum(v_t * v_t_lag) / n_atoms
                count += 1
            vacf[lag] = corr / count
        
        # 归一化
        vacf = vacf / vacf[0]
        
        vacf_data = {
            'lags': np.arange(max_lag),
            'vacf': vacf,
            'integral': np.trapezoid(vacf)  # 用于计算扩散系数
        }
        
        return vacf_data
    
    def _analyze_structure(self, atoms_list: List) -> Dict:
        """
        分析MD轨迹的结构特征
        
        Args:
            atoms_list: ASE Atoms对象列表
            
        Returns:
            structure_data: 结构数据
        """
        structure_data = {}
        
        # 密度分析
        densities = []
        for atoms in atoms_list:
            volume = atoms.get_volume()
            mass = np.sum(atoms.get_masses())
            density = mass / volume * 1.66054  # 转换为g/cm³
            densities.append(density)
        
        structure_data['density_mean'] = np.mean(densities)
        structure_data['density_std'] = np.std(densities)
        
        # 配位数分析 (简化)
        # 可以使用更复杂的算法如Voronoi分析
        
        # 径向分布函数 (RDF)
        if self.config.compute_rdf:
            rdf_data = self._compute_rdf(atoms_list[-1])
            structure_data['rdf'] = rdf_data
        
        self.structure_params = structure_data
        
        return structure_data
    
    def _compute_rdf(self, atoms, nbins: int = 100, rmax: Optional[float] = None) -> Dict:
        """
        计算径向分布函数
        
        Args:
            atoms: ASE Atoms对象
            nbins: RDF分箱数
            rmax: 最大距离
            
        Returns:
            rdf_data: RDF数据
        """
        from ase.geometry.analysis import Analysis
        
        if rmax is None:
            rmax = atoms.get_cell(). lengths().min() / 2
        
        # 使用ASE的RDF分析
        ana = Analysis(atoms)
        rdf = ana.get_rdf(rmax, nbins)
        
        r_bins = np.linspace(0, rmax, nbins)
        
        rdf_data = {
            'r': r_bins,
            'g_r': rdf[0][0] if rdf else np.zeros(nbins)
        }
        
        return rdf_data
    
    def extract_migration_barrier(self, 
                                   nudged_elastic_band_data: Dict) -> Dict:
        """
        从NEB计算提取迁移势垒
        
        Args:
            nudged_elastic_band_data: NEB计算结果
            
        Returns:
            barrier_data: 势垒数据
        """
        # 提取能量路径
        reaction_coord = nudged_elastic_band_data.get('reaction_coordinate', [])
        energies = nudged_elastic_band_data.get('energies', [])
        
        if len(energies) < 2:
            return {'error': 'Insufficient NEB data'}
        
        # 找到能垒
        e_max = max(energies)
        e_initial = energies[0]
        e_final = energies[-1]
        
        barrier_forward = e_max - e_initial
        barrier_reverse = e_max - e_final
        
        barrier_data = {
            'barrier_forward': barrier_forward,  # eV
            'barrier_reverse': barrier_reverse,
            'saddle_point_energy': e_max,
            'reaction_energy': e_final - e_initial,
            'reaction_coordinate': reaction_coord,
            'energy_path': energies
        }
        
        self.transport_params['migration_barrier'] = barrier_data
        
        logger.info(f"Extracted barrier: {barrier_forward:.3f} eV")
        
        return barrier_data
    
    def compute_time_correlation_function(self, 
                                          property_trajectory: np.ndarray,
                                          max_lag: Optional[int] = None) -> Dict:
        """
        计算通用时间关联函数
        
        C(t) = <A(t)A(0)>
        
        Args:
            property_trajectory: 性质随时间的变化
            max_lag: 最大滞后步数
            
        Returns:
            correlation_data: 关联函数数据
        """
        n_frames = len(property_trajectory)
        max_lag = max_lag or n_frames // 4
        
        # 中心化
        prop_centered = property_trajectory - property_trajectory.mean()
        
        # 计算关联函数
        correlation = np.zeros(max_lag)
        
        for lag in range(max_lag):
            if lag == 0:
                correlation[lag] = np.mean(prop_centered**2)
            else:
                correlation[lag] = np.mean(prop_centered[:-lag] * prop_centered[lag:])
        
        # 归一化
        correlation = correlation / correlation[0]
        
        # 计算关联时间
        correlation_time = np.trapezoid(correlation)
        
        correlation_data = {
            'lags': np.arange(max_lag),
            'correlation': correlation,
            'correlation_time': correlation_time
        }
        
        return correlation_data
    
    def generate_phase_field_params(self) -> Dict:
        """
        生成相场模型参数
        
        Returns:
            pf_params: 相场参数字典
        """
        params = {
            'transport': self.transport_params,
            'structure': self.structure_params
        }
        
        # 提取关键参数
        if 'diffusion_coefficient' in self.transport_params:
            D = self.transport_params['diffusion_coefficient']['value']
            params['M'] = D  # 迁移率 (假设爱因斯坦关系)
        
        if 'migration_barrier' in self.transport_params:
            barrier = self.transport_params['migration_barrier']['barrier_forward']
            params['activation_energy'] = barrier
        
        return params
    
    def get_atomistic_configuration(self, phase_field_concentration: np.ndarray,
                                    lattice_param: float = 4.0) -> 'Atoms':
        """
        将相场浓度场转换为原子构型
        
        Args:
            phase_field_concentration: 相场浓度场
            lattice_param: 晶格常数
            
        Returns:
            atoms: ASE Atoms对象
        """
        if not ASE_AVAILABLE:
            raise ImportError("ASE required for atomistic conversion")
        
        # 根据浓度场生成原子位置
        # 简化：在相场网格点上放置原子
        
        nx, ny = phase_field_concentration.shape[:2]
        
        positions = []
        symbols = []
        
        for i in range(nx):
            for j in range(ny):
                # 网格点坐标
                x = i * lattice_param
                y = j * lattice_param
                z = 0
                
                # 根据浓度决定原子类型
                c = phase_field_concentration[i, j]
                
                # 概率性放置原子
                if np.random.random() < c:
                    positions.append([x, y, z])
                    symbols.append('Li')
                else:
                    positions.append([x, y, z])
                    symbols.append('Co')
        
        # 创建晶胞
        cell = [nx * lattice_param, ny * lattice_param, lattice_param]
        
        atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
        
        return atoms
