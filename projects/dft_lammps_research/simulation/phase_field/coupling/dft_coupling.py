"""
DFT Coupling Module
===================
DFT耦合模块

实现相场模型与DFT计算的双向耦合:
1. 从DFT获取热力学参数 (化学势、扩散系数、弹性常数等)
2. 从DFT获取界面能、缺陷能
3. 将相场结果反馈给DFT进行验证
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# 尝试导入ASE
try:
    from ase import Atoms
    from ase.io import read, write
    from ase.calculators.vasp import Vasp
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    logger.warning("ASE not available. DFT coupling limited.")

# 尝试导入pymatgen
try:
    from pymatgen.core import Structure, Composition
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.entries.computed_entries import ComputedEntry
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False


@dataclass
class DFTCouplingConfig:
    """DFT耦合配置"""
    # DFT计算参数
    dft_code: str = "vasp"  # vasp, quantum_espresso, abacus
    dft_params: Dict[str, Any] = field(default_factory=dict)
    
    # 计算任务
    compute_bulk_energy: bool = True
    compute_interface_energy: bool = True
    compute_diffusion_barrier: bool = True
    compute_elastic_constants: bool = True
    
    # 界面结构
    interface_angles: List[float] = field(default_factory=lambda: [0, 15, 30, 45])
    interface_thickness: float = 1.0  # nm
    
    # 参数提取
    extract_thermodynamic_params: bool = True
    extract_kinetic_params: bool = True
    
    # 自动化
    auto_submit_jobs: bool = False
    check_job_status: bool = True
    
    # 数据库
    use_materials_project: bool = False
    mp_api_key: Optional[str] = None


class DFTCoupling:
    """
    DFT耦合类
    
    管理相场模型与DFT计算之间的数据交换。
    """
    
    def __init__(self, config: Optional[DFTCouplingConfig] = None):
        """
        初始化DFT耦合
        
        Args:
            config: DFT耦合配置
        """
        self.config = config or DFTCouplingConfig()
        
        # 存储DFT结果
        self.dft_results = {}
        self.thermodynamic_params = {}
        self.kinetic_params = {}
        
        # 工作目录
        self.work_dir = Path("./dft_coupling")
        self.work_dir.mkdir(exist_ok=True)
        
        logger.info(f"DFT coupling initialized ({self.config.dft_code})")
    
    def extract_from_dft_output(self, dft_output_path: str) -> Dict:
        """
        从DFT输出文件提取参数
        
        Args:
            dft_output_path: DFT输出文件/目录路径
            
        Returns:
            params: 提取的参数
        """
        dft_path = Path(dft_output_path)
        
        if self.config.dft_code == "vasp":
            return self._extract_from_vasp(dft_path)
        elif self.config.dft_code == "quantum_espresso":
            return self._extract_from_qe(dft_path)
        else:
            raise ValueError(f"Unsupported DFT code: {self.config.dft_code}")
    
    def _extract_from_vasp(self, vasp_dir: Path) -> Dict:
        """从VASP输出提取参数"""
        params = {}
        
        if ASE_AVAILABLE:
            try:
                # 读取OUTCAR
                outcar_path = vasp_dir / "OUTCAR"
                if outcar_path.exists():
                    atoms = read(str(outcar_path), format='vasp-out')
                    
                    # 提取能量
                    params['total_energy'] = float(atoms.get_potential_energy())
                    params['energy_per_atom'] = params['total_energy'] / len(atoms)
                    
                    # 提取力
                    forces = atoms.get_forces()
                    params['max_force'] = float(np.max(np.abs(forces)))
                    
                    # 提取应力
                    stress = atoms.get_stress()
                    params['stress'] = stress.tolist()
                    
                    logger.info(f"Extracted from VASP: E={params['energy_per_atom']:.4f} eV/atom")
                
                # 读取vasprun.xml获取更多信息
                vasprun_path = vasp_dir / "vasprun.xml"
                if vasprun_path.exists() and PYMATGEN_AVAILABLE:
                    from pymatgen.io.vasp import Vasprun
                    vasprun = Vasprun(str(vasprun_path))
                    
                    # 提取电子结构信息
                    params['band_gap'] = vasprun.eigenvalue_band_properties[0]
                    params['cbm'] = vasprun.eigenvalue_band_properties[1]
                    params['vbm'] = vasprun.eigenvalue_band_properties[2]
                    
            except Exception as e:
                logger.error(f"Error extracting VASP data: {e}")
        
        self.dft_results['bulk'] = params
        return params
    
    def _extract_from_qe(self, qe_dir: Path) -> Dict:
        """从Quantum ESPRESSO输出提取参数"""
        params = {}
        
        if ASE_AVAILABLE:
            try:
                # 读取QE输出
                for output_file in qe_dir.glob("*.out"):
                    atoms = read(str(output_file), format='espresso-out')
                    params['total_energy'] = float(atoms.get_potential_energy())
                    params['energy_per_atom'] = params['total_energy'] / len(atoms)
                    break
            except Exception as e:
                logger.error(f"Error extracting QE data: {e}")
        
        self.dft_results['bulk'] = params
        return params
    
    def compute_chemical_potential(self, compositions: List[Dict[str, float]],
                                   dft_results: List[Dict]) -> Callable:
        """
        计算成分依赖的化学势
        
        从DFT结果拟合化学势函数 μ(c)
        
        Args:
            compositions: 成分列表
            dft_results: 对应DFT结果
            
        Returns:
            mu_func: 化学势函数
        """
        # 提取能量数据
        c_values = []
        energies = []
        
        for comp, result in zip(compositions, dft_results):
            # 假设comp包含Li成分
            c_li = comp.get('Li', 0.5)
            c_values.append(c_li)
            energies.append(result.get('energy_per_atom', 0))
        
        c_values = np.array(c_values)
        energies = np.array(energies)
        
        # 多项式拟合自由能 f(c)
        coeffs = np.polyfit(c_values, energies, deg=4)
        
        # 化学势是自由能的导数
        mu_coeffs = np.polyder(coeffs)
        
        # 创建化学势函数
        def mu_func(c):
            return np.polyval(mu_coeffs, c)
        
        self.thermodynamic_params['chemical_potential'] = {
            'function': mu_func,
            'coefficients': coeffs.tolist()
        }
        
        logger.info(f"Computed chemical potential function from {len(c_values)} DFT points")
        
        return mu_func
    
    def compute_diffusion_coefficient(self, 
                                      barrier: float,
                                      temperature: float = 298.15,
                                      prefactor: float = 1e13) -> float:
        """
        计算扩散系数
        
        使用Arrhenius公式: D = D0 * exp(-Ea/kT)
        
        Args:
            barrier: 扩散势垒 (eV)
            temperature: 温度 (K)
            prefactor: 前置因子 (Hz)
            
        Returns:
            D: 扩散系数 (m²/s)
        """
        kB = 8.617333e-5  # eV/K
        D0 = prefactor * (1e-10)**2  # 假设跳跃距离1Å
        
        D = D0 * np.exp(-barrier / (kB * temperature))
        
        self.kinetic_params['diffusion_coefficient'] = {
            'value': D,
            'barrier': barrier,
            'temperature': temperature
        }
        
        logger.info(f"Computed D={D:.2e} m²/s (Ea={barrier:.3f} eV)")
        
        return D
    
    def compute_gradient_energy_coefficient(self, 
                                           interface_energy: float,
                                           interface_width: float) -> float:
        """
        计算梯度能量系数 κ
        
        关系: γ = (1/3) * sqrt(2*κ*A) 
        => κ = (3γ)² / (2A)
        
        Args:
            interface_energy: 界面能 (J/m²)
            interface_width: 界面宽度 (m)
            
        Returns:
            kappa: 梯度能量系数 (J/m)
        """
        # 简化模型
        kappa = interface_energy * interface_width / 2
        
        self.thermodynamic_params['gradient_energy_coefficient'] = kappa
        
        return kappa
    
    def compute_elastic_constants(self, dft_stress_strain: List[Dict]) -> np.ndarray:
        """
        计算弹性常数张量
        
        Args:
            dft_stress_strain: DFT计算的应力-应变数据
            
        Returns:
            C: 弹性常数矩阵 (6x6)
        """
        # 从DFT数据拟合弹性常数
        # 简化：假设各向同性
        
        # 提取数据
        strains = np.array([d['strain'] for d in dft_stress_strain])
        stresses = np.array([d['stress'] for d in dft_stress_strain])
        
        # 线性拟合
        # 对于各向同性材料: E = stress/strain
        E_estimate = np.mean(stresses / strains)
        
        # 构建各向同性弹性矩阵
        nu = 0.25  # 假设泊松比
        
        C = np.zeros((6, 6))
        lambda_lame = E_estimate * nu / ((1 + nu) * (1 - 2 * nu))
        mu_lame = E_estimate / (2 * (1 + nu))
        
        # 各向同性弹性张量
        for i in range(3):
            for j in range(3):
                C[i, j] = lambda_lame
            C[i, i] += 2 * mu_lame
        
        for i in range(3, 6):
            C[i, i] = mu_lame
        
        self.thermodynamic_params['elastic_constants'] = C
        
        logger.info(f"Computed elastic constants: E={E_estimate/1e9:.2f} GPa")
        
        return C
    
    def extract_interface_energy(self, slab_calculations: List[Dict]) -> Dict[str, float]:
        """
        提取界面能
        
        Args:
            slab_calculations: 不同晶面DFT计算结果
            
        Returns:
            gamma: 界面能字典
        """
        interface_energies = {}
        
        for calc in slab_calculations:
            miller = calc.get('miller_index', (1, 0, 0))
            e_surf = calc.get('surface_energy', 0)
            interface_energies[str(miller)] = e_surf
        
        self.thermodynamic_params['interface_energy'] = interface_energies
        
        return interface_energies
    
    def generate_phase_field_params(self) -> Dict:
        """
        生成相场模型参数
        
        整合所有DFT结果，生成可直接用于相场模拟的参数
        
        Returns:
            pf_params: 相场参数字典
        """
        params = {
            'thermodynamic': self.thermodynamic_params,
            'kinetic': self.kinetic_params,
            'dft_metadata': {
                'code': self.config.dft_code,
                'n_calculations': len(self.dft_results)
            }
        }
        
        # 提取关键参数
        if 'diffusion_coefficient' in self.kinetic_params:
            params['M'] = self.kinetic_params['diffusion_coefficient']['value']
        
        if 'gradient_energy_coefficient' in self.thermodynamic_params:
            params['kappa'] = self.thermodynamic_params['gradient_energy_coefficient']
        
        if 'elastic_constants' in self.thermodynamic_params:
            C = self.thermodynamic_params['elastic_constants']
            params['elastic'] = {
                'C11': float(C[0, 0]),
                'C12': float(C[0, 1]),
                'C44': float(C[3, 3])
            }
        
        # 保存到文件
        output_file = self.work_dir / "phase_field_params.json"
        with open(output_file, 'w') as f:
            # 转换函数为字符串
            params_save = params.copy()
            if 'chemical_potential' in self.thermodynamic_params:
                params_save['thermodynamic']['chemical_potential'] = 'Function'
            json.dump(params_save, f, indent=2)
        
        logger.info(f"Generated phase field parameters: {output_file}")
        
        return params
    
    def feedback_to_dft(self, phase_field_result: Dict,
                       output_dir: str = "./dft_input") -> List[Path]:
        """
        将相场结果反馈给DFT
        
        从相场结果提取关键结构，生成DFT输入
        
        Args:
            phase_field_result: 相场模拟结果
            output_dir: DFT输入输出目录
            
        Returns:
            dft_inputs: 生成的DFT输入文件列表
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        dft_inputs = []
        
        # 提取界面结构
        if 'interface_structure' in phase_field_result:
            interface = phase_field_result['interface_structure']
            if ASE_AVAILABLE:
                atoms = Atoms(
                    symbols=interface['symbols'],
                    positions=interface['positions'],
                    cell=interface['cell'],
                    pbc=True
                )
                poscar_path = output_path / "POSCAR_interface"
                write(str(poscar_path), atoms, format='vasp')
                dft_inputs.append(poscar_path)
        
        # 提取缺陷结构
        if 'defect_structures' in phase_field_result:
            for i, defect in enumerate(phase_field_result['defect_structures']):
                if ASE_AVAILABLE:
                    atoms = Atoms(
                        symbols=defect['symbols'],
                        positions=defect['positions'],
                        cell=defect['cell'],
                        pbc=True
                    )
                    poscar_path = output_path / f"POSCAR_defect_{i}"
                    write(str(poscar_path), atoms, format='vasp')
                    dft_inputs.append(poscar_path)
        
        logger.info(f"Generated {len(dft_inputs)} DFT input structures")
        
        return dft_inputs
    
    def validate_parameters(self, experimental_data: Optional[Dict] = None) -> Dict:
        """
        验证DFT提取的参数
        
        与实验数据或文献值对比
        
        Args:
            experimental_data: 实验参数字典 (可选)
            
        Returns:
            validation: 验证结果
        """
        validation = {
            'status': 'ok',
            'warnings': [],
            'errors': []
        }
        
        # 检查扩散系数合理性
        if 'diffusion_coefficient' in self.kinetic_params:
            D = self.kinetic_params['diffusion_coefficient']['value']
            if D < 1e-20 or D > 1e-8:
                validation['warnings'].append(
                    f"Diffusion coefficient {D:.2e} m²/s seems unusual"
                )
        
        # 与实验值对比
        if experimental_data:
            for key, exp_value in experimental_data.items():
                if key in self.thermodynamic_params:
                    calc_value = self.thermodynamic_params[key]
                    if isinstance(calc_value, (int, float)):
                        error = abs(calc_value - exp_value) / exp_value * 100
                        validation[key] = {
                            'calculated': calc_value,
                            'experimental': exp_value,
                            'error_percent': error
                        }
        
        return validation
