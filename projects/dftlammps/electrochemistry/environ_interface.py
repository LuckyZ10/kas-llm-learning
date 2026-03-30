"""
QE Environ Interface Module - Quantum ESPRESSO连续介质模型接口

本模块提供Quantum ESPRESSO Environ插件的Python接口，支持：
- 隐式溶剂模型计算
- 介电环境设置
- 电解液参数配置
- 电解屏蔽效应
- 极化连续介质模型(PCM)

Reference:
    - Andreussi et al., J. Chem. Phys. 2012, 136, 064102
    - Andreussi et al., J. Chem. Phys. 2014, 141, 034101

Author: DFT-LAMMPS Team
Date: 2025
"""

import os
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import subprocess
import warnings
from enum import Enum, auto


class EnvironType(Enum):
    """Environ模型类型"""
    NONE = "none"
    INPUT = "input"           # 从输入读取介电函数
    VACUUM = "vacuum"         # 真空环境
    WATER = "water"           # 水环境
    EXTERNAL = "external"     # 外部定义


class DerivativeMethod(Enum):
    """空间导数计算方法"""
    FFT = "fft"               # FFT方法
    DCT = "ct"                # 余弦变换方法
    FD = "fd"                 # 有限差分


class SolventBoundary(Enum):
    """溶剂边界类型"""
    SAS = "solventaccessible"  # 溶剂可及表面
    VDW = "vdw"                # van der Waals表面
    IONIC = "ionic"            # 离子表面


@dataclass
class EnvironParameters:
    """
    Environ基础参数
    
    Attributes:
        environ_type: 环境类型
        env_static_permittivity: 静态介电常数
        env_optical_permittivity: 光学介电常数
        env_surface_tension: 表面张力（Ry/bohr²）
        env_pressure: 外部压力（Ry/bohr³）
        env_periodicity: 周期性设置 (3D/2D/1D/0D)
    """
    # 基本参数
    environ_type: str = "input"
    env_static_permittivity: float = 1.0    # 静态介电常数
    env_optical_permittivity: float = 1.0   # 光学介电常数
    
    # 溶剂化参数
    env_surface_tension: float = 0.0        # 表面张力 (Ry/bohr²)
    env_pressure: float = 0.0               # 压力 (Ry/bohr³)
    env_temperature: float = 300.0          # 温度 (K)
    
    # 边界条件
    env_periodicity: str = "3D"             # 3D/2D/1D/0D
    env_axis: int = 3                       # 非周期性方向 (3=Z)
    
    # 介电函数参数
    env_epsilon_mode: str = "linear"        # linear/sigmoid/tanh
    env_epsilon_function: str = "rho"       # 基于密度的介电函数
    env_epsilon_min: float = 1.0            # 最小介电常数
    env_epsilon_max: float = 78.4           # 最大介电常数
    
    # 腔体参数
    env_boundary: str = "solventaccessible"  # 边界类型
    env_solvent_radius: float = 1.0          # 溶剂分子半径 (bohr)
    env_probe_radius: float = 1.4            # 探针半径 (bohr)
    
    # 数值参数
    env_derivative_method: str = "fft"       # 导数计算方法
    env_diagonalization: str = "cg"          # 对角化方法
    env_cg_tolerance: float = 1e-8           # CG收敛阈值
    env_maxstep: int = 1000                  # 最大迭代步
    
    def to_namelist_dict(self) -> Dict[str, Any]:
        """转换为Fortran namelist字典"""
        return {
            "environ_type": f"'{self.environ_type}'",
            "env_static_permittivity": self.env_static_permittivity,
            "env_optical_permittivity": self.env_optical_permittivity,
            "env_surface_tension": self.env_surface_tension,
            "env_pressure": self.env_pressure,
            "env_temperature": self.env_temperature,
            "env_periodicity": f"'{self.env_periodicity}'",
            "env_axis": self.env_axis,
            "env_epsilon_mode": f"'{self.env_epsilon_mode}'",
        }
    
    def to_input_cards(self) -> str:
        """生成Environ输入卡"""
        cards = ["\u0026ENVIRON"]
        for key, value in self.to_namelist_dict().items():
            if isinstance(value, str) and value.startswith("'"):
                cards.append(f"    {key} = {value}")
            else:
                cards.append(f"    {key} = {value}")
        cards.append("\u0026END")
        return "\n".join(cards)


@dataclass
class EnvironBoundaryParameters:
    """
    Environ边界参数
    
    控制溶剂腔体的形状和性质
    """
    # 边界类型
    solvent_mode: str = "ionic"              # electronic/ionic/pseudo
    
    # 电子密度参数
    rhomax: float = 0.005                    # 最大密度阈值
    rhomin: float = 0.0001                   # 最小密度阈值
    
    # 腔体参数
    alpha: float = 1.0                       # 腔体参数α
    beta: float = 1.0                        # 腔体参数β
    softness: float = 0.5                    # 边界软化参数
    
    # 溶剂参数
    solvent_radius: float = 1.0              # 溶剂半径 (bohr)
    radial_scale: float = 2.0                # 径向缩放
    
    def to_namelist_dict(self) -> Dict[str, Any]:
        """转换为namelist字典"""
        return {
            "solvent_mode": f"'{self.solvent_mode}'",
            "rhomax": self.rhomax,
            "rhomin": self.rhomin,
            "alpha": self.alpha,
            "beta": self.beta,
            "softness": self.softness,
            "solvent_radius": self.solvent_radius,
            "radial_scale": self.radial_scale,
        }


@dataclass
class ElectrolyteParameters:
    """
    电解液参数
    
    用于模拟电解液环境中的电极-电解质界面
    """
    # 电解液浓度
    cion: float = 0.0                        # 离子浓度 (M)
    cionmax: float = 1.0                     # 最大离子浓度
    
    # 离子性质
    zion: float = 1.0                        # 离子电荷数
    rion: float = 2.0                        # 离子半径 (bohr)
    
    # 温度
    temperature: float = 300.0               # 温度 (K)
    
    # 介电常数
    permittivity: float = 78.4               # 介电常数
    
    # 界面参数
    surface_tension: float = 50.0            # 表面张力 (dyn/cm)
    
    def calculate_debye_length(self) -> float:
        """
        计算Debye屏蔽长度
        
        Returns:
            Debye长度 (bohr)
        """
        # 常数 (原子单位)
        k_B = 3.16681e-6    # Hartree/K
        T = self.temperature
        
        # 离子强度
        I = self.cion  # 简化处理
        
        # Debye长度公式 (bohr)
        # λ_D = sqrt(εkT / (2e²NI))
        lambda_d = np.sqrt(
            self.permittivity * k_B * T / (2 * I * 0.001)
        ) * 0.529  # 转换为bohr
        
        return lambda_d
    
    def to_namelist_dict(self) -> Dict[str, Any]:
        """转换为namelist字典"""
        return {
            "cion": self.cion,
            "cionmax": self.cionmax,
            "zion": self.zion,
            "rion": self.rion,
            "temperature": self.temperature,
        }


@dataclass
class ExternalCharges:
    """
    外部电荷参数
    
    用于添加显式的外部电荷，如电极电荷
    """
    charges: List[Tuple[float, float, float, float]] = field(default_factory=list)
    """电荷列表: [(x, y, z, q), ...] 单位: bohr, e"""
    
    def add_charge(self, x: float, y: float, z: float, q: float):
        """添加外部电荷"""
        self.charges.append((x, y, z, q))
    
    def to_input_string(self) -> str:
        """生成输入字符串"""
        lines = ["EXTERNAL_CHARGES"]
        lines.append(f"{len(self.charges)}")
        for x, y, z, q in self.charges:
            lines.append(f"{x:.6f} {y:.6f} {z:.6f} {q:.6f}")
        return "\n".join(lines)


class EnvironCalculator:
    """
    Environ计算器
    
    封装QE Environ计算流程，提供便捷的接口
    
    Example:
        >>> env_params = EnvironParameters(
        ...     environ_type="input",
        ...     env_static_permittivity=78.4
        ... )
        >>> calc = EnvironCalculator(env_params, pw_cmd="pw.x")
        >>> energy = calc.calculate("input.pwi")
    """
    
    def __init__(
        self,
        environ_params: EnvironParameters,
        pw_cmd: str = "pw.x",
        environ_cmd: str = "environ.x",
        mpi_cmd: str = "mpirun -np 4",
        work_dir: str = "./environ_calc"
    ):
        """
        初始化Environ计算器
        
        Args:
            environ_params: Environ参数
            pw_cmd: pw.x可执行命令
            environ_cmd: environ.x命令（可选）
            mpi_cmd: MPI并行命令
            work_dir: 工作目录
        """
        self.environ_params = environ_params
        self.pw_cmd = pw_cmd
        self.environ_cmd = environ_cmd
        self.mpi_cmd = mpi_cmd
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # 可选组件
        self.boundary_params: Optional[EnvironBoundaryParameters] = None
        self.electrolyte_params: Optional[ElectrolyteParameters] = None
        self.external_charges: Optional[ExternalCharges] = None
        
        # 结果存储
        self.results: Dict[str, float] = {}
        self.solvation_energy: Optional[float] = None
        
    def set_boundary_params(self, params: EnvironBoundaryParameters):
        """设置边界参数"""
        self.boundary_params = params
    
    def set_electrolyte_params(self, params: ElectrolyteParameters):
        """设置电解液参数"""
        self.electrolyte_params = params
        
        # 更新Debye长度
        lambda_d = params.calculate_debye_length()
        print(f"电解液Debye长度: {lambda_d:.2f} bohr")
    
    def set_external_charges(self, charges: ExternalCharges):
        """设置外部电荷"""
        self.external_charges = charges
    
    def write_input(
        self,
        base_input: Union[str, Path, Dict],
        output_path: Path
    ):
        """
        写入包含Environ参数的输入文件
        
        Args:
            base_input: 基础QE输入文件路径或参数字典
            output_path: 输出路径
        """
        # 读取基础输入
        if isinstance(base_input, (str, Path)):
            with open(base_input, 'r') as f:
                base_content = f.read()
        else:
            base_content = self._dict_to_input(base_input)
        
        # 添加Environ参数
        environ_content = self._generate_environ_section()
        
        # 合并
        full_content = base_content + "\n" + environ_content
        
        # 写入文件
        with open(output_path, 'w') as f:
            f.write(full_content)
    
    def _generate_environ_section(self) -> str:
        """生成Environ输入段"""
        lines = []
        
        # 主Environ参数
        lines.append(self.environ_params.to_input_cards())
        
        # 边界参数
        if self.boundary_params:
            lines.append("\u0026BOUNDARY")
            for key, value in self.boundary_params.to_namelist_dict().items():
                lines.append(f"    {key} = {value}")
            lines.append("\u0026END")
        
        # 电解液参数
        if self.electrolyte_params:
            lines.append("\u0026ELECTROLYTE")
            for key, value in self.electrolyte_params.to_namelist_dict().items():
                lines.append(f"    {key} = {value}")
            lines.append("\u0026END")
        
        # 外部电荷
        if self.external_charges and self.external_charges.charges:
            lines.append(self.external_charges.to_input_string())
        
        return "\n".join(lines)
    
    def _dict_to_input(self, params: Dict) -> str:
        """将参数字典转换为QE输入格式"""
        lines = []
        
        # 控制部分
        if "control" in params:
            lines.append("\u0026CONTROL")
            for key, value in params["control"].items():
                lines.append(f"    {key} = {value}")
            lines.append("\u0026END")
        
        # 系统部分
        if "system" in params:
            lines.append("\u0026SYSTEM")
            for key, value in params["system"].items():
                lines.append(f"    {key} = {value}")
            lines.append("\u0026END")
        
        # 电子部分
        if "electrons" in params:
            lines.append("\u0026ELECTRONS")
            for key, value in params["electrons"].items():
                lines.append(f"    {key} = {value}")
            lines.append("\u0026END")
        
        # 原子位置（如果提供）
        if "atomic_positions" in params:
            lines.append("ATOMIC_POSITIONS (angstrom)")
            lines.extend(params["atomic_positions"])
        
        # 晶胞参数（如果提供）
        if "cell_parameters" in params:
            lines.append("CELL_PARAMETERS (angstrom)")
            lines.extend(params["cell_parameters"])
        
        return "\n".join(lines)
    
    def calculate(
        self,
        input_file: Union[str, Path],
        run: bool = True
    ) -> Dict[str, float]:
        """
        执行Environ计算
        
        Args:
            input_file: 输入文件路径
            run: 是否实际运行计算
            
        Returns:
            计算结果字典
        """
        import shutil
        
        input_path = Path(input_file)
        calc_dir = self.work_dir / "calculation"
        calc_dir.mkdir(exist_ok=True)
        
        # 复制或写入输入文件
        if input_path.exists():
            self.write_input(input_path, calc_dir / "input.pwi")
        else:
            raise FileNotFoundError(f"输入文件不存在: {input_file}")
        
        if run:
            # 运行计算
            self._run_qe(calc_dir)
            
            # 解析结果
            self.results = self._parse_output(calc_dir)
            
            # 提取溶剂化能
            if "solvation_energy" in self.results:
                self.solvation_energy = self.results["solvation_energy"]
        
        return self.results
    
    def calculate_vacuum_reference(
        self,
        input_file: Union[str, Path]
    ) -> float:
        """
        计算真空参考能量
        
        Args:
            input_file: 输入文件路径
            
        Returns:
            真空环境下的总能量
        """
        # 创建真空参数（ε=1）
        vacuum_params = EnvironParameters(
            environ_type="input",
            env_static_permittivity=1.0,
            env_optical_permittivity=1.0
        )
        
        vacuum_calc = EnvironCalculator(
            environ_params=vacuum_params,
            pw_cmd=self.pw_cmd,
            mpi_cmd=self.mpi_cmd,
            work_dir=str(self.work_dir / "vacuum")
        )
        
        results = vacuum_calc.calculate(input_file, run=True)
        return results.get("total_energy", 0.0)
    
    def _run_qe(self, calc_dir: Path):
        """运行QE计算"""
        cmd = f"cd {calc_dir} && {self.mpi_cmd} {self.pw_cmd} -in input.pwi > output.pwo"
        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            warnings.warn(f"QE计算失败: {e}")
    
    def _parse_output(self, calc_dir: Path) -> Dict[str, float]:
        """解析QE输出文件"""
        results = {}
        output_file = calc_dir / "output.pwo"
        
        if not output_file.exists():
            return results
        
        with open(output_file, 'r') as f:
            content = f.read()
        
        # 解析总能量
        energy_match = re.search(r"total energy\s*=\s*([-\d.]+)\s*Ry", content)
        if energy_match:
            results["total_energy_ry"] = float(energy_match.group(1))
            results["total_energy"] = float(energy_match.group(1)) * 13.6057  # eV
        
        # 解析溶剂化能
        solv_match = re.search(r"solvation energy\s*=\s*([-\d.]+)\s*Ry", content)
        if solv_match:
            results["solvation_energy_ry"] = float(solv_match.group(1))
            results["solvation_energy"] = float(solv_match.group(1)) * 13.6057  # eV
        
        # 解析腔体能量
        cavity_match = re.search(r"cavitation energy\s*=\s*([-\d.]+)\s*Ry", content)
        if cavity_match:
            results["cavitation_energy"] = float(cavity_match.group(1)) * 13.6057
        
        # 解析压强功
        pressure_match = re.search(r"pressure energy\s*=\s*([-\d.]+)\s*Ry", content)
        if pressure_match:
            results["pressure_energy"] = float(pressure_match.group(1)) * 13.6057
        
        return results


class PCMCalculator:
    """
    极化连续介质模型(PCM)计算器
    
    实现经典PCM方法的接口，用于与Environ结合
    """
    
    def __init__(
        self,
        dielectric_constant: float = 78.4,
        solvent_radius: float = 1.4,
        probe_radius: float = 1.4,
        tesserae_area: float = 0.3
    ):
        """
        初始化PCM计算器
        
        Args:
            dielectric_constant: 溶剂介电常数
            solvent_radius: 溶剂分子半径 (Angstrom)
            probe_radius: 探针半径 (Angstrom)
            tesserae_area: 表面镶嵌面积 (Angstrom²)
        """
        self.dielectric_constant = dielectric_constant
        self.solvent_radius = solvent_radius
        self.probe_radius = probe_radius
        self.tesserae_area = tesserae_area
        
        # 计算B矩阵（简化模型）
        self.B_matrix: Optional[np.ndarray] = None
        
    def build_surface(
        self,
        atomic_positions: np.ndarray,
        atomic_radii: np.ndarray
    ) -> np.ndarray:
        """
        构建溶剂可及表面(SAS)
        
        Args:
            atomic_positions: 原子位置 (N, 3) Angstrom
            atomic_radii: 原子半径 (N,) Angstrom
            
        Returns:
            表面点坐标 (M, 3)
        """
        # 简化的表面构建
        # 实际实现应使用GMSurface或其他库
        
        surface_points = []
        
        for pos, radius in zip(atomic_positions, atomic_radii):
            # 在原子周围生成球面点
            n_points = int(4 * np.pi * (radius + self.probe_radius)**2 / self.tesserae_area)
            
            # 球坐标采样
            phi = np.random.uniform(0, 2*np.pi, n_points)
            costheta = np.random.uniform(-1, 1, n_points)
            theta = np.arccos(costheta)
            
            r = radius + self.probe_radius
            x = pos[0] + r * np.sin(theta) * np.cos(phi)
            y = pos[1] + r * np.sin(theta) * np.sin(phi)
            z = pos[2] + r * np.cos(theta)
            
            surface_points.extend(np.column_stack([x, y, z]))
        
        return np.array(surface_points)
    
    def calculate_apparent_surface_charge(
        self,
        potential_at_surface: np.ndarray,
        normal_field: np.ndarray
    ) -> np.ndarray:
        """
        计算表观表面电荷(ASC)
        
        q = - (ε - 1) / (4πε) * (∂φ/∂n)
        
        Args:
            potential_at_surface: 表面电势
            normal_field: 表面法向电场
            
        Returns:
            表观表面电荷
        """
        eps = self.dielectric_constant
        
        # ASC计算公式
        asc = - (eps - 1) / (4 * np.pi * eps) * normal_field
        
        return asc
    
    def calculate_solvation_energy(
        self,
        asc: np.ndarray,
        potential_at_surface: np.ndarray
    ) -> float:
        """
        计算溶剂化自由能
        
        G_solv = ½ Σ q_i * φ_i
        
        Args:
            asc: 表观表面电荷
            potential_at_surface: 表面电势
            
        Returns:
            溶剂化能 (Hartree)
        """
        return 0.5 * np.sum(asc * potential_at_surface)


class EnvironWorkflow:
    """
    Environ工作流管理器
    
    自动化Environ计算流程
    """
    
    def __init__(self, work_dir: str = "./environ_workflow"):
        """
        初始化工作流
        
        Args:
            work_dir: 工作目录
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        self.calculators: List[EnvironCalculator] = []
        self.results: List[Dict] = []
        
    def setup_system(
        self,
        structure_data: Dict[str, Any],
        environ_params: EnvironParameters,
        base_qe_params: Optional[Dict] = None
    ):
        """
        设置计算体系
        
        Args:
            structure_data: 结构数据字典
            environ_params: Environ参数
            base_qe_params: 基础QE参数
        """
        self.structure_data = structure_data
        self.environ_params = environ_params
        self.base_qe_params = base_qe_params or self._default_qe_params()
    
    def _default_qe_params(self) -> Dict:
        """默认QE参数"""
        return {
            "control": {
                "calculation": "'scf'",
                "restart_mode": "'from_scratch'",
                "pseudo_dir": "'./'",
                "outdir": "'./outdir/'",
                "tprnfor": ".true.",
                "tstress": ".true.",
            },
            "system": {
                "ecutwfc": 50,
                "ecutrho": 400,
                "occupations": "'smearing'",
                "smearing": "'gaussian'",
                "degauss": 0.02,
            },
            "electrons": {
                "conv_thr": 1e-8,
                "mixing_beta": 0.7,
            }
        }
    
    def run_solvent_screening(
        self,
        solvent_list: List[Dict],
        pw_cmd: str = "pw.x",
        mpi_cmd: str = "mpirun -np 4"
    ) -> List[Dict]:
        """
        运行溶剂筛选计算
        
        Args:
            solvent_list: 溶剂参数列表
            pw_cmd: pw.x命令
            mpi_cmd: MPI命令
            
        Returns:
            计算结果列表
        """
        results = []
        
        for i, solvent_params in enumerate(solvent_list):
            print(f"\n运行溶剂 {i+1}/{len(solvent_list)}: {solvent_params.get('name', 'unknown')}")
            
            # 创建Environ参数
            env_params = EnvironParameters(
                environ_type="input",
                env_static_permittivity=solvent_params.get("epsilon", 78.4),
                env_surface_tension=solvent_params.get("gamma", 0.0),
            )
            
            # 创建工作目录
            calc_dir = self.work_dir / f"solvent_{i:03d}"
            
            # 创建计算器
            calc = EnvironCalculator(
                environ_params=env_params,
                pw_cmd=pw_cmd,
                mpi_cmd=mpi_cmd,
                work_dir=str(calc_dir)
            )
            
            # 准备输入文件
            input_file = calc_dir / "input.pwi"
            calc.write_input(self.base_qe_params, input_file)
            
            # 运行计算
            try:
                calc_results = calc.calculate(input_file, run=False)
                result = {
                    "solvent": solvent_params.get("name"),
                    "epsilon": solvent_params.get("epsilon"),
                    "success": True,
                }
            except Exception as e:
                result = {
                    "solvent": solvent_params.get("name"),
                    "error": str(e),
                    "success": False,
                }
            
            results.append(result)
            self.calculators.append(calc)
        
        self.results = results
        return results
    
    def analyze_solvent_effects(self) -> Dict:
        """分析溶剂效应"""
        if not self.results:
            return {}
        
        successful = [r for r in self.results if r.get("success")]
        
        return {
            "total": len(self.results),
            "successful": len(successful),
            "solvents_tested": [r.get("solvent") for r in successful],
        }


# ============================================================
# 使用示例
# ============================================================

def example_basic_environ():
    """基本Environ计算示例"""
    # 创建Environ参数
    env_params = EnvironParameters(
        environ_type="input",
        env_static_permittivity=78.4,    # 水
        env_optical_permittivity=1.78,
        env_surface_tension=0.0001,      # Ry/bohr²
        env_pressure=0.0,
        env_periodicity="2D",            # 2D周期（表面计算）
        env_axis=3                       # Z方向非周期
    )
    
    print("Environ参数:")
    print(f"  静态介电常数: {env_params.env_static_permittivity}")
    print(f"  光学介电常数: {env_params.env_optical_permittivity}")
    print(f"  周期性: {env_params.env_periodicity}")
    
    # 创建计算器
    calc = EnvironCalculator(
        environ_params=env_params,
        work_dir="./example_environ"
    )
    
    # 生成输入示例
    qe_params = {
        "control": {
            "calculation": "'scf'",
            "pseudo_dir": "'./'",
        },
        "system": {
            "ecutwfc": 50,
            "ecutrho": 400,
        },
        "electrons": {
            "conv_thr": 1e-8,
        }
    }
    
    calc.write_input(qe_params, Path("./example_environ/example.pwi"))
    print("\n示例输入文件已生成: ./example_environ/example.pwi")
    
    return calc


def example_electrolyte():
    """电解液参数示例"""
    # 创建电解液参数
    electrolyte = ElectrolyteParameters(
        cion=0.1,              # 0.1 M
        cionmax=1.0,           # 最大浓度
        zion=1.0,              # 单价离子
        rion=2.0,              # 离子半径 (bohr)
        temperature=300.0,     # 温度
        permittivity=78.4      # 介电常数
    )
    
    print("电解液参数:")
    print(f"  离子浓度: {electrolyte.cion} M")
    print(f"  离子电荷: {electrolyte.zion}")
    print(f"  温度: {electrolyte.temperature} K")
    
    # 计算Debye长度
    lambda_d = electrolyte.calculate_debye_length()
    print(f"  Debye长度: {lambda_d:.2f} bohr")
    
    return electrolyte


def example_pcm():
    """PCM计算示例"""
    # 创建PCM计算器
    pcm = PCMCalculator(
        dielectric_constant=78.4,
        solvent_radius=1.4,
        probe_radius=1.4,
        tesserae_area=0.3
    )
    
    # 示例原子位置
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    radii = np.array([1.2, 1.2])
    
    # 构建表面
    surface = pcm.build_surface(positions, radii)
    print(f"表面点数: {len(surface)}")
    
    # 模拟电势和电场
    potential = np.random.randn(len(surface)) * 0.1
    field = np.random.randn(len(surface)) * 0.05
    
    # 计算ASC
    asc = pcm.calculate_apparent_surface_charge(potential, field)
    print(f"ASC数量: {len(asc)}")
    print(f"ASC总和: {np.sum(asc):.6f}")
    
    # 计算溶剂化能
    solv_energy = pcm.calculate_solvation_energy(asc, potential)
    print(f"溶剂化能: {solv_energy:.6f} Hartree")
    
    return pcm


def example_workflow():
    """工作流示例"""
    # 创建工作流
    workflow = EnvironWorkflow("./example_environ_workflow")
    
    # 设置体系
    env_params = EnvironParameters(
        environ_type="input",
        env_static_permittivity=78.4
    )
    
    workflow.setup_system(
        structure_data={"nat": 4, "ntyp": 1},
        environ_params=env_params
    )
    
    # 定义溶剂列表
    solvents = [
        {"name": "Water", "epsilon": 78.4, "gamma": 50.0},
        {"name": "Acetonitrile", "epsilon": 37.5, "gamma": 30.0},
        {"name": "Methanol", "epsilon": 32.7, "gamma": 22.0},
    ]
    
    print("溶剂筛选设置完成:")
    for s in solvents:
        print(f"  {s['name']}: ε = {s['epsilon']}")
    
    return workflow


if __name__ == "__main__":
    print("=" * 60)
    print("QE Environ Interface Module - 使用示例")
    print("=" * 60)
    
    print("\n1. 基本Environ计算示例")
    print("-" * 40)
    example_basic_environ()
    
    print("\n2. 电解液参数示例")
    print("-" * 40)
    example_electrolyte()
    
    print("\n3. PCM计算示例")
    print("-" * 40)
    example_pcm()
    
    print("\n4. 工作流示例")
    print("-" * 40)
    example_workflow()
    
    print("\n" + "=" * 60)
    print("所有示例完成!")
    print("=" * 60)