"""
Constant Potential Module - 恒电势DFT计算

本模块实现恒电势计算功能，包括：
- 巨正则DFT (Grand Canonical DFT, GC-DFT)
- 电荷中性化方法
- 工作函数计算与调整
- 电容模型

Reference:
    - Sundararaman et al., J. Chem. Phys. 2017, 146, 114104
    - Zhang et al., Phys. Rev. B 2020, 102, 125105

Author: DFT-LAMMPS Team
Date: 2025
"""

import os
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import warnings
from enum import Enum
from scipy.optimize import minimize_scalar, brentq


class ChargeNeutralizationMethod(Enum):
    """电荷中性化方法"""
    NONE = "none"                     # 不中性化
    UNIFORM = "uniform"               # 均匀背景电荷
    GAUSSIAN = "gaussian"             # 高斯分布背景
    PLANE_AVERAGED = "plane"          # 平面平均
    POISSON_SOLVER = "poisson"        # Poisson求解器


class GrandPotentialMethod(Enum):
    """巨正则势计算方法"""
    LEGENDRE = "legendre"             # Legendre变换
    DIRECT = "direct"                 # 直接优化
    SCF_LOOP = "scf"                  # SCF循环


@dataclass
class ElectrodeParameters:
    """
    电极参数
    
    Attributes:
        potential: 目标电势 (V vs SHE)
        capacitance: 界面电容 (μF/cm²)
        area: 电极表面积 (Å²)
        work_function: 工作函数 (eV)
        fermi_level: Fermi能级 (eV)
    """
    potential: float = 0.0              # V vs SHE
    capacitance: float = 20.0           # μF/cm²
    area: float = 100.0                 # Å²
    work_function: float = 4.5          # eV
    fermi_level: float = -4.5           # eV (相对于真空)
    
    # 参考电势
    reference_potential: float = 4.44   # SHE vs 真空 (eV)
    
    def calculate_excess_charge(self, delta_phi: float) -> float:
        """
        计算过量电荷
        
        Q = C × A × Δφ
        
        Args:
            delta_phi: 电势差 (V)
            
        Returns:
            过量电荷 (e)
        """
        # 电容单位转换: μF/cm² → F/m²
        C = self.capacitance * 1e-6 / 1e4  # F/m²
        
        # 面积转换: Å² → m²
        A = self.area * 1e-20  # m²
        
        # 电荷: Q = C × V
        Q_coulomb = C * A * delta_phi  # C
        
        # 转换为电子数
        e = 1.602e-19  # C
        Q_electron = Q_coulomb / e
        
        return Q_electron
    
    def potential_to_charge(self, target_potential: float) -> float:
        """
        将目标电势转换为电荷
        
        Args:
            target_potential: 目标电势 (V vs SHE)
            
        Returns:
            所需电荷 (e)
        """
        delta_phi = target_potential - self.potential
        return self.calculate_excess_charge(delta_phi)


@dataclass
class GCDFTParameters:
    """
    巨正则DFT参数
    
    Attributes:
        mu_electron: 电子化学势 (eV)
        target_charge: 目标电荷 (e)
        temperature: 温度 (K)
        nelect: 电子数
        charge_tolerance: 电荷收敛阈值
    """
    mu_electron: float = -4.5           # eV
    target_charge: Optional[float] = None  # 目标电荷
    temperature: float = 298.15         # K
    nelect: float = 0.0                 # 当前电子数
    charge_tolerance: float = 0.01      # 电荷收敛阈值
    
    # 收敛参数
    max_iterations: int = 100
    mixing_beta: float = 0.5
    
    def calculate_grand_potential(
        self,
        total_energy: float,
        nelect: float
    ) -> float:
        """
        计算巨正则势
        
        Ω = E - μN
        
        Args:
            total_energy: 总能量 (eV)
            nelect: 电子数
            
        Returns:
            巨正则势 (eV)
        """
        return total_energy - self.mu_electron * nelect


class ConstantPotentialCalculator:
    """
    恒电势计算器
    
    实现恒电势DFT计算的核心功能
    
    Example:
        >>> electrode = ElectrodeParameters(potential=0.5, capacitance=20.0)
        >>> calc = ConstantPotentialCalculator(electrode)
        >>> calc.set_target_potential(0.8)  # V vs SHE
        >>> charge = calc.optimize_charge()
    """
    
    def __init__(
        self,
        electrode: ElectrodeParameters,
        method: ChargeNeutralizationMethod = ChargeNeutralizationMethod.UNIFORM,
        work_dir: str = "./constant_potential"
    ):
        """
        初始化恒电势计算器
        
        Args:
            electrode: 电极参数
            method: 电荷中性化方法
            work_dir: 工作目录
        """
        self.electrode = electrode
        self.method = method
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # 目标电势
        self.target_potential: Optional[float] = None
        self.current_charge: float = 0.0
        
        # GC-DFT参数
        self.gcdft_params: Optional[GCDFTParameters] = None
        
        # 结果存储
        self.results: Dict[str, float] = {}
        self.potential_vs_charge: List[Tuple[float, float]] = []
        
    def set_target_potential(self, potential: float):
        """
        设置目标电势
        
        Args:
            potential: 目标电势 (V vs SHE)
        """
        self.target_potential = potential
        
        # 估算所需电荷
        self.current_charge = self.electrode.potential_to_charge(potential)
        print(f"目标电势: {potential:.3f} V vs SHE")
        print(f"估算电荷: {self.current_charge:.4f} e")
    
    def set_gcdft_parameters(self, params: GCDFTParameters):
        """
        设置GC-DFT参数
        
        Args:
            params: GC-DFT参数
        """
        self.gcdft_params = params
    
    def optimize_charge(
        self,
        energy_calculator: Optional[Callable[[float], Tuple[float, float]]] = None
    ) -> float:
        """
        优化电荷以达到目标电势
        
        Args:
            energy_calculator: 能量计算函数 Q -> (E, actual_phi)
            
        Returns:
            优化后的电荷
        """
        if self.target_potential is None:
            raise ValueError("必须先设置目标电势")
        
        if energy_calculator is None:
            # 使用简化的能量模型
            energy_calculator = self._mock_energy_calculator
        
        # 定义目标函数: 最小化 (实际电势 - 目标电势)^2
        def objective(charge: float) -> float:
            _, actual_phi = energy_calculator(charge)
            return (actual_phi - self.target_potential)**2
        
        # 优化
        result = minimize_scalar(
            objective,
            bounds=(-5.0, 5.0),
            method='bounded'
        )
        
        optimal_charge = result.x
        self.current_charge = optimal_charge
        
        # 获取最终能量和电势
        energy, actual_phi = energy_calculator(optimal_charge)
        
        self.results = {
            "optimal_charge": optimal_charge,
            "total_energy": energy,
            "actual_potential": actual_phi,
            "target_potential": self.target_potential,
        }
        
        print(f"优化完成:")
        print(f"  最优电荷: {optimal_charge:.4f} e")
        print(f"  实际电势: {actual_phi:.4f} V")
        
        return optimal_charge
    
    def _mock_energy_calculator(self, charge: float) -> Tuple[float, float]:
        """
        简化的能量计算器（用于演示）
        
        模拟电势-电荷关系: φ = φ₀ + Q/C
        
        Args:
            charge: 电荷 (e)
            
        Returns:
            (能量, 电势)
        """
        # 电容模型: Δφ = Q/C
        C_eff = self.electrode.capacitance * 1e-6 / 1e4  # F/m²
        A = self.electrode.area * 1e-20  # m²
        e = 1.602e-19  # C
        
        # 电势变化 (V)
        delta_phi = charge * e / (C_eff * A)
        actual_phi = self.electrode.potential + delta_phi
        
        # 简化能量模型
        energy = self.electrode.work_function + 0.5 * charge * delta_phi
        
        return energy, actual_phi
    
    def calculate_capacitance(
        self,
        charge_range: Tuple[float, float] = (-1.0, 1.0),
        n_points: int = 20
    ) -> Dict[str, np.ndarray]:
        """
        计算微分电容
        
        Args:
            charge_range: 电荷范围 (e)
            n_points: 计算点数
            
        Returns:
            电容数据
        """
        charges = np.linspace(charge_range[0], charge_range[1], n_points)
        potentials = []
        
        for q in charges:
            _, phi = self._mock_energy_calculator(q)
            potentials.append(phi)
        
        potentials = np.array(potentials)
        
        # 计算微分电容 C = dQ/dV
        capacitance = np.gradient(charges, potentials)
        
        # 存储结果
        self.potential_vs_charge = list(zip(potentials, charges))
        
        return {
            "charge": charges,
            "potential": potentials,
            "capacitance": capacitance,
        }
    
    def generate_nelect_adjustment(
        self,
        base_nelect: float,
        target_charge: float
    ) -> float:
        """
        生成NELECT调整值
        
        Args:
            base_nelect: 基础电子数
            target_charge: 目标电荷
            
        Returns:
            调整后的NELECT
        """
        # 负电荷 = 更多电子
        return base_nelect + target_charge
    
    def write_vasp_incar(
        self,
        base_incar: Dict,
        nelect: float,
        output_path: Path
    ):
        """
        写入VASP INCAR（含NELECT调整）
        
        Args:
            base_incar: 基础INCAR参数
            nelect: 电子数
            output_path: 输出路径
        """
        incar = base_incar.copy()
        incar["NELECT"] = nelect
        
        # 添加电荷中性化参数
        if self.method == ChargeNeutralizationMethod.UNIFORM:
            incar["LVHAR"] = ".TRUE."
        
        with open(output_path, 'w') as f:
            f.write("# Constant Potential Calculation\n")
            for key, value in incar.items():
                f.write(f"{key:15s} = {value}\n")
    
    def write_quantum_espresso_input(
        self,
        base_input: Dict,
        nelect: float,
        output_path: Path
    ):
        """
        写入QE输入文件（含电荷调整）
        
        Args:
            base_input: 基础输入参数
            nelect: 电子数
            output_path: 输出路径
        """
        input_data = base_input.copy()
        
        # 更新电子数
        if "system" not in input_data:
            input_data["system"] = {}
        input_data["system"]["tot_charge"] = -nelect  # QE使用电荷而非电子数
        
        # 写入文件（简化格式）
        with open(output_path, 'w') as f:
            for section, params in input_data.items():
                f.write(f"\u0026{section.upper()}\n")
                for key, value in params.items():
                    if isinstance(value, str):
                        f.write(f"    {key} = {value}\n")
                    else:
                        f.write(f"    {key} = {value}\n")
                f.write("\u0026END\n\n")


class ChargeNeutralizer:
    """
    电荷中性化器
    
    实现各种电荷中性化方法
    """
    
    def __init__(
        self,
        method: ChargeNeutralizationMethod = ChargeNeutralizationMethod.UNIFORM,
        grid_shape: Tuple[int, int, int] = (100, 100, 100)
    ):
        """
        初始化中性化器
        
        Args:
            method: 中性化方法
            grid_shape: 电荷密度网格大小
        """
        self.method = method
        self.grid_shape = grid_shape
        
    def create_background_charge(
        self,
        total_charge: float,
        cell: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        创建背景电荷分布
        
        Args:
            total_charge: 总电荷 (e)
            cell: 晶胞参数 (3x3矩阵)
            
        Returns:
            电荷密度网格
        """
        grid = np.zeros(self.grid_shape)
        
        if self.method == ChargeNeutralizationMethod.UNIFORM:
            # 均匀分布
            grid.fill(-total_charge / np.prod(self.grid_shape))
            
        elif self.method == ChargeNeutralizationMethod.GAUSSIAN:
            # 高斯分布（中心集中）
            center = np.array(self.grid_shape) // 2
            sigma = min(self.grid_shape) / 4
            
            x = np.arange(self.grid_shape[0])
            y = np.arange(self.grid_shape[1])
            z = np.arange(self.grid_shape[2])
            
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            
            r2 = ((X - center[0])**2 + 
                  (Y - center[1])**2 + 
                  (Z - center[2])**2)
            
            gaussian = np.exp(-r2 / (2 * sigma**2))
            gaussian /= gaussian.sum()
            
            grid = -total_charge * gaussian
            
        elif self.method == ChargeNeutralizationMethod.PLANE_AVERAGED:
            # 平面平均（沿z方向）
            grid.fill(-total_charge / np.prod(self.grid_shape))
            # 可以添加更复杂的平面依赖
            
        return grid
    
    def apply_to_vasp_charge_density(
        self,
        charge_file: Union[str, Path],
        background_charge: np.ndarray,
        output_file: Union[str, Path]
    ):
        """
        将背景电荷应用到VASP电荷密度文件
        
        Args:
            charge_file: 输入CHGCAR路径
            background_charge: 背景电荷网格
            output_file: 输出路径
        """
        # 简化的实现
        # 实际应使用pymatgen等库解析CHGCAR
        warnings.warn("简化的电荷密度修改，实际使用需完善")
        
    def calculate_ewald_correction(
        self,
        charge: float,
        cell: np.ndarray,
        precision: float = 1e-6
    ) -> float:
        """
        计算Ewald自能校正
        
        Args:
            charge: 总电荷
            cell: 晶胞
            precision: 精度
            
        Returns:
            Ewald能量校正 (eV)
        """
        # 体积
        volume = np.abs(np.linalg.det(cell))
        
        # Ewald自能（均匀背景电荷）
        # E_self = -Q² / (2ε₀V) * α/√π
        # 简化计算
        alpha = 1.0  # Ewald参数
        eps_0 = 8.854e-12  # F/m
        e = 1.602e-19  # C
        
        # 转换体积到m³
        volume_m3 = volume * 1e-30
        
        # 能量校正 (J)
        E_joule = -(charge * e)**2 / (2 * eps_0 * volume_m3) * alpha / np.sqrt(np.pi)
        
        # 转换为eV
        E_ev = E_joule / e
        
        return E_ev


class GrandCanonicalDFT:
    """
    巨正则DFT实现
    
    使用巨正则系综进行恒电势计算
    
    Reference:
        - Sundararaman et al., J. Chem. Phys. 2017, 146, 114104
    """
    
    def __init__(
        self,
        temperature: float = 298.15,
        method: GrandPotentialMethod = GrandPotentialMethod.LEGENDRE
    ):
        """
        初始化GC-DFT
        
        Args:
            temperature: 温度 (K)
            method: 计算方法
        """
        self.temperature = temperature
        self.method = method
        self.kT = 8.617e-5 * temperature  # eV
        
        # 存储势能面
        self.energy_vs_charge: List[Tuple[float, float]] = []
        
    def legendre_transform(
        self,
        charges: np.ndarray,
        energies: np.ndarray,
        target_mu: float
    ) -> Tuple[float, float]:
        """
        Legendre变换求巨正则势
        
        Ω(μ) = min_N [E(N) - μN]
        
        Args:
            charges: 电荷数组 (N)
            energies: 能量数组 (E)
            target_mu: 目标化学势
            
        Returns:
            (最优电荷, 巨正则势)
        """
        # 计算Ω(N) = E(N) - μN
        grand_potentials = energies - target_mu * charges
        
        # 找到最小值
        min_idx = np.argmin(grand_potentials)
        optimal_charge = charges[min_idx]
        min_grand_potential = grand_potentials[min_idx]
        
        return optimal_charge, min_grand_potential
    
    def fit_energy_vs_charge(
        self,
        charges: np.ndarray,
        energies: np.ndarray,
        order: int = 3
    ) -> Callable[[float], float]:
        """
        拟合E vs Q曲线
        
        Args:
            charges: 电荷
            energies: 能量
            order: 多项式阶数
            
        Returns:
            拟合函数
        """
        # 多项式拟合
        coeffs = np.polyfit(charges, energies, order)
        
        def energy_function(q: float) -> float:
            return np.polyval(coeffs, q)
        
        return energy_function
    
    def calculate_electronic_entropy(
        self,
        eigenvalues: np.ndarray,
        fermi_level: float
    ) -> float:
        """
        计算电子熵
        
        Args:
            eigenvalues: 本征值数组
            fermi_level: Fermi能级
            
        Returns:
            电子熵贡献 (eV)
        """
        # Fermi-Dirac分布
        delta_e = eigenvalues - fermi_level
        f = 1.0 / (np.exp(delta_e / self.kT) + 1)
        
        # 熵: S = -k_B Σ [f ln f + (1-f) ln(1-f)]
        # 避免log(0)
        f = np.clip(f, 1e-10, 1 - 1e-10)
        
        entropy = -self.kT * np.sum(f * np.log(f) + (1 - f) * np.log(1 - f))
        
        return entropy
    
    def scf_charge_optimization(
        self,
        initial_charge: float,
        energy_func: Callable[[float], float],
        potential_func: Callable[[float], float],
        target_potential: float,
        tolerance: float = 0.01
    ) -> Dict:
        """
        SCF电荷优化
        
        Args:
            initial_charge: 初始电荷
            energy_func: 能量函数
            potential_func: 电势函数
            target_potential: 目标电势
            tolerance: 收敛阈值
            
        Returns:
            优化结果
        """
        charge = initial_charge
        history = []
        
        for iteration in range(100):
            # 计算当前电势
            current_potential = potential_func(charge)
            
            # 电势差
            delta_phi = target_potential - current_potential
            
            history.append({
                "iteration": iteration,
                "charge": charge,
                "potential": current_potential,
            })
            
            # 检查收敛
            if abs(delta_phi) < tolerance:
                break
            
            # 更新电荷（电容模型）
            # ΔQ = C × Δφ
            C_eff = 20.0  # μF/cm²，简化值
            delta_q = C_eff * 0.01 * delta_phi  # 简化计算
            
            charge += delta_q * 0.5  # 混合
        
        final_energy = energy_func(charge)
        
        return {
            "optimal_charge": charge,
            "final_energy": final_energy,
            "final_potential": potential_func(charge),
            "iterations": len(history),
            "history": history,
        }


class WorkFunctionAnalyzer:
    """
    工作函数分析器
    
    计算和调整电极工作函数
    """
    
    def __init__(self):
        """初始化分析器"""
        self.planar_average: Optional[np.ndarray] = None
        self.work_function: Optional[float] = None
        
    def calculate_planar_average(
        self,
        potential_grid: np.ndarray,
        axis: int = 2
    ) -> np.ndarray:
        """
        计算平面平均电势
        
        Args:
            potential_grid: 三维电势网格
            axis: 平均方向
            
        Returns:
            平面平均电势数组
        """
        # 沿指定轴平均
        axes = [0, 1, 2]
        axes.remove(axis)
        
        planar_avg = np.mean(potential_grid, axis=tuple(axes))
        self.planar_average = planar_avg
        
        return planar_avg
    
    def calculate_work_function(
        self,
        fermi_energy: float,
        vacuum_potential: Optional[float] = None
    ) -> float:
        """
        计算工作函数
        
        Φ = V_vacuum - E_Fermi
        
        Args:
            fermi_energy: Fermi能级
            vacuum_potential: 真空电势，默认为平面平均的最大值
            
        Returns:
            工作函数 (eV)
        """
        if vacuum_potential is None and self.planar_average is not None:
            vacuum_potential = np.max(self.planar_average)
        
        if vacuum_potential is None:
            raise ValueError("需要提供真空电势或先计算平面平均")
        
        self.work_function = vacuum_potential - fermi_energy
        return self.work_function
    
    def potential_to_she(self, potential: float, work_function: float) -> float:
        """
        将绝对电势转换为SHE标度
        
        E(vs SHE) = - (Φ - 4.44) - eU
        
        Args:
            potential: 绝对电势
            work_function: 工作函数
            
        Returns:
            SHE标度电势 (V)
        """
        # SHE相对于真空约为4.44 eV
        she_reference = 4.44
        
        return -(work_function - she_reference) - potential
    
    def calculate_dipole_correction(
        self,
        dipole_moment: float,
        cell_volume: float,
        axis: int = 2
    ) -> float:
        """
        计算偶极校正
        
        Args:
            dipole_moment: 偶极矩 (e·Å)
            cell_volume: 晶胞体积 (Å³)
            axis: 偶极方向
            
        Returns:
            偶极校正能量 (eV)
        """
        # 简化计算
        # E_dipole ∝ μ²/V
        eps_0 = 8.854e-12  # F/m
        
        # 单位转换
        mu_coulomb_m = dipole_moment * 1.602e-19 * 1e-10  # C·m
        volume_m3 = cell_volume * 1e-30  # m³
        
        # 能量 (J)
        E_joule = -mu_coulomb_m**2 / (2 * eps_0 * volume_m3)
        
        # 转换为eV
        E_ev = E_joule / 1.602e-19
        
        return E_ev


# ============================================================
# 使用示例
# ============================================================

def example_constant_potential():
    """恒电势计算示例"""
    print("恒电势计算示例")
    print("-" * 40)
    
    # 创建电极参数
    electrode = ElectrodeParameters(
        potential=0.0,        # 参考电势
        capacitance=20.0,     # μF/cm²
        area=100.0,           # Å²
        work_function=4.5     # eV
    )
    
    print(f"电极参数:")
    print(f"  参考电势: {electrode.potential} V vs SHE")
    print(f"  电容: {electrode.capacitance} μF/cm²")
    print(f"  面积: {electrode.area} Å²")
    print(f"  工作函数: {electrode.work_function} eV")
    
    # 计算电荷
    charge = electrode.calculate_excess_charge(0.5)  # +0.5 V
    print(f"\n+0.5 V所需的电荷: {charge:.4f} e")
    
    # 创建恒电势计算器
    calc = ConstantPotentialCalculator(electrode)
    
    # 设置目标电势
    calc.set_target_potential(0.8)  # 0.8 V vs SHE
    
    # 优化电荷（使用模拟器）
    optimal_charge = calc.optimize_charge()
    
    # 计算电容
    cap_data = calc.calculate_capacitance((-1.0, 1.0), 20)
    print(f"\n电容数据分析:")
    print(f"  电荷范围: [{cap_data['charge'][0]:.2f}, {cap_data['charge'][-1]:.2f}] e")
    print(f"  电势范围: [{cap_data['potential'][0]:.2f}, {cap_data['potential'][-1]:.2f}] V")
    print(f"  平均电容: {np.mean(cap_data['capacitance']):.2f} μF/cm²")
    
    return calc


def example_charge_neutralization():
    """电荷中性化示例"""
    print("\n电荷中性化示例")
    print("-" * 40)
    
    # 创建中性化器
    neutralizer = ChargeNeutralizer(
        method=ChargeNeutralizationMethod.GAUSSIAN,
        grid_shape=(50, 50, 50)
    )
    
    # 创建背景电荷
    total_charge = 0.5  # e
    bg_charge = neutralizer.create_background_charge(total_charge)
    
    print(f"总电荷: {total_charge:.2f} e")
    print(f"网格大小: {neutralizer.grid_shape}")
    print(f"背景电荷总和: {np.sum(bg_charge):.6f} e")
    print(f"电荷分布范围: [{np.min(bg_charge):.6f}, {np.max(bg_charge):.6f}] e/格点")
    
    # 计算Ewald校正
    cell = np.eye(3) * 10.0  # 10 Å立方晶胞
    ewald_corr = neutralizer.calculate_ewald_correction(total_charge, cell)
    print(f"\nEwald自能校正: {ewald_corr:.4f} eV")
    
    return neutralizer


def example_gcdft():
    """GC-DFT示例"""
    print("\nGC-DFT示例")
    print("-" * 40)
    
    # 创建GC-DFT对象
    gcdft = GrandCanonicalDFT(temperature=298.15)
    
    # 模拟数据: 能量vs电荷
    charges = np.linspace(-1.0, 1.0, 21)
    energies = charges**2 * 2.0 + charges * 0.5  # 简化的抛物线模型
    
    print("能量vs电荷数据 (前5点):")
    for q, e in zip(charges[:5], energies[:5]):
        print(f"  Q = {q:+.2f} e, E = {e:.4f} eV")
    
    # Legendre变换
    target_mu = -4.5  # eV
    optimal_q, omega = gcdft.legendre_transform(charges, energies, target_mu)
    
    print(f"\nLegendre变换结果:")
    print(f"  目标化学势: {target_mu:.2f} eV")
    print(f"  最优电荷: {optimal_q:.4f} e")
    print(f"  巨正则势: {omega:.4f} eV")
    
    # 拟合曲线
    energy_func = gcdft.fit_energy_vs_charge(charges, energies, order=2)
    
    # SCF优化
    result = gcdft.scf_charge_optimization(
        initial_charge=0.0,
        energy_func=energy_func,
        potential_func=lambda q: 0.1 * q,  # 简化电势函数
        target_potential=0.5,
        tolerance=0.01
    )
    
    print(f"\nSCF优化结果:")
    print(f"  最优电荷: {result['optimal_charge']:.4f} e")
    print(f"  最终能量: {result['final_energy']:.4f} eV")
    print(f"  迭代次数: {result['iterations']}")
    
    return gcdft


def example_work_function():
    """工作函数分析示例"""
    print("\n工作函数分析示例")
    print("-" * 40)
    
    analyzer = WorkFunctionAnalyzer()
    
    # 创建模拟电势网格
    grid = np.random.randn(50, 50, 50) * 0.1
    # 添加趋势模拟真空区域
    z = np.linspace(0, 10, 50)
    for i, z_val in enumerate(z):
        grid[:, :, i] += z_val * 0.5
    
    # 计算平面平均
    planar_avg = analyzer.calculate_planar_average(grid, axis=2)
    print(f"平面平均电势:")
    print(f"  最小值: {np.min(planar_avg):.4f} eV")
    print(f"  最大值: {np.max(planar_avg):.4f} eV (真空)")
    print(f"  平均值: {np.mean(planar_avg):.4f} eV")
    
    # 计算工作函数
    fermi_energy = 2.5  # eV
    phi = analyzer.calculate_work_function(fermi_energy)
    print(f"\n工作函数:")
    print(f"  Fermi能级: {fermi_energy:.2f} eV")
    print(f"  工作函数: {phi:.2f} eV")
    
    # 转换为SHE
    she_potential = analyzer.potential_to_she(0.0, phi)
    print(f"\nSHE转换:")
    print(f"  电势 vs SHE: {she_potential:.2f} V")
    
    return analyzer


def example_complete_workflow():
    """完整工作流程示例"""
    print("\n完整工作流程示例")
    print("-" * 40)
    
    # 步骤1: 设置电极参数
    electrode = ElectrodeParameters(
        potential=0.0,
        capacitance=25.0,
        area=150.0,
        work_function=4.8
    )
    
    # 步骤2: 设置目标电势
    target_U = 1.23  # OER电位
    
    # 步骤3: 计算所需电荷
    calc = ConstantPotentialCalculator(electrode)
    calc.set_target_potential(target_U)
    
    # 步骤4: 准备GC-DFT计算
    gcdft = GrandCanonicalDFT(temperature=298.15)
    
    # 步骤5: 电荷中性化设置
    neutralizer = ChargeNeutralizer(
        method=ChargeNeutralizationMethod.UNIFORM
    )
    
    print(f"工作流程:")
    print(f"  1. 电极工作函数: {electrode.work_function:.2f} eV")
    print(f"  2. 目标电势: {target_U:.2f} V vs SHE")
    print(f"  3. 估算电荷: {calc.current_charge:.4f} e")
    print(f"  4. 中性化方法: {neutralizer.method.value}")
    print(f"  5. GC-DFT温度: {gcdft.temperature:.1f} K")
    
    # 模拟计算结果
    print(f"\n模拟OER计算结果:")
    print(f"  平衡电势: 1.23 V")
    print(f"  目标过电位: 0.3 V")
    print(f"  实际工作电势: {target_U + 0.3:.2f} V")
    print(f"  所需额外电荷: {electrode.calculate_excess_charge(0.3):.4f} e")
    
    return calc, gcdft, neutralizer


if __name__ == "__main__":
    print("=" * 60)
    print("Constant Potential Module - 使用示例")
    print("=" * 60)
    
    example_constant_potential()
    example_charge_neutralization()
    example_gcdft()
    example_work_function()
    example_complete_workflow()
    
    print("\n" + "=" * 60)
    print("所有示例完成!")
    print("=" * 60)