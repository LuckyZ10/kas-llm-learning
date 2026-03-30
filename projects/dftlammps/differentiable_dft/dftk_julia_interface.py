"""
DFTK.jl Interface Module
========================

Julia DFTK.jl 的 Python 接口，利用 Julia 的自动微分生态。
提供平面波DFT的可微实现和几何优化梯度流。

核心功能：
- DFTK.jl 的 Python 封装
- Julia自动微分 (ForwardDiff/Zygote) 集成
- 平面波基组的解析梯度
- 结构优化的梯度流方法
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass
import subprocess
import json
import os
import tempfile


@dataclass
class DFTKConfig:
    """DFTK计算配置"""
    # 电子结构参数
    ecut: float = 15.0  # 平面波截断能 (Ha)
    kgrid: Tuple[int, int, int] = (3, 3, 3)  # K点网格
    kshift: Tuple[float, float, float] = (0.5, 0.5, 0.5)  # K点偏移
    
    # 交换关联泛函
    functional: str = "lda_xc_teter93"  # 或 "gga_x_pbe+gga_c_pbe"
    
    # SCF参数
    scf_maxiter: int = 100
    scf_tol: float = 1e-6
    mixing: str = "SimpleMixing"  # 或 "KerkerMixing", "DielectricMixing"
    mixing_beta: float = 0.7
    
    # 自洽迭代
    diag_algorithm: str = "lobpcg_hypre"  # 对角化算法
    
    # 自动微分设置
    autodiff_backend: str = "Zygote"  # "ForwardDiff" 或 "Zygote"
    
    # 并行设置
    n_threads: int = 4


@dataclass
class LatticeSystem:
    """晶格系统定义"""
    lattice: np.ndarray  # 晶格向量 (3, 3)，单位 Bohr
    positions: np.ndarray  # 原子位置 (N, 3)，分数坐标
    atomic_symbols: List[str]  # 元素符号列表
    magnetic_moments: Optional[np.ndarray] = None  # 磁矩


class DFTKInterface:
    """
    DFTK.jl 主接口类
    
    提供平面波DFT计算的核心功能，包括自动微分支持
    """
    
    # 内置赝势库映射
    PSP_LIBRARY = {
        'h': 'hgh/lda/h-q1', 'he': 'hgh/lda/he-q2',
        'li': 'hgh/lda/li-q3', 'be': 'hgh/lda/be-q4',
        'b': 'hgh/lda/b-q3', 'c': 'hgh/lda/c-q4',
        'n': 'hgh/lda/n-q5', 'o': 'hgh/lda/o-q6',
        'f': 'hgh/lda/f-q7', 'ne': 'hgh/lda/ne-q8',
        'na': 'hgh/lda/na-q9', 'mg': 'hgh/lda/mg-q10',
        'al': 'hgh/lda/al-q3', 'si': 'hgh/lda/si-q4',
        'p': 'hgh/lda/p-q5', 's': 'hgh/lda/s-q6',
        'cl': 'hgh/lda/cl-q7', 'ar': 'hgh/lda/ar-q8',
        'k': 'hgh/lda/k-q9', 'ca': 'hgh/lda/ca-q10',
    }
    
    def __init__(self, config: DFTKConfig):
        self.config = config
        self._check_julia_environment()
        
    def _check_julia_environment(self):
        """检查Julia环境"""
        try:
            result = subprocess.run(
                ['julia', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError("Julia未正确安装")
            print(f"Julia版本: {result.stdout.strip()}")
        except FileNotFoundError:
            print("警告: Julia未安装。DFTK.jl功能将不可用。")
            print("请安装Julia并运行: julia -e 'using Pkg; Pkg.add(\"DFTK\")'")
    
    def _get_psp_file(self, symbol: str) -> str:
        """获取赝势文件路径"""
        sym_lower = symbol.lower()
        if sym_lower in self.PSP_LIBRARY:
            return self.PSP_LIBRARY[sym_lower]
        else:
            # 默认使用HGH LDA赝势
            return f"hgh/lda/{sym_lower}-q{self._get_valence_electrons(symbol)}"
    
    def _get_valence_electrons(self, symbol: str) -> int:
        """获取价电子数 (简化)"""
        valence_map = {
            'H': 1, 'He': 2, 'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6,
            'F': 7, 'Ne': 8, 'Na': 1, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 5, 'S': 6,
            'Cl': 7, 'Ar': 8, 'K': 1, 'Ca': 2, 'Sc': 3, 'Ti': 4, 'V': 5, 'Cr': 6,
            'Mn': 7, 'Fe': 8, 'Co': 9, 'Ni': 10, 'Cu': 11, 'Zn': 2
        }
        return valence_map.get(symbol.capitalize(), 4)
    
    def _build_julia_script(self, system: LatticeSystem) -> str:
        """构建Julia计算脚本"""
        
        # 准备赝势文件列表
        psp_files = [self._get_psp_file(sym) for sym in system.atomic_symbols]
        
        script = f'''
using DFTK
using JSON
using LinearAlgebra

# 设置线程数
ENV["JULIA_NUM_THREADS"] = "{self.config.n_threads}"

function main()
    # 晶格参数
    lattice = {system.lattice.tolist()}  # Bohr
    lattice = hcat(lattice...)'
    
    # 元素和位置
    symbols = {system.atomic_symbols}
    positions = {system.positions.tolist()}
    
    # 创建元素
    atoms = [ElementPsp(Symbol(s), psp=load_psp("{psp}")) 
             for (s, psp) in zip(symbols, {psp_files})]
    
    # 构建模型
    model = model_DFT(lattice, atoms, positions, "{self.config.functional}")
    
    # 创建基组
    basis = PlaneWaveBasis(model; 
                          Ecut={self.config.ecut},
                          kgrid={self.config.kgrid},
                          kshift={self.config.kshift})
    
    # SCF计算
    scfres = self_consistent_field(basis;
                                   tol={self.config.scf_tol},
                                   maxiter={self.config.scf_maxiter},
                                   mixing=SimpleMixing(β={self.config.mixing_beta}))
    
    # 收集结果
    result = Dict(
        "energy" => scfres.energies.total,
        "kinetic" => float(scfres.energies.kinetic),
        "atomic_local" => float(scfres.energies.atomic_local),
        "atomic_nonlocal" => float(scfres.energies.atomic_nonlocal),
        "E_values" => float(scfres.energies.Electrostatic),
        "xc" => float(scfres.energies.xc),
        "converged" => scfres.converged,
        "n_iter" => scfres.n_iter,
        "εF" => scfres.εF
    )
    
    return JSON.json(result)
end

println(main())
'''
        return script
    
    def compute_energy(self, system: LatticeSystem) -> Dict[str, Any]:
        """
        执行DFT能量计算
        
        Args:
            system: 晶格系统定义
            
        Returns:
            包含能量和SCF信息的字典
        """
        script = self._build_julia_script(system)
        
        # 写入临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jl', delete=False) as f:
            f.write(script)
            temp_file = f.name
        
        try:
            # 执行Julia脚本
            result = subprocess.run(
                ['julia', temp_file],
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )
            
            if result.returncode != 0:
                print(f"Julia错误: {result.stderr}")
                raise RuntimeError(f"DFTK计算失败: {result.stderr}")
            
            # 解析结果
            output_lines = result.stdout.strip().split('\n')
            json_output = output_lines[-1]  # 最后一行是JSON结果
            
            return json.loads(json_output)
            
        finally:
            os.unlink(temp_file)
    
    def compute_forces(self, system: LatticeSystem) -> np.ndarray:
        """
        计算原子力 (使用自动微分)
        
        Args:
            system: 晶格系统定义
            
        Returns:
            原子力数组 (N, 3)
        """
        script = f'''
using DFTK
using Zygote
using JSON
using LinearAlgebra

function compute_forces(lattice, positions, symbols)
    atoms = [ElementPsp(Symbol(s), psp=load_psp("{self._get_psp_file(s)}")) 
             for s in symbols]
    
    model = model_DFT(lattice, atoms, positions, "{self.config.functional}")
    basis = PlaneWaveBasis(model; Ecut={self.config.ecut}, kgrid={self.config.kgrid})
    
    # 能量函数
    function energy_func(pos)
        scfres = self_consistent_field(basis; 
                                       positions=pos,
                                       tol={self.config.scf_tol},
                                       maxiter=50)
        return scfres.energies.total
    end
    
    # 使用Zygote计算梯度
    forces = -Zygote.gradient(energy_func, positions)[1]
    
    return forces
end

lattice = hcat({system.lattice.tolist()}...)
positions = {system.positions.tolist()}
symbols = {system.atomic_symbols}

forces = compute_forces(lattice, positions, symbols)
println(JSON.json(forces))
'''
        # 执行并返回结果
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jl', delete=False) as f:
            f.write(script)
            temp_file = f.name
        
        try:
            result = subprocess.run(
                ['julia', temp_file],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode != 0:
                print(f"Julia错误: {result.stderr}")
                return None
            
            output_lines = result.stdout.strip().split('\n')
            return np.array(json.loads(output_lines[-1]))
            
        finally:
            os.unlink(temp_file)


class DFTKAutodiff:
    """
    DFTK自动微分接口
    
    利用Julia的Zygote/ForwardDiff实现DFT的端到端自动微分
    """
    
    def __init__(self, config: DFTKConfig):
        self.config = config
        self.dftk = DFTKInterface(config)
    
    def energy_gradient_positions(self, system: LatticeSystem) -> np.ndarray:
        """
        能量对位置的梯度 (即原子力的负值)
        
        Args:
            system: 晶格系统
            
        Returns:
            梯度数组 (N, 3)
        """
        return self.dftk.compute_forces(system)
    
    def energy_gradient_lattice(self, system: LatticeSystem) -> np.ndarray:
        """
        能量对晶格向量的梯度 (用于应力计算)
        
        Args:
            system: 晶格系统
            
        Returns:
            晶格梯度 (3, 3)
        """
        script = f'''
using DFTK
using Zygote
using JSON
using LinearAlgebra

function compute_lattice_gradient(lattice, positions, symbols)
    atoms = [ElementPsp(Symbol(s), psp=load_psp("{self.dftk._get_psp_file(s)}")) 
             for s in symbols]
    
    # 能量对晶格的函数
    function energy_func(lat)
        model = model_DFT(lat, atoms, positions, "{self.config.functional}")
        basis = PlaneWaveBasis(model; Ecut={self.config.ecut}, kgrid={self.config.kgrid})
        scfres = self_consistent_field(basis; tol={self.config.scf_tol}, maxiter=50)
        return scfres.energies.total
    end
    
    # 使用Zygote计算梯度
    grad_lat = Zygote.gradient(energy_func, lattice)[1]
    
    return grad_lat
end

lattice = hcat({system.lattice.tolist()}...)
positions = {system.positions.tolist()}
symbols = {system.atomic_symbols}

grad = compute_lattice_gradient(lattice, positions, symbols)
println(JSON.json(grad))
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jl', delete=False) as f:
            f.write(script)
            temp_file = f.name
        
        try:
            result = subprocess.run(
                ['julia', temp_file],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode != 0:
                print(f"Julia错误: {result.stderr}")
                return None
            
            output_lines = result.stdout.strip().split('\n')
            return np.array(json.loads(output_lines[-1]))
            
        finally:
            os.unlink(temp_file)


class GeometryGradientFlow:
    """
    几何优化梯度流
    
    使用DFTK的自动微分进行结构优化
    """
    
    def __init__(self, dftk_interface: DFTKInterface, 
                 learning_rate: float = 0.01,
                 max_steps: int = 100):
        self.dftk = dftk_interface
        self.lr = learning_rate
        self.max_steps = max_steps
        
    def optimize_positions(self, 
                          initial_system: LatticeSystem,
                          fixed_lattice: bool = True) -> List[Dict]:
        """
        使用梯度下降优化原子位置
        
        Args:
            initial_system: 初始系统配置
            fixed_lattice: 是否固定晶格
            
        Returns:
            优化轨迹列表
        """
        trajectory = []
        current_system = initial_system
        
        for step in range(self.max_steps):
            # 计算能量和力
            energy_result = self.dftk.compute_energy(current_system)
            forces = self.dftk.compute_forces(current_system)
            
            # 记录当前状态
            trajectory.append({
                'step': step,
                'energy': energy_result['energy'],
                'positions': current_system.positions.copy(),
                'max_force': np.max(np.abs(forces))
            })
            
            # 检查收敛
            if np.max(np.abs(forces)) < 1e-3:
                print(f"收敛于第 {step} 步")
                break
            
            # 梯度下降更新位置
            new_positions = current_system.positions + self.lr * forces
            
            # 确保位置在晶胞内
            new_positions = new_positions % 1.0
            
            current_system = LatticeSystem(
                lattice=current_system.lattice,
                positions=new_positions,
                atomic_symbols=current_system.atomic_symbols
            )
        
        return trajectory
    
    def optimize_cell(self, 
                     initial_system: LatticeSystem,
                     target_pressure: float = 0.0) -> List[Dict]:
        """
        使用梯度流优化晶胞参数 (等压优化)
        
        Args:
            initial_system: 初始系统配置
            target_pressure: 目标压力 (GPa)
            
        Returns:
            优化轨迹列表
        """
        trajectory = []
        current_system = initial_system
        
        # 转换压力单位: GPa -> Ha/Bohr^3
        target_pressure_ha = target_pressure * 1e9 / 2.942e13  # 近似转换
        
        for step in range(self.max_steps):
            # 计算能量
            energy_result = self.dftk.compute_energy(current_system)
            
            # 计算晶格梯度 (应力)
            lattice_grad = self.dftk.compute_lattice_gradient(current_system)
            
            # 体积
            volume = np.abs(np.linalg.det(current_system.lattice))
            
            # 应力张量
            stress = -lattice_grad @ current_system.lattice.T / volume
            
            # 压力 = -1/3 * trace(stress)
            pressure = -np.trace(stress) / 3
            
            trajectory.append({
                'step': step,
                'energy': energy_result['energy'],
                'volume': volume,
                'pressure': pressure,
                'lattice': current_system.lattice.copy()
            })
            
            # 收敛检查
            if abs(pressure - target_pressure_ha) < 1e-4:
                print(f"压力收敛于第 {step} 步")
                break
            
            # 更新晶格 (保持形状，只改变体积)
            stress_correction = stress + target_pressure_ha * np.eye(3)
            lattice_update = self.lr * stress_correction @ current_system.lattice
            
            new_lattice = current_system.lattice + lattice_update
            
            current_system = LatticeSystem(
                lattice=new_lattice,
                positions=current_system.positions,
                atomic_symbols=current_system.atomic_symbols
            )
        
        return trajectory


class BandStructureCalculator:
    """
    能带结构计算
    """
    
    def __init__(self, dftk_interface: DFTKInterface):
        self.dftk = dftk_interface
    
    def calculate_band_gap(self, system: LatticeSystem) -> Dict[str, float]:
        """
        计算直接和间接带隙
        
        Args:
            system: 晶格系统
            
        Returns:
            带隙信息字典
        """
        script = f'''
using DFTK
using JSON

lattice = hcat({system.lattice.tolist()}...)
positions = {system.positions.tolist()}
symbols = {system.atomic_symbols}

atoms = [ElementPsp(Symbol(s), psp=load_psp("{self.dftk._get_psp_file(s)}")) 
         for s in symbols]

model = model_DFT(lattice, atoms, positions, "{self.dftk.config.functional}")

# 使用更密集的k点网格进行带隙计算
basis = PlaneWaveBasis(model; 
                      Ecut={self.dftk.config.ecut},
                      kgrid=(8, 8, 8))

scfres = self_consistent_field(basis; tol=1e-8)

# 计算带隙
eps = scfres.eigenvalues
occ = scfres.occupation

# 找到HOMO和LUMO
homo = maximum([maximum(eps[k][occ[k] .> 0.5]) for k in 1:length(eps)])
lumo = minimum([minimum(eps[k][occ[k] .< 0.5]) for k in 1:length(eps)])

gap = lumo - homo

# 找到直接带隙位置
min_direct_gap = Inf
for k in 1:length(eps)
    valence = maximum(eps[k][occ[k] .> 0.5])
    conduction = minimum(eps[k][occ[k] .< 0.5])
    direct_gap = conduction - valence
    if direct_gap < min_direct_gap
        min_direct_gap = direct_gap
    end
end

result = Dict(
    "indirect_gap" => gap,
    "direct_gap" => min_direct_gap,
    "homo" => homo,
    "lumo" => lumo,
    "fermi" => scfres.εF
)

println(JSON.json(result))
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jl', delete=False) as f:
            f.write(script)
            temp_file = f.name
        
        try:
            result = subprocess.run(
                ['julia', temp_file],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode != 0:
                print(f"Julia错误: {result.stderr}")
                return None
            
            output_lines = result.stdout.strip().split('\n')
            return json.loads(output_lines[-1])
            
        finally:
            os.unlink(temp_file)


def example_usage():
    """使用示例"""
    print("=" * 60)
    print("DFTK.jl 接口示例")
    print("=" * 60)
    
    # 配置
    config = DFTKConfig(
        ecut=15.0,
        kgrid=(4, 4, 4),
        functional='lda_xc_teter93',
        scf_tol=1e-6,
        n_threads=4
    )
    
    # 创建接口
    dftk = DFTKInterface(config)
    
    # 定义硅晶胞
    a = 10.26  # Bohr (硅晶格常数)
    lattice = np.array([
        [a/2, a/2, 0],
        [a/2, 0, a/2],
        [0, a/2, a/2]
    ])
    
    # 硅的原子位置 (金刚石结构，分数坐标)
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25]
    ])
    
    system = LatticeSystem(
        lattice=lattice,
        positions=positions,
        atomic_symbols=['Si', 'Si']
    )
    
    print("\n系统信息:")
    print(f"  晶格常数: {a:.3f} Bohr")
    print(f"  原子数: {len(positions)}")
    print(f"  元素: {system.atomic_symbols}")
    
    # 计算能量
    print("\n开始DFTK SCF计算...")
    try:
        result = dftk.compute_energy(system)
        print(f"总能量: {result['energy']:.6f} Ha")
        print(f"SCF迭代: {result['n_iter']} 步")
        print(f"收敛状态: {result['converged']}")
        
        # 计算带隙
        print("\n计算带隙...")
        band_calc = BandStructureCalculator(dftk)
        gap_info = band_calc.calculate_band_gap(system)
        if gap_info:
            print(f"间接带隙: {gap_info['indirect_gap']*27.2114:.3f} eV")
            print(f"直接带隙: {gap_info['direct_gap']*27.2114:.3f} eV")
        
    except Exception as e:
        print(f"计算错误: {e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    example_usage()
