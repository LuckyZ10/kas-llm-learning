"""
JAX-DFT Interface Module
========================

自动微分密度泛函理论接口，提供与JAX生态的无缝集成。
支持密度泛函的自动微分、力/应力的解析梯度计算。

核心功能：
- DFT能量的自动微分计算
- 原子力的解析梯度
- 应力张量的自动微分
- 与神经网络的混合微分
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax.experimental import optimizers
from functools import partial
from typing import Callable, Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class DFTConfig:
    """DFT计算配置"""
    xc_functional: str = 'lda_x+lda_c_pw'  # 交换关联泛函
    k_points: Tuple[int, int, int] = (4, 4, 4)  # K点网格
    grid_spacing: float = 0.16  # 实空间网格间距 (Bohr)
    ecut: float = 30.0  # 平面波截断能 (Ha)
    spin_polarized: bool = False  # 是否自旋极化
    temperature: float = 0.0  # 费米-狄拉克展宽温度
    

@dataclass
class SystemConfig:
    """系统配置"""
    positions: jnp.ndarray  # 原子位置 (N, 3)
    atomic_numbers: jnp.ndarray  # 原子序数 (N,)
    cell: jnp.ndarray  # 晶胞矩阵 (3, 3)
    pbc: bool = True  # 周期性边界条件


class JAXGrid:
    """JAX实现的实空间网格"""
    
    def __init__(self, cell: jnp.ndarray, spacing: float):
        """
        初始化实空间网格
        
        Args:
            cell: 晶胞矩阵 (3, 3)，单位 Bohr
            spacing: 网格间距，单位 Bohr
        """
        self.cell = cell
        self.spacing = spacing
        
        # 计算网格尺寸
        cell_lengths = jnp.linalg.norm(cell, axis=1)
        self.grid_sizes = jnp.ceil(cell_lengths / spacing).astype(jnp.int32)
        self.grid_sizes = jnp.maximum(self.grid_sizes, 1)
        
        # 实际网格间距
        self.actual_spacing = cell_lengths / self.grid_sizes
        
        # 创建网格
        self._create_grid()
        
    def _create_grid(self):
        """创建实空间网格点"""
        nx, ny, nz = self.grid_sizes
        
        # 生成网格索引
        x = jnp.arange(nx) / nx
        y = jnp.arange(ny) / ny
        z = jnp.arange(nz) / nz
        
        # 网格点在分数坐标
        grid_frac = jnp.stack(jnp.meshgrid(x, y, z, indexing='ij'), axis=-1)
        self.grid_frac = grid_frac.reshape(-1, 3)
        
        # 转换到笛卡尔坐标
        self.grid_cart = self.grid_frac @ self.cell
        
        # 网格体积
        self.dV = jnp.abs(jnp.linalg.det(self.cell)) / (nx * ny * nz)
        self.n_points = nx * ny * nz
        
    def get_grid_shape(self) -> Tuple[int, int, int]:
        """获取网格形状"""
        return tuple(self.grid_sizes.tolist())


class Pseudopotential:
    """赝势基类"""
    
    def __init__(self, atomic_number: int):
        self.Z = atomic_number
        self.symbol = self._get_element_symbol(atomic_number)
        
    def _get_element_symbol(self, Z: int) -> str:
        """获取元素符号"""
        symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                   'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                   'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
        if Z <= len(symbols):
            return symbols[Z - 1]
        return f'Z{Z}'
    
    def local_potential(self, r: jnp.ndarray) -> jnp.ndarray:
        """
        局域赝势
        
        Args:
            r: 距离数组
            
        Returns:
            局域势值
        """
        raise NotImplementedError
    
    def projector(self, r: jnp.ndarray, l: int) -> jnp.ndarray:
        """
        非局域投影函数
        
        Args:
            r: 距离数组
            l: 角动量量子数
            
        Returns:
            投影函数值
        """
        raise NotImplementedError


class HGHPseudo(Pseudopotential):
    """
    Hartwigsen-Goedecker-Hutter 赝势
    解析形式，适合自动微分
    """
    
    def __init__(self, atomic_number: int):
        super().__init__(atomic_number)
        self._load_parameters()
        
    def _load_parameters(self):
        """加载HGH参数"""
        # HGH赝势参数 (简化版本)
        hgh_params = {
            1: {'rloc': 0.2, 'c1': -4.17890044, 'c2': 0.72446331, 'c3': 0.0, 'c4': 0.0},
            6: {'rloc': 0.34, 'c1': -8.82982873, 'c2': 1.16959695, 'c3': 0.0, 'c4': 0.0},
            8: {'rloc': 0.28, 'c1': -15.87578969, 'c2': 2.09845195, 'c3': 0.0, 'c4': 0.0},
            14: {'rloc': 0.44, 'c1': -7.33610309, 'c2': 0.0, 'c3': 0.0, 'c4': 0.0},
        }
        
        if self.Z in hgh_params:
            params = hgh_params[self.Z]
            self.rloc = params['rloc']
            self.c = jnp.array([params['c1'], params['c2'], 
                               params['c3'], params['c4']])
        else:
            # 默认参数
            self.rloc = 0.5
            self.c = jnp.array([-5.0, 0.0, 0.0, 0.0])
    
    def local_potential(self, r: jnp.ndarray) -> jnp.ndarray:
        """
        HGH局域赝势
        V_loc(r) = -Z_ion/r * erf(r/sqrt(2)/rloc) + 
                   exp(-r^2/2/rloc^2) * (C1 + C2*r^2/rloc^2 + ...)
        """
        Z_ion = float(self.Z)  # 简化：核电荷 = 原子序数
        
        # 长程部分 (误差函数)
        x = r / (jnp.sqrt(2.0) * self.rloc)
        erf_part = -Z_ion / r * jax.scipy.special.erf(x)
        
        # 短程部分 (高斯展开)
        r2_rloc2 = (r / self.rloc) ** 2
        gaussian = jnp.exp(-0.5 * r2_rloc2)
        
        poly = (self.c[0] + 
                self.c[1] * r2_rloc2 + 
                self.c[2] * r2_rloc2 ** 2 + 
                self.c[3] * r2_rloc2 ** 3)
        
        # 避免r=0发散
        short_range = gaussian * poly
        
        return jnp.where(r > 1e-10, erf_part + short_range, 
                        -Z_ion / (jnp.sqrt(2 * jnp.pi) * self.rloc) + self.c[0])


class LDAFunctional:
    """
    局域密度近似 (LDA) 交换关联泛函
    使用PW92参数化
    """
    
    @staticmethod
    @jit
def exchange_energy_density(n: jnp.ndarray) -> jnp.ndarray:
        """
        LDA交换能密度
        
        eps_x = -3/4 * (3/pi)^(1/3) * n^(1/3)
        """
        # 添加小值避免数值问题
        n_safe = jnp.maximum(n, 1e-20)
        
        Cx = -0.75 * (3.0 / jnp.pi) ** (1.0/3.0)
        return Cx * n_safe ** (4.0/3.0)
    
    @staticmethod
    @jit
    def correlation_energy_density(n: jnp.ndarray) -> jnp.ndarray:
        """
        PW92 LDA关联能密度
        """
        n_safe = jnp.maximum(n, 1e-20)
        rs = (3.0 / (4.0 * jnp.pi * n_safe)) ** (1.0/3.0)
        
        # PW92参数 (非极化)
        A = 0.031091
        alpha1 = 0.21370
        beta1 = 7.5957
        beta2 = 3.5876
        beta3 = 1.6382
        beta4 = 0.49294
        
        # 关联能密度
        sqrt_rs = jnp.sqrt(rs)
        denom = 1.0 + beta1*sqrt_rs + beta2*rs + beta3*rs*sqrt_rs + beta4*rs*rs
        
        eps_c = -2.0 * A * (1.0 + alpha1*rs) * jnp.log(1.0 + 1.0 / (2.0 * A * denom))
        
        return n_safe * eps_c
    
    @classmethod
    @jit
    def energy_density(cls, n: jnp.ndarray) -> jnp.ndarray:
        """总交换关联能密度"""
        return cls.exchange_energy_density(n) + cls.correlation_energy_density(n)
    
    @classmethod
    @jit
    def potential(cls, n: jnp.ndarray) -> jnp.ndarray:
        """
        交换关联势 (V_xc = dE_xc/dn)
        使用自动微分计算
        """
        eps_xc = lambda density: jnp.sum(cls.energy_density(density))
        return grad(eps_xc)(n)


class DifferentiableDFT:
    """
    可微分DFT计算器
    
    核心类，提供自动微分功能：
    - 能量对位置的梯度 → 原子力
    - 能量对晶胞的梯度 → 应力
    """
    
    def __init__(self, config: DFTConfig):
        self.config = config
        self.grid = None
        self.pseudos = {}
        self._setup_functional()
        
    def _setup_functional(self):
        """设置交换关联泛函"""
        if 'lda' in self.config.xc_functional.lower():
            self.xc_functional = LDAFunctional()
        else:
            raise ValueError(f"不支持的泛函: {self.config.xc_functional}")
    
    def _get_pseudopotential(self, Z: int) -> Pseudopotential:
        """获取赝势"""
        if Z not in self.pseudos:
            self.pseudos[Z] = HGHPseudo(Z)
        return self.pseudos[Z]
    
    @partial(jit, static_argnums=(0,))
    def _compute_local_potential(self, positions: jnp.ndarray, 
                                  atomic_numbers: jnp.ndarray,
                                  cell: jnp.ndarray) -> jnp.ndarray:
        """
        计算局域赝势产生的有效势
        
        Args:
            positions: 原子位置 (N, 3)
            atomic_numbers: 原子序数 (N,)
            cell: 晶胞矩阵 (3, 3)
            
        Returns:
            网格上的局域势 (n_grid,)
        """
        if self.grid is None or not jnp.allclose(self.grid.cell, cell):
            self.grid = JAXGrid(cell, self.config.grid_spacing)
        
        grid_points = self.grid.grid_cart  # (n_grid, 3)
        n_grid = grid_points.shape[0]
        
        # 初始化局域势
        V_local = jnp.zeros(n_grid)
        
        # 每个原子贡献的局域势
        for i in range(len(positions)):
            Z = atomic_numbers[i]
            pseudo = self._get_pseudopotential(int(Z))
            
            # 计算到所有网格点的距离 (考虑周期性)
            ri = positions[i]
            dr = grid_points - ri[None, :]  # (n_grid, 3)
            
            # 应用周期性边界条件
            if self.config.pbc:
                # 转换到分数坐标，应用最小镜像约定
                frac_dr = dr @ jnp.linalg.inv(cell)
                frac_dr = frac_dr - jnp.rint(frac_dr)
                dr = frac_dr @ cell
            
            r = jnp.linalg.norm(dr, axis=1)
            
            # 添加局域势
            V_local = V_local + pseudo.local_potential(r)
        
        return V_local
    
    @partial(jit, static_argnums=(0,))
    def _compute_hartree_potential(self, density: jnp.ndarray,
                                    cell: jnp.ndarray) -> jnp.ndarray:
        """
        计算Hartree势 (泊松方程求解)
        使用FFT方法
        
        Args:
            density: 电子密度在网格上 (n_grid,)
            cell: 晶胞矩阵
            
        Returns:
            Hartree势 (n_grid,)
        """
        # 重新塑形为3D网格
        nx, ny, nz = self.grid.grid_sizes
        density_3d = density.reshape(nx, ny, nz)
        
        # FFT到倒空间
        density_G = jnp.fft.fftn(density_3d)
        
        # 倒空间格点
        recip_cell = 2 * jnp.pi * jnp.linalg.inv(cell.T)
        
        # 生成倒空间网格
        gx = jnp.fft.fftfreq(nx).reshape(-1, 1, 1) * recip_cell[0, 0] * nx
        gy = jnp.fft.fftfreq(ny).reshape(1, -1, 1) * recip_cell[1, 1] * ny
        gz = jnp.fft.fftfreq(nz).reshape(1, 1, -1) * recip_cell[2, 2] * nz
        
        G2 = gx**2 + gy**2 + gz**2
        
        # 避免G=0发散
        G2_safe = jnp.where(G2 < 1e-10, 1.0, G2)
        
        # Hartree势在倒空间: V_H(G) = 4π * n(G) / G^2
        V_H_G = 4.0 * jnp.pi * density_G / G2_safe
        V_H_G = jnp.where(G2 < 1e-10, 0.0, V_H_G)
        
        # FFT回实空间
        V_H = jnp.fft.ifftn(V_H_G).real
        
        return V_H.flatten()
    
    @partial(jit, static_argnums=(0,))
    def _compute_kinetic_energy(self, density: jnp.ndarray,
                                 cell: jnp.ndarray) -> float:
        """
        计算Thomas-Fermi动能 (简化模型)
        
        E_kin = ∫ (3/10)(3π²)^(2/3) n^(5/3) dr
        """
        n_safe = jnp.maximum(density, 1e-20)
        
        C_k = 2.871  # Thomas-Fermi常数
        e_kin_density = C_k * n_safe ** (5.0/3.0)
        
        return jnp.sum(e_kin_density) * self.grid.dV
    
    @partial(jit, static_argnums=(0,))
    def _compute_external_energy(self, density: jnp.ndarray,
                                  V_ext: jnp.ndarray) -> float:
        """
        计算外场能量
        E_ext = ∫ V_ext(r) * n(r) dr
        """
        return jnp.sum(V_ext * density) * self.grid.dV
    
    @partial(jit, static_argnums=(0,))
    def _compute_hartree_energy(self, density: jnp.ndarray,
                                 V_H: jnp.ndarray) -> float:
        """
        计算Hartree能量
        E_H = 1/2 ∫ V_H(r) * n(r) dr
        """
        return 0.5 * jnp.sum(V_H * density) * self.grid.dV
    
    @partial(jit, static_argnums=(0,))
    def _compute_xc_energy(self, density: jnp.ndarray) -> float:
        """
        计算交换关联能量
        E_xc = ∫ eps_xc(n) dr
        """
        exc_density = self.xc_functional.energy_density(density)
        return jnp.sum(exc_density) * self.grid.dV
    
    @partial(jit, static_argnums=(0,))
    def _self_consistent_field(self, initial_density: jnp.ndarray,
                                V_local: jnp.ndarray,
                                cell: jnp.ndarray,
                                max_iter: int = 100,
                                tol: float = 1e-6) -> Tuple[jnp.ndarray, Dict]:
        """
        自洽场迭代
        
        Args:
            initial_density: 初始电子密度
            V_local: 局域赝势
            cell: 晶胞矩阵
            max_iter: 最大迭代次数
            tol: 收敛容差
            
        Returns:
            (收敛后的密度, 收敛信息)
        """
        density = initial_density
        
        def scf_step(carry, _):
            rho = carry
            
            # 计算Hartree势
            V_H = self._compute_hartree_potential(rho, cell)
            
            # 计算交换关联势
            V_xc = self.xc_functional.potential(rho)
            
            # 总Kohn-Sham有效势
            V_eff = V_local + V_H + V_xc
            
            # 简化：直接更新密度 (实际应用需要求解KS方程)
            # 这里使用简单的混合策略
            new_rho = jnp.maximum(-V_eff / (2 * jnp.pi), 1e-10)
            
            # 归一化
            total_electrons = jnp.sum(new_rho) * self.grid.dV
            target_electrons = jnp.sum(initial_density) * self.grid.dV
            new_rho = new_rho * target_electrons / total_electrons
            
            # 混合
            mixing = 0.3
            rho_new = mixing * new_rho + (1 - mixing) * rho
            
            # 计算变化
            delta = jnp.linalg.norm(rho_new - rho) / jnp.linalg.norm(rho)
            
            return rho_new, delta
        
        # 使用scan进行迭代
        density, deltas = jax.lax.scan(scf_step, density, None, length=max_iter)
        
        converged = deltas[-1] < tol
        
        info = {
            'converged': converged,
            'final_delta': deltas[-1],
            'n_iter': max_iter
        }
        
        return density, info
    
    def compute_energy(self, system: SystemConfig,
                       return_forces: bool = False) -> Dict[str, Any]:
        """
        计算DFT能量
        
        Args:
            system: 系统配置
            return_forces: 是否计算力
            
        Returns:
            包含能量和可选力的字典
        """
        # 初始化网格
        self.grid = JAXGrid(system.cell, self.config.grid_spacing)
        
        # 计算局域势
        V_local = self._compute_local_potential(
            system.positions, system.atomic_numbers, system.cell
        )
        
        # 初始化电子密度 (叠加原子密度近似)
        n_electrons = jnp.sum(system.atomic_numbers)
        initial_density = jnp.ones(self.grid.n_points) * n_electrons / (
            jnp.abs(jnp.linalg.det(system.cell))
        )
        
        # 自洽场计算
        density, scf_info = self._self_consistent_field(
            initial_density, V_local, system.cell
        )
        
        # 计算各项能量
        V_H = self._compute_hartree_potential(density, system.cell)
        
        E_kin = self._compute_kinetic_energy(density, system.cell)
        E_ext = self._compute_external_energy(density, V_local)
        E_hartree = self._compute_hartree_energy(density, V_H)
        E_xc = self._compute_xc_energy(density)
        
        # 总能量
        E_total = E_kin + E_ext + E_hartree + E_xc
        
        result = {
            'energy': E_total,
            'energy_components': {
                'kinetic': E_kin,
                'external': E_ext,
                'hartree': E_hartree,
                'xc': E_xc
            },
            'density': density,
            'scf_info': scf_info
        }
        
        # 计算力 (如果需要)
        if return_forces:
            forces = self.compute_forces(system)
            result['forces'] = forces
        
        return result
    
    def compute_forces(self, system: SystemConfig) -> jnp.ndarray:
        """
        计算原子力 (Hellmann-Feynman + Pulay修正)
        
        使用自动微分计算能量对位置的梯度
        
        Args:
            system: 系统配置
            
        Returns:
            原子力 (N, 3)
        """
        def energy_fn(pos):
            sys = SystemConfig(
                positions=pos,
                atomic_numbers=system.atomic_numbers,
                cell=system.cell,
                pbc=system.pbc
            )
            result = self.compute_energy(sys, return_forces=False)
            return result['energy']
        
        # 使用JAX自动微分
        forces = -grad(energy_fn)(system.positions)
        
        return forces
    
    def compute_stress(self, system: SystemConfig) -> jnp.ndarray:
        """
        计算应力张量
        
        使用自动微分计算能量对晶胞的梯度
        
        Args:
            system: 系统配置
            
        Returns:
            应力张量 (3, 3)
        """
        def energy_fn(cell):
            sys = SystemConfig(
                positions=system.positions,
                atomic_numbers=system.atomic_numbers,
                cell=cell,
                pbc=system.pbc
            )
            result = self.compute_energy(sys, return_forces=False)
            return result['energy']
        
        # 应力 = -1/V * dE/depsilon (epsilon为应变)
        volume = jnp.abs(jnp.linalg.det(system.cell))
        
        # 计算能量对晶胞的梯度
        dE_dcell = grad(energy_fn)(system.cell)
        
        # 转换为应力
        # sigma_ij = -1/V * sum_k dE/dh_ik * h_jk
        stress = -dE_dcell @ system.cell.T / volume
        
        return stress


class NeuralDFT:
    """
    神经网络增强的DFT
    
    将机器学习势与DFT结合，实现混合能量计算
    """
    
    def __init__(self, 
                 dft_calculator: DifferentiableDFT,
                 neural_network: Callable,
                 mixing: float = 0.5):
        """
        Args:
            dft_calculator: DFT计算器
            neural_network: 神经网络势函数
            mixing: DFT与NN的混合比例
        """
        self.dft = dft_calculator
        self.nn = neural_network
        self.mixing = mixing
    
    def compute_energy(self, system: SystemConfig) -> Dict[str, Any]:
        """
        计算混合能量
        E = mixing * E_DFT + (1-mixing) * E_NN
        """
        # DFT能量
        dft_result = self.dft.compute_energy(system)
        E_dft = dft_result['energy']
        
        # 神经网络能量
        E_nn = self.nn(system.positions, system.atomic_numbers, system.cell)
        
        # 混合
        E_total = self.mixing * E_dft + (1 - self.mixing) * E_nn
        
        return {
            'energy': E_total,
            'energy_dft': E_dft,
            'energy_nn': E_nn,
            'density': dft_result.get('density')
        }
    
    def compute_forces(self, system: SystemConfig) -> jnp.ndarray:
        """
        计算混合力 (自动微分)
        """
        def energy_fn(pos):
            sys = SystemConfig(
                positions=pos,
                atomic_numbers=system.atomic_numbers,
                cell=system.cell,
                pbc=system.pbc
            )
            result = self.compute_energy(sys)
            return result['energy']
        
        return -grad(energy_fn)(system.positions)


def example_usage():
    """使用示例"""
    print("=" * 60)
    print("JAX-DFT 接口示例")
    print("=" * 60)
    
    # 创建DFT配置
    config = DFTConfig(
        xc_functional='lda_x+lda_c_pw',
        grid_spacing=0.2,
        ecut=30.0
    )
    
    # 创建DFT计算器
    dft = DifferentiableDFT(config)
    
    # 定义简单系统 (硅晶胞)
    a = 10.26  # Bohr
    cell = jnp.array([
        [a/2, a/2, 0],
        [a/2, 0, a/2],
        [0, a/2, a/2]
    ])
    
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [a/4, a/4, a/4]
    ]) @ jnp.linalg.inv(cell) @ cell  # 分数坐标转换
    
    # 归一化到晶胞内
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25]
    ]) @ cell
    
    atomic_numbers = jnp.array([14, 14])  # Si
    
    system = SystemConfig(
        positions=positions,
        atomic_numbers=atomic_numbers,
        cell=cell,
        pbc=True
    )
    
    print("\n系统信息:")
    print(f"  原子数: {len(positions)}")
    print(f"  晶胞:\n{cell}")
    print(f"  原子序数: {atomic_numbers}")
    
    # 计算能量
    print("\n开始DFT计算...")
    result = dft.compute_energy(system)
    
    print(f"\n总能量: {result['energy']:.6f} Ha")
    print("\n能量分解:")
    for key, val in result['energy_components'].items():
        print(f"  {key}: {val:.6f} Ha")
    
    print(f"\nSCF收敛: {result['scf_info']['converged']}")
    print(f"最终变化: {result['scf_info']['final_delta']:.2e}")
    
    # 计算力
    print("\n计算原子力 (使用自动微分)...")
    forces = dft.compute_forces(system)
    print(f"原子力:\n{forces}")
    
    # 计算应力
    print("\n计算应力张量...")
    stress = dft.compute_stress(system)
    print(f"应力张量:\n{stress}")
    
    print("\n" + "=" * 60)
    print("计算完成!")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()
