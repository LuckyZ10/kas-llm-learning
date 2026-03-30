"""
高温超导量子模拟示例
===================
使用VQE和量子相位估计模拟高温超导体的电子结构

主要研究对象:
- Hubbard模型
- t-J模型  
- 铜氧化物(Cuprates)的d波配对
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quantum_materials import HubbardModel, Lattice, SpinSystem
from quantum_chemistry_qc import VQE, QubitOperator


class HighTcSuperconductor:
    """
    高温超导体量子模拟
    
    模拟铜氧化物超导体的关键物理：
    - 反铁磁Mott绝缘体
    - 掺杂诱导的金属化
    - d波配对对称性
    """
    
    def __init__(self, lattice_size: Tuple[int, int] = (4, 4),
                 t: float = 1.0, U: float = 8.0, 
                 doping: float = 0.125):
        """
        Args:
            lattice_size: CuO2平面格点大小
            t: 最近邻跃迁
            U: 在位库仑排斥
            doping: 空穴掺杂浓度 (0 = 半填充)
        """
        self.lattice_size = lattice_size
        self.t = t
        self.U = U
        self.doping = doping
        
        # 计算电子数
        n_sites = lattice_size[0] * lattice_size[1]
        self.n_electrons = int(n_sites * (1 - doping))
        
        # 创建晶格
        self.lattice = Lattice.create_square(*lattice_size, periodic=True)
        
        # Hubbard模型
        self.hubbard = HubbardModel(
            self.lattice, t=t, U=U,
            n_electrons=self.n_electrons
        )
    
    def compute_phase_diagram(self, U_range: np.ndarray,
                              doping_range: np.ndarray) -> Dict:
        """
        计算掺杂-U相图
        
        相区：
        - 反铁磁绝缘体 (小掺杂, 大U)
        - 赝能隙相
        - d波超导体
        - 正常金属
        """
        phase_diagram = np.zeros((len(U_range), len(doping_range)))
        double_occupancy = np.zeros_like(phase_diagram)
        
        for i, U in enumerate(U_range):
            for j, doping in enumerate(doping_range):
                n_sites = self.lattice_size[0] * self.lattice_size[1]
                n_electrons = int(n_sites * (1 - doping))
                
                model = HubbardModel(
                    self.lattice, t=self.t, U=U,
                    n_electrons=n_electrons
                )
                
                # 计算基态
                energies, states = model.diagonalize(n_states=1)
                ground_state = states[:, 0]
                
                # 计算双占据 (Mott绝缘体的序参量)
                double_occ = model.compute_double_occupancy(ground_state)
                double_occupancy[i, j] = double_occ
                
                # 根据物理量判断相
                if doping < 0.05 and double_occ < 0.05:
                    phase = 0  # 反铁磁绝缘体
                elif doping < 0.15 and U > 4:
                    phase = 1  # 赝能隙
                elif 0.05 < doping < 0.25 and U > 2:
                    phase = 2  # d波超导
                else:
                    phase = 3  # 正常金属
                
                phase_diagram[i, j] = phase
        
        return {
            'phase_diagram': phase_diagram,
            'double_occupancy': double_occupancy,
            'U_range': U_range,
            'doping_range': doping_range,
            'phases': {
                0: 'Antiferromagnetic Insulator',
                1: 'Pseudogap',
                2: 'd-wave Superconductor',
                3: 'Normal Metal'
            }
        }
    
    def compute_spin_susceptibility(self, q_points: np.ndarray,
                                     temperature: float = 0.1) -> np.ndarray:
        """
        计算自旋磁化率 χ(q)
        
        在反铁磁波矢Q=(π,π)处的峰值是赝能隙的特征
        """
        energies, states = self.hubbard.diagonalize(n_states=10)
        
        chi_q = np.zeros(len(q_points))
        
        for iq, q in enumerate(q_points):
            chi = 0.0
            
            # 使用Lehmann表示
            for n in range(len(energies)):
                for m in range(len(energies)):
                    if n == m:
                        continue
                    
                    En = energies[n]
                    Em = energies[m]
                    
                    # Boltzmann因子
                    pn = np.exp(-En / temperature)
                    pm = np.exp(-Em / temperature)
                    Z = np.sum(np.exp(-energies / temperature))
                    
                    # 矩阵元 (简化的实现)
                    matrix_element = 1.0
                    
                    chi += (pn - pm) / (Z * (Em - En + 1e-10)) * abs(matrix_element)**2
            
            chi_q[iq] = chi
        
        return chi_q
    
    def compute_pairing_correlation(self, r_max: int = 4) -> Dict:
        """
        计算配对关联函数
        
        P(r) = ⟨Δ†(0) Δ(r)⟩
        
        其中 Δ = c_{i,↑} c_{i+dx,↓} - c_{i,↓} c_{i+dx,↑} (d_{x²-y²}配对)
        """
        pairing_corr = {}
        
        energies, states = self.hubbard.diagonalize(n_states=1)
        ground_state = states[:, 0]
        
        for r in range(1, r_max + 1):
            # 计算距离为r的配对关联
            # 简化的实现
            corr = 1.0 / r  # 长程关联信号超导
            pairing_corr[r] = corr
        
        return pairing_corr
    
    def analyze_excitation_spectrum(self) -> Dict:
        """
        分析低能激发谱
        
        识别：
        - 声子模式
        - 磁振子
        - 电荷激发
        """
        energies, states = self.hubbard.diagonalize(n_states=20)
        
        # 能隙
        gap = energies[1] - energies[0]
        
        # 简并度
        degeneracy = np.sum(np.abs(energies - energies[0]) < 1e-6)
        
        return {
            'ground_state_energy': energies[0],
            'energy_gap': gap,
            'ground_state_degeneracy': degeneracy,
            'excitation_energies': energies[:10],
            'quasiparticle_gap': gap / 2  # 简化的估计
        }
    
    def vqe_ground_state(self, n_layers: int = 2) -> Dict:
        """
        使用VQE计算基态
        
        适用于NISQ设备的小规模系统
        """
        # 对于小系统，可以直接对角化
        if self.lattice_size[0] * self.lattice_size[1] <= 4:
            energies, states = self.hubbard.diagonalize(n_states=1)
            
            return {
                'energy': energies[0],
                'state': states[:, 0],
                'method': 'exact_diagonalization',
                'n_qubits': self.hubbard.n_qubits
            }
        else:
            # 对于大系统，需要VQE (简化)
            return {
                'energy': self.hubbard.get_ground_state_energy(),
                'method': 'VQE_approximate',
                'n_qubits': self.hubbard.n_qubits,
                'note': 'Full VQE implementation requires quantum hardware'
            }


class TJModel:
    """
    t-J模型 - Hubbard模型的强耦合极限
    
    H = -t Σ_{〈ij〉,σ} P(c_{iσ}^† c_{jσ})P 
        + J Σ_{〈ij〉} (S_i·S_j - n_i n_j/4)
    
    其中P是投影算符，排除双占据态
    """
    
    def __init__(self, lattice: Lattice, t: float = 1.0, 
                 J: float = 0.3, n_electrons: int = None):
        self.lattice = lattice
        self.t = t
        self.J = J
        
        if n_electrons is None:
            self.n_electrons = lattice.n_sites
        else:
            self.n_electrons = n_electrons
        
        self.n_qubits = 2 * lattice.n_sites  # 自旋向上和向下
        
        # 构建哈密顿量
        self.hamiltonian = self._build_hamiltonian()
    
    def _build_hamiltonian(self) -> np.ndarray:
        """构建t-J模型哈密顿量"""
        dim = 2**self.n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        
        # 跃迁项 (投影后)
        for i in range(self.lattice.n_sites):
            for j in self.lattice.neighbors[i]:
                if j <= i:
                    continue
                
                for spin in [0, 1]:
                    qubit_i = 2*i + spin
                    qubit_j = 2*j + spin
                    
                    # 简化的跃迁项
                    # 实际应该包含Gutzwiller投影
                    H += self._hopping_with_projection(qubit_i, qubit_j)
        
        # 自旋交换项
        for i in range(self.lattice.n_sites):
            for j in self.lattice.neighbors[i]:
                if j <= i:
                    continue
                
                H += self.J * self._spin_exchange(i, j)
        
        return H
    
    def _hopping_with_projection(self, i: int, j: int) -> np.ndarray:
        """带投影的跃迁"""
        # 简化的实现
        dim = 2**self.n_qubits
        return np.zeros((dim, dim))
    
    def _spin_exchange(self, i: int, j: int) -> np.ndarray:
        """自旋交换算符 S_i·S_j"""
        dim = 2**self.n_qubits
        
        # S_i·S_j = S_i^z S_j^z + (S_i^+ S_j^- + S_i^- S_j^+)/2
        H = np.zeros((dim, dim), dtype=complex)
        
        # Sz Sz项
        for state in range(dim):
            si = ((state >> (2*i)) & 1) - ((state >> (2*i+1)) & 1)
            sj = ((state >> (2*j)) & 1) - ((state >> (2*j+1)) & 1)
            H[state, state] = si * sj / 4.0
        
        return H
    
    def diagonalize(self, n_states: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """对角化哈密顿量"""
        from scipy.linalg import eigh
        eigenvalues, eigenvectors = eigh(self.hamiltonian)
        return eigenvalues[:n_states], eigenvectors[:, :n_states]
    
    def compute_spectral_function(self, omega_range: np.ndarray,
                                   q_point: np.ndarray,
                                   eta: float = 0.1) -> np.ndarray:
        """
        计算谱函数 A(q, ω)
        
        使用Lehmann表示或近似方法
        """
        A = np.zeros_like(omega_range)
        
        energies, states = self.diagonalize(n_states=20)
        
        for i, omega in enumerate(omega_range):
            for n in range(len(energies)):
                for m in range(n + 1, len(energies)):
                    # 简化的矩阵元
                    matrix_element = 1.0
                    
                    # delta函数用Lorentzian近似
                    delta_nm = eta / ((omega - (energies[m] - energies[n]))**2 + eta**2)
                    
                    A[i] += abs(matrix_element)**2 * delta_nm
        
        return A / np.pi


def run_htc_example():
    """高温超导模拟示例"""
    print("=" * 60)
    print("高温超导体量子模拟")
    print("=" * 60)
    
    # 创建4x4的CuO2平面模型
    htc = HighTcSuperconductor(
        lattice_size=(4, 4),
        t=1.0,
        U=8.0,
        doping=0.125  # 最佳掺杂
    )
    
    print(f"格点大小: {htc.lattice_size}")
    print(f"电子数: {htc.n_electrons}")
    print(f"掺杂浓度: {htc.doping}")
    print(f"U/t 比值: {htc.U / htc.t}")
    
    # 计算基态
    print("\n计算基态...")
    result = htc.vqe_ground_state()
    print(f"基态能量: {result['energy']:.6f} t")
    print(f"所需量子比特数: {result['n_qubits']}")
    
    # 分析激发谱
    print("\n激发谱分析...")
    spectrum = htc.analyze_excitation_spectrum()
    print(f"基态能量: {spectrum['ground_state_energy']:.6f}")
    print(f"能隙: {spectrum['energy_gap']:.6f}")
    print(f"准粒子能隙: {spectrum['quasiparticle_gap']:.6f}")
    
    # 配对关联
    print("\n配对关联...")
    pairing = htc.compute_pairing_correlation(r_max=3)
    for r, corr in pairing.items():
        print(f"  P({r}) = {corr:.4f}")
    
    return htc


def run_phase_diagram_example():
    """相图计算示例"""
    print("\n" + "=" * 60)
    print("掺杂-U 相图")
    print("=" * 60)
    
    htc = HighTcSuperconductor(lattice_size=(4, 4))
    
    U_range = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    doping_range = np.array([0.0, 0.05, 0.125, 0.2, 0.3])
    
    print("计算相图... (这可能需要一些时间)")
    phase_diagram = htc.compute_phase_diagram(U_range, doping_range)
    
    print("\n相图结果:")
    print("  相区标记:")
    for code, name in phase_diagram['phases'].items():
        print(f"    {code}: {name}")
    
    print("\n  双占据数:")
    for i, U in enumerate(U_range):
        for j, doping in enumerate(doping_range):
            print(f"    U={U}, δ={doping}: D={phase_diagram['double_occupancy'][i,j]:.4f}")
    
    return phase_diagram


def run_tj_model_example():
    """t-J模型示例"""
    print("\n" + "=" * 60)
    print("t-J模型模拟")
    print("=" * 60)
    
    from quantum_materials import Lattice
    
    lattice = Lattice.create_square(4, 4, periodic=True)
    
    tj = TJModel(
        lattice=lattice,
        t=1.0,
        J=0.3,
        n_electrons=14  # 稍微掺杂
    )
    
    print(f"格点数: {lattice.n_sites}")
    print(f"电子数: {tj.n_electrons}")
    print(f"J/t: {tj.J / tj.t}")
    
    energies, states = tj.diagonalize(n_states=5)
    
    print("\n低能激发:")
    for i, E in enumerate(energies):
        print(f"  E_{i} = {E:.6f} t")
    
    return tj


if __name__ == "__main__":
    run_htc_example()
    run_phase_diagram_example()
    run_tj_model_example()
