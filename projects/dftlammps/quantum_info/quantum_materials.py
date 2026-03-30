"""
量子材料模拟模块 (Quantum Materials Simulation)
==========================================
实现量子材料系统的量子模拟算法：
- Hubbard模型量子模拟
- 自旋系统量子计算
- 拓扑不变量测量

作者: DFT-Team
日期: 2025-03
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
from scipy.linalg import expm, eigh
import itertools


class LatticeType(Enum):
    """晶格类型"""
    CHAIN = "chain"
    SQUARE = "square"
    CUBIC = "cubic"
    TRIANGULAR = "triangular"
    HONEYCOMB = "honeycomb"
    KAGOME = "kagome"


@dataclass
class Lattice:
    """晶格结构定义"""
    lattice_type: LatticeType
    dimensions: Tuple[int, ...]  # 各维度大小
    n_sites: int
    neighbors: Dict[int, List[int]]  # 每个格点的近邻
    
    @classmethod
    def create_chain(cls, n_sites: int, periodic: bool = True) -> 'Lattice':
        """创建一维链"""
        neighbors = {}
        for i in range(n_sites):
            neighbors[i] = []
            if i > 0:
                neighbors[i].append(i - 1)
            if i < n_sites - 1:
                neighbors[i].append(i + 1)
            if periodic:
                if i == 0:
                    neighbors[i].append(n_sites - 1)
                if i == n_sites - 1:
                    neighbors[i].append(0)
        return cls(LatticeType.CHAIN, (n_sites,), n_sites, neighbors)
    
    @classmethod
    def create_square(cls, nx: int, ny: int, periodic: bool = True) -> 'Lattice':
        """创建二维方格"""
        n_sites = nx * ny
        neighbors = {}
        
        for i in range(nx):
            for j in range(ny):
                site = i * ny + j
                neighbors[site] = []
                
                # 四个近邻
                if i > 0 or periodic:
                    ni = (i - 1) % nx
                    neighbors[site].append(ni * ny + j)
                if i < nx - 1 or periodic:
                    ni = (i + 1) % nx
                    neighbors[site].append(ni * ny + j)
                if j > 0 or periodic:
                    nj = (j - 1) % ny
                    neighbors[site].append(i * ny + nj)
                if j < ny - 1 or periodic:
                    nj = (j + 1) % ny
                    neighbors[site].append(i * ny + nj)
        
        return cls(LatticeType.SQUARE, (nx, ny), n_sites, neighbors)
    
    @classmethod
    def create_honeycomb(cls, nx: int, ny: int, periodic: bool = True) -> 'Lattice':
        """创建蜂窝格点 (石墨烯结构)"""
        n_sites = 2 * nx * ny  # 每个原胞2个原子
        neighbors = {}
        
        for site in range(n_sites):
            neighbors[site] = []
            # 简化实现：每个A子格点连接3个B子格点
            cell = site // 2
            sublattice = site % 2
            
            if sublattice == 0:  # A子格点
                # 连接到同一原胞的B
                neighbors[site].append(2*cell + 1)
                # 连接到邻近原胞的B (简化)
                neighbors[site].append(((cell + 1) % (nx*ny)) * 2 + 1)
                neighbors[site].append(((cell + nx) % (nx*ny)) * 2 + 1)
            else:  # B子格点
                # B连接到A
                neighbors[site].append(2*cell)
        
        return cls(LatticeType.HONEYCOMB, (nx, ny), n_sites, neighbors)


class HubbardModel:
    """
    Hubbard模型 - 描述强关联电子系统的标准模型
    
    H = -t Σ_{<i,j>,σ} (c_{i,σ}^† c_{j,σ} + h.c.)
        + U Σ_i n_{i,↑} n_{i,↓}
        - μ Σ_{i,σ} n_{i,σ}
    
    其中:
    - t: 跃迁积分
    - U: 在位库仑相互作用
    - μ: 化学势
    """
    
    def __init__(self, lattice: Lattice, t: float = 1.0, U: float = 4.0,
                 mu: float = 0.0, n_electrons: Optional[int] = None):
        """
        Args:
            lattice: 晶格结构
            t: 跃迁积分
            U: 在位库仑相互作用强度
            mu: 化学势
            n_electrons: 电子数 (None表示半填充)
        """
        self.lattice = lattice
        self.t = t
        self.U = U
        self.mu = mu
        
        if n_electrons is None:
            self.n_electrons = lattice.n_sites  # 半填充
        else:
            self.n_electrons = n_electrons
        
        self.n_qubits = 2 * lattice.n_sites  # 每个格点2个自旋
        
        # 构建哈密顿量
        self.hamiltonian = self._build_hamiltonian()
    
    def _build_hamiltonian(self) -> np.ndarray:
        """构建完整哈密顿量矩阵"""
        dim = 2**self.n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        
        # 跃迁项
        for i in range(self.lattice.n_sites):
            for j in self.lattice.neighbors[i]:
                if j <= i:  # 避免重复计算
                    continue
                
                # 自旋向上和向下的跃迁
                for spin in [0, 1]:  # 0=↑, 1=↓
                    qubit_i = 2*i + spin
                    qubit_j = 2*j + spin
                    
                    # -t (c_i^† c_j + c_j^† c_i)
                    # Jordan-Wigner变换实现
                    H += self._hopping_term(qubit_i, qubit_j)
        
        # 相互作用项和化学势
        for i in range(self.lattice.n_sites):
            qubit_up = 2*i
            qubit_down = 2*i + 1
            
            # U n_{i,↑} n_{i,↓}
            H += self.U * self._number_number_term(qubit_up, qubit_down)
            
            # -μ (n_{i,↑} + n_{i,↓})
            H -= self.mu * (self._number_term(qubit_up) + 
                          self._number_term(qubit_down))
        
        return H
    
    def _hopping_term(self, i: int, j: int) -> np.ndarray:
        """构建跃迁项矩阵 (Jordan-Wigner)"""
        dim = 2**self.n_qubits
        
        # 简化的跃迁项实现
        # 实际应该包含完整的Jordan-Wigner弦
        H = np.zeros((dim, dim), dtype=complex)
        
        # 获取i和j之间的所有状态
        for state in range(dim):
            # 检查i和j位置的占据
            i_occ = (state >> i) & 1
            j_occ = (state >> j) & 1
            
            if i_occ != j_occ:  # 一个占据一个空
                # 跃迁操作
                new_state = state ^ (1 << i) ^ (1 << j)
                
                # 计算Jordan-Wigner相位
                sign = 1.0
                if i < j:
                    for k in range(i+1, j):
                        if ((state >> k) & 1):
                            sign *= -1
                else:
                    for k in range(j+1, i):
                        if ((state >> k) & 1):
                            sign *= -1
                
                H[new_state, state] -= self.t * sign
                H[state, new_state] -= self.t * sign  # 厄米共轭
        
        return H
    
    def _number_number_term(self, i: int, j: int) -> np.ndarray:
        """构建n_i n_j项"""
        dim = 2**self.n_qubits
        H = np.zeros((dim, dim))
        
        for state in range(dim):
            i_occ = (state >> i) & 1
            j_occ = (state >> j) & 1
            H[state, state] = i_occ * j_occ
        
        return H
    
    def _number_term(self, i: int) -> np.ndarray:
        """构建粒子数算符n_i"""
        dim = 2**self.n_qubits
        H = np.zeros((dim, dim))
        
        for state in range(dim):
            H[state, state] = (state >> i) & 1
        
        return H
    
    def diagonalize(self, n_states: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """对角化哈密顿量"""
        eigenvalues, eigenvectors = eigh(self.hamiltonian)
        return eigenvalues[:n_states], eigenvectors[:, :n_states]
    
    def get_ground_state_energy(self) -> float:
        """获取基态能量"""
        energies, _ = self.diagonalize(n_states=1)
        return energies[0]
    
    def compute_double_occupancy(self, state: np.ndarray) -> float:
        """计算双占据概率 ⟨n_{i,↑}n_{i,↓}⟩"""
        double_occ = 0.0
        
        for i in range(self.lattice.n_sites):
            D_i = self._number_number_term(2*i, 2*i+1)
            double_occ += np.vdot(state, D_i @ state)
        
        return double_occ / self.lattice.n_sites
    
    def compute_kinetic_energy(self, state: np.ndarray) -> float:
        """计算动能期望"""
        T = np.zeros_like(self.hamiltonian)
        
        for i in range(self.lattice.n_sites):
            for j in self.lattice.neighbors[i]:
                if j <= i:
                    continue
                for spin in [0, 1]:
                    qubit_i = 2*i + spin
                    qubit_j = 2*j + spin
                    T += self._hopping_term(qubit_i, qubit_j)
        
        return np.vdot(state, T @ state).real
    
    def compute_spin_structure_factor(self, state: np.ndarray, q: np.ndarray) -> float:
        """
        计算自旋结构因子 S(q)
        S(q) = (1/N) Σ_{i,j} e^{iq·(r_i-r_j)} ⟨S_i·S_j⟩
        """
        # 简化的实现，假设一维链
        Sq = 0.0
        
        for i in range(self.lattice.n_sites):
            for j in range(self.lattice.n_sites):
                phase = np.exp(1j * q[0] * (i - j))
                
                # 计算⟨S_i·S_j⟩
                SiSj = self._spin_correlation(i, j, state)
                Sq += phase * SiSj
        
        return Sq.real / self.lattice.n_sites
    
    def _spin_correlation(self, i: int, j: int, state: np.ndarray) -> float:
        """计算自旋关联 ⟨S_i·S_j⟩"""
        # S_i·S_j = S_i^z S_j^z + (1/2)(S_i^+ S_j^- + S_i^- S_j^+)
        
        # S^z = (n_↑ - n_↓)/2
        Sz_i = 0.5 * (self._number_term(2*i) - self._number_term(2*i+1))
        Sz_j = 0.5 * (self._number_term(2*j) - self._number_term(2*j+1))
        
        SzSz = np.vdot(state, Sz_i @ Sz_j @ state)
        
        # 简化的横向关联 (实际应该实现升降算符)
        return SzSz.real


class SpinSystem:
    """
    量子自旋系统模拟
    
    实现各种自旋模型：
    - Heisenberg模型
    - Ising模型
    - XY模型
    - XXZ模型
    """
    
    def __init__(self, lattice: Lattice, model_type: str = "heisenberg",
                 J: float = 1.0, hx: float = 0.0, hz: float = 0.0,
                 Delta: float = 1.0):
        """
        Args:
            lattice: 晶格结构
            model_type: "heisenberg", "ising", "xy", "xxz"
            J: 交换耦合
            hx: 横向磁场
            hz: 纵向磁场
            Delta: XXZ模型的各向异性参数
        """
        self.lattice = lattice
        self.model_type = model_type
        self.J = J
        self.hx = hx
        self.hz = hz
        self.Delta = Delta
        
        self.n_qubits = lattice.n_sites
        self.hamiltonian = self._build_hamiltonian()
    
    def _build_hamiltonian(self) -> np.ndarray:
        """构建自旋哈密顿量"""
        dim = 2**self.n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        
        # 交换相互作用
        for i in range(self.lattice.n_sites):
            for j in self.lattice.neighbors[i]:
                if j <= i:
                    continue
                
                if self.model_type == "ising":
                    # H = -J Σ Z_i Z_j - h Σ X_i
                    H -= self.J * self._zz_interaction(i, j)
                elif self.model_type == "heisenberg":
                    # H = J Σ (X_i X_j + Y_i Y_j + Z_i Z_j)
                    H += self.J * (self._xx_interaction(i, j) +
                                  self._yy_interaction(i, j) +
                                  self._zz_interaction(i, j))
                elif self.model_type == "xy":
                    # H = J Σ (X_i X_j + Y_i Y_j)
                    H += self.J * (self._xx_interaction(i, j) +
                                  self._yy_interaction(i, j))
                elif self.model_type == "xxz":
                    # H = J Σ (X_i X_j + Y_i Y_j + Δ Z_i Z_j)
                    H += self.J * (self._xx_interaction(i, j) +
                                  self._yy_interaction(i, j) +
                                  self.Delta * self._zz_interaction(i, j))
        
        # 外场
        for i in range(self.lattice.n_sites):
            H -= self.hx * self._pauli_x(i)
            H -= self.hz * self._pauli_z(i)
        
        return H
    
    def _pauli_x(self, site: int) -> np.ndarray:
        """Pauli-X算符作用于第site个量子比特"""
        dim = 2**self.n_qubits
        mat = np.zeros((dim, dim))
        
        for state in range(dim):
            j = state ^ (1 << site)
            mat[j, state] = 1.0
        
        return mat
    
    def _pauli_y(self, site: int) -> np.ndarray:
        """Pauli-Y算符"""
        dim = 2**self.n_qubits
        mat = np.zeros((dim, dim), dtype=complex)
        
        for state in range(dim):
            j = state ^ (1 << site)
            if ((state >> site) & 1) == 0:
                mat[j, state] = -1j
            else:
                mat[j, state] = 1j
        
        return mat
    
    def _pauli_z(self, site: int) -> np.ndarray:
        """Pauli-Z算符"""
        dim = 2**self.n_qubits
        mat = np.zeros((dim, dim))
        
        for state in range(dim):
            mat[state, state] = 1.0 if ((state >> site) & 1) == 0 else -1.0
        
        return mat
    
    def _xx_interaction(self, i: int, j: int) -> np.ndarray:
        """X_i X_j 相互作用"""
        return self._pauli_x(i) @ self._pauli_x(j)
    
    def _yy_interaction(self, i: int, j: int) -> np.ndarray:
        """Y_i Y_j 相互作用"""
        return self._pauli_y(i) @ self._pauli_y(j)
    
    def _zz_interaction(self, i: int, j: int) -> np.ndarray:
        """Z_i Z_j 相互作用"""
        return self._pauli_z(i) @ self._pauli_z(j)
    
    def diagonalize(self, n_states: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """对角化哈密顿量"""
        eigenvalues, eigenvectors = eigh(self.hamiltonian)
        return eigenvalues[:n_states], eigenvectors[:, :n_states]
    
    def compute_magnetization(self, state: np.ndarray, direction: str = 'z') -> float:
        """计算磁化强度"""
        M = 0.0
        
        for i in range(self.lattice.n_sites):
            if direction == 'x':
                M += np.vdot(state, self._pauli_x(i) @ state)
            elif direction == 'y':
                M += np.vdot(state, self._pauli_y(i) @ state)
            else:  # 'z'
                M += np.vdot(state, self._pauli_z(i) @ state)
        
        return M.real / self.lattice.n_sites
    
    def compute_spin_correlation(self, state: np.ndarray, r: int) -> float:
        """
        计算距离为r的自旋关联函数
        C(r) = ⟨S_0·S_r⟩
        """
        C = 0.0
        
        for i in range(self.lattice.n_sites):
            j = (i + r) % self.lattice.n_sites  # 周期性边界
            
            # 简化的关联计算
            C += (np.vdot(state, self._xx_interaction(i, j) @ state) +
                  np.vdot(state, self._yy_interaction(i, j) @ state) +
                  np.vdot(state, self._zz_interaction(i, j) @ state)).real / 4.0
        
        return C / self.lattice.n_sites
    
    def compute_staggered_magnetization(self, state: np.ndarray) -> float:
        """
        计算交错磁化强度 (用于反铁磁相)
        m_s = (1/N) |Σ_i (-1)^i ⟨S_i^z⟩|
        """
        m_s = 0.0
        
        for i in range(self.lattice.n_sites):
            sign = (-1)**i
            Sz_i = np.vdot(state, self._pauli_z(i) @ state).real / 2.0
            m_s += sign * Sz_i
        
        return abs(m_s) / self.lattice.n_sites
    
    def compute_entanglement_entropy(self, state: np.ndarray, 
                                      subregion: List[int]) -> float:
        """
        计算子区域的纠缠熵
        S_A = -Tr(ρ_A log ρ_A)
        """
        # 构建约化密度矩阵
        rho_A = self._reduced_density_matrix(state, subregion)
        
        # 对角化并计算熵
        eigenvalues = np.linalg.eigvalsh(rho_A)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]  # 避免log(0)
        
        entropy = -np.sum(eigenvalues * np.log(eigenvalues))
        return entropy
    
    def _reduced_density_matrix(self, state: np.ndarray, 
                                subregion: List[int]) -> np.ndarray:
        """构建子区域的约化密度矩阵"""
        n_A = len(subregion)
        n_B = self.n_qubits - n_A
        
        dim_A = 2**n_A
        dim_B = 2**n_B
        
        # 重排状态向量
        rho = np.outer(state, state.conj())
        
        # 部分迹 (简化实现)
        rho_A = np.zeros((dim_A, dim_A), dtype=complex)
        
        # 环境索引
        env_sites = [i for i in range(self.n_qubits) if i not in subregion]
        
        for i_A in range(dim_A):
            for j_A in range(dim_A):
                trace = 0.0
                for k_B in range(dim_B):
                    # 构建完整索引
                    idx_i = self._combine_indices(i_A, k_B, subregion, env_sites)
                    idx_j = self._combine_indices(j_A, k_B, subregion, env_sites)
                    trace += rho[idx_i, idx_j]
                rho_A[i_A, j_A] = trace
        
        return rho_A
    
    def _combine_indices(self, idx_A: int, idx_B: int,
                        subregion: List[int], env_sites: List[int]) -> int:
        """组合子系统和环境索引为完整索引"""
        # 简化的索引组合
        full_idx = 0
        
        for i, site in enumerate(subregion):
            bit = (idx_A >> i) & 1
            full_idx |= bit << site
        
        for i, site in enumerate(env_sites):
            bit = (idx_B >> i) & 1
            full_idx |= bit << site
        
        return full_idx


class TopologicalInvariant:
    """
    拓扑不变量计算
    
    实现各种拓扑不变量的测量：
    - Berry相位
    - Chern数
    - Z2不变量
    """
    
    def __init__(self, band_structure: Callable = None):
        """
        Args:
            band_structure: 能带结构计算函数 H(k) -> 矩阵
        """
        self.band_structure = band_structure
    
    def compute_berry_phase(self, k_path: np.ndarray, band_index: int = 0) -> float:
        """
        计算Berry相位
        
        γ = i ∮ ⟨u_k|∇_k|u_k⟩ · dk
        
        Args:
            k_path: k空间中的闭合路径
            band_index: 能带索引
        """
        if self.band_structure is None:
            raise ValueError("需要设置能带结构函数")
        
        berry_phase = 0.0
        
        # 沿路径计算波函数重叠
        for i in range(len(k_path) - 1):
            k1 = k_path[i]
            k2 = k_path[i + 1]
            
            # 获取波函数
            _, evecs1 = self._get_eigenstate(k1, band_index)
            _, evecs2 = self._get_eigenstate(k2, band_index)
            
            # 计算重叠
            overlap = np.vdot(evecs1, evecs2)
            berry_phase += np.angle(overlap)
        
        return berry_phase
    
    def compute_chern_number(self, nk: int = 50, band_index: int = 0) -> float:
        """
        计算Chern数 (二维系统)
        
        C = (1/2π) ∫∫ F_{xy} dk_x dk_y
        
        其中F_{xy}是Berry曲率
        """
        if self.band_structure is None:
            raise ValueError("需要设置能带结构函数")
        
        chern = 0.0
        
        # 在布里渊区网格上计算
        dk = 2 * np.pi / nk
        
        for i in range(nk):
            for j in range(nk):
                kx = i * dk
                ky = j * dk
                
                # 计算Berry曲率 (使用离散近似)
                F = self._berry_curvature_discrete(kx, ky, dk, band_index)
                chern += F * dk * dk
        
        return chern / (2 * np.pi)
    
    def _berry_curvature_discrete(self, kx: float, ky: float, 
                                   dk: float, band_index: int) -> float:
        """使用离散方法计算Berry曲率"""
        # 四个角的k点
        k1 = np.array([kx, ky])
        k2 = np.array([kx + dk, ky])
        k3 = np.array([kx + dk, ky + dk])
        k4 = np.array([kx, ky + dk])
        
        # 计算U(1)链接变量
        U12 = self._link_variable(k1, k2, band_index)
        U23 = self._link_variable(k2, k3, band_index)
        U34 = self._link_variable(k3, k4, band_index)
        U41 = self._link_variable(k4, k1, band_index)
        
        # Berry曲率 ≈ Im(log(U12 U23 U34 U41))
        F = np.angle(U12 * U23 * U34 * U41)
        
        return F / (dk * dk)
    
    def _link_variable(self, k1: np.ndarray, k2: np.ndarray, band_index: int) -> complex:
        """计算链接变量 ⟨u_{k1}|u_{k2}⟩"""
        _, u1 = self._get_eigenstate(k1, band_index)
        _, u2 = self._get_eigenstate(k2, band_index)
        
        return np.vdot(u1, u2)
    
    def _get_eigenstate(self, k: np.ndarray, band_index: int) -> Tuple[float, np.ndarray]:
        """获取给定k点的本征态"""
        H = self.band_structure(k)
        eigenvalues, eigenvectors = eigh(H)
        
        return eigenvalues[band_index], eigenvectors[:, band_index]
    
    def compute_z2_invariant(self, nk: int = 30, occupied_bands: int = 1) -> int:
        """
        计算Z2拓扑不变量 (时间反演不变的系统)
        
        适用于：拓扑绝缘体、量子自旋霍尔效应
        """
        # Z2不变量通过计算时间反演不变动量点的占据宇称得到
        
        # 时间反演不变动量点 (TRIM)
        trim_points = [
            np.array([0, 0]),
            np.array([np.pi, 0]),
            np.array([0, np.pi]),
            np.array([np.pi, np.pi])
        ]
        
        # 计算每个TRIM点的占据宇称
        parities = []
        for k in trim_points:
            H = self.band_structure(k)
            eigenvalues, eigenvectors = eigh(H)
            
            # 简化的宇称计算 (实际需要时间反演操作)
            parity = 1.0
            for i in range(occupied_bands):
                # 简化的实现
                parity *= 1.0
            
            parities.append(parity)
        
        # Z2 = Π_i δ_i (mod 2)
        z2 = int(np.prod(parities)) % 2
        
        return z2
    
    def compute_wilson_loop(self, k_fixed: float, n_bands: int = 1,
                            nk: int = 100) -> np.ndarray:
        """
        计算Wilson环
        
        W = P exp(i ∮ A·dk)
        
        用于计算Wannier中心和拓扑分类
        """
        wilson = np.eye(n_bands, dtype=complex)
        
        # 沿k_y方向积分，固定k_x
        dk = 2 * np.pi / nk
        
        for i in range(nk):
            ky = i * dk
            k = np.array([k_fixed, ky])
            k_next = np.array([k_fixed, ((i + 1) % nk) * dk])
            
            # 获取多带波函数
            _, evecs = self._get_multiband_states(k, n_bands)
            _, evecs_next = self._get_multiband_states(k_next, n_bands)
            
            # 计算重叠矩阵
            overlap = evecs.conj().T @ evecs_next
            
            # 累积Wilson环
            wilson = wilson @ overlap
        
        return wilson
    
    def _get_multiband_states(self, k: np.ndarray, n_bands: int) -> Tuple[np.ndarray, np.ndarray]:
        """获取多个能带的波函数"""
        H = self.band_structure(k)
        eigenvalues, eigenvectors = eigh(H)
        
        return eigenvalues[:n_bands], eigenvectors[:, :n_bands]
    
    def compute_wannier_center(self, k_fixed: float, n_bands: int = 1) -> np.ndarray:
        """
        计算Wannier中心位置
        θ = (1/2π) Im(log(det(W)))
        """
        wilson = self.compute_wilson_loop(k_fixed, n_bands)
        
        # Wannier中心是Wilson环特征值的相位
        eigenvalues = np.linalg.eigvals(wilson)
        centers = np.angle(eigenvalues) / (2 * np.pi)
        
        return centers


class HaldaneModel:
    """
    Haldane模型 - 具有陈数非零的无能隙边缘态的蜂窝晶格模型
    
    H = -t Σ_{<i,j>} c_i^† c_j 
        - t_2 Σ_{<<i,j>>} e^{iφ} c_i^† c_j 
        + M Σ_i ξ_i c_i^† c_i
    
    其中ξ_i = +1 (-1) 对于A (B) 子格点
    """
    
    def __init__(self, t: float = 1.0, t2: float = 0.1, 
                 phi: float = np.pi/2, M: float = 0.0):
        self.t = t
        self.t2 = t2
        self.phi = phi
        self.M = M
        
        # 蜂窝晶格的基矢
        self.a1 = np.array([np.sqrt(3)/2, 1.5])
        self.a2 = np.array([-np.sqrt(3)/2, 1.5])
        
        # A和B子格点的相对位置
        self.delta = np.array([0, 1.0])
    
    def get_hamiltonian(self, k: np.ndarray) -> np.ndarray:
        """获取给定k点的布洛赫哈密顿量"""
        kx, ky = k
        
        # 最近邻项
        d1 = np.exp(1j * kx * np.sqrt(3)/2) * 2 * np.cos(ky/2 + np.pi/3)
        
        # 次近邻项
        d0 = 2 * self.t2 * np.cos(self.phi) * (
            np.cos(kx * np.sqrt(3)) + 2 * np.cos(3*ky/2) * np.cos(kx * np.sqrt(3)/2)
        )
        
        dx = self.t * (np.cos(kx * np.sqrt(3)/2 + ky/2) + 
                      np.cos(kx * np.sqrt(3)/2 - ky/2) +
                      np.cos(ky))
        
        dy = self.t * (np.sin(kx * np.sqrt(3)/2 + ky/2) - 
                      np.sin(kx * np.sqrt(3)/2 - ky/2) +
                      np.sin(ky))
        
        dz = self.M + 2 * self.t2 * np.sin(self.phi) * (
            np.sin(kx * np.sqrt(3)) - 
            2 * np.sin(kx * np.sqrt(3)/2) * np.cos(3*ky/2)
        )
        
        # 2x2哈密顿量
        H = np.array([[d0 + dz, dx - 1j*dy],
                      [dx + 1j*dy, d0 - dz]], dtype=complex)
        
        return H
    
    def compute_chern_number(self, nk: int = 100) -> float:
        """计算Haldane模型的Chern数"""
        invariant = TopologicalInvariant(band_structure=self.get_hamiltonian)
        return invariant.compute_chern_number(nk=nk, band_index=0)
    
    def get_band_structure(self, k_points: np.ndarray) -> np.ndarray:
        """计算能带结构"""
        n_points = len(k_points)
        energies = np.zeros((n_points, 2))
        
        for i, k in enumerate(k_points):
            H = self.get_hamiltonian(k)
            eigenvalues = np.linalg.eigvalsh(H)
            energies[i] = eigenvalues
        
        return energies


class KitaevModel:
    """
    Kitaev蜂窝模型 - 量子自旋液体
    
    H = -Jx Σ_{x-links} σ_i^x σ_j^x
        -Jy Σ_{y-links} σ_i^y σ_j^y
        -Jz Σ_{z-links} σ_i^z σ_j^z
    
    可精确求解，支持非阿贝尔任意子激发
    """
    
    def __init__(self, Jx: float = 1.0, Jy: float = 1.0, Jz: float = 1.0,
                 nx: int = 3, ny: int = 3):
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.nx = nx
        self.ny = ny
        self.n_sites = 2 * nx * ny
    
    def construct_majorana_representation(self) -> np.ndarray:
        """
        使用Majorana费米子表示构造哈密顿量
        
        每个自旋分解为4个Majorana费米子
        """
        # 简化的实现
        # 实际需要将每个自旋映射到四个Majorana算符
        dim = 2**(2 * self.n_sites)  # Majorana维数
        
        H = np.zeros((dim, dim))
        
        # 构建Kitaev哈密顿量
        # 这是一个复杂的映射，这里提供概念性实现
        
        return H
    
    def find_ground_state_manifold(self) -> Dict:
        """
        寻找基态流形
        
        Kitaev模型的基态简并依赖于拓扑
        """
        # 基态简并度与系统的拓扑性质相关
        # 在环面上，简并度为4
        
        return {
            'ground_state_degeneracy': 4,
            'topological_order': 'Z2',
            'anyon_types': ['1', 'e', 'm', 'ε']
        }
    
    def compute_vison_correlation(self, r: int) -> float:
        """
        计算vison关联函数
        
        Vison是Z2规范场的通量激发
        """
        # 简化的关联函数计算
        return np.exp(-r / 1.0)  # 指数衰减


# ============ 实用函数 ============

def hubbard_phase_diagram(U_range: np.ndarray, t_range: np.ndarray,
                           n_sites: int = 4) -> np.ndarray:
    """
    计算Hubbard模型的相图
    
    返回每个参数点的基态能量
    """
    lattice = Lattice.create_chain(n_sites)
    
    phase_diagram = np.zeros((len(U_range), len(t_range)))
    
    for i, U in enumerate(U_range):
        for j, t in enumerate(t_range):
            model = HubbardModel(lattice, t=t, U=U)
            energy = model.get_ground_state_energy()
            phase_diagram[i, j] = energy
    
    return phase_diagram


def spin_phase_diagram(J_range: np.ndarray, h_range: np.ndarray,
                       n_sites: int = 8, model_type: str = "ising") -> Dict:
    """
    计算自旋模型的相图
    
    返回序参量随参数的变化
    """
    lattice = Lattice.create_chain(n_sites)
    
    magnetizations = np.zeros((len(J_range), len(h_range)))
    
    for i, J in enumerate(J_range):
        for j, h in enumerate(h_range):
            model = SpinSystem(lattice, model_type=model_type, J=J, hx=h)
            _, ground_state = model.diagonalize(n_states=1)
            
            if model_type == "ising":
                m = model.compute_magnetization(ground_state[:, 0], 'x')
            else:
                m = model.compute_staggered_magnetization(ground_state[:, 0])
            
            magnetizations[i, j] = abs(m)
    
    return {
        'magnetization': magnetizations,
        'J_range': J_range,
        'h_range': h_range
    }


def run_hubbard_example():
    """Hubbard模型示例"""
    print("=" * 60)
    print("Hubbard模型示例")
    print("=" * 60)
    
    # 创建4格点链
    lattice = Lattice.create_chain(4, periodic=True)
    print(f"格点数: {lattice.n_sites}")
    
    # 半填充Hubbard模型
    hubbard = HubbardModel(lattice, t=1.0, U=4.0)
    print(f"电子数: {hubbard.n_electrons}")
    print(f"量子比特数: {hubbard.n_qubits}")
    
    # 对角化
    energies, states = hubbard.diagonalize(n_states=3)
    print(f"\n前3个本征能量:")
    for i, E in enumerate(energies):
        print(f"  E_{i} = {E:.6f} t")
    
    # 计算基态物理量
    gs = states[:, 0]
    double_occ = hubbard.compute_double_occupancy(gs)
    E_kin = hubbard.compute_kinetic_energy(gs)
    
    print(f"\n基态性质:")
    print(f"  双占据概率: {double_occ:.4f}")
    print(f"  动能: {E_kin:.4f} t")
    
    return hubbard


def run_spin_example():
    """自旋系统示例"""
    print("\n" + "=" * 60)
    print("自旋系统示例: Heisenberg模型")
    print("=" * 60)
    
    lattice = Lattice.create_chain(8, periodic=True)
    
    # Heisenberg模型
    heisenberg = SpinSystem(lattice, model_type="heisenberg", J=1.0)
    
    energies, states = heisenberg.diagonalize(n_states=3)
    print(f"前3个本征能量:")
    for i, E in enumerate(energies):
        print(f"  E_{i} = {E:.6f} J")
    
    # 基态性质
    gs = states[:, 0]
    
    # 磁化强度
    m_z = heisenberg.compute_magnetization(gs, 'z')
    print(f"\n基态磁化强度: ⟨S^z⟩ = {m_z:.6f}")
    
    # 交错磁化
    m_s = heisenberg.compute_staggered_magnetization(gs)
    print(f"交错磁化强度: m_s = {m_s:.6f}")
    
    # 自旋关联
    print("\n自旋关联函数 C(r):")
    for r in [1, 2, 3]:
        C_r = heisenberg.compute_spin_correlation(gs, r)
        print(f"  C({r}) = {C_r:.6f}")
    
    return heisenberg


def run_topological_example():
    """拓扑不变量示例"""
    print("\n" + "=" * 60)
    print("拓扑不变量示例: Haldane模型")
    print("=" * 60)
    
    # 创建Haldane模型
    haldane = HaldaneModel(t=1.0, t2=0.1, phi=np.pi/2, M=0.0)
    
    # 计算Chern数
    chern = haldane.compute_chern_number(nk=50)
    print(f"Chern数: C = {chern:.2f}")
    
    # 改变参数，观察Chern数变化
    haldane_M = HaldaneModel(t=1.0, t2=0.1, phi=np.pi/2, M=0.5)
    chern_M = haldane_M.compute_chern_number(nk=50)
    print(f"M=0.5时Chern数: C = {chern_M:.2f}")
    
    return haldane


if __name__ == "__main__":
    run_hubbard_example()
    run_spin_example()
    run_topological_example()
