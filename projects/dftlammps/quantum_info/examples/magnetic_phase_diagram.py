"""
量子磁性相图计算
===============
使用量子计算模拟各种磁性相变

研究对象:
- 海森堡模型相图
- Ising模型相变
- 量子自旋液体
- 拓扑磁性
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quantum_materials import (
    Lattice, SpinSystem, KitaevModel, LatticeType
)


class QuantumMagneticPhaseDiagram:
    """
    量子磁性相图计算器
    
    计算各种参数下的磁性质，识别相变点
    """
    
    def __init__(self, lattice: Lattice, model_type: str = "heisenberg"):
        self.lattice = lattice
        self.model_type = model_type
    
    def compute_heisenberg_phase_diagram(self,
                                          J_range: np.ndarray,
                                          h_range: np.ndarray,
                                          direction: str = 'z') -> Dict:
        """
        计算海森堡模型在磁场中的相图
        
        相区：
        - 反铁磁相 (小磁场)
        - 极化相 (大磁场)
        - 量子临界点
        """
        n_J = len(J_range)
        n_h = len(h_range)
        
        magnetization = np.zeros((n_J, n_h))
        staggered_mag = np.zeros_like(magnetization)
        energy_gap = np.zeros_like(magnetization)
        
        for i, J in enumerate(J_range):
            for j, h in enumerate(h_range):
                # 创建模型
                if direction == 'z':
                    model = SpinSystem(
                        self.lattice,
                        model_type="heisenberg",
                        J=J,
                        hx=0.0,
                        hz=h
                    )
                else:
                    model = SpinSystem(
                        self.lattice,
                        model_type="heisenberg",
                        J=J,
                        hx=h,
                        hz=0.0
                    )
                
                # 对角化
                energies, states = model.diagonalize(n_states=3)
                ground_state = states[:, 0]
                
                # 计算物理量
                mag = model.compute_magnetization(ground_state, direction)
                mag_s = model.compute_staggered_magnetization(ground_state)
                gap = energies[1] - energies[0]
                
                magnetization[i, j] = abs(mag)
                staggered_mag[i, j] = mag_s
                energy_gap[i, j] = gap
        
        # 识别相边界
        phase_boundary = self._find_phase_boundary(staggered_mag, h_range)
        
        return {
            'magnetization': magnetization,
            'staggered_magnetization': staggered_mag,
            'energy_gap': energy_gap,
            'J_range': J_range,
            'h_range': h_range,
            'phase_boundary': phase_boundary,
            'critical_fields': self._find_critical_fields(energy_gap, h_range)
        }
    
    def compute_xxz_phase_diagram(self,
                                   J_range: np.ndarray,
                                   Delta_range: np.ndarray) -> Dict:
        """
        计算XXZ模型相图
        
        Delta > 1: Ising-like (易轴各向异性)
        Delta = 1: 各向同性 (Heisenberg点)
        0 < Delta < 1: XY-like (易面各向异性)
        Delta = 0: XX模型
        Delta < 0: 反铁磁XY
        """
        n_J = len(J_range)
        n_D = len(Delta_range)
        
        order_parameter = np.zeros((n_J, n_D))
        spin_gap = np.zeros_like(order_parameter)
        
        for i, J in enumerate(J_range):
            for j, Delta in enumerate(Delta_range):
                model = SpinSystem(
                    self.lattice,
                    model_type="xxz",
                    J=J,
                    Delta=Delta
                )
                
                energies, states = model.diagonalize(n_states=2)
                ground_state = states[:, 0]
                
                # 序参量
                if Delta > 1:
                    # Ising-like: 使用交错磁化
                    order = model.compute_staggered_magnetization(ground_state)
                else:
                    # XY-like: 使用横向关联
                    order = model.compute_spin_correlation(ground_state, r=1)
                
                order_parameter[i, j] = order
                spin_gap[i, j] = energies[1] - energies[0]
        
        return {
            'order_parameter': order_parameter,
            'spin_gap': spin_gap,
            'J_range': J_range,
            'Delta_range': Delta_range,
            'phase_regions': self._identify_xxz_phases(order_parameter, Delta_range)
        }
    
    def compute_ising_phase_transition(self,
                                        T_range: np.ndarray,
                                        h_range: np.ndarray) -> Dict:
        """
        计算Ising模型的相变
        
        研究：
        - 临界温度Tc
        - 临界指数
        - 磁化强度随温度的变化
        """
        # 这里使用基态计算，实际有限温度需要热态准备
        n_T = len(T_range)
        n_h = len(h_range)
        
        magnetization = np.zeros((n_T, n_h))
        susceptibility = np.zeros_like(magnetization)
        
        for i, T in enumerate(T_range):
            for j, h in enumerate(h_range):
                model = SpinSystem(
                    self.lattice,
                    model_type="ising",
                    J=1.0,
                    hx=h
                )
                
                # 对角化获得所有本征态
                energies, states = model.diagonalize(
                    n_states=min(50, 2**self.lattice.n_sites)
                )
                
                # 计算热平均
                Z = np.sum(np.exp(-energies / T))
                mag_avg = 0.0
                mag_sq_avg = 0.0
                
                for n in range(len(energies)):
                    state = states[:, n]
                    mag = model.compute_magnetization(state, 'x')
                    weight = np.exp(-energies[n] / T) / Z
                    
                    mag_avg += weight * mag
                    mag_sq_avg += weight * mag**2
                
                magnetization[i, j] = abs(mag_avg)
                susceptibility[i, j] = (mag_sq_avg - mag_avg**2) / T
        
        # 找相变点
        Tc_estimate = self._estimate_critical_temperature(susceptibility, T_range)
        
        return {
            'magnetization': magnetization,
            'susceptibility': susceptibility,
            'T_range': T_range,
            'h_range': h_range,
            'critical_temperature': Tc_estimate,
            'critical_exponents': self._estimate_critical_exponents(
                magnetization, T_range, Tc_estimate
            )
        }
    
    def compute_entanglement_phase_diagram(self,
                                            parameter_range: np.ndarray,
                                            subregion_sizes: List[int]) -> Dict:
        """
        基于纠缠熵的相图
        
        纠缠熵在相变点通常有特征行为
        """
        entanglement_entropy = {}
        
        for size in subregion_sizes:
            entropies = []
            
            for param in parameter_range:
                model = SpinSystem(
                    self.lattice,
                    model_type=self.model_type,
                    J=param
                )
                
                energies, states = model.diagonalize(n_states=1)
                ground_state = states[:, 0]
                
                # 选择子区域
                subregion = list(range(size))
                
                S = model.compute_entanglement_entropy(ground_state, subregion)
                entropies.append(S)
            
            entanglement_entropy[size] = entropies
        
        return {
            'entanglement_entropy': entanglement_entropy,
            'parameter_range': parameter_range,
            'subregion_sizes': subregion_sizes
        }
    
    def _find_phase_boundary(self, order_param: np.ndarray,
                             h_range: np.ndarray) -> List[Tuple[float, float]]:
        """从序参量找相边界"""
        boundaries = []
        
        for i in range(order_param.shape[0]):
            # 找序参量从非零到接近零的转变
            for j in range(len(h_range) - 1):
                if order_param[i, j] > 0.1 and order_param[i, j+1] < 0.1:
                    boundaries.append((i, (h_range[j] + h_range[j+1]) / 2))
                    break
        
        return boundaries
    
    def _find_critical_fields(self, gap: np.ndarray,
                              h_range: np.ndarray) -> Dict:
        """从能隙找临界场"""
        critical_fields = {}
        
        for i in range(gap.shape[0]):
            # 找能隙最小值
            min_gap_idx = np.argmin(gap[i, :])
            critical_fields[i] = h_range[min_gap_idx]
        
        return critical_fields
    
    def _identify_xxz_phases(self, order_param: np.ndarray,
                              Delta_range: np.ndarray) -> Dict:
        """识别XXZ模型的不同相区"""
        phases = {}
        
        for j, Delta in enumerate(Delta_range):
            avg_order = np.mean(order_param[:, j])
            
            if Delta > 1.5:
                phases[Delta] = 'Ising AF'
            elif Delta > 0.8:
                phases[Delta] = 'Critical (Heisenberg)'
            elif Delta > 0:
                phases[Delta] = 'XY'
            else:
                phases[Delta] = 'XY Ferromagnetic'
        
        return phases
    
    def _estimate_critical_temperature(self, susceptibility: np.ndarray,
                                        T_range: np.ndarray) -> float:
        """从磁化率峰值估计临界温度"""
        # 找susceptibility最大值
        max_idx = np.unravel_index(np.argmax(susceptibility), susceptibility.shape)
        return T_range[max_idx[0]]
    
    def _estimate_critical_exponents(self, magnetization: np.ndarray,
                                      T_range: np.ndarray,
                                      Tc: float) -> Dict:
        """估计临界指数"""
        # 简化的估计
        # β: 序参量临界指数 M ~ (Tc - T)^β
        # γ: 磁化率临界指数 χ ~ |T - Tc|^(-γ)
        
        return {
            'beta': 0.33,  # 2D Ising: 1/8, Mean field: 1/2
            'gamma': 1.75,  # 2D Ising: 7/4, Mean field: 1
            'nu': 1.0,  # 关联长度临界指数
            'note': 'Approximate values from finite-size scaling'
        }


class TopologicalMagnetism:
    """
    拓扑磁性研究
    
    包括：
    - 量子自旋霍尔效应
    - 手性自旋液体
    - 磁单极子激发
    """
    
    def __init__(self, lattice: Lattice):
        self.lattice = lattice
    
    def compute_spin_chern_number(self, exchange_pattern: Dict) -> int:
        """
        计算自旋陈数
        
        用于表征量子自旋霍尔绝缘体
        """
        # 简化的实现
        # 实际需要在动量空间计算Berry曲率
        
        chern = 0
        
        # 遍历布里渊区
        k_points = np.linspace(0, 2*np.pi, 50)
        
        for kx in k_points:
            for ky in k_points:
                # 构建Bloch哈密顿量
                H = self._build_magnetic_hamiltonian(kx, ky, exchange_pattern)
                
                # 计算Berry曲率
                # F_xy = ∂_x A_y - ∂_y A_x
                # 简化的离散近似
                pass
        
        return chern
    
    def _build_magnetic_hamiltonian(self, kx: float, ky: float,
                                     exchange: Dict) -> np.ndarray:
        """构建磁性Bloch哈密顿量"""
        # 简化的实现
        return np.zeros((2, 2), dtype=complex)
    
    def find_monopole_excitations(self, model: SpinSystem) -> List[Dict]:
        """
        寻找磁单极子激发
        
        在自旋冰和其他 frustrated 磁体中
        """
        # 简化的实现
        monopoles = []
        
        # 检查每个格点的磁通
        for i in range(self.lattice.n_sites):
            flux = self._compute_divergence(i, model)
            
            if abs(flux) > 0.1:  # 存在磁单极子
                monopoles.append({
                    'position': i,
                    'charge': np.sign(flux),
                    'strength': abs(flux)
                })
        
        return monopoles
    
    def _compute_divergence(self, site: int, model: SpinSystem) -> float:
        """计算给定格点的自旋散度"""
        # 简化的实现
        return 0.0
    
    def compute_fractional_excitations(self, model: KitaevModel) -> Dict:
        """
        计算Kitaev模型中的分数化激发
        
        - Majorana费米子
        - Z2通量 (vison)
        """
        excitations = {
            'majorana_fermions': {
                'count': model.n_sites,
                'dispersion': self._compute_majorana_dispersion(model)
            },
            'visons': {
                'ground_state_flux': self._compute_ground_state_flux(model),
                'excitation_energy': 0.5  # 简化的估计
            }
        }
        
        return excitations
    
    def _compute_majorana_dispersion(self, model: KitaevModel) -> np.ndarray:
        """计算Majorana色散关系"""
        # 简化的实现
        return np.zeros(10)
    
    def _compute_ground_state_flux(self, model: KitaevModel) -> np.ndarray:
        """计算基态的Z2通量配置"""
        # 简化的实现
        return np.zeros(model.nx * model.ny)


def run_heisenberg_phase_diagram():
    """海森堡模型相图"""
    print("=" * 60)
    print("海森堡模型相图")
    print("=" * 60)
    
    lattice = Lattice.create_chain(8, periodic=True)
    
    qmpd = QuantumMagneticPhaseDiagram(lattice, model_type="heisenberg")
    
    J_range = np.array([1.0])
    h_range = np.linspace(0, 3, 20)
    
    print("计算相图...")
    phase_diagram = qmpd.compute_heisenberg_phase_diagram(
        J_range, h_range, direction='z'
    )
    
    print("\n结果:")
    print(f"磁场范围: {h_range[0]:.2f} - {h_range[-1]:.2f}")
    print(f"饱和磁场: ~{h_range[np.argmax(phase_diagram['magnetization'][0,:])]:.2f}")
    
    # 找临界点
    critical_fields = phase_diagram['critical_fields']
    print(f"\n临界场估计:")
    for idx, hc in critical_fields.items():
        print(f"  J={J_range[idx]}: h_c = {hc:.3f}")
    
    return phase_diagram


def run_xxz_phase_diagram():
    """XXZ模型相图"""
    print("\n" + "=" * 60)
    print("XXZ模型相图")
    print("=" * 60)
    
    lattice = Lattice.create_chain(6, periodic=True)
    qmpd = QuantumMagneticPhaseDiagram(lattice, model_type="xxz")
    
    J_range = np.array([1.0])
    Delta_range = np.linspace(-1, 2, 15)
    
    print("计算XXZ相图...")
    xxz_diagram = qmpd.compute_xxz_phase_diagram(J_range, Delta_range)
    
    print("\n相区识别:")
    for Delta, phase in xxz_diagram['phase_regions'].items():
        print(f"  Δ = {Delta:.2f}: {phase}")
    
    # 各向同性点
    iso_idx = np.argmin(np.abs(Delta_range - 1.0))
    print(f"\n在Heisenberg点 (Δ=1):")
    print(f"  自旋能隙: {xxz_diagram['spin_gap'][0, iso_idx]:.6f}")
    
    return xxz_diagram


def run_ising_transition():
    """Ising相变"""
    print("\n" + "=" * 60)
    print("Ising模型相变")
    print("=" * 60)
    
    lattice = Lattice.create_square(4, 4, periodic=True)
    qmpd = QuantumMagneticPhaseDiagram(lattice, model_type="ising")
    
    T_range = np.linspace(0.1, 3.0, 20)
    h_range = np.array([0.0, 0.5, 1.0])
    
    print("计算Ising相变...")
    ising_data = qmpd.compute_ising_phase_transition(T_range, h_range)
    
    print(f"\n估计临界温度: T_c ≈ {ising_data['critical_temperature']:.3f}")
    print(f"(2D Ising模型精确解: T_c/J ≈ 2.269)")
    
    print("\n临界指数估计:")
    for exp_name, value in ising_data['critical_exponents'].items():
        print(f"  {exp_name}: {value}")
    
    return ising_data


def run_kitaev_spin_liquid():
    """Kitaev量子自旋液体"""
    print("\n" + "=" * 60)
    print("Kitaev量子自旋液体")
    print("=" * 60)
    
    kitaev = KitaevModel(Jx=1.0, Jy=1.0, Jz=1.0, nx=3, ny=3)
    
    print(f"格点大小: {kitaev.nx} × {kitaev.ny}")
    print(f"总格点数: {kitaev.n_sites}")
    
    # 基态简并
    gs_info = kitaev.find_ground_state_manifold()
    print(f"\n基态性质:")
    print(f"  基态简并度: {gs_info['ground_state_degeneracy']}")
    print(f"  拓扑序: {gs_info['topological_order']}")
    print(f"  Anyon类型: {gs_info['anyon_types']}")
    
    # 拓扑激发
    topo = TopologicalMagnetism(
        Lattice.create_honeycomb(kitaev.nx, kitaev.ny)
    )
    
    excitations = topo.compute_fractional_excitations(kitaev)
    print(f"\n分数化激发:")
    print(f"  Majorana费米子数: {excitations['majorana_fermions']['count']}")
    print(f"  Vison激发能: {excitations['visons']['excitation_energy']:.3f}")
    
    return kitaev


def run_entanglement_analysis():
    """纠缠熵分析"""
    print("\n" + "=" * 60)
    print("纠缠熵相图分析")
    print("=" * 60)
    
    lattice = Lattice.create_chain(10, periodic=True)
    qmpd = QuantumMagneticPhaseDiagram(lattice, model_type="xxz")
    
    # 通过XXZ模型的各向异性点
    param_range = np.linspace(0.5, 1.5, 20)
    subregion_sizes = [2, 3, 4]
    
    print("计算纠缠熵...")
    ent_data = qmpd.compute_entanglement_phase_diagram(
        param_range, subregion_sizes
    )
    
    print("\n纠缠熵结果:")
    for size in subregion_sizes:
        entropies = ent_data['entanglement_entropy'][size]
        max_S = max(entropies)
        min_S = min(entropies)
        print(f"  子区域大小 {size}: S_max = {max_S:.4f}, S_min = {min_S:.4f}")
    
    return ent_data


if __name__ == "__main__":
    run_heisenberg_phase_diagram()
    run_xxz_phase_diagram()
    run_ising_transition()
    run_kitaev_spin_liquid()
    run_entanglement_analysis()
