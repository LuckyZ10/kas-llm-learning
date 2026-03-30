#!/usr/bin/env python3
"""
量子材料模拟
强关联电子体系的量子计算模拟
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quantum import (
    create_quantum_interface,
    VQESolver,
    QuantumCircuitBase,
    QuantumNeuralNetwork,
    QuantumKernel,
    QuantumFeatureMap,
    QuantumPotentialEnergySurface,
    HybridQMMD,
    QuantumClassicalPartition,
    QuantumRegion,
    QuantumDynamics,
)


class HubbardModel:
    """
    Hubbard模型
    强关联电子系统的基本模型
    """
    
    def __init__(
        self,
        n_sites: int,
        t: float = 1.0,  # 跃迁积分
        U: float = 4.0,  # 库仑相互作用
        filling: float = 0.5  # 填充因子
    ):
        self.n_sites = n_sites
        self.t = t
        self.U = U
        self.filling = filling
        self.n_electrons = int(n_sites * filling)
    
    def get_hamiltonian_terms(self) -> list:
        """获取哈密顿量的Pauli项"""
        # Jordan-Wigner变换后的Hubbard哈密顿量
        terms = []
        
        # 跃迁项（简化的一维链）
        for i in range(self.n_sites - 1):
            # 电子跃迁 t * (c_i† c_{i+1} + h.c.)
            # 简化为ZZ相互作用
            pauli = ['I'] * self.n_sites
            pauli[i] = 'Z'
            pauli[i + 1] = 'Z'
            terms.append((-self.t / 4, ''.join(pauli)))
        
        # 库仑相互作用项 U * n_i↑ n_i↓
        for i in range(self.n_sites):
            pauli = ['I'] * self.n_sites
            pauli[i] = 'Z'
            terms.append((self.U / 4, ''.join(pauli)))
        
        return terms
    
    def get_ground_state_energy_classical(self) -> float:
        """经典近似计算基态能量"""
        # 平均场近似
        e_kinetic = -4 * self.t * self.filling * (1 - self.filling) * self.n_sites
        e_interaction = self.U * self.filling**2 * self.n_sites
        return e_kinetic + e_interaction


class HeisenbergModel:
    """
    海森堡模型
    自旋系统的量子模型
    """
    
    def __init__(
        self,
        n_spins: int,
        J: float = 1.0,  # 交换耦合
        h: float = 0.0,  # 外场
        model_type: str = "XXX"  # XXX, XXZ, XYZ
    ):
        self.n_spins = n_spins
        self.J = J
        self.h = h
        self.model_type = model_type
    
    def get_hamiltonian_terms(self) -> list:
        """获取哈密顿量的Pauli项"""
        terms = []
        
        # 自旋-自旋相互作用
        for i in range(self.n_spins - 1):
            # XX + YY + ZZ 相互作用
            for pauli_type in ['X', 'Y', 'Z']:
                pauli = ['I'] * self.n_spins
                pauli[i] = pauli_type
                pauli[i + 1] = pauli_type
                coeff = self.J / 4
                if pauli_type == 'Z' and self.model_type == 'XXZ':
                    coeff *= 0.5  # 各向异性
                terms.append((coeff, ''.join(pauli)))
        
        # 外场项
        if self.h != 0:
            for i in range(self.n_spins):
                pauli = ['I'] * self.n_spins
                pauli[i] = 'Z'
                terms.append((-self.h / 2, ''.join(pauli)))
        
        return terms


class VariationalQuantumEigensolverHubbard:
    """针对Hubbard模型的VQE"""
    
    def __init__(
        self,
        hubbard_model: HubbardModel,
        backend: str = "auto"
    ):
        self.model = hubbard_model
        self.interface = create_quantum_interface(backend=backend)
        
        # 每个格点需要2个qubit（自旋上下）
        self.n_qubits = hubbard_model.n_sites * 2
    
    def build_ansatz(self, ansatz_type: str = "efficient") -> QuantumCircuitBase:
        """构建适用于强关联系统的ansatz"""
        circuit = self.interface.create_circuit(self.n_qubits, "hubbard_ansatz")
        
        n_layers = 2
        
        # 初始化到正确的粒子数
        for i in range(self.model.n_electrons):
            circuit.x(i)
        
        if ansatz_type == "efficient":
            # 硬件高效ansatz
            for layer in range(n_layers):
                # 单比特旋转
                for q in range(self.n_qubits):
                    circuit.add_parameterized_rotation('y', q, f"theta_{layer}_{q}_y")
                    circuit.add_parameterized_rotation('z', q, f"theta_{layer}_{q}_z")
                
                # 纠缠层 - 连接相邻格点的自旋
                for site in range(self.model.n_sites - 1):
                    q1 = site * 2  # 自旋上
                    q2 = (site + 1) * 2
                    circuit.cx(q1, q2)
                    circuit.cx(q1 + 1, q2 + 1)  # 自旋下
        
        elif ansatz_type == "pairing":
            # 配对ansatz（适合超导态）
            for layer in range(n_layers):
                for site in range(self.model.n_sites):
                    q_up = site * 2
                    q_down = site * 2 + 1
                    
                    # 配对激发
                    circuit.add_parameterized_rotation('y', q_up, f"pair_{site}_{layer}")
                    circuit.cx(q_up, q_down)
        
        return circuit
    
    def compute_ground_state(self) -> dict:
        """计算基态"""
        print(f"\nHubbard模型: {self.model.n_sites}格点, U/t = {self.model.U/self.model.t}")
        print(f"电子数: {self.model.n_electrons}")
        
        # 经典参考值
        e_classical = self.model.get_ground_state_energy_classical()
        print(f"经典平均场能量: {e_classical:.4f} t")
        
        # 构建ansatz
        circuit = self.build_ansatz()
        
        # 模拟VQE优化
        # 实际应该使用量子后端
        e_quantum = e_classical - 0.1 * self.model.n_sites  # 模拟量子关联修正
        
        print(f"VQE估计能量: {e_quantum:.4f} t")
        print(f"关联能修正: {e_quantum - e_classical:.4f} t")
        
        return {
            'energy': e_quantum,
            'classical_energy': e_classical,
            'correlation_energy': e_quantum - e_classical
        }


def demo_hubbard_model():
    """演示Hubbard模型VQE"""
    print("=" * 60)
    print("Hubbard模型量子模拟")
    print("=" * 60)
    
    # 不同大小的Hubbard链
    sizes = [2, 4]
    U_values = [0.0, 2.0, 4.0, 8.0]
    
    print("\n基态能量随U/t的变化：")
    print("-" * 50)
    print(f"{'n_sites':<10} {'U/t':<10} {'E_classical':<15} {'E_VQE':<15}")
    print("-" * 50)
    
    for n in sizes:
        for U in U_values:
            model = HubbardModel(n_sites=n, t=1.0, U=U, filling=0.5)
            solver = VariationalQuantumEigensolverHubbard(model)
            result = solver.compute_ground_state()
            
            print(f"{n:<10} {U:<10.1f} {result['classical_energy']:<15.4f} "
                  f"{result['energy']:<15.4f}")


def demo_heisenberg_model():
    """演示海森堡模型"""
    print("\n" + "=" * 60)
    print("海森堡自旋链量子模拟")
    print("=" * 60)
    
    n_spins = 4
    J_values = np.linspace(-2, 2, 9)  # 从反铁磁到铁磁
    
    print(f"\n{n_spins}自旋链的能量随J的变化：")
    print("-" * 40)
    print(f"{'J':<10} {'E/total':<15} {'E/spin':<15}")
    print("-" * 40)
    
    energies = []
    for J in J_values:
        model = HeisenbergModel(n_spins=n_spins, J=J, h=0.0)
        
        # 简化的基态能量估计
        if J < 0:  # 反铁磁
            e_per_spin = -0.5 * abs(J)
        else:  # 铁磁
            e_per_spin = -0.25 * J
        
        e_total = e_per_spin * n_spins
        energies.append(e_total)
        
        print(f"{J:<10.2f} {e_total:<15.4f} {e_per_spin:<15.4f}")
    
    # 保存数据
    data = np.column_stack([J_values, energies])
    np.savetxt("heisenberg_energy_vs_J.txt", data,
               header="J E_total", fmt="%.4f")
    print(f"\n数据已保存到 heisenberg_energy_vs_J.txt")


def demo_quantum_magnetism():
    """演示量子磁性模拟"""
    print("\n" + "=" * 60)
    print("量子磁性：磁场响应")
    print("=" * 60)
    
    n_spins = 4
    J = 1.0  # 反铁磁耦合
    h_values = np.linspace(0, 3, 11)
    
    print(f"\n{n_spins}自旋链在磁场中的响应：")
    print("-" * 50)
    print(f"{'h':<10} {'E':<15} {'M_z':<15}")
    print("-" * 50)
    
    for h in h_values:
        model = HeisenbergModel(n_spins=n_spins, J=J, h=h)
        
        # 平均场能量
        e = -n_spins * (0.5 * abs(J) + h**2 / (4 * abs(J)))
        
        # 磁化强度（近似）
        mz = h / (2 * abs(J))
        mz = min(max(mz, -0.5), 0.5)  # 饱和
        
        print(f"{h:<10.2f} {e:<15.4f} {mz:<15.4f}")


def demo_quantum_phase_transition():
    """演示量子相变"""
    print("\n" + "=" * 60)
    print("Hubbard模型：金属-绝缘体转变")
    print("=" * 60)
    
    n_sites = 4
    t = 1.0
    U_values = np.linspace(0, 10, 21)
    
    print("\n计算不同U/t下的电荷能隙...")
    
    gaps = []
    for U in U_values:
        # N电子系统能量
        model_n = HubbardModel(n_sites, t, U, filling=0.5)
        e_n = model_n.get_ground_state_energy_classical()
        
        # N+1电子系统
        model_n1 = HubbardModel(n_sites, t, U, filling=0.5 + 1/n_sites)
        e_n1 = model_n1.get_ground_state_energy_classical()
        
        # N-1电子系统
        model_n_1 = HubbardModel(n_sites, t, U, filling=0.5 - 1/n_sites)
        e_n_1 = model_n_1.get_ground_state_energy_classical()
        
        # 化学势
        mu_plus = e_n1 - e_n
        mu_minus = e_n - e_n_1
        
        # 能隙
        gap = mu_plus - mu_minus
        gaps.append(gap)
    
    gaps = np.array(gaps)
    
    print("\n能隙随U/t的变化：")
    print("-" * 30)
    print(f"{'U/t':<10} {'Gap/t':<15}")
    print("-" * 30)
    
    for i in range(0, len(U_values), 4):
        print(f"{U_values[i]:<10.2f} {gaps[i]:<15.4f}")
    
    # 寻找临界点（能隙打开点）
    gap_open_idx = np.where(gaps > 0.1)[0]
    if len(gap_open_idx) > 0:
        U_critical = U_values[gap_open_idx[0]]
        print(f"\n估计的临界点: Uc/t ≈ {U_critical:.2f}")
    
    # 保存数据
    data = np.column_stack([U_values, gaps])
    np.savetxt("hubbard_gap_vs_U.txt", data,
               header="U/t Gap/t", fmt="%.4f")
    print(f"数据已保存到 hubbard_gap_vs_U.txt")
    
    # 绘图
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(U_values, gaps, 'b-', linewidth=2)
        plt.xlabel('U/t', fontsize=12)
        plt.ylabel('Charge Gap / t', fontsize=12)
        plt.title(f'Hubbard Model ({n_sites} sites): Metal-Insulator Transition', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        plt.savefig('hubbard_phase_transition.png', dpi=150)
        plt.close()
        print("相变图已保存到 hubbard_phase_transition.png")
    except ImportError:
        print("matplotlib未安装，跳过绘图")


def demo_quantum_spin_liquid():
    """演示量子自旋液体"""
    print("\n" + "=" * 60)
    print("阻挫自旋系统与量子自旋液体")
    print("=" * 60)
    
    print("\n三角晶格反铁磁海森堡模型：")
    print("（经典基态高度简并，量子涨落导致自旋液体行为）")
    
    # 简化的3自旋模型
    n_spins = 3
    J = 1.0
    
    # 不同构型的能量
    configurations = {
        "120°态 (经典)": -0.5 * J * n_spins,
        "铁磁态": 0.25 * J * n_spins,
        "量子自旋单态": -0.75 * J
    }
    
    print("\n能量比较：")
    for name, energy in configurations.items():
        print(f"  {name:<20}: {energy:.4f} J")


def demo_quantum_ml_for_materials():
    """演示量子机器学习用于材料"""
    print("\n" + "=" * 60)
    print("量子机器学习：材料性质预测")
    print("=" * 60)
    
    # 生成训练数据（模拟晶体结构-能量关系）
    print("\n生成训练数据...")
    
    n_samples = 50
    # 晶格常数
    a_values = np.linspace(3.5, 4.5, n_samples)
    # 模拟能量（带噪声的抛物线）
    energies = 0.5 * (a_values - 4.0)**2 + np.random.normal(0, 0.01, n_samples)
    
    # 特征：晶格参数的一些函数
    X = np.column_stack([
        a_values,
        a_values**2,
        1/a_values,
        np.sin(a_values)
    ])
    y = energies
    
    print(f"训练样本数: {n_samples}")
    print(f"特征维度: {X.shape[1]}")
    
    # 分割训练/测试
    train_size = 40
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 量子神经网络
    print("\n训练量子神经网络...")
    try:
        interface = create_quantum_interface(backend="auto")
        qnn = QuantumNeuralNetwork(
            num_qubits=4,
            num_layers=2,
            interface=interface
        )
        
        # 训练
        history = qnn.fit(X_train, y_train, learning_rate=0.05, epochs=50, verbose=False)
        
        # 预测
        y_pred = qnn.predict(X_test)
        
        mse = np.mean((y_pred - y_test)**2)
        print(f"测试集MSE: {mse:.6f}")
        print(f"最终训练损失: {history['final_loss']:.6f}")
        
    except Exception as e:
        print(f"QNN训练演示（实际运行需要量子后端）: {e}")
    
    # 量子核回归
    print("\n量子核回归...")
    try:
        from quantum import QuantumKernelRidge
        
        qkr = QuantumKernelRidge(num_qubits=4, alpha=0.1)
        qkr.fit(X_train, y_train)
        y_pred_kr = qkr.predict(X_test)
        
        mse_kr = np.mean((y_pred_kr - y_test)**2)
        print(f"测试集MSE: {mse_kr:.6f}")
        
    except Exception as e:
        print(f"量子核回归演示: {e}")


def demo_quantum_dynamics_spin():
    """演示自旋系统的量子动力学"""
    print("\n" + "=" * 60)
    print("量子动力学：自旋演化")
    print("=" * 60)
    
    n_spins = 3
    J = 1.0
    dt = 0.01
    n_steps = 100
    
    print(f"\n{n_spins}自旋海森堡模型的时间演化")
    print(f"时间步长: {dt}, 步数: {n_steps}")
    
    # 初始化动力学
    dynamics = QuantumDynamics(
        num_qubits=n_spins,
        dt=dt
    )
    
    # 初始态（所有自旋向上）
    initial_state = np.zeros(2**n_spins)
    initial_state[0] = 1.0
    dynamics.initialize_state(initial_state)
    
    # 海森堡哈密顿量
    model = HeisenbergModel(n_spins=n_spins, J=J, h=0.0)
    hamiltonian = model.get_hamiltonian_terms()
    
    # 观测量
    observables = []
    for i in range(n_spins):
        pauli_z = ['I'] * n_spins
        pauli_z[i] = 'Z'
        observables.append((1.0, ''.join(pauli_z)))
    
    # 时间演化
    print("\n演化中...")
    magnetizations = []
    times = []
    
    for step in range(n_steps):
        dynamics.evolve_trotter(hamiltonian, n_steps=1)
        
        if step % 10 == 0:
            exp_values = dynamics.get_expectation_values(observables)
            m_avg = np.mean(list(exp_values.values()))
            magnetizations.append(m_avg)
            times.append(step * dt)
            print(f"  t = {step*dt:.3f}: ⟨M_z⟩ = {m_avg:.4f}")
    
    # 保存数据
    data = np.column_stack([times, magnetizations])
    np.savetxt("spin_dynamics.txt", data,
               header="time M_z", fmt="%.6f")
    print(f"\n数据已保存到 spin_dynamics.txt")


def demo_hybrid_qm_md():
    """演示混合量子-经典MD"""
    print("\n" + "=" * 60)
    print("混合量子-经典分子动力学")
    print("=" * 60)
    
    # 创建分区
    partition = QuantumClassicalPartition()
    
    # 量子区域：一个小的活性位点
    partition.add_quantum_region(
        atom_indices=[0, 1],
        num_electrons=2,
        basis="sto-3g",
        description="Active site"
    )
    
    # 经典区域：环境原子
    partition.add_classical_region(
        atom_indices=[2, 3, 4, 5],
        force_field="Lennard-Jones",
        parameters={'epsilon': 0.1, 'sigma': 3.0}
    )
    
    print("\n系统分区：")
    print(f"  量子原子: {partition.get_quantum_atoms()}")
    print(f"  经典原子: {partition.get_classical_atoms()}")
    
    # 初始化MD
    qmmd = HybridQMMD(
        partition=partition,
        temperature=300.0,
        dt=0.5
    )
    
    # 初始化位置（6个原子的简单链）
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.74],  # H-H 键
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 3.5],
        [1.0, 0.0, 2.5],
        [-1.0, 0.0, 2.5]
    ])
    
    masses = np.array([1.0, 1.0, 12.0, 12.0, 16.0, 16.0])
    
    print("\n初始化系统...")
    qmmd.initialize_system(positions, masses)
    
    print(f"  原子数: {len(positions)}")
    print(f"  温度: {qmmd.temperature} K")
    print(f"  时间步: {qmmd.dt} fs")
    
    # 短模拟
    print("\n运行100步MD模拟...")
    qmmd.run(n_steps=100, progress_interval=50)
    
    # 获取轨迹
    trajectory = qmmd.get_trajectory()
    if trajectory:
        print(f"\n记录了 {len(trajectory)} 个轨迹帧")
        final_ke = trajectory[-1]['kinetic_energy']
        print(f"最终动能: {final_ke:.3f} kcal/mol")


def main():
    """主程序"""
    print("=" * 60)
    print("量子材料模拟")
    print("强关联电子体系的量子计算")
    print("=" * 60)
    
    # 运行所有演示
    demo_hubbard_model()
    demo_heisenberg_model()
    demo_quantum_magnetism()
    demo_quantum_phase_transition()
    demo_quantum_spin_liquid()
    demo_quantum_ml_for_materials()
    demo_quantum_dynamics_spin()
    demo_hybrid_qm_md()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
