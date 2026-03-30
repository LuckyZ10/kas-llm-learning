"""
量子计算模块测试
===============
验证所有模块功能正确性
"""

import numpy as np
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quantum_info.quantum_chemistry_qc import (
    FermionOperator, QubitOperator, UCCSD, VQE, QuantumSimulator,
    h2_molecule_hamiltonian, compute_ground_state_exact
)

from quantum_info.quantum_materials import (
    Lattice, HubbardModel, SpinSystem, HaldaneModel
)

from quantum_error.surface_code import (
    SurfaceCodeLattice, MWPM_Decoder, SurfaceCodeSimulation
)


def test_fermion_operator():
    """测试费米子算符"""
    print("测试费米子算符...")
    
    # 创建产生算符
    op = FermionOperator([(1.0, [(0, 1), (1, 0)])])
    
    # 测试加法
    op2 = op + op
    assert len(op2.terms) > 0
    
    # 测试乘法
    op3 = op * 2.0
    assert abs(op3.terms[0][0] - 2.0) < 1e-10
    
    print("  ✓ FermionOperator 测试通过")


def test_qubit_operator():
    """测试量子比特算符"""
    print("测试量子比特算符...")
    
    # 创建Pauli-Z算符
    z_op = QubitOperator([(1.0, [(0, 'Z')])])
    
    # 转换为矩阵
    mat = z_op.to_matrix(1)
    expected = np.array([[1, 0], [0, -1]], dtype=complex)
    assert np.allclose(mat, expected)
    
    # 测试Pauli乘法
    x_op = QubitOperator([(1.0, [(0, 'X')])])
    y_op = QubitOperator([(1.0, [(0, 'Y')])])
    
    result = x_op * y_op
    assert len(result.terms) > 0
    
    print("  ✓ QubitOperator 测试通过")


def test_uccsd():
    """测试UCCSD ansatz"""
    print("测试UCCSD ansatz...")
    
    # 创建UCCSD
    uccsd = UCCSD(n_orbitals=2, n_electrons=2)
    
    # 检查参数数量
    assert uccsd.n_params > 0
    
    # 生成初始猜测
    params = uccsd.initial_guess("zeros")
    assert len(params) == uccsd.n_params
    
    # 生成电路
    circuit = uccsd.get_ansatz_circuit(params)
    assert isinstance(circuit, list)
    
    print(f"  ✓ UCCSD测试通过 (参数数: {uccsd.n_params})")


def test_quantum_simulator():
    """测试量子模拟器"""
    print("测试量子模拟器...")
    
    sim = QuantumSimulator(n_qubits=2)
    
    # 测试Hadamard门
    sim.apply_gate("H", [0], None)
    
    # 测试CNOT
    sim.apply_gate("CNOT", [0, 1], None)
    
    # 测试期望值
    z_op = QubitOperator([(1.0, [(0, 'Z')])])
    exp_val = sim.expectation_value(z_op)
    
    assert isinstance(exp_val, (complex, float, int))
    
    print("  ✓ QuantumSimulator 测试通过")


def test_h2_hamiltonian():
    """测试H2分子哈密顿量"""
    print("测试H2分子哈密顿量...")
    
    # 创建H2哈密顿量
    h2 = h2_molecule_hamiltonian(bond_length=0.74)
    
    # 检查算符
    assert len(h2.terms) > 0
    
    # 计算精确基态
    energy, state = compute_ground_state_exact(h2)
    
    print(f"  ✓ H2基态能量: {energy:.6f} Ha")
    assert energy < 0  # 束缚态能量为负


def test_lattice():
    """测试晶格创建"""
    print("测试晶格创建...")
    
    # 一维链
    chain = Lattice.create_chain(4, periodic=True)
    assert chain.n_sites == 4
    assert len(chain.neighbors) == 4
    
    # 二维方格
    square = Lattice.create_square(3, 3, periodic=True)
    assert square.n_sites == 9
    
    print(f"  ✓ Lattice测试通过 (链: {chain.n_sites}格点, 方格: {square.n_sites}格点)")


def test_hubbard_model():
    """测试Hubbard模型"""
    print("测试Hubbard模型...")
    
    lattice = Lattice.create_chain(4, periodic=True)
    hubbard = HubbardModel(lattice, t=1.0, U=4.0, n_electrons=4)
    
    # 检查哈密顿量
    assert hubbard.hamiltonian.shape[0] == 2**hubbard.n_qubits
    
    # 计算基态能量
    energy = hubbard.get_ground_state_energy()
    
    # 对角化
    energies, states = hubbard.diagonalize(n_states=2)
    
    print(f"  ✓ Hubbard模型测试通过 (基态能量: {energy:.4f})")


def test_spin_system():
    """测试自旋系统"""
    print("测试自旋系统...")
    
    lattice = Lattice.create_chain(4, periodic=True)
    
    # Heisenberg模型
    heisenberg = SpinSystem(lattice, model_type="heisenberg", J=1.0)
    energies, states = heisenberg.diagonalize(n_states=1)
    
    # 计算磁化强度
    mag = heisenberg.compute_magnetization(states[:, 0], 'z')
    
    print(f"  ✓ 自旋系统测试通过 (基态能量: {energies[0]:.4f}, 磁化: {mag:.4f})")


def test_haldane_model():
    """测试Haldane模型"""
    print("测试Haldane模型...")
    
    haldane = HaldaneModel(t=1.0, t2=0.1, phi=np.pi/2, M=0.0)
    
    # 计算能带结构
    k_points = np.array([[0, 0], [np.pi, 0], [np.pi, np.pi], [0, 0]])
    energies = haldane.get_band_structure(k_points)
    
    assert energies.shape[1] == 2  # 两个能带
    
    print(f"  ✓ Haldane模型测试通过 ({len(k_points)} k点)")


def test_surface_code_lattice():
    """测试表面码格点"""
    print("测试表面码格点...")
    
    lattice = SurfaceCodeLattice(distance=3)
    
    assert lattice.d == 3
    assert len(lattice.data_positions) == 9  # d x d
    
    print(f"  ✓ 表面码格点测试通过 (数据量子比特: {len(lattice.data_positions)})")


def test_surface_code_simulation():
    """测试表面码模拟"""
    print("测试表面码模拟...")
    
    sim = SurfaceCodeSimulation(distance=3, physical_error_rate=0.01)
    
    # 运行几个周期
    results = []
    for _ in range(5):
        result = sim.run_cycle()
        results.append(result)
    
    print(f"  ✓ 表面码模拟测试通过 ({len(results)} 周期)")


def test_mwpm_decoder():
    """测试MWPM解码器"""
    print("测试MWPM解码器...")
    
    lattice = SurfaceCodeLattice(distance=3)
    decoder = MWPM_Decoder(lattice)
    
    # 创建简单综合征
    from quantum_error.surface_code import Syndrome
    syndrome = Syndrome(
        x_syndrome=np.zeros((2, 2)),
        z_syndrome=np.array([[1, 0], [0, 1]])
    )
    
    # 解码
    x_errors, z_errors = decoder.decode(syndrome)
    
    print(f"  ✓ MWPM解码器测试通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("量子计算模块测试")
    print("=" * 60)
    
    tests = [
        test_fermion_operator,
        test_qubit_operator,
        test_uccsd,
        test_quantum_simulator,
        test_h2_hamiltonian,
        test_lattice,
        test_hubbard_model,
        test_spin_system,
        test_haldane_model,
        test_surface_code_lattice,
        test_surface_code_simulation,
        test_mwpm_decoder,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} 失败: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
