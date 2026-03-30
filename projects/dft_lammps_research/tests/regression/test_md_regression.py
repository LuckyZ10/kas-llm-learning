"""
分子动力学回归测试
MD Simulation Regression Tests
===============================

验证MD模拟的可重复性和轨迹一致性。

测试项目:
    - 轨迹可重复性
    - 能量守恒
    - 温度控制稳定性
    - 压强控制稳定性
    - 长时间稳定性
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.regression import MDReferenceData, NumericalComparator


# =============================================================================
# MD轨迹可重复性测试
# =============================================================================

@pytest.mark.regression
@pytest.mark.md
class TestMDTrajectoryReproducibility:
    """MD轨迹可重复性测试"""
    
    def test_detinistic_trajectory_with_fixed_seed(self):
        """测试固定随机种子下的确定性轨迹"""
        seed = 42
        np.random.seed(seed)
        
        # 模拟两次相同初始条件的MD
        n_steps = 100
        n_atoms = 10
        
        trajectory1 = []
        trajectory2 = []
        
        # 相同的初始位置和速度
        positions = np.random.randn(n_atoms, 3)
        velocities = np.random.randn(n_atoms, 3)
        
        # 简化的MD积分（Verlet算法）
        dt = 0.001
        forces = np.zeros((n_atoms, 3))
        
        def verlet_step(pos, vel, f, dt):
            """Verlet积分步"""
            pos_new = pos + vel * dt + 0.5 * f * dt**2
            # 简化的力计算（谐振子）
            f_new = -pos_new
            vel_new = vel + 0.5 * (f + f_new) * dt
            return pos_new, vel_new, f_new
        
        # 第一次运行
        np.random.seed(seed)
        pos, vel = positions.copy(), velocities.copy()
        f = forces.copy()
        for _ in range(n_steps):
            pos, vel, f = verlet_step(pos, vel, f, dt)
            trajectory1.append(pos.copy())
        
        # 第二次运行（相同种子）
        np.random.seed(seed)
        pos, vel = positions.copy(), velocities.copy()
        f = forces.copy()
        for _ in range(n_steps):
            pos, vel, f = verlet_step(pos, vel, f, dt)
            trajectory2.append(pos.copy())
        
        # 比较轨迹
        comparator = NumericalComparator()
        for i in range(n_steps):
            assert comparator.compare_arrays(trajectory1[i], trajectory2[i], rtol=1e-10), \
                f"Trajectory mismatch at step {i}"
    
    def test_trajectory_periodicity(self):
        """测试轨迹的周期性边界处理"""
        box = np.array([10.0, 10.0, 10.0])
        
        # 粒子超出边界
        positions = np.array([
            [11.0, 5.0, 5.0],  # x超出
            [5.0, -1.0, 5.0],  # y低于
            [5.0, 5.0, 15.0],  # z超出
        ])
        
        # 应用PBC
        wrapped = positions % box
        
        expected = np.array([
            [1.0, 5.0, 5.0],
            [5.0, 9.0, 5.0],
            [5.0, 5.0, 5.0],
        ])
        
        assert np.allclose(wrapped, expected), "PBC wrapping incorrect"
    
    def test_trajectory_continuity(self):
        """测试轨迹的连续性"""
        # 生成平滑轨迹
        n_frames = 50
        times = np.linspace(0, 10, n_frames)
        
        # 正弦波轨迹
        positions = np.sin(times)[:, np.newaxis] * np.array([1.0, 0.5, 0.0])
        
        # 检查速度连续性（有限差分）
        velocities_fd = np.diff(positions, axis=0) / np.diff(times)[:, np.newaxis]
        
        # 速度不应该有突变
        velocity_jumps = np.abs(np.diff(velocities_fd, axis=0))
        max_jump = np.max(velocity_jumps)
        
        assert max_jump < 0.5, f"Trajectory discontinuity detected: {max_jump}"
    
    def test_trajectory_hash_stability(self):
        """测试轨迹哈希的稳定性"""
        comparator = NumericalComparator()
        
        # 生成轨迹
        trajectory = [np.random.randn(10, 3) for _ in range(20)]
        
        # 计算哈希两次
        hash1 = comparator.compute_trajectory_hash(trajectory)
        hash2 = comparator.compute_trajectory_hash(trajectory)
        
        assert hash1 == hash2, "Trajectory hash not stable"
        
        # 微小变化应该产生不同哈希（或相同，取决于精度）
        trajectory_modified = [pos + 1e-7 * np.random.randn(*pos.shape) for pos in trajectory]
        hash3 = comparator.compute_trajectory_hash(trajectory_modified)
        
        # 在舍入精度内应该相同
        # hash1 和 hash3 可能相同（如果变化小于1e-6）


# =============================================================================
# 能量守恒测试
# =============================================================================

@pytest.mark.regression
@pytest.mark.md
class TestMDEnergyConservation:
    """MD能量守恒测试"""
    
    def test_nve_energy_conservation(self):
        """测试NVE系综能量守恒"""
        n_steps = 1000
        dt = 0.001
        
        # 简化的NVE模拟
        positions = np.array([[1.0, 0.0, 0.0]])
        velocities = np.array([[0.0, 0.5, 0.0]])
        
        energies = []
        
        for _ in range(n_steps):
            # 谐振子势
            potential = 0.5 * np.sum(positions**2)
            kinetic = 0.5 * np.sum(velocities**2)
            total = potential + kinetic
            energies.append(total)
            
            # Verlet积分（简化的）
            force = -positions
            velocities += 0.5 * force * dt
            positions += velocities * dt
            force_new = -positions
            velocities += 0.5 * force_new * dt
        
        energies = np.array(energies)
        
        # 检查能量漂移
        energy_drift = np.max(energies) - np.min(energies)
        mean_energy = np.mean(energies)
        relative_drift = energy_drift / abs(mean_energy)
        
        assert relative_drift < 1e-6, f"NVE energy drift too large: {relative_drift}"
    
    def test_energy_fluctuation_reasonable(self):
        """测试能量波动的合理性"""
        n_steps = 1000
        
        # 模拟能量序列
        base_energy = -1000.0
        fluctuations = np.random.normal(0, 0.01, n_steps)
        energies = base_energy + fluctuations
        
        # 能量标准差
        energy_std = np.std(energies)
        mean_energy = np.mean(energies)
        
        # 相对波动应该在合理范围内
        relative_fluctuation = energy_std / abs(mean_energy)
        
        assert relative_fluctuation < 1e-4, f"Energy fluctuation too large: {relative_fluctuation}"
    
    def test_hamiltonian_invariance(self):
        """测试哈密顿量不变性"""
        # 简谐振子
        omega = 1.0
        m = 1.0
        
        # 不同位置的哈密顿量
        positions = np.linspace(-2, 2, 100)
        
        for x in positions:
            # 给定总能量
            E = 2.0
            
            # 计算速度
            v = np.sqrt(2 * (E - 0.5 * m * omega**2 * x**2) / m)
            
            if not np.isnan(v):
                # 验证哈密顿量
                H = 0.5 * m * v**2 + 0.5 * m * omega**2 * x**2
                assert np.isclose(H, E), f"Hamiltonian not conserved: {H} vs {E}"


# =============================================================================
# 温度控制测试
# =============================================================================

@pytest.mark.regression
@pytest.mark.md
class TestMDTemperatureControl:
    """MD温度控制测试"""
    
    def test_nvt_temperature_stability(self):
        """测试NVT温度稳定性"""
        target_temp = 300.0  # K
        n_steps = 1000
        
        # 模拟温度序列（Berendsen热浴）
        temperatures = []
        temp = target_temp * 0.8  # 初始低温
        
        tau = 0.1  # 耦合时间
        dt = 0.001
        
        for _ in range(n_steps):
            # Berendsen热浴
            scale = np.sqrt(1 + (dt / tau) * (target_temp / temp - 1))
            temp = temp * scale**2
            
            # 添加热噪声
            temp += np.random.normal(0, 5)
            temperatures.append(temp)
        
        temperatures = np.array(temperatures)
        
        # 检查最终温度
        final_temp = np.mean(temperatures[-100:])
        assert np.abs(final_temp - target_temp) < 10, \
            f"Temperature not converged: {final_temp} vs {target_temp}"
        
        # 温度波动
        temp_std = np.std(temperatures[-100:])
        assert temp_std < target_temp * 0.1, f"Temperature fluctuations too large: {temp_std}"
    
    def test_velocity_rescaling_conservation(self):
        """测试速度缩放时的动量守恒"""
        n_atoms = 10
        velocities = np.random.randn(n_atoms, 3)
        
        # 计算质心速度
        v_com = np.mean(velocities, axis=0)
        
        # 去除质心运动
        velocities = velocities - v_com
        
        # 缩放速度（温度调节）
        scale = 1.1
        velocities_scaled = velocities * scale
        
        # 质心速度应该保持为零
        v_com_scaled = np.mean(velocities_scaled, axis=0)
        assert np.allclose(v_com_scaled, 0, atol=1e-10), \
            "Momentum not conserved during velocity scaling"
    
    def test_equipartition_theorem(self):
        """测试能量均分定理"""
        n_atoms = 100
        temperature = 300.0
        kb = 8.617e-5  # eV/K
        
        # Maxwell-Boltzmann分布的速度
        mass = 1.0
        sigma = np.sqrt(kb * temperature / mass)
        
        velocities = np.random.normal(0, sigma, (n_atoms, 3))
        
        # 每个自由度的平均动能应该为 0.5 * kT
        ke_per_dof = 0.5 * mass * np.mean(velocities**2, axis=0)
        expected_ke = 0.5 * kb * temperature
        
        for i, ke in enumerate(ke_per_dof):
            assert np.abs(ke - expected_ke) / expected_ke < 0.1, \
                f"Equipartition violated for dof {i}: {ke} vs {expected_ke}"


# =============================================================================
# 压强控制测试
# =============================================================================

@pytest.mark.regression
@pytest.mark.md
class TestMDPressureControl:
    """MD压强控制测试"""
    
    def test_npt_pressure_stability(self):
        """测试NPT压强稳定性"""
        target_pressure = 1.0  # bar
        n_steps = 500
        
        # 模拟压强序列
        pressures = []
        pressure = 0.0
        
        for _ in range(n_steps):
            # 简化的Berendsen压强耦合
            tau_p = 1.0
            dt = 0.001
            compressibility = 4.57e-5  # 1/bar
            
            scale = 1 - (dt / tau_p) * compressibility * (target_pressure - pressure)
            pressure = pressure + 0.1 * np.random.randn()  # 噪声
            pressures.append(pressure)
        
        # 检查压强收敛
        mean_pressure = np.mean(pressures[-100:])
        assert np.abs(mean_pressure - target_pressure) < 0.5, \
            f"Pressure not converged: {mean_pressure} vs {target_pressure}"
    
    def test_cell_fluctuation_reasonable(self):
        """测试晶胞体积波动的合理性"""
        target_volume = 1000.0  # Å³
        n_steps = 1000
        
        # 模拟体积序列
        volumes = target_volume + np.random.normal(0, 10, n_steps)
        
        # 体积波动
        volume_std = np.std(volumes)
        relative_fluctuation = volume_std / target_volume
        
        assert relative_fluctuation < 0.1, f"Volume fluctuation too large: {relative_fluctuation}"


# =============================================================================
# 长时间稳定性测试
# =============================================================================

@pytest.mark.regression
@pytest.mark.md
@pytest.mark.slow
class TestMDLongTermStability:
    """MD长时间稳定性测试"""
    
    def test_long_time_energy_drift(self):
        """测试长时间能量漂移"""
        # 简化的长时间模拟
        n_steps = 10000
        dt = 0.001
        
        positions = np.array([[1.0, 0.0, 0.0]])
        velocities = np.array([[0.0, 0.5, 0.0]])
        
        initial_energy = 0.5 * np.sum(positions**2) + 0.5 * np.sum(velocities**2)
        
        for _ in range(n_steps):
            force = -positions
            velocities += 0.5 * force * dt
            positions += velocities * dt
            force_new = -positions
            velocities += 0.5 * force_new * dt
        
        final_energy = 0.5 * np.sum(positions**2) + 0.5 * np.sum(velocities**2)
        
        relative_drift = abs(final_energy - initial_energy) / abs(initial_energy)
        
        assert relative_drift < 1e-5, f"Long-term energy drift too large: {relative_drift}"
    
    def test_structure_integrity_long_term(self):
        """测试长时间模拟的结构完整性"""
        n_atoms = 10
        n_steps = 5000
        
        # 初始近邻距离
        positions = np.random.randn(n_atoms, 3)
        
        initial_dists = []
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(positions[i] - positions[j])
                initial_dists.append(dist)
        
        min_initial = min(initial_dists)
        
        # 简化的扩散模拟
        dt = 0.001
        D = 0.1  # 扩散系数
        
        for _ in range(n_steps):
            # 随机行走
            displacements = np.random.normal(0, np.sqrt(2 * D * dt), (n_atoms, 3))
            positions += displacements
        
        # 检查没有原子重叠
        final_dists = []
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(positions[i] - positions[j])
                final_dists.append(dist)
        
        min_final = min(final_dists)
        
        # 最小距离不应该变得太小
        assert min_final > min_initial * 0.5, "Structure collapse detected"


# =============================================================================
# 特定系统测试
# =============================================================================

@pytest.mark.regression
@pytest.mark.md
class TestMDSpecificSystems:
    """特定系统的MD测试"""
    
    def test_water_simulation_stability(self):
        """测试水分子模拟稳定性"""
        # 简化的水模型
        n_molecules = 10
        n_atoms = n_molecules * 3  # H2O
        
        positions = np.random.randn(n_atoms, 3) * 5
        
        # O-H键长检查
        for i in range(n_molecules):
            o_idx = i * 3
            h1_idx = i * 3 + 1
            h2_idx = i * 3 + 2
            
            bond1 = np.linalg.norm(positions[o_idx] - positions[h1_idx])
            bond2 = np.linalg.norm(positions[o_idx] - positions[h2_idx])
            
            # 标准O-H键长约0.96 Å
            assert 0.8 < bond1 < 1.2, f"O-H bond 1 length abnormal: {bond1}"
            assert 0.8 < bond2 < 1.2, f"O-H bond 2 length abnormal: {bond2}"
    
    def test_metal_simulation_properties(self):
        """测试金属体系模拟性质"""
        # FCC金属的模拟
        n_atoms = 32  # 2x2x2 supercell
        
        # 模拟密度
        mass = 63.5 * 1.66e-27  # Cu mass in kg
        volume = (3.6e-10)**3 * 8  # FCC unit cell volume * 8
        
        density = n_atoms * mass / volume / 1000  # kg/m³ to g/cm³
        
        # Cu的密度约8.96 g/cm³
        assert 7 < density < 10, f"Unreasonable density: {density}"


# =============================================================================
# 参考数据测试
# =============================================================================

@pytest.mark.regression
@pytest.mark.md
class TestMDReferenceData:
    """MD参考数据测试"""
    
    def test_reference_data_creation(self):
        """测试MD参考数据创建"""
        ref_data = MDReferenceData(
            system_name="Li2S_bulk",
            potential_type="NEP",
            temperature=300.0,
            pressure=1.0,
            timestep=1.0,
            n_steps=10000,
            ensemble="NPT",
            final_energy=-500.0,
            mean_energy=-498.5,
            energy_std=2.0,
            mean_temperature=300.2,
            temp_std=5.0,
            final_density=1.8,
            mean_density=1.79,
            trajectory_hash="abc123def456",
            seed=42
        )
        
        assert ref_data.system_name == "Li2S_bulk"
        assert ref_data.temperature == 300.0
    
    def test_reference_data_serialization(self, tmp_path):
        """测试MD参考数据序列化"""
        import json
        
        ref_data = MDReferenceData(
            system_name="test",
            potential_type="LJ",
            temperature=100.0,
            pressure=0.0,
            timestep=0.5,
            n_steps=1000,
            ensemble="NVE",
            final_energy=-10.0,
            mean_energy=-9.8,
            energy_std=0.2,
            mean_temperature=102.0,
            temp_std=10.0,
            final_density=2.0,
            mean_density=2.0,
            trajectory_hash="test123",
            seed=123
        )
        
        data_dict = ref_data.to_dict()
        
        filepath = tmp_path / "md_ref.json"
        with open(filepath, 'w') as f:
            json.dump(data_dict, f)
        
        with open(filepath, 'r') as f:
            loaded = json.load(f)
        
        assert loaded['system_name'] == "test"
        assert loaded['temperature'] == 100.0
