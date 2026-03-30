"""
Phase Field Tests
=================
相场模块测试套件

包含所有核心模型、求解器和耦合功能的测试。
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from phase_field.core.cahn_hilliard import CahnHilliardSolver, CahnHilliardConfig
from phase_field.core.allen_cahn import AllenCahnSolver, AllenCahnConfig
from phase_field.core.electrochemical import ElectrochemicalPhaseField, ElectrochemicalConfig
from phase_field.core.mechanochemistry import MechanoChemicalSolver, MechanoChemicalConfig


class TestCahnHilliard:
    """Cahn-Hilliard模型测试"""
    
    def test_initialization(self):
        """测试求解器初始化"""
        config = CahnHilliardConfig(nx=64, ny=64, dx=1.0)
        solver = CahnHilliardSolver(config)
        
        assert solver.config.nx == 64
        assert solver.config.ny == 64
        assert solver.ndim == 2
    
    def test_field_initialization(self):
        """测试场初始化"""
        config = CahnHilliardConfig(nx=32, ny=32, c0=0.4)
        solver = CahnHilliardSolver(config)
        solver.initialize_fields(seed=42)
        
        assert solver.c is not None
        assert solver.c.shape == (32, 32)
        assert 0.3 < solver.c.mean() < 0.5  # 接近c0
    
    def test_energy_computation(self):
        """测试能量计算"""
        config = CahnHilliardConfig(nx=32, ny=32)
        solver = CahnHilliardSolver(config)
        solver.initialize_fields()
        
        energy = solver.compute_energy()
        assert energy >= 0  # 能量应为正
    
    def test_evolution_step(self):
        """测试演化步骤"""
        config = CahnHilliardConfig(nx=32, ny=32, dt=0.001)
        solver = CahnHilliardSolver(config)
        solver.initialize_fields()
        
        c_before = solver.c.copy()
        info = solver.evolve_step()
        c_after = solver.c.copy()
        
        assert 'dc_max' in info
        assert not np.allclose(c_before, c_after)  # 场应该变化
    
    def test_mass_conservation(self):
        """测试质量守恒"""
        config = CahnHilliardConfig(nx=32, ny=32)
        solver = CahnHilliardSolver(config)
        solver.initialize_fields()
        
        mass_before = solver.c.mean()
        
        # 运行多步
        for _ in range(10):
            solver.evolve_step()
        
        mass_after = solver.c.mean()
        
        # 质量应该基本守恒
        assert abs(mass_before - mass_after) < 1e-6


class TestAllenCahn:
    """Allen-Cahn模型测试"""
    
    def test_initialization(self):
        """测试求解器初始化"""
        config = AllenCahnConfig(nx=64, ny=64)
        solver = AllenCahnSolver(config)
        
        assert solver.config.nx == 64
        assert solver.ndim == 2
    
    def test_nucleation_initialization(self):
        """测试形核初始化"""
        config = AllenCahnConfig(
            nx=64, ny=64,
            initial_structure="nucleation",
            n_order_params=3
        )
        solver = AllenCahnSolver(config)
        solver.initialize_fields()
        
        assert len(solver.eta) == 3
    
    def test_grain_growth(self):
        """测试晶粒生长"""
        config = AllenCahnConfig(
            nx=32, ny=32,
            initial_structure="grains",
            n_order_params=4
        )
        solver = AllenCahnSolver(config)
        solver.initialize_fields()
        
        # 记录初始晶粒尺寸
        size_before = solver.get_average_grain_size()
        
        # 运行演化
        for _ in range(50):
            solver.evolve_step()
        
        size_after = solver.get_average_grain_size()
        
        # 晶粒应该长大
        assert size_after >= size_before


class TestElectrochemical:
    """电化学相场测试"""
    
    def test_initialization(self):
        """测试电化学模型初始化"""
        config = ElectrochemicalConfig(nx=32, ny=32)
        solver = ElectrochemicalPhaseField(config)
        
        assert solver.config.temperature == 298.15
    
    def test_potential_initialization(self):
        """测试电势初始化"""
        config = ElectrochemicalConfig(nx=32, ny=32)
        solver = ElectrochemicalPhaseField(config)
        solver.initialize_fields()
        
        assert solver.phi is not None
        assert solver.phi.shape == (32, 32)
    
    def test_butler_volmer(self):
        """测试Butler-Volmer方程"""
        config = ElectrochemicalConfig(nx=32, ny=32)
        solver = ElectrochemicalPhaseField(config)
        
        eta = np.array([0.1, 0.2, 0.3])
        j0 = np.array([1.0, 1.0, 1.0])
        
        j = solver._butler_volmer(eta, j0)
        
        assert len(j) == 3
        assert np.all(j > 0)  # 正向反应


class TestMechanoChemical:
    """力学-化学耦合测试"""
    
    def test_initialization(self):
        """测试力学-化学模型初始化"""
        config = MechanoChemicalConfig(nx=32, ny=32)
        solver = MechanoChemicalSolver(config)
        
        assert solver.config.E == 100.0
        assert solver.config.nu == 0.25
    
    def test_stress_computation(self):
        """测试应力计算"""
        config = MechanoChemicalConfig(nx=32, ny=32)
        solver = MechanoChemicalSolver(config)
        solver.initialize_fields()
        
        stress = solver._compute_hydrostatic_stress()
        assert stress is not None


class TestCoupling:
    """耦合模块测试"""
    
    def test_parameter_transfer(self):
        """测试参数传递"""
        from phase_field.coupling.parameter_transfer import ParameterTransfer
        
        transfer = ParameterTransfer()
        
        dft_params = {
            'M': 1e-14,  # m²/s
            'kappa': 1e-9  # J/m
        }
        
        pf_params = transfer.transfer_from_dft({'thermodynamic': dft_params})
        
        assert 'kappa' in pf_params


class TestApplications:
    """应用模块测试"""
    
    def test_sei_simulator(self):
        """测试SEI模拟器"""
        from phase_field.applications.sei_growth import SEIGrowthSimulator, SEIConfig
        
        config = SEIConfig(nx=32, ny=32)
        simulator = SEIGrowthSimulator(config)
        simulator.initialize_fields()
        
        assert 'organic' in simulator.phi
        
        # 测试演化
        info = simulator.evolve_step()
        assert 'sei_thickness' in info
    
    def test_precipitation_simulator(self):
        """测试沉淀相模拟器"""
        from phase_field.applications.precipitation import PrecipitationSimulator, PrecipConfig
        
        config = PrecipConfig(nx=32, ny=32)
        simulator = PrecipitationSimulator(config)
        simulator.initialize_fields()
        
        # 测试演化
        info = simulator.evolve_step()
        assert 'n_precipitates' in info


def run_tests():
    """运行所有测试"""
    print("Running Phase Field Module Tests...")
    print("=" * 60)
    
    test_classes = [
        TestCahnHilliard,
        TestAllenCahn,
        TestElectrochemical,
        TestMechanoChemical,
        TestCoupling,
        TestApplications
    ]
    
    passed = 0
    failed = 0
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()
        
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    method = getattr(instance, method_name)
                    method()
                    print(f"  ✓ {method_name}")
                    passed += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {e}")
                    failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
