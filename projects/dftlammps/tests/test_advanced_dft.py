#!/usr/bin/env python3
"""
测试高级DFT模块的示例脚本

运行此脚本验证模块是否正确安装
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """测试模块导入"""
    print("Testing imports...")
    
    try:
        from dftlammps.dft_advanced import (
            OpticalPropertyWorkflow,
            MagneticPropertyWorkflow,
            DefectCalculationWorkflow,
            NonlinearResponseWorkflow,
        )
        print("✓ dft_advanced imports successful")
    except Exception as e:
        print(f"✗ dft_advanced imports failed: {e}")
        return False
    
    try:
        from dftlammps.solvation import (
            VASPsolWorkflow,
            CP2KSolvationWorkflow,
        )
        print("✓ solvation imports successful")
    except Exception as e:
        print(f"✗ solvation imports failed: {e}")
        return False
    
    try:
        from dftlammps.advanced_dft_workflow import (
            AdvancedDFTWorkflow,
            AdvancedDFTConfig,
            CalculationType,
        )
        print("✓ advanced_dft_workflow imports successful")
    except Exception as e:
        print(f"✗ advanced_dft_workflow imports failed: {e}")
        return False
    
    return True


def test_data_structures():
    """测试数据结构创建"""
    print("\nTesting data structures...")
    
    import numpy as np
    
    try:
        from dftlammps.dft_advanced import ElasticTensor, DielectricFunction
        
        # 弹性张量
        C = np.random.rand(6, 6) * 100
        C = (C + C.T) / 2  # 对称化
        elastic = ElasticTensor(C_ij=C)
        print(f"✓ ElasticTensor created: B={elastic.bulk_modulus:.2f} GPa")
        
        # 介电函数
        energy = np.linspace(0, 10, 100)
        eps_real = 1 + 5 / (1 + (energy - 3)**2)
        eps_imag = 3 * np.exp(-(energy - 3)**2 / 0.5)
        
        dielec = DielectricFunction(
            energy=energy,
            eps_real=eps_real,
            eps_imag=eps_imag,
            method="RPA",
            code="TEST"
        )
        print(f"✓ DielectricFunction created: {len(energy)} points")
        
        return True
    except Exception as e:
        print(f"✗ Data structure test failed: {e}")
        return False


def test_spin_configurations():
    """测试自旋构型生成"""
    print("\nTesting spin configurations...")
    
    try:
        from ase.build import bulk
        from dftlammps.dft_advanced import SpinConfigurationGenerator
        
        # 创建简单结构
        fe = bulk('Fe', 'bcc', a=2.87)
        
        # 生成铁磁构型
        fm = SpinConfigurationGenerator.ferromagnetic(fe, magnitude=2.2)
        print(f"✓ FM configuration: {len(fm.indices)} spins")
        
        # 生成反铁磁构型
        afm = SpinConfigurationGenerator.antiferromagnetic_simple(
            fe, list(range(len(fe)))
        )
        print(f"✓ AFM configuration: {len(afm.indices)} spins")
        
        return True
    except Exception as e:
        print(f"✗ Spin configuration test failed: {e}")
        return False


def test_defect_generation():
    """测试缺陷结构生成"""
    print("\nTesting defect generation...")
    
    try:
        from ase.build import bulk
        from dftlammps.dft_advanced import DefectStructureGenerator
        
        # ZnO结构
        zno = bulk('ZnO', 'zincblende', a=4.5)
        
        generator = DefectStructureGenerator(zno)
        supercell = generator.create_supercell((2, 2, 2))
        print(f"✓ Supercell created: {len(supercell)} atoms")
        
        # 创建空位
        vacancy = generator.create_vacancy(supercell, 0)
        print(f"✓ Vacancy created: {len(vacancy)} atoms")
        
        # 创建间隙
        interstitial = generator.create_interstitial(
            supercell, 'Zn', 'octahedral'
        )
        print(f"✓ Interstitial created: {len(interstitial)} atoms")
        
        return True
    except Exception as e:
        print(f"✗ Defect generation test failed: {e}")
        return False


def test_workflow_initialization():
    """测试工作流初始化"""
    print("\nTesting workflow initialization...")
    
    try:
        from dftlammps.dft_advanced import (
            OpticalPropertyWorkflow,
            MagneticPropertyWorkflow,
        )
        from dftlammps.solvation import VASPsolWorkflow, VASPsolConfig
        
        # 光学工作流
        opt_workflow = OpticalPropertyWorkflow('vasp')
        print("✓ OpticalPropertyWorkflow initialized")
        
        # 磁性工作流
        mag_workflow = MagneticPropertyWorkflow('vasp')
        print("✓ MagneticPropertyWorkflow initialized")
        
        # 溶剂化工作流
        solv_config = VASPsolConfig(eb_k=80.0)
        solv_workflow = VASPsolWorkflow(solv_config)
        print("✓ VASPsolWorkflow initialized")
        
        return True
    except Exception as e:
        print(f"✗ Workflow initialization failed: {e}")
        return False


def main():
    """运行所有测试"""
    print("=" * 60)
    print("Advanced DFT Module Test Suite")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Data Structures", test_data_structures),
        ("Spin Configurations", test_spin_configurations),
        ("Defect Generation", test_defect_generation),
        ("Workflow Initialization", test_workflow_initialization),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, r in results if r)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
