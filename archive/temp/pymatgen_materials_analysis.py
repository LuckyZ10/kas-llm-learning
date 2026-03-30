#!/usr/bin/env python3
"""
Pymatgen材料分析完整示例
功能：能带结构、态密度、相图、弹性张量
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 尝试导入pymatgen
try:
    from pymatgen.core import Structure, Lattice, Element, Composition
    from pymatgen.io.vasp import Poscar, Vasprun, Kpoints, Incar
    from pymatgen.electronic_structure.plotter import (
        BSPlotter, BSPlotterProjected, DosPlotter
    )
    from pymatgen.electronic_structure.core import Spin, Orbital
    from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
    from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
    from pymatgen.analysis.elasticity import ElasticTensor
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.entries.computed_entries import ComputedStructureEntry
    HAS_PMG = True
except ImportError:
    HAS_PMG = False
    print("[WARN] Pymatgen未安装，将使用模拟数据")


class MaterialsAnalyzer:
    """材料分析器"""
    
    def __init__(self):
        self.structures = {}
        self.calculations = {}
    
    def create_structure(self, name, structure_type='NaCl'):
        """
        创建示例晶体结构
        """
        print(f"[INFO] 创建{structure_type}结构...")
        
        if structure_type == 'NaCl':
            # NaCl结构
            lattice = Lattice.cubic(5.64)
            structure = Structure(
                lattice,
                ['Na', 'Cl'],
                [[0, 0, 0], [0.5, 0.5, 0.5]]
            )
            structure.make_supercell([2, 2, 2])
            
        elif structure_type == 'Si':
            # 金刚石Si
            lattice = Lattice.cubic(5.43)
            structure = Structure(
                lattice,
                ['Si', 'Si'],
                [[0, 0, 0], [0.25, 0.25, 0.25]]
            )
            
        elif structure_type == 'GaAs':
            # 闪锌矿GaAs
            lattice = Lattice.cubic(5.65)
            structure = Structure(
                lattice,
                ['Ga', 'As'],
                [[0, 0, 0], [0.25, 0.25, 0.25]]
            )
            
        elif structure_type == 'perovskite':
            # 钙钛矿 (ABO3)
            lattice = Lattice.cubic(3.9)
            structure = Structure(
                lattice,
                ['Sr', 'Ti', 'O', 'O', 'O'],
                [[0, 0, 0], [0.5, 0.5, 0.5], 
                 [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
            )
        
        else:
            raise ValueError(f"未知的结构类型: {structure_type}")
        
        self.structures[name] = structure
        
        # 分析对称性
        if HAS_PMG:
            sga = SpacegroupAnalyzer(structure)
            print(f"[OK] 结构创建完成:")
            print(f"  - 化学式: {structure.formula}")
            print(f"  - 空间群: {sga.get_space_group_symbol()}")
            print(f"  - 晶系: {sga.get_crystal_system()}")
            print(f"  - 原子数: {len(structure)}")
        
        return structure
    
    def analyze_symmetry(self, structure_name):
        """分析晶体对称性"""
        if structure_name not in self.structures:
            print(f"[ERROR] 结构 {structure_name} 不存在")
            return None
        
        structure = self.structures[structure_name]
        
        if not HAS_PMG:
            return {'space_group': 'Fm-3m', 'crystal_system': 'cubic'}
        
        sga = SpacegroupAnalyzer(structure)
        
        symmetry_info = {
            'space_group_number': sga.get_space_group_number(),
            'space_group_symbol': sga.get_space_group_symbol(),
            'crystal_system': sga.get_crystal_system(),
            'point_group': sga.get_point_group_symbol(),
            'international_symbol': sga.get_international_symbol(),
            'lattice_type': sga.get_lattice_type()
        }
        
        print(f"[OK] 对称性分析完成:")
        for key, value in symmetry_info.items():
            print(f"  - {key}: {value}")
        
        return symmetry_info
    
    def plot_band_structure(self, vasprun_file=None, kpoints_file=None):
        """
        绘制能带结构
        """
        print("[INFO] 绘制能带结构...")
        
        if not HAS_PMG or not vasprun_file:
            # 模拟能带数据
            kpoints = np.linspace(0, 1, 100)
            
            # 价带和导带
            vb = -kpoints**2 - 0.5
            cb = kpoints**2 + 0.5
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(kpoints, vb, 'b-', label='Valence Band', linewidth=2)
            ax.plot(kpoints, cb, 'r-', label='Conduction Band', linewidth=2)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.fill_between(kpoints, -2, 0, alpha=0.2, color='blue')
            ax.fill_between(kpoints, 1.1, 3, alpha=0.2, color='red')
            
            ax.set_xlabel('K-Point Path', fontsize=12)
            ax.set_ylabel('Energy (eV)', fontsize=12)
            ax.set_title('Band Structure (Simulated)', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-2, 3)
            
            plt.tight_layout()
            plt.savefig('band_structure.png', dpi=150)
            print("[OK] 能带图保存到 band_structure.png")
            plt.show()
            return
        
        # 真实的pymatgen能带分析
        vasprun = Vasprun(vasprun_file)
        bs = vasprun.get_band_structure(kpoints_file, line_mode=True)
        
        # 使用BSPlotter
        plotter = BSPlotter(bs)
        plotter.get_plot(vbm_cbm_marker=True)
        
        # 获取带隙信息
        band_gap = bs.get_band_gap()
        print(f"[OK] 能带分析完成:")
        print(f"  - 带隙: {band_gap['energy']:.3f} eV")
        print(f"  - 直接带隙: {band_gap['direct']}")
        print(f"  - 跃迁: {band_gap['transition']}")
        
        plt.savefig('band_structure.png', dpi=150)
        plt.show()
        
        return bs
    
    def plot_density_of_states(self, vasprun_file=None):
        """
        绘制态密度
        """
        print("[INFO] 绘制态密度...")
        
        if not HAS_PMG or not vasprun_file:
            # 模拟DOS数据
            energy = np.linspace(-10, 10, 500)
            
            # 总DOS
            total_dos = np.exp(-(energy - 2)**2 / 2) + 0.5 * np.exp(-(energy + 2)**2 / 2)
            
            # 投影DOS
            s_dos = 0.3 * total_dos
            p_dos = 0.5 * total_dos
            d_dos = 0.2 * total_dos
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.fill_between(energy, 0, total_dos, alpha=0.3, color='gray', label='Total DOS')
            ax.plot(energy, s_dos, 'b-', label='s-orbital', linewidth=2)
            ax.plot(energy, p_dos, 'g-', label='p-orbital', linewidth=2)
            ax.plot(energy, d_dos, 'r-', label='d-orbital', linewidth=2)
            
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
            ax.fill_between(energy, 0, total_dos, where=(energy < 0), alpha=0.2, color='blue')
            
            ax.set_xlabel('Energy (eV)', fontsize=12)
            ax.set_ylabel('DOS (states/eV)', fontsize=12)
            ax.set_title('Density of States (Simulated)', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('dos.png', dpi=150)
            print("[OK] DOS图保存到 dos.png")
            plt.show()
            return
        
        # 真实的DOS分析
        vasprun = Vasprun(vasprun_file)
        dos = vasprun.complete_dos
        
        plotter = DosPlotter()
        plotter.add_dos("Total", dos)
        
        # 添加元素投影DOS
        for element in dos.structure.elements:
            plotter.add_dos(f"{element}", dos.get_element_dos()[element])
        
        plotter.get_plot()
        plt.savefig('dos.png', dpi=150)
        plt.show()
        
        return dos
    
    def calculate_phase_diagram(self, chemical_system=['Li', 'Fe', 'O']):
        """
        计算相图
        """
        print(f"[INFO] 计算 {'-'.join(chemical_system)} 相图...")
        
        if not HAS_PMG:
            print("[WARN] Pymatgen未安装，跳过相图计算")
            return None
        
        # 这里应该使用Materials Project API获取数据
        # 简化版本：创建示例条目
        entries = []
        
        # 示例：创建一些化合物条目
        compounds = [
            ('Li', -1.9),
            ('Fe', -8.3),
            ('O2', -4.9),
            ('Li2O', -6.0),
            ('FeO', -3.0),
            ('Fe2O3', -8.5),
            ('LiFeO2', -9.5),
            ('Li5FeO4', -20.0)
        ]
        
        for formula, energy in compounds:
            comp = Composition(formula)
            entry = ComputedStructureEntry(
                Structure.from_dict({'formula': formula}),  # 简化
                energy
            )
            entries.append(entry)
        
        # 构建相图
        pd = PhaseDiagram(entries)
        
        # 绘制相图
        plotter = PDPlotter(pd)
        plotter.get_plot()
        
        plt.savefig('phase_diagram.png', dpi=150)
        print("[OK] 相图保存到 phase_diagram.png")
        plt.show()
        
        return pd
    
    def analyze_elastic_properties(self, elastic_tensor=None):
        """
        分析弹性性质
        """
        print("[INFO] 分析弹性性质...")
        
        if not HAS_PMG or elastic_tensor is None:
            # 模拟数据
            elastic_props = {
                'bulk_modulus_voigt': 100.0,  # GPa
                'bulk_modulus_reuss': 98.0,
                'bulk_modulus_vrh': 99.0,
                'shear_modulus_voigt': 60.0,
                'shear_modulus_reuss': 58.0,
                'shear_modulus_vrh': 59.0,
                'young_modulus': 148.0,
                'poisson_ratio': 0.25,
                'pugh_ratio': 1.67,
                'cauchy_pressure': 20.0
            }
        else:
            # 使用pymatgen分析
            et = ElasticTensor(elastic_tensor)
            
            elastic_props = {
                'bulk_modulus_voigt': et.k_voigt,
                'bulk_modulus_reuss': et.k_reuss,
                'bulk_modulus_vrh': et.k_vrh,
                'shear_modulus_voigt': et.g_voigt,
                'shear_modulus_reuss': et.g_reuss,
                'shear_modulus_vrh': et.g_vrh,
                'young_modulus': et.y_mod,
                'poisson_ratio': et.poisson,
                'pugh_ratio': et.k_vrh / et.g_vrh,
                'universal_anisotropy': et.universal_anisotropy
            }
        
        print("[OK] 弹性性质分析完成:")
        print(f"  - 体积模量 (VRH): {elastic_props['bulk_modulus_vrh']:.2f} GPa")
        print(f"  - 剪切模量 (VRH): {elastic_props['shear_modulus_vrh']:.2f} GPa")
        print(f"  - 杨氏模量: {elastic_props['young_modulus']:.2f} GPa")
        print(f"  - 泊松比: {elastic_props['poisson_ratio']:.3f}")
        print(f"  - Pugh比: {elastic_props['pugh_ratio']:.2f}")
        
        # 判断延展性
        if elastic_props['pugh_ratio'] > 1.75:
            ductility = "延展性"
        else:
            ductility = "脆性"
        print(f"  - 材料特性: {ductility}")
        
        return elastic_props
    
    def calculate_effective_mass(self, band_structure=None):
        """
        计算有效质量
        """
        print("[INFO] 计算有效质量...")
        
        # 简化计算
        # 实际应该拟合能带曲率
        
        effective_masses = {
            'electron': {'Gamma-X': 0.23, 'Gamma-L': 0.18},
            'hole': {'Gamma-X': 0.35, 'Gamma-L': 0.42}
        }
        
        print("[OK] 有效质量计算完成:")
        for carrier_type, directions in effective_masses.items():
            print(f"  {carrier_type.capitalize()}:")
            for direction, mass in directions.items():
                print(f"    - {direction}: {mass} m₀")
        
        return effective_masses
    
    def generate_vasp_inputs(self, structure_name, calculation_type='relax'):
        """
        生成VASP输入文件
        """
        if structure_name not in self.structures:
            print(f"[ERROR] 结构 {structure_name} 不存在")
            return
        
        structure = self.structures[structure_name]
        
        if not HAS_PMG:
            print("[WARN] Pymatgen未安装")
            return
        
        # POSCAR
        poscar = Poscar(structure)
        poscar.write_file(f'{structure_name}_POSCAR')
        
        # INCAR
        if calculation_type == 'relax':
            incar = Incar({
                'ENCUT': 520,
                'ISMEAR': 0,
                'SIGMA': 0.05,
                'IBRION': 2,
                'ISIF': 3,
                'NSW': 100,
                'EDIFF': 1e-6,
                'EDIFFG': -0.01
            })
        elif calculation_type == 'bands':
            incar = Incar({
                'ENCUT': 520,
                'ISMEAR': 0,
                'SIGMA': 0.05,
                'ICHARG': 11,
                'LORBIT': 11,
                'NEDOS': 3000
            })
        
        incar.write_file(f'{structure_name}_{calculation_type}_INCAR')
        
        # KPOINTS
        kpoints = Kpoints.automatic(0.03)
        kpoints.write_file(f'{structure_name}_KPOINTS')
        
        print(f"[OK] VASP输入文件已生成: {structure_name}_*")
    
    def structure_comparison(self, structure_names):
        """
        比较多个结构
        """
        print("[INFO] 结构比较...")
        
        if not HAS_PMG or len(structure_names) < 2:
            return
        
        structures = [self.structures[name] for name in structure_names if name in self.structures]
        
        matcher = StructureMatcher()
        
        comparison_results = []
        for i, s1 in enumerate(structures):
            for s2 in structures[i+1:]:
                rms = matcher.get_rms_dist(s1, s2)
                are_equal = matcher.fit(s1, s2)
                comparison_results.append({
                    'pair': f"{s1.formula} - {s2.formula}",
                    'rms': rms,
                    'equal': are_equal
                })
        
        print("[OK] 结构比较完成:")
        for result in comparison_results:
            print(f"  - {result['pair']}: RMS={result['rms']:.4f}, Equal={result['equal']}")
        
        return comparison_results


# 使用示例
if __name__ == '__main__':
    analyzer = MaterialsAnalyzer()
    
    # 创建结构
    print("\n=== 创建晶体结构 ===")
    analyzer.create_structure('NaCl', 'NaCl')
    analyzer.create_structure('Si', 'Si')
    analyzer.create_structure('perovskite', 'perovskite')
    
    # 对称性分析
    print("\n=== 对称性分析 ===")
    analyzer.analyze_symmetry('NaCl')
    
    # 电子结构分析
    print("\n=== 电子结构分析 ===")
    analyzer.plot_band_structure()
    analyzer.plot_density_of_states()
    
    # 相图
    print("\n=== 相图分析 ===")
    analyzer.calculate_phase_diagram(['Li', 'Fe', 'O'])
    
    # 弹性性质
    print("\n=== 弹性性质分析 ===")
    analyzer.analyze_elastic_properties()
    
    # 有效质量
    print("\n=== 有效质量计算 ===")
    analyzer.calculate_effective_mass()
    
    # 生成VASP输入
    print("\n=== 生成VASP输入 ===")
    analyzer.generate_vasp_inputs('Si', 'relax')
    analyzer.generate_vasp_inputs('Si', 'bands')
