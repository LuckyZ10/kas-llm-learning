# DFT与其他软件接口指南

## 概述

现代计算材料科学需要多软件协同工作。本指南介绍DFT计算 (VASP/QE) 与主流材料模拟软件 (LAMMPS, ASE, Pymatgen) 的接口方法。

---

## 1. ASE (Atomic Simulation Environment)

### 安装

```bash
# 推荐安装
pip install ase

# 完整安装 (含可视化)
pip install ase[graphs] matplotlib

# 从conda
conda install -c conda-forge ase
```

### VASP接口

```python
#!/usr/bin/env python3
"""ASE与VASP接口示例"""

from ase import Atoms
from ase.build import bulk, surface
from ase.calculators.vasp import Vasp
from ase.optimize import BFGS
from ase.io import read, write

# 1. 基础VASP计算
def vasp_basic_calculation():
    """基础VASP计算设置"""
    
    # 构建结构
    si = bulk('Si', crystalstructure='diamond', a=5.43)
    
    # 设置VASP计算器
    calc = Vasp(
        xc='PBE',           # 泛函
        encut=400,          # 截断能
        kpts=[4, 4, 4],     # k点网格
        istart=0,           # 从头开始
        icharg=2,
        ismear=0,
        sigma=0.05,
        nsw=0,              # 单点计算
        directory='vasp_calc',  # 计算目录
        command='mpirun -np 4 vasp_std'  # 运行命令
    )
    
    si.calc = calc
    
    # 获取能量
    energy = si.get_potential_energy()
    print(f"Total energy: {energy:.4f} eV")
    
    # 获取力
    forces = si.get_forces()
    print(f"Max force: {np.abs(forces).max():.4f} eV/Å")
    
    return si

# 2. 结构优化
def vasp_relaxation():
    """使用ASE优化结构"""
    
    atoms = read('initial.vasp')
    
    calc = Vasp(
        xc='PBE',
        encut=500,
        kpts=[6, 6, 6],
        ibrion=2,           # 离子优化
        isif=3,             # 优化晶胞+离子
        nsw=100,
        ediffg=-0.02,       # 力收敛标准
        directory='relax'
    )
    
    atoms.calc = calc
    
    # 使用ASE优化器 (可选)
    # optimizer = BFGS(atoms, trajectory='relax.traj')
    # optimizer.run(fmax=0.02)
    
    # 或让VASP自己优化
    energy = atoms.get_potential_energy()
    
    # 保存优化后结构
    write('optimized.vasp', atoms, direct=True)
    
    return atoms

# 3. 能带计算工作流
def vasp_band_structure():
    """能带计算完整流程"""
    
    from ase.dft.kpoints import bandpath
    
    # 读取结构
    atoms = read('POSCAR')
    
    # Step 1: SCF计算
    calc_scf = Vasp(
        xc='PBE',
        encut=500,
        kpts=[8, 8, 8],
        istart=0,
        ismear=0,
        sigma=0.05,
        lorbit=11,
        directory='band_scf'
    )
    
    atoms.calc = calc_scf
    scf_energy = atoms.get_potential_energy()
    
    # Step 2: 能带计算
    # 高对称路径
    path = bandpath('GXWKGLUWLK', atoms.cell, npoints=100)
    
    calc_band = Vasp(
        xc='PBE',
        encut=500,
        kpts=path.kpts,     # 线形k点
        istart=1,           # 读取WAVECAR
        icharg=11,          # 非自洽
        ismear=0,
        sigma=0.05,
        lorbit=11,
        directory='band_nscf'
    )
    
    atoms.calc = calc_band
    atoms.get_potential_energy()
    
    # 读取能带数据
    from ase.calculators.vasp import VaspBandStructure
    bs = VaspBandStructure(directory='band_nscf')
    
    # 绘图
    import matplotlib.pyplot as plt
    bs.plot()
    plt.savefig('band_structure.png', dpi=150)
    
    return bs

# 4. 批量计算
def batch_calculations():
    """批量计算多个结构"""
    
    from ase.io import iread
    import os
    
    structures = iread('structures.xyz')  # 多个结构文件
    
    results = []
    for i, atoms in enumerate(structures):
        calc_dir = f'calc_{i:03d}'
        
        calc = Vasp(
            xc='PBE',
            encut=400,
            kpts=[4, 4, 4],
            directory=calc_dir
        )
        
        atoms.calc = calc
        
        try:
            energy = atoms.get_potential_energy()
            results.append({
                'index': i,
                'energy': energy,
                'formula': atoms.get_chemical_formula()
            })
            print(f"{i}: {atoms.get_chemical_formula()} = {energy:.4f} eV")
        except Exception as e:
            print(f"{i}: Failed - {e}")
    
    return results
```

### QE接口

```python
#!/usr/bin/env python3
"""ASE与Quantum ESPRESSO接口"""

from ase.calculators.espresso import Espresso
from ase.build import molecule

def qe_basic_calculation():
    """基础QE计算"""
    
    # 构建水分子
    h2o = molecule('H2O')
    h2o.center(vacuum=10)
    
    # QE计算器设置
    pseudopotentials = {
        'H': 'H.pbe-rrkjus_psl.1.0.0.UPF',
        'O': 'O.pbe-n-rrkjus_psl.1.0.0.UPF'
    }
    
    calc = Espresso(
        command='mpirun -np 4 pw.x -in PREFIX.pwi > PREFIX.pwo',
        pseudopotentials=pseudopotentials,
        pseudo_dir='/path/to/pseudo',
        kpts=[1, 1, 1],  # Gamma点
        input_data={
            'control': {
                'calculation': 'scf',
                'prefix': 'h2o',
                'outdir': './tmp',
            },
            'system': {
                'ecutwfc': 50,
                'ecutrho': 400,
            },
            'electrons': {
                'conv_thr': 1.0e-8,
            }
        },
        directory='qe_calc'
    )
    
    h2o.calc = calc
    energy = h2o.get_potential_energy()
    print(f"H2O energy: {energy:.4f} eV")
    
    return h2o

def qe_relaxation():
    """QE结构优化"""
    
    from ase.io import read
    
    atoms = read('input.xyz')
    
    calc = Espresso(
        command='mpirun -np 8 pw.x -in PREFIX.pwi > PREFIX.pwo',
        pseudopotentials={'Si': 'Si.pbe-n-kjpaw_psl.1.0.0.UPF'},
        pseudo_dir='./pseudo',
        kpts=[4, 4, 4],
        input_data={
            'control': {
                'calculation': 'relax',
                'prefix': 'si_relax',
                'outdir': './tmp',
            },
            'system': {
                'ecutwfc': 60,
                'ecutrho': 480,
            },
            'electrons': {
                'conv_thr': 1.0e-8,
            },
            'ions': {
                'ion_dynamics': 'bfgs',
            }
        }
    )
    
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    
    # ASE自动读取优化后结构
    write('relaxed.xyz', atoms)
    
    return atoms

# QE + ASE高级功能
def qe_phonon_ase():
    """使用ASE进行QE声子计算"""
    
    from ase.phonons import Phonons
    from ase.io import read
    
    atoms = read('POSCAR')
    
    # 设置QE计算器
    calc = Espresso(
        pseudopotentials={'Si': 'Si.pbe-n-kjpaw_psl.1.0.0.UPF'},
        pseudo_dir='./pseudo',
        kpts=[4, 4, 4],
        input_data={
            'control': {
                'calculation': 'scf',
                'prefix': 'si_ph',
            },
            'system': {
                'ecutwfc': 60,
            },
        }
    )
    
    # ASE有限位移法
    ph = Phonons(atoms, calc, supercell=(2, 2, 2))
    ph.run()
    
    # 读取力并计算声子
    ph.read(acoustic=True)
    
    # 获取DOS
    dos = ph.get_dos(kpts=[20, 20, 20])
    
    # 绘图
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(dos.get_energies(), dos.get_weights())
    ax.set_xlabel('Energy (meV)')
    ax.set_ylabel('DOS')
    plt.savefig('phonon_dos.png')
    
    return ph
```

---

## 2. Pymatgen

### 安装

```bash
pip install pymatgen

# 完整功能
pip install pymatgen[extra]

# 或conda
conda install -c conda-forge pymatgen
```

### 结构处理

```python
#!/usr/bin/env python3
"""Pymatgen结构处理与DFT接口"""

from pymatgen.core import Structure, Lattice, Molecule
from pymatgen.io.vasp import Poscar, Incar, Kpoints, Potcar
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet, MPNonSCFSet
from pymatgen.io.qe import PWInput

def pymatgen_vasp_workflow():
    """Pymatgen VASP工作流"""
    
    # 1. 构建结构
    lattice = Lattice.cubic(5.43)
    si = Structure(
        lattice,
        ['Si', 'Si'],
        [[0, 0, 0], [0.25, 0.25, 0.25]]
    )
    
    # 2. 使用Materials Project预设
    relax_set = MPRelaxSet(si)
    
    # 写入VASP输入
    relax_set.write_input('Si_relax')
    # 自动生成: INCAR, POSCAR, KPOINTS, POTCAR
    
    # 3. 自定义INCAR
    user_incar_settings = {
        'ENCUT': 600,
        'ISMEAR': 0,
        'SIGMA': 0.05,
        'NPAR': 4
    }
    
    custom_set = MPRelaxSet(
        si,
        user_incar_settings=user_incar_settings,
        user_kpoints_settings={'grid_density': 1000}
    )
    custom_set.write_input('Si_relax_custom')
    
    # 4. 多步工作流
    # Step 1: Relax
    relax = MPRelaxSet(si)
    relax.write_input('step1_relax')
    
    # Step 2: Static (读取step1的CONTCAR)
    # 假设已经完成relax
    relaxed_si = Structure.from_file('step1_relax/CONTCAR')
    static = MPStaticSet(relaxed_si)
    static.write_input('step2_static')
    
    # Step 3: Band
    band = MPNonSCFSet.from_prev_calc('step2_static', 
                                       mode='Line',
                                       standardize=True)
    band.write_input('step3_band')

def analyze_vasp_results():
    """分析VASP结果"""
    
    from pymatgen.io.vasp import Vasprun, Outcar
    from pymatgen.electronic_structure.plotter import DosPlotter, BSPlotter
    
    # 读取vasprun.xml
    vasprun = Vasprun('vasprun.xml', parse_dos=True, parse_eigen=True)
    
    # 获取带隙
    band_gap = vasprun.get_band_structure().get_band_gap()
    print(f"Band gap: {band_gap['energy']:.3f} eV")
    print(f"Direct: {band_gap['direct']}")
    
    # 获取DOS
    dos = vasprun.complete_dos
    
    # 绘制DOS
    plotter = DosPlotter()
    plotter.add_dos("Total", dos)
    plotter.add_dos_dict(dos.get_element_dos())
    plotter.save_plot('dos.png')
    
    # 能带结构
    bs = vasprun.get_band_structure(line_mode=True)
    bs_plotter = BSPlotter(bs)
    bs_plotter.save_plot('band.png')
    
    # 读取力 (Outcar)
    outcar = Outcar('OUTCAR')
    print(f"Total magnetization: {outcar.total_mag}")
    print(f"Forces shape: {outcar.read_table_pattern('total-force').shape}")

def pymatgen_qe_interface():
    """Pymatgen QE接口"""
    
    from pymatgen.io.qe import PWInput
    
    # 构建结构
    si = Structure(
        Lattice.cubic(5.43),
        ['Si', 'Si'],
        [[0, 0, 0], [0.25, 0.25, 0.25]]
    )
    
    # QE输入
    pw_input = PWInput(
        si,
        pseudo={'Si': 'Si.pbe-n-kjpaw_psl.1.0.0.UPF'},
        control={'calculation': 'scf', 'prefix': 'si'},
        system={'ecutwfc': 60, 'ecutrho': 480},
        kpoints_grid=[4, 4, 4]
    )
    
    # 写入文件
    pw_input.write_file('pw.in')
    
    # 或获取字符串
    input_str = pw_input.__str__()
    print(input_str)

# Materials Project数据库接口
def query_materials_project():
    """查询Materials Project数据库"""
    
    from pymatgen.ext.matproj import MPRester
    
    # 需要API key (从materialsproject.org获取)
    mpr = MPRester("YOUR_API_KEY")
    
    # 按化学式查询
    results = mpr.query("Si", properties=["task_id", "pretty_formula", 
                                         "band_gap", "formation_energy_per_atom"])
    
    for r in results:
        print(f"{r['task_id']}: {r['pretty_formula']}, "
              f"Gap={r['band_gap']:.2f} eV, "
              f"E_form={r['formation_energy_per_atom']:.3f} eV/atom")
    
    # 获取结构
    structure = mpr.get_structure_by_material_id("mp-149")  # Si
    structure.to(filename='Si_mp.vasp')
    
    # 获取能带
    bandstructure = mpr.get_bandstructure_by_material_id("mp-149")
    print(f"Band gap: {bandstructure.get_band_gap()['energy']:.3f} eV")
```

### 缺陷计算

```python
#!/usr/bin/env python3
"""Pymatgen缺陷计算工具"""

from pymatgen.analysis.defects.core import Vacancy, Substitution, Interstitial
from pymatgen.analysis.defects.generators import VacancyGenerator

def generate_defects():
    """生成缺陷结构"""
    
    from pymatgen.core import Structure
    
    # 读取完美晶体
    bulk = Structure.from_file('POSCAR')
    
    # 生成空位
    vac_gen = VacancyGenerator(bulk)
    
    for defect in vac_gen:
        print(f"Defect: {defect.name}")
        print(f"Multiplicity: {defect.multiplicity}")
        
        # 获取缺陷结构
        defect_structure = defect.get_supercell_structure(
            sc_mat=np.diag([3, 3, 3])
        )
        
        # 保存
        defect_structure.to(
            filename=f"vacancy_{defect.site.specie}.vasp"
        )
```

---

## 3. LAMMPS接口

### VASP与LAMMPS势函数拟合

```python
#!/usr/bin/env python3
"""DFT数据拟合LAMMPS势函数"""

import numpy as np
from ase import Atoms
from ase.calculators.vasp import Vasp
from ase.build import bulk
from ase.io import write

def generate_training_data():
    """生成DFT训练数据 (用于拟合势函数)"""
    
    configurations = []
    
    # 1. 平衡结构
    si = bulk('Si', crystalstructure='diamond', a=5.43)
    configurations.append(('equilibrium', si))
    
    # 2. 体积变形 (EOS数据)
    for scale in np.linspace(0.9, 1.1, 11):
        atoms = si.copy()
        atoms.set_cell(atoms.cell * scale, scale_atoms=True)
        configurations.append((f'vol_{scale:.2f}', atoms))
    
    # 3. 剪切变形
    for shear in np.linspace(-0.05, 0.05, 5):
        atoms = si.copy()
        cell = atoms.cell.copy()
        cell[0, 1] += shear * cell[1, 1]
        atoms.set_cell(cell, scale_atoms=True)
        configurations.append((f'shear_{shear:.3f}', atoms))
    
    # 4. 原子位移 (声子训练)
    from ase.build import make_supercell
    supercell = make_supercell(si, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    
    np.random.seed(42)
    for i in range(20):
        atoms = supercell.copy()
        displacement = np.random.randn(len(atoms), 3) * 0.1  # 0.1 Å
        atoms.positions += displacement
        configurations.append((f'phonon_{i}', atoms))
    
    # 运行DFT计算
    calc = Vasp(
        xc='PBE',
        encut=400,
        kpts=[4, 4, 4],
        ismear=0,
        sigma=0.05,
        nsw=0
    )
    
    training_data = []
    
    for name, atoms in configurations:
        print(f"Calculating {name}...")
        atoms.calc = calc
        
        try:
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            stress = atoms.get_stress()
            
            training_data.append({
                'name': name,
                'atoms': atoms,
                'energy': energy,
                'forces': forces,
                'stress': stress
            })
            
            # 保存为extended XYZ (含能量和力)
            atoms.info['energy'] = energy
            atoms.arrays['forces'] = forces
            write(f'training/{name}.xyz', atoms)
            
        except Exception as e:
            print(f"  Failed: {e}")
    
    return training_data

# 使用MTP拟合 (可选)
# 见 references/ml_potential_training.md
```

### LAMMPS输入生成

```python
#!/usr/bin/env python3
"""生成LAMMPS输入文件"""

def generate_lammps_input(potential_type='SW'):
    """生成LAMMPS输入文件"""
    
    if potential_type == 'SW':
        potential_block = """
# Stillinger-Weber potential for Si
pair_style sw
pair_coeff * * Si.sw Si
"""
    elif potential_type == 'Tersoff':
        potential_block = """
# Tersoff potential
pair_style tersoff
pair_coeff * * SiC.tersoff Si
"""
    elif potential_type == 'MEAM':
        potential_block = """
# MEAM potential
pair_style meam
pair_coeff * * library.meam Si Si.meam Si
"""
    elif potential_type == 'ML':
        potential_block = """
# Machine learning potential (MTP/ACE)
pair_style mlip almtp
pair_coeff * * Si.mtp Si
"""
    
    lammps_input = f"""
# LAMMPS input for Si MD
units metal
atom_style atomic
boundary p p p

# Read structure
read_data si.data

{potential_block}

# Neighbor list
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

# Thermodynamic output
thermo 100
thermo_style custom step temp pe ke etotal press vol

# Velocity initialization
velocity all create 300.0 12345

# Ensemble
fix 1 all nvt temp 300.0 300.0 0.1

# Run
timestep 0.001  # 1 fs
run 10000

# Output
dump 1 all custom 100 dump.lammpstrj id type x y z fx fy fz
"""
    
    with open('in.si_md', 'w') as f:
        f.write(lammps_input)
    
    print("LAMMPS input written to in.si_md")

# ASE与LAMMPS联用
def ase_lammps_interface():
    """ASE调用LAMMPS"""
    
    from ase.calculators.lammpsrun import LAMMPS
    from ase import Atoms
    from ase.build import bulk
    from ase.optimize import BFGS
    
    # 构建结构
    si = bulk('Si', crystalstructure='diamond', a=5.43)
    
    # LAMMPS计算器
    calc = LAMMPS(
        command='lmp',
        pair_style='sw',
        pair_coeff=['* * Si.sw Si'],
        specorder=['Si'],
        keep_alive=True
    )
    
    si.calc = calc
    
    # 优化
    optimizer = BFGS(si)
    optimizer.run(fmax=0.01)
    
    print(f"Optimized energy: {si.get_potential_energy():.4f} eV")
    print(f"Lattice constant: {si.cell[0,0]:.4f} Å")
```

---

## 4. 多软件联合工作流

```python
#!/usr/bin/env python3
"""多软件联合工作流示例"""

def dft_ml_md_workflow():
    """DFT → ML势 → 大尺度MD工作流"""
    
    workflow = """
    完整工作流:
    
    Step 1: DFT数据生成
    - 使用ASE+VASP生成训练数据
    - 覆盖: 平衡结构、EOS、声子、缺陷
    
    Step 2: ML势训练
    - 使用MTP/DeepMD/ACE
    - 主动学习迭代
    - 验证DFT精度
    
    Step 3: LAMMPS大尺度模拟
    - MD: 纳米尺度模拟
    - MC: 相图计算
    - MS: 缺陷演化
    
    Step 4: 验证
    - 对比DFT基准
    - 与实验对比
    """
    
    print(workflow)

def vasp_qe_comparison():
    """VASP与QE交叉验证"""
    
    from ase.build import bulk
    from ase.calculators.vasp import Vasp
    from ase.calculators.espresso import Espresso
    
    si = bulk('Si', a=5.43)
    
    # VASP计算
    vasp_calc = Vasp(
        xc='PBE',
        encut=500,
        kpts=[6, 6, 6],
        directory='compare_vasp'
    )
    
    si_vasp = si.copy()
    si_vasp.calc = vasp_calc
    e_vasp = si_vasp.get_potential_energy()
    
    # QE计算
    qe_calc = Espresso(
        pseudopotentials={'Si': 'Si.pbe-n-kjpaw_psl.1.0.0.UPF'},
        pseudo_dir='./pseudo',
        kpts=[6, 6, 6],
        input_data={
            'control': {'prefix': 'si', 'outdir': './tmp'},
            'system': {'ecutwfc': 60}
        },
        directory='compare_qe'
    )
    
    si_qe = si.copy()
    si_qe.calc = qe_calc
    e_qe = si_qe.get_potential_energy()
    
    print(f"VASP energy: {e_vasp:.6f} eV")
    print(f"QE energy:   {e_qe:.6f} eV")
    print(f"Difference:  {abs(e_vasp - e_qe):.6f} eV")
```

---

## 参考资源

- ASE文档: https://wiki.fysik.dtu.dk/ase/
- Pymatgen文档: https://pymatgen.org/
- LAMMPS手册: https://docs.lammps.org/
- ASE教程: https://wiki.fysik.dtu.dk/ase/tutorials/
- Pymatgen examples: https://pymatgen.org/examples.html

---

*文档版本: 1.0*
*最后更新: 2026-03-08*
