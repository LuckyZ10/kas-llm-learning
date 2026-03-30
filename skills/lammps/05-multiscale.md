# 05. 多尺度模拟

## 目录
- [多尺度方法概述](#多尺度方法概述)
- [QM/MM耦合](#qmmm耦合)
- [DFT-MD耦合](#dft-md耦合)
- [机器学习势多尺度](#机器学习势多尺度)
- [自适应分辨率](#自适应分辨率)
- [耦合接口与工具](#耦合接口与工具)

---

## 多尺度方法概述

### 多尺度层次

```
长度尺度与计算方法:
├── 埃级 (0.1-1 nm)
│   ├── 第一性原理: DFT, GW, Quantum Chemistry
│   └── 紧束缚: DFTB, LATTE
├── 纳米级 (1-100 nm)
│   ├── 全原子MD: AMBER, CHARMM, LAMMPS
│   └── 机器学习MD: DeepMD, ACE
├── 微米级 (100 nm - 10 μm)
│   ├── 粗粒化MD: Martini, MARTINI3, SIRAH
│   └── 耗散粒子动力学: DPD
└── 宏观级 (>10 μm)
    ├── 有限元: FEM
    └── 连续介质: CFD
```

### LAMMPS多尺度能力

```
LAMMPS多尺度框架:
├── QM/MM
│   ├── fix qmmm (简化接口)
│   └── 外部耦合 (LibQM/MM, ChemShell)
├── 机器学习
│   ├── DeepMD-kit
│   ├── TorchANI
│   └── NequIP/MACE
├── 粗粒化
│   ├── atom_style hybrid
│   ├── pair_style hybrid
│   └── 多分辨率方法
└── 外部耦合
    ├── MDI (MolSSI Driver Interface)
    ├── I-Pi
    └── Socket通讯
```

---

## QM/MM耦合

### 1. 理论框架

```
QM/MM能量分解:
E_total = E_QM(QM_atoms) + E_MM(MM_atoms) + E_QM-MM(coupling)

E_QM-MM = E_vdw(QM-MM) + E_elec(QM-MM)

边界处理:
├── 简单截断 (link atom)
├── 边界原子 (pseudobond)
└── 浸没边界 (ONIOM-style)
```

### 2. LAMMPS QM/MM接口

```lammps
# 需要编译with QM/MM支持
# 目前主要通过外部工具实现

# 方法1: 使用fix external
fix 1 all external pf/callback 10

# 方法2: 使用MDI接口 (推荐)
units real
atom_style full

read_data system.data

# 定义QM区域
group qm_atoms id 1 2 3 4 5
group mm_atoms subtract all qm_atoms

# MM力场
pair_style lj/cut/coul/long 10.0
pair_coeff * * 0.1 3.0
kspace_style pppm 1.0e-4

# MDI耦合到QM代码
fix 1 qm_atoms mdi/qm

# MM区域温度控制
fix 2 mm_atoms nvt temp 300.0 300.0 100.0

timestep 0.5
run 10000
```

### 3. 与DFTB+耦合

```python
# dftb_lammps.py
# 使用DFTB+作为QM引擎

from lammps import lammps
import subprocess
import numpy as np

class DFTBCalculator:
    def __init__(self, skf_path, charge=0):
        self.skf_path = skf_path
        self.charge = charge
    
    def calculate_forces(self, positions, atom_types, box):
        # 生成DFTB+输入
        self.write_dftb_input(positions, atom_types, box)
        
        # 运行DFTB+
        subprocess.run(['dftb+', 'dftb_in.hsd'])
        
        # 读取力
        forces = self.read_forces()
        energy = self.read_energy()
        
        return energy, forces
    
    def write_dftb_input(self, positions, atom_types, box):
        with open('dftb_in.hsd', 'w') as f:
            f.write(f"""
Geometry = GenFormat {{
    {len(positions)} C
    {' '.join(atom_types)}
    {self.gen_format_positions(positions, box)}
}}

Driver = {}

Hamiltonian = DFTB {{
    SCC = Yes
    SCCTolerance = 1e-6
    MaxSCCIterations = 200
    Charge = {self.charge}
    SlaterKosterFiles = Type2FileNames {{
        Prefix = "{self.skf_path}/"
        Separator = "-"
        Suffix = ".skf"
    }}
}}
""")

# LAMMPS回调函数
def qm_callback(caller, timestep, nlocal, ids, pos, fexternal):
    # 提取QM原子位置和类型
    qm_pos = np.array(pos).reshape(-1, 3)
    
    # 调用DFTB+
    energy, forces = dftb_calc.calculate_forces(qm_pos, atom_types, box)
    
    # 返回力到LAMMPS
    fexternal[:] = forces.flatten()
    
    return energy

# 初始化
lmp = lammps()
dftb_calc = DFTBCalculator('/path/to/skf')

# 注册回调
lmp.set_callback("external", qm_callback)

# 运行
lmp.file('input.lammps')
```

### 4. 与ORCA耦合

```python
# orca_lammps.py
# 使用ORCA作为QM引擎

import subprocess
import numpy as np

class ORCACalculator:
    def __init__(self, method='B3LYP', basis='def2-SVP', charge=0, mult=1):
        self.method = method
        self.basis = basis
        self.charge = charge
        self.mult = mult
    
    def calculate(self, positions, elements):
        # 生成ORCA输入
        with open('orca.inp', 'w') as f:
            f.write(f"""! {self.method} {self.basis} EnGrad

* xyz {self.charge} {self.mult}
""")
            for elem, pos in zip(elements, positions):
                f.write(f"{elem} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}\n")
            f.write("*\n")
        
        # 运行ORCA
        subprocess.run(['orca', 'orca.inp'], capture_output=True)
        
        # 读取结果
        energy = self.read_energy()
        forces = self.read_forces()
        
        return energy, forces
    
    def read_forces(self):
        forces = []
        with open('orca.engrad', 'r') as f:
            lines = f.readlines()
            # 解析力...
        return np.array(forces)
```

### 5. ChemShell/LibQM接口

```lammps
# 使用ChemShell进行QM/MM
# chemshell_qmmm.tcl

source $env(CHEMSH)/chemshell.tcl

# 加载体系
set fragments [load_fragments system.pun]
set qm_region [get_region $fragments qm_atoms]
set mm_region [get_region $fragments mm_atoms]

# 设置QM引擎
set qm_theory [create_theory orca {
    method: B3LYP
    basis: def2-SVP
}]

# 设置MM引擎 (LAMMPS)
set mm_theory [create_theory lammps {
    potential: charmm
    parameters: system.prm
}]

# QM/MM耦合
set qmmm_theory [create_qmmm_theory $qm_theory $mm_theory {
    qm_atoms: $qm_region
    embedding: electrostatic
}]

# 优化
optimize_coordinates $qmmm_theory

# 动力学
run_dynamics $qmmm_theory {
    ensemble: nvt
    temperature: 300
    timestep: 0.5
    steps: 10000
}
```

---

## DFT-MD耦合

### 1. CP2K与LAMMPS耦合

```bash
# 使用CP2K进行Born-Oppenheimer MD
# 然后与LAMMPS分析工具结合

# CP2K输入: dft_md.inp
&FORCE_EVAL
  METHOD Quickstep
  &DFT
    BASIS_SET_FILE_NAME BASIS_MOLOPT
    POTENTIAL_FILE_NAME POTENTIAL
    &MGRID
      CUTOFF 400
      REL_CUTOFF 50
    &END MGRID
    &QS
      EPS_DEFAULT 1.0E-12
    &END QS
    &SCF
      SCF_GUESS ATOMIC
      EPS_SCF 1.0E-6
      MAX_SCF 50
    &END SCF
    &XC
      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL
    &END XC
  &END DFT
  
  &SUBSYS
    &CELL
      ABC 12.0 12.0 12.0
    &END CELL
    &COORD
      @INCLUDE 'coords.xyz'
    &END COORD
  &END SUBSYS
&END FORCE_EVAL

&MOTION
  &MD
    ENSEMBLE NVT
    STEPS 10000
    TIMESTEP 0.5
    TEMPERATURE 300.0
    &THERMOSTAT
      TYPE NOSE
      REGION MASSIVE
      &NOSE
        TIMECON 100.0
      &END NOSE
    &END THERMOSTAT
  &END MD
&END MOTION
```

```python
# 后处理CP2K轨迹
# cp2k_postprocess.py

import numpy as np
from ase.io import read

# 读取CP2K轨迹
trajectory = read('dft-md-pos-1.xyz', index=':')

# 转换为LAMMPS格式用于分析
for i, atoms in enumerate(trajectory):
    atoms.write(f'dump_{i}.lammpstrj', format='lammpsdump')

# RDF分析
from ase import Atoms
from ase.geometry.analysis import Analysis

analyzer = Analysis(trajectory)
rdf = analyzer.get_rdf(rmax=10.0, nbins=100, elements=['O', 'O'])
```

### 2. VASP与LAMMPS耦合

```python
# vasp_lammps.py
# ASE接口

from ase.calculators.vasp import Vasp
from ase.md.langevin import Langevin
from ase import units
from ase.io import read

# 设置VASP计算器
calc = Vasp(
    xc='PBE',
    encut=400,
    ismear=0,
    sigma=0.1,
    ibrion=0,
    nsw=10000,
    potim=1.0,
    tebeg=300,
    mdalgo=2,  # Nose-Hoover
    smass=0,
)

# 读取结构
atoms = read('POSCAR')
atoms.calc = calc

# ASE MD (可导出到LAMMPS分析)
dyn = Langevin(atoms, 1.0*units.fs, 300*units.kB, 0.002)

def write_lammps_dump():
    atoms.write('traj.lammpstrj', append=True)

dyn.attach(write_lammps_dump, interval=100)
dyn.run(10000)
```

### 3. Quantum ESPRESSO与LAMMPS

```bash
# QE输入: pw.in
&CONTROL
  calculation='md'
  restart_mode='from_scratch'
  pseudo_dir='./pseudo/'
  outdir='./tmp/'
  dt=20
  nstep=1000
  iprint=10
  tempw=300.0
  ion_dynamics='verlet'
/
&SYSTEM
  ibrav=1
  celldm(1)=12.0
  nat=32
  ntyp=2
  ecutwfc=30.0
  ecutrho=240.0
/
&ELECTRONS
  conv_thr=1.0d-6
/
&IONS
  ion_temperature='rescaling'
/
ATOMIC_SPECIES
O 16.0 O.pbe-n-rrkjus_psl.0.1.UPF
H 1.0 H.pbe-rrkjus_psl.0.1.UPF
ATOMIC_POSITIONS (angstrom)
O  1.0  0.0  0.0
H  1.8  0.0  0.0
...
K_POINTS gamma
```

### 4. ONETEP/线性缩放DFT

```lammps
# ONETEP与LAMMPS耦合
# 适用于大体系QM/MM

# 在LAMMPS中使用fix external调用ONETEP
fix 1 qm_atoms external pf/callback 1
```

---

## 机器学习势多尺度

### 1. DeepMD-kit工作流

```bash
# DeepMD-kit完整工作流

# 1. DFT数据生成 (CP2K/VASP/Quantum ESPRESSO)
# 生成大量短轨迹

# 2. 数据准备
dpdata convert cp2k output/ -o deepmd_data

# 3. 训练输入: input.json
{
    "model": {
        "type_map": ["H", "O"],
        "descriptor": {
            "type": "se_e2_a",
            "rcut": 6.0,
            "rcut_smth": 0.5,
            "sel": [46, 92],
            "neuron": [25, 50, 100],
            "resnet_dt": false
        },
        "fitting_net": {
            "neuron": [240, 240, 240],
            "resnet_dt": true
        }
    },
    "learning_rate": {
        "type": "exp",
        "start_lr": 0.001,
        "stop_lr": 3.51e-8
    },
    "loss": {
        "type": "ener",
        "start_pref_e": 0.02,
        "limit_pref_e": 1,
        "start_pref_f": 1000,
        "limit_pref_f": 1
    },
    "training": {
        "numb_steps": 1000000,
        "batch_size": "auto",
        "disp_file": "lcurve.out",
        "save_freq": 10000
    }
}

# 4. 训练
dp train input.json

# 5. 冻结模型
dp freeze

# 6. LAMMPS中使用
# 编译: make yes-deepmd
gcc -std=c++11 -shared -fPIC -o deepmd_pair.so deepmd_pair.cpp \
    -I/path/to/deepmd/include \
    -L/path/to/deepmd/lib -ldeepmd_cc
```

```lammps
# LAMMPS中使用DeepMD
units metal
atom_style atomic

read_data water.data

# DeepMD势
pair_style deepmd graph.pb
pair_coeff * * 

timestep 0.0005  # 0.5 fs

fix 1 all nvt temp 330.0 330.0 0.5

dump 1 all custom 100 water.dump id type x y z

run 1000000
```

### 2. TorchANI集成

```python
# torchani_lammps.py

from lammps import lammps
import torch
import torchani

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载ANI模型
model = torchani.models.ANI2x(periodic_table_index=True).to(device)

def ani_callback(caller, timestep, nlocal, ids, pos, fexternal):
    """LAMMPS回调函数"""
    # 转换位置
    positions = torch.tensor(pos, dtype=torch.float32, device=device).reshape(-1, 3)
    
    # 假设原子类型已知
    species = torch.tensor([1, 6, 7, 8], device=device)  # H, C, N, O
    
    # 计算能量和力
    positions.requires_grad_(True)
    energy = model((species, positions.unsqueeze(0))).energies
    forces = -torch.autograd.grad(energy, positions)[0]
    
    # 返回力
    fexternal[:] = forces.cpu().numpy().flatten()
    
    return energy.item()

# 初始化LAMMPS
lmp = lammps()
lmp.set_callback("external", ani_callback)
lmp.file("input.lammps")
```

### 3. MACE势

```python
# mace_lammps.py
# MACE: 高精度等变消息传递网络

from mace.calculators import mace_mp
from ase.io import read
from ase.md.langevin import Langevin
from ase import units

# 加载预训练MACE模型
calc = mace_mp(model="medium", device='cuda')

# 读取结构
atoms = read('structure.xyz')
atoms.calc = calc

# MD运行
dyn = Langevin(atoms, 1.0*units.fs, 300*units.kB, 0.002)

# 输出LAMMPS格式
for i in range(10000):
    dyn.step()
    if i % 100 == 0:
        atoms.write(f'traj_{i:06d}.lammpstrj', format='lammpsdump')
```

### 4. 主动学习与迭代训练

```python
# active_learning.py

import numpy as np
from deepmd.infer import DeepPot

class ActiveLearner:
    def __init__(self, model_path, threshold=0.5):
        self.dp = DeepPot(model_path)
        self.threshold = threshold
        self.uncertain_configs = []
    
    def check_uncertainty(self, positions, cell, atom_types):
        """检查模型不确定性"""
        # 使用模型集成或预测方差
        e, f, v = self.dp.eval(positions, cell, atom_types)
        
        # 计算力的方差 (如果是集成模型)
        force_variance = np.var(f, axis=0)
        max_uncertainty = np.max(force_variance)
        
        if max_uncertainty > self.threshold:
            self.uncertain_configs.append({
                'positions': positions,
                'cell': cell,
                'types': atom_types
            })
            return True
        return False
    
    def relabel_with_dft(self):
        """用DFT重新标记不确定构型"""
        for config in self.uncertain_configs:
            # 调用DFT计算
            energy, forces = self.run_dft(config)
            # 添加到训练集
            self.add_to_training_set(config, energy, forces)
    
    def run_dft(self, config):
        # 调用CP2K/VASP
        pass

# 在MD中使用
learner = ActiveLearner('graph.pb', threshold=0.5)

# 每1000步检查不确定性
for step in range(100000):
    lmp.step()
    if step % 1000 == 0:
        positions = lmp.gather_atoms("x", 1, 3)
        is_uncertain = learner.check_uncertainty(positions, cell, types)
        if is_uncertain:
            print(f"Uncertain configuration at step {step}")
```

---

## 自适应分辨率

### 1. AdResS (自适应分辨率模拟)

```lammps
# 自适应分辨率方法
# 高分辨率区域: 全原子
# 低分辨率区域: 粗粒化

# 定义分辨率区域
region high_res block 0 20 0 20 0 20
region low_res block 20 100 0 20 0 20

group high_atoms region high_res
group low_atoms region low_res

# 高分辨率力场 (全原子)
pair_style hybrid/overlay lj/cut 10.0 lj/cut 15.0
pair_coeff 1 1 lj/cut 1 0.2381 3.405  # AA
pair_coeff 2 2 lj/cut 2 1.0 4.0        # CG

# 使用fix adress进行插值
fix 1 all adress 5.0 10.0  # 内半径 外半径
```

### 2. 多分辨率水模型

```lammps
# 自适应分辨率水
# 中心: SPC/E全原子水
# 外部: mW粗粒化水

units real
atom_style full

# 高分辨率区域
group aa_water id < 1000
# 低分辨率区域  
group cg_water id >= 1000

# 全原子水
pair_style lj/cut/coul/long 10.0
pair_coeff 1 1 0.15535 3.166  # O
kspace_style pppm 1.0e-4

# 粗粒化水 (mW模型)
pair_style sw
pair_coeff * * mW.sw mW

# 使用hybrid/overlay
pair_style hybrid/overlay lj/cut/coul/long 10.0 sw
pair_coeff 1 1 lj/cut/coul/long 0.15535 3.166
pair_coeff 2 2 sw mW.sw mW

# 插值函数
# 需要在CG和AA之间定义平滑过渡
```

### 3. H-AdResS (哈密顿量自适应)

```lammps
# H-AdResS实现
# 基于能量的自适应分辨率

# 定义权重函数
variable w equal "1.0 - step(r-15.0)"

# 应用加权力
fix 1 all addforce v_fx v_fy v_fz energy v_w
```

---

## 耦合接口与工具

### 1. MDI (MolSSI Driver Interface)

```lammps
# MDI标准接口
# 支持代码间通信

# LAMMPS作为MD引擎
units real
atom_style atomic

read_data system.data

pair_style lj/cut 10.0
pair_coeff * * 0.2381 3.405

# 启用MDI服务器模式
mdi_engine

# 或使用fix mdi
fix 1 all mdi/engine

run 10000
```

```python
# mdi_client.py
# 其他代码作为MDI客户端

import mdi

mdi.MDI_Init("-role DRIVER -method TCP -hostname localhost -port 8021")

# 连接到LAMMPS
mdi.MDI_Accept_Communicator()

# 发送命令
mdi.MDI_Send_command("@INIT_MD")
mdi.MDI_Send_command("@COORDS")

# 接收坐标
coords = mdi.MDI_Recv(3*natoms, mdi.MDI_DOUBLE)

# 计算QM力并发送回
forces = compute_qm_forces(coords)
mdi.MDI_Send_command("@FORCES")
mdi.MDI_Send(forces, 3*natoms, mdi.MDI_DOUBLE)
```

### 2. I-Pi (路径积分分子动力学)

```bash
# I-Pi与LAMMPS耦合
# 用于量子核效应

# i-pi输入: input.xml
<simulation verbosity='medium'>
  <output prefix='simulation'>
    <properties stride='10' filename='out'>
      [ step, time{picosecond}, conserved, temperature{kelvin}, 
        kinetic_md, potential, pressure_md ]
    </properties>
    <trajectory filename='pos' stride='100' format='xyz' cell_units='angstrom'>
      positions{angstrom}
    </trajectory>
  </output>
  
  <total_steps>100000</total_steps>
  <prng>
    <seed>32345</seed>
  </prng>
  
  <system>
    <initialize nbeads='4'>
      <file mode='xyz'>init.xyz</file>
    </initialize>
    <forces>
      <force forcefield='lammps' name='lammps'>
        <lammps>
          <command>lmp</command>
          <options>
            <input_file>in.lammps</input_file>
          </options>
        </lammps>
      </force>
    </forces>
    <ensemble>
      <temperature units='kelvin'>300</temperature>
    </ensemble>
    <motion mode='dynamics'>
      <dynamics mode='nvt'>
        <timestep units='femtosecond'>0.5</timestep>
        <thermostat mode='pile_l'>
          <tau units='femtosecond'>100</tau>
        </thermostat>
      </dynamics>
    </motion>
  </system>
</simulation>
```

### 3. 套接字通讯

```python
# socket_server.py
# LAMMPS作为socket服务器

import socket
import numpy as np

class LAMMPSSocketServer:
    def __init__(self, host='localhost', port=8080):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((host, port))
        self.socket.listen(1)
    
    def run(self):
        conn, addr = self.socket.accept()
        print(f"Connected by {addr}")
        
        while True:
            # 接收命令
            cmd = conn.recv(1024).decode().strip()
            
            if cmd == "GET_POS":
                # 发送原子位置
                positions = self.get_positions_from_lammps()
                conn.sendall(positions.tobytes())
                
            elif cmd == "SET_FORCES":
                # 接收力
                forces = np.frombuffer(conn.recv(1024), dtype=np.float64)
                self.set_forces_to_lammps(forces)
                
            elif cmd == "STEP":
                # 运行一步
                self.lammps_step()
                conn.sendall(b"DONE")
                
            elif cmd == "EXIT":
                break
        
        conn.close()

# 启动服务器
server = LAMMPSSocketServer()
server.run()
```

### 4. 多代码工作流

```bash
#!/bin/bash
# multiscale_workflow.sh

# 阶段1: DFT训练数据生成
for config in configs/*; do
    cp2k.popt -i ${config%.xyz}.inp -o dft_${config%.xyz}.out
done

# 阶段2: 训练ML势
dp train input.json
dp freeze

# 阶段3: LAMMPS大规模模拟
mpirun -np 64 lmp -in large_scale.in -v potential graph.pb

# 阶段4: 感兴趣区域QM/MM细化
python qmmm_refinement.py trajectory.dump

# 阶段5: 分析
python analyze_results.py
```
