# ML势 (机器学习势能面) 训练完整流程

## 1. ML势概述

### 1.1 为什么需要ML势？

| DFT局限性 | ML势解决方案 |
|-----------|--------------|
| 计算成本高 (~1000原子极限) | 线性标度，百万原子可行 |
| 时间尺度短 (~100ps) | 可达微秒-毫秒 |
| 特定泛函近似 | 可拟合任意精度参考数据 |
| 难以处理稀有事件 | 结合增强采样 |

### 1.2 主流ML势框架

| 框架 | 类型 | 特点 | 适用场景 |
|------|------|------|----------|
| **DP-GEN/DeepMD** | 深度神经网络 | 端到端、并行训练 | 大规模MD |
| **SNAP** | 谱距离 | 物理可解释 | 元素探索 |
| **GAP** | 高斯过程 | 不确定性量化 | 主动学习 |
| **MTP** | 矩张量势 | 快速评估 | 大体系 |
| **ACE** | 原子簇展开 | 高阶相互作用 | 复杂体系 |
| **CHGNet/M3GNet** | 图神经网络 | 预训练模型 | 快速部署 |
| **MACE** | 等变消息传递 | SO(3)等变 | 高精度 |

### 1.3 训练数据要求

**数据量**:
- 简单体系: 100-1000个构型
- 复杂反应: 10,000-100,000个构型
- 覆盖: 势能面各区域充分采样

**数据质量**:
- DFT级别: PBE/PBEsol (快速), r²SCAN (平衡), 杂化泛函 (高精度)
- 能量收敛: < 1 meV/atom
- 力收敛: 严格
- k点收敛: 对周期性体系至关重要

---

## 2. 数据准备与DFT计算

### 2.1 初始结构生成

```bash
mkdir mlff_workflow && cd mlff_workflow
mkdir 0_initial_structures
```

**结构来源**:
```bash
#!/bin/bash
# generate_initial_configs.sh

# 1. 实验晶体结构
# Materials Project API获取
curl -X GET "https://api.materialsproject.org/materials/core/?formula=LiFePO4&fields=structure" \
     -H "X-API-KEY: your_api_key" > lifepo4.json

# 2. 从头算分子动力学 (AIMD) 轨迹
# 使用不同温度、密度
for temp in 300 600 900 1200; do
    mkdir aimd_$temp
    cd aimd_$temp
    # 运行AIMD
    cp ../INCAR_AIMD INCAR
    sed -i "s/TEBEG.*/TEBEG = $temp/" INCAR
    mpirun -np 32 vasp_std
    cd ..
done

# 3. 晶格畸变 (rattle)
python <> 'EOF'
from ase.io import read, write
from ase.build import bulk
import numpy as np

# 读取平衡结构
atoms = bulk('Si', cubic=True)
atoms *= (2, 2, 2)  # 超胞

# 生成畸变结构
for i in range(100):
    rattle_atoms = atoms.copy()
    rattle_atoms.rattle(stdev=0.05, seed=i)  # 0.05 Å随机位移
    write(f'0_initial_structures/si_rattle_{i:03d}.vasp', rattle_atoms)
EOF

# 4. 表面/缺陷结构
# 使用ASE/Pymatgen构建
```

### 2.2 体积扫描 (EOS曲线)

```python
#!/usr/bin/env python3
"""生成EOS扫描结构"""
from ase.io import read, write
from ase.build import bulk
import numpy as np

atoms = bulk('Si', cubic=True)
atoms *= (2, 2, 2)

# 体积变形 ±15%
volumes = np.linspace(0.85, 1.15, 20)

for i, v in enumerate(volumes):
    scaled = atoms.copy()
    scaled.set_cell(atoms.get_cell() * v**(1/3), scale_atoms=True)
    write(f'eos_vol_{i:02d}.vasp', scaled)
```

### 2.3 DFT单点能计算

```bash
# 创建DFT计算目录
mkdir 1_dft_calculations
cd 1_dft_calculations

# 准备VASP输入
for struct in ../0_initial_structures/*.vasp; do
    name=$(basename $struct .vasp)
    mkdir $name
    cd $name
    
    cat > INCAR << 'EOF'
SYSTEM = MLFF Training Data

# 高精度设置
ENCUT = 600
EDIFF = 1E-8
ISMEAR = 0
SIGMA = 0.05

# PBE泛函 (可换为r2SCAN, HSE等)
GGA = PE

# 力计算
LCHARG = .FALSE.
LWAVE = .FALSE.

# 关键: 输出力和应力
# VASP默认输出力到OUTCAR
EOF
    
    cp $struct POSCAR
    # KPOINTS根据体系自动调整
    cat > KPOINTS <> 'EOF'
Automatic mesh
0
Gamma
4 4 4
0 0 0
EOF
    
    # 提交作业
    sbatch vasp_job.sh
    
    cd ..
done
```

### 2.4 结果提取与数据库构建

```python
#!/usr/bin/env python3
"""
extract_dft_data.py - 提取DFT结果并构建训练数据库
"""
import os
import numpy as np
from ase.io import read, write
from ase.io.vasp import read_vasp_out
import json

def extract_vasp_data(outcar_path):
    """从VASP OUTCAR提取能量、力、应力"""
    atoms = read(outcar_path, format='vasp-out')
    
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress(voigt=False)  # 3x3张量
    
    return {
        'energy': energy,
        'forces': forces.tolist(),
        'stress': stress.tolist(),
        'positions': atoms.positions.tolist(),
        'cell': atoms.cell.tolist(),
        'symbols': atoms.get_chemical_symbols(),
        'pbc': atoms.pbc.tolist()
    }

# 遍历所有计算结果
database = []
base_dir = '1_dft_calculations'

for subdir in os.listdir(base_dir):
    outcar_path = os.path.join(base_dir, subdir, 'OUTCAR')
    if os.path.exists(outcar_path):
        try:
            data = extract_vasp_data(outcar_path)
            data['id'] = subdir
            database.append(data)
            print(f"Extracted: {subdir}")
        except Exception as e:
            print(f"Failed to extract {subdir}: {e}")

# 保存为JSON (通用格式)
with open('training_data.json', 'w') as f:
    json.dump(database, f, indent=2)

# 转换为ASE db格式 (更高效)
from ase.db import connect

db = connect('training.db', append=False)
for data in database:
    atoms = Atoms(symbols=data['symbols'],
                  positions=data['positions'],
                  cell=data['cell'],
                  pbc=data['pbc'])
    db.write(atoms, 
             energy=data['energy'],
             forces=data['forces'],
             stress=data['stress'])

print(f"Database created: {len(database)} configurations")
```

---

## 3. DeepMD-kit 训练流程

### 3.1 环境安装

```bash
# 安装DeepMD-kit (conda推荐)
conda create -n deepmd deepmd-kit=2.2.* lammps cudatoolkit=11.6 -c conda-forge
conda activate deepmd

# 验证安装
dp --version
```

### 3.2 数据格式转换

```python
#!/usr/bin/env python3
"""
convert_to_deepmd.py - 转换为DeepMD格式
"""
import numpy as np
import json
from ase.db import connect
from collections import defaultdict

def convert_to_deepmd(db_path, output_dir):
    """将ASE数据库转换为DeepMD的raw格式"""
    db = connect(db_path)
    
    # 按元素类型分组
    type_map = defaultdict(list)
    
    for row in db.select():
        atoms = row.toatoms()
        symbols = atoms.get_chemical_symbols()
        unique_symbols = sorted(set(symbols))
        type_key = '-'.join(unique_symbols)
        type_map[type_key].append(row)
    
    # 为每组创建DeepMD数据
    for type_key, rows in type_map.items():
        type_dir = f"{output_dir}/{type_key}"
        os.makedirs(type_dir, exist_ok=True)
        
        # DeepMD格式: type.raw, type_map.raw, set.000/{coord,force,energy,box}
        all_coords = []
        all_forces = []
        all_energies = []
        all_boxes = []
        all_types = []
        
        symbol_to_type = {s: i for i, s in enumerate(unique_symbols)}
        
        for row in rows:
            atoms = row.toatoms()
            all_coords.append(atoms.positions.flatten())
            all_forces.append(np.array(row.forces).flatten())
            all_energies.append(row.energy)
            all_boxes.append(atoms.cell.array.flatten())
            all_types.append([symbol_to_type[s] for s in atoms.get_chemical_symbols()])
        
        # 写入文件
        np.savetxt(f'{type_dir}/type.raw', all_types[0], fmt='%d')
        with open(f'{type_dir}/type_map.raw', 'w') as f:
            f.write(' '.join(unique_symbols))
        
        set_dir = f'{type_dir}/set.000'
        os.makedirs(set_dir, exist_ok=True)
        np.save(f'{set_dir}/coord', np.array(all_coords))
        np.save(f'{set_dir}/force', np.array(all_forces))
        np.save(f'{set_dir}/energy', np.array(all_energies))
        np.save(f'{set_dir}/box', np.array(all_boxes))

convert_to_deepmd('training.db', 'deepmd_data')
```

### 3.3 训练脚本配置

**input.json** (DeepMD-kit配置):
```json
{
  "model": {
    "type_map": ["H", "C", "O", "Li", "Fe", "P"],
    "descriptor": {
      "type": "se_e2_a",
      "rcut": 6.0,
      "rcut_smth": 0.5,
      "sel": [200, 200, 200, 200, 200, 200],
      "neuron": [25, 50, 100],
      "resnet_dt": false,
      "axis_neuron": 16,
      "seed": 1,
      "activation_function": "tanh"
    },
    "fitting_net": {
      "neuron": [240, 240, 240],
      "resnet_dt": true,
      "seed": 1,
      "activation_function": "tanh"
    }
  },
  "learning_rate": {
    "type": "exp",
    "start_lr": 0.001,
    "stop_lr": 3.51e-8,
    "decay_steps": 5000
  },
  "loss": {
    "type": "ener",
    "start_pref_e": 0.02,
    "limit_pref_e": 1,
    "start_pref_f": 1000,
    "limit_pref_f": 1,
    "start_pref_v": 0.0,
    "limit_pref_v": 0.0
  },
  "training": {
    "training_data": {
      "systems": ["deepmd_data/Li-Fe-P-O"],
      "batch_size": "auto"
    },
    "validation_data": {
      "systems": ["deepmd_data/Li-Fe-P-O/validation"],
      "batch_size": "auto",
      "numb_btch": 1
    },
    "numb_steps": 1000000,
    "seed": 10,
    "disp_file": "lcurve.out",
    "save_freq": 10000,
    "max_ckpt_keep": 5
  }
}
```

### 3.4 启动训练

```bash
#!/bin/bash
# train_deepmd.sh

# 加载环境
conda activate deepmd

# 数据准备
dp convert-from deepmd_data -o training_data

# 训练
mpirun -np 4 dp train input.json

# 冻结模型 (导出为.pb文件)
dp freeze -o graph.pb

# 压缩模型 (减小体积，加快推理)
dp compress -i graph.pb -o graph_compressed.pb
```

### 3.5 训练监控与诊断

```python
#!/usr/bin/env python3
"""
monitor_training.py - 监控训练进度
"""
import numpy as np
import matplotlib.pyplot as plt

# 读取训练日志
data = np.loadtxt('lcurve.out', skiprows=1)

step = data[:, 0]
lr = data[:, 1]
energy_rmse_tr = data[:, 2]  # 训练集能量RMSE
energy_rmse_val = data[:, 3]  # 验证集能量RMSE
force_rmse_tr = data[:, 4]
force_rmse_val = data[:, 5]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 学习率
axes[0, 0].semilogy(step, lr)
axes[0, 0].set_xlabel('Step')
axes[0, 0].set_ylabel('Learning Rate')
axes[0, 0].set_title('Learning Rate Schedule')

# 能量RMSE
axes[0, 1].semilogy(step, energy_rmse_tr, label='Train')
axes[0, 1].semilogy(step, energy_rmse_val, label='Validation')
axes[0, 1].set_xlabel('Step')
axes[0, 1].set_ylabel('Energy RMSE (eV)')
axes[0, 1].set_title('Energy RMSE')
axes[0, 1].legend()

# 力RMSE
axes[1, 0].semilogy(step, force_rmse_tr, label='Train')
axes[1, 0].semilogy(step, force_rmse_val, label='Validation')
axes[1, 0].set_xlabel('Step')
axes[1, 0].set_ylabel('Force RMSE (eV/Å)')
axes[1, 0].set_title('Force RMSE')
axes[1, 0].legend()

# 验证误差 vs 训练误差 (检查过拟合)
axes[1, 1].loglog(energy_rmse_tr, energy_rmse_val, '.', alpha=0.5)
axes[1, 1].plot([1e-3, 10], [1e-3, 10], 'k--')
axes[1, 1].set_xlabel('Train RMSE')
axes[1, 1].set_ylabel('Validation RMSE')
axes[1, 1].set_title('Generalization Check')

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)

# 打印最终性能
print(f"Final Validation Energy RMSE: {energy_rmse_val[-1]:.6f} eV")
print(f"Final Validation Force RMSE: {force_rmse_val[-1]:.6f} eV/Å")

# 检查是否收敛
if energy_rmse_val[-1] < 0.01 and force_rmse_val[-1] < 0.1:
    print("✓ Training converged to good accuracy")
else:
    print("✗ Training may need more steps or data")
```

---

## 4. DP-GEN 主动学习工作流

### 4.1 DP-GEN简介

DP-GEN实现自动化迭代训练流程:
```
迭代循环:
1. 探索: ML势MD生成候选构型
2. 标注: DFT计算候选构型
3. 训练: 扩展训练集重新训练
4. 验证: 检查模型质量
```

### 4.2 配置文件

**param.json** (DP-GEN主配置):
```json
{
  "type_map": ["Li", "Fe", "P", "O"],
  "mass_map": [6.94, 55.85, 30.97, 16.0],
  "init_data_prefix": "./",
  "init_data_sys": ["deepmd_data/Li-Fe-P-O"],
  "sys_format": "deepmd/npy",
  
  "numb_models": 4,
  "default_training_param": {
    "model": {
      "descriptor": {
        "type": "se_e2_a",
        "rcut": 6.0,
        "rcut_smth": 0.5,
        "sel": [200, 200, 200, 200],
        "neuron": [25, 50, 100],
        "resnet_dt": false,
        "axis_neuron": 16,
        "seed": 1
      },
      "fitting_net": {
        "neuron": [240, 240, 240],
        "resnet_dt": true,
        "seed": 1
      }
    },
    "learning_rate": {
      "type": "exp",
      "start_lr": 0.001,
      "decay_steps": 5000
    },
    "loss": {
      "start_pref_e": 0.02,
      "limit_pref_e": 1,
      "start_pref_f": 1000,
      "limit_pref_f": 1
    },
    "training": {
      "numb_steps": 1000000,
      "seed": 10,
      "disp_file": "lcurve.out",
      "save_freq": 10000
    }
  },
  
  "model_devi_engine": "lammps",
  "model_devi_jobs": [
    {
      "sys_idx": [0],
      "temps": [50, 300, 600, 900],
      "press": [1.0, 1000.0, 5000.0, 10000.0],
      "trj_freq": 10,
      "nsteps": 10000,
      "ensemble": "npt",
      "neidelay": 1,
      "taut": 0.1,
      "taup": 0.5
    }
  ],
  "model_devi_f_trust_lo": 0.05,
  "model_devi_f_trust_hi": 0.15,
  
  "fp_style": "vasp",
  "shuffle_poscar": false,
  "fp_task_max": 200,
  "fp_task_min": 10,
  "fp_pp_path": "./",
  "fp_pp_files": ["Li.pbe-spn-kjpaw_psl.1.0.0.UPF", 
                  "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF",
                  "P.pbe-n-kjpaw_psl.1.0.0.UPF",
                  "O.pbe-n-kjpaw_psl.1.0.0.UPF"],
  "fp_incar": "./INCAR_fp"
}
```

**machine.json** (计算资源配置):
```json
{
  "train": [
    {
      "machine": {
        "machine_type": "slurm",
        "hostname": "hpc.university.edu",
        "port": 22,
        "username": "user",
        "work_path": "/scratch/user/dpgen/train"
      },
      "resources": {
        "numb_node": 1,
        "numb_gpu": 4,
        "task_per_node": 4,
        "partition": "gpu",
        "source_list": ["/opt/deepmd/env.sh"]
      },
      "python_path": "/opt/deepmd/bin/python"
    }
  ],
  "model_devi": [
    {
      "machine": {
        "machine_type": "slurm",
        "work_path": "/scratch/user/dpgen/md"
      },
      "resources": {
        "numb_node": 1,
        "numb_gpu": 1,
        "task_per_node": 8,
        "partition": "gpu"
      },
      "command": "lmp",
      "group_size": 50
    }
  ],
  "fp": [
    {
      "machine": {
        "machine_type": "slurm", 
        "work_path": "/scratch/user/dpgen/fp"
      },
      "resources": {
        "numb_node": 4,
        "task_per_node": 32,
        "cpu_per_node": 32,
        "partition": "cpu",
        "module_list": ["vasp/6.4.0"]
      },
      "command": "mpirun -np 128 vasp_std",
      "group_size": 20
    }
  ]
}
```

### 4.3 运行DP-GEN

```bash
#!/bin/bash
# run_dpgen.sh

# 提交DP-GEN工作流
dpgen run param.json machine.json

# 监控进度
watch -n 60 'tail -50 dpgen.log'
```

### 4.4 模型偏差分析

```python
#!/usr/bin/env python3
"""
analyze_model_devi.py - 分析模型偏差
"""
import numpy as np
import matplotlib.pyplot as plt
import glob

# 读取所有模型偏差文件
devi_files = glob.glob('model_devi/job*/model_devi.out')

all_devi = []
for f in devi_files:
    data = np.loadtxt(f)
    all_devi.append(data)

all_devi = np.vstack(all_devi)

# 分析
deavi_max = all_devi[:, 4]  # max deviation
step = all_devi[:, 0]

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(step, deavi_max, alpha=0.3, s=5)
ax.axhline(y=0.15, color='r', linestyle='--', label='Trust Hi')
ax.axhline(y=0.05, color='g', linestyle='--', label='Trust Lo')
ax.set_xlabel('MD Step')
ax.set_ylabel('Max Force Deviation (eV/Å)')
ax.set_title('Model Deviation During Exploration')
ax.legend()
ax.set_yscale('log')
plt.savefig('model_deviation.png', dpi=150)

# 统计候选构型数
candidates_hi = np.sum(deavi_max > 0.15)
candidates_lo = np.sum(deavi_max < 0.05)
candidates_mid = len(deavi_max) - candidates_hi - candidates_lo

print(f"Configurations selected for DFT: {candidates_hi}")
print(f"Good configurations (reliable): {candidates_lo}")
print(f"Gray zone: {candidates_mid}")
```

---

## 5. 预训练模型 (M3GNet/CHGNet)

### 5.1 快速部署预训练模型

```python
#!/usr/bin/env python3
"""
pretrained_model_demo.py - 使用预训练模型
"""
from ase import Atoms
from ase.io import read, write
from ase.md import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

# 使用M3GNet (Matten)
try:
    import matgl
    from matgl.ext.ase import PESCalculator
    
    # 加载预训练模型
    pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
    calc = PESCalculator(potential=pot)
    
    print("M3GNet loaded successfully")
    
except ImportError:
    print("MatGL not installed, trying CHGNet...")
    
    from chgnet.model import CHGNet
    from chgnet.model.dynamics import CHGNetCalculator
    
    chgnet = CHGNet.load()
    calc = CHGNetCalculator(model=chgnet)
    
    print("CHGNet loaded successfully")

# 设置计算
atoms = read('structure.vasp')
atoms.calc = calc

# 单点能计算
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()

print(f"Energy: {energy:.4f} eV")
print(f"Max force: {np.max(np.abs(forces)):.4f} eV/Å")

# MD模拟
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
dyn = Langevin(atoms, timestep=1*units.fs, temperature_K=300, friction=0.01)

def print_status(a=atoms):
    print(f"Step: {dyn.get_number_of_steps()}, E: {a.get_potential_energy():.4f} eV")

dyn.attach(print_status, interval=100)
dyn.run(10000)  # 10 ps
```

### 5.2 微调预训练模型

```python
#!/usr/bin/env python3
"""
finetune_chgnet.py - 微调CHGNet
"""
from chgnet.model import CHGNet
from chgnet.trainer import Trainer
from chgnet.data.dataset import StructureData, get_train_val_test_loader
import torch

# 加载预训练模型
model = CHGNet.load()

# 准备数据
data = StructureData.from_file('my_training_data.json')
train_loader, val_loader, test_loader = get_train_val_test_loader(
    data, batch_size=32, train_ratio=0.8, val_ratio=0.1
)

# 训练器
trainer = Trainer(
    model=model,
    targets="efs",  # energy, force, stress
    optimizer="Adam",
    scheduler="CosineAnnealingLR",
    criterion="MSE",
    epochs=100,
    learning_rate=1e-3,
    device="cuda"
)

# 微调
trainer.train(train_loader, val_loader, test_loader)

# 保存模型
torch.save(model.state_dict(), 'chgnet_finetuned.pth')
```

---

## 6. LAMMPS 中使用ML势

### 6.1 DeepMD 与 LAMMPS

```bash
# 编译LAMMPS with DeepMD
# 或直接用conda安装
conda install lammps deepmd-kit -c conda-forge
```

**lammps_input.in**:
```
# LAMMPS input with DeepMD potential

units           metal
atom_style      atomic
boundary        p p p

# 读取结构
read_data       structure.lmp

# DeepMD势 (使用冻结的graph.pb)
pair_style      deepmd graph.pb
pair_coeff      * *

# 邻居列表
neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

# 输出
thermo          100
thermo_style    custom step temp pe ke etotal press

# 模拟
timestep        0.001  # 1 fs

# NVT MD
fix             1 all nvt temp 300 300 0.1
run             100000  # 100 ps

# 或 NPT
unfix           1
fix             2 all npt temp 300 300 0.1 iso 1.0 1.0 1.0
dump            1 all atom 100 trajectory.dump
run             100000
```

```bash
# 运行
mpirun -np 4 lmp -in lammps_input.in
```

### 6.2 ACE 势在 LAMMPS 中

```
# ACE potential (MTP alternative)
pair_style      pace
pair_coeff      * * Si_Yace_Si.json Si
```

---

## 7. 验证与测试

### 7.1 性能基准测试

```python
#!/usr/bin/env python3
"""
benchmark_mlff.py - ML势验证
"""
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.eos import EquationOfState
import time

def benchmark_properties(atoms, calc):
    """测试关键物理性质"""
    results = {}
    
    atoms.calc = calc
    
    # 1. 晶格常数
    from ase.optimize import BFGS
    opt = BFGS(atoms)
    opt.run(fmax=0.01)
    results['lattice_constant'] = atoms.cell.lengths()
    
    # 2. 体模量 (EOS)
    volumes = []
    energies = []
    for scale in np.linspace(0.9, 1.1, 10):
        a = atoms.copy()
        a.set_cell(atoms.cell * scale**(1/3), scale_atoms=True)
        volumes.append(a.get_volume())
        energies.append(a.get_potential_energy())
    
    eos = EquationOfState(volumes, energies)
    v0, e0, B = eos.fit()
    results['bulk_modulus'] = B / 1e9  # GPa
    
    # 3. 声子谱 (需要phonopy)
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    
    phonopy_atoms = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell,
        scaled_positions=atoms.get_scaled_positions()
    )
    
    phonon = Phonopy(phonopy_atoms, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    phonon.generate_displacements(distance=0.01)
    
    forces = []
    for disp in phonon.supercells_with_displacements:
        a = Atoms(symbols=disp.symbols,
                  cell=disp.cell,
                  scaled_positions=disp.scaled_positions,
                  pbc=True)
        a.calc = calc
        forces.append(a.get_forces())
    
    phonon.forces = forces
    phonon.produce_force_constants()
    phonon.auto_band_structure()
    
    # 4. 计算速度
    atoms_r = atoms * (3, 3, 3)  # 27倍原子数
    start = time.time()
    _ = atoms_r.get_potential_energy()
    elapsed = time.time() - start
    results['time_per_atom'] = elapsed / len(atoms_r)
    
    return results

# 运行测试
atoms = read('Si_bulk.vasp')
results = benchmark_properties(atoms, calc)

print("MLFF Performance:")
print(f"  Lattice constant: {results['lattice_constant']}")
print(f"  Bulk modulus: {results['bulk_modulus']:.2f} GPa")
print(f"  Time per atom: {results['time_per_atom']*1000:.3f} ms")
```

### 7.2 与DFT对比

```python
#!/usr/bin/env python3
"""
compare_dft_mlff.py - 对比DFT和MLFF
"""
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from ase.calculators.vasp import Vasp

# 测试构型
test_structs = read('test_structures.xyz', ':')

dft_energies = []
mlff_energies = []
dft_forces = []
mlff_forces = []

for atoms in test_structs:
    # DFT计算
    atoms.calc = Vasp(xc='PBE', encut=400, ismear=0, sigma=0.05)
    dft_e = atoms.get_potential_energy()
    dft_f = atoms.get_forces()
    
    # MLFF计算
    atoms.calc = mlff_calc  # 你的ML势
    mlff_e = atoms.get_potential_energy()
    mlff_f = atoms.get_forces()
    
    dft_energies.append(dft_e)
    mlff_energies.append(mlff_e)
    dft_forces.extend(dft_f.flatten())
    mlff_forces.extend(mlff_f.flatten())

# 绘制对比
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 能量
axes[0].scatter(dft_energies, mlff_energies, alpha=0.5)
min_e, max_e = min(dft_energies), max(dft_energies)
axes[0].plot([min_e, max_e], [min_e, max_e], 'k--')
axes[0].set_xlabel('DFT Energy (eV)')
axes[0].set_ylabel('MLFF Energy (eV)')
axes[0].set_title('Energy Correlation')

rmse_e = np.sqrt(np.mean((np.array(dft_energies) - np.array(mlff_energies))**2))
axes[0].text(0.05, 0.95, f'RMSE: {rmse_e:.4f} eV', 
             transform=axes[0].transAxes, verticalalignment='top')

# 力
axes[1].scatter(dft_forces, mlff_forces, alpha=0.3, s=1)
min_f, max_f = min(dft_forces), max(dft_forces)
axes[1].plot([min_f, max_f], [min_f, max_f], 'k--')
axes[1].set_xlabel('DFT Force (eV/Å)')
axes[1].set_ylabel('MLFF Force (eV/Å)')
axes[1].set_title('Force Correlation')

rmse_f = np.sqrt(np.mean((np.array(dft_forces) - np.array(mlff_forces))**2))
axes[1].text(0.05, 0.95, f'RMSE: {rmse_f:.4f} eV/Å', 
             transform=axes[1].transAxes, verticalalignment='top')

plt.tight_layout()
plt.savefig('dft_mlff_comparison.png', dpi=150)
```

---

## 8. 高级主题

### 8.1 迁移学习

```python
#!/usr/bin/env python3
"""
transfer_learning.py - 跨体系迁移学习
"""
from deepmd.train import DeepPot

# 1. 加载已有模型
pretrained = DeepPot('graph_pretrained.pb')

# 2. 冻结部分层 (特征提取层)
# 只训练fitting_net

# 3. 在新体系上继续训练
train_on_new_system(pretrained, new_data)
```

### 8.2 多保真度训练

```python
#!/usr/bin/env python3
"""
multi_fidelity.py - 多保真度训练
使用大量低精度数据 + 少量高精度数据
"""
# 数据配置
low_fidelity_data = "pbe_data/"  # 大量数据
high_fidelity_data = "hse_data/"  # 少量数据

# 分阶段训练
# 阶段1: PBE数据预训练
train(input_json, low_fidelity_data, steps=500000)

# 阶段2: HSE数据微调
train(input_json, high_fidelity_data, steps=100000, 
      restart_from='stage1_model')
```

---

## 9. 故障排查

### 9.1 训练发散

**症状**: loss变为NaN或爆炸性增长

**对策**:
```json
{
  "learning_rate": {
    "start_lr": 0.0001,  // 降低学习率
    "stop_lr": 1e-8
  },
  "loss": {
    "start_pref_e": 0.001,  // 降低初始权重
    "start_pref_f": 100
  }
}
```

### 9.2 力预测不准确

**可能原因**:
- 训练数据力收敛不佳
- 截断半径太小
- 描述符维度不足

**检查**:
```python
# 检查力分布
dft_forces = [np.linalg.norm(f) for f in dft_forces]
plt.hist(dft_forces, bins=50)
plt.xlabel('|F| (eV/Å)')
# 确保覆盖0-10 eV/Å范围
```

### 9.3 MD不稳定

```python
# 能量漂移检查
def check_energy_conservation(traj):
    """检查NVE模拟能量守恒"""
    energies = [atoms.get_potential_energy() for atoms in traj]
    drift = max(energies) - min(energies)
    print(f"Energy drift: {drift:.4f} eV")
    if drift > 0.01:  # 10 meV阈值
        print("WARNING: Large energy drift, potential may be problematic")
```

---

## 10. 完整工作流总结

```bash
mlff_workflow/
├── 0_initial_structures/       # 初始结构准备
│   ├── rattle/                 # 热膨胀采样
│   ├── aimd_300K/              # AIMD轨迹
│   └── surfaces/               # 表面结构
├── 1_dft_calculations/         # DFT单点能计算
│   ├── run_dft.sh
│   └── extract_data.py
├── 2_training_data/            # 格式化训练数据
│   ├── deepmd_data/
│   ├── training.db
│   └── training_data.json
├── 3_training/                 # ML势训练
│   ├── input.json              # DeepMD配置
│   ├── run_training.sh
│   └── monitor.py
├── 4_active_learning/          # DP-GEN主动学习 (可选)
│   ├── param.json
│   ├── machine.json
│   └── run_dpgen.sh
├── 5_validation/               # 验证测试
│   ├── benchmark.py
│   ├── eos_test/
│   └── phonon_test/
└── 6_production/               # 生产使用
    ├── lammps_input.in
    └── run_md.sh
```

---

## 11. 参考资源

### 软件文档
- DeepMD-kit: https://docs.deepmodeling.com/
- DP-GEN: https://github.com/deepmodeling/dpgen
- M3GNet: https://github.com/materialsvirtuallab/m3gnet
- CHGNet: https://github.com/CederGroupHub/chgnet
- MACE: https://github.com/ACEsuit/mace

### 关键文献
1. **Zhang et al.** (2018). Deep Potential Molecular Dynamics. *PRL* 120, 143001.
2. **Zhang et al.** (2020). DP-GEN: A concurrent learning platform. *npj Comput. Mater.* 6, 92.
3. **Chen & Ong** (2022). A universal graph deep learning interatomic potential. *Nature* 1.
4. **Batatia et al.** (2022). MACE: Higher order equivariant message passing. *arXiv:2206.07697*.

### 最佳实践
- 数据多样性 > 数据量
- 检查DFT收敛 (力收敛标准)
- 主动学习大幅减少DFT计算量
- 始终保留验证集
- 定期重新训练 (新发现构型)
