# NequIP/MACE/Allegro 配置模板参考

## 1. NequIP 基础配置 (nequip_config.yaml)

```yaml
# 数据设置
dataset: ase
dataset_file_name: ./data/train.xyz
dataset_seed: 123

# 化学元素
 chemical_symbols:
  - Si
  - O

# 模型架构
model_builders:
  - SimpleIrrepsConfig
  - EnergyModel

r_max: 5.0  # 截断半径 (Å)
num_layers: 3
l_max: 2
num_features: 32

# 等变设置
irreps_edge_sh: 0e + 1o + 2e
edge_sh_normalization: norm
edge_sh_normalize: True

# MLP设置
num_basis: 8
invariant_layers: 2
invariant_neurons: 64
avg_num_neighbors: 15.0

# 训练设置
batch_size: 5
learning_rate: 0.001
num_epochs: 10000

# 损失函数
loss_coeff_energy: 1.0
loss_coeff_force: 10.0

# 优化器
optimizer_name: Adam
lr_scheduler_name: ReduceLROnPlateau
lr_scheduler_patience: 100
lr_scheduler_factor: 0.5
```

## 2. MACE 基础配置

### 2.1 训练脚本 (train_mace.sh)

```bash
python -m mace.cli.run_train \
    --name="mace_model" \
    --train_file="data/train.xyz" \
    --valid_fraction=0.1 \
    --test_file="data/test.xyz" \
    --config_type_weights='{"Default": 1.0}' \
    --model="MACE" \
    --hidden_irreps='128x0e + 128x1o' \
    --r_max=5.0 \
    --batch_size=10 \
    --max_num_neighbors=40 \
    --num_interactions=1 \
    --correlation=3 \
    --num_channels=128 \
    --max_L=2 \
    --loss='weighted' \
    --forces_weight=100.0 \
    --energy_weight=1.0 \
    --learning_rate=0.001 \
    --scheduler_patience=5 \
    --max_num_epochs=1000 \
    --device=cuda
```

### 2.2 微调基础模型 (finetune_mace.sh)

```bash
python -m mace.cli.run_train \
    --name="mace_finetuned" \
    --train_file="data/my_data.xyz" \
    --valid_fraction=0.1 \
    --model="MACE" \
    --load_model="MACE-MP-0.model" \
    --freeze_irreps='0e' \
    --hidden_irreps='128x0e + 128x1o' \
    --r_max=6.0 \
    --batch_size=5 \
    --num_interactions=1 \
    --max_num_epochs=500 \
    --learning_rate=0.0001 \
    --device=cuda
```

## 3. Allegro 配置 (allegro_config.yaml)

```yaml
# 数据设置
dataset: ase
dataset_file_name: ./data/train.xyz

# 化学元素
chemical_symbols:
  - Al
  - Cu
  - Zr

# 模型架构
model_builders:
  - AllegroConfig
  - EnergyModel

r_max: 5.0
num_layers: 2
l_max: 2

# Allegro特有参数
env_embed_multiplicity: 16
pair_embed_multiplicity: 16
two_body_latent_mlp_latent_dimensions: [64, 64, 64]

# 等变设置
irreps_edge_sh: 0e + 1o + 2e
edge_sh_normalization: norm

# 训练设置
batch_size: 4
learning_rate: 0.001
num_epochs: 5000

# 损失函数
loss_coeff_energy: 1.0
loss_coeff_force: 10.0
```

## 4. DeePMD-kit DPLR配置

### 4.1 Wannier模型配置 (dw.json)

```json
{
  "model": {
    "type_map": ["O", "H"],
    "descriptor": {
      "type": "se_e2_a",
      "rcut": 6.0,
      "rcut_smth": 0.5,
      "sel": [46, 92],
      "neuron": [25, 50, 100],
      "resnet_dt": false,
      "axis_neuron": 16,
      "type_one_side": true
    },
    "fitting_net": {
      "type": "dipole",
      "dipole_type": [0],
      "neuron": [128, 128, 128],
      "seed": 1
    }
  },
  "learning_rate": {
    "type": "exp",
    "start_lr": 0.001,
    "stop_lr": 3.51e-8,
    "decay_steps": 5000
  },
  "loss": {
    "type": "tensor",
    "pref": 0.0,
    "pref_atomic": 1.0
  },
  "training": {
    "numb_steps": 100000,
    "batch_size": 1,
    "systems": ["./data/wannier/"]
  }
}
```

### 4.2 DPLR能量模型配置 (dplr.json)

```json
{
  "model": {
    "type_map": ["O", "H"],
    "descriptor": {
      "type": "se_e2_a",
      "rcut": 6.0,
      "sel": [46, 92],
      "neuron": [25, 50, 100]
    },
    "fitting_net": {
      "neuron": [240, 240, 240],
      "resnet_dt": true
    },
    "modifier": {
      "type": "dipole_charge",
      "model_name": "dw.pb",
      "model_charge_map": [-8],
      "sys_charge_map": [6, 1],
      "ewald_h": 1.00,
      "ewald_beta": 0.40
    }
  },
  "learning_rate": {
    "type": "exp",
    "start_lr": 0.001
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
    "numb_steps": 100000,
    "batch_size": 1,
    "systems": ["./data/dplr/"]
  }
}
```

## 5. LAMMPS输入脚本模板

### 5.1 标准ML势MD (in.lammps)

```lammps
# 初始设置
units metal
boundary p p p
atom_style atomic

# 读取结构
read_data structure.data

# 势函数设置 (以DeepMD为例)
pair_style deepmd graph.pb
timestep 0.0005  # 0.5 fs

# 邻居列表
neighbor 1.0 bin
neigh_modify every 10 delay 0 check no

# 温度初始化
velocity all create 300.0 12345

# 固定约束（可选）
# fix 1 bottom setforce 0 0 0

# 热浴
fix 1 all nvt temp 300.0 300.0 0.1

# 输出
thermo 100
thermo_style custom step temp pe ke etotal press

# 轨迹输出
dump 1 all custom 100 traj.dump id type x y z fx fy fz

# 运行
run 100000
```

### 5.2 DPLR MD脚本 (in.dplr.lammps)

```lammps
units metal
boundary p p p
atom_style atomic

read_data water.data

# 原子组定义
group real_atom type 1 2      # O=1, H=2
group virtual_atom type 3      # Wannier中心

# DPLR势
pair_style deepmd ener.pb
tair_coeff * *

# 虚拟键设置
bond_style zero
bond_coeff *
special_bonds lj/coul 1 1 1 angle no

# 长程静电
kspace_style pppm/dplr 1e-5
kspace_modify gewald 0.40 diff ik mesh 64 64 64

# DPLR fix
fix 0 all dplr model ener.pb type_associate 1 3 bond_type 1
fix_modify 0 virial yes

# 温度计算（仅真实原子）
compute real_temp real_atom temp
compute real_press all pressure real_temp

# 热浴（仅真实原子）
fix 1 real_atom nvt temp 300.0 300.0 0.1
fix_modify 1 temp real_temp

timestep 0.0005

thermo 100
thermo_style custom step pe ke etotal c_real_temp c_real_press

dump 1 all custom 100 traj.dump id type x y z

run 100000
```

## 6. 主动学习Python脚本模板

### 6.1 不确定性查询 (active_learning.py)

```python
import numpy as np
import ase
from ase import Atoms
from ase.io import read, write
from ase.md import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

class ActiveLearner:
    def __init__(self, models, uncertainty_threshold=0.1):
        """
        models: 委员会模型列表
        uncertainty_threshold: 不确定性阈值
        """
        self.models = models
        self.threshold = uncertainty_threshold
        self.selected_structures = []
    
    def calculate_uncertainty(self, atoms: Atoms) -> float:
        """计算委员会不确定性"""
        forces_list = []
        for model in self.models:
            atoms_copy = atoms.copy()
            atoms_copy.calc = model
            forces_list.append(atoms_copy.get_forces())
        
        # 计算力预测的标准差
        forces_array = np.array(forces_list)
        uncertainty = np.std(forces_array, axis=0).mean()
        
        return uncertainty
    
    def select_structures(self, trajectory, output_file='selected.xyz'):
        """从轨迹中选择高不确定性构型"""
        uncertainties = []
        
        for atoms in trajectory:
            uq = self.calculate_uncertainty(atoms)
            uncertainties.append(uq)
        
        # 自适应阈值
        uq_mean = np.mean(uncertainties[:100])  # 前100步均值
        uq_std = np.std(uncertainties[:100])
        adaptive_threshold = uq_mean + 3 * uq_std
        
        # 选择
        selected = []
        for i, (atoms, uq) in enumerate(zip(trajectory, uncertainties)):
            if uq > adaptive_threshold:
                atoms.info['uncertainty'] = uq
                selected.append(atoms)
        
        write(output_file, selected)
        print(f"Selected {len(selected)} structures for DFT labeling")
        return selected

# 使用示例
def main():
    # 加载委员会模型
    from mace.calculators import MACECalculator
    
    committee = [
        MACECalculator(model_paths=f' committee_model_{i}.model', device='cuda')
        for i in range(5)
    ]
    
    # 初始化AL
    al = ActiveLearner(committee)
    
    # 读取MD轨迹
    traj = read('md_traj.xyz', index=':')
    
    # 选择结构
    selected = al.select_structures(traj)
    
    # 下一步：使用DFT计算这些结构的能量和力
    # 然后重新训练模型

if __name__ == '__main__':
    main()
```

## 7. 超参数调优建议

### 7.1 学习率
- 初始: 0.001 (Adam)
- 微调: 0.0001 或更低
- 使用ReduceLROnPlateau调度器

### 7.2 损失权重
- 能量: 1.0
- 力: 10-100 (取决于体系)
- 应力: 0.1-1.0 (如果需要)

### 7.3 截断半径
- 金属: 5-6 Å
- 共价体系: 4-5 Å
- 离子体系: 6-7 Å (考虑DPLR)

### 7.4 网络深度
- NequIP: 3-5层
- MACE: 1层消息传递 + 高阶特征
- Allegro: 1-2层

---

*配置模板收集时间: 2026-03-08*
