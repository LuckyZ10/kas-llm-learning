# 03 - ML势训练完整指南 | ML Potential Training Guide

> **学习目标**: 掌握DeePMD-kit、NEP等ML势的训练流程、参数调优和模型验证  
> **Learning Goal**: Master training workflows for DeePMD-kit, NEP, and model validation

---

## 📋 目录 | Table of Contents

1. [理论基础 | Theoretical Background](#1-理论基础--theoretical-background)
2. [数据准备 | Data Preparation](#2-数据准备--data-preparation)
3. [DeePMD-kit训练 | DeePMD-kit Training](#3-deepmd-kit训练--deepmd-kit-training)
4. [NEP训练 | NEP Training](#4-nep-training--nep-training)
5. [模型验证 | Model Validation](#5-模型验证--model-validation)
6. [模型压缩 | Model Compression](#6-模型压缩--model-compression)
7. [常见错误 | Common Errors](#7-常见错误--common-errors)
8. [练习题 | Exercises](#8-练习题--exercises)

---

## 1. 理论基础 | Theoretical Background

### 1.1 ML势概述 | ML Potential Overview

机器学习势函数(ML Potential)通过神经网络拟合DFT数据，实现：
- **精度**: 接近DFT的准确性
- **速度**: 比DFT快1000-10000倍
- **尺度**: 可模拟百万原子体系

```
┌────────────────────────────────────────────────────────────────┐
│                      ML势架构对比                               │
├──────────────┬─────────────────────────────────────────────────┤
│ DeepPot-SE   │ 平滑EAM-like描述符 + DNN                        │
│              │ 速度: ★★★★★  精度: ★★★★☆                      │
├──────────────┼─────────────────────────────────────────────────┤
│ DPA-2        │ Attention机制 + Transformer                     │
│              │ 速度: ★★★★☆  精度: ★★★★★                      │
├──────────────┼─────────────────────────────────────────────────┤
│ NEP          │ 神经进化势 + 单隐层NN                           │
│              │ 速度: ★★★★★  精度: ★★★★☆                      │
├──────────────┼─────────────────────────────────────────────────┤
│ M3GNet       │ 图卷积神经网络                                  │
│              │ 速度: ★★★★☆  精度: ★★★★☆                      │
└──────────────┴─────────────────────────────────────────────────┘
```

### 1.2 DeepPot-SE架构 | DeepPot-SE Architecture

```
输入坐标 (R)                        能量预测 (E)
    │                                    ▲
    ▼                                    │
┌─────────────────┐              ┌───────────────┐
│  邻居搜索        │              │   拟合网络     │
│  Neighbor Search │─────────────▶│  Fitting Net  │
└─────────────────┘   描述符      └───────────────┘
    │                   │                ▲
    ▼                   │                │
┌─────────────────┐     │         ┌───────────────┐
│   环境矩阵       │     │         │   嵌入网络     │
│  Environment    │─────┘         │  Embedding    │
│    Matrix       │               │   Network     │
└─────────────────┘               └───────────────┘
    ▲                                    │
    │                                    │
┌─────────────────┐                      │
│  平滑截断函数    │                      │
│ Smooth Switching│──────────────────────┘
└─────────────────┘
```

**能量表达式**:

$$
E = \sum_i E_i = \sum_i F_{fit}(\mathcal{D}(R^{(i)}))
$$

其中 $\mathcal{D}$ 是描述符网络，$F_{fit}$ 是拟合网络。

---

## 2. 数据准备 | Data Preparation

### 2.1 数据格式 | Data Format

DeePMD使用NumPy压缩格式(.npz)：

```
data/
├── training/
│   └── Li3PS4/
│       ├── set.000/
│       │   ├── box.npy      # 晶胞矩阵 (nframes, 9)
│       │   ├── coord.npy    # 坐标 (nframes, natoms*3)
│       │   ├── energy.npy   # 能量 (nframes,)
│       │   ├── force.npy    # 力 (nframes, natoms*3)
│       │   └── virial.npy   # 维里 (nframes, 9) [可选]
│       ├── type.raw         # 原子类型索引
│       └── type_map.raw     # 元素名称映射
└── validation/
    └── Li3PS4/
        └── ...
```

### 2.2 数据预处理脚本 | Data Preprocessing Script

```python
#!/usr/bin/env python3
"""
数据预处理工具 | Data Preprocessing Tool
"""
import dpdata
import numpy as np
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt

class DataPreprocessor:
    """训练数据预处理器 | Training data preprocessor"""
    
    def __init__(self, type_map: List[str]):
        """
        Args:
            type_map: 元素列表，如 ['Li', 'P', 'S']
        """
        self.type_map = type_map
        
    def convert_vasp_to_deepmd(self,
                               vasp_dirs: List[str],
                               output_dir: str,
                               train_ratio: float = 0.9) -> Tuple[str, str]:
        """
        将VASP输出转换为DeePMD格式
        
        Args:
            vasp_dirs: VASP计算目录列表
            output_dir: 输出目录
            train_ratio: 训练集比例
            
        Returns:
            (training_dir, validation_dir)
        """
        print("="*60)
        print("Converting VASP data to DeePMD format...")
        print("="*60)
        
        all_systems = []
        
        # 读取所有VASP输出
        for vasp_dir in vasp_dirs:
            outcar_path = Path(vasp_dir) / "OUTCAR"
            if not outcar_path.exists():
                print(f"⚠️  OUTCAR not found: {vasp_dir}")
                continue
            
            try:
                system = dpdata.LabeledSystem(str(outcar_path), fmt='vasp/outcar')
                
                # 过滤异常能量
                system = self._filter_outliers(system)
                
                all_systems.append(system)
                print(f"✓ Loaded {len(system)} frames from {vasp_dir}")
                
            except Exception as e:
                print(f"✗ Failed to load {vasp_dir}: {e}")
        
        if not all_systems:
            raise ValueError("No valid VASP data found!")
        
        # 合并系统
        multi_systems = dpdata.MultiSystems(*all_systems)
        
        # 创建输出目录
        train_dir = Path(output_dir) / "training"
        valid_dir = Path(output_dir) / "validation"
        train_dir.mkdir(parents=True, exist_ok=True)
        valid_dir.mkdir(parents=True, exist_ok=True)
        
        # 分割并保存
        for name, system in multi_systems.systems.items():
            n_frames = len(system)
            n_train = int(n_frames * train_ratio)
            
            # 随机打乱
            indices = np.random.permutation(n_frames)
            train_idx = indices[:n_train]
            valid_idx = indices[n_train:]
            
            # 保存
            train_system = system.sub_system(train_idx)
            valid_system = system.sub_system(valid_idx)
            
            train_system.to_deepmd_npy(str(train_dir / name))
            valid_system.to_deepmd_npy(str(valid_dir / name))
            
            print(f"\nSystem: {name}")
            print(f"  Total frames: {n_frames}")
            print(f"  Training: {n_train}")
            print(f"  Validation: {len(valid_idx)}")
        
        return str(train_dir), str(valid_dir)
    
    def _filter_outliers(self, system, energy_threshold: float = 5.0) -> dpdata.LabeledSystem:
        """过滤能量异常值"""
        energies = system.data['energies']
        n_atoms = sum(system.data['atom_numbs'])
        e_per_atom = energies / n_atoms
        
        mean_e = np.mean(e_per_atom)
        std_e = np.std(e_per_atom)
        
        # 保留在阈值范围内的帧
        valid_mask = np.abs(e_per_atom - mean_e) < energy_threshold * std_e
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) < len(energies):
            print(f"  Filtered {len(energies) - len(valid_indices)} outlier frames")
        
        return system.sub_system(valid_indices)
    
    def analyze_data_distribution(self, data_dir: str):
        """分析数据分布"""
        import glob
        
        all_energies = []
        all_forces = []
        
        # 收集所有数据
        for energy_file in Path(data_dir).rglob('energy.npy'):
            energies = np.load(energy_file)
            all_energies.extend(energies)
            
        for force_file in Path(data_dir).rglob('force.npy'):
            forces = np.load(force_file)
            all_forces.extend(forces.flatten())
        
        all_energies = np.array(all_energies)
        all_forces = np.array(all_forces)
        
        # 绘制分布图
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 能量分布
        axes[0, 0].hist(all_energies, bins=50, edgecolor='black')
        axes[0, 0].set_xlabel('Energy (eV)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Energy Distribution')
        
        # 能量每原子分布
        axes[0, 1].hist(all_energies / 32, bins=50, edgecolor='black')  # 假设32原子
        axes[0, 1].set_xlabel('Energy per Atom (eV/atom)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Energy per Atom Distribution')
        
        # 力分布
        axes[1, 0].hist(all_forces, bins=100, edgecolor='black')
        axes[1, 0].set_xlabel('Force (eV/Å)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Force Distribution')
        
        # 力分布(log scale)
        axes[1, 1].hist(np.abs(all_forces), bins=100, edgecolor='black')
        axes[1, 1].set_xlabel('|Force| (eV/Å)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_yscale('log')
        axes[1, 1].set_title('Force Magnitude (log scale)')
        
        plt.tight_layout()
        plt.savefig('data_distribution.png', dpi=150)
        
        # 打印统计信息
        print("\n" + "="*60)
        print("Data Statistics:")
        print("="*60)
        print(f"Total frames: {len(all_energies)}")
        print(f"\nEnergy (eV):")
        print(f"  Mean: {np.mean(all_energies):.4f}")
        print(f"  Std:  {np.std(all_energies):.4f}")
        print(f"  Min:  {np.min(all_energies):.4f}")
        print(f"  Max:  {np.max(all_energies):.4f}")
        print(f"\nForce (eV/Å):")
        print(f"  Mean: {np.mean(all_forces):.4f}")
        print(f"  Std:  {np.std(all_forces):.4f}")
        print(f"  Max:  {np.max(np.abs(all_forces)):.4f}")


# 使用示例
if __name__ == "__main__":
    preprocessor = DataPreprocessor(type_map=['Li', 'P', 'S'])
    
    # 转换数据
    train_dir, valid_dir = preprocessor.convert_vasp_to_deepmd(
        vasp_dirs=['./vasp_run1', './vasp_run2', './vasp_run3'],
        output_dir='./training_data',
        train_ratio=0.9
    )
    
    # 分析数据分布
    preprocessor.analyze_data_distribution(train_dir)
```

---

## 3. DeePMD-kit训练 | DeePMD-kit Training

### 3.1 输入文件详解 | Input File Details

```json
{
  "model": {
    "type_map": ["Li", "P", "S"],
    
    "descriptor": {
      "type": "se_e2_a",
      "rcut": 6.0,
      "rcut_smth": 0.5,
      "sel": [50, 50, 50],
      "neuron": [25, 50, 100],
      "resnet_dt": false,
      "axis_neuron": 16,
      "seed": 1,
      "type_one_side": true
    },
    
    "fitting_net": {
      "neuron": [240, 240, 240],
      "resnet_dt": true,
      "seed": 1
    }
  },
  
  "learning_rate": {
    "type": "exp",
    "decay_steps": 5000,
    "start_lr": 0.001,
    "stop_lr": 3.51e-8
  },
  
  "loss": {
    "type": "ener",
    "start_pref_e": 0.02,
    "limit_pref_e": 1,
    "start_pref_f": 1000,
    "limit_pref_f": 1,
    "start_pref_v": 0.01,
    "limit_pref_v": 1
  },
  
  "training": {
    "training_data": {
      "systems": ["./training_data/*"],
      "batch_size": "auto"
    },
    "validation_data": {
      "systems": ["./validation_data/*"],
      "batch_size": "auto"
    },
    "numb_steps": 1000000,
    "seed": 10,
    "disp_file": "lcurve.out",
    "disp_freq": 1000,
    "save_freq": 10000,
    "max_ckpt_keep": 5
  }
}
```

**参数说明**:

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `rcut` | 截断半径 | 6.0-8.0 Å |
| `rcut_smth` | 平滑截断宽度 | 0.5-1.0 Å |
| `sel` | 最大邻居数 | 按系统密度设置 |
| `neuron` | 描述符网络结构 | [25, 50, 100] |
| `axis_neuron` | 轴神经元数 | 8-16 |
| `fitting_neuron` | 拟合网络结构 | [240, 240, 240] |
| `start_lr` | 初始学习率 | 0.001 |
| `decay_steps` | 学习率衰减步数 | 5000-10000 |
| `start_pref_f` | 初始力权重 | 1000 |
| `limit_pref_f` | 最终力权重 | 1 |

### 3.2 训练脚本 | Training Script

```python
#!/usr/bin/env python3
"""
DeePMD-kit训练脚本 | DeePMD-kit Training Script
"""
import json
import subprocess
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict


@dataclass
class DeepMDConfig:
    """DeePMD配置"""
    # 模型架构
    type_map: List[str]
    descriptor_type: str = "se_e2_a"
    rcut: float = 6.0
    rcut_smth: float = 0.5
    sel: List[int] = None
    neuron: List[int] = None
    axis_neuron: int = 16
    fitting_neuron: List[int] = None
    
    # 训练参数
    start_lr: float = 0.001
    stop_lr: float = 3.51e-8
    decay_steps: int = 5000
    numb_steps: int = 1000000
    batch_size: str = "auto"
    
    # 损失权重
    start_pref_e: float = 0.02
    limit_pref_e: float = 1.0
    start_pref_f: float = 1000.0
    limit_pref_f: float = 1.0
    start_pref_v: float = 0.01
    limit_pref_v: float = 1.0
    
    # 数据路径
    training_data: str = "./training_data/*"
    validation_data: str = "./validation_data/*"
    
    # 输出
    output_dir: str = "./model"
    
    def __post_init__(self):
        if self.sel is None:
            self.sel = [50] * len(self.type_map)
        if self.neuron is None:
            self.neuron = [25, 50, 100]
        if self.fitting_neuron is None:
            self.fitting_neuron = [240, 240, 240]


class DeepMDTrainer:
    """DeePMD训练器"""
    
    def __init__(self, config: DeepMDConfig):
        self.config = config
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
    def generate_input(self, filename: str = "input.json"):
        """生成输入文件"""
        input_dict = {
            "model": {
                "type_map": self.config.type_map,
                "descriptor": {
                    "type": self.config.descriptor_type,
                    "rcut": self.config.rcut,
                    "rcut_smth": self.config.rcut_smth,
                    "sel": self.config.sel,
                    "neuron": self.config.neuron,
                    "resnet_dt": False,
                    "axis_neuron": self.config.axis_neuron,
                    "seed": 1,
                    "type_one_side": True
                },
                "fitting_net": {
                    "neuron": self.config.fitting_neuron,
                    "resnet_dt": True,
                    "seed": 1
                }
            },
            "learning_rate": {
                "type": "exp",
                "decay_steps": self.config.decay_steps,
                "start_lr": self.config.start_lr,
                "stop_lr": self.config.stop_lr
            },
            "loss": {
                "type": "ener",
                "start_pref_e": self.config.start_pref_e,
                "limit_pref_e": self.config.limit_pref_e,
                "start_pref_f": self.config.start_pref_f,
                "limit_pref_f": self.config.limit_pref_f,
                "start_pref_v": self.config.start_pref_v,
                "limit_pref_v": self.config.limit_pref_v
            },
            "training": {
                "training_data": {
                    "systems": [self.config.training_data],
                    "batch_size": self.config.batch_size
                },
                "validation_data": {
                    "systems": [self.config.validation_data],
                    "batch_size": self.config.batch_size
                },
                "numb_steps": self.config.numb_steps,
                "seed": 10,
                "disp_file": "lcurve.out",
                "disp_freq": 1000,
                "save_freq": 10000,
                "max_ckpt_keep": 5
            }
        }
        
        input_path = Path(self.config.output_dir) / filename
        with open(input_path, 'w') as f:
            json.dump(input_dict, f, indent=2)
        
        print(f"✓ Generated input file: {input_path}")
        return str(input_path)
    
    def train(self, input_file: str = None, restart: bool = False):
        """执行训练"""
        if input_file is None:
            input_file = self.generate_input()
        
        print("\n" + "="*60)
        print("Starting DeePMD training...")
        print("="*60)
        
        cmd = ["dp", "train", input_file]
        if restart:
            cmd.append("--restart")
        
        try:
            subprocess.run(cmd, cwd=self.config.output_dir, check=True)
            print("✓ Training completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"✗ Training failed: {e}")
            raise
    
    def freeze_model(self, model_name: str = "graph.pb"):
        """冻结模型"""
        print("\nFreezing model...")
        
        cmd = ["dp", "freeze", "-o", model_name]
        subprocess.run(cmd, cwd=self.config.output_dir, check=True)
        
        model_path = Path(self.config.output_dir) / model_name
        print(f"✓ Model frozen: {model_path}")
        return str(model_path)
    
    def compress_model(self, model_path: str, compressed_name: str = "graph-compress.pb"):
        """压缩模型"""
        print("\nCompressing model...")
        
        compressed_path = Path(self.config.output_dir) / compressed_name
        cmd = [
            "dp", "compress",
            "-i", model_path,
            "-o", str(compressed_path)
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✓ Model compressed: {compressed_path}")
            return str(compressed_path)
        except subprocess.CalledProcessError:
            print("⚠️ Compression failed, using uncompressed model")
            return model_path
    
    def train_ensemble(self, n_models: int = 4) -> List[str]:
        """训练模型集成"""
        model_paths = []
        
        for i in range(n_models):
            print(f"\n{'='*60}")
            print(f"Training model {i+1}/{n_models}")
            print(f"{'='*60}")
            
            # 修改种子
            model_dir = Path(self.config.output_dir) / f"model_{i}"
            model_dir.mkdir(exist_ok=True)
            
            config_copy = DeepMDConfig(**asdict(self.config))
            config_copy.output_dir = str(model_dir)
            # 使用不同种子
            
            trainer = DeepMDTrainer(config_copy)
            input_file = trainer.generate_input()
            trainer.train(input_file)
            
            model_path = trainer.freeze_model(f"graph_{i}.pb")
            model_paths.append(model_path)
        
        return model_paths
    
    def plot_learning_curve(self):
        """绘制学习曲线"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        lcurve_file = Path(self.config.output_dir) / "lcurve.out"
        if not lcurve_file.exists():
            print(f"Learning curve file not found: {lcurve_file}")
            return
        
        # 读取学习曲线数据
        data = np.loadtxt(lcurve_file)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        steps = data[:, 0]
        
        # 总损失
        axes[0, 0].semilogy(steps, data[:, 1], label='Total')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].grid(True)
        
        # 能量RMSE
        if data.shape[1] > 4:
            axes[0, 1].semilogy(steps, data[:, 4], label='Train')
            axes[0, 1].semilogy(steps, data[:, 5], label='Validation')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Energy RMSE (eV)')
            axes[0, 1].set_title('Energy RMSE')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # 力RMSE
        if data.shape[1] > 6:
            axes[1, 0].semilogy(steps, data[:, 6], label='Train')
            axes[1, 0].semilogy(steps, data[:, 7], label='Validation')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Force RMSE (eV/Å)')
            axes[1, 0].set_title('Force RMSE')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # 学习率
        if data.shape[1] > 2:
            axes[1, 1].semilogy(steps, data[:, 2])
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(Path(self.config.output_dir) / 'learning_curve.png', dpi=150)
        print(f"✓ Learning curve saved to {self.config.output_dir}/learning_curve.png")


# 主程序
if __name__ == "__main__":
    config = DeepMDConfig(
        type_map=["Li", "P", "S"],
        training_data="./training_data/*",
        validation_data="./validation_data/*",
        numb_steps=1000000,
        output_dir="./deepmd_model"
    )
    
    trainer = DeepMDTrainer(config)
    
    # 训练单个模型
    trainer.train()
    model_path = trainer.freeze_model()
    compressed = trainer.compress_model(model_path)
    
    # 绘制学习曲线
    trainer.plot_learning_curve()
    
    print("\n" + "="*60)
    print("Training pipeline completed!")
    print(f"Model: {compressed}")
    print("="*60)
```

### 3.3 训练监控 | Training Monitoring

```bash
# 实时监控训练 | Monitor training in real-time
watch -n 10 tail -n 20 lcurve.out

# 或绘制学习曲线 | Plot learning curve
python -c "import dpdata; dpdata.stat('lcurve.out')"
```

**学习曲线解读**:

```
# lcurve.out 格式
# step  lr  loss_tr  loss_val  ener_tr  ener_val  force_tr  force_val
0      0.001  1.23e+02  1.25e+02  0.5     0.52      10.0      10.5
1000   0.0009 8.5e+01   8.7e+01   0.3     0.32      8.0       8.2
...
```

**收敛标准**:
- Energy RMSE < 10 meV/atom
- Force RMSE < 100 meV/Å = 0.1 eV/Å
- Validation loss 稳定不上升

---

## 4. NEP训练 | NEP Training

### 4.1 NEP简介 | NEP Introduction

NEP (Neuro-Evolutionary Potential) 是GPUMD框架中的ML势：
- **速度**: GPU加速，比CPU快100倍
- **精度**: 与DeePMD相当
- **训练**: 使用进化算法，无需梯度下降

### 4.2 NEP输入文件 | NEP Input File

```bash
# nep.in - NEP训练配置
type Li P S                 # 元素类型
version 4                   # NEP版本
cutoff 6.0 4.0              # 径向和角向截断 (Å)
n_max 4 4                   # 径向和角向基函数数
basis_size 8 8              # 基函数大小
l_max 4                     # 角动量最大值
neuron 30                   # 神经网络隐层神经元数
population 50               # 种群大小
generation 100000           # 最大代数
batch 1000                  # 批次大小
```

### 4.3 NEP训练脚本 | NEP Training Script

```python
#!/usr/bin/env python3
"""
NEP训练脚本 | NEP Training Script
"""
import subprocess
import os
from pathlib import Path


class NEPTrainer:
    """NEP训练器"""
    
    def __init__(self, 
                 type_list: List[str],
                 cutoff_radial: float = 6.0,
                 cutoff_angular: float = 4.0,
                 n_max: int = 4,
                 basis_size: int = 8,
                 l_max: int = 4,
                 neuron: int = 30,
                 population: int = 50,
                 generation: int = 100000,
                 work_dir: str = "./nep_model"):
        self.type_list = type_list
        self.cutoff_radial = cutoff_radial
        self.cutoff_angular = cutoff_angular
        self.n_max = n_max
        self.basis_size = basis_size
        self.l_max = l_max
        self.neuron = neuron
        self.population = population
        self.generation = generation
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_input(self):
        """生成nep.in"""
        content = f"""type {' '.join(self.type_list)}
version 4
cutoff {self.cutoff_radial} {self.cutoff_angular}
n_max {self.n_max} {self.n_max}
basis_size {self.basis_size} {self.basis_size}
l_max {self.l_max}
neuron {self.neuron}
population {self.population}
generation {self.generation}
batch 1000
"""
        input_file = self.work_dir / "nep.in"
        with open(input_file, 'w') as f:
            f.write(content)
        print(f"✓ Generated {input_file}")
        return str(input_file)
    
    def prepare_data(self, train_xyz: str, test_xyz: str = None):
        """准备训练数据"""
        import shutil
        
        # 复制训练数据
        dst_train = self.work_dir / "train.xyz"
        shutil.copy(train_xyz, dst_train)
        print(f"✓ Copied training data to {dst_train}")
        
        # 复制测试数据
        if test_xyz and Path(test_xyz).exists():
            dst_test = self.work_dir / "test.xyz"
            shutil.copy(test_xyz, dst_test)
            print(f"✓ Copied test data to {dst_test}")
    
    def train(self):
        """执行训练"""
        print("\n" + "="*60)
        print("Starting NEP training...")
        print("="*60)
        
        # 生成输入
        self.generate_input()
        
        # 运行NEP
        try:
            result = subprocess.run(
                ["nep"],
                cwd=self.work_dir,
                capture_output=True,
                text=True
            )
            print(result.stdout)
            
            if result.returncode != 0:
                print(f"Error: {result.stderr}")
                raise RuntimeError("NEP training failed")
            
            print("✓ Training completed!")
            
        except FileNotFoundError:
            print("✗ 'nep' command not found. Please install GPUMD.")
            raise
    
    def get_model_path(self):
        """获取模型路径"""
        model_path = self.work_dir / "nep.txt"
        if model_path.exists():
            return str(model_path)
        return None


# 转换数据为XYZ格式
def convert_to_nep_xyz(dpdata_system, output_file: str):
    """将dpdata转换为NEP的xyz格式"""
    # 简化的转换函数
    # 实际实现需要根据NEP格式要求
    pass
```

---

## 5. 模型验证 | Model Validation

### 5.1 测试集验证 | Test Set Validation

```python
"""
模型验证脚本 | Model Validation Script
"""
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from deepmd.infer import DeepPot
from ase import Atoms


def test_model(model_path: str, test_data: str):
    """测试模型 | Test model"""
    print(f"\nTesting model: {model_path}")
    
    cmd = ["dp", "test", "-m", model_path, "-s", test_data, "-d", "test_results"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    
    # 解析结果
    energy_rmse = None
    force_rmse = None
    
    for line in result.stdout.split('\n'):
        if 'Energy RMSE' in line and 'Natoms' not in line:
            try:
                energy_rmse = float(line.split(':')[1].split()[0])
            except:
                pass
        elif 'Force RMSE' in line:
            try:
                force_rmse = float(line.split(':')[1].split()[0])
            except:
                pass
    
    return {
        'energy_rmse': energy_rmse,
        'force_rmse': force_rmse
    }


def plot_predictions(model_path: str, test_structures: List[Atoms]):
    """绘制预测vs真实值"""
    
    # 加载模型
    model = DeepPot(model_path)
    
    pred_energies = []
    true_energies = []
    pred_forces = []
    true_forces = []
    
    for atoms in test_structures:
        # 准备输入
        coord = atoms.get_positions().reshape(1, -1)
        cell = atoms.get_cell().array.reshape(1, -1)
        atype = np.array([model.get_type_map().index(s) 
                         for s in atoms.get_chemical_symbols()])
        
        # 预测
        e, f, v = model.eval(coord, cell, atype)
        
        pred_energies.append(e[0][0])
        true_energies.append(atoms.get_potential_energy())
        pred_forces.extend(f[0].flatten())
        true_forces.extend(atoms.get_forces().flatten())
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 能量对比
    axes[0].scatter(true_energies, pred_energies, alpha=0.5)
    min_e = min(min(true_energies), min(pred_energies))
    max_e = max(max(true_energies), max(pred_energies))
    axes[0].plot([min_e, max_e], [min_e, max_e], 'r--', label='y=x')
    axes[0].set_xlabel('DFT Energy (eV)')
    axes[0].set_ylabel('ML Predicted Energy (eV)')
    axes[0].set_title('Energy Prediction')
    axes[0].legend()
    axes[0].grid(True)
    
    # 力对比
    axes[1].scatter(true_forces, pred_forces, alpha=0.1)
    min_f = min(min(true_forces), min(pred_forces))
    max_f = max(max(true_forces), max(pred_forces))
    axes[1].plot([min_f, max_f], [min_f, max_f], 'r--', label='y=x')
    axes[1].set_xlabel('DFT Force (eV/Å)')
    axes[1].set_ylabel('ML Predicted Force (eV/Å)')
    axes[1].set_title('Force Prediction')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('prediction_comparison.png', dpi=150)
    print("✓ Prediction comparison saved to prediction_comparison.png")


# 计算RMSE
def compute_rmse(y_true, y_pred):
    """计算RMSE"""
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))


def compute_mae(y_true, y_pred):
    """计算MAE"""
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
```

### 5.2 MD验证 | MD Validation

```python
"""
MD验证 - 检查势函数在模拟中的稳定性
MD Validation - Check potential stability in simulations
"""
from ase import Atoms
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from deepmd.calculator import DP
import matplotlib.pyplot as plt


def validate_with_md(structure: Atoms,
                     model_path: str,
                     temperature: float = 300,
                     n_steps: int = 10000,
                     timestep: float = 1.0):
    """
    通过MD验证模型稳定性
    Validate model stability with MD
    """
    atoms = structure.copy()
    
    # 设置计算器
    calc = DP(model=model_path)
    atoms.calc = calc
    
    # 初始化速度
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    
    # MD运行器
    dyn = Langevin(atoms, timestep=timestep, temperature_K=temperature, friction=0.01)
    
    # 记录轨迹
    energies = []
    temperatures = []
    steps = []
    
    def log_data():
        e = atoms.get_potential_energy()
        ke = atoms.get_kinetic_energy()
        T = ke / (1.5 * len(atoms) * 8.617e-5)  # 近似温度
        
        energies.append(e)
        temperatures.append(T)
        steps.append(dyn.get_number_of_steps())
    
    dyn.attach(log_data, interval=100)
    
    # 运行MD
    print(f"Running MD at {temperature}K for {n_steps} steps...")
    dyn.run(n_steps)
    
    # 分析结果
    energy_drift = (energies[-1] - energies[0]) / n_steps
    
    print(f"\nMD Validation Results:")
    print(f"  Energy drift: {energy_drift:.6f} eV/step")
    print(f"  Mean temperature: {np.mean(temperatures):.1f} K")
    print(f"  Temperature std: {np.std(temperatures):.1f} K")
    
    # 判断稳定性
    if abs(energy_drift) < 1e-5 and np.std(temperatures) < temperature * 0.2:
        print("  ✓ Model appears stable")
        stable = True
    else:
        print("  ⚠️ Model may have stability issues")
        stable = False
    
    # 绘图
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    axes[0].plot(steps, energies)
    axes[0].set_ylabel('Energy (eV)')
    axes[0].set_title('MD Energy Trajectory')
    axes[0].grid(True)
    
    axes[1].plot(steps, temperatures)
    axes[1].axhline(y=temperature, color='r', linestyle='--', label='Target')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Temperature (K)')
    axes[1].set_title('MD Temperature')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('md_validation.png', dpi=150)
    
    return stable, energies, temperatures
```

---

## 6. 模型压缩 | Model Compression

### 6.1 压缩原理 | Compression Theory

模型压缩通过表格化存储神经网络输出：
- **速度提升**: 10-50倍
- **内存节省**: 20-100倍
- **精度损失**: < 1%

```
原始模型: 实时计算神经网络
          ┌─────────────┐
输入描述符 ──▶│  DNN计算   │──▶ 能量/力
          └─────────────┘
          
压缩模型: 查表插值
          ┌─────────────┐
输入描述符 ──▶│  查表+插值  │──▶ 能量/力
          └─────────────┘
```

### 6.2 压缩脚本 | Compression Script

```python
"""
模型压缩 | Model Compression
"""
import subprocess
from pathlib import Path


def compress_deepmd_model(input_model: str,
                          output_model: str = None,
                          step: float = 0.001,
                          extrapolate: float = 5.0):
    """
    压缩DeePMD模型
    
    Args:
        input_model: 输入模型路径
        output_model: 输出模型路径 (默认添加-compress后缀)
        step: 表格步长
        extrapolate: 外推参数
    """
    if output_model is None:
        output_model = input_model.replace('.pb', '-compress.pb')
    
    print(f"Compressing model: {input_model}")
    print(f"Output: {output_model}")
    
    cmd = [
        "dp", "compress",
        "-i", input_model,
        "-o", output_model,
        "-s", str(step),
        "-e", str(extrapolate)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Compression successful!")
        
        # 比较文件大小
        original_size = Path(input_model).stat().st_size / (1024**2)  # MB
        compressed_size = Path(output_model).stat().st_size / (1024**2)  # MB
        
        print(f"\nFile size comparison:")
        print(f"  Original:   {original_size:.2f} MB")
        print(f"  Compressed: {compressed_size:.2f} MB")
        print(f"  Ratio:      {original_size/compressed_size:.1f}x")
        
        return output_model
    else:
        print(f"✗ Compression failed: {result.stderr}")
        return None


def benchmark_model(model_path: str, n_iterations: int = 1000):
    """基准测试模型速度"""
    from deepmd.infer import DeepPot
    import numpy as np
    import time
    
    model = DeepPot(model_path)
    
    # 准备测试数据 (假设Li3PS4, 32原子)
    n_atoms = 32
    coord = np.random.rand(1, n_atoms * 3)
    cell = np.eye(3).reshape(1, 9) * 10
    atype = np.array([0]*12 + [1]*4 + [2]*16)  # Li, P, S
    
    # 预热
    for _ in range(10):
        model.eval(coord, cell, atype)
    
    # 计时
    start = time.time()
    for _ in range(n_iterations):
        model.eval(coord, cell, atype)
    elapsed = time.time() - start
    
    avg_time = elapsed / n_iterations * 1000  # ms
    
    print(f"\nBenchmark results for {model_path}:")
    print(f"  Average time: {avg_time:.3f} ms/evaluation")
    print(f"  Throughput:   {1000/avg_time:.1f} evaluations/sec")
    
    return avg_time


def compare_original_compressed(original: str, compressed: str):
    """比较原始模型和压缩模型"""
    print("\n" + "="*60)
    print("Comparing original and compressed models")
    print("="*60)
    
    # 文件大小
    orig_size = Path(original).stat().st_size / (1024**2)
    comp_size = Path(compressed).stat().st_size / (1024**2)
    
    print(f"\nFile size:")
    print(f"  Original:   {orig_size:.2f} MB")
    print(f"  Compressed: {comp_size:.2f} MB")
    print(f"  Ratio:      {orig_size/comp_size:.1f}x")
    
    # 速度测试
    print("\nSpeed benchmark:")
    orig_time = benchmark_model(original)
    comp_time = benchmark_model(compressed)
    
    print(f"\nSpeedup: {orig_time/comp_time:.1f}x")
```

---

## 7. 常见错误 | Common Errors

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `Neighbor list overflow` | sel设置太小 | 增加sel值 |
| `Loss is NaN` | 学习率过大或数据异常 | 降低学习率，检查数据 |
| `OOM` | 内存不足 | 减小batch_size |
| `Model not converging` | 网络容量不足或数据不够 | 增加网络宽度/深度，增加数据 |
| `Force RMSE too high` | 力权重设置不当 | 调整start_pref_f |
| `Dimension mismatch` | type_map与数据不匹配 | 检查type_map设置 |

---

## 8. 练习题 | Exercises

### 练习 1: 超参数调优 | Exercise 1: Hyperparameter Tuning

```python
# 尝试不同的网络结构
configs = [
    DeepMDConfig(type_map=["Li", "P", "S"], neuron=[10, 20, 40]),
    DeepMDConfig(type_map=["Li", "P", "S"], neuron=[25, 50, 100]),
    DeepMDConfig(type_map=["Li", "P", "S"], neuron=[50, 100, 200]),
]

for i, config in enumerate(configs):
    config.output_dir = f"./model_config_{i}"
    trainer = DeepMDTrainer(config)
    trainer.train()
    results = test_model(trainer.get_model_path(), "./validation_data")
    print(f"Config {i}: Energy RMSE = {results['energy_rmse']:.4f}")
```

### 练习 2: 集成模型 | Exercise 2: Ensemble Models

```python
# 训练4个模型并计算预测不确定性
config = DeepMDConfig(type_map=["Li", "P", "S"])
trainer = DeepMDTrainer(config)
model_paths = trainer.train_ensemble(n_models=4)

# 使用集成计算不确定性
from deepmd.infer.model_devi import calc_model_devi_v2
```

---

**下一步**: [04 - 主动学习实战](04_active_learning.md)
