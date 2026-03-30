#!/usr/bin/env python3
"""
ML势训练示例 - DeePMD-kit
ML Potential Training Example - DeePMD-kit

步骤:
1. 准备训练数据 (从VASP输出)
2. 生成输入文件
3. 训练模型
4. 冻结和压缩模型
"""

import json
import subprocess
from pathlib import Path
import sys
sys.path.insert(0, '/root/.openclaw/workspace/dft_lammps_research')


def prepare_training_data(vasp_dirs, output_dir="./training_data"):
    """
    准备训练数据
    
    Args:
        vasp_dirs: VASP计算目录列表
        output_dir: 输出目录
    """
    import dpdata
    import numpy as np
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    all_systems = []
    
    for vasp_dir in vasp_dirs:
        outcar = Path(vasp_dir) / "OUTCAR"
        if not outcar.exists():
            print(f"Warning: OUTCAR not found in {vasp_dir}")
            continue
        
        try:
            system = dpdata.LabeledSystem(str(outcar), fmt='vasp/outcar')
            all_systems.append(system)
            print(f"✓ Loaded {len(system)} frames from {vasp_dir}")
        except Exception as e:
            print(f"✗ Failed to load {vasp_dir}: {e}")
    
    if not all_systems:
        raise ValueError("No valid VASP data found")
    
    # 合并并分割
    multi_systems = dpdata.MultiSystems(*all_systems)
    
    train_dir = Path(output_dir) / "training"
    valid_dir = Path(output_dir) / "validation"
    
    for name, system in multi_systems.systems.items():
        n_frames = len(system)
        n_train = int(n_frames * 0.9)
        
        indices = np.random.permutation(n_frames)
        
        train_system = system.sub_system(indices[:n_train])
        valid_system = system.sub_system(indices[n_train:])
        
        train_system.to_deepmd_npy(str(train_dir / name))
        valid_system.to_deepmd_npy(str(valid_dir / name))
        
        print(f"{name}: {n_train} train, {len(indices)-n_train} valid frames")
    
    return str(train_dir), str(valid_dir)


def generate_input(type_map, output_file="input.json"):
    """生成DeePMD输入文件"""
    
    input_dict = {
        "model": {
            "type_map": type_map,
            "descriptor": {
                "type": "se_e2_a",
                "rcut": 6.0,
                "rcut_smth": 0.5,
                "sel": [50] * len(type_map),
                "neuron": [25, 50, 100],
                "resnet_dt": False,
                "axis_neuron": 16,
                "seed": 1,
                "type_one_side": True
            },
            "fitting_net": {
                "neuron": [240, 240, 240],
                "resnet_dt": True,
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
            "save_freq": 10000
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(input_dict, f, indent=2)
    
    print(f"✓ Generated input file: {output_file}")
    return output_file


def train_model(input_file="input.json"):
    """训练模型"""
    print("\n" + "="*60)
    print("Starting DeePMD training...")
    print("="*60)
    
    result = subprocess.run(
        ["dp", "train", input_file],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ Training completed successfully")
    else:
        print(f"✗ Training failed:\n{result.stderr}")
        raise RuntimeError("Training failed")


def freeze_model(output_name="graph.pb"):
    """冻结模型"""
    print("\nFreezing model...")
    
    result = subprocess.run(
        ["dp", "freeze", "-o", output_name],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"✓ Model frozen: {output_name}")
    else:
        print(f"✗ Freeze failed:\n{result.stderr}")


def compress_model(input_model="graph.pb", output_model="graph-compress.pb"):
    """压缩模型"""
    print("\nCompressing model...")
    
    result = subprocess.run(
        ["dp", "compress", "-i", input_model, "-o", output_model],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"✓ Model compressed: {output_model}")
        
        # 比较文件大小
        import os
        orig_size = os.path.getsize(input_model) / (1024**2)
        comp_size = os.path.getsize(output_model) / (1024**2)
        print(f"  Original: {orig_size:.2f} MB")
        print(f"  Compressed: {comp_size:.2f} MB")
        print(f"  Ratio: {orig_size/comp_size:.1f}x")
    else:
        print(f"⚠️ Compression failed, using uncompressed model")
        return input_model
    
    return output_model


def main():
    """主函数"""
    
    print("="*60)
    print("DeePMD-kit ML势训练示例")
    print("="*60)
    
    # 步骤1: 准备数据 (假设已有VASP计算结果)
    print("\n[1/4] 准备训练数据...")
    # train_dir, valid_dir = prepare_training_data(
    #     vasp_dirs=["./vasp_run1", "./vasp_run2"],
    #     output_dir="./training_data"
    # )
    
    # 步骤2: 生成输入文件
    print("[2/4] 生成输入文件...")
    input_file = generate_input(
        type_map=["Li", "P", "S"],
        output_file="input.json"
    )
    
    # 步骤3: 训练
    print("[3/4] 训练模型...")
    # train_model(input_file)
    print("(跳过训练步骤 - 请运行: dp train input.json)")
    
    # 步骤4: 冻结和压缩
    print("[4/4] 冻结和压缩模型...")
    # freeze_model()
    # compress_model()
    print("(跳过冻结步骤 - 请运行: dp freeze -o graph.pb)")
    
    print("\n" + "="*60)
    print("示例完成!")
    print("="*60)
    print("\n下一步:")
    print("1. 运行: dp train input.json")
    print("2. 运行: dp freeze -o graph.pb")
    print("3. 运行: dp compress -i graph.pb -o graph-compress.pb")


if __name__ == "__main__":
    main()
