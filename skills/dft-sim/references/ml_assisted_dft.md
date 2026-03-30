# 机器学习辅助DFT计算

## 概述

机器学习(ML)与DFT的结合正在改变材料计算的方式。从结构预测到性质预测，从势函数训练到主动学习采样，ML大幅提升了计算效率和探索空间。

**本模块涵盖**:
- 结构预测 (M3GNet/CHGNet)
- 性质预测 (CGCNN/MegNet)
- 势函数训练 (DeepMD/NequIP/MACE)
- 主动学习工作流

---

## 1. 结构预测

### 1.1 基于图神经网络的晶体结构预测

**M3GNet** (Materials 3D Graph Network) 是材料结构预测的强大工具。

```python
from pymatgen.core import Structure, Lattice
from m3gnet.models import M3GNet, Potential
import numpy as np

def predict_crystal_structure(composition, spacegroup=None):
    """
    使用M3GNet预测晶体结构
    
    Parameters:
    -----------
    composition : str
        化学式, 如 "LiCoO2"
    spacegroup : int, optional
        限制空间群
    """
    
    # 加载预训练模型
    potential = Potential(M3GNet.load())
    
    # 生成候选结构 (可使用pyxtal或USPEX生成初始猜测)
    candidate_structures = generate_candidate_structures(composition, spacegroup)
    
    results = []
    
    for struct in candidate_structures:
        # M3GNet快速能量评估
        energy = potential.get_potential_energy(struct)
        
        # 结构优化 (可选)
        from ase.optimize import BFGS
        atoms = struct.to_ase_atoms()
        atoms.calc = potential
        opt = BFGS(atoms)
        opt.run(fmax=0.05)
        
        relaxed_energy = potential.get_potential_energy(atoms)
        
        results.append({
            'structure': atoms,
            'energy': relaxed_energy,
            'initial_energy': energy
        })
    
    # 按能量排序
    results.sort(key=lambda x: x['energy'])
    
    print(f"预测到 {len(results)} 个候选结构")
    print(f"最稳定结构能量: {results[0]['energy']:.3f} eV/atom")
    
    return results

# 示例: 预测LiCoO₂结构
def example_lco_prediction():
    results = predict_crystal_structure("LiCoO2", spacegroup=166)
    
    # 与实验结构对比
    best_struct = results[0]['structure']
    
    print("\n预测结构参数:")
    print(f"  a = {best_struct.cell[0, 0]:.3f} Å")
    print(f"  c = {best_struct.cell[2, 2]:.3f} Å")
    print(f"  层间距 = {best_struct.cell[2, 2]/3:.3f} Å")
    
    return best_struct
```

### 1.2 替代传统结构搜索方法

| 方法 | 速度 | 精度 | 适用场景 |
|------|------|------|----------|
| USPEX | 慢 | 高 | 小体系，高精度需求 |
| CALYPSO | 慢 | 高 | 高压相预测 |
| AIRSS | 中 | 高 | 无机晶体 |
| **M3GNet** | **极快** | **中** | **快速筛选，大体系** |
| **CHGNet** | **快** | **中高** | **通用材料预测** |

### 1.3 CHGNet结构优化

CHGNet (Crystal Hamiltonian Graph Neural Network) 是另一个强大的预训练模型。

```python
from chgnet.model import CHGNet
from pymatgen.core import Structure

def chgnet_structure_relaxation(structure, fmax=0.05, max_steps=500):
    """
    使用CHGNet进行结构弛豫
    
    优势:
    - 速度比DFT快1000倍
    - 对磁性和电荷状态敏感
    - 支持大体系 (~1000原子)
    """
    
    # 加载预训练模型
    chgnet = CHGNet.load()
    
    # ASE接口
    from ase import Atoms
    from ase.optimize import FIRE
    
    atoms = structure.to_ase_atoms()
    atoms.calc = chgnet
    
    # 优化
    optimizer = FIRE(atoms)
    optimizer.run(fmax=fmax, steps=max_steps)
    
    # 获取优化后结构
    relaxed_structure = Structure(
        lattice=atoms.get_cell(),
        species=atoms.get_chemical_symbols(),
        coords=atoms.get_positions(),
        coords_are_cartesian=True
    )
    
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    
    print(f"优化完成:")
    print(f"  最终能量: {energy:.3f} eV")
    print(f"  最大力: {np.max(np.abs(forces)):.4f} eV/Å")
    
    return relaxed_structure, energy

# 批量筛选候选结构
def batch_screening(candidate_files):
    """批量筛选大量候选结构"""
    
    chgnet = CHGNet.load()
    energies = []
    
    for file in candidate_files:
        struct = Structure.from_file(file)
        
        # 快速单点能计算 (无优化)
        prediction = chgnet.predict_structure(struct)
        energy = prediction['energy']
        
        energies.append((file, energy))
    
    # 排序
    energies.sort(key=lambda x: x[1])
    
    print("前10个最稳定结构:")
    for i, (f, e) in enumerate(energies[:10]):
        print(f"  {i+1}. {f}: {e:.3f} eV/atom")
    
    return energies
```

---

## 2. 性质预测

### 2.1 CGCNN晶体图卷积网络

CGCNN (Crystal Graph Convolutional Neural Network) 用于直接从晶体结构预测性质。

```python
import torch
from cgcnn.data import CIFData, collate_pool
from cgcnn.model import CrystalGraphConvNet

def train_cgcnn_property_predictor(train_cifs, train_targets, property_name='band_gap'):
    """
    训练CGCNN性质预测模型
    
    Parameters:
    -----------
    train_cifs : list
        CIF文件路径列表
    train_targets : array
        目标性质值 (如带隙)
    """
    
    # 数据集
    dataset = CIFData(train_cifs, train_targets)
    
    # 数据加载器
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True,
        collate_fn=collate_pool
    )
    
    # 模型
    model = CrystalGraphConvNet(
        orig_atom_fea_len=92,
        nbr_fea_len=41,
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128,
        n_h=1,
        classification=False
    )
    
    # 训练
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(100):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            atom_fea, nbr_fea, nbr_idx, targets = batch
            
            optimizer.zero_grad()
            predictions = model(atom_fea, nbr_fea, nbr_idx, [])
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss/len(train_loader):.4f}")
    
    return model

def predict_properties(model, cif_files):
    """使用训练好的模型预测性质"""
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for cif in cif_files:
            struct = Structure.from_file(cif)
            # 转换为模型输入
            input_data = structure_to_cgcnn_input(struct)
            pred = model(*input_data)
            predictions.append(pred.item())
    
    return predictions
```

### 2.2 性质预测应用

```python
def materials_screening_workflow():
    """
    材料筛选工作流
    
    1. 使用ML模型快速筛选大量候选材料
    2. 选择Top候选进行DFT验证
    3. 反馈优化ML模型
    """
    
    # 候选材料库
    candidate_materials = load_materials_database()
    
    # Phase 1: ML快速筛选
    print("Phase 1: ML screening...")
    ml_predictions = {}
    
    for material in candidate_materials:
        # 使用CHGNet预测能量
        energy = chgnet_predict_energy(material)
        
        # 使用CGCNN预测带隙
        band_gap = cgcnn_predict_bandgap(material)
        
        # 筛选条件
        if energy < threshold_energy and 0.9 < band_gap < 2.0:
            ml_predictions[material.id] = {
                'energy': energy,
                'band_gap': band_gap
            }
    
    print(f"ML筛选: {len(ml_predictions)}/{len(candidate_materials)} 通过")
    
    # Phase 2: DFT验证
    print("\nPhase 2: DFT validation...")
    dft_results = {}
    
    for mat_id in list(ml_predictions.keys())[:50]:  # Top 50
        material = get_material(mat_id)
        
        # DFT计算
        dft_energy = vasp_relax(material)
        dft_gap = vasp_bandgap(material)
        
        dft_results[mat_id] = {
            'energy': dft_energy,
            'band_gap': dft_gap
        }
    
    # 计算ML误差
    ml_errors = []
    for mat_id in dft_results:
        ml_gap = ml_predictions[mat_id]['band_gap']
        dft_gap = dft_results[mat_id]['band_gap']
        ml_errors.append(abs(ml_gap - dft_gap))
    
    print(f"ML预测MAE: {np.mean(ml_errors):.2f} eV")
    
    return dft_results
```

---

## 3. 机器学习势函数

### 3.1 势函数对比

**2024-2025年重要进展**:

#### 通用ML势的PES软化问题 (Deng et al., 2025)
研究发现主流通用ML势(M3GNet, CHGNet, MACE-MP-0)存在系统性的**势能面软化**现象:

| 问题表现 | 原因 | 解决方案 |
|---------|------|---------|
| 表面/缺陷能量低估 | 训练数据偏向近平衡态 | 添加高能量OOD数据微调 |
| 离子迁移势垒低估 | PES曲率预测不足 | 增加模型容量或针对性训练 |
| 声子频率低估 | 训练数据偏差 | 包含振动模式数据 |
| 固溶体能量低估 | 组态空间采样不足 | 主动学习采样 |

**关键发现**:
- MACE (4.69M参数) 表现优于CHGNet和M3GNet
- 系统性误差可通过少量(~100)OOD数据高效修正
- 更大的模型容量有助于缓解软化问题

参考: B. Deng et al., *Nat. Commun.* (2025) - PES softening in uMLIPs

#### 微调基础模型时的灾难性遗忘 (2025)
对Fe系统的微调研究表明:
- **CHGNet** 和 **SevenNet-O**: 学习率≤0.0001时遗忘轻微
- **MACE**: 即使使用冻结层和数据重放，仍存在显著遗忘
- **建议**: 低学习率(≤0.0001)微调，谨慎选择架构

参考: J. Phase Equilibria Diffus. (2025)

### 3.2 MACE势训练

| 势函数 | 类型 | 精度 | 速度 | 推荐场景 | 最新版本 |
|--------|------|------|------|----------|----------|
| DeepMD | 深度神经网络 | 高 | 中 | 大尺度MD，常规研究 | v2.2+ |
| NequIP | E(3)等变网络 | 很高 | 中 | 高精度需求 | - |
| MACE | 等变消息传递 | 很高 | 快 | 原子模拟首选 | MACE-MP-0 |
| CHGNet | 预训练GNN | 中高 | 极快 | 快速筛选 | v0.2.0+ |
| M3GNet | 预训练GNN | 中 | 极快 | 结构预测 | - |
| **SevenNet-O** | **NequIP架构** | **很高** | **快** | **大规模并行MD** | **2024** |
| **MatterSim** | **Microsoft开发** | **高** | **快** | **通用材料** | **2024** |

### 3.2 MACE势训练

MACE (Message Passing Atomic Cluster Expansion) 是当前最先进的ML势之一。

```python
from mace.calculators import MACECalculator
import torch

def train_mace_potential(train_data, model_name='mace_run'):
    """
    训练MACE势函数
    
    Parameters:
    -----------
    train_data : list
        ASE Atoms对象列表 (含能量、力、应力)
    """
    
    # 保存训练数据为extxyz
    from ase.io import write
    write('training_data.xyz', train_data)
    
    # MACE训练命令
    mace_command = f"""
    python -m mace.run_train \\
        --name='{model_name}' \\
        --train_file='training_data.xyz' \\
        --valid_fraction=0.1 \\
        --test_file='test_data.xyz' \\
        --config_type_weights='{"Default": 1.0}' \\
        --E0s='{get_atomic_energies()}' \\
        --model='MACE' \\
        --hidden_irreps='128x0e + 128x1o' \\
        --r_max=5.0 \\
        --batch_size=5 \\
        --max_num_epochs=1000 \\
        --swa \\
        --start_swa=800 \\
        --ema \\
        --ema_decay=0.99 \\
        --amsgrad \\
        --device=cuda
    """
    
    import subprocess
    subprocess.run(mace_command, shell=True)
    
    print(f"MACE模型训练完成: {model_name}.model")
    
    return f'{model_name}.model'

def use_mace_calculator(model_path='mace_run.model'):
    """使用MACE势进行计算"""
    
    calc = MACECalculator(
        model_paths=model_path,
        device='cuda',
        default_dtype='float32'
    )
    
    # 示例: 结构优化
    from ase import Atoms
    from ase.optimize import BFGS
    
    atoms = Atoms(...)
    atoms.calc = calc
    
    opt = BFGS(atoms)
    opt.run(fmax=0.01)
    
    print(f"优化后能量: {atoms.get_potential_energy():.3f} eV")
    
    return atoms

# 大尺度MD模拟
def run_large_scale_md(mace_model, structure_file):
    """使用MACE势运行大尺度分子动力学"""
    
    from ase.io import read
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase import units
    
    # 加载结构
    atoms = read(structure_file)
    atoms.calc = MACECalculator(mace_model, device='cuda')
    
    # 设置MD
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    
    dyn = Langevin(
        atoms,
        timestep=1.0 * units.fs,
        temperature_K=300,
        friction=0.01
    )
    
    # 运行
    def print_status():
        print(f"Step {dyn.get_number_of_steps()}: "
              f"E = {atoms.get_potential_energy():.3f} eV, "
              f"T = {atoms.get_temperature():.1f} K")
    
    dyn.attach(print_status, interval=100)
    dyn.run(10000)  # 10 ps
    
    return atoms
```

### 3.3 主动学习工作流

主动学习是解决ML势泛化性的关键。

```python
def active_learning_workflow(initial_data, unlabeled_pool):
    """
    主动学习工作流
    
    迭代过程:
    1. 训练ML势
    2. 探索性MD模拟
    3. 识别不确定性高的构型
    4. DFT计算添加新数据
    5. 重复直到收敛
    """
    
    training_data = initial_data.copy()
    
    for iteration in range(10):  # 最大迭代次数
        print(f"\n=== Active Learning Iteration {iteration+1} ===")
        
        # 1. 训练ML势
        print("Training ML potential...")
        model = train_mace_potential(training_data, f'iter_{iteration}')
        
        # 2. 探索性MD
        print("Exploratory MD...")
        calc = MACECalculator(model, device='cuda')
        
        new_configs = []
        for temp in [300, 500, 800, 1200]:  # 不同温度探索
            configs = exploratory_md(calc, unlabeled_pool[0], temperature=temp)
            new_configs.extend(configs)
        
        # 3. 不确定性量化
        print("Uncertainty quantification...")
        uncertainties = []
        
        for config in new_configs:
            # committee disagreement或max force uncertainty
            uncertainty = estimate_uncertainty(model, config)
            uncertainties.append(uncertainty)
        
        # 选择高不确定性构型
        threshold = np.percentile(uncertainties, 95)
        selected_indices = np.where(uncertainties > threshold)[0]
        
        print(f"Selected {len(selected_indices)} configurations for DFT")
        
        if len(selected_indices) == 0:
            print("Convergence reached!")
            break
        
        # 4. DFT计算
        print("Running DFT calculations...")
        for idx in selected_indices[:20]:  # 每轮最多20个
            config = new_configs[idx]
            energy, forces = run_dft_single_point(config)
            
            config.info['energy'] = energy
            config.arrays['forces'] = forces
            training_data.append(config)
        
        print(f"Training set size: {len(training_data)}")
    
    return model, training_data

def estimate_uncertainty(model, atoms, n_ensemble=5):
    """
    估计构型不确定性
    
    方法1: Committee disagreement
    方法2: Max atomic force uncertainty
    """
    
    # 使用dropout或模型ensemble
    forces_list = []
    
    for _ in range(n_ensemble):
        forces = model.get_forces(atoms, dropout=True)
        forces_list.append(forces)
    
    forces_std = np.std(forces_list, axis=0)
    max_uncertainty = np.max(forces_std)
    
    return max_uncertainty
```

---

## 4. 高通量筛选工作流

### 4.1 完整自动化流程

```python
from dask import delayed, compute
from dask.distributed import Client

def high_throughput_screening_pipeline(materials_list):
    """
    高通量材料筛选管道
    
    使用Dask进行并行计算
    """
    
    client = Client(n_workers=8)
    
    results = []
    
    for material in materials_list:
        # Phase 1: 快速ML筛选
        ml_result = delayed(ml_screen)(material)
        
        # Phase 2: 条件DFT验证
        dft_result = delayed(conditional_dft)(ml_result)
        
        results.append(dft_result)
    
    # 执行
    computed_results = compute(*results)
    
    return computed_results

def ml_screen(material):
    """ML快速筛选"""
    
    # CHGNet能量
    chgnet = CHGNet.load()
    energy = chgnet_predict(chgnet, material)
    
    # 稳定性判断
    if energy > threshold:
        return {'pass': False, 'reason': 'unstable'}
    
    # CGCNN性质预测
    band_gap = cgcnn_predict(material)
    
    return {
        'pass': True,
        'material': material,
        'ml_energy': energy,
        'ml_gap': band_gap
    }

def conditional_dft(ml_result):
    """条件DFT计算"""
    
    if not ml_result['pass']:
        return ml_result
    
    # Top候选才进行DFT
    if ml_result['ml_gap'] < 0.5:  # 金属
        return ml_result
    
    # DFT验证
    material = ml_result['material']
    
    dft_energy = vasp_relax(material)
    dft_gap = vasp_bandgap(material)
    
    return {
        'material': material,
        'ml_energy': ml_result['ml_energy'],
        'dft_energy': dft_energy,
        'ml_gap': ml_result['ml_gap'],
        'dft_gap': dft_gap
    }
```

---

## 5. 工具与资源

### 5.1 推荐软件包

| 包名 | 功能 | 安装 |
|------|------|------|
| m3gnet | 结构预测 | `pip install m3gnet` |
| chgnet | 通用势函数 | `pip install chgnet` |
| mace | 高精度势 | `pip install mace-torch` |
| cgcnn | 性质预测 | GitHub |
| deepmd-kit | 深度势 | `conda install deepmd-kit` |
| nequip | 等变势 | `pip install nequip` |

### 5.2 预训练模型下载

```bash
# M3GNet
wget https://github.com/materialsvirtuallab/m3gnet/raw/main/pretrained/M3GNet-MP-2021.2.8-PES.zip

# CHGNet
# 自动下载于首次使用

# MACE-MP-0 (Materials Project大规模训练)
wget https://github.com/ACEsuit/mace/raw/main/mace_mp_0.model
```

---

## 6. 最佳实践

### 6.1 ML-DFT混合策略

```
┌─────────────────────────────────────────────┐
│         ML-DFT 分层计算策略                  │
├─────────────────────────────────────────────┤
│ Level 1: 预训练ML模型 (CHGNet/M3GNet)       │
│          - 快速筛选 10^5 候选               │
│          - 成本: ~1 CPU·hour                │
├─────────────────────────────────────────────┤
│ Level 2: 专用ML势 (MACE/NequIP)             │
│          - 训练于相关体系                   │
│          - 验证ML误差 < 5 meV/atom          │
│          - 成本: ~100 CPU·hours (训练)      │
├─────────────────────────────────────────────┤
│ Level 3: DFT验证                            │
│          - 关键性质的最终验证               │
│          - 成本: ~10^4 CPU·hours            │
└─────────────────────────────────────────────┘
```

### 6.2 误差控制

```python
def validate_ml_accuracy(ml_model, test_set):
    """
    验证ML模型精度
    
    关键指标:
    - 能量MAE < 10 meV/atom
    - 力MAE < 100 meV/Å
    - 应力MAE < 0.1 GPa
    """
    
    energy_errors = []
    force_errors = []
    
    for atoms in test_set:
        # ML预测
        ml_energy = ml_model.get_potential_energy(atoms)
        ml_forces = ml_model.get_forces(atoms)
        
        # DFT参考
        dft_energy = atoms.info['dft_energy']
        dft_forces = atoms.arrays['dft_forces']
        
        energy_errors.append(abs(ml_energy - dft_energy))
        force_errors.append(np.mean(np.abs(ml_forces - dft_forces)))
    
    print(f"能量MAE: {np.mean(energy_errors)*1000:.2f} meV/atom")
    print(f"力MAE: {np.mean(force_errors):.2f} meV/Å")
    
    # 判断是否满足精度要求
    if np.mean(energy_errors) < 0.01:  # 10 meV
        print("✓ ML模型满足精度要求")
        return True
    else:
        print("✗ 需要更多训练数据")
        return False
```

---

## 参考

1. C. Chen et al., *Nat. Comput. Sci.* 3, 838 (2023) - M3GNet
2. B. Deng et al., *Nat. Mach. Intell.* 5, 1031 (2023) - CHGNet
3. I. Batatia et al., *MRS Bulletin* 47, 995 (2022) - MACE
4. T. Xie et al., *Phys. Rev. Lett.* 120, 145301 (2018) - CGCNN
5. L. Zhang et al., *npj Comput. Mater.* 4, 1 (2018) - DeepMD
6. S. Batzner et al., *Nat. Commun.* 13, 2453 (2022) - NequIP
