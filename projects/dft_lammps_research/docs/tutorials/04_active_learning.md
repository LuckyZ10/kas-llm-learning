# 04 - 主动学习实战 | Active Learning in Practice

> **学习目标**: 掌握主动学习工作流，实现自动化模型迭代优化  
> **Learning Goal**: Master active learning workflows for automated model refinement

---

## 📋 目录 | Table of Contents

1. [理论基础 | Theoretical Background](#1-理论基础--theoretical-background)
2. [不确定性量化 | Uncertainty Quantification](#2-不确定性量化--uncertainty-quantification)
3. [探索策略 | Exploration Strategies](#3-探索策略--exploration-strategies)
4. [完整工作流 | Complete Workflow](#4-完整工作流--complete-workflow)
5. [DP-GEN使用 | DP-GEN Usage](#5-dp-gen使用--dp-gen-usage)
6. [高级技巧 | Advanced Techniques](#6-高级技巧--advanced-techniques)
7. [练习题 | Exercises](#7-练习题--exercises)

---

## 1. 理论基础 | Theoretical Background

### 1.1 主动学习循环 | Active Learning Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                    主动学习循环 (Active Learning Loop)            │
└─────────────────────────────────────────────────────────────────┘

    ┌─────────────┐         不确定性高          ┌─────────────┐
    │   初始训练   │──────────────────────────▶│   DFT计算   │
    │   Initial   │      High uncertainty      │    Label    │
    │   Training  │                            │             │
    └─────────────┘                            └─────────────┘
           ▲                                         │
           │                                         │
           │            ┌─────────────┐              │
           └────────────│   重新训练   │◀─────────────┘
              Retrain   │   Retrain   │    标注数据
                        └─────────────┘   Labeled data
                              ▲
                              │
                        ┌─────────────┐
                        │   探索阶段   │
                        │   Explore   │
                        └─────────────┘
                              │
                              ▼
                        ┌─────────────┐
                        │   ML-MD探索  │
                        │   ML-MD Run │
                        └─────────────┘
```

### 1.2 为什么需要主动学习？| Why Active Learning?

| 问题 | 传统方法 | 主动学习 |
|------|---------|----------|
| 数据效率 | 需要大量DFT计算 | 智能选择关键结构 |
| 覆盖范围 | 容易遗漏关键区域 | 自动探索构型空间 |
| 外推能力 | 在新环境失效 | 持续扩展适用范围 |
| 人力成本 | 需人工筛选结构 | 自动化迭代 |

---

## 2. 不确定性量化 | Uncertainty Quantification

### 2.1 模型偏差方法 | Model Deviation Method

使用模型集成(Ensemble)计算预测不确定性：

```python
"""
模型偏差计算 | Model Deviation Calculation
"""
import numpy as np
from deepmd.infer import DeepPot
from typing import List, Dict


class UncertaintyQuantifier:
    """不确定性量化器"""
    
    def __init__(self, model_paths: List[str]):
        """
        Args:
            model_paths: 模型路径列表 (通常4个模型)
        """
        self.models = []
        for path in model_paths:
            self.models.append(DeepPot(path))
        print(f"Loaded {len(self.models)} models for uncertainty quantification")
    
    def compute_model_deviation(self, atoms) -> Dict[str, float]:
        """
        计算模型偏差 (Model Deviation)
        
        Returns:
            dict with keys: 'forces', 'energy', 'virial', 'max_force_devi'
        """
        # 准备输入
        coord = atoms.get_positions().reshape(1, -1)
        cell = atoms.get_cell().array.reshape(1, -1)
        atype = np.array([self._get_type_index(s) 
                         for s in atoms.get_chemical_symbols()])
        
        # 收集所有模型的预测
        energies = []
        forces_list = []
        virials = []
        
        for model in self.models:
            e, f, v = model.eval(coord, cell, atype)
            energies.append(e[0][0])
            forces_list.append(f[0])
            virials.append(v[0][0] if v is not None else 0.0)
        
        # 计算偏差
        energies = np.array(energies)
        forces_array = np.array(forces_list)  # (n_models, n_atoms, 3)
        n_atoms = len(atoms)
        
        # 力偏差: 每个原子的力向量的标准差
        force_stds = np.std(forces_array, axis=0)  # (n_atoms, 3)
        force_devi_per_atom = np.linalg.norm(force_stds, axis=1)  # (n_atoms,)
        max_force_devi = np.max(force_devi_per_atom)
        
        # 能量偏差: 能量标准差 / 原子数
        energy_devi = np.std(energies) / n_atoms
        
        # 维里偏差
        virial_devi = np.std(virials) / n_atoms if len(virials) > 1 else 0.0
        
        return {
            'forces': max_force_devi,
            'energy': energy_devi,
            'virial': virial_devi,
            'max_force_devi': max_force_devi,
            'force_devi_per_atom': force_devi_per_atom,
        }
    
    def _get_type_index(self, symbol: str) -> int:
        """获取元素类型索引"""
        type_map = self.models[0].get_type_map()
        return type_map.index(symbol) if symbol in type_map else 0
    
    def select_candidates(self, 
                         structures: List,
                         f_trust_lo: float = 0.05,
                         f_trust_hi: float = 0.15) -> List:
        """
        选择候选结构进行DFT计算
        
        选择标准: θ_lo ≤ ε_F,max < θ_hi
        即模型不确定性在中等范围的结构
        
        Args:
            structures: 待筛选的结构列表
            f_trust_lo: 力偏差下限 (eV/Å)
            f_trust_hi: 力偏差上限 (eV/Å)
            
        Returns:
            候选结构列表
        """
        candidates = []
        stats = {'accurate': 0, 'candidate': 0, 'failed': 0}
        
        for atoms in structures:
            devi = self.compute_model_deviation(atoms)
            max_devi = devi['max_force_devi']
            
            if max_devi < f_trust_lo:
                stats['accurate'] += 1  # 模型确定，无需DFT
            elif max_devi >= f_trust_hi:
                stats['failed'] += 1    # 模型失效，可能需要排除
            else:
                stats['candidate'] += 1  # 候选结构
                candidates.append(atoms)
        
        print(f"Structure selection stats:")
        print(f"  Accurate:  {stats['accurate']} (模型确定)")
        print(f"  Candidate: {stats['candidate']} (需要DFT)")
        print(f"  Failed:    {stats['failed']} (可能无效)")
        
        return candidates
```

### 2.2 自适应阈值 | Adaptive Thresholds

```python
def adjust_thresholds(self, 
                     candidate_ratio: float,
                     target_ratio: float = 0.1) -> tuple:
    """
    自适应调整阈值
    
    Args:
        candidate_ratio: 当前候选结构比例
        target_ratio: 目标比例 (默认10%)
        
    Returns:
        (new_lo, new_hi)
    """
    current_lo = self.f_trust_lo
    current_hi = self.f_trust_hi
    
    if candidate_ratio > target_ratio * 1.5:
        # 候选太多，提高阈值
        new_lo = current_lo * 1.2
        new_hi = current_hi * 1.2
        print(f"Too many candidates ({candidate_ratio:.2%}), raising thresholds")
    elif candidate_ratio < target_ratio * 0.5:
        # 候选太少，降低阈值
        new_lo = current_lo * 0.8
        new_hi = max(current_hi * 0.9, new_lo + 0.05)
        print(f"Too few candidates ({candidate_ratio:.2%}), lowering thresholds")
    else:
        new_lo, new_hi = current_lo, current_hi
    
    return new_lo, new_hi
```

---

## 3. 探索策略 | Exploration Strategies

### 3.1 温度扰动探索 | Temperature Perturbation

```python
"""
结构探索器 | Structure Explorer
"""
from ase import Atoms
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
import numpy as np


class StructureExplorer:
    """结构空间探索器"""
    
    def __init__(self, calculator):
        """
        Args:
            calculator: ASE calculator (ML势)
        """
        self.calculator = calculator
    
    def explore_temperature(self,
                           base_structure: Atoms,
                           temperatures: List[float] = None,
                           n_steps: int = 50000,
                           sample_freq: int = 100) -> List[Atoms]:
        """
        温度扰动探索
        
        在不同温度下运行MD，采样构型空间
        
        Args:
            base_structure: 基础结构
            temperatures: 温度列表 (K)
            n_steps: MD总步数
            sample_freq: 采样频率
            
        Returns:
            采样结构列表
        """
        if temperatures is None:
            temperatures = [300, 500, 700, 900, 1100, 1300]
        
        all_structures = []
        
        for T in temperatures:
            print(f"Exploring at T={T}K...")
            
            atoms = base_structure.copy()
            atoms.calc = self.calculator
            
            # 初始化速度
            MaxwellBoltzmannDistribution(atoms, temperature_K=T)
            
            # NVT MD
            dyn = Langevin(atoms, 
                          timestep=1.0*units.fs,
                          temperature_K=T,
                          friction=0.01)
            
            # 采样
            structures = []
            def sample():
                structures.append(atoms.copy())
            
            dyn.attach(sample, interval=sample_freq)
            dyn.run(n_steps)
            
            all_structures.extend(structures)
            print(f"  Collected {len(structures)} structures")
        
        return all_structures
    
    def explore_pressure(self,
                        base_structure: Atoms,
                        pressures: List[float] = None,
                        temperature: float = 300) -> List[Atoms]:
        """
        压力扰动探索
        
        Args:
            pressures: 压力列表 (GPa)
            temperature: 温度 (K)
        """
        if pressures is None:
            pressures = [-5, 0, 5, 10, 20, 30, 50]  # -5GPa = 拉伸
        
        structures = []
        
        for P in pressures:
            print(f"Exploring at P={P}GPa...")
            
            atoms = base_structure.copy()
            atoms.calc = self.calculator
            
            MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
            
            # NPT MD
            dyn = NPTBerendsen(atoms,
                              timestep=1.0*units.fs,
                              temperature_K=temperature,
                              pressure_au=P * units.GPa,
                              taut=100*units.fs,
                              taup=1000*units.fs,
                              compressibility_au=4.57e-5/units.bar)
            
            # 平衡 + 采样
            dyn.run(10000)  # 平衡
            
            for _ in range(50):
                dyn.run(100)
                structures.append(atoms.copy())
        
        return structures
    
    def explore_deformation(self,
                           base_structure: Atoms,
                           strain_range: tuple = (-0.15, 0.15),
                           n_strains: int = 20) -> List[Atoms]:
        """
        结构变形探索
        
        应用各种应变模式
        """
        structures = []
        cell = base_structure.get_cell()
        
        strain_values = np.linspace(strain_range[0], strain_range[1], n_strains)
        
        for strain in strain_values:
            # 单轴应变 (x方向)
            atoms = base_structure.copy()
            deformation = np.eye(3)
            deformation[0, 0] = 1 + strain
            new_cell = cell @ deformation.T
            atoms.set_cell(new_cell, scale_atoms=True)
            structures.append(atoms)
            
            # 双轴应变
            atoms = base_structure.copy()
            deformation = np.eye(3)
            deformation[0, 0] = 1 + strain
            deformation[1, 1] = 1 + strain
            new_cell = cell @ deformation.T
            atoms.set_cell(new_cell, scale_atoms=True)
            structures.append(atoms)
            
            # 剪切应变
            atoms = base_structure.copy()
            deformation = np.eye(3)
            deformation[0, 1] = strain
            new_cell = cell @ deformation.T
            atoms.set_cell(new_cell, scale_atoms=True)
            structures.append(atoms)
        
        return structures
    
    def explore_surface(self,
                       bulk_structure: Atoms,
                       miller_indices: List[tuple] = None) -> List[Atoms]:
        """
        表面结构探索
        
        生成不同晶面的表面结构
        """
        from ase.build import surface
        
        if miller_indices is None:
            miller_indices = [(1,0,0), (1,1,0), (1,1,1), (2,1,0)]
        
        structures = []
        
        for hkl in miller_indices:
            try:
                slab = surface(bulk_structure, hkl, layers=6, vacuum=15)
                structures.append(slab)
                
                # 添加表面吸附位点变体
                for vacuum in [10, 15, 20]:
                    slab_v = surface(bulk_structure, hkl, layers=6, vacuum=vacuum)
                    structures.append(slab_v)
                    
            except Exception as e:
                print(f"Failed to create surface {hkl}: {e}")
        
        return structures
    
    def comprehensive_exploration(self, base_structure: Atoms) -> List[Atoms]:
        """
        综合探索策略
        
        结合多种探索方法
        """
        print("Starting comprehensive exploration...")
        
        structures = []
        
        # 1. 温度探索
        print("\n1. Temperature exploration")
        temp_structs = self.explore_temperature(base_structure)
        structures.extend(temp_structs)
        
        # 2. 压力探索
        print("\n2. Pressure exploration")
        press_structs = self.explore_pressure(base_structure)
        structures.extend(press_structs)
        
        # 3. 结构变形
        print("\n3. Structure deformation")
        deform_structs = self.explore_deformation(base_structure)
        structures.extend(deform_structs)
        
        # 去重 (简单的能量去重)
        structures = self._deduplicate(structures)
        
        print(f"\nTotal unique structures: {len(structures)}")
        return structures
    
    def _deduplicate(self, structures: List[Atoms], 
                    energy_tol: float = 0.01) -> List[Atoms]:
        """简单的去重"""
        if len(structures) <= 1:
            return structures
        
        unique = [structures[0]]
        
        for s in structures[1:]:
            is_dup = False
            for u in unique:
                # 比较元素组成
                if set(s.get_chemical_symbols()) != set(u.get_chemical_symbols()):
                    continue
                # 可以添加更多比较
                is_dup = True
                break
            
            if not is_dup:
                unique.append(s)
        
        return unique
```

---

## 4. 完整工作流 | Complete Workflow

### 4.1 主动学习主类 | Active Learning Main Class

```python
"""
主动学习工作流主类 | Active Learning Workflow Main Class
"""
import os
import json
import shutil
from pathlib import Path
from typing import List, Dict
from datetime import datetime

from ase import Atoms
from ase.io import read, write
from deepmd.calculator import DP


class ActiveLearningWorkflow:
    """主动学习工作流"""
    
    def __init__(self,
                 work_dir: str = "./active_learning",
                 f_trust_lo: float = 0.05,
                 f_trust_hi: float = 0.15,
                 max_iterations: int = 20):
        """
        Args:
            work_dir: 工作目录
            f_trust_lo: 力偏差下限 (eV/Å)
            f_trust_hi: 力偏差上限 (eV/Å)
            max_iterations: 最大迭代次数
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        self.f_trust_lo = f_trust_lo
        self.f_trust_hi = f_trust_hi
        self.max_iterations = max_iterations
        
        self.iteration = 0
        self.model_paths = []
        
        # 初始化组件
        self.quantifier = None
        self.explorer = None
        
        # 历史记录
        self.history = []
    
    def initialize(self,
                  initial_structures: List[Atoms],
                  dft_calculator):
        """
        初始化主动学习工作流
        
        Args:
            initial_structures: 初始结构列表
            dft_calculator: DFT计算器
        """
        print("="*60)
        print("Initializing Active Learning Workflow")
        print("="*60)
        
        # 保存初始结构
        init_file = self.work_dir / "initial_structures.xyz"
        write(init_file, initial_structures)
        
        # 第一轮DFT计算
        print("\n1. Labeling initial structures...")
        labeled_data = self._run_dft(initial_structures, iteration=0)
        
        # 初始训练
        print("\n2. Initial training...")
        self._train_models(labeled_data, iteration=0)
        
        print("\n✓ Initialization complete!")
    
    def _run_dft(self, structures: List[Atoms], iteration: int) -> str:
        """运行DFT计算并返回数据目录"""
        iter_dir = self.work_dir / f"iter_{iteration:02d}"
        dft_dir = iter_dir / "dft_calculations"
        dft_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Running DFT for {len(structures)} structures...")
        
        # 这里调用DFT计算
        # 简化版本，实际应提交到队列
        for i, atoms in enumerate(structures):
            struct_dir = dft_dir / f"struct_{i:04d}"
            struct_dir.mkdir(exist_ok=True)
            write(struct_dir / "POSCAR", atoms)
            # 提交DFT计算...
        
        # 收集结果
        data_dir = iter_dir / "labeled_data"
        self._collect_dft_results(dft_dir, data_dir)
        
        return str(data_dir)
    
    def _collect_dft_results(self, dft_dir: Path, output_dir: Path):
        """收集DFT结果并转换为DeePMD格式"""
        import dpdata
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        systems = []
        for struct_dir in dft_dir.glob("struct_*"):
            outcar = struct_dir / "OUTCAR"
            if outcar.exists():
                try:
                    system = dpdata.LabeledSystem(str(outcar), fmt='vasp/outcar')
                    systems.append(system)
                except:
                    pass
        
        if systems:
            multi_systems = dpdata.MultiSystems(*systems)
            for name, system in multi_systems.systems.items():
                system.to_deepmd_npy(str(output_dir / name))
    
    def _train_models(self, data_dir: str, iteration: int):
        """训练模型"""
        from code_templates.ml_potential_training import DeepMDTrainer, DeepMDConfig
        
        iter_dir = self.work_dir / f"iter_{iteration:02d}"
        model_dir = iter_dir / "models"
        
        config = DeepMDConfig(
            type_map=["Li", "P", "S"],  # 根据实际系统修改
            training_data=str(data_dir),
            output_dir=str(model_dir)
        )
        
        trainer = DeepMDTrainer(config)
        
        # 训练4个模型
        self.model_paths = trainer.train_ensemble(n_models=4)
        
        # 更新不确定性量化器
        self.quantifier = UncertaintyQuantifier(self.model_paths)
        
        # 更新探索器
        self.explorer = StructureExplorer(DP(model=self.model_paths[0]))
    
    def run_iteration(self) -> bool:
        """
        运行一个主动学习迭代
        
        Returns:
            是否收敛
        """
        self.iteration += 1
        
        print(f"\n{'='*60}")
        print(f"Active Learning Iteration {self.iteration}")
        print(f"{'='*60}")
        
        # 1. 探索
        print("\n1. Exploration phase...")
        base_structures = self._load_base_structures()
        explored_structs = self.explorer.comprehensive_exploration(base_structures[0])
        
        # 2. 选择候选
        print("\n2. Selecting candidates...")
        candidates = self.quantifier.select_candidates(
            explored_structs,
            f_trust_lo=self.f_trust_lo,
            f_trust_hi=self.f_trust_hi
        )
        
        if len(candidates) == 0:
            print("✓ No uncertain structures found. Converged!")
            return True
        
        print(f"  Selected {len(candidates)} candidates for DFT")
        
        # 3. DFT标注
        print("\n3. DFT labeling...")
        labeled_data = self._run_dft(candidates, iteration=self.iteration)
        
        # 4. 重新训练
        print("\n4. Retraining models...")
        self._merge_data(self.iteration)
        self._train_models(self._get_all_data_dir(), iteration=self.iteration)
        
        # 5. 评估
        print("\n5. Evaluating models...")
        metrics = self._evaluate_models()
        
        # 记录历史
        self.history.append({
            'iteration': self.iteration,
            'n_candidates': len(candidates),
            'metrics': metrics
        })
        
        # 检查收敛
        if self._check_convergence():
            print("\n✓ Convergence achieved!")
            return True
        
        return False
    
    def run(self, max_iterations: int = None):
        """
        运行完整主动学习循环
        """
        max_iter = max_iterations or self.max_iterations
        
        for i in range(max_iter):
            converged = self.run_iteration()
            if converged:
                break
        
        # 保存最终模型
        self._save_final_model()
        
        print(f"\n{'='*60}")
        print("Active Learning Complete!")
        print(f"{'='*60}")
        print(f"Total iterations: {self.iteration}")
        print(f"Final model: {self.model_paths[0]}")
    
    def _check_convergence(self) -> bool:
        """检查收敛性"""
        if len(self.history) < 3:
            return False
        
        # 检查最近3轮是否有候选结构
        recent = self.history[-3:]
        avg_candidates = sum(h['n_candidates'] for h in recent) / 3
        
        # 如果平均候选数 < 5，认为已收敛
        return avg_candidates < 5
    
    def _save_final_model(self):
        """保存最终模型"""
        final_dir = self.work_dir / "final_model"
        final_dir.mkdir(exist_ok=True)
        
        for i, model_path in enumerate(self.model_paths):
            shutil.copy(model_path, final_dir / f"graph_{i}.pb")
        
        # 保存历史
        with open(final_dir / "history.json", 'w') as f:
            json.dump(self.history, f, indent=2)
```

---

## 5. DP-GEN使用 | DP-GEN Usage

### 5.1 DP-GEN简介 | DP-GEN Introduction

DP-GEN是官方主动学习工作流工具：
- 自动化探索-标注-训练循环
- 支持多种探索策略
- 集成作业调度系统

### 5.2 DP-GEN配置 | DP-GEN Configuration

```json
{
  "type_map": ["Li", "P", "S"],
  "mass_map": [6.941, 30.974, 32.065],
  
  "init_data_prefix": "./data",
  "init_data_sys": ["init_data"],
  
  "sys_configs": [
    ["./configs/Li3PS4.vasp"],
    ["./configs/Li2S.vasp"],
    ["./configs/P2S5.vasp"]
  ],
  
  "numb_models": 4,
  
  "default_training_param": {
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
      "stop_lr": 3.51e-8,
      "decay_steps": 5000
    },
    "loss": {
      "start_pref_e": 0.02,
      "limit_pref_e": 1,
      "start_pref_f": 1000,
      "limit_pref_f": 1
    },
    "training": {
      "stop_batch": 1000000,
      "batch_size": "auto",
      "disp_file": "lcurve.out",
      "save_freq": 10000
    }
  },
  
  "model_devi_dt": 0.002,
  "model_devi_skip": 0,
  "model_devi_f_trust_lo": 0.05,
  "model_devi_f_trust_hi": 0.15,
  "model_devi_e_trust_lo": 0.05,
  "model_devi_e_trust_hi": 0.15,
  
  "model_devi_jobs": [
    {
      "sys_idx": [0],
      "temps": [50, 100, 200, 300, 500, 700],
      "press": [1.0, 10.0, 50.0],
      "trj_freq": 10,
      "nsteps": 1000
    }
  ],
  
  "fp_style": "vasp",
  "fp_task_max": 20,
  "fp_task_min": 5,
  "fp_pp_path": "./potentials",
  "fp_pp_files": {
    "Li": "Li.pbe",
    "P": "P.pbe",
    "S": "S.pbe"
  },
  "fp_incar": "./INCAR"
}
```

### 5.3 运行DP-GEN | Running DP-GEN

```bash
# 生成参数文件
dpgen init reaction param.json

# 运行主动学习
dpgen run param.json machine.json

# 自动提交作业
dpgen auto param.json machine.json
```

---

## 6. 高级技巧 | Advanced Techniques

### 6.1 并行探索 | Parallel Exploration

```python
from concurrent.futures import ProcessPoolExecutor

def parallel_explore(structures, calculator, n_workers=4):
    """并行结构探索"""
    
    def explore_single(structure):
        explorer = StructureExplorer(calculator)
        return explorer.explore_temperature(structure, temperatures=[300, 500])
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(explore_single, structures))
    
    # 合并结果
    all_structures = []
    for r in results:
        all_structures.extend(r)
    
    return all_structures
```

### 6.2 阶段式训练 | Staged Training

```python
def staged_training(self, stages):
    """
    分阶段训练
    
    阶段1: 初步收敛 (快速)
    阶段2: 精细优化 (中等)
    阶段3: 最终收敛 (慢速)
    """
    for i, stage in enumerate(stages):
        print(f"\n{'='*60}")
        print(f"Training Stage {i+1}/{len(stages)}")
        print(f"{'='*60}")
        
        # 更新训练参数
        config = DeepMDConfig(
            numb_steps=stage['steps'],
            start_lr=stage['lr'],
            # ...
        )
        
        # 训练
        trainer = DeepMDTrainer(config)
        trainer.train(restart=(i > 0))
```

---

## 7. 练习题 | Exercises

### 练习 1: 自定义探索策略

```python
# 实现针对电池材料的特殊探索策略
def explore_battery_materials(self, structure):
    """
    针对电池材料的探索：
    1. Li空位扩散路径
    2. 不同Li浓度
    3. 充放电状态
    """
    structures = []
    
    # Li空位探索
    n_li = sum(1 for s in structure.get_chemical_symbols() if s == 'Li')
    for vacancy_ratio in [0.0, 0.1, 0.2, 0.3]:
        n_vacancies = int(n_li * vacancy_ratio)
        # 创建空位...
    
    return structures
```

### 练习 2: 收敛分析

```python
# 分析主动学习收敛过程
def analyze_convergence(history):
    """分析主动学习收敛性"""
    import matplotlib.pyplot as plt
    
    iterations = [h['iteration'] for h in history]
    candidates = [h['n_candidates'] for h in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, candidates, 'bo-')
    plt.xlabel('Iteration')
    plt.ylabel('Number of Candidates')
    plt.title('Active Learning Convergence')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('convergence.png')
```

---

**下一步**: [05 - 高通量筛选案例](05_high_throughput.md)
