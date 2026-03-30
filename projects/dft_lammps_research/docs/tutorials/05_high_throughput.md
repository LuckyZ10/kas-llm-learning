# 05 - 高通量筛选案例 | High-Throughput Screening

> **学习目标**: 掌握高通量计算工作流，实现大规模材料筛选与性能预测  
> **Learning Goal**: Master high-throughput workflows for large-scale material screening

---

## 📋 目录 | Table of Contents

1. [理论基础 | Theoretical Background](#1-理论基础--theoretical-background)
2. [数据库查询 | Database Querying](#2-数据库查询--database-querying)
3. [工作流管理 | Workflow Management](#3-工作流管理--workflow-management)
4. [性能预测 | Property Prediction](#4-性能预测--property-prediction)
5. [结果分析 | Results Analysis](#5-结果分析--results-analysis)
6. [案例分析 | Case Studies](#6-案例分析--case-studies)
7. [练习题 | Exercises](#7-练习题--exercises)

---

## 1. 理论基础 | Theoretical Background

### 1.1 高通量计算框架 | HT Framework

```
┌─────────────────────────────────────────────────────────────────┐
│                  高通量材料筛选框架                               │
│              High-Throughput Materials Screening                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   结构获取   │───▶│   DFT计算   │───▶│   性质分析   │         │
│  │  Structures │    │    DFT      │    │ Properties  │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│        ▲                   │                   │                │
│        │                   ▼                   ▼                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  Materials  │    │   ML加速    │    │   机器学习   │         │
│  │   Project   │    │  ML Speedup │    │    ML       │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                              │                                  │
│                              ▼                                  │
│                       ┌─────────────┐                          │
│                       │   数据库    │                          │
│                       │  Database   │                          │
│                       └─────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 筛选流程 | Screening Pipeline

| 阶段 | 描述 | 方法 |
|------|------|------|
| 1. 候选生成 | 基于化学空间/结构模板 | Pymatgen, ASE |
| 2. 预筛选 | 快速稳定性判断 | 经验规则, ML预测 |
| 3. DFT计算 | 高精度性质计算 | VASP, QE |
| 4. 性质分析 | 提取目标性质 | 后处理脚本 |
| 5. 候选排序 | 多目标优化 | Pareto分析 |

---

## 2. 数据库查询 | Database Querying

### 2.1 Materials Project接口 | Materials Project API

```python
"""
Materials Project高通量查询
High-throughput querying from Materials Project
"""
from mp_api.client import MPRester
from pymatgen.core import Composition, Element
import pandas as pd


class MaterialsProjectInterface:
    """Materials Project接口"""
    
    def __init__(self, api_key: str = None):
        """
        Args:
            api_key: MP API密钥 (可从环境变量MP_API_KEY获取)
        """
        self.mpr = MPRester(api_key)
    
    def query_by_chemsys(self, 
                        elements: list,
                        max_entries: int = 1000) -> pd.DataFrame:
        """
        按化学系统查询
        
        Args:
            elements: 元素列表，如 ['Li', 'S']
            max_entries: 最大返回条目数
            
        Returns:
            DataFrame包含材料信息
        """
        docs = self.mpr.summary.search(
            elements=elements,
            fields=[
                "material_id",
                "formula_pretty",
                "structure",
                "energy_per_atom",
                "band_gap",
                "efermi",
                "formation_energy_per_atom",
                "energy_above_hull",
                "symmetry"
            ],
            num_chunks=1,
            chunk_size=max_entries
        )
        
        # 转换为DataFrame
        data = []
        for doc in docs:
            data.append({
                'material_id': doc.material_id,
                'formula': doc.formula_pretty,
                'energy_per_atom': doc.energy_per_atom,
                'band_gap': doc.band_gap,
                'efermi': doc.efermi,
                'formation_energy': doc.formation_energy_per_atom,
                'ehull': doc.energy_above_hull,
                'symmetry': doc.symmetry.symbol if doc.symmetry else None,
            })
        
        df = pd.DataFrame(data)
        print(f"Retrieved {len(df)} materials")
        return df
    
    def query_battery_candidates(self,
                                 working_ion: str = "Li",
                                 max_entries: int = 500) -> pd.DataFrame:
        """
        查询电池材料候选
        
        筛选标准:
        - 包含工作离子 (Li/Na/Mg)
        - 一定电压范围
        - 结构稳定
        """
        # 获取该离子的所有材料
        docs = self.mpr.summary.search(
            elements=[working_ion],
            fields=[
                "material_id",
                "formula_pretty",
                "structure",
                "energy_per_atom",
                "band_gap",
                "formation_energy_per_atom",
                "energy_above_hull"
            ],
            num_chunks=1,
            chunk_size=max_entries
        )
        
        data = []
        for doc in docs:
            # 稳定性筛选
            if doc.energy_above_hull < 0.1:  # 近稳定结构
                data.append({
                    'material_id': doc.material_id,
                    'formula': doc.formula_pretty,
                    'band_gap': doc.band_gap,
                    'formation_energy': doc.formation_energy_per_atom,
                    'ehull': doc.energy_above_hull,
                })
        
        return pd.DataFrame(data)
    
    def query_catalysis_candidates(self,
                                   host_elements: list,
                                   adsorbates: list = None) -> pd.DataFrame:
        """
        查询催化材料候选 (表面材料)
        """
        # 筛选具有合适带隙的金属或半导体
        docs = self.mpr.summary.search(
            elements=host_elements,
            fields=[
                "material_id",
                "formula_pretty",
                "structure",
                "band_gap",
                "symmetry"
            ]
        )
        
        data = []
        for doc in docs:
            # 选择金属或窄带隙半导体
            if doc.band_gap is None or doc.band_gap < 2.0:
                data.append({
                    'material_id': doc.material_id,
                    'formula': doc.formula_pretty,
                    'band_gap': doc.band_gap,
                    'symmetry': doc.symmetry.symbol if doc.symmetry else None,
                })
        
        return pd.DataFrame(data)
    
    def download_structures(self, 
                          material_ids: list,
                          output_dir: str = "./structures") -> list:
        """
        下载结构文件
        
        Args:
            material_ids: 材料ID列表
            output_dir: 输出目录
            
        Returns:
            下载的文件路径列表
        """
        from pathlib import Path
        import os
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        files = []
        
        for mp_id in material_ids:
            try:
                structure = self.mpr.get_structure_by_material_id(mp_id)
                
                # 保存为POSCAR
                filename = f"{mp_id}_{structure.formula.replace(' ', '')}.vasp"
                filepath = os.path.join(output_dir, filename)
                structure.to(filename=filepath)
                files.append(filepath)
                
            except Exception as e:
                print(f"Failed to download {mp_id}: {e}")
        
        print(f"Downloaded {len(files)} structures")
        return files


# 使用示例
if __name__ == "__main__":
    mp = MaterialsProjectInterface()
    
    # 查询Li-S体系
    df_li_s = mp.query_by_chemsys(['Li', 'S'])
    print(df_li_s.head())
    
    # 筛选稳定结构
    stable = df_li_s[df_li_s['ehull'] < 0.05]
    print(f"\nStable materials: {len(stable)}")
    
    # 下载前10个稳定结构
    top10 = stable.head(10)['material_id'].tolist()
    mp.download_structures(top10, "./mp_structures")
```

### 2.2 结构生成器 | Structure Generator

```python
"""
结构生成器 - 创建多样化候选结构
Structure Generator - Create diverse candidate structures
"""
from pymatgen.core import Structure, Lattice, Composition
from pymatgen.analysis.structure_matcher import StructureMatcher
from ase.build import bulk
import numpy as np


class StructureGenerator:
    """结构生成器"""
    
    def __init__(self):
        self.matcher = StructureMatcher()
    
    def generate_substitutions(self,
                               base_structure: Structure,
                               site_indices: list,
               replacement_elements: list) -> list:
        """
        生成替代合金结构
        
        Args:
            base_structure: 基础结构
            site_indices: 可替代位置索引
            replacement_elements: 替代元素列表
            
        Returns:
            新结构列表
        """
        from pymatgen.transformations.standard_transformations import \
            SubstitutionTransformation
        
        structures = []
        
        for elem in replacement_elements:
            for idx in site_indices:
                # 获取该位置的原子类型
                old_elem = str(base_structure[idx].specie)
                
                # 创建替代变换
                trans = SubstitutionTransformation(
                    {old_elem: elem}
                )
                
                try:
                    new_struct = trans.apply_transformation(base_structure)
                    structures.append(new_struct)
                except:
                    pass
        
        # 去重
        return self._deduplicate(structures)
    
    def generate_vacancies(self,
                          structure: Structure,
                          vacancy_concentrations: list = [0.0, 0.0625, 0.125, 0.25]) -> list:
        """
        生成空位结构
        """
        from pymatgen.transformations.defect_transformations import \
            VacancyTransformation
        
        structures = []
        
        for conc in vacancy_concentrations:
            # 计算需要移除的原子数
            n_remove = int(len(structure) * conc)
            
            if n_remove == 0:
                structures.append(structure)
                continue
            
            # 生成空位 (简化版本)
            for i in range(min(5, len(structure))):  # 限制构型数
                new_struct = structure.copy()
                # 移除第i个原子
                if i < len(new_struct):
                    new_struct.remove_sites([i])
                    structures.append(new_struct)
        
        return structures
    
    def generate_intercalation_structures(self,
                                         host_structure: Structure,
                                         guest_element: str,
                                         guest_fractions: list) -> list:
        """
        生成插层结构 (用于电池材料)
        """
        structures = []
        
        for frac in guest_fractions:
            # 计算需要的客体原子数
            n_guest = int(len(host_structure) * frac)
            
            if n_guest == 0:
                structures.append(host_structure)
                continue
            
            # 在间隙位置插入原子
            # 这里简化处理，实际应检测间隙位置
            new_struct = host_structure.copy()
            
            # 添加客体原子到间隙 (简化: 添加到原点)
            for _ in range(n_guest):
                new_struct.append(guest_element, [0.5, 0.5, 0.5])
            
            structures.append(new_struct)
        
        return structures
    
    def generate_surface_slabs(self,
                              bulk_structure: Structure,
                              miller_indices: list = None,
                              min_slab_size: float = 10.0,
                              min_vacuum_size: float = 15.0) -> list:
        """
        生成表面slab结构
        """
        from pymatgen.core.surface import SlabGenerator
        
        if miller_indices is None:
            miller_indices = [(1,0,0), (1,1,0), (1,1,1)]
        
        slabs = []
        
        for hkl in miller_indices:
            try:
                gen = SlabGenerator(
                    bulk_structure,
                    hkl,
                    min_slab_size=min_slab_size,
                    min_vacuum_size=min_vacuum_size
                )
                
                slab_structs = gen.get_slabs()
                slabs.extend(slab_structs)
                
            except Exception as e:
                print(f"Failed to generate slab {hkl}: {e}")
        
        return slabs
    
    def _deduplicate(self, structures: list) -> list:
        """结构去重"""
        unique = []
        
        for s in structures:
            is_dup = False
            for u in unique:
                if self.matcher.fit(s, u):
                    is_dup = True
                    break
            
            if not is_dup:
                unique.append(s)
        
        return unique
```

---

## 3. 工作流管理 | Workflow Management

### 3.1 批量DFT计算 | Batch DFT Calculator

```python
"""
高通量DFT计算管理器
High-Throughput DFT Manager
"""
import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
import concurrent.futures


@dataclass
class DFTTask:
    """DFT计算任务"""
    task_id: str
    input_file: str
    work_dir: str
    status: str = "pending"  # pending, running, completed, failed
    priority: int = 1


class HighThroughputDFTManager:
    """高通量DFT管理器"""
    
    def __init__(self,
                 max_parallel: int = 4,
                 queue_system: str = None,
                 dft_code: str = "vasp"):
        """
        Args:
            max_parallel: 最大并行任务数
            queue_system: 队列系统 (slurm, pbs, None)
            dft_code: DFT代码 (vasp, espresso)
        """
        self.max_parallel = max_parallel
        self.queue_system = queue_system
        self.dft_code = dft_code
        
        self.tasks = []
        self.results = []
    
    def add_task(self, structure_file: str, work_dir: str) -> str:
        """添加计算任务"""
        task_id = f"task_{len(self.tasks):04d}"
        
        task = DFTTask(
            task_id=task_id,
            input_file=structure_file,
            work_dir=work_dir
        )
        
        self.tasks.append(task)
        return task_id
    
    def run_local(self, task: DFTTask) -> Dict:
        """本地运行任务"""
        from ase.io import read
        from ase.calculators.vasp import Vasp
        
        try:
            # 读取结构
            atoms = read(task.input_file)
            
            # 设置计算器
            calc = Vasp(
                xc='PBE',
                encut=520,
                kpts=(4, 4, 4),
                ibrion=2,
                isif=3,
                nsw=200,
                command=f"mpirun -np {self.max_parallel} vasp_std"
            )
            
            atoms.calc = calc
            
            # 运行计算
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            
            result = {
                'task_id': task.task_id,
                'status': 'success',
                'energy': float(energy),
                'forces': forces.tolist(),
                'work_dir': task.work_dir
            }
            
        except Exception as e:
            result = {
                'task_id': task.task_id,
                'status': 'failed',
                'error': str(e)
            }
        
        return result
    
    def submit_slurm(self, task: DFTTask, script_template: str = None):
        """提交到Slurm队列"""
        if script_template is None:
            script_template = """#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -N 1
#SBATCH --ntasks-per-node={ncores}
#SBATCH -t 24:00:00
#SBATCH -p normal

cd {work_dir}
mpirun -np {ncores} vasp_std
"""
        
        script = script_template.format(
            job_name=task.task_id,
            ncores=self.max_parallel,
            work_dir=task.work_dir
        )
        
        script_file = Path(task.work_dir) / "submit.sh"
        with open(script_file, 'w') as f:
            f.write(script)
        
        # 提交作业
        result = subprocess.run(
            ['sbatch', str(script_file)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            task.status = "running"
            return job_id
        else:
            task.status = "failed"
            return None
    
    def run_batch(self, parallel: bool = True) -> List[Dict]:
        """批量运行所有任务"""
        print(f"Running {len(self.tasks)} tasks...")
        
        if parallel:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_parallel
            ) as executor:
                futures = {
                    executor.submit(self.run_local, task): task 
                    for task in self.tasks
                }
                
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    self.results.append(result)
                    print(f"Task {result['task_id']}: {result['status']}")
        else:
            for task in self.tasks:
                result = self.run_local(task)
                self.results.append(result)
        
        return self.results
    
    def save_results(self, output_file: str = "results.json"):
        """保存结果"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {output_file}")
```

---

## 4. 性能预测 | Property Prediction

### 4.1 离子电导率预测 | Ionic Conductivity Prediction

```python
"""
离子电导率预测工作流
Ionic Conductivity Prediction Workflow
"""
from ase import Atoms
import numpy as np


class IonicConductivityPredictor:
    """离子电导率预测器"""
    
    def __init__(self, ml_calculator):
        """
        Args:
            ml_calculator: ML势计算器
        """
        self.calculator = ml_calculator
    
    def predict_from_diffusion(self,
                              structure: Atoms,
                              temperature: float = 300,
                              n_steps: int = 100000) -> Dict:
        """
        通过MD扩散计算预测电导率
        
        使用Nernst-Einstein方程:
        σ = n * q² * D / (k_B * T)
        """
        from ase.md.langevin import Langevin
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        
        atoms = structure.copy()
        atoms.calc = self.calculator
        
        # 运行MD
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
        
        dyn = Langevin(
            atoms,
            timestep=1.0,
            temperature_K=temperature,
            friction=0.01
        )
        
        # 记录轨迹
        positions_0 = atoms.get_positions().copy()
        positions_list = []
        
        def record():
            positions_list.append(atoms.get_positions().copy())
        
        dyn.attach(record, interval=100)
        dyn.run(n_steps)
        
        # 计算扩散系数
        D = self._calculate_diffusion(positions_0, positions_list, timestep=1.0)
        
        # 计算电导率 (Nernst-Einstein)
        n_li = sum(1 for s in atoms.get_chemical_symbols() if s == 'Li')
        volume_cm3 = atoms.get_volume() * 1e-24
        n_density = n_li / volume_cm3
        
        q = 1.602e-19  # C
        k_B = 1.381e-23  # J/K
        
        sigma = n_density * q**2 * D / (k_B * temperature)  # S/cm
        
        return {
            'diffusion_coefficient': D,
            'ionic_conductivity': sigma,
            'temperature': temperature
        }
    
    def _calculate_diffusion(self,
                            positions_0: np.ndarray,
                            positions_list: list,
                            timestep: float = 1.0) -> float:
        """计算扩散系数 (MSD方法)"""
        # 简化实现
        # 实际应考虑周期性边界条件
        msd_values = []
        
        for pos in positions_list:
            msd = np.mean(np.sum((pos - positions_0)**2, axis=1))
            msd_values.append(msd)
        
        # 线性拟合
        times = np.arange(len(msd_values)) * timestep * 100  # fs
        
        # 使用中间50%数据
        start = len(times) // 4
        end = 3 * len(times) // 4
        
        slope = np.polyfit(times[start:end], msd_values[start:end], 1)[0]
        
        # D = slope / 6 (3D)
        D = slope / 6 * 1e-16  # cm²/s
        
        return D
```

---

## 5. 结果分析 | Results Analysis

### 5.1 数据可视化 | Data Visualization

```python
"""
高通量筛选结果分析
HT Screening Results Analysis
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_screening_results(results_file: str):
    """分析筛选结果"""
    
    # 读取结果
    df = pd.read_json(results_file)
    
    # 1. 能量分布
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].hist(df['energy_per_atom'], bins=50, edgecolor='black')
    axes[0, 0].set_xlabel('Energy per Atom (eV)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Energy Distribution')
    
    # 2. 带隙分布
    if 'band_gap' in df.columns:
        axes[0, 1].hist(df['band_gap'], bins=50, edgecolor='black')
        axes[0, 1].set_xlabel('Band Gap (eV)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Band Gap Distribution')
    
    # 3. 稳定性图
    if 'ehull' in df.columns:
        axes[1, 0].scatter(df['energy_per_atom'], df['ehull'], alpha=0.5)
        axes[1, 0].set_xlabel('Energy per Atom (eV)')
        axes[1, 0].set_ylabel('Energy Above Hull (eV/atom)')
        axes[1, 0].set_title('Stability Map')
        axes[1, 0].axhline(y=0.05, color='r', linestyle='--', label='Stability threshold')
        axes[1, 0].legend()
    
    # 4. 多目标优化 (Pareto前沿)
    if 'band_gap' in df.columns and 'energy_per_atom' in df.columns:
        # 选择稳定材料
        stable = df[df['ehull'] < 0.1]
        
        axes[1, 1].scatter(stable['band_gap'], stable['energy_per_atom'], alpha=0.5)
        axes[1, 1].set_xlabel('Band Gap (eV)')
        axes[1, 1].set_ylabel('Energy per Atom (eV)')
        axes[1, 1].set_title('Pareto Front (Stable Materials)')
    
    plt.tight_layout()
    plt.savefig('screening_analysis.png', dpi=150)
    print("Analysis saved to screening_analysis.png")
    
    # 统计信息
    print("\n" + "="*60)
    print("Screening Statistics:")
    print("="*60)
    print(f"Total materials: {len(df)}")
    if 'ehull' in df.columns:
        print(f"Stable materials (ehull < 0.1): {len(df[df['ehull'] < 0.1])}")
    print(f"\nTop 5 lowest energy:")
    print(df.nsmallest(5, 'energy_per_atom')[['formula', 'energy_per_atom']])


def find_pareto_front(df: pd.DataFrame, 
                     objectives: list) -> pd.DataFrame:
    """
    寻找Pareto前沿
    
    多目标优化：找到不被其他解支配的解
    """
    pareto = []
    
    for i, row in df.iterrows():
        is_dominated = False
        
        for j, other in df.iterrows():
            if i == j:
                continue
            
            # 检查是否被支配
            dominates = all(
                other[obj] <= row[obj] if minimize 
                else other[obj] >= row[obj]
                for obj, minimize in objectives
            ) and any(
                other[obj] < row[obj] if minimize 
                else other[obj] > row[obj]
                for obj, minimize in objectives
            )
            
            if dominates:
                is_dominated = True
                break
        
        if not is_dominated:
            pareto.append(row)
    
    return pd.DataFrame(pareto)
```

---

## 6. 案例分析 | Case Studies

### 6.1 固态电解质筛选 | Solid Electrolyte Screening

```python
"""
固态电解质高通量筛选案例
Solid Electrolyte High-Throughput Screening Case Study
"""


def screen_solid_electrolytes():
    """
    固态电解质筛选工作流
    
    目标: 寻找高离子电导率、电化学稳定、机械性能好的材料
    """
    
    # 1. 候选材料生成
    mp = MaterialsProjectInterface()
    
    # 查询Li-containing硫化物、氧化物
    candidates = mp.query_by_chemsys(['Li', 'S'])
    candidates = pd.concat([
        candidates,
        mp.query_by_chemsys(['Li', 'O'])
    ])
    
    # 2. 预筛选
    # 稳定性筛选
    stable = candidates[candidates['ehull'] < 0.1]
    
    # 带隙筛选 (电化学稳定性)
    insulating = stable[stable['band_gap'] > 2.0]
    
    print(f"Pre-screened candidates: {len(insulating)}")
    
    # 3. 下载结构
    mp_ids = insulating['material_id'].tolist()[:50]  # 限制数量
    structure_files = mp.download_structures(mp_ids, "./se_structures")
    
    # 4. ML势预测
    from deepmd.calculator import DP
    calc = DP(model="./Li-S-graph.pb")
    
    predictor = IonicConductivityPredictor(calc)
    
    results = []
    for struct_file in structure_files:
        from ase.io import read
        atoms = read(struct_file)
        
        result = predictor.predict_from_diffusion(atoms)
        result['structure_file'] = struct_file
        results.append(result)
    
    # 5. 结果分析
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('ionic_conductivity', ascending=False)
    
    print("\nTop 10 candidates by ionic conductivity:")
    print(df_results.head(10))
    
    return df_results
```

---

## 7. 练习题 | Exercises

### 练习 1: 催化剂筛选

```python
# 筛选CO2还原催化剂
# 目标: 低过电位、高选择性

def screen_co2rr_catalysts():
    """CO2还原催化剂筛选"""
    # 查询金属表面材料
    # 计算吸附能
    # 评估过电位
    pass
```

### 练习 2: 光伏材料筛选

```python
# 筛选钙钛矿光伏材料
# 目标: 合适带隙、高吸收系数、稳定性好

def screen_perovskite_pv():
    """钙钛矿光伏材料筛选"""
    # 查询钙钛矿结构
    # 计算带隙和形成能
    # 筛选合适候选
    pass
```

---

**下一步**: [06 - HPC集群使用指南](06_hpc_deployment.md)
