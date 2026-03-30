# DFT自动化工作流: AiiDA, Atomate, AIRSS

## 概述

自动化工作流是现代DFT计算的核心，实现从结构输入到结果分析的全流程自动化。本模块介绍三种主流框架：**AiiDA** (数据驱动)、**Atomate** (FireWorks工作流)、**AIRSS** (结构搜索)。

---

## 1. AiiDA (AI for Science)

AiiDA是基于Python的材料信息学平台，提供完整的数据溯源和工作流管理。

### 1.1 安装与配置

```bash
# 安装AiiDA
pip install aiida-core
pip install aiida-vasp
pip install aiida-quantumespresso

# 设置配置文件
verdi quicksetup
# 配置计算机和代码
verdi computer setup --config computer.yml
verdi code setup --config code.yml
```

**computer.yml配置示例**:
```yaml
---
label: "hpc_cluster"
hostname: "hpc.university.edu"
description: "University HPC cluster"
transport: ssh
scheduler: slurm
work_dir: "/scratch/{username}/aiida_run"
mpirun_command: "srun -n {tot_num_mpiprocs}"
mpiprocs_per_machine: 32
shebang: "#!/bin/bash"
```

### 1.2 基础工作流

```python
from aiida import orm, engine
from aiida.plugins import CalculationFactory, WorkflowFactory

def submit_vasp_calculation():
    """提交单个VASP计算"""
    
    # 加载代码
    code = orm.load_code('vasp-6.4.0@hpc_cluster')
    
    # 构建输入
    builder = code.get_builder()
    
    # 结构
    structure = orm.StructureData(pymatgen_structure=mg_structure)
    builder.structure = structure
    
    # K点
    kpoints = orm.KpointsData()
    kpoints.set_kpoints_mesh([8, 8, 8])
    builder.kpoints = kpoints
    
    # INCAR参数
    parameters = orm.Dict(dict={
        'SYSTEM': 'Si Bulk',
        'ENCUT': 520,
        'ISMEAR': -5,
        'EDIFF': 1E-8,
        'NSW': 0
    })
    builder.parameters = parameters
    
    # 赝势
    builder.pseudo = load_pseudo_family('PBE.54')
    
    # 资源
    builder.metadata.options = {
        'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 16},
        'max_wallclock_seconds': 3600,
        'queue_name': 'normal'
    }
    
    # 提交
    calculation = engine.submit(builder)
    
    print(f"Submitted calculation PK: {calculation.pk}")
    
    return calculation
```

### 1.3 复杂工作流: 能带计算

```python
from aiida.plugins import WorkflowFactory
from aiida.engine import submit

def run_band_structure_workflow():
    """
    能带结构计算工作流
    
    流程:
    1. 结构优化
    2. 自洽计算
    3. 非自洽能带计算
    4. 结果提取和可视化
    """
    
    # 使用AiiDA-VASP的工作流
    VaspBandsWorkChain = WorkflowFactory('vasp.bands')
    
    builder = VaspBandsWorkChain.get_builder()
    
    # 输入结构
    builder.structure = load_structure('Si.cif')
    
    # 代码
    builder.vasp.code = load_code('vasp-6.4.0@hpc')
    
    # 参数
    builder.vasp.parameters = {
        'relax': {
            'encut': 520,
            'ediff': 1E-8,
            'ediffg': -1E-3,
            'ibrion': 2,
            'isif': 3
        },
        'scf': {
            'encut': 520,
            'icharg': 0,
            'kspacing': 0.1
        },
        'bands': {
            'kpath': 'SC',  # 或自定义路径
            'kpoint_distance': 0.02
        }
    }
    
    # 运行
    results = submit(builder)
    
    return results
```

### 1.4 高通量筛选工作流

```python
from aiida import orm
from aiida.engine import run_get_node

def high_throughput_screening(structure_list, screening_criteria):
    """
    高通量筛选工作流
    
    并行计算大量结构，筛选符合条件的候选
    """
    
    from aiida.plugins import WorkflowFactory
    RelaxWorkChain = WorkflowFactory('vasp.relax')
    
    passed_structures = []
    
    for i, structure in enumerate(structure_list):
        print(f"Processing structure {i+1}/{len(structure_list)}")
        
        builder = RelaxWorkChain.get_builder()
        builder.structure = structure
        builder.vasp.code = load_code('vasp@hpc')
        
        # 快速设置
        builder.vasp.parameters = get_relax_parameters()
        
        # 运行并等待结果
        results, node = run_get_node(builder)
        
        # 提取能量和体积
        energy = results['total_energy'].value
        volume = results['output_structure'].get_cell_volume()
        
        # 筛选
        if screening_criteria['energy_threshold']:
            if energy < screening_criteria['energy_threshold']:
                if volume > screening_criteria['min_volume']:
                    passed_structures.append({
                        'structure': results['output_structure'],
                        'energy': energy,
                        'volume': volume
                    })
    
    print(f"筛选完成: {len(passed_structures)}/{len(structure_list)} 通过")
    
    return passed_structures
```

### 1.5 数据查询与追溯

```python
from aiida import orm
from aiida.orm import QueryBuilder

def query_calculations():
    """查询计算数据库"""
    
    # 查询所有VASP计算
    qb = QueryBuilder()
    qb.append(orm.CalcJobNode, 
              filters={'attributes.process_label': {'like': '%vasp%'}},
              tag='vasp')
    
    results = qb.all()
    print(f"Found {len(results)} VASP calculations")
    
    return results

def get_provenance(pk):
    """
    获取计算的血缘图
    
    AiiDA的核心优势: 完整的数据溯源
    """
    
    node = orm.load_node(pk)
    
    # 打印血缘
    print(f"Node {pk} provenance:")
    print(f"  Type: {node.node_type}")
    print(f"  Created: {node.ctime}")
    
    # 输入
    print("\n  Inputs:")
    for link_label, input_node in node.get_incoming().all_nodes_by_link_label().items():
        print(f"    {link_label}: {input_node.pk} ({input_node.node_type})")
    
    # 输出
    print("\n  Outputs:")
    for link_label, output_node in node.get_outgoing().all_nodes_by_link_label().items():
        print(f"    {link_label}: {output_node.pk} ({output_node.node_type})")
    
    # 生成图形
    from aiida.tools.visualization import graph_visualization
    graph = graph_visualization.get_graph([node], link_types=['input', 'output'])
    graph.render('provenance', format='png')
```

---

## 2. Atomate (Materials Project工作流)

Atomate基于FireWorks构建，与Materials Project数据格式无缝衔接。

### 2.1 安装与配置

```bash
# 安装
pip install atomate
pip install fireworks

# 配置
lpad init                              # 初始化LaunchPad
lpad -l my_launchpad.yaml init        # 或使用自定义配置
```

**my_launchpad.yaml**:
```yaml
host: localhost
port: 27017
name: atomate_db
username: null
password: null
```

### 2.2 基础工作流

```python
from fireworks import LaunchPad, Firework, Workflow
from atomate.vasp.workflows.base.core import get_wf
from atomate.vasp.powerups import *
from pymatgen import Structure

def create_band_structure_workflow(structure_file):
    """
    创建能带结构工作流
    
    使用Atomate预设工作流
    """
    
    # 加载结构
    structure = Structure.from_file(structure_file)
    
    # 获取预设工作流
    wf = get_wf(structure, "band_structure")
    
    # 添加powerups (增强功能)
    wf = add_modify_incar(wf, modify_incar_params={"ENCUT": 520})
    wf = use_custodian(wf)  # 添加错误处理
    
    # 提交到LaunchPad
    lpad = LaunchPad.auto_load()
    lpad.add_wf(wf)
    
    print(f"Added workflow with {len(wf.fws)} fireworks")
    
    return wf
```

### 2.3 运行工作流

```bash
# 查看待运行任务
lpad get_fws -s READY

# 运行单个任务
rlaunch singleshot

# 无限循环运行 (生产环境)
rlaunch rapidfire --nlaunches infinite

# 监控
lpad introspect
```

### 2.4 弹性常数工作流

```python
from atomate.vasp.workflows.base.elastic import get_wf_elastic_constant

def run_elastic_workflow(structure):
    """
    弹性常数计算工作流
    
    自动:
    1. 结构优化
    2. 生成形变结构
    3. 计算每个形变的应力
    4. 拟合弹性张量
    """
    
    wf = get_wf_elastic_constant(
        structure,
        vasp_cmd="mpirun -np 16 vasp_std",
        db_file="db.json",
        norm_deformations=[-0.02, -0.01, 0.01, 0.02],
        shear_deformations=[-0.02, -0.01, 0.01, 0.02]
    )
    
    # 添加优先级
    wf = add_priority(wf, 100)
    
    # 提交
    lpad = LaunchPad.auto_load()
    lpad.add_wf(wf)
    
    return wf
```

### 2.5 与Materials Project集成

```python
from mp_api.client import MPRester
from atomate.vasp.powerups import add_tags

def get_mp_structures(chemsys, property_filter=None):
    """
    从Materials Project获取结构并运行计算
    """
    
    with MPRester("YOUR_API_KEY") as mpr:
        # 查询材料
        docs = mpr.materials.summary.search(
            chemsys=chemsys,
            band_gap=(0.5, 2.0),  # 筛选带隙
            energy_above_hull=(0, 0.1)  # 稳定结构
        )
    
    workflows = []
    
    for doc in docs:
        structure = doc.structure
        
        # 创建工作流
        wf = get_wf(structure, "optimize_only")
        
        # 添加MP ID标签
        wf = add_tags(wf, [doc.material_id])
        
        workflows.append(wf)
    
    # 批量提交
    lpad = LaunchPad.auto_load()
    for wf in workflows:
        lpad.add_wf(wf)
    
    print(f"Submitted {len(workflows)} workflows from MP")
    
    return workflows
```

---

## 3. AIRSS (Ab Initio Random Structure Searching)

AIRSS是结构预测的经典方法，通过随机生成+对称约束寻找全局最优结构。

### 3.1 安装与配置

```bash
# 下载并编译
wget http://www.mtg.msm.cam.ac.uk/Codes/AIRSS/airss-0.9.1.tar.gz
tar -xzf airss-0.9.1.tar.gz
cd airss
make

# 设置环境变量
export PATH=$PATH:/path/to/airss/bin
export AIRSS_ROOT=/path/to/airss
```

### 3.2 结构搜索

**输入文件: LiCoO2.cell**:
```
%BLOCK LATTICE_CART
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 4.0
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
Li 0.0 0.0 0.0
Co 0.5 0.5 0.5
O  0.5 0.0 0.0
O  0.0 0.5 0.0
%ENDBLOCK POSITIONS_FRAC

# AIRSS约束
#POSAMP=0.1       # 位置扰动幅度
#SYMMOPS=4        # 最小对称操作数
#NFORM=4          # 公式单元数
#COMPACT          # 紧凑结构偏好
```

**运行搜索**:
```bash
# 生成随机结构
crush_cell LiCoO2.cell -n 100  # 生成100个候选

# 提交计算
airss.pl -max 100 -mpinp 16 -seed LiCoO2

# 或使用CASTEP接口
airss.pl -max 100 -castep -seed LiCoO2
```

### 3.3 Python接口

```python
import subprocess
import os
from pymatgen import Structure

def airss_structure_search(composition, num_structures=100, pressure=0):
    """
    使用AIRSS进行结构搜索
    
    Parameters:
    -----------
    composition : dict
        如 {'Li': 1, 'Co': 1, 'O': 2}
    num_structures : int
        生成候选结构数量
    """
    
    # 生成.cell输入文件
    cell_content = generate_airss_input(composition)
    
    with open(f"{composition}.cell", 'w') as f:
        f.write(cell_content)
    
    # 运行AIRSS
    cmd = [
        'airss.pl',
        '-max', str(num_structures),
        '-mpinp', '16',
        '-seed', composition
    ]
    
    if pressure != 0:
        cmd.extend(['-press', str(pressure)])
    
    subprocess.run(cmd)
    
    # 收集结果
    results = []
    
    for file in os.listdir('.'):
        if file.endswith('.res'):  # AIRSS结果文件
            structure, energy = parse_airss_res(file)
            results.append({
                'file': file,
                'structure': structure,
                'energy': energy
            })
    
    # 按能量排序
    results.sort(key=lambda x: x['energy'])
    
    print(f"搜索完成，找到 {len(results)} 个独特结构")
    print(f"最低能量: {results[0]['energy']:.4f} eV/atom")
    
    return results

def parse_airss_res(filename):
    """解析AIRSS结果文件"""
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # 解析结构
    lattice = []
    coords = []
    species = []
    
    for line in lines:
        if line.startswith('LATTICE'):
            # 解析晶格
            pass
        elif line.startswith('COORDINATES'):
            # 解析坐标
            pass
    
    structure = Structure(lattice, species, coords)
    
    # 解析能量
    energy = float([l for l in lines if 'energy' in l][0].split()[-1])
    
    return structure, energy
```

### 3.4 变组分搜索

```python
def variable_composition_search(elements, max_atoms=20):
    """
    变组分结构搜索
    
    探索给定元素的所有可能化学计量比
    """
    
    from itertools import product
    
    # 生成可能的化学计量比
    stoichiometries = []
    
    for n_atoms in range(2, max_atoms + 1):
        for combo in product(range(1, n_atoms), repeat=len(elements)):
            if sum(combo) == n_atoms:
                comp = dict(zip(elements, combo))
                stoichiometries.append(comp)
    
    # 对每个组分进行搜索
    all_results = {}
    
    for comp in stoichiometries:
        comp_str = ''.join([f"{k}{v}" for k, v in comp.items()])
        print(f"Searching {comp_str}...")
        
        results = airss_structure_search(comp, num_structures=50)
        all_results[comp_str] = results
    
    # 构建凸包
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    
    pd = PhaseDiagram([r['structure'] for r in all_results.values()])
    
    return pd, all_results
```

---

## 4. 框架对比与选择

| 特性 | AiiDA | Atomate | AIRSS |
|------|-------|---------|-------|
| **核心优势** | 数据溯源 | MP集成 | 结构搜索 |
| **数据库** | PostgreSQL | MongoDB | 文件系统 |
| **工作流引擎** | 自研 | FireWorks | Shell脚本 |
| **学习曲线** | 陡峭 | 中等 | 平缓 |
| **社区规模** | 大 | 中 | 小 |
| **推荐场景** | 研究项目 | 高通量筛选 | 结构预测 |

### 4.1 选择指南

```
┌─────────────────────────────────────────────────────┐
│                  工作流框架选择                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  需要完整数据溯源?                                   │
│     ├─ 是 → AiiDA                                   │
│     └─ 否 →                                         │
│           使用MP数据?                                │
│              ├─ 是 → Atomate                        │
│              └─ 否 →                                │
│                    结构预测为主?                      │
│                       ├─ 是 → AIRSS                 │
│                       └─ 否 → AiiDA (通用性强)      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 5. 高级技巧

### 5.1 AiiDA + 机器学习集成

```python
def aiida_ml_workflow(structure):
    """
    AiiDA与机器学习结合
    
    用ML快速预筛选，DFT精确计算
    """
    
    from aiida import orm, engine
    
    @engine.calcfunction
    def ml_screen(structure):
        """ML筛选计算函数"""
        
        # 加载ML模型
        from chgnet.model import CHGNet
        chgnet = CHGNet.load()
        
        # 预测
        atoms = structure.get_ase()
        atoms.calc = chgnet
        
        energy = atoms.get_potential_energy()
        
        # 返回结果
        return orm.Dict(dict={
            'ml_energy': energy,
            'pass_screening': energy < -5.0
        })
    
    # 运行ML筛选
    ml_result = ml_screen(structure)
    
    # 条件DFT
    if ml_result['pass_screening']:
        # 提交DFT工作流
        dft_wf = submit_vasp_workflow(structure)
        return dft_wf
    else:
        print("Structure failed ML screening")
        return None
```

### 5.2 错误恢复与重试

```python
from fireworks import FiretaskBase, explicit_serialize
from fireworks.core.rocket_launcher import rapidfire

@explicit_serialize
def VaspWithRetryTask(FiretaskBase):
    """
    带自动重试的VASP任务
    """
    
    _fw_name = "VaspWithRetryTask"
    
    def run_task(self, fw_spec):
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                result = run_vasp_calculation(fw_spec)
                return FWAction(stored_data={'result': result})
            except VaspConvergenceError:
                if attempt < max_retries - 1:
                    # 调整参数重试
                    modify_incar_for_convergence()
                    continue
                else:
                    raise
            except Exception as e:
                # 记录错误并通知
                send_notification(f"VASP failed: {e}")
                raise
```

---

## 参考

1. S. P. Huber et al., *Sci. Data* 7, 300 (2020) - AiiDA论文
2. K. Mathew et al., *Comput. Mater. Sci.* 139, 140 (2017) - Atomate论文
3. C. J. Pickard & R. J. Needs, *J. Phys. Condens. Matter* 23, 053201 (2011) - AIRSS论文
4. AiiDA文档: https://aiida.readthedocs.io/
5. Atomate文档: https://atomate.org/
6. AIRSS文档: http://www.mtg.msm.cam.ac.uk/Codes/AIRSS
