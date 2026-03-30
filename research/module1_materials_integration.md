# 材料数据库与自动化工具研究报告
## 模块1: Materials Project / AiiDA / Atomate 深度集成

**研究时间**: 2026-03-08 16:31+

---

## 1. Materials Project (材料项目)

### 1.1 核心架构
- **数据库规模**: 超过150万种材料计算数据
- **核心引擎**: pymatgen (Python Materials Genomics)
- **API演进**: 
  - Legacy API (2025年9月即将停用)
  - Next-Gen API (`mp-api`包) - 当前推荐

### 1.2 最新API特性 (2024-2025)
```python
from mp_api.client import MPRester

with MPRester() as mpr:
    # 任务端点获取历史结构数据
    tasks = mpr.materials.tasks.search(task_ids=['mvc-13350', 'mp-867303'])
    
    # 获取数据库版本
    db_version = mpr.get_database_version()
```

### 1.3 2025年数据库更新
- **2025年9月**: 插入电极集合修正，新增~1,200个文档
- **2025年6月**: 迁移~1,500个材料的DFPT声子数据
- **2025年4月**: 新增30,000个GNoME起源材料(r2SCAN计算)

---

## 2. AiiDA (自动化交互式基础设施)

### 2.1 核心特性
- **数据溯源**: 完整的有向图追踪(DAG)
- **插件生态**: 支持50+材料科学代码
- **可扩展性**: 支持exascale计算资源饱和

### 2.2 支持的计算代码
| 代码类型 | 具体软件 |
|---------|---------|
| DFT | Quantum ESPRESSO, VASP, CP2K, CASTEP, ABINIT |
| 光谱 | Yambo, Wannier90 |
| 分子动力学 | LAMMPS, i-PI |
| 其他 | SIESTA, Fleur, Crystal, NWChem |

### 2.3 Work Chain架构
```python
# 工作链支持中断恢复和错误处理
# 内置Exit Codes系统
# 支持动态决策重新提交计算
```

### 2.4 材料云集成 (Materials Cloud)
- 基于AiiDA的完整溯源模型
- 符合FAIR原则 (Findable, Accessible, Interoperable, Reusable)
- 支持教育资源、交互工具、模拟服务

---

## 3. Atomate / Atomate2

### 3.1 架构对比
| 特性 | Atomate1 | Atomate2 |
|------|----------|----------|
| 工作流引擎 | FireWorks | Jobflow |
| 数据库 | MongoDB | 多种后端支持 |
| 默认泛函 | PBE | PBEsol |
| 赝势 | PBE_52 | PBE_54 |

### 3.2 标准Workflows
- 结构优化 (OptimizeFW)
- 静态计算 (StaticFW)
- 能带结构 (BandStructureFW)
- 弹性常数 (ElasticFW)
- 声子计算 (PhononFW)
- NEB过渡态 (NEBFW)
- LOBSTER键合分析

### 3.3 VASP配置示例
```yaml
# ~/.atomate2.yaml
VASP_CMD: "mpirun -n 64 vasp_std > vasp.out"
VASP_GAMMA_CMD: "mpirun -n 64 vasp_gam > vasp.out"
VASP_NCL_CMD: "mpirun -n 64 vasp_ncl > vasp.out"
VASP_INCAR_UPDATES:
  NCORE: 8
  KPAR: 4
```

### 3.4 代码示例
```python
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.flows.lobster import VaspLobsterMaker
from jobflow import run_locally

# 创建LOBSTER键合分析工作流
lobster_flow = VaspLobsterMaker().make(structure)
run_locally(lobster_flow, create_folders=True)
```

---

## 4. 集成方案建议

### 4.1 推荐技术栈
```
Materials Project (数据源)
    ↓
mp-api / pymatgen (数据获取与处理)
    ↓
AiiDA (工作流管理与溯源)
    ↓
Atomate2 (高通量计算工作流)
    ↓
MongoDB (数据存储)
```

### 4.2 关键注意事项
1. **API迁移**: Legacy API将于2025年9月停用，需迁移至mp-api
2. **能量不可比**: Atomate2默认PBEsol与Materials Project的PBE能量不可直接比较
3. **输入集配置**: 需根据目标数据库调整INCAR参数

---

**模块1研究完成**
