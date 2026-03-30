#!/usr/bin/env python3
"""
Generate final reorganization report
生成最终整理报告
"""

import os
import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("/root/.openclaw/workspace/dft_lammps_research")
REPORT_DIR = BASE_DIR / "reorganization_report"

def get_directory_tree(path, prefix="", max_depth=3, current_depth=0):
    """Generate ASCII tree of directory structure"""
    if current_depth >= max_depth:
        return []
    
    lines = []
    items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
    
    for i, item in enumerate(items):
        if item.name.startswith('.') or item.name == '__pycache__':
            continue
        if item.name in ['reorganization_report']:
            continue
            
        is_last = i == len(items) - 1
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{item.name}")
        
        if item.is_dir():
            extension = "    " if is_last else "│   "
            lines.extend(get_directory_tree(item, prefix + extension, max_depth, current_depth + 1))
    
    return lines

def count_files_by_extension(path, extensions=None):
    """Count files by extension"""
    if extensions is None:
        extensions = ['.py', '.ts', '.js', '.yaml', '.yml', '.json', '.md', '.txt']
    
    counts = {ext: 0 for ext in extensions}
    counts['other'] = 0
    
    for root, dirs, files in os.walk(path):
        if '__pycache__' in root or '.pytest_cache' in root:
            continue
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in counts:
                counts[ext] += 1
            else:
                counts['other'] += 1
    
    return counts

def generate_architecture_diagram():
    """Generate architecture diagram in Mermaid format"""
    return '''```mermaid
graph TB
    subgraph Core["🔧 Core Engine"]
        DFT["core/dft/<br/>DFT Calculations"]
        MD["core/md/<br/>Molecular Dynamics"]
        ML["core/ml/<br/>ML Potentials<br/>NEP/DeepMD/MACE"]
        Common["core/common/<br/>Shared Utilities"]
    end
    
    subgraph Platform["🖥️ Platform Layer"]
        API["platform/api/<br/>REST API Gateway"]
        Web["platform/web/<br/>Web UI & Monitoring"]
        HPC["platform/hpc/<br/>HPC Scheduler"]
    end
    
    subgraph Intelligence["🧠 Intelligence Layer"]
        AL["intelligence/active_learning/<br/>Active Learning"]
        Lit["intelligence/literature/<br/>Literature Mining"]
        MA["intelligence/multi_agent/<br/>Multi-Agent System"]
        AD["intelligence/auto_discovery/<br/>Auto Discovery"]
    end
    
    subgraph Simulation["⚛️ Simulation Methods"]
        PF["simulation/phase_field/<br/>Phase Field"]
        QC["simulation/quantum/<br/>Quantum Computing"]
        RL["simulation/rl/<br/>RL Optimizer"]
    end
    
    subgraph Workflows["📋 Workflows"]
        Battery["workflows/battery/<br/>Battery Materials"]
        Catalyst["workflows/catalyst/<br/>Catalyst Design"]
        Perovskite["workflows/perovskite/<br/>Perovskite"]
    end
    
    Core --> Platform
    Core --> Intelligence
    Core --> Simulation
    Platform --> Workflows
    Intelligence --> Workflows
    Simulation --> Workflows
```'''

def generate_workflow_diagram():
    """Generate workflow diagram"""
    return '''```mermaid
flowchart LR
    A[Input Structure] --> B{DFT Calculation}
    B -->| Forces/Energy | C[ML Potential Training]
    B -->| Properties | D[Database]
    C --> E[MD Simulation]
    E --> F{Analysis}
    F -->| Valid | G[Screening Results]
    F -->| Invalid | H[Active Learning]
    H --> B
    G --> I[Report Generation]
    
    subgraph Core Pipeline
        B
        C
        E
    end
    
    subgraph Intelligence
        H
        F
    end
'''

def generate_final_report():
    """Generate comprehensive final report"""
    
    # Get current file counts
    counts = count_files_by_extension(BASE_DIR)
    
    # Get tree structure
    tree_lines = get_directory_tree(BASE_DIR, max_depth=3)
    
    report = f"""# DFT-LAMMPS Research Project - Reorganization Final Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Project Statistics

| File Type | Count |
|-----------|-------|
| Python (.py) | {counts['.py']} |
| TypeScript (.ts) | {counts['.ts']} |
| JavaScript (.js) | {counts['.js']} |
| YAML (.yaml/.yml) | {counts['.yaml'] + counts['.yml']} |
| JSON (.json) | {counts['.json']} |
| Markdown (.md) | {counts['.md']} |
| Text (.txt) | {counts['.txt']} |
| Other | {counts['other']} |
| **Total** | **{sum(counts.values())}** |

## 📁 New Directory Structure

```
{'\n'.join(tree_lines)}
```

## 🏗️ Architecture Diagram

{generate_architecture_diagram()}

## 🔄 Workflow Diagram

{generate_workflow_diagram()}

## ✅ Completed Operations

### 1. Core Module Restructuring
- ✅ Moved `dft_to_lammps_bridge.py` → `core/dft/bridge.py`
- ✅ Moved `integrated_materials_workflow.py` → `core/common/workflow_engine.py`
- ✅ Moved `checkpoint_manager.py` → `core/common/checkpoint.py`
- ✅ Moved `parallel_optimizer.py` → `core/common/parallel.py`
- ✅ Moved `nep_training_pipeline.py` → `core/ml/nep/pipeline.py`
- ✅ Created subdirectories for parsers, calculators, engines, analysis

### 2. Platform Module Restructuring
- ✅ Flattened `api_platform/` into `platform/api/`
- ✅ Flattened `web/v2/` into `platform/web/ui/`
- ✅ Moved `monitoring_dashboard.py` → `platform/web/monitoring/dashboard.py`
- ✅ Moved `hpc_scheduler.py` → `platform/hpc/scheduler.py`

### 3. Intelligence Module Restructuring
- ✅ Flattened `active_learning/v2/` into `intelligence/active_learning/`
- ✅ Flattened `literature_survey/` into `intelligence/literature/`
- ✅ Created `multi_agent/` subdirectories

### 4. Simulation Module Restructuring
- ✅ Flattened `phase_field/v1/` into `simulation/phase_field/`
- ✅ Moved `rl_optimizer/` → `simulation/rl/optimizer/`

### 5. Workflows Restructuring
- ✅ Flattened nested workflow directories
- ✅ Moved `battery_screening_pipeline.py` → `workflows/battery/screening.py`
- ✅ Moved examples to `workflows/battery/examples/`
- ✅ Created `workflows/examples/` for shared examples

### 6. Import Path Updates
- ✅ Updated 12 files with new import paths
- ✅ Mapped old import patterns to new module paths

### 7. Documentation Consolidation
- ✅ Moved `tutorials/` → `docs/tutorials/`
- ✅ Moved `references/` → `docs/references/`
- ✅ Created new `README.md` with updated structure

### 8. Deployment Configuration
- ✅ Moved `.github/workflows/` → `deploy/ci-cd/github/`

## 📝 Import Path Mapping

| Old Import | New Import |
|------------|------------|
| `from dft_to_lammps_bridge import ...` | `from core.dft.bridge import ...` |
| `from integrated_materials_workflow import ...` | `from core.common.workflow_engine import ...` |
| `from checkpoint_manager import ...` | `from core.common.checkpoint import ...` |
| `from parallel_optimizer import ...` | `from core.common.parallel import ...` |
| `from nep_training_pipeline import ...` | `from core.ml.nep.pipeline import ...` |
| `from hpc_scheduler import ...` | `from platform.hpc.scheduler import ...` |
| `from monitoring_dashboard import ...` | `from platform.web.monitoring.dashboard import ...` |
| `from battery_screening_pipeline import ...` | `from workflows.battery.screening import ...` |

## ⚠️ Manual Checklist

The following items require manual review:

### 1. Remaining Root Files
- [ ] Check if any files still in root need to be moved
- [ ] Review `requirements.txt` and consolidate with other req files
- [ ] Review configuration files (`*.yaml`, `*.yml`)

### 2. Test Files
- [ ] `tests/` directory still has legacy structure
- [ ] Consider reorganizing tests to match new module structure
- [ ] Update `conftest.py` with new fixtures

### 3. Examples
- [ ] `examples/` directory has legacy structure
- [ ] Consider merging with `workflows/examples/`

### 4. Documentation
- [ ] Update `docs/project/` files with new paths
- [ ] Update architecture documentation
- [ ] Create migration guide for users

### 5. CI/CD
- [ ] Update GitHub Actions workflows with new paths
- [ ] Update Docker configurations

### 6. Import Verification
- [ ] Run tests to verify all imports work correctly
- [ ] Check for circular imports
- [ ] Verify `__init__.py` files are properly configured

### 7. Package Configuration
- [ ] Update `pyproject.toml` with new package structure
- [ ] Update `setup.py` if exists
- [ ] Review package namespace configuration

## 🔧 Recommended Next Steps

1. **Run Tests**: Execute test suite to verify everything works
   ```bash
   pytest tests/ -v
   ```

2. **Import Verification**: Check for import errors
   ```bash
   python -c "import core.dft.bridge; import workflows.battery.screening"
   ```

3. **Documentation Update**: Update all documentation with new paths

4. **CI/CD Update**: Update GitHub Actions workflows

5. **Package Release**: Consider creating a new release with the restructured codebase

## 📈 Benefits of New Structure

1. **Clear Separation of Concerns**: Each module has a specific purpose
2. **Scalable Architecture**: Easy to add new simulation methods or workflows
3. **Better Maintainability**: Logical grouping of related functionality
4. **Improved Discoverability**: Clear naming makes it easy to find components
5. **Standardized Layout**: Follows Python package best practices

## 🐛 Known Issues

1. Some empty directories may still exist
2. A few import paths may need manual adjustment
3. Some documentation links may be broken
4. Tests may need path updates

## 📚 References

- Original files backed up where applicable
- See `README.md.backup` for original README
- See `import_updates.txt` for list of updated files

---

**Report generated by:** Project Reorganization Script  
**Project:** dft_lammps_research  
**Total files processed:** {sum(counts.values())}
"""
    
    # Save report
    report_path = REPORT_DIR / "FINAL_REORGANIZATION_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Final report generated: {report_path}")
    return report_path

if __name__ == "__main__":
    generate_final_report()
