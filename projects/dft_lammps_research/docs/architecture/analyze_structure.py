#!/usr/bin/env python3
"""
DFT-LAMMPS Research Project Structure Reorganization Script
项目结构重组脚本

This script analyzes the current project structure and creates a comprehensive
reorganization plan to achieve a clean, logical directory hierarchy.
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Base path
BASE_DIR = Path("/root/.openclaw/workspace/dft_lammps_research")
REPORT_DIR = BASE_DIR / "reorganization_report"

def ensure_dir(path):
    """Ensure directory exists"""
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_all_files():
    """Get all files in the project with their types"""
    files = {
        'python': [],
        'typescript': [],
        'javascript': [],
        'yaml': [],
        'json': [],
        'markdown': [],
        'txt': [],
        'other': []
    }
    
    exclude_patterns = ['__pycache__', '.pytest_cache', '.git', '.benchmarks', 
                        'reorganization_report', 'nep_checkpoints', 'nep_output',
                        'validation_results', 'test_reports']
    
    for root, dirs, filenames in os.walk(BASE_DIR):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_patterns and not d.startswith('.')]
        
        for filename in filenames:
            filepath = Path(root) / filename
            rel_path = filepath.relative_to(BASE_DIR)
            
            # Skip hidden files
            if filename.startswith('.'):
                continue
                
            ext = filepath.suffix.lower()
            
            if ext == '.py':
                files['python'].append(str(rel_path))
            elif ext == '.ts':
                files['typescript'].append(str(rel_path))
            elif ext == '.js':
                files['javascript'].append(str(rel_path))
            elif ext in ['.yaml', '.yml']:
                files['yaml'].append(str(rel_path))
            elif ext == '.json':
                files['json'].append(str(rel_path))
            elif ext in ['.md', '.rst']:
                files['markdown'].append(str(rel_path))
            elif ext == '.txt':
                files['txt'].append(str(rel_path))
            else:
                files['other'].append(str(rel_path))
    
    return files

def analyze_imports():
    """Analyze import statements in Python files"""
    imports = defaultdict(list)
    
    for py_file in BASE_DIR.rglob("*.py"):
        if '__pycache__' in str(py_file):
            continue
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('import ') or line.startswith('from '):
                        rel_path = py_file.relative_to(BASE_DIR)
                        imports[str(rel_path)].append(line)
        except Exception as e:
            print(f"Error reading {py_file}: {e}")
    
    return imports

def create_reorganization_plan():
    """Create the reorganization plan based on analysis"""
    
    plan = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'project': 'dft_lammps_research',
            'description': 'Project structure reorganization plan'
        },
        'move_operations': [],
        'merge_operations': [],
        'delete_operations': [],
        'create_operations': [],
        'import_updates': []
    }
    
    # Define move operations
    moves = [
        # Core modules
        {"source": "core/dft_to_lammps_bridge.py", "target": "core/dft/bridge.py", "description": "DFT bridge module"},
        {"source": "core/integrated_materials_workflow.py", "target": "core/common/workflow_engine.py", "description": "Integrated workflow engine"},
        {"source": "core/common/checkpoint_manager.py", "target": "core/common/checkpoint.py", "description": "Checkpoint management"},
        {"source": "core/common/parallel_optimizer.py", "target": "core/common/parallel.py", "description": "Parallel optimization utilities"},
        {"source": "core/ml/nep_training_pipeline.py", "target": "core/ml/nep/pipeline.py", "description": "NEP training pipeline"},
        
        # Platform modules
        {"source": "platform/monitoring_dashboard.py", "target": "platform/web/monitoring/dashboard.py", "description": "Monitoring dashboard"},
        {"source": "platform/hpc/hpc_scheduler.py", "target": "platform/hpc/scheduler.py", "description": "HPC scheduler"},
        {"source": "platform/api/api_platform", "target": "platform/api/", "description": "API platform (flatten)", "is_dir": True},
        {"source": "platform/web/v2", "target": "platform/web/ui/", "description": "Web UI v2", "is_dir": True},
        
        # Intelligence modules
        {"source": "intelligence/active_learning/v2", "target": "intelligence/active_learning/", "description": "Active learning v2 (flatten)", "is_dir": True, "merge": True},
        {"source": "intelligence/literature/literature_survey", "target": "intelligence/literature/", "description": "Literature survey (flatten)", "is_dir": True, "merge": True},
        
        # Simulation modules
        {"source": "simulation/phase_field/v1", "target": "simulation/phase_field/", "description": "Phase field (flatten)", "is_dir": True, "merge": True},
        {"source": "simulation/rl/rl_optimizer", "target": "simulation/rl/optimizer/", "description": "RL optimizer", "is_dir": True},
        
        # Workflow modules - fix nested structures
        {"source": "workflows/battery/solid_electrolyte", "target": "workflows/battery/", "description": "Battery workflow (flatten)", "is_dir": True, "merge": True},
        {"source": "workflows/battery/battery_screening_pipeline.py", "target": "workflows/battery/screening.py", "description": "Battery screening pipeline"},
        {"source": "workflows/battery/Li3PS4_workflow_example.py", "target": "workflows/battery/examples/Li3PS4.py", "description": "Li3PS4 example"},
        {"source": "workflows/catalyst/catalyst", "target": "workflows/catalyst/", "description": "Catalyst workflow (flatten)", "is_dir": True, "merge": True},
        {"source": "workflows/perovskite/perovskite", "target": "workflows/perovskite/", "description": "Perovskite workflow (flatten)", "is_dir": True, "merge": True},
        {"source": "workflows/screening_examples.py", "target": "workflows/examples/screening.py", "description": "Screening examples"},
        
        # Applications
        {"source": "applications", "target": "workflows/applications/", "description": "Application cases", "is_dir": True},
        
        # Examples consolidation
        {"source": "examples", "target": "examples/legacy/", "description": "Legacy examples", "is_dir": True, "merge": True},
        {"source": "code_templates", "target": "examples/templates/", "description": "Code templates", "is_dir": True, "merge": True},
        
        # Tests
        {"source": "tests", "target": "tests/legacy/", "description": "Legacy tests", "is_dir": True, "merge": True},
        {"source": "benchmarks", "target": "tests/benchmarks/", "description": "Benchmarks", "is_dir": True, "merge": True},
        
        # Documentation
        {"source": "docs", "target": "docs/project/", "description": "Project docs", "is_dir": True, "merge": True},
        {"source": "tutorials", "target": "docs/tutorials/", "description": "Tutorials", "is_dir": True, "merge": True},
        {"source": "references", "target": "docs/references/", "description": "References", "is_dir": True, "merge": True},
        
        # Scripts
        {"source": "scripts", "target": "scripts/tools/", "description": "Tool scripts", "is_dir": True, "merge": True},
        {"source": "generate_demo_data.py", "target": "scripts/generate_demo_data.py", "description": "Demo data generator"},
        
        # Docker
        {"source": "docker", "target": "deploy/docker/", "description": "Docker deployment", "is_dir": True, "merge": True},
        
        # CI/CD
        {"source": ".github", "target": "deploy/ci-cd/github/", "description": "GitHub workflows", "is_dir": True, "merge": True},
    ]
    
    plan['move_operations'] = moves
    
    # Define directories to create
    creates = [
        "core/dft/parsers/",
        "core/dft/calculators/",
        "core/md/engines/",
        "core/md/analysis/",
        "core/ml/deepmd/",
        "core/ml/mace/",
        "core/common/utils/",
        "core/common/models/",
        "platform/api/gateway/",
        "platform/api/auth/",
        "platform/api/integrations/",
        "platform/hpc/connectors/",
        "platform/hpc/monitoring/",
        "platform/web/ui/",
        "platform/web/monitoring/",
        "intelligence/multi_agent/agents/",
        "intelligence/multi_agent/orchestration/",
        "intelligence/auto_discovery/",
        "simulation/quantum/circuits/",
        "validation/experimental/",
        "validation/theoretical/",
        "examples/tutorials/",
        "examples/templates/",
        "docs/api/",
        "docs/architecture/",
    ]
    
    plan['create_operations'] = creates
    
    # Define files/directories to delete (empty or redundant)
    deletes = [
        "example_basic/",
        "example_advanced/",
        "example_al/",
        "example_ensemble/",
        "example_monitoring/",
        "high_throughput/",
        "ml_potentials/",
        "workflows/alloy/",  # empty
        "nep_training/",  # moved to core/ml/nep/
        "validation_results/",  # data directory
        "test_reports/",  # generated
    ]
    
    plan['delete_operations'] = deletes
    
    return plan

def generate_report():
    """Generate comprehensive reorganization report"""
    ensure_dir(REPORT_DIR)
    
    # Get file statistics
    files = get_all_files()
    total_files = sum(len(v) for v in files.values())
    
    stats = {
        'total_files': total_files,
        'python_files': len(files['python']),
        'typescript_files': len(files['typescript']),
        'javascript_files': len(files['javascript']),
        'yaml_files': len(files['yaml']),
        'json_files': len(files['json']),
        'markdown_files': len(files['markdown']),
        'other_files': len(files['other'])
    }
    
    # Save statistics
    with open(REPORT_DIR / "file_statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Save file lists
    with open(REPORT_DIR / "all_files.json", 'w') as f:
        json.dump(files, f, indent=2)
    
    # Create reorganization plan
    plan = create_reorganization_plan()
    
    with open(REPORT_DIR / "reorganization_plan.json", 'w') as f:
        json.dump(plan, f, indent=2)
    
    # Generate markdown report
    report_md = f"""# DFT-LAMMPS Research Project Reorganization Report

Generated: {datetime.now().isoformat()}

## File Statistics

| File Type | Count |
|-----------|-------|
| Python (.py) | {stats['python_files']} |
| TypeScript (.ts) | {stats['typescript_files']} |
| JavaScript (.js) | {stats['javascript_files']} |
| YAML (.yaml/.yml) | {stats['yaml_files']} |
| JSON (.json) | {stats['json_files']} |
| Markdown (.md) | {stats['markdown_files']} |
| Other | {stats['other_files']} |
| **Total** | **{stats['total_files']}** |

## Proposed Directory Structure

```
dft_lammps_research/
├── core/                    # Core engine (DFT/MD/ML unified interface)
│   ├── dft/                # DFT calculation module
│   ├── md/                 # Molecular dynamics module
│   ├── ml/                 # Machine learning potentials
│   │   ├── nep/           # NEP potential
│   │   ├── deepmd/        # DeePMD potential
│   │   └── mace/          # MACE potential
│   └── common/             # Shared utilities
├── platform/               # Platform layer
│   ├── api/                # API platform
│   ├── web/                # Web interface
│   │   ├── ui/            # Web UI
│   │   └── monitoring/    # Monitoring dashboard
│   └── hpc/                # HPC connectors
├── intelligence/           # Intelligence layer
│   ├── active_learning/    # Active learning
│   ├── literature/         # Literature intelligence
│   ├── multi_agent/        # Multi-agent coordination
│   └── auto_discovery/     # Automated discovery
├── simulation/             # Simulation methods
│   ├── phase_field/        # Phase field
│   ├── quantum/            # Quantum computing
│   └── rl/                 # RL optimizer
├── workflows/              # Workflow orchestration
│   ├── battery/            # Battery workflows
│   ├── catalyst/           # Catalyst workflows
│   ├── perovskite/         # Perovskite workflows
│   └── applications/       # Application cases
├── validation/             # Validation
│   ├── experimental/       # Experimental validation
│   └── theoretical/        # Theoretical validation
├── examples/               # Examples & templates
│   ├── templates/          # Code templates
│   └── tutorials/          # Tutorial files
├── tests/                  # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   ├── e2e/                # End-to-end tests
│   └── benchmarks/         # Performance benchmarks
├── docs/                   # Documentation
│   ├── project/            # Project documentation
│   ├── api/                # API documentation
│   ├── architecture/       # Architecture docs
│   ├── tutorials/          # Tutorials
│   └── references/         # References
├── scripts/                # Utility scripts
├── deploy/                 # Deployment configs
│   ├── docker/             # Docker configs
│   └── ci-cd/              # CI/CD configs
└── requirements/           # Dependency files
```

## Move Operations ({len(plan['move_operations'])})

"""
    
    for i, op in enumerate(plan['move_operations'], 1):
        report_md += f"{i}. **{op['source']}** → **{op['target']}**\n"
        report_md += f"   - Description: {op['description']}\n"
        if op.get('is_dir'):
            report_md += f"   - Type: Directory\n"
        if op.get('merge'):
            report_md += f"   - Action: Merge\n"
        report_md += "\n"
    
    report_md += f"""
## Create Operations ({len(plan['create_operations'])})

"""
    for i, path in enumerate(plan['create_operations'], 1):
        report_md += f"{i}. `{path}`\n"
    
    report_md += f"""

## Delete Operations ({len(plan['delete_operations'])})

**WARNING:** These directories will be removed (after verifying they're empty or redundant):

"""
    for i, path in enumerate(plan['delete_operations'], 1):
        report_md += f"{i}. `{path}`\n"
    
    report_md += """

## Next Steps

1. Review this report
2. Run the reorganization script with `--dry-run` to preview changes
3. Run the reorganization script with `--execute` to apply changes
4. Update import statements in moved files
5. Run tests to verify everything works
6. Update documentation

## Import Path Updates Required

After moving files, the following import patterns need to be updated:

- `from core.dft.bridge import ...` → `from core.dft.bridge import ...`
- `from core.common.workflow_engine import ...` → `from core.common.workflow_engine import ...`
- `from core.common.checkpoint import ...` → `from core.common.checkpoint import ...`
- `from core.common.parallel import ...` → `from core.common.parallel import ...`
- `from platform.hpc.scheduler import ...` → `from platform.hpc.scheduler import ...`
- `from platform.web.monitoring.dashboard import ...` → `from platform.web.monitoring.dashboard import ...`
"""
    
    with open(REPORT_DIR / "reorganization_report.md", 'w') as f:
        f.write(report_md)
    
    print(f"Report generated in: {REPORT_DIR}")
    print(f"Total files analyzed: {total_files}")
    print(f"Python files: {stats['python_files']}")
    print(f"Move operations: {len(plan['move_operations'])}")
    print(f"Create operations: {len(plan['create_operations'])}")
    print(f"Delete operations: {len(plan['delete_operations'])}")
    
    return plan

if __name__ == "__main__":
    plan = generate_report()
    print("\nReorganization plan created successfully!")
    print(f"View the report at: {REPORT_DIR / 'reorganization_report.md'}")
