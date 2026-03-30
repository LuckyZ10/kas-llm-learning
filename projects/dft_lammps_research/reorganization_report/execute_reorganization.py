#!/usr/bin/env python3
"""
DFT-LAMMPS Research Project Reorganization Executor
项目结构重组执行脚本

This script executes the reorganization plan created by analyze_structure.py
"""

import os
import sys
import shutil
import json
from pathlib import Path
from datetime import datetime

# Base path
BASE_DIR = Path("/root/.openclaw/workspace/dft_lammps_research")
LOG_FILE = BASE_DIR / "reorganization_report" / "execution_log.txt"

def log(msg):
    """Log message to console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg)
    with open(LOG_FILE, 'a') as f:
        f.write(log_msg + "\n")

def safe_move(src, dst, dry_run=False):
    """Safely move file or directory"""
    src_path = BASE_DIR / src
    dst_path = BASE_DIR / dst
    
    if not src_path.exists():
        log(f"SKIP: Source does not exist: {src}")
        return False
    
    if dst_path.exists():
        log(f"SKIP: Destination already exists: {dst}")
        return False
    
    # Ensure parent directory exists
    if not dry_run:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dry_run:
        log(f"DRY-RUN: Would move {src} → {dst}")
        return True
    
    try:
        if src_path.is_dir():
            shutil.move(str(src_path), str(dst_path))
        else:
            shutil.move(str(src_path), str(dst_path))
        log(f"MOVED: {src} → {dst}")
        return True
    except Exception as e:
        log(f"ERROR moving {src}: {e}")
        return False

def safe_merge(src, dst, dry_run=False):
    """Merge directory contents"""
    src_path = BASE_DIR / src
    dst_path = BASE_DIR / dst
    
    if not src_path.exists():
        log(f"SKIP: Source does not exist: {src}")
        return False
    
    if not src_path.is_dir():
        log(f"SKIP: Source is not a directory: {src}")
        return False
    
    if dry_run:
        log(f"DRY-RUN: Would merge {src} → {dst}")
        return True
    
    # Ensure destination exists
    dst_path.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    for item in src_path.iterdir():
        item_dst = dst_path / item.name
        if item_dst.exists():
            log(f"SKIP: {item.name} already exists in destination")
            continue
        try:
            shutil.move(str(item), str(item_dst))
            moved_count += 1
        except Exception as e:
            log(f"ERROR moving {item}: {e}")
    
    log(f"MERGED: {src} → {dst} ({moved_count} items)")
    
    # Remove empty source directory
    try:
        src_path.rmdir()
        log(f"REMOVED empty directory: {src}")
    except:
        pass
    
    return True

def safe_delete(path, dry_run=False):
    """Safely delete directory or file"""
    target = BASE_DIR / path
    
    if not target.exists():
        log(f"SKIP: Target does not exist: {path}")
        return False
    
    if dry_run:
        log(f"DRY-RUN: Would delete {path}")
        return True
    
    try:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        log(f"DELETED: {path}")
        return True
    except Exception as e:
        log(f"ERROR deleting {path}: {e}")
        return False

def create_directory(path, dry_run=False):
    """Create directory"""
    target = BASE_DIR / path
    
    if target.exists():
        log(f"SKIP: Directory already exists: {path}")
        return False
    
    if dry_run:
        log(f"DRY-RUN: Would create directory: {path}")
        return True
    
    try:
        target.mkdir(parents=True, exist_ok=True)
        log(f"CREATED: {path}")
        return True
    except Exception as e:
        log(f"ERROR creating {path}: {e}")
        return False

def fix_nested_structure(dry_run=False):
    """Fix nested duplicate directory structures"""
    log("=== Fixing Nested Structures ===")
    
    fixes = [
        # (source_nested, target_flat)
        ("platform/api/api_platform", "platform/api_contents"),
        ("platform/web/v2", "platform/web_ui"),
        ("intelligence/active_learning/v2", "intelligence/al_contents"),
        ("intelligence/literature/literature_survey", "intelligence/lit_contents"),
        ("simulation/phase_field/v1", "simulation/pf_contents"),
        ("simulation/rl/rl_optimizer", "simulation/rl_contents"),
        ("workflows/battery/solid_electrolyte", "workflows/battery_contents"),
        ("workflows/catalyst/catalyst", "workflows/catalyst_contents"),
        ("workflows/perovskite/perovskite", "workflows/perovskite_contents"),
    ]
    
    for src, temp in fixes:
        src_path = BASE_DIR / src
        if src_path.exists():
            safe_move(src, temp, dry_run)

def reorganize_core(dry_run=False):
    """Reorganize core module"""
    log("=== Reorganizing Core Module ===")
    
    # Move core files
    moves = [
        ("core/dft_to_lammps_bridge.py", "core/dft/bridge.py"),
        ("core/integrated_materials_workflow.py", "core/common/workflow_engine.py"),
        ("core/common/checkpoint_manager.py", "core/common/checkpoint.py"),
        ("core/common/parallel_optimizer.py", "core/common/parallel.py"),
        ("core/ml/nep_training_pipeline.py", "core/ml/nep/pipeline.py"),
    ]
    
    for src, dst in moves:
        safe_move(src, dst, dry_run)
    
    # Create additional directories
    dirs = [
        "core/dft/parsers",
        "core/dft/calculators",
        "core/md/engines",
        "core/md/analysis",
        "core/ml/deepmd",
        "core/ml/mace",
        "core/common/utils",
        "core/common/models",
    ]
    
    for d in dirs:
        create_directory(d, dry_run)

def reorganize_platform(dry_run=False):
    """Reorganize platform module"""
    log("=== Reorganizing Platform Module ===")
    
    # Move and flatten api_platform
    safe_merge("platform/api_contents", "platform/api/", dry_run)
    
    # Move and flatten web v2
    safe_merge("platform/web_ui", "platform/web/ui/", dry_run)
    
    # Move monitoring dashboard
    safe_move("platform/monitoring_dashboard.py", "platform/web/monitoring/dashboard.py", dry_run)
    
    # Move HPC scheduler
    safe_move("platform/hpc/hpc_scheduler.py", "platform/hpc/scheduler.py", dry_run)
    
    # Create additional directories
    dirs = [
        "platform/hpc/connectors",
        "platform/hpc/monitoring",
        "platform/web/monitoring",
    ]
    
    for d in dirs:
        create_directory(d, dry_run)

def reorganize_intelligence(dry_run=False):
    """Reorganize intelligence module"""
    log("=== Reorganizing Intelligence Module ===")
    
    # Flatten nested structures
    safe_merge("intelligence/al_contents", "intelligence/active_learning/", dry_run)
    safe_merge("intelligence/lit_contents", "intelligence/literature/", dry_run)
    
    # Create additional directories
    dirs = [
        "intelligence/multi_agent/agents",
        "intelligence/multi_agent/orchestration",
        "intelligence/auto_discovery",
    ]
    
    for d in dirs:
        create_directory(d, dry_run)

def reorganize_simulation(dry_run=False):
    """Reorganize simulation module"""
    log("=== Reorganizing Simulation Module ===")
    
    # Flatten nested structures
    safe_merge("simulation/pf_contents", "simulation/phase_field/", dry_run)
    safe_move("simulation/rl_contents", "simulation/rl/optimizer/", dry_run)
    
    # Create additional directories
    dirs = [
        "simulation/quantum/circuits",
    ]
    
    for d in dirs:
        create_directory(d, dry_run)

def reorganize_workflows(dry_run=False):
    """Reorganize workflows module"""
    log("=== Reorganizing Workflows Module ===")
    
    # Move workflow files
    safe_move("workflows/battery/battery_screening_pipeline.py", "workflows/battery/screening.py", dry_run)
    safe_move("workflows/battery/Li3PS4_workflow_example.py", "workflows/battery/examples/Li3PS4.py", dry_run)
    safe_move("workflows/screening_examples.py", "workflows/examples/screening.py", dry_run)
    
    # Flatten nested structures
    safe_merge("workflows/battery_contents", "workflows/battery/", dry_run)
    safe_merge("workflows/catalyst_contents", "workflows/catalyst/", dry_run)
    safe_merge("workflows/perovskite_contents", "workflows/perovskite/", dry_run)
    
    # Move applications
    safe_move("applications", "workflows/applications/", dry_run)

def reorganize_examples(dry_run=False):
    """Reorganize examples"""
    log("=== Reorganizing Examples ===")
    
    # Move existing examples
    safe_move("examples", "examples/legacy/", dry_run)
    safe_move("code_templates", "examples/templates/", dry_run)
    
    # Create directories
    dirs = [
        "examples/tutorials",
    ]
    
    for d in dirs:
        create_directory(d, dry_run)

def reorganize_tests(dry_run=False):
    """Reorganize tests"""
    log("=== Reorganizing Tests ===")
    
    # Move existing tests
    safe_move("tests", "tests/legacy/", dry_run)
    safe_move("benchmarks", "tests/benchmarks/", dry_run)
    
    # Create directories
    dirs = [
        "tests/unit",
        "tests/integration",
        "tests/e2e",
    ]
    
    for d in dirs:
        create_directory(d, dry_run)

def reorganize_docs(dry_run=False):
    """Reorganize documentation"""
    log("=== Reorganizing Documentation ===")
    
    # Move docs
    safe_move("docs", "docs/project/", dry_run)
    safe_move("tutorials", "docs/tutorials/", dry_run)
    safe_move("references", "docs/references/", dry_run)
    
    # Create directories
    dirs = [
        "docs/api",
        "docs/architecture",
    ]
    
    for d in dirs:
        create_directory(d, dry_run)

def reorganize_scripts(dry_run=False):
    """Reorganize scripts"""
    log("=== Reorganizing Scripts ===")
    
    # Move scripts
    safe_move("scripts", "scripts/tools/", dry_run)
    safe_move("generate_demo_data.py", "scripts/generate_demo_data.py", dry_run)

def reorganize_deploy(dry_run=False):
    """Reorganize deployment configs"""
    log("=== Reorganizing Deployment ===")
    
    # Move deployment configs
    safe_move("docker", "deploy/docker/", dry_run)
    safe_move(".github", "deploy/ci-cd/github/", dry_run)

def cleanup_empty_dirs(dry_run=False):
    """Clean up empty directories"""
    log("=== Cleaning Up Empty Directories ===")
    
    dirs_to_clean = [
        "example_basic",
        "example_advanced",
        "example_al",
        "example_ensemble",
        "example_monitoring",
        "high_throughput",
        "ml_potentials",
        "workflows/alloy",
        "nep_training",
    ]
    
    for d in dirs_to_clean:
        target = BASE_DIR / d
        if target.exists():
            if target.is_dir() and not any(target.iterdir()):
                safe_delete(d, dry_run)
            elif not target.is_dir():
                log(f"SKIP: {d} is not a directory")
            else:
                log(f"SKIP: {d} is not empty")

def create_new_readme(dry_run=False):
    """Create new main README"""
    if dry_run:
        log("DRY-RUN: Would create new README.md")
        return
    
    readme_content = '''# DFT-LAMMPS Research Platform

A comprehensive computational materials science platform integrating DFT calculations, molecular dynamics simulations, and machine learning potentials for high-throughput materials discovery.

## 🏗️ Architecture

```
dft_lammps_research/
├── core/                   # Core computational engine
│   ├── dft/               # DFT calculation modules
│   ├── md/                # Molecular dynamics modules
│   ├── ml/                # Machine learning potentials
│   └── common/            # Shared utilities
├── platform/              # Platform services
│   ├── api/               # REST API gateway
│   ├── web/               # Web interface
│   └── hpc/               # HPC connectors
├── intelligence/          # AI/ML intelligence layer
│   ├── active_learning/   # Active learning framework
│   ├── literature/        # Literature mining
│   ├── multi_agent/       # Multi-agent systems
│   └── auto_discovery/    # Automated discovery
├── simulation/            # Advanced simulation methods
│   ├── phase_field/       # Phase field simulations
│   ├── quantum/           # Quantum computing
│   └── rl/                # RL-based optimization
├── workflows/             # Application workflows
│   ├── battery/           # Battery materials
│   ├── catalyst/          # Catalyst design
│   └── applications/      # Other applications
├── validation/            # Validation framework
├── examples/              # Examples and templates
├── tests/                 # Test suite
├── docs/                  # Documentation
└── scripts/               # Utility scripts
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run basic workflow
python -m examples.tutorials.01_quick_start

# Launch web interface
cd platform/web/ui && npm install && npm run dev
```

## 📚 Documentation

- [User Guide](docs/project/README.md)
- [API Reference](docs/api/)
- [Architecture](docs/architecture/)
- [Tutorials](docs/tutorials/)

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run benchmarks
python -m tests.benchmarks.run_benchmarks
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

This project integrates multiple open-source tools including ASE, LAMMPS, VASP, DeepMD, and PyTorch.
'''
    
    readme_path = BASE_DIR / "README.md"
    
    # Backup old README
    if readme_path.exists():
        backup_path = BASE_DIR / "README.md.backup"
        shutil.copy(str(readme_path), str(backup_path))
        log(f"Backed up README.md to README.md.backup")
    
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    log("Created new README.md")

def main():
    """Main execution function"""
    dry_run = "--execute" not in sys.argv
    
    if dry_run:
        log("=" * 60)
        log("DRY RUN MODE - No changes will be made")
        log("Run with --execute to apply changes")
        log("=" * 60)
    else:
        log("=" * 60)
        log("EXECUTION MODE - Changes will be applied")
        log("=" * 60)
    
    # Clear log file
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    
    # Execute reorganization steps
    try:
        fix_nested_structure(dry_run)
        reorganize_core(dry_run)
        reorganize_platform(dry_run)
        reorganize_intelligence(dry_run)
        reorganize_simulation(dry_run)
        reorganize_workflows(dry_run)
        reorganize_examples(dry_run)
        reorganize_tests(dry_run)
        reorganize_docs(dry_run)
        reorganize_scripts(dry_run)
        reorganize_deploy(dry_run)
        cleanup_empty_dirs(dry_run)
        create_new_readme(dry_run)
        
        log("=" * 60)
        if dry_run:
            log("DRY RUN COMPLETED - No changes made")
            log("Run with --execute to apply changes")
        else:
            log("REORGANIZATION COMPLETED")
        log(f"See log at: {LOG_FILE}")
        log("=" * 60)
        
    except Exception as e:
        log(f"ERROR: {e}")
        import traceback
        log(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
