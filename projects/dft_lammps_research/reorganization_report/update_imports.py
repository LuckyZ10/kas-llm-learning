#!/usr/bin/env python3
"""
Update import statements after reorganization
更新导入路径脚本
"""

import os
import re
from pathlib import Path

BASE_DIR = Path("/root/.openclaw/workspace/dft_lammps_research")

# Define import mapping: old pattern -> new pattern
IMPORT_MAPPINGS = {
    # Core module imports
    r'from\s+dft_to_lammps_bridge\s+import': 'from core.dft.bridge import',
    r'import\s+dft_to_lammps_bridge': 'import core.dft.bridge as dft_to_lammps_bridge',
    
    r'from\s+integrated_materials_workflow\s+import': 'from core.common.workflow_engine import',
    r'import\s+integrated_materials_workflow': 'import core.common.workflow_engine as integrated_materials_workflow',
    
    r'from\s+checkpoint_manager\s+import': 'from core.common.checkpoint import',
    r'import\s+checkpoint_manager': 'import core.common.checkpoint as checkpoint_manager',
    
    r'from\s+parallel_optimizer\s+import': 'from core.common.parallel import',
    r'import\s+parallel_optimizer': 'import core.common.parallel as parallel_optimizer',
    
    r'from\s+nep_training_pipeline\s+import': 'from core.ml.nep.pipeline import',
    r'import\s+nep_training_pipeline': 'import core.ml.nep.pipeline as nep_training_pipeline',
    
    # Platform imports
    r'from\s+hpc_scheduler\s+import': 'from platform.hpc.scheduler import',
    r'import\s+hpc_scheduler': 'import platform.hpc.scheduler as hpc_scheduler',
    
    r'from\s+monitoring_dashboard\s+import': 'from platform.web.monitoring.dashboard import',
    r'import\s+monitoring_dashboard': 'import platform.web.monitoring.dashboard as monitoring_dashboard',
    
    # Workflow imports
    r'from\s+battery_screening_pipeline\s+import': 'from workflows.battery.screening import',
    r'import\s+battery_screening_pipeline': 'import workflows.battery.screening as battery_screening_pipeline',
    
    # Old relative imports that may need updating
    r'from\s+\.\.dft_to_lammps_bridge\s+import': 'from core.dft.bridge import',
    r'from\s+\.\.battery_screening_pipeline\s+import': 'from workflows.battery.screening import',
}

def update_file_imports(filepath):
    """Update imports in a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        for old_pattern, new_pattern in IMPORT_MAPPINGS.items():
            content = re.sub(old_pattern, new_pattern, content)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error updating {filepath}: {e}")
        return False

def scan_and_update():
    """Scan all Python files and update imports"""
    updated_files = []
    
    for py_file in BASE_DIR.rglob("*.py"):
        # Skip __pycache__ and reorganization_report
        if '__pycache__' in str(py_file) or 'reorganization_report' in str(py_file):
            continue
        
        if update_file_imports(py_file):
            updated_files.append(str(py_file.relative_to(BASE_DIR)))
    
    return updated_files

def main():
    print("Updating import statements...")
    updated = scan_and_update()
    print(f"Updated {len(updated)} files:")
    for f in updated:
        print(f"  - {f}")
    
    # Save report
    report_path = BASE_DIR / "reorganization_report" / "import_updates.txt"
    with open(report_path, 'w') as f:
        f.write("Files with updated imports:\n")
        f.write("\n".join(updated))
    
    print(f"\nReport saved to: {report_path}")

if __name__ == "__main__":
    main()
