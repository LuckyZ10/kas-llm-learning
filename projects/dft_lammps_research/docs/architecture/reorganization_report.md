# DFT-LAMMPS Research Project Reorganization Report

Generated: 2026-03-12T08:35:36.224511

## File Statistics

| File Type | Count |
|-----------|-------|
| Python (.py) | 234 |
| TypeScript (.ts) | 6 |
| JavaScript (.js) | 3 |
| YAML (.yaml/.yml) | 12 |
| JSON (.json) | 7 |
| Markdown (.md) | 53 |
| Other | 43 |
| **Total** | **365** |

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

## Move Operations (31)

1. **core/dft_to_lammps_bridge.py** → **core/dft/bridge.py**
   - Description: DFT bridge module

2. **core/integrated_materials_workflow.py** → **core/common/workflow_engine.py**
   - Description: Integrated workflow engine

3. **core/common/checkpoint_manager.py** → **core/common/checkpoint.py**
   - Description: Checkpoint management

4. **core/common/parallel_optimizer.py** → **core/common/parallel.py**
   - Description: Parallel optimization utilities

5. **core/ml/nep_training_pipeline.py** → **core/ml/nep/pipeline.py**
   - Description: NEP training pipeline

6. **platform/monitoring_dashboard.py** → **platform/web/monitoring/dashboard.py**
   - Description: Monitoring dashboard

7. **platform/hpc/hpc_scheduler.py** → **platform/hpc/scheduler.py**
   - Description: HPC scheduler

8. **platform/api/api_platform** → **platform/api/**
   - Description: API platform (flatten)
   - Type: Directory

9. **platform/web/v2** → **platform/web/ui/**
   - Description: Web UI v2
   - Type: Directory

10. **intelligence/active_learning/v2** → **intelligence/active_learning/**
   - Description: Active learning v2 (flatten)
   - Type: Directory
   - Action: Merge

11. **intelligence/literature/literature_survey** → **intelligence/literature/**
   - Description: Literature survey (flatten)
   - Type: Directory
   - Action: Merge

12. **simulation/phase_field/v1** → **simulation/phase_field/**
   - Description: Phase field (flatten)
   - Type: Directory
   - Action: Merge

13. **simulation/rl/rl_optimizer** → **simulation/rl/optimizer/**
   - Description: RL optimizer
   - Type: Directory

14. **workflows/battery/solid_electrolyte** → **workflows/battery/**
   - Description: Battery workflow (flatten)
   - Type: Directory
   - Action: Merge

15. **workflows/battery/battery_screening_pipeline.py** → **workflows/battery/screening.py**
   - Description: Battery screening pipeline

16. **workflows/battery/Li3PS4_workflow_example.py** → **workflows/battery/examples/Li3PS4.py**
   - Description: Li3PS4 example

17. **workflows/catalyst/catalyst** → **workflows/catalyst/**
   - Description: Catalyst workflow (flatten)
   - Type: Directory
   - Action: Merge

18. **workflows/perovskite/perovskite** → **workflows/perovskite/**
   - Description: Perovskite workflow (flatten)
   - Type: Directory
   - Action: Merge

19. **workflows/screening_examples.py** → **workflows/examples/screening.py**
   - Description: Screening examples

20. **applications** → **workflows/applications/**
   - Description: Application cases
   - Type: Directory

21. **examples** → **examples/legacy/**
   - Description: Legacy examples
   - Type: Directory
   - Action: Merge

22. **code_templates** → **examples/templates/**
   - Description: Code templates
   - Type: Directory
   - Action: Merge

23. **tests** → **tests/legacy/**
   - Description: Legacy tests
   - Type: Directory
   - Action: Merge

24. **benchmarks** → **tests/benchmarks/**
   - Description: Benchmarks
   - Type: Directory
   - Action: Merge

25. **docs** → **docs/project/**
   - Description: Project docs
   - Type: Directory
   - Action: Merge

26. **tutorials** → **docs/tutorials/**
   - Description: Tutorials
   - Type: Directory
   - Action: Merge

27. **references** → **docs/references/**
   - Description: References
   - Type: Directory
   - Action: Merge

28. **scripts** → **scripts/tools/**
   - Description: Tool scripts
   - Type: Directory
   - Action: Merge

29. **generate_demo_data.py** → **scripts/generate_demo_data.py**
   - Description: Demo data generator

30. **docker** → **deploy/docker/**
   - Description: Docker deployment
   - Type: Directory
   - Action: Merge

31. **.github** → **deploy/ci-cd/github/**
   - Description: GitHub workflows
   - Type: Directory
   - Action: Merge


## Create Operations (25)

1. `core/dft/parsers/`
2. `core/dft/calculators/`
3. `core/md/engines/`
4. `core/md/analysis/`
5. `core/ml/deepmd/`
6. `core/ml/mace/`
7. `core/common/utils/`
8. `core/common/models/`
9. `platform/api/gateway/`
10. `platform/api/auth/`
11. `platform/api/integrations/`
12. `platform/hpc/connectors/`
13. `platform/hpc/monitoring/`
14. `platform/web/ui/`
15. `platform/web/monitoring/`
16. `intelligence/multi_agent/agents/`
17. `intelligence/multi_agent/orchestration/`
18. `intelligence/auto_discovery/`
19. `simulation/quantum/circuits/`
20. `validation/experimental/`
21. `validation/theoretical/`
22. `examples/tutorials/`
23. `examples/templates/`
24. `docs/api/`
25. `docs/architecture/`


## Delete Operations (11)

**WARNING:** These directories will be removed (after verifying they're empty or redundant):

1. `example_basic/`
2. `example_advanced/`
3. `example_al/`
4. `example_ensemble/`
5. `example_monitoring/`
6. `high_throughput/`
7. `ml_potentials/`
8. `workflows/alloy/`
9. `nep_training/`
10. `validation_results/`
11. `test_reports/`


## Next Steps

1. Review this report
2. Run the reorganization script with `--dry-run` to preview changes
3. Run the reorganization script with `--execute` to apply changes
4. Update import statements in moved files
5. Run tests to verify everything works
6. Update documentation

## Import Path Updates Required

After moving files, the following import patterns need to be updated:

- `from dft_to_lammps_bridge import ...` → `from core.dft.bridge import ...`
- `from integrated_materials_workflow import ...` → `from core.common.workflow_engine import ...`
- `from checkpoint_manager import ...` → `from core.common.checkpoint import ...`
- `from parallel_optimizer import ...` → `from core.common.parallel import ...`
- `from hpc_scheduler import ...` → `from platform.hpc.scheduler import ...`
- `from monitoring_dashboard import ...` → `from platform.web.monitoring.dashboard import ...`
