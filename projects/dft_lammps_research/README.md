# DFT-LAMMPS Research Platform

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
