# ASDE - Automatic Scientific Discovery Engine

A comprehensive framework for automated scientific hypothesis generation, experimental planning, result analysis, and paper writing.

## Overview

ASDE implements a complete closed-loop scientific discovery pipeline:

1. **Knowledge Graph Construction** - Extract entities and relations from scientific literature
2. **Hypothesis Generation** - Generate novel hypotheses using causal reasoning and analogical inference
3. **Experimental Planning** - Optimize experiments using Bayesian optimization and active learning
4. **Result Analysis** - Perform rigorous statistical testing with effect size calculations
5. **Paper Generation** - Automatically generate scientific papers in Markdown and LaTeX formats

## Project Structure

```
dftlammps/asde/
├── __init__.py                      # Package initialization
├── hypothesis_generator.py          # Hypothesis generation strategies
├── experiment_planner.py            # Bayesian optimization & active learning
├── result_analyzer.py               # Statistical analysis & effect sizes
├── paper_writer.py                  # Automatic paper generation
├── requirements.txt                 # Python dependencies
├── test_system.py                   # System verification
└── knowledge_graph/
    ├── __init__.py
    ├── scientific_kg.py             # Knowledge graph & entity extraction
    ├── literature_miner.py          # arXiv/PubMed/CrossRef integration
    └── citation_network.py          # Citation network analysis

dftlammps/asde_examples/
├── __init__.py
├── autonomous_discovery.py          # Complete discovery pipeline demo
└── literature_review_bot.py         # Automated literature review
```

## Installation

```bash
pip install numpy scipy networkx aiohttp statsmodels scikit-learn
```

## Quick Start

### 1. Autonomous Materials Discovery

```python
from dftlammps.asde_examples.autonomous_discovery import autonomous_discovery_pipeline

# Run the complete pipeline
results = await autonomous_discovery_pipeline(
    research_topic="catalyst optimization",
    n_experiments=15,
    output_dir="./output"
)
```

### 2. Literature Review Bot

```python
from dftlammps.asde_examples.literature_review_bot import run_literature_review

# Generate automated literature review
results = await run_literature_review(
    topic="machine learning materials discovery",
    keywords=["neural networks", "DFT", "screening"],
    max_papers=50
)
```

### 3. Individual Components

```python
from dftlammps.asde import (
    HypothesisGenerator,
    ExperimentPlanner,
    ResultAnalyzer,
    PaperWriter,
)
from dftlammps.asde.knowledge_graph import ScientificKnowledgeGraph

# Build knowledge graph
kg = ScientificKnowledgeGraph()
kg.extract_from_text("Graphene has high thermal conductivity.")

# Generate hypotheses
generator = HypothesisGenerator(kg)
hypotheses = generator.generate(max_hypotheses=5)

# Plan experiments
planner = ExperimentPlanner(
    variables=[...],
    outcome_name="yield",
    maximize=True
)
next_exp = planner.suggest_next_experiment()

# Analyze results
analyzer = ResultAnalyzer()
result = analyzer.t_test(group1, group2)

# Write paper
writer = PaperWriter(title="...", authors=[...])
paper = writer.write_paper()
paper.save("paper.md", format="markdown")
```

## Core Components

### Hypothesis Generator

Implements multiple hypothesis generation strategies:
- **Causal Chain Strategy** - Explore causal paths in knowledge graph
- **Analogical Reasoning** - Transfer knowledge between similar materials
- **Compositional Strategy** - Combine methods and materials
- **Abductive Inference** - Infer best explanations
- **Gap-Driven Strategy** - Identify knowledge gaps

### Experiment Planner

Features:
- Bayesian optimization with Gaussian Process surrogate
- Multiple acquisition functions (Expected Improvement, UCB, PI)
- Latin hypercube sampling for initial design
- Active learning for efficient exploration

### Result Analyzer

Statistical capabilities:
- Parametric tests (t-test, ANOVA, correlation)
- Non-parametric tests (Mann-Whitney U, Wilcoxon)
- Effect sizes (Cohen's d, Cliff's delta, eta-squared)
- Power analysis
- Normality and homoscedasticity tests

### Paper Writer

Automatic generation of:
- Abstract
- Introduction with related work
- Methods section
- Results with statistical summaries
- Discussion with implications and limitations
- Conclusion
- References

Output formats: Markdown, LaTeX, JSON

### Knowledge Graph

Features:
- Entity extraction (materials, properties, methods, phenomena)
- Relation extraction (causal, correlational, compositional)
- Graph algorithms (shortest path, community detection, PageRank)
- Causal reasoning
- Integration with literature databases

### Literature Miner

Supports:
- arXiv API (physics, math, CS papers)
- PubMed/NCBI (biomedical literature)
- CrossRef (general scientific literature)
- Automatic deduplication
- Citation network construction

## Example Output

### Autonomous Discovery Pipeline

```
======================================================================
AUTONOMOUS SCIENTIFIC DISCOVERY ENGINE
======================================================================

STAGE 1: Knowledge Graph Construction
  Total entities: 25
  Total relations: 18
  
STAGE 2: Hypothesis Generation
  Generated 8 hypotheses:
    1. [MECHANISTIC] Palladium affects yield through temperature mediation
    2. [ANALOGICAL] Since Pt and Pd share properties, Pt may exhibit...
    ...

STAGE 3: Experimental Planning
  Phase 1: Initial Exploration (5 experiments)
  Phase 2: Bayesian Optimization (10 experiments)
  Best yield: 92.4%
  Optimal conditions: T=560K, P=42atm, Cat=Pd

STAGE 4: Statistical Analysis
  ANOVA: F=8.42, p=0.003 (significant)
  Effect size: η²=0.45 (large)

STAGE 5: Paper Generation
  Generated autonomous_discovery_paper.md
  Word count: 2,450
```

## Statistics

- **Total Lines of Code**: 6,199
- **Modules**: 12 Python files
- **Type Annotations**: Full type hints throughout
- **Test Coverage**: Demo scripts included

## References

This system implements concepts from:

- Bayesian Optimization: Jones et al. (1998), Frazier (2018)
- Active Learning: Settles (2009)
- Causal Reasoning: Pearl (2009)
- Scientific Discovery: Sparkes et al. (2010), King et al. (2009)

## License

MIT License - Research and educational use encouraged.

## Citation

If you use ASDE in your research, please cite:

```
@software{asde2024,
  title={ASDE: Automatic Scientific Discovery Engine},
  author={ASDE Development Team},
  year={2024}
}
```
