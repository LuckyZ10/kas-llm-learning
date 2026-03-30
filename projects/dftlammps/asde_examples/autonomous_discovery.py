#!/usr/bin/env python3
"""
Autonomous Discovery Example

Demonstrates the complete ASDE pipeline for autonomous materials discovery:
1. Knowledge graph construction from literature
2. Hypothesis generation
3. Experimental planning with Bayesian optimization
4. Result analysis and statistical validation
5. Automatic paper generation
"""

from __future__ import annotations

import asyncio
import json
import random
from datetime import datetime
from typing import Any

import numpy as np

# Import ASDE components
from dftlammps.asde import (
    HypothesisGenerator,
    ExperimentPlanner,
    HypothesisTester,
    ResultAnalyzer,
    PaperWriter,
    ExperimentalVariable,
    VariableType,
    ExperimentResult,
)
from dftlammps.asde.knowledge_graph import (
    ScientificKnowledgeGraph,
    LiteratureMiner,
    CitationNetwork,
    build_network_from_search_results,
)


class SimulatedExperimentRunner:
    """
    Simulates running experiments for demo purposes.
    In production, this would interface with real experimental equipment.
    """
    
    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.RandomState(seed)
        
        # Define ground truth: true optimal conditions
        self.true_optimal = {
            "temperature": 550,
            "pressure": 45,
            "catalyst": "Pd"
        }
    
    def run(self, condition: dict[str, Any]) -> float:
        """
        Simulate an experiment and return outcome.
        
        Ground truth model:
        - Optimal at T=550K, P=45atm, catalyst=Pd
        - Yield decreases quadratically from optimum
        - Catalysts ranked: Pd > Pt > Ni > Cu
        """
        temp = condition.get("temperature", 500)
        press = condition.get("pressure", 50)
        catalyst = condition.get("catalyst", "Ni")
        
        # Base yield from temperature and pressure (quadratic)
        temp_factor = -0.001 * (temp - 550)**2
        press_factor = -0.01 * (press - 45)**2
        
        # Catalyst factor
        catalyst_bonus = {
            "Pd": 15,
            "Pt": 10,
            "Ni": 5,
            "Cu": 0
        }.get(catalyst, 0)
        
        # Calculate yield (0-100 scale)
        base_yield = 70
        yield_value = base_yield + temp_factor + press_factor + catalyst_bonus
        
        # Add experimental noise
        noise = self.rng.normal(0, 3)
        yield_value += noise
        
        # Clip to valid range
        return float(np.clip(yield_value, 0, 100))


async def autonomous_discovery_pipeline(
    research_topic: str = "catalyst optimization",
    n_experiments: int = 15,
    output_dir: str = "/root/.openclaw/workspace/dftlammps/asde_examples/output"
) -> dict[str, Any]:
    """
    Run the complete autonomous discovery pipeline.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("AUTONOMOUS SCIENTIFIC DISCOVERY ENGINE")
    print("=" * 70)
    print(f"\nResearch Topic: {research_topic}")
    print(f"Planned Experiments: {n_experiments}")
    print()
    
    results = {
        "research_topic": research_topic,
        "start_time": datetime.now().isoformat(),
        "stages": {}
    }
    
    # =========================================================================
    # STAGE 1: Knowledge Graph Construction
    # =========================================================================
    print("\n" + "=" * 70)
    print("STAGE 1: Knowledge Graph Construction")
    print("=" * 70)
    
    kg = ScientificKnowledgeGraph()
    
    # Populate knowledge graph from sample texts
    sample_texts = [
        "Palladium catalysts exhibit high activity in hydrogenation reactions.",
        "Reaction temperature significantly affects catalyst performance.",
        "Optimal pressure depends on the catalyst material used.",
        "Platinum catalysts show moderate activity with good selectivity.",
        "Nickel catalysts are cost-effective for industrial applications.",
        "Catalyst deactivation occurs at temperatures above 600K.",
        "Molecular dynamics simulations predict adsorption energies.",
        "DFT calculations reveal electronic structure of catalyst surfaces.",
        "Phase transitions in catalysts affect active site availability.",
        "Temperature causes changes in reaction kinetics and thermodynamics.",
        "Pressure affects equilibrium conversion in gas-phase reactions.",
        "Selectivity depends on catalyst pore structure and surface area.",
    ]
    
    for i, text in enumerate(sample_texts):
        kg.extract_from_text(text, source=f"sample_{i}")
    
    kg_stats = kg.get_statistics()
    print(f"\nKnowledge Graph Statistics:")
    for key, value in kg_stats.items():
        print(f"  {key}: {value}")
    
    # Save knowledge graph
    kg.save(f"{output_dir}/knowledge_graph.json")
    print(f"\nKnowledge graph saved to {output_dir}/knowledge_graph.json")
    
    results["stages"]["knowledge_graph"] = {
        "entities": len(kg._entities),
        "relations": len(kg._relations),
        "statistics": kg_stats
    }
    
    # =========================================================================
    # STAGE 2: Hypothesis Generation
    # =========================================================================
    print("\n" + "=" * 70)
    print("STAGE 2: Hypothesis Generation")
    print("=" * 70)
    
    hypothesis_generator = HypothesisGenerator(kg)
    hypotheses = hypothesis_generator.generate(
        max_hypotheses=8,
        min_confidence=0.3,
        diversity_weight=0.4
    )
    
    print(f"\nGenerated {len(hypotheses)} hypotheses:\n")
    for i, h in enumerate(hypotheses, 1):
        print(f"{i}. [{h.hypothesis_type.name}] (confidence: {h.confidence:.2f})")
        print(f"   Statement: {h.statement}")
        print(f"   Strategy: {h.source_strategy}")
        print(f"   Testable predictions:")
        for pred in h.testable_predictions[:2]:
            print(f"      - {pred}")
        print()
    
    # Rank hypotheses
    ranked = hypothesis_generator.rank_hypotheses(hypotheses)
    print("Top 3 Ranked Hypotheses:")
    for h, score in ranked[:3]:
        print(f"  - {h.statement[:60]}... (score: {score:.3f})")
    
    # Select top hypotheses for testing
    hypotheses_to_test = [h for h, _ in ranked[:3]]
    
    results["stages"]["hypothesis_generation"] = {
        "total_generated": len(hypotheses),
        "selected_for_testing": len(hypotheses_to_test),
        "hypotheses": [h.to_dict() for h in hypotheses_to_test]
    }
    
    # =========================================================================
    # STAGE 3: Experimental Planning
    # =========================================================================
    print("\n" + "=" * 70)
    print("STAGE 3: Experimental Planning (Bayesian Optimization)")
    print("=" * 70)
    
    # Define experimental variables
    variables = [
        ExperimentalVariable(
            name="temperature",
            var_type=VariableType.CONTINUOUS,
            lower_bound=300,
            upper_bound=700,
            unit="K",
            description="Reaction temperature"
        ),
        ExperimentalVariable(
            name="pressure",
            var_type=VariableType.CONTINUOUS,
            lower_bound=1,
            upper_bound=100,
            unit="atm",
            description="Reaction pressure"
        ),
        ExperimentalVariable(
            name="catalyst",
            var_type=VariableType.CATEGORICAL,
            categories=["Pd", "Pt", "Ni", "Cu"],
            description="Catalyst material"
        ),
    ]
    
    # Create experiment planner
    planner = ExperimentPlanner(
        variables=variables,
        outcome_name="catalytic_yield",
        maximize=True,
        random_seed=42
    )
    
    # Create hypothesis tester
    tester = HypothesisTester(planner)
    
    # Generate validation experiments for hypotheses
    print("\nDesigning experiments for hypothesis validation...")
    for h in hypotheses_to_test:
        validation_exps = tester.design_validation_experiments(h, n_experiments=2)
        print(f"  {h.id}: {len(validation_exps)} validation experiments designed")
    
    # Run optimization
    print("\nRunning Bayesian optimization...")
    print("-" * 50)
    
    experiment_runner = SimulatedExperimentRunner(seed=42)
    experimental_results: list[ExperimentResult] = []
    
    # Initial experiments (Latin Hypercube)
    n_initial = 5
    initial_conditions = planner.suggest_initial_points(
        n_points=n_initial,
        method="latin_hypercube"
    )
    
    print(f"\nPhase 1: Initial Exploration ({n_initial} experiments)")
    for i, cond in enumerate(initial_conditions):
        outcome = experiment_runner.run(cond.values)
        result = ExperimentResult(
            condition_id=cond.id,
            values=cond.values,
            outcome=outcome,
            outcome_name="catalytic_yield",
            metadata={"phase": "initial", "iteration": i}
        )
        planner.update_with_result(result)
        experimental_results.append(result)
        print(f"  Exp {i+1:2d}: T={cond.values['temperature']:5.1f}K, "
              f"P={cond.values['pressure']:4.1f}atm, "
              f"Cat={cond.values['catalyst']:2s} → Yield={outcome:5.1f}%")
    
    # Optimization phase
    n_optimization = n_experiments - n_initial
    print(f"\nPhase 2: Bayesian Optimization ({n_optimization} experiments)")
    
    for i in range(n_optimization):
        next_exps = planner.suggest_next_experiment(batch_size=1)
        cond = next_exps[0]
        
        outcome = experiment_runner.run(cond.values)
        result = ExperimentResult(
            condition_id=cond.id,
            values=cond.values,
            outcome=outcome,
            outcome_name="catalytic_yield",
            metadata={
                "phase": "optimization",
                "iteration": i + n_initial,
                "predicted": cond.predicted_outcome,
                "uncertainty": cond.uncertainty,
                "acquisition_score": cond.acquisition_score
            }
        )
        planner.update_with_result(result)
        experimental_results.append(result)
        
        print(f"  Exp {i+n_initial+1:2d}: T={cond.values['temperature']:5.1f}K, "
              f"P={cond.values['pressure']:4.1f}atm, "
              f"Cat={cond.values['catalyst']:2s} → Yield={outcome:5.1f}% "
              f"(pred: {cond.predicted_outcome:.1f}±{cond.uncertainty:.1f})")
    
    # Get best result
    best_result = planner.get_best_observed()
    convergence_stats = planner.get_convergence_stats()
    
    print("\n" + "-" * 50)
    print("Optimization Complete!")
    print(f"  Best yield: {best_result.outcome:.2f}%")
    print(f"  Optimal conditions: T={best_result.values['temperature']:.1f}K, "
          f"P={best_result.values['pressure']:.1f}atm, "
          f"Cat={best_result.values['catalyst']}")
    print(f"  Improvements found: {convergence_stats['n_improvements']}")
    
    results["stages"]["experimental_planning"] = {
        "total_experiments": len(experimental_results),
        "best_yield": best_result.outcome if best_result else None,
        "optimal_conditions": best_result.values if best_result else None,
        "convergence": convergence_stats
    }
    
    # =========================================================================
    # STAGE 4: Result Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("STAGE 4: Statistical Analysis")
    print("=" * 70)
    
    analyzer = ResultAnalyzer(alpha=0.05)
    
    # Group results by catalyst
    catalyst_groups: dict[str, list[float]] = {"Pd": [], "Pt": [], "Ni": [], "Cu": []}
    for r in experimental_results:
        cat = r.values.get("catalyst", "unknown")
        if cat in catalyst_groups:
            catalyst_groups[cat].append(r.outcome)
    
    # Compare catalysts
    print("\nCatalyst Performance Comparison:")
    for cat, yields in catalyst_groups.items():
        if yields:
            summary = analyzer.summarize(np.array(yields))
            print(f"  {cat}: {summary.mean:.2f} ± {summary.std:.2f}% (n={summary.n})")
    
    # ANOVA across catalysts
    groups_for_anova = [np.array(y) for y in catalyst_groups.values() if y]
    group_names = [cat for cat, y in catalyst_groups.items() if y]
    
    anova_result = analyzer.anova(*groups_for_anova)
    print(f"\nANOVA Results:")
    print(f"  F-statistic: {anova_result.statistic:.4f}")
    print(f"  p-value: {anova_result.p_value:.4f}")
    print(f"  Significant: {anova_result.significant}")
    print(f"  Effect size ({anova_result.effect_size.measure}): "
          f"{anova_result.effect_size.value:.3f} ({anova_result.effect_size.interpretation})")
    
    # Correlation analysis: Temperature vs Yield
    temps = np.array([r.values["temperature"] for r in experimental_results])
    yields = np.array([r.outcome for r in experimental_results])
    
    corr_result = analyzer.correlation(temps, yields, method="pearson")
    print(f"\nTemperature-Yield Correlation:")
    print(f"  Pearson r: {corr_result.statistic:.4f}")
    print(f"  p-value: {corr_result.p_value:.4f}")
    print(f"  Interpretation: {corr_result.effect_size.interpretation}")
    
    # Comprehensive analysis
    comprehensive = analyzer.comprehensive_analysis(
        *groups_for_anova,
        group_names=group_names
    )
    
    results["stages"]["result_analysis"] = {
        "anova": anova_result.to_dict(),
        "correlation": corr_result.to_dict(),
        "catalyst_summaries": {
            cat: analyzer.summarize(np.array(y)).to_dict()
            for cat, y in catalyst_groups.items() if y
        }
    }
    
    # =========================================================================
    # STAGE 5: Paper Generation
    # =========================================================================
    print("\n" + "=" * 70)
    print("STAGE 5: Automatic Paper Generation")
    print("=" * 70)
    
    writer = PaperWriter(
        title=f"Autonomous Discovery of Optimal Catalysts: A Bayesian Optimization Study",
        authors=["ASDE Autonomous System", "AI Research Team"],
        keywords=["catalyst optimization", "Bayesian optimization", "active learning", "materials discovery"]
    )
    
    writer.add_hypotheses(hypotheses_to_test)
    
    # Convert results to dict format
    result_dicts = [r.to_dict() for r in experimental_results]
    writer.add_experimental_results(
        result_dicts,
        {
            "design_type": "Bayesian optimization with Gaussian Process surrogate",
            "n_variables": len(variables),
            "n_conditions": len(experimental_results),
            "variables": [
                {"name": v.name, "description": v.description, "type": v.var_type.name}
                for v in variables
            ]
        }
    )
    
    analysis_dicts = [
        anova_result.to_dict(),
        corr_result.to_dict()
    ]
    writer.add_analysis_results(analysis_dicts)
    
    # Generate paper
    paper = writer.write_paper()
    
    # Save in multiple formats
    paper.save(f"{output_dir}/autonomous_discovery_paper.md", format="markdown")
    paper.save(f"{output_dir}/autonomous_discovery_paper.tex", format="latex")
    paper.save(f"{output_dir}/autonomous_discovery_paper.json", format="json")
    
    print(f"\nPaper generated and saved:")
    print(f"  - Markdown: {output_dir}/autonomous_discovery_paper.md")
    print(f"  - LaTeX: {output_dir}/autonomous_discovery_paper.tex")
    print(f"  - JSON: {output_dir}/autonomous_discovery_paper.json")
    
    print(f"\nPaper Statistics:")
    print(f"  Title: {paper.title}")
    print(f"  Sections: {len(paper.sections)}")
    print(f"  Abstract: {len(paper.abstract)} characters")
    
    results["stages"]["paper_generation"] = {
        "title": paper.title,
        "sections": [s.title for s in paper.sections],
        "word_count": len(paper.to_markdown().split()),
        "files_generated": [
            "autonomous_discovery_paper.md",
            "autonomous_discovery_paper.tex",
            "autonomous_discovery_paper.json"
        ]
    }
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    
    results["end_time"] = datetime.now().isoformat()
    
    # Save full results
    with open(f"{output_dir}/pipeline_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFull results saved to: {output_dir}/pipeline_results.json")
    print("\nKey Findings:")
    print(f"  - Optimal catalyst: {best_result.values['catalyst'] if best_result else 'N/A'}")
    print(f"  - Maximum yield: {best_result.outcome:.2f}%" if best_result else "N/A")
    print(f"  - Catalyst effect significant: {anova_result.significant}")
    print(f"  - Temperature correlation: {corr_result.statistic:.3f}")
    
    return results


def main():
    """Run the autonomous discovery pipeline."""
    asyncio.run(autonomous_discovery_pipeline())


if __name__ == "__main__":
    main()
