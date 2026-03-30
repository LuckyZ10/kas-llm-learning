"""
LLM Hypothesis Demo - Demonstrates LLM-assisted hypothesis generation and explanation

This example demonstrates how to use the LLM integration module for:
1. Generating hypotheses from computational results
2. Explaining numerical results in natural language
3. Testing hypotheses against evidence
4. Iterative hypothesis refinement
"""

import asyncio
import json
from typing import Dict, List, Any

# Import the LLM integration modules
try:
    from dftlammps.llm_integration import (
        UnifiedLLMInterface,
        LLMConfig,
        LLMProvider,
        HypothesisExplainer,
        ExplanationStyle,
        NumericalContext,
        NumericalValue,
        Hypothesis,
    )
    from dftlammps.llm_integration.scientific_reasoning import (
        ScientificReasoningEngine,
        Evidence,
        EvidenceType,
        ConfidenceLevel,
    )
except ImportError:
    # Allow standalone execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from llm_integration import (
        UnifiedLLMInterface,
        LLMConfig,
        LLMProvider,
        HypothesisExplainer,
        ExplanationStyle,
        NumericalContext,
        NumericalValue,
        Hypothesis,
    )
    from llm_integration.scientific_reasoning import (
        ScientificReasoningEngine,
        Evidence,
        EvidenceType,
        ConfidenceLevel,
    )


async def demo_band_gap_analysis():
    """Demonstrate hypothesis generation for band gap analysis."""
    print("=" * 70)
    print("DEMO 1: Band Gap Analysis with Hypothesis Generation")
    print("=" * 70)
    
    # Create LLM interface (will use environment variables)
    try:
        config = LLMConfig.from_env(LLMProvider.OPENAI)
    except ValueError:
        print("Note: OPENAI_API_KEY not set. Using mock responses for demo.")
        config = None
    
    if config:
        llm = UnifiedLLMInterface(config)
        explainer = HypothesisExplainer(llm)
    else:
        explainer = None
    
    # Set up numerical context for a semiconductor calculation
    context = NumericalContext(
        system_description="Titanium dioxide (TiO2) anatase phase with oxygen vacancy",
        calculation_method="DFT-PBE with Hubbard U correction (U=4.2 eV)",
        numerical_results={
            "band_gap": NumericalValue(
                value=3.21,
                unit="eV",
                uncertainty=0.05,
                source="DFT-PBE+U"
            ),
            "formation_energy": NumericalValue(
                value=4.85,
                unit="eV",
                source="DFT-PBE+U"
            ),
            "vacancy_concentration": NumericalValue(
                value=1e17,
                unit="cm^-3",
                source="thermodynamic calculation"
            ),
        },
        comparison_values={
            "experimental_band_gap": NumericalValue(
                value=3.20,
                unit="eV",
                source="Experimental"
            ),
            "pristine_band_gap": NumericalValue(
                value=3.45,
                unit="eV",
                source="DFT-PBE+U"
            ),
        },
        validation_metrics={
            "lattice_constant_error": 0.8,  # percent
            "band_gap_error": 0.3,  # percent
        }
    )
    
    print("\nNumerical Context:")
    print(context.to_prompt_context())
    
    if explainer:
        # Generate hypotheses based on results
        print("\n" + "-" * 50)
        print("Generating Hypotheses from Results...")
        print("-" * 50)
        
        hypotheses = await explainer.generate_hypothesis_from_results(
            context=context,
            domain="semiconductor"
        )
        
        print(f"\nGenerated {len(hypotheses)} hypotheses:")
        for i, h in enumerate(hypotheses, 1):
            print(f"\nHypothesis {i}:")
            print(f"  Statement: {h.statement}")
            print(f"  Expected: {h.expected_outcome}")
            print(f"  Criteria: {', '.join(h.test_criteria[:2])}...")
        
        # Test first hypothesis
        if hypotheses:
            print("\n" + "-" * 50)
            print("Testing First Hypothesis...")
            print("-" * 50)
            
            explanation = await explainer.explain_results(
                context=context,
                hypothesis=hypotheses[0],
                style=ExplanationStyle.HYPOTHESIS_DRIVEN,
                domain="semiconductor"
            )
            
            print(f"\nConfidence Score: {explanation.confidence_score:.1%}")
            print(f"\nKey Findings:")
            for finding in explanation.key_findings[:3]:
                print(f"  • {finding}")
            
            print(f"\nImplications:")
            for impl in explanation.implications[:2]:
                print(f"  • {impl}")
    else:
        print("\n(Mock output - no LLM configured)")
        print("Would generate hypotheses about band gap reduction mechanisms...")


async def demo_catalysis_results():
    """Demonstrate catalysis results explanation."""
    print("\n" + "=" * 70)
    print("DEMO 2: Catalysis Results Interpretation")
    print("=" * 70)
    
    # Reaction pathway data
    context = NumericalContext(
        system_description="CO oxidation on Pt(111) surface",
        calculation_method="DFT with NEB transition state search",
        numerical_results={
            "adsorption_energy_CO": NumericalValue(-1.85, "eV"),
            "adsorption_energy_O2": NumericalValue(-1.23, "eV"),
            "reaction_barrier": NumericalValue(0.72, "eV"),
            "reaction_energy": NumericalValue(-1.45, "eV"),
        },
        comparison_values={
            "literature_barrier": NumericalValue(0.68, "eV", source="Literature"),
            "experimental_turnover": NumericalValue(1000, "s^-1", source="Experiment"),
        }
    )
    
    print("\nNumerical Context:")
    print(context.to_prompt_context())
    
    try:
        config = LLMConfig.from_env(LLMProvider.OPENAI)
        llm = UnifiedLLMInterface(config)
        explainer = HypothesisExplainer(llm)
        
        print("\n" + "-" * 50)
        print("Generating Explanation...")
        print("-" * 50)
        
        # Generate explanation in different styles
        for style in [ExplanationStyle.TECHNICAL, ExplanationStyle.EXECUTIVE]:
            print(f"\n{style.name} Style:")
            explanation = await explainer.explain_results(
                context=context,
                style=style,
                domain="catalysis"
            )
            
            print(f"  Confidence: {explanation.confidence_score:.1%}")
            print(f"  Key Findings ({len(explanation.key_findings)}):")
            for finding in explanation.key_findings[:2]:
                print(f"    - {finding[:80]}...")
                
    except ValueError:
        print("\n(Mock output - no LLM configured)")
        print("Would analyze catalytic activity and rate-limiting steps...")


async def demo_battery_materials():
    """Demonstrate battery materials analysis."""
    print("\n" + "=" * 70)
    print("DEMO 3: Battery Materials Analysis")
    print("=" * 70)
    
    # Battery material calculation results
    context = NumericalContext(
        system_description="LiFePO4 cathode material with Mg doping",
        calculation_method="DFT with HSE06 functional",
        numerical_results={
            "voltage": NumericalValue(3.45, "V"),
            "capacity": NumericalValue(170, "mAh/g"),
            "diffusion_barrier_Li": NumericalValue(0.27, "eV"),
            "electronic_conductivity": NumericalValue(1e-9, "S/cm"),
        },
        comparison_values={
            "theoretical_capacity": NumericalValue(170, "mAh/g"),
            "undoped_barrier": NumericalValue(0.31, "eV"),
        }
    )
    
    print("\nNumerical Context:")
    print(context.to_prompt_context())
    
    # Create hypothesis
    hypothesis = Hypothesis(
        statement="Mg doping improves Li-ion diffusion in LiFePO4",
        expected_outcome="Lower diffusion barrier compared to undoped material",
        test_criteria=[
            "Diffusion barrier < 0.30 eV",
            "Voltage profile maintained",
            "Structural stability preserved"
        ],
        domain="battery"
    )
    
    print(f"\nHypothesis: {hypothesis.statement}")
    
    try:
        config = LLMConfig.from_env(LLMProvider.OPENAI)
        llm = UnifiedLLMInterface(config)
        explainer = HypothesisExplainer(llm)
        
        explanation = await explainer.explain_results(
            context=context,
            hypothesis=hypothesis,
            style=ExplanationStyle.HYPOTHESIS_DRIVEN,
            domain="battery"
        )
        
        print(f"\nExplanation (excerpt):")
        print(f"{explanation.explanation[:300]}...")
        
        print(f"\nVerdict Indicators:")
        for finding in explanation.key_findings:
            if any(word in finding.lower() for word in ["support", "confirm", "refute", "evidence"]):
                print(f"  • {finding}")
                
    except ValueError:
        print("\n(Mock output - no LLM configured)")
        print("Would evaluate hypothesis against computed results...")


async def demo_comparison_analysis():
    """Demonstrate comparison of multiple calculations."""
    print("\n" + "=" * 70)
    print("DEMO 4: Multi-Calculation Comparison")
    print("=" * 70)
    
    # Multiple calculation contexts
    contexts = [
        NumericalContext(
            system_description="PBE functional calculation",
            calculation_method="DFT-PBE",
            numerical_results={
                "band_gap": NumericalValue(2.1, "eV"),
                "lattice_a": NumericalValue(3.89, "Å"),
            }
        ),
        NumericalContext(
            system_description="PBE+U calculation (U=4.0 eV)",
            calculation_method="DFT-PBE+U",
            numerical_results={
                "band_gap": NumericalValue(3.2, "eV"),
                "lattice_a": NumericalValue(3.91, "Å"),
            }
        ),
        NumericalContext(
            system_description="HSE06 calculation",
            calculation_method="DFT-HSE06",
            numerical_results={
                "band_gap": NumericalValue(3.5, "eV"),
                "lattice_a": NumericalValue(3.90, "Å"),
            }
        ),
    ]
    
    print(f"\nComparing {len(contexts)} calculation methods")
    
    try:
        config = LLMConfig.from_env(LLMProvider.OPENAI)
        llm = UnifiedLLMInterface(config)
        explainer = HypothesisExplainer(llm)
        
        comparison = await explainer.compare_calculations(
            contexts=contexts,
            comparison_focus="functional_performance",
            style=ExplanationStyle.METHODS_FOCUS
        )
        
        print(f"\nComparison Results:")
        print(f"Confidence: {comparison.confidence_score:.1%}")
        print(f"\nKey Findings:")
        for finding in comparison.key_findings:
            print(f"  • {finding}")
            
    except ValueError:
        print("\n(Mock output - no LLM configured)")
        print("Would compare functional performance for band gap prediction...")


async def demo_reasoning_chain():
    """Demonstrate scientific reasoning chain construction."""
    print("\n" + "=" * 70)
    print("DEMO 5: Scientific Reasoning Chain")
    print("=" * 70)
    
    try:
        config = LLMConfig.from_env(LLMProvider.OPENAI)
        llm = UnifiedLLMInterface(config)
        reasoning = ScientificReasoningEngine(llm)
        
        # Create evidence
        evidence = [
            Evidence(
                source="DFT Calculation",
                type=EvidenceType.COMPUTATIONAL,
                description="Calculated binding energy of -1.2 eV for CO on Pt",
                confidence=ConfidenceLevel.HIGH,
            ),
            Evidence(
                source="Experiment",
                type=EvidenceType.EXPERIMENTAL,
                description="Measured turn-over frequency of 1000 s^-1 at 300K",
                confidence=ConfidenceLevel.HIGH,
            ),
            Evidence(
                source="Literature",
                type=EvidenceType.LITERATURE,
                description="Similar systems show barrier of ~0.7 eV",
                confidence=ConfidenceLevel.MODERATE,
            ),
        ]
        
        print("\nEvidence:")
        for e in evidence:
            print(f"  • {e.type.value}: {e.description}")
        
        # Build reasoning chain
        chain = await reasoning.construct_reasoning_chain(
            goal="Determine if Pt is an effective catalyst for CO oxidation at room temperature",
            evidence=evidence,
            constraints=[
                "Must have barrier < 0.8 eV for room temperature activity",
                "Binding energy should be neither too strong nor too weak"
            ]
        )
        
        print(f"\nReasoning Chain ({len(chain.steps)} steps):")
        for step in chain.steps:
            print(f"\n  Step {step.step_number}:")
            print(f"    Assertion: {step.assertion}")
            print(f"    Confidence: {step.confidence:.1%}")
        
        print(f"\nConclusion: {chain.conclusion}")
        print(f"Overall Confidence: {chain.overall_confidence:.1%}")
        
        if chain.gaps:
            print(f"\nKnowledge Gaps:")
            for gap in chain.gaps:
                print(f"  • {gap}")
                
    except ValueError:
        print("\n(Mock output - no LLM configured)")
        print("Would construct reasoning chain from evidence...")


def print_summary():
    """Print summary of the demo."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
This demo showcased the LLM hypothesis integration capabilities:

1. BAND GAP ANALYSIS
   - Automatic hypothesis generation from DFT results
   - Testing hypotheses against numerical evidence
   - Multi-style explanations (technical/executive)

2. CATALYSIS RESULTS
   - Interpretation of reaction energetics
   - Comparison with literature values
   - Activity predictions

3. BATTERY MATERIALS
   - Hypothesis-driven analysis workflow
   - Property-performance correlations
   - Doping effect evaluation

4. COMPARISON ANALYSIS
   - Multi-method comparison
   - Method validation
   - Systematic benchmarking

5. REASONING CHAIN
   - Evidence-based reasoning
   - Confidence tracking
   - Knowledge gap identification

Key Features Demonstrated:
✓ Few-shot prompting for hypothesis generation
✓ Chain-of-thought for reasoning transparency
✓ Multi-style explanations for different audiences
✓ Evidence aggregation and weighting
✓ Structured JSON output parsing

To run with actual LLM responses:
  export OPENAI_API_KEY=your_key
  python llm_hypothesis_demo.py
""")


async def main():
    """Main demo function."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         LLM Hypothesis Generation and Explanation Demo              ║
║                                                                      ║
║  This demo shows how to use LLMs for scientific hypothesis           ║
║  generation, explanation, and reasoning from computational data.     ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    await demo_band_gap_analysis()
    await demo_catalysis_results()
    await demo_battery_materials()
    await demo_comparison_analysis()
    await demo_reasoning_chain()
    
    print_summary()


if __name__ == "__main__":
    asyncio.run(main())
