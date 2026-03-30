"""
Scientific Reasoning Engine Module

Provides advanced reasoning capabilities for scientific analysis including:
- Causal inference from computational data
- Counterfactual analysis (what-if scenarios)
- Reasoning chain construction and validation
- Evidence aggregation and weighting
- Hypothesis testing with uncertainty quantification

This module enables the LLM to perform structured scientific reasoning
beyond simple pattern matching.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
from collections import defaultdict
import asyncio
from abc import ABC, abstractmethod

from .llm_interface import (
    UnifiedLLMInterface,
    LLMConfig,
    LLMProvider,
    CompletionResponse,
)


class EvidenceType(Enum):
    """Types of evidence in scientific reasoning."""
    COMPUTATIONAL = "computational"  # DFT, MD, etc.
    EXPERIMENTAL = "experimental"    # Lab measurements
    THEORETICAL = "theoretical"      # Analytical models
    EMPIRICAL = "empirical"          # Previous observations
    LITERATURE = "literature"        # Published results


class ConfidenceLevel(Enum):
    """Confidence levels for assertions."""
    VERY_HIGH = 0.95
    HIGH = 0.80
    MODERATE = 0.60
    LOW = 0.40
    VERY_LOW = 0.20
    UNCERTAIN = 0.0


@dataclass
class Evidence:
    """A piece of evidence supporting or refuting a claim."""
    source: str
    type: EvidenceType
    description: str
    value: Optional[float] = None
    uncertainty: Optional[float] = None
    confidence: ConfidenceLevel = ConfidenceLevel.MODERATE
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "type": self.type.value,
            "description": self.description,
            "value": self.value,
            "uncertainty": self.uncertainty,
            "confidence": self.confidence.name,
            "metadata": self.metadata,
        }


@dataclass
class CausalFactor:
    """A factor in causal analysis."""
    name: str
    description: str
    type: str  # "intervention", "confounder", "mediator", "collider"
    evidence: List[Evidence] = field(default_factory=list)
    
    def add_evidence(self, evidence: Evidence) -> None:
        """Add evidence for this factor."""
        self.evidence.append(evidence)


@dataclass
class CausalRelationship:
    """A causal relationship between variables."""
    cause: str
    effect: str
    strength: float  # -1 to 1
    mechanism: Optional[str] = None
    evidence: List[Evidence] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cause": self.cause,
            "effect": self.effect,
            "strength": self.strength,
            "mechanism": self.mechanism,
            "evidence": [e.to_dict() for e in self.evidence],
            "assumptions": self.assumptions,
        }


@dataclass
class CounterfactualScenario:
    """A counterfactual scenario for analysis."""
    name: str
    description: str
    intervention: Dict[str, Any]  # What to change
    base_conditions: Dict[str, Any]  # Original conditions
    expected_outcome: Optional[str] = None
    confidence: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "intervention": self.intervention,
            "base_conditions": self.base_conditions,
            "expected_outcome": self.expected_outcome,
            "confidence": self.confidence,
        }


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain."""
    step_number: int
    assertion: str
    justification: str
    evidence: List[Evidence]
    dependencies: List[int]  # Previous steps this depends on
    confidence: float  # 0 to 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_number": self.step_number,
            "assertion": self.assertion,
            "justification": self.justification,
            "evidence": [e.to_dict() for e in self.evidence],
            "dependencies": self.dependencies,
            "confidence": self.confidence,
        }


@dataclass
class ReasoningChain:
    """A complete chain of reasoning."""
    goal: str
    steps: List[ReasoningStep]
    conclusion: str
    overall_confidence: float
    alternative_conclusions: List[Tuple[str, float]] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "conclusion": self.conclusion,
            "overall_confidence": self.overall_confidence,
            "alternative_conclusions": self.alternative_conclusions,
            "gaps": self.gaps,
        }
    
    def to_markdown(self) -> str:
        """Convert to markdown format."""
        lines = [
            f"# Reasoning: {self.goal}",
            "",
            "## Reasoning Steps",
            "",
        ]
        
        for step in self.steps:
            lines.append(f"### Step {step.step_number}")
            lines.append(f"**Assertion:** {step.assertion}")
            lines.append(f"**Justification:** {step.justification}")
            lines.append(f"**Confidence:** {step.confidence:.1%}")
            if step.dependencies:
                lines.append(f"**Depends on:** Steps {step.dependencies}")
            lines.append("")
        
        lines.extend([
            "## Conclusion",
            self.conclusion,
            "",
            f"**Overall Confidence:** {self.overall_confidence:.1%}",
        ])
        
        if self.alternative_conclusions:
            lines.extend(["", "## Alternative Conclusions"])
            for alt, conf in self.alternative_conclusions:
                lines.append(f"- {alt} (confidence: {conf:.1%})")
        
        if self.gaps:
            lines.extend(["", "## Knowledge Gaps"])
            for gap in self.gaps:
                lines.append(f"- {gap}")
        
        return "\n".join(lines)


@dataclass
class HypothesisTest:
    """Results of hypothesis testing."""
    hypothesis: str
    verdict: str  # "supported", "refuted", "inconclusive"
    confidence: float
    supporting_evidence: List[Evidence]
    conflicting_evidence: List[Evidence]
    effect_size: Optional[float] = None
    p_value: Optional[float] = None
    practical_significance: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hypothesis": self.hypothesis,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "supporting_evidence": [e.to_dict() for e in self.supporting_evidence],
            "conflicting_evidence": [e.to_dict() for e in self.conflicting_evidence],
            "effect_size": self.effect_size,
            "p_value": self.p_value,
            "practical_significance": self.practical_significance,
        }


class CausalInference:
    """Causal inference engine for determining cause-effect relationships."""
    
    CAUSAL_PROMPT_TEMPLATE = """You are a causal inference expert. Analyze the 
following variables and data to determine causal relationships.

Variables: {variables}

Observed Data:
{data}

Background Knowledge:
{background}

Using the principles of causal inference (considering confounders, mediators, 
colliders, and temporal ordering), identify:

1. Direct causal relationships
2. Indirect causal paths
3. Confounding variables
4. Potential interventions and their expected effects

For each causal relationship, provide:
- Cause and effect variables
- Estimated causal strength (-1 to 1)
- Proposed mechanism
- Confidence level
- Key assumptions

Format your response as JSON."""
    
    def __init__(self, llm_interface: Optional[UnifiedLLMInterface] = None):
        """Initialize causal inference engine.
        
        Args:
            llm_interface: LLM interface for inference
        """
        self.llm = llm_interface or self._create_default_llm()
        self.relationships: List[CausalRelationship] = []
        self.factors: Dict[str, CausalFactor] = {}
    
    def _create_default_llm(self) -> UnifiedLLMInterface:
        """Create default LLM interface."""
        try:
            config = LLMConfig.from_env(LLMProvider.OPENAI)
        except ValueError:
            for provider in [LLMProvider.ANTHROPIC, LLMProvider.DEEPSEEK]:
                try:
                    config = LLMConfig.from_env(provider)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError("No LLM API keys found")
        
        return UnifiedLLMInterface(config)
    
    async def infer_relationships(
        self,
        variables: List[str],
        data: Dict[str, List[float]],
        background_knowledge: Optional[str] = None,
    ) -> List[CausalRelationship]:
        """Infer causal relationships from data.
        
        Args:
            variables: List of variable names
            data: Observed data (variable -> values)
            background_knowledge: Domain knowledge
            
        Returns:
            List of inferred causal relationships
        """
        # Prepare data summary
        data_summary = []
        for var, values in data.items():
            if values:
                mean = sum(values) / len(values)
                std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
                data_summary.append(f"{var}: mean={mean:.3f}, std={std:.3f}, n={len(values)}")
        
        prompt = self.CAUSAL_PROMPT_TEMPLATE.format(
            variables=", ".join(variables),
            data="\n".join(data_summary),
            background=background_knowledge or "No specific background knowledge provided.",
        )
        
        schema = {
            "causal_relationships": [
                {
                    "cause": "string",
                    "effect": "string",
                    "strength": "number -1 to 1",
                    "mechanism": "string",
                    "confidence": "string",
                    "assumptions": ["string"],
                }
            ],
            "confounders": ["string"],
            "mediators": ["string"],
            "recommendations": ["string"],
        }
        
        result = await self.llm.generate_structured(
            prompt=prompt,
            output_schema=schema,
            temperature=0.2,
        )
        
        relationships = []
        for rel_data in result.get("causal_relationships", []):
            rel = CausalRelationship(
                cause=rel_data.get("cause", ""),
                effect=rel_data.get("effect", ""),
                strength=float(rel_data.get("strength", 0)),
                mechanism=rel_data.get("mechanism"),
                assumptions=rel_data.get("assumptions", []),
            )
            relationships.append(rel)
        
        self.relationships = relationships
        return relationships
    
    async def estimate_causal_effect(
        self,
        treatment: str,
        outcome: str,
        confounders: List[str],
        data: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        """Estimate the causal effect of treatment on outcome.
        
        Args:
            treatment: Treatment variable name
            outcome: Outcome variable name
            confounders: List of confounding variables
            data: Observed data
            
        Returns:
            Effect estimate with confidence
        """
        prompt = f"""Estimate the causal effect of {treatment} on {outcome}.

Consider the following confounders: {', '.join(confounders)}

Data Summary:
{json.dumps({k: f"mean={sum(v)/len(v):.3f}" for k, v in data.items() if v}, indent=2)}

Using backdoor criterion and adjustment, estimate:
1. Average Treatment Effect (ATE)
2. Confidence in the estimate
3. Sensitivity to unmeasured confounding
4. Key assumptions required

Format as JSON with fields: ate, confidence, sensitivity_analysis, assumptions."""
        
        schema = {
            "ate": "number - estimated average treatment effect",
            "confidence": "number 0-1",
            "confidence_interval": ["number", "number"],
            "sensitivity_analysis": "string - discussion of robustness",
            "assumptions": ["string"],
            "limitations": ["string"],
        }
        
        result = await self.llm.generate_structured(prompt, schema, temperature=0.2)
        return result
    
    def get_causal_graph(self) -> Dict[str, Any]:
        """Get causal graph representation.
        
        Returns:
            Graph structure for visualization
        """
        nodes = set()
        edges = []
        
        for rel in self.relationships:
            nodes.add(rel.cause)
            nodes.add(rel.effect)
            edges.append({
                "source": rel.cause,
                "target": rel.effect,
                "strength": rel.strength,
                "mechanism": rel.mechanism,
            })
        
        return {
            "nodes": [{"id": n, "label": n} for n in nodes],
            "edges": edges,
        }


class CounterfactualAnalysis:
    """Counterfactual analysis for what-if scenarios."""
    
    def __init__(self, llm_interface: Optional[UnifiedLLMInterface] = None):
        """Initialize counterfactual analyzer.
        
        Args:
            llm_interface: LLM interface
        """
        self.llm = llm_interface or self._create_default_llm()
        self.scenarios: List[CounterfactualScenario] = []
    
    def _create_default_llm(self) -> UnifiedLLMInterface:
        """Create default LLM interface."""
        try:
            config = LLMConfig.from_env(LLMProvider.OPENAI)
        except ValueError:
            for provider in [LLMProvider.ANTHROPIC, LLMProvider.DEEPSEEK]:
                try:
                    config = LLMConfig.from_env(provider)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError("No LLM API keys found")
        
        return UnifiedLLMInterface(config)
    
    async def generate_scenarios(
        self,
        base_conditions: Dict[str, Any],
        intervention_space: Dict[str, List[Any]],
        num_scenarios: int = 3,
        domain: str = "materials",
    ) -> List[CounterfactualScenario]:
        """Generate counterfactual scenarios.
        
        Args:
            base_conditions: Original/base conditions
            intervention_space: Possible interventions (param -> possible values)
            num_scenarios: Number of scenarios to generate
            domain: Scientific domain
            
        Returns:
            List of counterfactual scenarios
        """
        prompt = f"""Generate {num_scenarios} interesting counterfactual scenarios 
for a {domain} system.

Base Conditions:
{json.dumps(base_conditions, indent=2)}

Possible Interventions:
{json.dumps(intervention_space, indent=2)}

For each scenario:
1. Describe the intervention clearly
2. Explain what is being changed from base conditions
3. Predict the expected outcome with reasoning
4. Estimate confidence in the prediction

Generate scientifically interesting scenarios that explore:
- Extreme parameter values
- Combinations of interventions
- Unexpected or non-linear effects"""
        
        schema = {
            "scenarios": [
                {
                    "name": "string",
                    "description": "string",
                    "intervention": {},
                    "expected_outcome": "string",
                    "confidence": "number 0-1",
                    "reasoning": "string",
                }
            ]
        }
        
        result = await self.llm.generate_structured(prompt, schema, temperature=0.7)
        
        scenarios = []
        for s in result.get("scenarios", []):
            scenario = CounterfactualScenario(
                name=s.get("name", ""),
                description=s.get("description", ""),
                intervention=s.get("intervention", {}),
                base_conditions=base_conditions,
                expected_outcome=s.get("expected_outcome"),
                confidence=s.get("confidence", 0.5),
            )
            scenarios.append(scenario)
        
        self.scenarios = scenarios
        return scenarios
    
    async def analyze_counterfactual(
        self,
        scenario: CounterfactualScenario,
        causal_model: Optional[CausalInference] = None,
    ) -> Dict[str, Any]:
        """Analyze a specific counterfactual scenario.
        
        Args:
            scenario: Counterfactual scenario to analyze
            causal_model: Optional causal model for structured analysis
            
        Returns:
            Analysis results
        """
        prompt = f"""Analyze the following counterfactual scenario:

Scenario: {scenario.name}
Description: {scenario.description}

Intervention:
{json.dumps(scenario.intervention, indent=2)}

Base Conditions:
{json.dumps(scenario.base_conditions, indent=2)}

Provide:
1. Detailed prediction of outcomes
2. Mechanism explaining the effect
3. Key factors determining the result
4. Uncertainty sources
5. Testable predictions
6. Comparison with base case

Use scientific reasoning principles. Be explicit about assumptions."""
        
        schema = {
            "predicted_outcomes": ["string"],
            "mechanism": "string",
            "key_factors": ["string"],
            "uncertainty_sources": ["string"],
            "testable_predictions": ["string"],
            "comparison_with_base": "string",
            "confidence": "number 0-1",
            "recommendations": ["string"],
        }
        
        result = await self.llm.generate_structured(prompt, schema, temperature=0.3)
        return result
    
    async def compare_scenarios(
        self,
        scenarios: List[CounterfactualScenario],
        comparison_dimensions: List[str],
    ) -> Dict[str, Any]:
        """Compare multiple counterfactual scenarios.
        
        Args:
            scenarios: Scenarios to compare
            comparison_dimensions: Dimensions to compare on
            
        Returns:
            Comparison results
        """
        scenarios_json = [s.to_dict() for s in scenarios]
        
        prompt = f"""Compare the following counterfactual scenarios:

{json.dumps(scenarios_json, indent=2)}

Compare across these dimensions: {', '.join(comparison_dimensions)}

Provide:
1. Comparative analysis for each dimension
2. Trade-offs between scenarios
3. Robustness comparison
4. Recommendations based on objectives"""
        
        schema = {
            "dimension_analysis": {},
            "trade_offs": ["string"],
            "robustness_ranking": ["string"],
            "recommendations": ["string"],
            "key_insights": ["string"],
        }
        
        return await self.llm.generate_structured(prompt, schema, temperature=0.3)


class ScientificReasoningEngine:
    """Main scientific reasoning engine combining all capabilities."""
    
    def __init__(self, llm_interface: Optional[UnifiedLLMInterface] = None):
        """Initialize the reasoning engine.
        
        Args:
            llm_interface: LLM interface
        """
        self.llm = llm_interface or self._create_default_llm()
        self.causal_inference = CausalInference(self.llm)
        self.counterfactual = CounterfactualAnalysis(self.llm)
        self.reasoning_history: List[ReasoningChain] = []
    
    def _create_default_llm(self) -> UnifiedLLMInterface:
        """Create default LLM interface."""
        try:
            config = LLMConfig.from_env(LLMProvider.OPENAI)
        except ValueError:
            for provider in [LLMProvider.ANTHROPIC, LLMProvider.DEEPSEEK]:
                try:
                    config = LLMConfig.from_env(provider)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError("No LLM API keys found")
        
        return UnifiedLLMInterface(config)
    
    async def construct_reasoning_chain(
        self,
        goal: str,
        evidence: List[Evidence],
        constraints: Optional[List[str]] = None,
        max_steps: int = 10,
    ) -> ReasoningChain:
        """Construct a chain of reasoning from evidence to conclusion.
        
        Args:
            goal: Reasoning goal/question
            evidence: Available evidence
            constraints: Logical constraints
            max_steps: Maximum reasoning steps
            
        Returns:
            Reasoning chain
        """
        evidence_text = "\n".join([
            f"- {e.type.value}: {e.description} (confidence: {e.confidence.name})"
            for e in evidence
        ])
        
        prompt = f"""Construct a detailed reasoning chain to address: {goal}

Available Evidence:
{evidence_text}

Constraints:
{chr(10).join(f"- {c}" for c in (constraints or []))}

Construct a chain of logical reasoning:
1. Start from established facts/evidence
2. Build logical connections between steps
3. Identify dependencies between assertions
4. Reach a well-justified conclusion
5. Note any gaps or uncertainties

Each step should have:
- Clear assertion
- Justification based on evidence or previous steps
- Confidence assessment
- Dependencies on previous steps (if any)

Format as JSON with reasoning steps array and conclusion."""
        
        schema = {
            "steps": [
                {
                    "step_number": "integer",
                    "assertion": "string",
                    "justification": "string",
                    "evidence_indices": ["integer"],
                    "dependencies": ["integer"],
                    "confidence": "number 0-1",
                }
            ],
            "conclusion": "string",
            "overall_confidence": "number 0-1",
            "alternative_conclusions": [
                {"conclusion": "string", "confidence": "number"}
            ],
            "gaps": ["string"],
        }
        
        result = await self.llm.generate_structured(prompt, schema, temperature=0.2)
        
        steps = []
        for s in result.get("steps", []):
            step_evidence = [
                evidence[i] for i in s.get("evidence_indices", [])
                if i < len(evidence)
            ]
            
            step = ReasoningStep(
                step_number=s.get("step_number", 0),
                assertion=s.get("assertion", ""),
                justification=s.get("justification", ""),
                evidence=step_evidence,
                dependencies=s.get("dependencies", []),
                confidence=s.get("confidence", 0.5),
            )
            steps.append(step)
        
        chain = ReasoningChain(
            goal=goal,
            steps=steps,
            conclusion=result.get("conclusion", ""),
            overall_confidence=result.get("overall_confidence", 0.5),
            alternative_conclusions=[
                (a.get("conclusion", ""), a.get("confidence", 0))
                for a in result.get("alternative_conclusions", [])
            ],
            gaps=result.get("gaps", []),
        )
        
        self.reasoning_history.append(chain)
        return chain
    
    async def test_hypothesis(
        self,
        hypothesis: str,
        evidence: List[Evidence],
        threshold: float = 0.7,
    ) -> HypothesisTest:
        """Test a hypothesis against evidence.
        
        Args:
            hypothesis: Hypothesis to test
            evidence: Evidence to evaluate
            threshold: Confidence threshold for support
            
        Returns:
            Hypothesis test results
        """
        evidence_text = "\n".join([
            f"Evidence {i+1} ({e.type.value}): {e.description}"
            for i, e in enumerate(evidence)
        ])
        
        prompt = f"""Test the following hypothesis against available evidence.

Hypothesis: {hypothesis}

Evidence:
{evidence_text}

Evaluate:
1. Which evidence supports the hypothesis?
2. Which evidence contradicts the hypothesis?
3. Overall verdict (supported/refuted/inconclusive)
4. Confidence level
5. Effect size (if quantifiable)
6. Practical significance

Be objective and consider alternative explanations."""
        
        schema = {
            "verdict": "string - supported/refuted/inconclusive",
            "confidence": "number 0-1",
            "supporting_evidence_indices": ["integer"],
            "conflicting_evidence_indices": ["integer"],
            "effect_size": "number or null",
            "practical_significance": "string",
            "reasoning": "string",
            "suggested_tests": ["string"],
        }
        
        result = await self.llm.generate_structured(prompt, schema, temperature=0.2)
        
        supporting = [
            evidence[i] for i in result.get("supporting_evidence_indices", [])
            if i < len(evidence)
        ]
        conflicting = [
            evidence[i] for i in result.get("conflicting_evidence_indices", [])
            if i < len(evidence)
        ]
        
        return HypothesisTest(
            hypothesis=hypothesis,
            verdict=result.get("verdict", "inconclusive"),
            confidence=result.get("confidence", 0.5),
            supporting_evidence=supporting,
            conflicting_evidence=conflicting,
            effect_size=result.get("effect_size"),
            practical_significance=result.get("practical_significance"),
        )
    
    async def abductive_reasoning(
        self,
        observations: List[str],
        possible_explanations: List[str],
    ) -> Dict[str, Any]:
        """Perform abductive reasoning (inference to best explanation).
        
        Args:
            observations: Observed facts
            possible_explanations: Candidate explanations
            
        Returns:
            Analysis of best explanation
        """
        prompt = f"""Given the following observations, determine the best explanation.

Observations:
{chr(10).join(f"- {o}" for o in observations)}

Candidate Explanations:
{chr(10).join(f"{i+1}. {e}" for i, e in enumerate(possible_explanations))}

For each explanation, evaluate:
1. Explanatory power (how well it accounts for observations)
2. Simplicity (Occam's razor)
3. Consistency with background knowledge
4. Predictive scope
5. Overall plausibility

Provide ranking and detailed justification."""
        
        schema = {
            "explanation_analysis": [
                {
                    "explanation": "string",
                    "explanatory_power": "number 0-1",
                    "simplicity": "number 0-1",
                    "consistency": "number 0-1",
                    "predictive_scope": "number 0-1",
                    "overall_score": "number 0-1",
                }
            ],
            "best_explanation": "string",
            "confidence": "number 0-1",
            "reasoning": "string",
        }
        
        return await self.llm.generate_structured(prompt, schema, temperature=0.3)
    
    async def analogical_reasoning(
        self,
        target_problem: str,
        source_domains: List[str],
        target_features: List[str],
    ) -> Dict[str, Any]:
        """Use analogical reasoning to transfer knowledge from known domains.
        
        Args:
            target_problem: Problem to solve
            source_domains: Domains to draw analogies from
            target_features: Key features of target problem
            
        Returns:
            Analogical reasoning results
        """
        prompt = f"""Use analogical reasoning to analyze: {target_problem}

Target Features:
{chr(10).join(f"- {f}" for f in target_features)}

Source Domains for Analogy:
{chr(10).join(f"- {d}" for d in source_domains)}

For each source domain:
1. Identify structural similarities
2. Map concepts between domains
3. Infer new insights for target problem
4. Assess analogy strength
5. Note limitations of the analogy

Synthesize insights from all analogies."""
        
        schema = {
            "analogies": [
                {
                    "source_domain": "string",
                    "structural_similarities": ["string"],
                    "concept_mapping": {},
                    "inferred_insights": ["string"],
                    "analogy_strength": "number 0-1",
                    "limitations": ["string"],
                }
            ],
            "synthesized_insights": ["string"],
            "recommended_approach": "string",
        }
        
        return await self.llm.generate_structured(prompt, schema, temperature=0.4)
    
    def aggregate_evidence(
        self,
        evidence_list: List[Evidence],
        method: str = "weighted_average",
    ) -> Dict[str, Any]:
        """Aggregate multiple pieces of evidence.
        
        Args:
            evidence_list: Evidence to aggregate
            method: Aggregation method
            
        Returns:
            Aggregated assessment
        """
        # Group evidence by type
        by_type = defaultdict(list)
        for e in evidence_list:
            by_type[e.type].append(e)
        
        # Calculate weighted confidence
        total_weight = 0
        weighted_confidence = 0
        
        type_weights = {
            EvidenceType.EXPERIMENTAL: 1.0,
            EvidenceType.COMPUTATIONAL: 0.8,
            EvidenceType.THEORETICAL: 0.7,
            EvidenceType.LITERATURE: 0.6,
            EvidenceType.EMPIRICAL: 0.5,
        }
        
        for e in evidence_list:
            weight = type_weights.get(e.type, 0.5) * e.confidence.value
            weighted_confidence += weight
            total_weight += type_weights.get(e.type, 0.5)
        
        if total_weight > 0:
            weighted_confidence /= total_weight
        
        return {
            "aggregated_confidence": weighted_confidence,
            "evidence_by_type": {k.value: len(v) for k, v in by_type.items()},
            "strongest_evidence": [
                e.to_dict() for e in sorted(
                    evidence_list,
                    key=lambda x: x.confidence.value,
                    reverse=True
                )[:3]
            ],
            "consistency": "high" if len(set(e.confidence for e in evidence_list)) < 3 else "mixed",
        }
