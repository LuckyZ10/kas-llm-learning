"""
Hypothesis Explainer Module

Transforms numerical simulation results into natural language explanations
using LLMs. This module bridges the gap between quantitative computational
chemistry data and qualitative scientific understanding.

Features:
- Automated interpretation of DFT/MD results
- Context-aware explanation generation
- Multiple explanation styles (technical, pedagogical, executive)
- Uncertainty quantification in explanations
- Multi-modal result interpretation
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import asyncio
from collections import defaultdict

from .llm_interface import (
    UnifiedLLMInterface,
    LLMConfig,
    LLMProvider,
    CompletionResponse,
    Message,
)


class ExplanationStyle(Enum):
    """Styles of explanation generation."""
    TECHNICAL = auto()      # Detailed, jargon-heavy for experts
    PEDAGOGICAL = auto()    # Educational, with background explanations
    EXECUTIVE = auto()      # High-level summary for decision makers
    METHODS_FOCUS = auto()  # Focus on methodology and validation
    HYPOTHESIS_DRIVEN = auto()  # Connect results to hypotheses
    COMPARATIVE = auto()    # Compare with literature/other methods
    CRITICAL = auto()       # Include limitations and critiques


class ResultType(Enum):
    """Types of computational results."""
    ENERGY = "energy"
    FORCE = "force"
    STRESS = "stress"
    BAND_STRUCTURE = "band_structure"
    DOS = "density_of_states"
    PHONON = "phonon"
    MD_TRAJECTORY = "md_trajectory"
    REACTION_PATHWAY = "reaction_pathway"
    OPTIMIZATION = "optimization"
    SPECTROSCOPY = "spectroscopy"
    TRANSPORT = "transport"
    DEFECT = "defect"
    INTERFACE = "interface"
    THERMODYNAMIC = "thermodynamic"
    KINETIC = "kinetic"


@dataclass
class NumericalValue:
    """A numerical value with context."""
    value: float
    unit: str
    uncertainty: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    source: Optional[str] = None  # e.g., "DFT-PBE", "MD-300K"
    
    def format(self, precision: int = 3) -> str:
        """Format the value with appropriate precision."""
        val_str = f"{self.value:.{precision}f}"
        if self.uncertainty:
            unc_str = f"{self.uncertainty:.{precision}f}"
            return f"{val_str} ± {unc_str} {self.unit}"
        return f"{val_str} {self.unit}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "value": self.value,
            "unit": self.unit,
            "uncertainty": self.uncertainty,
            "confidence_interval": self.confidence_interval,
            "source": self.source,
        }


@dataclass
class NumericalContext:
    """Context for numerical results interpretation."""
    system_description: str
    calculation_method: str
    numerical_results: Dict[str, NumericalValue]
    comparison_values: Optional[Dict[str, NumericalValue]] = None
    literature_references: Optional[List[Dict[str, Any]]] = None
    convergence_info: Optional[Dict[str, Any]] = None
    validation_metrics: Optional[Dict[str, float]] = None
    
    def to_prompt_context(self) -> str:
        """Convert to a string for LLM prompting."""
        lines = [
            f"System: {self.system_description}",
            f"Method: {self.calculation_method}",
            "",
            "Numerical Results:",
        ]
        
        for name, value in self.numerical_results.items():
            lines.append(f"  {name}: {value.format()}")
        
        if self.comparison_values:
            lines.append("\nComparison Values:")
            for name, value in self.comparison_values.items():
                lines.append(f"  {name}: {value.format()}")
        
        if self.validation_metrics:
            lines.append("\nValidation Metrics:")
            for metric, val in self.validation_metrics.items():
                lines.append(f"  {metric}: {val:.4f}")
        
        return "\n".join(lines)


@dataclass
class ExplanationResult:
    """Result of hypothesis explanation."""
    explanation: str
    key_findings: List[str]
    implications: List[str]
    limitations: List[str]
    confidence_score: float  # 0-1
    style: ExplanationStyle
    follow_up_questions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_markdown(self) -> str:
        """Convert to markdown format."""
        sections = [
            "# Analysis Results",
            "",
            self.explanation,
            "",
            "## Key Findings",
        ]
        for finding in self.key_findings:
            sections.append(f"- {finding}")
        
        sections.extend(["", "## Implications"])
        for impl in self.implications:
            sections.append(f"- {impl}")
        
        sections.extend(["", "## Limitations"])
        for lim in self.limitations:
            sections.append(f"- {lim}")
        
        sections.extend([
            "",
            f"**Confidence Score:** {self.confidence_score:.2%}",
            "",
            "## Suggested Follow-up Questions",
        ])
        for q in self.follow_up_questions:
            sections.append(f"- {q}")
        
        return "\n".join(sections)


@dataclass
class Hypothesis:
    """A scientific hypothesis to be tested."""
    statement: str
    expected_outcome: str
    test_criteria: List[str]
    domain: str  # e.g., "catalysis", "battery", "semiconductor"
    
    def to_prompt(self) -> str:
        """Convert to prompt format."""
        return f"""Hypothesis: {self.statement}

Expected Outcome: {self.expected_outcome}

Test Criteria:
{chr(10).join(f"- {c}" for c in self.test_criteria)}"""


class HypothesisExplainer:
    """Main class for explaining numerical results in context of hypotheses."""
    
    # Domain-specific system prompts
    DOMAIN_PROMPTS = {
        "catalysis": """You are an expert computational catalysis researcher. 
Interpret catalytic activity data, binding energies, reaction barriers, and 
electronic structure results in the context of catalytic mechanisms and 
design principles.""",
        
        "battery": """You are an expert in computational materials science for 
energy storage. Interpret ionic conductivity, voltage profiles, diffusion 
coefficients, and structural stability in the context of battery performance.""",
        
        "semiconductor": """You are an expert in semiconductor physics and 
computational materials science. Interpret band structures, defect levels, 
carrier mobilities, and optical properties in the context of device applications.""",
        
        "molecular": """You are an expert computational chemist. Interpret 
thermodynamic properties, reaction pathways, spectroscopic data, and 
structural analysis in the context of molecular mechanisms.""",
        
        "materials": """You are an expert materials scientist. Interpret 
mechanical, thermal, electronic, and structural properties in the context 
of materials design and applications.""",
    }
    
    # Style-specific instruction modifiers
    STYLE_INSTRUCTIONS = {
        ExplanationStyle.TECHNICAL: """Provide a detailed, technically rigorous 
interpretation. Use appropriate scientific terminology. Discuss computational 
accuracy, method limitations, and comparison with established benchmarks.""",
        
        ExplanationStyle.PEDAGOGICAL: """Explain the results as if teaching an 
advanced graduate student. Include relevant background concepts, explain why 
the results matter, and connect to fundamental principles.""",
        
        ExplanationStyle.EXECUTIVE: """Provide a concise, high-level summary 
suitable for program managers or industry stakeholders. Focus on implications, 
potential applications, and strategic value. Minimize technical jargon.""",
        
        ExplanationStyle.METHODS_FOCUS: """Emphasize the methodology: validation 
approaches, convergence, error estimates, and comparison between methods. 
Discuss what can and cannot be concluded from the computational approach used.""",
        
        ExplanationStyle.HYPOTHESIS_DRIVEN: """Structure the explanation around 
the hypothesis being tested. Explicitly state whether results support or refute 
the hypothesis, the strength of evidence, and required confidence levels.""",
        
        ExplanationStyle.COMPARATIVE: """Compare results extensively with 
literature values, other computational methods, and experimental data where 
available. Discuss sources of discrepancy and agreement.""",
        
        ExplanationStyle.CRITICAL: """Provide a critical analysis including 
limitations, potential confounding factors, alternative interpretations, 
and suggestions for validation experiments.""",
    }
    
    def __init__(
        self,
        llm_interface: Optional[UnifiedLLMInterface] = None,
        default_style: ExplanationStyle = ExplanationStyle.TECHNICAL,
        default_domain: str = "materials",
    ):
        """Initialize the hypothesis explainer.
        
        Args:
            llm_interface: LLM interface to use
            default_style: Default explanation style
            default_domain: Default scientific domain
        """
        self.llm = llm_interface or self._create_default_llm()
        self.default_style = default_style
        self.default_domain = default_domain
    
    def _create_default_llm(self) -> UnifiedLLMInterface:
        """Create default LLM interface from environment."""
        try:
            config = LLMConfig.from_env(LLMProvider.OPENAI)
        except ValueError:
            # Fall back to other providers
            for provider in [LLMProvider.ANTHROPIC, LLMProvider.DEEPSEEK]:
                try:
                    config = LLMConfig.from_env(provider)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError("No LLM API keys found in environment")
        
        return UnifiedLLMInterface(config)
    
    def _build_system_prompt(
        self,
        domain: Optional[str] = None,
        style: Optional[ExplanationStyle] = None,
    ) -> str:
        """Build the system prompt for explanation generation."""
        domain = domain or self.default_domain
        style = style or self.default_style
        
        base_prompt = self.DOMAIN_PROMPTS.get(domain, self.DOMAIN_PROMPTS["materials"])
        style_instruction = self.STYLE_INSTRUCTIONS[style]
        
        return f"""{base_prompt}

{style_instruction}

When analyzing results:
1. Connect numerical values to physical/chemical meaning
2. Discuss significance relative to typical values in the field
3. Identify patterns, anomalies, or unexpected results
4. Suggest physical mechanisms that could explain observations
5. Note any limitations or uncertainties in the analysis

Structure your response as JSON with these fields:
- explanation: Main narrative explanation
- key_findings: List of key observations
- implications: List of scientific or practical implications
- limitations: List of caveats and limitations
- confidence_score: Number 0-1 indicating confidence
- follow_up_questions: List of suggested next steps"""
    
    async def explain_results(
        self,
        context: NumericalContext,
        hypothesis: Optional[Hypothesis] = None,
        style: Optional[ExplanationStyle] = None,
        domain: Optional[str] = None,
        include_structured_output: bool = True,
    ) -> ExplanationResult:
        """Generate explanation for numerical results.
        
        Args:
            context: Numerical context with results
            hypothesis: Optional hypothesis being tested
            style: Explanation style
            domain: Scientific domain
            include_structured_output: Whether to parse structured output
            
        Returns:
            Explanation result
        """
        style = style or self.default_style
        domain = domain or self.default_domain
        
        # Build the prompt
        prompt_parts = [
            "Please analyze the following computational results:",
            "",
            context.to_prompt_context(),
        ]
        
        if hypothesis:
            prompt_parts.extend([
                "",
                "Hypothesis Being Tested:",
                hypothesis.to_prompt(),
            ])
        
        if hypothesis:
            prompt_parts.extend([
                "",
                f"Does the evidence support or refute the hypothesis? "
                f"Evaluate based on the test criteria.",
            ])
        
        prompt = "\n".join(prompt_parts)
        
        # Generate explanation
        system_prompt = self._build_system_prompt(domain, style)
        
        if include_structured_output:
            schema = {
                "explanation": "string - detailed explanation of results",
                "key_findings": ["string - key observations"],
                "implications": ["string - scientific implications"],
                "limitations": ["string - caveats and limitations"],
                "confidence_score": "number 0-1",
                "follow_up_questions": ["string - suggested next steps"],
            }
            
            result = await self.llm.generate_structured(
                prompt=prompt,
                output_schema=schema,
                system_prompt=system_prompt,
                temperature=0.3,  # Lower temperature for structured output
            )
            
            return ExplanationResult(
                explanation=result.get("explanation", ""),
                key_findings=result.get("key_findings", []),
                implications=result.get("implications", []),
                limitations=result.get("limitations", []),
                confidence_score=float(result.get("confidence_score", 0.5)),
                style=style,
                follow_up_questions=result.get("follow_up_questions", []),
            )
        else:
            response = await self.llm.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.7,
            )
            
            return ExplanationResult(
                explanation=response.content,
                key_findings=[],
                implications=[],
                limitations=[],
                confidence_score=0.5,
                style=style,
                follow_up_questions=[],
            )
    
    async def explain_trajectory(
        self,
        trajectory_data: List[Dict[str, Any]],
        context: NumericalContext,
        focus: str = "trends",
        style: Optional[ExplanationStyle] = None,
    ) -> ExplanationResult:
        """Explain time-series or trajectory data.
        
        Args:
            trajectory_data: List of time-step data
            context: Numerical context
            focus: Analysis focus ("trends", "events", "statistics")
            style: Explanation style
            
        Returns:
            Explanation result
        """
        # Summarize trajectory data
        summary = self._summarize_trajectory(trajectory_data, focus)
        
        prompt = f"""Analyze the following trajectory data summary:

{summary}

Original Context:
{context.to_prompt_context()}

Focus on: {focus}

Provide insights about:
1. Overall trends and patterns
2. Significant events or transitions
3. Stability and convergence characteristics
4. Physical interpretation of dynamics"""
        
        system_prompt = self._build_system_prompt(context=None, style=style)
        
        schema = {
            "explanation": "string",
            "key_findings": ["string"],
            "implications": ["string"],
            "limitations": ["string"],
            "confidence_score": "number",
            "follow_up_questions": ["string"],
        }
        
        result = await self.llm.generate_structured(
            prompt=prompt,
            output_schema=schema,
            system_prompt=system_prompt,
            temperature=0.3,
        )
        
        return ExplanationResult(
            explanation=result.get("explanation", ""),
            key_findings=result.get("key_findings", []),
            implications=result.get("implications", []),
            limitations=result.get("limitations", []),
            confidence_score=float(result.get("confidence_score", 0.5)),
            style=style or self.default_style,
            follow_up_questions=result.get("follow_up_questions", []),
        )
    
    def _summarize_trajectory(
        self,
        trajectory_data: List[Dict[str, Any]],
        focus: str,
    ) -> str:
        """Summarize trajectory data for LLM consumption."""
        if not trajectory_data:
            return "No trajectory data provided."
        
        # Extract numerical properties
        numeric_keys = [
            k for k, v in trajectory_data[0].items()
            if isinstance(v, (int, float))
        ]
        
        summary_lines = [
            f"Trajectory contains {len(trajectory_data)} steps.",
            f"Tracked properties: {', '.join(numeric_keys)}",
            "",
            "Statistical Summary:",
        ]
        
        for key in numeric_keys:
            values = [step[key] for step in trajectory_data if key in step]
            if values:
                mean = sum(values) / len(values)
                min_val = min(values)
                max_val = max(values)
                std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
                
                summary_lines.append(
                    f"  {key}: mean={mean:.4f}, std={std:.4f}, "
                    f"range=[{min_val:.4f}, {max_val:.4f}]"
                )
        
        # Detect trends
        if focus in ["trends", "events"] and len(trajectory_data) > 1:
            summary_lines.append("\nTrends:")
            for key in numeric_keys[:3]:  # Limit to first 3 properties
                values = [step[key] for step in trajectory_data if key in step]
                if len(values) > 1:
                    # Simple trend detection
                    first_half = sum(values[:len(values)//2]) / (len(values)//2)
                    second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
                    
                    if abs(second_half - first_half) > 0.1 * abs(first_half):
                        direction = "increasing" if second_half > first_half else "decreasing"
                        summary_lines.append(f"  {key}: {direction} trend detected")
        
        return "\n".join(summary_lines)
    
    async def compare_calculations(
        self,
        contexts: List[NumericalContext],
        comparison_focus: str = "methodology",
        style: Optional[ExplanationStyle] = None,
    ) -> ExplanationResult:
        """Compare multiple calculation results.
        
        Args:
            contexts: List of numerical contexts to compare
            comparison_focus: What aspect to focus on
            style: Explanation style
            
        Returns:
            Explanation result
        """
        prompt_parts = [
            f"Compare the following {len(contexts)} calculations:",
            "",
        ]
        
        for i, ctx in enumerate(contexts, 1):
            prompt_parts.append(f"Calculation {i}:")
            prompt_parts.append(ctx.to_prompt_context())
            prompt_parts.append("")
        
        prompt_parts.append(f"Focus on: {comparison_focus}")
        prompt_parts.append(
            "\nAnalyze similarities, differences, systematic trends, and "
            "provide recommendations based on the comparison."
        )
        
        prompt = "\n".join(prompt_parts)
        system_prompt = self._build_system_prompt(None, style)
        
        schema = {
            "explanation": "string - comparative analysis",
            "key_findings": ["string - main observations from comparison"],
            "implications": ["string - implications of differences"],
            "limitations": ["string - limitations in comparison"],
            "confidence_score": "number 0-1",
            "follow_up_questions": ["string"],
        }
        
        result = await self.llm.generate_structured(
            prompt=prompt,
            output_schema=schema,
            system_prompt=system_prompt,
            temperature=0.3,
        )
        
        return ExplanationResult(
            explanation=result.get("explanation", ""),
            key_findings=result.get("key_findings", []),
            implications=result.get("implications", []),
            limitations=result.get("limitations", []),
            confidence_score=float(result.get("confidence_score", 0.5)),
            style=style or self.default_style,
            follow_up_questions=result.get("follow_up_questions", []),
        )
    
    async def generate_hypothesis_from_results(
        self,
        context: NumericalContext,
        domain: Optional[str] = None,
    ) -> List[Hypothesis]:
        """Generate hypotheses based on observed results.
        
        Args:
            context: Numerical context with results
            domain: Scientific domain
            
        Returns:
            List of generated hypotheses
        """
        prompt = f"""Based on the following computational results, generate 
3-5 testable scientific hypotheses that could explain the observations.

{context.to_prompt_context()}

For each hypothesis, provide:
1. A clear statement
2. Expected outcomes if true
3. Specific test criteria

Format as JSON array of objects with fields: statement, expected_outcome, test_criteria (array)."""
        
        system_prompt = self.DOMAIN_PROMPTS.get(domain or self.default_domain, self.DOMAIN_PROMPTS["materials"])
        
        schema = {
            "hypotheses": [
                {
                    "statement": "string - the hypothesis",
                    "expected_outcome": "string - what we expect if true",
                    "test_criteria": ["string - list of criteria"],
                }
            ]
        }
        
        result = await self.llm.generate_structured(
            prompt=prompt,
            output_schema=schema,
            system_prompt=system_prompt,
            temperature=0.7,  # Higher temperature for creativity
        )
        
        hypotheses = []
        for h in result.get("hypotheses", []):
            hypotheses.append(Hypothesis(
                statement=h.get("statement", ""),
                expected_outcome=h.get("expected_outcome", ""),
                test_criteria=h.get("test_criteria", []),
                domain=domain or self.default_domain,
            ))
        
        return hypotheses
    
    async def explain_band_structure(
        self,
        band_gap: NumericalValue,
        fermi_level: NumericalValue,
        key_features: List[Dict[str, Any]],
        material_type: str = "semiconductor",
        style: Optional[ExplanationStyle] = None,
    ) -> ExplanationResult:
        """Specialized explainer for band structure results.
        
        Args:
            band_gap: Band gap value
            fermi_level: Fermi level value
            key_features: List of key band structure features
            material_type: Type of material
            style: Explanation style
            
        Returns:
            Explanation result
        """
        context = NumericalContext(
            system_description=f"{material_type} material electronic structure",
            calculation_method="DFT band structure calculation",
            numerical_results={
                "band_gap": band_gap,
                "fermi_level": fermi_level,
            },
        )
        
        prompt = f"""Analyze the following band structure results:

Band Gap: {band_gap.format()}
Fermi Level: {fermi_level.format()}

Key Features:
{json.dumps(key_features, indent=2)}

Provide analysis including:
1. Electronic character (direct/indirect gap, metal/semiconductor/insulator)
2. Comparison with typical values for this material class
3. Implications for optical and transport properties
4. Reliability of DFT prediction (DFT typically underestimates gaps)"""
        
        system_prompt = self._build_system_prompt("semiconductor", style)
        
        schema = {
            "explanation": "string - comprehensive band structure analysis",
            "key_findings": ["string"],
            "implications": ["string - device/research implications"],
            "limitations": ["string - DFT limitations"],
            "confidence_score": "number",
            "follow_up_questions": ["string"],
        }
        
        result = await self.llm.generate_structured(
            prompt=prompt,
            output_schema=schema,
            system_prompt=system_prompt,
            temperature=0.3,
        )
        
        return ExplanationResult(
            explanation=result.get("explanation", ""),
            key_findings=result.get("key_findings", []),
            implications=result.get("implications", []),
            limitations=result.get("limitations", []),
            confidence_score=float(result.get("confidence_score", 0.5)),
            style=style or self.default_style,
            follow_up_questions=result.get("follow_up_questions", []),
        )
    
    async def explain_reaction_pathway(
        self,
        reactants: Dict[str, NumericalValue],
        products: Dict[str, NumericalValue],
        transition_states: List[Dict[str, NumericalValue]],
        reaction_coordinates: List[float],
        energies: List[float],
        style: Optional[ExplanationStyle] = None,
    ) -> ExplanationResult:
        """Specialized explainer for reaction pathway analysis.
        
        Args:
            reactants: Reactant energy values
            products: Product energy values
            transition_states: Transition state information
            reaction_coordinates: Reaction coordinate values
            energies: Energy profile
            style: Explanation style
            
        Returns:
            Explanation result
        """
        context = NumericalContext(
            system_description="Chemical reaction pathway",
            calculation_method="NEB or string method calculation",
            numerical_results={
                **{f"reactant_{k}": v for k, v in reactants.items()},
                **{f"product_{k}": v for k, v in products.items()},
            },
        )
        
        # Calculate key metrics
        delta_e = sum(p.value for p in products.values()) - sum(r.value for r in reactants.values())
        barrier = max(energies) - energies[0] if energies else 0
        
        prompt = f"""Analyze the following reaction pathway:

Energy Change (ΔE): {delta_e:.3f} eV
Activation Barrier: {barrier:.3f} eV

Number of Transition States: {len(transition_states)}
Reaction Coordinate Range: [{min(reaction_coordinates):.2f}, {max(reaction_coordinates):.2f}]

Pathway Characteristics:
- {'Exothermic' if delta_e < 0 else 'Endothermic'} reaction
- {'Low barrier - kinetically favorable' if barrier < 0.5 else 'Moderate barrier' if barrier < 1.0 else 'High barrier - kinetically hindered'}

Provide analysis of:
1. Thermodynamic favorability
2. Kinetic accessibility
3. Possible alternative pathways
4. Temperature effects and catalytic implications"""
        
        system_prompt = self._build_system_prompt("catalysis", style)
        
        schema = {
            "explanation": "string - reaction pathway analysis",
            "key_findings": ["string"],
            "implications": ["string - catalysis/implications"],
            "limitations": ["string"],
            "confidence_score": "number",
            "follow_up_questions": ["string"],
        }
        
        result = await self.llm.generate_structured(
            prompt=prompt,
            output_schema=schema,
            system_prompt=system_prompt,
            temperature=0.3,
        )
        
        return ExplanationResult(
            explanation=result.get("explanation", ""),
            key_findings=result.get("key_findings", []),
            implications=result.get("implications", []),
            limitations=result.get("limitations", []),
            confidence_score=float(result.get("confidence_score", 0.5)),
            style=style or self.default_style,
            follow_up_questions=result.get("follow_up_questions", []),
        )
