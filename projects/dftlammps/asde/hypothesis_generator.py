"""
Hypothesis Generator Module

Generates scientific hypotheses based on knowledge graphs and causal reasoning.
Implements multiple hypothesis generation strategies including analogical reasoning,
abductive inference, and causal chain exploration.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Protocol

import numpy as np

from ..knowledge_graph.scientific_kg import (
    Entity,
    EntityType,
    RelationType,
    ScientificKnowledgeGraph,
    CausalReasoner,
)


class HypothesisType(Enum):
    """Types of scientific hypotheses."""
    CAUSAL = auto()           # X causes Y
    CORRELATIONAL = auto()    # X correlates with Y
    MECHANISTIC = auto()      # X works through mechanism M
    PREDICTIVE = auto()       # If X then Y will occur
    COMPARATIVE = auto()      # X has greater effect than Y
    ANALOGICAL = auto()       # A is like B, so A has property of B
    COMPOSITIONAL = auto()    # Combining X and Y produces Z


@dataclass
class Hypothesis:
    """Represents a scientific hypothesis."""
    id: str
    statement: str
    hypothesis_type: HypothesisType
    confidence: float  # 0-1
    supporting_evidence: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    testable_predictions: list[str] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    novelty_score: float = 0.5  # How novel is this hypothesis?
    source_strategy: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate hypothesis."""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "statement": self.statement,
            "type": self.hypothesis_type.name,
            "confidence": self.confidence,
            "novelty_score": self.novelty_score,
            "source_strategy": self.source_strategy,
            "supporting_evidence": self.supporting_evidence,
            "assumptions": self.assumptions,
            "testable_predictions": self.testable_predictions,
            "entities": [e.name for e in self.entities],
            "metadata": self.metadata
        }


class HypothesisStrategy(ABC):
    """Abstract base class for hypothesis generation strategies."""
    
    def __init__(self, kg: ScientificKnowledgeGraph) -> None:
        self.kg = kg
        self.causal_reasoner = CausalReasoner(kg)
    
    @abstractmethod
    def generate(
        self,
        seed_entities: Optional[list[str]] = None,
        max_hypotheses: int = 5
    ) -> list[Hypothesis]: ...
    
    @property
    @abstractmethod
    def name(self) -> str: ...


class CausalChainStrategy(HypothesisStrategy):
    """
    Generate hypotheses by exploring causal chains in the knowledge graph.
    """
    
    @property
    def name(self) -> str:
        return "causal_chain"
    
    def generate(
        self,
        seed_entities: Optional[list[str]] = None,
        max_hypotheses: int = 5
    ) -> list[Hypothesis]:
        """Generate hypotheses by finding unexplored causal chains."""
        hypotheses: list[Hypothesis] = []
        
        # Get materials and properties
        materials = [
            e for e in self.kg._entities.values()
            if e.entity_type == EntityType.MATERIAL
        ]
        
        properties = [
            e for e in self.kg._entities.values()
            if e.entity_type == EntityType.PROPERTY
        ]
        
        if not materials or not properties:
            return hypotheses
        
        # For each material, find potential causal connections to properties
        for material in materials[:max_hypotheses]:
            # Find what properties this material might affect
            for prop in properties:
                # Check if direct connection exists
                existing = list(self.kg.query(
                    entity_type=EntityType.MATERIAL,
                    relation_type=RelationType.HAS_PROPERTY
                ))
                
                # Find indirect paths
                path = self.kg.find_path(material.id, prop.id, max_length=3)
                
                if path and len(path) > 2:  # Indirect connection
                    intermediate = path[1:-1]
                    
                    # Build hypothesis statement
                    chain_names = " → ".join(e.name for e in path)
                    statement = (
                        f"{material.name} affects {prop.name} through mechanism: {chain_names}"
                    )
                    
                    # Generate predictions
                    predictions = [
                        f"Modulating {intermediate[0].name} will change {prop.name}",
                        f"Blocking {intermediate[-1].name} will prevent {material.name} → {prop.name}",
                    ]
                    
                    hypothesis = Hypothesis(
                        id=f"cc_{material.id}_{prop.id}",
                        statement=statement,
                        hypothesis_type=HypothesisType.MECHANISTIC,
                        confidence=0.4 + 0.1 * len(path),  # Higher confidence for shorter paths
                        supporting_evidence=[f"Path exists in knowledge graph: {chain_names}"],
                        assumptions=[
                            "Causal relationships are transitive",
                            "Mechanism is conserved across contexts"
                        ],
                        testable_predictions=predictions,
                        entities=path,
                        novelty_score=0.6,
                        source_strategy=self.name
                    )
                    
                    hypotheses.append(hypothesis)
                    
                    if len(hypotheses) >= max_hypotheses:
                        break
            
            if len(hypotheses) >= max_hypotheses:
                break
        
        return hypotheses


class AnalogicalReasoningStrategy(HypothesisStrategy):
    """
    Generate hypotheses through analogical reasoning between similar materials.
    """
    
    @property
    def name(self) -> str:
        return "analogical"
    
    def generate(
        self,
        seed_entities: Optional[list[str]] = None,
        max_hypotheses: int = 5
    ) -> list[Hypothesis]:
        """Generate hypotheses by analogical transfer between similar materials."""
        hypotheses: list[Hypothesis] = []
        
        # Get all materials
        materials = [
            e for e in self.kg._entities.values()
            if e.entity_type == EntityType.MATERIAL
        ]
        
        # Find material pairs with similar properties
        material_properties: dict[str, list[Entity]] = {}
        for material in materials:
            props = self.kg.get_material_properties(material.name)
            material_properties[material.id] = [p for p, _ in props]
        
        # Find pairs with high property overlap
        for i, m1 in enumerate(materials):
            for m2 in materials[i+1:]:
                props1 = set(p.id for p in material_properties.get(m1.id, []))
                props2 = set(p.id for p in material_properties.get(m2.id, []))
                
                if not props1 or not props2:
                    continue
                
                overlap = len(props1 & props2) / max(len(props1), len(props2))
                
                if overlap > 0.5:  # High similarity
                    # Find properties unique to one
                    unique_to_m1 = props1 - props2
                    unique_to_m2 = props2 - props1
                    
                    # Generate analogical hypotheses
                    for prop_id in unique_to_m1:
                        prop = self.kg.get_entity(prop_id)
                        if prop:
                            statement = (
                                f"Since {m1.name} and {m2.name} share similar properties, "
                                f"{m2.name} may also exhibit {prop.name} like {m1.name}"
                            )
                            
                            hypothesis = Hypothesis(
                                id=f"ana_{m2.id}_{prop.id}",
                                statement=statement,
                                hypothesis_type=HypothesisType.ANALOGICAL,
                                confidence=0.5 + 0.3 * overlap,
                                supporting_evidence=[
                                    f"{m1.name} has {prop.name}",
                                    f"Materials share {overlap:.0%} of properties"
                                ],
                                assumptions=[
                                    "Similar materials have similar properties",
                                    "Property transfer is valid across these materials"
                                ],
                                testable_predictions=[
                                    f"Measure {prop.name} in {m2.name}",
                                    f"Compare {prop.name} values between {m1.name} and {m2.name}"
                                ],
                                entities=[m1, m2, prop],
                                novelty_score=0.5 + 0.3 * (1 - overlap),
                                source_strategy=self.name
                            )
                            
                            hypotheses.append(hypothesis)
                            
                            if len(hypotheses) >= max_hypotheses:
                                break
                    
                    for prop_id in unique_to_m2:
                        prop = self.kg.get_entity(prop_id)
                        if prop:
                            statement = (
                                f"Since {m1.name} and {m2.name} share similar properties, "
                                f"{m1.name} may also exhibit {prop.name} like {m2.name}"
                            )
                            
                            hypothesis = Hypothesis(
                                id=f"ana_{m1.id}_{prop.id}",
                                statement=statement,
                                hypothesis_type=HypothesisType.ANALOGICAL,
                                confidence=0.5 + 0.3 * overlap,
                                supporting_evidence=[
                                    f"{m2.name} has {prop.name}",
                                    f"Materials share {overlap:.0%} of properties"
                                ],
                                assumptions=[
                                    "Similar materials have similar properties",
                                    "Property transfer is valid across these materials"
                                ],
                                testable_predictions=[
                                    f"Measure {prop.name} in {m1.name}",
                                    f"Compare {prop.name} values between {m1.name} and {m2.name}"
                                ],
                                entities=[m1, m2, prop],
                                novelty_score=0.5 + 0.3 * (1 - overlap),
                                source_strategy=self.name
                            )
                            
                            hypotheses.append(hypothesis)
                            
                            if len(hypotheses) >= max_hypotheses:
                                break
                
                if len(hypotheses) >= max_hypotheses:
                    break
            
            if len(hypotheses) >= max_hypotheses:
                break
        
        return hypotheses


class CompositionalStrategy(HypothesisStrategy):
    """
    Generate hypotheses by composing existing knowledge elements.
    """
    
    @property
    def name(self) -> str:
        return "compositional"
    
    def generate(
        self,
        seed_entities: Optional[list[str]] = None,
        max_hypotheses: int = 5
    ) -> list[Hypothesis]:
        """Generate hypotheses by combining methods, materials, or phenomena."""
        hypotheses: list[Hypothesis] = []
        
        # Get entities by type
        methods = [
            e for e in self.kg._entities.values()
            if e.entity_type == EntityType.METHOD
        ]
        
        materials = [
            e for e in self.kg._entities.values()
            if e.entity_type == EntityType.MATERIAL
        ]
        
        phenomena = [
            e for e in self.kg._entities.values()
            if e.entity_type == EntityType.PHENOMENON
        ]
        
        # Generate method combination hypotheses
        if len(methods) >= 2:
            for i, m1 in enumerate(methods[:3]):
                for m2 in methods[i+1:min(i+3, len(methods))]:
                    statement = (
                        f"Combining {m1.name} with {m2.name} will yield more accurate "
                        f"predictions than either method alone"
                    )
                    
                    hypothesis = Hypothesis(
                        id=f"comp_{m1.id}_{m2.id}",
                        statement=statement,
                        hypothesis_type=HypothesisType.COMPOSITIONAL,
                        confidence=0.45,
                        supporting_evidence=[
                            f"{m1.name} has unique strengths",
                            f"{m2.name} has complementary strengths"
                        ],
                        assumptions=[
                            "Methods are complementary",
                            "Combination doesn't introduce systematic errors"
                        ],
                        testable_predictions=[
                            f"Compare {m1.name} vs {m2.name} vs combined approach",
                            f"Measure prediction accuracy for all three approaches"
                        ],
                        entities=[m1, m2],
                        novelty_score=0.6,
                        source_strategy=self.name
                    )
                    
                    hypotheses.append(hypothesis)
                    
                    if len(hypotheses) >= max_hypotheses:
                        break
                
                if len(hypotheses) >= max_hypotheses:
                    break
        
        # Generate material combination hypotheses
        if len(hypotheses) < max_hypotheses and len(materials) >= 2:
            for i, mat1 in enumerate(materials[:3]):
                for mat2 in materials[i+1:min(i+3, len(materials))]:
                    # Find properties both materials have
                    props1 = self.kg.get_material_properties(mat1.name)
                    props2 = self.kg.get_material_properties(mat2.name)
                    
                    common_props = set(p.name for p, _ in props1) & \
                                   set(p.name for p, _ in props2)
                    
                    if common_props:
                        prop_name = list(common_props)[0]
                        statement = (
                            f"A composite of {mat1.name} and {mat2.name} will exhibit "
                            f"enhanced {prop_name} compared to individual components"
                        )
                        
                        hypothesis = Hypothesis(
                            id=f"comp_mat_{mat1.id}_{mat2.id}",
                            statement=statement,
                            hypothesis_type=HypothesisType.COMPOSITIONAL,
                            confidence=0.4,
                            supporting_evidence=[
                                f"Both materials exhibit {prop_name}",
                                "Composites often show synergistic effects"
                            ],
                            assumptions=[
                                f"{prop_name} is additive or synergistic in composites",
                                "Materials are compatible"
                            ],
                            testable_predictions=[
                                f"Synthesize {mat1.name}-{mat2.name} composite",
                                f"Measure {prop_name} and compare to components"
                            ],
                            entities=[mat1, mat2],
                            novelty_score=0.55,
                            source_strategy=self.name
                        )
                        
                        hypotheses.append(hypothesis)
                        
                        if len(hypotheses) >= max_hypotheses:
                            break
                
                if len(hypotheses) >= max_hypotheses:
                    break
        
        return hypotheses


class AbductiveInferenceStrategy(HypothesisStrategy):
    """
    Generate hypotheses through abductive reasoning (inference to best explanation).
    """
    
    @property
    def name(self) -> str:
        return "abductive"
    
    def generate(
        self,
        seed_entities: Optional[list[str]] = None,
        max_hypotheses: int = 5
    ) -> list[Hypothesis]:
        """Generate hypotheses by finding best explanations for observations."""
        hypotheses: list[Hypothesis] = []
        
        # Find phenomena without clear causes
        phenomena = [
            e for e in self.kg._entities.values()
            if e.entity_type == EntityType.PHENOMENON
        ]
        
        variables = [
            e for e in self.kg._entities.values()
            if e.entity_type == EntityType.VARIABLE
        ]
        
        for phenomenon in phenomena[:max_hypotheses]:
            # Find potential explanatory variables
            for var in variables:
                # Check if there's a potential connection
                path = self.kg.find_path(var.id, phenomenon.id, max_length=2)
                
                if path:
                    statement = (
                        f"{var.name} is a primary cause of {phenomenon.name}, "
                        f"mediated by the mechanism: {' → '.join(e.name for e in path)}"
                    )
                    
                    hypothesis = Hypothesis(
                        id=f"abd_{var.id}_{phenomenon.id}",
                        statement=statement,
                        hypothesis_type=HypothesisType.CAUSAL,
                        confidence=0.4,
                        supporting_evidence=[
                            f"Path exists from {var.name} to {phenomenon.name}",
                            f"{var.name} is commonly manipulated in experiments"
                        ],
                        assumptions=[
                            f"{var.name} is controllable",
                            "Mechanism is primarily causal not correlational"
                        ],
                        testable_predictions=[
                            f"Varying {var.name} will modulate {phenomenon.name}",
                            f"Holding {var.name} constant will stabilize {phenomenon.name}"
                        ],
                        entities=path,
                        novelty_score=0.5,
                        source_strategy=self.name
                    )
                    
                    hypotheses.append(hypothesis)
                    break
        
        return hypotheses


class GapDrivenStrategy(HypothesisStrategy):
    """
    Generate hypotheses by identifying gaps in the knowledge graph.
    """
    
    @property
    def name(self) -> str:
        return "gap_driven"
    
    def generate(
        self,
        seed_entities: Optional[list[str]] = None,
        max_hypotheses: int = 5
    ) -> list[Hypothesis]:
        """Generate hypotheses by finding unexplored connections."""
        hypotheses: list[Hypothesis] = []
        
        # Find central entities with few connections
        central = self.kg.get_central_entities(top_k=10)
        
        for entity, centrality_score in central:
            # Count connections
            neighbors = self.kg.get_neighbors(entity.id)
            
            if len(neighbors) < 3:  # Underexplored entity
                # Find potential connections to other central entities
                for other_entity, other_score in central:
                    if other_entity.id == entity.id:
                        continue
                    
                    # Check if already connected
                    existing_path = self.kg.find_path(
                        entity.id,
                        other_entity.id,
                        max_length=2
                    )
                    
                    if not existing_path:
                        # Potential new connection
                        statement = (
                            f"There exists an undiscovered relationship between "
                            f"{entity.name} and {other_entity.name} that affects "
                            f"{entity.entity_type.name.lower()} behavior"
                        )
                        
                        hypothesis = Hypothesis(
                            id=f"gap_{entity.id}_{other_entity.id}",
                            statement=statement,
                            hypothesis_type=HypothesisType.CORRELATIONAL,
                            confidence=0.3,
                            supporting_evidence=[
                                f"{entity.name} is central (score: {centrality_score:.3f})",
                                f"{other_entity.name} is central (score: {other_score:.3f})",
                                "No direct connection exists in current knowledge"
                            ],
                            assumptions=[
                                "Central entities are likely connected",
                                "Connection is scientifically meaningful"
                            ],
                            testable_predictions=[
                                f"Search for correlation between {entity.name} and {other_entity.name}",
                                f"Design experiment to test {entity.name} → {other_entity.name} effect"
                            ],
                            entities=[entity, other_entity],
                            novelty_score=0.8,
                            source_strategy=self.name
                        )
                        
                        hypotheses.append(hypothesis)
                        
                        if len(hypotheses) >= max_hypotheses:
                            break
                
                if len(hypotheses) >= max_hypotheses:
                    break
        
        return hypotheses


class HypothesisGenerator:
    """
    Main hypothesis generation engine.
    
    Combines multiple strategies to generate diverse, high-quality
    scientific hypotheses from a knowledge graph.
    """
    
    def __init__(
        self,
        kg: ScientificKnowledgeGraph,
        strategies: Optional[list[type[HypothesisStrategy]]] = None
    ) -> None:
        self.kg = kg
        self.strategies: list[HypothesisStrategy] = []
        
        if strategies is None:
            strategies = [
                CausalChainStrategy,
                AnalogicalReasoningStrategy,
                CompositionalStrategy,
                AbductiveInferenceStrategy,
                GapDrivenStrategy,
            ]
        
        for strategy_class in strategies:
            self.strategies.append(strategy_class(kg))
    
    def generate(
        self,
        seed_entities: Optional[list[str]] = None,
        max_hypotheses: int = 10,
        min_confidence: float = 0.0,
        diversity_weight: float = 0.3
    ) -> list[Hypothesis]:
        """
        Generate hypotheses using all strategies.
        
        Args:
            seed_entities: Optional list of entity IDs to focus on
            max_hypotheses: Maximum number of hypotheses to return
            min_confidence: Minimum confidence threshold
            diversity_weight: Weight for diversity in selection (0-1)
        """
        all_hypotheses: list[Hypothesis] = []
        
        # Generate from each strategy
        per_strategy = max(1, max_hypotheses // len(self.strategies))
        
        for strategy in self.strategies:
            try:
                hypotheses = strategy.generate(seed_entities, per_strategy * 2)
                all_hypotheses.extend(hypotheses)
            except Exception:
                continue
        
        # Filter by confidence
        all_hypotheses = [
            h for h in all_hypotheses
            if h.confidence >= min_confidence
        ]
        
        # Deduplicate by statement
        seen_statements: set[str] = set()
        unique_hypotheses: list[Hypothesis] = []
        
        for h in all_hypotheses:
            key = h.statement.lower().strip()
            if key not in seen_statements:
                seen_statements.add(key)
                unique_hypotheses.append(h)
        
        # Select diverse subset using greedy algorithm
        selected = self._select_diverse_subset(
            unique_hypotheses,
            max_hypotheses,
            diversity_weight
        )
        
        # Sort by combined score
        selected.sort(
            key=lambda h: h.confidence + diversity_weight * h.novelty_score,
            reverse=True
        )
        
        return selected
    
    def _select_diverse_subset(
        self,
        hypotheses: list[Hypothesis],
        k: int,
        diversity_weight: float
    ) -> list[Hypothesis]:
        """Select a diverse subset of hypotheses using greedy algorithm."""
        if len(hypotheses) <= k:
            return hypotheses
        
        selected: list[Hypothesis] = []
        remaining = hypotheses.copy()
        
        while len(selected) < k and remaining:
            if not selected:
                # Select highest scoring first
                best = max(remaining, key=lambda h: h.confidence)
            else:
                # Select based on score and diversity
                def combined_score(h: Hypothesis) -> float:
                    base_score = h.confidence
                    # Diversity penalty: lower score if similar to already selected
                    min_similarity = min(
                        self._hypothesis_similarity(h, s)
                        for s in selected
                    )
                    diversity_bonus = (1 - min_similarity) * diversity_weight
                    return base_score + diversity_bonus
                
                best = max(remaining, key=combined_score)
            
            selected.append(best)
            remaining.remove(best)
        
        return selected
    
    def _hypothesis_similarity(self, h1: Hypothesis, h2: Hypothesis) -> float:
        """Calculate similarity between two hypotheses."""
        # Entity overlap
        entities1 = set(e.id for e in h1.entities)
        entities2 = set(e.id for e in h2.entities)
        
        if entities1 and entities2:
            entity_sim = len(entities1 & entities2) / max(len(entities1), len(entities2))
        else:
            entity_sim = 0
        
        # Type similarity
        type_sim = 1.0 if h1.hypothesis_type == h2.hypothesis_type else 0.0
        
        # Statement similarity (simple word overlap)
        words1 = set(h1.statement.lower().split())
        words2 = set(h2.statement.lower().split())
        stopwords = {'the', 'a', 'an', 'in', 'on', 'of', 'and', 'or', 'is', 'will'}
        words1 -= stopwords
        words2 -= stopwords
        
        if words1 and words2:
            text_sim = len(words1 & words2) / max(len(words1), len(words2))
        else:
            text_sim = 0
        
        return 0.4 * entity_sim + 0.3 * type_sim + 0.3 * text_sim
    
    def rank_hypotheses(
        self,
        hypotheses: list[Hypothesis],
        criteria: Optional[dict[str, float]] = None
    ) -> list[tuple[Hypothesis, float]]:
        """
        Rank hypotheses by multiple criteria.
        
        Default criteria: confidence, novelty, testability
        """
        if criteria is None:
            criteria = {
                "confidence": 0.4,
                "novelty": 0.3,
                "testability": 0.3
            }
        
        scored: list[tuple[Hypothesis, float]] = []
        
        for h in hypotheses:
            score = 0.0
            
            if "confidence" in criteria:
                score += criteria["confidence"] * h.confidence
            
            if "novelty" in criteria:
                score += criteria["novelty"] * h.novelty_score
            
            if "testability" in criteria:
                # Score by number of predictions
                testability = min(len(h.testable_predictions) / 3, 1.0)
                score += criteria["testability"] * testability
            
            scored.append((h, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def get_hypothesis_summary(self, hypotheses: list[Hypothesis]) -> dict[str, Any]:
        """Generate summary statistics for a set of hypotheses."""
        type_counts: dict[str, int] = {}
        strategy_counts: dict[str, int] = {}
        
        for h in hypotheses:
            type_counts[h.hypothesis_type.name] = type_counts.get(
                h.hypothesis_type.name, 0
            ) + 1
            strategy_counts[h.source_strategy] = strategy_counts.get(
                h.source_strategy, 0
            ) + 1
        
        return {
            "total_hypotheses": len(hypotheses),
            "type_distribution": type_counts,
            "strategy_distribution": strategy_counts,
            "avg_confidence": np.mean([h.confidence for h in hypotheses]) if hypotheses else 0,
            "avg_novelty": np.mean([h.novelty_score for h in hypotheses]) if hypotheses else 0,
            "high_confidence_count": sum(1 for h in hypotheses if h.confidence > 0.6)
        }


def demo():
    """Demo hypothesis generation."""
    from ..knowledge_graph.scientific_kg import ScientificKnowledgeGraph
    
    # Create knowledge graph with sample data
    kg = ScientificKnowledgeGraph()
    
    sample_texts = [
        "Graphene has high thermal conductivity measured by molecular dynamics.",
        "Silicon exhibits a band gap of 1.1 eV calculated using DFT.",
        "Perovskite materials show interesting electronic properties.",
        "Temperature causes phase transitions in materials.",
        "Strain affects the band gap of semiconductors.",
        "Density functional theory is used to calculate electronic structure.",
        "Monte Carlo simulations predict thermodynamic properties.",
    ]
    
    for text in sample_texts:
        kg.extract_from_text(text)
    
    # Generate hypotheses
    generator = HypothesisGenerator(kg)
    hypotheses = generator.generate(max_hypotheses=10)
    
    print("=== Generated Hypotheses ===\n")
    for i, h in enumerate(hypotheses, 1):
        print(f"{i}. [{h.hypothesis_type.name}] {h.statement}")
        print(f"   Confidence: {h.confidence:.2f} | Novelty: {h.novelty_score:.2f}")
        print(f"   Strategy: {h.source_strategy}")
        print(f"   Predictions:")
        for pred in h.testable_predictions[:2]:
            print(f"     - {pred}")
        print()
    
    # Summary
    print("=== Summary ===")
    summary = generator.get_hypothesis_summary(hypotheses)
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    demo()
