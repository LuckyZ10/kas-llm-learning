"""
Scientific Knowledge Graph Module

Provides entity-relationship extraction and knowledge graph construction
for scientific literature and experimental data.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Protocol, Iterator
from collections import defaultdict
import networkx as nx
from networkx.algorithms import community


class EntityType(Enum):
    """Types of scientific entities."""
    MATERIAL = auto()
    PROPERTY = auto()
    METHOD = auto()
    PHENOMENON = auto()
    CONCEPT = auto()
    VARIABLE = auto()
    UNIT = auto()
    ORGANIZATION = auto()
    PERSON = auto()


class RelationType(Enum):
    """Types of relationships between entities."""
    HAS_PROPERTY = auto()
    MEASURED_BY = auto()
    CAUSES = auto()
    INHIBITS = auto()
    CORRELATES_WITH = auto()
    DEPENDS_ON = auto()
    IS_A = auto()
    PART_OF = auto()
    USES_METHOD = auto()
    DISCOVERED_BY = auto()


@dataclass(frozen=True)
class Entity:
    """A node in the scientific knowledge graph."""
    id: str
    name: str
    entity_type: EntityType
    properties: dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return NotImplemented
        return self.id == other.id


@dataclass(frozen=True)
class Relation:
    """An edge in the scientific knowledge graph."""
    source: str
    target: str
    relation_type: RelationType
    confidence: float = 1.0
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        return hash((self.source, self.target, self.relation_type))


class EntityExtractor(Protocol):
    """Protocol for entity extraction strategies."""
    
    def extract(self, text: str) -> list[Entity]: ...


class RelationExtractor(Protocol):
    """Protocol for relation extraction strategies."""
    
    def extract(self, text: str, entities: list[Entity]) -> list[Relation]: ...


class RegexEntityExtractor:
    """Rule-based entity extractor using regex patterns."""
    
    PATTERNS: dict[EntityType, list[str]] = {
        EntityType.MATERIAL: [
            r'\b([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+)\b',  # Chemical formulas
            r'\b(graphene|silicon|germanium|perovskite|MOF|zeolite)\b',
            r'\b([a-z]+-based\s+materials?)\b',
        ],
        EntityType.PROPERTY: [
            r'\b(band[- ]?gap|conductivity|resistivity|mobility)\b',
            r'\b(thermal conductivity|specific heat|entropy)\b',
            r'\b(yield strength|elastic modulus|hardness)\b',
            r'\b(dielectric constant|permittivity|permeability)\b',
        ],
        EntityType.METHOD: [
            r'\b(DFT|density functional theory)\b',
            r'\b(MD|molecular dynamics)\b',
            r'\b(Monte[- ]?Carlo)\b',
            r'\b(XRD|X-ray diffraction)\b',
            r'\b(TEM|transmission electron microscopy)\b',
            r'\b(annealing|sintering|deposition)\b',
        ],
        EntityType.PHENOMENON: [
            r'\b(phase transition|phase separation)\b',
            r'\b(electron[- ]?phonon coupling)\b',
            r'\b(quantum confinement|quantum tunneling)\b',
            r'\b(superconductivity|ferromagnetism|piezoelectricity)\b',
        ],
        EntityType.VARIABLE: [
            r'\b(temperature|pressure|concentration)\b',
            r'\b(strain|stress|deformation)\b',
            r'\b(frequency|wavelength|energy)\b',
        ],
        EntityType.UNIT: [
            r'\b(eV|kcal/mol|GPa|MPa|K|°C|Pa|atm)\b',
            r'\b(nm|μm|mm|cm|m|Å)\b',
            r'\b(g/cm³|kg/m³|mol/L)\b',
        ],
    }
    
    def __init__(self) -> None:
        self._compiled: dict[EntityType, list[re.Pattern]] = {}
        self._entity_counter: dict[EntityType, int] = defaultdict(int)
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        for etype, patterns in self.PATTERNS.items():
            self._compiled[etype] = [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def extract(self, text: str) -> list[Entity]:
        """Extract entities from text using regex patterns."""
        entities: list[Entity] = []
        seen: set[tuple[str, EntityType]] = set()
        
        for entity_type, patterns in self._compiled.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    name = match.group(0)
                    key = (name.lower(), entity_type)
                    if key not in seen:
                        seen.add(key)
                        self._entity_counter[entity_type] += 1
                        entity = Entity(
                            id=f"{entity_type.name.lower()}_{self._entity_counter[entity_type]}",
                            name=name,
                            entity_type=entity_type,
                            properties={"source_text": match.group(0)}
                        )
                        entities.append(entity)
        
        return entities


class PatternRelationExtractor:
    """Rule-based relation extractor using syntactic patterns."""
    
    RELATION_PATTERNS: dict[RelationType, list[str]] = {
        RelationType.HAS_PROPERTY: [
            r'{MATERIAL}.*?(?:has|exhibits|shows|displays).*?{PROPERTY}',
            r'{PROPERTY}.*?(?:of|in).*?{MATERIAL}',
        ],
        RelationType.MEASURED_BY: [
            r'{PROPERTY}.*?(?:measured|determined|calculated).*?(?:by|using|with).*?{METHOD}',
            r'{METHOD}.*?{PROPERTY}',
        ],
        RelationType.CAUSES: [
            r'{VARIABLE}.*?(?:causes?|leads? to|results? in|induces?)',
            r'(?:due to|because of|caused by).*?{VARIABLE}',
        ],
        RelationType.DEPENDS_ON: [
            r'{PROPERTY}.*?(?:depends? on|varies? with|function of).*?{VARIABLE}',
            r'{VARIABLE}.*?effect.*?{PROPERTY}',
        ],
        RelationType.USES_METHOD: [
            r'(?:using|with|by|via).*?{METHOD}',
            r'{METHOD}.*?used',
        ],
    }
    
    def extract(self, text: str, entities: list[Entity]) -> list[Relation]:
        """Extract relations between entities."""
        relations: list[Relation] = []
        text_lower = text.lower()
        
        # Group entities by type
        by_type: dict[EntityType, list[Entity]] = defaultdict(list)
        for e in entities:
            by_type[e.entity_type].append(e)
        
        # Check for HAS_PROPERTY relations (Material -> Property)
        for material in by_type[EntityType.MATERIAL]:
            for prop in by_type[EntityType.PROPERTY]:
                # Simple proximity check
                mat_pos = text_lower.find(material.name.lower())
                prop_pos = text_lower.find(prop.name.lower())
                
                if mat_pos >= 0 and prop_pos >= 0 and abs(mat_pos - prop_pos) < 200:
                    confidence = 0.5 + 0.5 * (1 - abs(mat_pos - prop_pos) / 200)
                    relations.append(Relation(
                        source=material.id,
                        target=prop.id,
                        relation_type=RelationType.HAS_PROPERTY,
                        confidence=confidence,
                        evidence=[text[max(0, min(mat_pos, prop_pos)-50):
                                      min(len(text), max(mat_pos, prop_pos)+50)]]
                    ))
        
        # Check for MEASURED_BY relations (Property -> Method)
        for prop in by_type[EntityType.PROPERTY]:
            for method in by_type[EntityType.METHOD]:
                prop_pos = text_lower.find(prop.name.lower())
                meth_pos = text_lower.find(method.name.lower())
                
                if prop_pos >= 0 and meth_pos >= 0 and abs(prop_pos - meth_pos) < 150:
                    confidence = 0.4 + 0.4 * (1 - abs(prop_pos - meth_pos) / 150)
                    relations.append(Relation(
                        source=prop.id,
                        target=method.id,
                        relation_type=RelationType.MEASURED_BY,
                        confidence=confidence
                    ))
        
        # Check for DEPENDS_ON relations (Property -> Variable)
        for prop in by_type[EntityType.PROPERTY]:
            for var in by_type[EntityType.VARIABLE]:
                prop_pos = text_lower.find(prop.name.lower())
                var_pos = text_lower.find(var.name.lower())
                
                if prop_pos >= 0 and var_pos >= 0 and abs(prop_pos - var_pos) < 180:
                    # Check for dependency keywords
                    context_start = max(0, min(prop_pos, var_pos) - 30)
                    context_end = min(len(text), max(prop_pos, var_pos) + 30)
                    context = text_lower[context_start:context_end]
                    
                    dep_keywords = ['depend', 'function', 'vary', 'effect', 'influence', 'function']
                    if any(kw in context for kw in dep_keywords):
                        relations.append(Relation(
                            source=prop.id,
                            target=var.id,
                            relation_type=RelationType.DEPENDS_ON,
                            confidence=0.6
                        ))
        
        return relations


class ScientificKnowledgeGraph:
    """
    Knowledge graph for storing and querying scientific entities and relations.
    
    Built on NetworkX for efficient graph operations and community detection.
    """
    
    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self._entities: dict[str, Entity] = {}
        self._relations: set[tuple[str, str, RelationType]] = set()
        self._entity_extractor: EntityExtractor = RegexEntityExtractor()
        self._relation_extractor: RelationExtractor = PatternRelationExtractor()
    
    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the knowledge graph."""
        self._entities[entity.id] = entity
        self.graph.add_node(
            entity.id,
            name=entity.name,
            entity_type=entity.entity_type.name,
            **entity.properties
        )
    
    def add_relation(self, relation: Relation) -> None:
        """Add a relation to the knowledge graph."""
        key = (relation.source, relation.target, relation.relation_type)
        if key not in self._relations:
            self._relations.add(key)
            self.graph.add_edge(
                relation.source,
                relation.target,
                relation_type=relation.relation_type.name,
                confidence=relation.confidence,
                evidence=relation.evidence,
                **relation.metadata
            )
    
    def extract_from_text(self, text: str, source: Optional[str] = None) -> None:
        """Extract entities and relations from text and add to graph."""
        entities = self._entity_extractor.extract(text)
        
        # Add entities
        for entity in entities:
            if entity.id not in self._entities:
                if source:
                    entity.properties['source_document'] = source
                self.add_entity(entity)
        
        # Extract and add relations
        relations = self._relation_extractor.extract(text, entities)
        for relation in relations:
            if relation.source in self._entities and relation.target in self._entities:
                self.add_relation(relation)
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by ID."""
        return self._entities.get(entity_id)
    
    def get_neighbors(
        self,
        entity_id: str,
        relation_type: Optional[RelationType] = None
    ) -> list[tuple[Entity, Relation]]:
        """Get neighboring entities and their relations."""
        if entity_id not in self._entities:
            return []
        
        results: list[tuple[Entity, Relation]] = []
        
        for successor in self.graph.successors(entity_id):
            edge_data = self.graph.edges[entity_id, successor]
            rel_type = RelationType[edge_data['relation_type']]
            
            if relation_type is None or rel_type == relation_type:
                relation = Relation(
                    source=entity_id,
                    target=successor,
                    relation_type=rel_type,
                    confidence=edge_data.get('confidence', 1.0),
                    evidence=edge_data.get('evidence', [])
                )
                entity = self._entities[successor]
                results.append((entity, relation))
        
        return results
    
    def find_path(
        self,
        source: str,
        target: str,
        max_length: int = 5
    ) -> Optional[list[Entity]]:
        """Find a path between two entities."""
        try:
            path = nx.shortest_path(
                self.graph.to_undirected(),
                source=source,
                target=target
            )
            if len(path) <= max_length + 1:
                return [self._entities[n] for n in path]
            return None
        except nx.NetworkXNoPath:
            return None
    
    def find_communities(self) -> list[set[str]]:
        """Detect communities in the knowledge graph using Louvain algorithm."""
        if len(self.graph) == 0:
            return []
        
        # Convert to undirected for community detection
        undirected = self.graph.to_undirected()
        communities_gen = community.louvain_communities(undirected, seed=42)
        return [set(c) for c in communities_gen]
    
    def get_central_entities(self, top_k: int = 10) -> list[tuple[Entity, float]]:
        """Get most central entities by PageRank."""
        if len(self.graph) == 0:
            return []
        
        pagerank = nx.pagerank(self.graph)
        sorted_entities = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        
        return [
            (self._entities[eid], score)
            for eid, score in sorted_entities[:top_k]
            if eid in self._entities
        ]
    
    def query(
        self,
        entity_type: Optional[EntityType] = None,
        relation_type: Optional[RelationType] = None,
        min_confidence: float = 0.0
    ) -> Iterator[tuple[Entity, Entity, Relation]]:
        """Query the knowledge graph."""
        for source_id, target_id, edge_data in self.graph.edges(data=True):
            if edge_data.get('confidence', 1.0) < min_confidence:
                continue
            
            if relation_type and edge_data.get('relation_type') != relation_type.name:
                continue
            
            source = self._entities.get(source_id)
            target = self._entities.get(target_id)
            
            if source is None or target is None:
                continue
            
            if entity_type and source.entity_type != entity_type:
                continue
            
            relation = Relation(
                source=source_id,
                target=target_id,
                relation_type=RelationType[edge_data['relation_type']],
                confidence=edge_data.get('confidence', 1.0),
                evidence=edge_data.get('evidence', [])
            )
            
            yield (source, target, relation)
    
    def get_material_properties(self, material_name: str) -> list[tuple[Entity, float]]:
        """Get properties associated with a material."""
        results: list[tuple[Entity, float]] = []
        
        # Find material entity
        material = None
        for entity in self._entities.values():
            if entity.entity_type == EntityType.MATERIAL and \
               entity.name.lower() == material_name.lower():
                material = entity
                break
        
        if material is None:
            return results
        
        # Get HAS_PROPERTY relations
        for entity, relation in self.get_neighbors(material.id, RelationType.HAS_PROPERTY):
            results.append((entity, relation.confidence))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize knowledge graph to dictionary."""
        return {
            "entities": [
                {
                    "id": e.id,
                    "name": e.name,
                    "type": e.entity_type.name,
                    "properties": e.properties
                }
                for e in self._entities.values()
            ],
            "relations": [
                {
                    "source": r.source,
                    "target": r.target,
                    "type": r.relation_type.name,
                    "confidence": r.confidence
                }
                for r in [
                    Relation(
                        source=s,
                        target=t,
                        relation_type=RelationType[d['relation_type']],
                        confidence=d.get('confidence', 1.0)
                    )
                    for s, t, d in self.graph.edges(data=True)
                ]
            ]
        }
    
    def save(self, filepath: str) -> None:
        """Save knowledge graph to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> ScientificKnowledgeGraph:
        """Load knowledge graph from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        kg = cls()
        
        # Load entities
        for e_data in data['entities']:
            entity = Entity(
                id=e_data['id'],
                name=e_data['name'],
                entity_type=EntityType[e_data['type']],
                properties=e_data.get('properties', {})
            )
            kg.add_entity(entity)
        
        # Load relations
        for r_data in data['relations']:
            relation = Relation(
                source=r_data['source'],
                target=r_data['target'],
                relation_type=RelationType[r_data['type']],
                confidence=r_data.get('confidence', 1.0)
            )
            kg.add_relation(relation)
        
        return kg
    
    def get_statistics(self) -> dict[str, Any]:
        """Get graph statistics."""
        entity_counts = defaultdict(int)
        for entity in self._entities.values():
            entity_counts[entity.entity_type.name] += 1
        
        relation_counts = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            relation_counts[data['relation_type']] += 1
        
        return {
            "total_entities": len(self._entities),
            "total_relations": len(self._relations),
            "entity_types": dict(entity_counts),
            "relation_types": dict(relation_counts),
            "density": nx.density(self.graph),
            "is_connected": nx.is_weakly_connected(self.graph) if len(self.graph) > 0 else False,
            "num_communities": len(self.find_communities())
        }


class CausalReasoner:
    """
    Performs causal reasoning over the knowledge graph.
    """
    
    def __init__(self, kg: ScientificKnowledgeGraph) -> None:
        self.kg = kg
    
    def find_causal_chain(
        self,
        cause: str,
        effect: str,
        max_depth: int = 5
    ) -> Optional[list[Entity]]:
        """Find causal chain from cause to effect."""
        # Find entity IDs
        cause_id = None
        effect_id = None
        
        for entity in self.kg._entities.values():
            if entity.name.lower() == cause.lower():
                cause_id = entity.id
            if entity.name.lower() == effect.lower():
                effect_id = entity.id
        
        if cause_id is None or effect_id is None:
            return None
        
        # Use BFS to find path
        return self.kg.find_path(cause_id, effect_id, max_depth)
    
    def identify_confounders(
        self,
        variable1: str,
        variable2: str
    ) -> list[Entity]:
        """Identify potential confounding variables."""
        confounders: list[Entity] = []
        
        # Find common causes
        v1_causes: set[str] = set()
        v2_causes: set[str] = set()
        
        for source, target, relation in self.kg.query(relation_type=RelationType.CAUSES):
            if target.name.lower() == variable1.lower():
                v1_causes.add(source.id)
            if target.name.lower() == variable2.lower():
                v2_causes.add(source.id)
        
        common = v1_causes & v2_causes
        for entity_id in common:
            entity = self.kg.get_entity(entity_id)
            if entity:
                confounders.append(entity)
        
        return confounders
    
    def suggest_interventions(self, target_property: str) -> list[tuple[Entity, float]]:
        """Suggest variables that could intervene on a target property."""
        suggestions: list[tuple[Entity, float]] = []
        
        # Find entities with DEPENDS_ON relation to target
        for source, target, relation in self.kg.query(relation_type=RelationType.DEPENDS_ON):
            if target.name.lower() == target_property.lower():
                suggestions.append((source, relation.confidence))
        
        return sorted(suggestions, key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    # Demo
    kg = ScientificKnowledgeGraph()
    
    sample_text = """
    Graphene exhibits exceptional thermal conductivity measured by molecular dynamics.
    The band gap of silicon depends on temperature and strain. 
    DFT calculations show that perovskite materials have interesting electronic properties.
    Phase transitions in materials are caused by temperature changes.
    """
    
    kg.extract_from_text(sample_text, source="demo")
    
    print("Knowledge Graph Statistics:")
    stats = kg.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nExtracted Entities:")
    for entity in kg._entities.values():
        print(f"  {entity.id}: {entity.name} ({entity.entity_type.name})")
    
    print("\nMaterial Properties (graphene):")
    for prop, conf in kg.get_material_properties("graphene"):
        print(f"  {prop.name}: confidence={conf:.2f}")
    
    print("\nCentral Entities (PageRank):")
    for entity, score in kg.get_central_entities(top_k=5):
        print(f"  {entity.name}: {score:.4f}")
