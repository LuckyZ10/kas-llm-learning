"""
Knowledge Graph module for ASDE.

Provides tools for building and analyzing scientific knowledge graphs,
mining literature from various sources, and analyzing citation networks.
"""

from .scientific_kg import (
    Entity,
    EntityType,
    Relation,
    RelationType,
    ScientificKnowledgeGraph,
    CausalReasoner,
    EntityExtractor,
    RelationExtractor,
    RegexEntityExtractor,
    PatternRelationExtractor,
)

from .literature_miner import (
    Paper,
    LiteratureMiner,
    ArXivProvider,
    PubMedProvider,
    CrossRefProvider,
)

from .citation_network import (
    CitationNetwork,
    CitationNode,
    CitationEdge,
    build_network_from_search_results,
)

__all__ = [
    # Scientific KG
    'Entity',
    'EntityType',
    'Relation',
    'RelationType',
    'ScientificKnowledgeGraph',
    'CausalReasoner',
    'EntityExtractor',
    'RelationExtractor',
    'RegexEntityExtractor',
    'PatternRelationExtractor',
    # Literature Miner
    'Paper',
    'LiteratureMiner',
    'ArXivProvider',
    'PubMedProvider',
    'CrossRefProvider',
    # Citation Network
    'CitationNetwork',
    'CitationNode',
    'CitationEdge',
    'build_network_from_search_results',
]