"""
Explainability modules for GNNs
================================

Implements:
- GNNExplainer: Generates explanations by identifying important subgraphs
- PGExplainer: Parameterized explainer for global explanations
"""

from .gnn_explainer import GNNExplainer
from .pg_explainer import PGExplainer

__all__ = ["GNNExplainer", "PGExplainer"]
