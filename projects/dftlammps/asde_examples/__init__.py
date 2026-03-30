"""
ASDE Examples - Demonstration scripts for the Automatic Scientific Discovery Engine.

This package contains example scripts demonstrating the full ASDE pipeline:
- autonomous_discovery.py: Complete materials discovery workflow
- literature_review_bot.py: Automated literature review generation
"""

from .autonomous_discovery import autonomous_discovery_pipeline, SimulatedExperimentRunner
from .literature_review_bot import (
    LiteratureReviewBot,
    ReviewSection,
    run_literature_review,
)

__all__ = [
    'autonomous_discovery_pipeline',
    'SimulatedExperimentRunner',
    'LiteratureReviewBot',
    'ReviewSection',
    'run_literature_review',
]