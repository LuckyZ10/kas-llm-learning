"""
LLM Integration Module for DFT-LAMMPS Scientific Platform.

This module provides unified interfaces for integrating Large Language Models
into scientific research workflows, including hypothesis explanation, reasoning
enhancement, and paper writing assistance.
"""

from .llm_interface import (
    LLMProvider,
    LLMConfig,
    UnifiedLLMInterface,
    AsyncLLMInterface,
    StreamResponse,
    Message,
    Conversation,
    PromptEngineer,
)

from .hypothesis_explainer import (
    HypothesisExplainer,
    ExplanationResult,
    NumericalContext,
    NumericalValue,
    ExplanationStyle,
    Hypothesis,
    ResultType,
)

from .scientific_reasoning import (
    ScientificReasoningEngine,
    CausalInference,
    CounterfactualAnalysis,
    ReasoningChain,
    Evidence,
    EvidenceType,
    ConfidenceLevel,
)

from .paper_assistant import (
    PaperAssistant,
    PaperSection,
    ReviewResult,
    CitationManager,
    WritingStyle,
    CitationStyle,
    Citation,
    SectionDraft,
)

from .chat_interface import (
    ScientificChatInterface,
    ChatSession,
    ScientificContext,
    MultiTurnConversation,
    ChatMode,
    ResponseFormat,
)

__version__ = "1.0.0"
__all__ = [
    # LLM Interface
    "LLMProvider",
    "LLMConfig",
    "UnifiedLLMInterface",
    "AsyncLLMInterface",
    "StreamResponse",
    "Message",
    "Conversation",
    "PromptEngineer",
    # Hypothesis Explainer
    "HypothesisExplainer",
    "ExplanationResult",
    "NumericalContext",
    "NumericalValue",
    "ExplanationStyle",
    "Hypothesis",
    "ResultType",
    # Scientific Reasoning
    "ScientificReasoningEngine",
    "CausalInference",
    "CounterfactualAnalysis",
    "ReasoningChain",
    "Evidence",
    "EvidenceType",
    "ConfidenceLevel",
    # Paper Assistant
    "PaperAssistant",
    "PaperSection",
    "ReviewResult",
    "CitationManager",
    "WritingStyle",
    "CitationStyle",
    "Citation",
    "SectionDraft",
    # Chat Interface
    "ScientificChatInterface",
    "ChatSession",
    "ScientificContext",
    "MultiTurnConversation",
    "ChatMode",
    "ResponseFormat",
]
