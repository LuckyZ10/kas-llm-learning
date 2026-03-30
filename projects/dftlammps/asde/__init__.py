"""
ASDE - Automatic Scientific Discovery Engine

A comprehensive framework for automated scientific hypothesis generation,
experimental planning, result analysis, and paper writing.
"""

from .hypothesis_generator import (
    Hypothesis,
    HypothesisType,
    HypothesisGenerator,
    HypothesisStrategy,
    CausalChainStrategy,
    AnalogicalReasoningStrategy,
    CompositionalStrategy,
    AbductiveInferenceStrategy,
    GapDrivenStrategy,
)

from .experiment_planner import (
    ExperimentalVariable,
    VariableType,
    ExperimentalCondition,
    ExperimentResult,
    ExperimentPlanner,
    HypothesisTester,
    SurrogateModel,
    GaussianProcessSurrogate,
    AcquisitionFunction,
    ExpectedImprovement,
    UpperConfidenceBound,
)

from .result_analyzer import (
    ResultAnalyzer,
    TestResult,
    EffectSizeResult,
    DatasetSummary,
    TestType,
    EffectSizeType,
)

from .paper_writer import (
    PaperWriter,
    ScientificPaper,
    PaperSection,
)

__version__ = "0.1.0"
__all__ = [
    # Hypothesis Generator
    'Hypothesis',
    'HypothesisType',
    'HypothesisGenerator',
    'HypothesisStrategy',
    'CausalChainStrategy',
    'AnalogicalReasoningStrategy',
    'CompositionalStrategy',
    'AbductiveInferenceStrategy',
    'GapDrivenStrategy',
    # Experiment Planner
    'ExperimentalVariable',
    'VariableType',
    'ExperimentalCondition',
    'ExperimentResult',
    'ExperimentPlanner',
    'HypothesisTester',
    'SurrogateModel',
    'GaussianProcessSurrogate',
    'AcquisitionFunction',
    'ExpectedImprovement',
    'UpperConfidenceBound',
    # Result Analyzer
    'ResultAnalyzer',
    'TestResult',
    'EffectSizeResult',
    'DatasetSummary',
    'TestType',
    'EffectSizeType',
    # Paper Writer
    'PaperWriter',
    'ScientificPaper',
    'PaperSection',
]
