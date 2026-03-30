"""
Experimental Data Validation Module
===================================
实验数据对比与验证模块

This module provides comprehensive tools for comparing computational results
with experimental data, including:

1. **Data Connectors**: Interfaces to various experimental data sources
   - XRD pattern loaders
   - Electrochemical data parsers
   - Spectroscopy data readers
   - TEM image processors
   - Database interfaces (ICSD, Materials Project)

2. **Analysis Tools**: Comparison and validation algorithms
   - Structure comparison (XRD pattern matching)
   - Performance comparison (electrochemical properties)
   - Statistical analysis (MAE, RMSE, R²)
   - Visualization tools

3. **Validation Workflows**: Automated validation pipelines
   - Auto comparison workflows
   - Report generation
   - Outlier detection
   - Model parameter feedback

4. **Uncertainty Quantification**: Error analysis
   - Error propagation
   - Confidence intervals
   - Sensitivity analysis

Example Usage:
    >>> from experimental_validation import ValidationWorkflow
    >>> workflow = ValidationWorkflow()
    >>> workflow.load_computational_data("vasp_output/")
    >>> workflow.load_experimental_data("xrd_data.csv", data_type="xrd")
    >>> results = workflow.run_validation()
    >>> workflow.generate_report("validation_report.pdf")

Authors: DFT-MD Research Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "DFT-MD Research Team"

# Import main classes for convenient access
from .connectors.xrd_connector import XRDConnector
from .connectors.electrochemical_connector import ElectrochemicalConnector
from .connectors.spectroscopy_connector import SpectroscopyConnector
from .connectors.tem_connector import TEMConnector
from .connectors.database_connector import DatabaseConnector, ICSDConnector, MaterialsProjectConnector

from .analyzers.structure_analyzer import XRDComparator, StructureComparator
from .analyzers.performance_analyzer import ElectrochemicalComparator, PropertyComparator
from .analyzers.statistical_analyzer import StatisticalAnalyzer
from .analyzers.visualizer import ValidationVisualizer

from .workflows.validation_workflow import ValidationWorkflow
from .workflows.batch_validator import BatchValidator

from .uncertainty.error_propagation import ErrorPropagator
from .uncertainty.error_propagation import ErrorPropagator, ConfidenceIntervalEstimator, SensitivityAnalyzer

__all__ = [
    # Connectors
    'XRDConnector',
    'ElectrochemicalConnector',
    'SpectroscopyConnector',
    'TEMConnector',
    'DatabaseConnector',
    'ICSDConnector',
    'MaterialsProjectConnector',
    
    # Analyzers
    'XRDComparator',
    'StructureComparator',
    'ElectrochemicalComparator',
    'PropertyComparator',
    'StatisticalAnalyzer',
    'ValidationVisualizer',
    
    # Workflows
    'ValidationWorkflow',
    'BatchValidator',
    
    # Uncertainty
    'ErrorPropagator',
    'ConfidenceIntervalEstimator',
    'SensitivityAnalyzer',
]
