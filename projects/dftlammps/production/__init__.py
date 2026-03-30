"""
DFTLammps Production Module
生产运维模块

This module provides production operations and monitoring capabilities.

Modules:
    monitoring: Monitoring and alerting system
    logging: Log analysis and management
    ab_testing: A/B testing framework

Author: DFTLammps Platform Team
Version: 2.0.0
"""

from .monitoring import (
    MonitoringOrchestrator,
    Metric,
    AlertRule,
    AlertSeverity,
    SLIMonitor,
    AnomalyDetector
)

from .logging import (
    LogOrchestrator,
    LogEntry,
    LogLevel,
    LogParser,
    LogStorage
)

from .ab_testing import (
    ABTestOrchestrator,
    Experiment,
    Variant,
    ExperimentMetric,
    StatisticalEngine
)

__version__ = "2.0.0"
__all__ = [
    # Monitoring
    "MonitoringOrchestrator",
    "Metric",
    "AlertRule",
    "AlertSeverity",
    "SLIMonitor",
    "AnomalyDetector",
    
    # Logging
    "LogOrchestrator",
    "LogEntry",
    "LogLevel",
    "LogParser",
    "LogStorage",
    
    # A/B Testing
    "ABTestOrchestrator",
    "Experiment",
    "Variant",
    "ExperimentMetric",
    "StatisticalEngine"
]
