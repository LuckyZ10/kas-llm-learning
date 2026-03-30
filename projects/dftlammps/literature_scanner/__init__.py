"""
DFT-LAMMPS Literature Scanner

文献追踪和分析系统, 自动监控arXiv等材料科学文献,
识别研究趋势, 评估代码可复现性, 自动生成集成代码。

Modules:
    - arxiv_monitor: arXiv材料/计算/AI领域自动监控
    - paper_analyzer: 论文结构分析方法提取代码复现评估
    - trend_detector: 趋势检测新兴方法识别
    - auto_importer: 自动将新方法集成到平台
"""

__version__ = "0.1.0"

from .arxiv_monitor import (
    ArxivMonitor, ArxivPaper
)
from .paper_analyzer import (
    PaperAnalyzer, PaperAnalysis, MethodSection,
    CodeSnippet, assess_reproducibility
)
from .trend_detector import (
    TrendDetector, Trend, TrendReport,
    detect_trends, predict_future_trends
)
from .auto_importer import (
    AutoImporter, IntegrationPlan, CodeTemplate,
    analyze_method, generate_code
)

__all__ = [
    # arXiv Monitor
    'ArxivMonitor', 'ArxivPaper',
    # Paper Analyzer
    'PaperAnalyzer', 'PaperAnalysis', 'MethodSection', 'CodeSnippet',
    # Trend Detector
    'TrendDetector', 'Trend', 'TrendReport',
    # Auto Importer
    'AutoImporter', 'IntegrationPlan', 'CodeTemplate',
]
