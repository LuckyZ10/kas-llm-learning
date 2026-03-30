"""
智能文献综述与研究趋势分析系统
Literature Survey System

功能：
- 文献抓取与解析：arXiv、PubMed、CrossRef、Semantic Scholar
- 智能分析：主题建模、趋势分析、方法提取、知识图谱
- 综述生成：自动生成结构化综述报告
- 实时预警：关键词订阅、新论文推送、引用提醒
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from .fetcher import LiteratureFetcher
from .analysis.topic_modeling import TopicModeler
from .analysis.trend_analysis import TrendAnalyzer
from .analysis.knowledge_graph import KnowledgeGraphBuilder
from .generator.review_generator import ReviewGenerator
from .alert.alert_system import AlertSystem
from .config.database import DatabaseManager

__all__ = [
    'LiteratureFetcher',
    'TopicModeler',
    'TrendAnalyzer',
    'KnowledgeGraphBuilder',
    'ReviewGenerator',
    'AlertSystem',
    'DatabaseManager',
]
