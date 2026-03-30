# -*- coding: utf-8 -*-
"""
文献综述系统配置文件
"""

import os
from pathlib import Path

# 基础路径
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
PAPERS_DIR = DATA_DIR / "papers"
REPORTS_DIR = DATA_DIR / "reports"

# 创建目录
for dir_path in [DATA_DIR, CACHE_DIR, PAPERS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 数据源配置
DATA_SOURCES = {
    "arxiv": {
        "name": "arXiv",
        "base_url": "http://export.arxiv.org/api/query",
        "max_results": 100,
        "rate_limit": 3,  # 秒
        "categories": [
            "cond-mat.mtrl-sci",  # 材料科学
            "physics.chem-ph",     # 化学物理
            "cond-mat.mes-hall",   # 介观物理
            "physics.comp-ph",     # 计算物理
        ]
    },
    "pubmed": {
        "name": "PubMed",
        "base_url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        "api_key": os.getenv("PUBMED_API_KEY", ""),
        "rate_limit": 0.34,  # 秒 (3 requests per second without API key)
        "max_results": 100
    },
    "crossref": {
        "name": "CrossRef",
        "base_url": "https://api.crossref.org/works",
        "rate_limit": 0.1,
        "max_results": 100
    },
    "semantic_scholar": {
        "name": "Semantic Scholar",
        "base_url": "https://api.semanticscholar.org/graph/v1",
        "api_key": os.getenv("SEMANTIC_SCHOLAR_API_KEY", ""),
        "rate_limit": 1,
        "max_results": 100
    }
}

# 搜索关键词配置（DFT/分子动力学相关）
SEARCH_KEYWORDS = {
    "dft": [
        "density functional theory",
        "DFT calculation",
        "first-principles",
        "ab initio",
        "electronic structure",
        "band structure",
        "density of states"
    ],
    "md": [
        "molecular dynamics",
        "MD simulation",
        "atomistic simulation",
        "interatomic potential",
        "force field",
        "LAMMPS",
        "GROMACS"
    ],
    "ml_potentials": [
        "machine learning potential",
        "neural network potential",
        "MLIP",
        "deep potential",
        "behler-parrinello",
        "Gaussian approximation potential",
        "SNAP"
    ],
    "materials": [
        "solid electrolyte",
        "lithium battery",
        "battery material",
        "ionic conductivity",
        "diffusion coefficient",
        "phase transition"
    ]
}

# 分析配置
ANALYSIS_CONFIG = {
    "topic_modeling": {
        "method": "bertopic",  # 或 "lda"
        "n_topics": 10,
        "min_topic_size": 5,
        "language": "english"
    },
    "trend_analysis": {
        "window_size": 3,  # 年
        "min_papers": 5,
        "smoothing_factor": 0.3
    },
    "knowledge_graph": {
        "min_cooccurrence": 3,
        "max_nodes": 200,
        "edge_threshold": 0.1
    },
    "method_extraction": {
        "software_patterns": [
            "VASP", "Quantum ESPRESSO", "Gaussian", "ABINIT",
            "CASTEP", "GPAW", "Siesta", "WIEN2k",
            "LAMMPS", "GROMACS", "NAMD", "OpenMM",
            "ASE", "Pymatgen", "OVITO"
        ],
        "method_patterns": [
            "DFT", "DFT-D", "vdW-DF", "SCAN", "PBE", "B3LYP",
            "AIMD", "DFT-MD", "classical MD",
            "NEB", "CI-NEB", "metadynamics", "umbrella sampling",
            "cluster expansion", "CALPHAD"
        ]
    }
}

# 综述生成配置
REPORT_CONFIG = {
    "sections": [
        "abstract",
        "introduction",
        "methodology_overview",
        "key_findings",
        "trend_analysis",
        "method_comparison",
        "research_gaps",
        "future_outlook",
        "references"
    ],
    "max_papers_per_section": 20,
    "summary_length": 500,  # 词
    "citation_style": "ieee"  # 或 "apa", "nature"
}

# 预警配置
ALERT_CONFIG = {
    "check_interval": 86400,  # 秒 (每天)
    "max_new_papers": 50,
    "notification_channels": ["email", "webhook"],
    "weekly_digest_day": 0,  # 周一
    "citation_threshold": 10  # 被引次数阈值
}

# 缓存配置
CACHE_CONFIG = {
    "ttl": 86400 * 7,  # 7天
    "max_size": 1000,  # 最大条目数
    "compression": True
}

# 数据库配置（使用SQLite）
DATABASE_CONFIG = {
    "path": DATA_DIR / "literature.db",
    "backup_interval": 86400 * 7  # 7天
}

# 日志配置
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": DATA_DIR / "literature_survey.log"
}
