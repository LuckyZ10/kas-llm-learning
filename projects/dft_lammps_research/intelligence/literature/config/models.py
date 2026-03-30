"""
文献综述系统 - 数据库模型
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
import json


@dataclass
class Author:
    """作者信息"""
    name: str
    affiliation: Optional[str] = None
    email: Optional[str] = None
    orcid: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Citation:
    """引用信息"""
    cited_by: str  # 引用本文的论文ID
    citation_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cited_by": self.cited_by,
            "citation_count": self.citation_count,
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class Paper:
    """论文数据模型"""
    id: str
    title: str
    authors: List[Author]
    abstract: str
    publication_date: datetime
    journal: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pmid: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    
    # 内容
    keywords: List[str] = field(default_factory=list)
    full_text: Optional[str] = None
    sections: Dict[str, str] = field(default_factory=dict)
    
    # 元数据
    citation_count: int = 0
    reference_count: int = 0
    references: List[str] = field(default_factory=list)
    
    # 分类
    categories: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    
    # 方法信息
    methods: List[str] = field(default_factory=list)
    software: List[str] = field(default_factory=list)
    datasets: List[str] = field(default_factory=list)
    
    # 来源
    source: str = ""  # arxiv, pubmed, crossref等
    fetched_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # 分析结果
    embeddings: Optional[List[float]] = None
    sentiment_score: Optional[float] = None
    importance_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "title": self.title,
            "authors": [a.to_dict() for a in self.authors],
            "abstract": self.abstract,
            "publication_date": self.publication_date.isoformat(),
            "journal": self.journal,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "pmid": self.pmid,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "keywords": self.keywords,
            "full_text": self.full_text,
            "sections": self.sections,
            "citation_count": self.citation_count,
            "reference_count": self.reference_count,
            "references": self.references,
            "categories": self.categories,
            "topics": self.topics,
            "methods": self.methods,
            "software": self.software,
            "datasets": self.datasets,
            "source": self.source,
            "fetched_at": self.fetched_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "sentiment_score": self.sentiment_score,
            "importance_score": self.importance_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Paper':
        """从字典创建"""
        authors = [Author(**a) for a in data.get("authors", [])]
        return cls(
            id=data["id"],
            title=data["title"],
            authors=authors,
            abstract=data["abstract"],
            publication_date=datetime.fromisoformat(data["publication_date"]),
            journal=data.get("journal"),
            doi=data.get("doi"),
            arxiv_id=data.get("arxiv_id"),
            pmid=data.get("pmid"),
            url=data.get("url"),
            pdf_url=data.get("pdf_url"),
            keywords=data.get("keywords", []),
            full_text=data.get("full_text"),
            sections=data.get("sections", {}),
            citation_count=data.get("citation_count", 0),
            reference_count=data.get("reference_count", 0),
            references=data.get("references", []),
            categories=data.get("categories", []),
            topics=data.get("topics", []),
            methods=data.get("methods", []),
            software=data.get("software", []),
            datasets=data.get("datasets", []),
            source=data.get("source", ""),
            fetched_at=datetime.fromisoformat(data.get("fetched_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat())),
            sentiment_score=data.get("sentiment_score"),
            importance_score=data.get("importance_score")
        )
    
    def get_author_names(self) -> str:
        """获取作者名字列表"""
        return ", ".join([a.name for a in self.authors])
    
    def get_first_author(self) -> Optional[str]:
        """获取第一作者"""
        if self.authors:
            return self.authors[0].name
        return None


@dataclass
class ResearchTrend:
    """研究趋势"""
    topic: str
    year: int
    paper_count: int
    citation_count: int
    avg_citations: float
    growth_rate: float
    top_papers: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MethodComparison:
    """方法对比"""
    method_name: str
    paper_count: int
    avg_performance: Optional[float] = None
    datasets_used: List[str] = field(default_factory=list)
    software_used: List[str] = field(default_factory=list)
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchGap:
    """研究空白"""
    area: str
    description: str
    evidence: List[str] = field(default_factory=list)
    potential_impact: str = ""
    suggested_approaches: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LiteratureReview:
    """文献综述报告"""
    id: str
    title: str
    query: str
    created_at: datetime
    papers: List[Paper] = field(default_factory=list)
    
    # 分析结果
    topics: List[str] = field(default_factory=list)
    trends: List[ResearchTrend] = field(default_factory=list)
    methods: List[MethodComparison] = field(default_factory=list)
    gaps: List[ResearchGap] = field(default_factory=list)
    
    # 报告内容
    summary: str = ""
    sections: Dict[str, str] = field(default_factory=dict)
    
    # 统计
    total_papers: int = 0
    date_range: tuple = field(default_factory=lambda: (None, None))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "query": self.query,
            "created_at": self.created_at.isoformat(),
            "papers": [p.to_dict() for p in self.papers],
            "topics": self.topics,
            "trends": [t.to_dict() for t in self.trends],
            "methods": [m.to_dict() for m in self.methods],
            "gaps": [g.to_dict() for g in self.gaps],
            "summary": self.summary,
            "sections": self.sections,
            "total_papers": len(self.papers),
            "date_range": self.date_range
        }


@dataclass
class AlertSubscription:
    """预警订阅"""
    id: str
    name: str
    keywords: List[str]
    authors: List[str] = field(default_factory=list)
    journals: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    min_citations: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_check: Optional[datetime] = None
    is_active: bool = True
    notification_email: Optional[str] = None
    webhook_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "keywords": self.keywords,
            "authors": self.authors,
            "journals": self.journals,
            "categories": self.categories,
            "min_citations": self.min_citations,
            "created_at": self.created_at.isoformat(),
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "is_active": self.is_active,
            "notification_email": self.notification_email,
            "webhook_url": self.webhook_url
        }


@dataclass
class AlertNotification:
    """预警通知"""
    id: str
    subscription_id: str
    type: str  # new_paper, citation_alert, weekly_digest
    papers: List[Paper] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    is_read: bool = False
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "subscription_id": self.subscription_id,
            "type": self.type,
            "papers": [p.to_dict() for p in self.papers],
            "created_at": self.created_at.isoformat(),
            "is_read": self.is_read,
            "message": self.message
        }
