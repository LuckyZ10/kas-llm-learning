"""
arxiv_monitor.py
arXiv材料/计算/AI领域自动监控

自动获取arXiv最新论文, 筛选与材料科学、计算化学、AI相关的研究。
支持关键词过滤、作者追踪和智能摘要。

References:
- arXiv API: https://arxiv.org/help/api
- 2024进展: LLM辅助论文筛选和摘要
"""

import requests
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import time
import re
from collections import defaultdict
import feedparser


@dataclass
class ArxivPaper:
    """arXiv论文数据结构"""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published: datetime
    updated: datetime
    doi: Optional[str] = None
    journal_ref: Optional[str] = None
    primary_category: str = ""
    pdf_url: str = ""
    relevance_score: float = 0.0
    keywords_matched: List[str] = field(default_factory=list)


class ArxivMonitor:
    """
    arXiv监控器
    
    自动监控arXiv指定类别的最新论文
    """
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    # 材料科学相关类别
    RELEVANT_CATEGORIES = [
        'cond-mat.mtrl-sci',  # 材料科学
        'cond-mat.mes-hall',  # 介观系统和霍尔效应
        'cond-mat.str-el',    # 强关联电子
        'cond-mat.supr-con',  # 超导
        'physics.chem-ph',    # 化学物理
        'physics.comp-ph',    # 计算物理
        'cs.LG',              # 机器学习
        'cs.AI',              # 人工智能
        'cs.CL',              # 计算语言
        'stat.ML',            # 统计学习
    ]
    
    # 关键词库
    KEYWORDS = {
        'materials': [
            'battery', 'cathode', 'anode', 'electrolyte',
            'catalyst', 'perovskite', 'oxide', 'alloy',
            'semiconductor', 'superconductor', 'magnetic',
            'ferroelectric', 'piezoelectric', 'thermoelectric',
            'photovoltaic', 'solar cell', 'fuel cell',
            'Li-ion', 'Na-ion', 'solid-state',
            '2D materials', 'graphene', 'MXene', 'TMD',
            'MOF', 'COF', 'zeolite', 'nanostructure'
        ],
        'computation': [
            'DFT', 'density functional', 'molecular dynamics',
            'Monte Carlo', 'ab initio', 'first-principles',
            'machine learning potential', 'MLP', 'force field',
            'high-throughput', 'computational screening',
            'crystal structure prediction', 'phase diagram',
            'CALPHAD', 'finite element', 'multiscale'
        ],
        'ai_ml': [
            'deep learning', 'neural network', 'graph neural network',
            'transformer', 'GPT', 'large language model',
            'generative AI', 'diffusion model', 'reinforcement learning',
            'active learning', 'Bayesian optimization', 'surrogate model',
            'feature engineering', 'descriptor', 'representation learning',
            'transfer learning', 'few-shot learning', 'self-supervised'
        ],
        'methods': [
            'XRD', 'TEM', 'SEM', 'AFM', 'STM', 'spectroscopy',
            'NMR', 'XAS', 'XPS', 'Raman', 'FTIR',
            'electrochemical', 'impedance', 'cyclic voltammetry'
        ]
    }
    
    def __init__(
        self,
        search_categories: Optional[List[str]] = None,
        keywords: Optional[Dict[str, List[str]]] = None,
        authors_to_track: Optional[List[str]] = None
    ):
        self.search_categories = search_categories or self.RELEVANT_CATEGORIES
        self.keywords = keywords or self.KEYWORDS
        self.authors_to_track = authors_to_track or []
        
        self.paper_history: List[ArxivPaper] = []
        self.last_check: Optional[datetime] = None
        
    def fetch_recent_papers(
        self,
        days_back: int = 7,
        max_results: int = 100
    ) -> List[ArxivPaper]:
        """
        获取最近论文
        
        Args:
            days_back: 回溯天数
            max_results: 最大结果数
        """
        papers = []
        
        # 构建查询
        category_query = ' OR '.join(f'cat:{cat}' for cat in self.search_categories)
        
        # 日期范围
        start_date = datetime.now() - timedelta(days=days_back)
        date_query = f"submittedDate:[{start_date.strftime('%Y%m%d')}0000 TO NOW]"
        
        query = f"({category_query}) AND {date_query}"
        
        # 分批获取
        batch_size = 100
        for start in range(0, max_results, batch_size):
            params = {
                'search_query': query,
                'start': start,
                'max_results': min(batch_size, max_results - start),
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            try:
                response = requests.get(self.BASE_URL, params=params, timeout=30)
                response.raise_for_status()
                
                batch_papers = self._parse_feed(response.text)
                papers.extend(batch_papers)
                
                # 礼貌性等待
                time.sleep(3)
                
            except Exception as e:
                print(f"Error fetching papers: {e}")
                break
        
        # 去重和筛选
        papers = self._deduplicate_and_filter(papers)
        
        self.last_check = datetime.now()
        self.paper_history.extend(papers)
        
        return papers
    
    def _parse_feed(self, feed_xml: str) -> List[ArxivPaper]:
        """解析arXiv XML feed"""
        papers = []
        
        try:
            # 注册命名空间
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            root = ET.fromstring(feed_xml)
            
            for entry in root.findall('atom:entry', namespaces):
                try:
                    # 提取基本信息
                    arxiv_id = entry.find('atom:id', namespaces)
                    arxiv_id = arxiv_id.text.split('/')[-1] if arxiv_id else ""
                    
                    title = entry.find('atom:title', namespaces)
                    title = title.text.strip() if title else ""
                    
                    abstract = entry.find('atom:summary', namespaces)
                    abstract = abstract.text.strip() if abstract else ""
                    
                    # 作者
                    authors = []
                    for author in entry.findall('atom:author', namespaces):
                        name = author.find('atom:name', namespaces)
                        if name is not None:
                            authors.append(name.text)
                    
                    # 类别
                    categories = []
                    primary_cat = ""
                    for cat in entry.findall('atom:category', namespaces):
                        term = cat.get('term', '')
                        if term:
                            categories.append(term)
                            if not primary_cat:
                                primary_cat = term
                    
                    # 日期
                    published = entry.find('atom:published', namespaces)
                    published = datetime.fromisoformat(published.text.replace('Z', '+00:00')) if published else datetime.now()
                    
                    updated = entry.find('atom:updated', namespaces)
                    updated = datetime.fromisoformat(updated.text.replace('Z', '+00:00')) if updated else published
                    
                    # PDF链接
                    pdf_url = f"http://arxiv.org/pdf/{arxiv_id}.pdf"
                    
                    # DOI和期刊引用
                    doi_elem = entry.find('arxiv:doi', namespaces)
                    doi = doi_elem.text if doi_elem is not None else None
                    
                    journal_elem = entry.find('arxiv:journal_ref', namespaces)
                    journal_ref = journal_elem.text if journal_elem is not None else None
                    
                    paper = ArxivPaper(
                        arxiv_id=arxiv_id,
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        categories=categories,
                        published=published,
                        updated=updated,
                        doi=doi,
                        journal_ref=journal_ref,
                        primary_category=primary_cat,
                        pdf_url=pdf_url
                    )
                    
                    # 计算相关性
                    paper = self._score_relevance(paper)
                    
                    papers.append(paper)
                    
                except Exception as e:
                    print(f"Error parsing entry: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error parsing feed: {e}")
        
        return papers
    
    def _score_relevance(self, paper: ArxivPaper) -> ArxivPaper:
        """计算论文相关性分数"""
        score = 0.0
        matched_keywords = []
        
        text_to_check = f"{paper.title} {paper.abstract}".lower()
        
        # 检查各类关键词
        for category, keywords in self.keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_to_check:
                    # 标题匹配权重更高
                    if keyword.lower() in paper.title.lower():
                        score += 2.0
                    else:
                        score += 1.0
                    matched_keywords.append(keyword)
        
        # 作者追踪奖励
        for author in paper.authors:
            if any(tracked.lower() in author.lower() for tracked in self.authors_to_track):
                score += 3.0
        
        # 高影响力期刊引用
        if paper.journal_ref:
            high_impact = ['nature', 'science', 'phys. rev. lett.', 'jacs', 'angew']
            if any(journal in paper.journal_ref.lower() for journal in high_impact):
                score += 2.0
        
        paper.relevance_score = score
        paper.keywords_matched = list(set(matched_keywords))
        
        return paper
    
    def _deduplicate_and_filter(self, papers: List[ArxivPaper]) -> List[ArxivPaper]:
        """去重和筛选"""
        seen_ids = set()
        unique_papers = []
        
        for paper in papers:
            if paper.arxiv_id not in seen_ids:
                seen_ids.add(paper.arxiv_id)
                # 只保留相关性分数>0的论文
                if paper.relevance_score > 0:
                    unique_papers.append(paper)
        
        # 按相关性排序
        unique_papers.sort(key=lambda p: p.relevance_score, reverse=True)
        
        return unique_papers
    
    def get_high_relevance_papers(
        self,
        threshold: float = 3.0,
        top_k: int = 20
    ) -> List[ArxivPaper]:
        """获取高相关论文"""
        papers = [p for p in self.paper_history if p.relevance_score >= threshold]
        papers.sort(key=lambda p: p.relevance_score, reverse=True)
        return papers[:top_k]
    
    def get_papers_by_category(self, category: str) -> List[ArxivPaper]:
        """按类别获取论文"""
        return [p for p in self.paper_history if category in p.categories]
    
    def get_papers_by_keyword(self, keyword: str) -> List[ArxivPaper]:
        """按关键词获取论文"""
        return [p for p in self.paper_history if keyword in p.keywords_matched]
    
    def generate_report(
        self,
        days: int = 7,
        output_file: Optional[str] = None
    ) -> str:
        """生成监控报告"""
        recent_papers = [
            p for p in self.paper_history
            if (datetime.now() - p.published).days <= days
        ]
        
        report = f"""# arXiv Materials Science Monitor Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Period: Last {days} days

## Summary
- Total papers monitored: {len(recent_papers)}
- High relevance papers (score > 5): {len([p for p in recent_papers if p.relevance_score > 5])}
- Medium relevance papers (score 2-5): {len([p for p in recent_papers if 2 <= p.relevance_score <= 5])}

## Top Papers by Relevance
"""
        
        # 按相关性排序
        recent_papers.sort(key=lambda p: p.relevance_score, reverse=True)
        
        for i, paper in enumerate(recent_papers[:10], 1):
            report += f"""
### {i}. {paper.title}
- **arXiv ID**: {paper.arxiv_id}
- **Authors**: {', '.join(paper.authors[:3])}{' et al.' if len(paper.authors) > 3 else ''}
- **Relevance Score**: {paper.relevance_score:.1f}
- **Categories**: {', '.join(paper.categories[:3])}
- **Keywords**: {', '.join(paper.keywords_matched[:5])}
- **Published**: {paper.published.strftime('%Y-%m-%d')}
- **Abstract**: {paper.abstract[:300]}...

"""
        
        # 关键词统计
        all_keywords = defaultdict(int)
        for paper in recent_papers:
            for kw in paper.keywords_matched:
                all_keywords[kw] += 1
        
        report += "\n## Top Keywords\n"
        for kw, count in sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)[:20]:
            report += f"- {kw}: {count}\n"
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
        
        return report
    
    def export_to_json(self, filename: str):
        """导出论文数据到JSON"""
        data = []
        for paper in self.paper_history:
            data.append({
                'arxiv_id': paper.arxiv_id,
                'title': paper.title,
                'authors': paper.authors,
                'abstract': paper.abstract,
                'categories': paper.categories,
                'published': paper.published.isoformat(),
                'relevance_score': paper.relevance_score,
                'keywords_matched': paper.keywords_matched,
                'pdf_url': paper.pdf_url
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported {len(data)} papers to {filename}")


def demo():
    """演示"""
    print("=" * 60)
    print("arXiv Monitor Demo")
    print("=" * 60)
    
    # 创建监控器
    monitor = ArxivMonitor(
        search_categories=['cond-mat.mtrl-sci', 'cs.LG'],
        authors_to_track=['G. K. H. Madsen', 'A. Jain']
    )
    
    print("\nKeywords being tracked:")
    for category, keywords in monitor.keywords.items():
        print(f"  {category}: {len(keywords)} keywords")
    
    # 由于网络请求可能失败, 这里创建一些模拟数据用于演示
    print("\nCreating demo papers...")
    
    demo_papers = [
        ArxivPaper(
            arxiv_id="2401.00001",
            title="Deep Learning for Battery Material Discovery",
            authors=["John Doe", "Jane Smith"],
            abstract="We present a novel deep learning approach for discovering new battery cathode materials using graph neural networks and active learning.",
            categories=["cond-mat.mtrl-sci", "cs.LG"],
            published=datetime.now(),
            updated=datetime.now(),
            primary_category="cond-mat.mtrl-sci"
        ),
        ArxivPaper(
            arxiv_id="2401.00002",
            title="Transformer Models for Crystal Structure Prediction",
            authors=["Alice Wang", "Bob Chen"],
            abstract="This work introduces a transformer-based architecture for predicting stable crystal structures from composition alone.",
            categories=["cond-mat.mtrl-sci", "cs.CL"],
            published=datetime.now(),
            updated=datetime.now(),
            primary_category="cond-mat.mtrl-sci"
        ),
        ArxivPaper(
            arxiv_id="2401.00003",
            title="High-throughput DFT Screening of Perovskite Oxides",
            authors=["Carol Liu"],
            abstract="We perform high-throughput density functional theory calculations to screen perovskite oxides for catalytic applications.",
            categories=["cond-mat.mtrl-sci"],
            published=datetime.now(),
            updated=datetime.now(),
            primary_category="cond-mat.mtrl-sci"
        )
    ]
    
    # 评分
    for paper in demo_papers:
        monitor._score_relevance(paper)
        monitor.paper_history.append(paper)
    
    print(f"\nProcessed {len(demo_papers)} demo papers")
    
    # 输出高相关论文
    print("\nHigh relevance papers:")
    for paper in monitor.get_high_relevance_papers(threshold=0):
        print(f"\n  {paper.title}")
        print(f"    Score: {paper.relevance_score:.1f}")
        print(f"    Keywords: {paper.keywords_matched[:5]}")
    
    # 生成报告
    print("\nGenerating report...")
    report = monitor.generate_report(days=7)
    print(report[:1000])
    print("...")
    
    print("\n" + "=" * 60)
    print("arXiv Monitor Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
