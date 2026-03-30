"""
文献抓取器统一接口
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import hashlib

from ..config.models import Paper
from ..config.database import DatabaseManager
from .arxiv_fetcher import ArxivFetcher
from .pubmed_fetcher import PubMedFetcher
from .crossref_fetcher import CrossRefFetcher
from .semantic_scholar_fetcher import SemanticScholarFetcher


class LiteratureFetcher:
    """文献抓取器统一接口"""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()
        
        # 初始化各个抓取器
        self.fetchers = {
            "arxiv": ArxivFetcher(),
            "pubmed": PubMedFetcher(),
            "crossref": CrossRefFetcher(),
            "semanticscholar": SemanticScholarFetcher()
        }
    
    def search(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        max_results: int = 100,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        save_to_db: bool = True
    ) -> List[Paper]:
        """
        多源搜索文献
        
        Args:
            query: 搜索查询
            sources: 数据源列表，None表示全部
            max_results: 每个数据源的最大结果数
            date_from: 起始日期
            date_to: 结束日期
            save_to_db: 是否保存到数据库
        
        Returns:
            论文列表
        """
        sources = sources or list(self.fetchers.keys())
        all_papers = []
        
        for source in sources:
            if source not in self.fetchers:
                print(f"未知数据源: {source}")
                continue
            
            try:
                print(f"正在从 {source} 搜索...")
                fetcher = self.fetchers[source]
                
                # 根据不同源调用不同参数
                if source == "arxiv":
                    papers = fetcher.search(query, max_results=max_results)
                elif source == "pubmed":
                    papers = fetcher.search(query, max_results=max_results,
                                          date_from=date_from, date_to=date_to)
                elif source == "crossref":
                    papers = fetcher.search(query, max_results=max_results)
                else:  # semanticscholar
                    date_filter = None
                    if date_from:
                        date_filter = f">={date_from.year}"
                    papers = fetcher.search(query, max_results=max_results,
                                          publication_date_or_year=date_filter)
                
                print(f"  从 {source} 获取 {len(papers)} 篇论文")
                all_papers.extend(papers)
                
                # 保存到数据库
                if save_to_db:
                    for paper in papers:
                        self.db.save_paper(paper)
            
            except Exception as e:
                print(f"从 {source} 获取失败: {e}")
                continue
        
        # 去重
        unique_papers = self._deduplicate_papers(all_papers)
        
        print(f"总计获取 {len(unique_papers)} 篇唯一论文")
        return unique_papers
    
    def _deduplicate_papers(self, papers: List[Paper]) -> List[Paper]:
        """去重论文"""
        seen_ids = set()
        seen_dois = set()
        seen_arxiv = set()
        unique_papers = []
        
        for paper in papers:
            # 使用多种ID检查重复
            if paper.id in seen_ids:
                continue
            
            if paper.doi and paper.doi in seen_dois:
                continue
            
            if paper.arxiv_id and paper.arxiv_id in seen_arxiv:
                continue
            
            # 使用标题和作者模糊匹配
            if self._is_duplicate_by_content(paper, unique_papers):
                continue
            
            seen_ids.add(paper.id)
            if paper.doi:
                seen_dois.add(paper.doi)
            if paper.arxiv_id:
                seen_arxiv.add(paper.arxiv_id)
            
            unique_papers.append(paper)
        
        return unique_papers
    
    def _is_duplicate_by_content(self, paper: Paper, existing: List[Paper]) -> bool:
        """基于内容检查重复"""
        # 标题相似度检查（简化版）
        paper_title = paper.title.lower().strip()
        
        for existing_paper in existing:
            existing_title = existing_paper.title.lower().strip()
            
            # 完全匹配或高度相似
            if paper_title == existing_title:
                # 检查第一作者
                paper_first = paper.get_first_author()
                existing_first = existing_paper.get_first_author()
                
                if paper_first and existing_first:
                    if paper_first.split()[-1] == existing_first.split()[-1]:
                        return True
        
        return False
    
    def search_by_keywords(
        self,
        keywords: List[str],
        sources: Optional[List[str]] = None,
        max_results: int = 100,
        operator: str = "OR"
    ) -> List[Paper]:
        """
        按关键词搜索
        
        Args:
            keywords: 关键词列表
            sources: 数据源列表
            max_results: 最大结果数
            operator: 连接符 (AND/OR)
        
        Returns:
            论文列表
        """
        # 构建查询
        if operator.upper() == "AND":
            query = " AND ".join([f'"{kw}"' for kw in keywords])
        else:
            query = " OR ".join([f'"{kw}"' for kw in keywords])
        
        return self.search(query, sources, max_results)
    
    def fetch_recent(
        self,
        days: int = 7,
        sources: Optional[List[str]] = None,
        categories: Optional[List[str]] = None
    ) -> List[Paper]:
        """
        获取最近论文
        
        Args:
            days: 最近天数
            sources: 数据源列表
            categories: 分类列表（仅arXiv有效）
        
        Returns:
            论文列表
        """
        sources = sources or ["arxiv"]  # 默认使用arXiv
        all_papers = []
        
        for source in sources:
            if source == "arxiv":
                papers = self.fetchers["arxiv"].get_recent_papers(days, categories)
                all_papers.extend(papers)
                
                # 保存到数据库
                for paper in papers:
                    self.db.save_paper(paper)
        
        return self._deduplicate_papers(all_papers)
    
    def fetch_by_doi(self, doi: str) -> Optional[Paper]:
        """通过DOI获取论文"""
        # 首先检查数据库
        papers = self.db.search_papers(query=doi)
        for paper in papers:
            if paper.doi == doi:
                return paper
        
        # 从CrossRef获取
        paper = self.fetchers["crossref"].fetch_by_doi(doi)
        if paper:
            self.db.save_paper(paper)
            return paper
        
        return None
    
    def fetch_by_arxiv_id(self, arxiv_id: str) -> Optional[Paper]:
        """通过arXiv ID获取论文"""
        # 首先检查数据库
        paper = self.db.get_paper(f"arxiv:{arxiv_id}")
        if paper:
            return paper
        
        # 从arXiv获取
        paper = self.fetchers["arxiv"].fetch_by_id(arxiv_id)
        if paper:
            self.db.save_paper(paper)
            return paper
        
        return None
    
    def fetch_citations(self, paper: Paper) -> List[Paper]:
        """获取论文的引用"""
        citing_papers = []
        
        # 尝试从Semantic Scholar获取
        if paper.doi:
            paper_id = f"DOI:{paper.doi}"
            citations = self.fetchers["semanticscholar"].get_citations(paper_id)
            
            for cite in citations:
                citing_paper = cite.get("citingPaper", {})
                if citing_paper:
                    parsed = self.fetchers["semanticscholar"]._parse_paper(citing_paper)
                    if parsed:
                        citing_papers.append(parsed)
                        self.db.save_paper(parsed)
        
        return citing_papers
    
    def search_materials_dft_papers(
        self,
        max_results: int = 100
    ) -> List[Paper]:
        """搜索材料DFT相关论文"""
        # DFT和材料相关关键词
        dft_keywords = [
            "density functional theory",
            "DFT calculation",
            "first-principles",
            "electronic structure"
        ]
        
        materials_keywords = [
            "battery",
            "electrolyte",
            "lithium",
            "solid state",
            "ionic conductivity"
        ]
        
        # 构建查询
        dft_query = " OR ".join([f'"{kw}"' for kw in dft_keywords])
        materials_query = " OR ".join([f'"{kw}"' for kw in materials_keywords])
        
        query = f"({dft_query}) AND ({materials_query})"
        
        return self.search(query, max_results=max_results)
