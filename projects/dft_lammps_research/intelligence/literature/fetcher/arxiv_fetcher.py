"""
arXiv文献抓取器
"""

import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Optional, Dict, Any
import urllib.request
import urllib.parse

from ..config.models import Paper, Author
from ..config.settings import DATA_SOURCES


class ArxivFetcher:
    """arXiv文献抓取器"""
    
    def __init__(self):
        self.config = DATA_SOURCES["arxiv"]
        self.base_url = self.config["base_url"]
        self.max_results = self.config["max_results"]
        self.rate_limit = self.config["rate_limit"]
        self.last_request_time = 0
    
    def _rate_limit_wait(self):
        """限速等待"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def search(
        self,
        query: str,
        start: int = 0,
        max_results: Optional[int] = None,
        sort_by: str = "submittedDate",
        sort_order: str = "descending"
    ) -> List[Paper]:
        """
        搜索arXiv论文
        
        Args:
            query: 搜索查询
            start: 起始位置
            max_results: 最大结果数
            sort_by: 排序方式 (relevance, lastUpdatedDate, submittedDate)
            sort_order: 排序顺序 (ascending, descending)
        
        Returns:
            论文列表
        """
        max_results = max_results or self.max_results
        
        # 构建查询参数
        params = {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order
        }
        
        url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
        
        try:
            self._rate_limit_wait()
            
            headers = {
                "User-Agent": "LiteratureSurveyBot/1.0 (research@example.com)"
            }
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read()
            
            return self._parse_atom_feed(data)
        
        except Exception as e:
            print(f"arXiv搜索失败: {e}")
            return []
    
    def _parse_atom_feed(self, xml_data: bytes) -> List[Paper]:
        """解析Atom feed"""
        papers = []
        
        try:
            root = ET.fromstring(xml_data)
            
            # 定义命名空间
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom"
            }
            
            for entry in root.findall("atom:entry", ns):
                try:
                    paper = self._parse_entry(entry, ns)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    print(f"解析条目失败: {e}")
                    continue
        
        except ET.ParseError as e:
            print(f"XML解析错误: {e}")
        
        return papers
    
    def _parse_entry(self, entry: ET.Element, ns: Dict[str, str]) -> Optional[Paper]:
        """解析单个条目"""
        # 获取ID
        id_elem = entry.find("atom:id", ns)
        if id_elem is None:
            return None
        
        arxiv_id = id_elem.text.split("/")[-1].split("v")[0]
        paper_id = f"arxiv:{arxiv_id}"
        
        # 获取标题
        title_elem = entry.find("atom:title", ns)
        title = title_elem.text.strip() if title_elem else ""
        title = " ".join(title.split())  # 清理空白字符
        
        # 获取摘要
        summary_elem = entry.find("atom:summary", ns)
        abstract = summary_elem.text.strip() if summary_elem else ""
        
        # 获取作者
        authors = []
        for author_elem in entry.findall("atom:author", ns):
            name_elem = author_elem.find("atom:name", ns)
            if name_elem is not None:
                authors.append(Author(name=name_elem.text))
        
        # 获取发布日期
        published_elem = entry.find("atom:published", ns)
        if published_elem:
            pub_date = datetime.fromisoformat(published_elem.text.replace("Z", "+00:00"))
        else:
            pub_date = datetime.now()
        
        # 获取分类
        categories = []
        for cat_elem in entry.findall("atom:category", ns):
            term = cat_elem.get("term")
            if term:
                categories.append(term)
        
        # 获取PDF链接
        pdf_url = None
        for link in entry.findall("atom:link", ns):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
                break
        
        # 获取主页面URL
        url = None
        for link in entry.findall("atom:link", ns):
            if link.get("rel") == "alternate":
                url = link.get("href")
                break
        
        # 获取评论（可能包含期刊信息）
        comment_elem = entry.find("arxiv:comment", ns)
        journal = None
        if comment_elem is not None and comment_elem.text:
            # 尝试提取期刊信息
            comment = comment_elem.text
            if "published" in comment.lower() or "accepted" in comment.lower():
                journal = comment
        
        # 获取DOI
        doi_elem = entry.find("arxiv:doi", ns)
        doi = doi_elem.text if doi_elem is not None else None
        
        return Paper(
            id=paper_id,
            title=title,
            authors=authors,
            abstract=abstract,
            publication_date=pub_date,
            journal=journal,
            doi=doi,
            arxiv_id=arxiv_id,
            url=url,
            pdf_url=pdf_url,
            categories=categories,
            source="arxiv",
            fetched_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    def fetch_by_id(self, arxiv_id: str) -> Optional[Paper]:
        """通过arXiv ID获取论文"""
        query = f"id:{arxiv_id}"
        papers = self.search(query, max_results=1)
        return papers[0] if papers else None
    
    def fetch_by_category(
        self,
        category: str,
        date_from: Optional[datetime] = None,
        max_results: int = 100
    ) -> List[Paper]:
        """按分类获取论文"""
        query = f"cat:{category}"
        
        if date_from:
            date_str = date_from.strftime("%Y%m%d")
            query += f" AND submittedDate:[{date_str}0000 TO NOW]"
        
        return self.search(query, max_results=max_results)
    
    def get_recent_papers(
        self,
        days: int = 7,
        categories: Optional[List[str]] = None
    ) -> List[Paper]:
        """获取最近论文"""
        from datetime import timedelta
        
        date_from = datetime.now() - timedelta(days=days)
        
        all_papers = []
        cats = categories or self.config["categories"]
        
        for category in cats:
            papers = self.fetch_by_category(category, date_from)
            all_papers.extend(papers)
        
        # 去重
        seen_ids = set()
        unique_papers = []
        for paper in all_papers:
            if paper.id not in seen_ids:
                seen_ids.add(paper.id)
                unique_papers.append(paper)
        
        return sorted(unique_papers, key=lambda x: x.publication_date, reverse=True)
