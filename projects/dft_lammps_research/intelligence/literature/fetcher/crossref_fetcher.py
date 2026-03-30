"""
CrossRef文献抓取器
"""

import time
import json
import urllib.request
import urllib.parse
from datetime import datetime
from typing import List, Optional, Dict, Any

from ..config.models import Paper, Author
from ..config.settings import DATA_SOURCES


class CrossRefFetcher:
    """CrossRef文献抓取器"""
    
    def __init__(self):
        self.config = DATA_SOURCES["crossref"]
        self.base_url = self.config["base_url"]
        self.rate_limit = self.config["rate_limit"]
        self.last_request_time = 0
        self.mailto = "research@example.com"  # 礼貌池
    
    def _rate_limit_wait(self):
        """限速等待"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def search(
        self,
        query: str,
        max_results: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Paper]:
        """
        搜索CrossRef论文
        
        Args:
            query: 搜索查询
            max_results: 最大结果数
            filters: 过滤条件
        
        Returns:
            论文列表
        """
        papers = []
        rows = min(max_results, 1000)
        offset = 0
        
        while len(papers) < max_results:
            batch = self._search_batch(query, rows, offset, filters)
            if not batch:
                break
            
            papers.extend(batch)
            offset += rows
            
            if len(batch) < rows:
                break
        
        return papers[:max_results]
    
    def _search_batch(
        self,
        query: str,
        rows: int,
        offset: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[Paper]:
        """搜索一批论文"""
        params = {
            "query": query,
            "rows": rows,
            "offset": offset,
            "mailto": self.mailto,
            "select": "DOI,title,author,abstract,published-print,created,type,subject,link"
        }
        
        if filters:
            filter_parts = []
            for key, value in filters.items():
                filter_parts.append(f"{key}:{value}")
            params["filter"] = ",".join(filter_parts)
        
        url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
        
        try:
            self._rate_limit_wait()
            
            headers = {
                "User-Agent": "LiteratureSurveyBot/1.0 (mailto:research@example.com)"
            }
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read())
            
            items = data.get("message", {}).get("items", [])
            return [self._parse_item(item) for item in items if self._parse_item(item)]
        
        except Exception as e:
            print(f"CrossRef搜索失败: {e}")
            return []
    
    def _parse_item(self, item: Dict) -> Optional[Paper]:
        """解析单个条目"""
        doi = item.get("DOI")
        if not doi:
            return None
        
        paper_id = f"crossref:{doi}"
        
        # 获取标题
        title = ""
        titles = item.get("title", [])
        if titles:
            title = titles[0]
        
        # 获取作者
        authors = []
        for author_data in item.get("author", []):
            given = author_data.get("given", "")
            family = author_data.get("family", "")
            name = f"{given} {family}".strip()
            if name:
                affiliation = None
                affils = author_data.get("affiliation", [])
                if affils:
                    affiliation = affils[0].get("name")
                
                orcid = author_data.get("ORCID")
                authors.append(Author(name=name, affiliation=affiliation, orcid=orcid))
        
        # 获取摘要
        abstract = item.get("abstract", "")
        # CrossRef的摘要通常是JATS XML格式，需要清理
        if abstract:
            abstract = self._clean_abstract(abstract)
        
        # 获取日期
        pub_date = self._parse_date(item)
        
        # 获取期刊
        container = item.get("container-title", [])
        journal = container[0] if container else None
        
        # 获取主题
        subjects = item.get("subject", [])
        
        # 获取引用数
        cited_count = item.get("is-referenced-by-count", 0)
        
        # 构建URL
        url = f"https://doi.org/{doi}"
        
        # 查找PDF链接
        pdf_url = None
        for link in item.get("link", []):
            content_type = link.get("content-type", "")
            if "pdf" in content_type.lower():
                pdf_url = link.get("URL")
                break
        
        return Paper(
            id=paper_id,
            title=title,
            authors=authors,
            abstract=abstract,
            publication_date=pub_date,
            journal=journal,
            doi=doi,
            url=url,
            pdf_url=pdf_url,
            keywords=subjects,
            citation_count=cited_count,
            source="crossref",
            fetched_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    def _clean_abstract(self, abstract: str) -> str:
        """清理摘要文本"""
        import re
        
        # 移除JATS XML标签
        abstract = re.sub(r'<[^>]+>', ' ', abstract)
        
        # 清理多余空白
        abstract = " ".join(abstract.split())
        
        return abstract
    
    def _parse_date(self, item: Dict) -> datetime:
        """解析日期"""
        try:
            # 首先尝试published-print
            published = item.get("published-print", {}).get("date-parts", [])
            if not published:
                # 尝试created
                published = item.get("created", {}).get("date-parts", [])
            
            if published and published[0]:
                parts = published[0]
                year = parts[0] if len(parts) > 0 else 2000
                month = parts[1] if len(parts) > 1 else 1
                day = parts[2] if len(parts) > 2 else 1
                
                return datetime(year, month, day)
        
        except Exception as e:
            print(f"日期解析失败: {e}")
        
        return datetime.now()
    
    def fetch_by_doi(self, doi: str) -> Optional[Paper]:
        """通过DOI获取论文"""
        params = {
            "filter": f"doi:{doi}",
            "mailto": self.mailto
        }
        
        url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
        
        try:
            self._rate_limit_wait()
            
            headers = {
                "User-Agent": "LiteratureSurveyBot/1.0"
            }
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read())
            
            items = data.get("message", {}).get("items", [])
            if items:
                return self._parse_item(items[0])
        
        except Exception as e:
            print(f"获取DOI失败: {e}")
        
        return None
    
    def search_materials_science(self, max_results: int = 100) -> List[Paper]:
        """搜索材料科学论文"""
        filters = {
            "type": "journal-article"
        }
        
        query_parts = [
            "density functional theory",
            "molecular dynamics",
            "computational materials",
            "battery",
            "electrolyte"
        ]
        
        query = " OR ".join(query_parts)
        
        return self.search(query, max_results=max_results, filters=filters)
