"""
Semantic Scholar文献抓取器
"""

import time
import json
import urllib.request
import urllib.parse
from datetime import datetime
from typing import List, Optional, Dict, Any

from ..config.models import Paper, Author
from ..config.settings import DATA_SOURCES


class SemanticScholarFetcher:
    """Semantic Scholar文献抓取器"""
    
    def __init__(self):
        self.config = DATA_SOURCES["semantic_scholar"]
        self.base_url = self.config["base_url"]
        self.api_key = self.config["api_key"]
        self.rate_limit = self.config["rate_limit"]
        self.last_request_time = 0
    
    def _rate_limit_wait(self):
        """限速等待"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str) -> Optional[Dict]:
        """发起请求"""
        try:
            self._rate_limit_wait()
            
            headers = {
                "User-Agent": "LiteratureSurveyBot/1.0"
            }
            
            if self.api_key:
                headers["x-api-key"] = self.api_key
            
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read()
            
            return json.loads(data)
        
        except Exception as e:
            print(f"请求失败: {e}")
            return None
    
    def search(
        self,
        query: str,
        max_results: int = 100,
        fields: Optional[List[str]] = None,
        publication_date_or_year: Optional[str] = None
    ) -> List[Paper]:
        """
        搜索Semantic Scholar论文
        
        Args:
            query: 搜索查询
            max_results: 最大结果数
            fields: 返回字段
            publication_date_or_year: 发布日期/年份过滤
        
        Returns:
            论文列表
        """
        if fields is None:
            fields = [
                "paperId", "externalIds", "url", "title", "abstract",
                "venue", "year", "referenceCount", "citationCount",
                "publicationDate", "authors", "fieldsOfStudy", "openAccessPdf"
            ]
        
        papers = []
        offset = 0
        limit = min(100, max_results)
        
        while len(papers) < max_results:
            batch = self._search_batch(query, limit, offset, fields, publication_date_or_year)
            if not batch:
                break
            
            papers.extend(batch)
            offset += limit
            
            if len(batch) < limit:
                break
        
        return papers[:max_results]
    
    def _search_batch(
        self,
        query: str,
        limit: int,
        offset: int,
        fields: List[str],
        publication_date_or_year: Optional[str]
    ) -> List[Paper]:
        """搜索一批论文"""
        params = {
            "query": query,
            "limit": limit,
            "offset": offset,
            "fields": ",".join(fields)
        }
        
        if publication_date_or_year:
            params["publicationDateOrYear"] = publication_date_or_year
        
        url = f"{self.base_url}/paper/search?{urllib.parse.urlencode(params)}"
        
        data = self._make_request(url)
        if not data:
            return []
        
        items = data.get("data", [])
        return [self._parse_paper(item) for item in items if self._parse_paper(item)]
    
    def _parse_paper(self, item: Dict) -> Optional[Paper]:
        """解析论文"""
        paper_id = item.get("paperId")
        if not paper_id:
            return None
        
        # 获取外部ID
        external_ids = item.get("externalIds", {})
        doi = external_ids.get("DOI")
        arxiv_id = external_ids.get("ArXiv")
        pmid = external_ids.get("PubMed")
        
        # 构建ID
        if doi:
            paper_id = f"semanticscholar:doi:{doi}"
        else:
            paper_id = f"semanticscholar:{paper_id}"
        
        # 获取标题和摘要
        title = item.get("title", "")
        abstract = item.get("abstract", "") or ""
        
        # 获取作者
        authors = []
        for author_data in item.get("authors", []):
            name = author_data.get("name", "")
            if name:
                author_id = author_data.get("authorId")
                authors.append(Author(name=name, orcid=author_id))
        
        # 获取日期
        pub_date_str = item.get("publicationDate")
        if pub_date_str:
            try:
                pub_date = datetime.strptime(pub_date_str, "%Y-%m-%d")
            except:
                year = item.get("year", 2000)
                pub_date = datetime(year, 1, 1)
        else:
            year = item.get("year", 2000)
            pub_date = datetime(year, 1, 1)
        
        # 获取期刊
        journal = item.get("venue")
        
        # 获取引用信息
        citation_count = item.get("citationCount", 0)
        reference_count = item.get("referenceCount", 0)
        
        # 获取研究领域
        fields_of_study = item.get("fieldsOfStudy", [])
        
        # 获取PDF链接
        open_access = item.get("openAccessPdf")
        pdf_url = open_access.get("url") if open_access else None
        
        # 构建URL
        url = item.get("url") or f"https://www.semanticscholar.org/paper/{item.get('paperId')}"
        
        return Paper(
            id=paper_id,
            title=title,
            authors=authors,
            abstract=abstract,
            publication_date=pub_date,
            journal=journal,
            doi=doi,
            arxiv_id=arxiv_id,
            pmid=pmid,
            url=url,
            pdf_url=pdf_url,
            keywords=fields_of_study,
            citation_count=citation_count,
            reference_count=reference_count,
            source="semanticscholar",
            fetched_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    def fetch_paper_details(self, paper_id: str) -> Optional[Paper]:
        """获取论文详细信息"""
        fields = [
            "paperId", "externalIds", "url", "title", "abstract",
            "venue", "year", "referenceCount", "citationCount",
            "publicationDate", "authors", "fieldsOfStudy", "openAccessPdf",
            "references", "citations"
        ]
        
        url = f"{self.base_url}/paper/{paper_id}?fields={','.join(fields)}"
        
        data = self._make_request(url)
        if data:
            paper = self._parse_paper(data)
            
            # 解析参考文献
            if paper:
                refs = data.get("references", [])
                paper.references = [
                    r.get("paperId") for r in refs if r.get("paperId")
                ][:100]  # 限制数量
            
            return paper
        
        return None
    
    def get_citations(self, paper_id: str, limit: int = 100) -> List[Dict]:
        """获取引用该论文的文献"""
        url = f"{self.base_url}/paper/{paper_id}/citations?limit={limit}"
        
        data = self._make_request(url)
        if data:
            return data.get("data", [])
        
        return []
    
    def get_references(self, paper_id: str, limit: int = 100) -> List[Dict]:
        """获取该论文的参考文献"""
        url = f"{self.base_url}/paper/{paper_id}/references?limit={limit}"
        
        data = self._make_request(url)
        if data:
            return data.get("data", [])
        
        return []
    
    def search_by_author(self, author_name: str, max_results: int = 100) -> List[Paper]:
        """按作者搜索"""
        # 首先搜索作者
        url = f"{self.base_url}/author/search?query={urllib.parse.quote(author_name)}&limit=5"
        
        data = self._make_request(url)
        if not data or not data.get("data"):
            return []
        
        # 取第一个匹配的作者
        author = data["data"][0]
        author_id = author.get("authorId")
        
        if not author_id:
            return []
        
        # 获取作者的论文
        papers = []
        offset = 0
        limit = 100
        
        while len(papers) < max_results:
            url = f"{self.base_url}/author/{author_id}/papers?limit={limit}&offset={offset}"
            data = self._make_request(url)
            
            if not data or not data.get("data"):
                break
            
            for item in data["data"]:
                paper = self._parse_paper(item)
                if paper:
                    papers.append(paper)
            
            offset += limit
            
            if len(data["data"]) < limit:
                break
        
        return papers[:max_results]
