"""
PubMed文献抓取器
"""

import time
import json
import urllib.request
import urllib.parse
from datetime import datetime
from typing import List, Optional, Dict, Any

from ..config.models import Paper, Author
from ..config.settings import DATA_SOURCES


class PubMedFetcher:
    """PubMed文献抓取器"""
    
    def __init__(self):
        self.config = DATA_SOURCES["pubmed"]
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
        """发起请求并返回JSON"""
        try:
            self._rate_limit_wait()
            
            headers = {
                "User-Agent": "LiteratureSurveyBot/1.0 (research@example.com)"
            }
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
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> List[Paper]:
        """
        搜索PubMed论文
        
        Args:
            query: 搜索查询
            max_results: 最大结果数
            date_from: 起始日期
            date_to: 结束日期
        
        Returns:
            论文列表
        """
        # 构建搜索查询
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": min(max_results, 10000)
        }
        
        if date_from:
            search_params["mindate"] = date_from.strftime("%Y/%m/%d")
        if date_to:
            search_params["maxdate"] = date_to.strftime("%Y/%m/%d")
        
        if self.api_key:
            search_params["api_key"] = self.api_key
        
        search_url = f"{self.base_url}/esearch.fcgi?{urllib.parse.urlencode(search_params)}"
        
        # 执行搜索
        search_result = self._make_request(search_url)
        if not search_result or "esearchresult" not in search_result:
            return []
        
        id_list = search_result["esearchresult"].get("idlist", [])
        if not id_list:
            return []
        
        # 获取详细信息
        return self.fetch_details(id_list)
    
    def fetch_details(self, pmids: List[str]) -> List[Paper]:
        """获取论文详细信息"""
        if not pmids:
            return []
        
        # 分批获取（每次最多200个）
        batch_size = 200
        all_papers = []
        
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i+batch_size]
            papers = self._fetch_batch(batch)
            all_papers.extend(papers)
        
        return all_papers
    
    def _fetch_batch(self, pmids: List[str]) -> List[Paper]:
        """获取一批论文"""
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml"
        }
        
        if self.api_key:
            fetch_params["api_key"] = self.api_key
        
        fetch_url = f"{self.base_url}/efetch.fcgi?{urllib.parse.urlencode(fetch_params)}"
        
        try:
            self._rate_limit_wait()
            
            headers = {
                "User-Agent": "LiteratureSurveyBot/1.0"
            }
            req = urllib.request.Request(fetch_url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=60) as response:
                xml_data = response.read()
            
            return self._parse_pubmed_xml(xml_data)
        
        except Exception as e:
            print(f"获取详情失败: {e}")
            return []
    
    def _parse_pubmed_xml(self, xml_data: bytes) -> List[Paper]:
        """解析PubMed XML"""
        import xml.etree.ElementTree as ET
        
        papers = []
        
        try:
            root = ET.fromstring(xml_data)
            
            for article in root.findall(".//PubmedArticle"):
                try:
                    paper = self._parse_article(article)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    print(f"解析文章失败: {e}")
                    continue
        
        except ET.ParseError as e:
            print(f"XML解析错误: {e}")
        
        return papers
    
    def _parse_article(self, article: Any) -> Optional[Paper]:
        """解析单个文章"""
        # 获取PMID
        pmid_elem = article.find(".//PMID")
        if pmid_elem is None:
            return None
        
        pmid = pmid_elem.text
        paper_id = f"pubmed:{pmid}"
        
        # 获取标题
        title_elem = article.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None else ""
        
        # 获取摘要
        abstract_elem = article.find(".//Abstract/AbstractText")
        abstract = abstract_elem.text if abstract_elem is not None else ""
        
        # 如果没有摘要，尝试其他路径
        if not abstract:
            abstract_parts = article.findall(".//Abstract/AbstractText")
            abstract = " ".join([p.text for p in abstract_parts if p.text])
        
        # 获取作者
        authors = []
        for author_elem in article.findall(".//Author"):
            lastname = author_elem.find("LastName")
            forename = author_elem.find("ForeName")
            
            if lastname is not None:
                name = lastname.text
                if forename is not None:
                    name = f"{forename.text} {name}"
                
                affiliation = author_elem.find("AffiliationInfo/Affiliation")
                affil_text = affiliation.text if affiliation is not None else None
                
                authors.append(Author(name=name, affiliation=affil_text))
        
        # 获取期刊
        journal_elem = article.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None else None
        
        # 获取发布日期
        pub_date = self._parse_date(article)
        
        # 获取DOI
        doi_elem = article.find(".//ArticleId[@IdType='doi']")
        doi = doi_elem.text if doi_elem is not None else None
        
        # 获取关键词
        keywords = []
        for mesh in article.findall(".//MeshHeading/DescriptorName"):
            if mesh.text:
                keywords.append(mesh.text)
        
        # 构建URL
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        
        return Paper(
            id=paper_id,
            title=title,
            authors=authors,
            abstract=abstract,
            publication_date=pub_date,
            journal=journal,
            doi=doi,
            pmid=pmid,
            url=url,
            keywords=keywords,
            source="pubmed",
            fetched_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    def _parse_date(self, article: Any) -> datetime:
        """解析日期"""
        try:
            year_elem = article.find(".//PubDate/Year")
            month_elem = article.find(".//PubDate/Month")
            day_elem = article.find(".//PubDate/Day")
            
            year = int(year_elem.text) if year_elem is not None else 2000
            
            month_map = {
                "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
                "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
                "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
            }
            
            if month_elem is not None:
                month_text = month_elem.text
                month = month_map.get(month_text, int(month_text) if month_text.isdigit() else 1)
            else:
                month = 1
            
            day = int(day_elem.text) if day_elem is not None else 1
            
            return datetime(year, month, day)
        
        except Exception as e:
            print(f"日期解析失败: {e}")
            return datetime.now()
    
    def search_materials_chemistry(self, max_results: int = 100) -> List[Paper]:
        """搜索材料化学相关论文"""
        # 构建材料化学相关查询
        query_parts = [
            "density functional theory",
            "molecular dynamics",
            "battery materials",
            "solid electrolyte",
            "lithium ion",
            "computational chemistry"
        ]
        
        query = " OR ".join([f'"{q}"[Title/Abstract]' for q in query_parts])
        
        # 限制在材料化学相关期刊
        journal_filter = (
            '("Journal of Materials Chemistry"[Journal] OR '
            '"Chemistry of Materials"[Journal] OR '
            '"Physical Review Materials"[Journal] OR '
            '"npj Computational Materials"[Journal])'
        )
        
        full_query = f"({query}) AND {journal_filter}"
        
        return self.search(full_query, max_results=max_results)
