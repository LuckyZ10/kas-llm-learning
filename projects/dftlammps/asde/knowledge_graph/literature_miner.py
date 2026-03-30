"""
Literature Miner Module

Provides automated retrieval and processing of scientific literature
from arXiv, PubMed, and other sources.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional, Protocol
from urllib.parse import quote_plus, urlencode
import hashlib

import aiohttp


@dataclass
class Paper:
    """Represents a scientific paper."""
    id: str
    title: str
    authors: list[str]
    abstract: str
    year: int
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pmid: Optional[str] = None
    journal: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    citations: int = 0
    keywords: list[str] = field(default_factory=list)
    raw_data: dict[str, Any] = field(default_factory=dict, repr=False)
    
    @property
    def unique_id(self) -> str:
        """Generate unique ID for deduplication."""
        if self.doi:
            return f"doi:{self.doi}"
        elif self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        elif self.pmid:
            return f"pmid:{self.pmid}"
        else:
            # Hash of title + first author
            key = f"{self.title}:{self.authors[0] if self.authors else ''}"
            return f"hash:{hashlib.md5(key.encode()).hexdigest()[:12]}"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "year": self.year,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "pmid": self.pmid,
            "journal": self.journal,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "citations": self.citations,
            "keywords": self.keywords
        }


class SearchProvider(Protocol):
    """Protocol for literature search providers."""
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any
    ) -> list[Paper]: ...
    
    async def get_paper_by_id(self, paper_id: str) -> Optional[Paper]: ...


class ArXivProvider:
    """
    arXiv API provider for physics, math, and CS papers.
    Uses arXiv API: http://export.arxiv.org/api/query
    """
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def __init__(self, delay: float = 3.0) -> None:
        self.delay = delay  # arXiv requires 3 second delay between requests
        self._last_request: Optional[datetime] = None
    
    async def _wait_rate_limit(self) -> None:
        """Respect arXiv rate limiting."""
        if self._last_request is not None:
            elapsed = (datetime.now() - self._last_request).total_seconds()
            if elapsed < self.delay:
                await asyncio.sleep(self.delay - elapsed)
        self._last_request = datetime.now()
    
    def _parse_arxiv_id(self, entry_id: str) -> str:
        """Extract arXiv ID from entry ID URL."""
        # Entry ID format: http://arxiv.org/abs/2101.12345
        match = re.search(r'arxiv\.org/abs/([\d.]+)', entry_id)
        if match:
            return match.group(1)
        return entry_id.split('/')[-1]
    
    def _parse_entry(self, entry: ET.Element, ns: dict[str, str]) -> Paper:
        """Parse an Atom entry into a Paper object."""
        # Extract basic fields
        title_elem = entry.find('atom:title', ns)
        title = title_elem.text if title_elem is not None else "Unknown"
        
        summary_elem = entry.find('atom:summary', ns)
        abstract = summary_elem.text if summary_elem is not None else ""
        
        # Extract authors
        authors = []
        for author in entry.findall('atom:author', ns):
            name_elem = author.find('atom:name', ns)
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text)
        
        # Extract dates
        published_elem = entry.find('atom:published', ns)
        year = 2024
        if published_elem is not None:
            try:
                date = datetime.fromisoformat(published_elem.text.replace('Z', '+00:00'))
                year = date.year
            except (ValueError, AttributeError):
                pass
        
        # Extract arXiv ID
        id_elem = entry.find('atom:id', ns)
        arxiv_id = self._parse_arxiv_id(id_elem.text) if id_elem is not None else None
        
        # Extract links
        url = None
        pdf_url = None
        for link in entry.findall('atom:link', ns):
            href = link.get('href', '')
            rel = link.get('rel', '')
            title_attr = link.get('title', '')
            
            if rel == 'alternate':
                url = href
            elif title_attr == 'pdf':
                pdf_url = href
        
        # Extract categories as keywords
        keywords = []
        for category in entry.findall('arxiv:primary_category', ns):
            term = category.get('term')
            if term:
                keywords.append(term)
        for category in entry.findall('atom:category', ns):
            term = category.get('term')
            if term and term not in keywords:
                keywords.append(term)
        
        # Extract DOI if available
        doi = None
        for link in entry.findall('arxiv:doi', ns):
            if link.text:
                doi = link.text
                break
        
        return Paper(
            id=f"arxiv:{arxiv_id}" if arxiv_id else str(hash(title)),
            title=title,
            authors=authors,
            abstract=abstract.strip(),
            year=year,
            doi=doi,
            arxiv_id=arxiv_id,
            url=url,
            pdf_url=pdf_url,
            keywords=keywords
        )
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        start: int = 0,
        **kwargs: Any
    ) -> list[Paper]:
        """Search arXiv for papers."""
        await self._wait_rate_limit()
        
        # Build query parameters
        params = {
            "search_query": f"all:{query}",
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order
        }
        
        url = f"{self.BASE_URL}?{urlencode(params)}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return []
                
                text = await response.text()
        
        # Parse XML
        root = ET.fromstring(text)
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        papers = []
        for entry in root.findall('atom:entry', ns):
            try:
                paper = self._parse_entry(entry, ns)
                papers.append(paper)
            except Exception:
                continue
        
        return papers
    
    async def get_paper_by_id(self, arxiv_id: str) -> Optional[Paper]:
        """Retrieve a paper by arXiv ID."""
        await self._wait_rate_limit()
        
        params = {
            "id_list": arxiv_id,
            "max_results": 1
        }
        
        url = f"{self.BASE_URL}?{urlencode(params)}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return None
                
                text = await response.text()
        
        root = ET.fromstring(text)
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        entry = root.find('atom:entry', ns)
        if entry is None:
            return None
        
        return self._parse_entry(entry, ns)
    
    async def search_by_category(
        self,
        category: str,
        max_results: int = 10
    ) -> list[Paper]:
        """Search by arXiv category."""
        query = f"cat:{category}"
        return await self.search(query, max_results=max_results)


class PubMedProvider:
    """
    PubMed/NCBI E-utilities API provider for biomedical literature.
    Uses NCBI E-utilities: https://www.ncbi.nlm.nih.gov/books/NBK25501/
    """
    
    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    def __init__(self, api_key: Optional[str] = None, delay: float = 0.34) -> None:
        self.api_key = api_key
        self.delay = delay  # NCBI recommends max 3 requests/second without key
        self._last_request: Optional[datetime] = None
    
    async def _wait_rate_limit(self) -> None:
        """Respect NCBI rate limiting."""
        if self._last_request is not None:
            elapsed = (datetime.now() - self._last_request).total_seconds()
            if elapsed < self.delay:
                await asyncio.sleep(self.delay - elapsed)
        self._last_request = datetime.now()
    
    def _build_params(self, **kwargs: Any) -> dict[str, Any]:
        """Build request parameters with optional API key."""
        params = {"db": "pubmed", **kwargs}
        if self.api_key:
            params["api_key"] = self.api_key
        return params
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any
    ) -> list[Paper]:
        """Search PubMed for papers."""
        # First, search for IDs
        await self._wait_rate_limit()
        
        search_params = self._build_params(
            term=query,
            retmax=max_results,
            retmode="json",
            sort="relevance"
        )
        
        url = f"{self.ESEARCH_URL}?{urlencode(search_params)}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
        
        idlist = data.get('esearchresult', {}).get('idlist', [])
        if not idlist:
            return []
        
        # Fetch details for IDs
        return await self._fetch_papers_by_id(idlist)
    
    async def _fetch_papers_by_id(self, pmids: list[str]) -> list[Paper]:
        """Fetch paper details by PubMed IDs."""
        await self._wait_rate_limit()
        
        fetch_params = self._build_params(
            id=",".join(pmids),
            retmode="xml"
        )
        
        url = f"{self.EFETCH_URL}?{urlencode(fetch_params)}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return []
                
                text = await response.text()
        
        return self._parse_pubmed_xml(text)
    
    def _parse_pubmed_xml(self, xml_text: str) -> list[Paper]:
        """Parse PubMed XML response."""
        papers = []
        
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return papers
        
        for article in root.findall('.//PubmedArticle'):
            try:
                paper = self._parse_pubmed_article(article)
                if paper:
                    papers.append(paper)
            except Exception:
                continue
        
        return papers
    
    def _parse_pubmed_article(self, article: ET.Element) -> Optional[Paper]:
        """Parse a single PubMed article."""
        # Get PMID
        pmid_elem = article.find('.//PMID')
        pmid = pmid_elem.text if pmid_elem is not None else None
        
        # Get title
        title_elem = article.find('.//ArticleTitle')
        title = title_elem.text if title_elem is not None else "Unknown"
        
        # Get abstract
        abstract_texts = article.findall('.//AbstractText')
        abstract = " ".join(t.text or "" for t in abstract_texts)
        
        # Get authors
        authors = []
        for author in article.findall('.//Author'):
            last = author.find('LastName')
            first = author.find('ForeName')
            if last is not None and first is not None:
                authors.append(f"{last.text}, {first.text}")
            elif last is not None:
                authors.append(last.text or "")
        
        # Get year
        year = 2024
        year_elem = article.find('.//PubDate/Year')
        if year_elem is not None and year_elem.text:
            try:
                year = int(year_elem.text)
            except ValueError:
                pass
        
        # Get DOI
        doi = None
        for id_elem in article.findall('.//ArticleId'):
            if id_elem.get('IdType') == 'doi':
                doi = id_elem.text
                break
        
        # Get journal
        journal = None
        journal_elem = article.find('.//Journal/Title')
        if journal_elem is not None:
            journal = journal_elem.text
        
        # Get keywords
        keywords = []
        for keyword in article.findall('.//Keyword'):
            if keyword.text:
                keywords.append(keyword.text)
        
        # Get mesh terms
        for mesh in article.findall('.//MeshHeading/DescriptorName'):
            if mesh.text and mesh.text not in keywords:
                keywords.append(mesh.text)
        
        return Paper(
            id=f"pmid:{pmid}" if pmid else str(hash(title)),
            title=title,
            authors=authors,
            abstract=abstract.strip(),
            year=year,
            doi=doi,
            pmid=pmid,
            journal=journal,
            keywords=keywords
        )
    
    async def get_paper_by_id(self, pmid: str) -> Optional[Paper]:
        """Retrieve a paper by PubMed ID."""
        papers = await self._fetch_papers_by_id([pmid])
        return papers[0] if papers else None


class CrossRefProvider:
    """
    CrossRef API provider for general scientific literature.
    Uses CrossRef API: https://api.crossref.org
    """
    
    BASE_URL = "https://api.crossref.org/works"
    
    def __init__(self, mailto: Optional[str] = None) -> None:
        self.mailto = mailto
        self._last_request = datetime.now()
        self._min_delay = 1.0  # Be polite
    
    async def _wait_rate_limit(self) -> None:
        """Respect CrossRef rate limiting."""
        elapsed = (datetime.now() - self._last_request).total_seconds()
        if elapsed < self._min_delay:
            await asyncio.sleep(self._min_delay - elapsed)
        self._last_request = datetime.now()
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any
    ) -> list[Paper]:
        """Search CrossRef for papers."""
        await self._wait_rate_limit()
        
        params: dict[str, Any] = {
            "query": query,
            "rows": min(max_results, 20),  # API limit
            "sort": "relevance",
            "order": "desc"
        }
        
        if self.mailto:
            params["mailto"] = self.mailto
        
        url = f"{self.BASE_URL}?{urlencode(params)}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
        
        items = data.get('message', {}).get('items', [])
        papers = []
        
        for item in items:
            try:
                paper = self._parse_work(item)
                papers.append(paper)
            except Exception:
                continue
        
        return papers
    
    def _parse_work(self, item: dict[str, Any]) -> Paper:
        """Parse a CrossRef work item."""
        title = "Unknown"
        if item.get('title'):
            title = item['title'][0]
        
        authors = []
        for author in item.get('author', []):
            given = author.get('given', '')
            family = author.get('family', '')
            if given and family:
                authors.append(f"{family}, {given}")
            elif family:
                authors.append(family)
        
        year = 2024
        published = item.get('published-print') or item.get('published-online')
        if published and published.get('date-parts'):
            try:
                year = published['date-parts'][0][0]
            except (IndexError, TypeError):
                pass
        
        doi = item.get('DOI')
        
        # Build URL
        url = item.get('URL')
        if doi and not url:
            url = f"https://doi.org/{doi}"
        
        # Extract abstract if available
        abstract = item.get('abstract', '')
        # Remove XML tags from abstract
        abstract = re.sub(r'<[^>]+?>', '', abstract)
        
        # Get keywords
        keywords = item.get('subject', []) if isinstance(item.get('subject'), list) else []
        
        # Get citation count
        citations = item.get('is-referenced-by-count', 0)
        
        return Paper(
            id=f"doi:{doi}" if doi else str(hash(title)),
            title=title,
            authors=authors,
            abstract=abstract.strip(),
            year=year,
            doi=doi,
            journal=item.get('container-title', [None])[0],
            url=url,
            citations=citations,
            keywords=keywords,
            raw_data=item
        )
    
    async def get_paper_by_id(self, doi: str) -> Optional[Paper]:
        """Retrieve a paper by DOI."""
        await self._wait_rate_limit()
        
        params: dict[str, str] = {}
        if self.mailto:
            params["mailto"] = self.mailto
        
        url = f"https://api.crossref.org/works/{quote_plus(doi)}"
        if params:
            url += f"?{urlencode(params)}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
        
        item = data.get('message', {})
        if not item:
            return None
        
        return self._parse_work(item)


class LiteratureMiner:
    """
    Unified interface for mining scientific literature from multiple sources.
    """
    
    def __init__(
        self,
        arxiv: bool = True,
        pubmed: bool = True,
        crossref: bool = True,
        pubmed_api_key: Optional[str] = None,
        crossref_mailto: Optional[str] = None
    ) -> None:
        self.providers: dict[str, SearchProvider] = {}
        
        if arxiv:
            self.providers['arxiv'] = ArXivProvider()
        if pubmed:
            self.providers['pubmed'] = PubMedProvider(api_key=pubmed_api_key)
        if crossref:
            self.providers['crossref'] = CrossRefProvider(mailto=crossref_mailto)
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        sources: Optional[list[str]] = None,
        **kwargs: Any
    ) -> list[Paper]:
        """
        Search across multiple providers.
        
        Args:
            query: Search query string
            max_results: Maximum results per source
            sources: List of source names to use (None = all)
            **kwargs: Additional provider-specific parameters
        """
        providers_to_use = sources or list(self.providers.keys())
        
        tasks = []
        for name in providers_to_use:
            if name in self.providers:
                tasks.append(self.providers[name].search(query, max_results, **kwargs))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Deduplicate by unique_id
        seen: set[str] = set()
        papers: list[Paper] = []
        
        for result in results:
            if isinstance(result, Exception):
                continue
            for paper in result:
                uid = paper.unique_id
                if uid not in seen:
                    seen.add(uid)
                    papers.append(paper)
        
        # Sort by relevance (citations as proxy)
        papers.sort(key=lambda p: p.citations, reverse=True)
        
        return papers
    
    async def search_by_topic(
        self,
        topic: str,
        keywords: Optional[list[str]] = None,
        max_results: int = 10
    ) -> list[Paper]:
        """Search by topic with optional keyword filtering."""
        # Build query
        query = topic
        if keywords:
            query += " AND (" + " OR ".join(keywords) + ")"
        
        return await self.search(query, max_results=max_results)
    
    async def search_materials_papers(
        self,
        material: str,
        property_name: Optional[str] = None,
        max_results: int = 10
    ) -> list[Paper]:
        """Search for papers about specific materials."""
        query = material
        if property_name:
            query += f" {property_name}"
        
        # Try arXiv first for physics/materials papers
        papers = await self.search(query, max_results=max_results, sources=['arxiv'])
        
        # If not enough results, try CrossRef
        if len(papers) < max_results // 2:
            crossref_papers = await self.search(
                query,
                max_results=max_results,
                sources=['crossref']
            )
            papers.extend(crossref_papers)
        
        return papers[:max_results]
    
    async def get_recent_papers(
        self,
        query: str,
        days: int = 30,
        max_results: int = 10
    ) -> list[Paper]:
        """Get recent papers from the last N days."""
        papers = await self.search(query, max_results=max_results * 2)
        
        # Filter by date
        cutoff = datetime.now() - timedelta(days=days)
        recent = [p for p in papers if p.year >= cutoff.year]
        
        return recent[:max_results]
    
    async def get_paper_by_id(
        self,
        paper_id: str,
        source: Optional[str] = None
    ) -> Optional[Paper]:
        """Retrieve a paper by ID from specific or any source."""
        if source and source in self.providers:
            return await self.providers[source].get_paper_by_id(paper_id)
        
        # Try all sources
        for name, provider in self.providers.items():
            try:
                paper = await provider.get_paper_by_id(paper_id)
                if paper:
                    return paper
            except Exception:
                continue
        
        return None
    
    def save_papers(self, papers: list[Paper], filepath: str) -> None:
        """Save papers to JSON file."""
        data = {
            "saved_at": datetime.now().isoformat(),
            "count": len(papers),
            "papers": [p.to_dict() for p in papers]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_papers(self, filepath: str) -> list[Paper]:
        """Load papers from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        papers = []
        for p_data in data.get('papers', []):
            paper = Paper(
                id=p_data['id'],
                title=p_data['title'],
                authors=p_data['authors'],
                abstract=p_data['abstract'],
                year=p_data['year'],
                doi=p_data.get('doi'),
                arxiv_id=p_data.get('arxiv_id'),
                pmid=p_data.get('pmid'),
                journal=p_data.get('journal'),
                url=p_data.get('url'),
                pdf_url=p_data.get('pdf_url'),
                citations=p_data.get('citations', 0),
                keywords=p_data.get('keywords', [])
            )
            papers.append(paper)
        
        return papers


async def main():
    """Demo functionality."""
    miner = LiteratureMiner()
    
    # Search for materials papers
    print("Searching for graphene thermal conductivity papers...")
    papers = await miner.search_materials_papers(
        "graphene",
        "thermal conductivity",
        max_results=5
    )
    
    print(f"\nFound {len(papers)} papers:\n")
    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper.title}")
        print(f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
        print(f"   Year: {paper.year}")
        print(f"   DOI: {paper.doi or 'N/A'}")
        print(f"   Citations: {paper.citations}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
