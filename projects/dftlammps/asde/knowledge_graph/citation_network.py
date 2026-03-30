"""
Citation Network Analysis Module

Provides tools for analyzing citation networks, identifying influential papers,
and discovering research trends through graph analysis.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional, Iterator

import networkx as nx
import numpy as np
from networkx.algorithms import centrality, community, traversal

from .literature_miner import Paper


@dataclass
class CitationNode:
    """A node in the citation network."""
    paper_id: str
    paper: Optional[Paper] = None
    citation_count: int = 0
    reference_count: int = 0
    year: int = 0
    depth: int = 0  # Distance from seed papers
    
    def __hash__(self) -> int:
        return hash(self.paper_id)


@dataclass
class CitationEdge:
    """An edge in the citation network (citation relationship)."""
    source: str  # Citing paper
    target: str  # Cited paper
    weight: float = 1.0
    
    def __hash__(self) -> int:
        return hash((self.source, self.target))


class CitationNetwork:
    """
    Citation network for analyzing paper relationships and influence.
    """
    
    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self.papers: dict[str, Paper] = {}
        self._nodes: dict[str, CitationNode] = {}
        self._edges: set[tuple[str, str]] = set()
    
    def add_paper(self, paper: Paper, depth: int = 0) -> None:
        """Add a paper as a node in the network."""
        paper_id = paper.unique_id
        
        if paper_id not in self.papers:
            self.papers[paper_id] = paper
            self.graph.add_node(
                paper_id,
                title=paper.title,
                year=paper.year,
                citations=paper.citations,
                depth=depth
            )
            self._nodes[paper_id] = CitationNode(
                paper_id=paper_id,
                paper=paper,
                citation_count=paper.citations,
                year=paper.year,
                depth=depth
            )
    
    def add_citation(self, citing_id: str, cited_id: str, weight: float = 1.0) -> None:
        """Add a citation edge."""
        edge_key = (citing_id, cited_id)
        
        if edge_key not in self._edges:
            self._edges.add(edge_key)
            self.graph.add_edge(citing_id, cited_id, weight=weight)
            
            # Update node counts
            if citing_id in self._nodes:
                self._nodes[citing_id].reference_count += 1
    
    def build_from_papers(
        self,
        papers: list[Paper],
        citation_field: str = "citations"
    ) -> None:
        """Build network from a collection of papers."""
        # Add all papers as nodes
        for paper in papers:
            self.add_paper(paper)
        
        # Try to infer citations from common references
        self._infer_citations_from_references(papers)
    
    def _infer_citations_from_references(self, papers: list[Paper]) -> None:
        """Infer citation edges from reference lists."""
        paper_by_id = {p.unique_id: p for p in papers}
        
        # Look for reference information in raw_data
        for paper in papers:
            if not hasattr(paper, 'raw_data') or not paper.raw_data:
                continue
            
            references = paper.raw_data.get('reference', [])
            for ref in references:
                ref_doi = ref.get('DOI') if isinstance(ref, dict) else None
                if ref_doi:
                    ref_id = f"doi:{ref_doi}"
                    if ref_id in paper_by_id:
                        self.add_citation(paper.unique_id, ref_id)
    
    def get_influential_papers(
        self,
        method: str = "pagerank",
        top_k: int = 10
    ) -> list[tuple[Paper, float]]:
        """
        Identify most influential papers using various centrality measures.
        
        Methods:
            - pagerank: PageRank centrality
            - betweenness: Betweenness centrality
            - eigenvector: Eigenvector centrality
            - degree: In-degree centrality (citation count)
            - closeness: Closeness centrality
        """
        if len(self.graph) == 0:
            return []
        
        scores: dict[str, float] = {}
        
        if method == "pagerank":
            scores = centrality.pagerank(self.graph, alpha=0.85)
        elif method == "betweenness":
            scores = centrality.betweenness_centrality(self.graph)
        elif method == "eigenvector":
            try:
                scores = centrality.eigenvector_centrality_numpy(self.graph)
            except (nx.PowerIterationFailedConvergence, nx.AmbiguousSolution):
                # Fall back to degree centrality
                scores = dict(self.graph.in_degree())
        elif method == "degree":
            scores = dict(self.graph.in_degree())
        elif method == "closeness":
            # Use undirected for closeness
            scores = centrality.closeness_centrality(self.graph.to_undirected())
        else:
            raise ValueError(f"Unknown centrality method: {method}")
        
        # Sort and return top-k
        sorted_papers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        result = []
        for paper_id, score in sorted_papers[:top_k]:
            paper = self.papers.get(paper_id)
            if paper:
                result.append((paper, score))
        
        return result
    
    def get_research_clusters(self, resolution: float = 1.0) -> list[set[str]]:
        """
        Detect research clusters using community detection.
        """
        if len(self.graph) == 0:
            return []
        
        # Use undirected graph for community detection
        undirected = self.graph.to_undirected()
        
        # Louvain community detection
        communities_gen = community.louvain_communities(
            undirected,
            resolution=resolution,
            seed=42
        )
        
        return [set(c) for c in communities_gen]
    
    def get_cluster_representatives(
        self,
        clusters: Optional[list[set[str]]] = None,
        top_k: int = 3
    ) -> list[tuple[int, list[Paper]]]:
        """Get representative papers for each cluster."""
        if clusters is None:
            clusters = self.get_research_clusters()
        
        representatives = []
        pagerank = centrality.pagerank(self.graph, alpha=0.85) if len(self.graph) > 0 else {}
        
        for i, cluster in enumerate(clusters):
            # Get top papers by PageRank within cluster
            cluster_papers = [
                (self.papers.get(pid), pagerank.get(pid, 0))
                for pid in cluster
                if pid in self.papers
            ]
            cluster_papers = [(p, s) for p, s in cluster_papers if p is not None]
            cluster_papers.sort(key=lambda x: x[1], reverse=True)
            
            representatives.append((i, [p for p, _ in cluster_papers[:top_k]]))
        
        return representatives
    
    def find_research_frontier(self, years_back: int = 3) -> list[Paper]:
        """
        Identify recent highly-cited papers (research frontier).
        """
        from datetime import datetime
        
        current_year = datetime.now().year
        cutoff_year = current_year - years_back
        
        frontier = []
        for paper_id, paper in self.papers.items():
            if paper.year >= cutoff_year and paper.citations > 0:
                frontier.append(paper)
        
        # Sort by citations per year (normalized impact)
        frontier.sort(
            key=lambda p: p.citations / max(1, current_year - p.year + 1),
            reverse=True
        )
        
        return frontier
    
    def find_seminal_papers(self, top_k: int = 10) -> list[Paper]:
        """
        Identify foundational/seminal papers (highly cited, older).
        """
        from datetime import datetime
        
        current_year = datetime.now().year
        
        # Filter for papers at least 10 years old with high citations
        candidates = [
            p for p in self.papers.values()
            if current_year - p.year >= 10 and p.citations > 100
        ]
        
        # Sort by citation count
        candidates.sort(key=lambda p: p.citations, reverse=True)
        
        return candidates[:top_k]
    
    def trace_citation_chain(
        self,
        paper_id: str,
        direction: str = "backward",  # backward = references, forward = citations
        max_depth: int = 3
    ) -> list[list[str]]:
        """
        Trace citation chains from a paper.
        
        Args:
            paper_id: Starting paper ID
            direction: 'backward' for references, 'forward' for citations
            max_depth: Maximum chain length
        """
        if paper_id not in self.graph:
            return []
        
        chains: list[list[str]] = []
        
        if direction == "backward":
            # BFS through references (predecessors)
            for target in nx.bfs_tree(self.graph.reverse(), paper_id, depth_limit=max_depth):
                if target != paper_id:
                    try:
                        path = nx.shortest_path(
                            self.graph.reverse(),
                            paper_id,
                            target
                        )
                        chains.append(path)
                    except nx.NetworkXNoPath:
                        pass
        else:
            # BFS through citations (successors)
            for target in nx.bfs_tree(self.graph, paper_id, depth_limit=max_depth):
                if target != paper_id:
                    try:
                        path = nx.shortest_path(self.graph, paper_id, target)
                        chains.append(path)
                    except nx.NetworkXNoPath:
                        pass
        
        return chains
    
    def get_citation_context(
        self,
        paper_id: str
    ) -> dict[str, Any]:
        """
        Get citation context for a paper (who cites it, what it cites).
        """
        if paper_id not in self.graph:
            return {}
        
        node_data = self.graph.nodes[paper_id]
        
        # Papers that cite this paper
        citing = list(self.graph.predecessors(paper_id))
        
        # Papers this paper cites
        cited = list(self.graph.successors(paper_id))
        
        # Co-citation analysis (papers cited together with this one)
        co_cited: dict[str, int] = defaultdict(int)
        for citer in citing:
            for cited_paper in self.graph.successors(citer):
                if cited_paper != paper_id:
                    co_cited[cited_paper] += 1
        
        # Bibliographic coupling (papers citing the same papers)
        bib_coupling: dict[str, int] = defaultdict(int)
        for cited_paper in cited:
            for citer in self.graph.predecessors(cited_paper):
                if citer != paper_id:
                    bib_coupling[citer] += 1
        
        return {
            "paper_id": paper_id,
            "title": node_data.get('title', 'Unknown'),
            "cited_by_count": len(citing),
            "cited_by": citing[:20],  # Limit to top 20
            "references_count": len(cited),
            "references": cited[:20],
            "co_cited_papers": sorted(
                co_cited.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "bibliographically_coupled": sorted(
                bib_coupling.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
    
    def find_bridge_papers(self, top_k: int = 10) -> list[tuple[Paper, float]]:
        """
        Find papers that bridge different research areas (high betweenness).
        """
        if len(self.graph) == 0:
            return []
        
        betweenness = centrality.betweenness_centrality(self.graph)
        
        sorted_papers = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        
        result = []
        for paper_id, score in sorted_papers[:top_k]:
            paper = self.papers.get(paper_id)
            if paper:
                result.append((paper, score))
        
        return result
    
    def analyze_temporal_trends(self) -> dict[str, Any]:
        """
        Analyze publication and citation trends over time.
        """
        years: dict[int, dict[str, Any]] = defaultdict(
            lambda: {"publications": 0, "citations": 0, "papers": []}
        )
        
        for paper_id, paper in self.papers.items():
            year = paper.year
            years[year]["publications"] += 1
            years[year]["citations"] += paper.citations
            years[year]["papers"].append(paper_id)
        
        # Calculate average citations per paper per year
        for year in years:
            pubs = years[year]["publications"]
            years[year]["avg_citations"] = years[year]["citations"] / max(1, pubs)
        
        return {
            "yearly_data": dict(sorted(years.items())),
            "peak_year": max(years.items(), key=lambda x: x[1]["publications"])[0]
                if years else None,
            "citation_peak": max(years.items(), key=lambda x: x[1]["citations"])[0]
                if years else None
        }
    
    def calculate_h_index(self, paper_ids: Optional[list[str]] = None) -> int:
        """
        Calculate h-index for a set of papers (or all papers).
        """
        papers_to_check = paper_ids if paper_ids else list(self.papers.keys())
        
        citation_counts = sorted([
            self.papers[pid].citations
            for pid in papers_to_check
            if pid in self.papers
        ], reverse=True)
        
        h_index = 0
        for i, citations in enumerate(citation_counts, 1):
            if citations >= i:
                h_index = i
            else:
                break
        
        return h_index
    
    def get_network_stats(self) -> dict[str, Any]:
        """Get comprehensive network statistics."""
        if len(self.graph) == 0:
            return {"nodes": 0, "edges": 0}
        
        undirected = self.graph.to_undirected()
        
        # Degree distribution
        in_degrees = [d for _, d in self.graph.in_degree()]
        out_degrees = [d for _, d in self.graph.out_degree()]
        
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "is_connected": nx.is_weakly_connected(self.graph),
            "num_weakly_connected_components": nx.number_weakly_connected_components(
                self.graph
            ),
            "avg_in_degree": np.mean(in_degrees) if in_degrees else 0,
            "avg_out_degree": np.mean(out_degrees) if out_degrees else 0,
            "max_in_degree": max(in_degrees) if in_degrees else 0,
            "clustering_coefficient": nx.average_clustering(undirected),
            "diameter": nx.diameter(undirected) if nx.is_connected(undirected) else None
        }
    
    def export_to_graphml(self, filepath: str) -> None:
        """Export network to GraphML format."""
        nx.write_graphml(self.graph, filepath)
    
    def export_to_json(self, filepath: str) -> None:
        """Export network to JSON format."""
        data = {
            "nodes": [
                {
                    "id": node_id,
                    **self.graph.nodes[node_id]
                }
                for node_id in self.graph.nodes()
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    **data
                }
                for u, v, data in self.graph.edges(data=True)
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def find_gaps(self, min_citation_gap: int = 50) -> list[dict[str, Any]]:
        """
        Identify potential research gaps based on citation patterns.
        
        A gap exists when a highly cited paper has few connections to
        papers in different research areas.
        """
        gaps = []
        
        # Get communities
        communities = self.get_research_clusters()
        paper_to_community: dict[str, int] = {}
        for i, comm in enumerate(communities):
            for pid in comm:
                paper_to_community[pid] = i
        
        # Find papers with high citation but low cross-community citations
        for paper_id, paper in self.papers.items():
            if paper.citations < min_citation_gap:
                continue
            
            if paper_id not in paper_to_community:
                continue
            
            own_community = paper_to_community[paper_id]
            
            # Check citations from other communities
            citing = list(self.graph.predecessors(paper_id))
            cross_community_citations = sum(
                1 for citer in citing
                if paper_to_community.get(citer, own_community) != own_community
            )
            
            cross_ratio = cross_community_citations / max(1, len(citing))
            
            # Low cross-community citation suggests a gap
            if cross_ratio < 0.1 and len(citing) > 10:
                gaps.append({
                    "paper_id": paper_id,
                    "title": paper.title,
                    "citations": paper.citations,
                    "cross_community_citations": cross_community_citations,
                    "cross_ratio": cross_ratio,
                    "community": own_community,
                    "gap_type": "underexplored_connections"
                })
        
        return sorted(gaps, key=lambda x: x["citations"], reverse=True)


def build_network_from_search_results(
    papers: list[Paper],
    max_edges_per_paper: int = 10
) -> CitationNetwork:
    """
    Build a citation network from search results.
    
    Since we don't have full citation data, we infer relationships
    based on shared references, authors, and keywords.
    """
    network = CitationNetwork()
    
    # Add all papers
    for paper in papers:
        network.add_paper(paper)
    
    # Infer edges based on similarity
    for i, p1 in enumerate(papers):
        for p2 in papers[i+1:]:
            weight = _calculate_paper_similarity(p1, p2)
            
            # Add edge if similarity is high enough
            if weight > 0.3:
                # Direction: newer cites older
                if p1.year >= p2.year:
                    network.add_citation(p1.unique_id, p2.unique_id, weight)
                else:
                    network.add_citation(p2.unique_id, p1.unique_id, weight)
    
    return network


def _calculate_paper_similarity(p1: Paper, p2: Paper) -> float:
    """Calculate similarity between two papers."""
    score = 0.0
    
    # Author overlap
    authors1 = set(a.lower() for a in p1.authors)
    authors2 = set(a.lower() for a in p2.authors)
    if authors1 and authors2:
        author_overlap = len(authors1 & authors2) / max(len(authors1), len(authors2))
        score += author_overlap * 0.3
    
    # Keyword overlap
    keywords1 = set(k.lower() for k in p1.keywords)
    keywords2 = set(k.lower() for k in p2.keywords)
    if keywords1 and keywords2:
        keyword_overlap = len(keywords1 & keywords2) / max(len(keywords1), len(keywords2))
        score += keyword_overlap * 0.4
    
    # Title similarity (simple word overlap)
    words1 = set(p1.title.lower().split())
    words2 = set(p2.title.lower().split())
    common_words = words1 & words2
    # Remove common stopwords
    stopwords = {'the', 'a', 'an', 'in', 'on', 'of', 'and', 'or', 'for', 'to'}
    common_words -= stopwords
    if words1 and words2:
        title_sim = len(common_words) / max(len(words1), len(words2))
        score += title_sim * 0.3
    
    return min(score, 1.0)


async def demo():
    """Demo citation network analysis."""
    from .literature_miner import LiteratureMiner
    
    miner = LiteratureMiner()
    
    # Search for papers
    print("Searching for papers...")
    papers = await miner.search(
        "machine learning materials discovery",
        max_results=20
    )
    
    print(f"Found {len(papers)} papers")
    
    # Build network
    network = build_network_from_search_results(papers)
    
    # Analyze
    print("\n=== Network Statistics ===")
    stats = network.get_network_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n=== Most Influential Papers (PageRank) ===")
    influential = network.get_influential_papers(method="pagerank", top_k=5)
    for paper, score in influential:
        print(f"  {paper.title[:60]}... (score: {score:.4f})")
    
    print("\n=== Research Clusters ===")
    clusters = network.get_research_clusters()
    print(f"  Found {len(clusters)} clusters")
    
    representatives = network.get_cluster_representatives(clusters, top_k=2)
    for cluster_id, papers in representatives:
        print(f"\n  Cluster {cluster_id}:")
        for paper in papers:
            print(f"    - {paper.title[:50]}...")
    
    print("\n=== Bridge Papers ===")
    bridges = network.find_bridge_papers(top_k=3)
    for paper, score in bridges:
        print(f"  {paper.title[:60]}... (betweenness: {score:.4f})")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo())
