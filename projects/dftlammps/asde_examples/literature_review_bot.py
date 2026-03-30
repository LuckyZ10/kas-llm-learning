#!/usr/bin/env python3
"""
Literature Review Bot

Automatically generates comprehensive literature reviews by:
1. Mining papers from multiple sources (arXiv, PubMed, CrossRef)
2. Building citation networks
3. Extracting key themes and trends
4. Generating a structured literature review document
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from dftlammps.asde.knowledge_graph import (
    LiteratureMiner,
    CitationNetwork,
    ScientificKnowledgeGraph,
    build_network_from_search_results,
    Paper,
)
from dftlammps.asde.paper_writer import PaperWriter, ScientificPaper, PaperSection


@dataclass
class ReviewSection:
    """A section of the literature review."""
    title: str
    papers: list[Paper] = field(default_factory=list)
    key_themes: list[str] = field(default_factory=list)
    summary: str = ""


class LiteratureReviewBot:
    """
    Automated literature review generation system.
    """
    
    def __init__(
        self,
        topic: str,
        keywords: list[str],
        year_range: Optional[tuple[int, int]] = None
    ) -> None:
        self.topic = topic
        self.keywords = keywords
        self.year_range = year_range or (2015, datetime.now().year)
        self.miner = LiteratureMiner()
        self.papers: list[Paper] = []
        self.citation_network: Optional[CitationNetwork] = None
        self.knowledge_graph = ScientificKnowledgeGraph()
        
    async def search_literature(
        self,
        max_results: int = 50,
        sources: Optional[list[str]] = None
    ) -> list[Paper]:
        """Search for relevant papers across multiple sources."""
        print(f"Searching for papers on: {self.topic}")
        print(f"Keywords: {', '.join(self.keywords)}")
        
        # Build search query
        query = f"({self.topic})"
        if self.keywords:
            query += f" AND ({' OR '.join(self.keywords)})"
        
        # Search across sources
        all_papers: list[Paper] = []
        
        # Try arXiv for physics/materials papers
        print("\nSearching arXiv...")
        arxiv_papers = await self.miner.search(
            query,
            max_results=max_results // 2,
            sources=['arxiv']
        )
        print(f"  Found {len(arxiv_papers)} papers on arXiv")
        all_papers.extend(arxiv_papers)
        
        # Try CrossRef for broader coverage
        print("Searching CrossRef...")
        crossref_papers = await self.miner.search(
            query,
            max_results=max_results // 2,
            sources=['crossref']
        )
        print(f"  Found {len(crossref_papers)} papers on CrossRef")
        all_papers.extend(crossref_papers)
        
        # Filter by year range
        filtered_papers = [
            p for p in all_papers
            if self.year_range[0] <= p.year <= self.year_range[1]
        ]
        
        # Sort by citation count (as proxy for importance)
        filtered_papers.sort(key=lambda p: p.citations, reverse=True)
        
        self.papers = filtered_papers[:max_results]
        print(f"\nTotal unique papers collected: {len(self.papers)}")
        
        return self.papers
    
    def build_citation_network(self) -> CitationNetwork:
        """Build citation network from collected papers."""
        print("\nBuilding citation network...")
        
        self.citation_network = build_network_from_search_results(self.papers)
        
        stats = self.citation_network.get_network_stats()
        print(f"  Network nodes: {stats['nodes']}")
        print(f"  Network edges: {stats['edges']}")
        print(f"  Connected components: {stats['num_weakly_connected_components']}")
        
        return self.citation_network
    
    def extract_knowledge(self) -> ScientificKnowledgeGraph:
        """Extract knowledge from paper abstracts."""
        print("\nExtracting knowledge from abstracts...")
        
        for paper in self.papers:
            if paper.abstract:
                self.knowledge_graph.extract_from_text(
                    paper.abstract,
                    source=paper.unique_id
                )
        
        kg_stats = self.knowledge_graph.get_statistics()
        print(f"  Entities extracted: {kg_stats['total_entities']}")
        print(f"  Relations extracted: {kg_stats['total_relations']}")
        
        return self.knowledge_graph
    
    def analyze_trends(self) -> dict[str, Any]:
        """Analyze publication trends over time."""
        print("\nAnalyzing publication trends...")
        
        # Year distribution
        year_counts = Counter(p.year for p in self.papers)
        
        # Topic trends (using keywords)
        keyword_trends: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for paper in self.papers:
            for keyword in paper.keywords:
                keyword_trends[keyword][paper.year] += 1
        
        # Author trends
        author_counts = Counter()
        for paper in self.papers:
            for author in paper.authors[:3]:  # Top 3 authors per paper
                author_counts[author] += 1
        
        trends = {
            "year_distribution": dict(sorted(year_counts.items())),
            "keyword_trends": {
                kw: dict(years) for kw, years in keyword_trends.items()
            },
            "top_authors": dict(author_counts.most_common(10)),
            "total_citations": sum(p.citations for p in self.papers),
            "avg_citations": sum(p.citations for p in self.papers) / len(self.papers)
                if self.papers else 0
        }
        
        print(f"  Publication years: {min(year_counts.keys())} - {max(year_counts.keys())}")
        print(f"  Total citations: {trends['total_citations']}")
        print(f"  Average citations: {trends['avg_citations']:.1f}")
        
        return trends
    
    def identify_key_papers(self, top_k: int = 10) -> dict[str, list[tuple[Paper, Any]]]:
        """Identify key papers using multiple criteria."""
        print("\nIdentifying key papers...")
        
        if not self.citation_network:
            self.build_citation_network()
        
        key_papers = {}
        
        # Most cited
        by_citations = sorted(self.papers, key=lambda p: p.citations, reverse=True)[:top_k]
        key_papers['most_cited'] = [(p, p.citations) for p in by_citations]
        print(f"  Most cited: {by_citations[0].title[:50]}... ({by_citations[0].citations} citations)"
              if by_citations else "  No papers found")
        
        # Most influential (PageRank)
        if self.citation_network:
            influential = self.citation_network.get_influential_papers(
                method="pagerank",
                top_k=top_k
            )
            key_papers['most_influential'] = influential
            if influential:
                print(f"  Most influential: {influential[0][0].title[:50]}...")
        
        # Research frontier (recent highly cited)
        if self.citation_network:
            frontier = self.citation_network.find_research_frontier(years_back=3)[:top_k]
            key_papers['research_frontier'] = [(p, p.year) for p in frontier]
            if frontier:
                print(f"  Research frontier: {frontier[0].title[:50]}...")
        
        return key_papers
    
    def identify_research_clusters(self) -> list[ReviewSection]:
        """Identify and characterize research clusters."""
        print("\nIdentifying research clusters...")
        
        if not self.citation_network:
            self.build_citation_network()
        
        # Detect communities
        communities = self.citation_network.get_research_clusters()
        representatives = self.citation_network.get_cluster_representatives(
            communities,
            top_k=3
        )
        
        sections: list[ReviewSection] = []
        
        for cluster_id, papers in representatives:
            if not papers:
                continue
            
            # Extract themes from keywords
            all_keywords = []
            for p in papers:
                all_keywords.extend(p.keywords)
            
            top_themes = [kw for kw, _ in Counter(all_keywords).most_common(5)]
            
            section = ReviewSection(
                title=f"Research Theme {cluster_id + 1}",
                papers=papers,
                key_themes=top_themes
            )
            sections.append(section)
        
        print(f"  Identified {len(sections)} research clusters")
        
        return sections
    
    def find_gaps(self) -> list[dict[str, Any]]:
        """Identify potential research gaps."""
        print("\nIdentifying research gaps...")
        
        gaps: list[dict[str, Any]] = []
        
        if self.citation_network:
            # Use citation network gap detection
            network_gaps = self.citation_network.find_gaps(min_citation_gap=5)
            gaps.extend(network_gaps)
        
        # Check for emerging keywords (appearing recently)
        recent_papers = [p for p in self.papers if p.year >= datetime.now().year - 3]
        older_papers = [p for p in self.papers if p.year < datetime.now().year - 3]
        
        recent_keywords = Counter(kw for p in recent_papers for kw in p.keywords)
        older_keywords = Counter(kw for p in older_papers for kw in p.keywords)
        
        emerging = [
            kw for kw, count in recent_keywords.items()
            if count >= 3 and older_keywords.get(kw, 0) < 2
        ]
        
        if emerging:
            gaps.append({
                "type": "emerging_topic",
                "description": "Recently emerging research topics",
                "topics": emerging
            })
        
        print(f"  Found {len(gaps)} potential research gaps")
        
        return gaps
    
    def generate_review(self) -> ScientificPaper:
        """Generate comprehensive literature review."""
        print("\n" + "=" * 70)
        print("GENERATING LITERATURE REVIEW")
        print("=" * 70)
        
        # Gather all analysis
        trends = self.analyze_trends()
        key_papers = self.identify_key_papers(top_k=10)
        clusters = self.identify_research_clusters()
        gaps = self.find_gaps()
        
        # Create writer
        writer = PaperWriter(
            title=f"Literature Review: {self.topic.title()}",
            authors=["Literature Review Bot", "ASDE System"],
            keywords=self.keywords + ["literature review", "bibliometrics"]
        )
        
        # Add related work
        writer.add_related_work(self.papers[:20])
        
        # Build sections
        sections: list[PaperSection] = []
        
        # Introduction
        intro_content = (
            f"This review synthesizes the current state of research on {self.topic}. "
            f"A systematic search across major scientific databases yielded {len(self.papers)} "
            f"relevant publications from {self.year_range[0]} to {self.year_range[1]}. "
            f"The analysis reveals key trends, influential works, and emerging research directions."
        )
        sections.append(PaperSection(title="Introduction", content=intro_content))
        
        # Methodology
        method_content = (
            f"Literature was collected from arXiv and CrossRef databases using the search query: "
            f"'{self.topic}' with keywords: {', '.join(self.keywords)}. "
            f"Papers were filtered to include only those published between {self.year_range[0]} "
            f"and {self.year_range[1]}. Citation network analysis and knowledge graph extraction "
            f"were performed to identify key themes and research clusters."
        )
        sections.append(PaperSection(title="Methodology", content=method_content))
        
        # Publication Trends
        year_dist = trends['year_distribution']
        if year_dist:
            peak_year = max(year_dist.items(), key=lambda x: x[1])[0]
            trend_content = (
                f"Publication activity in this domain has varied over the study period. "
                f"Peak publication occurred in {peak_year} with {year_dist[peak_year]} papers. "
                f"The total corpus includes {trends['total_citations']} citations, "
                f"averaging {trends['avg_citations']:.1f} citations per paper."
            )
        else:
            trend_content = "Publication trends analysis was performed on the collected corpus."
        
        sections.append(PaperSection(
            title="Publication Trends",
            content=trend_content
        ))
        
        # Key Papers
        if key_papers.get('most_cited'):
            key_content = "The following papers have been identified as particularly influential:\\begin{itemize}\n"
            for paper, citations in key_papers['most_cited'][:5]:
                key_content += f"\\item \\textit{{{paper.title}}} ({paper.year}) - {citations} citations\n"
            key_content += "\\end{itemize}"
            sections.append(PaperSection(title="Key Papers", content=key_content))
        
        # Research Themes
        if clusters:
            themes_content = "The literature clusters into several distinct research themes:\\begin{enumerate}\n"
            for i, cluster in enumerate(clusters[:5], 1):
                themes_content += f"\\item \\textbf{{{cluster.title}}}: "
                themes_content += f"Characterized by focus on {', '.join(cluster.key_themes[:3])}. "
                themes_content += f"Key papers include works by {cluster.papers[0].authors[0] if cluster.papers else 'various authors'}.\n"
            themes_content += "\\end{enumerate}"
            sections.append(PaperSection(title="Research Themes", content=themes_content))
        
        # Research Gaps
        if gaps:
            gaps_content = "Several potential research gaps have been identified:\\begin{itemize}\n"
            for gap in gaps[:5]:
                if gap.get('type') == 'emerging_topic':
                    gaps_content += f"\\item Emerging topics: {', '.join(gap.get('topics', [])[:3])}\n"
                else:
                    gaps_content += f"\\item {gap.get('description', 'Research opportunity')}\n"
            gaps_content += "\\end{itemize}"
            sections.append(PaperSection(title="Research Gaps", content=gaps_content))
        
        # Conclusion
        conclusion_content = (
            f"This review has synthesized {len(self.papers)} papers on {self.topic}, "
            f"revealing {len(clusters)} major research clusters and several emerging trends. "
            f"The field shows {trends['avg_citations']:.0f} average citations per paper, indicating "
            f"{'high' if trends['avg_citations'] > 20 else 'moderate' if trends['avg_citations'] > 10 else 'low'} "
            f"research impact. Future work should address the identified research gaps, particularly "
            f"in areas with emerging keyword activity."
        )
        sections.append(PaperSection(title="Conclusion", content=conclusion_content))
        
        # Build paper
        abstract = (
            f"This systematic review analyzes {len(self.papers)} publications on {self.topic} "
            f"from {self.year_range[0]}-{self.year_range[1]}. Using citation network analysis and "
            f"knowledge extraction, we identify {len(clusters)} research clusters and "
            f"{len(gaps)} potential research gaps. The analysis reveals key trends and "
            f"highlights opportunities for future research."
        )
        
        paper = ScientificPaper(
            title=f"Literature Review: {self.topic.title()}",
            authors=["Literature Review Bot", "ASDE System"],
            abstract=abstract,
            sections=sections,
            keywords=self.keywords + ["literature review", "bibliometrics", "citation analysis"],
            references=[
                {
                    'key': f"ref{i}",
                    'authors': ', '.join(p.authors[:2]) + (' et al.' if len(p.authors) > 2 else ''),
                    'title': p.title,
                    'journal': p.journal or 'Preprint',
                    'year': str(p.year)
                }
                for i, p in enumerate(self.papers[:20], 1)
            ]
        )
        
        return paper


async def run_literature_review(
    topic: str = "machine learning materials discovery",
    keywords: Optional[list[str]] = None,
    max_papers: int = 30,
    output_dir: str = "/root/.openclaw/workspace/dftlammps/asde_examples/output"
) -> dict[str, Any]:
    """
    Run the complete literature review pipeline.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    if keywords is None:
        keywords = ["machine learning", "materials", "discovery", "screening"]
    
    print("=" * 70)
    print("LITERATURE REVIEW BOT")
    print("=" * 70)
    print(f"\nTopic: {topic}")
    print(f"Keywords: {', '.join(keywords)}")
    print(f"Max papers: {max_papers}")
    print()
    
    # Create bot
    bot = LiteratureReviewBot(
        topic=topic,
        keywords=keywords,
        year_range=(2018, datetime.now().year)
    )
    
    # Run pipeline
    await bot.search_literature(max_results=max_papers)
    
    if not bot.papers:
        print("\nNo papers found. Please check your search criteria.")
        return {"error": "No papers found"}
    
    bot.build_citation_network()
    bot.extract_knowledge()
    
    # Generate review
    paper = bot.generate_review()
    
    # Save outputs
    paper.save(f"{output_dir}/literature_review.md", format="markdown")
    paper.save(f"{output_dir}/literature_review.tex", format="latex")
    paper.save(f"{output_dir}/literature_review.json", format="json")
    
    # Save citation network
    if bot.citation_network:
        bot.citation_network.export_to_json(f"{output_dir}/citation_network.json")
    
    # Save knowledge graph
    bot.knowledge_graph.save(f"{output_dir}/review_knowledge_graph.json")
    
    print("\n" + "=" * 70)
    print("LITERATURE REVIEW COMPLETE")
    print("=" * 70)
    
    print(f"\nGenerated files:")
    print(f"  - Review (Markdown): {output_dir}/literature_review.md")
    print(f"  - Review (LaTeX): {output_dir}/literature_review.tex")
    print(f"  - Review (JSON): {output_dir}/literature_review.json")
    print(f"  - Citation Network: {output_dir}/citation_network.json")
    print(f"  - Knowledge Graph: {output_dir}/review_knowledge_graph.json")
    
    print(f"\nReview Statistics:")
    print(f"  Papers analyzed: {len(bot.papers)}")
    print(f"  Sections: {len(paper.sections)}")
    print(f"  Word count: {len(paper.to_markdown().split())}")
    
    return {
        "topic": topic,
        "papers_analyzed": len(bot.papers),
        "sections": len(paper.sections),
        "files_generated": [
            "literature_review.md",
            "literature_review.tex",
            "literature_review.json",
            "citation_network.json",
            "review_knowledge_graph.json"
        ]
    }


async def main():
    """Main entry point."""
    import sys
    
    # Parse arguments
    topic = sys.argv[1] if len(sys.argv) > 1 else "machine learning materials discovery"
    
    await run_literature_review(topic=topic)


if __name__ == "__main__":
    asyncio.run(main())
