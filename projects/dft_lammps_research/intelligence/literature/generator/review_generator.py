"""
综述报告生成器
自动生成结构化文献综述
"""

import uuid
from typing import List, Dict, Optional, Any
from datetime import datetime
from collections import defaultdict

from ..config.models import Paper, LiteratureReview, ResearchTrend, MethodComparison, ResearchGap
from ..config.database import DatabaseManager
from ..config.settings import REPORT_CONFIG
from ..analysis.topic_modeling import TopicModeler, KeywordExtractor
from ..analysis.trend_analysis import TrendAnalyzer, CitationAnalyzer
from ..analysis.method_extraction import MethodExtractor
from ..analysis.knowledge_graph import KnowledgeGraphBuilder


class ReviewGenerator:
    """综述生成器"""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()
        self.topic_modeler = TopicModeler()
        self.keyword_extractor = KeywordExtractor()
        self.trend_analyzer = TrendAnalyzer()
        self.citation_analyzer = CitationAnalyzer()
        self.method_extractor = MethodExtractor()
        self.kg_builder = KnowledgeGraphBuilder()
    
    def generate_review(
        self,
        papers: List[Paper],
        title: Optional[str] = None,
        query: str = "",
        include_sections: Optional[List[str]] = None
    ) -> LiteratureReview:
        """
        生成文献综述
        
        Args:
            papers: 论文列表
            title: 综述标题
            query: 查询条件
            include_sections: 包含的章节
        
        Returns:
            综述报告
        """
        if not papers:
            raise ValueError("论文列表为空")
        
        include_sections = include_sections or REPORT_CONFIG["sections"]
        
        # 生成ID和标题
        review_id = str(uuid.uuid4())
        if not title:
            keywords = self.keyword_extractor.extract(papers)[:3]
            title = f"Literature Review: {', '.join([k[0] for k in keywords])}"
        
        # 创建综述对象
        review = LiteratureReview(
            id=review_id,
            title=title,
            query=query,
            created_at=datetime.now(),
            papers=papers
        )
        
        # 执行各项分析
        print("执行主题建模...")
        review.topics = self._analyze_topics(papers)
        
        print("分析研究趋势...")
        review.trends = self._analyze_trends(papers)
        
        print("分析方法对比...")
        review.methods = self._analyze_methods(papers)
        
        print("识别研究空白...")
        review.gaps = self._identify_gaps(papers, review.trends)
        
        print("生成总结...")
        review.summary = self._generate_summary(review)
        
        print("生成章节内容...")
        review.sections = self._generate_sections(review, include_sections)
        
        # 设置统计信息
        review.total_papers = len(papers)
        dates = [p.publication_date for p in papers]
        review.date_range = (min(dates), max(dates))
        
        # 保存到数据库
        self.db.save_review(review)
        
        return review
    
    def _analyze_topics(self, papers: List[Paper]) -> List[str]:
        """分析主题"""
        topic_labels = self.topic_modeler.fit(papers)
        return list(topic_labels.values())
    
    def _analyze_trends(self, papers: List[Paper]) -> List[ResearchTrend]:
        """分析趋势"""
        return self.trend_analyzer.analyze(papers)
    
    def _analyze_methods(self, papers: List[Paper]) -> List[MethodComparison]:
        """分析方法"""
        return self.method_extractor.analyze_methods(papers)
    
    def _identify_gaps(
        self,
        papers: List[Paper],
        trends: List[ResearchTrend]
    ) -> List[ResearchGap]:
        """识别研究空白"""
        gaps = []
        
        # 基于趋势识别空白
        emerging = self.trend_analyzer.get_emerging_topics(papers, min_growth_rate=0.3)
        declining = self.trend_analyzer.get_declining_topics(papers, max_growth_rate=-0.1)
        
        # 新兴但研究不足的领域
        for topic, growth in emerging[:3]:
            topic_papers = [p for p in papers if topic in p.topics]
            if len(topic_papers) < 5:
                gaps.append(ResearchGap(
                    area=topic,
                    description=f"Emerging area with high growth rate ({growth:.1%}) but limited research coverage",
                    evidence=[f"Only {len(topic_papers)} papers found"],
                    potential_impact="High potential for breakthrough discoveries",
                    suggested_approaches=["Cross-disciplinary collaboration", "Large-scale computational screening"]
                ))
        
        # 方法组合空白
        methods_used = defaultdict(set)
        for paper in papers:
            for method in paper.methods:
                methods_used[method].add(paper.id)
        
        # 找出很少一起使用的方法组合
        common_combinations = [
            ("DFT", "Machine Learning"),
            ("Molecular Dynamics", "Machine Learning"),
            ("DFT", "Molecular Dynamics")
        ]
        
        for m1, m2 in common_combinations:
            if m1 in methods_used and m2 in methods_used:
                combined = methods_used[m1] & methods_used[m2]
                total = methods_used[m1] | methods_used[m2]
                
                if len(combined) < len(total) * 0.2:
                    gaps.append(ResearchGap(
                        area=f"Integration of {m1} and {m2}",
                        description=f"Limited integration between {m1} and {m2} methodologies",
                        evidence=[f"Only {len(combined)} papers use both methods out of {len(total)} total"],
                        potential_impact="Could combine accuracy and efficiency advantages",
                        suggested_approaches=["Develop multi-scale workflows", "Create hybrid potentials"]
                    ))
        
        # 基于关键词覆盖率的空白
        all_keywords = set()
        for paper in papers:
            all_keywords.update(paper.keywords)
        
        # 检查是否有重要的材料系统未被充分研究
        material_systems = ["Li", "Na", "K", "Mg", "Ca", "Zn"]
        studied_systems = set()
        
        for paper in papers:
            text = f"{paper.title} {paper.abstract}".lower()
            for system in material_systems:
                if system.lower() in text or f" {system} " in text:
                    studied_systems.add(system)
        
        unstudied = set(material_systems) - studied_systems
        if unstudied:
            gaps.append(ResearchGap(
                area="Novel Material Systems",
                description=f"Limited research on alternative material systems: {', '.join(unstudied)}",
                evidence=[f"Only {len(studied_systems)}/{len(material_systems)} systems studied"],
                potential_impact="Could discover new high-performance materials",
                suggested_approaches=["Expand computational screening", "Benchmark across systems"]
            ))
        
        return gaps
    
    def _generate_summary(self, review: LiteratureReview) -> str:
        """生成摘要总结"""
        lines = []
        
        lines.append(f"# {review.title}")
        lines.append("")
        lines.append(f"**Query:** {review.query}")
        lines.append(f"**Total Papers:** {len(review.papers)}")
        lines.append(f"**Date Range:** {review.date_range[0].strftime('%Y-%m-%d')} to {review.date_range[1].strftime('%Y-%m-%d')}")
        lines.append("")
        
        # 关键发现
        lines.append("## Key Findings")
        lines.append("")
        
        # 热门主题
        if review.topics:
            lines.append(f"- **Main Topics:** {', '.join(review.topics[:5])}")
        
        # 主要方法
        if review.methods:
            top_methods = sorted(review.methods, key=lambda x: x.paper_count, reverse=True)[:3]
            lines.append(f"- **Dominant Methods:** {', '.join([m.method_name for m in top_methods])}")
        
        # 引用影响
        impact = self.citation_analyzer.analyze_impact(review.papers)
        if impact:
            lines.append(f"- **Total Citations:** {impact.get('total_citations', 0)}")
            lines.append(f"- **Average Citations per Paper:** {impact.get('avg_citations', 0):.1f}")
        
        lines.append("")
        
        # 研究趋势
        lines.append("## Research Trends")
        lines.append("")
        
        if review.trends:
            # 找出增长最快的主题
            growth_rates = defaultdict(list)
            for trend in review.trends:
                growth_rates[trend.topic].append(trend.growth_rate)
            
            avg_growth = {
                topic: sum(rates) / len(rates)
                for topic, rates in growth_rates.items()
            }
            
            top_growing = sorted(avg_growth.items(), key=lambda x: x[1], reverse=True)[:3]
            
            lines.append("### Fastest Growing Areas")
            for topic, rate in top_growing:
                lines.append(f"- **{topic}:** {rate:+.1%} annual growth")
            
            lines.append("")
        
        # 研究空白
        if review.gaps:
            lines.append("## Identified Research Gaps")
            lines.append("")
            for gap in review.gaps[:3]:
                lines.append(f"### {gap.area}")
                lines.append(gap.description)
                lines.append("")
        
        return "\n".join(lines)
    
    def _generate_sections(
        self,
        review: LiteratureReview,
        sections: List[str]
    ) -> Dict[str, str]:
        """生成各章节内容"""
        content = {}
        
        if "introduction" in sections:
            content["introduction"] = self._generate_introduction(review)
        
        if "methodology_overview" in sections:
            content["methodology_overview"] = self._generate_methodology_overview(review)
        
        if "key_findings" in sections:
            content["key_findings"] = self._generate_key_findings(review)
        
        if "trend_analysis" in sections:
            content["trend_analysis"] = self._generate_trend_analysis(review)
        
        if "method_comparison" in sections:
            content["method_comparison"] = self._generate_method_comparison(review)
        
        if "research_gaps" in sections:
            content["research_gaps"] = self._generate_research_gaps(review)
        
        if "future_outlook" in sections:
            content["future_outlook"] = self._generate_future_outlook(review)
        
        if "references" in sections:
            content["references"] = self._generate_references(review)
        
        return content
    
    def _generate_introduction(self, review: LiteratureReview) -> str:
        """生成引言"""
        lines = ["# Introduction", ""]
        
        lines.append(f"This literature review analyzes {len(review.papers)} research papers")
        lines.append(f"published between {review.date_range[0].year} and {review.date_range[1].year}.")
        lines.append("")
        
        lines.append("## Scope and Objectives")
        lines.append("")
        lines.append("The primary objectives of this review are to:")
        lines.append("1. Identify major research themes and trends")
        lines.append("2. Compare computational methodologies")
        lines.append("3. Highlight significant findings")
        lines.append("4. Identify research gaps and future directions")
        lines.append("")
        
        return "\n".join(lines)
    
    def _generate_methodology_overview(self, review: LiteratureReview) -> str:
        """生成方法概述"""
        lines = ["# Methodology Overview", ""]
        
        if not review.methods:
            lines.append("No specific methods identified in the analyzed papers.")
            return "\n".join(lines)
        
        lines.append("## Computational Methods Used")
        lines.append("")
        
        for method in review.methods[:10]:
            lines.append(f"### {method.method_name}")
            lines.append(f"- **Usage:** {method.paper_count} papers")
            if method.software_used:
                lines.append(f"- **Common Software:** {', '.join(method.software_used[:5])}")
            if method.pros:
                lines.append(f"- **Advantages:** {', '.join(method.pros[:3])}")
            if method.cons:
                lines.append(f"- **Limitations:** {', '.join(method.cons[:3])}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_key_findings(self, review: LiteratureReview) -> str:
        """生成关键发现"""
        lines = ["# Key Findings", ""]
        
        # 高引用论文
        key_papers = self.citation_analyzer.find_key_papers(review.papers, top_n=10)
        
        if key_papers:
            lines.append("## Highly Cited Papers")
            lines.append("")
            for i, paper in enumerate(key_papers, 1):
                lines.append(f"{i}. **{paper.title}** ({paper.publication_date.year})")
                lines.append(f"   - {paper.get_author_names()}")
                lines.append(f"   - Citations: {paper.citation_count}")
                if paper.methods:
                    lines.append(f"   - Methods: {', '.join(paper.methods[:3])}")
                lines.append("")
        
        # 主题发现
        if review.topics:
            lines.append("## Major Research Themes")
            lines.append("")
            for topic in review.topics[:8]:
                topic_papers = [p for p in review.papers if topic in p.topics]
                lines.append(f"- **{topic}:** {len(topic_papers)} papers")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_trend_analysis(self, review: LiteratureReview) -> str:
        """生成趋势分析"""
        lines = ["# Trend Analysis", ""]
        
        if not review.trends:
            lines.append("Insufficient data for trend analysis.")
            return "\n".join(lines)
        
        # 按主题分组
        topic_trends = defaultdict(list)
        for trend in review.trends:
            topic_trends[trend.topic].append(trend)
        
        lines.append("## Publication Trends by Topic")
        lines.append("")
        
        for topic, trends in sorted(topic_trends.items(), 
                                   key=lambda x: sum(t.paper_count for t in x[1]), 
                                   reverse=True)[:8]:
            total_papers = sum(t.paper_count for t in trends)
            avg_growth = sum(t.growth_rate for t in trends) / len(trends)
            
            lines.append(f"### {topic}")
            lines.append(f"- **Total Papers:** {total_papers}")
            lines.append(f"- **Average Growth:** {avg_growth:+.1%}")
            
            # 显示年度数据
            for trend in sorted(trends, key=lambda x: x.year)[-3:]:
                lines.append(f"  - {trend.year}: {trend.paper_count} papers, {trend.citation_count} citations")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_method_comparison(self, review: LiteratureReview) -> str:
        """生成方法对比"""
        lines = ["# Method Comparison", ""]
        
        if len(review.methods) < 2:
            lines.append("Insufficient methods for comparison.")
            return "\n".join(lines)
        
        # 排序方法
        sorted_methods = sorted(review.methods, key=lambda x: x.paper_count, reverse=True)
        
        lines.append("## Usage Statistics")
        lines.append("")
        lines.append("| Method | Papers | Software | Pros | Cons |")
        lines.append("|--------|--------|----------|------|------|")
        
        for method in sorted_methods[:10]:
            pros = ", ".join(method.pros[:2]) if method.pros else "-"
            cons = ", ".join(method.cons[:2]) if method.cons else "-"
            sw = ", ".join(method.software_used[:2]) if method.software_used else "-"
            
            lines.append(f"| {method.method_name} | {method.paper_count} | {sw} | {pros} | {cons} |")
        
        lines.append("")
        
        return "\n".join(lines)
    
    def _generate_research_gaps(self, review: LiteratureReview) -> str:
        """生成研究空白"""
        lines = ["# Research Gaps", ""]
        
        if not review.gaps:
            lines.append("No significant research gaps identified.")
            return "\n".join(lines)
        
        for gap in review.gaps:
            lines.append(f"## {gap.area}")
            lines.append("")
            lines.append(f"**Description:** {gap.description}")
            lines.append("")
            
            if gap.evidence:
                lines.append("**Evidence:**")
                for ev in gap.evidence:
                    lines.append(f"- {ev}")
                lines.append("")
            
            if gap.potential_impact:
                lines.append(f"**Potential Impact:** {gap.potential_impact}")
                lines.append("")
            
            if gap.suggested_approaches:
                lines.append("**Suggested Approaches:**")
                for approach in gap.suggested_approaches:
                    lines.append(f"- {approach}")
                lines.append("")
        
        return "\n".join(lines)
    
    def _generate_future_outlook(self, review: LiteratureReview) -> str:
        """生成未来展望"""
        lines = ["# Future Outlook", ""]
        
        lines.append("## Emerging Trends")
        lines.append("")
        
        # 预测未来趋势
        predictions = self.trend_analyzer.predict_trends(review.trends, forecast_years=2)
        
        for topic, forecast in sorted(predictions.items(), 
                                     key=lambda x: x[1][-1][1] if x[1] else 0, 
                                     reverse=True)[:5]:
            if forecast:
                current = forecast[0][1] if forecast else 0
                future = forecast[-1][1] if forecast else 0
                lines.append(f"- **{topic}:** Expected to grow from {current:.0f} to {future:.0f} papers/year")
        
        lines.append("")
        
        lines.append("## Recommendations for Future Research")
        lines.append("")
        lines.append("Based on the identified gaps and trends, we recommend:")
        lines.append("")
        
        if review.gaps:
            for gap in review.gaps[:3]:
                lines.append(f"1. **{gap.area}:** {gap.suggested_approaches[0] if gap.suggested_approaches else 'Further investigation needed'}")
        
        lines.append("")
        
        return "\n".join(lines)
    
    def _generate_references(self, review: LiteratureReview) -> str:
        """生成参考文献"""
        lines = ["# References", ""]
        
        # 按年份排序
        sorted_papers = sorted(review.papers, 
                              key=lambda x: x.publication_date, 
                              reverse=True)
        
        for i, paper in enumerate(sorted_papers, 1):
            authors = paper.get_author_names()
            if len(authors) > 50:
                authors = authors[:50] + " et al."
            
            lines.append(f"[{i}] {authors}")
            lines.append(f"    \"{paper.title}\"")
            
            if paper.journal:
                lines.append(f"    {paper.journal}, {paper.publication_date.year}")
            else:
                lines.append(f"    {paper.publication_date.year}")
            
            if paper.doi:
                lines.append(f"    DOI: {paper.doi}")
            elif paper.arxiv_id:
                lines.append(f"    arXiv:{paper.arxiv_id}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def export_markdown(self, review: LiteratureReview, output_path: str):
        """导出为Markdown"""
        content = review.summary
        
        for section_name, section_content in review.sections.items():
            content += "\n\n" + section_content
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"综述已导出: {output_path}")
    
    def export_html(self, review: LiteratureReview, output_path: str):
        """导出为HTML"""
        try:
            import markdown
            
            md_content = review.summary
            for section_content in review.sections.values():
                md_content += "\n\n" + section_content
            
            html_content = markdown.markdown(md_content, extensions=['tables', 'toc'])
            
            html_template = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>{review.title}</title>
                <style>
                    body {{
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        max-width: 900px;
                        margin: 0 auto;
                        padding: 20px;
                        line-height: 1.6;
                        color: #333;
                    }}
                    h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                    h2 {{ color: #34495e; margin-top: 30px; }}
                    h3 {{ color: #7f8c8d; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    th {{ background-color: #3498db; color: white; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    code {{ background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
                    blockquote {{
                        border-left: 4px solid #3498db;
                        margin: 0;
                        padding-left: 20px;
                        color: #666;
                    }}
                </style>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_template)
            
            print(f"HTML已导出: {output_path}")
        
        except ImportError:
            print("需要安装markdown库: pip install markdown")
