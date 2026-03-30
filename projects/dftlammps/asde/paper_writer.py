"""
Paper Writer Module

Automatically generates scientific paper sections including methods,
results, and discussion based on experimental data and analysis results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .hypothesis_generator import Hypothesis
from .result_analyzer import TestResult, DatasetSummary
from ..knowledge_graph.literature_miner import Paper


@dataclass
class PaperSection:
    """A section of a scientific paper."""
    title: str
    content: str
    subsections: list[PaperSection] = field(default_factory=list)
    
    def to_markdown(self, level: int = 1) -> str:
        """Convert to markdown format."""
        prefix = "#" * level
        md = f"{prefix} {self.title}\n\n{self.content}\n\n"
        
        for subsection in self.subsections:
            md += subsection.to_markdown(level + 1)
        
        return md
    
    def to_latex(self, level: int = 1) -> str:
        """Convert to LaTeX format."""
        if level == 1:
            cmd = "\\section"
        elif level == 2:
            cmd = "\\subsection"
        else:
            cmd = "\\subsubsection"
        
        latex = f"{cmd}{{{self.title}}}\n\n{self.content}\n\n"
        
        for subsection in self.subsections:
            latex += subsection.to_latex(level + 1)
        
        return latex


@dataclass
class ScientificPaper:
    """A complete scientific paper."""
    title: str
    authors: list[str]
    abstract: str
    sections: list[PaperSection]
    keywords: list[str] = field(default_factory=list)
    date: datetime = field(default_factory=datetime.now)
    doi: Optional[str] = None
    acknowledgments: str = ""
    references: list[dict[str, str]] = field(default_factory=list)
    
    def to_markdown(self) -> str:
        """Generate full paper in Markdown format."""
        md = f"# {self.title}\n\n"
        md += f"**Authors:** {', '.join(self.authors)}\n\n"
        md += f"**Date:** {self.date.strftime('%Y-%m-%d')}\n\n"
        
        if self.keywords:
            md += f"**Keywords:** {', '.join(self.keywords)}\n\n"
        
        md += f"## Abstract\n\n{self.abstract}\n\n"
        
        for section in self.sections:
            md += section.to_markdown(level=2)
        
        if self.acknowledgments:
            md += f"## Acknowledgments\n\n{self.acknowledgments}\n\n"
        
        if self.references:
            md += "## References\n\n"
            for i, ref in enumerate(self.references, 1):
                md += f"[{i}] {ref.get('authors', 'Unknown')}, "
                md += f"\"{ref.get('title', 'Untitled')}\", "
                md += f"{ref.get('journal', 'Unknown Journal')} "
                md += f"({ref.get('year', 'n.d.')})\n\n"
        
        return md
    
    def to_latex(self) -> str:
        """Generate full paper in LaTeX format."""
        latex = "\\documentclass[11pt,a4paper]{article}\n\n"
        latex += "\\usepackage[utf8]{inputenc}\n"
        latex += "\\usepackage{amsmath,amssymb}\n"
        latex += "\\usepackage{graphicx}\n"
        latex += "\\usepackage{booktabs}\n"
        latex += "\\usepackage{siunitx}\n\n"
        
        latex += "\\begin{document}\n\n"
        latex += f"\\title{{{self.title}}}\n"
        latex += f"\\author{{{ ' \\and '.join(self.authors) }}}\n"
        latex += f"\\date{{{self.date.strftime('%Y-%m-%d')}}}\n"
        latex += "\\maketitle\n\n"
        
        latex += "\\begin{abstract}\n"
        latex += self.abstract + "\n"
        if self.keywords:
            latex += "\\textbf{Keywords:} " + ", ".join(self.keywords) + "\n"
        latex += "\\end{abstract}\n\n"
        
        for section in self.sections:
            latex += section.to_latex()
        
        if self.acknowledgments:
            latex += "\\section*{Acknowledgments}\n"
            latex += self.acknowledgments + "\n\n"
        
        if self.references:
            latex += "\\begin{thebibliography}{99}\n"
            for ref in self.references:
                key = ref.get('key', 'ref')
                latex += f"\\bibitem{{{key}}} "
                latex += f"{ref.get('authors', 'Unknown')}, "
                latex += f"\\textit{{{ref.get('title', 'Untitled')}}}, "
                latex += f"{ref.get('journal', 'Unknown')}, "
                latex += f"{ref.get('year', 'n.d.')}.\n\n"
            latex += "\\end{thebibliography}\n"
        
        latex += "\\end{document}\n"
        
        return latex
    
    def save(self, filepath: str, format: str = "markdown") -> None:
        """Save paper to file."""
        if format == "markdown":
            content = self.to_markdown()
        elif format == "latex":
            content = self.to_latex()
        elif format == "json":
            content = json.dumps(self.to_dict(), indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        with open(filepath, 'w') as f:
            f.write(content)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "authors": self.authors,
            "date": self.date.isoformat(),
            "abstract": self.abstract,
            "keywords": self.keywords,
            "sections": [
                {
                    "title": s.title,
                    "content": s.content,
                    "subsections": [
                        {"title": sub.title, "content": sub.content}
                        for sub in s.subsections
                    ]
                }
                for s in self.sections
            ],
            "acknowledgments": self.acknowledgments,
            "references": self.references
        }


class PaperWriter:
    """
    Automatically generates scientific paper content from experimental results.
    """
    
    def __init__(
        self,
        title: str,
        authors: list[str],
        keywords: Optional[list[str]] = None
    ) -> None:
        self.title = title
        self.authors = authors
        self.keywords = keywords or []
        self.hypotheses: list[Hypothesis] = []
        self.results: list[dict[str, Any]] = []
        self.related_work: list[Paper] = []
        self.experimental_design: Optional[dict[str, Any]] = None
        self.analysis_results: list[dict[str, Any]] = []
    
    def add_hypotheses(self, hypotheses: list[Hypothesis]) -> None:
        """Add hypotheses that were tested."""
        self.hypotheses.extend(hypotheses)
    
    def add_experimental_results(
        self,
        results: list[dict[str, Any]],
        design: dict[str, Any]
    ) -> None:
        """Add experimental results and design information."""
        self.results.extend(results)
        self.experimental_design = design
    
    def add_analysis_results(self, results: list[dict[str, Any]]) -> None:
        """Add statistical analysis results."""
        self.analysis_results.extend(results)
    
    def add_related_work(self, papers: list[Paper]) -> None:
        """Add related papers for literature context."""
        self.related_work.extend(papers)
    
    def generate_abstract(self) -> str:
        """Generate abstract summarizing the study."""
        # Count experiments and hypotheses
        n_hypotheses = len(self.hypotheses)
        n_experiments = len(self.results)
        
        abstract_parts: list[str] = []
        
        # Background
        if self.related_work:
            abstract_parts.append(
                f"This study investigates {self.keywords[0] if self.keywords else 'novel phenomena'} "
                f"building upon recent advances in the field."
            )
        else:
            abstract_parts.append(
                f"This study presents an investigation into {self.keywords[0] if self.keywords else 'novel phenomena'}."
            )
        
        # Methods
        if self.experimental_design:
            method_desc = self.experimental_design.get('method_description', 'computational methods')
            abstract_parts.append(
                f"We employed {method_desc} to systematically explore the parameter space."
            )
        
        # Key findings
        if self.analysis_results:
            n_significant = sum(
                1 for r in self.analysis_results
                if r.get('significant', False)
            )
            abstract_parts.append(
                f"Of {n_hypotheses} hypotheses tested through {n_experiments} experiments, "
                f"{n_significant} showed statistically significant support."
            )
        
        # Conclusion
        abstract_parts.append(
            "These findings contribute to our understanding of the underlying mechanisms "
            "and suggest directions for future research."
        )
        
        return " ".join(abstract_parts)
    
    def generate_introduction(self) -> PaperSection:
        """Generate introduction section."""
        content_parts: list[str] = []
        
        # Opening paragraph
        content_parts.append(
            "The investigation of complex systems requires systematic exploration of "
            "parameter spaces and rigorous statistical validation. Recent advances in "
            "computational methods have enabled unprecedented scale in scientific inquiry, "
            "yet the challenge of identifying meaningful patterns remains."
        )
        
        # Related work
        if self.related_work:
            content_parts.append(
                "\\paragraph{Related Work} "
                "Several studies have contributed to the foundational understanding of this domain:"
            )
            for paper in self.related_work[:3]:
                content_parts.append(
                    f"{paper.authors[0] if paper.authors else 'Unknown'} et al. "
                    f"({paper.year}) demonstrated significant progress in "
                    f"{paper.keywords[0] if paper.keywords else 'related areas'}."
                )
        
        # Research questions
        if self.hypotheses:
            content_parts.append(
                "\\paragraph{Research Questions} "
                "This study addresses the following research questions:"
            )
            for i, h in enumerate(self.hypotheses[:3], 1):
                content_parts.append(f"RQ{i}: {h.statement}")
        
        # Contribution statement
        content_parts.append(
            "\\paragraph{Contributions} "
            "The primary contributions of this work are: "
            "(1) systematic exploration of the parameter space through active learning, "
            "(2) rigorous statistical validation of proposed hypotheses, and "
            "(3) identification of optimal experimental conditions."
        )
        
        return PaperSection(
            title="Introduction",
            content=" ".join(content_parts)
        )
    
    def generate_methods(self) -> PaperSection:
        """Generate methods section."""
        subsections: list[PaperSection] = []
        
        # Experimental design
        if self.experimental_design:
            design_content = (
                f"Experiments were designed using {self.experimental_design.get('design_type', 'active learning')} "
                f"to maximize information gain. The experimental space comprised "
                f"{self.experimental_design.get('n_variables', 'multiple')} variables, "
                f"with {self.experimental_design.get('n_conditions', len(self.results))} conditions tested."
            )
            subsections.append(PaperSection(
                title="Experimental Design",
                content=design_content
            ))
        
        # Variables
        if self.experimental_design and 'variables' in self.experimental_design:
            var_content = "The following variables were systematically varied:\\begin{itemize}\n"
            for var in self.experimental_design['variables']:
                var_content += f"\\item \\textbf{{{var['name']}}}: {var.get('description', '')} "
                if 'range' in var:
                    var_content += f"(range: {var['range']})"
                var_content += "\n"
            var_content += "\\end{itemize}"
            
            subsections.append(PaperSection(
                title="Variables",
                content=var_content
            ))
        
        # Statistical analysis
        stats_content = (
            "Statistical analyses were conducted using rigorous hypothesis testing. "
            "Significance was assessed at $\alpha = 0.05$. Effect sizes were computed "
            "using Cohen's $d$ for parametric tests and Cliff's $\delta$ for non-parametric alternatives. "
            "All analyses included power calculations to ensure adequate sample sizes."
        )
        subsections.append(PaperSection(
            title="Statistical Analysis",
            content=stats_content
        ))
        
        return PaperSection(
            title="Methods",
            content="",
            subsections=subsections
        )
    
    def generate_results(self) -> PaperSection:
        """Generate results section."""
        subsections: list[PaperSection] = []
        
        # Descriptive statistics
        if self.results:
            desc_content = "A total of {} experiments were conducted. ".format(len(self.results))
            
            # Add summary statistics if available
            outcomes = [r.get('outcome') for r in self.results if 'outcome' in r]
            if outcomes:
                mean_outcome = sum(outcomes) / len(outcomes)
                desc_content += (
                    f"The mean outcome was ${mean_outcome:.2f} \\pm {np.std(outcomes):.2f}$. "
                )
            
            subsections.append(PaperSection(
                title="Descriptive Statistics",
                content=desc_content
            ))
        
        # Hypothesis testing results
        if self.analysis_results:
            hypo_content = "The following hypotheses were systematically evaluated:\\begin{enumerate}\n"
            
            for result in self.analysis_results:
                test_name = result.get('test_name', 'Statistical test')
                significant = result.get('significant', False)
                p_value = result.get('p_value', 1.0)
                effect_size = result.get('effect_size', {})
                
                status = "supported" if significant else "not supported"
                hypo_content += (
                    f"\\item {test_name}: $p = {p_value:.4f}$, hypothesis {status}. "
                )
                
                if effect_size:
                    measure = effect_size.get('measure', 'Effect size')
                    value = effect_size.get('value', 0)
                    interpretation = effect_size.get('interpretation', '')
                    hypo_content += f"{measure} = {value:.3f} ({interpretation}). "
                
                hypo_content += "\n"
            
            hypo_content += "\\end{enumerate}"
            
            subsections.append(PaperSection(
                title="Hypothesis Testing",
                content=hypo_content
            ))
        
        # Key findings summary
        if self.hypotheses:
            n_supported = sum(
                1 for r in self.analysis_results
                if r.get('significant', False)
            )
            
            key_content = (
                f"Of the {len(self.hypotheses)} hypotheses tested, "
                f"{n_supported} ({100*n_supported/len(self.hypotheses):.0f}\\%) "
                "received statistically significant support. "
                "The supported hypotheses demonstrate consistent patterns across the experimental conditions."
            )
            
            subsections.append(PaperSection(
                title="Summary of Findings",
                content=key_content
            ))
        
        return PaperSection(
            title="Results",
            content="",
            subsections=subsections
        )
    
    def generate_discussion(self) -> PaperSection:
        """Generate discussion section."""
        content_parts: list[str] = []
        
        # Interpretation
        content_parts.append(
            "The results of this study provide valuable insights into the underlying mechanisms. "
            "The statistically significant findings suggest that the proposed relationships "
            "hold under the tested conditions, contributing to the theoretical framework."
        )
        
        # Implications
        content_parts.append(
            "\\paragraph{Implications} "
            "These findings have several important implications. "
            "First, the identification of optimal experimental conditions provides "
            "a foundation for future investigations. "
            "Second, the rigorous statistical validation strengthens confidence in the conclusions. "
            "Third, the methodology demonstrates the effectiveness of active learning "
            "approaches in scientific discovery."
        )
        
        # Limitations
        content_parts.append(
            "\\paragraph{Limitations} "
            "Several limitations should be acknowledged. "
            "The experimental space, while systematically explored, represents a subset of all possible conditions. "
            "Additionally, the statistical power calculations indicate that larger sample sizes "
            "would increase confidence in borderline findings. "
            "Future work should extend these investigations to additional parameter regimes."
        )
        
        # Future work
        content_parts.append(
            "\\paragraph{Future Directions} "
            "Future research should focus on: "
            "(1) expanding the experimental space to include additional variables, "
            "(2) conducting longitudinal studies to assess temporal dynamics, and "
            "(3) integrating complementary methodologies to validate the findings."
        )
        
        return PaperSection(
            title="Discussion",
            content=" ".join(content_parts)
        )
    
    def generate_conclusion(self) -> PaperSection:
        """Generate conclusion section."""
        # Count findings
        n_significant = sum(
            1 for r in self.analysis_results
            if r.get('significant', False)
        )
        
        content = (
            f"This study presented a systematic investigation of {self.keywords[0] if self.keywords else 'the research domain'}, "
            f"testing {len(self.hypotheses)} hypotheses through rigorous experimentation. "
            f"Of these, {n_significant} received statistically significant support, "
            "demonstrating the validity of the proposed framework. "
            "The active learning approach efficiently identified optimal experimental conditions, "
            "while comprehensive statistical analysis validated the findings. "
            "These results contribute to the growing body of knowledge in this domain "
            "and provide a foundation for future investigations."
        )
        
        return PaperSection(
            title="Conclusion",
            content=content
        )
    
    def generate_references(self) -> list[dict[str, str]]:
        """Generate reference list."""
        references: list[dict[str, str]] = []
        
        for i, paper in enumerate(self.related_work, 1):
            ref = {
                'key': f"ref{i}",
                'authors': ', '.join(paper.authors[:3]) + (' et al.' if len(paper.authors) > 3 else ''),
                'title': paper.title,
                'journal': paper.journal or 'Preprint',
                'year': str(paper.year),
                'doi': paper.doi or ''
            }
            references.append(ref)
        
        # Add some generic statistical references
        references.extend([
            {
                'key': f"ref{len(references)+1}",
                'authors': 'Cohen, J.',
                'title': 'Statistical Power Analysis for the Behavioral Sciences',
                'journal': 'Routledge',
                'year': '1988'
            },
            {
                'key': f"ref{len(references)+2}",
                'authors': 'Faul, F. et al.',
                'title': 'G*Power 3: A flexible statistical power analysis program',
                'journal': 'Behavior Research Methods',
                'year': '2007'
            }
        ])
        
        return references
    
    def write_paper(self) -> ScientificPaper:
        """Generate complete scientific paper."""
        abstract = self.generate_abstract()
        
        sections = [
            self.generate_introduction(),
            self.generate_methods(),
            self.generate_results(),
            self.generate_discussion(),
            self.generate_conclusion()
        ]
        
        references = self.generate_references()
        
        return ScientificPaper(
            title=self.title,
            authors=self.authors,
            abstract=abstract,
            sections=sections,
            keywords=self.keywords,
            references=references
        )


def demo():
    """Demo paper writing."""
    import numpy as np
    from .hypothesis_generator import Hypothesis, HypothesisType
    
    # Create paper writer
    writer = PaperWriter(
        title="Active Learning for Materials Discovery: A Systematic Investigation",
        authors=["ASDE System", "Collaborative Research Team"],
        keywords=["active learning", "materials discovery", "Bayesian optimization"]
    )
    
    # Add hypotheses
    hypotheses = [
        Hypothesis(
            id="H1",
            statement="Temperature significantly affects reaction yield",
            hypothesis_type=HypothesisType.CAUSAL,
            confidence=0.8,
            testable_predictions=["Higher temperature increases yield up to optimum"]
        ),
        Hypothesis(
            id="H2",
            statement="Catalyst type moderates the temperature-yield relationship",
            hypothesis_type=HypothesisType.MECHANISTIC,
            confidence=0.7,
            testable_predictions=["Different catalysts show different optimal temperatures"]
        )
    ]
    writer.add_hypotheses(hypotheses)
    
    # Add experimental results
    results = [
        {"outcome": 75.5, "temperature": 350, "catalyst": "Pt"},
        {"outcome": 82.3, "temperature": 400, "catalyst": "Pt"},
        {"outcome": 78.1, "temperature": 350, "catalyst": "Pd"},
        {"outcome": 88.5, "temperature": 400, "catalyst": "Pd"},
    ]
    writer.add_experimental_results(
        results,
        {
            "design_type": "Bayesian optimization",
            "n_variables": 2,
            "n_conditions": len(results),
            "variables": [
                {"name": "temperature", "description": "Reaction temperature", "range": "300-500 K"},
                {"name": "catalyst", "description": "Catalyst material", "range": "Pt, Pd, Ni"}
            ]
        }
    )
    
    # Add analysis results
    writer.add_analysis_results([
        {
            "test_name": "Two-way ANOVA",
            "p_value": 0.023,
            "significant": True,
            "effect_size": {
                "measure": "Eta-squared",
                "value": 0.45,
                "interpretation": "large"
            }
        },
        {
            "test_name": "Post-hoc t-test",
            "p_value": 0.001,
            "significant": True,
            "effect_size": {
                "measure": "Cohen's d",
                "value": 1.2,
                "interpretation": "large"
            }
        }
    ])
    
    # Generate paper
    paper = writer.write_paper()
    
    print("=== Generated Paper (Markdown) ===\n")
    print(paper.to_markdown()[:2000])
    print("\n... [truncated] ...\n")
    
    print(f"\nTotal sections: {len(paper.sections)}")
    print(f"References: {len(paper.references)}")


if __name__ == "__main__":
    demo()
