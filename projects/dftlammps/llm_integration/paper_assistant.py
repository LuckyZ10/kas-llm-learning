"""
Paper Assistant Module

Provides comprehensive assistance for academic paper writing including:
- Section writing and refinement
- Literature review support
- Citation management
- Language polishing
- Reviewer response generation
- Structure recommendations

Supports multiple writing styles and academic disciplines.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime
from collections import defaultdict
import asyncio

from .llm_interface import (
    UnifiedLLMInterface,
    LLMConfig,
    LLMProvider,
    CompletionResponse,
    Message,
    Conversation,
)


class WritingStyle(Enum):
    """Academic writing styles."""
    FORMAL_ACADEMIC = auto()      # Standard journal style
    CONCISE_TECHNICAL = auto()    # Brief communications
    REVIEW_ARTICLE = auto()       # Comprehensive reviews
    PERSPECTIVE = auto()          # Opinion/commentary
    EDUCATIONAL = auto()          # Tutorial-style
    INTERDISCIPLINARY = auto()    # For broad audiences


class PaperSection(Enum):
    """Standard paper sections."""
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    METHODS = "methods"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    ACKNOWLEDGMENTS = "acknowledgments"
    REFERENCES = "references"
    SUPPLEMENTARY = "supplementary"


class CitationStyle(Enum):
    """Citation formatting styles."""
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    IEEE = "ieee"
    NATURE = "nature"
    SCIENCE = "science"
    ACS = "acs"
    APS = "aps"


@dataclass
class Citation:
    """A single citation entry."""
    id: str
    title: str
    authors: List[str]
    year: int
    journal: Optional[str] = None
    volume: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    abstract: Optional[str] = None
    cited_times: int = 0
    relevance_score: float = 0.5
    
    def format(self, style: CitationStyle) -> str:
        """Format citation according to style."""
        authors_str = self._format_authors(style)
        
        formatters = {
            CitationStyle.NATURE: lambda: f"{authors_str} {self.title}. {self.journal} {self.volume}, {self.pages} ({self.year}).",
            CitationStyle.SCIENCE: lambda: f"{authors_str}, {self.title}, {self.journal} {self.volume}, {self.pages} ({self.year}).",
            CitationStyle.IEEE: lambda: f"{authors_str}, \"{self.title},\" {self.journal}, vol. {self.volume}, pp. {self.pages}, {self.year}.",
            CitationStyle.ACS: lambda: f"{authors_str} {self.title}. {self.journal} {self.year}, {self.volume}, {self.pages}.",
            CitationStyle.APS: lambda: f"{authors_str}, {self.title}, {self.journal} {self.volume}, {self.pages} ({self.year}).",
        }
        
        formatter = formatters.get(style, formatters[CitationStyle.NATURE])
        return formatter()
    
    def _format_authors(self, style: CitationStyle) -> str:
        """Format author list."""
        if not self.authors:
            return "Unknown"
        
        if len(self.authors) == 1:
            return self.authors[0]
        elif len(self.authors) == 2:
            return f"{self.authors[0]} and {self.authors[1]}"
        elif len(self.authors) <= 6:
            return ", ".join(self.authors[:-1]) + ", and " + self.authors[-1]
        else:
            if style in [CitationStyle.NATURE, CitationStyle.SCIENCE]:
                return f"{self.authors[0]} et al."
            else:
                return ", ".join(self.authors[:3]) + ", et al."
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "journal": self.journal,
            "volume": self.volume,
            "pages": self.pages,
            "doi": self.doi,
            "url": self.url,
            "abstract": self.abstract,
            "relevance_score": self.relevance_score,
        }


@dataclass
class SectionDraft:
    """A draft of a paper section."""
    section: PaperSection
    content: str
    word_count: int = 0
    key_points: List[str] = field(default_factory=list)
    citations_needed: List[str] = field(default_factory=list)
    revision_history: List[Tuple[str, str]] = field(default_factory=list)  # (timestamp, note)
    
    def __post_init__(self):
        if not self.word_count:
            self.word_count = len(self.content.split())
    
    def add_revision(self, note: str) -> None:
        """Record a revision."""
        timestamp = datetime.now().isoformat()
        self.revision_history.append((timestamp, note))


@dataclass
class ReviewResult:
    """Results of paper review."""
    overall_score: float  # 1-10
    section_scores: Dict[str, float]
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    priority_revisions: List[str]
    estimated_impact: str
    
    def to_markdown(self) -> str:
        """Convert to markdown report."""
        lines = [
            "# Paper Review Report",
            "",
            f"**Overall Score:** {self.overall_score}/10",
            f"**Estimated Impact:** {self.estimated_impact}",
            "",
            "## Section Scores",
        ]
        
        for section, score in self.section_scores.items():
            lines.append(f"- {section}: {score}/10")
        
        lines.extend(["", "## Strengths"])
        for s in self.strengths:
            lines.append(f"- {s}")
        
        lines.extend(["", "## Weaknesses"])
        for w in self.weaknesses:
            lines.append(f"- {w}")
        
        lines.extend(["", "## Priority Revisions"])
        for r in self.priority_revisions:
            lines.append(f"1. {r}")
        
        lines.extend(["", "## Additional Suggestions"])
        for s in self.suggestions:
            lines.append(f"- {s}")
        
        return "\n".join(lines)


@dataclass
class PaperStructure:
    """Structure recommendations for a paper."""
    sections: List[PaperSection]
    estimated_word_counts: Dict[PaperSection, int]
    recommended_order: List[PaperSection]
    key_elements: Dict[PaperSection, List[str]]
    
    def to_outline(self) -> str:
        """Generate paper outline."""
        lines = ["# Paper Outline", ""]
        
        for section in self.recommended_order:
            word_count = self.estimated_word_counts.get(section, 0)
            lines.append(f"## {section.value.title()} (~{word_count} words)")
            
            for element in self.key_elements.get(section, []):
                lines.append(f"- {element}")
            lines.append("")
        
        return "\n".join(lines)


class CitationManager:
    """Manage citations and bibliography."""
    
    def __init__(self, style: CitationStyle = CitationStyle.NATURE):
        """Initialize citation manager.
        
        Args:
            style: Default citation style
        """
        self.style = style
        self.citations: Dict[str, Citation] = {}
        self.citation_counter: Dict[str, int] = defaultdict(int)
        self.cited_in_section: Dict[str, List[str]] = defaultdict(list)
    
    def add_citation(self, citation: Citation) -> str:
        """Add a citation to the database.
        
        Args:
            citation: Citation to add
            
        Returns:
            Citation ID
        """
        self.citations[citation.id] = citation
        return citation.id
    
    def cite(self, citation_id: str, section: Optional[str] = None) -> str:
        """Generate citation text.
        
        Args:
            citation_id: ID of citation
            section: Section where citation appears
            
        Returns:
            Formatted citation
        """
        if citation_id not in self.citations:
            return f"[CITATION NOT FOUND: {citation_id}]"
        
        self.citation_counter[citation_id] += 1
        
        if section:
            self.cited_in_section[section].append(citation_id)
        
        # Return citation marker (number or author-year)
        if self.style in [CitationStyle.NATURE, CitationStyle.SCIENCE, CitationStyle.IEEE]:
            # Find index in sorted citations
            sorted_ids = sorted(self.citations.keys())
            index = sorted_ids.index(citation_id) + 1
            return f"[{index}]"
        else:
            cite = self.citations[citation_id]
            return f"({cite.authors[0].split()[-1]}, {cite.year})"
    
    def generate_bibliography(self) -> str:
        """Generate bibliography section."""
        lines = ["# References", ""]
        
        # Sort by citation count (most cited first) or by order of appearance
        sorted_citations = sorted(
            self.citations.values(),
            key=lambda c: (-self.citation_counter[c.id], c.id)
        )
        
        for i, citation in enumerate(sorted_citations, 1):
            if self.style in [CitationStyle.NATURE, CitationStyle.SCIENCE]:
                lines.append(f"{i}. {citation.format(self.style)}")
            else:
                lines.append(citation.format(self.style))
        
        return "\n\n".join(lines)
    
    async def find_related_papers(
        self,
        llm: UnifiedLLMInterface,
        topic: str,
        num_papers: int = 5,
    ) -> List[Citation]:
        """Generate representative citations for a topic.
        
        Args:
            llm: LLM interface
            topic: Research topic
            num_papers: Number of papers to suggest
            
        Returns:
            List of suggested citations
        """
        prompt = f"""Suggest {num_papers} important papers related to: {topic}

For each paper, provide:
- Title
- Authors (2-3 main authors)
- Year (realistic recent year)
- Journal (appropriate for the field)
- Brief description of contribution

These should represent key works that would be cited in this area.
Format as JSON array."""
        
        schema = {
            "papers": [
                {
                    "title": "string",
                    "authors": ["string"],
                    "year": "integer",
                    "journal": "string",
                    "contribution": "string",
                }
            ]
        }
        
        result = await llm.generate_structured(prompt, schema, temperature=0.5)
        
        citations = []
        for i, p in enumerate(result.get("papers", [])):
            citation = Citation(
                id=f"ref_{i+1}",
                title=p.get("title", ""),
                authors=p.get("authors", []),
                year=p.get("year", datetime.now().year),
                journal=p.get("journal"),
                relevance_score=0.8,
            )
            citations.append(citation)
        
        return citations
    
    def check_citation_balance(self) -> Dict[str, Any]:
        """Analyze citation usage patterns.
        
        Returns:
            Analysis results
        """
        total_citations = sum(self.citation_counter.values())
        unique_citations = len([c for c in self.citation_counter.values() if c > 0])
        
        over_cited = [
            (cite_id, count) for cite_id, count in self.citation_counter.items()
            if count > total_citations * 0.2  # More than 20% of citations
        ]
        
        under_cited = [
            cite_id for cite_id, count in self.citation_counter.items()
            if count == 1
        ]
        
        return {
            "total_citation_instances": total_citations,
            "unique_citations_cited": unique_citations,
            "average_citations_per_paper": total_citations / max(unique_citations, 1),
            "over_cited_papers": over_cited,
            "single_use_citations": under_cited,
            "recommendation": (
                "Consider diversifying citations" if len(over_cited) > 2 
                else "Citation distribution looks balanced"
            ),
        }


class PaperAssistant:
    """Main paper writing assistant."""
    
    # Section-specific prompts
    SECTION_PROMPTS = {
        PaperSection.ABSTRACT: """Write a compelling abstract that:
1. States the problem clearly
2. Describes the methods used
3. Summarizes key results with specific numbers
4. States main conclusions and implications
5. Is 150-250 words

Use clear, accessible language while maintaining scientific precision.""",
        
        PaperSection.INTRODUCTION: """Write an introduction that:
1. Opens with the broad context and importance
2. Reviews relevant prior work (with citations)
3. Identifies the gap or problem addressed
4. States the specific objectives/hypotheses
5. Ends with a roadmap of the paper

Build a logical narrative leading to your contribution.""",
        
        PaperSection.METHODS: """Write a methods section that:
1. Provides sufficient detail for reproduction
2. Justifies methodological choices
3. Cites established methods appropriately
4. Includes relevant computational details
5. Describes validation approaches

Balance completeness with readability.""",
        
        PaperSection.RESULTS: """Write a results section that:
1. Presents findings objectively
2. Uses figures and tables effectively
3. Highlights significant trends and patterns
4. Includes statistical significance where applicable
5. Avoids interpretation (save for discussion)

Present results in a logical sequence.""",
        
        PaperSection.DISCUSSION: """Write a discussion that:
1. Interprets results in context of hypotheses
2. Compares with previous work
3. Explains mechanisms and implications
4. Acknowledges limitations
5. Suggests future directions

Provide insight, not just repetition of results.""",
        
        PaperSection.CONCLUSION: """Write a conclusion that:
1. Summarizes main findings
2. Restates significance
3. Provides take-home message
4. Suggests broader implications
5. Is concise and impactful

Avoid introducing new information.""",
    }
    
    def __init__(
        self,
        llm_interface: Optional[UnifiedLLMInterface] = None,
        writing_style: WritingStyle = WritingStyle.FORMAL_ACADEMIC,
        citation_style: CitationStyle = CitationStyle.NATURE,
    ):
        """Initialize paper assistant.
        
        Args:
            llm_interface: LLM interface
            writing_style: Default writing style
            citation_style: Citation style
        """
        self.llm = llm_interface or self._create_default_llm()
        self.writing_style = writing_style
        self.citation_manager = CitationManager(citation_style)
        self.drafts: Dict[PaperSection, SectionDraft] = {}
        self.conversation = Conversation()
    
    def _create_default_llm(self) -> UnifiedLLMInterface:
        """Create default LLM interface."""
        try:
            config = LLMConfig.from_env(LLMProvider.OPENAI)
        except ValueError:
            for provider in [LLMProvider.ANTHROPIC, LLMProvider.DEEPSEEK]:
                try:
                    config = LLMConfig.from_env(provider)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError("No LLM API keys found")
        
        return UnifiedLLMInterface(config)
    
    async def plan_structure(
        self,
        title: str,
        research_type: str,
        key_findings: List[str],
        target_journal: Optional[str] = None,
    ) -> PaperStructure:
        """Plan paper structure.
        
        Args:
            title: Paper title
            research_type: Type of research
            key_findings: Main findings
            target_journal: Target journal
            
        Returns:
            Recommended structure
        """
        prompt = f"""Plan the structure for a paper titled: {title}

Research Type: {research_type}
Key Findings:
{chr(10).join(f"- {f}" for f in key_findings)}

Target Journal: {target_journal or "General scientific journal"}

Recommend:
1. Which sections to include
2. Approximate word count for each section
3. Order of sections
4. Key elements each section should contain
5. Special considerations for this type of paper

Format as JSON."""
        
        schema = {
            "sections": ["string"],
            "word_counts": {},
            "section_order": ["string"],
            "key_elements": {},
            "special_considerations": ["string"],
        }
        
        result = await self.llm.generate_structured(prompt, schema, temperature=0.3)
        
        sections = []
        for s in result.get("sections", []):
            try:
                sections.append(PaperSection(s.lower()))
            except ValueError:
                continue
        
        word_counts = {}
        for section_str, count in result.get("word_counts", {}).items():
            try:
                section = PaperSection(section_str.lower())
                word_counts[section] = count
            except ValueError:
                continue
        
        return PaperStructure(
            sections=sections,
            estimated_word_counts=word_counts,
            recommended_order=sections,
            key_elements=result.get("key_elements", {}),
        )
    
    async def write_section(
        self,
        section: PaperSection,
        key_points: List[str],
        data_summary: Optional[str] = None,
        target_word_count: Optional[int] = None,
        context: Optional[str] = None,
    ) -> SectionDraft:
        """Write or draft a paper section.
        
        Args:
            section: Section to write
            key_points: Key points to include
            data_summary: Summary of data/results
            target_word_count: Target word count
            context: Additional context
            
        Returns:
            Section draft
        """
        section_guidance = self.SECTION_PROMPTS.get(section, "")
        
        prompt = f"""Write the {section.value} section of a scientific paper.

Guidance:
{section_guidance}

Key Points to Include:
{chr(10).join(f"- {p}" for p in key_points)}

{data_summary if data_summary else ""}

{context if context else ""}

Target Length: {target_word_count or "appropriate for section type"} words

Writing Style: {self.writing_style.name.replace('_', ' ').title()}

Write in a clear, professional academic style. Use active voice where appropriate."""
        
        response = await self.llm.complete(
            prompt=prompt,
            temperature=0.7,
            max_tokens=2000 if target_word_count and target_word_count > 500 else 1500,
        )
        
        draft = SectionDraft(
            section=section,
            content=response.content,
            key_points=key_points,
        )
        
        self.drafts[section] = draft
        return draft
    
    async def polish_text(
        self,
        text: str,
        polish_type: str = "general",
        target_audience: str = "expert",
    ) -> str:
        """Polish and improve text.
        
        Args:
            text: Text to polish
            polish_type: Type of polishing
            target_audience: Target audience level
            
        Returns:
            Polished text
        """
        polish_instructions = {
            "general": "Improve clarity, flow, and academic tone while preserving meaning.",
            "conciseness": "Make more concise without losing important information. Eliminate redundancy.",
            "clarity": "Improve clarity and readability. Simplify complex sentences. Define jargon.",
            "impact": "Strengthen the writing. Make arguments more compelling. Improve emphasis.",
            "grammar": "Fix grammar, punctuation, and style issues only. Do not change meaning.",
            "style": f"Adapt for {target_audience} audience. Adjust tone and terminology accordingly.",
        }
        
        instruction = polish_instructions.get(polish_type, polish_instructions["general"])
        
        prompt = f"""Polish the following academic text.

Instruction: {instruction}

Target Audience: {target_audience}

Original Text:
{text}

Provide the polished version only, without explanations."""
        
        response = await self.llm.complete(prompt, temperature=0.3)
        return response.content
    
    async def review_paper(
        self,
        sections: Dict[PaperSection, str],
        paper_type: str = "research_article",
    ) -> ReviewResult:
        """Review a complete paper or sections.
        
        Args:
            sections: Paper sections
            paper_type: Type of paper
            
        Returns:
            Review results
        """
        sections_text = "\n\n".join([
            f"## {s.value.upper()}\n\n{content}"
            for s, content in sections.items()
        ])
        
        prompt = f"""Review the following {paper_type} as an expert reviewer would.

{sections_text}

Provide a comprehensive review including:
1. Overall assessment and score (1-10)
2. Section-by-section scores
3. Main strengths
4. Main weaknesses
5. Specific suggestions for improvement
6. Priority revisions needed
7. Estimated impact/publishability

Be constructive but critical. Format as JSON."""
        
        schema = {
            "overall_score": "number 1-10",
            "section_scores": {},
            "strengths": ["string"],
            "weaknesses": ["string"],
            "suggestions": ["string"],
            "priority_revisions": ["string"],
            "estimated_impact": "string",
        }
        
        result = await self.llm.generate_structured(prompt, schema, temperature=0.3)
        
        return ReviewResult(
            overall_score=result.get("overall_score", 5),
            section_scores=result.get("section_scores", {}),
            strengths=result.get("strengths", []),
            weaknesses=result.get("weaknesses", []),
            suggestions=result.get("suggestions", []),
            priority_revisions=result.get("priority_revisions", []),
            estimated_impact=result.get("estimated_impact", "unknown"),
        )
    
    async def generate_reviewer_response(
        self,
        review_comments: List[str],
        paper_sections: Dict[PaperSection, str],
        tone: str = "professional",
    ) -> str:
        """Generate response to reviewer comments.
        
        Args:
            review_comments: Reviewer comments
            paper_sections: Current paper sections
            tone: Response tone
            
        Returns:
            Response letter draft
        """
        comments_text = "\n".join([
            f"{i+1}. {comment}" for i, comment in enumerate(review_comments)
        ])
        
        prompt = f"""Generate a response to the following reviewer comments.

Reviewer Comments:
{comments_text}

Tone: {tone}

For each comment:
1. Acknowledge the point professionally
2. Explain what changes were made (or why not)
3. Reference specific changes in the manuscript
4. Thank the reviewer for the suggestion

Format as a formal response letter with point-by-point replies."""
        
        response = await self.llm.complete(prompt, temperature=0.5)
        return response.content
    
    async def suggest_improvements(
        self,
        section: PaperSection,
        current_text: str,
        improvement_type: str = "general",
    ) -> List[str]:
        """Suggest improvements for a section.
        
        Args:
            section: Section type
            current_text: Current text
            improvement_type: Type of improvements
            
        Returns:
            List of suggestions
        """
        prompt = f"""Review the following {section.value} section and suggest improvements.

Current Text:
{current_text}

Focus on: {improvement_type}

Provide specific, actionable suggestions for improvement. Format as JSON array."""
        
        schema = {
            "suggestions": [
                {
                    "issue": "string - what's wrong or could be better",
                    "suggestion": "string - how to fix it",
                    "priority": "string - high/medium/low",
                }
            ]
        }
        
        result = await self.llm.generate_structured(prompt, schema, temperature=0.4)
        
        suggestions = []
        for s in result.get("suggestions", []):
            suggestions.append(f"[{s.get('priority', 'medium').upper()}] {s.get('issue', '')}: {s.get('suggestion', '')}")
        
        return suggestions
    
    async def generate_abstract_from_sections(
        self,
        sections: Dict[PaperSection, str],
        max_words: int = 250,
    ) -> str:
        """Generate abstract from paper sections.
        
        Args:
            sections: Paper sections
            max_words: Maximum words
            
        Returns:
            Generated abstract
        """
        # Extract key content from sections
        content_summary = []
        
        if PaperSection.INTRODUCTION in sections:
            content_summary.append(f"Background:\n{sections[PaperSection.INTRODUCTION][:500]}...")
        
        if PaperSection.METHODS in sections:
            content_summary.append(f"Methods:\n{sections[PaperSection.METHODS][:300]}...")
        
        if PaperSection.RESULTS in sections:
            content_summary.append(f"Results:\n{sections[PaperSection.RESULTS][:500]}...")
        
        if PaperSection.CONCLUSION in sections:
            content_summary.append(f"Conclusions:\n{sections[PaperSection.CONCLUSION][:300]}...")
        
        prompt = f"""Write an abstract based on the following paper content.

{chr(10).join(content_summary)}

Requirements:
- Maximum {max_words} words
- Include: background, methods, key results, conclusions
- Make it compelling and informative
- Use specific numbers where available

Write only the abstract text."""
        
        response = await self.llm.complete(prompt, temperature=0.5, max_tokens=500)
        return response.content
    
    def compile_paper(
        self,
        sections: Optional[Dict[PaperSection, str]] = None,
        include_bibliography: bool = True,
    ) -> str:
        """Compile sections into complete paper.
        
        Args:
            sections: Sections to compile (uses drafts if not provided)
            include_bibliography: Whether to include bibliography
            
        Returns:
            Complete paper text
        """
        if sections is None:
            sections = {s: d.content for s, d in self.drafts.items()}
        
        ordered_sections = [
            PaperSection.ABSTRACT,
            PaperSection.INTRODUCTION,
            PaperSection.METHODS,
            PaperSection.RESULTS,
            PaperSection.DISCUSSION,
            PaperSection.CONCLUSION,
            PaperSection.ACKNOWLEDGMENTS,
        ]
        
        parts = []
        for section in ordered_sections:
            if section in sections:
                parts.append(f"# {section.value.title()}\n\n{sections[section]}")
        
        if include_bibliography:
            parts.append(self.citation_manager.generate_bibliography())
        
        return "\n\n---\n\n".join(parts)
