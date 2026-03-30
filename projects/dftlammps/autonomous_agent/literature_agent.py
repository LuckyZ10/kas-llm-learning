"""
文献智能体模块
==============
实现自动文献检索、知识提取与整合、研究空白识别功能。
"""

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict
import hashlib


class LiteratureSource(Enum):
    """文献来源枚举"""
    PUBMED = "pubmed"
    ARXIV = "arxiv"
    GOOGLE_SCHOLAR = "google_scholar"
    WEB_OF_SCIENCE = "web_of_science"
    SCOPUS = "scopus"
    IEEE = "ieee"
    CHEMRXIV = "chemrxiv"
    MATERIALS_PROJECT = "materials_project"
    AFLOW = "aflow"
    OQMD = "oqmd"


class KnowledgeType(Enum):
    """知识类型枚举"""
    METHOD = "method"
    MATERIAL = "material"
    PROPERTY = "property"
    THEORY = "theory"
    EXPERIMENTAL = "experimental"
    COMPUTATIONAL = "computational"


@dataclass
class Paper:
    """论文数据类"""
    id: str = field(default_factory=lambda: f"paper_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}")
    title: str = ""
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    journal: str = ""
    year: int = 0
    doi: str = ""
    url: str = ""
    citations: int = 0
    keywords: List[str] = field(default_factory=list)
    source: LiteratureSource = LiteratureSource.ARXIV
    full_text: Optional[str] = None
    sections: Dict[str, str] = field(default_factory=dict)  # 章节内容
    figures: List[Dict] = field(default_factory=list)
    tables: List[Dict] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    retrieved_at: datetime = field(default_factory=datetime.now)
    relevance_score: float = 0.0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract[:200] + "..." if len(self.abstract) > 200 else self.abstract,
            "journal": self.journal,
            "year": self.year,
            "doi": self.doi,
            "citations": self.citations,
            "keywords": self.keywords,
            "relevance_score": self.relevance_score
        }


@dataclass
class KnowledgeUnit:
    """知识单元"""
    id: str = field(default_factory=lambda: f"ku_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}")
    content: str = ""
    knowledge_type: KnowledgeType = KnowledgeType.THEORY
    source_paper_id: str = ""
    source_section: str = ""
    confidence: float = 0.8
    extracted_at: datetime = field(default_factory=datetime.now)
    related_units: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    context: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.knowledge_type.value,
            "content": self.content[:100] + "..." if len(self.content) > 100 else self.content,
            "confidence": self.confidence,
            "entities": self.entities
        }


@dataclass
class ResearchGap:
    """研究空白"""
    id: str = field(default_factory=lambda: f"gap_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}")
    description: str = ""
    category: str = ""  # "knowledge", "methodology", "material", "application"
    importance: float = 0.5
    difficulty: float = 0.5
    related_papers: List[str] = field(default_factory=list)
    potential_impact: str = ""
    suggested_approaches: List[str] = field(default_factory=list)
    identified_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "description": self.description[:100] + "...",
            "category": self.category,
            "importance": self.importance,
            "difficulty": self.difficulty
        }


@dataclass
class ResearchTrend:
    """研究趋势"""
    topic: str = ""
    start_year: int = 0
    end_year: int = 0
    paper_counts: Dict[int, int] = field(default_factory=dict)
    growth_rate: float = 0.0
    hot_keywords: List[str] = field(default_factory=list)
    leading_authors: List[str] = field(default_factory=list)
    key_papers: List[str] = field(default_factory=list)


class BaseLiteratureSource(ABC):
    """文献源基类"""
    
    def __init__(self, source_type: LiteratureSource):
        self.source_type = source_type
        self.rate_limit_delay = 1.0  # 秒
        
    @abstractmethod
    async def search(self, query: str, filters: Optional[Dict] = None, max_results: int = 10) -> List[Paper]:
        """搜索文献"""
        pass
    
    @abstractmethod
    async def fetch_full_text(self, paper: Paper) -> str:
        """获取全文"""
        pass
    
    def _calculate_relevance(self, paper: Paper, query: str) -> float:
        """计算相关性分数"""
        score = 0.0
        query_terms = set(query.lower().split())
        
        # 标题匹配
        title_terms = set(paper.title.lower().split())
        title_match = len(query_terms & title_terms)
        score += title_match * 0.4
        
        # 摘要匹配
        abstract_terms = set(paper.abstract.lower().split())
        abstract_match = len(query_terms & abstract_terms)
        score += abstract_match * 0.1
        
        # 关键词匹配
        keyword_match = sum(1 for kw in paper.keywords if any(q in kw.lower() for q in query_terms))
        score += keyword_match * 0.3
        
        # 引用次数影响
        if paper.citations > 0:
            score += min(paper.citations / 1000, 0.2)
        
        # 年份影响（较新的文章权重稍高）
        if paper.year >= datetime.now().year - 5:
            score += 0.1
        
        return min(score, 1.0)


class SimulatedLiteratureSource(BaseLiteratureSource):
    """模拟文献源（用于测试）"""
    
    def __init__(self):
        super().__init__(LiteratureSource.ARXIV)
        self._mock_papers: List[Paper] = []
        self._generate_mock_data()
        
    def _generate_mock_data(self):
        """生成模拟数据"""
        mock_data = [
            {
                "title": "High-Entropy Alloys: A Critical Review",
                "authors": ["Y. Zhang", "T.T. Zuo", "Z. Tang"],
                "abstract": "High-entropy alloys (HEAs) have attracted significant attention due to their unique properties...",
                "journal": "Progress in Materials Science",
                "year": 2023,
                "keywords": ["high-entropy alloys", "mechanical properties", "microstructure"],
                "citations": 156
            },
            {
                "title": "Machine Learning for Catalyst Design",
                "authors": ["J. Smith", "A. Johnson", "K. Lee"],
                "abstract": "Recent advances in machine learning have enabled rapid screening of catalyst materials...",
                "journal": "Nature Catalysis",
                "year": 2024,
                "keywords": ["machine learning", "catalysis", "materials design"],
                "citations": 89
            },
            {
                "title": "Two-Dimensional Materials for Energy Storage",
                "authors": ["R. Wang", "L. Chen", "M. Liu"],
                "abstract": "Two-dimensional materials offer unique advantages for energy storage applications...",
                "journal": "Advanced Energy Materials",
                "year": 2023,
                "keywords": ["2D materials", "batteries", "supercapacitors"],
                "citations": 234
            },
            {
                "title": "DFT Calculations of Perovskite Oxides",
                "authors": ["P. Brown", "S. Davis", "N. Wilson"],
                "abstract": "Density functional theory calculations reveal the electronic structure of perovskite oxides...",
                "journal": "Physical Review B",
                "year": 2022,
                "keywords": ["DFT", "perovskite", "electronic structure"],
                "citations": 178
            },
            {
                "title": "Single-Atom Catalysts: Synthesis and Applications",
                "authors": ["H. Zhang", "Y. Li", "X. Wang"],
                "abstract": "Single-atom catalysts represent a new frontier in heterogeneous catalysis...",
                "journal": "Chemical Reviews",
                "year": 2024,
                "keywords": ["single-atom catalysis", "synthesis", "characterization"],
                "citations": 312
            }
        ]
        
        for data in mock_data:
            paper = Paper(
                title=data["title"],
                authors=data["authors"],
                abstract=data["abstract"],
                journal=data["journal"],
                year=data["year"],
                keywords=data["keywords"],
                citations=data["citations"]
            )
            self._mock_papers.append(paper)
    
    async def search(self, query: str, filters: Optional[Dict] = None, max_results: int = 10) -> List[Paper]:
        """模拟搜索"""
        await asyncio.sleep(0.5)  # 模拟网络延迟
        
        # 简单匹配
        query_lower = query.lower()
        results = []
        
        for paper in self._mock_papers:
            match_score = 0
            
            # 标题匹配
            if any(term in paper.title.lower() for term in query_lower.split()):
                match_score += 0.5
            
            # 摘要匹配
            if any(term in paper.abstract.lower() for term in query_lower.split()):
                match_score += 0.3
            
            # 关键词匹配
            for kw in paper.keywords:
                if any(term in kw.lower() for term in query_lower.split()):
                    match_score += 0.2
            
            if match_score > 0:
                paper.relevance_score = match_score
                results.append(paper)
        
        # 排序并限制结果
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:max_results]
    
    async def fetch_full_text(self, paper: Paper) -> str:
        """模拟获取全文"""
        await asyncio.sleep(0.3)
        
        # 生成模拟全文
        sections = {
            "introduction": f"This paper presents {paper.title.lower()}. Recent developments in this field...",
            "methods": "Density functional theory calculations were performed using VASP...",
            "results": "The results show significant improvements in performance...",
            "discussion": "These findings suggest important implications for future research...",
            "conclusion": "In conclusion, we have demonstrated..."
        }
        
        paper.sections = sections
        return "\n\n".join(sections.values())


class KnowledgeExtractor:
    """知识提取器"""
    
    def __init__(self):
        self.extraction_rules: Dict[KnowledgeType, List[Callable]] = defaultdict(list)
        self._setup_default_rules()
        
    def _setup_default_rules(self):
        """设置默认提取规则"""
        # 方法提取规则
        self.extraction_rules[KnowledgeType.METHOD].append(self._extract_methods)
        
        # 材料提取规则
        self.extraction_rules[KnowledgeType.MATERIAL].append(self._extract_materials)
        
        # 性质提取规则
        self.extraction_rules[KnowledgeType.PROPERTY].append(self._extract_properties)
        
        # 理论提取规则
        self.extraction_rules[KnowledgeType.THEORY].append(self._extract_theories)
    
    def add_extraction_rule(self, knowledge_type: KnowledgeType, rule: Callable):
        """添加提取规则"""
        self.extraction_rules[knowledge_type].append(rule)
    
    async def extract_from_paper(self, paper: Paper) -> List[KnowledgeUnit]:
        """
        从论文中提取知识
        
        分析论文内容，提取结构化知识单元。
        """
        knowledge_units = []
        
        # 确保有全文
        if not paper.sections:
            return knowledge_units
        
        # 从各个章节提取
        for section_name, section_content in paper.sections.items():
            section_units = await self._extract_from_section(
                section_content, section_name, paper.id
            )
            knowledge_units.extend(section_units)
        
        # 去重和合并
        knowledge_units = self._merge_similar_units(knowledge_units)
        
        return knowledge_units
    
    async def _extract_from_section(self, content: str, section_name: str, 
                                     paper_id: str) -> List[KnowledgeUnit]:
        """从章节提取知识"""
        units = []
        
        for knowledge_type, rules in self.extraction_rules.items():
            for rule in rules:
                extracted = rule(content)
                for item in extracted:
                    unit = KnowledgeUnit(
                        content=item["content"],
                        knowledge_type=knowledge_type,
                        source_paper_id=paper_id,
                        source_section=section_name,
                        confidence=item.get("confidence", 0.8),
                        entities=item.get("entities", []),
                        context=item.get("context", "")
                    )
                    units.append(unit)
        
        return units
    
    def _extract_methods(self, text: str) -> List[Dict]:
        """提取方法"""
        methods = []
        
        # 模式匹配
        method_patterns = [
            r"(?:using|via|by|with)\s+([A-Z][a-z]+(?:\s+[a-z]+){0,5}\s+(?:method|approach|technique|algorithm))",
            r"(?:density functional theory|DFT|molecular dynamics|MD|machine learning|ML|neural network|NN)",
            r"(?:experiment|simulation|calculation)\s+(?:was|were|is)\s+(?:performed|conducted|carried out)"
        ]
        
        for pattern in method_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                methods.append({
                    "content": f"Method: {match.group()}",
                    "confidence": 0.75,
                    "entities": [match.group(1) if match.lastindex else match.group()]
                })
        
        return methods
    
    def _extract_materials(self, text: str) -> List[Dict]:
        """提取材料"""
        materials = []
        
        # 材料名称模式
        material_patterns = [
            r"\b([A-Z][a-z]?\d*[A-Z][a-z]?\d*)\b",  # 化学式
            r"\b(graphene|carbon nanotube|MOF|COF|perovskite|HEA|alloy)\b",
            r"\b(single-atom|nanoparticle|nanowire|nanosheet)\s+catalyst\b"
        ]
        
        for pattern in material_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                materials.append({
                    "content": f"Material: {match.group()}",
                    "confidence": 0.8,
                    "entities": [match.group()]
                })
        
        return materials
    
    def _extract_properties(self, text: str) -> List[Dict]:
        """提取性质"""
        properties = []
        
        # 性质描述模式
        property_patterns = [
            r"(\d+\.?\d*)\s*(eV|GPa|K|MPa|mAh/g|m²/g)",
            r"\b(band gap|conductivity|stability|activity|selectivity)\s+(?:of|is|was)\s+(\d+\.?\d*)",
            r"\b(high|excellent|superior|outstanding)\s+(\w+\s+performance|conductivity|activity)"
        ]
        
        for pattern in property_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                properties.append({
                    "content": f"Property: {match.group()}",
                    "confidence": 0.85,
                    "entities": [match.group()]
                })
        
        return properties
    
    def _extract_theories(self, text: str) -> List[Dict]:
        """提取理论"""
        theories = []
        
        # 理论概念模式
        theory_patterns = [
            r"\b(d-band center|free energy|reaction mechanism|kinetic barrier)\b",
            r"\b(density functional theory|Hartree-Fock|Monte Carlo|molecular dynamics)\b",
            r"\b(thermodynamic|kinetic|electronic|structural)\s+(?:model|theory|principle)"
        ]
        
        for pattern in theory_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                theories.append({
                    "content": f"Theory: {match.group()}",
                    "confidence": 0.8,
                    "entities": [match.group()]
                })
        
        return theories
    
    def _merge_similar_units(self, units: List[KnowledgeUnit]) -> List[KnowledgeUnit]:
        """合并相似的知识单元"""
        if not units:
            return units
        
        merged = []
        seen = set()
        
        for unit in units:
            # 简单的去重：基于内容哈希
            content_hash = hashlib.md5(unit.content.encode()).hexdigest()[:8]
            if content_hash not in seen:
                seen.add(content_hash)
                merged.append(unit)
        
        return merged


class KnowledgeIntegrator:
    """知识整合器"""
    
    def __init__(self):
        self.knowledge_graph: Dict[str, Set[str]] = defaultdict(set)
        self.entity_index: Dict[str, List[str]] = defaultdict(list)
        self.contradictions: List[Dict] = []
        
    async def integrate(self, knowledge_units: List[KnowledgeUnit]) -> Dict:
        """
        整合知识
        
        将多个知识单元整合为一致的知识体系。
        """
        # 1. 构建知识图谱
        await self._build_knowledge_graph(knowledge_units)
        
        # 2. 检测矛盾
        contradictions = await self._detect_contradictions(knowledge_units)
        
        # 3. 解决冲突
        resolved = await self._resolve_conflicts(contradictions)
        
        # 4. 生成综合知识
        synthesized = await self._synthesize_knowledge(knowledge_units)
        
        return {
            "total_units": len(knowledge_units),
            "integrated_units": len(synthesized),
            "contradictions_found": len(contradictions),
            "contradictions_resolved": len(resolved),
            "knowledge_graph_nodes": len(self.knowledge_graph),
            "synthesized_knowledge": synthesized
        }
    
    async def _build_knowledge_graph(self, units: List[KnowledgeUnit]):
        """构建知识图谱"""
        for unit in units:
            # 索引实体
            for entity in unit.entities:
                self.entity_index[entity].append(unit.id)
            
            # 建立单元间关系
            for other_unit in units:
                if unit.id != other_unit.id:
                    shared_entities = set(unit.entities) & set(other_unit.entities)
                    if shared_entities:
                        self.knowledge_graph[unit.id].add(other_unit.id)
    
    async def _detect_contradictions(self, units: List[KnowledgeUnit]) -> List[Dict]:
        """检测矛盾"""
        contradictions = []
        
        # 按类型分组
        type_groups = defaultdict(list)
        for unit in units:
            type_groups[unit.knowledge_type].append(unit)
        
        # 在每个类型内检测矛盾
        for ktype, group in type_groups.items():
            for i, unit1 in enumerate(group):
                for unit2 in group[i+1:]:
                    if self._are_contradictory(unit1, unit2):
                        contradictions.append({
                            "unit1": unit1.id,
                            "unit2": unit2.id,
                            "type": ktype.value,
                            "confidence": min(unit1.confidence, unit2.confidence)
                        })
        
        self.contradictions = contradictions
        return contradictions
    
    def _are_contradictory(self, unit1: KnowledgeUnit, unit2: KnowledgeUnit) -> bool:
        """判断两个单元是否矛盾"""
        # 简化的矛盾检测：检查是否有相反的关键词
        opposite_terms = {
            "high": "low",
            "increase": "decrease",
            "positive": "negative",
            "stable": "unstable"
        }
        
        text1 = unit1.content.lower()
        text2 = unit2.content.lower()
        
        for term, opposite in opposite_terms.items():
            if term in text1 and opposite in text2:
                return True
            if opposite in text1 and term in text2:
                return True
        
        return False
    
    async def _resolve_conflicts(self, contradictions: List[Dict]) -> List[Dict]:
        """解决冲突"""
        resolved = []
        
        for conflict in contradictions:
            # 基于置信度解决
            if conflict["confidence"] < 0.6:
                # 低置信度矛盾，标记为需要验证
                resolved.append({
                    **conflict,
                    "resolution": "needs_verification",
                    "action": "design_experiment"
                })
            else:
                # 高置信度矛盾，保留高置信度的知识
                resolved.append({
                    **conflict,
                    "resolution": "keep_higher_confidence"
                })
        
        return resolved
    
    async def _synthesize_knowledge(self, units: List[KnowledgeUnit]) -> List[Dict]:
        """综合知识"""
        synthesized = []
        
        # 按类型聚合
        type_groups = defaultdict(list)
        for unit in units:
            type_groups[unit.knowledge_type].append(unit)
        
        # 为每个类型生成综合知识
        for ktype, group in type_groups.items():
            if len(group) > 1:
                # 聚合相似知识
                synthesized.append({
                    "type": ktype.value,
                    "synthesized_content": f"Multiple studies report on {ktype.value}...",
                    "source_count": len(group),
                    "consensus_level": self._calculate_consensus(group),
                    "key_entities": list(set(e for u in group for e in u.entities))[:10]
                })
        
        return synthesized
    
    def _calculate_consensus(self, units: List[KnowledgeUnit]) -> float:
        """计算共识程度"""
        if len(units) < 2:
            return 1.0
        
        # 基于内容相似度
        total_sim = 0
        count = 0
        for i, u1 in enumerate(units):
            for u2 in units[i+1:]:
                sim = self._content_similarity(u1.content, u2.content)
                total_sim += sim
                count += 1
        
        return total_sim / count if count > 0 else 0.0
    
    def _content_similarity(self, text1: str, text2: str) -> float:
        """计算内容相似度"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union) if union else 0.0


class GapAnalyzer:
    """研究空白分析器"""
    
    def __init__(self):
        self.gap_patterns: List[Callable] = []
        self._setup_default_patterns()
        
    def _setup_default_patterns(self):
        """设置默认模式"""
        self.gap_patterns = [
            self._identify_knowledge_gaps,
            self._identify_methodology_gaps,
            self._identify_material_gaps,
            self._identify_application_gaps
        ]
    
    async def identify_gaps(self, papers: List[Paper], 
                           knowledge_units: List[KnowledgeUnit]) -> List[ResearchGap]:
        """
        识别研究空白
        
        基于文献分析识别知识、方法、材料等方面的研究空白。
        """
        gaps = []
        
        for pattern in self.gap_patterns:
            found_gaps = await pattern(papers, knowledge_units)
            gaps.extend(found_gaps)
        
        # 评估和排序
        for gap in gaps:
            gap.importance = self._assess_importance(gap)
            gap.difficulty = self._assess_difficulty(gap)
        
        gaps.sort(key=lambda x: x.importance / (x.difficulty + 0.1), reverse=True)
        
        return gaps
    
    async def _identify_knowledge_gaps(self, papers: List[Paper], 
                                        knowledge_units: List[KnowledgeUnit]) -> List[ResearchGap]:
        """识别知识空白"""
        gaps = []
        
        # 分析知识覆盖范围
        covered_topics = set()
        for unit in knowledge_units:
            covered_topics.update(unit.entities)
        
        # 常见材料科学主题
        common_topics = {
            "electronic structure", "mechanical properties", "thermal stability",
            "catalytic mechanism", "synthesis method", "defect chemistry"
        }
        
        missing_topics = common_topics - covered_topics
        
        for topic in list(missing_topics)[:3]:
            gaps.append(ResearchGap(
                description=f"Limited understanding of {topic} in studied materials",
                category="knowledge",
                importance=0.7,
                related_papers=[p.id for p in papers],
                potential_impact=f"Understanding {topic} could lead to breakthrough discoveries"
            ))
        
        return gaps
    
    async def _identify_methodology_gaps(self, papers: List[Paper], 
                                          knowledge_units: List[KnowledgeUnit]) -> List[ResearchGap]:
        """识别方法论空白"""
        gaps = []
        
        # 分析方法使用情况
        methods_used = set()
        for unit in knowledge_units:
            if unit.knowledge_type == KnowledgeType.METHOD:
                methods_used.update(unit.entities)
        
        # 新兴方法
        emerging_methods = {
            "active learning", "Bayesian optimization", "generative AI",
            "neural network potential", "on-the-fly learning"
        }
        
        underutilized = emerging_methods - methods_used
        
        for method in list(underutilized)[:2]:
            gaps.append(ResearchGap(
                description=f"{method} is underutilized in this research area",
                category="methodology",
                importance=0.8,
                related_papers=[p.id for p in papers],
                potential_impact=f"Applying {method} could accelerate discovery",
                suggested_approaches=[f"Integrate {method} into the research workflow"]
            ))
        
        return gaps
    
    async def _identify_material_gaps(self, papers: List[Paper], 
                                       knowledge_units: List[KnowledgeUnit]) -> List[ResearchGap]:
        """识别材料空白"""
        gaps = []
        
        # 分析研究的材料类型
        materials_studied = set()
        for unit in knowledge_units:
            if unit.knowledge_type == KnowledgeType.MATERIAL:
                materials_studied.update(unit.entities)
        
        # 如果研究集中在特定材料，提示探索新材料
        if len(materials_studied) < 5:
            gaps.append(ResearchGap(
                description="Limited diversity of materials studied",
                category="material",
                importance=0.6,
                related_papers=[p.id for p in papers],
                potential_impact="Exploring new material compositions may reveal better properties",
                suggested_approaches=["Systematic screening of related compounds"]
            ))
        
        return gaps
    
    async def _identify_application_gaps(self, papers: List[Paper], 
                                          knowledge_units: List[KnowledgeUnit]) -> List[ResearchGap]:
        """识别应用空白"""
        gaps = []
        
        # 基于应用关键词分析
        application_keywords = ["battery", "catalysis", "solar", "sensor", "device"]
        mentioned_applications = set()
        
        for paper in papers:
            for keyword in paper.keywords:
                if any(app in keyword.lower() for app in application_keywords):
                    mentioned_applications.add(keyword.lower())
        
        # 识别未被充分探索的应用
        unexplored = set(application_keywords) - mentioned_applications
        
        if unexplored:
            gaps.append(ResearchGap(
                description=f"Potential applications not fully explored: {', '.join(list(unexplored)[:3])}",
                category="application",
                importance=0.7,
                related_papers=[p.id for p in papers],
                potential_impact="New applications could expand the impact of this research"
            ))
        
        return gaps
    
    def _assess_importance(self, gap: ResearchGap) -> float:
        """评估重要性"""
        # 基于相关论文数量和影响描述
        base_score = min(len(gap.related_papers) / 10, 0.5)
        
        if "breakthrough" in gap.potential_impact.lower():
            base_score += 0.3
        if "accelerate" in gap.potential_impact.lower():
            base_score += 0.2
        
        return min(base_score, 1.0)
    
    def _assess_difficulty(self, gap: ResearchGap) -> float:
        """评估难度"""
        # 基于类别估算难度
        difficulty_map = {
            "knowledge": 0.7,
            "methodology": 0.6,
            "material": 0.5,
            "application": 0.4
        }
        return difficulty_map.get(gap.category, 0.5)


class LiteratureAgent:
    """
    文献智能体主类
    
    整合文献检索、知识提取、知识整合和研究空白识别功能。
    """
    
    def __init__(self):
        self.sources: Dict[LiteratureSource, BaseLiteratureSource] = {}
        self.knowledge_extractor = KnowledgeExtractor()
        self.knowledge_integrator = KnowledgeIntegrator()
        self.gap_analyzer = GapAnalyzer()
        
        self.paper_cache: Dict[str, Paper] = {}
        self.knowledge_base: List[KnowledgeUnit] = []
        self.research_gaps: List[ResearchGap] = []
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 添加默认源
        self._add_default_sources()
        
    def _add_default_sources(self):
        """添加默认文献源"""
        self.register_source(SimulatedLiteratureSource())
        
    def register_source(self, source: BaseLiteratureSource):
        """注册文献源"""
        self.sources[source.source_type] = source
        self.logger.info(f"Registered literature source: {source.source_type.value}")
    
    async def search_literature(self, 
                                query: str, 
                                sources: Optional[List[LiteratureSource]] = None,
                                filters: Optional[Dict] = None,
                                max_results: int = 20) -> List[Paper]:
        """
        搜索文献
        
        从多个来源搜索相关文献。
        """
        self.logger.info(f"Searching literature for: {query}")
        
        all_papers = []
        
        # 确定要搜索的源
        if sources is None:
            sources = list(self.sources.keys())
        
        # 并行搜索多个源
        tasks = []
        for source_type in sources:
            if source_type in self.sources:
                source = self.sources[source_type]
                task = source.search(query, filters, max_results // len(sources))
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_papers.extend(result)
        
        # 去重和排序
        seen_titles = set()
        unique_papers = []
        for paper in all_papers:
            if paper.title not in seen_titles:
                seen_titles.add(paper.title)
                unique_papers.append(paper)
        
        unique_papers.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # 缓存结果
        for paper in unique_papers:
            self.paper_cache[paper.id] = paper
        
        self.logger.info(f"Found {len(unique_papers)} unique papers")
        return unique_papers[:max_results]
    
    async def extract_knowledge(self, papers: Optional[List[Paper]] = None) -> List[KnowledgeUnit]:
        """
        提取知识
        
        从论文中提取结构化知识。
        """
        if papers is None:
            papers = list(self.paper_cache.values())
        
        self.logger.info(f"Extracting knowledge from {len(papers)} papers")
        
        all_units = []
        
        for paper in papers:
            # 获取全文
            if paper.source in self.sources:
                source = self.sources[paper.source]
                if not paper.sections:
                    try:
                        await source.fetch_full_text(paper)
                    except Exception as e:
                        self.logger.warning(f"Failed to fetch full text for {paper.id}: {e}")
                        continue
            
            # 提取知识
            try:
                units = await self.knowledge_extractor.extract_from_paper(paper)
                all_units.extend(units)
                self.logger.debug(f"Extracted {len(units)} units from {paper.id}")
            except Exception as e:
                self.logger.error(f"Failed to extract knowledge from {paper.id}: {e}")
        
        # 添加到知识库
        self.knowledge_base.extend(all_units)
        
        self.logger.info(f"Extracted {len(all_units)} knowledge units in total")
        return all_units
    
    async def integrate_knowledge(self) -> Dict:
        """
        整合知识
        
        整合知识库中的所有知识单元。
        """
        self.logger.info(f"Integrating {len(self.knowledge_base)} knowledge units")
        
        result = await self.knowledge_integrator.integrate(self.knowledge_base)
        
        self.logger.info(f"Integration complete: {result['integrated_units']} units integrated")
        return result
    
    async def identify_research_gaps(self) -> List[ResearchGap]:
        """
        识别研究空白
        
        基于文献和知识分析识别研究空白。
        """
        self.logger.info("Identifying research gaps")
        
        papers = list(self.paper_cache.values())
        gaps = await self.gap_analyzer.identify_gaps(papers, self.knowledge_base)
        
        self.research_gaps.extend(gaps)
        
        self.logger.info(f"Identified {len(gaps)} research gaps")
        return gaps
    
    async def analyze_research_trends(self, topic: str, 
                                       years: int = 5) -> ResearchTrend:
        """
        分析研究趋势
        
        分析特定主题的研究趋势。
        """
        # 按年份统计论文数量
        papers = [p for p in self.paper_cache.values() if any(
            topic.lower() in kw.lower() for kw in p.keywords
        )]
        
        current_year = datetime.now().year
        paper_counts = {}
        
        for year in range(current_year - years, current_year + 1):
            count = sum(1 for p in papers if p.year == year)
            paper_counts[year] = count
        
        # 计算增长率
        values = list(paper_counts.values())
        if len(values) > 1 and values[0] > 0:
            growth_rate = (values[-1] - values[0]) / values[0]
        else:
            growth_rate = 0
        
        # 提取热门关键词
        all_keywords = []
        for p in papers:
            all_keywords.extend(p.keywords)
        
        from collections import Counter
        keyword_counts = Counter(all_keywords)
        hot_keywords = [kw for kw, count in keyword_counts.most_common(10)]
        
        # 识别领先作者
        all_authors = []
        for p in papers:
            all_authors.extend(p.authors)
        author_counts = Counter(all_authors)
        leading_authors = [a for a, count in author_counts.most_common(5)]
        
        trend = ResearchTrend(
            topic=topic,
            start_year=current_year - years,
            end_year=current_year,
            paper_counts=paper_counts,
            growth_rate=growth_rate,
            hot_keywords=hot_keywords,
            leading_authors=leading_authors,
            key_papers=[p.id for p in papers[:5]]
        )
        
        return trend
    
    async def generate_literature_review(self, topic: str) -> str:
        """
        生成文献综述
        
        自动生成特定主题的文献综述。
        """
        # 搜索相关文献
        papers = await self.search_literature(topic, max_results=20)
        
        # 提取知识
        await self.extract_knowledge(papers)
        
        # 整合知识
        integration = await self.integrate_knowledge()
        
        # 识别研究空白
        gaps = await self.identify_research_gaps()
        
        # 分析趋势
        trend = await self.analyze_research_trends(topic)
        
        # 生成综述文本
        review = f"""
# Literature Review: {topic}

## 1. Introduction

This review summarizes recent advances in {topic} based on analysis of {len(papers)} relevant publications.

## 2. Research Trends

The field has shown a {"rapid" if trend.growth_rate > 0.5 else "steady"} growth over the past {trend.end_year - trend.start_year} years,
with {trend.paper_counts.get(trend.end_year, 0)} publications in {trend.end_year}.

### 2.1 Hot Topics
- {chr(10).join(f"- {kw}" for kw in trend.hot_keywords[:5])}

### 2.2 Leading Researchers
{chr(10).join(f"- {author}" for author in trend.leading_authors[:3])}

## 3. Key Findings

{len(integration.get('synthesized_knowledge', []))} key knowledge areas have been identified:

"""
        
        for i, knowledge in enumerate(integration.get('synthesized_knowledge', [])[:5], 1):
            review += f"""
### 3.{i} {knowledge['type'].capitalize()}

{knowledge['synthesized_content']}
- Sources: {knowledge['source_count']} studies
- Consensus level: {knowledge['consensus_level']:.2f}
"""
        
        review += f"""

## 4. Research Gaps

{len(gaps)} major research gaps have been identified:

"""
        
        for i, gap in enumerate(gaps[:5], 1):
            review += f"""
### 4.{i} {gap.category.capitalize()} Gap

**Description**: {gap.description}

**Importance**: {gap.importance:.2f}/1.0

**Potential Impact**: {gap.potential_impact}

**Suggested Approaches**:
{chr(10).join(f"- {approach}" for approach in gap.suggested_approaches) if gap.suggested_approaches else "- Further investigation needed"}
"""
        
        review += f"""

## 5. Conclusions and Future Directions

Based on this analysis, future research in {topic} should focus on:
1. Addressing the identified knowledge gaps
2. Leveraging emerging methodologies
3. Exploring new material systems

---
*Generated by LiteratureAgent on {datetime.now().strftime("%Y-%m-%d")}*
"""
        
        return review
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            "papers_cached": len(self.paper_cache),
            "knowledge_units": len(self.knowledge_base),
            "research_gaps": len(self.research_gaps),
            "sources_available": [s.value for s in self.sources.keys()],
            "knowledge_by_type": self._count_by_type()
        }
    
    def _count_by_type(self) -> Dict:
        """按类型统计知识"""
        counts = defaultdict(int)
        for unit in self.knowledge_base:
            counts[unit.knowledge_type.value] += 1
        return dict(counts)


if __name__ == "__main__":
    # 测试代码
    async def test_literature_agent():
        agent = LiteratureAgent()
        
        # 1. 搜索文献
        print("=" * 60)
        print("1. Searching Literature")
        print("=" * 60)
        
        papers = await agent.search_literature(
            query="catalyst materials machine learning",
            max_results=10
        )
        
        print(f"\nFound {len(papers)} papers:")
        for i, paper in enumerate(papers[:3], 1):
            print(f"\n{i}. {paper.title}")
            print(f"   Authors: {', '.join(paper.authors)}")
            print(f"   Year: {paper.year}, Citations: {paper.citations}")
            print(f"   Relevance: {paper.relevance_score:.2f}")
        
        # 2. 提取知识
        print("\n" + "=" * 60)
        print("2. Extracting Knowledge")
        print("=" * 60)
        
        units = await agent.extract_knowledge(papers)
        print(f"\nExtracted {len(units)} knowledge units")
        
        for i, unit in enumerate(units[:5], 1):
            print(f"\n{i}. [{unit.knowledge_type.value}] {unit.content[:60]}...")
            print(f"   Confidence: {unit.confidence:.2f}")
            print(f"   Entities: {', '.join(unit.entities[:3])}")
        
        # 3. 整合知识
        print("\n" + "=" * 60)
        print("3. Integrating Knowledge")
        print("=" * 60)
        
        integration = await agent.integrate_knowledge()
        print(f"\nTotal units: {integration['total_units']}")
        print(f"Integrated units: {integration['integrated_units']}")
        print(f"Contradictions found: {integration['contradictions_found']}")
        
        # 4. 识别研究空白
        print("\n" + "=" * 60)
        print("4. Identifying Research Gaps")
        print("=" * 60)
        
        gaps = await agent.identify_research_gaps()
        print(f"\nFound {len(gaps)} research gaps:")
        
        for i, gap in enumerate(gaps[:3], 1):
            print(f"\n{i}. [{gap.category.upper()}] {gap.description[:60]}...")
            print(f"   Importance: {gap.importance:.2f}, Difficulty: {gap.difficulty:.2f}")
            print(f"   Impact: {gap.potential_impact[:60]}...")
        
        # 5. 分析趋势
        print("\n" + "=" * 60)
        print("5. Analyzing Research Trends")
        print("=" * 60)
        
        trend = await agent.analyze_research_trends("catalysis")
        print(f"\nTopic: {trend.topic}")
        print(f"Period: {trend.start_year}-{trend.end_year}")
        print(f"Growth rate: {trend.growth_rate:.2f}")
        print(f"Hot keywords: {', '.join(trend.hot_keywords[:5])}")
        
        # 6. 统计信息
        print("\n" + "=" * 60)
        print("6. Statistics")
        print("=" * 60)
        
        stats = agent.get_statistics()
        print(f"\nPapers cached: {stats['papers_cached']}")
        print(f"Knowledge units: {stats['knowledge_units']}")
        print(f"Knowledge by type: {stats['knowledge_by_type']}")
    
    asyncio.run(test_literature_agent())
