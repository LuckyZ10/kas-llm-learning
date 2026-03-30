"""
趋势分析模块
分析研究热点的时间演变
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from datetime import datetime, timedelta

from ..config.models import Paper, ResearchTrend


class TrendAnalyzer:
    """趋势分析器"""
    
    def __init__(self, window_size: int = 1):
        """
        Args:
            window_size: 时间窗口大小（年）
        """
        self.window_size = window_size
    
    def analyze(
        self,
        papers: List[Paper],
        topics: Optional[List[str]] = None
    ) -> List[ResearchTrend]:
        """
        分析研究趋势
        
        Args:
            papers: 论文列表
            topics: 主题列表，None则自动提取
        
        Returns:
            趋势列表
        """
        if not papers:
            return []
        
        # 如果没有提供主题，从论文中提取
        if topics is None:
            topics = self._extract_topics(papers)
        
        # 按年份分组统计
        yearly_data = self._group_by_year(papers)
        
        trends = []
        
        for topic in topics:
            for year in sorted(yearly_data.keys()):
                papers_in_year = yearly_data[year]
                
                # 统计该主题在该年的论文数
                topic_papers = [
                    p for p in papers_in_year
                    if topic in p.topics or topic in p.keywords or
                    topic.lower() in p.title.lower() or
                    topic.lower() in p.abstract.lower()
                ]
                
                if not topic_papers:
                    continue
                
                # 计算引用数
                total_citations = sum(p.citation_count for p in topic_papers)
                avg_citations = total_citations / len(topic_papers) if topic_papers else 0
                
                # 计算增长率
                prev_year = year - 1
                growth_rate = 0.0
                if prev_year in yearly_data:
                    prev_papers = [
                        p for p in yearly_data[prev_year]
                        if topic in p.topics or topic in p.keywords or
                        topic.lower() in p.title.lower()
                    ]
                    if prev_papers:
                        growth_rate = (len(topic_papers) - len(prev_papers)) / len(prev_papers)
                
                trends.append(ResearchTrend(
                    topic=topic,
                    year=year,
                    paper_count=len(topic_papers),
                    citation_count=total_citations,
                    avg_citations=avg_citations,
                    growth_rate=growth_rate,
                    top_papers=[p.id for p in sorted(topic_papers, 
                                                     key=lambda x: x.citation_count, 
                                                     reverse=True)[:5]]
                ))
        
        return trends
    
    def _extract_topics(self, papers: List[Paper]) -> List[str]:
        """提取主题"""
        all_topics = set()
        for paper in papers:
            all_topics.update(paper.topics)
        
        # 如果论文没有主题，从关键词提取
        if not all_topics:
            keyword_counts = defaultdict(int)
            for paper in papers:
                for kw in paper.keywords:
                    keyword_counts[kw] += 1
            
            # 返回最常见的10个关键词
            all_topics = set(sorted(keyword_counts.keys(), 
                                   key=lambda x: keyword_counts[x], 
                                   reverse=True)[:10])
        
        return list(all_topics)
    
    def _group_by_year(self, papers: List[Paper]) -> Dict[int, List[Paper]]:
        """按年份分组"""
        groups = defaultdict(list)
        for paper in papers:
            year = paper.publication_date.year
            groups[year].append(paper)
        return dict(groups)
    
    def get_hot_topics(
        self,
        papers: List[Paper],
        recent_years: int = 3
    ) -> List[Tuple[str, float]]:
        """
        获取热门主题
        
        Args:
            papers: 论文列表
            recent_years: 最近年数
        
        Returns:
            热门主题和热度分数
        """
        cutoff_year = datetime.now().year - recent_years
        recent_papers = [p for p in papers if p.publication_date.year >= cutoff_year]
        
        if not recent_papers:
            return []
        
        trends = self.analyze(recent_papers)
        
        # 计算每个主题的热度分数
        topic_scores = defaultdict(lambda: {"papers": 0, "citations": 0, "growth": 0})
        
        for trend in trends:
            topic_scores[trend.topic]["papers"] += trend.paper_count
            topic_scores[trend.topic]["citations"] += trend.citation_count
            topic_scores[trend.topic]["growth"] += trend.growth_rate
        
        # 计算综合热度分数
        hot_topics = []
        for topic, scores in topic_scores.items():
            # 热度 = 论文数 + 引用数*0.1 + 增长率*50
            heat_score = (scores["papers"] + 
                         scores["citations"] * 0.1 + 
                         max(0, scores["growth"]) * 50)
            hot_topics.append((topic, heat_score))
        
        return sorted(hot_topics, key=lambda x: x[1], reverse=True)
    
    def predict_trends(
        self,
        trends: List[ResearchTrend],
        forecast_years: int = 2
    ) -> Dict[str, List[Tuple[int, float]]]:
        """
        预测未来趋势（简单线性回归）
        
        Args:
            trends: 历史趋势
            forecast_years: 预测年数
        
        Returns:
            预测结果
        """
        # 按主题分组
        topic_data = defaultdict(list)
        for trend in trends:
            topic_data[trend.topic].append((trend.year, trend.paper_count))
        
        predictions = {}
        
        for topic, data in topic_data.items():
            if len(data) < 2:
                continue
            
            years = np.array([d[0] for d in data])
            counts = np.array([d[1] for d in data])
            
            # 简单线性回归
            z = np.polyfit(years, counts, 1)
            p = np.poly1d(z)
            
            # 预测
            last_year = max(years)
            forecast = []
            for i in range(1, forecast_years + 1):
                pred_year = last_year + i
                pred_count = max(0, p(pred_year))
                forecast.append((pred_year, pred_count))
            
            predictions[topic] = forecast
        
        return predictions
    
    def get_emerging_topics(
        self,
        papers: List[Paper],
        min_growth_rate: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        识别新兴主题
        
        Args:
            papers: 论文列表
            min_growth_rate: 最小增长率阈值
        
        Returns:
            新兴主题列表
        """
        trends = self.analyze(papers)
        
        # 按主题分组，计算平均增长率
        topic_growth = defaultdict(list)
        for trend in trends:
            topic_growth[trend.topic].append(trend.growth_rate)
        
        emerging = []
        for topic, growth_rates in topic_growth.items():
            avg_growth = np.mean(growth_rates)
            if avg_growth >= min_growth_rate:
                emerging.append((topic, avg_growth))
        
        return sorted(emerging, key=lambda x: x[1], reverse=True)
    
    def get_declining_topics(
        self,
        papers: List[Paper],
        max_growth_rate: float = -0.2
    ) -> List[Tuple[str, float]]:
        """
        识别衰退主题
        
        Args:
            papers: 论文列表
            max_growth_rate: 最大增长率阈值（负数）
        
        Returns:
            衰退主题列表
        """
        trends = self.analyze(papers)
        
        topic_growth = defaultdict(list)
        for trend in trends:
            topic_growth[trend.topic].append(trend.growth_rate)
        
        declining = []
        for topic, growth_rates in topic_growth.items():
            avg_growth = np.mean(growth_rates)
            if avg_growth <= max_growth_rate:
                declining.append((topic, avg_growth))
        
        return sorted(declining, key=lambda x: x[1])


class CitationAnalyzer:
    """引用分析器"""
    
    def analyze_impact(self, papers: List[Paper]) -> Dict[str, Any]:
        """
        分析引用影响
        
        Returns:
            影响指标
        """
        if not papers:
            return {}
        
        citation_counts = [p.citation_count for p in papers]
        
        return {
            "total_papers": len(papers),
            "total_citations": sum(citation_counts),
            "avg_citations": np.mean(citation_counts),
            "median_citations": np.median(citation_counts),
            "max_citations": max(citation_counts),
            "h_index": self._calculate_h_index(citation_counts),
            "highly_cited_papers": [
                {"id": p.id, "title": p.title, "citations": p.citation_count}
                for p in sorted(papers, key=lambda x: x.citation_count, reverse=True)[:10]
            ]
        }
    
    def _calculate_h_index(self, citations: List[int]) -> int:
        """计算h-index"""
        sorted_citations = sorted(citations, reverse=True)
        h = 0
        for i, c in enumerate(sorted_citations, 1):
            if c >= i:
                h = i
            else:
                break
        return h
    
    def get_citation_network(self, papers: List[Paper]) -> Dict[str, List[str]]:
        """
        获取引用网络
        
        Returns:
            引用关系图
        """
        network = {}
        paper_ids = {p.id for p in papers}
        
        for paper in papers:
            # 只保留在论文列表中的引用
            internal_refs = [ref for ref in paper.references if ref in paper_ids]
            if internal_refs:
                network[paper.id] = internal_refs
        
        return network
    
    def find_key_papers(self, papers: List[Paper], top_n: int = 10) -> List[Paper]:
        """
        找出关键论文（基于引用数、中心性等）
        
        Args:
            papers: 论文列表
            top_n: 返回数量
        
        Returns:
            关键论文列表
        """
        # 计算综合得分
        for paper in papers:
            # 引用得分（归一化）
            citation_score = min(paper.citation_count / 100, 1.0) if paper.citation_count > 0 else 0
            
            # 时间权重（越新越高）
            years_since = datetime.now().year - paper.publication_date.year
            recency_score = max(0, 1 - years_since / 10)
            
            # 综合得分
            paper.importance_score = citation_score * 0.6 + recency_score * 0.4
        
        return sorted(papers, key=lambda x: x.importance_score or 0, reverse=True)[:top_n]
