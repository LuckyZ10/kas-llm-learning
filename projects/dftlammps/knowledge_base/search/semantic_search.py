"""
Semantic Search - 语义搜索接口
=============================
提供基于向量的语义搜索功能，支持混合搜索和结果聚合。
"""

from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SearchMode(Enum):
    """搜索模式"""
    SEMANTIC = auto()      # 纯语义搜索
    KEYWORD = auto()       # 纯关键词搜索
    HYBRID = auto()        # 混合搜索
    FILTERED = auto()      # 过滤搜索


class SearchOperator(Enum):
    """搜索操作符"""
    AND = "and"
    OR = "or"
    NOT = "not"


@dataclass
class SearchQuery:
    """搜索查询"""
    query: str
    mode: SearchMode = SearchMode.HYBRID
    filters: Dict[str, Any] = field(default_factory=dict)
    top_k: int = 10
    threshold: float = 0.0
    
    # 混合搜索权重
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    
    # 高级选项
    include_vectors: bool = False
    include_metadata: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "mode": self.mode.name,
            "filters": self.filters,
            "top_k": self.top_k,
            "threshold": self.threshold,
            "semantic_weight": self.semantic_weight,
            "keyword_weight": self.keyword_weight
        }


@dataclass
class SearchResult:
    """搜索结果"""
    id: str
    score: float
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[List[float]] = None
    highlights: List[str] = field(default_factory=list)
    source: str = ""  # 来源 (semantic, keyword, hybrid)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "score": self.score,
            "content": self.content,
            "metadata": self.metadata,
            "highlights": self.highlights,
            "source": self.source
        }


@dataclass
class SearchConfig:
    """搜索配置"""
    # 向量存储
    vector_store: Optional[Any] = None
    
    # 文本存储 (用于关键词搜索)
    text_store: Optional[Any] = None
    
    # 嵌入提供者
    embedding_provider: Optional[Any] = None
    
    # 默认参数
    default_top_k: int = 10
    default_threshold: float = 0.0
    
    # 混合搜索配置
    rerank_enabled: bool = True
    rerank_model: Optional[str] = None
    
    # 结果处理
    deduplication: bool = True
    diversity_boost: bool = False
    
    # 缓存
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 秒


class SemanticSearch:
    """
    语义搜索引擎
    
    整合向量搜索和关键词搜索，提供高质量的搜索结果。
    """
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self._cache: Dict[str, Any] = {}
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """
        执行搜索
        
        Args:
            query: 搜索查询
            
        Returns:
            搜索结果列表
        """
        # 检查缓存
        cache_key = self._generate_cache_key(query)
        if self.config.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]
        
        # 根据模式选择搜索策略
        if query.mode == SearchMode.SEMANTIC:
            results = self._semantic_search(query)
        elif query.mode == SearchMode.KEYWORD:
            results = self._keyword_search(query)
        elif query.mode == SearchMode.HYBRID:
            results = self._hybrid_search(query)
        else:
            results = self._filtered_search(query)
        
        # 应用过滤
        results = self._apply_filters(results, query.filters)
        
        # 应用阈值
        if query.threshold > 0:
            results = [r for r in results if r.score >= query.threshold]
        
        # 去重
        if self.config.deduplication:
            results = self._deduplicate(results)
        
        # 重新排序
        if self.config.rerank_enabled:
            results = self._rerank(results, query)
        
        # 限制数量
        results = results[:query.top_k]
        
        # 缓存结果
        if self.config.cache_enabled:
            self._cache[cache_key] = results
        
        return results
    
    def _semantic_search(self, query: SearchQuery) -> List[SearchResult]:
        """语义搜索"""
        if self.config.vector_store is None:
            logger.warning("Vector store not configured")
            return []
        
        if self.config.embedding_provider is None:
            logger.warning("Embedding provider not configured")
            return []
        
        # 生成查询向量
        query_vector = self.config.embedding_provider.embed_text(query.query)
        
        # 执行向量搜索
        vector_results = self.config.vector_store.search(
            query_vector=query_vector,
            top_k=query.top_k * 2,  # 获取更多用于后续处理
            filter=query.filters if query.filters else None
        )
        
        # 转换为SearchResult
        results = []
        for vr in vector_results:
            result = SearchResult(
                id=vr.id,
                score=vr.score,
                content=vr.metadata.get("content", ""),
                metadata=vr.metadata,
                vector=vr.vector if query.include_vectors else None,
                source="semantic"
            )
            results.append(result)
        
        return results
    
    def _keyword_search(self, query: SearchQuery) -> List[SearchResult]:
        """关键词搜索"""
        if self.config.text_store is None:
            # 如果没有文本存储，尝试从向量存储获取
            logger.warning("Text store not configured")
            return []
        
        # 执行文本搜索
        text_results = self.config.text_store.search(
            query=query.query,
            filters=query.filters
        )
        
        results = []
        for tr in text_results:
            result = SearchResult(
                id=tr.get("id", ""),
                score=tr.get("score", 0.0),
                content=tr.get("content", ""),
                metadata=tr.get("metadata", {}),
                highlights=tr.get("highlights", []),
                source="keyword"
            )
            results.append(result)
        
        return results
    
    def _hybrid_search(self, query: SearchQuery) -> List[SearchResult]:
        """混合搜索"""
        # 并行执行语义搜索和关键词搜索
        semantic_results = self._semantic_search(query)
        keyword_results = self._keyword_search(query)
        
        # 归一化分数
        semantic_results = self._normalize_scores(semantic_results)
        keyword_results = self._normalize_scores(keyword_results)
        
        # 合并结果
        combined = {}
        
        # 添加语义搜索结果
        for r in semantic_results:
            combined[r.id] = SearchResult(
                id=r.id,
                score=r.score * query.semantic_weight,
                content=r.content,
                metadata=r.metadata,
                vector=r.vector,
                source="hybrid"
            )
        
        # 添加关键词搜索结果
        for r in keyword_results:
            if r.id in combined:
                combined[r.id].score += r.score * query.keyword_weight
            else:
                combined[r.id] = SearchResult(
                    id=r.id,
                    score=r.score * query.keyword_weight,
                    content=r.content,
                    metadata=r.metadata,
                    source="hybrid"
                )
        
        # 转换为列表并排序
        results = list(combined.values())
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def _filtered_search(self, query: SearchQuery) -> List[SearchResult]:
        """过滤搜索"""
        # 先获取所有候选结果
        base_query = SearchQuery(
            query=query.query,
            mode=SearchMode.SEMANTIC,
            top_k=query.top_k * 5,
            filters={}
        )
        results = self._semantic_search(base_query)
        
        # 应用严格过滤
        results = self._apply_filters(results, query.filters, strict=True)
        
        return results
    
    def _normalize_scores(self, results: List[SearchResult]) -> List[SearchResult]:
        """归一化分数到[0, 1]范围"""
        if not results:
            return results
        
        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return results
        
        for r in results:
            r.score = (r.score - min_score) / (max_score - min_score)
        
        return results
    
    def _apply_filters(
        self,
        results: List[SearchResult],
        filters: Dict[str, Any],
        strict: bool = False
    ) -> List[SearchResult]:
        """应用过滤器"""
        if not filters:
            return results
        
        filtered = []
        for r in results:
            match = True
            for key, value in filters.items():
                if key in r.metadata:
                    if isinstance(value, list):
                        if r.metadata[key] not in value:
                            match = False
                            break
                    elif r.metadata[key] != value:
                        match = False
                        break
                elif strict:
                    match = False
                    break
            
            if match:
                filtered.append(r)
        
        return filtered
    
    def _deduplicate(self, results: List[SearchResult]) -> List[SearchResult]:
        """去重"""
        seen = set()
        unique = []
        for r in results:
            if r.id not in seen:
                seen.add(r.id)
                unique.append(r)
        return unique
    
    def _rerank(self, results: List[SearchResult], query: SearchQuery) -> List[SearchResult]:
        """重新排序"""
        # 简单的重排序：考虑查询词在内容中的出现
        query_terms = set(query.query.lower().split())
        
        for r in results:
            content_lower = r.content.lower()
            term_matches = sum(1 for term in query_terms if term in content_lower)
            boost = min(term_matches * 0.1, 0.3)  # 最多提升0.3
            r.score = min(r.score + boost, 1.0)
        
        # 重新排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def _generate_cache_key(self, query: SearchQuery) -> str:
        """生成缓存键"""
        import hashlib
        key_str = f"{query.query}:{query.mode.name}:{query.top_k}:{sorted(query.filters.items())}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()


class HybridSearch:
    """
    混合搜索引擎
    
    支持多源搜索结果的智能融合。
    """
    
    def __init__(
        self,
        vector_store: Optional[Any] = None,
        graph_store: Optional[Any] = None,
        doc_store: Optional[Any] = None,
        embedding_provider: Optional[Any] = None
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.doc_store = doc_store
        self.embedding_provider = embedding_provider
    
    def search(
        self,
        query: str,
        sources: List[str] = None,
        top_k: int = 10
    ) -> Dict[str, List[SearchResult]]:
        """
        多源混合搜索
        
        Args:
            query: 查询字符串
            sources: 搜索源列表 ["vector", "graph", "document"]
            top_k: 每个源的返回数量
            
        Returns:
            按源分组的结果
        """
        sources = sources or ["vector", "graph", "document"]
        results = {}
        
        if "vector" in sources and self.vector_store:
            results["vector"] = self._search_vector(query, top_k)
        
        if "graph" in sources and self.graph_store:
            results["graph"] = self._search_graph(query, top_k)
        
        if "document" in sources and self.doc_store:
            results["document"] = self._search_document(query, top_k)
        
        return results
    
    def _search_vector(self, query: str, top_k: int) -> List[SearchResult]:
        """向量搜索"""
        if self.embedding_provider is None:
            return []
        
        query_vector = self.embedding_provider.embed_text(query)
        vector_results = self.vector_store.search(query_vector, top_k)
        
        return [
            SearchResult(
                id=vr.id,
                score=vr.score,
                content=vr.metadata.get("content", ""),
                metadata=vr.metadata,
                source="vector"
            )
            for vr in vector_results
        ]
    
    def _search_graph(self, query: str, top_k: int) -> List[SearchResult]:
        """图搜索 - 查找相关节点和路径"""
        # 提取查询中的实体
        entities = self._extract_entities(query)
        
        results = []
        for entity in entities:
            # 查找相关节点
            nodes = self.graph_store.get_nodes_by_label(
                label="Material",
                properties={"name": entity}
            )
            
            for node in nodes:
                results.append(SearchResult(
                    id=str(node.get("id", "")),
                    score=0.9,
                    content=node.get("name", ""),
                    metadata=node.get("properties", {}),
                    source="graph"
                ))
                
                # 获取相关路径
                related = self.graph_store.get_relationships(
                    node_id=str(node.get("id", "")),
                    direction="out"
                )
                for rel in related:
                    results.append(SearchResult(
                        id=str(rel.get("r", {}).get("id", "")),
                        score=0.7,
                        content=f"{node.get('name')} -> {rel.get('type', '')}",
                        metadata=rel,
                        source="graph"
                    ))
        
        return results[:top_k]
    
    def _search_document(self, query: str, top_k: int) -> List[SearchResult]:
        """文档搜索"""
        if self.doc_store is None:
            return []
        
        doc_results = self.doc_store.search(query, top_k)
        
        return [
            SearchResult(
                id=dr.get("id", ""),
                score=dr.get("score", 0),
                content=dr.get("content", ""),
                metadata=dr.get("metadata", {}),
                highlights=dr.get("highlights", []),
                source="document"
            )
            for dr in doc_results
        ]
    
    def _extract_entities(self, query: str) -> List[str]:
        """从查询中提取实体"""
        # 简单的实体提取：假设查询中的大写单词或化学式
        import re
        
        # 匹配化学式 (如 Li3PS4)
        chemical_pattern = r'[A-Z][a-z]?\d*'
        chemicals = re.findall(chemical_pattern, query)
        
        # 匹配引号中的词组
        quoted = re.findall(r'"([^"]+)"', query)
        
        # 匹配大写词
        words = query.split()
        capitalized = [w for w in words if w[0].isupper()]
        
        entities = list(set(chemicals + quoted + capitalized))
        return entities


class SearchAggregator:
    """
    搜索结果聚合器
    
    用于整合多个搜索结果，生成统一的输出。
    """
    
    def __init__(
        self,
        deduplication_threshold: float = 0.9,
        diversity_factor: float = 0.3
    ):
        self.deduplication_threshold = deduplication_threshold
        self.diversity_factor = diversity_factor
    
    def aggregate(
        self,
        results_by_source: Dict[str, List[SearchResult]],
        weights: Optional[Dict[str, float]] = None,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        聚合搜索结果
        
        Args:
            results_by_source: 按源分组的结果
            weights: 各源权重
            top_k: 返回数量
            
        Returns:
            聚合后的结果
        """
        weights = weights or {source: 1.0 for source in results_by_source.keys()}
        
        # 加权合并
        all_results = []
        for source, results in results_by_source.items():
            weight = weights.get(source, 1.0)
            for r in results:
                r.score *= weight
                all_results.append(r)
        
        # 排序
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # 去重
        all_results = self._deduplicate_by_content(all_results)
        
        # 多样性重排序
        if self.diversity_factor > 0:
            all_results = self._diversity_rerank(all_results)
        
        return all_results[:top_k]
    
    def _deduplicate_by_content(
        self,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """基于内容相似度去重"""
        unique = []
        for r in results:
            is_duplicate = False
            for u in unique:
                similarity = self._content_similarity(r.content, u.content)
                if similarity > self.deduplication_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(r)
        
        return unique
    
    def _content_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        # 简单的Jaccard相似度
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _diversity_rerank(self, results: List[SearchResult]) -> List[SearchResult]:
        """多样性重排序 - MMR算法"""
        if len(results) <= 1:
            return results
        
        selected = [results[0]]
        remaining = results[1:]
        
        while remaining and len(selected) < len(results):
            mmr_scores = []
            for r in remaining:
                # 计算与已选结果的最大相似度
                max_sim = max(
                    self._content_similarity(r.content, s.content)
                    for s in selected
                )
                # MMR分数
                mmr_score = (1 - self.diversity_factor) * r.score - self.diversity_factor * max_sim
                mmr_scores.append((r, mmr_score))
            
            # 选择MMR分数最高的
            mmr_scores.sort(key=lambda x: x[1], reverse=True)
            selected.append(mmr_scores[0][0])
            remaining.remove(mmr_scores[0][0])
        
        return selected
    
    def generate_summary(
        self,
        results: List[SearchResult],
        max_length: int = 200
    ) -> str:
        """生成搜索结果摘要"""
        if not results:
            return "未找到相关结果。"
        
        # 提取关键信息
        sources = set(r.source for r in results)
        top_result = results[0]
        
        summary = f"找到 {len(results)} 个相关结果。"
        summary += f"\n最相关结果来自{top_result.source}: "
        summary += top_result.content[:max_length]
        
        if len(top_result.content) > max_length:
            summary += "..."
        
        return summary


def create_search_engine(config: Dict[str, Any]) -> SemanticSearch:
    """
    工厂函数：创建搜索引擎
    
    Args:
        config: 配置字典
        
    Returns:
        搜索引擎实例
    """
    search_config = SearchConfig(**config)
    return SemanticSearch(search_config)
