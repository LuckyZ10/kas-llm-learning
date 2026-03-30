"""
主题建模模块
使用BERTopic或LDA进行主题分析
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
import re

from ..config.models import Paper


class TopicModeler:
    """主题建模器"""
    
    def __init__(self, method: str = "bertopic", n_topics: int = 10):
        self.method = method
        self.n_topics = n_topics
        self.model = None
        self.topic_labels = {}
    
    def fit(self, papers: List[Paper]) -> Dict[int, str]:
        """
        训练主题模型
        
        Args:
            papers: 论文列表
        
        Returns:
            主题标签字典
        """
        # 提取文本
        texts = []
        for paper in papers:
            text = f"{paper.title} {paper.abstract}"
            if paper.full_text:
                text += f" {paper.full_text[:5000]}"
            texts.append(self._preprocess(text))
        
        if self.method == "bertopic":
            return self._fit_bertopic(texts, papers)
        else:
            return self._fit_lda(texts, papers)
    
    def _fit_bertopic(self, texts: List[str], papers: List[Paper]) -> Dict[int, str]:
        """使用BERTopic训练"""
        try:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            from sklearn.feature_extraction.text import CountVectorizer
            
            # 创建向量化器
            vectorizer = CountVectorizer(
                stop_words="english",
                min_df=2,
                max_df=0.8,
                ngram_range=(1, 2)
            )
            
            # 创建模型
            self.model = BERTopic(
                vectorizer_model=vectorizer,
                nr_topics=self.n_topics,
                verbose=True
            )
            
            # 训练
            topics, probs = self.model.fit_transform(texts)
            
            # 为每个主题生成标签
            topic_info = self.model.get_topic_info()
            
            for topic_id in topic_info['Topic']:
                if topic_id == -1:
                    continue
                
                topic_words = self.model.get_topic(topic_id)
                if topic_words:
                    # 使用前3个词生成标签
                    words = [word for word, _ in topic_words[:3]]
                    label = " ".join(words).title()
                    self.topic_labels[topic_id] = label
            
            # 为论文分配主题
            for i, paper in enumerate(papers):
                if i < len(topics) and topics[i] != -1:
                    paper.topics = [self.topic_labels.get(topics[i], f"Topic {topics[i]}")]
            
            return self.topic_labels
        
        except ImportError:
            print("BERTopic未安装，回退到LDA")
            return self._fit_lda(texts, papers)
    
    def _fit_lda(self, texts: List[str], papers: List[Paper]) -> Dict[int, str]:
        """使用LDA训练"""
        try:
            from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
            from sklearn.decomposition import LatentDirichletAllocation
            
            # 向量化 - 使用更宽松的参数
            min_df = min(1, len(texts) - 1) if len(texts) > 1 else 1
            vectorizer = CountVectorizer(
                max_df=1.0,
                min_df=min_df,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            doc_term_matrix = vectorizer.fit_transform(texts)
            
            # 训练LDA
            lda = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=42,
                max_iter=20,
                learning_method='online'
            )
            
            lda.fit(doc_term_matrix)
            self.model = lda
            
            # 提取特征词
            feature_names = vectorizer.get_feature_names_out()
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-5:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                self.topic_labels[topic_idx] = " ".join(top_words).title()
            
            # 为论文分配主题
            doc_topics = lda.transform(doc_term_matrix)
            for i, paper in enumerate(papers):
                dominant_topic = doc_topics[i].argmax()
                paper.topics = [self.topic_labels[dominant_topic]]
            
            return self.topic_labels
        
        except ImportError:
            print("scikit-learn未安装，使用简单关键词聚类")
            return self._simple_clustering(texts, papers)
    
    def _simple_clustering(self, texts: List[str], papers: List[Paper]) -> Dict[int, str]:
        """简单关键词聚类"""
        # 定义主题关键词
        topic_keywords = {
            "DFT Calculations": ["dft", "density functional", "electronic structure", "band structure"],
            "Molecular Dynamics": ["molecular dynamics", "md simulation", "force field", "trajectory"],
            "Machine Learning": ["machine learning", "neural network", "deep learning", "prediction"],
            "Battery Materials": ["battery", "lithium", "electrolyte", "cathode", "anode"],
            "Catalysis": ["catalyst", "catalytic", "reaction", "activation energy"],
            "Thermodynamics": ["thermodynamic", "free energy", "enthalpy", "entropy"],
            "Transport Properties": ["diffusion", "conductivity", "transport", "migration"],
            "Phase Transitions": ["phase transition", "phase diagram", "melting", "crystallization"]
        }
        
        topic_assignments = {i: name for i, name in enumerate(topic_keywords.keys())}
        
        for paper in papers:
            text = f"{paper.title} {paper.abstract}".lower()
            scores = {}
            
            for topic_name, keywords in topic_keywords.items():
                score = sum(1 for kw in keywords if kw in text)
                if score > 0:
                    scores[topic_name] = score
            
            if scores:
                best_topic = max(scores.items(), key=lambda x: x[1])[0]
                paper.topics = [best_topic]
        
        return topic_assignments
    
    def _preprocess(self, text: str) -> str:
        """预处理文本"""
        # 转小写
        text = text.lower()
        
        # 移除特殊字符
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 移除多余空白
        text = ' '.join(text.split())
        
        return text
    
    def get_topic_distribution(self) -> Dict[str, int]:
        """获取主题分布"""
        if not self.topic_labels:
            return {}
        
        return dict(Counter(self.topic_labels.values()))
    
    def visualize_topics(self, output_path: Optional[str] = None):
        """可视化主题"""
        if self.method == "bertopic" and self.model:
            try:
                fig = self.model.visualize_topics()
                if output_path:
                    fig.write_html(output_path)
                return fig
            except Exception as e:
                print(f"可视化失败: {e}")
        
        return None


class KeywordExtractor:
    """关键词提取器"""
    
    def __init__(self, top_n: int = 10):
        self.top_n = top_n
    
    def extract(self, papers: List[Paper]) -> List[Tuple[str, int]]:
        """
        从论文中提取关键词
        
        Args:
            papers: 论文列表
        
        Returns:
            关键词和频率列表
        """
        all_text = ""
        for paper in papers:
            all_text += f" {paper.title} {paper.abstract}"
        
        # 简单频率统计
        words = self._tokenize(all_text.lower())
        
        # 过滤停用词
        stopwords = self._get_stopwords()
        words = [w for w in words if w not in stopwords and len(w) > 2]
        
        # 统计频率
        word_freq = Counter(words)
        
        return word_freq.most_common(self.top_n)
    
    def extract_bigrams(self, papers: List[Paper]) -> List[Tuple[str, int]]:
        """提取二元词组"""
        all_text = ""
        for paper in papers:
            all_text += f" {paper.title} {paper.abstract}"
        
        words = self._tokenize(all_text.lower())
        stopwords = self._get_stopwords()
        words = [w for w in words if w not in stopwords and len(w) > 2]
        
        # 构建二元组
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        
        bigram_freq = Counter(bigrams)
        return bigram_freq.most_common(self.top_n)
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        return re.findall(r'\b[a-z]+\b', text)
    
    def _get_stopwords(self) -> set:
        """获取停用词"""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'this', 'that', 'these', 'those', 'we', 'our', 'study',
            'paper', 'research', 'results', 'method', 'using', 'used', 'shown',
            'show', 'based', 'approach', 'analysis', 'work', 'also', 'however',
            'thus', 'therefore', 'furthermore', 'moreover', 'here', 'where',
            'when', 'what', 'how', 'why', 'which', 'who', 'all', 'each',
            'every', 'both', 'either', 'neither', 'one', 'two', 'three'
        }


class DomainClassifier:
    """领域分类器"""
    
    DOMAINS = {
        "computational_chemistry": ["dft", "quantum", "electronic", "wavefunction", "basis set"],
        "molecular_simulation": ["md", "molecular dynamics", "monte carlo", "force field"],
        "machine_learning": ["neural network", "deep learning", "ml", "ai", "prediction"],
        "materials_science": ["crystal", "lattice", "defect", "alloy", "ceramic"],
        "energy_storage": ["battery", "supercapacitor", "fuel cell", "electrolyte"],
        "catalysis": ["catalyst", "reaction", "mechanism", "activation"]
    }
    
    def classify(self, paper: Paper) -> List[str]:
        """分类论文领域"""
        text = f"{paper.title} {paper.abstract}".lower()
        
        scores = {}
        for domain, keywords in self.DOMAINS.items():
            score = sum(2 if kw in paper.title.lower() else 1 
                       for kw in keywords if kw in text)
            if score > 0:
                scores[domain] = score
        
        # 返回得分最高的领域
        if scores:
            max_score = max(scores.values())
            return [domain for domain, score in scores.items() if score >= max_score * 0.5]
        
        return ["general"]
