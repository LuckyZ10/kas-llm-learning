"""
领域相似度评估 (Domain Similarity)
计算材料空间距离，评估领域间相似性

作者: DFT-LAMMPS Team
版本: 1.0.0
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import wasserstein_distance, energy_distance
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings


@dataclass
class SimilarityConfig:
    """相似度计算配置"""
    method: str = "mmd"  # mmd, coral, proxy_a_distance, cosine, euclidean
    metric: str = "rbf"  # 用于某些方法的核函数
    bandwidth: float = 1.0
    n_components: int = 50  # PCA降维维度
    normalize: bool = True
    
    # 可视化配置
    viz_method: str = "tsne"  # tsne, pca, umap
    perplexity: float = 30.0


class DomainSimilarityMetrics:
    """域相似度度量集合"""
    
    @staticmethod
    def maximum_mean_discrepancy(
        X: np.ndarray,
        Y: np.ndarray,
        kernel: str = "rbf",
        bandwidth: float = 1.0
    ) -> float:
        """
        最大均值差异 (MMD)
        
        衡量两个分布之间的差异
        """
        n, m = len(X), len(Y)
        
        if kernel == "rbf":
            # 计算核矩阵
            K_xx = np.exp(
                -cdist(X, X, 'sqeuclidean') / (2 * bandwidth ** 2)
            )
            K_yy = np.exp(
                -cdist(Y, Y, 'sqeuclidean') / (2 * bandwidth ** 2)
            )
            K_xy = np.exp(
                -cdist(X, Y, 'sqeuclidean') / (2 * bandwidth ** 2)
            )
        else:  # linear
            K_xx = np.dot(X, X.T)
            K_yy = np.dot(Y, Y.T)
            K_xy = np.dot(X, Y.T)
        
        # MMD^2 = E[K(x,x')] + E[K(y,y')] - 2E[K(x,y)]
        mmd_squared = (
            K_xx.sum() - np.trace(K_xx)) / (n * (n - 1)) +
            K_yy.sum() - np.trace(K_yy) / (m * (m - 1)) -
            2 * K_xy.sum() / (n * m)
        )
        
        return np.sqrt(max(mmd_squared, 0))
    
    @staticmethod
    def correlation_alignment(
        X: np.ndarray,
        Y: np.ndarray
    ) -> float:
        """
        CORAL - Correlation Alignment Distance
        
        基于协方差矩阵的域距离
        """
        # 计算协方差矩阵
        cov_X = np.cov(X, rowvar=False)
        cov_Y = np.cov(Y, rowvar=False)
        
        # Frobenius范数差异
        diff = cov_X - cov_Y
        distance = np.sqrt(np.sum(diff ** 2))
        
        # 归一化
        return distance / (4 * X.shape[1] ** 2)
    
    @staticmethod
    def proxy_a_distance(
        X: np.ndarray,
        Y: np.ndarray,
        classifier: str = "linear"
    ) -> float:
        """
        Proxy A-Distance
        
        通过训练域分类器来估计域差异
        距离 = 2 * (1 - 2 * error)
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.model_selection import cross_val_score
        
        # 创建标签
        labels = np.concatenate([
            np.zeros(len(X)),
            np.ones(len(Y))
        ])
        
        features = np.vstack([X, Y])
        
        # 训练域分类器
        if classifier == "linear":
            clf = LogisticRegression(max_iter=1000)
        else:
            clf = SVC(kernel='rbf')
        
        # 交叉验证误差
        scores = cross_val_score(clf, features, labels, cv=5)
        error = 1 - scores.mean()
        
        # Proxy A-Distance
        pad = 2 * (1 - 2 * error)
        
        return max(0, pad)
    
    @staticmethod
    def wasserstein_distance_1d(
        X: np.ndarray,
        Y: np.ndarray
    ) -> float:
        """
        Wasserstein距离 (Earth Mover's Distance)
        """
        # 对每个维度计算距离，然后平均
        distances = []
        for i in range(X.shape[1]):
            dist = wasserstein_distance(X[:, i], Y[:, i])
            distances.append(dist)
        
        return np.mean(distances)
    
    @staticmethod
    def energy_distance_metric(
        X: np.ndarray,
        Y: np.ndarray
    ) -> float:
        """
        能量距离
        """
        n, m = len(X), len(Y)
        
        # E|X - Y|
        term1 = np.mean(cdist(X, Y, metric='euclidean'))
        
        # E|X - X'|
        if n > 1:
            term2 = np.mean(pdist(X, metric='euclidean'))
        else:
            term2 = 0
        
        # E|Y - Y'|
        if m > 1:
            term3 = np.mean(pdist(Y, metric='euclidean'))
        else:
            term3 = 0
        
        # Energy distance = 2*E|X-Y| - E|X-X'| - E|Y-Y'|
        return 2 * term1 - term2 - term3
    
    @staticmethod
    def cosine_similarity(
        X: np.ndarray,
        Y: np.ndarray
    ) -> float:
        """
        余弦相似度 (基于均值向量)
        """
        mean_X = X.mean(axis=0)
        mean_Y = Y.mean(axis=0)
        
        norm_X = np.linalg.norm(mean_X)
        norm_Y = np.linalg.norm(mean_Y)
        
        if norm_X == 0 or norm_Y == 0:
            return 0.0
        
        return np.dot(mean_X, mean_Y) / (norm_X * norm_Y)
    
    @staticmethod
    def euclidean_distance(
        X: np.ndarray,
        Y: np.ndarray
    ) -> float:
        """
        欧氏距离 (基于均值向量)
        """
        mean_X = X.mean(axis=0)
        mean_Y = Y.mean(axis=0)
        
        return np.linalg.norm(mean_X - mean_Y)
    
    @staticmethod
    def kl_divergence_approx(
        X: np.ndarray,
        Y: np.ndarray,
        bins: int = 20
    ) -> float:
        """
        KL散度近似 (基于直方图)
        """
        kl_div = 0.0
        
        for i in range(X.shape[1]):
            # 计算直方图
            min_val = min(X[:, i].min(), Y[:, i].min())
            max_val = max(X[:, i].max(), Y[:, i].max())
            
            hist_X, _ = np.histogram(
                X[:, i], bins=bins, range=(min_val, max_val), density=True
            )
            hist_Y, _ = np.histogram(
                Y[:, i], bins=bins, range=(min_val, max_val), density=True
            )
            
            # 添加平滑
            hist_X += 1e-10
            hist_Y += 1e-10
            
            # KL散度
            kl = np.sum(hist_X * np.log(hist_X / hist_Y))
            kl_div += abs(kl)
        
        return kl_div / X.shape[1]
    
    @staticmethod
    def frechet_inception_distance(
        X: np.ndarray,
        Y: np.ndarray
    ) -> float:
        """
        Fréchet Inception Distance (FID)
        
        假设高斯分布，计算分布间距离
        """
        mu_X = np.mean(X, axis=0)
        mu_Y = np.mean(Y, axis=0)
        
        sigma_X = np.cov(X, rowvar=False)
        sigma_Y = np.cov(Y, rowvar=False)
        
        # 均值差异
        mean_diff = np.sum((mu_X - mu_Y) ** 2)
        
        # 协方差差异
        covmean = sqrtm(sigma_X.dot(sigma_Y))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = mean_diff + np.trace(
            sigma_X + sigma_Y - 2 * covmean
        )
        
        return fid


class DomainSimilarityAnalyzer:
    """域相似度分析器"""
    
    def __init__(self, config: Optional[SimilarityConfig] = None):
        self.config = config or SimilarityConfig()
        self.metrics = DomainSimilarityMetrics()
        self.similarity_matrix = None
        self.domain_names = []
    
    def compute_similarity(
        self,
        domain1_features: np.ndarray,
        domain2_features: np.ndarray,
        method: Optional[str] = None
    ) -> float:
        """
        计算两个域之间的相似度
        """
        if method is None:
            method = self.config.method
        
        # 预处理
        if self.config.normalize:
            domain1_features = self._normalize(domain1_features)
            domain2_features = self._normalize(domain2_features)
        
        # 降维（如果指定）
        if self.config.n_components < min(domain1_features.shape[1],
                                           domain2_features.shape[1]):
            pca = PCA(n_components=self.config.n_components)
            combined = np.vstack([domain1_features, domain2_features])
            combined_reduced = pca.fit_transform(combined)
            
            n1 = len(domain1_features)
            domain1_features = combined_reduced[:n1]
            domain2_features = combined_reduced[n1:]
        
        # 计算相似度
        if method == "mmd":
            return self.metrics.maximum_mean_discrepancy(
                domain1_features, domain2_features,
                kernel=self.config.metric,
                bandwidth=self.config.bandwidth
            )
        elif method == "coral":
            return self.metrics.correlation_alignment(
                domain1_features, domain2_features
            )
        elif method == "proxy_a":
            return self.metrics.proxy_a_distance(
                domain1_features, domain2_features
            )
        elif method == "wasserstein":
            return self.metrics.wasserstein_distance_1d(
                domain1_features, domain2_features
            )
        elif method == "energy":
            return self.metrics.energy_distance_metric(
                domain1_features, domain2_features
            )
        elif method == "cosine":
            # 转换为距离 (1 - similarity)
            return 1 - self.metrics.cosine_similarity(
                domain1_features, domain2_features
            )
        elif method == "euclidean":
            return self.metrics.euclidean_distance(
                domain1_features, domain2_features
            )
        elif method == "kl":
            return self.metrics.kl_divergence_approx(
                domain1_features, domain2_features
            )
        elif method == "fid":
            return self.metrics.frechet_inception_distance(
                domain1_features, domain2_features
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def compute_pairwise_similarities(
        self,
        domains: Dict[str, np.ndarray],
        method: Optional[str] = None
    ) -> pd.DataFrame:
        """
        计算所有域对之间的相似度矩阵
        """
        import pandas as pd
        
        self.domain_names = list(domains.keys())
        n_domains = len(self.domain_names)
        
        similarity_matrix = np.zeros((n_domains, n_domains))
        
        for i, name1 in enumerate(self.domain_names):
            for j, name2 in enumerate(self.domain_names):
                if i == j:
                    similarity_matrix[i, j] = 0.0  # 自身距离为0
                elif i < j:
                    dist = self.compute_similarity(
                        domains[name1],
                        domains[name2],
                        method
                    )
                    similarity_matrix[i, j] = dist
                    similarity_matrix[j, i] = dist
        
        self.similarity_matrix = similarity_matrix
        
        return pd.DataFrame(
            similarity_matrix,
            index=self.domain_names,
            columns=self.domain_names
        )
    
    def find_most_similar(
        self,
        target_domain: str,
        n: int = 3
    ) -> List[Tuple[str, float]]:
        """
        找到与目标域最相似的n个域
        """
        if self.similarity_matrix is None:
            raise ValueError("Must compute similarity matrix first")
        
        idx = self.domain_names.index(target_domain)
        distances = self.similarity_matrix[idx]
        
        # 获取排序后的索引（排除自身）
        sorted_indices = np.argsort(distances)
        
        results = []
        for i in sorted_indices:
            if self.domain_names[i] != target_domain:
                results.append((self.domain_names[i], distances[i]))
            if len(results) >= n:
                break
        
        return results
    
    def compute_transferability_score(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
        source_labels: np.ndarray,
        target_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        计算迁移性评分
        
        综合考虑多种因素评估迁移可能性
        """
        scores = {}
        
        # 域距离
        domain_dist = self.compute_similarity(
            source_features, target_features, method="mmd"
        )
        scores["domain_distance"] = domain_dist
        
        # 标签分布相似性
        source_label_dist = np.bincount(source_labels.astype(int))
        target_label_dist = np.bincount(target_labels.astype(int))
        
        # 归一化
        source_label_dist = source_label_dist / source_label_dist.sum()
        target_label_dist = target_label_dist / target_label_dist.sum()
        
        # JS散度
        m = (source_label_dist + target_label_dist) / 2
        kl_source = np.sum(source_label_dist * np.log(
            source_label_dist / (m + 1e-10) + 1e-10
        ))
        kl_target = np.sum(target_label_dist * np.log(
            target_label_dist / (m + 1e-10) + 1e-10
        ))
        js_div = (kl_source + kl_target) / 2
        scores["label_distribution_distance"] = js_div
        
        # 综合迁移性评分 (越高越好迁移)
        transferability = 1.0 / (1 + domain_dist + js_div)
        scores["transferability_score"] = transferability
        
        # 建议
        if transferability > 0.7:
            scores["recommendation"] = "High transferability - Direct transfer recommended"
        elif transferability > 0.4:
            scores["recommendation"] = "Medium transferability - Fine-tuning recommended"
        else:
            scores["recommendation"] = "Low transferability - Domain adaptation required"
        
        return scores
    
    def visualize_domains(
        self,
        domains: Dict[str, np.ndarray],
        labels: Optional[Dict[str, np.ndarray]] = None,
        figsize: Tuple[int, int] = (12, 10)
    ) -> plt.Figure:
        """
        可视化多个域的分布
        """
        # 合并所有域
        all_features = []
        domain_labels_list = []
        
        for name, features in domains.items():
            all_features.append(features)
            domain_labels_list.extend([name] * len(features))
        
        all_features = np.vstack(all_features)
        
        # 降维
        if self.config.viz_method == "tsne":
            reducer = TSNE(
                n_components=2,
                perplexity=min(self.config.perplexity,
                              len(all_features) - 1),
                random_state=42
            )
        else:
            reducer = PCA(n_components=2)
        
        embedded = reducer.fit_transform(all_features)
        
        # 绘图
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(domains)))
        
        start_idx = 0
        for i, (name, features) in enumerate(domains.items()):
            end_idx = start_idx + len(features)
            
            ax.scatter(
                embedded[start_idx:end_idx, 0],
                embedded[start_idx:end_idx, 1],
                c=[colors[i]],
                label=name,
                alpha=0.6,
                s=30
            )
            
            start_idx = end_idx
        
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_title("Domain Distribution Visualization")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def plot_similarity_matrix(
        self,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        绘制相似度矩阵热力图
        """
        if self.similarity_matrix is None:
            raise ValueError("Must compute similarity matrix first")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(
            self.similarity_matrix,
            cmap="YlOrRd",
            aspect="auto"
        )
        
        ax.set_xticks(range(len(self.domain_names)))
        ax.set_yticks(range(len(self.domain_names)))
        ax.set_xticklabels(self.domain_names, rotation=45, ha="right")
        ax.set_yticklabels(self.domain_names)
        
        # 添加数值标注
        for i in range(len(self.domain_names)):
            for j in range(len(self.domain_names)):
                text = ax.text(
                    j, i,
                    f"{self.similarity_matrix[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if self.similarity_matrix[i, j] > 0.5 else "black",
                    fontsize=8
                )
        
        ax.set_title("Domain Similarity Matrix")
        plt.colorbar(im, ax=ax, label="Distance")
        
        plt.tight_layout()
        
        return fig
    
    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """归一化特征"""
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-8
        return (features - mean) / std


class MaterialSpaceDistance:
    """材料空间距离计算器"""
    
    def __init__(self):
        self.element_properties = self._load_element_properties()
    
    def _load_element_properties(self) -> Dict:
        """加载元素属性"""
        # 简化的元素属性表
        return {
            "electronegativity": {},
            "atomic_radius": {},
            "ionization_energy": {}
        }
    
    def compute_composition_distance(
        self,
        comp1: Dict[str, float],
        comp2: Dict[str, float]
    ) -> float:
        """
        计算两个成分之间的距离
        
        基于元素组成和比例
        """
        all_elements = set(comp1.keys()) | set(comp2.keys())
        
        vec1 = np.array([comp1.get(e, 0) for e in all_elements])
        vec2 = np.array([comp2.get(e, 0) for e in all_elements])
        
        # 归一化
        vec1 = vec1 / (vec1.sum() + 1e-10)
        vec2 = vec2 / (vec2.sum() + 1e-10)
        
        return np.linalg.norm(vec1 - vec2)
    
    def compute_structure_distance(
        self,
        struct1: Dict,
        struct2: Dict
    ) -> float:
        """
        计算两个结构之间的距离
        """
        # 基于结构特征的距离
        features1 = self._extract_structure_features(struct1)
        features2 = self._extract_structure_features(struct2)
        
        return np.linalg.norm(features1 - features2)
    
    def _extract_structure_features(self, structure: Dict) -> np.ndarray:
        """提取结构特征向量"""
        # 简化的特征提取
        features = []
        
        # 晶格参数
        if "lattice" in structure:
            lattice = structure["lattice"]
            features.extend([
                lattice.get("a", 0),
                lattice.get("b", 0),
                lattice.get("c", 0),
                lattice.get("alpha", 0),
                lattice.get("beta", 0),
                lattice.get("gamma", 0)
            ])
        
        # 空间群
        if "space_group" in structure:
            features.append(structure["space_group"])
        
        return np.array(features)
    
    def compute_property_distance(
        self,
        props1: Dict[str, float],
        props2: Dict[str, float]
    ) -> float:
        """
        计算性质空间的距离
        """
        common_props = set(props1.keys()) & set(props2.keys())
        
        if not common_props:
            return float('inf')
        
        distances = []
        for prop in common_props:
            v1, v2 = props1[prop], props2[prop]
            # 归一化差异
            diff = abs(v1 - v2) / (abs(v1) + abs(v2) + 1e-10)
            distances.append(diff)
        
        return np.mean(distances)


class TransferPathFinder:
    """迁移路径查找器"""
    
    def __init__(self, similarity_analyzer: DomainSimilarityAnalyzer):
        self.analyzer = similarity_analyzer
    
    def find_optimal_transfer_path(
        self,
        source: str,
        target: str,
        intermediate_domains: List[str],
        max_hops: int = 3
    ) -> List[Tuple[List[str], float]]:
        """
        寻找最优迁移路径
        
        通过中间域逐步迁移
        """
        from itertools import permutations
        
        if max_hops <= 1:
            # 直接迁移
            dist = self._get_distance(source, target)
            return [([source, target], dist)]
        
        best_paths = []
        
        # 尝试不同数量的中间域
        for n_intermediate in range(min(max_hops, len(intermediate_domains)) + 1):
            if n_intermediate == 0:
                # 直接迁移
                dist = self._get_distance(source, target)
                best_paths.append(([source, target], dist))
            else:
                # 尝试所有可能的中间域组合
                for intermediates in permutations(intermediate_domains, n_intermediate):
                    path = [source] + list(intermediates) + [target]
                    total_dist = self._compute_path_distance(path)
                    best_paths.append((path, total_dist))
        
        # 排序并返回最优路径
        best_paths.sort(key=lambda x: x[1])
        
        return best_paths[:5]  # 返回前5条
    
    def _get_distance(self, domain1: str, domain2: str) -> float:
        """获取两个域之间的距离"""
        if self.analyzer.similarity_matrix is None:
            raise ValueError("Similarity matrix not computed")
        
        i = self.analyzer.domain_names.index(domain1)
        j = self.analyzer.domain_names.index(domain2)
        
        return self.analyzer.similarity_matrix[i, j]
    
    def _compute_path_distance(self, path: List[str]) -> float:
        """计算路径的总距离"""
        total = 0.0
        for i in range(len(path) - 1):
            total += self._get_distance(path[i], path[i+1])
        return total


# 演示代码
if __name__ == "__main__":
    print("=" * 60)
    print("领域相似度评估演示 (Domain Similarity Demo)")
    print("=" * 60)
    
    # 生成模拟材料域数据
    np.random.seed(42)
    
    domains = {
        "battery_cathode": np.random.randn(200, 50) + np.array([1.0] * 50),
        "battery_anode": np.random.randn(200, 50) + np.array([0.5] * 50),
        "catalyst_oxide": np.random.randn(200, 50) + np.array([2.0] * 50),
        "catalyst_metal": np.random.randn(200, 50) + np.array([1.8] * 50),
        "semiconductor_2d": np.random.randn(200, 50) + np.array([-0.5] * 50),
        "semiconductor_bulk": np.random.randn(200, 50) + np.array([0.0] * 50),
        "metal_alloy": np.random.randn(200, 50) + np.array([1.2] * 50),
        "ceramic_oxide": np.random.randn(200, 50) + np.array([2.5] * 50)
    }
    
    print(f"\nCreated {len(domains)} material domains:")
    for name, features in domains.items():
        print(f"  {name}: {features.shape}")
    
    # 创建分析器
    config = SimilarityConfig(method="mmd")
    analyzer = DomainSimilarityAnalyzer(config)
    
    # 计算相似度矩阵
    print("\nComputing similarity matrix using MMD...")
    similarity_df = analyzer.compute_pairwise_similarities(domains)
    
    print("\nSimilarity Matrix (MMD Distance):")
    print(similarity_df.round(3))
    
    # 找到最相似的域
    print("\n" + "=" * 60)
    print("Most Similar Domains to 'battery_cathode':")
    print("=" * 60)
    
    similar = analyzer.find_most_similar("battery_cathode", n=3)
    for domain, dist in similar:
        print(f"  {domain}: {dist:.4f}")
    
    # 使用不同方法比较
    print("\n" + "=" * 60)
    print("Comparison of Different Similarity Metrics:")
    print("=" * 60)
    
    methods = ["mmd", "coral", "cosine", "euclidean"]
    X = domains["battery_cathode"]
    Y = domains["battery_anode"]
    
    for method in methods:
        dist = analyzer.compute_similarity(X, Y, method=method)
        print(f"  {method.upper()}: {dist:.4f}")
    
    # 计算迁移性评分
    print("\n" + "=" * 60)
    print("Transferability Analysis:")
    print("=" * 60)
    
    source_features = domains["battery_cathode"]
    target_features = domains["catalyst_oxide"]
    source_labels = np.random.randint(0, 5, len(source_features))
    target_labels = np.random.randint(0, 5, len(target_features))
    
    scores = analyzer.compute_transferability_score(
        source_features, target_features,
        source_labels, target_labels
    )
    
    for key, value in scores.items():
        if isinstance(value, str):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.4f}")
    
    # 可视化
    print("\nGenerating visualizations...")
    fig1 = analyzer.visualize_domains(domains)
    fig1.savefig("domain_visualization.png", dpi=150, bbox_inches="tight")
    print("Saved domain_visualization.png")
    
    fig2 = analyzer.plot_similarity_matrix()
    fig2.savefig("similarity_matrix.png", dpi=150, bbox_inches="tight")
    print("Saved similarity_matrix.png")
    
    # 寻找最优迁移路径
    print("\n" + "=" * 60)
    print("Optimal Transfer Paths:")
    print("=" * 60)
    
    path_finder = TransferPathFinder(analyzer)
    paths = path_finder.find_optimal_transfer_path(
        "battery_cathode",
        "ceramic_oxide",
        list(domains.keys()),
        max_hops=2
    )
    
    for i, (path, dist) in enumerate(paths[:3], 1):
        path_str = " -> ".join(path)
        print(f"  {i}. {path_str} (distance: {dist:.4f})")
    
    print("\n" + "=" * 60)
    print("Domain Similarity Demo Complete!")
    print("=" * 60)
