"""
可解释AI模块 - Explainable AI (XAI)
实现SHAP、LIME、概念激活向量等解释方法
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
from abc import ABC, abstractmethod


class Explanation(ABC):
    """解释基类"""
    
    @abstractmethod
    def explain(self, X: np.ndarray, **kwargs) -> Dict[str, Any]:
        """生成解释"""
        pass
    
    @abstractmethod
    def plot(self, **kwargs):
        """可视化解释"""
        pass


@dataclass
class FeatureImportance:
    """特征重要性"""
    feature_names: List[str]
    importances: np.ndarray
    std: Optional[np.ndarray] = None
    
    def top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """获取最重要的特征"""
        indices = np.argsort(np.abs(self.importances))[::-1][:n]
        return [(self.feature_names[i], self.importances[i]) for i in indices]
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "feature_names": self.feature_names,
            "importances": self.importances.tolist(),
            "std": self.std.tolist() if self.std is not None else None,
            "top_features": self.top_features()
        }


class SHAPExplainer(Explanation):
    """
    SHAP (SHapley Additive exPlanations) 解释器
    基于博弈论的模型解释方法
    """
    
    def __init__(
        self,
        model: Callable,
        background_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ):
        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names
        
        # 尝试导入shap库
        try:
            import shap
            self.shap_available = True
            self.shap = shap
        except ImportError:
            self.shap_available = False
            warnings.warn("shap not installed, using fallback implementation")
    
    def explain(
        self,
        X: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """
        计算SHAP值
        
        Args:
            X: 要解释的样本
        
        Returns:
            SHAP解释结果
        """
        if self.shap_available:
            return self._explain_shap_lib(X, **kwargs)
        else:
            return self._explain_fallback(X, **kwargs)
    
    def _explain_shap_lib(self, X: np.ndarray, **kwargs) -> Dict[str, Any]:
        """使用shap库计算"""
        if self.background_data is None:
            explainer = self.shap.KernelExplainer(self.model, X[:100])
        else:
            explainer = self.shap.KernelExplainer(self.model, self.background_data)
        
        shap_values = explainer.shap_values(X)
        
        # 处理多类别情况
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
        
        # 计算特征重要性
        if shap_values.ndim == 3:
            importances = np.abs(shap_values).mean(axis=(0, 2))
        else:
            importances = np.abs(shap_values).mean(axis=0)
        
        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(importances))]
        
        return {
            "shap_values": shap_values,
            "expected_value": explainer.expected_value,
            "feature_importance": FeatureImportance(
                feature_names,
                importances,
                std=np.std(shap_values, axis=0) if shap_values.ndim == 2 else None
            ),
            "method": "shap_kernel"
        }
    
    def _explain_fallback(
        self,
        X: np.ndarray,
        n_samples: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """备用的简化SHAP实现"""
        n_features = X.shape[1]
        
        # 使用背景数据的均值作为参考
        if self.background_data is not None:
            background_mean = self.background_data.mean(axis=0)
        else:
            background_mean = X.mean(axis=0)
        
        shap_values = np.zeros((len(X), n_features))
        
        # 对每个样本计算近似SHAP值
        for idx, x in enumerate(X):
            # 计算边际贡献
            for feature_idx in range(n_features):
                # 随机 coalition 采样
                contributions = []
                for _ in range(n_samples):
                    # 随机选择其他特征的子集
                    other_features = np.random.rand(n_features) > 0.5
                    other_features[feature_idx] = False
                    
                    # 包含该特征
                    x_with = background_mean.copy()
                    x_with[other_features] = x[other_features]
                    x_with[feature_idx] = x[feature_idx]
                    
                    # 不包含该特征
                    x_without = background_mean.copy()
                    x_without[other_features] = x[other_features]
                    
                    pred_with = self.model(x_with.reshape(1, -1))
                    pred_without = self.model(x_without.reshape(1, -1))
                    
                    contributions.append(pred_with - pred_without)
                
                shap_values[idx, feature_idx] = np.mean(contributions)
        
        importances = np.abs(shap_values).mean(axis=0)
        feature_names = self.feature_names or [f"feature_{i}" for i in range(n_features)]
        
        return {
            "shap_values": shap_values,
            "expected_value": self.model(background_mean.reshape(1, -1)),
            "feature_importance": FeatureImportance(feature_names, importances),
            "method": "shap_fallback"
        }
    
    def plot(self, shap_values: Optional[np.ndarray] = None, X: Optional[np.ndarray] = None, **kwargs):
        """可视化SHAP解释"""
        if not self.shap_available:
            print("SHAP plotting requires shap library")
            return
        
        if shap_values is None:
            result = self.explain(X)
            shap_values = result["shap_values"]
        
        self.shap.summary_plot(shap_values, X, feature_names=self.feature_names)


class LIMEExplainer(Explanation):
    """
    LIME (Local Interpretable Model-agnostic Explanations) 解释器
    局部可解释模型无关解释
    """
    
    def __init__(
        self,
        model: Callable,
        feature_names: Optional[List[str]] = None,
        mode: str = "regression"
    ):
        self.model = model
        self.feature_names = feature_names
        self.mode = mode
        
        try:
            import lime
            import lime.lime_tabular
            self.lime_available = True
            self.lime = lime
        except ImportError:
            self.lime_available = False
            warnings.warn("lime not installed, using fallback implementation")
    
    def explain(
        self,
        X: np.ndarray,
        instance_idx: int = 0,
        num_features: int = 10,
        num_samples: int = 500,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成LIME解释
        
        Args:
            X: 数据集
            instance_idx: 要解释的样本索引
            num_features: 返回的特征数
            num_samples: 扰动样本数
        
        Returns:
            LIME解释结果
        """
        if self.lime_available:
            return self._explain_lime_lib(X, instance_idx, num_features, num_samples)
        else:
            return self._explain_fallback(X[instance_idx], num_features, num_samples)
    
    def _explain_lime_lib(
        self,
        X: np.ndarray,
        instance_idx: int,
        num_features: int,
        num_samples: int
    ) -> Dict[str, Any]:
        """使用lime库"""
        feature_names = self.feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        if self.mode == "regression":
            explainer = self.lime.lime_tabular.LimeTabularExplainer(
                X,
                feature_names=feature_names,
                mode='regression'
            )
        else:
            explainer = self.lime.lime_tabular.LimeTabularExplainer(
                X,
                feature_names=feature_names,
                mode='classification'
            )
        
        exp = explainer.explain_instance(
            X[instance_idx],
            self.model,
            num_features=num_features,
            num_samples=num_samples
        )
        
        # 提取特征重要性
        feature_weights = exp.as_list()
        importances = np.zeros(len(feature_names))
        
        for feature_desc, weight in feature_weights:
            # 解析特征描述提取索引
            for i, name in enumerate(feature_names):
                if name in feature_desc:
                    importances[i] = weight
                    break
        
        return {
            "explanation": exp,
            "feature_weights": feature_weights,
            "feature_importance": FeatureImportance(feature_names, importances),
            "local_prediction": exp.local_pred,
            "method": "lime"
        }
    
    def _explain_fallback(
        self,
        x: np.ndarray,
        num_features: int,
        num_samples: int
    ) -> Dict[str, Any]:
        """备用的简化LIME实现"""
        n_features = len(x)
        
        # 生成扰动样本
        np.random.seed(42)
        perturbations = np.random.normal(0, 0.1, (num_samples, n_features))
        perturbed_samples = x + perturbations
        
        # 预测
        predictions = np.array([self.model(s.reshape(1, -1))[0] for s in perturbed_samples])
        original_pred = self.model(x.reshape(1, -1))[0]
        
        # 计算权重（基于距离）
        distances = np.linalg.norm(perturbations, axis=1)
        weights = np.exp(-distances / np.median(distances))
        
        # 加权线性回归
        X_perturbed = np.column_stack([np.ones(num_samples), perturbations])
        W = np.diag(weights)
        
        # 正规方程
        beta = np.linalg.lstsq(X_perturbed.T @ W @ X_perturbed, 
                               X_perturbed.T @ W @ (predictions - original_pred),
                               rcond=None)[0]
        
        coefficients = beta[1:]  # 排除截距
        
        # 取最重要的特征
        top_indices = np.argsort(np.abs(coefficients))[::-1][:num_features]
        
        feature_names = self.feature_names or [f"feature_{i}" for i in range(n_features)]
        
        return {
            "coefficients": coefficients,
            "intercept": beta[0],
            "feature_weights": [(feature_names[i], coefficients[i]) for i in top_indices],
            "feature_importance": FeatureImportance(feature_names, coefficients),
            "local_prediction": original_pred + beta[0],
            "method": "lime_fallback"
        }
    
    def plot(self, explanation: Optional[Dict] = None, **kwargs):
        """可视化LIME解释"""
        if explanation and "explanation" in explanation:
            exp = explanation["explanation"]
            exp.show_in_notebook()
        else:
            print("LIME visualization requires lime library")


class ConceptActivationVector:
    """
    CAV (Concept Activation Vector)
    概念激活向量，用于概念级别的解释
    """
    
    def __init__(
        self,
        layer_name: str,
        concept_names: List[str],
        bottleneck_dim: int = 128
    ):
        self.layer_name = layer_name
        self.concept_names = concept_names
        self.bottleneck_dim = bottleneck_dim
        self.concept_classifiers: Dict[str, Any] = {}
        self.cavs: Dict[str, np.ndarray] = {}
    
    def train_concept(
        self,
        concept_name: str,
        positive_examples: np.ndarray,
        negative_examples: np.ndarray
    ):
        """
        训练概念分类器
        
        Args:
            concept_name: 概念名称
            positive_examples: 正例激活 [n_pos, bottleneck_dim]
            negative_examples: 负例激活 [n_neg, bottleneck_dim]
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        
        # 准备数据
        X = np.vstack([positive_examples, negative_examples])
        y = np.concatenate([
            np.ones(len(positive_examples)),
            np.zeros(len(negative_examples))
        ])
        
        # 训练线性分类器
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X, y)
        
        self.concept_classifiers[concept_name] = clf
        
        # 提取CAV（权重向量）
        cav = clf.coef_[0]
        cav = cav / np.linalg.norm(cav)  # 归一化
        self.cavs[concept_name] = cav
    
    def get_concept_sensitivity(
        self,
        activations: np.ndarray,
        concept_name: str,
        model_gradient: Optional[Callable] = None
    ) -> float:
        """
        计算概念敏感度
        
        Args:
            activations: 激活值
            concept_name: 概念名称
            model_gradient: 模型梯度函数
        
        Returns:
            概念敏感度分数
        """
        if concept_name not in self.cavs:
            raise ValueError(f"Concept {concept_name} not trained")
        
        cav = self.cavs[concept_name]
        
        if model_gradient is None:
            # 使用激活值和CAV的点积作为敏感度
            sensitivity = np.dot(activations, cav)
        else:
            # 使用梯度计算TCAV分数
            gradient = model_gradient(activations)
            sensitivity = np.dot(gradient, cav)
        
        return float(sensitivity)
    
    def tcav_score(
        self,
        concept_name: str,
        class_id: int,
        activations: np.ndarray,
        model_gradient: Callable
    ) -> float:
        """
        计算TCAV (Testing with CAV) 分数
        
        Args:
            concept_name: 概念名称
            class_id: 类别ID
            activations: 激活值
            model_gradient: 模型梯度函数
        
        Returns:
            TCAV分数 [0, 1]
        """
        if concept_name not in self.cavs:
            raise ValueError(f"Concept {concept_name} not trained")
        
        cav = self.cavs[concept_name]
        
        # 计算每个样本的敏感度方向
        positive_count = 0
        for act in activations:
            gradient = model_gradient(act.reshape(1, -1))
            if np.dot(gradient.flatten(), cav) > 0:
                positive_count += 1
        
        tcav = positive_count / len(activations)
        return tcav
    
    def explain_instance(
        self,
        activation: np.ndarray,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        解释单个实例
        
        Args:
            activation: 激活值
            top_k: 返回前k个概念
        
        Returns:
            概念及其相关性列表
        """
        scores = []
        for concept_name, cav in self.cavs.items():
            score = np.dot(activation, cav)
            scores.append((concept_name, float(score)))
        
        scores.sort(key=lambda x: abs(x[1]), reverse=True)
        return scores[:top_k]


class IntegratedGradients:
    """
    积分梯度法
    基于路径的属性归因方法
    """
    
    def __init__(
        self,
        model: Callable,
        baseline: Optional[np.ndarray] = None
    ):
        self.model = model
        self.baseline = baseline
    
    def explain(
        self,
        X: np.ndarray,
        n_steps: int = 50,
        **kwargs
    ) -> Dict[str, Any]:
        """
        计算积分梯度
        
        Args:
            X: 输入样本
            n_steps: 积分步数
        
        Returns:
            归因结果
        """
        if self.baseline is None:
            baseline = np.zeros_like(X)
        else:
            baseline = self.baseline
        
        # 生成路径点
        alphas = np.linspace(0, 1, n_steps + 1)
        path_points = np.array([
            baseline + alpha * (X - baseline)
            for alpha in alphas
        ])
        
        # 计算梯度
        gradients = []
        for point in path_points:
            grad = self._compute_gradient(point)
            gradients.append(grad)
        
        gradients = np.array(gradients)
        
        # 近似积分
        avg_gradients = (gradients[:-1] + gradients[1:]) / 2
        integrated_gradients = (X - baseline) * avg_gradients.mean(axis=0)
        
        return {
            "attributions": integrated_gradients,
            "baseline": baseline,
            "n_steps": n_steps,
            "method": "integrated_gradients"
        }
    
    def _compute_gradient(self, x: np.ndarray) -> np.ndarray:
        """计算梯度（数值近似）"""
        epsilon = 1e-4
        gradient = np.zeros_like(x)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += epsilon
            x_minus[i] -= epsilon
            
            f_plus = self.model(x_plus.reshape(1, -1))[0]
            f_minus = self.model(x_minus.reshape(1, -1))[0]
            
            gradient[i] = (f_plus - f_minus) / (2 * epsilon)
        
        return gradient


class ExplanationAggregator:
    """
    解释聚合器
    整合多种解释方法的结果
    """
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names
        self.explanations: Dict[str, Dict] = {}
    
    def add_explanation(self, method: str, explanation: Dict):
        """添加解释结果"""
        self.explanations[method] = explanation
    
    def aggregate(
        self,
        weights: Optional[Dict[str, float]] = None
    ) -> FeatureImportance:
        """
        聚合解释结果
        
        Args:
            weights: 各方法的权重
        
        Returns:
            聚合后的特征重要性
        """
        if not self.explanations:
            raise ValueError("No explanations to aggregate")
        
        if weights is None:
            weights = {method: 1.0 for method in self.explanations.keys()}
        
        # 归一化权重
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # 收集所有特征重要性
        all_importances = []
        for method, exp in self.explanations.items():
            if "feature_importance" in exp:
                fi = exp["feature_importance"]
                # 归一化重要性
                normalized = fi.importances / (np.abs(fi.importances).sum() + 1e-10)
                all_importances.append(weights[method] * normalized)
        
        if not all_importances:
            raise ValueError("No feature importances found")
        
        # 加权平均
        aggregated = np.sum(all_importances, axis=0)
        
        n_features = len(aggregated)
        feature_names = self.feature_names or [f"feature_{i}" for i in range(n_features)]
        
        return FeatureImportance(feature_names, aggregated)
    
    def consensus_features(self, threshold: float = 0.5) -> List[str]:
        """
        找出各方法一致的重要特征
        
        Args:
            threshold: 一致性阈值
        
        Returns:
            一致重要的特征名列表
        """
        if not self.explanations:
            return []
        
        # 获取每个方法的前k个特征
        top_features_per_method = {}
        for method, exp in self.explanations.items():
            if "feature_importance" in exp:
                top_features = exp["feature_importance"].top_features()
                top_features_per_method[method] = set(f[0] for f in top_features)
        
        # 计算一致性
        feature_votes = defaultdict(int)
        for features in top_features_per_method.values():
            for f in features:
                feature_votes[f] += 1
        
        n_methods = len(top_features_per_method)
        consensus = [
            f for f, votes in feature_votes.items()
            if votes / n_methods >= threshold
        ]
        
        return sorted(consensus, key=lambda x: feature_votes[x], reverse=True)


class ExplainableAI:
    """
    可解释AI主类
    整合多种解释方法
    """
    
    def __init__(self, model: Callable, feature_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names
        self.explainers: Dict[str, Explanation] = {}
        self.cav: Optional[ConceptActivationVector] = None
    
    def add_shap(self, background_data: Optional[np.ndarray] = None):
        """添加SHAP解释器"""
        self.explainers["shap"] = SHAPExplainer(
            self.model, background_data, self.feature_names
        )
    
    def add_lime(self, mode: str = "regression"):
        """添加LIME解释器"""
        self.explainers["lime"] = LIMEExplainer(
            self.model, self.feature_names, mode
        )
    
    def add_integrated_gradients(self, baseline: Optional[np.ndarray] = None):
        """添加积分梯度解释器"""
        self.explainers["ig"] = IntegratedGradients(self.model, baseline)
    
    def setup_cav(
        self,
        layer_name: str,
        concept_names: List[str],
        bottleneck_dim: int = 128
    ):
        """设置CAV"""
        self.cav = ConceptActivationVector(
            layer_name, concept_names, bottleneck_dim
        )
    
    def explain(
        self,
        X: np.ndarray,
        methods: Optional[List[str]] = None,
        aggregate: bool = False
    ) -> Union[Dict[str, Dict], FeatureImportance]:
        """
        生成解释
        
        Args:
            X: 输入数据
            methods: 使用的解释方法
            aggregate: 是否聚合结果
        
        Returns:
            解释结果
        """
        if methods is None:
            methods = list(self.explainers.keys())
        
        results = {}
        for method in methods:
            if method in self.explainers:
                results[method] = self.explainers[method].explain(X)
        
        if aggregate and len(results) > 1:
            aggregator = ExplanationAggregator(self.feature_names)
            for method, result in results.items():
                aggregator.add_explanation(method, result)
            return aggregator.aggregate()
        
        return results
    
    def explain_with_cav(
        self,
        activations: np.ndarray,
        concept_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        使用CAV解释
        
        Args:
            activations: 网络激活值
            concept_name: 特定概念名称（None则返回所有）
        
        Returns:
            概念解释
        """
        if self.cav is None:
            raise ValueError("CAV not set up")
        
        if concept_name:
            score = self.cav.get_concept_sensitivity(activations, concept_name)
            return {concept_name: score}
        else:
            results = {}
            for name in self.cav.concept_names:
                results[name] = self.cav.get_concept_sensitivity(activations, name)
            return results
    
    def generate_report(self, X: np.ndarray) -> Dict[str, Any]:
        """生成完整的可解释性报告"""
        report = {
            "model_type": type(self.model).__name__,
            "input_shape": X.shape,
            "feature_names": self.feature_names,
        }
        
        # 收集所有解释
        explanations = self.explain(X)
        
        report["explanations"] = {}
        for method, result in explanations.items():
            if "feature_importance" in result:
                report["explanations"][method] = {
                    "top_features": result["feature_importance"].top_features(10),
                    "method": result.get("method", method)
                }
        
        # 如果有多于一种方法，计算一致性
        if len(explanations) > 1:
            aggregator = ExplanationAggregator(self.feature_names)
            for method, result in explanations.items():
                aggregator.add_explanation(method, result)
            
            report["consensus_features"] = aggregator.consensus_features()
            report["aggregated_importance"] = aggregator.aggregate().to_dict()
        
        return report


def demo():
    """演示可解释AI模块"""
    print("=" * 60)
    print("可解释AI模块演示")
    print("=" * 60)
    
    # 创建简单模型
    def simple_model(X: np.ndarray) -> np.ndarray:
        """简单的线性模型"""
        weights = np.array([2.0, -1.5, 0.5, 0.0, 3.0])
        return X @ weights + 1.0
    
    # 生成数据
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 5)
    feature_names = ["temperature", "pressure", "humidity", "wind", "energy"]
    
    print(f"\n数据形状: {X.shape}")
    print(f"特征: {feature_names}")
    
    # 创建解释器
    xai = ExplainableAI(simple_model, feature_names)
    
    print("\n1. SHAP解释")
    xai.add_shap(background_data=X[:50])
    shap_result = xai.explain(X[:5], methods=["shap"])
    print(f"   SHAP值形状: {shap_result['shap']['shap_values'].shape}")
    print(f"   顶级特征: {shap_result['shap']['feature_importance'].top_features(3)}")
    
    print("\n2. LIME解释")
    xai.add_lime(mode="regression")
    lime_result = xai.explain(X, methods=["lime"])
    print(f"   局部预测: {lime_result['lime']['local_prediction']:.4f}")
    print(f"   特征权重: {lime_result['lime']['feature_weights'][:3]}")
    
    print("\n3. 积分梯度")
    xai.add_integrated_gradients()
    ig_result = xai.explain(X[:1], methods=["ig"])
    print(f"   归因形状: {ig_result['ig']['attributions'].shape}")
    print(f"   归因和: {ig_result['ig']['attributions'].sum():.4f}")
    
    print("\n4. 聚合解释")
    aggregated = xai.explain(X[:5], aggregate=True)
    print(f"   聚合后的顶级特征: {aggregated.top_features(5)}")
    
    print("\n5. CAV概念解释")
    xai.setup_cav("layer1", ["high_temp", "low_pressure", "stable"])
    
    # 模拟训练CAV
    np.random.seed(42)
    for concept in ["high_temp", "low_pressure", "stable"]:
        pos = np.random.randn(50, 128)
        neg = np.random.randn(50, 128)
        xai.cav.train_concept(concept, pos, neg)
    
    # 解释实例
    activation = np.random.randn(128)
    cav_result = xai.explain_with_cav(activation)
    print(f"   概念敏感度: {cav_result}")
    
    print("\n6. 生成报告")
    report = xai.generate_report(X[:10])
    print(f"   报告包含 {len(report['explanations'])} 种解释方法")
    print(f"   一致特征: {report.get('consensus_features', [])}")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
