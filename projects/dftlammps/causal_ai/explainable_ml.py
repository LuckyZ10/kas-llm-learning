"""
可解释机器学习模块 - Explainable ML for Materials

本模块实现材料科学中的可解释AI方法：
- SHAP/LIME解释
- 注意力可视化
- 概念激活向量
- 特征重要性分析

作者: Causal AI Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
from collections import defaultdict
import itertools

# 尝试导入可视化库
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


@dataclass
class FeatureImportance:
    """特征重要性"""
    feature_name: str
    importance: float
    std: float = 0.0
    method: str = ""
    
    def __repr__(self):
        return f"{self.feature_name}: {self.importance:.4f} ± {self.std:.4f}"


@dataclass
class Explanation:
    """解释结果"""
    instance_id: int
    base_value: float
    prediction: float
    feature_contributions: Dict[str, float]
    method: str
    metadata: Dict = field(default_factory=dict)
    
    def top_features(self, n: int = 5) -> List[Tuple[str, float]]:
        """获取最重要的n个特征"""
        sorted_features = sorted(
            self.feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_features[:n]
    
    def plot(self, ax=None, **kwargs):
        """可视化解释"""
        if not HAS_MATPLOTLIB:
            warnings.warn("matplotlib not available")
            return None
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        features, values = zip(*self.top_features(n=kwargs.get('n_features', 10)))
        colors = ['green' if v > 0 else 'red' for v in values]
        
        y_pos = np.arange(len(features))
        ax.barh(y_pos, values, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Contribution to Prediction')
        ax.set_title(f'{self.method} Explanation (Prediction: {self.prediction:.3f})')
        
        plt.tight_layout()
        return ax


class SHAPExplainer:
    """
    SHAP解释器
    
    基于博弈论的特征重要性解释方法
    """
    
    def __init__(self, model: Callable, 
                 background_data: np.ndarray = None,
                 feature_names: List[str] = None):
        """
        初始化SHAP解释器
        
        Args:
            model: 预测模型（支持sklearn和pytorch/tensorflow）
            background_data: 背景数据（用于估计基线）
            feature_names: 特征名称
        """
        self.model = model
        self.background = background_data
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
        if HAS_SHAP:
            self._init_explainer()
    
    def _init_explainer(self):
        """初始化SHAP解释器"""
        if not HAS_SHAP:
            warnings.warn("SHAP not installed")
            return
        
        try:
            # 尝试不同类型的解释器
            import sklearn
            if hasattr(self.model, 'predict'):
                if self.background is not None:
                    self.explainer = shap.KernelExplainer(
                        self.model.predict, 
                        self.background
                    )
                else:
                    self.explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            warnings.warn(f"Failed to initialize SHAP explainer: {e}")
            # 回退到自定义实现
            self.explainer = None
    
    def explain(self, X: np.ndarray, instance_idx: int = 0) -> Explanation:
        """
        解释单个实例
        
        Args:
            X: 输入数据
            instance_idx: 要解释的实例索引
            
        Returns:
            解释结果
        """
        if HAS_SHAP and self.explainer is not None:
            return self._explain_shap(X, instance_idx)
        else:
            return self._explain_approximate(X, instance_idx)
    
    def _explain_shap(self, X: np.ndarray, instance_idx: int) -> Explanation:
        """使用SHAP库解释"""
        shap_values = self.explainer.shap_values(X[instance_idx:instance_idx+1])
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # 二分类问题
        
        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0]
        
        prediction = self.model.predict(X[instance_idx:instance_idx+1])
        if hasattr(prediction, 'flatten'):
            prediction = prediction.flatten()[0]
        
        feature_names = self.feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        contributions = {name: float(shap_values[0, i]) 
                        for i, name in enumerate(feature_names)}
        
        return Explanation(
            instance_id=instance_idx,
            base_value=float(base_value),
            prediction=float(prediction),
            feature_contributions=contributions,
            method="SHAP"
        )
    
    def _explain_approximate(self, X: np.ndarray, instance_idx: int) -> Explanation:
        """近似SHAP值计算（简化版）"""
        instance = X[instance_idx]
        n_features = len(instance)
        
        # 基线预测
        if self.background is not None:
            base_value = np.mean(self.model.predict(self.background))
        else:
            base_value = np.mean(self.model.predict(X))
        
        prediction = self.model.predict(instance.reshape(1, -1))[0]
        
        # 计算边际贡献（简化版）
        contributions = {}
        feature_names = self.feature_names or [f"feature_{i}" for i in range(n_features)]
        
        for i, name in enumerate(feature_names):
            # 特征置换
            X_perm = X.copy()
            np.random.shuffle(X_perm[:, i])
            
            # 边际贡献
            pred_with = self.model.predict(instance.reshape(1, -1))[0]
            pred_without = np.mean(self.model.predict(X_perm))
            
            contributions[name] = pred_with - pred_without
        
        # 归一化使其总和等于预测差
        total_contrib = sum(contributions.values())
        if abs(total_contrib) > 1e-10:
            scale = (prediction - base_value) / total_contrib
            contributions = {k: v * scale for k, v in contributions.items()}
        
        return Explanation(
            instance_id=instance_idx,
            base_value=float(base_value),
            prediction=float(prediction),
            feature_contributions=contributions,
            method="Approximate SHAP"
        )
    
    def explain_batch(self, X: np.ndarray) -> List[Explanation]:
        """批量解释"""
        return [self.explain(X, i) for i in range(len(X))]
    
    def global_importance(self, X: np.ndarray) -> List[FeatureImportance]:
        """
        计算全局特征重要性
        
        Args:
            X: 输入数据
            
        Returns:
            特征重要性列表
        """
        explanations = self.explain_batch(X)
        
        # 聚合特征重要性
        feature_importance = defaultdict(list)
        for exp in explanations:
            for feat, contrib in exp.feature_contributions.items():
                feature_importance[feat].append(abs(contrib))
        
        results = []
        for feat, values in feature_importance.items():
            results.append(FeatureImportance(
                feature_name=feat,
                importance=np.mean(values),
                std=np.std(values),
                method="SHAP"
            ))
        
        return sorted(results, key=lambda x: x.importance, reverse=True)
    
    def dependence_plot(self, feature: str, X: np.ndarray, 
                       interaction_feature: str = None, ax=None):
        """
        绘制SHAP依赖图
        
        Args:
            feature: 特征名
            X: 数据
            interaction_feature: 交互特征
            ax: matplotlib轴
        """
        if not HAS_MATPLOTLIB:
            warnings.warn("matplotlib not available")
            return None
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # 计算SHAP值
        explanations = self.explain_batch(X)
        feature_values = X[:, self.feature_names.index(feature)] if self.feature_names else X[:, 0]
        shap_values = [exp.feature_contributions[feature] for exp in explanations]
        
        if interaction_feature:
            interact_idx = self.feature_names.index(interaction_feature) if self.feature_names else 1
            colors = X[:, interact_idx]
            scatter = ax.scatter(feature_values, shap_values, c=colors, 
                               cmap='viridis', alpha=0.5)
            plt.colorbar(scatter, ax=ax, label=interaction_feature)
        else:
            ax.scatter(feature_values, shap_values, alpha=0.5)
        
        ax.set_xlabel(feature)
        ax.set_ylabel(f'SHAP value ({feature})')
        ax.set_title(f'SHAP Dependence Plot: {feature}')
        
        return ax
    
    def summary_plot(self, X: np.ndarray, max_display: int = 10, ax=None):
        """
        SHAP摘要图
        
        Args:
            X: 数据
            max_display: 显示的最大特征数
            ax: matplotlib轴
        """
        if not HAS_MATPLOTLIB:
            warnings.warn("matplotlib not available")
            return None
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        explanations = self.explain_batch(X)
        
        # 收集所有SHAP值
        all_shap = []
        for exp in explanations:
            all_shap.append(exp.feature_contributions)
        
        shap_df = pd.DataFrame(all_shap)
        
        # 计算特征重要性
        mean_abs_shap = shap_df.abs().mean().sort_values(ascending=True)
        mean_abs_shap = mean_abs_shap.tail(max_display)
        
        # 绘制
        for i, feature in enumerate(mean_abs_shap.index):
            values = shap_df[feature].values
            y_pos = np.ones(len(values)) * i
            colors = ['red' if v < 0 else 'blue' for v in values]
            ax.scatter(values, y_pos, c=colors, alpha=0.3, s=20)
        
        ax.set_yticks(range(len(mean_abs_shap)))
        ax.set_yticklabels(mean_abs_shap.index)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('SHAP value')
        ax.set_title('SHAP Summary Plot')
        
        return ax


class LIMEExplainer:
    """
    LIME解释器 (Local Interpretable Model-agnostic Explanations)
    
    局部可解释模型无关解释
    """
    
    def __init__(self, model: Callable,
                 feature_names: List[str] = None,
                 categorical_features: List[int] = None,
                 kernel_width: float = None):
        """
        初始化LIME解释器
        
        Args:
            model: 预测模型
            feature_names: 特征名称
            categorical_features: 分类特征索引
            kernel_width: 核宽度
        """
        self.model = model
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.kernel_width = kernel_width
        
    def explain(self, X: np.ndarray, instance_idx: int = 0,
               n_samples: int = 1000, n_features: int = 10) -> Explanation:
        """
        解释单个实例
        
        Args:
            X: 输入数据
            instance_idx: 要解释的实例索引
            n_samples: 扰动样本数
            n_features: 选择的特征数
            
        Returns:
            解释结果
        """
        instance = X[instance_idx]
        n_features_total = len(instance)
        
        # 生成扰动样本
        perturbed_samples = self._generate_perturbations(
            instance, X, n_samples
        )
        
        # 获取预测
        predictions = self.model.predict(perturbed_samples)
        
        # 计算距离权重
        distances = self._compute_distances(instance, perturbed_samples)
        weights = self._kernel(distances)
        
        # 拟合局部线性模型
        coefficients = self._fit_local_linear(
            perturbed_samples, predictions, weights, n_features
        )
        
        # 构建解释
        feature_names = self.feature_names or [f"feature_{i}" for i in range(n_features_total)]
        contributions = {name: float(coefficients.get(i, 0)) 
                        for i, name in enumerate(feature_names)}
        
        base_value = np.mean(predictions)
        prediction = self.model.predict(instance.reshape(1, -1))[0]
        
        return Explanation(
            instance_id=instance_idx,
            base_value=float(base_value),
            prediction=float(prediction),
            feature_contributions=contributions,
            method="LIME",
            metadata={'n_samples': n_samples, 'kernel_width': self.kernel_width}
        )
    
    def _generate_perturbations(self, instance: np.ndarray,
                                background: np.ndarray,
                                n_samples: int) -> np.ndarray:
        """生成扰动样本"""
        n_features = len(instance)
        samples = np.zeros((n_samples, n_features))
        
        for i in range(n_features):
            if i in self.categorical_features:
                # 分类特征：从背景中采样
                samples[:, i] = np.random.choice(background[:, i], n_samples)
            else:
                # 连续特征：基于标准差采样
                std = np.std(background[:, i])
                samples[:, i] = instance[i] + np.random.normal(0, std, n_samples)
        
        # 第一个样本是原始实例
        samples[0] = instance
        
        return samples
    
    def _compute_distances(self, instance: np.ndarray,
                          samples: np.ndarray) -> np.ndarray:
        """计算欧氏距离"""
        # 标准化
        std = np.std(samples, axis=0) + 1e-10
        normalized_instance = instance / std
        normalized_samples = samples / std
        
        # 欧氏距离
        distances = np.sqrt(np.sum((normalized_samples - normalized_instance) ** 2, axis=1))
        return distances
    
    def _kernel(self, distances: np.ndarray) -> np.ndarray:
        """指数核函数"""
        if self.kernel_width is None:
            self.kernel_width = np.sqrt(len(distances)) * 0.75
        
        return np.exp(-distances ** 2 / (2 * self.kernel_width ** 2))
    
    def _fit_local_linear(self, X: np.ndarray, y: np.ndarray,
                         weights: np.ndarray, n_features: int) -> Dict[int, float]:
        """拟合加权线性模型"""
        # 标准化
        X_mean = np.average(X, axis=0, weights=weights)
        X_std = np.sqrt(np.average((X - X_mean) ** 2, axis=0, weights=weights)) + 1e-10
        X_normalized = (X - X_mean) / X_std
        
        # 加权最小二乘
        W = np.diag(weights)
        try:
            beta = np.linalg.lstsq(X_normalized.T @ W @ X_normalized,
                                  X_normalized.T @ W @ y, rcond=None)[0]
        except:
            # 如果矩阵奇异，使用岭回归
            beta = np.linalg.lstsq(X_normalized.T @ W @ X_normalized + 0.1 * np.eye(X.shape[1]),
                                  X_normalized.T @ W @ y, rcond=None)[0]
        
        # 选择最重要的特征
        feature_importance = [(i, abs(beta[i])) for i in range(len(beta))]
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        selected = {i: beta[i] for i, _ in feature_importance[:n_features]}
        return selected
    
    def explain_batch(self, X: np.ndarray, **kwargs) -> List[Explanation]:
        """批量解释"""
        return [self.explain(X, i, **kwargs) for i in range(len(X))]


class AttentionVisualizer:
    """
    注意力可视化器
    
    用于分析神经网络中的注意力权重
    """
    
    def __init__(self, model: Any, attention_layer_name: str = None):
        """
        初始化注意力可视化器
        
        Args:
            model: 神经网络模型（PyTorch/TensorFlow）
            attention_layer_name: 注意力层名称
        """
        self.model = model
        self.attention_layer_name = attention_layer_name
        self.attention_weights = None
        
    def extract_attention(self, X: np.ndarray) -> np.ndarray:
        """
        提取注意力权重
        
        Args:
            X: 输入数据
            
        Returns:
            注意力权重矩阵
        """
        # 检测模型类型并相应处理
        model_type = self._detect_model_type()
        
        if model_type == "pytorch":
            return self._extract_pytorch_attention(X)
        elif model_type == "tensorflow":
            return self._extract_tensorflow_attention(X)
        else:
            raise ValueError("Unsupported model type")
    
    def _detect_model_type(self) -> str:
        """检测模型类型"""
        try:
            import torch
            if isinstance(self.model, torch.nn.Module):
                return "pytorch"
        except ImportError:
            pass
        
        try:
            import tensorflow as tf
            if isinstance(self.model, tf.keras.Model):
                return "tensorflow"
        except ImportError:
            pass
        
        return "unknown"
    
    def _extract_pytorch_attention(self, X: np.ndarray) -> np.ndarray:
        """提取PyTorch模型的注意力权重"""
        import torch
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X)
        
        attention_weights = []
        
        def hook_fn(module, input, output):
            # 假设输出包含注意力权重
            if isinstance(output, tuple):
                attention_weights.append(output[1].detach().numpy())
            else:
                attention_weights.append(output.detach().numpy())
        
        # 注册hook
        hooks = []
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or name == self.attention_layer_name:
                hooks.append(module.register_forward_hook(hook_fn))
        
        # 前向传播
        with torch.no_grad():
            _ = self.model(X_tensor)
        
        # 移除hooks
        for hook in hooks:
            hook.remove()
        
        if len(attention_weights) > 0:
            return np.array(attention_weights[0])
        else:
            return np.zeros((len(X), X.shape[1], X.shape[1]))
    
    def _extract_tensorflow_attention(self, X: np.ndarray) -> np.ndarray:
        """提取TensorFlow模型的注意力权重"""
        import tensorflow as tf
        
        # 创建模型以获取中间层输出
        layer_outputs = []
        layer_names = []
        
        for layer in self.model.layers:
            if 'attention' in layer.name.lower() or layer.name == self.attention_layer_name:
                layer_outputs.append(layer.output)
                layer_names.append(layer.name)
        
        if len(layer_outputs) == 0:
            return np.zeros((len(X), X.shape[1], X.shape[1]))
        
        # 创建提取模型
        activation_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=layer_outputs
        )
        
        # 获取注意力权重
        activations = activation_model.predict(X)
        
        if isinstance(activations, list):
            return activations[0]
        return activations
    
    def visualize_attention_map(self, attention_weights: np.ndarray,
                                tokens: List[str] = None,
                                ax=None, **kwargs):
        """
        可视化注意力热力图
        
        Args:
            attention_weights: 注意力权重矩阵 [batch, seq_len, seq_len]
            tokens: 标记列表
            ax: matplotlib轴
        """
        if not HAS_MATPLOTLIB:
            warnings.warn("matplotlib not available")
            return None
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # 取第一个样本的平均注意力
        if len(attention_weights.shape) == 3:
            attn_map = attention_weights[0]
        else:
            attn_map = attention_weights
        
        # 平均多头注意力
        if len(attn_map.shape) == 3:
            attn_map = np.mean(attn_map, axis=0)
        
        im = ax.imshow(attn_map, cmap='viridis', aspect='auto')
        
        if tokens:
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right')
            ax.set_yticklabels(tokens)
        
        plt.colorbar(im, ax=ax)
        ax.set_title(kwargs.get('title', 'Attention Map'))
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
        
        return ax
    
    def visualize_attention_flow(self, attention_weights: np.ndarray,
                                 tokens: List[str] = None,
                                 ax=None, **kwargs):
        """
        可视化注意力流
        
        Args:
            attention_weights: 注意力权重
            tokens: 标记列表
            ax: matplotlib轴
        """
        if not HAS_MATPLOTLIB:
            warnings.warn("matplotlib not available")
            return None
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        # 计算每个位置的总注意力
        if len(attention_weights.shape) == 3:
            attn = attention_weights[0]
        else:
            attn = attention_weights
        
        if len(attn.shape) == 3:
            attn = np.mean(attn, axis=0)
        
        row_sums = attn.sum(axis=1)
        col_sums = attn.sum(axis=0)
        
        x = np.arange(len(row_sums))
        
        ax.bar(x - 0.2, row_sums, 0.4, label='Outgoing (Query)', alpha=0.7)
        ax.bar(x + 0.2, col_sums, 0.4, label='Incoming (Key)', alpha=0.7)
        
        if tokens:
            ax.set_xticks(x)
            ax.set_xticklabels(tokens, rotation=45, ha='right')
        
        ax.set_xlabel('Position')
        ax.set_ylabel('Attention Weight')
        ax.set_title('Attention Flow')
        ax.legend()
        
        return ax


class ConceptActivationVectors:
    """
    概念激活向量 (CAV)
    
    用于解释神经网络中的概念层次
    """
    
    def __init__(self, model: Any, layer_name: str = None):
        """
        初始化CAV分析器
        
        Args:
            model: 神经网络模型
            layer_name: 目标层名称
        """
        self.model = model
        self.layer_name = layer_name
        self.concepts: Dict[str, Dict] = {}
        self.cavs: Dict[str, np.ndarray] = {}
        
    def define_concept(self, concept_name: str, 
                      positive_examples: np.ndarray,
                      negative_examples: np.ndarray = None):
        """
        定义概念
        
        Args:
            concept_name: 概念名称
            positive_examples: 正例激活
            negative_examples: 负例激活
        """
        self.concepts[concept_name] = {
            'positive': positive_examples,
            'negative': negative_examples
        }
    
    def train_cav(self, concept_name: str, method: str = "linear") -> np.ndarray:
        """
        训练概念激活向量
        
        Args:
            concept_name: 概念名称
            method: 分类方法 (linear, svm)
            
        Returns:
            CAV向量
        """
        concept = self.concepts[concept_name]
        positive = concept['positive']
        
        if concept['negative'] is not None:
            negative = concept['negative']
        else:
            # 使用随机负例
            negative = np.random.randn(*positive.shape)
        
        # 创建标签
        X = np.vstack([positive, negative])
        y = np.array([1] * len(positive) + [0] * len(negative))
        
        # 训练分类器
        if method == "linear":
            from sklearn.linear_model import SGDClassifier
            classifier = SGDClassifier(loss='log_loss', random_state=42)
        else:
            from sklearn.svm import LinearSVC
            classifier = LinearSVC(random_state=42)
        
        classifier.fit(X, y)
        
        # CAV是决策边界的法向量
        if hasattr(classifier, 'coef_'):
            cav = classifier.coef_[0]
        else:
            cav = classifier.coef_[0]
        
        # 归一化
        cav = cav / (np.linalg.norm(cav) + 1e-10)
        
        self.cavs[concept_name] = cav
        return cav
    
    def compute_tcav_score(self, concept_name: str,
                          input_activations: np.ndarray,
                          gradients: np.ndarray) -> float:
        """
        计算TCAV分数 (Testing with Concept Activation Vectors)
        
        Args:
            concept_name: 概念名称
            input_activations: 输入激活
            gradients: 关于输入的梯度
            
        Returns:
            TCAV分数
        """
        if concept_name not in self.cavs:
            raise ValueError(f"CAV for '{concept_name}' not trained. Call train_cav() first.")
        
        cav = self.cavs[concept_name]
        
        # 计算概念敏感度
        directional_derivatives = gradients @ cav
        
        # TCAV分数是正方向导数的比例
        tcav_score = np.mean(directional_derivatives > 0)
        
        return tcav_score
    
    def concept_importance(self, input_data: np.ndarray,
                          concept_names: List[str] = None) -> Dict[str, float]:
        """
        计算概念重要性
        
        Args:
            input_data: 输入数据
            concept_names: 要评估的概念列表
            
        Returns:
            概念重要性字典
        """
        if concept_names is None:
            concept_names = list(self.cavs.keys())
        
        # 获取激活和梯度
        activations = self._get_activations(input_data)
        gradients = self._get_gradients(input_data)
        
        importance = {}
        for concept in concept_names:
            if concept in self.cavs:
                score = self.compute_tcav_score(concept, activations, gradients)
                importance[concept] = score
        
        return importance
    
    def _get_activations(self, X: np.ndarray) -> np.ndarray:
        """获取层激活"""
        model_type = self._detect_model_type()
        
        if model_type == "pytorch":
            return self._get_pytorch_activations(X)
        elif model_type == "tensorflow":
            return self._get_tensorflow_activations(X)
        else:
            return X
    
    def _get_gradients(self, X: np.ndarray) -> np.ndarray:
        """获取梯度"""
        model_type = self._detect_model_type()
        
        if model_type == "pytorch":
            return self._get_pytorch_gradients(X)
        elif model_type == "tensorflow":
            return self._get_tensorflow_gradients(X)
        else:
            return np.zeros_like(X)
    
    def _detect_model_type(self) -> str:
        """检测模型类型"""
        try:
            import torch
            if isinstance(self.model, torch.nn.Module):
                return "pytorch"
        except ImportError:
            pass
        
        try:
            import tensorflow as tf
            if isinstance(self.model, tf.keras.Model):
                return "tensorflow"
        except ImportError:
            pass
        
        return "unknown"
    
    def _get_pytorch_activations(self, X: np.ndarray) -> np.ndarray:
        """获取PyTorch激活"""
        import torch
        
        activations = []
        
        def hook_fn(module, input, output):
            activations.append(output.detach().numpy())
        
        # 注册hook
        handle = None
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                handle = module.register_forward_hook(hook_fn)
                break
        
        # 前向传播
        self.model.eval()
        with torch.no_grad():
            _ = self.model(torch.FloatTensor(X))
        
        if handle:
            handle.remove()
        
        return activations[0] if activations else X
    
    def _get_tensorflow_activations(self, X: np.ndarray) -> np.ndarray:
        """获取TensorFlow激活"""
        import tensorflow as tf
        
        layer = self.model.get_layer(self.layer_name)
        activation_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=layer.output
        )
        
        return activation_model.predict(X)
    
    def _get_pytorch_gradients(self, X: np.ndarray) -> np.ndarray:
        """获取PyTorch梯度"""
        import torch
        
        X_tensor = torch.FloatTensor(X)
        X_tensor.requires_grad = True
        
        output = self.model(X_tensor)
        
        # 计算输出的梯度
        gradients = []
        for i in range(len(output)):
            if X_tensor.grad is not None:
                X_tensor.grad.zero_()
            output[i].backward(retain_graph=True)
            gradients.append(X_tensor.grad.numpy().flatten())
        
        return np.array(gradients)
    
    def _get_tensorflow_gradients(self, X: np.ndarray) -> np.ndarray:
        """获取TensorFlow梯度"""
        import tensorflow as tf
        
        X_tf = tf.Variable(X, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            predictions = self.model(X_tf)
        
        gradients = tape.gradient(predictions, X_tf)
        return gradients.numpy()


class IntegratedGradients:
    """
    积分梯度解释器
    
    用于属性归因的深度解释方法
    """
    
    def __init__(self, model: Callable, baseline: np.ndarray = None):
        """
        初始化积分梯度解释器
        
        Args:
            model: 预测模型
            baseline: 基线输入
        """
        self.model = model
        self.baseline = baseline
        
    def explain(self, X: np.ndarray, instance_idx: int = 0,
               n_steps: int = 50) -> Explanation:
        """
        解释单个实例
        
        Args:
            X: 输入数据
            instance_idx: 要解释的实例索引
            n_steps: 积分步数
            
        Returns:
            解释结果
        """
        instance = X[instance_idx]
        
        if self.baseline is None:
            baseline = np.zeros_like(instance)
        else:
            baseline = self.baseline
        
        # 生成插值路径
        alphas = np.linspace(0, 1, n_steps + 1)
        interpolated = np.array([
            baseline + alpha * (instance - baseline)
            for alpha in alphas
        ])
        
        # 计算梯度
        gradients = self._compute_gradients(interpolated)
        
        # 黎曼近似积分
        avg_gradients = np.mean(gradients[:-1], axis=0)
        
        # 积分梯度
        integrated_gradients = (instance - baseline) * avg_gradients
        
        # 构建解释
        prediction = self.model.predict(instance.reshape(1, -1))[0]
        base_prediction = self.model.predict(baseline.reshape(1, -1))[0]
        
        feature_names = [f"feature_{i}" for i in range(len(instance))]
        contributions = {name: float(integrated_gradients[i])
                        for i, name in enumerate(feature_names)}
        
        return Explanation(
            instance_id=instance_idx,
            base_value=float(base_prediction),
            prediction=float(prediction),
            feature_contributions=contributions,
            method="Integrated Gradients",
            metadata={'n_steps': n_steps}
        )
    
    def _compute_gradients(self, X: np.ndarray) -> np.ndarray:
        """计算梯度"""
        # 数值近似梯度
        eps = 1e-4
        gradients = np.zeros_like(X)
        
        for i in range(len(X)):
            for j in range(X.shape[1]):
                X_plus = X[i].copy()
                X_minus = X[i].copy()
                X_plus[j] += eps
                X_minus[j] -= eps
                
                pred_plus = self.model.predict(X_plus.reshape(1, -1))[0]
                pred_minus = self.model.predict(X_minus.reshape(1, -1))[0]
                
                gradients[i, j] = (pred_plus - pred_minus) / (2 * eps)
        
        return gradients
    
    def explain_batch(self, X: np.ndarray, **kwargs) -> List[Explanation]:
        """批量解释"""
        return [self.explain(X, i, **kwargs) for i in range(len(X))]


class PermutationImportance:
    """
    置换重要性
    
    通过置换特征值来估计重要性
    """
    
    def __init__(self, model: Callable, metric: str = "mse"):
        """
        初始化置换重要性计算器
        
        Args:
            model: 预测模型
            metric: 评估指标
        """
        self.model = model
        self.metric = metric
        
    def compute(self, X: np.ndarray, y: np.ndarray,
               n_repeats: int = 10) -> List[FeatureImportance]:
        """
        计算置换重要性
        
        Args:
            X: 特征数据
            y: 目标值
            n_repeats: 重复次数
            
        Returns:
            特征重要性列表
        """
        # 基准分数
        baseline_score = self._score(X, y)
        
        importances = []
        n_features = X.shape[1]
        
        for i in range(n_features):
            scores = []
            
            for _ in range(n_repeats):
                # 置换特征
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, i])
                
                # 计算分数下降
                permuted_score = self._score(X_permuted, y)
                score_drop = baseline_score - permuted_score
                scores.append(score_drop)
            
            importances.append(FeatureImportance(
                feature_name=f"feature_{i}",
                importance=np.mean(scores),
                std=np.std(scores),
                method="Permutation"
            ))
        
        return sorted(importances, key=lambda x: x.importance, reverse=True)
    
    def _score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算模型分数"""
        predictions = self.model.predict(X)
        
        if self.metric == "mse":
            return -np.mean((y - predictions) ** 2)
        elif self.metric == "mae":
            return -np.mean(np.abs(y - predictions))
        elif self.metric == "accuracy":
            return np.mean(y == predictions)
        elif self.metric == "r2":
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            return -np.mean((y - predictions) ** 2)


class ExplainableMLPipeline:
    """
    可解释ML管道
    
    整合多种解释方法的完整管道
    """
    
    def __init__(self, model: Callable, feature_names: List[str] = None):
        """
        初始化管道
        
        Args:
            model: 预测模型
            feature_names: 特征名称
        """
        self.model = model
        self.feature_names = feature_names
        self.explainers: Dict[str, Any] = {}
        self.explanations: Dict[str, List[Explanation]] = {}
        
    def add_explainer(self, name: str, explainer: Any):
        """添加解释器"""
        self.explainers[name] = explainer
        
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        拟合管道
        
        Args:
            X: 训练数据
            y: 目标值（可选）
        """
        # 初始化默认解释器
        if 'shap' not in self.explainers:
            self.explainers['shap'] = SHAPExplainer(
                self.model, background_data=X, feature_names=self.feature_names
            )
        
        if 'lime' not in self.explainers:
            self.explainers['lime'] = LIMEExplainer(
                self.model, feature_names=self.feature_names
            )
        
        if 'permutation' not in self.explainers and y is not None:
            self.explainers['permutation'] = PermutationImportance(self.model)
        
        if 'integrated_gradients' not in self.explainers:
            self.explainers['integrated_gradients'] = IntegratedGradients(self.model)
        
        return self
    
    def explain(self, X: np.ndarray, instance_idx: int = 0,
               methods: List[str] = None) -> Dict[str, Explanation]:
        """
        使用多种方法解释
        
        Args:
            X: 输入数据
            instance_idx: 实例索引
            methods: 要使用的解释方法
            
        Returns:
            各方法的解释结果
        """
        if methods is None:
            methods = list(self.explainers.keys())
        
        results = {}
        for method in methods:
            if method in self.explainers:
                try:
                    explainer = self.explainers[method]
                    if hasattr(explainer, 'explain'):
                        results[method] = explainer.explain(X, instance_idx)
                except Exception as e:
                    warnings.warn(f"Failed to explain with {method}: {e}")
        
        return results
    
    def global_explanation(self, X: np.ndarray, y: np.ndarray = None) -> Dict:
        """
        全局解释
        
        Args:
            X: 输入数据
            y: 目标值
            
        Returns:
            全局解释结果
        """
        results = {}
        
        # SHAP全局重要性
        if 'shap' in self.explainers:
            results['shap_importance'] = self.explainers['shap'].global_importance(X)
        
        # 置换重要性
        if 'permutation' in self.explainers and y is not None:
            results['permutation_importance'] = self.explainers['permutation'].compute(X, y)
        
        # 特征相关性
        if HAS_MATPLOTLIB:
            results['correlation_matrix'] = np.corrcoef(X.T)
        
        return results
    
    def compare_explanations(self, X: np.ndarray, instance_idx: int = 0) -> pd.DataFrame:
        """
        比较不同解释方法的结果
        
        Args:
            X: 输入数据
            instance_idx: 实例索引
            
        Returns:
            比较DataFrame
        """
        explanations = self.explain(X, instance_idx)
        
        # 收集所有特征
        all_features = set()
        for exp in explanations.values():
            all_features.update(exp.feature_contributions.keys())
        
        all_features = sorted(all_features)
        
        # 创建比较表
        data = {'feature': all_features}
        for method, exp in explanations.items():
            data[method] = [exp.feature_contributions.get(f, 0) for f in all_features]
        
        return pd.DataFrame(data)
    
    def plot_comparison(self, X: np.ndarray, instance_idx: int = 0,
                       top_n: int = 10, ax=None):
        """
        可视化解释方法比较
        
        Args:
            X: 输入数据
            instance_idx: 实例索引
            top_n: 显示前N个特征
            ax: matplotlib轴
        """
        if not HAS_MATPLOTLIB:
            warnings.warn("matplotlib not available")
            return None
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        comparison = self.compare_explanations(X, instance_idx)
        
        # 获取所有方法的平均重要性来确定top特征
        method_cols = [c for c in comparison.columns if c != 'feature']
        comparison['avg_importance'] = comparison[method_cols].abs().mean(axis=1)
        top_features = comparison.nlargest(top_n, 'avg_importance')['feature'].values
        
        # 绘制
        x = np.arange(len(top_features))
        width = 0.8 / len(method_cols)
        
        for i, method in enumerate(method_cols):
            values = [comparison[comparison['feature'] == f][method].values[0] 
                     for f in top_features]
            ax.bar(x + i * width, values, width, label=method, alpha=0.8)
        
        ax.set_xlabel('Feature')
        ax.set_ylabel('Contribution')
        ax.set_title('Explanation Comparison')
        ax.set_xticks(x + width * (len(method_cols) - 1) / 2)
        ax.set_xticklabels(top_features, rotation=45, ha='right')
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        return ax
    
    def stability_analysis(self, X: np.ndarray, 
                          n_perturbations: int = 10,
                          noise_level: float = 0.01) -> Dict:
        """
        解释稳定性分析
        
        Args:
            X: 输入数据
            n_perturbations: 扰动次数
            noise_level: 噪声水平
            
        Returns:
            稳定性分析结果
        """
        results = {}
        
        for method_name, explainer in self.explainers.items():
            if not hasattr(explainer, 'explain'):
                continue
            
            # 原始解释
            orig_exp = explainer.explain(X, 0)
            orig_contrib = np.array(list(orig_exp.feature_contributions.values()))
            
            # 扰动后的解释
            similarities = []
            for _ in range(n_perturbations):
                X_perturbed = X + np.random.normal(0, noise_level, X.shape)
                pert_exp = explainer.explain(X_perturbed, 0)
                pert_contrib = np.array(list(pert_exp.feature_contributions.values()))
                
                # 计算余弦相似度
                sim = np.dot(orig_contrib, pert_contrib) / (
                    np.linalg.norm(orig_contrib) * np.linalg.norm(pert_contrib) + 1e-10
                )
                similarities.append(sim)
            
            results[method_name] = {
                'mean_similarity': np.mean(similarities),
                'std_similarity': np.std(similarities),
                'min_similarity': np.min(similarities)
            }
        
        return results


def example_usage():
    """使用示例"""
    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    
    print("=" * 60)
    print("可解释机器学习示例")
    print("=" * 60)
    
    # 生成数据
    np.random.seed(42)
    X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
    feature_names = [f"Feature_{i}" for i in range(10)]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"\n模型性能: R² = {model.score(X_test, y_test):.4f}")
    
    # 创建解释管道
    pipeline = ExplainableMLPipeline(model, feature_names=feature_names)
    pipeline.fit(X_train, y_train)
    
    # 局部解释
    print("\n" + "-" * 40)
    print("局部解释 (实例 0)")
    print("-" * 40)
    
    explanations = pipeline.explain(X_test, instance_idx=0)
    for method, exp in explanations.items():
        print(f"\n{method.upper()}:")
        print(f"  预测值: {exp.prediction:.4f}")
        print(f"  Top 3 特征:")
        for feat, contrib in exp.top_features(3):
            print(f"    {feat}: {contrib:.4f}")
    
    # 全局解释
    print("\n" + "-" * 40)
    print("全局特征重要性")
    print("-" * 40)
    
    global_exp = pipeline.global_explanation(X_test, y_test)
    
    if 'shap_importance' in global_exp:
        print("\nSHAP重要性:")
        for imp in global_exp['shap_importance'][:5]:
            print(f"  {imp}")
    
    if 'permutation_importance' in global_exp:
        print("\n置换重要性:")
        for imp in global_exp['permutation_importance'][:5]:
            print(f"  {imp}")
    
    # 稳定性分析
    print("\n" + "-" * 40)
    print("解释稳定性分析")
    print("-" * 40)
    
    stability = pipeline.stability_analysis(X_test[:100])
    for method, metrics in stability.items():
        print(f"\n{method}:")
        print(f"  平均相似度: {metrics['mean_similarity']:.4f}")
        print(f"  标准差: {metrics['std_similarity']:.4f}")
    
    print("\n" + "=" * 60)
    print("示例完成")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()
