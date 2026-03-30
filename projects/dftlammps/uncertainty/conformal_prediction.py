"""
共形预测模块 - Conformal Prediction

本模块实现共形预测方法进行不确定性量化：
- 标准共形预测
- 自适应共形预测
- 多标签共形预测
- 时间序列共形预测

作者: Causal AI Team
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

try:
    from sklearn.model_selection import train_test_split
    from sklearn.base import BaseEstimator, RegressorMixin
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class ConformalPrediction:
    """共形预测结果"""
    point_prediction: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    confidence: float
    interval_width: np.ndarray
    coverage: float = 0.0
    
    def contains(self, y_true: np.ndarray) -> np.ndarray:
        """检查真实值是否在预测区间内"""
        return (y_true >= self.lower_bound) & (y_true <= self.upper_bound)
    
    def average_coverage(self, y_true: np.ndarray) -> float:
        """计算实际覆盖率"""
        return np.mean(self.contains(y_true))
    
    def average_interval_width(self) -> float:
        """计算平均区间宽度"""
        return np.mean(self.interval_width)


class StandardConformalPredictor:
    """
    标准共形预测器
    
    基于保形分数的预测区间估计
    """
    
    def __init__(self,
                 base_model: Any,
                 nonconformity: str = 'absolute',
                 normalization: bool = False):
        """
        初始化共形预测器
        
        Args:
            base_model: 基础预测模型
            nonconformity: 非一致性分数类型 ('absolute', 'normalized')
            normalization: 是否使用归一化
        """
        self.base_model = base_model
        self.nonconformity = nonconformity
        self.normalization = normalization
        
        self.calibration_scores: np.ndarray = None
        self.q_hat: float = None
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        拟合基础模型
        
        Args:
            X_train: 训练数据
            y_train: 训练标签
        """
        self.base_model.fit(X_train, y_train)
        return self
    
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray,
                 alpha: float = 0.1):
        """
        校准共形预测器
        
        Args:
            X_cal: 校准数据
            y_cal: 校准标签
            alpha: 错误率 (1-alpha = 置信度)
        """
        # 预测
        y_pred_cal = self.base_model.predict(X_cal)
        
        # 计算非一致性分数
        if self.nonconformity == 'absolute':
            scores = np.abs(y_cal - y_pred_cal)
        elif self.nonconformity == 'squared':
            scores = (y_cal - y_pred_cal) ** 2
        elif self.nonconformity == 'normalized':
            # 归一化非一致性分数
            residuals = np.abs(y_cal - y_pred_cal)
            # 使用残差的局部估计
            scores = residuals / (np.std(residuals) + 1e-10)
        else:
            scores = np.abs(y_cal - y_pred_cal)
        
        self.calibration_scores = scores
        
        # 计算分位数
        n = len(scores)
        # 使用修正的分位数
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        self.q_hat = np.quantile(scores, q_level, method='higher')
        
        return self
    
    def predict(self, X: np.ndarray, alpha: float = None) -> ConformalPrediction:
        """
        预测并构建预测区间
        
        Args:
            X: 输入数据
            alpha: 错误率（如果为None则使用校准时的值）
            
        Returns:
            共形预测结果
        """
        if self.q_hat is None:
            raise ValueError("Must call calibrate() before predict()")
        
        # 点预测
        y_pred = self.base_model.predict(X)
        
        # 构建预测区间
        if self.nonconformity == 'absolute':
            lower = y_pred - self.q_hat
            upper = y_pred + self.q_hat
        elif self.nonconformity == 'squared':
            width = np.sqrt(self.q_hat)
            lower = y_pred - width
            upper = y_pred + width
        elif self.nonconformity == 'normalized':
            lower = y_pred - self.q_hat
            upper = y_pred + self.q_hat
        else:
            lower = y_pred - self.q_hat
            upper = y_pred + self.q_hat
        
        interval_width = upper - lower
        
        return ConformalPrediction(
            point_prediction=y_pred,
            lower_bound=lower,
            upper_bound=upper,
            confidence=1 - (alpha if alpha else 0.1),
            interval_width=interval_width
        )
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                alpha: float = 0.1) -> Dict:
        """
        评估共形预测
        
        Args:
            X_test: 测试数据
            y_test: 测试标签
            alpha: 错误率
            
        Returns:
            评估指标
        """
        pred = self.predict(X_test, alpha)
        
        coverage = pred.average_coverage(y_test)
        avg_width = pred.average_interval_width()
        
        return {
            'target_coverage': 1 - alpha,
            'actual_coverage': coverage,
            'average_interval_width': avg_width,
            'coverage_gap': abs(coverage - (1 - alpha))
        }


class AdaptiveConformalPredictor:
    """
    自适应共形预测器
    
    根据输入特征自适应调整预测区间宽度
    """
    
    def __init__(self,
                 base_model: Any,
                 difficulty_model: Any = None):
        """
        初始化自适应共形预测器
        
        Args:
            base_model: 基础预测模型
            difficulty_model: 难度估计模型（预测残差大小）
        """
        self.base_model = base_model
        self.difficulty_model = difficulty_model
        
        self.calibration_scores: np.ndarray = None
        self.q_hat: float = None
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        拟合模型
        
        Args:
            X_train: 训练数据
            y_train: 训练标签
        """
        # 拟合基础模型
        self.base_model.fit(X_train, y_train)
        
        # 拟合难度模型（如果没有提供）
        if self.difficulty_model is None:
            # 使用残差作为难度估计
            y_pred_train = self.base_model.predict(X_train)
            residuals = np.abs(y_train - y_pred_train)
            
            # 简单使用特征范数作为难度代理
            self.difficulty_model = lambda X: np.linalg.norm(X, axis=1) + 1.0
        else:
            y_pred_train = self.base_model.predict(X_train)
            residuals = np.abs(y_train - y_pred_train)
            self.difficulty_model.fit(X_train, residuals)
        
        return self
    
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray,
                 alpha: float = 0.1):
        """
        校准
        
        Args:
            X_cal: 校准数据
            y_cal: 校准标签
            alpha: 错误率
        """
        y_pred_cal = self.base_model.predict(X_cal)
        
        # 估计难度
        difficulties = self._estimate_difficulty(X_cal)
        
        # 归一化非一致性分数
        residuals = np.abs(y_cal - y_pred_cal)
        scores = residuals / (difficulties + 1e-10)
        
        self.calibration_scores = scores
        
        # 计算分位数
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        self.q_hat = np.quantile(scores, q_level, method='higher')
        
        return self
    
    def _estimate_difficulty(self, X: np.ndarray) -> np.ndarray:
        """估计预测难度"""
        if callable(self.difficulty_model):
            return self.difficulty_model(X)
        else:
            return self.difficulty_model.predict(X)
    
    def predict(self, X: np.ndarray, alpha: float = None) -> ConformalPrediction:
        """
        预测
        
        Args:
            X: 输入数据
            alpha: 错误率
            
        Returns:
            共形预测结果
        """
        if self.q_hat is None:
            raise ValueError("Must call calibrate() before predict()")
        
        y_pred = self.base_model.predict(X)
        difficulties = self._estimate_difficulty(X)
        
        # 自适应区间宽度
        margin = self.q_hat * difficulties
        
        lower = y_pred - margin
        upper = y_pred + margin
        
        interval_width = upper - lower
        
        return ConformalPrediction(
            point_prediction=y_pred,
            lower_bound=lower,
            upper_bound=upper,
            confidence=1 - (alpha if alpha else 0.1),
            interval_width=interval_width
        )


class ConformalizedQuantileRegression:
    """
    共形化分位数回归
    
    结合分位数回归和共形校正
    """
    
    def __init__(self,
                 lower_quantile_model: Any,
                 upper_quantile_model: Any,
                 median_model: Any = None):
        """
        初始化共形化分位数回归
        
        Args:
            lower_quantile_model: 下分位数模型
            upper_quantile_model: 上分位数模型
            median_model: 中位数模型（可选）
        """
        self.lower_model = lower_quantile_model
        self.upper_model = upper_quantile_model
        self.median_model = median_model
        
        self.q_hat: float = 0.0
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        拟合分位数模型
        
        Args:
            X_train: 训练数据
            y_train: 训练标签
        """
        self.lower_model.fit(X_train, y_train)
        self.upper_model.fit(X_train, y_train)
        
        if self.median_model is not None:
            self.median_model.fit(X_train, y_train)
        
        return self
    
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray,
                 alpha: float = 0.1):
        """
        校准
        
        Args:
            X_cal: 校准数据
            y_cal: 校准标签
            alpha: 错误率
        """
        lower_pred = self.lower_model.predict(X_cal)
        upper_pred = self.upper_model.predict(X_cal)
        
        # 计算非一致性分数
        scores = np.maximum(lower_pred - y_cal, y_cal - upper_pred)
        
        # 计算分位数
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        self.q_hat = np.quantile(scores, q_level, method='higher')
        
        # 确保非负
        self.q_hat = max(0, self.q_hat)
        
        return self
    
    def predict(self, X: np.ndarray, alpha: float = None) -> ConformalPrediction:
        """
        预测
        
        Args:
            X: 输入数据
            alpha: 错误率
            
        Returns:
            共形预测结果
        """
        lower_base = self.lower_model.predict(X)
        upper_base = self.upper_model.predict(X)
        
        if self.median_model is not None:
            point_pred = self.median_model.predict(X)
        else:
            point_pred = (lower_base + upper_base) / 2
        
        # 应用共形校正
        lower = lower_base - self.q_hat
        upper = upper_base + self.q_hat
        
        interval_width = upper - lower
        
        return ConformalPrediction(
            point_prediction=point_pred,
            lower_bound=lower,
            upper_bound=upper,
            confidence=1 - (alpha if alpha else 0.1),
            interval_width=interval_width
        )


class MultiLabelConformalPredictor:
    """
    多标签共形预测器
    
    为每个标签提供预测集
    """
    
    def __init__(self,
                 base_models: List[Any],
                 nonconformity: str = 'absolute'):
        """
        初始化多标签共形预测器
        
        Args:
            base_models: 每个标签的基础模型列表
            nonconformity: 非一致性分数类型
        """
        self.base_models = base_models
        self.nonconformity = nonconformity
        
        self.calibration_scores: List[np.ndarray] = None
        self.q_hats: List[float] = None
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        拟合基础模型
        
        Args:
            X_train: 训练数据
            y_train: 训练标签 [n_samples, n_labels]
        """
        n_labels = y_train.shape[1]
        
        for i in range(n_labels):
            self.base_models[i].fit(X_train, y_train[:, i])
        
        return self
    
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray,
                 alpha: float = 0.1):
        """
        校准
        
        Args:
            X_cal: 校准数据
            y_cal: 校准标签
            alpha: 错误率
        """
        n_labels = y_cal.shape[1]
        self.calibration_scores = []
        self.q_hats = []
        
        for i in range(n_labels):
            y_pred_cal = self.base_models[i].predict(X_cal)
            
            if self.nonconformity == 'absolute':
                scores = np.abs(y_cal[:, i] - y_pred_cal)
            else:
                scores = (y_cal[:, i] - y_pred_cal) ** 2
            
            self.calibration_scores.append(scores)
            
            # 计算分位数
            n = len(scores)
            q_level = np.ceil((n + 1) * (1 - alpha)) / n
            q_hat = np.quantile(scores, q_level, method='higher')
            self.q_hats.append(q_hat)
        
        return self
    
    def predict(self, X: np.ndarray) -> List[ConformalPrediction]:
        """
        预测所有标签
        
        Args:
            X: 输入数据
            
        Returns:
            每个标签的共形预测结果列表
        """
        n_labels = len(self.base_models)
        predictions = []
        
        for i in range(n_labels):
            y_pred = self.base_models[i].predict(X)
            
            if self.nonconformity == 'absolute':
                lower = y_pred - self.q_hats[i]
                upper = y_pred + self.q_hats[i]
            else:
                width = np.sqrt(self.q_hats[i])
                lower = y_pred - width
                upper = y_pred + width
            
            pred = ConformalPrediction(
                point_prediction=y_pred,
                lower_bound=lower,
                upper_bound=upper,
                confidence=0.9,
                interval_width=upper - lower
            )
            predictions.append(pred)
        
        return predictions


class TimeSeriesConformalPredictor:
    """
    时间序列共形预测器
    
    为时间序列数据提供预测区间
    """
    
    def __init__(self,
                 base_model: Any,
                 lookback_window: int = 10,
                 seasonal_period: int = None):
        """
        初始化时间序列共形预测器
        
        Args:
            base_model: 基础预测模型
            lookback_window: 回溯窗口大小
            seasonal_period: 季节性周期（如果有）
        """
        self.base_model = base_model
        self.lookback_window = lookback_window
        self.seasonal_period = seasonal_period
        
        self.calibration_scores: np.ndarray = None
        self.q_hat: float = None
        
    def _create_features(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        为时间序列创建特征
        
        Args:
            series: 时间序列数据
            
        Returns:
            (特征, 目标)
        """
        n = len(series)
        X = []
        y = []
        
        for i in range(self.lookback_window, n):
            # 使用过去窗口的值作为特征
            features = series[i-self.lookback_window:i].flatten()
            
            # 添加季节性特征
            if self.seasonal_period:
                season = (i % self.seasonal_period) / self.seasonal_period
                features = np.append(features, [np.sin(2*np.pi*season), np.cos(2*np.pi*season)])
            
            X.append(features)
            y.append(series[i])
        
        return np.array(X), np.array(y)
    
    def fit(self, series: np.ndarray):
        """
        拟合模型
        
        Args:
            series: 时间序列数据
        """
        X, y = self._create_features(series)
        self.base_model.fit(X, y)
        self.series_mean = np.mean(series)
        self.series_std = np.std(series)
        return self
    
    def calibrate(self, series: np.ndarray, alpha: float = 0.1):
        """
        校准
        
        Args:
            series: 校准时间序列
            alpha: 错误率
        """
        X, y_true = self._create_features(series)
        y_pred = self.base_model.predict(X)
        
        # 计算非一致性分数
        scores = np.abs(y_true - y_pred)
        self.calibration_scores = scores
        
        # 计算分位数
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        self.q_hat = np.quantile(scores, q_level, method='higher')
        
        return self
    
    def predict(self, series: np.ndarray, horizon: int = 1) -> ConformalPrediction:
        """
        预测未来值
        
        Args:
            series: 历史时间序列
            horizon: 预测步长
            
        Returns:
            共形预测结果
        """
        if self.q_hat is None:
            raise ValueError("Must call calibrate() before predict()")
        
        predictions = []
        
        # 使用最后窗口进行多步预测
        current_window = series[-self.lookback_window:].flatten()
        
        for h in range(horizon):
            # 创建特征
            features = current_window.copy()
            
            if self.seasonal_period:
                future_idx = len(series) + h
                season = (future_idx % self.seasonal_period) / self.seasonal_period
                features = np.append(features, [np.sin(2*np.pi*season), np.cos(2*np.pi*season)])
            
            # 预测
            pred = self.base_model.predict(features.reshape(1, -1))[0]
            predictions.append(pred)
            
            # 更新窗口（使用预测值）
            current_window = np.append(current_window[1:], pred)
        
        predictions = np.array(predictions)
        
        # 构建区间（随horizon增加而扩大）
        margin = self.q_hat * (1 + 0.1 * np.arange(horizon))  # 随时间扩大
        
        lower = predictions - margin
        upper = predictions + margin
        
        return ConformalPrediction(
            point_prediction=predictions,
            lower_bound=lower,
            upper_bound=upper,
            confidence=0.9,
            interval_width=upper - lower
        )


class ConformalPredictionPipeline:
    """
    共形预测管道
    
    整合多种共形预测方法的完整流程
    """
    
    def __init__(self,
                 base_model: Any,
                 method: str = 'standard'):
        """
        初始化管道
        
        Args:
            base_model: 基础模型
            method: 共形预测方法 ('standard', 'adaptive', 'cqr')
        """
        self.method = method
        self.predictor = None
        
        if method == 'standard':
            self.predictor = StandardConformalPredictor(base_model)
        elif method == 'adaptive':
            self.predictor = AdaptiveConformalPredictor(base_model)
        elif method == 'cqr':
            raise ValueError("Use ConformalizedQuantileRegression directly for CQR")
        else:
            raise ValueError(f"Unknown method: {method}")
        
    def fit_calibrate(self, X: np.ndarray, y: np.ndarray,
                     calibration_split: float = 0.2,
                     alpha: float = 0.1) -> 'ConformalPredictionPipeline':
        """
        拟合和校准
        
        Args:
            X: 数据
            y: 标签
            calibration_split: 校准集比例
            alpha: 错误率
            
        Returns:
            self
        """
        if HAS_SKLEARN:
            X_train, X_cal, y_train, y_cal = train_test_split(
                X, y, test_size=calibration_split, random_state=42
            )
        else:
            # 简单分割
            n_cal = int(len(X) * calibration_split)
            indices = np.random.permutation(len(X))
            X_train, X_cal = X[indices[n_cal:]], X[indices[:n_cal]]
            y_train, y_cal = y[indices[n_cal:]], y[indices[:n_cal]]
        
        self.predictor.fit(X_train, y_train)
        self.predictor.calibrate(X_cal, y_cal, alpha)
        
        return self
    
    def predict(self, X: np.ndarray) -> ConformalPrediction:
        """预测"""
        return self.predictor.predict(X)
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                      n_folds: int = 5,
                      alpha: float = 0.1) -> Dict:
        """
        交叉验证
        
        Args:
            X: 数据
            y: 标签
            n_folds: 折数
            alpha: 错误率
            
        Returns:
            交叉验证结果
        """
        coverages = []
        widths = []
        
        fold_size = len(X) // n_folds
        indices = np.random.permutation(len(X))
        
        for fold in range(n_folds):
            # 分割
            test_idx = indices[fold * fold_size:(fold + 1) * fold_size]
            train_cal_idx = np.concatenate([
                indices[:fold * fold_size],
                indices[(fold + 1) * fold_size:]
            ])
            
            # 进一步分割训练和校准
            cal_size = len(train_cal_idx) // 5
            cal_idx = train_cal_idx[:cal_size]
            train_idx = train_cal_idx[cal_size:]
            
            X_train, X_cal, X_test = X[train_idx], X[cal_idx], X[test_idx]
            y_train, y_cal, y_test = y[train_idx], y[cal_idx], y[test_idx]
            
            # 训练
            self.predictor.fit(X_train, y_train)
            self.predictor.calibrate(X_cal, y_cal, alpha)
            
            # 评估
            pred = self.predictor.predict(X_test)
            coverages.append(pred.average_coverage(y_test))
            widths.append(pred.average_interval_width())
        
        return {
            'mean_coverage': np.mean(coverages),
            'std_coverage': np.std(coverages),
            'mean_width': np.mean(widths),
            'std_width': np.std(widths)
        }


def example_usage():
    """使用示例"""
    print("=" * 60)
    print("共形预测示例")
    print("=" * 60)
    
    if not HAS_SKLEARN:
        print("scikit-learn not available, skipping examples")
        return
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    
    # 生成数据
    np.random.seed(42)
    n_samples = 500
    X = np.random.uniform(-3, 3, (n_samples, 2))
    y = X[:, 0]**2 + np.sin(X[:, 1]) + 0.2 * np.random.randn(n_samples)
    
    print(f"\n样本数: {n_samples}")
    
    # 示例1: 标准共形预测
    print("\n" + "-" * 40)
    print("1. 标准共形预测")
    print("-" * 40)
    
    base_model = RandomForestRegressor(n_estimators=50, random_state=42)
    cp = StandardConformalPredictor(base_model)
    
    pipeline = ConformalPredictionPipeline(base_model, method='standard')
    pipeline.fit_calibrate(X, y, calibration_split=0.2, alpha=0.1)
    
    # 评估
    eval_results = pipeline.predictor.evaluate(X, y, alpha=0.1)
    print(f"目标覆盖率: {eval_results['target_coverage']:.2f}")
    print(f"实际覆盖率: {eval_results['actual_coverage']:.4f}")
    print(f"平均区间宽度: {eval_results['average_interval_width']:.4f}")
    
    # 示例2: 自适应共形预测
    print("\n" + "-" * 40)
    print("2. 自适应共形预测")
    print("-" * 40)
    
    adaptive_cp = AdaptiveConformalPredictor(
        RandomForestRegressor(n_estimators=50, random_state=42)
    )
    
    if HAS_SKLEARN:
        from sklearn.model_selection import train_test_split
        X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4, random_state=42)
        X_cal, X_test, y_cal, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)
    else:
        n_train = int(n_samples * 0.6)
        n_cal = int(n_samples * 0.2)
        X_train, X_cal, X_test = X[:n_train], X[n_train:n_train+n_cal], X[n_train+n_cal:]
        y_train, y_cal, y_test = y[:n_train], y[n_train:n_train+n_cal], y[n_train+n_cal:]
    
    adaptive_cp.fit(X_train, y_train)
    adaptive_cp.calibrate(X_cal, y_cal, alpha=0.1)
    
    pred_adaptive = adaptive_cp.predict(X_test)
    coverage_adaptive = pred_adaptive.average_coverage(y_test)
    
    print(f"实际覆盖率: {coverage_adaptive:.4f}")
    print(f"平均区间宽度: {pred_adaptive.average_interval_width():.4f}")
    
    # 示例3: 时间序列共形预测
    print("\n" + "-" * 40)
    print("3. 时间序列共形预测")
    print("-" * 40)
    
    # 生成时间序列
    t = np.arange(200)
    ts = np.sin(0.1 * t) + 0.5 * np.sin(0.3 * t) + 0.1 * np.random.randn(200)
    
    ts_model = Ridge(alpha=1.0)
    ts_cp = TimeSeriesConformalPredictor(ts_model, lookback_window=10)
    
    ts_cp.fit(ts[:150])
    ts_cp.calibrate(ts[150:180], alpha=0.1)
    
    pred_ts = ts_cp.predict(ts[:180], horizon=20)
    
    print(f"预测范围: 20步")
    print(f"平均区间宽度: {pred_ts.average_interval_width():.4f}")
    print(f"区间宽度范围: [{np.min(pred_ts.interval_width):.4f}, {np.max(pred_ts.interval_width):.4f}]")
    
    # 示例4: 交叉验证
    print("\n" + "-" * 40)
    print("4. 交叉验证")
    print("-" * 40)
    
    cv_results = pipeline.cross_validate(X, y, n_folds=3, alpha=0.1)
    
    print(f"平均覆盖率: {cv_results['mean_coverage']:.4f} ± {cv_results['std_coverage']:.4f}")
    print(f"平均区间宽度: {cv_results['mean_width']:.4f} ± {cv_results['std_width']:.4f}")
    
    print("\n" + "=" * 60)
    print("示例完成")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()
