"""
反事实推理引擎 - Counterfactual Reasoning Engine
实现反事实推理用于因果效应估计和决策支持
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from scipy.optimize import minimize
import warnings


@dataclass
class CounterfactualQuery:
    """反事实查询"""
    target_variable: str
    intervention: Dict[str, Any]  # 干预变量及其值
    factual_evidence: Dict[str, Any]  # 事实证据
    
    def __repr__(self):
        return f"What if {self.intervention} given {self.factual_evidence}?"


@dataclass
class CounterfactualResult:
    """反事实推理结果"""
    query: CounterfactualQuery
    factual_outcome: Any
    counterfactual_outcome: Any
    effect: float
    confidence: float = 0.0
    explanation: str = ""
    
    def __repr__(self):
        return (f"Counterfactual: {self.factual_outcome} -> {self.counterfactual_outcome} "
                f"(effect: {self.effect:.4f})")


class StructuralCausalModel:
    """
    结构因果模型 (SCM)
    用于反事实推理的形式化因果模型
    """
    
    def __init__(self):
        self.endogenous: Dict[str, Callable] = {}  # 内生变量函数
        self.exogenous: Dict[str, Tuple[float, float]] = {}  # 外生变量分布
        self.parents: Dict[str, List[str]] = {}  # 变量依赖关系
        self.noise_values: Dict[str, float] = {}  # 外生变量值（用于反事实）
    
    def add_variable(
        self,
        name: str,
        equation: Callable,
        exogenous_mean: float = 0.0,
        exogenous_std: float = 1.0,
        parents: Optional[List[str]] = None
    ):
        """
        添加变量
        
        Args:
            name: 变量名
            equation: 结构方程 f(parents, noise)
            exogenous_mean: 外生变量均值
            exogenous_std: 外生变量标准差
            parents: 父变量列表
        """
        self.endogenous[name] = equation
        self.exogenous[name] = (exogenous_mean, exogenous_std)
        self.parents[name] = parents or []
    
    def sample_noise(self) -> Dict[str, float]:
        """采样外生变量"""
        noise = {}
        for name, (mean, std) in self.exogenous.items():
            noise[name] = np.random.normal(mean, std)
        return noise
    
    def abduction(
        self,
        evidence: Dict[str, Any],
        method: str = "inversion"
    ) -> Dict[str, float]:
        """
        溯因步骤：根据证据推断外生变量值
        
        Args:
            evidence: 事实证据
            method: 推断方法
        
        Returns:
            外生变量值
        """
        noise_values = {}
        
        for var, value in evidence.items():
            if var in self.endogenous:
                # 反推外生变量
                parents = self.parents[var]
                parent_values = {p: evidence.get(p, 0) for p in parents}
                
                # 数值反演
                def objective(u):
                    pred = self.endogenous[var](**parent_values, U=u)
                    return (pred - value) ** 2
                
                result = minimize(objective, x0=0.0, method='BFGS')
                noise_values[var] = result.x[0]
            else:
                # 外生变量直接观测
                noise_values[var] = value
        
        self.noise_values = noise_values
        return noise_values
    
    def action(
        self,
        interventions: Dict[str, Any]
    ) -> 'StructuralCausalModel':
        """
        行动步骤：应用干预
        
        Args:
            interventions: 干预变量及其值
        
        Returns:
            修改后的SCM
        """
        # 创建修改后的SCM
        scm_modified = StructuralCausalModel()
        
        for name, equation in self.endogenous.items():
            if name in interventions:
                # 被干预的变量变为常数
                value = interventions[name]
                scm_modified.add_variable(
                    name,
                    lambda **kwargs: value,
                    parents=[]
                )
            else:
                # 保持原结构
                scm_modified.add_variable(
                    name,
                    equation,
                    parents=self.parents[name].copy()
                )
                scm_modified.exogenous[name] = self.exogenous[name]
        
        scm_modified.noise_values = self.noise_values.copy()
        
        return scm_modified
    
    def prediction(
        self,
        variables: Optional[List[str]] = None,
        interventions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        预测步骤：计算变量值
        
        Args:
            variables: 要预测的变量（None则预测所有）
            interventions: 干预
        
        Returns:
            变量值
        """
        if interventions:
            scm = self.action(interventions)
        else:
            scm = self
        
        if variables is None:
            variables = list(scm.endogenous.keys())
        
        # 拓扑排序计算
        values = {}
        computed = set()
        
        def compute(var):
            if var in computed:
                return values[var]
            
            if var not in scm.endogenous:
                return scm.noise_values.get(var, 0)
            
            # 计算父变量
            parent_values = {}
            for p in scm.parents[var]:
                parent_values[p] = compute(p)
            
            # 添加外生变量
            noise = scm.noise_values.get(var, 0)
            
            # 计算
            result = scm.endogenous[var](**parent_values, U=noise)
            values[var] = result
            computed.add(var)
            
            return result
        
        for var in variables:
            compute(var)
        
        return {var: values[var] for var in variables}
    
    def counterfactual(
        self,
        query: CounterfactualQuery
    ) -> CounterfactualResult:
        """
        三步反事实推理
        
        Args:
            query: 反事实查询
        
        Returns:
            反事实结果
        """
        # 步骤1: 溯因 (Abduction)
        self.abduction(query.factual_evidence)
        
        # 步骤2: 行动 (Action)
        modified_scm = self.action(query.intervention)
        modified_scm.noise_values = self.noise_values
        
        # 步骤3: 预测 (Prediction)
        factual = self.prediction([query.target_variable])
        counterfactual = modified_scm.prediction([query.target_variable])
        
        factual_val = factual[query.target_variable]
        counterfactual_val = counterfactual[query.target_variable]
        
        effect = counterfactual_val - factual_val
        
        return CounterfactualResult(
            query=query,
            factual_outcome=factual_val,
            counterfactual_outcome=counterfactual_val,
            effect=effect,
            explanation=f"If {query.intervention}, {query.target_variable} would be {counterfactual_val:.4f} "
                       f"instead of {factual_val:.4f}"
        )


class CausalForest:
    """
    因果森林
    用于异质性因果效应估计
    """
    
    def __init__(
        self,
        n_trees: int = 100,
        max_depth: int = 10,
        min_samples_leaf: int = 10
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.trees: List[Dict] = []
    
    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray
    ):
        """
        拟合因果森林
        
        Args:
            X: 协变量 [n_samples, n_features]
            T: 处理变量 [n_samples]
            Y: 结果变量 [n_samples]
        """
        n_samples = len(X)
        
        for _ in range(self.n_trees):
            # Bootstrap样本
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            T_boot = T[indices]
            Y_boot = Y[indices]
            
            # 构建树
            tree = self._build_tree(X_boot, T_boot, Y_boot, depth=0)
            self.trees.append(tree)
    
    def _build_tree(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        depth: int
    ) -> Dict:
        """递归构建树"""
        n_samples = len(X)
        
        # 停止条件
        if depth >= self.max_depth or n_samples < self.min_samples_leaf:
            # 计算CATE
            treated = Y[T == 1]
            control = Y[T == 0]
            
            if len(treated) > 0 and len(control) > 0:
                cate = np.mean(treated) - np.mean(control)
            else:
                cate = 0
            
            return {"leaf": True, "cate": cate, "samples": n_samples}
        
        # 寻找最佳分裂
        best_feature = None
        best_threshold = None
        best_gain = -np.inf
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            thresholds = np.percentile(X[:, feature], [25, 50, 75])
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                # 计算分裂增益
                gain = self._compute_split_gain(Y, T, left_mask, right_mask)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        if best_feature is None:
            # 无法分裂
            treated = Y[T == 1]
            control = Y[T == 0]
            cate = np.mean(treated) - np.mean(control) if len(treated) > 0 and len(control) > 0 else 0
            return {"leaf": True, "cate": cate, "samples": n_samples}
        
        # 分裂
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left_tree = self._build_tree(X[left_mask], T[left_mask], Y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], T[right_mask], Y[right_mask], depth + 1)
        
        return {
            "leaf": False,
            "feature": best_feature,
            "threshold": best_threshold,
            "left": left_tree,
            "right": right_tree
        }
    
    def _compute_split_gain(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        left_mask: np.ndarray,
        right_mask: np.ndarray
    ) -> float:
        """计算分裂增益"""
        # 简化的增益计算
        treated_left = Y[left_mask & (T == 1)]
        control_left = Y[left_mask & (T == 0)]
        treated_right = Y[right_mask & (T == 1)]
        control_right = Y[right_mask & (T == 0)]
        
        if len(treated_left) == 0 or len(control_left) == 0 or len(treated_right) == 0 or len(control_right) == 0:
            return 0
        
        cate_left = np.mean(treated_left) - np.mean(control_left)
        cate_right = np.mean(treated_right) - np.mean(control_right)
        
        # 增益是CATE差异的平方
        return (cate_left - cate_right) ** 2
    
    def predict_effect(self, X: np.ndarray) -> np.ndarray:
        """
        预测条件平均处理效应
        
        Args:
            X: 协变量
        
        Returns:
            CATE估计
        """
        effects = []
        
        for x in X:
            tree_effects = []
            for tree in self.trees:
                effect = self._traverse_tree(tree, x)
                tree_effects.append(effect)
            effects.append(np.mean(tree_effects))
        
        return np.array(effects)
    
    def _traverse_tree(self, tree: Dict, x: np.ndarray) -> float:
        """遍历树"""
        if tree["leaf"]:
            return tree["cate"]
        
        if x[tree["feature"]] <= tree["threshold"]:
            return self._traverse_tree(tree["left"], x)
        else:
            return self._traverse_tree(tree["right"], x)


class CounterfactualExplainer:
    """
    反事实解释器
    为预测提供反事实解释
    """
    
    def __init__(
        self,
        model: Callable,
        feature_names: Optional[List[str]] = None,
        continuous_features: Optional[List[str]] = None
    ):
        self.model = model
        self.feature_names = feature_names
        self.continuous_features = continuous_features or []
    
    def generate_counterfactual(
        self,
        x: np.ndarray,
        target_class: Optional[int] = None,
        target_prediction: Optional[float] = None,
        proximity_weight: float = 0.5,
        diversity_weight: float = 0.3
    ) -> Dict[str, Any]:
        """
        生成反事实样本
        
        Args:
            x: 原始样本
            target_class: 目标类别
            target_prediction: 目标预测值
            proximity_weight: 接近度权重
            diversity_weight: 多样性权重
        
        Returns:
            反事实样本
        """
        original_pred = self.model(x.reshape(1, -1))[0]
        
        if target_prediction is None and target_class is None:
            # 目标相反
            target_prediction = 1 - original_pred if original_pred < 1 else 0
        
        # 优化问题：找到最小的改变达到目标
        def objective(x_cf):
            pred = self.model(x_cf.reshape(1, -1))[0]
            
            # 预测损失
            if target_prediction is not None:
                pred_loss = (pred - target_prediction) ** 2
            else:
                pred_loss = 0
            
            # 接近度损失
            proximity_loss = np.sum((x_cf - x) ** 2)
            
            return pred_loss + proximity_weight * proximity_loss
        
        # 约束：保持某些特征不变
        bounds = []
        for i in range(len(x)):
            if self.feature_names and self.feature_names[i] in self.continuous_features:
                bounds.append((x[i] - 3, x[i] + 3))
            else:
                bounds.append((x[i], x[i]))  # 固定
        
        result = minimize(
            objective,
            x0=x,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        x_cf = result.x
        cf_pred = self.model(x_cf.reshape(1, -1))[0]
        
        # 找出改变的特征
        changes = {}
        for i, (orig, cf) in enumerate(zip(x, x_cf)):
            if abs(cf - orig) > 1e-6:
                name = self.feature_names[i] if self.feature_names else f"feature_{i}"
                changes[name] = {"from": orig, "to": cf, "change": cf - orig}
        
        return {
            "original": x,
            "counterfactual": x_cf,
            "original_prediction": original_pred,
            "counterfactual_prediction": cf_pred,
            "changes": changes,
            "distance": np.sqrt(np.sum((x_cf - x) ** 2))
        }
    
    def generate_diverse_counterfactuals(
        self,
        x: np.ndarray,
        n_counterfactuals: int = 3,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """生成多样化的反事实"""
        counterfactuals = []
        
        for i in range(n_counterfactuals):
            # 添加随机扰动以促进多样性
            x_perturbed = x + np.random.randn(len(x)) * 0.1
            cf = self.generate_counterfactual(x_perturbed, **kwargs)
            counterfactuals.append(cf)
        
        return counterfactuals


class PolicyOptimizer:
    """
    策略优化器
    基于因果效应学习最优策略
    """
    
    def __init__(
        self,
        n_actions: int,
        feature_dim: int
    ):
        self.n_actions = n_actions
        self.feature_dim = feature_dim
        self.policy: Optional[Callable] = None
    
    def fit_policy(
        self,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        method: str = "direct"
    ):
        """
        学习最优策略
        
        Args:
            X: 上下文特征
            A: 采取的动作
            Y: 观测结果
            method: 学习方法
        """
        if method == "direct":
            self._fit_direct_method(X, A, Y)
        elif method == "ipw":
            self._fit_ipw(X, A, Y)
    
    def _fit_direct_method(
        self,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray
    ):
        """直接方法：拟合结果回归然后优化"""
        from sklearn.ensemble import GradientBoostingRegressor
        
        # 为每个动作拟合模型
        self.models = []
        for a in range(self.n_actions):
            mask = A == a
            if np.sum(mask) > 0:
                model = GradientBoostingRegressor(n_estimators=50, random_state=42)
                model.fit(X[mask], Y[mask])
                self.models.append(model)
            else:
                self.models.append(None)
        
        # 定义策略：选择预期结果最好的动作
        def policy(x):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            
            expected_outcomes = []
            for model in self.models:
                if model:
                    expected_outcomes.append(model.predict(x)[0])
                else:
                    expected_outcomes.append(-np.inf)
            
            return np.argmax(expected_outcomes)
        
        self.policy = policy
    
    def _fit_ipw(
        self,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray
    ):
        """逆概率加权方法"""
        # 估计倾向得分
        from sklearn.linear_model import LogisticRegression
        
        # 简化为二分类
        if self.n_actions == 2:
            prop_model = LogisticRegression(random_state=42)
            prop_model.fit(X, A)
            propensities = prop_model.predict_proba(X)[:, 1]
            
            # IPW加权
            weights = np.where(A == 1, 1 / (propensities + 1e-6), 1 / (1 - propensities + 1e-6))
            
            # 加权回归
            from sklearn.linear_model import Ridge
            model = Ridge()
            model.fit(X, Y, sample_weight=weights)
            
            self.policy = lambda x: 1 if model.predict(x.reshape(1, -1))[0] > 0.5 else 0
    
    def recommend(self, X: np.ndarray) -> np.ndarray:
        """推荐动作"""
        if self.policy is None:
            raise ValueError("Policy not fitted yet")
        
        return np.array([self.policy(x) for x in X])
    
    def evaluate_policy(
        self,
        X: np.ndarray,
        true_outcomes: Dict[int, np.ndarray]
    ) -> float:
        """
        评估策略效果
        
        Args:
            X: 上下文
            true_outcomes: 每个动作的真实结果
        
        Returns:
            策略价值
        """
        recommendations = self.recommend(X)
        
        value = 0
        for i, rec in enumerate(recommendations):
            if rec in true_outcomes:
                value += true_outcomes[rec][i]
        
        return value / len(X)


def demo():
    """演示反事实推理"""
    print("=" * 60)
    print("反事实推理引擎演示")
    print("=" * 60)
    
    # 1. 结构因果模型示例
    print("\n1. 结构因果模型 (SCM)")
    scm = StructuralCausalModel()
    
    # 定义变量
    # X: 温度, Z: 添加剂, Y: 电池容量
    scm.add_variable(
        "X",
        lambda U: 25 + 5 * U,  # 温度 ~ N(25, 25)
        exogenous_mean=0,
        exogenous_std=1
    )
    
    scm.add_variable(
        "Z",
        lambda X, U: 1 if X > 30 and U > 0 else 0,  # 添加剂使用决策
        exogenous_mean=0,
        exogenous_std=1,
        parents=["X"]
    )
    
    scm.add_variable(
        "Y",
        lambda X, Z, U: 100 - 0.5 * (X - 25)**2 + 10 * Z + 2 * U,  # 容量
        exogenous_mean=0,
        exogenous_std=2,
        parents=["X", "Z"]
    )
    
    # 反事实查询
    factual_evidence = {"X": 35, "Z": 0, "Y": 85}
    
    query = CounterfactualQuery(
        target_variable="Y",
        intervention={"Z": 1},  # 如果使用了添加剂
        factual_evidence=factual_evidence
    )
    
    result = scm.counterfactual(query)
    print(f"   事实证据: {factual_evidence}")
    print(f"   干预: {query.intervention}")
    print(f"   反事实结果:")
    print(f"     实际容量: {result.factual_outcome:.2f}")
    print(f"     反事实容量: {result.counterfactual_outcome:.2f}")
    print(f"     因果效应: {result.effect:.2f}")
    print(f"     解释: {result.explanation}")
    
    # 2. 因果森林
    print("\n2. 因果森林 (Causal Forest)")
    np.random.seed(42)
    n_samples = 500
    
    # 生成数据
    X = np.random.randn(n_samples, 3)
    T = np.random.binomial(1, 0.5, n_samples)
    
    # 异质性处理效应
    cate = 2 + X[:, 0] - 0.5 * X[:, 1]
    Y = cate * T + X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n_samples)
    
    cf = CausalForest(n_trees=50, max_depth=5)
    cf.fit(X, T, Y)
    
    # 预测效应
    X_test = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    effects = cf.predict_effect(X_test)
    print(f"   协变量 {X_test[0]} 的CATE: {effects[0]:.3f}")
    print(f"   协变量 {X_test[1]} 的CATE: {effects[1]:.3f}")
    
    # 3. 反事实解释
    print("\n3. 反事实解释 (Counterfactual Explanations)")
    
    # 简单模型
    def simple_model(X):
        return (X[:, 0] > 0.5).astype(float)
    
    explainer = CounterfactualExplainer(
        simple_model,
        feature_names=["temp", "pressure", "time"],
        continuous_features=["temp", "pressure", "time"]
    )
    
    x = np.array([0.3, 0.5, 0.8])
    cf_result = explainer.generate_counterfactual(x, target_prediction=1.0)
    
    print(f"   原始样本: {cf_result['original']}")
    print(f"   原始预测: {cf_result['original_prediction']}")
    print(f"   反事实样本: {cf_result['counterfactual']}")
    print(f"   反事实预测: {cf_result['counterfactual_prediction']}")
    print(f"   需要改变: {list(cf_result['changes'].keys())}")
    
    # 4. 策略优化
    print("\n4. 策略优化 (Policy Optimization)")
    
    np.random.seed(42)
    n_samples = 300
    
    X = np.random.randn(n_samples, 2)
    # 最优策略：当X[0] > 0时选择动作1
    optimal_A = (X[:, 0] > 0).astype(int)
    Y = optimal_A * (X[:, 0] + 1) + (1 - optimal_A) * (-X[:, 0]) + np.random.randn(n_samples) * 0.1
    
    policy_opt = PolicyOptimizer(n_actions=2, feature_dim=2)
    policy_opt.fit_policy(X, optimal_A, Y, method="direct")
    
    # 测试策略
    X_test = np.array([[1.0, 0.0], [-1.0, 0.0], [0.5, 0.5]])
    recommendations = policy_opt.recommend(X_test)
    
    print(f"   测试样本: {X_test.tolist()}")
    print(f"   推荐动作: {recommendations}")
    print(f"   预期: 当X[0] > 0时选择1，否则选择0")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
