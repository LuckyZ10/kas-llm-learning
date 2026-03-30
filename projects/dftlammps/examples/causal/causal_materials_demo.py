"""
因果材料发现应用案例

本案例展示如何使用因果AI和可解释ML进行材料发现：
1. 材料性质的因果关系发现
2. 性质归因分析（使用SHAP/LIME）
3. 可靠材料预测（不确定性量化）
4. 物理方程发现

应用场景：锂离子电池材料设计
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

try:
    from dftlammps.causal_ai import (
        CausalDiscoveryPipeline, CausalAlgorithm,
        IndependenceTest, Intervention,
        ExplainableMLPipeline,
        SHAPExplainer, LIMEExplainer,
        SymbolicRegression, EquationDiscovery,
        MechanisticModelPipeline, PhysicalConstraint
    )
    from dftlammps.uncertainty import (
        DeepEnsemble, StandardConformalPredictor,
        ConformalPredictionPipeline,
        EnsemblePrediction, ConformalPrediction
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure to run from the correct directory")
    raise


def generate_battery_material_data(n_samples: int = 500, 
                                    noise_level: float = 0.05) -> pd.DataFrame:
    """
    生成合成电池材料数据
    
    模拟锂离子电池正极材料的性质
    
    Returns:
        DataFrame包含材料特征和性质
    """
    np.random.seed(42)
    
    # 基础材料特征
    # 1. 晶体结构参数
    lattice_constant = np.random.uniform(8.0, 8.5, n_samples)  # Å
    
    # 2. 过渡金属比例
    tm_ratio = np.random.uniform(0.3, 0.9, n_samples)  # 过渡金属/锂比例
    
    # 3. 氧空位浓度 (受合成条件影响)
    oxygen_partial_pressure = np.random.uniform(0.1, 1.0, n_samples)
    oxygen_vacancy = 0.1 * (1 - oxygen_partial_pressure) + 0.01 * np.random.randn(n_samples)
    oxygen_vacancy = np.clip(oxygen_vacancy, 0, 0.1)
    
    # 4. 掺杂浓度
    dopant_conc = np.random.uniform(0, 0.15, n_samples)
    
    # 因果机制建模
    
    # 离子电导率 (受晶体结构和氧空位影响)
    ionic_conductivity = (
        0.5 * (lattice_constant - 8.0) +
        2.0 * oxygen_vacancy -
        0.3 * dopant_conc +
        0.1 * np.random.randn(n_samples)
    )
    ionic_conductivity = np.exp(ionic_conductivity)  # 转换为实际尺度
    
    # 电子电导率 (受过渡金属比例和氧空位影响)
    electronic_conductivity = (
        1.5 * tm_ratio +
        1.0 * oxygen_vacancy -
        0.5 * dopant_conc +
        0.2 * np.random.randn(n_samples)
    )
    electronic_conductivity = np.exp(electronic_conductivity)
    
    # 容量 (受过渡金属比例和掺杂影响)
    capacity = (
        150 * tm_ratio +
        20 * (1 - dopant_conc) -
        30 * oxygen_vacancy +
        5 * np.random.randn(n_samples)
    )
    
    # 循环稳定性 (受表面重构影响)
    surface_reconstruction = 0.3 * oxygen_vacancy + 0.1 * dopant_conc
    cycle_stability = (
        100 - 50 * surface_reconstruction -
        20 * np.abs(tm_ratio - 0.8) +
        5 * np.random.randn(n_samples)
    )
    
    # 热稳定性
    thermal_stability = (
        80 + 10 * lattice_constant -
        30 * oxygen_vacancy -
        20 * dopant_conc +
        3 * np.random.randn(n_samples)
    )
    
    # 成本因子
    cost_factor = (
        0.5 * tm_ratio +
        2.0 * dopant_conc +
        0.3 * (lattice_constant - 8.0) +
        0.1 * np.random.randn(n_samples)
    )
    
    # 构建DataFrame
    data = pd.DataFrame({
        'lattice_constant': lattice_constant,
        'tm_ratio': tm_ratio,
        'oxygen_pressure': oxygen_partial_pressure,
        'oxygen_vacancy': oxygen_vacancy,
        'dopant_conc': dopant_conc,
        'surface_reconstruction': surface_reconstruction,
        'ionic_conductivity': ionic_conductivity,
        'electronic_conductivity': electronic_conductivity,
        'capacity': capacity,
        'cycle_stability': cycle_stability,
        'thermal_stability': thermal_stability,
        'cost_factor': cost_factor
    })
    
    return data


class CausalMaterialsAnalysis:
    """材料因果分析"""
    
    def __init__(self):
        self.causal_pipeline = None
        self.causal_graph = None
        
    def discover_causal_structure(self, data: pd.DataFrame,
                                  algorithm: str = 'pc') -> Dict:
        """
        发现材料性质的因果结构
        
        Args:
            data: 材料数据
            algorithm: 算法类型 ('pc', 'ges', 'notears')
            
        Returns:
            因果发现结果
        """
        print("\n" + "="*60)
        print("1. 因果关系发现")
        print("="*60)
        
        # 选择算法
        algo_map = {
            'pc': CausalAlgorithm.PC,
            'ges': CausalAlgorithm.GES,
            'notears': CausalAlgorithm.NOTEARS
        }
        
        self.causal_pipeline = CausalDiscoveryPipeline(
            algorithm=algo_map.get(algorithm, CausalAlgorithm.PC),
            independence_test=IndependenceTest.PEARSON,
            alpha=0.05,
            verbose=False
        )
        
        # 拟合
        self.causal_graph = self.causal_pipeline.fit(data)
        
        print(f"\n使用的算法: {algorithm.upper()}")
        print(f"发现的因果边数量: {len(self.causal_graph.edges)}")
        print("\n因果边:")
        for edge in self.causal_graph.edges:
            print(f"  {edge.source} --> {edge.target} "
                  f"(weight={edge.weight:.3f}, confidence={edge.confidence:.3f})")
        
        # 识别中介变量
        results = {
            'graph': self.causal_graph,
            'edges': [(e.source, e.target) for e in self.causal_graph.edges],
            'causal_effects': {}
        }
        
        # 计算关键因果效应
        if 'capacity' in data.columns and 'tm_ratio' in data.columns:
            self.causal_pipeline.estimate_intervention_effect(data)
            ate = self.causal_pipeline.intervention_estimator.estimate_ate(
                'tm_ratio', 'capacity', [0.5, 0.8]
            )
            results['causal_effects']['tm_ratio_on_capacity'] = ate
            print(f"\n因果效应 - 过渡金属比例对容量的影响:")
            print(f"  ATE = {ate[0.8] - ate[0.5]:.2f}")
        
        return results
    
    def analyze_mediation(self, treatment: str, outcome: str) -> Dict:
        """
        中介分析
        
        Args:
            treatment: 处理变量
            outcome: 结果变量
            
        Returns:
            中介分析结果
        """
        if self.causal_graph is None:
            raise ValueError("Must run discover_causal_structure() first")
        
        # 找到中介变量
        mediators = self._find_mediators(treatment, outcome)
        
        print(f"\n中介分析: {treatment} -> {outcome}")
        print(f"潜在中介变量: {mediators}")
        
        return {
            'treatment': treatment,
            'outcome': outcome,
            'mediators': mediators
        }
    
    def _find_mediators(self, treatment: str, outcome: str) -> List[str]:
        """找到中介变量"""
        # 同时是处理的后代和结果祖先的变量
        descendants = self.causal_graph.get_descendants(treatment)
        ancestors = self.causal_graph.get_ancestors(outcome)
        
        return list(descendants & ancestors)


class ExplainableMaterialsPrediction:
    """可解释材料预测"""
    
    def __init__(self, model=None):
        self.model = model
        self.explainer = None
        
    def fit_explainable_model(self, X: pd.DataFrame, y: pd.Series):
        """
        拟合可解释模型
        
        Args:
            X: 特征
            y: 目标
        """
        print("\n" + "="*60)
        print("2. 可解释材料性质预测")
        print("="*60)
        
        # 如果没有提供模型，使用随机森林
        if self.model is None:
            try:
                from sklearn.ensemble import RandomForestRegressor
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            except ImportError:
                print("scikit-learn not available")
                return
        
        # 拟合模型
        self.model.fit(X, y)
        print(f"\n模型类型: {type(self.model).__name__}")
        print(f"训练样本数: {len(X)}")
        print(f"特征数: {X.shape[1]}")
        print(f"特征: {list(X.columns)}")
        
        # 创建解释管道
        self.explainer = ExplainableMLPipeline(
            model=self.model,
            feature_names=list(X.columns)
        )
        self.explainer.fit(X.values, y.values)
        
    def analyze_feature_importance(self, X: pd.DataFrame) -> Dict:
        """
        分析特征重要性
        
        Args:
            X: 数据
            
        Returns:
            特征重要性分析结果
        """
        print("\n--- 特征重要性分析 ---")
        
        # 置换重要性
        from dftlammps.causal_ai import PermutationImportance
        perm_imp = PermutationImportance(self.model)
        importance = perm_imp.compute(X.values, self.model.predict(X.values))
        
        print("\n置换重要性 (Top 5):")
        for i, imp in enumerate(importance[:5]):
            print(f"  {i+1}. {imp.feature_name}: {imp.importance:.4f} ± {imp.std:.4f}")
        
        return {
            'permutation_importance': importance
        }
    
    def explain_instance(self, X: pd.DataFrame, instance_idx: int = 0) -> Dict:
        """
        解释单个预测
        
        Args:
            X: 数据
            instance_idx: 实例索引
            
        Returns:
            解释结果
        """
        print(f"\n--- 实例 {instance_idx} 的解释 ---")
        
        explanations = self.explainer.explain(X.values, instance_idx)
        
        results = {}
        for method, exp in explanations.items():
            print(f"\n{method.upper()} 解释:")
            print(f"  预测值: {exp.prediction:.4f}")
            print(f"  基线值: {exp.base_value:.4f}")
            print(f"  Top 5 贡献特征:")
            for i, (feat, contrib) in enumerate(exp.top_features(5)):
                direction = "增加" if contrib > 0 else "减少"
                print(f"    {i+1}. {feat}: {contrib:.4f} ({direction})")
            
            results[method] = exp
        
        return results
    
    def stability_check(self, X: pd.DataFrame, 
                       n_perturbations: int = 10) -> Dict:
        """
        解释稳定性检查
        
        Args:
            X: 数据
            n_perturbations: 扰动次数
            
        Returns:
            稳定性分析结果
        """
        print("\n--- 解释稳定性分析 ---")
        
        stability = self.explainer.stability_analysis(
            X.values, 
            n_perturbations=n_perturbations,
            noise_level=0.01
        )
        
        print("\n各方法的稳定性 (余弦相似度):")
        for method, metrics in stability.items():
            print(f"  {method}: {metrics['mean_similarity']:.4f} "
                  f"± {metrics['std_similarity']:.4f}")
        
        return stability


class ReliableMaterialsPrediction:
    """可靠的材料预测"""
    
    def __init__(self):
        self.ensemble_model = None
        self.conformal_predictor = None
        
    def fit_with_uncertainty(self, X: np.ndarray, y: np.ndarray,
                             method: str = 'ensemble') -> Dict:
        """
        带不确定性的模型拟合
        
        Args:
            X: 特征
            y: 目标
            method: 方法 ('ensemble', 'conformal')
            
        Returns:
            拟合结果
        """
        print("\n" + "="*60)
        print("3. 可靠材料预测（不确定性量化）")
        print("="*60)
        
        try:
            import torch
            import torch.nn as nn
            HAS_TORCH = True
        except ImportError:
            HAS_TORCH = False
            print("PyTorch not available, using sklearn models")
        
        if method == 'ensemble':
            return self._fit_ensemble(X, y, HAS_TORCH)
        elif method == 'conformal':
            return self._fit_conformal(X, y, HAS_TORCH)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _fit_ensemble(self, X, y, has_torch):
        """拟合集成模型"""
        print("\n方法: 深度集合 (Deep Ensemble)")
        
        if has_torch:
            import torch.nn as nn
            
            # 定义神经网络构建函数
            input_dim = X.shape[1]
            
            def build_model():
                return nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)
                )
            
            self.ensemble_model = DeepEnsemble(
                model_builder=build_model,
                n_members=5,
                bootstrap=True
            )
            
            self.ensemble_model.fit(
                X, y,
                epochs=200,
                batch_size=32,
                learning_rate=1e-3,
                verbose=False
            )
            
            print(f"集成成员数: 5")
            diversity = self.ensemble_model.diversity_metrics()
            print(f"多样性指标: {diversity}")
        else:
            from sklearn.ensemble import RandomForestRegressor
            
            # 使用sklearn的集成
            models = [
                RandomForestRegressor(n_estimators=50, random_state=i, max_depth=10+i)
                for i in range(5)
            ]
            
            for model in models:
                model.fit(X, y)
            
            self.ensemble_model = models
            print("使用RandomForest集成（5个成员）")
        
        return {'method': 'ensemble', 'status': 'fitted'}
    
    def _fit_conformal(self, X, y, has_torch):
        """拟合共形预测器"""
        print("\n方法: 共形预测 (Conformal Prediction)")
        
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
        except ImportError:
            print("scikit-learn not available")
            return {'method': 'conformal', 'status': 'failed'}
        
        base_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        self.conformal_predictor = StandardConformalPredictor(
            base_model,
            nonconformity='absolute'
        )
        
        # 分割数据
        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # 拟合和校准
        self.conformal_predictor.fit(X_train, y_train)
        self.conformal_predictor.calibrate(X_cal, y_cal, alpha=0.1)
        
        print(f"校准集大小: {len(X_cal)}")
        print(f"目标覆盖率: 90%")
        
        return {'method': 'conformal', 'status': 'fitted'}
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Dict:
        """
        带不确定性的预测
        
        Args:
            X: 输入数据
            
        Returns:
            预测结果和不确定性
        """
        results = {}
        
        if self.ensemble_model is not None:
            if isinstance(self.ensemble_model, list):
                # sklearn模型列表
                predictions = [m.predict(X) for m in self.ensemble_model]
                pred_matrix = np.column_stack(predictions)
                
                results['ensemble'] = {
                    'mean': np.mean(pred_matrix, axis=1),
                    'std': np.std(pred_matrix, axis=1),
                    'prediction_interval': (
                        np.percentile(pred_matrix, 5, axis=1),
                        np.percentile(pred_matrix, 95, axis=1)
                    )
                }
            else:
                # DeepEnsemble
                pred = self.ensemble_model.predict(X)
                lower, upper = pred.prediction_interval(confidence=0.9)
                
                results['ensemble'] = {
                    'mean': pred.mean.flatten(),
                    'std': np.sqrt(pred.variance.flatten()),
                    'prediction_interval': (lower.flatten(), upper.flatten())
                }
            
            print(f"\n集成预测不确定性:")
            print(f"  平均预测: {np.mean(results['ensemble']['mean']):.4f}")
            print(f"  平均标准差: {np.mean(results['ensemble']['std']):.4f}")
        
        if self.conformal_predictor is not None:
            pred_cp = self.conformal_predictor.predict(X, alpha=0.1)
            
            results['conformal'] = {
                'point': pred_cp.point_prediction,
                'lower': pred_cp.lower_bound,
                'upper': pred_cp.upper_bound,
                'interval_width': pred_cp.interval_width
            }
            
            print(f"\n共形预测区间:")
            print(f"  平均区间宽度: {np.mean(pred_cp.interval_width):.4f}")
            print(f"  平均点预测: {np.mean(pred_cp.point_prediction):.4f}")
        
        return results


class SymbolicEquationDiscovery:
    """符号方程发现"""
    
    def __init__(self):
        self.symbolic_regressor = None
        self.equation_discoverer = None
        
    def discover_equations(self, data: pd.DataFrame,
                          target: str,
                          predictors: List[str],
                          method: str = 'symbolic') -> Dict:
        """
        发现材料性质的符号方程
        
        Args:
            data: 材料数据
            target: 目标变量
            predictors: 预测变量
            method: 方法 ('symbolic', 'sparse')
            
        Returns:
            发现的方程
        """
        print("\n" + "="*60)
        print("4. 符号方程发现")
        print("="*60)
        
        X = data[predictors].values
        y = data[target].values
        
        if method == 'symbolic':
            return self._symbolic_regression(X, y, predictors, target)
        elif method == 'sparse':
            return self._sparse_discovery(X, y, predictors, target)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _symbolic_regression(self, X, y, features, target):
        """符号回归"""
        print("\n方法: 遗传编程符号回归")
        
        self.symbolic_regressor = SymbolicRegression(
            population_size=100,
            generations=50,
            mutation_rate=0.2,
            crossover_rate=0.7,
            max_depth=4,
            parsimony_coefficient=0.01
        )
        
        equation = self.symbolic_regressor.fit(X, y, variable_names=features)
        
        print(f"\n发现的方程:")
        print(f"  {target} = {equation.expression}")
        print(f"  复杂度: {equation.complexity}")
        print(f"  R² 分数: {equation.r2_score:.4f}")
        
        if equation.parameters:
            print(f"  优化参数: {equation.parameters}")
        
        return {
            'method': 'symbolic_regression',
            'equation': equation,
            'expression': equation.expression,
            'r2_score': equation.r2_score
        }
    
    def _sparse_discovery(self, X, y, features, target):
        """稀疏回归发现"""
        print("\n方法: 稀疏回归方程发现")
        
        self.equation_discoverer = EquationDiscovery(
            library_size=50,
            sparsity_threshold=0.1,
            alpha=0.01
        )
        
        equation = self.equation_discoverer.discover(X, y, method='sparse_regression')
        
        print(f"\n发现的方程:")
        print(f"  {target} = {equation.expression}")
        print(f"  非零项数: {equation.complexity}")
        print(f"  R² 分数: {equation.r2_score:.4f}")
        
        return {
            'method': 'sparse_regression',
            'equation': equation,
            'expression': equation.expression,
            'r2_score': equation.r2_score
        }


def run_full_demo():
    """运行完整演示"""
    print("\n" + "#"*60)
    print("# 因果AI与可解释材料发现 - 完整演示")
    print("#"*60)
    print("\n应用场景: 锂离子电池正极材料设计")
    
    # 1. 生成数据
    print("\n" + "-"*60)
    print("数据生成")
    print("-"*60)
    
    data = generate_battery_material_data(n_samples=500)
    print(f"生成 {len(data)} 个材料样本")
    print(f"\n数据预览:")
    print(data.head())
    print(f"\n数据统计:")
    print(data.describe())
    
    # 2. 因果发现
    causal_analysis = CausalMaterialsAnalysis()
    causal_results = causal_analysis.discover_causal_structure(data, algorithm='pc')
    
    # 中介分析示例
    mediation_results = causal_analysis.analyze_mediation(
        'tm_ratio', 'capacity'
    )
    
    # 3. 可解释预测
    # 选择特征和目标
    feature_cols = ['lattice_constant', 'tm_ratio', 'oxygen_vacancy', 
                   'dopant_conc', 'surface_reconstruction']
    target_col = 'capacity'
    
    X = data[feature_cols]
    y = data[target_col]
    
    explainable = ExplainableMaterialsPrediction()
    explainable.fit_explainable_model(X, y)
    
    # 特征重要性
    importance_results = explainable.analyze_feature_importance(X)
    
    # 实例解释
    instance_explanation = explainable.explain_instance(X, instance_idx=0)
    
    # 稳定性检查
    stability_results = explainable.stability_check(X, n_perturbations=5)
    
    # 4. 可靠预测
    reliable = ReliableMaterialsPrediction()
    
    # 使用集成方法
    reliable.fit_with_uncertainty(X.values, y.values, method='ensemble')
    uncertainty_results = reliable.predict_with_uncertainty(X.values[:10])
    
    # 使用共形预测
    reliable_cp = ReliableMaterialsPrediction()
    reliable_cp.fit_with_uncertainty(X.values, y.values, method='conformal')
    cp_results = reliable_cp.predict_with_uncertainty(X.values[:10])
    
    # 5. 符号方程发现
    symbolic = SymbolicEquationDiscovery()
    equation_results = symbolic.discover_equations(
        data,
        target='capacity',
        predictors=['lattice_constant', 'tm_ratio', 'oxygen_vacancy', 'dopant_conc'],
        method='symbolic'
    )
    
    # 稀疏回归发现
    sparse_results = symbolic.discover_equations(
        data,
        target='capacity',
        predictors=['lattice_constant', 'tm_ratio', 'oxygen_vacancy', 'dopant_conc'],
        method='sparse'
    )
    
    # 总结
    print("\n" + "="*60)
    print("演示总结")
    print("="*60)
    
    print("\n完成的主要任务:")
    print("  ✓ 因果关系发现: 识别了材料性质的因果结构")
    print("  ✓ 性质归因分析: 使用SHAP/LIME解释预测")
    print("  ✓ 可靠预测: 集成方法和共形预测提供不确定性量化")
    print("  ✓ 符号方程发现: 发现了可解释的数学方程")
    
    print("\n关键发现:")
    print(f"  • 过渡金属比例对容量有直接因果影响")
    print(f"  • 符号回归发现的方程 R² = {equation_results['r2_score']:.4f}")
    print(f"  • 共形预测提供 90% 覆盖率保证")
    
    print("\n" + "#"*60)
    print("# 演示完成")
    print("#"*60)
    
    return {
        'data': data,
        'causal_results': causal_results,
        'explanation_results': instance_explanation,
        'uncertainty_results': uncertainty_results,
        'equation_results': equation_results
    }


if __name__ == "__main__":
    results = run_full_demo()
