"""
材料性质预测器 - Material Property Predictor
可解释的材料性质预测系统
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import warnings


@dataclass
class Material:
    """材料定义"""
    name: str
    composition: Dict[str, float]
    structure_features: Dict[str, float]
    processing_params: Dict[str, float]
    properties: Optional[Dict[str, float]] = None


@dataclass
class PredictionExplanation:
    """预测解释"""
    prediction: float
    confidence: float
    feature_importance: Dict[str, float]
    similar_materials: List[Tuple[str, float]]
    reasoning_chain: List[str]
    uncertainty_estimate: Tuple[float, float]


class MaterialPropertyPredictor:
    """
    可解释材料性质预测器
    结合机器学习与符号推理进行可解释预测
    """
    
    def __init__(self, target_property: str):
        self.target_property = target_property
        self.model = None
        self.explainer = None
        self.material_database: List[Material] = []
        self.feature_importance_history: List[Dict] = []
        
        # 材料知识图谱
        self.knowledge_graph = None
        
        # 预测历史
        self.prediction_history: List[Dict] = []
    
    def initialize(self):
        """初始化系统"""
        import sys
        sys.path.insert(0, '/root/.openclaw/workspace/dftlammps')
        
        from neuro_symbolic.neural_perception import (
            NeuralPerceptionSystem, FeatureConfig, PatternConfig
        )
        from neuro_symbolic.explainable_ai import ExplainableAI
        
        # 初始化神经感知
        feature_config = FeatureConfig(
            input_dim=15,
            hidden_dims=[64, 128, 64],
            output_dim=32
        )
        self.neural_system = NeuralPerceptionSystem(feature_config)
        
        # 初始化知识图谱
        self._build_knowledge_graph()
        
        print(f"材料性质预测器已初始化: 目标属性 = {self.target_property}")
    
    def _build_knowledge_graph(self):
        """构建材料知识图谱"""
        import sys
        sys.path.insert(0, '/root/.openclaw/workspace/dftlammps')
        from neuro_symbolic.symbolic_reasoning import KnowledgeGraph
        
        kg = KnowledgeGraph()
        
        # 添加材料类别
        categories = ["metal", "ceramic", "polymer", "composite", "semiconductor"]
        for cat in categories:
            kg.add_node(cat, "Category")
        
        # 添加性质
        properties = ["hardness", "conductivity", "strength", "elasticity", "density"]
        for prop in properties:
            kg.add_node(prop, "Property")
        
        # 添加关系
        kg.add_edge("metal", "conductivity", "has_property", weight=0.9)
        kg.add_edge("ceramic", "hardness", "has_property", weight=0.85)
        kg.add_edge("polymer", "elasticity", "has_property", weight=0.8)
        
        self.knowledge_graph = kg
    
    def train(self, materials: List[Material], validation_split: float = 0.2):
        """
        训练预测模型
        
        Args:
            materials: 材料数据集
            validation_split: 验证集比例
        """
        self.material_database = materials
        
        # 准备数据
        X, y = self._prepare_training_data(materials)
        
        # 划分训练集和验证集
        n_total = len(X)
        n_val = int(n_total * validation_split)
        n_train = n_total - n_val
        
        indices = np.random.permutation(n_total)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # 训练简单模型（这里使用简化版）
        self.model = self._train_simple_model(X_train, y_train)
        
        # 初始化可解释AI
        import sys
        sys.path.insert(0, '/root/.openclaw/workspace/dftlammps')
        from neuro_symbolic.explainable_ai import ExplainableAI
        
        self.explainer = ExplainableAI(
            self.model,
            feature_names=self._get_feature_names()
        )
        
        # 添加解释方法
        self.explainer.add_shap(background_data=X_train[:50])
        self.explainer.add_lime(mode="regression")
        
        # 评估
        val_predictions = self.model(X_val)
        mse = np.mean((val_predictions - y_val) ** 2)
        
        print(f"训练完成: 训练集={n_train}, 验证集={n_val}")
        print(f"验证集MSE: {mse:.4f}")
    
    def _prepare_training_data(
        self,
        materials: List[Material]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """准备训练数据"""
        X_list = []
        y_list = []
        
        for mat in materials:
            if mat.properties and self.target_property in mat.properties:
                features = self._extract_features(mat)
                X_list.append(features)
                y_list.append(mat.properties[self.target_property])
        
        return np.array(X_list), np.array(y_list)
    
    def _extract_features(self, material: Material) -> np.ndarray:
        """提取特征向量"""
        features = []
        
        # 成分特征
        features.extend([
            material.composition.get('element_A', 0),
            material.composition.get('element_B', 0),
            material.composition.get('element_C', 0),
            material.composition.get('impurity', 0),
        ])
        
        # 结构特征
        features.extend([
            material.structure_features.get('crystal_structure_type', 0),
            material.structure_features.get('lattice_constant', 0),
            material.structure_features.get('coordination_number', 0),
            material.structure_features.get('defect_density', 0),
            material.structure_features.get('grain_size', 0),
            material.structure_features.get('porosity', 0),
        ])
        
        # 工艺参数
        features.extend([
            material.processing_params.get('sintering_temp', 0),
            material.processing_params.get('sintering_time', 0),
            material.processing_params.get('cooling_rate', 0),
            material.processing_params.get('pressure', 0),
            material.processing_params.get('atmosphere', 0),
        ])
        
        return np.array(features)
    
    def _get_feature_names(self) -> List[str]:
        """获取特征名称"""
        return [
            'element_A', 'element_B', 'element_C', 'impurity',
            'crystal_structure_type', 'lattice_constant', 'coordination_number',
            'defect_density', 'grain_size', 'porosity',
            'sintering_temp', 'sintering_time', 'cooling_rate', 'pressure', 'atmosphere'
        ]
    
    def _train_simple_model(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Any:
        """训练简单预测模型"""
        # 使用线性回归作为简化示例
        from numpy.linalg import lstsq
        
        X_b = np.column_stack([np.ones(len(X)), X])
        coeffs = lstsq(X_b, y, rcond=None)[0]
        
        def predict(X_new):
            X_new_b = np.column_stack([np.ones(len(X_new)), X_new])
            return X_new_b @ coeffs
        
        return predict
    
    def predict(
        self,
        material: Material,
        return_explanation: bool = True
    ) -> Union[float, Tuple[float, PredictionExplanation]]:
        """
        预测材料性质
        
        Args:
            material: 待预测材料
            return_explanation: 是否返回解释
        
        Returns:
            预测值和可选的解释
        """
        # 提取特征
        features = self._extract_features(material).reshape(1, -1)
        
        # 预测
        prediction = self.model(features)[0]
        
        if not return_explanation:
            return prediction
        
        # 生成解释
        explanation = self._generate_explanation(material, features, prediction)
        
        # 记录预测历史
        self.prediction_history.append({
            'material': material.name,
            'prediction': prediction,
            'explanation': explanation
        })
        
        return prediction, explanation
    
    def _generate_explanation(
        self,
        material: Material,
        features: np.ndarray,
        prediction: float
    ) -> PredictionExplanation:
        """生成预测解释"""
        # 特征重要性
        feature_importance = self._compute_feature_importance(features)
        
        # 找到相似材料
        similar_materials = self._find_similar_materials(material)
        
        # 生成推理链
        reasoning_chain = self._generate_reasoning_chain(
            material, feature_importance
        )
        
        # 不确定性估计
        uncertainty = self._estimate_uncertainty(features)
        
        # 置信度计算
        confidence = self._compute_confidence(features, similar_materials)
        
        return PredictionExplanation(
            prediction=prediction,
            confidence=confidence,
            feature_importance=feature_importance,
            similar_materials=similar_materials,
            reasoning_chain=reasoning_chain,
            uncertainty_estimate=uncertainty
        )
    
    def _compute_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        """计算特征重要性"""
        # 使用神经感知系统
        importance = self.neural_system.get_feature_importance(features)
        
        feature_names = self._get_feature_names()
        return {
            name: float(imp)
            for name, imp in zip(feature_names, importance)
        }
    
    def _find_similar_materials(
        self,
        material: Material,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """找到相似材料"""
        features = self._extract_features(material)
        
        similarities = []
        for db_material in self.material_database:
            db_features = self._extract_features(db_material)
            
            # 余弦相似度
            similarity = np.dot(features, db_features) / (
                np.linalg.norm(features) * np.linalg.norm(db_features) + 1e-10
            )
            similarities.append((db_material.name, float(similarity)))
        
        # 排序并返回前k个
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _generate_reasoning_chain(
        self,
        material: Material,
        feature_importance: Dict[str, float]
    ) -> List[str]:
        """生成推理链"""
        chain = []
        
        # 找出最重要的特征
        top_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # 基于知识图谱的推理
        if self.knowledge_graph:
            # 查找材料类别相关的性质
            for category in ["metal", "ceramic", "polymer"]:
                related = self.knowledge_graph.get_related(category, "has_property")
                if related:
                    chain.append(
                        f"基于知识图谱: {category} 材料通常具有 "
                        f"{', '.join([n.label for n in related[:2]])}"
                    )
        
        # 特征重要性推理
        for feature, importance in top_features:
            chain.append(
                f"{feature} 对预测贡献最大 (重要性: {importance:.3f})"
            )
        
        # 基于成分和工艺的推理
        if material.composition.get('impurity', 0) < 0.01:
            chain.append("高纯度成分预期带来更好的性能")
        
        if material.processing_params.get('sintering_temp', 0) > 1000:
            chain.append("高温烧结可能提高材料致密度")
        
        return chain
    
    def _estimate_uncertainty(self, features: np.ndarray) -> Tuple[float, float]:
        """估计预测不确定性"""
        # 基于与训练数据的距离估计不确定性
        if not self.material_database:
            return (0, 0)
        
        # 计算到训练数据的平均距离
        distances = []
        for mat in self.material_database[:50]:  # 样本子集
            mat_features = self._extract_features(mat)
            dist = np.linalg.norm(features - mat_features)
            distances.append(dist)
        
        avg_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        # 距离越大，不确定性越高
        uncertainty = min(avg_distance / 10, 1.0)
        
        return (prediction - uncertainty * 10, prediction + uncertainty * 10) if 'prediction' in dir() else (0, 0)
    
    def _compute_confidence(
        self,
        features: np.ndarray,
        similar_materials: List[Tuple[str, float]]
    ) -> float:
        """计算预测置信度"""
        # 基于相似度计算置信度
        if similar_materials:
            avg_similarity = np.mean([sim for _, sim in similar_materials])
            return float(avg_similarity)
        return 0.5
    
    def generate_counterfactual_suggestions(
        self,
        material: Material,
        target_value: float
    ) -> List[Dict[str, Any]]:
        """
        生成反事实改进建议
        
        Args:
            material: 当前材料
            target_value: 目标性质值
        
        Returns:
            改进建议列表
        """
        current_prediction, explanation = self.predict(material)
        
        suggestions = []
        
        gap = target_value - current_prediction
        
        if gap <= 0:
            suggestions.append({
                "message": "当前材料已达到或超过目标性能",
                "action": "无需改进"
            })
            return suggestions
        
        # 基于特征重要性生成建议
        for feature, importance in sorted(
            explanation.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]:
            if feature in material.composition:
                suggestions.append({
                    "target_property": self.target_property,
                    "current_value": current_prediction,
                    "target_value": target_value,
                    "suggested_change": f"调整 {feature} 成分",
                    "expected_impact": f"高（重要性: {importance:.3f}）",
                    "priority": "high" if importance > 0.5 else "medium"
                })
            elif feature in material.processing_params:
                suggestions.append({
                    "target_property": self.target_property,
                    "current_value": current_prediction,
                    "target_value": target_value,
                    "suggested_change": f"优化 {feature} 工艺参数",
                    "expected_impact": f"中（重要性: {importance:.3f}）",
                    "priority": "medium"
                })
        
        return suggestions
    
    def generate_full_report(self) -> str:
        """生成完整报告"""
        report = []
        report.append("=" * 70)
        report.append(f"材料性质预测报告: {self.target_property}")
        report.append("=" * 70)
        
        report.append(f"\n预测历史记录数: {len(self.prediction_history)}")
        
        if self.prediction_history:
            predictions = [p['prediction'] for p in self.prediction_history]
            report.append(f"预测值范围: {min(predictions):.2f} - {max(predictions):.2f}")
            report.append(f"预测均值: {np.mean(predictions):.2f} ± {np.std(predictions):.2f}")
        
        report.append(f"\n数据库中材料数: {len(self.material_database)}")
        
        # 特征重要性统计
        if self.feature_importance_history:
            avg_importance = {}
            for fi in self.feature_importance_history:
                for k, v in fi.items():
                    avg_importance[k] = avg_importance.get(k, 0) + v
            
            for k in avg_importance:
                avg_importance[k] /= len(self.feature_importance_history)
            
            report.append("\n平均特征重要性:")
            for feature, importance in sorted(
                avg_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]:
                report.append(f"  {feature}: {importance:.3f}")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)


def generate_sample_materials(n_samples: int = 100) -> List[Material]:
    """生成示例材料数据"""
    np.random.seed(42)
    materials = []
    
    material_types = [
        ("Al2O3_ceramic", "ceramic"),
        ("SiC_ceramic", "ceramic"),
        ("TiO2_ceramic", "ceramic"),
        ("Cu_metal", "metal"),
        ("Al_alloy", "metal"),
        ("PE_polymer", "polymer"),
        ("PVDF_polymer", "polymer"),
        ("C_fiber_composite", "composite"),
    ]
    
    for i in range(n_samples):
        name, mat_type = material_types[i % len(material_types)]
        name = f"{name}_{i}"
        
        # 成分
        composition = {
            'element_A': np.random.uniform(0.3, 0.9),
            'element_B': np.random.uniform(0, 0.3),
            'element_C': np.random.uniform(0, 0.2),
            'impurity': np.random.uniform(0, 0.05),
        }
        
        # 结构特征
        structure_features = {
            'crystal_structure_type': np.random.choice([0, 1, 2]),
            'lattice_constant': np.random.uniform(3, 6),
            'coordination_number': np.random.choice([4, 6, 8]),
            'defect_density': np.random.uniform(0, 0.1),
            'grain_size': np.random.uniform(0.1, 50),
            'porosity': np.random.uniform(0, 0.3),
        }
        
        # 工艺参数
        processing_params = {
            'sintering_temp': np.random.uniform(800, 1500),
            'sintering_time': np.random.uniform(1, 10),
            'cooling_rate': np.random.uniform(1, 100),
            'pressure': np.random.uniform(1, 100),
            'atmosphere': np.random.choice([0, 1, 2]),
        }
        
        # 性质（基于特征生成）
        hardness = (
            50 * composition['element_A'] +
            30 * (1 - structure_features['porosity']) +
            0.02 * processing_params['sintering_temp'] +
            np.random.normal(0, 5)
        )
        
        properties = {
            'hardness': max(0, hardness),
            'density': np.random.uniform(2, 8),
            'conductivity': np.random.uniform(0.01, 100),
        }
        
        materials.append(Material(
            name=name,
            composition=composition,
            structure_features=structure_features,
            processing_params=processing_params,
            properties=properties
        ))
    
    return materials


def run_material_property_demo():
    """运行材料性质预测演示"""
    print("=" * 70)
    print("材料性质预测器演示 - 可解释预测")
    print("=" * 70)
    
    # 创建预测器
    print("\n1. 初始化预测器...")
    predictor = MaterialPropertyPredictor(target_property="hardness")
    predictor.initialize()
    
    # 生成训练数据
    print("\n2. 生成训练数据...")
    training_materials = generate_sample_materials(n_samples=100)
    print(f"   生成了 {len(training_materials)} 个训练样本")
    
    # 训练模型
    print("\n3. 训练预测模型...")
    predictor.train(training_materials, validation_split=0.2)
    
    # 预测新材料
    print("\n4. 预测新材料性质...")
    test_materials = generate_sample_materials(n_samples=5)
    
    for mat in test_materials:
        prediction, explanation = predictor.predict(mat)
        
        print(f"\n   材料: {mat.name}")
        print(f"   预测硬度: {prediction:.2f}")
        print(f"   置信度: {explanation.confidence:.2%}")
        print(f"   不确定性区间: [{explanation.uncertainty_estimate[0]:.2f}, "
              f"{explanation.uncertainty_estimate[1]:.2f}]")
        
        print("   最重要特征:")
        for feature, importance in sorted(
            explanation.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]:
            print(f"     - {feature}: {importance:.3f}")
        
        print("   推理链:")
        for i, reasoning in enumerate(explanation.reasoning_chain[:3], 1):
            print(f"     {i}. {reasoning}")
        
        print("   相似材料:")
        for name, similarity in explanation.similar_materials:
            print(f"     - {name}: {similarity:.3f}")
    
    # 反事实建议
    print("\n5. 生成改进建议...")
    target_material = test_materials[0]
    current_pred, _ = predictor.predict(target_material)
    
    suggestions = predictor.generate_counterfactual_suggestions(
        target_material,
        target_value=current_pred * 1.2  # 目标提高20%
    )
    
    print(f"\n   目标: 将硬度从 {current_pred:.2f} 提高到 {current_pred * 1.2:.2f}")
    for suggestion in suggestions[:3]:
        print(f"   - {suggestion['suggested_change']}")
        print(f"     预期影响: {suggestion['expected_impact']}")
        print(f"     优先级: {suggestion['priority']}")
    
    # 生成完整报告
    print("\n6. 生成完整报告...")
    report = predictor.generate_full_report()
    print(report)
    
    print("\n" + "=" * 70)
    print("演示完成!")
    print("=" * 70)


def demo():
    """演示函数"""
    return run_material_property_demo()


if __name__ == "__main__":
    demo()
