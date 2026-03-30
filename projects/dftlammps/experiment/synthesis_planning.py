"""
合成规划模块 - Synthesis Planning Module

实现：
- 可合成性预测
- 合成路径规划
- 前驱体选择

基于机器学习模型和化学知识库
"""

import os
import json
import pickle
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from pathlib import Path
import numpy as np
from collections import defaultdict
import networkx as nx
from enum import Enum

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SynthesisDifficulty(Enum):
    """合成难度等级"""
    EASY = "easy"
    MODERATE = "moderate"
    DIFFICULT = "difficult"
    VERY_DIFFICULT = "very_difficult"
    IMPOSSIBLE = "impossible"


class ReactionType(Enum):
    """反应类型"""
    SOLID_STATE = "solid_state"
    SOL_GEL = "sol_gel"
    HYDROTHERMAL = "hydrothermal"
    CO_PRECIPITATION = "co_precipitation"
    MELT_SYNTHESIS = "melt_synthesis"
    VAPOR_TRANSPORT = "vapor_transport"
    ELECTROCHEMICAL = "electrochemical"
    MECHANOCHEMICAL = "mechanochemical"


@dataclass
class ChemicalCompound:
    """化学化合物"""
    formula: str
    name: str = ""
    molecular_weight: float = 0.0
    melting_point: Optional[float] = None  # K
    boiling_point: Optional[float] = None  # K
    stability: float = 1.0  # 0-1稳定性评分
    toxicity: float = 0.0  # 0-1毒性评分
    cost_per_gram: float = 0.0  # USD/g
    availability: float = 1.0  # 0-1可获得性
    storage_conditions: List[str] = field(default_factory=list)
    hazards: List[str] = field(default_factory=list)
    crystal_structure: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Precursor:
    """前驱体"""
    compound: ChemicalCompound
    stoichiometry: float
    purity_required: float = 0.99
    alternative_sources: List[ChemicalCompound] = field(default_factory=list)
    
    def total_cost(self) -> float:
        """计算总成本"""
        return self.compound.cost_per_gram * self.stoichiometry
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "compound": self.compound.to_dict(),
            "stoichiometry": self.stoichiometry,
            "purity_required": self.purity_required
        }


@dataclass
class SynthesisStep:
    """合成步骤"""
    step_number: int
    description: str
    reaction_type: ReactionType
    temperature: float  # K
    pressure: float = 101325  # Pa
    duration: float = 3600  # seconds
    atmosphere: str = "air"
    precursors: List[Precursor] = field(default_factory=list)
    products: List[ChemicalCompound] = field(default_factory=list)
    yield_estimate: float = 0.9
    critical_parameters: List[str] = field(default_factory=list)
    safety_notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "description": self.description,
            "reaction_type": self.reaction_type.value,
            "temperature": self.temperature,
            "pressure": self.pressure,
            "duration": self.duration,
            "atmosphere": self.atmosphere,
            "precursors": [p.to_dict() for p in self.precursors],
            "yield_estimate": self.yield_estimate
        }


@dataclass
class SynthesisRoute:
    """合成路线"""
    target: ChemicalCompound
    steps: List[SynthesisStep]
    total_yield: float = 0.0
    estimated_cost: float = 0.0
    estimated_time: float = 0.0
    difficulty_score: float = 0.0
    success_probability: float = 0.0
    alternative_routes: List['SynthesisRoute'] = field(default_factory=list)
    
    def __post_init__(self):
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """计算路线指标"""
        if not self.steps:
            return
        
        # 计算总产率
        self.total_yield = np.prod([step.yield_estimate for step in self.steps])
        
        # 计算总成本
        self.estimated_cost = sum(
            sum(p.total_cost() for p in step.precursors)
            for step in self.steps
        )
        
        # 计算总时间
        self.estimated_time = sum(step.duration for step in self.steps)
        
        # 计算难度评分
        difficulty_map = {
            ReactionType.SOLID_STATE: 1.0,
            ReactionType.CO_PRECIPITATION: 1.5,
            ReactionType.SOL_GEL: 2.0,
            ReactionType.HYDROTHERMAL: 2.5,
            ReactionType.MELT_SYNTHESIS: 2.0,
            ReactionType.VAPOR_TRANSPORT: 3.0,
            ReactionType.ELECTROCHEMICAL: 2.5,
            ReactionType.MECHANOCHEMICAL: 1.5
        }
        
        self.difficulty_score = np.mean([
            difficulty_map.get(step.reaction_type, 2.0) + 
            (step.temperature - 300) / 1000  # 温度惩罚
            for step in self.steps
        ])
        
        # 成功概率（基于产率和难度）
        self.success_probability = self.total_yield * (1 - self.difficulty_score / 10)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "total_yield": self.total_yield,
            "estimated_cost": self.estimated_cost,
            "estimated_time": self.estimated_time,
            "difficulty_score": self.difficulty_score,
            "success_probability": self.success_probability
        }


class SynthesisPredictor(ABC):
    """合成可行性预测器基类"""
    
    @abstractmethod
    def predict(self, target_formula: str, 
                available_precursors: List[ChemicalCompound]) -> Tuple[bool, float]:
        """
        预测合成可行性
        
        Returns:
            (是否可合成, 成功概率)
        """
        pass
    
    @abstractmethod
    def get_difficulty(self, target_formula: str) -> SynthesisDifficulty:
        """获取合成难度"""
        pass


class KnowledgeBasedPredictor(SynthesisPredictor):
    """基于知识库的预测器"""
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        self.compound_rules = self._init_compound_rules()
        
    def _load_knowledge_base(self, path: Optional[str]) -> Dict[str, Any]:
        """加载知识库"""
        default_kb = {
            "stable_oxides": ["SiO2", "Al2O3", "TiO2", "ZrO2", "Fe2O3", "Fe3O4"],
            "unstable_oxides": ["Au2O3", "Ag2O"],
            "stable_sulfides": ["ZnS", "CdS", "PbS", "Cu2S", "FeS2"],
            "water_soluble": ["NaCl", "KCl", "LiCl", "Na2SO4", "KNO3"],
            "air_sensitive": ["Li", "Na", "K", "Ca", "P2S5", "Li2S"],
            "moisture_sensitive": ["Li3N", "LiP", "Li2S", "TiCl4", "SiCl4"],
            "high_melting": ["WC", "TiC", "SiC", "Al2O3", "MgO", "ZrO2"],
            "synthesis_templates": {
                "oxide": {
                    "methods": ["solid_state", "sol_gel", "hydrothermal"],
                    "typical_temp": [500, 1200]
                },
                "sulfide": {
                    "methods": ["solid_state", "hydrothermal", "vapor_transport"],
                    "typical_temp": [300, 800],
                    "atmosphere": "inert"
                },
                "nitride": {
                    "methods": ["solid_state", "ammonothermal"],
                    "typical_temp": [600, 1000],
                    "atmosphere": "ammonia"
                },
                "halide": {
                    "methods": ["solid_state", "melt", "aqueous"],
                    "typical_temp": [200, 600]
                }
            }
        }
        
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                loaded_kb = json.load(f)
                default_kb.update(loaded_kb)
        
        return default_kb
    
    def _init_compound_rules(self) -> List[Callable]:
        """初始化化合物规则"""
        return [
            self._check_elemental_stability,
            self._check_oxidation_states,
            self._check_hydrolysis_risk,
            self._check_volatility,
            self._check_toxicity
        ]
    
    def predict(self, target_formula: str, 
                available_precursors: List[ChemicalCompound]) -> Tuple[bool, float]:
        """基于规则的合成可行性预测"""
        
        scores = []
        
        # 应用规则
        for rule in self.compound_rules:
            score = rule(target_formula)
            scores.append(score)
        
        # 检查前驱体可用性
        precursor_score = self._check_precursor_availability(
            target_formula, available_precursors
        )
        scores.append(precursor_score)
        
        # 检查已知合成路线
        route_score = self._check_known_routes(target_formula)
        scores.append(route_score)
        
        # 综合评分
        final_score = np.mean(scores)
        is_synthesizable = final_score > 0.3
        
        return is_synthesizable, final_score
    
    def get_difficulty(self, target_formula: str) -> SynthesisDifficulty:
        """评估合成难度"""
        is_possible, probability = self.predict(target_formula, [])
        
        if not is_possible:
            return SynthesisDifficulty.IMPOSSIBLE
        
        if probability > 0.9:
            return SynthesisDifficulty.EASY
        elif probability > 0.7:
            return SynthesisDifficulty.MODERATE
        elif probability > 0.5:
            return SynthesisDifficulty.DIFFICULT
        else:
            return SynthesisDifficulty.VERY_DIFFICULT
    
    def _check_elemental_stability(self, formula: str) -> float:
        """检查元素稳定性"""
        # 检查是否包含极不稳定元素
        unstable_elements = ["At", "Rn", "Fr", "Ra", "Po"]
        for elem in unstable_elements:
            if elem in formula:
                return 0.0
        
        # 检查放射性元素
        radioactive = ["U", "Th", "Pu", "Am", "Np"]
        for elem in radioactive:
            if elem in formula:
                return 0.3
        
        # 检查贵金属
        noble_metals = ["Au", "Pt", "Ir", "Os", "Rh", "Ru"]
        for elem in noble_metals:
            if elem in formula:
                return 0.6
        
        return 1.0
    
    def _check_oxidation_states(self, formula: str) -> float:
        """检查氧化态兼容性"""
        # 简化检查：假设大多数常见氧化态组合是可行的
        # 实际实现需要解析化学式并检查氧化态组合
        return 0.8
    
    def _check_hydrolysis_risk(self, formula: str) -> float:
        """检查水解风险"""
        moisture_sensitive = self.knowledge_base.get("moisture_sensitive", [])
        
        for compound in moisture_sensitive:
            if compound in formula:
                return 0.5  # 需要特殊处理
        
        return 1.0
    
    def _check_volatility(self, formula: str) -> float:
        """检查挥发性"""
        volatile_elements = ["Hg", "I", "Br", "S"]
        
        for elem in volatile_elements:
            if elem in formula:
                return 0.7
        
        return 1.0
    
    def _check_toxicity(self, formula: str) -> float:
        """检查毒性（影响合成可行性）"""
        highly_toxic = ["As", "Cd", "Hg", "Pb", "Tl", "Be"]
        
        for elem in highly_toxic:
            if elem in formula:
                return 0.6  # 需要特殊防护，但仍可合成
        
        return 1.0
    
    def _check_precursor_availability(self, target_formula: str,
                                     available_precursors: List[ChemicalCompound]) -> float:
        """检查前驱体可用性"""
        if not available_precursors:
            return 0.5  # 默认中等评分
        
        # 解析目标化学式
        elements = self._parse_formula(target_formula)
        
        available_elements = set()
        for precursor in available_precursors:
            precursor_elements = self._parse_formula(precursor.formula)
            available_elements.update(precursor_elements.keys())
        
        # 检查是否所有必需元素都有前驱体
        covered = sum(1 for elem in elements if elem in available_elements)
        return covered / len(elements) if elements else 0.5
    
    def _check_known_routes(self, target_formula: str) -> float:
        """检查是否有已知合成路线"""
        # 在知识库中查找
        for compound_type, info in self.knowledge_base.get("synthesis_templates", {}).items():
            # 简化匹配
            if compound_type in target_formula.lower():
                return 0.9
        
        return 0.5
    
    def _parse_formula(self, formula: str) -> Dict[str, float]:
        """解析化学式"""
        # 简化实现，实际应使用更复杂的解析
        import re
        pattern = r'([A-Z][a-z]*)(\d*\.?\d*)'
        matches = re.findall(pattern, formula)
        
        result = {}
        for element, count in matches:
            result[element] = float(count) if count else 1.0
        
        return result


class MLPredictor(SynthesisPredictor):
    """基于机器学习的合成可行性预测器"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.feature_extractor = None
        self.scaler = None
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            self._init_new_model()
    
    def _init_new_model(self):
        """初始化新模型"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
            self.feature_extractor = self._default_feature_extractor
        except ImportError:
            logger.warning("scikit-learn not available, using rule-based fallback")
            self.model = None
    
    def _load_model(self, path: str):
        """加载预训练模型"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data.get('model')
            self.scaler = data.get('scaler')
            self.feature_extractor = data.get('feature_extractor', self._default_feature_extractor)
    
    def save_model(self, path: str):
        """保存模型"""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_extractor': self.feature_extractor
            }, f)
    
    def _default_feature_extractor(self, formula: str) -> np.ndarray:
        """默认特征提取器"""
        features = []
        
        # 基础统计特征
        elements = self._parse_formula(formula)
        features.append(len(elements))  # 元素种类数
        features.append(sum(elements.values()))  # 总原子数
        features.append(max(elements.values()) if elements else 0)  # 最大化学计量
        
        # 电负性统计（简化）
        electronegativity = {
            'H': 2.20, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55,
            'N': 3.04, 'O': 3.44, 'F': 3.98, 'Na': 0.93, 'Mg': 1.31,
            'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16,
            'K': 0.82, 'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63,
            'Cr': 1.66, 'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91,
            'Cu': 1.90, 'Zn': 1.65, 'Ga': 1.81, 'Ge': 2.01, 'As': 2.18,
            'Se': 2.55, 'Br': 2.96
        }
        
        en_values = [electronegativity.get(elem, 1.5) for elem in elements]
        if en_values:
            features.extend([
                np.mean(en_values),
                np.std(en_values),
                max(en_values) - min(en_values)  # 电负性差
            ])
        else:
            features.extend([0, 0, 0])
        
        return np.array(features)
    
    def predict(self, target_formula: str, 
                available_precursors: List[ChemicalCompound]) -> Tuple[bool, float]:
        """使用ML模型预测"""
        if self.model is None:
            # 回退到基于规则的预测
            fallback = KnowledgeBasedPredictor()
            return fallback.predict(target_formula, available_precursors)
        
        # 提取特征
        features = self.feature_extractor(target_formula)
        features = features.reshape(1, -1)
        
        # 标准化
        if self.scaler:
            features = self.scaler.transform(features)
        
        # 预测
        probability = self.model.predict_proba(features)[0][1]
        is_synthesizable = probability > 0.5
        
        return is_synthesizable, probability
    
    def get_difficulty(self, target_formula: str) -> SynthesisDifficulty:
        """评估合成难度"""
        is_possible, probability = self.predict(target_formula, [])
        
        if not is_possible:
            return SynthesisDifficulty.IMPOSSIBLE
        
        if probability > 0.9:
            return SynthesisDifficulty.EASY
        elif probability > 0.7:
            return SynthesisDifficulty.MODERATE
        elif probability > 0.5:
            return SynthesisDifficulty.DIFFICULT
        else:
            return SynthesisDifficulty.VERY_DIFFICULT
    
    def train(self, formulas: List[str], labels: List[int]):
        """训练模型"""
        if self.model is None:
            self._init_new_model()
        
        # 提取特征
        X = np.array([self.feature_extractor(f) for f in formulas])
        y = np.array(labels)
        
        # 标准化
        X = self.scaler.fit_transform(X)
        
        # 训练
        self.model.fit(X, y)
        logger.info(f"Model trained on {len(formulas)} samples")
    
    def _parse_formula(self, formula: str) -> Dict[str, float]:
        """解析化学式"""
        import re
        pattern = r'([A-Z][a-z]*)(\d*\.?\d*)'
        matches = re.findall(pattern, formula)
        
        result = {}
        for element, count in matches:
            result[element] = float(count) if count else 1.0
        
        return result


class SynthesisPlanner:
    """合成路径规划器"""
    
    def __init__(self, predictor: Optional[SynthesisPredictor] = None):
        self.predictor = predictor or KnowledgeBasedPredictor()
        self.precursor_database = self._init_precursor_database()
        self.reaction_graph = self._build_reaction_graph()
        
    def _init_precursor_database(self) -> Dict[str, List[ChemicalCompound]]:
        """初始化前驱体数据库"""
        return {
            "Li": [
                ChemicalCompound("Li2CO3", "Lithium Carbonate", 73.89, 
                               melting_point=996, cost_per_gram=0.5),
                ChemicalCompound("LiOH", "Lithium Hydroxide", 23.95,
                               melting_point=723, cost_per_gram=1.0),
                ChemicalCompound("Li2O", "Lithium Oxide", 29.88,
                               melting_point=1957, cost_per_gram=2.0),
                ChemicalCompound("LiNO3", "Lithium Nitrate", 68.95,
                               melting_point=525, cost_per_gram=0.8)
            ],
            "Na": [
                ChemicalCompound("Na2CO3", "Sodium Carbonate", 105.99,
                               cost_per_gram=0.1),
                ChemicalCompound("NaOH", "Sodium Hydroxide", 40.00,
                               melting_point=591, cost_per_gram=0.2)
            ],
            "P": [
                ChemicalCompound("NH4H2PO4", "Ammonium Dihydrogen Phosphate", 115.03,
                               cost_per_gram=0.3),
                ChemicalCompound("P2O5", "Phosphorus Pentoxide", 141.94,
                               cost_per_gram=1.5),
                ChemicalCompound("H3PO4", "Phosphoric Acid", 98.00,
                               cost_per_gram=0.4)
            ],
            "S": [
                ChemicalCompound("H2SO4", "Sulfuric Acid", 98.08,
                               cost_per_gram=0.2, hazards=["corrosive"]),
                ChemicalCompound("Na2S", "Sodium Sulfide", 78.04,
                               cost_per_gram=0.5),
                ChemicalCompound("S", "Sulfur", 32.06,
                               melting_point=388, cost_per_gram=0.1)
            ],
            "Ti": [
                ChemicalCompound("TiO2", "Titanium Dioxide", 79.87,
                               melting_point=2116, cost_per_gram=0.8),
                ChemicalCompound("TiCl4", "Titanium Tetrachloride", 189.68,
                               boiling_point=409, cost_per_gram=2.0,
                               hazards=["moisture_sensitive", "corrosive"])
            ],
            "Si": [
                ChemicalCompound("SiO2", "Silicon Dioxide", 60.08,
                               melting_point=1996, cost_per_gram=0.3),
                ChemicalCompound("SiCl4", "Silicon Tetrachloride", 169.90,
                               cost_per_gram=1.5, hazards=["moisture_sensitive"])
            ]
        }
    
    def _build_reaction_graph(self) -> nx.DiGraph:
        """构建反应图"""
        G = nx.DiGraph()
        
        # 添加反应节点和边
        reactions = [
            # 固相反应
            ("Li2CO3", "Li2O", {"type": "decomposition", "temp": 1200}),
            ("CaCO3", "CaO", {"type": "decomposition", "temp": 900}),
            
            # 氧化反应
            ("TiO2", "BaTiO3", {"type": "solid_state", "temp": 1200}),
            ("SiO2", "SiC", {"type": "carbothermal", "temp": 1800}),
            
            # 硫化反应
            ("Li2CO3", "Li2S", {"type": "sulfidation", "temp": 800}),
            
            # 氮化反应
            ("Li3N", "Li2NH", {"type": "ammonolysis", "temp": 400}),
        ]
        
        for source, target, attrs in reactions:
            G.add_edge(source, target, **attrs)
        
        return G
    
    def plan_synthesis(self, target_formula: str, 
                      constraints: Optional[Dict[str, Any]] = None) -> List[SynthesisRoute]:
        """
        规划合成路线
        
        Args:
            target_formula: 目标化学式
            constraints: 约束条件
                - max_steps: 最大步骤数
                - max_cost: 最大成本
                - available_precursors: 可用前驱体列表
                - preferred_method: 首选方法
        
        Returns:
            合成路线列表（按可行性排序）
        """
        constraints = constraints or {}
        
        # 预测可行性
        available = constraints.get("available_precursors", [])
        is_possible, probability = self.predictor.predict(target_formula, available)
        
        if not is_possible:
            logger.warning(f"Target {target_formula} predicted to be difficult to synthesize")
        
        # 生成可能的路线
        routes = self._generate_routes(target_formula, constraints)
        
        # 评分和排序
        scored_routes = [(route, self._score_route(route, constraints)) for route in routes]
        scored_routes.sort(key=lambda x: x[1], reverse=True)
        
        return [route for route, _ in scored_routes]
    
    def _generate_routes(self, target_formula: str, 
                        constraints: Dict[str, Any]) -> List[SynthesisRoute]:
        """生成合成路线"""
        routes = []
        
        # 解析目标化学式
        target_elements = self._parse_formula(target_formula)
        
        # 策略1: 直接固相合成
        route1 = self._direct_solid_state_route(target_formula, target_elements)
        if route1:
            routes.append(route1)
        
        # 策略2: 共沉淀法
        route2 = self._co_precipitation_route(target_formula, target_elements)
        if route2:
            routes.append(route2)
        
        # 策略3: 溶胶-凝胶法
        route3 = self._sol_gel_route(target_formula, target_elements)
        if route3:
            routes.append(route3)
        
        # 策略4: 水热法
        route4 = self._hydrothermal_route(target_formula, target_elements)
        if route4:
            routes.append(route4)
        
        # 策略5: 基于反应图的路线
        graph_routes = self._graph_based_routes(target_formula)
        routes.extend(graph_routes)
        
        return routes
    
    def _direct_solid_state_route(self, target_formula: str,
                                  elements: Dict[str, float]) -> Optional[SynthesisRoute]:
        """生成固相合路线"""
        precursors = []
        
        for element, stoich in elements.items():
            available = self.precursor_database.get(element, [])
            if available:
                # 选择最合适的氧化物/碳酸盐前驱体
                best_precursor = min(available, 
                                   key=lambda p: p.cost_per_gram + (0 if p.melting_point else 1000))
                precursors.append(Precursor(best_precursor, stoich))
        
        if len(precursors) < len(elements):
            return None
        
        target = ChemicalCompound(target_formula, name=f"Target_{target_formula}")
        
        step = SynthesisStep(
            step_number=1,
            description=f"Solid state reaction of {' + '.join(p.compound.formula for p in precursors)}",
            reaction_type=ReactionType.SOLID_STATE,
            temperature=1273,  # 1000°C
            duration=14400,  # 4 hours
            atmosphere="air",
            precursors=precursors,
            products=[target],
            yield_estimate=0.85
        )
        
        route = SynthesisRoute(target=target, steps=[step])
        return route
    
    def _co_precipitation_route(self, target_formula: str,
                               elements: Dict[str, float]) -> Optional[SynthesisRoute]:
        """生成共沉淀路线"""
        # 筛选可形成可溶性盐的金属
        metal_precursors = []
        for element, stoich in elements.items():
            if element in ["Li", "Na", "K", "Mg", "Ca", "Ni", "Co", "Mn", "Fe", "Cu", "Zn"]:
                nitrate = ChemicalCompound(f"{element}(NO3)n", f"{element} Nitrate", 
                                         cost_per_gram=0.5)
                metal_precursors.append(Precursor(nitrate, stoich))
        
        if len(metal_precursors) < 2:
            return None
        
        target = ChemicalCompound(target_formula, name=f"Target_{target_formula}")
        precipitant = ChemicalCompound("NH4OH", "Ammonium Hydroxide", 35.05,
                                      cost_per_gram=0.2, hazards=["corrosive"])
        
        steps = [
            SynthesisStep(
                step_number=1,
                description="Dissolve metal nitrates in deionized water",
                reaction_type=ReactionType.CO_PRECIPITATION,
                temperature=298,
                duration=1800,
                precursors=metal_precursors
            ),
            SynthesisStep(
                step_number=2,
                description="Co-precipitation by adding NH4OH",
                reaction_type=ReactionType.CO_PRECIPITATION,
                temperature=298,
                duration=3600,
                precursors=[Precursor(precipitant, 2.0)],
                yield_estimate=0.90
            ),
            SynthesisStep(
                step_number=3,
                description="Calcination of precipitate",
                reaction_type=ReactionType.SOLID_STATE,
                temperature=873,
                duration=7200,
                products=[target],
                yield_estimate=0.95
            )
        ]
        
        route = SynthesisRoute(target=target, steps=steps)
        return route
    
    def _sol_gel_route(self, target_formula: str,
                      elements: Dict[str, float]) -> Optional[SynthesisRoute]:
        """生成溶胶-凝胶路线"""
        # 筛选可形成醇盐的元素
        alkoxide_precursors = []
        for element, stoich in elements.items():
            if element in ["Si", "Ti", "Al", "Zr"]:
                alkoxide = ChemicalCompound(f"{element}(OR)4", f"{element} Alkoxide",
                                          cost_per_gram=2.0,
                                          hazards=["moisture_sensitive"])
                alkoxide_precursors.append(Precursor(alkoxide, stoich))
        
        if not alkoxide_precursors:
            return None
        
        target = ChemicalCompound(target_formula, name=f"Target_{target_formula}")
        ethanol = ChemicalCompound("C2H5OH", "Ethanol", 46.07, cost_per_gram=0.1,
                                 hazards=["flammable"])
        
        steps = [
            SynthesisStep(
                step_number=1,
                description="Prepare alkoxide solution in ethanol",
                reaction_type=ReactionType.SOL_GEL,
                temperature=298,
                duration=1800,
                precursors=alkoxide_precursors + [Precursor(ethanol, 100.0)]
            ),
            SynthesisStep(
                step_number=2,
                description="Hydrolysis and gelation",
                reaction_type=ReactionType.SOL_GEL,
                temperature=333,
                duration=86400,
                yield_estimate=0.80
            ),
            SynthesisStep(
                step_number=3,
                description="Drying and calcination",
                reaction_type=ReactionType.SOL_GEL,
                temperature=773,
                duration=7200,
                products=[target],
                yield_estimate=0.95
            )
        ]
        
        route = SynthesisRoute(target=target, steps=steps)
        return route
    
    def _hydrothermal_route(self, target_formula: str,
                           elements: Dict[str, float]) -> Optional[SynthesisRoute]:
        """生成水热路线"""
        precursors = []
        
        for element, stoich in elements.items():
            available = self.precursor_database.get(element, [])
            if available:
                precursors.append(Precursor(available[0], stoich))
        
        if len(precursors) < len(elements):
            return None
        
        target = ChemicalCompound(target_formula, name=f"Target_{target_formula}")
        water = ChemicalCompound("H2O", "Deionized Water", 18.02, cost_per_gram=0.001)
        
        steps = [
            SynthesisStep(
                step_number=1,
                description="Prepare aqueous precursor solution",
                reaction_type=ReactionType.HYDROTHERMAL,
                temperature=298,
                duration=1800,
                precursors=precursors + [Precursor(water, 50.0)]
            ),
            SynthesisStep(
                step_number=2,
                description="Hydrothermal treatment in autoclave",
                reaction_type=ReactionType.HYDROTHERMAL,
                temperature=473,
                pressure=2000000,  # 2 MPa
                duration=86400,
                products=[target],
                yield_estimate=0.85
            )
        ]
        
        route = SynthesisRoute(target=target, steps=steps)
        return route
    
    def _graph_based_routes(self, target_formula: str) -> List[SynthesisRoute]:
        """基于反应图生成路线"""
        routes = []
        
        # 在反应图中查找路径
        if target_formula in self.reaction_graph:
            # 查找所有可能的前驱体
            predecessors = list(self.reaction_graph.predecessors(target_formula))
            
            for precursor in predecessors:
                edge_data = self.reaction_graph.get_edge_data(precursor, target_formula)
                
                target = ChemicalCompound(target_formula)
                precursor_compound = ChemicalCompound(precursor)
                
                step = SynthesisStep(
                    step_number=1,
                    description=f"{edge_data['type']} reaction",
                    reaction_type=ReactionType(edge_data['type']) 
                                 if edge_data['type'] in [e.value for e in ReactionType]
                                 else ReactionType.SOLID_STATE,
                    temperature=edge_data.get('temp', 1000),
                    precursors=[Precursor(precursor_compound, 1.0)],
                    products=[target]
                )
                
                route = SynthesisRoute(target=target, steps=[step])
                routes.append(route)
        
        return routes
    
    def _score_route(self, route: SynthesisRoute, 
                    constraints: Dict[str, Any]) -> float:
        """评分合成路线"""
        scores = []
        
        # 产率评分
        scores.append(route.total_yield * 0.3)
        
        # 成本评分（归一化）
        max_cost = constraints.get("max_cost", 1000)
        cost_score = max(0, 1 - route.estimated_cost / max_cost) * 0.2
        scores.append(cost_score)
        
        # 时间评分
        max_time = constraints.get("max_time", 7 * 24 * 3600)  # 1 week
        time_score = max(0, 1 - route.estimated_time / max_time) * 0.15
        scores.append(time_score)
        
        # 难度评分
        difficulty_score = max(0, 1 - route.difficulty_score / 5) * 0.2
        scores.append(difficulty_score)
        
        # 成功概率评分
        scores.append(route.success_probability * 0.15)
        
        return sum(scores)
    
    def select_precursors(self, target_formula: str,
                         criteria: Optional[Dict[str, Any]] = None) -> List[Precursor]:
        """
        选择最优前驱体
        
        Args:
            target_formula: 目标化学式
            criteria: 选择标准
                - minimize_cost: 最小化成本
                - maximize_purity: 最大化纯度
                - prefer_stable: 偏好稳定化合物
                - avoid_hazardous: 避免有害化合物
        
        Returns:
            最优前驱体列表
        """
        criteria = criteria or {}
        elements = self._parse_formula(target_formula)
        
        selected = []
        
        for element, stoich in elements.items():
            available = self.precursor_database.get(element, [])
            
            if not available:
                logger.warning(f"No precursors found for element: {element}")
                continue
            
            # 应用选择标准
            scored_precursors = []
            for compound in available:
                score = 0.0
                
                # 成本因素
                if criteria.get("minimize_cost", True):
                    score += (1 - compound.cost_per_gram / 10) * 0.3
                
                # 稳定性因素
                if criteria.get("prefer_stable", True):
                    score += compound.stability * 0.25
                
                # 纯度因素
                if criteria.get("maximize_purity", True):
                    score += 0.2
                
                # 安全性因素
                if criteria.get("avoid_hazardous", False):
                    if compound.hazards:
                        score -= len(compound.hazards) * 0.1
                else:
                    # 轻微惩罚有害化合物
                    score -= len(compound.hazards) * 0.05
                
                # 可获得性
                score += compound.availability * 0.15
                
                scored_precursors.append((compound, score))
            
            # 选择最高分的前驱体
            scored_precursors.sort(key=lambda x: x[1], reverse=True)
            best_compound = scored_precursors[0][0]
            
            selected.append(Precursor(best_compound, stoich))
        
        return selected
    
    def _parse_formula(self, formula: str) -> Dict[str, float]:
        """解析化学式"""
        import re
        pattern = r'([A-Z][a-z]*)(\d*\.?\d*)'
        matches = re.findall(pattern, formula)
        
        result = {}
        for element, count in matches:
            result[element] = float(count) if count else 1.0
        
        return result
    
    def optimize_reaction_conditions(self, route: SynthesisRoute,
                                    objective: str = "maximize_yield") -> SynthesisRoute:
        """优化反应条件"""
        optimized_route = SynthesisRoute(
            target=route.target,
            steps=route.steps.copy()
        )
        
        for step in optimized_route.steps:
            if objective == "maximize_yield":
                # 提高温度以提升反应速率
                step.temperature = min(step.temperature * 1.1, 1500)
                # 延长时间
                step.duration = min(step.duration * 1.2, 48 * 3600)
                step.yield_estimate = min(step.yield_estimate * 1.05, 0.99)
                
            elif objective == "minimize_time":
                # 提高温度缩短时间
                step.temperature = min(step.temperature * 1.2, 1500)
                step.duration = max(step.duration * 0.7, 600)
                step.yield_estimate = max(step.yield_estimate * 0.95, 0.5)
                
            elif objective == "minimize_cost":
                # 降低温度减少能耗
                step.temperature = max(step.temperature * 0.9, 300)
                step.duration = max(step.duration * 0.9, 600)
        
        optimized_route._calculate_metrics()
        return optimized_route
    
    def save_routes(self, routes: List[SynthesisRoute], filepath: str):
        """保存合成路线"""
        data = [route.to_dict() for route in routes]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_routes(self, filepath: str) -> List[SynthesisRoute]:
        """加载合成路线"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        routes = []
        for item in data:
            target = ChemicalCompound(**item["target"])
            steps = [SynthesisStep(**s) for s in item["steps"]]
            route = SynthesisRoute(target=target, steps=steps)
            routes.append(route)
        
        return routes


class PrecursorOptimizer:
    """前驱体优化器"""
    
    def __init__(self, planner: SynthesisPlanner):
        self.planner = planner
        
    def optimize_combination(self, target_formula: str,
                           available_precursors: List[ChemicalCompound],
                           n_combinations: int = 5) -> List[Dict[str, Any]]:
        """优化前驱体组合"""
        # 解析目标化学式
        elements = self.planner._parse_formula(target_formula)
        
        # 按元素分组前驱体
        precursor_by_element = defaultdict(list)
        for precursor in available_precursors:
            for element in elements:
                if element in precursor.formula:
                    precursor_by_element[element].append(precursor)
        
        # 生成组合
        combinations = self._generate_combinations(
            elements, precursor_by_element, n_combinations
        )
        
        # 评分组合
        scored_combinations = []
        for combo in combinations:
            score = self._score_combination(combo)
            scored_combinations.append({
                "precursors": combo,
                "score": score,
                "total_cost": sum(p.cost_per_gram for p in combo.values()),
                "avg_purity": np.mean([p.purity if hasattr(p, 'purity') else 0.99 
                                      for p in combo.values()]),
                "risk_level": self._assess_risk(combo)
            })
        
        # 按评分排序
        scored_combinations.sort(key=lambda x: x["score"], reverse=True)
        
        return scored_combinations
    
    def _generate_combinations(self, elements: Dict[str, float],
                              precursor_by_element: Dict[str, List[ChemicalCompound]],
                              n: int) -> List[Dict[str, ChemicalCompound]]:
        """生成前驱体组合"""
        combinations = []
        
        # 简化：为每个元素选择一个前驱体
        element_list = list(elements.keys())
        
        for _ in range(n):
            combo = {}
            for element in element_list:
                available = precursor_by_element.get(element, [])
                if available:
                    combo[element] = np.random.choice(available)
            
            if len(combo) == len(element_list):
                combinations.append(combo)
        
        return combinations
    
    def _score_combination(self, combo: Dict[str, ChemicalCompound]) -> float:
        """评分前驱体组合"""
        scores = []
        
        total_cost = sum(p.cost_per_gram for p in combo.values())
        scores.append(1 / (1 + total_cost))  # 成本越低分越高
        
        # 稳定性评分
        stability = np.mean([p.stability for p in combo.values()])
        scores.append(stability)
        
        # 可用性评分
        availability = np.mean([p.availability for p in combo.values()])
        scores.append(availability)
        
        # 安全性评分
        hazard_count = sum(len(p.hazards) for p in combo.values())
        scores.append(max(0, 1 - hazard_count * 0.1))
        
        return np.mean(scores)
    
    def _assess_risk(self, combo: Dict[str, ChemicalCompound]) -> str:
        """评估风险等级"""
        hazard_count = sum(len(p.hazards) for p in combo.values())
        
        if hazard_count == 0:
            return "low"
        elif hazard_count <= 2:
            return "moderate"
        elif hazard_count <= 4:
            return "high"
        else:
            return "critical"


# ==================== 主入口函数 ====================

def create_planner(predictor_type: str = "knowledge") -> SynthesisPlanner:
    """创建合成规划器"""
    if predictor_type == "ml":
        predictor = MLPredictor()
    else:
        predictor = KnowledgeBasedPredictor()
    
    return SynthesisPlanner(predictor)


def predict_synthesis_feasibility(formula: str, 
                                 available_precursors: Optional[List[str]] = None) -> Dict[str, Any]:
    """预测合成可行性"""
    predictor = KnowledgeBasedPredictor()
    
    precursors = []
    if available_precursors:
        for p_formula in available_precursors:
            precursors.append(ChemicalCompound(p_formula))
    
    is_possible, probability = predictor.predict(formula, precursors)
    difficulty = predictor.get_difficulty(formula)
    
    return {
        "formula": formula,
        "is_synthesizable": is_possible,
        "success_probability": probability,
        "difficulty": difficulty.value,
        "recommendations": []
    }


def plan_synthesis_route(target_formula: str, 
                        constraints: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """规划合成路线"""
    planner = create_planner()
    routes = planner.plan_synthesis(target_formula, constraints)
    return [route.to_dict() for route in routes]


# 示例用法
if __name__ == "__main__":
    # 预测合成可行性
    result = predict_synthesis_feasibility("Li3PS4")
    print(json.dumps(result, indent=2))
    
    # 规划合成路线
    routes = plan_synthesis_route("Li3PS4", {"max_cost": 100})
    print(f"\nGenerated {len(routes)} routes")
    for i, route in enumerate(routes[:3], 1):
        print(f"\nRoute {i}:")
        print(f"  Success probability: {route['success_probability']:.2%}")
        print(f"  Estimated cost: ${route['estimated_cost']:.2f}")
        print(f"  Steps: {len(route['steps'])}")
