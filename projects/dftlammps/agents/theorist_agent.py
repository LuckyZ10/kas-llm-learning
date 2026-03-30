"""
Theorist Agent - 理论家Agent
负责假设生成、模型推导、理论验证
"""
from __future__ import annotations
import asyncio
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import uuid

from ..multi_agent.agent_core import (
    BaseAgent, DeliberativeAgent, Message, MessageType,
    Observation, Action, AgentCapability, AgentStatus
)


@dataclass
class Hypothesis:
    """假设"""
    id: str
    statement: str
    assumptions: List[str] = field(default_factory=list)
    predictions: List[str] = field(default_factory=list)
    confidence: float = 0.5
    status: str = "proposed"  # proposed, testing, validated, rejected
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    parent_hypothesis: Optional[str] = None
    derived_hypotheses: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "assumptions": self.assumptions,
            "predictions": self.predictions,
            "confidence": self.confidence,
            "status": self.status,
            "evidence": self.evidence,
            "parent_hypothesis": self.parent_hypothesis,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class Model:
    """理论模型"""
    id: str
    name: str
    description: str
    equations: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    domain: str = ""
    validation_status: str = "unvalidated"
    validation_results: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "equations": self.equations,
            "parameters": self.parameters,
            "assumptions": self.assumptions,
            "domain": self.domain,
            "validation_status": self.validation_status
        }


@dataclass
class Theory:
    """理论"""
    id: str
    name: str
    description: str
    core_hypothesis: str
    supporting_hypotheses: List[str] = field(default_factory=list)
    models: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    status: str = "developing"  # developing, validated, rejected
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "core_hypothesis": self.core_hypothesis,
            "supporting_hypotheses": self.supporting_hypotheses,
            "models": self.models,
            "confidence": self.confidence,
            "status": self.status
        }


class TheoristAgent(DeliberativeAgent):
    """
    理论家Agent
    负责科学研究中的理论工作
    """
    
    def __init__(self, **kwargs):
        kwargs.setdefault("name", "TheoristAgent")
        kwargs.setdefault("description", "Generates hypotheses, develops models, validates theories")
        super().__init__(**kwargs)
        
        # 知识库
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.models: Dict[str, Model] = {}
        self.theories: Dict[str, Theory] = {}
        
        # 推理规则
        self.inference_rules: List[Callable] = []
        
        # 当前研究焦点
        self.current_research_topic: Optional[str] = None
        
        # 注册能力
        self._register_capabilities()
        
        # 注册消息处理器
        self.register_message_handler(
            MessageType.COMMUNICATION,
            self._handle_collaboration_message
        )
    
    def _register_capabilities(self) -> None:
        """注册专业能力"""
        self.register_capability(AgentCapability(
            name="generate_hypothesis",
            description="Generate new hypothesis from observations",
            handler=self._generate_hypothesis_handler
        ))
        
        self.register_capability(AgentCapability(
            name="derive_model",
            description="Derive mathematical model from hypothesis",
            handler=self._derive_model_handler
        ))
        
        self.register_capability(AgentCapability(
            name="validate_theory",
            description="Validate theory against evidence",
            handler=self._validate_theory_handler
        ))
        
        self.register_capability(AgentCapability(
            name="analyze_relationships",
            description="Analyze relationships between hypotheses",
            handler=self._analyze_relationships_handler
        ))
        
        self.register_capability(AgentCapability(
            name="propose_experiment",
            description="Propose experiments to test hypothesis",
            handler=self._propose_experiment_handler
        ))
    
    async def perceive(self) -> List[Observation]:
        """感知 - 获取相关信息"""
        observations = []
        
        # 检查是否有新的研究请求
        while not self.inbox.empty():
            try:
                message = self.inbox.get_nowait()
                if message.message_type == MessageType.COMMUNICATION:
                    observations.append(Observation(
                        source="message",
                        data=message.content,
                        confidence=0.9
                    ))
            except asyncio.QueueEmpty:
                break
        
        return observations
    
    async def reason(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """推理 - 理论推导"""
        decision = {
            "actions": [],
            "reasoning_log": []
        }
        
        # 检查工作记忆中的新假设请求
        trigger = context.get("trigger")
        
        if trigger:
            if trigger.get("type") == "generate_hypothesis":
                hypothesis = await self.generate_hypothesis(
                    trigger.get("problem_statement", ""),
                    trigger.get("observations", [])
                )
                decision["actions"].append({
                    "type": "generate_hypothesis",
                    "hypothesis": hypothesis.to_dict() if hypothesis else None
                })
            
            elif trigger.get("type") == "validate_model":
                model_id = trigger.get("model_id")
                if model_id in self.models:
                    validation_result = await self.validate_model(
                        self.models[model_id],
                        trigger.get("test_data", [])
                    )
                    decision["actions"].append({
                        "type": "validate_model",
                        "result": validation_result
                    })
        
        # 检查是否有待验证的假设
        pending_hypotheses = [
            h for h in self.hypotheses.values()
            if h.status == "proposed"
        ]
        
        if pending_hypotheses:
            # 选择最有希望的假设进行推导
            hypothesis = max(pending_hypotheses, key=lambda h: h.confidence)
            
            # 推导模型
            if hypothesis.id not in [m.get("hypothesis_id") for m in self.models.values()]:
                model = await self.derive_model_from_hypothesis(hypothesis)
                if model:
                    decision["actions"].append({
                        "type": "derive_model",
                        "model": model.to_dict()
                    })
        
        return decision
    
    async def act(self, decision: Dict[str, Any]) -> List[Action]:
        """行动 - 执行决策"""
        actions = []
        
        for action_data in decision.get("actions", []):
            action_type = action_data.get("type")
            
            if action_type == "generate_hypothesis":
                hypothesis_data = action_data.get("hypothesis")
                if hypothesis_data:
                    actions.append(Action(
                        action_type="broadcast_hypothesis",
                        params={"hypothesis": hypothesis_data},
                        priority=2
                    ))
            
            elif action_type == "derive_model":
                model_data = action_data.get("model")
                if model_data:
                    actions.append(Action(
                        action_type="broadcast_model",
                        params={"model": model_data},
                        priority=1
                    ))
            
            elif action_type == "validate_model":
                actions.append(Action(
                    action_type="report_validation",
                    params={"result": action_data.get("result")},
                    priority=1
                ))
        
        return actions
    
    # ===== 核心能力实现 =====
    
    async def generate_hypothesis(
        self,
        problem_statement: str,
        observations: List[Dict[str, Any]]
    ) -> Optional[Hypothesis]:
        """
        生成假设
        基于问题陈述和观察数据生成科学假设
        """
        hypothesis_id = f"hyp_{uuid.uuid4().hex[:8]}"
        
        # 分析观察数据中的模式
        patterns = self._extract_patterns(observations)
        
        # 基于现有知识和模式生成假设
        # 这里使用简化的逻辑，实际可以使用LLM或复杂推理
        
        if patterns:
            statement = f"Based on observed patterns: {patterns[0]}, we hypothesize that..."
            
            hypothesis = Hypothesis(
                id=hypothesis_id,
                statement=statement,
                assumptions=["Pattern is not due to random noise"],
                predictions=[
                    f"If {patterns[0]} then we expect...",
                    f"The relationship should hold under varying conditions..."
                ],
                confidence=0.6  # 初始置信度
            )
            
            self.hypotheses[hypothesis_id] = hypothesis
            
            # 存储到长期记忆
            self.long_term_memory.add_experience({
                "type": "hypothesis_generation",
                "hypothesis_id": hypothesis_id,
                "problem": problem_statement
            })
            
            return hypothesis
        
        return None
    
    async def derive_model_from_hypothesis(self, hypothesis: Hypothesis) -> Optional[Model]:
        """
        从假设推导数学模型
        """
        model_id = f"model_{uuid.uuid4().hex[:8]}"
        
        # 基于假设构建数学模型
        # 简化的实现，实际应该根据领域知识构建具体方程
        
        model = Model(
            id=model_id,
            name=f"Model for {hypothesis.id}",
            description=f"Mathematical model derived from hypothesis: {hypothesis.statement[:50]}...",
            equations=[
                "Y = f(X, θ) + ε",  # 通用形式
                "∂Y/∂t = ..."       # 动力学方程
            ],
            parameters={
                "θ": {"description": "Model parameters", "prior": "uniform"},
                "ε": {"description": "Noise term", "distribution": "normal"}
            },
            assumptions=hypothesis.assumptions.copy(),
            domain="general"
        )
        
        self.models[model_id] = model
        
        # 更新假设状态
        hypothesis.metadata["model_id"] = model_id
        
        return model
    
    async def validate_theory(
        self,
        theory: Theory,
        evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        验证理论
        """
        validation_results = {
            "theory_id": theory.id,
            "tests_passed": 0,
            "tests_failed": 0,
            "confidence_update": 0.0
        }
        
        # 检查核心假设
        if theory.core_hypothesis in self.hypotheses:
            core = self.hypotheses[theory.core_hypothesis]
            
            # 验证预测
            for prediction in core.predictions:
                # 检查是否有证据支持或反驳
                supporting = [e for e in evidence if e.get("supports") == prediction]
                contradicting = [e for e in evidence if e.get("contradicts") == prediction]
                
                if supporting and not contradicting:
                    validation_results["tests_passed"] += 1
                elif contradicting:
                    validation_results["tests_failed"] += 1
        
        # 更新理论置信度
        total_tests = validation_results["tests_passed"] + validation_results["tests_failed"]
        if total_tests > 0:
            theory.confidence = validation_results["tests_passed"] / total_tests
            
            if theory.confidence > 0.8:
                theory.status = "validated"
            elif theory.confidence < 0.3:
                theory.status = "rejected"
        
        theory.evidence.extend(evidence)
        
        return validation_results
    
    async def validate_model(
        self,
        model: Model,
        test_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        验证模型
        """
        results = {
            "model_id": model.id,
            "predictions_made": [],
            "fit_quality": 0.0,
            "validation_passed": False
        }
        
        if not test_data:
            return results
        
        # 使用模型进行预测
        # 简化的验证逻辑
        predictions = []
        for data_point in test_data:
            # 应用模型方程
            prediction = self._apply_model(model, data_point)
            predictions.append(prediction)
        
        # 计算拟合质量（简化）
        if len(predictions) == len(test_data):
            # 比较预测值和实际值
            errors = []
            for pred, actual in zip(predictions, test_data):
                if "actual" in actual:
                    error = abs(pred - actual["actual"])
                    errors.append(error)
            
            if errors:
                avg_error = sum(errors) / len(errors)
                results["fit_quality"] = max(0, 1 - avg_error)
                results["validation_passed"] = results["fit_quality"] > 0.7
        
        model.validation_results.append(results)
        model.validation_status = "validated" if results["validation_passed"] else "rejected"
        
        return results
    
    async def analyze_hypothesis_relationships(self) -> Dict[str, Any]:
        """
        分析假设间的关系
        识别支持、冲突、包含等关系
        """
        relationships = {
            "supports": [],
            "contradicts": [],
            "implies": [],
            "similar": []
        }
        
        hypotheses = list(self.hypotheses.values())
        
        for i, h1 in enumerate(hypotheses):
            for h2 in hypotheses[i+1:]:
                # 检查相似性（简单实现：检查关键词重叠）
                h1_words = set(h1.statement.lower().split())
                h2_words = set(h2.statement.lower().split())
                
                similarity = len(h1_words & h2_words) / len(h1_words | h2_words)
                
                if similarity > 0.5:
                    relationships["similar"].append((h1.id, h2.id, similarity))
                
                # 检查预测是否矛盾
                for pred1 in h1.predictions:
                    for pred2 in h2.predictions:
                        if self._predictions_contradict(pred1, pred2):
                            relationships["contradicts"].append((h1.id, h2.id))
        
        return relationships
    
    async def propose_experiments_for_hypothesis(
        self,
        hypothesis: Hypothesis
    ) -> List[Dict[str, Any]]:
        """
        为假设提出实验设计
        """
        experiments = []
        
        for prediction in hypothesis.predictions:
            experiment = {
                "hypothesis_id": hypothesis.id,
                "purpose": f"Test prediction: {prediction}",
                "variables": {
                    "independent": ["variable_1", "variable_2"],
                    "dependent": ["outcome"],
                    "controlled": ["temperature", "pressure"]
                },
                "measurements": ["measurement_1", "measurement_2"],
                "expected_result": f"Should observe {prediction}",
                "success_criteria": "Statistical significance p < 0.05"
            }
            experiments.append(experiment)
        
        return experiments
    
    # ===== 辅助方法 =====
    
    def _extract_patterns(self, observations: List[Dict[str, Any]]) -> List[str]:
        """从观察中提取模式"""
        patterns = []
        
        # 简化的模式识别
        # 实际应用中可以使用聚类、关联规则等算法
        
        for obs in observations:
            if "correlation" in obs:
                patterns.append(f"Correlation: {obs['correlation']}")
            if "trend" in obs:
                patterns.append(f"Trend: {obs['trend']}")
        
        return patterns
    
    def _apply_model(self, model: Model, data: Dict[str, Any]) -> float:
        """应用模型进行预测"""
        # 简化的模型应用
        # 实际应该根据具体模型方程计算
        return data.get("input", 0.0) * 0.8 + 0.1
    
    def _predictions_contradict(self, pred1: str, pred2: str) -> bool:
        """检查两个预测是否矛盾"""
        # 简化的矛盾检测
        # 实际应该使用语义分析
        contradictions = [
            ("increase", "decrease"),
            ("positive", "negative"),
            ("high", "low")
        ]
        
        pred1_lower = pred1.lower()
        pred2_lower = pred2.lower()
        
        for word1, word2 in contradictions:
            if (word1 in pred1_lower and word2 in pred2_lower) or \
               (word2 in pred1_lower and word1 in pred2_lower):
                return True
        
        return False
    
    # ===== 消息处理器 =====
    
    async def _handle_collaboration_message(self, message: Message) -> None:
        """处理协作消息"""
        content = message.content
        
        if content.get("request_type") == "generate_hypothesis":
            hypothesis = await self.generate_hypothesis(
                content.get("problem_statement", ""),
                content.get("observations", [])
            )
            
            # 回复
            await self.send_message(
                {"hypothesis": hypothesis.to_dict() if hypothesis else None},
                receiver_id=message.sender_id,
                message_type=MessageType.RESULT
            )
        
        elif content.get("request_type") == "validate":
            # 验证请求
            model_id = content.get("model_id")
            if model_id in self.models:
                result = await self.validate_model(
                    self.models[model_id],
                    content.get("test_data", [])
                )
                
                await self.send_message(
                    {"validation_result": result},
                    receiver_id=message.sender_id,
                    message_type=MessageType.RESULT
                )
    
    # ===== 能力处理器 =====
    
    async def _generate_hypothesis_handler(self, **kwargs) -> Dict[str, Any]:
        """生成假设处理器"""
        hypothesis = await self.generate_hypothesis(
            kwargs.get("problem_statement", ""),
            kwargs.get("observations", [])
        )
        return {"hypothesis": hypothesis.to_dict() if hypothesis else None}
    
    async def _derive_model_handler(self, **kwargs) -> Dict[str, Any]:
        """推导模型处理器"""
        hypothesis_id = kwargs.get("hypothesis_id")
        if hypothesis_id in self.hypotheses:
            model = await self.derive_model_from_hypothesis(
                self.hypotheses[hypothesis_id]
            )
            return {"model": model.to_dict() if model else None}
        return {"error": "Hypothesis not found"}
    
    async def _validate_theory_handler(self, **kwargs) -> Dict[str, Any]:
        """验证理论处理器"""
        theory_id = kwargs.get("theory_id")
        if theory_id in self.theories:
            result = await self.validate_theory(
                self.theories[theory_id],
                kwargs.get("evidence", [])
            )
            return {"validation": result}
        return {"error": "Theory not found"}
    
    async def _analyze_relationships_handler(self, **kwargs) -> Dict[str, Any]:
        """分析关系处理器"""
        relationships = await self.analyze_hypothesis_relationships()
        return {"relationships": relationships}
    
    async def _propose_experiment_handler(self, **kwargs) -> Dict[str, Any]:
        """提出实验处理器"""
        hypothesis_id = kwargs.get("hypothesis_id")
        if hypothesis_id in self.hypotheses:
            experiments = await self.propose_experiments_for_hypothesis(
                self.hypotheses[hypothesis_id]
            )
            return {"experiments": experiments}
        return {"error": "Hypothesis not found"}
    
    # ===== 公共API =====
    
    def get_hypotheses(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取假设列表"""
        hypotheses = self.hypotheses.values()
        if status:
            hypotheses = [h for h in hypotheses if h.status == status]
        return [h.to_dict() for h in hypotheses]
    
    def get_models(self) -> List[Dict[str, Any]]:
        """获取模型列表"""
        return [m.to_dict() for m in self.models.values()]
    
    def get_theories(self) -> List[Dict[str, Any]]:
        """获取理论列表"""
        return [t.to_dict() for t in self.theories.values()]
    
    def create_theory(
        self,
        name: str,
        description: str,
        core_hypothesis_id: str
    ) -> Optional[Theory]:
        """创建新理论"""
        if core_hypothesis_id not in self.hypotheses:
            return None
        
        theory_id = f"theory_{uuid.uuid4().hex[:8]}"
        
        theory = Theory(
            id=theory_id,
            name=name,
            description=description,
            core_hypothesis=core_hypothesis_id
        )
        
        self.theories[theory_id] = theory
        return theory
