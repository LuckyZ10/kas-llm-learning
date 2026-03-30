"""
Reviewer Agent - 审稿人Agent
负责批判性评估、漏洞识别、改进建议
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

from ..multi_agent.agent_core import (
    DeliberativeAgent, Message, MessageType,
    Observation, Action, AgentCapability
)


class ReviewSeverity(Enum):
    """评审严重级别"""
    CRITICAL = "critical"      # 严重问题，必须修复
    MAJOR = "major"            # 主要问题，应该修复
    MINOR = "minor"            # 次要问题，建议修复
    SUGGESTION = "suggestion"  # 改进建议
    PRAISE = "praise"          # 正面评价


@dataclass
class ReviewComment:
    """评审意见"""
    id: str
    target_id: str
    target_type: str  # hypothesis, model, experiment, paper, etc.
    severity: ReviewSeverity
    category: str  # logic, methodology, presentation, etc.
    comment: str
    location: Optional[str] = None
    suggestion: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "target_id": self.target_id,
            "target_type": self.target_type,
            "severity": self.severity.value,
            "category": self.category,
            "comment": self.comment,
            "location": self.location,
            "suggestion": self.suggestion,
            "created_at": self.created_at.isoformat(),
            "resolved": self.resolved
        }


@dataclass
class ReviewReport:
    """评审报告"""
    id: str
    target_id: str
    target_type: str
    overall_assessment: str
    recommendation: str  # accept, minor_revision, major_revision, reject
    confidence: float  # 0-1
    
    comments: List[ReviewComment] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    
    reviewed_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "target_id": self.target_id,
            "target_type": self.target_type,
            "overall_assessment": self.overall_assessment,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "comments": [c.to_dict() for c in self.comments],
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "reviewed_at": self.reviewed_at.isoformat(),
            "summary": self._generate_summary()
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """生成统计摘要"""
        severity_counts = {
            "critical": 0,
            "major": 0,
            "minor": 0,
            "suggestion": 0,
            "praise": 0
        }
        
        for comment in self.comments:
            severity_counts[comment.severity.value] += 1
        
        return {
            "total_comments": len(self.comments),
            "severity_distribution": severity_counts,
            "resolved_count": sum(1 for c in self.comments if c.resolved)
        }


class ReviewCriteria:
    """评审标准"""
    
    def __init__(
        self,
        name: str,
        description: str,
        weight: float = 1.0
    ):
        self.name = name
        self.description = description
        self.weight = weight
    
    def evaluate(self, target: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        评估目标
        返回 (分数, 意见列表)
        """
        return 0.5, ["Default evaluation"]


class ReviewerAgent(DeliberativeAgent):
    """
    审稿人Agent
    负责批判性评估科学工作
    """
    
    def __init__(self, **kwargs):
        kwargs.setdefault("name", "ReviewerAgent")
        kwargs.setdefault("description", "Critically evaluates scientific work, identifies flaws, suggests improvements")
        super().__init__(**kwargs)
        
        # 知识库
        self.review_reports: Dict[str, ReviewReport] = {}
        self.comments: Dict[str, ReviewComment] = {}
        self.review_criteria: Dict[str, ReviewCriteria] = {}
        
        # 评审历史
        self.reviewed_items: Set[str] = set()
        
        # 注册能力
        self._register_capabilities()
        
        # 注册消息处理器
        self.register_message_handler(
            MessageType.COMMUNICATION,
            self._handle_review_request
        )
        
        # 初始化标准
        self._initialize_criteria()
    
    def _initialize_criteria(self) -> None:
        """初始化评审标准"""
        self.review_criteria["logical_consistency"] = ReviewCriteria(
            name="Logical Consistency",
            description="Check for logical contradictions and consistency",
            weight=1.5
        )
        
        self.review_criteria["methodology"] = ReviewCriteria(
            name="Methodology",
            description="Evaluate experimental and analytical methods",
            weight=1.3
        )
        
        self.review_criteria["evidence_quality"] = ReviewCriteria(
            name="Evidence Quality",
            description="Assess the quality and sufficiency of evidence",
            weight=1.4
        )
        
        self.review_criteria["clarity"] = ReviewCriteria(
            name="Clarity",
            description="Evaluate clarity of presentation",
            weight=1.0
        )
        
        self.review_criteria["originality"] = ReviewCriteria(
            name="Originality",
            description="Assess novelty and originality",
            weight=1.2
        )
    
    def _register_capabilities(self) -> None:
        """注册专业能力"""
        self.register_capability(AgentCapability(
            name="review_hypothesis",
            description="Review scientific hypothesis",
            handler=self._review_hypothesis_handler
        ))
        
        self.register_capability(AgentCapability(
            name="review_model",
            description="Review mathematical model",
            handler=self._review_model_handler
        ))
        
        self.register_capability(AgentCapability(
            name="review_experiment",
            description="Review experimental design and results",
            handler=self._review_experiment_handler
        ))
        
        self.register_capability(AgentCapability(
            name="identify_flaws",
            description="Identify logical and methodological flaws",
            handler=self._identify_flaws_handler
        ))
        
        self.register_capability(AgentCapability(
            name="suggest_improvements",
            description="Suggest specific improvements",
            handler=self._suggest_improvements_handler
        ))
        
        self.register_capability(AgentCapability(
            name="verify_reproducibility",
            description="Check reproducibility of results",
            handler=self._verify_reproducibility_handler
        ))
    
    async def perceive(self) -> List[Observation]:
        """感知"""
        observations = []
        
        while not self.inbox.empty():
            try:
                message = self.inbox.get_nowait()
                observations.append(Observation(
                    source="review_request",
                    data=message.content,
                    confidence=0.95
                ))
            except asyncio.QueueEmpty:
                break
        
        return observations
    
    async def reason(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """推理"""
        decision = {"actions": [], "reasoning_log": []}
        
        trigger = context.get("trigger")
        
        if trigger and trigger.get("type") == "review_request":
            target_type = trigger.get("target_type")
            target_data = trigger.get("target_data")
            
            if target_type and target_data:
                decision["actions"].append({
                    "type": "conduct_review",
                    "target_type": target_type,
                    "target_data": target_data
                })
        
        return decision
    
    async def act(self, decision: Dict[str, Any]) -> List[Action]:
        """行动"""
        actions = []
        
        for action_data in decision.get("actions", []):
            action_type = action_data.get("type")
            
            if action_type == "conduct_review":
                actions.append(Action(
                    action_type="submit_review",
                    params={
                        "target_type": action_data.get("target_type"),
                        "target_data": action_data.get("target_data")
                    },
                    priority=3
                ))
        
        return actions
    
    # ===== 核心能力实现 =====
    
    async def review_hypothesis(self, hypothesis: Dict[str, Any]) -> ReviewReport:
        """
        评审假设
        """
        target_id = hypothesis.get("id", str(uuid.uuid4()))
        report_id = f"review_{target_id}"
        
        comments = []
        strengths = []
        weaknesses = []
        
        # 检查假设陈述的清晰度
        statement = hypothesis.get("statement", "")
        if len(statement) < 20:
            comments.append(ReviewComment(
                id=f"comment_{uuid.uuid4().hex[:8]}",
                target_id=target_id,
                target_type="hypothesis",
                severity=ReviewSeverity.MAJOR,
                category="clarity",
                comment="Hypothesis statement is too brief",
                suggestion="Expand the statement to clearly specify the expected relationship"
            ))
        else:
            strengths.append("Hypothesis statement is well-articulated")
        
        # 检查假设的可测试性
        predictions = hypothesis.get("predictions", [])
        if not predictions:
            comments.append(ReviewComment(
                id=f"comment_{uuid.uuid4().hex[:8]}",
                target_id=target_id,
                target_type="hypothesis",
                severity=ReviewSeverity.CRITICAL,
                category="testability",
                comment="No testable predictions provided",
                suggestion="Derive specific, falsifiable predictions from the hypothesis"
            ))
            weaknesses.append("Lacks testable predictions")
        else:
            strengths.append(f"Includes {len(predictions)} testable predictions")
            
            # 检查预测的精确性
            for i, pred in enumerate(predictions):
                if "should" not in pred.lower() and "will" not in pred.lower():
                    comments.append(ReviewComment(
                        id=f"comment_{uuid.uuid4().hex[:8]}",
                        target_id=target_id,
                        target_type="hypothesis",
                        severity=ReviewSeverity.MINOR,
                        category="precision",
                        comment=f"Prediction {i+1} lacks directional specificity",
                        location=f"prediction[{i}]"
                    ))
        
        # 检查假设的前提
        assumptions = hypothesis.get("assumptions", [])
        if not assumptions:
            comments.append(ReviewComment(
                id=f"comment_{uuid.uuid4().hex[:8]}",
                target_id=target_id,
                target_type="hypothesis",
                severity=ReviewSeverity.MAJOR,
                category="methodology",
                comment="No underlying assumptions stated",
                suggestion="Explicitly state the assumptions underlying this hypothesis"
            ))
        
        # 逻辑一致性检查
        for assumption in assumptions:
            for prediction in predictions:
                if self._check_contradiction(assumption, prediction):
                    comments.append(ReviewComment(
                        id=f"comment_{uuid.uuid4().hex[:8]}",
                        target_id=target_id,
                        target_type="hypothesis",
                        severity=ReviewSeverity.CRITICAL,
                        category="logic",
                        comment=f"Possible contradiction between assumption and prediction",
                        suggestion="Review the logical relationship between assumptions and predictions"
                    ))
                    weaknesses.append("Internal logical inconsistency detected")
        
        # 确定总体推荐
        critical_count = sum(1 for c in comments if c.severity == ReviewSeverity.CRITICAL)
        major_count = sum(1 for c in comments if c.severity == ReviewSeverity.MAJOR)
        
        if critical_count > 0:
            recommendation = "reject"
            overall_assessment = "The hypothesis has fundamental flaws that must be addressed"
        elif major_count > 1:
            recommendation = "major_revision"
            overall_assessment = "Significant revisions are required before this hypothesis can be considered"
        elif major_count == 1 or len(comments) > 3:
            recommendation = "minor_revision"
            overall_assessment = "The hypothesis is sound but needs refinement"
        else:
            recommendation = "accept"
            overall_assessment = "A well-formulated hypothesis with clear testable predictions"
        
        confidence = hypothesis.get("confidence", 0.5)
        
        report = ReviewReport(
            id=report_id,
            target_id=target_id,
            target_type="hypothesis",
            overall_assessment=overall_assessment,
            recommendation=recommendation,
            confidence=confidence,
            comments=comments,
            strengths=strengths,
            weaknesses=weaknesses
        )
        
        self.review_reports[report_id] = report
        self.reviewed_items.add(target_id)
        
        return report
    
    async def review_model(self, model: Dict[str, Any]) -> ReviewReport:
        """
        评审数学模型
        """
        target_id = model.get("id", str(uuid.uuid4()))
        report_id = f"review_{target_id}"
        
        comments = []
        strengths = []
        weaknesses = []
        
        # 检查模型方程
        equations = model.get("equations", [])
        if not equations:
            comments.append(ReviewComment(
                id=f"comment_{uuid.uuid4().hex[:8]}",
                target_id=target_id,
                target_type="model",
                severity=ReviewSeverity.CRITICAL,
                category="completeness",
                comment="No mathematical equations provided",
                suggestion="Include the formal mathematical description of the model"
            ))
        else:
            strengths.append(f"Model includes {len(equations)} equations")
        
        # 检查参数定义
        parameters = model.get("parameters", {})
        if not parameters:
            comments.append(ReviewComment(
                id=f"comment_{uuid.uuid4().hex[:8]}",
                target_id=target_id,
                target_type="model",
                severity=ReviewSeverity.MAJOR,
                category="completeness",
                comment="Model parameters not defined",
                suggestion="Provide complete parameter definitions with units and constraints"
            ))
        
        # 检查假设
        assumptions = model.get("assumptions", [])
        if len(assumptions) < 2:
            comments.append(ReviewComment(
                id=f"comment_{uuid.uuid4().hex[:8]}",
                target_id=target_id,
                target_type="model",
                severity=ReviewSeverity.MINOR,
                category="rigor",
                comment="Limited assumptions stated",
                suggestion="Consider additional simplifying assumptions and their implications"
            ))
        
        # 验证状态
        validation_status = model.get("validation_status", "unvalidated")
        if validation_status == "unvalidated":
            comments.append(ReviewComment(
                id=f"comment_{uuid.uuid4().hex[:8]}",
                target_id=target_id,
                target_type="model",
                severity=ReviewSeverity.MAJOR,
                category="validation",
                comment="Model has not been validated",
                suggestion="Validate the model against empirical data or analytical solutions"
            ))
            weaknesses.append("Lacks validation")
        else:
            strengths.append(f"Model validation status: {validation_status}")
        
        # 维度分析（简化）
        if equations:
            comments.append(ReviewComment(
                id=f"comment_{uuid.uuid4().hex[:8]}",
                target_id=target_id,
                target_type="model",
                severity=ReviewSeverity.SUGGESTION,
                category="methodology",
                comment="Consider performing dimensional analysis",
                suggestion="Verify dimensional consistency of all equations"
            ))
        
        # 确定推荐
        critical_count = sum(1 for c in comments if c.severity == ReviewSeverity.CRITICAL)
        
        if critical_count > 0:
            recommendation = "major_revision"
            overall_assessment = "The model requires substantial development"
        elif len([c for c in comments if c.severity == ReviewSeverity.MAJOR]) > 1:
            recommendation = "minor_revision"
            overall_assessment = "The model structure is adequate but needs refinement"
        else:
            recommendation = "accept"
            overall_assessment = "A well-structured mathematical model"
        
        report = ReviewReport(
            id=report_id,
            target_id=target_id,
            target_type="model",
            overall_assessment=overall_assessment,
            recommendation=recommendation,
            confidence=0.7,
            comments=comments,
            strengths=strengths,
            weaknesses=weaknesses
        )
        
        self.review_reports[report_id] = report
        self.reviewed_items.add(target_id)
        
        return report
    
    async def review_experiment(
        self,
        experiment_design: Dict[str, Any],
        experiment_results: Optional[Dict[str, Any]] = None
    ) -> ReviewReport:
        """
        评审实验
        """
        target_id = experiment_design.get("id", str(uuid.uuid4()))
        report_id = f"review_{target_id}"
        
        comments = []
        strengths = []
        weaknesses = []
        
        # 检查实验设计
        objective = experiment_design.get("objective", "")
        if not objective:
            comments.append(ReviewComment(
                id=f"comment_{uuid.uuid4().hex[:8]}",
                target_id=target_id,
                target_type="experiment",
                severity=ReviewSeverity.CRITICAL,
                category="clarity",
                comment="Experiment objective not clearly stated"
            ))
        
        # 检查变量定义
        independent_vars = experiment_design.get("independent_variables", [])
        dependent_vars = experiment_design.get("dependent_variables", [])
        control_vars = experiment_design.get("control_variables", [])
        
        if not independent_vars:
            comments.append(ReviewComment(
                id=f"comment_{uuid.uuid4().hex[:8]}",
                target_id=target_id,
                target_type="experiment",
                severity=ReviewSeverity.CRITICAL,
                category="design",
                comment="No independent variables defined"
            ))
        
        if not dependent_vars:
            comments.append(ReviewComment(
                id=f"comment_{uuid.uuid4().hex[:8]}",
                target_id=target_id,
                target_type="experiment",
                severity=ReviewSeverity.CRITICAL,
                category="design",
                comment="No dependent variables defined"
            ))
        
        if independent_vars and dependent_vars:
            strengths.append("Variables are clearly defined")
        
        # 检查样本量
        sample_size = experiment_design.get("sample_size", 0)
        if sample_size < 10:
            comments.append(ReviewComment(
                id=f"comment_{uuid.uuid4().hex[:8]}",
                target_id=target_id,
                target_type="experiment",
                severity=ReviewSeverity.MAJOR,
                category="statistical_power",
                comment=f"Sample size ({sample_size}) may be insufficient",
                suggestion="Conduct power analysis to determine adequate sample size"
            ))
        elif sample_size >= 30:
            strengths.append(f"Adequate sample size: {sample_size}")
        
        # 检查控制
        if not control_vars:
            comments.append(ReviewComment(
                id=f"comment_{uuid.uuid4().hex[:8]}",
                target_id=target_id,
                target_type="experiment",
                severity=ReviewSeverity.MAJOR,
                category="controls",
                comment="No control variables specified",
                suggestion="Identify and control potential confounding variables"
            ))
        else:
            strengths.append(f"Includes {len(control_vars)} control variables")
        
        # 检查成功标准
        success_criteria = experiment_design.get("success_criteria", {})
        if not success_criteria:
            comments.append(ReviewComment(
                id=f"comment_{uuid.uuid4().hex[:8]}",
                target_id=target_id,
                target_type="experiment",
                severity=ReviewSeverity.MAJOR,
                category="design",
                comment="Success criteria not defined",
                suggestion="Specify clear criteria for determining whether results support the hypothesis"
            ))
        
        # 评审结果（如果有）
        if experiment_results:
            data_points = experiment_results.get("data_points_count", 0)
            if data_points == 0:
                comments.append(ReviewComment(
                    id=f"comment_{uuid.uuid4().hex[:8]}",
                    target_id=target_id,
                    target_type="experiment",
                    severity=ReviewSeverity.CRITICAL,
                    category="results",
                    comment="No data collected"
                ))
            
            # 检查统计分析
            if "analysis" not in experiment_results:
                comments.append(ReviewComment(
                    id=f"comment_{uuid.uuid4().hex[:8]}",
                    target_id=target_id,
                    target_type="experiment",
                    severity=ReviewSeverity.MAJOR,
                    category="analysis",
                    comment="No statistical analysis reported"
                ))
        
        # 确定推荐
        critical_count = sum(1 for c in comments if c.severity == ReviewSeverity.CRITICAL)
        
        if critical_count > 1:
            recommendation = "reject"
            overall_assessment = "The experimental design has fundamental flaws"
        elif critical_count == 1:
            recommendation = "major_revision"
            overall_assessment = "The design needs significant improvement"
        elif len([c for c in comments if c.severity == ReviewSeverity.MAJOR]) > 0:
            recommendation = "minor_revision"
            overall_assessment = "The design is generally sound but needs refinement"
        else:
            recommendation = "accept"
            overall_assessment = "A well-designed experiment"
        
        report = ReviewReport(
            id=report_id,
            target_id=target_id,
            target_type="experiment",
            overall_assessment=overall_assessment,
            recommendation=recommendation,
            confidence=0.75,
            comments=comments,
            strengths=strengths,
            weaknesses=weaknesses
        )
        
        self.review_reports[report_id] = report
        self.reviewed_items.add(target_id)
        
        return report
    
    async def identify_flaws(self, target: Dict[str, Any], target_type: str) -> List[ReviewComment]:
        """
        识别漏洞
        """
        flaws = []
        
        # 通用漏洞检查
        
        # 1. 检查循环论证
        if self._detect_circular_reasoning(target):
            flaws.append(ReviewComment(
                id=f"flaw_{uuid.uuid4().hex[:8]}",
                target_id=target.get("id", ""),
                target_type=target_type,
                severity=ReviewSeverity.CRITICAL,
                category="logic",
                comment="Potential circular reasoning detected",
                suggestion="Ensure arguments do not assume what they are trying to prove"
            ))
        
        # 2. 检查选择性偏差
        if self._detect_selection_bias(target):
            flaws.append(ReviewComment(
                id=f"flaw_{uuid.uuid4().hex[:8]}",
                target_id=target.get("id", ""),
                target_type=target_type,
                severity=ReviewSeverity.MAJOR,
                category="bias",
                comment="Possible selection bias",
                suggestion="Review sampling methodology for representativeness"
            ))
        
        # 3. 检查混杂变量
        if self._detect_confounding(target):
            flaws.append(ReviewComment(
                id=f"flaw_{uuid.uuid4().hex[:8]}",
                target_id=target.get("id", ""),
                target_type=target_type,
                severity=ReviewSeverity.MAJOR,
                category="methodology",
                comment="Potential uncontrolled confounding variables",
                suggestion="Identify and control for all relevant confounders"
            ))
        
        return flaws
    
    async def suggest_improvements(
        self,
        target: Dict[str, Any],
        target_type: str,
        review_report: Optional[ReviewReport] = None
    ) -> List[Dict[str, Any]]:
        """
        提出改进建议
        """
        suggestions = []
        
        # 基于评审报告生成建议
        if review_report:
            for comment in review_report.comments:
                if comment.suggestion:
                    suggestions.append({
                        "category": comment.category,
                        "priority": comment.severity.value,
                        "description": comment.suggestion,
                        "related_issue": comment.comment
                    })
        
        # 通用改进建议
        if target_type == "hypothesis":
            suggestions.extend([
                {
                    "category": "formulation",
                    "priority": "suggestion",
                    "description": "Consider alternative hypotheses that could explain the same observations"
                },
                {
                    "category": "scope",
                    "priority": "suggestion",
                    "description": "Define the boundary conditions under which the hypothesis applies"
                }
            ])
        
        elif target_type == "experiment":
            suggestions.extend([
                {
                    "category": "design",
                    "priority": "suggestion",
                    "description": "Consider including negative controls"
                },
                {
                    "category": "analysis",
                    "priority": "suggestion",
                    "description": "Report effect sizes in addition to p-values"
                }
            ])
        
        return suggestions
    
    async def verify_reproducibility(
        self,
        experiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        验证可重复性
        """
        reproducibility_score = 0.0
        issues = []
        
        # 检查实验描述的完整性
        required_elements = [
            "objective", "independent_variables", "dependent_variables",
            "control_variables", "sample_size", "measurements"
        ]
        
        present_elements = sum(1 for elem in required_elements if elem in experiment)
        completeness = present_elements / len(required_elements)
        
        if completeness < 0.5:
            issues.append("Insufficient detail for reproduction")
            reproducibility_score += 0.1
        elif completeness < 0.8:
            issues.append("Some details missing for complete reproduction")
            reproducibility_score += 0.5
        else:
            reproducibility_score += 0.3
        
        # 检查是否有原始数据
        if "data_points" in experiment or "data_points_count" in experiment:
            reproducibility_score += 0.3
        else:
            issues.append("Raw data not provided")
        
        # 检查分析代码
        if "analysis_script" in experiment:
            reproducibility_score += 0.2
        else:
            issues.append("Analysis script not provided")
        
        # 检查材料和方法
        if "materials" in experiment and "methods" in experiment:
            reproducibility_score += 0.2
        else:
            issues.append("Detailed materials and methods needed")
        
        return {
            "score": min(reproducibility_score, 1.0),
            "issues": issues,
            "recommendations": [
                "Provide complete experimental protocols",
                "Share raw data and analysis code",
                "Include version information for software and materials"
            ]
        }
    
    # ===== 辅助方法 =====
    
    def _check_contradiction(self, statement1: str, statement2: str) -> bool:
        """检查矛盾"""
        # 简化的矛盾检测
        contradictions = [
            ("increase", "decrease"),
            ("positive", "negative"),
            ("all", "none"),
            ("always", "never")
        ]
        
        s1_lower = statement1.lower()
        s2_lower = statement2.lower()
        
        for word1, word2 in contradictions:
            if (word1 in s1_lower and word2 in s2_lower) or \
               (word2 in s1_lower and word1 in s2_lower):
                return True
        
        return False
    
    def _detect_circular_reasoning(self, target: Dict[str, Any]) -> bool:
        """检测循环论证"""
        # 简化实现
        return False
    
    def _detect_selection_bias(self, target: Dict[str, Any]) -> bool:
        """检测选择偏差"""
        # 简化实现
        if "sample_selection" in target:
            selection = target["sample_selection"]
            return "convenience" in str(selection).lower()
        return False
    
    def _detect_confounding(self, target: Dict[str, Any]) -> bool:
        """检测混杂变量"""
        # 简化实现
        control_vars = target.get("control_variables", [])
        return len(control_vars) < 2
    
    # ===== 消息处理器 =====
    
    async def _handle_review_request(self, message: Message) -> None:
        """处理评审请求"""
        content = message.content
        
        if content.get("request_type") == "review":
            target_type = content.get("target_type")
            target_data = content.get("target_data")
            
            if target_type == "hypothesis":
                report = await self.review_hypothesis(target_data)
            elif target_type == "model":
                report = await self.review_model(target_data)
            elif target_type == "experiment":
                report = await self.review_experiment(
                    target_data,
                    content.get("results")
                )
            else:
                report = None
            
            if report:
                await self.send_message(
                    {"review_report": report.to_dict()},
                    receiver_id=message.sender_id,
                    message_type=MessageType.RESULT
                )
    
    # ===== 能力处理器 =====
    
    async def _review_hypothesis_handler(self, **kwargs) -> Dict[str, Any]:
        report = await self.review_hypothesis(kwargs.get("hypothesis", {}))
        return {"report": report.to_dict()}
    
    async def _review_model_handler(self, **kwargs) -> Dict[str, Any]:
        report = await self.review_model(kwargs.get("model", {}))
        return {"report": report.to_dict()}
    
    async def _review_experiment_handler(self, **kwargs) -> Dict[str, Any]:
        report = await self.review_experiment(
            kwargs.get("experiment_design", {}),
            kwargs.get("experiment_results")
        )
        return {"report": report.to_dict()}
    
    async def _identify_flaws_handler(self, **kwargs) -> Dict[str, Any]:
        flaws = await self.identify_flaws(
            kwargs.get("target", {}),
            kwargs.get("target_type", "")
        )
        return {"flaws": [f.to_dict() for f in flaws]}
    
    async def _suggest_improvements_handler(self, **kwargs) -> Dict[str, Any]:
        report_id = kwargs.get("report_id")
        report = self.review_reports.get(report_id)
        
        suggestions = await self.suggest_improvements(
            kwargs.get("target", {}),
            kwargs.get("target_type", ""),
            report
        )
        return {"suggestions": suggestions}
    
    async def _verify_reproducibility_handler(self, **kwargs) -> Dict[str, Any]:
        result = await self.verify_reproducibility(kwargs.get("experiment", {}))
        return result
    
    # ===== 公共API =====
    
    def get_review_reports(self, target_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取评审报告"""
        if target_id:
            report_id = f"review_{target_id}"
            report = self.review_reports.get(report_id)
            return [report.to_dict()] if report else []
        
        return [r.to_dict() for r in self.review_reports.values()]
    
    def get_review_statistics(self) -> Dict[str, Any]:
        """获取评审统计"""
        total = len(self.review_reports)
        
        recommendations = {}
        for report in self.review_reports.values():
            rec = report.recommendation
            recommendations[rec] = recommendations.get(rec, 0) + 1
        
        total_comments = sum(len(r.comments) for r in self.review_reports.values())
        
        severity_counts = {"critical": 0, "major": 0, "minor": 0, "suggestion": 0, "praise": 0}
        for report in self.review_reports.values():
            for comment in report.comments:
                severity_counts[comment.severity.value] += 1
        
        return {
            "total_reviews": total,
            "recommendation_distribution": recommendations,
            "total_comments": total_comments,
            "severity_distribution": severity_counts,
            "items_reviewed": len(self.reviewed_items)
        }
