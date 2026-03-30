"""
Iterative Research - 迭代式研究工作流
实现假设→实验→评审→改进循环
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import uuid

from ..multi_agent.agent_core import Message, MessageType
from ..multi_agent.agent_orchestrator import AgentOrchestrator, Task, Workflow
from ..agents.theorist_agent import TheoristAgent, Hypothesis
from ..agents.experimentalist_agent import ExperimentalistAgent, Experiment
from ..agents.reviewer_agent import ReviewerAgent, ReviewReport


class ResearchPhase(Enum):
    """研究阶段"""
    HYPOTHESIS_GENERATION = auto()
    EXPERIMENT_DESIGN = auto()
    EXPERIMENT_EXECUTION = auto()
    DATA_ANALYSIS = auto()
    PEER_REVIEW = auto()
    REVISION = auto()
    COMPLETION = auto()


@dataclass
class IterationRecord:
    """迭代记录"""
    iteration_number: int
    phase: ResearchPhase
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration_number": self.iteration_number,
            "phase": self.phase.name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds
        }


@dataclass
class ResearchCycleConfig:
    """研究循环配置"""
    max_iterations: int = 5
    min_iterations: int = 1
    
    # 完成条件
    required_confidence: float = 0.8
    required_review_score: float = 0.7
    
    # 自动改进
    auto_improve: bool = True
    improvement_threshold: float = 0.5
    
    # 并行执行
    parallel_experiments: bool = False
    
    # 评审配置
    require_peer_review: bool = True
    min_reviewers: int = 1


class IterativeResearchWorkflow:
    """
    迭代式研究工作流
    实现科学研究的迭代循环
    """
    
    def __init__(
        self,
        orchestrator: AgentOrchestrator,
        config: Optional[ResearchCycleConfig] = None
    ):
        self.orchestrator = orchestrator
        self.config = config or ResearchCycleConfig()
        
        # 参与者
        self.theorist: Optional[TheoristAgent] = None
        self.experimentalist: Optional[ExperimentalistAgent] = None
        self.reviewer: Optional[ReviewerAgent] = None
        
        # 状态
        self.current_phase = ResearchPhase.HYPOTHESIS_GENERATION
        self.iteration_count = 0
        self.iteration_history: List[IterationRecord] = []
        
        # 当前研究对象
        self.current_hypothesis: Optional[Dict[str, Any]] = None
        self.current_experiment: Optional[Dict[str, Any]] = None
        self.current_review: Optional[Dict[str, Any]] = None
        
        # 结果
        self.final_results: Optional[Dict[str, Any]] = None
        self.completion_reason: str = ""
        
        # 回调
        self.on_phase_complete: Optional[Callable[[ResearchPhase, Dict], None]] = None
        self.on_iteration_complete: Optional[Callable[[int, Dict], None]] = None
        self.on_workflow_complete: Optional[Callable[[Dict], None]] = None
    
    def set_agents(
        self,
        theorist: TheoristAgent,
        experimentalist: ExperimentalistAgent,
        reviewer: ReviewerAgent
    ) -> None:
        """设置参与Agent"""
        self.theorist = theorist
        self.experimentalist = experimentalist
        self.reviewer = reviewer
        
        # 注册到编排器
        self.orchestrator.register_agent(theorist, ["theory"])
        self.orchestrator.register_agent(experimentalist, ["experiment"])
        self.orchestrator.register_agent(reviewer, ["review"])
    
    async def run(
        self,
        problem_statement: str,
        initial_observations: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        运行迭代研究循环
        """
        print(f"🔄 Starting Iterative Research Workflow")
        print(f"   Problem: {problem_statement[:60]}...")
        
        start_time = datetime.now()
        
        # 迭代循环
        while self.iteration_count < self.config.max_iterations:
            self.iteration_count += 1
            print(f"\n📊 Iteration {self.iteration_count}/{self.config.max_iterations}")
            
            iteration_start = datetime.now()
            
            # 阶段1: 假设生成/修订
            await self._run_hypothesis_phase(
                problem_statement,
                initial_observations if self.iteration_count == 1 else None
            )
            
            # 阶段2: 实验设计与执行
            await self._run_experiment_phase()
            
            # 阶段3: 同行评审
            if self.config.require_peer_review:
                await self._run_review_phase()
            
            # 阶段4: 检查完成条件
            should_continue, reason = self._evaluate_completion()
            
            iteration_duration = (datetime.now() - iteration_start).total_seconds()
            
            # 记录迭代
            record = IterationRecord(
                iteration_number=self.iteration_count,
                phase=self.current_phase,
                outputs={
                    "hypothesis": self.current_hypothesis,
                    "experiment": self.current_experiment,
                    "review": self.current_review
                },
                duration_seconds=iteration_duration
            )
            self.iteration_history.append(record)
            
            if self.on_iteration_complete:
                self.on_iteration_complete(self.iteration_count, record.to_dict())
            
            if not should_continue:
                self.completion_reason = reason
                break
        
        # 完成工作流
        total_duration = (datetime.now() - start_time).total_seconds()
        
        self.final_results = {
            "problem": problem_statement,
            "iterations": self.iteration_count,
            "final_hypothesis": self.current_hypothesis,
            "final_experiment": self.current_experiment,
            "final_review": self.current_review,
            "history": [r.to_dict() for r in self.iteration_history],
            "completion_reason": self.completion_reason,
            "total_duration_seconds": total_duration
        }
        
        if self.on_workflow_complete:
            self.on_workflow_complete(self.final_results)
        
        print(f"\n✅ Iterative Research Completed")
        print(f"   Total iterations: {self.iteration_count}")
        print(f"   Reason: {self.completion_reason}")
        
        return self.final_results
    
    async def _run_hypothesis_phase(
        self,
        problem_statement: str,
        initial_observations: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """运行假设生成阶段"""
        print(f"   📝 Phase: Hypothesis Generation")
        
        self.current_phase = ResearchPhase.HYPOTHESIS_GENERATION
        
        if self.iteration_count == 1:
            # 第一轮：从问题生成新假设
            if self.theorist:
                hypothesis = await self.theorist.generate_hypothesis(
                    problem_statement,
                    initial_observations or []
                )
                if hypothesis:
                    self.current_hypothesis = hypothesis.to_dict()
        else:
            # 后续轮：基于评审反馈改进
            if self.current_review and self.config.auto_improve:
                # 分析评审意见，改进假设
                improved = await self._improve_hypothesis(
                    self.current_hypothesis,
                    self.current_review
                )
                if improved:
                    self.current_hypothesis = improved
        
        if self.on_phase_complete:
            self.on_phase_complete(
                ResearchPhase.HYPOTHESIS_GENERATION,
                {"hypothesis": self.current_hypothesis}
            )
    
    async def _run_experiment_phase(self) -> None:
        """运行实验阶段"""
        print(f"   🔬 Phase: Experiment")
        
        self.current_phase = ResearchPhase.EXPERIMENT_DESIGN
        
        if not self.current_hypothesis:
            print("   ⚠️  No hypothesis to test")
            return
        
        if self.experimentalist:
            # 设计实验
            design = await self.experimentalist.design_experiment(
                self.current_hypothesis
            )
            
            if design:
                print(f"      Experiment designed: {design.name}")
                
                # 执行实验
                self.current_phase = ResearchPhase.EXPERIMENT_EXECUTION
                experiment = await self.experimentalist.execute_experiment(design.id)
                
                if experiment:
                    print(f"      Experiment completed: {len(experiment.data_points)} data points")
                    
                    # 分析数据
                    self.current_phase = ResearchPhase.DATA_ANALYSIS
                    analysis = await self.experimentalist.analyze_experiment(experiment.id)
                    
                    self.current_experiment = {
                        "design": design.to_dict(),
                        "execution": experiment.to_dict(),
                        "analysis": analysis.to_dict() if analysis else None
                    }
        
        if self.on_phase_complete:
            self.on_phase_complete(
                ResearchPhase.EXPERIMENT_EXECUTION,
                {"experiment": self.current_experiment}
            )
    
    async def _run_review_phase(self) -> None:
        """运行评审阶段"""
        print(f"   👁️  Phase: Peer Review")
        
        self.current_phase = ResearchPhase.PEER_REVIEW
        
        if not self.reviewer:
            return
        
        review_results = {}
        
        # 评审假设
        if self.current_hypothesis:
            hypothesis_review = await self.reviewer.review_hypothesis(
                self.current_hypothesis
            )
            review_results["hypothesis"] = hypothesis_review.to_dict()
        
        # 评审实验
        if self.current_experiment:
            experiment_design = self.current_experiment.get("design", {})
            experiment_results = self.current_experiment.get("execution", {})
            
            experiment_review = await self.reviewer.review_experiment(
                experiment_design,
                experiment_results
            )
            review_results["experiment"] = experiment_review.to_dict()
        
        self.current_review = review_results
        
        # 打印评审摘要
        for target_type, review in review_results.items():
            print(f"      {target_type}: {review.get('recommendation')}")
        
        if self.on_phase_complete:
            self.on_phase_complete(
                ResearchPhase.PEER_REVIEW,
                {"review": self.current_review}
            )
    
    async def _improve_hypothesis(
        self,
        hypothesis: Dict[str, Any],
        review: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        基于评审改进假设
        """
        improved = hypothesis.copy()
        
        hypothesis_review = review.get("hypothesis", {})
        comments = hypothesis_review.get("comments", [])
        
        # 处理各种评审意见
        for comment in comments:
            severity = comment.get("severity")
            category = comment.get("category")
            suggestion = comment.get("suggestion")
            
            if category == "clarity" and suggestion:
                # 改进陈述清晰度
                if "statement" in improved:
                    improved["statement"] += f" (Revised: {suggestion})"
            
            elif category == "testability":
                # 添加更多预测
                if "predictions" in improved:
                    improved["predictions"].append(
                        f"Additional prediction addressing: {suggestion}"
                    )
            
            elif category == "logic":
                # 修正逻辑问题
                improved["revisions_made"] = improved.get("revisions_made", []) + [
                    f"Fixed logic issue: {comment.get('comment')}"
                ]
        
        # 增加迭代版本号
        improved["iteration"] = self.iteration_count
        improved["improved_from"] = hypothesis.get("id")
        
        return improved
    
    def _evaluate_completion(self) -> Tuple[bool, str]:
        """
        评估是否应该继续迭代
        返回 (是否继续, 原因)
        """
        # 检查最小迭代次数
        if self.iteration_count < self.config.min_iterations:
            return True, "Minimum iterations not reached"
        
        # 检查是否有评审
        if not self.current_review:
            return True, "Awaiting review"
        
        # 检查假设质量
        if self.current_hypothesis:
            confidence = self.current_hypothesis.get("confidence", 0)
            if confidence >= self.config.required_confidence:
                # 检查评审是否接受
                hypothesis_review = self.current_review.get("hypothesis", {})
                if hypothesis_review.get("recommendation") == "accept":
                    return False, "Hypothesis validated and accepted"
        
        # 检查实验是否成功验证
        if self.current_experiment:
            analysis = self.current_experiment.get("analysis", {})
            findings = analysis.get("findings", []) if analysis else []
            
            for finding in findings:
                if finding.get("type") == "significant_effect":
                    return False, "Significant effect confirmed by experiment"
        
        # 检查是否达到最大迭代
        if self.iteration_count >= self.config.max_iterations:
            return False, "Maximum iterations reached"
        
        return True, "Continuing iteration"
    
    def get_progress_report(self) -> Dict[str, Any]:
        """获取进度报告"""
        return {
            "current_iteration": self.iteration_count,
            "current_phase": self.current_phase.name,
            "total_iterations": len(self.iteration_history),
            "hypothesis_status": "active" if self.current_hypothesis else "none",
            "experiment_status": "completed" if self.current_experiment else "pending",
            "review_status": "completed" if self.current_review else "pending"
        }


class AdaptiveIterativeResearch(IterativeResearchWorkflow):
    """
    自适应迭代研究
    根据中间结果动态调整策略
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptation_rules: List[Callable] = []
    
    def add_adaptation_rule(self, rule: Callable) -> None:
        """添加自适应规则"""
        self.adaptation_rules.append(rule)
    
    async def _evaluate_completion(self) -> Tuple[bool, str]:
        """评估并可能调整策略"""
        should_continue, reason = super()._evaluate_completion()
        
        # 应用自适应规则
        for rule in self.adaptation_rules:
            adjustment = rule(self)
            if adjustment:
                # 应用调整
                if adjustment.get("action") == "increase_iterations":
                    self.config.max_iterations += adjustment.get("amount", 1)
                    print(f"   📈 Adaptation: Increased max iterations to {self.config.max_iterations}")
                
                elif adjustment.get("action") == "skip_review":
                    self.config.require_peer_review = False
                    print(f"   ⚡ Adaptation: Skipping review for faster iteration")
        
        return should_continue, reason


class ParallelIterativeResearch(IterativeResearchWorkflow):
    """
    并行迭代研究
    同时测试多个假设
    """
    
    async def run(
        self,
        problem_statement: str,
        num_parallel_hypotheses: int = 3
    ) -> Dict[str, Any]:
        """
        运行并行研究
        """
        print(f"🔄 Starting Parallel Iterative Research")
        print(f"   Parallel hypotheses: {num_parallel_hypotheses}")
        
        # 生成多个假设
        hypotheses = []
        if self.theorist:
            for i in range(num_parallel_hypotheses):
                hypothesis = await self.theorist.generate_hypothesis(
                    problem_statement,
                    [{"variation": i}]  # 不同变体
                )
                if hypothesis:
                    hypotheses.append(hypothesis.to_dict())
        
        print(f"   Generated {len(hypotheses)} hypotheses")
        
        # 并行测试每个假设
        tasks = []
        for hypothesis in hypotheses:
            task = self._test_single_hypothesis(hypothesis)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 选择最佳结果
        valid_results = [r for r in results if isinstance(r, dict)]
        
        if valid_results:
            # 按评审分数排序
            best_result = max(
                valid_results,
                key=lambda r: self._get_result_score(r)
            )
            
            return {
                "problem": problem_statement,
                "hypotheses_tested": len(hypotheses),
                "best_result": best_result,
                "all_results": valid_results
            }
        
        return {"error": "No valid results"}
    
    async def _test_single_hypothesis(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """测试单个假设"""
        # 设计并执行实验
        if self.experimentalist:
            design = await self.experimentalist.design_experiment(hypothesis)
            if design:
                experiment = await self.experimentalist.execute_experiment(design.id)
                analysis = await self.experimentalist.analyze_experiment(experiment.id)
                
                # 评审
                review = None
                if self.reviewer:
                    review = await self.reviewer.review_experiment(
                        design.to_dict(),
                        experiment.to_dict()
                    )
                
                return {
                    "hypothesis": hypothesis,
                    "experiment": experiment.to_dict() if experiment else None,
                    "analysis": analysis.to_dict() if analysis else None,
                    "review": review.to_dict() if review else None
                }
        
        return {}
    
    def _get_result_score(self, result: Dict[str, Any]) -> float:
        """评估结果质量"""
        score = 0.0
        
        review = result.get("review", {})
        if review:
            if review.get("recommendation") == "accept":
                score += 1.0
            elif review.get("recommendation") == "minor_revision":
                score += 0.7
        
        analysis = result.get("analysis", {})
        if analysis:
            findings = analysis.get("findings", [])
            for finding in findings:
                if finding.get("type") == "significant_effect":
                    score += 0.5
        
        return score
