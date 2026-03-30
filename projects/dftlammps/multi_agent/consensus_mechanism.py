"""
Consensus Mechanism - 共识机制
实现多Agent决策达成一致
"""
from __future__ import annotations
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import statistics

from .agent_core import Message, MessageType


class VoteType(Enum):
    """投票类型"""
    APPROVE = auto()
    REJECT = auto()
    ABSTAIN = auto()
    PROPOSAL = auto()


class ConsensusStatus(Enum):
    """共识状态"""
    PENDING = auto()
    APPROVED = auto()
    REJECTED = auto()
    TIMED_OUT = auto()
    CONFLICT = auto()


@dataclass
class Vote:
    """投票"""
    voter_id: str
    vote_type: VoteType
    proposal_id: str
    weight: float = 1.0
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Proposal:
    """提案"""
    id: str
    proposer_id: str
    content: Dict[str, Any]
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    required_approvals: int = 1
    required_rejection_ratio: float = 0.5
    
    # 状态
    votes: Dict[str, Vote] = field(default_factory=dict)
    status: ConsensusStatus = ConsensusStatus.PENDING
    final_result: Optional[Dict[str, Any]] = None
    
    def get_vote_summary(self) -> Dict[str, Any]:
        """获取投票摘要"""
        approve_weight = sum(
            v.weight for v in self.votes.values()
            if v.vote_type == VoteType.APPROVE
        )
        reject_weight = sum(
            v.weight for v in self.votes.values()
            if v.vote_type == VoteType.REJECT
        )
        abstain_weight = sum(
            v.weight for v in self.votes.values()
            if v.vote_type == VoteType.ABSTAIN
        )
        total_weight = approve_weight + reject_weight + abstain_weight
        
        return {
            "total_votes": len(self.votes),
            "approve_weight": approve_weight,
            "reject_weight": reject_weight,
            "abstain_weight": abstain_weight,
            "total_weight": total_weight,
            "approval_ratio": approve_weight / total_weight if total_weight > 0 else 0,
            "rejection_ratio": reject_weight / total_weight if total_weight > 0 else 0
        }


class ConsensusAlgorithm(ABC):
    """共识算法基类"""
    
    @abstractmethod
    async def evaluate(self, proposal: Proposal, votes: List[Vote]) -> ConsensusStatus:
        """评估是否达成共识"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取算法名称"""
        pass


class MajorityConsensus(ConsensusAlgorithm):
    """
    多数决共识算法
    简单多数通过
    """
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold  # 通过阈值
    
    def get_name(self) -> str:
        return f"MajorityConsensus(threshold={self.threshold})"
    
    async def evaluate(self, proposal: Proposal, votes: List[Vote]) -> ConsensusStatus:
        if not votes:
            return ConsensusStatus.PENDING
        
        summary = proposal.get_vote_summary()
        
        if summary["approval_ratio"] >= self.threshold:
            return ConsensusStatus.APPROVED
        elif summary["rejection_ratio"] > proposal.required_rejection_ratio:
            return ConsensusStatus.REJECTED
        
        return ConsensusStatus.PENDING


class WeightedConsensus(ConsensusAlgorithm):
    """
    加权共识算法
    考虑投票权重
    """
    
    def __init__(
        self,
        approval_threshold: float = 0.6,
        min_participation: float = 0.7
    ):
        self.approval_threshold = approval_threshold
        self.min_participation = min_participation
    
    def get_name(self) -> str:
        return f"WeightedConsensus(approval={self.approval_threshold}, participation={self.min_participation})"
    
    async def evaluate(self, proposal: Proposal, votes: List[Vote]) -> ConsensusStatus:
        if not votes:
            return ConsensusStatus.PENDING
        
        summary = proposal.get_vote_summary()
        
        # 检查参与度
        # 假设总权重为1.0（需要外部设置）
        total_expected_weight = proposal.metadata.get("total_expected_weight", 1.0)
        participation = summary["total_weight"] / total_expected_weight
        
        if participation < self.min_participation:
            return ConsensusStatus.PENDING
        
        # 检查是否通过
        if summary["approval_ratio"] >= self.approval_threshold:
            return ConsensusStatus.APPROVED
        elif summary["rejection_ratio"] > (1 - self.approval_threshold):
            return ConsensusStatus.REJECTED
        
        return ConsensusStatus.PENDING


class BordaConsensus(ConsensusAlgorithm):
    """
    博尔达计数法
    用于多选项排序
    """
    
    def __init__(self):
        self.option_scores: Dict[str, float] = defaultdict(float)
    
    def get_name(self) -> str:
        return "BordaConsensus"
    
    async def evaluate(self, proposal: Proposal, votes: List[Vote]) -> ConsensusStatus:
        options = proposal.content.get("options", [])
        
        if not options or not votes:
            return ConsensusStatus.PENDING
        
        # 计算博尔达分数
        scores = defaultdict(float)
        
        for vote in votes:
            ranking = vote.metadata.get("ranking", [])
            for idx, option in enumerate(ranking):
                # 排名越靠前分数越高
                scores[option] += len(options) - idx
        
        # 选择最高分
        if scores:
            winner = max(scores.items(), key=lambda x: x[1])
            proposal.final_result = {"winner": winner[0], "scores": dict(scores)}
            return ConsensusStatus.APPROVED
        
        return ConsensusStatus.PENDING


class DelphiConsensus(ConsensusAlgorithm):
    """
    德尔菲法共识
    多轮迭代收敛
    """
    
    def __init__(
        self,
        max_rounds: int = 3,
        convergence_threshold: float = 0.8
    ):
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold
        self.rounds: Dict[str, int] = {}
        self.historical_votes: Dict[str, List[List[Vote]]] = defaultdict(list)
    
    def get_name(self) -> str:
        return f"DelphiConsensus(rounds={self.max_rounds})"
    
    async def evaluate(self, proposal: Proposal, votes: List[Vote]) -> ConsensusStatus:
        proposal_id = proposal.id
        
        # 记录当前轮投票
        if proposal_id not in self.historical_votes:
            self.rounds[proposal_id] = 0
        
        self.historical_votes[proposal_id].append(votes)
        current_round = self.rounds[proposal_id]
        
        if current_round >= self.max_rounds - 1:
            # 最后一轮，使用多数决
            approve_count = sum(1 for v in votes if v.vote_type == VoteType.APPROVE)
            if approve_count > len(votes) / 2:
                return ConsensusStatus.APPROVED
            return ConsensusStatus.REJECTED
        
        # 计算收敛度
        if current_round > 0:
            convergence = self._calculate_convergence(proposal_id)
            if convergence >= self.convergence_threshold:
                # 已收敛，提前结束
                summary = proposal.get_vote_summary()
                if summary["approval_ratio"] > 0.5:
                    return ConsensusStatus.APPROVED
                return ConsensusStatus.REJECTED
        
        self.rounds[proposal_id] += 1
        return ConsensusStatus.PENDING
    
    def _calculate_convergence(self, proposal_id: str) -> float:
        """计算收敛度"""
        history = self.historical_votes[proposal_id]
        if len(history) < 2:
            return 0.0
        
        # 比较最近两轮投票的相似度
        current = history[-1]
        previous = history[-2]
        
        current_approvals = {v.voter_id for v in current if v.vote_type == VoteType.APPROVE}
        previous_approvals = {v.voter_id for v in previous if v.vote_type == VoteType.APPROVE}
        
        if not current_approvals and not previous_approvals:
            return 1.0
        
        intersection = len(current_approvals & previous_approvals)
        union = len(current_approvals | previous_approvals)
        
        return intersection / union if union > 0 else 0.0


class ConsensusManager:
    """
    共识管理器
    管理多个提案的共识过程
    """
    
    def __init__(
        self,
        default_algorithm: Optional[ConsensusAlgorithm] = None
    ):
        self.algorithm = default_algorithm or MajorityConsensus()
        self.proposals: Dict[str, Proposal] = {}
        self.participants: Set[str] = set()
        self.vote_weights: Dict[str, float] = {}
        
        # 回调
        self.on_consensus_reached: Optional[Callable[[Proposal], None]] = None
        self.on_vote_received: Optional[Callable[[Vote], None]] = None
        
        # 超时检查任务
        self._timeout_task: Optional[asyncio.Task] = None
        self._running = False
    
    def set_default_algorithm(self, algorithm: ConsensusAlgorithm) -> None:
        """设置默认共识算法"""
        self.algorithm = algorithm
    
    def register_participant(self, agent_id: str, weight: float = 1.0) -> None:
        """注册参与者"""
        self.participants.add(agent_id)
        self.vote_weights[agent_id] = weight
    
    def unregister_participant(self, agent_id: str) -> None:
        """注销参与者"""
        self.participants.discard(agent_id)
        self.vote_weights.pop(agent_id, None)
    
    def create_proposal(
        self,
        proposal_id: str,
        proposer_id: str,
        content: Dict[str, Any],
        description: str,
        deadline_seconds: Optional[int] = None,
        required_approvals: Optional[int] = None,
        algorithm: Optional[ConsensusAlgorithm] = None
    ) -> Proposal:
        """创建新提案"""
        deadline = None
        if deadline_seconds:
            deadline = datetime.now() + timedelta(seconds=deadline_seconds)
        
        if required_approvals is None:
            required_approvals = len(self.participants) // 2 + 1
        
        proposal = Proposal(
            id=proposal_id,
            proposer_id=proposer_id,
            content=content,
            description=description,
            deadline=deadline,
            required_approvals=required_approvals
        )
        
        proposal.metadata["algorithm"] = algorithm or self.algorithm
        
        self.proposals[proposal_id] = proposal
        return proposal
    
    async def submit_vote(
        self,
        proposal_id: str,
        voter_id: str,
        vote_type: VoteType,
        reasoning: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """提交投票"""
        if proposal_id not in self.proposals:
            return False
        
        if voter_id not in self.participants:
            return False
        
        proposal = self.proposals[proposal_id]
        
        if proposal.status != ConsensusStatus.PENDING:
            return False
        
        # 检查是否超时
        if proposal.deadline and datetime.now() > proposal.deadline:
            proposal.status = ConsensusStatus.TIMED_OUT
            return False
        
        vote = Vote(
            voter_id=voter_id,
            vote_type=vote_type,
            proposal_id=proposal_id,
            weight=self.vote_weights.get(voter_id, 1.0),
            reasoning=reasoning,
            metadata=metadata or {}
        )
        
        proposal.votes[voter_id] = vote
        
        # 触发回调
        if self.on_vote_received:
            if asyncio.iscoroutinefunction(self.on_vote_received):
                await self.on_vote_received(vote)
            else:
                self.on_vote_received(vote)
        
        # 评估共识
        await self._evaluate_consensus(proposal_id)
        
        return True
    
    async def _evaluate_consensus(self, proposal_id: str) -> None:
        """评估是否达成共识"""
        proposal = self.proposals[proposal_id]
        
        if proposal.status != ConsensusStatus.PENDING:
            return
        
        algorithm = proposal.metadata.get("algorithm", self.algorithm)
        votes = list(proposal.votes.values())
        
        status = await algorithm.evaluate(proposal, votes)
        proposal.status = status
        
        if status in [ConsensusStatus.APPROVED, ConsensusStatus.REJECTED]:
            if self.on_consensus_reached:
                if asyncio.iscoroutinefunction(self.on_consensus_reached):
                    await self.on_consensus_reached(proposal)
                else:
                    self.on_consensus_reached(proposal)
    
    async def start(self) -> None:
        """启动共识管理器"""
        self._running = True
        self._timeout_task = asyncio.create_task(self._timeout_checker())
    
    async def stop(self) -> None:
        """停止共识管理器"""
        self._running = False
        if self._timeout_task:
            self._timeout_task.cancel()
            try:
                await self._timeout_task
            except asyncio.CancelledError:
                pass
    
    async def _timeout_checker(self) -> None:
        """超时检查器"""
        while self._running:
            now = datetime.now()
            
            for proposal in self.proposals.values():
                if (
                    proposal.status == ConsensusStatus.PENDING and
                    proposal.deadline and
                    now > proposal.deadline
                ):
                    proposal.status = ConsensusStatus.TIMED_OUT
            
            await asyncio.sleep(1)
    
    def get_proposal_status(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """获取提案状态"""
        if proposal_id not in self.proposals:
            return None
        
        proposal = self.proposals[proposal_id]
        
        return {
            "id": proposal.id,
            "proposer_id": proposal.proposer_id,
            "description": proposal.description,
            "status": proposal.status.name,
            "vote_summary": proposal.get_vote_summary(),
            "votes": [
                {
                    "voter_id": v.voter_id,
                    "vote_type": v.vote_type.name,
                    "reasoning": v.reasoning
                }
                for v in proposal.votes.values()
            ],
            "final_result": proposal.final_result,
            "deadline": proposal.deadline.isoformat() if proposal.deadline else None
        }
    
    def get_all_proposals(self) -> List[Dict[str, Any]]:
        """获取所有提案状态"""
        return [
            self.get_proposal_status(pid)
            for pid in self.proposals.keys()
        ]


class CollaborativeDecisionMaking:
    """
    协作决策
    支持多种决策策略
    """
    
    def __init__(self, consensus_manager: ConsensusManager):
        self.consensus_manager = consensus_manager
        self.decision_strategies: Dict[str, Callable] = {}
    
    def register_strategy(self, name: str, strategy: Callable) -> None:
        """注册决策策略"""
        self.decision_strategies[name] = strategy
    
    async def make_decision(
        self,
        options: List[Dict[str, Any]],
        criteria: List[str],
        participants: List[str],
        strategy: str = "weighted_voting"
    ) -> Dict[str, Any]:
        """
        协作决策
        
        Args:
            options: 选项列表
            criteria: 评估标准
            participants: 参与者列表
            strategy: 决策策略
        """
        if strategy not in self.decision_strategies:
            # 默认加权投票
            return await self._weighted_voting_decision(options, criteria, participants)
        
        return await self.decision_strategies[strategy](
            options, criteria, participants
        )
    
    async def _weighted_voting_decision(
        self,
        options: List[Dict[str, Any]],
        criteria: List[str],
        participants: List[str]
    ) -> Dict[str, Any]:
        """加权投票决策"""
        # 创建提案
        proposal_id = f"decision_{datetime.now().timestamp()}"
        
        proposal = self.consensus_manager.create_proposal(
            proposal_id=proposal_id,
            proposer_id="system",
            content={"options": options, "criteria": criteria},
            description="Collaborative decision making",
            algorithm=BordaConsensus()
        )
        
        # 等待投票完成
        while proposal.status == ConsensusStatus.PENDING:
            await asyncio.sleep(0.1)
        
        return {
            "proposal_id": proposal_id,
            "status": proposal.status.name,
            "result": proposal.final_result
        }
    
    async def expert_evaluation(
        self,
        problem: Dict[str, Any],
        experts: List[str],
        rounds: int = 3
    ) -> Dict[str, Any]:
        """
        专家评估（德尔菲法）
        """
        proposal_id = f"expert_{datetime.now().timestamp()}"
        
        # 第一轮：专家独立评估
        # 第二轮：分享匿名结果，重新评估
        # 第三轮：最终评估
        
        results = []
        for round_num in range(rounds):
            round_results = []
            
            for expert in experts:
                # 收集专家意见
                # 这里简化处理，实际应该向Agent发送消息获取评估
                evaluation = {
                    "expert": expert,
                    "round": round_num,
                    "assessment": "pending"
                }
                round_results.append(evaluation)
            
            results.append(round_results)
            
            # 如果不是最后一轮，匿名分享结果
            if round_num < rounds - 1:
                # 分享上一轮的中位数/平均结果
                pass
        
        return {
            "rounds": results,
            "final_consensus": "pending"
        }


class ConflictResolution:
    """
    冲突解决机制
    处理Agent间的分歧
    """
    
    def __init__(self, consensus_manager: ConsensusManager):
        self.consensus_manager = consensus_manager
        self.resolution_strategies = {
            "mediation": self._mediation,
            "arbitration": self._arbitration,
            "negotiation": self._negotiation,
            "voting": self._voting_resolution
        }
    
    async def resolve_conflict(
        self,
        conflict_id: str,
        conflicting_parties: List[str],
        conflict_description: str,
        strategy: str = "mediation"
    ) -> Dict[str, Any]:
        """
        解决冲突
        
        Args:
            conflict_id: 冲突ID
            conflicting_parties: 冲突各方
            conflict_description: 冲突描述
            strategy: 解决策略
        """
        if strategy not in self.resolution_strategies:
            strategy = "voting"
        
        return await self.resolution_strategies[strategy](
            conflict_id, conflicting_parties, conflict_description
        )
    
    async def _mediation(
        self,
        conflict_id: str,
        parties: List[str],
        description: str
    ) -> Dict[str, Any]:
        """调解方式解决冲突"""
        # 创建调解提案
        proposal_id = f"mediation_{conflict_id}"
        
        proposal = self.consensus_manager.create_proposal(
            proposal_id=proposal_id,
            proposer_id="mediator",
            content={"conflict": description, "parties": parties},
            description=f"Mediation for conflict: {description}",
            algorithm=MajorityConsensus(threshold=0.7)
        )
        
        return {
            "conflict_id": conflict_id,
            "resolution_type": "mediation",
            "proposal_id": proposal_id,
            "status": "pending_votes"
        }
    
    async def _arbitration(
        self,
        conflict_id: str,
        parties: List[str],
        description: str
    ) -> Dict[str, Any]:
        """仲裁方式解决冲突"""
        # 选择仲裁者（可以是协调者Agent或其他机制）
        arbitrator = "coordinator"
        
        return {
            "conflict_id": conflict_id,
            "resolution_type": "arbitration",
            "arbitrator": arbitrator,
            "decision": "pending_arbitrator_judgment"
        }
    
    async def _negotiation(
        self,
        conflict_id: str,
        parties: List[str],
        description: str
    ) -> Dict[str, Any]:
        """协商方式解决冲突"""
        # 启动多轮协商
        negotiation_rounds = []
        
        for round_num in range(3):
            round_offers = []
            
            for party in parties:
                # 收集各方的让步方案
                offer = {
                    "party": party,
                    "round": round_num,
                    "offer": f"offer_from_{party}_round_{round_num}"
                }
                round_offers.append(offer)
            
            negotiation_rounds.append(round_offers)
            
            # 检查是否达成一致
            # 简化：检查是否有相同的offer
        
        return {
            "conflict_id": conflict_id,
            "resolution_type": "negotiation",
            "rounds": negotiation_rounds,
            "outcome": "pending_evaluation"
        }
    
    async def _voting_resolution(
        self,
        conflict_id: str,
        parties: List[str],
        description: str
    ) -> Dict[str, Any]:
        """投票方式解决冲突"""
        proposal_id = f"conflict_{conflict_id}"
        
        proposal = self.consensus_manager.create_proposal(
            proposal_id=proposal_id,
            proposer_id="system",
            content={"conflict": description, "options": parties},
            description=f"Vote to resolve conflict: {description}",
            algorithm=MajorityConsensus(threshold=0.5)
        )
        
        return {
            "conflict_id": conflict_id,
            "resolution_type": "voting",
            "proposal_id": proposal_id
        }
