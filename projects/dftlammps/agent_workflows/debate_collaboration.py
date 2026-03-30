"""
Debate Collaboration - 辩论式协作工作流
多Agent辩论产生更优方案
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import uuid

from ..multi_agent.agent_core import Message, MessageType
from ..multi_agent.agent_orchestrator import AgentOrchestrator
from ..multi_agent.consensus_mechanism import (
    ConsensusManager, MajorityConsensus, VoteType, Proposal
)


class DebateRole(Enum):
    """辩论角色"""
    PROPONENT = auto()      # 支持者
    OPPONENT = auto()       # 反对者
    NEUTRAL = auto()        # 中立/评估者
    MODERATOR = auto()      # 主持人
    EXPERT = auto()         # 专家


class DebatePhase(Enum):
    """辩论阶段"""
    OPENING = auto()        # 开场陈述
    ARGUMENT = auto()       # 论点陈述
    REBUTTAL = auto()       # 反驳
    CROSS_EXAMINATION = auto()  # 交叉质询
    CLOSING = auto()        # 总结陈词
    EVALUATION = auto()     # 评估投票
    RESOLUTION = auto()     # 决议


@dataclass
class Argument:
    """论点"""
    id: str
    debater_id: str
    phase: DebatePhase
    content: str
    evidence: List[str] = field(default_factory=list)
    
    # 论证强度
    strength: float = 0.5
    relevance: float = 0.5
    
    # 互动
    responds_to: Optional[str] = None  # 回应哪个论点
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "debater_id": self.debater_id,
            "phase": self.phase.name,
            "content": self.content[:100] + "..." if len(self.content) > 100 else self.content,
            "evidence_count": len(self.evidence),
            "strength": self.strength,
            "relevance": self.relevance,
            "responds_to": self.responds_to,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class DebateRound:
    """辩论轮次"""
    round_number: int
    phase: DebatePhase
    arguments: List[Argument] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_number": self.round_number,
            "phase": self.phase.name,
            "argument_count": len(self.arguments),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "arguments": [a.to_dict() for a in self.arguments]
        }


@dataclass
class DebateConfig:
    """辩论配置"""
    max_rounds: int = 3
    min_rounds: int = 2
    
    # 时间限制
    time_limit_per_statement: int = 300  # 秒
    
    # 参与要求
    min_participants: int = 2
    max_participants: int = 8
    
    # 投票
    voting_method: str = "weighted"  # simple, weighted, borda
    consensus_threshold: float = 0.6
    
    # 评估标准
    criteria: List[str] = field(default_factory=lambda: [
        "logical_soundness",
        "evidence_quality",
        "practical_feasibility",
        "innovation"
    ])


class DebateCollaboration:
    """
    辩论式协作工作流
    通过结构化辩论产生更优方案
    """
    
    def __init__(
        self,
        orchestrator: AgentOrchestrator,
        config: Optional[DebateConfig] = None
    ):
        self.orchestrator = orchestrator
        self.config = config or DebateConfig()
        
        # 参与者
        self.participants: Dict[str, Dict[str, Any]] = {}  # agent_id -> {role, stance}
        self.debaters: List[str] = []
        
        # 辩论状态
        self.topic: str = ""
        self.proposition: Optional[Dict[str, Any]] = None
        self.current_phase = DebatePhase.OPENING
        self.current_round = 0
        self.rounds: List[DebateRound] = []
        
        # 论点库
        self.arguments: Dict[str, Argument] = {}
        self.argument_tree: Dict[str, List[str]] = {}  # argument_id -> 回应列表
        
        # 共识管理
        self.consensus_manager = ConsensusManager(
            default_algorithm=MajorityConsensus(threshold=self.config.consensus_threshold)
        )
        
        # 结果
        self.final_outcome: Optional[Dict[str, Any]] = None
        self.votes: Dict[str, Any] = {}
        
        # 回调
        self.on_round_complete: Optional[Callable[[int, DebatePhase], None]] = None
        self.on_argument_made: Optional[Callable[[Argument], None]] = None
        self.on_debate_complete: Optional[Callable[[Dict], None]] = None
    
    def register_participant(
        self,
        agent_id: str,
        role: DebateRole,
        stance: str = "neutral",  # for, against, neutral
        weight: float = 1.0
    ) -> None:
        """注册参与者"""
        self.participants[agent_id] = {
            "role": role,
            "stance": stance,
            "weight": weight,
            "arguments_made": 0
        }
        
        if role in [DebateRole.PROPONENT, DebateRole.OPPONENT]:
            self.debaters.append(agent_id)
        
        # 注册到共识管理器
        self.consensus_manager.register_participant(agent_id, weight)
    
    async def conduct_debate(
        self,
        topic: str,
        proposition: Dict[str, Any],
        agent_arguments: Optional[Dict[str, Callable]] = None
    ) -> Dict[str, Any]:
        """
        进行辩论
        
        Args:
            topic: 辩论主题
            proposition: 待辩论的提案
            agent_arguments: {agent_id: argument_generation_function}
        """
        print(f"🎤 Starting Debate: {topic}")
        print(f"   Proposition: {proposition.get('statement', '')[:60]}...")
        print(f"   Participants: {len(self.participants)}")
        
        self.topic = topic
        self.proposition = proposition
        
        start_time = datetime.now()
        
        # 开场陈述
        await self._conduct_phase(DebatePhase.OPENING, agent_arguments)
        
        # 主要辩论轮次
        for round_num in range(1, self.config.max_rounds + 1):
            self.current_round = round_num
            print(f"\n📢 Round {round_num}")
            
            # 论点陈述
            await self._conduct_phase(DebatePhase.ARGUMENT, agent_arguments)
            
            # 反驳
            await self._conduct_phase(DebatePhase.REBUTTAL, agent_arguments)
            
            # 检查是否提前结束
            if round_num >= self.config.min_rounds:
                if self._check_convergence():
                    print("   ✅ Convergence reached, ending debate early")
                    break
        
        # 总结陈词
        await self._conduct_phase(DebatePhase.CLOSING, agent_arguments)
        
        # 评估投票
        await self._conduct_evaluation()
        
        # 生成决议
        outcome = await self._generate_resolution()
        
        total_duration = (datetime.now() - start_time).total_seconds()
        
        self.final_outcome = {
            "topic": topic,
            "proposition": proposition,
            "total_rounds": self.current_round,
            "total_arguments": len(self.arguments),
            "votes": self.votes,
            "outcome": outcome,
            "duration_seconds": total_duration,
            "rounds": [r.to_dict() for r in self.rounds],
            "participants": {
                aid: {
                    "role": p["role"].name,
                    "stance": p["stance"],
                    "arguments_made": p["arguments_made"]
                }
                for aid, p in self.participants.items()
            }
        }
        
        if self.on_debate_complete:
            self.on_debate_complete(self.final_outcome)
        
        print(f"\n🏁 Debate Completed")
        print(f"   Outcome: {outcome.get('decision')}")
        print(f"   Confidence: {outcome.get('confidence', 0):.2f}")
        
        return self.final_outcome
    
    async def _conduct_phase(
        self,
        phase: DebatePhase,
        agent_arguments: Optional[Dict[str, Callable]] = None
    ) -> None:
        """进行辩论阶段"""
        print(f"   Phase: {phase.name}")
        
        self.current_phase = phase
        
        round_obj = DebateRound(
            round_number=self.current_round,
            phase=phase
        )
        
        # 收集各参与者的陈述
        for agent_id, participant_info in self.participants.items():
            role = participant_info["role"]
            
            # 根据阶段确定哪些角色参与
            should_participate = self._should_participate(role, phase)
            
            if should_participate:
                # 生成论点
                argument = await self._generate_argument(
                    agent_id,
                    phase,
                    agent_arguments.get(agent_id) if agent_arguments else None
                )
                
                if argument:
                    round_obj.arguments.append(argument)
                    self.arguments[argument.id] = argument
                    participant_info["arguments_made"] += 1
                    
                    if self.on_argument_made:
                        self.on_argument_made(argument)
        
        round_obj.end_time = datetime.now()
        self.rounds.append(round_obj)
        
        if self.on_round_complete:
            self.on_round_complete(self.current_round, phase)
    
    def _should_participate(self, role: DebateRole, phase: DebatePhase) -> bool:
        """确定角色是否在当前阶段参与"""
        if phase == DebatePhase.OPENING:
            return role in [DebateRole.PROPONENT, DebateRole.OPPONENT, DebateRole.MODERATOR]
        elif phase == DebatePhase.ARGUMENT:
            return role in [DebateRole.PROPONENT, DebateRole.OPPONENT, DebateRole.EXPERT]
        elif phase == DebatePhase.REBUTTAL:
            return role in [DebateRole.PROPONENT, DebateRole.OPPONENT]
        elif phase == DebatePhase.CLOSING:
            return role in [DebateRole.PROPONENT, DebateRole.OPPONENT]
        return False
    
    async def _generate_argument(
        self,
        agent_id: str,
        phase: DebatePhase,
        argument_func: Optional[Callable] = None
    ) -> Optional[Argument]:
        """生成论点"""
        participant = self.participants[agent_id]
        
        # 使用提供的函数或默认生成
        if argument_func:
            content = argument_func(self.topic, phase, participant["stance"])
        else:
            content = self._default_argument_generation(
                agent_id, phase, participant["stance"]
            )
        
        if content:
            argument_id = f"arg_{uuid.uuid4().hex[:8]}"
            
            # 确定回应哪个论点（如果是反驳阶段）
            responds_to = None
            if phase == DebatePhase.REBUTTAL:
                responds_to = self._find_argument_to_rebut(agent_id, participant["stance"])
            
            argument = Argument(
                id=argument_id,
                debater_id=agent_id,
                phase=phase,
                content=content,
                evidence=self._generate_evidence(content),
                strength=self._evaluate_argument_strength(content),
                relevance=0.8,
                responds_to=responds_to
            )
            
            return argument
        
        return None
    
    def _default_argument_generation(
        self,
        agent_id: str,
        phase: DebatePhase,
        stance: str
    ) -> str:
        """默认论点生成"""
        templates = {
            DebatePhase.OPENING: {
                "for": f"I strongly support this proposition because it addresses key challenges...",
                "against": f"I must oppose this proposition due to significant concerns...",
                "neutral": f"Let me present a balanced view of this proposition..."
            },
            DebatePhase.ARGUMENT: {
                "for": f"The evidence strongly suggests that this approach will yield positive results...",
                "against": f"However, we must consider the potential risks and limitations...",
                "neutral": f"From an objective standpoint, there are both merits and drawbacks..."
            },
            DebatePhase.REBUTTAL: {
                "for": f"The counter-arguments fail to account for...",
                "against": f"The supporting arguments overlook critical issues...",
                "neutral": f"Both sides raise valid points, but..."
            },
            DebatePhase.CLOSING: {
                "for": f"In conclusion, the benefits clearly outweigh the risks...",
                "against": f"In summary, the proposed approach is not viable at this time...",
                "neutral": f"Ultimately, further investigation is needed before proceeding..."
            }
        }
        
        phase_templates = templates.get(phase, {})
        return phase_templates.get(stance, f"Statement from {agent_id}")
    
    def _generate_evidence(self, content: str) -> List[str]:
        """为论点生成证据"""
        # 简化实现
        return [
            f"Evidence supporting: {content[:30]}...",
            "Reference to prior research",
            "Logical deduction"
        ]
    
    def _evaluate_argument_strength(self, content: str) -> float:
        """评估论点强度"""
        # 基于内容长度、关键词等简单评估
        strength = 0.5
        
        # 长度因素
        if len(content) > 100:
            strength += 0.1
        
        # 关键词因素
        strong_indicators = ["evidence", "research", "data", "analysis", "prove"]
        for indicator in strong_indicators:
            if indicator in content.lower():
                strength += 0.05
        
        return min(strength, 1.0)
    
    def _find_argument_to_rebut(self, agent_id: str, stance: str) -> Optional[str]:
        """找到要反驳的论点"""
        # 找到对方立场的最近论点
        opponent_stance = "against" if stance == "for" else "for"
        
        for arg_id, arg in reversed(list(self.arguments.items())):
            if arg.debater_id != agent_id:
                arg_stance = self.participants.get(arg.debater_id, {}).get("stance")
                if arg_stance == opponent_stance:
                    return arg_id
        
        return None
    
    def _check_convergence(self) -> bool:
        """检查是否已收敛到共识"""
        if len(self.rounds) < 2:
            return False
        
        # 检查最近两轮的论点是否重复
        recent_args = self.rounds[-1].arguments
        previous_args = self.rounds[-2].arguments
        
        # 简化：检查论点数量是否显著减少
        return len(recent_args) < len(previous_args) * 0.5
    
    async def _conduct_evaluation(self) -> None:
        """进行评估投票"""
        print(f"   Phase: Evaluation")
        
        self.current_phase = DebatePhase.EVALUATION
        
        # 创建提案
        proposal_id = f"debate_proposal_{uuid.uuid4().hex[:8]}"
        
        proposal = self.consensus_manager.create_proposal(
            proposal_id=proposal_id,
            proposer_id="debate_system",
            content={
                "topic": self.topic,
                "proposition": self.proposition
            },
            description=f"Vote on proposition: {self.topic}",
            deadline_seconds=60
        )
        
        # 收集投票
        for agent_id in self.participants:
            # 基于Agent的立场和论点生成投票
            participant = self.participants[agent_id]
            
            if participant["stance"] == "for":
                vote = VoteType.APPROVE
            elif participant["stance"] == "against":
                vote = VoteType.REJECT
            else:
                # 中立：基于论点强度决定
                avg_strength = self._calculate_avg_argument_strength(agent_id)
                vote = VoteType.APPROVE if avg_strength > 0.6 else VoteType.REJECT
            
            await self.consensus_manager.submit_vote(
                proposal_id=proposal_id,
                voter_id=agent_id,
                vote_type=vote,
                reasoning=f"Based on arguments presented in debate"
            )
        
        # 等待结果
        await asyncio.sleep(1)
        
        # 获取投票结果
        self.votes = self.consensus_manager.get_proposal_status(proposal_id) or {}
    
    def _calculate_avg_argument_strength(self, agent_id: str) -> float:
        """计算Agent的平均论点强度"""
        agent_arguments = [
            arg for arg in self.arguments.values()
            if arg.debater_id == agent_id
        ]
        
        if not agent_arguments:
            return 0.5
        
        return sum(arg.strength for arg in agent_arguments) / len(agent_arguments)
    
    async def _generate_resolution(self) -> Dict[str, Any]:
        """生成最终决议"""
        print(f"   Phase: Resolution")
        
        self.current_phase = DebatePhase.RESOLUTION
        
        vote_summary = self.votes.get("vote_summary", {})
        approve_ratio = vote_summary.get("approval_ratio", 0)
        
        # 确定决议
        if approve_ratio >= self.config.consensus_threshold:
            decision = "accepted"
            confidence = approve_ratio
        elif approve_ratio <= (1 - self.config.consensus_threshold):
            decision = "rejected"
            confidence = 1 - approve_ratio
        else:
            decision = "needs_revision"
            confidence = max(approve_ratio, 1 - approve_ratio)
        
        # 提取关键论点
        key_arguments_for = [
            arg.to_dict() for arg in self.arguments.values()
            if self.participants.get(arg.debater_id, {}).get("stance") == "for"
            and arg.strength > 0.7
        ]
        
        key_arguments_against = [
            arg.to_dict() for arg in self.arguments.values()
            if self.participants.get(arg.debater_id, {}).get("stance") == "against"
            and arg.strength > 0.7
        ]
        
        # 生成改进建议（如果被拒绝或需要修改）
        recommendations = []
        if decision in ["rejected", "needs_revision"]:
            recommendations = self._generate_recommendations()
        
        return {
            "decision": decision,
            "confidence": confidence,
            "vote_summary": vote_summary,
            "key_arguments_for": key_arguments_for[:3],
            "key_arguments_against": key_arguments_against[:3],
            "recommendations": recommendations,
            "alternative_proposals": self._generate_alternatives()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于反对论点生成建议
        against_args = [
            arg for arg in self.arguments.values()
            if self.participants.get(arg.debater_id, {}).get("stance") == "against"
        ]
        
        for arg in against_args:
            if "risk" in arg.content.lower():
                recommendations.append("Address risk factors identified by opponents")
            if "cost" in arg.content.lower():
                recommendations.append("Provide detailed cost-benefit analysis")
            if "evidence" in arg.content.lower():
                recommendations.append("Gather additional supporting evidence")
        
        if not recommendations:
            recommendations.append("Consider conducting pilot study")
            recommendations.append("Seek additional expert opinions")
        
        return list(set(recommendations))  # 去重
    
    def _generate_alternatives(self) -> List[Dict[str, Any]]:
        """生成替代方案"""
        alternatives = []
        
        # 基于辩论内容生成替代方案
        alternatives.append({
            "type": "modified_proposal",
            "description": "Original proposal with additional safeguards",
            "rationale": "Addresses concerns raised during debate"
        })
        
        alternatives.append({
            "type": "phased_approach",
            "description": "Implement in smaller, reversible phases",
            "rationale": "Reduces risk while allowing evaluation"
        })
        
        return alternatives
    
    def get_debate_summary(self) -> Dict[str, Any]:
        """获取辩论摘要"""
        return {
            "topic": self.topic,
            "current_phase": self.current_phase.name,
            "rounds_completed": len(self.rounds),
            "total_arguments": len(self.arguments),
            "participants": len(self.participants),
            "debater_distribution": {
                "for": sum(1 for p in self.participants.values() if p["stance"] == "for"),
                "against": sum(1 for p in self.participants.values() if p["stance"] == "against"),
                "neutral": sum(1 for p in self.participants.values() if p["stance"] == "neutral")
            }
        }


class StructuredDebate(DebateCollaboration):
    """
    结构化辩论
    使用特定的辩论格式（如林肯-道格拉斯制）
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.speech_order: List[Tuple[str, DebatePhase, int]] = []  # (agent_id, phase, time_limit)
    
    def set_speech_order(self, order: List[Tuple[str, DebatePhase, int]]) -> None:
        """设置发言顺序"""
        self.speech_order = order
    
    async def _conduct_phase(
        self,
        phase: DebatePhase,
        agent_arguments: Optional[Dict[str, Callable]] = None
    ) -> None:
        """使用预定发言顺序"""
        if not self.speech_order:
            return await super()._conduct_phase(phase, agent_arguments)
        
        round_obj = DebateRound(
            round_number=self.current_round,
            phase=phase
        )
        
        # 按照预定顺序
        for agent_id, speech_phase, time_limit in self.speech_order:
            if speech_phase == phase and agent_id in self.participants:
                argument = await self._generate_argument(
                    agent_id, phase, agent_arguments.get(agent_id) if agent_arguments else None
                )
                
                if argument:
                    round_obj.arguments.append(argument)
                    self.arguments[argument.id] = argument
                    self.participants[agent_id]["arguments_made"] += 1
        
        round_obj.end_time = datetime.now()
        self.rounds.append(round_obj)


class ExpertPanelDebate(DebateCollaboration):
    """
    专家组辩论
    多个专家Agent提供专业意见
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expertise_areas: Dict[str, List[str]] = {}
    
    def register_expert(
        self,
        agent_id: str,
        expertise_areas: List[str],
        weight: float = 1.0
    ) -> None:
        """注册专家"""
        self.register_participant(agent_id, DebateRole.EXPERT, "neutral", weight)
        self.expertise_areas[agent_id] = expertise_areas
    
    async def _generate_argument(
        self,
        agent_id: str,
        phase: DebatePhase,
        argument_func: Optional[Callable] = None
    ) -> Optional[Argument]:
        """生成专家意见"""
        argument = await super()._generate_argument(agent_id, phase, argument_func)
        
        if argument:
            # 添加专家领域标签
            argument.evidence.extend([
                f"Expertise: {area}"
                for area in self.expertise_areas.get(agent_id, [])
            ])
        
        return argument
    
    async def _generate_resolution(self) -> Dict[str, Any]:
        """生成带权重的专家决议"""
        outcome = await super()._generate_resolution()
        
        # 按专业领域加权
        domain_scores: Dict[str, float] = {}
        
        for agent_id, areas in self.expertise_areas.items():
            participant = self.participants.get(agent_id, {})
            weight = participant.get("weight", 1.0)
            
            for area in areas:
                if area not in domain_scores:
                    domain_scores[area] = 0
                domain_scores[area] += weight
        
        outcome["expertise_weighting"] = domain_scores
        
        return outcome
