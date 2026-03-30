"""
Hierarchical Team - 层级团队工作流
实现PI→博士后→学生层级管理
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import uuid

from ..multi_agent.agent_core import Message, MessageType
from ..multi_agent.agent_orchestrator import AgentOrchestrator, Task


class AcademicRank(Enum):
    """学术职级"""
    PI = "pi"                      # 首席研究员
    SENIOR_RESEARCHER = "senior"   # 高级研究员
    POSTDOC = "postdoc"            # 博士后
    PHD_STUDENT = "phd"            # 博士生
    MASTER_STUDENT = "master"      # 硕士生
    UNDERGRAD = "undergrad"        # 本科生
    VISITOR = "visitor"            # 访问学者


class ApprovalStatus(Enum):
    """审批状态"""
    DRAFT = auto()
    PENDING_REVIEW = auto()
    REVISION_REQUESTED = auto()
    APPROVED = auto()
    REJECTED = auto()


@dataclass
class ResearchMilestone:
    """研究里程碑"""
    id: str
    name: str
    description: str
    deadline: datetime
    deliverables: List[str] = field(default_factory=list)
    
    # 负责
    assigned_to: str = ""
    supervised_by: str = ""
    
    # 状态
    status: str = "planned"  # planned, in_progress, review, completed
    progress: float = 0.0
    
    # 评审
    submissions: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "deadline": self.deadline.isoformat(),
            "assigned_to": self.assigned_to,
            "status": self.status,
            "progress": self.progress
        }


@dataclass
class WorkPackage:
    """工作包"""
    id: str
    name: str
    description: str
    
    # 层级
    owner: str = ""  # 负责人
    contributors: List[str] = field(default_factory=list)
    
    # 任务
    subtasks: List[str] = field(default_factory=list)
    
    # 审批链
    approval_chain: List[str] = field(default_factory=list)
    current_approver_index: int = 0
    approval_status: ApprovalStatus = ApprovalStatus.DRAFT
    
    # 元数据
    created_at: datetime = field(default_factory=datetime.now)
    priority: str = "normal"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "owner": self.owner,
            "contributors": self.contributors,
            "approval_status": self.approval_status.name,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class LabMember:
    """实验室成员"""
    agent_id: str
    name: str
    rank: AcademicRank
    
    # 汇报关系
    reports_to: Optional[str] = None
    supervises: List[str] = field(default_factory=list)
    
    # 能力
    expertise: List[str] = field(default_factory=list)
    current_workload: int = 0
    max_workload: int = 5
    
    # 状态
    status: str = "active"
    joined_date: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "rank": self.rank.value,
            "reports_to": self.reports_to,
            "expertise": self.expertise,
            "workload": f"{self.current_workload}/{self.max_workload}",
            "status": self.status
        }


class HierarchicalTeam:
    """
    层级研究团队
    模拟学术实验室的层级结构
    """
    
    def __init__(self, orchestrator: AgentOrchestrator):
        self.orchestrator = orchestrator
        
        # 成员
        self.members: Dict[str, LabMember] = {}
        self.pi: Optional[str] = None
        
        # 研究项目
        self.research_programs: Dict[str, Dict[str, Any]] = {}
        self.milestones: Dict[str, ResearchMilestone] = {}
        self.work_packages: Dict[str, WorkPackage] = {}
        
        # 组织结构
        self.teams: Dict[str, Set[str]] = {}  # team_name -> member_ids
        
        # 会议和报告
        self.meetings: List[Dict[str, Any]] = []
        self.reports: Dict[str, List[Dict[str, Any]]] = {}
    
    def add_member(
        self,
        agent_id: str,
        name: str,
        rank: AcademicRank,
        reports_to: Optional[str] = None,
        expertise: Optional[List[str]] = None
    ) -> LabMember:
        """添加实验室成员"""
        member = LabMember(
            agent_id=agent_id,
            name=name,
            rank=rank,
            reports_to=reports_to,
            expertise=expertise or []
        )
        
        self.members[agent_id] = member
        
        # 设置PI
        if rank == AcademicRank.PI:
            self.pi = agent_id
        
        # 更新汇报关系
        if reports_to and reports_to in self.members:
            self.members[reports_to].supervises.append(agent_id)
        
        # 注册到编排器
        self.orchestrator.register_agent(
            self._create_agent_proxy(agent_id),
            groups=[rank.value]
        )
        
        return member
    
    def _create_agent_proxy(self, agent_id: str):
        """创建Agent代理"""
        from ..multi_agent.agent_core import BaseAgent
        
        class AgentProxy(BaseAgent):
            async def perceive(self):
                return []
            
            async def reason(self, context):
                return {}
            
            async def act(self, decision):
                return []
        
        return AgentProxy(agent_id=agent_id, name=agent_id)
    
    def create_research_program(
        self,
        program_id: str,
        title: str,
        description: str,
        pi_id: Optional[str] = None,
        objectives: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """创建研究项目"""
        program = {
            "id": program_id,
            "title": title,
            "description": description,
            "pi": pi_id or self.pi,
            "objectives": objectives or [],
            "milestones": [],
            "start_date": datetime.now().isoformat(),
            "status": "active"
        }
        
        self.research_programs[program_id] = program
        return program
    
    def create_milestone(
        self,
        program_id: str,
        name: str,
        description: str,
        deadline: datetime,
        assigned_to: str,
        deliverables: Optional[List[str]] = None
    ) -> Optional[ResearchMilestone]:
        """创建研究里程碑"""
        if program_id not in self.research_programs:
            return None
        
        if assigned_to not in self.members:
            return None
        
        milestone_id = f"milestone_{uuid.uuid4().hex[:8]}"
        
        assigned_member = self.members[assigned_to]
        
        milestone = ResearchMilestone(
            id=milestone_id,
            name=name,
            description=description,
            deadline=deadline,
            deliverables=deliverables or [],
            assigned_to=assigned_to,
            supervised_by=assigned_member.reports_to or self.pi
        )
        
        self.milestones[milestone_id] = milestone
        self.research_programs[program_id]["milestones"].append(milestone_id)
        
        # 增加工作负载
        assigned_member.current_workload += 1
        
        return milestone
    
    def create_work_package(
        self,
        milestone_id: str,
        name: str,
        description: str,
        owner: str,
        contributors: Optional[List[str]] = None,
        priority: str = "normal"
    ) -> Optional[WorkPackage]:
        """创建工作包"""
        if milestone_id not in self.milestones:
            return None
        
        package_id = f"wp_{uuid.uuid4().hex[:8]}"
        
        # 构建审批链
        approval_chain = self._build_approval_chain(owner)
        
        work_package = WorkPackage(
            id=package_id,
            name=name,
            description=description,
            owner=owner,
            contributors=contributors or [],
            approval_chain=approval_chain,
            priority=priority
        )
        
        self.work_packages[package_id] = work_package
        
        return work_package
    
    def _build_approval_chain(self, owner_id: str) -> List[str]:
        """构建审批链"""
        chain = []
        current = owner_id
        
        # 向上追溯汇报链
        while current:
            chain.append(current)
            member = self.members.get(current)
            if member:
                current = member.reports_to
            else:
                break
        
        # 确保PI在最后
        if self.pi and self.pi not in chain:
            chain.append(self.pi)
        
        return list(reversed(chain))  # 从PI开始
    
    async def submit_for_approval(
        self,
        work_package_id: str,
        submission: Dict[str, Any],
        submitter_id: str
    ) -> bool:
        """提交工作包审批"""
        if work_package_id not in self.work_packages:
            return False
        
        wp = self.work_packages[work_package_id]
        
        submission_record = {
            "id": f"sub_{uuid.uuid4().hex[:8]}",
            "submitter": submitter_id,
            "content": submission,
            "submitted_at": datetime.now().isoformat(),
            "status": "pending"
        }
        
        wp.submissions.append(submission_record)
        wp.approval_status = ApprovalStatus.PENDING_REVIEW
        
        # 通知当前审批人
        if wp.approval_chain:
            current_approver = wp.approval_chain[wp.current_approver_index]
            await self._notify_approver(current_approver, work_package_id, submission_record)
        
        return True
    
    async def _notify_approver(
        self,
        approver_id: str,
        work_package_id: str,
        submission: Dict[str, Any]
    ) -> None:
        """通知审批人"""
        # 实际实现中应该发送消息
        print(f"📧 Notification: {approver_id} has a submission to review ({work_package_id})")
    
    async def review_submission(
        self,
        work_package_id: str,
        reviewer_id: str,
        decision: str,  # approve, reject, request_revision
        comments: Optional[str] = None
    ) -> bool:
        """评审提交"""
        if work_package_id not in self.work_packages:
            return False
        
        wp = self.work_packages[work_package_id]
        
        # 检查是否是当前审批人
        if wp.approval_chain[wp.current_approver_index] != reviewer_id:
            return False
        
        if decision == "approve":
            wp.current_approver_index += 1
            
            # 检查是否所有审批人都通过
            if wp.current_approver_index >= len(wp.approval_chain):
                wp.approval_status = ApprovalStatus.APPROVED
                await self._on_approval_complete(work_package_id)
            else:
                # 通知下一个审批人
                next_approver = wp.approval_chain[wp.current_approver_index]
                await self._notify_approver(next_approver, work_package_id, {})
        
        elif decision == "reject":
            wp.approval_status = ApprovalStatus.REJECTED
        
        elif decision == "request_revision":
            wp.approval_status = ApprovalStatus.REVISION_REQUESTED
        
        # 记录评审意见
        review_record = {
            "reviewer": reviewer_id,
            "decision": decision,
            "comments": comments,
            "timestamp": datetime.now().isoformat()
        }
        
        if wp.submissions:
            wp.submissions[-1]["reviews"] = wp.submissions[-1].get("reviews", []) + [review_record]
        
        return True
    
    async def _on_approval_complete(self, work_package_id: str) -> None:
        """审批完成处理"""
        print(f"✅ Work package {work_package_id} fully approved")
        
        # 更新里程碑进度
        for milestone in self.milestones.values():
            # 简化：假设工作包属于某个里程碑
            milestone.progress = min(100, milestone.progress + 20)
    
    def get_org_chart(self) -> Dict[str, Any]:
        """获取组织结构图"""
        chart = {
            "pi": None,
            "teams": {}
        }
        
        for member_id, member in self.members.items():
            if member.rank == AcademicRank.PI:
                chart["pi"] = member.to_dict()
            
            # 按汇报关系组织
            if member.reports_to:
                if member.reports_to not in chart["teams"]:
                    chart["teams"][member.reports_to] = []
                chart["teams"][member.reports_to].append(member.to_dict())
        
        return chart
    
    def get_workload_distribution(self) -> Dict[str, Any]:
        """获取工作负载分布"""
        distribution = {
            "by_rank": {},
            "by_member": {}
        }
        
        for member in self.members.values():
            rank = member.rank.value
            if rank not in distribution["by_rank"]:
                distribution["by_rank"][rank] = {
                    "total_capacity": 0,
                    "assigned_work": 0,
                    "members": 0
                }
            
            distribution["by_rank"][rank]["total_capacity"] += member.max_workload
            distribution["by_rank"][rank]["assigned_work"] += member.current_workload
            distribution["by_rank"][rank]["members"] += 1
            
            distribution["by_member"][member.agent_id] = {
                "name": member.name,
                "rank": rank,
                "workload": f"{member.current_workload}/{member.max_workload}",
                "utilization": member.current_workload / member.max_workload
            }
        
        return distribution
    
    async def schedule_meeting(
        self,
        meeting_type: str,  # group, individual, committee
        participants: List[str],
        agenda: List[str],
        duration_minutes: int = 60
    ) -> Dict[str, Any]:
        """安排会议"""
        meeting_id = f"meeting_{uuid.uuid4().hex[:8]}"
        
        meeting = {
            "id": meeting_id,
            "type": meeting_type,
            "participants": participants,
            "agenda": agenda,
            "scheduled_at": datetime.now().isoformat(),
            "duration_minutes": duration_minutes,
            "status": "scheduled"
        }
        
        self.meetings.append(meeting)
        
        # 通知参与者
        for participant in participants:
            if participant in self.members:
                print(f"📅 Meeting scheduled for {participant}: {meeting_type}")
        
        return meeting
    
    async def conduct_group_meeting(
        self,
        meeting_id: str,
        discussion_topics: List[str]
    ) -> Dict[str, Any]:
        """进行组会"""
        meeting = next((m for m in self.meetings if m["id"] == meeting_id), None)
        if not meeting:
            return {"error": "Meeting not found"}
        
        print(f"\n🗣️  Group Meeting: {meeting['type']}")
        print(f"   Agenda: {meeting['agenda']}")
        
        meeting_minutes = {
            "meeting_id": meeting_id,
            "start_time": datetime.now().isoformat(),
            "attendees": meeting["participants"],
            "discussions": []
        }
        
        for topic in discussion_topics:
            print(f"\n   Topic: {topic}")
            
            # 收集各方汇报
            progress_updates = []
            for participant in meeting["participants"]:
                if participant in self.members:
                    member = self.members[participant]
                    
                    # 查找该成员负责的任务
                    assigned_milestones = [
                        m for m in self.milestones.values()
                        if m.assigned_to == participant
                    ]
                    
                    update = {
                        "member": member.name,
                        "rank": member.rank.value,
                        "milestones": [
                            {"name": m.name, "progress": m.progress, "status": m.status}
                            for m in assigned_milestones
                        ]
                    }
                    progress_updates.append(update)
                    
                    print(f"      {member.name} ({member.rank.value}): {len(assigned_milestones)} milestones")
            
            meeting_minutes["discussions"].append({
                "topic": topic,
                "progress_updates": progress_updates
            })
        
        meeting_minutes["end_time"] = datetime.now().isoformat()
        meeting["status"] = "completed"
        meeting["minutes"] = meeting_minutes
        
        return meeting_minutes
    
    def generate_progress_report(
        self,
        program_id: Optional[str] = None,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """生成进展报告"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "period": {
                "start": period_start.isoformat() if period_start else None,
                "end": period_end.isoformat() if period_end else None
            },
            "summary": {},
            "programs": [],
            "milestones": [],
            "team_stats": self.get_workload_distribution()
        }
        
        # 汇总项目进展
        for pid, program in self.research_programs.items():
            if program_id and pid != program_id:
                continue
            
            program_milestones = [
                self.milestones[mid] for mid in program.get("milestones", [])
                if mid in self.milestones
            ]
            
            completed = sum(1 for m in program_milestones if m.status == "completed")
            total = len(program_milestones)
            
            program_summary = {
                "id": pid,
                "title": program["title"],
                "milestone_progress": f"{completed}/{total}",
                "completion_rate": completed / total if total > 0 else 0
            }
            
            report["programs"].append(program_summary)
            
            # 里程碑详情
            for m in program_milestones:
                report["milestones"].append({
                    "id": m.id,
                    "name": m.name,
                    "program": program["title"],
                    "assigned_to": self.members.get(m.assigned_to, LabMember("", "Unknown", AcademicRank.VISITOR)).name,
                    "status": m.status,
                    "progress": m.progress,
                    "deadline": m.deadline.isoformat()
                })
        
        # 总体统计
        total_milestones = len(self.milestones)
        completed_milestones = sum(1 for m in self.milestones.values() if m.status == "completed")
        
        report["summary"] = {
            "total_programs": len(self.research_programs),
            "total_milestones": total_milestones,
            "completed_milestones": completed_milestones,
            "overall_completion": completed_milestones / total_milestones if total_milestones > 0 else 0,
            "team_size": len(self.members)
        }
        
        return report


class MentorshipSystem:
    """
    导师制系统
    管理PI-学生/博士后指导关系
    """
    
    def __init__(self, team: HierarchicalTeam):
        self.team = team
        self.mentorships: Dict[str, Dict[str, Any]] = {}  # mentee_id -> mentorship_info
        self.training_plans: Dict[str, List[Dict[str, Any]]] = {}
    
    def establish_mentorship(
        self,
        mentor_id: str,
        mentee_id: str,
        focus_areas: List[str]
    ) -> bool:
        """建立指导关系"""
        if mentor_id not in self.team.members or mentee_id not in self.team.members:
            return False
        
        self.mentorships[mentee_id] = {
            "mentor": mentor_id,
            "mentee": mentee_id,
            "established_at": datetime.now().isoformat(),
            "focus_areas": focus_areas,
            "meetings": [],
            "goals": []
        }
        
        return True
    
    def create_training_plan(
        self,
        mentee_id: str,
        skills: List[str],
        timeline_months: int = 12
    ) -> List[Dict[str, Any]]:
        """创建培训计划"""
        plan = []
        
        for i, skill in enumerate(skills):
            milestone = {
                "skill": skill,
                "target_month": (i + 1) * (timeline_months // len(skills)),
                "status": "planned",
                "evidence_required": f"Demonstrate competency in {skill}"
            }
            plan.append(milestone)
        
        self.training_plans[mentee_id] = plan
        return plan
    
    async def conduct_mentorship_meeting(
        self,
        mentee_id: str,
        meeting_type: str = "regular"
    ) -> Dict[str, Any]:
        """进行指导会议"""
        if mentee_id not in self.mentorships:
            return {"error": "No mentorship found"}
        
        mentorship = self.mentorships[mentee_id]
        mentor_id = mentorship["mentor"]
        
        meeting = {
            "type": meeting_type,
            "date": datetime.now().isoformat(),
            "mentor": self.team.members.get(mentor_id, LabMember("", "Unknown", AcademicRank.VISITOR)).name,
            "mentee": self.team.members.get(mentee_id, LabMember("", "Unknown", AcademicRank.VISITOR)).name,
            "topics_discussed": [],
            "action_items": [],
            "feedback": ""
        }
        
        if meeting_type == "progress_review":
            # 检查培训计划进展
            plan = self.training_plans.get(mentee_id, [])
            
            meeting["training_progress"] = [
                {"skill": p["skill"], "status": p["status"]}
                for p in plan
            ]
        
        mentorship["meetings"].append(meeting)
        
        return meeting
