#!/usr/bin/env python3
"""
Distributed Collaborative Discovery
分布式协同发现系统 - 全球科研协作平台

Features:
- Multi-institution collaboration
- Knowledge graph construction
- Discovery workflow orchestration
- Result verification and reproducibility
- Intellectual property management
- Real-time collaboration tools

Author: DFTLammps Platform
Version: 2.0.0
"""

import asyncio
import json
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable, Any, Union, Set, Tuple
from enum import Enum, auto
from collections import defaultdict
from datetime import datetime, timedelta
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InstitutionType(Enum):
    """机构类型"""
    UNIVERSITY = "university"
    RESEARCH_LAB = "research_lab"
    INDUSTRY = "industry"
    GOVERNMENT = "government"
    STARTUP = "startup"


class CollaborationRole(Enum):
    """协作角色"""
    LEAD = "lead"              # 主导方
    CONTRIBUTOR = "contributor"  # 贡献方
    REVIEWER = "reviewer"      # 审核方
    OBSERVER = "observer"      # 观察方


class DiscoveryTaskType(Enum):
    """发现任务类型"""
    SCREENING = "screening"          # 高通量筛选
    OPTIMIZATION = "optimization"    # 结构优化
    PROPERTY_PREDICTION = "property_prediction"  # 性质预测
    SYNTHESIS_PLANNING = "synthesis_planning"    # 合成规划
    VALIDATION = "validation"        # 实验验证


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REVIEW_PENDING = "review_pending"
    APPROVED = "approved"


class IPProtectionLevel(Enum):
    """知识产权保护级别"""
    PUBLIC = "public"              # 完全公开
    ACADEMIC = "academic"          # 学术共享
    COLLABORATION = "collaboration"  # 协作内共享
    PROPRIETARY = "proprietary"    # 专有
    RESTRICTED = "restricted"      # 严格受限


@dataclass
class Institution:
    """机构信息"""
    institution_id: str
    name: str
    institution_type: InstitutionType
    country: str
    region: str
    
    # 能力
    specializations: List[str] = field(default_factory=list)
    available_resources: List[str] = field(default_factory=list)
    
    # 联系信息
    principal_investigator: str = ""
    contact_email: str = ""
    
    # 声誉指标
    h_index: float = 0.0
    citation_count: int = 0
    
    def __post_init__(self):
        if not self.institution_id:
            self.institution_id = hashlib.md5(
                f"{self.name}-{time.time()}".encode()
            ).hexdigest()[:16]


@dataclass
class Researcher:
    """研究人员"""
    researcher_id: str
    name: str
    email: str
    institution_id: str
    
    expertise: List[str] = field(default_factory=list)
    publications: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.researcher_id:
            self.researcher_id = hashlib.md5(
                f"{self.email}-{time.time()}".encode()
            ).hexdigest()[:16]


@dataclass
class CollaborationAgreement:
    """协作协议"""
    agreement_id: str
    project_name: str
    description: str
    
    # 参与方
    lead_institution: Institution
    partner_institutions: List[Tuple[Institution, CollaborationRole]]
    
    # 知识产权条款
    ip_protection: IPProtectionLevel
    ip_sharing_terms: str = ""
    commercial_terms: str = ""
    
    # 数据治理
    data_sharing_policy: str = ""
    publication_policy: str = ""
    
    # 时间
    start_date: datetime = field(default_factory=datetime.utcnow)
    end_date: Optional[datetime] = None
    
    # 状态
    signed_by: List[str] = field(default_factory=list)
    status: str = "draft"  # draft, active, completed, terminated


@dataclass
class DiscoveryTask:
    """发现任务"""
    task_id: str
    task_type: DiscoveryTaskType
    description: str
    
    # 任务输入
    input_data: Dict[str, Any] = field(default_factory=dict)
    target_criteria: Dict[str, Any] = field(default_factory=dict)
    
    # 分配
    assigned_to: Optional[str] = None  # institution_id
    assigned_researcher: Optional[str] = None
    
    # 状态
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 5  # 1-10
    
    # 输出
    results: List[Dict] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)  # 文件/数据引用
    
    # 时间
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # 元数据
    dependencies: List[str] = field(default_factory=list)  # task_ids
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = hashlib.md5(
                f"{self.task_type.value}-{time.time()}".encode()
            ).hexdigest()[:16]


@dataclass
class KnowledgeNode:
    """知识图谱节点"""
    node_id: str
    node_type: str  # material, property, method, researcher, etc.
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # 来源
    source_task: Optional[str] = None
    confidence: float = 1.0
    verified: bool = False


@dataclass
class KnowledgeEdge:
    """知识图谱边"""
    edge_id: str
    source_id: str
    target_id: str
    relation_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


class KnowledgeGraph:
    """知识图谱"""
    
    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: Dict[str, KnowledgeEdge] = {}
        self.adjacency: Dict[str, List[str]] = defaultdict(list)
    
    def add_node(self, node: KnowledgeNode) -> str:
        """添加节点"""
        if not node.node_id:
            node.node_id = hashlib.md5(
                f"{node.node_type}-{node.name}-{time.time()}".encode()
            ).hexdigest()[:16]
        
        self.nodes[node.node_id] = node
        return node.node_id
    
    def add_edge(self, edge: KnowledgeEdge) -> str:
        """添加边"""
        if not edge.edge_id:
            edge.edge_id = hashlib.md5(
                f"{edge.source_id}-{edge.target_id}-{edge.relation_type}".encode()
            ).hexdigest()[:16]
        
        self.edges[edge.edge_id] = edge
        self.adjacency[edge.source_id].append(edge.target_id)
        return edge.edge_id
    
    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5
    ) -> List[List[str]]:
        """查找节点间的路径"""
        paths = []
        visited = set()
        
        def dfs(current: str, path: List[str], depth: int):
            if depth > max_depth:
                return
            
            if current == target_id:
                paths.append(path[:])
                return
            
            visited.add(current)
            
            for neighbor in self.adjacency[current]:
                if neighbor not in visited:
                    path.append(neighbor)
                    dfs(neighbor, path, depth + 1)
                    path.pop()
            
            visited.remove(current)
        
        dfs(source_id, [source_id], 0)
        return paths
    
    def find_similar(self, node_id: str, relation_type: Optional[str] = None) -> List[KnowledgeNode]:
        """查找相似节点"""
        if node_id not in self.nodes:
            return []
        
        similar = []
        for edge in self.edges.values():
            if edge.source_id == node_id:
                if relation_type is None or edge.relation_type == relation_type:
                    target = self.nodes.get(edge.target_id)
                    if target:
                        similar.append(target)
        
        return similar
    
    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": defaultdict(int),
            "relation_types": defaultdict(int)
        }


class TaskScheduler:
    """任务调度器"""
    
    def __init__(self):
        self.pending_tasks: List[DiscoveryTask] = []
        self.active_tasks: Dict[str, DiscoveryTask] = {}
        self.completed_tasks: Dict[str, DiscoveryTask] = {}
        
        self.institution_capabilities: Dict[str, List[DiscoveryTaskType]] = defaultdict(list)
        self.researcher_workloads: Dict[str, int] = defaultdict(int)
    
    def register_institution_capabilities(
        self,
        institution_id: str,
        capabilities: List[DiscoveryTaskType]
    ):
        """注册机构能力"""
        self.institution_capabilities[institution_id] = capabilities
    
    def submit_task(self, task: DiscoveryTask) -> str:
        """提交任务"""
        self.pending_tasks.append(task)
        self.pending_tasks.sort(key=lambda t: -t.priority)
        logger.info(f"Task submitted: {task.task_id} ({task.task_type.value})")
        return task.task_id
    
    async def schedule_tasks(self):
        """调度待处理任务"""
        to_schedule = []
        
        for task in self.pending_tasks[:]:
            # 检查依赖是否完成
            if all(
                dep in self.completed_tasks
                for dep in task.dependencies
            ):
                # 匹配合适的机构
                assigned = self._match_institution(task)
                if assigned:
                    task.assigned_to = assigned
                    task.status = TaskStatus.ASSIGNED
                    to_schedule.append(task)
                    self.pending_tasks.remove(task)
        
        return to_schedule
    
    def _match_institution(self, task: DiscoveryTask) -> Optional[str]:
        """匹配合适的机构"""
        candidates = []
        
        for inst_id, capabilities in self.institution_capabilities.items():
            if task.task_type in capabilities:
                # 计算负载分数
                workload = self.researcher_workloads.get(inst_id, 0)
                candidates.append((inst_id, workload))
        
        if not candidates:
            return None
        
        # 选择负载最低的
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
    
    async def start_task(self, task_id: str) -> bool:
        """开始任务"""
        task = next(
            (t for t in self.pending_tasks if t.task_id == task_id),
            None
        )
        
        if task:
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.utcnow()
            self.active_tasks[task_id] = task
            if task in self.pending_tasks:
                self.pending_tasks.remove(task)
            return True
        
        return False
    
    async def complete_task(self, task_id: str, results: List[Dict]) -> bool:
        """完成任务"""
        task = self.active_tasks.get(task_id)
        if not task:
            return False
        
        task.results = results
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.utcnow()
        
        del self.active_tasks[task_id]
        self.completed_tasks[task_id] = task
        
        logger.info(f"Task completed: {task_id}")
        return True
    
    def get_queue_stats(self) -> Dict:
        """获取队列统计"""
        return {
            "pending": len(self.pending_tasks),
            "active": len(self.active_tasks),
            "completed": len(self.completed_tasks),
            "by_type": defaultdict(int)
        }


class ResultVerifier:
    """结果验证器"""
    
    def __init__(self):
        self.verification_history: List[Dict] = []
        self.reproducibility_scores: Dict[str, float] = {}
    
    async def verify_result(
        self,
        task: DiscoveryTask,
        verification_method: str = "cross_institution"
    ) -> Dict:
        """验证结果"""
        
        verification_result = {
            "task_id": task.task_id,
            "method": verification_method,
            "timestamp": datetime.utcnow().isoformat(),
            "verified": False,
            "confidence": 0.0,
            "details": []
        }
        
        if verification_method == "cross_institution":
            # 跨机构验证
            verification_result["verified"] = await self._cross_institution_verify(task)
            verification_result["confidence"] = 0.85 if verification_result["verified"] else 0.3
        
        elif verification_method == "reproducibility":
            # 可重复性验证
            verification_result["verified"] = await self._reproducibility_check(task)
            verification_result["confidence"] = 0.9 if verification_result["verified"] else 0.4
        
        self.verification_history.append(verification_result)
        return verification_result
    
    async def _cross_institution_verify(self, task: DiscoveryTask) -> bool:
        """跨机构验证 (模拟)"""
        # 实际实现会提交给不同机构独立验证
        await asyncio.sleep(0.5)
        return len(task.results) > 0 and random.random() > 0.1
    
    async def _reproducibility_check(self, task: DiscoveryTask) -> bool:
        """可重复性检查 (模拟)"""
        await asyncio.sleep(0.5)
        return random.random() > 0.15


class CollaborationOrchestrator:
    """协作编排器"""
    
    def __init__(self):
        self.institutions: Dict[str, Institution] = {}
        self.researchers: Dict[str, Researcher] = {}
        self.agreements: Dict[str, CollaborationAgreement] = {}
        
        self.knowledge_graph = KnowledgeGraph()
        self.task_scheduler = TaskScheduler()
        self.result_verifier = ResultVerifier()
        
        self._running = False
    
    def register_institution(self, institution: Institution):
        """注册机构"""
        self.institutions[institution.institution_id] = institution
        logger.info(f"Registered institution: {institution.name}")
    
    def register_researcher(self, researcher: Researcher):
        """注册研究人员"""
        self.researchers[researcher.researcher_id] = researcher
    
    def create_collaboration(
        self,
        project_name: str,
        lead_institution: Institution,
        partners: List[Tuple[Institution, CollaborationRole]],
        ip_protection: IPProtectionLevel
    ) -> str:
        """创建协作项目"""
        agreement = CollaborationAgreement(
            agreement_id="",
            project_name=project_name,
            description="",
            lead_institution=lead_institution,
            partner_institutions=partners,
            ip_protection=ip_protection
        )
        
        self.agreements[agreement.agreement_id] = agreement
        logger.info(f"Created collaboration: {project_name}")
        return agreement.agreement_id
    
    async def submit_discovery_task(
        self,
        agreement_id: str,
        task_type: DiscoveryTaskType,
        description: str,
        input_data: Dict,
        priority: int = 5
    ) -> str:
        """提交发现任务"""
        task = DiscoveryTask(
            task_id="",
            task_type=task_type,
            description=description,
            input_data=input_data,
            priority=priority
        )
        
        return self.task_scheduler.submit_task(task)
    
    async def run_discovery_cycle(self):
        """运行发现周期"""
        # 调度任务
        scheduled = await self.task_scheduler.schedule_tasks()
        
        for task in scheduled:
            # 开始任务
            await self.task_scheduler.start_task(task.task_id)
            
            # 模拟执行
            await self._execute_task(task)
        
        return len(scheduled)
    
    async def _execute_task(self, task: DiscoveryTask):
        """执行任务 (模拟)"""
        logger.info(f"Executing task: {task.task_id}")
        
        # 模拟计算时间
        await asyncio.sleep(random.uniform(0.5, 2.0))
        
        # 生成模拟结果
        results = []
        
        if task.task_type == DiscoveryTaskType.SCREENING:
            results = [
                {"material_id": f"MAT-{i}", "score": random.random()}
                for i in range(10)
            ]
        
        elif task.task_type == DiscoveryTaskType.PROPERTY_PREDICTION:
            results = [
                {"property": "band_gap", "value": random.uniform(0.5, 5.0), "unit": "eV"}
            ]
        
        elif task.task_type == DiscoveryTaskType.OPTIMIZATION:
            results = [
                {"optimized_structure": "...", "energy": random.uniform(-100, -50)}
            ]
        
        # 完成任务
        await self.task_scheduler.complete_task(task.task_id, results)
        
        # 验证结果
        verification = await self.result_verifier.verify_result(task)
        
        # 更新知识图谱
        await self._update_knowledge_graph(task, results, verification)
    
    async def _update_knowledge_graph(
        self,
        task: DiscoveryTask,
        results: List[Dict],
        verification: Dict
    ):
        """更新知识图谱"""
        # 创建任务节点
        task_node = KnowledgeNode(
            node_id="",
            node_type="task",
            name=task.task_id,
            properties={
                "task_type": task.task_type.value,
                "verified": verification["verified"],
                "confidence": verification["confidence"]
            },
            source_task=task.task_id
        )
        
        task_node_id = self.knowledge_graph.add_node(task_node)
        
        # 创建结果节点
        for result in results:
            if "material_id" in result:
                material_node = KnowledgeNode(
                    node_id="",
                    node_type="material",
                    name=result["material_id"],
                    properties=result,
                    source_task=task.task_id,
                    confidence=verification["confidence"],
                    verified=verification["verified"]
                )
                
                material_node_id = self.knowledge_graph.add_node(material_node)
                
                # 创建关系
                self.knowledge_graph.add_edge(KnowledgeEdge(
                    edge_id="",
                    source_id=task_node_id,
                    target_id=material_node_id,
                    relation_type="discovered"
                ))
    
    def query_knowledge(
        self,
        query_type: str,
        params: Dict
    ) -> List[KnowledgeNode]:
        """查询知识图谱"""
        results = []
        
        if query_type == "by_type":
            node_type = params.get("type")
            results = [
                node for node in self.knowledge_graph.nodes.values()
                if node.node_type == node_type
            ]
        
        elif query_type == "similar":
            node_id = params.get("node_id")
            results = self.knowledge_graph.find_similar(node_id)
        
        return results
    
    def get_collaboration_stats(self) -> Dict:
        """获取协作统计"""
        return {
            "institutions": len(self.institutions),
            "researchers": len(self.researchers),
            "active_collaborations": len([
                a for a in self.agreements.values()
                if a.status == "active"
            ]),
            "knowledge_graph": self.knowledge_graph.get_stats(),
            "task_queue": self.task_scheduler.get_queue_stats(),
            "verifications": len(self.result_verifier.verification_history)
        }


# 示例使用
async def demo():
    """分布式协同发现演示"""
    
    orchestrator = CollaborationOrchestrator()
    
    # 注册机构
    print("=== 注册参与机构 ===")
    
    institutions = [
        Institution(
            institution_id="",
            name="MIT Materials Science",
            institution_type=InstitutionType.UNIVERSITY,
            country="USA",
            region="North America",
            specializations=["battery_materials", "computational_chemistry"],
            available_resources=["HPC", "experimental_facilities"]
        ),
        Institution(
            institution_id="",
            name="Max Planck Institute",
            institution_type=InstitutionType.RESEARCH_LAB,
            country="Germany",
            region="Europe",
            specializations=["catalysis", "surface_science"],
            available_resources=["synchrotron", "microscopy"]
        ),
        Institution(
            institution_id="",
            name="Tokyo Tech",
            institution_type=InstitutionType.UNIVERSITY,
            country="Japan",
            region="Asia Pacific",
            specializations=["semiconductors", "nanomaterials"],
            available_resources=["cleanroom", "simulation_cluster"]
        ),
    ]
    
    for inst in institutions:
        orchestrator.register_institution(inst)
        orchestrator.task_scheduler.register_institution_capabilities(
            inst.institution_id,
            [DiscoveryTaskType.SCREENING, DiscoveryTaskType.OPTIMIZATION]
        )
        print(f"  Registered: {inst.name} ({inst.country})")
    
    # 创建协作项目
    print("\n=== 创建协作项目 ===")
    
    agreement_id = orchestrator.create_collaboration(
        project_name="Global Battery Materials Discovery",
        lead_institution=institutions[0],
        partners=[
            (institutions[1], CollaborationRole.CONTRIBUTOR),
            (institutions[2], CollaborationRole.CONTRIBUTOR)
        ],
        ip_protection=IPProtectionLevel.ACADEMIC
    )
    
    print(f"  Created collaboration: Global Battery Materials Discovery")
    print(f"    Agreement ID: {agreement_id[:8]}...")
    print(f"    Lead: {institutions[0].name}")
    print(f"    Partners: {len(institutions) - 1}")
    print(f"    IP Protection: Academic")
    
    # 提交发现任务
    print("\n=== 提交发现任务 ===")
    
    tasks = [
        {
            "type": DiscoveryTaskType.SCREENING,
            "description": "Screen Li-Mn-O compounds for cathode materials",
            "input_data": {"elements": ["Li", "Mn", "O"], "structure_types": ["layered", "spinel"]},
            "priority": 9
        },
        {
            "type": DiscoveryTaskType.PROPERTY_PREDICTION,
            "description": "Predict voltage profiles of screened materials",
            "input_data": {"target_properties": ["voltage", "capacity", "stability"]},
            "priority": 8
        },
        {
            "type": DiscoveryTaskType.OPTIMIZATION,
            "description": "Optimize top 5 candidates",
            "input_data": {"candidates": [], "optimization_target": "energy_density"},
            "priority": 7,
            "dependencies": []  # 将在后面设置
        },
    ]
    
    task_ids = []
    for task_data in tasks:
        tid = await orchestrator.submit_discovery_task(
            agreement_id,
            task_data["type"],
            task_data["description"],
            task_data["input_data"],
            task_data["priority"]
        )
        task_ids.append(tid)
        print(f"  Submitted: {task_data['type'].value} ({tid[:8]}...)")
    
    # 运行发现周期
    print("\n=== 执行发现周期 ===")
    
    for cycle in range(3):
        scheduled = await orchestrator.run_discovery_cycle()
        print(f"  Cycle {cycle + 1}: {scheduled} tasks executed")
        await asyncio.sleep(0.5)
    
    # 查看知识图谱
    print("\n=== 知识图谱状态 ===")
    kg_stats = orchestrator.knowledge_graph.get_stats()
    print(f"  Total nodes: {kg_stats['total_nodes']}")
    print(f"  Total edges: {kg_stats['total_edges']}")
    
    # 查询发现
    print("\n=== 查询发现结果 ===")
    
    materials = orchestrator.query_knowledge("by_type", {"type": "material"})
    print(f"  Discovered materials: {len(materials)}")
    
    for mat in materials[:5]:
        print(f"    - {mat.name} (confidence: {mat.confidence:.2f}, verified: {mat.verified})")
    
    # 验证统计
    print("\n=== 验证统计 ===")
    print(f"  Total verifications: {len(orchestrator.result_verifier.verification_history)}")
    
    verified_count = sum(
        1 for v in orchestrator.result_verifier.verification_history
        if v["verified"]
    )
    print(f"  Verified: {verified_count}")
    
    # 协作统计
    print("\n=== 协作统计 ===")
    stats = orchestrator.get_collaboration_stats()
    print(f"  Institutions: {stats['institutions']}")
    print(f"  Active collaborations: {stats['active_collaborations']}")
    print(f"  Completed tasks: {stats['task_queue']['completed']}")


if __name__ == "__main__":
    import time
    asyncio.run(demo())
