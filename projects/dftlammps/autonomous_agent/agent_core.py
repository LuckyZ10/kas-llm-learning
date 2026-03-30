"""
自主智能体核心模块
=================
实现AI自主材料科学家的核心功能，包括目标理解、规划、执行和反思。
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
from collections import deque
import uuid
import numpy as np


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    ABORTED = auto()
    WAITING_FOR_DEPENDENCY = auto()


class AgentState(Enum):
    """智能体状态枚举"""
    IDLE = auto()
    PLANNING = auto()
    EXECUTING = auto()
    REFLECTING = auto()
    LEARNING = auto()
    ERROR = auto()


@dataclass
class Task:
    """任务数据类"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 5  # 1-10, 10 being highest
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    subtasks: List['Task'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error_message: Optional[str] = None
    max_retries: int = 3
    current_retry: int = 0
    timeout_seconds: int = 3600
    required_tools: List[str] = field(default_factory=list)
    estimated_cost: float = 0.0  # 计算资源成本估计


@dataclass
class Goal:
    """目标数据类"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    criteria: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[datetime] = None
    priority: int = 5
    parent_goal_id: Optional[str] = None
    subgoals: List['Goal'] = field(default_factory=list)
    achievement_threshold: float = 0.8  # 目标达成阈值
    current_progress: float = 0.0
    status: TaskStatus = TaskStatus.PENDING


@dataclass
class Reflection:
    """反思记录"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    context: str = ""
    observation: str = ""
    evaluation: str = ""
    insights: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)


@dataclass
class ToolCall:
    """工具调用记录"""
    tool_name: str
    parameters: Dict[str, Any]
    result: Any = None
    success: bool = False
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None


class BaseTool(ABC):
    """工具基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.usage_count = 0
        self.success_count = 0
        
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """执行工具"""
        pass
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """获取使用统计"""
        return {
            "name": self.name,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "success_rate": self.success_count / self.usage_count if self.usage_count > 0 else 0
        }


class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._tool_schemas: Dict[str, Dict] = {}
        
    def register(self, tool: BaseTool, schema: Optional[Dict] = None):
        """注册工具"""
        self._tools[tool.name] = tool
        self._tool_schemas[tool.name] = schema or {}
        
    def get(self, name: str) -> Optional[BaseTool]:
        """获取工具"""
        return self._tools.get(name)
        
    def list_tools(self) -> List[str]:
        """列出所有工具"""
        return list(self._tools.keys())
        
    def get_schema(self, name: str) -> Dict:
        """获取工具模式"""
        return self._tool_schemas.get(name, {})
        
    def search_tools(self, description: str) -> List[str]:
        """根据描述搜索工具"""
        matches = []
        for name, tool in self._tools.items():
            if description.lower() in tool.description.lower():
                matches.append(name)
        return matches


class Memory:
    """智能体内存系统"""
    
    def __init__(self, max_short_term: int = 100):
        self.short_term: deque = deque(maxlen=max_short_term)
        self.long_term: Dict[str, Any] = {}
        self.episodic: List[Dict] = []  # 情节记忆（经历）
        self.semantic: Dict[str, Any] = {}  # 语义记忆（知识）
        
    def add_to_short_term(self, item: Any):
        """添加到短期记忆"""
        self.short_term.append({
            "content": item,
            "timestamp": datetime.now()
        })
        
    def get_short_term(self, n: int = 10) -> List[Any]:
        """获取最近的短期记忆"""
        return [item["content"] for item in list(self.short_term)[-n:]]
        
    def store_long_term(self, key: str, value: Any, category: str = "general"):
        """存储到长期记忆"""
        if category not in self.long_term:
            self.long_term[category] = {}
        self.long_term[category][key] = {
            "value": value,
            "timestamp": datetime.now(),
            "access_count": 0
        }
        
    def retrieve_long_term(self, key: str, category: str = "general") -> Optional[Any]:
        """从长期记忆检索"""
        if category in self.long_term and key in self.long_term[category]:
            self.long_term[category][key]["access_count"] += 1
            return self.long_term[category][key]["value"]
        return None
        
    def add_episode(self, episode: Dict):
        """添加情节记忆"""
        episode["timestamp"] = datetime.now()
        self.episodic.append(episode)
        
    def search_episodes(self, query: str) -> List[Dict]:
        """搜索情节记忆"""
        results = []
        for episode in self.episodic:
            if any(query.lower() in str(v).lower() for v in episode.values()):
                results.append(episode)
        return results
        
    def add_knowledge(self, concept: str, knowledge: Any):
        """添加语义知识"""
        self.semantic[concept] = {
            "knowledge": knowledge,
            "timestamp": datetime.now()
        }
        
    def get_knowledge(self, concept: str) -> Optional[Any]:
        """获取语义知识"""
        if concept in self.semantic:
            return self.semantic[concept]["knowledge"]
        return None


class GoalDecomposer:
    """目标分解器"""
    
    def __init__(self):
        self.decomposition_patterns: Dict[str, Callable] = {}
        
    def register_pattern(self, pattern_name: str, pattern_func: Callable):
        """注册分解模式"""
        self.decomposition_patterns[pattern_name] = pattern_func
        
    def decompose(self, goal: Goal) -> List[Task]:
        """分解目标为任务"""
        # 分析目标类型并选择适当的分解策略
        tasks = []
        
        # 1. 信息收集任务
        info_task = Task(
            name=f"收集_{goal.id}_信息",
            description=f"收集目标 '{goal.description}' 所需的信息",
            priority=goal.priority,
            metadata={"goal_id": goal.id, "task_type": "information_gathering"}
        )
        tasks.append(info_task)
        
        # 2. 分析任务
        analysis_task = Task(
            name=f"分析_{goal.id}",
            description=f"分析目标 '{goal.description}' 的可行性和方法",
            priority=goal.priority,
            dependencies=[info_task.id],
            metadata={"goal_id": goal.id, "task_type": "analysis"}
        )
        tasks.append(analysis_task)
        
        # 3. 执行任务
        if goal.criteria.get("requires_experiment", False):
            exp_task = Task(
                name=f"执行实验_{goal.id}",
                description=f"为目标 '{goal.description}' 执行实验",
                priority=goal.priority,
                dependencies=[analysis_task.id],
                metadata={"goal_id": goal.id, "task_type": "experiment"},
                required_tools=["dft_calculator", "lammps_simulator"]
            )
            tasks.append(exp_task)
            
        # 4. 验证任务
        validation_task = Task(
            name=f"验证_{goal.id}",
            description=f"验证目标 '{goal.description}' 的完成情况",
            priority=goal.priority,
            dependencies=[t.id for t in tasks if t.metadata.get("task_type") != "information_gathering"],
            metadata={"goal_id": goal.id, "task_type": "validation"}
        )
        tasks.append(validation_task)
        
        return tasks


class Planner:
    """任务规划器"""
    
    def __init__(self):
        self.planning_strategies: Dict[str, Callable] = {}
        self.execution_history: List[Dict] = []
        
    def add_strategy(self, name: str, strategy: Callable):
        """添加规划策略"""
        self.planning_strategies[name] = strategy
        
    def create_plan(self, tasks: List[Task], constraints: Optional[Dict] = None) -> List[Task]:
        """创建执行计划"""
        # 拓扑排序处理依赖关系
        task_map = {t.id: t for t in tasks}
        in_degree = {t.id: 0 for t in tasks}
        
        for task in tasks:
            for dep in task.dependencies:
                if dep in in_degree:
                    in_degree[task.id] += 1
        
        # 优先级队列（按优先级排序）
        ready = [t for t in tasks if in_degree[t.id] == 0]
        ready.sort(key=lambda x: x.priority, reverse=True)
        
        plan = []
        while ready:
            task = ready.pop(0)
            plan.append(task)
            
            # 更新依赖计数
            for t in tasks:
                if task.id in t.dependencies:
                    in_degree[t.id] -= 1
                    if in_degree[t.id] == 0:
                        ready.append(t)
                        ready.sort(key=lambda x: x.priority, reverse=True)
        
        # 应用约束
        if constraints:
            plan = self._apply_constraints(plan, constraints)
        
        return plan
    
    def _apply_constraints(self, plan: List[Task], constraints: Dict) -> List[Task]:
        """应用约束条件"""
        # 资源约束
        if "max_concurrent" in constraints:
            # 限制并发任务数量
            pass
            
        # 时间约束
        if "deadline" in constraints:
            deadline = constraints["deadline"]
            # 确保计划能在截止时间前完成
            
        # 预算约束
        if "max_cost" in constraints:
            max_cost = constraints["max_cost"]
            current_cost = sum(t.estimated_cost for t in plan)
            if current_cost > max_cost:
                # 重新优化计划
                plan = self._optimize_for_cost(plan, max_cost)
                
        return plan
    
    def _optimize_for_cost(self, plan: List[Task], max_cost: float) -> List[Task]:
        """针对成本优化计划"""
        # 按性价比排序
        sorted_tasks = sorted(plan, key=lambda x: x.estimated_cost / x.priority)
        
        optimized = []
        total_cost = 0.0
        for task in sorted_tasks:
            if total_cost + task.estimated_cost <= max_cost:
                optimized.append(task)
                total_cost += task.estimated_cost
                
        return optimized


class Reflector:
    """反思系统"""
    
    def __init__(self):
        self.reflection_history: List[Reflection] = []
        self.reflection_triggers: List[Callable] = []
        
    def add_trigger(self, trigger: Callable):
        """添加反思触发器"""
        self.reflection_triggers.append(trigger)
        
    def should_reflect(self, context: Dict) -> bool:
        """判断是否应该进行反思"""
        for trigger in self.reflection_triggers:
            if trigger(context):
                return True
        return False
    
    async def reflect(self, context: Dict) -> Reflection:
        """执行反思"""
        reflection = Reflection(
            context=json.dumps(context, default=str),
            observation=context.get("observation", ""),
            evaluation=""
        )
        
        # 评估行动结果
        action_result = context.get("action_result")
        if action_result:
            if action_result.get("success"):
                reflection.evaluation = "行动成功执行"
                reflection.insights.append("成功因素需要分析")
            else:
                reflection.evaluation = f"行动失败: {action_result.get('error', '未知错误')}"
                reflection.action_items.append("分析失败原因并制定改进措施")
        
        # 生成洞察
        reflection.insights.extend(self._generate_insights(context))
        
        # 提取经验教训
        reflection.lessons_learned = self._extract_lessons(context)
        
        self.reflection_history.append(reflection)
        return reflection
    
    def _generate_insights(self, context: Dict) -> List[str]:
        """生成洞察"""
        insights = []
        
        # 分析执行时间
        execution_time = context.get("execution_time")
        if execution_time and execution_time > 300:
            insights.append("执行时间较长，考虑优化效率")
        
        # 分析成功率
        success_rate = context.get("success_rate")
        if success_rate is not None and success_rate < 0.7:
            insights.append("成功率较低，需要改进方法")
        
        return insights
    
    def _extract_lessons(self, context: Dict) -> List[str]:
        """提取经验教训"""
        lessons = []
        
        # 从失败中学习
        errors = context.get("errors", [])
        for error in errors:
            lessons.append(f"错误: {error} - 避免类似情况")
        
        # 从成功中学习
        successes = context.get("successes", [])
        for success in successes:
            lessons.append(f"成功经验: {success}")
        
        return lessons
    
    def get_reflection_summary(self, n: int = 5) -> Dict:
        """获取反思摘要"""
        recent = self.reflection_history[-n:] if len(self.reflection_history) >= n else self.reflection_history
        
        return {
            "total_reflections": len(self.reflection_history),
            "recent_insights": [r.insights for r in recent],
            "common_lessons": self._get_common_lessons(),
            "improvement_trends": self._analyze_trends()
        }
    
    def _get_common_lessons(self) -> List[str]:
        """获取常见经验教训"""
        all_lessons = []
        for r in self.reflection_history:
            all_lessons.extend(r.lessons_learned)
        
        # 统计频率
        from collections import Counter
        counter = Counter(all_lessons)
        return [lesson for lesson, count in counter.most_common(5)]
    
    def _analyze_trends(self) -> Dict:
        """分析改进趋势"""
        if len(self.reflection_history) < 2:
            return {"status": "insufficient_data"}
        
        recent = self.reflection_history[-10:]
        success_count = sum(1 for r in recent if "成功" in r.evaluation)
        
        return {
            "recent_success_rate": success_count / len(recent),
            "reflection_frequency": len(self.reflection_history) / max(1, len(set(r.timestamp.date() for r in self.reflection_history))),
            "trend": "improving" if success_count > len(recent) * 0.6 else "stable"
        }


class AutonomousAgent:
    """
    自主智能体核心类
    
    实现完整的自主材料科学家功能：
    - 目标理解与分解
    - 自主规划与决策
    - 工具调用与执行
    - 自我反思与修正
    """
    
    def __init__(self, name: str = "MaterialScientist"):
        self.name = name
        self.state = AgentState.IDLE
        self.memory = Memory()
        self.tool_registry = ToolRegistry()
        self.goal_decomposer = GoalDecomposer()
        self.planner = Planner()
        self.reflector = Reflector()
        
        self.active_goals: Dict[str, Goal] = {}
        self.task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []
        
        self.execution_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_execution_time": 0.0
        }
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{name}")
        self._setup_default_triggers()
        
    def _setup_default_triggers(self):
        """设置默认反思触发器"""
        # 任务失败时触发反思
        self.reflector.add_trigger(
            lambda ctx: ctx.get("action_result", {}).get("success") == False
        )
        
        # 执行时间过长时触发反思
        self.reflector.add_trigger(
            lambda ctx: ctx.get("execution_time", 0) > 600
        )
        
        # 连续失败时触发反思
        self.reflector.add_trigger(
            lambda ctx: ctx.get("consecutive_failures", 0) >= 3
        )
    
    async def understand_goal(self, description: str, criteria: Optional[Dict] = None) -> Goal:
        """
        理解目标
        
        分析用户输入的目标描述，提取关键信息，创建结构化的目标对象。
        """
        self.state = AgentState.PLANNING
        
        # 解析目标描述
        goal_analysis = self._analyze_goal_description(description)
        
        # 创建目标对象
        goal = Goal(
            description=description,
            criteria=criteria or {},
            priority=goal_analysis.get("priority", 5),
            achievement_threshold=goal_analysis.get("threshold", 0.8)
        )
        
        # 存储到内存
        self.active_goals[goal.id] = goal
        self.memory.add_to_short_term({
            "type": "goal_created",
            "goal_id": goal.id,
            "description": description
        })
        
        self.logger.info(f"理解目标: {description[:50]}...")
        return goal
    
    def _analyze_goal_description(self, description: str) -> Dict:
        """分析目标描述"""
        analysis = {
            "priority": 5,
            "threshold": 0.8,
            "requires_experiment": False,
            "complexity": "medium"
        }
        
        # 关键词分析
        high_priority_keywords = ["urgent", "critical", "important", "紧急", "关键"]
        experiment_keywords = ["experiment", "test", "simulate", "calculate", "实验", "测试", "模拟", "计算"]
        
        desc_lower = description.lower()
        
        if any(kw in desc_lower for kw in high_priority_keywords):
            analysis["priority"] = 9
            
        if any(kw in desc_lower for kw in experiment_keywords):
            analysis["requires_experiment"] = True
            analysis["complexity"] = "high"
        
        # 长度复杂度分析
        if len(description) > 200:
            analysis["complexity"] = "high"
        elif len(description) < 50:
            analysis["complexity"] = "low"
        
        return analysis
    
    async def decompose_goal(self, goal: Goal) -> List[Task]:
        """
        分解目标
        
        将目标分解为可执行的任务序列。
        """
        tasks = self.goal_decomposer.decompose(goal)
        
        # 关联任务到目标
        for task in tasks:
            task.metadata["goal_id"] = goal.id
        
        goal.subgoals = tasks
        
        self.logger.info(f"目标 '{goal.description[:30]}...' 分解为 {len(tasks)} 个任务")
        
        self.memory.add_to_short_term({
            "type": "goal_decomposed",
            "goal_id": goal.id,
            "task_count": len(tasks)
        })
        
        return tasks
    
    async def plan(self, tasks: List[Task], constraints: Optional[Dict] = None) -> List[Task]:
        """
        规划执行
        
        根据任务和约束创建执行计划。
        """
        self.state = AgentState.PLANNING
        
        plan = self.planner.create_plan(tasks, constraints)
        self.task_queue = plan
        
        self.memory.add_to_short_term({
            "type": "plan_created",
            "task_count": len(plan),
            "constraints": constraints
        })
        
        self.logger.info(f"创建执行计划，包含 {len(plan)} 个任务")
        return plan
    
    async def execute_task(self, task: Task) -> Task:
        """
        执行任务
        
        执行单个任务，处理执行过程中的所有细节。
        """
        self.state = AgentState.EXECUTING
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        self.logger.info(f"开始执行任务: {task.name}")
        
        try:
            # 检查依赖
            for dep_id in task.dependencies:
                dep_task = self._find_task(dep_id)
                if dep_task and dep_task.status != TaskStatus.COMPLETED:
                    task.status = TaskStatus.WAITING_FOR_DEPENDENCY
                    self.logger.warning(f"任务 {task.name} 等待依赖 {dep_id}")
                    return task
            
            # 执行任务
            start_time = datetime.now()
            
            if task.metadata.get("task_type") == "experiment":
                result = await self._execute_experiment(task)
            else:
                result = await self._execute_generic_task(task)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            self.completed_tasks.append(task)
            self.execution_stats["completed_tasks"] += 1
            self.execution_stats["total_execution_time"] += execution_time
            
            self.logger.info(f"任务 {task.name} 完成，耗时 {execution_time:.2f}s")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.current_retry += 1
            
            self.failed_tasks.append(task)
            self.execution_stats["failed_tasks"] += 1
            
            self.logger.error(f"任务 {task.name} 失败: {e}")
            
            # 检查是否需要重试
            if task.current_retry < task.max_retries:
                task.status = TaskStatus.PENDING
                self.logger.info(f"任务 {task.name} 将在稍后重试 ({task.current_retry}/{task.max_retries})")
        
        self.execution_stats["total_tasks"] += 1
        return task
    
    async def _execute_experiment(self, task: Task) -> Any:
        """执行实验任务"""
        # 获取所需工具
        tools_needed = task.required_tools
        
        results = {}
        for tool_name in tools_needed:
            tool = self.tool_registry.get(tool_name)
            if tool:
                tool_result = await tool.execute(**task.metadata.get("tool_params", {}))
                results[tool_name] = tool_result
            else:
                raise ValueError(f"工具 {tool_name} 未找到")
        
        return results
    
    async def _execute_generic_task(self, task: Task) -> Any:
        """执行通用任务"""
        # 模拟任务执行
        await asyncio.sleep(0.1)
        return {"status": "completed", "task_name": task.name}
    
    def _find_task(self, task_id: str) -> Optional[Task]:
        """查找任务"""
        for task in self.completed_tasks + self.failed_tasks + self.task_queue:
            if task.id == task_id:
                return task
        return None
    
    async def execute_plan(self) -> List[Task]:
        """
        执行计划
        
        按顺序执行任务队列中的所有任务。
        """
        completed = []
        
        while self.task_queue:
            task = self.task_queue.pop(0)
            
            # 跳过已完成的任务
            if task.status == TaskStatus.COMPLETED:
                continue
            
            result_task = await self.execute_task(task)
            completed.append(result_task)
            
            # 执行后反思
            context = {
                "task": task,
                "action_result": {"success": result_task.status == TaskStatus.COMPLETED},
                "execution_time": self.execution_stats["total_execution_time"]
            }
            
            if self.reflector.should_reflect(context):
                reflection = await self.reflector.reflect(context)
                self.logger.info(f"反思生成: {len(reflection.insights)} 条洞察")
        
        self.state = AgentState.IDLE
        return completed
    
    async def reflect(self, context: Optional[Dict] = None) -> Reflection:
        """
        自我反思
        
        对最近的行动进行反思和分析。
        """
        self.state = AgentState.REFLECTING
        
        if context is None:
            context = {
                "completed_tasks": len(self.completed_tasks),
                "failed_tasks": len(self.failed_tasks),
                "execution_stats": self.execution_stats,
                "success_rate": self.execution_stats["completed_tasks"] / max(1, self.execution_stats["total_tasks"])
            }
        
        reflection = await self.reflector.reflect(context)
        
        self.state = AgentState.IDLE
        return reflection
    
    async def learn(self, reflection: Reflection) -> Dict:
        """
        学习改进
        
        从反思中提取经验教训，更新知识库。
        """
        self.state = AgentState.LEARNING
        
        # 存储经验教训
        for lesson in reflection.lessons_learned:
            lesson_id = f"lesson_{uuid.uuid4().hex[:8]}"
            self.memory.store_long_term(lesson_id, lesson, category="lessons")
        
        # 存储洞察
        for insight in reflection.insights:
            insight_id = f"insight_{uuid.uuid4().hex[:8]}"
            self.memory.store_long_term(insight_id, insight, category="insights")
        
        # 更新策略
        improvements = self._generate_improvements(reflection)
        
        self.state = AgentState.IDLE
        
        return {
            "lessons_stored": len(reflection.lessons_learned),
            "insights_stored": len(reflection.insights),
            "improvements": improvements
        }
    
    def _generate_improvements(self, reflection: Reflection) -> List[str]:
        """生成改进措施"""
        improvements = []
        
        # 基于反思生成具体的改进建议
        if "失败" in reflection.evaluation:
            improvements.append("加强错误处理机制")
            improvements.append("增加预验证步骤")
        
        if "执行时间较长" in str(reflection.insights):
            improvements.append("优化任务并行执行")
            improvements.append("使用更高效的算法")
        
        return improvements
    
    async def run(self, goal_description: str, criteria: Optional[Dict] = None, constraints: Optional[Dict] = None) -> Dict:
        """
        完整执行流程
        
        从目标理解到最终执行的完整流程。
        """
        self.logger.info("=" * 60)
        self.logger.info("开始自主执行流程")
        self.logger.info("=" * 60)
        
        # 1. 理解目标
        goal = await self.understand_goal(goal_description, criteria)
        
        # 2. 分解目标
        tasks = await self.decompose_goal(goal)
        
        # 3. 规划
        plan = await self.plan(tasks, constraints)
        
        # 4. 执行
        completed = await self.execute_plan()
        
        # 5. 反思
        reflection = await self.reflect()
        
        # 6. 学习
        learning_result = await self.learn(reflection)
        
        # 7. 评估目标达成
        goal_achieved = self._evaluate_goal_achievement(goal, completed)
        
        self.logger.info("=" * 60)
        self.logger.info("自主执行流程完成")
        self.logger.info("=" * 60)
        
        return {
            "goal": goal,
            "tasks_completed": len(completed),
            "success_rate": len(self.completed_tasks) / max(1, self.execution_stats["total_tasks"]),
            "goal_achieved": goal_achieved,
            "reflection": reflection,
            "learning": learning_result
        }
    
    def _evaluate_goal_achievement(self, goal: Goal, completed_tasks: List[Task]) -> bool:
        """评估目标达成情况"""
        if not completed_tasks:
            return False
        
        success_count = sum(1 for t in completed_tasks if t.status == TaskStatus.COMPLETED)
        total_count = len(completed_tasks)
        
        achievement_rate = success_count / total_count if total_count > 0 else 0
        
        goal.current_progress = achievement_rate
        goal.status = TaskStatus.COMPLETED if achievement_rate >= goal.achievement_threshold else TaskStatus.FAILED
        
        return achievement_rate >= goal.achievement_threshold
    
    def get_status(self) -> Dict:
        """获取智能体状态"""
        return {
            "name": self.name,
            "state": self.state.name,
            "active_goals": len(self.active_goals),
            "pending_tasks": len(self.task_queue),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "execution_stats": self.execution_stats,
            "memory": {
                "short_term_size": len(self.memory.short_term),
                "long_term_categories": list(self.memory.long_term.keys()),
                "episodes_count": len(self.memory.episodic)
            }
        }
    
    def register_tool(self, tool: BaseTool):
        """注册工具"""
        self.tool_registry.register(tool)
        self.logger.info(f"注册工具: {tool.name}")


# 预定义的工具示例
class DFTTool(BaseTool):
    """DFT计算工具"""
    
    def __init__(self):
        super().__init__("dft_calculator", "执行DFT计算")
    
    async def execute(self, structure: Dict, parameters: Optional[Dict] = None, **kwargs) -> Dict:
        """执行DFT计算"""
        self.usage_count += 1
        
        try:
            # 模拟DFT计算
            await asyncio.sleep(0.5)
            
            result = {
                "energy": np.random.uniform(-100, -10),
                "forces": np.random.randn(len(structure.get("atoms", [])), 3).tolist(),
                "stress": np.random.randn(6).tolist(),
                "converged": True
            }
            
            self.success_count += 1
            return result
            
        except Exception as e:
            return {"error": str(e), "converged": False}


class LAMMPTool(BaseTool):
    """LAMMPS分子动力学模拟工具"""
    
    def __init__(self):
        super().__init__("lammps_simulator", "执行分子动力学模拟")
    
    async def execute(self, input_script: str, num_steps: int = 1000, **kwargs) -> Dict:
        """执行LAMMPS模拟"""
        self.usage_count += 1
        
        try:
            # 模拟MD计算
            await asyncio.sleep(0.5)
            
            # 生成模拟轨迹数据
            trajectory = {
                "timesteps": list(range(0, num_steps, 100)),
                "temperature": [300 + np.random.randn() * 10 for _ in range(num_steps // 100)],
                "pressure": [1.0 + np.random.randn() * 0.1 for _ in range(num_steps // 100)],
                "energy": [np.random.uniform(-1000, -500) + i * 0.01 for i in range(num_steps // 100)]
            }
            
            result = {
                "trajectory": trajectory,
                "final_temperature": trajectory["temperature"][-1],
                "final_energy": trajectory["energy"][-1],
                "completed_steps": num_steps
            }
            
            self.success_count += 1
            return result
            
        except Exception as e:
            return {"error": str(e), "completed_steps": 0}


class StructureAnalysisTool(BaseTool):
    """结构分析工具"""
    
    def __init__(self):
        super().__init__("structure_analyzer", "分析晶体和分子结构")
    
    async def execute(self, structure: Dict, analysis_type: str = "geometry", **kwargs) -> Dict:
        """分析结构"""
        self.usage_count += 1
        
        try:
            atoms = structure.get("atoms", [])
            
            if analysis_type == "geometry":
                result = {
                    "atom_count": len(atoms),
                    "bond_lengths": np.random.uniform(1.0, 3.0, size=len(atoms)).tolist(),
                    "angles": np.random.uniform(80, 120, size=len(atoms)).tolist(),
                    "coordination_numbers": np.random.randint(2, 8, size=len(atoms)).tolist()
                }
            elif analysis_type == "symmetry":
                result = {
                    "space_group": f"P-{np.random.randint(1, 230)}",
                    "point_group": "m-3m",
                    "symmetry_operations": np.random.randint(1, 50)
                }
            else:
                result = {"message": f"未知的分析类型: {analysis_type}"}
            
            self.success_count += 1
            return result
            
        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    # 测试代码
    async def test_agent():
        agent = AutonomousAgent("TestAgent")
        
        # 注册工具
        agent.register_tool(DFTTool())
        agent.register_tool(LAMMPTool())
        agent.register_tool(StructureAnalysisTool())
        
        # 运行示例目标
        result = await agent.run(
            goal_description="发现用于水分解的高效催化剂",
            criteria={
                "requires_experiment": True,
                "target_overpotential": 0.3,
                "stability_criteria": 1000  # 小时
            },
            constraints={
                "max_cost": 1000.0,
                "deadline": datetime(2026, 12, 31)
            }
        )
        
        print("\n执行结果:")
        print(f"任务完成数: {result['tasks_completed']}")
        print(f"成功率: {result['success_rate']:.2%}")
        print(f"目标达成: {result['goal_achieved']}")
        
        # 打印状态
        status = agent.get_status()
        print(f"\n智能体状态: {status}")
    
    asyncio.run(test_agent())
