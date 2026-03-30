"""
DFT-LAMMPS 工作流组合器
=======================
拖拽式/声明式/代码式三种方式

支持灵活的工作流定义和组合，满足不同场景需求。

Author: DFT-LAMMPS Team
Phase: 56 - Orchestration System
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any, Callable, Dict, Generic, List, Optional, Set, Tuple, TypeVar, Union,
    get_type_hints
)
import yaml

from .module_registry import (
    ModuleRegistry, Capability, CapabilityType, 
    RegisteredModule, ModuleInterface
)
from .capability_graph import (
    CapabilityGraph, CapabilityNode, CapabilityEdge,
    NodeType, EdgeType, CapabilityPath
)


logger = logging.getLogger("workflow_composer")


class WorkflowType(Enum):
    """工作流类型"""
    SEQUENTIAL = "sequential"      # 顺序执行
    PARALLEL = "parallel"          # 并行执行
    CONDITIONAL = "conditional"    # 条件执行
    LOOP = "loop"                  # 循环执行
    DAG = "dag"                    # 有向无环图


class StepStatus(Enum):
    """步骤状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class WorkflowStep:
    """工作流步骤"""
    id: str                                     # 步骤ID
    name: str                                   # 名称
    description: str = ""                       # 描述
    
    # 执行信息
    module_name: Optional[str] = None           # 执行模块
    capability_name: Optional[str] = None       # 使用的能力
    function_name: Optional[str] = None         # 函数名
    
    # 输入输出
    inputs: Dict[str, Any] = field(default_factory=dict)   # 输入映射
    outputs: Dict[str, Any] = field(default_factory=dict)  # 输出映射
    
    # 执行配置
    parameters: Dict[str, Any] = field(default_factory=dict)  # 参数
    condition: Optional[str] = None             # 执行条件
    retry_count: int = 3                        # 重试次数
    timeout: Optional[float] = None             # 超时时间
    
    # 依赖关系
    depends_on: List[str] = field(default_factory=list)   # 依赖步骤
    
    # 运行时状态
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "module_name": self.module_name,
            "capability_name": self.capability_name,
            "function_name": self.function_name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "parameters": self.parameters,
            "condition": self.condition,
            "retry_count": self.retry_count,
            "timeout": self.timeout,
            "depends_on": self.depends_on,
            "status": self.status.value
        }


@dataclass
class Workflow:
    """工作流定义"""
    id: str                                     # 工作流ID
    name: str                                   # 名称
    description: str = ""                       # 描述
    version: str = "1.0.0"                      # 版本
    
    # 执行类型
    workflow_type: WorkflowType = WorkflowType.SEQUENTIAL
    
    # 步骤定义
    steps: List[WorkflowStep] = field(default_factory=list)
    
    # 全局配置
    global_parameters: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    
    # 元数据
    tags: List[str] = field(default_factory=list)
    author: str = ""
    created_at: float = field(default_factory=lambda: __import__('time').time())
    
    # 输入输出定义
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "workflow_type": self.workflow_type.value,
            "steps": [s.to_dict() for s in self.steps],
            "global_parameters": self.global_parameters,
            "environment": self.environment,
            "tags": self.tags,
            "author": self.author,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema
        }
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证工作流定义"""
        errors = []
        
        # 检查步骤ID唯一性
        step_ids = [s.id for s in self.steps]
        if len(step_ids) != len(set(step_ids)):
            errors.append("Duplicate step IDs found")
        
        # 检查依赖有效性
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in step_ids:
                    errors.append(f"Step {step.id} depends on non-existent step {dep}")
        
        # 检查循环依赖
        if self._has_cycle():
            errors.append("Circular dependency detected")
        
        return len(errors) == 0, errors
    
    def _has_cycle(self) -> bool:
        """检测循环依赖"""
        graph = {s.id: s.depends_on for s in self.steps}
        visited = set()
        rec_stack = set()
        
        def has_cycle_util(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle_util(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                if has_cycle_util(node):
                    return True
        
        return False


@dataclass
class WorkflowExecution:
    """工作流执行实例"""
    workflow_id: str
    execution_id: str
    status: str = "pending"
    context: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    logs: List[Dict[str, Any]] = field(default_factory=list)


class WorkflowComposer(ABC):
    """工作流组合器基类"""
    
    @abstractmethod
    def compose(self, **kwargs) -> Workflow:
        """组合工作流"""
        pass


class DeclarativeComposer(WorkflowComposer):
    """
    声明式工作流组合器
    
    通过YAML/JSON定义工作流
    
    Example:
        composer = DeclarativeComposer()
        workflow = composer.compose(yaml_file="workflow.yaml")
    """
    
    def __init__(self, registry: Optional[ModuleRegistry] = None):
        self.registry = registry or ModuleRegistry.get_instance()
    
    def compose(
        self,
        yaml_file: Optional[str] = None,
        json_file: Optional[str] = None,
        yaml_content: Optional[str] = None,
        json_content: Optional[str] = None,
        **kwargs
    ) -> Workflow:
        """从声明式定义组合工作流"""
        
        # 加载定义
        if yaml_file:
            with open(yaml_file, 'r') as f:
                definition = yaml.safe_load(f)
        elif json_file:
            with open(json_file, 'r') as f:
                definition = json.load(f)
        elif yaml_content:
            definition = yaml.safe_load(yaml_content)
        elif json_content:
            definition = json.loads(json_content)
        else:
            raise ValueError("No workflow definition provided")
        
        # 解析工作流
        return self._parse_definition(definition)
    
    def _parse_definition(self, definition: Dict[str, Any]) -> Workflow:
        """解析定义"""
        workflow = Workflow(
            id=definition.get('id', str(uuid.uuid4())),
            name=definition.get('name', 'Unnamed Workflow'),
            description=definition.get('description', ''),
            version=definition.get('version', '1.0.0'),
            workflow_type=WorkflowType(definition.get('type', 'sequential')),
            global_parameters=definition.get('parameters', {}),
            environment=definition.get('environment', {}),
            tags=definition.get('tags', []),
            author=definition.get('author', ''),
            input_schema=definition.get('input_schema', {}),
            output_schema=definition.get('output_schema', {})
        )
        
        # 解析步骤
        for step_def in definition.get('steps', []):
            step = WorkflowStep(
                id=step_def.get('id', str(uuid.uuid4())),
                name=step_def.get('name', 'Unnamed Step'),
                description=step_def.get('description', ''),
                module_name=step_def.get('module'),
                capability_name=step_def.get('capability'),
                function_name=step_def.get('function'),
                inputs=step_def.get('inputs', {}),
                outputs=step_def.get('outputs', {}),
                parameters=step_def.get('parameters', {}),
                condition=step_def.get('condition'),
                retry_count=step_def.get('retry', 3),
                timeout=step_def.get('timeout'),
                depends_on=step_def.get('depends_on', [])
            )
            workflow.steps.append(step)
        
        # 验证
        valid, errors = workflow.validate()
        if not valid:
            raise ValueError(f"Workflow validation failed: {errors}")
        
        return workflow
    
    def export(
        self, 
        workflow: Workflow, 
        format: str = "yaml",
        file_path: Optional[str] = None
    ) -> str:
        """导出工作流定义"""
        data = workflow.to_dict()
        
        if format == "yaml":
            content = yaml.dump(data, default_flow_style=False, allow_unicode=True)
        elif format == "json":
            content = json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if file_path:
            with open(file_path, 'w') as f:
                f.write(content)
        
        return content


class CodeBasedComposer(WorkflowComposer):
    """
    代码式工作流组合器
    
    通过Python代码定义工作流
    
    Example:
        composer = CodeBasedComposer()
        
        with composer.workflow("battery_simulation") as wf:
            wf.step("import_structure", module="io", function="read_poscar")
            wf.step("relax", module="vasp", function="relax_structure", depends_on=["import_structure"])
            wf.step("calc_energy", module="vasp", function="calculate_energy", depends_on=["relax"])
    """
    
    def __init__(self, registry: Optional[ModuleRegistry] = None):
        self.registry = registry or ModuleRegistry.get_instance()
        self._current_workflow: Optional[Workflow] = None
    
    def compose(self, **kwargs) -> Workflow:
        """返回当前工作流"""
        if self._current_workflow:
            wf = self._current_workflow
            self._current_workflow = None
            return wf
        raise ValueError("No workflow being composed")
    
    def workflow(
        self,
        name: str,
        description: str = "",
        version: str = "1.0.0",
        workflow_type: WorkflowType = WorkflowType.SEQUENTIAL,
        **kwargs
    ) -> CodeBasedComposer:
        """开始定义工作流"""
        self._current_workflow = Workflow(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            version=version,
            workflow_type=workflow_type,
            **kwargs
        )
        return self
    
    def step(
        self,
        name: str,
        module: Optional[str] = None,
        capability: Optional[str] = None,
        function: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        depends_on: Optional[List[str]] = None,
        condition: Optional[str] = None,
        **kwargs
    ) -> CodeBasedComposer:
        """添加步骤"""
        if not self._current_workflow:
            raise RuntimeError("No active workflow. Use workflow() first.")
        
        step = WorkflowStep(
            id=str(uuid.uuid4()),
            name=name,
            module_name=module,
            capability_name=capability,
            function_name=function,
            inputs=inputs or {},
            outputs=outputs or {},
            parameters=parameters or {},
            depends_on=depends_on or [],
            condition=condition,
            **kwargs
        )
        
        self._current_workflow.steps.append(step)
        return self
    
    def parallel(
        self,
        *step_names: str
    ) -> CodeBasedComposer:
        """标记步骤并行执行"""
        if not self._current_workflow:
            raise RuntimeError("No active workflow")
        
        # 设置并行执行类型
        self._current_workflow.workflow_type = WorkflowType.PARALLEL
        
        # 添加并行组标记（通过步骤元数据）
        for name in step_names:
            step = self._find_step(name)
            if step:
                step.parameters['_parallel_group'] = True
        
        return self
    
    def condition(
        self,
        expression: str,
        then_steps: List[str],
        else_steps: Optional[List[str]] = None
    ) -> CodeBasedComposer:
        """添加条件分支"""
        if not self._current_workflow:
            raise RuntimeError("No active workflow")
        
        self._current_workflow.workflow_type = WorkflowType.CONDITIONAL
        
        for name in then_steps:
            step = self._find_step(name)
            if step:
                step.condition = expression
        
        if else_steps:
            for name in else_steps:
                step = self._find_step(name)
                if step:
                    step.condition = f"not ({expression})"
        
        return self
    
    def loop(
        self,
        iterator: str,
        items: str,
        body_steps: List[str]
    ) -> CodeBasedComposer:
        """添加循环"""
        if not self._current_workflow:
            raise RuntimeError("No active workflow")
        
        self._current_workflow.workflow_type = WorkflowType.LOOP
        
        # 标记循环步骤
        for name in body_steps:
            step = self._find_step(name)
            if step:
                step.parameters['_loop_iterator'] = iterator
                step.parameters['_loop_items'] = items
        
        return self
    
    def _find_step(self, name: str) -> Optional[WorkflowStep]:
        """查找步骤"""
        for step in self._current_workflow.steps:
            if step.name == name or step.id == name:
                return step
        return None
    
    def __enter__(self) -> CodeBasedComposer:
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._current_workflow:
            valid, errors = self._current_workflow.validate()
            if not valid:
                raise ValueError(f"Workflow validation failed: {errors}")


class DragDropComposer(WorkflowComposer):
    """
    拖拽式工作流组合器（数据模型）
    
    提供数据结构支持拖拽式界面，实际UI在前端实现
    """
    
    def __init__(self, registry: Optional[ModuleRegistry] = None):
        self.registry = registry or ModuleRegistry.get_instance()
        self.graph = CapabilityGraph()
    
    def compose(
        self,
        nodes: Optional[List[Dict[str, Any]]] = None,
        connections: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Workflow:
        """从拖拽式定义组合工作流"""
        workflow = Workflow(
            id=str(uuid.uuid4()),
            name=kwargs.get('name', 'Drag-Drop Workflow'),
            workflow_type=WorkflowType.DAG
        )
        
        node_map = {}  # 前端节点ID -> 步骤ID映射
        
        # 创建步骤
        for node in (nodes or []):
            step = WorkflowStep(
                id=str(uuid.uuid4()),
                name=node.get('label', 'Unnamed'),
                description=node.get('description', ''),
                module_name=node.get('module'),
                capability_name=node.get('capability'),
                position=node.get('position', {'x': 0, 'y': 0})
            )
            workflow.steps.append(step)
            node_map[node['id']] = step.id
        
        # 建立连接
        for conn in (connections or []):
            source_step = node_map.get(conn['source'])
            target_step = node_map.get(conn['target'])
            
            if source_step and target_step:
                target = next(s for s in workflow.steps if s.id == target_step)
                if source_step not in target.depends_on:
                    target.depends_on.append(source_step)
        
        return workflow
    
    def get_palette_items(self) -> List[Dict[str, Any]]:
        """获取可用的组件面板项目"""
        items = []
        
        # 按能力类型分组
        for cap_type in CapabilityType:
            group = {
                "category": cap_type.value,
                "items": []
            }
            
            modules = self.registry.find_modules(capability_type=cap_type)
            for mod in modules:
                for cap in mod.capabilities:
                    group["items"].append({
                        "type": "module",
                        "module_name": mod.metadata.name,
                        "capability_name": cap.name,
                        "label": cap.name,
                        "description": cap.description,
                        "icon": self._get_icon(cap_type),
                        "inputs": cap.input_schema,
                        "outputs": cap.output_schema
                    })
            
            if group["items"]:
                items.append(group)
        
        return items
    
    def get_workflow_visual(self, workflow: Workflow) -> Dict[str, Any]:
        """获取工作流的可视化表示"""
        nodes = []
        edges = []
        
        # 为每个步骤创建可视化节点
        for i, step in enumerate(workflow.steps):
            nodes.append({
                "id": step.id,
                "type": "step",
                "label": step.name,
                "description": step.description,
                "position": step.parameters.get('position', {'x': i * 200, 'y': 100}),
                "status": step.status.value,
                "data": {
                    "module": step.module_name,
                    "capability": step.capability_name
                }
            })
        
        # 创建边
        for step in workflow.steps:
            for dep_id in step.depends_on:
                edges.append({
                    "id": f"{dep_id}-{step.id}",
                    "source": dep_id,
                    "target": step.id,
                    "type": "dependency"
                })
        
        return {"nodes": nodes, "edges": edges}
    
    def _get_icon(self, cap_type: CapabilityType) -> str:
        """获取能力类型对应的图标"""
        icons = {
            CapabilityType.CALCULATION: "🔢",
            CapabilityType.SIMULATION: "🔄",
            CapabilityType.ANALYSIS: "📊",
            CapabilityType.ML: "🧠",
            CapabilityType.VISUALIZATION: "📈",
            CapabilityType.IO: "📁",
            CapabilityType.ORCHESTRATION: "🎼",
            CapabilityType.UTILITY: "🛠️"
        }
        return icons.get(cap_type, "📦")


class SmartComposer(WorkflowComposer):
    """
    智能工作流组合器
    
    基于目标自动组合工作流
    
    Example:
        composer = SmartComposer()
        workflow = composer.compose(
            goal="calculate_battery_ion_conductivity",
            input_structure="LiFePO4.cif",
            preferences={"accuracy": "high"}
        )
    """
    
    def __init__(
        self, 
        registry: Optional[ModuleRegistry] = None,
        graph: Optional[CapabilityGraph] = None
    ):
        self.registry = registry or ModuleRegistry.get_instance()
        self.graph = graph or CapabilityGraph()
    
    def compose(
        self,
        goal: str,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        preferences: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Workflow:
        """
        智能组合工作流
        
        Args:
            goal: 目标描述（如 "calculate_formation_energy"）
            inputs: 输入数据
            outputs: 期望输出
            constraints: 约束条件
            preferences: 偏好设置
        """
        inputs = inputs or {}
        outputs = outputs or []
        constraints = constraints or {}
        preferences = preferences or {}
        
        # 1. 解析目标
        required_capabilities = self._parse_goal(goal)
        
        # 2. 使用能力图谱找路径
        workflow = Workflow(
            id=str(uuid.uuid4()),
            name=f"Auto-generated: {goal}",
            workflow_type=WorkflowType.DAG
        )
        
        # 3. 为每个所需能力构建步骤
        previous_steps = []
        
        for cap_name in required_capabilities:
            # 查找提供该能力的模块
            providers = self.registry.find_modules(capability_name=cap_name)
            
            if not providers:
                logger.warning(f"No provider found for capability: {cap_name}")
                continue
            
            # 选择最佳提供者
            best_provider = self._select_best_provider(providers, preferences)
            
            # 创建步骤
            step = WorkflowStep(
                id=str(uuid.uuid4()),
                name=f"{cap_name}",
                module_name=best_provider.metadata.name,
                capability_name=cap_name,
                depends_on=[s.id for s in previous_steps],
                parameters=self._infer_parameters(cap_name, inputs, constraints)
            )
            
            workflow.steps.append(step)
            previous_steps.append(step)
        
        # 4. 优化工作流
        workflow = self._optimize_workflow(workflow, preferences)
        
        return workflow
    
    def _parse_goal(self, goal: str) -> List[str]:
        """解析目标为所需能力列表"""
        # 目标到能力映射
        goal_patterns = {
            r"energy|formation|cohesive": ["structure_import", "relaxation", "energy_calculation"],
            r"band.*gap|electronic": ["structure_import", "scf_calculation", "band_structure"],
            r"phonon|vibration|thermal": ["structure_import", "relaxation", "phonon_calculation"],
            r"md|molecular.*dynamic": ["structure_import", "md_simulation", "trajectory_analysis"],
            r"ion.*conduct|diffusion|transport": [
                "structure_import", "relaxation", "neb_calculation", "diffusion_analysis"
            ],
            r"catalyst|adsorption": ["structure_import", "surface_building", "adsorption_calculation"],
            r"battery|electrode": [
                "structure_import", "volume_calculation", "voltage_profile", "ion_diffusion"
            ],
            r"alloy|phase.*diagram": ["structure_import", "cluster_expansion", "phase_diagram"]
        }
        
        goal_lower = goal.lower()
        for pattern, capabilities in goal_patterns.items():
            if re.search(pattern, goal_lower):
                return capabilities
        
        # 默认返回通用流程
        return ["structure_import", "calculation"]
    
    def _select_best_provider(
        self, 
        providers: List[RegisteredModule],
        preferences: Dict[str, Any]
    ) -> RegisteredModule:
        """选择最佳模块提供者"""
        # 排序：活跃状态优先，版本较新优先
        active = [p for p in providers if p.state.name == "ACTIVE"]
        candidates = active if active else providers
        
        # 根据偏好过滤
        if preferences.get('accuracy') == 'high':
            # 选择高精度模块
            candidates = [p for p in candidates if 'accurate' in p.metadata.tags]
        
        if preferences.get('speed') == 'fast':
            # 选择快速模块
            candidates = [p for p in candidates if 'fast' in p.metadata.tags]
        
        # 按版本排序
        candidates.sort(key=lambda p: p.metadata.version, reverse=True)
        
        return candidates[0] if candidates else providers[0]
    
    def _infer_parameters(
        self,
        capability: str,
        inputs: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """推断步骤参数"""
        params = {}
        
        # 从约束中提取参数
        if 'encut' in constraints:
            params['encut'] = constraints['encut']
        if 'kpoints' in constraints:
            params['kpoints'] = constraints['kpoints']
        
        # 特殊处理
        if 'relaxation' in capability:
            params['force_threshold'] = constraints.get('force_threshold', 0.01)
        
        if 'md' in capability:
            params['temperature'] = inputs.get('temperature', 300)
            params['timestep'] = constraints.get('timestep', 1.0)
        
        return params
    
    def _optimize_workflow(
        self, 
        workflow: Workflow,
        preferences: Dict[str, Any]
    ) -> Workflow:
        """优化工作流"""
        # 合并相同模块的连续步骤
        workflow.steps = self._merge_redundant_steps(workflow.steps)
        
        # 识别可并行步骤
        if preferences.get('parallel', False):
            workflow.workflow_type = WorkflowType.PARALLEL
        
        # 验证
        valid, errors = workflow.validate()
        if not valid:
            logger.warning(f"Workflow optimization resulted in errors: {errors}")
        
        return workflow
    
    def _merge_redundant_steps(self, steps: List[WorkflowStep]) -> List[WorkflowStep]:
        """合并冗余步骤"""
        if len(steps) < 2:
            return steps
        
        merged = [steps[0]]
        
        for current in steps[1:]:
            previous = merged[-1]
            
            # 如果相同模块连续执行，考虑合并
            if (current.module_name == previous.module_name and
                current.capability_name == previous.capability_name):
                # 更新参数
                previous.parameters.update(current.parameters)
                # 更新依赖
                previous.depends_on = list(set(previous.depends_on + current.depends_on))
            else:
                merged.append(current)
        
        return merged


class WorkflowExecutor:
    """
    工作流执行器
    
    执行工作流定义，管理状态和上下文
    """
    
    def __init__(self, registry: Optional[ModuleRegistry] = None):
        self.registry = registry or ModuleRegistry.get_instance()
        self._executions: Dict[str, WorkflowExecution] = {}
    
    def execute(
        self, 
        workflow: Workflow,
        inputs: Optional[Dict[str, Any]] = None,
        dry_run: bool = False
    ) -> WorkflowExecution:
        """执行工作流"""
        import time
        
        execution = WorkflowExecution(
            workflow_id=workflow.id,
            execution_id=str(uuid.uuid4()),
            context=inputs or {}
        )
        
        self._executions[execution.execution_id] = execution
        execution.start_time = time.time()
        
        try:
            # 拓扑排序
            sorted_steps = self._topological_sort(workflow.steps)
            
            # 按顺序执行
            for step in sorted_steps:
                if dry_run:
                    logger.info(f"[DRY RUN] Would execute: {step.name}")
                    step.status = StepStatus.COMPLETED
                else:
                    self._execute_step(step, execution, workflow)
                
                if step.status == StepStatus.FAILED and step.retry_count > 0:
                    # 重试
                    for attempt in range(step.retry_count):
                        logger.info(f"Retrying {step.name}, attempt {attempt + 1}")
                        self._execute_step(step, execution, workflow)
                        if step.status == StepStatus.COMPLETED:
                            break
            
            execution.status = "completed"
            
        except Exception as e:
            execution.status = "failed"
            logger.error(f"Workflow execution failed: {e}")
        
        execution.end_time = time.time()
        return execution
    
    def _execute_step(
        self, 
        step: WorkflowStep,
        execution: WorkflowExecution,
        workflow: Workflow
    ) -> None:
        """执行单个步骤"""
        import time
        
        step.status = StepStatus.RUNNING
        step.start_time = time.time()
        
        try:
            # 检查条件
            if step.condition:
                if not self._evaluate_condition(step.condition, execution.context):
                    step.status = StepStatus.SKIPPED
                    return
            
            # 获取模块
            if step.module_name:
                module = self.registry.get_module(step.module_name)
                if not module or not module.instance:
                    raise RuntimeError(f"Module not found or not initialized: {step.module_name}")
                
                # 准备输入
                inputs = self._resolve_inputs(step, execution)
                
                # 执行
                # 这里需要实际的模块调用逻辑
                step.result = {"status": "executed", "inputs": inputs}
                step.status = StepStatus.COMPLETED
            else:
                # 纯数据处理步骤
                step.result = self._process_data_step(step, execution)
                step.status = StepStatus.COMPLETED
            
            # 保存结果到上下文
            for key, value in step.outputs.items():
                execution.context[key] = step.result
            
            execution.step_results[step.id] = step.result
            
        except Exception as e:
            step.status = StepStatus.FAILED
            step.error = str(e)
            logger.error(f"Step {step.name} failed: {e}")
        
        step.end_time = time.time()
    
    def _topological_sort(self, steps: List[WorkflowStep]) -> List[WorkflowStep]:
        """拓扑排序步骤"""
        step_map = {s.id: s for s in steps}
        in_degree = {s.id: len(s.depends_on) for s in steps}
        
        # 构建反向图
        dependents = defaultdict(list)
        for step in steps:
            for dep in step.depends_on:
                dependents[dep].append(step.id)
        
        # Kahn算法
        queue = [s.id for s in steps if in_degree[s.id] == 0]
        sorted_ids = []
        
        while queue:
            node_id = queue.pop(0)
            sorted_ids.append(node_id)
            
            for dependent in dependents[node_id]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        if len(sorted_ids) != len(steps):
            raise ValueError("Circular dependency detected")
        
        return [step_map[sid] for sid in sorted_ids]
    
    def _evaluate_condition(
        self, 
        condition: str, 
        context: Dict[str, Any]
    ) -> bool:
        """评估条件表达式"""
        # 简化的条件评估
        # 实际应用中可能需要更复杂的表达式解析
        try:
            # 替换上下文变量
            for key, value in context.items():
                condition = condition.replace(f"${key}", str(value))
            
            # 安全评估
            return eval(condition, {"__builtins__": {}}, {})
        except:
            return False
    
    def _resolve_inputs(
        self, 
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """解析步骤输入"""
        resolved = {}
        
        for key, value in step.inputs.items():
            if isinstance(value, str) and value.startswith("$"):
                # 引用上下文变量
                var_name = value[1:]
                resolved[key] = execution.context.get(var_name)
            else:
                resolved[key] = value
        
        return resolved
    
    def _process_data_step(
        self, 
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> Any:
        """处理纯数据步骤"""
        # 根据步骤名称推断操作
        if "merge" in step.name.lower():
            return self._merge_data(step.inputs, execution)
        elif "transform" in step.name.lower():
            return self._transform_data(step.inputs, step.parameters)
        else:
            return step.inputs
    
    def _merge_data(
        self, 
        inputs: Dict[str, Any],
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """合并数据"""
        result = {}
        for key in inputs:
            if isinstance(inputs[key], str) and inputs[key].startswith("$"):
                result[key] = execution.context.get(inputs[key][1:])
            else:
                result[key] = inputs[key]
        return result
    
    def _transform_data(
        self, 
        inputs: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Any:
        """转换数据"""
        # 根据参数进行数据转换
        transform_type = parameters.get('type', 'identity')
        
        if transform_type == 'select':
            keys = parameters.get('keys', [])
            return {k: inputs.get(k) for k in keys}
        
        return inputs
    
    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """获取执行实例"""
        return self._executions.get(execution_id)


# 便捷函数
def create_workflow(
    name: str,
    composer_type: str = "code",
    **kwargs
) -> WorkflowComposer:
    """创建工作流组合器"""
    if composer_type == "code":
        return CodeBasedComposer()
    elif composer_type == "declarative":
        return DeclarativeComposer()
    elif composer_type == "smart":
        return SmartComposer()
    else:
        raise ValueError(f"Unknown composer type: {composer_type}")


def load_workflow(file_path: str) -> Workflow:
    """从文件加载工作流"""
    composer = DeclarativeComposer()
    
    if file_path.endswith('.yaml') or file_path.endswith('.yml'):
        return composer.compose(yaml_file=file_path)
    elif file_path.endswith('.json'):
        return composer.compose(json_file=file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def save_workflow(
    workflow: Workflow, 
    file_path: str,
    format: str = "yaml"
) -> None:
    """保存工作流到文件"""
    composer = DeclarativeComposer()
    composer.export(workflow, format=format, file_path=file_path)