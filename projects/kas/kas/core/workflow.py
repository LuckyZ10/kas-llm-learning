"""
KAS Agent 工作流系统
多 Agent 协作编排
"""
import json
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

from kas.core.config import get_config


class StepStatus(Enum):
    """步骤状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """工作流步骤"""
    id: str
    agent_name: str
    task: str
    depends_on: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    output: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'id': self.id,
            'agent_name': self.agent_name,
            'task': self.task,
            'depends_on': self.depends_on,
            'status': self.status.value,
            'output': self.output,
            'error': self.error,
            'started_at': self.started_at,
            'completed_at': self.completed_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'WorkflowStep':
        """从字典创建"""
        step = cls(
            id=data['id'],
            agent_name=data['agent_name'],
            task=data['task'],
            depends_on=data.get('depends_on', []),
            status=StepStatus(data.get('status', 'pending'))
        )
        step.output = data.get('output')
        step.error = data.get('error')
        step.started_at = data.get('started_at')
        step.completed_at = data.get('completed_at')
        return step


@dataclass
class Workflow:
    """工作流定义"""
    name: str
    description: Optional[str] = None
    steps: List[WorkflowStep] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'name': self.name,
            'description': self.description,
            'steps': [s.to_dict() for s in self.steps],
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Workflow':
        """从字典创建"""
        workflow = cls(
            name=data['name'],
            description=data.get('description'),
            steps=[WorkflowStep.from_dict(s) for s in data.get('steps', [])],
            created_at=data.get('created_at', datetime.now().isoformat()),
            updated_at=data.get('updated_at', datetime.now().isoformat())
        )
        return workflow
    
    def add_step(self, agent_name: str, task: str, depends_on: List[str] = None) -> str:
        """添加步骤"""
        step_id = f"step_{len(self.steps)}"
        step = WorkflowStep(
            id=step_id,
            agent_name=agent_name,
            task=task,
            depends_on=depends_on or []
        )
        self.steps.append(step)
        self.updated_at = datetime.now().isoformat()
        return step_id
    
    def get_execution_order(self) -> List[List[str]]:
        """
        获取执行顺序（分层）
        返回: [[第一层步骤ID], [第二层步骤ID], ...]
        """
        if not self.steps:
            return []
        
        # 构建依赖图
        completed = set()
        remaining = {s.id: s for s in self.steps}
        order = []
        
        while remaining:
            # 找到当前可以执行的步骤（依赖已满足）
            executable = []
            for step_id, step in list(remaining.items()):
                if all(dep in completed for dep in step.depends_on):
                    executable.append(step_id)
            
            if not executable:
                # 有循环依赖
                raise ValueError("Workflow has circular dependencies")
            
            order.append(executable)
            for step_id in executable:
                completed.add(step_id)
                del remaining[step_id]
        
        return order


class WorkflowEngine:
    """工作流引擎"""
    
    def __init__(self):
        self.workflows_dir = Path(get_config().config_dir) / "workflows"
        self.workflows_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_workflow_path(self, name: str) -> Path:
        """获取工作流文件路径"""
        return self.workflows_dir / f"{name}.yaml"
    
    def create(self, name: str, description: str = None) -> Workflow:
        """创建工作流"""
        workflow = Workflow(name=name, description=description)
        self.save(workflow)
        return workflow
    
    def save(self, workflow: Workflow):
        """保存工作流"""
        path = self._get_workflow_path(workflow.name)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(workflow.to_dict(), f, allow_unicode=True, sort_keys=False)
    
    def load(self, name: str) -> Optional[Workflow]:
        """加载工作流"""
        path = self._get_workflow_path(name)
        if not path.exists():
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return Workflow.from_dict(data)
    
    def list_workflows(self) -> List[str]:
        """列出所有工作流"""
        workflows = []
        for path in self.workflows_dir.glob("*.yaml"):
            workflows.append(path.stem)
        return sorted(workflows)
    
    def delete(self, name: str) -> bool:
        """删除工作流"""
        path = self._get_workflow_path(name)
        if path.exists():
            path.unlink()
            return True
        return False
    
    def execute(
        self,
        workflow: Workflow,
        context: str,
        callback: Optional[Callable[[WorkflowStep, str], None]] = None,
        timeout: int = 300,
        use_mock: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        执行工作流

        Args:
            workflow: 工作流对象
            context: 任务上下文/输入
            callback: 步骤完成回调 (step, output) -> None
            timeout: 单步骤超时时间（秒），默认 300
            use_mock: 是否使用 mock 模式，默认从配置读取

        Returns:
            执行结果
        """
        from kas.core.chat import ChatEngine
        from kas.core.config import get_config
        import signal

        # 确定是否使用 mock
        if use_mock is None:
            config = get_config()
            use_mock = not config.has_api_key()

        results = {
            'workflow': workflow.name,
            'started_at': datetime.now().isoformat(),
            'steps': {},
            'success': True
        }

        # 获取执行顺序
        try:
            execution_order = workflow.get_execution_order()
        except ValueError as e:
            results['success'] = False
            results['error'] = f"工作流配置错误: {e}"
            return results

        # 步骤输出缓存
        step_outputs = {}

        def timeout_handler(signum, frame):
            raise TimeoutError(f"步骤执行超时（{timeout}秒）")

        # 按层执行
        for layer in execution_order:
            for step_id in layer:
                step = next(s for s in workflow.steps if s.id == step_id)

                # 更新状态
                step.status = StepStatus.RUNNING
                step.started_at = datetime.now().isoformat()

                try:
                    # 设置超时
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(timeout)

                    # 构建任务描述
                    task_desc = step.task
                    if context:
                        task_desc = f"上下文: {context}\n\n任务: {task_desc}"

                    # 添加上游步骤的输出作为参考
                    if step.depends_on:
                        refs = []
                        for dep_id in step.depends_on:
                            if dep_id in step_outputs:
                                refs.append(f"[{dep_id}]\n{step_outputs[dep_id]}")
                        if refs:
                            task_desc += "\n\n前置步骤输出:\n" + "\n---\n".join(refs)

                    # 执行 Agent
                    chat = ChatEngine(step.agent_name)
                    output = chat.run(task_desc, use_mock=use_mock)

                    # 取消超时
                    signal.alarm(0)
                    
                    # 更新步骤
                    step.output = output
                    step.status = StepStatus.COMPLETED
                    step.completed_at = datetime.now().isoformat()
                    step_outputs[step_id] = output
                    
                    results['steps'][step_id] = {
                        'status': 'completed',
                        'output': output
                    }
                    
                    if callback:
                        callback(step, output)

                except TimeoutError as e:
                    signal.alarm(0)  # 取消超时
                    step.status = StepStatus.FAILED
                    step.error = str(e)
                    step.completed_at = datetime.now().isoformat()
                    results['steps'][step_id] = {
                        'status': 'failed',
                        'error': str(e)
                    }
                    results['success'] = False

                except Exception as e:
                    signal.alarm(0)  # 取消超时
                    step.status = StepStatus.FAILED
                    step.error = str(e)
                    step.completed_at = datetime.now().isoformat()
                    results['steps'][step_id] = {
                        'status': 'failed',
                        'error': str(e)
                    }
                    results['success'] = False
        
        results['completed_at'] = datetime.now().isoformat()
        
        # 保存更新后的工作流状态
        self.save(workflow)
        
        return results


# 便捷函数
def get_workflow_engine() -> WorkflowEngine:
    """获取工作流引擎"""
    return WorkflowEngine()
