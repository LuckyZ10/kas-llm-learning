"""
DFT-LAMMPS 状态管理系统
=======================
工作流状态机、断点续算

管理工作流执行状态，支持断点续算和错误恢复。

Author: DFT-LAMMPS Team
Phase: 56 - Orchestration System
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Set, Tuple, TypeVar, Union

from ..orchestration.workflow_composer import Workflow, WorkflowStep, StepStatus


logger = logging.getLogger("state_manager")


class WorkflowState(Enum):
    """工作流状态"""
    PENDING = "pending"           # 等待执行
    RUNNING = "running"           # 运行中
    PAUSED = "paused"             # 暂停
    COMPLETED = "completed"       # 完成
    FAILED = "failed"             # 失败
    CANCELLED = "cancelled"       # 已取消
    RETRYING = "retrying"         # 重试中


class CheckpointStrategy(Enum):
    """检查点策略"""
    NONE = "none"                 # 不保存
    END_OF_STEP = "end_of_step"   # 每步结束保存
    ON_ERROR = "on_error"         # 错误时保存
    PERIODIC = "periodic"         # 定期保存


@dataclass
class Checkpoint:
    """检查点"""
    checkpoint_id: str
    workflow_id: str
    execution_id: str
    step_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    step_states: Dict[str, StepStatus] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "workflow_id": self.workflow_id,
            "execution_id": self.execution_id,
            "step_id": self.step_id,
            "timestamp": self.timestamp,
            "context": self.context,
            "step_states": {k: v.value for k, v in self.step_states.items()},
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Checkpoint:
        return cls(
            checkpoint_id=data["checkpoint_id"],
            workflow_id=data["workflow_id"],
            execution_id=data["execution_id"],
            step_id=data.get("step_id"),
            timestamp=data.get("timestamp", time.time()),
            context=data.get("context", {}),
            step_states={k: StepStatus(v) for k, v in data.get("step_states", {}).items()},
            metadata=data.get("metadata", {})
        )


@dataclass
class ExecutionRecord:
    """执行记录"""
    execution_id: str
    workflow_id: str
    state: WorkflowState
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    current_step: Optional[str] = None
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    
    @property
    def duration(self) -> Optional[float]:
        """执行时长"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        if self.start_time:
            return time.time() - self.start_time
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "workflow_id": self.workflow_id,
            "state": self.state.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "current_step": self.current_step,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "error_message": self.error_message,
            "context": self.context,
            "retry_count": self.retry_count,
            "duration": self.duration
        }


class StateStore(ABC):
    """状态存储抽象基类"""
    
    @abstractmethod
    def save_execution(self, record: ExecutionRecord) -> None:
        """保存执行记录"""
        pass
    
    @abstractmethod
    def load_execution(self, execution_id: str) -> Optional[ExecutionRecord]:
        """加载执行记录"""
        pass
    
    @abstractmethod
    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """保存检查点"""
        pass
    
    @abstractmethod
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """加载检查点"""
        pass
    
    @abstractmethod
    def list_checkpoints(self, workflow_id: str) -> List[Checkpoint]:
        """列出检查点"""
        pass
    
    @abstractmethod
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """删除检查点"""
        pass


class InMemoryStateStore(StateStore):
    """内存状态存储"""
    
    def __init__(self):
        self._executions: Dict[str, ExecutionRecord] = {}
        self._checkpoints: Dict[str, Checkpoint] = {}
    
    def save_execution(self, record: ExecutionRecord) -> None:
        self._executions[record.execution_id] = record
    
    def load_execution(self, execution_id: str) -> Optional[ExecutionRecord]:
        return self._executions.get(execution_id)
    
    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        return self._checkpoints.get(checkpoint_id)
    
    def list_checkpoints(self, workflow_id: str) -> List[Checkpoint]:
        return [
            cp for cp in self._checkpoints.values()
            if cp.workflow_id == workflow_id
        ]
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        if checkpoint_id in self._checkpoints:
            del self._checkpoints[checkpoint_id]
            return True
        return False


class FileStateStore(StateStore):
    """文件状态存储"""
    
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self._executions_dir = self.base_path / "executions"
        self._checkpoints_dir = self.base_path / "checkpoints"
        
        self._executions_dir.mkdir(exist_ok=True)
        self._checkpoints_dir.mkdir(exist_ok=True)
    
    def save_execution(self, record: ExecutionRecord) -> None:
        file_path = self._executions_dir / f"{record.execution_id}.json"
        with open(file_path, 'w') as f:
            json.dump(record.to_dict(), f, indent=2)
    
    def load_execution(self, execution_id: str) -> Optional[ExecutionRecord]:
        file_path = self._executions_dir / f"{execution_id}.json"
        if not file_path.exists():
            return None
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        return ExecutionRecord(**data)
    
    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        file_path = self._checkpoints_dir / f"{checkpoint.checkpoint_id}.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        file_path = self._checkpoints_dir / f"{checkpoint_id}.pkl"
        if not file_path.exists():
            return None
        
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def list_checkpoints(self, workflow_id: str) -> List[Checkpoint]:
        checkpoints = []
        for file_path in self._checkpoints_dir.glob("*.pkl"):
            cp = self.load_checkpoint(file_path.stem)
            if cp and cp.workflow_id == workflow_id:
                checkpoints.append(cp)
        return sorted(checkpoints, key=lambda x: x.timestamp, reverse=True)
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        file_path = self._checkpoints_dir / f"{checkpoint_id}.pkl"
        if file_path.exists():
            file_path.unlink()
            return True
        return False


class StateManager:
    """
    状态管理器
    
    管理工作流执行状态，支持断点续算
    
    Example:
        manager = StateManager()
        
        # 开始执行
        record = manager.start_execution("workflow_1")
        
        # 更新状态
        manager.update_step_status(record.execution_id, "step_1", StepStatus.COMPLETED)
        
        # 创建检查点
        manager.create_checkpoint(record.execution_id)
        
        # 暂停和恢复
        manager.pause_execution(record.execution_id)
        record = manager.resume_execution(record.execution_id)
    """
    
    def __init__(
        self,
        store: Optional[StateStore] = None,
        checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.END_OF_STEP
    ):
        self.store = store or InMemoryStateStore()
        self.checkpoint_strategy = checkpoint_strategy
        self._checkpoint_interval: float = 300.0  # 5分钟
        self._last_checkpoint: Dict[str, float] = {}
    
    def start_execution(
        self,
        workflow_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionRecord:
        """开始新的执行"""
        import uuid
        
        execution_id = str(uuid.uuid4())
        record = ExecutionRecord(
            execution_id=execution_id,
            workflow_id=workflow_id,
            state=WorkflowState.RUNNING,
            start_time=time.time(),
            context=context or {}
        )
        
        self.store.save_execution(record)
        logger.info(f"Started execution: {execution_id}")
        
        return record
    
    def update_step_status(
        self,
        execution_id: str,
        step_id: str,
        status: StepStatus,
        step_result: Optional[Any] = None
    ) -> None:
        """更新步骤状态"""
        record = self.store.load_execution(execution_id)
        if not record:
            raise ValueError(f"Execution not found: {execution_id}")
        
        record.current_step = step_id
        
        if status == StepStatus.COMPLETED:
            if step_id not in record.completed_steps:
                record.completed_steps.append(step_id)
            
            # 检查是否需要创建检查点
            if self.checkpoint_strategy == CheckpointStrategy.END_OF_STEP:
                self.create_checkpoint(execution_id, step_id)
        
        elif status == StepStatus.FAILED:
            if step_id not in record.failed_steps:
                record.failed_steps.append(step_id)
            
            if self.checkpoint_strategy in [CheckpointStrategy.ON_ERROR, CheckpointStrategy.END_OF_STEP]:
                self.create_checkpoint(execution_id, step_id)
        
        # 保存结果到上下文
        if step_result is not None:
            record.context[f"step_{step_id}_result"] = step_result
        
        self.store.save_execution(record)
    
    def complete_execution(
        self,
        execution_id: str,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """完成执行"""
        record = self.store.load_execution(execution_id)
        if not record:
            return
        
        record.state = WorkflowState.COMPLETED if success else WorkflowState.FAILED
        record.end_time = time.time()
        record.error_message = error_message
        
        self.store.save_execution(record)
        logger.info(f"Completed execution {execution_id}: {record.state.value}")
    
    def pause_execution(self, execution_id: str) -> ExecutionRecord:
        """暂停执行"""
        record = self.store.load_execution(execution_id)
        if not record:
            raise ValueError(f"Execution not found: {execution_id}")
        
        if record.state != WorkflowState.RUNNING:
            raise ValueError(f"Cannot pause execution in state: {record.state}")
        
        # 创建检查点
        self.create_checkpoint(execution_id, record.current_step)
        
        record.state = WorkflowState.PAUSED
        self.store.save_execution(record)
        
        logger.info(f"Paused execution: {execution_id}")
        return record
    
    def resume_execution(self, execution_id: str) -> ExecutionRecord:
        """恢复执行"""
        record = self.store.load_execution(execution_id)
        if not record:
            raise ValueError(f"Execution not found: {execution_id}")
        
        if record.state not in [WorkflowState.PAUSED, WorkflowState.FAILED, WorkflowState.RETRYING]:
            raise ValueError(f"Cannot resume execution in state: {record.state}")
        
        # 加载最新检查点
        checkpoints = self.store.list_checkpoints(record.workflow_id)
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: x.timestamp)
            # 恢复上下文
            record.context.update(latest_checkpoint.context)
            logger.info(f"Restored from checkpoint: {latest_checkpoint.checkpoint_id}")
        
        record.state = WorkflowState.RUNNING
        record.retry_count += 1
        self.store.save_execution(record)
        
        logger.info(f"Resumed execution: {execution_id}")
        return record
    
    def cancel_execution(self, execution_id: str) -> None:
        """取消执行"""
        record = self.store.load_execution(execution_id)
        if not record:
            return
        
        record.state = WorkflowState.CANCELLED
        record.end_time = time.time()
        
        self.store.save_execution(record)
        logger.info(f"Cancelled execution: {execution_id}")
    
    def create_checkpoint(
        self,
        execution_id: str,
        step_id: Optional[str] = None,
        context_override: Optional[Dict[str, Any]] = None
    ) -> Checkpoint:
        """创建检查点"""
        import uuid
        
        record = self.store.load_execution(execution_id)
        if not record:
            raise ValueError(f"Execution not found: {execution_id}")
        
        checkpoint_id = str(uuid.uuid4())
        
        # 构建步骤状态
        step_states = {}
        for step_id_completed in record.completed_steps:
            step_states[step_id_completed] = StepStatus.COMPLETED
        for step_id_failed in record.failed_steps:
            step_states[step_id_failed] = StepStatus.FAILED
        
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            workflow_id=record.workflow_id,
            execution_id=execution_id,
            step_id=step_id or record.current_step,
            context=context_override or record.context.copy(),
            step_states=step_states,
            metadata={
                "retry_count": record.retry_count,
                "checkpoint_strategy": self.checkpoint_strategy.value
            }
        )
        
        self.store.save_checkpoint(checkpoint)
        self._last_checkpoint[execution_id] = time.time()
        
        logger.debug(f"Created checkpoint: {checkpoint_id}")
        return checkpoint
    
    def restore_from_checkpoint(
        self,
        checkpoint_id: str
    ) -> Tuple[ExecutionRecord, Checkpoint]:
        """从检查点恢复"""
        checkpoint = self.store.load_checkpoint(checkpoint_id)
        if not checkpoint:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")
        
        record = self.store.load_execution(checkpoint.execution_id)
        if not record:
            raise ValueError(f"Execution not found: {checkpoint.execution_id}")
        
        # 恢复上下文
        record.context = checkpoint.context.copy()
        
        # 恢复步骤状态
        record.completed_steps = [
            step_id for step_id, status in checkpoint.step_states.items()
            if status == StepStatus.COMPLETED
        ]
        record.failed_steps = [
            step_id for step_id, status in checkpoint.step_states.items()
            if status == StepStatus.FAILED
        ]
        
        record.state = WorkflowState.PAUSED
        self.store.save_execution(record)
        
        logger.info(f"Restored from checkpoint: {checkpoint_id}")
        return record, checkpoint
    
    def get_execution_status(self, execution_id: str) -> Optional[WorkflowState]:
        """获取执行状态"""
        record = self.store.load_execution(execution_id)
        return record.state if record else None
    
    def list_executions(
        self,
        workflow_id: Optional[str] = None,
        state: Optional[WorkflowState] = None
    ) -> List[ExecutionRecord]:
        """列出执行记录"""
        # 这里简化实现，实际需要遍历存储
        return []
    
    def get_recovery_plan(
        self,
        execution_id: str
    ) -> Dict[str, Any]:
        """生成恢复计划"""
        record = self.store.load_execution(execution_id)
        if not record:
            raise ValueError(f"Execution not found: {execution_id}")
        
        # 找到最后失败的步骤
        last_failed = record.failed_steps[-1] if record.failed_steps else None
        
        # 获取检查点
        checkpoints = self.store.list_checkpoints(record.workflow_id)
        
        plan = {
            "execution_id": execution_id,
            "current_state": record.state.value,
            "completed_steps": record.completed_steps,
            "failed_steps": record.failed_steps,
            "last_failed_step": last_failed,
            "retry_count": record.retry_count,
            "available_checkpoints": [cp.checkpoint_id for cp in checkpoints],
            "recovery_options": []
        }
        
        # 生成恢复选项
        if record.state == WorkflowState.FAILED:
            plan["recovery_options"].append({
                "action": "retry_failed_step",
                "description": f"Retry failed step: {last_failed}"
            })
            
            if checkpoints:
                plan["recovery_options"].append({
                    "action": "rollback_to_checkpoint",
                    "description": f"Rollback to checkpoint: {checkpoints[0].checkpoint_id}"
                })
        
        return plan
    
    def periodic_checkpoint(self, execution_id: str) -> Optional[Checkpoint]:
        """定期创建检查点（如果策略允许）"""
        if self.checkpoint_strategy != CheckpointStrategy.PERIODIC:
            return None
        
        last = self._last_checkpoint.get(execution_id, 0)
        if time.time() - last >= self._checkpoint_interval:
            return self.create_checkpoint(execution_id)
        
        return None
    
    def cleanup_old_checkpoints(
        self,
        workflow_id: str,
        keep_count: int = 5
    ) -> int:
        """清理旧检查点"""
        checkpoints = self.store.list_checkpoints(workflow_id)
        checkpoints.sort(key=lambda x: x.timestamp, reverse=True)
        
        deleted = 0
        for cp in checkpoints[keep_count:]:
            if self.store.delete_checkpoint(cp.checkpoint_id):
                deleted += 1
        
        return deleted


class WorkflowRecovery:
    """
    工作流恢复器
    
    处理工作流失败后的恢复逻辑
    """
    
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
    
    def analyze_failure(
        self,
        execution_id: str
    ) -> Dict[str, Any]:
        """分析失败原因"""
        record = self.state_manager.store.load_execution(execution_id)
        if not record:
            return {"error": "Execution not found"}
        
        analysis = {
            "execution_id": execution_id,
            "failure_point": record.current_step,
            "error_message": record.error_message,
            "retry_count": record.retry_count,
            "failure_type": self._classify_failure(record.error_message),
            "suggestions": []
        }
        
        # 生成建议
        failure_type = analysis["failure_type"]
        
        if failure_type == "convergence":
            analysis["suggestions"].append("Increase maximum iterations")
            analysis["suggestions"].append("Adjust mixing parameters")
        elif failure_type == "memory":
            analysis["suggestions"].append("Reduce k-point grid density")
            analysis["suggestions"].append("Use iterative diagonalization")
        elif failure_type == "timeout":
            analysis["suggestions"].append("Increase timeout limit")
            analysis["suggestions"].append("Use checkpoint/restart")
        
        return analysis
    
    def attempt_recovery(
        self,
        execution_id: str,
        strategy: str = "retry"
    ) -> bool:
        """尝试恢复"""
        if strategy == "retry":
            return self._retry(execution_id)
        elif strategy == "rollback":
            return self._rollback(execution_id)
        elif strategy == "skip":
            return self._skip_step(execution_id)
        
        return False
    
    def _classify_failure(self, error_message: Optional[str]) -> str:
        """分类失败类型"""
        if not error_message:
            return "unknown"
        
        error_lower = error_message.lower()
        
        if "convergence" in error_lower or "scf" in error_lower:
            return "convergence"
        elif "memory" in error_lower or "alloc" in error_lower:
            return "memory"
        elif "timeout" in error_lower or "time" in error_lower:
            return "timeout"
        elif "input" in error_lower or "format" in error_lower:
            return "input"
        
        return "unknown"
    
    def _retry(self, execution_id: str) -> bool:
        """重试"""
        try:
            self.state_manager.resume_execution(execution_id)
            return True
        except Exception as e:
            logger.error(f"Retry failed: {e}")
            return False
    
    def _rollback(self, execution_id: str) -> bool:
        """回滚"""
        try:
            record = self.state_manager.store.load_execution(execution_id)
            if not record:
                return False
            
            checkpoints = self.state_manager.store.list_checkpoints(record.workflow_id)
            if not checkpoints:
                return False
            
            latest_checkpoint = max(checkpoints, key=lambda x: x.timestamp)
            self.state_manager.restore_from_checkpoint(latest_checkpoint.checkpoint_id)
            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def _skip_step(self, execution_id: str) -> bool:
        """跳过步骤"""
        try:
            record = self.state_manager.store.load_execution(execution_id)
            if not record:
                return False
            
            # 将失败的步骤标记为跳过
            if record.failed_steps:
                failed_step = record.failed_steps[-1]
                record.failed_steps.remove(failed_step)
                # 记录跳过的步骤
                record.context[f"skipped_{failed_step}"] = True
                
                record.state = WorkflowState.RUNNING
                self.state_manager.store.save_execution(record)
            
            return True
        except Exception as e:
            logger.error(f"Skip step failed: {e}")
            return False


# 便捷函数
def get_state_manager() -> StateManager:
    """获取全局状态管理器"""
    if not hasattr(get_state_manager, '_instance'):
        get_state_manager._instance = StateManager()
    return get_state_manager._instance


def create_checkpoint(execution_id: str, step_id: Optional[str] = None) -> Checkpoint:
    """便捷函数：创建检查点"""
    return get_state_manager().create_checkpoint(execution_id, step_id)


def resume_workflow(execution_id: str) -> ExecutionRecord:
    """便捷函数：恢复工作流"""
    return get_state_manager().resume_execution(execution_id)