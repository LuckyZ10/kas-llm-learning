"""
任务管理路由
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query, Request

from ...auth import (
    User,
    get_current_user_or_api_key,
    Permission,
    has_permission,
    audit_logger,
)
from ...models.schemas import (
    TaskCreate,
    TaskUpdate,
    TaskResponse,
    TaskListResponse,
    TaskQueueStatus,
    TaskStatus,
    TaskType,
    TaskPriority,
    APIResponse,
    PaginatedResponse,
)
from ...tasks import task_manager, celery_app
from ...utils import (
    NotFoundException,
    ForbiddenException,
    TaskNotFoundException,
    ValidationException,
)
from ...monitoring import api_metrics

router = APIRouter()

# 模拟任务存储
TASKS_DB = {}


@router.get("", response_model=PaginatedResponse)
async def list_tasks(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[TaskStatus] = None,
    task_type: Optional[TaskType] = None,
    search: Optional[str] = None,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """
    获取任务列表
    
    - **page**: 页码
    - **page_size**: 每页数量
    - **status**: 按状态筛选
    - **task_type**: 按类型筛选
    - **search**: 搜索关键词
    """
    # 获取用户的任务
    user_tasks = [
        t for t in TASKS_DB.values()
        if t.get("created_by") == current_user.id
    ]
    
    # 过滤
    if status:
        user_tasks = [t for t in user_tasks if t.get("status") == status]
    if task_type:
        user_tasks = [t for t in user_tasks if t.get("task_type") == task_type]
    if search:
        user_tasks = [t for t in user_tasks if search.lower() in str(t.get("name", "")).lower()]
    
    # 分页
    total = len(user_tasks)
    start = (page - 1) * page_size
    end = start + page_size
    
    return PaginatedResponse(
        success=True,
        data=user_tasks[start:end],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
        has_next=end < total,
        has_prev=page > 1,
    )


@router.post("", response_model=TaskResponse)
async def create_task(
    request: Request,
    task_data: TaskCreate,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """
    创建新任务
    
    根据task_type自动路由到相应的工作队列
    """
    if not has_permission(current_user.role, Permission.TASK_CREATE):
        raise ForbiddenException("Permission denied")
    
    import datetime
    from ...utils import generate_task_id
    
    task_id = generate_task_id()
    
    # 创建任务记录
    task = {
        "id": task_id,
        "name": task_data.name,
        "task_type": task_data.task_type,
        "status": TaskStatus.QUEUED,
        "priority": task_data.priority,
        "progress": 0.0,
        "created_by": current_user.id,
        "created_at": datetime.datetime.utcnow(),
        "started_at": None,
        "completed_at": None,
        "result": None,
        "error_message": None,
        "execution_time": None,
    }
    
    TASKS_DB[task_id] = task
    
    # 记录审计日志
    await audit_logger.log_action(
        user_id=current_user.id,
        action="create_task",
        resource_type="task",
        resource_id=task_id,
        details={"task_type": task_data.task_type.value, "name": task_data.name},
        ip_address=request.client.host if request.client else None,
    )
    
    # 记录指标
    api_metrics.record_task_submission(task_data.task_type.value, str(current_user.id))
    
    return TaskResponse(**task)


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: str,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """获取任务详情"""
    task = TASKS_DB.get(task_id)
    
    if not task:
        # 尝试从Celery获取
        celery_result = celery_app.AsyncResult(task_id)
        if celery_result.state != "PENDING":
            return TaskResponse(
                id=task_id,
                name="Unknown",
                task_type=TaskType.ANALYSIS,
                status=_map_celery_status(celery_result.state),
                priority=TaskPriority.NORMAL,
                progress=0.0,
                created_by=current_user.id,
                created_at=None,
                started_at=None,
                completed_at=None,
                result=celery_result.result if celery_result.ready() else None,
                error_message=None,
                execution_time=None,
            )
        raise TaskNotFoundException(task_id)
    
    # 检查权限
    if task["created_by"] != current_user.id and not has_permission(current_user.role, Permission.TASK_READ):
        raise ForbiddenException("Access denied")
    
    return TaskResponse(**task)


@router.put("/{task_id}", response_model=TaskResponse)
async def update_task(
    task_id: str,
    task_data: TaskUpdate,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """更新任务"""
    task = TASKS_DB.get(task_id)
    
    if not task:
        raise TaskNotFoundException(task_id)
    
    if task["created_by"] != current_user.id and not has_permission(current_user.role, Permission.TASK_UPDATE):
        raise ForbiddenException("Permission denied")
    
    # 更新字段
    if task_data.name:
        task["name"] = task_data.name
    if task_data.priority:
        task["priority"] = task_data.priority
    
    TASKS_DB[task_id] = task
    
    return TaskResponse(**task)


@router.delete("/{task_id}", response_model=APIResponse)
async def cancel_task(
    task_id: str,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """
    取消任务
    
    只能取消pending或running状态的任务
    """
    task = TASKS_DB.get(task_id)
    
    if not task:
        raise TaskNotFoundException(task_id)
    
    if task["created_by"] != current_user.id and not has_permission(current_user.role, Permission.TASK_CANCEL):
        raise ForbiddenException("Permission denied")
    
    # 检查状态
    if task["status"] not in [TaskStatus.PENDING, TaskStatus.QUEUED, TaskStatus.RUNNING]:
        raise ValidationException(f"Cannot cancel task with status: {task['status']}")
    
    # 取消Celery任务
    task_manager.revoke_task(task_id, terminate=True)
    
    # 更新状态
    task["status"] = TaskStatus.CANCELLED
    TASKS_DB[task_id] = task
    
    await audit_logger.log_action(
        user_id=current_user.id,
        action="cancel_task",
        resource_type="task",
        resource_id=task_id,
    )
    
    return APIResponse(
        success=True,
        message=f"Task {task_id} cancelled"
    )


@router.get("/{task_id}/status")
async def get_task_status(
    task_id: str,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """获取任务状态（轻量级查询）"""
    task = TASKS_DB.get(task_id)
    
    if not task:
        # 从Celery获取
        result = task_manager.get_task_status(task_id)
        return result
    
    if task["created_by"] != current_user.id and not has_permission(current_user.role, Permission.TASK_READ):
        raise ForbiddenException("Permission denied")
    
    return {
        "task_id": task_id,
        "status": task["status"],
        "progress": task["progress"],
    }


@router.get("/{task_id}/result")
async def get_task_result(
    task_id: str,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """获取任务结果"""
    task = TASKS_DB.get(task_id)
    
    if not task:
        # 从Celery获取
        celery_result = celery_app.AsyncResult(task_id)
        if celery_result.ready():
            return {
                "task_id": task_id,
                "status": celery_result.state,
                "result": celery_result.result,
            }
        raise TaskNotFoundException(task_id)
    
    if task["created_by"] != current_user.id and not has_permission(current_user.role, Permission.TASK_READ):
        raise ForbiddenException("Permission denied")
    
    if task["status"] != TaskStatus.COMPLETED:
        raise ValidationException(f"Task not completed. Current status: {task['status']}")
    
    return {
        "task_id": task_id,
        "status": task["status"],
        "result": task["result"],
    }


@router.get("/queue/status", response_model=List[TaskQueueStatus])
async def get_queue_status(
    current_user: User = Depends(get_current_user_or_api_key)
):
    """获取任务队列状态"""
    stats = task_manager.get_queue_stats()
    
    queues = []
    for queue_name in ["default", "dft", "md", "ml", "screening", "analysis"]:
        queues.append(TaskQueueStatus(
            queue_name=queue_name,
            pending_count=0,  # 从stats解析
            running_count=0,
            completed_count=0,
            failed_count=0,
            avg_wait_time=0.0,
            avg_execution_time=0.0,
        ))
    
    return queues


def _map_celery_status(celery_status: str) -> TaskStatus:
    """映射Celery状态到TaskStatus"""
    mapping = {
        "PENDING": TaskStatus.PENDING,
        "RECEIVED": TaskStatus.QUEUED,
        "STARTED": TaskStatus.RUNNING,
        "SUCCESS": TaskStatus.COMPLETED,
        "FAILURE": TaskStatus.FAILED,
        "REVOKED": TaskStatus.CANCELLED,
        "RETRY": TaskStatus.PENDING,
    }
    return mapping.get(celery_status, TaskStatus.PENDING)
