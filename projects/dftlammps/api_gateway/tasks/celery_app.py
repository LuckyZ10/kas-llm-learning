"""
Celery任务队列配置
分布式任务处理
"""

from celery import Celery
from celery.signals import task_prerun, task_postrun, task_failure
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)

# Celery配置
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# 创建Celery应用
celery_app = Celery(
    "dftlammps",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        "dftlammps.api_gateway.tasks.dft_tasks",
        "dftlammps.api_gateway.tasks.md_tasks",
        "dftlammps.api_gateway.tasks.ml_tasks",
        "dftlammps.api_gateway.tasks.screening_tasks",
        "dftlammps.api_gateway.tasks.analysis_tasks",
    ]
)

# Celery配置
celery_app.conf.update(
    # 任务序列化
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    
    # 时区
    timezone="UTC",
    enable_utc=True,
    
    # 任务结果过期时间
    result_expires=3600 * 24 * 7,  # 7天
    
    # 任务跟踪
    task_track_started=True,
    task_time_limit=3600 * 24,  # 24小时硬限制
    task_soft_time_limit=3600 * 23,  # 23小时软限制
    
    # Worker配置
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    
    # 任务路由
    task_routes={
        "dftlammps.api_gateway.tasks.dft_tasks.*": {"queue": "dft"},
        "dftlammps.api_gateway.tasks.md_tasks.*": {"queue": "md"},
        "dftlammps.api_gateway.tasks.ml_tasks.*": {"queue": "ml"},
        "dftlammps.api_gateway.tasks.screening_tasks.*": {"queue": "screening"},
        "dftlammps.api_gateway.tasks.analysis_tasks.*": {"queue": "analysis"},
    },
    
    # 任务默认队列
    task_default_queue="default",
    
    # 速率限制
    task_annotations={
        "*": {
            "rate_limit": "100/m",
        }
    },
    
    # 重试配置
    task_default_retry_delay=60,  # 1分钟后重试
    task_max_retries=3,
    
    # 结果后端配置
    result_backend_max_retries=10,
    result_backend_always_retry=True,
)


@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **extras):
    """任务开始前的处理"""
    logger.info(f"Task {task.name}[{task_id}] started")
    
    # 更新任务状态到数据库或缓存
    try:
        from ..models.schemas import TaskStatus
        update_task_status(task_id, TaskStatus.RUNNING)
    except Exception as e:
        logger.error(f"Failed to update task status: {e}")


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **extras):
    """任务完成后的处理"""
    logger.info(f"Task {task.name}[{task_id}] finished with state: {state}")
    
    try:
        from ..models.schemas import TaskStatus
        
        if state == "SUCCESS":
            update_task_status(task_id, TaskStatus.COMPLETED, result=retval)
        elif state == "FAILURE":
            update_task_status(task_id, TaskStatus.FAILED, error=str(retval))
        elif state == "REVOKED":
            update_task_status(task_id, TaskStatus.CANCELLED)
    except Exception as e:
        logger.error(f"Failed to update task status: {e}")


@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, args=None, kwargs=None, traceback=None, einfo=None, **extras):
    """任务失败处理"""
    logger.error(f"Task {sender.name}[{task_id}] failed: {exception}")
    
    try:
        # 记录错误详情
        from ..models.schemas import TaskStatus
        update_task_status(
            task_id,
            TaskStatus.FAILED,
            error=str(exception),
            error_traceback=str(traceback) if traceback else None
        )
        
        # 可以在这里发送告警通知
        send_failure_notification(task_id, sender.name, str(exception))
    except Exception as e:
        logger.error(f"Failed to handle task failure: {e}")


def update_task_status(task_id: str, status: str, result=None, error=None, error_traceback=None):
    """更新任务状态"""
    # 这里应该更新到数据库
    # 简化版本，实际应使用数据库
    logger.info(f"Updating task {task_id} status to {status}")
    
    # TODO: 实现数据库更新逻辑
    # from ..database import db
    # await db.execute(
    #     "UPDATE tasks SET status = ?, result = ?, error = ? WHERE id = ?",
    #     (status, json.dumps(result) if result else None, error, task_id)
    # )


def send_failure_notification(task_id: str, task_name: str, error: str):
    """发送任务失败通知"""
    logger.warning(f"Task failure notification: {task_name}[{task_id}] - {error}")
    
    # TODO: 实现通知逻辑（邮件、Slack、Webhook等）
    # notification_service.send_alert(
    #     level="error",
    #     message=f"Task {task_name} failed",
    #     details={"task_id": task_id, "error": error}
    # )


class TaskManager:
    """任务管理器"""
    
    def __init__(self):
        self.celery = celery_app
    
    def submit_task(self, task_name: str, args=None, kwargs=None, queue=None, priority=5, countdown=0):
        """提交任务"""
        return self.celery.send_task(
            task_name,
            args=args,
            kwargs=kwargs,
            queue=queue,
            priority=priority,
            countdown=countdown,
        )
    
    def revoke_task(self, task_id: str, terminate=False):
        """取消任务"""
        self.celery.control.revoke(task_id, terminate=terminate)
    
    def get_task_status(self, task_id: str):
        """获取任务状态"""
        result = self.celery.AsyncResult(task_id)
        return {
            "task_id": task_id,
            "status": result.status,
            "result": result.result if result.ready() else None,
        }
    
    def get_queue_stats(self):
        """获取队列统计"""
        inspect = self.celery.control.inspect()
        
        stats = {
            "active": inspect.active(),
            "scheduled": inspect.scheduled(),
            "reserved": inspect.reserved(),
            "revoked": inspect.revoked(),
            "stats": inspect.stats(),
        }
        
        return stats
    
    def purge_queue(self, queue_name=None):
        """清空队列"""
        if queue_name:
            self.celery.control.purge(queue=queue_name)
        else:
            self.celery.control.purge()


# 全局任务管理器实例
task_manager = TaskManager()
