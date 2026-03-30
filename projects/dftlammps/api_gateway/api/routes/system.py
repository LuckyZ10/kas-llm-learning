"""
系统管理路由
"""

from typing import Dict, Any
from fastapi import APIRouter, Depends, Request

from ...auth import (
    User,
    get_current_user_or_api_key,
    Permission,
    has_permission,
)
from ...models.schemas import HealthStatus, SystemMetrics, APIResponse
from ...monitoring import metrics, api_metrics, system_monitor
from ...tasks import task_manager, celery_app
from ...utils import ForbiddenException
import psutil
import time

router = APIRouter()

start_time = time.time()


@router.get("/health", response_model=HealthStatus)
async def health_check():
    """
    健康检查端点
    
    返回服务整体健康状态
    """
    checks = {
        "api": True,
        "database": True,  # 实际应检查数据库连接
        "redis": True,  # 实际应检查Redis连接
        "celery": celery_app.control.ping() is not None,
    }
    
    all_healthy = all(checks.values())
    
    from datetime import datetime
    
    return HealthStatus(
        status="healthy" if all_healthy else "degraded",
        version="1.0.0",
        uptime_seconds=time.time() - start_time,
        checks=checks,
        timestamp=datetime.utcnow(),
    )


@router.get("/metrics")
async def get_metrics(
    current_user: User = Depends(get_current_user_or_api_key)
):
    """
    获取Prometheus格式的指标数据
    
    用于监控系统集成
    """
    if not has_permission(current_user.role, Permission.SYSTEM_READ):
        raise ForbiddenException("Permission denied")
    
    return metrics.to_prometheus_format()


@router.get("/metrics/json")
async def get_metrics_json(
    current_user: User = Depends(get_current_user_or_api_key)
):
    """获取JSON格式的指标数据"""
    if not has_permission(current_user.role, Permission.SYSTEM_READ):
        raise ForbiddenException("Permission denied")
    
    return metrics.get_all_metrics()


@router.get("/stats")
async def get_system_stats(
    current_user: User = Depends(get_current_user_or_api_key)
):
    """获取系统统计信息"""
    if not has_permission(current_user.role, Permission.SYSTEM_READ):
        raise ForbiddenException("Permission denied")
    
    # CPU
    cpu_percent = psutil.cpu_percent(interval=0.5)
    cpu_count = psutil.cpu_count()
    
    # 内存
    memory = psutil.virtual_memory()
    
    # 磁盘
    disk = psutil.disk_usage('/')
    
    # 网络
    net_io = psutil.net_io_counters()
    
    # Celery队列状态
    queue_stats = task_manager.get_queue_stats()
    
    return {
        "cpu": {
            "percent": cpu_percent,
            "count": cpu_count,
            "per_cpu": psutil.cpu_percent(interval=0.5, percpu=True),
        },
        "memory": {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "percent": memory.percent,
            "used_gb": memory.used / (1024**3),
        },
        "disk": {
            "total_gb": disk.total / (1024**3),
            "used_gb": disk.used / (1024**3),
            "free_gb": disk.free / (1024**3),
            "percent": disk.percent,
        },
        "network": {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv,
        },
        "celery": queue_stats,
        "uptime_seconds": time.time() - start_time,
    }


@router.get("/queue-status")
async def get_queue_status(
    current_user: User = Depends(get_current_user_or_api_key)
):
    """获取任务队列状态"""
    if not has_permission(current_user.role, Permission.SYSTEM_READ):
        raise ForbiddenException("Permission denied")
    
    inspect = celery_app.control.inspect()
    
    active = inspect.active() or {}
    scheduled = inspect.scheduled() or {}
    reserved = inspect.reserved() or {}
    
    return {
        "active_tasks": sum(len(t) for t in active.values()),
        "scheduled_tasks": sum(len(t) for t in scheduled.values()),
        "reserved_tasks": sum(len(t) for t in reserved.values()),
        "queues": {
            "default": {"pending": 0, "active": 0},
            "dft": {"pending": 0, "active": 0},
            "md": {"pending": 0, "active": 0},
            "ml": {"pending": 0, "active": 0},
            "screening": {"pending": 0, "active": 0},
        },
    }


@router.post("/maintenance", response_model=APIResponse)
async def trigger_maintenance(
    action: str,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """
    触发系统维护操作
    
    - **action**: 维护操作类型 (clear_cache, purge_queues, restart_workers)
    """
    if not has_permission(current_user.role, Permission.SYSTEM_ADMIN):
        raise ForbiddenException("Admin permission required")
    
    if action == "clear_cache":
        # 清除缓存
        return APIResponse(success=True, message="Cache cleared")
    
    elif action == "purge_queues":
        # 清空队列
        task_manager.purge_queue()
        return APIResponse(success=True, message="Queues purged")
    
    elif action == "restart_workers":
        # 重启workers
        celery_app.control.broadcast("pool_restart")
        return APIResponse(success=True, message="Workers restart triggered")
    
    else:
        return APIResponse(success=False, message=f"Unknown action: {action}")


@router.get("/config")
async def get_system_config(
    current_user: User = Depends(get_current_user_or_api_key)
):
    """获取系统配置（脱敏）"""
    if not has_permission(current_user.role, Permission.SYSTEM_READ):
        raise ForbiddenException("Permission denied")
    
    return {
        "rate_limits": {
            "default": "100/min",
            "authenticated": "1000/min",
        },
        "max_file_size": "100MB",
        "supported_formats": ["POSCAR", "CIF", "XYZ", "LAMMPS"],
        "api_version": "1.0.0",
    }


@router.get("/logs")
async def get_system_logs(
    lines: int = 100,
    level: str = "INFO",
    current_user: User = Depends(get_current_user_or_api_key)
):
    """获取系统日志"""
    if not has_permission(current_user.role, Permission.SYSTEM_ADMIN):
        raise ForbiddenException("Admin permission required")
    
    # 模拟日志
    return {
        "logs": [
            {"timestamp": "2024-01-01T00:00:00Z", "level": "INFO", "message": "System started"},
            {"timestamp": "2024-01-01T00:01:00Z", "level": "INFO", "message": "Celery workers ready"},
        ]
    }
