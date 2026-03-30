"""
DFT-LAMMPS API Gateway Module

API网关与服务化部署模块 - Phase 69

提供RESTful API服务、任务队列、认证授权、限流监控等功能
"""

__version__ = "1.0.0"
__author__ = "DFT-LAMMPS Team"

# 核心组件
from .api.main import create_app
from .auth.security import (
    create_access_token,
    verify_password,
    get_password_hash,
    authenticate_user,
    get_current_user,
    get_current_active_user,
    User,
    Token,
)
from .tasks.celery_app import celery_app
from .monitoring.metrics import MetricsCollector
from .monitoring.rate_limiter import RateLimiter

__all__ = [
    "create_app",
    "create_access_token",
    "verify_password",
    "get_password_hash",
    "authenticate_user",
    "get_current_user",
    "get_current_active_user",
    "User",
    "Token",
    "celery_app",
    "MetricsCollector",
    "RateLimiter",
]
