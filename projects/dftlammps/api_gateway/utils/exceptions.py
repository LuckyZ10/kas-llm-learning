"""
异常处理
"""

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
    HTTP_409_CONFLICT,
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_429_TOO_MANY_REQUESTS,
    HTTP_500_INTERNAL_SERVER_ERROR,
)
from datetime import datetime
from typing import Any, Dict, Optional
import logging
import traceback

logger = logging.getLogger(__name__)


class APIException(Exception):
    """API异常基类"""
    
    def __init__(
        self,
        message: str,
        status_code: int = HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class BadRequestException(APIException):
    """错误请求异常"""
    
    def __init__(self, message: str = "Bad request", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=HTTP_400_BAD_REQUEST,
            error_code="BAD_REQUEST",
            details=details
        )


class UnauthorizedException(APIException):
    """未认证异常"""
    
    def __init__(self, message: str = "Authentication required"):
        super().__init__(
            message=message,
            status_code=HTTP_401_UNAUTHORIZED,
            error_code="UNAUTHORIZED"
        )


class ForbiddenException(APIException):
    """无权限异常"""
    
    def __init__(self, message: str = "Permission denied"):
        super().__init__(
            message=message,
            status_code=HTTP_403_FORBIDDEN,
            error_code="FORBIDDEN"
        )


class NotFoundException(APIException):
    """未找到异常"""
    
    def __init__(self, resource: str = "Resource", resource_id: Optional[str] = None):
        message = f"{resource} not found"
        if resource_id:
            message += f": {resource_id}"
        
        super().__init__(
            message=message,
            status_code=HTTP_404_NOT_FOUND,
            error_code="NOT_FOUND",
            details={"resource": resource, "id": resource_id}
        )


class ConflictException(APIException):
    """冲突异常"""
    
    def __init__(self, message: str = "Resource conflict"):
        super().__init__(
            message=message,
            status_code=HTTP_409_CONFLICT,
            error_code="CONFLICT"
        )


class ValidationException(APIException):
    """验证异常"""
    
    def __init__(self, message: str = "Validation error", errors: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            details={"errors": errors} if errors else None
        )


class RateLimitException(APIException):
    """限流异常"""
    
    def __init__(self, retry_after: int = 60):
        super().__init__(
            message="Rate limit exceeded",
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            error_code="RATE_LIMIT_EXCEEDED",
            details={"retry_after": retry_after}
        )
        self.retry_after = retry_after


class TaskNotFoundException(NotFoundException):
    """任务未找到异常"""
    
    def __init__(self, task_id: str):
        super().__init__(resource="Task", resource_id=task_id)


class CalculationException(APIException):
    """计算异常"""
    
    def __init__(self, message: str = "Calculation failed", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="CALCULATION_ERROR",
            details=details
        )


def create_error_response(
    exception: APIException,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """创建错误响应"""
    return {
        "success": False,
        "message": exception.message,
        "error_code": exception.error_code,
        "details": exception.details,
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": request_id,
    }


async def api_exception_handler(request: Request, exc: APIException):
    """API异常处理器"""
    request_id = getattr(request.state, "request_id", None)
    
    logger.error(
        f"API Exception: {exc.error_code} - {exc.message}",
        extra={"request_id": request_id, "details": exc.details}
    )
    
    response = create_error_response(exc, request_id)
    
    headers = {}
    if isinstance(exc, RateLimitException):
        headers["Retry-After"] = str(exc.retry_after)
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response,
        headers=headers
    )


async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理器"""
    request_id = getattr(request.state, "request_id", None)
    
    logger.warning(
        f"HTTP Exception: {exc.status_code} - {exc.detail}",
        extra={"request_id": request_id}
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": str(exc.detail),
            "error_code": f"HTTP_{exc.status_code}",
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
        }
    )


async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理器"""
    request_id = getattr(request.state, "request_id", None)
    
    logger.exception(
        f"Unhandled exception: {str(exc)}",
        extra={"request_id": request_id}
    )
    
    # 只在调试模式返回详细错误信息
    details = None
    # if settings.DEBUG:  # 可以添加调试模式检查
    #     details = {"traceback": traceback.format_exc()}
    
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "message": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "details": details,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
        }
    )


def setup_exception_handlers(app):
    """设置异常处理器"""
    app.add_exception_handler(APIException, api_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
    
    logger.info("Exception handlers registered")
