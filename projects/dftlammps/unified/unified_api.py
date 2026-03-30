"""
DFT+LAMMPS Unified API Layer
============================
统一API层 - 所有模块的统一接口、自动路由与分发、版本兼容性管理

功能：
1. 统一模块接口定义
2. 自动服务发现与路由
3. API版本管理
4. 请求/响应标准化
5. 中间件支持
"""

import json
import time
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Type, Union, TypeVar, Generic
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
import uuid
from datetime import datetime
import functools
import inspect
import importlib
import pkgutil
from collections import defaultdict
import logging

from .common import (
    DFTLAMMPSError, APIError, ValidationError, 
    get_logger, generate_id, log_execution, handle_errors
)
from .config_system import ConfigManager, GlobalConfig, ConfigBuilder

logger = get_logger("unified_api")

T = TypeVar('T')


# =============================================================================
# API 类型定义
# =============================================================================

class APIVersion(Enum):
    """API版本"""
    V1 = "1.0"
    V2 = "2.0"
    V3 = "3.0"
    LATEST = "2.0"


class HTTPMethod(Enum):
    """HTTP方法"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class ResponseStatus(Enum):
    """响应状态"""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    CANCELLED = "cancelled"


@dataclass
class APIRequest:
    """API请求对象"""
    path: str
    method: HTTPMethod = HTTPMethod.GET
    params: Dict[str, Any] = field(default_factory=dict)
    body: Optional[Any] = None
    headers: Dict[str, str] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: generate_id("req"))
    timestamp: float = field(default_factory=time.time)
    version: str = APIVersion.LATEST.value
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "request_id": self.request_id,
            "path": self.path,
            "method": self.method.value,
            "params": self.params,
            "body": self.body,
            "headers": self.headers,
            "timestamp": self.timestamp,
            "version": self.version
        }


@dataclass
class APIResponse:
    """API响应对象"""
    status: ResponseStatus
    data: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    request_id: str = ""
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def success(cls, data: Any = None, **kwargs) -> 'APIResponse':
        """创建成功响应"""
        return cls(
            status=ResponseStatus.SUCCESS,
            data=data,
            **kwargs
        )
    
    @classmethod
    def error(cls, message: str, code: str = "", details: Dict = None, **kwargs) -> 'APIResponse':
        """创建错误响应"""
        return cls(
            status=ResponseStatus.ERROR,
            error={
                "message": message,
                "code": code,
                "details": details or {}
            },
            **kwargs
        )
    
    @classmethod
    def pending(cls, task_id: str, **kwargs) -> 'APIResponse':
        """创建挂起响应"""
        return cls(
            status=ResponseStatus.PENDING,
            data={"task_id": task_id},
            **kwargs
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "status": self.status.value,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
        }
        if self.data is not None:
            result["data"] = self.data
        if self.error:
            result["error"] = self.error
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class RouteInfo:
    """路由信息"""
    path: str
    method: HTTPMethod
    handler: Callable
    name: str = ""
    version: str = APIVersion.LATEST.value
    middlewares: List[Callable] = field(default_factory=list)
    doc: str = ""
    auth_required: bool = False
    rate_limit: Optional[int] = None


# =============================================================================
# 中间件基类
# =============================================================================

class Middleware(ABC):
    """中间件基类"""
    
    @abstractmethod
    async def process_request(self, request: APIRequest) -> APIRequest:
        """处理请求"""
        pass
    
    @abstractmethod
    async def process_response(self, response: APIResponse, request: APIRequest) -> APIResponse:
        """处理响应"""
        pass


class AuthenticationMiddleware(Middleware):
    """认证中间件"""
    
    def __init__(self, token_validator: Optional[Callable] = None):
        self.token_validator = token_validator
        self._tokens: Dict[str, Dict[str, Any]] = {}
    
    async def process_request(self, request: APIRequest) -> APIRequest:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise APIError("Missing or invalid authorization header", status_code=401)
        
        token = auth_header[7:]
        if self.token_validator and not self.token_validator(token):
            raise APIError("Invalid token", status_code=401)
        
        return request
    
    async def process_response(self, response: APIResponse, request: APIRequest) -> APIResponse:
        return response


class RateLimitMiddleware(Middleware):
    """限流中间件"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self._requests: Dict[str, List[float]] = defaultdict(list)
    
    async def process_request(self, request: APIRequest) -> APIRequest:
        client_id = request.headers.get("X-Client-ID", "anonymous")
        now = time.time()
        
        # 清理旧请求记录
        self._requests[client_id] = [
            t for t in self._requests[client_id] 
            if now - t < 60
        ]
        
        if len(self._requests[client_id]) >= self.requests_per_minute:
            raise APIError("Rate limit exceeded", status_code=429)
        
        self._requests[client_id].append(now)
        return request
    
    async def process_response(self, response: APIResponse, request: APIRequest) -> APIResponse:
        return response


class LoggingMiddleware(Middleware):
    """日志中间件"""
    
    async def process_request(self, request: APIRequest) -> APIRequest:
        logger.info(f"API Request: {request.method.value} {request.path} [{request.request_id}]")
        return request
    
    async def process_response(self, response: APIResponse, request: APIRequest) -> APIResponse:
        logger.info(
            f"API Response: {response.status.value} "
            f"[{request.request_id}] {response.duration_ms:.2f}ms"
        )
        return response


class CORSMiddleware(Middleware):
    """CORS中间件"""
    
    def __init__(self, allowed_origins: List[str] = None):
        self.allowed_origins = allowed_origins or ["*"]
    
    async def process_request(self, request: APIRequest) -> APIRequest:
        return request
    
    async def process_response(self, response: APIResponse, request: APIRequest) -> APIResponse:
        response.metadata["cors_headers"] = {
            "Access-Control-Allow-Origin": ",".join(self.allowed_origins),
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, PATCH, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization"
        }
        return response


# =============================================================================
# 模块接口基类
# =============================================================================

class ModuleInterface(ABC):
    """模块接口基类 - 所有模块必须实现"""
    
    # 模块元数据
    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    dependencies: List[str] = []
    
    def __init__(self, config: Optional[GlobalConfig] = None):
        self.config = config
        self._initialized = False
        self._logger = get_logger(f"module.{self.name}")
    
    @abstractmethod
    async def initialize(self) -> None:
        """初始化模块"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """关闭模块"""
        pass
    
    @abstractmethod
    def get_routes(self) -> List[RouteInfo]:
        """获取模块路由列表"""
        pass
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            "status": "healthy" if self._initialized else "uninitialized",
            "module": self.name,
            "version": self.version
        }
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()


# =============================================================================
# 统一模块注册表
# =============================================================================

class ModuleRegistry:
    """模块注册表 - 管理所有模块"""
    
    def __init__(self):
        self._modules: Dict[str, ModuleInterface] = {}
        self._routes: Dict[str, RouteInfo] = {}
        self._dependencies: Dict[str, List[str]] = {}
    
    def register(self, module: ModuleInterface) -> None:
        """注册模块"""
        if module.name in self._modules:
            logger.warning(f"Module {module.name} already registered, overwriting")
        
        self._modules[module.name] = module
        self._dependencies[module.name] = module.dependencies
        
        # 注册路由
        for route in module.get_routes():
            route_key = f"{route.method.value}:{route.path}"
            self._routes[route_key] = route
        
        logger.info(f"Registered module: {module.name} v{module.version}")
    
    def unregister(self, name: str) -> None:
        """注销模块"""
        if name in self._modules:
            del self._modules[name]
            del self._dependencies[name]
            # 清理路由
            routes_to_remove = [
                k for k, r in self._routes.items() 
                if r.name == name
            ]
            for key in routes_to_remove:
                del self._routes[key]
    
    def get_module(self, name: str) -> Optional[ModuleInterface]:
        """获取模块"""
        return self._modules.get(name)
    
    def get_all_modules(self) -> Dict[str, ModuleInterface]:
        """获取所有模块"""
        return self._modules.copy()
    
    def get_route(self, method: str, path: str) -> Optional[RouteInfo]:
        """获取路由"""
        # 精确匹配
        route_key = f"{method}:{path}"
        if route_key in self._routes:
            return self._routes[route_key]
        
        # 模式匹配
        for key, route in self._routes.items():
            if self._match_path(route.path, path) and key.startswith(method):
                return route
        
        return None
    
    def _match_path(self, pattern: str, path: str) -> bool:
        """匹配路径模式"""
        pattern_parts = pattern.split("/")
        path_parts = path.split("/")
        
        if len(pattern_parts) != len(path_parts):
            return False
        
        for p_part, path_part in zip(pattern_parts, path_parts):
            if p_part.startswith("{") and p_part.endswith("}"):
                continue  # 路径参数
            if p_part != path_part:
                return False
        
        return True
    
    def get_dependency_order(self) -> List[str]:
        """获取依赖顺序"""
        visited = set()
        order = []
        
        def visit(name: str, stack: set):
            if name in stack:
                raise ValueError(f"Circular dependency detected: {name}")
            if name in visited:
                return
            
            stack.add(name)
            for dep in self._dependencies.get(name, []):
                if dep in self._modules:
                    visit(dep, stack)
            stack.remove(name)
            
            visited.add(name)
            order.append(name)
        
        for name in self._modules:
            visit(name, set())
        
        return order
    
    def list_routes(self) -> List[RouteInfo]:
        """列出所有路由"""
        return list(self._routes.values())


# =============================================================================
# 统一API路由器
# =============================================================================

class UnifiedAPIRouter:
    """
    统一API路由器
    
    功能：
    - 自动服务发现
    - 智能路由分发
    - 版本兼容性管理
    - 中间件链处理
    """
    
    def __init__(self, config: Optional[GlobalConfig] = None):
        self.config = config
        self.registry = ModuleRegistry()
        self.middlewares: List[Middleware] = []
        self._version_adapters: Dict[str, Callable] = {}
        self._initialized = False
        
        # 默认中间件
        self.add_middleware(LoggingMiddleware())
    
    def add_middleware(self, middleware: Middleware) -> None:
        """添加中间件"""
        self.middlewares.append(middleware)
    
    def register_module(self, module: ModuleInterface) -> None:
        """注册模块"""
        self.registry.register(module)
    
    def register_version_adapter(self, version: str, adapter: Callable) -> None:
        """注册版本适配器"""
        self._version_adapters[version] = adapter
    
    async def initialize(self) -> None:
        """初始化所有模块"""
        if self._initialized:
            return
        
        # 按依赖顺序初始化
        for name in self.registry.get_dependency_order():
            module = self.registry.get_module(name)
            if module:
                try:
                    await module.initialize()
                    logger.info(f"Initialized module: {name}")
                except Exception as e:
                    logger.error(f"Failed to initialize module {name}: {e}")
                    raise
        
        self._initialized = True
    
    async def shutdown(self) -> None:
        """关闭所有模块"""
        for name in reversed(self.registry.get_dependency_order()):
            module = self.registry.get_module(name)
            if module:
                try:
                    await module.shutdown()
                    logger.info(f"Shutdown module: {name}")
                except Exception as e:
                    logger.error(f"Error shutting down module {name}: {e}")
        
        self._initialized = False
    
    async def route(self, request: APIRequest) -> APIResponse:
        """
        路由请求
        
        Args:
            request: API请求对象
            
        Returns:
            APIResponse: 响应对象
        """
        start_time = time.time()
        
        try:
            # 执行请求中间件
            for middleware in self.middlewares:
                request = await middleware.process_request(request)
            
            # 查找路由
            route = self.registry.get_route(
                request.method.value, 
                request.path
            )
            
            if not route:
                response = APIResponse.error(
                    f"Route not found: {request.method.value} {request.path}",
                    code="ROUTE_NOT_FOUND",
                    request_id=request.request_id
                )
            else:
                # 版本适配
                if request.version != route.version:
                    request = self._adapt_request_version(request, route.version)
                
                # 执行处理函数
                handler_result = await self._execute_handler(route.handler, request)
                
                if isinstance(handler_result, APIResponse):
                    response = handler_result
                else:
                    response = APIResponse.success(
                        data=handler_result,
                        request_id=request.request_id
                    )
            
            # 执行响应中间件
            for middleware in self.middlewares:
                response = await middleware.process_response(response, request)
        
        except APIError as e:
            response = APIResponse.error(
                message=e.message,
                code=e.error_code,
                request_id=request.request_id
            )
        except Exception as e:
            logger.exception("Unexpected error in route handler")
            response = APIResponse.error(
                message=str(e),
                code="INTERNAL_ERROR",
                request_id=request.request_id
            )
        
        # 设置耗时
        response.duration_ms = (time.time() - start_time) * 1000
        response.request_id = request.request_id
        
        return response
    
    async def _execute_handler(self, handler: Callable, request: APIRequest) -> Any:
        """执行处理函数"""
        sig = inspect.signature(handler)
        params = list(sig.parameters.keys())
        
        if len(params) == 0:
            return await handler() if asyncio.iscoroutinefunction(handler) else handler()
        elif params[0] in ('self', 'cls'):
            # 类方法，需要实例
            return await handler(request) if asyncio.iscoroutinefunction(handler) else handler(request)
        else:
            return await handler(request) if asyncio.iscoroutinefunction(handler) else handler(request)
    
    def _adapt_request_version(self, request: APIRequest, target_version: str) -> APIRequest:
        """适配请求版本"""
        adapter = self._version_adapters.get(request.version)
        if adapter:
            return adapter(request, target_version)
        return request
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        modules_health = {}
        for name, module in self.registry.get_all_modules().items():
            modules_health[name] = module.health_check()
        
        return {
            "status": "healthy" if self._initialized else "initializing",
            "modules": modules_health,
            "total_modules": len(modules_health),
            "routes_count": len(self.registry.list_routes())
        }
    
    def generate_api_docs(self) -> Dict[str, Any]:
        """生成API文档"""
        routes = self.registry.list_routes()
        
        docs = {
            "version": APIVersion.LATEST.value,
            "generated_at": datetime.now().isoformat(),
            "routes": []
        }
        
        for route in routes:
            route_doc = {
                "path": route.path,
                "method": route.method.value,
                "name": route.name,
                "version": route.version,
                "auth_required": route.auth_required,
                "description": route.doc
            }
            docs["routes"].append(route_doc)
        
        return docs


# =============================================================================
# 便捷API装饰器
# =============================================================================

def api_route(path: str, 
             method: HTTPMethod = HTTPMethod.GET,
             version: str = APIVersion.LATEST.value,
             auth_required: bool = False,
             rate_limit: Optional[int] = None,
             doc: str = ""):
    """API路由装饰器"""
    def decorator(func: Callable) -> Callable:
        func._api_route = RouteInfo(
            path=path,
            method=method,
            handler=func,
            name=func.__name__,
            version=version,
            auth_required=auth_required,
            rate_limit=rate_limit,
            doc=doc or func.__doc__ or ""
        )
        return func
    return decorator


def get(path: str, **kwargs):
    """GET路由装饰器"""
    return api_route(path, HTTPMethod.GET, **kwargs)


def post(path: str, **kwargs):
    """POST路由装饰器"""
    return api_route(path, HTTPMethod.POST, **kwargs)


def put(path: str, **kwargs):
    """PUT路由装饰器"""
    return api_route(path, HTTPMethod.PUT, **kwargs)


def delete(path: str, **kwargs):
    """DELETE路由装饰器"""
    return api_route(path, HTTPMethod.DELETE, **kwargs)


# =============================================================================
# 标准模块实现示例
# =============================================================================

class HealthModule(ModuleInterface):
    """健康检查模块"""
    
    name = "health"
    version = "1.0.0"
    description = "System health check module"
    
    def __init__(self, router: UnifiedAPIRouter):
        super().__init__()
        self.router = router
    
    async def initialize(self) -> None:
        self._initialized = True
    
    async def shutdown(self) -> None:
        self._initialized = False
    
    def get_routes(self) -> List[RouteInfo]:
        return [
            RouteInfo(
                path="/health",
                method=HTTPMethod.GET,
                handler=self.check_health,
                name="health_check",
                doc="Get system health status"
            ),
            RouteInfo(
                path="/health/modules",
                method=HTTPMethod.GET,
                handler=self.list_modules,
                name="list_modules",
                doc="List all registered modules"
            )
        ]
    
    async def check_health(self, request: APIRequest) -> Dict[str, Any]:
        """健康检查"""
        return self.router.get_health_status()
    
    async def list_modules(self, request: APIRequest) -> List[Dict[str, Any]]:
        """列出模块"""
        modules = []
        for name, module in self.router.registry.get_all_modules().items():
            modules.append({
                "name": name,
                "version": module.version,
                "description": module.description,
                "health": module.health_check()
            })
        return modules


class DocsModule(ModuleInterface):
    """API文档模块"""
    
    name = "docs"
    version = "1.0.0"
    description = "API documentation module"
    
    def __init__(self, router: UnifiedAPIRouter):
        super().__init__()
        self.router = router
    
    async def initialize(self) -> None:
        self._initialized = True
    
    async def shutdown(self) -> None:
        self._initialized = False
    
    def get_routes(self) -> List[RouteInfo]:
        return [
            RouteInfo(
                path="/api/docs",
                method=HTTPMethod.GET,
                handler=self.get_docs,
                name="api_docs",
                doc="Get API documentation"
            ),
            RouteInfo(
                path="/api/routes",
                method=HTTPMethod.GET,
                handler=self.list_routes,
                name="list_routes",
                doc="List all API routes"
            )
        ]
    
    async def get_docs(self, request: APIRequest) -> Dict[str, Any]:
        """获取API文档"""
        return self.router.generate_api_docs()
    
    async def list_routes(self, request: APIRequest) -> List[Dict[str, Any]]:
        """列出路由"""
        routes = self.router.registry.list_routes()
        return [
            {
                "path": r.path,
                "method": r.method.value,
                "name": r.name,
                "version": r.version
            }
            for r in routes
        ]


# =============================================================================
# 全局API实例
# =============================================================================

# 全局路由器实例
_global_router: Optional[UnifiedAPIRouter] = None


def get_router(config: Optional[GlobalConfig] = None) -> UnifiedAPIRouter:
    """获取全局路由器"""
    global _global_router
    if _global_router is None:
        _global_router = UnifiedAPIRouter(config)
    return _global_router


def init_api(config: Optional[GlobalConfig] = None) -> UnifiedAPIRouter:
    """初始化API"""
    router = get_router(config)
    
    # 注册标准模块
    router.register_module(HealthModule(router))
    router.register_module(DocsModule(router))
    
    return router


# =============================================================================
# 便捷调用函数
# =============================================================================

async def call_api(path: str, 
                  method: str = "GET",
                  params: Optional[Dict] = None,
                  body: Optional[Any] = None,
                  headers: Optional[Dict] = None,
                  version: str = APIVersion.LATEST.value) -> APIResponse:
    """
    便捷API调用函数
    
    Args:
        path: API路径
        method: HTTP方法
        params: 查询参数
        body: 请求体
        headers: 请求头
        version: API版本
        
    Returns:
        APIResponse: 响应对象
    """
    request = APIRequest(
        path=path,
        method=HTTPMethod(method.upper()),
        params=params or {},
        body=body,
        headers=headers or {},
        version=version
    )
    
    router = get_router()
    return await router.route(request)


# 同步版本
def call_api_sync(path: str, **kwargs) -> APIResponse:
    """同步API调用"""
    return asyncio.run(call_api(path, **kwargs))
