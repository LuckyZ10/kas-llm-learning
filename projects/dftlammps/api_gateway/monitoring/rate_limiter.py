"""
限流器模块
支持多种限流策略：固定窗口、滑动窗口、令牌桶
"""

import time
import threading
from typing import Dict, Optional, Callable
from collections import deque
from functools import wraps
from fastapi import HTTPException, status, Request
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """限流器基类"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._storage: Dict[str, any] = {}
        self._lock = threading.Lock()
    
    def is_allowed(self, key: str) -> bool:
        """检查是否允许请求"""
        raise NotImplementedError
    
    def get_remaining(self, key: str) -> int:
        """获取剩余请求数"""
        raise NotImplementedError
    
    def get_reset_time(self, key: str) -> int:
        """获取重置时间（秒）"""
        raise NotImplementedError
    
    def _get_key(self, request: Request) -> str:
        """从请求生成限流key"""
        # 优先使用API Key，然后是IP地址
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"apikey:{api_key}"
        
        # 获取客户端IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
        
        return f"ip:{client_ip}"


class FixedWindowRateLimiter(RateLimiter):
    """固定窗口限流器"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        super().__init__(max_requests, window_seconds)
        self._timestamps: Dict[str, int] = {}  # key -> window start
    
    def is_allowed(self, key: str) -> bool:
        """检查是否允许请求"""
        with self._lock:
            now = int(time.time())
            current_window = now // self.window_seconds
            
            # 检查是否需要重置窗口
            if key not in self._storage or self._timestamps.get(key) != current_window:
                self._storage[key] = 0
                self._timestamps[key] = current_window
            
            # 检查限制
            if self._storage[key] < self.max_requests:
                self._storage[key] += 1
                return True
            
            return False
    
    def get_remaining(self, key: str) -> int:
        """获取剩余请求数"""
        with self._lock:
            now = int(time.time())
            current_window = now // self.window_seconds
            
            if key not in self._storage or self._timestamps.get(key) != current_window:
                return self.max_requests
            
            return max(0, self.max_requests - self._storage[key])
    
    def get_reset_time(self, key: str) -> int:
        """获取重置时间"""
        now = int(time.time())
        current_window = now // self.window_seconds
        next_window = (current_window + 1) * self.window_seconds
        return next_window - now


class SlidingWindowRateLimiter(RateLimiter):
    """滑动窗口限流器"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        super().__init__(max_requests, window_seconds)
        self._requests: Dict[str, deque] = {}  # key -> timestamps deque
    
    def is_allowed(self, key: str) -> bool:
        """检查是否允许请求"""
        with self._lock:
            now = time.time()
            window_start = now - self.window_seconds
            
            # 获取或创建请求队列
            if key not in self._requests:
                self._requests[key] = deque()
            
            request_queue = self._requests[key]
            
            # 移除窗口外的旧请求
            while request_queue and request_queue[0] < window_start:
                request_queue.popleft()
            
            # 检查限制
            if len(request_queue) < self.max_requests:
                request_queue.append(now)
                return True
            
            return False
    
    def get_remaining(self, key: str) -> int:
        """获取剩余请求数"""
        with self._lock:
            now = time.time()
            window_start = now - self.window_seconds
            
            if key not in self._requests:
                return self.max_requests
            
            request_queue = self._requests[key]
            
            # 移除旧请求
            while request_queue and request_queue[0] < window_start:
                request_queue.popleft()
            
            return max(0, self.max_requests - len(request_queue))
    
    def get_reset_time(self, key: str) -> int:
        """获取重置时间"""
        with self._lock:
            if key not in self._requests or not self._requests[key]:
                return 0
            
            oldest_request = self._requests[key][0]
            reset_time = oldest_request + self.window_seconds - time.time()
            return max(0, int(reset_time))


class TokenBucketRateLimiter(RateLimiter):
    """令牌桶限流器"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        super().__init__(max_requests, window_seconds)
        self._tokens: Dict[str, float] = {}  # key -> current tokens
        self._last_update: Dict[str, float] = {}  # key -> last update time
        self._rate = max_requests / window_seconds  # tokens per second
    
    def is_allowed(self, key: str) -> bool:
        """检查是否允许请求"""
        with self._lock:
            now = time.time()
            
            # 初始化或更新令牌数
            if key not in self._tokens:
                self._tokens[key] = self.max_requests
                self._last_update[key] = now
            else:
                # 补充令牌
                elapsed = now - self._last_update[key]
                tokens_to_add = elapsed * self._rate
                self._tokens[key] = min(self.max_requests, self._tokens[key] + tokens_to_add)
                self._last_update[key] = now
            
            # 检查并消耗令牌
            if self._tokens[key] >= 1:
                self._tokens[key] -= 1
                return True
            
            return False
    
    def get_remaining(self, key: str) -> int:
        """获取剩余请求数"""
        with self._lock:
            now = time.time()
            
            if key not in self._tokens:
                return self.max_requests
            
            # 更新令牌数
            elapsed = now - self._last_update.get(key, now)
            tokens_to_add = elapsed * self._rate
            current_tokens = min(self.max_requests, self._tokens[key] + tokens_to_add)
            
            return int(current_tokens)
    
    def get_reset_time(self, key: str) -> int:
        """获取重置时间"""
        with self._lock:
            if key not in self._tokens or self._tokens[key] >= 1:
                return 0
            
            # 计算获取下一个令牌需要的时间
            tokens_needed = 1 - self._tokens[key]
            seconds_needed = tokens_needed / self._rate
            return int(seconds_needed) + 1


# 全局限流器实例
_rate_limiters: Dict[str, RateLimiter] = {}


def get_rate_limiter(
    name: str = "default",
    strategy: str = "sliding_window",
    max_requests: int = 100,
    window_seconds: int = 60
) -> RateLimiter:
    """获取限流器实例"""
    if name not in _rate_limiters:
        if strategy == "fixed_window":
            _rate_limiters[name] = FixedWindowRateLimiter(max_requests, window_seconds)
        elif strategy == "sliding_window":
            _rate_limiters[name] = SlidingWindowRateLimiter(max_requests, window_seconds)
        elif strategy == "token_bucket":
            _rate_limiters[name] = TokenBucketRateLimiter(max_requests, window_seconds)
        else:
            raise ValueError(f"Unknown rate limiting strategy: {strategy}")
    
    return _rate_limiters[name]


def rate_limit(
    max_requests: int = 100,
    window_seconds: int = 60,
    strategy: str = "sliding_window",
    key_func: Optional[Callable] = None
):
    """限流装饰器"""
    limiter = get_rate_limiter(
        name=f"decorator_{strategy}",
        strategy=strategy,
        max_requests=max_requests,
        window_seconds=window_seconds
    )
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 获取请求对象
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                # 尝试从kwargs获取
                request = kwargs.get('request')
            
            if not request:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Rate limiting requires Request object"
                )
            
            # 生成限流key
            if key_func:
                key = key_func(request)
            else:
                key = limiter._get_key(request)
            
            # 检查限流
            if not limiter.is_allowed(key):
                reset_time = limiter.get_reset_time(key)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={
                        "X-RateLimit-Limit": str(max_requests),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(reset_time),
                        "Retry-After": str(reset_time),
                    }
                )
            
            # 添加限流头到响应
            response = await func(*args, **kwargs)
            
            # 注意：实际需要在响应中添加头部，这里简化处理
            return response
        
        return wrapper
    return decorator


class RateLimitMiddleware:
    """限流中间件"""
    
    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        strategy: str = "sliding_window",
        exclude_paths: Optional[list] = None
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.strategy = strategy
        self.exclude_paths = exclude_paths or ["/health", "/docs", "/openapi.json", "/api/v1/auth/login"]
        self.limiter = get_rate_limiter(
            name="middleware",
            strategy=strategy,
            max_requests=max_requests,
            window_seconds=window_seconds
        )
    
    async def __call__(self, request: Request, call_next):
        """处理请求"""
        from starlette.responses import JSONResponse
        
        path = request.url.path
        
        # 检查排除路径
        if any(path.startswith(exclude) for exclude in self.exclude_paths):
            response = await call_next(request)
            return response
        
        # 生成限流key
        key = self.limiter._get_key(request)
        
        # 检查限流
        if not self.limiter.is_allowed(key):
            reset_time = self.limiter.get_reset_time(key)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after": reset_time,
                },
                headers={
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_time),
                    "Retry-After": str(reset_time),
                }
            )
        
        # 继续处理
        response = await call_next(request)
        
        # 添加限流头部
        remaining = self.limiter.get_remaining(key)
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Window"] = str(self.window_seconds)
        
        return response
