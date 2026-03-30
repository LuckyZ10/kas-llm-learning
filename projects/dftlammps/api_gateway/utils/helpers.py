"""
工具函数
"""

import uuid
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def generate_uuid() -> str:
    """生成UUID"""
    return str(uuid.uuid4())


def generate_short_id() -> str:
    """生成短ID（8字符）"""
    return secrets.token_urlsafe(6)[:8]


def generate_task_id() -> str:
    """生成任务ID"""
    return f"task_{generate_short_id()}_{int(datetime.utcnow().timestamp())}"


def hash_string(s: str, algorithm: str = "sha256") -> str:
    """哈希字符串"""
    if algorithm == "sha256":
        return hashlib.sha256(s.encode()).hexdigest()
    elif algorithm == "md5":
        return hashlib.md5(s.encode()).hexdigest()
    elif algorithm == "sha1":
        return hashlib.sha1(s.encode()).hexdigest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def format_datetime(dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """格式化日期时间"""
    return dt.strftime(fmt)


def parse_datetime(s: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """解析日期时间"""
    return datetime.strptime(s, fmt)


def humanize_bytes(size_bytes: int) -> str:
    """人性化显示字节大小"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024
        i += 1
    
    return f"{size_bytes:.2f} {size_names[i]}"


def humanize_duration(seconds: float) -> str:
    """人性化显示持续时间"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"


def sanitize_filename(filename: str) -> str:
    """清理文件名"""
    import re
    # 移除危险字符
    filename = re.sub(r'[^<>":/\|?*]', "", filename)
    # 限制长度
    return filename[:255]


def validate_email(email: str) -> bool:
    """验证邮箱格式"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def paginate(
    items: list,
    page: int = 1,
    page_size: int = 20
) -> Dict[str, Any]:
    """分页"""
    total = len(items)
    total_pages = (total + page_size - 1) // page_size
    
    start = (page - 1) * page_size
    end = start + page_size
    
    return {
        "items": items[start:end],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1,
    }


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """重试装饰器"""
    import time
    from functools import wraps
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            current_delay = delay
            
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        raise
                    
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {current_delay}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
                    attempt += 1
            
            return None  # Should never reach here
        
        return wrapper
    
    return decorator


def cached(timeout: int = 300):
    """简单缓存装饰器"""
    from functools import wraps
    
    cache = {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            now = datetime.utcnow()
            
            if key in cache:
                result, timestamp = cache[key]
                if (now - timestamp).seconds < timeout:
                    return result
            
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            return result
        
        return wrapper
    
    return decorator


def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """截断字符串"""
    if len(s) <= max_length:
        return s
    return s[:max_length - len(suffix)] + suffix


def mask_sensitive(data: Dict[str, Any], sensitive_keys: list = None) -> Dict[str, Any]:
    """遮盖敏感信息"""
    if sensitive_keys is None:
        sensitive_keys = ["password", "token", "secret", "key", "api_key"]
    
    masked = {}
    for k, v in data.items():
        if any(sk in k.lower() for sk in sensitive_keys):
            masked[k] = "***"
        elif isinstance(v, dict):
            masked[k] = mask_sensitive(v, sensitive_keys)
        else:
            masked[k] = v
    
    return masked
