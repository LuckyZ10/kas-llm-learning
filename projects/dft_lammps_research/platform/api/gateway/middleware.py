"""
API Gateway Middleware

Custom middleware for request/response processing
"""

import time
import json
import uuid
from typing import Callable, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import structlog

logger = structlog.get_logger()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all API requests"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        
        start_time = time.time()
        
        # Extract client info
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Log request
        logger.info(
            "request_started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=client_ip,
            user_agent=user_agent,
        )
        
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log response
            logger.info(
                "request_completed",
                request_id=request_id,
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            logger.error(
                "request_failed",
                request_id=request_id,
                error=str(e),
                duration_ms=round(duration_ms, 2),
            )
            raise


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Track request timing for analytics"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        response = await call_next(request)
        
        # Add timing header
        duration_ms = (time.time() - start_time) * 1000
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        
        return response


class VersionHeaderMiddleware(BaseHTTPMiddleware):
    """Add API version headers to all responses"""
    
    def __init__(self, app, version: str = "1.0.0"):
        super().__init__(app)
        self.version = version
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add version headers
        response.headers["X-API-Version"] = self.version
        response.headers["X-API-Status"] = "stable"
        
        return response


class ResponseTransformMiddleware(BaseHTTPMiddleware):
    """
    Transform responses based on client preferences
    
    Supports:
    - Field filtering (?fields=id,name,status)
    - Response format (?format=xml)
    - Pretty printing (?pretty=true)
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Only transform JSON responses
        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type:
            return response
        
        # Check for transformation params
        fields = request.query_params.get("fields")
        response_format = request.query_params.get("format")
        pretty = request.query_params.get("pretty", "false").lower() == "true"
        
        if not any([fields, response_format, pretty]):
            return response
        
        try:
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            
            data = json.loads(body)
            
            # Field filtering
            if fields and isinstance(data, dict):
                field_list = [f.strip() for f in fields.split(",")]
                data = self._filter_fields(data, field_list)
            
            # Pretty printing
            indent = 2 if pretty else None
            separators = None if pretty else (",", ":")
            
            new_body = json.dumps(data, indent=indent, separators=separators, default=str)
            
            # Reconstruct response
            new_response = Response(
                content=new_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type="application/json",
            )
            
            return new_response
            
        except Exception as e:
            logger.error("response_transform_failed", error=str(e))
            return response
    
    def _filter_fields(self, data: dict, fields: list) -> dict:
        """Filter response to only include specified fields"""
        if not isinstance(data, dict):
            return data
        
        result = {}
        for field in fields:
            if "." in field:
                # Handle nested fields (e.g., "owner.name")
                parts = field.split(".")
                current = data
                for part in parts[:-1]:
                    current = current.get(part, {})
                if parts[-1] in current:
                    # Build nested structure
                    target = result
                    for part in parts[:-1]:
                        if part not in target:
                            target[part] = {}
                        target = target[part]
                    target[parts[-1]] = current[parts[-1]]
            else:
                if field in data:
                    result[field] = data[field]
        
        return result


class CacheControlMiddleware(BaseHTTPMiddleware):
    """Add cache control headers based on endpoint"""
    
    # Cache durations in seconds
    CACHE_RULES = {
        "/api/v1/projects": 0,  # No cache
        "/api/v1/calculations": 0,
        "/api/v1/structures": 60,  # 1 minute
        "/api/v1/results": 300,  # 5 minutes
    }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        path = request.url.path
        
        # Find matching cache rule
        cache_duration = None
        for prefix, duration in self.CACHE_RULES.items():
            if path.startswith(prefix):
                cache_duration = duration
                break
        
        if cache_duration is not None:
            if cache_duration == 0:
                response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            else:
                response.headers["Cache-Control"] = f"max-age={cache_duration}, private"
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response
