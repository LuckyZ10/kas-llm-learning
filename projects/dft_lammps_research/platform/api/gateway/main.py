"""
API Gateway - Main entry point for external API access

Production-grade API Gateway with:
- OAuth2 and API Key authentication
- Rate limiting and quota management
- Request/response transformation
- API versioning
- Caching and logging
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis.asyncio as redis
import structlog
import time
import uuid
from typing import Optional, Callable
import json

from api_platform.gateway.config import GatewayConfig
from api_platform.auth.oauth2 import OAuth2Manager
from api_platform.auth.api_key import APIKeyManager
from api_platform.auth.permissions import PermissionChecker
from api_platform.gateway.middleware import (
    RequestLoggingMiddleware,
    RequestTimingMiddleware,
    VersionHeaderMiddleware,
    ResponseTransformMiddleware,
)
from api_platform.gateway.router import gateway_router
from api_platform.webhooks.router import webhook_router
from api_platform.portal.router import portal_router

logger = structlog.get_logger()
security = HTTPBearer(auto_error=False)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)


class APIGateway:
    """Main API Gateway class"""
    
    def __init__(self):
        self.config = GatewayConfig()
        self.oauth2_manager = OAuth2Manager()
        self.api_key_manager = APIKeyManager()
        self.permission_checker = PermissionChecker()
        self.redis_client: Optional[redis.Redis] = None
        
    async def setup_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.REDIS_HOST,
                port=self.config.REDIS_PORT,
                db=self.config.REDIS_DB,
                decode_responses=True,
            )
            await self.redis_client.ping()
            logger.info("redis_connected", host=self.config.REDIS_HOST)
        except Exception as e:
            logger.error("redis_connection_failed", error=str(e))
            self.redis_client = None
    
    async def close_redis(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
    
    async def check_rate_limit(
        self, 
        request: Request, 
        client_id: str,
        tier: str = "free"
    ) -> bool:
        """Check if client has exceeded rate limit"""
        if not self.redis_client:
            return True
        
        # Get tier limits
        limits = self.config.RATE_LIMITS.get(tier, self.config.RATE_LIMITS["free"])
        requests_per_minute = limits["requests_per_minute"]
        requests_per_day = limits["requests_per_day"]
        
        # Check minute limit
        minute_key = f"ratelimit:{client_id}:minute"
        current_minute = int(time.time()) // 60
        minute_count = await self.redis_client.get(f"{minute_key}:{current_minute}")
        
        if minute_count and int(minute_count) >= requests_per_minute:
            return False
        
        # Check daily limit
        today = time.strftime("%Y-%m-%d")
        day_key = f"ratelimit:{client_id}:day:{today}"
        day_count = await self.redis_client.get(day_key)
        
        if day_count and int(day_count) >= requests_per_day:
            return False
        
        # Increment counters
        pipe = self.redis_client.pipeline()
        pipe.incr(f"{minute_key}:{current_minute}")
        pipe.expire(f"{minute_key}:{current_minute}", 120)  # 2 min TTL
        pipe.incr(day_key)
        pipe.expire(day_key, 86400 * 2)  # 2 days TTL
        await pipe.execute()
        
        return True
    
    async def authenticate_request(
        self,
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
    ) -> dict:
        """Authenticate request using OAuth2 or API Key"""
        # Try API Key first (from header or query param)
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        
        if api_key:
            client = await self.api_key_manager.validate_key(api_key)
            if client:
                return {
                    "client_id": client["client_id"],
                    "client_name": client["name"],
                    "tier": client.get("tier", "free"),
                    "auth_method": "api_key",
                    "permissions": client.get("permissions", ["read"]),
                }
        
        # Try OAuth2 Bearer token
        if credentials:
            token_data = await self.oauth2_manager.validate_token(credentials.credentials)
            if token_data:
                return {
                    "client_id": token_data["client_id"],
                    "client_name": token_data.get("name", "Unknown"),
                    "tier": token_data.get("tier", "free"),
                    "auth_method": "oauth2",
                    "permissions": token_data.get("permissions", ["read"]),
                }
        
        # No valid authentication
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    def create_application(self) -> FastAPI:
        """Create and configure FastAPI application"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await self.setup_redis()
            logger.info("api_gateway_started", version=self.config.VERSION)
            yield
            await self.close_redis()
            logger.info("api_gateway_stopped")
        
        app = FastAPI(
            title="DFT+LAMMPS API Platform",
            description="""
            Production-grade API for DFT+LAMMPS materials research platform.
            
            ## Authentication
            
            This API supports two authentication methods:
            
            1. **API Key**: Include your API key in the `X-API-Key` header
            2. **OAuth2**: Use Bearer token authentication
            
            ## Rate Limits
            
            - Free tier: 60 requests/minute, 10,000 requests/day
            - Pro tier: 300 requests/minute, 100,000 requests/day
            - Enterprise tier: Custom limits
            
            ## API Versions
            
            - Current: v1 (stable)
            - Beta: v2 (available for testing)
            """,
            version=self.config.VERSION,
            docs_url=None,  # Custom docs endpoint
            redoc_url=None,
            openapi_url="/openapi.json",
            lifespan=lifespan,
        )
        
        # Add rate limiter
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        
        # CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Compression middleware
        app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Custom middleware
        app.add_middleware(RequestLoggingMiddleware)
        app.add_middleware(RequestTimingMiddleware)
        app.add_middleware(VersionHeaderMiddleware, version=self.config.VERSION)
        
        # Include routers
        app.include_router(gateway_router, prefix="/api/v1")
        app.include_router(webhook_router, prefix="/api/v1/webhooks")
        app.include_router(portal_router, prefix="/portal")
        
        # Custom documentation endpoints
        @app.get("/docs", include_in_schema=False)
        async def swagger_ui():
            return get_swagger_ui_html(
                openapi_url="/openapi.json",
                title="DFT+LAMMPS API Documentation",
                oauth2_redirect_url="/docs/oauth2-redirect",
            )
        
        @app.get("/redoc", include_in_schema=False)
        async def redoc():
            return get_redoc_html(
                openapi_url="/openapi.json",
                title="DFT+LAMMPS API Documentation",
            )
        
        # Health check
        @app.get("/health", tags=["System"])
        async def health_check():
            """Health check endpoint"""
            health_status = {
                "status": "healthy",
                "version": self.config.VERSION,
                "timestamp": time.time(),
            }
            
            if self.redis_client:
                try:
                    await self.redis_client.ping()
                    health_status["redis"] = "connected"
                except:
                    health_status["redis"] = "disconnected"
            
            return health_status
        
        # Root endpoint
        @app.get("/", tags=["System"])
        async def root():
            """API information"""
            return {
                "name": "DFT+LAMMPS API Platform",
                "version": self.config.VERSION,
                "documentation": "/docs",
                "developer_portal": "/portal",
                "support": "https://support.dft-lammps.org",
            }
        
        # Custom OpenAPI schema
        def custom_openapi():
            if app.openapi_schema:
                return app.openapi_schema
            
            openapi_schema = get_openapi(
                title="DFT+LAMMPS API Platform",
                version=self.config.VERSION,
                description="Production-grade API for materials research",
                routes=app.routes,
            )
            
            # Add security schemes
            openapi_schema["components"]["securitySchemes"] = {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT",
                    "description": "OAuth2 Bearer token",
                },
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key",
                    "description": "API Key authentication",
                },
            }
            
            app.openapi_schema = openapi_schema
            return app.openapi_schema
        
        app.openapi = custom_openapi
        
        return app


# Create gateway instance
gateway = APIGateway()
app = gateway.create_application()
