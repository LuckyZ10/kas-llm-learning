"""
API Gateway Configuration
"""

import os
from typing import List, Dict
from pydantic_settings import BaseSettings


class GatewayConfig(BaseSettings):
    """Gateway configuration settings"""
    
    # Version
    VERSION: str = "1.0.0"
    API_VERSION: str = "v1"
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8080
    DEBUG: bool = False
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    
    # Redis settings
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    
    # Rate limiting
    RATE_LIMITS: Dict[str, Dict[str, int]] = {
        "free": {
            "requests_per_minute": 60,
            "requests_per_day": 10000,
            "max_projects": 5,
            "max_calculations_per_day": 100,
        },
        "pro": {
            "requests_per_minute": 300,
            "requests_per_day": 100000,
            "max_projects": 50,
            "max_calculations_per_day": 10000,
        },
        "enterprise": {
            "requests_per_minute": 1000,
            "requests_per_day": 1000000,
            "max_projects": -1,  # unlimited
            "max_calculations_per_day": -1,  # unlimited
        },
    }
    
    # OAuth2 settings
    OAUTH2_TOKEN_URL: str = "/api/v1/auth/token"
    OAUTH2_AUTHORIZE_URL: str = "/api/v1/auth/authorize"
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # API Key settings
    API_KEY_HEADER: str = "X-API-Key"
    API_KEY_MIN_LENGTH: int = 32
    
    # Webhook settings
    WEBHOOK_SECRET_HEADER: str = "X-Webhook-Signature"
    WEBHOOK_RETRY_ATTEMPTS: int = 5
    WEBHOOK_RETRY_DELAY_SECONDS: int = 60
    WEBHOOK_TIMEOUT_SECONDS: int = 30
    
    # Cache settings
    CACHE_TTL_SECONDS: int = 300  # 5 minutes
    CACHE_MAX_SIZE: int = 10000
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # Database (for API management)
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql+asyncpg://api_user:api_pass@localhost/api_platform"
    )
    
    # External service URLs
    DFT_SERVICE_URL: str = os.getenv("DFT_SERVICE_URL", "http://localhost:8000")
    LAMMPS_SERVICE_URL: str = os.getenv("LAMMPS_SERVICE_URL", "http://localhost:8001")
    WEBUI_SERVICE_URL: str = os.getenv("WEBUI_SERVICE_URL", "http://localhost:3000")
    
    # Feature flags
    ENABLE_WEBHOOKS: bool = True
    ENABLE_BATCH_OPERATIONS: bool = True
    ENABLE_REALTIME_STREAMING: bool = True
    ENABLE_USAGE_ANALYTICS: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True
