"""
Application Configuration
"""
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application
    APP_NAME: str = "DFT+LAMMPS Research Platform"
    DEBUG: bool = False
    SECRET_KEY: str = "your-secret-key-change-in-production"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/dft_lammps"
    DATABASE_POOL_SIZE: int = 20
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_PASSWORD: Optional[str] = None
    
    # File Storage
    UPLOAD_DIR: Path = Path("./uploads")
    STATIC_DIR: Path = Path("./static")
    EXPORT_DIR: Path = Path("./exports")
    
    # Workflow Paths
    WORK_DIR: Path = Path("./workdir")
    DFT_RESULTS_PATH: Path = Path("./battery_screening/dft_results")
    MD_RESULTS_PATH: Path = Path("./battery_screening/md_results")
    MODELS_PATH: Path = Path("./battery_screening/models")
    AL_WORKFLOW_PATH: Path = Path("./active_learning_workflow")
    SCREENING_DB_PATH: Path = Path("./screening_db")
    
    # Security
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 30  # 30 days
    
    # Real-time
    WS_HEARTBEAT_INTERVAL: int = 30
    FILE_WATCHER_ENABLED: bool = True
    
    # External Services
    SENTRY_DSN: Optional[str] = None
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "dft-lammps"
    
    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        
    def ensure_directories(self):
        """Ensure all required directories exist"""
        for path in [self.UPLOAD_DIR, self.STATIC_DIR, self.EXPORT_DIR, 
                     self.WORK_DIR, self.DFT_RESULTS_PATH, self.MD_RESULTS_PATH,
                     self.MODELS_PATH, self.AL_WORKFLOW_PATH, self.SCREENING_DB_PATH]:
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
