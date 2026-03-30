"""
Main FastAPI Application
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
import structlog

from app.core.config import settings
from app.core.events import startup_event, shutdown_event
from app.api.router import api_router
from app.websocket.manager import websocket_router
from app.core.logging import setup_logging

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    await startup_event()
    yield
    await shutdown_event()


def create_application() -> FastAPI:
    """Create and configure FastAPI application"""
    
    setup_logging()
    
    app = FastAPI(
        title="DFT+LAMMPS Research Platform API",
        description="Modern web interface for DFT+LAMMPS materials research workflow",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # API routes
    app.include_router(api_router, prefix="/api/v1")
    
    # WebSocket routes
    app.include_router(websocket_router, prefix="/ws")
    
    # Static files for uploads/exports
    app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")
    
    @app.get("/")
    async def root():
        return {
            "name": "DFT+LAMMPS Research Platform API",
            "version": "2.0.0",
            "docs": "/docs",
            "websocket": "/ws",
        }
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "version": "2.0.0"}
    
    return app


app = create_application()
