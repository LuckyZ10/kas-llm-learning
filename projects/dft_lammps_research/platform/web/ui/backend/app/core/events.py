"""
Application Lifecycle Events
"""
import structlog
from app.core.config import settings
from app.db.database import init_db, close_db
from app.services.file_watcher import start_file_watcher, stop_file_watcher
from app.services.redis_client import init_redis, close_redis
from app.websocket.manager import init_websocket_manager

logger = structlog.get_logger()


async def startup_event():
    """Initialize services on application startup"""
    logger.info("Starting up DFT+LAMMPS Research Platform API...")
    
    # Ensure directories exist
    settings.ensure_directories()
    logger.info("Directories initialized")
    
    # Initialize database
    await init_db()
    logger.info("Database initialized")
    
    # Initialize Redis
    await init_redis()
    logger.info("Redis initialized")
    
    # Initialize WebSocket manager
    await init_websocket_manager()
    logger.info("WebSocket manager initialized")
    
    # Start file watcher if enabled
    if settings.FILE_WATCHER_ENABLED:
        await start_file_watcher()
        logger.info("File watcher started")
    
    logger.info("Startup complete")


async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("Shutting down...")
    
    # Stop file watcher
    if settings.FILE_WATCHER_ENABLED:
        await stop_file_watcher()
        logger.info("File watcher stopped")
    
    # Close Redis connections
    await close_redis()
    logger.info("Redis connections closed")
    
    # Close database connections
    await close_db()
    logger.info("Database connections closed")
    
    logger.info("Shutdown complete")
