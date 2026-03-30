"""
File Watcher Service - Monitor file changes and emit events
"""
import asyncio
from pathlib import Path
from typing import Set
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent
import structlog

from app.core.config import settings
from app.websocket.manager import broadcast_log_message

logger = structlog.get_logger()

# Global observer
observer: Observer = None
watched_paths: Set[Path] = set()


class WorkflowFileHandler(FileSystemEventHandler):
    """Handler for file system events"""
    
    def __init__(self):
        self.last_events = {}
        self.debounce_seconds = 1.0
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        self._handle_event(event, "modified")
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        self._handle_event(event, "created")
    
    def _handle_event(self, event, event_type):
        import time
        
        current_time = time.time()
        file_path = Path(event.src_path)
        
        # Debounce events
        last_time = self.last_events.get(file_path, 0)
        if current_time - last_time < self.debounce_seconds:
            return
        
        self.last_events[file_path] = current_time
        
        # Log and broadcast
        logger.debug(f"File {event_type}", path=str(file_path))
        
        # Determine file type and broadcast appropriate message
        asyncio.create_task(self._broadcast_file_event(file_path, event_type))
    
    async def _broadcast_file_event(self, file_path: Path, event_type: str):
        """Broadcast file event to WebSocket clients"""
        
        # Training log updates
        if file_path.name == "lcurve.out":
            await broadcast_log_message(
                source="training",
                level="info",
                message=f"Training log updated",
                metadata={"file": str(file_path), "event": event_type}
            )
        
        # MD log updates
        elif file_path.name == "log.lammps":
            await broadcast_log_message(
                source="md_simulation",
                level="info",
                message=f"MD simulation log updated",
                metadata={"file": str(file_path), "event": event_type}
            )
        
        # Dump file updates
        elif file_path.suffix == ".lammpstrj":
            await broadcast_log_message(
                source="md_simulation",
                level="info",
                message=f"Trajectory file updated",
                metadata={"file": str(file_path), "event": event_type}
            )
        
        # Model file updates
        elif file_path.suffix in [".pb", ".pt"]:
            await broadcast_log_message(
                source="ml_training",
                level="info",
                message=f"Model file updated",
                metadata={"file": str(file_path), "event": event_type}
            )


async def start_file_watcher():
    """Start file watching service"""
    global observer
    
    if observer:
        return
    
    try:
        observer = Observer()
        handler = WorkflowFileHandler()
        
        # Watch key directories
        watch_paths = [
            settings.MODELS_PATH,
            settings.MD_RESULTS_PATH,
            settings.DFT_RESULTS_PATH,
            settings.AL_WORKFLOW_PATH,
        ]
        
        for path in watch_paths:
            if path.exists():
                observer.schedule(handler, str(path), recursive=True)
                watched_paths.add(path)
                logger.info(f"Watching directory", path=str(path))
        
        observer.start()
        logger.info("File watcher started")
        
    except Exception as e:
        logger.error(f"Failed to start file watcher", error=str(e))
        observer = None


async def stop_file_watcher():
    """Stop file watching service"""
    global observer
    
    if observer:
        observer.stop()
        observer.join()
        observer = None
        logger.info("File watcher stopped")
