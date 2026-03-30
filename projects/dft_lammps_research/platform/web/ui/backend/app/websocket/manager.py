"""
WebSocket Manager - Real-time communication hub
"""
import asyncio
import json
from typing import Dict, Set, List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import structlog

from app.services.redis_client import redis_client

logger = structlog.get_logger()
router = APIRouter()

# Global connection manager
class ConnectionManager:
    def __init__(self):
        # Map of room_id -> set of websocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Map of websocket -> user_id
        self.user_connections: Dict[WebSocket, str] = {}
        
    async def connect(self, websocket: WebSocket, room: str, user_id: str = "anonymous"):
        await websocket.accept()
        
        if room not in self.active_connections:
            self.active_connections[room] = set()
        
        self.active_connections[room].add(websocket)
        self.user_connections[websocket] = user_id
        
        logger.info(f"WebSocket connected", room=room, user_id=user_id)
        
        # Send connection confirmation
        await websocket.send_json({
            "type": "connection_established",
            "room": room,
            "user_id": user_id,
        })
    
    def disconnect(self, websocket: WebSocket, room: str):
        if room in self.active_connections:
            self.active_connections[room].discard(websocket)
            if not self.active_connections[room]:
                del self.active_connections[room]
        
        self.user_connections.pop(websocket, None)
        logger.info(f"WebSocket disconnected", room=room)
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send message", error=str(e))
    
    async def broadcast_to_room(self, room: str, message: dict):
        if room not in self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections[room]:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.active_connections[room].discard(conn)
    
    async def broadcast_to_all(self, message: dict):
        for room in self.active_connections:
            await self.broadcast_to_room(room, message)


# Global manager instance
manager = ConnectionManager()


@router.websocket("/{room}")
async def websocket_endpoint(websocket: WebSocket, room: str):
    """
    WebSocket endpoint for real-time updates
    Rooms: global, project:{id}, workflow:{id}, task:{id}, user:{id}
    """
    # Extract user from query params or token
    user_id = websocket.query_params.get("user_id", "anonymous")
    
    await manager.connect(websocket, room, user_id)
    
    try:
        while True:
            # Receive and handle messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            msg_type = message.get("type")
            
            if msg_type == "ping":
                await websocket.send_json({"type": "pong", "timestamp": message.get("timestamp")})
            
            elif msg_type == "subscribe":
                # Client wants to subscribe to additional rooms
                new_room = message.get("room")
                if new_room:
                    await manager.connect(websocket, new_room, user_id)
            
            elif msg_type == "unsubscribe":
                old_room = message.get("room")
                if old_room:
                    manager.disconnect(websocket, old_room)
            
            else:
                # Handle other message types
                await handle_client_message(message, websocket, room)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, room)
    except Exception as e:
        logger.error(f"WebSocket error", error=str(e), room=room)
        manager.disconnect(websocket, room)


async def handle_client_message(message: dict, websocket: WebSocket, room: str):
    """Handle incoming client messages"""
    msg_type = message.get("type")
    
    # Echo back for now, extend as needed
    await websocket.send_json({
        "type": "ack",
        "original_type": msg_type,
        "room": room,
    })


async def init_websocket_manager():
    """Initialize WebSocket manager"""
    logger.info("WebSocket manager initialized")


# Helper functions for broadcasting events
async def broadcast_task_update(task_id: str, status: str, data: dict = None):
    """Broadcast task status update"""
    message = {
        "type": "task_update",
        "task_id": task_id,
        "status": status,
        "data": data or {},
        "timestamp": asyncio.get_event_loop().time(),
    }
    await manager.broadcast_to_room(f"task:{task_id}", message)
    await manager.broadcast_to_room("global", message)


async def broadcast_workflow_update(workflow_id: str, status: str, progress: float = None):
    """Broadcast workflow status update"""
    message = {
        "type": "workflow_update",
        "workflow_id": workflow_id,
        "status": status,
        "progress": progress,
        "timestamp": asyncio.get_event_loop().time(),
    }
    await manager.broadcast_to_room(f"workflow:{workflow_id}", message)


async def broadcast_project_update(project_id: str, data: dict):
    """Broadcast project update"""
    message = {
        "type": "project_update",
        "project_id": project_id,
        "data": data,
        "timestamp": asyncio.get_event_loop().time(),
    }
    await manager.broadcast_to_room(f"project:{project_id}", message)


async def broadcast_system_stats(stats: dict):
    """Broadcast system statistics"""
    message = {
        "type": "system_stats",
        "stats": stats,
        "timestamp": asyncio.get_event_loop().time(),
    }
    await manager.broadcast_to_room("global", message)


async def broadcast_log_message(source: str, level: str, message: str, metadata: dict = None):
    """Broadcast log message"""
    msg = {
        "type": "log_message",
        "source": source,
        "level": level,
        "message": message,
        "metadata": metadata or {},
        "timestamp": asyncio.get_event_loop().time(),
    }
    await manager.broadcast_to_room("global", msg)
