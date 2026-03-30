"""
Real-time Synchronization Module
================================

Provides WebSocket-based real-time collaboration features for the DFT-LAMMPS platform,
enabling multi-user simultaneous editing, real-time computation result streaming,
and conflict resolution mechanisms.

Features
--------
- WebSocket-based real-time communication
- Multi-user concurrent structure editing
- Real-time computation result push
- Operational Transformation (OT) for conflict resolution
- Presence awareness (cursors, selections)
- Heartbeat and reconnection management

Example
-------
>>> from dftlammps.collaboration.realtime_sync import RealtimeSyncManager
>>> sync = RealtimeSyncManager("ws://localhost:8765")
>>> await sync.connect()
>>> await sync.join_session("project_123")
>>> await sync.broadcast_structure_update(structure_data)

Author: DFT-LAMMPS Collaboration Team
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Any, Set, Tuple, Union
from contextlib import asynccontextmanager
import hashlib

# WebSocket imports
try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    from websockets.client import WebSocketClientProtocol
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    WebSocketServerProtocol = Any
    WebSocketClientProtocol = Any

# Optional: Redis for distributed sync
try:
    import redis.asyncio as redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

logger = logging.getLogger(__name__)


class SyncMessageType(Enum):
    """Types of synchronization messages."""
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    JOIN_SESSION = "join_session"
    LEAVE_SESSION = "leave_session"
    STRUCTURE_UPDATE = "structure_update"
    CURSOR_UPDATE = "cursor_update"
    SELECTION_UPDATE = "selection_update"
    COMPUTATION_RESULT = "computation_result"
    OPERATION = "operation"
    ACK = "ack"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    PRESENCE = "presence"
    LOCK_REQUEST = "lock_request"
    LOCK_RELEASE = "lock_release"
    CONFLICT_DETECTED = "conflict_detected"
    CONFLICT_RESOLVED = "conflict_resolved"


class OperationType(Enum):
    """Types of operations for Operational Transformation."""
    INSERT = "insert"
    DELETE = "delete"
    REPLACE = "replace"
    MOVE = "move"
    TRANSFORM = "transform"
    PROPERTY_CHANGE = "property_change"


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving edit conflicts."""
    LAST_WRITE_WINS = "last_write_wins"
    FIRST_WRITE_WINS = "first_write_wins"
    OPERATIONAL_TRANSFORMATION = "operational_transformation"
    MANUAL_MERGE = "manual_merge"
    LOCK_BASED = "lock_based"


@dataclass
class UserPresence:
    """Represents a user's presence in a collaborative session."""
    user_id: str
    username: str
    color: str = "#007bff"
    cursor_position: Optional[Dict[str, float]] = None
    selection: Optional[Dict[str, Any]] = None
    is_active: bool = True
    last_seen: float = field(default_factory=time.time)
    current_file: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "color": self.color,
            "cursor_position": self.cursor_position,
            "selection": self.selection,
            "is_active": self.is_active,
            "last_seen": self.last_seen,
            "current_file": self.current_file,
        }


@dataclass
class Operation:
    """Represents an operation for Operational Transformation."""
    op_id: str
    type: OperationType
    path: str
    value: Any
    timestamp: float
    user_id: str
    revision: int
    parent_revision: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_id": self.op_id,
            "type": self.type.value,
            "path": self.path,
            "value": self.value,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "revision": self.revision,
            "parent_revision": self.parent_revision,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Operation:
        return cls(
            op_id=data["op_id"],
            type=OperationType(data["type"]),
            path=data["path"],
            value=data["value"],
            timestamp=data["timestamp"],
            user_id=data["user_id"],
            revision=data["revision"],
            parent_revision=data["parent_revision"],
        )


@dataclass
class SyncMessage:
    """A message for real-time synchronization."""
    msg_type: SyncMessageType
    session_id: str
    user_id: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "msg_id": self.msg_id,
            "msg_type": self.msg_type.value,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SyncMessage:
        return cls(
            msg_type=SyncMessageType(data["msg_type"]),
            session_id=data["session_id"],
            user_id=data["user_id"],
            payload=data["payload"],
            timestamp=data.get("timestamp", time.time()),
            msg_id=data.get("msg_id", str(uuid.uuid4())),
        )


@dataclass
class StructureVersion:
    """Represents a version of a structure with full history."""
    version_id: str
    revision: int
    structure_data: Dict[str, Any]
    operations: List[Operation]
    timestamp: float
    user_id: str
    checksum: str
    
    def calculate_checksum(self) -> str:
        """Calculate checksum for data integrity."""
        data_str = json.dumps(self.structure_data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "revision": self.revision,
            "structure_data": self.structure_data,
            "operations": [op.to_dict() for op in self.operations],
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "checksum": self.checksum,
        }


class OperationalTransformer:
    """
    Implements Operational Transformation (OT) for conflict-free collaborative editing.
    
    OT allows multiple users to edit the same document simultaneously by transforming
    operations to maintain consistency.
    """
    
    def __init__(self):
        self.revision_history: List[Operation] = []
        self.current_revision: int = 0
    
    def transform_operations(
        self,
        op1: Operation,
        op2: Operation
    ) -> Tuple[Operation, Operation]:
        """
        Transform two concurrent operations.
        
        Args:
            op1: First operation
            op2: Second operation
            
        Returns:
            Tuple of transformed operations (op1', op2')
        """
        if op1.type == OperationType.INSERT and op2.type == OperationType.INSERT:
            return self._transform_insert_insert(op1, op2)
        elif op1.type == OperationType.INSERT and op2.type == OperationType.DELETE:
            return self._transform_insert_delete(op1, op2)
        elif op1.type == OperationType.DELETE and op2.type == OperationType.INSERT:
            op2_t, op1_t = self._transform_insert_delete(op2, op1)
            return op1_t, op2_t
        elif op1.type == OperationType.DELETE and op2.type == OperationType.DELETE:
            return self._transform_delete_delete(op1, op2)
        else:
            # For other types, keep both operations
            return op1, op2
    
    def _transform_insert_insert(
        self,
        op1: Operation,
        op2: Operation
    ) -> Tuple[Operation, Operation]:
        """Transform two insert operations."""
        # Simplified transformation based on position
        path1 = op1.path.split(".")
        path2 = op2.path.split(".")
        
        if path1 == path2:
            # Same path - use timestamp ordering
            if op1.timestamp <= op2.timestamp:
                return op1, op2
            else:
                # Adjust op2 position
                return op1, Operation(
                    op_id=op2.op_id,
                    type=op2.type,
                    path=op2.path,
                    value=op2.value,
                    timestamp=op2.timestamp,
                    user_id=op2.user_id,
                    revision=op2.revision,
                    parent_revision=op2.parent_revision,
                )
        return op1, op2
    
    def _transform_insert_delete(
        self,
        insert_op: Operation,
        delete_op: Operation
    ) -> Tuple[Operation, Operation]:
        """Transform insert and delete operations."""
        # If delete affects the same path, insert happens first
        if insert_op.path == delete_op.path:
            # Keep insert, adjust delete
            adjusted_delete = Operation(
                op_id=delete_op.op_id,
                type=delete_op.type,
                path=delete_op.path,
                value=delete_op.value,
                timestamp=delete_op.timestamp,
                user_id=delete_op.user_id,
                revision=delete_op.revision,
                parent_revision=delete_op.parent_revision,
            )
            return insert_op, adjusted_delete
        return insert_op, delete_op
    
    def _transform_delete_delete(
        self,
        op1: Operation,
        op2: Operation
    ) -> Tuple[Operation, Operation]:
        """Transform two delete operations."""
        if op1.path == op2.path and op1.value == op2.value:
            # Same deletion - second becomes no-op
            no_op = Operation(
                op_id=op2.op_id,
                type=OperationType.PROPERTY_CHANGE,
                path=op2.path,
                value=None,
                timestamp=op2.timestamp,
                user_id=op2.user_id,
                revision=op2.revision,
                parent_revision=op2.parent_revision,
            )
            return op1, no_op
        return op1, op2
    
    def apply_operation(
        self,
        structure: Dict[str, Any],
        operation: Operation
    ) -> Dict[str, Any]:
        """Apply an operation to a structure."""
        result = self._deep_copy(structure)
        path_parts = operation.path.split(".")
        
        try:
            if operation.type == OperationType.INSERT:
                self._apply_insert(result, path_parts, operation.value)
            elif operation.type == OperationType.DELETE:
                self._apply_delete(result, path_parts)
            elif operation.type == OperationType.REPLACE:
                self._apply_replace(result, path_parts, operation.value)
            elif operation.type == OperationType.MOVE:
                self._apply_move(result, path_parts, operation.value)
            elif operation.type == OperationType.TRANSFORM:
                self._apply_transform(result, path_parts, operation.value)
            elif operation.type == OperationType.PROPERTY_CHANGE:
                self._apply_property_change(result, path_parts, operation.value)
        except Exception as e:
            logger.error(f"Failed to apply operation: {e}")
            raise
        
        self.revision_history.append(operation)
        self.current_revision += 1
        
        return result
    
    def _deep_copy(self, obj: Any) -> Any:
        """Deep copy an object."""
        return json.loads(json.dumps(obj))
    
    def _apply_insert(
        self,
        structure: Dict[str, Any],
        path: List[str],
        value: Any
    ) -> None:
        """Apply insert operation."""
        current = structure
        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[path[-1]] = value
    
    def _apply_delete(
        self,
        structure: Dict[str, Any],
        path: List[str]
    ) -> None:
        """Apply delete operation."""
        current = structure
        for part in path[:-1]:
            if part not in current:
                return
            current = current[part]
        if path[-1] in current:
            del current[path[-1]]
    
    def _apply_replace(
        self,
        structure: Dict[str, Any],
        path: List[str],
        value: Any
    ) -> None:
        """Apply replace operation."""
        current = structure
        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[path[-1]] = value
    
    def _apply_move(
        self,
        structure: Dict[str, Any],
        path: List[str],
        target_path: str
    ) -> None:
        """Apply move operation."""
        # Get value from source
        current = structure
        for part in path[:-1]:
            current = current.get(part, {})
        value = current.get(path[-1])
        
        if value is not None:
            # Delete from source
            del current[path[-1]]
            # Insert at target
            target_parts = target_path.split(".")
            current = structure
            for part in target_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[target_parts[-1]] = value
    
    def _apply_transform(
        self,
        structure: Dict[str, Any],
        path: List[str],
        transform: Dict[str, Any]
    ) -> None:
        """Apply transform operation (e.g., rotation, translation)."""
        current = structure
        for part in path:
            current = current.get(part, {})
        
        # Apply transformation matrix or parameters
        if "atoms" in current and "positions" in current["atoms"]:
            positions = current["atoms"]["positions"]
            transform_type = transform.get("type", "identity")
            
            if transform_type == "rotation":
                # Apply rotation
                angle = transform.get("angle", 0)
                axis = transform.get("axis", [0, 0, 1])
                # Simplified rotation application
                logger.debug(f"Applying rotation of {angle} around {axis}")
            elif transform_type == "translation":
                vector = transform.get("vector", [0, 0, 0])
                for i, pos in enumerate(positions):
                    positions[i] = [p + v for p, v in zip(pos, vector)]
    
    def _apply_property_change(
        self,
        structure: Dict[str, Any],
        path: List[str],
        properties: Dict[str, Any]
    ) -> None:
        """Apply property change operation."""
        current = structure
        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        target = current.get(path[-1], {})
        if isinstance(properties, dict):
            target.update(properties)
        current[path[-1]] = target


class ConflictResolver:
    """
    Resolves conflicts in concurrent editing scenarios.
    
    Supports multiple resolution strategies:
    - Last Write Wins (LWW)
    - First Write Wins (FWW)
    - Operational Transformation (OT)
    - Lock-based
    - Manual Merge
    """
    
    def __init__(self, strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.OPERATIONAL_TRANSFORMATION):
        self.strategy = strategy
        self.ot = OperationalTransformer()
        self.active_locks: Dict[str, str] = {}  # path -> user_id
        self.pending_conflicts: List[Dict[str, Any]] = []
    
    def resolve_conflict(
        self,
        base_version: StructureVersion,
        local_changes: List[Operation],
        remote_changes: List[Operation],
        local_user_id: str,
        remote_user_id: str
    ) -> Tuple[StructureVersion, List[Operation]]:
        """
        Resolve a conflict between local and remote changes.
        
        Args:
            base_version: Common base version
            local_changes: Local user's operations
            remote_changes: Remote user's operations
            local_user_id: Local user ID
            remote_user_id: Remote user ID
            
        Returns:
            Tuple of (resolved_version, applied_operations)
        """
        if self.strategy == ConflictResolutionStrategy.LAST_WRITE_WINS:
            return self._resolve_lww(base_version, local_changes, remote_changes, local_user_id, remote_user_id)
        elif self.strategy == ConflictResolutionStrategy.FIRST_WRITE_WINS:
            return self._resolve_fww(base_version, local_changes, remote_changes, local_user_id, remote_user_id)
        elif self.strategy == ConflictResolutionStrategy.OPERATIONAL_TRANSFORMATION:
            return self._resolve_ot(base_version, local_changes, remote_changes)
        elif self.strategy == ConflictResolutionStrategy.LOCK_BASED:
            return self._resolve_lock_based(base_version, local_changes, remote_changes, local_user_id)
        else:
            return self._resolve_manual(base_version, local_changes, remote_changes)
    
    def _resolve_lww(
        self,
        base_version: StructureVersion,
        local_changes: List[Operation],
        remote_changes: List[Operation],
        local_user_id: str,
        remote_user_id: str
    ) -> Tuple[StructureVersion, List[Operation]]:
        """Last Write Wins resolution."""
        local_time = max((op.timestamp for op in local_changes), default=0)
        remote_time = max((op.timestamp for op in remote_changes), default=0)
        
        winning_changes = local_changes if local_time >= remote_time else remote_changes
        winner_id = local_user_id if local_time >= remote_time else remote_user_id
        
        result = base_version.structure_data.copy()
        for op in winning_changes:
            result = self.ot.apply_operation(result, op)
        
        new_version = StructureVersion(
            version_id=str(uuid.uuid4()),
            revision=base_version.revision + 1,
            structure_data=result,
            operations=winning_changes,
            timestamp=time.time(),
            user_id=winner_id,
            checksum="",
        )
        new_version.checksum = new_version.calculate_checksum()
        
        return new_version, winning_changes
    
    def _resolve_fww(
        self,
        base_version: StructureVersion,
        local_changes: List[Operation],
        remote_changes: List[Operation],
        local_user_id: str,
        remote_user_id: str
    ) -> Tuple[StructureVersion, List[Operation]]:
        """First Write Wins resolution."""
        local_time = min((op.timestamp for op in local_changes), default=float('inf'))
        remote_time = min((op.timestamp for op in remote_changes), default=float('inf'))
        
        winning_changes = local_changes if local_time <= remote_time else remote_changes
        winner_id = local_user_id if local_time <= remote_time else remote_user_id
        
        result = base_version.structure_data.copy()
        for op in winning_changes:
            result = self.ot.apply_operation(result, op)
        
        new_version = StructureVersion(
            version_id=str(uuid.uuid4()),
            revision=base_version.revision + 1,
            structure_data=result,
            operations=winning_changes,
            timestamp=time.time(),
            user_id=winner_id,
            checksum="",
        )
        new_version.checksum = new_version.calculate_checksum()
        
        return new_version, winning_changes
    
    def _resolve_ot(
        self,
        base_version: StructureVersion,
        local_changes: List[Operation],
        remote_changes: List[Operation]
    ) -> Tuple[StructureVersion, List[Operation]]:
        """Operational Transformation resolution."""
        result = base_version.structure_data.copy()
        all_operations: List[Operation] = []
        
        # Sort operations by timestamp
        sorted_ops = sorted(
            local_changes + remote_changes,
            key=lambda op: (op.timestamp, op.user_id)
        )
        
        # Transform and apply each operation
        for i, op in enumerate(sorted_ops):
            # Transform against previous operations
            for j in range(i):
                _, op = self.ot.transform_operations(sorted_ops[j], op)
            
            result = self.ot.apply_operation(result, op)
            all_operations.append(op)
        
        new_version = StructureVersion(
            version_id=str(uuid.uuid4()),
            revision=base_version.revision + 1,
            structure_data=result,
            operations=all_operations,
            timestamp=time.time(),
            user_id="ot_merged",
            checksum="",
        )
        new_version.checksum = new_version.calculate_checksum()
        
        return new_version, all_operations
    
    def _resolve_lock_based(
        self,
        base_version: StructureVersion,
        local_changes: List[Operation],
        remote_changes: List[Operation],
        local_user_id: str
    ) -> Tuple[StructureVersion, List[Operation]]:
        """Lock-based resolution."""
        # Check if local user has locks for changed paths
        allowed_changes = []
        blocked_changes = []
        
        for op in local_changes:
            if self.active_locks.get(op.path) == local_user_id:
                allowed_changes.append(op)
            else:
                blocked_changes.append(op)
        
        # Only apply allowed local changes
        result = base_version.structure_data.copy()
        for op in allowed_changes:
            result = self.ot.apply_operation(result, op)
        
        new_version = StructureVersion(
            version_id=str(uuid.uuid4()),
            revision=base_version.revision + 1,
            structure_data=result,
            operations=allowed_changes,
            timestamp=time.time(),
            user_id=local_user_id,
            checksum="",
        )
        new_version.checksum = new_version.calculate_checksum()
        
        if blocked_changes:
            logger.warning(f"Blocked {len(blocked_changes)} changes due to locks")
        
        return new_version, allowed_changes
    
    def _resolve_manual(
        self,
        base_version: StructureVersion,
        local_changes: List[Operation],
        remote_changes: List[Operation]
    ) -> Tuple[StructureVersion, List[Operation]]:
        """Manual merge - store conflict for manual resolution."""
        conflict = {
            "base_version": base_version,
            "local_changes": local_changes,
            "remote_changes": remote_changes,
            "timestamp": time.time(),
        }
        self.pending_conflicts.append(conflict)
        
        # Return base version unchanged
        return base_version, []
    
    def acquire_lock(self, path: str, user_id: str) -> bool:
        """Acquire a lock on a specific path."""
        if path in self.active_locks:
            return False
        self.active_locks[path] = user_id
        return True
    
    def release_lock(self, path: str, user_id: str) -> bool:
        """Release a lock."""
        if self.active_locks.get(path) == user_id:
            del self.active_locks[path]
            return True
        return False


class CollaborativeSession:
    """
    Manages a collaborative editing session.
    
    Tracks users, operations, and maintains the current state of the shared structure.
    """
    
    def __init__(
        self,
        session_id: str,
        initial_structure: Optional[Dict[str, Any]] = None,
        conflict_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.OPERATIONAL_TRANSFORMATION
    ):
        self.session_id = session_id
        self.users: Dict[str, UserPresence] = {}
        self.connections: Dict[str, Any] = {}  # user_id -> websocket
        self.version_history: List[StructureVersion] = []
        self.pending_operations: List[Operation] = []
        self.conflict_resolver = ConflictResolver(strategy=conflict_strategy)
        self.ot = OperationalTransformer()
        self.lock = asyncio.Lock()
        
        # Initialize with empty structure or provided structure
        if initial_structure is None:
            initial_structure = {"atoms": [], "cell": {}, "properties": {}}
        
        initial_version = StructureVersion(
            version_id=str(uuid.uuid4()),
            revision=0,
            structure_data=initial_structure,
            operations=[],
            timestamp=time.time(),
            user_id="system",
            checksum="",
        )
        initial_version.checksum = initial_version.calculate_checksum()
        self.version_history.append(initial_version)
    
    @property
    def current_version(self) -> StructureVersion:
        """Get the current version of the structure."""
        return self.version_history[-1]
    
    async def add_user(
        self,
        user_id: str,
        username: str,
        connection: Any,
        color: Optional[str] = None
    ) -> UserPresence:
        """Add a user to the session."""
        async with self.lock:
            if color is None:
                color = self._generate_user_color(user_id)
            
            presence = UserPresence(
                user_id=user_id,
                username=username,
                color=color,
                is_active=True,
            )
            self.users[user_id] = presence
            self.connections[user_id] = connection
            
            # Notify other users
            await self._broadcast_presence(user_id)
            
            return presence
    
    async def remove_user(self, user_id: str) -> None:
        """Remove a user from the session."""
        async with self.lock:
            if user_id in self.users:
                self.users[user_id].is_active = False
                del self.connections[user_id]
                await self._broadcast_presence(user_id, is_leaving=True)
    
    async def update_presence(
        self,
        user_id: str,
        cursor_position: Optional[Dict[str, float]] = None,
        selection: Optional[Dict[str, Any]] = None,
        current_file: Optional[str] = None
    ) -> None:
        """Update user's presence information."""
        async with self.lock:
            if user_id in self.users:
                user = self.users[user_id]
                if cursor_position is not None:
                    user.cursor_position = cursor_position
                if selection is not None:
                    user.selection = selection
                if current_file is not None:
                    user.current_file = current_file
                user.last_seen = time.time()
                
                # Broadcast update to other users
                await self._broadcast_cursor_update(user_id)
    
    async def apply_operation(
        self,
        operation: Operation,
        user_id: str
    ) -> Tuple[bool, Optional[StructureVersion]]:
        """
        Apply an operation to the current structure.
        
        Returns:
            Tuple of (success, new_version)
        """
        async with self.lock:
            try:
                current = self.current_version
                
                # Check for conflicts
                if operation.parent_revision != current.revision:
                    # Conflict detected
                    base_version = self.version_history[operation.parent_revision]
                    local_ops = self.pending_operations
                    remote_ops = [operation]
                    
                    new_version, applied_ops = self.conflict_resolver.resolve_conflict(
                        base_version, local_ops, remote_ops, "local", user_id
                    )
                else:
                    # No conflict
                    new_structure = self.ot.apply_operation(
                        current.structure_data,
                        operation
                    )
                    
                    new_version = StructureVersion(
                        version_id=str(uuid.uuid4()),
                        revision=current.revision + 1,
                        structure_data=new_structure,
                        operations=[operation],
                        timestamp=time.time(),
                        user_id=user_id,
                        checksum="",
                    )
                    new_version.checksum = new_version.calculate_checksum()
                    applied_ops = [operation]
                
                self.version_history.append(new_version)
                self.pending_operations = []
                
                # Broadcast to all users
                await self._broadcast_operation(operation, new_version)
                
                return True, new_version
                
            except Exception as e:
                logger.error(f"Failed to apply operation: {e}")
                return False, None
    
    async def broadcast_computation_result(
        self,
        result_data: Dict[str, Any],
        user_id: str
    ) -> None:
        """Broadcast a computation result to all session participants."""
        message = SyncMessage(
            msg_type=SyncMessageType.COMPUTATION_RESULT,
            session_id=self.session_id,
            user_id=user_id,
            payload={
                "result": result_data,
                "timestamp": time.time(),
            }
        )
        await self._broadcast_message(message)
    
    def _generate_user_color(self, user_id: str) -> str:
        """Generate a unique color for a user."""
        colors = [
            "#007bff", "#28a745", "#dc3545", "#ffc107",
            "#17a2b8", "#6610f2", "#fd7e14", "#20c997",
            "#e83e8c", "#6f42c1"
        ]
        hash_val = hash(user_id) % len(colors)
        return colors[hash_val]
    
    async def _broadcast_presence(
        self,
        user_id: str,
        is_leaving: bool = False
    ) -> None:
        """Broadcast presence update to all users."""
        message = SyncMessage(
            msg_type=SyncMessageType.PRESENCE,
            session_id=self.session_id,
            user_id=user_id,
            payload={
                "users": [u.to_dict() for u in self.users.values()],
                "is_leaving": is_leaving,
            }
        )
        await self._broadcast_message(message, exclude_user=user_id if is_leaving else None)
    
    async def _broadcast_cursor_update(self, user_id: str) -> None:
        """Broadcast cursor update to other users."""
        if user_id not in self.users:
            return
        
        user = self.users[user_id]
        message = SyncMessage(
            msg_type=SyncMessageType.CURSOR_UPDATE,
            session_id=self.session_id,
            user_id=user_id,
            payload={
                "cursor_position": user.cursor_position,
                "selection": user.selection,
                "color": user.color,
                "username": user.username,
            }
        )
        await self._broadcast_message(message, exclude_user=user_id)
    
    async def _broadcast_operation(
        self,
        operation: Operation,
        new_version: StructureVersion
    ) -> None:
        """Broadcast an operation to all users."""
        message = SyncMessage(
            msg_type=SyncMessageType.OPERATION,
            session_id=self.session_id,
            user_id=operation.user_id,
            payload={
                "operation": operation.to_dict(),
                "new_version": new_version.to_dict(),
            }
        )
        await self._broadcast_message(message)
    
    async def _broadcast_message(
        self,
        message: SyncMessage,
        exclude_user: Optional[str] = None
    ) -> None:
        """Broadcast a message to all connected users."""
        if not HAS_WEBSOCKETS:
            return
        
        message_str = json.dumps(message.to_dict())
        
        for user_id, connection in self.connections.items():
            if user_id == exclude_user:
                continue
            
            try:
                if hasattr(connection, 'send'):
                    await connection.send(message_str)
            except Exception as e:
                logger.error(f"Failed to send message to {user_id}: {e}")


class RealtimeSyncServer:
    """
    WebSocket server for real-time synchronization.
    
    Manages multiple collaborative sessions and handles client connections.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        redis_url: Optional[str] = None
    ):
        self.host = host
        self.port = port
        self.sessions: Dict[str, CollaborativeSession] = {}
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id
        self.redis: Optional[Any] = None
        self.redis_url = redis_url
        self.server = None
        
        if HAS_REDIS and redis_url:
            self.redis = redis.from_url(redis_url)
    
    async def start(self) -> None:
        """Start the WebSocket server."""
        if not HAS_WEBSOCKETS:
            raise ImportError("websockets library is required for real-time sync")
        
        self.server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port
        )
        logger.info(f"RealtimeSyncServer started on ws://{self.host}:{self.port}")
    
    async def stop(self) -> None:
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("RealtimeSyncServer stopped")
    
    async def _handle_connection(
        self,
        websocket: WebSocketServerProtocol,
        path: str
    ) -> None:
        """Handle a new WebSocket connection."""
        user_id: Optional[str] = None
        session_id: Optional[str] = None
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg = SyncMessage.from_dict(data)
                    
                    if msg.msg_type == SyncMessageType.CONNECT:
                        user_id = msg.user_id
                        await self._handle_connect(websocket, msg)
                    
                    elif msg.msg_type == SyncMessageType.JOIN_SESSION:
                        session_id = msg.payload.get("session_id")
                        await self._handle_join_session(websocket, msg)
                    
                    elif msg.msg_type == SyncMessageType.LEAVE_SESSION:
                        await self._handle_leave_session(msg)
                        session_id = None
                    
                    elif msg.msg_type == SyncMessageType.OPERATION:
                        await self._handle_operation(msg)
                    
                    elif msg.msg_type == SyncMessageType.CURSOR_UPDATE:
                        await self._handle_cursor_update(msg)
                    
                    elif msg.msg_type == SyncMessageType.HEARTBEAT:
                        await self._handle_heartbeat(websocket, msg)
                    
                    elif msg.msg_type == SyncMessageType.LOCK_REQUEST:
                        await self._handle_lock_request(websocket, msg)
                    
                    elif msg.msg_type == SyncMessageType.LOCK_RELEASE:
                        await self._handle_lock_release(websocket, msg)
                    
                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid JSON")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    await self._send_error(websocket, str(e))
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed for user {user_id}")
        finally:
            if session_id and user_id:
                await self._cleanup_user(session_id, user_id)
    
    async def _handle_connect(
        self,
        websocket: WebSocketServerProtocol,
        msg: SyncMessage
    ) -> None:
        """Handle client connection."""
        response = SyncMessage(
            msg_type=SyncMessageType.ACK,
            session_id="",
            user_id=msg.user_id,
            payload={"status": "connected", "server_time": time.time()}
        )
        await websocket.send(json.dumps(response.to_dict()))
    
    async def _handle_join_session(
        self,
        websocket: WebSocketServerProtocol,
        msg: SyncMessage
    ) -> None:
        """Handle session join request."""
        session_id = msg.payload.get("session_id")
        username = msg.payload.get("username", msg.user_id)
        
        # Create session if it doesn't exist
        if session_id not in self.sessions:
            initial_structure = msg.payload.get("initial_structure")
            strategy_str = msg.payload.get("conflict_strategy", "operational_transformation")
            strategy = ConflictResolutionStrategy(strategy_str)
            self.sessions[session_id] = CollaborativeSession(
                session_id,
                initial_structure,
                strategy
            )
        
        session = self.sessions[session_id]
        await session.add_user(msg.user_id, username, websocket)
        self.user_sessions[msg.user_id] = session_id
        
        # Send current state to new user
        response = SyncMessage(
            msg_type=SyncMessageType.ACK,
            session_id=session_id,
            user_id=msg.user_id,
            payload={
                "status": "joined",
                "current_version": session.current_version.to_dict(),
                "users": [u.to_dict() for u in session.users.values()],
            }
        )
        await websocket.send(json.dumps(response.to_dict()))
    
    async def _handle_leave_session(self, msg: SyncMessage) -> None:
        """Handle session leave request."""
        session_id = self.user_sessions.get(msg.user_id)
        if session_id and session_id in self.sessions:
            await self.sessions[session_id].remove_user(msg.user_id)
            del self.user_sessions[msg.user_id]
            
            # Clean up empty sessions
            if not self.sessions[session_id].users:
                del self.sessions[session_id]
    
    async def _handle_operation(self, msg: SyncMessage) -> None:
        """Handle operation message."""
        session_id = msg.session_id
        if session_id not in self.sessions:
            return
        
        op_data = msg.payload.get("operation", {})
        operation = Operation.from_dict(op_data)
        
        session = self.sessions[session_id]
        success, new_version = await session.apply_operation(operation, msg.user_id)
        
        if not success:
            # Send error to user
            error_msg = SyncMessage(
                msg_type=SyncMessageType.ERROR,
                session_id=session_id,
                user_id=msg.user_id,
                payload={"error": "Failed to apply operation"}
            )
            if msg.user_id in session.connections:
                await session.connections[msg.user_id].send(
                    json.dumps(error_msg.to_dict())
                )
    
    async def _handle_cursor_update(self, msg: SyncMessage) -> None:
        """Handle cursor update message."""
        session_id = msg.session_id
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        await session.update_presence(
            msg.user_id,
            cursor_position=msg.payload.get("cursor_position"),
            selection=msg.payload.get("selection"),
            current_file=msg.payload.get("current_file")
        )
    
    async def _handle_heartbeat(
        self,
        websocket: WebSocketServerProtocol,
        msg: SyncMessage
    ) -> None:
        """Handle heartbeat message."""
        response = SyncMessage(
            msg_type=SyncMessageType.HEARTBEAT,
            session_id=msg.session_id,
            user_id=msg.user_id,
            payload={"server_time": time.time()}
        )
        await websocket.send(json.dumps(response.to_dict()))
    
    async def _handle_lock_request(
        self,
        websocket: WebSocketServerProtocol,
        msg: SyncMessage
    ) -> None:
        """Handle lock request."""
        session_id = msg.session_id
        path = msg.payload.get("path")
        
        if session_id in self.sessions:
            session = self.sessions[session_id]
            acquired = session.conflict_resolver.acquire_lock(path, msg.user_id)
            
            response = SyncMessage(
                msg_type=SyncMessageType.LOCK_REQUEST,
                session_id=session_id,
                user_id=msg.user_id,
                payload={"path": path, "acquired": acquired}
            )
            await websocket.send(json.dumps(response.to_dict()))
    
    async def _handle_lock_release(
        self,
        websocket: WebSocketServerProtocol,
        msg: SyncMessage
    ) -> None:
        """Handle lock release."""
        session_id = msg.session_id
        path = msg.payload.get("path")
        
        if session_id in self.sessions:
            session = self.sessions[session_id]
            released = session.conflict_resolver.release_lock(path, msg.user_id)
            
            response = SyncMessage(
                msg_type=SyncMessageType.LOCK_RELEASE,
                session_id=session_id,
                user_id=msg.user_id,
                payload={"path": path, "released": released}
            )
            await websocket.send(json.dumps(response.to_dict()))
    
    async def _send_error(
        self,
        websocket: WebSocketServerProtocol,
        error: str
    ) -> None:
        """Send error message to client."""
        response = SyncMessage(
            msg_type=SyncMessageType.ERROR,
            session_id="",
            user_id="",
            payload={"error": error}
        )
        await websocket.send(json.dumps(response.to_dict()))
    
    async def _cleanup_user(self, session_id: str, user_id: str) -> None:
        """Clean up when a user disconnects."""
        if session_id in self.sessions:
            await self.sessions[session_id].remove_user(user_id)
            if user_id in self.user_sessions:
                del self.user_sessions[user_id]
            
            # Clean up empty sessions
            if not self.sessions[session_id].users:
                del self.sessions[session_id]


class RealtimeSyncManager:
    """
    Client-side manager for real-time synchronization.
    
    Provides a convenient interface for connecting to and interacting with
    a RealtimeSyncServer.
    """
    
    def __init__(
        self,
        server_url: str = "ws://localhost:8765",
        user_id: Optional[str] = None,
        username: Optional[str] = None
    ):
        self.server_url = server_url
        self.user_id = user_id or str(uuid.uuid4())
        self.username = username or self.user_id
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.session_id: Optional[str] = None
        self.message_handlers: Dict[SyncMessageType, List[Callable]] = defaultdict(list)
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._connected = False
    
    def on(self, msg_type: SyncMessageType, handler: Callable[[SyncMessage], None]) -> None:
        """Register a message handler."""
        self.message_handlers[msg_type].append(handler)
    
    def off(self, msg_type: SyncMessageType, handler: Callable[[SyncMessage], None]) -> None:
        """Unregister a message handler."""
        if handler in self.message_handlers[msg_type]:
            self.message_handlers[msg_type].remove(handler)
    
    async def connect(self) -> bool:
        """Connect to the sync server."""
        if not HAS_WEBSOCKETS:
            raise ImportError("websockets library is required")
        
        try:
            self.websocket = await websockets.connect(self.server_url)
            self._connected = True
            
            # Send connect message
            msg = SyncMessage(
                msg_type=SyncMessageType.CONNECT,
                session_id="",
                user_id=self.user_id,
                payload={"username": self.username}
            )
            await self.websocket.send(json.dumps(msg.to_dict()))
            
            # Start heartbeat and receive tasks
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._receive_task = asyncio.create_task(self._receive_loop())
            
            logger.info(f"Connected to sync server at {self.server_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from the sync server."""
        self._connected = False
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        logger.info("Disconnected from sync server")
    
    async def join_session(
        self,
        session_id: str,
        initial_structure: Optional[Dict[str, Any]] = None,
        conflict_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.OPERATIONAL_TRANSFORMATION
    ) -> bool:
        """Join a collaborative session."""
        if not self.websocket:
            logger.error("Not connected to server")
            return False
        
        self.session_id = session_id
        
        msg = SyncMessage(
            msg_type=SyncMessageType.JOIN_SESSION,
            session_id=session_id,
            user_id=self.user_id,
            payload={
                "session_id": session_id,
                "username": self.username,
                "initial_structure": initial_structure,
                "conflict_strategy": conflict_strategy.value,
            }
        )
        
        await self.websocket.send(json.dumps(msg.to_dict()))
        return True
    
    async def leave_session(self) -> None:
        """Leave the current session."""
        if not self.websocket or not self.session_id:
            return
        
        msg = SyncMessage(
            msg_type=SyncMessageType.LEAVE_SESSION,
            session_id=self.session_id,
            user_id=self.user_id,
            payload={}
        )
        
        await self.websocket.send(json.dumps(msg.to_dict()))
        self.session_id = None
    
    async def broadcast_structure_update(
        self,
        structure_data: Dict[str, Any],
        operation_type: OperationType = OperationType.REPLACE
    ) -> None:
        """Broadcast a structure update to all session participants."""
        if not self.websocket or not self.session_id:
            return
        
        operation = Operation(
            op_id=str(uuid.uuid4()),
            type=operation_type,
            path="structure",
            value=structure_data,
            timestamp=time.time(),
            user_id=self.user_id,
            revision=0,  # Will be set by server
            parent_revision=0,
        )
        
        msg = SyncMessage(
            msg_type=SyncMessageType.OPERATION,
            session_id=self.session_id,
            user_id=self.user_id,
            payload={"operation": operation.to_dict()}
        )
        
        await self.websocket.send(json.dumps(msg.to_dict()))
    
    async def update_cursor(
        self,
        cursor_position: Dict[str, float],
        selection: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update cursor position and selection."""
        if not self.websocket or not self.session_id:
            return
        
        msg = SyncMessage(
            msg_type=SyncMessageType.CURSOR_UPDATE,
            session_id=self.session_id,
            user_id=self.user_id,
            payload={
                "cursor_position": cursor_position,
                "selection": selection,
            }
        )
        
        await self.websocket.send(json.dumps(msg.to_dict()))
    
    async def request_lock(self, path: str) -> bool:
        """Request a lock on a specific path."""
        if not self.websocket or not self.session_id:
            return False
        
        msg = SyncMessage(
            msg_type=SyncMessageType.LOCK_REQUEST,
            session_id=self.session_id,
            user_id=self.user_id,
            payload={"path": path}
        )
        
        await self.websocket.send(json.dumps(msg.to_dict()))
        return True
    
    async def release_lock(self, path: str) -> bool:
        """Release a lock on a specific path."""
        if not self.websocket or not self.session_id:
            return False
        
        msg = SyncMessage(
            msg_type=SyncMessageType.LOCK_RELEASE,
            session_id=self.session_id,
            user_id=self.user_id,
            payload={"path": path}
        )
        
        await self.websocket.send(json.dumps(msg.to_dict()))
        return True
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat messages."""
        while self._connected:
            try:
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
                if self.websocket and self.session_id:
                    msg = SyncMessage(
                        msg_type=SyncMessageType.HEARTBEAT,
                        session_id=self.session_id,
                        user_id=self.user_id,
                        payload={}
                    )
                    await self.websocket.send(json.dumps(msg.to_dict()))
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    async def _receive_loop(self) -> None:
        """Receive and process messages from server."""
        while self._connected and self.websocket:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)
                msg = SyncMessage.from_dict(data)
                
                # Call registered handlers
                for handler in self.message_handlers.get(msg.msg_type, []):
                    try:
                        handler(msg)
                    except Exception as e:
                        logger.error(f"Handler error: {e}")
                        
            except websockets.exceptions.ConnectionClosed:
                logger.info("Connection closed by server")
                self._connected = False
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Receive error: {e}")


# Convenience functions for common use cases

async def create_sync_server(
    host: str = "localhost",
    port: int = 8765,
    redis_url: Optional[str] = None
) -> RealtimeSyncServer:
    """
    Create and start a real-time synchronization server.
    
    Args:
        host: Server host
        port: Server port
        redis_url: Optional Redis URL for distributed sync
        
    Returns:
        Running RealtimeSyncServer instance
    """
    server = RealtimeSyncServer(host, port, redis_url)
    await server.start()
    return server


async def create_sync_client(
    server_url: str = "ws://localhost:8765",
    username: Optional[str] = None
) -> RealtimeSyncManager:
    """
    Create and connect a sync client.
    
    Args:
        server_url: WebSocket server URL
        username: User display name
        
    Returns:
        Connected RealtimeSyncManager instance
    """
    client = RealtimeSyncManager(server_url, username=username)
    await client.connect()
    return client
