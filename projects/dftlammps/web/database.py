#!/usr/bin/env python3
"""
Database Module
===============
Database interface for storing simulation results and metadata.

Supports:
- PostgreSQL (primary database for structured data)
- MongoDB (for flexible document storage)
- File storage (for large result files)

Author: DFT-LAMMPS Web Team
Version: 1.0.0
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager
import uuid
import hashlib

# SQLAlchemy for PostgreSQL
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, Boolean, Text, ForeignKey, JSON, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.sql import func

# MongoDB (optional)
try:
    from pymongo import MongoClient, ASCENDING, DESCENDING
    from bson import ObjectId
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    logging.warning("pymongo not installed. MongoDB features disabled.")

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# PostgreSQL configuration
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://dftlammps:dftlammps@localhost:5432/dftlammps"
)

# MongoDB configuration
MONGODB_URL = os.environ.get("MONGODB_URL", "mongodb://localhost:27017/")
MONGODB_DB_NAME = os.environ.get("MONGODB_DB_NAME", "dftlammps")

# File storage configuration
RESULTS_STORAGE_PATH = Path(os.environ.get("RESULTS_STORAGE_PATH", "./storage/results"))
RESULTS_STORAGE_PATH.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SQLAlchemy Models (PostgreSQL)
# =============================================================================

Base = declarative_base()

class User(Base):
    """User model."""
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False)
    full_name = Column(String(100))
    institution = Column(String(100))
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="researcher")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    tasks = relationship("Task", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")

class APIKey(Base):
    """API key model."""
    __tablename__ = "api_keys"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    name = Column(String(100))
    key_hash = Column(String(64), nullable=False)
    permissions = Column(JSON, default=list)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    last_used_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")

class Task(Base):
    """Task model for tracking simulations."""
    __tablename__ = "tasks"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    workflow_type = Column(String(50), nullable=False)
    status = Column(String(20), default="pending", index=True)
    priority = Column(Integer, default=5)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Progress
    progress = Column(Float, default=0.0)
    current_stage = Column(String(50))
    message = Column(Text)
    error_message = Column(Text)
    
    # Configuration
    config = Column(JSON)
    
    # Runtime info
    celery_task_id = Column(String(100))
    runtime_seconds = Column(Float)
    
    # Relationships
    user = relationship("User", back_populates="tasks")
    results = relationship("TaskResult", back_populates="task", uselist=False, cascade="all, delete-orphan")
    logs = relationship("TaskLog", back_populates="task", cascade="all, delete-orphan")
    files = relationship("ResultFile", back_populates="task", cascade="all, delete-orphan")

class TaskResult(Base):
    """Task results summary."""
    __tablename__ = "task_results"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id = Column(String(36), ForeignKey("tasks.id"), unique=True, nullable=False)
    
    # Results metadata
    success = Column(Boolean)
    summary = Column(JSON)  # Brief summary for quick access
    
    # Metrics
    dft_energy = Column(Float)
    diffusion_coefficient = Column(Float)
    conductivity = Column(Float)
    activation_energy = Column(Float)
    
    # Full results stored in MongoDB or files
    results_ref = Column(String(200))  # Reference to full results
    
    # Relationships
    task = relationship("Task", back_populates="results")

class TaskLog(Base):
    """Task execution logs."""
    __tablename__ = "task_logs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id = Column(String(36), ForeignKey("tasks.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    level = Column(String(20), default="INFO")  # DEBUG, INFO, WARNING, ERROR
    stage = Column(String(50))
    message = Column(Text)
    
    # Relationships
    task = relationship("Task", back_populates="logs")

class ResultFile(Base):
    """Result file metadata."""
    __tablename__ = "result_files"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id = Column(String(36), ForeignKey("tasks.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50))  # structure, trajectory, model, report, etc.
    file_size = Column(Integer)  # bytes
    checksum = Column(String(64))  # SHA-256
    storage_path = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)  # Additional file metadata
    
    # Relationships
    task = relationship("Task", back_populates="files")

class Structure(Base):
    """Cached structures from Materials Project."""
    __tablename__ = "structures"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    material_id = Column(String(20), unique=True, index=True)  # mp-1234
    formula = Column(String(50), index=True)
    name = Column(String(200))
    
    # Structure data
    lattice = Column(JSON)
    positions = Column(JSON)
    species = Column(JSON)
    
    # Properties
    energy_per_atom = Column(Float)
    band_gap = Column(Float)
    
    # Source
    source = Column(String(20), default="materials_project")  # mp, local, upload
    source_ref = Column(String(100))
    
    cached_at = Column(DateTime, default=datetime.utcnow)

# =============================================================================
# Database Manager
# =============================================================================

class DatabaseManager:
    """Main database manager for PostgreSQL."""
    
    def __init__(self, database_url: str = DATABASE_URL):
        self.engine = create_engine(database_url, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
    
    @contextmanager
    def get_session(self) -> Session:
        """Get database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    # User operations
    def create_user(self, user_data: Dict) -> User:
        """Create a new user."""
        with self.get_session() as session:
            user = User(**user_data)
            session.add(user)
            session.flush()  # Flush to get ID
            return user
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        with self.get_session() as session:
            return session.query(User).filter(User.id == user_id).first()
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        with self.get_session() as session:
            return session.query(User).filter(User.username == username).first()
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        with self.get_session() as session:
            return session.query(User).filter(User.email == email).first()
    
    def update_user(self, user_id: str, updates: Dict) -> bool:
        """Update user data."""
        with self.get_session() as session:
            result = session.query(User).filter(User.id == user_id).update(updates)
            return result > 0
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        with self.get_session() as session:
            result = session.query(User).filter(User.id == user_id).delete()
            return result > 0
    
    def list_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        """List users."""
        with self.get_session() as session:
            return session.query(User).offset(skip).limit(limit).all()
    
    # Task operations
    def create_task(self, task_data: Dict) -> Task:
        """Create a new task."""
        with self.get_session() as session:
            task = Task(**task_data)
            session.add(task)
            session.flush()
            return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        with self.get_session() as session:
            return session.query(Task).filter(Task.id == task_id).first()
    
    def update_task(self, task_id: str, updates: Dict) -> bool:
        """Update task data."""
        with self.get_session() as session:
            result = session.query(Task).filter(Task.id == task_id).update(updates)
            return result > 0
    
    def delete_task(self, task_id: str) -> bool:
        """Delete a task."""
        with self.get_session() as session:
            result = session.query(Task).filter(Task.id == task_id).delete()
            return result > 0
    
    def list_tasks(self, user_id: Optional[str] = None, 
                   status: Optional[str] = None,
                   skip: int = 0, limit: int = 100) -> List[Task]:
        """List tasks with filtering."""
        with self.get_session() as session:
            query = session.query(Task)
            
            if user_id:
                query = query.filter(Task.user_id == user_id)
            if status:
                query = query.filter(Task.status == status)
            
            return query.order_by(Task.created_at.desc()).offset(skip).limit(limit).all()
    
    def get_task_count(self, user_id: Optional[str] = None, 
                       status: Optional[str] = None) -> int:
        """Get task count with filtering."""
        with self.get_session() as session:
            query = session.query(Task)
            
            if user_id:
                query = query.filter(Task.user_id == user_id)
            if status:
                query = query.filter(Task.status == status)
            
            return query.count()
    
    # Task result operations
    def create_task_result(self, result_data: Dict) -> TaskResult:
        """Create task result entry."""
        with self.get_session() as session:
            result = TaskResult(**result_data)
            session.add(result)
            session.flush()
            return result
    
    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get result for a task."""
        with self.get_session() as session:
            return session.query(TaskResult).filter(TaskResult.task_id == task_id).first()
    
    # Task log operations
    def add_task_log(self, task_id: str, level: str, message: str, 
                    stage: str = "") -> TaskLog:
        """Add a log entry for a task."""
        with self.get_session() as session:
            log = TaskLog(
                task_id=task_id,
                level=level,
                message=message,
                stage=stage
            )
            session.add(log)
            session.flush()
            return log
    
    def get_task_logs(self, task_id: str, level: Optional[str] = None) -> List[TaskLog]:
        """Get logs for a task."""
        with self.get_session() as session:
            query = session.query(TaskLog).filter(TaskLog.task_id == task_id)
            if level:
                query = query.filter(TaskLog.level == level)
            return query.order_by(TaskLog.timestamp).all()
    
    # Result file operations
    def add_result_file(self, file_data: Dict) -> ResultFile:
        """Add result file metadata."""
        with self.get_session() as session:
            result_file = ResultFile(**file_data)
            session.add(result_file)
            session.flush()
            return result_file
    
    def get_result_files(self, task_id: str, file_type: Optional[str] = None) -> List[ResultFile]:
        """Get result files for a task."""
        with self.get_session() as session:
            query = session.query(ResultFile).filter(ResultFile.task_id == task_id)
            if file_type:
                query = query.filter(ResultFile.file_type == file_type)
            return query.all()
    
    def get_result_file(self, file_id: str) -> Optional[ResultFile]:
        """Get result file by ID."""}
        with self.get_session() as session:
            return session.query(ResultFile).filter(ResultFile.id == file_id).first()
    
    # Structure operations
    def cache_structure(self, structure_data: Dict) -> Structure:
        """Cache a structure."""
        with self.get_session() as session:
            # Check if already exists
            existing = session.query(Structure).filter(
                Structure.material_id == structure_data.get('material_id')
            ).first()
            
            if existing:
                # Update
                for key, value in structure_data.items():
                    setattr(existing, key, value)
                return existing
            
            structure = Structure(**structure_data)
            session.add(structure)
            session.flush()
            return structure
    
    def get_structure(self, material_id: str) -> Optional[Structure]:
        """Get cached structure by material ID."""
        with self.get_session() as session:
            return session.query(Structure).filter(
                Structure.material_id == material_id
            ).first()
    
    def search_structures(self, formula: Optional[str] = None,
                         skip: int = 0, limit: int = 100) -> List[Structure]:
        """Search cached structures."""
        with self.get_session() as session:
            query = session.query(Structure)
            if formula:
                query = query.filter(Structure.formula.like(f"%{formula}%"))
            return query.offset(skip).limit(limit).all()
    
    # Statistics
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        with self.get_session() as session:
            return {
                "total_users": session.query(User).count(),
                "total_tasks": session.query(Task).count(),
                "pending_tasks": session.query(Task).filter(Task.status == "pending").count(),
                "running_tasks": session.query(Task).filter(Task.status == "running").count(),
                "completed_tasks": session.query(Task).filter(Task.status == "completed").count(),
                "failed_tasks": session.query(Task).filter(Task.status == "failed").count(),
                "cached_structures": session.query(Structure).count(),
                "total_files": session.query(ResultFile).count(),
            }

# =============================================================================
# MongoDB Document Store
# =============================================================================

class DocumentStore:
    """MongoDB document store for flexible data."""
    
    def __init__(self, mongodb_url: str = MONGODB_URL, db_name: str = MONGODB_DB_NAME):
        if not MONGODB_AVAILABLE:
            raise ImportError("pymongo is required for DocumentStore")
        
        self.client = MongoClient(mongodb_url)
        self.db = self.client[db_name]
        
        # Collections
        self.results = self.db["results"]
        self.trajectories = self.db["trajectories"]
        self.analysis = self.db["analysis"]
        
        # Create indexes
        self._create_indexes()
    
    def _create_indexes(self):
        """Create database indexes."""
        self.results.create_index([("task_id", ASCENDING)], unique=True)
        self.results.create_index([("user_id", ASCENDING)])
        self.results.create_index([("created_at", DESCENDING)])
        
        self.trajectories.create_index([("task_id", ASCENDING)])
        self.trajectories.create_index([("material_id", ASCENDING)])
        
        self.analysis.create_index([("task_id", ASCENDING)])
    
    def store_result(self, task_id: str, user_id: str, results: Dict) -> str:
        """Store task results."""
        document = {
            "task_id": task_id,
            "user_id": user_id,
            "results": results,
            "created_at": datetime.utcnow()
        }
        
        result = self.results.replace_one(
            {"task_id": task_id},
            document,
            upsert=True
        )
        
        return str(result.upserted_id or result.matched_count)
    
    def get_result(self, task_id: str) -> Optional[Dict]:
        """Get task results."""
        return self.results.find_one({"task_id": task_id}, {"_id": 0})
    
    def store_trajectory(self, task_id: str, trajectory_data: Dict) -> str:
        """Store MD trajectory."""
        document = {
            "task_id": task_id,
            "trajectory": trajectory_data,
            "created_at": datetime.utcnow()
        }
        
        result = self.trajectories.insert_one(document)
        return str(result.inserted_id)
    
    def get_trajectory(self, task_id: str) -> Optional[Dict]:
        """Get trajectory data."""
        doc = self.trajectories.find_one({"task_id": task_id}, {"_id": 0})
        return doc.get("trajectory") if doc else None
    
    def store_analysis(self, task_id: str, analysis_type: str, data: Dict) -> str:
        """Store analysis results."""
        document = {
            "task_id": task_id,
            "analysis_type": analysis_type,
            "data": data,
            "created_at": datetime.utcnow()
        }
        
        result = self.analysis.replace_one(
            {"task_id": task_id, "analysis_type": analysis_type},
            document,
            upsert=True
        )
        
        return str(result.upserted_id or result.matched_count)
    
    def get_analysis(self, task_id: str, analysis_type: Optional[str] = None) -> Union[Dict, List[Dict]]:
        """Get analysis results."""
        if analysis_type:
            doc = self.analysis.find_one(
                {"task_id": task_id, "analysis_type": analysis_type},
                {"_id": 0}
            )
            return doc.get("data") if doc else None
        else:
            docs = self.analysis.find({"task_id": task_id}, {"_id": 0})
            return [{"type": d["analysis_type"], "data": d["data"]} for d in docs]
    
    def delete_task_data(self, task_id: str):
        """Delete all data for a task."""
        self.results.delete_one({"task_id": task_id})
        self.trajectories.delete_many({"task_id": task_id})
        self.analysis.delete_many({"task_id": task_id})

# =============================================================================
# File Storage Manager
# =============================================================================

class FileStorageManager:
    """Manager for storing large result files."""
    
    def __init__(self, base_path: Path = RESULTS_STORAGE_PATH):
        self.base_path = base_path
    
    def _get_task_dir(self, task_id: str) -> Path:
        """Get storage directory for a task."""
        # Use hash-based subdirectory to avoid too many files in one directory
        hash_prefix = hashlib.md5(task_id.encode()).hexdigest()[:2]
        task_dir = self.base_path / hash_prefix / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir
    
    def store_file(self, task_id: str, filename: str, content: bytes,
                  file_type: str = "", metadata: Optional[Dict] = None) -> Dict:
        """Store a file."""
        task_dir = self._get_task_dir(task_id)
        file_path = task_dir / filename
        
        # Write file
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Calculate checksum
        checksum = hashlib.sha256(content).hexdigest()
        
        # Save metadata
        meta_path = task_dir / f"{filename}.meta"
        meta = {
            "filename": filename,
            "file_type": file_type,
            "file_size": len(content),
            "checksum": checksum,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat()
        }
        
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
        
        return {
            "storage_path": str(file_path.relative_to(self.base_path)),
            "file_size": len(content),
            "checksum": checksum
        }
    
    def get_file(self, task_id: str, filename: str) -> Optional[bytes]:
        """Get file content."""
        file_path = self._get_task_dir(task_id) / filename
        
        if not file_path.exists():
            return None
        
        with open(file_path, 'rb') as f:
            return f.read()
    
    def get_file_path(self, task_id: str, filename: str) -> Optional[Path]:
        """Get file path."""
        file_path = self._get_task_dir(task_id) / filename
        return file_path if file_path.exists() else None
    
    def delete_task_files(self, task_id: str) -> bool:
        """Delete all files for a task."""
        import shutil
        task_dir = self._get_task_dir(task_id)
        
        if task_dir.exists():
            shutil.rmtree(task_dir)
            return True
        return False
    
    def list_files(self, task_id: str) -> List[Dict]:
        """List all files for a task."""
        task_dir = self._get_task_dir(task_id)
        files = []
        
        for meta_file in task_dir.glob("*.meta"):
            with open(meta_file, 'r') as f:
                files.append(json.load(f))
        
        return files
    
    def verify_checksum(self, task_id: str, filename: str) -> bool:
        """Verify file integrity."""
        task_dir = self._get_task_dir(task_id)
        file_path = task_dir / filename
        meta_path = task_dir / f"{filename}.meta"
        
        if not file_path.exists() or not meta_path.exists():
            return False
        
        # Read stored checksum
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            stored_checksum = meta.get("checksum")
        
        # Calculate current checksum
        with open(file_path, 'rb') as f:
            current_checksum = hashlib.sha256(f.read()).hexdigest()
        
        return stored_checksum == current_checksum

# =============================================================================
# Unified Database Interface
# =============================================================================

class Database:
    """Unified database interface combining PostgreSQL, MongoDB, and file storage."""
    
    def __init__(self):
        self.postgres = DatabaseManager()
        self.files = FileStorageManager()
        
        # MongoDB is optional
        try:
            self.mongodb = DocumentStore()
            self.use_mongodb = True
        except Exception as e:
            logger.warning(f"MongoDB not available: {e}")
            self.mongodb = None
            self.use_mongodb = False
    
    # Task operations
    def create_task(self, task_data: Dict) -> Task:
        """Create a new task."""
        return self.postgres.create_task(task_data)
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self.postgres.get_task(task_id)
    
    def update_task(self, task_id: str, updates: Dict) -> bool:
        """Update task."""
        return self.postgres.update_task(task_id, updates)
    
    def delete_task(self, task_id: str) -> bool:
        """Delete task and all associated data."""
        # Delete from MongoDB
        if self.use_mongodb:
            self.mongodb.delete_task_data(task_id)
        
        # Delete files
        self.files.delete_task_files(task_id)
        
        # Delete from PostgreSQL
        return self.postgres.delete_task(task_id)
    
    def store_task_results(self, task_id: str, user_id: str, results: Dict):
        """Store task results in appropriate storage."""
        # Large results go to MongoDB or files
        if self.use_mongodb:
            self.mongodb.store_result(task_id, user_id, results)
        
        # Summary goes to PostgreSQL
        summary = self._create_summary(results)
        self.postgres.create_task_result({
            "task_id": task_id,
            "success": True,
            "summary": summary,
            "dft_energy": results.get("dft", {}).get("energy_per_atom"),
            "diffusion_coefficient": results.get("analysis", {}).get("activation_energy"),
        })
    
    def _create_summary(self, results: Dict) -> Dict:
        """Create a summary of results."""
        return {
            "has_dft": "dft" in results,
            "has_ml": "ml_models" in results,
            "has_md": "trajectories" in results,
            "has_analysis": "analysis" in results,
        }
    
    def get_task_results(self, task_id: str) -> Optional[Dict]:
        """Get task results."""
        # Try MongoDB first
        if self.use_mongodb:
            results = self.mongodb.get_result(task_id)
            if results:
                return results.get("results")
        
        # Fall back to PostgreSQL summary
        result_record = self.postgres.get_task_result(task_id)
        if result_record:
            return result_record.summary
        
        return None
    
    def store_file(self, task_id: str, filename: str, content: bytes,
                  file_type: str = "", metadata: Optional[Dict] = None) -> ResultFile:
        """Store a result file."""
        # Store file
        storage_info = self.files.store_file(task_id, filename, content, file_type, metadata)
        
        # Record in database
        return self.postgres.add_result_file({
            "task_id": task_id,
            "filename": filename,
            "file_type": file_type,
            "file_size": storage_info["file_size"],
            "checksum": storage_info["checksum"],
            "storage_path": storage_info["storage_path"]
        })
    
    def get_file(self, task_id: str, filename: str) -> Optional[bytes]:
        """Get file content."""
        return self.files.get_file(task_id, filename)
    
    def add_task_log(self, task_id: str, level: str, message: str, stage: str = ""):
        """Add task log entry."""
        return self.postgres.add_task_log(task_id, level, message, stage)
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        stats = self.postgres.get_statistics()
        
        if self.use_mongodb:
            stats["mongodb_documents"] = {
                "results": self.mongodb.results.count_documents({}),
                "trajectories": self.mongodb.trajectories.count_documents({}),
                "analysis": self.mongodb.analysis.count_documents({})
            }
        
        return stats

# Global database instance
db = Database()

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Test database connection
    print("Testing database connection...")
    
    # PostgreSQL
    try:
        stats = db.postgres.get_statistics()
        print(f"PostgreSQL connected. Stats: {stats}")
    except Exception as e:
        print(f"PostgreSQL connection failed: {e}")
    
    # MongoDB
    if db.use_mongodb:
        try:
            result = db.mongodb.store_result("test_task", "test_user", {"test": "data"})
            print(f"MongoDB connected. Test result: {result}")
        except Exception as e:
            print(f"MongoDB test failed: {e}")
    else:
        print("MongoDB not available")
    
    # File storage
    try:
        test_content = b"Test file content"
        file_info = db.files.store_file("test_task", "test.txt", test_content, "test")
        print(f"File storage test: {file_info}")
    except Exception as e:
        print(f"File storage test failed: {e}")
