"""
DFT-LAMMPS Web Module
=====================
Web interface and API for DFT-LAMMPS platform.

Modules:
- api_server: FastAPI REST API
- task_queue: Celery task queue
- auth: Authentication and authorization
- database: Database interface

Author: DFT-LAMMPS Web Team
Version: 1.0.0
"""

__version__ = "1.0.0"

from .api_server import app, API_VERSION
from .auth import AuthService, UserRole, Permission
from .database import Database, DatabaseManager

__all__ = [
    'app',
    'API_VERSION',
    'AuthService',
    'UserRole',
    'Permission',
    'Database',
    'DatabaseManager',
]
