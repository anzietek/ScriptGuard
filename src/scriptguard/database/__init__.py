"""
ScriptGuard Database Module
Provides PostgreSQL database management, versioning, and deduplication.
"""

from .db_schema import (
    DatabasePool,
    create_database_schema,
    get_connection,
    return_connection,
    refresh_statistics
)
from .dataset_manager import DatasetManager
from .deduplication import deduplicate_samples, compute_hash

__all__ = [
    "DatabasePool",
    "create_database_schema",
    "get_connection",
    "return_connection",
    "refresh_statistics",
    "DatasetManager",
    "deduplicate_samples",
    "compute_hash",
]
