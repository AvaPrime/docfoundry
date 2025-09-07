"""Configuration module for DocFoundry.

Provides configuration management for database, embeddings, and other components.
"""

from .database import (
    DatabaseConfig,
    DatabaseType,
    DatabaseFactory,
    db_factory,
    get_db_adapter,
    initialize_database,
    close_database
)

__all__ = [
    'DatabaseConfig',
    'DatabaseType', 
    'DatabaseFactory',
    'db_factory',
    'get_db_adapter',
    'initialize_database',
    'close_database'
]