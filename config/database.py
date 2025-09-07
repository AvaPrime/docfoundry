"""Database configuration and factory for DocFoundry.

Provides unified interface for database operations with support for
both SQLite (development) and PostgreSQL (production) backends.
"""

import os
import logging
from typing import Union, Optional
from enum import Enum
from pydantic import BaseModel, Field

from indexer.postgres_adapter import PostgresAdapter, PostgresConfig

logger = logging.getLogger(__name__)


class DatabaseType(str, Enum):
    """Supported database types."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"


class DatabaseConfig(BaseModel):
    """Database configuration."""
    type: DatabaseType = Field(default=DatabaseType.SQLITE, description="Database type")
    
    # SQLite configuration
    sqlite_path: str = Field(default="docfoundry.db", description="SQLite database path")
    
    # PostgreSQL configuration
    postgres: PostgresConfig = Field(default_factory=PostgresConfig, description="PostgreSQL configuration")
    
    # Connection settings
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Maximum pool overflow")
    pool_timeout: int = Field(default=30, description="Pool timeout in seconds")
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create configuration from environment variables."""
        db_type = os.getenv('DOCFOUNDRY_DB_TYPE', 'sqlite').lower()
        
        if db_type == 'postgresql':
            postgres_config = PostgresConfig(
                host=os.getenv('POSTGRES_HOST', 'localhost'),
                port=int(os.getenv('POSTGRES_PORT', '5432')),
                database=os.getenv('POSTGRES_DB', 'docfoundry'),
                user=os.getenv('POSTGRES_USER', 'docfoundry'),
                password=os.getenv('POSTGRES_PASSWORD', ''),
                min_connections=int(os.getenv('POSTGRES_MIN_CONNECTIONS', '5')),
                max_connections=int(os.getenv('POSTGRES_MAX_CONNECTIONS', '20')),
                command_timeout=int(os.getenv('POSTGRES_COMMAND_TIMEOUT', '60'))
            )
            
            return cls(
                type=DatabaseType.POSTGRESQL,
                postgres=postgres_config,
                pool_size=int(os.getenv('DB_POOL_SIZE', '10')),
                max_overflow=int(os.getenv('DB_MAX_OVERFLOW', '20')),
                pool_timeout=int(os.getenv('DB_POOL_TIMEOUT', '30'))
            )
        else:
            return cls(
                type=DatabaseType.SQLITE,
                sqlite_path=os.getenv('SQLITE_PATH', 'docfoundry.db'),
                pool_size=int(os.getenv('DB_POOL_SIZE', '10')),
                max_overflow=int(os.getenv('DB_MAX_OVERFLOW', '20')),
                pool_timeout=int(os.getenv('DB_POOL_TIMEOUT', '30'))
            )


class DatabaseFactory:
    """Factory for creating database adapters."""
    
    _instance: Optional['DatabaseFactory'] = None
    _adapter: Optional[Union[PostgresAdapter, 'SQLiteAdapter']] = None
    _config: Optional[DatabaseConfig] = None
    
    def __new__(cls) -> 'DatabaseFactory':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def initialize(self, config: Optional[DatabaseConfig] = None):
        """Initialize database adapter based on configuration."""
        if config is None:
            config = DatabaseConfig.from_env()
        
        self._config = config
        
        if config.type == DatabaseType.POSTGRESQL:
            logger.info("Initializing PostgreSQL adapter")
            self._adapter = PostgresAdapter(config.postgres)
            await self._adapter.initialize()
        else:
            logger.info("Initializing SQLite adapter")
            # Import here to avoid circular imports
            from indexer.sqlite_adapter import SQLiteAdapter
            self._adapter = SQLiteAdapter(config.sqlite_path)
            await self._adapter.initialize()
        
        logger.info(f"Database adapter initialized: {config.type}")
    
    async def close(self):
        """Close database connections."""
        if self._adapter:
            await self._adapter.close()
            self._adapter = None
            logger.info("Database adapter closed")
    
    def get_adapter(self) -> Union[PostgresAdapter, 'SQLiteAdapter']:
        """Get the current database adapter."""
        if self._adapter is None:
            raise RuntimeError("Database adapter not initialized. Call initialize() first.")
        return self._adapter
    
    def get_config(self) -> DatabaseConfig:
        """Get the current database configuration."""
        if self._config is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._config
    
    def is_postgresql(self) -> bool:
        """Check if using PostgreSQL backend."""
        return self._config and self._config.type == DatabaseType.POSTGRESQL
    
    def is_sqlite(self) -> bool:
        """Check if using SQLite backend."""
        return self._config and self._config.type == DatabaseType.SQLITE


# Global database factory instance
db_factory = DatabaseFactory()


async def get_db_adapter() -> Union[PostgresAdapter, 'SQLiteAdapter']:
    """Get database adapter instance."""
    return db_factory.get_adapter()


async def initialize_database(config: Optional[DatabaseConfig] = None):
    """Initialize database with configuration."""
    await db_factory.initialize(config)


async def close_database():
    """Close database connections."""
    await db_factory.close()