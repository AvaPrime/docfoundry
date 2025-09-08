#!/usr/bin/env python3
"""Migration utility for adding lineage metadata to chunks table.

This script handles schema updates for both SQLite and PostgreSQL databases
to add lineage tracking capabilities to the chunks table.
"""

import sqlite3
import psycopg2
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class LineageMigration:
    """Handles lineage metadata migration for both SQLite and PostgreSQL."""
    
    def __init__(self, db_config: Dict[str, Any]):
        """Initialize migration with database configuration.
        
        Args:
            db_config: Database configuration with 'type', 'path' (SQLite) or connection params (PostgreSQL)
        """
        self.db_config = db_config
        self.db_type = db_config.get('type', 'sqlite').lower()
        
    def check_lineage_columns_exist(self) -> bool:
        """Check if lineage metadata columns already exist.
        
        Returns:
            True if columns exist, False otherwise
        """
        try:
            if self.db_type == 'sqlite':
                return self._check_sqlite_columns()
            elif self.db_type == 'postgresql':
                return self._check_postgres_columns()
            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")
        except Exception as e:
            logger.error(f"Error checking lineage columns: {e}")
            return False
    
    def _check_sqlite_columns(self) -> bool:
        """Check if lineage columns exist in SQLite chunks table."""
        conn = sqlite3.connect(self.db_config['path'])
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(chunks)")
            columns = [row[1] for row in cursor.fetchall()]
            
            lineage_columns = {
                'chunker_version', 'chunker_config', 'embedding_model',
                'embedding_version', 'embedding_dimensions', 'processing_timestamp', 'lineage_id'
            }
            
            return lineage_columns.issubset(set(columns))
        finally:
            conn.close()
    
    def _check_postgres_columns(self) -> bool:
        """Check if lineage columns exist in PostgreSQL chunks table."""
        conn = psycopg2.connect(**self.db_config['connection'])
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'chunks' AND table_schema = 'public'
            """)
            columns = [row[0] for row in cursor.fetchall()]
            
            lineage_columns = {
                'chunker_version', 'chunker_config', 'embedding_model',
                'embedding_version', 'embedding_dimensions', 'processing_timestamp', 'lineage_id'
            }
            
            return lineage_columns.issubset(set(columns))
        finally:
            conn.close()
    
    def run_migration(self) -> Dict[str, Any]:
        """Run the lineage metadata migration.
        
        Returns:
            Dictionary with migration results
        """
        try:
            # Check if migration is needed
            if self.check_lineage_columns_exist():
                logger.info("Lineage metadata columns already exist, skipping migration")
                return {
                    'success': True,
                    'message': 'Migration not needed - columns already exist',
                    'columns_added': 0,
                    'chunks_updated': 0
                }
            
            logger.info(f"Starting lineage metadata migration for {self.db_type} database")
            
            if self.db_type == 'sqlite':
                return self._migrate_sqlite()
            elif self.db_type == 'postgresql':
                return self._migrate_postgres()
            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")
                
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'columns_added': 0,
                'chunks_updated': 0
            }
    
    def _migrate_sqlite(self) -> Dict[str, Any]:
        """Run SQLite migration."""
        conn = sqlite3.connect(self.db_config['path'])
        try:
            cursor = conn.cursor()
            
            # Add lineage metadata columns (SQLite doesn't support CURRENT_TIMESTAMP as default)
            lineage_columns = [
                "ALTER TABLE chunks ADD COLUMN chunker_version TEXT DEFAULT 'v1.0.0'",
                "ALTER TABLE chunks ADD COLUMN chunker_config TEXT DEFAULT '{}'",
                "ALTER TABLE chunks ADD COLUMN embedding_model TEXT DEFAULT 'sentence-transformers/all-MiniLM-L6-v2'",
                "ALTER TABLE chunks ADD COLUMN embedding_version TEXT DEFAULT 'v1.0.0'",
                "ALTER TABLE chunks ADD COLUMN embedding_dimensions INTEGER DEFAULT 384",
                "ALTER TABLE chunks ADD COLUMN processing_timestamp TEXT",
                "ALTER TABLE chunks ADD COLUMN lineage_id TEXT"
            ]
            
            columns_added = 0
            for sql in lineage_columns:
                try:
                    cursor.execute(sql)
                    columns_added += 1
                    logger.debug(f"Added column: {sql}")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" in str(e).lower():
                        logger.debug(f"Column already exists, skipping: {sql}")
                    else:
                        raise
            
            # Create indexes
            index_queries = [
                "CREATE INDEX IF NOT EXISTS idx_chunks_lineage_id ON chunks(lineage_id)",
                "CREATE INDEX IF NOT EXISTS idx_chunks_chunker_version ON chunks(chunker_version)",
                "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_model ON chunks(embedding_model)",
                "CREATE INDEX IF NOT EXISTS idx_chunks_processing_timestamp ON chunks(processing_timestamp)"
            ]
            
            for sql in index_queries:
                cursor.execute(sql)
                logger.debug(f"Created index: {sql}")
            
            # Update existing chunks with default lineage metadata
            chunks_updated = self._populate_existing_chunks_sqlite(cursor)
            
            conn.commit()
            
            logger.info(f"SQLite migration completed: {columns_added} columns added, {chunks_updated} chunks updated")
            return {
                'success': True,
                'message': 'SQLite migration completed successfully',
                'columns_added': columns_added,
                'chunks_updated': chunks_updated
            }
            
        finally:
            conn.close()
    
    def _migrate_postgres(self) -> Dict[str, Any]:
        """Run PostgreSQL migration."""
        conn = psycopg2.connect(**self.db_config['connection'])
        try:
            cursor = conn.cursor()
            
            # Add lineage metadata columns
            lineage_columns = [
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunker_version TEXT DEFAULT 'v1.0.0'",
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunker_config JSONB DEFAULT '{}'",
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_model TEXT DEFAULT 'sentence-transformers/all-MiniLM-L6-v2'",
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_version TEXT DEFAULT 'v1.0.0'",
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_dimensions INTEGER DEFAULT 384",
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS processing_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS lineage_id TEXT"
            ]
            
            columns_added = len(lineage_columns)
            for sql in lineage_columns:
                cursor.execute(sql)
                logger.debug(f"Added column: {sql}")
            
            # Create indexes
            index_queries = [
                "CREATE INDEX IF NOT EXISTS idx_chunks_lineage_id ON chunks(lineage_id)",
                "CREATE INDEX IF NOT EXISTS idx_chunks_chunker_version ON chunks(chunker_version)",
                "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_model ON chunks(embedding_model)",
                "CREATE INDEX IF NOT EXISTS idx_chunks_processing_timestamp ON chunks(processing_timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_chunks_chunker_config ON chunks USING gin(chunker_config)"
            ]
            
            for sql in index_queries:
                cursor.execute(sql)
                logger.debug(f"Created index: {sql}")
            
            # Create lineage analysis view
            cursor.execute("""
                CREATE OR REPLACE VIEW chunk_lineage_analysis AS
                SELECT 
                    chunker_version,
                    embedding_model,
                    embedding_version,
                    embedding_dimensions,
                    COUNT(*) as chunk_count,
                    MIN(processing_timestamp) as first_processed,
                    MAX(processing_timestamp) as last_processed
                FROM chunks 
                WHERE chunker_version IS NOT NULL
                GROUP BY chunker_version, embedding_model, embedding_version, embedding_dimensions
                ORDER BY last_processed DESC
            """)
            
            # Create lineage ID generation function
            cursor.execute("""
                CREATE OR REPLACE FUNCTION generate_lineage_id(
                    p_chunker_version TEXT,
                    p_chunker_config JSONB,
                    p_embedding_model TEXT,
                    p_embedding_version TEXT,
                    p_embedding_dimensions INTEGER
                ) RETURNS TEXT AS $$
                BEGIN
                    RETURN encode(digest(
                        p_chunker_version || '|' || 
                        p_chunker_config::text || '|' || 
                        p_embedding_model || '|' || 
                        p_embedding_version || '|' || 
                        p_embedding_dimensions::text, 
                        'sha256'
                    ), 'hex');
                END;
                $$ LANGUAGE plpgsql IMMUTABLE;
            """)
            
            # Update existing chunks with default lineage metadata
            chunks_updated = self._populate_existing_chunks_postgres(cursor)
            
            conn.commit()
            
            logger.info(f"PostgreSQL migration completed: {columns_added} columns added, {chunks_updated} chunks updated")
            return {
                'success': True,
                'message': 'PostgreSQL migration completed successfully',
                'columns_added': columns_added,
                'chunks_updated': chunks_updated
            }
            
        finally:
            conn.close()
    
    def _populate_existing_chunks_sqlite(self, cursor) -> int:
        """Populate existing chunks with default lineage metadata in SQLite."""
        # Generate lineage ID for default configuration
        default_config = json.dumps({
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'min_chunk_size': 100,
            'max_tokens': 512,
            'overlap_tokens': 64,
            'chars_per_token': 4
        })
        
        import hashlib
        lineage_content = f"v1.0.0|{default_config}|sentence-transformers/all-MiniLM-L6-v2|v1.0.0|384"
        default_lineage_id = hashlib.sha256(lineage_content.encode()).hexdigest()
        
        # Get current timestamp for SQLite
        current_time = datetime.utcnow().isoformat()
        
        # Update chunks that don't have lineage metadata
        cursor.execute("""
            UPDATE chunks 
            SET 
                chunker_config = ?,
                lineage_id = ?,
                processing_timestamp = ?
            WHERE lineage_id IS NULL
        """, (default_config, default_lineage_id, current_time))
        
        return cursor.rowcount
    
    def _populate_existing_chunks_postgres(self, cursor) -> int:
        """Populate existing chunks with default lineage metadata in PostgreSQL."""
        # Update chunks that don't have lineage metadata
        cursor.execute("""
            UPDATE chunks 
            SET 
                chunker_config = '{
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                    "min_chunk_size": 100,
                    "max_tokens": 512,
                    "overlap_tokens": 64,
                    "chars_per_token": 4
                }'::jsonb,
                lineage_id = generate_lineage_id(
                    chunker_version,
                    '{
                        "chunk_size": 1000,
                        "chunk_overlap": 200,
                        "min_chunk_size": 100,
                        "max_tokens": 512,
                        "overlap_tokens": 64,
                        "chars_per_token": 4
                    }'::jsonb,
                    embedding_model,
                    embedding_version,
                    embedding_dimensions
                ),
                processing_timestamp = CURRENT_TIMESTAMP
            WHERE lineage_id IS NULL
        """)
        
        return cursor.rowcount


def run_lineage_migration(db_config: Dict[str, Any]) -> Dict[str, Any]:
    """Run lineage metadata migration.
    
    Args:
        db_config: Database configuration
    
    Returns:
        Migration results
    """
    migration = LineageMigration(db_config)
    return migration.run_migration()


if __name__ == "__main__":
    import argparse
    import sys
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Run lineage metadata migration')
    parser.add_argument('--db-type', choices=['sqlite', 'postgresql'], default='sqlite',
                       help='Database type')
    parser.add_argument('--db-path', help='SQLite database path')
    parser.add_argument('--db-host', help='PostgreSQL host')
    parser.add_argument('--db-port', type=int, default=5432, help='PostgreSQL port')
    parser.add_argument('--db-name', help='PostgreSQL database name')
    parser.add_argument('--db-user', help='PostgreSQL username')
    parser.add_argument('--db-password', help='PostgreSQL password')
    
    args = parser.parse_args()
    
    if args.db_type == 'sqlite':
        if not args.db_path:
            print("Error: --db-path is required for SQLite")
            sys.exit(1)
        
        db_config = {
            'type': 'sqlite',
            'path': args.db_path
        }
    else:
        if not all([args.db_host, args.db_name, args.db_user]):
            print("Error: --db-host, --db-name, and --db-user are required for PostgreSQL")
            sys.exit(1)
        
        db_config = {
            'type': 'postgresql',
            'connection': {
                'host': args.db_host,
                'port': args.db_port,
                'database': args.db_name,
                'user': args.db_user,
                'password': args.db_password or ''
            }
        }
    
    # Run migration
    result = run_lineage_migration(db_config)
    
    if result['success']:
        print(f"Migration completed successfully: {result['message']}")
        print(f"Columns added: {result['columns_added']}")
        print(f"Chunks updated: {result['chunks_updated']}")
    else:
        print(f"Migration failed: {result['error']}")
        sys.exit(1)