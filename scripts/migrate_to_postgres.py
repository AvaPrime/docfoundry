#!/usr/bin/env python3
"""Migration script from SQLite to PostgreSQL with pgvector.

This script migrates existing DocFoundry data from SQLite to PostgreSQL,
preserving all documents, chunks, and embeddings while upgrading to pgvector.
"""

import asyncio
import sqlite3
import logging
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from indexer.postgres_adapter import PostgresAdapter, PostgresConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SQLiteToPostgresMigrator:
    """Handles migration from SQLite to PostgreSQL."""
    
    def __init__(self, sqlite_path: str, postgres_config: PostgresConfig):
        self.sqlite_path = sqlite_path
        self.postgres_config = postgres_config
        self.postgres_adapter = PostgresAdapter(postgres_config)
        
    async def migrate(self, batch_size: int = 1000, verify: bool = True):
        """Perform the complete migration."""
        logger.info("Starting migration from SQLite to PostgreSQL")
        
        try:
            # Initialize PostgreSQL connection
            await self.postgres_adapter.initialize()
            
            # Execute schema
            schema_path = Path(__file__).parent.parent / "indexer" / "postgres_schema.sql"
            await self.postgres_adapter.execute_schema(str(schema_path))
            logger.info("PostgreSQL schema initialized")
            
            # Connect to SQLite
            sqlite_conn = sqlite3.connect(self.sqlite_path)
            sqlite_conn.row_factory = sqlite3.Row
            
            # Migrate data
            await self._migrate_documents(sqlite_conn, batch_size)
            await self._migrate_chunks(sqlite_conn, batch_size)
            
            # Verify migration if requested
            if verify:
                await self._verify_migration(sqlite_conn)
            
            sqlite_conn.close()
            logger.info("Migration completed successfully")
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
        finally:
            await self.postgres_adapter.close()
    
    async def _migrate_documents(self, sqlite_conn: sqlite3.Connection, batch_size: int):
        """Migrate documents table."""
        logger.info("Migrating documents...")
        
        cursor = sqlite_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        total_docs = cursor.fetchone()[0]
        logger.info(f"Found {total_docs} documents to migrate")
        
        if total_docs == 0:
            logger.info("No documents to migrate")
            return
        
        # Get all documents
        cursor.execute("""
            SELECT id, path, title, source_url, captured_at, hash, 
                   content_type, language, word_count
            FROM documents
            ORDER BY id
        """)
        
        migrated_count = 0
        doc_id_mapping = {}  # SQLite ID -> PostgreSQL ID mapping
        
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            
            for row in rows:
                try:
                    # Convert captured_at if it exists
                    captured_at = None
                    if row['captured_at']:
                        try:
                            captured_at = datetime.fromisoformat(row['captured_at'].replace('Z', '+00:00'))
                        except:
                            captured_at = None
                    
                    # Insert document into PostgreSQL
                    pg_doc_id = await self.postgres_adapter.insert_document(
                        path=row['path'],
                        title=row['title'],
                        source_url=row['source_url'],
                        content_type=row['content_type'] or 'text/markdown',
                        language=row['language'],
                        word_count=row['word_count']
                    )
                    
                    # Store mapping for chunks migration
                    doc_id_mapping[row['id']] = pg_doc_id
                    migrated_count += 1
                    
                    if migrated_count % 100 == 0:
                        logger.info(f"Migrated {migrated_count}/{total_docs} documents")
                        
                except Exception as e:
                    logger.error(f"Failed to migrate document {row['path']}: {e}")
                    continue
        
        logger.info(f"Successfully migrated {migrated_count} documents")
        self.doc_id_mapping = doc_id_mapping
    
    async def _migrate_chunks(self, sqlite_conn: sqlite3.Connection, batch_size: int):
        """Migrate chunks table with embeddings."""
        logger.info("Migrating chunks...")
        
        cursor = sqlite_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks")
        total_chunks = cursor.fetchone()[0]
        logger.info(f"Found {total_chunks} chunks to migrate")
        
        if total_chunks == 0:
            logger.info("No chunks to migrate")
            return
        
        # Get all chunks
        cursor.execute("""
            SELECT document_id, chunk_id, heading, anchor, text, url, 
                   retrieved_at, token_len, lang, hash, embedding
            FROM chunks
            ORDER BY document_id, chunk_id
        """)
        
        migrated_count = 0
        skipped_count = 0
        
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            
            for row in rows:
                try:
                    # Get PostgreSQL document ID
                    sqlite_doc_id = row['document_id']
                    if sqlite_doc_id not in self.doc_id_mapping:
                        logger.warning(f"Document ID {sqlite_doc_id} not found in mapping, skipping chunk {row['chunk_id']}")
                        skipped_count += 1
                        continue
                    
                    pg_doc_id = self.doc_id_mapping[sqlite_doc_id]
                    
                    # Convert embedding from BLOB to numpy array
                    embedding = None
                    if row['embedding']:
                        try:
                            embedding = np.frombuffer(row['embedding'], dtype=np.float32)
                            # Validate embedding dimension
                            if len(embedding) != 1536:  # OpenAI embedding dimension
                                logger.warning(f"Unexpected embedding dimension {len(embedding)} for chunk {row['chunk_id']}")
                                embedding = None
                        except Exception as e:
                            logger.warning(f"Failed to convert embedding for chunk {row['chunk_id']}: {e}")
                            embedding = None
                    
                    # Parse heading path if available
                    h_path = None
                    if row['heading']:
                        # Simple heuristic: split by common separators
                        h_path = [h.strip() for h in row['heading'].split(' > ') if h.strip()]
                        if not h_path:
                            h_path = [row['heading']]
                    
                    # Insert chunk into PostgreSQL
                    await self.postgres_adapter.insert_chunk(
                        document_id=pg_doc_id,
                        chunk_id=row['chunk_id'],
                        text=row['text'],
                        heading=row['heading'],
                        anchor=row['anchor'],
                        h_path=h_path,
                        url=row['url'],
                        token_len=row['token_len'],
                        lang=row['lang'],
                        embedding=embedding
                    )
                    
                    migrated_count += 1
                    
                    if migrated_count % 500 == 0:
                        logger.info(f"Migrated {migrated_count}/{total_chunks} chunks")
                        
                except Exception as e:
                    logger.error(f"Failed to migrate chunk {row['chunk_id']}: {e}")
                    skipped_count += 1
                    continue
        
        logger.info(f"Successfully migrated {migrated_count} chunks, skipped {skipped_count}")
    
    async def _verify_migration(self, sqlite_conn: sqlite3.Connection):
        """Verify migration integrity."""
        logger.info("Verifying migration...")
        
        # Get SQLite counts
        cursor = sqlite_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        sqlite_doc_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM chunks")
        sqlite_chunk_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
        sqlite_embedded_count = cursor.fetchone()[0]
        
        # Get PostgreSQL counts
        pg_stats = await self.postgres_adapter.get_database_stats()
        
        # Compare counts
        doc_match = sqlite_doc_count == pg_stats['document_count']
        chunk_match = sqlite_chunk_count == pg_stats['chunk_count']
        embedding_match = sqlite_embedded_count == pg_stats['embedded_chunk_count']
        
        logger.info(f"Documents: SQLite={sqlite_doc_count}, PostgreSQL={pg_stats['document_count']}, Match={doc_match}")
        logger.info(f"Chunks: SQLite={sqlite_chunk_count}, PostgreSQL={pg_stats['chunk_count']}, Match={chunk_match}")
        logger.info(f"Embeddings: SQLite={sqlite_embedded_count}, PostgreSQL={pg_stats['embedded_chunk_count']}, Match={embedding_match}")
        
        if doc_match and chunk_match and embedding_match:
            logger.info("✅ Migration verification passed")
        else:
            logger.warning("⚠️ Migration verification found discrepancies")
    
    async def create_backup(self, backup_path: str):
        """Create a backup of the SQLite database before migration."""
        import shutil
        
        if os.path.exists(self.sqlite_path):
            shutil.copy2(self.sqlite_path, backup_path)
            logger.info(f"SQLite backup created at {backup_path}")
        else:
            logger.warning(f"SQLite database not found at {self.sqlite_path}")


async def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(description="Migrate DocFoundry from SQLite to PostgreSQL")
    parser.add_argument("--sqlite-path", required=True, help="Path to SQLite database")
    parser.add_argument("--pg-host", default="localhost", help="PostgreSQL host")
    parser.add_argument("--pg-port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--pg-database", default="docfoundry", help="PostgreSQL database name")
    parser.add_argument("--pg-user", default="docfoundry", help="PostgreSQL user")
    parser.add_argument("--pg-password", default="", help="PostgreSQL password")
    parser.add_argument("--batch-size", type=int, default=1000, help="Migration batch size")
    parser.add_argument("--no-verify", action="store_true", help="Skip migration verification")
    parser.add_argument("--backup", help="Create SQLite backup before migration")
    
    args = parser.parse_args()
    
    # Create PostgreSQL config
    pg_config = PostgresConfig(
        host=args.pg_host,
        port=args.pg_port,
        database=args.pg_database,
        user=args.pg_user,
        password=args.pg_password
    )
    
    # Create migrator
    migrator = SQLiteToPostgresMigrator(args.sqlite_path, pg_config)
    
    try:
        # Create backup if requested
        if args.backup:
            await migrator.create_backup(args.backup)
        
        # Perform migration
        await migrator.migrate(
            batch_size=args.batch_size,
            verify=not args.no_verify
        )
        
        logger.info("Migration completed successfully!")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())