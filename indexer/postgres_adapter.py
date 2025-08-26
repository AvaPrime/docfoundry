"""PostgreSQL database adapter for DocFoundry.

Provides PostgreSQL-specific implementations for database operations,
replacing SQLite functionality with pgvector support.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import hashlib

import asyncpg
import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PostgresConfig(BaseModel):
    """PostgreSQL connection configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "docfoundry"
    user: str = "docfoundry"
    password: str = ""
    min_connections: int = 5
    max_connections: int = 20
    command_timeout: int = 60


class PostgresAdapter:
    """PostgreSQL database adapter with pgvector support."""
    
    def __init__(self, config: PostgresConfig):
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None
        
    async def initialize(self):
        """Initialize connection pool and ensure schema exists."""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                command_timeout=self.config.command_timeout
            )
            logger.info("PostgreSQL connection pool initialized")
            
            # Ensure pgvector extension is available
            async with self.pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                logger.info("pgvector extension ensured")
                
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            raise
    
    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")
    
    async def execute_schema(self, schema_path: str):
        """Execute schema SQL file."""
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        
        async with self.pool.acquire() as conn:
            await conn.execute(schema_sql)
        logger.info(f"Schema executed from {schema_path}")
    
    async def insert_document(self, path: str, title: str = None, 
                            source_url: str = None, content_type: str = 'text/markdown',
                            language: str = None, word_count: int = None) -> int:
        """Insert a new document and return its ID."""
        content_hash = hashlib.sha256(path.encode()).hexdigest()[:16]
        
        async with self.pool.acquire() as conn:
            doc_id = await conn.fetchval(
                """
                INSERT INTO documents (path, title, source_url, hash, content_type, language, word_count)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (path) DO UPDATE SET
                    title = EXCLUDED.title,
                    source_url = EXCLUDED.source_url,
                    hash = EXCLUDED.hash,
                    content_type = EXCLUDED.content_type,
                    language = EXCLUDED.language,
                    word_count = EXCLUDED.word_count,
                    updated_at = NOW()
                RETURNING id
                """,
                path, title, source_url, content_hash, content_type, language, word_count
            )
        return doc_id
    
    async def insert_chunk(self, document_id: int, chunk_id: str, text: str,
                          heading: str = None, anchor: str = None, 
                          h_path: List[str] = None, url: str = None,
                          token_len: int = None, lang: str = None,
                          embedding: np.ndarray = None) -> int:
        """Insert a new chunk with optional embedding."""
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        
        # Convert numpy array to list for pgvector
        embedding_list = embedding.tolist() if embedding is not None else None
        
        async with self.pool.acquire() as conn:
            chunk_db_id = await conn.fetchval(
                """
                INSERT INTO chunks (document_id, chunk_id, heading, anchor, text, 
                                  h_path, url, token_len, lang, hash, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    heading = EXCLUDED.heading,
                    anchor = EXCLUDED.anchor,
                    text = EXCLUDED.text,
                    h_path = EXCLUDED.h_path,
                    url = EXCLUDED.url,
                    token_len = EXCLUDED.token_len,
                    lang = EXCLUDED.lang,
                    hash = EXCLUDED.hash,
                    embedding = EXCLUDED.embedding,
                    updated_at = NOW()
                RETURNING id
                """,
                document_id, chunk_id, heading, anchor, text,
                h_path, url, token_len, lang, content_hash, embedding_list
            )
        return chunk_db_id
    
    async def search_hybrid(self, query: str, embedding: np.ndarray = None,
                           limit: int = 10, offset: int = 0,
                           filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Perform hybrid search combining full-text and vector similarity."""
        embedding_list = embedding.tolist() if embedding is not None else None
        
        # Build filter conditions
        filter_conditions = []
        filter_params = []
        param_count = 0
        
        if filters:
            if 'language' in filters:
                param_count += 1
                filter_conditions.append(f"c.lang = ${param_count}")
                filter_params.append(filters['language'])
            
            if 'source_url' in filters:
                param_count += 1
                filter_conditions.append(f"d.source_url LIKE ${param_count}")
                filter_params.append(f"%{filters['source_url']}%")
        
        where_clause = ""
        if filter_conditions:
            where_clause = "WHERE " + " AND ".join(filter_conditions)
        
        # Hybrid search query with RRF (Reciprocal Rank Fusion)
        search_sql = f"""
        WITH fts_results AS (
            SELECT c.id, c.chunk_id, c.text, c.heading, c.anchor, c.url,
                   d.path, d.title, d.source_url,
                   ts_rank(to_tsvector('english', c.text), plainto_tsquery('english', $1)) as fts_score,
                   ROW_NUMBER() OVER (ORDER BY ts_rank(to_tsvector('english', c.text), plainto_tsquery('english', $1)) DESC) as fts_rank
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            {where_clause}
            AND to_tsvector('english', c.text) @@ plainto_tsquery('english', $1)
        ),
        vector_results AS (
            SELECT c.id, c.chunk_id, c.text, c.heading, c.anchor, c.url,
                   d.path, d.title, d.source_url,
                   1 - (c.embedding <=> $2::vector) as similarity_score,
                   ROW_NUMBER() OVER (ORDER BY c.embedding <=> $2::vector) as vector_rank
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            {where_clause}
            AND c.embedding IS NOT NULL
        )
        SELECT COALESCE(f.id, v.id) as id,
               COALESCE(f.chunk_id, v.chunk_id) as chunk_id,
               COALESCE(f.text, v.text) as text,
               COALESCE(f.heading, v.heading) as heading,
               COALESCE(f.anchor, v.anchor) as anchor,
               COALESCE(f.url, v.url) as url,
               COALESCE(f.path, v.path) as path,
               COALESCE(f.title, v.title) as title,
               COALESCE(f.source_url, v.source_url) as source_url,
               COALESCE(f.fts_score, 0) as fts_score,
               COALESCE(v.similarity_score, 0) as similarity_score,
               -- RRF scoring: 1/(k + rank) where k=60
               COALESCE(1.0/(60 + f.fts_rank), 0) + COALESCE(1.0/(60 + v.vector_rank), 0) as rrf_score
        FROM fts_results f
        FULL OUTER JOIN vector_results v ON f.id = v.id
        ORDER BY rrf_score DESC
        LIMIT $3 OFFSET $4
        """
        
        params = [query, embedding_list] + filter_params + [limit, offset]
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(search_sql, *params)
        
        return [dict(row) for row in rows]
    
    async def search_semantic(self, embedding: np.ndarray, limit: int = 10,
                             threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Perform semantic search using vector similarity."""
        embedding_list = embedding.tolist()
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT c.id, c.chunk_id, c.text, c.heading, c.anchor, c.url,
                       d.path, d.title, d.source_url,
                       1 - (c.embedding <=> $1::vector) as similarity_score
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.embedding IS NOT NULL
                  AND 1 - (c.embedding <=> $1::vector) >= $2
                ORDER BY c.embedding <=> $1::vector
                LIMIT $3
                """,
                embedding_list, threshold, limit
            )
        
        return [dict(row) for row in rows]
    
    async def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by its ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT c.*, d.path, d.title, d.source_url
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.chunk_id = $1
                """,
                chunk_id
            )
        
        return dict(row) if row else None
    
    async def record_click_feedback(self, session_id: str, user_id: str,
                                   query: str, chunk_id: str, position: int,
                                   clicked: bool = True, dwell_time: float = None):
        """Record click feedback for learning-to-rank."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO click_feedback (session_id, user_id, query, chunk_id, 
                                          position, clicked, dwell_time)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                session_id, user_id, query, chunk_id, position, clicked, dwell_time
            )
    
    async def get_click_feedback_stats(self, query: str = None, 
                                      days: int = 30) -> Dict[str, Any]:
        """Get click feedback statistics for learning-to-rank."""
        conditions = ["timestamp >= NOW() - INTERVAL '%s days'" % days]
        params = []
        
        if query:
            conditions.append("query = $1")
            params.append(query)
        
        where_clause = "WHERE " + " AND ".join(conditions)
        
        async with self.pool.acquire() as conn:
            stats = await conn.fetchrow(
                f"""
                SELECT COUNT(*) as total_clicks,
                       COUNT(DISTINCT session_id) as unique_sessions,
                       AVG(position) as avg_click_position,
                       AVG(dwell_time) as avg_dwell_time
                FROM click_feedback
                {where_clause}
                """,
                *params
            )
        
        return dict(stats) if stats else {}
    
    async def cleanup_old_data(self, days: int = 90):
        """Clean up old data to maintain performance."""
        async with self.pool.acquire() as conn:
            # Clean old click feedback
            deleted_clicks = await conn.fetchval(
                "DELETE FROM click_feedback WHERE timestamp < NOW() - INTERVAL '%s days' RETURNING COUNT(*)",
                days
            )
            
            # Clean old search sessions
            deleted_sessions = await conn.fetchval(
                "DELETE FROM search_sessions WHERE created_at < NOW() - INTERVAL '%s days' RETURNING COUNT(*)",
                days
            )
            
            logger.info(f"Cleaned up {deleted_clicks} old click feedback records and {deleted_sessions} old search sessions")
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics for monitoring."""
        async with self.pool.acquire() as conn:
            stats = await conn.fetchrow(
                """
                SELECT 
                    (SELECT COUNT(*) FROM documents) as document_count,
                    (SELECT COUNT(*) FROM chunks) as chunk_count,
                    (SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL) as embedded_chunk_count,
                    (SELECT COUNT(*) FROM sources WHERE enabled = true) as active_source_count,
                    (SELECT COUNT(*) FROM jobs WHERE status = 'running') as running_jobs,
                    (SELECT COUNT(*) FROM click_feedback WHERE timestamp >= NOW() - INTERVAL '24 hours') as recent_clicks
                """
            )
        
        return dict(stats) if stats else {}