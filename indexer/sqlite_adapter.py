"""SQLite database adapter for DocFoundry.

Provides SQLite-specific implementations for database operations,
maintaining backward compatibility with existing functionality.
"""

import sqlite3
import logging
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SQLiteAdapter:
    """SQLite database adapter with unified interface."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        
    async def initialize(self):
        """Initialize SQLite connection and ensure schema exists."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys = ON")
            
            # Ensure schema exists
            schema_path = Path(__file__).parent / "schema.sql"
            if schema_path.exists():
                with open(schema_path, 'r', encoding='utf-8') as f:
                    schema_sql = f.read()
                self.conn.executescript(schema_sql)
                self.conn.commit()
            
            logger.info(f"SQLite adapter initialized: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SQLite: {e}")
            raise
    
    async def close(self):
        """Close SQLite connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("SQLite connection closed")
    
    async def execute_schema(self, schema_path: str):
        """Execute schema SQL file."""
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        
        self.conn.executescript(schema_sql)
        self.conn.commit()
        logger.info(f"Schema executed from {schema_path}")
    
    async def insert_document(self, path: str, title: str = None, 
                            source_url: str = None, content_type: str = 'text/markdown',
                            language: str = None, word_count: int = None) -> int:
        """Insert a new document and return its ID."""
        content_hash = hashlib.sha256(path.encode()).hexdigest()[:16]
        
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO documents (path, title, source_url, hash, content_type, language, word_count, captured_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (path, title, source_url, content_hash, content_type, language, word_count, datetime.now().isoformat())
        )
        
        doc_id = cursor.lastrowid
        self.conn.commit()
        return doc_id
    
    async def insert_chunk(self, document_id: int, chunk_id: str, text: str,
                          heading: str = None, anchor: str = None, 
                          h_path: List[str] = None, url: str = None,
                          token_len: int = None, lang: str = None,
                          embedding: np.ndarray = None) -> int:
        """Insert a new chunk with optional embedding."""
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        
        # Convert numpy array to bytes for SQLite BLOB
        embedding_blob = embedding.tobytes() if embedding is not None else None
        
        # Convert h_path to JSON string for SQLite
        h_path_json = json.dumps(h_path) if h_path else None
        
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO chunks (document_id, chunk_id, heading, anchor, text, 
                                         url, token_len, lang, hash, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (document_id, chunk_id, heading, anchor, text, url, token_len, lang, content_hash, embedding_blob)
        )
        
        chunk_db_id = cursor.lastrowid
        self.conn.commit()
        return chunk_db_id
    
    async def search_hybrid(self, query: str, embedding: np.ndarray = None,
                           limit: int = 10, offset: int = 0,
                           filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Perform hybrid search combining full-text and vector similarity."""
        # Build filter conditions
        filter_conditions = []
        filter_params = []
        
        if filters:
            if 'language' in filters:
                filter_conditions.append("c.lang = ?")
                filter_params.append(filters['language'])
            
            if 'source_url' in filters:
                filter_conditions.append("d.source_url LIKE ?")
                filter_params.append(f"%{filters['source_url']}%")
        
        where_clause = ""
        if filter_conditions:
            where_clause = "WHERE " + " AND ".join(filter_conditions)
        
        # For SQLite, we'll do a simpler hybrid approach
        # First get FTS results
        fts_query = f"""
        SELECT c.id, c.chunk_id, c.text, c.heading, c.anchor, c.url,
               d.path, d.title, d.source_url, c.embedding,
               chunks_fts.rank as fts_score
        FROM chunks_fts
        JOIN chunks c ON chunks_fts.rowid = c.rowid
        JOIN documents d ON c.document_id = d.id
        {where_clause}
        {'AND' if where_clause else 'WHERE'} chunks_fts MATCH ?
        ORDER BY chunks_fts.rank
        LIMIT ? OFFSET ?
        """
        
        params = filter_params + [query, limit, offset]
        cursor = self.conn.cursor()
        cursor.execute(fts_query, params)
        
        results = []
        for row in cursor.fetchall():
            result = dict(row)
            
            # Calculate similarity score if embedding is available
            similarity_score = 0.0
            if embedding is not None and row['embedding']:
                try:
                    chunk_embedding = np.frombuffer(row['embedding'], dtype=np.float32)
                    # Cosine similarity
                    similarity_score = float(np.dot(embedding, chunk_embedding) / 
                                           (np.linalg.norm(embedding) * np.linalg.norm(chunk_embedding)))
                except Exception as e:
                    logger.warning(f"Failed to calculate similarity for chunk {row['chunk_id']}: {e}")
            
            result['similarity_score'] = similarity_score
            result['fts_score'] = row['fts_score']
            # Simple RRF-like scoring
            result['rrf_score'] = (result['fts_score'] + similarity_score) / 2
            
            # Remove embedding from result
            result.pop('embedding', None)
            results.append(result)
        
        # Sort by combined score
        results.sort(key=lambda x: x['rrf_score'], reverse=True)
        return results
    
    async def search_semantic(self, embedding: np.ndarray, limit: int = 10,
                             threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Perform semantic search using vector similarity."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT c.id, c.chunk_id, c.text, c.heading, c.anchor, c.url,
                   d.path, d.title, d.source_url, c.embedding
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.embedding IS NOT NULL
            """
        )
        
        results = []
        for row in cursor.fetchall():
            try:
                chunk_embedding = np.frombuffer(row['embedding'], dtype=np.float32)
                # Cosine similarity
                similarity_score = float(np.dot(embedding, chunk_embedding) / 
                                       (np.linalg.norm(embedding) * np.linalg.norm(chunk_embedding)))
                
                if similarity_score >= threshold:
                    result = dict(row)
                    result['similarity_score'] = similarity_score
                    result.pop('embedding', None)  # Remove embedding from result
                    results.append(result)
                    
            except Exception as e:
                logger.warning(f"Failed to calculate similarity for chunk {row['chunk_id']}: {e}")
                continue
        
        # Sort by similarity score and limit
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:limit]
    
    async def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by its ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT c.*, d.path, d.title, d.source_url
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.chunk_id = ?
            """,
            (chunk_id,)
        )
        
        row = cursor.fetchone()
        if row:
            result = dict(row)
            result.pop('embedding', None)  # Remove embedding from result
            return result
        return None
    
    async def record_click_feedback(self, session_id: str, user_id: str,
                                   query: str, chunk_id: str, position: int,
                                   clicked: bool = True, dwell_time: float = None):
        """Record click feedback for learning-to-rank."""
        # For SQLite, we'll create a simple table if it doesn't exist
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS click_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_id TEXT,
                query TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                clicked BOOLEAN DEFAULT 1,
                dwell_time REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        
        self.conn.execute(
            """
            INSERT INTO click_feedback (session_id, user_id, query, chunk_id, 
                                      position, clicked, dwell_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, user_id, query, chunk_id, position, clicked, dwell_time)
        )
        self.conn.commit()
    
    async def get_click_feedback_stats(self, query: str = None, 
                                      days: int = 30) -> Dict[str, Any]:
        """Get click feedback statistics for learning-to-rank."""
        conditions = ["timestamp >= datetime('now', '-{} days')".format(days)]
        params = []
        
        if query:
            conditions.append("query = ?")
            params.append(query)
        
        where_clause = "WHERE " + " AND ".join(conditions)
        
        cursor = self.conn.cursor()
        cursor.execute(
            f"""
            SELECT COUNT(*) as total_clicks,
                   COUNT(DISTINCT session_id) as unique_sessions,
                   AVG(position) as avg_click_position,
                   AVG(dwell_time) as avg_dwell_time
            FROM click_feedback
            {where_clause}
            """,
            params
        )
        
        row = cursor.fetchone()
        return dict(row) if row else {}
    
    async def cleanup_old_data(self, days: int = 90):
        """Clean up old data to maintain performance."""
        cursor = self.conn.cursor()
        
        # Clean old click feedback if table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='click_feedback'"
        )
        if cursor.fetchone():
            cursor.execute(
                "DELETE FROM click_feedback WHERE timestamp < datetime('now', '-{} days')".format(days)
            )
            deleted_clicks = cursor.rowcount
            logger.info(f"Cleaned up {deleted_clicks} old click feedback records")
        
        self.conn.commit()
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics for monitoring."""
        cursor = self.conn.cursor()
        
        # Get basic counts
        cursor.execute("SELECT COUNT(*) FROM documents")
        document_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
        embedded_chunk_count = cursor.fetchone()[0]
        
        # Check for click feedback table
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='click_feedback'"
        )
        recent_clicks = 0
        if cursor.fetchone():
            cursor.execute(
                "SELECT COUNT(*) FROM click_feedback WHERE timestamp >= datetime('now', '-1 day')"
            )
            recent_clicks = cursor.fetchone()[0]
        
        return {
            'document_count': document_count,
            'chunk_count': chunk_count,
            'embedded_chunk_count': embedded_chunk_count,
            'active_source_count': 0,  # Not implemented in SQLite version
            'running_jobs': 0,  # Not implemented in SQLite version
            'recent_clicks': recent_clicks
        }