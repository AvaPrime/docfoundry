"""Document indexing pipeline for DocFoundry.

Provides functionality to index crawled documents into the vector database.
"""

import sqlite3
import hashlib
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path
from .differential_chunker import DifferentialChunker

logger = logging.getLogger(__name__)

def index_documents(documents: List[Dict[str, Any]], 
                   db_path: str,
                   embedding_manager = None) -> Dict[str, Any]:
    """Index documents into the database.
    
    Args:
        documents: List of document dictionaries with content and metadata
        db_path: Path to the SQLite database
        embedding_manager: Optional embedding manager for vector indexing
    
    Returns:
        Dictionary with indexing results
    """
    indexed_count = 0
    errors = []
    
    try:
        # Ensure database directory exists
        db_path_obj = Path(db_path)
        db_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                title TEXT,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                source_name TEXT,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                etag TEXT,
                last_modified TEXT,
                last_crawled TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                heading TEXT,
                anchor TEXT,
                text TEXT NOT NULL,
                embedding BLOB,
                chunk_index INTEGER,
                content_hash TEXT,
                token_count INTEGER,
                metadata TEXT,
                -- Lineage metadata columns
                chunker_version TEXT DEFAULT 'v1.0.0',
                chunker_config TEXT DEFAULT '{}',
                embedding_model TEXT DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
                embedding_version TEXT DEFAULT 'v1.0.0',
                embedding_dimensions INTEGER DEFAULT 384,
                processing_timestamp TEXT,
                lineage_id TEXT
            )
        """)
        
        # Create indexes for lineage metadata
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_lineage_id ON chunks(lineage_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chunker_version ON chunks(chunker_version)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_embedding_model ON chunks(embedding_model)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_processing_timestamp ON chunks(processing_timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)")
        
        # Create indexes for incremental crawling
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_last_crawled ON documents(last_crawled)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash)")
        
        conn.commit()
        
        for doc in documents:
            try:
                # Generate document ID from URL
                doc_id = hashlib.sha256(doc['url'].encode()).hexdigest()[:16]
                
                # Generate content hash
                content_hash = hashlib.sha256(doc['content'].encode()).hexdigest()
                
                # Check if document already exists with same content
                cursor.execute(
                    "SELECT content_hash FROM documents WHERE id = ?",
                    (doc_id,)
                )
                existing = cursor.fetchone()
                
                if existing and existing[0] == content_hash:
                    logger.debug(f"Document {doc['url']} already indexed with same content")
                    continue
                
                # Insert or update document
                cursor.execute("""
                    INSERT OR REPLACE INTO documents 
                    (id, url, title, content, content_hash, source_name, metadata, etag, last_modified, last_crawled)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    doc_id,
                    doc['url'],
                    doc.get('title', ''),
                    doc['content'],
                    content_hash,
                    doc.get('source_name', ''),
                    str(doc.get('metadata', {})),
                    doc.get('etag'),
                    doc.get('last_modified')
                ))
                
                indexed_count += 1
                logger.debug(f"Indexed document: {doc['url']}")
                
            except Exception as e:
                error_msg = f"Failed to index document {doc.get('url', 'unknown')}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Indexed {indexed_count} documents")
        
        return {
            "indexed_count": indexed_count,
            "total_documents": len(documents),
            "errors": errors,
            "success": len(errors) == 0
        }
        
    except Exception as e:
        error_msg = f"Failed to index documents: {str(e)}"
        logger.error(error_msg)
        return {
            "indexed_count": indexed_count,
            "total_documents": len(documents),
            "errors": errors + [error_msg],
            "success": False
        }

def get_document_by_id(doc_id: str, db_path: str) -> Optional[Dict[str, Any]]:
    """Get a document by ID from the database.
    
    Args:
        doc_id: Document ID
        db_path: Path to the SQLite database
    
    Returns:
        Document dictionary or None if not found
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM documents WHERE id = ?",
            (doc_id,)
        )
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "id": row[0],
                "url": row[1],
                "title": row[2],
                "content": row[3],
                "content_hash": row[4],
                "source_name": row[5],
                "indexed_at": row[6],
                "metadata": row[7],
                "etag": row[8],
                "last_modified": row[9],
                "last_crawled": row[10]
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to get document {doc_id}: {str(e)}")
        return None

def get_document_metadata_for_url(url: str, db_path: str) -> Optional[Dict[str, Any]]:
    """Get document metadata for incremental crawling
    
    Args:
        url: Document URL
        db_path: Path to the SQLite database
    
    Returns:
        Document metadata dictionary or None if not found
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        doc_id = hashlib.sha256(url.encode()).hexdigest()[:16]
        cursor.execute(
            "SELECT etag, last_modified, content_hash, last_crawled FROM documents WHERE id = ?",
            (doc_id,)
        )
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'etag': row[0],
                'last_modified': row[1],
                'content_hash': row[2],
                'last_crawled': row[3]
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to get document metadata for {url}: {str(e)}")
        return None

def should_recrawl_document(url: str, db_path: str, max_age_hours: int = 24) -> bool:
    """Determine if a document should be re-crawled based on age and metadata
    
    Args:
        url: Document URL
        db_path: Path to the SQLite database
        max_age_hours: Maximum age in hours before re-crawling
    
    Returns:
        True if document should be re-crawled, False otherwise
    """
    try:
        metadata = get_document_metadata_for_url(url, db_path)
        if not metadata:
            return True  # New document, should crawl
        
        # Check if document is older than max_age_hours
        if metadata['last_crawled']:
            from datetime import datetime, timedelta
            last_crawled = datetime.fromisoformat(metadata['last_crawled'])
            if datetime.now() - last_crawled > timedelta(hours=max_age_hours):
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"Failed to check recrawl status for {url}: {str(e)}")
        return True  # Default to crawling on error

def list_documents(db_path: str, source_name: str = None, limit: int = 100) -> List[Dict[str, Any]]:
    """List documents from the database.
    
    Args:
        db_path: Path to the SQLite database
        source_name: Optional source name filter
        limit: Maximum number of documents to return
    
    Returns:
        List of document dictionaries
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        if source_name:
            cursor.execute(
                "SELECT * FROM documents WHERE source_name = ? ORDER BY indexed_at DESC LIMIT ?",
                (source_name, limit)
            )
        else:
            cursor.execute(
                "SELECT * FROM documents ORDER BY indexed_at DESC LIMIT ?",
                (limit,)
            )
        
        rows = cursor.fetchall()
        conn.close()
        
        documents = []
        for row in rows:
            documents.append({
                "id": row[0],
                "url": row[1],
                "title": row[2],
                "content": row[3][:200] + "..." if len(row[3]) > 200 else row[3],  # Truncate content
                "content_hash": row[4],
                "source_name": row[5],
                "indexed_at": row[6],
                "metadata": row[7]
            })
        
        return documents
        
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        return []

def delete_documents_by_source(source_name: str, db_path: str) -> Dict[str, Any]:
    """Delete all documents from a specific source.
    
    Args:
        source_name: Name of the source
        db_path: Path to the SQLite database
    
    Returns:
        Dictionary with deletion results
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Count documents to be deleted
        cursor.execute(
            "SELECT COUNT(*) FROM documents WHERE source_name = ?",
            (source_name,)
        )
        count = cursor.fetchone()[0]
        
        # Delete document chunks first (foreign key constraint)
        cursor.execute("""
            DELETE FROM chunks 
            WHERE document_id IN (
                SELECT id FROM documents WHERE source_name = ?
            )
        """, (source_name,))
        
        # Delete documents
        cursor.execute(
            "DELETE FROM documents WHERE source_name = ?",
            (source_name,)
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Deleted {count} documents from source '{source_name}'")
        
        return {
            "deleted_count": count,
            "source_name": source_name,
            "success": True
        }
        
    except Exception as e:
        error_msg = f"Failed to delete documents from source '{source_name}': {str(e)}"
        logger.error(error_msg)
        return {
            "deleted_count": 0,
            "source_name": source_name,
            "error": error_msg,
            "success": False
        }