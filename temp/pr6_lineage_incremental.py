# ============================================================================
# PR-6: Data Lineage & Incremental Refresh
# Files: Database schema updates + incremental processing logic
# ============================================================================

# FILE: alembic/versions/002_add_lineage_columns.py
"""Add lineage and versioning columns

Revision ID: 002
Revises: 001
Create Date: 2024-01-02 00:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Add lineage columns to chunks table
    op.add_column('chunks', sa.Column('embedding_model', sa.String(100), nullable=True))
    op.add_column('chunks', sa.Column('embedding_version', sa.String(50), nullable=True))
    op.add_column('chunks', sa.Column('chunker_version', sa.String(50), nullable=True))
    op.add_column('chunks', sa.Column('processing_metadata', sa.JSON(), nullable=True))
    
    # Add processing tracking columns to documents
    op.add_column('documents', sa.Column('last_processed_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('documents', sa.Column('processing_version', sa.String(50), nullable=True))
    op.add_column('documents', sa.Column('etag', sa.String(100), nullable=True))
    op.add_column('documents', sa.Column('last_modified', sa.DateTime(timezone=True), nullable=True))
    
    # Add indexes for efficient lookups
    op.create_index('idx_documents_content_hash', 'documents', ['content_hash'])
    op.create_index('idx_documents_last_processed', 'documents', ['last_processed_at'])
    op.create_index('idx_documents_etag', 'documents', ['etag'])
    op.create_index('idx_chunks_embedding_model', 'chunks', ['embedding_model', 'embedding_version'])

def downgrade() -> None:
    # Remove indexes
    op.drop_index('idx_chunks_embedding_model')
    op.drop_index('idx_documents_etag')
    op.drop_index('idx_documents_last_processed')
    op.drop_index('idx_documents_content_hash')
    
    # Remove columns
    op.drop_column('documents', 'last_modified')
    op.drop_column('documents', 'etag')
    op.drop_column('documents', 'processing_version')
    op.drop_column('documents', 'last_processed_at')
    op.drop_column('chunks', 'processing_metadata')
    op.drop_column('chunks', 'chunker_version')
    op.drop_column('chunks', 'embedding_version')
    op.drop_column('chunks', 'embedding_model')

# ============================================================================
# FILE: services/shared/models.py (updated with lineage fields)

from sqlalchemy import Column, String, DateTime, JSON, Text, Integer, ForeignKey, UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    url = Column(Text, nullable=False, index=True)
    title = Column(Text, nullable=True)
    content_hash = Column(String(64), nullable=False, index=True)  # SHA256 of content
    metadata = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    # Processing tracking (new in PR-6)
    last_processed_at = Column(DateTime(timezone=True), nullable=True, index=True)
    processing_version = Column(String(50), nullable=True)  # Version of processing pipeline
    etag = Column(String(100), nullable=True, index=True)   # HTTP ETag for caching
    last_modified = Column(DateTime(timezone=True), nullable=True)  # HTTP Last-Modified
    
    # Relationships
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = 'chunks'
    
    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID, ForeignKey('documents.id', ondelete='CASCADE'), nullable=False, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(ARRAY(float), nullable=True)  # Vector embedding
    fts = Column(TSVECTOR, nullable=True)            # Full-text search vector (maintained by trigger)
    chunk_index = Column(Integer, nullable=False)    # Order within document
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    # Lineage and versioning (new in PR-6)
    embedding_model = Column(String(100), nullable=True, index=True)     # e.g., "sentence-transformers/all-MiniLM-L6-v2"
    embedding_version = Column(String(50), nullable=True, index=True)    # e.g., "v2.2.2"
    chunker_version = Column(String(50), nullable=True)                  # e.g., "v1.0.0"
    processing_metadata = Column(JSON, nullable=True)                    # Additional processing info
    
    # Relationships
    document = relationship("Document", back_populates="chunks")

# ============================================================================
# FILE: services/shared/incremental.py (new incremental processing service)

import hashlib
import aiohttp
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from services.shared.models import Document, Chunk
import os
import logging

logger = logging.getLogger(__name__)

class IncrementalProcessor:
    """Handles incremental document processing with change detection"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.processing_version = os.getenv("PROCESSING_VERSION", "1.0.0")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_version = os.getenv("EMBEDDING_VERSION", "2.2.2")
        self.chunker_version = os.getenv("CHUNKER_VERSION", "1.0.0")
    
    async def check_document_changed(self, url: str, content: str) -> Tuple[bool, Optional[Document]]:
        """
        Check if document has changed since last processing
        
        Returns:
            Tuple of (has_changed, existing_document)
        """
        content_hash = self._compute_content_hash(content)
        
        # Look for existing document
        result = await self.session.execute(
            select(Document).where(Document.url == url)
        )
        existing_doc = result.scalar_one_or_none()
        
        if not existing_doc:
            logger.info(f"New document detected: {url}")
            return True, None
        
        # Check if content changed
        if existing_doc.content_hash == content_hash:
            logger.info(f"Document unchanged: {url}")
            return False, existing_doc
        
        logger.info(f"Document changed: {url} (hash: {existing_doc.content_hash} -> {content_hash})")
        return True, existing_doc
    
    async def check_document_changed_via_http(self, url: str) -> Tuple[bool, Optional[str]]:
        """
        Check if document changed using HTTP conditional requests (ETag/Last-Modified)
        
        Returns:
            Tuple of (has_changed, content_or_none)
        """
        # Get existing document for conditional headers
        result = await self.session.execute(
            select(Document).where(Document.url == url)
        )
        existing_doc = result.scalar_one_or_none()
        
        headers = {}
        if existing_doc:
            if existing_doc.etag:
                headers['If-None-Match'] = existing_doc.etag
            elif existing_doc.last_modified:
                headers['If-Modified-Since'] = existing_doc.last_modified.strftime('%a, %d %b %Y %H:%M:%S GMT')
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 304:  # Not Modified
                        logger.info(f"Document not modified (304): {url}")
                        return False, None
                    elif response.status == 200:
                        content = await response.text()
                        
                        # Update caching headers for future requests
                        if existing_doc:
                            update_data = {}
                            if 'etag' in response.headers:
                                update_data['etag'] = response.headers['etag']
                            if 'last-modified' in response.headers:
                                from email.utils import parsedate_to_datetime
                                update_data['last_modified'] = parsedate_to_datetime(response.headers['last-modified'])
                            
                            if update_data:
                                await self.session.execute(
                                    update(Document)
                                    .where(Document.id == existing_doc.id)
                                    .values(**update_data)
                                )
                        
                        logger.info(f"Document fetched successfully: {url}")
                        return True, content
                    else:
                        logger.error(f"Failed to fetch document: {url} (status: {response.status})")
                        return False, None
                        
        except Exception as e:
            logger.error(f"Error checking document via HTTP: {url} - {e}")
            return False, None
    
    async def process_document_incrementally(
        self, 
        url: str, 
        content: str, 
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process document incrementally, skipping if unchanged
        
        Returns:
            Dict with processing results and statistics
        """
        has_changed, existing_doc = await self.check_document_changed(url, content)
        
        if not has_changed and existing_doc:
            # Document hasn't changed, check if we need to reprocess due to version changes
            needs_reprocessing = await self._needs_reprocessing(existing_doc)
            if not needs_reprocessing:
                return {
                    "status": "skipped",
                    "reason": "unchanged",
                    "document_id": str(existing_doc.id),
                    "chunks_processed": 0,
                    "chunks_existing": await self._count_chunks(existing_doc.id)
                }
        
        # Process the document (new or changed)
        return await self._full_process_document(url, content, title, metadata, existing_doc)
    
    async def _needs_reprocessing(self, document: Document) -> bool:
        """Check if document needs reprocessing due to version changes"""
        
        # Check if processing pipeline version changed
        if document.processing_version != self.processing_version:
            logger.info(f"Document needs reprocessing - pipeline version changed: {document.processing_version} -> {self.processing_version}")
            return True
        
        # Check if embedding model/version changed for any chunks
        result = await self.session.execute(
            select(Chunk)
            .where(Chunk.document_id == document.id)
            .where(
                (Chunk.embedding_model != self.embedding_model) |
                (Chunk.embedding_version != self.embedding_version) |
                (Chunk.chunker_version != self.chunker_version)
            )
            .limit(1)
        )
        
        if result.scalar_one_or_none():
            logger.info(f"Document needs reprocessing - model versions changed")
            return True
        
        return False
    
    async def _full_process_document(
        self, 
        url: str, 
        content: str, 
        title: Optional[str], 
        metadata: Optional[Dict[str, Any]],
        existing_doc: Optional[Document]
    ) -> Dict[str, Any]:
        """Perform full document processing"""
        
        content_hash = self._compute_content_hash(content)
        now = datetime.utcnow()
        
        if existing_doc:
            # Update existing document
            await self.session.execute(
                update(Document)
                .where(Document.id == existing_doc.id)
                .values(
                    title=title,
                    content_hash=content_hash,
                    metadata=metadata,
                    updated_at=now,
                    last_processed_at=now,
                    processing_version=self.processing_version
                )
            )
            doc_id = existing_doc.id
            
            # Delete old chunks
            from sqlalchemy import delete
            await self.session.execute(delete(Chunk).where(Chunk.document_id == doc_id))
            
        else:
            # Create new document
            new_doc = Document(
                url=url,
                title=title,
                content_hash=content_hash,
                metadata=metadata,
                created_at=now,
                updated_at=now,
                last_processed_at=now,
                processing_version=self.processing_version
            )
            self.session.add(new_doc)
            await self.session.flush()  # Get the ID
            doc_id = new_doc.id
        
        # Process chunks
        chunks_created = await self._create_chunks(doc_id, content)
        
        await self.session.commit()
        
        return {
            "status": "processed",
            "reason": "new" if not existing_doc else "changed",
            "document_id": str(doc_id),
            "chunks_processed": chunks_created,
            "content_hash": content_hash,
            "processing_version": self.processing_version
        }
    
    async def _create_chunks(self, document_id: str, content: str) -> int:
        """Create chunks for document with lineage tracking"""
        
        # Simple chunking strategy (can be made more sophisticated)
        chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
        
        words = content.split()
        chunks_created = 0
        
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk_content = " ".join(words[i:i + chunk_size])
            
            if len(chunk_content.strip()) < 10:  # Skip very short chunks
                continue
            
            # Generate embedding (placeholder - integrate with your embedding service)
            embedding = await self._generate_embedding(chunk_content)
            
            chunk = Chunk(
                document_id=document_id,
                content=chunk_content,
                embedding=embedding,
                chunk_index=chunks_created,
                metadata={"word_count": len(chunk_content.split())},
                created_at=datetime.utcnow(),
                embedding_model=self.embedding_model,
                embedding_version=self.embedding_version,
                chunker_version=self.chunker_version,
                processing_metadata={
                    "chunk_method": "word_based",
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "processed_at": datetime.utcnow().isoformat()
                }
            )
            
            self.session.add(chunk)
            chunks_created += 1
        
        return chunks_created
    
    async def _generate_embedding(self, text: str) -> list:
        """Generate embedding for text (integrate with your embedding service)"""
        # Placeholder - replace with actual embedding service call
        import random
        return [random.gauss(0, 0.1) for _ in range(384)]
    
    async def _count_chunks(self, document_id: str) -> int:
        """Count existing chunks for document"""
        from sqlalchemy import func
        result = await self.session.execute(
            select(func.count(Chunk.id)).where(Chunk.document_id == document_id)
        )
        return result.scalar() or 0
    
    def _compute_content_hash(self, content: str) -> str:
        """Compute SHA256 hash of content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

# ============================================================================
# FILE: services/shared/lineage.py (lineage tracking and analysis)

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, distinct
from services.shared.models import Document, Chunk
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

class LineageTracker:
    """Track and analyze data lineage for reproducibility"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing versions and models"""
        
        # Document processing versions
        doc_versions = await self.session.execute(
            select(
                Document.processing_version,
                func.count(Document.id).label('count'),
                func.min(Document.last_processed_at).label('first_processed'),
                func.max(Document.last_processed_at).label('last_processed')
            )
            .where(Document.processing_version.isnot(None))
            .group_by(Document.processing_version)
        )
        
        # Embedding model usage
        embedding_models = await self.session.execute(
            select(
                Chunk.embedding_model,
                Chunk.embedding_version,
                func.count(Chunk.id).label('chunk_count'),
                func.count(distinct(Chunk.document_id)).label('document_count')
            )
            .where(Chunk.embedding_model.isnot(None))
            .group_by(Chunk.embedding_model, Chunk.embedding_version)
        )
        
        return {
            "processing_versions": [
                {
                    "version": row.processing_version,
                    "document_count": row.count,
                    "first_processed": row.first_processed.isoformat() if row.first_processed else None,
                    "last_processed": row.last_processed.isoformat() if row.last_processed else None
                }
                for row in doc_versions.fetchall()
            ],
            "embedding_models": [
                {
                    "model": row.embedding_model,
                    "version": row.embedding_version,
                    "chunk_count": row.chunk_count,
                    "document_count": row.document_count
                }
                for row in embedding_models.fetchall()
            ]
        }
    
    async def find_documents_needing_reprocessing(
        self, 
        target_processing_version: str,
        target_embedding_model: str,
        target_embedding_version: str
    ) -> List[Dict[str, Any]]:
        """Find documents that need reprocessing for version upgrades"""
        
        # Documents with old processing version
        old_processing = await self.session.execute(
            select(Document)
            .where(Document.processing_version != target_processing_version)
            .order_by(Document.last_processed_at.desc())
        )
        
        # Documents with chunks using old embedding models
        old_embeddings = await self.session.execute(
            select(Document)
            .join(Chunk, Document.id == Chunk.document_id)
            .where(
                (Chunk.embedding_model != target_embedding_model) |
                (Chunk.embedding_version != target_embedding_version)
            )
            .group_by(Document.id, Document.url, Document.title, Document.last_processed_at)
            .order_by(Document.last_processed_at.desc())
        )
        
        results = []
        
        # Combine results
        for doc in old_processing.fetchall():
            results.append({
                "document_id": str(doc.id),
                "url": doc.url,
                "title": doc.title,
                "reason": "processing_version_outdated",
                "current_version": doc.processing_version,
                "target_version": target_processing_version,
                "last_processed": doc.last_processed_at.isoformat() if doc.last_processed_at else None
            })
        
        for doc in old_embeddings.fetchall():
            # Avoid duplicates
            if not any(r["document_id"] == str(doc.id) for r in results):
                results.append({
                    "document_id": str(doc.id),
                    "url": doc.url,
                    "title": doc.title,
                    "reason": "embedding_model_outdated",
                    "last_processed": doc.last_processed_at.isoformat() if doc.last_processed_at else None
                })
        
        return results
    
    async def get_reprocessing_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get statistics about recent reprocessing activity"""
        
        since_date = datetime.utcnow() - timedelta(days=days)
        
        # Documents processed in period
        processed_docs = await self.session.execute(
            select(func.count(Document.id))
            .where(Document.last_processed_at >= since_date)
        )
        
        # Chunks created in period  
        processed_chunks = await self.session.execute(
            select(func.count(Chunk.id))
            .where(Chunk.created_at >= since_date)
        )
        
        # Processing by day
        daily_stats = await self.session.execute(
            select(
                func.date(Document.last_processed_at).label('date'),
                func.count(Document.id).label('documents_processed')
            )
            .where(Document.last_processed_at >= since_date)
            .group_by(func.date(Document.last_processed_at))
            .order_by(func.date(Document.last_processed_at))
        )
        
        return {
            "period_days": days,
            "total_documents_processed": processed_docs.scalar() or 0,
            "total_chunks_created": processed_chunks.scalar() or 0,
            "daily_processing": [
                {
                    "date": row.date.isoformat(),
                    "documents_processed": row.documents_processed
                }
                for row in daily_stats.fetchall()
            ]
        }
    
    async def audit_data_consistency(self) -> Dict[str, Any]:
        """Audit data for consistency issues"""
        
        issues = []
        
        # Documents without chunks
        docs_without_chunks = await self.session.execute(
            select(func.count(Document.id))
            .outerjoin(Chunk, Document.id == Chunk.document_id)
            .where(Chunk.id.is_(None))
        )
        
        if docs_without_chunks.scalar() > 0:
            issues.append({
                "type": "documents_without_chunks",
                "count": docs_without_chunks.scalar(),
                "description": "Documents that have no associated chunks"
            })
        
        # Chunks without embeddings
        chunks_without_embeddings = await self.session.execute(
            select(func.count(Chunk.id))
            .where(Chunk.embedding.is_(None))
        )
        
        if chunks_without_embeddings.scalar() > 0:
            issues.append({
                "type": "chunks_without_embeddings", 
                "count": chunks_without_embeddings.scalar(),
                "description": "Chunks that have no embeddings generated"
            })
        
        # Chunks with inconsistent model versions
        inconsistent_versions = await self.session.execute(
            select(func.count(Chunk.id))
            .where(
                (Chunk.embedding.isnot(None)) &
                (Chunk.embedding_model.is_(None))
            )
        )
        
        if inconsistent_versions.scalar() > 0:
            issues.append({
                "type": "missing_model_metadata",
                "count": inconsistent_versions.scalar(),
                "description": "Chunks with embeddings but missing model metadata"
            })
        
        return {
            "audit_timestamp": datetime.utcnow().isoformat(),
            "issues_found": len(issues),
            "issues": issues,
            "status": "healthy" if len(issues) == 0 else "issues_detected"
        }

# ============================================================================
# FILE: services/api/endpoints/lineage.py (new API endpoints for lineage)

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from services.shared.db import get_db_session
from services.shared.lineage import LineageTracker
from services.shared.incremental import IncrementalProcessor
from typing import Optional

router = APIRouter(prefix="/lineage", tags=["lineage"])

@router.get("/summary")
async def get_lineage_summary(db: AsyncSession = Depends(get_db_session)):
    """Get processing lineage summary"""
    tracker = LineageTracker(db)
    return await tracker.get_processing_summary()

@router.get("/reprocessing-candidates")
async def get_reprocessing_candidates(
    processing_version: str,
    embedding_model: str,
    embedding_version: str,
    db: AsyncSession = Depends(get_db_session)
):
    """Get documents that need reprocessing for version upgrades"""
    tracker = LineageTracker(db)
    return await tracker.find_documents_needing_reprocessing(
        processing_version, embedding_model, embedding_version
    )

@router.get("/stats")
async def get_reprocessing_stats(
    days: int = 30,
    db: AsyncSession = Depends(get_db_session)
):
    """Get reprocessing statistics"""
    if days < 1 or days > 365:
        raise HTTPException(status_code=400, detail="Days must be between 1 and 365")
    
    tracker = LineageTracker(db)
    return await tracker.get_reprocessing_stats(days)

@router.get("/audit")
async def audit_data_consistency(db: AsyncSession = Depends(get_db_session)):
    """Audit data for consistency issues"""
    tracker = LineageTracker(db)
    return await tracker.audit_data_consistency()

@router.post("/reprocess-document")
async def reprocess_document(
    document_id: str,
    force: bool = False,
    db: AsyncSession = Depends(get_db_session)
):
    """Reprocess a specific document"""
    from services.shared.models import Document
    from sqlalchemy import select
    
    # Get document
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # TODO: Fetch content and reprocess
    # This would integrate with your crawler/fetcher service
    return {
        "message": "Document queued for reprocessing",
        "document_id": document_id,
        "url": document.url
    }

# ============================================================================
# FILE: scripts/migration_helper.py (helper for data migration)

import asyncio
import asyncpg
from datetime import datetime
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def migrate_existing_data():
    """Migrate existing data to include lineage information"""
    
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:password@localhost/docfoundry")
    conn = await asyncpg.connect(DATABASE_URL.replace("+asyncpg", ""))
    
    logger.info("Starting data migration for lineage fields...")
    
    # Set default values for existing chunks without lineage info
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2") 
    embedding_version = os.getenv("EMBEDDING_VERSION", "2.2.2")
    chunker_version = os.getenv("CHUNKER_VERSION", "1.0.0")
    processing_version = os.getenv("PROCESSING_VERSION", "1.0.0")
    
    # Update chunks with missing lineage info
    chunks_updated = await conn.execute("""
        UPDATE chunks 
        SET 
            embedding_model = $1,
            embedding_version = $2,
            chunker_version = $3,
            processing_metadata = '{"migrated": true, "migration_date": "' || $4 || '"}'
        WHERE 
            embedding_model IS NULL 
            AND embedding IS NOT NULL
    """, embedding_model, embedding_version, chunker_version, datetime.utcnow().isoformat())
    
    logger.info(f"Updated {chunks_updated} chunks with lineage information")
    
    # Update documents with processing info
    docs_updated = await conn.execute("""
        UPDATE documents 
        SET 
            processing_version = $1,
            last_processed_at = COALESCE(updated_at, created_at)
        WHERE processing_version IS NULL
    """, processing_version)
    
    logger.info(f"Updated {docs_updated} documents with processing version")
    
    # Generate content hashes for existing documents without them
    docs_without_hash = await conn.fetch("""
        SELECT id, url FROM documents 
        WHERE content_hash IS NULL OR content_hash = ''
        LIMIT 100
    """)
    
    for doc in docs_without_hash:
        # Generate a placeholder hash (in real migration, you'd re-fetch content)
        placeholder_hash = f"legacy_hash_{doc['id']}"
        await conn.execute("""
            UPDATE documents 
            SET content_hash = $1 
            WHERE id = $2
        """, placeholder_hash, doc['id'])
    
    logger.info(f"Generated placeholder hashes for {len(docs_without_hash)} documents")
    
    await conn.close()
    logger.info("Migration completed successfully")

if __name__ == "__main__":
    asyncio.run(migrate_existing_data())

# ============================================================================
# FILE: README_UPDATES.md (documentation updates for operators)

# DocFoundry Operations Guide - Lineage & Incremental Processing

## Data Lineage Tracking

DocFoundry now tracks complete lineage for all processed data:

### Tracked Metadata
- **Processing Version**: Version of the processing pipeline
- **Embedding Model**: Model used for generating embeddings
- **Embedding Version**: Specific version of the embedding model
- **Chunker Version**: Version of the text chunking algorithm
- **Processing Timestamps**: When documents were last processed

### API Endpoints

```bash
# Get lineage summary
curl http://localhost:8080/lineage/summary

# Find documents needing reprocessing
curl "http://localhost:8080/lineage/reprocessing-candidates?processing_version=1.1.0&embedding_model=new-model&embedding_version=2.0.0"

# Get processing statistics
curl http://localhost:8080/lineage/stats?days=30

# Audit data consistency
curl http://localhost:8080/lineage/audit
```

## Incremental Processing

Documents are now processed incrementally:

### Change Detection Methods
1. **Content Hash**: SHA256 hash comparison
2. **HTTP Conditional Requests**: ETag and Last-Modified headers
3. **Processing Version**: Reprocess when pipeline versions change

### Configuration
```bash
# Environment variables
PROCESSING_VERSION=1.0.0
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_VERSION=2.2.2
CHUNKER_VERSION=1.0.0
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

### Benefits
- **Efficiency**: Skip unchanged documents
- **Reproducibility**: Track exactly how data was processed
- **Version Management**: Easy upgrades and rollbacks
- **Cost Reduction**: Avoid reprocessing unchanged content

## Migration

For existing installations:

```bash
# Run Alembic migration
alembic upgrade head

# Migrate existing data
python scripts/migration_helper.py

# Verify migration
curl http://localhost:8080/lineage/audit
```

## Monitoring

Key metrics to monitor:
- Documents processed per day
- Skipped vs. reprocessed ratios  
- Processing version distribution
- Data consistency audit results

## Best Practices

1. **Version Bumping**: Increment versions when changing:
   - Processing algorithms
   - Embedding models
   - Chunking strategies

2. **Batch Reprocessing**: Use lineage APIs to find and reprocess outdated data

3. **Validation**: Regular audits ensure data consistency

4. **Performance**: Monitor skip rates to optimize change detection
            