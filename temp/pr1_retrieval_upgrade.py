# ============================================================================
# PR-1: Retrieval Quality Upgrade
# Files: Alembic migration + Production retrieval service + DB connection config
# ============================================================================

# FILE: alembic/versions/001_base_schema.py
"""Base schema with FTS and vector indexes

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Ensure required extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    op.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm')
    
    # Create documents table
    op.create_table('documents',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('url', sa.Text(), nullable=False),
        sa.Column('title', sa.Text(), nullable=True),
        sa.Column('content_hash', sa.String(64), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create chunks table
    op.create_table('chunks',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('document_id', sa.UUID(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding', postgresql.ARRAY(sa.Float), nullable=True),
        sa.Column('fts', postgresql.TSVECTOR(), nullable=True),  # Maintained FTS column
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index('idx_documents_url', 'documents', ['url'])
    op.create_index('idx_documents_content_hash', 'documents', ['content_hash'])
    op.create_index('idx_documents_created_at', 'documents', ['created_at'])
    op.create_index('idx_chunks_document_id', 'chunks', ['document_id'])
    
    # Vector index (IVFFLAT for similarity search)
    op.execute('CREATE INDEX chunks_embedding_idx ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)')
    
    # FTS indexes
    op.execute('CREATE INDEX chunks_fts_idx ON chunks USING gin(fts)')
    op.execute('CREATE INDEX chunks_content_trgm_idx ON chunks USING gin(content gin_trgm_ops)')
    
    # Trigger to maintain FTS column
    op.execute("""
        CREATE OR REPLACE FUNCTION update_fts_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.fts := to_tsvector('english', NEW.content);
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    op.execute("""
        CREATE TRIGGER chunks_fts_trigger
        BEFORE INSERT OR UPDATE ON chunks
        FOR EACH ROW EXECUTE FUNCTION update_fts_column();
    """)

def downgrade() -> None:
    op.execute('DROP TRIGGER IF EXISTS chunks_fts_trigger ON chunks')
    op.execute('DROP FUNCTION IF EXISTS update_fts_column()')
    op.drop_table('chunks')
    op.drop_table('documents')

# ============================================================================
# FILE: services/shared/retrieval.py
# ============================================================================

import asyncio
from typing import List, Tuple, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, func
import numpy as np
from dataclasses import dataclass
import os

@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]

class ProductionRetrievalService:
    """Production-grade hybrid retrieval: Vector + BM25 + RRF"""
    
    def __init__(self, session: AsyncSession, embedding_service):
        self.session = session
        self.embedding_service = embedding_service
        self.rrf_k = float(os.getenv("RRF_K", "60.0"))
        
    async def hybrid_search(
        self, 
        query: str, 
        site: Optional[str] = None,
        limit: int = 10,
        vector_weight: float = 0.7
    ) -> List[SearchResult]:
        """
        Hybrid search combining vector similarity and BM25 text search with RRF fusion
        """
        # Get both vector and BM25 results in parallel
        dense_task = asyncio.create_task(self.vector_search(query, site, limit * 5))
        sparse_task = asyncio.create_task(self.bm25_search(query, site, limit * 5))
        
        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
        
        # Apply RRF fusion
        fused_results = self.rrf_fusion(dense_results, sparse_results, self.rrf_k)
        
        return fused_results[:limit]
    
    async def vector_search(
        self, 
        query: str, 
        site: Optional[str] = None, 
        limit: int = 50
    ) -> List[Tuple[str, float]]:
        """Vector similarity search using pgvector"""
        
        # Generate query embedding
        query_embedding = await self.embedding_service.embed_text(query)
        
        # Build query with optional site filtering
        site_filter = ""
        params = {"query_embedding": query_embedding, "limit": limit}
        
        if site:
            site_filter = "AND d.url LIKE :site_pattern"
            params["site_pattern"] = f"%{site}%"
        
        query_sql = f"""
            SELECT 
                c.id,
                1 - (c.embedding <=> :query_embedding) as similarity_score
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.embedding IS NOT NULL
            {site_filter}
            ORDER BY c.embedding <=> :query_embedding
            LIMIT :limit
        """
        
        result = await self.session.execute(text(query_sql), params)
        return [(row.id, row.similarity_score) for row in result.fetchall()]
    
    async def bm25_search(
        self, 
        query: str, 
        site: Optional[str] = None, 
        limit: int = 50
    ) -> List[Tuple[str, float]]:
        """BM25 text search using PostgreSQL's ts_rank_cd"""
        
        # Build query with optional site filtering
        site_filter = ""
        params = {"query": query, "limit": limit}
        
        if site:
            site_filter = "AND d.url LIKE :site_pattern"
            params["site_pattern"] = f"%{site}%"
        
        query_sql = f"""
            SELECT 
                c.id,
                ts_rank_cd(c.fts, plainto_tsquery('english', :query), 32) as bm25_score
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.fts @@ plainto_tsquery('english', :query)
            {site_filter}
            ORDER BY ts_rank_cd(c.fts, plainto_tsquery('english', :query), 32) DESC
            LIMIT :limit
        """
        
        result = await self.session.execute(text(query_sql), params)
        return [(row.id, row.bm25_score) for row in result.fetchall()]
    
    def rrf_fusion(
        self, 
        dense_results: List[Tuple[str, float]], 
        sparse_results: List[Tuple[str, float]], 
        k: float = 60.0
    ) -> List[str]:
        """
        Reciprocal Rank Fusion to combine vector and BM25 results
        Enhanced version that uses actual BM25 scores for better ranking
        """
        # Create rank mappings
        dense_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(dense_results)}
        sparse_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sparse_results)}
        sparse_scores = {doc_id: score for doc_id, score in sparse_results}
        
        # Get all unique document IDs
        all_docs = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        
        # Calculate RRF scores
        rrf_scores = {}
        for doc_id in all_docs:
            dense_score = 1 / (k + dense_ranks.get(doc_id, len(dense_results) + 1))
            sparse_score = 1 / (k + sparse_ranks.get(doc_id, len(sparse_results) + 1))
            
            # Weight by actual BM25 score if available
            if doc_id in sparse_scores:
                sparse_score *= (1 + sparse_scores[doc_id])
            
            rrf_scores[doc_id] = dense_score + sparse_score
        
        # Sort by RRF score and return document IDs
        return sorted(all_docs, key=lambda x: rrf_scores[x], reverse=True)
    
    async def optimize_vector_index(self) -> None:
        """Optimize IVFFLAT index based on current data size"""
        
        # Get current chunk count
        result = await self.session.execute(text("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"))
        chunk_count = result.scalar()
        
        if chunk_count == 0:
            return
        
        # Calculate optimal lists parameter (approximately sqrt(rows))
        optimal_lists = max(1, int(np.sqrt(chunk_count)))
        
        # Recreate index with optimal parameters
        await self.session.execute(text("DROP INDEX IF EXISTS chunks_embedding_idx"))
        await self.session.execute(text(f"""
            CREATE INDEX chunks_embedding_idx ON chunks 
            USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = {optimal_lists})
        """))
        await self.session.commit()

# ============================================================================
# FILE: services/shared/db.py (connection configuration)
# ============================================================================

from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import os

# Database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/docfoundry")

# Create async engine
async_engine = create_async_engine(
    DATABASE_URL,
    echo=os.getenv("SQL_ECHO", "false").lower() == "true",
    pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
    max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
)

# Configure connection parameters for optimal pgvector performance
@event.listens_for(async_engine.sync_engine, "connect")
def configure_connection(dbapi_connection, connection_record):
    """Configure connection-level parameters for optimal performance"""
    with dbapi_connection.cursor() as cur:
        # Optimize IVFFLAT probes for query performance
        probes = os.getenv("IVFFLAT_PROBES", "10")
        cur.execute(f"SET ivfflat.probes = {probes}")
        
        # Set default text search configuration
        cur.execute("SET default_text_search_config = 'english'")
        
        # Optimize for read-heavy workloads
        cur.execute("SET random_page_cost = 1.1")  # Assume SSD storage
        
        # Connection-level performance tuning
        cur.execute("SET work_mem = '256MB'")  # For complex queries
        cur.execute("SET maintenance_work_mem = '512MB'")  # For index operations

# Session factory
AsyncSessionLocal = sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)

async def get_db_session() -> AsyncSession:
    """Dependency for getting database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# Utility function for index optimization
async def optimize_all_indexes():
    """Optimize all vector indexes - run after bulk data loading"""
    async with AsyncSessionLocal() as session:
        retrieval_service = ProductionRetrievalService(session, None)
        await retrieval_service.optimize_vector_index()
