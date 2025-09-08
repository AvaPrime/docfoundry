"""Retrieval quality upgrade with FTS and vector indexes

Revision ID: 004
Revises: 003
Create Date: 2024-01-15 10:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision: str = '004'
down_revision: Union[str, None] = '003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Ensure required PostgreSQL extensions are available
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute("CREATE EXTENSION IF NOT EXISTS btree_gin")
    
    # Add FTS columns to documents table
    op.add_column('documents', sa.Column('fts_vector', postgresql.TSVECTOR(), nullable=True))
    op.add_column('documents', sa.Column('content_hash', sa.String(64), nullable=True))
    
    # Add FTS columns to chunks table
    op.add_column('chunks', sa.Column('fts_vector', postgresql.TSVECTOR(), nullable=True))
    op.add_column('chunks', sa.Column('content_hash', sa.String(64), nullable=True))
    
    # Create GIN indexes for full-text search
    op.create_index('idx_documents_fts', 'documents', ['fts_vector'], postgresql_using='gin')
    op.create_index('idx_chunks_fts', 'chunks', ['fts_vector'], postgresql_using='gin')
    
    # Create indexes for content hashes (deduplication)
    op.create_index('idx_documents_content_hash', 'documents', ['content_hash'])
    op.create_index('idx_chunks_content_hash', 'chunks', ['content_hash'])
    
    # Create trigram indexes for fuzzy text matching
    op.create_index('idx_documents_title_trgm', 'documents', ['title'], postgresql_using='gin', postgresql_ops={'title': 'gin_trgm_ops'})
    op.create_index('idx_chunks_text_trgm', 'chunks', ['text'], postgresql_using='gin', postgresql_ops={'text': 'gin_trgm_ops'})
    
    # Create composite indexes for hybrid search
    op.create_index('idx_chunks_embedding_fts', 'chunks', ['embedding', 'fts_vector'])
    op.create_index('idx_documents_hash_updated', 'documents', ['content_hash', 'updated_at'])
    
    # Update FTS vectors for existing data
    op.execute("""
        UPDATE documents 
        SET fts_vector = to_tsvector('english', COALESCE(title, '') || ' ' || COALESCE(path, ''))
        WHERE fts_vector IS NULL
    """)
    
    op.execute("""
        UPDATE chunks 
        SET fts_vector = to_tsvector('english', COALESCE(heading, '') || ' ' || COALESCE(text, ''))
        WHERE fts_vector IS NULL
    """)
    
    # Create triggers to automatically update FTS vectors
    op.execute("""
        CREATE OR REPLACE FUNCTION update_documents_fts()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.fts_vector = to_tsvector('english', COALESCE(NEW.title, '') || ' ' || COALESCE(NEW.path, ''));
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    op.execute("""
        CREATE TRIGGER documents_fts_update
        BEFORE INSERT OR UPDATE ON documents
        FOR EACH ROW EXECUTE FUNCTION update_documents_fts();
    """)
    
    op.execute("""
        CREATE OR REPLACE FUNCTION update_chunks_fts()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.fts_vector = to_tsvector('english', COALESCE(NEW.heading, '') || ' ' || COALESCE(NEW.text, ''));
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    op.execute("""
        CREATE TRIGGER chunks_fts_update
        BEFORE INSERT OR UPDATE ON chunks
        FOR EACH ROW EXECUTE FUNCTION update_chunks_fts();
    """)

def downgrade() -> None:
    # Drop triggers
    op.execute("DROP TRIGGER IF EXISTS chunks_fts_update ON chunks")
    op.execute("DROP TRIGGER IF EXISTS documents_fts_update ON documents")
    
    # Drop functions
    op.execute("DROP FUNCTION IF EXISTS update_chunks_fts()")
    op.execute("DROP FUNCTION IF EXISTS update_documents_fts()")
    
    # Drop indexes
    op.drop_index('idx_documents_hash_updated')
    op.drop_index('idx_chunks_embedding_fts')
    op.drop_index('idx_chunks_text_trgm')
    op.drop_index('idx_documents_title_trgm')
    op.drop_index('idx_chunks_content_hash')
    op.drop_index('idx_documents_content_hash')
    op.drop_index('idx_chunks_fts')
    op.drop_index('idx_documents_fts')
    
    # Drop columns
    op.drop_column('chunks', 'content_hash')
    op.drop_column('chunks', 'fts_vector')
    op.drop_column('documents', 'content_hash')
    op.drop_column('documents', 'fts_vector')