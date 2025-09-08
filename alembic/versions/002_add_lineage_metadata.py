"""Add lineage and versioning columns"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None

def upgrade():
    # Add lineage metadata columns to chunks table
    op.add_column("chunks", sa.Column("chunker_version", sa.Text(), nullable=True))
    op.add_column("chunks", sa.Column("chunker_config", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("chunks", sa.Column("embedding_model", sa.Text(), nullable=True))
    op.add_column("chunks", sa.Column("embedding_version", sa.Text(), nullable=True))
    op.add_column("chunks", sa.Column("embedding_dimensions", sa.Integer(), nullable=True))
    op.add_column("chunks", sa.Column("processing_timestamp", sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=True))
    op.add_column("chunks", sa.Column("lineage_id", sa.Text(), nullable=True))
    op.add_column("chunks", sa.Column("processing_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    
    # Add processing tracking columns to documents
    op.add_column("documents", sa.Column("last_processed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("documents", sa.Column("processing_version", sa.Text(), nullable=True))
    op.add_column("documents", sa.Column("etag", sa.Text(), nullable=True))
    op.add_column("documents", sa.Column("last_modified", sa.DateTime(timezone=True), nullable=True))
    
    # Create indexes for lineage queries
    op.create_index("idx_chunks_chunker_version", "chunks", ["chunker_version"])
    op.create_index("idx_chunks_embedding_model", "chunks", ["embedding_model"])
    op.create_index("idx_chunks_lineage_id", "chunks", ["lineage_id"])
    op.create_index("idx_chunks_processing_timestamp", "chunks", ["processing_timestamp"])
    op.create_index("idx_documents_content_hash", "documents", ["content_hash"])
    op.create_index("idx_documents_last_processed", "documents", ["last_processed_at"])
    op.create_index("idx_documents_etag", "documents", ["etag"])

def downgrade():
    # Drop indexes
    op.drop_index("idx_documents_etag", "documents")
    op.drop_index("idx_documents_last_processed", "documents")
    op.drop_index("idx_documents_content_hash", "documents")
    op.drop_index("idx_chunks_processing_timestamp", "chunks")
    op.drop_index("idx_chunks_lineage_id", "chunks")
    op.drop_index("idx_chunks_embedding_model", "chunks")
    op.drop_index("idx_chunks_chunker_version", "chunks")
    
    # Drop columns from documents
    op.drop_column("documents", "last_modified")
    op.drop_column("documents", "etag")
    op.drop_column("documents", "processing_version")
    op.drop_column("documents", "last_processed_at")
    
    # Drop columns from chunks
    op.drop_column("chunks", "processing_metadata")
    op.drop_column("chunks", "lineage_id")
    op.drop_column("chunks", "processing_timestamp")
    op.drop_column("chunks", "embedding_dimensions")
    op.drop_column("chunks", "embedding_version")
    op.drop_column("chunks", "embedding_model")
    op.drop_column("chunks", "chunker_config")
    op.drop_column("chunks", "chunker_version")