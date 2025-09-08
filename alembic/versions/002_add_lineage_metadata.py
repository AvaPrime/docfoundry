"""Add lineage metadata to chunks"""
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
    
    # Create indexes for lineage queries
    op.create_index("idx_chunks_chunker_version", "chunks", ["chunker_version"])
    op.create_index("idx_chunks_embedding_model", "chunks", ["embedding_model"])
    op.create_index("idx_chunks_lineage_id", "chunks", ["lineage_id"])
    op.create_index("idx_chunks_processing_timestamp", "chunks", ["processing_timestamp"])

def downgrade():
    # Drop indexes
    op.drop_index("idx_chunks_processing_timestamp", "chunks")
    op.drop_index("idx_chunks_lineage_id", "chunks")
    op.drop_index("idx_chunks_embedding_model", "chunks")
    op.drop_index("idx_chunks_chunker_version", "chunks")
    
    # Drop columns
    op.drop_column("chunks", "lineage_id")
    op.drop_column("chunks", "processing_timestamp")
    op.drop_column("chunks", "embedding_dimensions")
    op.drop_column("chunks", "embedding_version")
    op.drop_column("chunks", "embedding_model")
    op.drop_column("chunks", "chunker_config")
    op.drop_column("chunks", "chunker_version")