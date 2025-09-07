-- Migration: Add lineage metadata to chunks table
-- This migration adds fields to track chunker and embedding model versions
-- for better A/B testing and model management

-- For PostgreSQL
-- Add lineage metadata columns to chunks table
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunker_version TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunker_config JSONB;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_model TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_version TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_dimensions INTEGER;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS processing_timestamp TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS lineage_id TEXT;

-- Create index for lineage queries
CREATE INDEX IF NOT EXISTS idx_chunks_chunker_version ON chunks(chunker_version);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_model ON chunks(embedding_model);
CREATE INDEX IF NOT EXISTS idx_chunks_lineage_id ON chunks(lineage_id);
CREATE INDEX IF NOT EXISTS idx_chunks_processing_timestamp ON chunks(processing_timestamp);

-- For SQLite (alternative syntax)
-- Note: SQLite doesn't support IF NOT EXISTS for ALTER TABLE
-- These would need to be run conditionally in application code

/*
-- SQLite version (run conditionally)
ALTER TABLE chunks ADD COLUMN chunker_version TEXT;
ALTER TABLE chunks ADD COLUMN chunker_config TEXT; -- JSON as TEXT in SQLite
ALTER TABLE chunks ADD COLUMN embedding_model TEXT;
ALTER TABLE chunks ADD COLUMN embedding_version TEXT;
ALTER TABLE chunks ADD COLUMN embedding_dimensions INTEGER;
ALTER TABLE chunks ADD COLUMN processing_timestamp TEXT; -- ISO timestamp as TEXT
ALTER TABLE chunks ADD COLUMN lineage_id TEXT;

-- SQLite indexes
CREATE INDEX IF NOT EXISTS idx_chunks_chunker_version ON chunks(chunker_version);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_model ON chunks(embedding_model);
CREATE INDEX IF NOT EXISTS idx_chunks_lineage_id ON chunks(lineage_id);
CREATE INDEX IF NOT EXISTS idx_chunks_processing_timestamp ON chunks(processing_timestamp);
*/

-- Create a view for lineage analysis
CREATE OR REPLACE VIEW chunk_lineage_summary AS
SELECT 
    chunker_version,
    embedding_model,
    embedding_version,
    COUNT(*) as chunk_count,
    MIN(processing_timestamp) as first_processed,
    MAX(processing_timestamp) as last_processed,
    COUNT(DISTINCT document_id) as document_count
FROM chunks 
WHERE chunker_version IS NOT NULL
GROUP BY chunker_version, embedding_model, embedding_version
ORDER BY last_processed DESC;

-- Create a function to generate lineage IDs
CREATE OR REPLACE FUNCTION generate_lineage_id(
    p_chunker_version TEXT,
    p_embedding_model TEXT,
    p_embedding_version TEXT
) RETURNS TEXT AS $$
BEGIN
    RETURN CONCAT(
        COALESCE(p_chunker_version, 'unknown'), 
        '-',
        COALESCE(p_embedding_model, 'unknown'),
        '-',
        COALESCE(p_embedding_version, 'unknown')
    );
END;
$$ LANGUAGE plpgsql;

-- Update existing chunks with default lineage metadata
-- This should be run after the migration to populate existing data
/*
UPDATE chunks 
SET 
    chunker_version = 'v1.0.0',
    embedding_model = 'sentence-transformers/all-MiniLM-L6-v2',
    embedding_version = 'v1.0.0',
    embedding_dimensions = 384,
    processing_timestamp = NOW(),
    lineage_id = generate_lineage_id('v1.0.0', 'sentence-transformers/all-MiniLM-L6-v2', 'v1.0.0')
WHERE chunker_version IS NULL;
*/