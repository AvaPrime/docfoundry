-- PostgreSQL schema with pgvector extension for DocFoundry
-- Migration from SQLite to PostgreSQL for production scalability

-- Enable pgvector extension for vector operations
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable foreign key constraints
SET foreign_keys = ON;

-- Documents table with enhanced metadata
CREATE TABLE IF NOT EXISTS documents (
  id SERIAL PRIMARY KEY,
  path TEXT UNIQUE NOT NULL,
  title TEXT,
  source_url TEXT,
  captured_at TIMESTAMPTZ DEFAULT NOW(),
  hash TEXT,
  content_type TEXT DEFAULT 'text/markdown',
  language TEXT,
  word_count INTEGER,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chunks table with pgvector support
CREATE TABLE IF NOT EXISTS chunks (
  id SERIAL PRIMARY KEY,
  document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  chunk_id TEXT UNIQUE NOT NULL,
  heading TEXT,
  anchor TEXT,
  text TEXT NOT NULL,
  h_path TEXT[], -- Array of heading hierarchy
  url TEXT,
  retrieved_at TIMESTAMPTZ,
  token_len INTEGER,
  lang TEXT,
  hash TEXT,
  embedding VECTOR(1536), -- OpenAI embedding dimension
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Sources table for source configuration
CREATE TABLE IF NOT EXISTS sources (
  id SERIAL PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  base_urls TEXT[] NOT NULL,
  sitemaps TEXT[],
  include_patterns TEXT[],
  exclude_patterns TEXT[],
  rate_limit DECIMAL(5,2) DEFAULT 0.5,
  backoff_base DECIMAL(5,2) DEFAULT 0.25,
  backoff_max DECIMAL(5,2) DEFAULT 8.0,
  depth INTEGER DEFAULT 4,
  priority TEXT DEFAULT 'normal' CHECK (priority IN ('low', 'normal', 'high')),
  auth_config JSONB,
  license_hint TEXT,
  enabled BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Jobs table for background processing
CREATE TABLE IF NOT EXISTS jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_type TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'completed', 'failed')),
  source_id INTEGER REFERENCES sources(id),
  parameters JSONB,
  result JSONB,
  error_message TEXT,
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Click feedback table for learning-to-rank
CREATE TABLE IF NOT EXISTS click_feedback (
  id SERIAL PRIMARY KEY,
  session_id TEXT,
  user_id TEXT,
  query TEXT NOT NULL,
  chunk_id TEXT NOT NULL,
  position INTEGER NOT NULL,
  clicked BOOLEAN DEFAULT TRUE,
  dwell_time DECIMAL(10,3),
  timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Search sessions for analytics
CREATE TABLE IF NOT EXISTS search_sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id TEXT,
  query TEXT NOT NULL,
  results_count INTEGER,
  response_time_ms INTEGER,
  search_type TEXT DEFAULT 'hybrid',
  filters JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);
CREATE INDEX IF NOT EXISTS idx_documents_captured_at ON documents(captured_at);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id ON chunks(chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(hash);
CREATE INDEX IF NOT EXISTS idx_chunks_lang ON chunks(lang);

-- Vector index for similarity search (IVFFlat with cosine distance)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks 
  USING ivfflat (embedding vector_cosine_ops) 
  WITH (lists = 100);

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_chunks_text_fts ON chunks 
  USING gin(to_tsvector('english', text));

CREATE INDEX IF NOT EXISTS idx_chunks_heading_fts ON chunks 
  USING gin(to_tsvector('english', heading));

CREATE INDEX IF NOT EXISTS idx_sources_name ON sources(name);
CREATE INDEX IF NOT EXISTS idx_sources_enabled ON sources(enabled);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_job_type ON jobs(job_type);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);

CREATE INDEX IF NOT EXISTS idx_click_feedback_query ON click_feedback(query);
CREATE INDEX IF NOT EXISTS idx_click_feedback_chunk_id ON click_feedback(chunk_id);
CREATE INDEX IF NOT EXISTS idx_click_feedback_session_id ON click_feedback(session_id);
CREATE INDEX IF NOT EXISTS idx_click_feedback_timestamp ON click_feedback(timestamp);

CREATE INDEX IF NOT EXISTS idx_search_sessions_user_id ON search_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_search_sessions_created_at ON search_sessions(created_at);

-- Update triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_chunks_updated_at BEFORE UPDATE ON chunks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_sources_updated_at BEFORE UPDATE ON sources
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_jobs_updated_at BEFORE UPDATE ON jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();