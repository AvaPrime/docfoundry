# PostgreSQL Migration Guide

This document describes the migration from SQLite to PostgreSQL with pgvector support for improved scalability and vector operations.

## Overview

DocFoundry now supports both SQLite (development) and PostgreSQL (production) databases through a unified adapter interface. The PostgreSQL implementation includes:

- **pgvector extension** for efficient vector similarity search
- **Connection pooling** for better performance
- **Async operations** for improved concurrency
- **Enhanced indexing** for faster queries
- **Click feedback tracking** for learning-to-rank

## Architecture Changes

### Database Adapters

- `SQLiteAdapter`: Maintains backward compatibility with existing SQLite databases
- `PostgresAdapter`: New PostgreSQL implementation with pgvector support
- `DatabaseConfig`: Unified configuration management

### Key Features

1. **Vector Search**: Native pgvector support for semantic search
2. **Hybrid Search**: Combines full-text search with vector similarity
3. **Click Tracking**: Enhanced feedback system for learning-to-rank
4. **Connection Pooling**: Efficient database connection management
5. **Async Operations**: Non-blocking database operations

## Migration Process

### Prerequisites

1. **PostgreSQL 15+** with pgvector extension
2. **Python dependencies**: `asyncpg`, `psycopg2-binary`, `pgvector`
3. **Environment configuration**: Database connection settings

### Step 1: Install PostgreSQL and pgvector

```bash
# Install PostgreSQL (Ubuntu/Debian)
sudo apt update
sudo apt install postgresql postgresql-contrib

# Install pgvector extension
sudo apt install postgresql-15-pgvector

# Or compile from source
git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

### Step 2: Create Database and User

```sql
-- Connect as postgres user
sudo -u postgres psql

-- Create database and user
CREATE DATABASE docfoundry;
CREATE USER docfoundry WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE docfoundry TO docfoundry;

-- Connect to docfoundry database
\c docfoundry

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Grant usage on schema
GRANT USAGE ON SCHEMA public TO docfoundry;
GRANT CREATE ON SCHEMA public TO docfoundry;
```

### Step 3: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env file
DATABASE_TYPE=postgresql
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=docfoundry
POSTGRES_USER=docfoundry
POSTGRES_PASSWORD=your_secure_password
```

### Step 4: Initialize PostgreSQL Schema

```bash
# Run schema initialization
python -c "import asyncio; from config import initialize_database; asyncio.run(initialize_database())"
```

### Step 5: Migrate Data (Optional)

If you have existing SQLite data:

```bash
# Run migration script
python scripts/migrate_to_postgres.py
```

## Database Schema

### Core Tables

```sql
-- Documents table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    title TEXT,
    source_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chunks table with vector embeddings
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    heading TEXT,
    anchor TEXT,
    embedding VECTOR(1536),  -- OpenAI embedding dimension
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Click feedback for learning-to-rank
CREATE TABLE click_feedback (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    chunk_id INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    session_id TEXT,
    user_id TEXT,
    dwell_time FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Indexes

```sql
-- Vector similarity search index
CREATE INDEX chunks_embedding_idx ON chunks 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Full-text search index
CREATE INDEX chunks_text_gin_idx ON chunks USING gin(to_tsvector('english', text));

-- Performance indexes
CREATE INDEX chunks_document_id_idx ON chunks(document_id);
CREATE INDEX click_feedback_query_idx ON click_feedback(query);
CREATE INDEX click_feedback_session_idx ON click_feedback(session_id);
```

## API Changes

### Unified Interface

All search endpoints now use the database adapter pattern:

```python
@app.post("/search")
async def search(req: SearchRequest, db = Depends(get_db)):
    results = await db.search_fulltext(req.q, req.k)
    return {"results": results}

@app.post("/search/semantic")
async def semantic_search(req: SemanticSearchRequest, db = Depends(get_db)):
    results = await db.search_semantic(req.q, req.k, req.min_similarity)
    return {"results": results, "search_type": "semantic"}

@app.post("/search/hybrid")
async def hybrid_search(req: HybridSearchRequest, db = Depends(get_db)):
    results = await db.search_hybrid(req.q, req.k, req.rrf_k, req.min_similarity)
    return {"results": results, "search_type": "hybrid_rrf"}
```

### Enhanced Features

1. **Async Operations**: All database operations are now asynchronous
2. **Connection Pooling**: Automatic connection management
3. **Error Handling**: Improved error handling and logging
4. **Performance Monitoring**: Built-in query performance tracking

## Performance Considerations

### Vector Search Optimization

- **IVFFlat Index**: Approximate nearest neighbor search for large datasets
- **Embedding Dimensions**: Optimized for OpenAI embeddings (1536 dimensions)
- **Index Tuning**: Configurable `lists` parameter for index performance

### Connection Pooling

```python
# PostgreSQL connection pool configuration
pool_config = {
    'min_size': 5,
    'max_size': 20,
    'max_queries': 50000,
    'max_inactive_connection_lifetime': 300.0
}
```

### Query Optimization

- **Prepared Statements**: Reduced query parsing overhead
- **Batch Operations**: Efficient bulk inserts and updates
- **Index Usage**: Optimized queries for vector and text search

## Monitoring and Observability

### Database Metrics

- Connection pool utilization
- Query execution times
- Vector search performance
- Index usage statistics

### Health Checks

```python
@app.get("/health")
async def health_check():
    db_status = await db.health_check()
    return {
        "status": "healthy" if db_status else "unhealthy",
        "database": "postgresql" if isinstance(db, PostgresAdapter) else "sqlite",
        "timestamp": datetime.utcnow().isoformat()
    }
```

## Troubleshooting

### Common Issues

1. **pgvector Extension Missing**
   ```
   ERROR: extension "vector" is not available
   ```
   Solution: Install pgvector extension

2. **Connection Pool Exhaustion**
   ```
   ERROR: too many connections
   ```
   Solution: Adjust pool size or check for connection leaks

3. **Vector Index Performance**
   ```
   SLOW: Vector similarity queries taking too long
   ```
   Solution: Tune IVFFlat index parameters or increase `lists`

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with database query logging
export POSTGRES_LOG_QUERIES=true
```

## Migration Rollback

To rollback to SQLite:

1. Update environment: `DATABASE_TYPE=sqlite`
2. Restart application
3. Existing SQLite database will be used

## Next Steps

1. **Performance Tuning**: Monitor and optimize query performance
2. **Scaling**: Consider read replicas for high-traffic deployments
3. **Backup Strategy**: Implement regular database backups
4. **Security**: Configure SSL/TLS for production deployments

## References

- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [PostgreSQL Performance Tuning](https://wiki.postgresql.org/wiki/Performance_Optimization)
- [FastAPI Async Database](https://fastapi.tiangolo.com/advanced/async-sql-databases/)