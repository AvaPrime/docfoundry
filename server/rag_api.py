from fastapi import FastAPI, Body, HTTPException, Depends
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Union, AsyncGenerator
import pathlib, json, datetime, re, logging, uuid, asyncio, time
import sys
import os
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "indexer"))
from embeddings import EmbeddingManager
from .jobs import job_manager, register_default_handlers, JobStatus

# Import database configuration
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from config import get_db_adapter, initialize_database, DatabaseConfig
from indexer.postgres_adapter import PostgresAdapter
from indexer.sqlite_adapter import SQLiteAdapter

# Import learning-to-rank for click feedback
try:
    from learning_to_rank import LearningToRankReranker
except ImportError:
    LearningToRankReranker = None
    logging.warning("Learning-to-rank module not available for click feedback")

# Import monitoring system
try:
    from server.monitoring import metrics_collector, MonitoringMiddleware, track_search_performance, SearchMetrics
except ImportError:
    metrics_collector = None
    MonitoringMiddleware = None
    track_search_performance = None
    SearchMetrics = None
    logging.warning("Monitoring system not available")

# Import OpenTelemetry for observability
try:
    from observability.telemetry import (
        init_telemetry, 
        shutdown_telemetry, 
        instrument_fastapi_app,
        trace_function,
        trace_span,
        record_counter,
        record_histogram,
        record_gauge
    )
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False
    logging.warning("OpenTelemetry not available - running without observability")

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "docfoundry.db"
DOCS_DIR = BASE_DIR / "docs"

app = FastAPI(title="DocFoundry RAG API", version="0.2.0")

# Add monitoring middleware
if MonitoringMiddleware and metrics_collector:
    app.add_middleware(MonitoringMiddleware, metrics_collector=metrics_collector)

# Initialize OpenTelemetry instrumentation
if TELEMETRY_AVAILABLE:
    # Initialize telemetry with environment configuration
    init_telemetry(
        service_name=os.getenv("OTEL_SERVICE_NAME", "docfoundry"),
        service_version="0.2.0",
        environment=os.getenv("ENVIRONMENT", "development"),
        otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        enable_prometheus=True
    )
    # Instrument FastAPI app
    instrument_fastapi_app(app)
    logging.info("OpenTelemetry instrumentation enabled")

# Global database adapter
db_adapter: Optional[Union[PostgresAdapter, SQLiteAdapter]] = None

# Initialize embedding manager for semantic search
embedding_manager = None

# Initialize learning-to-rank reranker for click feedback
ltr_reranker = None

@app.on_event("startup")
async def startup_event():
    """Initialize database, components, and job manager on startup."""
    global db_adapter, embedding_manager, ltr_reranker
    
    try:
        # Initialize database
        await initialize_database()
        db_adapter = await get_db_adapter()
        logging.info(f"Database initialized: {type(db_adapter).__name__}")
        
        # Initialize embedding manager
        if isinstance(db_adapter, SQLiteAdapter):
            embedding_manager = EmbeddingManager(str(DB_PATH))
        else:
            # For PostgreSQL, we'll need to adapt the embedding manager
            embedding_manager = EmbeddingManager(db_adapter=db_adapter)
        logging.info("Embedding manager initialized")
        
        # Initialize learning-to-rank reranker
        if LearningToRankReranker:
            if isinstance(db_adapter, SQLiteAdapter):
                ltr_reranker = LearningToRankReranker(str(DB_PATH))
            else:
                ltr_reranker = LearningToRankReranker(db_adapter=db_adapter)
            logging.info("LTR reranker initialized")
        
        # Initialize job manager
        try:
            register_default_handlers()
            await job_manager.start()
            logging.info("Job manager initialized successfully")
        except Exception as e:
            logging.warning(f"Failed to initialize job manager: {e}")
            # Don't fail startup if Redis is not available
            
    except Exception as e:
        logging.error(f"Failed to initialize application: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global db_adapter
    
    # Shutdown job manager
    try:
        await job_manager.shutdown()
        logging.info("Job manager shutdown completed")
    except Exception as e:
        logging.error(f"Error during job manager shutdown: {e}")
    
    # Close database connections
    if db_adapter:
        await db_adapter.close()
        logging.info("Database connections closed")
    
    # Shutdown telemetry
    if TELEMETRY_AVAILABLE:
        shutdown_telemetry()
        logging.info("OpenTelemetry shutdown completed")

async def get_db():
    """Dependency to get database adapter."""
    if db_adapter is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    return db_adapter

class SearchRequest(BaseModel):
    q: str
    k: int = 5

class SemanticSearchRequest(BaseModel):
    q: str
    k: int = 5
    min_similarity: float = 0.3

class HybridSearchRequest(BaseModel):
    q: str
    k: int = 5
    rrf_k: int = 60
    min_similarity: float = 0.3

class ClickFeedbackRequest(BaseModel):
    query: str
    chunk_id: str
    position: int
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    dwell_time: Optional[float] = None

class SearchSessionRequest(BaseModel):
    session_id: Optional[str] = None
    user_id: Optional[str] = None

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "DocFoundry RAG API",
        "version": "0.2.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


@app.get("/metrics")
async def get_metrics(hours: int = 24):
    """Get API performance metrics."""
    if not metrics_collector:
        return {"error": "Monitoring not available"}
    
    try:
        summary_stats = metrics_collector.get_summary_stats(hours=hours)
        endpoint_stats = metrics_collector.get_endpoint_stats()
        
        return {
            "summary": summary_stats,
            "endpoints": endpoint_stats,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"Error retrieving metrics: {e}")
        return {"error": "Failed to retrieve metrics"}


@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with system metrics."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "version": "0.2.0",
        "components": {}
    }
    
    # Check database connectivity
    try:
        if db_adapter:
            # Test database connection
            await db_adapter.get_stats()
            health_status["components"]["database"] = {"status": "healthy"}
        else:
            health_status["components"]["database"] = {"status": "unavailable"}
    except Exception as e:
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check embedding manager
    try:
        if embedding_manager:
            health_status["components"]["embeddings"] = {"status": "healthy"}
        else:
            health_status["components"]["embeddings"] = {"status": "unavailable"}
    except Exception as e:
        health_status["components"]["embeddings"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Add system metrics if available
    if metrics_collector:
        try:
            recent_stats = metrics_collector.get_summary_stats(hours=1)
            health_status["system"] = recent_stats.get("system", {})
            health_status["performance"] = {
                "avg_response_time": recent_stats.get("requests", {}).get("avg_response_time", 0),
                "error_rate": recent_stats.get("requests", {}).get("error_rate", 0),
                "cache_hit_rate": recent_stats.get("cache", {}).get("hits", 0) / max(recent_stats.get("cache", {}).get("total_requests", 1), 1)
            }
        except Exception as e:
            logging.warning(f"Could not retrieve system metrics: {e}")
    
    return health_status


@app.get("/health")
def health():
    return {"ok": True, "time": datetime.datetime.utcnow().isoformat()+"Z"}

@app.post("/search")
async def search(req: SearchRequest, db: Union[PostgresAdapter, SQLiteAdapter] = Depends(get_db)):
    """Perform full-text search using the database adapter."""
    try:
        results = await db.search_fulltext(req.q, req.k)
        return {"results": results}
    except Exception as e:
        logging.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Search failed")

@app.post("/search/semantic")
async def semantic_search(req: SemanticSearchRequest, db: Union[PostgresAdapter, SQLiteAdapter] = Depends(get_db)):
    """Perform semantic search using embeddings with caching optimization"""
    if TELEMETRY_AVAILABLE:
        record_counter("search_requests_total", {"type": "semantic"})
    
    # Generate cache key for query
    cache_key = f"semantic:{hash(req.q)}:{req.k}:{req.min_similarity}"
    
    # Track search performance
    if track_search_performance and metrics_collector:
        async with track_search_performance(metrics_collector, req.q, "semantic") as search_metrics:
            try:
                # Check cache first (if available)
                cached_result = None
                if hasattr(db, 'get_cached_result'):
                    cached_result = await db.get_cached_result(cache_key)
                    if cached_result:
                        search_metrics.cache_hit = True
                        search_metrics.results_count = len(cached_result)
                        if TELEMETRY_AVAILABLE:
                            record_counter("cache_hits_total", {"type": "semantic"})
                        return {"results": cached_result, "search_type": "semantic", "cached": True}
                
                if TELEMETRY_AVAILABLE:
                    with trace_span("semantic_search") as span:
                        span.set_attribute("query", req.q)
                        span.set_attribute("k", req.k)
                        span.set_attribute("min_similarity", req.min_similarity)
                        
                        # Add query optimization for large result sets
                        optimized_k = min(req.k * 2, 100)  # Fetch more for better ranking
                        
                        # Generate embedding from query text
                        query_embedding = embedding_manager.generate_embedding(req.q)
                        
                        # Track database query time
                        db_start = time.time()
                        results = await db.search_semantic(query_embedding, optimized_k, req.min_similarity)
                        search_metrics.db_query_time = time.time() - db_start
                        
                        # Apply post-processing and limit to requested k
                        results = results[:req.k]
                        
                        search_metrics.results_count = len(results)
                        span.set_attribute("results_count", len(results))
                        record_histogram("search_results_count", len(results), {"type": "semantic"})
                        record_counter("search_requests_success", {"type": "semantic"})
                else:
                    optimized_k = min(req.k * 2, 100)
                    # Generate embedding from query text
                    query_embedding = embedding_manager.generate_embedding(req.q)
                    db_start = time.time()
                    results = await db.search_semantic(query_embedding, optimized_k, req.min_similarity)
                    search_metrics.db_query_time = time.time() - db_start
                    results = results[:req.k]
                    search_metrics.results_count = len(results)
                
                # Cache results for future requests
                if hasattr(db, 'cache_result'):
                    await db.cache_result(cache_key, results, ttl=3600)  # 1 hour TTL
                    
                return {"results": results, "search_type": "semantic", "cached": False}
            except Exception as e:
                if TELEMETRY_AVAILABLE:
                    record_counter("search_requests_error", {"type": "semantic", "error": str(type(e).__name__)})
                logging.error(f"Semantic search error: {e}")
                raise HTTPException(status_code=500, detail="Semantic search failed")
    else:
        # Fallback without monitoring
        try:
            # Check cache first (if available)
            cached_result = None
            if hasattr(db, 'get_cached_result'):
                cached_result = await db.get_cached_result(cache_key)
                if cached_result:
                    if TELEMETRY_AVAILABLE:
                        record_counter("cache_hits_total", {"type": "semantic"})
                    return {"results": cached_result, "search_type": "semantic", "cached": True}
            
            if TELEMETRY_AVAILABLE:
                with trace_span("semantic_search") as span:
                    span.set_attribute("query", req.q)
                    span.set_attribute("k", req.k)
                    span.set_attribute("min_similarity", req.min_similarity)
                    
                    # Add query optimization for large result sets
                    optimized_k = min(req.k * 2, 100)  # Fetch more for better ranking
                    # Generate embedding from query text
                    query_embedding = embedding_manager.generate_embedding(req.q)
                    results = await db.search_semantic(query_embedding, optimized_k, req.min_similarity)
                    
                    # Apply post-processing and limit to requested k
                    results = results[:req.k]
                    
                    span.set_attribute("results_count", len(results))
                    record_histogram("search_results_count", len(results), {"type": "semantic"})
                    record_counter("search_requests_success", {"type": "semantic"})
            else:
                optimized_k = min(req.k * 2, 100)
                # Generate embedding from query text
                query_embedding = embedding_manager.generate_embedding(req.q)
                results = await db.search_semantic(query_embedding, optimized_k, req.min_similarity)
                results = results[:req.k]
            
            # Cache results for future requests
            if hasattr(db, 'cache_result'):
                await db.cache_result(cache_key, results, ttl=3600)  # 1 hour TTL
                
            return {"results": results, "search_type": "semantic", "cached": False}
        except Exception as e:
            if TELEMETRY_AVAILABLE:
                record_counter("search_requests_error", {"type": "semantic", "error": str(type(e).__name__)})
            logging.error(f"Semantic search error: {e}")
            raise HTTPException(status_code=500, detail="Semantic search failed")

@app.post("/search/hybrid")
async def hybrid_search(req: HybridSearchRequest, db: Union[PostgresAdapter, SQLiteAdapter] = Depends(get_db)):
    """Perform hybrid search combining FTS and semantic search using RRF with optimization"""
    if TELEMETRY_AVAILABLE:
        record_counter("search_requests_total", {"type": "hybrid"})
    
    # Generate cache key for hybrid query
    cache_key = f"hybrid:{hash(req.q)}:{req.k}:{req.rrf_k}:{req.min_similarity}"
    
    # Track search performance
    if track_search_performance and metrics_collector:
        async with track_search_performance(metrics_collector, req.q, "hybrid") as search_metrics:
            try:
                # Check cache first
                cached_result = None
                if hasattr(db, 'get_cached_result'):
                    cached_result = await db.get_cached_result(cache_key)
                    if cached_result:
                        search_metrics.cache_hit = True
                        search_metrics.results_count = len(cached_result)
                        if TELEMETRY_AVAILABLE:
                            record_counter("cache_hits_total", {"type": "hybrid"})
                        return {"results": cached_result, "search_type": "hybrid_rrf", "cached": True}
                
                if TELEMETRY_AVAILABLE:
                    with trace_span("hybrid_search") as span:
                        span.set_attribute("query", req.q)
                        span.set_attribute("k", req.k)
                        span.set_attribute("rrf_k", req.rrf_k)
                        span.set_attribute("min_similarity", req.min_similarity)
                        
                        # Optimize hybrid search with parallel execution
                        start_time = time.time()
                        results = await db.search_hybrid(req.q, req.k, req.rrf_k, req.min_similarity)
                        search_time = time.time() - start_time
                        search_metrics.db_query_time = search_time
                        
                        span.set_attribute("results_count", len(results))
                        span.set_attribute("search_duration_ms", search_time * 1000)
                        record_histogram("search_results_count", len(results), {"type": "hybrid"})
                        record_histogram("search_duration_seconds", search_time, {"type": "hybrid"})
                        record_counter("search_requests_success", {"type": "hybrid"})
                else:
                    start_time = time.time()
                    results = await db.search_hybrid(req.q, req.k, req.rrf_k, req.min_similarity)
                    search_metrics.db_query_time = time.time() - start_time
                
                search_metrics.results_count = len(results)
                
                # Cache results with shorter TTL for hybrid (more dynamic)
                if hasattr(db, 'cache_result'):
                    await db.cache_result(cache_key, results, ttl=1800)  # 30 minutes TTL
                    
                return {"results": results, "search_type": "hybrid_rrf", "cached": False}
            except Exception as e:
                if TELEMETRY_AVAILABLE:
                    record_counter("search_requests_error", {"type": "hybrid", "error": str(type(e).__name__)})
                logging.error(f"Hybrid search error: {e}")
                raise HTTPException(status_code=500, detail="Hybrid search failed")
    else:
        # Fallback without monitoring
        try:
            # Check cache first
            cached_result = None
            if hasattr(db, 'get_cached_result'):
                cached_result = await db.get_cached_result(cache_key)
                if cached_result:
                    if TELEMETRY_AVAILABLE:
                        record_counter("cache_hits_total", {"type": "hybrid"})
                    return {"results": cached_result, "search_type": "hybrid_rrf", "cached": True}
            
            if TELEMETRY_AVAILABLE:
                with trace_span("hybrid_search") as span:
                    span.set_attribute("query", req.q)
                    span.set_attribute("k", req.k)
                    span.set_attribute("rrf_k", req.rrf_k)
                    span.set_attribute("min_similarity", req.min_similarity)
                    
                    # Optimize hybrid search with parallel execution
                    start_time = time.time()
                    results = await db.search_hybrid(req.q, req.k, req.rrf_k, req.min_similarity)
                    search_time = time.time() - start_time
                    
                    span.set_attribute("results_count", len(results))
                    span.set_attribute("search_duration_ms", search_time * 1000)
                    record_histogram("search_results_count", len(results), {"type": "hybrid"})
                    record_histogram("search_duration_seconds", search_time, {"type": "hybrid"})
                    record_counter("search_requests_success", {"type": "hybrid"})
            else:
                results = await db.search_hybrid(req.q, req.k, req.rrf_k, req.min_similarity)
            
            # Cache results with shorter TTL for hybrid (more dynamic)
            if hasattr(db, 'cache_result'):
                await db.cache_result(cache_key, results, ttl=1800)  # 30 minutes TTL
                
            return {"results": results, "search_type": "hybrid_rrf", "cached": False}
        except Exception as e:
            if TELEMETRY_AVAILABLE:
                record_counter("search_requests_error", {"type": "hybrid", "error": str(type(e).__name__)})
            logging.error(f"Hybrid search error: {e}")
            raise HTTPException(status_code=500, detail="Hybrid search failed")

# Streaming search endpoints
async def stream_search_results(results: List[dict], search_type: str) -> AsyncGenerator[str, None]:
    """Stream search results as Server-Sent Events."""
    # Send initial metadata
    yield f"data: {json.dumps({'type': 'metadata', 'search_type': search_type, 'total_results': len(results)})}\n\n"
    
    # Stream results one by one
    for i, result in enumerate(results):
        result_data = {
            'type': 'result',
            'index': i,
            'data': result
        }
        yield f"data: {json.dumps(result_data)}\n\n"
        
        # Add small delay to simulate progressive loading
        await asyncio.sleep(0.01)
    
    # Send completion signal
    yield f"data: {json.dumps({'type': 'complete', 'message': 'Search completed'})}\n\n"

@app.post("/search/semantic/stream")
async def semantic_search_stream(req: SemanticSearchRequest, db: Union[PostgresAdapter, SQLiteAdapter] = Depends(get_db)):
    """Perform semantic search with streaming results."""
    if TELEMETRY_AVAILABLE:
        record_counter("search_requests_total", {"type": "semantic_stream"})
    
    try:
        if TELEMETRY_AVAILABLE:
            with trace_span("semantic_search_stream") as span:
                span.set_attribute("query", req.q)
                span.set_attribute("k", req.k)
                span.set_attribute("min_similarity", req.min_similarity)
                
                results = await db.search_semantic(req.q, req.k, req.min_similarity)
                
                span.set_attribute("results_count", len(results))
                record_histogram("search_results_count", len(results), {"type": "semantic_stream"})
                record_counter("search_requests_success", {"type": "semantic_stream"})
        else:
            results = await db.search_semantic(req.q, req.k, req.min_similarity)
        
        return StreamingResponse(
            stream_search_results(results, "semantic"),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*"
            }
        )
    except Exception as e:
        if TELEMETRY_AVAILABLE:
            record_counter("search_requests_error", {"type": "semantic_stream", "error": str(type(e).__name__)})
        logging.error(f"Semantic search stream error: {e}")
        raise HTTPException(status_code=500, detail="Semantic search stream failed")

@app.post("/search/hybrid/stream")
async def hybrid_search_stream(req: HybridSearchRequest, db: Union[PostgresAdapter, SQLiteAdapter] = Depends(get_db)):
    """Perform hybrid search with streaming results."""
    if TELEMETRY_AVAILABLE:
        record_counter("search_requests_total", {"type": "hybrid_stream"})
    
    try:
        if TELEMETRY_AVAILABLE:
            with trace_span("hybrid_search_stream") as span:
                span.set_attribute("query", req.q)
                span.set_attribute("k", req.k)
                span.set_attribute("rrf_k", req.rrf_k)
                span.set_attribute("min_similarity", req.min_similarity)
                
                results = await db.search_hybrid(req.q, req.k, req.rrf_k, req.min_similarity)
                
                span.set_attribute("results_count", len(results))
                record_histogram("search_results_count", len(results), {"type": "hybrid_stream"})
                record_counter("search_requests_success", {"type": "hybrid_stream"})
        else:
            results = await db.search_hybrid(req.q, req.k, req.rrf_k, req.min_similarity)
        
        return StreamingResponse(
            stream_search_results(results, "hybrid"),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*"
            }
        )
    except Exception as e:
        if TELEMETRY_AVAILABLE:
            record_counter("search_requests_error", {"type": "hybrid_stream", "error": str(type(e).__name__)})
        logging.error(f"Hybrid search stream error: {e}")
        raise HTTPException(status_code=500, detail="Hybrid search stream failed")

@app.get("/doc")
def get_doc(path: str):
    p = BASE_DIR / path
    if not p.resolve().is_file() or not str(p).startswith(str(DOCS_DIR.resolve())):
        return JSONResponse({"error": "not found"}, status_code=404)
    return PlainTextResponse(p.read_text(encoding="utf-8", errors="ignore"))

class CaptureRequest(BaseModel):
    url: str
    title: str
    content: str = ""

class IngestRequest(BaseModel):
    source_name: Optional[str] = None
    urls: Optional[List[str]] = None
    reindex: bool = False

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str = "Job enqueued successfully"

def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:80] or "untitled"

@app.post("/capture")
def capture(req: CaptureRequest):
    slug = slugify(req.title or req.url)
    path = DOCS_DIR / "research" / f"{slug}.md"
    fm = f"---\ntitle: {req.title}\nsource_url: {req.url}\ncaptured_at: {datetime.datetime.utcnow().isoformat()}Z\n---\n\n"
    path.write_text(fm + req.content, encoding="utf-8")
    return {"saved": str(path.relative_to(BASE_DIR))}

@app.post("/ingest", response_model=JobResponse)
async def ingest(req: IngestRequest):
    """Enqueue ingestion job for a source or URLs"""
    if not req.source_name and not req.urls:
        raise HTTPException(status_code=400, detail="Either source_name or urls must be provided")
    
    if TELEMETRY_AVAILABLE:
        record_counter("ingest_jobs_total")
    
    try:
        if TELEMETRY_AVAILABLE:
            with trace_span("ingest_job_enqueue") as span:
                span.set_attribute("source_name", req.source_name or "unknown")
                span.set_attribute("reindex", req.reindex)
                if req.urls:
                    span.set_attribute("url_count", len(req.urls))
                    record_histogram("ingest_url_count", len(req.urls))
                
                # Prepare job parameters
                job_params = {
                    "source_name": req.source_name,
                    "urls": req.urls or [],
                    "reindex": req.reindex
                }
                
                # Enqueue crawl job
                job_id = await job_manager.enqueue_job("crawl_source", job_params)
                span.set_attribute("job_id", job_id)
                
                # If reindex is requested, enqueue reindex job as well
                if req.reindex:
                    reindex_params = {"source_filter": req.source_name}
                    await job_manager.enqueue_job("reindex", reindex_params)
                
                record_counter("ingest_jobs_enqueued")
        else:
            # Prepare job parameters
            job_params = {
                "source_name": req.source_name,
                "urls": req.urls or [],
                "reindex": req.reindex
            }
            
            # Enqueue crawl job
            job_id = await job_manager.enqueue_job("crawl_source", job_params)
            
            # If reindex is requested, enqueue reindex job as well
            if req.reindex:
                reindex_params = {"source_filter": req.source_name}
                await job_manager.enqueue_job("reindex", reindex_params)
        
        return JobResponse(
            job_id=job_id,
            status="queued",
            message="Ingestion job enqueued successfully"
        )
    except Exception as e:
        if TELEMETRY_AVAILABLE:
            record_counter("ingest_jobs_error", {"error": str(type(e).__name__)})
        raise HTTPException(status_code=500, detail=f"Failed to enqueue job: {str(e)}")

@app.post("/feedback/click")
async def log_click_feedback(req: ClickFeedbackRequest, db: Union[PostgresAdapter, SQLiteAdapter] = Depends(get_db)):
    """Log click feedback for learning-to-rank improvement"""
    if TELEMETRY_AVAILABLE:
        record_counter("feedback_clicks_total")
    
    try:
        if TELEMETRY_AVAILABLE:
            with trace_span("log_click_feedback") as span:
                span.set_attribute("query", req.query)
                span.set_attribute("chunk_id", req.chunk_id)
                span.set_attribute("position", req.position)
                if req.dwell_time:
                    span.set_attribute("dwell_time", req.dwell_time)
                    record_histogram("feedback_dwell_time", req.dwell_time)
                
                # Generate session ID if not provided
                session_id = req.session_id or str(uuid.uuid4())
                
                # Log the click event using database adapter
                await db.record_click_feedback(
                    query=req.query,
                    chunk_id=req.chunk_id,
                    position=req.position,
                    session_id=session_id,
                    user_id=req.user_id,
                    dwell_time=req.dwell_time
                )
                
                span.set_attribute("session_id", session_id)
                record_counter("feedback_clicks_success")
        else:
            # Generate session ID if not provided
            session_id = req.session_id or str(uuid.uuid4())
            
            # Log the click event using database adapter
            await db.record_click_feedback(
                query=req.query,
                chunk_id=req.chunk_id,
                position=req.position,
                session_id=session_id,
                user_id=req.user_id,
                dwell_time=req.dwell_time
            )
        
        return {
            "success": True,
            "message": "Click feedback logged successfully",
            "session_id": session_id
        }
    except Exception as e:
        if TELEMETRY_AVAILABLE:
            record_counter("feedback_clicks_error", {"error": str(type(e).__name__)})
        logging.error(f"Failed to log click feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to log click feedback: {str(e)}")

@app.post("/feedback/session")
def create_search_session(req: SearchSessionRequest):
    """Create or retrieve a search session for tracking user interactions"""
    try:
        # Generate session ID if not provided
        session_id = req.session_id or str(uuid.uuid4())
        
        return {
            "session_id": session_id,
            "user_id": req.user_id,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z"
        }
    except Exception as e:
        logging.error(f"Failed to create search session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create search session: {str(e)}")

@app.get("/feedback/stats")
async def get_feedback_stats(query: Optional[str] = None, days: int = 30, db: Union[PostgresAdapter, SQLiteAdapter] = Depends(get_db)):
    """Get click feedback statistics for analysis"""
    try:
        stats = await db.get_stats()
        
        return {
            "database_stats": stats,
            "query_filter": query,
            "days_range": days
        }
    except Exception as e:
        logging.error(f"Failed to get feedback stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get feedback stats: {str(e)}")

@app.post("/feedback/update-model")
def update_ltr_model(learning_rate: float = 0.01):
    """Manually trigger learning-to-rank model update"""
    if not ltr_reranker:
        raise HTTPException(status_code=503, detail="Learning-to-rank system not available")
    
    try:
        ltr_reranker.update_model(learning_rate=learning_rate)
        
        return {
            "success": True,
            "message": "Learning-to-rank model updated successfully",
            "learning_rate": learning_rate,
            "updated_at": datetime.datetime.utcnow().isoformat() + "Z"
        }
    except Exception as e:
        logging.error(f"Failed to update LTR model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update LTR model: {str(e)}")

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status and logs"""
    try:
        job_record = await job_manager.get_job_status(job_id)
        if not job_record:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {
            "id": job_record.id,
            "status": job_record.status.value,
            "type": job_record.type,
            "created_at": job_record.created_at.isoformat(),
            "started_at": job_record.started_at.isoformat() if job_record.started_at else None,
            "completed_at": job_record.completed_at.isoformat() if job_record.completed_at else None,
            "logs": job_record.logs,
            "error": job_record.error,
            "result": job_record.result
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@app.get("/jobs")
async def list_jobs(status: Optional[str] = None, limit: int = 100):
    """List jobs with optional status filter"""
    try:
        job_status = None
        if status:
            try:
                job_status = JobStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        jobs = await job_manager.list_jobs(job_status, limit)
        
        return {
            "jobs": [
                {
                    "id": job.id,
                    "type": job.type,
                    "status": job.status.value,
                    "created_at": job.created_at.isoformat(),
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "error": job.error
                }
                for job in jobs
            ],
            "total": len(jobs)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")

# Application lifecycle events
# Job manager initialization is now handled in the main startup event above
