from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from typing import List, Optional
import sqlite3, pathlib, json, datetime, re, logging
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "indexer"))
from embeddings import EmbeddingManager
from .jobs import job_manager, register_default_handlers, JobStatus

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "docfoundry.db"
DOCS_DIR = BASE_DIR / "docs"

app = FastAPI(title="DocFoundry RAG API", version="0.1.0")

# Initialize embedding manager for semantic search
embedding_manager = None
try:
    embedding_manager = EmbeddingManager(str(DB_PATH))
except Exception as e:
    logging.warning(f"Failed to initialize embedding manager: {e}")

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

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

@app.get("/health")
def health():
    return {"ok": True, "time": datetime.datetime.utcnow().isoformat()+"Z"}

@app.post("/search")
def search(req: SearchRequest):
    conn = db()
    try:
        rows = conn.execute("""
            SELECT d.path, d.title, d.source_url, c.heading, c.anchor,
                   snippet(chunks_fts, 0, '<b>', '</b>', '…', 8) AS snippet
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN documents d ON d.id = c.document_id
            WHERE chunks_fts MATCH ?
            LIMIT ?
        """, (req.q, req.k)).fetchall()
    except Exception:
        rows = conn.execute("""
            SELECT d.path, d.title, d.source_url, c.heading, c.anchor, substr(c.text, 1, 240) as snippet
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.text LIKE ?
            LIMIT ?
        """, (f"%{req.q}%", req.k)).fetchall()
    results = [dict(r) for r in rows]
    return {"results": results}

@app.post("/search/semantic")
def semantic_search(req: SemanticSearchRequest):
    """Perform semantic search using embeddings"""
    if not embedding_manager:
        return JSONResponse({"error": "Semantic search not available - embedding manager not initialized"}, status_code=503)
    
    try:
        results = embedding_manager.semantic_search(req.q, req.k, req.min_similarity)
        formatted_results = []
        for chunk_data, similarity in results:
            formatted_results.append({
                "path": chunk_data["path"],
                "title": chunk_data["title"],
                "source_url": chunk_data["source_url"],
                "heading": chunk_data["heading"],
                "anchor": chunk_data["anchor"],
                "snippet": chunk_data["text"][:240] + "..." if len(chunk_data["text"]) > 240 else chunk_data["text"],
                "similarity": round(similarity, 3)
            })
        return {"results": formatted_results, "search_type": "semantic"}
    except Exception as e:
        return JSONResponse({"error": f"Semantic search failed: {str(e)}"}, status_code=500)

@app.post("/search/hybrid")
def hybrid_search(req: HybridSearchRequest):
    """Perform hybrid search combining FTS and semantic search using RRF"""
    if not embedding_manager:
        return JSONResponse({"error": "Hybrid search not available - embedding manager not initialized"}, status_code=503)
    
    try:
        results = embedding_manager.hybrid_search(req.q, req.k, req.rrf_k, req.min_similarity)
        formatted_results = []
        for chunk_data, rrf_score in results:
            formatted_results.append({
                "path": chunk_data["path"],
                "title": chunk_data["title"],
                "source_url": chunk_data["source_url"],
                "heading": chunk_data["heading"],
                "anchor": chunk_data["anchor"],
                "snippet": chunk_data["text"][:240] + "..." if len(chunk_data["text"]) > 240 else chunk_data["text"],
                "rrf_score": round(rrf_score, 4)
            })
        return {"results": formatted_results, "search_type": "hybrid_rrf"}
    except Exception as e:
        return JSONResponse({"error": f"Hybrid search failed: {str(e)}"}, status_code=500)

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
    
    try:
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
        raise HTTPException(status_code=500, detail=f"Failed to enqueue job: {str(e)}")

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
@app.on_event("startup")
async def startup_event():
    """Initialize job manager on startup"""
    try:
        await job_manager.initialize()
        register_default_handlers()
        logging.info("Job manager initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize job manager: {e}")
        # Don't fail startup if Redis is not available

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown job manager on app shutdown"""
    try:
        await job_manager.shutdown()
        logging.info("Job manager shutdown completed")
    except Exception as e:
        logging.error(f"Error during job manager shutdown: {e}")
