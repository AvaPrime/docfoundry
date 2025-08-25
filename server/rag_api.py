from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
import sqlite3, pathlib, json, datetime, re, logging
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "indexer"))
from embeddings import EmbeddingManager

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
    semantic_weight: float = 0.7
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
    """Perform hybrid search combining FTS and semantic search"""
    if not embedding_manager:
        return JSONResponse({"error": "Hybrid search not available - embedding manager not initialized"}, status_code=503)
    
    try:
        results = embedding_manager.hybrid_search(req.q, req.k, req.semantic_weight)
        formatted_results = []
        for chunk_data, score in results:
            formatted_results.append({
                "path": chunk_data["path"],
                "title": chunk_data["title"],
                "source_url": chunk_data["source_url"],
                "heading": chunk_data["heading"],
                "anchor": chunk_data["anchor"],
                "snippet": chunk_data["text"][:240] + "..." if len(chunk_data["text"]) > 240 else chunk_data["text"],
                "score": round(score, 3)
            })
        return {"results": formatted_results, "search_type": "hybrid"}
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
