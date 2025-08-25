from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
import sqlite3, pathlib, json, datetime, re

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "docfoundry.db"
DOCS_DIR = BASE_DIR / "docs"

app = FastAPI(title="DocFoundry RAG API", version="0.1.0")

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

class SearchRequest(BaseModel):
    q: str
    k: int = 5

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
