# Builds a SQLite FTS5 index from docs/*.md.

import os, re, sys, pathlib, hashlib, sqlite3

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DOCS_DIR = BASE_DIR / "docs"
DB_PATH = BASE_DIR / "data" / "docfoundry.db"

def read_markdown(p: pathlib.Path):
    text = p.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    title = m.group(1).strip() if m else p.stem
    fm = {}
    if text.startswith('---'):
        end = text.find('---', 3)
        if end != -1:
            head = text[3:end].strip()
            for line in head.splitlines():
                if ':' in line:
                    k,v = line.split(':', 1)
                    fm[k.strip()] = v.strip().strip("'\"")
            text = text[end+3:]
    return title, fm, text

def chunk_markdown(md: str):
    parts = re.split(r"(?m)^(##+)\s+(.*)$", md)
    chunks = []
    if parts:
        preamble = parts[0].strip()
        if preamble:
            chunks.append(("Introduction", "introduction", preamble))
        for i in range(1, len(parts), 3):
            heading = parts[i+1].strip()
            block = parts[i+2].strip()
            anchor = re.sub(r'[^a-z0-9]+', '-', heading.lower()).strip('-')
            paragraphs = re.split(r"\n\s*\n", block)
            cur=""
            for para in paragraphs:
                if len(cur)+len(para) > 1000:
                    if cur.strip():
                        chunks.append((heading, anchor, cur.strip()))
                        cur=""
                cur += ("\n\n" if cur else "") + para
            if cur.strip():
                chunks.append((heading, anchor, cur.strip()))
    return chunks

def ensure_schema(conn: sqlite3.Connection):
    schema = (BASE_DIR / "indexer" / "schema.sql").read_text(encoding="utf-8")
    conn.executescript(schema)

def index_file(conn, path: pathlib.Path):
    rel = path.relative_to(BASE_DIR).as_posix()
    title, fm, md = read_markdown(path)
    source_url = fm.get("source_url", "")
    captured_at = fm.get("captured_at", "")
    h = hashlib.sha256(md.encode("utf-8")).hexdigest()

    cur = conn.execute("SELECT id, hash FROM documents WHERE path=?", (rel,))
    row = cur.fetchone()
    if row and row[1] == h:
        return 0

    if row:
        doc_id = row[0]
        conn.execute("DELETE FROM chunks WHERE document_id=?", (doc_id,))
        conn.execute("UPDATE documents SET title=?, source_url=?, captured_at=?, hash=? WHERE id=?",
                     (title, source_url, captured_at, h, doc_id))
    else:
        conn.execute("INSERT INTO documents(path, title, source_url, captured_at, hash) VALUES(?,?,?,?,?)",
                     (rel, title, source_url, captured_at, h))
        doc_id = conn.execute("SELECT id FROM documents WHERE path=?", (rel,)).fetchone()[0]

    chunks = chunk_markdown(md)
    for heading, anchor, text in chunks:
        conn.execute("INSERT INTO chunks(document_id, heading, anchor, text) VALUES(?,?,?,?)",
                     (doc_id, heading, anchor, text))
        conn.execute("INSERT INTO chunks_fts(rowid, text, heading, anchor, path) VALUES(last_insert_rowid(), ?, ?, ?, ?)",
                     (text, heading, anchor, rel))
    return 1

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA case_sensitive_like=OFF")
    ensure_schema(conn)
    conn.execute("DELETE FROM chunks_fts")
    added = 0
    for p in DOCS_DIR.rglob("*.md"):
        added += index_file(conn, p)
    conn.commit()
    print(f"Indexed {added} files into {DB_PATH}")

if __name__ == "__main__":
    main()
