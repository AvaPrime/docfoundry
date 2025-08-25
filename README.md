# DocFoundry (Scaffold)

A lightweight **Docs Intelligence System** that discovers, ingests, normalizes, indexes,
and serves third‑party docs alongside your own — then exposes them to agents, IDEs, and browsers.

## What’s here
- **MkDocs site** for humans: `mkdocs.yml`, `docs/`
- **Source registry** for crawl rules: `sources/*.yaml`
- **Pipelines**: repo ingest (stub), HTML crawler → Markdown, feed ingest (stub) in `pipelines/`
- **Indexer**: builds a SQLite FTS5 index of Markdown chunks (optional embeddings hook) in `indexer/`
- **RAG API** (FastAPI): `/search`, `/doc`, `/capture` in `server/`
- **VS Code extension**: quick search and open results (`extensions/vscode`)
- **Chrome extension**: capture current page into research list (`browser/chrome-extension`)
- **Makefile** for common tasks

> Note: This scaffold favors **working locally with SQLite + FTS5** first.
> You can upgrade to **Postgres + pgvector** later by swapping the storage layer in `indexer/` and `server/`.

## Quick start
```bash
# 1) Python deps (use a venv in your environment)
pip install -r requirements.txt

# 2) Build the initial index from the docs/ folder
python indexer/build_index.py

# 3) Run the API
uvicorn server.rag_api:app --reload --port 8001

# 4) Preview docs (optional)
mkdocs serve -a 127.0.0.1:8000
```

## Typical flow
1. Add or edit a source file in `sources/` (e.g. `openrouter.yaml`).
2. Crawl HTML → Markdown: `python pipelines/html_ingest.py sources/openrouter.yaml`
3. Re-index: `python indexer/build_index.py`
4. Search: `curl -X POST :8001/search -H 'content-type: application/json' -d '{"q":"provider routing"}'`
5. Use VS Code extension command **DocFoundry: Search Docs** to query from the editor.
6. Use the Chrome extension to capture a web page into `docs/research/` with frontmatter.

## Upgrade path
- Swap SQLite for **Postgres + pgvector** and move embeddings to `vector` column.
- Add **Temporal** workflows for scheduled polling & webhooks.
- Implement a real **MCP server** on stdio / sockets to serve resources to MCP clients.
- Add **license & robots** policy checks in pipelines.
