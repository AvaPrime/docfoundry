# Stub: ingest RSS/Atom feeds (release notes, changelogs) into research notes.
import sys, pathlib, datetime, hashlib
try:
    import feedparser
except Exception:
    print("Missing dependency: feedparser. Install with `pip install feedparser`."); raise

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
RESEARCH_DIR = BASE_DIR / "docs" / "research"

def slugify(s: str) -> str:
    import re
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s

def main(url: str):
    d = feedparser.parse(url)
    for e in d.entries:
        title = e.get("title", "untitled")
        link = e.get("link", "")
        date = e.get("updated", e.get("published", ""))
        content = e.get("summary", "")
        slug = slugify(title)[:60]
        path = RESEARCH_DIR / f"{slug}.md"
        fm = f"---\ntitle: {title}\nsource_url: {link}\ncaptured_at: {datetime.datetime.utcnow().isoformat()}Z\n---\n\n"
        path.write_text(fm + content, encoding="utf-8")
        print(f"Wrote {path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipelines/feed_ingest.py <feed_url>")
        sys.exit(1)
    main(sys.argv[1])
