# Focused HTML to Markdown crawler driven by a single source YAML.
# Writes Markdown into docs/vendors/<source_id>/... with provenance frontmatter.

import sys, os, re, urllib.parse, pathlib, json, datetime
from typing import Dict, Any, Set, List

import yaml, requests
from bs4 import BeautifulSoup
from trafilatura import extract

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DOCS_DIR = BASE_DIR / "docs" / "vendors"

def slugify_path(url_path: str) -> str:
    path = url_path
    if path.endswith("/"):
        path += "index"
    path = re.sub(r"[^a-zA-Z0-9/_-]", "-", path).strip("/")
    if not path:
        path = "index"
    return path + ".md"

def within_rules(url: str, home: str, allow_globs: List[str], deny_globs: List[str]) -> bool:
    from fnmatch import fnmatch
    parsed = urllib.parse.urlparse(url)
    if not url.startswith(home):
        return False
    path = parsed.path or "/"
    if any(fnmatch(path, pat.replace("**", "*")) for pat in deny_globs):
        return False
    if not allow_globs:
        return True
    return any(fnmatch(path, pat.replace("**", "*")) for pat in allow_globs)

def discover_links(start_url: str, html: str, home: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.select("a[href]"):
        href = urllib.parse.urljoin(start_url, a["href"])
        if href.startswith(home):
            out.append(href.split("#")[0])
    seen=set(); res=[]
    for h in out:
        if h not in seen:
            seen.add(h); res.append(h)
    return res

def write_markdown(out_dir: pathlib.Path, url: str, md: str, meta: Dict[str, Any]):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = slugify_path(urllib.parse.urlparse(url).path)
    fp = out_dir / path
    front = "---\n" + "\n".join(f"{k}: {json.dumps(v)}" for k,v in meta.items()) + "\n---\n\n"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(front + md, encoding="utf-8")
    return fp

def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    src_id = cfg["id"]
    home = cfg["home"].rstrip("/")
    allow = cfg.get("crawl", {}).get("allow", [])
    deny = cfg.get("crawl", {}).get("deny", [])
    seeds = cfg.get("crawl", {}).get("seeds", [home])

    out_dir = DOCS_DIR / src_id
    out_dir.mkdir(parents=True, exist_ok=True)

    q = list(seeds)
    seen: Set[str] = set()

    budget_pages = int(os.environ.get("CRAWL_BUDGET_PAGES", "200"))
    timeout = 20

    while q and len(seen) < budget_pages:
        url = q.pop(0)
        if url in seen:
            continue
        seen.add(url)

        if not within_rules(url, home, allow, deny):
            continue

        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "DocFoundry/0.1"})
            r.raise_for_status()
            html = r.text
        except Exception as e:
            print(f"Skip {url}: {e}", file=sys.stderr)
            continue

        md = extract(html, output="markdown", include_links=True, include_tables=True)
        if not md or len(md.strip()) < 40:
            soup = BeautifulSoup(html, "html.parser")
            md = soup.get_text("\n")

        meta = {
            "source_url": url,
            "captured_at": datetime.datetime.utcnow().isoformat() + "Z",
            "source_id": src_id,
        }
        fp = write_markdown(out_dir, url, md, meta)
        print(f"Wrote {fp.relative_to(BASE_DIR)}")

        for link in discover_links(url, html, home):
            if link not in seen:
                q.append(link)

    print(f"Done. Pages seen: {len(seen)}; Output: {out_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipelines/html_ingest.py sources/<source>.yaml", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
