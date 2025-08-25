# Stub: if upstream docs are in a repo, prefer pulling Markdown directly.
# This script would typically use GitPython to clone or update a submodule and
# copy whitelisted paths into docs/vendors/<source_id>.

import sys, pathlib, shutil, os, subprocess, yaml

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DOCS_DIR = BASE_DIR / "docs" / "vendors"

def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    src_id = cfg["id"]
    repo = cfg.get("repo")
    if not repo:
        print("No repo specified in config.")
        return
    vendor_dir = DOCS_DIR / src_id
    vendor_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = BASE_DIR / ".cache" / src_id
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    if not (cache_dir / ".git").exists():
        subprocess.check_call(["git", "clone", "--depth", "1", repo, str(cache_dir)])
    else:
        subprocess.check_call(["git", "-C", str(cache_dir), "pull", "--ff-only"])
    for p in cache_dir.rglob("*.md"):
        rel = p.relative_to(cache_dir)
        out = vendor_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        print(f"Copied {rel}")
    print("Repo ingest complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipelines/repo_ingest.py sources/<source>.yaml")
        sys.exit(1)
    main(sys.argv[1])
