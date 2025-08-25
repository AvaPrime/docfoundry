# Minimal placeholder for an MCP-like server (JSON over stdout).
# Real MCP uses JSON-RPC over stdio/sockets; this is just a sketch.

import sys, json, pathlib, sqlite3

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "docfoundry.db"

def list_resources(limit=200):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT path, title FROM documents ORDER BY path LIMIT ?", (limit,))
    return [{"uri": f"docfoundry://{row[0]}", "title": row[1]} for row in cur.fetchall()]

def read_resource(uri: str):
    path = uri.replace("docfoundry://", "")
    p = BASE_DIR / path
    try:
        return (BASE_DIR / path).read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return f"ERROR: {e}"

def main():
    if len(sys.argv) == 1:
        print(json.dumps({"resources": list_resources()}, indent=2))
    elif sys.argv[1] == "read" and len(sys.argv) > 2:
        print(read_resource(sys.argv[2]))
    else:
        print("Usage: python server/mcp_server.py [read docfoundry://<path>]")

if __name__ == "__main__":
    main()
