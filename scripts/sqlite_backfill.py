import sqlite3, os

DB_PATH = os.environ.get("SQLITE_PATH", "docfoundry.db")
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

def has_column(table, column):
    c.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in c.fetchall())

def add_col_if_missing(table, column, col_def):
    if not has_column(table, column):
        c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}")

# Align with verification expectations
add_col_if_missing("documents", "etag", "TEXT")
add_col_if_missing("documents", "last_modified", "TEXT")
add_col_if_missing("documents", "last_crawled", "TEXT")

conn.commit()
conn.close()
print("✅ SQLite schema backfill complete")