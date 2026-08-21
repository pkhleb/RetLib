import sqlite3
from pathlib import Path
from contextlib import contextmanager
from retlib.parser import ParseResult

DEFAULT_DB_PATH = ".retlib/index.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT    NOT NULL UNIQUE,
    checksum    TEXT    NOT NULL,
    last_indexed TEXT   NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS symbols (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id         INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    name            TEXT    NOT NULL,
    qualified_name  TEXT    NOT NULL UNIQUE,
    kind            TEXT    NOT NULL,
    start_line      INTEGER NOT NULL,
    end_line        INTEGER NOT NULL,
    source          TEXT    NOT NULL,
    docstring       TEXT,
    embedding       BLOB
);

CREATE TABLE IF NOT EXISTS edges (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_qualified_name TEXT NOT NULL,
    target_qualified_name TEXT NOT NULL,
    kind            TEXT NOT NULL,
    UNIQUE(source_qualified_name, target_qualified_name, kind)
);

CREATE INDEX IF NOT EXISTS idx_symbols_qualified_name ON symbols(qualified_name);
CREATE INDEX IF NOT EXISTS idx_symbols_file_id        ON symbols(file_id);
CREATE INDEX IF NOT EXISTS idx_symbols_kind           ON symbols(kind);
CREATE INDEX IF NOT EXISTS idx_edges_source           ON edges(source_qualified_name);
CREATE INDEX IF NOT EXISTS idx_edges_target           ON edges(target_qualified_name);
CREATE INDEX IF NOT EXISTS idx_edges_kind             ON edges(kind);
"""


def _db_path(path: str | None) -> Path:
    return Path(path) if path else Path(DEFAULT_DB_PATH)


@contextmanager
def _connect(db_path: str | None):
    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str | None = None):
    """Create the database and schema if they don't exist."""
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)


def get_file_checksum(file_path: str, db_path: str | None = None) -> str | None:
    """Return the stored checksum for a file, or None if not indexed."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT checksum FROM files WHERE path = ?", (file_path,)
        ).fetchone()
        return row[0] if row else None


def index_result(result: ParseResult, db_path: str | None = None):
    """
    Persist a ParseResult into the database.
    If the file is already indexed with the same checksum, skip it.
    If the checksum has changed, delete and replace.
    """
    with _connect(db_path) as conn:
        existing = conn.execute(
            "SELECT id, checksum FROM files WHERE path = ?", (result.file_path,)
        ).fetchone()

        if existing:
            file_id, stored_checksum = existing
            if stored_checksum == result.checksum:
                return  # unchanged — nothing to do

            # file changed — delete symbols and edges, then reinsert
            # edges are deleted via ON DELETE CASCADE from symbols → files
            conn.execute("DELETE FROM files WHERE id = ?", (file_id,))

        # insert file record
        cursor = conn.execute(
            "INSERT INTO files (path, checksum) VALUES (?, ?)",
            (result.file_path, result.checksum),
        )
        file_id = cursor.lastrowid

        # insert symbols
        conn.executemany(
            """
            INSERT OR IGNORE INTO symbols
                (file_id, name, qualified_name, kind, start_line, end_line, source, docstring)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    file_id,
                    s.name,
                    s.qualified_name,
                    s.kind,
                    s.start_line,
                    s.end_line,
                    s.source,
                    s.docstring,
                )
                for s in result.symbols
            ],
        )

        # insert edges — ignore duplicates across files
        conn.executemany(
            """
            INSERT OR IGNORE INTO edges
                (source_qualified_name, target_qualified_name, kind)
            VALUES (?, ?, ?)
            """,
            [
                (e.source_qualified_name, e.target_qualified_name, e.kind)
                for e in result.edges
            ],
        )


def index_results(results: list[ParseResult], db_path: str | None = None):
    """Persist a list of ParseResults, skipping unchanged files."""
    for result in results:
        index_result(result, db_path)


def get_symbol(qualified_name: str, db_path: str | None = None) -> dict | None:
    """Fetch a single symbol by qualified name."""
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT qualified_name, name, kind, start_line, end_line, source, docstring, f.path
            FROM symbols s
            JOIN files f ON s.file_id = f.id
            WHERE s.qualified_name = ?
            """,
            (qualified_name,),
        ).fetchone()
        if not row:
            return None
        return {
            "qualified_name": row[0],
            "name": row[1],
            "kind": row[2],
            "start_line": row[3],
            "end_line": row[4],
            "source": row[5],
            "docstring": row[6],
            "file_path": row[7],
        }


def get_neighbors(
    qualified_name: str,
    edge_kinds: list[str] | None = None,
    direction: str = "out",
    db_path: str | None = None,
) -> list[dict]:
    """
    Get neighboring symbols via edges.

    direction:
        'out'  — edges where source = qualified_name (e.g. CALLS, IMPORTS)
        'in'   — edges where target = qualified_name (e.g. CALLED_BY)
        'both' — both directions
    edge_kinds:
        filter to specific edge types, e.g. ['CALLS', 'IMPORTS']
        None means all kinds
    """
    kind_filter = ""
    params: list = []

    if direction == "out":
        base = "SELECT target_qualified_name AS neighbor, kind FROM edges WHERE source_qualified_name = ?"
        params.append(qualified_name)
    elif direction == "in":
        base = "SELECT source_qualified_name AS neighbor, kind FROM edges WHERE target_qualified_name = ?"
        params.append(qualified_name)
    else:  # both
        base = """
            SELECT target_qualified_name AS neighbor, kind FROM edges WHERE source_qualified_name = ?
            UNION
            SELECT source_qualified_name AS neighbor, kind FROM edges WHERE target_qualified_name = ?
        """
        params.extend([qualified_name, qualified_name])

    if edge_kinds:
        placeholders = ",".join("?" * len(edge_kinds))
        kind_filter = f" AND kind IN ({placeholders})"
        params.extend(edge_kinds)

    with _connect(db_path) as conn:
        rows = conn.execute(base + kind_filter, params).fetchall()
        return [{"qualified_name": r[0], "edge_kind": r[1]} for r in rows]


def search_symbols(
    name: str,
    kind: str | None = None,
    db_path: str | None = None,
) -> list[dict]:
    """Simple name-based symbol search (substring match)."""
    query = "SELECT qualified_name, name, kind, f.path FROM symbols s JOIN files f ON s.file_id = f.id WHERE s.name LIKE ?"
    params: list = [f"%{name}%"]
    if kind:
        query += " AND s.kind = ?"
        params.append(kind)
    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
        return [
            {"qualified_name": r[0], "name": r[1], "kind": r[2], "file_path": r[3]}
            for r in rows
        ]


def stats(db_path: str | None = None) -> dict:
    """Return basic counts for observability."""
    with _connect(db_path) as conn:
        files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        symbols = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
        edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        by_kind = conn.execute(
            "SELECT kind, COUNT(*) FROM symbols GROUP BY kind"
        ).fetchall()
        edge_by_kind = conn.execute(
            "SELECT kind, COUNT(*) FROM edges GROUP BY kind"
        ).fetchall()
        return {
            "files": files,
            "symbols": symbols,
            "edges": edges,
            "symbols_by_kind": dict(by_kind),
            "edges_by_kind": dict(edge_by_kind),
        }
