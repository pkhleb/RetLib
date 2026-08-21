"""
index.py — BM25 lexical search and embedding-based vector search over symbols.

Embedding model: sentence-transformers/all-MiniLM-L6-v2 (384 dims, CPU-friendly).
TODO: consider swapping to a code-specific model (e.g. voyage-code-2, cohere-embed)
      for better semantic signal on code retrieval tasks.

Each symbol is embedded as:
    "{qualified_name} {docstring} {source}"
This captures name, intent, and implementation in a single vector.

Storage: embeddings stored as float32 blobs in SQLite, loaded into memory at
search time for cosine similarity. Linear scan is fine for single-codebase scale.
For larger corpora (10k+ symbols), swap to faiss ANN index.
"""

import sqlite3
import struct
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# TODO: swap this model for a code-specific one once baseline is established
#       candidates: nomic-embed-text, voyage-code-2
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model — only downloaded on first use."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _symbol_text(row: tuple) -> str:
    """
    Build the text to embed for a symbol.
    row = (qualified_name, docstring, source)
    """
    qualified_name, docstring, source = row
    parts = [qualified_name]
    if docstring:
        parts.append(docstring)
    if source:
        parts.append(source)
    return " ".join(parts)


def _encode_embedding(vec: np.ndarray) -> bytes:
    """Pack a float32 numpy array into bytes for SQLite storage."""
    return struct.pack(f"{len(vec)}f", *vec)


def _decode_embedding(blob: bytes) -> np.ndarray:
    """Unpack bytes from SQLite into a float32 numpy array."""
    n = len(blob) // 4
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ── embedding index ───────────────────────────────────────────────────────────

def build_embedding_index(db_path: str, force: bool = False):
    """
    Generate and store embeddings for all symbols that don't have one yet.
    Set force=True to regenerate all embeddings (e.g. after switching models).
    """
    conn = _connect(db_path)
    try:
        if force:
            conn.execute("UPDATE symbols SET embedding = NULL")
            conn.commit()

        rows = conn.execute(
            """
            SELECT id, qualified_name, docstring, source
            FROM symbols
            WHERE embedding IS NULL
            """
        ).fetchall()

        if not rows:
            print("[index] all symbols already embedded, nothing to do")
            return

        print(f"[index] embedding {len(rows)} symbols...")
        model = _get_model()

        ids = [r[0] for r in rows]
        texts = [_symbol_text(r[1:]) for r in rows]

        # batch encode — sentence-transformers handles batching internally
        vecs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        conn.executemany(
            "UPDATE symbols SET embedding = ? WHERE id = ?",
            [(_encode_embedding(vec), id_) for vec, id_ in zip(vecs, ids)],
        )
        conn.commit()
        print(f"[index] done — {len(rows)} embeddings stored")
    finally:
        conn.close()


def vector_search(
    query: str,
    db_path: str,
    top_k: int = 10,
    kind: str | None = None,
) -> list[dict]:
    """
    Embed the query and return the top_k most similar symbols by cosine similarity.

    Returns list of dicts with keys:
        qualified_name, kind, file_path, score
    """
    model = _get_model()
    query_vec = model.encode(query, convert_to_numpy=True).astype(np.float32)

    conn = _connect(db_path)
    try:
        q = """
            SELECT s.qualified_name, s.kind, f.path, s.embedding
            FROM symbols s
            JOIN files f ON s.file_id = f.id
            WHERE s.embedding IS NOT NULL
        """
        params: list = []
        if kind:
            q += " AND s.kind = ?"
            params.append(kind)

        rows = conn.execute(q, params).fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    scored = []
    for qualified_name, sym_kind, file_path, blob in rows:
        vec = _decode_embedding(blob)
        score = _cosine_similarity(query_vec, vec)
        scored.append({
            "qualified_name": qualified_name,
            "kind": sym_kind,
            "file_path": file_path,
            "score": score,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ── BM25 lexical index ────────────────────────────────────────────────────────

class BM25Index:
    """
    In-memory BM25 index over symbol text.
    Rebuilt from the database on each instantiation — cheap for single-codebase scale.
    """

    def __init__(self, db_path: str, kind: str | None = None):
        conn = _connect(db_path)
        try:
            q = """
                SELECT s.qualified_name, s.kind, f.path, s.docstring, s.source
                FROM symbols s
                JOIN files f ON s.file_id = f.id
            """
            params: list = []
            if kind:
                q += " WHERE s.kind = ?"
                params.append(kind)
            rows = conn.execute(q, params).fetchall()
        finally:
            conn.close()

        self._meta = [
            {"qualified_name": r[0], "kind": r[1], "file_path": r[2]}
            for r in rows
        ]
        texts = [_symbol_text((r[0], r[3], r[4])) for r in rows]
        tokenized = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Return top_k results by BM25 score."""
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        return [
            {**self._meta[i], "score": float(scores[i])}
            for i in ranked
            if scores[i] > 0
        ]


# ── hybrid search ─────────────────────────────────────────────────────────────

def hybrid_search(
    query: str,
    db_path: str,
    top_k: int = 10,
    bm25_weight: float = 0.3,
    vector_weight: float = 0.7,
    kind: str | None = None,
) -> list[dict]:
    """
    Combine BM25 and vector search with weighted score fusion (RRF-style normalization).

    bm25_weight + vector_weight should sum to 1.0.
    Default weighting favors semantic similarity over lexical — adjust as needed.

    Returns list of dicts with keys:
        qualified_name, kind, file_path, score, bm25_score, vector_score
    """
    bm25_index = BM25Index(db_path, kind=kind)
    bm25_results = bm25_index.search(query, top_k=top_k * 2)
    vector_results = vector_search(query, db_path, top_k=top_k * 2, kind=kind)

    # normalize scores to [0, 1]
    def normalize(results: list[dict], key: str = "score") -> dict[str, float]:
        if not results:
            return {}
        max_score = max(r[key] for r in results)
        if max_score == 0:
            return {r["qualified_name"]: 0.0 for r in results}
        return {r["qualified_name"]: r[key] / max_score for r in results}

    bm25_norm = normalize(bm25_results)
    vector_norm = normalize(vector_results)

    all_names = set(bm25_norm) | set(vector_norm)

    combined = []
    for name in all_names:
        b = bm25_norm.get(name, 0.0)
        v = vector_norm.get(name, 0.0)
        combined.append({
            "qualified_name": name,
            "score": bm25_weight * b + vector_weight * v,
            "bm25_score": b,
            "vector_score": v,
        })

    combined.sort(key=lambda x: x["score"], reverse=True)
    top = combined[:top_k]

    # enrich with kind and file_path from either result set
    meta: dict[str, dict] = {}
    for r in bm25_results + vector_results:
        if r["qualified_name"] not in meta:
            meta[r["qualified_name"]] = {
                "kind": r.get("kind"),
                "file_path": r.get("file_path"),
            }

    return [
        {**r, **meta.get(r["qualified_name"], {})}
        for r in top
    ]
