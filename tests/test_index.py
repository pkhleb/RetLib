"""
Tests for index.py.

The embedding model (all-MiniLM-L6-v2) requires network access to download,
so all tests mock the model with deterministic vectors. This lets us test
the full pipeline — encoding, storage, retrieval, scoring — without a network
dependency or GPU.
"""

import numpy as np
import pytest
import retlib.index as idx
from retlib.graph import init_db, index_result
from retlib.index import (
    BM25Index,
    _decode_embedding,
    _encode_embedding,
    build_embedding_index,
    hybrid_search,
    vector_search,
)
from retlib.parser import ParseResult, Symbol, Edge

# ── mock model ────────────────────────────────────────────────────────────────

class DeterministicModel:
    """
    Returns a fixed embedding per text based on a hash of the text.
    Same text always gets the same vector — lets us test similarity logic.
    """
    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return self._vec(texts)
        return np.stack([self._vec(t) for t in texts])

    def _vec(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.random(384).astype(np.float32)
        return v / np.linalg.norm(v)  # unit vector


@pytest.fixture(autouse=True)
def mock_model(monkeypatch):
    """Replace the global model with the deterministic mock for every test."""
    monkeypatch.setattr(idx, "_model", DeterministicModel())


# ── helpers ───────────────────────────────────────────────────────────────────

def make_symbol(
    qualified_name: str,
    kind: str = "function",
    file_path: str = "foo.py",
    docstring: str = None,
    source: str = "def foo(): pass",
) -> Symbol:
    return Symbol(
        file_path=file_path,
        name=qualified_name.split(".")[-1],
        qualified_name=qualified_name,
        kind=kind,
        start_line=1,
        end_line=5,
        source=source,
        docstring=docstring,
    )


def make_result(
    file_path: str = "foo.py",
    checksum: str = "abc",
    symbols: list = None,
    edges: list = None,
) -> ParseResult:
    return ParseResult(
        file_path=file_path,
        checksum=checksum,
        symbols=symbols or [],
        edges=edges or [],
    )


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    return path


@pytest.fixture
def populated_db(db):
    """DB with a small set of symbols already indexed and embedded."""
    syms = [
        make_symbol(
            "parse_file", kind="function",
            docstring="Parse a Python file and extract symbols.",
            source="def parse_file(path):\n    with open(path) as f:\n        return ast.parse(f.read())",
        ),
        make_symbol(
            "parse_directory", kind="function",
            docstring="Parse all Python files in a directory.",
            source="def parse_directory(root):\n    for path in Path(root).rglob('*.py'):\n        yield parse_file(path)",
        ),
        make_symbol(
            "SymbolVisitor", kind="class",
            docstring="AST visitor that extracts symbols from a Python file.",
            source="class SymbolVisitor(ast.NodeVisitor):\n    def __init__(self):\n        self.symbols = []",
        ),
        make_symbol(
            "SymbolVisitor.visit_FunctionDef", kind="method",
            docstring="Visit a function definition node and record it as a symbol.",
            source="def visit_FunctionDef(self, node):\n    self.symbols.append(node.name)\n    self.generic_visit(node)",
        ),
        make_symbol(
            "init_db", kind="function",
            docstring="Initialize the SQLite database and create tables.",
            source="def init_db(db_path):\n    conn = sqlite3.connect(db_path)\n    conn.executescript(SCHEMA)",
        ),
        make_symbol(
            "get_symbol", kind="function",
            docstring="Fetch a symbol by its qualified name from the database.",
            source="def get_symbol(name, db_path):\n    conn = sqlite3.connect(db_path)\n    return conn.execute('SELECT * FROM symbols WHERE qualified_name = ?', (name,)).fetchone()",
        ),
    ]
    index_result(make_result(symbols=syms), db)
    build_embedding_index(db)
    return db


# ── _encode_embedding / _decode_embedding ─────────────────────────────────────

def test_encode_decode_roundtrip():
    vec = np.random.rand(384).astype(np.float32)
    assert np.array_equal(vec, _decode_embedding(_encode_embedding(vec)))


def test_encode_decode_preserves_values():
    vec = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    recovered = _decode_embedding(_encode_embedding(vec))
    np.testing.assert_allclose(recovered, vec, rtol=1e-6)


def test_encode_produces_bytes():
    vec = np.random.rand(384).astype(np.float32)
    assert isinstance(_encode_embedding(vec), bytes)


def test_decode_produces_ndarray():
    vec = np.random.rand(384).astype(np.float32)
    result = _decode_embedding(_encode_embedding(vec))
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32


# ── build_embedding_index ─────────────────────────────────────────────────────

def test_build_embedding_index_populates_embeddings(db):
    syms = [make_symbol("foo"), make_symbol("bar")]
    index_result(make_result(symbols=syms), db)
    build_embedding_index(db)

    import sqlite3
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT embedding FROM symbols WHERE embedding IS NOT NULL").fetchall()
    conn.close()
    assert len(rows) == 2


def test_build_embedding_index_skips_already_embedded(db):
    syms = [make_symbol("foo")]
    index_result(make_result(symbols=syms), db)
    build_embedding_index(db)

    # second call should be a no-op — mock would produce same vector anyway
    # just verify it doesn't raise
    build_embedding_index(db)

    import sqlite3
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT embedding FROM symbols WHERE embedding IS NOT NULL").fetchall()
    conn.close()
    assert len(rows) == 1


def test_build_embedding_index_force_regenerates(db):
    syms = [make_symbol("foo")]
    index_result(make_result(symbols=syms), db)
    build_embedding_index(db)

    import sqlite3
    conn = sqlite3.connect(db)
    before = conn.execute("SELECT embedding FROM symbols").fetchone()[0]

    # force=True with same mock model produces same vector
    build_embedding_index(db, force=True)
    after = conn.execute("SELECT embedding FROM symbols").fetchone()[0]
    conn.close()

    # both should be non-null
    assert before is not None
    assert after is not None


def test_build_embedding_index_empty_db(db):
    # should not raise on empty db
    build_embedding_index(db)


# ── vector_search ─────────────────────────────────────────────────────────────

def test_vector_search_returns_results(populated_db):
    results = vector_search("parse python file", populated_db, top_k=3)
    assert len(results) <= 3
    assert len(results) > 0


def test_vector_search_result_keys(populated_db):
    results = vector_search("parse", populated_db, top_k=1)
    assert len(results) == 1
    r = results[0]
    assert "qualified_name" in r
    assert "kind" in r
    assert "file_path" in r
    assert "score" in r


def test_vector_search_scores_between_0_and_1(populated_db):
    results = vector_search("parse", populated_db, top_k=5)
    for r in results:
        assert -1.0 <= r["score"] <= 1.0


def test_vector_search_sorted_by_score_descending(populated_db):
    results = vector_search("parse", populated_db, top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_vector_search_kind_filter(populated_db):
    results = vector_search("parse", populated_db, top_k=10, kind="function")
    assert all(r["kind"] == "function" for r in results)


def test_vector_search_kind_filter_method(populated_db):
    results = vector_search("visit", populated_db, top_k=10, kind="method")
    assert all(r["kind"] == "method" for r in results)


def test_vector_search_top_k_respected(populated_db):
    results = vector_search("parse", populated_db, top_k=2)
    assert len(results) <= 2


def test_vector_search_empty_db(db):
    results = vector_search("anything", db, top_k=5)
    assert results == []


def test_vector_search_same_query_same_results(populated_db):
    r1 = vector_search("parse python file", populated_db, top_k=3)
    r2 = vector_search("parse python file", populated_db, top_k=3)
    assert [r["qualified_name"] for r in r1] == [r["qualified_name"] for r in r2]


# ── BM25Index ─────────────────────────────────────────────────────────────────

def test_bm25_returns_results(populated_db):
    bm25 = BM25Index(populated_db)
    results = bm25.search("parse", top_k=3)
    assert len(results) > 0


def test_bm25_result_keys(populated_db):
    bm25 = BM25Index(populated_db)
    results = bm25.search("parse", top_k=1)
    assert len(results) == 1
    r = results[0]
    assert "qualified_name" in r
    assert "kind" in r
    assert "file_path" in r
    assert "score" in r


def test_bm25_sorted_by_score_descending(populated_db):
    bm25 = BM25Index(populated_db)
    results = bm25.search("symbol visitor", top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_bm25_no_match_returns_empty(populated_db):
    bm25 = BM25Index(populated_db)
    results = bm25.search("xyzzy_nonexistent_token", top_k=5)
    assert results == []


def test_bm25_top_k_respected(populated_db):
    bm25 = BM25Index(populated_db)
    results = bm25.search("parse", top_k=2)
    assert len(results) <= 2


def test_bm25_kind_filter(populated_db):
    bm25 = BM25Index(populated_db, kind="function")
    results = bm25.search("parse", top_k=10)
    assert all(r["kind"] == "function" for r in results)


def test_bm25_symbol_name_in_query_scores_high(populated_db):
    bm25 = BM25Index(populated_db)
    results = bm25.search("get_symbol", top_k=5)
    names = [r["qualified_name"] for r in results]
    assert "get_symbol" in names
    # should be top result
    assert names[0] == "get_symbol"


def test_bm25_empty_db(db):
    bm25 = BM25Index(db)
    results = bm25.search("anything", top_k=5)
    assert results == []


# ── hybrid_search ─────────────────────────────────────────────────────────────

def test_hybrid_returns_results(populated_db):
    results = hybrid_search("parse python file", populated_db, top_k=3)
    assert len(results) > 0


def test_hybrid_result_keys(populated_db):
    results = hybrid_search("parse", populated_db, top_k=1)
    assert len(results) == 1
    r = results[0]
    assert "qualified_name" in r
    assert "score" in r
    assert "bm25_score" in r
    assert "vector_score" in r
    assert "kind" in r
    assert "file_path" in r


def test_hybrid_sorted_by_score_descending(populated_db):
    results = hybrid_search("parse", populated_db, top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_top_k_respected(populated_db):
    results = hybrid_search("parse", populated_db, top_k=2)
    assert len(results) <= 2


def test_hybrid_kind_filter(populated_db):
    results = hybrid_search("parse", populated_db, top_k=10, kind="function")
    assert all(r["kind"] == "function" for r in results)


def test_hybrid_scores_are_weighted_combination(populated_db):
    results = hybrid_search(
        "parse", populated_db, top_k=5,
        bm25_weight=0.3, vector_weight=0.7
    )
    for r in results:
        expected = 0.3 * r["bm25_score"] + 0.7 * r["vector_score"]
        assert abs(r["score"] - expected) < 1e-5


def test_hybrid_bm25_only_weight(populated_db):
    results = hybrid_search(
        "get_symbol", populated_db, top_k=5,
        bm25_weight=1.0, vector_weight=0.0
    )
    for r in results:
        assert abs(r["score"] - r["bm25_score"]) < 1e-5


def test_hybrid_vector_only_weight(populated_db):
    results = hybrid_search(
        "parse", populated_db, top_k=5,
        bm25_weight=0.0, vector_weight=1.0
    )
    for r in results:
        assert abs(r["score"] - r["vector_score"]) < 1e-5


def test_hybrid_empty_db(db):
    results = hybrid_search("anything", db, top_k=5)
    assert results == []
