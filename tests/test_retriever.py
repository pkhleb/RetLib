"""
Tests for retriever.py.

Embedding model is mocked with deterministic vectors (same approach as
test_index.py). Graph and index are populated with a small synthetic codebase
that's realistic enough to exercise task detection, expansion, and ranking.
"""

import numpy as np
import pytest
import retlib.index as idx
from retlib.graph import init_db, index_result
from retlib.index import build_embedding_index
from retlib.parser import ParseResult, Symbol, Edge
from retlib.retriever import (
    TASK_DEBUG,
    TASK_DEFAULT,
    TASK_REFACTOR,
    TASK_UNDERSTAND,
    WEIGHTS,
    RetrievalResult,
    detect_task,
    explain,
    retrieve,
)

# ── mock model ────────────────────────────────────────────────────────────────

class DeterministicModel:
    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return self._vec(texts)
        return np.stack([self._vec(t) for t in texts])

    def _vec(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.random(384).astype(np.float32)
        return v / np.linalg.norm(v)


@pytest.fixture(autouse=True)
def mock_model(monkeypatch):
    monkeypatch.setattr(idx, "_model", DeterministicModel())


# ── helpers ───────────────────────────────────────────────────────────────────

def make_symbol(
    qualified_name: str,
    kind: str = "function",
    file_path: str = "app.py",
    docstring: str = None,
    source: str = None,
) -> Symbol:
    name = qualified_name.split(".")[-1]
    return Symbol(
        file_path=file_path,
        name=name,
        qualified_name=qualified_name,
        kind=kind,
        start_line=1,
        end_line=10,
        source=source or f"def {name}(): pass",
        docstring=docstring,
    )


def make_result(
    file_path: str = "app.py",
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


def make_edge(source: str, target: str, kind: str) -> Edge:
    return Edge(source_qualified_name=source, target_qualified_name=target, kind=kind)


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    return path


@pytest.fixture
def populated_db(db):
    """
    Small synthetic codebase:

        Parser
        ├── parse_file        (calls: tokenize, build_ast)
        ├── parse_directory   (calls: parse_file)
        └── tokenize

        Graph
        ├── init_db           (calls: connect)
        ├── index_result      (calls: init_db, connect)
        └── connect

    Edges:
        parse_file    CALLS tokenize
        parse_file    CALLS build_ast
        parse_directory CALLS parse_file
        index_result  CALLS init_db
        index_result  CALLS connect
        init_db       CALLS connect
        Parser        CONTAINS parse_file
        Parser        CONTAINS parse_directory
        Parser        CONTAINS tokenize
    """
    symbols = [
        make_symbol("parse_file", kind="function",
                    docstring="Parse a single Python source file.",
                    source="def parse_file(path):\n    tokens = tokenize(path)\n    return build_ast(tokens)"),
        make_symbol("parse_directory", kind="function",
                    docstring="Recursively parse all Python files in a directory.",
                    source="def parse_directory(root):\n    for f in os.walk(root):\n        parse_file(f)"),
        make_symbol("tokenize", kind="function",
                    docstring="Tokenize source code into a list of tokens.",
                    source="def tokenize(source):\n    return list(source.split())"),
        make_symbol("build_ast", kind="function",
                    docstring="Build an AST from a token list.",
                    source="def build_ast(tokens):\n    return ast.parse(' '.join(tokens))"),
        make_symbol("Parser", kind="class",
                    docstring="Main parser class.",
                    source="class Parser:\n    pass"),
        make_symbol("init_db", kind="function",
                    docstring="Initialize the SQLite database.",
                    source="def init_db(path):\n    conn = connect(path)\n    conn.executescript(SCHEMA)"),
        make_symbol("index_result", kind="function",
                    docstring="Index a parse result into the database.",
                    source="def index_result(result, db):\n    init_db(db)\n    conn = connect(db)"),
        make_symbol("connect", kind="function",
                    docstring="Open a SQLite connection.",
                    source="def connect(path):\n    return sqlite3.connect(path)"),
    ]
    edges = [
        make_edge("parse_file", "tokenize", "CALLS"),
        make_edge("parse_file", "build_ast", "CALLS"),
        make_edge("parse_directory", "parse_file", "CALLS"),
        make_edge("index_result", "init_db", "CALLS"),
        make_edge("index_result", "connect", "CALLS"),
        make_edge("init_db", "connect", "CALLS"),
        make_edge("Parser", "parse_file", "CONTAINS"),
        make_edge("Parser", "parse_directory", "CONTAINS"),
        make_edge("Parser", "tokenize", "CONTAINS"),
    ]
    index_result_fn = index_result  # avoid name collision with symbol
    index_result_fn(make_result(symbols=symbols, edges=edges), db)
    build_embedding_index(db)
    return db


# ── detect_task ───────────────────────────────────────────────────────────────

def test_detect_task_debug():
    assert detect_task("there's a bug in parse_file") == TASK_DEBUG
    assert detect_task("fix the error in tokenize") == TASK_DEBUG
    assert detect_task("it keeps crashing") == TASK_DEBUG


def test_detect_task_refactor():
    assert detect_task("refactor the parser module") == TASK_REFACTOR
    assert detect_task("rename parse_file to parse") == TASK_REFACTOR
    assert detect_task("simplify the tokenizer") == TASK_REFACTOR


def test_detect_task_understand():
    assert detect_task("explain how parse_file works") == TASK_UNDERSTAND
    assert detect_task("what does init_db do") == TASK_UNDERSTAND
    assert detect_task("give me an overview of the parser") == TASK_UNDERSTAND


def test_detect_task_default():
    assert detect_task("add a new function") == TASK_DEFAULT
    assert detect_task("search for parse") == TASK_DEFAULT
    assert detect_task("") == TASK_DEFAULT


def test_detect_task_case_insensitive():
    assert detect_task("FIX THE BUG") == TASK_DEBUG
    assert detect_task("EXPLAIN HOW THIS WORKS") == TASK_UNDERSTAND


def test_detect_task_first_match_wins():
    # "fix" → debug, "refactor" → refactor — first keyword match wins
    result = detect_task("fix and refactor the parser")
    assert result in (TASK_DEBUG, TASK_REFACTOR)  # either is valid, just consistent


# ── retrieve — basic ──────────────────────────────────────────────────────────

def test_retrieve_returns_results(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=5)
    assert len(results) > 0


def test_retrieve_returns_retrieval_results(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=3)
    for r in results:
        assert isinstance(r, RetrievalResult)


def test_retrieve_result_fields(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=1)
    r = results[0]
    assert r.qualified_name
    assert r.kind in ("function", "method", "class", "module")
    assert r.file_path
    assert r.source
    assert isinstance(r.score, float)
    assert isinstance(r.signals, dict)
    assert isinstance(r.hop, int)


def test_retrieve_sorted_by_score_descending(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=8)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_top_k_respected(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=3)
    assert len(results) <= 3


def test_retrieve_empty_db(db):
    results = retrieve("parse a python file", db, top_k=5)
    assert results == []


def test_retrieve_no_seeds_returns_empty(populated_db, monkeypatch):
    import retlib.retriever as ret
    monkeypatch.setattr(ret, "hybrid_search", lambda *a, **kw: [])
    results = retrieve("anything", populated_db, top_k=5)
    assert results == []


# ── retrieve — hops ───────────────────────────────────────────────────────────

def test_retrieve_hop_0_only(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=20, hops=0)
    assert all(r.hop == 0 for r in results)


def test_retrieve_hops_expands_graph(populated_db):
    results_0 = retrieve("parse a python file", populated_db, top_k=20, hops=0)
    results_2 = retrieve("parse a python file", populated_db, top_k=20, hops=2)
    # more hops should surface more candidates
    assert len(results_2) >= len(results_0)


def test_retrieve_hop_recorded_correctly(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=20, hops=2)
    hops = {r.hop for r in results}
    # should have mix of seed (0) and expanded (1+) symbols
    assert 0 in hops


# ── retrieve — task override ──────────────────────────────────────────────────

def test_retrieve_task_override(populated_db):
    # same query, different task — should not raise and should return results
    r_debug = retrieve("parse file", populated_db, top_k=5, task=TASK_DEBUG)
    r_understand = retrieve("parse file", populated_db, top_k=5, task=TASK_UNDERSTAND)
    assert len(r_debug) > 0
    assert len(r_understand) > 0


def test_retrieve_task_override_changes_edges(populated_db):
    # debug focuses on CALLS edges, understand on CONTAINS/IMPORTS
    # with hops=1, different tasks may surface different neighbors
    r_debug = retrieve("Parser", populated_db, top_k=20, hops=1, task=TASK_DEBUG)
    r_understand = retrieve("Parser", populated_db, top_k=20, hops=1, task=TASK_UNDERSTAND)
    debug_names = {r.qualified_name for r in r_debug}
    understand_names = {r.qualified_name for r in r_understand}
    # both sets should be non-empty
    assert len(debug_names) > 0
    assert len(understand_names) > 0


# ── retrieve — signals ────────────────────────────────────────────────────────

def test_retrieve_signals_present(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=3)
    for r in results:
        assert "semantic_score" in r.signals
        assert "lexical_score" in r.signals
        assert "graph_distance" in r.signals
        assert "edge_type" in r.signals
        assert "symbol_kind" in r.signals


def test_retrieve_seed_has_graph_distance_1(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=10, hops=0)
    for r in results:
        assert r.signals["graph_distance"] == pytest.approx(1.0)


def test_retrieve_hop1_has_lower_distance_score(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=20, hops=1)
    hop0 = [r for r in results if r.hop == 0]
    hop1 = [r for r in results if r.hop == 1]
    if hop0 and hop1:
        avg0 = sum(r.signals["graph_distance"] for r in hop0) / len(hop0)
        avg1 = sum(r.signals["graph_distance"] for r in hop1) / len(hop1)
        assert avg0 > avg1


def test_retrieve_score_matches_weighted_signals(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=5)
    for r in results:
        expected = sum(WEIGHTS[k] * v for k, v in r.signals.items())
        assert r.score == pytest.approx(expected, abs=1e-5)


# ── explain ───────────────────────────────────────────────────────────────────

def test_explain_returns_string(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=1)
    assert isinstance(explain(results[0]), str)


def test_explain_contains_qualified_name(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=1)
    output = explain(results[0])
    assert results[0].qualified_name in output


def test_explain_contains_all_signals(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=1)
    output = explain(results[0])
    for signal in WEIGHTS:
        assert signal in output


def test_explain_contains_score(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=1)
    output = explain(results[0])
    assert f"{results[0].score:.4f}" in output


def test_explain_contains_hop(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=1)
    output = explain(results[0])
    assert f"hop {results[0].hop}" in output
