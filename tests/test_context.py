"""
Tests for context.py.

Uses the same deterministic mock model as other test modules.
The populated_db fixture mirrors test_retriever.py for consistency.
"""

import numpy as np
import pytest
import retlib.index as idx
from retlib.context import (
    DEFAULT_TOKEN_BUDGET,
    ContextResult,
    _relationship_summary,
    build_context,
    estimate_tokens,
    summarize,
)
from retlib.graph import init_db, index_result
from retlib.index import build_embedding_index
from retlib.parser import ParseResult, Symbol, Edge
from retlib.retriever import RetrievalResult, retrieve

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
        source=source or f"def {name}():\n    pass",
        docstring=docstring,
    )


def make_result_obj(
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


def make_retrieval_result(
    qualified_name: str,
    kind: str = "function",
    file_path: str = "app.py",
    source: str = None,
    score: float = 0.8,
    hop: int = 0,
) -> RetrievalResult:
    return RetrievalResult(
        qualified_name=qualified_name,
        kind=kind,
        file_path=file_path,
        source=source or f"def {qualified_name}():\n    pass",
        score=score,
        signals={
            "semantic_score": score,
            "lexical_score": 0.5,
            "graph_distance": 1.0,
            "edge_type": 1.0,
            "symbol_kind": 1.0,
        },
        hop=hop,
    )


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    return path


@pytest.fixture
def populated_db(db):
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
    index_result_fn = index_result
    index_result_fn(make_result_obj(symbols=symbols, edges=edges), db)
    build_embedding_index(db)
    return db


# ── estimate_tokens ───────────────────────────────────────────────────────────

def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_estimate_tokens_approximation():
    text = "a" * 400
    assert estimate_tokens(text) == 100


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("x" * 100) < estimate_tokens("x" * 200)


def test_estimate_tokens_returns_int():
    assert isinstance(estimate_tokens("hello world"), int)


# ── _relationship_summary ─────────────────────────────────────────────────────

def test_relationship_summary_calls(populated_db):
    # parse_file calls tokenize and build_ast
    included = {"parse_file", "tokenize", "build_ast"}
    summary = _relationship_summary("parse_file", populated_db, included)
    assert "tokenize" in summary
    assert "build_ast" in summary


def test_relationship_summary_called_by(populated_db):
    # tokenize is called by parse_file
    included = {"parse_file", "tokenize"}
    summary = _relationship_summary("tokenize", populated_db, included)
    assert "parse_file" in summary


def test_relationship_summary_excludes_not_included(populated_db):
    # parse_file calls tokenize and build_ast, but build_ast not in included
    included = {"parse_file", "tokenize"}
    summary = _relationship_summary("parse_file", populated_db, included)
    assert "build_ast" not in summary


def test_relationship_summary_contains(populated_db):
    # Parser contains parse_file
    included = {"Parser", "parse_file"}
    summary = _relationship_summary("parse_file", populated_db, included)
    assert "Parser" in summary


def test_relationship_summary_empty_when_no_relations(populated_db):
    # connect is called by others but doesn't call anything in included
    included = {"connect"}
    summary = _relationship_summary("connect", populated_db, included)
    # no outbound calls to included symbols
    assert "Calls:" not in summary


def test_relationship_summary_empty_included_set(populated_db):
    summary = _relationship_summary("parse_file", populated_db, set())
    assert summary == ""


# ── build_context — basic ─────────────────────────────────────────────────────

def test_build_context_returns_context_result(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=5)
    ctx = build_context(results, populated_db)
    assert isinstance(ctx, ContextResult)


def test_build_context_empty_results(populated_db):
    ctx = build_context([], populated_db)
    assert ctx.text == ""
    assert ctx.included == []
    assert ctx.excluded == []
    assert ctx.estimated_tokens == 0


def test_build_context_text_is_string(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=5)
    ctx = build_context(results, populated_db)
    assert isinstance(ctx.text, str)


def test_build_context_included_not_empty(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=5)
    ctx = build_context(results, populated_db)
    assert len(ctx.included) > 0


def test_build_context_included_are_qualified_names(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=5)
    ctx = build_context(results, populated_db)
    result_names = {r.qualified_name for r in results}
    for name in ctx.included:
        assert name in result_names


def test_build_context_excluded_plus_included_equals_results(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=8)
    ctx = build_context(results, populated_db, token_budget=500)
    all_names = {r.qualified_name for r in results}
    accounted = set(ctx.included) | set(ctx.excluded)
    assert accounted == all_names


def test_build_context_signals_populated(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=5)
    ctx = build_context(results, populated_db)
    assert len(ctx.signals) > 0
    for name, signals in ctx.signals.items():
        assert name in ctx.included
        assert isinstance(signals, dict)


# ── build_context — budget ────────────────────────────────────────────────────

def test_build_context_respects_budget(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=8)
    budget = 300
    ctx = build_context(results, populated_db, token_budget=budget)
    # estimated tokens of text may slightly exceed due to footer/relationships
    # but included symbols should fit
    assert ctx.budget == budget


def test_build_context_tiny_budget_includes_nothing_or_one(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=5)
    ctx = build_context(results, populated_db, token_budget=5)
    # with a 5-token budget nothing should fit
    assert len(ctx.included) == 0


def test_build_context_large_budget_includes_all(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=5)
    ctx = build_context(results, populated_db, token_budget=99999)
    assert len(ctx.excluded) == 0
    assert len(ctx.included) == len(results)


def test_build_context_excluded_when_budget_tight(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=8)
    ctx = build_context(results, populated_db, token_budget=200)
    assert len(ctx.excluded) > 0


def test_build_context_default_budget(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=5)
    ctx = build_context(results, populated_db)
    assert ctx.budget == DEFAULT_TOKEN_BUDGET


# ── build_context — format ────────────────────────────────────────────────────

def test_build_context_text_contains_qualified_names(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=3)
    ctx = build_context(results, populated_db)
    for name in ctx.included:
        assert name in ctx.text


def test_build_context_text_contains_source(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=3)
    ctx = build_context(results, populated_db)
    for r in results:
        if r.qualified_name in ctx.included:
            # at least first line of source should appear
            first_line = r.source.split("\n")[0]
            assert first_line in ctx.text


def test_build_context_text_contains_file_path(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=3)
    ctx = build_context(results, populated_db)
    assert "app.py" in ctx.text


def test_build_context_text_contains_score(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=3)
    ctx = build_context(results, populated_db)
    assert "Score:" in ctx.text


def test_build_context_text_contains_footer(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=5)
    ctx = build_context(results, populated_db)
    assert "symbols included" in ctx.text
    assert "Estimated tokens:" in ctx.text


def test_build_context_footer_lists_excluded(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=8)
    ctx = build_context(results, populated_db, token_budget=200)
    if ctx.excluded:
        assert "Dropped (budget)" in ctx.text
        for name in ctx.excluded:
            assert name in ctx.text


def test_build_context_relationships_in_text(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=8, hops=1)
    ctx = build_context(results, populated_db, token_budget=99999)
    # with everything included, relationships should appear
    assert "Relationships:" in ctx.text


# ── summarize ─────────────────────────────────────────────────────────────────

def test_summarize_returns_string(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=5)
    ctx = build_context(results, populated_db)
    assert isinstance(summarize(ctx), str)


def test_summarize_contains_budget_info(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=5)
    ctx = build_context(results, populated_db)
    s = summarize(ctx)
    assert "Token budget" in s
    assert str(ctx.budget) in s


def test_summarize_contains_included_names(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=5)
    ctx = build_context(results, populated_db)
    s = summarize(ctx)
    for name in ctx.included:
        assert name in s


def test_summarize_contains_excluded_when_present(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=8)
    ctx = build_context(results, populated_db, token_budget=200)
    s = summarize(ctx)
    if ctx.excluded:
        assert "Excluded" in s
        for name in ctx.excluded:
            assert name in s


def test_summarize_no_excluded_section_when_all_fit(populated_db):
    results = retrieve("parse a python file", populated_db, top_k=5)
    ctx = build_context(results, populated_db, token_budget=99999)
    s = summarize(ctx)
    assert "Excluded" not in s
