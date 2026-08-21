import os
import tempfile
import pytest
from retlib.parser import ParseResult, Symbol, Edge
from retlib.graph import (
    init_db,
    index_result,
    index_results,
    get_file_checksum,
    get_symbol,
    get_neighbors,
    search_symbols,
    stats,
)

# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path):
    """Fresh database for each test."""
    path = str(tmp_path / "test.db")
    init_db(path)
    return path


def make_result(
    file_path: str = "foo.py",
    checksum: str = "abc123",
    symbols: list[Symbol] = None,
    edges: list[Edge] = None,
) -> ParseResult:
    return ParseResult(
        file_path=file_path,
        checksum=checksum,
        symbols=symbols or [],
        edges=edges or [],
    )


def make_symbol(
    qualified_name: str,
    name: str = None,
    kind: str = "function",
    file_path: str = "foo.py",
    start_line: int = 1,
    end_line: int = 5,
    source: str = "def foo(): pass",
    docstring: str = None,
) -> Symbol:
    return Symbol(
        file_path=file_path,
        name=name or qualified_name.split(".")[-1],
        qualified_name=qualified_name,
        kind=kind,
        start_line=start_line,
        end_line=end_line,
        source=source,
        docstring=docstring,
    )


def make_edge(source: str, target: str, kind: str) -> Edge:
    return Edge(source_qualified_name=source, target_qualified_name=target, kind=kind)


# ── init_db ───────────────────────────────────────────────────────────────────

def test_init_db_creates_file(tmp_path):
    path = str(tmp_path / "new.db")
    init_db(path)
    assert os.path.exists(path)


def test_init_db_idempotent(tmp_path):
    path = str(tmp_path / "new.db")
    init_db(path)
    init_db(path)  # should not raise


def test_init_db_creates_nested_dirs(tmp_path):
    path = str(tmp_path / "a" / "b" / "index.db")
    init_db(path)
    assert os.path.exists(path)


# ── index_result ──────────────────────────────────────────────────────────────

def test_index_result_inserts_file(db):
    result = make_result()
    index_result(result, db)
    checksum = get_file_checksum("foo.py", db)
    assert checksum == "abc123"


def test_index_result_inserts_symbols(db):
    sym = make_symbol("foo")
    result = make_result(symbols=[sym])
    index_result(result, db)
    fetched = get_symbol("foo", db)
    assert fetched is not None
    assert fetched["qualified_name"] == "foo"


def test_index_result_inserts_edges(db):
    result = make_result(edges=[make_edge("foo", "bar", "CALLS")])
    index_result(result, db)
    neighbors = get_neighbors("foo", db_path=db)
    assert any(n["qualified_name"] == "bar" for n in neighbors)


def test_index_result_skips_unchanged(db):
    sym = make_symbol("foo")
    result = make_result(symbols=[sym])
    index_result(result, db)
    # modify in memory but keep same checksum — second call should be no-op
    sym2 = make_symbol("bar")
    result2 = make_result(checksum="abc123", symbols=[sym, sym2])
    index_result(result2, db)
    # bar should NOT be in db since checksum matched
    assert get_symbol("bar", db) is None


def test_index_result_replaces_on_checksum_change(db):
    sym = make_symbol("foo")
    result = make_result(checksum="v1", symbols=[sym])
    index_result(result, db)

    sym2 = make_symbol("bar")
    result2 = make_result(checksum="v2", symbols=[sym2])
    index_result(result2, db)

    assert get_symbol("foo", db) is None
    assert get_symbol("bar", db) is not None


def test_index_result_updates_checksum(db):
    index_result(make_result(checksum="v1"), db)
    index_result(make_result(checksum="v2"), db)
    assert get_file_checksum("foo.py", db) == "v2"


def test_index_result_duplicate_edges_ignored(db):
    edge = make_edge("foo", "bar", "CALLS")
    result = make_result(edges=[edge, edge])
    index_result(result, db)  # should not raise
    neighbors = get_neighbors("foo", db_path=db)
    bar_neighbors = [n for n in neighbors if n["qualified_name"] == "bar"]
    assert len(bar_neighbors) == 1


def test_index_results_multiple_files(db):
    r1 = make_result(file_path="a.py", checksum="c1", symbols=[make_symbol("alpha", file_path="a.py")])
    r2 = make_result(file_path="b.py", checksum="c2", symbols=[make_symbol("beta", file_path="b.py")])
    index_results([r1, r2], db)
    assert get_symbol("alpha", db) is not None
    assert get_symbol("beta", db) is not None


# ── get_file_checksum ─────────────────────────────────────────────────────────

def test_get_file_checksum_missing_returns_none(db):
    assert get_file_checksum("nonexistent.py", db) is None


def test_get_file_checksum_returns_stored(db):
    index_result(make_result(checksum="deadbeef"), db)
    assert get_file_checksum("foo.py", db) == "deadbeef"


# ── get_symbol ────────────────────────────────────────────────────────────────

def test_get_symbol_missing_returns_none(db):
    assert get_symbol("nonexistent", db) is None


def test_get_symbol_returns_all_fields(db):
    sym = make_symbol(
        "MyClass.my_method",
        kind="method",
        start_line=10,
        end_line=20,
        source="def my_method(self): pass",
        docstring="A method.",
    )
    index_result(make_result(symbols=[sym]), db)
    fetched = get_symbol("MyClass.my_method", db)
    assert fetched["qualified_name"] == "MyClass.my_method"
    assert fetched["kind"] == "method"
    assert fetched["start_line"] == 10
    assert fetched["end_line"] == 20
    assert fetched["source"] == "def my_method(self): pass"
    assert fetched["docstring"] == "A method."
    assert fetched["file_path"] == "foo.py"


# ── get_neighbors ─────────────────────────────────────────────────────────────

def test_get_neighbors_out(db):
    result = make_result(edges=[make_edge("foo", "bar", "CALLS")])
    index_result(result, db)
    neighbors = get_neighbors("foo", direction="out", db_path=db)
    assert any(n["qualified_name"] == "bar" for n in neighbors)


def test_get_neighbors_in(db):
    result = make_result(edges=[make_edge("foo", "bar", "CALLS")])
    index_result(result, db)
    neighbors = get_neighbors("bar", direction="in", db_path=db)
    assert any(n["qualified_name"] == "foo" for n in neighbors)


def test_get_neighbors_both(db):
    result = make_result(edges=[
        make_edge("foo", "bar", "CALLS"),
        make_edge("baz", "foo", "CALLS"),
    ])
    index_result(result, db)
    neighbors = get_neighbors("foo", direction="both", db_path=db)
    names = {n["qualified_name"] for n in neighbors}
    assert "bar" in names
    assert "baz" in names


def test_get_neighbors_edge_kind_filter(db):
    result = make_result(edges=[
        make_edge("foo", "bar", "CALLS"),
        make_edge("foo", "baz", "IMPORTS"),
    ])
    index_result(result, db)
    neighbors = get_neighbors("foo", edge_kinds=["CALLS"], db_path=db)
    names = {n["qualified_name"] for n in neighbors}
    assert "bar" in names
    assert "baz" not in names


def test_get_neighbors_no_results(db):
    index_result(make_result(), db)
    assert get_neighbors("foo", db_path=db) == []


def test_get_neighbors_edge_kind_in_result(db):
    result = make_result(edges=[make_edge("foo", "bar", "CALLS")])
    index_result(result, db)
    neighbors = get_neighbors("foo", db_path=db)
    assert neighbors[0]["edge_kind"] == "CALLS"


# ── search_symbols ────────────────────────────────────────────────────────────

def test_search_symbols_exact(db):
    index_result(make_result(symbols=[make_symbol("parse_file")]), db)
    results = search_symbols("parse_file", db_path=db)
    assert any(r["qualified_name"] == "parse_file" for r in results)


def test_search_symbols_substring(db):
    index_result(make_result(symbols=[
        make_symbol("parse_file"),
        make_symbol("parse_dir"),
        make_symbol("build_graph"),
    ]), db)
    results = search_symbols("parse", db_path=db)
    names = {r["qualified_name"] for r in results}
    assert "parse_file" in names
    assert "parse_dir" in names
    assert "build_graph" not in names


def test_search_symbols_kind_filter(db):
    index_result(make_result(symbols=[
        make_symbol("MyClass", kind="class"),
        make_symbol("my_func", kind="function"),
    ]), db)
    results = search_symbols("my", kind="class", db_path=db)
    assert all(r["kind"] == "class" for r in results)
    names = {r["qualified_name"] for r in results}
    assert "MyClass" in names
    assert "my_func" not in names


def test_search_symbols_no_results(db):
    index_result(make_result(), db)
    assert search_symbols("nonexistent", db_path=db) == []


# ── stats ─────────────────────────────────────────────────────────────────────

def test_stats_empty_db(db):
    s = stats(db)
    assert s["files"] == 0
    assert s["symbols"] == 0
    assert s["edges"] == 0


def test_stats_counts(db):
    result = make_result(
        symbols=[
            make_symbol("foo", kind="function"),
            make_symbol("MyClass", kind="class"),
        ],
        edges=[make_edge("foo", "MyClass", "CALLS")],
    )
    index_result(result, db)
    s = stats(db)
    assert s["files"] == 1
    assert s["symbols"] == 2
    assert s["edges"] == 1
    assert s["symbols_by_kind"]["function"] == 1
    assert s["symbols_by_kind"]["class"] == 1
    assert s["edges_by_kind"]["CALLS"] == 1
