"""
retlib — code retrieval library for LLM context assembly.

Pipeline:
    parse → index → retrieve → build_context → LLM

Quickstart:
    from retlib import init_db, index_directory, retrieve, build_context

    init_db("my_project/.retlib/index.db")
    index_directory("my_project/", "my_project/.retlib/index.db")

    results = retrieve("how does authentication work", "my_project/.retlib/index.db")
    ctx = build_context(results, "my_project/.retlib/index.db")

    print(ctx.text)   # pass to your LLM
    print(summarize(ctx))  # observability summary
"""

from retlib.graph import init_db, index_result, stats
from retlib.index import build_embedding_index
from retlib.retriever import retrieve, explain
from retlib.context import build_context, summarize
from retlib.parser import parse_file, parse_directory

import os
from pathlib import Path


def index_directory(
    root: str,
    db_path: str,
    skip_dirs: set[str] | None = None,
    force: bool = False,
) -> dict:
    """
    Parse and index all Python files in a directory.

    Incremental by default — skips files whose checksum hasn't changed.
    Set force=True to re-embed all symbols (e.g. after swapping embedding model).

    Returns a summary dict with counts of files indexed and symbols stored.
    """
    results = parse_directory(root, skip_dirs=skip_dirs)
    for result in results:
        index_result(result, db_path)
    build_embedding_index(db_path, force=force)
    return stats(db_path)


__all__ = [
    # setup
    "init_db",
    "index_directory",
    # lower-level indexing (for incremental updates)
    "index_result",
    "parse_file",
    "parse_directory",
    # retrieval
    "retrieve",
    "build_context",
    # observability
    "explain",
    "summarize",
    "stats",
]
