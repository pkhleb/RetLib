# retlib

A code retrieval library for LLM context assembly. Given a natural language query and a Python codebase, retlib returns the most relevant source spans — ranked, budget-constrained, and ready to pass to an LLM.

## Why

Naive approaches to code context (dump the whole file, dump the whole directory) break down quickly. Token budgets are finite, and irrelevant code degrades LLM response quality. retlib addresses this with a structured retrieval pipeline:

```
parse → index → retrieve → build_context → LLM
```

Each stage is independently inspectable, so you can see exactly which symbols were retrieved, why they ranked where they did, and what got dropped due to budget constraints.

## Architecture

### 1. Parse (`parser.py`)
Extracts symbols and relationships from Python source using `ast.NodeVisitor`. Each `.py` file yields:
- **Symbols** — modules, classes, functions, methods with source spans and docstrings
- **Edges** — `CALLS`, `CONTAINS`, `IMPORTS`, `INHERITS` relationships between symbols

### 2. Index (`graph.py`, `index.py`)
Persists the code graph to SQLite and generates embeddings for semantic search.
- **Graph** — symbols and edges stored in SQLite with indexed lookups by name, kind, and relationship
- **BM25** — in-memory lexical index over symbol names, docstrings, and source
- **Embeddings** — `all-MiniLM-L6-v2` vectors stored as blobs in SQLite; incremental (only re-embeds changed files)

> **Note:** `all-MiniLM-L6-v2` is a general-purpose model chosen for its small size and CPU friendliness. A code-specific model (e.g. `voyage-code-2`, `nomic-embed-text`) would improve retrieval quality and is a one-line swap.

### 3. Retrieve (`retriever.py`)
Converts a query into a ranked list of relevant symbols.

1. **Task detection** — classifies query intent (`debug`, `refactor`, `understand`, `default`) via keyword matching, selecting appropriate graph edge types for expansion
2. **Seed search** — hybrid BM25 + vector search finds initial candidate symbols
3. **Graph expansion** — walks outward from seeds through typed edges for a fixed number of hops, surfacing related symbols the query didn't explicitly name
4. **Ranking** — manually weighted linear combination of five signals:

| Signal | Weight | Description |
|---|---|---|
| `semantic_score` | 0.35 | Cosine similarity to query embedding |
| `lexical_score` | 0.25 | BM25 score |
| `graph_distance` | 0.25 | Proximity to seed symbols (decays per hop) |
| `edge_type` | 0.10 | Reward for high-value edge types (CALLS > IMPORTS) |
| `symbol_kind` | 0.05 | Preference for functions/methods over modules |

> **Note:** Task detection uses keyword matching and ranking uses manually chosen weights. Both are flagged for future improvement — learned task classification and a trained ranker (e.g. LambdaMART) are natural next steps once labeled examples are available.

### 4. Build context (`context.py`)
Assembles the final context string within a token budget.
- Symbols included in rank order until budget is exhausted
- Each block includes: qualified name, file path, score, hop distance, relationship summary, source
- Relationship summaries only reference other symbols in the context window
- Footer reports included/excluded symbols and estimated token usage

Token count is estimated as `len(text) // 4`. True token count is available post-hoc from the LLM API response and should be used to calibrate the budget over time.

## Quickstart

```python
from retlib import init_db, index_directory, retrieve, build_context, summarize

# one-time setup
db = "my_project/.retlib/index.db"
init_db(db)
index_directory("my_project/", db)

# query
results = retrieve("how does authentication work", db)
ctx = build_context(results, db)

print(ctx.text)       # pass to your LLM
print(summarize(ctx)) # observability summary
```

## Observability

Every stage surfaces its reasoning:

```python
from retlib import retrieve, build_context, explain, summarize

results = retrieve("fix the bug in parse_file", db, top_k=8, hops=2)

# per-symbol score breakdown
for r in results:
    print(explain(r))

# context assembly summary
ctx = build_context(results, db, token_budget=2048)
print(summarize(ctx))
```

`explain()` output:
```
Symbol:   parse_file (function)
File:     src/parser.py  |  Score: 0.8312  (hop 0 from seed)
Signals:
  semantic_score       0.921 × 0.35 = 0.3224
  lexical_score        0.783 × 0.25 = 0.1958
  graph_distance       1.000 × 0.25 = 0.2500
  edge_type            1.000 × 0.10 = 0.1000
  symbol_kind          1.000 × 0.05 = 0.0500
```

`summarize()` output:
```
Token budget: 1847 / 2048 used
Included (6): parse_file, tokenize, build_ast, Parser, parse_directory, init_db
Excluded (2): connect, index_result
```

## Installation

```bash
pip install -e .
```

Dependencies: `sentence-transformers`, `rank-bm25`, `numpy`

## Project structure

```
retlib/
├── retlib/
│   ├── __init__.py     # public API
│   ├── parser.py       # AST parsing, symbol and edge extraction
│   ├── graph.py        # SQLite schema, graph persistence, neighbor traversal
│   ├── index.py        # BM25 index, embedding generation, hybrid search
│   ├── retriever.py    # task detection, graph expansion, ranking
│   └── context.py      # token budgeting, context assembly
└── tests/
    ├── test_parser.py
    ├── test_graph.py
    ├── test_index.py
    ├── test_retriever.py
    └── test_context.py
```

## Known limitations and planned improvements

- **Python only** — the AST parser targets Python. Adding support for other languages would require either tree-sitter or per-language parsers.
- **Embedding model** — `all-MiniLM-L6-v2` is general-purpose. A code-specific model would meaningfully improve semantic retrieval quality.
- **Task detection** — keyword matching is brittle. A learned classifier or lightweight LLM call would handle paraphrasing and ambiguous intent better.
- **Graph expansion** — fixed-hop expansion is simple but wasteful. Budget-based expansion with priority queuing would surface higher-value neighbors first.
- **Ranking** — manually chosen weights work but don't generalize. A trained ranker (LambdaMART, a small MLP) would improve once labeled retrieval examples are available.
- **Token estimation** — `len // 4` is an approximation. Wiring in actual token counts from LLM API responses would let the budget self-calibrate.
