"""
retriever.py — Query-driven symbol retrieval over the code graph.

Pipeline:
    query → task detection → seed symbols (hybrid search)
          → graph expansion (fixed hops, task-aware edge selection)
          → ranking (manually weighted linear combination)
          → ranked RetrievalResult list

TODOs flagged for future improvement:
    - TODO(task-detection): replace keyword matching with a learned classifier
      or lightweight LLM call for more robust intent detection.
    - TODO(expansion): replace fixed-hop expansion with budget-based expansion
      that prioritizes higher-ranked neighbors and stops at N total symbols.
    - TODO(ranking): replace manually weighted linear combination with a
      learned ranking model (e.g. LambdaMART, a small MLP) trained on
      labeled retrieval examples.
"""

from dataclasses import dataclass, field
from retlib.graph import get_neighbors, get_symbol
from retlib.index import hybrid_search

# ── task types ────────────────────────────────────────────────────────────────

TASK_DEBUG = "debug"
TASK_REFACTOR = "refactor"
TASK_UNDERSTAND = "understand"
TASK_DEFAULT = "default"

# Keywords used for task detection.
# TODO(task-detection): this is naive keyword matching — replace with a
# classifier or LLM call for robustness against paraphrasing.
TASK_KEYWORDS: dict[str, list[str]] = {
    TASK_DEBUG: [
        "bug", "error", "exception", "crash", "fail", "broken", "fix",
        "debug", "traceback", "stack trace", "not working", "wrong",
    ],
    TASK_REFACTOR: [
        "refactor", "rename", "move", "extract", "reorganize", "clean",
        "restructure", "simplify", "improve", "rewrite",
    ],
    TASK_UNDERSTAND: [
        "explain", "understand", "how does", "what does", "overview",
        "summarize", "describe", "walk me through", "what is",
    ],
}

# Edge types to expand per task.
# Debugging follows the call graph (who calls what, what gets called).
# Refactoring needs callers and references (what would break).
# Understanding needs structure (containment, imports, type relationships).
TASK_EDGE_KINDS: dict[str, list[str]] = {
    TASK_DEBUG:     ["CALLS", "CONTAINS"],
    TASK_REFACTOR:  ["CALLS", "CONTAINS", "IMPORTS"],
    TASK_UNDERSTAND: ["CONTAINS", "IMPORTS", "INHERITS"],
    TASK_DEFAULT:   ["CALLS", "CONTAINS", "IMPORTS"],
}

# ── ranking weights ───────────────────────────────────────────────────────────

# Manually chosen weights for ranking signals.
# TODO(ranking): replace with a learned ranker once labeled examples exist.
# Weights should sum to 1.0 for interpretability, but are normalized anyway.

WEIGHTS = {
    "semantic_score": 0.35,   # cosine similarity from vector search
    "lexical_score":  0.25,   # BM25 score
    "graph_distance": 0.25,   # proximity to seed symbols (closer = higher)
    "edge_type":      0.10,   # reward for high-value edge types
    "symbol_kind":    0.05,   # slight preference for functions/methods
}

# Reward multiplier per edge type — reflects typical relevance to user queries.
EDGE_TYPE_SCORES: dict[str, float] = {
    "CALLS":    1.0,
    "CONTAINS": 0.8,
    "IMPORTS":  0.6,
    "INHERITS": 0.5,
}

# Symbol kind preference.
SYMBOL_KIND_SCORES: dict[str, float] = {
    "function": 1.0,
    "method":   1.0,
    "class":    0.8,
    "module":   0.5,
}

# Graph distance decay: score = DISTANCE_DECAY ** hop_count
# Seeds (hop 0) score 1.0, 1-hop neighbors score 0.7, 2-hop score 0.49, etc.
DISTANCE_DECAY = 0.7

# ── result type ───────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    qualified_name: str
    kind: str
    file_path: str
    source: str
    score: float                        # final weighted score
    signals: dict = field(default_factory=dict)  # per-signal scores for observability
    hop: int = 0                        # graph distance from nearest seed


# ── internals ─────────────────────────────────────────────────────────────────

def detect_task(query: str) -> str:
    """
    Identify the task type from the query using keyword matching.
    Returns the first matching task type, or TASK_DEFAULT.

    TODO(task-detection): replace with a classifier or LLM call.
    """
    query_lower = query.lower()
    for task, keywords in TASK_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            return task
    return TASK_DEFAULT


def _expand(
    seeds: list[dict],
    edge_kinds: list[str],
    db_path: str,
    hops: int,
) -> dict[str, dict]:
    """
    Expand from seed symbols through the graph for `hops` levels.

    Returns a dict mapping qualified_name → {
        "hop": int,
        "edge_kind": str | None,   # edge type used to reach this node
        "via": str | None,         # qualified_name of the node that led here
    }

    Seeds are at hop 0. Their neighbors at hop 1, and so on.
    Already-visited nodes are not revisited.
    """
    visited: dict[str, dict] = {}

    for seed in seeds:
        name = seed["qualified_name"]
        if name not in visited:
            visited[name] = {"hop": 0, "edge_kind": None, "via": None}

    frontier = list(visited.keys())

    for hop in range(1, hops + 1):
        next_frontier = []
        for node in frontier:
            neighbors = get_neighbors(
                node,
                edge_kinds=edge_kinds,
                direction="both",
                db_path=db_path,
            )
            for n in neighbors:
                name = n["qualified_name"]
                if name not in visited:
                    visited[name] = {
                        "hop": hop,
                        "edge_kind": n["edge_kind"],
                        "via": node,
                    }
                    next_frontier.append(name)
        frontier = next_frontier

    return visited


def _normalize(scores: list[float]) -> list[float]:
    """Min-max normalize a list of scores to [0, 1]."""
    if not scores:
        return scores
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def _score(
    name: str,
    expansion_info: dict,
    seed_scores: dict[str, dict],
) -> tuple[float, dict]:
    """
    Compute the final score and per-signal breakdown for a candidate symbol.

    seed_scores: mapping from qualified_name → {"semantic": float, "lexical": float}
                 for symbols that appeared in the initial hybrid search.
    """
    info = expansion_info[name]
    hop = info["hop"]

    # semantic and lexical scores — non-zero only for seeds
    seed = seed_scores.get(name, {})
    semantic = seed.get("semantic", 0.0)
    lexical = seed.get("lexical", 0.0)

    # graph distance score
    distance_score = DISTANCE_DECAY ** hop

    # edge type score
    edge_kind = info.get("edge_kind")
    edge_score = EDGE_TYPE_SCORES.get(edge_kind, 0.5) if edge_kind else 1.0

    signals = {
        "semantic_score": semantic,
        "lexical_score": lexical,
        "graph_distance": distance_score,
        "edge_type": edge_score,
        "symbol_kind": 0.0,  # filled in after symbol fetch
    }

    total = sum(WEIGHTS[k] * v for k, v in signals.items())
    return total, signals


# ── public API ────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    db_path: str,
    top_k: int = 10,
    hops: int = 2,
    seed_k: int = 5,
    task: str | None = None,
) -> list[RetrievalResult]:
    """
    Retrieve the most relevant symbols for a query.

    Args:
        query:    Natural language query or code snippet.
        db_path:  Path to the retlib SQLite database.
        top_k:    Number of results to return.
        hops:     Graph expansion depth.
                  TODO(expansion): replace with budget-based expansion.
        seed_k:   Number of seed symbols from hybrid search.
        task:     Override task detection ('debug', 'refactor', 'understand').
                  If None, detected automatically from query.

    Returns:
        List of RetrievalResult sorted by score descending.
        Each result includes per-signal scores for observability.
    """
    # 1. detect task
    detected_task = task or detect_task(query)
    edge_kinds = TASK_EDGE_KINDS[detected_task]

    # 2. hybrid search for seeds
    seeds = hybrid_search(query, db_path, top_k=seed_k)

    if not seeds:
        return []

    # build seed score lookup
    seed_scores: dict[str, dict] = {
        s["qualified_name"]: {
            "semantic": s.get("vector_score", 0.0),
            "lexical": s.get("bm25_score", 0.0),
        }
        for s in seeds
    }

    # 3. graph expansion
    expansion = _expand(seeds, edge_kinds, db_path, hops)

    # 4. score all candidates
    candidates = []
    for name, info in expansion.items():
        raw_score, signals = _score(name, expansion, seed_scores)
        candidates.append((name, info["hop"], raw_score, signals))

    # 5. fetch symbol metadata and finalize scores
    results = []
    for name, hop, raw_score, signals in candidates:
        sym = get_symbol(name, db_path)
        if sym is None:
            continue  # symbol in graph but not in db (cross-file edge target)

        # fill in symbol kind score now that we have the symbol
        kind_score = SYMBOL_KIND_SCORES.get(sym["kind"], 0.5)
        signals["symbol_kind"] = kind_score
        final_score = sum(WEIGHTS[k] * v for k, v in signals.items())

        results.append(RetrievalResult(
            qualified_name=sym["qualified_name"],
            kind=sym["kind"],
            file_path=sym["file_path"],
            source=sym["source"],
            score=final_score,
            signals=signals,
            hop=hop,
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


def explain(result: RetrievalResult) -> str:
    """
    Return a human-readable explanation of why a symbol was retrieved.
    Useful for observability and debugging the retrieval pipeline.
    """
    lines = [
        f"Symbol:   {result.qualified_name} ({result.kind})",
        f"File:     {result.file_path}",
        f"Score:    {result.score:.4f}  (hop {result.hop} from seed)",
        f"Signals:",
    ]
    for signal, value in result.signals.items():
        weight = WEIGHTS.get(signal, 0.0)
        contribution = weight * value
        lines.append(f"  {signal:<20} {value:.3f} × {weight:.2f} = {contribution:.4f}")
    return "\n".join(lines)
