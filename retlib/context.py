"""
context.py — Token-budget-aware context assembly from retrieved symbols.

Takes a ranked list of RetrievalResult objects and builds a formatted
context string to pass to the LLM, respecting a token budget.

Token counting:
    Estimated as len(text) // 4 — a standard approximation.
    NOTE: this will not match the exact token count of any specific model.
    True token count is available post-hoc from the LLM API response
    (response.usage.prompt_tokens) and should be used to calibrate the
    budget constant over time.

Assembly order:
    1. Symbols are taken in ranked order (highest score first).
    2. For each symbol, we include: header, relationship summary, source.
    3. We stop when the next symbol would exceed the budget.
    4. A budget summary is appended at the end for observability.
"""

from dataclasses import dataclass, field
from retlib.graph import get_neighbors
from retlib.retriever import RetrievalResult

DEFAULT_TOKEN_BUDGET = 4096


# ── token estimation ──────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """
    Estimate token count as len(text) // 4.
    Standard approximation — actual count varies by model and tokenizer.
    """
    return len(text) // 4


# ── relationship summary ──────────────────────────────────────────────────────

def _relationship_summary(
    qualified_name: str,
    db_path: str,
    included_names: set[str],
) -> str:
    """
    Build a one-line relationship summary for a symbol.
    Only references symbols that are also included in the context, to avoid
    confusing the LLM with names it has no source for.
    """
    lines = []

    callers = get_neighbors(
        qualified_name, edge_kinds=["CALLS"], direction="in", db_path=db_path
    )
    caller_names = [n["qualified_name"] for n in callers if n["qualified_name"] in included_names]
    if caller_names:
        lines.append(f"Called by: {', '.join(caller_names)}")

    callees = get_neighbors(
        qualified_name, edge_kinds=["CALLS"], direction="out", db_path=db_path
    )
    callee_names = [n["qualified_name"] for n in callees if n["qualified_name"] in included_names]
    if callee_names:
        lines.append(f"Calls: {', '.join(callee_names)}")

    parents = get_neighbors(
        qualified_name, edge_kinds=["CONTAINS"], direction="in", db_path=db_path
    )
    parent_names = [n["qualified_name"] for n in parents if n["qualified_name"] in included_names]
    if parent_names:
        lines.append(f"Contained in: {', '.join(parent_names)}")

    bases = get_neighbors(
        qualified_name, edge_kinds=["INHERITS"], direction="out", db_path=db_path
    )
    base_names = [n["qualified_name"] for n in bases if n["qualified_name"] in included_names]
    if base_names:
        lines.append(f"Inherits from: {', '.join(base_names)}")

    return " | ".join(lines) if lines else ""


# ── context block ─────────────────────────────────────────────────────────────

def _format_symbol_block(
    result: RetrievalResult,
    db_path: str,
    included_names: set[str],
) -> str:
    """
    Format a single symbol as a context block.

    Structure:
        === qualified_name (kind) ===
        File: file_path  |  Score: 0.xxxx  |  Hop: N
        [relationship summary if any]

        <source>
    """
    header = f"=== {result.qualified_name} ({result.kind}) ==="
    meta = f"File: {result.file_path}  |  Score: {result.score:.4f}  |  Hop: {result.hop}"

    rel = _relationship_summary(result.qualified_name, db_path, included_names)
    rel_line = f"Relationships: {rel}" if rel else ""

    parts = [header, meta]
    if rel_line:
        parts.append(rel_line)
    parts.append("")  # blank line before source
    parts.append(result.source)

    return "\n".join(parts)


# ── assembly ──────────────────────────────────────────────────────────────────

@dataclass
class ContextResult:
    text: str                          # the assembled context string
    included: list[str]                # qualified names of included symbols
    excluded: list[str]                # qualified names dropped due to budget
    estimated_tokens: int              # estimated token count of text
    budget: int                        # token budget used
    signals: dict = field(default_factory=dict)  # per-symbol scores for observability


def build_context(
    results: list[RetrievalResult],
    db_path: str,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> ContextResult:
    """
    Assemble a context string from ranked retrieval results within a token budget.

    Symbols are included in rank order until the budget is exhausted.
    Relationship summaries only reference other included symbols.

    Args:
        results:       Ranked list from retriever.retrieve().
        db_path:       Path to the retlib SQLite database.
        token_budget:  Max estimated tokens for the context block.

    Returns:
        ContextResult with the assembled text and observability metadata.
    """
    if not results:
        return ContextResult(
            text="",
            included=[],
            excluded=[],
            estimated_tokens=0,
            budget=token_budget,
        )

    # first pass: greedily select symbols within budget
    # we do two passes so relationship summaries can reference all included symbols
    included_results: list[RetrievalResult] = []
    excluded_names: list[str] = []
    tokens_used = 0

    for result in results:
        # estimate cost of this block (header + meta + source, no relationships yet)
        block_estimate = estimate_tokens(
            f"=== {result.qualified_name} ({result.kind}) ===\n"
            f"File: {result.file_path}\n\n"
            f"{result.source}\n\n"
        )
        if tokens_used + block_estimate <= token_budget:
            included_results.append(result)
            tokens_used += block_estimate
        else:
            excluded_names.append(result.qualified_name)

    included_names = {r.qualified_name for r in included_results}

    # second pass: format blocks with relationship summaries
    blocks = []
    for result in included_results:
        blocks.append(_format_symbol_block(result, db_path, included_names))

    # budget summary footer
    footer_lines = [
        "",
        "---",
        f"Context: {len(included_results)} symbols included",
        f"Estimated tokens: {tokens_used} / {token_budget}",
    ]
    if excluded_names:
        footer_lines.append(
            f"Dropped (budget): {', '.join(excluded_names)}"
        )

    footer = "\n".join(footer_lines)
    full_text = "\n\n".join(blocks) + footer

    return ContextResult(
        text=full_text,
        included=list(included_names),
        excluded=excluded_names,
        estimated_tokens=estimate_tokens(full_text),
        budget=token_budget,
        signals={r.qualified_name: r.signals for r in included_results},
    )


def summarize(context_result: ContextResult) -> str:
    """
    Return a compact summary of what was included and why.
    Useful for observability without printing the full context.
    """
    lines = [
        f"Token budget: {context_result.estimated_tokens} / {context_result.budget} used",
        f"Included ({len(context_result.included)}): {', '.join(context_result.included)}",
    ]
    if context_result.excluded:
        lines.append(f"Excluded ({len(context_result.excluded)}): {', '.join(context_result.excluded)}")
    return "\n".join(lines)
