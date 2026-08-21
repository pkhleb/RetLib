import ast
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass
class Symbol:
    file_path: str
    name: str
    qualified_name: str  # e.g. MyClass.my_method
    kind: str  # module, class, function, method
    start_line: int
    end_line: int
    source: str
    docstring: Optional[str] = None

@dataclass
class Edge:
    source_qualified_name: str
    target_qualified_name: str
    kind: str  # CONTAINS, CALLS, IMPORTS, INHERITS

@dataclass
class ParseResult:
    file_path: str
    checksum: str
    symbols: list[Symbol] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

class SymbolVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str, source_lines: list[str]):
        self.file_path = file_path
        self.source_lines = source_lines
        self.symbols: list[Symbol] = []
        self.edges: list[Edge] = []
        self._scope_stack: list[str] = []

    def _current_scope(self) -> Optional[str]:
        return self._scope_stack[-1] if self._scope_stack else None

    def _qualified_name(self, name: str) -> str:
        if self._scope_stack:
            return f"{self._scope_stack[-1]}.{name}"
        return name

    def _get_source(self, node: ast.AST) -> str:
        lines = self.source_lines[node.lineno - 1:node.end_lineno]
        return "\n".join(lines)

    def _get_docstring(self, node: ast.AST) -> Optional[str]:
        return ast.get_docstring(node)

    def _current_kind(self) -> str:
        if self._scope_stack:
            for sym in self.symbols:
                if sym.qualified_name == self._scope_stack[-1] and sym.kind == "class":
                    return "method"
        return "function"

    def visit_ClassDef(self, node: ast.ClassDef):
        qualified_name = self._qualified_name(node.name)

        symbol = Symbol(
            file_path=self.file_path,
            name=node.name,
            qualified_name=qualified_name,
            kind="class",
            start_line=node.lineno,
            end_line=node.end_lineno,
            source=self._get_source(node),
            docstring=self._get_docstring(node),
        )
        self.symbols.append(symbol)

        parent = self._current_scope()
        if parent:
            self.edges.append(Edge(
                source_qualified_name=parent,
                target_qualified_name=qualified_name,
                kind="CONTAINS",
            ))

        for base in node.bases:
            if isinstance(base, ast.Name):
                self.edges.append(Edge(
                    source_qualified_name=qualified_name,
                    target_qualified_name=base.id,
                    kind="INHERITS",
                ))
            elif isinstance(base, ast.Attribute):
                self.edges.append(Edge(
                    source_qualified_name=qualified_name,
                    target_qualified_name=f"{base.value.id}.{base.attr}" if isinstance(base.value, ast.Name) else base.attr,
                    kind="INHERITS",
                ))

        self._scope_stack.append(qualified_name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        qualified_name = self._qualified_name(node.name)
        kind = self._current_kind()

        symbol = Symbol(
            file_path=self.file_path,
            name=node.name,
            qualified_name=qualified_name,
            kind=kind,
            start_line=node.lineno,
            end_line=node.end_lineno,
            source=self._get_source(node),
            docstring=self._get_docstring(node),
        )
        self.symbols.append(symbol)

        parent = self._current_scope()
        if parent:
            self.edges.append(Edge(
                source_qualified_name=parent,
                target_qualified_name=qualified_name,
                kind="CONTAINS",
            ))

        self._scope_stack.append(qualified_name)
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                called = self._resolve_call(child)
                if called:
                    self.edges.append(Edge(
                        source_qualified_name=qualified_name,
                        target_qualified_name=called,
                        kind="CALLS",
                    ))
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.visit(child)
        self._scope_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def _resolve_call(self, node: ast.Call) -> Optional[str]:
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                return f"{node.func.value.id}.{node.func.attr}"
            return node.func.attr
        return None

    def visit_Import(self, node: ast.Import):
        scope = self._current_scope()
        for alias in node.names:
            self.edges.append(Edge(
                source_qualified_name=scope or "__module__",
                target_qualified_name=alias.name,
                kind="IMPORTS",
            ))

    def visit_ImportFrom(self, node: ast.ImportFrom):
        scope = self._current_scope()
        module = node.module or ""
        for alias in node.names:
            target = f"{module}.{alias.name}" if module else alias.name
            self.edges.append(Edge(
                source_qualified_name=scope or "__module__",
                target_qualified_name=target,
                kind="IMPORTS",
            ))


def parse_file(file_path: str) -> ParseResult:
    path = Path(file_path)
    source = path.read_text(encoding="utf-8")
    checksum = hashlib.sha256(source.encode()).hexdigest()
    source_lines = source.splitlines()

    tree = ast.parse(source, filename=file_path)

    module_symbol = Symbol(
        file_path=file_path,
        name=path.stem,
        qualified_name=path.stem,
        kind="module",
        start_line=1,
        end_line=len(source_lines),
        source=source,
        docstring=ast.get_docstring(tree),
    )

    visitor = SymbolVisitor(file_path=file_path, source_lines=source_lines)
    visitor.visit(tree)

    return ParseResult(
        file_path=file_path,
        checksum=checksum,
        symbols=[module_symbol] + visitor.symbols,
        edges=visitor.edges,
    )


def parse_directory(root: str, skip_dirs: set[str] = None) -> list[ParseResult]:
    if skip_dirs is None:
        skip_dirs = {"__pycache__", ".git", ".venv", "node_modules"}

    results = []
    for path in Path(root).rglob("*.py"):
        if any(part in skip_dirs for part in path.parts):
            continue
        try:
            results.append(parse_file(str(path)))
        except Exception as e:
            print(f"[parser] skipped {path}: {e}")

    return results
