import textwrap
import tempfile
import os
import pytest
from retlib.parser import parse_file, parse_directory, Symbol, Edge

# ── helpers ──────────────────────────────────────────────────────────────────

def write_temp(source: str, suffix=".py") -> str:
    """Write source to a temp file and return the path."""
    source = textwrap.dedent(source)
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    f.write(source)
    f.close()
    return f.name

def symbols_by_kind(result, kind):
    return [s for s in result.symbols if s.kind == kind]

def edges_by_kind(result, kind):
    return [e for e in result.edges if e.kind == kind]

def find_symbol(result, qualified_name):
    for s in result.symbols:
        if s.qualified_name == qualified_name:
            return s
    return None

def find_edge(result, source, target, kind):
    for e in result.edges:
        if e.source_qualified_name == source and e.target_qualified_name == target and e.kind == kind:
            return e
    return None

# ── symbol extraction ─────────────────────────────────────────────────────────

def test_module_symbol_always_present():
    path = write_temp("x = 1\n")
    try:
        result = parse_file(path)
        modules = symbols_by_kind(result, "module")
        assert len(modules) == 1
    finally:
        os.unlink(path)

def test_function_extraction():
    path = write_temp("""
        def foo():
            pass

        def bar():
            pass
    """)
    try:
        result = parse_file(path)
        funcs = {s.qualified_name for s in symbols_by_kind(result, "function")}
        assert "foo" in funcs
        assert "bar" in funcs
    finally:
        os.unlink(path)

def test_class_extraction():
    path = write_temp("""
        class MyClass:
            pass
    """)
    try:
        result = parse_file(path)
        classes = symbols_by_kind(result, "class")
        assert len(classes) == 1
        assert classes[0].qualified_name == "MyClass"
    finally:
        os.unlink(path)

def test_method_extraction():
    path = write_temp("""
        class MyClass:
            def my_method(self):
                pass
    """)
    try:
        result = parse_file(path)
        methods = symbols_by_kind(result, "method")
        assert len(methods) == 1
        assert methods[0].qualified_name == "MyClass.my_method"
    finally:
        os.unlink(path)

def test_method_vs_function_distinction():
    path = write_temp("""
        def standalone():
            pass

        class MyClass:
            def method(self):
                pass
    """)
    try:
        result = parse_file(path)
        funcs = {s.qualified_name for s in symbols_by_kind(result, "function")}
        methods = {s.qualified_name for s in symbols_by_kind(result, "method")}
        assert "standalone" in funcs
        assert "MyClass.method" in methods
        assert "MyClass.method" not in funcs
        assert "standalone" not in methods
    finally:
        os.unlink(path)

def test_nested_class_method():
    path = write_temp("""
        class Outer:
            class Inner:
                def inner_method(self):
                    pass
    """)
    try:
        result = parse_file(path)
        method = find_symbol(result, "Outer.Inner.inner_method")
        assert method is not None
        assert method.kind == "method"
    finally:
        os.unlink(path)

def test_async_function_extracted():
    path = write_temp("""
        async def async_func():
            pass
    """)
    try:
        result = parse_file(path)
        funcs = {s.qualified_name for s in symbols_by_kind(result, "function")}
        assert "async_func" in funcs
    finally:
        os.unlink(path)

def test_line_numbers():
    path = write_temp("""
        def foo():
            pass

        def bar():
            pass
    """)
    try:
        result = parse_file(path)
        foo = find_symbol(result, "foo")
        bar = find_symbol(result, "bar")
        assert foo is not None and bar is not None
        assert foo.start_line < bar.start_line
    finally:
        os.unlink(path)

def test_docstring_extraction():
    path = write_temp("""
        def foo():
            \"\"\"This is a docstring.\"\"\"
            pass
    """)
    try:
        result = parse_file(path)
        foo = find_symbol(result, "foo")
        assert foo is not None
        assert foo.docstring == "This is a docstring."
    finally:
        os.unlink(path)

def test_no_docstring_is_none():
    path = write_temp("""
        def foo():
            pass
    """)
    try:
        result = parse_file(path)
        foo = find_symbol(result, "foo")
        assert foo is not None
        assert foo.docstring is None
    finally:
        os.unlink(path)

def test_source_span_captured():
    path = write_temp("""
        def foo():
            x = 1
            return x
    """)
    try:
        result = parse_file(path)
        foo = find_symbol(result, "foo")
        assert foo is not None
        assert "def foo" in foo.source
        assert "return x" in foo.source
    finally:
        os.unlink(path)

def test_checksum_is_stable():
    source = "def foo():\n    pass\n"
    path = write_temp(source)
    try:
        r1 = parse_file(path)
        r2 = parse_file(path)
        assert r1.checksum == r2.checksum
    finally:
        os.unlink(path)

def test_checksum_changes_with_content():
    path = write_temp("def foo():\n    pass\n")
    try:
        r1 = parse_file(path)
        with open(path, "w") as f:
            f.write("def foo():\n    return 1\n")
        r2 = parse_file(path)
        assert r1.checksum != r2.checksum
    finally:
        os.unlink(path)

# ── edges ─────────────────────────────────────────────────────────────────────

def test_contains_edge_class_to_method():
    path = write_temp("""
        class MyClass:
            def my_method(self):
                pass
    """)
    try:
        result = parse_file(path)
        edge = find_edge(result, "MyClass", "MyClass.my_method", "CONTAINS")
        assert edge is not None
    finally:
        os.unlink(path)

def test_calls_edge():
    path = write_temp("""
        def helper():
            pass

        def main():
            helper()
    """)
    try:
        result = parse_file(path)
        edge = find_edge(result, "main", "helper", "CALLS")
        assert edge is not None
    finally:
        os.unlink(path)

def test_calls_attribute_method():
    path = write_temp("""
        def foo(obj):
            obj.bar()
    """)
    try:
        result = parse_file(path)
        edge = find_edge(result, "foo", "obj.bar", "CALLS")
        assert edge is not None
    finally:
        os.unlink(path)

def test_imports_edge_plain():
    path = write_temp("""
        import os
    """)
    try:
        result = parse_file(path)
        edge = find_edge(result, "__module__", "os", "IMPORTS")
        assert edge is not None
    finally:
        os.unlink(path)

def test_imports_edge_from():
    path = write_temp("""
        from pathlib import Path
    """)
    try:
        result = parse_file(path)
        edge = find_edge(result, "__module__", "pathlib.Path", "IMPORTS")
        assert edge is not None
    finally:
        os.unlink(path)

def test_inherits_edge():
    path = write_temp("""
        class Base:
            pass

        class Child(Base):
            pass
    """)
    try:
        result = parse_file(path)
        edge = find_edge(result, "Child", "Base", "INHERITS")
        assert edge is not None
    finally:
        os.unlink(path)

def test_inherits_dotted_base():
    path = write_temp("""
        import ast

        class MyVisitor(ast.NodeVisitor):
            pass
    """)
    try:
        result = parse_file(path)
        edge = find_edge(result, "MyVisitor", "ast.NodeVisitor", "INHERITS")
        assert edge is not None
    finally:
        os.unlink(path)

# ── parse_directory ───────────────────────────────────────────────────────────

def test_parse_directory_finds_py_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        for name in ["a.py", "b.py"]:
            with open(os.path.join(tmpdir, name), "w") as f:
                f.write("x = 1\n")
        results = parse_directory(tmpdir)
        paths = {r.file_path for r in results}
        assert any("a.py" in p for p in paths)
        assert any("b.py" in p for p in paths)

def test_parse_directory_skips_pycache():
    with tempfile.TemporaryDirectory() as tmpdir:
        pycache = os.path.join(tmpdir, "__pycache__")
        os.makedirs(pycache)
        with open(os.path.join(pycache, "cached.py"), "w") as f:
            f.write("x = 1\n")
        with open(os.path.join(tmpdir, "real.py"), "w") as f:
            f.write("x = 1\n")
        results = parse_directory(tmpdir)
        paths = {r.file_path for r in results}
        assert not any("__pycache__" in p for p in paths)
        assert any("real.py" in p for p in paths)

def test_parse_directory_skips_bad_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "good.py"), "w") as f:
            f.write("x = 1\n")
        with open(os.path.join(tmpdir, "bad.py"), "w") as f:
            f.write("def (:\n")  # syntax error
        results = parse_directory(tmpdir)
        paths = {r.file_path for r in results}
        assert any("good.py" in p for p in paths)
        assert not any("bad.py" in p for p in paths)

def test_parse_directory_recurses():
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, "sub")
        os.makedirs(subdir)
        with open(os.path.join(subdir, "deep.py"), "w") as f:
            f.write("x = 1\n")
        results = parse_directory(tmpdir)
        paths = {r.file_path for r in results}
        assert any("deep.py" in p for p in paths)
