"""angr/rustylib/vex_engine.pyi matches the built extension's surface (angr-vmi2l.282).

The stub is hand-maintained, so nothing else notices when a ``#[pymethods]``
method, ``#[pyfunction]`` or ``#[getter]`` is added, removed or re-signatured.
This diffs every public name, its kind (method, staticmethod, property, class
attribute) and its parameter names and literal defaults against
``inspect.signature`` of the imported module. A failure lists stub lines to
paste for what the stub lacks.
"""

from __future__ import annotations

import ast
import inspect
import os

import pytest

from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration engine not built")

STUB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "angr",
    "rustylib",
    "vex_engine.pyi",
)

# Module attributes the stub deliberately leaves out: the test-only hook module
# (its header says so). Dunders other than __version__ are skipped by name.
UNSTUBBED = {"_testing"}


class _Required:
    def __repr__(self):
        return "<required>"


_NO_DEFAULT = _Required()


def _stub_params(fn: ast.FunctionDef) -> list[tuple[str, object]]:
    a = fn.args
    positional = a.posonlyargs + a.args
    defaults = [_NO_DEFAULT] * (len(positional) - len(a.defaults)) + list(a.defaults)
    out = []
    for arg, default in zip(positional, defaults):
        out.append((arg.arg, default))
    if a.vararg:
        out.append(("*" + a.vararg.arg, _NO_DEFAULT))
    for arg, default in zip(a.kwonlyargs, a.kw_defaults):
        out.append((arg.arg, _NO_DEFAULT if default is None else default))
    if a.kwarg:
        out.append(("**" + a.kwarg.arg, _NO_DEFAULT))
    if out and out[0][0] in ("self", "cls"):
        out = out[1:]
    return [(n, d if d is _NO_DEFAULT else ast.literal_eval(d)) for n, d in out]


def _decorators(fn: ast.FunctionDef) -> set[str]:
    return {
        d.id if isinstance(d, ast.Name) else d.attr
        for d in fn.decorator_list
        if isinstance(d, ast.Name | ast.Attribute)
    }


def _member_kind(fn: ast.FunctionDef) -> str | None:
    decorators = _decorators(fn)
    if "setter" in decorators:
        return None
    if "property" in decorators:
        return "property"
    if "staticmethod" in decorators:
        return "staticmethod"
    if "classmethod" in decorators:
        return "classmethod"
    return "method"


def _parse_stub():
    with open(STUB, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    stub: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            members: dict[str, tuple[str, list | None]] = {}
            for b in node.body:
                if isinstance(b, ast.FunctionDef):
                    kind = _member_kind(b)
                    if kind is not None:
                        members[b.name] = (kind, None if kind == "property" else _stub_params(b))
                elif isinstance(b, ast.AnnAssign):
                    members[b.target.id] = ("attr", None)
            stub[node.name] = members
        elif isinstance(node, ast.FunctionDef):
            stub[node.name] = ("function", _stub_params(node))
        elif isinstance(node, ast.AnnAssign):
            stub[node.target.id] = ("attr", None)
    return stub


def _runtime_params(obj) -> list[tuple[str, object]] | None:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    out = []
    for p in sig.parameters.values():
        if p.name in ("self", "cls", "$self", "$cls"):
            continue
        prefix = {p.VAR_POSITIONAL: "*", p.VAR_KEYWORD: "**"}.get(p.kind, "")
        out.append((prefix + p.name, _NO_DEFAULT if p.default is p.empty else p.default))
    return out


def _same_params(stub_params: list, runtime_params: list | None) -> bool:
    """PyO3 renders a default it cannot spell as a literal (``Vec::new()``) as
    ``...``, which matches any stub default."""
    if runtime_params is None:
        return True
    if len(stub_params) != len(runtime_params):
        return False
    for (sn, sd), (rn, rd) in zip(stub_params, runtime_params):
        if sn != rn or (sd is _NO_DEFAULT) != (rd is _NO_DEFAULT):
            return False
        if rd is not Ellipsis and sd != rd:
            return False
    return True


def _runtime_kind(cls: type, name: str) -> str:
    raw = inspect.getattr_static(cls, name)
    if isinstance(raw, staticmethod):
        return "staticmethod"
    if isinstance(raw, classmethod):
        return "classmethod"
    if inspect.isdatadescriptor(raw):
        return "property"
    if callable(raw):
        return "method"
    return "attr"


def _suggest(name: str, kind: str, params: list | None, indent: str = "") -> str:
    def render(ps):
        parts = [] if kind in ("staticmethod", "function") else ["self"]
        for n, d in ps or []:
            if d is _NO_DEFAULT:
                parts.append(f"{n}: Any")
            else:
                parts.append(f"{n}: Any = {'...' if d is Ellipsis else repr(d)}")
        return ", ".join(parts)

    if kind == "attr":
        return f"{indent}{name}: Any"
    if kind == "property":
        return f"{indent}@property\n{indent}def {name}(self) -> Any: ..."
    deco = f"{indent}@staticmethod\n" if kind == "staticmethod" else ""
    return f"{deco}{indent}def {name}({render(params)}) -> Any: ..."


def _drift() -> list[str]:
    import angr.rustylib.vex_engine as ve  # pylint: disable=import-outside-toplevel

    stub = _parse_stub()
    problems: list[str] = []
    runtime = {n for n in dir(ve) if not n.startswith("__") or n == "__version__"} - UNSTUBBED
    for name in sorted(runtime - set(stub)):
        obj = getattr(ve, name)
        if isinstance(obj, type):
            problems.append(f"stub lacks class {name}")
        elif callable(obj):
            problems.append(f"stub lacks:\n{_suggest(name, 'function', _runtime_params(obj))}")
        else:
            problems.append(f"stub lacks:\n{_suggest(name, 'attr', None)}")
    for name in sorted(set(stub) - runtime):
        problems.append(f"stub names {name}, which the module does not have")

    for name in sorted(runtime & set(stub)):
        obj = getattr(ve, name)
        entry = stub[name]
        if not isinstance(obj, type):
            kind, params = entry
            if callable(obj) and not _same_params(params, _runtime_params(obj)):
                problems.append(f"{name}: stub params {params} vs runtime {_runtime_params(obj)}")
            continue
        if not isinstance(entry, dict):
            problems.append(f"{name}: a class at runtime but not in the stub")
            continue
        own = {m for m in vars(obj) if not m.startswith("__")}
        if issubclass(obj, BaseException):
            own -= set(vars(BaseException))
        for m in sorted(own - set(entry)):
            kind = _runtime_kind(obj, m)
            params = None if kind in ("property", "attr") else _runtime_params(getattr(obj, m))
            problems.append(f"{name} stub lacks:\n{_suggest(m, kind, params, '    ')}")
        for m in sorted(set(entry) - own - {"__init__"}):
            problems.append(f"{name}.{m} is in the stub but not on the runtime class")
        for m in sorted(own & set(entry)):
            stub_kind, stub_params = entry[m]
            kind = _runtime_kind(obj, m)
            if stub_kind != kind:
                problems.append(f"{name}.{m}: stub says {stub_kind}, runtime is {kind}")
                continue
            if kind in ("property", "attr"):
                continue
            rt = _runtime_params(getattr(obj, m))
            if not _same_params(stub_params, rt):
                problems.append(f"{name}.{m}: stub params {stub_params} vs runtime {rt}")
        if "__init__" in entry:
            rt = _runtime_params(obj)
            if not _same_params(entry["__init__"][1], rt):
                problems.append(f"{name}.__init__: stub params {entry['__init__'][1]} vs runtime {rt}")
    return problems


def test_stub_matches_the_built_extension():
    problems = _drift()
    assert not problems, "vex_engine.pyi drifted from the extension:\n" + "\n".join(problems)
