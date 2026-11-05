#!/usr/bin/env python3
"""Reject broad Python fallbacks that swallow ``MemoryError`` (angr-mx00p.132.66).

The Rust exploration glue under ``angr/exploration/`` wraps many calls in
``except Exception`` cat-(b)/(c) fallbacks that log and continue. Near an
address-space cap the exception they catch is often a ``MemoryError``, and
continuing turns it into a silently wrong state: a page left blank, a hook
result dropped (angr-mx00p.132.65). Out of memory must fail loud.

Detection, by ``ast`` over every ``*.py`` file in ``angr/exploration/``:

* an ``except`` handler that is bare, or names ``Exception`` /
  ``BaseException`` (alone or in a tuple), unless an earlier handler of the
  same ``try`` names ``MemoryError`` or the handler's last statement is a
  ``raise``;
* a ``suppress(...)`` call (``contextlib.suppress`` or a bare import) whose
  arguments name ``Exception`` / ``BaseException``. Use
  ``angr.exploration._fallback.suppress_recoverable`` instead, which
  re-raises ``MemoryError``.

The usual fix is an ``except MemoryError: raise`` clause ahead of the broad
one. A site is exempt when its line or the line above carries
``memoryerror-ok: <why>``. The site set is empty and must stay empty, so
there is no baseline.

Usage::

    tools/audit_memory_error_swallow.py               # report sites (exit 1 if any)
    tools/audit_memory_error_swallow.py --self-test   # prove the detector fires
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCAN = "angr/exploration"
MARKER = "memoryerror-ok:"
BROAD = {"Exception", "BaseException"}
# Fewer files than this means the glue moved out of the scan root.
MIN_FILES = 8


def _names(node: ast.expr | None) -> set[str]:
    if node is None:
        return set()
    if isinstance(node, ast.Tuple):
        return set().union(*(_names(e) for e in node.elts))
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        return {node.attr}
    return set()


def _is_suppress(call: ast.Call) -> bool:
    func = call.func
    return (isinstance(func, ast.Name) and func.id == "suppress") or (
        isinstance(func, ast.Attribute) and func.attr == "suppress"
    )


def find_sites(source: str) -> list[tuple[int, str]]:
    """(line, kind) for each unexempted swallowing site in ``source``."""
    lines = source.splitlines()

    def exempt(lineno: int) -> bool:
        return any(MARKER in lines[i - 1] for i in (lineno, lineno - 1) if 1 <= i <= len(lines))

    sites = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            guarded = False
            for handler in node.handlers:
                names = _names(handler.type)
                if "MemoryError" in names:
                    guarded = True
                    continue
                if handler.type is not None and not names & BROAD:
                    continue
                if guarded or isinstance(handler.body[-1], ast.Raise) or exempt(handler.lineno):
                    continue
                sites.append((handler.lineno, "except"))
        elif isinstance(node, ast.Call) and _is_suppress(node):
            if any(_names(a) & BROAD for a in node.args) and not exempt(node.lineno):
                sites.append((node.lineno, "suppress"))
    return sorted(sites)


def scan(repo_root: Path = REPO) -> list[tuple[str, int, str]]:
    files = sorted((repo_root / SCAN).glob("*.py"))
    if len(files) < MIN_FILES:
        raise SystemExit(f"audit_memory_error_swallow: only {len(files)} files under {SCAN}")
    out = []
    for path in files:
        rel = str(path.relative_to(repo_root))
        out.extend((rel, line, kind) for line, kind in find_sites(path.read_text()))
    return out


_FIRES = [
    "try:\n    f()\nexcept Exception:\n    pass\n",
    "try:\n    f()\nexcept:\n    pass\n",
    "try:\n    f()\nexcept (ValueError, BaseException) as e:\n    log(e)\n",
    "try:\n    f()\nexcept Exception:\n    pass\nexcept MemoryError:\n    raise\n",
    "with contextlib.suppress(Exception):\n    f()\n",
    "with suppress(KeyError, Exception):\n    f()\n",
]
_QUIET = [
    "try:\n    f()\nexcept MemoryError:\n    raise\nexcept Exception:\n    pass\n",
    "try:\n    f()\nexcept (MemoryError, KeyError):\n    raise\nexcept Exception:\n    pass\n",
    "try:\n    f()\nexcept Exception as e:\n    log(e)\n    raise\n",
    "try:\n    f()\nexcept Exception as e:\n    raise RuntimeError from e\n",
    "try:\n    f()\nexcept ValueError:\n    pass\n",
    "try:\n    f()\n# memoryerror-ok: test\nexcept Exception:\n    pass\n",
    "with contextlib.suppress(KeyError):\n    f()\n",
    "with suppress_recoverable():\n    f()\n",
]


def self_test() -> int:
    bad = [s for s in _FIRES if not find_sites(s)] + [s for s in _QUIET if find_sites(s)]
    for s in bad:
        print(f"self-test mismatch:\n{s}")
    if not bad:
        print(f"audit_memory_error_swallow self-test OK ({len(_FIRES)} fire, {len(_QUIET)} quiet)")
    return 1 if bad else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--self-test", action="store_true")
    if parser.parse_args().self_test:
        return self_test()
    sites = scan()
    for rel, line, kind in sites:
        print(f"{rel}:{line}: broad {kind} swallows MemoryError")
    if sites:
        print(f"\n{len(sites)} site(s); add `except MemoryError: raise` ahead, or use suppress_recoverable")
        return 1
    print("audit_memory_error_swallow OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
