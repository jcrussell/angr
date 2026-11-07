#!/usr/bin/env python3
"""Reject untagged broad Python fallbacks in ``angr/exploration/`` (angr-vmi2l.270).

A broad handler that logs and continues is one of three things, and the
reader of the site needs to know which: cat-(a) EXPECTED CONTROL FLOW (a
planned absorb), cat-(b) FALLBACK WITH LOSS (the caller takes a stale or
default value and execution degrades), or cat-(c) WRONG-ANSWER RISK (the
result may be wrong downstream, so the site logs a warning). The Rust side
spells this ``// SILENT(cat-…)`` and ``audit_silent_fallback.py`` gates it;
this is the Python half.

Detection, by ``ast`` over every ``*.py`` file in ``angr/exploration/``:

* an ``except`` handler that is bare, or names ``Exception`` /
  ``BaseException`` (alone or in a tuple), unless its last statement is a
  ``raise``;
* a ``with suppress_recoverable()`` block, or a ``suppress(...)`` naming
  ``Exception`` / ``BaseException`` (``audit_memory_error_swallow.py``
  rejects the latter on its own).

A site is tagged when ``cat-(a)``, ``cat-(b)`` or ``cat-(c)`` (the parentheses
optional) appears on one of its lines (the ``except`` handler or ``with``
block), or in the comment block directly above its ``except`` / ``with``
line, or directly above the handler's ``try:``. The site set is empty and
must stay empty, so there is no baseline.

Usage::

    tools/audit_fallback_tag_py.py               # report sites (exit 1 if any)
    tools/audit_fallback_tag_py.py --self-test   # prove the detector fires
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCAN = "angr/exploration"
TAG = re.compile(r"cat-\(?[abc]\)?")
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


def _call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def find_sites(source: str) -> list[tuple[int, str]]:
    """(line, kind) for each untagged broad fallback in ``source``."""
    lines = source.splitlines()

    def tag_on(lineno: int) -> bool:
        return 1 <= lineno <= len(lines) and bool(TAG.search(lines[lineno - 1]))

    def tag_above(lineno: int) -> bool:
        i = lineno - 1
        while i >= 1 and lines[i - 1].lstrip().startswith("#"):
            if tag_on(i):
                return True
            i -= 1
        return False

    def tagged(start: int, end: int, *anchors: int) -> bool:
        return any(tag_on(i) for i in range(start, end + 1)) or any(tag_above(a) for a in (start, *anchors))

    sites = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            for handler in node.handlers:
                if handler.type is not None and not _names(handler.type) & BROAD:
                    continue
                if isinstance(handler.body[-1], ast.Raise):
                    continue
                if not tagged(handler.lineno, handler.end_lineno, node.lineno):
                    sites.append((handler.lineno, "except"))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                call = item.context_expr
                if not isinstance(call, ast.Call):
                    continue
                name = _call_name(call)
                broad = name == "suppress_recoverable" or (
                    name == "suppress" and any(_names(a) & BROAD for a in call.args)
                )
                if broad and not tagged(node.lineno, node.end_lineno):
                    sites.append((node.lineno, name))
    return sorted(sites)


def scan(repo_root: Path = REPO) -> list[tuple[str, int, str]]:
    files = sorted((repo_root / SCAN).glob("*.py"))
    if len(files) < MIN_FILES:
        raise SystemExit(f"audit_fallback_tag_py: only {len(files)} files under {SCAN}")
    out = []
    for path in files:
        rel = str(path.relative_to(repo_root))
        out.extend((rel, line, kind) for line, kind in find_sites(path.read_text()))
    return out


_FIRES = [
    "try:\n    f()\nexcept Exception:\n    pass\n",
    "try:\n    f()\nexcept:\n    pass\n",
    "try:\n    f()\nexcept (ValueError, BaseException) as e:\n    log(e)\n",
    "try:\n    f()\nexcept MemoryError:\n    raise\nexcept Exception:\n    pass\n",
    "with suppress_recoverable():\n    f()\n",
    "with contextlib.suppress(KeyError, Exception):\n    f()\n",
    # A tag on an unrelated comment two lines up, past code, does not count.
    "# cat-(a) EXPECTED CONTROL FLOW: x\ng()\ntry:\n    f()\nexcept Exception:\n    pass\n",
    # A tag in a sibling handler does not cover this one.
    "try:\n    f()\nexcept KeyError:\n    pass  # cat-(a)\nexcept Exception:\n    pass\n",
]
_QUIET = [
    "try:\n    f()\nexcept Exception:\n    # cat-(b) FALLBACK WITH LOSS: x\n    pass\n",
    "try:\n    f()\nexcept Exception:  # cat-(a) EXPECTED CONTROL FLOW: x\n    pass\n",
    "try:\n    f()\n# cat-(c) WRONG-ANSWER RISK: x\nexcept Exception:\n    pass\n",
    "# cat-(b) FALLBACK WITH LOSS: x\n# more\ntry:\n    f()\nexcept Exception:\n    pass\n",
    "try:\n    f()\nexcept Exception as e:\n    log(e)\n    raise\n",
    "try:\n    f()\nexcept ValueError:\n    pass\n",
    "# cat-a: x\nwith suppress_recoverable():\n    f()\n",
    "with suppress_recoverable():\n    f()  # cat-(b) FALLBACK WITH LOSS: x\n",
    "with contextlib.suppress(KeyError):\n    f()\n",
]


def self_test() -> int:
    bad = [s for s in _FIRES if not find_sites(s)] + [s for s in _QUIET if find_sites(s)]
    for s in bad:
        print(f"self-test mismatch:\n{s}")
    if not bad:
        print(f"audit_fallback_tag_py self-test OK ({len(_FIRES)} fire, {len(_QUIET)} quiet)")
    return 1 if bad else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--self-test", action="store_true")
    if parser.parse_args().self_test:
        return self_test()
    sites = scan()
    for rel, line, kind in sites:
        print(f"{rel}:{line}: broad {kind} fallback has no cat-(a|b|c) tag")
    if sites:
        print(f"\n{len(sites)} site(s); tag each with `# cat-(a|b|c) <LABEL>: <why>`")
        return 1
    print("audit_fallback_tag_py OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
