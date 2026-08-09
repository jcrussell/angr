#!/usr/bin/env python3
"""Shared source-scanning helpers for the ``tools/audit_*.py`` baseline lints.

The audit scripts in this directory all do the same two boring things before
they can look for their own pattern: strip comments/string literals so prose
cannot be mistaken for code, and skip Rust test modules. Keeping one copy here
means a fix to the (deliberately simple) blanker benefits every lint.

Pure stdlib, no repo imports — the audit scripts run standalone in CI.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["blank_noise", "enclosing_fn", "is_test_file"]

import re

_FN_RE = re.compile(r"\bfn\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*[(<]")


def blank_noise(src: str) -> str:
    """Replace comments and string/char literals with spaces, preserving offsets.

    Keeps line/column arithmetic intact so reported line numbers stay true,
    while stopping a ``//`` comment or a doc block from being parsed as code.
    """
    out = list(src)
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                out[i] = " "
                i += 1
        elif c == "/" and i + 1 < n and src[i + 1] == "*":
            depth, start = 1, i
            i += 2
            while i < n and depth:
                if src.startswith("/*", i):
                    depth += 1
                    i += 2
                elif src.startswith("*/", i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
            for j in range(start, i):
                if src[j] != "\n":
                    out[j] = " "
        elif c == '"':
            start = i
            i += 1
            while i < n and src[i] != '"':
                i += 2 if src[i] == "\\" else 1
            i = min(i + 1, n)
            for j in range(start, i):
                if src[j] != "\n":
                    out[j] = " "
        else:
            i += 1
    return "".join(out)


def is_test_file(path: Path | str) -> bool:
    """True for the repo's Rust test-module naming conventions."""
    rel = Path(path).as_posix()
    name = Path(path).name
    return (
        name.endswith("_tests.rs")
        or name.startswith("tests_")
        or name in ("test_helpers.rs", "property_tests.rs")
        or "/tests/" in rel
        or rel.startswith("tests/")
    )


def enclosing_fn(src: str, pos: int) -> str | None:
    """Name of the nearest preceding ``fn`` declaration before ``pos``.

    An approximation — it does not track brace depth, so a hit in a module's
    top-level (outside any fn) is attributed to the last fn above it. Adequate
    for the audit lints, which only use the name to build a stable,
    line-number-independent baseline key.
    """
    last = None
    for m in _FN_RE.finditer(src, 0, pos):
        last = m
    return last.group("name") if last else None
