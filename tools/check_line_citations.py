#!/usr/bin/env python3
"""Baseline-audited guard against raw line-number citations in Rust comments.

angr-9ke6b (the pre-PR audit of native/angr/src/) turned up a recurring bug
class: doc comments and inline comments that cite an EXACT line number in
another file (``arch/mod.rs:638``) or in the same file (``line ~478``) as
part of an invariant, cross-reference, or "see also" note -- including a
widespread and otherwise-valuable convention of citing the upstream Python
implementation's line numbers (``engines/successors.py:203``) to document
behavior parity. Those citations drift silently as the cited code moves --
CLAUDE.md's bd-memory "symbol-anchor convention" exists for exactly this
failure mode, but only covers bd memory authoring, not in-source comments.

Mirrors tools/audit_silent_fallback.py's shape rather than diffing against a
base ref: a diff-scoped version would need to pick a base branch, and this
repo's PRs land against `rust-symex` today but may eventually target `master`
directly for the final upstream merge, at which point a base-ref diff would
dump the branch's entire multi-year backlog of (mostly legitimate) citations
onto one unlucky PR with no escape hatch. Scanning the whole tree against a
checked-in baseline is base-ref-independent and self-documenting: existing
citations are grandfathered once, and the baseline should only shrink as they
get converted to symbol references.

Usage::

    tools/check_line_citations.py                 # report new un-baselined citations
    tools/check_line_citations.py --list           # list ALL detected citations
    tools/check_line_citations.py --update-baseline

Exit 0 when there are no new citations (or on --list / --update-baseline),
1 when new ones are found. Pure stdlib.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "native" / "angr" / "src"
BASELINE_PATH = REPO_ROOT / "tools" / "line_citations_baseline.txt"

# file.rs:123 / some/path/foo.py:45 -- cite-another-location-by-line-number.
FILE_LINE_RE = re.compile(r"[\w./-]+\.(?:rs|py):\d+")
# "line 638" / "lines ~478-490" -- same shape without a filename.
LINE_WORD_RE = re.compile(r"\blines?\s*~?\d+\b", re.IGNORECASE)


def _iter_rs_files() -> list[Path]:
    return sorted(SRC_ROOT.rglob("*.rs"))


def _key(rel: str, stripped: str) -> str:
    """Line-number-independent baseline key so unrelated edits don't churn it."""
    return f"{rel}\t{stripped}"


def scan() -> list[tuple[str, int, str]]:
    """Return (rel, lineno, stripped) for every citation-shaped comment line."""
    hits: list[tuple[str, int, str]] = []
    for path in _iter_rs_files():
        rel = path.relative_to(SRC_ROOT).as_posix()
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for idx, raw in enumerate(lines):
            stripped = raw.strip()
            if not stripped.startswith("//"):
                continue
            if FILE_LINE_RE.search(stripped) or LINE_WORD_RE.search(stripped):
                hits.append((rel, idx + 1, stripped))
    return hits


def load_baseline() -> set[str]:
    if not BASELINE_PATH.exists():
        return set()
    keys = set()
    for line in BASELINE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        keys.add(line)
    return keys


def write_baseline(hits: list[tuple[str, int, str]]) -> int:
    keys = sorted({_key(rel, stripped) for rel, _, stripped in hits})
    header = (
        "# Line-number-citation baseline -- see tools/check_line_citations.py\n"
        "# One key per known citation: <relpath-under-native/angr/src>\\t<stripped-comment>.\n"
        "# Regenerate with: tools/check_line_citations.py --update-baseline\n"
        "# Shrinking this file (by citing a symbol name instead of a line number) is the goal.\n"
    )
    BASELINE_PATH.write_text(header + "\n".join(keys) + "\n", encoding="utf-8")
    return len(keys)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="list ALL detected citations")
    ap.add_argument("--update-baseline", action="store_true", help="rewrite the baseline from current citations")
    args = ap.parse_args()

    if not SRC_ROOT.is_dir():
        print(f"error: source root not found: {SRC_ROOT}", file=sys.stderr)
        return 2

    hits = scan()

    if args.list:
        for rel, lineno, stripped in hits:
            print(f"{rel}:{lineno}\t{stripped}")
        print(f"\n{len(hits)} citation(s) found.")
        return 0

    if args.update_baseline:
        n = write_baseline(hits)
        print(f"wrote {n} baseline keys to {BASELINE_PATH.relative_to(REPO_ROOT)}")
        return 0

    baseline = load_baseline()
    new_hits = [h for h in hits if _key(h[0], h[2]) not in baseline]

    if not new_hits:
        print(f"OK: {len(hits)} line-number citations, {len(baseline)} baselined, 0 new.")
        return 0

    print(f"FOUND {len(new_hits)} new comment(s) citing a raw line number:\n")
    for rel, lineno, stripped in new_hits:
        print(f"  {rel}:{lineno}\t{stripped}")
    print(
        "\nCite the function/struct/const name instead of a line number -- line "
        "citations drift silently as code moves (same symbol-anchor convention "
        "CLAUDE.md documents for bd memories, applied to in-source comments). "
        "If this is intentional pre-existing debt, run --update-baseline."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
