#!/usr/bin/env python3
"""Baseline-audited guard against raw line-number citations in comments and docs.

angr-9ke6b (the pre-PR audit of native/angr/src/) turned up a recurring bug
class: doc comments and inline comments that cite an EXACT line number in
another file (``arch/mod.rs:638``) or in the same file (``line ~478``) as
part of an invariant, cross-reference, or "see also" note -- including a
widespread and otherwise-valuable convention of citing the upstream Python
implementation's line numbers (``engines/successors.py:203``) to document
behavior parity. Those citations drift silently as the cited code moves --
AGENTS.md's symbol-anchor rule exists for exactly this
failure mode; it covers bd memories, in-source comments and ``docs/**/*.rst``.

Mirrors tools/audit_silent_fallback.py's shape rather than diffing against a
base ref: a diff-scoped version would need to pick a base branch, and this
repo's PRs land against `rust-symex` today but may eventually target `master`
directly for the final upstream merge, at which point a base-ref diff would
dump the branch's entire multi-year backlog of (mostly legitimate) citations
onto one unlucky PR with no escape hatch. Scanning the whole tree against a
checked-in baseline is base-ref-independent and self-documenting: existing
citations are grandfathered once, and the baseline should only shrink as they
get converted to symbol references.

Scanned surfaces: ``//``-comments under the ``src/`` of every fork-owned
workspace member (``rust_source_utils.scan_roots``) and *every* line of
``docs/**/*.rst``. The docs half was added by angr-kw5f6 after angr-sqfj8.126
found ``docs/advanced-topics/rust_engine.rst``'s typed-exception table citing
``engine.rs``/``errors.rs`` line numbers that had drifted 60-95 lines with zero
gate signal -- prose rots exactly the way comments do, and the ``.rst`` files
here are dense with ``file.py:NNN`` cross-references.

Usage::

    tools/check_line_citations.py                 # report new un-baselined citations
    tools/check_line_citations.py --list           # list ALL detected citations
    tools/check_line_citations.py --update-baseline
    tools/check_line_citations.py --self-test      # prove it can fail

Exit 0 when there are no new citations (or on --list / --update-baseline),
1 when new ones are found. Pure stdlib.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# The scanner below is this gate's own; only root resolution is shared.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rust_source_utils import ScanRootError, require_floor, rust_files, scan_roots, self_test_scan_roots

REPO_ROOT = Path(__file__).resolve().parent.parent
# Comment hygiene holds in every fork-owned crate.
SCAN = {"native/angr": ("",), "native/angr-macros": ("",)}
EXCLUDE: dict[str, str] = {}
# Fewer Rust files than this means the code moved out of the scan roots.
MIN_FILES = 400
BASELINE_PATH = REPO_ROOT / "tools" / "line_citations_baseline.txt"

# file.rs:123 / some/path/foo.py:45 -- cite-another-location-by-line-number.
FILE_LINE_RE = re.compile(r"[\w./-]+\.(?:rs|py):\d+")
# "line 638" / "lines ~478-490" -- same shape without a filename.
LINE_WORD_RE = re.compile(r"\blines?\s*~?\d+\b", re.IGNORECASE)


def _iter_sources(repo_root: Path = REPO_ROOT) -> list[Path]:
    """Every file whose citations are gated, in a stable order.

    ``.rs`` sources are filtered down to comment lines by :func:`_is_scannable`;
    ``.rst`` docs are scanned whole, since the entire file is prose. Raises
    ScanRootError on a bad root or a Rust scan below ``MIN_FILES``.
    """
    rs = rust_files(scan_roots(repo_root, SCAN, EXCLUDE))
    require_floor(rs, MIN_FILES, "check_line_citations")
    docs = repo_root / "docs"
    if not docs.is_dir():
        raise ScanRootError(f"scan root {docs} is missing")
    return rs + sorted(docs.rglob("*.rst"))


def _is_scannable(path: Path, stripped: str) -> bool:
    """Whether this line of ``path`` is eligible to be a citation."""
    if path.suffix == ".rs":
        return stripped.startswith("//")
    return bool(stripped)


def _key(rel: str, stripped: str) -> str:
    """Line-number-independent baseline key so unrelated edits don't churn it."""
    return f"{rel}\t{stripped}"


def scan_text(rel: str, text: str) -> list[tuple[str, int, str]]:
    """Return (rel, lineno, stripped) for every citation-shaped line of ``text``.

    ``rel``'s suffix picks the surface (``.rs`` comment lines vs whole
    ``.rst``), so :func:`scan` and :func:`self_test` share one matcher.
    """
    path = Path(rel)
    hits: list[tuple[str, int, str]] = []
    for idx, raw in enumerate(text.splitlines()):
        stripped = raw.strip()
        if not _is_scannable(path, stripped):
            continue
        if FILE_LINE_RE.search(stripped) or LINE_WORD_RE.search(stripped):
            hits.append((rel, idx + 1, stripped))
    return hits


def scan() -> list[tuple[str, int, str]]:
    """Return (rel, lineno, stripped) for every citation-shaped line."""
    hits: list[tuple[str, int, str]] = []
    for path in _iter_sources():
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        hits.extend(scan_text(rel, text))
    return hits


def new_citations(hits: list[tuple[str, int, str]], baseline: set[str]) -> list[tuple[str, int, str]]:
    """The hits whose line-number-independent key is not in ``baseline``."""
    return [h for h in hits if _key(h[0], h[2]) not in baseline]


# Synthetic sources for --self-test: a file:NNN and a "line NNN" citation on
# each surface, plus must-NOT-fire controls (a plain comment, a non-comment
# .rs line, plain prose).
_SELF_TEST_RS = """\
// see foo.rs:123 for the invariant
/// Mirrors the guard at line 42 of the caller.
// A plain comment naming foo.rs and a line of code, no number.
let line_42 = bar.rs; // not a comment line, so not scanned
"""
_SELF_TEST_RST = """\
The table lives in ``engine.rs:88``.
See lines ~478-490 of the upstream file.
A plain prose line with no citation.
"""
_SELF_TEST_EXPECTED = [
    ("native/angr/src/fixture.rs", 1, "// see foo.rs:123 for the invariant"),
    ("native/angr/src/fixture.rs", 2, "/// Mirrors the guard at line 42 of the caller."),
    ("docs/fixture.rst", 1, "The table lives in ``engine.rs:88``."),
    ("docs/fixture.rst", 2, "See lines ~478-490 of the upstream file."),
]


def self_test() -> int:
    """Prove the detector and the baseline filter can both fail.

    A gate never seen to fail is indistinguishable from one that cannot. This
    checks the matcher (positive and negative fixtures through
    :func:`scan_text`) and the baseline semantics (a baselined key is
    suppressed even after its line moves; an un-baselined one is not).
    """
    failures: list[str] = []
    got = scan_text("native/angr/src/fixture.rs", _SELF_TEST_RS) + scan_text("docs/fixture.rst", _SELF_TEST_RST)
    if got != _SELF_TEST_EXPECTED:
        failures.append(f"matcher: expected {_SELF_TEST_EXPECTED}\n  got      {got}")

    # Baseline: grandfather the first .rs citation, then shift every line down.
    baseline = {_key("native/angr/src/fixture.rs", "// see foo.rs:123 for the invariant")}
    shifted = scan_text("native/angr/src/fixture.rs", "// padding\n" + _SELF_TEST_RS)
    fresh = new_citations(shifted, baseline)
    want = [("native/angr/src/fixture.rs", 3, "/// Mirrors the guard at line 42 of the caller.")]
    if fresh != want:
        failures.append(f"baseline: expected {want}\n  got      {fresh}")
    failures.extend(self_test_scan_roots(_iter_sources, SCAN))

    if failures:
        print("SELF-TEST FAILED")
        for f in failures:
            print(f"  {f}")
        return 1
    print(
        f"self-test OK: {len(got)} citations flagged as expected "
        "(.rs file:NNN, .rs line NNN, .rst file:NNN, .rst lines NNN; clean controls silent), "
        "baseline suppresses a moved grandfathered key; "
        "an unclassified member, a missing root and a low file count fail"
    )
    return 0


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
        "# One key per known citation: <repo-relative-path>\\t<stripped-line>.\n"
        "# Regenerate with: tools/check_line_citations.py --update-baseline\n"
        "# Shrinking this file (by citing a symbol name instead of a line number) is the goal.\n"
    )
    BASELINE_PATH.write_text(header + "\n".join(keys) + "\n", encoding="utf-8")
    return len(keys)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="list ALL detected citations")
    ap.add_argument("--update-baseline", action="store_true", help="rewrite the baseline from current citations")
    ap.add_argument("--self-test", action="store_true", help="prove the detector fires on synthetic citations")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    try:
        hits = scan()
    except ScanRootError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

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
    new_hits = new_citations(hits, baseline)

    if not new_hits:
        print(f"OK: {len(hits)} line-number citations, {len(baseline)} baselined, 0 new.")
        return 0

    print(f"FOUND {len(new_hits)} new comment(s) citing a raw line number:\n")
    for rel, lineno, stripped in new_hits:
        print(f"  {rel}:{lineno}\t{stripped}")
    print(
        "\nCite the function/struct/const name instead of a line number -- line "
        "citations drift silently as code moves (same symbol-anchor convention "
        "AGENTS.md's hygiene rules require for bd memories, comments and docs). "
        "If this is intentional pre-existing debt, run --update-baseline."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
