#!/usr/bin/env python3
"""Steady-guard coverage audit for ``RustExplorationManager`` (bd angr-91vj9.2).

``#[angr_macros::steady_guarded]`` exists so a config mutator on
``RustExplorationManager`` cannot forget to ``steady_config_guard()`` — i.e.
finalize a live steady parallel session — before touching state that the
worker threads snapshot. It is applied dozens of times across
``native/angr/src/exploration/manager_methods*.rs``, but it is *opt-in per
function*, so a newly added mutator silently inherits the bug the macro was
built to prevent (round-3 audit found exactly that in ``set_max_history``,
filed as angr-c7xno.21).

This script closes the loop the same way ``tools/audit_silent_fallback.py`` and
``tools/check_line_citations.py`` do: a heuristic detector plus a checked-in
baseline of the already-known gaps. Only *new* unguarded mutators fail.

Detection heuristic — a function is "steady-guard eligible" when all hold:

  * it lives in ``manager_methods*.rs`` (excluding ``*_tests.rs``),
  * it is declared ``pub fn`` (the ``#[pymethods]`` surface; private helpers
    are reached through a guarded public entry point),
  * its receiver is ``&mut self``, and
  * its name starts with one of :data:`MUTATOR_PREFIXES`.

A hit is exempt when either:

  * the function carries ``#[angr_macros::steady_guarded]``, or
  * its doc block documents the exception, i.e. contains a line matching
    ``NOT ... steady_guarded`` (the convention already used by
    ``load_snapshot_bytes`` and ``set_parallel_frontier_residency``: the guard
    must run at a point the macro's unconditional prologue cannot express).

Everything else must either be fixed or appear in the baseline.

Usage::

    tools/audit_steady_guard_coverage.py                 # report new gaps
    tools/audit_steady_guard_coverage.py --list          # list ALL eligible fns
    tools/audit_steady_guard_coverage.py --update-baseline

Exit 0 when there are no new unguarded mutators (or on ``--list`` /
``--update-baseline``), 1 otherwise. Pure stdlib.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Repo layout: this file lives at <repo>/tools/audit_steady_guard_coverage.py
REPO_ROOT = Path(__file__).resolve().parent.parent
METHODS_DIR = REPO_ROOT / "native" / "angr" / "src" / "exploration"
BASELINE_PATH = REPO_ROOT / "tools" / "steady_guard_baseline.txt"

# Name prefixes that mark a function as a config *mutator* rather than an
# accessor or a run-loop entry point. Kept deliberately narrow: every prefix
# here already has at least one guarded member, so widening the list is a
# decision to be made with the baseline in hand, not a silent default.
MUTATOR_PREFIXES = (
    "set_",
    "clear_",
    "register_",
    "unregister_",
    "enable_",
    "disable_",
    "add_",
    "remove_",
    "load_",
    "configure_",
)

GUARD_ATTR = "angr_macros::steady_guarded"
# Documented-exception marker in the doc block, e.g.
#   /// Deliberately NOT `#[angr_macros::steady_guarded]`: the guard must ...
EXCEPTION_RE = re.compile(r"NOT\b[^\n]*steady_guarded")

FN_RE = re.compile(r"^pub fn (?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^>]*>)?\s*\(")


def _iter_method_files() -> list[Path]:
    files = []
    for p in sorted(METHODS_DIR.glob("manager_methods*.rs")):
        if p.name.endswith("_tests.rs"):
            continue
        files.append(p)
    return files


def _consume_attribute(lines: list[str], i: int) -> tuple[str, int]:
    """Join a (possibly multi-line) attribute starting at ``i``.

    Returns the joined text and the index of the line *after* the attribute.
    """
    text = lines[i].strip()
    j = i
    while text.count("[") > text.count("]") and j + 1 < len(lines):
        j += 1
        text += " " + lines[j].strip()
    return text, j + 1


def _consume_signature(lines: list[str], i: int) -> tuple[str, int]:
    """Join a (possibly multi-line) fn signature starting at ``i``.

    Returns the text up to and including the opening brace (or ``;`` for a
    trait-style declaration) and the index of the line after it.
    """
    text = lines[i].strip()
    j = i
    while "{" not in text and not text.endswith(";") and j + 1 < len(lines):
        j += 1
        text += " " + lines[j].strip()
    return text, j + 1


def scan() -> list[tuple[str, int, str, bool, bool]]:
    """Return (rel, lineno, fn_name, guarded, documented_exception) per hit."""
    hits: list[tuple[str, int, str, bool, bool]] = []
    for path in _iter_method_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

        guarded = False
        exception = False
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if stripped.startswith("#["):
                text, i = _consume_attribute(lines, i)
                if GUARD_ATTR in text:
                    guarded = True
                continue
            if stripped.startswith("//"):
                if EXCEPTION_RE.search(stripped):
                    exception = True
                i += 1
                continue
            if not stripped:
                i += 1
                continue
            m = FN_RE.match(stripped)
            if m:
                sig, nxt = _consume_signature(lines, i)
                name = m.group("name")
                if name.startswith(MUTATOR_PREFIXES) and "&mut self" in sig:
                    hits.append((rel, i + 1, name, guarded, exception))
                guarded = exception = False
                i = nxt
                continue
            # Any other code line ends the current doc/attribute block.
            guarded = exception = False
            i += 1
    return hits


def _key(rel: str, name: str) -> str:
    """Line-number-independent baseline key: <relpath>\\t<fn name>."""
    return f"{rel}\t{name}"


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


def write_baseline(gaps: list[tuple[str, int, str, bool, bool]]) -> int:
    keys = sorted({_key(rel, name) for rel, _, name, _, _ in gaps})
    header = (
        "# Steady-guard coverage baseline -- see tools/audit_steady_guard_coverage.py\n"
        "# One key per known-unguarded manager mutator: <relpath>\\t<fn name>.\n"
        "# Regenerate with: tools/audit_steady_guard_coverage.py --update-baseline\n"
        "# Shrinking this file (by adding #[angr_macros::steady_guarded], or by\n"
        "# documenting the exception with a `NOT ... steady_guarded` doc line) is\n"
        "# the goal. Growing it needs a reason in the commit message.\n"
    )
    BASELINE_PATH.write_text(header + "\n".join(keys) + "\n", encoding="utf-8")
    return len(keys)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="list ALL eligible mutators (guarded + not)")
    ap.add_argument("--update-baseline", action="store_true", help="rewrite the baseline from current gaps")
    args = ap.parse_args()

    if not METHODS_DIR.is_dir():
        print(f"error: exploration source dir not found: {METHODS_DIR}", file=sys.stderr)
        return 2

    hits = scan()
    gaps = [h for h in hits if not h[3] and not h[4]]

    if args.list:
        for rel, lineno, name, guarded, exception in hits:
            mark = "GUARDED" if guarded else ("EXEMPT " if exception else "GAP    ")
            print(f"{mark} {rel}:{lineno}\t{name}")
        n_guarded = sum(1 for h in hits if h[3])
        n_exempt = sum(1 for h in hits if not h[3] and h[4])
        print(f"\n{len(hits)} eligible mutators ({n_guarded} guarded, {n_exempt} documented-exempt, {len(gaps)} gaps)")
        return 0

    if args.update_baseline:
        n = write_baseline(gaps)
        print(f"wrote {n} baseline keys to {BASELINE_PATH.relative_to(REPO_ROOT)}")
        return 0

    baseline = load_baseline()
    new_gaps = [h for h in gaps if _key(h[0], h[2]) not in baseline]

    if not new_gaps:
        n_guarded = sum(1 for h in hits if h[3])
        print(f"OK: {len(hits)} eligible mutators, {n_guarded} guarded, {len(gaps)} baselined gaps, 0 new unguarded.")
        return 0

    print(f"FOUND {len(new_gaps)} new unguarded manager mutator(s):\n")
    for rel, lineno, name, _, _ in new_gaps:
        print(f"  {rel}:{lineno}\t{name}")
    print(
        "\nAdd `#[angr_macros::steady_guarded]` above each function, or -- if the\n"
        "guard genuinely cannot be the unconditional first statement -- document\n"
        "why with a doc line matching `NOT ... steady_guarded` (see\n"
        "`load_snapshot_bytes`). Use --update-baseline only to defer a gap you\n"
        "have deliberately decided not to close."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
