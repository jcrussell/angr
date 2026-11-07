#!/usr/bin/env python3
"""Grep gate: ban raw ``z3::SatResult::{Sat,Unsat}`` matches and raw solver checks
in Rust core code (bd angr-qwyti.3, angr-vmi2l.307).

Interpreting a Z3 ``check()`` outcome must go through the single
forcing-function ``SatOutcome::decided`` (in
``native/angr-symex-core/src/symbolic/solver_build.rs``), which turns ``Unknown``
(timeout / incompleteness) into ``None`` so every call site is forced to decide
what a timeout means. Conflating ``Unknown`` with ``Unsat`` is the
angr-ph300.43 / angr-n0irt.1 bug class (see the ``invariant-z3-unknown-not-unsat``
memory): it fabricates extrema, permanently pins ``sat_cache=false``, and prunes
feasible branches.

This gate scans production Rust under the ``SCAN`` crates' code (comments and
string literals blanked) and fails when it finds any of the ``BANNED`` shapes:

  * a ``SatResult::Sat`` or ``SatResult::Unsat`` token;
  * a ``SatResult::*`` / ``SatResult::{..}`` import or a ``SatResult as``
    alias, which would bring the variants in under a name the first rule
    cannot see (angr-vmi2l.307);
  * ``.decided().unwrap_or(..)`` / ``.unwrap_or_default()``, which folds
    ``None`` (Unknown) back into a boolean: the collapse ``decided`` exists to
    prevent;
  * a raw ``.check(`` / ``.check_assumptions(`` outside ``solver_build.rs``.
    ``timed_check`` / ``timed_check_assumptions`` there count every check in
    ``Z3_CHECK_COUNT`` / ``Z3_TIMEOUT_COUNT`` and the per-site stats, and a raw
    check skips them (and can be matched against ``SatResult::Unknown``
    without naming the banned variants).

Two contexts are allowed:

  * a line (or the two lines above it) tagged ``// satresult-exempt`` — reserved
    for the ``SatOutcome`` impl and the per-variant stats counter in
    ``timed_check``, which legitimately distinguish all three outcomes;
  * test code (files named ``*_tests.rs`` or living under a ``*_tests/`` dir),
    where asserting a raw solver verdict is the point of the test.

``SatResult::Unknown`` on its own is always allowed — naming the timeout variant
is never the bug; collapsing it into a sat/unsat boolean is. A pattern is
matched over the whole blanked file, so a ``.decided()`` split from its
``.unwrap_or(`` across lines still fires; the hit reports its first line.

Usage::

    tools/audit_satresult.py            # gate: exit 1 on any violation
    tools/audit_satresult.py --list     # list every banned site, exempt and test ones too
    tools/audit_satresult.py --self-test  # prove it can fail

Pure stdlib; runs no symbolic execution.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# The scanner below is this gate's own; only root resolution is shared.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rust_source_utils import (
    ScanRootError,
    blank_noise,
    require_floor,
    rust_files,
    scan_roots,
    self_test_scan_roots,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN = {
    "native/angr": ("",),
    "native/angr-bv": ("",),
    "native/angr-vex": ("",),
    "native/angr-symex-core": ("",),
    "native/angr-symex-native": ("",),
}
EXCLUDE = {"native/angr-symex-macros": "proc-macro crate: compile-time code generation, it never checks a solver"}
# Fewer files than this means the code moved out of the scan roots.
MIN_FILES = 400

# Only the boolean-collapsing variants are banned; Unknown is fine to name.
BANNED = re.compile(
    r"SatResult::(?:Sat|Unsat)\b"
    r"|SatResult::\s*[*{]"
    r"|\bSatResult\s+as\s+\w"
    r"|\.decided\(\)\s*\.unwrap_or(?:_default)?\("
)
# The only file allowed to call the solver directly: its timed wrappers.
RAW_CHECK = re.compile(r"\.check(?:_assumptions)?\(")
RAW_CHECK_HOME = "solver_build.rs"
EXEMPT_TAG = "satresult-exempt"
# The rationale tag sits on the match arm's own line or just above it.
EXEMPT_LOOKBACK = 2


def _is_test_file(path: Path) -> bool:
    if path.name.endswith("_tests.rs"):
        return True
    return any(part.endswith("_tests") for part in path.parts)


def _tagged(lines: list[str], idx: int) -> bool:
    lo = max(0, idx - EXEMPT_LOOKBACK)
    return any(EXEMPT_TAG in lines[j] for j in range(lo, idx + 1))


def scan_text(path: Path, text: str, list_all: bool) -> list[tuple[Path, int, str]]:
    """Return (path, lineno, stripped) for every unexempt site in ``text``.

    ``path`` drives the test-file skip; :func:`scan` and :func:`self_test`
    share this matcher.
    """
    hits: list[tuple[Path, int, str]] = []
    if not list_all and _is_test_file(path):
        return hits
    lines = text.splitlines()
    code = blank_noise(text)
    starts = [m.start() for m in BANNED.finditer(code)]
    if path.name != RAW_CHECK_HOME:
        starts.extend(m.start() for m in RAW_CHECK.finditer(code))
    for idx in sorted({code.count("\n", 0, pos) for pos in starts}):
        if not list_all and _tagged(lines, idx):
            continue
        hits.append((path, idx + 1, lines[idx].strip()))
    return hits


def scan_files(repo_root: Path = REPO_ROOT) -> list[Path]:
    """Every file the real run reads; raises ScanRootError on a bad root or a low count."""
    files = rust_files(scan_roots(repo_root, SCAN, EXCLUDE))
    require_floor(files, MIN_FILES, "audit_satresult")
    return files


def scan(list_all: bool) -> list[tuple[Path, int, str]]:
    hits: list[tuple[Path, int, str]] = []
    for path in scan_files():
        rel = path.relative_to(REPO_ROOT)
        hits.extend(scan_text(rel, path.read_text(), list_all))
    return hits


# Synthetic source for --self-test: raw Sat/Unsat matches (must fire), the
# same line with a `satresult-exempt` tag on it or within EXEMPT_LOOKBACK
# above (the shape of the `SatOutcome::decided` impl in solver_build.rs), and
# a bare `SatResult::Unknown` (always allowed).
_SELF_TEST_SRC = """\
fn raw(solver: &z3::Solver) -> bool {
    match solver.check() {
        z3::SatResult::Sat => true,
        _ => false,
    }
}
fn raw_unsat(r: z3::SatResult) -> bool {
    r == z3::SatResult::Unsat
}
fn tagged_inline(r: z3::SatResult) -> bool {
    matches!(r, z3::SatResult::Sat) // satresult-exempt: why
}
impl SatOutcome for z3::SatResult {
    fn decided(self) -> Option<bool> {
        match self {
            // satresult-exempt: this impl is the sanctioned mapping site.
            z3::SatResult::Sat => Some(true),
            z3::SatResult::Unsat => Some(false),
            z3::SatResult::Unknown => None,
        }
    }
}
fn timeout_only(r: z3::SatResult) -> bool {
    r == z3::SatResult::Unknown
}
use z3::SatResult::*;
use z3::SatResult::{Sat, Unsat};
use z3::{SatResult as R, Solver};
fn collapse(s: &z3::Solver) -> bool {
    timed_check(s, CheckSite::Eval).decided().unwrap_or(false)
        || timed_check(s, CheckSite::Eval)
            .decided()
            .unwrap_or_default()
}
fn unknown_only(s: &z3::Solver, t: &[z3::ast::Bool]) -> bool {
    !matches!(s.check_assumptions(t), z3::SatResult::Unknown)
}
fn decided_ok(s: &z3::Solver) -> Option<bool> {
    // a comment naming z3::SatResult::Sat and solver.check() is prose
    let _ = "z3::SatResult::Unsat";
    timed_check(s, CheckSite::Eval).decided()
}
"""
_SELF_TEST_EXPECTED = [
    (2, "match solver.check() {"),
    (3, "z3::SatResult::Sat => true,"),
    (8, "r == z3::SatResult::Unsat"),
    (26, "use z3::SatResult::*;"),
    (27, "use z3::SatResult::{Sat, Unsat};"),
    (28, "use z3::{SatResult as R, Solver};"),
    (30, "timed_check(s, CheckSite::Eval).decided().unwrap_or(false)"),
    (32, ".decided()"),
    (36, "!matches!(s.check_assumptions(t), z3::SatResult::Unknown)"),
]
# The timed wrappers' home may call the solver directly: the same fixture
# there loses only its raw-check hits.
_SELF_TEST_RAW_CHECK_HOME = Path("native/angr-symex-core/src/symbolic/solver_build.rs")
_SELF_TEST_HOME_EXPECTED = [hit for hit in _SELF_TEST_EXPECTED if hit[0] not in (2, 36)]
_SELF_TEST_PROD = Path("native/angr-symex-core/src/symbolic/fixture.rs")
_SELF_TEST_TEST_PATHS = (
    Path("native/angr-symex-core/src/symbolic/fixture_tests.rs"),
    Path("native/angr-symex-core/src/symbolic/context_tests/fixture.rs"),
)


def self_test() -> int:
    """Prove the detector can fail, and that the tag and test-file skip work.

    A gate never seen to fail is indistinguishable from one that cannot. The
    fixture runs through :func:`scan_text`, the real per-file matcher with its
    test-file skip and exempt-tag lookback.
    """
    failures: list[str] = []
    got = [(ln, code) for _, ln, code in scan_text(_SELF_TEST_PROD, _SELF_TEST_SRC, list_all=False)]
    if got != _SELF_TEST_EXPECTED:
        failures.append(f"production path: expected {_SELF_TEST_EXPECTED}\n  got      {got}")
    got_home = [(ln, code) for _, ln, code in scan_text(_SELF_TEST_RAW_CHECK_HOME, _SELF_TEST_SRC, list_all=False)]
    if got_home != _SELF_TEST_HOME_EXPECTED:
        failures.append(f"{RAW_CHECK_HOME}: expected {_SELF_TEST_HOME_EXPECTED}\n  got      {got_home}")
    for test_path in _SELF_TEST_TEST_PATHS:
        got_test = scan_text(test_path, _SELF_TEST_SRC, list_all=False)
        if got_test:
            failures.append(f"test path {test_path}: expected no hits, got {got_test}")
    failures.extend(self_test_scan_roots(scan_files, SCAN))

    if failures:
        print("SELF-TEST FAILED")
        for f in failures:
            print(f"  {f}")
        return 1
    print(
        "self-test OK: raw SatResult::Sat / ::Unsat, glob/group/alias imports, "
        "decided().unwrap_or(..) collapses (also split across lines) and raw "
        "checks outside solver_build.rs flagged; comments and strings, satresult-exempt "
        "(inline and lookback), SatResult::Unknown and *_tests paths pass; "
        "an unclassified member, a missing root and a low file count fail"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--list",
        action="store_true",
        help="list every banned site (including exempt/test)",
    )
    ap.add_argument("--self-test", action="store_true", help="prove the detector fires on a synthetic raw match")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    try:
        hits = scan(list_all=args.list)
    except ScanRootError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.list:
        for rel, ln, code in hits:
            print(f"{rel}:{ln}: {code}")
        print(f"\n{len(hits)} banned site(s) total.")
        return 0

    if hits:
        print(
            "ERROR: raw SatResult::{Sat,Unsat} match, SatResult glob/alias import, "
            "decided() collapse or raw solver check (angr-qwyti.3, angr-vmi2l.307).\n"
            "Check through `timed_check` / `timed_check_assumptions` and route the "
            "result through `.decided()` (Some(true)=Sat, Some(false)=Unsat, "
            "None=Unknown), deciding None explicitly,\n"
            "or, for a site that must distinguish all three, tag the line "
            "`// satresult-exempt: <reason>`.\n",
            file=sys.stderr,
        )
        for rel, ln, code in hits:
            print(f"  {rel}:{ln}: {code}", file=sys.stderr)
        return 1

    print("OK: no raw SatResult::{Sat,Unsat} match, decided() collapse or raw check outside the helpers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
