#!/usr/bin/env python3
"""Rounding-mode threading audit for the VEX op family (bd angr-91vj9.7).

``VEXOps`` exposes a family of near-identical ``*_rm`` conversion/arithmetic
helpers. Each takes a VEX rounding-mode operand and hands a *concrete* closure
to a shared ``*_rm`` dispatcher (``float_to_int_rm``, ``float_to_float_rm``,
…). Those dispatchers exist for exactly one reason: to thread the rounding mode
down into the concrete path alongside the symbolic Z3 FP path. A closure that
drops the mode silently computes round-to-nearest-even for every mode, and the
symbolic and concrete paths then disagree.

That is not hypothetical: ``f64_to_f32_rm``'s closure named its mode parameter
``_rm`` (Rust's "intentionally unused" convention) and used a bare ``as f32``
cast, so RZ/RU/RD were silently wrong while every sibling correctly routed
through ``apply_rounding_f32``/``apply_rounding_f64`` (angr-c7xno.85). Nothing
flagged the divergence — the compiler is happy with a leading-underscore
parameter, which is precisely what makes this shape invisible.

This script closes the loop the same way ``tools/audit_silent_fallback.py``,
``tools/check_line_citations.py`` and ``tools/audit_steady_guard_coverage.py``
do: a heuristic detector plus a checked-in baseline of already-known gaps. Only
*new* mode-dropping closures fail.

Detection heuristic — inside any ``fn <name>_rm(...)`` in
``native/angr/src/vex/**/*.rs`` (test modules excluded), every closure literal
of arity >= 2 is checked. Its **last** parameter is the rounding mode by
convention (``|v, rm|``, ``|a, b, rm|``). A closure is a gap when:

  * that parameter is named ``_rm`` / ``_`` / any ``_``-prefixed name — an
    explicit "I am discarding the rounding mode", or
  * it is named ``rm`` but the closure body never mentions ``rm``.

A gap is exempt when the line above the closure (or the closure's own trailing
comment) documents it with a marker matching :data:`EXEMPT_RE`, i.e. contains
``rm-ignored:`` followed by a rationale — the escape hatch for the genuine
cases (e.g. a libm-backed transcendental where Rust only offers RNE).
Everything else must be fixed or appear in the baseline.

Usage::

    tools/audit_rounding_mode_threading.py                 # report new gaps
    tools/audit_rounding_mode_threading.py --list          # list ALL closures
    tools/audit_rounding_mode_threading.py --self-test     # prove it can fail
    tools/audit_rounding_mode_threading.py --update-baseline

Exit 0 when there are no new mode-dropping closures (or on ``--list`` /
``--update-baseline``), 1 otherwise. Pure stdlib.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Repo layout: this file lives at <repo>/tools/audit_rounding_mode_threading.py
REPO_ROOT = Path(__file__).resolve().parent.parent
VEX_DIR = REPO_ROOT / "native" / "angr" / "src" / "vex"
BASELINE_PATH = REPO_ROOT / "tools" / "rounding_mode_baseline.txt"

# Documented-exception marker, e.g.
#   // rm-ignored: Rust libm is round-to-nearest-even only.
EXEMPT_RE = re.compile(r"rm-ignored:")

FN_RM_RE = re.compile(r"\bfn\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*_rm)\s*[(<]")
ANY_FN_RE = re.compile(r"\bfn\s+[A-Za-z_][A-Za-z0-9_]*\s*[(<]")
# A closure header: `|a, b|` / `|v, _rm|`. Excludes `||` (arity 0) and the
# `a || b` boolean-or shape, since both alternations require an identifier.
CLOSURE_RE = re.compile(r"\|\s*(?P<params>[A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)+)\s*\|")


def _iter_source_files() -> list[Path]:
    """Non-test Rust sources under native/angr/src/vex/."""
    files = []
    for p in sorted(VEX_DIR.rglob("*.rs")):
        name = p.name
        if name.startswith("tests_") or name.endswith("_tests.rs") or name in ("test_helpers.rs", "property_tests.rs"):
            continue
        files.append(p)
    return files


def _blank_noise(src: str) -> str:
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


def _enclosing_fn(src: str, pos: int) -> str | None:
    """Name of the nearest preceding ``fn <...>_rm`` declaration, if any.

    Good enough for this file set: the ``*_rm`` helpers are short, flat
    functions in an ``impl`` block, so "nearest preceding" and "enclosing"
    coincide. A closure sitting after the last ``_rm`` fn in the file but
    outside it would be a false positive; none exist today, and the exemption
    marker covers one if it ever appears.
    """
    last = None
    for m in FN_RM_RE.finditer(src, 0, pos):
        last = m
    if last is None:
        return None
    # Reject when any other fn was declared between that `_rm` fn and the
    # closure — then the closure belongs to the later fn, not to this one.
    if ANY_FN_RE.search(src, last.end(), pos) is not None:
        return None
    return last.group("name")


def _closure_body(src: str, start: int) -> str:
    """Text of the closure body beginning at ``start`` (just past the params).

    Scans forward tracking bracket depth; stops at the ``,`` or closing bracket
    that ends the closure expression in its enclosing call.
    """
    depth = 0
    i, n = start, len(src)
    while i < n:
        c = src[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0:
                break
            depth -= 1
        elif (c == "," and depth == 0) or (c == ";" and depth == 0):
            break
        i += 1
    return src[start:i]


def scan_text(rel: str, raw: str) -> list[tuple[str, int, str, str, bool, bool]]:
    """Return (rel, lineno, fn_name, params, drops_rm, exempt) per closure."""
    hits: list[tuple[str, int, str, str, bool, bool]] = []
    if "_rm" not in raw:
        return hits
    src = _blank_noise(raw)
    raw_lines = raw.splitlines()

    for m in CLOSURE_RE.finditer(src):
        fn_name = _enclosing_fn(src, m.start())
        if fn_name is None:
            continue
        params = [p.strip() for p in m.group("params").split(",")]
        rm_param = params[-1]
        body = _closure_body(src, m.end())
        if rm_param.startswith("_"):
            drops = True
        elif rm_param == "rm":
            drops = not re.search(r"\brm\b", body)
        else:
            # Last param is not a rounding mode (e.g. `|lhs, rhs|` in a lane
            # helper that happens to sit inside a `_rm` fn).
            continue
        lineno = src.count("\n", 0, m.start()) + 1
        window = raw_lines[max(0, lineno - 3) : lineno]
        exempt = any(EXEMPT_RE.search(line) for line in window)
        hits.append((rel, lineno, fn_name, m.group("params"), drops, exempt))
    return hits


def scan() -> list[tuple[str, int, str, str, bool, bool]]:
    hits: list[tuple[str, int, str, str, bool, bool]] = []
    for path in _iter_source_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        hits.extend(scan_text(rel, path.read_text(encoding="utf-8", errors="replace")))
    return hits


# Synthetic source for --self-test: three closures the detector must classify
# as threaded / dropped-via-underscore / dropped-via-unused, plus one exempt.
_SELF_TEST_SRC = """
impl VEXOps {
    fn good_rm(rm: RustBV, arg: RustBV) -> R {
        Self::float_to_int_rm(rm, arg, |v, rm| Self::apply_rounding_f32(v, rm) as u128)
    }
    fn underscore_rm(rm: RustBV, arg: RustBV) -> R {
        Self::float_to_float_rm(rm, arg, |v, _rm| (v as f32).to_bits() as u128)
    }
    fn unused_rm(rm: RustBV, arg: RustBV) -> R {
        // A comment mentioning rm must not count as a use.
        Self::float_to_int_rm(rm, arg, |v, rm| (v as i32) as u128)
    }
    fn exempt_rm(rm: RustBV, arg: RustBV) -> R {
        // rm-ignored: libm is round-to-nearest-even only.
        Self::float_to_int_rm(rm, arg, |v, _rm| v.sin() as u128)
    }
    fn plain(a: u32) -> u32 {
        (0..a).map(|x, y| x + y).sum()
    }
}
"""

_SELF_TEST_EXPECTED = [
    ("good_rm", "v, rm", False, False),
    ("underscore_rm", "v, _rm", True, False),
    ("unused_rm", "v, rm", True, False),
    ("exempt_rm", "v, _rm", True, True),
]


def self_test() -> int:
    """Prove the detector can actually fail (and can be exempted).

    A lint that has never been seen to fire is indistinguishable from one that
    cannot — this repo has been bitten by that before (see the valgrind gate's
    ``--self-test`` note in CLAUDE.md).
    """
    got = [(fn, params, drops, exempt) for _, _, fn, params, drops, exempt in scan_text("<self-test>", _SELF_TEST_SRC)]
    if got != _SELF_TEST_EXPECTED:
        print("SELF-TEST FAILED\n  expected:")
        for row in _SELF_TEST_EXPECTED:
            print(f"    {row}")
        print("  got:")
        for row in got:
            print(f"    {row}")
        return 1
    print(f"self-test OK: {len(got)} closures classified as expected (1 threaded, 2 gaps, 1 exempt)")
    return 0


def _key(rel: str, fn_name: str, params: str) -> str:
    """Line-number-independent baseline key: <relpath>\\t<fn>\\t<params>."""
    return f"{rel}\t{fn_name}\t{params}"


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


def write_baseline(gaps: list[tuple[str, int, str, str, bool, bool]]) -> int:
    keys = sorted({_key(rel, fn, params) for rel, _, fn, params, _, _ in gaps})
    header = (
        "# Rounding-mode threading baseline -- see tools/audit_rounding_mode_threading.py\n"
        "# One key per known mode-dropping closure: <relpath>\\t<fn>\\t<closure params>.\n"
        "# Regenerate with: tools/audit_rounding_mode_threading.py --update-baseline\n"
        "# This file should stay EMPTY. Threading the mode through (see\n"
        "# VEXOps::apply_rounding_f32 / apply_rounding_f64 / narrow_f64_to_f32_rm)\n"
        "# or documenting the exception with an `rm-ignored:` comment are both\n"
        "# preferable; growing it needs a reason in the commit message.\n"
    )
    BASELINE_PATH.write_text(header + ("\n".join(keys) + "\n" if keys else ""), encoding="utf-8")
    return len(keys)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="list ALL rm closures (threaded + dropped)")
    ap.add_argument("--update-baseline", action="store_true", help="rewrite the baseline from current gaps")
    ap.add_argument("--self-test", action="store_true", help="prove the detector fires on a synthetic gap")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if not VEX_DIR.is_dir():
        print(f"error: vex source dir not found: {VEX_DIR}", file=sys.stderr)
        return 2

    hits = scan()
    gaps = [h for h in hits if h[4] and not h[5]]

    if args.list:
        for rel, lineno, fn, params, drops, exempt in hits:
            mark = "THREADED" if not drops else ("EXEMPT  " if exempt else "GAP     ")
            print(f"{mark} {rel}:{lineno}\t{fn}\t|{params}|")
        n_threaded = sum(1 for h in hits if not h[4])
        n_exempt = sum(1 for h in hits if h[4] and h[5])
        print(f"\n{len(hits)} rm closures ({n_threaded} threaded, {n_exempt} documented-exempt, {len(gaps)} gaps)")
        return 0

    if args.update_baseline:
        n = write_baseline(gaps)
        print(f"wrote {n} baseline keys to {BASELINE_PATH.relative_to(REPO_ROOT)}")
        return 0

    baseline = load_baseline()
    new_gaps = [h for h in gaps if _key(h[0], h[2], h[3]) not in baseline]

    if not new_gaps:
        n_threaded = sum(1 for h in hits if not h[4])
        print(f"OK: {len(hits)} rm closures, {n_threaded} thread the mode, {len(gaps)} baselined gaps, 0 new.")
        return 0

    print(f"FOUND {len(new_gaps)} closure(s) dropping the VEX rounding mode:\n")
    for rel, lineno, fn, params, _, _ in new_gaps:
        print(f"  {rel}:{lineno}\t{fn}\t|{params}|")
    print(
        "\nThread the mode through the concrete path (see `VEXOps::apply_rounding_f32`,\n"
        "`apply_rounding_f64`, `narrow_f64_to_f32_rm`) so it matches the symbolic Z3 FP\n"
        "path. If the mode genuinely cannot be honoured (e.g. Rust libm is\n"
        "round-to-nearest-even only), document it with an `rm-ignored: <why>` comment\n"
        "on the line above. Use --update-baseline only to defer a gap you have\n"
        "deliberately decided not to close."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
