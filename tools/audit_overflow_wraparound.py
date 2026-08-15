#!/usr/bin/env python3
"""Integer overflow / wraparound audit (bd angr-03vl4, round-4 retrospective).

``[profile.release]`` (repo-root ``Cargo.toml``) sets no ``overflow-checks``,
so a bare ``+``/``-`` on a guest-controlled or Z3-derived address/size value
does not panic in the shipped ``.so`` -- it silently wraps, turning a range
check or an allocation size into a wrong answer rather than a crash.
``cargo test --profile release-checked`` (the CI test job) *does* enable
overflow-checks, but that only catches the bug if some test happens to drive
the value near a 64-bit boundary; it is a runtime safety net, not a static
one.

This was round 4's dominant bug class (``invariant-proc-address-arith-wrapping``,
``invariant-rust-concrete-arith-must-wrap``,
``invariant-overflow-fix-refuse-not-saturate-identities``): 3 P1s and a
double-digit count of P2/P3s across memory/, interpreter/, symbolic/,
state/, syscalls/ and procedures/, all now fixed by hand but with nothing
stopping the same shape from being reintroduced. Every fixed site uses a
``.wrapping_*()``/``.saturating_*()``/``.checked_*()`` *method call* rather
than a bare operator, so this script's heuristic is simple: flag a bare
``+``/``-`` binary operator where at least one operand's name looks like an
address/size/offset/count. Because the safe pattern is always a method call,
not an operator, no explicit exclusion for already-fixed sites is needed --
the operator scan just never matches them.

This closes the loop the same way ``tools/audit_rounding_mode_threading.py``
and ``tools/audit_sp_default_zero.py`` do: a heuristic detector plus a
checked-in baseline of already-triaged sites. Only *new* sites fail.

Detection heuristic -- in any non-test ``.rs`` file under
``native/angr/src/{memory,interpreter,symbolic,state,syscalls,procedures}/``,
a bare ``+`` or ``-`` (not ``+=``/``-=``/``->``, not unary) whose left or
right operand is a simple identifier / dotted field access (or, on the
right, a numeric literal) whose last ``_``-separated component is one of
:data:`NAME_COMPONENTS`.

A site is exempt when the line above it (or its own trailing comment)
carries a marker matching :data:`EXEMPT_RE`, i.e. ``overflow-ok:`` followed
by a rationale -- the escape hatch for arithmetic that only *looks*
address-shaped (e.g. a bounds-checked ``usize`` loop counter that can never
reach a guest-controlled value).

Usage::

    tools/audit_overflow_wraparound.py                 # report new sites
    tools/audit_overflow_wraparound.py --list          # list ALL matched sites
    tools/audit_overflow_wraparound.py --self-test     # prove it can fail
    tools/audit_overflow_wraparound.py --update-baseline

Exit 0 when there are no new sites (or on ``--list`` / ``--update-baseline``),
1 otherwise. Pure stdlib.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rust_source_utils import blank_noise, enclosing_fn, is_test_file

# Repo layout: this file lives at <repo>/tools/audit_overflow_wraparound.py
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "native" / "angr" / "src"
# The 6 subsystems the round-4 retrospective named. vex/ has its own written
# convention (invariant-rust-concrete-arith-must-wrap) and can be folded in
# as a later extension.
SUBSYSTEMS = ("memory", "interpreter", "symbolic", "state", "syscalls", "procedures")
SCAN_DIRS = tuple(SRC_DIR / s for s in SUBSYSTEMS)
BASELINE_PATH = REPO_ROOT / "tools" / "overflow_baseline.txt"

# Documented-exception marker, e.g.
#   // overflow-ok: usize loop counter bounded by `size`, never guest-controlled.
EXEMPT_RE = re.compile(r"overflow-ok:")

# `_`-separated name components that mark an operand as address/size-shaped.
NAME_COMPONENTS = frozenset(
    {
        "addr",
        "base",
        "dst",
        "src",
        "ptr",
        "offset",
        "size",
        "len",
        "length",
        "count",
        "start",
        "end",
        "brk",
        "stride",
    }
)

_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
# Dotted field access (`self.field.subfield`) or a plain identifier.
_TOKEN = rf"(?:{_IDENT}\.)*{_IDENT}"
# A numeric literal, e.g. `8`, `1u64`, `0x1000`.
_NUM = r"\d[\w]*"
_OPERAND = rf"(?:{_TOKEN}|{_NUM})"

# A bare `+`/`-` between two operands. The lookahead after the operator
# excludes `+=`/`-=`/`->`; there is no lhs-adjacency requirement for unary
# minus since OPERAND must immediately precede the operator.
BINOP_RE = re.compile(rf"(?P<lhs>{_OPERAND})[ \t]*(?P<op>[+-])(?![=>])[ \t]*(?P<rhs>{_OPERAND})")


def _iter_source_files() -> list[Path]:
    """Non-test Rust sources under the 6 scanned subsystem dirs."""
    return [p for d in SCAN_DIRS if d.is_dir() for p in sorted(d.rglob("*.rs")) if not is_test_file(p)]


def _core_name(operand: str) -> str | None:
    """Last dotted segment of an operand, or None for a numeric literal."""
    if operand[0].isdigit():
        return None
    return operand.rsplit(".", 1)[-1]


def _is_addr_shaped(operand: str) -> bool:
    name = _core_name(operand)
    if name is None:
        return False
    return bool(NAME_COMPONENTS.intersection(name.split("_")))


def _normalize(text: str) -> str:
    """Collapse whitespace so a wrapped expression and a one-liner share a key."""
    return re.sub(r"\s+", "", text)


def scan_text(rel: str, raw: str) -> list[tuple[str, int, str, str, bool]]:
    """Return (rel, lineno, fn_name, shape, exempt) per address-shaped bare +/- site."""
    hits: list[tuple[str, int, str, str, bool]] = []
    src = blank_noise(raw)
    raw_lines = raw.splitlines()

    for m in BINOP_RE.finditer(src):
        if not (_is_addr_shaped(m.group("lhs")) or _is_addr_shaped(m.group("rhs"))):
            continue
        lineno = src.count("\n", 0, m.start()) + 1
        window = raw_lines[max(0, lineno - 3) : lineno]
        exempt = any(EXEMPT_RE.search(line) for line in window)
        shape = _normalize(m.group(0))
        hits.append((rel, lineno, enclosing_fn(src, m.start()) or "<top-level>", shape, exempt))
    return hits


def scan() -> list[tuple[str, int, str, str, bool]]:
    hits: list[tuple[str, int, str, str, bool]] = []
    for path in _iter_source_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        hits.extend(scan_text(rel, path.read_text(encoding="utf-8", errors="replace")))
    return hits


# Synthetic source for --self-test: a wrapping control, an unrelated-name
# control (bare + but neither operand is address-shaped), a real gap, a
# documented-exempt gap, and two must-NOT-fire compound-assignment/arrow
# controls.
_SELF_TEST_SRC = """
impl Thing {
    fn safe_wrapping(&self, addr: u64, offset: u64) -> u64 {
        addr.wrapping_add(offset)
    }
    fn unrelated_add(&self, total: u64, extra: u64) -> u64 {
        total + extra
    }
    fn raw_addr_add(&self, addr: u64, offset: u64) -> u64 {
        addr + offset
    }
    fn raw_size_sub(&self, size: u64) -> u64 {
        // overflow-ok: size is always >= 1 here, checked by caller.
        size - 1
    }
    fn compound_assign(&mut self, addr: u64) {
        addr += 1;
    }
    fn arrow_return(&self) -> u64 {
        0
    }
}
"""

_SELF_TEST_EXPECTED = [
    ("raw_addr_add", "addr+offset", False),
    ("raw_size_sub", "size-1", True),
]


def self_test() -> int:
    """Prove the detector can actually fail (and can be exempted).

    A lint that has never been seen to fire is indistinguishable from one
    that cannot -- this repo has been bitten by that before (see the
    valgrind gate's ``--self-test`` note in CLAUDE.md). This runs ahead of
    the real check in CI.
    """
    got = [(fn, shape, exempt) for _, _, fn, shape, exempt in scan_text("<self-test>", _SELF_TEST_SRC)]
    if got != _SELF_TEST_EXPECTED:
        print("SELF-TEST FAILED\n  expected:")
        for row in _SELF_TEST_EXPECTED:
            print(f"    {row}")
        print("  got:")
        for row in got:
            print(f"    {row}")
        return 1
    print(f"self-test OK: {len(got)} sites classified as expected (1 gap, 1 documented-exempt, 4 clean controls)")
    return 0


def _key(rel: str, fn_name: str, shape: str) -> str:
    """Line-number-independent baseline key: <relpath>\\t<fn>\\t<shape>."""
    return f"{rel}\t{fn_name}\t{shape}"


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


def write_baseline(gaps: list[tuple[str, int, str, str, bool]]) -> int:
    keys = sorted({_key(rel, fn, shape) for rel, _, fn, shape, _ in gaps})
    header = (
        "# Integer overflow / wraparound baseline -- see tools/audit_overflow_wraparound.py\n"
        "# One key per known address-shaped bare +/- site: <relpath>\\t<fn>\\t<normalized shape>.\n"
        "# Regenerate with: tools/audit_overflow_wraparound.py --update-baseline\n"
        "# This file should stay EMPTY. Fixing the site with wrapping_*/saturating_*/\n"
        "# checked_* (see invariant-overflow-fix-refuse-not-saturate-identities for\n"
        "# which one applies) or documenting a genuine non-risk with an\n"
        "# `overflow-ok:` comment are both preferable; growing it needs a reason in\n"
        "# the commit message.\n"
    )
    BASELINE_PATH.write_text(header + ("\n".join(keys) + "\n" if keys else ""), encoding="utf-8")
    return len(keys)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="list ALL address-shaped bare +/- sites")
    ap.add_argument("--update-baseline", action="store_true", help="rewrite the baseline from current gaps")
    ap.add_argument("--self-test", action="store_true", help="prove the detector fires on a synthetic gap")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    for d in SCAN_DIRS:
        if not d.is_dir():
            print(f"error: source dir not found: {d}", file=sys.stderr)
            return 2

    hits = scan()
    gaps = [h for h in hits if not h[4]]

    if args.list:
        for rel, lineno, fn, shape, exempt in hits:
            mark = "EXEMPT " if exempt else "GAP    "
            print(f"{mark} {rel}:{lineno}\t{fn}\t{shape}")
        print(
            f"\n{len(hits)} address-shaped bare +/- sites ({len(hits) - len(gaps)} documented-exempt, {len(gaps)} gaps)"
        )
        return 0

    if args.update_baseline:
        n = write_baseline(gaps)
        print(f"wrote {n} baseline keys to {BASELINE_PATH.relative_to(REPO_ROOT)}")
        return 0

    baseline = load_baseline()
    new_gaps = [h for h in gaps if _key(h[0], h[2], h[3]) not in baseline]

    if not new_gaps:
        print(f"OK: {len(hits)} address-shaped bare +/- sites, {len(gaps)} baselined, 0 new.")
        return 0

    print(f"FOUND {len(new_gaps)} new address-shaped bare +/- site(s):\n")
    for rel, lineno, fn, shape, _ in new_gaps:
        print(f"  {rel}:{lineno}\t{fn}\t{shape}")
    print(
        "\nA guest-controlled or Z3-derived address/size value feeding a bare +/-\n"
        "silently wraps in the shipped release build ([profile.release] has no\n"
        "overflow-checks). Fix with wrapping_*/saturating_*/checked_* -- see\n"
        "invariant-overflow-fix-refuse-not-saturate-identities for which one applies\n"
        "(saturating for sizes/lengths, checked+refuse for identity allocators like\n"
        "fd numbers). If this really is a bounds-checked, non-guest-controlled value\n"
        "(e.g. a usize loop counter), document it with an `overflow-ok: <why>` comment\n"
        "on the line above. Use --update-baseline only to defer a site you have\n"
        "deliberately decided not to fix yet."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
