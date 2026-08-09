#!/usr/bin/env python3
"""Stack-pointer / return-address "default to 0" audit (bd angr-91vj9.4).

Reading the stack pointer or the return address can fail: the register may hold
a symbolic value, or the `[sp]` slot may not be concretely readable. Every such
accessor therefore returns an `Option`/`RustBV`, and the *only* sanctioned way
to collapse that to a plain `u64` is one of the logged helpers:

  * `VEXInterpreter::get_stack_pointer_or_log` (`interpreter/prefetch.rs`)
  * `VEXInterpreter::get_return_addr_or_log`  (`interpreter/simprocedures.rs`)
  * `RustExplorationManager::get_return_addr_or_log` (`exploration/helpers.rs`)

Each is `SILENT(cat-c)`-tagged and emits a `log::warn!`, because a symbolic SP
silently becoming the literal `0` is a wrong-answer risk: the value is exported
verbatim (e.g. as `CallStackEntry.stack_ptr`) and downstream code cannot tell it
from a genuine SP of 0.

The helpers existed and were still bypassed. The identical shape has now been
found and fixed three separate times:

  * angr-sqfj8.62 — `get_return_addr().unwrap_or(0)` at 6 call sites
  * angr-sqfj8.63 — `get_offset_u64(sp_offset).unwrap_or(0)`
  * angr-c7xno.29 — `state.get_sp().as_u64().unwrap_or(0)` in
    `run_loop_single.rs` + `core_outcome_handlers.rs`, corrupting a symbolic SP
    into a bogus concrete pointer on the native-procedure return path

Nothing stopped a new call site from reaching for the raw accessor, since the
raw accessor and the helper are equally easy to reach for. This script closes
that loop the same way `tools/audit_silent_fallback.py`,
`tools/check_line_citations.py`, `tools/audit_steady_guard_coverage.py` and
`tools/audit_rounding_mode_threading.py` do: a heuristic detector plus a
checked-in baseline of already-known sites. Only *new* sites fail.

Detection heuristic — in any non-test `.rs` file under `native/angr/src/`, an
SP/return-address accessor (see :data:`ACCESSORS`) whose call is followed by a
pure method chain ending in `.unwrap_or(0)` / `.unwrap_or_default()`. The
generic register reader `get_offset_u64` only counts when its arguments mention
an `sp`-ish offset — it is a general accessor, and only the SP use is a
wrong-answer risk.

A site is exempt when the line above it (or its own trailing comment) carries a
marker matching :data:`EXEMPT_RE`, i.e. `sp-default-ok:` followed by a
rationale — the escape hatch for a site where 0 genuinely is the right answer
(e.g. an SP-relative *size* computation that is meaningless without a stack).

Usage::

    tools/audit_sp_default_zero.py                 # report new sites
    tools/audit_sp_default_zero.py --list          # list ALL accessor sites
    tools/audit_sp_default_zero.py --self-test     # prove it can fail
    tools/audit_sp_default_zero.py --update-baseline

Exit 0 when there are no new raw-default sites (or on `--list` /
`--update-baseline`), 1 otherwise. Pure stdlib.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rust_source_utils import blank_noise, enclosing_fn, is_test_file

# Repo layout: this file lives at <repo>/tools/audit_sp_default_zero.py
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "native" / "angr" / "src"
BASELINE_PATH = REPO_ROOT / "tools" / "sp_default_baseline.txt"

# Documented-exception marker, e.g.
#   // sp-default-ok: this is a stack *depth*, 0 is the correct empty answer.
EXEMPT_RE = re.compile(r"sp-default-ok:")

# Accessors whose failure mode is "symbolic/unreadable SP or return address".
ACCESSORS = (
    "get_sp",
    "get_sp_value",
    "get_stack_pointer",
    "get_return_addr",
    "get_offset_u64",
)
ACCESSOR_RE = re.compile(r"\b(?P<name>" + "|".join(ACCESSORS) + r")\s*\(")

# `get_offset_u64` is the generic "read register at offset" reader; only its
# SP use is in scope (angr-sqfj8.63 was `get_offset_u64(sp_offset)`).
SP_ARG_RE = re.compile(r"\bsp\b|sp_offset|stack_pointer", re.IGNORECASE)

# A method chain of simple calls (`.as_u64()`, `.map(f)`) terminating in a
# silent zero default. Anchored at the accessor's closing paren.
CHAIN_TO_ZERO_RE = re.compile(
    # No `\A` — `Pattern.match(s, pos)` already anchors at `pos`, whereas `\A`
    # would demand the *string* start and silently never fire.
    r"(?:\s*\.\s*[A-Za-z_][A-Za-z0-9_]*\s*\([^()]*\))*"
    r"\s*\.\s*(?P<term>unwrap_or\s*\(\s*0(?:u8|u16|u32|u64|usize)?\s*\)|unwrap_or_default\s*\(\s*\))"
)


def _iter_source_files() -> list[Path]:
    """Non-test Rust sources under native/angr/src/."""
    return [p for p in sorted(SRC_DIR.rglob("*.rs")) if not is_test_file(p)]


def _match_paren(src: str, open_idx: int) -> int | None:
    """Index just past the `)` matching the `(` at ``open_idx``."""
    depth, i, n = 0, open_idx, len(src)
    while i < n:
        c = src[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def _normalize(text: str) -> str:
    """Collapse whitespace so a wrapped chain and a one-liner share a key."""
    return re.sub(r"\s+", "", text)


def scan_text(rel: str, raw: str) -> list[tuple[str, int, str, str, bool]]:
    """Return (rel, lineno, fn_name, shape, exempt) per raw-default site."""
    hits: list[tuple[str, int, str, str, bool]] = []
    if not any(name in raw for name in ACCESSORS):
        return hits
    src = blank_noise(raw)
    raw_lines = raw.splitlines()

    for m in ACCESSOR_RE.finditer(src):
        end = _match_paren(src, m.end() - 1)
        if end is None:
            continue
        if m.group("name") == "get_offset_u64" and not SP_ARG_RE.search(src[m.end() : end]):
            continue
        chain = CHAIN_TO_ZERO_RE.match(src, end)
        if chain is None:
            continue
        lineno = src.count("\n", 0, m.start()) + 1
        shape = _normalize(src[m.start() : chain.end()])
        # The accessor's own line plus the two above it (a wrapped chain often
        # carries the marker on the line introducing the statement).
        window = raw_lines[max(0, lineno - 3) : lineno]
        exempt = any(EXEMPT_RE.search(line) for line in window)
        hits.append((rel, lineno, enclosing_fn(src, m.start()) or "<top-level>", shape, exempt))
    return hits


def scan() -> list[tuple[str, int, str, str, bool]]:
    hits: list[tuple[str, int, str, str, bool]] = []
    for path in _iter_source_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        hits.extend(scan_text(rel, path.read_text(encoding="utf-8", errors="replace")))
    return hits


# Synthetic source for --self-test: the three historically-fixed shapes, the
# documented-exempt escape hatch, and two must-NOT-fire controls.
_SELF_TEST_SRC = """
impl Thing {
    fn uses_the_helper(&self) -> u64 {
        // A comment saying get_sp().as_u64().unwrap_or(0) must not count.
        self.get_stack_pointer_or_log("call site")
    }
    fn propagates(&self, state: &RustSimState) -> Option<u64> {
        state.get_sp().as_u64()
    }
    fn raw_sp(&self, state: &RustSimState) -> u64 {
        state.get_sp().as_u64().unwrap_or(0)
    }
    fn raw_sp_value(&self) -> u64 {
        self.registers.get_sp_value().unwrap_or_default()
    }
    fn raw_ret(&self) -> u64 {
        self.get_return_addr().unwrap_or(0u64)
    }
    fn raw_offset(&self) -> u64 {
        self.registers.get_offset_u64(sp_offset, self.ctx).unwrap_or(0)
    }
    fn unrelated_offset(&self) -> u64 {
        self.registers.get_offset_u64(offset, self.ctx).unwrap_or(0)
    }
    fn documented(&self, state: &RustSimState) -> u64 {
        // sp-default-ok: a stack *depth*, 0 is the correct empty answer.
        state.get_sp().as_u64().unwrap_or(0)
    }
}
"""

_SELF_TEST_EXPECTED = [
    ("raw_sp", "get_sp().as_u64().unwrap_or(0)", False),
    ("raw_sp_value", "get_sp_value().unwrap_or_default()", False),
    ("raw_ret", "get_return_addr().unwrap_or(0u64)", False),
    ("raw_offset", "get_offset_u64(sp_offset,self.ctx).unwrap_or(0)", False),
    ("documented", "get_sp().as_u64().unwrap_or(0)", True),
]


def self_test() -> int:
    """Prove the detector can actually fail (and can be exempted).

    A lint that has never been seen to fire is indistinguishable from one that
    cannot — this repo has been bitten by that before (see the valgrind gate's
    `--self-test` note in CLAUDE.md). With an empty baseline the real check
    passes vacuously, so this runs ahead of it in CI.
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
    print(f"self-test OK: {len(got)} sites classified as expected (4 gaps, 1 documented-exempt, 3 clean controls)")
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
        "# SP / return-address default-to-zero baseline -- see tools/audit_sp_default_zero.py\n"
        "# One key per known raw-default site: <relpath>\\t<fn>\\t<normalized shape>.\n"
        "# Regenerate with: tools/audit_sp_default_zero.py --update-baseline\n"
        "# This file should stay EMPTY. Routing through the logged helpers\n"
        "# (VEXInterpreter::get_stack_pointer_or_log / get_return_addr_or_log,\n"
        "# RustExplorationManager::get_return_addr_or_log) or documenting the\n"
        "# exception with an `sp-default-ok:` comment are both preferable;\n"
        "# growing it needs a reason in the commit message.\n"
    )
    BASELINE_PATH.write_text(header + ("\n".join(keys) + "\n" if keys else ""), encoding="utf-8")
    return len(keys)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="list ALL raw-default sites (gaps + exempt)")
    ap.add_argument("--update-baseline", action="store_true", help="rewrite the baseline from current gaps")
    ap.add_argument("--self-test", action="store_true", help="prove the detector fires on a synthetic gap")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if not SRC_DIR.is_dir():
        print(f"error: rust source dir not found: {SRC_DIR}", file=sys.stderr)
        return 2

    hits = scan()
    gaps = [h for h in hits if not h[4]]

    if args.list:
        for rel, lineno, fn, shape, exempt in hits:
            mark = "EXEMPT " if exempt else "GAP    "
            print(f"{mark} {rel}:{lineno}\t{fn}\t{shape}")
        print(f"\n{len(hits)} raw-default sites ({len(hits) - len(gaps)} documented-exempt, {len(gaps)} gaps)")
        return 0

    if args.update_baseline:
        n = write_baseline(gaps)
        print(f"wrote {n} baseline keys to {BASELINE_PATH.relative_to(REPO_ROOT)}")
        return 0

    baseline = load_baseline()
    new_gaps = [h for h in gaps if _key(h[0], h[2], h[3]) not in baseline]

    if not new_gaps:
        print(f"OK: {len(hits)} raw-default sites, {len(gaps)} baselined, 0 new.")
        return 0

    print(f"FOUND {len(new_gaps)} site(s) silently defaulting a stack pointer / return address to 0:\n")
    for rel, lineno, fn, shape, _ in new_gaps:
        print(f"  {rel}:{lineno}\t{fn}\t{shape}")
    print(
        "\nRoute through the logged fallback helpers instead — a symbolic SP collapsing\n"
        "to the literal 0 is exported verbatim and cannot be told apart from a genuine\n"
        "SP of 0 downstream:\n"
        "  VEXInterpreter::get_stack_pointer_or_log (interpreter/prefetch.rs)\n"
        "  VEXInterpreter::get_return_addr_or_log   (interpreter/simprocedures.rs)\n"
        "  RustExplorationManager::get_return_addr_or_log (exploration/helpers.rs)\n"
        "For the native-procedure return path specifically, call\n"
        "`exploration::helpers::advance_sp_past_return_addr`, which bumps a symbolic\n"
        "SP symbolically instead of concretizing it.\n"
        "If 0 genuinely is the right answer here, document it with an\n"
        "`sp-default-ok: <why>` comment on the line above. Use --update-baseline only\n"
        "to defer a site you have deliberately decided not to fix."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
