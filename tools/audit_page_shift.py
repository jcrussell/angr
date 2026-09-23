#!/usr/bin/env python3
"""Open-coded page arithmetic audit (bd angr-fs8kb.91).

An address and a page number are both bare `u64`, and the conversion between
them is a shift by `PageIndex::SHIFT`. Open-coding that shift -- or the page
size / page mask it derives from -- is this repo's most persistent recurring
defect class: [`PageIndex`]'s own doc comment in `memory/page.rs` records that
*every* audit round has turned one up.

  * angr-c7xno.49 -- `PAGE_SIZE` expressed three incompatible ways inside
    `interpreter/` alone; motivated `Address::page_num` / `Address::page_base`
  * angr-sqfj8.140 -- `syscalls/page.rs` redefining the page constants
  * angr-fs8kb.70 -- five more sites, two of them *inside* the
    `end_page_inclusive` / `end_page_exclusive` helpers that exist to
    centralize the formula

Every other recurring class here has been closed by a baseline-gated audit
script (`audit_silent_fallback.py`, `check_line_citations.py`,
`audit_rounding_mode_threading.py`, `audit_sp_default_zero.py`,
`audit_overflow_wraparound.py`); this one had nothing, so it kept coming back
one review round at a time. This script closes that loop the same way: a
heuristic detector plus a checked-in baseline of already-triaged sites. Only
*new* sites fail.

Two detectors, over any non-test `.rs` file under `native/angr/src/`:

1. :data:`SHIFT_CONST_RE` / :data:`SHIFT_LIT_RE` -- a `>>`/`<<` by a page
   shift, either named (`PageIndex::SHIFT`, `Self::SHIFT`, `PAGE_SHIFT` --
   those constants exist for nothing else, so they are flagged unconditionally)
   or the bare literal `12` applied to an address/page-shaped operand
   (:data:`NAME_COMPONENTS`). The literal form needs the operand check because
   `1 << 12` is a perfectly ordinary bit constant.

2. :data:`LITERAL_RE` -- a page-size / page-mask *magic literal*
   (:data:`PAGE_LITERALS`) used as an operand of an arithmetic or masking
   operator whose other side is address/page-shaped, or passed to a method
   called on an address-shaped receiver (`length.div_ceil(0x1000)`). This is
   the same class one level down: `addr & !0xFFF` is `Address::page_base()`
   spelled by hand, and a site that hard-codes `0x1000` cannot follow if
   `PAGE_SIZE` ever changes. Adjacency to a shaped operand is what keeps the
   detector off the many legitimate `const MAX_...: usize = 4096;` buffer
   bounds, which sit to the right of an `=` and never next to an operator.

Sanctioned spellings are all *method calls* -- `Address::page_num`,
`Address::page_base`, `Address::page_offset`, `PageIndex::of`,
`PageIndex::base_addr`, `PageIndex::range_covering` -- plus the `PAGE_SIZE` /
`PAGE_MASK` constants, so (as with the overflow gate) the operator scan simply
never matches an already-fixed site and no exclusion list is needed.

A site is exempt when the marker :data:`EXEMPT_RE` -- `page-shift-ok:`
followed by a rationale -- appears on the site's own line or either of the two
lines above it. The four definitions of the conversion itself
(`Address::page_num` / `page_base` / `page_offset`, `PageIndex::of` /
`base_addr`) carry that marker: something has to do the shift.

Usage::

    tools/audit_page_shift.py                 # report new sites
    tools/audit_page_shift.py --list          # list ALL matched sites
    tools/audit_page_shift.py --self-test     # prove it can fail
    tools/audit_page_shift.py --update-baseline

Exit 0 when there are no new sites (or on `--list` / `--update-baseline`),
1 otherwise. Pure stdlib.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rust_source_utils import blank_noise, enclosing_fn, is_test_file

# Repo layout: this file lives at <repo>/tools/audit_page_shift.py
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "native" / "angr" / "src"
BASELINE_PATH = REPO_ROOT / "tools" / "page_shift_baseline.txt"

# Documented-exception marker, e.g.
#   // page-shift-ok: this *is* the definition of the address->page conversion.
EXEMPT_RE = re.compile(r"page-shift-ok:")

# `_`-separated name components that mark an operand as address/page-shaped.
# `len`/`length`/`size` are in the set because page *rounding* (`div_ceil`,
# align-up) is the second half of the same formula.
NAME_COMPONENTS = frozenset(
    {
        "addr",
        "address",
        "base",
        "brk",
        "end",
        "len",
        "length",
        "offset",
        "page",
        "pages",
        "pfn",
        "ptr",
        "size",
        "start",
    }
)

# Magic spellings of PAGE_SIZE / PAGE_MASK. `4095` and `0xFFF` are the mask;
# `4096` and `0x1000` the size.
PAGE_LITERALS = ("0x1000", "0X1000", "4096", "0xFFF", "0xfff", "0XFFF", "4095")

_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
# Dotted field access (`self.page.base`) or a plain identifier.
_TOKEN = rf"(?:{_IDENT}\.)*{_IDENT}"
_LIT = "|".join(re.escape(lit) for lit in PAGE_LITERALS)

# Detector 1a: shift by a *named* page shift. Those constants have exactly one
# purpose, so no operand check is needed. `\bSHIFT\b` deliberately does not
# match `arm_flag_shift::SHIFT_N` and friends.
SHIFT_CONST_RE = re.compile(rf"(?P<op>>>|<<)[ \t]*(?:{_IDENT}::)*(?:PAGE_SHIFT|SHIFT)\b")

# Detector 1b: shift by the bare literal 12, on an address/page-shaped operand.
SHIFT_LIT_RE = re.compile(rf"(?P<lhs>{_TOKEN})[ \t]*(?P<op>>>|<<)[ \t]*12\b")

# Detector 2: a page literal adjacent to an address/page-shaped operand, in
# any of the three shapes it takes in practice.
#   `addr & !0xFFF`   -- shaped operand on the left of an operator
#   `0x1000 + addr`   -- shaped operand on the right
#   `length.div_ceil(0x1000)` -- argument to a method on a shaped receiver
LITERAL_RE = re.compile(
    rf"(?P<lhs>{_TOKEN})[ \t]*(?P<op>[-+*/%&|])[ \t]*!?[ \t]*(?P<lit>{_LIT})\b"
    rf"|(?P<lit2>{_LIT})[ \t]*(?P<op2>[-+*/%&|])[ \t]*(?P<rhs>{_TOKEN})"
    rf"|(?P<recv>{_TOKEN})[ \t]*\.[ \t]*{_IDENT}[ \t]*\([ \t]*(?P<lit3>{_LIT})\b"
)


def _iter_source_files() -> list[Path]:
    """Non-test Rust sources under native/angr/src/."""
    return [p for p in sorted(SRC_DIR.rglob("*.rs")) if not is_test_file(p)]


def _is_shaped(operand: str) -> bool:
    """True when any `_`-separated component of the operand's last dotted
    segment is page-shaped.

    *Any* component rather than the trailing one (which is what
    ``audit_overflow_wraparound.py``'s compound detector keys on): here both
    ends of the conversion are shaped names and they put the marker in
    different positions -- ``page_num``/``page_addr`` lead with it,
    ``start_page``/``aligned_length`` trail it.
    """
    name = operand.rsplit(".", 1)[-1]
    return any(part.lower() in NAME_COMPONENTS for part in name.split("_"))


def _normalize(text: str) -> str:
    """Collapse whitespace so a wrapped expression and a one-liner share a key."""
    return re.sub(r"\s+", "", text)


def scan_text(rel: str, raw: str) -> list[tuple[str, int, str, str, bool]]:
    """Return (rel, lineno, fn_name, shape, exempt) per open-coded page-math site."""
    src = blank_noise(raw)
    raw_lines = raw.splitlines()
    found: dict[int, tuple[int, str]] = {}

    def record(start: int, end: int) -> None:
        # One hit per line: the same expression often matches two detectors
        # (`addr >> 12` vs `addr & 0xFFF`), and a per-line key keeps the
        # baseline stable when one of them is rewritten.
        lineno = src.count("\n", 0, start) + 1
        found.setdefault(lineno, (start, _normalize(src[start:end])))

    for m in SHIFT_CONST_RE.finditer(src):
        record(m.start(), m.end())
    for m in SHIFT_LIT_RE.finditer(src):
        if _is_shaped(m.group("lhs")):
            record(m.start(), m.end())
    for m in LITERAL_RE.finditer(src):
        operand = m.group("lhs") or m.group("rhs") or m.group("recv")
        if operand and _is_shaped(operand):
            record(m.start(), m.end())

    hits: list[tuple[str, int, str, str, bool]] = []
    for lineno, (start, shape) in sorted(found.items()):
        # The site's own line (so a trailing comment works) plus the two above.
        window = raw_lines[max(0, lineno - 3) : lineno]
        exempt = any(EXEMPT_RE.search(line) for line in window)
        hits.append((rel, lineno, enclosing_fn(src, start) or "<top-level>", shape, exempt))
    return hits


def scan() -> list[tuple[str, int, str, str, bool]]:
    hits: list[tuple[str, int, str, str, bool]] = []
    for path in _iter_source_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        hits.extend(scan_text(rel, path.read_text(encoding="utf-8", errors="replace")))
    return hits


# Synthetic source for --self-test: each detector's gap shape, the
# documented-exempt escape hatch, and the must-NOT-fire controls that made the
# operand/adjacency checks necessary in the first place.
_SELF_TEST_SRC = """
const MAX_FORMAT_LEN: usize = 4096;
const BLOCK_CACHE_CAPACITY: usize = 4096;

impl Thing {
    fn sanctioned(&self, addr: u64) -> u64 {
        Address::new(addr).page_base() + PageIndex::of(addr).get() + (addr & PAGE_MASK)
    }
    fn open_coded_shift(&self, addr: u64) -> u64 {
        addr >> 12
    }
    fn open_coded_unshift(&self, page_num: u64) -> u64 {
        page_num << 12
    }
    fn borrowed_const(&self, addr: u64) -> u64 {
        addr >> PageIndex::SHIFT
    }
    fn open_coded_mask(&self, addr: u64) -> u64 {
        addr & !0xFFF
    }
    fn open_coded_round(&self, length: u64) -> u64 {
        length.div_ceil(0x1000)
    }
    fn literal_on_the_left(&self, base: u64) -> u64 {
        0x1000 + base
    }
    fn plain_bit_constant(&self) -> u64 {
        1 << 12
    }
    fn unrelated_flag_shift(&self, dep1: u64) -> u64 {
        (dep1 >> arm_flag_shift::SHIFT_N) & 1
    }
    fn unrelated_buffer(&self) -> Vec<u8> {
        vec![0u8; 0x1000]
    }
    fn documented(&self, addr: u64) -> u64 {
        // page-shift-ok: this *is* the definition of the conversion.
        addr >> 12
    }
}
"""

_SELF_TEST_EXPECTED = [
    ("open_coded_shift", "addr>>12", False),
    ("open_coded_unshift", "page_num<<12", False),
    ("borrowed_const", ">>PageIndex::SHIFT", False),
    ("open_coded_mask", "addr&!0xFFF", False),
    ("open_coded_round", "length.div_ceil(0x1000", False),
    ("literal_on_the_left", "0x1000+base", False),
    ("documented", "addr>>12", True),
]


def self_test() -> int:
    """Prove the detector can actually fire (and can be exempted).

    A lint that has never been seen to fail is indistinguishable from one that
    cannot -- this repo has been bitten by that before (see the valgrind gate's
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
    print(f"self-test OK: {len(got)} sites classified as expected (6 gaps, 1 documented-exempt, 5 clean controls)")
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
        "# Open-coded page-arithmetic baseline -- see tools/audit_page_shift.py\n"
        "# One key per known site: <relpath>\\t<fn>\\t<normalized shape>.\n"
        "# Regenerate with: tools/audit_page_shift.py --update-baseline\n"
        "# This file should stay EMPTY. Routing through Address::page_num /\n"
        "# page_base / page_offset, PageIndex::of / base_addr / range_covering,\n"
        "# or the PAGE_SIZE / PAGE_MASK constants, or documenting the exception\n"
        "# with a `page-shift-ok:` comment, are all preferable; growing it needs\n"
        "# a reason in the commit message.\n"
    )
    BASELINE_PATH.write_text(header + ("\n".join(keys) + "\n" if keys else ""), encoding="utf-8")
    return len(keys)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="list ALL matched sites (gaps + exempt)")
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
        print(f"\n{len(hits)} page-math sites ({len(hits) - len(gaps)} documented-exempt, {len(gaps)} gaps)")
        return 0

    if args.update_baseline:
        n = write_baseline(gaps)
        print(f"wrote {n} baseline keys to {BASELINE_PATH.relative_to(REPO_ROOT)}")
        return 0

    baseline = load_baseline()
    new_gaps = [h for h in gaps if _key(h[0], h[2], h[3]) not in baseline]

    if not new_gaps:
        print(f"OK: {len(hits)} page-math sites, {len(gaps)} baselined, 0 new.")
        return 0

    print(f"FOUND {len(new_gaps)} site(s) open-coding address<->page arithmetic:\n")
    for rel, lineno, fn, shape, _ in new_gaps:
        print(f"  {rel}:{lineno}\t{fn}\t{shape}")
    print(
        "\nUse the centralized conversions instead -- a hand-written shift or a\n"
        "hard-coded 0x1000/0xFFF cannot follow if PAGE_SIZE changes, and every\n"
        "audit round so far has found one that had already drifted:\n"
        "  Address::page_num / page_base / page_offset  (memory/address.rs)\n"
        "  PageIndex::of / base_addr / range_covering   (memory/page.rs)\n"
        "  PAGE_SIZE / PAGE_MASK constants              (memory/page.rs)\n"
        "If the open-coded form genuinely is right here, document it with a\n"
        "`page-shift-ok: <why>` comment on the line above. Use --update-baseline\n"
        "only to defer a site you have deliberately decided not to fix."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
