#!/usr/bin/env python3
"""Silent-fallback tagging audit for the Rust core (bd angr-qwyti.1).

Ports the Python ``except Exception`` cat-x rationale convention (bd memory
``bare-except-cat-x-convention``) to ``native/angr/src/``. A *silent fallback*
is a site that discards an error/absent value and lets execution continue with
a degraded (stale / None / default) result -- exactly the shape that turns a
solver ``Unknown`` or a missing page into a wrong answer with no log line.

The convention: every such site that can affect *state correctness* carries a

    // SILENT(cat-a): <rationale>   expected control flow (planned absorb)
    // SILENT(cat-b): <rationale>   fallback with loss (degrades silently)
    // SILENT(cat-c): <rationale>   wrong-answer risk -- MUST also log::warn!

comment, either trailing the site's own line or in the comment/code run
directly above it (see :func:`_has_tag` for where that run stops). The tag
is annotation-only: this gate does not check that a cat-c site logs. A
fallback that must log goes through the ``silent_default!`` macro, whose
``cat_c`` arm logs warn.

This runs as a CI gate (``ci.yml::rust_check``) against a checked-in baseline: the detected shapes
have real false positives (plenty of legitimate self-only-by-design early
returns exist), so precision is proven against a checked-in *baseline* of the
currently-known-untagged sites. The baseline is a multiset: one line per
untagged occurrence, so a second identical site in a baselined file is new.
The script reports untagged sites beyond the baseline's count for their key,
and baseline lines that no longer match a live site (stale: prune them with
``--update-baseline``, a deletions-only diff). Either fails the gate; tag a
new site rather than growing the baseline.

Detected shapes (deliberately narrow -- the noisy ``.unwrap_or*`` /
``let _ =`` families are out of scope until precision is proven):

  * ``return Ok(None)``  -- early-return-Ok that drops a computed result
  * ``.ok();``           -- Result->Option swallow discarding the error; the
    same line matcher also catches it as the tail of a ``let`` binding
    (``let x = f().ok();``), the bound form

Test modules and the modules in ``UPSTREAM_OWNED`` (taken verbatim from
upstream angr, so not ours to tag) are not scanned.

Usage::

    tools/audit_silent_fallback.py                 # report new untagged sites
    tools/audit_silent_fallback.py --list          # list ALL detected sites
    tools/audit_silent_fallback.py --update-baseline
    tools/audit_silent_fallback.py --self-test     # prove it can fail

Exit 0 when there are no new untagged sites (or on --list / --update-baseline),
1 when new untagged sites are found. Pure stdlib; runs no symbolic execution.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

# The scanner below is this gate's own; only root resolution is shared.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rust_source_utils import ScanRootError, require_floor, rust_files, scan_roots, self_test_scan_roots

# Repo layout: this file lives at <repo>/tools/audit_silent_fallback.py
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "native" / "angr" / "src"
SCAN = {
    "native/angr": ("",),
    "native/angr-bv": ("",),
    "native/angr-vex": ("",),
    "native/angr-symex-core": ("",),
    "native/angr-symex-native": ("",),
}
EXCLUDE = {"native/angr-symex-macros": "proc-macro crate: a failure there is a compile error, not a runtime fallback"}
# Fewer non-test, fork-owned files than this means the code moved out of the scan roots.
MIN_FILES = 200
BASELINE_PATH = REPO_ROOT / "tools" / "silent_fallback_baseline.txt"

# Either the `// SILENT(cat-x): <rationale>` comment, or an invocation of the
# `silent_default!` macro (`lib.rs`), whose category argument *is* the tag —
# requiring a redundant comment beside it would push contributors back toward
# the hand-written form the macro exists to replace (angr-91vj9.6). The macro
# arm matches on the category, so an invocation cannot compile without one;
# that makes the bare `silent_default!(` enough, and it keeps matching when
# rustfmt wraps the category onto its own line (this regex is applied one line
# at a time, so anchoring on `silent_default!(\s*cat_c` would miss that shape).
# Where a match counts is :func:`_has_tag`'s business (angr-vmi2l.309).
TAG_RE = re.compile(r"//\s*SILENT\(cat-[abc]\):|silent_default!\(")

# Each shape: (name, compiled matcher over the *stripped* line).
SHAPES: list[tuple[str, re.Pattern[str]]] = [
    ("return-ok-none", re.compile(r"return Ok\(None\)")),
    # Match `=> return Ok(None),` arms too (covered by the substring above).
    ("ok-swallow", re.compile(r"\.ok\(\);\s*$")),
]


def _is_test_rel(rel: str) -> bool:
    """Whether ``rel`` (relative to ``SRC_ROOT``) is a test module.

    Test modules are skipped: their fallbacks are not production state paths.
    """
    return rel.endswith("_tests.rs") or "/tests/" in rel or rel.startswith("tests/")


# Modules taken verbatim from upstream angr (angr-mx00p.16). The SILENT(cat-x)
# convention is a fork process marker; tagging these would write it into
# upstream-owned files, so the gate skips them. A directory entry ends in `/`.
# Drop an entry once the fork takes ownership of that module.
UPSTREAM_OWNED = (
    "ailment/",
    "automaton/",
    "cfg_graph.rs",
    "function_graph.rs",
    "fuzzer.rs",
    "fuzzer/",
    "icicle.rs",
    "segmentlist.rs",
)


def _is_upstream_owned(rel: str) -> bool:
    """Whether ``rel`` (relative to ``SRC_ROOT``) is an upstream-owned module."""
    return any(rel.startswith(e) if e.endswith("/") else rel == e for e in UPSTREAM_OWNED)


def _iter_rs_files(repo_root: Path = REPO_ROOT) -> list[Path]:
    """The fork-owned production files the real run scans.

    Raises ScanRootError on a bad root or a scan below ``MIN_FILES``.
    """
    files = []
    for root in scan_roots(repo_root, SCAN, EXCLUDE):
        for p in rust_files([root]):
            rel = p.relative_to(root).as_posix()
            if _is_test_rel(rel) or _is_upstream_owned(rel):
                continue
            files.append(p)
    require_floor(files, MIN_FILES, "audit_silent_fallback")
    return files


def _key(rel: str, shape: str, stripped: str) -> str:
    """Line-number-independent baseline key.

    Keying on (path, shape, stripped-code) rather than a line number means a
    baselined site survives unrelated edits that shift it up or down the file;
    only genuinely new sites churn the baseline.
    """
    return f"{rel}\t{shape}\t{stripped}"


# An item header: the walk up from a site never crosses into another fn.
FN_HEADER_RE = re.compile(r"^(?:pub(?:\([^)]*\))?\s+)?(?:(?:async|const|unsafe|extern(?:\s+\"[^\"]*\")?)\s+)*fn\s")


# A trailing `// ...` comment, so `x.ok(); // SILENT(cat-b): why` is still a
# `.ok();` site. The leading whitespace keeps a `"http://"` literal intact.
TRAILING_COMMENT_RE = re.compile(r"\s+//.*$")


def _code(stripped: str) -> str:
    """A stripped line minus its trailing comment: what ``SHAPES`` match."""
    return TRAILING_COMMENT_RE.sub("", stripped)


def _is_site(stripped: str) -> bool:
    """Whether a stripped, non-comment line carries one of ``SHAPES``."""
    return not stripped.startswith("//") and any(m.search(_code(stripped)) for _, m in SHAPES)


def _has_tag(lines: list[str], idx: int) -> bool:
    """Whether the site on ``lines[idx]`` carries its own tag.

    The tag is on the site's line, or in the run of lines directly above it.
    The walk up stops at a blank line, another site, a fn header, or a line
    opening with ``}`` (the end of a sibling block). So one site's tag cannot
    exempt a neighbour, nor a site in the next fn (angr-vmi2l.309).
    """
    if TAG_RE.search(lines[idx]):
        return True
    for j in range(idx - 1, -1, -1):
        stripped = lines[j].strip()
        if not stripped or _is_site(stripped) or FN_HEADER_RE.match(stripped) or stripped.startswith("}"):
            return False
        if TAG_RE.search(stripped):
            return True
    return False


def scan_text(rel: str, text: str) -> list[tuple[str, int, str, str, bool]]:
    """Return (rel, lineno, shape, stripped, tagged) for every site in ``text``."""
    hits: list[tuple[str, int, str, str, bool]] = []
    lines = text.splitlines()
    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        # Ignore commented-out code so a `// return Ok(None)` note is inert.
        if stripped.startswith("//"):
            continue
        code = _code(stripped)
        for shape, matcher in SHAPES:
            if matcher.search(code):
                hits.append((rel, idx + 1, shape, code, _has_tag(lines, idx)))
                break
    return hits


def scan() -> list[tuple[str, int, str, str, bool]]:
    """Return (rel, lineno, shape, stripped, tagged) for every detected site."""
    hits: list[tuple[str, int, str, str, bool]] = []
    for path in _iter_rs_files():
        # Engine-crate keys stay SRC_ROOT-relative (the baseline's format);
        # another member's file is keyed from the repo root.
        base = SRC_ROOT if path.is_relative_to(SRC_ROOT) else REPO_ROOT
        rel = path.relative_to(base).as_posix()
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        hits.extend(scan_text(rel, text))
    return hits


# Synthetic sources for --self-test. The untagged file carries each detected
# shape (bare `return Ok(None)`, bare `.ok();`, the `let` tail) plus
# must-NOT-fire controls; the tagged file is the same sites, each with a
# `// SILENT(cat-b):` tag (one of them trailing). They are separate texts so
# the tagged one cannot bind into the untagged one.
_SELF_TEST_UNTAGGED = """\
fn early(&self) -> Result<Option<u64>> {
    if self.empty() {
        return Ok(None);
    }
    // return Ok(None) in a comment is inert
    Ok(Some(1))
}
fn swallow(&self) {
    self.flush().ok();
    let page = self.load(addr).ok();
    let kept = self.load(addr).ok().unwrap_or_default();
}
"""
_SELF_TEST_TAGGED = """\
fn early(&self) -> Result<Option<u64>> {
    if self.empty() {
        // SILENT(cat-b): why
        return Ok(None);
    }
    Ok(Some(1))
}
fn swallow(&self) {
    // SILENT(cat-b): why
    self.flush().ok();
    let page = self.load(addr).ok(); // SILENT(cat-b): why
}
"""
# Neighbour-tag evasions (angr-vmi2l.309): in each, a tag or a
# `silent_default!` belongs to something else, and the `.ok();` site is
# untagged. They cover a site in the next fn, the next site in a run, a
# blank-line gap, and a sibling block's tag.
_SELF_TEST_NEIGHBOURS = """\
fn a(&self) -> u64 {
    silent_default!(cat_b, self.parse(), 0, "unrelated")
}
fn b(&self) {
    self.flush().ok();
}
fn c(&self) -> Result<Option<u64>> {
    // SILENT(cat-a): expected
    return Ok(None);
}
fn d(&self) {
    // SILENT(cat-a): covers the first only
    self.first().ok();
    self.flush().ok();
    // SILENT(cat-a): a gap ends the comment run

    self.flush().ok();
    if x {
        // SILENT(cat-a): the if arm's own fallback
        self.first().ok();
    }
    self.flush().ok();
}
"""
_SELF_TEST_EXPECTED_NEIGHBOURS = [
    ("ok-swallow", "self.flush().ok();", False),
    ("return-ok-none", "return Ok(None);", True),
    ("ok-swallow", "self.first().ok();", True),
    ("ok-swallow", "self.flush().ok();", False),
    ("ok-swallow", "self.flush().ok();", False),
    ("ok-swallow", "self.first().ok();", True),
    ("ok-swallow", "self.flush().ok();", False),
]
_SELF_TEST_EXPECTED_UNTAGGED = [
    ("return-ok-none", "return Ok(None);", False),
    ("ok-swallow", "self.flush().ok();", False),
    ("ok-swallow", "let page = self.load(addr).ok();", False),
]
_SELF_TEST_EXPECTED_TAGGED = [(shape, code, True) for shape, code, _ in _SELF_TEST_EXPECTED_UNTAGGED]


def self_test() -> int:
    """Prove the detector can fail, can be silenced by a tag, and skips tests
    and upstream-owned modules.

    A gate never seen to fail is indistinguishable from one that cannot. The
    fixtures run through :func:`scan_text` (the real per-file matcher, with its
    comment skip and tag window), :func:`_is_test_rel` and
    :func:`_is_upstream_owned` (the real file skips), and the untagged hits
    through the real baseline filter.
    """
    failures: list[str] = []
    for name, src, want in (
        ("untagged", _SELF_TEST_UNTAGGED, _SELF_TEST_EXPECTED_UNTAGGED),
        ("tagged", _SELF_TEST_TAGGED, _SELF_TEST_EXPECTED_TAGGED),
        ("neighbours", _SELF_TEST_NEIGHBOURS, _SELF_TEST_EXPECTED_NEIGHBOURS),
    ):
        got = [(shape, code, tagged) for _, _, shape, code, tagged in scan_text("fixture.rs", src)]
        if got != want:
            failures.append(f"{name}: expected {want}\n  got      {got}")

    # A baselined untagged site is suppressed; the others still report.
    untagged = [h for h in scan_text("fixture.rs", _SELF_TEST_UNTAGGED) if not h[4]]
    baseline = Counter([_key("fixture.rs", "return-ok-none", "return Ok(None);")])
    fresh = [h[3] for h in new_untagged(untagged, baseline)]
    if fresh != ["self.flush().ok();", "let page = self.load(addr).ok();"]:
        failures.append(f"baseline: got {fresh}")
    if stale_baseline(untagged, baseline):
        failures.append(f"baseline: a used line reads stale: {stale_baseline(untagged, baseline)}")

    # The baseline counts: one line covers one of the four identical untagged
    # `self.flush().ok();` sites, so three are new; a line nothing uses is stale.
    flush = _key("fixture.rs", "ok-swallow", "self.flush().ok();")
    gone = _key("fixture.rs", "ok-swallow", "self.gone().ok();")
    untagged = [h for h in scan_text("fixture.rs", _SELF_TEST_NEIGHBOURS) if not h[4]]
    fresh_lines = [h[1] for h in new_untagged(untagged, Counter([flush, gone]))]
    if fresh_lines != [h[1] for h in untagged[1:]]:
        failures.append(f"baseline count: expected the last 3 of {[h[1] for h in untagged]}, got {fresh_lines}")
    if stale_baseline(untagged, Counter([flush, gone])) != [gone]:
        failures.append(f"stale baseline: got {stale_baseline(untagged, Counter([flush, gone]))}")

    for rel, want_test in (
        ("memory/page_tests.rs", True),
        ("memory/tests/page.rs", True),
        ("tests/page.rs", True),
        ("memory/page.rs", False),
    ):
        if _is_test_rel(rel) != want_test:
            failures.append(f"test-file skip: {rel!r} expected {want_test}")

    for rel, want_upstream in (
        ("ailment/ail_expr.rs", True),
        ("cfg_graph.rs", True),
        ("fuzzer/mod.rs", True),
        ("cfg_graph_extra.rs", False),
        ("memory/cfg_graph.rs", False),
        ("lib.rs", False),
    ):
        if _is_upstream_owned(rel) != want_upstream:
            failures.append(f"upstream-owned skip: {rel!r} expected {want_upstream}")

    failures.extend(self_test_scan_roots(_iter_rs_files, SCAN))

    if failures:
        print("SELF-TEST FAILED")
        for f in failures:
            print(f"  {f}")
        return 1
    print(
        "self-test OK: untagged return Ok(None) / .ok(); / let-tail flagged, "
        "the same sites tagged SILENT(cat-b) above or trailing pass, a neighbour's tag exempts nothing, "
        "the baseline counts occurrences and reports stale lines, test-file and upstream-owned skips honoured; "
        "an unclassified member, a missing root and a low file count fail"
    )
    return 0


def load_baseline() -> Counter[str]:
    """The baseline as a multiset: a key listed twice covers two sites."""
    if not BASELINE_PATH.exists():
        return Counter()
    keys: Counter[str] = Counter()
    for line in BASELINE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        keys[line] += 1
    return keys


def new_untagged(
    untagged: list[tuple[str, int, str, str, bool]], baseline: Counter[str]
) -> list[tuple[str, int, str, str, bool]]:
    """The untagged sites beyond the baseline's count for their key.

    Within one key the earliest sites are the baselined ones, so the
    surplus reported is the last occurrences in the file.
    """
    left = Counter(baseline)
    fresh = []
    for h in untagged:
        k = _key(h[0], h[2], h[3])
        if left[k] > 0:
            left[k] -= 1
        else:
            fresh.append(h)
    return fresh


def stale_baseline(untagged: list[tuple[str, int, str, str, bool]], baseline: Counter[str]) -> list[str]:
    """Baseline lines no live untagged site uses, one entry per surplus line."""
    surplus = baseline - Counter(_key(h[0], h[2], h[3]) for h in untagged)
    return sorted(surplus.elements())


def write_baseline(untagged: list[tuple[str, int, str, str, bool]]) -> int:
    keys = sorted(_key(rel, shape, stripped) for rel, _, shape, stripped, _ in untagged)
    header = (
        "# Silent-fallback audit baseline -- see tools/audit_silent_fallback.py\n"
        "# One line per known-untagged site: <relpath>\\t<shape>\\t<stripped-code>;\n"
        "# a key repeats once per occurrence.\n"
        "# Regenerate with: tools/audit_silent_fallback.py --update-baseline\n"
        "# Shrinking this file (by tagging a site with // SILENT(cat-x):) is the goal.\n"
    )
    BASELINE_PATH.write_text(header + "\n".join(keys) + "\n", encoding="utf-8")
    return len(keys)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="list ALL detected sites (tagged + untagged)")
    ap.add_argument("--update-baseline", action="store_true", help="rewrite the baseline from current untagged sites")
    ap.add_argument("--self-test", action="store_true", help="prove the detector fires on synthetic untagged sites")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    try:
        hits = scan()
    except ScanRootError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    untagged = [h for h in hits if not h[4]]

    if args.list:
        for rel, lineno, shape, stripped, tagged in hits:
            mark = "TAGGED " if tagged else "UNTAG  "
            print(f"{mark}{rel}:{lineno}\t[{shape}]\t{stripped}")
        print(f"\n{len(hits)} sites ({len(hits) - len(untagged)} tagged, {len(untagged)} untagged)")
        return 0

    if args.update_baseline:
        n = write_baseline(untagged)
        print(f"wrote {n} baseline keys to {BASELINE_PATH.relative_to(REPO_ROOT)}")
        return 0

    baseline = load_baseline()
    new_sites = new_untagged(untagged, baseline)
    stale = stale_baseline(untagged, baseline)

    if not new_sites and not stale:
        print(
            f"OK: {len(hits)} silent-fallback sites, "
            f"{len(hits) - len(untagged)} tagged, {len(untagged)} baselined, 0 new untagged, 0 stale."
        )
        return 0

    if new_sites:
        print(f"FOUND {len(new_sites)} new untagged silent-fallback site(s):\n")
        for rel, lineno, shape, stripped, _ in new_sites:
            print(f"  {rel}:{lineno}\t[{shape}]\t{stripped}")
        print(
            "\nAdd a `// SILENT(cat-a|b|c): <rationale>` comment directly above each site "
            "(or trailing it),\nwith no blank line between. A fallback that must log "
            "goes through `silent_default!`."
        )
    if stale:
        print(f"\nFOUND {len(stale)} stale baseline line(s) matching no live untagged site:\n")
        for key in stale:
            print(f"  {key}")
        print("\nPrune them with --update-baseline (the diff must be deletions only).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
