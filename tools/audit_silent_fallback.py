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

comment within a small window above the site. cat-a/b are annotation-only;
cat-c additionally demands a warn.

This runs as a CI gate (``ci.yml::rust_check``) against a checked-in baseline: the detected shapes
have real false positives (plenty of legitimate self-only-by-design early
returns exist), so precision is proven against a checked-in *baseline* of the
currently-known-untagged sites. The script reports only sites that are neither
tagged nor in the baseline -- i.e. *new* untagged silent fallbacks introduced
since the baseline was taken. A new untagged site fails the gate; tag it
rather than growing the baseline.

Detected shapes (deliberately narrow -- the noisy ``.unwrap_or*`` /
``let _ =`` families are out of scope until precision is proven):

  * ``return Ok(None)``  -- early-return-Ok that drops a computed result
  * ``.ok();``           -- Result->Option swallow discarding the error; the
    same line matcher also catches it as the tail of a ``let`` binding
    (``let x = f().ok();``), the bound form

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
from pathlib import Path

# Repo layout: this file lives at <repo>/tools/audit_silent_fallback.py
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "native" / "angr" / "src"
BASELINE_PATH = REPO_ROOT / "tools" / "silent_fallback_baseline.txt"

# Window (lines before / after the site) scanned for a SILENT(cat-x) tag.
# Mirrors the Python audit's 20-before / 12-after asymmetry: the rationale
# almost always precedes the swallow.
LOOKBACK = 20
LOOKAHEAD = 12

# Either the `// SILENT(cat-x): <rationale>` comment, or an invocation of the
# `silent_default!` macro (`lib.rs`), whose category argument *is* the tag —
# requiring a redundant comment beside it would push contributors back toward
# the hand-written form the macro exists to replace (angr-91vj9.6). The macro
# arm matches on the category, so an invocation cannot compile without one;
# that makes the bare `silent_default!(` enough, and it keeps matching when
# rustfmt wraps the category onto its own line (this regex is applied one line
# at a time, so anchoring on `silent_default!(\s*cat_c` would miss that shape).
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


def _iter_rs_files() -> list[Path]:
    files = []
    for p in sorted(SRC_ROOT.rglob("*.rs")):
        rel = p.relative_to(SRC_ROOT).as_posix()
        if _is_test_rel(rel):
            continue
        files.append(p)
    return files


def _key(rel: str, shape: str, stripped: str) -> str:
    """Line-number-independent baseline key.

    Keying on (path, shape, stripped-code) rather than a line number means a
    baselined site survives unrelated edits that shift it up or down the file;
    only genuinely new sites churn the baseline.
    """
    return f"{rel}\t{shape}\t{stripped}"


def _has_tag(lines: list[str], idx: int) -> bool:
    lo = max(0, idx - LOOKBACK)
    hi = min(len(lines), idx + LOOKAHEAD + 1)
    return any(TAG_RE.search(lines[j]) for j in range(lo, hi))


def scan_text(rel: str, text: str) -> list[tuple[str, int, str, str, bool]]:
    """Return (rel, lineno, shape, stripped, tagged) for every site in ``text``."""
    hits: list[tuple[str, int, str, str, bool]] = []
    lines = text.splitlines()
    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        # Ignore commented-out code so a `// return Ok(None)` note is inert.
        if stripped.startswith("//"):
            continue
        for shape, matcher in SHAPES:
            if matcher.search(stripped):
                hits.append((rel, idx + 1, shape, stripped, _has_tag(lines, idx)))
                break
    return hits


def scan() -> list[tuple[str, int, str, str, bool]]:
    """Return (rel, lineno, shape, stripped, tagged) for every detected site."""
    hits: list[tuple[str, int, str, str, bool]] = []
    for path in _iter_rs_files():
        rel = path.relative_to(SRC_ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        hits.extend(scan_text(rel, text))
    return hits


# Synthetic sources for --self-test. The untagged file carries each detected
# shape (bare `return Ok(None)`, bare `.ok();`, the `let` tail) plus
# must-NOT-fire controls; the tagged file is the same sites, each with a
# `// SILENT(cat-b):` tag. They are separate texts, so a tag in one cannot
# fall inside the other's LOOKBACK/LOOKAHEAD window.
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
    // SILENT(cat-b): why
    let page = self.load(addr).ok();
}
"""
_SELF_TEST_EXPECTED_UNTAGGED = [
    ("return-ok-none", "return Ok(None);", False),
    ("ok-swallow", "self.flush().ok();", False),
    ("ok-swallow", "let page = self.load(addr).ok();", False),
]
_SELF_TEST_EXPECTED_TAGGED = [(shape, code, True) for shape, code, _ in _SELF_TEST_EXPECTED_UNTAGGED]


def self_test() -> int:
    """Prove the detector can fail, can be silenced by a tag, and skips tests.

    A gate never seen to fail is indistinguishable from one that cannot. The
    fixtures run through :func:`scan_text` (the real per-file matcher, with its
    comment skip and tag window) and :func:`_is_test_rel` (the real test-file
    skip), and the untagged hits through the real baseline filter.
    """
    failures: list[str] = []
    for name, src, want in (
        ("untagged", _SELF_TEST_UNTAGGED, _SELF_TEST_EXPECTED_UNTAGGED),
        ("tagged", _SELF_TEST_TAGGED, _SELF_TEST_EXPECTED_TAGGED),
    ):
        got = [(shape, code, tagged) for _, _, shape, code, tagged in scan_text("fixture.rs", src)]
        if got != want:
            failures.append(f"{name}: expected {want}\n  got      {got}")

    # A baselined untagged site is suppressed; the others still report.
    untagged = [h for h in scan_text("fixture.rs", _SELF_TEST_UNTAGGED) if not h[4]]
    baseline = {_key("fixture.rs", "return-ok-none", "return Ok(None);")}
    fresh = [h[3] for h in new_untagged(untagged, baseline)]
    if fresh != ["self.flush().ok();", "let page = self.load(addr).ok();"]:
        failures.append(f"baseline: got {fresh}")

    for rel, want_test in (
        ("memory/page_tests.rs", True),
        ("memory/tests/page.rs", True),
        ("tests/page.rs", True),
        ("memory/page.rs", False),
    ):
        if _is_test_rel(rel) != want_test:
            failures.append(f"test-file skip: {rel!r} expected {want_test}")

    if failures:
        print("SELF-TEST FAILED")
        for f in failures:
            print(f"  {f}")
        return 1
    print(
        "self-test OK: untagged return Ok(None) / .ok(); / let-tail flagged, "
        "the same sites tagged SILENT(cat-b) pass, baseline and test-file skip honoured"
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


def new_untagged(
    untagged: list[tuple[str, int, str, str, bool]], baseline: set[str]
) -> list[tuple[str, int, str, str, bool]]:
    """The untagged sites whose line-number-independent key is not baselined."""
    return [h for h in untagged if _key(h[0], h[2], h[3]) not in baseline]


def write_baseline(untagged: list[tuple[str, int, str, str, bool]]) -> int:
    keys = sorted({_key(rel, shape, stripped) for rel, _, shape, stripped, _ in untagged})
    header = (
        "# Silent-fallback audit baseline -- see tools/audit_silent_fallback.py\n"
        "# One key per known-untagged site: <relpath>\\t<shape>\\t<stripped-code>.\n"
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

    if not SRC_ROOT.is_dir():
        print(f"error: source root not found: {SRC_ROOT}", file=sys.stderr)
        return 2

    hits = scan()
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

    if not new_sites:
        print(
            f"OK: {len(hits)} silent-fallback sites, "
            f"{len(hits) - len(untagged)} tagged, {len(untagged)} baselined, 0 new untagged."
        )
        return 0

    print(f"FOUND {len(new_sites)} new untagged silent-fallback site(s):\n")
    for rel, lineno, shape, stripped, _ in new_sites:
        print(f"  {rel}:{lineno}\t[{shape}]\t{stripped}")
    print(
        "\nAdd a `// SILENT(cat-a|b|c): <rationale>` comment above each site "
        "(cat-c must also log::warn!),\nor run --update-baseline if this is an "
        "intentional new fallback to defer."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
