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

This is an AUDIT tool, NOT a hard CI gate (see the bead): the detected shapes
have real false positives (plenty of legitimate self-only-by-design early
returns exist), so precision is proven against a checked-in *baseline* of the
currently-known-untagged sites. The script reports only sites that are neither
tagged nor in the baseline -- i.e. *new* untagged silent fallbacks introduced
since the baseline was taken. Wire it as a blocking gate only once the
false-positive rate is understood (a later, separate bead).

Detected shapes (deliberately narrow -- the noisy ``.unwrap_or*`` /
``let _ =`` families are out of scope until precision is proven):

  * ``return Ok(None)``  -- early-return-Ok that drops a computed result
  * ``.ok();``           -- Result->Option swallow discarding the error
  * ``.ok()`` as the tail of a ``let`` binding -- same swallow, bound form

Usage::

    tools/audit_silent_fallback.py                 # report new untagged sites
    tools/audit_silent_fallback.py --list          # list ALL detected sites
    tools/audit_silent_fallback.py --update-baseline

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


def _iter_rs_files() -> list[Path]:
    files = []
    for p in sorted(SRC_ROOT.rglob("*.rs")):
        rel = p.relative_to(SRC_ROOT).as_posix()
        # Skip test modules: their fallbacks are not production state paths.
        if rel.endswith("_tests.rs") or "/tests/" in rel or rel.startswith("tests/"):
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


def scan() -> list[tuple[str, int, str, str, bool]]:
    """Return (rel, lineno, shape, stripped, tagged) for every detected site."""
    hits: list[tuple[str, int, str, str, bool]] = []
    for path in _iter_rs_files():
        rel = path.relative_to(SRC_ROOT).as_posix()
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
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
    args = ap.parse_args()

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
    new_sites = [h for h in untagged if _key(h[0], h[2], h[3]) not in baseline]

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
