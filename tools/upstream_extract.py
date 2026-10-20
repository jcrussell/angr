#!/usr/bin/env python3
"""Scrub fork-process references from code headed for an upstream PR (angr-mx00p.77).

The fork's working branch keeps its process vocabulary: bead ids
(``angr-xxxxx``), bd memory citations, the ``SILENT(cat-x)`` tag and the
``*-ok:`` exemption markers of the ``tools/audit_*.py`` gates. None of that
means anything upstream, so a PR branch is cut from fork files and then run
through this script, which rewrites the prose and refuses what it cannot.

Rewrites happen only in prose regions, never in code:

* ``.rs``: the text after ``//`` / ``///`` / ``//!`` (outside a string
  literal) and the body of ``reason = "..."`` strings;
* ``.py``: the text after ``#`` (outside a string literal) and lines inside
  triple-quoted strings;
* ``.rst`` / ``.md`` / ``.toml`` comments: the whole line (``.toml`` only after
  ``#``).

Rewrites, in order (:data:`RULES`):

1. a parenthetical made only of bead ids (``(angr-ab12c)``,
   ``(bd angr-ab12c.3, angr-de45f)``, ``(angr-ab12c Phase 2)``) is deleted;
2. a bead id inside a longer parenthetical is deleted with its separator
   (``(workspace Cargo.toml, angr-1cue)`` -> ``(workspace Cargo.toml)``);
3. a leading ``angr-xxxxx:`` / ``angr-xxxxx.N:`` tag is deleted;
4. ``SILENT(cat-a|b|c)`` and the ``overflow-ok:`` / ``page-shift-ok:`` /
   ``sp-default-ok:`` / ``rm-ignored:`` / ``satresult-exempt:`` markers are
   deleted, keeping the reason that follows;
5. a ``(bd memory `key`)`` / ``(see bd memory `key`)`` citation is deleted.

A comment line left empty by a rewrite is dropped. Anything still matching
:data:`LEFTOVER_RE` afterwards (a bare ``see angr-x``, a bead id used as a
noun, a bd memory named in running prose) cannot be rewritten mechanically:
it is reported with ``file:line`` and the run exits 1, so the comment gets a
hand rewrite on the fork (angr-mx00p.78) instead of a dangling reference.

:data:`EXCLUDE` lists the fork-only paths that must never enter an upstream
PR; ``--paths`` mode checks a file list (e.g. ``git diff --name-only``)
against it.

Usage::

    tools/upstream_extract.py --check FILE...    # report, change nothing
    tools/upstream_extract.py --apply FILE...    # rewrite in place
    tools/upstream_extract.py --paths FILE...    # flag excluded paths
    tools/upstream_extract.py --self-test

Exit 0 when nothing is left to fix by hand (and no excluded path was given),
1 otherwise. Pure stdlib; fork-only (listed in :data:`EXCLUDE` itself).
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
from pathlib import Path

BEAD = r"angr-[a-z0-9]{4,5}(?:\.\d+)*"
BEAD_REF = rf"(?:bd |bead |see |cf\. )?{BEAD}(?: [Pp]hase \w+| [Ss]lice \w+| pt\d+)?"

RULES: list[tuple[re.Pattern[str], str]] = [
    # 1. a parenthetical of bead ids only
    (re.compile(rf"\s*\((?:{BEAD_REF})(?:\s*(?:,|;|/|and)\s*{BEAD_REF})*\)"), ""),
    # 2. a bead id inside a longer parenthetical: trailing, then leading
    (re.compile(rf"\s*[,;]\s*{BEAD_REF}(?=\s*[)\]])"), ""),
    (re.compile(rf"(?<=[(\[]){BEAD_REF}\s*[,;]\s*"), ""),
    # 3. a leading tag
    (re.compile(rf"^(\s*)(?:bd )?{BEAD}(?: [Pp]hase \w+)?:\s+"), r"\1"),
    # 4. audit-gate markers
    (re.compile(r"SILENT\(cat-[abc]\)\s*:?\s*"), ""),
    (re.compile(r"\b(?:overflow-ok|page-shift-ok|sp-default-ok|rm-ignored|satresult-exempt):\s*"), ""),
    # 5. memory citations
    (re.compile(r"\s*\((?:see )?(?:the )?bd memor(?:y|ies) `[a-z0-9-]+`(?:\s*(?:,|and)\s*`[a-z0-9-]+`)*\)"), ""),
]

LEFTOVER_RE = re.compile(rf"{BEAD}|\bbd memor|\bbd (?:recall|remember|show|create)\b|\bbeads?\b|SILENT\(|-ok:")

EXCLUDE = [
    "AGENTS.md",
    "TRIAGE.md",
    "CLAUDE.md",
    ".github/copilot-instructions.md",
    ".claude/*",
    ".ralph/*",
    ".beads/*",
    ".cargo/*",
    "corpus_tests/*",
    "tools/ralph-*",
    "tools/audit_*",
    "tools/check_line_citations.py",
    "tools/rust_source_utils.py",
    "tools/upstream_extract.py",
    "tools/decisions/*",
    "tools/draft_patches/*",
    "tools/upstream_patches/*",
    "tools/*_baseline.txt",
    "tools/OFFLINE_WORKFLOWS.md",
    "docs/blog/*",
    "docs/dashboard/*",
    "docs/advanced-topics/perf/*",
    "docs/advanced-topics/rust_engine_decisions.rst",
    "docs/advanced-topics/rust_bimodal_variance.rst",
    ".github/workflows/perf-dashboard.yml",
    "tests/benchmarks/*",
]

RS_REASON_RE = re.compile(r'(reason\s*=\s*")((?:[^"\\]|\\.)*)(")')


def _scrub(text: str) -> str:
    for rx, rep in RULES:
        text = rx.sub(rep, text)
    return text


def _split_code_comment(line: str, marker: str) -> int | None:
    """Index where a line comment starts, skipping string literals."""
    in_str: str | None = None
    i = 0
    while i < len(line):
        c = line[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == in_str:
                in_str = None
        elif c in "\"'":
            # Rust lifetimes / char literals: only treat ' as a quote in Python
            if c == "'" and marker == "//":
                i += 1
                continue
            in_str = c
        elif line.startswith(marker, i):
            return i
        i += 1
    return None


def rewrite_text(path: str, src: str) -> tuple[str, list[tuple[int, str]]]:
    """Return (rewritten source, [(lineno, leftover line)])."""
    suffix = Path(path).suffix
    out: list[str] = []
    leftovers: list[tuple[int, str]] = []
    in_doc: str | None = None
    for lineno, line in enumerate(src.splitlines(keepends=True), 1):
        body = line.rstrip("\n")
        nl = line[len(body) :]
        new = body
        prose = ""
        if suffix in (".rst", ".md", ".txt"):
            new = _scrub(body)
            prose = new
        elif suffix == ".rs":
            new = RS_REASON_RE.sub(lambda m: m.group(1) + _scrub(m.group(2)) + m.group(3), body)
            prose = "".join(m.group(2) for m in RS_REASON_RE.finditer(new))
            k = _split_code_comment(new, "//")
            if k is not None:
                head, comment = new[:k], new[k:]
                m = re.match(r"(//[/!]?\s?)(.*)", comment)
                assert m is not None
                scrubbed = _scrub(m.group(2))
                new = head + m.group(1) + scrubbed
                prose += scrubbed
                if not scrubbed.strip() and m.group(2).strip():
                    new = head.rstrip()
        elif suffix in (".py", ".toml"):
            if in_doc is not None:
                new = _scrub(body)
                prose = new
                if in_doc in body:
                    in_doc = None
            else:
                k = _split_code_comment(body, "#")
                if k is not None:
                    head, comment = body[:k], body[k:]
                    m = re.match(r"(#\s?)(.*)", comment)
                    assert m is not None
                    scrubbed = _scrub(m.group(2))
                    new = head + m.group(1) + scrubbed
                    prose = scrubbed
                    if not scrubbed.strip() and m.group(2).strip():
                        new = head.rstrip()
                if suffix == ".py":
                    for q in ('"""', "'''"):
                        if body.count(q) == 1:
                            in_doc = q
                            new = _scrub(new)
                            prose += new
                            break
                        if body.count(q) >= 2:
                            new = _scrub(new)
                            prose += new
        if new != body and not new.strip() and body.strip():
            continue  # a comment-only line emptied by the scrub
        out.append(new.rstrip() + nl if new != body else line)
        if LEFTOVER_RE.search(prose):
            leftovers.append((lineno, new.strip()))
    return "".join(out), leftovers


def excluded(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pat) for pat in EXCLUDE)


SELF_TESTS: list[tuple[str, str, str, bool]] = [
    # (filename, input, expected output, expect leftover)
    ("a.rs", "    /// to eax clears rax[63:32] (angr-fs8kb.40).\n", "    /// to eax clears rax[63:32].\n", False),
    ("a.rs", "// angr-sqfj8.109: the loop above confirmed it\n", "// the loop above confirmed it\n", False),
    ("a.rs", "x += 1; // overflow-ok: counts are bounded\n", "x += 1; // counts are bounded\n", False),
    ("a.rs", "    // SILENT(cat-b): a failed peek just skips\n", "    // a failed peek just skips\n", False),
    (
        "a.rs",
        '    reason = "guard (workspace `Cargo.toml`, angr-1cue)"\n',
        '    reason = "guard (workspace `Cargo.toml`)"\n',
        False,
    ),
    ("a.rs", "    // (angr-ab12c)\n    let x = 1;\n", "    let x = 1;\n", False),
    ("a.rs", "    // see angr-ab12c for why\n", "    // see angr-ab12c for why\n", True),
    ("a.rs", '    let s = "angr-ab12c"; // keep\n', '    let s = "angr-ab12c"; // keep\n', False),
    (
        "a.rs",
        "// see bd memory `invariant-x` (angr-ab12c, angr-de45f Phase 2)\n",
        "// see bd memory `invariant-x`\n",
        True,
    ),
    ("a.rs", "// held off (bd memory `invariant-x`).\n", "// held off.\n", False),
    ("a.py", "    x = 1  # cat ok (angr-ab12c.3)\n", "    x = 1  # cat ok\n", False),
    (
        "a.py",
        '    """Doc.\n\n    Fixed in angr-ab12c.\n    """\n',
        '    """Doc.\n\n    Fixed in angr-ab12c.\n    """\n',
        True,
    ),
    ("a.py", '    """One line (angr-ab12c)."""\n', '    """One line."""\n', False),
    ("a.rst", "The pin (angr-iaol.1) holds.\n", "The pin holds.\n", False),
    ("a.rs", "    let p = 'a'; // angr-ab12c: note\n", "    let p = 'a'; // note\n", False),
]


def self_test() -> int:
    bad = 0
    for name, src, want, want_left in SELF_TESTS:
        got, left = rewrite_text(name, src)
        if got != want or bool(left) != want_left:
            bad += 1
            print(f"FAIL {name}: {src!r}\n  got  {got!r} leftovers={left}\n  want {want!r} leftovers={want_left}")
    for p, want in (("AGENTS.md", True), ("tools/audit_page_shift.py", True), ("native/angr/src/lib.rs", False)):
        if excluded(p) != want:
            bad += 1
            print(f"FAIL excluded({p!r}) != {want}")
    print("self-test:", "OK" if not bad else f"{bad} failure(s)")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--paths", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    ap.add_argument("files", nargs="*")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.paths:
        hits = [f for f in args.files if excluded(f)]
        for f in hits:
            print(f"excluded: {f}")
        return 1 if hits else 0
    total_left = 0
    changed = 0
    for f in args.files:
        p = Path(f)
        if p.suffix not in (".rs", ".py", ".rst", ".md", ".toml", ".txt") or not p.is_file():
            continue
        src = p.read_text()
        new, left = rewrite_text(f, src)
        if new != src:
            changed += 1
            if args.apply:
                p.write_text(new)
        for lineno, text in left:
            print(f"{f}:{lineno}: {text[:160]}")
        total_left += len(left)
    verb = "rewrote" if args.apply else "would rewrite"
    print(f"{verb} {changed} file(s); {total_left} line(s) need a hand rewrite", file=sys.stderr)
    return 1 if total_left else 0


if __name__ == "__main__":
    sys.exit(main())
