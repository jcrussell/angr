#!/usr/bin/env python3
"""Diff-based pairing check: new cat_b/cat_c silent fallbacks need a test.

`tools/audit_silent_fallback.py` gates the *tagging* convention (every silent
fallback in `native/angr/src/` carries a `// SILENT(cat-a|b|c):` rationale, or
uses the `silent_default!` macro). It is a whole-tree-vs-baseline scan and
deliberately does not look at test coverage -- adding the tag is enough to
satisfy it.

This script is a narrower, diff-based companion closing the next gap up. Per
the analysis in `/home/ubuntu/.claude/plans/we-keep-finding-bugs-optimized-acorn.md`
("What NOT to chase with more tests"), the two *logged* silent-fallback
categories --

    cat_b   fallback with loss (degrades silently, `log::debug!`)
    cat_c   wrong-answer risk (MUST also `log::warn!`)

-- are exactly the historically-recurring "silently wrong instead of loud
error" bug family (39 of the 162 audit-found logic bugs), and are too
heterogeneous for one generic sweep to cover proactively. What *does*
generalize, confirmed by a 25/25 sample of this repo's own bug-fix commits, is
the discipline of shipping a same-commit regression test proving the
fallback's chosen value is *correct* for the edge case, not merely "doesn't
crash". `cat_a` (expected control flow) is deliberately exempt -- there is no
"wrong answer" for it to get right, so a test would be vacuous busywork
(CLAUDE.md's "Silent-fallback tagging (Rust)" section).

The rule this script enforces: a diff that adds a new `silent_default!(cat_b`
or `silent_default!(cat_c` invocation, or a new `// SILENT(cat-b):` /
`// SILENT(cat-c):` comment, in `native/angr/src/*.rs` must *also* touch at
least one test file in the same diff. It cannot verify the test actually
covers the right thing -- that is unavoidably a human-review judgment call --
so a structurally-present pairing passes with a note asking the reviewer to
confirm relevance; only a *complete absence* of any touched test file fails.

Site detection deliberately mirrors (a cat_b/cat_c-only restriction of)
`TAG_RE` in `audit_silent_fallback.py`, kept as a literal sibling regex rather
than an import: the two scripts scan fundamentally different things (whole
tree vs. a diff) and have no shared runtime state, so importing one from the
other would just be action-at-a-distance coupling for a four-line regex. Keep
both in sync by hand if the tagging convention's syntax ever changes (both are
anchored in CLAUDE.md's "Silent-fallback tagging (Rust)" section, which is the
actual source of truth). `silent_default!(` alone is enough to flag the macro
form without also checking which category follows: the macro (`lib.rs`) has
no `cat_a` arm, so *any* invocation is inherently cat_b or cat_c already, and
the category token commonly lands on its own rustfmt-wrapped line rather than
beside the opening paren (see real call sites in `vex/opcode_map.rs`), which a
same-line `cat_b|cat_c` match would miss.

## Diff range

- ``--diff-file <path>`` (or ``-`` for stdin): read a unified diff verbatim,
  skip git entirely. Used by ``--self-test`` and by anyone who wants to feed a
  pre-built diff (e.g. from ``git diff`` piped through a filter).
- ``--base-ref <ref>``: diff ``<ref>...HEAD`` (merge-base form) if `<ref>` is
  locally resolvable, else diff literally against `<ref>`.
- Otherwise, PR context: GitHub Actions sets ``GITHUB_BASE_REF`` (just the
  branch name, e.g. ``master``) for ``pull_request`` events. `actions/checkout`
  defaults to a shallow, single-branch clone of the PR's own ref, so
  `origin/<base>` is never fetched -- this script fetches it itself
  (``--depth=1``; only the tree is needed, not history). If a merge-base
  against `HEAD` exists, diff `origin/<base>...HEAD` (scoped to just this
  branch's commits); a shallow clone can leave the two histories without a
  common ancestor, in which case fall back to the direct two-dot tree diff
  `origin/<base>..HEAD`.
- Otherwise (``push``/``merge_group``/``workflow_dispatch``, or local/non-PR
  use -- anything without ``GITHUB_BASE_REF``): diff against the previous
  commit, ``HEAD~1``. `actions/checkout` defaults to ``fetch-depth: 1`` (a
  single-commit clone), under which ``HEAD~1`` does not exist -- a bare
  ``git diff HEAD~1`` would exit 128 and, left unhandled, get swallowed by
  ``_run_diff``'s own error handling into a false "OK, no new sites" (this
  shipped as a real bug: the `rust_check` job's ``on:`` triggers include
  ``push`` to ``master`` and ``merge_group``, both of which hit this branch
  with no `GITHUB_BASE_REF` set). So before trusting ``HEAD~1``, deepen the
  shallow clone by one commit (``git fetch --deepen=1 origin <HEAD-sha>``) if
  it does not already resolve, then re-check. If it still does not resolve
  (genuinely no parent -- the repo's very first commit -- or the deepen fetch
  itself failed, e.g. no ``origin`` remote / no network), raise
  ``NoDiffRangeError`` rather than silently falling through to an empty diff:
  this script's entire purpose is catching silent failures, so it must not
  become one itself.

Usage::

    tools/audit_silent_fallback_test_pairing.py                    # PR / local diff
    tools/audit_silent_fallback_test_pairing.py --base-ref origin/master
    tools/audit_silent_fallback_test_pairing.py --diff-file some.patch
    tools/audit_silent_fallback_test_pairing.py --self-test        # prove it can pass/fail

Exit 0 = OK (no new cat_b/cat_c sites, or a test file was also touched),
1 = new cat_b/cat_c site(s) with no test file touched anywhere in the diff,
    OR no usable diff range could be determined (see ``NoDiffRangeError``
    above -- refusing to silently report OK when the comparison base is
    unknown).
Pure stdlib; shells out to ``git`` only to acquire the diff (not needed for
``--diff-file`` / ``--self-test``).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rust_source_utils import is_test_file

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATHSPEC = "native/angr/src"

# See the module docstring's "Site detection" paragraph for why these two
# alternatives (and not a same-line `cat_b|cat_c` match) are the right shape.
TAG_COMMENT_RE = re.compile(r"^//\s*SILENT\(cat-[bc]\):")
MACRO_CALL_RE = re.compile(r"silent_default!\(")

HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def _git(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False)


def _ref_exists(ref: str) -> bool:
    return _git(["rev-parse", "--verify", "--quiet", ref]).returncode == 0


class NoDiffRangeError(RuntimeError):
    """Raised when no diff-able range can be determined.

    This must never be swallowed into an empty diff -- that is precisely the
    "silently report OK" failure mode this script exists to prevent
    elsewhere (see the module docstring's "Diff range" section, last bullet).
    """


def determine_range(explicit_base: str | None) -> tuple[str, str]:
    """Pick a ``git diff``-able range. Returns (range_arg, human description).

    Raises ``NoDiffRangeError`` if none can be determined -- see that class's
    docstring for why this is not an empty-string / empty-diff return.
    """
    if explicit_base:
        if _ref_exists(explicit_base):
            return f"{explicit_base}...HEAD", f"explicit --base-ref {explicit_base} (merge-base diff)"
        return explicit_base, f"explicit --base-ref {explicit_base} (literal range)"

    base_ref = os.environ.get("GITHUB_BASE_REF")  # set only for pull_request[_target] events
    if base_ref:
        remote_ref = f"origin/{base_ref}"
        _git(["fetch", "--depth=1", "origin", base_ref])
        if _ref_exists(remote_ref):
            if _git(["merge-base", remote_ref, "HEAD"]).returncode == 0:
                return f"{remote_ref}...HEAD", f"PR base {remote_ref} (three-dot, merge-base found)"
            return f"{remote_ref}..HEAD", f"PR base {remote_ref} (two-dot, no merge-base in shallow clone)"

    # Non-PR trigger (push, merge_group, workflow_dispatch, or plain local
    # use with no GITHUB_BASE_REF set). Compare against the previous commit
    # -- but `actions/checkout` defaults to a single-commit (fetch-depth: 1)
    # shallow clone, so HEAD~1 does not exist yet. Deepen by one commit
    # before trusting it; this is a no-op (git no-ops the fetch) on an
    # already-unshallow local clone.
    if not _ref_exists("HEAD~1"):
        head_sha = _git(["rev-parse", "HEAD"]).stdout.strip()
        if head_sha:
            _git(["fetch", "--deepen=1", "origin", head_sha])

    if _ref_exists("HEAD~1"):
        return "HEAD~1", "HEAD~1 (previous commit, deepened shallow clone if needed)"

    # Genuinely no parent commit is reachable -- either this really is the
    # repo's first commit, or the deepen fetch above failed for some
    # external reason (no `origin` remote, no network, etc). Silently
    # falling through to an empty diff here would report "OK: no new
    # cat_b/cat_c silent-fallback sites" no matter what the push actually
    # changed -- exactly the bug this script was written to catch elsewhere.
    # Fail loudly instead of guessing.
    raise NoDiffRangeError(
        "could not resolve a diff range: HEAD~1 does not exist (even after "
        "attempting to deepen the clone by one commit) and no GITHUB_BASE_REF "
        "is set. This is expected only for the repository's very first "
        "commit; otherwise it means the deepen fetch failed (no 'origin' "
        "remote, no network, etc). Refusing to silently report OK -- pass "
        "--base-ref explicitly to specify a comparison point."
    )


def _run_diff(range_arg: str, pathspec: str | None) -> str:
    args = ["diff", "--unified=0", "--no-color", range_arg]
    if pathspec:
        args += ["--", pathspec]
    proc = _git(args)
    if proc.returncode != 0:
        print(f"warning: `git diff {' '.join(args)}` failed: {proc.stderr.strip()}", file=sys.stderr)
        return ""
    return proc.stdout


def _run_name_only(range_arg: str) -> list[str]:
    proc = _git(["diff", "--name-only", range_arg])
    if proc.returncode != 0:
        print(f"warning: `git diff --name-only {range_arg}` failed: {proc.stderr.strip()}", file=sys.stderr)
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def parse_added_lines(diff_text: str) -> list[tuple[str, int, str]]:
    """Return (relpath, new_lineno, content-without-leading-plus) per ``+`` line.

    Handles both a ``--unified=0`` diff (context lines absent) and an ordinary
    one (context lines present, only used to keep ``new_lineno`` accurate).
    """
    hits: list[tuple[str, int, str]] = []
    current_file: str | None = None
    new_lineno: int | None = None
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            path = line[4:].split("\t", 1)[0]
            current_file = None if path == "/dev/null" else path.removeprefix("b/")
            new_lineno = None
            continue
        if line.startswith("@@"):
            m = HUNK_HEADER_RE.match(line)
            new_lineno = int(m.group(1)) if m else None
            continue
        if line.startswith("+"):
            if current_file is not None and new_lineno is not None:
                hits.append((current_file, new_lineno, line[1:]))
                new_lineno += 1
            continue
        if line.startswith(("-", "\\")):
            continue  # removed line / "\ No newline at end of file": doesn't advance new_lineno
        if new_lineno is not None:
            new_lineno += 1  # context line
    return hits


def _is_new_lossy_site(stripped: str) -> bool:
    """True for an added line that is a new cat_b/cat_c silent-fallback site.

    A ``//``-prefixed line only counts via the explicit tag comment (so a
    commented-out ``silent_default!(cat_b, ...)`` call, or a doc comment that
    merely *mentions* the macro as a usage example -- see `lib.rs`'s own doc
    comment for the macro -- is correctly not a site). A non-comment line
    counts via the macro-call form.
    """
    if stripped.startswith("//"):
        return bool(TAG_COMMENT_RE.match(stripped))
    return bool(MACRO_CALL_RE.search(stripped))


def find_new_sites(diff_text: str) -> list[tuple[str, int, str]]:
    """New cat_b/cat_c sites added in ``diff_text``, restricted to non-test .rs files."""
    hits = []
    for relpath, lineno, content in parse_added_lines(diff_text):
        if not relpath.endswith(".rs") or is_test_file(relpath):
            continue
        stripped = content.strip()
        if _is_new_lossy_site(stripped):
            hits.append((relpath, lineno, stripped))
    return hits


def has_paired_test(touched: list[str]) -> bool:
    return any(is_test_file(f) for f in touched)


def evaluate(diff_text: str, touched: list[str]) -> tuple[int, str]:
    """Return (exit_code, message) for a given (site-scan diff, touched-files list)."""
    sites = find_new_sites(diff_text)
    if not sites:
        return 0, "OK: no new cat_b/cat_c silent-fallback sites in this diff."

    if has_paired_test(touched):
        lines = [
            f"NOTE: {len(sites)} new cat_b/cat_c silent-fallback site(s) found, "
            "and this diff also touches a test file:",
        ]
        for rel, lineno, stripped in sites:
            lines.append(f"  {rel}:{lineno}\t{stripped}")
        lines.append(
            "\nPairing looks structurally present -- a reviewer should still confirm "
            "the test actually asserts the fallback's chosen value/behavior for the "
            "new edge case, not just that nothing panics."
        )
        return 0, "\n".join(lines)

    lines = [f"FOUND {len(sites)} new cat_b/cat_c silent-fallback site(s) with no test file touched:\n"]
    for rel, lineno, stripped in sites:
        lines.append(f"  {rel}:{lineno}\t{stripped}")
    lines.append(
        "\ncat_b/cat_c mark a fallback that silently degrades or risks a wrong answer -- "
        "exactly the shape this repo's audits keep re-finding (39 historical instances,\n"
        'see CLAUDE.md\'s "Silent-fallback tagging (Rust)" section). Add a regression test '
        "in the same change proving the fallback's chosen default is *correct* for this\n"
        "edge case (not just that it doesn't panic), touching a file matching `*_tests.rs` "
        "or under a `tests/` directory. If this really is `cat_a` (expected control flow,\n"
        "no wrong answer to get right), use that tag instead -- it is exempt from this check."
    )
    return 1, "\n".join(lines)


# ---------------------------------------------------------------------------
# --self-test: synthetic diffs proving the check can both pass and fail.
# A lint that has never been seen to fire is indistinguishable from one that
# cannot (same reasoning as the valgrind gate's --self-test, CLAUDE.md).
# ---------------------------------------------------------------------------


def _synthetic_diff(path: str, added_lines: list[str], hunk_start: int = 10) -> str:
    body = "\n".join(f"+{line}" for line in added_lines)
    count = len(added_lines)
    return (
        f"diff --git a/{path} b/{path}\n"
        f"index 1111111..2222222 100644\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        f"@@ -{hunk_start},0 +{hunk_start},{count} @@\n"
        f"{body}\n"
    )


_SELF_TEST_CASES: list[tuple[str, str, list[str], int]] = [
    # (name, diff_text, touched_files, expected_exit_code)
    (
        "macro-call, no test touched -> FAIL",
        _synthetic_diff(
            "native/angr/src/vex/opcode_map.rs",
            [
                "    silent_default!(",
                "        cat_c,",
                "        parse_type(ty_str),",
                "        IRType::I64,",
                '        "unknown type"',
                "    )",
            ],
        ),
        ["native/angr/src/vex/opcode_map.rs"],
        1,
    ),
    (
        "macro-call, test file also touched -> PASS (structural)",
        _synthetic_diff(
            "native/angr/src/vex/opcode_map.rs",
            ["    silent_default!(", "        cat_b,", "        parsed,", "        0,", '        "msg"', "    )"],
        ),
        ["native/angr/src/vex/opcode_map.rs", "native/angr/src/vex/opcode_map_tests.rs"],
        0,
    ),
    (
        "tag comment cat-c, no test touched -> FAIL",
        _synthetic_diff(
            "native/angr/src/arch/mod.rs",
            [
                "// SILENT(cat-c): a symbolic SP collapsing to 0 is a wrong-answer risk.",
                "let sp = 0u64;",
            ],
        ),
        ["native/angr/src/arch/mod.rs"],
        1,
    ),
    (
        "tag comment cat-a (exempt) -> PASS, not even flagged",
        _synthetic_diff(
            "native/angr/src/state/mod.rs",
            ["// SILENT(cat-a): expected control flow, planned absorb.", "return Ok(None);"],
        ),
        ["native/angr/src/state/mod.rs"],
        0,
    ),
    (
        "commented-out macro call (dead code) -> PASS, not a real site",
        _synthetic_diff(
            "native/angr/src/state/mod.rs",
            ['// silent_default!(cat_b, old_expr(), 0, "dead code, do not use")'],
        ),
        ["native/angr/src/state/mod.rs"],
        0,
    ),
    (
        "unrelated diff, no silent-fallback lines -> PASS",
        _synthetic_diff("native/angr/src/state/mod.rs", ["let x = 1 + 1;", 'println!("{x}");']),
        ["native/angr/src/state/mod.rs"],
        0,
    ),
    (
        "new site in a *_tests.rs file itself -> PASS, test files excluded from site scan",
        _synthetic_diff(
            "native/angr/src/silent_default_tests.rs",
            ['    let got = silent_default!(cat_c, None::<u64>, 0, "demo");'],
        ),
        ["native/angr/src/silent_default_tests.rs"],
        0,
    ),
]


# ---------------------------------------------------------------------------
# --self-test, part 2: real-git scenarios exercising determine_range() itself.
#
# Every case in `_SELF_TEST_CASES` above hands a pre-built diff straight to
# `evaluate()`, bypassing `determine_range()` / `_run_diff()` / `_run_name_only()`
# entirely -- so that suite could never have caught the shipped bug: a
# `push`/`merge_group` trigger (no `GITHUB_BASE_REF`) hitting a shallow
# (fetch-depth: 1) `actions/checkout` clone, where `HEAD~1` doesn't exist,
# `git diff HEAD~1` exits 128, and `_run_diff`'s own error handling quietly
# turned that into an empty diff and a false "OK". These scenarios build real,
# disposable git repos and invoke the script as a subprocess against them, to
# prove the deepen-or-fail-loudly logic in `determine_range()` actually
# engages -- not just that `evaluate()` classifies a hand-built diff
# correctly.
#
# `determine_range()`'s git calls are pinned to `REPO_ROOT`
# (`Path(__file__).resolve().parent.parent`) by design, so it works
# regardless of the caller's cwd against whatever repo it actually lives in.
# To exercise it against a *disposable* repo (so these scenarios never touch
# this actual repo's git history), the script has to physically live under
# that disposable repo's `tools/` directory -- hence copying it there rather
# than invoking it in place.
# ---------------------------------------------------------------------------


def _write_tool_copies(dest_tools_dir) -> None:
    import shutil

    dest_tools_dir.mkdir(parents=True, exist_ok=True)
    src_dir = Path(__file__).resolve().parent
    for name in ("audit_silent_fallback_test_pairing.py", "rust_source_utils.py"):
        shutil.copy(src_dir / name, dest_tools_dir / name)


def _init_git_repo(repo) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "selftest@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Self Test"], cwd=repo, check=True)


def _commit_all(repo, message: str) -> None:
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo, check=True)


def _run_script_subprocess(repo):
    env = dict(os.environ)
    env.pop("GITHUB_BASE_REF", None)  # simulate a push/merge_group trigger, not pull_request
    return subprocess.run(
        [sys.executable, "tools/audit_silent_fallback_test_pairing.py"],
        cwd=repo,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _self_test_no_parent_commit_fails_loudly() -> str | None:
    """The repo's very first commit, no ``origin`` remote at all.

    ``HEAD~1`` genuinely cannot exist and there is nothing to deepen from --
    the same observable shape as the shipped bug's shallow-clone case (no
    ``GITHUB_BASE_REF``, ``HEAD~1`` unresolvable). Must fail loudly, not
    print "OK".
    """
    import tempfile

    with tempfile.TemporaryDirectory(prefix="audit_pairing_selftest_noparent_") as tmp:
        repo = Path(tmp)
        _init_git_repo(repo)
        _write_tool_copies(repo / "tools")
        (repo / "native" / "angr" / "src").mkdir(parents=True)
        (repo / "native" / "angr" / "src" / "placeholder.rs").write_text("// placeholder\n")
        _commit_all(repo, "single initial commit, no parent, no origin")

        proc = _run_script_subprocess(repo)
        if proc.returncode == 0:
            return (
                "no-parent-commit scenario: expected a non-zero (loud-failure) exit, got 0 -- "
                f"stdout: {proc.stdout!r} stderr: {proc.stderr!r}"
            )
        if "OK:" in proc.stdout:
            return (
                "no-parent-commit scenario: silently printed an OK message instead of failing -- "
                f"this is exactly the shipped bug. stdout: {proc.stdout!r}"
            )
        if "NoDiffRangeError" not in proc.stderr and "could not resolve a diff range" not in proc.stderr:
            return (
                "no-parent-commit scenario: expected a diff-range error on stderr, got: "
                f"{proc.stderr!r} (stdout: {proc.stdout!r})"
            )
    return None


def _self_test_shallow_clone_deepens_and_succeeds() -> str | None:
    """A real shallow (depth-1) clone with genuine history behind it.

    Mirrors what ``actions/checkout`` leaves on a ``push`` runner: a
    single-commit clone of a repo that *does* have a parent commit and an
    ``origin`` remote to fetch it from. Confirms ``HEAD~1`` is unresolvable
    beforehand, that the script's deepen step makes it resolvable, and that
    the script exits 0 with a real (non-error) diff-range description --
    proving the fix's happy path, not just its failure path.
    """
    import tempfile

    with tempfile.TemporaryDirectory(prefix="audit_pairing_selftest_shallow_") as tmp:
        upstream = Path(tmp) / "upstream"
        shallow = Path(tmp) / "shallow"
        upstream.mkdir()
        _init_git_repo(upstream)
        _write_tool_copies(upstream / "tools")
        (upstream / "native" / "angr" / "src").mkdir(parents=True)
        (upstream / "native" / "angr" / "src" / "placeholder.rs").write_text("// placeholder v1\n")
        _commit_all(upstream, "parent commit")
        (upstream / "native" / "angr" / "src" / "placeholder.rs").write_text("// placeholder v2\n")
        _commit_all(upstream, "child commit (what the shallow clone will see as HEAD)")

        clone = subprocess.run(
            ["git", "clone", "--quiet", "--no-local", "--depth", "1", str(upstream), str(shallow)],
            capture_output=True,
            text=True,
            check=False,
        )
        if clone.returncode != 0:
            return f"shallow-clone scenario: `git clone --depth 1` setup failed: {clone.stderr!r}"

        pre_check = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", "HEAD~1"],
            cwd=shallow,
            capture_output=True,
            text=True,
            check=False,
        )
        if pre_check.returncode == 0:
            return "shallow-clone scenario: setup bug -- HEAD~1 already resolves before the script even ran"

        proc = _run_script_subprocess(shallow)
        if proc.returncode != 0:
            return (
                "shallow-clone scenario: expected exit 0 (deepen should have recovered a real diff "
                f"range), got {proc.returncode}. stdout: {proc.stdout!r} stderr: {proc.stderr!r}"
            )
        if "HEAD~1" not in proc.stderr:
            return f"shallow-clone scenario: expected the HEAD~1 diff-range description on stderr, got: {proc.stderr!r}"

        post_check = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", "HEAD~1"],
            cwd=shallow,
            capture_output=True,
            text=True,
            check=False,
        )
        if post_check.returncode != 0:
            return (
                "shallow-clone scenario: HEAD~1 still does not resolve after running the script -- "
                "the deepen fetch did not actually happen"
            )
    return None


_GIT_RANGE_SELF_TEST_SCENARIOS = [
    ("no parent commit, no origin -> fails loudly, never OK", _self_test_no_parent_commit_fails_loudly),
    ("shallow clone with real history -> deepens HEAD~1, succeeds", _self_test_shallow_clone_deepens_and_succeeds),
]


def self_test() -> int:
    failures = []
    for name, diff_text, touched, expected in _SELF_TEST_CASES:
        got, _msg = evaluate(diff_text, touched)
        if got != expected:
            failures.append(f"{name}: expected exit {expected}, got {got}")

    for name, scenario in _GIT_RANGE_SELF_TEST_SCENARIOS:
        error = scenario()
        if error:
            failures.append(f"{name}: {error}")

    if failures:
        print("SELF-TEST FAILED:")
        for msg in failures:
            print(f"  {msg}")
        return 1

    print(
        f"self-test OK: {len(_SELF_TEST_CASES)} synthetic diff scenarios + "
        f"{len(_GIT_RANGE_SELF_TEST_SCENARIOS)} real-git determine_range() scenarios classified as expected."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-ref", help="diff against this ref instead of auto-detecting one")
    ap.add_argument("--diff-file", help="read a unified diff from this path (or '-' for stdin) instead of running git")
    ap.add_argument("--self-test", action="store_true", help="prove the check can both pass and fail")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if args.diff_file:
        text = sys.stdin.read() if args.diff_file == "-" else Path(args.diff_file).read_text(encoding="utf-8")
        # A pre-built diff carries every changed file; reuse it for both the
        # site scan (filtered to native/angr/src below) and the touched-files
        # check, rather than shelling out to git twice.
        site_diff_text = text
        touched = []
        current_file = None
        for line in text.splitlines():
            if line.startswith("+++ "):
                path = line[4:].split("\t", 1)[0]
                current_file = None if path == "/dev/null" else path.removeprefix("b/")
                if current_file:
                    touched.append(current_file)
        code, message = evaluate(site_diff_text, touched)
        print(message)
        return code

    try:
        range_arg, description = determine_range(args.base_ref)
    except NoDiffRangeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"diff range: {description}", file=sys.stderr)
    site_diff_text = _run_diff(range_arg, SRC_PATHSPEC)
    touched = _run_name_only(range_arg)
    code, message = evaluate(site_diff_text, touched)
    print(message)
    return code


if __name__ == "__main__":
    sys.exit(main())
