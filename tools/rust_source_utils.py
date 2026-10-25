#!/usr/bin/env python3
"""Shared source-scanning helpers for the ``tools/audit_*.py`` baseline lints.

Before the importing audit scripts can look for their own pattern they need
the same groundwork: find the sources to scan (``scan_roots``, ``rust_files``,
``require_floor``), strip comments/string literals so prose cannot be
mistaken for code (``blank_noise``), skip Rust test modules
(``is_test_file``), and name the ``fn`` a match sits in (``enclosing_fn``).
Keeping one copy here means a fix to the (deliberately simple) blanker
benefits every lint that imports it.

Every gate, including the ones that keep their own scanner, resolves its
roots through ``scan_roots`` (angr-mx00p.116.9): roots come from the
repo-root ``Cargo.toml`` workspace members, so a crate added to the workspace
fails every gate until each one scans or excludes it, and a configured root
that moved or vanished fails instead of scanning nothing.

Pure stdlib, no repo imports — the audit scripts run standalone in CI.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "UPSTREAM_MEMBERS",
    "ScanRootError",
    "blank_noise",
    "enclosing_fn",
    "is_test_file",
    "require_floor",
    "rust_files",
    "scan_roots",
    "self_test_scan_roots",
    "workspace_members",
]

import re
import tempfile
import tomllib
from collections.abc import Callable, Mapping, Sequence

# Workspace members no gate scans: upstream-owned crates the fork does not
# write its process markers into.
UPSTREAM_MEMBERS = frozenset(
    {
        "native/clarirs-core",
        "native/clarirs-num",
        "native/clarirs-vsa",
        "native/clarirs-z3",
    }
)


class ScanRootError(Exception):
    """A gate's scan roots do not match the workspace, or it scanned too little."""


def workspace_members(repo_root: Path) -> list[str]:
    """The ``[workspace] members`` of ``repo_root/Cargo.toml``."""
    manifest = repo_root / "Cargo.toml"
    try:
        data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ScanRootError(f"cannot read workspace members from {manifest}: {e}") from e
    members = data.get("workspace", {}).get("members")
    if not members:
        raise ScanRootError(f"{manifest} lists no workspace members")
    return list(members)


def scan_roots(
    repo_root: Path,
    scan: Mapping[str, Sequence[str]],
    exclude: Mapping[str, str],
) -> list[Path]:
    """The directories a gate scans: ``<member>/src/<subdir>`` for each entry.

    ``scan`` maps a workspace member to the subdirectories of its ``src`` the
    gate covers (``("",)`` for the whole ``src``); ``exclude`` maps a member
    the gate deliberately skips to the reason. Raises :class:`ScanRootError`
    when a workspace member is in neither map (and is not one of
    :data:`UPSTREAM_MEMBERS`), when either map names a non-member, or when a
    configured directory is missing.
    """
    members = workspace_members(repo_root)
    unclassified = [m for m in members if m not in scan and m not in exclude and m not in UPSTREAM_MEMBERS]
    if unclassified:
        raise ScanRootError(
            f"workspace member(s) {unclassified} are neither scanned nor excluded by this gate: "
            "add each to its scan roots, or to its exclusions with a reason"
        )
    stale = [m for m in (*scan, *exclude) if m not in members]
    if stale:
        raise ScanRootError(f"configured member(s) {stale} are not workspace members of {repo_root / 'Cargo.toml'}")
    roots = []
    for member, subdirs in scan.items():
        for sub in subdirs:
            root = repo_root / member / "src" / sub
            if not root.is_dir():
                raise ScanRootError(f"scan root {root.relative_to(repo_root).as_posix()} is missing")
            roots.append(root)
    return roots


def rust_files(roots: Sequence[Path]) -> list[Path]:
    """Every ``.rs`` file under ``roots``, sorted, each once."""
    return sorted({p for root in roots for p in root.rglob("*.rs")})


def require_floor(files: Sequence[Path], floor: int, gate: str) -> None:
    """Raise :class:`ScanRootError` when ``gate`` scanned fewer than ``floor`` files.

    The floor catches a root that still exists but no longer holds the code
    (a move into another crate, a broken filter): a gate that scans nothing
    passes.
    """
    if len(files) < floor:
        raise ScanRootError(
            f"{gate}: scanned {len(files)} Rust file(s), below its floor of {floor}; "
            "did the code move out of its scan roots?"
        )


def self_test_scan_roots(
    scan_files: Callable[[Path], object],
    scan: Mapping[str, Sequence[str]],
    check_floor: bool = True,
) -> list[str]:
    """Prove a gate's own ``scan_files(repo_root)`` fails on a bad workspace.

    Runs it against fixture repos built from the real workspace members: an
    extra unclassified member, a configured root that is missing, and (with
    ``check_floor``) every root present but holding one file each, which is
    under any real floor. Returns one message per fixture that did NOT raise
    :class:`ScanRootError`.
    """
    members = workspace_members(Path(__file__).resolve().parent.parent)
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        fixtures = [("unclassified member", ["native/new-crate"], True), ("missing root", [], False)]
        if check_floor:
            fixtures.append(("file floor", [], True))
        for name, extra, populate in fixtures:
            repo = base / name.replace(" ", "_")
            repo.mkdir()
            listed = ", ".join(f'"{m}"' for m in [*members, *extra])
            (repo / "Cargo.toml").write_text(f"[workspace]\nmembers = [{listed}]\n", encoding="utf-8")
            if populate:
                for member, subdirs in scan.items():
                    for sub in subdirs:
                        root = repo / member / "src" / sub
                        root.mkdir(parents=True, exist_ok=True)
                        (root / "fixture.rs").write_text("fn f() {}\n", encoding="utf-8")
            try:
                scan_files(repo)
            except ScanRootError:
                continue
            failures.append(f"scan roots: the {name} fixture did not fail")
    return failures

_FN_RE = re.compile(r"\bfn\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*[(<]")


def blank_noise(src: str) -> str:
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


def is_test_file(path: Path | str) -> bool:
    """True for the repo's Rust test-module naming conventions."""
    p = Path(path)
    rel = p.as_posix()
    name = p.name
    return (
        # `*_tests_support.rs` is the shared-fixture module a >2000-line test
        # file splits into (see bd memory `invariant-file-size-split-convention`)
        # — same test-only status as the `*_tests.rs` siblings that consume it.
        name.endswith(("_tests.rs", "_tests_support.rs"))
        or name.startswith("tests_")
        or name in ("test_helpers.rs", "property_tests.rs")
        or "/tests/" in rel
        or rel.startswith("tests/")
        or any(part.endswith("_tests") for part in p.parent.parts)
    )


def enclosing_fn(src: str, pos: int) -> str | None:
    """Name of the nearest preceding ``fn`` declaration before ``pos``.

    An approximation — it does not track brace depth, so a hit in a module's
    top-level (outside any fn) is attributed to the last fn above it. Adequate
    for the audit lints, which only use the name to build a stable,
    line-number-independent baseline key.
    """
    last = None
    for m in _FN_RE.finditer(src, 0, pos):
        last = m
    return last.group("name") if last else None
