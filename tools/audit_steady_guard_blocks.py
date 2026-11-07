#!/usr/bin/env python3
"""Require ``steady_guard_checked`` on every ``#[pymethods] impl RustExplorationManager`` (angr-vmi2l.226).

``#[angr_symex_macros::steady_guard_checked]`` makes every ``&mut self``
method of the block it sits on carry ``#[steady_guarded]`` or
``#[steady_guard_exempt(reason = "…")]``, so a Python-callable mutator cannot
skip ``steady_config_guard()`` and change config a live steady session has
already snapshotted. The macro checks only the blocks it is written on. A new
``manager_methods_<area>.rs`` split whose block leaves it off compiles, and
its mutators run unguarded. This gate closes that gap at the block level.

Detection: in the PyO3 shell (``native/angr/src/``, test files included),
each inherent ``impl RustExplorationManager {`` whose run of outer attributes
contains one whose path ends in ``pymethods`` must also carry one whose path
ends in ``steady_guard_checked``, written *above* the ``pymethods`` one.
Attributes expand top-down, so a checker written below ``#[pymethods]``
would only see pyo3's expansion. Comments and string literals are blanked
first (``rust_source_utils.blank_noise``), so a ``reason = "…pymethods…"``
string or a doc mention never counts. A plain ``impl RustExplorationManager``
(no ``#[pymethods]``) is Rust-only and not Python-callable, so it is skipped.

A macro that emits its own ``#[pymethods]`` block (``callback_setters!``,
``inspect_test_entries!``) is out of reach of this text scan. Neither is
invoked on ``RustExplorationManager``. The second also expands only ``&self``
entries.

The site set is empty and must stay empty, so there is no baseline and no
exemption marker: a block that should not be guarded writes
``#[steady_guard_exempt(reason = "…")]`` on its methods instead.

Usage::

    tools/audit_steady_guard_blocks.py               # report blocks (exit 1 if any)
    tools/audit_steady_guard_blocks.py --self-test   # prove the detector fires
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rust_source_utils import ScanRootError, blank_noise, require_floor, rust_files, scan_roots, self_test_scan_roots

REPO = Path(__file__).resolve().parent.parent
SCAN = {"native/angr": ("",)}
EXCLUDE = {
    "native/angr-bv": "PyO3-free (audit_core_pyo3_free.py): it cannot hold a #[pymethods] block",
    "native/angr-vex": "PyO3-free (audit_core_pyo3_free.py): it cannot hold a #[pymethods] block",
    "native/angr-symex-core": "PyO3-free (audit_core_pyo3_free.py): it cannot hold a #[pymethods] block",
    "native/angr-symex-native": "PyO3-free (audit_core_pyo3_free.py): it cannot hold a #[pymethods] block",
    "native/angr-symex-macros": "proc-macro crate: its impl blocks are quote! fixtures, not the manager",
}
# Fewer files than this means the shell moved out of SCAN.
MIN_FILES = 150
# Fewer manager #[pymethods] blocks than this means the detector stopped matching them.
MIN_BLOCKS = 8

IMPL_RE = re.compile(r"\bimpl\s*(?:<[^>{]*>\s*)?(?:(?:crate|self|super|[A-Za-z_]\w*)::)*RustExplorationManager\s*\{")


def scan_files(repo_root: Path = REPO) -> list[Path]:
    """The Rust files the real run scans; raises ScanRootError on a bad root or a low count."""
    files = rust_files(scan_roots(repo_root, SCAN, EXCLUDE))
    require_floor(files, MIN_FILES, "audit_steady_guard_blocks")
    return files


def _attr_paths(code: str, impl_pos: int) -> list[str]:
    """The paths of the outer attributes directly above ``impl_pos``, top to bottom.

    Walks backward over whitespace and balanced ``#[...]`` groups; stops at
    anything else (the previous item's ``}`` or ``;``, a ``pub``, …).
    """
    paths: list[str] = []
    i = impl_pos
    while True:
        j = i - 1
        while j >= 0 and code[j].isspace():
            j -= 1
        if j < 0 or code[j] != "]":
            break
        depth, k = 0, j
        while k >= 0:
            if code[k] == "]":
                depth += 1
            elif code[k] == "[":
                depth -= 1
                if depth == 0:
                    break
            k -= 1
        if k <= 0 or code[k - 1] != "#":
            break
        body = code[k + 1 : j]
        paths.append(re.match(r"\s*([\w:\s]*)", body).group(1).replace(" ", "").strip())
        i = k - 1
    paths.reverse()
    return paths


def _last_segment(path: str) -> str:
    return path.rsplit("::", 1)[-1]


def scan(src: str) -> tuple[list[tuple[int, str]], int]:
    """``(line, why)`` for each ``#[pymethods]`` manager block lacking the checker above it, and the block count."""
    code = blank_noise(src)
    hits = []
    blocks = 0
    for m in IMPL_RE.finditer(code):
        segs = [_last_segment(p) for p in _attr_paths(code, m.start())]
        if "pymethods" not in segs:
            continue
        blocks += 1
        line = code.count("\n", 0, m.start()) + 1
        if "steady_guard_checked" not in segs:
            hits.append((line, "no #[steady_guard_checked]"))
        elif segs.index("steady_guard_checked") > segs.index("pymethods"):
            hits.append((line, "#[steady_guard_checked] is below #[pymethods]"))
    return hits, blocks


SELF_TEST_SRC = """\
#[angr_symex_macros::steady_guard_checked]
#[allow(unreachable_pub, reason = "pyo3 `#[pymethods]` surface")]
#[pymethods]
impl RustExplorationManager {
    #[steady_guarded]
    fn ok(&mut self) {}
}

#[allow(unreachable_pub, reason = "steady_guard_checked is only named in this string")]
#[pymethods]
impl RustExplorationManager {
    fn unguarded(&mut self) {}
}

// #[steady_guard_checked] in a comment does not count
#[pyo3::pymethods]
impl super::RustExplorationManager {
    fn unguarded(&mut self) {}
}

#[pymethods]
#[steady_guard_checked]
impl RustExplorationManager {
    fn checked_too_late(&mut self) {}
}

/// A Rust-only block: `#[pymethods]` in a doc comment does not make it Python-callable.
impl RustExplorationManager {
    fn helper(&mut self) {}
}

#[pymethods]
impl RustExplorationManagerFactory {
    fn other_type(&mut self) {}
}
"""


def self_test() -> int:
    hits, blocks = scan(SELF_TEST_SRC)
    want = [
        (11, "no #[steady_guard_checked]"),
        (17, "no #[steady_guard_checked]"),
        (23, "#[steady_guard_checked] is below #[pymethods]"),
    ]
    if hits != want:
        print(f"self-test FAILED: flagged {hits}, want {want}", file=sys.stderr)
        return 1
    if blocks != 4:
        print(f"self-test FAILED: counted {blocks} #[pymethods] manager blocks, want 4", file=sys.stderr)
        return 1
    root_failures = self_test_scan_roots(scan_files, SCAN)
    if root_failures:
        print(f"self-test FAILED: {root_failures}", file=sys.stderr)
        return 1
    print(
        f"self-test OK: flagged the unchecked and mis-ordered blocks at lines {[h[0] for h in hits]}, "
        "skipped the checked, Rust-only and other-type blocks; "
        "an unclassified member, a missing root and a low file count fail"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="prove the detector fires on synthetic blocks")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    try:
        files = scan_files()
    except ScanRootError as e:
        print(f"audit_steady_guard_blocks: {e}", file=sys.stderr)
        return 1
    failures = []
    total_blocks = 0
    for path in files:
        hits, blocks = scan(path.read_text(encoding="utf-8"))
        total_blocks += blocks
        failures.extend(f"{path.relative_to(REPO).as_posix()}:{line}: {why}" for line, why in hits)
    if total_blocks < MIN_BLOCKS:
        print(
            f"audit_steady_guard_blocks: found {total_blocks} #[pymethods] impl RustExplorationManager block(s), "
            f"below its floor of {MIN_BLOCKS}; did the detector or the manager's layout change?",
            file=sys.stderr,
        )
        return 1
    if failures:
        print(
            "audit_steady_guard_blocks: #[pymethods] impl RustExplorationManager without the checker:", file=sys.stderr
        )
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        print(
            "write #[angr_symex_macros::steady_guard_checked] above the block's #[pymethods] "
            "(see the macro doc in native/angr-symex-macros/src/lib.rs)",
            file=sys.stderr,
        )
        return 1
    print(f"audit_steady_guard_blocks: OK ({total_blocks} manager blocks in {len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
