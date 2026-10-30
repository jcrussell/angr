#!/usr/bin/env python3
"""Reject imports of the ``angr`` package from Rust test code (angr-vmi2l.208).

The embedded interpreter in a ``cargo test`` binary cannot import the
top-level ``angr`` package: CI's ``rust_test`` job installs no angr, cargo's
test cwd is ``native/angr/``, and the Ralph gate sets no ``PYTHONPATH``. A test
that guards on ``py.import("angr.rustylib.claripy")`` and returns early on
failure therefore returns before its first assertion and reports a pass, in
every lane that is supposed to check it. That is how a dozen regression tests
went silently dead.

A test that needs claripy gets it from
``crate::claripy_bridge::claripy_module(py)``, which builds the module
in-process, or wraps a clarirs AST with ``claripy_bridge::claripy_object``.
Neither needs ``angr`` on ``sys.path``.

Detection: in any Rust test file under ``native/angr/src/`` (see
``rust_source_utils.is_test_file``), a non-comment line calling
``import("angr.`` / ``import("angr")``. Any such call is flagged, not just the
early-return shapes: an import that fails in CI either skips silently or
panics, and neither one tests anything. The site set is empty and must stay
empty, so there is no baseline.

A site is exempt when its line or the line above carries
``import-guard-ok: <why>``.

Usage::

    tools/audit_test_import_guard.py               # report sites (exit 1 if any)
    tools/audit_test_import_guard.py --self-test   # prove the detector fires
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rust_source_utils import ScanRootError, is_test_file, require_floor, rust_files, scan_roots, self_test_scan_roots

REPO = Path(__file__).resolve().parent.parent
SCAN = {"native/angr": ("",), "native/angr-bv": ("",)}
EXCLUDE = {"native/angr-macros": "proc-macro crate: its tests never start Python"}
# Fewer test files than this means the tests moved out of the scan roots.
MIN_FILES = 150


def scan_files(repo_root: Path = REPO) -> list[Path]:
    """The Rust test files the real run scans; raises ScanRootError on a bad root or a low count."""
    files = [p for p in rust_files(scan_roots(repo_root, SCAN, EXCLUDE)) if is_test_file(p.relative_to(repo_root))]
    require_floor(files, MIN_FILES, "audit_test_import_guard")
    return files

IMPORT_RE = re.compile(r"""\bimport(?:_bound)?\s*\(\s*(?:py\s*,\s*)?"angr(?:\.|")""")
EXEMPT_RE = re.compile(r"import-guard-ok:\s*\S")


def scan(src: str) -> list[int]:
    """1-based line numbers of unexempted ``angr`` imports in ``src``."""
    lines = src.splitlines()
    hits = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(("//", "///", "//!")):
            continue
        if not IMPORT_RE.search(line):
            continue
        if EXEMPT_RE.search(line) or (i > 0 and EXEMPT_RE.search(lines[i - 1])):
            continue
        hits.append(i + 1)
    return hits


SELF_TEST_SRC = """\
#[test]
fn guarded_match() {
    Python::attach(|py| {
        let claripy = match py.import("angr.rustylib.claripy") {
            Ok(m) => m,
            Err(_) => return,
        };
    });
}
#[test]
fn guarded_if() {
    Python::attach(|py| {
        if py.import("angr.rustylib.claripy").is_err() {
            return;
        }
    });
}
#[test]
fn guarded_let_else() {
    Python::attach(|py| {
        let Ok(claripy) = PyModule::import(py, "angr.rustylib.claripy") else { return };
    });
}
#[test]
fn fine() {
    Python::attach(|py| {
        // the old py.import("angr.rustylib.claripy") guard is gone
        let claripy = crate::claripy_bridge::claripy_module(py).expect("claripy");
        // import-guard-ok: synthetic exemption for the self-test
        let _ = py.import("angr.rustylib.claripy");
    });
}
"""


def self_test() -> int:
    hits = scan(SELF_TEST_SRC)
    want = [4, 13, 21]
    if hits != want:
        print(f"self-test FAILED: flagged lines {hits}, want {want}", file=sys.stderr)
        return 1
    if not is_test_file("native/angr/src/callbacks/callbacks_tests.rs"):
        print("self-test FAILED: is_test_file rejects a *_tests.rs path", file=sys.stderr)
        return 1
    root_failures = self_test_scan_roots(scan_files, SCAN)
    if root_failures:
        print(f"self-test FAILED: {root_failures}", file=sys.stderr)
        return 1
    print(
        f"self-test OK: flagged the three guard shapes at lines {hits}, skipped comment and exemption; "
        "an unclassified member, a missing root and a low file count fail"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="prove the detector fires on synthetic guards")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    try:
        files = scan_files()
    except ScanRootError as e:
        print(f"audit_test_import_guard: {e}", file=sys.stderr)
        return 1
    found = []
    for path in files:
        rel = path.relative_to(REPO)
        for line in scan(path.read_text(encoding="utf-8")):
            found.append(f"{rel}:{line}")
    if found:
        print("Rust test code imports the angr package, which cargo test cannot import (angr-vmi2l.208):")
        for site in found:
            print(f"  {site}")
        print("Use crate::claripy_bridge::claripy_module(py) / claripy_object instead,")
        print("or mark a genuine case with `import-guard-ok: <why>`.")
        return 1
    print(f"audit_test_import_guard: no angr imports in {len(files)} Rust test files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
