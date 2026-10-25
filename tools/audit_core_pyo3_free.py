#!/usr/bin/env python3
"""Keep PyO3 out of the engine core, its tests included (angr-mx00p.75).

The core modules (``arch/``, ``memory/``, ``symbolic/``, ``vex/``,
``interpreter/`` and ``state/`` under ``native/angr/src/``) are what the crate
split (angr-mx00p.51) moves into a crate with no ``pyo3`` dependency. Their
code reaches Python only through the PyO3 shell (``EngineHost`` for the
interpreter, ``HostAst`` for the state's overlays), and their tests run
against plain-Rust doubles (``interpreter::TestHost``), so none of them starts
an interpreter.

A test that needs Python passes anyway in a batch run: any earlier test in
the same process has already called ``Python::initialize``. So a missing
initialize, or a new one, shows up only when the module runs alone; this
gate catches it statically instead.

Detection: in every ``.rs`` file under those directories, prod and test
alike, any PyO3 token in code (comments and string literals blanked by
``rust_source_utils.blank_noise``): the ``pyo3`` path, the ``Python`` token
type (``Python::initialize``, ``Python::attach``, ``Python<'_>``), the ``Py``
smart pointer and the ``Py*`` types and attribute macros
(``#[pyclass]``, ``#[pymethods]``, ...). Scanning whole directories rather
than test-named files means a test module of any name is covered.

The only exemptions are the files in ``ALLOWED``, each with the reason the
dependency is upstream's to remove. The set of other sites is empty and must
stay empty, so there is no baseline.

Usage::

    tools/audit_core_pyo3_free.py               # report sites (exit 1 if any)
    tools/audit_core_pyo3_free.py --self-test   # prove the detector fires
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rust_source_utils import ScanRootError, blank_noise, require_floor, rust_files, scan_roots, self_test_scan_roots

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "native" / "angr" / "src"
CORE_DIRS = ("arch", "memory", "symbolic", "vex", "interpreter", "state")
SCAN = {"native/angr": CORE_DIRS}
EXCLUDE = {"native/angr-macros": "proc-macro crate: compile-time code generation, not the engine core"}
# Fewer files than this means the core moved out of CORE_DIRS.
MIN_FILES = 150


def scan_files(repo_root: Path = REPO) -> list[Path]:
    """Every core file the real run scans; raises ScanRootError on a bad root or a low count."""
    files = rust_files(scan_roots(repo_root, SCAN, EXCLUDE))
    require_floor(files, MIN_FILES, "audit_core_pyo3_free")
    return files

# Files that keep PyO3 for an upstream reason, relative to SRC.
ALLOWED = {
    "vex/libvex_lifter.rs": (
        "built on upstream's ailment::vex_ffi (init_symbols(py)), which lives in "
        "the rustylib crate; moving it needs vex_ffi in its own crate (RFC question)"
    ),
}

PYO3_RE = re.compile(
    r"""\bpyo3\b
      | \bPython\b
      | \bPy\s*<
      | \bPy(?:Result|Err|Any|Object|Module|Dict|List|Tuple|Bytes|Type|Ref|RefMut)\b
      | \#\s*\[\s*py(?:class|methods|function|module|o3)\b""",
    re.VERBOSE,
)


def scan(src: str) -> list[int]:
    """1-based line numbers of PyO3 tokens in the code of ``src``."""
    code = blank_noise(src)
    lines = sorted({code.count("\n", 0, m.start()) + 1 for m in PYO3_RE.finditer(code)})
    return lines


SELF_TEST_SRC = """\
use pyo3::prelude::*;
// Python::initialize() in a comment is prose, not code.
#[test]
fn starts_python() {
    Python::initialize();
    let msg = "Python::attach in a string is data";
    Python::attach(|_py| {});
}
fn takes_token(py: Python<'_>) -> PyResult<Py<PyAny>> {
    todo!()
}
#[pyclass]
struct S;
fn fine(host: &TestHost) {
    let python_like = 1; // an identifier that merely contains the word
}
"""


def self_test() -> int:
    hits = scan(SELF_TEST_SRC)
    want = [1, 5, 7, 9, 12]
    if hits != want:
        print(f"self-test FAILED: flagged lines {hits}, want {want}", file=sys.stderr)
        return 1
    root_failures = self_test_scan_roots(scan_files, SCAN)
    if root_failures:
        print(f"self-test FAILED: {root_failures}", file=sys.stderr)
        return 1
    print(
        f"self-test OK: flagged lines {hits} (import, initialize, attach, token types, pyclass); "
        "skipped comment, string and identifier; an unclassified member, a missing root and a low file count fail"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="prove the detector fires on synthetic sites")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    try:
        files = scan_files()
    except ScanRootError as e:
        print(f"audit_core_pyo3_free: {e}", file=sys.stderr)
        return 1
    found = []
    scanned = len(files)
    for path in files:
        rel = path.relative_to(SRC).as_posix()
        if rel in ALLOWED:
            continue
        for line in scan(path.read_text(encoding="utf-8")):
            found.append(f"native/angr/src/{rel}:{line}")
    stale = [rel for rel in ALLOWED if not (SRC / rel).is_file()]
    if stale:
        print(f"audit_core_pyo3_free: ALLOWED names files that no longer exist: {stale}", file=sys.stderr)
        return 1
    if found:
        print("PyO3 in the engine core (angr-mx00p.75):")
        for site in found:
            print(f"  {site}")
        print("Reach Python through the PyO3 shell (EngineHost, HostAst, a *_bridge module),")
        print("and test against interpreter::TestHost instead of starting Python.")
        return 1
    print(f"audit_core_pyo3_free: OK, {scanned} core files scanned, no PyO3 outside {sorted(ALLOWED)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
