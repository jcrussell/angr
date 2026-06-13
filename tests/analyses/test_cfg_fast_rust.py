"""Smoke tests for ``proj.analyses.CFGFast()`` under a project that has had
an entry state wrapped in :class:`RustExplorationManager`.

CFGFast is a static analysis (lifter + heuristics, no SimState exploration),
so the Rust engine should not change its output. These tests assert that
constructing a Rust manager against the project leaves CFGFast producing a
function set identical to the Python-engine baseline.

Three binaries are exercised, drawn from the angr-examples bench corpus:
fauxware, csgames2018 (KeygenMe), and csaw_wyvern (wyvern).
"""

from __future__ import annotations

__package__ = __package__ or "tests.analyses"  # pylint:disable=redefined-builtin

import os

import pytest

import angr

try:
    from angr.exploration import RustExplorationManager

    RUST_EXPLORATION_AVAILABLE = True
except ImportError:
    RUST_EXPLORATION_AVAILABLE = False


EXAMPLES_DIR = os.environ.get("ANGR_EXAMPLES_DIR") or os.path.expanduser("~/repos/angr-examples/examples")

# (example dir name, binary file name within that dir)
BINARIES = [
    ("fauxware", "fauxware"),
    ("csgames2018", "KeygenMe"),
    ("csaw_wyvern", "wyvern"),
]


def _binary_path(example: str, binary: str) -> str:
    return os.path.join(EXAMPLES_DIR, example, binary)


def _exec_func_addrs(proj: angr.Project, cfg) -> set[int]:
    """Return CFGFast function addresses that live inside an executable section.

    CFGFast also surfaces indirect-reference candidates in `.data`/`.bss` as
    `FunctionManager` entries, and the exact set of those non-executable
    addresses can vary run-to-run (it depends on iteration order over
    reference-discovery worklists). Those addresses are not meaningful CFG
    nodes, so the smoke test restricts comparison to executable ranges where
    the function set is stable.
    """
    exec_ranges = [(s.vaddr, s.vaddr + s.memsize) for s in proj.loader.main_object.sections if s.is_executable]
    return {addr for addr in cfg.kb.functions.keys() if any(lo <= addr < hi for lo, hi in exec_ranges)}


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
@pytest.mark.parametrize("example,binary", BINARIES, ids=[b[0] for b in BINARIES])
class TestCFGFastRust:
    """CFGFast still produces the Python-baseline function set on a project
    that has had an entry state attached to a :class:`RustExplorationManager`.
    """

    def test_cfg_fast_matches_python_baseline(self, example: str, binary: str):
        path = _binary_path(example, binary)
        if not os.path.exists(path):
            pytest.skip(f"binary not found at {path}")

        # Baseline: a fresh project, plain CFGFast.
        baseline_proj = angr.Project(path, auto_load_libs=False)
        baseline_cfg = baseline_proj.analyses.CFGFast()
        baseline_funcs = _exec_func_addrs(baseline_proj, baseline_cfg)
        assert baseline_funcs, "baseline CFGFast recovered no in-text functions"
        assert baseline_proj.entry in baseline_funcs

        # Rust-engine-aware: build a separate project, wrap an entry state in
        # a RustExplorationManager (verifies the manager constructs cleanly
        # against this binary), then run CFGFast on the *same* project.
        rust_proj = angr.Project(path, auto_load_libs=False)
        entry = rust_proj.factory.entry_state()
        mgr = RustExplorationManager(rust_proj, [entry])
        assert mgr.active, "RustExplorationManager did not pick up the entry state"

        rust_cfg = rust_proj.analyses.CFGFast()
        rust_funcs = _exec_func_addrs(rust_proj, rust_cfg)

        assert rust_funcs == baseline_funcs, (
            f"CFGFast in-text function set diverged on {example}: "
            f"baseline-only={sorted(baseline_funcs - rust_funcs)}, "
            f"rust-only={sorted(rust_funcs - baseline_funcs)}"
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
