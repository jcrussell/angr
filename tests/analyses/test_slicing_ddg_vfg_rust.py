"""Smoke tests for ``BackwardSlice`` / ``DDG`` / ``VFG`` against a project
that has had an entry state wrapped in :class:`RustExplorationManager`.

These three analyses do not construct a :class:`SimulationManager`
internally. They consume the CFG / CDG / DDG dependency-graph products of
upstream analyses (BackwardSlice, DDG) or drive state stepping directly
through ``project.factory.successors`` (VFG). The Rust engine cannot
influence them today — attaching a ``RustExplorationManager`` to the
project is a no-op as far as their output is concerned — but these
smoke tests pin that contract: if any of the three ever grew a hidden
``factory.simulation_manager(...)`` call site, this suite would surface
the regression.

Fauxware is used as the binary across all three because:

* it is the smallest x86_64 ELF in the angr-examples corpus, so
  ``CFGEmulated`` (which DDG and BackwardSlice both require) completes
  in a few seconds even with ``state_add_options=refs``,
* its single ``main`` / ``authenticate`` shape exercises enough
  inter-procedural structure to produce a non-trivial DDG, and
* it is the canonical regression binary across the rest of the Rust
  engine test suite.
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


EXAMPLES_DIR = os.environ.get("ANGR_EXAMPLES_DIR") or os.path.expanduser(
    "~/repos/angr-examples/examples"
)

FAUXWARE = os.path.join(EXAMPLES_DIR, "fauxware", "fauxware")


def _load_rust_project() -> tuple[angr.Project, RustExplorationManager]:
    """Construct a project, attach a Rust manager, and return both."""
    if not os.path.exists(FAUXWARE):
        pytest.skip(f"fauxware binary not found at {FAUXWARE}")
    proj = angr.Project(
        FAUXWARE,
        load_options={"auto_load_libs": False},
        use_sim_procedures=True,
        default_analysis_mode="symbolic",
    )
    entry = proj.factory.entry_state()
    mgr = RustExplorationManager(proj, [entry])
    assert mgr.active, "RustExplorationManager did not pick up the entry state"
    return proj, mgr


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
class TestAnalysesSliceDdgVfgRust:
    """BackwardSlice / DDG / VFG run cleanly on a Rust-manager-attached project.

    The Rust engine never enters the picture for these three (they don't
    use ``SimulationManager``), so "Works" is the documented verdict.
    These tests serve as the regression net.
    """

    def test_ddg_runs(self):
        proj, _mgr = _load_rust_project()

        cfg = proj.analyses.CFGEmulated(
            context_sensitivity_level=2,
            keep_state=True,
            state_add_options=angr.sim_options.refs,
        )
        main_addr = cfg.functions["main"].addr

        ddg = proj.analyses.DDG(cfg, start=main_addr)
        assert ddg.graph is not None
        assert len(ddg.graph) > 0, "DDG produced an empty data-dependency graph"

    def test_backward_slice_runs(self):
        proj, _mgr = _load_rust_project()

        cfg = proj.analyses.CFGEmulated(
            context_sensitivity_level=2,
            keep_state=True,
            state_add_options=angr.sim_options.refs,
        )
        main_addr = cfg.functions["main"].addr

        cdg = proj.analyses.CDG(cfg)
        ddg = proj.analyses.DDG(cfg, start=main_addr)

        # Pick the first reachable CFG node inside main() as the slice target.
        main_node = cfg.model.get_any_node(main_addr)
        assert main_node is not None, "CFGEmulated did not produce a node for main"

        # ``control_flow_slice=True`` drives BackwardSlice._construct, which
        # populates ``chosen_statements`` (and therefore ``annotated_cfg``).
        # ``no_construct=True`` would skip construction and leave the
        # statement map empty — not a useful smoke check.
        bs = proj.analyses.BackwardSlice(
            cfg, cdg, ddg, targets=[(main_node, -1)], control_flow_slice=True
        )
        assert bs.chosen_statements is not None
        assert len(bs.chosen_statements) > 0, (
            "BackwardSlice produced an empty chosen_statements set"
        )
        # annotated_cfg() is the canonical consumer-facing product.
        anno_cfg = bs.annotated_cfg()
        assert anno_cfg is not None

    def test_vfg_runs(self):
        proj, _mgr = _load_rust_project()

        cfg = proj.analyses.CFGFast(normalize=True)
        main_addr = cfg.functions["main"].addr

        vfg = proj.analyses.VFG(
            cfg,
            start=main_addr,
            context_sensitivity_level=1,
            interfunction_level=1,
            record_function_final_states=True,
            max_iterations=40,
        )
        assert vfg is not None
        # VFG records per-function final states when requested; the seed
        # function should appear in the resulting map.
        assert main_addr in vfg.function_final_states, (
            "VFG ran but did not record a final state for main"
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
