"""Contract test: ``use_rust_engine`` on ``factory.simulation_manager()`` is
a per-call selector and is **not** inherited by subsequent calls or by
internal-analysis SM constructions.

This pins the inheritance contract documented in
``docs/advanced-topics/rust_engine.rst`` under
"Analyses compatibility > Inheritance contract". Any future change that
introduces a project-level Rust-engine default (e.g. setting an attribute
on the project or factory) needs to update either this contract or the
table of per-analysis verdicts so the safer default cannot regress
silently.

The test does **not** drive an analysis end-to-end — it inspects the
return type of ``factory.simulation_manager(...)`` (and the
``factory.successors`` engine dispatch) under the same call patterns
that internal analyses use.
"""

from __future__ import annotations

__package__ = __package__ or "tests.engines"  # pylint:disable=redefined-builtin

import os

import pytest

import angr
from angr.sim_manager import SimulationManager

# Availability guard and examples-dir resolution live in conftest (angr-7gdp).
from tests.engines.conftest import (
    EXAMPLES_DIR,
    RUST_EXPLORATION_AVAILABLE,
    RustExplorationManager,
)

FAUXWARE = os.path.join(EXAMPLES_DIR, "fauxware", "fauxware")


@pytest.fixture
def proj():
    if not os.path.exists(FAUXWARE):
        pytest.skip(f"fauxware binary not found at {FAUXWARE}")
    return angr.Project(FAUXWARE, auto_load_libs=False)


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
class TestFactoryRustInheritance:
    """``factory.simulation_manager(use_rust_engine=...)`` chooses per-call;
    omitting the kwarg always lands on the Python ``SimulationManager``.
    """

    def test_default_returns_python_manager(self, proj):
        """No kwarg → Python ``SimulationManager`` (never Rust)."""
        sm = proj.factory.simulation_manager()
        assert isinstance(sm, SimulationManager)
        assert not isinstance(sm, RustExplorationManager)

    def test_use_rust_engine_true_returns_rust_manager(self, proj):
        """Explicit opt-in → ``RustExplorationManager``."""
        sm = proj.factory.simulation_manager(use_rust_engine=True)
        assert isinstance(sm, RustExplorationManager)

    def test_use_rust_engine_false_returns_python_manager(self, proj):
        """Explicit opt-out (kwarg=False) → Python ``SimulationManager``."""
        sm = proj.factory.simulation_manager(use_rust_engine=False)
        assert isinstance(sm, SimulationManager)
        assert not isinstance(sm, RustExplorationManager)

    def test_rust_choice_does_not_persist_across_calls(self, proj):
        """Constructing a Rust manager once must not bias later calls.

        Pins the "no project-level toggle" half of the contract: if any
        future change stored ``use_rust_engine`` on ``proj`` or
        ``proj.factory``, this test would fail when the second call
        unexpectedly returned a Rust manager.
        """
        rust_sm = proj.factory.simulation_manager(use_rust_engine=True)
        assert isinstance(rust_sm, RustExplorationManager)

        default_sm = proj.factory.simulation_manager()
        assert isinstance(default_sm, SimulationManager)
        assert not isinstance(default_sm, RustExplorationManager)

    def test_simgr_alias_honors_per_call_selector(self, proj):
        """The ``factory.simgr(...)`` alias defers to
        ``simulation_manager(...)`` and so must apply the same
        per-call rule. Pins the alias against accidental divergence.
        """
        default_simgr = proj.factory.simgr()
        assert isinstance(default_simgr, SimulationManager)
        assert not isinstance(default_simgr, RustExplorationManager)

        rust_simgr = proj.factory.simgr(use_rust_engine=True)
        assert isinstance(rust_simgr, RustExplorationManager)

    def test_default_engine_is_not_rust_after_rust_manager(self, proj):
        """``factory.successors(state, ...)`` dispatches to
        ``factory.default_engine.process(...)``. Internal analyses
        (``CFGEmulated``, ``Identifier``, ``VFG``, ``Jumptable``
        resolver) step states through this path; the contract is that
        it always uses the Python engine even after a
        ``RustExplorationManager`` has been built against the project.
        """
        from angr.engines.vex.heavy.heavy import HeavyVEXMixin

        _ = proj.factory.simulation_manager(use_rust_engine=True)
        engine = proj.factory.default_engine
        # The default engine is ``UberEngine`` / ``UberEnginePcode``;
        # both inherit from HeavyVEXMixin (the Python VEX interpreter).
        # If a future change rewires this to a Rust engine, that
        # change needs to either update this test or update the
        # contract doc.
        assert isinstance(engine, HeavyVEXMixin), (
            f"factory.default_engine became {type(engine).__name__}; "
            "the inheritance contract requires it to remain a Python "
            "engine. Update docs/advanced-topics/rust_engine.rst "
            "(Inheritance contract section) if this is intentional."
        )

    def test_internal_analysis_call_pattern_gets_python(self, proj):
        """Simulate the exact call signature an internal analysis uses
        (``self.project.factory.simulation_manager(state, resilience=True)``)
        — including the ``resilience`` kwarg the jumptable resolver
        passes — and confirm the result is the Python ``SimulationManager``.

        Pins the contract for the call site shape that matters in
        practice; the per-analysis verdict table in the docs reads
        from this contract.
        """
        state = proj.factory.entry_state()
        sm = proj.factory.simulation_manager(state, resilience=True)
        assert isinstance(sm, SimulationManager)
        assert not isinstance(sm, RustExplorationManager)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
