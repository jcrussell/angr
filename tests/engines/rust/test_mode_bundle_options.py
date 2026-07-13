"""The four non-``symbolic``-bundle divergence-risk SimOptions (angr-op0dn.14.9).

After angr-op0dn.14.7 drained the ``symbolic`` bundle, four silent options
remained, all riding bundles a user reaches only by passing ``mode=``:
``AVOID_MULTIVALUED_READS`` / ``AVOID_MULTIVALUED_WRITES`` (``fastpath``) and
``PRODUCE_ZERODIV_SUCCESSORS`` / ``ZERO_FILL_UNCONSTRAINED_REGISTERS``
(``tracing``). Three needed no work — Rust was already doing what they ask —
and only ``PRODUCE_ZERODIV_SUCCESSORS`` is a real gap, so it is promoted to
raise and the dispatcher routes it to Python.
"""

from __future__ import annotations

import pytest

import angr
from angr.exploration import (
    RustExplorationManager,
    rust_engine_eligible,
    rust_unsupported_options,
)
from angr.exploration.rust_manager import _RAISE_OPTION_NAMES
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)

# Every real state carries EXTENDED_IROP_SUPPORT, and its *absence* is itself a
# gate (the inverse-polarity one, angr-op0dn.14.7) — so a probe option set has
# to include it or `rust_unsupported_options` reports that miss instead.
_BASE = {"EXTENDED_IROP_SUPPORT"}


def _census():
    import sys
    from pathlib import Path

    bench = Path(angr.__file__).parent.parent / "tests" / "benchmarks"
    sys.path.insert(0, str(bench))
    try:
        from parity_census import collect
    finally:
        sys.path.remove(str(bench))
    return collect()


class TestZeroDivSuccessors:
    """``PRODUCE_ZERODIV_SUCCESSORS``: the one real gap of the four."""

    def test_promoted_to_raise(self):
        assert "PRODUCE_ZERODIV_SUCCESSORS" in _RAISE_OPTION_NAMES
        assert rust_unsupported_options(_BASE | {angr.options.PRODUCE_ZERODIV_SUCCESSORS}) == [
            "PRODUCE_ZERODIV_SUCCESSORS"
        ]

    def test_manager_raises(self, fauxware_project):
        state = fauxware_project.factory.entry_state(add_options={angr.options.PRODUCE_ZERODIV_SUCCESSORS})
        with pytest.raises(NotImplementedError, match="PRODUCE_ZERODIV_SUCCESSORS"):
            RustExplorationManager(fauxware_project, [state])

    def test_routes_to_python(self, fauxware_project):
        """The raise must be routable, not fatal, under auto-dispatch."""
        state = fauxware_project.factory.entry_state(add_options={angr.options.PRODUCE_ZERODIV_SUCCESSORS})
        eligible, _ = rust_engine_eligible(fauxware_project, [state], {})
        assert eligible is False

    def test_rides_only_the_tracing_bundle(self):
        """If it ever joins `symbolic`, the raise would fire for every user."""
        riders = {m for m, opts in angr.options.modes.items() if angr.options.PRODUCE_ZERODIV_SUCCESSORS in opts}
        assert riders == {"tracing"}


class TestAvoidMultivalued:
    """``AVOID_MULTIVALUED_READS`` / ``_WRITES``: honored, not ignored (angr-tfic)."""

    @pytest.mark.parametrize("name", ["AVOID_MULTIVALUED_READS", "AVOID_MULTIVALUED_WRITES"])
    def test_not_gated(self, name):
        assert rust_unsupported_options(_BASE | {getattr(angr.options, name)}) == []

    @pytest.mark.parametrize("name", ["AVOID_MULTIVALUED_READS", "AVOID_MULTIVALUED_WRITES"])
    def test_exonerated_in_the_census(self, name):
        census = _census()
        assert name in census["exonerated_options"]
        assert name not in census["silent_live_options"]

    def test_manager_runs_with_both_set(self, fauxware_project):
        state = fauxware_project.factory.entry_state(
            add_options={angr.options.AVOID_MULTIVALUED_READS, angr.options.AVOID_MULTIVALUED_WRITES}
        )
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.run(max_steps=3)


class TestZeroFillRegisters:
    """``ZERO_FILL_UNCONSTRAINED_REGISTERS``: matches Rust's default (angr-rhe2)."""

    def test_not_gated(self):
        assert rust_unsupported_options(_BASE | {angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}) == []

    def test_exonerated_in_the_census(self):
        census = _census()
        assert "ZERO_FILL_UNCONSTRAINED_REGISTERS" in census["exonerated_options"]

    def test_symbol_fill_sibling_still_raises(self):
        """The divergent half of the pair must stay loud."""
        assert "SYMBOL_FILL_UNCONSTRAINED_REGISTERS" in _RAISE_OPTION_NAMES

    def test_manager_runs_with_it_set(self, fauxware_project):
        state = fauxware_project.factory.entry_state(add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS})
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.run(max_steps=3)


class TestFlipGate:
    """Acceptance: no live silent option ships in *any* default mode bundle."""

    def test_no_silent_option_in_any_default_bundle(self):
        census = _census()
        assert census["silent_default_mode_options"] == []
        assert census["silent_entry_state_options"] == []
