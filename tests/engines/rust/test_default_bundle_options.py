"""The three default-bundle divergence-risk SimOptions (angr-op0dn.14.7).

``EXTENDED_IROP_SUPPORT``, ``TRACK_CONSTRAINT_ACTIONS`` and
``TRACK_MEMORY_MAPPING`` all ship in the ``symbolic`` mode bundle that a plain
``factory.entry_state()`` hands out. That makes them the one class of option the
tri-state dispatcher cannot route around — routing on a bundle option routes
*every* user to Python — so each had to be resolved individually:

* ``EXTENDED_IROP_SUPPORT`` — inverse-polarity gate. Set (the default) it is
  honored transparently; *unset* it asks Python for the narrow IR-op table,
  which Rust has no mode for, so the absence raises.
* ``TRACK_CONSTRAINT_ACTIONS`` — attributable at its consumption site: reading
  ``state.history.actions`` on a Rust-owned state warns once.
* ``TRACK_MEMORY_MAPPING`` — vestigial; nothing reads it, so ignoring it cannot
  diverge.
"""

from __future__ import annotations

import warnings

import pytest

import angr
from angr.exploration import (
    RustExplorationManager,
    rust_engine_eligible,
    rust_missing_required_options,
    rust_unsupported_options,
)
from angr.exploration.rust_manager import _REQUIRED_OPTION_NAMES
from angr.sim_manager import SimulationManager
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)


class TestRequiredOptionGate:
    """``_REQUIRED_OPTION_NAMES``: options Rust refuses by their *absence*."""

    def test_extended_irop_support_is_the_required_option(self):
        assert {"EXTENDED_IROP_SUPPORT"} == _REQUIRED_OPTION_NAMES

    def test_none_options_have_no_opinion(self):
        """A bare container, not a state — nothing to judge."""
        assert rust_missing_required_options(None) == []
        assert rust_unsupported_options(None) == []

    def test_empty_option_set_offends(self):
        assert rust_missing_required_options(set()) == ["EXTENDED_IROP_SUPPORT"]
        assert rust_unsupported_options(set()) == ["unset EXTENDED_IROP_SUPPORT"]

    def test_option_present_is_clean(self):
        assert rust_missing_required_options({"EXTENDED_IROP_SUPPORT"}) == []
        assert rust_unsupported_options({"EXTENDED_IROP_SUPPORT"}) == []

    def test_both_polarities_reported_together(self):
        assert rust_unsupported_options({"CALLLESS"}) == [
            "CALLLESS",
            "unset EXTENDED_IROP_SUPPORT",
        ]

    def test_manager_raises_when_the_option_is_removed(self, fauxware_project):
        state = fauxware_project.factory.entry_state(remove_options={angr.options.EXTENDED_IROP_SUPPORT})
        with pytest.raises(NotImplementedError, match="unset EXTENDED_IROP_SUPPORT"):
            RustExplorationManager(fauxware_project, [state])

    def test_entry_state_carries_the_option(self, fauxware_project):
        state = fauxware_project.factory.entry_state()
        assert angr.options.EXTENDED_IROP_SUPPORT in state.options
        assert rust_unsupported_options(state.options) == []

    def test_removed_option_routes_to_python(self, fauxware_project):
        state = fauxware_project.factory.entry_state(remove_options={angr.options.EXTENDED_IROP_SUPPORT})
        eligible, reason = rust_engine_eligible(fauxware_project, [state], {})
        assert eligible is False
        assert "unset EXTENDED_IROP_SUPPORT" in reason


class TestTrackConstraintActions:
    """Attributed at the read site, not at manager construction."""

    def test_default_state_does_not_raise(self, fauxware_project):
        """It ships in the `symbolic` bundle — raising would break every user."""
        state = fauxware_project.factory.entry_state()
        assert angr.options.TRACK_CONSTRAINT_ACTIONS in state.options
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.run(max_steps=3)

    def test_reading_history_actions_warns(self, fauxware_project):
        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.run(max_steps=1)
        materialized = mgr.active[0]

        from angr.exploration.rust_state_export import _RustOwnedSimStateHistory

        _RustOwnedSimStateHistory._WARNED = False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            list(materialized.history.actions)
        assert any("TRACK_CONSTRAINT_ACTIONS" in str(w.message) for w in caught)


class TestTrackMemoryMapping:
    """Vestigial: nothing in angr reads it, so Rust ignoring it cannot diverge."""

    def test_no_python_reader(self):
        """The only in-tree reference *adds* it to an option set."""
        import pathlib

        root = pathlib.Path(angr.__file__).parent
        readers = [
            (path, line)
            for path in root.rglob("*.py")
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if "TRACK_MEMORY_MAPPING" in line
            and "in " in line
            and "options.add" not in line
            and not line.lstrip().startswith("#")
        ]
        assert readers == [], f"TRACK_MEMORY_MAPPING gained a reader; re-open angr-op0dn.14.7: {readers}"

    def test_default_state_does_not_raise(self, fauxware_project):
        state = fauxware_project.factory.entry_state()
        assert angr.options.TRACK_MEMORY_MAPPING in state.options
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.run(max_steps=3)

    def test_auto_dispatch_still_picks_rust(self, fauxware_project):
        """Neither of the two attributable options may push a plain state to Python."""
        state = fauxware_project.factory.entry_state()
        eligible, _ = rust_engine_eligible(fauxware_project, [state], {})
        assert eligible is True


class TestFlipGate:
    """The `symbolic`-bundle silent surface — the acceptance criterion."""

    def test_no_silent_option_in_the_entry_state_bundle(self):
        import sys
        from pathlib import Path

        bench = Path(angr.__file__).parent.parent / "tests" / "benchmarks"
        sys.path.insert(0, str(bench))
        try:
            from parity_census import collect
        finally:
            sys.path.remove(str(bench))

        census = collect()
        assert census["silent_entry_state_options"] == []

    def test_python_engine_still_honors_a_narrow_op_table(self, fauxware_project):
        """The gate must not have changed Python-engine behavior."""
        state = fauxware_project.factory.entry_state(remove_options={angr.options.EXTENDED_IROP_SUPPORT})
        simgr = SimulationManager(fauxware_project, [state])
        assert not isinstance(simgr, RustExplorationManager)
        simgr.run(n=3)
