"""The config-gated ``default_engine="rust"`` flip (angr-op0dn.14.6).

Two opt-in doors onto the auto-dispatch mechanism that angr-op0dn.14.5 built:
``angr.Project(binary, engine="rust")`` and ``ANGR_DEFAULT_ENGINE=rust``. Both
make ``factory.simulation_manager()`` — called with *no* ``use_rust_engine``
argument — route through the eligibility predicate instead of unconditionally
building a Python ``SimulationManager``.

The invariants this file pins:

* **Flag off = zero diff.** A stock project still builds a plain
  ``SimulationManager`` for an unset ``use_rust_engine``.
* **The flip swaps the manager, not the SimEngine.** ``engine="rust"`` must
  leave ``default_engine_factory`` on ``UberEngine`` — ``block()``, the CFG, and
  every lifting path keep using it.
* **Explicit always wins.** ``use_rust_engine=True/False`` overrides the project
  default in both directions.
* **The predicate still governs.** An ineligible workload under the flag routes
  to Python transparently, exactly as ``use_rust_engine=None`` does; the flag
  raises the *default*, it does not weaken the eligibility gate.
"""

from __future__ import annotations

import os

import pytest

import angr
from angr.engines import UberEngine
from angr.errors import AngrError
from angr.exploration import RustExplorationManager, rust_auto_dispatch_enabled
from angr.sim_manager import SimulationManager
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, TEST_BINARIES_DIR

pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)


def _fauxware(**kwargs):
    binary_path = os.path.join(TEST_BINARIES_DIR, "fauxware")
    if not os.path.exists(binary_path):
        pytest.skip("fauxware binary not found")
    return angr.Project(binary_path, auto_load_libs=False, **kwargs)


@pytest.fixture
def rust_project():
    return _fauxware(engine="rust")


class TestFlagOff:
    """Default until GO: nothing changes."""

    def test_stock_project_defaults_to_python(self, fauxware_project):
        mgr = fauxware_project.factory.simulation_manager(fauxware_project.factory.entry_state())
        assert type(mgr) is SimulationManager
        assert mgr.dispatch_reason == "project default engine is python"

    def test_stock_project_is_not_rust_default(self, fauxware_project):
        assert fauxware_project.factory._rust_default is False

    def test_global_auto_switch_is_still_off(self):
        """The flip is per-project opt-in; it must not flip the process-wide gate."""
        _fauxware(engine="rust")
        assert rust_auto_dispatch_enabled() is False


class TestProjectEngineSentinel:
    def test_default_engine_factory_stays_uberengine(self, rust_project):
        """The flip swaps the MANAGER, not the SimEngine block()/CFG/lifting use."""
        assert rust_project.factory.default_engine_factory is UberEngine
        assert rust_project.factory.default_engine.__class__ is UberEngine
        # ...and the lifting path it drives still works.
        assert rust_project.factory.block(rust_project.entry).size > 0

    def test_unset_use_rust_engine_selects_rust(self, rust_project):
        mgr = rust_project.factory.simulation_manager(rust_project.factory.entry_state())
        assert isinstance(mgr, RustExplorationManager)
        assert "rust engine supports this project" in mgr.dispatch_reason

    def test_no_seed_state_selects_rust(self, rust_project):
        """The entry_state() the factory seeds for itself is routed too."""
        assert isinstance(rust_project.factory.simulation_manager(), RustExplorationManager)

    def test_simgr_alias_honors_the_default(self, rust_project):
        assert isinstance(rust_project.factory.simgr(), RustExplorationManager)

    def test_routed_manager_runs(self, rust_project):
        mgr = rust_project.factory.simulation_manager(rust_project.factory.entry_state())
        mgr.step()
        assert mgr.active

    def test_no_global_auto_switch_needed(self, rust_project):
        """Opting the project in IS the switch — ANGR_RUST_AUTO stays irrelevant."""
        assert rust_auto_dispatch_enabled() is False
        assert isinstance(rust_project.factory.simulation_manager(), RustExplorationManager)

    def test_unknown_engine_string_raises(self):
        with pytest.raises(AngrError, match="Unknown engine"):
            _fauxware(engine="pcode")

    def test_simengine_class_still_accepted(self):
        proj = _fauxware(engine=UberEngine)
        assert proj.factory.default_engine_factory is UberEngine
        assert proj.factory._rust_default is False
        assert type(proj.factory.simulation_manager()) is SimulationManager


class TestExplicitOverridesTheDefault:
    def test_explicit_false_stays_python(self, rust_project):
        mgr = rust_project.factory.simulation_manager(rust_project.factory.entry_state(), use_rust_engine=False)
        assert type(mgr) is SimulationManager
        assert mgr.dispatch_reason == "explicit use_rust_engine=False"

    def test_explicit_true_keeps_its_loud_contract(self, rust_project):
        """The flag must not soften the explicit opt-in's raise."""
        with pytest.raises(NotImplementedError, match="resilience"):
            rust_project.factory.simulation_manager(
                rust_project.factory.entry_state(), use_rust_engine=True, resilience=True
            )


class TestPredicateStillGoverns:
    """Under the flag, an ineligible workload is still a Python workload."""

    def test_raise_option_routes_to_python(self, rust_project):
        state = rust_project.factory.entry_state(add_options={"TRACK_MEMORY_ACTIONS"})
        mgr = rust_project.factory.simulation_manager(state)
        assert type(mgr) is SimulationManager
        assert "TRACK_MEMORY_ACTIONS" in mgr.dispatch_reason

    def test_unsupported_kwarg_routes_to_python(self, rust_project):
        mgr = rust_project.factory.simulation_manager(rust_project.factory.entry_state(), resilience=True)
        assert type(mgr) is SimulationManager
        assert "resilience" in mgr.dispatch_reason

    def test_unimplemented_arch_routes_to_python(self, rust_project, monkeypatch):
        monkeypatch.setattr("angr.exploration.rust_manager.rust_supports_arch", lambda _name: False)
        mgr = rust_project.factory.simulation_manager(rust_project.factory.entry_state())
        assert type(mgr) is SimulationManager
        assert "not implemented" in mgr.dispatch_reason


class TestEnvGate:
    """``ANGR_DEFAULT_ENGINE=rust`` is the process-wide equivalent of the sentinel.

    Read at project-construction time (not import), so the variable can be set
    by a test/CI job without reloading angr.
    """

    def test_env_rust_defaults_new_projects_to_rust(self, monkeypatch):
        monkeypatch.setenv("ANGR_DEFAULT_ENGINE", "rust")
        proj = _fauxware()
        assert proj.factory._rust_default is True
        assert isinstance(proj.factory.simulation_manager(), RustExplorationManager)

    def test_env_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("ANGR_DEFAULT_ENGINE", "RuSt")
        assert _fauxware().factory._rust_default is True

    @pytest.mark.parametrize("value", ["", "python", "0"])
    def test_other_env_values_leave_the_default_alone(self, monkeypatch, value):
        monkeypatch.setenv("ANGR_DEFAULT_ENGINE", value)
        proj = _fauxware()
        assert proj.factory._rust_default is False
        assert type(proj.factory.simulation_manager()) is SimulationManager

    def test_env_applies_even_with_an_explicit_simengine_class(self, monkeypatch):
        """The env gate picks the manager; the SimEngine class stays the caller's."""
        monkeypatch.setenv("ANGR_DEFAULT_ENGINE", "rust")
        proj = _fauxware(engine=UberEngine)
        assert proj.factory.default_engine_factory is UberEngine
        assert isinstance(proj.factory.simulation_manager(), RustExplorationManager)


class TestParityUnderTheFlag:
    """A flag-on run and a stock Python run reach the same answer.

    The routing decision is only safe if the routed-to engine agrees with the
    one it displaced. Two halves, matching the two routes the flag can take:
    an eligible workload (goes to Rust) and an ineligible one (stays on Python).
    """

    @staticmethod
    def _solve(proj, **state_kwargs):
        state = proj.factory.entry_state(**state_kwargs)
        mgr = proj.factory.simulation_manager(state)
        mgr.explore(find=lambda s: b"Welcome" in s.posix.dumps(1), max_steps=60)
        assert mgr.found, "fauxware backdoor not reached"
        found = mgr.found[0]
        return type(mgr), found.addr, found.posix.dumps(0)

    def test_eligible_workload_matches_python(self, rust_project, fauxware_project):
        rust_kind, _, rust_stdin = self._solve(rust_project)
        py_kind, _, py_stdin = self._solve(fauxware_project)

        assert issubclass(rust_kind, RustExplorationManager)
        assert py_kind is SimulationManager
        # The solved input is the comparable artifact, not the found address: an
        # output-content predicate trips at whatever block boundary the engine
        # happens to check on, and the two engines batch steps differently.
        # SOSNEAKY is the fauxware backdoor password; the rest of stdin is free.
        assert b"SOSNEAKY" in rust_stdin
        assert b"SOSNEAKY" in py_stdin

    def test_ineligible_workload_is_untouched(self, rust_project, fauxware_project):
        """An action-tracking state routes to Python under the flag and behaves identically."""
        opts = {"add_options": {"TRACK_MEMORY_ACTIONS"}}
        routed_kind, routed_addr, _ = self._solve(rust_project, **opts)
        py_kind, py_addr, _ = self._solve(fauxware_project, **opts)

        assert routed_kind is SimulationManager
        assert py_kind is SimulationManager
        assert routed_addr == py_addr


class TestFactoryPickle:
    """``_rust_default`` survives the factory's own __getstate__/__setstate__."""

    def test_roundtrip_preserves_the_flag(self, rust_project):
        import pickle

        restored = pickle.loads(pickle.dumps(rust_project.factory))
        assert restored._rust_default is True
        assert restored.default_engine_factory is UberEngine
