"""The ``engine="rust"`` project default (angr-op0dn.14.6, angr-mx00p.62).

``engine="rust"`` is the one engine-selection knob. On ``Project(binary,
engine="rust")`` (or for every project, ``ANGR_DEFAULT_ENGINE=rust``) it makes
``factory.simulation_manager()`` calls that name no engine route through the
eligibility predicate (``test_auto_dispatch.py``) instead of unconditionally
building a Python ``SimulationManager``. On ``simulation_manager(...,
engine="rust")`` it demands the Rust engine and raises when it cannot honor
the workload.

The invariants this file pins:

* **Flag off = zero diff.** A stock project still builds a plain
  ``SimulationManager``.
* **The flip swaps the manager, not the SimEngine.** ``engine="rust"`` must
  leave ``default_engine_factory`` on ``UberEngine`` — ``block()``, the CFG, and
  every lifting path keep using it.
* **Naming the engine stays loud.** ``simulation_manager(engine="rust")``
  raises on an opted-in project exactly as on a stock one.
* **Removed knobs fail loudly.** ``use_rust_engine=`` is an unknown keyword and
  ``ANGR_RUST_AUTO`` set is an error, never a silently ignored setting.
"""

from __future__ import annotations

import pytest

from angr.engines import UberEngine
from angr.errors import AngrError
from angr.exploration import RustExplorationManager
from angr.sim_manager import SimulationManager
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, load_fauxware

pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)


@pytest.fixture
def rust_project():
    return load_fauxware(engine="rust")


class TestFlagOff:
    """Default: nothing changes."""

    def test_stock_project_defaults_to_python(self, fauxware_project):
        mgr = fauxware_project.factory.simulation_manager(fauxware_project.factory.entry_state())
        assert type(mgr) is SimulationManager
        assert "dispatch_reason" not in vars(mgr)

    def test_stock_project_is_not_rust_default(self, fauxware_project):
        assert fauxware_project._rust_default is False


class TestProjectEngineSentinel:
    def test_default_engine_factory_stays_uberengine(self, rust_project):
        """The flip swaps the MANAGER, not the SimEngine block()/CFG/lifting use."""
        assert rust_project.factory.default_engine_factory is UberEngine
        assert rust_project.factory.default_engine.__class__ is UberEngine
        # ...and the lifting path it drives still works.
        assert rust_project.factory.block(rust_project.entry).size > 0

    def test_unset_engine_selects_rust(self, rust_project):
        assert rust_project._rust_default is True
        mgr = rust_project.factory.simulation_manager(rust_project.factory.entry_state())
        assert isinstance(mgr, RustExplorationManager)

    def test_no_seed_state_selects_rust(self, rust_project):
        """The entry_state() the factory seeds for itself is routed too."""
        assert isinstance(rust_project.factory.simulation_manager(), RustExplorationManager)

    def test_simgr_alias_honors_the_default(self, rust_project):
        assert isinstance(rust_project.factory.simgr(), RustExplorationManager)

    def test_routed_manager_runs(self, rust_project):
        mgr = rust_project.factory.simulation_manager(rust_project.factory.entry_state())
        mgr.step()
        assert mgr.active

    @pytest.mark.parametrize("engine", ["pcode", "python", "Rust"])
    def test_unknown_engine_string_raises(self, engine):
        with pytest.raises(AngrError, match="Unknown engine"):
            load_fauxware(engine=engine)

    @pytest.mark.parametrize("engine", ["pcode", "python", "Rust", UberEngine])
    def test_unknown_per_call_engine_raises(self, fauxware_project, engine):
        with pytest.raises(AngrError, match="Unknown engine"):
            fauxware_project.factory.simulation_manager(engine=engine)

    def test_simengine_class_still_accepted(self):
        proj = load_fauxware(engine=UberEngine)
        assert proj.factory.default_engine_factory is UberEngine
        assert proj._rust_default is False
        assert type(proj.factory.simulation_manager()) is SimulationManager


class TestNamingTheEngineStaysLoud:
    def test_explicit_rust_keeps_its_loud_contract(self, rust_project):
        """The project default must not soften the per-call opt-in's raise."""
        with pytest.raises(NotImplementedError, match="resilience"):
            rust_project.factory.simulation_manager(rust_project.factory.entry_state(), engine="rust", resilience=True)

    def test_python_on_a_rust_project_is_a_plain_simulation_manager(self, rust_project):
        """The per-call way back to Python is the stock constructor."""
        mgr = SimulationManager(rust_project, active_states=[rust_project.factory.entry_state()])
        assert type(mgr) is SimulationManager


class TestRemovedKnobs:
    """The fork-only selection API is gone outright; using it fails loudly (angr-mx00p.62)."""

    @pytest.mark.parametrize("value", [True, False, None])
    def test_use_rust_engine_is_an_unknown_kwarg(self, fauxware_project, value):
        with pytest.raises(TypeError, match="use_rust_engine"):
            fauxware_project.factory.simulation_manager(use_rust_engine=value)

    def test_use_rust_engine_is_unknown_on_a_rust_project(self, rust_project):
        """Unsupported kwarg → ineligible → the Python constructor rejects it."""
        with pytest.raises(TypeError, match="use_rust_engine"):
            rust_project.factory.simulation_manager(use_rust_engine=True)

    def test_use_rust_engine_with_engine_rust_raises(self, fauxware_project):
        with pytest.raises(NotImplementedError, match="use_rust_engine"):
            fauxware_project.factory.simulation_manager(engine="rust", use_rust_engine=True)

    @pytest.mark.parametrize("value", ["1", "0", "true"])
    def test_angr_rust_auto_set_raises_at_project_construction(self, monkeypatch, value):
        monkeypatch.setenv("ANGR_RUST_AUTO", value)
        with pytest.raises(AngrError, match="ANGR_RUST_AUTO"):
            load_fauxware()

    def test_angr_rust_auto_empty_is_unset(self, monkeypatch):
        monkeypatch.setenv("ANGR_RUST_AUTO", "")
        assert load_fauxware()._rust_default is False

    @pytest.mark.parametrize("name", ["set_rust_auto_dispatch", "rust_auto_dispatch_enabled"])
    def test_python_switch_is_gone(self, name):
        import angr.exploration

        assert not hasattr(angr.exploration, name)


class TestEnvGate:
    """``ANGR_DEFAULT_ENGINE=rust`` is the process-wide default of ``Project(engine=...)``.

    Read at project-construction time (not import), so the variable can be set
    by a test/CI job without reloading angr.
    """

    def test_env_rust_defaults_new_projects_to_rust(self, monkeypatch):
        monkeypatch.setenv("ANGR_DEFAULT_ENGINE", "rust")
        proj = load_fauxware()
        assert proj._rust_default is True
        assert isinstance(proj.factory.simulation_manager(), RustExplorationManager)

    def test_env_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("ANGR_DEFAULT_ENGINE", "RuSt")
        assert load_fauxware()._rust_default is True

    @pytest.mark.parametrize("value", ["", "python", "0"])
    def test_other_env_values_leave_the_default_alone(self, monkeypatch, value):
        monkeypatch.setenv("ANGR_DEFAULT_ENGINE", value)
        proj = load_fauxware()
        assert proj._rust_default is False
        assert type(proj.factory.simulation_manager()) is SimulationManager

    def test_env_applies_even_with_an_explicit_simengine_class(self, monkeypatch):
        """The env gate picks the manager; the SimEngine class stays the caller's."""
        monkeypatch.setenv("ANGR_DEFAULT_ENGINE", "rust")
        proj = load_fauxware(engine=UberEngine)
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
