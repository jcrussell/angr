"""The tri-state engine dispatcher (angr-op0dn.14.5.2).

``factory.simulation_manager(use_rust_engine=None)`` means *auto*: run on the
Rust engine when it can execute the workload faithfully, and on the Python
engine when it cannot — never raising, never diverging. That is what converts
the Rust engine's loud surface (raise-options, unimplemented arches, unsupported
kwargs) from "a regression the day the default flips" into "transparently routed
to Python".

Auto mode is off until angr-op0dn.14.6 flips it, so the default path here must
be byte-identical to today's ``use_rust_engine=False``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from angr.exploration import (
    RustExplorationManager,
    rust_auto_dispatch_enabled,
    rust_engine_eligible,
    rust_supports_arch,
    set_rust_auto_dispatch,
)
from angr.exploration.rust_manager import _RAISE_OPTION_NAMES
from angr.sim_manager import SimulationManager
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)


@pytest.fixture
def auto_on():
    """Turn auto-dispatch on for one test and restore the process default."""
    previous = rust_auto_dispatch_enabled()
    set_rust_auto_dispatch(True)
    yield
    set_rust_auto_dispatch(previous)


def _fake_project(arch_name: str = "AMD64"):
    return SimpleNamespace(arch=SimpleNamespace(name=arch_name))


def _fake_state(*options: str):
    # Every mode bundle carries EXTENDED_IROP_SUPPORT, so every real state has
    # it; a state *without* it asks for the narrow IR-op table, which the Rust
    # engine cannot honor (_REQUIRED_OPTION_NAMES — see TestRequiredOptionGate
    # in test_default_bundle_options.py).
    return SimpleNamespace(options={"EXTENDED_IROP_SUPPORT", *options})


class TestArchPredicate:
    """``rust_supports_arch`` mirrors the native ``arch_from_name``."""

    @pytest.mark.parametrize("name", ["X86", "AMD64", "ARM", "ARM64", "MIPS32", "MIPS64"])
    def test_implemented_arches(self, name):
        assert rust_supports_arch(name) is True

    @pytest.mark.parametrize("name", ["PPC32", "PPC64", "S390X"])
    def test_python_only_arches(self, name):
        """Python angr supports these; the Rust interpreter does not."""
        assert rust_supports_arch(name) is False

    @pytest.mark.parametrize("name", ["", None, "RISCV64", "Soot"])
    def test_unknown_arch(self, name):
        assert rust_supports_arch(name) is False


class TestEligibilityPredicate:
    def test_plain_amd64_state_is_eligible(self):
        eligible, reason = rust_engine_eligible(_fake_project(), [_fake_state("SYMBOLIC")], {})
        assert eligible is True
        assert "rust engine" in reason

    @pytest.mark.parametrize("arch", ["PPC32", "PPC64", "S390X"])
    def test_unimplemented_arch_routes_python(self, arch):
        eligible, reason = rust_engine_eligible(_fake_project(arch), [_fake_state()], {})
        assert eligible is False
        assert arch in reason

    @pytest.mark.parametrize("option", sorted(_RAISE_OPTION_NAMES))
    def test_every_raise_option_routes_python(self, option):
        """Each option the manager would refuse makes the workload ineligible."""
        eligible, reason = rust_engine_eligible(_fake_project(), [_fake_state(option)], {})
        assert eligible is False
        assert option in reason

    def test_offending_option_on_a_later_seed_state_counts(self):
        states = [_fake_state("SYMBOLIC"), _fake_state("CALLLESS")]
        eligible, reason = rust_engine_eligible(_fake_project(), states, {})
        assert eligible is False
        assert "CALLLESS" in reason

    def test_unsupported_kwargs_route_python(self):
        eligible, reason = rust_engine_eligible(_fake_project(), [_fake_state()], {"hierarchy": None})
        assert eligible is False
        assert "hierarchy" in reason

    def test_rust_kwargs_stay_eligible(self):
        eligible, _ = rust_engine_eligible(_fake_project(), [_fake_state()], {"save_unconstrained": True})
        assert eligible is True

    def test_unsupported_inspect_breakpoint_routes_python(self, fauxware_project):
        state = fauxware_project.factory.entry_state()
        # One of the five events the Rust engine refuses to register (it raises
        # NotImplementedError on a Rust-owned state) — see _INSPECT_EVENT_SPECS.
        state.inspect.b("engine_process", action=lambda _s: None)
        eligible, reason = rust_engine_eligible(fauxware_project, [state], {})
        assert eligible is False
        assert "engine_process" in reason

    def test_supported_inspect_breakpoint_stays_eligible(self, fauxware_project):
        state = fauxware_project.factory.entry_state()
        state.inspect.b("mem_read", action=lambda _s: None)
        eligible, _ = rust_engine_eligible(fauxware_project, [state], {})
        assert eligible is True


class TestFactoryDispatch:
    def test_auto_is_off_by_default(self, fauxware_project):
        """Zero behavioral diff until angr-op0dn.14.6 flips the flag."""
        assert rust_auto_dispatch_enabled() is False
        state = fauxware_project.factory.entry_state()
        mgr = fauxware_project.factory.simulation_manager(state, use_rust_engine=None)
        assert type(mgr) is SimulationManager
        assert "auto engine dispatch is off" in mgr.dispatch_reason

    def test_auto_selects_rust_when_eligible(self, fauxware_project, auto_on):
        state = fauxware_project.factory.entry_state()
        mgr = fauxware_project.factory.simulation_manager(state, use_rust_engine=None)
        assert isinstance(mgr, RustExplorationManager)

    def test_auto_falls_back_to_python_on_raise_option(self, fauxware_project, auto_on):
        """The whole point: a state the Rust engine refuses must not raise."""
        state = fauxware_project.factory.entry_state(add_options={"TRACK_MEMORY_ACTIONS"})
        mgr = fauxware_project.factory.simulation_manager(state, use_rust_engine=None)
        assert type(mgr) is SimulationManager
        assert "TRACK_MEMORY_ACTIONS" in mgr.dispatch_reason

    def test_auto_falls_back_to_python_on_unsupported_kwarg(self, fauxware_project, auto_on):
        mgr = fauxware_project.factory.simulation_manager(
            fauxware_project.factory.entry_state(), use_rust_engine=None, resilience=True
        )
        assert type(mgr) is SimulationManager
        assert "resilience" in mgr.dispatch_reason

    def test_auto_falls_back_to_python_on_unimplemented_arch(self, fauxware_project, auto_on, monkeypatch):
        monkeypatch.setattr("angr.exploration.rust_manager.rust_supports_arch", lambda _name: False)
        mgr = fauxware_project.factory.simulation_manager(fauxware_project.factory.entry_state(), use_rust_engine=None)
        assert type(mgr) is SimulationManager
        assert "not implemented" in mgr.dispatch_reason

    def test_auto_routed_rust_manager_runs(self, fauxware_project, auto_on):
        """The auto-selected manager is a working one, not just the right type."""
        state = fauxware_project.factory.entry_state()
        mgr = fauxware_project.factory.simulation_manager(state, use_rust_engine=None)
        mgr.step()
        assert mgr.active

    def test_explicit_true_still_raises_on_unsupported_kwarg(self, fauxware_project, auto_on):
        """Explicit opt-in keeps its loud contract even with auto mode on."""
        with pytest.raises(NotImplementedError, match="resilience"):
            fauxware_project.factory.simulation_manager(
                fauxware_project.factory.entry_state(), use_rust_engine=True, resilience=True
            )

    def test_explicit_false_stays_python_with_auto_on(self, fauxware_project, auto_on):
        mgr = fauxware_project.factory.simulation_manager(fauxware_project.factory.entry_state(), use_rust_engine=False)
        assert type(mgr) is SimulationManager
        assert mgr.dispatch_reason == "explicit use_rust_engine=False"

    def test_simgr_alias_honors_auto(self, fauxware_project, auto_on):
        mgr = fauxware_project.factory.simgr(fauxware_project.factory.entry_state(), use_rust_engine=None)
        assert isinstance(mgr, RustExplorationManager)
