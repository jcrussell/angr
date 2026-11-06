"""The engine dispatcher behind ``Project(engine="rust")`` (angr-op0dn.14.5.2, angr-mx00p.62).

On a project built with ``engine="rust"``, ``factory.simulation_manager()``
called with no ``engine`` builds a ``SimulationManager`` with a
``RustBackend`` (angr-mx00p.136.19), which steps each state on the Rust
engine when it can execute it faithfully and declines it to the Python engine
when it cannot — never raising, never diverging. That is what converts the
Rust engine's loud surface (raise-options, unimplemented arches) into
"transparently stepped on Python", decided per state per step and counted in
``mgr.backend.stats["declined"]``. ``simulation_manager(engine="rust")`` keeps
the loud contract: its first such decline raises NotImplementedError at step
time.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

import angr
from angr.exploration import (
    rust_engine_eligible,
    rust_supports_arch,
)
from angr.exploration.rust_backend import RustBackend
from angr.exploration.rust_manager import _RAISE_OPTION_NAMES
from angr.sim_manager import SimulationManager
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, load_fauxware

pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)

# The options that actually gate SimAction recording (angr/engines/vex/heavy/actions.py).
# TRACK_ACTION_HISTORY is deliberately not one of them — see TestActionTrackingRoutes.
ACTION_TRACKING_OPTIONS = frozenset(
    {
        "TRACK_MEMORY_ACTIONS",
        "TRACK_REGISTER_ACTIONS",
        "TRACK_TMP_ACTIONS",
        "TRACK_JMP_ACTIONS",
        "TRACK_OP_ACTIONS",
    }
)


@pytest.fixture(scope="module")
def rust_project():
    """A ``Project(engine="rust")`` fauxware project, shared read-only across the module."""
    return load_fauxware(engine="rust")


def _main_state(project, **kwargs):
    """A state at fauxware's ``main``, past the init prefix the backend declines for its own routing."""
    return project.factory.blank_state(addr=project.loader.find_symbol("main").rebased_addr, **kwargs)


def _stepped(caplog, project, state, **kwargs):
    """Build a manager over *state* through *project*'s factory, step it once, and return it with the
    backend's logged decline messages."""
    mgr = project.factory.simulation_manager(state, **kwargs)
    assert type(mgr) is SimulationManager
    assert isinstance(mgr.backend, RustBackend)
    with caplog.at_level(logging.DEBUG, logger="angr.exploration.rust_backend"):
        caplog.clear()
        mgr.step()
    declines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("simgr backend: python")]
    return mgr, declines


def _assert_declined_once(mgr, declines, needle: str) -> None:
    """The one stepped state was declined, for a reason naming *needle*, and stepped on Python."""
    stats = mgr.backend.stats
    assert stats["claimed"] == 0
    [(reason, count)] = stats["declined"].items()
    assert needle in reason
    assert count == 1
    assert declines == [f"simgr backend: python ({reason})"]


def _fake_project(arch_name: str = "AMD64", engine=angr.engines.UberEngine):
    return SimpleNamespace(arch=SimpleNamespace(name=arch_name), factory=SimpleNamespace(default_engine_factory=engine))


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

    def test_unicorn_routes_python(self):
        """angr-mx00p.132.10: a unicorn step runs many concrete blocks, so a
        default-routed manager stepping one block at a time errored later than
        Python (test_0div's run(n=5) + step())."""
        eligible, reason = rust_engine_eligible(_fake_project(), [_fake_state("UNICORN")], {})
        assert eligible is False
        assert "UNICORN" in reason

    @pytest.mark.parametrize("option", ["FAST_MEMORY", "FAST_REGISTERS"])
    def test_fast_memory_routes_python(self, option):
        """angr-mx00p.132.39: the option swaps in a FastMemory plugin the state sync cannot read."""
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

    @pytest.mark.parametrize("event", ["simprocedure", "syscall"])
    def test_bounce_only_inspect_breakpoint_routes_python(self, fauxware_project, event):
        """angr-mx00p.132.20: these BPs register on Rust but miss every natively
        served call, so a seed overriding a syscall result found nothing."""
        state = fauxware_project.factory.entry_state()
        state.inspect.b(event, action=lambda _s: None)
        eligible, reason = rust_engine_eligible(fauxware_project, [state], {})
        assert eligible is False
        assert event in reason

    def test_supported_inspect_breakpoint_stays_eligible(self, fauxware_project):
        state = fauxware_project.factory.entry_state()
        state.inspect.b("mem_read", when=angr.BP_AFTER, action=None)
        eligible, _ = rust_engine_eligible(fauxware_project, [state], {})
        assert eligible is True

    @pytest.mark.parametrize(
        ("event", "when"),
        [("mem_read", angr.BP_BEFORE), ("instruction", angr.BP_AFTER), ("reg_write", angr.BP_BOTH)],
    )
    def test_breakpoint_on_an_unfired_phase_routes_to_python(self, fauxware_project, event, when):
        """Rust fires mem_read only after and instruction only before, so a
        seed BP on the other phase (the default ``BP_BEFORE`` on mem_read) would
        never fire there (angr-vmi2l.203)."""
        state = fauxware_project.factory.entry_state()
        state.inspect.b(event, when=when, action=None)
        eligible, reason = rust_engine_eligible(fauxware_project, [state], {})
        assert eligible is False
        assert event in reason

    @pytest.mark.parametrize("action", [lambda _s: None, angr.BP_IPDB])
    def test_inspect_action_stays_eligible(self, fauxware_project, action):
        """A BP action reads the mid-step view, so it no longer routes a seed
        to Python (angr-mx00p.132.53.3)."""
        state = fauxware_project.factory.entry_state()
        state.inspect.b("mem_read", when=angr.BP_AFTER, action=None)
        state.inspect.b("instruction", action=action)
        eligible, _ = rust_engine_eligible(fauxware_project, [state], {})
        assert eligible is True

    def test_smc_analysis_finds_smc_on_a_rust_project(self):
        project = angr.load_shellcode(
            "inc dword ptr [here+1]\nhere:\nmov rax, 0xdeadbeef\nret", "amd64", selfmodifying_code=True, engine="rust"
        )
        assert project.analyses.SMC(project.entry).result


class TestFactoryDispatch:
    def test_stock_project_never_consults_the_predicate(self, fauxware_project, caplog):
        """Zero behavioral diff for a project that did not opt in."""
        with caplog.at_level(logging.DEBUG, logger="angr.factory"):
            mgr = fauxware_project.factory.simulation_manager(fauxware_project.factory.entry_state())
        assert type(mgr) is SimulationManager
        assert not [r for r in caplog.records if r.name == "angr.factory"]

    def test_selects_rust_when_eligible(self, rust_project, caplog):
        mgr, declines = _stepped(caplog, rust_project, _main_state(rust_project))
        stats = mgr.backend.stats
        assert stats["claimed"] == 1
        assert not stats["declined"]
        assert not declines
        assert mgr.active

    def test_falls_back_to_python_on_raise_option(self, rust_project, caplog):
        """The whole point: a state the Rust engine refuses must not raise."""
        state = _main_state(rust_project, add_options={"TRACK_MEMORY_ACTIONS"})
        mgr, declines = _stepped(caplog, rust_project, state)
        _assert_declined_once(mgr, declines, "TRACK_MEMORY_ACTIONS")
        assert mgr.active

    def test_manager_kwarg_stays_on_rust(self, rust_project):
        """A SimulationManager-only kwarg no longer routes anything: the manager honors it itself."""
        hierarchy = angr.state_hierarchy.StateHierarchy()
        mgr = rust_project.factory.simulation_manager(_main_state(rust_project), hierarchy=hierarchy)
        assert isinstance(mgr.backend, RustBackend)
        assert mgr._hierarchy is hierarchy
        mgr.step()
        assert mgr.backend.stats["claimed"] == 1

    def test_falls_back_to_python_on_unimplemented_arch(self, rust_project, monkeypatch, caplog):
        monkeypatch.setattr("angr.exploration.rust_backend.rust_supports_arch", lambda _name: False)
        mgr, declines = _stepped(caplog, rust_project, _main_state(rust_project))
        _assert_declined_once(mgr, declines, "not implemented")
        assert mgr.active

    def test_routed_rust_manager_runs(self, rust_project):
        """The auto-selected manager is a working one, not just the right type."""
        mgr = rust_project.factory.simulation_manager(rust_project.factory.entry_state())
        mgr.step()
        assert mgr.active

    def test_symbolizer_on_a_routed_manager_hands_off_to_python(self, rust_project):
        """SimSymbolizer needs angr's paged memory, and the caller adds it after routing (angr-mx00p.132.37)."""
        mgr = rust_project.factory.simulation_manager(rust_project.factory.entry_state())
        ref = mgr.one_active
        ref.regs.rbx = 0x1234
        ref.symbolizer.set_symbolization_for_all_pages()
        assert type(mgr) is SimulationManager
        assert ref.solver.eval_one(ref.regs.rbx) == 0x1234
        assert mgr.one_active.symbolizer is ref.symbolizer
        ref.symbolizer.resymbolize()
        count = ref.symbolizer.symbolized_count
        assert count
        mgr.run()
        assert not mgr.errored
        assert mgr.one_deadended.symbolizer.symbolized_count > count

    def test_congruency_check_seeds_both_states_like_python(self, rust_project):
        """CongruencyCheck's left state (UNICORN) routes the pair to Python, and the manager owns both
        (angr-mx00p.132.39: the left state was appended to a copy of a Rust manager's active stash)."""
        ca = rust_project.analyses.CongruencyCheck(throw=True)
        # The option sets of tests/common.py::run_simple_unicorn_congruency_check, which congruent fauxware runs under.
        dropped = {angr.options.LAZY_SOLVES, angr.options.TRACK_MEMORY_MAPPING, angr.options.COMPOSITE_SOLVER}
        ca.set_state_options(
            left_add_options=angr.options.unicorn,
            left_remove_options=dropped,
            right_add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS},
            right_remove_options=dropped,
        )
        assert type(ca.simgr) is SimulationManager
        assert len(ca.simgr.left) == 1
        assert len(ca.simgr.right) == 1
        assert angr.options.UNICORN in ca.simgr.one_left.options
        assert angr.options.UNICORN not in ca.simgr.one_right.options
        ca.run(depth=1)

    def test_explicit_rust_still_raises_on_a_declined_state(self, rust_project):
        """Naming the engine keeps its loud contract even on an opted-in project."""
        state = _main_state(rust_project, add_options={"TRACK_MEMORY_ACTIONS"})
        mgr = rust_project.factory.simulation_manager(state, engine="rust")
        with pytest.raises(NotImplementedError, match="TRACK_MEMORY_ACTIONS"):
            mgr.step()

    def test_simgr_alias_routes_too(self, rust_project):
        mgr = rust_project.factory.simgr(rust_project.factory.entry_state())
        assert type(mgr) is SimulationManager
        assert isinstance(mgr.backend, RustBackend)

    def test_no_dispatch_reason_attribute(self, rust_project):
        """The routing decision is logged, not stored on the manager (angr-mx00p.62)."""
        state = rust_project.factory.entry_state(add_options={"TRACK_MEMORY_ACTIONS"})
        assert "dispatch_reason" not in vars(rust_project.factory.simulation_manager(state))


class TestActionTrackingRoutes:
    """Action-tracking workloads run on Python, transparently (angr-op0dn.14.3).

    The Rust engine deliberately records no ``SimAction``s: they wrap claripy
    objects Python-side, so emitting them natively would violate the
    Rust/Python boundary, and no mode bundle sets a ``TRACK_*_ACTIONS`` option
    by default. The backend's per-state decline — not a native implementation —
    is the answer: every action-tracking option is raise-listed, so a
    default-routed backend declines an action-tracking state and it steps on
    the Python engine, with ``state.history.actions`` populated exactly as it
    is today.
    """

    def test_family_is_raise_listed(self):
        """The five options that actually gate action recording, and only those."""
        raise_listed = {n for n in _RAISE_OPTION_NAMES if n.startswith("TRACK_") and n.endswith("_ACTIONS")}
        assert raise_listed == ACTION_TRACKING_OPTIONS

    @pytest.mark.parametrize("option", sorted(ACTION_TRACKING_OPTIONS))
    def test_routes_each_action_option_to_python(self, rust_project, option, caplog):
        state = _main_state(rust_project, add_options={option})
        mgr, declines = _stepped(caplog, rust_project, state)
        _assert_declined_once(mgr, declines, option)

    @pytest.mark.parametrize("option", sorted(ACTION_TRACKING_OPTIONS))
    def test_explicit_rust_still_raises_on_each_action_option(self, fauxware_project, option):
        """Loudness preserved: naming the engine still refuses the workload, at its first step."""
        state = fauxware_project.factory.entry_state(add_options={option})
        mgr = fauxware_project.factory.simulation_manager(state, engine="rust")
        with pytest.raises(NotImplementedError, match=option):
            mgr.step()

    def test_track_action_history_alone_stays_on_rust(self, rust_project, caplog):
        """TRACK_ACTION_HISTORY does not gate action recording (demoted, angr-fkvt)."""
        state = _main_state(rust_project, add_options={"TRACK_ACTION_HISTORY"})
        mgr, declines = _stepped(caplog, rust_project, state)
        assert mgr.backend.stats["claimed"] == 1
        assert not declines

    def test_routed_manager_records_actions_like_python(self, rust_project, fauxware_project):
        """The parity that matters: the routed run's actions match a stock Python run."""
        options = {"TRACK_MEMORY_ACTIONS", "TRACK_REGISTER_ACTIONS"}

        def _action_kinds(project):
            state = project.factory.entry_state(add_options=options)
            mgr = project.factory.simulation_manager(state)
            mgr.run(n=5)
            stepped = mgr.active + mgr.deadended
            assert stepped
            return [sorted(a.type for a in s.history.actions) for s in stepped]

        routed = _action_kinds(rust_project)
        baseline = _action_kinds(fauxware_project)
        assert routed == baseline
        assert any(kinds for kinds in routed), "action tracking recorded nothing at all"


class _CountingEngine(angr.engines.UberEngine):
    calls = 0

    def process_successors(self, successors, **kwargs):
        type(self).calls += 1
        return super().process_successors(successors, **kwargs)


class TestCustomEngine:
    """angr-vmi2l.281: the Rust engine replaces the whole SimEngine, so a project
    built with a SimEngine subclass must not lose its overrides to it."""

    def test_predicate_declines_a_custom_engine(self):
        eligible, reason = rust_engine_eligible(_fake_project(engine=_CountingEngine), [_fake_state()], {})
        assert (eligible, reason) == (False, "project uses a custom SimEngine _CountingEngine")

    def test_env_default_route_runs_the_custom_engine(self, monkeypatch):
        monkeypatch.setenv("ANGR_DEFAULT_ENGINE", "rust")
        proj = load_fauxware(engine=_CountingEngine)
        simgr = proj.factory.simgr()
        assert isinstance(simgr.backend, RustBackend)
        _CountingEngine.calls = 0
        simgr.run(n=5)
        assert _CountingEngine.calls > 0
        assert simgr.backend.stats["claimed"] == 0
        assert set(simgr.backend.stats["declined"]) == {"project uses a custom SimEngine _CountingEngine"}

    def test_explicit_rust_engine_raises_on_a_custom_engine(self):
        proj = load_fauxware(engine=_CountingEngine)
        simgr = proj.factory.simgr(engine="rust")
        with pytest.raises(NotImplementedError, match="custom SimEngine _CountingEngine"):
            simgr.step()
