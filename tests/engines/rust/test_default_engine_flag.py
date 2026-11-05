"""The ``engine="rust"`` project default (angr-op0dn.14.6, angr-mx00p.62).

``engine="rust"`` is the one engine-selection knob. On ``Project(binary,
engine="rust")`` (or for every project, ``ANGR_DEFAULT_ENGINE=rust``) it makes
``factory.simulation_manager()`` calls that name no engine build a
``SimulationManager`` with a ``RustBackend`` (angr-mx00p.136.19), which
decides per state per step whether the Rust engine steps it
(``test_auto_dispatch.py``). On ``simulation_manager(..., engine="rust")`` the
backend is strict: the first state it would decline raises NotImplementedError
at that step.

The invariants this file pins:

* **Flag off = zero diff.** A stock project still builds a plain
  ``SimulationManager`` with no backend.
* **The flip adds a backend, not a SimEngine.** ``engine="rust"`` must
  leave ``default_engine_factory`` on ``UberEngine`` — ``block()``, the CFG, and
  every lifting path keep using it.
* **Naming the engine stays loud.** ``simulation_manager(engine="rust")``
  raises on an opted-in project exactly as on a stock one.
* **Removed knobs fail loudly.** ``use_rust_engine=`` is an unknown keyword and
  ``ANGR_RUST_AUTO`` set is an error, never a silently ignored setting.
"""

from __future__ import annotations

import os

import pytest

from angr import claripy
from angr.engines import UberEngine
from angr.errors import AngrError
from angr.exploration import RustExplorationManager
from angr.exploration.rust_backend import RustBackend
from angr.exploration_techniques import DFS, Director, ExplorationTechnique, UniqueSearch
from angr.sim_manager import SimulationManager
from tests.engines.conftest import BIN_LOCATION, RUST_EXPLORATION_AVAILABLE, load_fauxware

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
        """The flip adds a stepping BACKEND, not the SimEngine block()/CFG/lifting use."""
        assert rust_project.factory.default_engine_factory is UberEngine
        assert rust_project.factory.default_engine.__class__ is UberEngine
        # ...and the lifting path it drives still works.
        assert rust_project.factory.block(rust_project.entry).size > 0

    def test_unset_engine_selects_rust(self, rust_project):
        assert rust_project._rust_default is True
        mgr = rust_project.factory.simulation_manager(rust_project.factory.entry_state())
        assert type(mgr) is SimulationManager
        assert isinstance(mgr.backend, RustBackend)
        assert mgr.backend._strict is False

    def test_no_seed_state_selects_rust(self, rust_project):
        """The entry_state() the factory seeds for itself is routed too."""
        assert isinstance(rust_project.factory.simulation_manager().backend, RustBackend)

    def test_simgr_alias_honors_the_default(self, rust_project):
        assert isinstance(rust_project.factory.simgr().backend, RustBackend)

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
        state = rust_project.factory.entry_state(add_options={"TRACK_MEMORY_ACTIONS"})
        mgr = rust_project.factory.simulation_manager(state, engine="rust")
        assert mgr.backend._strict is True
        with pytest.raises(NotImplementedError, match="TRACK_MEMORY_ACTIONS"):
            mgr.step()

    def test_python_on_a_rust_project_is_a_plain_simulation_manager(self, rust_project):
        """The per-call way back to Python is the stock constructor."""
        mgr = SimulationManager(rust_project, active_states=[rust_project.factory.entry_state()])
        assert type(mgr) is SimulationManager


class _CountingSuccessors(ExplorationTechnique):
    """A successors() override, the hook the Rust manager never dispatches."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def successors(self, simgr, state, **kwargs):
        self.calls += 1
        return simgr.successors(state, **kwargs)


class TestSuccessorsTechniqueHandsOff:
    """A default-routed backend declines a state whose manager has a successors() technique, so the
    technique runs on the Python step (angr-mx00p.132.25)."""

    def test_default_route_hands_off_before_stepping(self, rust_project):
        mgr = rust_project.factory.simulation_manager(rust_project.factory.entry_state())
        dfs = mgr.use_technique(DFS())
        tech = mgr.use_technique(_CountingSuccessors())

        assert type(mgr) is SimulationManager
        assert dfs in mgr._techniques
        assert tech in mgr._techniques
        mgr.step()
        assert tech.calls == 1
        assert mgr.active

    def test_named_engine_raises_at_step(self, rust_project):
        mgr = rust_project.factory.simulation_manager(rust_project.factory.entry_state(), engine="rust")
        tech = mgr.use_technique(_CountingSuccessors())
        with pytest.raises(NotImplementedError, match="_CountingSuccessors"):
            mgr.step()
        assert tech.calls == 0

    def test_technique_added_after_a_step_declines_the_next(self, rust_project):
        mgr = rust_project.factory.simulation_manager(rust_project.factory.entry_state())
        mgr.step()
        tech = mgr.use_technique(_CountingSuccessors())
        stepped = len(mgr.active)
        assert stepped
        mgr.step()
        assert tech.calls == stepped
        assert mgr.backend.stats["declined"]["techniques override successors(): _CountingSuccessors"] == stepped

    def test_director_hands_off(self, rust_project):
        """Director's step() hands simgr.active to CFGEmulated, which rejects a proxy (angr-mx00p.132.27)."""
        mgr = rust_project.factory.simulation_manager(rust_project.factory.entry_state())
        director = mgr.use_technique(Director(num_fallback_states=1))

        assert type(mgr) is SimulationManager
        assert director in mgr._techniques

    def test_unique_search_hands_off(self, rust_project):
        """UniqueSearch moves its pick back with ``s is unique_state``, which a step hook's proxies never match
        (angr-mx00p.132.32)."""
        mgr = rust_project.factory.simulation_manager(rust_project.factory.entry_state())
        unique = mgr.use_technique(UniqueSearch())

        assert type(mgr) is SimulationManager
        assert unique in mgr._techniques
        mgr.step()
        assert len(mgr.active) == 1
        assert unique.uniqueness == {}


class TestStepRunArgsHandOff:
    """``step(num_inst=...)`` stops mid-block, as SimulationManager.step does (angr-mx00p.132.23): a
    default-routed backend declines the state for that step, so the Python engine honors the run argument.

    StringObfuscationFinder steps to just before a call with ``num_inst``; the
    Rust step dropped the kwarg and ran the call, so the callee read its stack
    arguments 8 bytes off.
    """

    def test_default_route_hands_off_and_honors_num_inst(self, rust_project):
        entry = rust_project.entry
        second_insn = rust_project.factory.block(entry).instruction_addrs[1]
        mgr = rust_project.factory.simulation_manager(rust_project.factory.entry_state())

        mgr.step(num_inst=1)

        assert type(mgr) is SimulationManager
        assert [s.addr for s in mgr.active] == [second_insn]

    def test_hand_off_keeps_a_write_made_before_stepping(self, rust_project, fauxware_project):
        """A non-init seed carries over as the caller left it, not as the seed was (angr-mx00p.114)."""
        addr = rust_project.entry + 0x10
        insn_bytes = b"\x48\x83\xc0\x0f"  # add rax, 0xf
        results = []
        for proj in (rust_project, fauxware_project):
            mgr = proj.factory.simulation_manager(proj.factory.blank_state(addr=addr))
            mgr.one_active.regs.rax = 3
            mgr.step(insn_bytes=insn_bytes)
            assert type(mgr) is SimulationManager
            results.append(mgr.one_active.solver.eval(mgr.one_active.regs.rax))
        assert results == [0x12, 0x12]

    def test_hand_off_replays_writes_onto_an_init_seed(self, rust_project, fauxware_project):
        """An init seed hands off as the pristine seed plus the caller's writes, where Python steps it (angr-mx00p.132.64).

        The constructor already stepped the entry seed to main on Python, so the export the caller wrote to is at
        main; the handoff must still step from the entry, carrying the register, memory and constraint writes made
        through the export, and none made to a copy of it.
        """
        x = claripy.BVS("handoff_x", 32)
        results = []
        for proj in (rust_project, fauxware_project):
            mgr = proj.factory.simulation_manager()
            state = mgr.one_active
            state.regs.rax = 3
            state.memory.store(0x500000, x, endness="Iend_LE")
            state.solver.add(x == 0x41)
            clone = state.copy()
            clone.regs.rax = 9
            mgr.step(num_inst=1)
            assert type(mgr) is SimulationManager
            (succ,) = mgr.active
            results.append(
                (
                    succ.addr,
                    succ.solver.eval(succ.regs.rax),
                    succ.solver.eval_upto(succ.memory.load(0x500000, 4, endness="Iend_LE"), 2),
                )
            )
        second_insn = fauxware_project.factory.block(fauxware_project.entry).instruction_addrs[1]
        assert results == [(second_insn, 3, [0x41])] * 2

    def test_stash_active_stays_rust(self, rust_project):
        main = rust_project.loader.find_symbol("main").rebased_addr
        mgr = rust_project.factory.simulation_manager(rust_project.factory.blank_state(addr=main))
        mgr.step(stash="active")
        assert mgr.backend.stats["claimed"] == 1
        assert not mgr.backend.stats["declined"]

    def test_named_engine_raises(self, rust_project):
        mgr = rust_project.factory.simulation_manager(rust_project.factory.entry_state(), engine="rust")
        with pytest.raises(NotImplementedError, match="num_inst"):
            mgr.step(num_inst=1)

    def test_stepped_manager_declines_and_honors_num_inst(self, rust_project, fauxware_project):
        """A manager the backend already stepped declines the num_inst step and lands where Python does."""
        managers = []
        for proj in (rust_project, fauxware_project):
            main = proj.loader.find_symbol("main").rebased_addr
            mgr = proj.factory.simulation_manager(proj.factory.blank_state(addr=main))
            mgr.step()
            managers.append((mgr, len(mgr.active)))
            mgr.step(num_inst=1)
        (rust, rust_stepped), (python, _) = managers
        assert sorted(s.addr for s in rust.active) == sorted(s.addr for s in python.active)
        stats = rust.backend.stats
        assert stats["claimed"] == 1
        assert stats["declined"]["step passes run arguments: num_inst"] == rust_stepped


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
        with pytest.raises(TypeError, match="use_rust_engine"):
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
        assert isinstance(proj.factory.simulation_manager().backend, RustBackend)

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
        """The env gate picks the backend; the SimEngine class stays the caller's."""
        monkeypatch.setenv("ANGR_DEFAULT_ENGINE", "rust")
        proj = load_fauxware(engine=UberEngine)
        assert proj.factory.default_engine_factory is UberEngine
        assert isinstance(proj.factory.simulation_manager().backend, RustBackend)


class TestParityUnderTheFlag:
    """A flag-on run and a stock Python run reach the same answer.

    The routing decision is only safe if the routed-to engine agrees with the
    one it displaced. Two halves, matching the two routes the backend can take:
    an eligible workload (stepped on Rust) and an ineligible one (declined to Python).
    """

    @staticmethod
    def _solve(proj, **state_kwargs):
        state = proj.factory.entry_state(**state_kwargs)
        mgr = proj.factory.simulation_manager(state)
        # No max_steps=: it was a RustExplorationManager.explore() budget; on a SimulationManager it is an
        # engine run argument, which the backend declines and the Python engine ignores.
        mgr.explore(find=lambda s: b"Welcome" in s.posix.dumps(1))
        assert mgr.found, "fauxware backdoor not reached"
        found = mgr.found[0]
        return mgr.backend, found.addr, found.posix.dumps(0)

    def test_eligible_workload_matches_python(self, rust_project, fauxware_project):
        rust_backend, _, rust_stdin = self._solve(rust_project)
        py_backend, _, py_stdin = self._solve(fauxware_project)

        assert isinstance(rust_backend, RustBackend)
        stats = rust_backend.stats
        assert stats["claimed"] + stats["fast_path_runs"] >= 1, "the rust engine stepped nothing"
        assert py_backend is None
        # The solved input is the comparable artifact, not the found address: an
        # output-content predicate trips at whatever block boundary the engine
        # happens to check on, and the two engines batch steps differently.
        # SOSNEAKY is the fauxware backdoor password; the rest of stdin is free.
        assert b"SOSNEAKY" in rust_stdin
        assert b"SOSNEAKY" in py_stdin

    def test_ineligible_workload_is_untouched(self, rust_project, fauxware_project):
        """An action-tracking state is declined to Python under the flag and behaves identically."""
        opts = {"add_options": {"TRACK_MEMORY_ACTIONS"}}
        routed_backend, routed_addr, _ = self._solve(rust_project, **opts)
        py_backend, py_addr, _ = self._solve(fauxware_project, **opts)

        assert isinstance(routed_backend, RustBackend)
        stats = routed_backend.stats
        assert stats["claimed"] == 0
        assert stats["declined"]["state options not honored by the rust engine: TRACK_MEMORY_ACTIONS"] >= 1
        assert py_backend is None
        assert routed_addr == py_addr


class TestDefaultRoutedStepIsOneBlock:
    """A default-routed ``step()`` runs one block and exports its IRSB (angr-mx00p.132.24).

    Upstream ``test_slicing`` reads ``simgr.active[0].scratch.irsb`` after one
    ``step()`` from armel fauxware's ``0x86dc``. Under Rust the step chained into
    the fallthrough's ``bl accepted``, the export left ``scratch.irsb`` None, and
    the history entry named the block's last ``PUT(pc)`` rather than its start.
    """

    def test_step_exports_the_executed_block(self):
        import angr

        path = os.path.join(BIN_LOCATION, "tests", "armel", "fauxware")
        if not os.path.exists(path):
            pytest.skip(f"binary not found: {path}")
        project = angr.Project(path, auto_load_libs=False, engine="rust")
        mgr = project.factory.simulation_manager(project.factory.blank_state(addr=0x86DC))
        assert isinstance(mgr.backend, RustBackend)

        mgr.step()

        assert mgr.backend.stats["claimed"] == 1
        assert sorted(s.addr for s in mgr.active) == [0x86F0, 0x86F8]
        for state in mgr.active:
            assert state.scratch.bbl_addr == 0x86DC
            assert state.scratch.irsb.addr == 0x86DC
            # The executed block, not a re-lift at the successor.
            assert state.scratch.irsb.default_exit_target == 0x86F0


class TestDefaultRoutedSyscallIsItsOwnStep:
    """A default-routed ``step()`` stops at a syscall exit (angr-mx00p.132.12).

    Python runs a block and the syscall it exits into as two steps. Upstream
    ``test_unsupported_syscall_simos`` counts them: under Rust the first
    ``step()`` ran both, so the state errored a step early, and with
    ``BYPASS_UNSUPPORTED_SYSCALL`` the second step ran past the stub.
    """

    @staticmethod
    def _seed():
        import angr

        project = angr.load_shellcode("int 0x80", "x86", engine="rust")
        state = project.factory.entry_state()
        state.regs.eax = 4
        return project, state

    def test_syscall_errors_on_the_second_step(self):
        project, state = self._seed()
        mgr = project.factory.simulation_manager(state)
        assert isinstance(mgr.backend, RustBackend)

        mgr.step()
        assert mgr.backend.stats["claimed"] == 1
        assert [s.addr for s in mgr.active] == [0x2]
        assert mgr.active[0].history.jumpkind == "Ijk_Sys_int128"
        assert not mgr.errored
        mgr.step()
        assert not mgr.active
        assert len(mgr.errored) == 1

    def test_bypassed_syscall_stays_active(self):
        import angr

        project, state = self._seed()
        state.options.add(angr.options.BYPASS_UNSUPPORTED_SYSCALL)
        mgr = project.factory.simulation_manager(state)

        mgr.step()
        mgr.step()
        assert len(mgr.active) == 1
        assert not mgr.errored

    def test_python_stepped_seed_runs_its_pending_syscall(self):
        project, state = self._seed()
        python_mgr = SimulationManager(project, active_states=[state])
        python_mgr.step()
        (stepped,) = python_mgr.active
        assert stepped.history.jumpkind.startswith("Ijk_Sys")

        mgr = project.factory.simulation_manager(stepped)
        assert isinstance(mgr.backend, RustBackend)
        mgr.step()
        assert mgr.backend.stats["claimed"] == 1
        # The syscall, not the block at 0x2.
        assert not mgr.active
        assert len(mgr.errored) == 1


class TestForkSiblingHistory:
    """A deferred-fork sibling's history ends at the branch (angr-mx00p.132.44).

    From armel fauxware's ``0x86dc`` the interpreter continues on the
    fallthrough (``0x86f0``) and defers ``0x86f8``. The sibling was forked from
    the continuing state, so it carried that path's history: chained, the
    whole run through ``bl accepted``; granular, an edge to ``0x86f0``.
    """

    @pytest.mark.parametrize("granular", [False, True])
    def test_sibling_history_ends_at_the_branch(self, granular):
        import angr

        path = os.path.join(BIN_LOCATION, "tests", "armel", "fauxware")
        if not os.path.exists(path):
            pytest.skip(f"binary not found: {path}")
        project = angr.Project(path, auto_load_libs=False, engine="rust")
        # Deferred-fork bookkeeping is the private manager's, so drive one directly.
        mgr = RustExplorationManager(project, active_states=[project.factory.blank_state(addr=0x86DC)])
        mgr.set_block_granular(granular)

        mgr._rust_mgr.run(1)

        rust = mgr._rust_mgr
        ids = rust.get_state_ids("active")
        sibling = [sid for i, sid in enumerate(ids) if rust.get_state_pc("active", i) == 0x86F8]
        assert len(sibling) == 1
        assert rust.get_state_detailed_history(sibling[0]) == [(0x86DC, 0, 0x86F8)]

    @pytest.mark.parametrize("granular", [False, True])
    def test_sibling_recent_bbl_addrs_end_at_the_branch(self, granular):
        """The sibling's ``recent_bbl_addrs`` names the branch block (angr-mx00p.132.45).

        The plain per-step history is the fork base's, ending at the continuing
        path's pc; the bbl tail reads the detailed history instead (angr-v625t).
        """
        import angr

        path = os.path.join(BIN_LOCATION, "tests", "armel", "fauxware")
        if not os.path.exists(path):
            pytest.skip(f"binary not found: {path}")
        project = angr.Project(path, auto_load_libs=False, engine="rust")
        # Deferred-fork bookkeeping is the private manager's, so drive one directly.
        mgr = RustExplorationManager(project, active_states=[project.factory.blank_state(addr=0x86DC)])
        mgr.set_block_granular(granular)

        mgr._rust_mgr.run(1)

        rust = mgr._rust_mgr
        ids = rust.get_state_ids("active")
        sibling = [sid for i, sid in enumerate(ids) if rust.get_state_pc("active", i) == 0x86F8]
        assert len(sibling) == 1
        assert rust.get_state_bbl_history_tail(sibling[0], 1) == [0x86DC]
        assert rust.get_state_bbl_history_tail(sibling[0], 0) == [0x86DC]


class TestExportHistoryChain:
    """Exports share their history ancestors up to the branch (angr-mx00p.132.16).

    An export used to get one history node on top of the seed's, so the found
    and avoid states of x86_64 fauxware met at the seed and
    ``closest_common_ancestor`` named the seed's block, not the branch on
    ``strcmp``'s result at ``0x40068e``.
    """

    def test_found_and_avoid_diverge_at_the_branch(self):
        import angr
        from angr.state_plugins.history import HistoryIter

        path = os.path.join(BIN_LOCATION, "tests", "x86_64", "fauxware")
        if not os.path.exists(path):
            pytest.skip(f"binary not found: {path}")
        project = angr.Project(path, auto_load_libs=False, engine="rust")
        mgr = project.factory.simulation_manager().explore(find=0x4006ED, avoid=[0x4006AA, 0x4006FD])

        found = mgr.found[0]
        other = (mgr.avoid + mgr.active)[0]
        ancestor = found.history.closest_common_ancestor(other.history)
        divergent = next(iter(HistoryIter(found.history, end=ancestor)))
        assert divergent.recent_bbl_addrs == [0x40068E]
        assert divergent.jumpkind == "Ijk_Boring"

    def test_stop_hook_state_records_the_block_into_it(self):
        """A block whose exit lands on a hook (find/avoid stop, SimProcedure) is in the history.

        On i386 fauxware the avoid address ``0x8048564`` is the branch's taken
        exit, so the avoid state's last block, ``0x8048559``, exits straight
        into the stop hook. That edge was dropped, so its history ended at
        ``strcmp`` and the divergence point was ``strcmp``'s node. The PLT
        stub that jumps into a SimProcedure's hook was dropped the same way.
        """
        import angr
        from angr.state_plugins.history import HistoryIter

        path = os.path.join(BIN_LOCATION, "tests", "i386", "fauxware")
        if not os.path.exists(path):
            pytest.skip(f"binary not found: {path}")
        project = angr.Project(path, auto_load_libs=False, engine="rust")
        mgr = project.factory.simulation_manager().explore(find=0x80485C9, avoid=[0x80485DD, 0x8048564])

        found = mgr.found[0]
        avoid = mgr.avoid[0]
        assert avoid.addr == 0x8048564
        assert avoid.history.addr == 0x8048559
        assert avoid.history.jump_target.concrete_value == 0x8048564
        assert 0x80483E0 in list(avoid.history.bbl_addrs)  # strcmp@plt
        assert found.history.addr == 0x80486CA  # the call to accepted()
        ancestor = found.history.closest_common_ancestor(avoid.history)
        divergent = next(iter(HistoryIter(found.history, end=ancestor)))
        assert divergent.recent_bbl_addrs == [0x8048559]
