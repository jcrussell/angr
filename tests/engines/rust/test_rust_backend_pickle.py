"""Pickling a RustBackend manager or state hands its states back to Python (angr-mx00p.136.10).

An export is a live view of a Rust state, which does not pickle. A pickled
export is a copy with plugins of its own, holding the Rust path constraints
(``SimState.__getstate__``), and a pickled manager drops its backend, so an
unpickled run steps on Python unless it is given a backend again. Pickling
leaves the export itself bound to its Rust state.
"""

from __future__ import annotations

import gc
import os
import pickle

import pytest

import angr
from angr.exploration.rust_backend import RustBackend
from angr.exploration.rust_state_binding import binding_of
from angr.exploration_techniques.spiller import Spiller
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, TEST_BINARIES_DIR

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust extension not built")

FIND = {"i386": 0x080485C9, "x86_64": 0x4006ED, "armel": 0x85F0, "mips": 0x4009FC}
AVOID = {
    "i386": [0x080485DD, 0x08048564],
    "x86_64": [0x4006AA, 0x4006FD],
    "armel": [0x86F8, 0x857C],
    "mips": [0x400A10, 0x400774],
}
SOSNEAKY = b"\x00\x00\x00\x00\x00\x00\x00\x00\x00SOSNEAKY\x00"
# Steps until fauxware's states are past the Python init prefix and stepped in
# Rust, and short of FIND: by 40 steps mips's SOSNEAKY state is past it, on
# either engine.
PAST_INIT_STEPS = 30


def _fauxware_simgr(arch):
    project = angr.Project(os.path.join(TEST_BINARIES_DIR, "..", arch, "fauxware"), auto_load_libs=False)
    backend = RustBackend(project)
    simgr = angr.SimulationManager(project, [project.factory.entry_state()], backend=backend)
    simgr.run(n=PAST_INIT_STEPS, until=lambda sm: False)
    assert backend.stats["claimed"] > 0
    assert simgr.active and all(binding_of(s) is not None for s in simgr.active)
    return simgr


@pytest.mark.parametrize("arch", sorted(FIND))
def test_an_unpickled_manager_finds_fauxware_on_python(arch):
    simgr = _fauxware_simgr(arch)
    pickled = pickle.dumps(simgr, pickle.HIGHEST_PROTOCOL)
    del simgr
    gc.collect()
    simgr = pickle.loads(pickled)
    assert simgr._backend is None
    assert all(binding_of(s) is None for s in simgr.active)
    simgr.explore(find=FIND[arch], avoid=AVOID[arch])
    assert not simgr.errored
    assert simgr.found[0].posix.dumps(0) == SOSNEAKY


@pytest.mark.parametrize("arch", sorted(FIND))
def test_a_factory_built_rust_manager_pickles(arch):
    # The factory route of upstream test_fauxware's test_pickling: a
    # Project(engine="rust") manager is a SimulationManager with a RustBackend,
    # so it pickles (angr-mx00p.132.34).
    project = angr.Project(os.path.join(TEST_BINARIES_DIR, "..", arch, "fauxware"), auto_load_libs=False, engine="rust")
    simgr = project.factory.simulation_manager().run(n=PAST_INIT_STEPS, until=lambda sm: False)
    assert isinstance(simgr.backend, RustBackend)
    assert simgr.backend.stats["claimed"] > 0
    pickled = pickle.dumps(simgr, pickle.HIGHEST_PROTOCOL)
    del project, simgr
    gc.collect()
    simgr = pickle.loads(pickled)
    assert simgr.backend is None
    simgr.explore(find=FIND[arch], avoid=AVOID[arch])
    assert not simgr.errored
    assert simgr.found[0].posix.dumps(0) == SOSNEAKY


def test_a_factory_built_rust_manager_runs_the_spiller():
    # The factory route of upstream test_spiller's test_palindrome2: Spiller's
    # step hook drives a SimulationManager with a RustBackend, which spills and
    # restores exports (angr-mx00p.132.31). fauxware forks once, so the spill
    # goes straight to the vault, with no staging stash.
    project = angr.Project(
        os.path.join(TEST_BINARIES_DIR, "..", "x86_64", "fauxware"), auto_load_libs=False, engine="rust"
    )
    simgr = project.factory.simulation_manager()
    spiller = Spiller(min=1, max=1, staging_stash=None, staging_min=0, staging_max=0)
    simgr.use_technique(spiller)
    simgr.run()
    assert isinstance(simgr.backend, RustBackend)
    assert simgr.backend.stats["claimed"] > 0
    assert spiller._ever_pickled > 0
    assert spiller._ever_unpickled == spiller._ever_pickled
    assert not simgr.errored
    assert SOSNEAKY in [s.posix.dumps(0) for s in simgr.deadended]


def test_an_export_held_out_of_the_stashes_keeps_its_options():
    # Spiller's vault can hand back the live export it was given, after the
    # rest stepped without it. Its Rust state waited in the backend's parked
    # stash, whose options entry the cache cleanup pruned, so the export came
    # back with no options and a concrete solver (angr-l0k4o).
    project = angr.Project(
        os.path.join(TEST_BINARIES_DIR, "..", "x86_64", "fauxware"), auto_load_libs=False, engine="rust"
    )
    simgr = project.factory.simulation_manager()
    while len(simgr.active) < 2:
        simgr.step()
    # As the vault does: the claimed copies are new lineage roots.
    simgr.active[:] = [pickle.loads(pickle.dumps(s, pickle.HIGHEST_PROTOCOL)) for s in simgr.active]
    simgr.step()
    held = simgr.active.pop()
    sid = binding_of(held)._id
    mgr = simgr.backend._manager()
    assert sid in mgr._py_state_options
    while simgr.active:
        simgr.step()
    assert sid in mgr._py_state_options
    simgr.active.append(held)
    simgr.run()
    assert isinstance(simgr.backend, RustBackend)
    assert len(simgr.deadended) == 3
    for state in simgr.deadended:
        assert angr.sim_options.SYMBOLIC in state.options
    assert SOSNEAKY in [s.posix.dumps(0) for s in simgr.deadended]


def test_a_lineage_with_no_metadata_record_keeps_its_options_and_globals():
    # A fork's first access to its options or globals, with no entry on it or
    # any ancestor, used to mint an empty one, and the next export then lost
    # every option (a concrete solver) and every global (angr-kfchf).
    project = angr.Project(
        os.path.join(TEST_BINARIES_DIR, "..", "x86_64", "fauxware"), auto_load_libs=False, engine="rust"
    )
    entry = project.factory.entry_state()
    entry.globals["seeded"] = 1
    simgr = project.factory.simulation_manager(entry)
    while len(simgr.active) < 2:
        simgr.step()
    mgr = simgr.backend._manager()
    for state in simgr.active:
        for sid in mgr._state_ancestry(binding_of(state)._id):
            mgr._py_state_options.pop(sid, None)
            mgr._py_state_globals.pop(sid, None)
    for state in simgr.active:
        sid = binding_of(state)._id
        assert angr.sim_options.SYMBOLIC in mgr.get_state_options_py(sid)
        assert mgr.get_state_globals_py(sid).get("seeded") == 1
        mgr._apply_state_metadata(state, sid)
        assert angr.sim_options.SYMBOLIC in state.options
        assert state.globals.get("seeded") == 1
    simgr.run()
    assert not simgr.errored
    for state in simgr.deadended:
        assert angr.sim_options.SYMBOLIC in state.options
        assert state.globals.get("seeded") == 1
    assert SOSNEAKY in [s.posix.dumps(0) for s in simgr.deadended]


@pytest.mark.parametrize("arch", sorted(FIND))
def test_a_pickled_export_keeps_the_after_main_continuation(arch):
    # Python ran __libc_start_main's self.call(main); Rust frames carry no
    # procedure_data, so a detached call stack must get it back on the frame
    # main returns to, or after_main errors on Python (angr-mx00p.136.32).
    simgr = _fauxware_simgr(arch)
    main = simgr._project.loader.find_symbol("main").rebased_addr
    plain = pickle.loads(pickle.dumps(simgr.active[0], pickle.HIGHEST_PROTOCOL))
    (main_frame,) = [f for f in plain.callstack if f.func_addr == main]
    assert main_frame.next.procedure_data is not None
    assert main_frame.next.procedure_data[4] == main_frame.ret_addr


def test_an_unpickled_manager_given_a_backend_steps_in_rust():
    simgr = pickle.loads(pickle.dumps(_fauxware_simgr("x86_64"), pickle.HIGHEST_PROTOCOL))
    backend = RustBackend(simgr._project)
    simgr._backend = backend
    simgr.explore(find=FIND["x86_64"], avoid=AVOID["x86_64"])
    # An address-only explore of claimable states runs natively (angr-mx00p.136.14).
    assert backend.stats["fast_path_runs"] == 1
    assert simgr.found[0].posix.dumps(0) == SOSNEAKY


def test_pickling_an_export_leaves_it_bound():
    simgr = _fauxware_simgr("x86_64")
    state = simgr.active[0]
    binding = binding_of(state)
    constraints = [str(c) for c in state.solver.constraints]
    plain = pickle.loads(pickle.dumps(state, pickle.HIGHEST_PROTOCOL))
    assert binding_of(state) is binding
    assert binding_of(plain) is None
    assert plain.addr == state.addr
    assert plain.regs.sp.concrete_value == state.regs.sp.concrete_value
    assert [str(c) for c in plain.solver.constraints] == constraints
    # Bytes the path leaves free may take other values, but the stdin read so far is kept.
    assert len(plain.posix.dumps(0)) == len(state.posix.dumps(0)) > 0


def test_an_errored_records_state_pickles():
    # jmp +0; ud2: the first step is stepped in Rust, so the errored source is an export.
    project = angr.load_shellcode(b"\xeb\x00\x0f\x0b", arch="amd64", load_address=0x400000)
    simgr = angr.SimulationManager(project, [project.factory.blank_state(addr=0x400000)], backend=RustBackend(project))
    simgr.run(n=3, until=lambda sm: False)
    (record,) = simgr.errored
    assert binding_of(record.state) is not None
    # The record's traceback does not pickle, on either engine; its state does.
    loaded = pickle.loads(pickle.dumps(record.state, pickle.HIGHEST_PROTOCOL))
    assert binding_of(loaded) is None
    assert loaded.addr == 0x400002


def test_a_deep_copy_keeps_the_backend_and_steps_apart():
    simgr = _fauxware_simgr("x86_64")
    before = sorted(s.addr for s in simgr.active)
    copied = simgr.copy(deep=True)
    assert copied._backend is simgr._backend
    copied.run(n=5, until=lambda sm: False)
    assert sorted(s.addr for s in simgr.active) == before
    simgr.explore(find=FIND["x86_64"], avoid=AVOID["x86_64"])
    assert simgr.found[0].posix.dumps(0) == SOSNEAKY


def test_a_spilled_export_round_trips_through_the_vault():
    # The vault pickler asks for a persistent id of every object, the detached copy's weak
    # self-references included, so the copy must outlive the dump.
    simgr = _fauxware_simgr("x86_64")
    state = simgr.active.pop()
    addr, constraints = state.addr, [str(c) for c in state.solver.constraints]
    spiller = Spiller()
    spiller._pickle([state])
    # The vault hands back a live original, so only a collected one is read from the pickle.
    del state
    gc.collect()
    (loaded,) = spiller._unpickle(1)
    assert binding_of(loaded) is None
    assert loaded.addr == addr
    assert [str(c) for c in loaded.solver.constraints] == constraints


def test_an_exports_block_count_follows_its_history():
    # LengthLimiter cuts on history.block_count, which an export's history node must carry.
    simgr = _fauxware_simgr("x86_64")
    for state in simgr.active:
        assert state.history.block_count == state.history.depth > PAST_INIT_STEPS // 2
