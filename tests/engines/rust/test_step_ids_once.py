"""RustExplorationManager._step_ids_once: one SimulationManager step per state id (angr-mx00p.136.3).

From fauxware's main (both engines seeded with the same Python-made state), each
step must give the Python engine's active addresses and stash counts: the puts,
read and strcmp SimProcedure steps, the strcmp fork and the open() call. The
exit steps differ by one step (angr-mx00p.136.30), so they are xfail-strict.
"""

from __future__ import annotations

import pytest

from angr.exploration.rust_manager import RustExplorationManager, resolve_main_address
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, load_fauxware

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust extension not built")

PUTS, READ, STRCMP, OPEN = 0x700000, 0x700010, 0x700020, 0x700028
PUTS_PLT = 0x400510
# The steps from main before the first state exits (main returns into __libc_start_main).
STEPS_BEFORE_EXIT = 33
STEPS_TO_END = 46
BUCKETS = ("active", "deadended", "unconstrained", "errored")


@pytest.fixture(scope="module")
def project():
    return load_fauxware()


@pytest.fixture(scope="module")
def main_state(project):
    main = resolve_main_address(project)
    simgr = project.factory.simulation_manager(project.factory.entry_state())
    for _ in range(40):
        if simgr.active[0].addr == main:
            return simgr.active[0]
        simgr.step()
    raise AssertionError("the Python engine never reached main")


def _python_steps(project, seed, n):
    simgr = project.factory.simulation_manager(seed.copy())
    trace = []
    for _ in range(n):
        simgr.step()
        counts = {b: len(simgr.stashes.get(b, [])) for b in BUCKETS if b != "errored"}
        counts["errored"] = len(simgr.errored)
        trace.append((sorted(s.addr for s in simgr.active), counts))
    return trace


def _rust_steps(project, seed, n):
    mgr = RustExplorationManager(project, [seed.copy()])
    trace, children = [], []
    for _ in range(n):
        step = mgr._step_ids_once(mgr._rust_mgr.get_state_ids("active"))
        counts = mgr.stash_counts()
        trace.append((sorted(s.addr for s in mgr.active), {b: counts.get(b, 0) for b in BUCKETS}))
        children.append(step)
    return mgr, trace, children


def test_steps_match_python_through_procedures_and_fork(project, main_state):
    expected = _python_steps(project, main_state, STEPS_BEFORE_EXIT)
    _, actual, children = _rust_steps(project, main_state, STEPS_BEFORE_EXIT)
    for i, (want, got) in enumerate(zip(expected, actual)):
        assert got == want, f"step {i}: rust {got} != python {want}"
    seen = {addr for addrs, _ in actual for addr in addrs}
    assert {PUTS, READ, STRCMP, OPEN} <= seen
    assert any(len(c.get("active", [])) == 2 for step in children for c in step.values()), "no fork step"


@pytest.mark.xfail(strict=True, reason="angr-mx00p.136.30: exit deadends a step earlier in Rust")
def test_exit_step_count_matches_python(project, main_state):
    expected = _python_steps(project, main_state, STEPS_TO_END)
    _, actual, _ = _rust_steps(project, main_state, STEPS_TO_END)
    assert actual == expected
    assert actual[-1] == ([], {"active": 0, "deadended": 3, "unconstrained": 0, "errored": 0})


def test_children_are_attributed_to_their_source(project, main_state):
    mgr, _, children = _rust_steps(project, main_state, STEPS_TO_END)
    ends = set()
    for step in children:
        for sid, buckets in step.items():
            # A state that does not fork keeps its id; a fork adds one new id.
            assert sid in buckets.get("active", []) or sid in buckets.get("deadended", []), (sid, buckets)
            ends.update(cid for cid in buckets.get("deadended", []))
    assert sorted(ends) == sorted(mgr._rust_mgr.get_state_ids("deadended"))
    assert not mgr._rust_mgr.get_state_ids("active")


def test_restores_settings_and_parked_states(project, main_state):
    mgr = RustExplorationManager(project, [main_state.copy(), main_state.copy()])
    first, second = mgr._rust_mgr.get_state_ids("active")
    drop_terminal = mgr._rust_mgr.stats()["drop_terminal_states"]
    assert mgr.set_block_granular(False) is False

    step = mgr._step_ids_once([first])

    assert list(step) == [first]
    assert sorted(mgr._rust_mgr.get_state_ids("active")) == sorted([first, second])
    assert {s.addr for s in mgr.active} == {main_state.addr, PUTS_PLT}
    assert mgr.set_block_granular(False) is False
    assert mgr._rust_mgr.stats()["drop_terminal_states"] == drop_terminal


def test_unknown_id_raises_and_leaves_active_alone(project, main_state):
    mgr = RustExplorationManager(project, [main_state.copy()])
    ids = mgr._rust_mgr.get_state_ids("active")
    with pytest.raises(ValueError, match="not active"):
        mgr._step_ids_once([max(ids) + 100])
    assert mgr._rust_mgr.get_state_ids("active") == ids
