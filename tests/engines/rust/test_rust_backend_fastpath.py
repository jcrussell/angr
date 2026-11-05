"""SimulationManager+RustBackend: an explore() or a plain run() runs natively (angr-mx00p.136.14, angr-mx00p.136.15).

With no ``until`` or step arguments and only an Explorer (plus
the default Suggestions) installed, :meth:`SimulationManager.run` hands the
rest of the run, and of its ``n`` bound, to :meth:`RustBackend.try_run`, which
explores in one native run and exports the states it ends with. An ``until``
keeps the per-step backend path; both must find the same states, with the same
stdin, in the same stashes.
"""

from __future__ import annotations

import os

import pytest

import angr
from angr import claripy
from angr.exploration.rust_backend import RustBackend
from tests.engines.conftest import EXAMPLES_DIR, FAUXWARE_PATH, RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust extension not built")


# An ``until`` that never stops the run: it keeps explore() on the per-step path.
def PER_STEP_UNTIL(_simgr):
    return False


AIS3_PATH = os.path.join(EXAMPLES_DIR, "ais3_crackme", "ais3_crackme")
R100_PATH = os.path.join(EXAMPLES_DIR, "defcamp_r100", "r100")

ARGV1 = claripy.BVS("argv1", 100 * 8)
AIS3_FLAG = b"ais3{I_tak3_g00d_n0t3s}"
# r100's main. Its ptrace check runs from an init constructor, and the state where ptrace fails spins
# in a `jmp .` there, inside the Python init prefix, for good: from entry the stash is never all claimed.
R100_MAIN = 0x4007E8

# name -> (path, find, avoid)
CASES = {
    "fauxware": (FAUXWARE_PATH, 0x4006ED, [0x4006AA, 0x4006FD]),
    "defcamp_r100": (R100_PATH, 0x400844, 0x400855),
    "ais3_crackme": (AIS3_PATH, 0x400602, None),
}


def _simgr(name, from_entry=False):
    path = CASES[name][0]
    if not os.path.exists(path):
        pytest.skip(f"{name} binary not found: {path}")
    project = angr.Project(path, auto_load_libs=False)
    if name == "ais3_crackme":
        state = project.factory.entry_state(args=[project.filename, ARGV1])
    elif name == "defcamp_r100" and not from_entry:
        state = project.factory.call_state(R100_MAIN)
    else:
        state = project.factory.entry_state()
    backend = RustBackend(project)
    return angr.SimulationManager(project, [state], backend=backend), backend


def _summary(simgr):
    # stdin only where a find pins it: elsewhere it is under-constrained and dumps() picks any model.
    return {
        name: sorted(
            (s.addr, s.history.depth, s.posix.dumps(0) if name in ("found", "hits") else None, s.posix.dumps(1))
            for s in states
        )
        for name, states in simgr.stashes.items()
        if states
    }


def _explore(name, fast, **kwargs):
    _, find, avoid = CASES[name]
    simgr, backend = _simgr(name)
    simgr.explore(find=find, avoid=avoid, until=None if fast else PER_STEP_UNTIL, **kwargs)
    return simgr, backend


@pytest.mark.parametrize("name", sorted(CASES))
def test_fast_path_leaves_the_stashes_the_per_step_path_leaves(name):
    fast, fast_backend = _explore(name, fast=True)
    per_step, per_step_backend = _explore(name, fast=False)

    assert fast_backend.stats["fast_path_runs"] == 1
    assert per_step_backend.stats["fast_path_runs"] == 0
    assert per_step_backend.stats["claimed"] > fast_backend.stats["claimed"]
    assert len(fast.found) == 1
    assert _summary(fast) == _summary(per_step)
    assert [(type(r.error), r.state.addr) for r in fast.errored] == [
        (type(r.error), r.state.addr) for r in per_step.errored
    ]
    if name == "ais3_crackme":
        for simgr in (fast, per_step):
            assert simgr.found[0].solver.eval(ARGV1, cast_to=bytes).split(b"\x00")[0] == AIS3_FLAG


def test_fast_path_honors_num_find_and_custom_stashes():
    kwargs = {"num_find": 2, "find_stash": "hits", "avoid_stash": "misses"}
    fast, fast_backend = _explore("ais3_crackme", fast=True, **kwargs)
    per_step, _ = _explore("ais3_crackme", fast=False, **kwargs)

    assert fast_backend.stats["fast_path_runs"] == 1
    assert not fast.stashes.get("found")
    assert _summary(fast) == _summary(per_step)


def test_a_second_explore_counts_only_its_own_finds():
    runs = []
    for fast in (True, False):
        simgr, backend = _simgr("ais3_crackme")
        _, find, _ = CASES["ais3_crackme"]
        simgr.explore(find=find, until=None if fast else PER_STEP_UNTIL)
        simgr.explore(find=find, until=None if fast else PER_STEP_UNTIL)
        runs.append((simgr, backend))
    (fast, fast_backend), (per_step, _) = runs

    assert fast_backend.stats["fast_path_runs"] == 2
    assert _summary(fast) == _summary(per_step)


def _stdout_has(text):
    return lambda s: text in s.posix.dumps(1)


# name -> (find, avoid) predicates, as csgames2018 / fauxware scripts write them (angr-mx00p.136.15)
PREDICATE_CASES = {
    "fauxware": (_stdout_has(b"Welcome"), _stdout_has(b"Go away")),
    "ais3_crackme": (_stdout_has(b"Correct"), None),
    "defcamp_r100": (_stdout_has(b"Nice!"), 0x400855),
}


@pytest.mark.parametrize("name", sorted(PREDICATE_CASES))
def test_a_predicate_find_runs_natively_with_the_per_step_stashes(name):
    find, avoid = PREDICATE_CASES[name]
    runs = []
    for fast in (True, False):
        simgr, backend = _simgr(name)
        simgr.explore(find=find, avoid=avoid, until=None if fast else PER_STEP_UNTIL)
        runs.append((simgr, backend))
    (fast, fast_backend), (per_step, per_step_backend) = runs

    assert fast_backend.stats["fast_path_runs"] == 1
    assert per_step_backend.stats["fast_path_runs"] == 0
    assert len(fast.found) == 1
    assert _summary(fast) == _summary(per_step)


# r100 spends 30 steps in the init prefix from entry; the budgets end after it.
@pytest.mark.parametrize("n", [None, 40, 60])
def test_a_state_the_backend_declines_steps_on_python_beside_the_native_run(n):
    # From entry, r100's ptrace-failure state never leaves the init prefix: it steps on Python in each
    # native round, its finds count toward num_find, and it ends in the stash it would on the per-step
    # path (angr-mx00p.136.40.2).
    _, find, avoid = CASES["defcamp_r100"]
    runs = []
    for fast in (True, False):
        simgr, backend = _simgr("defcamp_r100", from_entry=True)
        simgr.explore(find=find, avoid=avoid, n=n, until=None if fast else PER_STEP_UNTIL)
        runs.append((simgr, backend))
    (fast, fast_backend), (per_step, per_step_backend) = runs

    assert fast_backend.stats["fast_path_runs"] == 1
    assert per_step_backend.stats["fast_path_runs"] == 0
    assert fast_backend.stats["declined"]["state is in the python init prefix"] > 0
    assert fast_backend.stats["declined"]["a fast-path run steps its declined states on python"] > 0
    assert "_rust_fast_path_held" not in fast.stashes
    assert _summary(fast) == _summary(per_step)
    if n is None:
        assert len(fast.found) == 1


def test_a_declined_state_keeps_a_dfs_run_on_the_per_step_path():
    simgr, backend = _simgr("defcamp_r100", from_entry=True)
    _, find, avoid = CASES["defcamp_r100"]
    simgr.use_technique(angr.exploration_techniques.DFS())
    simgr.explore(find=find, avoid=avoid)

    assert backend.stats["fast_path_runs"] == 0
    assert backend.stats["claimed"] > 0
    assert len(simgr.found) == 1


# (name, max_length): a limit no path reaches, and limits that cut before and around the find
LIMITER_CASES = [
    ("fauxware", 1000),
    ("fauxware", 30),
    ("fauxware", 20),
    ("ais3_crackme", 1000),
    ("ais3_crackme", 20),
    ("defcamp_r100", 8),
]


@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.parametrize(("name", "max_length"), LIMITER_CASES)
def test_a_length_limiter_runs_natively_with_the_per_step_stashes(name, max_length, drop):
    _, find, avoid = CASES[name]
    runs = []
    for fast in (True, False):
        simgr, backend = _simgr(name)
        simgr.use_technique(angr.exploration_techniques.LengthLimiter(max_length, drop=drop))
        simgr.explore(find=find, avoid=avoid, until=None if fast else PER_STEP_UNTIL)
        runs.append((simgr, backend))
    (fast, fast_backend), (per_step, per_step_backend) = runs

    assert fast_backend.stats["fast_path_runs"] == 1
    assert per_step_backend.stats["fast_path_runs"] == 0
    assert ("cut" in fast.stashes) == ("cut" in per_step.stashes) == (not drop)
    assert _summary(fast) == _summary(per_step)


def test_a_length_limiter_counts_the_blocks_run_before_the_explore():
    runs = []
    for fast in (True, False):
        simgr, backend = _simgr("fauxware")
        simgr.run(until=lambda sm: sm.active[0].history.block_count >= 5)
        simgr.use_technique(angr.exploration_techniques.LengthLimiter(30))
        _, find, avoid = CASES["fauxware"]
        simgr.explore(find=find, avoid=avoid, until=None if fast else PER_STEP_UNTIL)
        runs.append((simgr, backend))
    (fast, fast_backend), (per_step, _) = runs

    assert fast_backend.stats["fast_path_runs"] == 1
    assert _summary(fast) == _summary(per_step)


def _ordered(simgr):
    # DFS's deferred order decides which state it resumes, so the stash order is compared too.
    return {name: [(s.addr, s.history.depth) for s in states] for name, states in simgr.stashes.items() if states}


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_dfs_run_replays_its_deferrals_natively(name):
    # fauxware's DFS arm runs block 0x400699 holding the mid-block avoid 0x4006aa, which the fast path's
    # disarmed rounds must still split the block at (angr-mx00p.132.60).
    _, find, avoid = CASES[name]
    runs = []
    for fast in (True, False):
        simgr, backend = _simgr(name)
        dfs = angr.exploration_techniques.DFS()
        simgr.use_technique(dfs)
        simgr.explore(find=find, avoid=avoid, until=None if fast else PER_STEP_UNTIL)
        runs.append((simgr, backend, dfs._random.getstate()))
    (fast, fast_backend, fast_rng), (per_step, per_step_backend, per_step_rng) = runs

    assert fast_backend.stats["fast_path_runs"] == 1
    assert per_step_backend.stats["fast_path_runs"] == 0
    assert len(fast.found) == 1
    assert _ordered(fast) == _ordered(per_step)
    assert _summary(fast) == _summary(per_step)
    assert fast_rng == per_step_rng


def test_a_dfs_with_states_already_deferred_keeps_the_per_step_path():
    simgr, backend = _simgr("fauxware")
    simgr.use_technique(angr.exploration_techniques.DFS())
    simgr.run(until=lambda sm: len(sm.deferred) > 0)
    # Resume the arm that reaches find, so the other arm stays deferred for the whole explore.
    simgr.move("active", "_held")
    simgr.move("deferred", "active")
    simgr.move("_held", "deferred")
    _, find, avoid = CASES["fauxware"]
    simgr.explore(find=find, avoid=avoid)

    assert backend.stats["fast_path_runs"] == 0
    assert len(simgr.found) == 1


FORK_TREE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "benchmarks", "synthetic_examples", "cow_fork_scaling", "fork_tree_8"
)


def _fork_tree_simgr():
    # cow_fork_scaling's setup: 2**8 leaves, each returning through the blank state's symbolic return address.
    project = angr.Project(FORK_TREE_PATH, auto_load_libs=False)
    state = project.factory.blank_state(addr=project.loader.find_symbol("main").rebased_addr)
    state.posix.stdin.content.append((claripy.BVS("stdin", 16 * 8), claripy.BVV(16, state.arch.bits)))
    backend = RustBackend(project)
    return angr.SimulationManager(project, [state], backend=backend), backend


@pytest.mark.parametrize("name", ["fauxware", "fork_tree"])
def test_a_plain_run_runs_natively_with_the_per_step_stashes(name):
    runs = []
    for fast in (True, False):
        simgr, backend = _fork_tree_simgr() if name == "fork_tree" else _simgr(name)
        simgr.run(until=None if fast else PER_STEP_UNTIL)
        runs.append((simgr, backend))
    (fast, fast_backend), (per_step, per_step_backend) = runs

    assert fast_backend.stats["fast_path_runs"] == 1
    assert per_step_backend.stats["fast_path_runs"] == 0
    assert not fast.active
    assert _summary(fast) == _summary(per_step)
    if name == "fork_tree":
        assert len(fast.unconstrained) == 2**8


# fauxware and ais3 start at entry: their init prefix steps on Python, so the backend takes the run with
# fewer steps left (a budget of 12 or less ends inside the prefix). r100 starts at main.
BOUNDED_CASES = [
    ("fauxware", 20),
    ("fauxware", 40),
    ("fauxware", 60),
    ("ais3_crackme", 20),
    ("ais3_crackme", 30),
    ("defcamp_r100", 1),
    ("defcamp_r100", 12),
    ("defcamp_r100", 20),
    ("defcamp_r100", 40),
]


@pytest.mark.parametrize(("name", "n"), BOUNDED_CASES)
def test_a_bounded_explore_runs_natively_with_the_per_step_stashes(name, n):
    # The budget's last step leaves its arrivals active, as Explorer filters before it steps (angr-mx00p.136.40.1).
    runs = [_explore(name, fast, n=n) for fast in (True, False)]
    (fast, fast_backend), (per_step, per_step_backend) = runs

    assert fast_backend.stats["fast_path_runs"] == 1
    assert per_step_backend.stats["fast_path_runs"] == 0
    assert _summary(fast) == _summary(per_step)


@pytest.mark.parametrize("n", [1, 4, 9])
def test_a_bounded_run_runs_natively_with_the_per_step_stashes(n):
    runs = []
    for fast in (True, False):
        simgr, backend = _fork_tree_simgr()
        simgr.run(n=n, until=None if fast else PER_STEP_UNTIL)
        runs.append((simgr, backend))
    (fast, fast_backend), (per_step, per_step_backend) = runs

    assert fast_backend.stats["fast_path_runs"] == 1
    assert per_step_backend.stats["fast_path_runs"] == 0
    assert _summary(fast) == _summary(per_step)
