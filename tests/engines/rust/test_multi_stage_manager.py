"""A manager built from another manager's found state leaves that manager alone (angr-vmi2l.2)."""

from __future__ import annotations

import gc
import os
import weakref

import pytest

import angr
from angr.exploration.rust_manager import RustExplorationManager

BINARIES = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "binaries", "tests")
FAUXWARE = os.path.join(BINARIES, "x86_64", "fauxware")
AUTHENTICATED = 0x4006ED

pytestmark = pytest.mark.skipif(not os.path.exists(FAUXWARE), reason="binaries checkout missing")


def _stage_one():
    proj = angr.Project(FAUXWARE, auto_load_libs=False)
    s1 = RustExplorationManager(proj, [proj.factory.entry_state()])
    s1.explore(find=AUTHENTICATED, num_find=2)
    assert len(s1.found) == 2
    return proj, s1


def _detached_like(s1, found):
    """A found state carrying the transfer attributes a detached export gets."""
    state = found.copy()
    state.scratch.rust_mgr = s1._rust_mgr
    state.scratch.rust_found_state_id = found._lazy_state_id
    return state


@pytest.mark.parametrize("detached", [False, True], ids=["export", "detached"])
def test_stage_two_leaves_source_manager_intact(detached):
    proj, s1 = _stage_one()
    counts = {name: len(states) for name, states in s1.stashes.items()}
    f0, f1 = s1.found[0], s1.found[1]
    seed = _detached_like(s1, f0) if detached else f0
    s2 = RustExplorationManager(proj, [seed], max_active_states=3)

    assert s2._rust_mgr is not s1._rust_mgr
    assert {name: len(states) for name, states in s1.stashes.items()} == counts
    # The sibling stays usable and the stage-2 seed keeps its path constraints.
    assert b"SOSNEAKY" in f0.posix.dumps(0) or b"SOSNEAKY" in f1.posix.dumps(0)
    assert s2.active[0].posix.dumps(0) == f0.posix.dumps(0)


def test_stage_two_kwargs_apply_to_its_own_manager():
    proj, s1 = _stage_one()
    s2 = RustExplorationManager(proj, [_detached_like(s1, s1.found[0])], max_active_states=3)
    assert s2._rust_mgr.get_max_active_states() == 3
    assert s1._rust_mgr.get_max_active_states() != 3


def test_stage_two_does_not_keep_stage_one_alive():
    """Once the caller drops it, the stage-1 manager and its core are freed (angr-f4ldp).

    The stage-2 manager kept them alive two ways: its pristine seed copy kept
    the export's proxies, and its root mirror's solver fallback kept the source
    core. A multi-stage loop then held every earlier stage and ran out of memory.
    """
    proj, s1 = _stage_one()
    expected = s1.found[0].posix.dumps(0)
    s2 = RustExplorationManager(proj, [s1.found[0]])
    # The PyO3 core takes no weak reference: count the live ones instead.
    core_type = type(s1._rust_mgr)
    gc.collect()
    cores = sum(type(o) is core_type for o in gc.get_objects())
    s1_ref = weakref.ref(s1)
    del s1
    gc.collect()

    assert s1_ref() is None
    assert sum(type(o) is core_type for o in gc.get_objects()) == cores - 1
    # The stage-2 seed keeps its path constraints and stdin without stage 1.
    assert s2.active[0].posix.dumps(0) == expected
    (seed,) = s2._initial_seed_states
    assert seed.posix.dumps(0) == expected


def test_stage_two_root_mirror_answers_from_its_own_core():
    """The imported root's solver fallback forks the stage-2 core's state, not the source's (angr-f4ldp)."""
    proj, s1 = _stage_one()
    s2 = RustExplorationManager(proj, [s1.found[0]])
    (sid,) = s2._rust_mgr.get_state_ids("active")
    mirror = s2._state_cache[sid]
    assert mirror.scratch.rust_mgr is s2._rust_mgr
    assert mirror.scratch.rust_found_state_id == sid
    fallback = mirror.solver._rust_fallback_instance
    assert fallback._rust_mgr is s2._rust_mgr
    assert fallback._state_id == sid
