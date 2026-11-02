# pylint: disable=missing-class-docstring,no-self-use
"""The guest's initial ``entry_state(env=...)`` must reach Rust's env map (angr-6cp06.12).

angr models the initial environment purely in memory: ``simos/linux.py`` dumps a
``KEY=VALUE`` string table onto the stack and stores the ``envp`` array pointer
as ``state.posix.environ``; the Python ``getenv`` SimProcedure walks that array.
Rust's native ``getenv``/``setenv``/``putenv`` instead read ``RustSimState``'s
byte-keyed environment map, which used to start empty and was only ever written
by those same native procs --- so a guest ``getenv("FLAG_CHECK")`` on an
``entry_state(env={"FLAG_CHECK": "1"})`` state returned NULL under the Rust
engine where Python finds the value.

``RustExplorationManager._seed_environ_to_rust`` now walks the ``envp`` array at
state-add time and pushes the concrete pairs through ``seed_environment``.

Observable: ``seed_environment`` is insert-if-absent and returns the number of
keys it actually inserted, so re-seeding a key the initial bridge already
installed returns 0 while a never-seen key returns 1.
"""

from __future__ import annotations

import os

import pytest

import angr
from angr import claripy
from angr.exploration.rust_manager import RustExplorationManager
from tests.common import bin_location
from tests.engines.conftest import load_pbounce

_ENV = {"FLAG_CHECK": "1", "EMPTY_VAR": "", "PATH": "/usr/bin:/bin"}


@pytest.fixture(scope="module")
def seeded_mgr():
    """A Rust manager over an ``entry_state(env=...)`` state, plus its state id."""
    project = load_pbounce()
    state = project.factory.entry_state(env=_ENV)
    mgr = RustExplorationManager(project, [state])
    sid = mgr._rust_mgr.get_state_ids("active")[0]
    return mgr, sid


class TestEntryStateEnvironSeed:
    @pytest.mark.parametrize("key", sorted(_ENV))
    def test_entry_state_env_var_reaches_rust_map(self, seeded_mgr, key):
        """Each ``env=`` key is already present, so a re-seed inserts nothing.

        Includes ``EMPTY_VAR`` --- an empty value must be a *present* key, not a
        miss, or native ``setenv(..., overwrite=0)`` would clobber it.
        """
        mgr, sid = seeded_mgr
        assert mgr._rust_mgr.seed_environment(sid, [(key.encode(), _ENV[key].encode())]) == 0

    def test_absent_key_still_inserts(self, seeded_mgr):
        """Control: the return value is not trivially 0 for every input."""
        mgr, sid = seeded_mgr
        assert mgr._rust_mgr.seed_environment(sid, [(b"NEVER_SET_BY_HARNESS", b"x")]) == 1

    def test_no_env_seeds_nothing(self):
        """A default ``entry_state()`` has an empty envp array --- nothing to seed."""
        project = load_pbounce()
        mgr = RustExplorationManager(project, [project.factory.entry_state()])
        sid = mgr._rust_mgr.get_state_ids("active")[0]
        assert mgr._rust_mgr.seed_environment(sid, [(b"FLAG_CHECK", b"1")]) == 1


def _symbolic_env_state():
    """``test_getenv`` with ``TEST_ENV1`` symbolic: its envp entry is ``TEST_ENV1=<BVS>``."""
    flag = claripy.Concat(*[claripy.BVS(f"flag_{i}", 8) for i in range(30)])
    env = {"PATH": "/home/angr/", "TEST_ENV1": flag, "JAVA_HOME": "jdk-install-dir"}
    project = angr.Project(os.path.join(bin_location, "tests", "x86_64", "test_getenv"), engine="rust")
    return project, project.factory.entry_state(env=env), flag


class TestSymbolicEnvEntry:
    def test_concrete_bytes_of_a_mixed_entry_reach_rust_memory(self):
        """The stack-page sync keeps ``TEST_ENV1=`` beside the symbolic value (angr-mx00p.132.2).

        The entry is one symbolic_data store, and ``concrete_load`` holds zeros
        across all of it; the targeted scan used to re-add only the symbolic
        bytes, so Rust read the key as NULs.
        """
        project, state, _ = _symbolic_env_state()
        envp = state.solver.eval(state.posix.environ)
        ptr = state.solver.eval(state.memory.load(envp + 8, 8, endness=project.arch.memory_endness))
        simgr = project.factory.simulation_manager(state)
        loaded = simgr.active[0].memory.load(ptr, 10)
        assert simgr.active[0].solver.eval(loaded, cast_to=bytes) == b"TEST_ENV1="

    def test_getenv_of_a_symbolic_entry_returns_it(self):
        """Native getenv cannot hold a symbolic value, so the env family runs on Python.

        Before, native getenv returned NULL for ``TEST_ENV1`` and the Bingo
        path left the flag unconstrained.
        """
        project, state, flag = _symbolic_env_state()
        simgr = project.factory.simulation_manager(state)
        simgr.explore()
        bingo = [s for s in simgr.deadended if b"Bingo" in s.posix.dumps(1)]
        assert len(bingo) == 1
        assert bingo[0].solver.eval(flag, cast_to=bytes).split(b"\x00")[0] == b"this_is_a_string!"


def test_printf_substitutes_a_concrete_env_value():
    """Native printf formats ``%s`` as Python's does; it used to print ``!! Bingo %s`` (angr-vmi2l.109)."""
    env = {"PATH": "/home/angr/", "TEST_ENV1": "this_is_a_string!"}
    project = angr.Project(os.path.join(bin_location, "tests", "x86_64", "test_getenv"), engine="rust")
    simgr = project.factory.simulation_manager(project.factory.entry_state(env=env))
    simgr.explore()
    assert len(simgr.deadended) == 1
    lines = simgr.deadended[0].posix.dumps(1).decode().splitlines()
    assert "!! Bingo this_is_a_string!" in lines
    assert "TEST_ENV2: __NULL__" in lines
