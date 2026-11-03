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


_LOAD, _HOOK, _NAME = 0x400000, 0x500000, 0x602000
# movabs rax, _HOOK; call rax; jmp $
_CALL_HOOK = bytes([0x48, 0xB8]) + _HOOK.to_bytes(8, "little") + bytes([0xFF, 0xD0, 0xEB, 0xFE])


def _getenv_home_rax(engine: str, factory: str):
    """Call a stock ``getenv("HOME")`` hook on a state with no modeled environment."""
    project = angr.load_shellcode(_CALL_HOOK, arch="AMD64", load_address=_LOAD)
    proto = angr.SIM_LIBRARIES["libc.so.6"][0].get_prototype("getenv")
    project.hook(_HOOK, angr.SIM_PROCEDURES["posix"]["getenv"](prototype=proto), replace=True)
    opts = {angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS, angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY}
    if factory == "call_state":
        state = project.factory.call_state(_LOAD, add_options=opts)
    else:
        state = project.factory.blank_state(addr=_LOAD, add_options=opts)
    assert state.posix.environ is None
    state.regs.rsp = 0x7FFFFFFF0800
    state.memory.store(_NAME, b"HOME\0")
    state.regs.rdi = _NAME
    if engine == "rust":
        mgr = RustExplorationManager(project, [state])
        for _ in range(4):
            mgr.run(max_steps=1)
        states = list(mgr.active_proxies())
    else:
        simgr = project.factory.simulation_manager(state)
        for _ in range(3):
            simgr.step()
        states = simgr.active
    assert [s.addr for s in states] == [_LOAD + 12]
    return states[0], states[0].regs.rax


class TestGetenvWithoutModeledEnviron:
    """No ``posix.environ`` means an unknown environment, not an empty one (angr-vmi2l.124).

    Python's getenv returns a fresh ``getenv__<name>`` symbol, keeping both the
    NULL and non-NULL branches; native getenv used to answer a concrete NULL.
    """

    @pytest.mark.parametrize("engine", ["py", "rust"])
    @pytest.mark.parametrize("factory", ["call_state", "blank_state"])
    def test_getenv_keeps_both_branches(self, engine, factory):
        state, rax = _getenv_home_rax(engine, factory)
        assert rax.symbolic
        assert state.solver.satisfiable([rax == 0])
        assert state.solver.satisfiable([rax != 0])
