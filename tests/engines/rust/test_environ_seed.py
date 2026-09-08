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
from angr.exploration.rust_manager import RustExplorationManager

_SYNTH_NAME = "fork_solve_pbounce_W3_S2_M8_B1"
_SYNTH_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "benchmarks",
    "synthetic_examples",
    _SYNTH_NAME,
    _SYNTH_NAME,
)

_ENV = {"FLAG_CHECK": "1", "EMPTY_VAR": "", "PATH": "/usr/bin:/bin"}


@pytest.fixture(scope="module")
def seeded_mgr():
    """A Rust manager over an ``entry_state(env=...)`` state, plus its state id."""
    if not os.path.exists(_SYNTH_PATH):
        pytest.skip(f"synthetic bench binary not found: {_SYNTH_PATH}")
    project = angr.Project(_SYNTH_PATH, auto_load_libs=False)
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

    def test_warm_disk_cache_hit_keeps_environ(self):
        """A warm disk-init-cache hit must not drop the environment.

        ``_deserialize_init_state`` rebuilds a *blank* state from the pickle, so
        every ``state.posix`` pointer --- including the ``envp`` array pointer
        the seed bridge walks --- used to be lost on a warm hit even though the
        string table itself is inside the cached stack page. Clearing the
        in-memory ``_init_cache`` forces the disk path.
        """
        if not os.path.exists(_SYNTH_PATH):
            pytest.skip(f"synthetic bench binary not found: {_SYNTH_PATH}")
        project = angr.Project(_SYNTH_PATH, auto_load_libs=False)
        # Cold run: populates both the in-memory and the disk init cache.
        RustExplorationManager(project, [project.factory.entry_state(env=_ENV)])
        RustExplorationManager._init_cache.clear()
        mgr = RustExplorationManager(project, [project.factory.entry_state(env=_ENV)])
        sid = mgr._rust_mgr.get_state_ids("active")[0]
        assert mgr._rust_mgr.seed_environment(sid, [(b"FLAG_CHECK", b"1")]) == 0

    def test_no_env_seeds_nothing(self):
        """A default ``entry_state()`` has an empty envp array --- nothing to seed."""
        if not os.path.exists(_SYNTH_PATH):
            pytest.skip(f"synthetic bench binary not found: {_SYNTH_PATH}")
        project = angr.Project(_SYNTH_PATH, auto_load_libs=False)
        mgr = RustExplorationManager(project, [project.factory.entry_state()])
        sid = mgr._rust_mgr.get_state_ids("active")[0]
        assert mgr._rust_mgr.seed_environment(sid, [(b"FLAG_CHECK", b"1")]) == 1
