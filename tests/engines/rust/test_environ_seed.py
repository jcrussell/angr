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

import pytest

from angr.exploration.rust_manager import RustExplorationManager
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
