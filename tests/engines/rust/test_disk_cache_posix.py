# pylint: disable=missing-class-docstring,no-self-use
"""A warm disk-init-cache hit must restore ``posix.argv/argc/environ/auxv`` (angr-7tmoz).

``rust_disk_cache.py::_deserialize_init_state`` rebuilds a *blank* state and
restores registers + memory pages, so every pointer
``simos/linux.py::state_entry`` parks in the ``posix`` plugin is lost unless it
was explicitly persisted. angr-6cp06.12 persisted ``environ`` (the Rust getenv
seed bridge needs it); ``argv`` / ``argc`` / ``auxv`` stayed dropped, so a
Python SimProcedure that reads ``state.posix.argv`` --- ``__libc_start_main``
being the load-bearing one --- saw ``None`` on a cache-warm run and a real
pointer on a cold one.

Two layers of coverage: a round-trip over the module-level extract/restore
helpers, and an end-to-end check that the pickle a real cold run writes to
``~/.cache/angr_rust_init`` deserializes back to the same four values.
"""

from __future__ import annotations

import os
from unittest import mock

import claripy
import pytest

import angr
from angr.exploration.rust_disk_cache import (
    _POSIX_POINTER_FIELDS,
    _extract_posix_argc,
    _extract_posix_pointer,
    _restore_posix_entry_fields,
)
from angr.exploration.rust_manager import RustExplorationManager

_SYNTH_NAME = "fork_solve_pbounce_W3_S2_M8_B1"
_SYNTH_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "benchmarks",
    "synthetic_examples",
    _SYNTH_NAME,
    _SYNTH_NAME,
)

_ARGS = ["prog", "alpha", "beta"]
_ENV = {"FLAG_CHECK": "1"}

_POSIX_FIELDS = (*_POSIX_POINTER_FIELDS, "argc")
_CACHE_KEY = "test_angr7tmoz_posix"


def _project():
    if not os.path.exists(_SYNTH_PATH):
        pytest.skip(f"synthetic bench binary not found: {_SYNTH_PATH}")
    return angr.Project(_SYNTH_PATH, auto_load_libs=False)


def _snapshot(state) -> dict:
    """The four posix fields as plain ints, for value comparison across states."""
    out = {}
    for field in _POSIX_FIELDS:
        val = getattr(state.posix, field)
        out[field] = None if val is None else state.solver.eval(val)
    return out


class TestPosixEntryFieldRoundTrip:
    """The extract/restore helper pair, without touching the filesystem."""

    @pytest.fixture(scope="class")
    def states(self):
        project = _project()
        entry = project.factory.entry_state(args=_ARGS, env=_ENV)
        data = {f"posix_{f}": _extract_posix_pointer(entry, f) for f in _POSIX_POINTER_FIELDS}
        data["posix_argc"] = _extract_posix_argc(entry)
        blank = project.factory.blank_state(addr=entry.addr)
        _restore_posix_entry_fields(blank, data)
        return entry, blank

    def test_blank_state_lacks_the_fields_to_begin_with(self):
        """Control: this is the hole the restore fills, not a no-op."""
        project = _project()
        blank = project.factory.blank_state(addr=project.entry)
        assert all(getattr(blank.posix, f) is None for f in _POSIX_FIELDS)

    @pytest.mark.parametrize("field", _POSIX_FIELDS)
    def test_field_round_trips_by_value(self, states, field):
        entry, blank = states
        assert _snapshot(blank)[field] == _snapshot(entry)[field]

    @pytest.mark.parametrize("field", _POSIX_FIELDS)
    def test_field_round_trips_as_a_claripy_bv(self, states, field):
        """Not as a plain int: ``simos/linux.py`` calls ``argc.sign_extend`` and
        ``__libc_start_main`` stores ``argv`` straight into a register."""
        _entry, blank = states
        assert isinstance(getattr(blank.posix, field), claripy.ast.Base)

    def test_argc_keeps_its_width(self, states):
        """``state_entry`` builds a 32-bit argc, narrower than the arch word;
        widening it on restore would change what ``sign_extend(32)`` yields."""
        entry, blank = states
        assert blank.posix.argc.size() == entry.posix.argc.size()

    def test_absent_fields_are_left_alone(self):
        """A pre-angr-7tmoz pickle (or an unreadable field) must not crash the
        restore or write a bogus zero pointer."""
        project = _project()
        blank = project.factory.blank_state(addr=project.entry)
        _restore_posix_entry_fields(blank, {"posix_environ": 0x1000})
        assert blank.posix.environ is not None
        assert blank.posix.argv is None
        assert blank.posix.argc is None

    def test_symbolic_argc_is_not_persisted(self):
        """Concretizing it would silently drop a symbolic argc; None keeps the
        blank-state default instead."""
        project = _project()
        state = project.factory.entry_state(args=_ARGS, argc=claripy.BVS("argc", 32))
        assert _extract_posix_argc(state) is None


class TestWarmDiskCacheKeepsPosix:
    """The full pickle round-trip, through the real save + load entry points."""

    @pytest.fixture(scope="class")
    def cold_and_warm(self, tmp_path_factory):
        """Pickle an ``entry_state(args=..., env=...)`` and read it back.

        Redirects the cache directory at a tmp path for the whole fixture, so
        the run neither reads nor writes ``~/.cache/angr_rust_init`` --- and, in
        particular, cannot be short-circuited into a skip by a pickle a previous
        run of this file left behind.
        """
        project = _project()
        cache_dir = str(tmp_path_factory.mktemp("angr_rust_init"))
        # `mock.patch.object`, not a manual setattr/restore pair: the mixin
        # defines `_disk_cache_dir` as an inherited staticmethod, so restoring
        # it by hand re-binds a *plain function* onto the class, which then
        # eats `self` as its first argument and breaks disk caching for every
        # later test in the session.
        with mock.patch.object(RustExplorationManager, "_disk_cache_dir", staticmethod(lambda: cache_dir)):
            mgr = RustExplorationManager(project, [project.factory.entry_state(args=_ARGS, env=_ENV)])
            cold = project.factory.entry_state(args=_ARGS, env=_ENV)
            mgr._save_init_to_disk_cache(_CACHE_KEY, cold)
            assert os.path.exists(os.path.join(cache_dir, f"{_CACHE_KEY}.pkl")), "save wrote no pickle"
            warm, _mem_cache = mgr._load_init_from_disk_cache(_CACHE_KEY)
        assert warm is not None, "disk cache miss on the key just written"
        return _snapshot(cold), _snapshot(warm)

    @pytest.mark.parametrize("field", _POSIX_FIELDS)
    def test_warm_hit_matches_cold_state(self, cold_and_warm, field):
        cold, warm = cold_and_warm
        assert warm[field] == cold[field]

    def test_cold_state_actually_had_the_pointers(self, cold_and_warm):
        """Guards the comparison above from passing vacuously on None == None."""
        cold, _warm = cold_and_warm
        assert all(cold[f] is not None for f in _POSIX_FIELDS)
