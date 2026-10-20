# pylint: disable=missing-class-docstring,no-self-use
"""Regression tests: a harness's own stdin symbol must be solved through the
raw read syscall and a file-backed stdin (angr-23n3r). test_stdin_seed.py covers
the libc ``read`` procedure with packet stdin.

``blank_state(stdin=BVS)`` makes ``posix.stdin`` a ``SimFileStream``, which has
no ``content`` packet list. ``_seed_stdin_to_rust`` only flattened packet
lists, so for a file-backed stdin it silently seeded nothing: the native
``read(0, ...)`` minted fresh ``stdin_*`` symbols, the path constraints were
phrased over those, and ``found.posix.dumps(0)`` evaluated the harness's
unconstrained BVS. ekopartyctf2016_rev250 got a team name the binary rejects.
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration engine not built")

# sub rsp,0x20 ; read(0, rsp, 4) ; cmp dword [rsp], "abcd" ; jne bad ; good: jmp good ; bad: jmp bad
SHELLCODE = bytes.fromhex("4883ec2031c031ff4889e6ba040000000f05813c24616263647502ebfeebfe")
LOAD = 0x400000
GOOD = LOAD + 0x1B
BAD = LOAD + 0x1D


def _found_stdin(make_state):
    proj = angr.load_shellcode(SHELLCODE, arch="amd64", simos="linux", load_address=LOAD)
    state, sym = make_state(proj)
    mgr = RustExplorationManager(proj, [state])
    mgr.explore(find=GOOD, avoid=BAD)
    assert mgr.found, "the read bytes never reached the success branch"
    found = mgr.found[0]
    return found, sym


class TestHarnessSeededStdin:
    def test_file_backed_stdin_symbol_is_solved(self):
        def make(proj):
            sym = claripy.BVS("seeded_stdin", 4 * 8)
            return proj.factory.blank_state(addr=LOAD, stdin=sym), sym

        found, sym = _found_stdin(make)
        assert found.posix.dumps(0)[:4] == b"abcd"
        assert found.solver.eval(sym, cast_to=bytes) == b"abcd"

    def test_packet_stdin_symbol_is_solved(self):
        """The packet-list shape that was already seeded keeps working."""

        def make(proj):
            sym = claripy.BVS("seeded_packets", 4 * 8)
            state = proj.factory.blank_state(addr=LOAD)
            state.posix.stdin.content.append((sym, claripy.BVV(4, 64)))
            return state, sym

        found, sym = _found_stdin(make)
        assert found.solver.eval(sym, cast_to=bytes) == b"abcd"
