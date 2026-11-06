# pylint: disable=missing-class-docstring,no-self-use
"""Regression tests: a proxy's ``posix.dumps(0)`` lays a seeded stdin out as the
export does (angr-vmi2l.267).

A seeded ``posix.stdin`` reaches Rust as fd 0's content (``_seed_stdin_to_rust``),
not as stdin symbols, and the export puts it back from the seed state's stream.
The proxy evaluated only the symbols native reads minted, so a ``find`` predicate
saw ``b""`` (or only the bytes read past the seed) where ``mgr.found[0]`` saw the
seed.
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


def _proxy_and_export_stdin(make_state):
    proj = angr.load_shellcode(SHELLCODE, arch="amd64", simos="linux", load_address=LOAD)
    seen = []

    def at_good(s):
        if s.addr != GOOD:
            return False
        seen.append(s.posix.dumps(0))
        return True

    mgr = RustExplorationManager(proj, [make_state(proj)])
    mgr.explore(find=at_good, avoid=BAD)
    assert mgr.found, "the read bytes never reached the success branch"
    assert seen, "the find predicate never saw the success state"
    return seen[-1], mgr.found[0].posix.dumps(0)


class TestProxySeededStdin:
    def test_file_backed_concrete_seed(self):
        def make(proj):
            stdin = angr.SimFileStream(name="stdin", content=b"abcd", has_end=True)
            return proj.factory.blank_state(addr=LOAD, stdin=stdin)

        proxy, export = _proxy_and_export_stdin(make)
        assert export == b"abcd"
        assert proxy == export

    def test_symbolic_packet_seed(self):
        def make(proj):
            state = proj.factory.blank_state(addr=LOAD)
            state.posix.stdin.content.append((claripy.BVS("seeded", 4 * 8), claripy.BVV(4, 64)))
            return state

        proxy, export = _proxy_and_export_stdin(make)
        assert export == b"abcd"
        assert proxy == export

    def test_seed_followed_by_minted_bytes(self):
        """A two-byte seed, then two bytes the native read mints past it, in stream order."""

        def make(proj):
            state = proj.factory.blank_state(addr=LOAD)
            state.posix.stdin.content.append((claripy.BVV(b"ab"), claripy.BVV(2, 64)))
            return state

        proxy, export = _proxy_and_export_stdin(make)
        assert export == b"abcd"
        assert proxy == export
