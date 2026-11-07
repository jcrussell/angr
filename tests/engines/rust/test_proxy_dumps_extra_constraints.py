# pylint: disable=missing-class-docstring,no-self-use
"""Regression tests: a proxy's ``posix.dumps(0, extra_constraints=...)`` honours
the extras, as ``SimSystemPosix.dumps`` does (angr-vmi2l.335).

``RustPosixProxy.dumps`` took only ``fd``, so a ``find`` predicate passing
``extra_constraints`` raised ``TypeError``. Each case runs the same predicate on
the Python engine and on the Rust proxy and compares what it saw.
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration engine not built")

# sub rsp,0x20 ; read(0, rsp, 4) ; good: jmp good -- stdin stays unconstrained
SHELLCODE = bytes.fromhex("4883ec2031c031ff4889e6ba040000000f05ebfe")
LOAD = 0x400000
GOOD = LOAD + 0x12


def _seen_by_find(engine, make_state, query):
    proj = angr.load_shellcode(SHELLCODE, arch="amd64", simos="linux", load_address=LOAD)
    seen = []

    def at_good(s):
        if s.addr != GOOD:
            return False
        seen.append(query(s))
        return True

    state = make_state(proj)
    if engine == "rust":
        mgr = RustExplorationManager(proj, [state])
    else:
        mgr = proj.factory.simulation_manager(state)
    mgr.explore(find=at_good)
    assert mgr.found, f"{engine}: the read never reached the success state"
    assert seen, f"{engine}: the find predicate never saw the success state"
    return seen[-1]


def _both(make_state, query):
    return _seen_by_find("python", make_state, query), _seen_by_find("rust", make_state, query)


class TestProxyDumpsExtraConstraints:
    def test_seeded_symbol_follows_the_extra(self):
        seeded = claripy.BVS("seeded", 4 * 8)

        def make(proj):
            state = proj.factory.blank_state(addr=LOAD)
            state.posix.stdin.content.append((seeded, claripy.BVV(4, 64)))
            return state

        def query(s):
            return s.posix.dumps(0, extra_constraints=[seeded == int.from_bytes(b"wxyz", "big")])

        python, rust = _both(make, query)
        assert python == b"wxyz"
        assert rust == python

    def test_minted_bytes_follow_an_extra_over_memory(self):
        def make(proj):
            return proj.factory.blank_state(addr=LOAD)

        def query(s):
            return s.posix.dumps(0, extra_constraints=[s.memory.load(s.regs.rsp, 4) == b"wxyz"])

        python, rust = _both(make, query)
        assert python == b"wxyz"
        assert rust == python

    def test_output_fd_accepts_and_ignores_extras(self):
        def make(proj):
            return proj.factory.blank_state(addr=LOAD)

        def query(s):
            return s.posix.dumps(1, extra_constraints=[claripy.true()])

        python, rust = _both(make, query)
        assert python == b""
        assert rust == python
