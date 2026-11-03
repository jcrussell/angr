# pylint: disable=missing-class-docstring,no-self-use
"""Regression tests: symbolic stdout from a bounced proc (angr-vmi2l.264).

A Python hook writes ``"hi" + u`` to stdout, then a native ``puts("OK")``
runs. The write-back used to drop a symbolic segment from Rust's fd buffer,
so the found state lost the native ``OK\\n`` and a proxy saw only it.
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from angr.exploration.rust_state_export import _stream_is_prefix_of
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration engine not built")

LOAD, HOOK, PUTS, MSG = 0x400000, 0x500000, 0x500100, 0x600000
# mov rax,HOOK; call rax; mov rdi,MSG; mov rax,PUTS; call rax; jmp $
CODE = (
    bytes([0x48, 0xB8])
    + HOOK.to_bytes(8, "little")
    + bytes([0xFF, 0xD0, 0x48, 0xBF])
    + MSG.to_bytes(8, "little")
    + bytes([0x48, 0xB8])
    + PUTS.to_bytes(8, "little")
    + bytes([0xFF, 0xD0, 0xEB, 0xFE])
)
END = LOAD + len(CODE) - 2
YZ, AB = int.from_bytes(b"yz", "big"), int.from_bytes(b"ab", "big")


def _setup(constraint):
    p = angr.load_shellcode(CODE, arch="AMD64", load_address=LOAD)
    u = claripy.BVS("u", 16)

    class Write(angr.SimProcedure):
        def run(self):  # pylint: disable=arguments-differ
            self.state.add_constraints(constraint(u))
            self.state.posix.get_fd(1).write_data(claripy.Concat(claripy.BVV(b"hi"), u))
            return 0

    p.hook(HOOK, Write(), replace=True)
    puts = angr.SIM_PROCEDURES["libc"]["puts"](prototype=angr.SIM_LIBRARIES["libc.so.6"][0].get_prototype("puts"))
    p.hook(PUTS, puts, replace=True)
    st = p.factory.blank_state(
        addr=LOAD,
        add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS, angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY},
    )
    st.regs.rsp = 0x7FFFFFFF0800
    st.memory.store(MSG, b"OK\0")
    return p, st, u


def _found_stdout(engine, constraint, find=END):
    p, st, u = _setup(constraint)
    mgr = RustExplorationManager(p, [st]) if engine == "rust" else p.factory.simulation_manager(st)
    mgr.explore(find=find)
    return mgr, u


@pytest.mark.parametrize("engine", ["python", "rust"])
def test_single_valued_output_precedes_native_output(engine):
    mgr, _ = _found_stdout(engine, lambda u: u == YZ)
    assert [f.posix.dumps(1) for f in mgr.found] == [b"hiyzOK\n"]


def test_proxy_sees_bounced_and_native_output():
    mgr, _ = _found_stdout("rust", lambda u: u == YZ)
    assert [px.posix.dumps(1) for px in mgr.found_proxies()] == [b"hiyzOK\n"]


def test_find_predicate_sees_bounced_output():
    mgr, _ = _found_stdout("rust", lambda u: u == YZ, find=lambda s: s.posix.dumps(1) == b"hiyzOK\n")
    assert len(mgr.found) == 1


def test_multi_valued_output_keeps_its_ast_on_export():
    mgr, u = _found_stdout("rust", lambda u: claripy.Or(u == YZ, u == AB))
    (found,) = mgr.found
    found.add_constraints(u == AB)
    assert found.posix.dumps(1) == b"hiabOK\n"


def test_symbolic_packet_rust_does_not_cover_is_not_a_prefix():
    _, st, u = _setup(lambda u: u == YZ)
    st.posix.get_fd(1).write_data(b"hi")
    st.posix.get_fd(1).write_data(u)
    assert _stream_is_prefix_of(st, st.posix.stdout, b"hiyz")
    assert _stream_is_prefix_of(st, st.posix.stdout, b"hixxOK")
    assert not _stream_is_prefix_of(st, st.posix.stdout, b"hiy")
    assert not _stream_is_prefix_of(st, st.posix.stdout, b"hoyz")
