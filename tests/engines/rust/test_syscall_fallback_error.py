"""A Python syscall fallback that raises errors the state, as the Python engine does (angr-vmi2l.156).

The callback's ``except`` arm resumed at ``state.addr + 1``, one byte into
the instruction after the syscall, with none of the syscall's effects: amd64
ran on decoding garbage, and MIPS hit an IR decode error at the odd address
(angr-vmi2l.331). Each case issues an ``mmap`` with symbolic flags, which the
native handler declines and Python's ``mmap`` rejects with ``SimPosixError``:
symbolic flags on amd64, a symbolic address on mips32.
"""

from __future__ import annotations

import struct

import pytest

import angr
from angr import claripy
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

LOAD = 0x400000


def _amd64():
    # mov rax, 9; mov rdi, 0; mov rsi, 0x1000; mov rdx, 3; mov r8, -1; mov r9, 0;
    # syscall; mov r12, rax
    code = (
        b"\x48\xc7\xc0\x09\x00\x00\x00"
        + b"\x48\xc7\xc7\x00\x00\x00\x00"
        + b"\x48\xc7\xc6\x00\x10\x00\x00"
        + b"\x48\xc7\xc2\x03\x00\x00\x00"
        + b"\x49\xc7\xc0\xff\xff\xff\xff"
        + b"\x49\xc7\xc1\x00\x00\x00\x00"
        + b"\x0f\x05"
        + b"\x49\x89\xc4"
    )
    project = angr.load_shellcode(code + b"\xeb\xfe", arch="AMD64", simos="linux", load_address=LOAD)
    state = project.factory.entry_state(addr=LOAD)
    state.regs.r10 = claripy.BVS("flags", 64)
    return project, state, LOAD + len(code)


def _mips32():
    def w(x):
        return struct.pack(">I", x & 0xFFFFFFFF)

    def li(rt, v):
        return w(0x3C000000 | (rt << 16) | (v >> 16)) + w(0x34000000 | (rt << 21) | (rt << 16) | (v & 0xFFFF))

    # mmap2(<symbolic a0>, 0x2000, 3, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0); syscall; move s0, v0
    code = li(4, 0) + li(2, 4210) + li(5, 0x2000) + li(6, 3) + li(7, 0x802) + w(0xC) + w(0x00408021)
    project = angr.load_shellcode(code + w(0x1000FFFF) + w(0), arch="MIPS32", simos="linux", load_address=LOAD)
    # Start past ``li a0`` so a0 keeps its symbolic value.
    state = project.factory.entry_state(addr=LOAD + 8)
    state.regs.a0 = claripy.BVS("addr", 32)
    # Linux reads args 5-6 at sp+16; SimCCO32LinuxSyscall reads them at sp+0.
    sp = state.solver.eval(state.regs.sp)
    for base in (0, 16):
        state.memory.store(sp + base, claripy.BVV(-1, 32), endness="Iend_BE")
        state.memory.store(sp + base + 4, claripy.BVV(0, 32), endness="Iend_BE")
    return project, state, LOAD + len(code)


@pytest.mark.parametrize("build", [_amd64, _mips32], ids=["amd64", "mips32"])
def test_raising_syscall_fallback_errors_the_state(build):
    outcomes = {}
    for engine in ("python", "rust"):
        project, state, end = build()
        if engine == "rust":
            mgr = RustExplorationManager(project, [state])
        else:
            mgr = project.factory.simulation_manager(state)
        mgr.explore(find=end, n=50)
        errors = [str(getattr(e, "error", e)) for e in mgr.errored]
        outcomes[engine] = (len(mgr.found), len(mgr.active), len(errors))
        assert all("Cannot handle symbolic" in msg for msg in errors), errors
    assert outcomes["rust"] == outcomes["python"] == (0, 0, 1)
