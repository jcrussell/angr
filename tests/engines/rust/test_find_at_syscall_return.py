"""A find/avoid address that is a syscall's return address waits for the syscall (angr-vmi2l.328).

A block-granular Rust step stops at an ``Ijk_Sys*`` exit with the state's pc
at the return address and the syscall pending. Python points that successor's
ip at the syscall's SimProcedure, so its explorer matches the return address
only after the syscall has run. Rust used to match the pending state, so the
found/avoided state lacked the syscall's return value and memory effects.
"""

from __future__ import annotations

import struct

import pytest

import angr
from angr.exploration.rust_manager import RustExplorationManager

LOAD = 0x400000
BUF = 0x30000000


def _sys(nr: int) -> bytes:
    return b"\xb8" + struct.pack("<I", nr) + b"\x0f\x05"


# getcwd(BUF, 256); END: nop; jmp $
CODE = b"\x48\xbf" + struct.pack("<Q", BUF) + b"\xbe\x00\x01\x00\x00" + _sys(79)
END = LOAD + len(CODE)


@pytest.mark.parametrize("stash", ["found", "avoid"])
@pytest.mark.parametrize("engine", ["py", "rust"])
def test_find_avoid_at_syscall_return_sees_syscall_effects(engine, stash):
    proj = angr.load_shellcode(CODE + b"\x90\xeb\xfe", arch="amd64", simos="linux", load_address=LOAD)
    state = proj.factory.blank_state(addr=LOAD, add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS})
    state.memory.store(BUF, b"\0" * 0x100)
    mgr = RustExplorationManager(proj, [state]) if engine == "rust" else proj.factory.simulation_manager(state)
    if stash == "found":
        mgr.explore(find=END, n=20)
    else:
        mgr.explore(find=END + 1, avoid=END, n=20)
    states = mgr.found if stash == "found" else mgr.avoid
    assert [s.addr for s in states] == [END]
    (s,) = states
    cwd = b"/home/user"
    assert s.solver.eval(s.regs.rax) == len(cwd) + 1
    assert s.solver.eval(s.memory.load(BUF, len(cwd) + 1), cast_to=bytes) == cwd + b"\0"
