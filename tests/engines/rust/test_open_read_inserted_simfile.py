"""Raw ``open``+``read`` syscalls on an ``fs.insert``-ed SimFile read its content (angr-vmi2l.327).

The reported zero-byte read was a find at the ``read`` syscall's return
address matching the Rust state before the syscall ran (angr-vmi2l.328). Each
case opens a relative, absolute, or post-``chdir`` relative name, reads three
bytes, and compares with the Python engine.
"""

from __future__ import annotations

import struct

import pytest

import angr
from angr.exploration.rust_manager import RustExplorationManager

LOAD = 0x400000
DATA = 0x30000000
DIR = DATA
NAME = DATA + 0x100
BUF = DATA + 0x200


def _mov(reg: str, v: int) -> bytes:
    return {"rdi": b"\x48\xbf", "rsi": b"\x48\xbe", "rdx": b"\x48\xba"}[reg] + struct.pack("<Q", v)


def _sys(nr: int) -> bytes:
    return b"\xb8" + struct.pack("<I", nr) + b"\x0f\x05"


OPEN_READ = (
    _mov("rdi", NAME) + _mov("rsi", 0) + _sys(2)  # open(NAME, O_RDONLY)
    + b"\x48\x89\xc7" + _mov("rsi", BUF) + _mov("rdx", 3) + _sys(0)  # read(fd, BUF, 3)
)  # fmt: skip


def _run(engine: str, chdir: bytes | None, name: bytes):
    code = (_mov("rdi", DIR) + _sys(80) if chdir is not None else b"") + OPEN_READ
    end = LOAD + len(code)
    proj = angr.load_shellcode(code + b"\x90\xeb\xfe", arch="amd64", simos="linux", load_address=LOAD)
    state = proj.factory.blank_state(addr=LOAD, add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS})
    state.memory.store(DATA, b"\0" * 0x300)
    if chdir is not None:
        state.memory.store(DIR, chdir + b"\0")
    state.memory.store(NAME, name + b"\0")
    state.fs.insert(b"/home/user/x", angr.SimFile(b"/home/user/x", content=b"XYZ", has_end=True))
    mgr = RustExplorationManager(proj, [state]) if engine == "rust" else proj.factory.simulation_manager(state)
    mgr.explore(find=end, n=20)
    assert len(mgr.found) == 1
    (found,) = mgr.found
    return found.solver.eval(found.regs.rax), found.solver.eval(found.memory.load(BUF, 3), cast_to=bytes)


@pytest.mark.parametrize(
    ("chdir", "name"),
    [(None, b"x"), (None, b"/home/user/x"), (b"/home", b"user/x")],
    ids=["relative", "absolute", "relative_after_chdir"],
)
@pytest.mark.parametrize("engine", ["py", "rust"])
def test_open_read_inserted_simfile(engine, chdir, name):
    assert _run(engine, chdir, name) == (3, b"XYZ")
