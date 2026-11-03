"""Raw ``lseek`` with a negative offset matches the Python engine (angr-vmi2l.113).

On i386 ``off_t`` is 32 bits, so the native handler must sign-extend it: a
zero-extended ``-1`` seeks to ``+0xffffffff``. On both arches a negative
target fails with -1 and leaves the position alone, where Rust used to clamp
it to 0 and report success. Each case opens an ``fs.insert``-ed file, seeks,
reads two bytes, and compares the seek result, read count and buffer.
"""

from __future__ import annotations

import struct

import pytest

import angr
from angr.exploration.rust_manager import RustExplorationManager

LOAD = 0x400000
DATA = 0x30000000
NAME = DATA
BUF = DATA + 0x100
CONTENT = b"ABCDEF"


def _x86(off: int, whence: int) -> bytes:
    def mov(op: int, v: int) -> bytes:
        return bytes([op]) + struct.pack("<I", v & 0xFFFFFFFF)

    return (
        mov(0xBB, NAME) + mov(0xB9, 0) + mov(0xB8, 5) + b"\xcd\x80"  # open(NAME, O_RDONLY)
        + b"\x89\xc6"  # mov esi, eax
        + b"\x89\xf3" + mov(0xB9, off) + mov(0xBA, whence) + mov(0xB8, 19) + b"\xcd\x80"  # lseek
        + b"\x89\xc7"  # mov edi, eax
        + b"\x89\xf3" + mov(0xB9, BUF) + mov(0xBA, 2) + mov(0xB8, 3) + b"\xcd\x80"  # read(fd, BUF, 2)
    )  # fmt: skip


def _amd64(off: int, whence: int) -> bytes:
    def mov(reg: bytes, v: int) -> bytes:
        return b"\x48" + reg + struct.pack("<Q", v & 0xFFFFFFFFFFFFFFFF)

    def sys(nr: int) -> bytes:
        return b"\xb8" + struct.pack("<I", nr) + b"\x0f\x05"

    rdi, rsi, rdx = b"\xbf", b"\xbe", b"\xba"
    return (
        mov(rdi, NAME) + mov(rsi, 0) + sys(2)  # open(NAME, O_RDONLY)
        + b"\x49\x89\xc4"  # mov r12, rax
        + b"\x4c\x89\xe7" + mov(rsi, off) + mov(rdx, whence) + sys(8)  # lseek
        + b"\x49\x89\xc5"  # mov r13, rax
        + b"\x4c\x89\xe7" + mov(rsi, BUF) + mov(rdx, 2) + sys(0)  # read(fd, BUF, 2)
    )  # fmt: skip


def _run(engine: str, arch: str, off: int, whence: int):
    code = (_x86 if arch == "x86" else _amd64)(off, whence)
    end = LOAD + len(code)
    proj = angr.load_shellcode(code + b"\x90\xeb\xfe", arch=arch, simos="linux", load_address=LOAD)
    state = proj.factory.blank_state(addr=LOAD, add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS})
    state.memory.store(DATA, b"\0" * 0x200)
    state.memory.store(NAME, b"/g\0")
    state.fs.insert(b"/g", angr.SimFile(b"/g", content=CONTENT, has_end=True))
    mgr = RustExplorationManager(proj, [state]) if engine == "rust" else proj.factory.simulation_manager(state)
    mgr.explore(find=end, n=20)
    assert len(mgr.found) == 1
    (found,) = mgr.found
    seek_reg, ret_reg = ("edi", "eax") if arch == "x86" else ("r13", "rax")
    bits = proj.arch.bits
    seek = found.solver.eval(getattr(found.regs, seek_reg))
    if seek >> (bits - 1):
        seek -= 1 << bits
    return (
        seek,
        found.solver.eval(getattr(found.regs, ret_reg)),
        found.solver.eval(found.memory.load(BUF, 2), cast_to=bytes),
    )


@pytest.mark.parametrize(
    ("off", "whence", "want"),
    [
        (-2, 2, (4, 2, b"EF")),  # SEEK_END - 2
        (-1, 2, (5, 1, b"F\0")),  # SEEK_END - 1
        (-5, 0, (-1, 2, b"AB")),  # negative SEEK_SET fails, position stays 0
        (-7, 2, (-1, 2, b"AB")),  # SEEK_END past the start fails
    ],
    ids=["end_minus_2", "end_minus_1", "neg_set", "end_before_start"],
)
@pytest.mark.parametrize("arch", ["x86", "amd64"])
@pytest.mark.parametrize("engine", ["py", "rust"])
def test_lseek_negative_offset(engine, arch, off, whence, want):
    assert _run(engine, arch, off, whence) == want
