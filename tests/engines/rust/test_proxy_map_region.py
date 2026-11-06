"""``state.memory.map_region`` / ``unmap_region`` on a Rust state map and drop Rust pages (angr-vmi2l.146).

``RustMemoryProxy.map_region`` raised for every range outside the Rust heap,
so each Python fallback for a mapping syscall errored the state:
a file-backed mmap, a brk that collides with a mapped page, and an mmap whose
hint is mapped (mmap.py's retry loop needs ``SimMemoryError``). Each case
runs shellcode on both engines and compares the result register, the
mapping's permissions and its first bytes.
"""

from __future__ import annotations

import struct

import pytest

import angr
from angr import claripy
from angr.errors import SimMemoryError
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

LOAD = 0x400000
PATH = 0x600000
SYSCALL = b"\x0f\x05"
MOV_R12_RAX = b"\x49\x89\xc4"
MOV_R13_RAX = b"\x49\x89\xc5"
MOV_R8_RAX = b"\x49\x89\xc0"
MOV_RDI_R13 = b"\x4c\x89\xef"

_MOVABS = {
    "rax": b"\x48\xb8",
    "rdi": b"\x48\xbf",
    "rsi": b"\x48\xbe",
    "rdx": b"\x48\xba",
    "r10": b"\x49\xba",
    "r8": b"\x49\xb8",
    "r9": b"\x49\xb9",
}


def _mov(reg: str, value: int) -> bytes:
    return _MOVABS[reg] + struct.pack("<q", value)


def _add_rdi(value: int) -> bytes:
    return b"\x48\x81\xc7" + struct.pack("<i", value)


# open("/f", 0) -> r8; mmap(0, 0x1000, PROT_READ, MAP_PRIVATE, r8, 0) -> r12
FILEMAP = (
    _mov("rax", 2)
    + _mov("rdi", PATH)
    + _mov("rsi", 0)
    + SYSCALL
    + MOV_R8_RAX
    + _mov("rax", 9)
    + _mov("rdi", 0)
    + _mov("rsi", 0x1000)
    + _mov("rdx", 1)
    + _mov("r10", 2)
    + _mov("r9", 0)
    + SYSCALL
    + MOV_R12_RAX
)

# r13 = brk(0); mmap(r13 + 0x2000, 0x1000, RW, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0);
# brk(r13 + 0x4000) -> r12, which collides with the fixed page
BRK_COLLIDE = (
    _mov("rax", 12)
    + _mov("rdi", 0)
    + SYSCALL
    + MOV_R13_RAX
    + MOV_RDI_R13
    + _add_rdi(0x2000)
    + _mov("rax", 9)
    + _mov("rsi", 0x1000)
    + _mov("rdx", 3)
    + _mov("r10", 0x32)
    + _mov("r8", -1)
    + _mov("r9", 0)
    + SYSCALL
    + _mov("rax", 12)
    + MOV_RDI_R13
    + _add_rdi(0x4000)
    + SYSCALL
    + MOV_R12_RAX
)


def _explore(project, state, find: int, engine: str):
    mgr = RustExplorationManager(project, [state]) if engine == "rust" else project.factory.simulation_manager(state)
    mgr.explore(find=find, n=100)
    assert len(mgr.found) == 1, f"{engine}: errored={list(mgr.errored)}"
    return mgr.found[0]


def _amd64_found(code: bytes, engine: str, file_content: bytes | None = None):
    project = angr.load_shellcode(code + b"\xeb\xfe", arch="AMD64", simos="linux", load_address=LOAD)
    state = project.factory.entry_state(addr=LOAD, add_options={angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY})
    if file_content is not None:
        state.memory.store(PATH, b"/f\0")
        state.fs.insert("/f", angr.SimFile("/f", content=file_content, has_end=True))
    return _explore(project, state, LOAD + len(code), engine)


def _mapping(state, addr: int) -> tuple:
    perms = state.solver.eval(state.memory.permissions(addr))
    return addr, perms, state.solver.eval(state.memory.load(addr, 8), cast_to=bytes)


def _filemap_case(engine: str):
    found = _amd64_found(FILEMAP, engine, file_content=b"ABCDEFGH" * 0x200)
    return _mapping(found, found.solver.eval(found.regs.r12))


def _brk_case(engine: str):
    found = _amd64_found(BRK_COLLIDE, engine)
    brk = found.solver.eval(found.regs.r12)
    start = found.solver.eval(found.regs.r13)
    return brk - start, found.solver.eval(found.memory.permissions(start + 0x1000))


def test_file_backed_mmap_fallback_maps_the_file():
    expected = _filemap_case("python")
    assert _filemap_case("rust") == expected
    assert expected == (0xC1000000, 1, b"ABCDEFGH")


def test_brk_collision_stops_the_break_at_the_mapped_page():
    expected = _brk_case("python")
    assert _brk_case("rust") == expected
    assert expected == (0x2000, 7)


def _be32(value: int) -> bytes:
    return struct.pack(">I", value & 0xFFFFFFFF)


def _li(reg: int, value: int) -> bytes:
    value &= 0xFFFFFFFF
    return _be32(0x3C000000 | (reg << 16) | (value >> 16)) + _be32(
        0x34000000 | (reg << 21) | (reg << 16) | (value & 0xFFFF)
    )


def _mips_hint_case(engine: str):
    # mmap2(PATH, 0x2000, RW, MAP_PRIVATE | MAP_ANONYMOUS(0x800), -1, 0) -> s0, with PATH mapped:
    # native declines the colliding hint and mmap.py retries at mmap_base.
    code = _li(2, 4210) + _li(4, PATH) + _li(5, 0x2000) + _li(6, 3) + _li(7, 0x802) + _be32(0xC) + _be32(0x00408021)
    project = angr.load_shellcode(code + _be32(0x1000FFFF) + _be32(0), arch="MIPS32", simos="linux", load_address=LOAD)
    state = project.factory.entry_state(addr=LOAD, add_options={angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY})
    state.memory.store(PATH, b"\0" * 0x40)
    sp = state.solver.eval(state.regs.sp)
    # args 5-6 (fd, offset) at sp+16 per Linux o32, and at sp+0 where SimCCO32LinuxSyscall reads them
    for off, value in ((16, -1), (20, 0), (0, -1), (4, 0)):
        state.memory.store(sp + off, claripy.BVV(value & 0xFFFFFFFF, 32), endness="Iend_BE")
    found = _explore(project, state, LOAD + len(code), engine)
    return _mapping(found, found.solver.eval(found.regs.s0)), found.heap.mmap_base


def test_mmap_hint_collision_retries_at_mmap_base():
    expected = _mips_hint_case("python")
    assert _mips_hint_case("rust") == expected
    assert expected[0][:2] == (0xC1000000, 3)


@pytest.mark.parametrize("engine", ["python", "rust"])
def test_map_region_collision_and_unmap_on_a_found_state(engine):
    found = _amd64_found(FILEMAP, engine, file_content=b"A" * 0x1000)
    page = found.solver.eval(found.regs.r12)
    with pytest.raises(SimMemoryError) as info:
        found.memory.map_region(page - 0x1000, 0x2000, 3)
    assert info.value.args[1] == page
    assert found.solver.eval(found.memory.permissions(page - 0x1000)) == 3
    found.memory.unmap_region(page - 0x1000, 0x2000)
    for addr in (page - 0x1000, page):
        with pytest.raises(SimMemoryError):
            found.memory.permissions(addr)
