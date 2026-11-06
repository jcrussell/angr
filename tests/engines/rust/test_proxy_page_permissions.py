"""``state.memory.permissions`` on a Rust state reads and sets the Rust page's bits (angr-vmi2l.330).

``RustMemoryProxy.permissions`` returned RWX (7) for every address and
dropped writes. Amd64 shellcode mmaps one anonymous RW page into ``r12``;
each case reads or sets permissions on the found state and compares both
engines.
"""

from __future__ import annotations

import struct

import pytest

import angr
from angr.errors import SimMemoryError
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

LOAD = 0x400000
UNMAPPED = 0x7000_0000


def _movabs(prefix: bytes, value: int) -> bytes:
    return prefix + struct.pack("<q", value)


# mmap(0, 0x1000, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0) -> r12
CODE = (
    _movabs(b"\x48\xbf", 0)  # rdi
    + _movabs(b"\x48\xbe", 0x1000)  # rsi
    + _movabs(b"\x48\xba", 3)  # rdx
    + _movabs(b"\x49\xba", 0x22)  # r10
    + _movabs(b"\x49\xb8", -1)  # r8
    + _movabs(b"\x49\xb9", 0)  # r9
    + _movabs(b"\x48\xb8", 9)  # rax = SYS_mmap
    + b"\x0f\x05"  # syscall
    + b"\x49\x89\xc4"  # mov r12, rax
)


def _found(engine: str):
    project = angr.load_shellcode(CODE + b"\xeb\xfe", arch="AMD64", simos="linux", load_address=LOAD)
    state = project.factory.entry_state(addr=LOAD)
    mgr = RustExplorationManager(project, [state]) if engine == "rust" else project.factory.simulation_manager(state)
    mgr.explore(find=LOAD + len(CODE), n=50)
    assert len(mgr.found) == 1
    found = mgr.found[0]
    return found, found.solver.eval(found.regs.r12)


def _perms(state, addr: int) -> int:
    return state.solver.eval(state.memory.permissions(addr))


def _read_case(engine: str):
    found, page = _found(engine)
    return _perms(found, page), _perms(found, LOAD)


def _write_case(engine: str):
    found, page = _found(engine)
    old = _perms(found, page)
    returned = found.solver.eval(found.memory.permissions(page, 1))
    return old, returned, _perms(found, page)


def test_reads_the_page_permissions():
    expected = _read_case("python")
    assert _read_case("rust") == expected
    assert expected[0] == 3


def test_write_sets_the_page_permissions():
    expected = _write_case("python")
    assert _write_case("rust") == expected
    assert expected == (3, 3, 1)


@pytest.mark.parametrize("engine", ["python", "rust"])
def test_unmapped_page_raises(engine):
    found, _ = _found(engine)
    with pytest.raises(SimMemoryError):
        found.memory.permissions(UNMAPPED)
    with pytest.raises(SimMemoryError):
        found.memory.permissions(UNMAPPED, 3)
