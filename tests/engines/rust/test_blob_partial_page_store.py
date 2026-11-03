"""A store into a partly backed page of a blob image keeps the image's bytes (angr-vmi2l.159).

The blob covers less than a page, so the loader page cache had no whole page
for it and the first store mapped a zero page over the image: neighbours,
code and the export then read 0. Each case is compared with the Python
engine.
"""

from __future__ import annotations

import struct

import pytest

import angr
from angr.exploration.rust_manager import RustExplorationManager

LOAD = 0x400000
FIND = LOAD + 0x80
STORE, READ = LOAD + 0x100, LOAD + 0x108
SMC = LOAD + 0x40
OUTSIDE = LOAD + 0x300


def _jmp_find() -> bytes:
    # mov rdx, FIND; jmp rdx
    return b"\x48\xba" + struct.pack("<Q", FIND) + b"\xff\xe2"


def _blob(code: bytes) -> bytes:
    blob = bytearray(code.ljust(0x200, b"\x90"))
    blob[0x80:0x82] = b"\xeb\xfe"
    blob[0x100:0x110] = bytes(range(0x50, 0x60))
    # mov eax, 1; jmp FIND, patched to mov eax, 7 by the SMC case
    blob[0x40:0x51] = b"\xb8\x01\x00\x00\x00" + _jmp_find()
    return bytes(blob)


# mov byte [STORE], 2; mov rbx, [READ]; mov rcx, [OUTSIDE]; jmp FIND
NEIGHBOUR = (
    b"\xc6\x04\x25" + struct.pack("<I", STORE) + b"\x02"
    + b"\x48\x8b\x1c\x25" + struct.pack("<I", READ)
    + b"\x48\x8b\x0c\x25" + struct.pack("<I", OUTSIDE)
    + _jmp_find()
)  # fmt: skip
# mov byte [SMC+1], 7; mov rax, SMC; jmp rax
PATCH = b"\xc6\x04\x25" + struct.pack("<I", SMC + 1) + b"\x07" + b"\x48\xb8" + struct.pack("<Q", SMC) + b"\xff\xe0"


def _found(engine: str, code: bytes, zero_fill: bool):
    proj = angr.load_shellcode(_blob(code), arch="amd64", load_address=LOAD)
    options = {angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
    if zero_fill:
        options.add(angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY)
    state = proj.factory.blank_state(addr=LOAD, add_options=options)
    mgr = RustExplorationManager(proj, [state]) if engine == "rust" else proj.factory.simulation_manager(state)
    mgr.explore(find=FIND, n=20)
    (found,) = mgr.found
    return found


@pytest.mark.parametrize("zero_fill", [False, True], ids=["default", "zero_fill"])
@pytest.mark.parametrize("engine", ["py", "rust"])
def test_store_keeps_image_neighbour(engine, zero_fill):
    f = _found(engine, NEIGHBOUR, zero_fill)
    assert f.solver.eval_upto(f.regs.rbx, 2) == [0x5F5E5D5C5B5A5958]
    assert f.solver.eval_upto(f.memory.load(READ, 8, endness="Iend_LE"), 2) == [0x5F5E5D5C5B5A5958]
    assert f.solver.eval_upto(f.memory.load(STORE, 2), 2) == [0x0251]
    # The rest of the blob's page is zero, as the Python engine's page
    # initialization pads it, whatever the zero-fill option.
    assert f.solver.eval_upto(f.regs.rcx, 2) == [0]


@pytest.mark.parametrize("zero_fill", [False, True], ids=["default", "zero_fill"])
@pytest.mark.parametrize("engine", ["py", "rust"])
def test_self_modifying_code_in_sub_page_blob(engine, zero_fill):
    f = _found(engine, PATCH, zero_fill)
    assert f.solver.eval_upto(f.memory.load(SMC, 2), 2) == [0xB807]
    # The Python engine lifts from the loader's backer, so it runs the
    # unpatched `mov eax, 1`; Rust runs the patched bytes.
    assert f.solver.eval_upto(f.regs.eax, 2) == [7 if engine == "rust" else 1]
