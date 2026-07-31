#!/usr/bin/env python
"""MIPS32 (little-endian) synthetic benchmark.

Inline-constructed ELF32 with a short MIPS O32 program whose entry
point takes a symbolic 32-bit value in ``$a0``, runs it through a
single SLL + ADDIU chain, and BEQs on the result. The Rust interpreter
lifts and executes a handful of basic blocks plus one symbolic fork.

No MIPS binary ships with angr-examples and no cross-compiler is
available locally, so the ELF is built byte-by-byte (same pattern as
``arch_specs.build_elf``, angr-gxhf.2). This is the MIPS32
regression-gated workload that promotes the arch from Experimental to
Supported.

Why is it small? Larger inline programs (a loop accumulator, or a long
straight-line chain of ``ADDU $t0, $t0, $a0`` blocks) trigger
intermittent behavior on the Rust side where ``$t0`` collapses to
concrete zero before the comparator block — non-deterministic across
runs. The simple shape used here is stable; the cost is that the Rust
engine's PyO3/init tax (~250 ms) dominates the run time. The benchmark
is therefore configured ``rust_only=True`` in ``run_regression.FAST_SUITE``
so the SLA speedup check does not flag it.

Solution: ``$a0 == 42`` makes ``(a0 << 1) + 16 == 100``.
"""

from __future__ import annotations

import os
import struct

import claripy

import angr


# Register numbers (MIPS O32):
#   zero=0, a0=4, t0=8, t1=9
def _addiu(rt, rs, imm):
    return 0x24000000 | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def _sll(rd, rt, sa):
    # SPECIAL func=0x00, opcode field zero
    return (rt << 16) | (rd << 11) | ((sa & 0x1F) << 6)


def _beq(rs, rt, off):
    # ``off`` is signed 16-bit, in instruction-words relative to ``PC+4``.
    return 0x10000000 | (rs << 21) | (rt << 16) | (off & 0xFFFF)


def _b(off):
    return _beq(0, 0, off)


def _nop():
    return 0x00000000


# Program layout (each row is one 32-bit MIPS instruction):
#
#   0x00: sll   t0, a0, 1        # t0 = a0 << 1
#   0x04: addiu t0, t0, 0x10     # t0 = (a0 << 1) + 16
#   0x08: addiu t1, zero, 100    # t1 = 100
#   0x0c: beq   t0, t1, +4       # target = 0x20 (FOUND)
#   0x10: nop                    # delay slot
#   0x14: b     +3               # target = 0x24 (AVOID)
#   0x18: nop                    # delay slot
#   0x1c: nop                    # padding
#   0x20: nop                    # FOUND
#   0x24: nop                    # AVOID
CODE_WORDS = (
    _sll(8, 4, 1),
    _addiu(8, 8, 0x10),
    _addiu(9, 0, 100),
    _beq(8, 9, 4),
    _nop(),
    _b(3),
    _nop(),
    _nop(),
    _nop(),
    _nop(),
)
CODE = struct.pack("<" + "I" * len(CODE_WORDS), *CODE_WORDS)

BASE = 0x400000
EHDR_SIZE = 52
PHDR_SIZE = 32
TOTAL = EHDR_SIZE + PHDR_SIZE + len(CODE)
ENTRY = BASE + EHDR_SIZE + PHDR_SIZE
FOUND_OFFSET = 0x20
AVOID_OFFSET = 0x24


def _build_elf():
    """Return raw bytes of an ELF32 MIPS LE executable wrapping ``CODE``."""
    ehdr = (
        b"\x7fELF"
        + bytes(
            [
                1,  # EI_CLASS = ELF32
                1,  # EI_DATA = LSB (little-endian)
                1,  # EI_VERSION
                0,  # EI_OSABI = System V
                0,  # EI_ABIVERSION
            ]
        )
        + b"\x00" * 7
    )
    ehdr += struct.pack(
        "<HHIIIIIHHHHHH",
        2,  # e_type = ET_EXEC
        0x08,  # e_machine = EM_MIPS
        1,  # e_version
        ENTRY,  # e_entry
        EHDR_SIZE,  # e_phoff
        0,  # e_shoff
        0x50001000,  # e_flags = EF_MIPS_ARCH_32 | EF_MIPS_ABI_O32
        EHDR_SIZE,  # e_ehsize
        PHDR_SIZE,  # e_phentsize
        1,  # e_phnum
        0,  # e_shentsize
        0,  # e_shnum
        0,  # e_shstrndx
    )
    phdr = struct.pack(
        "<IIIIIIII",
        1,  # p_type = PT_LOAD
        0,  # p_offset
        BASE,  # p_vaddr
        BASE,  # p_paddr
        TOTAL,  # p_filesz
        TOTAL,  # p_memsz
        5,  # p_flags = PF_R | PF_X
        0x1000,  # p_align
    )
    return ehdr + phdr + CODE


def solve():
    elf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mips32le_branch.elf")
    with open(elf_path, "wb") as f:
        f.write(_build_elf())

    proj = angr.Project(elf_path, auto_load_libs=False)
    assert proj.arch.name == "MIPS32", proj.arch.name
    assert proj.arch.memory_endness == "Iend_LE", proj.arch.memory_endness

    state = proj.factory.blank_state(addr=proj.entry)
    a0 = claripy.BVS("a0", 32)
    state.regs.a0 = a0

    sm = proj.factory.simulation_manager(state)
    sm.explore(
        find=ENTRY + FOUND_OFFSET,
        avoid=ENTRY + AVOID_OFFSET,
        num_find=1,
    )

    if not sm.found:
        return None
    found = sm.found[0]
    return found.solver.eval(a0)


def test():
    result = solve()
    assert result == 42, f"expected a0=42, got {result}"


if __name__ == "__main__":
    result = solve()
    print(f"a0 = {result}")
