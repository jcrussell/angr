#!/usr/bin/env python
"""AArch64 (little-endian) synthetic benchmark.

Inline-constructed ELF64 with a short AArch64 program whose entry point
takes a symbolic 64-bit value in ``x0`` (used as ``w0``), runs it through
a doubling + add + compare chain, and branches on the result. The Rust
interpreter lifts and executes a handful of basic blocks plus one
symbolic fork.

No AArch64 binary ships with angr-examples and no cross-compiler is
available locally, so the ELF is built byte-by-byte (same pattern as
``test_aarch64_explore_real_elf``). This is the AArch64 regression-gated
workload that promotes the arch from Experimental to Supported.

The program intentionally avoids NEON SIMD ops — those still use the
``NeonUnimplemented`` scaffold and would panic. Only base scalar ops
(``ADD``, ``MOVZ``, ``CMP``, ``B.cond``, ``B``, ``NOP``) are used.

The program is short, so Rust's PyO3 init tax (~250 ms) dominates the
runtime. Marked ``rust_only=True`` in ``FAST_SUITE`` so the 0.5x SLA gate
is skipped; the regression gate still validates correctness + 15% timing
drift against the baseline ``rust_time``.

Solution: ``w0 == 42`` makes ``2*w0 + 16 == 100``.
"""

from __future__ import annotations

import os
import struct

import claripy

import angr

# AArch64 little-endian instructions (verified against ARMv8 ARM, see
# invariant-aarch64-inline-test-opcodes memory):
#
#   0x00: add  w0, w0, w0       0B000000   ; w0 = 2*w0
#   0x04: add  w0, w0, #16      11004000   ; w0 = 2*w0 + 16
#   0x08: movz w1, #100         52800C81   ; w1 = 100
#   0x0c: cmp  w0, w1           6B01001F   ; subs wzr, w0, w1
#   0x10: b.eq +8               54000040   ; -> 0x18 (FOUND)
#   0x14: b    +12              14000003   ; -> 0x20 (AVOID)
#   0x18: nop                   D503201F   ; FOUND
#   0x1c: nop                   D503201F
#   0x20: nop                   D503201F   ; AVOID
#   0x24: nop                   D503201F   ; padding
CODE_WORDS = (
    0x0B000000,  # add w0, w0, w0
    0x11004000,  # add w0, w0, #16
    0x52800C81,  # movz w1, #100
    0x6B01001F,  # cmp w0, w1
    0x54000040,  # b.eq +8 -> found
    0x14000003,  # b +12 -> avoid
    0xD503201F,  # nop (found)
    0xD503201F,  # nop
    0xD503201F,  # nop (avoid)
    0xD503201F,  # nop (padding)
)
CODE = struct.pack("<" + "I" * len(CODE_WORDS), *CODE_WORDS)

BASE = 0x400000
EHDR_SIZE = 64
PHDR_SIZE = 56
TOTAL = EHDR_SIZE + PHDR_SIZE + len(CODE)
ENTRY = BASE + EHDR_SIZE + PHDR_SIZE
FOUND_OFFSET = 0x18
AVOID_OFFSET = 0x20


def _build_elf():
    """Return raw bytes of an ELF64 AArch64 LE executable wrapping ``CODE``."""
    ehdr = (
        b"\x7fELF"
        + bytes(
            [
                2,  # EI_CLASS = ELF64
                1,  # EI_DATA = LSB (little-endian)
                1,  # EI_VERSION
                0,  # EI_OSABI = System V
                0,  # EI_ABIVERSION
            ]
        )
        + b"\x00" * 7
    )
    ehdr += struct.pack(
        "<HHIQQQIHHHHHH",
        2,  # e_type = ET_EXEC
        0xB7,  # e_machine = EM_AARCH64
        1,  # e_version
        ENTRY,  # e_entry
        EHDR_SIZE,  # e_phoff
        0,  # e_shoff
        0,  # e_flags
        EHDR_SIZE,  # e_ehsize
        PHDR_SIZE,  # e_phentsize
        1,  # e_phnum
        0,  # e_shentsize
        0,  # e_shnum
        0,  # e_shstrndx
    )
    phdr = struct.pack(
        "<IIQQQQQQ",
        1,  # p_type = PT_LOAD
        5,  # p_flags = PF_R | PF_X
        0,  # p_offset
        BASE,  # p_vaddr
        BASE,  # p_paddr
        TOTAL,  # p_filesz
        TOTAL,  # p_memsz
        0x1000,  # p_align
    )
    return ehdr + phdr + CODE


def solve():
    elf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aarch64le_branch.elf")
    with open(elf_path, "wb") as f:
        f.write(_build_elf())

    proj = angr.Project(elf_path, auto_load_libs=False)
    assert proj.arch.name == "AARCH64", proj.arch.name
    assert proj.arch.memory_endness == "Iend_LE", proj.arch.memory_endness

    state = proj.factory.blank_state(addr=proj.entry)
    x0 = claripy.BVS("x0", 64)
    state.regs.x0 = x0

    sm = proj.factory.simulation_manager(state)
    sm.explore(
        find=ENTRY + FOUND_OFFSET,
        avoid=ENTRY + AVOID_OFFSET,
        num_find=1,
    )

    if not sm.found:
        return None
    found = sm.found[0]
    return found.solver.eval(x0)


def test():
    result = solve()
    assert result == 42, f"expected x0=42, got {result}"


if __name__ == "__main__":
    result = solve()
    print(f"x0 = {result}")
