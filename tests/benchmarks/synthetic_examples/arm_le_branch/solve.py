#!/usr/bin/env python
"""ARM (ARMEL, little-endian) synthetic benchmark.

Inline-constructed ELF32 with a short ARM program whose entry point
takes a symbolic 32-bit value in ``r0``, runs it through a doubling +
add + compare chain, and branches on the result. The Rust interpreter
lifts and executes a handful of basic blocks plus one symbolic fork.

The angr-examples Android validate binary
(``test_arm32_explore_real_binary``) covers a real-world ARM ELF
end-to-end, but it requires the angr-examples checkout on disk and
skips in environments where it is absent — unsuitable as a
regression-gated workload. This synthetic ships its ELF inline so the
benchmark runs unconditionally, matching the ``aarch64_le_branch`` and
``mips32_le_branch`` pattern. Promotes ARM from Experimental to
Supported in the support matrix.

Avoids NEON / VFP and Thumb encodings — only base ARM scalar ops
(``ADD``, ``MOV imm``, ``CMP``, ``BEQ``, ``B``) so dispatch stays on
the non-scaffolded path.

Solution: ``r0 == 42`` makes ``2*r0 + 16 == 100``.
"""

from __future__ import annotations

import os
import struct

import claripy

import angr

# ARM (AL condition = 0xE) little-endian instruction encodings.
# Each is a 32-bit ARM word; the file stores them little-endian
# (per ELF EI_DATA = LSB on ARMEL).
#
#   0x00: ADD  r0, r0, r0       0xE0800000  ; r0 = 2*r0
#   0x04: ADD  r0, r0, #16      0xE2800010  ; r0 = 2*r0 + 16
#   0x08: MOV  r1, #100         0xE3A01064  ; r1 = 100
#   0x0c: CMP  r0, r1           0xE1500001  ; flags <- r0 - r1
#   0x10: BEQ  +0  -> 0x18      0x0A000000  ; target = PC+8+0 = 0x18
#   0x14: B    +4  -> 0x20      0xEA000001  ; target = PC+8+4 = 0x20
#   0x18: NOP  (found)          0xE1A00000  ; MOV r0,r0
#   0x1c: NOP                   0xE1A00000
#   0x20: NOP  (avoid)          0xE1A00000
#   0x24: NOP                   0xE1A00000
CODE_WORDS = (
    0xE0800000,  # ADD r0, r0, r0
    0xE2800010,  # ADD r0, r0, #16
    0xE3A01064,  # MOV r1, #100
    0xE1500001,  # CMP r0, r1
    0x0A000000,  # BEQ +0 (target 0x18)
    0xEA000001,  # B   +4 (target 0x20)
    0xE1A00000,  # NOP (found)
    0xE1A00000,  # NOP
    0xE1A00000,  # NOP (avoid)
    0xE1A00000,  # NOP (padding)
)
CODE = struct.pack("<" + "I" * len(CODE_WORDS), *CODE_WORDS)

BASE = 0x10000
EHDR_SIZE = 52
PHDR_SIZE = 32
TOTAL = EHDR_SIZE + PHDR_SIZE + len(CODE)
ENTRY = BASE + EHDR_SIZE + PHDR_SIZE
FOUND_OFFSET = 0x18
AVOID_OFFSET = 0x20


def _build_elf():
    """Return raw bytes of an ELF32 ARM LE executable wrapping ``CODE``."""
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
        0x28,  # e_machine = EM_ARM
        1,  # e_version
        ENTRY,  # e_entry
        EHDR_SIZE,  # e_phoff
        0,  # e_shoff
        0x05000000,  # e_flags = EF_ARM_EABI v5
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
    elf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "armle_branch.elf")
    with open(elf_path, "wb") as f:
        f.write(_build_elf())

    proj = angr.Project(elf_path, auto_load_libs=False)
    assert proj.arch.name == "ARMEL", proj.arch.name
    assert proj.arch.memory_endness == "Iend_LE", proj.arch.memory_endness

    state = proj.factory.blank_state(addr=proj.entry)
    r0 = claripy.BVS("r0", 32)
    state.regs.r0 = r0

    sm = proj.factory.simulation_manager(state)
    sm.explore(
        find=ENTRY + FOUND_OFFSET,
        avoid=ENTRY + AVOID_OFFSET,
        num_find=1,
    )

    if not sm.found:
        return None
    found = sm.found[0]
    return found.solver.eval(r0)


def test():
    result = solve()
    assert result == 42, f"expected r0=42, got {result}"


if __name__ == "__main__":
    result = solve()
    print(f"r0 = {result}")
