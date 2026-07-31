"""Shared per-architecture spec table for the multi-arch Rust-engine harness.

Coverage of a minority architecture used to depend on someone remembering to
hand-author a test for that specific arch, so arch-specific bugs slipped
through disproportionately (the ``.1`` big-endian ``htonl``/``htons``/``pipe``
byte-order bug, the ``.111`` ``scanf``/``sprintf`` length-modifier width bug,
the ``.160`` unmapped NEON unsigned vector ``CmpGT``, ...). This module holds
the single table those tests sweep, so *adding an assertion* or *adding an
architecture* buys the whole matrix at once.

Two tables live here:

:data:`ARCH_SPECS`
    One :class:`ArchSpec` per architecture the Rust engine supports — the same
    six rows as ``ALL_ARCHES`` in ``native/angr/src/arch/mod.rs``. Carries the
    archinfo id, the ELF ``e_machine``/``e_flags``/class, the endianness
    flavours the arch ships in, the hand-assembled "solve for 42" program the
    end-to-end explore sweep runs, and the calling-convention register names
    the native-SimProcedure round-trip sweep pokes.

:data:`ARCHES`
    ``{rust_arch: archinfo_id}``, derived from :data:`ARCH_SPECS`. Consumed by
    ``test_arch_offset_parity.py``.

:func:`build_elf` replaces the five copy-pasted minimal-ELF builders that used
to live inline in ``test_multiarch.py``; they differed only in ELF class,
byte order, ``e_machine`` and ``e_flags``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

# ELF ``e_machine`` values (see ``elf.h``).
EM_386 = 3
EM_MIPS = 8
EM_ARM = 40
EM_X86_64 = 62
EM_AARCH64 = 0xB7

# ELF ``e_flags`` values that cle/archinfo actually look at.
EF_ARM_EABI_VER5 = 0x05000000
EF_MIPS_ARCH_32 = 0x50001000  # EF_MIPS_ARCH_32 | EF_MIPS_ABI_O32
EF_MIPS_ARCH_64 = 0x60000000  # N64 is implied by EI_CLASS=ELFCLASS64


@dataclass(frozen=True)
class EndianVariant:
    """One endianness flavour of an architecture.

    ``cle_arch`` is the name cle's Blob backend / ``archinfo.arch_from_id``
    accept for this flavour; the endness fields are the values a correctly
    decoded project must report, so the ELF half of the explore sweep is a
    real check on cle's ``EI_DATA``/``e_machine`` parsing rather than a
    tautology (Blob takes the arch name verbatim, ELF does not).
    """

    label: str
    cle_arch: str
    little: bool
    memory_endness: str
    instruction_endness: str


@dataclass(frozen=True)
class ProcSpec:
    """Calling-convention registers for the native-SimProcedure round trip.

    Only link-register architectures are described: the sweep asserts SP is
    *untouched* because the return address arrives in a register, which is
    false for the stack-based x86/amd64 conventions.
    """

    arg_reg: str
    ret_addr_reg: str
    ret_reg: str
    stack_base: int


@dataclass(frozen=True)
class ArchSpec:
    """Everything the multi-arch sweeps need to know about one architecture."""

    #: Name accepted by ``RustSimState`` / ``_RustExplorationManager``.
    rust_arch: str
    #: ``archinfo.arch_from_id`` id, used as the foreign source of truth.
    archinfo_id: str
    #: Value ``proj.arch.name`` must report once loaded.
    proj_arch_name: str
    bits: int
    e_machine: int
    e_flags: int
    base_addr: int
    variants: tuple[EndianVariant, ...]
    #: The "solve for 42" program, as fixed-width instruction words packed in
    #: the variant's byte order. Mutually exclusive with :attr:`code_bytes`.
    code_words: tuple[int, ...] | None
    #: The "solve for 42" program as literal bytes, for variable-length ISAs
    #: (x86/amd64) whose encoding does not change with the data endianness.
    code_bytes: bytes | None
    #: Byte offsets of the find / avoid targets within the program.
    find_off: int
    avoid_off: int
    #: Register the symbolic input is planted in, via ``state.regs.<name>``.
    input_reg: str
    input_bits: int
    #: Register names the state round-trip / fork-isolation sweeps poke. Sizes
    #: are looked up in archinfo rather than restated here, so a row cannot
    #: encode a wrong width. Each arch lists a scratch register, an argument
    #: register, the stack pointer and one more, so a wrong offset shows up as
    #: cross-talk between the four rather than a clean pass.
    probe_regs: tuple[str, ...]
    #: ``None`` for the stack-return-address architectures.
    proc: ProcSpec | None

    def code_for(self, variant: EndianVariant) -> bytes:
        """The explore program's bytes in ``variant``'s storage byte order."""
        if self.code_bytes is not None:
            return self.code_bytes
        assert self.code_words is not None, f"{self.rust_arch}: no explore program"
        prefix = "<" if variant.little else ">"
        return struct.pack(prefix + "I" * len(self.code_words), *self.code_words)


# ---------------------------------------------------------------------------
# The table.
#
# Every program below drives its symbolic input register to 42 on the path
# that reaches ``find_off`` (either ``2*x + 16 == 100`` or a direct
# ``x == 42`` compare), and to something else on the ``avoid_off`` path.
# ---------------------------------------------------------------------------

#: i386, little-endian only.
#:   0x00: 01 C0        add eax, eax        ; eax = 2*eax
#:   0x02: 83 C0 10     add eax, 0x10       ; eax = 2*eax + 16
#:   0x05: 83 F8 64     cmp eax, 100
#:   0x08: 74 06        je  +6  -> 0x10     (found)
#:   0x0a: EB 0C        jmp +12 -> 0x18     (avoid)
#:   0x0c: 90 90 90 90  pad
#:   0x10: 8x 90        found region
#:   0x18: 90           avoid
_X86_CODE = bytes.fromhex("01C083C01083F8647406EB0C90909090909090909090909090")

#: x86-64, little-endian only. Same shape as ``_X86_CODE`` on RDI.
#:   0x00: 48 01 FF        add rdi, rdi
#:   0x03: 48 83 C7 10     add rdi, 0x10
#:   0x07: 48 83 FF 64     cmp rdi, 100
#:   0x0b: 74 03           je  -> 0x10      (found)
#:   0x0d: EB 09           jmp -> 0x18      (avoid)
#:   0x0f: 90              pad
#:   0x10: 8x 90           found region
#:   0x18: 90              avoid
_AMD64_CODE = bytes.fromhex("4801FF4883C7104883FF647403EB0990909090909090909090")

#: ARM, AL condition (0xE). Words are packed per-variant, which is what makes
#: the ARMEB (BE32) row a genuine instruction-fetch byte-order test.
#:   0x00: ADD  r0, r0, r0    0xE0800000
#:   0x04: ADD  r0, r0, #16   0xE2800010
#:   0x08: MOV  r1, #100      0xE3A01064
#:   0x0c: CMP  r0, r1        0xE1500001
#:   0x10: BEQ  +0  -> 0x18   0x0A000000  (found)
#:   0x14: B    +4  -> 0x20   0xEA000001  (avoid)
#:   0x18: NOP               0xE1A00000
#:   0x1c: NOP               0xE1A00000
#:   0x20: NOP               0xE1A00000
_ARM_WORDS = (
    0xE0800000,
    0xE2800010,
    0xE3A01064,
    0xE1500001,
    0x0A000000,
    0xEA000001,
    0xE1A00000,
    0xE1A00000,
    0xE1A00000,
)

#: AArch64, little-endian only.
#:   0x00: MOV  w1, #42       52800541
#:   0x04: CMP  w0, w1        6B01001F  (SUBS wzr, w0, w1)
#:   0x08: B.EQ +8 -> 0x10    54000040  (found)
#:   0x0c: B   +12 -> 0x18    14000003  (avoid)
#:   0x10: NOP                D503201F
#:   0x14: NOP                D503201F
#:   0x18: NOP                D503201F
_ARM64_WORDS = (
    0x52800541,
    0x6B01001F,
    0x54000040,
    0x14000003,
    0xD503201F,
    0xD503201F,
    0xD503201F,
)

#: MIPS, with delay slots. The base-integer encodings below are identical on
#: MIPS32 and MIPS64 (register fields are still 5 bits; ADDIU sign-extends the
#: immediate to 64 bits on MIPS64), so both rows share the program.
#:   0x00: ADDIU t0, zero, 42     2408002A
#:   0x04: BEQ   a0, t0, +3       10880003  -> 0x14 (found)
#:   0x08: NOP  (delay slot)      00000000
#:   0x0c: B     +2               10000002  -> 0x18 (avoid)
#:   0x10: NOP  (delay slot)      00000000
#:   0x14: NOP  (found)           00000000
#:   0x18: NOP  (avoid)           00000000
_MIPS_WORDS = (
    0x2408002A,
    0x10880003,
    0x00000000,
    0x10000002,
    0x00000000,
    0x00000000,
    0x00000000,
)

#: One row per architecture the Rust engine supports. Order matches the
#: historical ``ARCHES`` dict in ``test_arch_offset_parity`` so that module's
#: parametrize ids are unchanged.
ARCH_SPECS: tuple[ArchSpec, ...] = (
    ArchSpec(
        rust_arch="amd64",
        archinfo_id="AMD64",
        proj_arch_name="AMD64",
        bits=64,
        e_machine=EM_X86_64,
        e_flags=0,
        base_addr=0x400000,
        variants=(EndianVariant("amd64", "amd64", True, "Iend_LE", "Iend_BE"),),
        code_words=None,
        code_bytes=_AMD64_CODE,
        find_off=0x10,
        avoid_off=0x18,
        input_reg="rdi",
        input_bits=64,
        probe_regs=("rax", "rdi", "rsp", "rbp"),
        # amd64 CALL pushes the return address; the SP-untouched assertion in
        # the round-trip sweep does not apply.
        proc=None,
    ),
    ArchSpec(
        rust_arch="x86",
        archinfo_id="X86",
        proj_arch_name="X86",
        bits=32,
        e_machine=EM_386,
        e_flags=0,
        base_addr=0x400000,
        variants=(EndianVariant("x86", "x86", True, "Iend_LE", "Iend_BE"),),
        code_words=None,
        code_bytes=_X86_CODE,
        find_off=0x10,
        avoid_off=0x18,
        input_reg="eax",
        input_bits=32,
        probe_regs=("eax", "ecx", "esp", "ebp"),
        proc=None,  # CALL pushes the return address, see the amd64 row.
    ),
    ArchSpec(
        rust_arch="arm",
        archinfo_id="ARMEL",
        proj_arch_name="ARMEL",
        bits=32,
        e_machine=EM_ARM,
        e_flags=EF_ARM_EABI_VER5,
        base_addr=0x400000,
        variants=(
            EndianVariant("armel", "armel", True, "Iend_LE", "Iend_LE"),
            # archinfo maps ``armeb`` onto ARMEL with BE memory *and* BE
            # instruction fetch (the BE32 model).
            EndianVariant("armeb", "armeb", False, "Iend_BE", "Iend_BE"),
        ),
        code_words=_ARM_WORDS,
        code_bytes=None,
        find_off=0x18,
        avoid_off=0x20,
        input_reg="r0",
        input_bits=32,
        probe_regs=("r0", "r1", "sp", "lr"),
        # BL leaves the return address in LR (R14); SP must not move.
        proc=ProcSpec(arg_reg="r0", ret_addr_reg="lr", ret_reg="r0", stack_base=0x7FFF0000),
    ),
    ArchSpec(
        rust_arch="arm64",
        archinfo_id="AArch64",
        proj_arch_name="AARCH64",
        bits=64,
        e_machine=EM_AARCH64,
        e_flags=0,
        base_addr=0x400000,
        variants=(EndianVariant("aarch64", "aarch64", True, "Iend_LE", "Iend_LE"),),
        code_words=_ARM64_WORDS,
        code_bytes=None,
        find_off=0x10,
        avoid_off=0x18,
        input_reg="x0",
        input_bits=64,
        probe_regs=("x0", "x1", "sp", "x30"),
        # BL leaves the return address in X30 (LR); SP must not move.
        proc=ProcSpec(arg_reg="x0", ret_addr_reg="x30", ret_reg="x0", stack_base=0x7FFFFFFFE000),
    ),
    ArchSpec(
        rust_arch="mips32",
        archinfo_id="MIPS32",
        proj_arch_name="MIPS32",
        bits=32,
        e_machine=EM_MIPS,
        e_flags=EF_MIPS_ARCH_32,
        base_addr=0x400000,
        variants=(
            EndianVariant("mips32be", "mips", False, "Iend_BE", "Iend_BE"),
            EndianVariant("mips32le", "mipsel", True, "Iend_LE", "Iend_BE"),
        ),
        code_words=_MIPS_WORDS,
        code_bytes=None,
        find_off=0x14,
        avoid_off=0x18,
        input_reg="a0",
        input_bits=32,
        probe_regs=("v0", "a0", "sp", "t0"),
        # O32: JAL leaves the return address in $ra (R31); SP must not move.
        proc=ProcSpec(arg_reg="a0", ret_addr_reg="ra", ret_reg="v0", stack_base=0x7FFF0000),
    ),
    ArchSpec(
        rust_arch="mips64",
        archinfo_id="MIPS64",
        proj_arch_name="MIPS64",
        bits=64,
        e_machine=EM_MIPS,
        e_flags=EF_MIPS_ARCH_64,
        base_addr=0x400000,
        variants=(
            EndianVariant("mips64be", "mips64", False, "Iend_BE", "Iend_BE"),
            EndianVariant("mips64le", "mips64el", True, "Iend_LE", "Iend_BE"),
        ),
        code_words=_MIPS_WORDS,
        code_bytes=None,
        find_off=0x14,
        avoid_off=0x18,
        input_reg="a0",
        input_bits=64,
        probe_regs=("v0", "a0", "sp", "t0"),
        # N64: same $ra/$v0 shape as O32, wider registers.
        proc=ProcSpec(arg_reg="a0", ret_addr_reg="ra", ret_reg="v0", stack_base=0x7FFF0000),
    ),
)

#: ``{rust_arch: archinfo_id}`` — the table ``test_arch_offset_parity`` sweeps.
ARCHES: dict[str, str] = {spec.rust_arch: spec.archinfo_id for spec in ARCH_SPECS}


def spec_for(rust_arch: str) -> ArchSpec:
    """The :class:`ArchSpec` whose ``rust_arch`` is ``rust_arch``."""
    for spec in ARCH_SPECS:
        if spec.rust_arch == rust_arch:
            return spec
    raise KeyError(rust_arch)


def build_elf(spec: ArchSpec, variant: EndianVariant, code: bytes, entry_skip: int = 0) -> tuple[bytes, int]:
    """Wrap ``code`` in a minimal single-PT_LOAD ET_EXEC ELF.

    Replaces the five near-identical inline builders ``test_multiarch.py``
    used to carry; they differed only in ELF class (32 vs 64), byte order,
    ``e_machine`` and ``e_flags``, all of which are :class:`ArchSpec` /
    :class:`EndianVariant` columns now.

    The segment maps file offset 0 at ``spec.base_addr``, so ``code`` starts
    right after the headers. ``entry_skip`` moves ``e_entry`` that many bytes
    further in (used to skip a leading subroutine). Returns
    ``(elf_bytes, entry_vaddr)``.
    """
    endian = "<" if variant.little else ">"
    is64 = spec.bits == 64
    ehdr_size = 64 if is64 else 52
    phdr_size = 56 if is64 else 32
    total = ehdr_size + phdr_size + len(code)
    entry = spec.base_addr + ehdr_size + phdr_size + entry_skip

    e_ident = (
        b"\x7fELF"
        + bytes(
            [
                2 if is64 else 1,  # EI_CLASS
                1 if variant.little else 2,  # EI_DATA
                1,  # EI_VERSION
                0,  # EI_OSABI = System V
                0,  # EI_ABIVERSION
            ]
        )
        + b"\x00" * 7  # EI_PAD
    )

    header_fields = (
        2,  # e_type = ET_EXEC
        spec.e_machine,
        1,  # e_version
        entry,  # e_entry
        ehdr_size,  # e_phoff
        0,  # e_shoff
        spec.e_flags,
        ehdr_size,  # e_ehsize
        phdr_size,  # e_phentsize
        1,  # e_phnum
        0,  # e_shentsize
        0,  # e_shnum
        0,  # e_shstrndx
    )
    if is64:
        ehdr = e_ident + struct.pack(endian + "HHIQQQIHHHHHH", *header_fields)
        # ELF64 program header: p_flags follows p_type.
        phdr = struct.pack(
            endian + "IIQQQQQQ",
            1,  # p_type = PT_LOAD
            5,  # p_flags = PF_R | PF_X
            0,  # p_offset
            spec.base_addr,  # p_vaddr
            spec.base_addr,  # p_paddr
            total,  # p_filesz
            total,  # p_memsz
            0x1000,  # p_align
        )
    else:
        ehdr = e_ident + struct.pack(endian + "HHIIIIIHHHHHH", *header_fields)
        # ELF32 program header: p_flags sits near the end instead.
        phdr = struct.pack(
            endian + "IIIIIIII",
            1,  # p_type = PT_LOAD
            0,  # p_offset
            spec.base_addr,  # p_vaddr
            spec.base_addr,  # p_paddr
            total,  # p_filesz
            total,  # p_memsz
            5,  # p_flags = PF_R | PF_X
            0x1000,  # p_align
        )

    assert len(ehdr) == ehdr_size
    assert len(phdr) == phdr_size
    return ehdr + phdr + code, entry
