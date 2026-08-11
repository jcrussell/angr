//! x86 (32-bit) architecture definition.
//!
//! This module provides the register layout and architecture-specific
//! information for 32-bit x86.

use super::{Arch, RegEntry, impl_arch_registers};
use crate::vex::VexArch;

/// x86 (32-bit) architecture.
#[derive(Debug, Clone, Copy)]
pub(crate) struct X86;

// x86 VEX guest state offsets (from archinfo.ArchX86)
// These match pyvex's register layout for compatibility.
pub(crate) mod offsets {
    // General purpose registers
    pub(crate) const EAX: u32 = 8;
    pub(crate) const ECX: u32 = 12;
    pub(crate) const EDX: u32 = 16;
    pub(crate) const EBX: u32 = 20;
    pub(crate) const ESP: u32 = 24;
    pub(crate) const EBP: u32 = 28;
    pub(crate) const ESI: u32 = 32;
    pub(crate) const EDI: u32 = 36;

    // Flags thunks
    pub(crate) const CC_OP: u32 = 40;
    pub(crate) const CC_DEP1: u32 = 44;
    pub(crate) const CC_DEP2: u32 = 48;
    pub(crate) const CC_NDEP: u32 = 52;

    // Other flags
    pub(crate) const DFLAG: u32 = 56;
    pub(crate) const IDFLAG: u32 = 60;
    pub(crate) const ACFLAG: u32 = 64;
    pub(crate) const EIP: u32 = 68;

    // FPU registers (before SSE in archinfo layout)
    // VEX stores the x87 stack as `guest_FPREG[8]` of `ULong`, i.e. 8 x 8-byte
    // slots (64 bytes total, ending where FPTAG starts) -- MMX-style 64-bit
    // storage, not true 80-bit x87 extended precision. Per-index offset is
    // therefore `FPREG + n * 8`. Same layout as AMD64's `offsets::FPREG`; the
    // widths diverge only below it, where x86's FPROUND/FC3210/FTOP are `UInt`
    // (4 bytes) against AMD64's `ULong` (VexGuestX86State,
    // libvex_guest_x86.h).
    pub(crate) const FPREG: u32 = 72;
    // `guest_FPTAG` is UChar[8] — one x87 tag byte per FPREG slot.
    pub(crate) const FPTAG: u32 = 136;
    pub(crate) const FPROUND: u32 = 144;
    pub(crate) const FC3210: u32 = 148;
    pub(crate) const FTOP: u32 = 152;

    // SSE
    pub(crate) const SSEROUND: u32 = 156;
    pub(crate) const XMM0: u32 = 160;
    pub(crate) const XMM1: u32 = 176;
    pub(crate) const XMM2: u32 = 192;
    pub(crate) const XMM3: u32 = 208;
    pub(crate) const XMM4: u32 = 224;
    pub(crate) const XMM5: u32 = 240;
    pub(crate) const XMM6: u32 = 256;
    pub(crate) const XMM7: u32 = 272;

    // Segment selectors (after XMM registers)
    pub(crate) const CS: u32 = 288;
    pub(crate) const DS: u32 = 290;
    pub(crate) const ES: u32 = 292;
    pub(crate) const FS: u32 = 294;
    pub(crate) const GS: u32 = 296;
    pub(crate) const SS: u32 = 298;

    // Segment base addresses
    pub(crate) const LDT: u32 = 304;
    pub(crate) const GDT: u32 = 312;
    // VEX bookkeeping tail (VexGuestX86State, libvex_guest_x86.h). x86 has no
    // guest_FS_CONST/GS_CONST — that pair is amd64-only — so 320/324 are
    // guest_EMNOTE/guest_CMSTART, not scratch space (angr-rfxc7).
    pub(crate) const EMNOTE: u32 = 320;
    pub(crate) const CMSTART: u32 = 324;
    pub(crate) const CMLEN: u32 = 328;
    pub(crate) const NRADDR: u32 = 332;
    pub(crate) const SC_CLASS: u32 = 336;
    pub(crate) const IP_AT_SYSCALL: u32 = 340;

    // Total guest state size (must cover all registers). 344 is the end of
    // guest_IP_AT_SYSCALL; everything past it is the struct's explicit
    // padding1..3, not guest state, so archinfo and VEX agree here.
    pub(crate) const GUEST_STATE_SIZE: usize = 344;
}

// Canonical registers: drive `register_name(offset)` reverse lookups.
const CANONICAL: &[RegEntry] = &[
    // 32-bit GPRs
    ("eax", offsets::EAX, 4),
    ("ecx", offsets::ECX, 4),
    ("edx", offsets::EDX, 4),
    ("ebx", offsets::EBX, 4),
    ("esp", offsets::ESP, 4),
    ("ebp", offsets::EBP, 4),
    ("esi", offsets::ESI, 4),
    ("edi", offsets::EDI, 4),
    ("eip", offsets::EIP, 4),
    // Flags thunks
    ("cc_op", offsets::CC_OP, 4),
    ("cc_dep1", offsets::CC_DEP1, 4),
    ("cc_dep2", offsets::CC_DEP2, 4),
    ("cc_ndep", offsets::CC_NDEP, 4),
    ("dflag", offsets::DFLAG, 4),
    ("idflag", offsets::IDFLAG, 4),
    ("acflag", offsets::ACFLAG, 4),
    // SSE control register (4B per archinfo.ArchX86)
    ("sseround", offsets::SSEROUND, 4),
    // XMM registers (128-bit). Canonical, not aliased, to match AMD64's split:
    // they are their own registers, not sub-names of a GPR, and `REGISTER_NAMES`
    // exports them — so `register_name(offset)` must resolve them (angr-9ke6b.9).
    ("xmm0", offsets::XMM0, 16),
    ("xmm1", offsets::XMM1, 16),
    ("xmm2", offsets::XMM2, 16),
    ("xmm3", offsets::XMM3, 16),
    ("xmm4", offsets::XMM4, 16),
    ("xmm5", offsets::XMM5, 16),
    ("xmm6", offsets::XMM6, 16),
    ("xmm7", offsets::XMM7, 16),
    // FPU. `fpreg`'s 64 bytes and `fptag`'s 8 are the VEX `guest_FPREG[8]` /
    // `guest_FPTAG[8]` arrays — see the provenance comment on `offsets::FPREG`.
    ("fpreg", offsets::FPREG, 64),
    ("fptag", offsets::FPTAG, 8),
    ("fpround", offsets::FPROUND, 4),
    ("fc3210", offsets::FC3210, 4),
    ("ftop", offsets::FTOP, 4),
];

// Aliases: alternate names, sub-registers, and segments.
// `register_offset` and `register_size` consult these as a fallback;
// `register_name` does not.
const ALIASES: &[RegEntry] = &[
    // Architecture-independent full-width aliases (match archinfo). The legacy
    // 16-bit `sp`/`bp` sub-registers are deliberately absent: archinfo binds
    // these two names to the full-width stack/base pointer (X86 `sp` is
    // (24, 4)), so a `set_register("sp", ...)` must not truncate to 2 bytes.
    ("sp", offsets::ESP, 4),
    ("bp", offsets::EBP, 4),
    // The two architecture-independent instruction-pointer spellings, matching
    // the ARM/ARM64/MIPS tables (angr-9ke6b.217) and archinfo's ArchX86
    // pc=(68,4)/ip=(68,4). `ip` is the *instruction pointer* on every arch,
    // never an ABI scratch register — see the angr-itm3u note in arch/arm.rs
    // (angr-690nc).
    ("pc", offsets::EIP, 4),
    ("ip", offsets::EIP, 4),
    // 16-bit
    ("ax", offsets::EAX, 2),
    ("cx", offsets::ECX, 2),
    ("dx", offsets::EDX, 2),
    ("bx", offsets::EBX, 2),
    ("si", offsets::ESI, 2),
    ("di", offsets::EDI, 2),
    // 8-bit low
    ("al", offsets::EAX, 1),
    ("cl", offsets::ECX, 1),
    ("dl", offsets::EDX, 1),
    ("bl", offsets::EBX, 1),
    // 8-bit high (offset+1)
    ("ah", offsets::EAX + 1, 1),
    ("ch", offsets::ECX + 1, 1),
    ("dh", offsets::EDX + 1, 1),
    ("bh", offsets::EBX + 1, 1),
    // Flag aliases
    ("d", offsets::DFLAG, 4),
    ("id", offsets::IDFLAG, 4),
    ("ac", offsets::ACFLAG, 4),
    // Segments (16-bit selectors)
    ("cs", offsets::CS, 2),
    ("ds", offsets::DS, 2),
    ("es", offsets::ES, 2),
    ("fs", offsets::FS, 2),
    ("gs", offsets::GS, 2),
    ("ss", offsets::SS, 2),
    // VEX bookkeeping tail (see the offsets note above).
    ("emnote", offsets::EMNOTE, 4),
    ("cmstart", offsets::CMSTART, 4),
    ("cmlen", offsets::CMLEN, 4),
    ("nraddr", offsets::NRADDR, 4),
    ("sc_class", offsets::SC_CLASS, 4),
    ("ip_at_syscall", offsets::IP_AT_SYSCALL, 4),
    // Segment-base descriptor tables (archinfo.ArchX86: 8B each, zero-init).
    // Surfaced for TLS-aware analyses that read state.regs.ldt/gdt.
    ("ldt", offsets::LDT, 8),
    ("gdt", offsets::GDT, 8),
];

// Registers exported to / imported from Python (see `Arch::register_names`).
// `fpreg` is omitted for the same u128-width reason as on AMD64 (angr-9ke6b.6);
// every other `CANONICAL` entry must be here, which is what
// `mod_tests::every_narrow_canonical_register_is_exported` enforces.
// `sseround` was the last straggler: it lives in `CANONICAL` and in AMD64's
// export list, but was left out here, so a Python-side `state.regs.sseround`
// write on an X86 state was dropped by `_supported_register_names`
// (angr-sqfj8.1).
const REGISTER_NAMES: &[&str] = &[
    "eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi", "eip", "cc_op", "cc_dep1", "cc_dep2",
    "cc_ndep", "dflag", "idflag", "acflag", "sseround", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4",
    "xmm5", "xmm6", "xmm7", "fptag", "fpround", "fc3210", "ftop",
];

impl Arch for X86 {
    fn vex_arch(&self) -> VexArch {
        VexArch::X86
    }

    fn name(&self) -> &'static str {
        "X86"
    }

    fn bits(&self) -> u32 {
        32
    }

    fn state_size(&self) -> usize {
        offsets::GUEST_STATE_SIZE
    }

    fn ip_offset(&self) -> u32 {
        offsets::EIP
    }

    fn sp_offset(&self) -> u32 {
        offsets::ESP
    }

    fn bp_offset(&self) -> Option<u32> {
        Some(offsets::EBP)
    }

    impl_arch_registers!(CANONICAL, ALIASES, REGISTER_NAMES);

    fn syscall_num_offset(&self) -> Option<u32> {
        Some(offsets::EAX)
    }

    fn is_little_endian(&self) -> bool {
        true
    }
}

test_submod!("x86_tests.rs" => tests);
