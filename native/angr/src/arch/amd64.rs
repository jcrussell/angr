//! AMD64 (x86-64) architecture definition.
//!
//! This module provides the register layout and architecture-specific
//! information for AMD64.

use super::{Arch, RegEntry, impl_arch_registers};
use crate::vex::VexArch;

/// AMD64 architecture.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AMD64;

// AMD64 VEX guest state offsets (from VEX/pub/libvex_guest_amd64.h)
// These match the VexGuestAMD64State structure layout.
pub(crate) mod offsets {
    pub(crate) const RAX: u32 = 16;
    pub(crate) const RCX: u32 = 24;
    pub(crate) const RDX: u32 = 32;
    pub(crate) const RBX: u32 = 40;
    pub(crate) const RSP: u32 = 48;
    pub(crate) const RBP: u32 = 56;
    pub(crate) const RSI: u32 = 64;
    pub(crate) const RDI: u32 = 72;
    pub(crate) const R8: u32 = 80;
    pub(crate) const R9: u32 = 88;
    pub(crate) const R10: u32 = 96;
    pub(crate) const R11: u32 = 104;
    pub(crate) const R12: u32 = 112;
    pub(crate) const R13: u32 = 120;
    pub(crate) const R14: u32 = 128;
    pub(crate) const R15: u32 = 136;

    // Flags
    pub(crate) const CC_OP: u32 = 144;
    pub(crate) const CC_DEP1: u32 = 152;
    pub(crate) const CC_DEP2: u32 = 160;
    pub(crate) const CC_NDEP: u32 = 168;

    // Other
    pub(crate) const DFLAG: u32 = 176;
    pub(crate) const RIP: u32 = 184;
    pub(crate) const ACFLAG: u32 = 192;
    pub(crate) const IDFLAG: u32 = 200;

    // Segment registers. FS_CONST is the FS base address (archinfo offset 208);
    // GS_CONST is the GS base, which archinfo places at offset 1032 — well past
    // the XMM/FPU bank. The 216 slot belongs to SSEROUND, not GS_CONST — an
    // earlier layout wrongly defined GS_CONST there, aliasing the two.
    pub(crate) const FS_CONST: u32 = 208;

    // SSE control (per archinfo: 4B uint32_t padded to 8B in archinfo's table)
    pub(crate) const SSEROUND: u32 = 216;

    // XMM registers (128-bit each, 16 bytes) - offsets per archinfo
    pub(crate) const XMM0: u32 = 224;
    pub(crate) const XMM1: u32 = 256;
    pub(crate) const XMM2: u32 = 288;
    pub(crate) const XMM3: u32 = 320;
    pub(crate) const XMM4: u32 = 352;
    pub(crate) const XMM5: u32 = 384;
    pub(crate) const XMM6: u32 = 416;
    pub(crate) const XMM7: u32 = 448;
    pub(crate) const XMM8: u32 = 480;
    pub(crate) const XMM9: u32 = 512;
    pub(crate) const XMM10: u32 = 544;
    pub(crate) const XMM11: u32 = 576;
    pub(crate) const XMM12: u32 = 608;
    pub(crate) const XMM13: u32 = 640;
    pub(crate) const XMM14: u32 = 672;
    pub(crate) const XMM15: u32 = 704;

    // FPU state (per archinfo)
    pub(crate) const FTOP: u32 = 896;
    // VEX stores the x87 stack as `guest_FPREG[8]` of `ULong`, i.e. 8 x 8-byte
    // slots (64 bytes total, ending where FPTAG starts) -- MMX-style 64-bit
    // storage, not true 80-bit x87 extended precision. Per-index offset is
    // therefore `FPREG + n * 8`.
    pub(crate) const FPREG: u32 = 904;
    pub(crate) const FPTAG: u32 = 968;
    pub(crate) const FPROUND: u32 = 976;
    pub(crate) const FC3210: u32 = 984;

    // Late VEX state (per archinfo): the emulation-note word and the
    // self-modifying-code / redirect bookkeeping fields precede the real
    // GS_CONST slot at offset 1032. `guest_EMNOTE` is a UInt (4 bytes) with
    // 4 bytes of padding after it; the rest are ULongs. Named for parity with
    // X86/ARM/ARM64 (angr-9ke6b.10).
    pub(crate) const EMNOTE: u32 = 992;
    pub(crate) const CMSTART: u32 = 1000;
    pub(crate) const CMLEN: u32 = 1008;
    pub(crate) const NRADDR: u32 = 1016;
    pub(crate) const GS_CONST: u32 = 1032;
    pub(crate) const IP_AT_SYSCALL: u32 = 1040;

    // Total guest state size: archinfo's last register (ss_seg) ends at 1060.
    pub(crate) const GUEST_STATE_SIZE: usize = 1060;
}

// Canonical registers: each entry is `(name, offset, size_bytes)`. These
// drive `register_name(offset)` reverse lookups.
const CANONICAL: &[RegEntry] = &[
    // 64-bit GPRs
    ("rax", offsets::RAX, 8),
    ("rcx", offsets::RCX, 8),
    ("rdx", offsets::RDX, 8),
    ("rbx", offsets::RBX, 8),
    ("rsp", offsets::RSP, 8),
    ("rbp", offsets::RBP, 8),
    ("rsi", offsets::RSI, 8),
    ("rdi", offsets::RDI, 8),
    ("r8", offsets::R8, 8),
    ("r9", offsets::R9, 8),
    ("r10", offsets::R10, 8),
    ("r11", offsets::R11, 8),
    ("r12", offsets::R12, 8),
    ("r13", offsets::R13, 8),
    ("r14", offsets::R14, 8),
    ("r15", offsets::R15, 8),
    ("rip", offsets::RIP, 8),
    // Flags thunks (64-bit)
    ("cc_op", offsets::CC_OP, 8),
    ("cc_dep1", offsets::CC_DEP1, 8),
    ("cc_dep2", offsets::CC_DEP2, 8),
    ("cc_ndep", offsets::CC_NDEP, 8),
    ("dflag", offsets::DFLAG, 8),
    ("acflag", offsets::ACFLAG, 8),
    ("idflag", offsets::IDFLAG, 8),
    // Segments
    ("fs_const", offsets::FS_CONST, 8),
    ("gs_const", offsets::GS_CONST, 8),
    // SSE control register (8B per archinfo)
    ("sseround", offsets::SSEROUND, 8),
    // XMM registers (128-bit)
    ("xmm0", offsets::XMM0, 16),
    ("xmm1", offsets::XMM1, 16),
    ("xmm2", offsets::XMM2, 16),
    ("xmm3", offsets::XMM3, 16),
    ("xmm4", offsets::XMM4, 16),
    ("xmm5", offsets::XMM5, 16),
    ("xmm6", offsets::XMM6, 16),
    ("xmm7", offsets::XMM7, 16),
    ("xmm8", offsets::XMM8, 16),
    ("xmm9", offsets::XMM9, 16),
    ("xmm10", offsets::XMM10, 16),
    ("xmm11", offsets::XMM11, 16),
    ("xmm12", offsets::XMM12, 16),
    ("xmm13", offsets::XMM13, 16),
    ("xmm14", offsets::XMM14, 16),
    ("xmm15", offsets::XMM15, 16),
    // FPU
    ("fpreg", offsets::FPREG, 64),
    ("fptag", offsets::FPTAG, 8),
    ("fpround", offsets::FPROUND, 8),
    ("fc3210", offsets::FC3210, 8),
    // guest_FTOP is a UInt in VEX's amd64 guest state, not a ULong.
    ("ftop", offsets::FTOP, 4),
];

// Aliases: alternate names or sub-registers sharing a canonical offset.
// `register_offset` and `register_size` consult these as a fallback;
// `register_name` does not.
const ALIASES: &[RegEntry] = &[
    // Architecture-independent full-width aliases (match archinfo).
    ("sp", offsets::RSP, 8),
    ("bp", offsets::RBP, 8),
    // The two architecture-independent instruction-pointer spellings, matching
    // the ARM/ARM64/MIPS tables (angr-9ke6b.217) and archinfo's ArchAMD64
    // pc=(184,8)/ip=(184,8). `ip` is the *instruction pointer* on every arch,
    // never an ABI scratch register — see the angr-itm3u note in arch/arm.rs
    // (angr-690nc).
    ("pc", offsets::RIP, 8),
    ("ip", offsets::RIP, 8),
    // 32-bit sub-registers (low 32 of RAX etc.)
    ("eax", offsets::RAX, 4),
    ("ecx", offsets::RCX, 4),
    ("edx", offsets::RDX, 4),
    ("ebx", offsets::RBX, 4),
    ("esp", offsets::RSP, 4),
    ("ebp", offsets::RBP, 4),
    ("esi", offsets::RSI, 4),
    ("edi", offsets::RDI, 4),
    ("r8d", offsets::R8, 4),
    ("r9d", offsets::R9, 4),
    ("r10d", offsets::R10, 4),
    ("r11d", offsets::R11, 4),
    ("r12d", offsets::R12, 4),
    ("r13d", offsets::R13, 4),
    ("r14d", offsets::R14, 4),
    ("r15d", offsets::R15, 4),
    ("eip", offsets::RIP, 4),
    // 16-bit
    ("ax", offsets::RAX, 2),
    ("cx", offsets::RCX, 2),
    ("dx", offsets::RDX, 2),
    ("bx", offsets::RBX, 2),
    // NB: no 16-bit `sp`/`bp` entries. archinfo binds those two names to the
    // architecture-independent full-width stack/base pointer (AMD64 `sp` is
    // (48, 8)), so they live in the 64-bit block below; the legacy 16-bit
    // sub-registers are unreachable by name here, exactly as in archinfo.
    ("si", offsets::RSI, 2),
    ("di", offsets::RDI, 2),
    // 8-bit low
    ("al", offsets::RAX, 1),
    ("cl", offsets::RCX, 1),
    ("dl", offsets::RDX, 1),
    ("bl", offsets::RBX, 1),
    ("spl", offsets::RSP, 1),
    ("bpl", offsets::RBP, 1),
    ("sil", offsets::RSI, 1),
    ("dil", offsets::RDI, 1),
    ("r8b", offsets::R8, 1),
    ("r9b", offsets::R9, 1),
    ("r10b", offsets::R10, 1),
    ("r11b", offsets::R11, 1),
    ("r12b", offsets::R12, 1),
    ("r13b", offsets::R13, 1),
    ("r14b", offsets::R14, 1),
    ("r15b", offsets::R15, 1),
    // 8-bit high (offset+1)
    ("ah", offsets::RAX + 1, 1),
    ("ch", offsets::RCX + 1, 1),
    ("dh", offsets::RDX + 1, 1),
    ("bh", offsets::RBX + 1, 1),
    // Flag aliases
    ("d", offsets::DFLAG, 8),
    ("ac", offsets::ACFLAG, 8),
    ("id", offsets::IDFLAG, 8),
    // Segment aliases (the base registers, not the 16-bit selectors)
    ("fs", offsets::FS_CONST, 8),
    ("gs", offsets::GS_CONST, 8),
    // VEX bookkeeping tail (see the offsets note above). guest_EMNOTE is a
    // UInt; the rest are ULongs.
    ("emnote", offsets::EMNOTE, 4),
    ("cmstart", offsets::CMSTART, 8),
    ("cmlen", offsets::CMLEN, 8),
    ("nraddr", offsets::NRADDR, 8),
    ("ip_at_syscall", offsets::IP_AT_SYSCALL, 8),
];

// Registers exported to / imported from Python (see `Arch::register_names`).
// Deliberately omits `fpreg`: it is 64 bytes wide and the named-register
// channel is u128-capped, so it cannot round-trip here. The x87 control/status
// words below are all <= 8 bytes and do round-trip (angr-9ke6b.6).
const REGISTER_NAMES: &[&str] = &[
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13",
    "r14", "r15", "rip", "cc_op", "cc_dep1", "cc_dep2", "cc_ndep", "dflag", "acflag", "idflag",
    "fs_const", "gs_const", "sseround", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
    "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15", "fptag",
    "fpround", "fc3210", "ftop",
];

impl Arch for AMD64 {
    fn vex_arch(&self) -> VexArch {
        VexArch::AMD64
    }

    fn name(&self) -> &'static str {
        "AMD64"
    }

    fn bits(&self) -> u32 {
        64
    }

    fn state_size(&self) -> usize {
        offsets::GUEST_STATE_SIZE
    }

    fn ip_offset(&self) -> u32 {
        offsets::RIP
    }

    fn sp_offset(&self) -> u32 {
        offsets::RSP
    }

    fn bp_offset(&self) -> Option<u32> {
        Some(offsets::RBP)
    }

    impl_arch_registers!(CANONICAL, ALIASES, REGISTER_NAMES);

    fn syscall_num_offset(&self) -> Option<u32> {
        Some(offsets::RAX)
    }

    fn is_little_endian(&self) -> bool {
        true
    }
}

test_submod!("amd64_tests.rs" => tests);
