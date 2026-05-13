//! AMD64 (x86-64) architecture definition.
//!
//! This module provides the register layout and architecture-specific
//! information for AMD64.

use super::{Arch, RegEntry, impl_arch_registers};
use crate::vex::VexArch;

/// AMD64 architecture.
#[derive(Debug, Clone, Copy)]
pub struct AMD64;

// AMD64 VEX guest state offsets (from VEX/pub/libvex_guest_amd64.h)
// These match the VexGuestAMD64State structure layout.
mod offsets {
    pub const RAX: u32 = 16;
    pub const RCX: u32 = 24;
    pub const RDX: u32 = 32;
    pub const RBX: u32 = 40;
    pub const RSP: u32 = 48;
    pub const RBP: u32 = 56;
    pub const RSI: u32 = 64;
    pub const RDI: u32 = 72;
    pub const R8: u32 = 80;
    pub const R9: u32 = 88;
    pub const R10: u32 = 96;
    pub const R11: u32 = 104;
    pub const R12: u32 = 112;
    pub const R13: u32 = 120;
    pub const R14: u32 = 128;
    pub const R15: u32 = 136;

    // Flags
    pub const CC_OP: u32 = 144;
    pub const CC_DEP1: u32 = 152;
    pub const CC_DEP2: u32 = 160;
    pub const CC_NDEP: u32 = 168;

    // Other
    pub const DFLAG: u32 = 176;
    pub const RIP: u32 = 184;
    pub const ACFLAG: u32 = 192;
    pub const IDFLAG: u32 = 200;

    // Segment registers
    pub const FS_CONST: u32 = 208;
    pub const GS_CONST: u32 = 216;

    // SSE control (per archinfo)
    pub const SSEROUND: u32 = 216;

    // XMM registers (128-bit each, 16 bytes) - offsets per archinfo
    pub const XMM0: u32 = 224;
    pub const XMM1: u32 = 256;
    pub const XMM2: u32 = 288;
    pub const XMM3: u32 = 320;
    pub const XMM4: u32 = 352;
    pub const XMM5: u32 = 384;
    pub const XMM6: u32 = 416;
    pub const XMM7: u32 = 448;
    pub const XMM8: u32 = 480;
    pub const XMM9: u32 = 512;
    pub const XMM10: u32 = 544;
    pub const XMM11: u32 = 576;
    pub const XMM12: u32 = 608;
    pub const XMM13: u32 = 640;
    pub const XMM14: u32 = 672;
    pub const XMM15: u32 = 704;

    // FPU state (per archinfo)
    pub const FTOP: u32 = 896;
    pub const FPREG: u32 = 904; // 8 x 80-bit FP registers
    pub const FPTAG: u32 = 968;
    pub const FPROUND: u32 = 976;
    pub const FC3210: u32 = 984;

    // Total guest state size
    pub const GUEST_STATE_SIZE: usize = 992;
}

// Canonical registers: each entry is `(name, offset, size_bytes)`. These
// drive `register_name(offset)` reverse lookups. SSEROUND is intentionally
// omitted because its offset (216) collides with GS_CONST.
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
    ("ftop", offsets::FTOP, 8),
];

// Aliases: alternate names or sub-registers sharing a canonical offset.
// `register_offset` and `register_size` consult these as a fallback;
// `register_name` does not.
const ALIASES: &[RegEntry] = &[
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
    ("sp", offsets::RSP, 2),
    ("bp", offsets::RBP, 2),
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
    // SSE rounding (collides with GS_CONST; lookup by name returns 216)
    ("sseround", offsets::SSEROUND, 8),
];

const REGISTER_NAMES: &[&str] = &[
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13",
    "r14", "r15", "rip", "cc_op", "cc_dep1", "cc_dep2", "cc_ndep", "dflag", "acflag", "idflag",
    "fs_const", "gs_const", "sseround", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
    "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15",
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

    fn argument_registers(&self) -> &[u32] {
        // System V AMD64 ABI: rdi, rsi, rdx, rcx, r8, r9
        &[
            offsets::RDI,
            offsets::RSI,
            offsets::RDX,
            offsets::RCX,
            offsets::R8,
            offsets::R9,
        ]
    }

    fn return_register(&self) -> u32 {
        offsets::RAX
    }

    fn syscall_num_offset(&self) -> Option<u32> {
        Some(offsets::RAX)
    }

    fn is_little_endian(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amd64_basics() {
        let arch = AMD64;

        assert_eq!(arch.bits(), 64);
        assert_eq!(arch.name(), "AMD64");
        assert!(arch.is_little_endian());
    }

    #[test]
    fn test_register_lookup() {
        let arch = AMD64;

        assert_eq!(arch.register_offset("rax"), Some(16));
        assert_eq!(arch.register_size("rax"), Some(8));

        assert_eq!(arch.register_offset("eax"), Some(16));
        assert_eq!(arch.register_size("eax"), Some(4));

        assert_eq!(arch.register_offset("al"), Some(16));
        assert_eq!(arch.register_size("al"), Some(1));

        assert_eq!(arch.register_offset("ah"), Some(17)); // High byte
        assert_eq!(arch.register_size("ah"), Some(1));
    }

    #[test]
    fn test_special_registers() {
        let arch = AMD64;

        assert_eq!(arch.ip_offset(), 184);
        assert_eq!(arch.sp_offset(), 48);
        assert_eq!(arch.bp_offset(), Some(56));
    }
}
