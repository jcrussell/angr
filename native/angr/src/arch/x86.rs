//! x86 (32-bit) architecture definition.
//!
//! This module provides the register layout and architecture-specific
//! information for 32-bit x86.

use super::{Arch, RegEntry, impl_arch_registers};
use crate::vex::VexArch;

/// x86 (32-bit) architecture.
#[derive(Debug, Clone, Copy)]
pub struct X86;

// x86 VEX guest state offsets (from archinfo.ArchX86)
// These match pyvex's register layout for compatibility.
#[allow(dead_code)]
mod offsets {
    // General purpose registers
    pub const EAX: u32 = 8;
    pub const ECX: u32 = 12;
    pub const EDX: u32 = 16;
    pub const EBX: u32 = 20;
    pub const ESP: u32 = 24;
    pub const EBP: u32 = 28;
    pub const ESI: u32 = 32;
    pub const EDI: u32 = 36;

    // Flags thunks
    pub const CC_OP: u32 = 40;
    pub const CC_DEP1: u32 = 44;
    pub const CC_DEP2: u32 = 48;
    pub const CC_NDEP: u32 = 52;

    // Other flags
    pub const DFLAG: u32 = 56;
    pub const IDFLAG: u32 = 60;
    pub const ACFLAG: u32 = 64;
    pub const EIP: u32 = 68;

    // FPU registers (before SSE in archinfo layout)
    pub const FPREG: u32 = 72;
    pub const FPTAG: u32 = 136;
    pub const FPROUND: u32 = 144;
    pub const FC3210: u32 = 148;
    pub const FTOP: u32 = 152;

    // SSE
    pub const SSEROUND: u32 = 156;
    pub const XMM0: u32 = 160;
    pub const XMM1: u32 = 176;
    pub const XMM2: u32 = 192;
    pub const XMM3: u32 = 208;
    pub const XMM4: u32 = 224;
    pub const XMM5: u32 = 240;
    pub const XMM6: u32 = 256;
    pub const XMM7: u32 = 272;

    // Segment selectors (after XMM registers)
    pub const CS: u32 = 288;
    pub const DS: u32 = 290;
    pub const ES: u32 = 292;
    pub const FS: u32 = 294;
    pub const GS: u32 = 296;
    pub const SS: u32 = 298;

    // Segment base addresses
    pub const LDT: u32 = 304;
    pub const GDT: u32 = 312;
    // Note: fs_const/gs_const not in archinfo for x86, using placeholders
    pub const FS_CONST: u32 = 320;
    pub const GS_CONST: u32 = 324;

    // Total guest state size (must cover all registers)
    pub const GUEST_STATE_SIZE: usize = 344;
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
];

// Aliases: alternate names, sub-registers, segments, and SIMD registers.
// `register_offset` and `register_size` consult these as a fallback;
// `register_name` does not.
const ALIASES: &[RegEntry] = &[
    // 16-bit
    ("ax", offsets::EAX, 2),
    ("cx", offsets::ECX, 2),
    ("dx", offsets::EDX, 2),
    ("bx", offsets::EBX, 2),
    ("sp", offsets::ESP, 2),
    ("bp", offsets::EBP, 2),
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
    ("fs_const", offsets::FS_CONST, 4),
    ("gs_const", offsets::GS_CONST, 4),
    // SSE
    ("sseround", offsets::SSEROUND, 4),
    ("xmm0", offsets::XMM0, 16),
    ("xmm1", offsets::XMM1, 16),
    ("xmm2", offsets::XMM2, 16),
    ("xmm3", offsets::XMM3, 16),
    ("xmm4", offsets::XMM4, 16),
    ("xmm5", offsets::XMM5, 16),
    ("xmm6", offsets::XMM6, 16),
    ("xmm7", offsets::XMM7, 16),
    // FPU
    ("fpreg", offsets::FPREG, 64),
    ("fptag", offsets::FPTAG, 4),
    ("fpround", offsets::FPROUND, 4),
    ("fc3210", offsets::FC3210, 4),
    ("ftop", offsets::FTOP, 4),
];

const REGISTER_NAMES: &[&str] = &[
    "eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi", "eip", "cc_op", "cc_dep1", "cc_dep2",
    "cc_ndep", "dflag", "idflag", "acflag", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
    "xmm7",
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

    fn argument_registers(&self) -> &[u32] {
        // cdecl: arguments on stack, but we list potential register args
        &[]
    }

    fn return_register(&self) -> u32 {
        offsets::EAX
    }

    fn syscall_num_offset(&self) -> Option<u32> {
        Some(offsets::EAX)
    }

    fn is_little_endian(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_x86_basics() {
        let arch = X86;

        assert_eq!(arch.bits(), 32);
        assert_eq!(arch.name(), "X86");
        assert!(arch.is_little_endian());
    }

    #[test]
    fn test_register_lookup() {
        let arch = X86;

        assert_eq!(arch.register_offset("eax"), Some(8));
        assert_eq!(arch.register_size("eax"), Some(4));

        assert_eq!(arch.register_offset("ax"), Some(8));
        assert_eq!(arch.register_size("ax"), Some(2));

        assert_eq!(arch.register_offset("al"), Some(8));
        assert_eq!(arch.register_size("al"), Some(1));
    }

    #[test]
    fn test_special_registers() {
        let arch = X86;

        assert_eq!(arch.ip_offset(), 68);
        assert_eq!(arch.sp_offset(), 24);
        assert_eq!(arch.bp_offset(), Some(28));
    }
}
