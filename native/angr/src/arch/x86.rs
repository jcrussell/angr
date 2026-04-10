//! x86 (32-bit) architecture definition.
//!
//! This module provides the register layout and architecture-specific
//! information for 32-bit x86.

use super::Arch;
use crate::vex::VexArch;

/// x86 (32-bit) architecture.
#[derive(Debug, Clone, Copy)]
pub struct X86;

// x86 VEX guest state offsets (from archinfo.ArchX86)
// These match pyvex's register layout for compatibility.
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

    fn register_offset(&self, name: &str) -> Option<u32> {
        let name_lower = name.to_lowercase();
        match name_lower.as_str() {
            // 32-bit registers
            "eax" => Some(offsets::EAX),
            "ecx" => Some(offsets::ECX),
            "edx" => Some(offsets::EDX),
            "ebx" => Some(offsets::EBX),
            "esp" => Some(offsets::ESP),
            "ebp" => Some(offsets::EBP),
            "esi" => Some(offsets::ESI),
            "edi" => Some(offsets::EDI),
            "eip" => Some(offsets::EIP),

            // 16-bit registers
            "ax" => Some(offsets::EAX),
            "cx" => Some(offsets::ECX),
            "dx" => Some(offsets::EDX),
            "bx" => Some(offsets::EBX),
            "sp" => Some(offsets::ESP),
            "bp" => Some(offsets::EBP),
            "si" => Some(offsets::ESI),
            "di" => Some(offsets::EDI),

            // 8-bit registers
            "al" => Some(offsets::EAX),
            "cl" => Some(offsets::ECX),
            "dl" => Some(offsets::EDX),
            "bl" => Some(offsets::EBX),
            "ah" => Some(offsets::EAX + 1),
            "ch" => Some(offsets::ECX + 1),
            "dh" => Some(offsets::EDX + 1),
            "bh" => Some(offsets::EBX + 1),

            // Flags thunks
            "cc_op" => Some(offsets::CC_OP),
            "cc_dep1" => Some(offsets::CC_DEP1),
            "cc_dep2" => Some(offsets::CC_DEP2),
            "cc_ndep" => Some(offsets::CC_NDEP),
            "d" | "dflag" => Some(offsets::DFLAG),
            "id" | "idflag" => Some(offsets::IDFLAG),
            "ac" | "acflag" => Some(offsets::ACFLAG),

            // Segments
            "cs" => Some(offsets::CS),
            "ds" => Some(offsets::DS),
            "es" => Some(offsets::ES),
            "fs" => Some(offsets::FS),
            "gs" => Some(offsets::GS),
            "ss" => Some(offsets::SS),
            "fs_const" => Some(offsets::FS_CONST),
            "gs_const" => Some(offsets::GS_CONST),

            // SSE
            "sseround" => Some(offsets::SSEROUND),
            "xmm0" => Some(offsets::XMM0),
            "xmm1" => Some(offsets::XMM1),
            "xmm2" => Some(offsets::XMM2),
            "xmm3" => Some(offsets::XMM3),
            "xmm4" => Some(offsets::XMM4),
            "xmm5" => Some(offsets::XMM5),
            "xmm6" => Some(offsets::XMM6),
            "xmm7" => Some(offsets::XMM7),

            // FPU
            "fpreg" => Some(offsets::FPREG),
            "fptag" => Some(offsets::FPTAG),
            "fpround" => Some(offsets::FPROUND),
            "fc3210" => Some(offsets::FC3210),
            "ftop" => Some(offsets::FTOP),

            _ => None,
        }
    }

    fn register_size(&self, name: &str) -> Option<u32> {
        let name_lower = name.to_lowercase();
        match name_lower.as_str() {
            // 32-bit registers
            "eax" | "ecx" | "edx" | "ebx" | "esp" | "ebp" | "esi" | "edi" | "eip" => Some(4),

            // 16-bit registers
            "ax" | "cx" | "dx" | "bx" | "sp" | "bp" | "si" | "di" => Some(2),

            // 8-bit registers
            "al" | "cl" | "dl" | "bl" | "ah" | "ch" | "dh" | "bh" => Some(1),

            // Flags thunks (32-bit)
            "cc_op" | "cc_dep1" | "cc_dep2" | "cc_ndep" | "d" | "dflag" | "id" | "idflag" | "ac" | "acflag" => Some(4),

            // Segments (16-bit selectors, 32-bit bases)
            "cs" | "ds" | "es" | "fs" | "gs" | "ss" => Some(2),
            "fs_const" | "gs_const" => Some(4),

            // SSE
            "sseround" => Some(4),
            "xmm0" | "xmm1" | "xmm2" | "xmm3" | "xmm4" | "xmm5" | "xmm6" | "xmm7" => Some(16),

            // FPU
            "fpreg" => Some(64),
            "fptag" | "fpround" | "fc3210" | "ftop" => Some(4),

            _ => None,
        }
    }

    fn register_name(&self, offset: u32) -> Option<&'static str> {
        match offset {
            offsets::EAX => Some("eax"),
            offsets::ECX => Some("ecx"),
            offsets::EDX => Some("edx"),
            offsets::EBX => Some("ebx"),
            offsets::ESP => Some("esp"),
            offsets::EBP => Some("ebp"),
            offsets::ESI => Some("esi"),
            offsets::EDI => Some("edi"),
            offsets::EIP => Some("eip"),
            offsets::CC_OP => Some("cc_op"),
            offsets::CC_DEP1 => Some("cc_dep1"),
            offsets::CC_DEP2 => Some("cc_dep2"),
            offsets::CC_NDEP => Some("cc_ndep"),
            offsets::DFLAG => Some("dflag"),
            offsets::IDFLAG => Some("idflag"),
            offsets::ACFLAG => Some("acflag"),
            _ => None,
        }
    }

    fn register_names(&self) -> &[&'static str] {
        &[
            "eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi", "eip",
            "cc_op", "cc_dep1", "cc_dep2", "cc_ndep", "dflag", "idflag", "acflag",
            "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7",
        ]
    }

    fn argument_registers(&self) -> &[u32] {
        // cdecl: arguments on stack, but we list potential register args
        &[]
    }

    fn return_register(&self) -> u32 {
        offsets::EAX
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
