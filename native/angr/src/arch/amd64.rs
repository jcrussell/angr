//! AMD64 (x86-64) architecture definition.
//!
//! This module provides the register layout and architecture-specific
//! information for AMD64.

use super::Arch;
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

    fn register_offset(&self, name: &str) -> Option<u32> {
        let name_lower = name.to_lowercase();
        match name_lower.as_str() {
            // 64-bit registers
            "rax" => Some(offsets::RAX),
            "rcx" => Some(offsets::RCX),
            "rdx" => Some(offsets::RDX),
            "rbx" => Some(offsets::RBX),
            "rsp" => Some(offsets::RSP),
            "rbp" => Some(offsets::RBP),
            "rsi" => Some(offsets::RSI),
            "rdi" => Some(offsets::RDI),
            "r8" => Some(offsets::R8),
            "r9" => Some(offsets::R9),
            "r10" => Some(offsets::R10),
            "r11" => Some(offsets::R11),
            "r12" => Some(offsets::R12),
            "r13" => Some(offsets::R13),
            "r14" => Some(offsets::R14),
            "r15" => Some(offsets::R15),
            "rip" => Some(offsets::RIP),

            // 32-bit registers (same offsets, smaller size)
            "eax" => Some(offsets::RAX),
            "ecx" => Some(offsets::RCX),
            "edx" => Some(offsets::RDX),
            "ebx" => Some(offsets::RBX),
            "esp" => Some(offsets::RSP),
            "ebp" => Some(offsets::RBP),
            "esi" => Some(offsets::RSI),
            "edi" => Some(offsets::RDI),
            "r8d" => Some(offsets::R8),
            "r9d" => Some(offsets::R9),
            "r10d" => Some(offsets::R10),
            "r11d" => Some(offsets::R11),
            "r12d" => Some(offsets::R12),
            "r13d" => Some(offsets::R13),
            "r14d" => Some(offsets::R14),
            "r15d" => Some(offsets::R15),
            "eip" => Some(offsets::RIP),

            // 16-bit registers
            "ax" => Some(offsets::RAX),
            "cx" => Some(offsets::RCX),
            "dx" => Some(offsets::RDX),
            "bx" => Some(offsets::RBX),
            "sp" => Some(offsets::RSP),
            "bp" => Some(offsets::RBP),
            "si" => Some(offsets::RSI),
            "di" => Some(offsets::RDI),

            // 8-bit registers
            "al" => Some(offsets::RAX),
            "cl" => Some(offsets::RCX),
            "dl" => Some(offsets::RDX),
            "bl" => Some(offsets::RBX),
            "ah" => Some(offsets::RAX + 1),
            "ch" => Some(offsets::RCX + 1),
            "dh" => Some(offsets::RDX + 1),
            "bh" => Some(offsets::RBX + 1),
            "spl" => Some(offsets::RSP),
            "bpl" => Some(offsets::RBP),
            "sil" => Some(offsets::RSI),
            "dil" => Some(offsets::RDI),
            "r8b" => Some(offsets::R8),
            "r9b" => Some(offsets::R9),
            "r10b" => Some(offsets::R10),
            "r11b" => Some(offsets::R11),
            "r12b" => Some(offsets::R12),
            "r13b" => Some(offsets::R13),
            "r14b" => Some(offsets::R14),
            "r15b" => Some(offsets::R15),

            // Flags thunks
            "cc_op" => Some(offsets::CC_OP),
            "cc_dep1" => Some(offsets::CC_DEP1),
            "cc_dep2" => Some(offsets::CC_DEP2),
            "cc_ndep" => Some(offsets::CC_NDEP),
            "d" | "dflag" => Some(offsets::DFLAG),
            "ac" | "acflag" => Some(offsets::ACFLAG),
            "id" | "idflag" => Some(offsets::IDFLAG),

            // Segments
            "fs" | "fs_const" => Some(offsets::FS_CONST),
            "gs" | "gs_const" => Some(offsets::GS_CONST),

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
            "xmm8" => Some(offsets::XMM8),
            "xmm9" => Some(offsets::XMM9),
            "xmm10" => Some(offsets::XMM10),
            "xmm11" => Some(offsets::XMM11),
            "xmm12" => Some(offsets::XMM12),
            "xmm13" => Some(offsets::XMM13),
            "xmm14" => Some(offsets::XMM14),
            "xmm15" => Some(offsets::XMM15),

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
            // 64-bit registers
            "rax" | "rcx" | "rdx" | "rbx" | "rsp" | "rbp" | "rsi" | "rdi" | "r8" | "r9"
            | "r10" | "r11" | "r12" | "r13" | "r14" | "r15" | "rip" => Some(8),

            // 32-bit registers
            "eax" | "ecx" | "edx" | "ebx" | "esp" | "ebp" | "esi" | "edi" | "r8d" | "r9d"
            | "r10d" | "r11d" | "r12d" | "r13d" | "r14d" | "r15d" | "eip" => Some(4),

            // 16-bit registers
            "ax" | "cx" | "dx" | "bx" | "sp" | "bp" | "si" | "di" => Some(2),

            // 8-bit registers
            "al" | "cl" | "dl" | "bl" | "ah" | "ch" | "dh" | "bh" | "spl" | "bpl" | "sil"
            | "dil" | "r8b" | "r9b" | "r10b" | "r11b" | "r12b" | "r13b" | "r14b" | "r15b" => {
                Some(1)
            }

            // Flags thunks (64-bit)
            "cc_op" | "cc_dep1" | "cc_dep2" | "cc_ndep" | "d" | "dflag" | "ac" | "acflag" | "id" | "idflag" => Some(8),

            // Segments (64-bit)
            "fs" | "fs_const" | "gs" | "gs_const" => Some(8),

            // SSE (64-bit for rounding, 128-bit for XMM)
            "sseround" => Some(8),
            "xmm0" | "xmm1" | "xmm2" | "xmm3" | "xmm4" | "xmm5" | "xmm6" | "xmm7" | "xmm8"
            | "xmm9" | "xmm10" | "xmm11" | "xmm12" | "xmm13" | "xmm14" | "xmm15" => Some(16),

            // FPU
            "fpreg" => Some(64), // 8 x 64-bit (simplified)
            "fptag" | "fpround" | "fc3210" | "ftop" => Some(8),

            _ => None,
        }
    }

    fn register_name(&self, offset: u32) -> Option<&'static str> {
        match offset {
            offsets::RAX => Some("rax"),
            offsets::RCX => Some("rcx"),
            offsets::RDX => Some("rdx"),
            offsets::RBX => Some("rbx"),
            offsets::RSP => Some("rsp"),
            offsets::RBP => Some("rbp"),
            offsets::RSI => Some("rsi"),
            offsets::RDI => Some("rdi"),
            offsets::R8 => Some("r8"),
            offsets::R9 => Some("r9"),
            offsets::R10 => Some("r10"),
            offsets::R11 => Some("r11"),
            offsets::R12 => Some("r12"),
            offsets::R13 => Some("r13"),
            offsets::R14 => Some("r14"),
            offsets::R15 => Some("r15"),
            offsets::RIP => Some("rip"),
            offsets::CC_OP => Some("cc_op"),
            offsets::CC_DEP1 => Some("cc_dep1"),
            offsets::CC_DEP2 => Some("cc_dep2"),
            offsets::CC_NDEP => Some("cc_ndep"),
            offsets::DFLAG => Some("dflag"),
            offsets::ACFLAG => Some("acflag"),
            offsets::IDFLAG => Some("idflag"),
            offsets::FS_CONST => Some("fs_const"),
            offsets::GS_CONST => Some("gs_const"),
            offsets::SSEROUND => Some("sseround"),
            offsets::XMM0 => Some("xmm0"),
            offsets::XMM1 => Some("xmm1"),
            offsets::XMM2 => Some("xmm2"),
            offsets::XMM3 => Some("xmm3"),
            offsets::XMM4 => Some("xmm4"),
            offsets::XMM5 => Some("xmm5"),
            offsets::XMM6 => Some("xmm6"),
            offsets::XMM7 => Some("xmm7"),
            offsets::XMM8 => Some("xmm8"),
            offsets::XMM9 => Some("xmm9"),
            offsets::XMM10 => Some("xmm10"),
            offsets::XMM11 => Some("xmm11"),
            offsets::XMM12 => Some("xmm12"),
            offsets::XMM13 => Some("xmm13"),
            offsets::XMM14 => Some("xmm14"),
            offsets::XMM15 => Some("xmm15"),
            _ => None,
        }
    }

    fn register_names(&self) -> &[&'static str] {
        &[
            "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11",
            "r12", "r13", "r14", "r15", "rip", "cc_op", "cc_dep1", "cc_dep2", "cc_ndep", "dflag",
            "acflag", "idflag", "fs_const", "gs_const", "sseround", "xmm0", "xmm1", "xmm2",
            "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12",
            "xmm13", "xmm14", "xmm15",
        ]
    }

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
