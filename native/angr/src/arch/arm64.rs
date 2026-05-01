//! ARM64 (AArch64) architecture definition.
//!
//! This module provides the register layout and architecture-specific
//! information for 64-bit ARM.

use super::Arch;
use crate::vex::VexArch;

/// ARM64 (AArch64) architecture.
#[derive(Debug, Clone, Copy)]
pub struct ARM64;

// ARM64 VEX guest state offsets (from VEX/pub/libvex_guest_arm64.h)
// These match the VexGuestARM64State structure layout.
mod offsets {
    // General purpose registers (X0-X30)
    pub const X0: u32 = 16;
    pub const X1: u32 = 24;
    pub const X2: u32 = 32;
    pub const X3: u32 = 40;
    pub const X4: u32 = 48;
    pub const X5: u32 = 56;
    pub const X6: u32 = 64;
    pub const X7: u32 = 72;
    pub const X8: u32 = 80;
    pub const X9: u32 = 88;
    pub const X10: u32 = 96;
    pub const X11: u32 = 104;
    pub const X12: u32 = 112;
    pub const X13: u32 = 120;
    pub const X14: u32 = 128;
    pub const X15: u32 = 136;
    pub const X16: u32 = 144;
    pub const X17: u32 = 152;
    pub const X18: u32 = 160;
    pub const X19: u32 = 168;
    pub const X20: u32 = 176;
    pub const X21: u32 = 184;
    pub const X22: u32 = 192;
    pub const X23: u32 = 200;
    pub const X24: u32 = 208;
    pub const X25: u32 = 216;
    pub const X26: u32 = 224;
    pub const X27: u32 = 232;
    pub const X28: u32 = 240;
    pub const X29: u32 = 248; // FP (frame pointer)
    pub const X30: u32 = 256; // LR (link register)

    // Stack pointer (XSP)
    pub const XSP: u32 = 264;

    // Program counter
    pub const PC: u32 = 272;

    // Condition code thunks
    pub const CC_OP: u32 = 280;
    pub const CC_DEP1: u32 = 288;
    pub const CC_DEP2: u32 = 296;
    pub const CC_NDEP: u32 = 304;

    // Thread pointer
    pub const TPIDR_EL0: u32 = 312;

    // NEON/SIMD registers (V0-V31, 128-bit each)
    pub const Q0: u32 = 320;
    pub const Q1: u32 = 336;
    pub const Q2: u32 = 352;
    pub const Q3: u32 = 368;
    pub const Q4: u32 = 384;
    pub const Q5: u32 = 400;
    pub const Q6: u32 = 416;
    pub const Q7: u32 = 432;
    pub const Q8: u32 = 448;
    pub const Q9: u32 = 464;
    pub const Q10: u32 = 480;
    pub const Q11: u32 = 496;
    pub const Q12: u32 = 512;
    pub const Q13: u32 = 528;
    pub const Q14: u32 = 544;
    pub const Q15: u32 = 560;
    pub const Q16: u32 = 576;
    pub const Q17: u32 = 592;
    pub const Q18: u32 = 608;
    pub const Q19: u32 = 624;
    pub const Q20: u32 = 640;
    pub const Q21: u32 = 656;
    pub const Q22: u32 = 672;
    pub const Q23: u32 = 688;
    pub const Q24: u32 = 704;
    pub const Q25: u32 = 720;
    pub const Q26: u32 = 736;
    pub const Q27: u32 = 752;
    pub const Q28: u32 = 768;
    pub const Q29: u32 = 784;
    pub const Q30: u32 = 800;
    pub const Q31: u32 = 816;

    // FPCR/FPSR
    pub const QCFLAG: u32 = 832;
    pub const FPCR: u32 = 836;

    // Total guest state size
    pub const GUEST_STATE_SIZE: usize = 848;
}

impl Arch for ARM64 {
    fn vex_arch(&self) -> VexArch {
        VexArch::ARM64
    }

    fn name(&self) -> &'static str {
        "ARM64"
    }

    fn bits(&self) -> u32 {
        64
    }

    fn state_size(&self) -> usize {
        offsets::GUEST_STATE_SIZE
    }

    fn ip_offset(&self) -> u32 {
        offsets::PC
    }

    fn sp_offset(&self) -> u32 {
        offsets::XSP
    }

    fn bp_offset(&self) -> Option<u32> {
        Some(offsets::X29) // X29 is FP on ARM64
    }

    fn register_offset(&self, name: &str) -> Option<u32> {
        let name_lower = name.to_lowercase();
        match name_lower.as_str() {
            // 64-bit X registers
            "x0" => Some(offsets::X0),
            "x1" => Some(offsets::X1),
            "x2" => Some(offsets::X2),
            "x3" => Some(offsets::X3),
            "x4" => Some(offsets::X4),
            "x5" => Some(offsets::X5),
            "x6" => Some(offsets::X6),
            "x7" => Some(offsets::X7),
            "x8" => Some(offsets::X8),
            "x9" => Some(offsets::X9),
            "x10" => Some(offsets::X10),
            "x11" => Some(offsets::X11),
            "x12" => Some(offsets::X12),
            "x13" => Some(offsets::X13),
            "x14" => Some(offsets::X14),
            "x15" => Some(offsets::X15),
            "x16" => Some(offsets::X16),
            "x17" => Some(offsets::X17),
            "x18" => Some(offsets::X18),
            "x19" => Some(offsets::X19),
            "x20" => Some(offsets::X20),
            "x21" => Some(offsets::X21),
            "x22" => Some(offsets::X22),
            "x23" => Some(offsets::X23),
            "x24" => Some(offsets::X24),
            "x25" => Some(offsets::X25),
            "x26" => Some(offsets::X26),
            "x27" => Some(offsets::X27),
            "x28" => Some(offsets::X28),
            "x29" | "fp" => Some(offsets::X29),
            "x30" | "lr" => Some(offsets::X30),

            // 32-bit W registers (lower 32 bits of X registers)
            "w0" => Some(offsets::X0),
            "w1" => Some(offsets::X1),
            "w2" => Some(offsets::X2),
            "w3" => Some(offsets::X3),
            "w4" => Some(offsets::X4),
            "w5" => Some(offsets::X5),
            "w6" => Some(offsets::X6),
            "w7" => Some(offsets::X7),
            "w8" => Some(offsets::X8),
            "w9" => Some(offsets::X9),
            "w10" => Some(offsets::X10),
            "w11" => Some(offsets::X11),
            "w12" => Some(offsets::X12),
            "w13" => Some(offsets::X13),
            "w14" => Some(offsets::X14),
            "w15" => Some(offsets::X15),
            "w16" => Some(offsets::X16),
            "w17" => Some(offsets::X17),
            "w18" => Some(offsets::X18),
            "w19" => Some(offsets::X19),
            "w20" => Some(offsets::X20),
            "w21" => Some(offsets::X21),
            "w22" => Some(offsets::X22),
            "w23" => Some(offsets::X23),
            "w24" => Some(offsets::X24),
            "w25" => Some(offsets::X25),
            "w26" => Some(offsets::X26),
            "w27" => Some(offsets::X27),
            "w28" => Some(offsets::X28),
            "w29" => Some(offsets::X29),
            "w30" => Some(offsets::X30),

            // Stack pointer and PC
            "sp" | "xsp" => Some(offsets::XSP),
            "pc" => Some(offsets::PC),

            // Condition code thunks
            "cc_op" => Some(offsets::CC_OP),
            "cc_dep1" => Some(offsets::CC_DEP1),
            "cc_dep2" => Some(offsets::CC_DEP2),
            "cc_ndep" => Some(offsets::CC_NDEP),

            // Thread pointer
            "tpidr_el0" => Some(offsets::TPIDR_EL0),

            // SIMD/NEON Q registers (128-bit)
            "q0" | "v0" => Some(offsets::Q0),
            "q1" | "v1" => Some(offsets::Q1),
            "q2" | "v2" => Some(offsets::Q2),
            "q3" | "v3" => Some(offsets::Q3),
            "q4" | "v4" => Some(offsets::Q4),
            "q5" | "v5" => Some(offsets::Q5),
            "q6" | "v6" => Some(offsets::Q6),
            "q7" | "v7" => Some(offsets::Q7),
            "q8" | "v8" => Some(offsets::Q8),
            "q9" | "v9" => Some(offsets::Q9),
            "q10" | "v10" => Some(offsets::Q10),
            "q11" | "v11" => Some(offsets::Q11),
            "q12" | "v12" => Some(offsets::Q12),
            "q13" | "v13" => Some(offsets::Q13),
            "q14" | "v14" => Some(offsets::Q14),
            "q15" | "v15" => Some(offsets::Q15),
            "q16" | "v16" => Some(offsets::Q16),
            "q17" | "v17" => Some(offsets::Q17),
            "q18" | "v18" => Some(offsets::Q18),
            "q19" | "v19" => Some(offsets::Q19),
            "q20" | "v20" => Some(offsets::Q20),
            "q21" | "v21" => Some(offsets::Q21),
            "q22" | "v22" => Some(offsets::Q22),
            "q23" | "v23" => Some(offsets::Q23),
            "q24" | "v24" => Some(offsets::Q24),
            "q25" | "v25" => Some(offsets::Q25),
            "q26" | "v26" => Some(offsets::Q26),
            "q27" | "v27" => Some(offsets::Q27),
            "q28" | "v28" => Some(offsets::Q28),
            "q29" | "v29" => Some(offsets::Q29),
            "q30" | "v30" => Some(offsets::Q30),
            "q31" | "v31" => Some(offsets::Q31),

            // D registers (lower 64-bit of Q registers)
            "d0" => Some(offsets::Q0),
            "d1" => Some(offsets::Q1),
            "d2" => Some(offsets::Q2),
            "d3" => Some(offsets::Q3),
            "d4" => Some(offsets::Q4),
            "d5" => Some(offsets::Q5),
            "d6" => Some(offsets::Q6),
            "d7" => Some(offsets::Q7),
            "d8" => Some(offsets::Q8),
            "d9" => Some(offsets::Q9),
            "d10" => Some(offsets::Q10),
            "d11" => Some(offsets::Q11),
            "d12" => Some(offsets::Q12),
            "d13" => Some(offsets::Q13),
            "d14" => Some(offsets::Q14),
            "d15" => Some(offsets::Q15),
            "d16" => Some(offsets::Q16),
            "d17" => Some(offsets::Q17),
            "d18" => Some(offsets::Q18),
            "d19" => Some(offsets::Q19),
            "d20" => Some(offsets::Q20),
            "d21" => Some(offsets::Q21),
            "d22" => Some(offsets::Q22),
            "d23" => Some(offsets::Q23),
            "d24" => Some(offsets::Q24),
            "d25" => Some(offsets::Q25),
            "d26" => Some(offsets::Q26),
            "d27" => Some(offsets::Q27),
            "d28" => Some(offsets::Q28),
            "d29" => Some(offsets::Q29),
            "d30" => Some(offsets::Q30),
            "d31" => Some(offsets::Q31),

            // FPCR
            "qcflag" => Some(offsets::QCFLAG),
            "fpcr" => Some(offsets::FPCR),

            _ => None,
        }
    }

    fn register_size(&self, name: &str) -> Option<u32> {
        let name_lower = name.to_lowercase();
        match name_lower.as_str() {
            // 64-bit X registers
            "x0" | "x1" | "x2" | "x3" | "x4" | "x5" | "x6" | "x7" | "x8" | "x9" | "x10"
            | "x11" | "x12" | "x13" | "x14" | "x15" | "x16" | "x17" | "x18" | "x19" | "x20"
            | "x21" | "x22" | "x23" | "x24" | "x25" | "x26" | "x27" | "x28" | "x29" | "fp"
            | "x30" | "lr" | "sp" | "xsp" | "pc" => Some(8),

            // 32-bit W registers
            "w0" | "w1" | "w2" | "w3" | "w4" | "w5" | "w6" | "w7" | "w8" | "w9" | "w10"
            | "w11" | "w12" | "w13" | "w14" | "w15" | "w16" | "w17" | "w18" | "w19" | "w20"
            | "w21" | "w22" | "w23" | "w24" | "w25" | "w26" | "w27" | "w28" | "w29" | "w30" => {
                Some(4)
            }

            // CC thunks (64-bit)
            "cc_op" | "cc_dep1" | "cc_dep2" | "cc_ndep" | "tpidr_el0" => Some(8),

            // Q/V registers (128-bit)
            "q0" | "v0" | "q1" | "v1" | "q2" | "v2" | "q3" | "v3" | "q4" | "v4" | "q5" | "v5"
            | "q6" | "v6" | "q7" | "v7" | "q8" | "v8" | "q9" | "v9" | "q10" | "v10" | "q11"
            | "v11" | "q12" | "v12" | "q13" | "v13" | "q14" | "v14" | "q15" | "v15" | "q16"
            | "v16" | "q17" | "v17" | "q18" | "v18" | "q19" | "v19" | "q20" | "v20" | "q21"
            | "v21" | "q22" | "v22" | "q23" | "v23" | "q24" | "v24" | "q25" | "v25" | "q26"
            | "v26" | "q27" | "v27" | "q28" | "v28" | "q29" | "v29" | "q30" | "v30" | "q31"
            | "v31" => Some(16),

            // D registers (64-bit)
            "d0" | "d1" | "d2" | "d3" | "d4" | "d5" | "d6" | "d7" | "d8" | "d9" | "d10"
            | "d11" | "d12" | "d13" | "d14" | "d15" | "d16" | "d17" | "d18" | "d19" | "d20"
            | "d21" | "d22" | "d23" | "d24" | "d25" | "d26" | "d27" | "d28" | "d29" | "d30"
            | "d31" => Some(8),

            // FPCR (32-bit)
            "qcflag" | "fpcr" => Some(4),

            _ => None,
        }
    }

    fn register_name(&self, offset: u32) -> Option<&'static str> {
        match offset {
            offsets::X0 => Some("x0"),
            offsets::X1 => Some("x1"),
            offsets::X2 => Some("x2"),
            offsets::X3 => Some("x3"),
            offsets::X4 => Some("x4"),
            offsets::X5 => Some("x5"),
            offsets::X6 => Some("x6"),
            offsets::X7 => Some("x7"),
            offsets::X8 => Some("x8"),
            offsets::X9 => Some("x9"),
            offsets::X10 => Some("x10"),
            offsets::X11 => Some("x11"),
            offsets::X12 => Some("x12"),
            offsets::X13 => Some("x13"),
            offsets::X14 => Some("x14"),
            offsets::X15 => Some("x15"),
            offsets::X16 => Some("x16"),
            offsets::X17 => Some("x17"),
            offsets::X18 => Some("x18"),
            offsets::X19 => Some("x19"),
            offsets::X20 => Some("x20"),
            offsets::X21 => Some("x21"),
            offsets::X22 => Some("x22"),
            offsets::X23 => Some("x23"),
            offsets::X24 => Some("x24"),
            offsets::X25 => Some("x25"),
            offsets::X26 => Some("x26"),
            offsets::X27 => Some("x27"),
            offsets::X28 => Some("x28"),
            offsets::X29 => Some("fp"),
            offsets::X30 => Some("lr"),
            offsets::XSP => Some("sp"),
            offsets::PC => Some("pc"),
            _ => None,
        }
    }

    fn register_names(&self) -> &[&'static str] {
        &[
            "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12",
            "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24",
            "x25", "x26", "x27", "x28", "fp", "lr", "sp", "pc",
        ]
    }

    fn argument_registers(&self) -> &[u32] {
        // AAPCS64: x0-x7
        &[
            offsets::X0,
            offsets::X1,
            offsets::X2,
            offsets::X3,
            offsets::X4,
            offsets::X5,
            offsets::X6,
            offsets::X7,
        ]
    }

    fn return_register(&self) -> u32 {
        offsets::X0
    }

    fn syscall_num_offset(&self) -> Option<u32> {
        // Linux AArch64 syscall convention puts the syscall number in X8
        Some(offsets::X8)
    }

    fn is_little_endian(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arm64_basics() {
        let arch = ARM64;

        assert_eq!(arch.bits(), 64);
        assert_eq!(arch.name(), "ARM64");
        assert!(arch.is_little_endian());
    }

    #[test]
    fn test_register_lookup() {
        let arch = ARM64;

        assert_eq!(arch.register_offset("x0"), Some(16));
        assert_eq!(arch.register_size("x0"), Some(8));

        assert_eq!(arch.register_offset("w0"), Some(16));
        assert_eq!(arch.register_size("w0"), Some(4));

        assert_eq!(arch.register_offset("sp"), Some(264));
        assert_eq!(arch.register_offset("xsp"), Some(264));

        assert_eq!(arch.register_offset("lr"), Some(256));
        assert_eq!(arch.register_offset("x30"), Some(256));

        assert_eq!(arch.register_offset("fp"), Some(248));
        assert_eq!(arch.register_offset("x29"), Some(248));
    }

    #[test]
    fn test_special_registers() {
        let arch = ARM64;

        assert_eq!(arch.ip_offset(), 272); // PC
        assert_eq!(arch.sp_offset(), 264); // SP
        assert_eq!(arch.bp_offset(), Some(248)); // FP (X29)
    }
}
