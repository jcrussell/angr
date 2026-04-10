//! ARM (32-bit) architecture definition.
//!
//! This module provides the register layout and architecture-specific
//! information for 32-bit ARM.

use super::Arch;
use crate::vex::VexArch;

/// ARM (32-bit) architecture.
#[derive(Debug, Clone, Copy)]
pub struct ARM;

// ARM VEX guest state offsets (from VEX/pub/libvex_guest_arm.h)
// These match the VexGuestARMState structure layout.
mod offsets {
    // General purpose registers (R0-R15)
    pub const R0: u32 = 8;
    pub const R1: u32 = 12;
    pub const R2: u32 = 16;
    pub const R3: u32 = 20;
    pub const R4: u32 = 24;
    pub const R5: u32 = 28;
    pub const R6: u32 = 32;
    pub const R7: u32 = 36;
    pub const R8: u32 = 40;
    pub const R9: u32 = 44;
    pub const R10: u32 = 48;
    pub const R11: u32 = 52; // FP (frame pointer)
    pub const R12: u32 = 56; // IP (intra-procedure scratch)
    pub const R13: u32 = 60; // SP (stack pointer)
    pub const R14: u32 = 64; // LR (link register)
    pub const R15T: u32 = 68; // PC with Thumb bit

    // Condition code thunks
    pub const CC_OP: u32 = 72;
    pub const CC_DEP1: u32 = 76;
    pub const CC_DEP2: u32 = 80;
    pub const CC_NDEP: u32 = 84;

    // Thumb/ARM mode flag
    pub const QFLAG32: u32 = 88;
    pub const GEFLAG0: u32 = 92;
    pub const GEFLAG1: u32 = 96;
    pub const GEFLAG2: u32 = 100;
    pub const GEFLAG3: u32 = 104;

    // NEON/VFP registers (D0-D31, Q0-Q15)
    pub const D0: u32 = 112;
    pub const D1: u32 = 120;
    pub const D2: u32 = 128;
    pub const D3: u32 = 136;
    pub const D4: u32 = 144;
    pub const D5: u32 = 152;
    pub const D6: u32 = 160;
    pub const D7: u32 = 168;
    pub const D8: u32 = 176;
    pub const D9: u32 = 184;
    pub const D10: u32 = 192;
    pub const D11: u32 = 200;
    pub const D12: u32 = 208;
    pub const D13: u32 = 216;
    pub const D14: u32 = 224;
    pub const D15: u32 = 232;
    pub const D16: u32 = 240;
    pub const D17: u32 = 248;
    pub const D18: u32 = 256;
    pub const D19: u32 = 264;
    pub const D20: u32 = 272;
    pub const D21: u32 = 280;
    pub const D22: u32 = 288;
    pub const D23: u32 = 296;
    pub const D24: u32 = 304;
    pub const D25: u32 = 312;
    pub const D26: u32 = 320;
    pub const D27: u32 = 328;
    pub const D28: u32 = 336;
    pub const D29: u32 = 344;
    pub const D30: u32 = 352;
    pub const D31: u32 = 360;

    // FPSCR
    pub const FPSCR: u32 = 368;

    // TPIDRURO (thread pointer)
    pub const TPIDRURO: u32 = 372;

    // IT state for conditional execution
    pub const ITSTATE: u32 = 376;

    // Total guest state size
    pub const GUEST_STATE_SIZE: usize = 380;
}

impl Arch for ARM {
    fn vex_arch(&self) -> VexArch {
        VexArch::ARM
    }

    fn name(&self) -> &'static str {
        "ARM"
    }

    fn bits(&self) -> u32 {
        32
    }

    fn state_size(&self) -> usize {
        offsets::GUEST_STATE_SIZE
    }

    fn ip_offset(&self) -> u32 {
        offsets::R15T
    }

    fn sp_offset(&self) -> u32 {
        offsets::R13
    }

    fn bp_offset(&self) -> Option<u32> {
        Some(offsets::R11) // R11 is typically FP on ARM
    }

    fn register_offset(&self, name: &str) -> Option<u32> {
        let name_lower = name.to_lowercase();
        match name_lower.as_str() {
            // General purpose registers
            "r0" => Some(offsets::R0),
            "r1" => Some(offsets::R1),
            "r2" => Some(offsets::R2),
            "r3" => Some(offsets::R3),
            "r4" => Some(offsets::R4),
            "r5" => Some(offsets::R5),
            "r6" => Some(offsets::R6),
            "r7" => Some(offsets::R7),
            "r8" => Some(offsets::R8),
            "r9" => Some(offsets::R9),
            "r10" => Some(offsets::R10),
            "r11" | "fp" => Some(offsets::R11),
            "r12" | "ip" => Some(offsets::R12),
            "r13" | "sp" => Some(offsets::R13),
            "r14" | "lr" => Some(offsets::R14),
            "r15" | "pc" | "r15t" => Some(offsets::R15T),

            // Condition code thunks
            "cc_op" => Some(offsets::CC_OP),
            "cc_dep1" => Some(offsets::CC_DEP1),
            "cc_dep2" => Some(offsets::CC_DEP2),
            "cc_ndep" => Some(offsets::CC_NDEP),

            // Flags
            "qflag32" => Some(offsets::QFLAG32),
            "geflag0" => Some(offsets::GEFLAG0),
            "geflag1" => Some(offsets::GEFLAG1),
            "geflag2" => Some(offsets::GEFLAG2),
            "geflag3" => Some(offsets::GEFLAG3),

            // VFP/NEON D registers
            "d0" => Some(offsets::D0),
            "d1" => Some(offsets::D1),
            "d2" => Some(offsets::D2),
            "d3" => Some(offsets::D3),
            "d4" => Some(offsets::D4),
            "d5" => Some(offsets::D5),
            "d6" => Some(offsets::D6),
            "d7" => Some(offsets::D7),
            "d8" => Some(offsets::D8),
            "d9" => Some(offsets::D9),
            "d10" => Some(offsets::D10),
            "d11" => Some(offsets::D11),
            "d12" => Some(offsets::D12),
            "d13" => Some(offsets::D13),
            "d14" => Some(offsets::D14),
            "d15" => Some(offsets::D15),
            "d16" => Some(offsets::D16),
            "d17" => Some(offsets::D17),
            "d18" => Some(offsets::D18),
            "d19" => Some(offsets::D19),
            "d20" => Some(offsets::D20),
            "d21" => Some(offsets::D21),
            "d22" => Some(offsets::D22),
            "d23" => Some(offsets::D23),
            "d24" => Some(offsets::D24),
            "d25" => Some(offsets::D25),
            "d26" => Some(offsets::D26),
            "d27" => Some(offsets::D27),
            "d28" => Some(offsets::D28),
            "d29" => Some(offsets::D29),
            "d30" => Some(offsets::D30),
            "d31" => Some(offsets::D31),

            // Q registers (128-bit, overlapping D registers)
            "q0" => Some(offsets::D0),
            "q1" => Some(offsets::D2),
            "q2" => Some(offsets::D4),
            "q3" => Some(offsets::D6),
            "q4" => Some(offsets::D8),
            "q5" => Some(offsets::D10),
            "q6" => Some(offsets::D12),
            "q7" => Some(offsets::D14),
            "q8" => Some(offsets::D16),
            "q9" => Some(offsets::D18),
            "q10" => Some(offsets::D20),
            "q11" => Some(offsets::D22),
            "q12" => Some(offsets::D24),
            "q13" => Some(offsets::D26),
            "q14" => Some(offsets::D28),
            "q15" => Some(offsets::D30),

            // Other
            "fpscr" => Some(offsets::FPSCR),
            "tpidruro" => Some(offsets::TPIDRURO),
            "itstate" => Some(offsets::ITSTATE),

            _ => None,
        }
    }

    fn register_size(&self, name: &str) -> Option<u32> {
        let name_lower = name.to_lowercase();
        match name_lower.as_str() {
            // 32-bit general purpose registers
            "r0" | "r1" | "r2" | "r3" | "r4" | "r5" | "r6" | "r7" | "r8" | "r9" | "r10"
            | "r11" | "fp" | "r12" | "ip" | "r13" | "sp" | "r14" | "lr" | "r15" | "pc"
            | "r15t" => Some(4),

            // CC thunks (32-bit)
            "cc_op" | "cc_dep1" | "cc_dep2" | "cc_ndep" => Some(4),

            // Flags (32-bit)
            "qflag32" | "geflag0" | "geflag1" | "geflag2" | "geflag3" => Some(4),

            // D registers (64-bit)
            "d0" | "d1" | "d2" | "d3" | "d4" | "d5" | "d6" | "d7" | "d8" | "d9" | "d10"
            | "d11" | "d12" | "d13" | "d14" | "d15" | "d16" | "d17" | "d18" | "d19" | "d20"
            | "d21" | "d22" | "d23" | "d24" | "d25" | "d26" | "d27" | "d28" | "d29" | "d30"
            | "d31" => Some(8),

            // Q registers (128-bit)
            "q0" | "q1" | "q2" | "q3" | "q4" | "q5" | "q6" | "q7" | "q8" | "q9" | "q10"
            | "q11" | "q12" | "q13" | "q14" | "q15" => Some(16),

            // Other (32-bit)
            "fpscr" | "tpidruro" | "itstate" => Some(4),

            _ => None,
        }
    }

    fn register_name(&self, offset: u32) -> Option<&'static str> {
        match offset {
            offsets::R0 => Some("r0"),
            offsets::R1 => Some("r1"),
            offsets::R2 => Some("r2"),
            offsets::R3 => Some("r3"),
            offsets::R4 => Some("r4"),
            offsets::R5 => Some("r5"),
            offsets::R6 => Some("r6"),
            offsets::R7 => Some("r7"),
            offsets::R8 => Some("r8"),
            offsets::R9 => Some("r9"),
            offsets::R10 => Some("r10"),
            offsets::R11 => Some("r11"),
            offsets::R12 => Some("r12"),
            offsets::R13 => Some("sp"),
            offsets::R14 => Some("lr"),
            offsets::R15T => Some("pc"),
            offsets::CC_OP => Some("cc_op"),
            offsets::CC_DEP1 => Some("cc_dep1"),
            offsets::CC_DEP2 => Some("cc_dep2"),
            offsets::CC_NDEP => Some("cc_ndep"),
            _ => None,
        }
    }

    fn register_names(&self) -> &[&'static str] {
        &[
            "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "r12",
            "sp", "lr", "pc", "cc_op", "cc_dep1", "cc_dep2", "cc_ndep",
        ]
    }

    fn argument_registers(&self) -> &[u32] {
        // AAPCS: r0-r3
        &[offsets::R0, offsets::R1, offsets::R2, offsets::R3]
    }

    fn return_register(&self) -> u32 {
        offsets::R0
    }

    fn is_little_endian(&self) -> bool {
        true // ARM can be big or little endian, but little is more common
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arm_basics() {
        let arch = ARM;

        assert_eq!(arch.bits(), 32);
        assert_eq!(arch.name(), "ARM");
        assert!(arch.is_little_endian());
    }

    #[test]
    fn test_register_lookup() {
        let arch = ARM;

        assert_eq!(arch.register_offset("r0"), Some(8));
        assert_eq!(arch.register_size("r0"), Some(4));

        assert_eq!(arch.register_offset("sp"), Some(60));
        assert_eq!(arch.register_offset("r13"), Some(60));

        assert_eq!(arch.register_offset("lr"), Some(64));
        assert_eq!(arch.register_offset("r14"), Some(64));

        assert_eq!(arch.register_offset("pc"), Some(68));
    }

    #[test]
    fn test_special_registers() {
        let arch = ARM;

        assert_eq!(arch.ip_offset(), 68); // PC
        assert_eq!(arch.sp_offset(), 60); // SP
        assert_eq!(arch.bp_offset(), Some(52)); // R11/FP
    }
}
