//! ARM (32-bit) architecture definition.
//!
//! This module provides the register layout and architecture-specific
//! information for 32-bit ARM.

use super::{Arch, RegEntry, impl_arch_registers};
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

// Canonical registers: drive `register_name(offset)` reverse lookups.
// ARM's register_name returns the rN form for r0..r12 and the conventional
// "sp"/"lr"/"pc" names for r13..r15t — the table mirrors that.
const CANONICAL: &[RegEntry] = &[
    ("r0", offsets::R0, 4),
    ("r1", offsets::R1, 4),
    ("r2", offsets::R2, 4),
    ("r3", offsets::R3, 4),
    ("r4", offsets::R4, 4),
    ("r5", offsets::R5, 4),
    ("r6", offsets::R6, 4),
    ("r7", offsets::R7, 4),
    ("r8", offsets::R8, 4),
    ("r9", offsets::R9, 4),
    ("r10", offsets::R10, 4),
    ("r11", offsets::R11, 4),
    ("r12", offsets::R12, 4),
    ("sp", offsets::R13, 4),
    ("lr", offsets::R14, 4),
    ("pc", offsets::R15T, 4),
    ("cc_op", offsets::CC_OP, 4),
    ("cc_dep1", offsets::CC_DEP1, 4),
    ("cc_dep2", offsets::CC_DEP2, 4),
    ("cc_ndep", offsets::CC_NDEP, 4),
];

const ALIASES: &[RegEntry] = &[
    // r11/r12/r13/r14/r15 alternate names
    ("fp", offsets::R11, 4),
    ("ip", offsets::R12, 4),
    ("r13", offsets::R13, 4),
    ("r14", offsets::R14, 4),
    ("r15", offsets::R15T, 4),
    ("r15t", offsets::R15T, 4),
    // Flags
    ("qflag32", offsets::QFLAG32, 4),
    ("geflag0", offsets::GEFLAG0, 4),
    ("geflag1", offsets::GEFLAG1, 4),
    ("geflag2", offsets::GEFLAG2, 4),
    ("geflag3", offsets::GEFLAG3, 4),
    // VFP/NEON D registers (64-bit)
    ("d0", offsets::D0, 8),
    ("d1", offsets::D1, 8),
    ("d2", offsets::D2, 8),
    ("d3", offsets::D3, 8),
    ("d4", offsets::D4, 8),
    ("d5", offsets::D5, 8),
    ("d6", offsets::D6, 8),
    ("d7", offsets::D7, 8),
    ("d8", offsets::D8, 8),
    ("d9", offsets::D9, 8),
    ("d10", offsets::D10, 8),
    ("d11", offsets::D11, 8),
    ("d12", offsets::D12, 8),
    ("d13", offsets::D13, 8),
    ("d14", offsets::D14, 8),
    ("d15", offsets::D15, 8),
    ("d16", offsets::D16, 8),
    ("d17", offsets::D17, 8),
    ("d18", offsets::D18, 8),
    ("d19", offsets::D19, 8),
    ("d20", offsets::D20, 8),
    ("d21", offsets::D21, 8),
    ("d22", offsets::D22, 8),
    ("d23", offsets::D23, 8),
    ("d24", offsets::D24, 8),
    ("d25", offsets::D25, 8),
    ("d26", offsets::D26, 8),
    ("d27", offsets::D27, 8),
    ("d28", offsets::D28, 8),
    ("d29", offsets::D29, 8),
    ("d30", offsets::D30, 8),
    ("d31", offsets::D31, 8),
    // Q registers (128-bit, overlap pairs of D registers)
    ("q0", offsets::D0, 16),
    ("q1", offsets::D2, 16),
    ("q2", offsets::D4, 16),
    ("q3", offsets::D6, 16),
    ("q4", offsets::D8, 16),
    ("q5", offsets::D10, 16),
    ("q6", offsets::D12, 16),
    ("q7", offsets::D14, 16),
    ("q8", offsets::D16, 16),
    ("q9", offsets::D18, 16),
    ("q10", offsets::D20, 16),
    ("q11", offsets::D22, 16),
    ("q12", offsets::D24, 16),
    ("q13", offsets::D26, 16),
    ("q14", offsets::D28, 16),
    ("q15", offsets::D30, 16),
    // Other
    ("fpscr", offsets::FPSCR, 4),
    ("tpidruro", offsets::TPIDRURO, 4),
    ("itstate", offsets::ITSTATE, 4),
];

const REGISTER_NAMES: &[&str] = &[
    "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "r12", "sp", "lr",
    "pc", "cc_op", "cc_dep1", "cc_dep2", "cc_ndep",
];

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

    impl_arch_registers!(CANONICAL, ALIASES, REGISTER_NAMES);

    fn argument_registers(&self) -> &[u32] {
        // AAPCS: r0-r3
        &[offsets::R0, offsets::R1, offsets::R2, offsets::R3]
    }

    fn return_register(&self) -> u32 {
        offsets::R0
    }

    fn syscall_num_offset(&self) -> Option<u32> {
        // EABI Linux syscall convention puts the syscall number in R7
        Some(offsets::R7)
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

    #[test]
    fn test_neon_q_and_d_registers() {
        // Smoke test for the NEON D/Q scaffolding (angr-bkcs.1).
        // ARM has D0..D31 (64-bit) and Q0..Q15 (128-bit) overlapping in
        // pairs of D registers: Q0=D0..D1, Q1=D2..D3, ..., Q15=D30..D31.
        let arch = ARM;

        // D registers are 8 bytes wide.
        assert_eq!(arch.register_offset("d0"), Some(112));
        assert_eq!(arch.register_size("d0"), Some(8));
        assert_eq!(arch.register_offset("d31"), Some(112 + 31 * 8));

        // Q registers are 16 bytes wide and start at the matching even
        // D register: Qn lives at offset of D(2n).
        for n in 0..16usize {
            let q = format!("q{}", n);
            let d_even = format!("d{}", 2 * n);
            assert_eq!(arch.register_offset(&q), arch.register_offset(&d_even));
            assert_eq!(arch.register_size(&q), Some(16));
        }
    }
}
