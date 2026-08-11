//! ARM64 (AArch64) architecture definition.
//!
//! This module provides the register layout and architecture-specific
//! information for 64-bit ARM.

use super::{Arch, RegEntry, impl_arch_registers};
use crate::vex::VexArch;

/// ARM64 (AArch64) architecture.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ARM64;

// ARM64 VEX guest state offsets (from VEX/pub/libvex_guest_arm64.h)
// These match the VexGuestARM64State structure layout.
pub(crate) mod offsets {
    // General purpose registers (X0-X30)
    pub(crate) const X0: u32 = 16;
    pub(crate) const X1: u32 = 24;
    pub(crate) const X2: u32 = 32;
    pub(crate) const X3: u32 = 40;
    pub(crate) const X4: u32 = 48;
    pub(crate) const X5: u32 = 56;
    pub(crate) const X6: u32 = 64;
    pub(crate) const X7: u32 = 72;
    pub(crate) const X8: u32 = 80;
    pub(crate) const X9: u32 = 88;
    pub(crate) const X10: u32 = 96;
    pub(crate) const X11: u32 = 104;
    pub(crate) const X12: u32 = 112;
    pub(crate) const X13: u32 = 120;
    pub(crate) const X14: u32 = 128;
    pub(crate) const X15: u32 = 136;
    pub(crate) const X16: u32 = 144;
    pub(crate) const X17: u32 = 152;
    pub(crate) const X18: u32 = 160;
    pub(crate) const X19: u32 = 168;
    pub(crate) const X20: u32 = 176;
    pub(crate) const X21: u32 = 184;
    pub(crate) const X22: u32 = 192;
    pub(crate) const X23: u32 = 200;
    pub(crate) const X24: u32 = 208;
    pub(crate) const X25: u32 = 216;
    pub(crate) const X26: u32 = 224;
    pub(crate) const X27: u32 = 232;
    pub(crate) const X28: u32 = 240;
    pub(crate) const X29: u32 = 248; // FP (frame pointer)
    pub(crate) const X30: u32 = 256; // LR (link register)

    // Stack pointer (XSP)
    pub(crate) const XSP: u32 = 264;

    // Program counter
    pub(crate) const PC: u32 = 272;

    // Condition code thunks
    pub(crate) const CC_OP: u32 = 280;
    pub(crate) const CC_DEP1: u32 = 288;
    pub(crate) const CC_DEP2: u32 = 296;
    pub(crate) const CC_NDEP: u32 = 304;

    // Thread pointer
    pub(crate) const TPIDR_EL0: u32 = 312;

    // NEON/SIMD registers (V0-V31, 128-bit each)
    pub(crate) const Q0: u32 = 320;
    pub(crate) const Q1: u32 = 336;
    pub(crate) const Q2: u32 = 352;
    pub(crate) const Q3: u32 = 368;
    pub(crate) const Q4: u32 = 384;
    pub(crate) const Q5: u32 = 400;
    pub(crate) const Q6: u32 = 416;
    pub(crate) const Q7: u32 = 432;
    pub(crate) const Q8: u32 = 448;
    pub(crate) const Q9: u32 = 464;
    pub(crate) const Q10: u32 = 480;
    pub(crate) const Q11: u32 = 496;
    pub(crate) const Q12: u32 = 512;
    pub(crate) const Q13: u32 = 528;
    pub(crate) const Q14: u32 = 544;
    pub(crate) const Q15: u32 = 560;
    pub(crate) const Q16: u32 = 576;
    pub(crate) const Q17: u32 = 592;
    pub(crate) const Q18: u32 = 608;
    pub(crate) const Q19: u32 = 624;
    pub(crate) const Q20: u32 = 640;
    pub(crate) const Q21: u32 = 656;
    pub(crate) const Q22: u32 = 672;
    pub(crate) const Q23: u32 = 688;
    pub(crate) const Q24: u32 = 704;
    pub(crate) const Q25: u32 = 720;
    pub(crate) const Q26: u32 = 736;
    pub(crate) const Q27: u32 = 752;
    pub(crate) const Q28: u32 = 768;
    pub(crate) const Q29: u32 = 784;
    pub(crate) const Q30: u32 = 800;
    pub(crate) const Q31: u32 = 816;

    // Tail of VexGuestARM64State, in declaration order. `guest_QCFLAG` is a
    // U128 (the sticky FPSR.QC saturation flag), so the next field starts at
    // 848 — not 836. See /usr/include/valgrind/libvex_guest_arm64.h and
    // archinfo ArchAArch64.registers, which agree on every offset below.
    pub(crate) const QCFLAG: u32 = 832;
    pub(crate) const EMNOTE: u32 = 848;
    pub(crate) const CMSTART: u32 = 856;
    pub(crate) const CMLEN: u32 = 864;
    pub(crate) const NRADDR: u32 = 872;
    pub(crate) const IP_AT_SYSCALL: u32 = 880;
    pub(crate) const FPCR: u32 = 888;

    // Total guest state size. After guest_FPCR come 4 bytes of padding and
    // the LL/SC fallback block (guest_LLSC_{SIZE,ADDR,DATA_LO64,DATA_HI64},
    // 896..928). We do not name those, but the buffer must cover them or
    // IR-level PUTs to them are silently dropped by RegisterFile::put.
    pub(crate) const GUEST_STATE_SIZE: usize = 928;
}

// Canonical registers: drive `register_name(offset)` reverse lookups.
// X29/X30/XSP use their conventional aliases (fp/lr/sp) as the canonical
// reverse-lookup name, matching the prior register_name behavior.
const CANONICAL: &[RegEntry] = &[
    ("x0", offsets::X0, 8),
    ("x1", offsets::X1, 8),
    ("x2", offsets::X2, 8),
    ("x3", offsets::X3, 8),
    ("x4", offsets::X4, 8),
    ("x5", offsets::X5, 8),
    ("x6", offsets::X6, 8),
    ("x7", offsets::X7, 8),
    ("x8", offsets::X8, 8),
    ("x9", offsets::X9, 8),
    ("x10", offsets::X10, 8),
    ("x11", offsets::X11, 8),
    ("x12", offsets::X12, 8),
    ("x13", offsets::X13, 8),
    ("x14", offsets::X14, 8),
    ("x15", offsets::X15, 8),
    ("x16", offsets::X16, 8),
    ("x17", offsets::X17, 8),
    ("x18", offsets::X18, 8),
    ("x19", offsets::X19, 8),
    ("x20", offsets::X20, 8),
    ("x21", offsets::X21, 8),
    ("x22", offsets::X22, 8),
    ("x23", offsets::X23, 8),
    ("x24", offsets::X24, 8),
    ("x25", offsets::X25, 8),
    ("x26", offsets::X26, 8),
    ("x27", offsets::X27, 8),
    ("x28", offsets::X28, 8),
    ("fp", offsets::X29, 8),
    ("lr", offsets::X30, 8),
    ("sp", offsets::XSP, 8),
    ("pc", offsets::PC, 8),
];

const ALIASES: &[RegEntry] = &[
    // X29/X30/XSP alternate names
    ("x29", offsets::X29, 8),
    // Architecture-independent frame-pointer name, mirroring archinfo's
    // ArchAArch64 bp=(248,8) and the `bp`/`fp` interchangeability documented
    // on `Arch::bp_offset` (angr-03vl4.1).
    ("bp", offsets::X29, 8),
    ("x30", offsets::X30, 8),
    ("xsp", offsets::XSP, 8),
    // 32-bit W registers (lower 32 bits of X registers)
    ("w0", offsets::X0, 4),
    ("w1", offsets::X1, 4),
    ("w2", offsets::X2, 4),
    ("w3", offsets::X3, 4),
    ("w4", offsets::X4, 4),
    ("w5", offsets::X5, 4),
    ("w6", offsets::X6, 4),
    ("w7", offsets::X7, 4),
    ("w8", offsets::X8, 4),
    ("w9", offsets::X9, 4),
    ("w10", offsets::X10, 4),
    ("w11", offsets::X11, 4),
    ("w12", offsets::X12, 4),
    ("w13", offsets::X13, 4),
    ("w14", offsets::X14, 4),
    ("w15", offsets::X15, 4),
    ("w16", offsets::X16, 4),
    ("w17", offsets::X17, 4),
    ("w18", offsets::X18, 4),
    ("w19", offsets::X19, 4),
    ("w20", offsets::X20, 4),
    ("w21", offsets::X21, 4),
    ("w22", offsets::X22, 4),
    ("w23", offsets::X23, 4),
    ("w24", offsets::X24, 4),
    ("w25", offsets::X25, 4),
    ("w26", offsets::X26, 4),
    ("w27", offsets::X27, 4),
    ("w28", offsets::X28, 4),
    ("w29", offsets::X29, 4),
    ("w30", offsets::X30, 4),
    // CC thunks
    ("cc_op", offsets::CC_OP, 8),
    ("cc_dep1", offsets::CC_DEP1, 8),
    ("cc_dep2", offsets::CC_DEP2, 8),
    ("cc_ndep", offsets::CC_NDEP, 8),
    // Thread pointer
    ("tpidr_el0", offsets::TPIDR_EL0, 8),
    // SIMD/NEON Q/V registers (128-bit)
    ("q0", offsets::Q0, 16),
    ("v0", offsets::Q0, 16),
    ("q1", offsets::Q1, 16),
    ("v1", offsets::Q1, 16),
    ("q2", offsets::Q2, 16),
    ("v2", offsets::Q2, 16),
    ("q3", offsets::Q3, 16),
    ("v3", offsets::Q3, 16),
    ("q4", offsets::Q4, 16),
    ("v4", offsets::Q4, 16),
    ("q5", offsets::Q5, 16),
    ("v5", offsets::Q5, 16),
    ("q6", offsets::Q6, 16),
    ("v6", offsets::Q6, 16),
    ("q7", offsets::Q7, 16),
    ("v7", offsets::Q7, 16),
    ("q8", offsets::Q8, 16),
    ("v8", offsets::Q8, 16),
    ("q9", offsets::Q9, 16),
    ("v9", offsets::Q9, 16),
    ("q10", offsets::Q10, 16),
    ("v10", offsets::Q10, 16),
    ("q11", offsets::Q11, 16),
    ("v11", offsets::Q11, 16),
    ("q12", offsets::Q12, 16),
    ("v12", offsets::Q12, 16),
    ("q13", offsets::Q13, 16),
    ("v13", offsets::Q13, 16),
    ("q14", offsets::Q14, 16),
    ("v14", offsets::Q14, 16),
    ("q15", offsets::Q15, 16),
    ("v15", offsets::Q15, 16),
    ("q16", offsets::Q16, 16),
    ("v16", offsets::Q16, 16),
    ("q17", offsets::Q17, 16),
    ("v17", offsets::Q17, 16),
    ("q18", offsets::Q18, 16),
    ("v18", offsets::Q18, 16),
    ("q19", offsets::Q19, 16),
    ("v19", offsets::Q19, 16),
    ("q20", offsets::Q20, 16),
    ("v20", offsets::Q20, 16),
    ("q21", offsets::Q21, 16),
    ("v21", offsets::Q21, 16),
    ("q22", offsets::Q22, 16),
    ("v22", offsets::Q22, 16),
    ("q23", offsets::Q23, 16),
    ("v23", offsets::Q23, 16),
    ("q24", offsets::Q24, 16),
    ("v24", offsets::Q24, 16),
    ("q25", offsets::Q25, 16),
    ("v25", offsets::Q25, 16),
    ("q26", offsets::Q26, 16),
    ("v26", offsets::Q26, 16),
    ("q27", offsets::Q27, 16),
    ("v27", offsets::Q27, 16),
    ("q28", offsets::Q28, 16),
    ("v28", offsets::Q28, 16),
    ("q29", offsets::Q29, 16),
    ("v29", offsets::Q29, 16),
    ("q30", offsets::Q30, 16),
    ("v30", offsets::Q30, 16),
    ("q31", offsets::Q31, 16),
    ("v31", offsets::Q31, 16),
    // D registers (lower 64-bit of Q registers)
    ("d0", offsets::Q0, 8),
    ("d1", offsets::Q1, 8),
    ("d2", offsets::Q2, 8),
    ("d3", offsets::Q3, 8),
    ("d4", offsets::Q4, 8),
    ("d5", offsets::Q5, 8),
    ("d6", offsets::Q6, 8),
    ("d7", offsets::Q7, 8),
    ("d8", offsets::Q8, 8),
    ("d9", offsets::Q9, 8),
    ("d10", offsets::Q10, 8),
    ("d11", offsets::Q11, 8),
    ("d12", offsets::Q12, 8),
    ("d13", offsets::Q13, 8),
    ("d14", offsets::Q14, 8),
    ("d15", offsets::Q15, 8),
    ("d16", offsets::Q16, 8),
    ("d17", offsets::Q17, 8),
    ("d18", offsets::Q18, 8),
    ("d19", offsets::Q19, 8),
    ("d20", offsets::Q20, 8),
    ("d21", offsets::Q21, 8),
    ("d22", offsets::Q22, 8),
    ("d23", offsets::Q23, 8),
    ("d24", offsets::Q24, 8),
    ("d25", offsets::Q25, 8),
    ("d26", offsets::Q26, 8),
    ("d27", offsets::Q27, 8),
    ("d28", offsets::Q28, 8),
    ("d29", offsets::Q29, 8),
    ("d30", offsets::Q30, 8),
    ("d31", offsets::Q31, 8),
    // Guest-state tail (see the offsets module for the layout rationale).
    ("qcflag", offsets::QCFLAG, 16),
    ("emnote", offsets::EMNOTE, 4),
    ("cmstart", offsets::CMSTART, 8),
    ("cmlen", offsets::CMLEN, 8),
    ("nraddr", offsets::NRADDR, 8),
    ("ip_at_syscall", offsets::IP_AT_SYSCALL, 8),
    ("fpcr", offsets::FPCR, 4),
];

const REGISTER_NAMES: &[&str] = &[
    "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14",
    "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27",
    "x28", "fp", "lr", "sp", "pc",
];

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

    impl_arch_registers!(CANONICAL, ALIASES, REGISTER_NAMES);

    fn syscall_num_offset(&self) -> Option<u32> {
        // Linux AArch64 syscall convention puts the syscall number in X8
        Some(offsets::X8)
    }

    fn is_little_endian(&self) -> bool {
        true
    }
}

#[cfg(test)]
#[path = "arm64_tests.rs"]
mod tests;
