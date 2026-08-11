//! ARM (32-bit) architecture definition.
//!
//! This module provides the register layout and architecture-specific
//! information for 32-bit ARM.

use super::{Arch, RegEntry, impl_arch_registers};
use crate::vex::VexArch;

/// ARM (32-bit) architecture.
///
/// Name mirrors archinfo's `ArchARM` / the VEX `Ijk`/guest-state spelling, and
/// is matched against `archinfo` arch names elsewhere in `arch::mod`, so it
/// keeps the all-caps form. (Only visible to clippy since angr-9ke6b.214 made
/// `arch` a `pub(crate)` module — style lints skip exported items.)
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct ARM;

// ARM VEX guest state offsets (from VEX/pub/libvex_guest_arm.h)
// These match the VexGuestARMState structure layout.
pub(crate) mod offsets {
    // General purpose registers (R0-R15)
    pub(crate) const R0: u32 = 8;
    pub(crate) const R1: u32 = 12;
    pub(crate) const R2: u32 = 16;
    pub(crate) const R3: u32 = 20;
    pub(crate) const R4: u32 = 24;
    pub(crate) const R5: u32 = 28;
    pub(crate) const R6: u32 = 32;
    pub(crate) const R7: u32 = 36;
    pub(crate) const R8: u32 = 40;
    pub(crate) const R9: u32 = 44;
    pub(crate) const R10: u32 = 48;
    pub(crate) const R11: u32 = 52; // FP (frame pointer)
    pub(crate) const R12: u32 = 56; // IP (intra-procedure scratch)
    pub(crate) const R13: u32 = 60; // SP (stack pointer)
    pub(crate) const R14: u32 = 64; // LR (link register)
    pub(crate) const R15T: u32 = 68; // PC with Thumb bit

    // Condition code thunks
    pub(crate) const CC_OP: u32 = 72;
    pub(crate) const CC_DEP1: u32 = 76;
    pub(crate) const CC_DEP2: u32 = 80;
    pub(crate) const CC_NDEP: u32 = 84;

    // Thumb/ARM mode flag
    pub(crate) const QFLAG32: u32 = 88;
    pub(crate) const GEFLAG0: u32 = 92;
    pub(crate) const GEFLAG1: u32 = 96;
    pub(crate) const GEFLAG2: u32 = 100;
    pub(crate) const GEFLAG3: u32 = 104;

    // Emulation-warning note
    pub(crate) const EMNOTE: u32 = 108;

    // Chunk-marker / self-modifying-code fields and syscall bookkeeping.
    // These occupy the 112-127 VEX block that a prior revision omitted,
    // which shifted every VFP/FPSCR/TPIDRURO field -16 vs the real layout
    // (angr-ihfe5). Keep them named so IR PUT/GET offsets resolve.
    pub(crate) const CMSTART: u32 = 112;
    pub(crate) const CMLEN: u32 = 116;
    pub(crate) const NRADDR: u32 = 120;
    pub(crate) const IP_AT_SYSCALL: u32 = 124;

    // NEON/VFP registers (D0-D31, Q0-Q15)
    pub(crate) const D0: u32 = 128;
    pub(crate) const D1: u32 = 136;
    pub(crate) const D2: u32 = 144;
    pub(crate) const D3: u32 = 152;
    pub(crate) const D4: u32 = 160;
    pub(crate) const D5: u32 = 168;
    pub(crate) const D6: u32 = 176;
    pub(crate) const D7: u32 = 184;
    pub(crate) const D8: u32 = 192;
    pub(crate) const D9: u32 = 200;
    pub(crate) const D10: u32 = 208;
    pub(crate) const D11: u32 = 216;
    pub(crate) const D12: u32 = 224;
    pub(crate) const D13: u32 = 232;
    pub(crate) const D14: u32 = 240;
    pub(crate) const D15: u32 = 248;
    pub(crate) const D16: u32 = 256;
    pub(crate) const D17: u32 = 264;
    pub(crate) const D18: u32 = 272;
    pub(crate) const D19: u32 = 280;
    pub(crate) const D20: u32 = 288;
    pub(crate) const D21: u32 = 296;
    pub(crate) const D22: u32 = 304;
    pub(crate) const D23: u32 = 312;
    pub(crate) const D24: u32 = 320;
    pub(crate) const D25: u32 = 328;
    pub(crate) const D26: u32 = 336;
    pub(crate) const D27: u32 = 344;
    pub(crate) const D28: u32 = 352;
    pub(crate) const D29: u32 = 360;
    pub(crate) const D30: u32 = 368;
    pub(crate) const D31: u32 = 376;

    // FPSCR
    pub(crate) const FPSCR: u32 = 384;

    // TPIDRURO (thread pointer)
    pub(crate) const TPIDRURO: u32 = 388;

    // IT state for conditional execution
    pub(crate) const ITSTATE: u32 = 392;

    // Total guest state size. ITSTATE ends at 396; VexGuestARMState carries
    // 8-byte-aligned ULong D-registers, so sizeof rounds up to 400.
    pub(crate) const GUEST_STATE_SIZE: usize = 400;
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
    // r11/r13/r14/r15 alternate names.
    //
    // "ip" is R15T, the PC — NOT r12 (angr-itm3u, angr-690nc). The ARM ABI
    // calls r12 the intra-procedure-call scratch register "ip", but
    // archinfo/angr use "ip" as the architecture-independent *instruction
    // pointer* alias on every arch (ARMEL: (68, 4) = R15T). The ABI reading
    // lived here once and made a Python-side write through the name "ip" land
    // in r12 instead of the PC — a silent cross-register corruption. r12 stays
    // reachable as "r12".
    ("ip", offsets::R15T, 4),
    ("fp", offsets::R11, 4),
    // Architecture-independent frame-pointer name, mirroring archinfo's
    // ArchARMEL bp=(52,4) and the `bp`/`fp` interchangeability documented on
    // `Arch::bp_offset` (angr-03vl4.1).
    ("bp", offsets::R11, 4),
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
    // Emulation note + chunk-marker/syscall bookkeeping (VEX 108-127 block)
    ("emnote", offsets::EMNOTE, 4),
    ("cmstart", offsets::CMSTART, 4),
    ("cmlen", offsets::CMLEN, 4),
    ("nraddr", offsets::NRADDR, 4),
    ("ip_at_syscall", offsets::IP_AT_SYSCALL, 4),
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

    fn syscall_num_offset(&self) -> Option<u32> {
        // EABI Linux syscall convention puts the syscall number in R7
        Some(offsets::R7)
    }

    fn is_little_endian(&self) -> bool {
        true // ARM can be big or little endian, but little is more common
    }
}

test_submod!("arm_tests.rs" => tests);
