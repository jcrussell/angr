//! MIPS architecture definitions.
//!
//! This module provides the register layout and architecture-specific
//! information for MIPS32 and MIPS64.

use super::{Arch, RegEntry, impl_arch_registers};
use crate::vex::VexArch;

/// MIPS32 architecture.
#[derive(Debug, Clone, Copy)]
pub struct MIPS32;

/// MIPS64 architecture.
#[derive(Debug, Clone, Copy)]
pub struct MIPS64;

// MIPS32 VEX guest state offsets (from VEX/pub/libvex_guest_mips32.h)
mod offsets32 {
    pub const R0: u32 = 8; // zero
    pub const R1: u32 = 12; // at
    pub const R2: u32 = 16; // v0
    pub const R3: u32 = 20; // v1
    pub const R4: u32 = 24; // a0
    pub const R5: u32 = 28; // a1
    pub const R6: u32 = 32; // a2
    pub const R7: u32 = 36; // a3
    pub const R8: u32 = 40; // t0
    pub const R9: u32 = 44; // t1
    pub const R10: u32 = 48; // t2
    pub const R11: u32 = 52; // t3
    pub const R12: u32 = 56; // t4
    pub const R13: u32 = 60; // t5
    pub const R14: u32 = 64; // t6
    pub const R15: u32 = 68; // t7
    pub const R16: u32 = 72; // s0
    pub const R17: u32 = 76; // s1
    pub const R18: u32 = 80; // s2
    pub const R19: u32 = 84; // s3
    pub const R20: u32 = 88; // s4
    pub const R21: u32 = 92; // s5
    pub const R22: u32 = 96; // s6
    pub const R23: u32 = 100; // s7
    pub const R24: u32 = 104; // t8
    pub const R25: u32 = 108; // t9
    pub const R26: u32 = 112; // k0
    pub const R27: u32 = 116; // k1
    pub const R28: u32 = 120; // gp
    pub const R29: u32 = 124; // sp
    pub const R30: u32 = 128; // fp (s8)
    pub const R31: u32 = 132; // ra

    pub const PC: u32 = 136;
    pub const HI: u32 = 140;
    pub const LO: u32 = 144;

    // FPU registers
    pub const F0: u32 = 152;
    pub const F1: u32 = 160;
    pub const F2: u32 = 168;
    pub const F3: u32 = 176;
    pub const F4: u32 = 184;
    pub const F5: u32 = 192;
    pub const F6: u32 = 200;
    pub const F7: u32 = 208;
    pub const F8: u32 = 216;
    pub const F9: u32 = 224;
    pub const F10: u32 = 232;
    pub const F11: u32 = 240;
    pub const F12: u32 = 248;
    pub const F13: u32 = 256;
    pub const F14: u32 = 264;
    pub const F15: u32 = 272;
    pub const F16: u32 = 280;
    pub const F17: u32 = 288;
    pub const F18: u32 = 296;
    pub const F19: u32 = 304;
    pub const F20: u32 = 312;
    pub const F21: u32 = 320;
    pub const F22: u32 = 328;
    pub const F23: u32 = 336;
    pub const F24: u32 = 344;
    pub const F25: u32 = 352;
    pub const F26: u32 = 360;
    pub const F27: u32 = 368;
    pub const F28: u32 = 376;
    pub const F29: u32 = 384;
    pub const F30: u32 = 392;
    pub const F31: u32 = 400;

    pub const FIR: u32 = 408;
    pub const FCCR: u32 = 412;
    pub const FEXR: u32 = 416;
    pub const FENR: u32 = 420;
    pub const FCSR: u32 = 424;

    pub const GUEST_STATE_SIZE: usize = 432;
}

// MIPS64 VEX guest state offsets (from VEX/pub/libvex_guest_mips64.h)
mod offsets64 {
    pub const R0: u32 = 16; // zero
    pub const R1: u32 = 24; // at
    pub const R2: u32 = 32; // v0
    pub const R3: u32 = 40; // v1
    pub const R4: u32 = 48; // a0
    pub const R5: u32 = 56; // a1
    pub const R6: u32 = 64; // a2
    pub const R7: u32 = 72; // a3
    pub const R8: u32 = 80; // t0/a4
    pub const R9: u32 = 88; // t1/a5
    pub const R10: u32 = 96; // t2/a6
    pub const R11: u32 = 104; // t3/a7
    pub const R12: u32 = 112; // t4
    pub const R13: u32 = 120; // t5
    pub const R14: u32 = 128; // t6
    pub const R15: u32 = 136; // t7
    pub const R16: u32 = 144; // s0
    pub const R17: u32 = 152; // s1
    pub const R18: u32 = 160; // s2
    pub const R19: u32 = 168; // s3
    pub const R20: u32 = 176; // s4
    pub const R21: u32 = 184; // s5
    pub const R22: u32 = 192; // s6
    pub const R23: u32 = 200; // s7
    pub const R24: u32 = 208; // t8
    pub const R25: u32 = 216; // t9
    pub const R26: u32 = 224; // k0
    pub const R27: u32 = 232; // k1
    pub const R28: u32 = 240; // gp
    pub const R29: u32 = 248; // sp
    pub const R30: u32 = 256; // fp (s8)
    pub const R31: u32 = 264; // ra

    pub const PC: u32 = 272;
    pub const HI: u32 = 280;
    pub const LO: u32 = 288;

    // FPU registers (64-bit each) — VEX guest_f0..guest_f31 (libvex_guest_mips64.h)
    pub const F0: u32 = 296;
    pub const F1: u32 = 304;
    pub const F2: u32 = 312;
    pub const F3: u32 = 320;
    pub const F4: u32 = 328;
    pub const F5: u32 = 336;
    pub const F6: u32 = 344;
    pub const F7: u32 = 352;
    pub const F8: u32 = 360;
    pub const F9: u32 = 368;
    pub const F10: u32 = 376;
    pub const F11: u32 = 384;
    pub const F12: u32 = 392;
    pub const F13: u32 = 400;
    pub const F14: u32 = 408;
    pub const F15: u32 = 416;
    pub const F16: u32 = 424;
    pub const F17: u32 = 432;
    pub const F18: u32 = 440;
    pub const F19: u32 = 448;
    pub const F20: u32 = 456;
    pub const F21: u32 = 464;
    pub const F22: u32 = 472;
    pub const F23: u32 = 480;
    pub const F24: u32 = 488;
    pub const F25: u32 = 496;
    pub const F26: u32 = 504;
    pub const F27: u32 = 512;
    pub const F28: u32 = 520;
    pub const F29: u32 = 528;
    pub const F30: u32 = 536;
    pub const F31: u32 = 544;

    pub const FIR: u32 = 552;
    pub const FCCR: u32 = 556;
    pub const FEXR: u32 = 560;
    pub const FENR: u32 = 564;
    pub const FCSR: u32 = 568;

    pub const GUEST_STATE_SIZE: usize = 592;
}

// MIPS32 canonical: ABI mnemonics drive register_name reverse lookup.
const CANONICAL_MIPS32: &[RegEntry] = &[
    ("zero", offsets32::R0, 4),
    ("at", offsets32::R1, 4),
    ("v0", offsets32::R2, 4),
    ("v1", offsets32::R3, 4),
    ("a0", offsets32::R4, 4),
    ("a1", offsets32::R5, 4),
    ("a2", offsets32::R6, 4),
    ("a3", offsets32::R7, 4),
    ("t0", offsets32::R8, 4),
    ("t1", offsets32::R9, 4),
    ("t2", offsets32::R10, 4),
    ("t3", offsets32::R11, 4),
    ("t4", offsets32::R12, 4),
    ("t5", offsets32::R13, 4),
    ("t6", offsets32::R14, 4),
    ("t7", offsets32::R15, 4),
    ("s0", offsets32::R16, 4),
    ("s1", offsets32::R17, 4),
    ("s2", offsets32::R18, 4),
    ("s3", offsets32::R19, 4),
    ("s4", offsets32::R20, 4),
    ("s5", offsets32::R21, 4),
    ("s6", offsets32::R22, 4),
    ("s7", offsets32::R23, 4),
    ("t8", offsets32::R24, 4),
    ("t9", offsets32::R25, 4),
    ("k0", offsets32::R26, 4),
    ("k1", offsets32::R27, 4),
    ("gp", offsets32::R28, 4),
    ("sp", offsets32::R29, 4),
    ("fp", offsets32::R30, 4),
    ("ra", offsets32::R31, 4),
    ("pc", offsets32::PC, 4),
    ("hi", offsets32::HI, 4),
    ("lo", offsets32::LO, 4),
];

const ALIASES_MIPS32: &[RegEntry] = &[
    // Numeric rN aliases
    ("r0", offsets32::R0, 4),
    ("r1", offsets32::R1, 4),
    ("r2", offsets32::R2, 4),
    ("r3", offsets32::R3, 4),
    ("r4", offsets32::R4, 4),
    ("r5", offsets32::R5, 4),
    ("r6", offsets32::R6, 4),
    ("r7", offsets32::R7, 4),
    ("r8", offsets32::R8, 4),
    ("r9", offsets32::R9, 4),
    ("r10", offsets32::R10, 4),
    ("r11", offsets32::R11, 4),
    ("r12", offsets32::R12, 4),
    ("r13", offsets32::R13, 4),
    ("r14", offsets32::R14, 4),
    ("r15", offsets32::R15, 4),
    ("r16", offsets32::R16, 4),
    ("r17", offsets32::R17, 4),
    ("r18", offsets32::R18, 4),
    ("r19", offsets32::R19, 4),
    ("r20", offsets32::R20, 4),
    ("r21", offsets32::R21, 4),
    ("r22", offsets32::R22, 4),
    ("r23", offsets32::R23, 4),
    ("r24", offsets32::R24, 4),
    ("r25", offsets32::R25, 4),
    ("r26", offsets32::R26, 4),
    ("r27", offsets32::R27, 4),
    ("r28", offsets32::R28, 4),
    ("r29", offsets32::R29, 4),
    ("r30", offsets32::R30, 4),
    ("r31", offsets32::R31, 4),
    // $N aliases
    ("$0", offsets32::R0, 4),
    ("$1", offsets32::R1, 4),
    ("$2", offsets32::R2, 4),
    ("$3", offsets32::R3, 4),
    ("$4", offsets32::R4, 4),
    ("$5", offsets32::R5, 4),
    ("$6", offsets32::R6, 4),
    ("$7", offsets32::R7, 4),
    ("$8", offsets32::R8, 4),
    ("$9", offsets32::R9, 4),
    ("$10", offsets32::R10, 4),
    ("$11", offsets32::R11, 4),
    ("$12", offsets32::R12, 4),
    ("$13", offsets32::R13, 4),
    ("$14", offsets32::R14, 4),
    ("$15", offsets32::R15, 4),
    ("$16", offsets32::R16, 4),
    ("$17", offsets32::R17, 4),
    ("$18", offsets32::R18, 4),
    ("$19", offsets32::R19, 4),
    ("$20", offsets32::R20, 4),
    ("$21", offsets32::R21, 4),
    ("$22", offsets32::R22, 4),
    ("$23", offsets32::R23, 4),
    ("$24", offsets32::R24, 4),
    ("$25", offsets32::R25, 4),
    ("$26", offsets32::R26, 4),
    ("$27", offsets32::R27, 4),
    ("$28", offsets32::R28, 4),
    ("$29", offsets32::R29, 4),
    ("$30", offsets32::R30, 4),
    ("$31", offsets32::R31, 4),
    // R30 alias s8 (same offset as fp)
    ("s8", offsets32::R30, 4),
    // FPU registers (64-bit on MIPS32 VEX state)
    ("f0", offsets32::F0, 8),
    ("f1", offsets32::F1, 8),
    ("f2", offsets32::F2, 8),
    ("f3", offsets32::F3, 8),
    ("f4", offsets32::F4, 8),
    ("f5", offsets32::F5, 8),
    ("f6", offsets32::F6, 8),
    ("f7", offsets32::F7, 8),
    ("f8", offsets32::F8, 8),
    ("f9", offsets32::F9, 8),
    ("f10", offsets32::F10, 8),
    ("f11", offsets32::F11, 8),
    ("f12", offsets32::F12, 8),
    ("f13", offsets32::F13, 8),
    ("f14", offsets32::F14, 8),
    ("f15", offsets32::F15, 8),
    ("f16", offsets32::F16, 8),
    ("f17", offsets32::F17, 8),
    ("f18", offsets32::F18, 8),
    ("f19", offsets32::F19, 8),
    ("f20", offsets32::F20, 8),
    ("f21", offsets32::F21, 8),
    ("f22", offsets32::F22, 8),
    ("f23", offsets32::F23, 8),
    ("f24", offsets32::F24, 8),
    ("f25", offsets32::F25, 8),
    ("f26", offsets32::F26, 8),
    ("f27", offsets32::F27, 8),
    ("f28", offsets32::F28, 8),
    ("f29", offsets32::F29, 8),
    ("f30", offsets32::F30, 8),
    ("f31", offsets32::F31, 8),
    ("$f0", offsets32::F0, 8),
    ("$f1", offsets32::F1, 8),
    ("$f2", offsets32::F2, 8),
    ("$f3", offsets32::F3, 8),
    ("$f4", offsets32::F4, 8),
    ("$f5", offsets32::F5, 8),
    ("$f6", offsets32::F6, 8),
    ("$f7", offsets32::F7, 8),
    ("$f8", offsets32::F8, 8),
    ("$f9", offsets32::F9, 8),
    ("$f10", offsets32::F10, 8),
    ("$f11", offsets32::F11, 8),
    ("$f12", offsets32::F12, 8),
    ("$f13", offsets32::F13, 8),
    ("$f14", offsets32::F14, 8),
    ("$f15", offsets32::F15, 8),
    ("$f16", offsets32::F16, 8),
    ("$f17", offsets32::F17, 8),
    ("$f18", offsets32::F18, 8),
    ("$f19", offsets32::F19, 8),
    ("$f20", offsets32::F20, 8),
    ("$f21", offsets32::F21, 8),
    ("$f22", offsets32::F22, 8),
    ("$f23", offsets32::F23, 8),
    ("$f24", offsets32::F24, 8),
    ("$f25", offsets32::F25, 8),
    ("$f26", offsets32::F26, 8),
    ("$f27", offsets32::F27, 8),
    ("$f28", offsets32::F28, 8),
    ("$f29", offsets32::F29, 8),
    ("$f30", offsets32::F30, 8),
    ("$f31", offsets32::F31, 8),
    // FPU control registers (32-bit)
    ("fir", offsets32::FIR, 4),
    ("fccr", offsets32::FCCR, 4),
    ("fexr", offsets32::FEXR, 4),
    ("fenr", offsets32::FENR, 4),
    ("fcsr", offsets32::FCSR, 4),
];

const REGISTER_NAMES_MIPS32: &[&str] = &[
    "zero", "at", "v0", "v1", "a0", "a1", "a2", "a3", "t0", "t1", "t2", "t3", "t4", "t5", "t6",
    "t7", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "t8", "t9", "k0", "k1", "gp", "sp", "fp",
    "ra", "pc", "hi", "lo",
];

// MIPS64 canonical: preserves the original (intentionally narrow) reverse
// lookup behavior — only zero/v0/a0/sp/fp/ra/pc are reverse-mapped, matching
// the prior register_name match arms. Other GPR mnemonics live in ALIASES.
const CANONICAL_MIPS64: &[RegEntry] = &[
    ("zero", offsets64::R0, 8),
    ("v0", offsets64::R2, 8),
    ("a0", offsets64::R4, 8),
    ("sp", offsets64::R29, 8),
    ("fp", offsets64::R30, 8),
    ("ra", offsets64::R31, 8),
    ("pc", offsets64::PC, 8),
];

const ALIASES_MIPS64: &[RegEntry] = &[
    ("at", offsets64::R1, 8),
    ("v1", offsets64::R3, 8),
    ("a1", offsets64::R5, 8),
    ("a2", offsets64::R6, 8),
    ("a3", offsets64::R7, 8),
    ("a4", offsets64::R8, 8),
    ("a5", offsets64::R9, 8),
    ("a6", offsets64::R10, 8),
    ("a7", offsets64::R11, 8),
    ("t0", offsets64::R8, 8),
    ("t1", offsets64::R9, 8),
    ("t2", offsets64::R10, 8),
    ("t3", offsets64::R11, 8),
    ("t4", offsets64::R12, 8),
    ("t5", offsets64::R13, 8),
    ("t6", offsets64::R14, 8),
    ("t7", offsets64::R15, 8),
    ("s0", offsets64::R16, 8),
    ("s1", offsets64::R17, 8),
    ("s2", offsets64::R18, 8),
    ("s3", offsets64::R19, 8),
    ("s4", offsets64::R20, 8),
    ("s5", offsets64::R21, 8),
    ("s6", offsets64::R22, 8),
    ("s7", offsets64::R23, 8),
    ("t8", offsets64::R24, 8),
    ("t9", offsets64::R25, 8),
    ("k0", offsets64::R26, 8),
    ("k1", offsets64::R27, 8),
    ("gp", offsets64::R28, 8),
    ("s8", offsets64::R30, 8),
    ("hi", offsets64::HI, 8),
    ("lo", offsets64::LO, 8),
    // Numeric rN aliases
    ("r0", offsets64::R0, 8),
    ("r1", offsets64::R1, 8),
    ("r2", offsets64::R2, 8),
    ("r3", offsets64::R3, 8),
    ("r4", offsets64::R4, 8),
    ("r5", offsets64::R5, 8),
    ("r6", offsets64::R6, 8),
    ("r7", offsets64::R7, 8),
    ("r8", offsets64::R8, 8),
    ("r9", offsets64::R9, 8),
    ("r10", offsets64::R10, 8),
    ("r11", offsets64::R11, 8),
    ("r12", offsets64::R12, 8),
    ("r13", offsets64::R13, 8),
    ("r14", offsets64::R14, 8),
    ("r15", offsets64::R15, 8),
    ("r16", offsets64::R16, 8),
    ("r17", offsets64::R17, 8),
    ("r18", offsets64::R18, 8),
    ("r19", offsets64::R19, 8),
    ("r20", offsets64::R20, 8),
    ("r21", offsets64::R21, 8),
    ("r22", offsets64::R22, 8),
    ("r23", offsets64::R23, 8),
    ("r24", offsets64::R24, 8),
    ("r25", offsets64::R25, 8),
    ("r26", offsets64::R26, 8),
    ("r27", offsets64::R27, 8),
    ("r28", offsets64::R28, 8),
    ("r29", offsets64::R29, 8),
    ("r30", offsets64::R30, 8),
    ("r31", offsets64::R31, 8),
    // $N aliases
    ("$0", offsets64::R0, 8),
    ("$1", offsets64::R1, 8),
    ("$2", offsets64::R2, 8),
    ("$3", offsets64::R3, 8),
    ("$4", offsets64::R4, 8),
    ("$5", offsets64::R5, 8),
    ("$6", offsets64::R6, 8),
    ("$7", offsets64::R7, 8),
    ("$8", offsets64::R8, 8),
    ("$9", offsets64::R9, 8),
    ("$10", offsets64::R10, 8),
    ("$11", offsets64::R11, 8),
    ("$12", offsets64::R12, 8),
    ("$13", offsets64::R13, 8),
    ("$14", offsets64::R14, 8),
    ("$15", offsets64::R15, 8),
    ("$16", offsets64::R16, 8),
    ("$17", offsets64::R17, 8),
    ("$18", offsets64::R18, 8),
    ("$19", offsets64::R19, 8),
    ("$20", offsets64::R20, 8),
    ("$21", offsets64::R21, 8),
    ("$22", offsets64::R22, 8),
    ("$23", offsets64::R23, 8),
    ("$24", offsets64::R24, 8),
    ("$25", offsets64::R25, 8),
    ("$26", offsets64::R26, 8),
    ("$27", offsets64::R27, 8),
    ("$28", offsets64::R28, 8),
    ("$29", offsets64::R29, 8),
    ("$30", offsets64::R30, 8),
    ("$31", offsets64::R31, 8),
    // FPU registers (64-bit on MIPS64 VEX state)
    ("f0", offsets64::F0, 8),
    ("f1", offsets64::F1, 8),
    ("f2", offsets64::F2, 8),
    ("f3", offsets64::F3, 8),
    ("f4", offsets64::F4, 8),
    ("f5", offsets64::F5, 8),
    ("f6", offsets64::F6, 8),
    ("f7", offsets64::F7, 8),
    ("f8", offsets64::F8, 8),
    ("f9", offsets64::F9, 8),
    ("f10", offsets64::F10, 8),
    ("f11", offsets64::F11, 8),
    ("f12", offsets64::F12, 8),
    ("f13", offsets64::F13, 8),
    ("f14", offsets64::F14, 8),
    ("f15", offsets64::F15, 8),
    ("f16", offsets64::F16, 8),
    ("f17", offsets64::F17, 8),
    ("f18", offsets64::F18, 8),
    ("f19", offsets64::F19, 8),
    ("f20", offsets64::F20, 8),
    ("f21", offsets64::F21, 8),
    ("f22", offsets64::F22, 8),
    ("f23", offsets64::F23, 8),
    ("f24", offsets64::F24, 8),
    ("f25", offsets64::F25, 8),
    ("f26", offsets64::F26, 8),
    ("f27", offsets64::F27, 8),
    ("f28", offsets64::F28, 8),
    ("f29", offsets64::F29, 8),
    ("f30", offsets64::F30, 8),
    ("f31", offsets64::F31, 8),
    ("$f0", offsets64::F0, 8),
    ("$f1", offsets64::F1, 8),
    ("$f2", offsets64::F2, 8),
    ("$f3", offsets64::F3, 8),
    ("$f4", offsets64::F4, 8),
    ("$f5", offsets64::F5, 8),
    ("$f6", offsets64::F6, 8),
    ("$f7", offsets64::F7, 8),
    ("$f8", offsets64::F8, 8),
    ("$f9", offsets64::F9, 8),
    ("$f10", offsets64::F10, 8),
    ("$f11", offsets64::F11, 8),
    ("$f12", offsets64::F12, 8),
    ("$f13", offsets64::F13, 8),
    ("$f14", offsets64::F14, 8),
    ("$f15", offsets64::F15, 8),
    ("$f16", offsets64::F16, 8),
    ("$f17", offsets64::F17, 8),
    ("$f18", offsets64::F18, 8),
    ("$f19", offsets64::F19, 8),
    ("$f20", offsets64::F20, 8),
    ("$f21", offsets64::F21, 8),
    ("$f22", offsets64::F22, 8),
    ("$f23", offsets64::F23, 8),
    ("$f24", offsets64::F24, 8),
    ("$f25", offsets64::F25, 8),
    ("$f26", offsets64::F26, 8),
    ("$f27", offsets64::F27, 8),
    ("$f28", offsets64::F28, 8),
    ("$f29", offsets64::F29, 8),
    ("$f30", offsets64::F30, 8),
    ("$f31", offsets64::F31, 8),
    // FPU control registers (32-bit)
    ("fir", offsets64::FIR, 4),
    ("fccr", offsets64::FCCR, 4),
    ("fexr", offsets64::FEXR, 4),
    ("fenr", offsets64::FENR, 4),
    ("fcsr", offsets64::FCSR, 4),
];

const REGISTER_NAMES_MIPS64: &[&str] = &[
    "zero", "at", "v0", "v1", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "t4", "t5", "t6",
    "t7", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "t8", "t9", "k0", "k1", "gp", "sp", "fp",
    "ra", "pc", "hi", "lo",
];

impl Arch for MIPS32 {
    fn vex_arch(&self) -> VexArch {
        VexArch::MIPS32
    }

    fn name(&self) -> &'static str {
        "MIPS32"
    }

    fn bits(&self) -> u32 {
        32
    }

    fn state_size(&self) -> usize {
        offsets32::GUEST_STATE_SIZE
    }

    fn ip_offset(&self) -> u32 {
        offsets32::PC
    }

    fn sp_offset(&self) -> u32 {
        offsets32::R29
    }

    fn bp_offset(&self) -> Option<u32> {
        Some(offsets32::R30) // fp/s8
    }

    impl_arch_registers!(CANONICAL_MIPS32, ALIASES_MIPS32, REGISTER_NAMES_MIPS32);

    fn argument_registers(&self) -> &[u32] {
        // O32 ABI: a0-a3
        &[offsets32::R4, offsets32::R5, offsets32::R6, offsets32::R7]
    }

    fn return_register(&self) -> u32 {
        offsets32::R2 // v0
    }

    fn syscall_num_offset(&self) -> Option<u32> {
        // O32 Linux puts the syscall number in v0 ($2)
        Some(offsets32::R2)
    }

    fn is_little_endian(&self) -> bool {
        true // MIPS can be either, default to little
    }
}

impl Arch for MIPS64 {
    fn vex_arch(&self) -> VexArch {
        VexArch::MIPS64
    }

    fn name(&self) -> &'static str {
        "MIPS64"
    }

    fn bits(&self) -> u32 {
        64
    }

    fn state_size(&self) -> usize {
        offsets64::GUEST_STATE_SIZE
    }

    fn ip_offset(&self) -> u32 {
        offsets64::PC
    }

    fn sp_offset(&self) -> u32 {
        offsets64::R29
    }

    fn bp_offset(&self) -> Option<u32> {
        Some(offsets64::R30)
    }

    impl_arch_registers!(CANONICAL_MIPS64, ALIASES_MIPS64, REGISTER_NAMES_MIPS64);

    fn argument_registers(&self) -> &[u32] {
        // N64 ABI: a0-a7
        &[
            offsets64::R4,
            offsets64::R5,
            offsets64::R6,
            offsets64::R7,
            offsets64::R8,
            offsets64::R9,
            offsets64::R10,
            offsets64::R11,
        ]
    }

    fn return_register(&self) -> u32 {
        offsets64::R2
    }

    fn syscall_num_offset(&self) -> Option<u32> {
        // N64 Linux puts the syscall number in v0 ($2)
        Some(offsets64::R2)
    }

    fn is_little_endian(&self) -> bool {
        true
    }
}

#[cfg(test)]
#[path = "mips_tests.rs"]
mod tests;
