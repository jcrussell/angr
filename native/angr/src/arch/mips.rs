//! MIPS architecture definitions.
//!
//! This module provides the register layout and architecture-specific
//! information for MIPS32 and MIPS64.

use super::Arch;
use crate::vex::VexArch;

/// MIPS32 architecture.
#[derive(Debug, Clone, Copy)]
pub struct MIPS32;

/// MIPS64 architecture.
#[derive(Debug, Clone, Copy)]
pub struct MIPS64;

// MIPS32 VEX guest state offsets (from VEX/pub/libvex_guest_mips32.h)
mod offsets32 {
    pub const R0: u32 = 8;    // zero
    pub const R1: u32 = 12;   // at
    pub const R2: u32 = 16;   // v0
    pub const R3: u32 = 20;   // v1
    pub const R4: u32 = 24;   // a0
    pub const R5: u32 = 28;   // a1
    pub const R6: u32 = 32;   // a2
    pub const R7: u32 = 36;   // a3
    pub const R8: u32 = 40;   // t0
    pub const R9: u32 = 44;   // t1
    pub const R10: u32 = 48;  // t2
    pub const R11: u32 = 52;  // t3
    pub const R12: u32 = 56;  // t4
    pub const R13: u32 = 60;  // t5
    pub const R14: u32 = 64;  // t6
    pub const R15: u32 = 68;  // t7
    pub const R16: u32 = 72;  // s0
    pub const R17: u32 = 76;  // s1
    pub const R18: u32 = 80;  // s2
    pub const R19: u32 = 84;  // s3
    pub const R20: u32 = 88;  // s4
    pub const R21: u32 = 92;  // s5
    pub const R22: u32 = 96;  // s6
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
    pub const R0: u32 = 16;   // zero
    pub const R1: u32 = 24;   // at
    pub const R2: u32 = 32;   // v0
    pub const R3: u32 = 40;   // v1
    pub const R4: u32 = 48;   // a0
    pub const R5: u32 = 56;   // a1
    pub const R6: u32 = 64;   // a2
    pub const R7: u32 = 72;   // a3
    pub const R8: u32 = 80;   // t0/a4
    pub const R9: u32 = 88;   // t1/a5
    pub const R10: u32 = 96;  // t2/a6
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

    // FPU registers (64-bit each)
    pub const F0: u32 = 304;
    pub const F1: u32 = 312;
    pub const F2: u32 = 320;
    pub const F3: u32 = 328;
    pub const F4: u32 = 336;
    pub const F5: u32 = 344;
    pub const F6: u32 = 352;
    pub const F7: u32 = 360;
    pub const F8: u32 = 368;
    pub const F9: u32 = 376;
    pub const F10: u32 = 384;
    pub const F11: u32 = 392;
    pub const F12: u32 = 400;
    pub const F13: u32 = 408;
    pub const F14: u32 = 416;
    pub const F15: u32 = 424;
    pub const F16: u32 = 432;
    pub const F17: u32 = 440;
    pub const F18: u32 = 448;
    pub const F19: u32 = 456;
    pub const F20: u32 = 464;
    pub const F21: u32 = 472;
    pub const F22: u32 = 480;
    pub const F23: u32 = 488;
    pub const F24: u32 = 496;
    pub const F25: u32 = 504;
    pub const F26: u32 = 512;
    pub const F27: u32 = 520;
    pub const F28: u32 = 528;
    pub const F29: u32 = 536;
    pub const F30: u32 = 544;
    pub const F31: u32 = 552;

    pub const FIR: u32 = 560;
    pub const FCCR: u32 = 564;
    pub const FEXR: u32 = 568;
    pub const FENR: u32 = 572;
    pub const FCSR: u32 = 576;

    pub const GUEST_STATE_SIZE: usize = 592;
}

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

    fn register_offset(&self, name: &str) -> Option<u32> {
        let name_lower = name.to_lowercase();
        match name_lower.as_str() {
            // Numeric names
            "r0" | "zero" | "$0" => Some(offsets32::R0),
            "r1" | "at" | "$1" => Some(offsets32::R1),
            "r2" | "v0" | "$2" => Some(offsets32::R2),
            "r3" | "v1" | "$3" => Some(offsets32::R3),
            "r4" | "a0" | "$4" => Some(offsets32::R4),
            "r5" | "a1" | "$5" => Some(offsets32::R5),
            "r6" | "a2" | "$6" => Some(offsets32::R6),
            "r7" | "a3" | "$7" => Some(offsets32::R7),
            "r8" | "t0" | "$8" => Some(offsets32::R8),
            "r9" | "t1" | "$9" => Some(offsets32::R9),
            "r10" | "t2" | "$10" => Some(offsets32::R10),
            "r11" | "t3" | "$11" => Some(offsets32::R11),
            "r12" | "t4" | "$12" => Some(offsets32::R12),
            "r13" | "t5" | "$13" => Some(offsets32::R13),
            "r14" | "t6" | "$14" => Some(offsets32::R14),
            "r15" | "t7" | "$15" => Some(offsets32::R15),
            "r16" | "s0" | "$16" => Some(offsets32::R16),
            "r17" | "s1" | "$17" => Some(offsets32::R17),
            "r18" | "s2" | "$18" => Some(offsets32::R18),
            "r19" | "s3" | "$19" => Some(offsets32::R19),
            "r20" | "s4" | "$20" => Some(offsets32::R20),
            "r21" | "s5" | "$21" => Some(offsets32::R21),
            "r22" | "s6" | "$22" => Some(offsets32::R22),
            "r23" | "s7" | "$23" => Some(offsets32::R23),
            "r24" | "t8" | "$24" => Some(offsets32::R24),
            "r25" | "t9" | "$25" => Some(offsets32::R25),
            "r26" | "k0" | "$26" => Some(offsets32::R26),
            "r27" | "k1" | "$27" => Some(offsets32::R27),
            "r28" | "gp" | "$28" => Some(offsets32::R28),
            "r29" | "sp" | "$29" => Some(offsets32::R29),
            "r30" | "fp" | "s8" | "$30" => Some(offsets32::R30),
            "r31" | "ra" | "$31" => Some(offsets32::R31),

            "pc" => Some(offsets32::PC),
            "hi" => Some(offsets32::HI),
            "lo" => Some(offsets32::LO),

            // FPU registers
            "f0" | "$f0" => Some(offsets32::F0),
            "f1" | "$f1" => Some(offsets32::F1),
            "f2" | "$f2" => Some(offsets32::F2),
            "f3" | "$f3" => Some(offsets32::F3),
            "f4" | "$f4" => Some(offsets32::F4),
            "f5" | "$f5" => Some(offsets32::F5),
            "f6" | "$f6" => Some(offsets32::F6),
            "f7" | "$f7" => Some(offsets32::F7),
            "f8" | "$f8" => Some(offsets32::F8),
            "f9" | "$f9" => Some(offsets32::F9),
            "f10" | "$f10" => Some(offsets32::F10),
            "f11" | "$f11" => Some(offsets32::F11),
            "f12" | "$f12" => Some(offsets32::F12),
            "f13" | "$f13" => Some(offsets32::F13),
            "f14" | "$f14" => Some(offsets32::F14),
            "f15" | "$f15" => Some(offsets32::F15),
            "f16" | "$f16" => Some(offsets32::F16),
            "f17" | "$f17" => Some(offsets32::F17),
            "f18" | "$f18" => Some(offsets32::F18),
            "f19" | "$f19" => Some(offsets32::F19),
            "f20" | "$f20" => Some(offsets32::F20),
            "f21" | "$f21" => Some(offsets32::F21),
            "f22" | "$f22" => Some(offsets32::F22),
            "f23" | "$f23" => Some(offsets32::F23),
            "f24" | "$f24" => Some(offsets32::F24),
            "f25" | "$f25" => Some(offsets32::F25),
            "f26" | "$f26" => Some(offsets32::F26),
            "f27" | "$f27" => Some(offsets32::F27),
            "f28" | "$f28" => Some(offsets32::F28),
            "f29" | "$f29" => Some(offsets32::F29),
            "f30" | "$f30" => Some(offsets32::F30),
            "f31" | "$f31" => Some(offsets32::F31),

            "fir" => Some(offsets32::FIR),
            "fccr" => Some(offsets32::FCCR),
            "fexr" => Some(offsets32::FEXR),
            "fenr" => Some(offsets32::FENR),
            "fcsr" => Some(offsets32::FCSR),

            _ => None,
        }
    }

    fn register_size(&self, name: &str) -> Option<u32> {
        let name_lower = name.to_lowercase();
        match name_lower.as_str() {
            // All GPRs are 32-bit
            "r0" | "zero" | "$0" | "r1" | "at" | "$1" | "r2" | "v0" | "$2" | "r3" | "v1"
            | "$3" | "r4" | "a0" | "$4" | "r5" | "a1" | "$5" | "r6" | "a2" | "$6" | "r7"
            | "a3" | "$7" | "r8" | "t0" | "$8" | "r9" | "t1" | "$9" | "r10" | "t2" | "$10"
            | "r11" | "t3" | "$11" | "r12" | "t4" | "$12" | "r13" | "t5" | "$13" | "r14"
            | "t6" | "$14" | "r15" | "t7" | "$15" | "r16" | "s0" | "$16" | "r17" | "s1"
            | "$17" | "r18" | "s2" | "$18" | "r19" | "s3" | "$19" | "r20" | "s4" | "$20"
            | "r21" | "s5" | "$21" | "r22" | "s6" | "$22" | "r23" | "s7" | "$23" | "r24"
            | "t8" | "$24" | "r25" | "t9" | "$25" | "r26" | "k0" | "$26" | "r27" | "k1"
            | "$27" | "r28" | "gp" | "$28" | "r29" | "sp" | "$29" | "r30" | "fp" | "s8"
            | "$30" | "r31" | "ra" | "$31" | "pc" | "hi" | "lo" => Some(4),

            // FPU registers are 64-bit
            "f0" | "$f0" | "f1" | "$f1" | "f2" | "$f2" | "f3" | "$f3" | "f4" | "$f4" | "f5"
            | "$f5" | "f6" | "$f6" | "f7" | "$f7" | "f8" | "$f8" | "f9" | "$f9" | "f10"
            | "$f10" | "f11" | "$f11" | "f12" | "$f12" | "f13" | "$f13" | "f14" | "$f14"
            | "f15" | "$f15" | "f16" | "$f16" | "f17" | "$f17" | "f18" | "$f18" | "f19"
            | "$f19" | "f20" | "$f20" | "f21" | "$f21" | "f22" | "$f22" | "f23" | "$f23"
            | "f24" | "$f24" | "f25" | "$f25" | "f26" | "$f26" | "f27" | "$f27" | "f28"
            | "$f28" | "f29" | "$f29" | "f30" | "$f30" | "f31" | "$f31" => Some(8),

            "fir" | "fccr" | "fexr" | "fenr" | "fcsr" => Some(4),

            _ => None,
        }
    }

    fn register_name(&self, offset: u32) -> Option<&'static str> {
        match offset {
            offsets32::R0 => Some("zero"),
            offsets32::R1 => Some("at"),
            offsets32::R2 => Some("v0"),
            offsets32::R3 => Some("v1"),
            offsets32::R4 => Some("a0"),
            offsets32::R5 => Some("a1"),
            offsets32::R6 => Some("a2"),
            offsets32::R7 => Some("a3"),
            offsets32::R8 => Some("t0"),
            offsets32::R9 => Some("t1"),
            offsets32::R10 => Some("t2"),
            offsets32::R11 => Some("t3"),
            offsets32::R12 => Some("t4"),
            offsets32::R13 => Some("t5"),
            offsets32::R14 => Some("t6"),
            offsets32::R15 => Some("t7"),
            offsets32::R16 => Some("s0"),
            offsets32::R17 => Some("s1"),
            offsets32::R18 => Some("s2"),
            offsets32::R19 => Some("s3"),
            offsets32::R20 => Some("s4"),
            offsets32::R21 => Some("s5"),
            offsets32::R22 => Some("s6"),
            offsets32::R23 => Some("s7"),
            offsets32::R24 => Some("t8"),
            offsets32::R25 => Some("t9"),
            offsets32::R26 => Some("k0"),
            offsets32::R27 => Some("k1"),
            offsets32::R28 => Some("gp"),
            offsets32::R29 => Some("sp"),
            offsets32::R30 => Some("fp"),
            offsets32::R31 => Some("ra"),
            offsets32::PC => Some("pc"),
            offsets32::HI => Some("hi"),
            offsets32::LO => Some("lo"),
            _ => None,
        }
    }

    fn register_names(&self) -> &[&'static str] {
        &[
            "zero", "at", "v0", "v1", "a0", "a1", "a2", "a3", "t0", "t1", "t2", "t3", "t4",
            "t5", "t6", "t7", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "t8", "t9",
            "k0", "k1", "gp", "sp", "fp", "ra", "pc", "hi", "lo",
        ]
    }

    fn argument_registers(&self) -> &[u32] {
        // O32 ABI: a0-a3
        &[
            offsets32::R4,
            offsets32::R5,
            offsets32::R6,
            offsets32::R7,
        ]
    }

    fn return_register(&self) -> u32 {
        offsets32::R2 // v0
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

    fn register_offset(&self, name: &str) -> Option<u32> {
        let name_lower = name.to_lowercase();
        match name_lower.as_str() {
            "r0" | "zero" | "$0" => Some(offsets64::R0),
            "r1" | "at" | "$1" => Some(offsets64::R1),
            "r2" | "v0" | "$2" => Some(offsets64::R2),
            "r3" | "v1" | "$3" => Some(offsets64::R3),
            "r4" | "a0" | "$4" => Some(offsets64::R4),
            "r5" | "a1" | "$5" => Some(offsets64::R5),
            "r6" | "a2" | "$6" => Some(offsets64::R6),
            "r7" | "a3" | "$7" => Some(offsets64::R7),
            "r8" | "a4" | "t0" | "$8" => Some(offsets64::R8),
            "r9" | "a5" | "t1" | "$9" => Some(offsets64::R9),
            "r10" | "a6" | "t2" | "$10" => Some(offsets64::R10),
            "r11" | "a7" | "t3" | "$11" => Some(offsets64::R11),
            "r12" | "t4" | "$12" => Some(offsets64::R12),
            "r13" | "t5" | "$13" => Some(offsets64::R13),
            "r14" | "t6" | "$14" => Some(offsets64::R14),
            "r15" | "t7" | "$15" => Some(offsets64::R15),
            "r16" | "s0" | "$16" => Some(offsets64::R16),
            "r17" | "s1" | "$17" => Some(offsets64::R17),
            "r18" | "s2" | "$18" => Some(offsets64::R18),
            "r19" | "s3" | "$19" => Some(offsets64::R19),
            "r20" | "s4" | "$20" => Some(offsets64::R20),
            "r21" | "s5" | "$21" => Some(offsets64::R21),
            "r22" | "s6" | "$22" => Some(offsets64::R22),
            "r23" | "s7" | "$23" => Some(offsets64::R23),
            "r24" | "t8" | "$24" => Some(offsets64::R24),
            "r25" | "t9" | "$25" => Some(offsets64::R25),
            "r26" | "k0" | "$26" => Some(offsets64::R26),
            "r27" | "k1" | "$27" => Some(offsets64::R27),
            "r28" | "gp" | "$28" => Some(offsets64::R28),
            "r29" | "sp" | "$29" => Some(offsets64::R29),
            "r30" | "fp" | "s8" | "$30" => Some(offsets64::R30),
            "r31" | "ra" | "$31" => Some(offsets64::R31),

            "pc" => Some(offsets64::PC),
            "hi" => Some(offsets64::HI),
            "lo" => Some(offsets64::LO),

            _ => None,
        }
    }

    fn register_size(&self, name: &str) -> Option<u32> {
        let name_lower = name.to_lowercase();
        match name_lower.as_str() {
            // All GPRs are 64-bit
            "r0" | "zero" | "$0" | "r1" | "at" | "$1" | "r2" | "v0" | "$2" | "r3" | "v1"
            | "$3" | "r4" | "a0" | "$4" | "r5" | "a1" | "$5" | "r6" | "a2" | "$6" | "r7"
            | "a3" | "$7" | "r8" | "a4" | "t0" | "$8" | "r9" | "a5" | "t1" | "$9" | "r10"
            | "a6" | "t2" | "$10" | "r11" | "a7" | "t3" | "$11" | "r12" | "t4" | "$12"
            | "r13" | "t5" | "$13" | "r14" | "t6" | "$14" | "r15" | "t7" | "$15" | "r16"
            | "s0" | "$16" | "r17" | "s1" | "$17" | "r18" | "s2" | "$18" | "r19" | "s3"
            | "$19" | "r20" | "s4" | "$20" | "r21" | "s5" | "$21" | "r22" | "s6" | "$22"
            | "r23" | "s7" | "$23" | "r24" | "t8" | "$24" | "r25" | "t9" | "$25" | "r26"
            | "k0" | "$26" | "r27" | "k1" | "$27" | "r28" | "gp" | "$28" | "r29" | "sp"
            | "$29" | "r30" | "fp" | "s8" | "$30" | "r31" | "ra" | "$31" | "pc" | "hi"
            | "lo" => Some(8),

            _ => None,
        }
    }

    fn register_name(&self, offset: u32) -> Option<&'static str> {
        match offset {
            offsets64::R0 => Some("zero"),
            offsets64::R2 => Some("v0"),
            offsets64::R4 => Some("a0"),
            offsets64::R29 => Some("sp"),
            offsets64::R30 => Some("fp"),
            offsets64::R31 => Some("ra"),
            offsets64::PC => Some("pc"),
            _ => None,
        }
    }

    fn register_names(&self) -> &[&'static str] {
        &[
            "zero", "at", "v0", "v1", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "t4",
            "t5", "t6", "t7", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "t8", "t9",
            "k0", "k1", "gp", "sp", "fp", "ra", "pc", "hi", "lo",
        ]
    }

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

    fn is_little_endian(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mips32_basics() {
        let arch = MIPS32;

        assert_eq!(arch.bits(), 32);
        assert_eq!(arch.name(), "MIPS32");
    }

    #[test]
    fn test_mips64_basics() {
        let arch = MIPS64;

        assert_eq!(arch.bits(), 64);
        assert_eq!(arch.name(), "MIPS64");
    }

    #[test]
    fn test_mips32_register_lookup() {
        let arch = MIPS32;

        assert_eq!(arch.register_offset("v0"), Some(16));
        assert_eq!(arch.register_offset("$2"), Some(16));
        assert_eq!(arch.register_offset("r2"), Some(16));

        assert_eq!(arch.register_offset("sp"), Some(124));
        assert_eq!(arch.register_offset("$29"), Some(124));

        assert_eq!(arch.register_size("v0"), Some(4));
    }

    #[test]
    fn test_mips64_register_lookup() {
        let arch = MIPS64;

        assert_eq!(arch.register_offset("v0"), Some(32));
        assert_eq!(arch.register_size("v0"), Some(8));
    }
}
