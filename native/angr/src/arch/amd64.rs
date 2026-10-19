//! AMD64 (x86-64) architecture definition.
//!
//! This module provides the register layout and architecture-specific
//! information for AMD64.

use super::{Arch, RegEntry, impl_arch_registers};
use crate::vex::VexArch;

/// AMD64 architecture.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AMD64;

// AMD64 VEX guest state offsets (from VEX/pub/libvex_guest_amd64.h)
// These match the VexGuestAMD64State structure layout.
pub(crate) mod offsets {
    pub(crate) const RAX: u32 = 16;
    pub(crate) const RCX: u32 = 24;
    pub(crate) const RDX: u32 = 32;
    pub(crate) const RBX: u32 = 40;
    pub(crate) const RSP: u32 = 48;
    pub(crate) const RBP: u32 = 56;
    pub(crate) const RSI: u32 = 64;
    pub(crate) const RDI: u32 = 72;
    pub(crate) const R8: u32 = 80;
    pub(crate) const R9: u32 = 88;
    pub(crate) const R10: u32 = 96;
    pub(crate) const R11: u32 = 104;
    pub(crate) const R12: u32 = 112;
    pub(crate) const R13: u32 = 120;
    pub(crate) const R14: u32 = 128;
    pub(crate) const R15: u32 = 136;

    // Flags
    pub(crate) const CC_OP: u32 = 144;
    pub(crate) const CC_DEP1: u32 = 152;
    pub(crate) const CC_DEP2: u32 = 160;
    pub(crate) const CC_NDEP: u32 = 168;

    // Other
    pub(crate) const DFLAG: u32 = 176;
    pub(crate) const RIP: u32 = 184;
    pub(crate) const ACFLAG: u32 = 192;
    pub(crate) const IDFLAG: u32 = 200;

    // Segment registers. FS_CONST is the FS base address (archinfo offset 208);
    // GS_CONST is the GS base, which archinfo places well past the vector/FPU
    // bank (see below). The 216 slot belongs to SSEROUND, not GS_CONST — an
    // earlier layout wrongly defined GS_CONST there, aliasing the two.
    pub(crate) const FS_CONST: u32 = 208;

    // SSE control (per archinfo: 4B uint32_t padded to 8B in archinfo's table)
    pub(crate) const SSEROUND: u32 = 216;

    // XMM registers (128-bit each). VEX lays the vector bank out as 32
    // 512-bit ZMM slots (AVX-512 guest state, pyvex 10): XMMn is the low 16
    // bytes of ZMMn, so the stride is 64.
    pub(crate) const XMM0: u32 = 224;
    pub(crate) const XMM1: u32 = 288;
    pub(crate) const XMM2: u32 = 352;
    pub(crate) const XMM3: u32 = 416;
    pub(crate) const XMM4: u32 = 480;
    pub(crate) const XMM5: u32 = 544;
    pub(crate) const XMM6: u32 = 608;
    pub(crate) const XMM7: u32 = 672;
    pub(crate) const XMM8: u32 = 736;
    pub(crate) const XMM9: u32 = 800;
    pub(crate) const XMM10: u32 = 864;
    pub(crate) const XMM11: u32 = 928;
    pub(crate) const XMM12: u32 = 992;
    pub(crate) const XMM13: u32 = 1056;
    pub(crate) const XMM14: u32 = 1120;
    pub(crate) const XMM15: u32 = 1184;
    pub(crate) const XMM16: u32 = 1248;
    pub(crate) const XMM17: u32 = 1312;
    pub(crate) const XMM18: u32 = 1376;
    pub(crate) const XMM19: u32 = 1440;
    pub(crate) const XMM20: u32 = 1504;
    pub(crate) const XMM21: u32 = 1568;
    pub(crate) const XMM22: u32 = 1632;
    pub(crate) const XMM23: u32 = 1696;
    pub(crate) const XMM24: u32 = 1760;
    pub(crate) const XMM25: u32 = 1824;
    pub(crate) const XMM26: u32 = 1888;
    pub(crate) const XMM27: u32 = 1952;
    pub(crate) const XMM28: u32 = 2016;
    pub(crate) const XMM29: u32 = 2080;
    pub(crate) const XMM30: u32 = 2144;
    pub(crate) const XMM31: u32 = 2208;

    // AVX-512 opmask registers k0-k7 (8 bytes each), after the ZMM bank.
    pub(crate) const K0: u32 = 2336;
    pub(crate) const K1: u32 = 2344;
    pub(crate) const K2: u32 = 2352;
    pub(crate) const K3: u32 = 2360;
    pub(crate) const K4: u32 = 2368;
    pub(crate) const K5: u32 = 2376;
    pub(crate) const K6: u32 = 2384;
    pub(crate) const K7: u32 = 2392;

    // FPU state (per archinfo), after the 32 ZMM slots and the K0-K7 opmask
    // registers (2336..2400).
    pub(crate) const FTOP: u32 = 2400;
    // VEX stores the x87 stack as `guest_FPREG[8]` of `ULong`, i.e. 8 x 8-byte
    // slots (64 bytes total, ending where FPTAG starts) -- MMX-style 64-bit
    // storage, not true 80-bit x87 extended precision. Per-index offset is
    // therefore `FPREG + n * 8`.
    pub(crate) const FPREG: u32 = 2408;
    pub(crate) const FPTAG: u32 = 2472;
    pub(crate) const FPROUND: u32 = 2480;
    pub(crate) const FC3210: u32 = 2488;

    // Late VEX state (per archinfo): the emulation-note word and the
    // self-modifying-code / redirect bookkeeping fields precede the real
    // GS_CONST slot. `guest_EMNOTE` is a UInt (4 bytes) with 4 bytes of
    // padding after it; the rest are ULongs. Named for parity with
    // X86/ARM/ARM64 (angr-9ke6b.10).
    pub(crate) const EMNOTE: u32 = 2496;
    pub(crate) const CMSTART: u32 = 2504;
    pub(crate) const CMLEN: u32 = 2512;
    pub(crate) const NRADDR: u32 = 2520;
    pub(crate) const GS_CONST: u32 = 2536;
    pub(crate) const IP_AT_SYSCALL: u32 = 2544;

    // Total guest state size: sizeof(VexGuestAMD64State) in the AVX-512 build
    // pyvex ships. archinfo's last register (cr8) ends at 2648; the control
    // register tail runs to the end, and IR-level PUTs into it must not be
    // dropped by RegisterFile::put.
    pub(crate) const GUEST_STATE_SIZE: usize = 2704;
}

// Canonical registers: each entry is `(name, offset, size_bytes)`. These
// drive `register_name(offset)` reverse lookups.
const CANONICAL: &[RegEntry] = &[
    // 64-bit GPRs
    ("rax", offsets::RAX, 8),
    ("rcx", offsets::RCX, 8),
    ("rdx", offsets::RDX, 8),
    ("rbx", offsets::RBX, 8),
    ("rsp", offsets::RSP, 8),
    ("rbp", offsets::RBP, 8),
    ("rsi", offsets::RSI, 8),
    ("rdi", offsets::RDI, 8),
    ("r8", offsets::R8, 8),
    ("r9", offsets::R9, 8),
    ("r10", offsets::R10, 8),
    ("r11", offsets::R11, 8),
    ("r12", offsets::R12, 8),
    ("r13", offsets::R13, 8),
    ("r14", offsets::R14, 8),
    ("r15", offsets::R15, 8),
    ("rip", offsets::RIP, 8),
    // Flags thunks (64-bit)
    ("cc_op", offsets::CC_OP, 8),
    ("cc_dep1", offsets::CC_DEP1, 8),
    ("cc_dep2", offsets::CC_DEP2, 8),
    ("cc_ndep", offsets::CC_NDEP, 8),
    ("dflag", offsets::DFLAG, 8),
    ("acflag", offsets::ACFLAG, 8),
    ("idflag", offsets::IDFLAG, 8),
    // Segments
    ("fs_const", offsets::FS_CONST, 8),
    ("gs_const", offsets::GS_CONST, 8),
    // SSE control register (8B per archinfo)
    ("sseround", offsets::SSEROUND, 8),
    // XMM registers (128-bit)
    ("xmm0", offsets::XMM0, 16),
    ("xmm1", offsets::XMM1, 16),
    ("xmm2", offsets::XMM2, 16),
    ("xmm3", offsets::XMM3, 16),
    ("xmm4", offsets::XMM4, 16),
    ("xmm5", offsets::XMM5, 16),
    ("xmm6", offsets::XMM6, 16),
    ("xmm7", offsets::XMM7, 16),
    ("xmm8", offsets::XMM8, 16),
    ("xmm9", offsets::XMM9, 16),
    ("xmm10", offsets::XMM10, 16),
    ("xmm11", offsets::XMM11, 16),
    ("xmm12", offsets::XMM12, 16),
    ("xmm13", offsets::XMM13, 16),
    ("xmm14", offsets::XMM14, 16),
    ("xmm15", offsets::XMM15, 16),
    // AVX-512 vector registers 16-31 (low 128 bits) and opmasks
    ("xmm16", offsets::XMM16, 16),
    ("xmm17", offsets::XMM17, 16),
    ("xmm18", offsets::XMM18, 16),
    ("xmm19", offsets::XMM19, 16),
    ("xmm20", offsets::XMM20, 16),
    ("xmm21", offsets::XMM21, 16),
    ("xmm22", offsets::XMM22, 16),
    ("xmm23", offsets::XMM23, 16),
    ("xmm24", offsets::XMM24, 16),
    ("xmm25", offsets::XMM25, 16),
    ("xmm26", offsets::XMM26, 16),
    ("xmm27", offsets::XMM27, 16),
    ("xmm28", offsets::XMM28, 16),
    ("xmm29", offsets::XMM29, 16),
    ("xmm30", offsets::XMM30, 16),
    ("xmm31", offsets::XMM31, 16),
    ("k0", offsets::K0, 8),
    ("k1", offsets::K1, 8),
    ("k2", offsets::K2, 8),
    ("k3", offsets::K3, 8),
    ("k4", offsets::K4, 8),
    ("k5", offsets::K5, 8),
    ("k6", offsets::K6, 8),
    ("k7", offsets::K7, 8),
    // FPU
    ("fpreg", offsets::FPREG, 64),
    ("fptag", offsets::FPTAG, 8),
    ("fpround", offsets::FPROUND, 8),
    ("fc3210", offsets::FC3210, 8),
    // guest_FTOP is a UInt in VEX's amd64 guest state, not a ULong.
    ("ftop", offsets::FTOP, 4),
];

// Aliases: alternate names or sub-registers sharing a canonical offset.
// `register_offset` and `register_size` consult these as a fallback;
// `register_name` does not.
const ALIASES: &[RegEntry] = &[
    // Architecture-independent full-width aliases (match archinfo).
    ("sp", offsets::RSP, 8),
    ("bp", offsets::RBP, 8),
    // The two architecture-independent instruction-pointer spellings, matching
    // the ARM/ARM64/MIPS tables (angr-9ke6b.217) and archinfo's ArchAMD64
    // pc=(184,8)/ip=(184,8). `ip` is the *instruction pointer* on every arch,
    // never an ABI scratch register — see the angr-itm3u note in arch/arm.rs
    // (angr-690nc).
    ("pc", offsets::RIP, 8),
    ("ip", offsets::RIP, 8),
    // 32-bit sub-registers (low 32 of RAX etc.)
    ("eax", offsets::RAX, 4),
    ("ecx", offsets::RCX, 4),
    ("edx", offsets::RDX, 4),
    ("ebx", offsets::RBX, 4),
    ("esp", offsets::RSP, 4),
    ("ebp", offsets::RBP, 4),
    ("esi", offsets::RSI, 4),
    ("edi", offsets::RDI, 4),
    ("r8d", offsets::R8, 4),
    ("r9d", offsets::R9, 4),
    ("r10d", offsets::R10, 4),
    ("r11d", offsets::R11, 4),
    ("r12d", offsets::R12, 4),
    ("r13d", offsets::R13, 4),
    ("r14d", offsets::R14, 4),
    ("r15d", offsets::R15, 4),
    ("eip", offsets::RIP, 4),
    // 16-bit
    ("ax", offsets::RAX, 2),
    ("cx", offsets::RCX, 2),
    ("dx", offsets::RDX, 2),
    ("bx", offsets::RBX, 2),
    // NB: no 16-bit `sp`/`bp` entries. archinfo binds those two names to the
    // architecture-independent full-width stack/base pointer (AMD64 `sp` is
    // (48, 8)), so they live in the 64-bit block below; the legacy 16-bit
    // sub-registers are unreachable by name here, exactly as in archinfo.
    ("si", offsets::RSI, 2),
    ("di", offsets::RDI, 2),
    // 8-bit low
    ("al", offsets::RAX, 1),
    ("cl", offsets::RCX, 1),
    ("dl", offsets::RDX, 1),
    ("bl", offsets::RBX, 1),
    ("spl", offsets::RSP, 1),
    ("bpl", offsets::RBP, 1),
    ("sil", offsets::RSI, 1),
    ("dil", offsets::RDI, 1),
    ("r8b", offsets::R8, 1),
    ("r9b", offsets::R9, 1),
    ("r10b", offsets::R10, 1),
    ("r11b", offsets::R11, 1),
    ("r12b", offsets::R12, 1),
    ("r13b", offsets::R13, 1),
    ("r14b", offsets::R14, 1),
    ("r15b", offsets::R15, 1),
    // 8-bit high (offset+1)
    ("ah", offsets::RAX + 1, 1),
    ("ch", offsets::RCX + 1, 1),
    ("dh", offsets::RDX + 1, 1),
    ("bh", offsets::RBX + 1, 1),
    // Flag aliases
    ("d", offsets::DFLAG, 8),
    ("ac", offsets::ACFLAG, 8),
    ("id", offsets::IDFLAG, 8),
    // Segment aliases (the base registers, not the 16-bit selectors)
    ("fs", offsets::FS_CONST, 8),
    ("gs", offsets::GS_CONST, 8),
    // VEX bookkeeping tail (see the offsets note above). guest_EMNOTE is a
    // UInt; the rest are ULongs.
    ("emnote", offsets::EMNOTE, 4),
    ("cmstart", offsets::CMSTART, 8),
    ("cmlen", offsets::CMLEN, 8),
    ("nraddr", offsets::NRADDR, 8),
    ("ip_at_syscall", offsets::IP_AT_SYSCALL, 8),
    // AVX-512 views of the ZMM bank, laid out as archinfo's ArchAMD64 does:
    // zmmN is the whole 64-byte slot and zmmNhy its upper 32 bytes; for N >= 16
    // also ymmN (low 32), ymmNhx (bytes 16..32) and xmmN's 8-byte lq/hq halves.
    ("zmm0", offsets::XMM0, 64),
    ("zmm0hy", offsets::XMM0 + 32, 32),
    ("zmm1", offsets::XMM1, 64),
    ("zmm1hy", offsets::XMM1 + 32, 32),
    ("zmm2", offsets::XMM2, 64),
    ("zmm2hy", offsets::XMM2 + 32, 32),
    ("zmm3", offsets::XMM3, 64),
    ("zmm3hy", offsets::XMM3 + 32, 32),
    ("zmm4", offsets::XMM4, 64),
    ("zmm4hy", offsets::XMM4 + 32, 32),
    ("zmm5", offsets::XMM5, 64),
    ("zmm5hy", offsets::XMM5 + 32, 32),
    ("zmm6", offsets::XMM6, 64),
    ("zmm6hy", offsets::XMM6 + 32, 32),
    ("zmm7", offsets::XMM7, 64),
    ("zmm7hy", offsets::XMM7 + 32, 32),
    ("zmm8", offsets::XMM8, 64),
    ("zmm8hy", offsets::XMM8 + 32, 32),
    ("zmm9", offsets::XMM9, 64),
    ("zmm9hy", offsets::XMM9 + 32, 32),
    ("zmm10", offsets::XMM10, 64),
    ("zmm10hy", offsets::XMM10 + 32, 32),
    ("zmm11", offsets::XMM11, 64),
    ("zmm11hy", offsets::XMM11 + 32, 32),
    ("zmm12", offsets::XMM12, 64),
    ("zmm12hy", offsets::XMM12 + 32, 32),
    ("zmm13", offsets::XMM13, 64),
    ("zmm13hy", offsets::XMM13 + 32, 32),
    ("zmm14", offsets::XMM14, 64),
    ("zmm14hy", offsets::XMM14 + 32, 32),
    ("zmm15", offsets::XMM15, 64),
    ("zmm15hy", offsets::XMM15 + 32, 32),
    ("zmm16", offsets::XMM16, 64),
    ("zmm16hy", offsets::XMM16 + 32, 32),
    ("zmm17", offsets::XMM17, 64),
    ("zmm17hy", offsets::XMM17 + 32, 32),
    ("zmm18", offsets::XMM18, 64),
    ("zmm18hy", offsets::XMM18 + 32, 32),
    ("zmm19", offsets::XMM19, 64),
    ("zmm19hy", offsets::XMM19 + 32, 32),
    ("zmm20", offsets::XMM20, 64),
    ("zmm20hy", offsets::XMM20 + 32, 32),
    ("zmm21", offsets::XMM21, 64),
    ("zmm21hy", offsets::XMM21 + 32, 32),
    ("zmm22", offsets::XMM22, 64),
    ("zmm22hy", offsets::XMM22 + 32, 32),
    ("zmm23", offsets::XMM23, 64),
    ("zmm23hy", offsets::XMM23 + 32, 32),
    ("zmm24", offsets::XMM24, 64),
    ("zmm24hy", offsets::XMM24 + 32, 32),
    ("zmm25", offsets::XMM25, 64),
    ("zmm25hy", offsets::XMM25 + 32, 32),
    ("zmm26", offsets::XMM26, 64),
    ("zmm26hy", offsets::XMM26 + 32, 32),
    ("zmm27", offsets::XMM27, 64),
    ("zmm27hy", offsets::XMM27 + 32, 32),
    ("zmm28", offsets::XMM28, 64),
    ("zmm28hy", offsets::XMM28 + 32, 32),
    ("zmm29", offsets::XMM29, 64),
    ("zmm29hy", offsets::XMM29 + 32, 32),
    ("zmm30", offsets::XMM30, 64),
    ("zmm30hy", offsets::XMM30 + 32, 32),
    ("zmm31", offsets::XMM31, 64),
    ("zmm31hy", offsets::XMM31 + 32, 32),
    ("ymm16", offsets::XMM16, 32),
    ("ymm16hx", offsets::XMM16 + 16, 16),
    ("xmm16lq", offsets::XMM16, 8),
    ("xmm16hq", offsets::XMM16 + 8, 8),
    ("ymm17", offsets::XMM17, 32),
    ("ymm17hx", offsets::XMM17 + 16, 16),
    ("xmm17lq", offsets::XMM17, 8),
    ("xmm17hq", offsets::XMM17 + 8, 8),
    ("ymm18", offsets::XMM18, 32),
    ("ymm18hx", offsets::XMM18 + 16, 16),
    ("xmm18lq", offsets::XMM18, 8),
    ("xmm18hq", offsets::XMM18 + 8, 8),
    ("ymm19", offsets::XMM19, 32),
    ("ymm19hx", offsets::XMM19 + 16, 16),
    ("xmm19lq", offsets::XMM19, 8),
    ("xmm19hq", offsets::XMM19 + 8, 8),
    ("ymm20", offsets::XMM20, 32),
    ("ymm20hx", offsets::XMM20 + 16, 16),
    ("xmm20lq", offsets::XMM20, 8),
    ("xmm20hq", offsets::XMM20 + 8, 8),
    ("ymm21", offsets::XMM21, 32),
    ("ymm21hx", offsets::XMM21 + 16, 16),
    ("xmm21lq", offsets::XMM21, 8),
    ("xmm21hq", offsets::XMM21 + 8, 8),
    ("ymm22", offsets::XMM22, 32),
    ("ymm22hx", offsets::XMM22 + 16, 16),
    ("xmm22lq", offsets::XMM22, 8),
    ("xmm22hq", offsets::XMM22 + 8, 8),
    ("ymm23", offsets::XMM23, 32),
    ("ymm23hx", offsets::XMM23 + 16, 16),
    ("xmm23lq", offsets::XMM23, 8),
    ("xmm23hq", offsets::XMM23 + 8, 8),
    ("ymm24", offsets::XMM24, 32),
    ("ymm24hx", offsets::XMM24 + 16, 16),
    ("xmm24lq", offsets::XMM24, 8),
    ("xmm24hq", offsets::XMM24 + 8, 8),
    ("ymm25", offsets::XMM25, 32),
    ("ymm25hx", offsets::XMM25 + 16, 16),
    ("xmm25lq", offsets::XMM25, 8),
    ("xmm25hq", offsets::XMM25 + 8, 8),
    ("ymm26", offsets::XMM26, 32),
    ("ymm26hx", offsets::XMM26 + 16, 16),
    ("xmm26lq", offsets::XMM26, 8),
    ("xmm26hq", offsets::XMM26 + 8, 8),
    ("ymm27", offsets::XMM27, 32),
    ("ymm27hx", offsets::XMM27 + 16, 16),
    ("xmm27lq", offsets::XMM27, 8),
    ("xmm27hq", offsets::XMM27 + 8, 8),
    ("ymm28", offsets::XMM28, 32),
    ("ymm28hx", offsets::XMM28 + 16, 16),
    ("xmm28lq", offsets::XMM28, 8),
    ("xmm28hq", offsets::XMM28 + 8, 8),
    ("ymm29", offsets::XMM29, 32),
    ("ymm29hx", offsets::XMM29 + 16, 16),
    ("xmm29lq", offsets::XMM29, 8),
    ("xmm29hq", offsets::XMM29 + 8, 8),
    ("ymm30", offsets::XMM30, 32),
    ("ymm30hx", offsets::XMM30 + 16, 16),
    ("xmm30lq", offsets::XMM30, 8),
    ("xmm30hq", offsets::XMM30 + 8, 8),
    ("ymm31", offsets::XMM31, 32),
    ("ymm31hx", offsets::XMM31 + 16, 16),
    ("xmm31lq", offsets::XMM31, 8),
    ("xmm31hq", offsets::XMM31 + 8, 8),
];

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

    impl_arch_registers!(CANONICAL, ALIASES);

    fn syscall_num_offset(&self) -> Option<u32> {
        Some(offsets::RAX)
    }

    fn is_little_endian(&self) -> bool {
        true
    }
}

test_submod!("amd64_tests.rs" => tests);
