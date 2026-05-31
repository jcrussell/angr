//! Calling convention implementations for argument extraction.
//!
//! This module provides calling convention support for extracting function
//! arguments from registers and stack. This is used to pre-extract arguments
//! before returning to Python for SimProcedure execution.

use crate::arch::RegisterFile;
use crate::memory::SymbolicMemory;
use crate::symbolic::{RustBV, SymContext};
use crate::vex::Endness;

/// Errors produced by [`CallingConvention::extract_args`] when the stack
/// portion of the argument list cannot be read.
///
/// Procedures previously relied on silent fabrication (the trait method
/// would mint a fresh `RustBV::symbolic("stack_arg_N", …)` whenever
/// `mem.load_concrete_lazy` failed). That made real stack-setup bugs
/// indistinguishable from intentional symbolic input. Returning an
/// explicit error variant forces callers to decide policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtractionError {
    /// The caller asked for stack-resident arguments but did not supply a
    /// memory view. The trait method has no way to materialise stack
    /// arguments without it.
    MemoryUnavailable,
    /// The stack pointer is symbolic. We cannot compute the stack-slot
    /// addresses without committing to a concrete SP, which would silently
    /// pin the value of a symbol the caller may want to reason about.
    SpSymbolic,
    /// A specific stack slot could not be read from memory (typically
    /// unmapped page or permission failure). `arg_index` is the
    /// zero-based position of the failing argument within the full
    /// `num_args` request; `addr` is the absolute address that failed.
    StackUnmapped { arg_index: usize, addr: u64 },
    /// The caller requested more arguments than the ABI exposes via
    /// registers, but the ABI has no stack path (e.g. syscalls on most
    /// architectures). Indicates a SimProcedure / syscall-handler
    /// misconfiguration declaring a higher `num_args` than the kernel
    /// ABI supports.
    RegisterOverflow { requested: usize, available: usize },
}

impl core::fmt::Display for ExtractionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MemoryUnavailable => write!(
                f,
                "extract_args: stack argument requested without a memory view",
            ),
            Self::SpSymbolic => write!(
                f,
                "extract_args: stack pointer is symbolic; cannot compute stack-slot addresses",
            ),
            Self::StackUnmapped { arg_index, addr } => write!(
                f,
                "extract_args: stack argument {arg_index} at address {addr:#x} is unmapped",
            ),
            Self::RegisterOverflow {
                requested,
                available,
            } => write!(
                f,
                "extract_args: requested {requested} arguments but ABI exposes only {available} registers and no stack path",
            ),
        }
    }
}

impl std::error::Error for ExtractionError {}

/// Calling convention trait for extracting function arguments.
pub trait CallingConvention: Send + Sync {
    /// Get the name of this calling convention.
    fn name(&self) -> &'static str;

    /// Get the register offsets used for integer/pointer arguments.
    fn arg_registers(&self) -> &[u32];

    /// Get the register offsets used for syscall arguments.
    ///
    /// Defaults to `arg_registers()`. Override on architectures where the
    /// syscall ABI differs from the C ABI (e.g. amd64 Linux uses R10 in
    /// place of RCX for the 4th argument).
    fn syscall_arg_registers(&self) -> &[u32] {
        self.arg_registers()
    }

    /// Get the register offsets used for floating-point arguments.
    fn fp_arg_registers(&self) -> &[u32];

    /// Get the size of a pointer in bytes.
    fn pointer_size(&self) -> u32;

    /// Get the stack offset where arguments start (after return address).
    fn stack_arg_offset(&self) -> u64;

    /// Get the endianness for memory access.
    fn endness(&self) -> Endness;

    /// Get the return value register offset.
    fn return_register(&self) -> u32;

    /// Whether a `call`-style instruction pushes the return address onto the
    /// stack (true) or stores it in a link/return register (false).
    ///
    /// x86/AMD64 use stack-based return-addr semantics — `call` pushes
    /// `ret_addr` to `[sp]` and `ret` pops it, so when a native SimProcedure
    /// finishes, the dispatcher must increment SP by `pointer_size`.
    /// ARM/ARM64 use BL which writes the return address into LR (R14/X30) and
    /// MIPS uses JAL which writes it to $ra (R31); for these architectures
    /// the dispatcher must NOT adjust SP after a native SimProcedure returns.
    fn pops_return_addr(&self) -> bool {
        true
    }

    /// Extract up to N arguments from registers and memory.
    ///
    /// Arguments are extracted in order: first from registers, then from
    /// stack. Returns a vector of extracted argument values on success.
    ///
    /// If the request can be satisfied entirely from registers (i.e.
    /// `num_args <= arg_registers().len()`), this always succeeds — `memory`
    /// is unused. Otherwise the trait reads the stack-resident slots from
    /// `memory`; any failure (no memory view, symbolic SP, unmapped slot)
    /// produces a structured [`ExtractionError`] so the caller can decide
    /// whether to abort, retry, or fabricate placeholders explicitly.
    fn extract_args(
        &self,
        regs: &RegisterFile,
        memory: Option<&SymbolicMemory>,
        ctx: &SymContext,
        num_args: usize,
    ) -> Result<Vec<RustBV>, ExtractionError> {
        let mut args = Vec::with_capacity(num_args);
        let arg_regs = self.arg_registers();
        let ptr_size = self.pointer_size();

        for &offset in arg_regs.iter() {
            if args.len() >= num_args {
                break;
            }
            args.push(regs.get(offset, ptr_size, ctx));
        }

        if args.len() >= num_args {
            return Ok(args);
        }

        let mem = memory.ok_or(ExtractionError::MemoryUnavailable)?;
        let sp = regs.get(regs.arch().sp_offset(), ptr_size, ctx);
        let sp_val = sp.as_u64().ok_or(ExtractionError::SpSymbolic)?;
        let stack_start = sp_val + self.stack_arg_offset();
        let already = args.len();
        let remaining = num_args - already;

        for i in 0..remaining {
            let addr = stack_start + (i as u64 * ptr_size as u64);
            let value = mem.load_concrete_lazy(addr, ptr_size, ctx).map_err(|_| {
                ExtractionError::StackUnmapped {
                    arg_index: already + i,
                    addr,
                }
            })?;
            args.push(value);
        }

        Ok(args)
    }

    /// Get the return address from the stack.
    ///
    /// For x86/AMD64, this is at [rsp] after a call instruction.
    fn get_return_addr(
        &self,
        regs: &RegisterFile,
        memory: Option<&SymbolicMemory>,
        ctx: &SymContext,
    ) -> Option<u64> {
        let ptr_size = self.pointer_size();
        let sp = regs.get(regs.arch().sp_offset(), ptr_size, ctx);
        let sp_val = sp.as_u64()?;

        if let Some(mem) = memory {
            if let Ok(ret_val) = mem.load_concrete_lazy(sp_val, ptr_size, ctx) {
                return ret_val.as_u64();
            }
        }

        None
    }
}

/// AMD64 System V ABI calling convention (Linux, macOS, BSD).
///
/// Integer/pointer arguments: RDI, RSI, RDX, RCX, R8, R9
/// Floating-point arguments: XMM0-XMM7
/// Return value: RAX (with RDX for 128-bit)
/// Stack alignment: 16 bytes at function entry
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemVAMD64;

impl SystemVAMD64 {
    pub const ARCH_ALIASES: &'static [&'static str] = &["amd64", "x86_64", "x64"];
}

impl CallingConvention for SystemVAMD64 {
    fn name(&self) -> &'static str {
        "SystemV_AMD64"
    }

    fn arg_registers(&self) -> &[u32] {
        // RDI, RSI, RDX, RCX, R8, R9
        &[72, 64, 32, 24, 80, 88]
    }

    fn syscall_arg_registers(&self) -> &[u32] {
        // Linux amd64 syscall ABI: RDI, RSI, RDX, R10, R8, R9.
        // Differs from the C ABI at the 4th argument: R10 (96) vs RCX (24).
        &[72, 64, 32, 96, 80, 88]
    }

    fn fp_arg_registers(&self) -> &[u32] {
        // XMM0-XMM7
        &[224, 256, 288, 320, 352, 384, 416, 448]
    }

    fn pointer_size(&self) -> u32 {
        8
    }

    fn stack_arg_offset(&self) -> u64 {
        // After call: [rsp] = return address, args start at [rsp + 8]
        8
    }

    fn endness(&self) -> Endness {
        Endness::Little
    }

    fn return_register(&self) -> u32 {
        16 // RAX
    }
}

/// Microsoft x64 calling convention (Windows).
///
/// Integer/pointer arguments: RCX, RDX, R8, R9
/// Floating-point arguments: XMM0-XMM3 (shadow space on stack)
/// Return value: RAX
/// Shadow space: 32 bytes reserved on stack
#[derive(Debug, Clone, Copy, Default)]
pub struct MicrosoftX64;

impl MicrosoftX64 {
    /// MicrosoftX64 is not the default for any arch in `default_cc_for_arch`;
    /// callers select it explicitly when they know they're dealing with a
    /// Windows binary.
    pub const ARCH_ALIASES: &'static [&'static str] = &[];
}

impl CallingConvention for MicrosoftX64 {
    fn name(&self) -> &'static str {
        "Microsoft_x64"
    }

    fn arg_registers(&self) -> &[u32] {
        // RCX, RDX, R8, R9
        &[24, 32, 80, 88]
    }

    fn fp_arg_registers(&self) -> &[u32] {
        // XMM0-XMM3
        &[224, 256, 288, 320]
    }

    fn pointer_size(&self) -> u32 {
        8
    }

    fn stack_arg_offset(&self) -> u64 {
        // After call: [rsp] = return address
        // Then 32 bytes of shadow space
        // Args start at [rsp + 8 + 32] = [rsp + 40]
        40
    }

    fn endness(&self) -> Endness {
        Endness::Little
    }

    fn return_register(&self) -> u32 {
        16 // RAX
    }
}

/// x86 cdecl calling convention.
///
/// All arguments on stack, right-to-left.
/// Return value: EAX (with EDX for 64-bit)
/// Caller cleans up stack.
#[derive(Debug, Clone, Copy, Default)]
pub struct Cdecl;

impl Cdecl {
    pub const ARCH_ALIASES: &'static [&'static str] = &["x86", "i386", "i686"];
}

impl CallingConvention for Cdecl {
    fn name(&self) -> &'static str {
        "cdecl"
    }

    fn arg_registers(&self) -> &[u32] {
        // No register arguments in cdecl
        &[]
    }

    fn syscall_arg_registers(&self) -> &[u32] {
        // Linux i386 syscall ABI (int 0x80): EBX, ECX, EDX, ESI, EDI, EBP.
        // The C ABI is empty (cdecl is stack-only) but syscalls bypass
        // libc, so we must define the kernel ABI explicitly here.
        // x86 VEX offsets: EBX=20, ECX=12, EDX=16, ESI=32, EDI=36, EBP=28.
        &[20, 12, 16, 32, 36, 28]
    }

    fn fp_arg_registers(&self) -> &[u32] {
        // No register FP arguments
        &[]
    }

    fn pointer_size(&self) -> u32 {
        4
    }

    fn stack_arg_offset(&self) -> u64 {
        // After call: [esp] = return address, args start at [esp + 4]
        4
    }

    fn endness(&self) -> Endness {
        Endness::Little
    }

    fn return_register(&self) -> u32 {
        // EAX in x86 VEX register file (offset differs from amd64's RAX=16)
        8
    }
}

/// ARM EABI calling convention.
///
/// Integer/pointer arguments: R0-R3
/// Return value: R0 (with R1 for 64-bit)
/// Return address: LR (R14) — ARM uses BL which stores return addr in LR, not stack.
#[derive(Debug, Clone, Copy, Default)]
pub struct ARMEABI;

impl ARMEABI {
    pub const ARCH_ALIASES: &'static [&'static str] = &["arm", "armel", "armhf"];
}

impl CallingConvention for ARMEABI {
    fn name(&self) -> &'static str {
        "ARM_EABI"
    }

    fn arg_registers(&self) -> &[u32] {
        // R0-R3 (ARM VEX offsets: R0=8, R1=12, R2=16, R3=20)
        &[8, 12, 16, 20]
    }

    fn syscall_arg_registers(&self) -> &[u32] {
        // Linux ARM EABI syscall ABI: R0-R5 hold args (R7 holds the
        // syscall number). The C ABI only uses R0-R3, so syscalls with
        // 4+ args (e.g. mmap2 with 6, rt_sigaction with 4) need the
        // wider window or extracted args would be zero-padded.
        // ARM VEX offsets: R0=8, R1=12, R2=16, R3=20, R4=24, R5=28.
        &[8, 12, 16, 20, 24, 28]
    }

    fn fp_arg_registers(&self) -> &[u32] {
        // D0-D7 for hard-float
        &[]
    }

    fn pointer_size(&self) -> u32 {
        4
    }

    fn stack_arg_offset(&self) -> u64 {
        // BL doesn't push return address to stack on ARM
        // Args start at [sp + 0]
        0
    }

    fn endness(&self) -> Endness {
        Endness::Little
    }

    fn return_register(&self) -> u32 {
        8 // R0
    }

    /// On ARM, BL stores the return address in LR (R14), not on the stack.
    fn get_return_addr(
        &self,
        regs: &RegisterFile,
        _memory: Option<&SymbolicMemory>,
        ctx: &SymContext,
    ) -> Option<u64> {
        // LR offset = 64 (R14 in ARM VEX guest state)
        let lr = regs.get(64, 4, ctx);
        lr.as_u64()
    }

    fn pops_return_addr(&self) -> bool {
        false
    }
}

/// AArch64 (ARM64) calling convention.
///
/// Integer/pointer arguments: X0-X7
/// Floating-point arguments: V0-V7
/// Return value: X0 (with X1 for 128-bit)
/// Return address: LR (X30) — ARM64 uses BL which stores return addr in X30.
#[derive(Debug, Clone, Copy, Default)]
pub struct AArch64CC;

impl AArch64CC {
    pub const ARCH_ALIASES: &'static [&'static str] = &["arm64", "aarch64"];
}

impl CallingConvention for AArch64CC {
    fn name(&self) -> &'static str {
        "AArch64"
    }

    fn arg_registers(&self) -> &[u32] {
        // X0-X7 (ARM64 VEX offsets)
        &[16, 24, 32, 40, 48, 56, 64, 72]
    }

    fn fp_arg_registers(&self) -> &[u32] {
        // V0-V7
        &[]
    }

    fn pointer_size(&self) -> u32 {
        8
    }

    fn stack_arg_offset(&self) -> u64 {
        0
    }

    fn endness(&self) -> Endness {
        Endness::Little
    }

    fn return_register(&self) -> u32 {
        16 // X0
    }

    /// On AArch64, BL stores the return address in X30 (LR), not on the stack.
    fn get_return_addr(
        &self,
        regs: &RegisterFile,
        _memory: Option<&SymbolicMemory>,
        ctx: &SymContext,
    ) -> Option<u64> {
        // X30 (LR) offset = 256 in ARM64 VEX guest state
        let lr = regs.get(256, 8, ctx);
        lr.as_u64()
    }

    fn pops_return_addr(&self) -> bool {
        false
    }
}

/// MIPS O32 calling convention (32-bit MIPS, the default Linux ABI).
///
/// Integer/pointer arguments: $a0-$a3 (R4-R7)
/// Return value: $v0 (R2). $v1 (R3) holds the upper half for 64-bit returns.
/// Return address: $ra (R31) — JAL writes the return address into $ra, not
/// the stack. Note that O32 reserves 16 bytes of stack space for the four
/// register-passed args; additional args land at [sp + 16].
#[derive(Debug, Clone, Copy, Default)]
pub struct MipsO32;

impl MipsO32 {
    pub const ARCH_ALIASES: &'static [&'static str] = &["mips", "mips32", "mipsel", "mipsbe"];
}

impl CallingConvention for MipsO32 {
    fn name(&self) -> &'static str {
        "MIPS_O32"
    }

    fn arg_registers(&self) -> &[u32] {
        // $a0-$a3 (MIPS32 VEX offsets: R4=24, R5=28, R6=32, R7=36)
        &[24, 28, 32, 36]
    }

    fn fp_arg_registers(&self) -> &[u32] {
        &[]
    }

    fn pointer_size(&self) -> u32 {
        4
    }

    fn stack_arg_offset(&self) -> u64 {
        // O32 reserves 16 bytes for $a0-$a3 in the caller's frame.
        // Stack-passed args (5+) start at [sp + 16].
        16
    }

    fn endness(&self) -> Endness {
        // MIPS may be either-endian; default to little. The arch's own
        // `is_little_endian()` is the authoritative endianness for memory
        // access on a given state.
        Endness::Little
    }

    fn return_register(&self) -> u32 {
        16 // $v0 (R2)
    }

    fn get_return_addr(
        &self,
        regs: &RegisterFile,
        _memory: Option<&SymbolicMemory>,
        ctx: &SymContext,
    ) -> Option<u64> {
        // $ra (R31) offset = 132 in MIPS32 VEX guest state
        let ra = regs.get(132, 4, ctx);
        ra.as_u64()
    }

    fn pops_return_addr(&self) -> bool {
        false
    }
}

/// MIPS N64 calling convention (64-bit MIPS, the default Linux ABI).
///
/// Integer/pointer arguments: $a0-$a7 (R4-R11). N64 widens the O32
/// four-register window to eight by repurposing $t0-$t3 as $a4-$a7.
/// Return value: $v0 (R2). Return address: $ra (R31) — JAL writes the
/// return address into $ra, not the stack.
///
/// Unlike O32, N64 does NOT reserve a stack save area for the
/// register-passed arguments; stack-passed args (9+) start at [sp + 0].
#[derive(Debug, Clone, Copy, Default)]
pub struct MipsN64;

impl MipsN64 {
    pub const ARCH_ALIASES: &'static [&'static str] =
        &["mips64", "mips64el", "mips64le", "mips64be"];
}

impl CallingConvention for MipsN64 {
    fn name(&self) -> &'static str {
        "MIPS_N64"
    }

    fn arg_registers(&self) -> &[u32] {
        // $a0-$a7 (MIPS64 VEX offsets: R4=48, R5=56, R6=64, R7=72,
        // R8=80, R9=88, R10=96, R11=104). See arch/mips.rs offsets64.
        &[48, 56, 64, 72, 80, 88, 96, 104]
    }

    fn fp_arg_registers(&self) -> &[u32] {
        &[]
    }

    fn pointer_size(&self) -> u32 {
        8
    }

    fn stack_arg_offset(&self) -> u64 {
        // N64 does not reserve a save area for register args
        // (unlike O32's 16-byte window). Stack-passed args start at [sp].
        0
    }

    fn endness(&self) -> Endness {
        // MIPS may be either-endian; default to little. The arch's own
        // `is_little_endian()` is the authoritative endianness for memory
        // access on a given state.
        Endness::Little
    }

    fn return_register(&self) -> u32 {
        32 // $v0 (R2) in MIPS64 VEX guest state
    }

    fn get_return_addr(
        &self,
        regs: &RegisterFile,
        _memory: Option<&SymbolicMemory>,
        ctx: &SymContext,
    ) -> Option<u64> {
        // $ra (R31) offset = 264 in MIPS64 VEX guest state
        let ra = regs.get(264, 8, ctx);
        ra.as_u64()
    }

    fn pops_return_addr(&self) -> bool {
        false
    }
}

/// Get the default calling convention for an architecture.
///
/// Driven by each CC's inherent `ARCH_ALIASES` constant — adding a new
/// alias only requires updating the relevant impl block. Unknown arch
/// names cause a panic so that mis-routed argument extraction (silently
/// reading AMD64 RDI/RSI for a foreign arch) fails loudly instead of
/// producing wrong-but-plausible values. See the latent x86 Cdecl
/// return-register bug (commit 5329d8222) for the failure mode this
/// guards against.
pub fn default_cc_for_arch(arch_name: &str) -> Box<dyn CallingConvention> {
    let lower = arch_name.to_lowercase();
    let lower_str = lower.as_str();

    if SystemVAMD64::ARCH_ALIASES.contains(&lower_str) {
        Box::new(SystemVAMD64)
    } else if Cdecl::ARCH_ALIASES.contains(&lower_str) {
        Box::new(Cdecl)
    } else if ARMEABI::ARCH_ALIASES.contains(&lower_str) {
        Box::new(ARMEABI)
    } else if AArch64CC::ARCH_ALIASES.contains(&lower_str) {
        Box::new(AArch64CC)
    } else if MipsO32::ARCH_ALIASES.contains(&lower_str) {
        Box::new(MipsO32)
    } else if MipsN64::ARCH_ALIASES.contains(&lower_str) {
        Box::new(MipsN64)
    } else {
        panic!(
            "default_cc_for_arch: no calling convention registered for arch {arch_name:?}. \
             Register it in ARCH_ALIASES on the relevant CC, or add a new CallingConvention impl. \
             Silent fallback to SystemV_AMD64 would mis-route argument extraction."
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::AMD64;

    #[test]
    fn test_systemv_amd64_args() {
        let cc = SystemVAMD64;
        assert_eq!(cc.name(), "SystemV_AMD64");
        assert_eq!(cc.arg_registers().len(), 6);
        assert_eq!(cc.pointer_size(), 8);
        assert_eq!(cc.stack_arg_offset(), 8);
    }

    #[test]
    fn test_systemv_amd64_syscall_args_use_r10_not_rcx() {
        // Linux amd64 syscall ABI uses RDI, RSI, RDX, R10, R8, R9.
        // Differs from C ABI at the 4th arg: R10 (offset 96) replaces RCX
        // (offset 24). Required for 4+ arg syscalls (mmap takes 6).
        let cc = SystemVAMD64;
        assert_eq!(
            cc.syscall_arg_registers(),
            &[72, 64, 32, 96, 80, 88][..],
            "amd64 syscall ABI must use R10 (96), not RCX (24), at arg 4",
        );
        assert_eq!(
            cc.arg_registers()[3],
            24,
            "C ABI arg 4 is RCX (sanity check)"
        );
    }

    #[test]
    fn test_default_syscall_args_match_arg_registers() {
        // CCs that don't override syscall_arg_registers should fall back to
        // the C ABI registers. AArch64 and MipsO32 don't override (their
        // C ABI register windows X0-X7 / $a0-$a3 cover the syscall ABI).
        // Cdecl/SystemV_AMD64/ARMEABI all override, so don't use them here.
        let cc = AArch64CC;
        assert_eq!(cc.syscall_arg_registers(), cc.arg_registers());
        let cc = MipsO32;
        assert_eq!(cc.syscall_arg_registers(), cc.arg_registers());
    }

    #[test]
    fn test_cdecl_syscall_args_use_linux_i386_abi() {
        // Linux i386 syscall ABI (int 0x80): EBX, ECX, EDX, ESI, EDI, EBP.
        // The C ABI cdecl is empty (stack-only), so this MUST be overridden
        // or extract_syscall_args would zero-pad every arg. x86 VEX offsets:
        // EBX=20, ECX=12, EDX=16, ESI=32, EDI=36, EBP=28.
        let cc = Cdecl;
        assert_eq!(
            cc.syscall_arg_registers(),
            &[20, 12, 16, 32, 36, 28][..],
            "x86 syscall ABI must be EBX, ECX, EDX, ESI, EDI, EBP",
        );
        assert!(
            cc.arg_registers().is_empty(),
            "C ABI sanity check (cdecl is stack-only)"
        );
    }

    #[test]
    fn test_arm_eabi_syscall_args_extend_to_r5() {
        // Linux ARM EABI syscall ABI: R0-R5 hold args (R7 holds the syscall
        // number). The C ABI only uses R0-R3, so syscalls with 4+ args
        // (mmap2 with 6, rt_sigaction with 4) require the wider register
        // window. ARM VEX offsets: R0=8, R1=12, R2=16, R3=20, R4=24, R5=28.
        let cc = ARMEABI;
        assert_eq!(
            cc.syscall_arg_registers(),
            &[8, 12, 16, 20, 24, 28][..],
            "ARM Linux syscall ABI must extend the C ABI window to R5",
        );
        assert_eq!(
            cc.arg_registers(),
            &[8, 12, 16, 20][..],
            "C ABI sanity check (R0-R3 only)"
        );
    }

    #[test]
    fn test_cdecl_args() {
        let cc = Cdecl;
        assert_eq!(cc.name(), "cdecl");
        assert!(cc.arg_registers().is_empty()); // All args on stack
        assert_eq!(cc.pointer_size(), 4);
        assert_eq!(cc.stack_arg_offset(), 4);
    }

    #[test]
    fn test_return_register_offsets_per_arch() {
        // Each calling convention must use the right return-register offset
        // for its architecture's VEX register file. EAX (x86) ≠ RAX (amd64).
        // Bug: cdecl previously returned 16 (RAX) which routed native procedure
        // results to EDX in 32-bit x86 binaries, leaving EAX unset — and the
        // caller's cmp/jne against EAX kept the stale prior value, masking
        // any fork the symbolic return value would have produced.
        // EAX in x86 VEX guest state = offset 8.
        assert_eq!(Cdecl.return_register(), 8);
        // RAX in amd64 VEX guest state = offset 16.
        assert_eq!(SystemVAMD64.return_register(), 16);
        assert_eq!(MicrosoftX64.return_register(), 16);
        // ARM r0 = offset 8 (R0 in ARM VEX guest state).
        assert_eq!(ARMEABI.return_register(), 8);
        // AArch64 X0 = offset 16 (X0 in ARM64 VEX guest state).
        assert_eq!(AArch64CC.return_register(), 16);
        // MIPS32 $v0 (R2) = offset 16 (R2 in MIPS32 VEX guest state).
        assert_eq!(MipsO32.return_register(), 16);
        // MIPS64 $v0 (R2) = offset 32 (R2 in MIPS64 VEX guest state).
        assert_eq!(MipsN64.return_register(), 32);
    }

    #[test]
    fn test_pops_return_addr_per_arch() {
        // Stack-based ABIs (x86/AMD64): return addr lives at [sp], so the
        // dispatcher must increment SP after a native procedure returns.
        assert!(SystemVAMD64.pops_return_addr());
        assert!(MicrosoftX64.pops_return_addr());
        assert!(Cdecl.pops_return_addr());
        // Register-based ABIs (ARM/ARM64/MIPS): return addr lives in
        // LR/X30/$ra, and SP must be left untouched.
        assert!(!ARMEABI.pops_return_addr());
        assert!(!AArch64CC.pops_return_addr());
        assert!(!MipsO32.pops_return_addr());
        assert!(!MipsN64.pops_return_addr());
    }

    #[test]
    fn test_mips_n64_arg_registers() {
        // N64 widens the O32 four-register window to eight: $a0-$a7
        // (R4-R11), repurposing $t0-$t3. VEX MIPS64 offsets for R4-R11
        // are 48, 56, 64, 72, 80, 88, 96, 104.
        let cc = MipsN64;
        assert_eq!(cc.name(), "MIPS_N64");
        assert_eq!(cc.arg_registers(), &[48, 56, 64, 72, 80, 88, 96, 104]);
        assert_eq!(cc.pointer_size(), 8);
        // N64 does not reserve a save area for register args (unlike O32's
        // 16-byte window).
        assert_eq!(cc.stack_arg_offset(), 0);
    }

    #[test]
    fn test_default_cc_for_arch_registry() {
        // Each arch alias must resolve to the documented CC. If you add a new
        // alias, extend the relevant ARCH_ALIASES constant and add a row here.
        let cases: &[(&str, &str)] = &[
            ("amd64", "SystemV_AMD64"),
            ("x86_64", "SystemV_AMD64"),
            ("x64", "SystemV_AMD64"),
            ("AMD64", "SystemV_AMD64"), // case-insensitive
            ("x86", "cdecl"),
            ("i386", "cdecl"),
            ("i686", "cdecl"),
            ("arm", "ARM_EABI"),
            ("armel", "ARM_EABI"),
            ("armhf", "ARM_EABI"),
            ("arm64", "AArch64"),
            ("aarch64", "AArch64"),
            ("mips", "MIPS_O32"),
            ("mips32", "MIPS_O32"),
            ("mipsel", "MIPS_O32"),
            ("mipsbe", "MIPS_O32"),
            ("mips64", "MIPS_N64"),
            ("mips64el", "MIPS_N64"),
            ("mips64le", "MIPS_N64"),
            ("mips64be", "MIPS_N64"),
            ("MIPS64", "MIPS_N64"), // case-insensitive
        ];
        for (arch, expected) in cases {
            assert_eq!(
                default_cc_for_arch(arch).name(),
                *expected,
                "default_cc_for_arch({arch:?}) should resolve to {expected}",
            );
        }
    }

    #[test]
    #[should_panic(expected = "no calling convention registered")]
    fn test_default_cc_for_arch_unknown_panics() {
        // Unknown archs must panic loudly, not silently fall back to
        // SystemV_AMD64 (which would mis-route argument extraction —
        // see commit 5329d8222 for the x86 Cdecl variant of this bug).
        let _ = default_cc_for_arch("ppc");
    }

    #[test]
    fn test_arch_aliases_disjoint() {
        // No alias may belong to more than one CC, otherwise default_cc_for_arch
        // becomes order-dependent.
        let groups: &[(&str, &[&str])] = &[
            ("SystemV_AMD64", SystemVAMD64::ARCH_ALIASES),
            ("Microsoft_x64", MicrosoftX64::ARCH_ALIASES),
            ("cdecl", Cdecl::ARCH_ALIASES),
            ("ARM_EABI", ARMEABI::ARCH_ALIASES),
            ("AArch64", AArch64CC::ARCH_ALIASES),
            ("MIPS_O32", MipsO32::ARCH_ALIASES),
            ("MIPS_N64", MipsN64::ARCH_ALIASES),
        ];
        for (i, (name_a, aliases_a)) in groups.iter().enumerate() {
            for (name_b, aliases_b) in &groups[i + 1..] {
                for a in *aliases_a {
                    assert!(
                        !aliases_b.contains(a),
                        "alias {a:?} appears in both {name_a} and {name_b}",
                    );
                }
            }
        }
    }

    #[test]
    fn test_extract_args_from_regs() {
        let ctx = SymContext::new_mock();
        let cc = SystemVAMD64;
        let mut regs = RegisterFile::new(Box::new(AMD64));

        // Set up some argument values
        // RDI (arg1) = 0x1000
        regs.put(72, RustBV::concrete(0x1000, 64));
        // RSI (arg2) = 0x2000
        regs.put(64, RustBV::concrete(0x2000, 64));
        // RDX (arg3) = 0x3000
        regs.put(32, RustBV::concrete(0x3000, 64));

        let args = cc.extract_args(&regs, None, &ctx, 3).expect("regs-only");
        assert_eq!(args.len(), 3);
        assert_eq!(args[0].as_u64(), Some(0x1000));
        assert_eq!(args[1].as_u64(), Some(0x2000));
        assert_eq!(args[2].as_u64(), Some(0x3000));
    }

    #[test]
    fn test_extract_args_stack_without_memory_errors() {
        // amd64 SystemV has 6 register slots; asking for a 7th forces the
        // trait to consult memory. With no memory view supplied, it must
        // return MemoryUnavailable rather than silently fabricate a
        // placeholder.
        let ctx = SymContext::new_mock();
        let cc = SystemVAMD64;
        let regs = RegisterFile::new(Box::new(AMD64));
        let err = cc
            .extract_args(&regs, None, &ctx, 7)
            .expect_err("no memory => should fail");
        assert!(
            matches!(err, ExtractionError::MemoryUnavailable),
            "expected MemoryUnavailable, got {err:?}",
        );
    }

    #[test]
    fn test_extract_args_register_only_path_ignores_missing_memory() {
        // The reverse: 6 register args with no memory view must succeed —
        // no stack slot is ever consulted.
        let ctx = SymContext::new_mock();
        let cc = SystemVAMD64;
        let regs = RegisterFile::new(Box::new(AMD64));
        let args = cc
            .extract_args(&regs, None, &ctx, 6)
            .expect("6 args fit in registers; memory unused");
        assert_eq!(args.len(), 6);
    }
}
