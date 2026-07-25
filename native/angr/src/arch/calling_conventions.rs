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
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ExtractionError {
    /// The caller asked for stack-resident arguments but did not supply a
    /// memory view. The trait method has no way to materialise stack
    /// arguments without it.
    #[error("extract_args: stack argument requested without a memory view")]
    MemoryUnavailable,
    /// The stack pointer is symbolic. We cannot compute the stack-slot
    /// addresses without committing to a concrete SP, which would silently
    /// pin the value of a symbol the caller may want to reason about.
    #[error("extract_args: stack pointer is symbolic; cannot compute stack-slot addresses")]
    SpSymbolic,
    /// A specific stack slot could not be read from memory (typically
    /// unmapped page or permission failure). `arg_index` is the
    /// zero-based position of the failing argument within the full
    /// `num_args` request; `addr` is the absolute address that failed.
    #[error("extract_args: stack argument {arg_index} at address {addr:#x} is unmapped")]
    StackUnmapped { arg_index: usize, addr: u64 },
    /// The caller requested more arguments than the ABI exposes via
    /// registers, but the ABI has no stack path (e.g. syscalls on most
    /// architectures). Indicates a SimProcedure / syscall-handler
    /// misconfiguration declaring a higher `num_args` than the kernel
    /// ABI supports.
    #[error(
        "extract_args: requested {requested} arguments but ABI exposes only {available} registers and no stack path"
    )]
    RegisterOverflow { requested: usize, available: usize },
}

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

    /// Stack offset (relative to SP) where syscall arguments past the register
    /// window begin, or `None` if this ABI never spills syscall args to the
    /// stack.
    ///
    /// Most Linux syscall ABIs (amd64, x86, ARM, AArch64) expose every syscall
    /// argument we care about in registers, so a request for more args than
    /// the register window holds is a genuine [`ExtractionError::RegisterOverflow`]
    /// and the caller falls through to the Python syscall callback. MIPS O32 is
    /// the exception: it passes syscall args 5+ on the stack at `sp+16` (the
    /// same save area as the C ABI), so it overrides this to `Some(16)`.
    fn syscall_stack_arg_offset(&self) -> Option<u64> {
        None
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

    /// Byte offset of the register slot carrying a scalar `double` return
    /// value, or `None` when this ABI has no single-register FP-return slot we
    /// model (x86 returns in the x87 `st0` stack; ARM EABI soft-float returns
    /// a double in the `r0:r1` integer pair; MIPS O32/N64 use `$f0`, which our
    /// register files do not expose yet).
    ///
    /// The value is written as the 64-bit IEEE-754 bit pattern into the low 64
    /// bits of that offset (both AMD64 `xmm0` and AArch64 `v0` are 128-bit
    /// little-endian vector registers whose low half holds the scalar).
    /// Consumers must suppress the dispatcher's default integer-return store
    /// (return `Ok(None)`) when they use this slot. See `procedures/strtod.rs`.
    fn fp_return_register(&self) -> Option<u32> {
        None
    }

    /// The Linux syscall error register and its errno threshold, if this ABI
    /// carries a *separate* success/failure flag alongside the return value.
    ///
    /// On most architectures (amd64, x86, ARM, AArch64) the kernel encodes
    /// errors directly in the return register as a small negative value, so
    /// this returns `None`. MIPS is the notable exception: the O32/N64 Linux
    /// ABIs set `$a3` to 0 on success and non-zero on error, with `$v0`
    /// holding the *positive* errno. PowerPC uses `cr0.SO` similarly.
    ///
    /// The tuple is `(error_register_offset, errno_start)` where `errno_start`
    /// mirrors angr's `SYSCALL_ERRNO_START` (the signed threshold above which
    /// an unsigned return value is treated as `-errno`). See
    /// `linux_syscall_update_error_reg` in `angr/calling_conventions.py`.
    fn syscall_error_register(&self) -> Option<(u32, i64)> {
        None
    }

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

    /// Register offset of the link/return-address register on ABIs where
    /// `call` writes the return address to a register rather than the stack
    /// (`pops_return_addr() == false`): ARM/ARM64 LR (R14/X30), MIPS `$ra`
    /// (R31). Returns `None` on stack-return ABIs (x86/AMD64) and on any ABI
    /// that has not yet wired this up.
    ///
    /// Used by the native sub-call dispatcher (S2, bead angr-5gf0s) to make a
    /// guest routine return to the resume sentinel on link-register ABIs. Until
    /// an arch overrides it, link-register sub-calls fall back to the Python
    /// SimProcedure path.
    fn link_register(&self) -> Option<u32> {
        None
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
    /// For x86/AMD64, this is at \[rsp\] after a call instruction.
    fn get_return_addr(
        &self,
        regs: &RegisterFile,
        memory: Option<&SymbolicMemory>,
        ctx: &SymContext,
    ) -> Option<u64> {
        let ptr_size = self.pointer_size();
        let sp = regs.get(regs.arch().sp_offset(), ptr_size, ctx);
        let sp_val = sp.as_u64()?;

        if let Some(mem) = memory
            && let Ok(ret_val) = mem.load_concrete_lazy(sp_val, ptr_size, ctx)
        {
            return ret_val.as_u64();
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

    fn fp_return_register(&self) -> Option<u32> {
        Some(224) // XMM0 (low 64 bits carry the scalar double)
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
    pub const ARCH_ALIASES: &'static [&'static str] = &["x86", "i386", "i486", "i586", "i686"];
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
    pub const ARCH_ALIASES: &'static [&'static str] = &["arm", "armel", "armhf", "armv7", "armv7l"];
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
    pub const ARCH_ALIASES: &'static [&'static str] = &["arm64", "aarch64", "armv8"];
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

    fn fp_return_register(&self) -> Option<u32> {
        Some(320) // Q0/V0 (low 64 bits carry the scalar double)
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
    pub const ARCH_ALIASES: &'static [&'static str] = &["mips", "mips32", "mipsel", "mipsle"];
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

    fn syscall_stack_arg_offset(&self) -> Option<u64> {
        // O32 syscalls share the C ABI's 16-byte save area: args 5+ live at
        // [sp + 16]. Enables native dispatch of 6-arg syscalls (futex 4238,
        // epoll_pwait 4313, mmap2 4210) when SP is concrete.
        Some(16)
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

    fn syscall_error_register(&self) -> Option<(u32, i64)> {
        // $a3 (R7) = offset 36 in MIPS32 VEX guest state. errno_start matches
        // SimCCO32LinuxSyscall.SYSCALL_ERRNO_START (-1133).
        Some((36, -1133))
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

    fn syscall_error_register(&self) -> Option<(u32, i64)> {
        // $a3 (R7) = offset 72 in MIPS64 VEX guest state. errno_start matches
        // SimCCN64LinuxSyscall.SYSCALL_ERRNO_START (-1133).
        Some((72, -1133))
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
    cc_for_arch(arch_name).unwrap_or_else(|| {
        panic!(
            "default_cc_for_arch: no calling convention registered for arch {arch_name:?}. \
             Register it in ARCH_ALIASES on the relevant CC, or add a new CallingConvention impl. \
             Silent fallback to SystemV_AMD64 would mis-route argument extraction."
        )
    })
}

/// Non-panicking variant of [`default_cc_for_arch`]: resolve the default
/// calling convention for `arch_name`, or `None` when no CC is registered.
///
/// Callers that can degrade gracefully (e.g. a native SimProcedure that defers
/// to Python on an unmodelled ABI) use this; the interpreter's argument
/// extraction path — where a wrong CC silently yields wrong-but-plausible
/// arguments — keeps the panicking wrapper.
pub fn cc_for_arch(arch_name: &str) -> Option<Box<dyn CallingConvention>> {
    let lower = arch_name.to_lowercase();
    let lower_str = lower.as_str();

    if SystemVAMD64::ARCH_ALIASES.contains(&lower_str) {
        Some(Box::new(SystemVAMD64))
    } else if Cdecl::ARCH_ALIASES.contains(&lower_str) {
        Some(Box::new(Cdecl))
    } else if ARMEABI::ARCH_ALIASES.contains(&lower_str) {
        Some(Box::new(ARMEABI))
    } else if AArch64CC::ARCH_ALIASES.contains(&lower_str) {
        Some(Box::new(AArch64CC))
    } else if MipsO32::ARCH_ALIASES.contains(&lower_str) {
        Some(Box::new(MipsO32))
    } else if MipsN64::ARCH_ALIASES.contains(&lower_str) {
        Some(Box::new(MipsN64))
    } else {
        None
    }
}

#[cfg(test)]
#[path = "calling_conventions_tests.rs"]
mod calling_conventions_tests;
