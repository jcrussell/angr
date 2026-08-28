//! Per-ABI calling convention tables.
//!
//! Each [`CallingConvention`] impl is a table of ABI facts — argument /
//! syscall-argument register windows, pointer size, stack-spill offsets,
//! return and link registers. The extraction *logic* that consumes them lives
//! once in `exploration::helpers::extract_args_with_abi` (plain code span, not
//! an intra-doc link: `helpers` is private to `exploration`, so the path is
//! not nameable from here),
//! which walks a `RustSimState`; this module deliberately holds no second copy
//! of that walk (angr-9iny6).

use crate::arch::RegisterFile;
use crate::memory::SymbolicMemory;
use crate::symbolic::SymContext;

// Every ABI register table below is expressed as named VEX guest-state
// constants rather than bare integers. The per-arch `offsets` modules carry
// the provenance comments (`VEX/pub/libvex_guest_*.h` / archinfo) and are the
// single source of truth; restating the numbers here once caused the x86
// `Cdecl::return_register` bug (RAX's offset used for EAX).
use super::amd64::offsets as amd64_off;
use super::arm::offsets as arm_off;
use super::arm64::offsets as arm64_off;
use super::mips::{offsets32 as mips32_off, offsets64 as mips64_off};
use super::x86::offsets as x86_off;

/// Errors produced by `exploration::helpers::extract_args_with_abi`
/// when the stack portion of the argument list cannot be read.
///
/// Argument extraction previously relied on silent fabrication (it would mint
/// a fresh `RustBV::symbolic("stack_arg_N", …)` whenever the stack load
/// failed). That made real stack-setup bugs indistinguishable from intentional
/// symbolic input. Returning an explicit error variant forces callers to
/// decide policy.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum ExtractionError {
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
    /// The requested argument count exceeds [`MAX_EXTRACT_ARGS`]. Only
    /// reachable from a caller-supplied count (`register_simprocedure`'s
    /// `num_args` crosses the Python boundary verbatim), so the request is a
    /// misconfiguration rather than anything the guest can drive.
    #[error("extract_args: requested {requested} arguments, exceeding the maximum of {max}")]
    TooManyArgs { requested: usize, max: usize },
}

/// Upper bound on the argument count `extract_args_with_abi` will honor.
///
/// `num_args` reaches that helper straight from Python
/// (`RustExplorationManager::register_simprocedure` stores it verbatim, and
/// `run_loop_single::step_one` only widens it with `.max()`), so without a
/// bound the first statement — a `Vec::with_capacity(num_args)` — turns an
/// ordinary API call like `register_simprocedure(addr, "foo", 1 << 63, false)`
/// into a `capacity overflow` panic or an `alloc::handle_alloc_error` abort
/// that no `catch_unwind` can contain (angr-0jh0j.23).
///
/// 64 is generous by two orders of magnitude against real ABIs: the widest
/// register window here is 8, the largest `num_args()` any native procedure
/// declares is 5, and `MAX_VARARGS` in `procedures::fortify_printf` is 6.
/// A request above it declines cleanly, which makes the dispatcher fall back
/// to the Python SimProcedure rather than crash the host process.
pub(crate) const MAX_EXTRACT_ARGS: usize = 64;

/// Calling convention trait: the per-ABI facts argument extraction, native
/// sub-calls and return-value stores need.
pub(crate) trait CallingConvention: Send + Sync {
    /// Get the name of this calling convention.
    ///
    /// Diagnostic-only: exercised by `calling_conventions_tests` (which asserts
    /// the per-ABI spellings and the `default_cc_for_arch` mapping), but no
    /// production code logs it (angr-9ke6b.214).
    #[cfg_attr(not(test), allow(dead_code))]
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
    ///
    /// No production caller yet — argument extraction is integer-only.
    /// Retained (rather than deleted) because each impl encodes non-obvious
    /// ABI provenance (AAPCS-VFP `D0-D7`, AArch64 `V0-V7`, MIPS'
    /// deliberately-empty table) that FP-argument support will need verbatim
    /// (angr-9ke6b.214). `calling_conventions_tests::fp_arg_registers_*` pins
    /// every table so the provenance cannot rot silently before that support
    /// lands, which is what makes keeping it defensible (angr-9ke6b.218 item 7).
    #[cfg_attr(not(test), allow(dead_code))]
    fn fp_arg_registers(&self) -> &[u32];

    /// Get the size of a pointer in bytes.
    fn pointer_size(&self) -> u32;

    /// Get the stack offset where arguments start (after return address).
    fn stack_arg_offset(&self) -> u64;

    /// Get the return value register offset.
    fn return_register(&self) -> u32;

    /// Byte offset of the register slot carrying a scalar `double` return
    /// value, or `None` when this ABI has no single-register FP-return slot we
    /// model (x86 returns in the x87 `st0` stack; ARM EABI soft-float returns
    /// a double in the `r0:r1` integer pair; MIPS O32/N64 return in `$f0`
    /// *only* for hard-float binaries — a `-msoft-float` build returns the
    /// double in the `$v0`/`$v1` integer pair, and nothing in a `RegisterFile`
    /// tells the two apart, so picking `$f0` would silently write the wrong
    /// slot for half the corpus). The MIPS offsets themselves are exposed
    /// (`mips32_off::F0`/`mips64_off::F0`, aliased `f0`/`$f0`); the blocker is
    /// the same float-ABI-variant ambiguity that keeps
    /// [`MipsO32::fp_arg_registers`] deliberately empty, not a missing
    /// register (angr-sqfj8.9).
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
    /// (R31). Returns `None` on stack-return ABIs (x86/AMD64), where the
    /// return address lives at `[sp]` instead.
    ///
    /// Used by the native sub-call dispatcher (S2, bead angr-5gf0s) to make a
    /// guest routine return to the resume sentinel on link-register ABIs:
    /// `setup_native_subcall` writes the sentinel into this register instead of
    /// over `[sp]`. A link-register ABI that leaves this `None` still works —
    /// setup bails with `SubcallSetupError::UnsupportedAbi` and the proc falls
    /// back to the Python SimProcedure path — so this is a fast-path enabler,
    /// not a correctness requirement.
    ///
    /// The override must name the same register the convention's
    /// `get_return_addr` reads; `calling_conventions_tests::test_link_register_matches_get_return_addr_register`
    /// locks that. Wiring the ARM/ARM64/MIPS overrides up additionally required
    /// `RustExplorationManager::get_return_addr` (`exploration/helpers.rs`) to
    /// resolve through the convention rather than reading `[sp]` unconditionally
    /// — the serial `NativeProcDisposition::SubCall` arm in
    /// `RustExplorationManager::step_one` (`exploration/run_loop_single.rs`)
    /// feeds it into `NativeSubcall::caller_return_addr`, which the resume path
    /// installs as the PC (bead angr-9ke6b.3).
    fn link_register(&self) -> Option<u32> {
        None
    }

    /// Get the return address at a call boundary.
    ///
    /// On stack-return ABIs (x86/AMD64, `pops_return_addr() == true`) this is
    /// \[rsp\] after a `call` instruction, and reading it needs `memory`; with
    /// `memory == None` the convention declines and the caller is expected to
    /// read the slot itself.
    ///
    /// On link-register ABIs (ARM/ARM64/MIPS) the address is in the register
    /// named by [`Self::link_register`] and `memory` is unused. Keeping both
    /// answers on this one default is what makes `link_register()` and
    /// `get_return_addr()` agree by construction rather than by two hardcoded
    /// copies of the same offset (bead angr-9ke6b.3).
    fn get_return_addr(
        &self,
        regs: &RegisterFile,
        memory: Option<&SymbolicMemory>,
        ctx: &SymContext,
    ) -> Option<u64> {
        let ptr_size = self.pointer_size();
        if !self.pops_return_addr() {
            // A link-register ABI that has not named its LR gets `None`, never
            // a stack read: `[sp]` holds no return address there.
            let lr = self.link_register()?;
            return regs.get(lr, ptr_size, ctx).as_u64();
        }
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
pub(crate) struct SystemVAMD64;

impl CallingConvention for SystemVAMD64 {
    fn name(&self) -> &'static str {
        "SystemV_AMD64"
    }

    fn arg_registers(&self) -> &[u32] {
        &[
            amd64_off::RDI,
            amd64_off::RSI,
            amd64_off::RDX,
            amd64_off::RCX,
            amd64_off::R8,
            amd64_off::R9,
        ]
    }

    fn syscall_arg_registers(&self) -> &[u32] {
        // Linux amd64 syscall ABI differs from the C ABI at the 4th
        // argument: R10 in place of RCX (the `syscall` instruction
        // clobbers RCX with the return address).
        &[
            amd64_off::RDI,
            amd64_off::RSI,
            amd64_off::RDX,
            amd64_off::R10,
            amd64_off::R8,
            amd64_off::R9,
        ]
    }

    fn fp_arg_registers(&self) -> &[u32] {
        &[
            amd64_off::XMM0,
            amd64_off::XMM1,
            amd64_off::XMM2,
            amd64_off::XMM3,
            amd64_off::XMM4,
            amd64_off::XMM5,
            amd64_off::XMM6,
            amd64_off::XMM7,
        ]
    }

    fn pointer_size(&self) -> u32 {
        8
    }

    fn stack_arg_offset(&self) -> u64 {
        // After call: [rsp] = return address, args start at [rsp + 8]
        8
    }

    fn return_register(&self) -> u32 {
        amd64_off::RAX
    }

    fn fp_return_register(&self) -> Option<u32> {
        // XMM0's low 64 bits carry the scalar double.
        Some(amd64_off::XMM0)
    }
}

/// x86 cdecl calling convention.
///
/// All arguments on stack, right-to-left.
/// Return value: EAX (with EDX for 64-bit)
/// Caller cleans up stack.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Cdecl;

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
        &[
            x86_off::EBX,
            x86_off::ECX,
            x86_off::EDX,
            x86_off::ESI,
            x86_off::EDI,
            x86_off::EBP,
        ]
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

    fn return_register(&self) -> u32 {
        // EAX in the x86 VEX register file — a different offset from amd64's
        // RAX, which is exactly the confusion that bare literals invited here.
        x86_off::EAX
    }
}

/// ARM EABI calling convention.
///
/// Integer/pointer arguments: R0-R3
/// Return value: R0 (with R1 for 64-bit)
/// Return address: LR (R14) — ARM uses BL which stores return addr in LR, not stack.
/// Name mirrors the ABI's own spelling ("ARM EABI"); see the note on
/// `arch::arm::ARM` for why clippy only started seeing it in angr-9ke6b.214.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ARMEABI;

impl CallingConvention for ARMEABI {
    fn name(&self) -> &'static str {
        "ARM_EABI"
    }

    fn arg_registers(&self) -> &[u32] {
        // R0-R3
        &[arm_off::R0, arm_off::R1, arm_off::R2, arm_off::R3]
    }

    fn syscall_arg_registers(&self) -> &[u32] {
        // Linux ARM EABI syscall ABI: R0-R5 hold args (R7 holds the
        // syscall number). The C ABI only uses R0-R3, so syscalls with
        // 4+ args (e.g. mmap2 with 6, rt_sigaction with 4) need the
        // wider window or extracted args would be zero-padded.
        &[
            arm_off::R0,
            arm_off::R1,
            arm_off::R2,
            arm_off::R3,
            arm_off::R4,
            arm_off::R5,
        ]
    }

    fn fp_arg_registers(&self) -> &[u32] {
        // D0-D7 for hard-float (AAPCS-VFP). Soft-float ARM EABI passes
        // doubles in the integer register pairs instead, so consumers must
        // know which variant the binary was built for.
        &[
            arm_off::D0,
            arm_off::D1,
            arm_off::D2,
            arm_off::D3,
            arm_off::D4,
            arm_off::D5,
            arm_off::D6,
            arm_off::D7,
        ]
    }

    fn pointer_size(&self) -> u32 {
        4
    }

    fn stack_arg_offset(&self) -> u64 {
        // BL doesn't push return address to stack on ARM
        // Args start at [sp + 0]
        0
    }

    fn return_register(&self) -> u32 {
        arm_off::R0
    }

    /// On ARM, BL stores the return address in LR (R14), not on the stack.
    fn link_register(&self) -> Option<u32> {
        Some(arm_off::R14)
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
pub(crate) struct AArch64;

impl CallingConvention for AArch64 {
    fn name(&self) -> &'static str {
        "AArch64"
    }

    fn arg_registers(&self) -> &[u32] {
        // X0-X7
        &[
            arm64_off::X0,
            arm64_off::X1,
            arm64_off::X2,
            arm64_off::X3,
            arm64_off::X4,
            arm64_off::X5,
            arm64_off::X6,
            arm64_off::X7,
        ]
    }

    fn fp_arg_registers(&self) -> &[u32] {
        // V0-V7, i.e. the low half of the Q0-Q7 SIMD bank. Same slots
        // `fp_return_register` already uses for the scalar double return.
        &[
            arm64_off::Q0,
            arm64_off::Q1,
            arm64_off::Q2,
            arm64_off::Q3,
            arm64_off::Q4,
            arm64_off::Q5,
            arm64_off::Q6,
            arm64_off::Q7,
        ]
    }

    fn pointer_size(&self) -> u32 {
        8
    }

    fn stack_arg_offset(&self) -> u64 {
        0
    }

    fn return_register(&self) -> u32 {
        arm64_off::X0
    }

    fn fp_return_register(&self) -> Option<u32> {
        // Q0/V0's low 64 bits carry the scalar double.
        Some(arm64_off::Q0)
    }

    /// On AArch64, BL stores the return address in X30 (LR), not on the stack.
    fn link_register(&self) -> Option<u32> {
        Some(arm64_off::X30)
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
pub(crate) struct MipsO32;

impl CallingConvention for MipsO32 {
    fn name(&self) -> &'static str {
        "MIPS_O32"
    }

    fn arg_registers(&self) -> &[u32] {
        // $a0-$a3
        &[
            mips32_off::R4,
            mips32_off::R5,
            mips32_off::R6,
            mips32_off::R7,
        ]
    }

    fn fp_arg_registers(&self) -> &[u32] {
        // Deliberately empty: O32 hard-float passes doubles in $f12/$f14,
        // but which of $f12/$f14 vs $a0-$a3 a given argument lands in depends
        // on the float/int position rules and on whether the binary is
        // soft-float (`-msoft-float` puts everything in the integer window).
        // We do not model that dispatch, so an extractor handed these offsets
        // would read the wrong slot rather than no slot at all. That
        // positional ambiguity is the whole reason; the offsets themselves
        // exist (`mips32_off::F12`/`F14`, reachable via `register_offset`).
        // angr-9ke6b.218 item 7 removed an older clause here claiming they
        // have "no canonical `register_name` entry" — true, but it does not
        // distinguish MIPS: ARM's `d0`-`d7` and AArch64's `q0`-`q7` are
        // alias-only too, and those tables are populated.
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

    fn return_register(&self) -> u32 {
        mips32_off::R2 // $v0
    }

    fn syscall_error_register(&self) -> Option<(u32, i64)> {
        // $a3 (R7). errno_start matches
        // SimCCO32LinuxSyscall.SYSCALL_ERRNO_START (-1133).
        Some((mips32_off::R7, -1133))
    }

    /// On MIPS, JAL stores the return address in $ra (R31), not on the stack.
    fn link_register(&self) -> Option<u32> {
        Some(mips32_off::R31)
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
pub(crate) struct MipsN64;

impl CallingConvention for MipsN64 {
    fn name(&self) -> &'static str {
        "MIPS_N64"
    }

    fn arg_registers(&self) -> &[u32] {
        // $a0-$a7 = R4-R11 (N64 repurposes $t0-$t3 as $a4-$a7).
        &[
            mips64_off::R4,
            mips64_off::R5,
            mips64_off::R6,
            mips64_off::R7,
            mips64_off::R8,
            mips64_off::R9,
            mips64_off::R10,
            mips64_off::R11,
        ]
    }

    fn fp_arg_registers(&self) -> &[u32] {
        // Deliberately empty, same reasoning as `MipsO32::fp_arg_registers`:
        // N64 passes floats in $f12-$f19, but the int/float slot assignment
        // and the soft-float variant are not modelled.
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

    fn return_register(&self) -> u32 {
        mips64_off::R2 // $v0
    }

    fn syscall_error_register(&self) -> Option<(u32, i64)> {
        // $a3 (R7). errno_start matches
        // SimCCN64LinuxSyscall.SYSCALL_ERRNO_START (-1133).
        Some((mips64_off::R7, -1133))
    }

    /// On MIPS, JAL stores the return address in $ra (R31), not on the stack.
    fn link_register(&self) -> Option<u32> {
        Some(mips64_off::R31)
    }

    fn pops_return_addr(&self) -> bool {
        false
    }
}

/// Get the default calling convention for an architecture.
///
/// Driven by [`crate::arch::registry::ALL_ARCHES`] — adding a new alias only requires
/// adding it to that table's row. Unknown arch names cause a panic so that
/// mis-routed argument extraction (silently reading AMD64 RDI/RSI for a
/// foreign arch) fails loudly instead of producing wrong-but-plausible
/// values. See the latent x86 Cdecl return-register bug (commit 5329d8222)
/// for the failure mode this guards against.
pub(crate) fn default_cc_for_arch(arch_name: &str) -> Box<dyn CallingConvention> {
    cc_for_arch(arch_name).unwrap_or_else(|| {
        panic!(
            "default_cc_for_arch: no calling convention registered for arch {arch_name:?}. \
             Register it in ALL_ARCHES (arch/registry.rs), or add a new CallingConvention impl. \
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
pub(crate) fn cc_for_arch(arch_name: &str) -> Option<Box<dyn CallingConvention>> {
    crate::arch::arch_desc_from_name(arch_name).map(|d| (d.make_cc)())
}

test_submod!("calling_conventions_tests.rs" => calling_conventions_tests);
