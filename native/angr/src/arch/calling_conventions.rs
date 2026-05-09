//! Calling convention implementations for argument extraction.
//!
//! This module provides calling convention support for extracting function
//! arguments from registers and stack. This is used to pre-extract arguments
//! before returning to Python for SimProcedure execution.

use crate::arch::RegisterFile;
use crate::memory::SymbolicMemory;
use crate::symbolic::{RustBV, SymContext};
use crate::vex::Endness;

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

    /// Extract up to N arguments from registers and memory.
    ///
    /// Arguments are extracted in order: first from registers, then from stack.
    /// Returns a vector of extracted argument values.
    fn extract_args(
        &self,
        regs: &RegisterFile,
        memory: Option<&SymbolicMemory>,
        ctx: &SymContext,
        num_args: usize,
    ) -> Vec<RustBV> {
        let mut args = Vec::with_capacity(num_args);
        let arg_regs = self.arg_registers();
        let ptr_size = self.pointer_size();

        // First extract from registers
        for (_i, &offset) in arg_regs.iter().enumerate() {
            if args.len() >= num_args {
                break;
            }
            let value = regs.get(offset, ptr_size, ctx);
            args.push(value);
        }

        // If we need more args, get them from stack
        if args.len() < num_args {
            if let Some(mem) = memory {
                let sp = regs.get(regs.arch().sp_offset(), ptr_size, ctx);
                if let Some(sp_val) = sp.as_u64() {
                    let stack_start = sp_val + self.stack_arg_offset();

                    for i in 0..(num_args - args.len()) {
                        let addr = stack_start + (i as u64 * ptr_size as u64);
                        match mem.load_concrete_lazy(addr, ptr_size, ctx) {
                            Ok(value) => args.push(value),
                            Err(_) => {
                                // Can't read stack - push a symbolic placeholder
                                args.push(RustBV::symbolic(
                                    ctx,
                                    &format!("stack_arg_{}", i),
                                    ptr_size * 8,
                                ));
                            }
                        }
                    }
                }
            }
        }

        args
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
}

/// Get the default calling convention for an architecture.
///
/// Driven by each CC's inherent `ARCH_ALIASES` constant — adding a new
/// alias only requires updating the relevant impl block. Unknown arch
/// names fall back to `SystemVAMD64`.
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
    } else {
        Box::new(SystemVAMD64)
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
        assert_eq!(cc.arg_registers()[3], 24, "C ABI arg 4 is RCX (sanity check)");
    }

    #[test]
    fn test_default_syscall_args_match_arg_registers() {
        // CCs that don't override syscall_arg_registers should fall back to
        // the C ABI registers. Verify on a non-amd64 CC.
        let cc = Cdecl;
        assert_eq!(cc.syscall_arg_registers(), cc.arg_registers());
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
            // Unknown falls back to SystemV_AMD64.
            ("mips", "SystemV_AMD64"),
            ("ppc", "SystemV_AMD64"),
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
    fn test_arch_aliases_disjoint() {
        // No alias may belong to more than one CC, otherwise default_cc_for_arch
        // becomes order-dependent.
        let groups: &[(&str, &[&str])] = &[
            ("SystemV_AMD64", SystemVAMD64::ARCH_ALIASES),
            ("Microsoft_x64", MicrosoftX64::ARCH_ALIASES),
            ("cdecl", Cdecl::ARCH_ALIASES),
            ("ARM_EABI", ARMEABI::ARCH_ALIASES),
            ("AArch64", AArch64CC::ARCH_ALIASES),
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

        let args = cc.extract_args(&regs, None, &ctx, 3);
        assert_eq!(args.len(), 3);
        assert_eq!(args[0].as_u64(), Some(0x1000));
        assert_eq!(args[1].as_u64(), Some(0x2000));
        assert_eq!(args[2].as_u64(), Some(0x3000));
    }
}
