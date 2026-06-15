//! Tests for calling convention argument extraction.
//!
//! Extracted from `calling_conventions.rs` (see bd `rust-mod-tests-sibling-extraction`).

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
