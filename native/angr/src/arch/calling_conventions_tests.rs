//! Tests for calling convention argument extraction.
//!
//! Extracted from `calling_conventions.rs` (see bd `rust-mod-tests-sibling-extraction`).

use super::*;
use crate::arch::{ALL_ARCHES, AMD64};

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
    // Every spelling ALL_ARCHES accepts — canonical name and aliases — must
    // resolve to that row's CC, and matching must stay case-insensitive.
    for desc in ALL_ARCHES {
        let arch = desc.name;
        let expected = (desc.make_cc)().name();
        for spelling in std::iter::once(desc.name).chain(desc.aliases.iter().copied()) {
            assert_eq!(
                default_cc_for_arch(spelling).name(),
                expected,
                "{arch}: default_cc_for_arch({spelling:?}) should resolve to {expected}",
            );
            assert_eq!(
                default_cc_for_arch(&spelling.to_uppercase()).name(),
                expected,
                "{arch}: default_cc_for_arch must be case-insensitive for {spelling:?}",
            );
        }
    }
}

/// `mips64be` used to sit in `MipsN64::ARCH_ALIASES` while `arch_from_name`
/// rejected it — a name the CC registry knew and the arch registry did not.
/// The `ALL_ARCHES` merge resolved that in favor of rejecting it everywhere
/// (MIPS64 BE states are built as `mips64` + `Iend_BE`). Pin the decision so a
/// future edit has to be deliberate.
#[test]
fn test_mips64be_is_not_a_registered_arch_name() {
    assert!(
        crate::arch::arch_from_name("mips64be").is_none(),
        "mips64be must not be an accepted arch name; use mips64 with Iend_BE",
    );
    assert!(
        cc_for_arch("mips64be").is_none(),
        "cc_for_arch must agree with arch_from_name on mips64be",
    );
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
    // No spelling may belong to more than one ALL_ARCHES row, otherwise
    // `arch_desc_from_name` (and through it `default_cc_for_arch`) becomes
    // order-dependent.
    let spellings = |d: &'static crate::arch::ArchDesc| -> Vec<&'static str> {
        std::iter::once(d.name)
            .chain(d.aliases.iter().copied())
            .collect()
    };
    for (i, desc_a) in ALL_ARCHES.iter().enumerate() {
        let a_names = spellings(desc_a);
        for desc_b in &ALL_ARCHES[i + 1..] {
            let b_names = spellings(desc_b);
            for a in &a_names {
                assert!(
                    !b_names.iter().any(|b| b.eq_ignore_ascii_case(a)),
                    "{}: spelling {a:?} also appears in {}",
                    desc_a.name,
                    desc_b.name,
                );
            }
        }
    }
}

#[test]
fn test_arch_from_name_names_all_have_a_cc() {
    // Completeness invariant (bd angr-n0irt.6): every arch-name alias that
    // `arch_from_name` accepts must also resolve to a calling convention via
    // `cc_for_arch`. Otherwise `RustExplorationManager(arch=<alias>)` panics
    // in `default_cc_for_arch` *after* `arch_from_name` already accepted the
    // name — a recognized alias failing more violently than an unrecognized
    // one. `test_arch_aliases_disjoint` guards overlap; this guards coverage.
    //
    // Both sides now read the same ALL_ARCHES table, so this is a structural
    // check that neither lookup grew a filter of its own.
    for desc in ALL_ARCHES {
        let arch = desc.name;
        for name in std::iter::once(desc.name).chain(desc.aliases.iter().copied()) {
            let built = crate::arch::arch_from_name(name);
            assert!(
                built.is_some(),
                "{arch}: arch_from_name({name:?}) should be recognized",
            );
            assert_eq!(
                built.map(|a| a.name()),
                Some(desc.name),
                "{arch}: arch_from_name({name:?}) resolved to the wrong arch",
            );
            assert!(
                cc_for_arch(name).is_some(),
                "{arch}: arch_from_name accepts {name:?} but cc_for_arch has no CC for it \
                 (would panic in default_cc_for_arch)",
            );
        }
    }
}

/// Cross-check the CC register tables against the `Arch` register-name tables:
/// every offset a CC hands out must name the register the ABI documents. The
/// hardcoded-literal tests above are an independent oracle for the *numbers*;
/// this one ties those numbers to `Arch::register_name`, so a CC table and a
/// register table can no longer drift apart silently.
#[test]
fn test_cc_arg_registers_resolve_to_expected_register_names() {
    // (arch canonical name, C-ABI args, syscall args, return register)
    let expected: &[(&str, &[&str], &[&str], &str)] = &[
        (
            "X86",
            &[],
            &["ebx", "ecx", "edx", "esi", "edi", "ebp"],
            "eax",
        ),
        (
            "AMD64",
            &["rdi", "rsi", "rdx", "rcx", "r8", "r9"],
            &["rdi", "rsi", "rdx", "r10", "r8", "r9"],
            "rax",
        ),
        (
            "ARM",
            &["r0", "r1", "r2", "r3"],
            &["r0", "r1", "r2", "r3", "r4", "r5"],
            "r0",
        ),
        (
            "ARM64",
            &["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"],
            &["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"],
            "x0",
        ),
        (
            "MIPS32",
            &["a0", "a1", "a2", "a3"],
            &["a0", "a1", "a2", "a3"],
            "v0",
        ),
        (
            "MIPS64",
            &["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"],
            &["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"],
            "v0",
        ),
    ];

    let names = |arch: &dyn crate::arch::Arch, offsets: &[u32]| -> Vec<&'static str> {
        offsets
            .iter()
            .map(|&off| {
                arch.register_name(off).unwrap_or_else(|| {
                    panic!(
                        "{}: CC offset {off} has no canonical register name",
                        arch.name()
                    )
                })
            })
            .collect()
    };

    assert_eq!(
        expected.len(),
        ALL_ARCHES.len(),
        "expected-name table must cover every ALL_ARCHES row",
    );
    for desc in ALL_ARCHES {
        let arch = desc.name;
        let (_, want_args, want_syscall_args, want_ret) = expected
            .iter()
            .find(|(n, _, _, _)| *n == desc.name)
            .unwrap_or_else(|| panic!("{arch}: no expected-name row"));
        let cc = (desc.make_cc)();
        let a = (desc.make_arch)();

        assert_eq!(
            a.name(),
            desc.name,
            "{arch}: ALL_ARCHES name must equal Arch::name()",
        );
        assert_eq!(
            names(a.as_ref(), cc.arg_registers()),
            *want_args,
            "{arch}: C-ABI argument registers",
        );
        assert_eq!(
            names(a.as_ref(), cc.syscall_arg_registers()),
            *want_syscall_args,
            "{arch}: syscall argument registers",
        );
        assert_eq!(
            a.register_name(cc.return_register()),
            Some(*want_ret),
            "{arch}: return register",
        );
    }
}

/// Characterization (not aspiration) of `CallingConvention::link_register`.
///
/// The eventual invariant is `pops_return_addr() == false` implies
/// `link_register().is_some()` — a link-register ABI that does not name its LR
/// makes `setup_native_subcall` bail with `SubcallSetupError::UnsupportedAbi`
/// and silently defers every native sub-call to Python. Today **no** CC
/// overrides it, and that is load-bearing: the serial sub-call arm in
/// `RustExplorationManager::step_one` fills `NativeSubcall::caller_return_addr`
/// from `RustExplorationManager::get_return_addr`, which reads `[sp]`
/// unconditionally, so on ARM/ARM64/MIPS the `UnsupportedAbi` bail is the only
/// thing stopping a garbage return address from reaching
/// `state.set_pc(frame.caller_return_addr)`. See the `link_register` doc
/// comment for the full write-up.
///
/// So this test pins the status quo in both directions. When the serial path is
/// fixed and the overrides land, flip the `is_none()` arm to `is_some()` — the
/// name-resolution assertion below is the half that survives unchanged.
#[test]
fn test_link_register_is_unwired_pending_serial_subcall_fix() {
    for desc in ALL_ARCHES {
        let arch = desc.name;
        let cc = (desc.make_cc)();
        if cc.pops_return_addr() {
            assert!(
                cc.link_register().is_none(),
                "{arch}: stack-return ABI must not claim a link register",
            );
        } else {
            assert!(
                cc.link_register().is_none(),
                "{arch}: link_register() is now wired up — the serial sub-call \
                 path in run_loop.rs must be fixed to resolve caller_return_addr \
                 through the calling convention before this is safe",
            );
        }
        // Whatever an ABI does claim must be a real register on that arch.
        if let Some(off) = cc.link_register() {
            assert!(
                (desc.make_arch)().register_name(off).is_some(),
                "{arch}: link_register offset {off} has no canonical register name",
            );
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
