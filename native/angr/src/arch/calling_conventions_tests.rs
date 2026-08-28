//! Tests for the per-ABI calling convention tables.
//!
//! Extracted from `calling_conventions.rs` (see bd `rust-mod-tests-sibling-extraction`).

use super::*;
use crate::arch::{ALL_ARCHES, Arch, MIPS32, MIPS64};
use crate::symbolic::RustBV;

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
    // the C ABI registers. AArch64 and MipsO32 don't override, for two
    // different reasons:
    //   - AArch64: the C ABI window X0-X7 is a strict *superset* of the
    //     Linux syscall window X0-X5 (X8 carries the syscall number). No
    //     Linux syscall takes more than 6 args, so the extra X6/X7 slots
    //     are never read and the superset is harmless.
    //   - MipsO32: the syscall ABI passes args in $a0-$a3 exactly like the
    //     C ABI ($v0 carries the number, args 5+ spill to the stack in
    //     both), so the fallback is an exact match, not a superset.
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

/// The MIPS O32/N64 Linux syscall ABIs report failure in a *second* register
/// (`$a3`) rather than encoding it in the return value, and
/// `CcSnapshot::write_syscall_return` splits the two apart from this tuple.
/// A drifted offset or threshold would silently mis-classify every MIPS
/// syscall, so pin both halves — and pin that no other ABI claims one, since a
/// spurious `Some` would rewrite successful returns as `-errno` on an arch
/// whose kernel never sets a flag register.
#[test]
fn test_syscall_error_register_only_on_mips_and_names_a3() {
    // Mirrors SimCC{O32,N64}LinuxSyscall.SYSCALL_ERRNO_START in
    // angr/calling_conventions.py.
    const MIPS_ERRNO_START: i64 = -1133;

    assert_eq!(
        MipsO32.syscall_error_register(),
        Some((MIPS32.register_offset("a3").unwrap(), MIPS_ERRNO_START)),
    );
    assert_eq!(
        MipsN64.syscall_error_register(),
        Some((MIPS64.register_offset("a3").unwrap(), MIPS_ERRNO_START)),
    );
    // $a3 is R7 on both, but the MIPS64 register file has 8-byte slots, so the
    // two offsets differ and neither tuple may be reused for the other arch.
    assert_ne!(
        MipsO32.syscall_error_register(),
        MipsN64.syscall_error_register(),
    );

    for desc in ALL_ARCHES {
        let arch = desc.name;
        let cc = (desc.make_cc)();
        let is_mips = arch == "MIPS32" || arch == "MIPS64";
        match cc.syscall_error_register() {
            None => assert!(
                !is_mips,
                "{arch}: the MIPS Linux syscall ABI must name its $a3 errno flag",
            ),
            Some((off, errno_start)) => {
                assert!(
                    is_mips,
                    "{arch}: only MIPS splits the errno flag into a second register; \
                     a spurious Some rewrites every large successful return as -errno",
                );
                assert_eq!(
                    (desc.make_arch)().register_name(off),
                    Some("a3"),
                    "{arch}: syscall_error_register offset {off} must be $a3",
                );
                assert_eq!(errno_start, MIPS_ERRNO_START, "{arch}: errno threshold");
            }
        }
    }
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

/// `CallingConvention::link_register` invariant (bead angr-9ke6b.3).
///
/// `pops_return_addr() == false` implies `link_register().is_some()` — a
/// link-register ABI that does not name its LR makes `setup_native_subcall`
/// bail with `SubcallSetupError::UnsupportedAbi` and silently defers every
/// native sub-call to Python. The converse also holds: a stack-return ABI
/// (x86/AMD64) must not claim one, since its return address lives at `[sp]`.
#[test]
fn test_link_register_set_on_link_register_abis() {
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
                cc.link_register().is_some(),
                "{arch}: link-register ABI must name its LR, otherwise every \
                 native sub-call falls back to the Python SimProcedure path",
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

/// `link_register()` must name the *same* register `get_return_addr()` reads,
/// otherwise `setup_native_subcall` would write the resume sentinel into one
/// register while the dispatcher captured the caller return address from
/// another. The trait's default `get_return_addr` derives one from the other,
/// so this locks that any future per-CC override keeps them in sync.
#[test]
fn test_link_register_matches_get_return_addr_register() {
    let ctx = SymContext::new_mock();
    for desc in ALL_ARCHES {
        let arch = desc.name;
        let cc = (desc.make_cc)();
        let Some(lr) = cc.link_register() else {
            continue;
        };
        let mut regs = RegisterFile::new((desc.make_arch)());
        let marker = 0x0040_1234u64;
        regs.put(lr, RustBV::concrete(marker as u128, cc.pointer_size() * 8));
        assert_eq!(
            cc.get_return_addr(&regs, None, &ctx),
            Some(marker),
            "{arch}: get_return_addr must read the register link_register() names",
        );
    }
}

// Argument extraction itself is not tested here: the one implementation lives
// in `exploration::helpers::extract_args_with_abi` and is pinned by
// `exploration/helpers_tests.rs` (register window, stack spill, SpSymbolic,
// StackUnmapped, RegisterOverflow). The register *tables* this module owns are
// pinned by the per-ABI tests above and below (angr-9iny6).

// --- fp_arg_registers provenance pins (angr-9ke6b.218 item 7) ---------------
//
// `fp_arg_registers` has no production caller: argument extraction is
// integer-only. It is kept because each table encodes ABI provenance that
// FP-argument support will need verbatim, and these tests are what makes
// keeping it defensible — without them the tables could rot silently in the
// years before the first real consumer arrives.

#[test]
fn fp_arg_registers_amd64_is_xmm0_through_xmm7() {
    let cc = SystemVAMD64;
    assert_eq!(
        cc.fp_arg_registers(),
        &[
            amd64_off::XMM0,
            amd64_off::XMM1,
            amd64_off::XMM2,
            amd64_off::XMM3,
            amd64_off::XMM4,
            amd64_off::XMM5,
            amd64_off::XMM6,
            amd64_off::XMM7,
        ][..],
        "SysV AMD64 passes the first 8 FP args in XMM0-XMM7",
    );
    assert_eq!(
        cc.fp_arg_registers().first().copied(),
        cc.fp_return_register(),
        "XMM0 is both the first FP arg slot and the scalar-double return slot",
    );
}

#[test]
fn fp_arg_registers_x86_cdecl_is_empty() {
    // cdecl passes every float on the stack; there is no FP register window.
    assert!(Cdecl.fp_arg_registers().is_empty());
}

#[test]
fn fp_arg_registers_arm_is_d0_through_d7() {
    // AAPCS-VFP (hard-float). Soft-float ARM EABI uses the integer pairs
    // instead, which this table deliberately does not model.
    assert_eq!(
        ARMEABI.fp_arg_registers(),
        &[
            arm_off::D0,
            arm_off::D1,
            arm_off::D2,
            arm_off::D3,
            arm_off::D4,
            arm_off::D5,
            arm_off::D6,
            arm_off::D7,
        ][..],
    );
}

#[test]
fn fp_arg_registers_arm64_is_v0_through_v7() {
    let cc = AArch64CC;
    assert_eq!(
        cc.fp_arg_registers(),
        &[
            arm64_off::Q0,
            arm64_off::Q1,
            arm64_off::Q2,
            arm64_off::Q3,
            arm64_off::Q4,
            arm64_off::Q5,
            arm64_off::Q6,
            arm64_off::Q7,
        ][..],
        "AAPCS64 passes the first 8 FP args in V0-V7, the low halves of Q0-Q7",
    );
    assert_eq!(
        cc.fp_arg_registers().first().copied(),
        cc.fp_return_register(),
        "V0 is both the first FP arg slot and the scalar-double return slot",
    );
}

#[test]
fn fp_arg_registers_mips_is_empty_by_design() {
    // Both MIPS tables are empty ON PURPOSE: O32 passes doubles in $f12/$f14
    // and N64 in $f12-$f19, but which of those vs the integer window a given
    // argument lands in depends on the float/int position rules and on whether
    // the binary is soft-float. We do not model that dispatch, so an extractor
    // handed these offsets would read the wrong slot rather than no slot.
    assert!(MipsO32.fp_arg_registers().is_empty());
    assert!(MipsN64.fp_arg_registers().is_empty());
    // The f-registers exist in the register file, so emptiness is a modelling
    // decision, not a missing-offset one.
    assert_eq!(MIPS32.register_offset("f12"), Some(mips32_off::F12));
    assert_eq!(MIPS64.register_offset("f12"), Some(mips64_off::F12));
}

#[test]
fn fp_arg_register_offsets_are_distinct_and_evenly_strided() {
    // What a future FP-argument extractor actually needs from these tables:
    // N distinct slots, ascending, one register width apart, so `regs[i]` is
    // the i-th FP argument. (Deliberately NOT asserted: that `register_name`
    // resolves them. It does not — ARM's `d0`-`d7` and AArch64's `q0`-`q7`
    // live in each arch's ALIASES table, and `register_name` searches only
    // CANONICAL. That is the same position MIPS' `$f12`/`$f14` are in, so
    // naming is not what separates the populated tables from the empty ones.)
    let cases: [(&dyn CallingConvention, &str, u32); 3] = [
        // 32, not 16: VEX lays amd64 out as 256-bit YMM slots and XMMn is
        // the low half of YMMn.
        (&SystemVAMD64, "amd64", 32),
        (&ARMEABI, "arm", 8),
        (&AArch64CC, "arm64", 16),
    ];
    for (cc, label, stride) in cases {
        let regs = cc.fp_arg_registers();
        assert_eq!(regs.len(), 8, "{label} FP arg window is 8 registers wide");
        let unique: std::collections::HashSet<u32> = regs.iter().copied().collect();
        assert_eq!(
            unique.len(),
            regs.len(),
            "{label} FP arg offsets must be distinct"
        );
        for (i, w) in regs.windows(2).enumerate() {
            assert_eq!(
                w[1] - w[0],
                stride,
                "{label} FP arg slots {i} and {} are not one register apart",
                i + 1
            );
        }
    }
}
