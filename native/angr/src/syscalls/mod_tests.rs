// NativeSyscallRegistry and support-helper unit tests. Declared as a direct
// child of `syscalls` so `use super::*` reaches the module's private items
// (`BY_NAME`, the registry's handler map).

use super::*;
use crate::symbolic::RustBV;

#[test]
fn exit_handler_returns_exit_outcome() {
    use crate::state::RustSimState;
    let h = exit::NativeExitSyscall;
    let mut state = RustSimState::new("amd64").expect("amd64 state");
    let outcome = h.call(&mut state, &[]).expect("exit handler succeeds");
    assert!(matches!(outcome, SyscallOutcome::Exit));
    assert_eq!(h.name(), "exit");
    assert_eq!(h.num_args(), 0);
}

#[test]
fn fresh_byte_names_uniquifies_per_batch_and_index() {
    // angr-9ke6b.158: the naming scheme all five mint-N-symbolic-bytes call
    // sites now share. Two properties are load-bearing (see `fresh_symbolic`
    // for why): the counter bumps once per *batch* so a batch's names are
    // `<prefix>_<id>_0..n`, and two batches from the same callsite never
    // collide (which would alias them to the same Z3 constant).
    const COUNTER: &str = "mod_tests::fresh_byte_names_batches";

    let first = fresh_byte_names("sys_probe", COUNTER, 3);
    let second = fresh_byte_names("sys_probe", COUNTER, 3);

    assert_eq!(
        first,
        vec!["sys_probe_0_0", "sys_probe_0_1", "sys_probe_0_2"]
    );
    assert_eq!(
        second,
        vec!["sys_probe_1_0", "sys_probe_1_1", "sys_probe_1_2"]
    );
    assert!(
        first.iter().all(|n| !second.contains(n)),
        "batches must not share names"
    );
}

#[test]
fn fresh_byte_names_empty_batch_still_bumps_the_counter() {
    // A zero-length request yields no names but must not hand the *next*
    // batch the id it would have used — every handler guards count==0 before
    // calling, so this only pins that the shared helper can't silently
    // reuse an id if a future caller drops that guard.
    const COUNTER: &str = "mod_tests::fresh_byte_names_empty";

    assert!(fresh_byte_names("sys_probe", COUNTER, 0).is_empty());
    assert_eq!(
        fresh_byte_names("sys_probe", COUNTER, 1),
        vec!["sys_probe_1_0"]
    );
}

/// angr-91vj9.1: `bounded_value` is the single boundary definition every
/// bounded syscall arg funnels through, so pin `<= max` as inclusive here
/// rather than re-deriving it at each handler (an at-cap `mmap`/`mprotect`
/// would need 65536 eagerly-allocated pages to observe the same edge).
#[test]
fn bounded_value_cap_is_inclusive() {
    assert_eq!(
        bounded_value("probe", MAX_MAP_SIZE, MAX_MAP_SIZE),
        BoundedArg::Within(MAX_MAP_SIZE)
    );
    assert_eq!(
        bounded_value("probe", MAX_MAP_SIZE - 1, MAX_MAP_SIZE),
        BoundedArg::Within(MAX_MAP_SIZE - 1)
    );
    assert_eq!(
        bounded_value("probe", MAX_MAP_SIZE + 1, MAX_MAP_SIZE),
        BoundedArg::Exceeds
    );
    assert_eq!(
        bounded_value("probe", u64::MAX, MAX_MAP_SIZE),
        BoundedArg::Exceeds
    );
}

#[test]
fn extract_bounded_concrete_arg_bounds_and_still_falls_back_on_symbolic() {

    assert_eq!(
        extract_bounded_concrete_arg(&RustBV::concrete(0x1000, 64), "probe", MAX_MAP_SIZE)
            .expect("concrete"),
        BoundedArg::Within(0x1000)
    );
    assert_eq!(
        extract_bounded_concrete_arg(
            &RustBV::concrete(u128::from(MAX_MAP_SIZE) + 1, 64),
            "probe",
            MAX_MAP_SIZE
        )
        .expect("concrete"),
        BoundedArg::Exceeds
    );

    // A symbolic arg is neither Within nor Exceeds — the cap must not turn a
    // Python fallback into a silent refusal.
    match extract_bounded_concrete_arg(
        &RustBV::symbol("probe_sym", 64),
        "probe",
        MAX_MAP_SIZE,
    ) {
        Err(SyscallError::SymbolicArgument(name)) => assert_eq!(name, "probe"),
        other => panic!("expected SymbolicArgument, got {other:?}"),
    }
}

// ======================================================================
// By-name registration (angr-mx00p.35)
// ======================================================================

fn angr_table(abi: &str, rows: &[(u64, &str)]) -> (String, HashMap<u64, String>) {
    (
        abi.to_string(),
        rows.iter().map(|&(n, name)| (n, name.to_string())).collect(),
    )
}

fn handler_name(r: &NativeSyscallRegistry, abi: SyscallAbi, num: u64) -> Option<&'static str> {
    r.handlers.get(&abi)?.get(&num).map(|h| h.name())
}

#[test]
fn from_angr_tables_resolves_numbers_by_name() {
    let r = NativeSyscallRegistry::from_angr_tables(&[
        angr_table("amd64", &[(0, "read"), (60, "exit"), (231, "exit_group"), (14, "rt_sigprocmask")]),
        angr_table("i386", &[(1, "exit"), (195, "stat64"), (106, "stat")]),
    ])
    .expect("known ABIs");
    assert_eq!(handler_name(&r, SyscallAbi::Amd64, 0), Some("read"));
    assert_eq!(handler_name(&r, SyscallAbi::Amd64, 60), Some("exit"));
    assert_eq!(handler_name(&r, SyscallAbi::Amd64, 231), Some("exit"));
    // A name with no row stays with Python.
    assert_eq!(handler_name(&r, SyscallAbi::Amd64, 14), None);
    assert_eq!(handler_name(&r, SyscallAbi::I386, 1), Some("exit"));
    assert_eq!(handler_name(&r, SyscallAbi::I386, 195), Some("stat"));
    // i386's pre-LFS stat is out of the `stat` row's scope.
    assert_eq!(handler_name(&r, SyscallAbi::I386, 106), None);
}

#[test]
fn from_angr_tables_skips_ignored_abis_and_rejects_unknown_ones() {
    let r = NativeSyscallRegistry::from_angr_tables(&[
        angr_table("arm", &[(1, "exit")]),
        angr_table("armhf", &[(0x90_0001, "exit")]),
        angr_table("mips-n32", &[(6058, "exit")]),
    ])
    .expect("armhf and mips-n32 are skipped");
    assert_eq!(handler_name(&r, SyscallAbi::Arm, 1), Some("exit"));
    assert_eq!(r.table().len(), 1, "nothing registered for the ignored ABIs");

    let err = NativeSyscallRegistry::from_angr_tables(&[angr_table("ppc", &[(1, "exit")])])
        .err()
        .expect("an ABI the engine was not written for must not dispatch");
    assert!(err.contains("ppc"), "{err}");
}

#[test]
fn mmap_name_means_the_struct_argument_form_on_arm_and_mips_o32() {
    for (abi, expected) in [
        (SyscallAbi::Amd64, "mmap"),
        (SyscallAbi::Aarch64, "mmap"),
        (SyscallAbi::MipsN64, "mmap"),
        (SyscallAbi::Arm, "old_mmap"),
        (SyscallAbi::MipsO32, "old_mmap"),
    ] {
        let h = NativeSyscallRegistry::handler_for(abi, "mmap").expect("mmap row");
        assert_eq!(h.name(), expected, "{abi:?}");
    }
    assert_eq!(
        NativeSyscallRegistry::handler_for(SyscallAbi::I386, "old_mmap").map(|h| h.name()),
        Some("old_mmap")
    );
}

#[test]
fn names_angr_lists_but_the_engine_must_not_run_are_out_of_scope() {
    // Pre-LFS struct stat has no writer on the 32-bit ABIs; mips-n64 has no
    // stat writer at all; aarch64 has no legacy stat/lstat.
    for abi in [SyscallAbi::I386, SyscallAbi::Arm, SyscallAbi::MipsO32, SyscallAbi::MipsN64] {
        for name in ["stat", "lstat", "fstat"] {
            assert!(NativeSyscallRegistry::handler_for(abi, name).is_none(), "{abi:?} {name}");
        }
    }
    assert!(NativeSyscallRegistry::handler_for(SyscallAbi::MipsN64, "newfstatat").is_none());
    assert!(NativeSyscallRegistry::handler_for(SyscallAbi::Aarch64, "stat").is_none());
    assert!(NativeSyscallRegistry::handler_for(SyscallAbi::Aarch64, "fstat").is_some());
    // arch_prctl writes fs_const / gs_const, which only AMD64 has.
    assert!(NativeSyscallRegistry::handler_for(SyscallAbi::I386, "arch_prctl").is_none());
    assert!(NativeSyscallRegistry::handler_for(SyscallAbi::Amd64, "arch_prctl").is_some());
    assert!(NativeSyscallRegistry::handler_for(SyscallAbi::Amd64, "rt_sigprocmask").is_none());
    // DECREE names never resolve on a Linux ABI and vice versa.
    assert!(NativeSyscallRegistry::handler_for(SyscallAbi::I386, "transmit").is_none());
    assert!(NativeSyscallRegistry::handler_for(SyscallAbi::Cgc, "write").is_none());
    assert_eq!(
        NativeSyscallRegistry::handler_for(SyscallAbi::Cgc, "_terminate").map(|h| h.name()),
        Some("exit")
    );
}

#[test]
fn by_name_rows_never_overlap() {
    for (i, a) in BY_NAME.iter().enumerate() {
        for b in &BY_NAME[i + 1..] {
            if a.name == b.name {
                assert!(
                    !a.scope.iter().any(|abi| b.scope.contains(abi)),
                    "two rows for {:?} share an ABI",
                    a.name
                );
            }
        }
    }
}

#[test]
fn amd64_dispatches_only_the_syscall_instruction() {
    let r = NativeSyscallRegistry::from_angr_tables(&[
        angr_table("i386", &[(1, "exit")]),
        angr_table("amd64", &[(1, "write"), (60, "exit")]),
    ])
    .expect("known ABIs");
    let name = |arch: &str, jk: JumpKind, num: u64| r.get_for_exit(arch, jk, num).map(|h| h.name());
    assert_eq!(name("AMD64", JumpKind::Sys_syscall, 1), Some("write"));
    // int 0x80 in 64-bit code is the i386 ABI: never the amd64 handler.
    assert_eq!(name("AMD64", JumpKind::Sys_int128, 1), None);
    assert_eq!(name("AMD64", JumpKind::Sys_int, 60), None);
    // On X86 every syscall exit is the i386 ABI.
    assert_eq!(name("X86", JumpKind::Sys_int128, 1), Some("exit"));
    assert_eq!(name("X86", JumpKind::Sys_sysenter, 1), Some("exit"));
}

/// A CGC project's SimOS lists only `cgcabi`, and a registry holding the
/// DECREE table dispatches every syscall through it: 2 is `transmit`, not
/// i386 `fork`.
#[test]
fn a_decree_registry_dispatches_through_the_decree_table() {
    let r = NativeSyscallRegistry::from_angr_tables(&[angr_table(
        "cgcabi",
        &[(1, "_terminate"), (2, "transmit")],
    )])
    .expect("cgcabi");
    assert_eq!(
        r.get_for_exit("X86", JumpKind::Sys_int128, 2).map(|h| h.name()),
        Some("transmit")
    );
    let linux = NativeSyscallRegistry::from_angr_tables(&[angr_table("i386", &[(2, "fork")])])
        .expect("i386");
    assert!(linux.get_for_exit("X86", JumpKind::Sys_int128, 2).is_none());
}
