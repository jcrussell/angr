// angr-ist1: NativeSyscallRegistry unit tests, extracted out of the former
// in-file `mod tests` (~1077 lines) into a sibling file to shrink
// syscalls/mod.rs below the god-object threshold. Declared as a direct child
// of `syscalls` so `use super::*` reaches the module's private items.

use super::*;

#[test]
fn default_registry_has_amd64_exit_handlers() {
    let r = NativeSyscallRegistry::new();
    assert!(r.get("AMD64", 0).is_some(), "read (0) should be registered");
    assert!(
        r.get("AMD64", 1).is_some(),
        "write (1) should be registered"
    );
    assert!(
        r.get("AMD64", 60).is_some(),
        "exit (60) should be registered"
    );
    assert!(
        r.get("AMD64", 231).is_some(),
        "exit_group (231) should be registered"
    );
    assert!(
        r.get("AMD64", 10).is_some(),
        "mprotect (10) should be registered"
    );
    assert!(
        r.get("AMD64", 12).is_some(),
        "brk (12) should be registered"
    );
    assert!(
        r.get("AMD64", 11).is_some(),
        "munmap (11) should be registered"
    );
    assert!(r.get("AMD64", 9).is_some(), "mmap (9) should be registered");
    assert!(
        r.get("AMD64", 13).is_some(),
        "rt_sigaction (13) should be registered"
    );
    assert!(
        r.get("AMD64", 96).is_some(),
        "gettimeofday (96) should be registered"
    );
    assert!(
        r.get("AMD64", 158).is_some(),
        "arch_prctl (158) should be registered"
    );
    assert!(
        r.get("AMD64", 201).is_some(),
        "time (201) should be registered"
    );
    assert!(
        r.get("AMD64", 228).is_some(),
        "clock_gettime (228) should be registered"
    );
    assert!(
        r.get("X86", 60).is_none(),
        "amd64 numbers don't apply to x86"
    );
}

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

// ======================================================================
// Per-arch dispatch tests (angr-7xms)
// ======================================================================
//
// These verify that the Linux syscall numbers for each arch route to a
// registered handler, and that arch isolation holds (an x86 number does
// not collide with the AMD64 table). Each arch test enumerates the
// representative syscalls registered in `register_<arch>` so adding /
// removing one without updating the test fails noisily.

#[test]
fn x86_syscall_numbers_route_to_handlers() {
    let r = NativeSyscallRegistry::new();
    for (num, label) in [
        (1, "exit"),
        (3, "read"),
        (4, "write"),
        (13, "time"),
        (20, "getpid"),
        (24, "getuid"),
        (45, "brk"),
        (47, "getgid"),
        (49, "geteuid"),
        (50, "getegid"),
        (64, "getppid"),
        (78, "gettimeofday"),
        (91, "munmap"),
        (125, "mprotect"),
        (174, "rt_sigaction"),
        (199, "getuid32"),
        (200, "getgid32"),
        (201, "geteuid32"),
        (202, "getegid32"),
        (224, "gettid"),
        (252, "exit_group"),
        (265, "clock_gettime"),
    ] {
        assert!(
            r.get("X86", num).is_some(),
            "X86 syscall {num} ({label}) should be registered",
        );
    }
    // i386 mmap family (angr-6gmc): old_mmap (90, struct-arg) and
    // mmap2 (192, page-offset) are now native.
    assert!(
        r.get("X86", 90).is_some(),
        "x86 old_mmap (90) should be registered"
    );
    assert!(
        r.get("X86", 192).is_some(),
        "x86 mmap2 (192) should be registered"
    );
}

#[test]
fn arm_syscall_numbers_route_to_handlers() {
    let r = NativeSyscallRegistry::new();
    for (num, label) in [
        (1, "exit"),
        (3, "read"),
        (4, "write"),
        (13, "time"),
        (20, "getpid"),
        (24, "getuid"),
        (45, "brk"),
        (47, "getgid"),
        (49, "geteuid"),
        (50, "getegid"),
        (64, "getppid"),
        (78, "gettimeofday"),
        (91, "munmap"),
        (125, "mprotect"),
        (174, "rt_sigaction"),
        (199, "getuid32"),
        (200, "getgid32"),
        (201, "geteuid32"),
        (202, "getegid32"),
        (224, "gettid"),
        (248, "exit_group"),
        (263, "clock_gettime"),
    ] {
        assert!(
            r.get("ARM", num).is_some(),
            "ARM syscall {num} ({label}) should be registered",
        );
    }
    // ARM mmap family (angr-6gmc): old_mmap (90) and mmap2 (192) native.
    assert!(
        r.get("ARM", 90).is_some(),
        "ARM old_mmap (90) should be registered"
    );
    assert!(
        r.get("ARM", 192).is_some(),
        "ARM mmap2 (192) should be registered"
    );
}

#[test]
fn arm64_syscall_numbers_route_to_handlers() {
    let r = NativeSyscallRegistry::new();
    for (num, label) in [
        (63, "read"),
        (64, "write"),
        (93, "exit"),
        (94, "exit_group"),
        (113, "clock_gettime"),
        (134, "rt_sigaction"),
        (169, "gettimeofday"),
        (172, "getpid"),
        (173, "getppid"),
        (174, "getuid"),
        (175, "geteuid"),
        (176, "getgid"),
        (177, "getegid"),
        (178, "gettid"),
        (214, "brk"),
        (215, "munmap"),
        (222, "mmap"),
        (226, "mprotect"),
    ] {
        assert!(
            r.get("ARM64", num).is_some(),
            "ARM64 syscall {num} ({label}) should be registered",
        );
    }
}

#[test]
fn mips32_syscall_numbers_route_to_handlers() {
    let r = NativeSyscallRegistry::new();
    for (num, label) in [
        (4001, "exit"),
        (4003, "read"),
        (4004, "write"),
        (4013, "time"),
        (4020, "getpid"),
        (4024, "getuid"),
        (4045, "brk"),
        (4047, "getgid"),
        (4049, "geteuid"),
        (4050, "getegid"),
        (4064, "getppid"),
        (4078, "gettimeofday"),
        (4091, "munmap"),
        (4125, "mprotect"),
        (4194, "rt_sigaction"),
        (4222, "gettid"),
        (4246, "exit_group"),
        (4263, "clock_gettime"),
    ] {
        assert!(
            r.get("MIPS32", num).is_some(),
            "MIPS32 syscall {num} ({label}) should be registered",
        );
    }
    // MIPS32 old_mmap (4090) registered (angr-6gmc). mmap2 (4210)
    // registered (angr-tvod) — 6 args, args 5-6 read from [sp+16].
    assert!(
        r.get("MIPS32", 4090).is_some(),
        "MIPS32 old_mmap (4090) should be registered"
    );
    assert!(
        r.get("MIPS32", 4210).is_some(),
        "MIPS32 mmap2 (4210) should be registered (angr-tvod)"
    );
}

#[test]
fn mips32_at_family_uses_angr_divergent_numbers() {
    // angr-cudgw.1 / bd memory `angr-syscall-mips-o32-table-divergence`:
    // angr's MIPS-O32 syscall table (linux_kernel.py, `mips-o32` block)
    // diverges from upstream Linux <asm/unistd_o32.h> for part of the
    // *at family — several entries sit +2 above their upstream number.
    // Native handlers MUST register against angr's numbers, because the
    // dispatcher receives the angr-issued number when a name-based
    // syscall fires; using the upstream number would silently
    // de-register the handler. This test pins the divergent numbers so a
    // naive "match upstream" edit re-reddens instead of regressing.
    let r = NativeSyscallRegistry::new();
    // (name, angr number, upstream number). angr == upstream for
    // openat/readlinkat/faccessat; +2 for mkdirat/unlinkat/renameat.
    for (name, angr_num, upstream_num) in [
        ("openat", 4288u64, 4288u64),
        ("mkdirat", 4289, 4287),
        ("unlinkat", 4294, 4292),
        ("renameat", 4295, 4293),
        ("readlinkat", 4298, 4298),
        ("faccessat", 4300, 4300),
    ] {
        assert!(
            r.get("MIPS32", angr_num).is_some(),
            "MIPS32 {name} must register at angr number {angr_num}",
        );
        if angr_num != upstream_num {
            // The upstream slot must NOT carry THIS name's handler — a +2
            // edit toward upstream would land here and de-register the name.
            // The slot may legitimately hold a *different* handler: since
            // angr-11djq.5.3, 4293 (upstream renameat) is angr's fstatat64
            // and IS registered natively, so we check the handler's name
            // rather than emptiness. (4287/4292 stay unmapped.)
            if let Some(h) = r.get("MIPS32", upstream_num) {
                assert_ne!(
                    h.name(),
                    name,
                    "MIPS32 upstream number {upstream_num} must not carry \
                     {name} — angr's table uses {angr_num}",
                );
            }
        }
    }
}

#[test]
fn mips64_syscall_numbers_route_to_handlers() {
    // angr-smtv: MIPS64 N64 dispatch. Numbers from asm/unistd_n64.h
    // (also mirrored in angr/procedures/definitions/linux_kernel.py
    // `mips-n64` table).
    let r = NativeSyscallRegistry::new();
    for (num, label) in [
        (5000, "read"),
        (5001, "write"),
        (5009, "mmap"),
        (5010, "mprotect"),
        (5011, "munmap"),
        (5012, "brk"),
        (5013, "rt_sigaction"),
        (5038, "getpid"),
        (5058, "exit"),
        (5070, "fcntl"),
        (5094, "gettimeofday"),
        (5100, "getuid"),
        (5102, "getgid"),
        (5105, "geteuid"),
        (5106, "getegid"),
        (5108, "getppid"),
        (5178, "gettid"),
        (5205, "exit_group"),
        (5222, "clock_gettime"),
    ] {
        assert!(
            r.get("MIPS64", num).is_some(),
            "MIPS64 syscall {num} ({label}) should be registered",
        );
    }
    // N64 uses the modern 6-arg mmap (5009); old_mmap (90 on i386/arm
    // or 4090 on MIPS32-O32) and mmap2 do not exist in the N64 table.
    assert!(
        r.get("MIPS64", 5009).is_some(),
        "MIPS64 mmap (5009) should be registered"
    );
    // angr-9ke6b.154: the stat family stays unregistered on N64 —
    // `write_stat_for_arch` has no MIPS64 writer, so a registration
    // would only add a dispatch hop before falling back to Python.
    for n in [5004u64, 5005, 5006, 5252] {
        assert!(
            r.get("MIPS64", n).is_none(),
            "MIPS64 stat-family ({n}) must fall back to Python"
        );
    }
    // N64 is already 64-bit so it has no separate fcntl64.
    assert!(
        r.get("MIPS64", 5070).is_some(),
        "MIPS64 fcntl (5070) should be registered"
    );
}

#[test]
fn dbb1_fd_io_and_startup_handlers_route_per_arch() {
    // angr-dbb1: lseek/readv/writev/uname/set_tid_address/set_robust_list
    // + pread64/pwrite64 registered on every arch with per-arch numbers;
    // getrandom only on AMD64 + X86 (absent from angr's other tables).
    let r = NativeSyscallRegistry::new();
    // (arch, lseek, readv, writev, uname, set_tid, set_robust, pread, pwrite)
    let table = [
        ("AMD64", 8, 19, 20, 63, 218, 273, 17, 18),
        ("X86", 19, 145, 146, 122, 258, 311, 180, 181),
        ("ARM", 19, 145, 146, 122, 256, 338, 180, 181),
        ("ARM64", 62, 65, 66, 160, 96, 99, 67, 68),
        ("MIPS32", 4019, 4145, 4146, 4122, 4252, 4309, 4200, 4201),
        ("MIPS64", 5008, 5018, 5019, 5061, 5212, 5268, 5016, 5017),
    ];
    for &(arch, lseek, readv, writev, uname, set_tid, set_robust, pread, pwrite) in &table {
        for (num, name) in [
            (lseek, "lseek"),
            (readv, "readv"),
            (writev, "writev"),
            (uname, "uname"),
            (set_tid, "set_tid_address"),
            (set_robust, "set_robust_list"),
            (pread, "pread64"),
            (pwrite, "pwrite64"),
        ] {
            let h = r
                .get(arch, num)
                .unwrap_or_else(|| panic!("{arch} {name} ({num}) should be registered"));
            assert_eq!(h.name(), name, "{arch} {num} routed to wrong handler");
        }
    }
    // getrandom: present on AMD64 (318) + X86 (355); absent elsewhere.
    assert_eq!(r.get("AMD64", 318).unwrap().name(), "getrandom");
    assert_eq!(r.get("X86", 355).unwrap().name(), "getrandom");
}

#[test]
fn arch_namespaces_are_isolated() {
    // Same numeric value must NOT collide across arches. e.g. X86 syscall
    // 1 = exit, ARM 1 = exit, but ARM64 1 is unassigned. Each lookup
    // must respect the arch key.
    let r = NativeSyscallRegistry::new();
    assert!(r.get("ARM64", 1).is_none(), "ARM64 has no syscall 1");
    assert!(r.get("AMD64", 1).is_some(), "AMD64 syscall 1 = write");
    // 60 = AMD64 exit, but X86's exit is 1, not 60.
    assert!(r.get("X86", 60).is_none(), "X86 has no syscall 60");
    // 4001 = MIPS32 exit. Other arches should not see it.
    assert!(r.get("AMD64", 4001).is_none(), "AMD64 has no syscall 4001");
    assert!(r.get("X86", 4001).is_none(), "X86 has no syscall 4001");
    // 5058 = MIPS64 exit. Other arches (including MIPS32) should not
    // see it; the MIPS-O32 and MIPS-N64 numbering spaces are disjoint.
    assert!(r.get("MIPS64", 5058).is_some(), "MIPS64 exit = 5058");
    assert!(
        r.get("MIPS32", 5058).is_none(),
        "MIPS32 has no syscall 5058"
    );
    assert!(r.get("AMD64", 5058).is_none(), "AMD64 has no syscall 5058");
    assert!(
        r.get("MIPS64", 4001).is_none(),
        "MIPS64 has no syscall 4001"
    );
}

#[test]
fn identity_syscalls_registered_on_all_arches() {
    // angr-0hif.3: getpid / getppid / gettid / getuid / geteuid /
    // getgid / getegid should be hookable on every supported arch.
    // arch → (getpid, getppid, gettid, getuid, geteuid, getgid, getegid)
    let r = NativeSyscallRegistry::new();
    for (arch, nums) in [
        ("AMD64", (39, 110, 186, 102, 107, 104, 108)),
        ("X86", (20, 64, 224, 24, 49, 47, 50)),
        ("ARM", (20, 64, 224, 24, 49, 47, 50)),
        ("ARM64", (172, 173, 178, 174, 175, 176, 177)),
        ("MIPS32", (4020, 4064, 4222, 4024, 4049, 4047, 4050)),
        ("MIPS64", (5038, 5108, 5178, 5100, 5105, 5102, 5106)),
    ] {
        let (pid, ppid, tid, uid, euid, gid, egid) = nums;
        for (num, label) in [
            (pid, "getpid"),
            (ppid, "getppid"),
            (tid, "gettid"),
            (uid, "getuid"),
            (euid, "geteuid"),
            (gid, "getgid"),
            (egid, "getegid"),
        ] {
            let h = r
                .get(arch, num)
                .unwrap_or_else(|| panic!("{arch} {label} ({num}) handler missing"));
            assert_eq!(h.name(), label, "{arch} syscall {num} should be {label}");
            assert_eq!(h.num_args(), 0, "{arch} {label} takes 0 args");
        }
    }
}

#[test]
fn setuid_setgid_registered_on_all_arches() {
    // angr-pqgu: setuid / setgid have no Python SimProcedure, so the
    // unhandled-syscall path falls through to syscall_stub which
    // returns a fresh symbolic value. The native handlers mirror
    // that via SyscallOutcome::ContinueSymbolic; they should be
    // registered (with name() == "setuid"/"setgid", 1 arg) on every
    // supported arch. Legacy and LFS variants on x86/ARM both alias
    // to the same handler — the syscall semantics are identical
    // and angr's prototype is also the same.
    let r = NativeSyscallRegistry::new();
    // arch → (setuid_num, setgid_num)
    for (arch, setuid, setgid) in [
        ("AMD64", 105u64, 106u64),
        ("X86", 23, 46),
        ("ARM", 23, 46),
        ("ARM64", 146, 144),
        ("MIPS32", 4023, 4046),
        ("MIPS64", 5103, 5104),
    ] {
        let u = r
            .get(arch, setuid)
            .unwrap_or_else(|| panic!("{arch} setuid ({setuid}) missing"));
        assert_eq!(u.name(), "setuid", "{arch} {setuid} should be setuid");
        assert_eq!(u.num_args(), 1, "{arch} setuid takes 1 arg");
        let g = r
            .get(arch, setgid)
            .unwrap_or_else(|| panic!("{arch} setgid ({setgid}) missing"));
        assert_eq!(g.name(), "setgid", "{arch} {setgid} should be setgid");
        assert_eq!(g.num_args(), 1, "{arch} setgid takes 1 arg");
    }
    // x86/ARM LFS variants (32-bit uid_t/gid_t) share the handler.
    for arch in ["X86", "ARM"] {
        let u = r
            .get(arch, 213)
            .unwrap_or_else(|| panic!("{arch} setuid32 (213) missing"));
        assert_eq!(u.name(), "setuid");
        let g = r
            .get(arch, 214)
            .unwrap_or_else(|| panic!("{arch} setgid32 (214) missing"));
        assert_eq!(g.name(), "setgid");
    }
}

#[test]
fn memory_extras_registered_on_all_arches() {
    // angr-0hif.4: madvise / mremap / msync / mlock / munlock /
    // mlockall / munlockall have no Python SimProcedure, so the
    // unhandled-syscall path falls through to `syscall_stub` which
    // returns a fresh symbolic. The native handlers mirror that via
    // SyscallOutcome::ContinueSymbolic; they must be registered with
    // the right name + arity on every supported arch.
    let r = NativeSyscallRegistry::new();
    // arch -> (madvise, mremap, msync, mlock, munlock, mlockall, munlockall)
    let table: &[(&str, [u64; 7])] = &[
        ("AMD64", [28, 25, 26, 149, 150, 151, 152]),
        ("X86", [219, 163, 144, 150, 151, 152, 153]),
        ("ARM", [220, 163, 144, 150, 151, 152, 153]),
        ("ARM64", [233, 216, 227, 228, 229, 230, 231]),
        ("MIPS32", [4218, 4167, 4144, 4154, 4155, 4156, 4157]),
        ("MIPS64", [5027, 5024, 5025, 5146, 5147, 5148, 5149]),
    ];
    let labels = [
        ("madvise", 3usize),
        ("mremap", 5),
        ("msync", 3),
        ("mlock", 2),
        ("munlock", 2),
        ("mlockall", 1),
        ("munlockall", 0),
    ];
    for (arch, nums) in table {
        for (i, (label, nargs)) in labels.iter().enumerate() {
            let num = nums[i];
            let h = r
                .get(arch, num)
                .unwrap_or_else(|| panic!("{arch} {label} ({num}) handler missing"));
            assert_eq!(h.name(), *label, "{arch} syscall {num} should be {label}");
            assert_eq!(
                h.num_args(),
                *nargs,
                "{arch} {label} should take {nargs} args"
            );
        }
    }
}

#[test]
fn signals_registered_on_all_arches() {
    // angr-0hif.6: kill / tgkill / rt_sigreturn / pause / alarm.
    // kill, rt_sigreturn, pause, alarm have no Python SimProcedure;
    // tgkill is concrete-0 (matches procedures/linux_kernel/tgkill.py).
    // ARM64 does NOT define `pause` (29 on i386/arm) or `alarm` (27)
    // in asm-generic, so only kill/tgkill/rt_sigreturn land there.
    let r = NativeSyscallRegistry::new();

    // (arch, kill, tgkill, rt_sigreturn, pause-or-None, alarm-or-None)
    type SignalRow = (&'static str, u64, u64, u64, Option<u64>, Option<u64>);
    let table: &[SignalRow] = &[
        ("AMD64", 62, 234, 15, Some(34), Some(37)),
        ("X86", 37, 270, 173, Some(29), Some(27)),
        ("ARM", 37, 268, 173, Some(29), Some(27)),
        ("ARM64", 129, 131, 139, None, None),
        ("MIPS32", 4037, 4266, 4193, Some(4029), Some(4027)),
        ("MIPS64", 5060, 5225, 5211, Some(5033), Some(5037)),
    ];

    for &(arch, kill_n, tgkill_n, rtret_n, pause_n, alarm_n) in table {
        let kill = r
            .get(arch, kill_n)
            .unwrap_or_else(|| panic!("{arch} kill ({kill_n}) missing"));
        assert_eq!(kill.name(), "kill");
        assert_eq!(kill.num_args(), 2);

        let tg = r
            .get(arch, tgkill_n)
            .unwrap_or_else(|| panic!("{arch} tgkill ({tgkill_n}) missing"));
        assert_eq!(tg.name(), "tgkill");
        assert_eq!(tg.num_args(), 3);

        let rtret = r
            .get(arch, rtret_n)
            .unwrap_or_else(|| panic!("{arch} rt_sigreturn ({rtret_n}) missing"));
        assert_eq!(rtret.name(), "rt_sigreturn");
        assert_eq!(rtret.num_args(), 0);

        if let Some(n) = pause_n {
            let p = r
                .get(arch, n)
                .unwrap_or_else(|| panic!("{arch} pause ({n}) missing"));
            assert_eq!(p.name(), "pause");
            assert_eq!(p.num_args(), 0);
        }
        if let Some(n) = alarm_n {
            let a = r
                .get(arch, n)
                .unwrap_or_else(|| panic!("{arch} alarm ({n}) missing"));
            assert_eq!(a.name(), "alarm");
            assert_eq!(a.num_args(), 1);
        }
    }
}

#[test]
fn rlimit_registered_on_all_arches() {
    // angr-0hif.7: getrlimit / setrlimit / prlimit64.
    // getrlimit mirrors procedures/linux_kernel/getrlimit.py — the
    // RLIMIT_STACK branch writes 8388608 to *rlim and returns 0;
    // other resources return a fresh symbolic. setrlimit/prlimit64
    // have no Python SimProcedure and fall through to syscall_stub.
    // x86 and ARM also expose `ugetrlimit` (191), aliased to the
    // same handler since Python defines ugetrlimit(getrlimit).
    let r = NativeSyscallRegistry::new();

    // (arch, getrlimit, setrlimit, prlimit64, ugetrlimit-or-None)
    let table: &[(&str, u64, u64, u64, Option<u64>)] = &[
        ("AMD64", 97, 160, 302, None),
        ("X86", 76, 75, 340, Some(191)),
        ("ARM", 76, 75, 369, Some(191)),
        ("ARM64", 163, 164, 261, None),
        ("MIPS32", 4076, 4075, 4338, None),
        // N64 has no `ugetrlimit` alias — `getrlimit` is already
        // 64-bit-clean at 5095.
        ("MIPS64", 5095, 5155, 5297, None),
    ];

    for &(arch, get_n, set_n, pr_n, uget_n) in table {
        let g = r
            .get(arch, get_n)
            .unwrap_or_else(|| panic!("{arch} getrlimit ({get_n}) missing"));
        assert_eq!(g.name(), "getrlimit");
        assert_eq!(g.num_args(), 2);

        let s = r
            .get(arch, set_n)
            .unwrap_or_else(|| panic!("{arch} setrlimit ({set_n}) missing"));
        assert_eq!(s.name(), "setrlimit");
        assert_eq!(s.num_args(), 2);

        let p = r
            .get(arch, pr_n)
            .unwrap_or_else(|| panic!("{arch} prlimit64 ({pr_n}) missing"));
        assert_eq!(p.name(), "prlimit64");
        assert_eq!(p.num_args(), 4);

        if let Some(un) = uget_n {
            let u = r
                .get(arch, un)
                .unwrap_or_else(|| panic!("{arch} ugetrlimit ({un}) missing"));
            // Alias: ugetrlimit shares the getrlimit handler.
            assert_eq!(
                u.name(),
                "getrlimit",
                "{arch} ugetrlimit must alias to getrlimit"
            );
        }
    }
}

#[test]
fn concurrency_registered_on_all_arches() {
    // angr-0hif.7: futex + eventfd/eventfd2 + epoll_create/_create1/
    // _ctl/_wait. ARM64 (asm-generic) drops legacy epoll_create,
    // epoll_wait, and 1-arg eventfd — only the *_create1 / epoll_ctl
    // / epoll_pwait / eventfd2 variants exist.
    let r = NativeSyscallRegistry::new();

    // (arch, futex, eventfd-or-None, eventfd2, epoll_create-or-None,
    //  epoll_create1, epoll_ctl, epoll_wait-or-None)
    #[allow(clippy::type_complexity)]
    let table: &[(
        &str,
        u64,
        Option<u64>,
        u64,
        Option<u64>,
        u64,
        u64,
        Option<u64>,
    )] = &[
        ("AMD64", 202, Some(284), 290, Some(213), 291, 233, Some(232)),
        ("X86", 240, Some(323), 328, Some(254), 329, 255, Some(256)),
        ("ARM", 240, Some(351), 356, Some(250), 357, 251, Some(252)),
        ("ARM64", 98, None, 19, None, 20, 21, None),
        (
            "MIPS32",
            4238,
            Some(4319),
            4325,
            Some(4248),
            4326,
            4249,
            Some(4250),
        ),
        (
            "MIPS64",
            5194,
            Some(5278),
            5284,
            Some(5207),
            5285,
            5208,
            Some(5209),
        ),
    ];

    for &(arch, futex_n, evfd_n, evfd2_n, ec_n, ec1_n, ectl_n, ewait_n) in table {
        let f = r
            .get(arch, futex_n)
            .unwrap_or_else(|| panic!("{arch} futex ({futex_n}) missing"));
        assert_eq!(f.name(), "futex");
        assert_eq!(f.num_args(), 6);

        let e2 = r
            .get(arch, evfd2_n)
            .unwrap_or_else(|| panic!("{arch} eventfd2 ({evfd2_n}) missing"));
        assert_eq!(e2.name(), "eventfd2");
        assert_eq!(e2.num_args(), 2);

        let ec1 = r
            .get(arch, ec1_n)
            .unwrap_or_else(|| panic!("{arch} epoll_create1 ({ec1_n}) missing"));
        assert_eq!(ec1.name(), "epoll_create1");
        assert_eq!(ec1.num_args(), 1);

        let ectl = r
            .get(arch, ectl_n)
            .unwrap_or_else(|| panic!("{arch} epoll_ctl ({ectl_n}) missing"));
        assert_eq!(ectl.name(), "epoll_ctl");
        assert_eq!(ectl.num_args(), 4);

        if let Some(n) = evfd_n {
            let e = r
                .get(arch, n)
                .unwrap_or_else(|| panic!("{arch} eventfd ({n}) missing"));
            assert_eq!(e.name(), "eventfd");
            assert_eq!(e.num_args(), 1);
        }
        if let Some(n) = ec_n {
            let h = r
                .get(arch, n)
                .unwrap_or_else(|| panic!("{arch} epoll_create ({n}) missing"));
            assert_eq!(h.name(), "epoll_create");
            assert_eq!(h.num_args(), 1);
        }
        if let Some(n) = ewait_n {
            let h = r
                .get(arch, n)
                .unwrap_or_else(|| panic!("{arch} epoll_wait ({n}) missing"));
            assert_eq!(h.name(), "epoll_wait");
            assert_eq!(h.num_args(), 4);
        }
    }
}

#[test]
fn file_path_stubs_registered_on_all_arches() {
    // Originally angr-0hif.1 symbolic-return subset (lstat /
    // newfstatat / readlink / readlinkat); every entry has since
    // been promoted to a real handler:
    //   * faccessat (angr-6009) — NativeAccessSyscall + dirfd.
    //   * lstat / newfstatat (angr-poao) — stat()-shaped clones.
    //   * readlink / readlinkat (angr-wv38) — always -1
    //     (no-symlink FileSystem).
    // Per-handler semantics are pinned by dedicated tests in
    // syscalls/file_path.rs; this test still asserts name + arity.
    //
    // Per-arch availability:
    //   * AArch64 asm-generic ABI dropped legacy `lstat` and `readlink`
    //     (only *at variants exist).
    //   * 32-bit Linux i386 / ARM EABI use the LFS `lstat64` (196) and
    //     `fstatat64` numbers, both wired to the lstat / newfstatat
    //     handlers (angr-11djq.5.1 / .5.2). The two arches disagree on the
    //     `fstatat64` number: i386 is **300** (327 there is signalfd4),
    //     ARM EABI is **327** (angr-sqfj8.107). MIPS32 O32 also uses the LFS
    //     `lstat64` (4214) and `fstatat64` (4293, wired to the newfstatat
    //     handler) numbers (angr-11djq.5.3). The legacy `lstat` numbers
    //     (i386/ARM 107, MIPS32 4107) stay on Python — the Rust writers
    //     emit the `*64` layout only (angr-9ke6b.226).
    //   * MIPS64 N64 has no `struct stat` writer in `write_stat_for_arch`,
    //     so its whole stat family stays on Python (angr-9ke6b.154).
    let r = NativeSyscallRegistry::new();

    // (arch, lstat-or-None, newfstatat-or-None, readlink-or-None,
    //  readlinkat, faccessat)
    type FilePathRow = (
        &'static str,
        Option<u64>,
        Option<u64>,
        Option<u64>,
        u64,
        u64,
    );
    let table: &[FilePathRow] = &[
        ("AMD64", Some(6), Some(262), Some(89), 267, 269),
        // X86 / ARM: LFS lstat64 (196) + fstatat64, no legacy 107. The
        // fstatat64 numbers differ: 300 on i386, 327 on ARM EABI.
        ("X86", Some(196), Some(300), Some(85), 305, 307),
        ("ARM", Some(196), Some(327), Some(85), 332, 334),
        ("ARM64", None, Some(79), None, 78, 48),
        ("MIPS32", Some(4214), Some(4293), Some(4085), 4298, 4300),
        // N64 has both lstat (5006) and newfstatat (5252) in its table,
        // but neither is registered: `write_stat_for_arch` has no MIPS64
        // `struct stat` writer, so both stay on Python (angr-9ke6b.154).
        ("MIPS64", None, None, Some(5087), 5257, 5259),
    ];

    for &(arch, lstat_n, nfstatat_n, readlink_n, readlinkat_n, faccessat_n) in table {
        let rla = r
            .get(arch, readlinkat_n)
            .unwrap_or_else(|| panic!("{arch} readlinkat ({readlinkat_n}) missing"));
        assert_eq!(rla.name(), "readlinkat");
        assert_eq!(rla.num_args(), 4);

        let fa = r
            .get(arch, faccessat_n)
            .unwrap_or_else(|| panic!("{arch} faccessat ({faccessat_n}) missing"));
        assert_eq!(fa.name(), "faccessat");
        assert_eq!(fa.num_args(), 3);

        if let Some(n) = lstat_n {
            let h = r
                .get(arch, n)
                .unwrap_or_else(|| panic!("{arch} lstat ({n}) missing"));
            assert_eq!(h.name(), "lstat");
            assert_eq!(h.num_args(), 2);
        }
        if let Some(n) = nfstatat_n {
            let h = r
                .get(arch, n)
                .unwrap_or_else(|| panic!("{arch} newfstatat ({n}) missing"));
            assert_eq!(h.name(), "newfstatat");
            assert_eq!(h.num_args(), 4);
        }
        if let Some(n) = readlink_n {
            let h = r
                .get(arch, n)
                .unwrap_or_else(|| panic!("{arch} readlink ({n}) missing"));
            assert_eq!(h.name(), "readlink");
            assert_eq!(h.num_args(), 3);
        }
    }

    // angr-9ke6b.226 regression: the pre-LFS stat-family numbers must
    // stay unregistered on the 32-bit arches. ARM used to register 107
    // to the lstat handler, which writes the `struct stat64` layout
    // (`write_arm_stat`) — a wrong-offset buffer instead of a clean
    // fall-through to Python's proc.
    for &(arch, legacy) in &[
        ("X86", [106u64, 107, 108]),
        ("ARM", [106, 107, 108]),
        ("MIPS32", [4106, 4107, 4108]),
    ] {
        for n in legacy {
            assert!(
                r.get(arch, n).is_none(),
                "{arch} legacy stat-family ({n}) must fall back to Python"
            );
        }
    }
}

/// angr-sqfj8.107: `fstatat64` has a *different* number on i386 (300) than on
/// ARM EABI (327), and the i386 table originally used the ARM number. That is
/// not a missed fast path but a wrong answer: 327 on i386 is `signalfd4`, so a
/// guest `signalfd4(fd, mask, sizemask, flags)` was dispatched to the
/// newfstatat handler, which reinterpreted the args as
/// `(dirfd, pathname, statbuf, flag)` and wrote a bogus `struct stat64` into
/// whatever address sat in the third argument.
///
/// The numbers below are angr's own, from
/// `angr/procedures/definitions/linux_kernel.py`'s
/// `syscall_number_mapping["i386"]` / `["arm"]` — that map is the contract the
/// registry must match, since it is what selects the Python `SimProcedure` we
/// are shadowing.
#[test]
fn fstatat64_uses_per_arch_number_and_does_not_shadow_signalfd4() {
    let r = NativeSyscallRegistry::new();

    // Positive: each arch's own fstatat64 number reaches the handler.
    for &(arch, n) in &[("X86", 300u64), ("ARM", 327)] {
        let h = r
            .get(arch, n)
            .unwrap_or_else(|| panic!("{arch} fstatat64 ({n}) missing"));
        assert_eq!(h.name(), "newfstatat");
        assert_eq!(h.num_args(), 4);
    }

    // Negative: the *other* arch's number must not be registered, or a
    // different syscall silently gets the stat writer. i386 327 = signalfd4,
    // ARM EABI 300 = semctl; neither has a native handler, so both must fall
    // through to Python.
    for &(arch, n, actual) in &[("X86", 327u64, "signalfd4"), ("ARM", 300, "semctl")] {
        assert!(
            r.get(arch, n).is_none(),
            "{arch} {n} is {actual}, not fstatat64 — must fall back to Python"
        );
    }
}

#[test]
fn file_descriptor_stubs_registered_on_all_arches() {
    // angr-0hif.5 originally registered fcntl / fcntl64 / ioctl /
    // pipe / pipe2 as pure stubs. angr-aig2 promoted fcntl /
    // fcntl64 / ioctl to concrete-cmd dispatch — they still fall
    // through to a fresh symbolic for unhandled cmds, so the
    // registry shape and `name()` / arity remain unchanged. None
    // of these have a Python `SimProcedure` bound in the kernel
    // library — `posix/fcntl.py` is registered on the libc side
    // only; ioctl/pipe/pipe2 have no proc at all.
    //
    // Per-arch availability:
    //   * AArch64 asm-generic omits legacy `pipe` (only pipe2 at 59)
    //     and `fcntl64` (unified fcntl at 25).
    //   * 32-bit i386 / ARM EABI / MIPS32 O32 carry both `fcntl`
    //     and `fcntl64`; AMD64 has only `fcntl` (no fcntl64).
    //
    // dup / dup2 / dup3 (angr-vp19) ARE registered — they mutate
    // RustSimState::file_system() directly, matching the libc
    // `procedures/fileops::NativeDup` precedent. State.posix.fd
    // intentionally not synced (see file_descriptor.rs doc).
    let r = NativeSyscallRegistry::new();

    // (arch, fcntl, fcntl64-or-None, ioctl, pipe-or-None, pipe2)
    type FcntlRow = (&'static str, u64, Option<u64>, u64, Option<u64>, u64);
    let table: &[FcntlRow] = &[
        ("AMD64", 72, None, 16, Some(22), 293),
        ("X86", 55, Some(221), 54, Some(42), 331),
        ("ARM", 55, Some(221), 54, Some(42), 359),
        ("ARM64", 25, None, 29, None, 59),
        ("MIPS32", 4055, Some(4220), 4054, Some(4042), 4328),
        // N64 is already 64-bit — no separate fcntl64.
        ("MIPS64", 5070, None, 5015, Some(5021), 5287),
    ];

    for &(arch, fcntl_n, fcntl64_n, ioctl_n, pipe_n, pipe2_n) in table {
        let f = r
            .get(arch, fcntl_n)
            .unwrap_or_else(|| panic!("{arch} fcntl ({fcntl_n}) missing"));
        assert_eq!(f.name(), "fcntl");
        assert_eq!(f.num_args(), 3);

        let io = r
            .get(arch, ioctl_n)
            .unwrap_or_else(|| panic!("{arch} ioctl ({ioctl_n}) missing"));
        assert_eq!(io.name(), "ioctl");
        assert_eq!(io.num_args(), 3);

        let p2 = r
            .get(arch, pipe2_n)
            .unwrap_or_else(|| panic!("{arch} pipe2 ({pipe2_n}) missing"));
        assert_eq!(p2.name(), "pipe2");
        assert_eq!(p2.num_args(), 2);

        if let Some(n) = fcntl64_n {
            let h = r
                .get(arch, n)
                .unwrap_or_else(|| panic!("{arch} fcntl64 ({n}) missing"));
            assert_eq!(h.name(), "fcntl64");
            assert_eq!(h.num_args(), 3);
        }
        if let Some(n) = pipe_n {
            let h = r
                .get(arch, n)
                .unwrap_or_else(|| panic!("{arch} pipe ({n}) missing"));
            assert_eq!(h.name(), "pipe");
            assert_eq!(h.num_args(), 1);
        }
    }

    // dup / dup2 / dup3 (angr-vp19): native handlers backed by
    // RustSimState::file_system(). ARM64 asm-generic only has dup
    // (23) and dup3 (24); no legacy dup2.
    let dup_table: &[(&str, u64, Option<u64>, u64)] = &[
        ("AMD64", 32, Some(33), 292),
        ("X86", 41, Some(63), 330),
        ("ARM", 41, Some(63), 358),
        ("ARM64", 23, None, 24),
        ("MIPS32", 4041, Some(4063), 4327),
        ("MIPS64", 5031, Some(5032), 5286),
    ];
    for &(arch, dup_n, dup2_n, dup3_n) in dup_table {
        let d = r
            .get(arch, dup_n)
            .unwrap_or_else(|| panic!("{arch} dup ({dup_n}) missing"));
        assert_eq!(d.name(), "dup");
        assert_eq!(d.num_args(), 1);

        if let Some(n) = dup2_n {
            let d2 = r
                .get(arch, n)
                .unwrap_or_else(|| panic!("{arch} dup2 ({n}) missing"));
            assert_eq!(d2.name(), "dup2");
            assert_eq!(d2.num_args(), 2);
        }

        let d3 = r
            .get(arch, dup3_n)
            .unwrap_or_else(|| panic!("{arch} dup3 ({dup3_n}) missing"));
        assert_eq!(d3.name(), "dup3");
        assert_eq!(d3.num_args(), 3);
    }
}

#[test]
fn directory_syscalls_registered_on_all_arches() {
    // angr-0hif.2: chdir / fchdir / getcwd back the per-state cwd;
    // mkdir/mkdirat/rmdir/unlink/unlinkat/rename/renameat/renameat2
    // mirror syscall_stub (no Python proc). Per-arch availability:
    //   * ARM64 asm-generic drops legacy mkdir/rmdir/unlink/rename
    //     (only *at variants exist) and has no renameat2 in angr.
    //   * ARM EABI / MIPS-O32 have no renameat2 in angr's table.
    let r = NativeSyscallRegistry::new();

    // (arch, getcwd, chdir, fchdir, mkdirat, unlinkat, renameat,
    //  mkdir-or-None, rmdir-or-None, unlink-or-None, rename-or-None,
    //  renameat2-or-None)
    #[allow(clippy::type_complexity)]
    let table: &[(
        &str,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        Option<u64>,
        Option<u64>,
        Option<u64>,
        Option<u64>,
        Option<u64>,
    )] = &[
        (
            "AMD64",
            79,
            80,
            81,
            258,
            263,
            264,
            Some(83),
            Some(84),
            Some(87),
            Some(82),
            Some(316),
        ),
        (
            "X86",
            183,
            12,
            133,
            296,
            301,
            302,
            Some(39),
            Some(40),
            Some(10),
            Some(38),
            Some(353),
        ),
        (
            "ARM",
            183,
            12,
            133,
            323,
            328,
            329,
            Some(39),
            Some(40),
            Some(10),
            Some(38),
            None,
        ),
        (
            "ARM64", 17, 49, 50, 34, 35, 38, None, None, None, None, None,
        ),
        (
            "MIPS32",
            4203,
            4012,
            4133,
            4289,
            4294,
            4295,
            Some(4039),
            Some(4040),
            Some(4010),
            Some(4038),
            None,
        ),
        (
            "MIPS64",
            5077,
            5078,
            5079,
            5248,
            5253,
            5254,
            Some(5081),
            Some(5082),
            Some(5085),
            Some(5080),
            None,
        ),
    ];

    for &(arch, gc_n, cd_n, fcd_n, mka_n, ula_n, rea_n, mk_n, rm_n, ul_n, rn_n, rea2_n) in table {
        let g = r
            .get(arch, gc_n)
            .unwrap_or_else(|| panic!("{arch} getcwd ({gc_n}) missing"));
        assert_eq!(g.name(), "getcwd");
        assert_eq!(g.num_args(), 2);

        let c = r
            .get(arch, cd_n)
            .unwrap_or_else(|| panic!("{arch} chdir ({cd_n}) missing"));
        assert_eq!(c.name(), "chdir");
        assert_eq!(c.num_args(), 1);

        let fc = r
            .get(arch, fcd_n)
            .unwrap_or_else(|| panic!("{arch} fchdir ({fcd_n}) missing"));
        assert_eq!(fc.name(), "fchdir");
        assert_eq!(fc.num_args(), 1);

        let mka = r
            .get(arch, mka_n)
            .unwrap_or_else(|| panic!("{arch} mkdirat ({mka_n}) missing"));
        assert_eq!(mka.name(), "mkdirat");
        assert_eq!(mka.num_args(), 3);

        let ula = r
            .get(arch, ula_n)
            .unwrap_or_else(|| panic!("{arch} unlinkat ({ula_n}) missing"));
        assert_eq!(ula.name(), "unlinkat");
        assert_eq!(ula.num_args(), 3);

        let rea = r
            .get(arch, rea_n)
            .unwrap_or_else(|| panic!("{arch} renameat ({rea_n}) missing"));
        assert_eq!(rea.name(), "renameat");
        assert_eq!(rea.num_args(), 4);

        if let Some(n) = mk_n {
            let h = r
                .get(arch, n)
                .unwrap_or_else(|| panic!("{arch} mkdir ({n}) missing"));
            assert_eq!(h.name(), "mkdir");
            assert_eq!(h.num_args(), 2);
        }
        if let Some(n) = rm_n {
            let h = r
                .get(arch, n)
                .unwrap_or_else(|| panic!("{arch} rmdir ({n}) missing"));
            assert_eq!(h.name(), "rmdir");
            assert_eq!(h.num_args(), 1);
        }
        if let Some(n) = ul_n {
            let h = r
                .get(arch, n)
                .unwrap_or_else(|| panic!("{arch} unlink ({n}) missing"));
            assert_eq!(h.name(), "unlink");
            assert_eq!(h.num_args(), 1);
        }
        if let Some(n) = rn_n {
            let h = r
                .get(arch, n)
                .unwrap_or_else(|| panic!("{arch} rename ({n}) missing"));
            assert_eq!(h.name(), "rename");
            assert_eq!(h.num_args(), 2);
        }
        if let Some(n) = rea2_n {
            let h = r
                .get(arch, n)
                .unwrap_or_else(|| panic!("{arch} renameat2 ({n}) missing"));
            assert_eq!(h.name(), "renameat2");
            assert_eq!(h.num_args(), 5);
        }
    }
}

#[test]
fn all_arches_have_exit_handler() {
    // Sanity smoke: every supported arch can deadend on exit. Catches
    // accidental dropping of the exit registration during a refactor.
    let r = NativeSyscallRegistry::new();
    // arch_name, exit number
    for (arch, num) in [
        ("AMD64", 60),
        ("X86", 1),
        ("ARM", 1),
        ("ARM64", 93),
        ("MIPS32", 4001),
        ("MIPS64", 5058),
    ] {
        let h = r
            .get(arch, num)
            .unwrap_or_else(|| panic!("{arch} exit ({num}) handler missing"));
        assert_eq!(h.name(), "exit", "{arch} syscall {num} should be exit");
        assert_eq!(h.num_args(), 0, "{arch} exit takes 0 args");
    }
}

#[test]
fn fresh_byte_names_uniquifies_per_batch_and_index() {
    // angr-9ke6b.158: the naming scheme all five mint-N-symbolic-bytes call
    // sites now share. Two properties are load-bearing (see `fresh_symbolic`
    // for why): the counter bumps once per *batch* so a batch's names are
    // `<prefix>_<id>_0..n`, and two batches from the same callsite never
    // collide (which would alias them to the same Z3 constant).
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    let first = fresh_byte_names("sys_probe", &COUNTER, 3);
    let second = fresh_byte_names("sys_probe", &COUNTER, 3);

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
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    assert!(fresh_byte_names("sys_probe", &COUNTER, 0).is_empty());
    assert_eq!(
        fresh_byte_names("sys_probe", &COUNTER, 1),
        vec!["sys_probe_1_0"]
    );
}
