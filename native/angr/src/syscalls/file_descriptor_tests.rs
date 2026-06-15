//! Tests for file-descriptor syscall handlers (pipe/dup/dup2/dup3 etc., extracted from file_descriptor.rs).

use super::*;
use crate::state::RustSimState;
use crate::symbolic::RustBV;
use crate::syscalls::{NativeSyscall, SyscallOutcome};

/// `pipe` / `pipe2` are still pure stubs (no concrete-arg
/// dispatch), so they must return a fresh symbolic `RustBV` of
/// width `arch().bits()` on every supported arch with distinct
/// symbol IDs across successive calls.
#[test]
fn pipe_stub_handlers_return_fresh_symbolic_on_all_arches() {
    let cases: &[(&'static dyn NativeSyscall, &str, usize)] = &[
        (&NativePipeSyscall, "pipe", 1),
        (&NativePipe2Syscall, "pipe2", 2),
    ];

    for arch in ["amd64", "x86", "armel", "aarch64", "mipsel"] {
        let mut state = RustSimState::new(arch).expect("state");
        let bits = state.arch().bits();
        for &(handler, label, nargs) in cases {
            assert_eq!(handler.name(), label);
            assert_eq!(handler.num_args(), nargs, "{label} arity");
            let args: Vec<RustBV> = (0..nargs).map(|_| RustBV::concrete(0, bits)).collect();
            let outcome = handler
                .call(&mut state, &args)
                .unwrap_or_else(|e| panic!("{arch} {label}: {e:?}"));
            let ret = match outcome {
                SyscallOutcome::ContinueSymbolic { ret } => ret,
                other => panic!("{arch} {label}: expected ContinueSymbolic, got {other:?}"),
            };
            assert_eq!(ret.width(), bits);
            assert!(ret.as_u64().is_none(), "{arch} {label} must be symbolic");

            let outcome2 = handler.call(&mut state, &args).unwrap();
            let ret2 = match outcome2 {
                SyscallOutcome::ContinueSymbolic { ret } => ret,
                _ => unreachable!(),
            };
            let (id1, id2) = match (&ret, &ret2) {
                (RustBV::Symbolic { id: a, .. }, RustBV::Symbolic { id: b, .. }) => (*a, *b),
                _ => panic!("{arch} {label}: expected Symbolic variant"),
            };
            assert_ne!(
                id1, id2,
                "{arch} {label} successive calls must yield distinct symbol IDs"
            );
        }
    }
}

/// `fcntl` / `fcntl64` handle the four trivial getter / setter
/// `cmd`s natively and fall through to a fresh symbolic for
/// everything else. Exercises both dispatches on every supported
/// arch so per-arch register-width handling stays exercised.
#[test]
fn fcntl_dispatch_concrete_cmds_across_arches() {
    for arch in ["amd64", "x86", "armel", "aarch64", "mipsel"] {
        let mut state = RustSimState::new(arch).expect("state");
        let bits = state.arch().bits();

        let bv = |v: u64| RustBV::concrete(u128::from(v), bits);
        // F_GETFL on fd=0 (stdin = O_RDONLY = 0).
        for handler in [
            &NativeFcntlSyscall as &dyn NativeSyscall,
            &NativeFcntl64Syscall as &dyn NativeSyscall,
        ] {
            let label = handler.name();

            let getfl = handler
                .call(&mut state, &[bv(0), bv(F_GETFL), bv(0)])
                .unwrap();
            match getfl {
                SyscallOutcome::Continue { ret } => {
                    assert_eq!(ret, 0, "{arch} {label} F_GETFL(stdin) expected O_RDONLY");
                }
                other => panic!("{arch} {label} F_GETFL expected Continue, got {other:?}"),
            }

            // F_GETFL on fd=1 (stdout = O_WRONLY = 1).
            let getfl_w = handler
                .call(&mut state, &[bv(1), bv(F_GETFL), bv(0)])
                .unwrap();
            match getfl_w {
                SyscallOutcome::Continue { ret } => {
                    assert_eq!(ret, 1, "{arch} {label} F_GETFL(stdout) expected O_WRONLY");
                }
                other => panic!("{arch} {label} F_GETFL(1) expected Continue, got {other:?}"),
            }

            // F_GETFL on closed fd → -EBADF.
            let getfl_bad = handler
                .call(&mut state, &[bv(99), bv(F_GETFL), bv(0)])
                .unwrap();
            match getfl_bad {
                SyscallOutcome::Continue { ret } => assert_eq!(ret, NEG_EBADF),
                other => panic!("{arch} {label} F_GETFL(99) expected Continue, got {other:?}"),
            }

            // F_GETFD / F_SETFD / F_SETFL all return 0 regardless of fd.
            for cmd in [F_GETFD, F_SETFD, F_SETFL] {
                let outcome = handler.call(&mut state, &[bv(0), bv(cmd), bv(0)]).unwrap();
                match outcome {
                    SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
                    other => {
                        panic!("{arch} {label} cmd={cmd} expected Continue, got {other:?}")
                    }
                }
            }

            // F_DUPFD (cmd=0) is unhandled → falls through to fresh symbolic.
            let unhandled = handler.call(&mut state, &[bv(0), bv(0), bv(0)]).unwrap();
            match unhandled {
                SyscallOutcome::ContinueSymbolic { ret } => {
                    assert_eq!(ret.width(), bits);
                    assert!(ret.as_u64().is_none());
                }
                other => {
                    panic!("{arch} {label} cmd=F_DUPFD expected ContinueSymbolic, got {other:?}")
                }
            }
        }
    }
}

/// Symbolic `cmd` aborts the native dispatch and surfaces a
/// `SyscallError::SymbolicArgument` so the caller falls back to
/// the Python `syscall_stub` path — matches every other
/// concrete-arg native syscall.
#[test]
fn fcntl_symbolic_cmd_falls_back_to_python() {
    let mut state = RustSimState::new("amd64").expect("state");
    let sym_cmd = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "cmd_sym", 64)
    };
    let err = NativeFcntlSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0, 64), sym_cmd, RustBV::concrete(0, 64)],
        )
        .unwrap_err();
    assert!(matches!(err, SyscallError::SymbolicArgument(_)));
}

/// `ioctl(_, TIOCGWINSZ, _)` returns `-ENOTTY` on the arches we
/// model (no terminal is attached to fd 0/1/2). MIPS uses a
/// different constant; verify both branches.
#[test]
fn ioctl_tiocgwinsz_returns_enotty() {
    for (arch, cmd) in [
        ("amd64", TIOCGWINSZ_GENERIC),
        ("x86", TIOCGWINSZ_GENERIC),
        ("armel", TIOCGWINSZ_GENERIC),
        ("aarch64", TIOCGWINSZ_GENERIC),
        ("mipsel", TIOCGWINSZ_MIPS),
    ] {
        let mut state = RustSimState::new(arch).expect("state");
        let bits = state.arch().bits();
        let outcome = NativeIoctlSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(0, bits),
                    RustBV::concrete(u128::from(cmd), bits),
                    RustBV::concrete(0, bits),
                ],
            )
            .unwrap();
        match outcome {
            SyscallOutcome::Continue { ret } => {
                assert_eq!(ret, NEG_ENOTTY, "{arch} TIOCGWINSZ should ENOTTY");
            }
            other => panic!("{arch} TIOCGWINSZ expected Continue, got {other:?}"),
        }
    }
}

/// `ioctl` with a `cmd` we do not handle (e.g. `FIONREAD`) falls
/// through to a fresh symbolic return on every arch, mirroring
/// the prior stub behavior. Symbolic `cmd` triggers the
/// SymbolicArgument fallback path.
#[test]
fn ioctl_unknown_cmd_falls_back_to_symbolic() {
    let mut state = RustSimState::new("amd64").expect("state");
    // FIONREAD on x86/amd64 = 0x541B — not handled here, falls back.
    let outcome = NativeIoctlSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(0, 64),
                RustBV::concrete(0x541B, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();
    match outcome {
        SyscallOutcome::ContinueSymbolic { ret } => {
            assert_eq!(ret.width(), 64);
            assert!(ret.as_u64().is_none());
        }
        other => panic!("FIONREAD expected ContinueSymbolic, got {other:?}"),
    }

    let sym_cmd = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "ioctl_cmd_sym", 64)
    };
    let err = NativeIoctlSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0, 64), sym_cmd, RustBV::concrete(0, 64)],
        )
        .unwrap_err();
    assert!(matches!(err, SyscallError::SymbolicArgument(_)));
}

/// `dup(oldfd)` on a fresh state allocates a fresh fd aliasing the
/// requested one. Repeat the smoke across every supported arch so
/// per-arch register-width truncation in the syscall dispatcher is
/// exercised by callers; the handler itself returns `u64` (the
/// dispatcher writes the low `arch().bits()` bits to rax/r0/...).
#[test]
fn dup_handlers_allocate_and_alias_across_arches() {
    for arch in ["amd64", "x86", "armel", "aarch64", "mipsel"] {
        let mut state = RustSimState::new(arch).expect("state");
        let bits = state.arch().bits();

        // Seed: dup of stdin (fd=0, pre-registered open).
        let outcome = NativeDupSyscall
            .call(&mut state, &[RustBV::concrete(0, bits)])
            .unwrap();
        let newfd = match outcome {
            SyscallOutcome::Continue { ret } => ret,
            other => panic!("{arch} dup: expected Continue, got {other:?}"),
        };
        assert_eq!(newfd, 3, "{arch} dup(0) should allocate next_fd=3");
        assert!(state.file_system_ref().is_open(newfd as u32));

        // dup of a never-opened fd returns -EBADF.
        let bad = NativeDupSyscall
            .call(&mut state, &[RustBV::concrete(99, bits)])
            .unwrap();
        match bad {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, NEG_EBADF),
            _ => panic!("{arch} dup(99): expected Continue"),
        }
    }
}

/// `dup2(oldfd, newfd)` makes `newfd` an alias of `oldfd`. Verify the
/// behavior on a fresh state, on collision (closes target first), and
/// the EBADF return for out-of-range / closed-source paths.
#[test]
fn dup2_handler_covers_alias_collision_and_ebadf() {
    let mut state = RustSimState::new("amd64").expect("state");

    // dup2(0, 5) — newfd=5 was not open; should return 5.
    let outcome = NativeDup2Syscall
        .call(
            &mut state,
            &[RustBV::concrete(0, 64), RustBV::concrete(5, 64)],
        )
        .unwrap();
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 5),
        _ => panic!("dup2(0,5) expected Continue"),
    }
    assert!(state.file_system_ref().is_open(5));

    // dup2(0, 0) — same fd, open; returns newfd unchanged.
    let same = NativeDup2Syscall
        .call(
            &mut state,
            &[RustBV::concrete(0, 64), RustBV::concrete(0, 64)],
        )
        .unwrap();
    match same {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
        _ => panic!("dup2(0,0) expected Continue"),
    }

    // dup2(99, 6) — oldfd never opened; returns -EBADF.
    let bad_old = NativeDup2Syscall
        .call(
            &mut state,
            &[RustBV::concrete(99, 64), RustBV::concrete(6, 64)],
        )
        .unwrap();
    match bad_old {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, NEG_EBADF),
        _ => panic!("dup2(99,6) expected Continue"),
    }

    // dup2(0, 4096) — out-of-range newfd; returns -EBADF.
    let bad_new = NativeDup2Syscall
        .call(
            &mut state,
            &[RustBV::concrete(0, 64), RustBV::concrete(4096, 64)],
        )
        .unwrap();
    match bad_new {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, NEG_EBADF),
        _ => panic!("dup2(0,4096) expected Continue"),
    }
}

/// `dup3(oldfd, newfd, flags)` — same semantics as dup2 except the
/// extra `flags` argument is accepted (O_CLOEXEC modeling is out of
/// scope, so flags are ignored). Verify dispatch + flags-tolerance.
#[test]
fn dup3_handler_ignores_flags_and_matches_dup2() {
    let mut state = RustSimState::new("amd64").expect("state");

    // dup3(0, 7, 0) → newfd=7
    let outcome = NativeDup3Syscall
        .call(
            &mut state,
            &[
                RustBV::concrete(0, 64),
                RustBV::concrete(7, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 7),
        _ => panic!("dup3(0,7,0) expected Continue"),
    }
    assert!(state.file_system_ref().is_open(7));

    // dup3(0, 8, O_CLOEXEC=0x80000) → newfd=8 (flag accepted+ignored)
    let cloexec = NativeDup3Syscall
        .call(
            &mut state,
            &[
                RustBV::concrete(0, 64),
                RustBV::concrete(8, 64),
                RustBV::concrete(0x80000, 64),
            ],
        )
        .unwrap();
    match cloexec {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 8),
        _ => panic!("dup3(0,8,O_CLOEXEC) expected Continue"),
    }

    // dup3(99, 9, 0) — oldfd never opened; -EBADF.
    let bad_old = NativeDup3Syscall
        .call(
            &mut state,
            &[
                RustBV::concrete(99, 64),
                RustBV::concrete(9, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();
    match bad_old {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, NEG_EBADF),
        _ => panic!("dup3(99,9,0) expected Continue"),
    }

    // dup3(0, 5000, 0) — out-of-range newfd; -EBADF.
    let bad_new = NativeDup3Syscall
        .call(
            &mut state,
            &[
                RustBV::concrete(0, 64),
                RustBV::concrete(5000, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();
    match bad_new {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, NEG_EBADF),
        _ => panic!("dup3(0,5000,0) expected Continue"),
    }
}
