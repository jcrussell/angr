//! File-descriptor control syscall handlers — `fcntl`, `fcntl64`,
//! `ioctl`, `pipe`, `pipe2` (the stub-fallthrough subset of
//! `angr-0hif.5`) and `dup`, `dup2`, `dup3` (`angr-vp19`).
//!
//! ## fcntl / fcntl64 (angr-aig2)
//!
//! Dispatched on the concrete `cmd` arg. The four "trivial getter /
//! setter" ops are handled natively without touching Python:
//!
//! * `F_GETFD` (1): close-on-exec flags. We do not model FD_CLOEXEC,
//!   so the answer is always `0`.
//! * `F_SETFD` (2): set close-on-exec flags. Ignored; returns `0`
//!   (success).
//! * `F_GETFL` (3): file status flags. Reads the `FdFlags` (RO/WO/RW)
//!   from the Rust `FileSystem` and returns the matching `O_RDONLY` /
//!   `O_WRONLY` / `O_RDWR` value. `-EBADF` when the fd is not open.
//! * `F_SETFL` (4): set file status flags. Ignored; returns `0`
//!   (success). The new flags are not stored — `FdFlags` only models
//!   the access mode, not the per-syscall blocking / append bits.
//!
//! Unrecognized `cmd` values (including `F_DUPFD`, locks, leases,
//! pipe-size getters / setters, ...) fall through to a fresh symbolic
//! `RustBV` of width `arch().bits()`, matching the prior
//! `syscall_stub_fcntl` behavior. A symbolic `cmd` also falls back to
//! the stub via `SyscallError::SymbolicArgument`.
//!
//! `posix/fcntl.py` defines a `fcntl` `SimProcedure` but
//! `angr/procedures/definitions/linux_kernel.py` does NOT bind it
//! into the `SimSyscallLibrary` (only `dup` from `posix/` is wired
//! into the kernel table). So pure-Python angr also hits
//! `syscall_stub` on every `fcntl` — the Rust handler is strictly
//! more informative than baseline for the four handled cmds and
//! identical to baseline for everything else.
//!
//! ## ioctl (angr-aig2)
//!
//! Dispatched on the concrete `cmd` arg. We handle one terminal-
//! ioctl that real binaries call early in `main()` to size a tty
//! and degrade gracefully on `ENOTTY`:
//!
//! * `TIOCGWINSZ`: get window size. Our model has no terminal
//!   attached to fd 0/1/2 (they back onto byte streams via
//!   `FileSystem`), so we return `-ENOTTY` (`-25`) verbatim. The
//!   user-supplied `struct winsize *` buffer is left untouched —
//!   POSIX allows this on error.
//!
//! `cmd` differs per-arch: `0x5413` on x86 / amd64 / arm / aarch64,
//! `0x40087468` on the MIPS family. The dispatcher reads the right
//! constant from `state.arch().name()` so a single handler covers
//! every registered arch.
//!
//! `FIONREAD` is deliberately NOT handled — its third argument is an
//! `int *` that we would have to write a byte count back into, and
//! the existing native procedures handle that pattern but the
//! syscall-side memory-write helpers are not yet plumbed through
//! here. Falls back to the symbolic stub.
//!
//! Unrecognized `cmd` (including symbolic) falls through to
//! `RustBV::symbolic` of width `arch().bits()`.
//!
//! ## pipe / pipe2
//!
//! No Python `SimProcedure`; both fall through to `syscall_stub`
//! via a fresh symbolic return.
//!
//! ## dup / dup2 / dup3 (angr-vp19)
//!
//! `dup`, `dup2`, `dup3` mutate the per-state FD table. We model that
//! directly on `RustSimState::file_system()` rather than syncing back
//! into Python's `state.posix.fd` — same precedent as
//! `procedures/fileops::NativeDup` / `NativeDup2`, which already
//! manipulate the Rust-side `FileSystem` without touching the Python
//! plugin. Python `procedures/posix/dup.py` allocates a "lowest-free"
//! fd; the Rust `FileSystem::dup` uses a monotonic `next_fd` instead, a
//! divergence that only surfaces when an exploration closes an fd and
//! later expects the freed slot to be reused. No current bench / test
//! relies on slot reuse, so we keep the Rust monotonic allocator.
//!
//! Error returns use the kernel ABI: `-EBADF` (negative errno) on
//! invalid `oldfd` and on out-of-range `newfd` for `dup2`/`dup3`
//! (matches Python's 4096-fd ulimits ceiling).

use super::{
    NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg, fresh_symbolic, stub_syscall,
};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// `-EBADF` as a 64-bit two's-complement value. The kernel ABI returns
/// negative errno in the return register on failure; truncation to the
/// arch's register width happens in the dispatcher when it writes the
/// `Continue { ret }` value.
const NEG_EBADF: u64 = (-9_i64) as u64;
/// `-ENOTTY` (Linux errno 25) for the `TIOCGWINSZ` ioctl on a
/// non-terminal fd.
const NEG_ENOTTY: u64 = (-25_i64) as u64;

/// Upper bound on `newfd` that Python `procedures/posix/dup.py` enforces
/// (the default ulimits ceiling). Out-of-range values return EBADF.
const NEWFD_LIMIT: u64 = 4096;

// fcntl / ioctl `cmd` constants — uniform across Linux ABIs we support.
const F_GETFD: u64 = 1;
const F_SETFD: u64 = 2;
const F_GETFL: u64 = 3;
const F_SETFL: u64 = 4;

/// `TIOCGWINSZ` ioctl number on x86 / amd64 / arm / aarch64 (the
/// `asm-generic/ioctls.h` family).
const TIOCGWINSZ_GENERIC: u64 = 0x5413;
/// `TIOCGWINSZ` ioctl number on MIPS (`asm-mips/ioctls.h` flavor —
/// embeds the struct size into the upper bits).
const TIOCGWINSZ_MIPS: u64 = 0x40087468;

/// Per-arch `TIOCGWINSZ` lookup. Returns `None` for arches we have not
/// audited so the dispatcher falls through to the symbolic stub rather
/// than acting on a guessed constant. Arch names follow the
/// `Arch::name()` strings ("AMD64", "X86", "ARM", "ARM64", "MIPS32",
/// "MIPS64").
fn tiocgwinsz_for_arch(arch_name: &str) -> Option<u64> {
    match arch_name {
        "AMD64" | "X86" | "ARM" | "ARM64" => Some(TIOCGWINSZ_GENERIC),
        "MIPS32" | "MIPS64" => Some(TIOCGWINSZ_MIPS),
        _ => None,
    }
}

/// Build the fresh-symbolic fallback return used when fcntl / ioctl /
/// pipe* hit a `cmd` we do not handle natively. Borrows the solver
/// context once, mirroring the `stub_syscall!` macro expansion.
fn symbolic_return(state: &mut RustSimState, name: &'static str) -> SyscallOutcome {
    let bits = state.arch().bits();
    let ret = {
        let ctx = state.solver().borrow();
        fresh_symbolic(&ctx, name, bits)
    };
    SyscallOutcome::ContinueSymbolic { ret }
}

/// Shared dispatch for `fcntl` / `fcntl64`. Both syscalls take
/// `(fd, cmd, arg)` and differ only in the kernel-side handling of
/// the LFS `arg` payload (locks); the four trivial getter / setter
/// cmds we handle do not read `arg`, so the dispatch is identical.
fn fcntl_dispatch(
    state: &mut RustSimState,
    args: &[RustBV],
    stub_name: &'static str,
) -> Result<SyscallOutcome, SyscallError> {
    let fd = extract_concrete_arg(&args[0], "fcntl fd")?;
    let cmd = extract_concrete_arg(&args[1], "fcntl cmd")?;
    // args[2] = arg — unused by the handled cmds. For F_SETFL it
    // would be the new flags; we accept any value (including
    // symbolic) without forcing concretization.
    let _ = args.get(2);

    match cmd {
        F_GETFD | F_SETFD => Ok(SyscallOutcome::Continue { ret: 0 }),
        F_GETFL => {
            let ret = match state.file_system_ref().fd_info(fd as u32) {
                Some((_, _, flags, _, is_open)) if is_open => flags as u64,
                _ => NEG_EBADF,
            };
            Ok(SyscallOutcome::Continue { ret })
        }
        F_SETFL => Ok(SyscallOutcome::Continue { ret: 0 }),
        _ => Ok(symbolic_return(state, stub_name)),
    }
}

/// `fcntl(fd, cmd, arg) → int` — see module doc for handled `cmd`s.
pub struct NativeFcntlSyscall;

impl NativeSyscall for NativeFcntlSyscall {
    fn name(&self) -> &'static str {
        "fcntl"
    }

    fn num_args(&self) -> usize {
        3
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        fcntl_dispatch(state, args, "syscall_stub_fcntl")
    }
}

/// `fcntl64(fd, cmd, arg) → long` — LFS-flavored variant on 32-bit
/// arches; shares the same dispatch as `fcntl` for the trivial cmds.
pub struct NativeFcntl64Syscall;

impl NativeSyscall for NativeFcntl64Syscall {
    fn name(&self) -> &'static str {
        "fcntl64"
    }

    fn num_args(&self) -> usize {
        3
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        fcntl_dispatch(state, args, "syscall_stub_fcntl64")
    }
}

/// `ioctl(fd, cmd, arg) → int` — see module doc. Currently handles
/// `TIOCGWINSZ` → `-ENOTTY` (no terminal model); everything else
/// falls back to a fresh symbolic return.
pub struct NativeIoctlSyscall;

impl NativeSyscall for NativeIoctlSyscall {
    fn name(&self) -> &'static str {
        "ioctl"
    }

    fn num_args(&self) -> usize {
        3
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        let _fd = extract_concrete_arg(&args[0], "ioctl fd")?;
        let cmd = extract_concrete_arg(&args[1], "ioctl cmd")?;
        let _ = args.get(2);

        let arch_name = state.arch().name();
        if Some(cmd) == tiocgwinsz_for_arch(arch_name) {
            return Ok(SyscallOutcome::Continue { ret: NEG_ENOTTY });
        }
        Ok(symbolic_return(state, "syscall_stub_ioctl"))
    }
}

// pipe(fildes*) → int — no Python SimProcedure.
stub_syscall!(NativePipeSyscall, "pipe", "syscall_stub_pipe", 1);
// pipe2(fildes*, flags) → int — no Python SimProcedure.
stub_syscall!(NativePipe2Syscall, "pipe2", "syscall_stub_pipe2", 2);

/// `dup(oldfd) → newfd` — allocate a fresh fd that aliases `oldfd`.
/// Returns `-EBADF` if `oldfd` is not open.
pub struct NativeDupSyscall;

impl NativeSyscall for NativeDupSyscall {
    fn name(&self) -> &'static str {
        "dup"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        let oldfd = extract_concrete_arg(&args[0], "oldfd")?;
        let ret = match state.file_system().dup(oldfd as u32) {
            Some(newfd) => newfd as u64,
            None => NEG_EBADF,
        };
        Ok(SyscallOutcome::Continue { ret })
    }
}

/// `dup2(oldfd, newfd) → newfd` — make `newfd` an alias of `oldfd`,
/// closing the previous `newfd` if it was open. Returns `-EBADF` if
/// `oldfd` is closed or `newfd` is out of range.
pub struct NativeDup2Syscall;

impl NativeSyscall for NativeDup2Syscall {
    fn name(&self) -> &'static str {
        "dup2"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        let oldfd = extract_concrete_arg(&args[0], "oldfd")?;
        let newfd = extract_concrete_arg(&args[1], "newfd")?;
        if newfd >= NEWFD_LIMIT {
            return Ok(SyscallOutcome::Continue { ret: NEG_EBADF });
        }
        let ret = match state.file_system().dup2(oldfd as u32, newfd as u32) {
            Some(fd) => fd as u64,
            None => NEG_EBADF,
        };
        Ok(SyscallOutcome::Continue { ret })
    }
}

/// `dup3(oldfd, newfd, flags) → newfd` — like `dup2`, with a `flags`
/// argument (only `O_CLOEXEC = 0x80000` defined). We do not model
/// close-on-exec, so the flag is accepted and ignored.
pub struct NativeDup3Syscall;

impl NativeSyscall for NativeDup3Syscall {
    fn name(&self) -> &'static str {
        "dup3"
    }

    fn num_args(&self) -> usize {
        3
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        let oldfd = extract_concrete_arg(&args[0], "oldfd")?;
        let newfd = extract_concrete_arg(&args[1], "newfd")?;
        // args[2] = flags — O_CLOEXEC modeling is out of scope; ignore.
        let _flags = extract_concrete_arg(&args[2], "flags")?;
        if newfd >= NEWFD_LIMIT {
            return Ok(SyscallOutcome::Continue { ret: NEG_EBADF });
        }
        let ret = match state.file_system().dup2(oldfd as u32, newfd as u32) {
            Some(fd) => fd as u64,
            None => NEG_EBADF,
        };
        Ok(SyscallOutcome::Continue { ret })
    }
}

#[cfg(test)]
mod tests {
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
                    other => panic!(
                        "{arch} {label} cmd=F_DUPFD expected ContinueSymbolic, got {other:?}"
                    ),
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
}
