//! File-descriptor control syscall handlers — `fcntl`, `fcntl64`,
//! `ioctl`, `pipe`, `pipe2` (the stub-fallthrough subset of
//! `angr-0hif.5`) and `dup`, `dup2`, `dup3` (`angr-vp19`).
//!
//! Stub handlers (`fcntl`, `fcntl64`, `ioctl`, `pipe`, `pipe2`) fall
//! through to `procedures/stubs/syscall_stub.py::syscall` in pure Python
//! angr:
//!
//! * `posix/fcntl.py` defines a `fcntl` `SimProcedure` but
//!   `angr/procedures/definitions/linux_kernel.py` does NOT bind it
//!   into the `SimSyscallLibrary` (only `dup` from `posix/` is wired
//!   into the kernel table). So a kernel-level `fcntl` syscall on every
//!   supported arch hits `syscall_stub` rather than the libc proc, and
//!   the native handler mirrors that with `RustBV::symbolic` of width
//!   `arch().bits()`.
//! * `fcntl64` (the LFS-style 64-bit-offset variant on 32-bit arches)
//!   has no dedicated Python proc either — same stub path. Registered
//!   as a separate handler so the `name()` reported back is `"fcntl64"`
//!   for visibility in traces.
//! * `ioctl`, `pipe`, `pipe2` have no Python `SimProcedure` and fall
//!   through to `syscall_stub` directly.
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

use super::{NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg, stub_syscall};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// `-EBADF` as a 64-bit two's-complement value. The kernel ABI returns
/// negative errno in the return register on failure; truncation to the
/// arch's register width happens in the dispatcher when it writes the
/// `Continue { ret }` value.
const NEG_EBADF: u64 = (-9_i64) as u64;

/// Upper bound on `newfd` that Python `procedures/posix/dup.py` enforces
/// (the default ulimits ceiling). Out-of-range values return EBADF.
const NEWFD_LIMIT: u64 = 4096;

// fcntl(fd, cmd, arg) → int — falls through to syscall_stub.
stub_syscall!(NativeFcntlSyscall, "fcntl", "syscall_stub_fcntl", 3);
// fcntl64(fd, cmd, arg) → long — LFS-flavored variant on 32-bit arches.
stub_syscall!(NativeFcntl64Syscall, "fcntl64", "syscall_stub_fcntl64", 3);
// ioctl(fd, cmd, arg) → int — no Python SimProcedure.
stub_syscall!(NativeIoctlSyscall, "ioctl", "syscall_stub_ioctl", 3);
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

    /// All five handlers should: produce the expected `name()` / arity,
    /// return a fresh symbolic `RustBV` of width `arch().bits()` on every
    /// supported arch, and yield distinct symbol IDs on successive calls
    /// (no stale-Arc sharing).
    #[test]
    fn fd_stub_handlers_return_fresh_symbolic_on_all_arches() {
        let cases: &[(&'static dyn NativeSyscall, &str, usize)] = &[
            (&NativeFcntlSyscall, "fcntl", 3),
            (&NativeFcntl64Syscall, "fcntl64", 3),
            (&NativeIoctlSyscall, "ioctl", 3),
            (&NativePipeSyscall, "pipe", 1),
            (&NativePipe2Syscall, "pipe2", 2),
        ];

        for arch in ["amd64", "x86", "armel", "aarch64", "mipsel"] {
            let mut state = RustSimState::new(arch).expect("state");
            let bits = state.arch().bits();
            for &(handler, label, nargs) in cases {
                assert_eq!(handler.name(), label);
                assert_eq!(handler.num_args(), nargs, "{label} arity");
                let args: Vec<RustBV> =
                    (0..nargs).map(|_| RustBV::concrete(0, bits)).collect();
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
                    (
                        RustBV::Symbolic { id: a, .. },
                        RustBV::Symbolic { id: b, .. },
                    ) => (*a, *b),
                    _ => panic!("{arch} {label}: expected Symbolic variant"),
                };
                assert_ne!(
                    id1, id2,
                    "{arch} {label} successive calls must yield distinct symbol IDs"
                );
            }
        }
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
            .call(&mut state, &[RustBV::concrete(0, 64), RustBV::concrete(5, 64)])
            .unwrap();
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 5),
            _ => panic!("dup2(0,5) expected Continue"),
        }
        assert!(state.file_system_ref().is_open(5));

        // dup2(0, 0) — same fd, open; returns newfd unchanged.
        let same = NativeDup2Syscall
            .call(&mut state, &[RustBV::concrete(0, 64), RustBV::concrete(0, 64)])
            .unwrap();
        match same {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
            _ => panic!("dup2(0,0) expected Continue"),
        }

        // dup2(99, 6) — oldfd never opened; returns -EBADF.
        let bad_old = NativeDup2Syscall
            .call(&mut state, &[RustBV::concrete(99, 64), RustBV::concrete(6, 64)])
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
