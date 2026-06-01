//! File-descriptor control syscall handlers — `fcntl`, `fcntl64`,
//! `ioctl`, `pipe`, `pipe2` (the stub-fallthrough subset of
//! `angr-0hif.5`).
//!
//! All of these syscalls fall through to
//! `procedures/stubs/syscall_stub.py::syscall` in pure Python angr:
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
//! ## What's intentionally NOT here
//!
//! `dup`, `dup2`, `dup3` have dedicated Python procs in
//! `procedures/posix/dup.py` that mutate `state.posix.fd`. Native
//! parity needs the FD table plumbed into `RustSimState` — the same
//! blocker as `angr-k3ol` (file-path with real procs). Until that
//! infra lands, the syscall dispatcher falls back to Python for these
//! so the FD-table side effects continue to apply.

use super::{NativeSyscall, SyscallError, SyscallOutcome};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Macro to declare a stub syscall handler with arbitrary arity.
///
/// Generates a unit struct implementing `NativeSyscall` whose `call`
/// returns a fresh `RustBV::symbolic` of width `arch().bits()` (matching
/// Python's `syscall_stub.py::syscall` for syscalls with no dedicated
/// `SimProcedure` registered in the kernel library). Args are ignored.
///
/// Duplicated from `concurrency.rs` / `rlimit.rs` / `memory_extras.rs` /
/// `signals.rs` / `file_path.rs` / `identity::stub_syscall_1arg!` — the
/// seventh caller is past the dedup threshold. Tracked as a follow-up:
/// extracting a crate-wide `pub(crate) macro_rules!` would save ~30 lines
/// per file. Until then keep each module self-contained.
macro_rules! stub_syscall {
    ($ty:ident, $label:expr, $sym_name:expr, $nargs:expr) => {
        pub struct $ty;

        impl NativeSyscall for $ty {
            fn name(&self) -> &'static str {
                $label
            }

            fn num_args(&self) -> usize {
                $nargs
            }

            fn call(
                &self,
                state: &mut RustSimState,
                _args: &[RustBV],
            ) -> Result<SyscallOutcome, SyscallError> {
                let bits = state.arch().bits();
                let ret = {
                    let ctx = state.solver().borrow();
                    RustBV::symbolic(&ctx, $sym_name, bits)
                };
                Ok(SyscallOutcome::ContinueSymbolic { ret })
            }
        }
    };
}

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
