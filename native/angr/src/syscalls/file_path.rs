//! File-path syscall handlers — symbolic-return subset (angr-0hif.1).
//!
//! Covers `lstat`, `newfstatat`, `readlink`, `readlinkat`, `faccessat`.
//! None of these have a dedicated Python `SimProcedure` in
//! `angr/procedures/linux_kernel/` — the unhandled-syscall path falls
//! through to `procedures/stubs/syscall_stub.py::syscall`, which returns
//! `state.solver.Unconstrained("syscall_stub_<name>", returnty.size, ...)`.
//!
//! The native handlers below match that exactly: ignore the args (no
//! semantic effect on simulated process state) and emit a fresh
//! `RustBV::symbolic` sized to `arch().bits()` (the C `long` return
//! width on every supported Linux arch). The dispatcher routes this
//! through `SyscallOutcome::ContinueSymbolic`.
//!
//! What is intentionally NOT covered here (deferred to a follow-up
//! pass — see bd `follow-up to angr-0hif.1`):
//!
//! * `open`, `openat`, `close` — Python procs in `procedures/posix/`
//!   that allocate / release FDs from `state.posix.fd` and create
//!   `SimFile`s. Needs the FD-table model ported into `RustSimState`.
//! * `stat`, `fstat` — Python procs in `procedures/linux_kernel/` that
//!   call `state.posix.fstat_with_result` and write arch-specific
//!   `struct stat` layouts. Needs the posix plumbing above plus a
//!   per-arch field-layout table.
//! * `access` — Python proc in `procedures/linux_kernel/access.py`
//!   that concretizes the symbolic path under current constraints,
//!   queries `state.fs.get(path)`, and returns `0` or `-1`. Needs
//!   `state.fs` plumbing into Rust state.
//!
//! On those, the unhandled-syscall path continues to dispatch to the
//! Python `_handle_syscall_callback`, preserving full semantics.

use super::stub_syscall;

// lstat(pathname, statbuf) → long
stub_syscall!(NativeLstatSyscall, "lstat", "syscall_stub_lstat", 2);
// newfstatat(dfd, filename, statbuf, flag) → long
stub_syscall!(NativeNewfstatatSyscall, "newfstatat", "syscall_stub_newfstatat", 4);
// readlink(path, buf, bufsiz) → long
stub_syscall!(NativeReadlinkSyscall, "readlink", "syscall_stub_readlink", 3);
// readlinkat(dfd, path, buf, bufsiz) → long
stub_syscall!(NativeReadlinkatSyscall, "readlinkat", "syscall_stub_readlinkat", 4);
// faccessat(dfd, filename, mode) → long
stub_syscall!(NativeFaccessatSyscall, "faccessat", "syscall_stub_faccessat", 3);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::RustSimState;
    use crate::symbolic::RustBV;
    use crate::syscalls::{NativeSyscall, SyscallOutcome};

    /// Sweep all five stub handlers across every supported arch and
    /// verify they return a fresh `RustBV::symbolic` of width
    /// `arch().bits()`. Successive invocations must yield distinct
    /// fresh symbols (different `RustBV::Symbolic.id`), matching
    /// `syscall_stub.py::syscall` semantics where each call gets a
    /// new `Unconstrained` BV.
    #[test]
    fn stub_handlers_return_fresh_symbolic_on_all_arches() {
        // (handler, expected name, arity)
        let cases: &[(&'static dyn NativeSyscall, &str, usize)] = &[
            (&NativeLstatSyscall, "lstat", 2),
            (&NativeNewfstatatSyscall, "newfstatat", 4),
            (&NativeReadlinkSyscall, "readlink", 3),
            (&NativeReadlinkatSyscall, "readlinkat", 4),
            (&NativeFaccessatSyscall, "faccessat", 3),
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
                    .unwrap_or_else(|e| panic!("{arch} {label} errored: {e:?}"));
                let ret = match outcome {
                    SyscallOutcome::ContinueSymbolic { ret } => ret,
                    other => panic!(
                        "{arch} {label} expected ContinueSymbolic, got {other:?}"
                    ),
                };
                assert_eq!(ret.width(), bits, "{arch} {label} width");
                assert!(!ret.is_concrete(), "{arch} {label} should be symbolic");

                // Second call: must produce a *distinct* fresh symbol.
                let outcome2 = handler
                    .call(&mut state, &args)
                    .unwrap_or_else(|e| panic!("{arch} {label} 2nd call errored: {e:?}"));
                let ret2 = match outcome2 {
                    SyscallOutcome::ContinueSymbolic { ret } => ret,
                    other => panic!(
                        "{arch} {label} 2nd: expected ContinueSymbolic, got {other:?}"
                    ),
                };
                let (id1, id2) = match (&ret, &ret2) {
                    (
                        RustBV::Symbolic { id: a, .. },
                        RustBV::Symbolic { id: b, .. },
                    ) => (*a, *b),
                    _ => panic!("{arch} {label} returns must be Symbolic"),
                };
                assert_ne!(
                    id1, id2,
                    "{arch} {label} successive calls must yield distinct fresh symbols",
                );
            }
        }
    }
}
