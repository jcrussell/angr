//! Memory-advisory syscall handlers.
//!
//! Mirrors angr's behavior for `madvise`, `mremap`, `msync`, `mlock`,
//! `munlock`, `mlockall`, and `munlockall`. None of these have a
//! dedicated Python `SimProcedure` — the unhandled-syscall path falls
//! through to `procedures/stubs/syscall_stub.py::syscall`, which returns
//! `state.solver.Unconstrained("syscall_stub_<name>", returnty.size, ...)`.
//!
//! The native handlers below match that exactly: ignore the args (no
//! semantic effect on simulated process state) and emit a fresh
//! `RustBV::symbolic` sized to `arch().bits()` (the C `long` return
//! width on every supported Linux arch). The dispatcher routes this
//! through `SyscallOutcome::ContinueSymbolic`.
//!
//! Notes:
//! * `mremap` is registered as a stub for parity with Python; angr's
//!   `syscall_stub` does NOT update page tables, so this matches.
//!   Full semantic mremap (move/resize mappings + page-table coordination
//!   with the existing mmap/mprotect/munmap handlers) is tracked
//!   separately so the campaign close here is a pure parity win.
//! * Advisory ops (`mlock`, `munlock`, `mlockall`, `munlockall`,
//!   `madvise`, `msync`) are inherently no-ops in the symbolic VM —
//!   the symbolic return surfaces any binary that branches on it.

use super::{NativeSyscall, SyscallError, SyscallOutcome};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Macro to declare a stub syscall handler with arbitrary arity.
///
/// Generates a unit struct implementing `NativeSyscall` whose `call`
/// returns a fresh `RustBV::symbolic` of width `arch().bits()` (matching
/// Python's `syscall_stub.py::syscall` for syscalls with no dedicated
/// `SimProcedure`). Args are intentionally ignored.
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

// madvise(start, len, behavior) → long
stub_syscall!(NativeMadviseSyscall, "madvise", "syscall_stub_madvise", 3);
// mremap(addr, old_len, new_len, flags, new_addr) → long
stub_syscall!(NativeMremapSyscall, "mremap", "syscall_stub_mremap", 5);
// msync(start, len, flags) → long
stub_syscall!(NativeMsyncSyscall, "msync", "syscall_stub_msync", 3);
// mlock(start, len) → long
stub_syscall!(NativeMlockSyscall, "mlock", "syscall_stub_mlock", 2);
// munlock(start, len) → long
stub_syscall!(NativeMunlockSyscall, "munlock", "syscall_stub_munlock", 2);
// mlockall(flags) → long
stub_syscall!(NativeMlockallSyscall, "mlockall", "syscall_stub_mlockall", 1);
// munlockall() → long
stub_syscall!(NativeMunlockallSyscall, "munlockall", "syscall_stub_munlockall", 0);

#[cfg(test)]
mod tests {
    use super::*;

    /// All seven handlers must return a fresh `RustBV::symbolic` of width
    /// `arch().bits()` on every supported arch, mirroring `syscall_stub`
    /// semantics bit-for-bit. Successive invocations must yield distinct
    /// fresh symbols (different `RustBV::Symbolic.id`).
    #[test]
    fn memory_extras_return_fresh_symbolic_on_all_arches() {
        // (handler, expected name, arity, dummy_arg_count_for_args_slice)
        // The args slice length doesn't matter — handlers ignore it — but
        // we pass `num_args()`-many zero BVs to mirror the dispatcher.
        let cases: &[(&'static dyn NativeSyscall, &str, usize)] = &[
            (&NativeMadviseSyscall, "madvise", 3),
            (&NativeMremapSyscall, "mremap", 5),
            (&NativeMsyncSyscall, "msync", 3),
            (&NativeMlockSyscall, "mlock", 2),
            (&NativeMunlockSyscall, "munlock", 2),
            (&NativeMlockallSyscall, "mlockall", 1),
            (&NativeMunlockallSyscall, "munlockall", 0),
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
                assert_eq!(
                    ret.width(),
                    bits,
                    "{arch} {label} return width should match arch().bits()",
                );
                assert!(
                    ret.as_u64().is_none(),
                    "{arch} {label} return must be symbolic",
                );

                // Distinct fresh symbol on second invocation.
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
                    _ => panic!("{arch} {label} returns should be Symbolic"),
                };
                assert_ne!(
                    id1, id2,
                    "{arch} {label} successive calls must yield distinct fresh symbols",
                );
            }
        }
    }
}
