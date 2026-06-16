// Tests for syscalls/memory_extras.rs (extracted sibling module).
// Covers the seven memory-extras syscall handlers returning fresh symbolics.
use super::*;
use crate::state::RustSimState;
use crate::symbolic::RustBV;
use crate::syscalls::{NativeSyscall, SyscallOutcome};

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

            let args: Vec<RustBV> = (0..nargs).map(|_| RustBV::concrete(0, bits)).collect();
            let outcome = handler
                .call(&mut state, &args)
                .unwrap_or_else(|e| panic!("{arch} {label} errored: {e:?}"));
            let ret = match outcome {
                SyscallOutcome::ContinueSymbolic { ret } => ret,
                other => panic!("{arch} {label} expected ContinueSymbolic, got {other:?}"),
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
                (RustBV::Symbolic { id: a, .. }, RustBV::Symbolic { id: b, .. }) => (*a, *b),
                _ => panic!("{arch} {label} returns should be Symbolic"),
            };
            assert_ne!(
                id1, id2,
                "{arch} {label} successive calls must yield distinct fresh symbols",
            );
        }
    }
}
