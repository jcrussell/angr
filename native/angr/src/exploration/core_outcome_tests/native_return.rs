//! angr-sqfj8.37: the native-return SP bump is gated on `pops_return_addr`, so
//! it fires on stack-return ABIs and not on link-register ones — and
//! angr-c7xno.29: a symbolic SP must stay symbolic across it.
//!
//! Drives the same hook-dispatch fixtures as the sibling `hooks` module
//! ([`dispatch_hook_on_arch`], in the parent), but varies the architecture and
//! the seeded SP rather than the find/avoid configuration.

use super::*;

const SP_SEED: u64 = 0x7fff_0000;

/// Run the native-return path on `arch` and hand back the successor's SP.
fn native_return_sp(arch: &str) -> u64 {
    let outcome = dispatch_hook_on_arch(arch, SpSeed::Concrete(SP_SEED), |_| {});
    assert_eq!(
        outcome.counters.native_calls, 1,
        "{arch}: native proc must have run"
    );
    match outcome.ret {
        CoreReturn::Continue(succ) => {
            assert_eq!(succ.len(), 1);
            assert_eq!(succ[0].0.pc(), HOOK_RET, "{arch}: landed at return address");
            succ[0]
                .0
                .get_sp()
                .as_u64()
                .unwrap_or_else(|| panic!("{arch}: SP went symbolic"))
        }
        _ => panic!("{arch}: expected Continue"),
    }
}

/// Stack-return ABIs pop the return address, so the parallel path advances SP
/// by one pointer — the control for the link-register cases below.
#[test]
fn native_return_advances_sp_on_stack_return_abi() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        assert_eq!(native_return_sp("amd64"), SP_SEED + 8);
        assert_eq!(native_return_sp("x86"), SP_SEED + 4);
    });
}

/// ARM/ARM64/MIPS return through LR/X30/$ra: nothing was pushed, so nothing may
/// be popped. Before angr-sqfj8.37 `handle_simprocedure_core` bumped SP
/// unconditionally — only `step_one` (run_loop_single.rs) gated it — so every
/// native proc return under the wave/steady/worker engines silently dropped a
/// pointer-sized live stack slot.
#[test]
fn native_return_leaves_sp_untouched_on_link_register_abis() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        for arch in ["armel", "aarch64", "mips32", "mips64"] {
            assert_eq!(
                native_return_sp(arch),
                SP_SEED,
                "{arch} returns via a link register; SP must not move"
            );
        }
    });
}

/// A symbolic SP must survive the native-return bump (angr-c7xno.29).
///
/// `extract_args_with_abi` only reads SP when the argument count overflows the
/// register window, so a zero-arg native proc runs to completion with SP still
/// symbolic. The pre-fix `get_sp().as_u64().unwrap_or(0)` then rewrote that SP
/// to a bogus concrete `ptr_size` (8 on amd64, 4 on x86) — the shared
/// `advance_sp_past_return_addr` helper adds the pointer symbolically instead.
#[test]
fn native_return_keeps_symbolic_sp_symbolic_on_stack_return_abi() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        for arch in ["amd64", "x86"] {
            let outcome = dispatch_hook_on_arch(arch, SpSeed::Symbolic, |_| {});
            let CoreReturn::Continue(succ) = outcome.ret else {
                panic!("{arch}: expected Continue");
            };
            let sp = succ[0].0.get_sp();
            assert!(
                sp.as_u64().is_none(),
                "{arch}: symbolic SP was concretized to {:?} — the unwrap_or(0) bug",
                sp.as_u64()
            );
        }
    });
}

/// Control for the test above on the link-register ABIs: the bump is skipped
/// entirely there, so the SP register must come out as the *same* symbol it
/// went in as. (This one passed pre-fix too — it guards against the shared
/// helper regressing into an unconditional bump.)
#[test]
fn native_return_keeps_symbolic_sp_symbolic_on_link_register_abis() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        for arch in ["armel", "aarch64", "mips32", "mips64"] {
            let outcome = dispatch_hook_on_arch(arch, SpSeed::Symbolic, |_| {});
            let CoreReturn::Continue(succ) = outcome.ret else {
                panic!("{arch}: expected Continue");
            };
            let sp = succ[0].0.get_sp();
            assert!(
                matches!(sp, crate::symbolic::RustBV::Symbolic { .. }),
                "{arch}: SP must be the untouched symbol, got {sp:?}"
            );
        }
    });
}
