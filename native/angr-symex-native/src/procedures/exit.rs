//! Native exit/abort implementations.
//!
//! These are no-return procedures that terminate the current state.
//! They don't need Python callbacks since they just deadend the state.
//!
//! Each returns its exit status, which the dispatcher hands to
//! `RustSimState::record_exit` for the export's `terminate` event (see
//! `NativeSimProcedure::no_return`), as Python's `SimProcedure.exit` records
//! it: exit/_exit pass their argument, abort and `__stack_chk_fail` (aliases
//! of Python's `abort`) pass 1 (angr-vmi2l.151).
//!
//! exit/_exit declare their `status` arg as `bv` (not `concrete`) so a
//! symbolic exit status is passed through rather than forcing a Python
//! fallback — preserving the always-deadend behavior regardless of arg form.

crate::declare_proc! {
    /// Native exit implementation.
    ///
    /// ```c
    /// void exit(int status);
    /// ```
    name = "exit",
    struct = NativeExit,
    args = [status: bv],
    no_return = true,
    call |_state| {
        Ok(Some(status))
    }
}

crate::declare_proc! {
    /// Native _exit implementation (same as exit for our purposes).
    name = "_exit",
    struct = NativeUnderscoreExit,
    args = [status: bv],
    no_return = true,
    call |_state| {
        Ok(Some(status))
    }
}

crate::declare_proc! {
    /// Native abort implementation.
    name = "abort",
    struct = NativeAbort,
    args = [],
    no_return = true,
    call |state| {
        Ok(Some(abort_status(state)))
    }
}

crate::declare_proc! {
    /// Native `__stack_chk_fail` implementation.
    ///
    /// Stack-protector failure handler emitted by GCC/Clang. Like abort, it
    /// never returns — the state is deadended by the dispatcher when
    /// `no_return()` is true. Already recognized as a terminal in the Python
    /// callback dispatcher's `_SIMPROC_NO_RET_TERMINAL` set, so the native
    /// path keeps semantics identical while skipping the Python round-trip.
    name = "__stack_chk_fail",
    struct = NativeStackChkFail,
    args = [],
    no_return = true,
    call |state| {
        Ok(Some(abort_status(state)))
    }
}

/// Python's `abort.py` exit status, `self.exit(1)` at the arch's width.
fn abort_status(state: &crate::state::RustSimState) -> crate::symbolic::RustBV {
    crate::symbolic::RustBV::concrete(1, state.arch().bits())
}

test_submod!("exit_tests.rs" => tests);
