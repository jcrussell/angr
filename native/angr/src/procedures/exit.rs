//! Native exit/abort implementations.
//!
//! These are no-return procedures that terminate the current state.
//! They don't need Python callbacks since they just deadend the state.
//!
//! exit/_exit declare their `status` arg as `bv` (not `concrete`) so a
//! symbolic exit status is simply ignored rather than forcing a Python
//! fallback — preserving the always-deadend behavior regardless of arg form.

crate::declare_proc! {
    /// Native exit implementation.
    ///
    /// ```c
    /// void exit(int status);
    /// ```
    name = "exit",
    struct = NativeExit,
    args = [_status: bv],
    no_return = true,
    call |_state| {
        // No-return: just return None to signal state should be deadended.
        Ok(None)
    }
}

crate::declare_proc! {
    /// Native _exit implementation (same as exit for our purposes).
    name = "_exit",
    struct = NativeUnderscoreExit,
    args = [_status: bv],
    no_return = true,
    call |_state| {
        Ok(None)
    }
}

crate::declare_proc! {
    /// Native abort implementation.
    name = "abort",
    struct = NativeAbort,
    args = [],
    no_return = true,
    call |_state| {
        Ok(None)
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
    call |_state| {
        Ok(None)
    }
}

#[cfg(test)]
#[path = "exit_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod tests;
