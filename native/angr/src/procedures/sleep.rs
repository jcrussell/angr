//! Native POSIX no-op timers: sleep / usleep.
//!
//! Angr's Python SimProcedures (`angr/procedures/posix/{sleep,usleep}.py`) each
//! ignore their argument and `return 0` — symbolic execution does not model
//! wall-clock delay, so a sleep is a no-op that reports "slept the full
//! duration" (sleep) / success (usleep). The native side previously had
//! neither, so a PLT libc call to `sleep`/`usleep` round-tripped to Python.
//!
//! The argument is taken as a raw `bv` and discarded: parity holds for both
//! concrete and symbolic durations (Python never inspects it), so there is no
//! reason to force concretization and fall back to Python on a symbolic count.
//!
//! The return BV is sized to `arch().bits()`, matching how angr's
//! `SimProcedure.ret(0)` builds a `BVV(0, arch.bits)`.

use crate::symbolic::RustBV;

crate::declare_proc! {
    /// ```c
    /// unsigned int sleep(unsigned int seconds);
    /// ```
    /// Returns 0 (the amount of time left to sleep — none, since we "slept"
    /// the whole duration).
    name = "sleep",
    struct = NativeSleep,
    args = [_seconds: bv],
    call |state| {
        Ok(Some(RustBV::concrete(0, state.arch().bits())))
    }
}

crate::declare_proc! {
    /// ```c
    /// int usleep(useconds_t usec);
    /// ```
    /// Returns 0 (success).
    name = "usleep",
    struct = NativeUsleep,
    args = [_usec: bv],
    call |state| {
        Ok(Some(RustBV::concrete(0, state.arch().bits())))
    }
}

#[cfg(test)]
#[path = "sleep_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod tests;
