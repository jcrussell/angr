//! Native syslog-family no-op stubs: openlog / closelog.
//!
//! Angr's Python SimProcedures (`angr/procedures/libc/{openlog,closelog}.py`)
//! are pure stubs that `return` without touching state — they configure the
//! syslog connection, which has no observable effect under symbolic execution.
//! The native side previously had neither, so a PLT libc call to
//! `openlog`/`closelog` round-tripped to Python.
//!
//! Both are void: `call()` returns `Ok(None)` so the run-loop performs the
//! return-address dance but writes no return register (parity with the void
//! Python stubs). Arguments are taken as raw `bv` and discarded, so a symbolic
//! `ident`/`option`/`facility` never forces concretization or a Python
//! fallback.
//!
//! `syslog(3)` itself (the variadic format-string logger,
//! `procedures/posix/syslog.py`, a `FormatParser` subclass) is intentionally
//! NOT modeled here — it needs the format machinery and is left to Python.

crate::declare_proc! {
    /// ```c
    /// void openlog(const char *ident, int option, int facility);
    /// ```
    /// No-op stub: configures syslog, no observable symbolic effect.
    name = "openlog",
    struct = NativeOpenlog,
    args = [_ident: bv, _option: bv, _facility: bv],
    call |_state| {
        Ok(None)
    }
}

crate::declare_proc! {
    /// ```c
    /// void closelog(void);
    /// ```
    /// No-op stub: closes the syslog descriptor, no observable symbolic effect.
    name = "closelog",
    struct = NativeCloselog,
    args = [],
    call |_state| {
        Ok(None)
    }
}

#[cfg(test)]
#[path = "syslog_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod tests;
