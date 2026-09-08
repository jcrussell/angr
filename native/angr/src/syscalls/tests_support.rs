//! Fixtures shared by the `*_tests` modules under `syscalls` that exercise the
//! `MAX_IO_SIZE` / `MAX_IOVCNT` bounds checks (`fd_io`, `read`, `write`).
//!
//! Those modules are cousins, not siblings, so — unlike
//! `file_path_tests_support`, whose consumers all sit under `file_path` — the
//! items here are `pub(crate)` and reached by their full path. The module
//! itself is private to `syscalls`, which is enough: privacy reaches every
//! descendant, and every consumer is one.

use super::SyscallError;

/// Assert `err` is the `Other` fallback naming `what` as over-limit.
///
/// Deliberately matches on the message and not just `SyscallError::Other(_)`:
/// every fallback arm in these handlers is an `Other`, so the weaker
/// assertion passes even when the bounce came from an unrelated check.
pub(crate) fn assert_over_limit(err: &SyscallError, what: &str) {
    assert!(
        matches!(err, SyscallError::Other(m) if m.contains(what) && m.contains("exceeds limit")),
        "expected an over-limit fallback mentioning {what}, got {err:?}"
    );
}
