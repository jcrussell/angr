//! Native perror implementation.
//!
//! ```c
//! void perror(const char *s);
//! ```
//!
//! Matches Python `procedures/posix/perror.py`, which forwards to
//! `write(2, s, strlen(s))` — i.e. it writes only the user-supplied string to
//! stderr, without the glibc `": <errno message>\n"` suffix. We mirror that:
//! scan the NUL-terminated string and append it to fd 2's buffer (the same
//! `write_fd` / `FileSystem::write` path as `NativeWrite`), returning void.
//!
//! Falls back to Python when the string pointer or any byte is symbolic, when
//! the string is not NUL-terminated within `MAX_PERROR_LEN`, or when fd 2 is
//! not open in the Rust `FileSystem` (so the symbolic-file model can handle it
//! — see `invariant-rust-filesystem-no-python-sync`).

use super::ProcedureError;
use super::strings::scan_concrete_until_null;

const MAX_PERROR_LEN: usize = 4096;

const STDERR_FD: u32 = 2;

crate::declare_proc! {
    name = "perror",
    struct = NativePerror,
    args = [string: concrete],
    call |state| {
        if !state.file_system_ref().is_open(STDERR_FD) {
            return Err(ProcedureError::Other(
                "perror: fd=2 (stderr) not open in Rust FileSystem, falls back to Python"
                    .to_string(),
            ));
        }
        let bytes = scan_concrete_until_null(state, string, MAX_PERROR_LEN, "perror string")?;
        // A refusal means fd 2 was rebound (dup2) onto a bounded-symbolic-
        // content fd, now demoted — bounce to Python (angr-0xyq2 Phase 2
        // choke point; see FileSystem::write).
        if !state.write_fd(STDERR_FD, &bytes) {
            return Err(ProcedureError::Other(
                "perror to stderr with symbolic content falls back to Python (demoted)"
                    .to_string(),
            ));
        }
        Ok(None)
    }
}

#[cfg(test)]
#[path = "perror_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod perror_tests;
