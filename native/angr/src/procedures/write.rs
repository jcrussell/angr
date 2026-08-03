//! Native write implementation.
//!
//! Handles `write(fd, buf, count)` for any fd that is open in the Rust
//! `FileSystem` (stdout/stderr pre-registered; user fds created via
//! `NativeOpen`/`NativePipe`/`NativeDup`/`NativeDup2`/`NativeFopen` etc.).
//! Writing to fd=0 (stdin) or to fds tracked only on the Python side falls
//! back to Python so the symbolic-file model can handle them.
//!
//! ## Fd-table sync invariant (angr-8j16)
//!
//! Rust's `FileSystem` uses a monotonic fd counter, while Python's
//! `state.posix.fd` uses lowest-free. The two tables are NOT kept in sync.
//! The native handlers cover only fds that exist in Rust's table; symbolic
//! fds, fds created on the Python side, or fds backed by symbolic content
//! all fall back to Python (same rule as the "Fd-table sync invariant"
//! section of the `procedures::read` module doc).

use super::ProcedureError;
use crate::symbolic::RustBV;

const MAX_WRITE_SIZE: u64 = 4096;

crate::declare_proc! {
    /// write: serve any fd open in the Rust `FileSystem` (not fd=0).
    ///
    /// ```c
    /// ssize_t write(int fd, const void *buf, size_t count);
    /// ```
    ///
    /// Symbolic bytes in `[buf, buf+count)` or counts beyond `MAX_WRITE_SIZE`
    /// fall back to Python.
    name = "write",
    struct = NativeWrite,
    args = [fd: concrete, buf: concrete, count: concrete],
    call |state| {
        if fd == 0 {
            return Err(ProcedureError::Other(
                "write to fd=0 (stdin) falls back to Python".to_string(),
            ));
        }
        let fd_u32 = fd as u32;
        if !state.file_system_ref().is_open(fd_u32) {
            return Err(ProcedureError::Other(format!(
                "write to fd={fd} (not open in Rust FileSystem) falls back to Python"
            )));
        }
        // Zero-length write: POSIX no-op — return 0 natively WITHOUT
        // demoting bounded symbolic content (angr-0xyq2 A3).
        if count == 0 {
            let bits = state.arch().bits();
            return Ok(Some(RustBV::concrete(0, bits)));
        }
        // Write-demotion (angr-0xyq2 Phase 2): a write to a file with bounded
        // symbolic content drops the content (all sibling fds + registry) and
        // bounces to Python, so this write and all later I/O on the file are
        // consistently Python-owned. Gated BEFORE the size/symbolic-byte
        // bounces below — those fall back to Python too, and must not leave
        // stale native serving behind.
        if state.file_system().demote_symbolic_content(fd_u32) {
            return Err(ProcedureError::Other(format!(
                "write to fd={fd} with symbolic content falls back to Python (demoted)"
            )));
        }

        if count > MAX_WRITE_SIZE {
            return Err(ProcedureError::Other(format!(
                "write count {count} exceeds limit"
            )));
        }

        let mut bytes = Vec::with_capacity(count as usize);
        for i in 0..count {
            match state.memory_load(buf.wrapping_add(i), 1) {
                Ok(bv) => {
                    if let Some(val) = bv.as_u64() {
                        bytes.push(val as u8);
                    } else {
                        return Err(ProcedureError::SymbolicArgument(format!(
                            "symbolic byte at buf+{i}"
                        )));
                    }
                }
                Err(e) => {
                    return Err(e.into());
                }
            }
        }

        // Unreachable after the gate above; kept as choke-point insurance
        // (see FileSystem::write) so no future reordering can mutate a
        // symbolic-content fd natively.
        if !state.write_fd(fd_u32, &bytes) {
            return Err(ProcedureError::Other(format!(
                "write to fd={fd} with symbolic content falls back to Python (demoted)"
            )));
        }

        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(count as u128, bits)))
    }
}

#[cfg(test)]
#[path = "write_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod tests;
