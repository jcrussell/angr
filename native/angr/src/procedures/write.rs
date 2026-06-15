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
//! all fall back to Python. See bd memory
//! `invariant-rust-filesystem-no-python-sync`.

use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const MAX_WRITE_SIZE: u64 = 4096;

/// Native write implementation.
///
/// ```c
/// ssize_t write(int fd, const void *buf, size_t count);
/// ```
///
/// Handles any fd that is open in the Rust `FileSystem` and is not fd=0
/// (stdin). Symbolic bytes in `[buf, buf+count)` or counts beyond
/// `MAX_WRITE_SIZE` fall back to Python.
pub struct NativeWrite;

impl NativeSimProcedure for NativeWrite {
    fn name(&self) -> &'static str {
        "write"
    }

    fn num_args(&self) -> usize {
        3
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let fd = extract_concrete_arg(&args[0], "fd")?;

        if fd == 0 {
            return Err(ProcedureError::Other(
                "write to fd=0 (stdin) falls back to Python".to_string(),
            ));
        }
        let fd_u32 = fd as u32;
        if !state.file_system_ref().is_open(fd_u32) {
            return Err(ProcedureError::Other(format!(
                "write to fd={} (not open in Rust FileSystem) falls back to Python",
                fd
            )));
        }

        let buf = extract_concrete_arg(&args[1], "buf")?;
        let count = extract_concrete_arg(&args[2], "count")?;

        if count > MAX_WRITE_SIZE {
            return Err(ProcedureError::Other(format!(
                "write count {} exceeds limit",
                count
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
                            "symbolic byte at buf+{}",
                            i
                        )));
                    }
                }
                Err(e) => {
                    return Err(e.into());
                }
            }
        }

        state.write_fd(fd_u32, &bytes);

        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(count as u128, bits)))
    }
}

#[cfg(test)]
#[path = "write_tests.rs"]
mod tests;
