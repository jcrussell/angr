//! Native read implementation.
//!
//! Handles read(fd, buf, count) for stdin (fd=0) by creating symbolic bytes.
//! Other file descriptors fall back to Python.

use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

use std::sync::atomic::{AtomicU64, Ordering};

const MAX_READ_SIZE: u64 = 4096;

/// Counter for unique stdin read variable names.
static READ_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Native read implementation.
///
/// ```c
/// ssize_t read(int fd, void *buf, size_t count);
/// ```
///
/// Only handles fd=0 (stdin) natively by storing symbolic bytes.
/// Other fds fall back to Python.
pub struct NativeRead;

impl NativeSimProcedure for NativeRead {
    fn name(&self) -> &'static str {
        "read"
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

        // Only handle stdin natively
        if fd != 0 {
            return Err(ProcedureError::Other(format!(
                "read from fd={} not supported natively",
                fd
            )));
        }

        let buf = extract_concrete_arg(&args[1], "buf")?;
        let count = extract_concrete_arg(&args[2], "count")?;

        if count > MAX_READ_SIZE {
            return Err(ProcedureError::Other(format!(
                "read count {} exceeds limit",
                count
            )));
        }

        if count == 0 {
            let bits = state.arch().bits();
            return Ok(Some(RustBV::concrete(0, bits)));
        }

        // Create symbolic bytes and store to buffer
        let read_id = READ_COUNTER.fetch_add(1, Ordering::Relaxed);

        // First, create all symbolic bytes (needs solver borrow)
        let sym_bytes: Vec<RustBV> = {
            let ctx = state.solver().borrow();
            (0..count)
                .map(|i| {
                    let name = format!("stdin_{}_{}", read_id, i);
                    RustBV::symbolic(&ctx, &name, 8)
                })
                .collect()
        };

        // Then store them (needs mutable state, solver borrow released)
        for (i, sym_byte) in sym_bytes.into_iter().enumerate() {
            state
                .memory_store(buf.wrapping_add(i as u64), sym_byte)
                .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
        }

        // Return count (symbolic read returns full count)
        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(count as u128, bits)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;

    #[test]
    fn test_read_stdin() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory(0x2000, 0x1000, Permission::RWX);

        let result = NativeRead
            .call(
                &mut state,
                &[
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(4, 64),
                ],
            )
            .unwrap();

        // Should return count
        assert_eq!(result.unwrap().as_u64(), Some(4));

        // Each byte should be symbolic
        for i in 0..4u64 {
            let byte = state.memory_load(0x2000 + i, 1).unwrap();
            assert!(byte.as_u64().is_none()); // symbolic
        }
    }

    #[test]
    fn test_read_zero_count() {
        let mut state = RustSimState::new("amd64").unwrap();
        let result = NativeRead
            .call(
                &mut state,
                &[
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(0));
    }

    #[test]
    fn test_read_non_stdin_fails() {
        let mut state = RustSimState::new("amd64").unwrap();
        let result = NativeRead.call(
            &mut state,
            &[
                RustBV::concrete(3, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(4, 64),
            ],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_read_too_large() {
        let mut state = RustSimState::new("amd64").unwrap();
        let result = NativeRead.call(
            &mut state,
            &[
                RustBV::concrete(0, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(5000, 64),
            ],
        );
        assert!(result.is_err());
    }
}
