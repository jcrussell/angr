//! Native write implementation.
//!
//! Handles write(fd, buf, count) for stdout (fd=1) and stderr (fd=2).
//! Other file descriptors fall back to Python.

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
/// Handles fd=1 (stdout) and fd=2 (stderr) natively. Other fds fall back to Python.
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

        // Handle stdout (fd=1) and stderr (fd=2) natively
        if fd != 1 && fd != 2 {
            return Err(ProcedureError::Other(format!(
                "write to fd={} not supported natively",
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

        // Read bytes from memory and append to stdout buffer
        let mut bytes = Vec::with_capacity(count as usize);
        for i in 0..count {
            match state.memory_load(buf.wrapping_add(i), 1) {
                Ok(bv) => {
                    if let Some(val) = bv.as_u64() {
                        bytes.push(val as u8);
                    } else {
                        // Symbolic byte — can't handle natively
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

        state.write_fd(fd as u32, &bytes);

        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(count as u128, bits)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;

    #[test]
    fn test_write_stdout() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"hello", Permission::RWX);

        let result = NativeWrite
            .call(
                &mut state,
                &[
                    RustBV::concrete(1, 64),
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(5, 64),
                ],
            )
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(5));
        assert_eq!(state.stdout_buffer(), b"hello");
    }

    #[test]
    fn test_write_stderr() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"err", Permission::RWX);

        let result = NativeWrite
            .call(
                &mut state,
                &[
                    RustBV::concrete(2, 64),
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(3, 64),
                ],
            )
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(3));
        assert_eq!(state.fd_buffer(2), b"err");
    }

    #[test]
    fn test_write_unsupported_fd() {
        let mut state = RustSimState::new("amd64").unwrap();
        let result = NativeWrite.call(
            &mut state,
            &[
                RustBV::concrete(3, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(1, 64),
            ],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_write_too_large() {
        let mut state = RustSimState::new("amd64").unwrap();
        let result = NativeWrite.call(
            &mut state,
            &[
                RustBV::concrete(1, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(5000, 64),
            ],
        );
        assert!(result.is_err());
    }
}
