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
    fn test_write_unknown_fd_falls_back() {
        // fd=3 is not open in the FileSystem → Native falls back.
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"x", Permission::RWX);
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
    fn test_write_stdin_falls_back() {
        // fd=0 is open (stdin) but we refuse to write — fall back so the
        // Python posix model can produce EBADF.
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"x", Permission::RWX);
        let result = NativeWrite.call(
            &mut state,
            &[
                RustBV::concrete(0, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(1, 64),
            ],
        );
        assert!(result.is_err());
        // The native handler must not have appended to stdin's content.
        assert_eq!(state.file_system_ref().fd_content(0), b"");
    }

    #[test]
    fn test_write_user_fd_appends_to_filesystem() {
        // Open fd=3 via NativeOpen path; native write should append into the
        // file's content buffer without entering Python.
        let mut state = RustSimState::new("amd64").unwrap();
        state.file_system().open(
            "out.bin".to_string(),
            crate::state::FdFlags::WriteOnly,
        );
        state.map_memory_data(0x1000, b"hello", Permission::RWX);

        let result = NativeWrite
            .call(
                &mut state,
                &[
                    RustBV::concrete(3, 64),
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(5, 64),
                ],
            )
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(5));
        assert_eq!(state.file_system_ref().fd_content(3), b"hello");
        // stdout untouched.
        assert_eq!(state.stdout_buffer(), b"");
    }

    #[test]
    fn test_write_closed_fd_falls_back() {
        // open + close → is_open false → fall back.
        let mut state = RustSimState::new("amd64").unwrap();
        state.file_system().open(
            "out.bin".to_string(),
            crate::state::FdFlags::WriteOnly,
        );
        assert!(state.file_system().close(3));
        state.map_memory_data(0x1000, b"x", Permission::RWX);

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
