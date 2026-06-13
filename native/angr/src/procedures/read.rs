//! Native read implementation.
//!
//! Handles `read(fd, buf, count)` for:
//!   - fd=0 (stdin) by storing fresh symbolic bytes and recording the symbol
//!     names so the Python state export can splice them into `posix.dumps(0)`.
//!   - any other fd open in the Rust `FileSystem` that has remaining concrete
//!     content (pos < content_len) — bytes are served from the FS buffer,
//!     advancing the position.
//!
//! Falls back to Python for:
//!   - symbolic fd/buf/count
//!   - count beyond `MAX_READ_SIZE`
//!   - fds not open in the Rust `FileSystem` (which includes any fd Python
//!     created via its symbolic-file plumbing; see fd-table invariant below).
//!   - non-stdin open fds with empty / fully-consumed content (Python's
//!     symbolic-file model owns those reads).
//!
//! ## Fd-table sync invariant (angr-8j16)
//!
//! Rust's `FileSystem` uses a monotonic fd counter, Python's `state.posix.fd`
//! uses lowest-free. They are NOT kept in sync. The native handlers cover
//! only fds in Rust's table; everything else falls back so the Python
//! symbolic-file model can take over. See bd memory
//! `invariant-rust-filesystem-no-python-sync`.

use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg, symbol_counter};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const MAX_READ_SIZE: u64 = 4096;

/// Native read implementation.
///
/// ```c
/// ssize_t read(int fd, void *buf, size_t count);
/// ```
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

        if fd == 0 {
            return read_stdin_symbolic(state, buf, count);
        }

        let fd_u32 = fd as u32;
        let (open, content_len) = match state.file_system_ref().fd_info(fd_u32) {
            Some((_, _, _, len, is_open)) => (is_open, len),
            None => (false, 0),
        };
        if !open {
            return Err(ProcedureError::Other(format!(
                "read from fd={} (not open in Rust FileSystem) falls back to Python",
                fd
            )));
        }
        // Empty / no-content fds defer to Python so the symbolic-file model
        // (cle simfs + SimFile) can produce symbolic bytes.
        if content_len == 0 {
            return Err(ProcedureError::Other(format!(
                "read from fd={} has no concrete content; falling back to Python",
                fd
            )));
        }

        let bytes = state.file_system().read(fd_u32, count as usize);
        let n = bytes.len();
        for (i, b) in bytes.iter().enumerate() {
            state.memory_store(buf.wrapping_add(i as u64), RustBV::concrete(*b as u128, 8))?;
        }
        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(n as u128, bits)))
    }
}

fn read_stdin_symbolic(
    state: &mut RustSimState,
    buf: u64,
    count: u64,
) -> Result<Option<RustBV>, ProcedureError> {
    let read_id = symbol_counter("read");

    let names: Vec<String> = (0..count)
        .map(|i| format!("stdin_{}_{}", read_id, i))
        .collect();
    let sym_bytes: Vec<RustBV> = {
        let ctx = state.solver().borrow();
        names
            .iter()
            .map(|name| RustBV::symbolic(&ctx, name, 8))
            .collect()
    };

    for name in &names {
        state.record_stdin_symbol(name.clone(), 8);
    }

    for (i, sym_byte) in sym_bytes.into_iter().enumerate() {
        state.memory_store(buf.wrapping_add(i as u64), sym_byte)?;
    }

    let bits = state.arch().bits();
    Ok(Some(RustBV::concrete(count as u128, bits)))
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
    fn test_read_records_stdin_symbols() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory(0x2000, 0x1000, Permission::RWX);

        assert!(!state.has_stdin_symbols());
        let _ = NativeRead
            .call(
                &mut state,
                &[
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(3, 64),
                ],
            )
            .unwrap();

        let symbols = state.stdin_symbols();
        assert_eq!(symbols.len(), 3);
        for (_, bits) in symbols {
            assert_eq!(*bits, 8);
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
    fn test_read_unknown_fd_falls_back() {
        // fd=3 is not open in the FileSystem → fall back.
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory(0x2000, 0x1000, crate::memory::Permission::RWX);
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
    fn test_read_user_fd_with_content_serves_natively() {
        // Open fd=3 with concrete content; native read should serve from FS.
        let mut state = RustSimState::new("amd64").unwrap();
        state.file_system().open_with_content(
            "in.bin".to_string(),
            crate::state::FdFlags::ReadOnly,
            b"abcdef".to_vec(),
        );
        state.map_memory(0x2000, 0x1000, crate::memory::Permission::RWX);

        // Read 3 bytes; should return 3 and copy "abc" to memory.
        let result = NativeRead
            .call(
                &mut state,
                &[
                    RustBV::concrete(3, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(3, 64),
                ],
            )
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(3));
        for (i, &want) in b"abc".iter().enumerate() {
            let byte = state.memory_load(0x2000 + i as u64, 1).unwrap();
            assert_eq!(byte.as_u64(), Some(want as u64));
        }
        // Position advanced.
        let info = state.file_system_ref().fd_info(3).unwrap();
        assert_eq!(info.1, 3);
    }

    #[test]
    fn test_read_user_fd_eof_returns_zero() {
        // Read past the end of the content returns 0 (Linux EOF semantics).
        let mut state = RustSimState::new("amd64").unwrap();
        state.file_system().open_with_content(
            "in.bin".to_string(),
            crate::state::FdFlags::ReadOnly,
            b"ab".to_vec(),
        );
        state.map_memory(0x2000, 0x1000, crate::memory::Permission::RWX);

        // Drain 2 bytes.
        NativeRead
            .call(
                &mut state,
                &[
                    RustBV::concrete(3, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(2, 64),
                ],
            )
            .unwrap();
        // Next read: should return 0.
        let result = NativeRead
            .call(
                &mut state,
                &[
                    RustBV::concrete(3, 64),
                    RustBV::concrete(0x2010, 64),
                    RustBV::concrete(4, 64),
                ],
            )
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(0));
    }

    #[test]
    fn test_read_empty_content_fd_falls_back() {
        // fd open but no content → defer to Python (symbolic-file model).
        let mut state = RustSimState::new("amd64").unwrap();
        state
            .file_system()
            .open("in.bin".to_string(), crate::state::FdFlags::ReadOnly);
        state.map_memory(0x2000, 0x1000, crate::memory::Permission::RWX);
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
    fn test_read_closed_fd_falls_back() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.file_system().open_with_content(
            "in.bin".to_string(),
            crate::state::FdFlags::ReadOnly,
            b"x".to_vec(),
        );
        assert!(state.file_system().close(3));
        state.map_memory(0x2000, 0x1000, crate::memory::Permission::RWX);
        let result = NativeRead.call(
            &mut state,
            &[
                RustBV::concrete(3, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(1, 64),
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
