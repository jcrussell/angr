//! Native file operation implementations: open, close, lseek.
//!
//! These procedures track file descriptor state in the FileSystem.
//! open() allocates a new fd, close() marks it closed, lseek() adjusts position.
//! Only handles concrete arguments; symbolic arguments fall back to Python.

use crate::state::RustSimState;
use crate::symbolic::RustBV;
use super::{NativeSimProcedure, ProcedureError};

/// Native open implementation.
///
/// ```c
/// int open(const char *pathname, int flags, ...);
/// ```
///
/// Reads the pathname string from memory, allocates a new fd in the FileSystem.
/// Returns the new fd number, or -1 on error.
pub struct NativeOpen;

impl NativeSimProcedure for NativeOpen {
    fn name(&self) -> &'static str {
        "open"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let pathname_addr = args[0].as_u64().ok_or_else(|| {
            ProcedureError::SymbolicArgument("pathname".to_string())
        })?;
        let flags = args[1].as_u64().ok_or_else(|| {
            ProcedureError::SymbolicArgument("flags".to_string())
        })?;

        // Read pathname string from memory (max 256 bytes)
        let mut name = Vec::new();
        for i in 0..256u64 {
            match state.memory_load(pathname_addr.wrapping_add(i), 1) {
                Ok(bv) => {
                    if let Some(val) = bv.as_u64() {
                        if val == 0 {
                            break;
                        }
                        name.push(val as u8);
                    } else {
                        return Err(ProcedureError::SymbolicArgument(
                            "symbolic byte in pathname".to_string()
                        ));
                    }
                }
                Err(e) => return Err(ProcedureError::MemoryError(e.to_string())),
            }
        }

        let pathname = String::from_utf8_lossy(&name).to_string();
        let fd_flags = crate::state::FdFlags::from_posix(flags as u32);
        let fd = state.file_system().open(pathname, fd_flags);

        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(fd as u128, bits)))
    }
}

/// Native close implementation.
///
/// ```c
/// int close(int fd);
/// ```
///
/// Marks the fd as closed. Returns 0 on success, -1 if not open.
pub struct NativeClose;

impl NativeSimProcedure for NativeClose {
    fn name(&self) -> &'static str {
        "close"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let fd = args[0].as_u64().ok_or_else(|| {
            ProcedureError::SymbolicArgument("fd".to_string())
        })?;

        let success = state.file_system().close(fd as u32);
        let bits = state.arch().bits();
        let ret = if success { 0u128 } else { (-1i64 as u64) as u128 };
        Ok(Some(RustBV::concrete(ret, bits)))
    }
}

/// Native lseek implementation.
///
/// ```c
/// off_t lseek(int fd, off_t offset, int whence);
/// ```
///
/// Adjusts the file position. Returns the new position, or -1 on error.
pub struct NativeLseek;

impl NativeSimProcedure for NativeLseek {
    fn name(&self) -> &'static str {
        "lseek"
    }

    fn num_args(&self) -> usize {
        3
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let fd = args[0].as_u64().ok_or_else(|| {
            ProcedureError::SymbolicArgument("fd".to_string())
        })?;
        let offset = args[1].as_u64().ok_or_else(|| {
            ProcedureError::SymbolicArgument("offset".to_string())
        })?;
        let whence = args[2].as_u64().ok_or_else(|| {
            ProcedureError::SymbolicArgument("whence".to_string())
        })?;

        let bits = state.arch().bits();
        match state.file_system().seek(fd as u32, offset as i64, whence as u32) {
            Some(new_pos) => Ok(Some(RustBV::concrete(new_pos as u128, bits))),
            None => {
                let ret = (-1i64 as u64) as u128;
                Ok(Some(RustBV::concrete(ret, bits)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;

    #[test]
    fn test_open_basic() {
        let mut state = RustSimState::new("amd64").unwrap();
        // Write pathname "test.txt\0" to memory
        state.map_memory_data(0x1000, b"test.txt\0", Permission::RWX);

        let result = NativeOpen.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
        ).unwrap();

        // Should return fd 3 (first user fd after stdin/stdout/stderr)
        assert_eq!(result.unwrap().as_u64(), Some(3));

        // Verify fd is tracked
        assert!(state.file_system_ref().is_open(3));
        let info = state.file_system_ref().fd_info(3).unwrap();
        assert_eq!(info.0, "test.txt");
    }

    #[test]
    fn test_open_multiple() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"a.txt\0", Permission::RWX);
        state.map_memory_data(0x2000, b"b.txt\0", Permission::RWX);

        let r1 = NativeOpen.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
        ).unwrap();
        let r2 = NativeOpen.call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0, 64)],
        ).unwrap();

        assert_eq!(r1.unwrap().as_u64(), Some(3));
        assert_eq!(r2.unwrap().as_u64(), Some(4));
    }

    #[test]
    fn test_close_basic() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"test.txt\0", Permission::RWX);

        // Open a file
        NativeOpen.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
        ).unwrap();

        // Close it
        let result = NativeClose.call(
            &mut state,
            &[RustBV::concrete(3, 64)],
        ).unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(0)); // success
        assert!(!state.file_system_ref().is_open(3));
    }

    #[test]
    fn test_close_not_open() {
        let mut state = RustSimState::new("amd64").unwrap();

        let result = NativeClose.call(
            &mut state,
            &[RustBV::concrete(99, 64)],
        ).unwrap();

        // Should return -1 (not open)
        let val = result.unwrap().as_u64().unwrap();
        assert_eq!(val, u64::MAX); // -1 as u64
    }

    #[test]
    fn test_lseek_set() {
        let mut state = RustSimState::new("amd64").unwrap();
        // Open and write some content
        state.file_system().open_with_content(
            "test.txt".to_string(),
            crate::state::FdFlags::ReadOnly,
            vec![0u8; 100],
        );

        // SEEK_SET to position 42
        let result = NativeLseek.call(
            &mut state,
            &[RustBV::concrete(3, 64), RustBV::concrete(42, 64), RustBV::concrete(0, 64)],
        ).unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(42));
    }

    #[test]
    fn test_lseek_cur() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.file_system().open_with_content(
            "test.txt".to_string(),
            crate::state::FdFlags::ReadOnly,
            vec![0u8; 100],
        );

        // Seek to 10
        NativeLseek.call(
            &mut state,
            &[RustBV::concrete(3, 64), RustBV::concrete(10, 64), RustBV::concrete(0, 64)],
        ).unwrap();

        // SEEK_CUR +5
        let result = NativeLseek.call(
            &mut state,
            &[RustBV::concrete(3, 64), RustBV::concrete(5, 64), RustBV::concrete(1, 64)],
        ).unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(15));
    }

    #[test]
    fn test_lseek_end() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.file_system().open_with_content(
            "test.txt".to_string(),
            crate::state::FdFlags::ReadOnly,
            vec![0u8; 100],
        );

        // SEEK_END + 0
        let result = NativeLseek.call(
            &mut state,
            &[RustBV::concrete(3, 64), RustBV::concrete(0, 64), RustBV::concrete(2, 64)],
        ).unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(100));
    }
}
