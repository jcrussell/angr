//! Native file operation implementations: open, close, lseek, dup, dup2, pipe.
//!
//! These procedures track file descriptor state in the FileSystem.
//! open() allocates a new fd, close() marks it closed, lseek() adjusts position.
//! dup/dup2 duplicate a fd; pipe creates a (read_fd, write_fd) pair.
//! Only handles concrete arguments; symbolic arguments fall back to Python.

use crate::state::RustSimState;
use crate::symbolic::RustBV;
use super::{extract_concrete_arg, NativeSimProcedure, ProcedureError};

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
        let pathname_addr = extract_concrete_arg(&args[0], "pathname")?;
        let flags = extract_concrete_arg(&args[1], "flags")?;

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
        let fd = extract_concrete_arg(&args[0], "fd")?;

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
        let fd = extract_concrete_arg(&args[0], "fd")?;
        let offset = extract_concrete_arg(&args[1], "offset")?;
        let whence = extract_concrete_arg(&args[2], "whence")?;

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

/// Native dup implementation.
///
/// ```c
/// int dup(int oldfd);
/// ```
///
/// Allocates a new fd that refers to the same underlying file as `oldfd`.
/// Returns the new fd, or -1 if `oldfd` is not open.
pub struct NativeDup;

impl NativeSimProcedure for NativeDup {
    fn name(&self) -> &'static str {
        "dup"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let oldfd = extract_concrete_arg(&args[0], "oldfd")?;

        let bits = state.arch().bits();
        match state.file_system().dup(oldfd as u32) {
            Some(newfd) => Ok(Some(RustBV::concrete(newfd as u128, bits))),
            None => Ok(Some(RustBV::concrete((-1i64 as u64) as u128, bits))),
        }
    }
}

/// Native dup2 implementation.
///
/// ```c
/// int dup2(int oldfd, int newfd);
/// ```
///
/// Makes `newfd` a copy of `oldfd`, closing `newfd` first if it was open.
/// Returns `newfd` on success, or -1 if `oldfd` is not open.
pub struct NativeDup2;

impl NativeSimProcedure for NativeDup2 {
    fn name(&self) -> &'static str {
        "dup2"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let oldfd = extract_concrete_arg(&args[0], "oldfd")?;
        let newfd = extract_concrete_arg(&args[1], "newfd")?;

        let bits = state.arch().bits();
        match state.file_system().dup2(oldfd as u32, newfd as u32) {
            Some(fd) => Ok(Some(RustBV::concrete(fd as u128, bits))),
            None => Ok(Some(RustBV::concrete((-1i64 as u64) as u128, bits))),
        }
    }
}

/// Native pipe implementation.
///
/// ```c
/// int pipe(int pipefd[2]);
/// ```
///
/// Creates a (read_fd, write_fd) pair and writes them as two consecutive
/// 32-bit ints to `pipefd`. Returns 0 on success.
pub struct NativePipe;

impl NativeSimProcedure for NativePipe {
    fn name(&self) -> &'static str {
        "pipe"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let pipefd_addr = extract_concrete_arg(&args[0], "pipefd")?;

        let (read_fd, write_fd) = state.file_system().pipe();

        // Write the two 32-bit fds as raw bytes at pipefd[0..4] and pipefd[4..8].
        // Use byte-by-byte little/big-endian encoding so pipefd[0] reads back correctly
        // through the standard 32-bit memory load (which respects arch endianness).
        let little = state.arch().is_little_endian();
        let read_bytes = if little {
            (read_fd as u32).to_le_bytes()
        } else {
            (read_fd as u32).to_be_bytes()
        };
        let write_bytes = if little {
            (write_fd as u32).to_le_bytes()
        } else {
            (write_fd as u32).to_be_bytes()
        };
        for (i, b) in read_bytes.iter().enumerate() {
            state.memory_store(pipefd_addr.wrapping_add(i as u64), RustBV::concrete(*b as u128, 8))?;
        }
        for (i, b) in write_bytes.iter().enumerate() {
            state.memory_store(pipefd_addr.wrapping_add(4 + i as u64), RustBV::concrete(*b as u128, 8))?;
        }

        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(0, bits)))
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

    #[test]
    fn test_dup_basic() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"a.txt\0", Permission::RWX);
        // Open fd=3
        NativeOpen.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
        ).unwrap();

        // dup(3) should allocate fd=4
        let result = NativeDup.call(&mut state, &[RustBV::concrete(3, 64)]).unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(4));

        // Both fds should be open and refer to the same name.
        assert!(state.file_system_ref().is_open(3));
        assert!(state.file_system_ref().is_open(4));
        assert_eq!(state.file_system_ref().fd_info(4).unwrap().0, "a.txt");
    }

    #[test]
    fn test_dup_closed_fd() {
        let mut state = RustSimState::new("amd64").unwrap();
        // dup of a never-opened fd returns -1.
        let result = NativeDup.call(&mut state, &[RustBV::concrete(99, 64)]).unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(u64::MAX));
    }

    #[test]
    fn test_dup_after_close() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"a.txt\0", Permission::RWX);
        NativeOpen.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
        ).unwrap();
        NativeClose.call(&mut state, &[RustBV::concrete(3, 64)]).unwrap();

        // dup of a closed fd returns -1.
        let result = NativeDup.call(&mut state, &[RustBV::concrete(3, 64)]).unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(u64::MAX));
    }

    #[test]
    fn test_dup2_basic() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"a.txt\0", Permission::RWX);
        NativeOpen.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
        ).unwrap();

        // dup2(3, 7) — newfd=7 was not open
        let result = NativeDup2.call(
            &mut state,
            &[RustBV::concrete(3, 64), RustBV::concrete(7, 64)],
        ).unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(7));
        assert!(state.file_system_ref().is_open(7));
        assert_eq!(state.file_system_ref().fd_info(7).unwrap().0, "a.txt");

        // next_fd should now be past 7 so subsequent open doesn't collide.
        assert!(state.file_system_ref().next_fd() > 7);
    }

    #[test]
    fn test_dup2_closes_target() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"a.txt\0", Permission::RWX);
        state.map_memory_data(0x2000, b"b.txt\0", Permission::RWX);
        // Open two fds: 3=a.txt, 4=b.txt
        NativeOpen.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
        ).unwrap();
        NativeOpen.call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0, 64)],
        ).unwrap();

        // dup2(3, 4) — overwrite fd=4 with a duplicate of fd=3
        let result = NativeDup2.call(
            &mut state,
            &[RustBV::concrete(3, 64), RustBV::concrete(4, 64)],
        ).unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(4));
        // fd=4 now refers to a.txt (the old b.txt is overwritten).
        assert_eq!(state.file_system_ref().fd_info(4).unwrap().0, "a.txt");
    }

    #[test]
    fn test_dup2_same_fd() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"a.txt\0", Permission::RWX);
        NativeOpen.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
        ).unwrap();

        // dup2(3, 3) is a no-op when oldfd is open; returns 3.
        let result = NativeDup2.call(
            &mut state,
            &[RustBV::concrete(3, 64), RustBV::concrete(3, 64)],
        ).unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(3));
        assert!(state.file_system_ref().is_open(3));
    }

    #[test]
    fn test_dup2_oldfd_not_open() {
        let mut state = RustSimState::new("amd64").unwrap();
        let result = NativeDup2.call(
            &mut state,
            &[RustBV::concrete(99, 64), RustBV::concrete(7, 64)],
        ).unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(u64::MAX));
        assert!(!state.file_system_ref().is_open(7));
    }

    #[test]
    fn test_pipe_basic() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, &[0u8; 16], Permission::RWX);

        // pipe(pipefd) — should return 0 and allocate two fds at 3 and 4.
        let result = NativePipe.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64)],
        ).unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(0));

        // Read pipefd[0] (4 bytes at 0x1000) and pipefd[1] (4 bytes at 0x1004).
        let read_fd_bv = state.memory_load(0x1000, 4).unwrap();
        let write_fd_bv = state.memory_load(0x1004, 4).unwrap();
        assert_eq!(read_fd_bv.as_u64(), Some(3));
        assert_eq!(write_fd_bv.as_u64(), Some(4));

        // Both fds open with correct flags.
        assert!(state.file_system_ref().is_open(3));
        assert!(state.file_system_ref().is_open(4));
        let info_r = state.file_system_ref().fd_info(3).unwrap();
        let info_w = state.file_system_ref().fd_info(4).unwrap();
        // ReadOnly = 0, WriteOnly = 1
        assert_eq!(info_r.2, 0);
        assert_eq!(info_w.2, 1);
    }

    #[test]
    fn test_pipe_then_dup2_to_stdin() {
        // Realistic pattern: pipe(p); dup2(p[0], 0) — redirects stdin to read end.
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, &[0u8; 16], Permission::RWX);

        NativePipe.call(&mut state, &[RustBV::concrete(0x1000, 64)]).unwrap();
        // pipefd[0] = 3, pipefd[1] = 4

        // dup2(3, 0): redirect stdin.
        let result = NativeDup2.call(
            &mut state,
            &[RustBV::concrete(3, 64), RustBV::concrete(0, 64)],
        ).unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(0));

        // fd=0 is now the pipe read end, not the original stdin.
        assert_eq!(state.file_system_ref().fd_info(0).unwrap().0, "<pipe:r>");
        // fd=3 is still open (dup2 doesn't close oldfd).
        assert!(state.file_system_ref().is_open(3));
        assert!(state.file_system_ref().is_open(4));
    }
}
