//! Native file operation implementations: open, close, lseek, dup, dup2, pipe,
//! fopen/fclose/fseek/ftell/rewind/fdopen, access.
//!
//! These procedures track file descriptor state in the FileSystem.
//! open() allocates a new fd, close() marks it closed, lseek() adjusts position.
//! dup/dup2 duplicate a fd; pipe creates a (read_fd, write_fd) pair.
//! fopen/fdopen allocate a `_IO_FILE` struct on the heap and store the fd at the
//! arch-specific `_fileno` offset; fclose/fseek/ftell/rewind dispatch off it.
//! Only handles concrete arguments; symbolic arguments fall back to Python.

use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg};
use crate::state::{FdFlags, RustSimState};
use crate::symbolic::RustBV;

/// `_IO_FILE` size and fd offset per arch, mirroring
/// `cle.backends.externs.simdata.io_file.io_file_data_for_arch`.
/// Returns `(fd_offset, total_size)`.
fn io_file_for_arch(name: &str) -> Option<(u64, u64)> {
    match name {
        "AMD64" => Some((112, 216)),
        "X86" => Some((56, 148)),
        "ARM" | "ARMEL" | "ARMHF" => Some((14, 84)),
        "ARM64" | "AARCH64" => Some((20, 152)),
        "MIPS32" => Some((56, 148)),
        "MIPS64" => Some((112, 216)),
        _ => None,
    }
}

const MAX_FOPEN_PATH_LEN: u64 = 256;
const MAX_FOPEN_MODE_LEN: u64 = 8;

/// Read a NUL-terminated string from memory up to `max_len` bytes. Concrete only.
fn read_cstring(
    state: &RustSimState,
    addr: u64,
    max_len: u64,
    name: &str,
) -> Result<Vec<u8>, ProcedureError> {
    let mut out = Vec::new();
    for i in 0..max_len {
        let bv = state.memory_load(addr.wrapping_add(i), 1)?;
        let v = bv
            .as_u64()
            .ok_or_else(|| ProcedureError::SymbolicArgument(format!("symbolic byte in {name}")))?;
        if v == 0 {
            return Ok(out);
        }
        out.push(v as u8);
    }
    Err(ProcedureError::Other(format!(
        "{name} not NUL-terminated within {max_len} bytes"
    )))
}

/// Convert an fopen-style mode string (e.g. `"r"`, `"w+b"`) to FdFlags.
/// Returns None for unrecognized modes (caller falls back to Python).
fn parse_fopen_mode(mode: &[u8]) -> Option<FdFlags> {
    let mut bytes: Vec<u8> = mode.to_vec();
    if matches!(bytes.last(), Some(b'b') | Some(b't')) {
        bytes.pop();
    }
    bytes.retain(|c| *c != b'c' && *c != b'e');
    match bytes.as_slice() {
        b"r" => Some(FdFlags::ReadOnly),
        b"r+" => Some(FdFlags::ReadWrite),
        b"w" => Some(FdFlags::WriteOnly),
        b"w+" => Some(FdFlags::ReadWrite),
        b"a" => Some(FdFlags::WriteOnly),
        b"a+" => Some(FdFlags::ReadWrite),
        _ => None,
    }
}

/// Read a 32-bit fd from a FILE struct on the given arch. Returns the signed fd
/// (so -1 sentinels are preserved) and any symbolic / arch-resolution errors.
fn read_fileno(state: &RustSimState, file_ptr: u64) -> Result<i32, ProcedureError> {
    let arch_name = state.arch().name();
    let (fd_off, _) = io_file_for_arch(arch_name)
        .ok_or_else(|| ProcedureError::Other(format!("no _IO_FILE layout for arch {arch_name}")))?;
    let bv = state.memory_load(file_ptr.wrapping_add(fd_off), 4)?;
    let raw = bv
        .as_u64()
        .ok_or_else(|| ProcedureError::SymbolicArgument("FILE._fileno".to_string()))?;
    Ok(raw as u32 as i32)
}

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
                            "symbolic byte in pathname".to_string(),
                        ));
                    }
                }
                Err(e) => return Err(e.into()),
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
        let ret = if success {
            0u128
        } else {
            (-1i64 as u64) as u128
        };
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
        match state
            .file_system()
            .seek(fd as u32, offset as i64, whence as u32)
        {
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
            read_fd.to_le_bytes()
        } else {
            read_fd.to_be_bytes()
        };
        let write_bytes = if little {
            write_fd.to_le_bytes()
        } else {
            write_fd.to_be_bytes()
        };
        for (i, b) in read_bytes.iter().enumerate() {
            state.memory_store(
                pipefd_addr.wrapping_add(i as u64),
                RustBV::concrete(*b as u128, 8),
            )?;
        }
        for (i, b) in write_bytes.iter().enumerate() {
            state.memory_store(
                pipefd_addr.wrapping_add(4 + i as u64),
                RustBV::concrete(*b as u128, 8),
            )?;
        }

        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(0, bits)))
    }
}

/// Native fopen implementation.
///
/// ```c
/// FILE *fopen(const char *pathname, const char *mode);
/// ```
///
/// Opens an fd, allocates an `_IO_FILE` struct on the heap, writes the fd at the
/// arch-specific `_fileno` offset, and returns the struct pointer. Returns 0 on
/// unrecognized modes / unsupported arches / unmapped heap pages so the Python
/// implementation can take over.
pub struct NativeFopen;

impl NativeSimProcedure for NativeFopen {
    fn name(&self) -> &'static str {
        "fopen"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let path_addr = extract_concrete_arg(&args[0], "pathname")?;
        let mode_addr = extract_concrete_arg(&args[1], "mode")?;

        let path = read_cstring(state, path_addr, MAX_FOPEN_PATH_LEN, "pathname")?;
        let mode = read_cstring(state, mode_addr, MAX_FOPEN_MODE_LEN, "mode")?;
        let flags = parse_fopen_mode(&mode).ok_or_else(|| {
            ProcedureError::Other(format!(
                "unsupported fopen mode {:?}",
                String::from_utf8_lossy(&mode)
            ))
        })?;

        let arch_name = state.arch().name();
        let (fd_off, struct_size) = io_file_for_arch(arch_name).ok_or_else(|| {
            ProcedureError::Other(format!("no _IO_FILE layout for arch {arch_name}"))
        })?;

        let path_str = String::from_utf8_lossy(&path).to_string();
        let fd = state.file_system().open(path_str, flags);

        let file_ptr = state.heap_alloc(struct_size);
        state.memory_store(
            file_ptr.wrapping_add(fd_off),
            RustBV::concrete(fd as u128, 32),
        )?;

        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(file_ptr as u128, bits)))
    }
}

/// Native fdopen implementation.
///
/// ```c
/// FILE *fdopen(int fd, const char *mode);
/// ```
///
/// Allocates an `_IO_FILE` struct for an already-open fd. Returns 0 if `fd` is
/// not open in the FileSystem so the Python proc can fall back to its
/// symbolic-fd path.
pub struct NativeFdopen;

impl NativeSimProcedure for NativeFdopen {
    fn name(&self) -> &'static str {
        "fdopen"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let fd = extract_concrete_arg(&args[0], "fd")? as u32 as i32;
        let mode_addr = extract_concrete_arg(&args[1], "mode")?;

        let mode = read_cstring(state, mode_addr, MAX_FOPEN_MODE_LEN, "mode")?;
        parse_fopen_mode(&mode).ok_or_else(|| {
            ProcedureError::Other(format!(
                "unsupported fdopen mode {:?}",
                String::from_utf8_lossy(&mode)
            ))
        })?;

        let bits = state.arch().bits();
        if fd < 0 || !state.file_system_ref().is_open(fd as u32) {
            return Ok(Some(RustBV::concrete(0, bits)));
        }

        let arch_name = state.arch().name();
        let (fd_off, struct_size) = io_file_for_arch(arch_name).ok_or_else(|| {
            ProcedureError::Other(format!("no _IO_FILE layout for arch {arch_name}"))
        })?;

        let file_ptr = state.heap_alloc(struct_size);
        state.memory_store(
            file_ptr.wrapping_add(fd_off),
            RustBV::concrete(fd as u32 as u128, 32),
        )?;

        Ok(Some(RustBV::concrete(file_ptr as u128, bits)))
    }
}

/// Native fclose implementation.
///
/// ```c
/// int fclose(FILE *stream);
/// ```
///
/// Reads `stream->_fileno` and calls FileSystem::close. Returns 0 on success
/// and -1 if the fd was not open.
pub struct NativeFclose;

impl NativeSimProcedure for NativeFclose {
    fn name(&self) -> &'static str {
        "fclose"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let file_ptr = extract_concrete_arg(&args[0], "stream")?;
        let fd = read_fileno(state, file_ptr)?;

        let bits = state.arch().bits();
        if fd < 0 {
            return Ok(Some(RustBV::concrete((-1i64 as u64) as u128, bits)));
        }
        let ret = if state.file_system().close(fd as u32) {
            0u128
        } else {
            (-1i64 as u64) as u128
        };
        Ok(Some(RustBV::concrete(ret, bits)))
    }
}

/// Native fseek implementation.
///
/// ```c
/// int fseek(FILE *stream, long offset, int whence);
/// ```
///
/// Reads `stream->_fileno` and seeks via FileSystem::seek. Returns 0 on
/// success and -1 on failure (matching glibc).
pub struct NativeFseek;

impl NativeSimProcedure for NativeFseek {
    fn name(&self) -> &'static str {
        "fseek"
    }

    fn num_args(&self) -> usize {
        3
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let file_ptr = extract_concrete_arg(&args[0], "stream")?;
        let offset = extract_concrete_arg(&args[1], "offset")? as i64;
        let whence = extract_concrete_arg(&args[2], "whence")? as u32;

        let fd = read_fileno(state, file_ptr)?;
        let bits = state.arch().bits();
        if fd < 0 {
            return Ok(Some(RustBV::concrete((-1i64 as u64) as u128, bits)));
        }
        match state.file_system().seek(fd as u32, offset, whence) {
            Some(_) => Ok(Some(RustBV::concrete(0, bits))),
            None => Ok(Some(RustBV::concrete((-1i64 as u64) as u128, bits))),
        }
    }
}

/// Native ftell implementation.
///
/// ```c
/// long ftell(FILE *stream);
/// ```
///
/// Reads `stream->_fileno` and returns the current position via
/// FileSystem::fd_info. Returns -1 if the fd is not tracked.
pub struct NativeFtell;

impl NativeSimProcedure for NativeFtell {
    fn name(&self) -> &'static str {
        "ftell"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let file_ptr = extract_concrete_arg(&args[0], "stream")?;
        let fd = read_fileno(state, file_ptr)?;

        let bits = state.arch().bits();
        if fd < 0 {
            return Ok(Some(RustBV::concrete((-1i64 as u64) as u128, bits)));
        }
        match state.file_system_ref().fd_info(fd as u32) {
            Some((_, pos, _, _, _)) => Ok(Some(RustBV::concrete(pos as u128, bits))),
            None => Ok(Some(RustBV::concrete((-1i64 as u64) as u128, bits))),
        }
    }
}

/// Native rewind implementation.
///
/// ```c
/// void rewind(FILE *stream);
/// ```
///
/// Equivalent to `fseek(stream, 0, SEEK_SET)` but returns void. The exploration
/// manager handles void returns by leaving the return register untouched.
pub struct NativeRewind;

impl NativeSimProcedure for NativeRewind {
    fn name(&self) -> &'static str {
        "rewind"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let file_ptr = extract_concrete_arg(&args[0], "stream")?;
        let fd = read_fileno(state, file_ptr)?;

        if fd >= 0 {
            // SEEK_SET = 0
            let _ = state.file_system().seek(fd as u32, 0, 0);
        }
        Ok(None)
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

        let result = NativeOpen
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
            )
            .unwrap();

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

        let r1 = NativeOpen
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
            )
            .unwrap();
        let r2 = NativeOpen
            .call(
                &mut state,
                &[RustBV::concrete(0x2000, 64), RustBV::concrete(0, 64)],
            )
            .unwrap();

        assert_eq!(r1.unwrap().as_u64(), Some(3));
        assert_eq!(r2.unwrap().as_u64(), Some(4));
    }

    #[test]
    fn test_close_basic() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"test.txt\0", Permission::RWX);

        // Open a file
        NativeOpen
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
            )
            .unwrap();

        // Close it
        let result = NativeClose
            .call(&mut state, &[RustBV::concrete(3, 64)])
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(0)); // success
        assert!(!state.file_system_ref().is_open(3));
    }

    #[test]
    fn test_close_not_open() {
        let mut state = RustSimState::new("amd64").unwrap();

        let result = NativeClose
            .call(&mut state, &[RustBV::concrete(99, 64)])
            .unwrap();

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
        let result = NativeLseek
            .call(
                &mut state,
                &[
                    RustBV::concrete(3, 64),
                    RustBV::concrete(42, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();

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
        NativeLseek
            .call(
                &mut state,
                &[
                    RustBV::concrete(3, 64),
                    RustBV::concrete(10, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();

        // SEEK_CUR +5
        let result = NativeLseek
            .call(
                &mut state,
                &[
                    RustBV::concrete(3, 64),
                    RustBV::concrete(5, 64),
                    RustBV::concrete(1, 64),
                ],
            )
            .unwrap();

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
        let result = NativeLseek
            .call(
                &mut state,
                &[
                    RustBV::concrete(3, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(2, 64),
                ],
            )
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(100));
    }

    #[test]
    fn test_dup_basic() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"a.txt\0", Permission::RWX);
        // Open fd=3
        NativeOpen
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
            )
            .unwrap();

        // dup(3) should allocate fd=4
        let result = NativeDup
            .call(&mut state, &[RustBV::concrete(3, 64)])
            .unwrap();
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
        let result = NativeDup
            .call(&mut state, &[RustBV::concrete(99, 64)])
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(u64::MAX));
    }

    #[test]
    fn test_dup_after_close() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"a.txt\0", Permission::RWX);
        NativeOpen
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
            )
            .unwrap();
        NativeClose
            .call(&mut state, &[RustBV::concrete(3, 64)])
            .unwrap();

        // dup of a closed fd returns -1.
        let result = NativeDup
            .call(&mut state, &[RustBV::concrete(3, 64)])
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(u64::MAX));
    }

    #[test]
    fn test_dup2_basic() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"a.txt\0", Permission::RWX);
        NativeOpen
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
            )
            .unwrap();

        // dup2(3, 7) — newfd=7 was not open
        let result = NativeDup2
            .call(
                &mut state,
                &[RustBV::concrete(3, 64), RustBV::concrete(7, 64)],
            )
            .unwrap();
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
        NativeOpen
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
            )
            .unwrap();
        NativeOpen
            .call(
                &mut state,
                &[RustBV::concrete(0x2000, 64), RustBV::concrete(0, 64)],
            )
            .unwrap();

        // dup2(3, 4) — overwrite fd=4 with a duplicate of fd=3
        let result = NativeDup2
            .call(
                &mut state,
                &[RustBV::concrete(3, 64), RustBV::concrete(4, 64)],
            )
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(4));
        // fd=4 now refers to a.txt (the old b.txt is overwritten).
        assert_eq!(state.file_system_ref().fd_info(4).unwrap().0, "a.txt");
    }

    #[test]
    fn test_dup2_same_fd() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"a.txt\0", Permission::RWX);
        NativeOpen
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
            )
            .unwrap();

        // dup2(3, 3) is a no-op when oldfd is open; returns 3.
        let result = NativeDup2
            .call(
                &mut state,
                &[RustBV::concrete(3, 64), RustBV::concrete(3, 64)],
            )
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(3));
        assert!(state.file_system_ref().is_open(3));
    }

    #[test]
    fn test_dup2_oldfd_not_open() {
        let mut state = RustSimState::new("amd64").unwrap();
        let result = NativeDup2
            .call(
                &mut state,
                &[RustBV::concrete(99, 64), RustBV::concrete(7, 64)],
            )
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(u64::MAX));
        assert!(!state.file_system_ref().is_open(7));
    }

    #[test]
    fn test_pipe_basic() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, &[0u8; 16], Permission::RWX);

        // pipe(pipefd) — should return 0 and allocate two fds at 3 and 4.
        let result = NativePipe
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap();
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

        NativePipe
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap();
        // pipefd[0] = 3, pipefd[1] = 4

        // dup2(3, 0): redirect stdin.
        let result = NativeDup2
            .call(
                &mut state,
                &[RustBV::concrete(3, 64), RustBV::concrete(0, 64)],
            )
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(0));

        // fd=0 is now the pipe read end, not the original stdin.
        assert_eq!(state.file_system_ref().fd_info(0).unwrap().0, "<pipe:r>");
        // fd=3 is still open (dup2 doesn't close oldfd).
        assert!(state.file_system_ref().is_open(3));
        assert!(state.file_system_ref().is_open(4));
    }

    // --- fopen / fdopen / fclose / fseek / ftell / rewind ---

    /// Set up an amd64 state with the heap region mapped so heap_alloc-backed
    /// writes can land. Mirrors the pattern used in malloc tests.
    fn setup_amd64_state() -> RustSimState {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory(0xC000_0000, 0x10000, Permission::RWX);
        state
    }

    #[test]
    fn test_fopen_reads_path_and_writes_fileno() {
        let mut state = setup_amd64_state();
        state.map_memory_data(0x1000, b"data.bin\0", Permission::RWX);
        state.map_memory_data(0x2000, b"r\0", Permission::RWX);

        let result = NativeFopen
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
            )
            .unwrap()
            .unwrap();
        let file_ptr = result.as_u64().unwrap();
        assert_ne!(file_ptr, 0);

        // FILE._fileno at offset 112 on amd64 should be fd=3 (first user fd).
        let fd_bv = state.memory_load(file_ptr + 112, 4).unwrap();
        assert_eq!(fd_bv.as_u64(), Some(3));
        assert!(state.file_system_ref().is_open(3));
        assert_eq!(state.file_system_ref().fd_info(3).unwrap().0, "data.bin");
        // r → ReadOnly (0)
        assert_eq!(state.file_system_ref().fd_info(3).unwrap().2, 0);
    }

    #[test]
    fn test_fopen_write_mode_flags_writeonly() {
        let mut state = setup_amd64_state();
        state.map_memory_data(0x1000, b"out.txt\0", Permission::RWX);
        state.map_memory_data(0x2000, b"w\0", Permission::RWX);

        NativeFopen
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
            )
            .unwrap();
        // w → WriteOnly (1)
        assert_eq!(state.file_system_ref().fd_info(3).unwrap().2, 1);
    }

    #[test]
    fn test_fopen_rw_plus_mode() {
        let mut state = setup_amd64_state();
        state.map_memory_data(0x1000, b"rw.txt\0", Permission::RWX);
        state.map_memory_data(0x2000, b"r+\0", Permission::RWX);

        NativeFopen
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
            )
            .unwrap();
        // r+ → ReadWrite (2)
        assert_eq!(state.file_system_ref().fd_info(3).unwrap().2, 2);
    }

    #[test]
    fn test_fopen_binary_suffix_ignored() {
        let mut state = setup_amd64_state();
        state.map_memory_data(0x1000, b"bin.bin\0", Permission::RWX);
        state.map_memory_data(0x2000, b"rb\0", Permission::RWX);

        NativeFopen
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
            )
            .unwrap();
        // rb → r → ReadOnly (0)
        assert_eq!(state.file_system_ref().fd_info(3).unwrap().2, 0);
    }

    #[test]
    fn test_fopen_unknown_mode_falls_back() {
        let mut state = setup_amd64_state();
        state.map_memory_data(0x1000, b"x.txt\0", Permission::RWX);
        state.map_memory_data(0x2000, b"xyz\0", Permission::RWX);

        let result = NativeFopen.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_fopen_unterminated_path_errors() {
        let mut state = setup_amd64_state();
        // No NUL within mapped region — read_cstring will hit a memory error
        // before reaching MAX_FOPEN_PATH_LEN (page boundary triggers Unmapped).
        state.map_memory_data(0x1000, &vec![b'A'; 0x1000], Permission::RWX);
        state.map_memory_data(0x2000, b"r\0", Permission::RWX);

        let result = NativeFopen.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_fclose_releases_fd() {
        let mut state = setup_amd64_state();
        state.map_memory_data(0x1000, b"f.txt\0", Permission::RWX);
        state.map_memory_data(0x2000, b"r\0", Permission::RWX);

        let file_bv = NativeFopen
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
            )
            .unwrap()
            .unwrap();
        let file_ptr = file_bv.as_u64().unwrap();
        assert!(state.file_system_ref().is_open(3));

        let ret = NativeFclose
            .call(&mut state, &[RustBV::concrete(file_ptr as u128, 64)])
            .unwrap()
            .unwrap();
        assert_eq!(ret.as_u64(), Some(0));
        assert!(!state.file_system_ref().is_open(3));
    }

    #[test]
    fn test_fclose_returns_minus_one_for_closed_fd() {
        let mut state = setup_amd64_state();
        // Manually construct a FILE struct with a stale fd.
        state.map_memory_data(0x10000, &vec![0u8; 0x1000], Permission::RWX);
        state
            .memory_store(0x10000 + 112, RustBV::concrete(42, 32))
            .unwrap();

        let ret = NativeFclose
            .call(&mut state, &[RustBV::concrete(0x10000, 64)])
            .unwrap()
            .unwrap();
        assert_eq!(ret.as_u64(), Some(u64::MAX));
    }

    #[test]
    fn test_fclose_negative_fileno_returns_minus_one() {
        let mut state = setup_amd64_state();
        state.map_memory_data(0x10000, &vec![0u8; 0x1000], Permission::RWX);
        state
            .memory_store(0x10000 + 112, RustBV::concrete((-1i32 as u32) as u128, 32))
            .unwrap();

        let ret = NativeFclose
            .call(&mut state, &[RustBV::concrete(0x10000, 64)])
            .unwrap()
            .unwrap();
        assert_eq!(ret.as_u64(), Some(u64::MAX));
    }

    #[test]
    fn test_fseek_set_then_ftell() {
        let mut state = setup_amd64_state();
        state.file_system().open_with_content(
            "data".to_string(),
            FdFlags::ReadOnly,
            vec![0u8; 100],
        );
        // Manually build a FILE struct pointing at fd=3.
        state.map_memory_data(0x20000, &vec![0u8; 0x1000], Permission::RWX);
        state
            .memory_store(0x20000 + 112, RustBV::concrete(3, 32))
            .unwrap();

        // fseek(fp, 42, SEEK_SET)
        let ret = NativeFseek
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x20000, 64),
                    RustBV::concrete(42, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap()
            .unwrap();
        assert_eq!(ret.as_u64(), Some(0));

        // ftell(fp) == 42
        let pos = NativeFtell
            .call(&mut state, &[RustBV::concrete(0x20000, 64)])
            .unwrap()
            .unwrap();
        assert_eq!(pos.as_u64(), Some(42));
    }

    #[test]
    fn test_fseek_invalid_whence_returns_minus_one() {
        let mut state = setup_amd64_state();
        state.file_system().open_with_content(
            "data".to_string(),
            FdFlags::ReadOnly,
            vec![0u8; 100],
        );
        state.map_memory_data(0x20000, &vec![0u8; 0x1000], Permission::RWX);
        state
            .memory_store(0x20000 + 112, RustBV::concrete(3, 32))
            .unwrap();

        // whence=99 is invalid → -1
        let ret = NativeFseek
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x20000, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(99, 64),
                ],
            )
            .unwrap()
            .unwrap();
        assert_eq!(ret.as_u64(), Some(u64::MAX));
    }

    #[test]
    fn test_fseek_unknown_fd_returns_minus_one() {
        let mut state = setup_amd64_state();
        state.map_memory_data(0x20000, &vec![0u8; 0x1000], Permission::RWX);
        state
            .memory_store(0x20000 + 112, RustBV::concrete(99, 32))
            .unwrap();

        let ret = NativeFseek
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x20000, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap()
            .unwrap();
        assert_eq!(ret.as_u64(), Some(u64::MAX));
    }

    #[test]
    fn test_ftell_unknown_fd_returns_minus_one() {
        let mut state = setup_amd64_state();
        state.map_memory_data(0x20000, &vec![0u8; 0x1000], Permission::RWX);
        // fd=77 never registered.
        state
            .memory_store(0x20000 + 112, RustBV::concrete(77, 32))
            .unwrap();

        let pos = NativeFtell
            .call(&mut state, &[RustBV::concrete(0x20000, 64)])
            .unwrap()
            .unwrap();
        assert_eq!(pos.as_u64(), Some(u64::MAX));
    }

    #[test]
    fn test_rewind_returns_none_and_resets_position() {
        let mut state = setup_amd64_state();
        state.file_system().open_with_content(
            "data".to_string(),
            FdFlags::ReadOnly,
            vec![0u8; 100],
        );
        state.map_memory_data(0x20000, &vec![0u8; 0x1000], Permission::RWX);
        state
            .memory_store(0x20000 + 112, RustBV::concrete(3, 32))
            .unwrap();

        // Seek to a non-zero position first.
        NativeFseek
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x20000, 64),
                    RustBV::concrete(50, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();
        assert_eq!(state.file_system_ref().fd_info(3).unwrap().1, 50);

        // rewind returns void (None) and resets position to 0.
        let ret = NativeRewind
            .call(&mut state, &[RustBV::concrete(0x20000, 64)])
            .unwrap();
        assert!(ret.is_none());
        assert_eq!(state.file_system_ref().fd_info(3).unwrap().1, 0);
    }

    #[test]
    fn test_fdopen_existing_fd() {
        let mut state = setup_amd64_state();
        // Open via FileSystem directly so fd 3 is known to be open.
        state
            .file_system()
            .open("foo".to_string(), FdFlags::ReadOnly);
        state.map_memory_data(0x2000, b"r\0", Permission::RWX);

        let fp = NativeFdopen
            .call(
                &mut state,
                &[RustBV::concrete(3, 64), RustBV::concrete(0x2000, 64)],
            )
            .unwrap()
            .unwrap();
        let file_ptr = fp.as_u64().unwrap();
        assert_ne!(file_ptr, 0);

        let fd_bv = state.memory_load(file_ptr + 112, 4).unwrap();
        assert_eq!(fd_bv.as_u64(), Some(3));
    }

    #[test]
    fn test_fdopen_unknown_fd_returns_null() {
        let mut state = setup_amd64_state();
        state.map_memory_data(0x2000, b"r\0", Permission::RWX);

        // fd 42 is not registered → fdopen returns NULL.
        let fp = NativeFdopen
            .call(
                &mut state,
                &[RustBV::concrete(42, 64), RustBV::concrete(0x2000, 64)],
            )
            .unwrap()
            .unwrap();
        assert_eq!(fp.as_u64(), Some(0));
    }

    #[test]
    fn test_fdopen_negative_fd_returns_null() {
        let mut state = setup_amd64_state();
        state.map_memory_data(0x2000, b"r\0", Permission::RWX);

        let fp = NativeFdopen
            .call(
                &mut state,
                &[
                    RustBV::concrete((-1i64 as u64) as u128, 64),
                    RustBV::concrete(0x2000, 64),
                ],
            )
            .unwrap()
            .unwrap();
        assert_eq!(fp.as_u64(), Some(0));
    }
}
