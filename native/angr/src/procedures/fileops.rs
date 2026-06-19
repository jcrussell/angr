//! Native file operation implementations: open, close, lseek, dup, dup2, pipe,
//! fopen/fclose/fseek/ftell/rewind/fdopen, access.
//!
//! These procedures track file descriptor state in the FileSystem.
//! open() allocates a new fd, close() marks it closed, lseek() adjusts position.
//! dup/dup2 duplicate a fd; pipe creates a (read_fd, write_fd) pair.
//! fopen/fdopen allocate a `_IO_FILE` struct on the heap and store the fd at the
//! arch-specific `_fileno` offset; fclose/fseek/ftell/rewind dispatch off it.
//! Only handles concrete arguments; symbolic arguments fall back to Python.

use super::ProcedureError;
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

crate::declare_proc! {
    /// Native open implementation.
    ///
    /// ```c
    /// int open(const char *pathname, int flags, ...);
    /// ```
    ///
    /// Reads the pathname string from memory, allocates a new fd in the FileSystem.
    /// Returns the new fd number, or -1 on error.
    name = "open",
    struct = NativeOpen,
    args = [pathname_addr: concrete, flags: concrete],
    call |state| {
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
        let fd_flags = FdFlags::from_posix(flags as u32);
        let fd = state.file_system().open(pathname, fd_flags);

        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(fd as u128, bits)))
    }
}

crate::declare_proc! {
    /// Native close implementation.
    ///
    /// ```c
    /// int close(int fd);
    /// ```
    ///
    /// Marks the fd as closed. Returns 0 on success, -1 if not open.
    name = "close",
    struct = NativeClose,
    args = [fd: concrete],
    call |state| {
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

crate::declare_proc! {
    /// Native lseek implementation.
    ///
    /// ```c
    /// off_t lseek(int fd, off_t offset, int whence);
    /// ```
    ///
    /// Adjusts the file position. Returns the new position, or -1 on error.
    name = "lseek",
    struct = NativeLseek,
    args = [fd: concrete, offset: concrete, whence: concrete],
    call |state| {
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

crate::declare_proc! {
    /// Native dup implementation.
    ///
    /// ```c
    /// int dup(int oldfd);
    /// ```
    ///
    /// Allocates a new fd that refers to the same underlying file as `oldfd`.
    /// Returns the new fd, or -1 if `oldfd` is not open.
    name = "dup",
    struct = NativeDup,
    args = [oldfd: concrete],
    call |state| {
        let bits = state.arch().bits();
        match state.file_system().dup(oldfd as u32) {
            Some(newfd) => Ok(Some(RustBV::concrete(newfd as u128, bits))),
            None => Ok(Some(RustBV::concrete((-1i64 as u64) as u128, bits))),
        }
    }
}

crate::declare_proc! {
    /// Native dup2 implementation.
    ///
    /// ```c
    /// int dup2(int oldfd, int newfd);
    /// ```
    ///
    /// Makes `newfd` a copy of `oldfd`, closing `newfd` first if it was open.
    /// Returns `newfd` on success, or -1 if `oldfd` is not open.
    name = "dup2",
    struct = NativeDup2,
    args = [oldfd: concrete, newfd: concrete],
    call |state| {
        let bits = state.arch().bits();
        match state.file_system().dup2(oldfd as u32, newfd as u32) {
            Some(fd) => Ok(Some(RustBV::concrete(fd as u128, bits))),
            None => Ok(Some(RustBV::concrete((-1i64 as u64) as u128, bits))),
        }
    }
}

crate::declare_proc! {
    /// Native pipe implementation.
    ///
    /// ```c
    /// int pipe(int pipefd[2]);
    /// ```
    ///
    /// Creates a (read_fd, write_fd) pair and writes them as two consecutive
    /// 32-bit ints to `pipefd`. Returns 0 on success.
    name = "pipe",
    struct = NativePipe,
    args = [pipefd_addr: concrete],
    call |state| {
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

crate::declare_proc! {
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
    name = "fopen",
    struct = NativeFopen,
    args = [path_addr: concrete, mode_addr: concrete],
    call |state| {
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

crate::declare_proc! {
    /// Native fdopen implementation.
    ///
    /// ```c
    /// FILE *fdopen(int fd, const char *mode);
    /// ```
    ///
    /// Allocates an `_IO_FILE` struct for an already-open fd. Returns 0 if `fd` is
    /// not open in the FileSystem so the Python proc can fall back to its
    /// symbolic-fd path.
    name = "fdopen",
    struct = NativeFdopen,
    args = [fd_raw: concrete, mode_addr: concrete],
    call |state| {
        let fd = fd_raw as u32 as i32;
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

crate::declare_proc! {
    /// Native fclose implementation.
    ///
    /// ```c
    /// int fclose(FILE *stream);
    /// ```
    ///
    /// Reads `stream->_fileno` and calls FileSystem::close. Returns 0 on success
    /// and -1 if the fd was not open.
    name = "fclose",
    struct = NativeFclose,
    args = [file_ptr: concrete],
    call |state| {
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

crate::declare_proc! {
    /// Native fseek implementation.
    ///
    /// ```c
    /// int fseek(FILE *stream, long offset, int whence);
    /// ```
    ///
    /// Reads `stream->_fileno` and seeks via FileSystem::seek. Returns 0 on
    /// success and -1 on failure (matching glibc).
    name = "fseek",
    struct = NativeFseek,
    args = [file_ptr: concrete, offset_raw: concrete, whence_raw: concrete],
    call |state| {
        let offset = offset_raw as i64;
        let whence = whence_raw as u32;
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

crate::declare_proc! {
    /// Native ftell implementation.
    ///
    /// ```c
    /// long ftell(FILE *stream);
    /// ```
    ///
    /// Reads `stream->_fileno` and returns the current position via
    /// FileSystem::fd_info. Returns -1 if the fd is not tracked.
    name = "ftell",
    struct = NativeFtell,
    args = [file_ptr: concrete],
    call |state| {
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

crate::declare_proc! {
    /// Native rewind implementation.
    ///
    /// ```c
    /// void rewind(FILE *stream);
    /// ```
    ///
    /// Equivalent to `fseek(stream, 0, SEEK_SET)` but returns void. The exploration
    /// manager handles void returns by leaving the return register untouched.
    name = "rewind",
    struct = NativeRewind,
    args = [file_ptr: concrete],
    call |state| {
        let fd = read_fileno(state, file_ptr)?;

        if fd >= 0 {
            // SEEK_SET = 0
            let _ = state.file_system().seek(fd as u32, 0, 0);
        }
        Ok(None)
    }
}

#[cfg(test)]
#[path = "fileops_tests.rs"]
mod fileops_tests;
