//! Native file operation implementations: open, close, lseek, dup, dup2, pipe,
//! fopen/fclose/fseek/ftell/rewind/fdopen, access.
//!
//! These procedures track file descriptor state in the FileSystem.
//! open() allocates a new fd, close() marks it closed, lseek() adjusts position.
//! dup/dup2 duplicate a fd; pipe creates a (read_fd, write_fd) pair.
//! fopen/fdopen allocate a `_IO_FILE` struct on the heap and store the fd at the
//! arch-specific `_fileno` offset; fclose/fseek/ftell/rewind dispatch off it.
//! Only handles concrete arguments; symbolic arguments fall back to Python.

use super::strings::{
    ScanOutcome, null_exists_constraint, scan_concrete_bounded, scan_concrete_until_null,
    scan_for_null_symbolic,
};
use super::{ProcedureError, arch_word};
use crate::state::{FdFlags, RustSimState};
use crate::symbolic::RustBV;

/// Longest pathname `open` will read out of memory.
const MAX_PATH: u64 = 256;

/// Read the NUL-terminated pathname at `addr`, concretizing symbolic bytes the
/// way angr's Python `open` does (`procedures/posix/open.py`): it inline-calls
/// `strlen` and then `solver.eval`s the loaded path expression — the path is
/// *concretized*, not constrained, so the symbolic bytes keep every value they
/// could have had. Only the terminator assertion `strlen` itself contributes
/// (see [`null_exists_constraint`]) lands on the state.
///
/// Without this, one symbolic byte anywhere in the pathname bounced the whole
/// call to Python — it was fauxware's sole Python crossing, since fauxware
/// opens the *username* buffer it just read from stdin (angr-gorvf.13).
fn read_pathname(state: &mut RustSimState, addr: u64) -> Result<Vec<u8>, ProcedureError> {
    let bytes = match scan_for_null_symbolic(state, addr, MAX_PATH)? {
        ScanOutcome::AllConcrete { length } => {
            // `scan_for_null_symbolic` only reports `length == max` when it ran
            // the whole window without hitting a terminator (a found null is
            // always at an index `< max`). Python's `open` uses an unbounded
            // `strlen`, so a longer pathname is a real path there — returning
            // the 256-byte prefix would silently open a *different* file.
            // Error out so dispatch falls back, matching sibling
            // `read_cstring`/`scan_concrete_until_null` (angr-sqfj8.80).
            if length == MAX_PATH {
                return Err(ProcedureError::MaxIterations(MAX_PATH as usize));
            }
            let (buf, _) = scan_concrete_bounded(state, addr, length as usize, "pathname")?;
            return Ok(buf);
        }
        ScanOutcome::Symbolic { bytes } => bytes,
    };

    // Bytes before the first symbolic one were all concrete and non-NUL.
    let prefix_len = bytes[0].0 as usize;
    let (mut name, _) = scan_concrete_bounded(state, addr, prefix_len, "pathname")?;

    // The inline strlen Python performs asserts a terminator exists in the
    // window; mirror it so the two engines constrain the buffer identically.
    let pruning = {
        let ctx = state.solver().borrow();
        null_exists_constraint(&bytes, &ctx)
    };
    if let Some(c) = pruning {
        state.add_constraint(c);
    }

    let ctx = state.solver().borrow();
    for (_, byte) in &bytes {
        match ctx.eval(byte) {
            Some(0) => break,
            Some(v) => name.push(v as u8),
            None => {
                return Err(ProcedureError::SymbolicArgument(
                    "unsatisfiable pathname".to_string(),
                ));
            }
        }
    }
    Ok(name)
}

/// `_IO_FILE` size and fd offset per arch, mirroring
/// `cle.backends.externs.simdata.io_file.io_file_data_for_arch`.
/// Returns `(fd_offset, total_size)`. Single source of truth for the arch ->
/// `_fileno` offset mapping.
/// Arch names match what `Arch::name()` returns (only `"ARM"`/`"ARM64"` for the
/// ARM family — the `ARMEL`/`ARMHF`/`AARCH64` aliases never reach here).
pub(super) fn io_file_for_arch(name: &str) -> Option<(u64, u64)> {
    match name {
        "AMD64" => Some((112, 216)),
        "X86" => Some((56, 148)),
        "ARM" => Some((14, 84)),
        "ARM64" => Some((20, 152)),
        "MIPS32" => Some((56, 148)),
        "MIPS64" => Some((112, 216)),
        _ => None,
    }
}

const MAX_FOPEN_PATH_LEN: u64 = 256;
const MAX_FOPEN_MODE_LEN: u64 = 8;

/// Read a NUL-terminated string from memory up to `max_len` bytes. Concrete
/// only: a symbolic byte (or exhausting `max_len` without a null) errors and
/// falls back to Python.
fn read_cstring(
    state: &mut RustSimState,
    addr: u64,
    max_len: u64,
    name: &str,
) -> Result<Vec<u8>, ProcedureError> {
    scan_concrete_until_null(state, addr, max_len as usize, name)
}

/// Convert an fopen-style mode string (e.g. `"r"`, `"w+b"`, `"rb+"`) to FdFlags.
/// Returns None for unrecognized modes (caller falls back to Python).
///
/// glibc semantics: the first character selects the base access mode
/// (`r`/`w`/`a`); the remaining flag characters (`+`, `b`, `t`, `c`, `e`, `m`,
/// `x`) may appear in **any order**. Only `+` upgrades the access mode to
/// read+write — the rest are buffering/sharing/exclusivity hints that do not
/// affect the FdFlags mapping. Parsing the trailing flags positionally (only
/// popping a trailing `b`/`t`) silently missed valid orderings like `"rb+"`.
fn parse_fopen_mode(mode: &[u8]) -> Option<FdFlags> {
    let first = *mode.first()?;
    // Flag characters glibc accepts after the base mode. Anything outside this
    // set is genuinely unrecognized, so defer to Python rather than guess.
    const FLAG_CHARS: &[u8] = b"+btcexm";
    if mode[1..].iter().any(|c| !FLAG_CHARS.contains(c)) {
        return None;
    }
    let read_write = mode[1..].contains(&b'+');
    match (first, read_write) {
        (b'r', false) => Some(FdFlags::ReadOnly),
        (b'r', true) => Some(FdFlags::ReadWrite),
        (b'w', false) => Some(FdFlags::WriteOnly),
        (b'w', true) => Some(FdFlags::ReadWrite),
        (b'a', false) => Some(FdFlags::WriteOnly),
        (b'a', true) => Some(FdFlags::ReadWrite),
        _ => None,
    }
}

/// Read a 32-bit fd from a FILE struct on the given arch. Returns the signed fd
/// (so -1 sentinels are preserved) and any symbolic / arch-resolution errors.
pub(crate) fn read_fileno(state: &RustSimState, file_ptr: u64) -> Result<i32, ProcedureError> {
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
        // Read the pathname (max 256 bytes), concretizing symbolic bytes as
        // Python's open does.
        let name = read_pathname(state, pathname_addr)?;

        let pathname = String::from_utf8_lossy(&name).to_string();
        let fd_flags = FdFlags::from_posix(flags as u32);
        let fd = state.file_system().open(pathname, fd_flags);

        Ok(Some(arch_word(state, u64::from(fd))))
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
        let ret = if success { 0 } else { -1i64 as u64 };
        Ok(Some(arch_word(state, ret)))
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
        match state
            .file_system()
            .seek(fd as u32, offset as i64, whence as u32)
        {
            Some(new_pos) => Ok(Some(arch_word(state, new_pos))),
            None => Ok(Some(arch_word(state, -1i64 as u64))),
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
        match state.file_system().dup(oldfd as u32) {
            Some(newfd) => Ok(Some(arch_word(state, u64::from(newfd)))),
            None => Ok(Some(arch_word(state, -1i64 as u64))),
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
        match state.file_system().dup2(oldfd as u32, newfd as u32) {
            Some(fd) => Ok(Some(arch_word(state, u64::from(fd)))),
            None => Ok(Some(arch_word(state, -1i64 as u64))),
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
        let little = state.is_little_endian();
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

        Ok(Some(arch_word(state, 0u64)))
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

        Ok(Some(arch_word(state, file_ptr)))
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

        if fd < 0 || !state.file_system_ref().is_open(fd as u32) {
            return Ok(Some(arch_word(state, 0u64)));
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

        Ok(Some(arch_word(state, file_ptr)))
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

        if fd < 0 {
            return Ok(Some(arch_word(state, -1i64 as u64)));
        }
        let ret = if state.file_system().close(fd as u32) {
            0
        } else {
            -1i64 as u64
        };
        Ok(Some(arch_word(state, ret)))
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
    aliases = ["fseeko"],
    call |state| {
        let offset = offset_raw as i64;
        let whence = whence_raw as u32;
        let fd = read_fileno(state, file_ptr)?;
        if fd < 0 {
            return Ok(Some(arch_word(state, -1i64 as u64)));
        }
        match state.file_system().seek(fd as u32, offset, whence) {
            Some(_) => Ok(Some(arch_word(state, 0u64))),
            None => Ok(Some(arch_word(state, -1i64 as u64))),
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
    aliases = ["ftello"],
    call |state| {
        let fd = read_fileno(state, file_ptr)?;

        if fd < 0 {
            return Ok(Some(arch_word(state, -1i64 as u64)));
        }
        match state.file_system_ref().fd_info(fd as u32) {
            Some((_, pos, _, _, _)) => Ok(Some(arch_word(state, pos))),
            None => Ok(Some(arch_word(state, -1i64 as u64))),
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

test_submod!("fileops_tests.rs" => fileops_tests);
