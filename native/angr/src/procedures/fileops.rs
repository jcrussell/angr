//! Native fd-table primitives: open, close, lseek, dup, dup2, pipe.
//!
//! These procedures track file descriptor state in the FileSystem.
//! open() allocates a new fd, close() marks it closed, lseek() adjusts position.
//! dup/dup2 duplicate a fd; pipe creates a (read_fd, write_fd) pair.
//!
//! The `FILE *`-based stdio primitives layered on top of these
//! (fopen/fdopen/fclose/fseek/ftell/rewind, plus the shared `_fileno`
//! resolution helpers) live in the sibling [`stream_ops`](super::stream_ops)
//! module.
//!
//! Only handles concrete arguments; symbolic arguments fall back to Python.

use super::strings::{
    MAX_PATH_SCAN as MAX_PATH, ScanOutcome, null_exists_constraint, scan_concrete_bounded,
    scan_for_null_symbolic,
};
use super::{ProcedureError, arch_word};
use crate::state::{FdFlags, RustSimState};
use crate::symbolic::RustBV;

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
            // Error out so dispatch falls back, matching
            // `stream_ops::read_cstring_strict` / `scan_concrete_until_null`
            // (angr-sqfj8.80).
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
        let append = FdFlags::posix_has_append(flags as u32);
        // `None` = the fd space is exhausted (angr-03vl4.88); bounce to Python
        // rather than wrapping `next_fd` and handing out stdin as a fresh file.
        let fd = state
            .file_system()
            .open_with_append(pathname, fd_flags, append)
            .ok_or_else(|| ProcedureError::Other("open: fd space exhausted".to_string()))?;

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
    ///
    /// An `oldfd` that does not fit in a `u32` is "not open" rather than
    /// truncated: `oldfd as u32` would alias an unrelated descriptor
    /// (`0x1_0000_0000` → 0 → stdin) — angr-03vl4.52.
    name = "dup",
    struct = NativeDup,
    args = [oldfd: concrete],
    call |state| {
        let duped = match u32::try_from(oldfd) {
            Ok(oldfd) => state.file_system().dup(oldfd),
            Err(_) => None,
        };
        match duped {
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
    /// Returns `newfd` on success, or -1 if `oldfd` is not open, if either fd
    /// does not fit in a `u32` (truncating would alias an unrelated
    /// descriptor), or if `newfd` is at/above
    /// [`MAX_FD`](crate::state::MAX_FD) — angr-03vl4.52. Unlike the `dup2`
    /// *syscall* there is no layer above this one applying that cap, so a
    /// guest `dup2(fd, 0xFFFFFFFF)` used to reach the unbounded `next_fd`
    /// bump inside [`FileSystem::dup2`](crate::state::FileSystem::dup2).
    name = "dup2",
    struct = NativeDup2,
    args = [oldfd: concrete, newfd: concrete],
    call |state| {
        let dup2ed = match (u32::try_from(oldfd), u32::try_from(newfd)) {
            (Ok(oldfd), Ok(newfd)) => state.file_system().dup2(oldfd, newfd),
            _ => None,
        };
        match dup2ed {
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
        // `None` = the fd space cannot supply a consecutive pair (angr-03vl4.55);
        // bounce to Python rather than aliasing two ends onto one fd.
        let (read_fd, write_fd) = state
            .file_system()
            .pipe()
            .ok_or_else(|| ProcedureError::Other("pipe: fd space exhausted".to_string()))?;

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

test_submod!("fileops_tests.rs" => fileops_tests);
