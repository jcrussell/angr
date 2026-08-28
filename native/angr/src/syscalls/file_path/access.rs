//! `access` / `faccessat` syscall handlers.
//!
//!
//! `access(pathname, mode) → 0 | -1` queries
//! `RustSimState::file_system().is_path_known(path)`, which mirrors the
//! Python proc `procedures/linux_kernel/access.py::run` (returns `-1`
//! when `state.fs.get(path)` is `None`, else `0`). The path set is
//! populated by `open` / `open_with_content` calls — pre-populated
//! Python `state.fs` entries are NOT mirrored unless an explicit
//! `register_known_path` call is made on the Rust state. Same
//! trade-off as `open` / `openat` here.
//!
//!
//! `faccessat(dfd, pathname, mode) → 0 | -1` — clone of `access` with
//! dirfd handling. Absolute paths and `AT_FDCWD` query
//! `FileSystem::is_path_known`. Relative paths with any other dirfd
//! return `-1` (we do not model directory fds — matches `openat`'s
//! policy). The `mode` arg is ignored. This was previously a stub.

// faccessat(dfd, filename, mode) → long — see NativeFaccessatSyscall impl below.

use super::{NEG_ONE, dirfd_allows, read_path};
use crate::state::RustSimState;
use crate::symbolic::RustBV;
use crate::syscalls::{NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg};

/// `access(pathname, mode) → 0 | -1` — return `0` if the path has been
/// registered as known (via prior `open` / `openat` or
/// `FileSystem::register_known_path`), `-1` otherwise. Mirrors
/// `procedures/linux_kernel/access.py::run`. The `mode` arg
/// (`F_OK` / `R_OK` / ...) is ignored — the Python proc also ignores it.
/// Empty path → `-1` (defensive: Python would also miss in `state.fs`).
pub(crate) struct NativeAccessSyscall;

impl NativeSyscall for NativeAccessSyscall {
    fn name(&self) -> &'static str {
        "access"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        let pathname_addr = extract_concrete_arg(&args[0], "access pathname")?;
        // args[1] = mode — Python proc ignores it; so do we.

        let path = read_path(state, pathname_addr, "access")?;
        if path.is_empty() {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }
        let ret = if state.file_system_ref().is_path_known(&path) {
            0
        } else {
            NEG_ONE
        };
        Ok(SyscallOutcome::Continue { ret })
    }
}

/// `faccessat(dirfd, pathname, mode) → 0 | -1` — clone of `access` with
/// dirfd handling. Absolute paths and `AT_FDCWD` query
/// `FileSystem::is_path_known`. Relative paths with any other dirfd
/// return `-1` (we do not model directory fds — matches `openat`'s
/// policy). The `mode` arg is ignored — Python's stub also ignores it.
/// Empty path → `-1`.
pub(crate) struct NativeFaccessatSyscall;

impl NativeSyscall for NativeFaccessatSyscall {
    fn name(&self) -> &'static str {
        "faccessat"
    }

    fn num_args(&self) -> usize {
        3
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        let dirfd = extract_concrete_arg(&args[0], "faccessat dirfd")?;
        let pathname_addr = extract_concrete_arg(&args[1], "faccessat pathname")?;
        // args[2] = mode — ignored, mirroring NativeAccessSyscall.

        let path = read_path(state, pathname_addr, "faccessat")?;
        if path.is_empty() {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }
        if !dirfd_allows(&path, dirfd) {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }
        let ret = if state.file_system_ref().is_path_known(&path) {
            0
        } else {
            NEG_ONE
        };
        Ok(SyscallOutcome::Continue { ret })
    }
}
