//! `open` / `openat` / `close` — the FD-allocating handlers (angr-k3ol.1).
//!
//!
//! `open`, `openat`, `close` allocate / release FDs against
//! `RustSimState::file_system()`, mirroring the existing
//! `procedures/fileops::NativeOpen` / `NativeClose` libc procs (which
//! are already wired through `state.file_system()` and therefore
//! diverge from Python `state.posix.fd` in the same way — see the
//! `syscall-vs-procedure-dispatch` bd memory). The Python proc
//! `procedures/posix/open.py` returns `-1` when `state.fs.get(path)`
//! is `None` and creation flags are absent; we always allocate a fresh
//! fd since the Rust-side `FileSystem` does not mirror Python's
//! `state.fs`. This is the same trade-off the libc procedure made.

use super::{NEG_ONE, dirfd_allows, read_path};
use crate::state::{FdFlags, RustSimState};
use crate::symbolic::RustBV;
use crate::syscalls::{NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg};

/// `open(pathname, flags, mode) → fd` — allocate a fresh fd in the
/// Rust `FileSystem` keyed by `pathname`. Mirrors
/// `procedures/fileops::NativeOpen`; the only divergence from
/// `procedures/posix/open.py` is that we do not consult Python's
/// `state.fs` map (which is not mirrored into Rust state), so we never
/// return `-1` for "file doesn't exist". Empty path → `-1` (same as
/// the Python proc).
pub(crate) struct NativeOpenSyscall;

impl NativeSyscall for NativeOpenSyscall {
    fn name(&self) -> &'static str {
        "open"
    }

    fn num_args(&self) -> usize {
        3
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        let pathname_addr = extract_concrete_arg(&args[0], "open pathname")?;
        let flags = extract_concrete_arg(&args[1], "open flags")?;
        // args[2] = mode — irrelevant in Rust's FileSystem model.

        let path = read_path(state, pathname_addr, "open")?;
        if path.is_empty() {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }
        // `None` = the fd space is exhausted (angr-03vl4.88); bounce to Python
        // rather than wrapping `next_fd` and handing out stdin as a fresh file.
        let fd = state
            .file_system()
            .open(path, FdFlags::from_posix(flags as u32))
            .ok_or_else(|| SyscallError::Other("open: fd space exhausted".to_string()))?;
        Ok(SyscallOutcome::Continue { ret: fd as u64 })
    }
}

/// `openat(dirfd, pathname, flags, mode) → fd` — like `open`, plus
/// the `dirfd` arg for relative paths. We only handle absolute paths
/// and the `AT_FDCWD` sentinel (mirroring
/// `procedures/linux_kernel/openat.py`, which returns `-1` for any
/// other dirfd). The relative-path-from-dirfd case is not modeled in
/// Python either.
pub(crate) struct NativeOpenatSyscall;

impl NativeSyscall for NativeOpenatSyscall {
    fn name(&self) -> &'static str {
        "openat"
    }

    fn num_args(&self) -> usize {
        4
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        let dirfd = extract_concrete_arg(&args[0], "openat dirfd")?;
        let pathname_addr = extract_concrete_arg(&args[1], "openat pathname")?;
        let flags = extract_concrete_arg(&args[2], "openat flags")?;
        // args[3] = mode — irrelevant in Rust's FileSystem model.

        let path = read_path(state, pathname_addr, "openat")?;
        if path.is_empty() {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }
        if !dirfd_allows(&path, dirfd) {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }
        // `None` = the fd space is exhausted (angr-03vl4.88) — see `NativeOpenSyscall`.
        let fd = state
            .file_system()
            .open(path, FdFlags::from_posix(flags as u32))
            .ok_or_else(|| SyscallError::Other("openat: fd space exhausted".to_string()))?;
        Ok(SyscallOutcome::Continue { ret: fd as u64 })
    }
}

/// `close(fd) → 0 | -1` — mark `fd` closed in Rust's `FileSystem`.
/// Returns `-1` if the fd was never opened by Rust (mirrors
/// `state.posix.close` returning falsy).
pub(crate) struct NativeCloseSyscall;

impl NativeSyscall for NativeCloseSyscall {
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
    ) -> Result<SyscallOutcome, SyscallError> {
        let fd = extract_concrete_arg(&args[0], "close fd")?;
        let ret = if state.file_system().close(fd as u32) {
            0
        } else {
            NEG_ONE
        };
        Ok(SyscallOutcome::Continue { ret })
    }
}
