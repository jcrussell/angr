//! `readlink` / `readlinkat` syscall handlers (angr-wv38).
//!
//! `readlink(pathname, buf, bufsiz) → ssize_t` and
//! `readlinkat(dirfd, pathname, buf, bufsiz) → ssize_t` consult the
//! `FileSystem` symlink table (angr-11djq.6.2):
//!
//! * If `pathname` is a registered symlink (via `FileSystem::add_symlink`)
//!   → write `min(target_len, bufsiz)` raw target bytes into `buf` (no
//!   NUL terminator, matching `readlink(2)`) and return that count.
//! * Otherwise → `-1`, buffer untouched (the path is "not a symlink"
//!   `EINVAL` / "no such file" `ENOENT`).
//!
//! The symlink table is empty by default, so with no `add_symlink` call
//! the handlers return `-1` for every path — byte-identical to the prior
//! always-`-1` behavior. That default is strictly more accurate than the
//! original fresh-symbolic stub return, which would let solver-permitted
//! paths take the "positive return" branch and explore code reading
//! non-existent symlink data. The known regression risk (binaries that
//! branch on a positive return from `/proc/self/exe`-discovery code) was
//! cleared by the angr-examples bench sweep. Python's symbolic-stub
//! `readlink` has no symlink model, so this is a deliberate accuracy
//! divergence (same trade-off as `access` / `stat`).
//!
//! The handlers still read `pathname` into a Rust `String` before
//! returning, mirroring `NativeAccessSyscall` / `NativeFaccessatSyscall`:
//! a symbolic path byte routes through `SyscallError::SymbolicArgument`
//! and falls back to the Python `syscall_stub`. `readlinkat` also reads
//! `dirfd` and applies the `openat` policy (absolute / `AT_FDCWD` →
//! resolve; relative + other dirfd → `-1`) even though the dirfd
//! ultimately does not affect the return — keeps the handler shape
//! symmetric with `NativeFaccessatSyscall` / `NativeOpenatSyscall` and
//! falls back on symbolic dirfd.
//!
//! Both work on every arch the syscalls are registered on:
//! `readlink` (AMD64 89, X86 85, ARM EABI 85, MIPS32 4085, MIPS64 5087)
//! and `readlinkat` (AMD64 267, X86 305, ARM EABI 332, ARM64 78,
//! MIPS32 4298, MIPS64 5257). No arch-specific layout to write.

// readlink(path, buf, bufsiz) → long — see NativeReadlinkSyscall impl below.
// readlinkat(dfd, path, buf, bufsiz) → long — see NativeReadlinkatSyscall impl below.

use super::{NEG_ONE, dirfd_allows, read_path};
use crate::state::RustSimState;
use crate::symbolic::RustBV;
use crate::syscalls::{NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg};

/// `readlink(pathname, buf, bufsiz) → ssize_t` — resolves `pathname`
/// against the `FileSystem` symlink table (empty by default → `-1`).
/// On a hit, writes `min(target_len, bufsiz)` raw target bytes to `buf`
/// and returns that count (see `write_symlink_target`). `pathname` is
/// read into a Rust `String` first, so a symbolic path byte routes
/// through `SyscallError::SymbolicArgument` and falls back to the Python
/// `syscall_stub` (matches the `NativeAccessSyscall` pattern). An empty
/// `pathname` short-circuits to `-1` (matches `readlinkat`).
pub(crate) struct NativeReadlinkSyscall;

impl NativeSyscall for NativeReadlinkSyscall {
    fn name(&self) -> &'static str {
        "readlink"
    }

    fn num_args(&self) -> usize {
        3
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        let pathname_addr = extract_concrete_arg(&args[0], "readlink pathname")?;
        let path = read_path(state, pathname_addr, "readlink")?;
        if path.is_empty() {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }
        write_symlink_target(state, &path, &args[1], &args[2], "readlink")
    }
}

/// `readlinkat(dirfd, pathname, buf, bufsiz) → ssize_t` — clone of
/// `readlink` with `openat`-style dirfd handling. Absolute paths and
/// `AT_FDCWD` resolve via `read_path`; relative paths with any other
/// dirfd short-circuit to `-1` without touching the path (mirrors
/// `NativeOpenatSyscall` / `NativeFaccessatSyscall`). On an absolute /
/// `AT_FDCWD` path the symlink-table lookup runs via
/// `write_symlink_target` (empty table → `-1`). Falls back to Python on
/// symbolic dirfd.
pub(crate) struct NativeReadlinkatSyscall;

impl NativeSyscall for NativeReadlinkatSyscall {
    fn name(&self) -> &'static str {
        "readlinkat"
    }

    fn num_args(&self) -> usize {
        4
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        let dirfd = extract_concrete_arg(&args[0], "readlinkat dirfd")?;
        let pathname_addr = extract_concrete_arg(&args[1], "readlinkat pathname")?;

        let path = read_path(state, pathname_addr, "readlinkat")?;
        if path.is_empty() {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }
        if !dirfd_allows(&path, dirfd) {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }
        write_symlink_target(state, &path, &args[2], &args[3], "readlinkat")
    }
}

/// Shared `readlink` / `readlinkat` tail: if `path` is a registered
/// symlink, write `min(target_len, bufsiz)` raw target bytes (no NUL
/// terminator, matching `readlink(2)`) into `buf` and return that count;
/// otherwise return `-1` (the path is "not a symlink" / does not exist —
/// the empty-table default). `buf` / `bufsiz` are only read on a symlink
/// hit, so a symbolic `buf` / `bufsiz` on a non-symlink path still
/// returns `-1` without falling back to Python (a symbolic operand on a
/// genuine symlink routes through `SymbolicArgument` → Python).
fn write_symlink_target(
    state: &mut RustSimState,
    path: &str,
    buf_arg: &RustBV,
    bufsiz_arg: &RustBV,
    label: &str,
) -> Result<SyscallOutcome, SyscallError> {
    let target = match state.file_system_ref().readlink_target(path) {
        Some(t) => t.to_vec(),
        None => return Ok(SyscallOutcome::Continue { ret: NEG_ONE }),
    };
    let buf = extract_concrete_arg(buf_arg, &format!("{label} buf"))?;
    let bufsiz = extract_concrete_arg(bufsiz_arg, &format!("{label} bufsiz"))?;
    let n = (target.len() as u64).min(bufsiz);
    for (i, b) in target.iter().take(n as usize).enumerate() {
        state.memory_store(buf.wrapping_add(i as u64), RustBV::concrete(*b as u128, 8))?;
    }
    Ok(SyscallOutcome::Continue { ret: n })
}
