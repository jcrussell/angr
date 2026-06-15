//! File-path syscall handlers.
//!
//! ## `readlink` / `readlinkat` (angr-wv38)
//!
//! `readlink(pathname, buf, bufsiz) → ssize_t` and
//! `readlinkat(dirfd, pathname, buf, bufsiz) → ssize_t` always return
//! `-1`. The Rust `FileSystem` model has no symlinks, so:
//!
//! * For any known path: `-1` with `errno = EINVAL` (not a symlink).
//! * For any unknown path: `-1` with `errno = ENOENT` (no such file).
//!
//! Either way the buffer is left untouched (real Linux only writes on a
//! positive return). This is strictly more accurate than the previous
//! fresh-symbolic stub return — the previous stub would let solver-
//! permitted paths take the "positive return" branch and explore code
//! reading non-existent symlink data. The known regression risk
//! (binaries that branch on a positive return from `/proc/self/exe`-
//! discovery code) was cleared by the angr-examples bench sweep.
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
//!
//! ## `faccessat` (angr-6009)
//!
//! `faccessat(dfd, pathname, mode) → 0 | -1` — clone of `access` with
//! dirfd handling. Absolute paths and `AT_FDCWD` query
//! `FileSystem::is_path_known`. Relative paths with any other dirfd
//! return `-1` (we do not model directory fds — matches `openat`'s
//! policy). The `mode` arg is ignored. This was previously a stub.
//!
//! ## FD-allocating handlers (angr-k3ol.1)
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
//!
//! ## `access` (angr-k3ol.2)
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
//! ## `fstat` (angr-k3ol.3)
//!
//! `fstat(fd, statbuf) → 0 | -1` reads `(name, _, _, content_len, _)`
//! from `FileSystem::fd_info(fd)` and writes a per-arch `struct stat`
//! to `statbuf`. Mirrors `procedures/linux_kernel/fstat.py`, which
//! delegates to `state.posix.fstat_with_result`. The Rust handler
//! diverges in two intentional ways:
//!
//! * `st_mode` is written as the concrete constant `S_IFREG | 0o755`
//!   instead of a fresh symbolic BVS (Python's
//!   `fstat_with_result` mints `BVS("st_mode", 32)`). The concrete
//!   value matches a regular file's permissions and avoids spawning
//!   a symbolic the binary will likely just compare against
//!   `S_IFREG`. Anything that depends on a symbolic mode must use
//!   the Python proc.
//! * `st_size` comes from `content_len` (whatever was last written
//!   to the fd's backing buffer) — concrete, not symbolic.
//!
//! Per-arch struct layouts cover AMD64 and ARM64 (the two 64-bit
//! arches with a sys-call 5 / 80 `fstat` and a Python
//! implementation). x86 / ARM / MIPS32 carry the legacy 32-bit
//! `struct stat`; the Python proc itself raises on those — there is
//! no benefit to a Rust copy, so fstat for those arches continues to
//! fall through to the Python error path.
//!
//! Unknown fd → `-1` with no buffer write (matches
//! `fstat_with_result`'s `result = -1` branch).
//!
//! ## `stat` (angr-k3ol.4)
//!
//! `stat(pathname, statbuf) → 0 | -1` resolves `pathname` via
//! `read_path`, returns `-1` for empty / unknown paths (via
//! `FileSystem::is_path_known`), and otherwise writes a per-arch
//! `struct stat` using the same `write_amd64_stat` /
//! `write_aarch64_stat` helpers used by `fstat`. The size field is
//! sourced from `FileSystem::content_size_for_path(path)` (largest
//! `content_len` across any fd that opened the path) — `0` if the
//! path was registered via `register_known_path` without a content
//! payload. This diverges from `procedures/linux_kernel/stat.py`,
//! which opens a temp fd, calls `fstat`, then closes. The Rust path
//! never mutates the fd table, so the next-fd counter is stable
//! across stat queries. Arch coverage: AMD64 only — ARM64 has no
//! legacy `stat` syscall (only `newfstatat` 79, already a stub).
//! Other arches (x86 / ARM EABI / MIPS32 carry the legacy 32-bit
//! `struct stat`) fall back to Python's error path, matching the
//! `fstat` policy.
//!
//! Symbolic pathname pointer / pathname byte → `SymbolicArgument`
//! (dispatcher falls back to Python). Unmapped statbuf surfaces a
//! `MemoryError` via the `?` conversion. Unsupported arch returns
//! `Other("unsupported arch …")` — checked FIRST before reading the
//! path, so a state on a non-AMD64 arch never even attempts to
//! traverse memory.
//!
//! ## `lstat` / `newfstatat` (angr-poao)
//!
//! `lstat(pathname, statbuf) → 0 | -1` and
//! `newfstatat(dirfd, pathname, statbuf, flag) → 0 | -1` clone the
//! `stat` semantics with two adjustments:
//!
//! * `lstat` would normally diverge on symbolic links — but the
//!   `FileSystem` model has no symlinks (open / openat never produce
//!   one), so it collapses to the same write as `stat`. AMD64 only,
//!   matching the `stat` policy.
//! * `newfstatat` adds `openat`-style dirfd handling: absolute paths
//!   and `AT_FDCWD` resolve via `FileSystem`; relative paths with any
//!   other dirfd return `-1` (we do not model directory fds). The
//!   `flag` arg (incl. `AT_EMPTY_PATH` 0x1000, which would route to
//!   `NativeFstatSyscall(dirfd)`) is ignored — deferred. Arch coverage
//!   is AMD64 + ARM64 (both have a 64-bit `struct stat` and a Python
//!   `fstat.py` reference); `newfstatat` is in fact the only stat-shaped
//!   syscall on ARM64's asm-generic ABI.
//!
//! Both reuse `write_amd64_stat` / `write_aarch64_stat` from `fstat`
//! and `FileSystem::content_size_for_path` from `stat`. Arch-check
//! happens FIRST (before any memory read), matching the `stat` policy.

use super::{NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg};
use crate::state::{FdFlags, RustSimState};
use crate::symbolic::RustBV;

// readlink(path, buf, bufsiz) → long — see NativeReadlinkSyscall impl below.
// readlinkat(dfd, path, buf, bufsiz) → long — see NativeReadlinkatSyscall impl below.
// faccessat(dfd, filename, mode) → long — see NativeFaccessatSyscall impl below.
// lstat(pathname, statbuf) → long — see NativeLstatSyscall impl below.
// newfstatat(dfd, filename, statbuf, flag) → long — see NativeNewfstatatSyscall impl below.

/// Upper bound on the NUL-terminated path we will read from memory.
/// Matches `procedures/fileops.rs::MAX_FOPEN_PATH_LEN` (256 bytes).
const MAX_PATH_LEN: u64 = 256;

/// `AT_FDCWD` in unsigned 32-bit form (-100 reinterpreted). Linux's
/// `openat(2)` treats this as "use the current working directory" for
/// relative paths. `procedures/linux_kernel/openat.py` also matches
/// against this exact unsigned value.
const AT_FDCWD_UNSIGNED: u64 = 4_294_967_196;

/// `-1` (as `u64`) — kernel ABI failure return for `open` / `openat` /
/// `close` mirroring `procedures/posix/open.py::run` (`return -1`).
/// The dispatcher truncates to `arch().bits()` when writing the return
/// register.
const NEG_ONE: u64 = u64::MAX;

/// Read a NUL-terminated path from memory at `addr`, up to
/// `MAX_PATH_LEN`. Returns `SymbolicArgument` on the first symbolic
/// byte (the syscall then falls back to Python). Errors out with
/// `Other` if no NUL is seen within the limit.
fn read_path(state: &RustSimState, addr: u64, label: &str) -> Result<String, SyscallError> {
    let mut bytes: Vec<u8> = Vec::new();
    for i in 0..MAX_PATH_LEN {
        let bv = state.memory_load(addr.wrapping_add(i), 1)?;
        let v = bv
            .as_u64()
            .ok_or_else(|| SyscallError::SymbolicArgument(format!("{label} path byte")))?;
        if v == 0 {
            return Ok(String::from_utf8_lossy(&bytes).to_string());
        }
        bytes.push(v as u8);
    }
    Err(SyscallError::Other(format!(
        "{label} path not NUL-terminated within {MAX_PATH_LEN} bytes"
    )))
}

/// `open(pathname, flags, mode) → fd` — allocate a fresh fd in the
/// Rust `FileSystem` keyed by `pathname`. Mirrors
/// `procedures/fileops::NativeOpen`; the only divergence from
/// `procedures/posix/open.py` is that we do not consult Python's
/// `state.fs` map (which is not mirrored into Rust state), so we never
/// return `-1` for "file doesn't exist". Empty path → `-1` (same as
/// the Python proc).
pub struct NativeOpenSyscall;

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
        let _ = args.get(2);

        let path = read_path(state, pathname_addr, "open")?;
        if path.is_empty() {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }
        let fd = state
            .file_system()
            .open(path, FdFlags::from_posix(flags as u32));
        Ok(SyscallOutcome::Continue { ret: fd as u64 })
    }
}

/// `openat(dirfd, pathname, flags, mode) → fd` — like `open`, plus
/// the `dirfd` arg for relative paths. We only handle absolute paths
/// and the `AT_FDCWD` sentinel (mirroring
/// `procedures/linux_kernel/openat.py`, which returns `-1` for any
/// other dirfd). The relative-path-from-dirfd case is not modeled in
/// Python either.
pub struct NativeOpenatSyscall;

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
        let _ = args.get(3);

        let path = read_path(state, pathname_addr, "openat")?;
        if path.is_empty() {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }
        let absolute = path.starts_with('/');
        if !absolute && dirfd != AT_FDCWD_UNSIGNED {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }
        let fd = state
            .file_system()
            .open(path, FdFlags::from_posix(flags as u32));
        Ok(SyscallOutcome::Continue { ret: fd as u64 })
    }
}

/// `close(fd) → 0 | -1` — mark `fd` closed in Rust's `FileSystem`.
/// Returns `-1` if the fd was never opened by Rust (mirrors
/// `state.posix.close` returning falsy).
pub struct NativeCloseSyscall;

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

/// `access(pathname, mode) → 0 | -1` — return `0` if the path has been
/// registered as known (via prior `open` / `openat` or
/// `FileSystem::register_known_path`), `-1` otherwise. Mirrors
/// `procedures/linux_kernel/access.py::run`. The `mode` arg
/// (`F_OK` / `R_OK` / ...) is ignored — the Python proc also ignores it.
/// Empty path → `-1` (defensive: Python would also miss in `state.fs`).
pub struct NativeAccessSyscall;

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
        let _ = args.get(1);

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
pub struct NativeFaccessatSyscall;

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
        let _ = args.get(2);

        let path = read_path(state, pathname_addr, "faccessat")?;
        if path.is_empty() {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }
        let absolute = path.starts_with('/');
        if !absolute && dirfd != AT_FDCWD_UNSIGNED {
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

/// `readlink(pathname, buf, bufsiz) → -1` — always returns `-1` because
/// the Rust `FileSystem` model has no symlinks (every path is "not a
/// symlink" → `EINVAL`, every unknown path → `ENOENT`). The buffer is
/// left untouched. `pathname` is still read into a Rust `String` so
/// that a symbolic path byte routes through `SyscallError::SymbolicArgument`
/// and falls back to the Python `syscall_stub` (matches the
/// `NativeAccessSyscall` pattern). `buf` / `bufsiz` are not validated:
/// they would only matter on a positive return, which never happens
/// here.
pub struct NativeReadlinkSyscall;

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
        // args[1] = buf, args[2] = bufsiz — ignored (we return -1 and
        // never write the buffer).
        let _ = (args.get(1), args.get(2));

        let _path = read_path(state, pathname_addr, "readlink")?;
        Ok(SyscallOutcome::Continue { ret: NEG_ONE })
    }
}

/// `readlinkat(dirfd, pathname, buf, bufsiz) → -1` — clone of
/// `readlink` with `openat`-style dirfd handling. Absolute paths and
/// `AT_FDCWD` resolve via `read_path`; relative paths with any other
/// dirfd short-circuit to `-1` without touching the path (mirrors
/// `NativeOpenatSyscall` / `NativeFaccessatSyscall`). The end result
/// is `-1` either way — the dirfd branch only exists so the handler
/// shape stays symmetric with the rest of the `*at` family and falls
/// back to Python on symbolic dirfd.
pub struct NativeReadlinkatSyscall;

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
        // args[2] = buf, args[3] = bufsiz — ignored (we return -1).
        let _ = (args.get(2), args.get(3));

        let path = read_path(state, pathname_addr, "readlinkat")?;
        if path.is_empty() {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }
        let absolute = path.starts_with('/');
        if !absolute && dirfd != AT_FDCWD_UNSIGNED {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }
        Ok(SyscallOutcome::Continue { ret: NEG_ONE })
    }
}

/// Concrete defaults for the `struct stat` fields. `fstat_with_result`
/// in Python returns a symbolic `st_mode` and `st_size` plus a
/// `st_blksize` of `0x400`; the Rust handler swaps `st_mode` for
/// `S_IFREG | 0o755` (regular file, rwxr-xr-x) and `st_size` for the
/// concrete length of the fd's backing buffer (`content_len`). Other
/// fields stay zero, matching the Python defaults.
const S_IFREG_0755: u64 = 0o100_755;
const ST_BLKSIZE: u64 = 0x400;

/// AMD64 `struct stat` layout — total 0x90 bytes. Mirrors
/// `angr/procedures/linux_kernel/fstat.py::_store_amd64`. Writes are
/// arch-LE per `RustSimState::memory.endness`.
///
/// Field widths (offsets cumulative):
/// `dev` u64, `ino` u64, `nlink` u64, `mode` u32, `uid` u32,
/// `gid` u32, pad u32, `rdev` u64, `size` u64, `blksize` u64,
/// `blocks` u64, `atime+nsec` u64×2, `mtime+nsec` u64×2,
/// `ctime+nsec` u64×2, pad u64×3.
fn write_amd64_stat(state: &mut RustSimState, buf: u64, size: u64) -> Result<(), SyscallError> {
    let store_u64 = |state: &mut RustSimState, off: u64, val: u64| -> Result<(), SyscallError> {
        state.memory_store(buf + off, RustBV::concrete(val as u128, 64))?;
        Ok(())
    };
    let store_u32 = |state: &mut RustSimState, off: u64, val: u32| -> Result<(), SyscallError> {
        state.memory_store(buf + off, RustBV::concrete(val as u128, 32))?;
        Ok(())
    };

    store_u64(state, 0x00, 0)?; // st_dev
    store_u64(state, 0x08, 0)?; // st_ino
    store_u64(state, 0x10, 0)?; // st_nlink
    store_u32(state, 0x18, S_IFREG_0755 as u32)?; // st_mode
    store_u32(state, 0x1C, 0)?; // st_uid
    store_u32(state, 0x20, 0)?; // st_gid
    store_u32(state, 0x24, 0)?; // pad
    store_u64(state, 0x28, 0)?; // st_rdev
    store_u64(state, 0x30, size)?; // st_size
    store_u64(state, 0x38, ST_BLKSIZE)?; // st_blksize
    store_u64(state, 0x40, 0)?; // st_blocks
    store_u64(state, 0x48, 0)?; // st_atime
    store_u64(state, 0x50, 0)?; // st_atimensec
    store_u64(state, 0x58, 0)?; // st_mtime
    store_u64(state, 0x60, 0)?; // st_mtimensec
    store_u64(state, 0x68, 0)?; // st_ctime
    store_u64(state, 0x70, 0)?; // st_ctimensec
    store_u64(state, 0x78, 0)?; // pad
    store_u64(state, 0x80, 0)?; // pad
    store_u64(state, 0x88, 0)?; // pad
    Ok(())
}

/// AArch64 `struct stat` layout — total 0x80 bytes. Mirrors
/// `_store_aarch64` (note: `nlink` is u32 here, `blksize` is u32, and
/// the field order around mode/nlink/uid/gid differs from AMD64).
fn write_aarch64_stat(state: &mut RustSimState, buf: u64, size: u64) -> Result<(), SyscallError> {
    let store_u64 = |state: &mut RustSimState, off: u64, val: u64| -> Result<(), SyscallError> {
        state.memory_store(buf + off, RustBV::concrete(val as u128, 64))?;
        Ok(())
    };
    let store_u32 = |state: &mut RustSimState, off: u64, val: u32| -> Result<(), SyscallError> {
        state.memory_store(buf + off, RustBV::concrete(val as u128, 32))?;
        Ok(())
    };

    store_u64(state, 0x00, 0)?; // st_dev
    store_u64(state, 0x08, 0)?; // st_ino
    store_u32(state, 0x10, S_IFREG_0755 as u32)?; // st_mode
    store_u32(state, 0x14, 0)?; // st_nlink
    store_u32(state, 0x18, 0)?; // st_uid
    store_u32(state, 0x1C, 0)?; // st_gid
    store_u64(state, 0x20, 0)?; // st_rdev
    store_u64(state, 0x28, 0)?; // pad
    store_u64(state, 0x30, size)?; // st_size
    store_u32(state, 0x38, ST_BLKSIZE as u32)?; // st_blksize
    store_u32(state, 0x3C, 0)?; // pad
    store_u64(state, 0x40, 0)?; // st_blocks
    store_u64(state, 0x48, 0)?; // st_atime
    store_u64(state, 0x50, 0)?; // st_atimensec
    store_u64(state, 0x58, 0)?; // st_mtime
    store_u64(state, 0x60, 0)?; // st_mtimensec
    store_u64(state, 0x68, 0)?; // st_ctime
    store_u64(state, 0x70, 0)?; // st_ctimensec
    store_u64(state, 0x78, 0)?; // pad
    Ok(())
}

/// `fstat(fd, statbuf) → 0 | -1` — look up `fd` in the Rust
/// `FileSystem`, fill a per-arch `struct stat` at `statbuf`. Returns
/// `-1` when the fd is unknown to the Rust state (matches
/// `state.posix.fstat_with_result`'s `result = -1` branch). Falls
/// back to Python on symbolic fd / buf or unsupported arch.
pub struct NativeFstatSyscall;

impl NativeSyscall for NativeFstatSyscall {
    fn name(&self) -> &'static str {
        "fstat"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        let fd = extract_concrete_arg(&args[0], "fstat fd")?;
        let buf = extract_concrete_arg(&args[1], "fstat statbuf")?;

        // arch check first: avoid mutating state.memory if we will fall
        // back to Python anyway.
        let arch_name = state.arch().name();
        if arch_name != "AMD64" && arch_name != "ARM64" {
            return Err(SyscallError::Other(format!(
                "fstat: unsupported arch {arch_name} (only AMD64/ARM64 have a Rust handler)"
            )));
        }

        // Look up fd — `content.len()` is `content_len` (index 3 in the
        // tuple). Borrow ends before any memory_store.
        let size_opt = state.file_system_ref().fd_info(fd as u32).map(|t| t.3);
        let Some(size) = size_opt else {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        };

        match arch_name {
            "AMD64" => write_amd64_stat(state, buf, size as u64)?,
            "ARM64" => write_aarch64_stat(state, buf, size as u64)?,
            _ => unreachable!(),
        }
        Ok(SyscallOutcome::Continue { ret: 0 })
    }
}

/// `stat(pathname, statbuf) → 0 | -1` — resolve `pathname`, look up
/// its content length via `FileSystem::content_size_for_path`, write a
/// per-arch `struct stat` using the existing `write_amd64_stat` /
/// `write_aarch64_stat` helpers, return `0`. Unknown / empty path
/// returns `-1` with no buffer write. Arch coverage: AMD64 only
/// (ARM64's asm-generic ABI dropped legacy `stat` — only `newfstatat`
/// remains, already a stub). Unsupported arch returns `Other` BEFORE
/// touching the path, mirroring the `fstat` policy.
pub struct NativeStatSyscall;

impl NativeSyscall for NativeStatSyscall {
    fn name(&self) -> &'static str {
        "stat"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        // Arch check first: avoid traversing memory if we will fall
        // back to Python anyway. AMD64 is the only arch that retains
        // a legacy `stat` syscall *and* has a working Python proc.
        let arch_name = state.arch().name();
        if arch_name != "AMD64" {
            return Err(SyscallError::Other(format!(
                "stat: unsupported arch {arch_name} (only AMD64 has a Rust handler)"
            )));
        }

        let pathname_addr = extract_concrete_arg(&args[0], "stat pathname")?;
        let buf = extract_concrete_arg(&args[1], "stat statbuf")?;

        let path = read_path(state, pathname_addr, "stat")?;
        if path.is_empty() {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }

        let fs = state.file_system_ref();
        if !fs.is_path_known(&path) {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }
        let size = fs.content_size_for_path(&path).unwrap_or(0) as u64;

        write_amd64_stat(state, buf, size)?;
        Ok(SyscallOutcome::Continue { ret: 0 })
    }
}

/// `lstat(pathname, statbuf) → 0 | -1` — `stat`-shaped clone that
/// would normally diverge on symbolic links. The Rust `FileSystem`
/// model has no symlinks (open / openat never create one), so
/// `lstat` collapses to `stat` semantics: resolve `pathname`, look up
/// its content length via `FileSystem::content_size_for_path`, write
/// the AMD64 `struct stat` via `write_amd64_stat`, return `0`.
/// Unknown / empty path returns `-1` with no buffer write. Arch
/// coverage: AMD64 only — ARM64's asm-generic ABI dropped legacy
/// `lstat` entirely; x86 / ARM EABI / MIPS32 carry the legacy 32-bit
/// `struct stat` with no Python proc. Unsupported arch returns
/// `Other` BEFORE touching the path, mirroring the `stat` policy.
pub struct NativeLstatSyscall;

impl NativeSyscall for NativeLstatSyscall {
    fn name(&self) -> &'static str {
        "lstat"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        let arch_name = state.arch().name();
        if arch_name != "AMD64" {
            return Err(SyscallError::Other(format!(
                "lstat: unsupported arch {arch_name} (only AMD64 has a Rust handler)"
            )));
        }

        let pathname_addr = extract_concrete_arg(&args[0], "lstat pathname")?;
        let buf = extract_concrete_arg(&args[1], "lstat statbuf")?;

        let path = read_path(state, pathname_addr, "lstat")?;
        if path.is_empty() {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }

        let fs = state.file_system_ref();
        if !fs.is_path_known(&path) {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }
        let size = fs.content_size_for_path(&path).unwrap_or(0) as u64;

        write_amd64_stat(state, buf, size)?;
        Ok(SyscallOutcome::Continue { ret: 0 })
    }
}

/// `newfstatat(dirfd, pathname, statbuf, flag) → 0 | -1` — `stat`-shaped
/// clone with `openat`-style dirfd handling and per-arch `struct stat`
/// layout. Mirrors `NativeStatSyscall`, plus the dirfd policy from
/// `NativeOpenatSyscall`: absolute paths and `AT_FDCWD` resolve via
/// `FileSystem`; relative paths with any other dirfd return `-1` (we
/// do not model directory fds). The `flag` arg (incl. `AT_EMPTY_PATH`
/// 0x1000) is ignored — `AT_EMPTY_PATH`'s "stat the dirfd directly"
/// semantics would require dispatching to `NativeFstatSyscall(dirfd)`
/// and is deferred. Arch coverage: AMD64 + ARM64 (both use 64-bit
/// `struct stat` and have a Python `fstat.py` reference). Other arches
/// (x86 / ARM EABI / MIPS32) use the legacy 32-bit struct stat with
/// no Python proc — returns `Other` BEFORE touching state.
pub struct NativeNewfstatatSyscall;

impl NativeSyscall for NativeNewfstatatSyscall {
    fn name(&self) -> &'static str {
        "newfstatat"
    }

    fn num_args(&self) -> usize {
        4
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        let arch_name = state.arch().name();
        if arch_name != "AMD64" && arch_name != "ARM64" {
            return Err(SyscallError::Other(format!(
                "newfstatat: unsupported arch {arch_name} (only AMD64/ARM64 have a Rust handler)"
            )));
        }

        let dirfd = extract_concrete_arg(&args[0], "newfstatat dirfd")?;
        let pathname_addr = extract_concrete_arg(&args[1], "newfstatat pathname")?;
        let buf = extract_concrete_arg(&args[2], "newfstatat statbuf")?;
        // args[3] = flag — AT_EMPTY_PATH / AT_SYMLINK_NOFOLLOW are not
        // modeled (see doc comment).
        let _ = args.get(3);

        let path = read_path(state, pathname_addr, "newfstatat")?;
        if path.is_empty() {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }
        let absolute = path.starts_with('/');
        if !absolute && dirfd != AT_FDCWD_UNSIGNED {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }

        let fs = state.file_system_ref();
        if !fs.is_path_known(&path) {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }
        let size = fs.content_size_for_path(&path).unwrap_or(0) as u64;

        match arch_name {
            "AMD64" => write_amd64_stat(state, buf, size)?,
            "ARM64" => write_aarch64_stat(state, buf, size)?,
            _ => unreachable!(),
        }
        Ok(SyscallOutcome::Continue { ret: 0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;
    use crate::state::RustSimState;
    use crate::symbolic::{RustBV, SymContext};
    use crate::syscalls::{NativeSyscall, SyscallOutcome};

    /// Map an RW page at 0x2000 and stage a NUL-terminated path byte-by-byte.
    fn stage_path(state: &mut RustSimState, addr: u64, path: &[u8]) {
        state.map_memory(addr & !0xfff, 0x1000, Permission::RWX);
        for (i, b) in path.iter().enumerate() {
            state
                .memory_store(addr + i as u64, RustBV::concrete(*b as u128, 8))
                .expect("store path byte");
        }
        state
            .memory_store(addr + path.len() as u64, RustBV::concrete(0, 8))
            .expect("store NUL");
    }

    /// Build an amd64 state with `path` staged (NUL-terminated) at 0x2000 —
    /// the default staging address shared by most file_path syscall tests.
    fn state_with_path(path: &[u8]) -> RustSimState {
        let mut state = RustSimState::new("amd64").expect("state");
        stage_path(&mut state, 0x2000, path);
        state
    }

    /// Unwrap a `Continue` outcome's return value, panicking with context
    /// otherwise.
    fn expect_continue(outcome: SyscallOutcome) -> u64 {
        match outcome {
            SyscallOutcome::Continue { ret } => ret,
            other => panic!("expected Continue, got {other:?}"),
        }
    }

    /// Assert that `err` is a `SymbolicArgument` whose message contains
    /// `needle`.
    fn assert_symbolic_arg(err: SyscallError, needle: &str) {
        match err {
            SyscallError::SymbolicArgument(msg) => assert!(
                msg.contains(needle),
                "expected SymbolicArgument message to contain {needle:?}, got {msg:?}",
            ),
            other => panic!("expected SymbolicArgument, got {other:?}"),
        }
    }

    /// Assert that `err` is a `Memory` error (message not inspected).
    fn assert_memory_err(err: SyscallError) {
        match err {
            SyscallError::Memory(_) => {}
            other => panic!("expected Memory error, got {other:?}"),
        }
    }

    // angr-0hif.1 stub-sweep deleted — every file_path stub has now
    // been promoted: faccessat (angr-6009), lstat/newfstatat (angr-poao),
    // readlink/readlinkat (angr-wv38). Per-handler semantics are pinned
    // by the dedicated tests below.

    #[test]
    fn open_allocates_fresh_fd_and_records_name() {
        let mut state = state_with_path(b"/tmp/example.txt");

        let outcome = NativeOpenSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64), // O_RDONLY
                    RustBV::concrete(0, 64),
                ],
            )
            .expect("open ok");
        let fd = expect_continue(outcome);
        assert_eq!(fd, 3, "first allocated fd should be 3");
        assert!(state.file_system_ref().is_open(fd as u32));
        let (name, _, _, _, _) = state.file_system_ref().fd_info(fd as u32).unwrap();
        assert_eq!(name, "/tmp/example.txt");
    }

    #[test]
    fn open_empty_path_returns_minus_one() {
        let mut state = RustSimState::new("amd64").expect("state");
        // Just stage a NUL at addr 0x2000.
        state.map_memory(0x2000, 0x1000, Permission::RWX);
        state
            .memory_store(0x2000, RustBV::concrete(0, 8))
            .expect("store nul");

        let out = NativeOpenSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();
        assert_eq!(expect_continue(out), NEG_ONE);
    }

    #[test]
    fn open_symbolic_path_byte_falls_back() {
        let mut state = RustSimState::new("amd64").expect("state");
        state.map_memory(0x2000, 0x1000, Permission::RWX);
        let sym_byte = {
            let ctx = state.solver().borrow();
            RustBV::symbolic(&ctx, "first_path_byte", 8)
        };
        state.memory_store(0x2000, sym_byte).unwrap();

        let err = NativeOpenSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect_err("must fall back");
        assert_symbolic_arg(err, "open");
    }

    #[test]
    fn open_symbolic_pathname_addr_falls_back() {
        let mut state = RustSimState::new("amd64").expect("state");
        let sym_ptr = {
            let ctx = state.solver().borrow();
            RustBV::symbolic(&ctx, "pathname_ptr", 64)
        };
        let err = NativeOpenSyscall
            .call(
                &mut state,
                &[sym_ptr, RustBV::concrete(0, 64), RustBV::concrete(0, 64)],
            )
            .expect_err("must fall back");
        assert_symbolic_arg(err, "pathname");
    }

    #[test]
    fn openat_absolute_path_allocates_fd_ignoring_dirfd() {
        let mut state = state_with_path(b"/etc/hosts");

        let outcome = NativeOpenatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(99, 64), // arbitrary dirfd — ignored for absolute paths
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect("openat ok");
        match outcome {
            SyscallOutcome::Continue { ret } => {
                assert_eq!(ret, 3);
                assert!(state.file_system_ref().is_open(ret as u32));
            }
            other => panic!("expected Continue, got {other:?}"),
        }
    }

    #[test]
    fn openat_relative_path_with_at_fdcwd_allocates_fd() {
        let mut state = state_with_path(b"flag.txt");

        let outcome = NativeOpenatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(AT_FDCWD_UNSIGNED as u128, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect("openat ok");
        assert_eq!(expect_continue(outcome), 3);
    }

    #[test]
    fn openat_relative_path_without_at_fdcwd_returns_minus_one() {
        let mut state = state_with_path(b"flag.txt");
        let prev_next_fd = state.file_system_ref().next_fd();

        let outcome = NativeOpenatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(7, 64), // arbitrary dirfd ≠ AT_FDCWD
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect("openat ok");
        assert_eq!(expect_continue(outcome), NEG_ONE);
        // No fd should have been allocated.
        assert_eq!(state.file_system_ref().next_fd(), prev_next_fd);
    }

    #[test]
    fn close_open_fd_returns_zero_and_marks_closed() {
        let mut state = RustSimState::new("amd64").expect("state");
        // Allocate a fresh fd via the file system directly.
        let fd = state.file_system().open("/tmp/x".into(), FdFlags::ReadOnly);
        assert!(state.file_system_ref().is_open(fd));

        let outcome = NativeCloseSyscall
            .call(&mut state, &[RustBV::concrete(fd as u128, 64)])
            .unwrap();
        assert_eq!(expect_continue(outcome), 0);
        assert!(!state.file_system_ref().is_open(fd));
    }

    #[test]
    fn close_unknown_fd_returns_minus_one() {
        let mut state = RustSimState::new("amd64").expect("state");
        // fd=99 was never allocated.
        let outcome = NativeCloseSyscall
            .call(&mut state, &[RustBV::concrete(99, 64)])
            .unwrap();
        assert_eq!(expect_continue(outcome), NEG_ONE);
    }

    #[test]
    fn close_symbolic_fd_falls_back() {
        let mut state = RustSimState::new("amd64").expect("state");
        let ctx = SymContext::new();
        let sym_fd = RustBV::symbolic(&ctx, "fd", 64);
        let err = NativeCloseSyscall
            .call(&mut state, &[sym_fd])
            .expect_err("must fall back");
        assert_symbolic_arg(err, "fd");
    }

    #[test]
    fn access_unknown_path_returns_minus_one() {
        let mut state = state_with_path(b"/no/such/file");

        let out = NativeAccessSyscall
            .call(
                &mut state,
                &[RustBV::concrete(0x2000, 64), RustBV::concrete(0, 64)],
            )
            .expect("access ok");
        assert_eq!(expect_continue(out), NEG_ONE);
    }

    #[test]
    fn access_after_open_returns_zero() {
        let mut state = state_with_path(b"/tmp/exists.txt");

        // Open registers the path.
        NativeOpenSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect("open ok");

        // Re-stage path at a different addr to prove access reads it
        // fresh (not relying on caller-side cached state).
        stage_path(&mut state, 0x3000, b"/tmp/exists.txt");

        let out = NativeAccessSyscall
            .call(
                &mut state,
                &[RustBV::concrete(0x3000, 64), RustBV::concrete(0, 64)],
            )
            .expect("access ok");
        assert_eq!(expect_continue(out), 0);
    }

    #[test]
    fn access_registered_path_returns_zero() {
        let mut state = RustSimState::new("amd64").expect("state");
        // Mirror the Python state.fs.insert path: register without
        // allocating an fd.
        state
            .file_system()
            .register_known_path("/etc/passwd".to_string());
        stage_path(&mut state, 0x2000, b"/etc/passwd");

        let out = NativeAccessSyscall
            .call(
                &mut state,
                &[RustBV::concrete(0x2000, 64), RustBV::concrete(0, 64)],
            )
            .expect("access ok");
        assert_eq!(expect_continue(out), 0);
    }

    #[test]
    fn access_empty_path_returns_minus_one() {
        let mut state = RustSimState::new("amd64").expect("state");
        state.map_memory(0x2000, 0x1000, Permission::RWX);
        state
            .memory_store(0x2000, RustBV::concrete(0, 8))
            .expect("store NUL");

        let out = NativeAccessSyscall
            .call(
                &mut state,
                &[RustBV::concrete(0x2000, 64), RustBV::concrete(0, 64)],
            )
            .expect("access ok");
        assert_eq!(expect_continue(out), NEG_ONE);
    }

    #[test]
    fn access_symbolic_path_byte_falls_back() {
        let mut state = RustSimState::new("amd64").expect("state");
        state.map_memory(0x2000, 0x1000, Permission::RWX);
        let sym_byte = {
            let ctx = state.solver().borrow();
            RustBV::symbolic(&ctx, "first_path_byte", 8)
        };
        state.memory_store(0x2000, sym_byte).unwrap();

        let err = NativeAccessSyscall
            .call(
                &mut state,
                &[RustBV::concrete(0x2000, 64), RustBV::concrete(0, 64)],
            )
            .expect_err("must fall back");
        assert_symbolic_arg(err, "access");
    }

    #[test]
    fn access_symbolic_pathname_addr_falls_back() {
        let mut state = RustSimState::new("amd64").expect("state");
        let sym_ptr = {
            let ctx = state.solver().borrow();
            RustBV::symbolic(&ctx, "pathname_ptr", 64)
        };
        let err = NativeAccessSyscall
            .call(&mut state, &[sym_ptr, RustBV::concrete(0, 64)])
            .expect_err("must fall back");
        assert_symbolic_arg(err, "pathname");
    }

    #[test]
    fn access_round_trip_sweeps_supported_arches() {
        // AArch64 has no legacy `access` syscall (asm-generic only ships
        // faccessat), but the handler itself is arch-agnostic — exercise
        // it from each `RustSimState::new(...)` arch to confirm the
        // path-read + lookup path is independent of pointer width.
        for arch in ["amd64", "x86", "armel", "aarch64", "mipsel"] {
            let mut state = RustSimState::new(arch).expect("state");
            let bits = state.arch().bits();
            stage_path(&mut state, 0x2000, b"/tmp/access-rt");

            // Unknown → -1.
            let out = NativeAccessSyscall
                .call(
                    &mut state,
                    &[RustBV::concrete(0x2000, bits), RustBV::concrete(0, bits)],
                )
                .expect("access");
            match out {
                SyscallOutcome::Continue { ret } => {
                    assert_eq!(ret, NEG_ONE, "{arch} pre-open")
                }
                other => panic!("{arch} pre-open: got {other:?}"),
            }

            // Register, then known → 0.
            state
                .file_system()
                .register_known_path("/tmp/access-rt".to_string());
            let out2 = NativeAccessSyscall
                .call(
                    &mut state,
                    &[RustBV::concrete(0x2000, bits), RustBV::concrete(0, bits)],
                )
                .expect("access");
            match out2 {
                SyscallOutcome::Continue { ret } => {
                    assert_eq!(ret, 0, "{arch} post-register")
                }
                other => panic!("{arch} post-register: got {other:?}"),
            }
        }
    }

    /// `AT_FDCWD` reinterpreted as unsigned 64-bit — same constant the
    /// handler matches on. Kept inline so the test stays self-contained.
    const TEST_AT_FDCWD: u64 = AT_FDCWD_UNSIGNED;

    #[test]
    fn faccessat_unknown_path_returns_minus_one() {
        let mut state = state_with_path(b"/no/such/file");

        let out = NativeFaccessatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(TEST_AT_FDCWD as u128, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64), // mode (ignored)
                ],
            )
            .expect("faccessat ok");
        assert_eq!(expect_continue(out), NEG_ONE);
    }

    #[test]
    fn faccessat_after_open_returns_zero_with_at_fdcwd() {
        let mut state = state_with_path(b"/tmp/fa-exists.txt");

        NativeOpenSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect("open ok");

        let out = NativeFaccessatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(TEST_AT_FDCWD as u128, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect("faccessat ok");
        assert_eq!(expect_continue(out), 0);
    }

    #[test]
    fn faccessat_absolute_path_ignores_dirfd() {
        // Absolute paths bypass dirfd entirely — any value should resolve.
        let mut state = RustSimState::new("amd64").expect("state");
        state
            .file_system()
            .register_known_path("/etc/passwd".to_string());
        stage_path(&mut state, 0x2000, b"/etc/passwd");

        let out = NativeFaccessatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(42, 64), // arbitrary dirfd
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect("faccessat ok");
        assert_eq!(expect_continue(out), 0);
    }

    #[test]
    fn faccessat_relative_path_non_atfdcwd_returns_minus_one() {
        // Mirrors NativeOpenatSyscall: we do not model dirfd directories,
        // so relative paths with a non-AT_FDCWD dirfd cannot be resolved.
        let mut state = RustSimState::new("amd64").expect("state");
        // Register the bare name in case the handler ever resolved it
        // without consulting dirfd — proves we are NOT doing that.
        state
            .file_system()
            .register_known_path("local.txt".to_string());
        stage_path(&mut state, 0x2000, b"local.txt");

        let out = NativeFaccessatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(7, 64), // arbitrary dirfd != AT_FDCWD
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect("faccessat ok");
        assert_eq!(expect_continue(out), NEG_ONE);
    }

    #[test]
    fn faccessat_empty_path_returns_minus_one() {
        let mut state = RustSimState::new("amd64").expect("state");
        state.map_memory(0x2000, 0x1000, Permission::RWX);
        state
            .memory_store(0x2000, RustBV::concrete(0, 8))
            .expect("store NUL");

        let out = NativeFaccessatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(TEST_AT_FDCWD as u128, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect("faccessat ok");
        assert_eq!(expect_continue(out), NEG_ONE);
    }

    #[test]
    fn faccessat_symbolic_path_byte_falls_back() {
        let mut state = RustSimState::new("amd64").expect("state");
        state.map_memory(0x2000, 0x1000, Permission::RWX);
        let sym_byte = {
            let ctx = state.solver().borrow();
            RustBV::symbolic(&ctx, "first_path_byte", 8)
        };
        state.memory_store(0x2000, sym_byte).unwrap();

        let err = NativeFaccessatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(TEST_AT_FDCWD as u128, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect_err("must fall back");
        assert_symbolic_arg(err, "faccessat");
    }

    #[test]
    fn faccessat_symbolic_dirfd_falls_back() {
        let mut state = state_with_path(b"/tmp/x");
        let sym_dirfd = {
            let ctx = state.solver().borrow();
            RustBV::symbolic(&ctx, "dirfd", 64)
        };

        let err = NativeFaccessatSyscall
            .call(
                &mut state,
                &[
                    sym_dirfd,
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect_err("must fall back");
        assert_symbolic_arg(err, "dirfd");
    }

    #[test]
    fn faccessat_round_trip_sweeps_supported_arches() {
        // Like access_round_trip_sweeps_supported_arches but for the
        // *at variant — arch independence of the dispatch + lookup paths.
        for arch in ["amd64", "x86", "armel", "aarch64", "mipsel"] {
            let mut state = RustSimState::new(arch).expect("state");
            let bits = state.arch().bits();
            stage_path(&mut state, 0x2000, b"/tmp/faccessat-rt");
            // AT_FDCWD truncates to the arch's pointer width — the raw
            // constant is 32-bit signed -100 reinterpreted unsigned, which
            // is the same value Python's openat.py uses on every arch.
            let atfdcwd = if bits == 64 {
                TEST_AT_FDCWD as u128
            } else {
                (TEST_AT_FDCWD & ((1u64 << bits) - 1)) as u128
            };

            let out = NativeFaccessatSyscall
                .call(
                    &mut state,
                    &[
                        RustBV::concrete(atfdcwd, bits),
                        RustBV::concrete(0x2000, bits),
                        RustBV::concrete(0, bits),
                    ],
                )
                .expect("faccessat");
            match out {
                SyscallOutcome::Continue { ret } => {
                    assert_eq!(ret, NEG_ONE, "{arch} pre-register")
                }
                other => panic!("{arch} pre-register: got {other:?}"),
            }

            state
                .file_system()
                .register_known_path("/tmp/faccessat-rt".to_string());
            let out2 = NativeFaccessatSyscall
                .call(
                    &mut state,
                    &[
                        RustBV::concrete(atfdcwd, bits),
                        RustBV::concrete(0x2000, bits),
                        RustBV::concrete(0, bits),
                    ],
                )
                .expect("faccessat");
            match out2 {
                SyscallOutcome::Continue { ret } => {
                    assert_eq!(ret, 0, "{arch} post-register")
                }
                other => panic!("{arch} post-register: got {other:?}"),
            }
        }
    }

    // ---- readlink / readlinkat (angr-wv38) ----

    #[test]
    fn readlink_unknown_path_returns_minus_one() {
        let mut state = state_with_path(b"/no/such/path");

        let out = NativeReadlinkSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0x3000, 64), // buf — must NOT be touched
                    RustBV::concrete(256, 64),    // bufsiz
                ],
            )
            .expect("readlink ok");
        assert_eq!(expect_continue(out), NEG_ONE);
    }

    #[test]
    fn readlink_known_path_still_returns_minus_one() {
        // Even for paths the FileSystem knows about, readlink must return
        // -1 (EINVAL — not a symlink). The FileSystem has no symlinks.
        let mut state = state_with_path(b"/tmp/known");
        state
            .file_system()
            .register_known_path("/tmp/known".to_string());

        let out = NativeReadlinkSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0x3000, 64),
                    RustBV::concrete(256, 64),
                ],
            )
            .expect("readlink ok");
        assert_eq!(expect_continue(out), NEG_ONE);
    }

    #[test]
    fn readlink_empty_path_returns_minus_one() {
        let mut state = RustSimState::new("amd64").expect("state");
        state.map_memory(0x2000, 0x1000, Permission::RWX);
        state
            .memory_store(0x2000, RustBV::concrete(0, 8))
            .expect("store nul");

        let out = NativeReadlinkSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect("readlink ok");
        assert_eq!(expect_continue(out), NEG_ONE);
    }

    #[test]
    fn readlink_buf_is_not_modified_on_failure() {
        // The buffer must NOT be written: real Linux only fills it on a
        // positive return, and we always return -1.
        let mut state = state_with_path(b"/whatever");
        // Pre-mark buf with a sentinel; after the call it must still be
        // there (we never wrote to it).
        state.map_memory(0x3000, 0x1000, Permission::RWX);
        for i in 0..8 {
            state
                .memory_store(0x3000 + i, RustBV::concrete(0xAA, 8))
                .expect("store sentinel");
        }

        let _ = NativeReadlinkSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0x3000, 64),
                    RustBV::concrete(8, 64),
                ],
            )
            .expect("readlink ok");

        for i in 0..8u64 {
            let bv = state.memory_load(0x3000 + i, 1).expect("load");
            assert_eq!(
                bv.as_u64().unwrap(),
                0xAA,
                "buf byte {i} was touched (must be untouched on -1 return)"
            );
        }
    }

    #[test]
    fn readlink_symbolic_path_byte_falls_back() {
        let mut state = RustSimState::new("amd64").expect("state");
        state.map_memory(0x2000, 0x1000, Permission::RWX);
        let sym_byte = {
            let ctx = state.solver().borrow();
            RustBV::symbolic(&ctx, "first_path_byte", 8)
        };
        state.memory_store(0x2000, sym_byte).unwrap();

        let err = NativeReadlinkSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect_err("must fall back");
        assert_symbolic_arg(err, "readlink");
    }

    #[test]
    fn readlink_symbolic_pathname_addr_falls_back() {
        let mut state = RustSimState::new("amd64").expect("state");
        let sym_ptr = {
            let ctx = state.solver().borrow();
            RustBV::symbolic(&ctx, "pathname_ptr", 64)
        };
        let err = NativeReadlinkSyscall
            .call(
                &mut state,
                &[sym_ptr, RustBV::concrete(0, 64), RustBV::concrete(0, 64)],
            )
            .expect_err("must fall back");
        assert_symbolic_arg(err, "pathname");
    }

    #[test]
    fn readlink_round_trip_sweeps_supported_arches() {
        // readlink is registered on every arch except ARM64; the handler
        // itself has no arch-specific code. Sweep all arches it can be
        // dispatched on to pin arch-independence.
        for arch in ["amd64", "x86", "armel", "mipsel"] {
            let mut state = RustSimState::new(arch).expect("state");
            let bits = state.arch().bits();
            stage_path(&mut state, 0x2000, b"/tmp/whatever");
            let out = NativeReadlinkSyscall
                .call(
                    &mut state,
                    &[
                        RustBV::concrete(0x2000, bits),
                        RustBV::concrete(0, bits),
                        RustBV::concrete(0, bits),
                    ],
                )
                .expect("readlink");
            match out {
                SyscallOutcome::Continue { ret } => {
                    assert_eq!(ret, NEG_ONE, "{arch} readlink ret")
                }
                other => panic!("{arch}: got {other:?}"),
            }
        }
    }

    #[test]
    fn readlinkat_at_fdcwd_unknown_path_returns_minus_one() {
        let mut state = state_with_path(b"/no/such/path");

        let out = NativeReadlinkatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(TEST_AT_FDCWD as u128, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0x3000, 64),
                    RustBV::concrete(256, 64),
                ],
            )
            .expect("readlinkat ok");
        assert_eq!(expect_continue(out), NEG_ONE);
    }

    #[test]
    fn readlinkat_known_path_still_returns_minus_one() {
        let mut state = state_with_path(b"/tmp/known");
        state
            .file_system()
            .register_known_path("/tmp/known".to_string());

        let out = NativeReadlinkatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(TEST_AT_FDCWD as u128, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0x3000, 64),
                    RustBV::concrete(256, 64),
                ],
            )
            .expect("readlinkat ok");
        assert_eq!(expect_continue(out), NEG_ONE);
    }

    #[test]
    fn readlinkat_absolute_path_ignores_dirfd() {
        // Absolute path: dirfd does not matter, still -1.
        let mut state = state_with_path(b"/abs/path");

        let out = NativeReadlinkatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(99, 64), // arbitrary dirfd
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0x3000, 64),
                    RustBV::concrete(256, 64),
                ],
            )
            .expect("readlinkat ok");
        assert_eq!(expect_continue(out), NEG_ONE);
    }

    #[test]
    fn readlinkat_relative_path_non_atfdcwd_returns_minus_one() {
        // Relative path with arbitrary dirfd: still -1 (would be -1
        // anyway, but the short-circuit branch exists for symmetry with
        // faccessat / openat).
        let mut state = state_with_path(b"relative/path");

        let out = NativeReadlinkatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(99, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0x3000, 64),
                    RustBV::concrete(256, 64),
                ],
            )
            .expect("readlinkat ok");
        assert_eq!(expect_continue(out), NEG_ONE);
    }

    #[test]
    fn readlinkat_empty_path_returns_minus_one() {
        let mut state = RustSimState::new("amd64").expect("state");
        state.map_memory(0x2000, 0x1000, Permission::RWX);
        state
            .memory_store(0x2000, RustBV::concrete(0, 8))
            .expect("store nul");

        let out = NativeReadlinkatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(TEST_AT_FDCWD as u128, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect("readlinkat ok");
        assert_eq!(expect_continue(out), NEG_ONE);
    }

    #[test]
    fn readlinkat_symbolic_dirfd_falls_back() {
        let mut state = RustSimState::new("amd64").expect("state");
        let sym_dirfd = {
            let ctx = state.solver().borrow();
            RustBV::symbolic(&ctx, "dirfd", 64)
        };
        let err = NativeReadlinkatSyscall
            .call(
                &mut state,
                &[
                    sym_dirfd,
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect_err("must fall back");
        assert_symbolic_arg(err, "dirfd");
    }

    #[test]
    fn readlinkat_symbolic_pathname_addr_falls_back() {
        let mut state = RustSimState::new("amd64").expect("state");
        let sym_ptr = {
            let ctx = state.solver().borrow();
            RustBV::symbolic(&ctx, "pathname_ptr", 64)
        };
        let err = NativeReadlinkatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(TEST_AT_FDCWD as u128, 64),
                    sym_ptr,
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect_err("must fall back");
        assert_symbolic_arg(err, "pathname");
    }

    #[test]
    fn readlinkat_round_trip_sweeps_supported_arches() {
        // readlinkat is registered on every arch including ARM64 (78).
        for arch in ["amd64", "x86", "armel", "aarch64", "mipsel"] {
            let mut state = RustSimState::new(arch).expect("state");
            let bits = state.arch().bits();
            stage_path(&mut state, 0x2000, b"/tmp/whatever");
            let atfdcwd = if bits == 64 {
                TEST_AT_FDCWD as u128
            } else {
                (TEST_AT_FDCWD & ((1u64 << bits) - 1)) as u128
            };
            let out = NativeReadlinkatSyscall
                .call(
                    &mut state,
                    &[
                        RustBV::concrete(atfdcwd, bits),
                        RustBV::concrete(0x2000, bits),
                        RustBV::concrete(0, bits),
                        RustBV::concrete(0, bits),
                    ],
                )
                .expect("readlinkat");
            match out {
                SyscallOutcome::Continue { ret } => {
                    assert_eq!(ret, NEG_ONE, "{arch} readlinkat ret")
                }
                other => panic!("{arch}: got {other:?}"),
            }
        }
    }

    /// Read `size` bytes of LE-packed u64 from memory at `addr`.
    fn read_u64_le(state: &RustSimState, addr: u64) -> u64 {
        let bv = state.memory_load(addr, 8).expect("load");
        bv.as_u64().expect("concrete")
    }

    fn read_u32_le(state: &RustSimState, addr: u64) -> u32 {
        let bv = state.memory_load(addr, 4).expect("load");
        bv.as_u64().expect("concrete") as u32
    }

    #[test]
    fn fstat_unknown_fd_returns_minus_one() {
        let mut state = RustSimState::new("amd64").expect("state");
        state.map_memory(0x4000, 0x1000, Permission::RW);

        let out = NativeFstatSyscall
            .call(
                &mut state,
                &[RustBV::concrete(99, 64), RustBV::concrete(0x4000, 64)],
            )
            .expect("fstat ok");
        assert_eq!(expect_continue(out), NEG_ONE);
        // Buffer must NOT have been touched on the failure path
        // (read 0s from the freshly-mapped page).
        assert_eq!(read_u64_le(&state, 0x4000), 0);
    }

    #[test]
    fn fstat_known_fd_writes_amd64_layout_and_returns_zero() {
        let mut state = RustSimState::new("amd64").expect("state");
        state.map_memory(0x4000, 0x1000, Permission::RW);

        // Seed a file with concrete content so content_len = 13.
        let fd = state.file_system().open_with_content(
            "/tmp/hello".into(),
            FdFlags::ReadOnly,
            b"hello, world!".to_vec(),
        );

        let out = NativeFstatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(fd as u128, 64),
                    RustBV::concrete(0x4000, 64),
                ],
            )
            .expect("fstat ok");
        assert_eq!(expect_continue(out), 0);

        // st_size at offset 0x30
        assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 13);
        // st_mode at offset 0x18 (u32)
        assert_eq!(read_u32_le(&state, 0x4000 + 0x18), S_IFREG_0755 as u32);
        // st_blksize at offset 0x38
        assert_eq!(read_u64_le(&state, 0x4000 + 0x38), ST_BLKSIZE);
        // st_dev at offset 0 — zero
        assert_eq!(read_u64_le(&state, 0x4000), 0);
        // st_ctimensec at 0x70 — zero
        assert_eq!(read_u64_le(&state, 0x4000 + 0x70), 0);
    }

    #[test]
    fn fstat_known_fd_writes_aarch64_layout_and_returns_zero() {
        let mut state = RustSimState::new("aarch64").expect("state");
        state.map_memory(0x4000, 0x1000, Permission::RW);

        let fd = state.file_system().open_with_content(
            "/tmp/arm".into(),
            FdFlags::ReadOnly,
            vec![0u8; 4096],
        );

        let out = NativeFstatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(fd as u128, 64),
                    RustBV::concrete(0x4000, 64),
                ],
            )
            .expect("fstat ok");
        assert_eq!(expect_continue(out), 0);

        // AArch64-specific: st_mode at 0x10 (NOT 0x18 like AMD64).
        assert_eq!(read_u32_le(&state, 0x4000 + 0x10), S_IFREG_0755 as u32);
        // st_nlink at 0x14 — zero u32
        assert_eq!(read_u32_le(&state, 0x4000 + 0x14), 0);
        // st_size at 0x30
        assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 4096);
        // st_blksize at 0x38 is u32 here
        assert_eq!(read_u32_le(&state, 0x4000 + 0x38), ST_BLKSIZE as u32);
        // The last padding word at 0x78 — zero
        assert_eq!(read_u64_le(&state, 0x4000 + 0x78), 0);
    }

    #[test]
    fn fstat_unsupported_arch_falls_back() {
        // X86 / ARM / MIPS32 have legacy 32-bit struct stat and no
        // Python implementation in fstat.py either, so the handler
        // intentionally errors out and lets the dispatcher fall back
        // to the Python proc (which itself raises).
        for arch in ["x86", "armel", "mipsel"] {
            let mut state = RustSimState::new(arch).expect("state");
            let bits = state.arch().bits();
            let fd = state
                .file_system()
                .open("/tmp/foo".into(), FdFlags::ReadOnly);

            let err = NativeFstatSyscall
                .call(
                    &mut state,
                    &[
                        RustBV::concrete(fd as u128, bits),
                        RustBV::concrete(0x4000, bits),
                    ],
                )
                .expect_err("{arch}: must surface as Other");
            match err {
                SyscallError::Other(msg) => {
                    assert!(msg.contains("unsupported arch"), "{arch}: got {msg:?}",)
                }
                other => panic!("{arch}: expected Other, got {other:?}"),
            }
        }
    }

    #[test]
    fn fstat_symbolic_fd_falls_back() {
        let mut state = RustSimState::new("amd64").expect("state");
        let sym_fd = {
            let ctx = state.solver().borrow();
            RustBV::symbolic(&ctx, "fd", 64)
        };
        let err = NativeFstatSyscall
            .call(&mut state, &[sym_fd, RustBV::concrete(0x4000, 64)])
            .expect_err("must fall back");
        assert_symbolic_arg(err, "fd");
    }

    #[test]
    fn fstat_symbolic_buf_falls_back() {
        let mut state = RustSimState::new("amd64").expect("state");
        let sym_buf = {
            let ctx = state.solver().borrow();
            RustBV::symbolic(&ctx, "statbuf", 64)
        };
        let err = NativeFstatSyscall
            .call(&mut state, &[RustBV::concrete(0, 64), sym_buf])
            .expect_err("must fall back");
        assert_symbolic_arg(err, "statbuf");
    }

    #[test]
    fn fstat_unmapped_buf_surfaces_error() {
        let mut state = RustSimState::new("amd64").expect("state");
        let fd = state.file_system().open("/tmp/x".into(), FdFlags::ReadOnly);
        // Do NOT map the destination page — store should error.
        let err = NativeFstatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(fd as u128, 64),
                    RustBV::concrete(0x8000, 64),
                ],
            )
            .expect_err("unmapped should error");
        // The MemoryError surfaces as SyscallError via the `?` conversion.
        assert_memory_err(err);
    }

    #[test]
    fn stat_unknown_path_returns_minus_one() {
        let mut state = state_with_path(b"/tmp/never-registered");
        state.map_memory(0x4000, 0x1000, Permission::RW);

        let out = NativeStatSyscall
            .call(
                &mut state,
                &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x4000, 64)],
            )
            .expect("stat ok");
        assert_eq!(expect_continue(out), NEG_ONE);
        // Buffer must NOT have been touched on the failure path.
        assert_eq!(read_u64_le(&state, 0x4000), 0);
    }

    #[test]
    fn stat_empty_path_returns_minus_one() {
        let mut state = RustSimState::new("amd64").expect("state");
        state.map_memory(0x2000, 0x1000, Permission::RWX);
        state
            .memory_store(0x2000, RustBV::concrete(0, 8))
            .expect("nul");
        state.map_memory(0x4000, 0x1000, Permission::RW);

        let out = NativeStatSyscall
            .call(
                &mut state,
                &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x4000, 64)],
            )
            .expect("stat ok");
        assert_eq!(expect_continue(out), NEG_ONE);
    }

    #[test]
    fn stat_known_path_with_content_writes_amd64_layout() {
        let mut state = state_with_path(b"/tmp/sized");
        state.map_memory(0x4000, 0x1000, Permission::RW);

        // Seed an fd with 13 bytes so content_size_for_path returns Some(13).
        let _fd = state.file_system().open_with_content(
            "/tmp/sized".into(),
            FdFlags::ReadOnly,
            b"hello, world!".to_vec(),
        );

        let out = NativeStatSyscall
            .call(
                &mut state,
                &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x4000, 64)],
            )
            .expect("stat ok");
        assert_eq!(expect_continue(out), 0);

        // st_size at offset 0x30
        assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 13);
        // st_mode at offset 0x18 (u32)
        assert_eq!(read_u32_le(&state, 0x4000 + 0x18), S_IFREG_0755 as u32);
        // st_blksize at offset 0x38
        assert_eq!(read_u64_le(&state, 0x4000 + 0x38), ST_BLKSIZE);
    }

    #[test]
    fn stat_registered_path_without_fd_uses_zero_size() {
        let mut state = state_with_path(b"/etc/registered-only");
        state.map_memory(0x4000, 0x1000, Permission::RW);

        // Register without allocating an fd — content_size_for_path → None.
        state
            .file_system()
            .register_known_path("/etc/registered-only".to_string());

        let out = NativeStatSyscall
            .call(
                &mut state,
                &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x4000, 64)],
            )
            .expect("stat ok");
        assert_eq!(expect_continue(out), 0);
        // st_size defaults to 0 when content_size_for_path is None.
        assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 0);
        assert_eq!(read_u32_le(&state, 0x4000 + 0x18), S_IFREG_0755 as u32);
    }

    #[test]
    fn stat_unsupported_arch_falls_back() {
        // ARM64 has no legacy stat (only newfstatat). x86 / armel /
        // mipsel carry the legacy 32-bit struct stat with no Python
        // proc support — handler errors out so the dispatcher falls
        // back to Python's error path.
        for arch in ["x86", "armel", "aarch64", "mipsel"] {
            let mut state = RustSimState::new(arch).expect("state");
            let bits = state.arch().bits();
            // Register the path so we'd otherwise succeed.
            state
                .file_system()
                .register_known_path("/tmp/foo".to_string());

            let err = NativeStatSyscall
                .call(
                    &mut state,
                    &[
                        RustBV::concrete(0x2000, bits),
                        RustBV::concrete(0x4000, bits),
                    ],
                )
                .expect_err("must surface as Other");
            match err {
                SyscallError::Other(msg) => {
                    assert!(msg.contains("unsupported arch"), "{arch}: got {msg:?}",)
                }
                other => panic!("{arch}: expected Other, got {other:?}"),
            }
        }
    }

    #[test]
    fn stat_symbolic_pathname_addr_falls_back() {
        let mut state = RustSimState::new("amd64").expect("state");
        let sym_ptr = {
            let ctx = state.solver().borrow();
            RustBV::symbolic(&ctx, "pathname_ptr", 64)
        };
        let err = NativeStatSyscall
            .call(&mut state, &[sym_ptr, RustBV::concrete(0x4000, 64)])
            .expect_err("must fall back");
        assert_symbolic_arg(err, "pathname");
    }

    #[test]
    fn stat_symbolic_buf_falls_back() {
        let mut state = RustSimState::new("amd64").expect("state");
        let sym_buf = {
            let ctx = state.solver().borrow();
            RustBV::symbolic(&ctx, "statbuf", 64)
        };
        let err = NativeStatSyscall
            .call(&mut state, &[RustBV::concrete(0x2000, 64), sym_buf])
            .expect_err("must fall back");
        assert_symbolic_arg(err, "statbuf");
    }

    #[test]
    fn stat_unmapped_buf_surfaces_error() {
        let mut state = state_with_path(b"/tmp/known");
        state
            .file_system()
            .register_known_path("/tmp/known".to_string());
        // Do NOT map the statbuf page.
        let err = NativeStatSyscall
            .call(
                &mut state,
                &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x8000, 64)],
            )
            .expect_err("unmapped should error");
        assert_memory_err(err);
    }

    #[test]
    fn stat_uses_largest_content_len_across_fds_for_same_path() {
        let mut state = state_with_path(b"/tmp/shared");
        state.map_memory(0x4000, 0x1000, Permission::RW);

        // Two fds for the same name with different sizes. Helper takes
        // the max (deterministic regardless of iteration order).
        let _fd_small = state.file_system().open_with_content(
            "/tmp/shared".into(),
            FdFlags::ReadOnly,
            vec![0u8; 4],
        );
        let _fd_big = state.file_system().open_with_content(
            "/tmp/shared".into(),
            FdFlags::ReadOnly,
            vec![0u8; 17],
        );

        let out = NativeStatSyscall
            .call(
                &mut state,
                &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x4000, 64)],
            )
            .expect("stat ok");
        assert_eq!(expect_continue(out), 0);
        assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 17);
    }

    // ===== lstat (angr-poao) =====
    //
    // lstat is a stat() clone in our model (no symlinks in FileSystem),
    // so most of these mirror the stat tests above. The unsupported-arch
    // case differs slightly (lstat dropped on ARM64 — newfstatat is the
    // only stat-shaped syscall there).

    #[test]
    fn lstat_unknown_path_returns_minus_one() {
        let mut state = state_with_path(b"/tmp/never-registered");
        state.map_memory(0x4000, 0x1000, Permission::RW);

        let out = NativeLstatSyscall
            .call(
                &mut state,
                &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x4000, 64)],
            )
            .expect("lstat ok");
        assert_eq!(expect_continue(out), NEG_ONE);
        // Buffer must NOT have been touched on the failure path.
        assert_eq!(read_u64_le(&state, 0x4000), 0);
    }

    #[test]
    fn lstat_empty_path_returns_minus_one() {
        let mut state = RustSimState::new("amd64").expect("state");
        state.map_memory(0x2000, 0x1000, Permission::RWX);
        state
            .memory_store(0x2000, RustBV::concrete(0, 8))
            .expect("nul");
        state.map_memory(0x4000, 0x1000, Permission::RW);

        let out = NativeLstatSyscall
            .call(
                &mut state,
                &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x4000, 64)],
            )
            .expect("lstat ok");
        assert_eq!(expect_continue(out), NEG_ONE);
    }

    #[test]
    fn lstat_known_path_with_content_writes_amd64_layout() {
        let mut state = state_with_path(b"/tmp/lsized");
        state.map_memory(0x4000, 0x1000, Permission::RW);

        let _fd = state.file_system().open_with_content(
            "/tmp/lsized".into(),
            FdFlags::ReadOnly,
            b"abc".to_vec(),
        );

        let out = NativeLstatSyscall
            .call(
                &mut state,
                &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x4000, 64)],
            )
            .expect("lstat ok");
        assert_eq!(expect_continue(out), 0);

        // st_size at offset 0x30
        assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 3);
        // st_mode at offset 0x18 (u32)
        assert_eq!(read_u32_le(&state, 0x4000 + 0x18), S_IFREG_0755 as u32);
        // st_blksize at offset 0x38
        assert_eq!(read_u64_le(&state, 0x4000 + 0x38), ST_BLKSIZE);
    }

    #[test]
    fn lstat_unsupported_arch_falls_back() {
        // ARM64 asm-generic ABI dropped legacy lstat (only newfstatat).
        // x86 / armel / mipsel carry the legacy 32-bit struct stat with
        // no Python proc support — handler errors out so the dispatcher
        // falls back to Python's error path.
        for arch in ["x86", "armel", "aarch64", "mipsel"] {
            let mut state = RustSimState::new(arch).expect("state");
            let bits = state.arch().bits();
            state
                .file_system()
                .register_known_path("/tmp/foo".to_string());

            let err = NativeLstatSyscall
                .call(
                    &mut state,
                    &[
                        RustBV::concrete(0x2000, bits),
                        RustBV::concrete(0x4000, bits),
                    ],
                )
                .expect_err("must surface as Other");
            match err {
                SyscallError::Other(msg) => {
                    assert!(msg.contains("unsupported arch"), "{arch}: got {msg:?}",)
                }
                other => panic!("{arch}: expected Other, got {other:?}"),
            }
        }
    }

    #[test]
    fn lstat_symbolic_pathname_addr_falls_back() {
        let mut state = RustSimState::new("amd64").expect("state");
        let sym_ptr = {
            let ctx = state.solver().borrow();
            RustBV::symbolic(&ctx, "pathname_ptr", 64)
        };
        let err = NativeLstatSyscall
            .call(&mut state, &[sym_ptr, RustBV::concrete(0x4000, 64)])
            .expect_err("must fall back");
        assert_symbolic_arg(err, "pathname");
    }

    #[test]
    fn lstat_unmapped_buf_surfaces_error() {
        let mut state = state_with_path(b"/tmp/known-l");
        state
            .file_system()
            .register_known_path("/tmp/known-l".to_string());
        // Do NOT map the statbuf page.
        let err = NativeLstatSyscall
            .call(
                &mut state,
                &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x8000, 64)],
            )
            .expect_err("unmapped should error");
        assert_memory_err(err);
    }

    // ===== newfstatat (angr-poao) =====

    /// Same AT_FDCWD constant used by NativeNewfstatatSyscall.
    const TEST_AT_FDCWD_NFA: u64 = AT_FDCWD_UNSIGNED;

    #[test]
    fn newfstatat_unknown_path_returns_minus_one() {
        let mut state = state_with_path(b"/no/such/file");
        state.map_memory(0x4000, 0x1000, Permission::RW);

        let out = NativeNewfstatatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(TEST_AT_FDCWD_NFA as u128, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0x4000, 64),
                    RustBV::concrete(0, 64), // flag
                ],
            )
            .expect("newfstatat ok");
        assert_eq!(expect_continue(out), NEG_ONE);
        // Buffer must not have been touched on the failure path.
        assert_eq!(read_u64_le(&state, 0x4000), 0);
    }

    #[test]
    fn newfstatat_at_fdcwd_known_path_amd64_layout() {
        let mut state = state_with_path(b"/tmp/nfa.txt");
        state.map_memory(0x4000, 0x1000, Permission::RW);
        let _fd = state.file_system().open_with_content(
            "/tmp/nfa.txt".into(),
            FdFlags::ReadOnly,
            b"hello, world!".to_vec(),
        );

        let out = NativeNewfstatatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(TEST_AT_FDCWD_NFA as u128, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0x4000, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect("newfstatat ok");
        assert_eq!(expect_continue(out), 0);

        // AMD64-specific offsets (same as fstat/stat).
        assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 13);
        assert_eq!(read_u32_le(&state, 0x4000 + 0x18), S_IFREG_0755 as u32);
        assert_eq!(read_u64_le(&state, 0x4000 + 0x38), ST_BLKSIZE);
    }

    #[test]
    fn newfstatat_at_fdcwd_known_path_aarch64_layout() {
        // ARM64 has no legacy lstat/stat — newfstatat (79) is the only
        // stat-shaped syscall on the asm-generic ABI. Critical that the
        // ARM64 struct stat layout is used here.
        let mut state = RustSimState::new("aarch64").expect("state");
        stage_path(&mut state, 0x2000, b"/tmp/arm-nfa");
        state.map_memory(0x4000, 0x1000, Permission::RW);
        let _fd = state.file_system().open_with_content(
            "/tmp/arm-nfa".into(),
            FdFlags::ReadOnly,
            vec![0u8; 4096],
        );

        let out = NativeNewfstatatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(TEST_AT_FDCWD_NFA as u128, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0x4000, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect("newfstatat ok");
        assert_eq!(expect_continue(out), 0);

        // ARM64-specific: st_mode at 0x10 (u32), st_nlink at 0x14, blksize is u32.
        assert_eq!(read_u32_le(&state, 0x4000 + 0x10), S_IFREG_0755 as u32);
        assert_eq!(read_u32_le(&state, 0x4000 + 0x14), 0);
        assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 4096);
        assert_eq!(read_u32_le(&state, 0x4000 + 0x38), ST_BLKSIZE as u32);
    }

    #[test]
    fn newfstatat_absolute_path_ignores_dirfd() {
        // Absolute paths bypass dirfd entirely.
        let mut state = RustSimState::new("amd64").expect("state");
        state
            .file_system()
            .register_known_path("/etc/passwd".to_string());
        stage_path(&mut state, 0x2000, b"/etc/passwd");
        state.map_memory(0x4000, 0x1000, Permission::RW);

        let out = NativeNewfstatatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(42, 64), // arbitrary dirfd — absolute path
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0x4000, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect("newfstatat ok");
        assert_eq!(expect_continue(out), 0);
        // Empty content (registered without fd) — st_size = 0.
        assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 0);
    }

    #[test]
    fn newfstatat_relative_path_non_atfdcwd_returns_minus_one() {
        // Mirrors NativeOpenatSyscall / NativeFaccessatSyscall: we do
        // not model dirfd directories, so relative paths with a
        // non-AT_FDCWD dirfd cannot be resolved.
        let mut state = RustSimState::new("amd64").expect("state");
        state
            .file_system()
            .register_known_path("local.txt".to_string());
        stage_path(&mut state, 0x2000, b"local.txt");
        state.map_memory(0x4000, 0x1000, Permission::RW);

        let out = NativeNewfstatatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(7, 64), // arbitrary dirfd != AT_FDCWD
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0x4000, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect("newfstatat ok");
        assert_eq!(expect_continue(out), NEG_ONE);
        // Buffer must NOT have been touched on the failure path.
        assert_eq!(read_u64_le(&state, 0x4000), 0);
    }

    #[test]
    fn newfstatat_empty_path_returns_minus_one() {
        let mut state = RustSimState::new("amd64").expect("state");
        state.map_memory(0x2000, 0x1000, Permission::RWX);
        state
            .memory_store(0x2000, RustBV::concrete(0, 8))
            .expect("nul");
        state.map_memory(0x4000, 0x1000, Permission::RW);

        let out = NativeNewfstatatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(TEST_AT_FDCWD_NFA as u128, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0x4000, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect("newfstatat ok");
        assert_eq!(expect_continue(out), NEG_ONE);
    }

    #[test]
    fn newfstatat_unsupported_arch_falls_back() {
        // x86 / armel / mipsel carry the legacy 32-bit struct stat with
        // no Python proc support. Handler errors out before any memory
        // read.
        for arch in ["x86", "armel", "mipsel"] {
            let mut state = RustSimState::new(arch).expect("state");
            let bits = state.arch().bits();
            state
                .file_system()
                .register_known_path("/tmp/foo".to_string());

            let err = NativeNewfstatatSyscall
                .call(
                    &mut state,
                    &[
                        RustBV::concrete((TEST_AT_FDCWD_NFA & ((1u64 << bits) - 1)) as u128, bits),
                        RustBV::concrete(0x2000, bits),
                        RustBV::concrete(0x4000, bits),
                        RustBV::concrete(0, bits),
                    ],
                )
                .expect_err("must surface as Other");
            match err {
                SyscallError::Other(msg) => {
                    assert!(msg.contains("unsupported arch"), "{arch}: got {msg:?}",)
                }
                other => panic!("{arch}: expected Other, got {other:?}"),
            }
        }
    }

    #[test]
    fn newfstatat_symbolic_dirfd_falls_back() {
        let mut state = state_with_path(b"/tmp/x");
        let sym_dirfd = {
            let ctx = state.solver().borrow();
            RustBV::symbolic(&ctx, "dirfd", 64)
        };

        let err = NativeNewfstatatSyscall
            .call(
                &mut state,
                &[
                    sym_dirfd,
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0x4000, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect_err("must fall back");
        assert_symbolic_arg(err, "dirfd");
    }

    #[test]
    fn newfstatat_symbolic_pathname_addr_falls_back() {
        let mut state = RustSimState::new("amd64").expect("state");
        let sym_ptr = {
            let ctx = state.solver().borrow();
            RustBV::symbolic(&ctx, "pathname_ptr", 64)
        };
        let err = NativeNewfstatatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(TEST_AT_FDCWD_NFA as u128, 64),
                    sym_ptr,
                    RustBV::concrete(0x4000, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect_err("must fall back");
        assert_symbolic_arg(err, "pathname");
    }

    #[test]
    fn newfstatat_unmapped_buf_surfaces_error() {
        let mut state = state_with_path(b"/tmp/known-nfa");
        state
            .file_system()
            .register_known_path("/tmp/known-nfa".to_string());
        // Do NOT map the statbuf page.
        let err = NativeNewfstatatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(TEST_AT_FDCWD_NFA as u128, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0x8000, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect_err("unmapped should error");
        assert_memory_err(err);
    }

    #[test]
    fn open_close_round_trip_sweeps_supported_arches() {
        for arch in ["amd64", "x86", "armel", "aarch64", "mipsel"] {
            let mut state = RustSimState::new(arch).expect("state");
            let bits = state.arch().bits();
            stage_path(&mut state, 0x2000, b"/tmp/roundtrip");

            let fd_out = NativeOpenSyscall
                .call(
                    &mut state,
                    &[
                        RustBV::concrete(0x2000, bits),
                        RustBV::concrete(0, bits),
                        RustBV::concrete(0, bits),
                    ],
                )
                .expect("open");
            let fd = match fd_out {
                SyscallOutcome::Continue { ret } => ret,
                other => panic!("{arch} open: expected Continue, got {other:?}"),
            };
            assert!(state.file_system_ref().is_open(fd as u32), "{arch}");

            let close_out = NativeCloseSyscall
                .call(&mut state, &[RustBV::concrete(fd as u128, bits)])
                .expect("close");
            match close_out {
                SyscallOutcome::Continue { ret } => assert_eq!(ret, 0, "{arch}"),
                other => panic!("{arch} close: expected Continue, got {other:?}"),
            }
            assert!(!state.file_system_ref().is_open(fd as u32), "{arch}");
        }
    }
}
