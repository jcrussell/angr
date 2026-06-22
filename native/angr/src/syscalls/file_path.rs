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
//! Per-arch struct layouts cover AMD64 + ARM64 (64-bit `struct stat`,
//! `fstat`) and i386 (`struct stat64` via the LFS `fstat64`/`stat64`/
//! `lstat64`/`fstatat64` syscalls — `write_i386_stat`, mirroring
//! `fstat64.py::_store_i386`, angr-11djq.5.1). The legacy i386 `fstat`
//! (108, old 32-bit `struct stat`) has no Rust writer — its Python
//! proc raises — so it still falls through. ARM / MIPS32 likewise carry
//! a legacy 32-bit `struct stat` with no Rust writer and fall through
//! to the Python error path.
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

/// i386 `struct stat64` layout (LFS variant — the one 32-bit glibc
/// actually emits via `stat64`/`lstat64`/`fstat64`/`fstatat64`). Mirrors
/// `angr/procedures/linux_kernel/fstat64.py::_store_i386` field-for-field,
/// including its overlapping writes: the value widths come from
/// `posix.fstat_with_result`'s `Stat` tuple (`st_dev`/`st_ino`/`st_nlink`/
/// `st_rdev`/`st_size`/`st_blksize`/`st_blocks`/times are 64-bit;
/// `st_mode`/`st_uid`/`st_gid` are 32-bit), NOT the packed struct field
/// widths, so several 64-bit stores spill into the next field and are
/// overwritten by the following store. Replaying Python's exact order
/// reproduces its byte output. All fields but `st_mode` (concrete
/// `S_IFREG | 0o755`), `st_size` and `st_blksize` are zero — matching the
/// AMD64/AArch64 Rust handlers (Python mints a symbolic `st_mode`; the
/// Rust path uses a concrete constant — see `write_amd64_stat`).
fn write_i386_stat(state: &mut RustSimState, buf: u64, size: u64) -> Result<(), SyscallError> {
    let store_u64 = |state: &mut RustSimState, off: u64, val: u64| -> Result<(), SyscallError> {
        state.memory_store(buf + off, RustBV::concrete(val as u128, 64))?;
        Ok(())
    };
    let store_u32 = |state: &mut RustSimState, off: u64, val: u32| -> Result<(), SyscallError> {
        state.memory_store(buf + off, RustBV::concrete(val as u128, 32))?;
        Ok(())
    };

    store_u64(state, 0x00, 0)?; // st_dev
    store_u64(state, 0x0C, 0)?; // st_ino (64-bit; low half overlaps st_mode, both zero)
    store_u32(state, 0x10, S_IFREG_0755 as u32)?; // st_mode
    store_u64(state, 0x14, 0)?; // st_nlink (64-bit; overlaps st_uid, both zero)
    store_u32(state, 0x18, 0)?; // st_uid
    store_u32(state, 0x1C, 0)?; // st_gid
    store_u64(state, 0x20, 0)?; // st_rdev
    store_u64(state, 0x2C, size)?; // st_size
    store_u64(state, 0x34, ST_BLKSIZE)?; // st_blksize (64-bit; upper overlaps st_blocks)
    store_u64(state, 0x38, 0)?; // st_blocks
    store_u32(state, 0x3C, 0)?; // padding
    store_u64(state, 0x40, 0)?; // st_atime
    store_u64(state, 0x44, 0)?; // st_atimensec
    store_u64(state, 0x48, 0)?; // st_mtime
    store_u64(state, 0x4C, 0)?; // st_mtimensec
    store_u64(state, 0x50, 0)?; // st_ctime
    store_u64(state, 0x54, 0)?; // st_ctimensec
    store_u64(state, 0x5C, 0)?; // st_ino (verification copy)
    Ok(())
}

/// Dispatch the per-arch `struct stat` writer. Callers must arch-guard
/// first (each stat-family handler accepts a slightly different arch set
/// — legacy `stat`/`lstat` are AMD64+X86 only, `fstat`/`newfstatat` add
/// ARM64), so the `_` arm is defensive rather than a normal path.
fn write_stat_for_arch(
    state: &mut RustSimState,
    arch_name: &str,
    buf: u64,
    size: u64,
) -> Result<(), SyscallError> {
    match arch_name {
        "AMD64" => write_amd64_stat(state, buf, size),
        "ARM64" => write_aarch64_stat(state, buf, size),
        "X86" => write_i386_stat(state, buf, size),
        other => Err(SyscallError::Other(format!(
            "stat: no struct-stat writer for arch {other}"
        ))),
    }
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
        if arch_name != "AMD64" && arch_name != "ARM64" && arch_name != "X86" {
            return Err(SyscallError::Other(format!(
                "fstat: unsupported arch {arch_name} (only AMD64/ARM64/X86 have a Rust handler)"
            )));
        }

        // Look up fd — `content.len()` is `content_len` (index 3 in the
        // tuple). Borrow ends before any memory_store.
        let size_opt = state.file_system_ref().fd_info(fd as u32).map(|t| t.3);
        let Some(size) = size_opt else {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        };

        write_stat_for_arch(state, arch_name, buf, size as u64)?;
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
        if arch_name != "AMD64" && arch_name != "X86" {
            return Err(SyscallError::Other(format!(
                "stat: unsupported arch {arch_name} (only AMD64/X86 have a Rust handler)"
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

        write_stat_for_arch(state, arch_name, buf, size)?;
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
        if arch_name != "AMD64" && arch_name != "X86" {
            return Err(SyscallError::Other(format!(
                "lstat: unsupported arch {arch_name} (only AMD64/X86 have a Rust handler)"
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

        write_stat_for_arch(state, arch_name, buf, size)?;
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
        if arch_name != "AMD64" && arch_name != "ARM64" && arch_name != "X86" {
            return Err(SyscallError::Other(format!(
                "newfstatat: unsupported arch {arch_name} (only AMD64/ARM64/X86 have a Rust handler)"
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

        write_stat_for_arch(state, arch_name, buf, size)?;
        Ok(SyscallOutcome::Continue { ret: 0 })
    }
}

#[cfg(test)]
#[path = "file_path_tests.rs"]
mod file_path_tests;
