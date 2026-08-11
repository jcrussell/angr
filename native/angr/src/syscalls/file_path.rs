//! File-path syscall handlers.
//!
//! ## `readlink` / `readlinkat` (angr-wv38)
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
//! `fstat(fd, statbuf) → 0 | -1` reads the fd's content length from
//! `FileSystem::effective_size(fd)` and writes a per-arch `struct stat`
//! to `statbuf`. Mirrors `procedures/linux_kernel/fstat.py`, which
//! delegates to `state.posix.fstat_with_result`. The Rust handler
//! diverges in two intentional ways:
//!
//! * `st_mode` is written as a concrete constant — `S_IFREG | 0o755`
//!   for every fd/regular-file path, `S_IFLNK | 0777` for a symlink
//!   under `lstat` — instead of a fresh symbolic BVS (Python's
//!   `fstat_with_result` mints `BVS("st_mode", 32)`). The concrete
//!   value matches the file type's real permissions and avoids
//!   spawning a symbolic the binary will likely just compare against
//!   `S_IFREG`. Anything that depends on a symbolic mode must use
//!   the Python proc.
//! * `st_size` comes from `content_len` (whatever was last written
//!   to the fd's backing buffer) — concrete, not symbolic.
//!
//! Per-arch struct layouts cover AMD64 + ARM64 (64-bit `struct stat`,
//! `fstat`), i386 (`struct stat64` via the LFS `fstat64`/`stat64`/
//! `lstat64`/`fstatat64` syscalls — `write_i386_stat`, mirroring
//! `fstat64.py::_store_i386`, angr-11djq.5.1), and ARM EABI / MIPS32
//! (their own 32-bit LFS `struct stat64` layouts — `write_arm_stat` /
//! `write_mips32_stat`). `write_stat_for_arch` is the single dispatch
//! point; anything outside those five arches returns `Other` and falls
//! through to Python. On i386 and MIPS32 the legacy pre-LFS numbers
//! (106/107/108 and 4106/4107/4108, old 32-bit `struct stat`) are
//! deliberately left unregistered: modern 32-bit glibc emits the `*64`
//! variants, and angr's own Python map has no writer for the legacy
//! layout either — see the registration-table comments in `mod.rs`.
//!
//! Unknown fd → `-1` with no buffer write (matches
//! `fstat_with_result`'s `result = -1` branch).
//!
//! ## `stat` (angr-k3ol.4)
//!
//! `stat(pathname, statbuf) → 0 | -1` resolves `pathname` via
//! `read_path`, returns `-1` for empty / unknown paths (via
//! `stat_lookup_follow`, which walks the symlink table then checks
//! `FileSystem::is_path_known`), and otherwise writes a per-arch
//! `struct stat` via the same `write_stat_for_arch` dispatch used by
//! `fstat`. The size field is sourced from
//! `FileSystem::content_size_for_path(path)` (largest
//! `content_len` across any fd that opened the path) — `0` if the
//! path was registered via `register_known_path` without a content
//! payload. This diverges from `procedures/linux_kernel/stat.py`,
//! which opens a temp fd, calls `fstat`, then closes. The Rust path
//! never mutates the fd table, so the next-fd counter is stable
//! across stat queries. Arch coverage: AMD64 (legacy `stat` 4) plus
//! X86 / ARM EABI / MIPS32 via their LFS `stat64` numbers (195 / 195 /
//! 4213). ARM64 is absent by design, not by omission — its asm-generic
//! ABI dropped legacy `stat` entirely, leaving only `newfstatat` (79).
//! Every other arch falls back to Python's error path.
//!
//! Symbolic pathname pointer / pathname byte → `SymbolicArgument`
//! (dispatcher falls back to Python). Unmapped statbuf surfaces a
//! `MemoryError` via the `?` conversion. Unsupported arch returns
//! `Other("unsupported arch …")` — checked FIRST before reading the
//! path, so a state on an unsupported arch never even attempts to
//! traverse memory.
//!
//! ## `lstat` / `newfstatat` (angr-poao)
//!
//! `lstat(pathname, statbuf) → 0 | -1` and
//! `newfstatat(dirfd, pathname, statbuf, flag) → 0 | -1` clone the
//! `stat` semantics with two adjustments:
//!
//! * `lstat` does not follow symlinks: a path in the `FileSystem`
//!   symlink table (`add_symlink`, the same registry `readlink` reads)
//!   gets `st_mode = S_IFLNK | 0777` and `st_size = target.len()`,
//!   while `stat` / `newfstatat` walk the link to its target (up to
//!   `MAX_SYMLINK_HOPS`) and stat that. Both share
//!   `stat_lookup_nofollow` / `stat_lookup_follow` (angr-9ke6b.235;
//!   before that, both consulted `is_path_known` only, so a
//!   symlink-only path stat'd as unknown even though `readlink`
//!   resolved it). A dangling link or an over-long chain is `-1`
//!   (`ENOENT` / `ELOOP`). Same arch coverage as `stat`:
//!   AMD64 + X86 + ARM + MIPS32. Caveat: MIPS32's `struct stat64`
//!   layout writes no `st_mode` field at all (see
//!   `write_mips32_stat`), so there `lstat` on a symlink is
//!   distinguishable from a regular file only by `st_size`.
//! * `newfstatat` adds `openat`-style dirfd handling: absolute paths
//!   and `AT_FDCWD` resolve via `FileSystem`; relative paths with any
//!   other dirfd return `-1` (we do not model directory fds). Its
//!   `flag` arg honors `AT_SYMLINK_NOFOLLOW` (0x100) by routing to
//!   `stat_lookup_nofollow` (angr-zueuw) — ARM64 dropped legacy
//!   `lstat`, so glibc's `lstat()` there lowers to
//!   `fstatat(AT_FDCWD, path, buf, AT_SYMLINK_NOFOLLOW)` and this is
//!   the only no-follow entry point on that arch. `AT_EMPTY_PATH`
//!   (0x1000), which would route to `NativeFstatSyscall(dirfd)`, is
//!   still deferred (empty path → `-1`). Arch coverage
//!   is the full `write_stat_for_arch` set (AMD64 / ARM64 / X86 / ARM /
//!   MIPS32) — one arch wider than `stat` / `lstat`, since `newfstatat`
//!   is the only stat-shaped syscall on ARM64's asm-generic ABI (79).
//!
//! Both reuse `write_stat_for_arch` from `fstat` and
//! `FileSystem::content_size_for_path` from `stat`. Arch-check
//! happens FIRST (before any memory read), matching the `stat` policy.

use super::{NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg};
use crate::procedures::strings::MAX_PATH_SCAN as MAX_PATH_LEN;
use crate::state::{FdFlags, RustSimState};
use crate::symbolic::RustBV;

// readlink(path, buf, bufsiz) → long — see NativeReadlinkSyscall impl below.
// readlinkat(dfd, path, buf, bufsiz) → long — see NativeReadlinkatSyscall impl below.
// faccessat(dfd, filename, mode) → long — see NativeFaccessatSyscall impl below.
// lstat(pathname, statbuf) → long — see NativeLstatSyscall impl below.
// newfstatat(dfd, filename, statbuf, flag) → long — see NativeNewfstatatSyscall impl below.

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

/// The shared `*at` dirfd policy: `true` when this handler can resolve
/// `path` on its own, i.e. the path is absolute (dirfd irrelevant) or
/// `dirfd` is the `AT_FDCWD` sentinel. A relative path against a real
/// dirfd is not modeled — Rust's `FileSystem` has no per-fd directory —
/// so the caller returns `-1`, mirroring
/// `procedures/linux_kernel/openat.py`.
///
/// Every `*at` handler in this module (`openat`, `faccessat`,
/// `readlinkat`, `newfstatat`) must gate on this, so that adding real
/// dirfd support is a one-site change instead of four (angr-c7xno.82).
fn dirfd_allows(path: &str, dirfd: u64) -> bool {
    path.starts_with('/') || dirfd == AT_FDCWD_UNSIGNED
}

/// Read a NUL-terminated path from memory at `addr`, up to
/// `MAX_PATH_LEN`. Returns `SymbolicArgument` on the first symbolic
/// byte (the syscall then falls back to Python). Errors out with
/// `Other` if no NUL is seen within the limit.
///
/// NOTE: the cap-exceeded behavior here deliberately differs from its
/// sibling `directory.rs::read_concrete_cstring`, which *truncates* at
/// its (larger, 4096) `PATH_MAX` cap instead of erroring. The 256-byte
/// `MAX_PATH_LEN` here is a defensive scan bound for `open`/`openat`
/// where a runaway unterminated path almost certainly signals a bad
/// pointer, so erroring (→ Python fallback) is safer than silently
/// opening a truncated name. Keep this asymmetry in mind when adding a
/// new path syscall — pick the reader whose cap semantics you want.
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

/// Concrete defaults for the `struct stat` fields. `fstat_with_result`
/// in Python returns a symbolic `st_mode` and `st_size` plus a
/// `st_blksize` of `0x400`; the Rust handler swaps `st_mode` for
/// `S_IFREG | 0o755` (regular file, rwxr-xr-x — the `mode` every
/// stat-family handler passes to `write_stat_for_arch` except `lstat`
/// on a symlink, which passes `S_IFLNK_0777`) and `st_size` for the
/// concrete length of the fd's backing buffer (`content_len`). Other
/// fields stay zero, matching the Python defaults.
const S_IFREG_0755: u64 = 0o100_755;
const ST_BLKSIZE: u64 = 0x400;

/// `st_mode` for a symlink: `S_IFLNK | 0777`. Real Linux always reports
/// `0777` permission bits on a symlink, so there is no `0755` analogue
/// here. Written by `NativeLstatSyscall` when the path is registered in
/// `FileSystem`'s symlink table (`FileSystem::readlink_target`).
const S_IFLNK_0777: u64 = 0o120_777;

/// `AT_SYMLINK_NOFOLLOW` — `newfstatat`'s flag bit that selects
/// `lstat` semantics. ARM64's asm-generic ABI has no legacy `lstat`
/// syscall, so glibc's `lstat()` there lowers to
/// `fstatat(AT_FDCWD, path, buf, AT_SYMLINK_NOFOLLOW)`; honoring this
/// bit is what makes `NativeNewfstatatSyscall` the ARM64 equivalent of
/// `NativeLstatSyscall` (angr-zueuw).
const AT_SYMLINK_NOFOLLOW: u64 = 0x100;

/// Maximum symlink hops `stat_lookup_follow` will traverse before giving
/// up. Linux's own limit is 40 (`ELOOP`); 8 is plenty for the tiny
/// hand-registered link tables the Rust `FileSystem` models, and bounds
/// the walk on a cyclic registration (`/a → /b`, `/b → /a`).
const MAX_SYMLINK_HOPS: usize = 8;

/// Resolve `path` for the link-*following* stat-family handlers (`stat`,
/// `newfstatat` without `AT_SYMLINK_NOFOLLOW`). Walks the symlink table
/// up to `MAX_SYMLINK_HOPS` times, then requires the final path to be a
/// known regular file. Returns `(st_size, st_mode)`, or `None` when the
/// path is unknown, the link dangles, or the chain exceeds the hop limit
/// (all three are `-1` at the syscall boundary, matching `ENOENT` /
/// `ELOOP`).
fn stat_lookup_follow(fs: &crate::state::FileSystem, path: &str) -> Option<(u64, u32)> {
    let mut cur = path.to_string();
    for _ in 0..MAX_SYMLINK_HOPS {
        let Some(target) = fs.readlink_target(&cur) else {
            // Not a symlink — it must be a known regular file.
            return stat_lookup_nofollow(fs, &cur);
        };
        // Symlink targets are raw bytes; a non-UTF-8 target cannot name
        // a path in the `String`-keyed known-path set, so it dangles.
        cur = String::from_utf8(target.to_vec()).ok()?;
    }
    None
}

/// Resolve `path` for the link-*preserving* handler (`lstat`). A
/// registered symlink reports `S_IFLNK | 0777` sized to its raw target
/// bytes (what `readlink` would return); otherwise the path must be a
/// known regular file. Returns `(st_size, st_mode)`, or `None` for an
/// unknown path (`-1` / `ENOENT`).
fn stat_lookup_nofollow(fs: &crate::state::FileSystem, path: &str) -> Option<(u64, u32)> {
    if let Some(target) = fs.readlink_target(path) {
        return Some((target.len() as u64, S_IFLNK_0777 as u32));
    }
    if !fs.is_path_known(path) {
        return None;
    }
    Some((
        fs.content_size_for_path(path).unwrap_or(0) as u64,
        S_IFREG_0755 as u32,
    ))
}

/// Shared `struct stat` field writers used by every per-arch layout below.
/// Each takes the destination `buf` base plus the field `off`, so the arch
/// writers no longer redefine identical store closures (audit angr-myzjx.17).
///
/// `buf + off` is spelled `wrapping_add`: `buf` reaches us straight from
/// `extract_concrete_arg(&args[1], "…statbuf")` with no upper-bound check, and
/// `[profile.release]` disables overflow checks — so a bare `+` would wrap
/// silently in the shipped `.so` but panic under CI's `release-checked`
/// profile. Same rule as `invariant-proc-address-arith-wrapping`.
fn store_stat_u64(
    state: &mut RustSimState,
    buf: u64,
    off: u64,
    val: u64,
) -> Result<(), SyscallError> {
    state.memory_store(buf.wrapping_add(off), RustBV::concrete(val as u128, 64))?;
    Ok(())
}
fn store_stat_u32(
    state: &mut RustSimState,
    buf: u64,
    off: u64,
    val: u32,
) -> Result<(), SyscallError> {
    state.memory_store(buf.wrapping_add(off), RustBV::concrete(val as u128, 32))?;
    Ok(())
}
/// 96-bit zero pad (3 × 32-bit words), matching `claripy.BVV(0, 32 * 3)`.
/// MIPS32 is the only layout that needs it.
fn store_stat_zero96(state: &mut RustSimState, buf: u64, off: u64) -> Result<(), SyscallError> {
    state.memory_store(buf.wrapping_add(off), RustBV::concrete(0, 96))?;
    Ok(())
}

/// AMD64 `struct stat` layout — total 0x90 bytes. Mirrors
/// `angr/procedures/linux_kernel/fstat.py::_store_amd64`. Writes are
/// arch-LE per `RustSimState::memory.endness`.
///
/// Field widths (offsets cumulative):
/// `dev` u64, `ino` u64, `nlink` u64, `mode` u32, `uid` u32,
/// `gid` u32, pad u32, `rdev` u64, `size` u64, `blksize` u64,
/// `blocks` u64, `atime+nsec` u64×2, `mtime+nsec` u64×2,
/// `ctime+nsec` u64×2, pad u64×3.
fn write_amd64_stat(
    state: &mut RustSimState,
    buf: u64,
    size: u64,
    mode: u32,
) -> Result<(), SyscallError> {
    store_stat_u64(state, buf, 0x00, 0)?; // st_dev
    store_stat_u64(state, buf, 0x08, 0)?; // st_ino
    store_stat_u64(state, buf, 0x10, 0)?; // st_nlink
    store_stat_u32(state, buf, 0x18, mode)?; // st_mode
    store_stat_u32(state, buf, 0x1C, 0)?; // st_uid
    store_stat_u32(state, buf, 0x20, 0)?; // st_gid
    store_stat_u32(state, buf, 0x24, 0)?; // pad
    store_stat_u64(state, buf, 0x28, 0)?; // st_rdev
    store_stat_u64(state, buf, 0x30, size)?; // st_size
    store_stat_u64(state, buf, 0x38, ST_BLKSIZE)?; // st_blksize
    store_stat_u64(state, buf, 0x40, 0)?; // st_blocks
    store_stat_u64(state, buf, 0x48, 0)?; // st_atime
    store_stat_u64(state, buf, 0x50, 0)?; // st_atimensec
    store_stat_u64(state, buf, 0x58, 0)?; // st_mtime
    store_stat_u64(state, buf, 0x60, 0)?; // st_mtimensec
    store_stat_u64(state, buf, 0x68, 0)?; // st_ctime
    store_stat_u64(state, buf, 0x70, 0)?; // st_ctimensec
    store_stat_u64(state, buf, 0x78, 0)?; // pad
    store_stat_u64(state, buf, 0x80, 0)?; // pad
    store_stat_u64(state, buf, 0x88, 0)?; // pad
    Ok(())
}

/// AArch64 `struct stat` layout — total 0x80 bytes. Mirrors
/// `_store_aarch64` (note: `nlink` is u32 here, `blksize` is u32, and
/// the field order around mode/nlink/uid/gid differs from AMD64).
fn write_aarch64_stat(
    state: &mut RustSimState,
    buf: u64,
    size: u64,
    mode: u32,
) -> Result<(), SyscallError> {
    store_stat_u64(state, buf, 0x00, 0)?; // st_dev
    store_stat_u64(state, buf, 0x08, 0)?; // st_ino
    store_stat_u32(state, buf, 0x10, mode)?; // st_mode
    store_stat_u32(state, buf, 0x14, 0)?; // st_nlink
    store_stat_u32(state, buf, 0x18, 0)?; // st_uid
    store_stat_u32(state, buf, 0x1C, 0)?; // st_gid
    store_stat_u64(state, buf, 0x20, 0)?; // st_rdev
    store_stat_u64(state, buf, 0x28, 0)?; // pad
    store_stat_u64(state, buf, 0x30, size)?; // st_size
    store_stat_u32(state, buf, 0x38, ST_BLKSIZE as u32)?; // st_blksize
    store_stat_u32(state, buf, 0x3C, 0)?; // pad
    store_stat_u64(state, buf, 0x40, 0)?; // st_blocks
    store_stat_u64(state, buf, 0x48, 0)?; // st_atime
    store_stat_u64(state, buf, 0x50, 0)?; // st_atimensec
    store_stat_u64(state, buf, 0x58, 0)?; // st_mtime
    store_stat_u64(state, buf, 0x60, 0)?; // st_mtimensec
    store_stat_u64(state, buf, 0x68, 0)?; // st_ctime
    store_stat_u64(state, buf, 0x70, 0)?; // st_ctimensec
    store_stat_u64(state, buf, 0x78, 0)?; // pad
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
/// Rust path uses the caller-supplied concrete `mode` — see
/// `write_amd64_stat`).
fn write_i386_stat(
    state: &mut RustSimState,
    buf: u64,
    size: u64,
    mode: u32,
) -> Result<(), SyscallError> {
    store_stat_u64(state, buf, 0x00, 0)?; // st_dev
    store_stat_u64(state, buf, 0x0C, 0)?; // st_ino (64-bit; low half overlaps st_mode, both zero)
    store_stat_u32(state, buf, 0x10, mode)?; // st_mode
    store_stat_u64(state, buf, 0x14, 0)?; // st_nlink (64-bit; overlaps st_uid, both zero)
    store_stat_u32(state, buf, 0x18, 0)?; // st_uid
    store_stat_u32(state, buf, 0x1C, 0)?; // st_gid
    store_stat_u64(state, buf, 0x20, 0)?; // st_rdev
    store_stat_u64(state, buf, 0x2C, size)?; // st_size
    store_stat_u64(state, buf, 0x34, ST_BLKSIZE)?; // st_blksize (64-bit; upper overlaps st_blocks)
    store_stat_u64(state, buf, 0x38, 0)?; // st_blocks
    store_stat_u32(state, buf, 0x3C, 0)?; // padding
    store_stat_u64(state, buf, 0x40, 0)?; // st_atime
    store_stat_u64(state, buf, 0x44, 0)?; // st_atimensec
    store_stat_u64(state, buf, 0x48, 0)?; // st_mtime
    store_stat_u64(state, buf, 0x4C, 0)?; // st_mtimensec
    store_stat_u64(state, buf, 0x50, 0)?; // st_ctime
    store_stat_u64(state, buf, 0x54, 0)?; // st_ctimensec
    store_stat_u64(state, buf, 0x5C, 0)?; // st_ino (verification copy)
    Ok(())
}

/// ARM (32-bit EABI) `struct stat64` layout (LFS variant — the one
/// 32-bit glibc emits via `stat64`/`lstat64`/`fstat64`/`fstatat64`).
/// Mirrors `angr/procedures/linux_kernel/fstat64.py::_store_arm`
/// field-for-field. As with `write_i386_stat`, the value widths come
/// from `posix.fstat_with_result`'s `Stat` tuple (`st_dev`/`st_ino`/
/// `st_nlink`/`st_rdev`/`st_size`/`st_blksize`/`st_blocks`/times are
/// 64-bit; `st_mode`/`st_uid`/`st_gid` are 32-bit), NOT the packed
/// struct field widths, so several 64-bit stores spill into the next
/// field and are overwritten by the following store. Replaying Python's
/// exact order reproduces its byte output. The ARM layout differs from
/// i386 mainly around `st_size` (0x30 here vs 0x2C on i386, no padding
/// store) and it ends with the same "weird verification" `st_ino` copy
/// (0x60 vs i386's 0x5C). All fields but `st_mode` (concrete
/// `S_IFREG | 0o755`), `st_size` and `st_blksize` are zero — matching
/// the other Rust handlers (Python mints a symbolic `st_mode`; the Rust
/// path uses the caller-supplied concrete `mode` — see
/// `write_amd64_stat`).
fn write_arm_stat(
    state: &mut RustSimState,
    buf: u64,
    size: u64,
    mode: u32,
) -> Result<(), SyscallError> {
    store_stat_u64(state, buf, 0x00, 0)?; // st_dev
    store_stat_u64(state, buf, 0x0C, 0)?; // st_ino (64-bit; low half overlaps st_mode, both zero)
    store_stat_u32(state, buf, 0x10, mode)?; // st_mode
    store_stat_u64(state, buf, 0x14, 0)?; // st_nlink (64-bit; overlaps st_uid, both zero)
    store_stat_u32(state, buf, 0x18, 0)?; // st_uid
    store_stat_u32(state, buf, 0x1C, 0)?; // st_gid
    store_stat_u64(state, buf, 0x20, 0)?; // st_rdev
    store_stat_u64(state, buf, 0x30, size)?; // st_size
    store_stat_u64(state, buf, 0x38, ST_BLKSIZE)?; // st_blksize (64-bit; upper overlaps st_blocks)
    store_stat_u64(state, buf, 0x40, 0)?; // st_blocks
    store_stat_u64(state, buf, 0x48, 0)?; // st_atime
    store_stat_u64(state, buf, 0x4C, 0)?; // st_atimensec
    store_stat_u64(state, buf, 0x50, 0)?; // st_mtime
    store_stat_u64(state, buf, 0x54, 0)?; // st_mtimensec
    store_stat_u64(state, buf, 0x58, 0)?; // st_ctime
    store_stat_u64(state, buf, 0x5C, 0)?; // st_ctimensec
    store_stat_u64(state, buf, 0x60, 0)?; // st_ino (verification copy)
    Ok(())
}

/// MIPS32 O32 `struct stat64` layout (LFS variant — the one 32-bit MIPS
/// glibc emits via `stat64`/`lstat64`/`fstat64`/`fstatat64`). Mirrors
/// `angr/procedures/linux_kernel/fstat64.py::_store_mips32` field-for-field,
/// including its overlapping writes. Two things differ from the i386/ARM
/// writers:
///   1. `_store_mips32` uses `endness=self.state.arch.memory_endness`
///      (not a hardcoded `Iend_LE`), because MIPS32 is big-endian (MIPS32EL
///      is little). `memory_store` already applies the state's arch endness,
///      so plain concrete stores reproduce Python's byte order on both.
///   2. The MIPS layout writes NO `st_mode` and NO `st_nlink` field — angr's
///      `_store_mips32` simply omits them (the struct is flagged "NOT CORRECT"
///      upstream). So unlike the other Rust writers the `mode` argument is
///      ignored (hence `_mode`) — every field but `st_size` (0x30) and
///      `st_blksize` (0x50) is zero. The 96-bit zero stores at 0x04/0x24 plus
///      the overlapping field stores fully cover bytes 0x00..0x5F with no gap,
///      so no stale memory leaks where `st_mode` would sit.
///
/// As with the i386/ARM writers the value widths come from
/// `posix.fstat_with_result`'s `Stat` tuple (`st_dev`/`st_ino`/`st_rdev`/
/// `st_size`/`st_blksize`/`st_blocks`/times are 64-bit; `st_uid`/`st_gid`
/// are 32-bit), NOT the packed struct widths — replaying Python's exact
/// store order reproduces its byte output.
fn write_mips32_stat(
    state: &mut RustSimState,
    buf: u64,
    size: u64,
    _mode: u32,
) -> Result<(), SyscallError> {
    store_stat_u64(state, buf, 0x00, 0)?; // st_dev
    store_stat_zero96(state, buf, 0x04)?; // 96-bit zero pad (overlaps st_dev upper)
    store_stat_u64(state, buf, 0x10, 0)?; // st_ino
    store_stat_u32(state, buf, 0x18, 0)?; // st_uid
    store_stat_u32(state, buf, 0x1C, 0)?; // st_gid
    store_stat_u64(state, buf, 0x20, 0)?; // st_rdev
    store_stat_zero96(state, buf, 0x24)?; // 96-bit zero pad (overlaps st_rdev upper)
    store_stat_u64(state, buf, 0x30, size)?; // st_size
    store_stat_u64(state, buf, 0x38, 0)?; // st_atime
    store_stat_u64(state, buf, 0x3C, 0)?; // st_atimensec (overlaps st_atime upper)
    store_stat_u64(state, buf, 0x40, 0)?; // st_mtime
    store_stat_u64(state, buf, 0x44, 0)?; // st_mtimensec
    store_stat_u64(state, buf, 0x48, 0)?; // st_ctime
    store_stat_u64(state, buf, 0x4C, 0)?; // st_ctimensec
    store_stat_u64(state, buf, 0x50, ST_BLKSIZE)?; // st_blksize (64-bit; upper overlaps the zero pad below)
    store_stat_u32(state, buf, 0x54, 0)?; // 32-bit zero pad
    store_stat_u64(state, buf, 0x58, 0)?; // st_blocks
    Ok(())
}

/// Dispatch the per-arch `struct stat` writer. Callers must arch-guard
/// first (each stat-family handler accepts a slightly different arch set
/// — legacy `stat`/`lstat` are AMD64+X86+ARM only, `fstat`/`newfstatat`
/// add ARM64), so the `_` arm is defensive rather than a normal path.
fn write_stat_for_arch(
    state: &mut RustSimState,
    arch_name: &str,
    buf: u64,
    size: u64,
    mode: u32,
) -> Result<(), SyscallError> {
    match arch_name {
        "AMD64" => write_amd64_stat(state, buf, size, mode),
        "ARM64" => write_aarch64_stat(state, buf, size, mode),
        "X86" => write_i386_stat(state, buf, size, mode),
        "ARM" => write_arm_stat(state, buf, size, mode),
        "MIPS32" => write_mips32_stat(state, buf, size, mode),
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
pub(crate) struct NativeFstatSyscall;

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
        if arch_name != "AMD64"
            && arch_name != "ARM64"
            && arch_name != "X86"
            && arch_name != "ARM"
            && arch_name != "MIPS32"
        {
            return Err(SyscallError::Other(format!(
                "fstat: unsupported arch {arch_name} (only AMD64/ARM64/X86/ARM/MIPS32 have a Rust handler)"
            )));
        }

        // Look up fd — effective_size is the symbolic byte count when
        // bounded symbolic content is attached (angr-0xyq2), else the
        // concrete content length. Borrow ends before any memory_store.
        let size_opt = state.file_system_ref().effective_size(fd as u32);
        let Some(size) = size_opt else {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        };

        write_stat_for_arch(state, arch_name, buf, size as u64, S_IFREG_0755 as u32)?;
        Ok(SyscallOutcome::Continue { ret: 0 })
    }
}

/// `stat(pathname, statbuf) → 0 | -1` — resolve `pathname`, look it up
/// via `stat_lookup_follow` (walks the `FileSystem` symlink table, then
/// takes the target's content length from
/// `FileSystem::content_size_for_path`), write a per-arch `struct stat`
/// using the existing `write_amd64_stat` / `write_i386_stat` /
/// `write_arm_stat` helpers, return `0`. Unknown / empty path, dangling
/// link or over-long link chain returns `-1` with no buffer write. Arch
/// coverage:
/// AMD64 + X86 + ARM + MIPS32 (the latter three via their LFS `stat64`
/// number; ARM64's asm-generic ABI dropped legacy `stat` — only
/// `newfstatat` remains).
/// Unsupported arch returns `Other` BEFORE touching the path, mirroring
/// the `fstat` policy.
pub(crate) struct NativeStatSyscall;

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
        // back to Python anyway. ARM64 is absent because its
        // asm-generic ABI has no `stat` number at all (only
        // `newfstatat`), not because the writer is missing.
        let arch_name = state.arch().name();
        if arch_name != "AMD64" && arch_name != "X86" && arch_name != "ARM" && arch_name != "MIPS32"
        {
            return Err(SyscallError::Other(format!(
                "stat: unsupported arch {arch_name} (only AMD64/X86/ARM/MIPS32 have a Rust handler)"
            )));
        }

        let pathname_addr = extract_concrete_arg(&args[0], "stat pathname")?;
        let buf = extract_concrete_arg(&args[1], "stat statbuf")?;

        let path = read_path(state, pathname_addr, "stat")?;
        if path.is_empty() {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }

        let Some((size, mode)) = stat_lookup_follow(state.file_system_ref(), &path) else {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        };

        write_stat_for_arch(state, arch_name, buf, size, mode)?;
        Ok(SyscallOutcome::Continue { ret: 0 })
    }
}

/// `lstat(pathname, statbuf) → 0 | -1` — `stat`-shaped clone that does
/// NOT follow symlinks: resolve `pathname`, look it up via
/// `stat_lookup_nofollow` (registered symlink → `S_IFLNK | 0777` sized
/// to the raw target bytes; otherwise a known regular file →
/// `S_IFREG | 0755` sized by `FileSystem::content_size_for_path`),
/// write the per-arch `struct stat` via `write_stat_for_arch`, return
/// `0`. Unknown / empty path returns `-1` with no buffer write. Arch
/// coverage: AMD64 + X86 + ARM + MIPS32 (the latter three via their
/// LFS `lstat64` number); ARM64's asm-generic ABI dropped legacy
/// `lstat` entirely. The legacy numbers (i386/ARM 107, MIPS32 4107)
/// are deliberately left to Python — their pre-LFS `struct stat`
/// layout does not match what the `*64` writers emit (angr-9ke6b.226).
/// Unsupported
/// arch returns `Other` BEFORE touching the path, mirroring the `stat`
/// policy.
pub(crate) struct NativeLstatSyscall;

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
        if arch_name != "AMD64" && arch_name != "X86" && arch_name != "ARM" && arch_name != "MIPS32"
        {
            return Err(SyscallError::Other(format!(
                "lstat: unsupported arch {arch_name} (only AMD64/X86/ARM/MIPS32 have a Rust handler)"
            )));
        }

        let pathname_addr = extract_concrete_arg(&args[0], "lstat pathname")?;
        let buf = extract_concrete_arg(&args[1], "lstat statbuf")?;

        let path = read_path(state, pathname_addr, "lstat")?;
        if path.is_empty() {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }

        let Some((size, mode)) = stat_lookup_nofollow(state.file_system_ref(), &path) else {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        };

        write_stat_for_arch(state, arch_name, buf, size, mode)?;
        Ok(SyscallOutcome::Continue { ret: 0 })
    }
}

/// `newfstatat(dirfd, pathname, statbuf, flag) → 0 | -1` — `stat`-shaped
/// clone with `openat`-style dirfd handling and per-arch `struct stat`
/// layout. Mirrors `NativeStatSyscall`, plus the dirfd policy from
/// `NativeOpenatSyscall`: absolute paths and `AT_FDCWD` resolve via
/// `FileSystem`; relative paths with any other dirfd return `-1` (we
/// do not model directory fds). `AT_SYMLINK_NOFOLLOW` (0x100) in `flag`
/// selects `stat_lookup_nofollow` (`lstat` semantics — a registered
/// symlink reports its own `S_IFLNK | 0777` mode and target-length
/// size); without it, symlinks are walked to their target via
/// `stat_lookup_follow`. That bit is load-bearing on ARM64, whose
/// asm-generic ABI has no legacy `lstat` syscall, so `newfstatat` is the
/// only no-follow stat entry point there (angr-zueuw). A symbolic `flag`
/// is `SymbolicArgument` (falls back to Python) rather than an assumed
/// follow, since the two branches now report different `st_mode`s.
/// `AT_EMPTY_PATH` (0x1000) is still ignored — its "stat the dirfd
/// directly" semantics would require dispatching to
/// `NativeFstatSyscall(dirfd)`; an empty path stays `-1`.
/// Arch coverage: AMD64 + ARM64 + X86 + ARM (AMD64/
/// ARM64 use the 64-bit `struct stat`; X86/ARM use the LFS `struct
/// stat64` via `fstatat64`, mirroring `fstat64.py`). MIPS32 also uses the
/// LFS `struct stat64` via `fstatat64` (4293). Unsupported arch returns
/// `Other` BEFORE touching state.
pub(crate) struct NativeNewfstatatSyscall;

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
        if arch_name != "AMD64"
            && arch_name != "ARM64"
            && arch_name != "X86"
            && arch_name != "ARM"
            && arch_name != "MIPS32"
        {
            return Err(SyscallError::Other(format!(
                "newfstatat: unsupported arch {arch_name} (only AMD64/ARM64/X86/ARM/MIPS32 have a Rust handler)"
            )));
        }

        let dirfd = extract_concrete_arg(&args[0], "newfstatat dirfd")?;
        let pathname_addr = extract_concrete_arg(&args[1], "newfstatat pathname")?;
        let buf = extract_concrete_arg(&args[2], "newfstatat statbuf")?;
        // args[3] = flag. AT_SYMLINK_NOFOLLOW is honored below;
        // AT_EMPTY_PATH is not modeled (see doc comment). A symbolic
        // flag defers to Python — the bit changes the reported st_mode.
        let flag = extract_concrete_arg(&args[3], "newfstatat flag")?;
        let nofollow = flag & AT_SYMLINK_NOFOLLOW != 0;

        let path = read_path(state, pathname_addr, "newfstatat")?;
        if path.is_empty() {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }
        if !dirfd_allows(&path, dirfd) {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }

        let fs = state.file_system_ref();
        let looked_up = if nofollow {
            stat_lookup_nofollow(fs, &path)
        } else {
            stat_lookup_follow(fs, &path)
        };
        let Some((size, mode)) = looked_up else {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        };

        write_stat_for_arch(state, arch_name, buf, size, mode)?;
        Ok(SyscallOutcome::Continue { ret: 0 })
    }
}

#[cfg(test)]
#[path = "file_path_tests.rs"]
mod file_path_tests;
