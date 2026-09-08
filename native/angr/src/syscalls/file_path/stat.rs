//! `fstat` / `stat` / `lstat` / `newfstatat` syscall handlers.
//!
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
//! point and `stat_layouts::STAT_LAYOUT_WRITERS` the table it and every
//! handler's `require_stat_arch` guard read; an arch absent from that
//! table returns `Other` and falls through to Python. On i386 and MIPS32 the legacy pre-LFS numbers
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

// lstat(pathname, statbuf) → long — see NativeLstatSyscall impl below.
// newfstatat(dfd, filename, statbuf, flag) → long — see NativeNewfstatatSyscall impl below.

use super::stat_layouts::{S_IFLNK_0777, S_IFREG_0755, require_stat_arch, write_stat_for_arch};
use super::{NEG_ONE, dirfd_allows, read_path};
use crate::state::RustSimState;
use crate::symbolic::RustBV;
use crate::syscalls::{NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg};

/// `AT_SYMLINK_NOFOLLOW` — `newfstatat`'s flag bit that selects
/// `lstat` semantics. ARM64's asm-generic ABI has no legacy `lstat`
/// syscall, so glibc's `lstat()` there lowers to
/// `fstatat(AT_FDCWD, path, buf, AT_SYMLINK_NOFOLLOW)`; honoring this
/// bit is what makes `NativeNewfstatatSyscall` the ARM64 equivalent of
/// `NativeLstatSyscall` (angr-zueuw).
pub(super) const AT_SYMLINK_NOFOLLOW: u64 = 0x100;

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

        // Arch check first: avoid mutating state.memory if we will fall
        // back to Python anyway. `allow_arm64 = true` — `fstat` takes an
        // fd, so it exists on ARM64's asm-generic ABI (80) like everywhere
        // else.
        let arch_name = state.arch().name();
        require_stat_arch("fstat", arch_name, true)?;

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
        // back to Python anyway. `allow_arm64 = false` because that ABI
        // has no `stat` number at all (only `newfstatat`), not because
        // the writer is missing.
        let arch_name = state.arch().name();
        require_stat_arch("stat", arch_name, false)?;

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
        // `allow_arm64 = false`: same reason as `stat` above — no legacy
        // `lstat` number on the asm-generic ABI.
        let arch_name = state.arch().name();
        require_stat_arch("lstat", arch_name, false)?;

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
        // `allow_arm64 = true` — one arch wider than `stat` / `lstat`,
        // since `newfstatat` (79) is the only stat-shaped syscall on
        // ARM64's asm-generic ABI.
        let arch_name = state.arch().name();
        require_stat_arch("newfstatat", arch_name, true)?;

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
