//! Per-arch `struct stat` field constants and layout writers shared by
//! every handler in [`super::stat`].
//!
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

use crate::state::RustSimState;
use crate::symbolic::RustBV;
use crate::syscalls::SyscallError;

/// Concrete defaults for the `struct stat` fields. `fstat_with_result`
/// in Python returns a symbolic `st_mode` and `st_size` plus a
/// `st_blksize` of `0x400`; the Rust handler swaps `st_mode` for
/// `S_IFREG | 0o755` (regular file, rwxr-xr-x — the `mode` every
/// stat-family handler passes to `write_stat_for_arch` except `lstat`
/// on a symlink, which passes `S_IFLNK_0777`) and `st_size` for the
/// concrete length of the fd's backing buffer (`content_len`). Other
/// fields stay zero, matching the Python defaults.
pub(super) const S_IFREG_0755: u64 = 0o100_755;
pub(super) const ST_BLKSIZE: u64 = 0x400;

/// `st_mode` for a symlink: `S_IFLNK | 0777`. Real Linux always reports
/// `0777` permission bits on a symlink, so there is no `0755` analogue
/// here. Written by `NativeLstatSyscall` when the path is registered in
/// `FileSystem`'s symlink table (`FileSystem::readlink_target`).
pub(super) const S_IFLNK_0777: u64 = 0o120_777;

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
pub(super) fn write_stat_for_arch(
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
