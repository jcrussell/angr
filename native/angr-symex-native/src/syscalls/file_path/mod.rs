//! File-path syscall handlers.
//!
//! The handler families live in sibling submodules, each carrying the
//! prose that used to sit in this header:
//!
//! * [`open_close`] — `open` / `openat` / `close`
//! * [`access`] — `access` / `faccessat`
//! * [`readlink`] — `readlink` / `readlinkat`
//! * [`stat`] — `fstat` / `stat` / `lstat` / `newfstatat`
//! * [`stat_layouts`] — the per-arch `struct stat` writers `stat` shares
//!
//! This file keeps only what more than one family needs: the `AT_FDCWD`
//! sentinel, the shared `*at` dirfd policy ([`dirfd_allows`]) and the
//! NUL-terminated path reader ([`read_path`]).

mod access;
mod open_close;
mod readlink;
mod stat;
mod stat_layouts;

pub(crate) use access::{NativeAccessSyscall, NativeFaccessatSyscall};
pub(crate) use open_close::{NativeCloseSyscall, NativeOpenSyscall, NativeOpenatSyscall};
pub(crate) use readlink::{NativeReadlinkSyscall, NativeReadlinkatSyscall};
pub(crate) use stat::{
    NativeFstatSyscall, NativeLstatSyscall, NativeNewfstatatSyscall, NativeStatSyscall,
};

use crate::procedures::strings::MAX_PATH_SCAN as MAX_PATH_LEN;
use crate::state::RustSimState;
use crate::syscalls::SyscallError;

/// `AT_FDCWD` in unsigned 32-bit form (-100 reinterpreted). Linux's
/// `openat(2)` treats this as "use the current working directory" for
/// relative paths. `procedures/linux_kernel/openat.py` also matches
/// against this exact unsigned value. `dirfd` is a C `int`, so it is
/// compared at 32 bits: aarch64 glibc, musl and Go pass it sign-extended
/// to the full register (angr-vmi2l.135).
const AT_FDCWD_UNSIGNED: u32 = 0xffff_ff9c;

// `NEG_ONE` is the kernel-ABI failure return for `open` / `openat` /
// `close` here, mirroring `procedures/posix/open.py::run` (`return -1`).
use super::errno::NEG_ONE;

/// The shared `*at` dirfd policy: `Ok` when this handler can resolve
/// `path` on its own, i.e. the path is absolute (dirfd irrelevant) or
/// the low 32 bits of `dirfd` are the `AT_FDCWD` sentinel. A relative
/// path against a real dirfd is not modeled — Rust's `FileSystem` has no
/// per-fd directory — so it returns `Other` and Python decides
/// (angr-vmi2l.135).
///
/// Every `*at` handler in this module (`openat`, `faccessat`,
/// `readlinkat`, `newfstatat`) must gate on this, so that adding real
/// dirfd support is a one-site change instead of four (angr-c7xno.82).
fn dirfd_allows(path: &str, dirfd: u64, syscall: &str) -> Result<(), SyscallError> {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "dirfd is a C int: the kernel reads only the low 32 bits"
    )]
    let dirfd = dirfd as u32;
    if path.starts_with('/') || dirfd == AT_FDCWD_UNSIGNED {
        Ok(())
    } else {
        Err(SyscallError::Other(format!(
            "{syscall} of relative {path:?} against dirfd {dirfd}: not modeled, Python decides"
        )))
    }
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

test_submod!("../file_path_tests_support.rs" => file_path_tests_support);
test_submod!("../file_path_tests.rs" => file_path_tests);
test_submod!("../file_path_readlink_tests.rs" => file_path_readlink_tests);
test_submod!("../file_path_fstat_tests.rs" => file_path_fstat_tests);
test_submod!("../file_path_stat_tests.rs" => file_path_stat_tests);
test_submod!("../file_path_lstat_tests.rs" => file_path_lstat_tests);
test_submod!("../file_path_newfstatat_tests.rs" => file_path_newfstatat_tests);
test_submod!("../file_path_symlink_tests.rs" => file_path_symlink_tests);
