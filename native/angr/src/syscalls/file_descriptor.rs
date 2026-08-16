//! File-descriptor control syscall handlers — `fcntl`, `fcntl64`,
//! `ioctl`, `pipe`, `pipe2` (the stub-fallthrough subset of
//! `angr-0hif.5`) and `dup`, `dup2`, `dup3` (`angr-vp19`).
//!
//! ## fcntl / fcntl64 (angr-aig2)
//!
//! Dispatched on the concrete `cmd` arg. The four "trivial getter /
//! setter" ops are handled natively without touching Python:
//!
//! * `F_GETFD` (1): close-on-exec flags. We do not model FD_CLOEXEC,
//!   so the answer is always `0`.
//! * `F_SETFD` (2): set close-on-exec flags. Ignored; returns `0`
//!   (success).
//! * `F_GETFL` (3): file status flags. Reads the `FdFlags` (RO/WO/RW)
//!   from the Rust `FileSystem` and returns the matching `O_RDONLY` /
//!   `O_WRONLY` / `O_RDWR` value. `-EBADF` when the fd is not open.
//! * `F_SETFL` (4): set file status flags. Ignored; returns `0`
//!   (success). The new flags are not stored — `FdFlags` only models
//!   the access mode, not the per-syscall blocking / append bits.
//!
//! Unrecognized `cmd` values (including `F_DUPFD`, locks, leases,
//! pipe-size getters / setters, ...) fall through to a fresh symbolic
//! `RustBV` of width `arch().bits()`, matching the prior
//! `syscall_stub_fcntl` behavior. A symbolic `cmd` also falls back to
//! the stub via `SyscallError::SymbolicArgument`.
//!
//! `posix/fcntl.py` defines a `fcntl` `SimProcedure` but
//! `angr/procedures/definitions/linux_kernel.py` does NOT bind it
//! into the `SimSyscallLibrary` (only `dup` from `posix/` is wired
//! into the kernel table). So pure-Python angr also hits
//! `syscall_stub` on every `fcntl` — the Rust handler is strictly
//! more informative than baseline for the four handled cmds and
//! identical to baseline for everything else.
//!
//! ## ioctl (angr-aig2)
//!
//! Dispatched on the concrete `cmd` arg. We handle one terminal-
//! ioctl that real binaries call early in `main()` to size a tty
//! and degrade gracefully on `ENOTTY`:
//!
//! * `TIOCGWINSZ`: get window size. Our model has no terminal
//!   attached to fd 0/1/2 (they back onto byte streams via
//!   `FileSystem`), so we return `-ENOTTY` (`-25`) verbatim. The
//!   user-supplied `struct winsize *` buffer is left untouched —
//!   POSIX allows this on error.
//!
//! `cmd` differs per-arch: `0x5413` on x86 / amd64 / arm / aarch64,
//! `0x40087468` on the MIPS family. The dispatcher reads the right
//! constant from `state.arch().name()` so a single handler covers
//! every registered arch.
//!
//! `FIONREAD` is deliberately NOT handled — its third argument is an
//! `int *` that we would have to write a byte count back into, and
//! the existing native procedures handle that pattern but the
//! syscall-side memory-write helpers are not yet plumbed through
//! here. Falls back to the symbolic stub.
//!
//! Unrecognized `cmd` (including symbolic) falls through to
//! `RustBV::symbolic` of width `arch().bits()`.
//!
//! ## pipe / pipe2
//!
//! No Python `SimProcedure`; both fall through to `syscall_stub`
//! via a fresh symbolic return.
//!
//! ## dup / dup2 / dup3 (angr-vp19)
//!
//! `dup`, `dup2`, `dup3` mutate the per-state FD table. We model that
//! directly on `RustSimState::file_system()` rather than syncing back
//! into Python's `state.posix.fd` — same precedent as
//! `procedures/fileops::NativeDup` / `NativeDup2`, which already
//! manipulate the Rust-side `FileSystem` without touching the Python
//! plugin. Like Python `procedures/posix/dup.py`, `FileSystem::dup`
//! allocates the lowest free fd, so a closed slot is reused (angr-9ke6b.119
//! replaced the earlier monotonic `next_fd` allocator, which diverged from
//! both POSIX and Python once an exploration closed an fd). The one
//! remaining divergence is Python's multi-gap scan quirk — see
//! `FileSystem::dup`'s doc comment.
//!
//! Error returns use the kernel ABI: `-EBADF` (negative errno) on
//! invalid `oldfd` and on out-of-range `newfd` for `dup2`/`dup3`
//! (matches Python's 4096-fd ulimits ceiling).

use super::{
    NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg, stub_syscall,
    symbolic_outcome,
};
use crate::state::{MAX_FD, RustSimState};
use crate::symbolic::RustBV;

use super::errno::{NEG_EBADF, NEG_ENOTTY};

/// Upper bound on `newfd` that Python `procedures/posix/dup.py` enforces
/// (the default ulimits ceiling). Out-of-range values return EBADF.
///
/// Shares the [`MAX_FD`] value with [`FileSystem::dup2`](crate::state::FileSystem::dup2),
/// which enforces the same cap for the callers that do not come through this
/// layer (the native `dup2` SimProcedure) — angr-03vl4.52. The check stays
/// here as well because Python's ordering puts the `oldfd == newfd` success
/// *before* it.
const NEWFD_LIMIT: u64 = MAX_FD as u64;

/// Shared `dup2`/`dup3` body. Mirrors angr's Python `dup2`/`dup3`
/// (`procedures/posix/dup.py`) check ordering **exactly** for parity:
///   1. `oldfd` not open        → `-EBADF`
///   2. `oldfd == newfd`        → return `newfd` (no-op success, *before* the
///      range check — so `dup2(fd, fd)` on an open high fd succeeds even when
///      `fd >= NEWFD_LIMIT`, matching Python)
///   3. `newfd >= NEWFD_LIMIT`  → `-EBADF`
///   4. alias `newfd` → `oldfd`
///
/// NOTE: real `dup3(2)` returns `-EINVAL` when `oldfd == newfd`, but angr's
/// Python `dup3` does not — it returns `newfd` like `dup2`. The Rust engine
/// targets Python-engine parity, so we deliberately match Python (no EINVAL)
/// here; adding EINVAL would diverge from the reference engine. See
/// bd `angr-1yge9.7`.
fn dup2_body(state: &mut RustSimState, oldfd: u64, newfd: u64) -> u64 {
    if !state.file_system().is_open(oldfd as u32) {
        return NEG_EBADF;
    }
    if oldfd == newfd {
        return newfd;
    }
    if newfd >= NEWFD_LIMIT {
        return NEG_EBADF;
    }
    match state.file_system().dup2(oldfd as u32, newfd as u32) {
        Some(fd) => fd as u64,
        None => NEG_EBADF,
    }
}

// fcntl / ioctl `cmd` constants — uniform across Linux ABIs we support.
const F_GETFD: u64 = 1;
const F_SETFD: u64 = 2;
const F_GETFL: u64 = 3;
const F_SETFL: u64 = 4;

/// `TIOCGWINSZ` ioctl number on x86 / amd64 / arm / aarch64 (the
/// `asm-generic/ioctls.h` family).
const TIOCGWINSZ_GENERIC: u64 = 0x5413;
/// `TIOCGWINSZ` ioctl number on MIPS (`asm-mips/ioctls.h` flavor —
/// embeds the struct size into the upper bits).
const TIOCGWINSZ_MIPS: u64 = 0x40087468;

/// Per-arch `TIOCGWINSZ` lookup. Returns `None` for arches we have not
/// audited so the dispatcher falls through to the symbolic stub rather
/// than acting on a guessed constant. Arch names follow the
/// `Arch::name()` strings ("AMD64", "X86", "ARM", "ARM64", "MIPS32",
/// "MIPS64").
fn tiocgwinsz_for_arch(arch_name: &str) -> Option<u64> {
    match arch_name {
        "AMD64" | "X86" | "ARM" | "ARM64" => Some(TIOCGWINSZ_GENERIC),
        "MIPS32" | "MIPS64" => Some(TIOCGWINSZ_MIPS),
        _ => None,
    }
}

/// Shared dispatch for `fcntl` / `fcntl64`. Both syscalls take
/// `(fd, cmd, arg)` and differ only in the kernel-side handling of
/// the LFS `arg` payload (locks); the four trivial getter / setter
/// cmds we handle do not read `arg`, so the dispatch is identical.
fn fcntl_dispatch(
    state: &RustSimState,
    args: &[RustBV],
    stub_name: &'static str,
) -> Result<SyscallOutcome, SyscallError> {
    let fd = extract_concrete_arg(&args[0], "fcntl fd")?;
    let cmd = extract_concrete_arg(&args[1], "fcntl cmd")?;
    // args[2] = arg — unused by the handled cmds. For F_SETFL it
    // would be the new flags; we accept any value (including
    // symbolic) without forcing concretization.

    match cmd {
        F_GETFD | F_SETFD => Ok(SyscallOutcome::Continue { ret: 0 }),
        F_GETFL => {
            let ret = match state.file_system_ref().fd_info(fd as u32) {
                Some((_, _, flags, _, is_open)) if is_open => flags as u64,
                _ => NEG_EBADF,
            };
            Ok(SyscallOutcome::Continue { ret })
        }
        F_SETFL => Ok(SyscallOutcome::Continue { ret: 0 }),
        _ => Ok(symbolic_outcome(state, stub_name)),
    }
}

/// `fcntl(fd, cmd, arg) → int` — see module doc for handled `cmd`s.
pub(crate) struct NativeFcntlSyscall;

impl NativeSyscall for NativeFcntlSyscall {
    fn name(&self) -> &'static str {
        "fcntl"
    }

    fn num_args(&self) -> usize {
        3
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        fcntl_dispatch(state, args, "syscall_stub_fcntl")
    }
}

/// `fcntl64(fd, cmd, arg) → long` — LFS-flavored variant on 32-bit
/// arches; shares the same dispatch as `fcntl` for the trivial cmds.
pub(crate) struct NativeFcntl64Syscall;

impl NativeSyscall for NativeFcntl64Syscall {
    fn name(&self) -> &'static str {
        "fcntl64"
    }

    fn num_args(&self) -> usize {
        3
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        fcntl_dispatch(state, args, "syscall_stub_fcntl64")
    }
}

/// `ioctl(fd, cmd, arg) → int` — see module doc. Currently handles
/// `TIOCGWINSZ` → `-ENOTTY` (no terminal model); everything else
/// falls back to a fresh symbolic return.
pub(crate) struct NativeIoctlSyscall;

impl NativeSyscall for NativeIoctlSyscall {
    fn name(&self) -> &'static str {
        "ioctl"
    }

    fn num_args(&self) -> usize {
        3
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        let _fd = extract_concrete_arg(&args[0], "ioctl fd")?;
        let cmd = extract_concrete_arg(&args[1], "ioctl cmd")?;
        // args[2] = arg — unused by the handled cmds; accept any value
        // (including symbolic) without forcing concretization.

        let arch_name = state.arch().name();
        if Some(cmd) == tiocgwinsz_for_arch(arch_name) {
            return Ok(SyscallOutcome::Continue { ret: NEG_ENOTTY });
        }
        Ok(symbolic_outcome(state, "syscall_stub_ioctl"))
    }
}

// pipe(fildes*) → int — no Python SimProcedure.
stub_syscall!(NativePipeSyscall, "pipe", "syscall_stub_pipe", 1);
// pipe2(fildes*, flags) → int — no Python SimProcedure.
stub_syscall!(NativePipe2Syscall, "pipe2", "syscall_stub_pipe2", 2);

/// `dup(oldfd) → newfd` — allocate a fresh fd that aliases `oldfd`.
/// Returns `-EBADF` if `oldfd` is not open.
pub(crate) struct NativeDupSyscall;

impl NativeSyscall for NativeDupSyscall {
    fn name(&self) -> &'static str {
        "dup"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        let oldfd = extract_concrete_arg(&args[0], "oldfd")?;
        let ret = match state.file_system().dup(oldfd as u32) {
            Some(newfd) => newfd as u64,
            None => NEG_EBADF,
        };
        Ok(SyscallOutcome::Continue { ret })
    }
}

/// `dup2(oldfd, newfd) → newfd` — make `newfd` an alias of `oldfd`,
/// closing the previous `newfd` if it was open. Returns `-EBADF` if
/// `oldfd` is closed or `newfd` is out of range.
pub(crate) struct NativeDup2Syscall;

impl NativeSyscall for NativeDup2Syscall {
    fn name(&self) -> &'static str {
        "dup2"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        let oldfd = extract_concrete_arg(&args[0], "oldfd")?;
        let newfd = extract_concrete_arg(&args[1], "newfd")?;
        let ret = dup2_body(state, oldfd, newfd);
        Ok(SyscallOutcome::Continue { ret })
    }
}

/// `dup3(oldfd, newfd, flags) → newfd` — like `dup2`, with a `flags`
/// argument (only `O_CLOEXEC = 0x80000` defined). We do not model
/// close-on-exec, so the flag is accepted and ignored — including a
/// symbolic one, which never forces a Python fallback.
pub(crate) struct NativeDup3Syscall;

impl NativeSyscall for NativeDup3Syscall {
    fn name(&self) -> &'static str {
        "dup3"
    }

    fn num_args(&self) -> usize {
        3
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        let oldfd = extract_concrete_arg(&args[0], "oldfd")?;
        let newfd = extract_concrete_arg(&args[1], "newfd")?;
        // args[2] = flags — O_CLOEXEC modeling is out of scope, so the
        // value is never read; accept any value (including symbolic)
        // without forcing concretization, same as fcntl_dispatch/ioctl.
        // Parity with Python `dup3` (no EINVAL for oldfd==newfd — see dup2_body).
        let ret = dup2_body(state, oldfd, newfd);
        Ok(SyscallOutcome::Continue { ret })
    }
}

test_submod!("file_descriptor_tests.rs" => file_descriptor_tests);
