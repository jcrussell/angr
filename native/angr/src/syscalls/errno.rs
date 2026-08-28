//! Kernel-ABI negative-errno return values, shared by the syscall handlers.
//!
//! Linux syscalls signal failure by returning `-errno` in the return
//! register. Handlers hand that value back as
//! `SyscallOutcome::Continue { ret }`; the dispatcher
//! (`exploration::core_outcome_handlers::handle_syscall_core`) truncates it
//! to `arch().bits()` when it writes the return register, so every constant
//! here is the 64-bit two's-complement pattern regardless of the guest arch.
//!
//! These lived as per-file `const NEG_*` definitions in `file_path.rs`,
//! `file_descriptor.rs`, `directory.rs`, `sigaction.rs` and `sim_time.rs`
//! until angr-0jh0j.60. Values never diverged, but each new failure mode
//! had no existing constant to reach for and minted another one-off; add
//! the missing errno here instead (`neg` keeps that a one-liner).

/// The 64-bit two's-complement pattern for `-errno`.
const fn neg(errno: u32) -> u64 {
    (-(errno as i64)) as u64
}

/// `-1` — the generic failure return used where the mirrored Python
/// `SimProcedure` returns a bare `-1` rather than a specific errno (e.g.
/// `procedures/posix/open.py::run`), and where the syscall's own ABI
/// makes `-1` the error sentinel rather than a negated errno (the
/// `sim_time.rs` clock family).
pub(crate) const NEG_ONE: u64 = neg(1);

/// `-EBADF` (9) — bad file descriptor.
pub(crate) const NEG_EBADF: u64 = neg(9);

/// `-EFAULT` (14) — bad address passed by the guest.
pub(crate) const NEG_EFAULT: u64 = neg(14);

/// `-EINVAL` (22) — invalid argument.
pub(crate) const NEG_EINVAL: u64 = neg(22);

/// `-ENOTTY` (25) — inappropriate ioctl for device (non-terminal fd).
pub(crate) const NEG_ENOTTY: u64 = neg(25);

/// `-ERANGE` (34) — result too large for the caller's buffer.
pub(crate) const NEG_ERANGE: u64 = neg(34);

test_submod!("errno_tests.rs" => errno_tests);
