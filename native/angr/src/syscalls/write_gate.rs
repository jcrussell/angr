//! The demotion-gate prologue every native write-path syscall shares.
//!
//! `write`, `writev` and `pwrite64` differ only in how they name their length
//! argument, how they gather the bytes, and which `FileSystem` sink they hand
//! them to. The *ordering* of the checks in front of that — specifically where
//! `demote_symbolic_content` sits relative to the concrete-zero short-circuit
//! and to the `?` on the remaining arguments — is one correctness invariant
//! shared by all three (angr-0xyq2 A3/A4). It used to be hand-repeated in each
//! handler with only comment cross-references holding the three copies in
//! lockstep (angr-fs8kb.54).
//!
//! [`write_path_gate`] is that prologue, written once:
//!
//! 1. Extract the fd. A *symbolic* fd could name any bounded symbolic file, so
//!    demote them all (`FileSystem::demote_all_symbolic_content`) before
//!    bouncing — A4; O(1) when no symbolic file is attached.
//! 2. fd 0 (stdin) is Python's symbolic-packet model; bounce.
//! 3. An fd not open in the Rust `FileSystem` is Python's symbolic-file model;
//!    bounce.
//! 4. A **concrete** zero length is a POSIX no-op and must NOT demote, so it
//!    short-circuits ahead of the gate — A3. A *symbolic* length deliberately
//!    falls through to step 5, so the fd demotes before its bounce.
//! 5. Run the gate: an fd carrying symbolic content demotes and bounces to
//!    Python, which owns the symbolic-content write path.
//!
//! Everything after that is the per-syscall body, supplied as a closure. It
//! runs strictly after the gate, so it is free to `?` on its remaining
//! arguments — those bounces are already demoted. Each body ends by rechecking
//! its own sink's return (`RustSimState::write_fd` / `FileSystem::write_at`),
//! whose "still symbolic" answer is unreachable after step 5 but kept as
//! choke-point insurance; `WriteTarget::demoted` builds the one fallback
//! message both that recheck and the gate itself report.

use super::{SyscallError, SyscallOutcome};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// The validated write target handed to a [`write_path_gate`] body.
pub(crate) struct WriteTarget {
    /// The fd as the guest passed it — concrete, nonzero, and open in the Rust
    /// `FileSystem`. Reported verbatim in the fallback messages.
    fd: u64,
    /// The syscall's name, for those messages.
    name: &'static str,
}

impl WriteTarget {
    /// The `FileSystem` table key for this fd.
    pub(crate) fn fd_u32(&self) -> u32 {
        self.fd as u32
    }

    /// The fallback error every write-path handler reports once the fd is known
    /// to carry symbolic content Python must own — both from the gate itself
    /// and from the post-write choke-point recheck (see the module docs).
    pub(crate) fn demoted(&self) -> SyscallError {
        SyscallError::Other(format!(
            "{} to fd={} with symbolic content falls back to Python (demoted)",
            self.name, self.fd
        ))
    }
}

/// Run the shared write-path prologue (steps 1-5 in the module docs) for
/// `name`, then hand off to `body`.
///
/// `fd_arg` and `len_arg` are the syscall's fd and length arguments — `count`
/// for `write`, `iovcnt` for `writev`, `nbyte` for `pwrite64`. `body` returns
/// the value to place in the return register.
pub(crate) fn write_path_gate<F>(
    state: &mut RustSimState,
    name: &'static str,
    fd_arg: &RustBV,
    len_arg: &RustBV,
    body: F,
) -> Result<SyscallOutcome, SyscallError>
where
    F: FnOnce(&mut RustSimState, &WriteTarget) -> Result<u64, SyscallError>,
{
    // Step 1. `RustBV::as_u64` rather than `extract_concrete_arg` so the
    // `"{name} fd"` context is only formatted on the symbolic path — `writev`
    // is glibc stdio's flush path, hit on essentially every printf.
    let Some(fd) = fd_arg.as_u64() else {
        state.file_system().demote_all_symbolic_content();
        return Err(SyscallError::SymbolicArgument(format!("{name} fd")));
    };
    // Step 2.
    if fd == 0 {
        return Err(SyscallError::Other(format!(
            "{name} to fd=0 (stdin) falls back to Python"
        )));
    }
    let target = WriteTarget { fd, name };
    // Step 3.
    if !state.file_system_ref().is_open(target.fd_u32()) {
        return Err(SyscallError::Other(format!(
            "{name} to fd={fd} (not open in Rust FileSystem) falls back to Python"
        )));
    }
    // Step 4. A symbolic length is `None` here and falls through on purpose.
    if len_arg.as_u64() == Some(0) {
        return Ok(SyscallOutcome::Continue { ret: 0 });
    }
    // Step 5 — write-demotion (angr-0xyq2 Phase 2), see procedures/write.rs.
    if state.file_system().demote_symbolic_content(target.fd_u32()) {
        return Err(target.demoted());
    }
    let ret = body(state, &target)?;
    Ok(SyscallOutcome::Continue { ret })
}
