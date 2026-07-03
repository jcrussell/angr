//! amd64 read syscall handler.
//!
//! Mirrors `procedures/read.rs::NativeRead`. Handles `sys_read(fd, buf, count)`
//! for:
//!   - `fd == 0` (stdin) by writing `count` fresh symbolic bytes into memory
//!     and returning `count` in rax.
//!   - any other fd open in the Rust `FileSystem` that has remaining
//!     concrete content (pos < content_len). Serves bytes from the FS buffer
//!     and advances the position.
//!   - fds with bounded symbolic content (`content_sym`, angr-0xyq2
//!     Phase 2): the registered per-byte BVs are stored to the buffer
//!     natively via `FileSystem::read_sym`; EOF returns 0. Counts are
//!     clamped to `MAX_SYMFILE_SERVE_SIZE` (= the export cap, so any
//!     registered file serves in one call, matching Python), never
//!     bounced — a fallback would split the position cursor (angr-8j16).
//!
//! Falls back to the Python `_handle_syscall_callback` path
//! (`state.posix.get_fd(fd).read(...)`) for: symbolic args, counts beyond
//! `MAX_READ_SIZE` (stdin / concrete-content fds), fds not open in Rust's
//! `FileSystem`, and non-stdin open fds with neither concrete nor symbolic
//! content (Python's symbolic-file model owns those reads).
//!
//! The dirty pages produced here are picked up by `_replay_rust_dirty_pages`
//! (rust_state_sync.py) on the next Python callback, so a downstream Python
//! SimProc (e.g. strcmp) observes the bytes — see memory
//! `invariant-3tek2-replay-ordering`.
//!
//! See `procedures/read.rs` for the fd-table sync invariant (angr-8j16).

use std::sync::atomic::{AtomicU64, Ordering};

use super::{NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg};
use crate::procedures::strings::write_bv_bytes;
use crate::state::MAX_SYMFILE_SERVE_SIZE;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const MAX_READ_SIZE: u64 = 4096;

/// Counter for unique stdin variable names. Independent of the
/// `procedures/read.rs` counter so name collisions only happen when the
/// same callsite is hit twice (in which case they're already disambiguated
/// by exploration order, like the procedure-level counter).
static SYS_READ_COUNTER: AtomicU64 = AtomicU64::new(0);

pub struct NativeReadSyscall;

impl NativeSyscall for NativeReadSyscall {
    fn name(&self) -> &'static str {
        "read"
    }

    fn num_args(&self) -> usize {
        3
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        if args.len() < 3 {
            return Err(SyscallError::Other(format!(
                "read expected 3 args, got {}",
                args.len()
            )));
        }
        let fd = extract_concrete_arg(&args[0], "read fd")?;
        let buf = extract_concrete_arg(&args[1], "read buf")?;
        let count = extract_concrete_arg(&args[2], "read count")?;

        if count == 0 {
            return Ok(SyscallOutcome::Continue { ret: 0 });
        }

        if fd == 0 {
            if count > MAX_READ_SIZE {
                return Err(SyscallError::Other(format!(
                    "read count {} exceeds limit",
                    count
                )));
            }
            return read_symbolic(state, buf, count, "sys_read");
        }

        let fd_u32 = fd as u32;
        let (open, content_len) = match state.file_system_ref().fd_info(fd_u32) {
            Some((_, _, _, len, is_open)) => (is_open, len),
            None => (false, 0),
        };
        if !open {
            return Err(SyscallError::Other(format!(
                "read from fd={} (not open in Rust FileSystem) falls back to Python",
                fd
            )));
        }
        // Bounded symbolic file content (angr-0xyq2 Phase 2): serve the
        // per-byte BVs registered for this fd's path natively, advancing the
        // position. Checked before the symbolic-stream branch — a finite
        // symbolic *file* returns 0 at EOF rather than minting fresh bytes.
        // Clamped to MAX_SYMFILE_SERVE_SIZE (= the export cap; whole file in
        // one call, matching Python) instead of bounced: a Python fallback
        // would split the position cursor, since natively-minted fds are not
        // mirrored into Python (angr-8j16).
        let clamped = count.min(MAX_SYMFILE_SERVE_SIZE) as usize;
        if let Some(sym_bytes) = state.file_system().read_sym(fd_u32, clamped) {
            let n = sym_bytes.len();
            write_bv_bytes(state, buf, sym_bytes)?;
            if n > 0 {
                crate::symbolic::record_symfile_read_native();
            }
            return Ok(SyscallOutcome::Continue { ret: n as u64 });
        }
        if count > MAX_READ_SIZE {
            return Err(SyscallError::Other(format!(
                "read count {} exceeds limit",
                count
            )));
        }
        if content_len == 0 {
            // No concrete bytes left. A symbolic-stream fd (the stdin model)
            // mints fresh symbolic bytes natively; any other fd defers to
            // Python's symbolic-file model.
            if state.file_system_ref().is_symbolic(fd_u32) {
                return read_symbolic(state, buf, count, &format!("sys_read_fd{fd}"));
            }
            return Err(SyscallError::Other(format!(
                "read from fd={} has no concrete content; falling back to Python",
                fd
            )));
        }

        let bytes = state.file_system().read(fd_u32, count as usize);
        let n = bytes.len();
        for (i, b) in bytes.iter().enumerate() {
            state.memory_store(buf.wrapping_add(i as u64), RustBV::concrete(*b as u128, 8))?;
        }
        Ok(SyscallOutcome::Continue { ret: n as u64 })
    }
}

/// Mint `count` fresh symbolic bytes named `<prefix>_<id>_<i>` into `buf` and
/// return `count` in rax. Shared by the stdin path (fd 0) and symbolic-stream
/// fds. The `prefix` disambiguates per-fd names; `SYS_READ_COUNTER` keeps each
/// call's bytes uniquely named even at the same callsite.
fn read_symbolic(
    state: &mut RustSimState,
    buf: u64,
    count: u64,
    prefix: &str,
) -> Result<SyscallOutcome, SyscallError> {
    let read_id = SYS_READ_COUNTER.fetch_add(1, Ordering::Relaxed);

    let sym_bytes: Vec<RustBV> = {
        let ctx = state.solver().borrow();
        (0..count)
            .map(|i| {
                let name = format!("{}_{}_{}", prefix, read_id, i);
                RustBV::symbolic(&ctx, &name, 8)
            })
            .collect()
    };

    write_bv_bytes(state, buf, sym_bytes)?;

    Ok(SyscallOutcome::Continue { ret: count })
}

#[cfg(test)]
#[path = "read_tests.rs"]
mod read_tests;
