//! amd64 read syscall handler.
//!
//! Mirrors `procedures/read.rs::NativeRead`. Handles `sys_read(fd, buf, count)`
//! for:
//!   - `fd == 0` (stdin) by writing `count` fresh symbolic bytes into memory
//!     and returning `count` in rax.
//!   - any other fd open in the Rust `FileSystem` that has remaining
//!     concrete content (pos < content_len). Serves bytes from the FS buffer
//!     and advances the position.
//!
//! Falls back to the Python `_handle_syscall_callback` path
//! (`state.posix.get_fd(fd).read(...)`) for: symbolic args, counts beyond
//! `MAX_READ_SIZE`, fds not open in Rust's `FileSystem`, and non-stdin open
//! fds with empty / fully-consumed content (Python's symbolic-file model
//! owns those reads).
//!
//! The dirty pages produced here are picked up by `_replay_rust_dirty_pages`
//! (rust_state_sync.py) on the next Python callback, so a downstream Python
//! SimProc (e.g. strcmp) observes the bytes — see memory
//! `invariant-3tek2-replay-ordering`.
//!
//! See `procedures/read.rs` for the fd-table sync invariant (angr-8j16).

use std::sync::atomic::{AtomicU64, Ordering};

use super::{NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg};
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

        if count > MAX_READ_SIZE {
            return Err(SyscallError::Other(format!(
                "read count {} exceeds limit",
                count
            )));
        }

        if count == 0 {
            return Ok(SyscallOutcome::Continue { ret: 0 });
        }

        if fd == 0 {
            return read_stdin_symbolic(state, buf, count);
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
        if content_len == 0 {
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

fn read_stdin_symbolic(
    state: &mut RustSimState,
    buf: u64,
    count: u64,
) -> Result<SyscallOutcome, SyscallError> {
    let read_id = SYS_READ_COUNTER.fetch_add(1, Ordering::Relaxed);

    let sym_bytes: Vec<RustBV> = {
        let ctx = state.solver().borrow();
        (0..count)
            .map(|i| {
                let name = format!("sys_read_{}_{}", read_id, i);
                RustBV::symbolic(&ctx, &name, 8)
            })
            .collect()
    };

    for (i, sym_byte) in sym_bytes.into_iter().enumerate() {
        state.memory_store(buf.wrapping_add(i as u64), sym_byte)?;
    }

    Ok(SyscallOutcome::Continue { ret: count })
}

#[cfg(test)]
#[path = "read_tests.rs"]
mod read_tests;
