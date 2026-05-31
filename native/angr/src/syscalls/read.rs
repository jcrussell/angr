//! amd64 read syscall handler.
//!
//! Mirrors `procedures/read.rs::NativeRead`: handles `sys_read(fd, buf, count)`
//! when `fd == 0` (stdin) by writing `count` fresh symbolic bytes into memory
//! at `buf` and returning `count` in rax. Other fds, symbolic args, or counts
//! beyond `MAX_READ_SIZE` fall back to the Python `_handle_syscall_callback`
//! path which dispatches through `state.posix.get_fd(fd).read(...)`.
//!
//! The dirty pages produced here are picked up by
//! `_replay_rust_dirty_pages` (rust_state_sync.py) on the next Python
//! callback, so a downstream Python SimProc (e.g. strcmp) will observe the
//! symbolic stdin bytes — same contract as NativeRead. See memory
//! `invariant-3tek2-replay-ordering`.

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
        if fd != 0 {
            return Err(SyscallError::Other(format!(
                "read from fd={} not supported natively",
                fd
            )));
        }
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

        let read_id = SYS_READ_COUNTER.fetch_add(1, Ordering::Relaxed);

        // Build symbolic bytes first (needs solver borrow), then store to
        // memory (needs &mut state with the solver borrow released).
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;
    use crate::state::RustSimState;
    use crate::symbolic::{RustBV, SymContext};

    fn fresh_state_with_buf() -> RustSimState {
        let mut state = RustSimState::new("amd64").expect("amd64 state");
        state.map_memory(0x2000, 0x1000, Permission::RWX);
        state
    }

    #[test]
    fn handler_metadata() {
        let h = NativeReadSyscall;
        assert_eq!(h.name(), "read");
        assert_eq!(h.num_args(), 3);
    }

    #[test]
    fn read_stdin_returns_count_and_writes_symbolic_bytes() {
        let h = NativeReadSyscall;
        let mut state = fresh_state_with_buf();

        let outcome = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(4, 64),
                ],
            )
            .expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 4),
            _ => panic!("expected Continue"),
        }
        for i in 0..4u64 {
            let byte = state.memory_load(0x2000 + i, 1).expect("loaded");
            assert!(byte.as_u64().is_none(), "byte {} should be symbolic", i);
        }
    }

    #[test]
    fn read_zero_count_returns_zero_without_writing() {
        let h = NativeReadSyscall;
        let mut state = fresh_state_with_buf();
        // Pre-fill so we can detect any write.
        state.map_memory_data(0x2000, b"abcd", Permission::RWX);

        let outcome = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
            _ => panic!("expected Continue"),
        }
        // Pre-fill must be intact.
        for (i, &b) in b"abcd".iter().enumerate() {
            let byte = state.memory_load(0x2000 + i as u64, 1).expect("loaded");
            assert_eq!(byte.as_u64(), Some(b as u64));
        }
    }

    #[test]
    fn read_non_stdin_falls_back() {
        let h = NativeReadSyscall;
        let mut state = fresh_state_with_buf();
        let err = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(3, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(4, 64),
                ],
            )
            .expect_err("must fall back");
        assert!(matches!(err, SyscallError::Other(_)));
    }

    #[test]
    fn read_too_large_falls_back() {
        let h = NativeReadSyscall;
        let mut state = fresh_state_with_buf();
        let err = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(5000, 64),
                ],
            )
            .expect_err("must fall back");
        assert!(matches!(err, SyscallError::Other(_)));
    }

    #[test]
    fn symbolic_fd_falls_back() {
        let h = NativeReadSyscall;
        let mut state = fresh_state_with_buf();
        let ctx = SymContext::new();
        let sym_fd = RustBV::symbolic(&ctx, "fd", 64);
        let err = h
            .call(
                &mut state,
                &[
                    sym_fd,
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(4, 64),
                ],
            )
            .expect_err("must fall back");
        match err {
            SyscallError::SymbolicArgument(msg) => {
                assert!(msg.contains("fd"), "message should name fd, got {msg:?}");
            }
            other => panic!("expected SymbolicArgument, got {other:?}"),
        }
    }

    #[test]
    fn symbolic_buf_falls_back() {
        let h = NativeReadSyscall;
        let mut state = fresh_state_with_buf();
        let ctx = SymContext::new();
        let sym_buf = RustBV::symbolic(&ctx, "buf", 64);
        let err = h
            .call(
                &mut state,
                &[RustBV::concrete(0, 64), sym_buf, RustBV::concrete(4, 64)],
            )
            .expect_err("must fall back");
        assert!(matches!(err, SyscallError::SymbolicArgument(_)));
    }

    #[test]
    fn symbolic_count_falls_back() {
        let h = NativeReadSyscall;
        let mut state = fresh_state_with_buf();
        let ctx = SymContext::new();
        let sym_count = RustBV::symbolic(&ctx, "count", 64);
        let err = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0x2000, 64),
                    sym_count,
                ],
            )
            .expect_err("must fall back");
        assert!(matches!(err, SyscallError::SymbolicArgument(_)));
    }
}
