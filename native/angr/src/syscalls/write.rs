//! amd64 write syscall handler.
//!
//! Mirrors `procedures/write.rs::NativeWrite`: handles `sys_write(fd, buf,
//! count)` for `fd == 1` (stdout) and `fd == 2` (stderr) by reading
//! `count` concrete bytes from memory at `buf` and appending them to the
//! corresponding fd buffer via `state.write_fd`. Returns `count`.
//!
//! Symbolic args, symbolic bytes in [buf, buf+count), other fds, and counts
//! beyond `MAX_WRITE_SIZE` fall back to the Python `_handle_syscall_callback`
//! path. The Python path goes through `state.posix.get_fd(fd).write(...)`,
//! which has the symbolic-content + non-stdio fd plumbing.

use super::{NativeSyscall, SyscallError, SyscallOutcome};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const MAX_WRITE_SIZE: u64 = 4096;

pub struct NativeWriteSyscall;

impl NativeSyscall for NativeWriteSyscall {
    fn name(&self) -> &'static str {
        "write"
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
                "write expected 3 args, got {}",
                args.len()
            )));
        }
        let fd = args[0]
            .as_u64()
            .ok_or_else(|| SyscallError::SymbolicArgument("write fd".into()))?;
        if fd != 1 && fd != 2 {
            return Err(SyscallError::Other(format!(
                "write to fd={} not supported natively",
                fd
            )));
        }
        let buf = args[1]
            .as_u64()
            .ok_or_else(|| SyscallError::SymbolicArgument("write buf".into()))?;
        let count = args[2]
            .as_u64()
            .ok_or_else(|| SyscallError::SymbolicArgument("write count".into()))?;

        if count > MAX_WRITE_SIZE {
            return Err(SyscallError::Other(format!(
                "write count {} exceeds limit",
                count
            )));
        }

        let mut bytes = Vec::with_capacity(count as usize);
        for i in 0..count {
            match state.memory_load(buf.wrapping_add(i), 1) {
                Ok(bv) => match bv.as_u64() {
                    Some(val) => bytes.push(val as u8),
                    None => {
                        return Err(SyscallError::SymbolicArgument(format!(
                            "symbolic byte at buf+{i}"
                        )));
                    }
                },
                Err(e) => {
                    return Err(SyscallError::Other(format!("memory_load: {e}")));
                }
            }
        }

        state.write_fd(fd as u32, &bytes);
        Ok(SyscallOutcome::Continue { ret: count })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;
    use crate::state::RustSimState;
    use crate::symbolic::{RustBV, SymContext};

    #[test]
    fn handler_metadata() {
        let h = NativeWriteSyscall;
        assert_eq!(h.name(), "write");
        assert_eq!(h.num_args(), 3);
    }

    #[test]
    fn write_stdout_appends_and_returns_count() {
        let h = NativeWriteSyscall;
        let mut state = RustSimState::new("amd64").expect("amd64 state");
        state.map_memory_data(0x1000, b"hello", Permission::RWX);

        let outcome = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(1, 64),
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(5, 64),
                ],
            )
            .expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 5),
            _ => panic!("expected Continue"),
        }
        assert_eq!(state.stdout_buffer(), b"hello");
    }

    #[test]
    fn write_stderr_appends_and_returns_count() {
        let h = NativeWriteSyscall;
        let mut state = RustSimState::new("amd64").expect("amd64 state");
        state.map_memory_data(0x1000, b"err", Permission::RWX);

        let outcome = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(2, 64),
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(3, 64),
                ],
            )
            .expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 3),
            _ => panic!("expected Continue"),
        }
        assert_eq!(state.fd_buffer(2), b"err");
    }

    #[test]
    fn write_unsupported_fd_falls_back() {
        let h = NativeWriteSyscall;
        let mut state = RustSimState::new("amd64").expect("amd64 state");
        state.map_memory_data(0x1000, b"x", Permission::RWX);
        let err = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(3, 64),
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(1, 64),
                ],
            )
            .expect_err("must fall back");
        assert!(matches!(err, SyscallError::Other(_)));
        // No append must have occurred.
        assert_eq!(state.stdout_buffer(), b"");
    }

    #[test]
    fn write_too_large_falls_back() {
        let h = NativeWriteSyscall;
        let mut state = RustSimState::new("amd64").expect("amd64 state");
        let err = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(1, 64),
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(5000, 64),
                ],
            )
            .expect_err("must fall back");
        assert!(matches!(err, SyscallError::Other(_)));
    }

    #[test]
    fn write_symbolic_fd_falls_back() {
        let h = NativeWriteSyscall;
        let mut state = RustSimState::new("amd64").expect("amd64 state");
        let ctx = SymContext::new();
        let sym_fd = RustBV::symbolic(&ctx, "fd", 64);
        let err = h
            .call(
                &mut state,
                &[
                    sym_fd,
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(1, 64),
                ],
            )
            .expect_err("must fall back");
        match err {
            SyscallError::SymbolicArgument(msg) => {
                assert!(msg.contains("fd"), "message should name fd, got {msg:?}",)
            }
            other => panic!("expected SymbolicArgument, got {other:?}"),
        }
    }

    #[test]
    fn write_symbolic_byte_falls_back() {
        let h = NativeWriteSyscall;
        let mut state = RustSimState::new("amd64").expect("amd64 state");
        // Map some memory and write a symbolic byte at 0x1000.
        state.map_memory(0x1000, 0x1000, Permission::RWX);
        let ctx = SymContext::new();
        let sym_byte = RustBV::symbolic(&ctx, "b", 8);
        state.memory_store(0x1000, sym_byte).expect("ok");

        let err = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(1, 64),
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(1, 64),
                ],
            )
            .expect_err("must fall back");
        match err {
            SyscallError::SymbolicArgument(msg) => assert!(
                msg.contains("buf+0"),
                "message should name the symbolic offset, got {msg:?}",
            ),
            other => panic!("expected SymbolicArgument, got {other:?}"),
        }
        // No partial write to stdout must have occurred.
        assert_eq!(state.stdout_buffer(), b"");
    }
}
