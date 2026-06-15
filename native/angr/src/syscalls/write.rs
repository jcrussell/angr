//! amd64 write syscall handler.
//!
//! Mirrors `procedures/write.rs::NativeWrite`: handles `sys_write(fd, buf,
//! count)` for any fd open in the Rust `FileSystem` (stdout/stderr
//! pre-registered, plus any user fd from `NativeOpen`/`NativePipe`/etc.).
//! Reads `count` concrete bytes from memory at `buf` and appends them to the
//! fd's content buffer via `state.write_fd`. Returns `count`.
//!
//! Falls back to Python (`_handle_syscall_callback`) for: symbolic args,
//! symbolic bytes in `[buf, buf+count)`, fd=0 (stdin), fds not open in
//! Rust's `FileSystem`, and counts beyond `MAX_WRITE_SIZE`. The Python path
//! goes through `state.posix.get_fd(fd).write(...)`, which owns the
//! symbolic-content + symbolic-fd plumbing.
//!
//! See `procedures/write.rs` for the fd-table sync invariant (angr-8j16).

use super::{NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg};
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
        let fd = extract_concrete_arg(&args[0], "write fd")?;
        if fd == 0 {
            return Err(SyscallError::Other(
                "write to fd=0 (stdin) falls back to Python".to_string(),
            ));
        }
        let fd_u32 = fd as u32;
        if !state.file_system_ref().is_open(fd_u32) {
            return Err(SyscallError::Other(format!(
                "write to fd={} (not open in Rust FileSystem) falls back to Python",
                fd
            )));
        }
        let buf = extract_concrete_arg(&args[1], "write buf")?;
        let count = extract_concrete_arg(&args[2], "write count")?;

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
                    return Err(e.into());
                }
            }
        }

        state.write_fd(fd_u32, &bytes);
        Ok(SyscallOutcome::Continue { ret: count })
    }
}

#[cfg(test)]
#[path = "write_tests.rs"]
mod write_tests;
