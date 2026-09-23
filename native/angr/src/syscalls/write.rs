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
//! The symbolic-fd / zero-length / content-demotion ordering in front of the
//! actual write is shared with `writev` and `pwrite64` (`syscalls/fd_io.rs`)
//! and lives in [`crate::syscalls::write_path_gate`]; this
//! handler supplies only the gather-and-sink body.
//!
//! See `procedures/write.rs` for the fd-table sync invariant (angr-8j16).

use super::require_syscall_args;
use super::{
    MAX_IO_SIZE as MAX_WRITE_SIZE, NativeSyscall, SyscallError, SyscallOutcome,
    extract_concrete_arg, gather_concrete_bytes, write_path_gate,
};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

pub(crate) struct NativeWriteSyscall;

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
        require_syscall_args!(self, args);
        write_path_gate(state, self.name(), &args[0], &args[2], |state, target| {
            let buf = extract_concrete_arg(&args[1], "write buf")?;
            let count = extract_concrete_arg(&args[2], "write count")?;
            if count > MAX_WRITE_SIZE {
                return Err(SyscallError::Other(format!(
                    "write count {count} exceeds limit"
                )));
            }
            let bytes = gather_concrete_bytes(state, buf, count, "write buf")?;
            // Choke-point insurance — unreachable after the gate (see
            // syscalls::write_gate).
            if !state.write_fd(target.fd_u32(), &bytes) {
                return Err(target.demoted());
            }
            Ok(count)
        })
    }
}

test_submod!("write_tests.rs" => write_tests);
