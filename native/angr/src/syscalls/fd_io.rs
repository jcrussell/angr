//! amd64 (and per-arch) file-descriptor I/O syscalls: lseek / readv / writev.
//!
//! These mirror the seek + vectored-I/O semantics that previously cost a
//! Python `_handle_syscall_callback` round-trip (angr-6ylm):
//!
//!   - `lseek(fd, offset, whence)` mirrors `procedures/fileops.rs::NativeLseek`
//!     and `linux_kernel/lseek.py`: adjusts the `FileSystem` position via
//!     `FileSystem::seek` and returns the new position, or -1 on a bad whence.
//!     Falls back to Python for symbolic args and fds not open in the Rust
//!     `FileSystem` (Python's symbolic-file model owns those).
//!   - `readv(fd, iov, iovcnt)` / `writev(fd, iov, iovcnt)` mirror
//!     `linux_kernel/iovec.py`: walk the `struct iovec[]` (two pointer-width
//!     words per element — `iov_base`, `iov_len`) and dispatch each segment
//!     through the same native read/write logic as `read`/`write`. `writev`
//!     is glibc stdio's flush path, so it is hit on essentially every printf.
//!
//! Symbolic `iov` / `iovcnt`, oversize requests, and fds not handled natively
//! all fall back to Python per the established `SyscallError` pattern. To
//! avoid a partial mutation before a fallback, `writev` gathers every
//! concrete byte across all segments *before* touching the fd buffer — a
//! symbolic byte aborts with `SymbolicArgument` and leaves the fd untouched.
//!
//! Not handled here (deferred to Python, see angr-6ylm follow-up): `pread64`
//! / `pwrite64`. Positioned writes need an offset-honoring `FileSystem::write`
//! (the current model is append-only), so they stay on the Python path.

use std::sync::atomic::{AtomicU64, Ordering};

use super::{NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Per-segment and aggregate caps. Mirror `read`/`write`'s `MAX_*_SIZE`
/// (4096) per segment; cap the segment count so a bogus `iovcnt` can't spin
/// the handler. Anything larger falls back to Python.
const MAX_IO_SIZE: u64 = 4096;
const MAX_IOVCNT: u64 = 1024;

/// Counter for unique readv-from-stdin variable names. Independent of the
/// `read` handler's counter (see `read::SYS_READ_COUNTER` rationale).
static SYS_READV_COUNTER: AtomicU64 = AtomicU64::new(0);

/// `lseek(fd, offset, whence)` — adjust the file position.
pub struct NativeLseekSyscall;

impl NativeSyscall for NativeLseekSyscall {
    fn name(&self) -> &'static str {
        "lseek"
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
                "lseek expected 3 args, got {}",
                args.len()
            )));
        }
        let fd = extract_concrete_arg(&args[0], "lseek fd")?;
        let offset = extract_concrete_arg(&args[1], "lseek offset")?;
        let whence = extract_concrete_arg(&args[2], "lseek whence")?;

        // Fds not open in the Rust FileSystem are owned by Python's
        // symbolic-file model — defer rather than return -1.
        if !state.file_system_ref().is_open(fd as u32) {
            return Err(SyscallError::Other(format!(
                "lseek on fd={fd} (not open in Rust FileSystem) falls back to Python"
            )));
        }

        match state
            .file_system()
            .seek(fd as u32, offset as i64, whence as u32)
        {
            Some(new_pos) => Ok(SyscallOutcome::Continue { ret: new_pos }),
            // Bad whence — mirror NativeLseek returning -1.
            None => Ok(SyscallOutcome::Continue {
                ret: (-1i64) as u64,
            }),
        }
    }
}

/// Read the `(iov_base, iov_len)` pair for element `idx` from a `struct
/// iovec[]` at `iov`. Each field is one pointer-width word.
fn read_iovec(
    state: &RustSimState,
    iov: u64,
    idx: u64,
    word: u32,
) -> Result<(u64, u64), SyscallError> {
    let stride = u64::from(word) * 2;
    let base_addr = iov.wrapping_add(idx * stride);
    let len_addr = base_addr.wrapping_add(u64::from(word));
    let base = state
        .memory_load(base_addr, word)?
        .as_u64()
        .ok_or_else(|| SyscallError::SymbolicArgument(format!("iov_base[{idx}]")))?;
    let len = state
        .memory_load(len_addr, word)?
        .as_u64()
        .ok_or_else(|| SyscallError::SymbolicArgument(format!("iov_len[{idx}]")))?;
    Ok((base, len))
}

/// `writev(fd, iov, iovcnt)` — concatenate iovec segments to the fd buffer.
pub struct NativeWritevSyscall;

impl NativeSyscall for NativeWritevSyscall {
    fn name(&self) -> &'static str {
        "writev"
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
                "writev expected 3 args, got {}",
                args.len()
            )));
        }
        let fd = extract_concrete_arg(&args[0], "writev fd")?;
        if fd == 0 {
            return Err(SyscallError::Other(
                "writev to fd=0 (stdin) falls back to Python".to_string(),
            ));
        }
        if !state.file_system_ref().is_open(fd as u32) {
            return Err(SyscallError::Other(format!(
                "writev to fd={fd} (not open in Rust FileSystem) falls back to Python"
            )));
        }
        let iov = extract_concrete_arg(&args[1], "writev iov")?;
        let iovcnt = extract_concrete_arg(&args[2], "writev iovcnt")?;
        if iovcnt > MAX_IOVCNT {
            return Err(SyscallError::Other(format!(
                "writev iovcnt {iovcnt} exceeds limit"
            )));
        }

        let word = state.arch().bytes();

        // Gather every concrete byte across all segments BEFORE mutating the
        // fd buffer — a symbolic byte must leave the fd untouched so the
        // Python fallback path produces the single authoritative write.
        let mut bytes: Vec<u8> = Vec::new();
        for idx in 0..iovcnt {
            let (base, len) = read_iovec(state, iov, idx, word)?;
            if len > MAX_IO_SIZE {
                return Err(SyscallError::Other(format!(
                    "writev iov_len {len} exceeds limit"
                )));
            }
            for i in 0..len {
                let bv = state.memory_load(base.wrapping_add(i), 1)?;
                match bv.as_u64() {
                    Some(v) => bytes.push(v as u8),
                    None => {
                        return Err(SyscallError::SymbolicArgument(format!(
                            "symbolic byte in iov[{idx}]+{i}"
                        )));
                    }
                }
            }
        }

        let total = bytes.len() as u64;
        state.write_fd(fd as u32, &bytes);
        Ok(SyscallOutcome::Continue { ret: total })
    }
}

/// `readv(fd, iov, iovcnt)` — scatter-read into iovec segments.
pub struct NativeReadvSyscall;

impl NativeSyscall for NativeReadvSyscall {
    fn name(&self) -> &'static str {
        "readv"
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
                "readv expected 3 args, got {}",
                args.len()
            )));
        }
        let fd = extract_concrete_arg(&args[0], "readv fd")?;
        let iov = extract_concrete_arg(&args[1], "readv iov")?;
        let iovcnt = extract_concrete_arg(&args[2], "readv iovcnt")?;
        if iovcnt > MAX_IOVCNT {
            return Err(SyscallError::Other(format!(
                "readv iovcnt {iovcnt} exceeds limit"
            )));
        }

        let word = state.arch().bytes();

        // Decode all segments up front (symbolic iov fields → fallback) so a
        // mid-loop symbolic field doesn't leave a partial scatter.
        let mut segments: Vec<(u64, u64)> = Vec::with_capacity(iovcnt as usize);
        for idx in 0..iovcnt {
            let (base, len) = read_iovec(state, iov, idx, word)?;
            if len > MAX_IO_SIZE {
                return Err(SyscallError::Other(format!(
                    "readv iov_len {len} exceeds limit"
                )));
            }
            segments.push((base, len));
        }

        if fd == 0 {
            // stdin: fill every segment with fresh symbolic bytes, mirroring
            // `read`'s stdin path. Returns the requested total.
            let mut total = 0u64;
            for (base, len) in segments {
                let read_id = SYS_READV_COUNTER.fetch_add(1, Ordering::Relaxed);
                let sym_bytes: Vec<RustBV> = {
                    let ctx = state.solver().borrow();
                    (0..len)
                        .map(|i| RustBV::symbolic(&ctx, format!("sys_readv_{read_id}_{i}"), 8))
                        .collect()
                };
                for (i, b) in sym_bytes.into_iter().enumerate() {
                    state.memory_store(base.wrapping_add(i as u64), b)?;
                }
                total += len;
            }
            return Ok(SyscallOutcome::Continue { ret: total });
        }

        // User fd: serve concrete content from the FileSystem, advancing the
        // position per segment (mirrors `read`). Fds not open or without
        // concrete content fall back to Python.
        let (open, content_len) = match state.file_system_ref().fd_info(fd as u32) {
            Some((_, _, _, len, is_open)) => (is_open, len),
            None => (false, 0),
        };
        if !open {
            return Err(SyscallError::Other(format!(
                "readv from fd={fd} (not open in Rust FileSystem) falls back to Python"
            )));
        }
        if content_len == 0 {
            return Err(SyscallError::Other(format!(
                "readv from fd={fd} has no concrete content; falling back to Python"
            )));
        }

        let mut total = 0u64;
        for (base, len) in segments {
            let bytes = state.file_system().read(fd as u32, len as usize);
            let n = bytes.len();
            for (i, b) in bytes.iter().enumerate() {
                state.memory_store(base.wrapping_add(i as u64), RustBV::concrete(*b as u128, 8))?;
            }
            total += n as u64;
            if n < len as usize {
                break; // EOF — stop scattering further segments.
            }
        }
        Ok(SyscallOutcome::Continue { ret: total })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;
    use crate::state::{FdFlags, RustSimState};
    use crate::symbolic::{RustBV, SymContext};

    fn fresh_state() -> RustSimState {
        let mut state = RustSimState::new("amd64").expect("amd64 state");
        state.map_memory(0x2000, 0x2000, Permission::RWX);
        state
    }

    /// Lay out a `struct iovec[]` at `iov_addr` pointing at `segments`
    /// `(base, len)` pairs (amd64: 8-byte words).
    fn write_iovec_array(state: &mut RustSimState, iov_addr: u64, segments: &[(u64, u64)]) {
        for (idx, (base, len)) in segments.iter().enumerate() {
            let elem = iov_addr + (idx as u64) * 16;
            state
                .memory_store(elem, RustBV::concrete(*base as u128, 64))
                .unwrap();
            state
                .memory_store(elem + 8, RustBV::concrete(*len as u128, 64))
                .unwrap();
        }
    }

    #[test]
    fn metadata() {
        assert_eq!(NativeLseekSyscall.name(), "lseek");
        assert_eq!(NativeLseekSyscall.num_args(), 3);
        assert_eq!(NativeWritevSyscall.name(), "writev");
        assert_eq!(NativeReadvSyscall.name(), "readv");
    }

    #[test]
    fn lseek_set_returns_new_position() {
        let mut state = fresh_state();
        state.file_system().open_with_content(
            "f.bin".to_string(),
            FdFlags::ReadOnly,
            b"abcdefgh".to_vec(),
        );
        // SEEK_SET to 3.
        let out = NativeLseekSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(3, 64),
                    RustBV::concrete(3, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect("ok");
        match out {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 3),
            _ => panic!("expected Continue"),
        }
        assert_eq!(state.file_system_ref().fd_info(3).unwrap().1, 3);
    }

    #[test]
    fn lseek_bad_whence_returns_neg1() {
        let mut state = fresh_state();
        state.file_system().open("f".to_string(), FdFlags::ReadOnly);
        let out = NativeLseekSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(3, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(99, 64),
                ],
            )
            .expect("ok");
        match out {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, (-1i64) as u64),
            _ => panic!("expected Continue"),
        }
    }

    #[test]
    fn lseek_unknown_fd_falls_back() {
        let mut state = fresh_state();
        let err = NativeLseekSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(7, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect_err("fallback");
        assert!(matches!(err, SyscallError::Other(_)));
    }

    #[test]
    fn lseek_symbolic_fd_falls_back() {
        let mut state = fresh_state();
        let ctx = SymContext::new();
        let err = NativeLseekSyscall
            .call(
                &mut state,
                &[
                    RustBV::symbolic(&ctx, "fd", 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect_err("fallback");
        assert!(matches!(err, SyscallError::SymbolicArgument(_)));
    }

    #[test]
    fn writev_concatenates_segments_to_stdout() {
        let mut state = fresh_state();
        state.map_memory_data(0x3000, b"hello", Permission::RWX);
        state.map_memory_data(0x3100, b" world", Permission::RWX);
        write_iovec_array(&mut state, 0x2000, &[(0x3000, 5), (0x3100, 6)]);

        let out = NativeWritevSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(1, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(2, 64),
                ],
            )
            .expect("ok");
        match out {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 11),
            _ => panic!("expected Continue"),
        }
        assert_eq!(state.stdout_buffer(), b"hello world");
    }

    #[test]
    fn writev_symbolic_byte_leaves_fd_untouched() {
        let mut state = fresh_state();
        state.map_memory_data(0x3000, b"ok", Permission::RWX);
        // Second segment has a symbolic byte.
        let ctx = SymContext::new();
        state
            .memory_store(0x3100, RustBV::symbolic(&ctx, "b", 8))
            .unwrap();
        write_iovec_array(&mut state, 0x2000, &[(0x3000, 2), (0x3100, 1)]);

        let err = NativeWritevSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(1, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(2, 64),
                ],
            )
            .expect_err("fallback");
        assert!(matches!(err, SyscallError::SymbolicArgument(_)));
        // No partial write to stdout.
        assert_eq!(state.stdout_buffer(), b"");
    }

    #[test]
    fn writev_stdin_falls_back() {
        let mut state = fresh_state();
        write_iovec_array(&mut state, 0x2000, &[(0x3000, 1)]);
        let err = NativeWritevSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(1, 64),
                ],
            )
            .expect_err("fallback");
        assert!(matches!(err, SyscallError::Other(_)));
    }

    #[test]
    fn writev_user_fd_appends() {
        let mut state = fresh_state();
        state
            .file_system()
            .open("out".to_string(), FdFlags::WriteOnly);
        state.map_memory_data(0x3000, b"abc", Permission::RWX);
        write_iovec_array(&mut state, 0x2000, &[(0x3000, 3)]);
        NativeWritevSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(3, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(1, 64),
                ],
            )
            .expect("ok");
        assert_eq!(state.file_system_ref().fd_content(3), b"abc");
    }

    #[test]
    fn readv_user_fd_scatters_content() {
        let mut state = fresh_state();
        state.file_system().open_with_content(
            "in".to_string(),
            FdFlags::ReadOnly,
            b"abcdef".to_vec(),
        );
        write_iovec_array(&mut state, 0x2000, &[(0x3000, 3), (0x3100, 3)]);
        let out = NativeReadvSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(3, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(2, 64),
                ],
            )
            .expect("ok");
        match out {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 6),
            _ => panic!("expected Continue"),
        }
        for (i, &b) in b"abc".iter().enumerate() {
            assert_eq!(
                state.memory_load(0x3000 + i as u64, 1).unwrap().as_u64(),
                Some(b as u64)
            );
        }
        for (i, &b) in b"def".iter().enumerate() {
            assert_eq!(
                state.memory_load(0x3100 + i as u64, 1).unwrap().as_u64(),
                Some(b as u64)
            );
        }
    }

    #[test]
    fn readv_stdin_writes_symbolic() {
        let mut state = fresh_state();
        write_iovec_array(&mut state, 0x2000, &[(0x3000, 4)]);
        let out = NativeReadvSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(1, 64),
                ],
            )
            .expect("ok");
        match out {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 4),
            _ => panic!("expected Continue"),
        }
        for i in 0..4u64 {
            assert!(
                state.memory_load(0x3000 + i, 1).unwrap().as_u64().is_none(),
                "byte {i} should be symbolic"
            );
        }
    }

    #[test]
    fn readv_unknown_fd_falls_back() {
        let mut state = fresh_state();
        write_iovec_array(&mut state, 0x2000, &[(0x3000, 4)]);
        let err = NativeReadvSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(9, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(1, 64),
                ],
            )
            .expect_err("fallback");
        assert!(matches!(err, SyscallError::Other(_)));
    }
}
