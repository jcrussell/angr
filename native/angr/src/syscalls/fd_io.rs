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
//!     `readv` also mirrors `read`'s is_symbolic fast path (angr-myzjx.12): a
//!     symbolic-stream fd (the stdin model, `FileSystem::open_symbolic`) with
//!     no concrete content scatters fresh symbolic bytes natively rather than
//!     bouncing to Python.
//!
//! Symbolic `iov` / `iovcnt`, oversize requests, and fds not handled natively
//! all fall back to Python per the established `SyscallError` pattern. To
//! avoid a partial mutation before a fallback, `writev` gathers every
//! concrete byte across all segments *before* touching the fd buffer — a
//! symbolic byte aborts with `SymbolicArgument` and leaves the fd untouched.
//!
//! Bounded symbolic file content (angr-0xyq2 Phase 2): `readv` / `pread64`
//! serve the fd's registered per-byte BVs (`FileSystem::read_sym` /
//! `read_sym_at`), clamping requests to `MAX_SYMFILE_SERVE_SIZE` (= the
//! export cap, so any registered file serves in one call, matching Python)
//! instead of bouncing — a fallback would split the position cursor
//! (angr-8j16). `writev` / `pwrite64` demote the content
//! (`demote_symbolic_content`) and bounce to Python before mutating;
//! zero-length writes are a no-demotion no-op, and a symbolic fd demotes
//! everything (`demote_all_symbolic_content`) before falling back.
//!
//!   - `pread64(fd, buf, nbyte, offset)` / `pwrite64(fd, buf, nbyte, offset)`
//!     (angr-dbb1) mirror `posix/pread64.py` / `pwrite64.py`: positioned I/O
//!     that does NOT disturb the fd's current position. `pread64` serves
//!     concrete `FileSystem` content via `read_at`; `pwrite64` overwrites at
//!     the offset via the offset-honoring `FileSystem::write_at` (unlike the
//!     append-only `write`). Symbolic offset / symbolic data / fds not open
//!     natively fall back to Python.

use std::sync::atomic::{AtomicU64, Ordering};

use super::{MAX_IO_SIZE, NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg};
use crate::procedures::strings::write_bv_bytes;
use crate::state::MAX_SYMFILE_SERVE_SIZE;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Cap the segment count so a bogus `iovcnt` can't spin the handler. The
/// per-segment byte cap is the shared `MAX_IO_SIZE` (see `syscalls::mod`).
const MAX_IOVCNT: u64 = 1024;

/// Counter for unique readv-from-stdin variable names. Independent of the
/// `read` handler's counter (see `read::SYS_READ_COUNTER` rationale).
static SYS_READV_COUNTER: AtomicU64 = AtomicU64::new(0);

/// `lseek(fd, offset, whence)` — adjust the file position.
pub(crate) struct NativeLseekSyscall;

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
pub(crate) struct NativeWritevSyscall;

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
        let fd = match extract_concrete_arg(&args[0], "writev fd") {
            Ok(fd) => fd,
            Err(e) => {
                // Symbolic fd on a write path: any bounded symbolic file
                // could be the target — hand them all to Python before the
                // fallback (angr-0xyq2 A4; O(1) when none attached).
                state.file_system().demote_all_symbolic_content();
                return Err(e);
            }
        };
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
        // Deferred `?` + zero-check before the demote gate — see
        // syscalls/write.rs (A3/A4 ordering).
        let iov = extract_concrete_arg(&args[1], "writev iov");
        let iovcnt = extract_concrete_arg(&args[2], "writev iovcnt");
        if let Ok(0) = iovcnt {
            return Ok(SyscallOutcome::Continue { ret: 0 });
        }
        // Write-demotion (angr-0xyq2 Phase 2) — see procedures/write.rs.
        if state.file_system().demote_symbolic_content(fd as u32) {
            return Err(SyscallError::Other(format!(
                "writev to fd={fd} with symbolic content falls back to Python (demoted)"
            )));
        }
        let iov = iov?;
        let iovcnt = iovcnt?;
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
        // Unreachable after the gate above; choke-point insurance (see
        // FileSystem::write).
        if !state.write_fd(fd as u32, &bytes) {
            return Err(SyscallError::Other(format!(
                "writev to fd={fd} with symbolic content falls back to Python (demoted)"
            )));
        }
        Ok(SyscallOutcome::Continue { ret: total })
    }
}

/// `readv(fd, iov, iovcnt)` — scatter-read into iovec segments.
pub(crate) struct NativeReadvSyscall;

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

        // Bounded-symbolic-content fds clamp oversized segments instead of
        // bouncing (see the serve loop below), so the per-segment cap only
        // applies to the stdin / concrete paths.
        let serve_sym = fd != 0 && state.file_system_ref().has_content_sym(fd as u32);

        // Decode all segments up front (symbolic iov fields → fallback) so a
        // mid-loop symbolic field doesn't leave a partial scatter.
        let mut segments: Vec<(u64, u64)> = Vec::with_capacity(iovcnt as usize);
        for idx in 0..iovcnt {
            let (base, len) = read_iovec(state, iov, idx, word)?;
            if !serve_sym && len > MAX_IO_SIZE {
                return Err(SyscallError::Other(format!(
                    "readv iov_len {len} exceeds limit"
                )));
            }
            segments.push((base, len));
        }

        if fd == 0 {
            // stdin: fill every segment with fresh symbolic bytes, mirroring
            // `read`'s stdin path. Returns the requested total.
            return scatter_symbolic(state, &segments, "sys_readv");
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
        // Bounded symbolic file content (angr-0xyq2 Phase 2): scatter the
        // registered per-byte BVs per segment via read_sym, mirroring the
        // concrete loop below (position advances; EOF stops scattering).
        // Segments are clamped to MAX_SYMFILE_SERVE_SIZE (= the export cap;
        // a whole registered file fits in one segment, matching Python),
        // ending the scatter there. A mid-loop `None` (impossible while the
        // predicate holds, but not worth an `expect`) degrades to a short
        // read too.
        if serve_sym {
            let mut total = 0u64;
            for (base, len) in segments {
                let serve = (len as usize).min(MAX_SYMFILE_SERVE_SIZE as usize);
                let Some(sym_bytes) = state.file_system().read_sym(fd as u32, serve) else {
                    break;
                };
                let n = sym_bytes.len();
                write_bv_bytes(state, base, sym_bytes)?;
                total += n as u64;
                if (n as u64) < len {
                    break; // EOF or clamp — stop scattering further segments.
                }
            }
            // One counter bump per guest readv that served bytes (NOT per
            // segment) — keeps `symfile_reads_native` comparable across
            // read/fread/readv/pread64.
            if total > 0 {
                crate::symbolic::record_symfile_read_native();
            }
            return Ok(SyscallOutcome::Continue { ret: total });
        }
        if content_len == 0 {
            // No concrete bytes left. A symbolic-stream fd (the stdin model)
            // scatters fresh symbolic bytes natively, mirroring `read`'s
            // is_symbolic fast path; any other fd defers to Python's
            // symbolic-file model. (pread64 deliberately lacks this: a
            // positioned read on a stream has no clear position semantics.)
            if state.file_system_ref().is_symbolic(fd as u32) {
                return scatter_symbolic(state, &segments, &format!("sys_readv_fd{fd}"));
            }
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

/// Scatter fresh symbolic bytes across the iovec `segments`, mirroring
/// `read::read_symbolic`. Shared by `readv`'s stdin path (fd 0) and
/// symbolic-stream fds (`FileSystem::is_symbolic`). Each segment mints `len`
/// uniquely-named bytes (`<prefix>_<id>_<i>`); returns the requested total in
/// rax (a symbolic stream never hits EOF).
fn scatter_symbolic(
    state: &mut RustSimState,
    segments: &[(u64, u64)],
    prefix: &str,
) -> Result<SyscallOutcome, SyscallError> {
    let mut total = 0u64;
    for &(base, len) in segments {
        let read_id = SYS_READV_COUNTER.fetch_add(1, Ordering::Relaxed);
        let sym_bytes: Vec<RustBV> = {
            let ctx = state.solver().borrow();
            (0..len)
                .map(|i| RustBV::symbolic(&ctx, format!("{prefix}_{read_id}_{i}"), 8))
                .collect()
        };
        write_bv_bytes(state, base, sym_bytes)?;
        total += len;
    }
    Ok(SyscallOutcome::Continue { ret: total })
}

/// `pread64(fd, buf, nbyte, offset)` — positioned read; file position
/// unaffected. Mirrors `posix/pread64.py`.
pub(crate) struct NativePread64Syscall;

impl NativeSyscall for NativePread64Syscall {
    fn name(&self) -> &'static str {
        "pread64"
    }

    fn num_args(&self) -> usize {
        4
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        if args.len() < 4 {
            return Err(SyscallError::Other(format!(
                "pread64 expected 4 args, got {}",
                args.len()
            )));
        }
        let fd = extract_concrete_arg(&args[0], "pread64 fd")?;
        let buf = extract_concrete_arg(&args[1], "pread64 buf")?;
        let nbyte = extract_concrete_arg(&args[2], "pread64 nbyte")?;
        // Symbolic offset is unsupported by Python's pread64 (raises
        // SimPosixError); fall back so Python produces the authoritative error.
        let offset = extract_concrete_arg(&args[3], "pread64 offset")?;
        // stdin (fd=0) is owned by Python's symbolic-packet model; positioned
        // reads of it are unusual — defer rather than fabricate.
        if fd == 0 {
            return Err(SyscallError::Other(
                "pread64 from fd=0 (stdin) falls back to Python".to_string(),
            ));
        }
        let (open, content_len) = match state.file_system_ref().fd_info(fd as u32) {
            Some((_, _, _, len, is_open)) => (is_open, len),
            None => (false, 0),
        };
        if !open {
            return Err(SyscallError::Other(format!(
                "pread64 from fd={fd} (not open in Rust FileSystem) falls back to Python"
            )));
        }
        // Bounded symbolic file content (angr-0xyq2 Phase 2): positioned
        // serve of the registered per-byte BVs; the fd position is untouched.
        // nbyte is clamped to MAX_SYMFILE_SERVE_SIZE (= the export cap;
        // whole file in one call, matching Python) — see module docs — so
        // the cap bounce below only applies to concrete content.
        let clamped = nbyte.min(MAX_SYMFILE_SERVE_SIZE) as usize;
        if let Some(sym_bytes) = state
            .file_system_ref()
            .read_sym_at(fd as u32, offset, clamped)
        {
            let n = sym_bytes.len() as u64;
            write_bv_bytes(state, buf, sym_bytes)?;
            if n > 0 {
                crate::symbolic::record_symfile_read_native();
            }
            return Ok(SyscallOutcome::Continue { ret: n });
        }
        if nbyte > MAX_IO_SIZE {
            return Err(SyscallError::Other(format!(
                "pread64 nbyte {nbyte} exceeds limit"
            )));
        }
        if content_len == 0 {
            return Err(SyscallError::Other(format!(
                "pread64 from fd={fd} has no concrete content; falling back to Python"
            )));
        }

        let bytes = state
            .file_system_ref()
            .read_at(fd as u32, offset, nbyte as usize);
        let n = bytes.len() as u64;
        for (i, b) in bytes.iter().enumerate() {
            state.memory_store(buf.wrapping_add(i as u64), RustBV::concrete(*b as u128, 8))?;
        }
        Ok(SyscallOutcome::Continue { ret: n })
    }
}

/// `pwrite64(fd, buf, nbyte, offset)` — positioned write; file position
/// unaffected. Mirrors `posix/pwrite64.py`. Uses the offset-honoring
/// `FileSystem::write_at` (overwrite at offset) rather than append-only
/// `write`.
pub(crate) struct NativePwrite64Syscall;

impl NativeSyscall for NativePwrite64Syscall {
    fn name(&self) -> &'static str {
        "pwrite64"
    }

    fn num_args(&self) -> usize {
        4
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        if args.len() < 4 {
            return Err(SyscallError::Other(format!(
                "pwrite64 expected 4 args, got {}",
                args.len()
            )));
        }
        let fd = match extract_concrete_arg(&args[0], "pwrite64 fd") {
            Ok(fd) => fd,
            Err(e) => {
                // Symbolic fd on a write path — see writev above (A4).
                state.file_system().demote_all_symbolic_content();
                return Err(e);
            }
        };
        if fd == 0 {
            return Err(SyscallError::Other(
                "pwrite64 to fd=0 (stdin) falls back to Python".to_string(),
            ));
        }
        if !state.file_system_ref().is_open(fd as u32) {
            return Err(SyscallError::Other(format!(
                "pwrite64 to fd={fd} (not open in Rust FileSystem) falls back to Python"
            )));
        }
        // Deferred `?` + zero-check before the demote gate — see
        // syscalls/write.rs (A3/A4 ordering).
        let buf = extract_concrete_arg(&args[1], "pwrite64 buf");
        let nbyte = extract_concrete_arg(&args[2], "pwrite64 nbyte");
        let offset = extract_concrete_arg(&args[3], "pwrite64 offset");
        if let Ok(0) = nbyte {
            return Ok(SyscallOutcome::Continue { ret: 0 });
        }
        // Write-demotion (angr-0xyq2 Phase 2) — see procedures/write.rs.
        if state.file_system().demote_symbolic_content(fd as u32) {
            return Err(SyscallError::Other(format!(
                "pwrite64 to fd={fd} with symbolic content falls back to Python (demoted)"
            )));
        }
        let buf = buf?;
        let nbyte = nbyte?;
        let offset = offset?;
        if nbyte > MAX_IO_SIZE {
            return Err(SyscallError::Other(format!(
                "pwrite64 nbyte {nbyte} exceeds limit"
            )));
        }

        // Gather every concrete byte BEFORE mutating the fd — a symbolic byte
        // must leave the fd untouched so the Python fallback produces the
        // single authoritative write (same discipline as writev).
        let mut bytes: Vec<u8> = Vec::with_capacity(nbyte as usize);
        for i in 0..nbyte {
            let bv = state.memory_load(buf.wrapping_add(i), 1)?;
            match bv.as_u64() {
                Some(v) => bytes.push(v as u8),
                None => {
                    return Err(SyscallError::SymbolicArgument(format!(
                        "symbolic byte in pwrite64 buf+{i}"
                    )));
                }
            }
        }

        let total = bytes.len() as u64;
        // Unreachable after the gate above; choke-point insurance (see
        // FileSystem::write_at).
        if !state.file_system().write_at(fd as u32, offset, &bytes) {
            return Err(SyscallError::Other(format!(
                "pwrite64 to fd={fd} with symbolic content falls back to Python (demoted)"
            )));
        }
        Ok(SyscallOutcome::Continue { ret: total })
    }
}

#[cfg(test)]
#[path = "fd_io_tests.rs"]
mod fd_io_tests;
