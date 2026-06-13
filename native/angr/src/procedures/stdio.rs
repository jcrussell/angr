//! Native stdio shim implementations: fwrite, fflush, setvbuf.
//!
//! These are the highest-volume libc stdio procedures the Python fallback
//! still serviced (per angr-otjw spike). fflush / setvbuf are no-ops that
//! always return 0; fwrite resolves the FILE struct's `_fileno` field via
//! an arch-specific offset and reuses the NativeWrite path for stdout/stderr.

use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const MAX_FWRITE_SIZE: u64 = 4096;

/// `_IO_FILE._fileno` byte offset per arch, mirroring
/// `cle.backends.externs.simdata.io_file.io_file_data_for_arch`.
fn fd_offset_for_arch(name: &str) -> Option<u64> {
    match name {
        "AMD64" => Some(112),
        "X86" => Some(56),
        "ARM" => Some(14),
        "ARM64" => Some(20),
        "MIPS32" => Some(56),
        "MIPS64" => Some(112),
        _ => None,
    }
}

/// Native fwrite implementation.
///
/// ```c
/// size_t fwrite(const void *src, size_t size, size_t nmemb, FILE *stream);
/// ```
///
/// Resolves `stream->_fileno` from the FILE struct and writes the payload
/// to the matching fd buffer (only fds 1/2 handled — others fall back).
/// Returns `size * nmemb` on success, matching angr's Python fwrite
/// (which delegates to SimFileDescriptor.write and returns byte count).
pub struct NativeFwrite;

impl NativeSimProcedure for NativeFwrite {
    fn name(&self) -> &'static str {
        "fwrite"
    }

    fn num_args(&self) -> usize {
        4
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let src = extract_concrete_arg(&args[0], "src")?;
        let size = extract_concrete_arg(&args[1], "size")?;
        let nmemb = extract_concrete_arg(&args[2], "nmemb")?;
        let file_ptr = extract_concrete_arg(&args[3], "file_ptr")?;

        let total = size.saturating_mul(nmemb);
        if total > MAX_FWRITE_SIZE {
            return Err(ProcedureError::Other(format!(
                "fwrite byte count {} exceeds limit",
                total
            )));
        }

        // Resolve fd from the FILE struct.
        let arch_name = state.arch().name();
        let fd_off = fd_offset_for_arch(arch_name).ok_or_else(|| {
            ProcedureError::Other(format!(
                "fwrite: no _IO_FILE fd offset for arch {arch_name}"
            ))
        })?;
        let fd_bv = state.memory_load(file_ptr.wrapping_add(fd_off), 4)?;
        let fd_raw = fd_bv
            .as_u64()
            .ok_or_else(|| ProcedureError::SymbolicArgument("FILE._fileno".to_string()))?;
        let fd_signed = fd_raw as u32 as i32;

        let bits = state.arch().bits();
        if fd_signed < 0 {
            // FILE not backed by a real fd — propagate -1 per fwrite spec.
            return Ok(Some(RustBV::concrete((-1i64 as u64) as u128, bits)));
        }
        if fd_signed != 1 && fd_signed != 2 {
            return Err(ProcedureError::Other(format!(
                "fwrite to fd={fd_signed} not supported natively"
            )));
        }

        let mut bytes = Vec::with_capacity(total as usize);
        for i in 0..total {
            match state.memory_load(src.wrapping_add(i), 1) {
                Ok(bv) => match bv.as_u64() {
                    Some(val) => bytes.push(val as u8),
                    None => {
                        return Err(ProcedureError::SymbolicArgument(format!(
                            "symbolic byte at src+{i}"
                        )));
                    }
                },
                Err(e) => return Err(e.into()),
            }
        }
        state.write_fd(fd_signed as u32, &bytes);

        Ok(Some(RustBV::concrete(total as u128, bits)))
    }
}

/// Native fflush implementation.
///
/// ```c
/// int fflush(FILE *stream);
/// ```
///
/// angr's Python proc returns 0 unconditionally — we match.
pub struct NativeFflush;

impl NativeSimProcedure for NativeFflush {
    fn name(&self) -> &'static str {
        "fflush"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        _args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(0, bits)))
    }
}

/// Native setvbuf implementation.
///
/// ```c
/// int setvbuf(FILE *stream, char *buf, int type, size_t size);
/// ```
///
/// angr's Python proc returns 0 unconditionally — we match.
pub struct NativeSetvbuf;

impl NativeSimProcedure for NativeSetvbuf {
    fn name(&self) -> &'static str {
        "setvbuf"
    }

    fn num_args(&self) -> usize {
        4
    }

    fn call(
        &self,
        state: &mut RustSimState,
        _args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(0, bits)))
    }
}

/// Read a 32-bit fd from a FILE struct on the given arch. Returns the signed
/// fd (so -1 sentinels are preserved). Mirrors fileops.rs::read_fileno so the
/// stdio shims don't need to depend on private helpers there.
fn read_fileno_for_stream(state: &RustSimState, file_ptr: u64) -> Result<i32, ProcedureError> {
    let arch_name = state.arch().name();
    let fd_off = fd_offset_for_arch(arch_name).ok_or_else(|| {
        ProcedureError::Other(format!("no _IO_FILE fd offset for arch {arch_name}"))
    })?;
    let bv = state.memory_load(file_ptr.wrapping_add(fd_off), 4)?;
    let raw = bv
        .as_u64()
        .ok_or_else(|| ProcedureError::SymbolicArgument("FILE._fileno".to_string()))?;
    Ok(raw as u32 as i32)
}

/// Native feof implementation.
///
/// ```c
/// int feof(FILE *stream);
/// ```
///
/// Resolves `stream->_fileno`, then returns 1 if the fd's read position is at
/// (or past) the content buffer end, and 0 otherwise. The Python proc
/// (`procedures/libc/feof.py`) wraps the same check in a claripy `If` against
/// `simfd.eof()`. In our model the content buffer is always concrete, so the
/// boolean is concrete too.
///
/// Returns -1 if the fd is not tracked (matches the Python `simfd is None`
/// → `None` short-circuit by signaling "fallback" via a non-zero status). We
/// can't return None from a procedure with a non-void signature, so a
/// best-effort 0 (not EOF) is the safe default — Python's `feof` returns
/// `None` in that case, which the caller would coerce to 0 anyway.
pub struct NativeFeof;

impl NativeSimProcedure for NativeFeof {
    fn name(&self) -> &'static str {
        "feof"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let file_ptr = extract_concrete_arg(&args[0], "stream")?;
        let fd = read_fileno_for_stream(state, file_ptr)?;
        let bits = state.arch().bits();
        if fd < 0 {
            return Ok(Some(RustBV::concrete(0, bits)));
        }
        let at_eof = match state.file_system_ref().fd_info(fd as u32) {
            Some((_, pos, _, len, _)) => pos as usize >= len,
            None => false,
        };
        Ok(Some(RustBV::concrete(if at_eof { 1 } else { 0 }, bits)))
    }
}

/// Native ferror implementation.
///
/// ```c
/// int ferror(FILE *stream);
/// ```
///
/// angr's FileSystem model does not track per-fd I/O errors, so this is always
/// 0 (no error). glibc.json declares `ferror` but ships no Python proc, so
/// without this native entry the dispatcher falls back to angr's default
/// (unbound) handling. Always returning 0 matches a successfully-read stream
/// and is the most useful default for the symbolic-execution use case.
pub struct NativeFerror;

impl NativeSimProcedure for NativeFerror {
    fn name(&self) -> &'static str {
        "ferror"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        _args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(0, bits)))
    }
}

/// Maximum string length scanned by fputs. Matches MAX_FWRITE_SIZE so the two
/// stdio write paths have the same upper bound on payload size.
const MAX_FPUTS_LEN: u64 = 4096;

/// Native fputs implementation.
///
/// ```c
/// int fputs(const char *s, FILE *stream);
/// ```
///
/// Resolves `stream->_fileno`, reads a NUL-terminated string from `s` (up to
/// MAX_FPUTS_LEN bytes), and appends it to the fd buffer. Returns 1 on
/// success and -1 on a closed/negative fd (matching the Python proc's `-1`
/// short-circuit when `simfd is None`).
///
/// Only stdout (1) and stderr (2) are handled inline — other fds fall back to
/// the FileSystem write path the same way fwrite does so the bytes land in
/// the tracked fd buffer regardless of underlying backing.
pub struct NativeFputs;

impl NativeSimProcedure for NativeFputs {
    fn name(&self) -> &'static str {
        "fputs"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let str_addr = extract_concrete_arg(&args[0], "s")?;
        let file_ptr = extract_concrete_arg(&args[1], "stream")?;
        let fd = read_fileno_for_stream(state, file_ptr)?;
        let bits = state.arch().bits();
        if fd < 0 {
            return Ok(Some(RustBV::concrete((-1i64 as u64) as u128, bits)));
        }

        let mut bytes = Vec::new();
        for i in 0..MAX_FPUTS_LEN {
            let bv = state.memory_load(str_addr.wrapping_add(i), 1)?;
            let v = bv.as_u64().ok_or_else(|| {
                ProcedureError::SymbolicArgument(format!("symbolic byte at s+{i}"))
            })?;
            if v == 0 {
                break;
            }
            bytes.push(v as u8);
            if i + 1 == MAX_FPUTS_LEN {
                return Err(ProcedureError::Other(format!(
                    "fputs source not NUL-terminated within {MAX_FPUTS_LEN} bytes"
                )));
            }
        }
        state.write_fd(fd as u32, &bytes);
        Ok(Some(RustBV::concrete(1, bits)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;

    fn setup_file_struct(state: &mut RustSimState, file_ptr: u64, fd: i32) {
        let arch_name = state.arch().name();
        let off = fd_offset_for_arch(arch_name).expect("test arch supported");
        // Map enough room for the FILE struct + buf.
        state.map_memory_data(file_ptr & !0xfff, &vec![0u8; 0x4000], Permission::RWX);
        let fd_bv = RustBV::concrete(fd as u32 as u128, 32);
        state.memory_store(file_ptr + off, fd_bv).unwrap();
    }

    #[test]
    fn test_fwrite_stdout() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"hello world", Permission::RWX);
        let file_ptr = 0x10000u64;
        setup_file_struct(&mut state, file_ptr, 1);

        let result = NativeFwrite
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(1, 64),
                    RustBV::concrete(5, 64),
                    RustBV::concrete(file_ptr as u128, 64),
                ],
            )
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(5));
        assert_eq!(state.stdout_buffer(), b"hello");
    }

    #[test]
    fn test_fwrite_stderr_nmemb_times_size() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"abcdefgh", Permission::RWX);
        let file_ptr = 0x10000u64;
        setup_file_struct(&mut state, file_ptr, 2);

        let result = NativeFwrite
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(2, 64),
                    RustBV::concrete(3, 64),
                    RustBV::concrete(file_ptr as u128, 64),
                ],
            )
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(6));
        assert_eq!(state.fd_buffer(2), b"abcdef");
    }

    #[test]
    fn test_fwrite_unsupported_fd_falls_back() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"abc", Permission::RWX);
        let file_ptr = 0x10000u64;
        setup_file_struct(&mut state, file_ptr, 5);

        let result = NativeFwrite.call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(1, 64),
                RustBV::concrete(3, 64),
                RustBV::concrete(file_ptr as u128, 64),
            ],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_fwrite_negative_fd_returns_minus_one() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"abc", Permission::RWX);
        let file_ptr = 0x10000u64;
        setup_file_struct(&mut state, file_ptr, -1);

        let result = NativeFwrite
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(1, 64),
                    RustBV::concrete(3, 64),
                    RustBV::concrete(file_ptr as u128, 64),
                ],
            )
            .unwrap();

        // -1 as size_t on amd64
        assert_eq!(result.unwrap().as_u64(), Some(u64::MAX));
    }

    #[test]
    fn test_fwrite_too_large() {
        let mut state = RustSimState::new("amd64").unwrap();
        let file_ptr = 0x10000u64;
        setup_file_struct(&mut state, file_ptr, 1);

        let result = NativeFwrite.call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(1, 64),
                RustBV::concrete(8192, 64),
                RustBV::concrete(file_ptr as u128, 64),
            ],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_fflush_returns_zero() {
        let mut state = RustSimState::new("amd64").unwrap();
        let result = NativeFflush
            .call(&mut state, &[RustBV::concrete(0x12345678, 64)])
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(0));
    }

    #[test]
    fn test_setvbuf_returns_zero() {
        let mut state = RustSimState::new("amd64").unwrap();
        let result = NativeSetvbuf
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x10000, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(0));
    }

    // --- feof / ferror / fputs ---

    #[test]
    fn test_feof_at_start_of_empty_fd_is_eof() {
        // An open-but-empty fd: position 0 >= content_len 0 → EOF (1).
        let mut state = RustSimState::new("amd64").unwrap();
        let fd = state
            .file_system()
            .open("empty.txt".to_string(), crate::state::FdFlags::ReadOnly);
        let file_ptr = 0x10000u64;
        setup_file_struct(&mut state, file_ptr, fd as i32);

        let result = NativeFeof
            .call(&mut state, &[RustBV::concrete(file_ptr as u128, 64)])
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(1));
    }

    #[test]
    fn test_feof_with_content_available_returns_zero() {
        // Position 0 with non-empty content → not EOF (0).
        let mut state = RustSimState::new("amd64").unwrap();
        let fd = state.file_system().open_with_content(
            "data.txt".to_string(),
            crate::state::FdFlags::ReadOnly,
            b"hello".to_vec(),
        );
        let file_ptr = 0x10000u64;
        setup_file_struct(&mut state, file_ptr, fd as i32);

        let result = NativeFeof
            .call(&mut state, &[RustBV::concrete(file_ptr as u128, 64)])
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(0));
    }

    #[test]
    fn test_feof_after_consuming_all_bytes_is_eof() {
        // Read everything, position == content_len → EOF.
        let mut state = RustSimState::new("amd64").unwrap();
        let fd = state.file_system().open_with_content(
            "data.txt".to_string(),
            crate::state::FdFlags::ReadOnly,
            b"abc".to_vec(),
        );
        let _ = state.file_system().read(fd, 3);
        let file_ptr = 0x10000u64;
        setup_file_struct(&mut state, file_ptr, fd as i32);

        let result = NativeFeof
            .call(&mut state, &[RustBV::concrete(file_ptr as u128, 64)])
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(1));
    }

    #[test]
    fn test_feof_negative_fd_returns_zero() {
        let mut state = RustSimState::new("amd64").unwrap();
        let file_ptr = 0x10000u64;
        setup_file_struct(&mut state, file_ptr, -1);
        let result = NativeFeof
            .call(&mut state, &[RustBV::concrete(file_ptr as u128, 64)])
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(0));
    }

    #[test]
    fn test_ferror_always_zero() {
        let mut state = RustSimState::new("amd64").unwrap();
        let result = NativeFerror
            .call(&mut state, &[RustBV::concrete(0xdeadbeef, 64)])
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(0));
    }

    #[test]
    fn test_fputs_writes_to_stdout() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"hello\0", Permission::RWX);
        let file_ptr = 0x10000u64;
        setup_file_struct(&mut state, file_ptr, 1);

        let result = NativeFputs
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(file_ptr as u128, 64),
                ],
            )
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(1));
        assert_eq!(state.stdout_buffer(), b"hello");
    }

    #[test]
    fn test_fputs_writes_to_stderr() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"oops\0", Permission::RWX);
        let file_ptr = 0x10000u64;
        setup_file_struct(&mut state, file_ptr, 2);

        let result = NativeFputs
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(file_ptr as u128, 64),
                ],
            )
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(1));
        assert_eq!(state.fd_buffer(2), b"oops");
    }

    #[test]
    fn test_fputs_negative_fd_returns_minus_one() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"abc\0", Permission::RWX);
        let file_ptr = 0x10000u64;
        setup_file_struct(&mut state, file_ptr, -1);

        let result = NativeFputs
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(file_ptr as u128, 64),
                ],
            )
            .unwrap();
        // -1 as size_t on amd64
        assert_eq!(result.unwrap().as_u64(), Some(u64::MAX));
    }

    #[test]
    fn test_fputs_empty_string_writes_nothing_returns_one() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"\0", Permission::RWX);
        let file_ptr = 0x10000u64;
        setup_file_struct(&mut state, file_ptr, 1);

        let result = NativeFputs
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(file_ptr as u128, 64),
                ],
            )
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(1));
        assert_eq!(state.stdout_buffer(), b"");
    }
}
