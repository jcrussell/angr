//! Native stdio shim implementations: fwrite, fflush, setvbuf.
//!
//! These are the highest-volume libc stdio procedures the Python fallback
//! still serviced (per angr-otjw spike). fflush / setvbuf are no-ops that
//! always return 0; fwrite resolves the FILE struct's `_fileno` field via
//! the shared [`super::fileops::resolve_stream_fd_or_demote_all`] helper and
//! reuses the NativeWrite path for stdout/stderr.

use super::arch_word;
use super::fileops::{read_fileno, resolve_stream_fd_or_demote_all};
use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;
use crate::syscalls::MAX_IO_SIZE as MAX_FWRITE_SIZE;

/// Native fwrite implementation.
///
/// ```c
/// size_t fwrite(const void *src, size_t size, size_t nmemb, FILE *stream);
/// ```
///
/// Resolves `stream->_fileno` from the FILE struct and writes the payload
/// to the matching fd buffer. Any non-negative fd is handled inline (like
/// `NativeFputs`); a negative `_fileno` propagates -1.
/// Returns `size * nmemb` on success, matching angr's Python fwrite
/// (which delegates to SimFileDescriptor.write and returns byte count).
pub(crate) struct NativeFwrite;

impl NativeSimProcedure for NativeFwrite {
    fn name(&self) -> &'static str {
        "fwrite"
    }

    fn aliases(&self) -> &'static [&'static str] {
        &["fwrite_unlocked"]
    }

    fn num_args(&self) -> usize {
        4
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        // Resolve the fd FIRST: the write-intent bounces below must know the
        // target fd so they can demote its bounded symbolic content before
        // falling back (angr-0xyq2 A4, enforced by the helper).
        let fd_signed = resolve_stream_fd_or_demote_all(state, &args[3])?;

        if fd_signed < 0 {
            // FILE not backed by a real fd — propagate -1 per fwrite spec.
            return Ok(Some(arch_word(state, -1i64 as u64)));
        }
        // Any non-negative fd is serviced via write_fd (FileSystem::write
        // appends to the fd's content buffer), matching NativeFputs and
        // Python fwrite's `simfd.write` for an arbitrary fd. No fd-1/2
        // narrowing — that was a stale holdover from when fwrite only reused
        // the stdout/stderr NativeWrite path.

        // Deferred `?`: a concrete zero-length fwrite is a POSIX no-op that
        // returns 0 WITHOUT demoting (A3); every other post-resolution
        // bounce (symbolic size/nmemb/src, oversize) demotes first via the
        // gate below (A4).
        let src = extract_concrete_arg(&args[0], "src");
        let size = extract_concrete_arg(&args[1], "size");
        let nmemb = extract_concrete_arg(&args[2], "nmemb");
        if let (Ok(s), Ok(n)) = (&size, &nmemb)
            && s.saturating_mul(*n) == 0
        {
            return Ok(Some(arch_word(state, 0u64)));
        }

        // Write-demotion (angr-0xyq2 Phase 2): a write to a file with
        // bounded symbolic content hands the file to Python for good.
        if state
            .file_system()
            .demote_symbolic_content(fd_signed as u32)
        {
            return Err(ProcedureError::Other(format!(
                "fwrite to fd={fd_signed} with symbolic content falls back to Python (demoted)"
            )));
        }

        let src = src?;
        let size = size?;
        let nmemb = nmemb?;
        let total = size.saturating_mul(nmemb);
        if total > MAX_FWRITE_SIZE {
            return Err(ProcedureError::Other(format!(
                "fwrite byte count {total} exceeds limit"
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
        // Unreachable after the gate above; choke-point insurance (see
        // FileSystem::write).
        if !state.write_fd(fd_signed as u32, &bytes) {
            return Err(ProcedureError::Other(format!(
                "fwrite to fd={fd_signed} with symbolic content falls back to Python (demoted)"
            )));
        }

        Ok(Some(arch_word(state, total)))
    }
}

/// Native fflush implementation.
///
/// ```c
/// int fflush(FILE *stream);
/// ```
///
/// angr's Python proc returns 0 unconditionally — we match.
pub(crate) struct NativeFflush;

impl NativeSimProcedure for NativeFflush {
    fn name(&self) -> &'static str {
        "fflush"
    }

    fn aliases(&self) -> &'static [&'static str] {
        &["fflush_unlocked"]
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        _args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        Ok(Some(arch_word(state, 0u64)))
    }
}

/// Native setvbuf implementation.
///
/// ```c
/// int setvbuf(FILE *stream, char *buf, int type, size_t size);
/// ```
///
/// angr's Python proc returns 0 unconditionally — we match.
pub(crate) struct NativeSetvbuf;

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
        Ok(Some(arch_word(state, 0u64)))
    }
}

/// Native setbuf implementation.
///
/// ```c
/// void setbuf(FILE *stream, char *buf);
/// ```
///
/// angr's Python proc (`procedures/libc/setbuf.py`) is a void no-op
/// (`run(stream, buf): return`). We match: both args are ignored and no
/// return register is written (`Ok(None)`), so the proc just performs the
/// return-address dance — parity holds for symbolic args too.
pub(crate) struct NativeSetbuf;

impl NativeSimProcedure for NativeSetbuf {
    fn name(&self) -> &'static str {
        "setbuf"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        _state: &mut RustSimState,
        _args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        Ok(None)
    }
}

/// Native feof implementation.
///
/// ```c
/// int feof(FILE *stream);
/// ```
///
/// Resolves `stream->_fileno`, then returns 1 if the fd's read position is at
/// (or past) the content end (`FileSystem::fd_pos_and_size`, whose length leg
/// is the max of the concrete buffer length and the bounded symbolic
/// `content_sym` length, angr-0xyq2),
/// and 0 otherwise. The Python proc (`procedures/libc/feof.py`) wraps the
/// same check in a claripy `If` against `simfd.eof()`. In our model the
/// content *length* is always concrete (bounded symbolic files have a
/// concrete size), so the boolean is concrete too.
///
/// Both "no such fd" cases — a negative/closed `stream->_fileno`, and an fd
/// `fd_pos_and_size` does not track — return a best-effort 0 (not EOF), never
/// a distinguished error status. The Python proc short-circuits on
/// `simfd is None` by returning `None`, which leaves the return register
/// untouched; a native proc has no way to express that, and 0 is the safe
/// default (a caller's `while (!feof(f))` keeps making progress rather than
/// treating an untracked stream as exhausted). Covered by
/// `test_feof_negative_fd_returns_zero`.
pub(crate) struct NativeFeof;

impl NativeSimProcedure for NativeFeof {
    fn name(&self) -> &'static str {
        "feof"
    }

    fn aliases(&self) -> &'static [&'static str] {
        &["feof_unlocked"]
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
        let fd = read_fileno(state, file_ptr)?;
        if fd < 0 {
            return Ok(Some(arch_word(state, 0u64)));
        }
        // effective_len = max(concrete, symbolic content_sym length)
        // (angr-0xyq2 Phase 2) — identical to the concrete content length
        // for fds without bounded symbolic content. Single map lookup:
        // feof is hot in `while (!feof(f))` guest loops.
        let at_eof = match state.file_system_ref().fd_pos_and_size(fd as u32) {
            Some((pos, len)) => pos as usize >= len,
            None => false,
        };
        Ok(Some(arch_word(state, u64::from(at_eof))))
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
pub(crate) struct NativeFerror;

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
        Ok(Some(arch_word(state, 0u64)))
    }
}

/// Maximum string length scanned by fputs. `fputs` takes no length argument, so
/// this is a C-string scan bound rather than a payload-size one: it derives from
/// the shared [`super::strings::MAX_STRING_SCAN`] home every other NUL-scanning
/// procedure uses, widened to `u64` for the byte-offset loop below. Both shared
/// caps are 4096 today, so the two stdio write paths still agree on their upper
/// bound on payload size.
const MAX_FPUTS_LEN: u64 = super::strings::MAX_STRING_SCAN as u64;

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
/// Any non-negative fd is handled inline: the bytes are appended to the
/// tracked fd buffer via `write_fd` (`FileSystem::write`), regardless of
/// underlying backing. `NativeFwrite` writes the same way.
pub(crate) struct NativeFputs;

impl NativeSimProcedure for NativeFputs {
    fn name(&self) -> &'static str {
        "fputs"
    }

    fn aliases(&self) -> &'static [&'static str] {
        &["fputs_unlocked"]
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
        let fd = resolve_stream_fd_or_demote_all(state, &args[1])?;
        if fd < 0 {
            return Ok(Some(arch_word(state, -1i64 as u64)));
        }

        // No pre-scan demote gate: a zero-length fputs("") must stay a
        // no-demotion no-op (A3), and only the scan can tell. The write
        // choke point (FileSystem::write) demotes-and-refuses non-empty
        // writes to symbolic-content fds; scan bounces demote explicitly
        // below (A4) so a Python-handled write never leaves stale serving.
        let mut bytes = Vec::new();
        let scan = (|| -> Result<(), ProcedureError> {
            for i in 0..MAX_FPUTS_LEN {
                let bv = state.memory_load(str_addr.wrapping_add(i), 1)?;
                let v = bv.as_u64().ok_or_else(|| {
                    ProcedureError::SymbolicArgument(format!("symbolic byte at s+{i}"))
                })?;
                if v == 0 {
                    return Ok(());
                }
                bytes.push(v as u8);
            }
            Err(ProcedureError::Other(format!(
                "fputs source not NUL-terminated within {MAX_FPUTS_LEN} bytes"
            )))
        })();
        if let Err(e) = scan {
            // Write-intent bounce with the fd resolved — demote first (A4).
            state.file_system().demote_symbolic_content(fd as u32);
            return Err(e);
        }
        if !state.write_fd(fd as u32, &bytes) {
            return Err(ProcedureError::Other(format!(
                "fputs to fd={fd} with symbolic content falls back to Python (demoted)"
            )));
        }
        Ok(Some(arch_word(state, 1u64)))
    }
}

test_submod!("stdio_tests.rs" => stdio_tests);
