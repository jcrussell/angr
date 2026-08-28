//! Native fwrite / fwrite_unlocked implementation.
//!
//! ```c
//! size_t fwrite(const void *src, size_t size, size_t nmemb, FILE *stream);
//! ```
//!
//! The write-side counterpart to [`super::fread`]: both resolve the backing fd
//! from `stream->_fileno` and then service the transfer against the Rust
//! `FileSystem` rather than bouncing to Python. Unlike `fread`, the `_unlocked`
//! spelling needs no separate type — it is a plain alias (see
//! [`NativeFwrite::aliases`]), because there is no per-variant argument
//! reshuffling to do.
//!
//! The no-op stdio shims that surround this call in a real program
//! (`fflush` / `setvbuf` / `setbuf`) and the status/write siblings
//! (`feof` / `ferror` / `fputs`) live in [`super::stdio`].

use super::arch_word;
use super::fileops::resolve_stream_fd_or_demote_all;
use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;
use crate::syscalls::MAX_IO_SIZE as MAX_FWRITE_SIZE;

/// Native fwrite implementation.
///
/// Resolves `stream->_fileno` from the FILE struct and writes the payload
/// to the matching fd buffer. Any non-negative fd is handled inline (like
/// [`super::stdio::NativeFputs`]); a negative `_fileno` propagates -1.
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

test_submod!("fwrite_tests.rs" => fwrite_tests);
