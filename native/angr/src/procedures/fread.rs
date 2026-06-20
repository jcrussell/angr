//! Native fread / fread_unlocked implementation.
//!
//! ```c
//! size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
//! ```
//!
//! Resolves the backing fd from `stream->_fileno` (like fseek/ftell/fwrite),
//! then:
//!   - serves concrete content from the Rust `FileSystem` when the fd is open
//!     and still has unconsumed bytes (`pos < content_len`), advancing the
//!     position — mirroring `read.rs`.
//!
//! Falls back to Python (returns `Err`) for:
//!   - symbolic `stream` pointer or symbolic `FILE._fileno`;
//!   - symbolic `size` / `nmemb` (the macro rejects these before the body runs);
//!   - `total = size * nmemb` beyond `MAX_FREAD_SIZE`;
//!   - fds NOT open in the Rust `FileSystem` (Python's symbolic-file model owns
//!     those — see the fd-table invariant in `read.rs`);
//!   - fds with no concrete content (symbolic SimFiles) — Python's symbolic-file
//!     model owns those, and synthesizing fresh bytes here would drop the
//!     SimFile's own constraints.
//!
//! NOTE (angr-m674p): this native fread does NOT fix asisctffinals2015_license.
//! There `fread` is called with a SYMBOLIC `size` (the file size, rbp at
//! 0x400a4e), so the macro returns `SymbolicArgument` before this body runs and
//! the call falls back to Python. The license hang is the Rust→Python state
//! export for that callback churning the shared Z3 solver on the symbolic
//! `filesize_*` — NOT the fread logic itself. See bd memory
//! `license-timeout-fread-root-cause`. This procedure is an additive win for
//! concrete-content fread only.

use super::ProcedureError;
use super::strings::write_concrete_bytes;
use crate::procedures::fileops::read_fileno;
use crate::symbolic::RustBV;

const MAX_FREAD_SIZE: u64 = 4096;

crate::declare_proc! {
    /// fread: serve concrete FS content or synthesize symbolic bytes for a
    /// natively-opened file. See module docs for the fallback matrix.
    name = "fread",
    struct = NativeFread,
    args = [dst: concrete, size: concrete, nmemb: concrete, stream: bv],
    call |state| {
        let bits = state.arch().bits();

        // size * nmemb; a zero count returns 0 items read.
        let total = size.saturating_mul(nmemb);
        if total == 0 {
            return Ok(Some(RustBV::concrete(0, bits)));
        }
        if total > MAX_FREAD_SIZE {
            return Err(ProcedureError::Other(format!(
                "fread total {} exceeds limit",
                total
            )));
        }

        // Resolve the backing fd from the FILE struct. Symbolic stream pointer
        // or symbolic _fileno falls back to Python.
        let stream_ptr = stream.as_u64().ok_or_else(|| {
            ProcedureError::SymbolicArgument("fread stream pointer".to_string())
        })?;
        let fd = read_fileno(state, stream_ptr)?;
        if fd < 0 {
            // Invalid stream: 0 items read.
            return Ok(Some(RustBV::concrete(0, bits)));
        }
        let fd_u32 = fd as u32;

        let (open, content_len) = match state.file_system_ref().fd_info(fd_u32) {
            Some((_, _, _, len, is_open)) => (is_open, len),
            None => (false, 0),
        };
        if !open {
            // Not a Rust-tracked fd — Python's symbolic-file model owns it.
            return Err(ProcedureError::Other(format!(
                "fread from fd={} (not open in Rust FileSystem) falls back to Python",
                fd
            )));
        }

        // Empty / fully-consumed content defers to Python so the symbolic-file
        // model (cle simfs + SimFile) can produce symbolic bytes — mirroring
        // read.rs. Synthesizing fresh symbolic bytes here would lose the
        // SimFile's own constraints (e.g. the license bench's `byte != '\n'`),
        // diverging from Python and risking path explosion.
        if content_len == 0 {
            return Err(ProcedureError::Other(format!(
                "fread from fd={} has no concrete content; falling back to Python",
                fd
            )));
        }

        // Serve concrete bytes from the FS buffer, advancing position.
        let bytes = state.file_system().read(fd_u32, total as usize);
        let n = bytes.len();
        write_concrete_bytes(state, dst, &bytes)?;
        // fread returns the number of complete items read.
        let items = (n as u64) / size;
        Ok(Some(RustBV::concrete(items as u128, bits)))
    }
}

crate::declare_proc! {
    /// fread_unlocked: identical semantics to fread (no locking in our model).
    name = "fread_unlocked",
    struct = NativeFreadUnlocked,
    args = [dst: bv, size: bv, nmemb: bv, stream: bv],
    call |state| {
        NativeFread.call(state, &[dst, size, nmemb, stream])
    }
}

#[cfg(test)]
#[path = "fread_tests.rs"]
mod fread_tests;
