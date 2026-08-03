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
//!   - serves bounded symbolic content (`content_sym`, angr-0xyq2 Phase 2)
//!     via `FileSystem::read_sym`, returning the truncated item count
//!     (bytes served / size) with the position advanced by bytes served —
//!     Python fread semantics (`simfd.read(dst, size*nm)` then `ret // size`).
//!     Totals are clamped to `MAX_SYMFILE_SERVE_SIZE` (= the export cap, so
//!     any registered file serves in one call — a smaller cap would round
//!     `served / size` to 0 forever for items larger than it), never
//!     bounced — a fallback would split the position cursor (angr-8j16).
//!
//! Falls back to Python (returns `Err`) for:
//!   - symbolic `stream` pointer or symbolic `FILE._fileno`;
//!   - symbolic `size` / `nmemb` (the macro rejects these before the body runs);
//!   - `total = size * nmemb` beyond `MAX_FREAD_SIZE` (non-`content_sym` fds);
//!   - fds NOT open in the Rust `FileSystem` (Python's symbolic-file model owns
//!     those — see the fd-table invariant in `read.rs`);
//!   - fds with neither concrete nor registered symbolic content — Python's
//!     symbolic-file model owns those, and synthesizing fresh bytes here would
//!     drop the SimFile's own constraints.
//!
//! NOTE (angr-m674p, superseded by angr-0xyq2): pre-Phase-3, this native
//! fread did NOT fix asisctffinals2015_license — `fread`'s `size` came from
//! fstat/ftell on the symbolic file (symbolic `filesize_*`), so the macro
//! bounced `SymbolicArgument` and the Rust→Python export churned the shared
//! Z3 solver. With the bounded-symbolic-content export (angr-0xyq2 Phase 3),
//! the file is registered natively and `effective_len` makes fstat/ftell
//! CONCRETE, so the same fread arrives with a concrete size and is served
//! from `content_sym` — the license bench now passes in ~0.9s.

use super::ProcedureError;
use super::strings::{write_bv_bytes, write_concrete_bytes};
use crate::procedures::fileops::read_fileno;
use crate::state::MAX_SYMFILE_SERVE_SIZE;
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
                "fread from fd={fd} (not open in Rust FileSystem) falls back to Python"
            )));
        }

        // Bounded symbolic file content (angr-0xyq2 Phase 2): serve the
        // registered per-byte BVs natively. Matches Python fread
        // (`procedures/libc/fread.py`) exactly: `simfd.read(dst, size*nm)`
        // consumes ALL available bytes up to size*nmemb (position advances by
        // bytes read, not by items*size) and the return is the truncated item
        // count `ret // size`. Clamped to MAX_SYMFILE_SERVE_SIZE (= the
        // export cap; whole file in one call, matching Python) rather than
        // bounced — see the module docs.
        let clamped = total.min(MAX_SYMFILE_SERVE_SIZE) as usize;
        if let Some(sym_bytes) = state.file_system().read_sym(fd_u32, clamped) {
            let n = sym_bytes.len();
            write_bv_bytes(state, dst, sym_bytes)?;
            if n > 0 {
                crate::symbolic::record_symfile_read_native();
            }
            let items = (n as u64) / size;
            return Ok(Some(RustBV::concrete(items as u128, bits)));
        }
        if total > MAX_FREAD_SIZE {
            return Err(ProcedureError::Other(format!(
                "fread total {total} exceeds limit"
            )));
        }
        // Empty / fully-consumed content defers to Python so the symbolic-file
        // model (cle simfs + SimFile) can produce symbolic bytes. This
        // deliberately DIVERGES from read.rs, which for content_len==0 mints
        // fresh symbolic bytes in-place via read_file_symbolic (angr-gorvf.15)
        // to avoid the Python round-trip. fread keeps bouncing on purpose:
        // synthesizing fresh symbolic bytes here would lose the SimFile's own
        // constraints (e.g. the license bench's `byte != '\n'`), diverging from
        // Python and risking path explosion. Do not "unify" this with
        // read.rs's fresh-mint path without re-checking the license bench.
        if content_len == 0 {
            return Err(ProcedureError::Other(format!(
                "fread from fd={fd} has no concrete content; falling back to Python"
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
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod fread_tests;
