//! Native fgets, fgetc, and getchar implementations.
//!
//! These procedures handle stdin input by creating symbolic bytes.
//! fgets stores symbolic bytes to a buffer, fgetc/getchar return a single symbolic byte.
//!
//! # Behavior
//!
//! - For stdin (resolved `_fileno == 0`): creates symbolic BVS variables
//!   representing user input
//! - fgets: stores up to (size-1) symbolic bytes + NUL terminator at buf, and
//!   constrains every byte but the last to be non-newline (a full read cannot
//!   contain an embedded newline — real fgets stops at the first one). Short
//!   reads / EOF are not modeled (would need variable-length semantics).
//! - fgetc: returns one symbolic byte zero-extended to int
//! - getchar: equivalent to fgetc(stdin) — always stdin, no FILE* arg
//! - A non-stdin FILE* (resolved `_fileno > 0`) falls back to Python's SimFile
//!   model, which serves concrete file content and adds the EOF/newline
//!   constraints this native path does not. An invalid stream (`_fileno < 0`)
//!   returns the error sentinel, matching Python `fgets`/`fgetc` (which return
//!   -1 when the backing SimFileDescriptor is missing). A symbolic FILE* or
//!   symbolic `_fileno` also falls back to Python.

use super::{ProcedureError, symbol_counter};
use crate::procedures::fileops::read_fileno;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const MAX_FGETS_SIZE: u64 = 4096;

/// Resolve a `FILE *stream` argument to its backing fd for a read-side stdio
/// procedure (`fgets`/`fgetc`).
///
/// Mirrors Python's `stream->_fileno` resolution with one pragmatic carve-out:
/// the cle-provided standard streams (`stdin`/`stdout`/`stderr`) live in the
/// `cle##externs` object, which is mapped lazily in Rust and is *not fetchable
/// from a SimProcedure context* (only the VEX interpreter can fetch lazy pages).
/// So `read_fileno` hits an unmapped page and errors for the very common
/// `fgets(buf, n, stdin)` call, forcing a ~100ms Python fallback per call
/// (angr-defcamp_r100 regressed 88% after the native fd-resolution landed).
///
/// A `FILE *` whose struct is unmapped in Rust can only be a cle standard
/// stream — `fopen`'d files allocate their `_IO_FILE` in Rust memory (native
/// `fopen`), so their `_fileno` resolves normally. For a read like `fgets`, a
/// cle standard stream is overwhelmingly stdin (reading from stdout/stderr is a
/// programming error that does not occur in practice), and Python resolves
/// stdin's `_fileno` to 0. So on a *memory* error we serve fd 0 natively,
/// matching Python's stdin path. A *symbolic* `_fileno` still falls back to
/// Python (we cannot pick a branch).
fn resolve_stream_fd(state: &RustSimState, stream: u64) -> Result<i32, ProcedureError> {
    match read_fileno(state, stream) {
        Ok(fd) => Ok(fd),
        // FILE struct not in Rust memory => cle standard stream => stdin.
        Err(ProcedureError::Memory(_)) => Ok(0),
        Err(e) => Err(e),
    }
}

crate::declare_proc! {
    /// Native fgets implementation.
    ///
    /// ```c
    /// char *fgets(char *s, int size, FILE *stream);
    /// ```
    ///
    /// Creates (size-1) symbolic bytes and a NUL terminator at the buffer.
    /// Returns the buffer address on success.
    ///
    /// Resolves the backing fd via [`resolve_stream_fd`]: only stdin (fd 0) uses
    /// this native symbolic-stdin path. A non-stdin file stream falls back to
    /// Python, an invalid fd returns -1, and a symbolic FILE*/`_fileno` falls
    /// back to Python.
    name = "fgets",
    struct = NativeFgets,
    args = [buf: concrete, size: concrete, stream: concrete],
    call |state| {
        let bits = state.arch().bits();
        if size == 0 {
            // fgets with size 0 returns NULL
            return Ok(Some(RustBV::concrete(0, bits)));
        }

        if size > MAX_FGETS_SIZE {
            return Err(ProcedureError::Other(format!(
                "fgets size {} exceeds limit",
                size
            )));
        }

        // Resolve the backing fd. Only stdin (fd 0) is served natively.
        let fd = resolve_stream_fd(state, stream)?;
        if fd < 0 {
            // Invalid stream: Python fgets returns -1 (missing SimFileDescriptor).
            return Ok(Some(RustBV::concrete((-1i64 as u64) as u128, bits)));
        }
        if fd != 0 {
            return Err(ProcedureError::Other(format!(
                "fgets from fd={} (non-stdin) falls back to Python",
                fd
            )));
        }

        let read_count = size - 1; // fgets reads at most size-1 bytes
        let read_id = symbol_counter("fgets");

        // Create symbolic bytes and record for stdin tracking
        let names: Vec<String> = (0..read_count)
            .map(|i| format!("stdin_fgets_{}_{}", read_id, i))
            .collect();

        let sym_bytes: Vec<RustBV> = {
            let ctx = state.solver().borrow();
            names
                .iter()
                .map(|name| RustBV::symbolic(&ctx, name, 8))
                .collect()
        };

        // Newline path constraint: a *full* read of `read_count` bytes means no
        // newline terminated the read before its final byte — real fgets stops
        // at (and includes) the first newline, so any embedded newline followed
        // by further data is impossible. Constrain every byte except the last
        // to be non-newline; the last byte may legitimately be the line's
        // terminating newline (a line exactly filling the buffer). Without this,
        // the native path over-approximates and keeps infeasible
        // newline-in-middle-of-a-full-read states that Python's SimFile model
        // (procedures/libc/fgets.py) prunes. We do not model short reads / EOF
        // here — that needs variable-length read semantics (see angr-abora).
        let newline_conds: Vec<RustBV> = {
            let ctx = state.solver().borrow();
            let nl = RustBV::concrete(b'\n' as u128, 8);
            sym_bytes
                .iter()
                .take((read_count as usize).saturating_sub(1))
                .map(|b| b.ne(&nl, &ctx))
                .collect()
        };

        // Record stdin symbols for posix.dumps(0) export
        for name in &names {
            state.record_stdin_symbol(name.clone(), 8);
        }

        // Store symbolic bytes to buffer
        for (i, sym_byte) in sym_bytes.into_iter().enumerate() {
            state.memory_store(buf.wrapping_add(i as u64), sym_byte)?;
        }

        // Apply the newline constraints after storing (add_constraint borrows
        // the solver, which the byte-creation block above held).
        for cond in newline_conds {
            state.add_constraint(cond);
        }

        // Store NUL terminator
        state.memory_store(buf.wrapping_add(read_count), RustBV::concrete(0, 8))?;

        // Return buffer address
        Ok(Some(RustBV::concrete(buf as u128, bits)))
    }
}

crate::declare_proc! {
    /// Native fgetc implementation.
    ///
    /// ```c
    /// int fgetc(FILE *stream);
    /// ```
    ///
    /// Returns one symbolic byte zero-extended to int size.
    ///
    /// Resolves the backing fd via [`resolve_stream_fd`]: only stdin (fd 0) is
    /// served natively. A non-stdin stream falls back to Python, an invalid fd
    /// returns -1 (EOF sentinel, matching Python `fgetc`), and a symbolic
    /// FILE*/`_fileno` falls back to Python.
    name = "fgetc",
    struct = NativeFgetc,
    args = [stream: concrete],
    aliases = ["fgetc_unlocked"],
    call |state| {
        let fd = resolve_stream_fd(state, stream)?;
        if fd < 0 {
            // Invalid stream: Python fgetc returns -1 (missing descriptor).
            return Ok(Some(RustBV::concrete((-1i64 as u64) as u128, 32)));
        }
        if fd != 0 {
            return Err(ProcedureError::Other(format!(
                "fgetc from fd={} (non-stdin) falls back to Python",
                fd
            )));
        }
        let read_id = symbol_counter("fgetc");
        let name = format!("stdin_fgetc_{}", read_id);
        let result = {
            let ctx = state.solver().borrow();
            let sym_byte = RustBV::symbolic(&ctx, &name, 8);
            // Zero-extend to int (32-bit, matching C int type)
            sym_byte.zero_extend(32, &ctx)
        };
        // Record for posix.dumps(0) export
        state.record_stdin_symbol(name, 8);
        Ok(Some(result))
    }
}

crate::declare_proc! {
    /// Native getchar implementation.
    ///
    /// ```c
    /// int getchar(void);
    /// ```
    ///
    /// Equivalent to fgetc(stdin). Returns one symbolic byte zero-extended to int.
    name = "getchar",
    struct = NativeGetchar,
    args = [],
    aliases = ["getchar_unlocked"],
    call |state| {
        let read_id = symbol_counter("getchar");
        let name = format!("stdin_getchar_{}", read_id);
        let result = {
            let ctx = state.solver().borrow();
            let sym_byte = RustBV::symbolic(&ctx, &name, 8);
            // Zero-extend to int (32-bit, matching C int type)
            sym_byte.zero_extend(32, &ctx)
        };
        // Record for posix.dumps(0) export
        state.record_stdin_symbol(name, 8);
        Ok(Some(result))
    }
}

crate::declare_proc! {
    /// Native getc implementation (alias for fgetc).
    name = "getc",
    struct = NativeGetc,
    args = [stream: bv],
    aliases = ["getc_unlocked"],
    call |state| {
        NativeFgetc.call(state, std::slice::from_ref(&stream))
    }
}

#[cfg(test)]
#[path = "fgets_tests.rs"]
mod tests;
