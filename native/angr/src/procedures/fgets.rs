//! Native fgets, fgetc, and getchar implementations.
//!
//! These procedures handle stdin input by creating symbolic bytes.
//! fgets stores symbolic bytes to a buffer, fgetc/getchar return a single symbolic byte.
//!
//! # Behavior
//!
//! - For stdin (resolved `_fileno == 0`): creates symbolic BVS variables
//!   representing user input
//! - fgets: stores up to (size-1) symbolic bytes + NUL terminator at buf
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
use crate::symbolic::RustBV;

const MAX_FGETS_SIZE: u64 = 4096;

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
    /// Resolves the backing fd from `stream->_fileno`: only stdin (fd 0) uses
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
        let fd = read_fileno(state, stream)?;
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

        // Record stdin symbols for posix.dumps(0) export
        for name in &names {
            state.record_stdin_symbol(name.clone(), 8);
        }

        // Store symbolic bytes to buffer
        for (i, sym_byte) in sym_bytes.into_iter().enumerate() {
            state.memory_store(buf.wrapping_add(i as u64), sym_byte)?;
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
    /// Resolves `stream->_fileno`: only stdin (fd 0) is served natively. A
    /// non-stdin stream falls back to Python, an invalid fd returns -1 (EOF
    /// sentinel, matching Python `fgetc`), and a symbolic FILE*/`_fileno`
    /// falls back to Python.
    name = "fgetc",
    struct = NativeFgetc,
    args = [stream: concrete],
    call |state| {
        let fd = read_fileno(state, stream)?;
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
    call |state| {
        NativeFgetc.call(state, std::slice::from_ref(&stream))
    }
}

#[cfg(test)]
#[path = "fgets_tests.rs"]
mod tests;
