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
//!   contain an embedded newline — real fgets stops at the first one). With the
//!   SHORT_READS SimOption set, instead models a variable-length read: a
//!   symbolic real_size in [0, size-1], a NUL at the real_size offset, and a
//!   symbolic return of real_size (matching Python procedures/libc/fgets.py
//!   case 2). Default (SHORT_READS off) stays the non-forking full read.
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
        let short_reads = state.has_option("SHORT_READS");

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

        // Record stdin symbols for posix.dumps(0) export (shared by both paths).
        for name in &names {
            state.record_stdin_symbol(name.clone(), 8);
        }

        if short_reads {
            // Variable-length / short-read model, gated behind the SHORT_READS
            // SimOption (angr-efvao). Mirrors Python procedures/libc/fgets.py
            // case 2: a symbolic `real_size` in [0, size-1], per-byte
            // newline/EOF constraints, a NUL stored at the real_size offset,
            // and a *symbolic return* of real_size — which is the actual
            // downstream fork source (Python's fgets returns real_size, not the
            // buffer pointer). Without SHORT_READS the default path below stays
            // byte-identical and non-forking, so the perf gate is unaffected;
            // enabling SHORT_READS opts into the same state multiplication it
            // already causes Python-side.
            let (real_size, constraints, store_bytes) = {
                let ctx = state.solver().borrow();
                let real_size =
                    RustBV::symbolic(&ctx, format!("fgets_realsize_{}", read_id), bits);
                // EOF is unknown for native symbolic stdin; a fresh symbolic
                // bit soundly over-approximates `simfd.eof()` (the solver may
                // pick eof=true to justify a short read, matching Python).
                let eof = RustBV::symbolic(&ctx, format!("fgets_eof_{}", read_id), 1);
                let nl = RustBV::concrete(b'\n' as u128, 8);
                let nul = RustBV::concrete(0, 8);

                let mut constraints: Vec<RustBV> = Vec::with_capacity(read_count as usize + 1);
                // 0 <= real_size <= size-1 (lower bound is implicit for unsigned).
                constraints.push(real_size.ule(&RustBV::concrete(read_count as u128, bits), &ctx));

                // For each returned byte i:
                //   If(i+1 != real_size,            byte != '\n',
                //      Or(i+2 == size, eof, byte == '\n'))
                // i.e. a non-final byte cannot be a newline, and the final
                // returned byte is justified by running out of space, EOF, or
                // being the terminating newline.
                for (i, byte) in sym_bytes.iter().enumerate() {
                    let idx = i as u64;
                    let cond = RustBV::concrete((idx + 1) as u128, bits).ne(&real_size, &ctx);
                    let then_b = byte.ne(&nl, &ctx);
                    let else_b = if idx + 2 == size {
                        // i+2 == size is a concrete tautology for the last byte.
                        RustBV::concrete(1, 1)
                    } else {
                        eof.or(&byte.eq(&nl, &ctx), &ctx)
                    };
                    constraints.push(cond.ite(&then_b, &else_b, &ctx));
                }

                // Emulate Python's `store(dst, data, size=real_size)` + NUL at
                // dst+real_size as concrete-position ITE stores: byte p is the
                // NUL exactly when real_size == p, else the data byte. Keeping
                // every store at a fixed address avoids symbolic-address
                // concretization (which would enumerate up to size-1 addresses)
                // and stays concrete-loadable downstream. The final slot (index
                // read_count == size-1) is the NUL of a full read; clamping it
                // to NUL unconditionally is harmless for shorter reads (it sits
                // beyond real_size, in undefined-content territory).
                let mut store_bytes: Vec<RustBV> = Vec::with_capacity(read_count as usize + 1);
                for (p, byte) in sym_bytes.iter().enumerate() {
                    let at = RustBV::concrete(p as u128, bits).eq(&real_size, &ctx);
                    store_bytes.push(at.ite(&nul, byte, &ctx));
                }
                store_bytes.push(nul.clone());
                (real_size, constraints, store_bytes)
            };

            // Store the computed bytes at fixed offsets.
            for (i, b) in store_bytes.into_iter().enumerate() {
                state.memory_store(buf.wrapping_add(i as u64), b)?;
            }
            // Apply constraints after storing (each borrows the solver).
            for c in constraints {
                state.add_constraint(c);
            }

            // Return symbolic real_size — the downstream fork source.
            return Ok(Some(real_size));
        }

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
