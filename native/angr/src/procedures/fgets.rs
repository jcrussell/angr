//! Native fgets, fgetc, and getchar implementations.
//!
//! These procedures handle stdin input by creating symbolic bytes.
//! fgets stores symbolic bytes to a buffer, fgetc/getchar return a single symbolic byte.
//!
//! **Panic policy / enforcement (angr-qwyti.11, angr-9ke6b.212):** this module
//! carries `#![deny(clippy::unwrap_used, clippy::expect_used)]` — the FILE\*
//! and buffer arguments are guest-supplied and every failure to resolve them
//! already returns a `ProcedureError` that falls back to Python. The single
//! statement-level `#[allow]` below (in [`read_stdin_char`]) is the
//! `mint_stdin_bytes` one-name contract, not an input check.
#![deny(clippy::unwrap_used, clippy::expect_used)]
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
//! - fgetc: returns one symbolic byte zero-extended to int. Under SHORT_READS,
//!   returns `If(eof, -1, byte)` to model the EOF (-1) return.
//! - getchar: equivalent to fgetc(stdin) — always stdin, no FILE* arg
//! - A non-stdin FILE* (resolved `_fileno > 0`) falls back to Python's SimFile
//!   model, which serves concrete file content and adds the EOF/newline
//!   constraints this native path does not. An invalid stream (`_fileno < 0`)
//!   returns the error sentinel, matching Python `fgets`/`fgetc` (which return
//!   -1 when the backing SimFileDescriptor is missing). A symbolic FILE* or
//!   symbolic `_fileno` also falls back to Python.

use super::stdin_common::mint_stdin_bytes;
use super::{ProcedureError, symbol_counter};
use crate::procedures::fileops::read_fileno;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const MAX_FGETS_SIZE: u64 = 4096;

/// Default `SimStateLibc.max_gets_size` (angr/state_plugins/libc.py). `gets` has
/// no size argument, so it reads at most `MAX_GETS_SIZE - 1` bytes. The Python
/// knob is a static state attribute not exposed to Rust; a user who raises it to
/// model a larger overflow would diverge here (rare — the default is what nearly
/// every harness uses).
const MAX_GETS_SIZE: u64 = 256;

/// Apply the variable-length symbolic-line ("case 2") model shared by Python's
/// `gets` (always) and `fgets` under SHORT_READS: build a symbolic `real_size`
/// in `[0, size-1]`, add the per-byte newline/EOF constraints, store the data
/// with a NUL at the `real_size` offset (as fixed-address ITE stores), and
/// return the symbolic `real_size`.
///
/// `sym_bytes` are the `size-1` already-created symbolic input bytes (the caller
/// records them as stdin symbols). `label`/`read_id` name the fresh
/// `real_size`/`eof` symbols. Mirrors `procedures/libc/{fgets,gets}.py` case 2.
///
/// Keeping every store at a fixed address avoids symbolic-address
/// concretization (which would enumerate up to `size-1` addresses) and stays
/// concrete-loadable downstream. The final slot (index `size-1`) is the NUL of a
/// full read; clamping it to NUL unconditionally is harmless for shorter reads
/// (it sits beyond `real_size`, in undefined-content territory).
pub(crate) fn store_symbolic_line(
    state: &mut RustSimState,
    buf: u64,
    size: u64,
    sym_bytes: &[RustBV],
    label: &str,
    read_id: u64,
) -> Result<RustBV, ProcedureError> {
    let bits = state.arch().bits();
    let read_count = size - 1;
    let (real_size, constraints, store_bytes) = {
        let ctx = state.solver().borrow();
        let real_size = RustBV::symbolic(&ctx, format!("{label}_realsize_{read_id}"), bits);
        // EOF is unknown for native symbolic stdin; a fresh symbolic bit soundly
        // over-approximates `simfd.eof()` (the solver may pick eof=true to
        // justify a short read, matching Python).
        let eof = RustBV::symbolic(&ctx, format!("{label}_eof_{read_id}"), 1);
        let nl = RustBV::concrete(b'\n' as u128, 8);
        let nul = RustBV::concrete(0, 8);

        let mut constraints: Vec<RustBV> = Vec::with_capacity(read_count as usize + 1);
        // 0 <= real_size <= size-1 (lower bound is implicit for unsigned).
        constraints.push(real_size.ule(&RustBV::concrete(read_count as u128, bits), &ctx));

        // For each returned byte i:
        //   If(i+1 != real_size,            byte != '\n',
        //      Or(i+2 == size, eof, byte == '\n'))
        // i.e. a non-final byte cannot be a newline, and the final returned byte
        // is justified by running out of space, EOF, or being the newline.
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
        // dst+real_size: byte p is the NUL exactly when real_size == p.
        let mut store_bytes: Vec<RustBV> = Vec::with_capacity(read_count as usize + 1);
        for (p, byte) in sym_bytes.iter().enumerate() {
            let at = RustBV::concrete(p as u128, bits).eq(&real_size, &ctx);
            store_bytes.push(at.ite(&nul, byte, &ctx));
        }
        store_bytes.push(nul);
        (real_size, constraints, store_bytes)
    };

    // Store the computed bytes at fixed offsets.
    super::strings::write_bv_bytes(state, buf, store_bytes)?;
    // Apply constraints after storing (each borrows the solver).
    for c in constraints {
        state.add_constraint(c);
    }
    Ok(real_size)
}

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
    aliases = ["fgets_unlocked"],
    call |state| {
        let bits = state.arch().bits();
        if size == 0 {
            // fgets with size 0 returns NULL
            return Ok(Some(RustBV::concrete(0, bits)));
        }

        if size > MAX_FGETS_SIZE {
            return Err(ProcedureError::Other(format!(
                "fgets size {size} exceeds limit"
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
                "fgets from fd={fd} (non-stdin) falls back to Python"
            )));
        }

        let read_count = size - 1; // fgets reads at most size-1 bytes
        let read_id = symbol_counter("fgets");
        let short_reads = state.has_option("SHORT_READS");

        // Create symbolic bytes and record for stdin tracking
        let names: Vec<String> = (0..read_count)
            .map(|i| format!("stdin_fgets_{read_id}_{i}"))
            .collect();

        // Mint the leaves, consuming any harness-seeded fd-0 bytes and recording
        // the unseeded ones for posix.dumps(0) export (shared by both paths).
        let sym_bytes: Vec<RustBV> = mint_stdin_bytes(state, &names);

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
            // Return symbolic real_size — the downstream fork source.
            let real_size = store_symbolic_line(state, buf, size, &sym_bytes, "fgets", read_id)?;
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
        super::strings::write_bv_bytes(state, buf, sym_bytes)?;

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

/// Read one symbolic byte from stdin and widen it to a C `int` return value,
/// applying the SHORT_READS EOF model.
///
/// Shared body of `fgetc` and `getchar`: Python defines `getchar` as
/// `fgetc(stdin)`, so the two must not diverge — a change to the eof-bit
/// naming or the -1 sentinel has to land in exactly one place.
///
/// `tag` selects the per-procedure [`symbol_counter`] and names the fresh
/// symbols (`stdin_<tag>_<id>` for the byte, `<tag>_eof_<id>` for the eof
/// bit), so the two callers keep distinct symbol namespaces.
///
/// Returns `zero_extend(byte, 32)` by default. Under the SHORT_READS
/// SimOption, returns `If(eof, -1, byte)` for a fresh symbolic eof bit, which
/// soundly over-approximates `simfd.eof()` (the solver may pick `eof=true` to
/// justify a zero-length read) — mirroring Python `fgetc`'s
/// `If(real_length == 0, -1, byte)` and the fgets short-read path above
/// (angr-qx81x).
fn read_stdin_char(state: &mut RustSimState, tag: &'static str) -> RustBV {
    let read_id = symbol_counter(tag);
    let name = format!("stdin_{tag}_{read_id}");
    let short_reads = state.has_option("SHORT_READS");
    // Mints the leaf, binds it to a harness-seeded fd-0 byte if there is one,
    // and records it for posix.dumps(0) export when there is not.
    #[allow(
        clippy::expect_used,
        reason = "`mint_stdin_bytes` returns exactly one `RustBV` per requested name (its own doc contract, and it builds the vec by mapping over `names`), and the call passes a single-element slice via `slice::from_ref`, so the vec always holds one element"
    )]
    let sym_byte = mint_stdin_bytes(state, std::slice::from_ref(&name))
        .pop()
        .expect("mint_stdin_bytes returns one BV per name");
    let ctx = state.solver().borrow();
    // Zero-extend to int (32-bit, matching C int type)
    let byte_ze = sym_byte.zero_extend(32, &ctx);
    if short_reads {
        let eof = RustBV::symbolic(&ctx, format!("{tag}_eof_{read_id}"), 1);
        let neg_one = RustBV::concrete((-1i64 as u64) as u128, 32);
        eof.ite(&neg_one, &byte_ze, &ctx)
    } else {
        byte_ze
    }
}

crate::declare_proc! {
    /// Native fgetc implementation.
    ///
    /// ```c
    /// int fgetc(FILE *stream);
    /// ```
    ///
    /// Returns one symbolic byte zero-extended to int size, or `If(eof, -1, byte)`
    /// under the SHORT_READS SimOption — see [`read_stdin_char`], the body shared
    /// with `getchar`. Default (SHORT_READS off) never returns the EOF sentinel
    /// from the symbolic-stdin path.
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
                "fgetc from fd={fd} (non-stdin) falls back to Python"
            )));
        }
        Ok(Some(read_stdin_char(state, "fgetc")))
    }
}

crate::declare_proc! {
    /// Native getchar implementation.
    ///
    /// ```c
    /// int getchar(void);
    /// ```
    ///
    /// Equivalent to fgetc(stdin), and shares [`read_stdin_char`] with it so the
    /// two cannot drift: returns one symbolic byte zero-extended to int, or
    /// `If(eof, -1, byte)` under the SHORT_READS SimOption. Unlike `fgetc` there
    /// is no FILE\* to resolve — `getchar` always reads fd 0.
    name = "getchar",
    struct = NativeGetchar,
    args = [],
    aliases = ["getchar_unlocked"],
    call |state| {
        Ok(Some(read_stdin_char(state, "getchar")))
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

crate::declare_proc! {
    /// Native gets implementation.
    ///
    /// ```c
    /// char *gets(char *s);
    /// ```
    ///
    /// `gets` has no size argument and always reads from stdin, so this serves
    /// the symbolic-stdin path unconditionally (fd 0), reading at most
    /// `MAX_GETS_SIZE - 1` symbolic bytes. Unlike `fgets`, Python's `gets`
    /// (`procedures/libc/gets.py`) applies the variable-length case-2 model for
    /// symbolic input *always* — it is not gated on SHORT_READS — so this native
    /// path mirrors that: [`store_symbolic_line`] adds the per-byte newline/EOF
    /// constraints and a NUL at the symbolic `real_size` offset. The return value
    /// is the buffer pointer `s` (matching Python), not `real_size`.
    name = "gets",
    struct = NativeGets,
    args = [buf: concrete],
    call |state| {
        let bits = state.arch().bits();
        let read_count = MAX_GETS_SIZE - 1;
        let read_id = symbol_counter("gets");

        // Create symbolic stdin bytes and record them for posix.dumps(0) export.
        let names: Vec<String> = (0..read_count)
            .map(|i| format!("stdin_gets_{read_id}_{i}"))
            .collect();
        let sym_bytes: Vec<RustBV> = mint_stdin_bytes(state, &names);

        store_symbolic_line(state, buf, MAX_GETS_SIZE, &sym_bytes, "gets", read_id)?;

        // gets returns the destination buffer pointer.
        Ok(Some(RustBV::concrete(buf as u128, bits)))
    }
}

#[cfg(test)]
#[path = "fgets_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod tests;
