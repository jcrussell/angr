//! Native puts implementation.
//!
//! puts writes a string to stdout followed by a newline.
//! Reads the string from memory and appends it (plus newline) to the
//! state's stdout buffer for predicate evaluation.
//!
//! # Behavior
//!
//! - Reads the string at arg\[0\] byte-by-byte until NUL or MAX_PUTS_LEN
//! - Appends string + newline to state.stdout_buffer
//! - Returns length + 1 (for the appended newline)
//! - Falls back to Python if address is symbolic

use crate::symbolic::RustBV;

const MAX_PUTS_LEN: usize = 4096;

crate::declare_proc! {
    /// Native puts implementation.
    ///
    /// ```c
    /// int puts(const char *s);
    /// ```
    ///
    /// Returns a non-negative value on success (we return strlen(s) + 1).
    name = "puts",
    struct = NativePuts,
    args = [s: concrete],
    call |state| {
        // Read the concrete prefix byte-by-byte; a symbolic byte, failed load,
        // or the cap quietly stops the scan (puts prints what it has).
        let buf = crate::procedures::strings::scan_concrete_lossy(state, s, MAX_PUTS_LEN);

        // Append string + newline to stdout buffer. A refusal (fd 1 dup2'd
        // onto a bounded-symbolic-content fd, now demoted) bounces to
        // Python — see FileSystem::write. The newline write cannot be
        // refused once the first write succeeded (the fd is demoted, not
        // re-attached, between them).
        if !(state.write_stdout(&buf) && state.write_stdout(b"\n")) {
            return Err(crate::procedures::ProcedureError::Other(
                "puts to stdout with symbolic content falls back to Python (demoted)".to_string(),
            ));
        }

        let len = buf.len() as u128 + 1; // +1 for newline
        Ok(Some(RustBV::concrete(len, 32)))
    }
}

crate::declare_proc! {
    /// Native putchar implementation.
    ///
    /// ```c
    /// int putchar(int c);
    /// ```
    ///
    /// Writes one byte to stdout. Returns the character written (as int).
    name = "putchar",
    struct = NativePutchar,
    args = [c: concrete],
    call |state| {
        let byte = (c & 0xFF) as u8;
        // Refusal contract — see NativePuts.
        if !state.write_stdout(&[byte]) {
            return Err(crate::procedures::ProcedureError::Other(
                "putchar to stdout with symbolic content falls back to Python (demoted)"
                    .to_string(),
            ));
        }
        Ok(Some(RustBV::concrete(byte as u128, 32)))
    }
}

crate::declare_proc! {
    /// Native fputc implementation.
    ///
    /// ```c
    /// int fputc(int c, FILE *stream);
    /// ```
    ///
    /// Resolves `stream->_fileno` and appends the low byte to that fd's
    /// buffer via `write_fd`, matching Python `fputc` (which writes `c[7:0]`
    /// to the resolved SimFileDescriptor). Returns `c & 0xFF` on success and
    /// -1 on a closed/negative fd. A symbolic FILE* (concrete extraction
    /// fails) falls back to Python. Mirrors NativeFputs / NativeFwrite, which
    /// also route any non-negative fd through `write_fd` rather than assuming
    /// stdout.
    name = "fputc",
    struct = NativeFputc,
    args = [c: concrete, stream: concrete],
    aliases = ["fputc_unlocked"],
    call |state| {
        let byte = (c & 0xFF) as u8;
        let fd = match crate::procedures::stdio::read_fileno_for_stream(state, stream) {
            Ok(fd) => fd,
            Err(e) => {
                // Unresolvable fd on a write path: any bounded symbolic
                // file could be the target (angr-0xyq2 A4; O(1) when none
                // attached).
                state.file_system().demote_all_symbolic_content();
                return Err(e);
            }
        };
        if fd < 0 {
            return Ok(Some(RustBV::concrete((-1i64 as u64) as u128, 32)));
        }
        // A refusal means the fd carried bounded symbolic content (now
        // demoted) — bounce to Python, which owns the file from here
        // (angr-0xyq2 Phase 2 choke point; see FileSystem::write).
        if !state.write_fd(fd as u32, &[byte]) {
            return Err(crate::procedures::ProcedureError::Other(format!(
                "fputc to fd={fd} with symbolic content falls back to Python (demoted)"
            )));
        }
        Ok(Some(RustBV::concrete(byte as u128, 32)))
    }
}

crate::declare_proc! {
    /// Native putc implementation (alias for fputc).
    ///
    /// `putc` is a macro alias for `fputc` in glibc — same fd resolution.
    name = "putc",
    struct = NativePutc,
    args = [c: concrete, stream: concrete],
    aliases = ["putc_unlocked"],
    call |state| {
        let byte = (c & 0xFF) as u8;
        let fd = match crate::procedures::stdio::read_fileno_for_stream(state, stream) {
            Ok(fd) => fd,
            Err(e) => {
                // See NativeFputc.
                state.file_system().demote_all_symbolic_content();
                return Err(e);
            }
        };
        if fd < 0 {
            return Ok(Some(RustBV::concrete((-1i64 as u64) as u128, 32)));
        }
        // Refusal contract — see NativeFputc.
        if !state.write_fd(fd as u32, &[byte]) {
            return Err(crate::procedures::ProcedureError::Other(format!(
                "putc to fd={fd} with symbolic content falls back to Python (demoted)"
            )));
        }
        Ok(Some(RustBV::concrete(byte as u128, 32)))
    }
}

#[cfg(test)]
#[path = "puts_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod tests;
