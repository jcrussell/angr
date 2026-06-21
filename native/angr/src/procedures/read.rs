//! Native read implementation.
//!
//! Handles `read(fd, buf, count)` for:
//!   - fd=0 (stdin) by storing fresh symbolic bytes and recording the symbol
//!     names so the Python state export can splice them into `posix.dumps(0)`.
//!     Returns the full concrete `count` by default; under the SHORT_READS
//!     SimOption it instead returns a symbolic size in `[0, count]` (the lone
//!     fork source), mirroring Python's SimPacket short-read path.
//!   - any other fd open in the Rust `FileSystem` that has remaining concrete
//!     content (pos < content_len) — bytes are served from the FS buffer,
//!     advancing the position.
//!
//! Falls back to Python for:
//!   - symbolic fd/buf/count
//!   - count beyond `MAX_READ_SIZE`
//!   - fds not open in the Rust `FileSystem` (which includes any fd Python
//!     created via its symbolic-file plumbing; see fd-table invariant below).
//!   - non-stdin open fds with empty / fully-consumed content (Python's
//!     symbolic-file model owns those reads).
//!
//! ## Fd-table sync invariant (angr-8j16)
//!
//! Rust's `FileSystem` uses a monotonic fd counter, Python's `state.posix.fd`
//! uses lowest-free. They are NOT kept in sync. The native handlers cover
//! only fds in Rust's table; everything else falls back so the Python
//! symbolic-file model can take over. See bd memory
//! `invariant-rust-filesystem-no-python-sync`.

use super::strings::write_concrete_bytes;
use super::{ProcedureError, symbol_counter};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const MAX_READ_SIZE: u64 = 4096;

crate::declare_proc! {
    /// read: serve stdin (fd=0) symbolic bytes or concrete FS content.
    ///
    /// ```c
    /// ssize_t read(int fd, void *buf, size_t count);
    /// ```
    name = "read",
    struct = NativeRead,
    args = [fd: concrete, buf: concrete, count: concrete],
    call |state| {
        if count > MAX_READ_SIZE {
            return Err(ProcedureError::Other(format!(
                "read count {} exceeds limit",
                count
            )));
        }

        if count == 0 {
            let bits = state.arch().bits();
            return Ok(Some(RustBV::concrete(0, bits)));
        }

        if fd == 0 {
            return read_stdin_symbolic(state, buf, count);
        }

        let fd_u32 = fd as u32;
        let (open, content_len) = match state.file_system_ref().fd_info(fd_u32) {
            Some((_, _, _, len, is_open)) => (is_open, len),
            None => (false, 0),
        };
        if !open {
            return Err(ProcedureError::Other(format!(
                "read from fd={} (not open in Rust FileSystem) falls back to Python",
                fd
            )));
        }
        // Empty / no-content fds defer to Python so the symbolic-file model
        // (cle simfs + SimFile) can produce symbolic bytes.
        if content_len == 0 {
            return Err(ProcedureError::Other(format!(
                "read from fd={} has no concrete content; falling back to Python",
                fd
            )));
        }

        let bytes = state.file_system().read(fd_u32, count as usize);
        let n = bytes.len();
        write_concrete_bytes(state, buf, &bytes)?;
        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(n as u128, bits)))
    }
}

fn read_stdin_symbolic(
    state: &mut RustSimState,
    buf: u64,
    count: u64,
) -> Result<Option<RustBV>, ProcedureError> {
    let read_id = symbol_counter("read");

    let names: Vec<String> = (0..count)
        .map(|i| format!("stdin_{}_{}", read_id, i))
        .collect();
    let sym_bytes: Vec<RustBV> = {
        let ctx = state.solver().borrow();
        names
            .iter()
            .map(|name| RustBV::symbolic(&ctx, name, 8))
            .collect()
    };

    for name in &names {
        state.record_stdin_symbol(name.clone(), 8);
    }

    for (i, sym_byte) in sym_bytes.into_iter().enumerate() {
        state.memory_store(buf.wrapping_add(i as u64), sym_byte)?;
    }

    let bits = state.arch().bits();

    // Short-read model, gated behind the SHORT_READS SimOption (angr-kf0uy).
    // Mirrors Python's storage/file.py SimPacket path: under SHORT_READS the
    // read returns a symbolic `real_size` constrained to <= count, rather than
    // the full count. This is the only fork source — the `count` symbolic bytes
    // already filled into `buf` are unchanged (Python likewise generates the
    // full packet and lets the symbolic return size tell the caller how many
    // bytes are valid; content beyond real_size is undefined). Unlike fgets
    // (angr-efvao) there are no newline/NUL/buffer semantics to model, so the
    // symbolic return is the entire change. Without SHORT_READS the default
    // below stays byte-identical and non-forking, so the perf gate is
    // unaffected; enabling it opts into the same state multiplication Python
    // already incurs.
    if state.has_option("SHORT_READS") {
        let real_size = {
            let ctx = state.solver().borrow();
            let real_size = RustBV::symbolic(&ctx, format!("read_realsize_{}", read_id), bits);
            // 0 <= real_size <= count (lower bound implicit for unsigned).
            let bound = real_size.ule(&RustBV::concrete(count as u128, bits), &ctx);
            (real_size, bound)
        };
        let (real_size, bound) = real_size;
        state.add_constraint(bound);
        return Ok(Some(real_size));
    }

    Ok(Some(RustBV::concrete(count as u128, bits)))
}

#[cfg(test)]
#[path = "read_tests.rs"]
mod read_tests;
