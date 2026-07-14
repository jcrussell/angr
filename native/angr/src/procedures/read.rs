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
//!   - fds with bounded symbolic content (`content_sym`, angr-0xyq2 Phase 2):
//!     the registered per-byte BVs are stored to the buffer natively via
//!     `FileSystem::read_sym`, advancing the position; EOF returns 0. Counts
//!     are clamped to `MAX_SYMFILE_SERVE_SIZE` (= the export cap, so any
//!     registered file serves in one call, matching Python `SimFile.read`),
//!     never bounced — a Python fallback would split the position cursor,
//!     since natively-minted fds are not mirrored into Python (angr-8j16).
//!   - open fds with neither concrete nor symbolic content (angr-gorvf.15):
//!     fresh symbolic bytes are minted, mirroring Python's fresh-`SimFile`
//!     model for an unknown/empty file. See `read_file_symbolic`.
//!
//! Falls back to Python for:
//!   - symbolic fd/buf/count
//!   - count beyond `MAX_READ_SIZE` (stdin and concrete-content fds only)
//!   - fds not open in the Rust `FileSystem` (which includes any fd Python
//!     created via its symbolic-file plumbing; see fd-table invariant below).
//!
//! ## Fd-table sync invariant (angr-8j16)
//!
//! Rust's `FileSystem` uses a monotonic fd counter, Python's `state.posix.fd`
//! uses lowest-free. They are NOT kept in sync. The native handlers cover
//! only fds in Rust's table; everything else falls back so the Python
//! symbolic-file model can take over. See bd memory
//! `invariant-rust-filesystem-no-python-sync`.

use super::stdin_common::mint_stdin_bytes;
use super::strings::{write_bv_bytes, write_concrete_bytes};
use super::{ProcedureError, symbol_counter};
use crate::state::MAX_SYMFILE_SERVE_SIZE;
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
        if count == 0 {
            let bits = state.arch().bits();
            return Ok(Some(RustBV::concrete(0, bits)));
        }

        if fd == 0 {
            if count > MAX_READ_SIZE {
                return Err(ProcedureError::Other(format!(
                    "read count {count} exceeds limit"
                )));
            }
            return read_stdin_symbolic(state, buf, count);
        }

        let fd_u32 = fd as u32;
        let (open, content_len) = match state.file_system_ref().fd_info(fd_u32) {
            Some((_, _, _, len, is_open)) => (is_open, len),
            None => (false, 0),
        };
        if !open {
            return Err(ProcedureError::Other(format!(
                "read from fd={fd} (not open in Rust FileSystem) falls back to Python"
            )));
        }
        // Bounded symbolic file content (angr-0xyq2 Phase 2): serve the
        // per-byte BVs registered for this fd's path natively, advancing the
        // position. An empty vec means EOF — return 0, matching the concrete
        // EOF shape below and Python SimFile's max(0, min(count, size - pos)).
        // Clamped to MAX_SYMFILE_SERVE_SIZE (= the export cap; whole file in
        // one call, matching Python) rather than bounced — see module docs.
        let clamped = count.min(MAX_SYMFILE_SERVE_SIZE) as usize;
        if let Some(sym_bytes) = state.file_system().read_sym(fd_u32, clamped) {
            let n = sym_bytes.len();
            write_bv_bytes(state, buf, sym_bytes)?;
            if n > 0 {
                crate::symbolic::record_symfile_read_native();
            }
            let bits = state.arch().bits();
            return Ok(Some(RustBV::concrete(n as u128, bits)));
        }
        if count > MAX_READ_SIZE {
            return Err(ProcedureError::Other(format!(
                "read count {count} exceeds limit"
            )));
        }
        // A write-demoted file (angr-0xyq2 Phase 2) is Python-owned: the
        // write bounced, so Python's `SimFile` holds bytes this FileSystem
        // never saw. Minting fresh symbolic bytes here would discard them —
        // keep bouncing (angr-8kk32).
        if state.file_system_ref().is_demoted_fd(fd_u32) {
            return Err(ProcedureError::Other(format!(
                "read from fd={fd} on a write-demoted file falls back to Python"
            )));
        }
        // No concrete bytes and no bounded symbolic content: the fd is backed
        // by a file the native FS has no content for. Mint fresh symbolic
        // bytes rather than bouncing (angr-gorvf.15) — see
        // `read_file_symbolic`.
        if content_len == 0 {
            return read_file_symbolic(state, fd_u32, buf, count);
        }

        let bytes = state.file_system().read(fd_u32, count as usize);
        let n = bytes.len();
        write_concrete_bytes(state, buf, &bytes)?;
        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(n as u128, bits)))
    }
}

/// Serve a read from an fd whose backing file has no content in the native
/// `FileSystem` (no concrete bytes, no bounded `content_sym`): mint `count`
/// fresh symbolic bytes named `file_<fd>_<id>_<i>` into `buf`, advance the
/// fd position, and return `count`.
///
/// This is Python's model for an unknown/empty file: `open` drops a fresh
/// `SimFile` with symbolic content into `state.fs`, and reading it mints
/// symbolic bytes. Bouncing instead (the pre-angr-gorvf.15 behavior) merely
/// moved the Python round-trip from `open` to `read`, and the natively-opened
/// fd is not mirrored into Python anyway (angr-8j16), so the bounce had no
/// Python-side fd to read from.
///
/// Reads never hit EOF here (the stream model — same as
/// `FileDescriptor::symbolic`, which this also covers). A file with a *finite*
/// symbolic size is the `content_sym` registry path, handled by the caller
/// before this is reached.
fn read_file_symbolic(
    state: &mut RustSimState,
    fd: u32,
    buf: u64,
    count: u64,
) -> Result<Option<RustBV>, ProcedureError> {
    let read_id = symbol_counter("read");
    let sym_bytes: Vec<RustBV> = {
        let ctx = state.solver().borrow();
        (0..count)
            .map(|i| RustBV::symbolic(&ctx, format!("file_{fd}_{read_id}_{i}"), 8))
            .collect()
    };
    write_bv_bytes(state, buf, sym_bytes)?;
    state.file_system().seek(fd, count as i64, 1); // SEEK_CUR
    let bits = state.arch().bits();
    Ok(Some(RustBV::concrete(count as u128, bits)))
}

fn read_stdin_symbolic(
    state: &mut RustSimState,
    buf: u64,
    count: u64,
) -> Result<Option<RustBV>, ProcedureError> {
    let read_id = symbol_counter("read");

    // Harness-seeded stdin (angr-mb09c): mint the fresh `stdin_N_i` leaves and
    // bind them to any bytes the harness seeded onto fd 0. See
    // `stdin_common::mint_stdin_bytes` for why the seed is bound by constraint
    // rather than stored into the buffer.
    let names: Vec<String> = (0..count).map(|i| format!("stdin_{read_id}_{i}")).collect();
    let sym_bytes = mint_stdin_bytes(state, &names);

    write_bv_bytes(state, buf, sym_bytes)?;

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
            let real_size = RustBV::symbolic(&ctx, format!("read_realsize_{read_id}"), bits);
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
