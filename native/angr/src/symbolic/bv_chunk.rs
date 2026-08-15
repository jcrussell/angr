//! Chunked conversion between concrete byte buffers and `RustBV::Concrete`.
//!
//! `RustBV::Concrete` is backed by a `u128`, so a single concrete BV round-trips
//! at most **16 bytes**. Every memory entry point that moves a `&[u8]` in or out
//! of the engine therefore has to split the range into `<= 16`-byte chunks. That
//! loop was hand-copied into five call sites across two layers before
//! angr-9ke6b.230 landed it here.
//!
//! This module lives next to [`RustBV`] rather than in `exploration/helpers.rs`
//! because the constraint it encodes belongs to `RustBV`, and because both
//! `state::RustSimState` (lower layer) and `exploration::RustExplorationManager`
//! (upper layer) need it — `state/` cannot import from `exploration/`.
//!
//! The helpers are generic over the caller's error type so this module does not
//! pull `pyo3` into the symbolic layer; the `#[pymethods]` call sites
//! instantiate `E = PyErr`.

use super::RustBV;

/// Widest concrete chunk a single `RustBV::Concrete` can round-trip, in bytes
/// (`u128` = 16 bytes). The reason every helper here loops.
pub const MAX_CONCRETE_CHUNK: usize = 16;

/// Largest byte count accepted by a memory-read entry point that materializes
/// the range into a `Vec<u8>` (or a single `RustBV`) for Python.
///
/// The `size` on `RustExplorationManager.get_state_memory` /
/// `get_state_memory_ast` / `get_pending_memory` / `pending_memory_load` and
/// `RustSimState.memory_load` is caller-supplied and crosses the PyO3 boundary
/// verbatim, so a script that derives it from guest-controlled data can ask for
/// `u32::MAX` bytes. 16 MiB is four thousand pages — orders of magnitude past
/// any real read (a buffer, a string, a stack frame, a segment) — while still
/// refusing the ~4 GiB request that would OOM or hang the host regardless of
/// whether the range is even mapped (angr-0jh0j.11, same DoS class as
/// angr-c7xno.67/.80/.81/.94).
pub const MAX_CONCRETE_LOAD_BYTES: u32 = 0x100_0000;

/// Reject a memory-read byte count above [`MAX_CONCRETE_LOAD_BYTES`], naming
/// `op` in the message.
///
/// Returns the message rather than a typed error for the same reason
/// [`check_bv_width`](super::check_bv_width) does: the callers report through
/// different channels, and this module stays free of `pyo3`. Every Python-facing
/// entry point that ends in [`load_concrete_bytes_chunked`] — or in a single
/// `memory_load` of the whole range — calls this first.
pub fn check_concrete_load_size(op: &str, size: u32) -> Result<(), String> {
    if size > MAX_CONCRETE_LOAD_BYTES {
        return Err(format!(
            "{op}: size {size} exceeds the maximum supported memory-read size \
             {MAX_CONCRETE_LOAD_BYTES}"
        ));
    }
    Ok(())
}

/// Pack a concrete byte slice into `<= 16`-byte `RustBV::concrete` chunks and
/// store each via the caller-supplied sink.
///
/// angr-5aj8: packing more than `MAX_CONCRETE_CHUNK` bytes into a single value
/// shift-overflows for byte indices `>= 16`, and the downstream `store_concrete`
/// page-fill loop then emits a 16-byte-cycle pattern across the entire
/// `data.len()` range, corrupting memory wholesale. This is the canonical safe
/// loop; the sink closure abstracts the only divergence between call sites
/// (which memory API to write through, and how to map its error).
///
/// An empty `data` stores nothing and never invokes the sink.
pub fn store_concrete_bytes_chunked<E, F>(addr: u64, data: &[u8], mut store: F) -> Result<(), E>
where
    F: FnMut(u64, RustBV) -> Result<(), E>,
{
    let mut offset = 0usize;
    while offset < data.len() {
        let remaining = data.len() - offset;
        let chunk_size = remaining.min(MAX_CONCRETE_CHUNK);
        // overflow-ok: `offset < data.len()` and `chunk_size <= data.len() -
        // offset`, so the end index is at most `data.len()` (<= `isize::MAX`).
        let chunk = &data[offset..offset + chunk_size];
        let width = (chunk_size * 8) as u32;
        let mut value: u128 = 0;
        for (i, &b) in chunk.iter().enumerate() {
            value |= (b as u128) << (i * 8);
        }
        let bv = RustBV::concrete(value, width);
        // Guest address arithmetic wraps mod 2^64 (a range starting near the top
        // of the address space continues at 0), per
        // `invariant-rust-concrete-arith-must-wrap`. A bare `+` here panics
        // under `overflow-checks` — and `panic = "abort"` makes that SIGABRT the
        // whole Python process — while release wraps anyway (angr-xloth.2).
        store(addr.wrapping_add(offset as u64), bv)?;
        offset += chunk_size;
    }
    Ok(())
}

/// Little-endian byte `i` of a `u128` payload, and `0` once `i` reaches
/// [`MAX_CONCRETE_CHUNK`].
///
/// The bare `(val >> (i * 8)) as u8` this replaces is only correct below the
/// payload: at `i >= 16` the shift amount reaches 128, which panics under
/// `overflow-checks` (the `release-checked` profile CI runs `cargo test` with)
/// and otherwise wraps mod 128, repeating the low bytes in a 16-byte cycle —
/// the memory-corruption shape of angr-5aj8 and the register shape of
/// angr-0jh0j.1/.2. Zero is the *correct* answer there, not a fallback: a
/// `Concrete` wider than 128 bits stores only these low bits and the ones above
/// them are implicitly zero (see `value_ops::bits_beyond_storage`).
pub fn u128_le_byte(val: u128, i: usize) -> u8 {
    if i >= MAX_CONCRETE_CHUNK {
        return 0;
    }
    (val >> (i * 8)) as u8
}

/// Unpack the low `size` bytes of a `u128` into a little-endian byte vector.
///
/// Callers that need more than `MAX_CONCRETE_CHUNK` bytes of *data* must chunk
/// first (see [`load_concrete_bytes_chunked`]) — a single `u128` has no more to
/// give, and per [`u128_le_byte`] every further byte here is zero.
/// Read-side counterpart of the pack loop in [`store_concrete_bytes_chunked`].
pub fn u128_to_le_bytes(val: u128, size: usize) -> Vec<u8> {
    (0..size).map(|i| u128_le_byte(val, i)).collect()
}

/// Read `size` bytes at `addr` in `<= 16`-byte chunks via the caller-supplied
/// source, concatenating the results.
///
/// Read-side mirror of [`store_concrete_bytes_chunked`], and the reason both
/// exist is the same `u128` width: `as_u128()` / [`u128_to_le_bytes`] only
/// round-trip 16 bytes, so a single wider load either truncates the tail, wraps
/// mod 128 into a repeating 16-byte pattern, or (for a fully concrete region)
/// fails as "symbolic" because `as_u128()` returns `None` (angr-ph300.19,
/// angr-ph300.53). The source closure abstracts the only divergence between
/// call sites: which memory API to read through and how to map its failure. It
/// is called once per chunk with `(chunk_addr, chunk_size)` and must return
/// exactly `chunk_size` bytes; the first error propagates and the partial
/// prefix is dropped.
///
/// Callers do their state / pending lookup **once** around this call and read
/// inside that single borrow. The `exploration` hand-copies this replaced
/// (`_get_state_memory`, `_get_pending_memory`, `_pending_memory_load`) each
/// recursed into themselves per chunk, redoing the `with_state` / `with_pending`
/// lookup — and, for the pending pair, re-borrowing the solver — every 16 bytes
/// (angr-9ke6b.82).
///
/// `size == 0` yields an empty vector without invoking the source at all,
/// matching [`store_concrete_bytes_chunked`]'s empty-slice behavior.
///
/// The up-front reservation is capped at [`MAX_CONCRETE_LOAD_BYTES`] so an
/// unvalidated `size` cannot turn this into an immediate multi-gigabyte
/// allocation before a single chunk has been proven readable; past that cap the
/// vector grows as chunks actually arrive. That is defence in depth, not the
/// guard — Python-facing callers refuse such a `size` outright via
/// [`check_concrete_load_size`] (angr-0jh0j.11).
pub fn load_concrete_bytes_chunked<E, F>(addr: u64, size: u32, mut load: F) -> Result<Vec<u8>, E>
where
    F: FnMut(u64, u32) -> Result<Vec<u8>, E>,
{
    let mut out = Vec::with_capacity(size.min(MAX_CONCRETE_LOAD_BYTES) as usize);
    let mut offset = 0u32;
    while offset < size {
        // overflow-ok: `offset < size` is the loop condition.
        let chunk_size = (size - offset).min(MAX_CONCRETE_CHUNK as u32);
        // Wraps mod 2^64 for the same reason the store side does — see
        // `store_concrete_bytes_chunked`.
        out.extend_from_slice(&load(addr.wrapping_add(offset as u64), chunk_size)?);
        offset += chunk_size;
    }
    Ok(out)
}

test_submod!("bv_chunk_tests.rs" => bv_chunk_tests);
