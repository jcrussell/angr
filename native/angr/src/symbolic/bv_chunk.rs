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
        let chunk = &data[offset..offset + chunk_size];
        let width = (chunk_size * 8) as u32;
        let mut value: u128 = 0;
        for (i, &b) in chunk.iter().enumerate() {
            value |= (b as u128) << (i * 8);
        }
        let bv = RustBV::concrete(value, width);
        store(addr + offset as u64, bv)?;
        offset += chunk_size;
    }
    Ok(())
}

/// Unpack the low `size` bytes of a `u128` into a little-endian byte vector.
///
/// `size` must be `<= MAX_CONCRETE_CHUNK`; for `i >= 16` the `val >> (i * 8)`
/// shift wraps mod 128 and would repeat earlier bytes (callers that need wider
/// reads chunk first — see [`load_concrete_bytes_chunked`]). Read-side
/// counterpart of the pack loop in [`store_concrete_bytes_chunked`].
pub fn u128_to_le_bytes(val: u128, size: usize) -> Vec<u8> {
    (0..size).map(|i| (val >> (i * 8)) as u8).collect()
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
pub fn load_concrete_bytes_chunked<E, F>(addr: u64, size: u32, mut load: F) -> Result<Vec<u8>, E>
where
    F: FnMut(u64, u32) -> Result<Vec<u8>, E>,
{
    let mut out = Vec::with_capacity(size as usize);
    let mut offset = 0u32;
    while offset < size {
        let chunk_size = (size - offset).min(MAX_CONCRETE_CHUNK as u32);
        out.extend_from_slice(&load(addr + offset as u64, chunk_size)?);
        offset += chunk_size;
    }
    Ok(out)
}

#[cfg(test)]
#[path = "bv_chunk_tests.rs"]
mod bv_chunk_tests;
