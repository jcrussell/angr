//! Buffer for pending concrete stores accumulated during VEX block execution.
//!
//! Wraps a `Vec<(u64, Vec<u8>)>` (chronological insertion order) with a
//! per-byte address → store-index map so concrete-addr loads can fast-skip the
//! reverse linear scan when no pending store overlaps the load address.

use rustc_hash::FxHashMap;

/// Pending concrete stores buffered for a single VEX block.
///
/// Stores are appended in execution order and flushed on block boundary or
/// auto-flush threshold. Loads forward through pending stores by checking the
/// most recent store covering each load address.
#[derive(Default)]
pub(crate) struct PendingStoreBuffer {
    /// Stores in chronological insertion order: (address, data).
    stores: Vec<(u64, Vec<u8>)>,
    /// Maps each byte address covered by a pending store to the index in
    /// `stores` of the most recently pushed store covering it.
    ///
    /// This lets loads skip the linear reverse scan when no pending store
    /// touches the load address. When a store does cover the load address,
    /// the indexed entry is checked first; only if that store is smaller than
    /// the load do we fall back to a reverse scan to find an earlier
    /// fully-covering store. That earlier store's bytes are kept up to date
    /// by `push` (see below), so the fallback never returns stale data.
    byte_index: FxHashMap<u64, usize>,
}

/// Byte range of a `size`-byte load at `addr` within a store buffer of length
/// `s_len` based at `s_addr`, or `None` when that store does not fully cover
/// the load.
///
/// The distance is computed with `wrapping_sub` so neither `s_addr + s_len`
/// nor `addr + size` is ever formed: a store may sit at the top of the guest
/// address space and wrap, which is exactly how `push` indexes its bytes. A
/// load below the store's base wraps to a huge offset and so fails the bounds
/// check, which is what the old `s_addr <= addr` guard did.
fn covering_range(
    s_addr: u64,
    s_len: usize,
    addr: u64,
    size: usize,
) -> Option<std::ops::Range<usize>> {
    let off = usize::try_from(addr.wrapping_sub(s_addr)).ok()?;
    let end = off.checked_add(size)?;
    (end <= s_len).then_some(off..end)
}

impl PendingStoreBuffer {
    pub(crate) fn with_capacity(cap: usize) -> Self {
        Self {
            stores: Vec::with_capacity(cap),
            byte_index: FxHashMap::with_capacity_and_hasher(cap * 4, Default::default()),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.stores.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.stores.is_empty()
    }

    pub(crate) fn push(&mut self, addr: u64, data: Vec<u8>) {
        let idx = self.stores.len();
        let len = data.len() as u64;
        let new_end = addr.saturating_add(len);

        // Patch any earlier store whose byte range overlaps this new store's
        // range so the earlier store's buffer reflects the newest bytes for
        // the overlapping region. This is what lets try_load's reverse-scan
        // fallback (used when the byte_index-indexed store is smaller than
        // the requested load) return an up-to-date value for an earlier,
        // wider store instead of stale pre-overwrite bytes: the fallback
        // scans stores by address-range coverage alone, ignoring
        // byte_index, so every store's own buffer must stay internally
        // coherent regardless of whether byte_index currently points at it.
        for (s_addr, s_data) in self.stores.iter_mut() {
            let s_end = s_addr.saturating_add(s_data.len() as u64);
            let overlap_start = addr.max(*s_addr);
            let overlap_end = new_end.min(s_end);
            // Neither end can outrun its buffer: both came from a
            // `saturating_add`, which only ever *under*-estimates a range that
            // crosses the top of the address space, so `end - base` stays
            // `<= buffer.len()`.
            if overlap_start < overlap_end {
                let s_off = (overlap_start - *s_addr) as usize; // overflow-ok: overlap_start = max(addr, s_addr)
                let new_off = (overlap_start - addr) as usize; // overflow-ok: overlap_start = max(addr, s_addr)
                let overlap_len = (overlap_end - overlap_start) as usize; // overflow-ok: guarded by the `if`
                let s_range = s_off..s_off + overlap_len; // overflow-ok: overlap_end <= saturating s_end
                let new_range = new_off..new_off + overlap_len; // overflow-ok: overlap_end <= saturating new_end
                s_data[s_range].copy_from_slice(&data[new_range]);
            }
        }

        for offset in 0..len {
            // Guest addresses wrap at the top of the address space; `try_load`
            // recovers the byte offset with the matching `wrapping_sub`.
            self.byte_index.insert(addr.wrapping_add(offset), idx);
        }
        self.stores.push((addr, data));
    }

    pub(crate) fn iter(&self) -> std::slice::Iter<'_, (u64, Vec<u8>)> {
        self.stores.iter()
    }

    pub(crate) fn as_slice(&self) -> &[(u64, Vec<u8>)] {
        &self.stores
    }

    pub(crate) fn drain(&mut self) -> std::vec::Drain<'_, (u64, Vec<u8>)> {
        self.byte_index.clear();
        self.stores.drain(..)
    }

    #[cfg(test)]
    pub(crate) fn clear(&mut self) {
        self.stores.clear();
        self.byte_index.clear();
    }

    /// Find bytes for a load at `[addr, addr+size)` from a single pending store
    /// that fully covers the range. Returns None if no such store exists, in
    /// which case the caller should fall through to other lookup paths.
    ///
    /// If a smaller store at `addr` was pushed after a larger covering store,
    /// we fall back to a reverse scan to find the earlier, still
    /// fully-covering store. That earlier store's buffer is kept patched
    /// up-to-date on every `push` (see `push`'s doc comment), so this
    /// correctly returns last-write-wins bytes rather than stale data from
    /// before the later, narrower store overwrote part of its range.
    pub(crate) fn try_load(&self, addr: u64, size: usize) -> Option<&[u8]> {
        let &idx = self.byte_index.get(&addr)?;
        let (store_addr, store_data) = &self.stores[idx];
        // store_addr <= addr is guaranteed: byte_index[addr] = idx means
        // store idx covers the byte at addr.
        if let Some(range) = covering_range(*store_addr, store_data.len(), addr, size) {
            return Some(&store_data[range]);
        }
        // The most recent store covering addr is smaller than the load.
        // Fall back to reverse scan to find an earlier fully-covering store.
        for (s_addr, s_data) in self.stores.iter().rev() {
            if let Some(range) = covering_range(*s_addr, s_data.len(), addr, size) {
                return Some(&s_data[range]);
            }
        }
        None
    }

    /// Assemble a load at `[addr, addr+size)` from *several* pending stores
    /// when no single store covers the whole range.
    ///
    /// `try_load` can only ever hand back a slice of one store's buffer, so a
    /// byte-wise write loop (store 1 byte at `addr`, then 1 byte at
    /// `addr + 1`) followed by a wider read-back missed the buffer entirely
    /// and fell through to stale pre-write memory (angr-6cp06.69). Callers
    /// should try `try_load` first — it is allocation-free — and only reach
    /// for this on a miss.
    ///
    /// Every byte of the range must be buffered; a partially covered load
    /// still returns `None`, because merging buffered bytes with the caller's
    /// lower layers (flushed stores, concrete cache, Python callback) needs
    /// machinery that does not live here.
    ///
    /// `byte_index` records, per byte, the index of the most recent store
    /// covering it, so gathering byte-by-byte is last-write-wins by
    /// construction — unlike `try_load`'s reverse scan, it does not lean on
    /// `push`'s overlap patching.
    pub(crate) fn try_load_assembled(&self, addr: u64, size: usize) -> Option<Vec<u8>> {
        // Fast-skip the common miss (no pending store at the load address)
        // before allocating the output buffer.
        if !self.byte_index.contains_key(&addr) {
            return None;
        }
        let mut out = Vec::with_capacity(size);
        for offset in 0..size as u64 {
            // Guest addresses wrap at the top of the address space, matching
            // how `push` indexes a store's bytes.
            let byte_addr = addr.wrapping_add(offset);
            let &idx = self.byte_index.get(&byte_addr)?;
            let (store_addr, store_data) = &self.stores[idx];
            let off = usize::try_from(byte_addr.wrapping_sub(*store_addr)).ok()?;
            out.push(*store_data.get(off)?);
        }
        Some(out)
    }

    /// Find bytes from a pending store whose base address exactly equals
    /// `addr` and whose length is at least `size`. Used by callers that need
    /// the value originally written at `addr` (not arbitrary bytes from a
    /// larger covering store).
    pub(crate) fn try_load_exact(&self, addr: u64, size: usize) -> Option<&[u8]> {
        let &idx = self.byte_index.get(&addr)?;
        let (store_addr, store_data) = &self.stores[idx];
        if *store_addr == addr && store_data.len() >= size {
            return Some(&store_data[..size]);
        }
        for (s_addr, s_data) in self.stores.iter().rev() {
            if *s_addr == addr && s_data.len() >= size {
                return Some(&s_data[..size]);
            }
        }
        None
    }
}

test_submod!("pending_store_tests.rs" => tests);
