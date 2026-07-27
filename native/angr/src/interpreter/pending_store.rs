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
            if overlap_start < overlap_end {
                let s_off = (overlap_start - *s_addr) as usize;
                let new_off = (overlap_start - addr) as usize;
                let overlap_len = (overlap_end - overlap_start) as usize;
                s_data[s_off..s_off + overlap_len]
                    .copy_from_slice(&data[new_off..new_off + overlap_len]);
            }
        }

        for offset in 0..len {
            self.byte_index.insert(addr + offset, idx);
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
        let load_end = addr.checked_add(size as u64)?;
        let store_end = store_addr.saturating_add(store_data.len() as u64);
        if load_end <= store_end {
            let offset = (addr - store_addr) as usize;
            return Some(&store_data[offset..offset + size]);
        }
        // The most recent store covering addr is smaller than the load.
        // Fall back to reverse scan to find an earlier fully-covering store.
        for (s_addr, s_data) in self.stores.iter().rev() {
            if *s_addr <= addr && addr + size as u64 <= *s_addr + s_data.len() as u64 {
                let off = (addr - *s_addr) as usize;
                return Some(&s_data[off..off + size]);
            }
        }
        None
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

#[cfg(test)]
#[path = "pending_store_tests.rs"]
mod tests;
