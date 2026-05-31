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
    /// the load do we fall back to a reverse scan to find an earlier covering
    /// store (matches the prior reverse-scan-first-fully-covering semantics).
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
    /// Preserves the prior semantics of "most recent fully-covering store wins":
    /// if a small store at `addr` was pushed after a larger covering store, we
    /// fall back to a reverse scan to find the earlier covering store.
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
mod tests {
    use super::*;

    #[test]
    fn empty_buffer_misses() {
        let buf = PendingStoreBuffer::with_capacity(8);
        assert!(buf.try_load(0x100, 4).is_none());
    }

    #[test]
    fn single_store_full_cover() {
        let mut buf = PendingStoreBuffer::with_capacity(8);
        buf.push(0x100, vec![1, 2, 3, 4]);
        assert_eq!(buf.try_load(0x100, 4).unwrap(), &[1, 2, 3, 4]);
        assert_eq!(buf.try_load(0x101, 2).unwrap(), &[2, 3]);
    }

    #[test]
    fn miss_when_no_overlap() {
        let mut buf = PendingStoreBuffer::with_capacity(8);
        buf.push(0x200, vec![1, 2, 3, 4]);
        assert!(buf.try_load(0x100, 4).is_none());
        assert!(buf.try_load(0x205, 4).is_none()); // beyond end
    }

    #[test]
    fn most_recent_full_cover_wins() {
        let mut buf = PendingStoreBuffer::with_capacity(8);
        buf.push(0x100, vec![1, 2, 3, 4]);
        buf.push(0x100, vec![9, 8, 7, 6]);
        assert_eq!(buf.try_load(0x100, 4).unwrap(), &[9, 8, 7, 6]);
    }

    #[test]
    fn small_store_falls_back_to_earlier() {
        // Larger covering store followed by smaller store at same addr.
        // Old semantics: reverse scan finds earlier fully-covering store first.
        let mut buf = PendingStoreBuffer::with_capacity(8);
        buf.push(0x100, vec![1, 2, 3, 4]); // idx 0
        buf.push(0x100, vec![9]); // idx 1, partial
        // Load size 4 at 0x100: idx 1 doesn't fully cover; reverse scan finds idx 0.
        assert_eq!(buf.try_load(0x100, 4).unwrap(), &[1, 2, 3, 4]);
    }

    #[test]
    fn drain_clears_index() {
        let mut buf = PendingStoreBuffer::with_capacity(8);
        buf.push(0x100, vec![1, 2, 3, 4]);
        let _: Vec<_> = buf.drain().collect();
        assert!(buf.is_empty());
        assert!(buf.try_load(0x100, 4).is_none());
    }

    #[test]
    fn clear_clears_index() {
        let mut buf = PendingStoreBuffer::with_capacity(8);
        buf.push(0x100, vec![1, 2, 3, 4]);
        buf.clear();
        assert!(buf.is_empty());
        assert!(buf.try_load(0x100, 4).is_none());
    }

    #[test]
    fn try_load_exact_requires_base_match() {
        let mut buf = PendingStoreBuffer::with_capacity(8);
        buf.push(0x100, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(
            buf.try_load_exact(0x100, 8).unwrap(),
            &[1, 2, 3, 4, 5, 6, 7, 8]
        );
        // Loading at offset within the store should fail (not an exact base match).
        assert!(buf.try_load_exact(0x101, 4).is_none());
    }
}
