//! Self-modifying-code / code-page dirtying and cached-IRSB invalidation.
use super::*;

impl<'a> VEXInterpreter<'a> {
    /// Whether the page containing `addr` has been overwritten via a store.
    /// Native-lift callers must avoid this page since they read from the
    /// immutable `concrete_memory` buffer that does not see the new bytes.
    pub fn is_code_page_dirtied(&self, addr: u64) -> bool {
        self.dirtied_code_pages.contains(&(addr >> 12))
    }

    /// Whether any page intersecting `[addr, addr + len)` has been written.
    pub fn is_code_range_dirtied(&self, addr: u64, len: u64) -> bool {
        if self.dirtied_code_pages.is_empty() || len == 0 {
            return false;
        }
        let first_page = addr >> 12;
        let last_page = (addr.saturating_add(len - 1)) >> 12;
        for page_num in first_page..=last_page {
            if self.dirtied_code_pages.contains(&page_num) {
                return true;
            }
        }
        false
    }

    /// Mark code pages overlapping `[addr, addr + size)` as dirtied and
    /// invalidate cached IRSBs whose byte ranges cover any of those bytes.
    /// Caller must verify the address is in a binary region first.
    pub fn invalidate_code_at(&mut self, addr: u64, size: usize) {
        if size == 0 {
            return;
        }
        let end = addr.saturating_add(size as u64 - 1);
        let first_page = addr >> 12;
        let last_page = end >> 12;
        for page_num in first_page..=last_page {
            self.dirtied_code_pages.insert(page_num);
        }
        // Find cached IRSBs whose [start, start + irsb.size()) overlaps the
        // write. LruCache::iter is O(N) but N <= BLOCK_CACHE_CAPACITY and this
        // fires only on rare in-binary stores, so the cost is bounded.
        let mut to_remove: Vec<u64> = Vec::new();
        for (block_addr, irsb) in self.block_cache.iter() {
            let block_end = block_addr.saturating_add(irsb.size() as u64);
            if *block_addr <= end && block_end > addr {
                to_remove.push(*block_addr);
            }
        }
        for block_addr in to_remove {
            self.block_cache.pop(&block_addr);
        }
    }
}
