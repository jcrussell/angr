//! Glue helpers between the address concretizer and the page table.
//!
//! `load_symbolic_unified` calls these to check candidate addresses before
//! handing them off to the ITE-tree builders. They live here, separate from
//! the load/store logic in `memory/mod.rs`, so the relationship between
//! "concretizer told us X addresses" and "page table is ready for X" stays
//! easy to find.

use super::page::PageIndex;
use super::{MemoryError, SymbolicMemory};
use crate::symbolic::record_mem_lazy_page_fault;

impl SymbolicMemory {
    /// Check every page of every candidate access `[cand, cand + size)`
    /// before the ITE builder loads them.
    ///
    /// A missing page that a lazy region covers, or that `host_has_page`
    /// says the host can serve, returns `UnmappedPageInRegion` so the
    /// interpreter fetches it and retries, the same cue the store side's
    /// `install_multi_for_candidates_safe` gives. Any other missing page is
    /// one neither side holds: its candidate stays in the set and its bytes
    /// read as a fresh filler (`load_concrete_or_unconstrained`), as in
    /// Python's default memory. No candidate is ever dropped, so the ITE and
    /// the address disjunction cover the concretizer's whole set
    /// (angr-vmi2l.177).
    pub(super) fn check_candidate_pages(
        &self,
        addrs: &[u64],
        size: u32,
        host_has_page: &dyn Fn(u64) -> bool,
    ) -> Result<(), MemoryError> {
        for &cand in addrs {
            for page in PageIndex::range_covering(cand, u64::from(size)) {
                let page_addr = page.base_addr();
                if !self.is_mapped(page_addr)
                    && (self.is_addr_in_lazy_region(page_addr) || host_has_page(page_addr))
                {
                    record_mem_lazy_page_fault();
                    return Err(MemoryError::UnmappedPageInRegion { page_addr });
                }
            }
        }
        Ok(())
    }
}
