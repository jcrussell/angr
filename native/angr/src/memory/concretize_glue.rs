//! Glue helpers between the address concretizer and the page table.
//!
//! `load_symbolic_unified` calls these to prep candidate addresses before
//! handing them off to the ITE-tree builders. They live here, separate from
//! the load/store logic in `memory/mod.rs`, so the relationship between
//! "concretizer told us X addresses" and "page table is ready for X" stays
//! easy to find.

use super::SymbolicMemory;

impl SymbolicMemory {
    /// Filter the candidate address list down to the ones whose pages are
    /// already mapped, so the ITE-tree builder can load each one safely.
    ///
    /// Unmapped addresses are dropped — the caller should detect an empty or
    /// shorter return list and fall back to Python (which has the actual
    /// backer data) instead of speculatively materializing zero pages here.
    ///
    /// # Arguments
    /// * `addrs` - List of candidate addresses
    /// * `size` - Size of the access in bytes (currently unused)
    ///
    /// # Returns
    /// List of addresses whose pages are mapped.
    pub fn prepare_addresses_for_ite(&self, addrs: &[u64], _size: u32) -> Vec<u64> {
        // Keep only addresses whose page is already mapped. Unmapped pages are
        // intentionally dropped rather than auto-mapped as zeros: Python may
        // have actual backer data (file contents, initialized data) for them,
        // so the caller falls back to the Python callback. Speculatively
        // materializing zero pages here would cause state divergence.
        addrs
            .iter()
            .copied()
            .filter(|&addr| self.pages.contains_key(&(addr >> 12)))
            .collect()
    }

    /// Prepare a strided memory region (no-op - kept for API compatibility).
    ///
    /// # Note
    /// This function previously auto-mapped zero pages for unmapped addresses,
    /// but that caused state divergence with Python's actual backer data.
    /// Now it does nothing - unmapped pages will trigger Python fallback.
    ///
    /// # Arguments
    /// * `base` - Base address of the strided pattern
    /// * `stride` - Stride between consecutive addresses
    /// * `count` - Number of addresses in the pattern
    /// * `size` - Size of each access in bytes
    pub(super) fn prepare_strided_region(&self, _base: u64, _stride: u64, _count: u64, _size: u32) {
        // No longer auto-maps zero pages.
        // The interpreter will fall back to Python callback which can provide
        // actual backer data instead of speculative zeros.
        //
        // NOTE: Strided loads/stores may fail and trigger Python fallback.
        // This is intentional - Python has the correct memory state.
    }
}
