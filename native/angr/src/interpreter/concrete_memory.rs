//! Concrete-memory region cache for fast reads of read-only binary
//! sections (`.text`, `.rodata`) without going through Python callbacks.
use super::*;

/// A concrete memory region cached locally in Rust.
/// Uses `Arc<Vec<u8>>` for O(1) cloning — binary data is shared, not copied.
#[derive(Clone)]
pub(crate) struct ConcreteMemoryRegion {
    /// Base address of the region.
    pub base: u64,
    /// Size of the region in bytes.
    pub size: u64,
    /// The concrete data (shared via Arc to avoid copying per step).
    pub data: Arc<Vec<u8>>,
}

impl ConcreteMemoryRegion {
    /// Byte offset of `addr` within this region, wrapping at the top of the
    /// address space. An `addr` below `base` (or past the region) yields a
    /// value `>= self.size`, which is what makes the single-comparison
    /// containment tests below overflow-free: `base + size` is never formed.
    #[inline]
    fn offset_of(&self, addr: u64) -> u64 {
        addr.wrapping_sub(self.base)
    }

    /// Whether `addr` falls inside this region. Equivalent to
    /// `addr >= base && addr < base + size` for a region that does not wrap,
    /// but cannot overflow for one abutting the top of the address space.
    #[inline]
    pub(crate) fn contains(&self, addr: u64) -> bool {
        self.offset_of(addr) < self.size
    }

    /// Read bytes from this region. Returns None if out of bounds.
    #[inline]
    pub(crate) fn read(&self, addr: u64, size: usize) -> Option<&[u8]> {
        // An `addr` below `base` wraps to a huge offset here, so it fails the
        // bounds check below exactly as the old explicit `addr < self.base`
        // guard did. `checked_add` because a wrapped offset plus `size` would
        // otherwise wrap back into range and index the slice out of bounds.
        let offset = usize::try_from(self.offset_of(addr)).ok()?;
        let end = offset.checked_add(size)?;
        if end > self.data.len() {
            return None;
        }
        Some(&self.data[offset..end])
    }
}

impl<'a> VEXInterpreter<'a> {
    /// Add a concrete memory region for fast local access.
    ///
    /// This allows the interpreter to read from binary sections (e.g., .text, .rodata)
    /// without going through Python callbacks, significantly improving performance.
    ///
    /// Production always has the region bytes behind an `Arc` already (the
    /// binary-region copy in `step_core`), so it calls
    /// `add_concrete_memory_shared`; only the interpreter tests hand over an
    /// owned `Vec` (angr-9ke6b.214).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn add_concrete_memory(&mut self, base: u64, data: Vec<u8>) {
        let size = data.len() as u64;
        Arc::make_mut(&mut self.concrete_memory).push(ConcreteMemoryRegion {
            base,
            size,
            data: Arc::new(data),
        });
        self.concrete_memory_sorted = false;
    }

    /// Add a concrete memory region using pre-shared Arc data (O(1) clone).
    pub(crate) fn add_concrete_memory_shared(&mut self, base: u64, data: Arc<Vec<u8>>) {
        let size = data.len() as u64;
        Arc::make_mut(&mut self.concrete_memory).push(ConcreteMemoryRegion { base, size, data });
        self.concrete_memory_sorted = false;
    }

    /// Sort concrete memory regions by base address for binary search.
    pub(super) fn sort_concrete_memory(&mut self) {
        if !self.concrete_memory_sorted && self.concrete_memory.len() > 1 {
            Arc::make_mut(&mut self.concrete_memory).sort_by_key(|r| r.base);
            self.concrete_memory_sorted = true;
        }
    }

    /// Try to read from concrete memory cache using binary search.
    /// Returns Some(data) if the address range is fully contained in a cached region.
    #[inline]
    pub(super) fn try_read_concrete_memory(&self, addr: u64, size: usize) -> Option<&[u8]> {
        if self.concrete_memory.is_empty() {
            return None;
        }

        // Use binary search if we have many regions
        if self.concrete_memory.len() > 4 && self.concrete_memory_sorted {
            // Binary search: find the region where base <= addr
            let idx = self.concrete_memory.partition_point(|r| r.base <= addr);
            if idx > 0 {
                // Check the region just before this index
                let region = &self.concrete_memory[idx - 1];
                if let Some(data) = region.read(addr, size) {
                    return Some(data);
                }
            }
            return None;
        }

        // Linear scan for small number of regions
        for region in self.concrete_memory.iter() {
            if let Some(data) = region.read(addr, size) {
                return Some(data);
            }
        }
        None
    }

    /// Read up to `max_size` bytes starting at `addr` from the load-time binary
    /// regions, truncating at the end of the containing region. Returns `None`
    /// when `addr` falls outside every region.
    ///
    /// Unlike [`Self::try_read_concrete_memory`] this tolerates a short read —
    /// the native lifter stops at the block boundary, so a prefix suffices.
    /// These are the *load-time* bytes: callers must not use them for a page a
    /// store has dirtied (see `is_code_range_dirtied`).
    #[cfg(feature = "libvex-ffi")]
    pub(super) fn read_concrete_prefix(&self, addr: u64, max_size: usize) -> Option<&[u8]> {
        for region in self.concrete_memory.iter() {
            if region.contains(addr) {
                // overflow-ok: `contains` proves the wrapping offset is
                // `< region.size`, i.e. an in-bounds index into `data`.
                let offset = region.offset_of(addr) as usize;
                let end = offset.saturating_add(max_size).min(region.data.len());
                return Some(&region.data[offset..end]);
            }
        }
        None
    }

    /// Check if an address is within loaded binary (concrete memory) regions.
    /// Used to distinguish internal function calls from external/library calls.
    pub(crate) fn is_in_binary(&self, addr: u64) -> bool {
        self.concrete_memory
            .iter()
            .any(|region| region.contains(addr))
    }
}

test_submod!("concrete_memory_tests.rs" => tests);
