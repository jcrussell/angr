//! Concrete-memory region cache for fast reads of read-only binary
//! sections (`.text`, `.rodata`) without going through Python callbacks.
use super::*;

/// A concrete memory region cached locally in Rust.
/// Uses `Arc<Vec<u8>>` for O(1) cloning — binary data is shared, not copied.
#[derive(Clone)]
pub struct ConcreteMemoryRegion {
    /// Base address of the region.
    pub base: u64,
    /// Size of the region in bytes.
    pub size: u64,
    /// The concrete data (shared via Arc to avoid copying per step).
    pub data: Arc<Vec<u8>>,
}

impl ConcreteMemoryRegion {
    /// Check if this region contains the given address range.
    #[inline]
    pub fn contains(&self, addr: u64, size: u64) -> bool {
        addr >= self.base && addr + size <= self.base + self.size
    }

    /// Read bytes from this region. Returns None if out of bounds.
    #[inline]
    pub fn read(&self, addr: u64, size: usize) -> Option<&[u8]> {
        if addr < self.base {
            return None;
        }
        let offset = (addr - self.base) as usize;
        if offset + size > self.data.len() {
            return None;
        }
        Some(&self.data[offset..offset + size])
    }
}

impl<'a> VEXInterpreter<'a> {
    /// Add a concrete memory region for fast local access.
    ///
    /// This allows the interpreter to read from binary sections (e.g., .text, .rodata)
    /// without going through Python callbacks, significantly improving performance.
    pub fn add_concrete_memory(&mut self, base: u64, data: Vec<u8>) {
        let size = data.len() as u64;
        Arc::make_mut(&mut self.concrete_memory).push(ConcreteMemoryRegion {
            base,
            size,
            data: Arc::new(data),
        });
        self.concrete_memory_sorted = false;
    }

    /// Add a concrete memory region using pre-shared Arc data (O(1) clone).
    pub fn add_concrete_memory_shared(&mut self, base: u64, data: Arc<Vec<u8>>) {
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

    /// Clear all concrete memory regions.
    pub fn clear_concrete_memory(&mut self) {
        Arc::make_mut(&mut self.concrete_memory).clear();
        self.concrete_memory_sorted = false;
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

    /// Check if an address is within loaded binary (concrete memory) regions.
    /// Used to distinguish internal function calls from external/library calls.
    pub fn is_in_binary(&self, addr: u64) -> bool {
        self.concrete_memory
            .iter()
            .any(|region| addr >= region.base && addr < region.base + region.size)
    }
}
