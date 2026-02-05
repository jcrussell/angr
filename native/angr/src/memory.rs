//! Symbolic memory system for the VEX execution engine.
//!
//! This module provides a paged memory model with:
//! - O(1) forking via copy-on-write (using Grudge's RustPage)
//! - Mixed concrete/symbolic value storage
//! - Efficient symbolic address handling

use std::collections::HashMap;
use std::sync::Arc;

use im::OrdMap;

use crate::symbolic::{RustBV, SymContext};
use crate::vex::Endness;

/// Page size in bytes (4KB).
pub const PAGE_SIZE: u64 = 4096;

/// Page mask for address calculation.
pub const PAGE_MASK: u64 = PAGE_SIZE - 1;

/// Errors from memory operations.
#[derive(Debug, Clone)]
pub enum MemoryError {
    /// Unmapped memory access.
    Unmapped { addr: u64, size: u64 },
    /// Permission violation.
    Permission {
        addr: u64,
        required: Permission,
        actual: Permission,
    },
    /// Unresolvable symbolic address.
    SymbolicAddress { description: String },
    /// Out of bounds access.
    OutOfBounds { addr: u64, size: u64 },
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryError::Unmapped { addr, size } => {
                write!(f, "unmapped memory at 0x{:x} (size {})", addr, size)
            }
            MemoryError::Permission {
                addr,
                required,
                actual,
            } => {
                write!(
                    f,
                    "permission violation at 0x{:x}: required {:?}, have {:?}",
                    addr, required, actual
                )
            }
            MemoryError::SymbolicAddress { description } => {
                write!(f, "symbolic address: {}", description)
            }
            MemoryError::OutOfBounds { addr, size } => {
                write!(f, "out of bounds access at 0x{:x} (size {})", addr, size)
            }
        }
    }
}

impl std::error::Error for MemoryError {}

/// Memory permissions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Permission {
    pub read: bool,
    pub write: bool,
    pub execute: bool,
}

impl Permission {
    pub const NONE: Permission = Permission {
        read: false,
        write: false,
        execute: false,
    };
    pub const R: Permission = Permission {
        read: true,
        write: false,
        execute: false,
    };
    pub const RW: Permission = Permission {
        read: true,
        write: true,
        execute: false,
    };
    pub const RX: Permission = Permission {
        read: true,
        write: false,
        execute: true,
    };
    pub const RWX: Permission = Permission {
        read: true,
        write: true,
        execute: true,
    };

    pub fn from_bits(bits: u8) -> Self {
        Permission {
            read: bits & 0x4 != 0,
            write: bits & 0x2 != 0,
            execute: bits & 0x1 != 0,
        }
    }

    pub fn to_bits(&self) -> u8 {
        let mut bits = 0u8;
        if self.read {
            bits |= 0x4;
        }
        if self.write {
            bits |= 0x2;
        }
        if self.execute {
            bits |= 0x1;
        }
        bits
    }

    pub fn allows(&self, other: Permission) -> bool {
        (!other.read || self.read)
            && (!other.write || self.write)
            && (!other.execute || self.execute)
    }
}

/// A memory page with copy-on-write semantics.
#[derive(Clone)]
pub struct MemoryPage {
    /// Concrete data (4KB) - uses Arc for CoW.
    data: Arc<Vec<u8>>,
    /// Permissions for this page.
    permissions: Permission,
    /// Base address of the page.
    base_addr: u64,
    /// Symbolic bytes (offset -> marker).
    /// We store these separately from the concrete data.
    symbolic: Arc<HashMap<u16, u8>>,
}

impl MemoryPage {
    /// Create a new zeroed page.
    pub fn new(base_addr: u64, permissions: Permission) -> Self {
        MemoryPage {
            data: Arc::new(vec![0u8; PAGE_SIZE as usize]),
            permissions,
            base_addr,
            symbolic: Arc::new(HashMap::new()),
        }
    }

    /// Create a page from existing data.
    pub fn from_data(base_addr: u64, data: Vec<u8>, permissions: Permission) -> Self {
        let mut page_data = vec![0u8; PAGE_SIZE as usize];
        let copy_len = data.len().min(PAGE_SIZE as usize);
        page_data[..copy_len].copy_from_slice(&data[..copy_len]);

        MemoryPage {
            data: Arc::new(page_data),
            permissions,
            base_addr,
            symbolic: Arc::new(HashMap::new()),
        }
    }

    /// Get the base address.
    pub fn base_addr(&self) -> u64 {
        self.base_addr
    }

    /// Get the permissions.
    pub fn permissions(&self) -> Permission {
        self.permissions
    }

    /// Set the permissions.
    pub fn set_permissions(&mut self, perm: Permission) {
        self.permissions = perm;
    }

    /// Check if this page has any symbolic bytes.
    pub fn has_symbolic(&self) -> bool {
        !self.symbolic.is_empty()
    }

    /// Get the symbolic byte offsets.
    pub fn symbolic_offsets(&self) -> Vec<u16> {
        self.symbolic.keys().copied().collect()
    }

    /// Load bytes from this page (concrete only).
    pub fn load_concrete(&self, offset: u16, size: u16) -> Vec<u8> {
        let start = offset as usize;
        let end = (start + size as usize).min(PAGE_SIZE as usize);
        self.data[start..end].to_vec()
    }

    /// Store bytes to this page (concrete).
    pub fn store_concrete(&mut self, offset: u16, bytes: &[u8]) {
        // Copy-on-write: if shared, make a unique copy
        let data = Arc::make_mut(&mut self.data);
        let start = offset as usize;
        let end = (start + bytes.len()).min(PAGE_SIZE as usize);
        data[start..end].copy_from_slice(&bytes[..(end - start)]);

        // Clear symbolic markers for overwritten bytes
        let sym = Arc::make_mut(&mut self.symbolic);
        for i in offset..(offset + bytes.len() as u16) {
            sym.remove(&i);
        }
    }

    /// Mark bytes as symbolic.
    pub fn mark_symbolic(&mut self, offset: u16, size: u16) {
        let sym = Arc::make_mut(&mut self.symbolic);
        for i in offset..(offset + size) {
            sym.insert(i, 1);
        }
    }

    /// Check if a byte is symbolic.
    pub fn is_symbolic(&self, offset: u16) -> bool {
        self.symbolic.contains_key(&offset)
    }

    /// Fork this page (O(1) via Arc).
    pub fn fork(&self) -> Self {
        MemoryPage {
            data: Arc::clone(&self.data),
            permissions: self.permissions,
            base_addr: self.base_addr,
            symbolic: Arc::clone(&self.symbolic),
        }
    }
}

/// Symbolic memory model.
///
/// This provides a paged memory model with:
/// - O(1) forking via CoW
/// - Mixed concrete/symbolic storage
/// - Endianness-aware loads and stores
pub struct SymbolicMemory {
    /// Pages indexed by page number (addr >> 12).
    pages: OrdMap<u64, MemoryPage>,
    /// Symbolic objects (for values that span multiple bytes).
    symbolic_objects: HashMap<u64, RustBV>,
    /// Next symbolic object ID.
    next_sym_id: u64,
    /// Default permissions for new pages.
    default_permissions: Permission,
    /// Endianness for this memory.
    endness: Endness,
}

impl SymbolicMemory {
    /// Create a new empty memory.
    pub fn new(endness: Endness) -> Self {
        SymbolicMemory {
            pages: OrdMap::new(),
            symbolic_objects: HashMap::new(),
            next_sym_id: 0,
            default_permissions: Permission::RWX,
            endness,
        }
    }

    /// Get the endianness.
    pub fn endness(&self) -> Endness {
        self.endness
    }

    /// Map a memory region.
    pub fn map(&mut self, addr: u64, size: u64, permissions: Permission) {
        let start_page = addr >> 12;
        let end_page = (addr + size + PAGE_SIZE - 1) >> 12;

        for page_num in start_page..end_page {
            let base = page_num << 12;
            if !self.pages.contains_key(&page_num) {
                self.pages.insert(page_num, MemoryPage::new(base, permissions));
            }
        }
    }

    /// Map a region and initialize with data.
    pub fn map_data(&mut self, addr: u64, data: &[u8], permissions: Permission) {
        let start_page = addr >> 12;
        let offset_in_page = addr & PAGE_MASK;

        let mut remaining = data;
        let mut current_addr = addr;

        while !remaining.is_empty() {
            let page_num = current_addr >> 12;
            let page_offset = (current_addr & PAGE_MASK) as usize;
            let bytes_in_page = (PAGE_SIZE as usize - page_offset).min(remaining.len());

            // Get or create page
            let page = self.pages.entry(page_num).or_insert_with(|| {
                MemoryPage::new(page_num << 12, permissions)
            });

            // Write data
            let mut page = page.clone();
            page.store_concrete(page_offset as u16, &remaining[..bytes_in_page]);
            self.pages.insert(page_num, page);

            remaining = &remaining[bytes_in_page..];
            current_addr += bytes_in_page as u64;
        }
    }

    /// Unmap a memory region.
    pub fn unmap(&mut self, addr: u64, size: u64) {
        let start_page = addr >> 12;
        let end_page = (addr + size + PAGE_SIZE - 1) >> 12;

        for page_num in start_page..end_page {
            self.pages.remove(&page_num);
        }
    }

    /// Check if an address is mapped.
    pub fn is_mapped(&self, addr: u64) -> bool {
        let page_num = addr >> 12;
        self.pages.contains_key(&page_num)
    }

    /// Load bytes from memory as a RustBV.
    pub fn load(
        &self,
        addr: RustBV,
        size: u32,
        ctx: &SymContext,
    ) -> Result<RustBV, MemoryError> {
        // For symbolic addresses, we need to concretize or fork
        let concrete_addr = match addr.as_u64() {
            Some(a) => a,
            None => {
                // Try to evaluate the address
                match ctx.eval(&addr) {
                    Some(a) => a as u64,
                    None => {
                        return Err(MemoryError::SymbolicAddress {
                            description: "could not resolve address".to_string(),
                        });
                    }
                }
            }
        };

        self.load_concrete(concrete_addr, size, ctx)
    }

    /// Load from a concrete address.
    pub fn load_concrete(
        &self,
        addr: u64,
        size: u32,
        ctx: &SymContext,
    ) -> Result<RustBV, MemoryError> {
        let mut bytes = Vec::with_capacity(size as usize);
        let mut has_symbolic = false;

        // Check for stored symbolic object first
        if let Some(sym) = self.symbolic_objects.get(&addr) {
            if sym.width() == size * 8 {
                return Ok(sym.clone());
            }
        }

        for i in 0..size {
            let byte_addr = addr + i as u64;
            let page_num = byte_addr >> 12;
            let offset = (byte_addr & PAGE_MASK) as u16;

            if let Some(page) = self.pages.get(&page_num) {
                if page.is_symbolic(offset) {
                    has_symbolic = true;
                }
                let byte = page.load_concrete(offset, 1);
                bytes.push(byte.get(0).copied().unwrap_or(0));
            } else {
                return Err(MemoryError::Unmapped {
                    addr: byte_addr,
                    size: 1,
                });
            }
        }

        if has_symbolic {
            // Return stored symbolic object if available
            if let Some(sym) = self.symbolic_objects.get(&addr) {
                return Ok(sym.clone());
            }
            // Otherwise create a fresh symbolic value
            return Err(MemoryError::SymbolicAddress {
                description: "symbolic bytes not fully tracked".to_string(),
            });
        }

        // Convert bytes to value based on endianness
        let value = match self.endness {
            Endness::Little => {
                let mut v: u128 = 0;
                for (i, &byte) in bytes.iter().enumerate() {
                    v |= (byte as u128) << (i * 8);
                }
                v
            }
            Endness::Big => {
                let mut v: u128 = 0;
                for &byte in &bytes {
                    v = (v << 8) | (byte as u128);
                }
                v
            }
        };

        Ok(RustBV::concrete(value, size * 8))
    }

    /// Store a value to memory.
    pub fn store(
        &mut self,
        addr: RustBV,
        value: RustBV,
        ctx: &SymContext,
    ) -> Result<(), MemoryError> {
        // For symbolic addresses, we need to concretize
        let concrete_addr = match addr.as_u64() {
            Some(a) => a,
            None => {
                match ctx.eval(&addr) {
                    Some(a) => a as u64,
                    None => {
                        return Err(MemoryError::SymbolicAddress {
                            description: "could not resolve address for store".to_string(),
                        });
                    }
                }
            }
        };

        self.store_concrete(concrete_addr, value)
    }

    /// Store to a concrete address.
    pub fn store_concrete(
        &mut self,
        addr: u64,
        value: RustBV,
    ) -> Result<(), MemoryError> {
        let size = value.width() / 8;

        // Check if pages are mapped
        let start_page = addr >> 12;
        let end_page = (addr + size as u64 + PAGE_SIZE - 1) >> 12;

        for page_num in start_page..end_page {
            if !self.pages.contains_key(&page_num) {
                return Err(MemoryError::Unmapped {
                    addr: page_num << 12,
                    size: PAGE_SIZE,
                });
            }
        }

        // If symbolic, store in symbolic_objects
        if value.is_symbolic() {
            self.symbolic_objects.insert(addr, value.clone());
            // Also mark pages as having symbolic bytes
            for i in 0..size {
                let byte_addr = addr + i as u64;
                let page_num = byte_addr >> 12;
                let offset = (byte_addr & PAGE_MASK) as u16;

                if let Some(page) = self.pages.get(&page_num) {
                    let mut page = page.clone();
                    page.mark_symbolic(offset, 1);
                    self.pages.insert(page_num, page);
                }
            }
            return Ok(());
        }

        // Concrete store
        let concrete_val = value.to_u128();

        // Convert to bytes based on endianness
        let bytes: Vec<u8> = match self.endness {
            Endness::Little => (0..size).map(|i| (concrete_val >> (i * 8)) as u8).collect(),
            Endness::Big => (0..size)
                .rev()
                .map(|i| (concrete_val >> (i * 8)) as u8)
                .collect(),
        };

        // Write to pages
        let mut remaining = &bytes[..];
        let mut current_addr = addr;

        while !remaining.is_empty() {
            let page_num = current_addr >> 12;
            let page_offset = (current_addr & PAGE_MASK) as u16;
            let bytes_in_page = ((PAGE_SIZE - page_offset as u64) as usize).min(remaining.len());

            if let Some(page) = self.pages.get(&page_num) {
                let mut page = page.clone();
                page.store_concrete(page_offset, &remaining[..bytes_in_page]);
                self.pages.insert(page_num, page);
            }

            remaining = &remaining[bytes_in_page..];
            current_addr += bytes_in_page as u64;
        }

        // Clear any symbolic object at this address
        self.symbolic_objects.remove(&addr);

        Ok(())
    }

    /// Fork the memory (O(1) via CoW).
    pub fn fork(&self) -> Self {
        SymbolicMemory {
            pages: self.pages.clone(), // OrdMap clones in O(1)
            symbolic_objects: self.symbolic_objects.clone(),
            next_sym_id: self.next_sym_id,
            default_permissions: self.default_permissions,
            endness: self.endness,
        }
    }

    /// Get page count.
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Get total mapped size in bytes.
    pub fn mapped_size(&self) -> u64 {
        self.pages.len() as u64 * PAGE_SIZE
    }
}

impl Clone for SymbolicMemory {
    fn clone(&self) -> Self {
        self.fork()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_concrete() {
        let ctx = SymContext::new_mock();

        let mut mem = SymbolicMemory::new(Endness::Little);

        // Map a page
        mem.map(0x1000, 0x1000, Permission::RWX);

        // Store and load
        let addr = RustBV::concrete(0x1000, 64);
        let value = RustBV::concrete(0x12345678, 32);
        mem.store(addr.clone(), value.clone(), &ctx).unwrap();

        let loaded = mem.load(addr, 4, &ctx).unwrap();
        assert_eq!(loaded.as_u64(), Some(0x12345678));
    }

    #[test]
    fn test_memory_endianness() {
        let ctx = SymContext::new_mock();

        // Little-endian memory
        let mut mem_le = SymbolicMemory::new(Endness::Little);
        mem_le.map(0x1000, 0x1000, Permission::RWX);
        mem_le
            .store_concrete(0x1000, RustBV::concrete(0x12345678, 32))
            .unwrap();

        // First byte should be 0x78 (low byte)
        let byte0 = mem_le.load_concrete(0x1000, 1, &ctx).unwrap();
        assert_eq!(byte0.as_u64(), Some(0x78));

        // Big-endian memory
        let mut mem_be = SymbolicMemory::new(Endness::Big);
        mem_be.map(0x1000, 0x1000, Permission::RWX);
        mem_be
            .store_concrete(0x1000, RustBV::concrete(0x12345678, 32))
            .unwrap();

        // First byte should be 0x12 (high byte)
        let byte0 = mem_be.load_concrete(0x1000, 1, &ctx).unwrap();
        assert_eq!(byte0.as_u64(), Some(0x12));
    }

    #[test]
    fn test_memory_fork() {
        let ctx = SymContext::new_mock();

        let mut mem1 = SymbolicMemory::new(Endness::Little);
        mem1.map(0x1000, 0x1000, Permission::RWX);
        mem1.store_concrete(0x1000, RustBV::concrete(0xAAAA, 16))
            .unwrap();

        // Fork
        let mut mem2 = mem1.fork();

        // Modify mem2
        mem2.store_concrete(0x1000, RustBV::concrete(0xBBBB, 16))
            .unwrap();

        // mem1 should still have original value
        let val1 = mem1.load_concrete(0x1000, 2, &ctx).unwrap();
        assert_eq!(val1.as_u64(), Some(0xAAAA));

        // mem2 should have new value
        let val2 = mem2.load_concrete(0x1000, 2, &ctx).unwrap();
        assert_eq!(val2.as_u64(), Some(0xBBBB));
    }

    #[test]
    fn test_memory_map_data() {
        let ctx = SymContext::new_mock();

        let mut mem = SymbolicMemory::new(Endness::Little);

        // Map data
        let data: Vec<u8> = (0..=255u8).collect();
        mem.map_data(0x1000, &data, Permission::RX);

        // Check some bytes
        let byte0 = mem.load_concrete(0x1000, 1, &ctx).unwrap();
        assert_eq!(byte0.as_u64(), Some(0));

        let byte100 = mem.load_concrete(0x1064, 1, &ctx).unwrap();
        assert_eq!(byte100.as_u64(), Some(100));
    }
}
